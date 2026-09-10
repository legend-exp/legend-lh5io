# ruff: noqa: ARG001

from __future__ import annotations

import awkward as ak
import h5py
import lgdo
import numpy as np
import pytest
from lgdo import types

import lh5


def test_init():
    lh5.LH5Store()


def test_gimme_file(lgnd_file, tmptestdir):
    # get a file and check that it's cached
    with lh5.LH5Store(keep_open=True) as store:
        full_path = store.base_path / lgnd_file
        f = store.gimme_file(lgnd_file)
        assert isinstance(f, h5py.File)
        assert store.files[full_path] == f

        with pytest.raises(FileNotFoundError):
            store.gimme_file("non-existent-file")

    # leaving the context should clear the cache
    assert full_path not in store.files

    # no caching if keep_open is False
    with lh5.LH5Store(keep_open=False) as store:
        f = store.gimme_file(lgnd_file)
        assert isinstance(f, h5py.File)
        assert full_path not in store.files

    # test cache
    files = [tmptestdir / f"test-cache{i}.lh5" for i in range(3)]
    with lh5.LH5Store(keep_open=2, default_mode="of") as store:
        for file in files:
            f = store.gimme_file(file)
        assert files[0] not in store.files
        assert files[1] in store.files
        assert files[2] in store.files


def test_close_file(lgnd_file):
    store = lh5.LH5Store(keep_open=True)
    full_path = store.base_path / lgnd_file

    f = store.gimme_file(lgnd_file)
    assert isinstance(f, h5py.File)
    assert store.files[full_path] == f
    store.close(lgnd_file)
    assert not f.id.valid
    assert full_path not in store.files

    f = store.gimme_file(lgnd_file)
    store.close()
    assert not f.id.valid
    assert full_path not in store.files

    with lh5.LH5Store(keep_open=True) as st:
        f = st.gimme_file(lgnd_file)
    assert not f.id.valid


def test_gimme_group(lgnd_file, tmptestdir):
    f = h5py.File(lgnd_file)
    store = lh5.LH5Store()
    g = store.gimme_group("/geds", f)
    assert isinstance(g, h5py.Group)

    f = h5py.File(f"{tmptestdir}/testfile.lh5", mode="w")
    g = store.gimme_group("/geds", f, grp_attrs={"attr1": 1}, overwrite=True)
    assert isinstance(g, h5py.Group)


def test_write_objects(tmptestdir):
    # test writing in all wo_modes and with several other arguments

    # test writing all object types at once by putting them in a Table
    struct = lgdo.Struct(
        {
            "table": lgdo.Table(
                {
                    "array": lgdo.Array(np.arange(10)),
                    "aoesa": lgdo.ArrayOfEqualSizedArrays(
                        nda=np.arange(100).reshape((10, 10))
                    ),
                    "waveform": lgdo.WaveformTable(
                        values=lgdo.ArrayOfEqualSizedArrays(
                            nda=np.arange(100).reshape((10, 10)), attrs={"unit": "ADC"}
                        ),
                        t0=lgdo.Array(np.arange(10), attrs={"unit": "ns"}),
                        dt=10,
                        dt_units="ns",
                    ),
                    "vov": lgdo.VectorOfVectors(
                        flattened_data=np.arange(100),
                        cumulative_length=lgdo.Array(
                            [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
                        ),
                    ),
                }
            )
        }
    )

    struct_append = lgdo.Struct(
        {
            "table": lgdo.Table(
                {
                    "array": lgdo.Array(np.arange(10, 12)),
                    "aoesa": lgdo.ArrayOfEqualSizedArrays(
                        nda=np.arange(100, 120).reshape((2, 10))
                    ),
                    "waveform": lgdo.WaveformTable(
                        values=lgdo.ArrayOfEqualSizedArrays(
                            nda=np.arange(100, 120).reshape((2, 10)),
                            attrs={"unit": "ADC"},
                        ),
                        t0=lgdo.Array(np.arange(10, 12), attrs={"unit": "ns"}),
                        dt=10,
                        dt_units="ns",
                    ),
                    "vov": lgdo.VectorOfVectors(
                        flattened_data=np.arange(100, 144),
                        cumulative_length=lgdo.Array([21, 44]),
                    ),
                }
            )
        }
    )

    struct_combined = lgdo.Struct(
        {
            "table": lgdo.Table(
                {
                    "array": lgdo.Array(np.arange(12)),
                    "aoesa": lgdo.ArrayOfEqualSizedArrays(
                        nda=np.arange(120).reshape((12, 10))
                    ),
                    "waveform": lgdo.WaveformTable(
                        values=lgdo.ArrayOfEqualSizedArrays(
                            nda=np.arange(120).reshape((12, 10)), attrs={"unit": "ADC"}
                        ),
                        t0=lgdo.Array(np.arange(12), attrs={"unit": "ns"}),
                        dt=10,
                        dt_units="ns",
                    ),
                    "vov": lgdo.VectorOfVectors(
                        flattened_data=np.arange(144),
                        cumulative_length=lgdo.Array(
                            [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]
                        ),
                    ),
                }
            )
        }
    )

    # append node with new file
    outfile = tmptestdir / "test-write-objects.lh5"
    with lh5.LH5Store(keep_open=True, default_mode="a") as store:
        store.write(struct, "struct", outfile, group="/data")
        assert store.read("/data/struct", outfile) == struct
        store.write(struct_append, "struct", outfile, group="/data")
        assert store.read("/data/struct", outfile) == struct_combined

    # overwrite on existing file
    with lh5.LH5Store(keep_open=True, default_mode="o") as store:
        store.write(struct, "struct", outfile, group="/data")
        assert store.read("/data/struct", outfile) == struct
        store.write(struct_append, "struct", outfile, group="/data", write_start=10)
        assert store.read("/data/struct", outfile) == struct_combined
        store.write(
            struct, "struct", outfile, group="/data", write_start=5, start_row=5
        )
        assert store.read("/data/struct", outfile) == struct
        store.write(struct_append, "struct", outfile, group="/data")
        assert store.read("/data/struct", outfile) == struct_append

    # overwrite_file; second write should automatically swap to append!
    with lh5.LH5Store(keep_open=True, default_mode="of") as store:
        store.write(struct, "struct", outfile, group="/data")
        assert store.read("/data/struct", outfile) == struct
        store.write(struct_append, "struct", outfile, group="/data")
        assert store.read("/data/struct", outfile) == struct_combined

    # append_column
    with lh5.LH5Store(keep_open=True, default_mode="ac") as store:
        # cannot append if already exists
        with pytest.raises(lh5.io.exceptions.LH5EncodeError):
            store.write(struct.table, "table", outfile, group="/data/struct")

        # ac won't create new structs
        with pytest.raises(lh5.io.exceptions.LH5EncodeError):
            store.write(struct_append, "struct2", outfile, group="/data")

        # data should be unchanged
        assert store.read("/data/struct", outfile) == struct_combined

        # ac will create new table/column
        new_col = lgdo.Array(-np.arange(12))
        store.write(new_col, "array2", outfile, group="/data/struct/table")
        assert store.read("/data/struct/table/array2", outfile) == new_col

    # overwrite_file; second write with keep_open=False, second write should not append!
    with lh5.LH5Store(keep_open=False, default_mode="of") as store:
        store.write(struct, "struct", outfile, group="/data")
        assert lh5.read("/data/struct", outfile) == struct
        store.write(struct_append, "struct", outfile, group="/data")
        assert lh5.read("/data/struct", outfile) == struct_append


def test_write_safe(tmptestdir):
    # write_safe should create new file
    lh5_st = lh5.LH5Store()
    struct = lgdo.Struct()
    struct.add_field("scalar", lgdo.Scalar(value=10, attrs={"sth": 1}))
    lh5_st.write(
        struct,
        "struct",
        f"{tmptestdir}/tmp-pygama-write_safe_store.lh5",
        group="/data",
        start_row=1,
        n_rows=3,
        wo_mode="w",
    )
    assert lh5.ls(f"{tmptestdir}/tmp-pygama-write_safe_store.lh5")

    # write_safe should add a new group to an existing file
    lh5_st = lh5.LH5Store()
    struct = lgdo.Struct()
    struct.add_field("scalar", lgdo.Scalar(value=10, attrs={"sth": 1}))
    lh5_st.write(
        struct,
        "struct2",
        f"{tmptestdir}/tmp-pygama-write_safe_store.lh5",
        group="/data",
        start_row=1,
        n_rows=3,
        wo_mode="w",
    )
    assert lh5.ls(f"{tmptestdir}/tmp-pygama-write_safe_store.lh5", "data/") == [
        "data/struct",
        "data/struct2",
    ]

    # write_safe should not allow writing to existing dataset
    lh5_st = lh5.LH5Store()
    with pytest.raises(lh5.io.exceptions.LH5EncodeError):
        lh5_st.write(
            struct,
            "struct",
            f"{tmptestdir}/tmp-pygama-write_safe_store.lh5",
            group="/data",
            start_row=1,
            n_rows=3,
            wo_mode="w",
        )


def test_read_n_rows(lh5_file):
    store = lh5.LH5Store()
    assert store.read_n_rows("/data/struct_full/aoesa", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/array", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/scalar", lh5_file) is None
    assert store.read_n_rows("/data/struct_full/table", lh5_file) == 4
    assert store.read_n_rows("/data/struct_full/voev", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/vov", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/vov3d", lh5_file) == 5
    assert store.read_n_rows("/data/struct_full/wftable", lh5_file) == 10
    assert store.read_n_rows("/data/struct_full/wftable_enc/values", lh5_file) == 10


def test_read_size_in_bytes(lh5_file):
    store = lh5.LH5Store()
    assert store.read_size_in_bytes("/data/struct_full/aoesa", lh5_file) == 100
    assert store.read_size_in_bytes("/data/struct_full/array", lh5_file) == 40
    assert store.read_size_in_bytes("/data/struct_full/scalar", lh5_file) == 8
    assert store.read_size_in_bytes("/data/struct_full/table", lh5_file) == 144
    # assert store.read_size_in_bytes("/data/struct_full/voev", lh5_file) == ?
    assert store.read_size_in_bytes("/data/struct_full/vov", lh5_file) == 144
    assert store.read_size_in_bytes("/data/struct_full/vov3d", lh5_file) == 232
    assert store.read_size_in_bytes("/data/struct_full/wftable", lh5_file) == 20160
    assert (
        store.read_size_in_bytes("/data/struct_full/wftable_enc", lh5_file)
        == 80 + 80 + 40000
    )


def test_get_buffer(lh5_file):
    store = lh5.LH5Store()
    buf = store.get_buffer("/data/struct_full/wftable_enc", lh5_file)
    assert isinstance(buf.values, types.ArrayOfEqualSizedArrays)


def test_read_scalar(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/scalar", lh5_file)
    assert isinstance(lh5_obj, lgdo.Scalar)
    assert lh5_obj.value == 10
    assert lh5_obj.attrs["sth"] == 1
    with h5py.File(lh5_file) as h5f:
        assert h5f["/data/struct/scalar"].compression is None


def test_read_array(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/array", lh5_file)
    assert isinstance(lh5_obj, types.Array)
    assert (lh5_obj.nda == np.array([2, 3, 4])).all()
    assert len(lh5_obj) == 3
    with h5py.File(lh5_file) as h5f:
        assert (
            h5f["/data/struct/array"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )

    lh5_obj = store.read("/data/struct_full/array2d", lh5_file)
    assert isinstance(lh5_obj, types.Array)
    assert lh5_obj == types.Array(shape=(23, 56), fill_val=69, dtype=int)


def test_read_array_slice(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct_full/array", lh5_file, start_row=1, n_rows=3)
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert lh5_obj == lgdo.Array([2, 3, 4])

    lh5_obj = store.read(
        "/data/struct_full/array", [lh5_file, lh5_file], start_row=1, n_rows=6
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 6
    assert lh5_obj == lgdo.Array([2, 3, 4, 5, 1, 2])


def test_read_array_fancy_idx(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct_full/array", lh5_file, idx=3)
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 1
    assert lh5_obj == lgdo.Array([4])

    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct_full/array", lh5_file, idx=[0, 3, 4])
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert lh5_obj == lgdo.Array([1, 4, 5])

    # Test reading multiple files
    lh5_obj = store.read(
        "/data/struct_full/array", [lh5_file, lh5_file], idx=[[0, 3, 4], [0, 3, 4]]
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 6
    assert lh5_obj == lgdo.Array([1, 4, 5, 1, 4, 5])

    lh5_obj = store.read(
        "/data/struct_full/array", [lh5_file, lh5_file], idx=[0, 3, 4, 5, 8, 9]
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 6
    assert lh5_obj == lgdo.Array([1, 4, 5, 1, 4, 5])

    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct_full/array", [lh5_file, lh5_file], idx=[0, 3, 4])
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert lh5_obj == lgdo.Array([1, 4, 5])

    # Test with out of range index
    lh5_obj = store.read("/data/struct_full/array", lh5_file, idx=[0, 3, 4, 100])
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert lh5_obj == lgdo.Array([1, 4, 5])

    # Test with out of order index
    with pytest.raises(ValueError):
        store.read("/data/struct_full/array", lh5_file, idx=[0, 4, 3])

    # Test with out of order index
    with pytest.raises(ValueError):
        store.read("/data/struct_full/array", lh5_file, idx=[[0], [3, 5]])

    # Test with boolean mask
    lh5_obj = store.read(
        "/data/struct_full/array", lh5_file, idx=np.array([1, 0, 0, 1, 1], "bool")
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert lh5_obj == lgdo.Array([1, 4, 5])

    lh5_obj = store.read(
        "/data/struct_full/array",
        [lh5_file, lh5_file],
        idx=np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1], "bool"),
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 6
    assert lh5_obj == lgdo.Array([1, 4, 5, 1, 4, 5])

    # Test interaction with start_rows and n_rows
    lh5_obj = store.read(
        "/data/struct_full/array", lh5_file, start_row=1, n_rows=1, idx=[0, 3, 4]
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 1
    assert lh5_obj == lgdo.Array([4])

    # Test idx with list of ranges
    lh5_obj = store.read(
        "/data/struct_full/array", lh5_file, idx=np.array([[0, 1], [3, 5]])
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert lh5_obj == lgdo.Array([1, 4, 5])

    # test with list of ranges with no gap
    lh5_obj = store.read(
        "/data/struct_full/array", lh5_file, idx=np.array([[0, 1], [1, 3]])
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert lh5_obj == lgdo.Array([1, 2, 3])

    # Test idx with list of ranges and start_rows and n_rows
    lh5_obj = store.read(
        "/data/struct_full/array",
        lh5_file,
        start_row=1,
        n_rows=1,
        idx=np.array([[0, 1], [3, 5]]),
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 1
    assert lh5_obj == lgdo.Array([4])

    # Test idx with list of ranges and start_rows and n_rows
    lh5_obj = store.read(
        "/data/struct_full/array", lh5_file, idx=np.array([[0, 1], [3, 10]])
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert lh5_obj == lgdo.Array([1, 4, 5])

    lh5_obj = store.read(
        "/data/struct_full/array", lh5_file, idx=np.array([[0, 1], [3, 5], [7, 10]])
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert lh5_obj == lgdo.Array([1, 4, 5])

    lh5_obj = store.read("/data/struct_full/array", lh5_file, idx=np.array([[10, 15]]))
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 0

    # Test idx with list of ranges and multiple files
    lh5_obj = store.read(
        "/data/struct_full/array",
        [lh5_file] * 3,
        idx=np.array([[0, 3], [4, 7], [11, 13]]),
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 8
    assert lh5_obj == lgdo.Array([1, 2, 3, 5, 1, 2, 2, 3])

    lh5_obj = store.read("/data/struct_full/array", [lh5_file] * 3, idx=12)
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 1
    assert lh5_obj == lgdo.Array([3])

    # This should interpret as a 1D array for each file
    lh5_obj = store.read(
        "/data/struct_full/array", [lh5_file] * 2, idx=[[0, 3], [0, 3]]
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 4
    assert lh5_obj == lgdo.Array([1, 4, 1, 4])

    lh5_obj = store.read("/data/struct_full/array", [lh5_file] * 2, idx=[1, None])
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 6
    assert lh5_obj == lgdo.Array([2, 1, 2, 3, 4, 5])

    # This should interpret as a 2D array across all files
    lh5_obj = store.read(
        "/data/struct_full/array", [lh5_file] * 2, idx=np.array([[0, 3], [5, 8]])
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 6
    assert lh5_obj == lgdo.Array([1, 2, 3, 1, 2, 3])

    with pytest.raises(ValueError):
        store.read("/data/struct_full/array", [lh5_file] * 2, idx=[[1, 3], 4, 5])


def test_read_vov(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/vov", lh5_file)
    assert isinstance(lh5_obj, types.VectorOfVectors)

    assert lh5_obj == lgdo.VectorOfVectors(
        [[3, 4, 5], [2], [4, 8, 9, 7]], attrs={"myattr": 2}
    )

    assert len(lh5_obj) == 3
    assert lh5_obj.attrs["myattr"] == 2

    lh5_obj = store.read("/data/struct/vov", [lh5_file, lh5_file])
    assert len(lh5_obj) == 6
    assert lh5_obj == lgdo.VectorOfVectors(
        [[3, 4, 5], [2], [4, 8, 9, 7], [3, 4, 5], [2], [4, 8, 9, 7]],
        attrs={"myattr": 2},
    )

    with h5py.File(lh5_file) as h5f:
        assert (
            h5f["/data/struct/vov/cumulative_length"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/vov/flattened_data"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )

    lh5_obj = store.read("/data/struct/vov3d", lh5_file)
    assert isinstance(lh5_obj, types.VectorOfVectors)

    assert ak.all(
        lh5_obj.view_as("ak") == ak.Array([[[2], [4, 8, 9, 7]], [[5, 3, 1]], [[3], []]])
    )


def test_read_vov_fancy_idx(lh5_file):
    store = lh5.LH5Store()

    lh5_obj = store.read("/data/struct_full/vov", lh5_file, idx=[0], n_rows=1)
    assert isinstance(lh5_obj, types.VectorOfVectors)

    lh5_obj = store.read("/data/struct_full/vov", lh5_file, idx=[0, 2])
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert lh5_obj == types.VectorOfVectors([[1, 2], [2]], attrs={"myattr": 2})
    assert len(lh5_obj) == 2

    lh5_obj = store.read("/data/struct_full/vov", lh5_file, idx=[0, 2, 10])
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert lh5_obj == types.VectorOfVectors([[1, 2], [2]], attrs={"myattr": 2})
    assert len(lh5_obj) == 2

    lh5_obj = store.read("/data/struct_full/vov3d", lh5_file, idx=[0, 2])
    assert isinstance(lh5_obj, types.VectorOfVectors)

    assert lh5_obj == types.VectorOfVectors([[[1, 2], [3, 4, 5]], [[5, 3, 1]]])
    assert len(lh5_obj) == 2

    # Out-of-range indices should be culled (and not raise).
    lh5_obj = store.read("/data/struct_full/vov3d", lh5_file, idx=[0, 10_000])
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert lh5_obj == types.VectorOfVectors([[[1, 2], [3, 4, 5]]])
    assert len(lh5_obj) == 1

    # A fully out-of-range idx should yield an empty object.
    lh5_obj = store.read("/data/struct_full/vov3d", lh5_file, idx=[10_000])
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert len(lh5_obj) == 0

    lh5_obj = store.read(
        "/data/struct_full/vov", lh5_file, start_row=1, n_rows=1, idx=[0, 2, 3]
    )
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert lh5_obj == types.VectorOfVectors([[2]], attrs={"myattr": 2})
    assert len(lh5_obj) == 1


def test_read_voev(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/voev", lh5_file, decompress=False)
    assert isinstance(lh5_obj, types.VectorOfEncodedVectors)

    desired = [np.array([3, 4, 5]), np.array([2]), np.array([4, 8, 9, 7])]

    for i in range(len(desired)):
        assert (desired[i] == lh5_obj[i][0]).all()

    assert len(lh5_obj) == 3

    lh5_obj = store.read("/data/struct/voev", [lh5_file, lh5_file], decompress=False)
    assert isinstance(lh5_obj, types.VectorOfEncodedVectors)
    assert len(lh5_obj) == 6

    with h5py.File(lh5_file) as h5f:
        assert h5f["/data/struct/voev/encoded_data/flattened_data"].compression is None
        assert (
            h5f["/data/struct/voev/encoded_data/cumulative_length"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/voev/decoded_size"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )


def test_read_voev_fancy_idx(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read(
        "/data/struct_full/voev", lh5_file, idx=[0, 2], decompress=False
    )
    assert isinstance(lh5_obj, types.VectorOfEncodedVectors)

    desired = [np.array([1, 2]), np.array([2])]

    for i in range(len(desired)):
        assert (desired[i] == lh5_obj[i][0]).all()

    assert len(lh5_obj) == 2


def test_read_aoesa(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/aoesa", lh5_file)
    assert isinstance(lh5_obj, types.ArrayOfEqualSizedArrays)
    assert (lh5_obj.nda == np.full((3, 5), fill_value=42)).all()


def test_read_aoesa_fancy_idx(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/aoesa", lh5_file, idx=[0, 2])
    assert isinstance(lh5_obj, types.ArrayOfEqualSizedArrays)
    assert (lh5_obj.nda == np.full((2, 5), fill_value=42)).all()

    lh5_obj = store.read("/data/struct/aoesa", lh5_file, idx=[0, 2, 10])
    assert isinstance(lh5_obj, types.ArrayOfEqualSizedArrays)
    assert (lh5_obj.nda == np.full((2, 5), fill_value=42)).all()

    store = lh5.LH5Store()
    lh5_obj = store.read(
        "/data/struct/aoesa", lh5_file, start_row=1, n_rows=1, idx=[0, 2, 3]
    )
    assert isinstance(lh5_obj, types.ArrayOfEqualSizedArrays)
    assert (lh5_obj.nda == np.full((1, 5), fill_value=42)).all()


def test_read_table(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/table", lh5_file)
    assert isinstance(lh5_obj, types.Table)
    assert len(lh5_obj) == 3

    lh5_obj = store.read("/data/struct/table", [lh5_file, lh5_file])
    assert len(lh5_obj) == 6
    assert lh5_obj.attrs["stuff"] == 5
    assert lh5_obj["a"].attrs["attr"] == 9


def test_read_table_fancy_idx(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/table", lh5_file, idx=[0, 2])
    assert isinstance(lh5_obj, types.Table)
    assert len(lh5_obj) == 2

    lh5_obj = store.read("/data/struct/table", lh5_file, idx=[0, 2, 10])
    assert isinstance(lh5_obj, types.Table)
    assert len(lh5_obj) == 2

    lh5_obj = store.read("/data/struct/table", lh5_file, idx=[])
    assert isinstance(lh5_obj, types.Table)
    assert len(lh5_obj) == 0

    store = lh5.LH5Store()
    lh5_obj = store.read(
        "/data/struct/table", lh5_file, start_row=1, n_rows=1, idx=[0, 2, 3]
    )
    assert isinstance(lh5_obj, types.Table)
    assert len(lh5_obj) == 1


def test_read_empty_struct(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/empty_struct", lh5_file)
    assert isinstance(lh5_obj, types.Struct)
    assert list(lh5_obj.keys()) == []


def test_read_hdf5_compressed_data(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/table", lh5_file)

    assert "compression" not in lh5_obj["b"].attrs
    with h5py.File(lh5_file) as h5f:
        assert (
            h5f["/data/struct/table/a"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert h5f["/data/struct/table/b"].compression == "gzip"
        assert h5f["/data/struct/table/c"].compression == "gzip"
        assert h5f["/data/struct/table/c"].compression_opts == 9
        assert h5f["/data/struct/table/d"].compression is None


def test_read_wftable(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/wftable", lh5_file)
    assert isinstance(lh5_obj, types.WaveformTable)
    assert len(lh5_obj) == 3

    lh5_obj = store.read("/data/struct/wftable", [lh5_file, lh5_file])
    assert len(lh5_obj) == 6
    assert lh5_obj.values.attrs["custom"] == 8

    with h5py.File(lh5_file) as h5f:
        assert (
            h5f["/data/struct/wftable/values"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/wftable/t0"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/wftable/dt"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )


def test_read_wftable_encoded(lh5_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/data/struct/wftable_enc", lh5_file, decompress=False)
    assert isinstance(lh5_obj, types.WaveformTable)
    assert isinstance(lh5_obj.values, types.ArrayOfEncodedEqualSizedArrays)
    assert len(lh5_obj) == 3
    assert lh5_obj.values.attrs["codec"] == "radware_sigcompress"
    assert "codec_shift" in lh5_obj.values.attrs

    lh5_obj = store.read("/data/struct/wftable_enc/values", lh5_file)
    assert isinstance(lh5_obj, lgdo.ArrayOfEqualSizedArrays)
    assert len(lh5_obj) == 3

    lh5_obj = store.read("/data/struct/wftable_enc", lh5_file)
    assert isinstance(lh5_obj, lgdo.WaveformTable)
    assert isinstance(lh5_obj.values, lgdo.ArrayOfEqualSizedArrays)
    assert len(lh5_obj) == 3

    lh5_obj_chain = store.read(
        "/data/struct/wftable_enc", [lh5_file, lh5_file], decompress=False
    )
    assert len(lh5_obj_chain) == 6
    assert isinstance(lh5_obj_chain.values, lgdo.ArrayOfEncodedEqualSizedArrays)

    lh5_obj_chain = store.read(
        "/data/struct/wftable_enc", [lh5_file, lh5_file], decompress=True
    )
    assert isinstance(lh5_obj_chain.values, lgdo.ArrayOfEqualSizedArrays)
    assert np.array_equal(lh5_obj_chain.values[:3], lh5_obj.values)
    assert np.array_equal(lh5_obj_chain.values[3:], lh5_obj.values)
    assert len(lh5_obj_chain) == 6

    with h5py.File(lh5_file, locking=False) as h5f:
        assert (
            h5f[
                "/data/struct/wftable_enc/values/encoded_data/flattened_data"
            ].compression
            is None
        )
        assert h5f["/data/struct/wftable_enc/values/decoded_size"].compression is None
        assert (
            h5f["/data/struct/wftable_enc/t0"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )
        assert (
            h5f["/data/struct/wftable_enc/dt"].compression
            is lh5.io.settings.DEFAULT_HDF5_SETTINGS["compression"]
        )


def test_read_with_field_mask(lh5_file):
    store = lh5.LH5Store()

    lh5_obj = store.read("/data/struct_full", lh5_file, field_mask=["array"])
    assert list(lh5_obj.keys()) == ["array"]

    lh5_obj = store.read("/data/struct_full", lh5_file, field_mask=("array", "table"))
    assert sorted(lh5_obj.keys()) == ["array", "table"]

    lh5_obj = store.read("/data/struct_full", lh5_file, field_mask={"array": True})
    assert list(lh5_obj.keys()) == ["array"]

    lh5_obj = store.read(
        "/data/struct_full", lh5_file, field_mask={"vov": False, "voev": False}
    )
    assert sorted(lh5_obj.keys()) == [
        "aoesa",
        "array",
        "array2d",
        "empty_struct",
        "scalar",
        "table",
        "vov3d",
        "wftable",
        "wftable_enc",
    ]


def test_read_with_nested_field_mask(lh5_file):
    store = lh5.LH5Store()

    lh5_obj = store.read(
        "/data", lh5_file, decompress=False, field_mask=["struct/table"]
    )
    assert sorted(lh5_obj.struct.keys()) == ["table"]
    assert sorted(lh5_obj.struct.table.keys()) == ["a", "b", "c", "d", "e"]

    lh5_obj = store.read("/data", lh5_file, field_mask=["struct/table/a"])
    assert sorted(lh5_obj.struct.table.keys()) == ["a"]

    lh5_obj = store.read(
        "/data", lh5_file, decompress=False, field_mask={"struct/table/b": False}
    )
    assert sorted(lh5_obj.struct.table.keys()) == ["a", "c", "d", "e"]

    lh5_obj = store.read(
        "/data",
        lh5_file,
        decompress=False,
        field_mask=["struct/table", "struct/table/b"],
    )
    assert sorted(lh5_obj.struct.table.keys()) == ["a", "b", "c", "d", "e"]

    lh5_obj = store.read(
        "/data",
        lh5_file,
        decompress=False,
        field_mask={"struct/table": False, "struct/table/b": True},
    )
    assert sorted(lh5_obj.struct.table.keys()) == ["b"]

    lh5_obj = store.read(
        "/data",
        lh5_file,
        decompress=False,
        field_mask={"struct/table": True, "struct/table/b": False},
    )
    assert sorted(lh5_obj.struct.table.keys()) == ["a", "c", "d", "e"]


def test_read_lgnd_array(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj = store.read("/geds/raw/baseline", lgnd_file)
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 100

    lh5_obj = store.read("/geds/raw/waveform/values", lgnd_file)
    assert isinstance(lh5_obj, types.ArrayOfEqualSizedArrays)


def test_read_lgnd_array_fancy_idx(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj = store.read("/geds/raw/baseline", lgnd_file, idx=[2, 4, 6, 9, 11, 16, 68])
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 7
    assert (lh5_obj.nda == [13508, 14353, 14525, 14341, 15079, 11675, 13995]).all()

    lh5_obj = store.read(
        "/geds/raw/baseline",
        lgnd_file,
        start_row=5,
        n_rows=3,
        idx=[2, 4, 6, 9, 11, 16, 68],
    )
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 3
    assert (lh5_obj.nda == [14525, 14341, 15079]).all()


def test_read_lgnd_vov(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj = store.read("/geds/raw/tracelist", lgnd_file)
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert len(lh5_obj) == 100


def test_read_lgnd_vov_fancy_idx(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj = store.read("/geds/raw/tracelist", lgnd_file, idx=[2, 4, 6, 9, 11, 16, 68])
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert len(lh5_obj) == 7
    assert (lh5_obj.cumulative_length.nda == [1, 2, 3, 4, 5, 6, 7]).all()
    assert (lh5_obj.flattened_data.nda == [40, 60, 64, 60, 64, 28, 60]).all()

    lh5_obj = store.read("/geds/raw/tracelist", lgnd_file, idx=[])
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert len(lh5_obj) == 0

    lh5_obj = store.read("/geds/raw/tracelist", [lgnd_file] * 3, idx=[250])
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert len(lh5_obj) == 1

    lh5_obj = store.read(
        "/geds/raw/tracelist",
        lgnd_file,
        start_row=5,
        n_rows=3,
        idx=[2, 4, 6, 9, 11, 16, 68],
    )
    assert isinstance(lh5_obj, types.VectorOfVectors)
    assert len(lh5_obj) == 3
    assert (lh5_obj.cumulative_length.nda == [1, 2, 3]).all()
    assert (lh5_obj.flattened_data.nda == [64, 60, 64]).all()


def test_read_array_concatenation(lgnd_file):
    store = lh5.LH5Store()
    lh5_obj = store.read("/geds/raw/baseline", [lgnd_file, lgnd_file])
    assert isinstance(lh5_obj, types.Array)
    assert len(lh5_obj) == 200


def test_read_lgnd_waveform_table(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj = store.read("/geds/raw/waveform", lgnd_file)
    assert isinstance(lh5_obj, types.WaveformTable)

    lh5_obj = store.read(
        "/geds/raw/waveform",
        lgnd_file,
        start_row=10,
        n_rows=10,
        field_mask=["t0", "dt"],
    )

    assert isinstance(lh5_obj, types.Table)
    assert list(lh5_obj.keys()) == ["t0", "dt"]
    assert len(lh5_obj) == 10


def test_read_lgnd_waveform_table_fancy_idx(lgnd_file):
    store = lh5.LH5Store()

    lh5_obj = store.read(
        "/geds/raw/waveform",
        lgnd_file,
        idx=[7, 9, 25, 27, 33, 38, 46, 52, 57, 59, 67, 71, 72, 82, 90, 92, 93, 94, 97],
    )
    assert isinstance(lh5_obj, types.WaveformTable)
    assert len(lh5_obj) == 19

    lh5_obj = store.read(
        "/geds/raw/waveform",
        lgnd_file,
        idx=[
            7,
            9,
            25,
            27,
            33,
            38,
            46,
            52,
            57,
            59,
            67,
            71,
            72,
            82,
            90,
            92,
            93,
            94,
            97,
            1000,
        ],
    )
    assert isinstance(lh5_obj, types.WaveformTable)
    assert len(lh5_obj) == 19

    lh5_obj2 = store.read(
        "/geds/raw/waveform",
        lgnd_file,
        start_row=10,
        n_rows=3,
        idx=[7, 9, 25, 27, 33, 38, 46, 52, 57, 59, 67, 71, 72, 82, 90, 92, 93, 94, 97],
    )
    assert isinstance(lh5_obj2, types.WaveformTable)
    assert len(lh5_obj2) == 3
    assert lh5_obj2 == lh5_obj[2:5]


def test_read_compressed_lgnd_waveform_table(lgnd_file, enc_lgnd_file):
    store = lh5.LH5Store()
    wft = store.read("/geds/raw/waveform", enc_lgnd_file)
    assert isinstance(wft.values, types.ArrayOfEqualSizedArrays)
    assert "compression" not in wft.values.attrs


def test_read_histogram_testdata(lgnd_test_data):
    file = lgnd_test_data.get_path("lh5/lgdo-histograms.lh5")

    h1 = lh5.read("test_histogram_range", file)
    assert isinstance(h1, types.Histogram)
    assert h1.binning[0].is_range

    h2 = lh5.read("test_histogram_variable", file)
    assert isinstance(h2, types.Histogram)
    assert not h2.binning[0].is_range

    h3 = lh5.read("test_histogram_range_w_attrs", file)
    assert isinstance(h3, types.Histogram)
    assert h3.binning[0].is_range
    assert h3.binning[0]["binedges"].getattrs() == {"units": "m"}


def test_read_histogram_multiple(lgnd_test_data):
    file = lgnd_test_data.get_path("lh5/lgdo-histograms.lh5")
    with pytest.raises(lh5.io.exceptions.LH5DecodeError):
        lh5.read("test_histogram_range", [file, file])


def test_views(tmptestdir):
    test_file = f"{tmptestdir}/test_view.lh5"
    external_file = f"{tmptestdir}/test_view_external.lh5"

    array = lgdo.Array(np.arange(100, dtype=np.int64))
    array2 = lgdo.Array(-np.arange(100, dtype=np.int64))

    entries_1d = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89], dtype=np.int64)
    entries_2d = np.array([[0, 2], [10, 13], [98, 100]], dtype=np.int64)
    expected_1d = np.copy(entries_1d)
    expected_2d = np.concatenate(
        [np.arange(a, b, dtype=np.int64) for a, b in entries_2d]
    )

    with lh5.LH5Store(keep_open=True, default_mode="of") as store:
        store.write(array, "array", test_file, group="/data")
        store.write(array2, "array2", test_file, group="/data")

        store.write_view(
            "/data/array",
            entries_1d,
            "hard_1d",
            test_file,
            link_type="hard",
            group="/views",
        )

        store.write_view(
            "/data/array",
            entries_1d,
            "soft_1d",
            test_file,
            link_type="soft",
            group="/views",
        )

        store.write_view(
            "/data/array",
            entries_1d,
            "external_1d",
            external_file,
            link_type="external",
            external_file=test_file,
            group="/views",
        )

        store.write_view(
            "/data/array",
            entries_2d,
            "hard_2d",
            test_file,
            link_type="hard",
            group="/views",
        )

        store.write_view(
            "/data/array",
            entries_2d,
            "soft_2d",
            test_file,
            link_type="soft",
            group="/views",
        )

        store.write_view(
            "/data/array",
            entries_2d,
            "external_2d",
            external_file,
            link_type="external",
            external_file=test_file,
            group="/views",
        )

    with lh5.LH5Store(keep_open=True, default_mode="r") as store:
        lh5_file = store.gimme_file(test_file)
        ext_file = store.gimme_file(external_file)

        assert store.read_n_rows("/views/hard_1d", lh5_file) == len(expected_1d)
        assert (
            store.read_size_in_bytes("/views/hard_1d", lh5_file) == expected_1d.nbytes
        )
        ar_h1d = store.read("/views/hard_1d", test_file)
        assert isinstance(ar_h1d, types.Array)
        assert np.all(ar_h1d.nda == expected_1d)
        assert isinstance(
            lh5_file["views/hard_1d"].get("data", getlink=True), h5py.HardLink
        )

        assert store.read_n_rows("/views/soft_1d", lh5_file) == len(expected_1d)
        assert (
            store.read_size_in_bytes("/views/soft_1d", lh5_file) == expected_1d.nbytes
        )
        ar_s1d = store.read("/views/soft_1d", test_file)
        assert isinstance(ar_s1d, types.Array)
        assert np.all(ar_s1d.nda == expected_1d)
        assert isinstance(
            lh5_file["views/soft_1d"].get("data", getlink=True), h5py.SoftLink
        )

        assert store.read_n_rows("/views/external_1d", external_file) == len(
            expected_1d
        )
        assert (
            store.read_size_in_bytes("/views/external_1d", external_file)
            == expected_1d.nbytes
        )
        ar_e1d = store.read("/views/external_1d", external_file)
        assert isinstance(ar_e1d, types.Array)
        assert np.all(ar_e1d.nda == expected_1d)
        assert isinstance(
            ext_file["views/external_1d"].get("data", getlink=True), h5py.ExternalLink
        )

        assert store.read_n_rows("/views/hard_2d", lh5_file) == len(expected_2d)
        assert (
            store.read_size_in_bytes("/views/hard_2d", lh5_file) == expected_2d.nbytes
        )
        ar_h2d = store.read("/views/hard_2d", test_file)
        assert isinstance(ar_h2d, types.Array)
        assert np.all(ar_h2d.nda == expected_2d)
        assert isinstance(
            lh5_file["views/hard_2d"].get("data", getlink=True), h5py.HardLink
        )

        assert store.read_n_rows("/views/soft_2d", lh5_file) == len(expected_2d)
        assert (
            store.read_size_in_bytes("/views/soft_2d", lh5_file) == expected_2d.nbytes
        )
        ar_s2d = store.read("/views/soft_2d", test_file)
        assert isinstance(ar_s2d, types.Array)
        assert np.all(ar_s2d.nda == expected_2d)
        assert isinstance(
            lh5_file["views/soft_2d"].get("data", getlink=True), h5py.SoftLink
        )

        assert store.read_n_rows("/views/external_2d", external_file) == len(
            expected_2d
        )
        assert (
            store.read_size_in_bytes("/views/external_2d", external_file)
            == expected_2d.nbytes
        )
        ar_e2d = store.read("/views/external_2d", external_file)
        assert isinstance(ar_e2d, types.Array)
        assert np.all(ar_e2d.nda == expected_2d)
        assert isinstance(
            ext_file["views/external_2d"].get("data", getlink=True), h5py.ExternalLink
        )

        del lh5_file
        del ext_file

    # we shouldn't be able to write_safe to an existing view...
    with (
        lh5.LH5Store(keep_open=True, default_mode="w") as store,
        pytest.raises(lh5.io.exceptions.LH5EncodeError),
    ):
        store.write_view(
            "/data/array",
            entries_1d,
            "hard_1d",
            test_file,
            link_type="hard",
            group="/views",
        )

    # append
    entries_app = np.array([91, 92, 93, 94, 95], dtype=np.int64)
    expected_app = np.concatenate([expected_1d, entries_app])
    with lh5.LH5Store(keep_open=True, default_mode="a") as store:
        store.write_view(
            "/data/array",
            entries_app,
            "hard_1d",
            test_file,
            link_type="hard",
            group="/views",
        )

        store.write_view(
            "/data/array",
            entries_app,
            "soft_1d",
            test_file,
            link_type="soft",
            group="/views",
        )

        store.write_view(
            "/data/array",
            entries_app,
            "external_1d",
            external_file,
            link_type="external",
            external_file=test_file,
            group="/views",
        )

        # error if we use a different array
        with pytest.raises(lh5.io.exceptions.LH5EncodeError):
            store.write_view(
                "/data/array2",
                entries_app,
                "hard_1d",
                test_file,
                link_type="hard",
                group="/views",
            )

        with pytest.raises(lh5.io.exceptions.LH5EncodeError):
            store.write_view(
                "/data/array2",
                entries_app,
                "soft_1d",
                test_file,
                link_type="soft",
                group="/views",
            )

        with pytest.raises(lh5.io.exceptions.LH5EncodeError):
            store.write_view(
                "/data/array2",
                entries_app,
                "external_1d",
                external_file,
                link_type="external",
                external_file=test_file,
                group="/views",
            )

    with lh5.LH5Store(keep_open=True, default_mode="r") as store:
        lh5_file = store.gimme_file(test_file)
        ext_file = store.gimme_file(external_file)

        ar_h1d = store.read("/views/hard_1d", test_file)
        assert isinstance(ar_h1d, types.Array)
        assert np.all(ar_h1d.nda == expected_app)
        assert isinstance(
            lh5_file["views/hard_1d"].get("data", getlink=True), h5py.HardLink
        )

        ar_s1d = store.read("/views/soft_1d", test_file)
        assert isinstance(ar_s1d, types.Array)
        assert np.all(ar_s1d.nda == expected_app)
        assert isinstance(
            lh5_file["views/soft_1d"].get("data", getlink=True), h5py.SoftLink
        )

        ar_e1d = store.read("/views/external_1d", external_file)
        assert isinstance(ar_e1d, types.Array)
        assert np.all(ar_e1d.nda == expected_app)
        assert isinstance(
            ext_file["views/external_1d"].get("data", getlink=True), h5py.ExternalLink
        )

        del lh5_file
        del ext_file

    # overwrite
    with lh5.LH5Store(keep_open=True, default_mode="o") as store:
        store.write_view(
            "/data/array2",
            entries_1d,
            "hard_1d",
            test_file,
            link_type="hard",
            group="/views",
        )

        store.write_view(
            "/data/array2",
            entries_1d,
            "soft_1d",
            test_file,
            link_type="soft",
            group="/views",
        )

        store.write_view(
            "/data/array2",
            entries_1d,
            "external_1d",
            external_file,
            link_type="external",
            external_file=test_file,
            group="/views",
        )

    with lh5.LH5Store(keep_open=True, default_mode="r") as store:
        ar_h1d = store.read("/views/hard_1d", test_file)
        assert isinstance(ar_h1d, types.Array)
        assert np.all(ar_h1d.nda == -expected_1d)

        ar_s1d = store.read("/views/soft_1d", test_file)
        assert isinstance(ar_s1d, types.Array)
        assert np.all(ar_s1d.nda == -expected_1d)

        ar_e1d = store.read("/views/external_1d", external_file)
        assert isinstance(ar_e1d, types.Array)
        assert np.all(ar_e1d.nda == -expected_1d)

    # Test automatic deduction of link type from inputs
    with lh5.LH5Store(keep_open=True, default_mode="o") as store:
        lh5_file = store.gimme_file(test_file)
        data_array = store.gimme_group("/data/array", lh5_file)
        data_array2 = store.gimme_group("/data/array2", lh5_file)

        store.write_view(
            data_array,
            entries_1d,
            "hard_1d",
            test_file,
            group="/views",
        )

        store.write_view(
            "/data/array",
            entries_1d,
            "soft_1d",
            test_file,
            group="/views",
        )

        store.write_view(
            data_array,
            entries_1d,
            "external_1d",
            external_file,
            external_file=test_file,
            group="/views",
        )

        # should error if group doesn't match external file
        with pytest.raises(lh5.io.exceptions.LH5EncodeError):
            store.write_view(
                data_array,
                entries_1d,
                "external_1d",
                external_file,
                external_file=external_file,
                group="/views",
            )

        # should figure out file on its own if it is external
        store.write_view(
            data_array2,
            entries_1d,
            "external2_1d",
            external_file,
            group="/views",
        )

        del lh5_file
        del data_array
        del data_array2

    with lh5.LH5Store(keep_open=True, default_mode="r") as store:
        ar_h1d = store.read("/views/hard_1d", test_file)
        assert isinstance(ar_h1d, types.Array)
        assert np.all(ar_h1d.nda == expected_1d)

        ar_s1d = store.read("/views/soft_1d", test_file)
        assert isinstance(ar_s1d, types.Array)
        assert np.all(ar_s1d.nda == expected_1d)

        ar_e1d = store.read("/views/external_1d", external_file)
        assert isinstance(ar_e1d, types.Array)
        assert np.all(ar_e1d.nda == expected_1d)

        ar_e1d = store.read("/views/external2_1d", external_file)
        assert isinstance(ar_e1d, types.Array)
        assert np.all(ar_e1d.nda == -expected_1d)

    # Test other sorts of errors...
    with lh5.LH5Store(keep_open=True, default_mode="a") as store:
        with pytest.raises(TypeError):
            store.write_view(
                "/data/array",
                np.array(entries_1d, dtype="float32"),
                "hard_1d",
                test_file,
                link_type="hard",
                group="/views",
            )

        with pytest.raises(lh5.io.exceptions.LH5EncodeError):
            store.write_view(
                "/data/array",
                np.array([5, 4, 3, 2, 1], dtype="int64"),
                "hard_1d",
                test_file,
                link_type="hard",
                group="/views",
            )

        with pytest.raises(lh5.io.exceptions.LH5EncodeError):
            store.write_view(
                "/data/array",
                np.arange(12, dtype="int64").reshape((3, 4)),
                "hard_1d",
                test_file,
                link_type="hard",
                group="/views",
            )

        with pytest.raises(ValueError):
            store.write_view(
                "/data/array",
                entries_1d,
                "hard_1d",
                test_file,
                link_type="hard",
                external_file=external_file,
                group="/views",
            )

        with pytest.raises(ValueError):
            store.write_view(
                "/data/array",
                entries_1d,
                "soft_1d",
                test_file,
                link_type="soft",
                external_file=external_file,
                group="/views",
            )

        store.write_view(
            "/data/array3",
            entries_1d,
            "soft_missing",
            test_file,
            link_type="soft",
            group="/views",
        )
        with pytest.raises(lh5.io.exceptions.LH5DecodeError):
            store.read("/views/soft_missing", test_file)

    # test with empty entries list
    with lh5.LH5Store(keep_open=True, default_mode="of") as store:
        store.write(array, "array", test_file, group="/data")

        store.write_view(
            "/data/array",
            np.array([], dtype="int64").reshape((0,)),
            "hard_1d",
            test_file,
            link_type="hard",
            group="/views",
        )

    with lh5.LH5Store(keep_open=True, default_mode="r") as store:
        ar_h1d = store.read("/views/hard_1d", test_file)
        assert isinstance(ar_h1d, types.Array)
        assert np.all(ar_h1d.nda == np.array([], dtype="int64"))
