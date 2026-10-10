from __future__ import annotations

import lgdo
import numpy as np
import pytest
from lgdo import types

import lh5


def test_read(lh5_file):
    lh5_obj = lh5.read("/data/struct/scalar", lh5_file)
    assert isinstance(lh5_obj, lgdo.Scalar)
    assert lh5_obj.value == 10
    assert lh5_obj.attrs["sth"] == 1


def test_write(tmptestdir):
    struct = lgdo.Struct()
    struct.add_field("scalar", lgdo.Scalar(value=10, attrs={"sth": 1}))
    lh5.write(
        struct,
        "struct",
        f"{tmptestdir}/tmp-pygama-lgdo-types2.lh5",
        group="/data",
        start_row=1,
        n_rows=3,
        wo_mode="overwrite_file",
    )
    assert lh5.ls(f"{tmptestdir}/tmp-pygama-lgdo-types2.lh5")


def test_write_safe(tmptestdir):
    # write_safe should create new file
    struct = lgdo.Struct()
    struct.add_field("scalar", lgdo.Scalar(value=10, attrs={"sth": 1}))
    lh5.write(
        struct,
        "struct",
        f"{tmptestdir}/tmp-pygama-write_safe.lh5",
        group="/data",
        start_row=1,
        n_rows=3,
        wo_mode="w",
    )
    assert lh5.ls(f"{tmptestdir}/tmp-pygama-write_safe.lh5")

    # write_safe should add a new group to an existing file
    struct = lgdo.Struct()
    struct.add_field("scalar", lgdo.Scalar(value=10, attrs={"sth": 1}))
    lh5.write(
        struct,
        "struct2",
        f"{tmptestdir}/tmp-pygama-write_safe.lh5",
        group="/data",
        start_row=1,
        n_rows=3,
        wo_mode="w",
    )
    assert lh5.ls(f"{tmptestdir}/tmp-pygama-write_safe.lh5", "data/") == [
        "data/struct",
        "data/struct2",
    ]

    # write_safe should not allow writing to existing dataset
    with pytest.raises(lh5.io.exceptions.LH5EncodeError):
        lh5.write(
            struct,
            "struct",
            f"{tmptestdir}/tmp-pygama-write_safe.lh5",
            group="/data",
            start_row=1,
            n_rows=3,
            wo_mode="w",
        )


def test_read_as(lh5_file):
    store = lh5.LH5Store()
    obj1 = store.read("/data/struct/table", lh5_file, start_row=1)
    obj1 = obj1.view_as("pd", with_units=True)

    obj2 = lh5.read_as(
        "/data/struct/table", lh5_file, "pd", start_row=1, with_units=True
    )
    assert obj1.equals(obj2)

    obj2 = lh5.read_as("/data/struct/table", [lh5_file], "ak")


def test_read_multiple_files(lh5_file):
    lh5_obj = lh5.read("/data/struct/array", [lh5_file, lh5_file, lh5_file])
    assert isinstance(lh5_obj, lgdo.Array)
    assert len(lh5_obj) == 9
    assert (lh5_obj.nda == np.array([2, 3, 4] * 3)).all()

    lh5_obj = lh5.read(
        "/data/struct/array", [lh5_file, lh5_file, lh5_file], idx=[1, 3, 5, 7]
    )
    assert len(lh5_obj) == 4
    assert (lh5_obj.nda == np.array([3, 2, 4, 3])).all()


def test_read_hdf5plugin_compression(lgnd_file_new_format):
    lh5_obj = lh5.read("/raw/B00000D/baseline", lgnd_file_new_format)
    assert isinstance(lh5_obj, lgdo.Array)
    assert lh5_obj.nda[0] == 14620


def test_views(tmptestdir):
    # Note: this test just tests core.write_view, not the inner workings.
    # test_lh5_store has more extensive tests
    test_file = f"{tmptestdir}/test_view.lh5"
    array = lgdo.Array(np.arange(100, dtype=np.int64))
    entries_1d = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89], dtype=np.int64)
    expected_1d = np.copy(entries_1d)

    lh5.write(array, "array", test_file, group="/data")
    lh5.write_view(
        "/data/array",
        entries_1d,
        "hard_1d",
        test_file,
        link_type="hard",
        group="/views",
    )

    ar_h1d = lh5.read("/views/hard_1d", test_file)
    assert isinstance(ar_h1d, types.Array)
    assert np.all(ar_h1d.nda == expected_1d)
