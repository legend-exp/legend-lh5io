"""Tests for the file-space strategy, chunk policy and page-buffered reads.

Covers the three layout changes: files default to the compact (FSM) strategy
with paged aggregation as an explicit opt-in via ``fs_page_size``; the
``chunk_nbytes`` byte target replaces h5py first-write auto-chunking; and
reads accept a ``page_buffer``, falling back gracefully on non-paged files.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from lgdo import types

import lh5
from lh5.io import settings


def _strategy(path):
    with h5py.File(path) as f:
        return f.id.get_create_plist().get_file_space_strategy()


def _make_table(n=1000):
    return types.Table(
        size=n,
        col_dict={
            "col1": types.Array(np.arange(n, dtype="float64")),
            "col2": types.Array(np.random.default_rng(1).random(n, dtype="float32")),
        },
    )


def test_write_defaults_to_fsm_strategy(tmptestdir):
    out = f"{tmptestdir}/fsm_default.lh5"
    lh5.write(_make_table(), "tbl", out, wo_mode="overwrite_file")
    strategy, _, _ = _strategy(out)
    assert strategy == h5py.h5f.FSPACE_STRATEGY_FSM_AGGR


def test_write_fs_page_size_opts_into_paged(tmptestdir):
    out = f"{tmptestdir}/paged.lh5"
    lh5.write(_make_table(), "tbl", out, wo_mode="overwrite_file", fs_page_size="64KiB")
    strategy, _, _ = _strategy(out)
    assert strategy == h5py.h5f.FSPACE_STRATEGY_PAGE
    with h5py.File(out) as f:
        assert f.id.get_create_plist().get_file_space_page_size() == 64 * 1024


def test_write_page_buffer_alias_warns_and_pages(tmptestdir):
    out = f"{tmptestdir}/paged_legacy.lh5"
    with pytest.warns(DeprecationWarning, match="fs_page_size"):
        lh5.write(_make_table(), "tbl", out, wo_mode="overwrite_file", page_buffer=4096)
    strategy, _, _ = _strategy(out)
    assert strategy == h5py.h5f.FSPACE_STRATEGY_PAGE


def test_store_write_defaults_to_fsm(tmptestdir):
    out = f"{tmptestdir}/store_fsm.lh5"
    lh5.LH5Store().write(_make_table(), "tbl", out, wo_mode="overwrite_file")
    strategy, _, _ = _strategy(out)
    assert strategy == h5py.h5f.FSPACE_STRATEGY_FSM_AGGR


def test_default_chunking_hits_byte_target(tmptestdir):
    out = f"{tmptestdir}/chunks.lh5"
    # write in two small appends: chunk shape must NOT freeze at 100 rows
    tbl = _make_table(100)
    lh5.write(tbl, "tbl", out, wo_mode="overwrite_file")
    lh5.write(tbl, "tbl", out, wo_mode="append")
    target = settings.DEFAULT_HDF5_SETTINGS["chunk_nbytes"]
    with h5py.File(out) as f:
        d = f["tbl/col1"]  # float64
        assert d.chunks == (target // 8,)
        d = f["tbl/col2"]  # float32
        assert d.chunks == (target // 4,)


def test_explicit_chunks_override_byte_target(tmptestdir):
    out = f"{tmptestdir}/chunks_explicit.lh5"
    lh5.write(_make_table(), "tbl", out, wo_mode="overwrite_file", chunks=(50,))
    with h5py.File(out) as f:
        assert f["tbl/col1"].chunks == (50,)


def test_chunking_2d_rows(tmptestdir):
    out = f"{tmptestdir}/chunks2d.lh5"
    aoesa = types.ArrayOfEqualSizedArrays(nda=np.zeros((100, 256), dtype="float32"))
    lh5.write(aoesa, "wfs", out, wo_mode="overwrite_file")
    target = settings.DEFAULT_HDF5_SETTINGS["chunk_nbytes"]
    with h5py.File(out) as f:
        rows = target // (256 * 4)
        assert f["wfs"].chunks == (rows, 256)


def test_read_with_page_buffer_on_paged_file(tmptestdir):
    out = f"{tmptestdir}/paged_read.lh5"
    tbl = _make_table()
    lh5.write(tbl, "tbl", out, wo_mode="overwrite_file", fs_page_size="64KiB")
    back = lh5.read("tbl", out, page_buffer="1MiB")
    assert np.array_equal(back["col1"].nda, tbl["col1"].nda)


def test_read_with_page_buffer_falls_back_on_fsm_file(tmptestdir):
    out = f"{tmptestdir}/fsm_read.lh5"
    tbl = _make_table()
    lh5.write(tbl, "tbl", out, wo_mode="overwrite_file")
    # non-paged file: must silently fall back to an unbuffered open
    back = lh5.read("tbl", out, page_buffer="1MiB")
    assert np.array_equal(back["col2"].nda, tbl["col2"].nda)


def test_store_read_page_buffer(tmptestdir):
    out = f"{tmptestdir}/store_paged_read.lh5"
    tbl = _make_table()
    lh5.write(tbl, "tbl", out, wo_mode="overwrite_file", fs_page_size="64KiB")
    store = lh5.LH5Store(page_buffer="1MiB")
    back = store.read("tbl", out)
    assert len(back) == 1000
    assert np.array_equal(back["col1"].nda, tbl["col1"].nda)


def test_iterator_page_buffer_passthrough(tmptestdir):
    out = f"{tmptestdir}/it_paged.lh5"
    tbl = _make_table()
    lh5.write(tbl, "tbl", out, wo_mode="overwrite_file", fs_page_size="64KiB")
    it = lh5.LH5Iterator(out, "tbl", buffer_len=300, page_buffer="1MiB")
    assert it.lh5_st.page_buffer == 2**20
    n = sum(len(chunk) for chunk in it)
    assert n == 1000


def test_env_default_page_buffer(tmptestdir, monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_PAGE_BUFFER", 2**20)
    out = f"{tmptestdir}/env_paged.lh5"
    tbl = _make_table()
    lh5.write(tbl, "tbl", out, wo_mode="overwrite_file", fs_page_size="64KiB")
    back = lh5.read("tbl", out)  # no explicit page_buffer: uses default
    assert np.array_equal(back["col1"].nda, tbl["col1"].nda)
    assert lh5.LH5Store().page_buffer == 2**20


@pytest.mark.parametrize(
    ("size", "nbytes"),
    [("16MiB", 16 * 2**20), ("4kb", 4000), ("1.5GiB", int(1.5 * 2**30)), (512, 512)],
)
def test_parse_datasize(size, nbytes):
    assert settings.parse_datasize(size) == nbytes
