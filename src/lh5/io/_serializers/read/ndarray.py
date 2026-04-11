from __future__ import annotations

import logging
import sys
from bisect import bisect_left

import h5py
import hdf5plugin  # noqa: F401, used for compression plugins
import numpy as np
from lgdo.types import Array

from ... import datatype
from ...exceptions import LH5DecodeError
from .utils import read_attrs

log = logging.getLogger(__name__)


def _h5_read_ndarray(
    h5d,
    fname,
    oname,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    obj_buf=None,
    obj_buf_start=0,
):
    if obj_buf is not None and not isinstance(obj_buf, Array):
        msg = "object buffer is not an Array"
        raise LH5DecodeError(msg, fname, oname)

    # compute the number of rows to read
    # we culled idx above for start_row and n_rows, now we have to apply
    # the constraint of the length of the dataset
    try:
        fspace = h5d.get_space()
        ds_n_rows = fspace.shape[0]
    except AttributeError as e:
        msg = "does not seem to be an HDF5 dataset"
        raise LH5DecodeError(msg, fname, oname) from e

    idx = np.asarray(idx) if idx is not None else None
    if idx is None:
        n_rows_to_read = ds_n_rows - start_row
    elif idx.ndim == 1:
        if len(idx) > 0 and idx[-1] >= ds_n_rows:
            log.warning("idx indexed past the end of the array in the file. Culling...")
            n_rows_to_read = bisect_left(idx, ds_n_rows)
            idx = idx[:n_rows_to_read]
            if len(idx) == 0:
                log.warning("idx empty after culling.")
        n_rows_to_read = len(idx)
    elif idx.ndim == 2 and idx.shape[1] == 2:
        if len(idx) > 0 and idx[-1, 1] > ds_n_rows:
            log.warning("idx indexed past the end of the array in the file. Culling...")
            n_pairs = bisect_left(idx[:, 0], ds_n_rows)
            idx = idx[:n_pairs, :]
            if len(idx) == 0:
                log.warning("idx empty after culling.")
            else:
                idx[-1, 1] = min(idx[-1, 1], ds_n_rows)
        n_rows_to_read = (idx[:, 1] - idx[:, 0]).sum()
    n_rows_to_read = min(n_rows_to_read, n_rows)

    if idx is None:
        fspace.select_hyperslab(
            (start_row,) + (0,) * (h5d.rank - 1),
            (1,) * h5d.rank,
            None,
            (n_rows_to_read, *fspace.shape[1:]),
        )
    else:
        fspace.select_none()
        if len(idx) == 0:
            pass
        elif idx.ndim == 1:
            h5py._selector.Selector(fspace).make_selection((idx,))
        elif idx.ndim == 2 and idx.shape[1] == 2:
            _build_h5s_from_list_of_ranges(fspace, idx)
        else:
            msg = "idx array must be 1D or 2D with shape (N, 2)"
            raise ValueError(msg)

    # Now read the array
    if obj_buf is not None and n_rows_to_read > 0:
        buf_size = obj_buf_start + n_rows_to_read
        if len(obj_buf) < buf_size:
            obj_buf.resize(buf_size)

        mspace = h5py.h5s.create_simple(obj_buf.nda.shape)
        mspace.select_hyperslab(
            (obj_buf_start,) + (0,) * (h5d.rank - 1),
            (1,) * h5d.rank,
            None,
            (n_rows_to_read, *fspace.shape[1:]),
        )
        h5d.read(mspace, fspace, obj_buf.nda)
        nda = obj_buf.nda
    elif n_rows == 0:
        tmp_shape = (0, *h5d.shape[1:])
        nda = np.empty(tmp_shape, h5d.dtype)
    else:
        mspace = h5py.h5s.create_simple((n_rows_to_read, *fspace.shape[1:]))
        nda = np.empty(mspace.shape, h5d.dtype)
        h5d.read(mspace, fspace, nda)

    # Finally, set attributes and return objects
    attrs = read_attrs(h5d, fname, oname)

    # special handling for bools
    # (c and Julia store as uint8 so cast to bool)
    if datatype.get_nested_datatype_string(attrs["datatype"]) == "bool":
        nda = nda.astype(np.bool_, copy=False)

    return (nda, attrs, n_rows_to_read)


def _build_h5s_from_list_of_ranges(h5s, ranges):
    # recursive merge algorithm to build h5s from list of ranges
    if ranges.shape[0] > 1:
        space2 = h5s.copy()
        _build_h5s_from_list_of_ranges(h5s, ranges[: ranges.shape[0] // 2, :])
        _build_h5s_from_list_of_ranges(space2, ranges[ranges.shape[0] // 2 :, :])
        h5s.modify_select(space2)
    elif ranges.shape[0] == 1:
        h5s.select_hyperslab(
            (ranges[0, 0],) + (0,) * (len(h5s.shape) - 1),
            (1,) * len(h5s.shape),
            None,
            (ranges[0, 1] - ranges[0, 0], *h5s.shape[1:]),
        )
