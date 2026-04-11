from __future__ import annotations

import logging
import sys

import h5py
import numba
import numpy as np
from lgdo.types import (
    Array,
    VectorOfVectors,
)

from ... import datatype as dtypeutils
from ...exceptions import LH5DecodeError
from .array import (
    _h5_read_array,
)
from .utils import read_attrs

log = logging.getLogger(__name__)


def _h5_read_vector_of_vectors(
    h5g,
    fname,
    oname,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    obj_buf=None,
    obj_buf_start=0,
):
    if obj_buf is not None and not isinstance(obj_buf, VectorOfVectors):
        msg = "object buffer is not a VectorOfVectors"
        raise LH5DecodeError(msg, fname, oname)

    # read out cumulative_length
    cumulen_buf = None if obj_buf is None else obj_buf.cumulative_length
    h5d_cl = h5py.h5d.open(h5g, b"cumulative_length")
    idx = np.asarray(idx) if idx is not None else None

    cumulative_length, n_rows_read = _h5_read_array(
        h5d_cl,
        fname,
        f"{oname}/cumulative_length",
        start_row=start_row,
        n_rows=n_rows,
        idx=idx,
        obj_buf=cumulen_buf,
        obj_buf_start=obj_buf_start,
    )
    # get a view of just what was read out for cleaner code below
    this_cumulen_nda = cumulative_length.nda[
        obj_buf_start : obj_buf_start + n_rows_read
    ]

    # fix this_cumulen_nda so that cumulative_lengths match in-memory layout
    # and find the ranges of values to read out for flattened_data (fd_idx)
    fd_start = 0
    if idx is None or n_rows_read == 0:
        fd_idx = None

        # determine the start_row and n_rows for the flattened_data readout
        if start_row > 0 and n_rows_read > 0:
            # need to read out the cumulen sample -before- the first sample
            # read above in order to get the starting row of the first
            # vector to read out in flattened_data
            fspace = h5d_cl.get_space()
            fspace.select_elements([[start_row - 1]])
            mspace = h5py.h5s.create(h5py.h5s.SCALAR)
            fd_start = np.empty((), h5d_cl.dtype)
            h5d_cl.read(mspace, fspace, fd_start)

            # check limits for values that will be used subsequently
            if this_cumulen_nda[-1] < fd_start:
                log.debug(
                    f"this_cumulen_nda[-1] = {this_cumulen_nda[-1]}, "
                    f"fd_start = {fd_start}, "
                    f"start_row = {start_row}, "
                    f"n_rows_read = {n_rows_read}"
                )
                msg = (
                    f"cumulative_length non-increasing between entries "
                    f"{start_row} and {start_row + n_rows_read}"
                )
                raise LH5DecodeError(msg, fname, oname)

        # subtract offset of flattened_data from cumulative_length
        this_cumulen_nda -= fd_start

        # determine the number of rows for the flattened_data readout
        fd_n_rows = this_cumulen_nda[-1] if n_rows_read > 0 else 0

    elif idx.ndim == 1:
        # get the starting indices for each array in flattened data:
        # the starting index for array[i] is cumulative_length[i-1]
        fstarts, _ = _h5_read_array(
            h5d_cl,
            fname,
            f"{oname}/cumulative_length",
            start_row=start_row,
            n_rows=n_rows,
            idx=(idx if idx[0] > 0 else idx[1:]) - 1,
            obj_buf=Array(
                shape=len(this_cumulen_nda), fill_val=0, dtype=this_cumulen_nda.dtype
            ),
            obj_buf_start=0 if idx[0] > 0 else 1,
        )

        # get 2D array of start/stop and cumulative lengths
        mask = this_cumulen_nda > fstarts  # remove len 0 entries
        fd_idx = np.stack([fstarts[mask], this_cumulen_nda[mask]], axis=1)
        np.cumsum(this_cumulen_nda - fstarts, out=this_cumulen_nda)
        fd_n_rows = this_cumulen_nda[-1]

    elif idx.ndim == 2 and idx.shape[1] == 2:
        # get starting indices for contiguous blocks of arrays in flattened data:
        # the starting index for block[i] is cumulative_length[idx[i,0]-1]
        fstarts, _ = _h5_read_array(
            h5d_cl,
            fname,
            f"{oname}/cumulative_length",
            start_row=start_row,
            n_rows=n_rows,
            idx=(idx[:, 0] if idx[0, 0] > 0 else idx[1:, 0]) - 1,
            obj_buf=Array(shape=len(idx), fill_val=0, dtype=this_cumulen_nda.dtype),
            obj_buf_start=0 if idx[0, 0] > 0 else 1,
        )

        # get 2D array of start/stop and cumulative
        fd_idx = _h5_get_2D_fd_idx_and_cumulen(fstarts.nda, this_cumulen_nda)
        fd_n_rows = this_cumulen_nda[-1]

    h5d_cl.close()

    # If we started with a partially-filled buffer, add the
    # appropriate offset for the start of the in-memory flattened
    # data for this read.
    fd_buf_start = np.uint32(0)
    if obj_buf_start > 0:
        fd_buf_start = cumulative_length.nda[obj_buf_start - 1]
        this_cumulen_nda += fd_buf_start

    # Now prepare the object buffer if necessary
    fd_buf = None
    if obj_buf is not None:
        fd_buf = obj_buf.flattened_data
        # grow fd_buf if necessary to hold the data
        fdb_size = fd_buf_start + fd_n_rows
        if len(fd_buf) < fdb_size:
            fd_buf.resize(fdb_size)

    # now read
    h5o = h5py.h5o.open(h5g, b"flattened_data")
    h5a_dtype = h5py.h5a.open(h5o, b"datatype")
    val = np.empty((), "O")
    h5a_dtype.read(val)
    lgdotype = dtypeutils.datatype(val.item().decode())
    if lgdotype is Array:
        _func = _h5_read_array
    elif lgdotype is VectorOfVectors:
        _func = _h5_read_vector_of_vectors
    else:
        msg = "type {lgdotype.__name__} is not supported"
        raise LH5DecodeError(msg, fname, f"{oname}/flattened_data")

    flattened_data, _ = _func(
        h5o,
        fname,
        f"{oname}/flattened_data",
        start_row=fd_start,
        n_rows=fd_n_rows,
        idx=fd_idx,
        obj_buf=fd_buf,
        obj_buf_start=fd_buf_start,
    )
    h5o.close()

    if obj_buf is not None:
        # if the buffer is partially filled, cumulative_length will be invalid
        # (i.e. non monotonically increasing). Let's fix that but filling the
        # rest of the array with the length of flattened_data
        if n_rows_read > 0:
            end = obj_buf_start + n_rows_read
            obj_buf.cumulative_length.nda[end:] = obj_buf.cumulative_length.nda[end - 1]

        return obj_buf, n_rows_read

    return (
        VectorOfVectors(
            flattened_data=flattened_data,
            cumulative_length=cumulative_length,
            attrs=read_attrs(h5g, fname, oname),
        ),
        n_rows_read,
    )


@numba.njit
def _h5_get_2D_fd_idx_and_cumulen(fstarts, this_cumulen_nda):
    # helper to get 2D ranges for flattened data and update cumulen in place
    fd_idx = np.empty((len(fstarts), 2), dtype=fstarts.dtype)
    i_fd = 0
    i_start = 0
    fd_idx[0, 0] = fstarts[0]
    last_cumulen = fstarts[0]
    start = fstarts[0]
    for i_cl in range(len(this_cumulen_nda)):
        # advance through fstarts if we have moved into a new block of arrays
        if i_start < len(fstarts) - 1 and this_cumulen_nda[i_cl] > fstarts[i_start + 1]:
            i_start += 1
            start = fstarts[i_start]

        # if we have a gap between ranges, add a new fd_idx pair and use start
        if last_cumulen < start:
            fd_idx[i_fd, 1] = last_cumulen
            i_fd += 1
            fd_idx[i_fd, 0] = start
            last_cumulen = start

        # correct cumulens for this range
        this_len = this_cumulen_nda[i_cl] - last_cumulen
        last_cumulen = this_cumulen_nda[i_cl]
        if i_cl == 0:
            this_cumulen_nda[i_cl] = this_len
        else:
            this_cumulen_nda[i_cl] = this_cumulen_nda[i_cl - 1] + this_len
    fd_idx[i_fd, 1] = last_cumulen
    return fd_idx[: i_fd + 1]
