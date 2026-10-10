from __future__ import annotations

import logging
import sys
from contextlib import closing

import h5py
import numpy as np

from ...exceptions import LH5DecodeError
from . import composite
from .ndarray import _h5_read_ndarray

log = logging.getLogger(__name__)


def _h5_read_view(
    h5g,
    fname,
    oname,
    view_type,
    start_row=0,
    n_rows=sys.maxsize,
    idx=None,
    field_mask=None,
    obj_buf=None,
    obj_buf_start=0,
    decompress=True,
):
    # Read the entries for the view
    try:
        h5d_ent = h5py.h5d.open(h5g, b"entries")
    except KeyError:
        msg = "entries not found"
        raise LH5DecodeError(msg, fname, oname) from None

    with closing(h5d_ent):
        if view_type == "view{entries}":
            if len(h5d_ent.shape) != 1:
                msg = "entries must be a 1D array of integers for view{entries}"
                raise LH5DecodeError(msg, fname, oname)
            entries, _, n_rows = _h5_read_ndarray(
                h5d_ent,
                fname,
                f"{oname}/entries",
                start_row=start_row,
                n_rows=n_rows,
                idx=idx,
            )
        elif view_type == "view{slices}":
            if len(h5d_ent.shape) != 2 or h5d_ent.shape[1] != 2:
                msg = "entries must be a 2D array of shape (n, 2) for view{slices}"
                raise LH5DecodeError(msg, fname, oname)

            # for slices, we must manually handle start row/n_rows/idx
            entries, _, _ = _h5_read_ndarray(
                h5d_ent,
                fname,
                f"{oname}/entries",
            )

            if len(entries) == 0:
                if idx is not None and len(idx) > 0:
                    log.warning(
                        "idx indexed past the end of the array in the file. Culling..."
                    )
                    log.warning("idx empty after culling.")

            elif idx is None:
                # get entries, adjusted for start_row and n_rows
                cum_entries = np.cumsum(np.diff(entries, axis=1)).ravel()
                stop_row = min(n_rows + start_row, cum_entries[-1])

                if stop_row <= start_row:
                    entries = np.empty((0), dtype=entries.dtype)
                else:
                    i_start = np.searchsorted(cum_entries, start_row, "right")
                    i_stop = np.searchsorted(cum_entries, stop_row, "left")

                    entries = entries[i_start : i_stop + 1, :]
                    entries[0, 0] += start_row - (
                        cum_entries[i_start - 1] if i_start > 0 else 0
                    )
                    entries[-1, 1] += stop_row - cum_entries[i_stop]

            elif idx.ndim == 1:
                # list of indices: calculate the data array indices by finding the
                # index within each range that corresponds to the idx list
                # Note: indices should already be sorted and trimmed for start_row and
                # n_rows by composite.py
                new_idx = np.empty_like(idx)
                cum_entries = np.cumsum(np.diff(entries, axis=1)).ravel()
                idx_range = np.searchsorted(idx, cum_entries, "left")
                i_start = 0
                for i_range, i_end in enumerate(idx_range):
                    offset = entries[i_range, 0] - (
                        cum_entries[i_range - 1] if i_range > 0 else 0
                    )
                    new_idx[i_start:i_end] = idx[i_start:i_end] + offset
                    i_start = i_end

                if i_end < len(idx):
                    log.warning(
                        "idx indexed past the end of the array in the file. Culling..."
                    )
                    new_idx = new_idx[:i_end]
                entries = new_idx
                if len(entries) == 0:
                    log.warning("idx empty after culling.")

            elif idx.ndim == 2 and idx.shape[1] == 2:
                # list of ranges: calculate the data array indices by finding the
                # index within each range that corresponds to the idx list
                # Note: indices should already be sorted and trimmed for start_row and
                # n_rows by composite.py
                new_idx = []
                cum_entries = np.cumsum(np.diff(entries, axis=1)).ravel()

                for row_start, row_stop in idx:
                    i_start = np.searchsorted(cum_entries, row_start, "right")
                    i_stop = np.searchsorted(cum_entries, row_stop, "left")
                    if i_start >= len(cum_entries):
                        break

                    this_entries = np.copy(entries[i_start : i_stop + 1, :])
                    this_entries[0, 0] += row_start - (
                        cum_entries[i_start - 1] if i_start > 0 else 0
                    )
                    if i_stop < len(cum_entries):
                        this_entries[-1, 1] += row_stop - cum_entries[i_stop]
                    new_idx.append(this_entries)

                if len(idx) > 0 and i_stop >= len(cum_entries):
                    log.warning(
                        "idx indexed past the end of the array in the file. Culling..."
                    )
                entries = (
                    np.concatenate(new_idx)
                    if len(new_idx) > 0
                    else np.zeros((0), dtype="int")
                )
                if len(entries) == 0:
                    log.warning("idx empty after culling.")

        else:
            msg = f"unknown view type: {view_type}"
            raise LH5DecodeError(msg, fname, oname)

    if not np.issubdtype(entries.dtype, np.integer):
        msg = "entries is not an integer array"
        raise LH5DecodeError(msg, fname, oname)

    # Now read the data, selecting the correct entries
    try:
        h5o = h5py.h5o.open(h5g, b"data")
    except KeyError as e:
        msg = f"view {oname} does not link to data"
        raise LH5DecodeError(msg, fname, oname) from e

    with closing(h5o):
        return composite._h5_read_lgdo(
            h5o,
            fname,
            oname,
            start_row=0,
            n_rows=sys.maxsize,
            idx=entries,
            field_mask=field_mask,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
            decompress=decompress,
        )
