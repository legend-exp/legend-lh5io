from __future__ import annotations

import logging
import sys

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
    h5d_ent = h5py.h5d.open(h5g, b"entries")
    entries, _, n_rows = _h5_read_ndarray(
        h5d_ent,
        fname,
        f"{oname}/entries",
        start_row=start_row,
        n_rows=n_rows,
        idx=idx,
    )
    h5d_ent.close()
    if not np.issubdtype(entries.dtype, np.integer):
        msg = "entries is not an integer array"
        raise LH5DecodeError(msg, fname, oname)

    # Check that the view type matches the shape of entries
    if view_type == "view{entries}":
        if len(entries.shape) != 1:
            msg = "entries must be a 1D array of integers for view{entries}"
            raise LH5DecodeError(msg, fname, oname)
    elif view_type == "view{slices}":
        if len(entries.shape) != 2 or entries.shape[1] != 2:
            msg = "entries must be a 2D array of shape (n, 2) for view{slices}"
            raise LH5DecodeError(msg, fname, oname)
    else:
        msg = f"unknown view type: {view_type}"
        raise LH5DecodeError(msg, fname, oname)

    # Now read the data, selecting the correct entries
    try:
        h5o = h5py.h5o.open(h5g, b"data")
    except KeyError as e:
        msg = f"view {oname} does not link to data"
        raise LH5DecodeError(msg, fname, oname) from e

    output = composite._h5_read_lgdo(
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
    h5o.close()
    return output
