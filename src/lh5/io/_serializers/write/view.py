from __future__ import annotations

import logging
from typing import Literal

import h5py
import numpy as np
from lgdo import Array

from ... import utils
from ...exceptions import LH5EncodeError
from .array import _h5_write_array

log = logging.getLogger(__name__)


def _h5_write_view(
    target: h5py.Group | h5py.Dataset | str,
    entries: np.ndarray,
    name: str,
    lh5_file: h5py.File,
    link_type: Literal[None, "hard", "soft", "external"] = None,
    external_file: str | h5py.File | None = None,
    group: str = "/",
    start_row: int = 0,
    n_rows: int | None = None,
    wo_mode: str = "a",
    write_start: int = 0,
    **h5py_kwargs,
):
    if not (
        isinstance(entries, np.ndarray) and issubclass(entries.dtype.type, np.integer)
    ):
        msg = "entries must be ints"
        raise TypeError(msg)

    # Check validity of entries
    if len(entries) > 0 and not (
        np.all(entries.ravel()[1:] > entries.ravel()[:-1]) and entries.ravel()[0] >= 0
    ):
        msg = "entries must be positive and in ascending order"
        raise LH5EncodeError(msg, lh5_file, group, name)

    # check type of view and define datatype string
    if len(entries.shape) == 1:
        view_type = "view{entries}"
    elif len(entries.shape) == 2 and entries.shape[1] == 2:
        view_type = "view{slices}"
    else:
        msg = "entries must have shape (n,) or (n, 2)"
        raise LH5EncodeError(msg, lh5_file, group, name)

    # create/get the view
    wo_mode = utils.normalize_womode(wo_mode)
    overwrite = wo_mode == "o"
    group = utils.get_h5_group(group, lh5_file)
    if wo_mode == "w" and name in group:
        msg = f"can't overwrite '{name}' in wo_mode 'write_safe'"
        raise LH5EncodeError(msg, lh5_file, group, name)
    view = utils.get_h5_group(name, group, overwrite=overwrite)
    if view.attrs.setdefault("datatype", view_type) != view_type and not overwrite:
        msg = f"cannot write a `{view_type}` to '{name}' (`{view.attrs['datatype']}')"
        raise LH5EncodeError(msg, lh5_file, group, name)

    # Deduce link type if needed
    if isinstance(external_file, h5py.File):
        external_file = external_file.filename
    if link_type is None:
        if isinstance(target, (h5py.Group, h5py.Dataset)):
            if external_file is not None and external_file != target.file.filename:
                msg = f"external_file {external_file} does not match target file {target.file.filename}"
                raise LH5EncodeError(msg, lh5_file, group, name)
            if external_file is None and target.file != lh5_file:
                external_file = target.file.filename

            link_type = "hard" if external_file is None else "external"
        elif isinstance(target, str):
            link_type = "soft" if external_file is None else "external"

    # Add the link
    if link_type == "external":
        if external_file is None:
            msg = "external_file required for external links"
            raise ValueError(msg)
        if isinstance(target, (h5py.Group, h5py.Dataset)):
            target = target.name

        link = view.get("data", getlink=True)
        if link is None:
            view["data"] = h5py.ExternalLink(external_file, target)
        elif not (
            isinstance(link, h5py.ExternalLink)
            and link.path == target
            and link.filename == external_file
        ):
            if not overwrite:
                msg = "existing HDF5 link is different from target. Cannot append"
                raise LH5EncodeError(msg, lh5_file, group, name)
            del view["data"]
            view["data"] = h5py.ExternalLink(external_file, target)

    elif link_type == "hard":
        if external_file is not None:
            msg = "external_file must be None for hard links"
            raise ValueError(msg)
        if not isinstance(target, (h5py.Group, h5py.Dataset)):
            target = utils.get_h5_group(target, lh5_file)

        link = view.get("data", getlink=True)
        if link is None:
            view["data"] = target
        elif not (isinstance(link, h5py.HardLink) and view["data"].id == target.id):
            if not overwrite:
                msg = "existing HDF5 link is different from target"
                raise LH5EncodeError(msg, lh5_file, group, name)
            del view["data"]
            view["data"] = target

    elif link_type == "soft":
        if external_file is not None:
            msg = "external_file must be None for soft links"
            raise ValueError(msg)
        if isinstance(target, (h5py.Group, h5py.Dataset)):
            target = target.name

        link = view.get("data", getlink=True)
        if link is None:
            view["data"] = h5py.SoftLink(target)
        elif not (isinstance(link, h5py.SoftLink) and link.path == target):
            if not overwrite:
                msg = "existing HDF5 link is different from target"
                raise LH5EncodeError(msg, lh5_file, group, name)
            del view["data"]
            view["data"] = h5py.SoftLink(target)

    # write entries
    _h5_write_array(
        Array(entries),
        "entries",
        lh5_file,
        group=view,
        start_row=start_row,
        n_rows=n_rows,
        wo_mode=wo_mode,
        write_start=write_start,
        **h5py_kwargs,
    )

    # remove datatype from the entry attrs; it is not a part of the view spec
    entries_attrs = view["entries"].attrs
    if "datatype" in entries_attrs:
        del view["entries"].attrs["datatype"]
