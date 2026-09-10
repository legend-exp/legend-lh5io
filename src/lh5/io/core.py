from __future__ import annotations

import bisect
import inspect
import logging
import sys
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from inspect import signature
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from lgdo import types
from numpy.typing import ArrayLike

from . import _serializers
from .exceptions import LH5DecodeError
from .utils import normalize_womode, read_n_rows

log = logging.getLogger(__name__)


def read(
    name: str,
    lh5_file: str | Path | h5py.File | Sequence[str | Path | h5py.File],
    start_row: int = 0,
    n_rows: int = sys.maxsize,
    idx: ArrayLike | Sequence[ArrayLike] = None,
    use_h5idx: bool = False,
    field_mask: Mapping[str, bool] | Sequence[str] | None = None,
    obj_buf: types.LGDO = None,
    obj_buf_start: int = 0,
    decompress: bool = True,
    locking: bool = False,
) -> types.LGDO | tuple[types.LGDO, int]:
    """Read LH5 object data from a file.

    Parameters
    ----------
    name
        Name of the LH5 object to be read (including its group path).
    lh5_file
        The file(s) containing the object to be read out. If a list of
        files, array-like object data will be concatenated into the output
        object.
    start_row
        Starting entry for the object read (for array-like objects). For a
        list of files, only applies to the first file.
    n_rows
        The maximum number of rows to read (for array-like objects). The
        actual number of rows read will be returned as one of the return
        values (see below).
    idx
        For NumPy-style "fancy indexing" for the read to select only
        some rows, e.g. after applying some cuts to particular columns.
        Only selection along the first axis is supported, so tuple
        arguments must be one-tuples.  A 2D array of shape (N, 2) can be.
        used to provide a list of ranges. To use with a list of files,
        can pass in a list/tuple of `idx`'s (one for each file) or use a
        long contiguous list (e.g. built from a previous identical read).
        If used in conjunction with `start_row` and `n_rows`, will be
        sliced to obey those constraints, where `n_rows` is interpreted as
        the (max) number of *selected* values (in `idx`) to be read out.

        Note
        ----
        If a list of arrays is provided with the same length as number of
        files, it will be split among the files even if it was intended as
        a list of ranges. In that case, explicitly provide the list as a
        numpy array of shape (n, 2). If a list of arrays has a length different
        from the number of files, attempt to interpret as shape (n, 2) array.
    use_h5idx
        deprecated and has no effect.
    field_mask
        For tables and structs, determines which fields get read out.
        Nested struct elements can be accessed by using ``/`` as a separator
        (e.g. refer to field ``a`` inside the table ``table`` which is stored
        inside the struct ``struct`` as ``struct/table/a``). If a dict
        is used, a default dict will be made with the default set to the
        opposite of the first element in the dict. This way if one specifies
        a few fields at ``False``, all but those fields will be read out,
        while if one specifies just a few fields as ``True``, only those
        fields will be read out. If a list is provided, the listed fields
        will be set to ``True``, while the rest will default to ``False``.
    obj_buf
        Read directly into memory provided in `obj_buf`. Note: the buffer
        will be resized to accommodate the data retrieved.
    obj_buf_start
        Start location in ``obj_buf`` for read. For concatenating data to
        array-like objects.
    decompress
        Decompress data encoded with LGDO's compression routines right
        after reading. The option has no effect on data encoded with HDF5
        built-in filters, which is always decompressed upstream by HDF5.
    locking
        Lock HDF5 file while reading

    Returns
    -------
    object
        the read-out object
    """
    if use_h5idx:
        log.warning("use_h5idx is deprecated and has no effect.")

    close_after = False
    if isinstance(lh5_file, h5py.File):
        try:
            lh5_obj = lh5_file[name]
        except KeyError as oe:
            raise LH5DecodeError(oe, lh5_file, name) from oe
    elif isinstance(lh5_file, (str, Path)):
        try:
            lh5_file = h5py.File(str(Path(lh5_file)), mode="r", locking=locking)
        except (OSError, FileExistsError) as oe:
            raise LH5DecodeError(oe, lh5_file) from oe

        close_after = True
        try:
            lh5_obj = lh5_file[name]
        except KeyError as ke:
            raise LH5DecodeError(str(ke), lh5_file, name) from ke
    else:
        if obj_buf is not None:
            obj_buf.resize(obj_buf_start)
        else:
            obj_buf_start = 0

        if idx is None:
            pass
        elif (
            isinstance(idx, Sequence)
            and len(idx) == len(lh5_file)
            and not all(np.isscalar(i) for i in idx)
        ):
            # list of idx for each file
            idx = [np.array(idx_i) if idx_i is not None else None for idx_i in idx]
        else:
            # long list of idx to be split among files
            try:
                idx = np.array(idx)
                if idx.ndim == 0:
                    idx = idx.reshape(1)
                assert idx.ndim == 1 or (idx.ndim == 2 and idx.shape[1] == 2)
            except (TypeError, ValueError, AssertionError) as e:
                msg = "idx must be of shape (), (n,) or (n, 2)"
                raise ValueError(msg) from e

        for i, h5f in enumerate(lh5_file):
            if isinstance(idx, list):
                # a list of lists: must be one per file
                idx_i = idx[i]
            elif idx is not None:
                # find subset of idx contained in this file
                n_rows_i = read_n_rows(name, h5f)
                if len(idx) == 0:
                    # no more idx to read
                    break
                if idx.ndim == 1:
                    # split idx into part for this file and part for remaining
                    if idx.dtype == np.dtype("?"):
                        idx_i = idx[:n_rows_i]
                        idx = idx[n_rows_i:]
                    else:
                        n_rows_to_read_i = bisect.bisect_left(idx, n_rows_i)
                        idx_i = idx[:n_rows_to_read_i]
                        idx = idx[n_rows_to_read_i:] - n_rows_i
                elif idx.ndim == 2 and idx.shape[1] == 2:
                    i_last_valid = bisect.bisect_left(idx[:, 1], n_rows_i)
                    if i_last_valid >= len(idx):
                        idx_i = idx
                        idx = np.array([], dtype=idx.dtype)
                    elif idx[i_last_valid, 0] >= n_rows_i:
                        idx_i = idx[:i_last_valid]
                        idx = idx[i_last_valid:] - n_rows_i
                    else:
                        # if n_rows_i is in the middle of a range, split the range
                        idx_i = idx[: i_last_valid + 1]
                        idx = idx[i_last_valid:] - n_rows_i
                        idx_i[-1, 1] = n_rows_i
                        idx[0, 0] = 0
            else:
                idx_i = None

            obj_buf_start_i = len(obj_buf) if obj_buf else 0
            n_rows_i = n_rows - (obj_buf_start_i - obj_buf_start)

            obj_buf = read(
                name,
                h5f,
                start_row=start_row if i == 0 else 0,
                n_rows=n_rows_i,
                idx=idx_i,
                field_mask=field_mask,
                obj_buf=obj_buf,
                obj_buf_start=obj_buf_start_i,
                decompress=decompress,
            )

            if obj_buf is None or (len(obj_buf) - obj_buf_start) >= n_rows:
                return obj_buf
        return obj_buf

    if idx is not None:
        try:
            idx = np.array(idx)
            if idx.ndim == 0:
                idx = idx.reshape(1)
            if idx.dtype == np.dtype("?"):
                idx = np.where(idx)[0]
            assert idx.ndim == 1 or (idx.ndim == 2 and idx.shape[1] == 2)
        except (TypeError, ValueError, AssertionError) as e:
            msg = "idx must be of shape (), (n,) or (n, 2)"
            raise ValueError(msg) from e

    try:
        obj, n_rows_read = _serializers._h5_read_lgdo(
            lh5_obj.id,
            lh5_obj.file.filename,
            lh5_obj.name,
            start_row=start_row,
            n_rows=n_rows,
            idx=idx,
            field_mask=field_mask,
            obj_buf=obj_buf,
            obj_buf_start=obj_buf_start,
            decompress=decompress,
        )
        with suppress(AttributeError):
            obj.resize(obj_buf_start + n_rows_read)
        return obj
    finally:
        if close_after:
            lh5_file.close()


def write(
    obj: types.LGDO,
    name: str,
    lh5_file: str | Path | h5py.File,
    group: str | h5py.Group = "/",
    start_row: int = 0,
    n_rows: int | None = None,
    wo_mode: str = "append",
    write_start: int = 0,
    page_buffer: int = 0,
    **h5py_kwargs,
) -> None:
    """Write an LGDO into an LH5 file.

    If the `obj` :class:`.LGDO` has a `compression` attribute, its value is
    interpreted as the algorithm to be used to compress `obj` before
    writing to disk. The type of `compression` can be:

    string, kwargs dictionary, hdf5plugin filter
      interpreted as the name of a built-in or custom `HDF5 compression
      filter <https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline>`_
      (``"gzip"``, ``"lzf"``, :mod:`hdf5plugin` filter object etc.) and
      passed directly to :meth:`h5py.Group.create_dataset`.

    :class:`.WaveformCodec` object
      If `obj` is a :class:`.WaveformTable` and ``obj.values`` holds the
      attribute, compress ``values`` using this algorithm. More
      documentation about the supported waveform compression algorithms at
      :mod:`.lgdo.compression`.

    If the `obj` :class:`.LGDO` has a `hdf5_settings` attribute holding a
    dictionary, it is interpreted as a list of keyword arguments to be
    forwarded directly to :meth:`h5py.Group.create_dataset` (exactly like
    the first format of `compression` above). This is the preferred way to
    specify HDF5 dataset options such as chunking etc. If compression
    options are specified, they take precedence over those set with the
    `compression` attribute.

    Note
    ----
    The `compression` LGDO attribute takes precedence over the default HDF5
    compression settings. The `hdf5_settings` attribute takes precedence
    over `compression`. These attributes are not written to disk.

    Note
    ----
    HDF5 compression is skipped for the `encoded_data.flattened_data`
    dataset of :class:`.VectorOfEncodedVectors` and
    :class:`.ArrayOfEncodedEqualSizedArrays`.

    Parameters
    ----------
    obj
        LH5 object. If object is array-like, writes `n_rows` starting from
        `start_row` in `obj`.
    name
        name of the object in the output HDF5 file.
    lh5_file
        HDF5 file name or :class:`h5py.File` object.
    group
        HDF5 group name or :class:`h5py.Group` object in which `obj` should
        be written.
    start_row
        first row in `obj` to be written.
    n_rows
        number of rows in `obj` to be written.
    wo_mode
        - ``write_safe`` or ``w``: only proceed with writing if the
          object does not already exist in the file.
        - ``append`` or ``a``: append along axis 0 (the first dimension)
          of array-like objects and array-like subfields of structs.
          :class:`~.lgdo.scalar.Scalar` objects get overwritten.
        - ``overwrite`` or ``o``: replace data in the file if present,
          starting from `write_start`. Note: overwriting with `write_start` =
          end of array is the same as ``append``.
        - ``overwrite_file`` or ``of``: delete file if present prior to
          writing to it. `write_start` should be 0 (it's ignored).
        - ``append_column`` or ``ac``: append fields/columns from an
          :class:`~.lgdo.struct.Struct` `obj` (and derived types such as
          :class:`~.lgdo.table.Table`) only if there is an existing
          :class:`~.lgdo.struct.Struct` in the `lh5_file` with the same `name`.
          If there are matching fields, it errors out. If appending to a
          ``Table`` and the size of the new column is different from the size
          of the existing table, it errors out.
    write_start
        row in the output file (if already existing) to start overwriting
        from.
    page_buffer
        enable paged aggregation with a buffer of this size in bytes.
        Only used when creating a new file. Useful when writing a file
        with a large number of small datasets. This is a short-hand for
        ``(fs_strategy="page", fs_page_size=page_buffer)``
    **h5py_kwargs
        additional keyword arguments forwarded to
        :meth:`h5py.Group.create_dataset` to specify, for example, an HDF5
        compression filter to be applied before writing non-scalar
        datasets. **Note: `compression` ignored if compression is specified
        as an `obj` attribute.**
    """

    wo_mode = normalize_womode(wo_mode)
    if page_buffer > 0 and (
        (isinstance(lh5_file, (str, Path)) and not Path(lh5_file).is_file())
        or wo_mode in ("w", "of")
    ):
        h5py_kwargs.update(
            {
                "fs_strategy": "page",
                "fs_page_size": page_buffer,
            }
        )
    return _serializers._h5_write_lgdo(
        obj,
        name,
        lh5_file,
        group=group,
        start_row=start_row,
        n_rows=n_rows,
        wo_mode=wo_mode,
        write_start=write_start,
        **h5py_kwargs,
    )


def read_as(
    name: str,
    lh5_file: str | Path | h5py.File | Sequence[str | Path | h5py.File],
    library: str,
    **kwargs,
) -> Any:
    """Read LH5 data from disk straight into a third-party data format view.

    This function is nothing more than a shortcut chained call to
    :func:`.read` and to :meth:`.LGDO.view_as`.

    Parameters
    ----------
    name
        LH5 object name on disk.
    lh5_file
        LH5 file name.
    library
        string ID of the third-party data format library (``np``, ``pd``,
        ``ak``, etc).

    See Also
    --------
    .read, .LGDO.view_as
    """
    # determine which keyword arguments should be forwarded to read() and which
    # should be forwarded to view_as()
    read_kwargs = inspect.signature(read).parameters.keys()

    kwargs1 = {}
    kwargs2 = {}
    for k, v in kwargs.items():
        if k in read_kwargs:
            kwargs1[k] = v
        else:
            kwargs2[k] = v

    # read the LGDO from disk
    # NOTE: providing a buffer does not make much sense
    obj = read(name, lh5_file, **kwargs1)

    if isinstance(obj, tuple):
        obj = obj[0]

    # and finally return a view
    return obj.view_as(library, **kwargs2)


def write_view(
    target: str | h5py.Group | h5py.Dataset,
    entries: np.ndarray,
    name: str,
    lh5_file: str | Path | h5py.File,
    link_type: str = None,
    external_file: str | Path | None = None,
    group: str | h5py.Group = "/",
    start_row: int = 0,
    n_rows: int | None = None,
    wo_mode: str = "append",
    write_start: int = 0,
    page_buffer: int = 0,
    **h5py_kwargs,
):
    """Create a view of an array or table in an LH5 file.
    A LH5 view is a reference to an existing array or table in the
    same or another LH5 file, along with an entry list specifying the
    rows to read out. The entry list can be stored as a 1D array of
    row indices or a 2D array of row ranges (start, end); the entries
    correspond to the outer-most dimension of the target array or table only.

    Parameters
    ----------
    target
        The target array or table to be referenced. Can be a string
        corresponding to the path of the target in the file, or an
        :class:`h5py.Group` or :class:`h5py.Dataset` object. A string
        must be provided if the view is a SoftLink or ExternalLink.
    entries
        A 1D array of row indices or a (n,2) shape array of row ranges (start, end).
        Indices must be in ascending order.
    name
        Name of the view in the output LH5 file.
    lh5_file
        HDF5 file name or :class:`h5py.File` object.
    link_type
        Type of link used to reference the target. Can be ``hard`` (default),
        ``soft`` or ``external``. If ``external``, the `external_file` arg
        is required.
    external_file
        External HDF5 file containing target referenced by view. Only used
        if `link_type` is ``external``.
    group
        HDF5 group name or :class:`h5py.Group` object in which `obj` should
        be written.
    start_row
        first row in `obj` to be written.
    n_rows
        number of rows in `obj` to be written.
    wo_mode
        - ``write_safe`` or ``w``: only proceed with writing if the
          object does not already exist in the file.
        - ``append`` or ``a``: append along axis 0 (the first dimension)
          of array-like objects and array-like subfields of structs.
          :class:`~.lgdo.scalar.Scalar` objects get overwritten.
        - ``overwrite`` or ``o``: replace data in the file if present,
          starting from `write_start`. Note: overwriting with `write_start` =
          end of array is the same as ``append``.
        - ``overwrite_file`` or ``of``: delete file if present prior to
          writing to it. `write_start` should be 0 (it's ignored).
        - ``append_column`` or ``ac``: append fields/columns from an
          :class:`~.lgdo.struct.Struct` `obj` (and derived types such as
          :class:`~.lgdo.table.Table`) only if there is an existing
          :class:`~.lgdo.struct.Struct` in the `lh5_file` with the same `name`.
          If there are matching fields, it errors out. If appending to a
          ``Table`` and the size of the new column is different from the size
          of the existing table, it errors out.
    write_start
        row in the output file (if already existing) to start overwriting
        from.
    page_buffer
        enable paged aggregation with a buffer of this size in bytes.
        Only used when creating a new file. Useful when writing a file
        with a large number of small datasets. This is a short-hand for
        ``(fs_strategy="page", fs_page_size=page_buffer)``
    **h5py_kwargs
        additional keyword arguments forwarded to
        :meth:`h5py.Group.create_dataset` to specify, for example, an HDF5
        compression filter to be applied before writing non-scalar
        datasets. **Note: `compression` ignored if compression is specified
        as an `obj` attribute.**
    """
    wo_mode = normalize_womode(wo_mode)
    if page_buffer > 0 and (
        (isinstance(lh5_file, (str, Path)) and not Path(lh5_file).is_file())
        or wo_mode in ("w", "of")
    ):
        h5py_kwargs.update(
            {
                "fs_strategy": "page",
                "fs_page_size": page_buffer,
            }
        )

    file_kwargs = {
        k: h5py_kwargs[k] for k in h5py_kwargs & signature(h5py.File).parameters.keys()
    }
    h5py_kwargs = {k: h5py_kwargs[k] for k in h5py_kwargs - file_kwargs.keys()}

    with (
        h5py.File(
            lh5_file,
            mode="w" if wo_mode == "of" or not Path(lh5_file).exists() else "a",
            **file_kwargs,
        )
        if not isinstance(lh5_file, h5py.File)
        else nullcontext(lh5_file) as fh
    ):
        return _serializers._h5_write_view(
            target,
            entries,
            name,
            fh,
            link_type=link_type,
            external_file=external_file,
            group=group,
            start_row=start_row,
            n_rows=n_rows,
            wo_mode=wo_mode,
            write_start=write_start,
            **h5py_kwargs,
        )
