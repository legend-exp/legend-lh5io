"""Truncate lh5 files. Useful for generating test data."""

from __future__ import annotations

import fnmatch
import logging
import re
import sys
from typing import Protocol

import awkward as ak
from lgdo.types import LGDO, Array, Struct, Table, VectorOfVectors, WaveformTable

import lh5
from lh5 import read, read_as

log = logging.getLogger(__name__)


class LGDOMappable(Protocol):
    def __call__(self, lgdo: str, input_array: ak.Array) -> ak.Array: ...


class LGDOTruncator(LGDOMappable, Protocol):
    """Try to get info on how much rows to read before performing the actual read
    for performance improvement."""

    def start_row(self, lgdo: str) -> int: ...
    def n_rows(self, lgdo: str) -> int: ...


def _is_included(
    name: str,
    *,
    include_list: list[str] | None = None,
    exclude_list: list[str] | None = None,
) -> bool:
    if exclude_list is not None:
        for exc in exclude_list:
            if fnmatch.fnmatch(name, exc.strip("/")):
                return False
    if include_list is not None:
        # match also parents if descendant is included
        include_list_ = []
        for inc in include_list:
            segments = inc.strip("/").split("/")
            include_list_ += ["/".join(segments[: i + 1]) for i in range(len(segments))]
        return any(fnmatch.fnmatch(name, inc.strip("/")) for inc in include_list_)
    return True


def map_lgdo_arrays(
    func: LGDOMappable,
    lgdo: LGDO,
    name: str,
    *,
    include_list: list[str] | None = None,
    exclude_list: list[str] | None = None,
) -> LGDO | None:
    """Map a function acting on awkward arrays contained in the lgdo tree on the tree.
    The tree structure itself is not altered (compare to map in functional languages),
    except if branches are excluded (explicitly or because they are not in the
    include_list passed).
    Also, attributes are propagated unchanged."""
    if not _is_included(name, include_list=include_list, exclude_list=exclude_list):
        msg = f"{name} does not match pattern incl={include_list}, excl={exclude_list}"
        log.debug(msg)
        return None
    if isinstance(lgdo, WaveformTable):
        # before Table as WaveformTable inherits from Table
        t0 = map_lgdo_arrays(
            func,
            lgdo.t0,
            name + "/t0",
            include_list=include_list,
            exclude_list=exclude_list,
        )  # Array
        dt = map_lgdo_arrays(
            func,
            lgdo.dt,
            name + "/dt",
            include_list=include_list,
            exclude_list=exclude_list,
        )  # Array
        values = map_lgdo_arrays(
            func,
            lgdo.values,
            name + "/values",
            include_list=include_list,
            exclude_list=exclude_list,
        )  # AoesA (or VoV?)
        # WaveformTable is treated as all-or-nothing: if any component is excluded,
        # drop the entire WaveformTable.
        if t0 is None or dt is None or values is None:
            msg = (
                f"WaveformTable at {name} excluded due to include/exclude rules "
                f"(t0={t0 is not None}, dt={dt is not None}, values={values is not None})"
            )
            log.debug(msg)
            return None
        return WaveformTable(t0=t0, dt=dt, values=values, attrs=lgdo.attrs)
    if isinstance(lgdo, (Struct, Table)):
        mp = {
            key: map_lgdo_arrays(
                func,
                val,
                name + "/" + key,
                include_list=include_list,
                exclude_list=exclude_list,
            )
            for key, val in lgdo.items()
        }
        mp = {key: val for key, val in mp.items() if val is not None}
        attrs = lgdo.attrs
        if isinstance(lgdo, Table):
            # need to drop datatype in case part of its content will not be included,
            # as table column names are part of datatype attribute
            attrs.pop("datatype", None)
        return type(lgdo)(mp, attrs=attrs)  # inherit attrs from original
    if isinstance(lgdo, (VectorOfVectors, Array)):  # treat as leave here
        ak_array = lgdo.view_as("ak")
        assert isinstance(ak_array, ak.Array)
        ak_array = func(name, ak_array)  # apply the function here
        cls = type(lgdo)
        if isinstance(lgdo, VectorOfVectors):
            lgdo = cls(
                data=ak_array, attrs=lgdo.attrs
            )  # VoV or anything inheriting from it
        elif isinstance(lgdo, Array):
            lgdo = cls(nda=ak_array, attrs=lgdo.attrs)  # might be Array or AoesA
        else:
            msg = "Array-type lgdo cannot be identified"
            raise RuntimeError(msg)
        return lgdo
    msg = f"Cannot map LGDO at {name}: {type(lgdo)}"  # e.g. Scalar. Might find a way to treat this.
    raise RuntimeError(msg)


def map_lgdo_arrays_on_file(
    infile: str,
    outfile: str,
    func: LGDOTruncator,
    overwrite: bool = False,
    *,
    include_list: list | None = None,
    exclude_list: list | None = None,
) -> None:
    """Run function func on all vectorofvectors and all arrays in the file.
    1st arg of func is the name of the lgdo, 2nd is the array within.
    """

    # list of root lgdo names. Actually 2nd level, since / is the real root.
    # But I don't want to load all at once for memory reasons.
    root_lgdo_names = lh5.ls(infile, recursive=False)

    msg = f"objects in {infile}: {root_lgdo_names}"
    log.info(msg)

    first_done = False
    store = lh5.LH5Store()

    # loop over lgdo objects
    for root_name in root_lgdo_names:
        msg = f"mapping over {root_name}"
        log.debug(msg)
        if not _is_included(
            root_name, include_list=include_list, exclude_list=exclude_list
        ):
            # early check to enhance performance
            msg = f"{root_name} does not match pattern incl={include_list}, excl={exclude_list}"
            log.debug(msg)
            continue
        log.info(
            f"Reading {root_name} from {infile} with start_row={func.start_row(root_name)}, n_rows={func.n_rows(root_name)}"
        )
        lh5_obj = read(
            root_name,
            infile,
            start_row=func.start_row(root_name),
            n_rows=func.n_rows(root_name),
        )

        lh5_obj = map_lgdo_arrays(
            func,
            lh5_obj,
            root_name,
            include_list=include_list,
            exclude_list=exclude_list,
        )

        if lh5_obj is None:
            msg = f"{root_name} mapped to Null"
            log.debug(msg)
            continue

        if first_done is False:
            msg = f"creating output file {outfile}"
            log.info(msg)

            store.write(
                lh5_obj,
                root_name,
                outfile,
                wo_mode="overwrite_file"
                if (overwrite and not first_done)
                else "write_safe",
            )
            first_done = True

        else:
            msg = f"appending to {outfile}"
            log.debug(msg)

            # if isinstance(lh5_obj, Table): # should be done already...
            #    _inplace_table_filter(lgdo, lh5_obj, obj_list)

            store.write(lh5_obj, root_name, outfile, wo_mode="append")


def truncate_array_channel(
    input_array: ak.Array,
    table_key_trunc: ak.Array,
    row_in_table_trunc: ak.Array,
    channel_key: int,
):
    row_indices = ak.flatten(row_in_table_trunc[table_key_trunc == channel_key])
    return input_array[row_indices]


class HitBasedTruncator(LGDOTruncator):
    def __init__(self, table_key_trunc, row_in_table_trunc) -> None:
        self.table_key_trunc = table_key_trunc
        self.row_in_table_trunc = row_in_table_trunc

    def _row_indices(self, lgdo: str) -> ak.Array:
        match = re.search(r"^ch(\d+)(?:/|$)", lgdo)
        if match:
            channel_key = int(match.group(1))
            # print(f"Extracted chid: {channel_key}")
        else:
            msg = f"Cannot deduce channel key from {lgdo}"
            raise RuntimeError(msg)
        return ak.flatten(self.row_in_table_trunc[self.table_key_trunc == channel_key])

    def start_row(self, lgdo: str) -> int:
        try:
            row_indices = self._row_indices(lgdo)
            return int(ak.min(row_indices))
        except RuntimeError:
            return 0

    def n_rows(self, lgdo: str) -> int:
        try:
            row_indices = self._row_indices(lgdo)
            return int(ak.max(row_indices) - ak.min(row_indices) + 1)
        except RuntimeError:
            return sys.maxsize

    def __call__(self, lgdo: str, input_array: ak.Array) -> ak.Array:
        # Need to subtract start_row in case it's non-zero,
        # because arrays read with start_row parameter will be shifted
        row_indices = self._row_indices(lgdo) - self.start_row(lgdo)
        return input_array[row_indices]


# this creates the truncator function for hit-ordered arrays
def create_hit_ordered_truncation_func(
    tcm_file: str, length_or_slice: int | slice
) -> LGDOTruncator:
    row_in_table = read_as("hardware_tcm_1/row_in_table", tcm_file, "ak")
    table_key = read_as("hardware_tcm_1/table_key", tcm_file, "ak")
    # truncate
    slice_ = (
        length_or_slice
        if isinstance(length_or_slice, slice)
        else slice(length_or_slice)
    )
    row_in_table_trunc = row_in_table[slice_]
    table_key_trunc = table_key[slice_]  # was :length

    return HitBasedTruncator(
        table_key_trunc=table_key_trunc, row_in_table_trunc=row_in_table_trunc
    )


class EvtBasedTruncator(LGDOTruncator):
    def __init__(self, slice_: slice) -> None:
        self.slice_ = slice_

    def start_row(self, lgdo: str) -> int:  # noqa: ARG002
        return self.slice_.start or 0

    def n_rows(self, lgdo: str) -> int:  # noqa: ARG002
        if self.slice_.stop is None:
            return sys.maxsize
        return self.slice_.stop - (self.slice_.start or 0)

    def __call__(self, lgdo: str, input_array: ak.Array) -> ak.Array:  # noqa: ARG002
        shifted_slice = (
            self.slice_
            if self.slice_.start is None
            else slice(0, self.slice_.stop - self.slice_.start)
        )
        # Need to shift by self.slice_.start
        # because arrays read with start_row parameter will be shifted
        return input_array[shifted_slice]


# this creates the truncator function for evt-ordered arrays (tcm, evt)
def create_evt_ordered_truncation_func(
    length_or_slice: int | slice,
) -> LGDOTruncator:
    slice_ = (
        length_or_slice
        if isinstance(length_or_slice, slice)
        else slice(length_or_slice)
    )

    return EvtBasedTruncator(slice_=slice_)


def truncate(
    infile: str,
    outfile: str,
    length_or_slice: int | slice,
    overwrite: bool = False,
    *,
    tcm_file: str | None = None,  # required for hit-ordered
    include_list: list[str] | None = None,
    exclude_list: list[str] | None = None,
    file_type: str | None = None,  # auto-deduce from name
) -> None:
    """Truncate an LH5 file and write the result to a new file.

    This function produces a truncated copy of `infile` at `outfile` by applying
    `length_or_slice` to array-like LGDOs contained in the file. There are two
    supported truncation modes depending on the ordering of the input file:

    - evt-ordered (file types: ``evt``, ``tcm``, ``any-evt``): arrays are truncated
      by applying `length_or_slice` directly to each array (simple slicing).
    - hit-ordered (file types: ``raw``, ``dsp``, ``hit``, ``any-hit``): rows belong
      to channels and the truncation must preserve only those rows that fall into
      the requested ``length_or_slice`` of the corresponding TCM mapping. For this
      mode a `tcm_file` must be provided; the function reads
      ``hardware_tcm_1/row_in_table`` and ``hardware_tcm_1/table_key`` from the
      TCM file to build the per-channel row selection.

    Parameters
    ----------
    infile:
        Path to the input LH5 file to truncate.
    outfile:
        Path to the resulting truncated LH5 file to create.
    length_or_slice:
        Integer number of rows to keep (keeps first N rows) or a ``slice``
        object to select rows. The semantics differ slightly between
        evt- and hit-ordered files (see above).
    overwrite:
        If True, the output file will be overwritten when created; otherwise
        a safe write/append strategy is used.
    tcm_file:
        Path to a TCM LH5 file. Required for hit-ordered truncation to map
        channel keys to row indices.
    include_list:
        Optional list of fnmatch patterns selecting LGDO paths to include.
    exclude_list:
        Optional list of fnmatch patterns selecting LGDO paths to exclude.
    file_type:
        Optional override of the file type (auto-deduced from `infile` when
        not provided). Use values like ``evt``, ``tcm``, ``raw``, ``dsp``,
        ``hit``, or the generic ``any-evt``/``any-hit``.
    """

    if file_type is None:
        if (match := re.search(r"tier_([a-z0-9]+)\.lh5", infile)) is not None:
            file_type = match.group(1)
        else:
            msg = f"Cannot deduce file type from filename {infile}"
            raise RuntimeError(msg)
    evt_ordered_types = ["evt", "tcm", "any-evt"]
    hit_ordered_types = ["raw", "dsp", "hit", "any-hit"]
    if file_type in evt_ordered_types:
        hit_ordered = False
    elif file_type in hit_ordered_types:
        hit_ordered = True
    else:
        msg = (
            f"Unknown file type {file_type}: impossible to deduce if hit- or evt-ordered. "
            "Use any-hit or any-evt as file type if the file type is not one of the standard types."
        )
        raise RuntimeError(msg)
    msg = "hit-ordered data" if hit_ordered else "evt-ordered data"
    log.info(msg)

    if hit_ordered:
        if tcm_file is None:
            msg = "tcm_file is required for hit-ordered files, but is None"
            raise RuntimeError(msg)
        truncator = create_hit_ordered_truncation_func(
            tcm_file=tcm_file, length_or_slice=length_or_slice
        )
    else:
        truncator = create_evt_ordered_truncation_func(length_or_slice=length_or_slice)
    map_lgdo_arrays_on_file(
        infile=infile,
        outfile=outfile,
        func=truncator,
        overwrite=overwrite,
        include_list=include_list,
        exclude_list=exclude_list,
    )
