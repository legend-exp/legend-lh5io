from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator, Mapping
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import chain
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Literal

import awkward as ak
import numpy as np
import pandas as pd
from hist import Hist, axis
from lgdo.types import LGDOCollection, Table
from lgdo.units import default_units_registry as ureg
from numpy.typing import NDArray
from rich import console, progress

from .store import LH5Store
from .utils import expand_path

log = logging.getLogger(__name__)


class LH5Iterator(Iterator):
    """Iterate over chunks of entries from LH5 files.

    The iterator reads ``buffer_len`` entries at a time from one or more
    files.  The LGDO instance returned at each iteration is reused to avoid
    reallocations, so copy the data if it should be preserved.

    Examples
    --------
    Iterate through a table one chunk at a time and call ``process`` on each chunk::

            from lh5 import LH5Iterator
            for table in LH5Iterator("data.lh5", "geds/raw/energy", buffer_len=100):
                process(table)

    ``LH5Iterator`` can also be used for random access::

            it = LH5Iterator(files, groups)
            table = it.read(i_entry)

    In case of multiple files or an entry selection, ``i_entry`` refers to the
    global event index across all files.

    When instantiating an iterator you must provide a list of files and the
    HDF5 groups to read.  Optional parameters allow field masking, event
    selection and pairing the iterator with a "friend" iterator that is read in
    parallel.  Several properties are available to obtain the provenance of the
    data currently loaded:

    - ``current_i_entry`` -- index within the entry list of the first entry in
      the buffer
    - ``current_local_entries`` -- entry numbers relative to the file the data
      came from
    - ``current_global_entries`` -- entry number relative to the full dataset
    - ``current_files`` -- file name corresponding to each entry in the buffer
    - ``current_groups`` -- group name corresponding to each entry in the
      buffer
    """

    def __init__(
        self,
        lh5_files: str | Collection[str | Collection[str]],
        groups: str | Collection[str | Collection[str]],
        *,
        base_path: str = "",
        entry_list: Collection[int] | Collection[Collection[int]] = None,
        entry_mask: Collection[bool] | Collection[Collection[bool]] = None,
        i_start: int = 0,
        n_entries: int = None,
        field_mask: Mapping[str, bool] | Collection[str] = None,
        group_data: Mapping[Collection] | ak.Array = None,
        buffer_len: int = "100*MB",
        file_cache: int = 10,
        ds_map: NDArray[int] = None,
        friend: Collection[LH5Iterator] = None,
        friend_prefix: str = "",
        friend_suffix: str = "",
        safe_mode: bool = True,
        h5py_open_mode: str = "r",
    ) -> None:
        """
        Constructor for LH5Iterator. Must provide a file or collection of
        files, and an lh5 group or collection of groups to read data from.

        Collections of files and groups can be nested. At the top level, we
        expect the same number of entries (one set of files to one set of groups).
        For each corresponding pair of sets, we will loop over each pairing of
        a file and group, with an inner loop over the groups to minimize the
        opening of files. Wildcards used for files will be expanded and applied
        in the inner loop (i.e. each file in a wildcard will read the same groups).
        If groups is an un-nested collection of strings, use all groups for all files.

        Examples
        --------
        Read "ch1/table" and "ch2/table" from "file.lh5"::

            LH5Iterator("/path/to/file.lh5", ["ch1/table", "ch2/table"])

        Read "ch1" from all lh5 files in "/path1", then read "ch1" and "ch2" from
        "/path2/file.lh5", and then read "ch1/2/3" from both "file1.lh5" and "file2.lh5"::

            LH5Iterator(
                ["/path1/*.lh5", "/path2/file.lh5", ["/path3/file1.lh5", "/path3/file2.lh5"]],
                ["ch1/table", ["ch1/table", "ch2/table"], ["ch1/table", "ch2/table", "ch3/table"]]
            )

        Parameters
        ----------
        lh5_files
            file(s) to read from (see above). May include wildcards and environment variables.
        groups
            HDF5 group(s) to read (see above).
        base_path
            directory path prepended to all file names.
        entry_list
            list of entry numbers to read. If a nested list is provided,
            expect one top-level list for each file, containing a list of
            local entries. If a list of ints is provided, use global entries.
        entry_mask
            mask of entries to read. If a list of arrays is provided, expect
            one for each file. Ignore if a selection list is provided.
        i_start
            index of first entry to start at when iterating
        n_entries
            number of entries to read before terminating iteration
        field_mask
            mask of which fields to read. See :meth:`LH5Store.read` for
            more details.
        group_data
            mapping of values corresponding to each provided lh5 group. Values
            will be duplicated for each entry in each dataset, corresponding to
            the correct group, and added to the output table. This should have
            same structure as ``groups``.
        buffer_len
            number of entries in tables yielded by iterator. Can be provided as
            a value with a unit of memory; in this case, use the estimated number
            of rows that will yield tables that require the provided memory.
            Defaults to ``"100*MB"``.
        file_cache
            maximum number of files to keep open at a time
        ds_map
            cumulative entries in datasets corresponding to file/group pairs.
            This can be provided on construction to speed up random or sparse
            access; otherwise, we sequentially read the size of each group.
            WARNING: no checks for accuracy are performed so only use this if
            you know what you are doing!
        friend
            a \"friend\" LH5Iterator that will be joined to this one, and read
            in parallel. The friend should have the same length and entry list.
            Each iteration will return a single LH5 Table containing columns from
            both iterators. The buffer_len will be set to the minimum of the two.
        friend_prefix
            prefix for fields in friend iterator for resolving naming conflicts
        friend_suffix
            suffix for fields in friend iterator for resolving naming conflicts
        safe_mode
            if ``True`` and a friend iterator has a different number of files,
            groups, or elements in a dataset, raise an Exception.
        h5py_open_mode
            file open mode used when acquiring file handles. ``r`` (default)
            opens files read-only while ``a`` allows opening files for
            write-appending as well.
        """

        if h5py_open_mode == "read":
            h5py_open_mode = "r"
        if h5py_open_mode == "append":
            h5py_open_mode = "a"
        if h5py_open_mode not in ["r", "a"]:
            msg = f"unknown h5py_open_mode '{h5py_open_mode}'"
            raise ValueError(msg)

        self.lh5_st = LH5Store(
            base_path=base_path, keep_open=file_cache, default_mode=h5py_open_mode
        )

        # convert lh5_files into a nested list
        if isinstance(lh5_files, str):
            self.lh5_files = [expand_path(lh5_files, list=True, base_path=base_path)]
        elif not isinstance(lh5_files, Collection):
            msg = "lh5_files must be a string or list of strings"
            raise ValueError(msg)
        else:
            self.lh5_files = []
            for f in lh5_files:
                if isinstance(f, str):
                    self.lh5_files.append(
                        expand_path(f, list=True, base_path=base_path)
                    )
                elif isinstance(f, Collection) and all(
                    isinstance(name, str) for name in f
                ):
                    for name in f:
                        flist = expand_path(name, list=True, base_path=base_path)
                        if len(flist) == 0:
                            log.warning(f"{name} did not match any files")
                        self.lh5_files.append(flist)
                else:
                    msg = "lh5_files must be a collection of strings with up to two levels of nesting"
                    raise ValueError(msg)

        if isinstance(group_data, pd.DataFrame):
            group_data = ak.Array(group_data.to_dict(orient="list"))

        # convert groups into a nested list
        if isinstance(groups, str):
            self.groups = [[groups]] * len(self.lh5_files)
            if group_data is not None:
                group_data = [group_data] * len(self.lh5_files)
        elif not isinstance(groups, Collection):
            msg = "group must be a string or collection of strings"
            raise ValueError(msg)
        elif all(isinstance(g, str) for g in groups):
            self.groups = [groups] * len(self.lh5_files)
            if group_data is not None:
                group_data = [group_data] * len(self.lh5_files)
        else:
            self.groups = []
            for g in groups:
                if isinstance(g, str):
                    g = [g]  # noqa: PLW2901
                elif not (
                    isinstance(g, Collection)
                    and all(isinstance(name, str) for name in g)
                ):
                    msg = "groups must be a collection of strings with up to two levels of nesting"
                    raise ValueError(msg)
                self.groups.append(g)

        # check that groups and lh5_files are compatible
        if len(self.groups) != len(self.lh5_files):
            msg = "lh5_files and groups could not be broadcast onto one another"
            raise ValueError(msg)

        # build map of cumulative number of datasets for file/group pairs and number of files/groups
        self._fggroups = np.array(
            [
                (len(f) * len(g), len(g))
                for f, g in zip(self.lh5_files, self.groups, strict=True)
            ]
        )
        self._fggroups[:, 0] = np.cumsum(self._fggroups[:, 0])
        self.group_data = None
        self._broadcast_group_data = None

        # track offset and number of datasets; used by _select_datasets
        self._ds_offset = 0
        self.n_datasets = self._fggroups[-1, 0] if len(self._fggroups) > 0 else 0

        if entry_list is not None and entry_mask is not None:
            msg = "entry_list and entry_mask arguments are mutually exclusive"
            raise ValueError(msg)

        # Map to last row in each file
        if ds_map is None:
            self.ds_map = np.full(self.n_datasets, np.iinfo("q").max, "q")
        else:
            self.ds_map = np.array(ds_map)

        # Map to last iterator entry for each file
        self.entry_map = np.full(self.n_datasets, np.iinfo("q").max, "q")

        self.friend = []
        self.friend_prefix = []
        self.friend_suffix = []

        if self.n_datasets == 0:
            msg = f"can't open any files from {lh5_files}"
            raise RuntimeError(msg)

        # lh5 buffer will contain all fields and be used for I/O (with a field mask)
        self.lh5_buffer = self.lh5_st.get_buffer(
            self.get_group(0),
            self.get_file(0),
            size=0,
        )

        def get_available_fields(tab):
            ret = set()
            if not isinstance(tab, Table):
                return ret

            for k, v in tab.items():
                ret |= {k}
                if isinstance(v, Table):
                    ret |= {f"{k}/{field}" for field in get_available_fields(v)}
            return ret

        self.available_fields = get_available_fields(self.lh5_buffer)

        # set field mask and buffer length
        # Note: group_data is added to table here!
        self.reset_field_mask(field_mask)
        self.buffer_len = buffer_len

        # add group data
        self.set_group_data(group_data)

        # Attach the friend(s)
        self.safe_mode = safe_mode
        if friend is None:
            friend = []
        elif isinstance(friend, LH5Iterator):
            friend = [friend]

        if len(friend) > 0:
            fr_buf_len = min(fr.buffer_len for fr in friend)
            self.buffer_len = min(self.buffer_len, fr_buf_len)

        if isinstance(friend_prefix, str):
            friend_prefix = [friend_prefix] * len(friend)
        if isinstance(friend_suffix, str):
            friend_suffix = [friend_suffix] * len(friend)
        for fr, prefix, suffix in zip(
            friend, friend_prefix, friend_suffix, strict=False
        ):
            self.add_friend(fr, prefix, suffix)

        self.i_start = i_start
        self.n_entries = n_entries
        self.current_i_entry = 0
        self.next_i_entry = 0
        self.current_local_entries = np.empty(0, "q")
        self.current_global_entries = np.empty(0, "q")

        # List of entry indices from each file
        self.local_entry_list = None
        self.global_entry_list = None
        if entry_list is not None:
            entry_list = list(entry_list)
            if len(entry_list) > 0 and isinstance(entry_list[0], (int, np.integer)):
                self.local_entry_list = [None] * self.n_datasets
                self.global_entry_list = np.array(entry_list, "q")
                self.global_entry_list.sort()

            else:
                self.local_entry_list = [[]] * self.n_datasets
                for i_ds, local_list in enumerate(entry_list):
                    self.local_entry_list[i_ds] = np.array(local_list, "q")
                    self.local_entry_list[i_ds].sort()

        elif entry_mask is not None:
            # Convert entry mask into an entry list
            if isinstance(entry_mask, pd.Series):
                entry_mask = entry_mask.to_numpy()
            if isinstance(entry_mask, np.ndarray):
                self.local_entry_list = [None] * self.n_datasets
                self.global_entry_list = np.nonzero(entry_mask)[0]
            else:
                self.local_entry_list = [[]] * self.n_datasets
                for i_ds, local_mask in enumerate(entry_mask):
                    self.local_entry_list[i_ds] = np.nonzero(local_mask)[0]

    def __del__(self):
        if hasattr(self, "lh5_st") and self.lh5_st is not None:
            self.lh5_st.close()

    def get_file(self, i_ds: int) -> str:
        """Get file name for dataset i_ds"""
        if i_ds < 0 or i_ds >= self.n_datasets:
            msg = f"dataset index {i_ds} out of range"
            raise IndexError(msg)
        i_ds += self._ds_offset

        i_set = np.searchsorted(self._fggroups[:, 0], i_ds, "right")
        offset = self._fggroups[i_set - 1, 0] if i_set > 0 else 0
        ngroups = self._fggroups[i_set, 1]
        return self.lh5_files[i_set][(i_ds - offset) // ngroups]

    def get_group(self, i_ds: int) -> str:
        """Get group name for dataset i_ds"""
        if i_ds < 0 or i_ds >= self.n_datasets:
            msg = f"dataset index {i_ds} out of range"
            raise IndexError(msg)
        i_ds += self._ds_offset

        i_set = np.searchsorted(self._fggroups[:, 0], i_ds, "right")
        offset = self._fggroups[i_set - 1, 0] if i_set > 0 else 0
        ngroups = self._fggroups[i_set, 1]
        return self.groups[i_set][(i_ds - offset) % ngroups]

    def get_group_data(self, i_ds: int) -> ak.Record | None:
        """Get group data for dataset i_ds"""
        if self.group_data is None:
            return None

        if i_ds < 0 or i_ds >= self.n_datasets:
            msg = f"dataset index {i_ds} out of range"
            raise IndexError(msg)
        i_ds += self._ds_offset

        i_set = np.searchsorted(self._fggroups[:, 0], i_ds, "right")
        gp_data = self.group_data[i_set]
        offset = self._fggroups[i_set - 1, 0] if i_set > 0 else 0
        ngroups = self._fggroups[i_set, 1]

        ret_data = {}
        for f in gp_data.fields:
            val = (
                gp_data[f]
                if self._broadcast_group_data[f]
                else gp_data[f][(i_ds - offset) % ngroups]
            )
            if val is None:
                # convert None to a valid value based on the dtype
                if f in self.lh5_buffer:
                    dtype = self.lh5_buffer[f].dtype
                else:
                    # Yeesh...Get the dtype of an optional numpy array
                    dtype = self.group_data[f].type
                    while not isinstance(dtype, ak.types.NumpyType):
                        dtype = dtype.content
                    dtype = np.dtype(str(dtype))

                if dtype.kind == "u":
                    val = np.iinfo(dtype).max
                elif dtype.kind == "i":
                    val = np.iinfo(dtype).min
                elif dtype.kind in ("f", "c"):
                    val = np.nan
                elif dtype.kind in ("S", "T", "U"):
                    val = ""
                else:
                    msg = f"Could not handle missing values for {dtype}"
                    raise ValueError(msg)

            ret_data[f] = val
        return ak.Record(ret_data)

    def _get_ds_cumlen(self, i_ds: int) -> int:
        """Helper to get cumulative dataset length of file/groups"""
        if i_ds < 0:
            return 0
        if i_ds >= self.n_datasets:
            return self._get_ds_cumlen(self.n_datasets - 1)
        fcl = self.ds_map[i_ds]

        # if we haven't already calculated, calculate for all files up to i_ds
        if fcl == np.iinfo("q").max:
            i_start = np.searchsorted(self.ds_map, np.iinfo("q").max)
            fcl = self.ds_map[i_start - 1] if i_start > 0 else 0

            for i in range(i_start, i_ds + 1):
                if i >= self.n_datasets:
                    break
                fcl += self.lh5_st.read_n_rows(self.get_group(i), self.get_file(i))
                self.ds_map[i] = fcl
        return fcl

    def _get_ds_cumentries(self, i_ds: int) -> int:
        """Helper to get cumulative iterator entries in file/groups"""
        if i_ds < 0:
            return 0
        if i_ds >= self.n_datasets:
            return self._get_ds_cumentries(self.n_datasets - 1)
        n = self.entry_map[i_ds]

        # if we haven't already calculated, calculate for all files up to i_ds
        if n == np.iinfo("q").max:
            i_start = np.searchsorted(self.entry_map, np.iinfo("q").max)
            n = self.entry_map[i_start - 1] if i_start > 0 else 0

            for i in range(i_start, i_ds + 1):
                elist = self.get_ds_entrylist(i)
                fcl = self._get_ds_cumlen(i)
                if elist is None:
                    # no entry list provided
                    n = fcl
                else:
                    n += len(elist)
                    # check that file entries fall inside of file
                    if len(elist) > 0 and elist[-1] >= fcl:
                        log.warning(f"Found entries out of range for file {i}")
                        n += np.searchsorted(elist, fcl, "right") - len(elist)
                self.entry_map[i] = n
        return n

    def get_ds_entrylist(self, i_ds: int) -> np.ndarray:
        """Helper to get entry list for dataset"""
        if i_ds < 0 or i_ds >= self.n_datasets:
            msg = f"dataset index {i_ds} out of range"
            raise IndexError(msg)

        # If no entry list is provided
        if self.local_entry_list is None:
            return None

        elist = self.local_entry_list[i_ds]
        if elist is None:
            # Get local entrylist for this dataset from global entry list
            f_start = self._get_ds_cumlen(i_ds - 1)
            f_end = self._get_ds_cumlen(i_ds)
            i_start = self._get_ds_cumentries(i_ds - 1)
            i_stop = np.searchsorted(self.global_entry_list, f_end, "right")
            elist = np.array(self.global_entry_list[i_start:i_stop], "q") - f_start
            self.local_entry_list[i_ds] = elist
        return elist

    def get_global_entrylist(self) -> np.ndarray:
        """Get global entry list, constructing it if needed"""
        if self.global_entry_list is None and self.local_entry_list is not None:
            self.global_entry_list = np.zeros(len(self), "q")
            for i_ds in range(self.n_datasets):
                i_start = self._get_ds_cumentries(i_ds - 1)
                i_stop = self._get_ds_cumentries(i_ds)
                f_start = self._get_ds_cumlen(i_ds - 1)
                self.global_entry_list[i_start:i_stop] = (
                    self.get_ds_entrylist(i_ds) + f_start
                )
        return self.global_entry_list

    def read(self, i_entry: int, n_entries: int | None = None) -> Table:
        """Read a chunk of events starting at global entry `i_entry`."""
        self.lh5_buffer.resize(0)

        if n_entries is None:
            n_entries = self.buffer_len
        elif n_entries == 0:
            return self.lh5_buffer
        elif n_entries > self.buffer_len:
            msg = "n_entries cannot be larger than buffer_len"
            raise ValueError(msg)

        if len(self.current_local_entries) < n_entries:
            self.current_local_entries = np.empty(n_entries, "q")
            self.current_global_entries = np.empty(n_entries, "q")

        # if dataset hasn't been opened yet, search through datasets
        # sequentially until we find the right one
        i_ds = np.searchsorted(self.entry_map, i_entry, "right")
        if i_ds < self.n_datasets and self.entry_map[i_ds] == np.iinfo("q").max:
            while i_ds < self.n_datasets and i_entry >= self._get_ds_cumentries(i_ds):
                i_ds += 1

        if i_ds == self.n_datasets:
            return self.lh5_buffer
        local_i_entry = i_entry - self._get_ds_cumentries(i_ds - 1)

        while len(self.lh5_buffer) < n_entries and i_ds < self.n_datasets:
            # Loop through datasets
            local_idx = self.get_ds_entrylist(i_ds)
            if local_idx is not None and len(local_idx) == 0:
                i_ds += 1
                local_i_entry = 0
                continue

            i_local = local_i_entry if local_idx is None else local_idx[local_i_entry]

            buf_start = len(self.lh5_buffer)

            if len(self.field_mask) > 0 or not isinstance(self.lh5_buffer, Table):
                self.lh5_buffer = self.lh5_st.read(
                    self.get_group(i_ds),
                    self.get_file(i_ds),
                    start_row=i_local,
                    n_rows=n_entries - buf_start,
                    idx=local_idx,
                    field_mask=self.field_mask,
                    obj_buf=self.lh5_buffer,
                    obj_buf_start=buf_start,
                )
            else:
                self.lh5_buffer.resize(
                    min(n_entries, self._get_ds_cumentries(i_ds) - i_entry)
                )

            if local_idx is None:
                self.current_local_entries[buf_start : len(self.lh5_buffer)] = (
                    np.arange(
                        local_i_entry, local_i_entry + len(self.lh5_buffer) - buf_start
                    )
                )
            else:
                self.current_local_entries[buf_start : len(self.lh5_buffer)] = (
                    local_idx[
                        local_i_entry : local_i_entry + len(self.lh5_buffer) - buf_start
                    ]
                )
            self.current_global_entries[buf_start : len(self.lh5_buffer)] = (
                self.current_local_entries[buf_start : len(self.lh5_buffer)]
                + self._get_ds_cumlen(i_ds - 1)
            )

            if self.group_data is not None:
                data = self.get_group_data(i_ds)
                for f in data.fields:
                    self.lh5_buffer[f][buf_start:] = data[f]

            i_ds += 1
            local_i_entry = 0

        self.current_i_entry = i_entry
        if len(self.current_local_entries) > len(self.lh5_buffer):
            self.current_local_entries = np.resize(
                self.current_local_entries, len(self.lh5_buffer)
            )
            self.current_global_entries = np.resize(
                self.current_global_entries, len(self.lh5_buffer)
            )

        for friend in self.friend:
            friend.read(i_entry, n_entries)

            # check if entries in all datasets to current are all equal
            if (
                self.safe_mode
                and self._get_ds_cumentries(i_ds) != friend._get_ds_cumentries(i_ds)
                and np.any(self.entry_map[:i_ds] != friend.entry_map[:i_ds])
            ):
                i_diff = np.argmax(self.entry_map[:i_ds] != friend.entry_map[:i_ds])
                msg = (
                    f"with safe_mode = True, require that datasets have same sizes between friends. "
                    f"File {self.get_file(i_diff)} group {self.get_group(i_diff)} differs from "
                    f"file {friend.lh5_files[i_diff]} group {friend.groups[i_diff]}."
                )
                raise RuntimeError(msg)

        return self.lh5_buffer

    @property
    def buffer_len(self):
        return self._buffer_len

    @buffer_len.setter
    def buffer_len(self, buffer_len: str | ureg.Quantity | int):
        if isinstance(buffer_len, str):
            buffer_len = ureg.Quantity(buffer_len)
        if isinstance(buffer_len, ureg.Quantity):
            for i_ds in range(self.n_datasets):
                f = self.get_file(i_ds)
                g = self.get_group(i_ds)
                n_row = max(self.lh5_st.read_n_rows(g, f), 1)
                size_in_bytes = self.lh5_st.read_size_in_bytes(g, f) / n_row
                if size_in_bytes > 0:
                    buffer_len = int(buffer_len / (size_in_bytes * ureg.B))
                    break
            if isinstance(buffer_len, ureg.Quantity):
                buffer_len = int(buffer_len / ureg.B)

        self._buffer_len = buffer_len
        for fr in self.friend:
            fr.buffer_len = buffer_len

    def add_friend(self, friend: LH5Iterator, prefix: str = "", suffix: str = ""):
        """Add a friend which will be iterated alongside this, returning a Table
        joining the contents of each.

        Parameters
        ----------
        friend
            LH5Iterator to be friended to this one
        prefix
            string prepended to field names; useful for disambiguating conflicts
        suffix
            string appended to field names; useful for disambiguating conflicts
        """
        if not isinstance(friend, LH5Iterator):
            msg = "Friend must be an LH5Iterator"
            raise ValueError(msg)

        if self.safe_mode and self.n_datasets != friend.n_datasets:
            msg = (
                f"with safe_mode = True, friend iterator must have same number of datasets. "
                f"Found {self.n_datasets} in self, and {friend.n_datasets} in friend."
            )
            raise RuntimeError(msg)

        # set buffer_lens to be equal
        if friend.buffer_len > self.buffer_len:
            friend.buffer_len = self.buffer_len
        elif friend.buffer_len < self.buffer_len:
            self.buffer_len = friend.buffer_len
        friend.lh5_buffer.resize(len(self.lh5_buffer))

        self.friend += [friend]
        self.friend_prefix += [prefix]
        self.friend_suffix += [suffix]
        self.lh5_buffer.join(
            friend.lh5_buffer,
            keep_mine=True,
            prefix=prefix,
            suffix=suffix,
        )

    def set_group_data(self, group_data: Mapping[Collection] | ak.Array):
        """
        Set group data which will be joined to each table based on the current file/group.
        Should have same structure of groups, with one entry per file and either one subentry
        per group or one entry that will be broadcast across all groups.

        Note: unlike in the constructor, the group_data will not be broadcast over all files!
        """
        old_fields = (
            set(self.group_data.fields) if self.group_data is not None else set()
        )

        if group_data is not None:
            self.group_data = ak.Array(group_data)
            if not self.group_data.fields:
                msg = "group_data must have named fields"
                raise ValueError(msg)
            if len(self.group_data) != len(self.lh5_files):
                msg = "group_data must have same structure as groups"
                raise ValueError(msg)

            # check if we need to broadcast group data over groups
            self._broadcast_group_data = {}
            for f in self.group_data.fields:
                if self.group_data[f].ndim > 1:
                    if ak.any(ak.num(self.group_data[f]) != self._fggroups[:, 1]):
                        msg = f"group_data field '{f}' has incompatible length with groups"
                        raise ValueError(msg)
                    self._broadcast_group_data[f] = False
                else:
                    self._broadcast_group_data[f] = True

            # replace old group data with new in buffer
            for f in old_fields:
                self.lh5_buffer.remove_field(f)

            tb_gd = deepcopy(Table(ak.Array([self.get_group_data(0)])))
            tb_gd.resize(len(self.lh5_buffer))
            self.lh5_buffer.join(tb_gd)
        else:
            self.group_data = None
            self._broadcast_group_data = None

    def reset_field_mask(
        self,
        mask: Collection[str]
        | Mapping[str, bool]
        | Collection[Collection[str]]
        | Collection[Mapping[str, bool]]
        | None,
        warn_missing=True,
    ):
        """Replaces the field mask of this iterator and any friends with mask.

        - If ``None``, set this and all friends to have no mask.
        - If a collection of strings or mapping from strings to bools, set the mask
          for this and all friends; in the case of a conflict, use first column found. If a
          prefix or suffix is included for the friend, it must be included in this mask
        - If a collection of collections, use the first item to set this mask, and subsequent
          items to set friend masks. In this case, do not include prefixes or suffixes in names
        """
        if mask is None:
            self.field_mask = self.available_fields

            for fr in self.friend:
                fr.reset_field_mask(None)

            remaining_fields = set()

        elif isinstance(mask, Mapping):
            mask = {k.replace(".", "/"): v for k, v in mask.items()}

            self.field_mask = {
                field: mask[field] for field in self.available_fields if field in mask
            }
            mask = {
                field: mask[field] for field in mask if field not in self.field_mask
            }

            for fr, pre, suf in zip(
                self.friend, self.friend_prefix, self.friend_suffix, strict=False
            ):
                mask_lookup = {
                    f"{pre}{field}{suf}": field for field in fr.available_fields
                }
                fr_mask = {
                    mask_lookup[field]: mask[field]
                    for field in mask_lookup
                    if field in mask
                }
                fr.reset_field_mask(fr_mask)
                mask = {
                    field: mask[field] for field in mask if field not in mask_lookup
                }

            remaining_fields = mask

        elif isinstance(mask, Collection) and all(isinstance(m, str) for m in mask):
            mask = {f.replace(".", "/") for f in mask}
            self.field_mask = mask & set(self.available_fields)
            mask -= self.field_mask

            for fr, pre, suf in zip(
                self.friend, self.friend_prefix, self.friend_suffix, strict=False
            ):
                mask_lookup = {
                    f"{pre}{field}{suf}": field for field in fr.available_fields
                }
                fr_mask = {mask_lookup[field] for field in mask if field in mask_lookup}
                fr.reset_field_mask(fr_mask)
                mask -= set(mask_lookup)

            remaining_fields = mask

        # Create a new buffer and move any elements from the old into the new
        def copy_data(old_buffer, new_buffer):
            if isinstance(new_buffer, Table):
                for k, v in new_buffer.items():
                    if k in old_buffer:
                        new_buffer[k] = copy_data(v, old_buffer[k])
                return new_buffer
            return old_buffer

        if len(self.field_mask) > 0 or not isinstance(self.lh5_buffer, Table):
            self.lh5_buffer = copy_data(
                self.lh5_buffer,
                self.lh5_st.get_buffer(
                    self.get_group(0),
                    self.get_file(0),
                    size=len(self.lh5_buffer),
                    field_mask=self.field_mask,
                ),
            )
        else:
            self.lh5_buffer = Table(size=0)

        for fr, pre, suf in zip(
            self.friend, self.friend_prefix, self.friend_suffix, strict=False
        ):
            self.lh5_buffer.join(
                fr.lh5_buffer,
                keep_mine=True,
                prefix=pre,
                suffix=suf,
            )

        # join Table with group metadata; repeat first record to initialize
        if self.group_data is not None:
            # deepcopy required to prevent ownership conflict
            tb_gd = deepcopy(Table(ak.Array([self.get_group_data(0)])))
            tb_gd.resize(len(self.lh5_buffer))
            if isinstance(remaining_fields, dict):
                for f in tb_gd:
                    if f in remaining_fields:
                        del remaining_fields[f]
            else:
                remaining_fields -= set(tb_gd)
            self.lh5_buffer.join(tb_gd)

        if warn_missing and len(remaining_fields) > 0:
            log.warning(f"Fields {remaining_fields} in field mask were not found")

    @property
    def current_files(self) -> NDArray[str]:
        """Return list of file names for entries in buffer"""
        cur_files = np.zeros(len(self.lh5_buffer), dtype=np.dtypes.StringDType)
        i_ds = np.searchsorted(self.entry_map, self.current_i_entry, "right")
        ds_start = self._get_ds_cumentries(i_ds - 1)
        i_local = self.current_i_entry - ds_start
        i = 0

        while i < len(cur_files):
            # number of entries to read from this file
            ds_end = self._get_ds_cumentries(i_ds)
            n = min(ds_end - ds_start - i_local, len(cur_files) - i)
            cur_files[i : i + n] = self.get_file(i_ds)

            i_ds += 1
            ds_start = ds_end
            i_local = 0
            i += n

        return cur_files

    @property
    def current_groups(self) -> NDArray[str]:
        """Return list of group names for entries in buffer"""
        cur_groups = np.zeros(len(self.lh5_buffer), dtype=np.dtypes.StringDType)
        i_ds = np.searchsorted(self.entry_map, self.current_i_entry, "right")
        ds_start = self._get_ds_cumentries(i_ds - 1)
        i_local = self.current_i_entry - ds_start
        i = 0

        while i < len(cur_groups):
            # number of entries to read from this file
            ds_end = self._get_ds_cumentries(i_ds)
            n = min(ds_end - ds_start - i_local, len(cur_groups) - i)
            cur_groups[i : i + n] = self.get_group(i_ds)

            i_ds += 1
            ds_start = ds_end
            i_local = 0
            i += n

        return cur_groups

    def __len__(self) -> int:
        """Return the total number of entries to be read."""
        if len(self.entry_map) == 0:
            return 0
        if self.n_entries is None:
            return self._get_ds_cumentries(self.n_datasets - 1)
        # only check as many files as we strictly need to
        for i in range(self.n_datasets):
            if self.n_entries < self._get_ds_cumentries(i):
                return self.n_entries
        return self._get_ds_cumentries(self.n_datasets - 1)

    def __iter__(self) -> LH5Iterator:
        """Loop through entries in blocks of size buffer_len."""
        self.current_i_entry = 0
        self.next_i_entry = self.i_start
        return self

    def __next__(self) -> Table:
        """Read next buffer_len entries and return lh5_table and iterator entry."""
        n_entries = self.n_entries
        if n_entries is not None:
            n_entries = min(
                self.buffer_len, n_entries + self.i_start - self.next_i_entry
            )

        buf = self.read(self.next_i_entry, n_entries)
        if len(buf) == 0:
            self.lh5_st.close()
            raise StopIteration
        self.next_i_entry = self.current_i_entry + len(buf)
        return buf

    def __deepcopy__(self, memo):
        """Deep copy everything except lh5_st"""
        result = LH5Iterator.__new__(LH5Iterator)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "lh5_st":
                result.lh5_st = LH5Store(
                    base_path=self.lh5_st.base_path, keep_open=self.lh5_st.keep_open
                )
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def __getstate__(self):
        """Do not try to pickle lh5_st or lh5_buffer"""
        return dict(
            self.__dict__,
            lh5_st={
                "base_path": self.lh5_st.base_path,
                "keep_open": self.lh5_st.keep_open,
            },
            lh5_buffer=None,
        )

    def __setstate__(self, d):
        """Reinitialize lh5_st and lh5_buffer to avoid potential issues"""
        self.__dict__ = d
        self.lh5_st = LH5Store(**(d["lh5_st"]))
        self.lh5_st.gimme_file(self.get_file(0))

        self.lh5_buffer = Table(size=0)

        # recursively walk through friends and append field_masks
        def build_field_mask(it):
            field_mask = it.field_mask
            for fr in it.friend:
                if isinstance(field_mask, Mapping):
                    field_mask.update(fr.field_mask)
                else:
                    field_mask |= fr.field_mask
            return field_mask

        self.reset_field_mask(build_field_mask(self))

    def _select_datasets(self, i_beg, i_end):
        """Reduce list of files and groups; used by _generate_workers"""
        self._ds_offset = i_beg
        self.n_datasets = i_end - i_beg

        if i_beg > 0:
            np.subtract(
                self.ds_map,
                self.ds_map[i_beg - 1],
                out=self.ds_map,
                where=self.ds_map != np.iinfo("q").max,
            )
            np.subtract(
                self.entry_map,
                self.entry_map[i_beg - 1],
                out=self.entry_map,
                where=self.entry_map != np.iinfo("q").max,
            )
        self.ds_map = self.ds_map[i_beg:i_end]
        self.entry_map = self.entry_map[i_beg:i_end]

        if self.local_entry_list is not None:
            self.local_entry_list = self.local_entry_list[i_beg:i_end]
        self.global_entry_list = None

        for fr in self.friend:
            fr._select_datasets(i_beg, i_end)

    def _generate_workers(self, n_workers: int):
        """Create `n_workers` copies of this iterator, dividing the datasets (file/group pairs)
        between them. These are intended for parallel use."""
        i_datasets = np.linspace(0, self.n_datasets, n_workers + 1).astype("int")
        # if we have an entry list, get local entries for all files
        if self.local_entry_list is not None:
            for i in range(self.n_datasets):
                self.get_ds_entrylist(i)

        worker_its = []
        for i_worker in range(n_workers):
            it = deepcopy(self)
            it._select_datasets(i_datasets[i_worker], i_datasets[i_worker + 1])
            worker_its += [it]

        return worker_its

    def map(
        self,
        fun: Callable[[Table, LH5Iterator], Any],
        aggregate: Callable = None,
        init: Any = None,
        begin: Callable[[LH5Iterator], None] = None,
        terminate: Callable[[LH5Iterator], None] = None,
        processes: int = None,
        executor: Executor = None,
        executor_mode: Literal["process", "thread", None] = None,
        progress_queue: Queue = None,
        job_id: int | Collection[int] = 0,
    ) -> Iterator[Any]:
        """Map function over iterator blocks.

        Returns order-preserving list of outputs. Can be multi-threaded
        provided there are no attempts to modify existing objects.
        Multi-threading splits the iterator into multiple independent
        streams with an approximately equal number of files/groups,
        concurrently processed under a single program multiple data
        model. Results will be returned asynchronously for each process.

        Note: see :meth:`query` and :meth:`hist`

        Example
        -------
        Process a table and sum the products at the end::

            def process(lh5_tab, lh5_it):
                ...process the table
                return result_of_processing

            results = lh5_it.map(process, processes=4)
            # results are an iterator over lists
            result = sum(val for result in results for val in result)

        Process a table as above, using aggregate to sum the results::

            def process(lh5_tab, lh5_it):
                ...process the table
                return result_of_processing

            result = lh5_it.map(process, aggregate=np.add, init=0, processes=4)

        Process a table using a more arbitrary output::

            class Result:
                def __init__(self):
                    ...initialize

                @classmethod
                def process_table(tab):
                    ...process the table

                def aggregate(self, result):
                    ...add data from processing into object

            result = lh5_it.map(
                Result.process_table,
                aggregate=Result.aggregate,
                init=Result(),
                processes=4
            )

        Parameters
        ----------
        fun:
            function with signature ``fun(lh5_obj: Table, it: LH5Iterator) -> Any``
            Outputs of function will be collected in list and returned
        aggregate:
            function used to iterably combine outputs of ``fun`` for each block
            of data. Should have two inputs; first input should be the type of
            the aggregate, and second of the type returned by ``fun``. This function
            can either return the result, or perform the aggregation in-place on
            the first element and return ``None``. If using multi-processing, ``map``
            will return an async-iterator over the aggregated results from each process.
            If ``None``, do not aggregate and instead return will iterate over
            result for each block.
        init:
            initial value used for aggregation. If using an aggregating function
            and ``init`` is ``None``, perform a deep copy of the first element
        begin:
            function with signature ``fun(it: LH5Iterator)`` that is run before we
            loop through a chunk of the iterator
        terminate:
            function with the signature ``fun(it: LH5Iterator)`` that is run after
            we finish looping through a chunk of the iterator
        processes:
            number of processes to use if executor is provided. If ``None``, use
            all available processes/threads.
        executor:
            :class:`concurrent.futures.Executor` object for managing parallelism.
        executor_mode:
            mode for transferring data between threads/processes, based on executor. This
            affects how aggregators, and internal states if objects are passed. Options:

            - process: multiprocessing-like; the executor is assumed to handle inter-process
              communication (likely through pickling), and objects are assumed to be isolated
            - thread: threading-like; memory is shared between threads, so we explicitly
              copy data before sending to threads to ensure isolation
            - ``None``: default; use process for ProcessPoolExecutor and InterpreterPoolExecutor,
              and thread for ThreadPoolExecutor; must be explicit for others!

        progress_queue:
            :class:`multiprocessing.Queue` object to which progress information will be
            communicated back to main process. Returns a mapping with keys:
            - task_id: the job_id passed to this function
            - total: total number of datasets to be processed
            - completed: number of datasets that have been processed
            - entries: number of entries that have been processed
            - status: "Initializing", "Processing", "Terminating" or "Finished"
        job_id:
            index of first process (see ``task_id`` above; subsequent processes will
            increment by 1) or list of ``task_ids`` for each process
        """

        # if no aggregate is provided, append results to a list
        if aggregate is None:
            init = []
            aggregate = _append_copy

        if executor is None:
            return _map_helper(
                fun,
                aggregate,
                init,
                begin,
                terminate,
                self,
                job_id,
                progress_queue=progress_queue,
            )

        if processes is None:
            if hasattr(executor, "_max_workers"):
                processes = executor._max_workers
            elif hasattr(executor, "_threads"):
                processes = len(executor._threads)
            else:
                msg = f"Must explicitly provide number of processes for {type(executor).__name__}"
                raise ValueError(msg)

        it_pool = self._generate_workers(processes)

        if executor_mode is None:
            if type(executor).__name__ in (
                "ProcessPoolExecutor",
                "InterpreterPoolExecutor",
            ):
                executor_mode = "process"
            elif type(executor).__name__ == "ThreadPoolExecutor":
                executor_mode = "thread"
            else:
                msg = f"Could not deduce executor_mode for {type(executor).__name__}. Please specify"
                raise ValueError(msg)

        if executor_mode == "process":
            result = executor.map(
                partial(
                    _map_helper,
                    fun,
                    aggregate,
                    init,
                    begin,
                    terminate,
                    progress_queue=progress_queue,
                ),
                it_pool,
                job_id
                if isinstance(job_id, Collection)
                else range(job_id, job_id + processes),
            )
        elif executor_mode == "thread":
            result = executor.map(
                _map_helper,
                [deepcopy(fun) for _ in range(processes)],
                [deepcopy(aggregate) for _ in range(processes)],
                [deepcopy(init) for _ in range(processes)],
                [deepcopy(begin) for _ in range(processes)],
                [deepcopy(terminate) for _ in range(processes)],
                it_pool,
                job_id
                if isinstance(job_id, Collection)
                else range(job_id, job_id + processes),
            )

        # If no aggregator was given, chain iterators
        if aggregate is _append_copy:
            return chain.from_iterable(result)
        return result

    def query(
        self,
        where: Callable | str,
        *,
        fields: Collection[str] | Mapping[str, str | None] = None,
        processes: Executor | int = None,
        executor: Executor = None,
        executor_mode: Literal["process", "thread", None] = None,
        library: str = None,
        progress: progress.Progress | console.Console | bool = True,
    ):
        """
        Query the data files in the iterator

        Returns the selected data as a single table in one of several formats.

        .. danger::

            This function uses :func:`eval` to evaluate string expressions. Do not
            use with untrusted input, as this can lead to arbitrary code execution.

        Examples
        --------
        Query data using a string selection::

            tab = lh5_it.query("(col1 == 0) & (col2 > 100)")

        Query data using a function::

            def select(lh5_tab, lh5_it):
                ...process data and produce a new table
                return result

            tab = lh5_it.query(select)

        Parameters
        ----------
        where:
            A filter function for selecting data entries. Can be:

            - A function that returns reduced data, with signature
              ``fun(lh5_obj: Table, it: LH5Iterator)``. Can return:

              - :class:`numpy.ndarray`: if 1D list of values; if 2D list of lists of
                values in same order as axes
              - ``Collection[ArrayLike]``: return list of values in same order as axes
              - ``Mapping[str, ArrayLike]``: mapping from axis name to values
              - ``pandas.DataFrame``: pandas dataframe. Treat as mapping from column
                name to values

            - A string expression. This will call ``eval``, with the table columns
              provided as local variables formatted as :meth:`awkward.Array`, and
              access to :mod:`awkward` (or ``ak``) and :mod:`numpy` (or ``np``).
              Return a table formatted according to ``library``

        fields:
            list of fields to return. If ``None`` return all fields in ``field_mask``.
            If a mapping is provided, key corresponds to name of field in this iterator,
            and value is an alias to name in returned table; if alias is ``None``, do
            not rename.

        processes:
            number of processes. If ``None``, use number equal to threads available
            to ``executor`` (if provided), or else do not parallelize
        executor:
            :class:`concurrent.futures.Executor` object for managing parallelism.
            If ``None``, create a :class:`concurrent.futures.ProcessPoolExecutor`
            with number of processes equal to ``processes``.
        executor_mode:
            mode for transferring data between threads/processes, based on executor. This
            affects how aggregators, and internal states if objects are passed. Options:

            - process: multiprocessing-like; the executor is assumed to handle inter-process
              communication (likely through pickling), and objects are assumed to be isolated
            - thread: threading-like; memory is shared between threads, so we explicitly
              copy data before sending to threads to ensure isolation
            - ``None``: default; use process for ProcessPoolExecutor and InterpreterPoolExecutor,
              and thread for ThreadPoolExecutor; must be explicit for others!

        library:
            library to convert the columns to when using a string expression for ``where``.
            See :meth:`Table.eval`.
        progress:
            if ``True`` draw progress bar; can also provide an existing rich ``Progress``
            or ``Console`` object
        """
        if where is None:
            where = _identity

        if isinstance(where, str):
            where = _table_query(where, library, fields)

        test = where(self.lh5_buffer, self)

        with ExitStack() as stack:
            if processes is None and isinstance(executor, Executor):
                processes = executor._max_workers

            if executor is None and isinstance(processes, int):
                executor = stack.enter_context(ProcessPoolExecutor(processes))

            prog = (
                stack.enter_context(MapProgress(processes, executor, progress))
                if progress
                else None
            )

            pq = prog.queue if prog else None
            if isinstance(test, LGDOCollection):
                it = self.map(
                    where,
                    processes=processes,
                    executor=executor,
                    executor_mode=executor_mode,
                    aggregate=Table.append,
                    progress_queue=pq,
                )
                if isinstance(it, LGDOCollection):
                    return it
                ret = deepcopy(next(it))
                for res in it:
                    ret.append(res)
                return ret
            if isinstance(test, pd.DataFrame):
                return pd.concat(
                    list(
                        self.map(
                            where,
                            processes=processes,
                            executor=executor,
                            progress_queue=pq,
                        )
                    ),
                    ignore_index=True,
                )
            if isinstance(test, np.ndarray):
                return np.concatenate(
                    list(
                        self.map(
                            where,
                            processes=processes,
                            executor=executor,
                            progress_queue=pq,
                        )
                    )
                )
            if isinstance(test, ak.Array):
                return ak.concatenate(
                    list(
                        self.map(
                            where,
                            processes=processes,
                            executor=executor,
                            progress_queue=pq,
                        )
                    )
                )

        msg = f"Cannot call query with return type {test.__class__}. "
        "Allowed return types: LGDOCollection, np.array, pd.DataFrame, ak.Array"
        raise ValueError(msg)

    def hist(
        self,
        ax: Hist | axis | Collection[axis],
        where: Callable | str = None,
        keys: Collection[str] | str = None,
        processes: Executor | int = None,
        executor: Executor = None,
        executor_mode: Literal["process", "thread", None] = None,
        progress: progress.Progress | console.Console | bool = True,
        **hist_kwargs,
    ) -> Hist:
        """
        Fill a histogram from data produced by a query selecting on ``where``. If
        ``where`` is ``None``, fill with all data fetched by iterator.

        .. danger::

            This function uses :func:`eval` to evaluate string expressions. Do not
            use with untrusted input, as this can lead to arbitrary code execution.

        Examples
        --------
        Build a 1D histogram of values in ``col3`` with a string selection::

            h = lh5_it.hist(
                hist.axis.Regular(100, 0, 500, label="col3"),
                where="(col1 == 0) & (col2 > 100)",
                keys="col3",
            )

        Build a 2D histogram with a value axis and string-category axis after
        applying some processing::

            def get_val(lh5_tab, lh5_it):
                ...process data
                return value, category

            h = lh5_it.hist(
                [hist.axis.Regular(100, 0, 500, label="Value"),
                 hist.axis.StrCategory([], growth=True, label="Category")],
                where=get_val,
            )

        Parameters
        ----------
        ax:
            :class:`hist.axis` object(s) used to construct the histogram. Can provide a
            :class:`hist.Hist` which will be filled as well.
        where:
            A filter function for selecting data entries to put into the histogram. Can be:

            - A function that returns reduced data, with signature
              ``fun(lh5_obj: Table, it: LH5Iterator)``. Can return:

              - :class:`numpy.ndarray`: if 1D list of values; if 2D list of lists of
                values in same order as axes
              - ``Collection[ArrayLike]``: return list of values in same order as axes
              - ``Mapping[str, ArrayLike]``: mapping from axis name to values
              - :class:`pandas.DataFrame`: treat as mapping from column name to values

            - A string expression. This will call ``eval``, with the table columns
              provided as local variables formatted as :meth:`awkward.Array`, and
              access to :mod:`awkward` (or ``ak``) and :mod:`numpy` (or ``np``).

        keys:
            list of keys fields corresponding to axes. Use if where
            returns a mapping with names different from axis names.
        processes:
            number of processes. If ``None``, use number equal to threads available
            to ``executor`` (if provided), or else do not parallelize
        executor:
            :class:`concurrent.futures.Executor` object for managing parallelism.
            If ``None``, create a :class:`concurrent.futures.ProcessPoolExecutor`
            with number of processes equal to ``processes``.
        executor_mode:
            mode for transferring data between threads/processes, based on executor. This
            affects how aggregators, and internal states if objects are passed. Options:

            - process: multiprocessing-like; the executor is assumed to handle inter-process
              communication (likely through pickling), and objects are assumed to be isolated
            - thread: threading-like; memory is shared between threads, so we explicitly
              copy data before sending to threads to ensure isolation
            - ``None``: default; use process for ProcessPoolExecutor and InterpreterPoolExecutor,
              and thread for ThreadPoolExecutor; must be explicit for others!

        progress:
            if ``True`` draw progress bar; can also provide an existing rich ``Progress``
            or ``Console`` object
        hist_kwargs:
            additional keyword arguments for constructing :class:`hist.Hist`.
        """

        # get initial hist for each thread
        if isinstance(ax, axis.AxesMixin):
            h = Hist(ax, **hist_kwargs)
        elif isinstance(ax, Collection):
            h = Hist(*ax, **hist_kwargs)
        elif isinstance(ax, Hist):
            h = ax.copy()
            h[...] = 0

        if where is None:
            where = _identity
        elif isinstance(where, str):
            where = _table_query(where, "ak", None)

        with ExitStack() as stack:
            if executor is None and isinstance(processes, int):
                executor = stack.enter_context(ProcessPoolExecutor(processes))

            prog = (
                stack.enter_context(MapProgress(processes, executor, progress))
                if progress
                else None
            )

            if processes is None and isinstance(executor, Executor):
                processes = executor._max_workers

            h = self.map(
                where,
                processes=processes,
                executor=executor,
                executor_mode=executor_mode,
                aggregate=_hist_filler(keys),
                init=h,
                progress_queue=prog.queue if prog else None,
            )
            if isinstance(h, Iterator):
                h = sum(h)

        if isinstance(ax, Hist):
            ax += h
            return ax
        return h


# Would that python multiprocessing allowed lambdas...
def _identity(val, _):
    return val


def _append_copy(list, val):
    """Helper for aggregating tables in query"""
    list.append(deepcopy(val))


def _map_helper(
    fun,
    aggregator,
    init,
    begin,
    terminate,
    it,
    i_job: int,
    *,
    progress_queue: Queue = None,
):
    """Helper for executing init, begin and terminate functions when calling map"""
    if progress_queue is not None:
        progress_queue.put(
            {
                "task_id": i_job,
                "total": it.n_datasets,
                "completed": 0.0,
                "entries": 0,
                "status": "Initializing",
                "finished": False,
            }
        )

    if begin:
        begin(it)

    aggregate = init
    for tab in it:
        if progress_queue is not None:
            i_ds = np.searchsorted(it.entry_map, it.current_i_entry, "right")
            progress_queue.put(
                {
                    "task_id": i_job,
                    "total": it.n_datasets,
                    "completed": i_ds
                    + (it.current_i_entry - it._get_ds_cumentries(i_ds - 1))
                    / (it._get_ds_cumentries(i_ds) - it._get_ds_cumentries(i_ds - 1)),
                    "entries": it.current_i_entry,
                    "status": "Processing",
                    "finished": False,
                }
            )

        result = fun(tab, it)

        if aggregate is None:
            # if no init, initialize on first entry
            aggregate = deepcopy(result)
        else:
            res = aggregator(aggregate, result)
            if res is not None:
                aggregate = res

    if progress_queue is not None:
        progress_queue.put(
            {
                "task_id": i_job,
                "total": it.n_datasets,
                "completed": it.n_datasets,
                "entries": it._get_ds_cumentries(it.n_datasets),
                "status": "Terminating",
                "finished": False,
            }
        )

    if terminate:
        terminate(it)

    if progress_queue is not None:
        progress_queue.put(
            {
                "task_id": i_job,
                "total": it.n_datasets,
                "completed": it.n_datasets,
                "entries": it._get_ds_cumentries(it.n_datasets),
                "status": "Finished",
                "finished": True,
            }
        )

    return aggregate


@dataclass
class _table_query:
    """Helper for when query is called on a string"""

    expr: str
    library: str
    fields: Collection[str] | Mapping[str, str | None] | None

    def __post_init__(self):
        # turn collection into mapping
        if self.fields is not None and not isinstance(self.fields, Mapping):
            self.fields = dict.fromkeys(self.fields)

    def __call__(self, tab, _):
        """Evaluate selection and return selected elements"""
        args = {f: a.view_as("ak", with_units=False) for f, a in tab.items()}
        if self.fields is not None:
            for k, v in self.fields.items():
                if v is not None:
                    args[v] = tab[k].view_as("ak", with_units=False)

        if self.expr:
            mask = eval(
                self.expr,
                {"np": np, "numpy": np, "ak": ak, "awkward": ak},
                args,
            )
            ret = tab[mask]
        else:
            ret = tab

        if self.fields is not None:
            ret = Table(
                {(k if f is None else f): ret[k] for k, f in self.fields.items()}
            )

        if self.library is None:
            return ret
        return ret.view_as(self.library, with_units=False)


class _hist_filler:
    """Helper for filling histogram"""

    def __init__(self, keys):
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            elif not isinstance(keys, list):
                keys = list(keys)
        self.keys = keys

    def __call__(self, hist, data):
        if isinstance(data, np.ndarray) and len(data.shape) == 1:
            hist.fill(data)
        elif isinstance(data, np.ndarray) and len(data.shape) == 2:
            hist.fill(*data)
        elif isinstance(data, pd.DataFrame):
            if self.keys is not None:
                hist.fill(*[data.eval(k) for k in self.keys])
            else:
                hist.fill(**data)
        elif isinstance(data, Mapping):
            if self.keys is not None:
                hist.fill(
                    *[
                        eval(
                            k,
                            {
                                "np": np,
                                "numpy": np,
                                "pd": pd,
                                "pandas": pd,
                                "ak": ak,
                                "awkward": ak,
                            },
                            data,
                        )
                        for k in self.keys
                    ]
                )
            else:
                hist.fill(**data)
        elif isinstance(data, ak.Array):
            if self.keys is not None:
                hist.fill(
                    *[
                        ak.ravel(
                            eval(
                                k,
                                {"np": np, "numpy": np, "ak": ak, "awkward": ak},
                                {f: data[f] for f in data.fields},
                            )
                        )
                        for k in self.keys
                    ]
                )
            else:
                hist.fill(*[ak.ravel(data[f]) for f in data.fields])
        elif isinstance(data, Collection):
            hist.fill(*data)
        else:
            msg = "data returned by where is not compatible with hist. Must be a 1d or 2d numpy array, a list of arrays, or a mapping from str to array"
            raise ValueError(msg)


class MapProgress(Thread):
    """Helper for tracking progress of threads in map

    Basic Usage::

        with MapProgress(task_list) as prog:
            iter.map(..., progress_queue = prog.queue)
    """

    def __init__(
        self,
        tasks: list | int,
        executor: Executor,
        prog: progress.Progress | console.Console = None,
        update_period: float = 0.1,
    ):
        """
        Parameters
        ----------
        tasks
            list of descriptions to prepend to progress bars. Can also provide the number of
            tasks, in which case description will be set to ``"#i"``.
        prog
            rich Progress or Console object to add bars to. Use to customize the bar.
        update_period
            frequency in seconds to update progress bars.
        """
        if isinstance(prog, progress.Progress):
            self.progress = prog
        else:
            self.progress = progress.Progress(
                progress.TextColumn("{task.description:>5}: {task.fields[status]:<12}"),
                progress.BarColumn(),
                progress.TaskProgressColumn(),
                progress.TextColumn(
                    "{task.completed:.1f}/{task.total} ds, {task.fields[entries]} rows"
                ),
                progress.TimeElapsedColumn(),
                progress.TimeRemainingColumn(compact=True),
                console=prog if isinstance(prog, console.Console) else None,
                auto_refresh=False,
            )

        if isinstance(tasks, int):
            tasks = [f"#{i}" for i in range(tasks)]
        elif isinstance(tasks, str):
            tasks = [tasks]
        elif tasks is None:
            tasks = [":"]
        for desc in tasks:
            self.progress.add_task(
                desc, total=None, completed=0.0, finished=False, status="", entries=0
            )
        self.update_period = update_period

        self.manager = None
        self.queue = None
        if executor is None:
            self.queue = Queue()
        elif type(executor).__name__ == "ProcessPoolExecutor":
            import multiprocessing  # noqa: PLC0415

            self.manager = multiprocessing.Manager()
            self.queue = self.manager.Queue()
        elif type(executor).__name__ == "InterpreterPoolExecutor":
            self.manager = None
            from concurrent import interpreters  # noqa: PLC0415

            self.queue = interpreters.create_queue()
        elif type(executor).__name__ == "ThreadPoolExecutor":
            self.queue = Queue()
        else:
            log.warning(
                f"Cannot pass messages from {type(executor).__name__} to progress bar. Progress will not be shown."
            )
        self.done = Event()
        super().__init__(daemon=True)

    def run(self):
        self.progress.start()

        # update every update_period s
        while not self.done.wait(self.update_period):
            while True:
                try:
                    progress_info = self.queue.get(block=False)
                except (AttributeError, Empty):
                    break
                self.progress.update(**progress_info)
            self.progress.refresh()

        # one final update to make sure we get all the way to 100%
        while True:
            try:
                progress_info = self.queue.get(block=False)
            except (AttributeError, Empty):
                break
            self.progress.update(**progress_info)
        self.progress.refresh()
        self.progress.stop()

    def __enter__(self) -> MapProgress:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.done.set()
        self.join()
