"""Routines for reading and writing LEGEND Data Objects in HDF5 files.

Currently the primary on-disk format for LGDO objects is LEGEND HDF5 (LH5) files. IO
is done via the class :class:`.store.LH5Store`. LH5 files can also be
browsed easily in Python like any `HDF5 <https://www.hdfgroup.org>`_ file using
`h5py <https://www.h5py.org>`_.
"""

from __future__ import annotations

# import this so users can transparently decode data compressed with hdf5plugin
# filters
import hdf5plugin  # noqa: F401

from . import concat, truncate
from .core import read, read_as, write
from .iterator import LH5Iterator, MapProgress
from .settings import default_hdf5_settings
from .store import LH5Store
from .tools import ls, show
from .utils import read_n_rows

__all__ = [
    "LH5Iterator",
    "LH5Store",
    "MapProgress",
    "concat",
    "default_hdf5_settings",
    "ls",
    "read",
    "read_as",
    "read_n_rows",
    "show",
    "truncate",
    "write",
]
