from __future__ import annotations

# import this so users can transparently decode data compressed with hdf5plugin
# filters
import hdf5plugin  # noqa: F401

from ._version import version as __version__
from .io import LH5Iterator, LH5Store, ls, read, read_as, read_n_rows, show, write

__all__ = [
    "LH5Iterator",
    "LH5Store",
    "__version__",
    "ls",
    "read",
    "read_as",
    "read_n_rows",
    "show",
    "write",
]
