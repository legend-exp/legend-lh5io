from __future__ import annotations

import os
from typing import Any

_SIZE_SUFFIXES = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
}


def parse_datasize(size: str | int) -> int:
    """Parse a data size such as ``"16MiB"``, ``"4096"`` or ``16`` into bytes."""
    if isinstance(size, int):
        return size
    s = str(size).strip().lower()
    for suffix, mult in sorted(_SIZE_SUFFIXES.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(float(s))


DEFAULT_PAGE_BUFFER: int = parse_datasize(os.getenv("LH5_PAGE_BUFFER", "0"))
"""Default page-buffer size (bytes) for reading paged LH5 files.

Initialised from the ``LH5_PAGE_BUFFER`` environment variable (binary and
decimal suffixes accepted, e.g. ``16MiB``), so page-buffered reads can be
enabled fleet-wide without touching call sites. ``0`` disables page
buffering. Only files written with the paged file-space strategy can use a
page buffer; plain files fall back to an unbuffered open.
"""


def default_hdf5_settings() -> dict[str, Any]:
    """Returns the HDF5 settings for writing data to disk reset to the package defaults.

    Examples
    --------
    >>> import lh5
    >>> lh5.DEFAULT_HDF5_SETTINGS["compression"] = "lzf"
    >>> lh5.write(data, "data", "file.lh5")  # compressed with LZF
    >>> lh5.DEFAULT_HDF5_SETTINGS = lh5.default_hdf5_settings()
    >>> lh5.write(data, "data", "file.lh5", "of")  # compressed with default settings (GZIP)
    """

    return {
        "shuffle": True,
        "compression": "gzip",
        # target chunk size in bytes; translated per dataset into an h5py
        # "chunks" tuple when the user provides no explicit chunking. Without
        # it, h5py auto-chunking freezes the chunk shape from the first
        # (possibly tiny) buffered write.
        "chunk_nbytes": 256 * 1024,
    }


DEFAULT_HDF5_SETTINGS: dict[str, ...] = default_hdf5_settings()
"""Global dictionary storing the default HDF5 settings for writing data to disk.

Modify this global variable before writing data to disk with this package.

Examples
--------
>>> import lh5
>>> lh5.DEFAULT_HDF5_SETTINGS["compression"] = "lzf"
>>> lh5.write(data, "data", "file.lh5")  # compressed with LZF
"""
