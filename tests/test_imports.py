"""Verify the package and its subpackages import cleanly and that every name
listed in ``__all__`` is actually exposed.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import lh5

SUBMODULES = [
    name
    for _, name, _ in pkgutil.walk_packages(lh5.__path__, prefix=lh5.__name__ + ".")
]


@pytest.mark.parametrize("modname", [lh5.__name__, *SUBMODULES])
def test_module_importable(modname):
    importlib.import_module(modname)


@pytest.mark.parametrize("modname", [lh5.__name__, *SUBMODULES])
def test_all_names_resolved(modname):
    mod = importlib.import_module(modname)
    for name in getattr(mod, "__all__", []):
        assert hasattr(mod, name), f"{modname}.__all__ lists missing name {name!r}"


def test_top_level_star_import():
    namespace: dict = {}
    exec("from lh5 import *", namespace)
    for name in lh5.__all__:
        assert name in namespace
