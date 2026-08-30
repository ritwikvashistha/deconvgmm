"""The convMMD facade re-exports exactly the private implementation objects.

``src/deconvgmm/convmmd.py`` imports from ``deconvgmm._impl.convmmd`` /
``convmmd_fit``, which the wheel maps from ``development/``. This test simulates
that mapping (no built wheel needed) and asserts every ``__all__`` name is a
faithful re-export of the same object. It also checks that convMMD is exposed on
the public ``deconvgmm`` top-level surface.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from development import convmmd as impl_convmmd
from development import convmmd_fit as impl_convmmd_fit
from development import convmmd_grouped as impl_convmmd_grouped
from development import general_validation as impl_general_validation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACADE_PATH = PROJECT_ROOT / "src" / "deconvgmm" / "convmmd.py"
TOP_LEVEL_INIT = PROJECT_ROOT / "src" / "deconvgmm" / "__init__.py"


def _load_facade():
    """Load the facade with ``deconvgmm._impl`` aliased to ``development``."""

    modules = {}
    for name, module in (
        ("deconvgmm", None),
        ("deconvgmm._impl", None),
        ("deconvgmm._impl.convmmd", impl_convmmd),
        ("deconvgmm._impl.convmmd_fit", impl_convmmd_fit),
        ("deconvgmm._impl.convmmd_grouped", impl_convmmd_grouped),
        ("deconvgmm._impl.general_validation", impl_general_validation),
    ):
        if module is None:
            module = types.ModuleType(name)
        modules[name] = sys.modules.get(name)
        sys.modules[name] = module
    try:
        spec = importlib.util.spec_from_file_location(
            "deconvgmm.convmmd", FACADE_PATH
        )
        facade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(facade)
        return facade
    finally:
        for name, previous in modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_facade_reexports_exact_implementation_objects():
    facade = _load_facade()
    assert facade.__all__ == sorted(facade.__all__), "keep __all__ sorted"
    impl_modules = (
        impl_convmmd,
        impl_convmmd_fit,
        impl_convmmd_grouped,
        impl_general_validation,
    )
    for name in facade.__all__:
        value = getattr(facade, name)
        source = None
        for module in impl_modules:
            source = getattr(module, name, None)
            if source is not None:
                break
        assert source is not None, f"{name} is not defined in the impl modules"
        assert value is source, f"{name} is not a faithful re-export"


def test_facade_all_matches_public_names():
    facade = _load_facade()
    public = {name for name in vars(facade) if not name.startswith("_")}
    # Drop re-imported module objects (there are none expected) and keep symbols.
    public = {name for name in public if not isinstance(getattr(facade, name), type(sys))}
    assert public == set(facade.__all__)


def test_convmmd_is_a_public_top_level_module():
    """convMMD is now a first-class public module in the ``deconvgmm`` surface."""

    tree = ast.parse(TOP_LEVEL_INIT.read_text(encoding="utf-8"))
    top_level_all = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "__all__" for target in node.targets
        ):
            top_level_all = [element.value for element in node.value.elts]
    assert top_level_all is not None
    assert "convmmd" in top_level_all
