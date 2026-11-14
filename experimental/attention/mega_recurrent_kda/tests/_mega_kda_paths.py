"""Load this checkout's KDA operators and shared precision helpers only."""

import hashlib
import importlib
import importlib.util
import os
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys

ATTENTION_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ATTENTION_ROOT / "mega_recurrent_kda/tests"


def _package(name, root):
    if name not in sys.modules:
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(root)]
        sys.modules[name] = importlib.util.module_from_spec(spec)


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    # Reuse a public import of this exact file, avoiding duplicate Torch schema
    # registration. A module from another checkout is never accepted here.
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if isinstance(filename, str) and Path(filename).resolve() == path.resolve():
            sys.modules[name] = module
            return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_sample(relative_path):
    relative = Path(relative_path)
    if (
        len(relative.parts) != 2
        or relative.parts[0] != "mega_recurrent_kda"
        or relative.name != relative.parts[0] + ".py"
    ):
        raise ValueError(f"Unsupported KDA source: {relative_path!r}")
    arch = os.environ.get("KDA_ARCH", "arch35")
    if arch not in ("arch35", "arch92"):
        raise ValueError("KDA_ARCH must be arch35 or arch92")
    root = ATTENTION_ROOT / relative.parts[0] / "op_dsl" / arch
    path = root / relative.name
    if not path.is_file():
        raise FileNotFoundError(path)
    prefix = "_ops_kda_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    _package(prefix, root)
    return _load(prefix + "." + relative.stem, path)


def load_test_support():
    prefix = (
        "_ops_kda_tests_" + hashlib.sha256(str(TEST_ROOT).encode()).hexdigest()[:16]
    )
    _package(prefix, TEST_ROOT)
    runtime = importlib.import_module(prefix + ".kda_ttk_ops")
    inputs = importlib.import_module(prefix + ".assets.inputs")
    return runtime, inputs
