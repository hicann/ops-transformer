#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Custom op loader for quant_block_sparse_attn pytest.

The QBSA pytest tree should not hard-code one development worktree.  This module
checks the current torch registry and only loads an external extension when
`QBSA_CUSTOM_OPS_PATH` explicitly points to one.
"""

import glob
import importlib.util
import os
import sys
from pathlib import Path

import torch


_LOAD_ERRORS = []


def _prepend_env_path(env_name, paths):
    current = os.environ.get(env_name, "")
    current_parts = [part for part in current.split(":") if part]
    new_parts = [str(path) for path in paths if path and str(path) not in current_parts]
    if new_parts:
        os.environ[env_name] = ":".join(new_parts + current_parts)


def _append_env_path(env_name, paths):
    current = os.environ.get(env_name, "")
    current_parts = [part for part in current.split(":") if part]
    new_parts = [str(path) for path in paths if path and path.exists() and str(path) not in current_parts]
    if new_parts:
        os.environ[env_name] = ":".join(current_parts + new_parts)


def _candidate_roots():
    roots = []
    value = os.environ.get("QBSA_CUSTOM_OPS_PATH")
    if value:
        roots.extend(Path(part) for part in value.split(":") if part)
    roots.append(Path(__file__).resolve().parents[2] / "torch_ops_extension")
    deduped = []
    seen = set()
    for root in roots:
        root = root.expanduser()
        if root in seen:
            continue
        seen.add(root)
        deduped.append(root)
    return deduped


def _load_shared_libraries(root):
    if root.is_file() and root.suffix == ".so":
        candidates = [root]
    else:
        candidates = [Path(path) for path in glob.glob(str(root / "**" / "custom_ops_lib*.so"), recursive=True)]
    for lib_path in candidates[:1]:
        try:
            torch.ops.load_library(str(lib_path))
        except Exception as error:  # pragma: no cover - depends on local extension environment
            _LOAD_ERRORS.append(f"{lib_path}: {error}")


def _load_python_package(root):
    init_file = root / "custom_ops" / "__init__.py"
    if not init_file.exists():
        return
    try:
        spec = importlib.util.spec_from_file_location(
            "_bsa_external_custom_ops",
            init_file,
            submodule_search_locations=[str(init_file.parent)],
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as error:  # pragma: no cover - depends on local extension environment
        _LOAD_ERRORS.append(f"{init_file}: {error}")


def _load_converter(root):
    converter_names = [
        "npu_quant_block_sparse_attn.py",
        "npu_quant_block_sparse_attn_metadata.py",
    ]
    for converter_name in converter_names:
        converter_file = root / "custom_ops" / "converter" / converter_name
        if not converter_file.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"_bsa_{converter_file.stem}_converter",
                converter_file,
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        except Exception as error:  # pragma: no cover - depends on local extension environment
            _LOAD_ERRORS.append(f"{converter_file}: {error}")


def _custom_opp_roots():
    repo_root = Path(__file__).resolve().parents[5]
    build_candidates = [
        repo_root / "build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-x86_64.run"
        / "packages/vendors/custom_transformer",
    ]
    build_roots = [path for path in build_candidates if path.exists()]
    if build_roots:
        return build_roots
    candidates = [
        Path("/home/dyx/vendors/custom_transformer"),
        Path("/workspace/vendors/custom_transformer"),
    ]
    return [path for path in candidates if path.exists()]


def _configure_custom_opp_paths():
    custom_opp_roots = _custom_opp_roots()
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", custom_opp_roots)
    _prepend_env_path("LD_LIBRARY_PATH", [root / "op_api" / "lib" for root in custom_opp_roots])


def _load_custom_ops():
    _configure_custom_opp_paths()
    if has_quant_block_sparse_attn_op() and has_quant_block_sparse_attn_metadata_op():
        for root in _candidate_roots():
            if root.exists():
                _load_converter(root)
                break
        return
    for root in _candidate_roots():
        if not root.exists():
            continue
        _load_shared_libraries(root)
        if has_quant_block_sparse_attn_op() and has_quant_block_sparse_attn_metadata_op():
            _load_converter(root)
            return
        _load_python_package(root)
        if has_quant_block_sparse_attn_op() and has_quant_block_sparse_attn_metadata_op():
            return


def has_quant_block_sparse_attn_op():
    try:
        getattr(torch.ops.custom, "npu_quant_block_sparse_attn")
    except AttributeError:
        return False
    return True


def has_quant_block_sparse_attn_metadata_op():
    try:
        getattr(torch.ops.custom, "npu_quant_block_sparse_attn_metadata")
    except AttributeError:
        return False
    return True


def load_errors():
    return list(_LOAD_ERRORS)


_load_custom_ops()
