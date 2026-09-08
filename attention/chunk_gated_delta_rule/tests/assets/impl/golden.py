#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK golden adapter for ChunkGatedDeltaRule.

The CPU reference is loaded from tests/pytest/chunk_gated_delta_rule_golden.py.
The benchmark (third-party) is loaded from tests/pytest/chunk_gated_delta_rule_benchmark.py.

Both golden and benchmark are computed here. Golden outputs are returned to the
framework for comparison; benchmark outputs are stored in _GOLDEN_CONTEXT for
the custom compare to retrieve and perform three-party cross_check.
"""

import gc
import importlib.util
import logging
import re
import sys
import numpy
import torch
import torch.nn.functional as F
from pathlib import Path


logger = logging.getLogger(__name__)


PYTEST_GOLDEN_MODULE = None
PYTEST_BENCHMARK_MODULE = None

_GOLDEN_CONTEXT = {}

_BENCH_DIR_SUFFIX = ".bench"
_BENCH_FILE_COUNT = 2
_BENCH_FILE_PATTERN = re.compile(
    r"^bench_([0-9]+)_([A-Za-z0-9][A-Za-z0-9_.-]*?)__shape_(scalar|[0-9]+(?:x[0-9]+)*)\.bin$"
)


def load_pytest_golden_module():
    """Load tests/pytest/chunk_gated_delta_rule_golden.py as the canonical CPU golden."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "chunk_gated_delta_rule_golden.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"cgdr_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_GOLDEN_MODULE = module
    return PYTEST_GOLDEN_MODULE


def load_pytest_benchmark_module():
    """Load tests/pytest/chunk_gated_delta_rule_benchmark.py as the third-party benchmark."""
    global PYTEST_BENCHMARK_MODULE
    if PYTEST_BENCHMARK_MODULE is not None:
        return PYTEST_BENCHMARK_MODULE
    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "chunk_gated_delta_rule_benchmark.py"
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"cgdr_pytest_benchmark_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
    PYTEST_BENCHMARK_MODULE = module
    return PYTEST_BENCHMARK_MODULE


__golden__ = {
    "e2e": {"torch_npu.npu_chunk_gated_delta_rule": "cpu_chunk_gated_delta_rule"},
    "aclnn": {"aclnnChunkGatedDeltaRule": "aclnn_chunk_gated_delta_rule_golden"},
}


_PARAM_ORDER = [
    "beta",
    "initial_state",
    "actual_seq_lengths",
    "scale",
    "g",
]


def _extract_param(args, kwargs, name, index, default=None):
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return default


def _run_cpu_golden(
    query, key, value, beta, initial_state, actual_seq_lengths, scale, g
):
    """Run the CPU golden reference and return (out, final_state).

    Receives torch tensors (CPU). The pytest golden expects:
      q/k/v/g/beta: [1, T, ...] (unsqueeze(0))
      initial_state: [B, Nv, Dk, Dv] (transposed from [B, Nv, Dv, Dk])
      cu_seqlens: cumsum of actual_seq_lengths (padded with 0 at front)
    Returns:
      out: [T, Nv, Dv]
      final_state: [B, Nv, Dv, Dk]
    """
    mod = load_pytest_golden_module()

    q = query.unsqueeze(0).to(torch.float32)
    k = key.unsqueeze(0).to(torch.float32)
    v = value.unsqueeze(0).to(torch.float32)
    beta_t = beta.unsqueeze(0).to(torch.float32)

    if g is None:
        g = torch.zeros((v.shape[1], v.shape[2]), dtype=torch.float32, device=v.device)
    g_t = g.unsqueeze(0).to(torch.float32)

    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)

    cu_seqlens = F.pad(actual_seq_lengths, (1, 0)).cumsum(dim=0).to(torch.int64)

    state_transposed = initial_state.transpose(-1, -2).clone().to(torch.float32)

    o, state = mod.chunk_gated_delta_rule_npu(
        q,
        k,
        v,
        g_t,
        beta_t,
        scale=scale,
        initial_state=state_transposed,
        cu_seqlens=cu_seqlens,
        chunk_size=64,
    )
    o_out = o[0].to(query.dtype)
    state_out = state.transpose(-1, -2).to(initial_state.dtype)
    return o_out, state_out


def _run_benchmark(
    query, key, value, beta, initial_state, actual_seq_lengths, scale, g
):
    """Run the third-party benchmark and return (out_bench, state_bench).

    The benchmark (chunk_gdn_benchmark_opt) uses a different bf16 implementation
    and serves as the third-party reference for cross_check comparison.
    """
    bench_mod = load_pytest_benchmark_module()

    q = query
    k = key
    v = value
    beta_b = beta
    state = initial_state

    if g is None:
        g = torch.zeros((v.shape[0], v.shape[1]), dtype=torch.float32, device=v.device)

    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)

    asl_list = actual_seq_lengths.tolist()
    o_bench, state_bench = bench_mod.chunk_gdn_benchmark_opt(
        q,
        k,
        v,
        beta_b,
        scale,
        state,
        asl_list,
        g=g,
        chunk_size=64,
    )
    return o_bench, state_bench


def _compute_and_store(
    query,
    key,
    value,
    beta,
    initial_state,
    actual_seq_lengths,
    scale,
    g,
    testcase_name=None,
):
    """Compute both golden and benchmark, store benchmark in _GOLDEN_CONTEXT.

    Returns (out_golden, state_golden) for the framework to compare against NPU.
    """
    o_g, state_g = _run_cpu_golden(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        scale,
        g,
    )

    gc.collect()

    o_b, state_b = _run_benchmark(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        scale,
        g,
    )

    _GOLDEN_CONTEXT["bench_out"] = o_b
    _GOLDEN_CONTEXT["bench_state"] = state_b
    _GOLDEN_CONTEXT["bench_case"] = testcase_name

    _save_bench(testcase_name, (o_b, state_b))

    return o_g, state_g


def cpu_chunk_gated_delta_rule(query, key, value, *args, **kwargs):
    """Golden reference for torch_npu.npu_chunk_gated_delta_rule.

    Receives the same parameters as the NPU API (positional or keyword).
    Returns [out, final_state] to align with NPU outputs.
    Benchmark outputs are stored in _GOLDEN_CONTEXT for three-party compare.
    """
    p = {}
    for i, name in enumerate(_PARAM_ORDER):
        p[name] = _extract_param(args, kwargs, name, i)

    o, state = _compute_and_store(
        query,
        key,
        value,
        p["beta"],
        p["initial_state"],
        p["actual_seq_lengths"],
        p["scale"],
        p["g"],
        testcase_name=kwargs.get("testcase_name"),
    )
    return [o, state]


def aclnn_chunk_gated_delta_rule_golden(
    query,
    key,
    value,
    beta,
    initialState,
    actualSeqLengths,
    gOptional,
    scaleValue,
    out,
    finalState,
    **kwargs,
):
    """Golden reference for aclnnChunkGatedDeltaRule.

    Parameter names follow aclnnChunkGatedDeltaRuleGetWorkspaceSize (without
    workspaceSize and executor). Returns [out_golden, finalState_golden].
    Benchmark outputs are stored in _GOLDEN_CONTEXT for three-party compare.
    """
    o, state = _compute_and_store(
        query,
        key,
        value,
        beta,
        initialState,
        actualSeqLengths,
        scaleValue,
        gOptional,
        testcase_name=kwargs.get("testcase_name"),
    )
    return [o, state]


def get_golden_context():
    return _GOLDEN_CONTEXT


def _bench_to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.dtype in (torch.bfloat16, torch.float16):
            value = value.to(torch.float32)
        return value.numpy()
    return numpy.asarray(value)


def _manual_bench_dirs(testcase_name, mode):
    """Sibling bench dirs (<case_dir>.bench) matching the manual-data mode."""
    try:
        from ttk.core_modules.manual_data import case_directory_name
        from ttk.utilities import get_global_storage
    except ImportError:
        return []
    switches = get_global_storage()
    if getattr(switches, "manual_data_mode", None) != mode:
        return []
    roots = tuple(getattr(switches, "manual_data_dirs", ()) or ())
    stem = case_directory_name(testcase_name) + _BENCH_DIR_SUFFIX
    return [Path(root) / stem for root in roots]


def _encode_shape(shape):
    shape = tuple(int(dimension) for dimension in shape)
    return "scalar" if not shape else "x".join(str(dimension) for dimension in shape)


def _save_bench(testcase_name, bench_outputs):
    """Persist benchmark outputs beside the manual-data case dir at prepare.

    The TTK manual-data case dir has a strict file schema (input/scalar/golden
    with slot counts fixed by the CSV), so the benchmark lives in a sibling
    directory and is written only when goldens are dumped (--dump in,golden).
    Uses the same bin convention as TTK golden files: raw bytes with the dtype
    and shape encoded in the filename (bench_<i>_<dtype>__shape_<DxN>.bin).
    """
    if testcase_name is None:
        return
    bench_dirs = _manual_bench_dirs(testcase_name, "prepare")
    if not bench_dirs:
        return
    try:
        from ttk.utilities import get_global_storage, dump_to_file
    except ImportError:
        return
    dump_config = getattr(get_global_storage(), "dump_config", None)
    if dump_config is None or not dump_config.is_golden_enabled():
        return
    bench_dir = bench_dirs[0]
    bench_dir.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(bench_outputs):
        array = _bench_to_numpy(value)
        stem = f"bench_{index}_{array.dtype.name}__shape_{_encode_shape(array.shape)}"
        dump_to_file(array, str(bench_dir), stem, file_format="bin")
    logger.info("[%s] persisted benchmark data at %s", testcase_name, bench_dir)


def _load_bench(testcase_name):
    """Load benchmark outputs persisted at prepare; None when unavailable."""
    if testcase_name is None:
        return None
    try:
        from ttk.utilities import load_numpy_data
    except ImportError:
        return None
    for bench_dir in _manual_bench_dirs(testcase_name, "replay"):
        entries = {}
        for path in sorted(bench_dir.glob("bench_*.bin")):
            match = _BENCH_FILE_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            entries[int(match.group(1))] = (match.group(2), match.group(3), path)
        indexes = sorted(entries)
        if indexes != list(range(_BENCH_FILE_COUNT)):
            continue
        try:
            values = []
            for index in indexes:
                dtype_name, shape_token, path = entries[index]
                shape = (
                    ()
                    if shape_token == "scalar"
                    else tuple(int(d) for d in shape_token.split("x"))
                )
                values.append(
                    load_numpy_data(str(path), numpy.dtype(dtype_name), shape).copy()
                )
        except Exception:
            logger.warning(
                "[%s] failed to load persisted benchmark: %s", testcase_name, path
            )
            continue
        logger.info("[%s] loaded persisted benchmark from %s", testcase_name, bench_dir)
        return values
    return None


def _flatten_values(values):
    flat = []
    for item in values or ():
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten_values(item))
        else:
            flat.append(item)
    return flat


def _scalar_to_float(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        return float(value.reshape(-1)[0])
    return float(value)


def _recompute_bench_from_context(compare_context):
    """Recompute the benchmark from inputs restored by manual-data replay.

    Both CSV layouts (e2e and aclnn) place q, k, v, beta, initial_state,
    actual_seq_lengths, g in the first seven tensor slots; scale is the first
    aclnn scalar or the e2e 'scale' attribute.
    """
    tensors = _flatten_values(getattr(compare_context, "input_tensors", None))
    scalars = _flatten_values(getattr(compare_context, "input_scalars", None))
    attributes = dict(getattr(compare_context, "attributes", None) or {})
    if len(tensors) < 7:
        raise ValueError(
            f"compare_context provides {len(tensors)} tensors, expected at least 7"
        )

    query, key, value, beta, initial_state, actual_seq_lengths, g = tensors[:7]

    scale = None
    for scalar in scalars:
        scale = _scalar_to_float(scalar)
        if scale is not None:
            break
    if scale is None:
        for name in ("scale", "scaleValue"):
            if attributes.get(name) is not None:
                scale = _scalar_to_float(attributes[name])
                break

    o_b, state_b = _run_benchmark(
        query, key, value, beta, initial_state, actual_seq_lengths, scale, g
    )
    return o_b, state_b


def resolve_bench(compare_context=None):
    """Return (bench_out, bench_state) for the three-party compare hook.

    Prefers the in-memory benchmark computed alongside the golden function for
    the current testcase (direct and input-only replay). Full-mode manual-data
    replay skips the golden function, so the benchmark persisted at prepare is
    loaded from the <case_dir>.bench sibling of the manual-data directory —
    mirroring how the golden itself is restored from bin. Recomputing from the
    restored inputs remains a fallback for datasets prepared without it.
    """
    name = getattr(compare_context, "testcase_name", None)
    if name is None or _GOLDEN_CONTEXT.get("bench_case") != name:
        if name is None:
            return _GOLDEN_CONTEXT.get("bench_out"), _GOLDEN_CONTEXT.get("bench_state")
        loaded = _load_bench(name)
        if loaded is None:
            logger.warning(
                "[%s] persisted benchmark not found, recomputing from restored inputs",
                name,
            )
            loaded = list(_recompute_bench_from_context(compare_context))
        _GOLDEN_CONTEXT["bench_out"], _GOLDEN_CONTEXT["bench_state"] = loaded
        _GOLDEN_CONTEXT["bench_case"] = name
    return _GOLDEN_CONTEXT.get("bench_out"), _GOLDEN_CONTEXT.get("bench_state")
