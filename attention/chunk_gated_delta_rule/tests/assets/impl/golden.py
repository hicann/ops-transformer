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
import sys
import torch
import torch.nn.functional as F
from pathlib import Path


PYTEST_GOLDEN_MODULE = None
PYTEST_BENCHMARK_MODULE = None

_GOLDEN_CONTEXT = {}


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
    query, key, value, beta, initial_state, actual_seq_lengths, scale, g
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
    )
    return [o, state]


def get_golden_context():
    return _GOLDEN_CONTEXT
