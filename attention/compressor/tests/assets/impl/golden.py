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

import importlib.util
import sys
import types
from pathlib import Path

import torch
import numpy as np
import gc

try:
    from ttk.utilities.container_utils import get_global_storage
except Exception:
    get_global_storage = None


def _get_compare_method():
    if get_global_storage is not None:
        try:
            return getattr(get_global_storage(), "compare_method", None)
        except Exception:
            return None
    return None


PYTEST_GOLDEN_MODULE = None

_GOLDEN_CONTEXT = {}


def load_pytest_golden_module():
    """Load tests/pytest/compressor_golden.py as the canonical CPU golden."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE

    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "compressor_golden.py"

    for mod_name in ("torch_npu", "torchair"):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            if mod_name == "torch_npu":
                stub.npu = types.SimpleNamespace(
                    set_device=lambda *a, **kw: None,
                    config=types.SimpleNamespace(allow_internal_format=True),
                )
                testing_mod = types.ModuleType("torch_npu.testing")
                testcase_mod = types.ModuleType("torch_npu.testing.testcase")
                testcase_mod.TestCase = object
                testcase_mod.run_tests = lambda *a, **kw: None
                testing_mod.testcase = testcase_mod
                stub.testing = testing_mod
            sys.modules[mod_name] = stub

    _saved_modules = {}
    for _mod_name in ("cann_ops_transformer", "cann_ops_transformer.ops"):
        if _mod_name in sys.modules:
            _saved_modules[_mod_name] = sys.modules[_mod_name]
    cann_stub = types.ModuleType("cann_ops_transformer")
    ops_stub = types.ModuleType("cann_ops_transformer.ops")
    ops_stub.compressor = lambda *a, **kw: None
    cann_stub.ops = ops_stub
    sys.modules["cann_ops_transformer"] = cann_stub
    sys.modules["cann_ops_transformer.ops"] = ops_stub

    _np_random_state = np.random.get_state()
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"compressor_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
        np.random.set_state(_np_random_state)
        for _mod_name, _saved in _saved_modules.items():
            sys.modules[_mod_name] = _saved
        for _mod_name in ("cann_ops_transformer", "cann_ops_transformer.ops"):
            if _mod_name not in _saved_modules and _mod_name in sys.modules:
                del sys.modules[_mod_name]

    PYTEST_GOLDEN_MODULE = module
    return PYTEST_GOLDEN_MODULE


def ttk_to_cpu(tensor):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        return tensor.detach().cpu()
    return tensor


def ttk_tensor_to_list(val):
    if val is None:
        return None
    if torch.is_tensor(val):
        val = val.detach().cpu().reshape(-1).tolist()
    elif hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        return [int(v) for v in val]
    return [int(val)]


def run_cpu_compressor(
    x,
    wkv,
    wgate,
    state_cache_cpu,
    ape_cpu,
    block_table,
    cmp_ratio,
    coff,
    cache_mode,
    start_pos_list,
    cu_seqlens_list,
    seqused_list,
    grad_enabled=False,
):
    pytest_golden = load_pytest_golden_module()

    x_dtype = x.dtype if x is not None else torch.bfloat16

    state_cache_f32 = state_cache_cpu.to(torch.float32)
    half_dim = state_cache_f32.shape[-1] // 2

    kv_state_golden = state_cache_f32[:, :, :half_dim].contiguous().clone()
    score_state_golden = state_cache_f32[:, :, half_dim:].contiguous().clone()

    update_kv = torch.zeros(kv_state_golden.shape, dtype=torch.bool)
    update_score = torch.zeros(score_state_golden.shape, dtype=torch.bool)

    if ape_cpu is not None and torch.is_tensor(ape_cpu):
        ape_cpu = ape_cpu.to(torch.float32)

    cmp_ratio_val = int(cmp_ratio) if cmp_ratio is not None else 4
    coff_val = int(coff) if coff is not None else 1
    cache_mode_val = int(cache_mode) if cache_mode is not None else 1

    if start_pos_list is None:
        if cu_seqlens_list is not None:
            B = len(cu_seqlens_list) - 1
        elif x is not None and x.dim() == 3:
            B = x.shape[0]
        else:
            B = 1
        start_pos_list = [0] * B

    cmp_kv, cmp_kv_mask, softmax, kv, mid_result_mask = pytest_golden.cpu_compressor(
        x,
        wkv,
        wgate,
        kv_state_golden,
        score_state_golden,
        update_kv,
        update_score,
        ape_cpu,
        block_table=block_table,
        cu_seqlens=cu_seqlens_list,
        seqused=seqused_list,
        start_pos=start_pos_list,
        cmp_ratio=cmp_ratio_val,
        coff=coff_val,
        cache_mode=cache_mode_val,
        grad_enabled=grad_enabled,
    )

    golden_state_cache = torch.zeros_like(state_cache_f32)
    golden_state_cache[:, :, :half_dim] = kv_state_golden
    golden_state_cache[:, :, half_dim:] = score_state_golden
    del kv_state_golden, score_state_golden, state_cache_f32
    gc.collect()
    return (
        cmp_kv.to(x_dtype),
        cmp_kv_mask,
        golden_state_cache.to(state_cache_cpu.dtype),
        update_kv,
        update_score,
        x_dtype,
        softmax,
        kv,
        mid_result_mask,
    )


def _run_benchmark(
    x,
    wkv,
    wgate,
    state_cache_cpu,
    ape_cpu,
    block_table,
    cmp_ratio,
    coff,
    cache_mode,
    start_pos_list,
    cu_seqlens_list,
    seqused_list,
    grad_enabled=False,
):
    """Run benchmark (float64 precision) for cross_check three-way comparison.

    Calls cpu_compressor_fp64 in pytest/compressor_golden.py, which performs
    the same computation as cpu_compressor but with float64 throughout
    (matmul, softmax, state read/write), avoiding the float32 downcast in
    the original cpu_compressor.
    """
    pytest_golden = load_pytest_golden_module()

    x_f64 = x.to(torch.float64) if x is not None else None
    wkv_f64 = wkv.to(torch.float64)
    wgate_f64 = wgate.to(torch.float64)
    state_cache_f64 = state_cache_cpu.to(torch.float64)
    ape_f64 = ape_cpu.to(torch.float64) if ape_cpu is not None else None

    half_dim = state_cache_f64.shape[-1] // 2
    kv_state = state_cache_f64[:, :, :half_dim].contiguous().clone()
    score_state = state_cache_f64[:, :, half_dim:].contiguous().clone()
    update_kv = torch.zeros(kv_state.shape, dtype=torch.bool)
    update_score = torch.zeros(score_state.shape, dtype=torch.bool)

    cmp_ratio_val = int(cmp_ratio) if cmp_ratio is not None else 4
    coff_val = int(coff) if coff is not None else 1
    cache_mode_val = int(cache_mode) if cache_mode is not None else 1

    if start_pos_list is None:
        if cu_seqlens_list is not None:
            B = len(cu_seqlens_list) - 1
        elif x is not None and x.dim() == 3:
            B = x.shape[0]
        else:
            B = 1
        start_pos_list = [0] * B

    cmp_kv, cmp_kv_mask, softmax, kv, mid_result_mask = (
        pytest_golden.cpu_compressor_fp64(
            x_f64,
            wkv_f64,
            wgate_f64,
            kv_state,
            score_state,
            update_kv,
            update_score,
            ape_f64,
            block_table=block_table,
            cu_seqlens=cu_seqlens_list,
            seqused=seqused_list,
            start_pos=start_pos_list,
            cmp_ratio=cmp_ratio_val,
            coff=coff_val,
            cache_mode=cache_mode_val,
            grad_enabled=grad_enabled,
        )
    )

    bench_state_cache = torch.zeros_like(state_cache_f64)
    bench_state_cache[:, :, :half_dim] = kv_state
    bench_state_cache[:, :, half_dim:] = score_state

    del x_f64, wkv_f64, wgate_f64, state_cache_f64, ape_f64
    del kv_state, score_state
    gc.collect()

    if grad_enabled:
        return [cmp_kv, softmax, kv, bench_state_cache]
    return [cmp_kv, bench_state_cache]


def cpu_compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    cmp_ratio=4,
    *,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    coff=1,
    cache_mode=1,
    **kwargs,
):
    x_cpu = ttk_to_cpu(x)
    wkv_cpu = ttk_to_cpu(wkv)
    wgate_cpu = ttk_to_cpu(wgate)
    ape_cpu = ttk_to_cpu(ape)
    state_cache_cpu = ttk_to_cpu(state_cache)
    block_table = ttk_to_cpu(state_block_table)

    start_pos_list = ttk_tensor_to_list(start_pos) if start_pos is not None else None
    cu_seqlens_list = ttk_tensor_to_list(cu_seqlens) if cu_seqlens is not None else None
    seqused_list = ttk_tensor_to_list(seqused) if seqused is not None else None

    if start_pos_list is None:
        if cu_seqlens_list is not None:
            B = len(cu_seqlens_list) - 1
        elif x_cpu is not None and x_cpu.dim() == 3:
            B = x_cpu.shape[0]
        else:
            B = 1
        start_pos_list = [0] * B

    (
        cmp_kv,
        cmp_kv_mask,
        golden_state_cache,
        update_kv,
        update_score,
        x_dtype,
        softmax,
        kv,
        mid_result_mask,
    ) = run_cpu_compressor(
        x_cpu,
        wkv_cpu,
        wgate_cpu,
        state_cache_cpu,
        ape_cpu,
        block_table,
        cmp_ratio,
        coff,
        cache_mode,
        start_pos_list,
        cu_seqlens_list,
        seqused_list,
    )
    is_th = x_cpu.dim() == 2
    _GOLDEN_CONTEXT["cmp_kv_mask"] = cmp_kv_mask
    _GOLDEN_CONTEXT["update_kv"] = update_kv
    _GOLDEN_CONTEXT["update_score"] = update_score
    _GOLDEN_CONTEXT["data_type"] = str(x_dtype)
    _GOLDEN_CONTEXT["cmp_ratio"] = cmp_ratio
    _GOLDEN_CONTEXT["start_pos_list"] = start_pos_list
    _GOLDEN_CONTEXT["seqused_list"] = seqused_list
    _GOLDEN_CONTEXT["cu_seqlens_list"] = cu_seqlens_list
    _GOLDEN_CONTEXT["is_th"] = is_th
    bench_outputs = None
    if _get_compare_method() == "cross_check":
        bench_outputs = _run_benchmark(
            x_cpu,
            wkv_cpu,
            wgate_cpu,
            state_cache_cpu,
            ape_cpu,
            block_table,
            cmp_ratio,
            coff,
            cache_mode,
            start_pos_list,
            cu_seqlens_list,
            seqused_list,
        )
    _GOLDEN_CONTEXT["bench_outputs"] = bench_outputs
    gc.collect()
    return [
        cmp_kv,
        golden_state_cache,
    ]


def aclnn_compressor_golden(
    x,
    wkv,
    wgate,
    stateCacheRef,
    ape,
    stateBlockTable,
    cuSeqlens,
    seqused,
    startPos,
    cmpRatio,
    coff,
    cacheMode,
    stateCacheStrideDim0,
    gradEnabled,
    cmpKv,
    softmaxScoreOut,
    kvOut,
    **kwargs,
):
    x_cpu = ttk_to_cpu(x)
    wkv_cpu = ttk_to_cpu(wkv)
    wgate_cpu = ttk_to_cpu(wgate)
    ape_cpu = ttk_to_cpu(ape)
    state_cache_cpu = ttk_to_cpu(stateCacheRef)
    block_table = ttk_to_cpu(stateBlockTable)

    start_pos_list = ttk_tensor_to_list(startPos) if startPos is not None else None
    cu_seqlens_list = ttk_tensor_to_list(cuSeqlens) if cuSeqlens is not None else None
    seqused_list = ttk_tensor_to_list(seqused) if seqused is not None else None

    (
        cmp_kv,
        cmp_kv_mask,
        golden_state_cache,
        update_kv,
        update_score,
        x_dtype,
        softmax,
        kv,
        mid_result_mask,
    ) = run_cpu_compressor(
        x_cpu,
        wkv_cpu,
        wgate_cpu,
        state_cache_cpu,
        ape_cpu,
        block_table,
        cmpRatio,
        coff,
        cacheMode,
        start_pos_list,
        cu_seqlens_list,
        seqused_list,
        gradEnabled,
    )

    is_th = x_cpu.dim() == 2 if x_cpu is not None else False
    _GOLDEN_CONTEXT["cmp_kv_mask"] = cmp_kv_mask
    _GOLDEN_CONTEXT["update_kv"] = update_kv
    _GOLDEN_CONTEXT["update_score"] = update_score
    _GOLDEN_CONTEXT["data_type"] = str(x_dtype)
    _GOLDEN_CONTEXT["cmp_ratio"] = cmpRatio
    _GOLDEN_CONTEXT["start_pos_list"] = start_pos_list
    _GOLDEN_CONTEXT["seqused_list"] = seqused_list
    _GOLDEN_CONTEXT["cu_seqlens_list"] = cu_seqlens_list
    _GOLDEN_CONTEXT["is_th"] = is_th
    _GOLDEN_CONTEXT["gradEnabled"] = gradEnabled
    _GOLDEN_CONTEXT["mid_result_mask"] = mid_result_mask

    bench_outputs = None
    if _get_compare_method() == "cross_check":
        bench_outputs = _run_benchmark(
            x_cpu,
            wkv_cpu,
            wgate_cpu,
            state_cache_cpu,
            ape_cpu,
            block_table,
            cmpRatio,
            coff,
            cacheMode,
            start_pos_list,
            cu_seqlens_list,
            seqused_list,
            grad_enabled=gradEnabled,
        )
    _GOLDEN_CONTEXT["bench_outputs"] = bench_outputs

    return [cmp_kv, softmax, kv, golden_state_cache]


def get_golden_context():
    return _GOLDEN_CONTEXT


def rebuild_golden_context(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    cmp_ratio,
    *,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    coff=1,
    cache_mode=1,
):
    _GOLDEN_CONTEXT.clear()
    try:
        cpu_compressor(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            cmp_ratio,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            start_pos=start_pos,
            coff=coff,
            cache_mode=cache_mode,
        )
    except Exception:
        pass


def rebuild_golden_context_from_compare_context(compare_context, api_kind="e2e"):
    if compare_context is None:
        return
    try:
        tensors = compare_context.input_tensors
        attrs = dict(compare_context.attributes or {})
        tensors = list(tensors) if tensors is not None else []

        def _t(idx):
            return tensors[idx] if idx < len(tensors) else None

        def _attr(*names, default=None):
            for n in names:
                if n in attrs:
                    return attrs[n]
            return default

        cmp_ratio = _attr("cmp_ratio", "cmpRatio", default=4)
        coff = _attr("coff", default=1)
        cache_mode = _attr("cache_mode", "cacheMode", default=1)
        _grad_enabled = _attr("grad_enabled", "gradEnabled", default=False)

        if api_kind == "aclnn":
            rebuild_golden_context(
                _t(0),
                _t(1),
                _t(2),
                _t(3),
                _t(4),
                cmp_ratio,
                state_block_table=_t(5),
                cu_seqlens=_t(6),
                seqused=_t(7),
                start_pos=_t(8),
                coff=coff,
                cache_mode=cache_mode,
            )
        else:
            rebuild_golden_context(
                _t(0),
                _t(1),
                _t(2),
                _t(3),
                _t(4),
                cmp_ratio,
                state_block_table=_t(5),
                cu_seqlens=_t(6),
                seqused=_t(7),
                start_pos=_t(8),
                coff=coff,
                cache_mode=cache_mode,
            )
    except Exception:
        pass
