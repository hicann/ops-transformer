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


PYTEST_GOLDEN_MODULE = None

_GOLDEN_CONTEXT = {}


def load_pytest_golden_module():
    """Load tests/pytest/quant_compressor_golden.py as the canonical CPU golden."""
    global PYTEST_GOLDEN_MODULE
    if PYTEST_GOLDEN_MODULE is not None:
        return PYTEST_GOLDEN_MODULE

    pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
    module_path = pytest_dir / "quant_compressor_golden.py"

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

    if "cann_ops_transformer" not in sys.modules:
        cann_stub = types.ModuleType("cann_ops_transformer")
        ops_stub = types.ModuleType("cann_ops_transformer.ops")
        quant_stub = types.ModuleType("cann_ops_transformer.ops.quant_compressor")
        quant_stub.CacheMode = type(
            "CacheMode", (), {"LINEAR_BUFFER": 1, "RING_BUFFER": 2}
        )
        quant_stub.QuantMode = type(
            "QuantMode", (), {"A8W8_A_HIFP8_PER_TENSOR_W_HIFP8_PER_CHANNEL": 1}
        )
        ops_stub.quant_compressor = quant_stub
        cann_stub.ops = ops_stub
        sys.modules["cann_ops_transformer"] = cann_stub
        sys.modules["cann_ops_transformer.ops"] = ops_stub
        sys.modules["cann_ops_transformer.ops.quant_compressor"] = quant_stub

    _np_random_state = np.random.get_state()
    sys.path.insert(0, str(pytest_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"quant_compressor_pytest_golden_{abs(hash(module_path))}", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(pytest_dir))
        except ValueError:
            pass
        np.random.set_state(_np_random_state)

    PYTEST_GOLDEN_MODULE = module
    return PYTEST_GOLDEN_MODULE


def ttk_to_cpu(tensor):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        return tensor.detach().cpu()
    if isinstance(tensor, np.ndarray):
        if tensor.dtype == np.dtype("hifloat8"):
            return torch.from_numpy(tensor.view(np.uint8))
        return torch.from_numpy(tensor)
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


def run_cpu_quant_compressor(
    x,
    wkv,
    wgate,
    x_descale,
    wkv_descale,
    wgate_descale,
    state_cache_cpu,
    ape_cpu,
    block_table,
    cmp_ratio,
    coff,
    cache_mode,
    quant_mode,
    start_pos_list,
    cu_seqlens_list,
    seqused_list,
):
    pytest_golden = load_pytest_golden_module()

    out_dtype = torch.bfloat16

    state_cache_f32 = state_cache_cpu.to(torch.float32)
    half_dim = state_cache_f32.shape[-1] // 2

    kv_state_golden = state_cache_f32[:, :, :half_dim].contiguous().clone()
    score_state_golden = state_cache_f32[:, :, half_dim:].contiguous().clone()

    update_kv = torch.zeros(kv_state_golden.shape, dtype=torch.bool)
    update_score = torch.zeros(score_state_golden.shape, dtype=torch.bool)

    if ape_cpu is not None and torch.is_tensor(ape_cpu):
        ape_cpu = ape_cpu.to(torch.float32)

    if x_descale is not None and torch.is_tensor(x_descale):
        x_descale = x_descale.to(torch.float32)
    if wkv_descale is not None and torch.is_tensor(wkv_descale):
        wkv_descale = wkv_descale.to(torch.float32)
    if wgate_descale is not None and torch.is_tensor(wgate_descale):
        wgate_descale = wgate_descale.to(torch.float32)

    cmp_ratio_val = int(cmp_ratio) if cmp_ratio is not None else 4
    coff_val = int(coff) if coff is not None else 1
    cache_mode_val = int(cache_mode) if cache_mode is not None else 1
    quant_mode_val = int(quant_mode) if quant_mode is not None else 1

    if start_pos_list is None:
        if cu_seqlens_list is not None:
            B = len(cu_seqlens_list) - 1
        elif x is not None and x.dim() == 3:
            B = x.shape[0]
        else:
            B = 1
        start_pos_list = [0] * B

    cmp_kv, cmp_kv_mask = pytest_golden.cpu_compressor(
        x,
        wkv,
        wgate,
        x_descale,
        wkv_descale,
        wgate_descale,
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
        quant_mode=quant_mode_val,
    )

    golden_state_cache = torch.zeros_like(state_cache_f32)
    golden_state_cache[:, :, :half_dim] = kv_state_golden
    golden_state_cache[:, :, half_dim:] = score_state_golden

    return (
        cmp_kv.to(out_dtype),
        cmp_kv_mask,
        golden_state_cache.to(state_cache_cpu.dtype),
        update_kv,
        update_score,
        out_dtype,
    )


def cpu_quant_compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    quant_mode,
    cmp_ratio,
    *,
    x_descale=None,
    wkv_descale=None,
    wgate_descale=None,
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
    x_descale_cpu = ttk_to_cpu(x_descale)
    wkv_descale_cpu = ttk_to_cpu(wkv_descale)
    wgate_descale_cpu = ttk_to_cpu(wgate_descale)

    start_pos_list = ttk_tensor_to_list(start_pos) if start_pos is not None else None
    cu_seqlens_list = ttk_tensor_to_list(cu_seqlens) if cu_seqlens is not None else None
    seqused_list = ttk_tensor_to_list(seqused) if seqused is not None else None

    # run_cpu_quant_compressor 内部会在 start_pos_list 为 None 时默认 [0]*B，
    # 但那个默认值不会返回。这里提前设默认值，确保 _GOLDEN_CONTEXT 拿到非 None。
    if start_pos_list is None:
        if cu_seqlens_list is not None:
            B = len(cu_seqlens_list) - 1
        elif x_cpu is not None and x_cpu.dim() == 3:
            B = x_cpu.shape[0]
        else:
            B = 1
        start_pos_list = [0] * B

    cmp_kv, cmp_kv_mask, golden_state_cache, update_kv, update_score, out_dtype = (
        run_cpu_quant_compressor(
            x_cpu,
            wkv_cpu,
            wgate_cpu,
            x_descale_cpu,
            wkv_descale_cpu,
            wgate_descale_cpu,
            state_cache_cpu,
            ape_cpu,
            block_table,
            cmp_ratio,
            coff,
            cache_mode,
            quant_mode,
            start_pos_list,
            cu_seqlens_list,
            seqused_list,
        )
    )
    is_th = x_cpu.dim() == 2
    _GOLDEN_CONTEXT["cmp_kv_mask"] = cmp_kv_mask
    _GOLDEN_CONTEXT["update_kv"] = update_kv
    _GOLDEN_CONTEXT["update_score"] = update_score
    _GOLDEN_CONTEXT["data_type"] = str(out_dtype)
    _GOLDEN_CONTEXT["cmp_ratio"] = cmp_ratio
    _GOLDEN_CONTEXT["start_pos_list"] = start_pos_list
    _GOLDEN_CONTEXT["seqused_list"] = seqused_list
    _GOLDEN_CONTEXT["cu_seqlens_list"] = cu_seqlens_list
    _GOLDEN_CONTEXT["is_th"] = is_th

    return [
        cmp_kv,
        golden_state_cache,
    ]


def aclnn_quant_compressor_golden(
    x,
    wkv,
    wgate,
    stateCacheRef,
    ape,
    xDescale,
    wkvDescale,
    wgateDescale,
    stateBlockTable,
    cuSeqlens,
    seqused,
    startPos,
    quantMode,
    cmpRatio,
    coff,
    cacheMode,
    stateCacheStrideDim0,
    cmpKv,
    **kwargs,
):
    x_cpu = ttk_to_cpu(x)
    wkv_cpu = ttk_to_cpu(wkv)
    wgate_cpu = ttk_to_cpu(wgate)
    ape_cpu = ttk_to_cpu(ape)
    state_cache_cpu = ttk_to_cpu(stateCacheRef)
    block_table = ttk_to_cpu(stateBlockTable)
    x_descale_cpu = ttk_to_cpu(xDescale)
    wkv_descale_cpu = ttk_to_cpu(wkvDescale)
    wgate_descale_cpu = ttk_to_cpu(wgateDescale)

    start_pos_list = ttk_tensor_to_list(startPos) if startPos is not None else None
    cu_seqlens_list = ttk_tensor_to_list(cuSeqlens) if cuSeqlens is not None else None
    seqused_list = ttk_tensor_to_list(seqused) if seqused is not None else None

    cmp_kv, cmp_kv_mask, golden_state_cache, update_kv, update_score, out_dtype = (
        run_cpu_quant_compressor(
            x_cpu,
            wkv_cpu,
            wgate_cpu,
            x_descale_cpu,
            wkv_descale_cpu,
            wgate_descale_cpu,
            state_cache_cpu,
            ape_cpu,
            block_table,
            cmpRatio,
            coff,
            cacheMode,
            quantMode,
            start_pos_list,
            cu_seqlens_list,
            seqused_list,
        )
    )

    is_th = x_cpu.dim() == 2 if x_cpu is not None else False
    _GOLDEN_CONTEXT["cmp_kv_mask"] = cmp_kv_mask
    _GOLDEN_CONTEXT["update_kv"] = update_kv
    _GOLDEN_CONTEXT["update_score"] = update_score
    _GOLDEN_CONTEXT["data_type"] = str(out_dtype)
    _GOLDEN_CONTEXT["cmp_ratio"] = cmpRatio
    _GOLDEN_CONTEXT["start_pos_list"] = start_pos_list
    _GOLDEN_CONTEXT["seqused_list"] = seqused_list
    _GOLDEN_CONTEXT["cu_seqlens_list"] = cu_seqlens_list
    _GOLDEN_CONTEXT["is_th"] = is_th

    # Return order MUST align with output_tensor_indexes=(3,12):
    #   idx 3 = stateCacheRef (inplace output)  -> golden_state_cache
    #   idx 12 = cmpKvOut (pure output)         -> cmp_kv
    # load_goldens validates saved golden shape against device output shape in
    # this same order; a mismatch raises MANUAL_DATA_READ_FAILURE.
    return [golden_state_cache, cmp_kv]


def get_golden_context():
    return _GOLDEN_CONTEXT


def rebuild_golden_context(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    quant_mode,
    cmp_ratio,
    *,
    x_descale=None,
    wkv_descale=None,
    wgate_descale=None,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    coff=1,
    cache_mode=1,
):
    """Recompute and populate _GOLDEN_CONTEXT (cmp_kv_mask / update_kv / update_score).

    In TTK e2e replay mode the golden plugin is not re-executed (goldens are
    loaded from bin files), so the module-level _GOLDEN_CONTEXT stays empty and
    compare() receives None masks, which makes kv_state_origin / score_state_origin
    fall back to N/A. This helper re-runs the CPU golden on the prepared inputs
    (already produced by customize_inputs) to restore the masks.

    The context is reset before recomputation: _GOLDEN_CONTEXT is module-level and
    would otherwise leak across testcases in the same process (case B would reuse
    case A's masks). In prepare mode the golden plugin runs afterwards and
    overwrites the context again, so the reset is harmless.
    """
    _GOLDEN_CONTEXT.clear()
    try:
        cpu_quant_compressor(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            quant_mode,
            cmp_ratio,
            x_descale=x_descale,
            wkv_descale=wkv_descale,
            wgate_descale=wgate_descale,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            start_pos=start_pos,
            coff=coff,
            cache_mode=cache_mode,
        )
    except Exception:
        # Best-effort: if context rebuild fails, leave context empty so the
        # existing N/A fallback behaviour is preserved rather than crashing
        # the whole comparison.
        pass


def rebuild_golden_context_from_compare_context(compare_context, api_kind="e2e"):
    """Rebuild _GOLDEN_CONTEXT from a TTK CompareContext (replay mode).

    In TTK e2e/aclnn replay mode the golden plugin is not re-executed (goldens
    are loaded from bin files), so the module-level _GOLDEN_CONTEXT stays empty
    and compare() receives None masks, which makes kv_state_origin /
    score_state_origin fall back to N/A. This helper extracts the prepared
    inputs + attributes carried by CompareContext (populated from testcase.tensors
    / testcase.attributes, which are available in both prepare and replay) and
    re-runs the CPU golden to restore the masks.

    api_kind: "e2e" or "aclnn" — selects the tensor position layout.
    """
    if compare_context is None:
        return
    try:
        tensors = compare_context.input_tensors
        attrs = dict(compare_context.attributes or {})
        tensors = list(tensors) if tensors is not None else []

        def _t(idx):
            return tensors[idx] if idx < len(tensors) else None

        # e2e/kernel CSV attributes use snake_case, aclnn CSV uses camelCase
        # (matching aclnn header parameter names). Fall back to defaults if absent.
        def _attr(*names, default=None):
            for n in names:
                if n in attrs:
                    return attrs[n]
            return default

        quant_mode = _attr("quant_mode", "quantMode", default=1)
        cmp_ratio = _attr("cmp_ratio", "cmpRatio", default=4)
        coff = _attr("coff", default=1)
        cache_mode = _attr("cache_mode", "cacheMode", default=1)

        if api_kind == "aclnn":
            # aclnn signature:
            # x, wkv, wgate, stateCacheRef, ape, xDescale, wkvDescale, wgateDescale,
            # stateBlockTable, cuSeqlens, seqused, startPos, ..., cmpKvOut
            rebuild_golden_context(
                _t(0),
                _t(1),
                _t(2),
                _t(3),
                _t(4),
                quant_mode,
                cmp_ratio,
                x_descale=_t(5),
                wkv_descale=_t(6),
                wgate_descale=_t(7),
                state_block_table=_t(8),
                cu_seqlens=_t(9),
                seqused=_t(10),
                start_pos=_t(11),
                coff=coff,
                cache_mode=cache_mode,
            )
        else:
            # e2e / kernel signature:
            # x, wkv, wgate, state_cache, ape, x_descale, wkv_descale, wgate_descale,
            # state_block_table, cu_seqlens, seqused, start_pos
            rebuild_golden_context(
                _t(0),
                _t(1),
                _t(2),
                _t(3),
                _t(4),
                quant_mode,
                cmp_ratio,
                x_descale=_t(5),
                wkv_descale=_t(6),
                wgate_descale=_t(7),
                state_block_table=_t(8),
                cu_seqlens=_t(9),
                seqused=_t(10),
                start_pos=_t(11),
                coff=coff,
                cache_mode=cache_mode,
            )
    except Exception:
        pass


def kernel_quant_compressor_golden(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    x_descale=None,
    wkv_descale=None,
    wgate_descale=None,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    **kwargs,
):
    """Golden for kernel mode: 12 tensors positionally + attributes as kwargs."""
    return cpu_quant_compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        kwargs.get("quant_mode", 1),
        kwargs.get("cmp_ratio", 4),
        x_descale=x_descale,
        wkv_descale=wkv_descale,
        wgate_descale=wgate_descale,
        state_block_table=state_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        coff=kwargs.get("coff", 1),
        cache_mode=kwargs.get("cache_mode", 1),
        **{
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "quant_mode",
                "cmp_ratio",
                "coff",
                "cache_mode",
                "state_cache_stride_dim0",
                "full_soc_version",
                "short_soc_version",
            )
        },
    )
