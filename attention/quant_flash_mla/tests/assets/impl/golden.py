#!/usr/bin/python3
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


"""TTK golden plugin for quant_flash_mla.

Recovers BNSD data from the actual TTK tensor slots, calls the CPU reference,
converts output to the target layout, and returns
(attention_out, lse_out) for comparison.
"""

import importlib.util
import logging
import os
import sys

import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_ASSETS_DIR, "..")  # assets/
if _GOLDEN_DIR not in sys.path:
    sys.path.insert(0, _GOLDEN_DIR)

_GOLDEN_MODULE_NAME = "qfmla_assets_golden_reference"
if _GOLDEN_MODULE_NAME in sys.modules:
    golden_mod = sys.modules[_GOLDEN_MODULE_NAME]
else:
    _GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "qfmla_golden.py")
    _GOLDEN_SPEC = importlib.util.spec_from_file_location(
        _GOLDEN_MODULE_NAME, _GOLDEN_PATH
    )
    golden_mod = importlib.util.module_from_spec(_GOLDEN_SPEC)
    sys.modules[_GOLDEN_MODULE_NAME] = golden_mod
    _GOLDEN_SPEC.loader.exec_module(golden_mod)

logger = logging.getLogger(__name__)


def _load_local(stem):
    name = f"qfmla_assets_impl_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ASSETS_DIR, stem + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _configure_reference(
    cfg,
    quant_mode,
    mask_mode,
    softmax_scale,
    max_seqlen_q,
    max_seqlen_kv,
    return_softmax_lse,
    kwargs,
):
    # Match QFA's per-call numerical-reference configuration. Generation
    # seeds/ranges belong to inputs.py and are neither read nor reset here.
    # Cross-check uses the global reference; two-way uses NPU sections if available.
    npu = getattr(torch, "npu", None)
    golden_mod.ENABLE_SECTION_GOLDEN = (
        not _bench_enabled() and npu is not None and npu.is_available()
    )
    logger.info(
        "[GOLDEN] NPU metadata section reference enabled: %s",
        golden_mod.ENABLE_SECTION_GOLDEN,
    )
    golden_mod.FP8_DTYPE = torch.float8_e4m3fn
    golden_mod.P_SCALE = 1.0
    golden_mod._csv_precision_tolerances = kwargs.get("precision_tolerances")
    golden_mod._csv_absolute_precision = kwargs.get("absolute_precision")
    for name in ("B", "N_q", "N_kv", "D", "D_V"):
        setattr(golden_mod, name, cfg[name])
    golden_mod.QFMLA_QK_HEAD_DIM = cfg["D"]
    golden_mod.QFMLA_V_HEAD_DIM = cfg["D_V"]
    golden_mod.QFMLA_QUANT_MODE = quant_mode
    golden_mod.SEQUSED_Q = list(cfg["seqused_q_vals"])
    golden_mod.CACHE_SEQLENS = list(cfg["cache_seqlens_vals"])
    golden_mod.CU_SEQLENS_Q = cfg["cu_seqlens_q_vals"]
    golden_mod.ENABLE_PA = True
    golden_mod.ENABLE_LSE = return_softmax_lse
    golden_mod.GOLDEN_MODE = True
    golden_mod.BLOCK_SIZE = cfg["block_size"]
    golden_mod.MASK_MODE = mask_mode
    golden_mod.INPUT_LAYOUT = cfg["input_layout"]
    golden_mod.OUTPUT_LAYOUT = cfg["output_layout"]
    golden_mod.KV_CACHE_LAYOUT = cfg["kv_cache_layout"]
    golden_mod.SCALE_VALUE = softmax_scale
    golden_mod.MAX_SEQLEN_Q = max_seqlen_q
    golden_mod.MAX_SEQLEN_KV = max_seqlen_kv
    # Only NPU two-way runs obtain metadata here. CPU-only and cross-check
    # runs use the global reference, without a separate metadata-selection switch.


def _bench_enabled():
    """是否生成三方 bench: 仅 CLI 显式 --compare cross_check 时生成。

    默认(未指定 --compare)不生成, 避免两方回归白跑一遍 fp64 bench。
    """
    try:
        from ttk.switches import get_global_storage
    except Exception:
        try:
            from ttk.utilities.container_utils import get_global_storage
        except Exception:
            return False
    try:
        method = getattr(get_global_storage(), "compare_method", None)
    except Exception:  # pylint: disable=broad-except
        return False
    return method is not None and str(method).lower() == "cross_check"


_BENCH_CACHE = {}
_LAST_BENCH = None
# Independent fixed-block approximation for the cross-check third-party slot.
_REF_CACHE = {}


def peek_bench(testcase_name=None):
    """取当前用例的 bench 输出; 未生成时返回 None。"""
    if testcase_name is not None:
        return _BENCH_CACHE.get(str(testcase_name))
    return _LAST_BENCH


def peek_sectioned_ref(testcase_name=None):
    """取当前用例的分段量化参考; 未生成时返回 None。"""
    return _REF_CACHE.get(str(testcase_name))


def _convert_tnd_full_rows(
    bnsd, seqused, cu_values, total_t, to_ntd=False, fill_value=0.0
):
    """TND/NTD输出按算子语义转换: T=全量T(q行数), 每批按cu偏移定位,
    有效行=seqused[b], 无效行attn补0/lse补inf (NPU实测: 无效行attn=0, lse=inf)。
    bnsd: attn为(B,N,S,Dv) / lse为(B,N,S,1)"""
    fill_val = fill_value
    if to_ntd:
        out = torch.full(
            (bnsd.shape[1], total_t, bnsd.shape[3]), fill_val, dtype=bnsd.dtype
        )
    else:
        out = torch.full(
            (total_t, bnsd.shape[1], bnsd.shape[3]), fill_val, dtype=bnsd.dtype
        )
    for b in range(bnsd.shape[0]):
        act_s = int(seqused[b])
        if act_s <= 0:
            continue
        start = int(cu_values[b])
        valid = bnsd[b, :, :act_s, :]  # (N, act_s, D)
        if to_ntd:
            out[:, start : start + act_s, :] = valid
        else:
            out[start : start + act_s, :, :] = valid.permute(1, 0, 2)
    return out.contiguous()


def _convert_outputs(
    result_bnsd,
    lse_bnsd,
    q,
    cfg,
    kwargs,
    golden_seqused,
    return_softmax_lse,
    layout_q,
    to_bf16=True,
):
    """Restore output layout; LSE layout follows query, not attention output."""
    output_layout = cfg["output_layout"]
    cu = cfg["cu_seqlens_q_vals"]
    if output_layout == "BNSD":
        result = result_bnsd.contiguous()
    elif output_layout == "BSND":
        result = result_bnsd.permute(0, 2, 1, 3).contiguous()
    elif output_layout in ("TND", "NTD"):
        if layout_q == "TND":
            result = _convert_tnd_full_rows(
                result_bnsd,
                golden_seqused,
                cu,
                int(q.shape[0]),
                to_ntd=output_layout == "NTD",
            )
        else:
            result = result_bnsd.permute(0, 2, 1, 3).flatten(0, 1).contiguous()
            if output_layout == "NTD":
                result = result.permute(1, 0, 2).contiguous()
    else:
        raise ValueError(f"[GOLDEN] cannot represent layout_out={output_layout!r}")
    lse = None
    if return_softmax_lse and lse_bnsd is not None:
        if layout_q == "TND":
            lse = _convert_tnd_full_rows(
                lse_bnsd,
                golden_seqused,
                cu,
                int(q.shape[0]),
                to_ntd=True,
                fill_value=float("inf"),
            ).squeeze(-1)
        else:
            lse = lse_bnsd.squeeze(-1).contiguous()
    return (result.to(torch.bfloat16) if to_bf16 else result), lse


def _cache_bench(
    testcase_name, cfg, kwargs, q, golden_seqused, return_softmax_lse, layout_q, tensors
):
    """生成并缓存当前用例的 fp64 bench, 供 compare 侧三方比对读取。"""
    global _LAST_BENCH
    name = str(testcase_name)
    _BENCH_CACHE.pop(name, None)
    _LAST_BENCH = None
    _REF_CACHE.pop(name, None)
    if not _bench_enabled():
        return
    q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v = tensors
    bench_bnsd, bench_lse_bnsd = golden_mod.fp8_fullquant_mla_bench(
        q_fp8,
        k_fp8,
        v_fp8,
        deq_q,
        deq_k,
        deq_v,
        golden_seqused,
        list(cfg["cache_seqlens_vals"]),
    )
    bench_result, bench_lse = _convert_outputs(
        bench_bnsd,
        bench_lse_bnsd,
        q,
        cfg,
        kwargs,
        golden_seqused,
        return_softmax_lse,
        layout_q,
        to_bf16=False,
    )
    _LAST_BENCH = [bench_result, bench_lse]
    _BENCH_CACHE[name] = _LAST_BENCH

    # Fixed-block third-party approximation; it is not the target metadata schedule.
    # Its failure must remain visible when cross_check was explicitly requested.
    ref = golden_mod.fp8_fullquant_mla_sectioned_golden(
        q_fp8,
        k_fp8,
        v_fp8,
        deq_q,
        deq_k,
        deq_v,
        golden_seqused,
        list(cfg["cache_seqlens_vals"]),
    )
    if ref is not None:
        ref_result, ref_lse = _convert_outputs(
            ref[0],
            ref[1],
            q,
            cfg,
            kwargs,
            golden_seqused,
            return_softmax_lse,
            layout_q,
            to_bf16=True,
        )
        _REF_CACHE[name] = [ref_result, ref_lse]
        logger.info("[%s] sectioned ref: result=%s", name, tuple(ref_result.shape))
    logger.info(
        "[%s] bench: result=%s lse=%s",
        name,
        tuple(bench_result.shape),
        None if bench_lse is None else tuple(bench_lse.shape),
    )


def cpu_qfmla(
    q,
    k_cache,
    q_descale,
    k_descale,
    block_table,
    cache_seqlens,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=None,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_v=512,
    layout_q="TND",
    layout_kv="PA_BNBD",
    layout_out="TND",
    return_softmax_lse=False,
    **kwargs,
):
    """Compute CPU golden, bench and reference from the same actual inputs.

    Negative cases without a meaningful numerical reference should select
    TTK's golden_api=disable. Do not repair them here or return a fake PASS.
    """
    global _LAST_BENCH
    testcase_name = str(kwargs.get("testcase_name", "unknown"))
    _BENCH_CACHE.pop(testcase_name, None)
    _REF_CACHE.pop(testcase_name, None)
    _LAST_BENCH = None
    cfg, tensors = _load_local("restore_inputs").restore_inputs(
        q,
        k_cache,
        q_descale,
        k_descale,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        seqused_q,
        layout_q,
        layout_kv,
        layout_out,
        512 if head_dim_v is None else head_dim_v,
        quant_mode,
    )
    if softmax_scale is None:
        softmax_scale = q.size(-1) ** -0.5
    _configure_reference(
        cfg,
        quant_mode,
        mask_mode,
        softmax_scale,
        max_seqlen_q,
        max_seqlen_kv,
        return_softmax_lse,
        kwargs,
    )
    q_data, k_data, v_data, deq_q, deq_k, deq_v = tensors
    result_bnsd, lse_bnsd = golden_mod.cpu_fp8_fullquant_mla_golden(
        q_data,
        k_data,
        v_data,
        deq_q,
        deq_k,
        deq_v,
        torch.ones(1, dtype=torch.float32),
        cfg["seqused_q_vals"],
        cfg["cache_seqlens_vals"],
    )
    outputs = _convert_outputs(
        result_bnsd,
        lse_bnsd,
        q,
        cfg,
        kwargs,
        cfg["seqused_q_vals"],
        return_softmax_lse,
        layout_q,
    )
    _cache_bench(
        testcase_name,
        cfg,
        kwargs,
        q,
        cfg["seqused_q_vals"],
        return_softmax_lse,
        layout_q,
        tensors,
    )
    if _bench_enabled():
        _load_local("offline_cross_check").save(
            testcase_name,
            [
                q,
                k_cache,
                q_descale,
                k_descale,
                block_table,
                cache_seqlens,
                cu_seqlens_q,
                seqused_q,
                attn_mask,
            ],
            {
                "quant_mode": quant_mode,
                "softmax_scale": softmax_scale,
                "mask_mode": mask_mode,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_kv": max_seqlen_kv,
                "head_dim_v": head_dim_v,
                "layout_q": layout_q,
                "layout_kv": layout_kv,
                "layout_out": layout_out,
                "return_softmax_lse": return_softmax_lse,
            },
            peek_bench(testcase_name),
            peek_sectioned_ref(testcase_name),
        )
    logger.info(
        "[%s] golden consumed actual slots: q=%s k=%s output=%s",
        testcase_name,
        tuple(q_data.shape),
        tuple(k_data.shape),
        tuple(outputs[0].shape),
    )
    return outputs
