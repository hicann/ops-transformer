#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""TTK e2e golden for mixed_quant_flash_attn.

直接调用 pytest 的 core.cpu.tforward 作为 CPU golden。
TTK 传入的 tensor 可能是 numpy array，需转为 torch tensor 后传给 tforward。
"""

__golden__ = {
    "e2e": {
        "torch.ops.cann_ops_transformer.mixed_quant_flash_attn": "cpu_mixed_quant_flash_attn"
    }
}

import sys
import traceback

import numpy as np
import torch

from core.cpu import run_cpu_golden

try:
    from ttk.utilities.container_utils import get_global_storage
except ImportError:
    try:
        from ttk.switches import get_global_storage
    except ImportError:
        get_global_storage = None


_GOLDEN_CONTEXT = {}


def _record_bench_error(exc):
    """第三方 golden 失败时打印根因到 stderr，避免静默吞掉。"""
    detail = f"{type(exc).__name__}: {exc}"
    print(f"[golden] third-party golden failed: {detail}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)


def _get_compare_method():
    if get_global_storage is not None:
        try:
            return getattr(get_global_storage(), "compare_method", None)
        except Exception:
            return None
    return None


def _to_torch(val, dtype=None):
    """把 numpy/torch/scalar 转为 torch tensor。
    处理 numpy float8_e8m0 和 bfloat16 等非原生 dtype（通过 uint16/uint8 view）。
    """
    if val is None:
        return None
    if torch.is_tensor(val):
        return val.detach().cpu()
    if isinstance(val, np.ndarray):
        dt = str(val.dtype)
        if dt.startswith("float8"):
            return torch.from_numpy(val.view(np.uint8).copy()).view(
                torch.float8_e8m0fnu
            )
        if dt == "bfloat16":
            return torch.from_numpy(val.view(np.uint16).copy()).view(torch.bfloat16)
        return torch.from_numpy(val.copy())
    if isinstance(val, (list, tuple)):
        return torch.tensor(val, dtype=dtype or torch.int32)
    return torch.tensor(val)


def _run_third_party_golden(input_kwargs):
    # 第三方切换：注释当前 return，取消另一行注释；同进程和独立进程重建共用此处。
    return run_cpu_golden(input_kwargs, mode="npu")  # NPU 小算子
    # return run_cpu_golden(input_kwargs, mode="cpu")  # CPU 低精


def cpu_mixed_quant_flash_attn(
    q,
    k,
    v,
    k_descale,
    v_descale,
    block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    quant_mode=1,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_attn_out="BSND",
    return_softmax_lse=False,
    batch_size=1,
    num_heads_q=1,
    num_heads_kv=1,
    head_dim=128,
    **kwargs,
):
    """TTK golden adapter: 调用 pytest tforward 计算 CPU golden。"""
    # numpy → torch 转换
    input_kwargs = {
        "q": _to_torch(q, torch.bfloat16),
        "k": _to_torch(k, torch.uint8),
        "v": _to_torch(v, torch.uint8),
        "k_descale": _to_torch(k_descale),
        "v_descale": _to_torch(v_descale),
        "block_table": _to_torch(block_table, torch.int32),
        "cu_seqlens_q": _to_torch(cu_seqlens_q, torch.int32),
        "seqused_q": _to_torch(seqused_q, torch.int32),
        "seqused_kv": _to_torch(seqused_kv, torch.int32),
        "sinks": _to_torch(sinks, torch.float32),
        "attn_mask": _to_torch(attn_mask, torch.int8),
        "quant_mode": int(quant_mode),
        "softmax_scale": float(softmax_scale),
        "mask_mode": int(mask_mode),
        "win_left": int(win_left),
        "win_right": int(win_right),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_kv": int(max_seqlen_kv),
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_attn_out": layout_attn_out,
        "return_softmax_lse": bool(return_softmax_lse),
        "sinner": int(kwargs.get("sinner", 512)),
        "souter": int(kwargs.get("souter", 32)),
    }

    cross_check = _get_compare_method() == "cross_check"
    result = run_cpu_golden(input_kwargs, mode="cpu_high" if cross_check else "cpu")
    if cross_check:
        # FP64 CPU result is the truth; bench_outputs holds the third-party result.
        _GOLDEN_CONTEXT["data_type"] = str(getattr(q, "dtype", "bfloat16"))
        try:
            bench = _run_third_party_golden(input_kwargs)
            _GOLDEN_CONTEXT["bench_outputs"] = (bench["attn_out"], bench["softmax_lse"])
        except Exception as exc:
            _GOLDEN_CONTEXT["bench_outputs"] = None
            _record_bench_error(exc)
    return result["attn_out"], result["softmax_lse"]


def get_golden_context():
    return _GOLDEN_CONTEXT


def rebuild_golden_context_from_compare_context(compare_context):
    """Re-run the selected third-party golden from compare_context.

    TTK 可能在独立进程里跑 golden 与 compare，cross_check 需要在 compare 进程里
    重建 bench_outputs。仅计算第三方结果（golden 本体已由 TTK 传入）。
    """
    if compare_context is None:
        return
    try:
        tensors = (
            list(compare_context.input_tensors)
            if compare_context.input_tensors is not None
            else []
        )
        attrs = dict(compare_context.attributes or {})

        def _t(idx):
            return tensors[idx] if idx < len(tensors) else None

        def _attr(*names, default=None):
            for n in names:
                if n in attrs:
                    return attrs[n]
            return default

        input_kwargs = {
            "q": _to_torch(_t(0), torch.bfloat16),
            "k": _to_torch(_t(1), torch.uint8),
            "v": _to_torch(_t(2), torch.uint8),
            "k_descale": _to_torch(_t(3)),
            "v_descale": _to_torch(_t(4)),
            "block_table": _to_torch(_t(5), torch.int32),
            "cu_seqlens_q": _to_torch(_t(6), torch.int32),
            "seqused_q": _to_torch(_t(7), torch.int32),
            "seqused_kv": _to_torch(_t(8), torch.int32),
            "sinks": _to_torch(_t(9), torch.float32),
            "attn_mask": _to_torch(_t(10), torch.int8),
            "quant_mode": int(_attr("quant_mode", "quant_compute_mode", default=1)),
            "softmax_scale": float(_attr("softmax_scale", default=1.0)),
            "mask_mode": int(_attr("mask_mode", default=0)),
            "win_left": int(_attr("win_left", default=-1)),
            "win_right": int(_attr("win_right", default=-1)),
            "max_seqlen_q": int(_attr("max_seqlen_q", default=-1)),
            "max_seqlen_kv": int(_attr("max_seqlen_kv", default=-1)),
            "layout_q": _attr("layout_q", default="BSND"),
            "layout_kv": _attr("layout_kv", default="PA_BBND"),
            "layout_attn_out": _attr("layout_attn_out", default="BSND"),
            "return_softmax_lse": bool(_attr("return_softmax_lse", default=False)),
            "sinner": int(_attr("sinner", default=512)),
            "souter": int(_attr("souter", default=32)),
        }
        _GOLDEN_CONTEXT["data_type"] = str(input_kwargs["q"].dtype)
        bench = _run_third_party_golden(input_kwargs)
        _GOLDEN_CONTEXT["bench_outputs"] = (bench["attn_out"], bench["softmax_lse"])
    except Exception as exc:
        _GOLDEN_CONTEXT["bench_outputs"] = None
        _record_bench_error(exc)
