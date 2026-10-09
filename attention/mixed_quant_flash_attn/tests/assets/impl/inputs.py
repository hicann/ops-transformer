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

"""TTK e2e customize_inputs for mixed_quant_flash_attn.

直接调用 pytest 的 core.gen_data.generate_inputs 生成全部输入，写回 TTK 传入的 tensor。
这样 ttk 和 pytest 用完全相同的输入数据（相同种子、相同生成逻辑）。
"""

__input__ = {"e2e": {"mqfa_ttk_ops.call_npu": "generate_mixed_quant_inputs"}}

import os
import math
import numpy as np
import torch

# 直接 import pytest 的 gen_data
from core.gen_data import generate_inputs


def _to_int_list(val):
    if val is None:
        return None
    if torch.is_tensor(val):
        return val.detach().cpu().reshape(-1).tolist()
    if isinstance(val, np.ndarray):
        return val.reshape(-1).tolist()
    if isinstance(val, (list, tuple)):
        return [int(v) for v in val]
    return [int(val)]


def _to_q_dtype(tensor):
    """从传入的 q tensor 的 dtype 推导 generate_inputs 的 q_dtype 字符串。"""
    dtype = tensor.dtype
    if torch.is_tensor(tensor):
        return (
            "fp16"
            if dtype == torch.float16
            else "bf16"
            if dtype == torch.bfloat16
            else "fp32"
        )
    dt = str(dtype)
    return "fp16" if dt == "float16" else "bf16" if dt == "bfloat16" else "fp32"


def _resolve_axis_layout(
    q, k, layout_q, layout_kv, cu_seqlens_q, seqused_q, seqused_kv
):
    if layout_q == "BNSD":
        b, n1, s1, d = q.shape
    elif layout_q == "BSND":
        b, s1, n1, d = q.shape
    elif layout_q == "TND":
        n1, d = q.shape[1], q.shape[2]
        if cu_seqlens_q is not None:
            b = len(_to_int_list(cu_seqlens_q)) - 1
        elif seqused_q is not None:
            b = len(_to_int_list(seqused_q))
        else:
            b = 1
        s1 = q.shape[0]
    else:
        b, n1, s1, d = q.shape
    if "PA" in layout_kv:
        block_num = k.shape[0]
        if layout_kv == "PA_NZ":
            n2, block_size = k.shape[1], k.shape[3]
        elif layout_kv == "PA_BBND":
            block_size, n2 = k.shape[1], k.shape[2]
        elif layout_kv == "PA_BNBD":
            n2, block_size = k.shape[1], k.shape[2]
        else:
            n2, block_size = k.shape[1], k.shape[-1]
    else:
        n2 = k.shape[1] if layout_kv == "TND" else (k.shape[-2] if k.dim() >= 2 else 1)
        block_num = block_size = 0
    if seqused_kv is not None:
        s2 = max(_to_int_list(seqused_kv) or [0])
    else:
        s2 = (
            block_size * (block_num // max(b, 1))
            if "PA" in layout_kv
            else (k.shape[1] if layout_kv in ("BNSD", "BSND") else k.shape[0])
        )
    return b, n1, n2, s1, s2, d, block_num, block_size


def _write_back(tensor, src):
    """把 pytest 生成的 torch tensor 数据写回 TTK 传入的 tensor（可能是 numpy 或 torch）。

    TTK 传入的 numpy tensor 可能是 as_strided 创建的非连续 view。
    不能用 reshape(-1) 赋值——numpy 对非连续 view 的 reshape 返回 copy，
    数据写不回底层 storage。必须用 [...] = 或 np.copyto 直接对 view 赋值。
    """
    if tensor is None or src is None:
        return
    if isinstance(tensor, np.ndarray):
        # numpy: 按 raw bytes 拷贝（处理 float8_e8m0 等自定义 dtype）
        if tensor.dtype == src.dtype:
            src_np = src.numpy() if torch.is_tensor(src) else np.asarray(src)
            tensor[...] = (
                src_np.reshape(tensor.shape) if src_np.size == tensor.size else src_np
            )
        else:
            # dtype 不同: view 为相同 itemsize 的整数类型，再逐元素赋值
            item_bytes = tensor.dtype.itemsize
            if item_bytes == 1:
                dst = tensor.view(np.uint8)
                src_view = (
                    src.view(torch.uint8)
                    if src.dtype == torch.float8_e8m0fnu
                    else src.view(torch.int8)
                    if src.dtype == torch.int8
                    else src.view(torch.uint8)
                )
            elif item_bytes == 2:
                dst = tensor.view(np.int16)
                src_view = (
                    src.view(torch.int16)
                    if src.dtype == torch.bfloat16
                    else src.view(torch.int16)
                )
            elif item_bytes == 4:
                dst = tensor.view(np.uint32)
                src_view = (
                    src.view(torch.uint32)
                    if src.dtype == torch.float32
                    else src.view(torch.int32)
                )
            else:
                dst = tensor.ravel()
                src_view = src.ravel()
            src_np = (
                src_view.numpy() if torch.is_tensor(src_view) else np.asarray(src_view)
            )
            dst[...] = src_np.reshape(dst.shape) if src_np.size == dst.size else src_np
    elif torch.is_tensor(tensor):
        if tensor.dtype == src.dtype:
            tensor.copy_(src)
        elif tensor.dtype == torch.float8_e8m0fnu:
            tensor.view(torch.uint8).copy_(src.view(torch.uint8))
        elif tensor.dtype == torch.bfloat16:
            tensor.view(torch.int16).copy_(src.view(torch.int16))
        else:
            tensor.copy_(src.to(tensor.dtype))


def generate_mixed_quant_inputs(
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
    quant_compute_mode=1,
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
    """In-place customize_inputs hook: 调用 pytest generate_inputs 生成全部输入并写回。"""
    b, n1, n2, s1, s2, d, block_num, block_size = _resolve_axis_layout(
        q, k, layout_q, layout_kv, cu_seqlens_q, seqused_q, seqused_kv
    )

    # 构造 pytest generate_inputs 的 params dict
    seed = int(os.environ.get("MQFA_SEED", "21"))
    # 从 ttk 传入的 input_ranges 解析 q_range/kv_range（CSV input_data_ranges）
    # 顺序: (q_range, k_range, v_range, k_descale_range, v_descale_range, block_table_range, ...)
    input_ranges = kwargs.get("input_ranges") or kwargs.get("input_data_ranges")
    if input_ranges and len(input_ranges) >= 3:
        _q_range = tuple(input_ranges[0])
        _kv_range = tuple(input_ranges[1])
    else:
        _q_range = tuple(
            float(x) for x in os.environ.get("MQFA_Q_RANGE", "-1.0,1.0").split(",")
        )
        _kv_range = tuple(
            float(x) for x in os.environ.get("MQFA_KV_RANGE", "-10.0,10.0").split(",")
        )

    params = {
        "B": batch_size,
        "N1": n1,
        "N2": n2,
        "S1": s1,
        "S2": s2,
        "D": head_dim,
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_attn_out": layout_attn_out,
        "block_size": block_size,
        "block_num": block_num,
        "quant_compute_mode": int(quant_compute_mode),
        "mask_mode": int(mask_mode),
        "win_left": int(win_left),
        "win_right": int(win_right),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_kv": int(max_seqlen_kv),
        "softmax_scale": float(softmax_scale),
        "return_softmax_lse": bool(return_softmax_lse),
        "seqused_q": _to_int_list(seqused_q),
        "seqused_kv": _to_int_list(seqused_kv),
        "cu_seqlens_q": _to_int_list(cu_seqlens_q),
        "q_range": _q_range,
        "kv_range": _kv_range,
        "q_dtype": _to_q_dtype(q),
    }

    # 调用 pytest 的 generate_inputs 生成全部 tensor
    tensors = generate_inputs(params, seed=seed)

    # DEBUG: dump params and tensor shape/stride for cross-check
    import os as _os

    if _os.environ.get("MQFA_DEBUG_INPUTS"):
        import sys as _sys

        print(
            "[DEBUG inputs] params: B=%s N1=%s N2=%s S1=%s S2=%s D=%s block_num=%s block_size=%s"
            % (b, n1, n2, s1, s2, d, block_num, block_size),
            file=_sys.stderr,
            flush=True,
        )
        print(
            "[DEBUG inputs] layout_q=%s layout_kv=%s layout_attn_out=%s"
            % (layout_q, layout_kv, layout_attn_out),
            file=_sys.stderr,
            flush=True,
        )
        for _k in [
            "q",
            "k",
            "v",
            "k_descale",
            "v_descale",
            "block_table",
            "seqused_q",
            "seqused_kv",
            "attn_mask",
            "cu_seqlens_q",
            "sinks",
        ]:
            _t = tensors.get(_k)
            if _t is not None:
                print(
                    "[DEBUG inputs] %s: shape=%s stride=%s dtype=%s"
                    % (_k, tuple(_t.shape), tuple(_t.stride()), _t.dtype),
                    file=_sys.stderr,
                    flush=True,
                )
            else:
                print("[DEBUG inputs] %s: None" % _k, file=_sys.stderr, flush=True)
        # ttk 传入的 tensor shape/stride
        for _name, _t in [
            ("ttk_q", q),
            ("ttk_k", k),
            ("ttk_v", v),
            ("ttk_k_descale", k_descale),
            ("ttk_v_descale", v_descale),
            ("ttk_block_table", block_table),
            ("ttk_seqused_q", seqused_q),
            ("ttk_seqused_kv", seqused_kv),
            ("ttk_attn_mask", attn_mask),
            ("ttk_metadata", metadata),
        ]:
            if _t is not None:
                _sh = tuple(_t.shape) if hasattr(_t, "shape") else "N/A"
                _st = tuple(_t.stride()) if hasattr(_t, "stride") else "N/A"
                _dt = str(getattr(_t, "dtype", "N/A"))
                print(
                    "[DEBUG inputs] %s: shape=%s stride=%s dtype=%s"
                    % (_name, _sh, _st, _dt),
                    file=_sys.stderr,
                    flush=True,
                )
            else:
                print("[DEBUG inputs] %s: None" % _name, file=_sys.stderr, flush=True)

    # 写回 TTK 传入的 tensor
    _write_back(q, tensors.get("q"))
    _write_back(k, tensors.get("k"))
    _write_back(v, tensors.get("v"))
    _write_back(k_descale, tensors.get("k_descale"))
    _write_back(v_descale, tensors.get("v_descale"))
    _write_back(block_table, tensors.get("block_table"))
    _write_back(cu_seqlens_q, tensors.get("cu_seqlens_q"))
    _write_back(seqused_q, tensors.get("seqused_q"))
    _write_back(seqused_kv, tensors.get("seqused_kv"))
    _write_back(sinks, tensors.get("sinks"))
    _write_back(attn_mask, tensors.get("attn_mask"))

    # metadata: pytest gen_data 不生成（设为 None），由 npu.py 在 NPU 端生成。
    # ttk 也需要 metadata，调 pytest 的 metadata op 生成。
    if metadata is not None:
        from core.meta_cpu.mixed_quant_flash_attn_metadata_op import (
            MixedQuantFlashAttnMetadataOp,
        )

        meta_op = MixedQuantFlashAttnMetadataOp()
        _meta_sq = _to_int_list(seqused_q) if seqused_q is not None else None
        _meta_skv = _to_int_list(seqused_kv) if seqused_kv is not None else None
        meta_list = meta_op.npu_mixed_quant_flash_attn_metadata(
            num_heads_q=n1,
            num_heads_kv=n2,
            head_dim=d,
            quant_compute_mode=int(quant_compute_mode),
            batch_size=b if layout_q != "TND" else None,
            cu_seqlens_q=_to_int_list(cu_seqlens_q)
            if cu_seqlens_q is not None
            else None,
            seqused_q=_meta_sq,
            seqused_kv=_meta_skv,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_kv=int(max_seqlen_kv),
            mask_mode=int(mask_mode),
            win_left=int(win_left),
            win_right=int(win_right),
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_attn_out,
        )
        _meta_src = torch.tensor(meta_list, dtype=torch.int32)
        # metadata 实际生成 size 可能 != CSV view_shape，用截断填充（numpy/torch 两条路径都要 min 截断）
        if torch.is_tensor(metadata):
            _meta_dst = metadata.view(torch.int32).reshape(-1)
            _n = min(_meta_dst.numel(), _meta_src.numel())
            _meta_dst[:_n] = _meta_src[:_n]
        else:
            _meta_dst = np.asarray(metadata).view(np.int32).reshape(-1)
            _n = min(_meta_dst.size, _meta_src.numel())
            _meta_dst[:_n] = _meta_src.numpy().reshape(-1)[:_n]
