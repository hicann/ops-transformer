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


"""TTK customize_inputs plugin for quant_flash_mla.

Generates BNSD data via the golden module's generate_data(), converts
to target layout + PA cache, and writes in-place into TTK-allocated slots.
"""

import logging
import math
import os
import sys

import numpy as np
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_ASSETS_DIR, "..")  # assets/
if _GOLDEN_DIR not in sys.path:
    sys.path.insert(0, _GOLDEN_DIR)
# golden 模块依赖 pytest/common/ 下的 result_compare_method
_COMMON_DIR = os.path.join(_ASSETS_DIR, "..", "..", "pytest", "common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

import qfmla_golden as golden_mod

logger = logging.getLogger(__name__)

# Reverse mapping: op layout_kv → golden KV_CACHE_LAYOUT
_KV_LAYOUT_REVERSE = {
    "PA_BNBD": "BnNBsD",
    "PA_BBND": "BnBsND",
    "PA_NZ": "NZ",
}


def generate_qfmla_inputs(
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
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_v=512,
    layout_q="BSND",
    layout_kv="PA_BNBD",
    layout_out="BSND",
    return_softmax_lse=False,
    **kwargs,
):
    """Generate MLA FP8 input data and write into TTK tensor slots in-place.

    Sidecar attributes via **kwargs:
      B, N_q, N_kv, D, D_V, D_rope,
      seqused_q_values, cache_seqlens_values,
      kv_cache_layout (golden-side name, e.g. "BnNBsD"),
      block_size, input_layout, output_layout,
      q_data_range, k_data_range, v_data_range,
      seed_q, seed_k, seed_v, seed_qr, seed_kr,
      p_scale, graph_path
    """
    # ---- 1. Read sidecar attributes ----
    B = kwargs.get("B", 1)
    N_q = kwargs.get("N_q", 6)
    N_kv = kwargs.get("N_kv", 1)
    D = kwargs.get("D", 512)
    D_V = kwargs.get("D_V", 512)
    D_rope = kwargs.get("D_rope", 64)
    block_size = kwargs.get("block_size", 128)
    seqused_q_vals = kwargs.get("seqused_q_values", [1])
    cache_seqlens_vals = kwargs.get("cache_seqlens_values", [1024])
    kv_cache_layout = kwargs.get("kv_cache_layout") or _KV_LAYOUT_REVERSE.get(
        layout_kv, "BnNBsD"
    )
    input_layout = kwargs.get("input_layout", layout_q)
    output_layout = kwargs.get("output_layout", layout_out)
    p_scale_val = kwargs.get("p_scale", 1.0)
    graph_path = kwargs.get("graph_path", 0)

    # 异常用例容忍: N_q/N_kv/D 可能被注入非法值 (如 -1)。非法值不能用于数据合成
    # (golden generate_data 会裸崩, 掩盖真正要验证的异常属性), 按 CSV slot 形状
    # 推导回退值完成数据生成; 原始非法属性仍由 ttk 透传给 metadata/主算子, 由
    # 算子 host 侧校验拦截并给出明确报错 (如 "numHeadsQ must be greater than 0")。
    def _positive_or_derive(name, value, slot, axis):
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            ivalue = 0
        if ivalue > 0:
            return ivalue
        shape = tuple(int(dim) for dim in slot.shape) if hasattr(slot, "shape") else ()
        idx = axis if axis >= 0 else len(shape) + axis
        if not 0 <= idx < len(shape) or shape[idx] <= 0:
            raise ValueError(
                f"[INPUTS] attribute {name}={value!r} invalid (must be positive "
                f"integer) and cannot be derived from CSV slot shape {shape}"
            )
        logger.warning(
            "[INPUTS] attribute %s=%r invalid (must be positive integer) - derive "
            "%s=%d from CSV slot shape %s for data synthesis; the invalid value is "
            "still passed through to the op for interception",
            name,
            value,
            name,
            shape[idx],
            shape,
        )
        return shape[idx]

    # q slot head axis: BSND → axis 2, TND → axis 1
    _q_head_axis = 2 if str(layout_q).upper() == "BSND" else 1
    # k_cache slot head axis: PA_BNBD/NZ → axis 1, PA_BBND → axis 2
    _k_head_axis = 2 if str(layout_kv).upper() == "PA_BBND" else 1
    N_q = _positive_or_derive("N_q", N_q, q, _q_head_axis)
    N_kv = _positive_or_derive("N_kv", N_kv, k_cache, _k_head_axis)
    # D: q slot 最后一维是 D_total = D + D_rope; 推导出 D_total 后减去 D_rope 得 D
    D_total = _positive_or_derive("D", int(D) + int(D_rope), q, -1)
    D = D_total - int(D_rope)

    logger.info(
        "[INPUTS] B=%s N_q=%s N_kv=%s D=%s D_rope=%s layout_q=%s kv_layout=%s block_size=%s",
        B,
        N_q,
        N_kv,
        D,
        D_rope,
        layout_q,
        kv_cache_layout,
        block_size,
    )

    # ---- 2. Set golden module globals ----
    golden_mod.B = B
    golden_mod.N_q = N_q
    golden_mod.N_kv = N_kv
    golden_mod.D = D
    golden_mod.D_V = D_V
    golden_mod.D_rope = D_rope
    golden_mod.QFMLA_QK_HEAD_DIM = D + D_rope
    golden_mod.QFMLA_V_HEAD_DIM = D_V
    golden_mod.QFMLA_QUANT_MODE = quant_mode
    golden_mod.SEQUSED_Q = list(seqused_q_vals)
    golden_mod.CACHE_SEQLENS = list(cache_seqlens_vals)
    golden_mod.ENABLE_PA = True  # MLA always uses PA
    golden_mod.ENABLE_LSE = return_softmax_lse
    golden_mod.GOLDEN_MODE = True
    golden_mod.BLOCK_SIZE = block_size
    golden_mod.MASK_MODE = mask_mode
    golden_mod.INPUT_LAYOUT = input_layout
    golden_mod.OUTPUT_LAYOUT = output_layout
    golden_mod.KV_CACHE_LAYOUT = kv_cache_layout
    golden_mod.P_SCALE = float(p_scale_val)
    golden_mod.GRAPH_PATH = graph_path
    golden_mod.NUM_BLOCKS = kwargs.get("num_blocks", 0)
    golden_mod.SCALE_VALUE = softmax_scale
    golden_mod.MAX_SEQLEN_Q = max_seqlen_q
    golden_mod.MAX_SEQLEN_KV = max_seqlen_kv

    # Data ranges
    if kwargs.get("q_data_range"):
        golden_mod.Q_DATA_RANGE = tuple(kwargs["q_data_range"])
    if kwargs.get("k_data_range"):
        golden_mod.K_DATA_RANGE = tuple(kwargs["k_data_range"])
    if kwargs.get("v_data_range"):
        golden_mod.V_DATA_RANGE = tuple(kwargs["v_data_range"])

    # Seeds
    for s in ("SEED_Q", "SEED_K", "SEED_V", "SEED_QR", "SEED_KR", "SEED_BLOCK_TABLE"):
        key = s.lower()
        if key in kwargs:
            setattr(golden_mod, s, kwargs[key])

    # ---- 3. Generate BNSD data ----
    (q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale_t, qr_bf16, kr_bf16) = (
        golden_mod.generate_data(device="cpu")
    )

    logger.info(
        "[INPUTS] generated: q=%s k=%s deq_q=%s deq_k=%s",
        tuple(q_fp8.shape),
        tuple(k_fp8.shape),
        tuple(deq_q.shape),
        tuple(deq_k.shape),
    )

    # ---- 4. Convert Q to target layout ----
    q_final = golden_mod.convert_q_bnsd_to_layout(q_fp8, seqused_q_vals, input_layout)

    # ---- 5. Convert K to PA cache ----
    block_table_t = golden_mod.create_block_table(
        cache_seqlens_vals, block_size, num_blocks=golden_mod.NUM_BLOCKS
    )

    k_pa = golden_mod.bnsd_to_k_cache(
        k_fp8,
        cache_seqlens_vals,
        block_size,
        block_table_t,
        num_blocks=golden_mod.NUM_BLOCKS,
        kv_layout=kv_cache_layout,
    )

    # ---- 6. Prepare descale ----
    # q_descale: use golden's convert_scale_to_layout (reads INPUT_LAYOUT global)
    deq_q_final = golden_mod.convert_scale_to_layout(deq_q, seqused_q_vals, "deq_q")
    # k_descale: scalar (1,)
    deq_k_final = deq_k.float().reshape(-1)[:1].contiguous()

    # ---- 7. Write to tensor slots in-place ----
    def _inplace_write(slot, src, name):
        """形状一致时原位 bit-exact copy_; 不一致时 WARN 并跳过写入。

        异常用例的 CSV slot 形状常与按属性推导的计算结果不一致。裸 copy_ 的
        broadcast 报错会掩盖真正注入的异常属性, 且无法定位。跳过写入让非法输入
        按 CSV 声明的形状透传到算子, 由 host 侧 checker 给出明确拦截报错。
        注: ttk 在算子调用前会从 numpy 底层存储重建 runtime tensor, 插件侧
        resize_/set_ 改形状不会传导, 只有共享内存的原位数据写入有效。
        """
        if slot is None:
            return
        if not isinstance(src, torch.Tensor):
            src = torch.as_tensor(src)
        if isinstance(slot, torch.Tensor):
            if slot.numel() == 0:
                return
            if tuple(slot.shape) != tuple(src.shape):
                logger.warning(
                    "[INPUTS] %s shape mismatch: CSV slot %s != computed %s - skip "
                    "in-place write (keep CSV slot data/shape), let op-side "
                    "validation intercept the inconsistent case",
                    name,
                    tuple(slot.shape),
                    tuple(src.shape),
                )
                return
            slot.copy_(src.to(slot.dtype))
        else:
            arr = np.asarray(slot)
            if arr.size == 0:
                return
            if "float8" in str(src.dtype):
                src_np = src.view(torch.uint8).cpu().numpy()
            else:
                src_np = src.cpu().numpy()
            if tuple(arr.shape) != tuple(src_np.shape):
                logger.warning(
                    "[INPUTS] %s shape mismatch: CSV slot %s != computed %s - skip "
                    "in-place write (keep CSV slot data/shape), let op-side "
                    "validation intercept the inconsistent case",
                    name,
                    tuple(arr.shape),
                    tuple(src_np.shape),
                )
                return
            arr[...] = src_np

    _inplace_write(q, q_final, "q")
    _inplace_write(k_cache, k_pa, "k_cache")
    _inplace_write(q_descale, deq_q_final, "q_descale")
    _inplace_write(k_descale, deq_k_final, "k_descale")
    _inplace_write(block_table, block_table_t, "block_table")
    _inplace_write(
        cache_seqlens,
        torch.tensor(cache_seqlens_vals, dtype=torch.int32),
        "cache_seqlens",
    )

    # seqused_q slot (kw_only tensor)
    _inplace_write(
        seqused_q, torch.tensor(seqused_q_vals, dtype=torch.int32), "seqused_q"
    )

    # cu_seqlens_q slot (only for TND layout)
    if cu_seqlens_q is not None and layout_q == "TND":
        cu_vals = [0]
        for s in seqused_q_vals:
            cu_vals.append(cu_vals[-1] + s)
        _inplace_write(
            cu_seqlens_q, torch.tensor(cu_vals, dtype=torch.int32), "cu_seqlens_q"
        )

    # attn_mask slot (only for causal mask_mode=3)
    if attn_mask is not None and mask_mode == 3:
        mask_shape = (2048, 2048)
        causal_mask = torch.triu(torch.ones(mask_shape, dtype=torch.int8), diagonal=1)
        _inplace_write(attn_mask, causal_mask, "attn_mask")

    logger.info(
        "[INPUTS] in-place writes complete: q=%s k_cache=%s bt=%s",
        tuple(q.shape) if hasattr(q, "shape") else "?",
        tuple(k_cache.shape) if hasattr(k_cache, "shape") else "?",
        tuple(block_table_t.shape),
    )
