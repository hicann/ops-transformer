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

import importlib.util
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
    layout_q="TND",
    layout_kv="PA_BNBD",
    layout_out="TND",
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
    # Match QFA's plugin/no-plugin layering for structural empty-tensor cases.
    # With the plugin enabled, custom input generation cannot synthesize a
    # meaningful payload for an empty core tensor, so fail in INPUT_GEN.  With
    # the plugin disabled this hook is bypassed and the original CSV tensor is
    # passed to the Torch/host validation path.  Value-only anomalies (for
    # example negative cache_seqlens/seqused_q) are deliberately not rejected
    # here; they still use safe synthesis data and reach metadata validation.
    for name, tensor in (("q", q), ("k_cache", k_cache)):
        shape = tuple(int(dim) for dim in tensor.shape)
        if any(dim == 0 for dim in shape):
            raise ValueError(
                f"[INPUTS] {name} has an empty dimension in shape {shape}; "
                "custom input generation does not support empty core tensors"
            )

    # ---- 1. Read sidecar attributes ----
    N_q = kwargs.get("N_q", 6)
    N_kv = kwargs.get("N_kv", 1)
    D = kwargs.get("D", 512)
    D_V = kwargs.get("D_V", 512)
    D_rope = kwargs.get("D_rope", 64)
    block_size = kwargs.get("block_size", 128)
    # Keep the raw values for the final in-place write.  Negative/value-error
    # cases must reach metadata unchanged, but those values cannot be used to
    # allocate the CPU tensors that provide the otherwise-valid test payload.
    seqused_q_vals = list(kwargs.get("seqused_q_values", [1]))
    cache_seqlens_vals = list(kwargs.get("cache_seqlens_values", [1024]))
    cu_seqlens_q_vals = kwargs.get("cu_seqlens_q_values")
    if cu_seqlens_q_vals is not None:
        cu_seqlens_q_vals = list(cu_seqlens_q_vals)
    kv_cache_layout = kwargs.get("kv_cache_layout") or _KV_LAYOUT_REVERSE.get(
        layout_kv, "BnNBsD"
    )
    input_layout = kwargs.get("input_layout", layout_q)
    output_layout = kwargs.get("output_layout", layout_out)
    p_scale_val = kwargs.get("p_scale", 1.0)
    graph_path = kwargs.get("graph_path", 0)

    # Only use the allocated cache_seqlens slot to choose the synthesis batch.
    # Invalid cu_seqlens_q shapes/values must not change the otherwise-valid
    # payload shape, otherwise input generation fails before metadata sees the
    # malformed tensor.
    cache_shape = tuple(int(dim) for dim in cache_seqlens.shape)
    if len(cache_shape) == 1 and cache_shape[0] > 0:
        B = cache_shape[0]
    elif cache_seqlens_vals:
        B = max(1, len(cache_seqlens_vals))
    else:
        B = max(1, int(kwargs.get("B", 1)))

    # QMLA only supports TND.  Use TND and a positive block size for payload
    # synthesis while preserving the raw layout_q/block_size in the actual
    # operator arguments and tensor slots.
    synthesis_input_layout = "TND"
    synthesis_output_layout = "TND"
    synthesis_block_size = int(block_size)
    if synthesis_block_size <= 0:
        synthesis_block_size = 128

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
    # D: q slot 末维是真实的 D_total (CSV slot 配置); CSV 的 D 属性语义不统一
    # (有些 case D=D_nope=512, 有些 case D=D_total=576), 不能盲目 D+D_rope。
    # 以 q slot 末维为准反推 D_nope, 不一致时以 q slot 为准 (CSV slot 是算子
    # 实际接收的形状, 是真实 QK head_dim); 原 CSV 的 D 属性仍由 ttk 透传给
    # metadata/主算子, 不影响算子侧拦截。
    _q_last_dim = int(q.shape[-1]) if (hasattr(q, "shape") and q.dim() > 0) else 0
    if _q_last_dim > 0:
        D_total = _q_last_dim
    else:
        D_total = _positive_or_derive("D", int(D) + int(D_rope), q, -1)
    if int(D) + int(D_rope) != D_total:
        logger.warning(
            "[INPUTS] CSV D=%d + D_rope=%d = %d != q slot last dim=%d; "
            "CSV 'D' likely uses D_total convention (D_total=%d, D_nope=%d) - "
            "using q slot dim as D_total for data synthesis",
            int(D),
            int(D_rope),
            int(D) + int(D_rope),
            D_total,
            D_total,
            D_total - int(D_rope),
        )
    D = D_total - int(D_rope)

    # Synthesis-only sequence lengths. Prefer a valid cumulative sequence from
    # the case, otherwise derive a positive partition from the CSV q slot. The
    # raw seqused_q/cu_seqlens_q values are still written below.
    synthesis_seqused_q_vals = None
    if (
        cu_seqlens_q_vals is not None
        and len(cu_seqlens_q_vals) == int(B) + 1
        and cu_seqlens_q_vals
        and cu_seqlens_q_vals[0] == 0
        and all(
            cu_seqlens_q_vals[i + 1] >= cu_seqlens_q_vals[i]
            for i in range(len(cu_seqlens_q_vals) - 1)
        )
    ):
        candidate = [
            cu_seqlens_q_vals[i + 1] - cu_seqlens_q_vals[i] for i in range(int(B))
        ]
        if candidate and max(candidate) > 0:
            synthesis_seqused_q_vals = candidate
    if synthesis_seqused_q_vals is None:
        q_shape = tuple(int(dim) for dim in q.shape) if hasattr(q, "shape") else ()
        batch_for_synthesis = max(int(B), 1)
        total_q = (
            q_shape[0] if len(q_shape) == 3 and q_shape[0] > 0 else batch_for_synthesis
        )
        base, extra = divmod(total_q, batch_for_synthesis)
        synthesis_seqused_q_vals = [
            base + (1 if i < extra else 0) for i in range(batch_for_synthesis)
        ]
        if max(synthesis_seqused_q_vals) <= 0:
            synthesis_seqused_q_vals = [1] * batch_for_synthesis

    synthesis_cache_seqlens_vals = cache_seqlens_vals
    if (
        not synthesis_cache_seqlens_vals
        or len(synthesis_cache_seqlens_vals) != int(B)
        or any(int(v) <= 0 for v in synthesis_cache_seqlens_vals)
    ):
        synthesis_cache_seqlens_vals = [1] * max(int(B), 1)

    # Invalid D/D_rope values are operator attributes and/or reflected by the
    # CSV slot shape. Use the supported dimensions only for payload synthesis;
    # shape mismatches are deliberately skipped by _inplace_write below.
    if D <= 0 or int(D_rope) <= 0:
        D = 512
        D_rope = 64

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
    golden_mod.SEQUSED_Q = list(synthesis_seqused_q_vals)
    golden_mod.CACHE_SEQLENS = list(synthesis_cache_seqlens_vals)
    golden_mod.ENABLE_PA = True  # MLA always uses PA
    golden_mod.ENABLE_LSE = return_softmax_lse
    golden_mod.GOLDEN_MODE = True
    golden_mod.BLOCK_SIZE = synthesis_block_size
    golden_mod.MASK_MODE = mask_mode
    golden_mod.INPUT_LAYOUT = synthesis_input_layout
    golden_mod.OUTPUT_LAYOUT = synthesis_output_layout
    golden_mod.KV_CACHE_LAYOUT = kv_cache_layout
    golden_mod.P_SCALE = float(p_scale_val)
    golden_mod.GRAPH_PATH = graph_path
    # num_blocks: 从 k_cache slot 首维推导物理 block 数 (对齐 QFA inputs.py 的
    # `total_blocks = int(dequant_scale_k.shape[0])` 风格)。CSV 无 num_blocks 字段,
    # 用 kwargs.get("num_blocks", 0) 会恒为 0, 导致 create_block_table 生成空
    # block_table → 算子侧 "max_seqlen_kv must not exceed block_table capacity"
    # 拦截。k_cache slot 首维是算子实际接收的物理 block 数, 是真实容量来源。
    k_slot_blocks = (
        int(k_cache.shape[0])
        if (hasattr(k_cache, "shape") and k_cache.dim() >= 1)
        else 0
    )
    num_blocks = (
        int(kwargs.get("num_blocks", 0))
        if int(kwargs.get("num_blocks", 0)) > 0
        else k_slot_blocks
    )
    golden_mod.NUM_BLOCKS = num_blocks
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
    q_final = golden_mod.convert_q_bnsd_to_layout(
        q_fp8, synthesis_seqused_q_vals, synthesis_input_layout
    )

    # ---- 5. Convert K to PA cache ----
    block_table_t = golden_mod.create_block_table(
        synthesis_cache_seqlens_vals,
        synthesis_block_size,
        num_blocks=golden_mod.NUM_BLOCKS,
    )

    k_pa = golden_mod.bnsd_to_k_cache(
        k_fp8,
        synthesis_cache_seqlens_vals,
        synthesis_block_size,
        block_table_t,
        num_blocks=golden_mod.NUM_BLOCKS,
        kv_layout=kv_cache_layout,
    )

    # ---- 6. Prepare descale ----
    # q_descale: use golden's convert_scale_to_layout (reads INPUT_LAYOUT global)
    deq_q_final = golden_mod.convert_scale_to_layout(
        deq_q, synthesis_seqused_q_vals, "deq_q"
    )
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
        cu_vals = cu_seqlens_q_vals
        if cu_vals is None:
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
