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

Generates BNSD data with explicit per-case parameters, converts
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
    """Generate MLA FP8 input data and write into TTK tensor slots in-place.

    Explicit *_values override vector contents; failed vector writes are errors.
    Actual tensor slots determine core dimensions, physical pages and block size.
    input_ranges takes precedence over legacy q_data_range/k_data_range.
    seed_q/seed_k/seed_block_table affect synthesis only. Metadata sidecars are
    forwarded separately by npu_preprocess and are not main-op shape arguments.
    """
    mismatch_policy = kwargs.get("input_mismatch_policy", "passthrough")
    if mismatch_policy not in ("passthrough", "error"):
        raise ValueError("input_mismatch_policy must be 'passthrough' or 'error'")
    skipped_slots = []

    # Shapes and explicitly supplied vectors describe the payload to synthesize.
    # This generator does not invent safe replacements for malformed sequences.
    # Pure validation cases that cannot be synthesized must supply raw/manual data.
    seqused_q_vals = kwargs.get("seqused_q_values")
    cache_seqlens_vals = kwargs.get("cache_seqlens_values")
    cu_seqlens_q_vals = kwargs.get("cu_seqlens_q_values")

    def _values(slot, explicit):
        if explicit is not None:
            return list(explicit)
        if slot is None:
            return None
        if isinstance(slot, torch.Tensor):
            return slot.detach().cpu().tolist() if slot.numel() else None
        arr = np.asarray(slot)
        return arr.tolist() if arr.size else None

    cache_lengths = _values(cache_seqlens, cache_seqlens_vals)
    cu_values = _values(cu_seqlens_q, cu_seqlens_q_vals)
    seq_values = _values(seqused_q, seqused_q_vals)
    if not cache_lengths or any(
        not isinstance(x, (int, np.integer)) or x < 0 for x in cache_lengths
    ):
        raise ValueError(
            "[INPUTS] cannot synthesize cache data without nonnegative batch lengths"
        )
    B = len(cache_lengths)
    synthesis_cache_seqlens_vals = cache_lengths
    synthesis_input_layout = layout_q
    kv_cache_layout = _KV_LAYOUT_REVERSE.get(layout_kv)
    if kv_cache_layout is None:
        raise ValueError(
            f"[INPUTS] cannot synthesize cache layout {layout_kv!r}; use raw inputs for layout negatives"
        )
    if layout_q not in ("TND", "BSND", "BNSD"):
        raise ValueError(
            f"[INPUTS] cannot synthesize query layout {layout_q!r}; use raw inputs for layout negatives"
        )
    q_shape = tuple(int(dim) for dim in q.shape)
    k_shape = tuple(int(dim) for dim in k_cache.shape)
    q_head_axis = 2 if layout_q == "BSND" else 1
    k_head_axis = 2 if layout_kv == "PA_BBND" else 1
    # Main-op dimensions come from its tensor slots. Metadata sidecar attributes
    # remain untouched, but are not independent main-op dimension arguments.
    N_q, N_kv, D_total = q_shape[q_head_axis], k_shape[k_head_axis], q_shape[-1]
    D_V = 512 if head_dim_v is None else head_dim_v
    block_axis = 3 if layout_kv == "PA_NZ" else (1 if layout_kv == "PA_BBND" else 2)
    block_size = k_shape[block_axis]
    synthesis_block_size = block_size
    if block_size <= 0:
        raise ValueError("[INPUTS] cannot synthesize pages with an empty block axis")
    padding_cu = None
    if layout_q == "TND":
        if cu_values is not None:
            if len(cu_values) != B + 1:
                raise ValueError("[INPUTS] cu vector cannot describe the cache batches")
            synthesis_seqused_q_vals = [b - a for a, b in zip(cu_values, cu_values[1:])]
            padding_cu = cu_values
        elif seq_values is not None:
            synthesis_seqused_q_vals = seq_values
        else:
            raise ValueError(
                "[INPUTS] TND synthesis needs actual cu offsets or explicit query lengths"
            )
    else:
        synthesis_seqused_q_vals = [q_shape[1 if layout_q == "BSND" else 2]] * B
    if len(synthesis_seqused_q_vals) != B or any(
        not isinstance(x, (int, np.integer)) or x < 0 for x in synthesis_seqused_q_vals
    ):
        raise ValueError(
            "[INPUTS] cannot allocate query data from the supplied lengths"
        )
    p_scale_val = 1.0
    logger.info(
        "[INPUTS] B=%s N_q=%s N_kv=%s D=%s D_V=%s layout_q=%s kv_layout=%s block_size=%s",
        B,
        N_q,
        N_kv,
        D_total,
        D_V,
        layout_q,
        kv_cache_layout,
        block_size,
    )

    # Resolve every generation setting in this call, like QFA inputs.py.
    # input_ranges is supplied by TTK from CSV input_data_ranges; slot 2 is
    # q_descale, not a separate V input.
    input_ranges = kwargs.get("input_ranges")

    def _range_of(index, legacy):
        value = (
            input_ranges[index]
            if input_ranges is not None and index < len(input_ranges)
            else None
        )
        if value is None:
            value = kwargs.get(legacy)
        return (-1.0, 1.0) if value is None else (float(value[0]), float(value[1]))

    q_range = _range_of(0, "q_data_range")
    k_range = _range_of(1, "k_data_range")
    seed_q = 54 if kwargs.get("seed_q") is None else kwargs["seed_q"]
    seed_k = 3 if kwargs.get("seed_k") is None else kwargs["seed_k"]
    seed_block_table = (
        1234 if kwargs.get("seed_block_table") is None else kwargs["seed_block_table"]
    )
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
    num_blocks = k_slot_blocks

    # ---- 3. Generate BNSD data ----
    (q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale_t, qr_bf16, kr_bf16) = (
        golden_mod.generate_case_data(
            batch_size=B,
            num_heads_q=N_q,
            num_heads_kv=N_kv,
            head_dim_qk=D_total,
            head_dim_v=D_V,
            seqused_q=synthesis_seqused_q_vals,
            cache_seqlens=synthesis_cache_seqlens_vals,
            quant_mode=quant_mode,
            q_data_range=q_range,
            k_data_range=k_range,
            seed_q=seed_q,
            seed_k=seed_k,
            p_scale=float(p_scale_val),
            device="cpu",
        )
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
        q_fp8,
        synthesis_seqused_q_vals,
        synthesis_input_layout,
        cu_seqlens=padding_cu if synthesis_input_layout == "TND" else None,
    )

    # ---- 5. Convert K to PA cache ----
    # 槽位宽度与 QFA 对齐: 取 CSV block_table 声明的第二维 (每 batch 最大块槽位),
    # 与物理块数 NUM_BLOCKS 解耦, 支持 num_blocks > 第二维 的用例。
    _bt_slot_shape = (
        tuple(int(d) for d in block_table.shape)
        if hasattr(block_table, "shape")
        else ()
    )
    block_table_t = golden_mod.create_block_table(
        synthesis_cache_seqlens_vals,
        synthesis_block_size,
        num_blocks=num_blocks,
        seed=seed_block_table,
        max_blocks=_bt_slot_shape[1] if len(_bt_slot_shape) >= 2 else None,
    )

    k_pa = golden_mod.bnsd_to_k_cache(
        k_fp8,
        synthesis_cache_seqlens_vals,
        synthesis_block_size,
        block_table_t,
        num_blocks=num_blocks,
        kv_layout=kv_cache_layout,
    )

    # ---- 6. Prepare descale ----
    # q_descale layout is explicit; do not depend on the reference module state.
    deq_q_final = golden_mod.convert_scale_to_layout(
        deq_q,
        synthesis_seqused_q_vals,
        "deq_q",
        input_layout=synthesis_input_layout,
        cu_seqlens=padding_cu if synthesis_input_layout == "TND" else None,
    )
    # k_descale: scalar (1,)
    deq_k_final = deq_k.float().reshape(-1)[:1].contiguous()

    # ---- 7. Write to tensor slots in-place ----
    def _inplace_write(slot, src, name, *, strict=False):
        """Core slots may preserve a shape mismatch; explicit vectors never do.

        Core shape mismatches preserve the existing slot, like QFA. That does
        not prove any requested sidecar value reached the operator. Explicit
        vector/mask requests use strict writes and fail if they cannot be copied.
        注: ttk 在算子调用前会从 numpy 底层存储重建 runtime tensor, 插件侧
        resize_/set_ 改形状不会传导, 只有共享内存的原位数据写入有效。
        """
        if slot is None or src is None:
            return
        if not isinstance(src, torch.Tensor):
            src = torch.as_tensor(src)
        if isinstance(slot, torch.Tensor):
            if slot.numel() == 0 and not strict:
                return
            if tuple(slot.shape) != tuple(src.shape):
                if strict or mismatch_policy == "error":
                    raise ValueError(
                        f"[INPUTS] {name} slot {tuple(slot.shape)} != generated {tuple(src.shape)}"
                    )
                skipped_slots.append(name)
                logger.warning(
                    "[INPUTS] %s shape mismatch: CSV slot %s != computed %s - skip "
                    "in-place write (keep CSV slot data/shape), let op-side "
                    "validation intercept the inconsistent case",
                    name,
                    tuple(slot.shape),
                    tuple(src.shape),
                )
                return
            if (
                name in ("q", "k_cache")
                and src.element_size() == slot.element_size() == 1
            ):
                slot.view(torch.uint8).copy_(src.view(torch.uint8))
            else:
                slot.copy_(src.to(slot.dtype))
        else:
            arr = np.asarray(slot)
            if arr.size == 0 and not strict:
                return
            if "float8" in str(src.dtype):
                src_np = src.view(torch.uint8).cpu().numpy()
            else:
                src_np = src.cpu().numpy()
            if tuple(arr.shape) != tuple(src_np.shape):
                if strict or mismatch_policy == "error":
                    raise ValueError(
                        f"[INPUTS] {name} slot {tuple(arr.shape)} != generated {tuple(src_np.shape)}"
                    )
                skipped_slots.append(name)
                logger.warning(
                    "[INPUTS] %s shape mismatch: CSV slot %s != computed %s - skip "
                    "in-place write (keep CSV slot data/shape), let op-side "
                    "validation intercept the inconsistent case",
                    name,
                    tuple(arr.shape),
                    tuple(src_np.shape),
                )
                return
            if (
                name in ("q", "k_cache")
                and src.element_size() == arr.dtype.itemsize == 1
            ):
                arr.view(np.uint8)[...] = src.view(torch.uint8).cpu().numpy()
            else:
                arr[...] = src_np

    _inplace_write(q, q_final, "q")
    _inplace_write(k_cache, k_pa, "k_cache")
    _inplace_write(q_descale, deq_q_final, "q_descale")
    _inplace_write(k_descale, deq_k_final, "k_descale")
    _inplace_write(block_table, block_table_t, "block_table")
    # Only explicit vector overrides are written. A failed write is a generation
    # error, never a claim that the requested negative value reached the operator.
    for slot, values, name in (
        (cache_seqlens, cache_seqlens_vals, "cache_seqlens"),
        (seqused_q, seqused_q_vals, "seqused_q"),
        (cu_seqlens_q, cu_seqlens_q_vals, "cu_seqlens_q"),
    ):
        if values is not None:
            _inplace_write(slot, torch.as_tensor(values), name, strict=True)

    if attn_mask is not None and mask_mode == 3:
        mask_shape = kwargs.get("attn_mask_shape")
        mask_shape = (2048, 2048) if mask_shape is None else tuple(mask_shape)
        causal_mask = torch.triu(torch.ones(mask_shape, dtype=torch.int8), diagonal=1)
        _inplace_write(attn_mask, causal_mask, "attn_mask", strict=True)

    if skipped_slots:
        logger.warning(
            "[INPUTS] kept original slots under passthrough policy: %s", skipped_slots
        )
    logger.info(
        "[INPUTS] in-place writes complete: q=%s k_cache=%s bt=%s",
        tuple(q.shape) if hasattr(q, "shape") else "?",
        tuple(k_cache.shape) if hasattr(k_cache, "shape") else "?",
        tuple(block_table_t.shape),
    )
