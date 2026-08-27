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

import logging
import math
import os
import sys
from typing import List

import numpy
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import quant_flash_attn_golden as mxfp8_golden_mod
import quant_flash_attn_fp8_golden as fp8_golden_mod

logger = logging.getLogger(__name__)

__input__ = {"e2e": {"qfa_wrapper.npu_qfa": "generate_qfa_mxfp8_inputs"}}

_SEED_MAP = {"q": 54, "k": 3, "v": 4}


def _write_int32_list(slot, values, slot_name):
    """把 attributes 里的 int 值列表写进 CSV tensor slot（覆盖 ttk 随机生成）。

    空 slot（shape (0,)，无数据）或 values 为 None 时跳过。
    dst 可为 numpy 或 CPU torch tensor（ttk 经 backend.inputs_from_numpy 传入）。

    按 slot 的目标 dtype 写入（不硬编码 int32）——保留 CSV tensor_dtypes。
    正常用例 slot 是 int32；异常用例若把 cu_seqlens/seqused 设为 int8，
    此处用 int8 写（值可能溢出并抛错，或被 AICPU 校验拦截），tensor_dtypes 真正生效。
    """
    if slot is None:
        return
    if values is None:
        return
    # 空 slot（shape (0,)，无数据）→ 跳过，不覆盖（PA 布局下 cu_seqlens/seqused 可能
    # 留空，属性 value 是无意义的残留值）。
    if isinstance(slot, torch.Tensor):
        if slot.numel() == 0:
            return
    elif numpy.asarray(slot).size == 0:
        return
    if isinstance(slot, torch.Tensor):
        arr = torch.as_tensor(list(values), dtype=slot.dtype)
        if tuple(slot.shape) != tuple(arr.shape):
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: CSV slot {tuple(slot.shape)} "
                f"!= attribute value {tuple(arr.shape)} ({values!r}). "
                f"Check Excel shape vs cu_seqlens/seqused value."
            )
        slot.copy_(arr)
    else:
        target_dtype = numpy.asarray(slot).dtype
        arr = numpy.array(list(values), dtype=target_dtype)
        if tuple(slot.shape) != tuple(arr.shape):
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: CSV slot {tuple(slot.shape)} "
                f"!= attribute value {tuple(arr.shape)} ({values!r}). "
                f"Check Excel shape vs cu_seqlens/seqused value."
            )
        slot[...] = arr


def generate_qfa_mxfp8_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    p_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q_t: torch.Tensor,
    cu_seqlens_kv_t: torch.Tensor,
    seqused_q_t: torch.Tensor,
    seqused_kv_t: torch.Tensor,
    sinks_t: torch.Tensor,
    attn_mask_t: torch.Tensor,
    metadata_t: torch.Tensor,
    *,
    batch_size: int,
    N_q: int,
    N_kv: int,
    D: int,
    cu_seqlens_q: List[int],
    cu_seqlens_kv: List[int],
    seqused_q: List[int],
    seqused_kv: List[int],
    max_seqlen_q: int,
    max_seqlen_kv: int,
    enable_pa: bool,
    kv_cache_layout: str,
    block_size: int,
    mask_mode: int,
    q_scale_layout: str,
    quant_mode: int = 1,
    enable_lse: bool = False,
    graph_path: int = 0,
    input_layout: str = "TND",
    is_contiguous: bool = True,
    device_id: int = 0,
    softmax_scale: float = None,
    data_range_q: float = 1.0,
    data_range_k: float = 1.0,
    data_range_v: float = 1.0,
    layout_q: str = "TND",
    layout_q_descale: str = "TND",
    layout_kv: str = "TND",
    layout_out: str = "TND",
    **kwargs,
):
    # cu_seqlens -> actual_seq (差分还原)
    cu_seqlens_q = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]
    actual_seq_q = (
        [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]
        if len(cu_seqlens_q) > 1
        else [0]
    )
    if len(cu_seqlens_kv) > 1:
        actual_seq_kv = [
            cu_seqlens_kv[i + 1] - cu_seqlens_kv[i]
            for i in range(len(cu_seqlens_kv) - 1)
        ]
    elif seqused_kv is not None and len(seqused_kv) > 0:
        actual_seq_kv = list(seqused_kv)
    else:
        actual_seq_kv = [0]

    # batch_size: CSV 原始值 (可为 -1), 透传给 metadata;
    # B: 从 cu_seqlens_q 推导的正整数, 供 BNSD 张量生成。
    B = max(1, len(cu_seqlens_q) - 1) if cu_seqlens_q and len(cu_seqlens_q) >= 2 else 1

    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D

    for gkey, gval in [
        ("B", B),
        ("N_q", N_q),
        ("N_kv", N_kv),
        ("D", D),
        ("FP8_DTYPE", torch.float8_e4m3fn),
        ("QUANT_GROUP_SIZE", 32),
        ("CU_SEQLENS_Q", cu_seqlens_q),
        ("CU_SEQLENS_KV", cu_seqlens_kv if cu_seqlens_kv else None),
        ("SEQUSED_Q", list(seqused_q) if seqused_q is not None else None),
        ("SEQUSED_KV", list(seqused_kv) if seqused_kv is not None else None),
        ("MAX_SEQLEN_Q", max_seqlen_q),
        ("MAX_SEQLEN_KV", max_seqlen_kv),
        ("ENABLE_PA", enable_pa),
        ("KV_CACHE_LAYOUT", kv_cache_layout),
        ("BLOCK_SIZE", block_size),
        ("Q_SCALE_LAYOUT", q_scale_layout),
        ("INPUT_LAYOUT", input_layout),
        ("LAYOUT_Q", layout_q),
        ("LAYOUT_Q_DESCALE", layout_q_descale),
        ("LAYOUT_KV", layout_kv),
        ("LAYOUT_OUT", layout_out),
    ]:
        setattr(mxfp8_golden_mod, gkey, gval)

    # csv input_data_ranges 列由 ttk 作为 input_ranges kwarg 传入
    input_ranges = kwargs.get("input_ranges")

    def _range_of(idx, default=(-1.0, 1.0)):
        if input_ranges is None or idx >= len(input_ranges):
            return default
        r = input_ranges[idx]
        if r is None:
            return default
        return (float(r[0]), float(r[1]))

    rq_min, rq_max = _range_of(0)
    rk_min, rk_max = _range_of(1)
    rv_min, rv_max = _range_of(2)

    # ----- Step 1: 生成 bf16 BNSD random  -----
    torch.manual_seed(_SEED_MAP["q"])
    q_bf16 = (
        torch.rand(B, N_q, max_sq, D, dtype=torch.bfloat16) * (rq_max - rq_min) + rq_min
    )
    torch.manual_seed(_SEED_MAP["k"])
    k_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rk_max - rk_min)
        + rk_min
    )
    torch.manual_seed(_SEED_MAP["v"])
    v_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rv_max - rv_min)
        + rv_min
    )

    # ----- Step 2: 量化 bf16 -> fp8 BNSD + fp32 scale BNSD  -----
    fp8_dtype = torch.float8_e4m3fn
    group_size = 32
    fp8_max = 448.0

    quant_scale_q_bnsd = mxfp8_golden_mod.get_mxfp8_per_token_group_quant_scale(
        q_bf16, fp8_dtype, group_size
    )
    quant_scale_k_bnsd = mxfp8_golden_mod.get_mxfp8_per_token_group_quant_scale(
        k_bf16, fp8_dtype, group_size
    )
    quant_scale_v_bnsd = mxfp8_golden_mod.get_mxfp8_per_channel_group_quant_scale(
        v_bf16, fp8_dtype, group_size
    )

    q_fp8_bnsd_f32 = (
        mxfp8_golden_mod.mxfp8_per_token_group_quant(
            q_bf16, quant_scale_q_bnsd, group_size
        )
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    k_fp8_bnsd_f32 = (
        mxfp8_golden_mod.mxfp8_per_token_group_quant(
            k_bf16, quant_scale_k_bnsd, group_size
        )
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )
    v_fp8_bnsd_f32 = (
        mxfp8_golden_mod.mxfp8_per_channel_group_quant(
            v_bf16, quant_scale_v_bnsd, group_size
        )
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )

    # ----- Step 3: layout 转换 (BNSD -> final layout) -----
    q_fp8_final = mxfp8_golden_mod.convert_q_bnsd_to_layout(
        q_fp8_bnsd_f32, actual_seq_q, "TND", cu_seqlens=cu_seqlens_q
    )
    q_scale_final_layout = mxfp8_golden_mod.convert_q_scale_bnsd_to_layout(
        quant_scale_q_bnsd, actual_seq_q, q_scale_layout, cu_seqlens=cu_seqlens_q
    )

    if enable_pa:
        total_blocks_k = int(dequant_scale_k.shape[0])
        total_blocks_v = int(dequant_scale_v.shape[0])
        total_blocks = min(total_blocks_k, total_blocks_v)

        if block_table.size == 0:
            raise ValueError(
                "[INPUTS] enable_pa=True but block_table CSV shape is (0,) — "
                "Excel AI column empty for PA row"
            )
        max_blocks = block_table.shape[1] if block_table.ndim >= 2 else 0
        if max_blocks <= 0:
            raise ValueError(
                f"[INPUTS] PA mode block_table max_blocks={max_blocks} invalid"
            )
        torch.manual_seed(42)  # 确定性 block_table
        blockid_pool = torch.randperm(total_blocks, dtype=torch.int32)
        bt_real = torch.full((B, max_blocks), -1, dtype=torch.int32)
        idx = 0
        for b in range(B):
            n_blocks = math.ceil(actual_seq_kv[b] / block_size)
            for j in range(n_blocks):
                bt_real[b, j] = blockid_pool[idx % total_blocks]
                idx += 1
        k_fp8_final = mxfp8_golden_mod.mxfp8_pa_preprocessing(
            k_fp8_bnsd_f32,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=False,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_k,
        )
        v_fp8_final = mxfp8_golden_mod.mxfp8_pa_preprocessing(
            v_fp8_bnsd_f32,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=False,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_v,
        )

        k_scale_pa = mxfp8_golden_mod.mxfp8_pa_preprocessing(
            quant_scale_k_bnsd,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=True,
            is_vscale=False,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_k,
        )
        v_scale_pa = mxfp8_golden_mod.mxfp8_pa_preprocessing(
            quant_scale_v_bnsd,
            actual_seq_kv,
            block_size,
            bt_real,
            is_scale=True,
            is_vscale=True,
            kv_layout=kv_cache_layout,
            total_blocks=total_blocks_v,
        )
        k_scale_final_layout = k_scale_pa
        v_scale_final_layout = v_scale_pa
    else:
        # 非 PA 模式: k/v -> TND
        k_fp8_final = mxfp8_golden_mod.convert_kv_bnsd_to_layout(
            k_fp8_bnsd_f32,
            actual_seq_kv,
            "TND",
            cu_seqlens=cu_seqlens_kv if cu_seqlens_kv else None,
        )
        v_fp8_final = mxfp8_golden_mod.convert_kv_bnsd_to_layout(
            v_fp8_bnsd_f32,
            actual_seq_kv,
            "TND",
            cu_seqlens=cu_seqlens_kv if cu_seqlens_kv else None,
        )
        # k/v scale: convert 函数内部做 pack, 输入 UNPACKED 4D scale (B, N, S, D)
        k_scale_final_layout = mxfp8_golden_mod.convert_k_scale_bnsd_to_layout(
            quant_scale_k_bnsd,
            actual_seq_kv,
            "TND",
            cu_seqlens=cu_seqlens_kv if cu_seqlens_kv else None,
        )
        v_scale_final_layout = mxfp8_golden_mod.convert_v_scale_bnsd_to_layout(
            quant_scale_v_bnsd, actual_seq_kv, "TND"
        )

    # ----- Step 4: fp32 scale -> e8m0 -----

    q_scale_e8m0 = mxfp8_golden_mod.fp32_to_e8m0fnu_safe(
        q_scale_final_layout, "Q scale"
    )
    k_scale_e8m0 = mxfp8_golden_mod.fp32_to_e8m0fnu_safe(
        k_scale_final_layout, "K scale"
    )
    v_scale_e8m0 = mxfp8_golden_mod.fp32_to_e8m0fnu_safe(
        v_scale_final_layout, "V scale"
    )

    def _inplace_write(dst_np, src_torch, slot_name):
        if tuple(dst_np.shape) != tuple(src_torch.shape):
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: CSV storage {tuple(dst_np.shape)} "
                f"!= computed {tuple(src_torch.shape)}. "
                f"Check Excel shape vs layout conversion logic."
            )
        dst_dtype_np = dst_np.dtype
        ts = str(src_torch.dtype)
        if "float8" in ts:
            src_np = src_torch.view(dst_dtype_np)
        else:
            src_np = src_torch.numpy()
        if src_np.dtype != dst_dtype_np:
            raise ValueError(
                f"[INPUTS] {slot_name} dtype mismatch: CSV storage {dst_dtype_np} "
                f"!= computed {src_np.dtype} (torch {src_torch.dtype}). "
                f"Check CSV dtype vs inputs.py quant/layout output dtype."
            )
        # numpy in-place 拷贝 (同 dtype 同 shape, bit-exact 内存拷贝)
        dst_np[...] = src_np

    q.copy_(q_fp8_final)
    k.copy_(k_fp8_final)
    v.copy_(v_fp8_final)
    dequant_scale_q.copy_(q_scale_e8m0)
    dequant_scale_k.copy_(k_scale_e8m0)
    dequant_scale_v.copy_(v_scale_e8m0)

    # ----- p_scale: in-place 写 fp32 -----
    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    p_scale[...] = p_scale_value

    if enable_pa:
        block_table[...] = bt_real
    else:
        block_table[...] = 0

    # ----- 新增 slot 8-14: 用 attributes 真实值覆盖 ttk 随机生成的 cu_seqlens/seqused。
    # sinks/metadata (slot 12,14) 无 value，保持 None 语义（CSV shape (0,) → 空 tensor，
    # wrapper/golden 不消费）；attn_mask (slot 13) 不传值覆盖——mask 由 golden
    # _build_causal_mask() 按 attributes 里 attn_mask_shape 重建。 -----
    _write_int32_list(cu_seqlens_q_t, cu_seqlens_q, "cu_seqlens_q (slot 8)")
    _write_int32_list(cu_seqlens_kv_t, cu_seqlens_kv, "cu_seqlens_kv (slot 9)")
    _write_int32_list(seqused_q_t, seqused_q, "seqused_q (slot 10)")
    _write_int32_list(seqused_kv_t, seqused_kv, "seqused_kv (slot 11)")

    logger.info(
        "[INPUTS] in-place wrote fp8 q/k/v (q=%s), e8m0 descale (dq=%s, dk=%s, dv=%s), "
        "fp32 p_scale, int32 block_table (pa=%s)",
        tuple(q.shape),
        tuple(dequant_scale_q.shape),
        tuple(dequant_scale_k.shape),
        tuple(dequant_scale_v.shape),
        enable_pa,
    )


# ==============================================================================
# GQA FP8 全量化输入生成 (quant_mode=6)
# Q/K: per-token-head, V: per-head, descale=FP32 (非 e8m0)
# layout_q=NTD, layout_q_descale=NT, layout_kv=PA_BNBD (K cache 含 scale rows),
# layout_out=TND, 仅 PA 模式
# ==============================================================================


def generate_qfa_gqa_fp8_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    p_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q_t: torch.Tensor,
    cu_seqlens_kv_t: torch.Tensor,
    seqused_q_t: torch.Tensor,
    seqused_kv_t: torch.Tensor,
    sinks_t: torch.Tensor,
    attn_mask_t: torch.Tensor,
    metadata_t: torch.Tensor,
    *,
    batch_size: int,
    N_q: int,
    N_kv: int,
    D: int,
    cu_seqlens_q: List[int],
    cu_seqlens_kv: List[int],
    seqused_q: List[int],
    seqused_kv: List[int],
    max_seqlen_q: int,
    max_seqlen_kv: int,
    enable_pa: bool,
    kv_cache_layout: str,
    block_size: int,
    mask_mode: int,
    q_scale_layout: str,
    quant_mode: int = 6,
    enable_lse: bool = False,
    graph_path: int = 0,
    input_layout: str = "NTD",
    is_contiguous: bool = True,
    device_id: int = 0,
    softmax_scale: float = None,
    data_range_q: float = 1.0,
    data_range_k: float = 1.0,
    data_range_v: float = 1.0,
    **kwargs,
):
    """GQA FP8 输入生成 (quant_mode=6, 仅 PA)

    输出 slot 约定 (in-place 写入 ttk 分配的 numpy slot):
      q slot:              NTD [N,T,D] FP8_E4M3
      k slot:              PA K cache [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8 (末 K_SCALE_ROWS 行存 FP32 deq_k)
      v slot:              PA V cache [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8
      dequant_scale_q slot: NT [N,T] FP32
      dequant_scale_k slot: PA 块状 [Bn,N_kv,block_size] FP32 (与 K cache 数据部分布局对齐;
                            NPU/golden 均从 K cache 提取 deq_k, 此 slot 仅形状对齐)
      dequant_scale_v slot: [N_kv] FP32
      p_scale slot:        [1] FP32
      block_table slot:    [B,max_blocks] int32
    """
    if not enable_pa:
        raise NotImplementedError("GQA FP8 (quant_mode=6) 仅支持 PA 模式")

    # cu_seqlens -> actual_seq (差分还原)
    cu_seqlens_q = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]
    actual_seq_q = (
        [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)]
        if len(cu_seqlens_q) > 1
        else [0]
    )
    if len(cu_seqlens_kv) > 1:
        actual_seq_kv = [
            cu_seqlens_kv[i + 1] - cu_seqlens_kv[i]
            for i in range(len(cu_seqlens_kv) - 1)
        ]
    elif seqused_kv is not None and len(seqused_kv) > 0:
        actual_seq_kv = list(seqused_kv)
    else:
        actual_seq_kv = [0]

    # batch_size: CSV 原始值 (可为 -1), 透传给 metadata;
    # B: 从 cu_seqlens_q 推导的正整数, 供 BNSD 张量生成。
    B = max(1, len(cu_seqlens_q) - 1) if cu_seqlens_q and len(cu_seqlens_q) >= 2 else 1

    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D

    # 注入 golden 全局变量 (prepare_npu_inputs 依赖)
    for gkey, gval in [
        ("B", B),
        ("N_q", N_q),
        ("N_kv", N_kv),
        ("D", D),
        ("FP8_DTYPE", torch.float8_e4m3fn),
        ("QUANT_MODE", 6),
        ("CU_SEQLENS_Q", cu_seqlens_q),
        ("CU_SEQLENS_KV", cu_seqlens_kv if cu_seqlens_kv else None),
        ("SEQUSED_Q", list(seqused_q) if seqused_q is not None else None),
        ("SEQUSED_KV", list(seqused_kv) if seqused_kv is not None else None),
        ("MAX_SEQLEN_Q", max_seqlen_q),
        ("MAX_SEQLEN_KV", max_seqlen_kv),
        ("ENABLE_PA", enable_pa),
        ("KV_CACHE_LAYOUT", kv_cache_layout),
        ("BLOCK_SIZE", block_size),
        ("Q_SCALE_LAYOUT", q_scale_layout),
        ("INPUT_LAYOUT", input_layout),
        ("SPARSE_MODE", mask_mode),
    ]:
        setattr(fp8_golden_mod, gkey, gval)

    # csv input_data_ranges 列由 ttk 作为 input_ranges kwarg 传入
    input_ranges = kwargs.get("input_ranges")

    def _range_of(idx, default=(-1.0, 1.0)):
        if input_ranges is None or idx >= len(input_ranges):
            return default
        r = input_ranges[idx]
        if r is None:
            return default
        return (float(r[0]), float(r[1]))

    rq_min, rq_max = _range_of(0)
    rk_min, rk_max = _range_of(1)
    rv_min, rv_max = _range_of(2)

    # ----- Step 1: 生成 bf16 BNSD random  -----
    torch.manual_seed(_SEED_MAP["q"])
    q_bf16 = (
        torch.rand(B, N_q, max_sq, D, dtype=torch.bfloat16) * (rq_max - rq_min) + rq_min
    )
    torch.manual_seed(_SEED_MAP["k"])
    k_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rk_max - rk_min)
        + rk_min
    )
    torch.manual_seed(_SEED_MAP["v"])
    v_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rv_max - rv_min)
        + rv_min
    )

    # ----- Step 2: GQA FP8 量化 (per-token-head Q/K, per-head V), descale=FP32 -----
    fp8_dtype = torch.float8_e4m3fn

    quant_scale_q_bnsd = fp8_golden_mod.get_fp8_per_token_head_quant_scale(q_bf16)
    quant_scale_k_bnsd = fp8_golden_mod.get_fp8_per_token_head_quant_scale(k_bf16)
    quant_scale_v_bnsd = fp8_golden_mod.get_fp8_per_head_quant_scale(v_bf16)

    # descale = 1/scale, FP32
    dequant_scale_q_bnsd = (1.0 / quant_scale_q_bnsd).contiguous().float()
    dequant_scale_k_bnsd = (1.0 / quant_scale_k_bnsd).contiguous().float()
    dequant_scale_v_bnsd = (1.0 / quant_scale_v_bnsd).contiguous().float()

    q_fp8_bnsd = fp8_golden_mod.quant_fp16_to_fp8(q_bf16, quant_scale_q_bnsd)
    k_fp8_bnsd = fp8_golden_mod.quant_fp16_to_fp8(k_bf16, quant_scale_k_bnsd)
    v_fp8_bnsd = fp8_golden_mod.quant_fp16_to_fp8(v_bf16, quant_scale_v_bnsd)

    # ----- Step 3: layout 转换 (BNSD -> final layout) -----
    # Q: BNSD -> NTD [N,T,D]
    q_fp8_final = fp8_golden_mod.convert_q_bnsd_to_ntd(q_fp8_bnsd, actual_seq_q)
    # deq_q: BNSD [B,N,S,1] -> NT [N,T] FP32
    deq_q_final = fp8_golden_mod.convert_scale_to_layout_gqa(
        dequant_scale_q_bnsd, actual_seq_q, "deq_q"
    )
    # deq_v: [1,N_kv,1,1] -> [N_kv] FP32
    deq_v_final = fp8_golden_mod.convert_scale_to_layout_gqa(
        dequant_scale_v_bnsd, actual_seq_kv, "deq_v"
    )

    # block_table (确定性, seed=42 与 mxfp8 inputs 一致)
    total_blocks_k = int(dequant_scale_k.shape[0]) if dequant_scale_k.ndim >= 1 else 0
    total_blocks_v = int(dequant_scale_v.shape[0]) if dequant_scale_v.ndim >= 1 else 0
    # CSV 分配的 k/v slot 形状决定物理 block 数 (slot 第一维)
    k_slot_blocks = int(k.shape[0]) if k.ndim >= 4 else 0
    v_slot_blocks = int(v.shape[0]) if v.ndim >= 4 else 0
    total_blocks = (
        min(k_slot_blocks, v_slot_blocks) if (k_slot_blocks and v_slot_blocks) else 0
    )

    if block_table.size == 0:
        raise ValueError(
            "[INPUTS GQA FP8] enable_pa=True but block_table CSV shape is (0,) — "
            "Excel AI column empty for PA row"
        )
    max_blocks = block_table.shape[1] if block_table.ndim >= 2 else 0
    if max_blocks <= 0:
        raise ValueError(
            f"[INPUTS GQA FP8] PA mode block_table max_blocks={max_blocks} invalid"
        )
    torch.manual_seed(42)
    blockid_pool = (
        torch.randperm(total_blocks, dtype=torch.int32)
        if total_blocks > 0
        else torch.zeros(0, dtype=torch.int32)
    )
    bt_real = torch.full((B, max_blocks), -1, dtype=torch.int32)
    idx = 0
    for b in range(B):
        n_blocks = math.ceil(actual_seq_kv[b] / block_size)
        for j in range(n_blocks):
            bt_real[b, j] = blockid_pool[idx % total_blocks] if total_blocks > 0 else 0
            idx += 1

    # K cache: BNSD -> PA_BNBD, 末 K_SCALE_ROWS 行存 FP32 deq_k
    k_fp8_final = fp8_golden_mod.bnsd_to_k_cache_gqa(
        k_fp8_bnsd,
        dequant_scale_k_bnsd,
        actual_seq_kv,
        block_size,
        bt_real,
        num_blocks=total_blocks,
    )
    # V cache: BNSD -> PA_BNBD (含 K_SCALE_ROWS 占位行, 无 scale 数据)
    v_fp8_final = fp8_golden_mod.bnsd_to_v_cache_gqa(
        v_fp8_bnsd,
        actual_seq_kv,
        block_size,
        bt_real,
        num_blocks=total_blocks,
    )

    # deq_k slot: 从 K cache 末 K_SCALE_ROWS 行 (uint8 view → FP32) 直接提取
    # k_fp8_final: [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8 → uint8 → FP32 reshape
    # 末 block_size 个 FP32 值 = 嵌入的 K scale, 形状 [Bn,N_kv,block_size] 匹配 slot
    k_f32 = (
        k_fp8_final.view(torch.uint8)
        .view(k_fp8_final.shape[0], k_fp8_final.shape[1], -1)
        .view(torch.float32)
    )
    deq_k_slot = k_f32[:, :, -block_size:].contiguous()

    # ----- Step 4: in-place 写入 ttk 分配的 slot -----
    # TTK 在 use_torch=True 时把 numpy slot 转成 torch tensor 传给本函数,
    # 因此 dst 是 torch tensor (dtype=torch.float8_e4m3fn / torch.float32 / torch.int32),
    # 用 torch 原生 .copy_() 做 bit-exact in-place 写入 (支持 fp8)。
    def _inplace_write(dst, src_torch, slot_name):
        if tuple(dst.shape) != tuple(src_torch.shape):
            raise ValueError(
                f"[INPUTS GQA FP8] {slot_name} shape mismatch: slot {tuple(dst.shape)} "
                f"!= computed {tuple(src_torch.shape)}. "
                f"Check Excel shape vs layout conversion logic."
            )
        if str(dst.dtype) != str(src_torch.dtype):
            raise ValueError(
                f"[INPUTS GQA FP8] {slot_name} dtype mismatch: slot {dst.dtype} "
                f"!= computed {src_torch.dtype}."
            )
        dst.copy_(src_torch)

    _inplace_write(q, q_fp8_final, "q (slot 0, NTD)")
    _inplace_write(k, k_fp8_final, "k (slot 1, PA cache + scale rows)")
    _inplace_write(v, v_fp8_final, "v (slot 2, PA cache)")
    _inplace_write(dequant_scale_q, deq_q_final, "descale_q (slot 3, NT FP32)")
    _inplace_write(dequant_scale_k, deq_k_slot, "descale_k (slot 4, BNSD FP32)")
    _inplace_write(dequant_scale_v, deq_v_final, "descale_v (slot 5, [N_kv] FP32)")

    # p_scale: in-place 写 fp32
    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    p_scale.copy_(torch.tensor([float(p_scale_value)], dtype=torch.float32))

    # block_table: int32
    block_table.copy_(bt_real.to(torch.int32))

    # ----- 新增 slot 8-14: 用 attributes 真实值覆盖 ttk 随机生成的 cu_seqlens/seqused。
    # sinks/metadata (slot 12,14) 无 value，保持 None 语义（空 tensor）；attn_mask (slot 13)
    # 不传值覆盖，mask 由 golden 按 attn_mask_shape 重建。 -----
    _write_int32_list(cu_seqlens_q_t, cu_seqlens_q, "cu_seqlens_q (slot 8)")
    _write_int32_list(cu_seqlens_kv_t, cu_seqlens_kv, "cu_seqlens_kv (slot 9)")
    _write_int32_list(seqused_q_t, seqused_q, "seqused_q (slot 10)")
    _write_int32_list(seqused_kv_t, seqused_kv, "seqused_kv (slot 11)")

    logger.info(
        "[INPUTS GQA FP8] in-place wrote fp8 q (NTD %s), k/v (PA cache %s, %s), "
        "fp32 descale (dq NT %s, dk BNSD %s, dv %s), fp32 p_scale, int32 block_table",
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(dequant_scale_q.shape),
        tuple(dequant_scale_k.shape),
        tuple(dequant_scale_v.shape),
    )


def generate_qfa_hif8_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    p_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q_t: torch.Tensor,
    cu_seqlens_kv_t: torch.Tensor,
    seqused_q_t: torch.Tensor,
    seqused_kv_t: torch.Tensor,
    sinks_t: torch.Tensor,
    attn_mask_t: torch.Tensor,
    metadata_t: torch.Tensor,
    *,
    batch_size: int,
    N_q: int,
    N_kv: int,
    D: int,
    cu_seqlens_q: List[int],
    cu_seqlens_kv: List[int],
    seqused_q: List[int],
    seqused_kv: List[int],
    max_seqlen_q: int,
    max_seqlen_kv: int,
    enable_pa: bool,
    kv_cache_layout: str,
    block_size: int,
    mask_mode: int,
    q_scale_layout: str,
    quant_mode: int = 0,
    enable_lse: bool = False,
    graph_path: int = 0,
    input_layout: str = "TND",
    layout_q: str = None,
    layout_kv: str = None,
    layout_out: str = None,
    is_contiguous: bool = True,
    device_id: int = 0,
    softmax_scale: float = None,
    data_range_q: float = 1.0,
    data_range_k: float = 1.0,
    data_range_v: float = 1.0,
    **kwargs,
):
    """HIF8 输入生成 (quant_mode=0, per-tensor, 仅 TND, 无 PA)。

    输出 slot 约定 (in-place 写入 ttk 分配的 numpy slot):
      q slot:    TND [T_q, N_q, D] uint8 (hifloat8 编码)
      k slot:    TND [T_kv, N_kv, D] uint8
      v slot:    TND [T_kv, N_kv, D] uint8
      dq slot:   (1,) float32 per-tensor descale
      dk slot:   (1,) float32 per-tensor descale
      dv slot:   (1,) float32 per-tensor descale
      p_scale:   (1,) float32
      block_table: (0,) int32 (无 PA)
    """
    import quant_flash_attn_hif8_golden as hif8_golden_mod

    input_layout = layout_q or input_layout
    cu_seqlens_q = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]

    B = (
        batch_size
        if batch_size is not None and batch_size > 0
        else (
            max(1, len(cu_seqlens_q) - 1)
            if cu_seqlens_q and len(cu_seqlens_q) >= 2
            else len(list(seqused_q))
            if seqused_q is not None and len(list(seqused_q)) > 0
            else 1
        )
    )

    if len(cu_seqlens_q) > 1:
        actual_seq_q = [
            cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)
        ]
    elif seqused_q is not None and len(list(seqused_q)) > 0:
        actual_seq_q = list(seqused_q)
    elif max_seqlen_q is not None and max_seqlen_q > 0:
        actual_seq_q = [max_seqlen_q] * B
    else:
        actual_seq_q = [0]
    if len(cu_seqlens_kv) > 1:
        actual_seq_kv = [
            cu_seqlens_kv[i + 1] - cu_seqlens_kv[i]
            for i in range(len(cu_seqlens_kv) - 1)
        ]
    elif seqused_kv is not None and len(list(seqused_kv)) > 0:
        actual_seq_kv = list(seqused_kv)
    elif max_seqlen_kv is not None and max_seqlen_kv > 0:
        actual_seq_kv = [max_seqlen_kv] * B
    else:
        actual_seq_kv = [0]

    # B 可能为-1
    if B is None or B <= 0:
        B = len(actual_seq_q)

    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D

    for gkey, gval in [
        ("B", B),
        ("N_q", N_q),
        ("N_kv", N_kv),
        ("D", D),
        ("CU_SEQLENS_Q", cu_seqlens_q),
        ("CU_SEQLENS_KV", cu_seqlens_kv if cu_seqlens_kv else None),
        ("SEQUSED_Q", list(seqused_q) if seqused_q is not None else None),
        ("SEQUSED_KV", list(seqused_kv) if seqused_kv is not None else None),
        ("MAX_SEQLEN_Q", max_seqlen_q),
        ("MAX_SEQLEN_KV", max_seqlen_kv),
        ("SPARSE_MODE", mask_mode),
        ("Q_SCALE_LAYOUT", q_scale_layout),
        ("INPUT_LAYOUT", input_layout),
        ("IS_CONTIGUOUS", is_contiguous),
        ("DEVICE_ID", device_id),
        ("GRAPH_PATH", graph_path),
        ("SOFTMAX_SCALE", softmax_scale),
    ]:
        setattr(hif8_golden_mod, gkey, gval)

    input_ranges = kwargs.get("input_ranges")

    def _range_of(idx, default=(-1.0, 1.0)):
        if input_ranges is None or idx >= len(input_ranges):
            return default
        r = input_ranges[idx]
        if r is None:
            return default
        return (float(r[0]), float(r[1]))

    rq_min, rq_max = _range_of(0, (-1.0, 1.0))
    rk_min, rk_max = _range_of(1, (-1.0, 1.0))
    rv_min, rv_max = _range_of(2, (-1.0, 1.0))

    data_range_q_eff = (
        data_range_q if data_range_q is not None and data_range_q > 0 else 1.0
    )
    data_range_k_eff = (
        data_range_k if data_range_k is not None and data_range_k > 0 else 1.0
    )
    data_range_v_eff = (
        data_range_v if data_range_v is not None and data_range_v > 0 else 1.0
    )

    torch.manual_seed(_SEED_MAP["q"])
    q_fp16 = (
        torch.rand(B, N_q, max_sq, D, dtype=torch.float16) * 2 - 1
    ) * data_range_q_eff
    torch.manual_seed(_SEED_MAP["k"])
    k_fp16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1
    ) * data_range_k_eff
    torch.manual_seed(_SEED_MAP["v"])
    v_fp16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.float16) * 2 - 1
    ) * data_range_v_eff

    quant_scale_q = hif8_golden_mod.get_hif8_per_tensor_quant_scale(
        q_fp16.to(torch.float32)
    )
    quant_scale_k = hif8_golden_mod.get_hif8_per_tensor_quant_scale(
        k_fp16.to(torch.float32)
    )
    quant_scale_v = hif8_golden_mod.get_hif8_per_tensor_quant_scale(
        v_fp16.to(torch.float32)
    )

    q_hif8_bnsd = hif8_golden_mod.hif8_per_tensor_quant(
        q_fp16.to(torch.float32), quant_scale_q
    )
    k_hif8_bnsd = hif8_golden_mod.hif8_per_tensor_quant(
        k_fp16.to(torch.float32), quant_scale_k
    )
    v_hif8_bnsd = hif8_golden_mod.hif8_per_tensor_quant(
        v_fp16.to(torch.float32), quant_scale_v
    )

    q_final = (
        hif8_golden_mod.convert_q_bnsd_to_layout(
            q_hif8_bnsd,
            actual_seq_q,
            input_layout,
            cu_seqlens=cu_seqlens_q
            if (input_layout == "TND" and len(cu_seqlens_q) > 1)
            else None,
        )
        .contiguous()
        .view(torch.float8_e4m3fn)
    )
    k_final = (
        hif8_golden_mod.convert_kv_bnsd_to_layout(
            k_hif8_bnsd,
            actual_seq_kv,
            input_layout,
            cu_seqlens=cu_seqlens_kv
            if (input_layout == "TND" and len(cu_seqlens_kv) > 1)
            else None,
        )
        .contiguous()
        .view(torch.float8_e4m3fn)
    )
    v_final = (
        hif8_golden_mod.convert_kv_bnsd_to_layout(
            v_hif8_bnsd,
            actual_seq_kv,
            input_layout,
            cu_seqlens=cu_seqlens_kv
            if (input_layout == "TND" and len(cu_seqlens_kv) > 1)
            else None,
        )
        .contiguous()
        .view(torch.float8_e4m3fn)
    )

    def _inplace_write(dst, src_torch, slot_name):
        if tuple(dst.shape) != tuple(src_torch.shape):
            raise ValueError(
                f"[INPUTS HIF8] {slot_name} shape mismatch: slot {tuple(dst.shape)} "
                f"!= computed {tuple(src_torch.shape)}."
            )
        dst.copy_(src_torch.view(dst.dtype))

    _inplace_write(q, q_final, "q (slot 0)")
    _inplace_write(k, k_final, "k (slot 1)")
    _inplace_write(v, v_final, "v (slot 2)")
    _inplace_write(dequant_scale_q, quant_scale_q, "descale_q (slot 3)")
    _inplace_write(dequant_scale_k, quant_scale_k, "descale_k (slot 4)")
    _inplace_write(dequant_scale_v, quant_scale_v, "descale_v (slot 5)")

    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    p_scale.copy_(torch.tensor([float(p_scale_value)], dtype=torch.float32))
    block_table.copy_(torch.zeros_like(block_table))

    # ----- slot 8-14: 用 attributes 真实值覆盖 ttk 随机生成的 cu_seqlens/seqused -----
    _write_int32_list(cu_seqlens_q_t, cu_seqlens_q, "cu_seqlens_q (slot 8)")
    _write_int32_list(cu_seqlens_kv_t, cu_seqlens_kv, "cu_seqlens_kv (slot 9)")
    _write_int32_list(seqused_q_t, seqused_q, "seqused_q (slot 10)")
    _write_int32_list(seqused_kv_t, seqused_kv, "seqused_kv (slot 11)")

    logger.info(
        "[INPUTS] HIF8 in-place wrote q/k/v (q=%s), fp32 descale (dq=%s, dk=%s, dv=%s), "
        "fp32 p_scale, int32 block_table (pa=False)",
        tuple(q.shape),
        tuple(dequant_scale_q.shape),
        tuple(dequant_scale_k.shape),
        tuple(dequant_scale_v.shape),
    )
