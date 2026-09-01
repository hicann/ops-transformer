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

import numpy
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.join(_ASSETS_DIR, "..")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
import quant_flash_attn_golden as mxfp8_golden_mod
import quant_flash_attn_fp8_golden as fp8_golden_mod
import quant_flash_attn_hif8_golden as hif8_golden_mod

logger = logging.getLogger(__name__)

__golden__ = {"e2e": {"qfa_wrapper.npu_qfa": "cpu_qfa_mxfp8"}}


# ==============================================================================
# 框架 fp8/e8m0 -> torch 转换
# ==============================================================================


def _to_torch_fp8(t):
    if isinstance(t, torch.Tensor):
        if t.dtype == torch.float8_e4m3fn:
            return t
        if t.dtype == torch.uint8:
            return t.view(torch.float8_e4m3fn)
        return t.to(torch.float8_e4m3fn)
    arr = numpy.asarray(t)
    if arr.dtype == numpy.uint8:
        return torch.from_numpy(arr).view(torch.float8_e4m3fn)
    return torch.from_numpy(arr.view(numpy.uint8)).view(torch.float8_e4m3fn)


def _to_torch_e8m0(t):
    if isinstance(t, torch.Tensor):
        if t.dtype == torch.float8_e8m0fnu:
            return t
        if t.dtype == torch.uint8:
            return t.view(torch.float8_e8m0fnu)
        raise ValueError(f"_to_torch_e8m0: unexpected torch dtype {t.dtype}")
    arr = numpy.asarray(t)
    if arr.dtype == numpy.uint8:
        return torch.from_numpy(arr).view(torch.float8_e8m0fnu)
    return torch.from_numpy(arr.view(numpy.uint8)).view(torch.float8_e8m0fnu)


def _to_torch_int32(t):
    if isinstance(t, torch.Tensor):
        return t.to(torch.int32) if t.dtype != torch.int32 else t
    return torch.from_numpy(numpy.asarray(t)).to(torch.int32)


def _tnd_to_bnsd_fixed(tensor_tnd, seq_lens, cu_seqlens=None):
    tensor = (
        tensor_tnd
        if isinstance(tensor_tnd, torch.Tensor)
        else torch.as_tensor(tensor_tnd)
    )
    B = len(seq_lens)
    N = tensor.shape[1]
    D = tensor.shape[2]
    max_seq = max(seq_lens) if seq_lens else 0
    result = torch.zeros((B, N, max_seq, D), dtype=tensor.dtype, device=tensor.device)
    if cu_seqlens is not None:
        for b in range(B):
            act_s = seq_lens[b]
            if act_s <= 0:
                continue
            offset = cu_seqlens[b]
            result[b, :, :act_s, :] = tensor[offset : offset + act_s, :, :].permute(
                1, 0, 2
            )
    else:
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            if act_s > 0:
                result[b, :, :act_s, :] = tensor[t : t + act_s, :, :].permute(1, 0, 2)
            t += act_s
    return result.contiguous()


def _assert_no_e8m0_nan(t_e8m0, name="scale"):
    if t_e8m0.numel() == 0:
        return
    nan_count = int((t_e8m0.view(torch.uint8) == 0xFF).sum().item())
    if nan_count:
        raise ValueError(
            f"{name}: {nan_count} e8m0 bytes are 0xFF (NaN sentinel) — "
            "bin data corrupted or inputs.py sanitize bypassed"
        )


# ==============================================================================
# PA paged layout 逆转换辅助
# ==============================================================================


def _reverse_layout_to_cache(pa_tensor, kv_layout, is_scale, is_vscale):
    layout = (kv_layout or "BnNBsD").upper()
    if layout in ("BNNBSD", "PA_BNBD"):
        return pa_tensor
    if layout in ("BNBSND", "PA_BBND"):
        return pa_tensor.transpose(1, 2).contiguous()
    if layout == "PA_NZ":
        return mxfp8_golden_mod.pa_reverse_permute_nz(pa_tensor, is_scale, is_vscale)
    raise ValueError(f"Unsupported kv_layout: {kv_layout}")


def _tnd_qk_scale_to_bnsd_grouped(
    scale_tnd_packed, seq_lens, cu_seqlens=None, q_scale_layout="TND", num_kv_heads=None
):
    layout = mxfp8_golden_mod.canonical_q_scale_layout(q_scale_layout)
    if layout == "N2TGD":
        if num_kv_heads is None:
            raise ValueError("num_kv_heads required for N2TGD layout")
        tnd_packed = scale_tnd_packed.permute(1, 0, 2, 3, 4).contiguous()
        N_kv, T, G, Dg_half, _ = scale_tnd_packed.shape
        N_q = N_kv * G
        tnd_packed = tnd_packed.reshape(T, N_q, Dg_half, 2)
    else:
        tnd_packed = scale_tnd_packed

    T, N, Dg_half, _pair = tnd_packed.shape
    Dg = Dg_half * 2
    tnd_grouped = tnd_packed.reshape(T, N, Dg)
    return _tnd_to_bnsd_fixed(tnd_grouped, seq_lens, cu_seqlens=cu_seqlens)


def _tnd_v_scale_to_bnsd_grouped(
    scale_tnd_packed, seq_lens, cu_seqlens=None, group_size=32
):
    T, N, D, _pair = scale_tnd_packed.shape
    sg_per_batch = [math.ceil(s / group_size) for s in seq_lens]
    sg_max = max(sg_per_batch) if sg_per_batch else 0
    B = len(seq_lens)
    result = torch.zeros(
        (B, N, sg_max, D), dtype=scale_tnd_packed.dtype, device=scale_tnd_packed.device
    )

    t = 0
    for b in range(B):
        sg = sg_per_batch[b]
        if sg <= 0:
            continue
        sg_padded = sg + (sg % 2)
        s_out_b = sg_padded // 2
        if cu_seqlens is not None:
            t_start = t
        else:
            t_start = t
        t_end = t_start + s_out_b
        chunk = scale_tnd_packed[t_start:t_end, :, :, :]
        recovered = torch.zeros(
            (sg_padded, N, D), dtype=chunk.dtype, device=chunk.device
        )
        recovered[0::2, :, :] = chunk[:, :, :, 0]
        recovered[1::2, :, :] = chunk[:, :, :, 1]
        result[b, :, :sg, :] = recovered[:sg, :, :].permute(1, 0, 2)
        t = t_end
    return result.contiguous()


def _pa_k_scale_to_bnsd_grouped(pa_cache, seq_lens, block_size, block_table):
    pa_cache = (
        pa_cache if isinstance(pa_cache, torch.Tensor) else torch.as_tensor(pa_cache)
    )
    block_table = (
        block_table
        if isinstance(block_table, torch.Tensor)
        else torch.as_tensor(block_table)
    )
    Bn, N, Bs, Dg_half, _pair = pa_cache.shape
    Dg = Dg_half * 2
    B = block_table.shape[0]
    max_skv = max(seq_lens) if seq_lens else 0
    result = torch.zeros(
        (B, N, max_skv, Dg_half, 2), dtype=pa_cache.dtype, device=pa_cache.device
    )
    num_blocks = [math.ceil(s / Bs) for s in seq_lens]
    for b in range(B):
        bid_table = block_table[b]
        for blk_idx in range(num_blocks[b]):
            blockid = int(bid_table[blk_idx])
            block_offset = blk_idx * Bs
            valid_len = min(Bs, seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            result[b, :, block_offset : block_offset + valid_len] = pa_cache[
                blockid, :, :valid_len
            ]
    return result[:, :, :max_skv, :, :].reshape(B, N, max_skv, Dg).contiguous()


def _pa_v_scale_to_bnsd_grouped(
    pa_cache, seq_lens, block_size, block_table, group_size=32
):
    pa_cache = (
        pa_cache if isinstance(pa_cache, torch.Tensor) else torch.as_tensor(pa_cache)
    )
    block_table = (
        block_table
        if isinstance(block_table, torch.Tensor)
        else torch.as_tensor(block_table)
    )
    Bn, N, pack_bs, D, _pair = pa_cache.shape
    B = block_table.shape[0]
    v_scale_pack_ratio = group_size * 2
    pack_seq_lens = [math.ceil(s / v_scale_pack_ratio) for s in seq_lens]
    max_packed = max(pack_seq_lens) if pack_seq_lens else 0
    result_packed = torch.zeros(
        (B, N, max_packed, D, 2), dtype=pa_cache.dtype, device=pa_cache.device
    )
    num_blocks = [math.ceil(s / pack_bs) for s in pack_seq_lens]
    for b in range(B):
        bid_table = block_table[b]
        for blk_idx in range(num_blocks[b]):
            blockid = int(bid_table[blk_idx])
            block_offset = blk_idx * pack_bs
            valid_len = min(pack_bs, pack_seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            result_packed[b, :, block_offset : block_offset + valid_len] = pa_cache[
                blockid, :, :valid_len
            ]
    result_packed = result_packed[:, :, :max_packed, :, :].contiguous()
    B_out, N_out, S_packed, D_out, _ = result_packed.shape
    Sg_recovered = S_packed * 2
    unpacked = (
        result_packed.permute(0, 1, 2, 4, 3)
        .contiguous()
        .reshape(B_out, N_out, Sg_recovered, D_out)
    )
    sg_per_batch = [math.ceil(s / group_size) for s in seq_lens]
    sg_max = max(sg_per_batch) if sg_per_batch else 0
    return unpacked[:, :, :sg_max, :].contiguous()


def _apply_golden_globals(attrs, quant_mode=1):
    """把 case 属性注入 golden 模块全局变量 (按 quant_mode 选择目标模块).

    quant_mode=6 → fp8_golden_mod (GQA FP8 全量化路径)
    quant_mode=0 → hif8_golden_mod (HIF8 per-tensor 量化路径)
    其他 → mxfp8_golden_mod (MXFP8 路径)
    """
    if quant_mode == 6:
        target = fp8_golden_mod
    elif quant_mode == 0:
        target = hif8_golden_mod
    else:
        target = mxfp8_golden_mod
    mapping = {
        "B": "B",
        "N_q": "N_q",
        "N_kv": "N_kv",
        "D": "D",
        "enable_pa": "ENABLE_PA",
        "kv_cache_layout": "KV_CACHE_LAYOUT",
        "block_size": "BLOCK_SIZE",
        "mask_mode": "SPARSE_MODE",
        "attn_mask_shape": "ATTN_MASK_SHAPE",
        "q_scale_layout": "Q_SCALE_LAYOUT",
        "enable_lse": "ENABLE_LSE",
        "input_layout": "INPUT_LAYOUT",
        "is_contiguous": "IS_CONTIGUOUS",
        "device_id": "DEVICE_ID",
        "graph_path": "GRAPH_PATH",
        "cu_seqlens_q": "CU_SEQLENS_Q",
        "cu_seqlens_kv": "CU_SEQLENS_KV",
        "seqused_q": "SEQUSED_Q",
        "seqused_kv": "SEQUSED_KV",
        "max_seqlen_q": "MAX_SEQLEN_Q",
        "max_seqlen_kv": "MAX_SEQLEN_KV",
        "softmax_scale": "SOFTMAX_SCALE",
        "head_dim_v": "HEAD_DIM_V",
    }
    for attr_key, golden_key in mapping.items():
        if attr_key in attrs:
            setattr(target, golden_key, attrs[attr_key])
    setattr(target, "FP8_DTYPE", torch.float8_e4m3fn)
    setattr(target, "QUANT_GROUP_SIZE", 32)
    p_scale = attrs.get("p_scale_value", 1.0)
    if isinstance(p_scale, torch.Tensor):
        p_scale = float(p_scale.item())
    else:
        p_scale = float(p_scale)
    setattr(target, "P_SCALE", p_scale)


def _cu_seqlens_to_actual(cu_seqlens):
    """cu_seqlens → actual_seq (差分还原)。

    cu_seqlens = [0, 3, 7, 10] → actual_seq = [3, 4, 3]
    """
    if cu_seqlens is None or len(cu_seqlens) < 2:
        return []
    return [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)]


def _tolist_t(t):
    """从 CPU tensor slot (_t) 读回 list；空/None → None。

    cu_seqlens/seqused 的真实值由 inputs.py in-place 写入 tensor slot，
    golden 收到的是 CPU tensor，直接 .detach().cpu().tolist() 取回。
    """
    if t is None:
        return None
    return t.detach().cpu().tolist() if t.numel() > 0 else None


def cpu_qfa_mxfp8(
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
    **kwargs,
):
    # csv precision_tolerances / absolute_precision 经 testcase.attributes → kwargs 传入,
    # 暂存到 mxfp8_golden_mod 供 compare 插件读取（ttk 不把 testcase 直接传给 custom compare）。
    mxfp8_golden_mod._csv_precision_tolerances = kwargs.get("precision_tolerances")
    mxfp8_golden_mod._csv_absolute_precision = kwargs.get("absolute_precision")

    # cu_seqlens/seqused 的真实值由 inputs.py in-place 写入 tensor slot (_t)。
    # golden 端 _t 是 CPU tensor，直接读回 list。
    cu_seqlens_q_list = _tolist_t(cu_seqlens_q_t)
    cu_seqlens_kv_list = _tolist_t(cu_seqlens_kv_t)
    seqused_q_list = _tolist_t(seqused_q_t)
    seqused_kv_list = _tolist_t(seqused_kv_t)

    # actual_seq (有效长度) 用于 CPU golden 的 attention mask 构建——必须与 NPU 侧
    # prepare_npu_inputs 里的 _actual_seq_q/kv() 取值逻辑一致：
    #   优先用 seqused (实际有效长度，可小于 padded 范围)；
    #   seqused 为 None 时才用 cu_seqlens 差分 (物理 TND 范围)。
    actual_seq_q = (
        list(seqused_q_list)
        if (seqused_q_list is not None and len(seqused_q_list) > 0)
        else _cu_seqlens_to_actual(cu_seqlens_q_list)
    )
    actual_seq_kv = (
        list(seqused_kv_list)
        if (seqused_kv_list is not None and len(seqused_kv_list) > 0)
        else _cu_seqlens_to_actual(cu_seqlens_kv_list)
    )

    # batch_size: CSV 原始值 (可为 -1), 透传给 metadata;
    # B: 从 cu_seqlens_q 推导的正整数, 供 shape assert 与 BNSD 逆转换。
    B = (
        max(1, len(cu_seqlens_q_list) - 1)
        if cu_seqlens_q_list and len(cu_seqlens_q_list) >= 2
        else 1
    )

    # p_scale_value 从 CSV attributes 经 kwargs 传入
    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    else:
        p_scale_value = float(p_scale_value)

    _apply_golden_globals(
        {
            "B": B,
            "N_q": N_q,
            "N_kv": N_kv,
            "D": D,
            "cu_seqlens_q": cu_seqlens_q_list,
            "cu_seqlens_kv": cu_seqlens_kv_list,
            "seqused_q": seqused_q_list,
            "seqused_kv": seqused_kv_list,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
            "enable_pa": enable_pa,
            "kv_cache_layout": kv_cache_layout,
            "block_size": block_size,
            "mask_mode": mask_mode,
            "q_scale_layout": q_scale_layout,
            "enable_lse": enable_lse,
            "input_layout": input_layout,
            "is_contiguous": is_contiguous,
            "device_id": device_id,
            "graph_path": graph_path,
            "softmax_scale": softmax_scale,
            "p_scale_value": p_scale_value,
        },
        quant_mode=1,
    )

    group_size = 32

    # actual_seq 推导 max_seq (用于 assertion 和 PA 逆转换的 scatter 范围)
    max_sq = max(actual_seq_q) if actual_seq_q else D
    max_skv = max(actual_seq_kv) if actual_seq_kv else D
    # D=64→2, D=72→4 D=128→4.
    _num_groups = math.ceil(D / group_size)
    Dg = _num_groups + (_num_groups % 2)

    # Q: TND -> BNSD
    q_fp8_tnd = _to_torch_fp8(q)
    q_fp8_bnsd = _tnd_to_bnsd_fixed(
        q_fp8_tnd, actual_seq_q, cu_seqlens=cu_seqlens_q_list
    )

    # Q scale: e8m0 -> fp32
    dq_e8m0 = _to_torch_e8m0(dequant_scale_q)
    _assert_no_e8m0_nan(dq_e8m0, "descale_q")
    dq_fp32 = mxfp8_golden_mod.e8m0_to_fp32(dq_e8m0)
    _assert_no_e8m0_nan(dq_fp32, "descale_q")
    dq_bnsd_grouped = _tnd_qk_scale_to_bnsd_grouped(
        dq_fp32,
        actual_seq_q,
        cu_seqlens=cu_seqlens_q_list,
        q_scale_layout=q_scale_layout,
        num_kv_heads=N_kv,
    )
    assert dq_bnsd_grouped.shape == (B, N_q, max_sq, Dg), (
        f"Q scale grouped shape mismatch: got {tuple(dq_bnsd_grouped.shape)}, "
        f"expected {(B, N_q, max_sq, Dg)}"
    )

    # K/V: PA paged / TND -> BNSD
    if enable_pa:
        bt = _to_torch_int32(block_table)
        if bt.numel() == 0 or bt.ndim < 2:
            raise ValueError(
                f"[GOLDEN] enable_pa=True but block_table invalid: shape={tuple(bt.shape)}"
            )

        k_pa = _to_torch_fp8(k)
        k_cache = _reverse_layout_to_cache(
            k_pa, kv_cache_layout, is_scale=False, is_vscale=False
        )
        k_fp8_bnsd = mxfp8_golden_mod.pa_to_bnsd_data(
            k_cache, actual_seq_kv, block_size, bt, kv_layout=kv_cache_layout
        )
        # PA_NZ 等布局会对 D 维做对齐 padding (如 D=72→96, 32 对齐),
        # 裁剪到 logical D, 排除 padding 列 (fill_value=0), 与 NPU 算子
        # 按 head_dim=logical D 计算的行为一致. 对 D∈{64,96,128} 等整除值是 no-op.
        if k_fp8_bnsd.shape[-1] > D:
            k_fp8_bnsd = k_fp8_bnsd[:, :, :, :D].contiguous()
        # V data: 同 K
        v_pa = _to_torch_fp8(v)
        v_cache = _reverse_layout_to_cache(
            v_pa, kv_cache_layout, is_scale=False, is_vscale=False
        )
        v_fp8_bnsd = mxfp8_golden_mod.pa_to_bnsd_data(
            v_cache, actual_seq_kv, block_size, bt, kv_layout=kv_cache_layout
        )
        if v_fp8_bnsd.shape[-1] > D:
            v_fp8_bnsd = v_fp8_bnsd[:, :, :, :D].contiguous()

        # K scale: e8m0 -> fp32 -> reverse layout -> out_cache (Bn,N,Bs,Dg//2,2) -> grouped BNSD
        dk_e8m0 = _to_torch_e8m0(dequant_scale_k)
        _assert_no_e8m0_nan(dk_e8m0, "descale_k")
        dk_fp32 = mxfp8_golden_mod.e8m0_to_fp32(dk_e8m0)
        dk_cache = _reverse_layout_to_cache(
            dk_fp32, kv_cache_layout, is_scale=True, is_vscale=False
        )
        dk_bnsd_grouped = _pa_k_scale_to_bnsd_grouped(
            dk_cache, actual_seq_kv, block_size, bt
        )
        assert dk_bnsd_grouped.shape == (B, N_kv, max_skv, Dg), (
            f"K scale grouped shape mismatch: got {tuple(dk_bnsd_grouped.shape)}, "
            f"expected {(B, N_kv, max_skv, Dg)}"
        )

        # V scale: e8m0 -> fp32 -> reverse layout -> out_cache (Bn,N,pack_bs,D,2) -> grouped BNSD
        dv_e8m0 = _to_torch_e8m0(dequant_scale_v)
        _assert_no_e8m0_nan(dv_e8m0, "descale_v")
        dv_fp32 = mxfp8_golden_mod.e8m0_to_fp32(dv_e8m0)
        dv_cache = _reverse_layout_to_cache(
            dv_fp32, kv_cache_layout, is_scale=True, is_vscale=True
        )
        dv_bnsd_grouped = _pa_v_scale_to_bnsd_grouped(
            dv_cache, actual_seq_kv, block_size, bt, group_size=group_size
        )
        # V scale grouped: (B, N_kv, Sg_max, D), Sg_max = ceil(max_skv/32)
        # PA_NZ 布局 V scale 的 D 维会 16 对齐 padding (如 D=72→80),
        # 物理 D >= logical D, 裁剪到 logical D 排除 padding (fill_value=E8M0_MIN_POSITIVE).
        Sg_max = math.ceil(max_skv / group_size) if max_skv > 0 else 0
        assert dv_bnsd_grouped.shape[:3] == (B, N_kv, Sg_max), (
            f"V scale grouped first 3 dims mismatch: got {tuple(dv_bnsd_grouped.shape[:3])}, "
            f"expected {(B, N_kv, Sg_max)}"
        )
        assert dv_bnsd_grouped.shape[3] >= D, (
            f"V scale grouped D={dv_bnsd_grouped.shape[3]} < logical D={D}"
        )
        if dv_bnsd_grouped.shape[3] > D:
            dv_bnsd_grouped = dv_bnsd_grouped[:, :, :, :D].contiguous()
    else:
        # 非 PA: K/V 都是 TND -> BNSD
        k_fp8_tnd = _to_torch_fp8(k)
        k_fp8_bnsd = _tnd_to_bnsd_fixed(
            k_fp8_tnd, actual_seq_kv, cu_seqlens=cu_seqlens_kv_list
        )
        if k_fp8_bnsd.shape[-1] > D:
            k_fp8_bnsd = k_fp8_bnsd[:, :, :, :D].contiguous()
        v_fp8_tnd = _to_torch_fp8(v)
        v_fp8_bnsd = _tnd_to_bnsd_fixed(
            v_fp8_tnd, actual_seq_kv, cu_seqlens=cu_seqlens_kv_list
        )
        if v_fp8_bnsd.shape[-1] > D:
            v_fp8_bnsd = v_fp8_bnsd[:, :, :, :D].contiguous()

        # K scale: e8m0 -> fp32 -> TND grouped -> BNSD grouped
        dk_e8m0 = _to_torch_e8m0(dequant_scale_k)
        _assert_no_e8m0_nan(dk_e8m0, "descale_k")
        dk_fp32 = mxfp8_golden_mod.e8m0_to_fp32(dk_e8m0)
        dk_bnsd_grouped = _tnd_qk_scale_to_bnsd_grouped(
            dk_fp32,
            actual_seq_kv,
            cu_seqlens=cu_seqlens_kv_list,
            q_scale_layout="TND",
            num_kv_heads=N_kv,
        )
        assert dk_bnsd_grouped.shape == (B, N_kv, max_skv, Dg), (
            f"K scale grouped shape mismatch: got {tuple(dk_bnsd_grouped.shape)}, "
            f"expected {(B, N_kv, max_skv, Dg)}"
        )

        # V scale: e8m0 -> fp32 -> TND v-scale grouped -> BNSD grouped
        dv_e8m0 = _to_torch_e8m0(dequant_scale_v)
        _assert_no_e8m0_nan(dv_e8m0, "descale_v")
        dv_fp32 = mxfp8_golden_mod.e8m0_to_fp32(dv_e8m0)
        dv_bnsd_grouped = _tnd_v_scale_to_bnsd_grouped(
            dv_fp32, actual_seq_kv, cu_seqlens=cu_seqlens_kv_list, group_size=group_size
        )
        Sg_max = math.ceil(max_skv / group_size) if max_skv > 0 else 0
        # TND V scale D 维可能因对齐 padding 而大于 logical D, 同 PA 分支处理.
        assert dv_bnsd_grouped.shape[:3] == (B, N_kv, Sg_max), (
            f"V scale grouped first 3 dims mismatch: got {tuple(dv_bnsd_grouped.shape[:3])}, "
            f"expected {(B, N_kv, Sg_max)}"
        )
        assert dv_bnsd_grouped.shape[3] >= D, (
            f"V scale grouped D={dv_bnsd_grouped.shape[3]} < logical D={D}"
        )
        if dv_bnsd_grouped.shape[3] > D:
            dv_bnsd_grouped = dv_bnsd_grouped[:, :, :, :D].contiguous()

    # ----- 调 cpu_mxfp8_golden (BNSD fp8 + group 维 fp32 scale) -----
    cpu_out, cpu_lse = mxfp8_golden_mod.cpu_mxfp8_golden(
        q_fp8_bnsd,
        k_fp8_bnsd,
        v_fp8_bnsd,
        dq_bnsd_grouped,
        dk_bnsd_grouped,
        dv_bnsd_grouped,
        p_scale_value,
        actual_seq_q,
        actual_seq_kv,
        softmax_scale=softmax_scale,
    )

    compare_layout = "TND" if enable_pa else input_layout
    cpu_out_aligned = mxfp8_golden_mod.convert_q_bnsd_to_layout(
        cpu_out,
        actual_seq_q,
        compare_layout,
        cu_seqlens=cu_seqlens_q_list if enable_pa else None,
    )

    if enable_lse:
        cpu_lse_aligned = mxfp8_golden_mod.convert_q_bnsd_to_layout(
            cpu_lse,
            actual_seq_q,
            compare_layout,
            cu_seqlens=cu_seqlens_q_list if enable_pa else None,
        )
        # TND padding 位置填 inf 以匹配 NPU 行为：NPU 对超出实际序列长度的 Q 位置输出 inf LSE
        if enable_pa and cu_seqlens_q_list is not None:
            cpu_lse_aligned = mxfp8_golden_mod.fill_tnd_padding(
                cpu_lse_aligned,
                actual_seq_q,
                cu_seqlens_q_list,
                fill_value=float("inf"),
            )

        if cpu_lse_aligned.ndim == 3 and cpu_lse_aligned.shape[-1] == 1:
            cpu_lse_aligned = cpu_lse_aligned.squeeze(-1).contiguous()
        # NPU LSE 输出已改为 N-major 排布 (N, T): N 在外, T 在内
        # CPU golden 经 convert 后是 [T, N] (T-major), TND case 需转成 [N, T] 对齐
        if compare_layout == "TND" and cpu_lse_aligned.ndim == 2:
            cpu_lse_aligned = cpu_lse_aligned.permute(1, 0).contiguous()
        result = [cpu_out_aligned, cpu_lse_aligned]

    else:
        result = [cpu_out_aligned, None]

    return result


# ==============================================================================
# GQA FP8 全量化 CPU golden (quant_mode=6)
# Q/K: per-token-head, V: per-head, descale=FP32 (非 e8m0, 无 group 维扩展)
# layout_q=NTD, layout_q_descale=NT, layout_kv=PA_BNBD (K cache 含 scale rows),
# layout_out=TND, 仅 PA 模式
# ==============================================================================


def cpu_qfa_gqa_fp8(
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
    """GQA FP8 CPU golden (quant_mode=6, 仅 PA)

    前 8 个入参 slot (0 q, 1 k, 2 v, 3 dequant_scale_q, 4 dequant_scale_k,
    5 dequant_scale_v, 6 p_scale, 7 block_table) 由 inputs.py 写入真实值;
    后 7 个 tensor 参数 (cu_seqlens_q_t .. metadata_t, slot 8-14) 中，
    cu_seqlens/seqused 的真实值由 inputs.py 写入，本函数从 _t 读回 list 使用。
      q: NTD [N,T,D] FP8
      k: PA K cache [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8 (末 K_SCALE_ROWS 行存 FP32 deq_k)
      v: PA V cache [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8
      dequant_scale_q: NT [N,T] FP32
      dequant_scale_k: BNSD [B,N_kv,max_skv,1] FP32 (供本函数直接用, 无需从 K cache 提取)
      dequant_scale_v: [N_kv] FP32
    返回: [cpu_out_aligned, cpu_lse_aligned 或 None]
    """
    if not enable_pa:
        raise NotImplementedError("GQA FP8 (quant_mode=6) 仅支持 PA 模式")

    # csv precision_tolerances 暂存到 fp8_golden_mod 供 compare 读取
    fp8_golden_mod._csv_precision_tolerances = kwargs.get("precision_tolerances")
    fp8_golden_mod._csv_absolute_precision = kwargs.get("absolute_precision")

    # cu_seqlens/seqused 的真实值由 inputs.py in-place 写入 tensor slot (_t)。
    # golden 端 _t 是 CPU tensor，直接读回 list。
    cu_seqlens_q_list = _tolist_t(cu_seqlens_q_t)
    cu_seqlens_kv_list = _tolist_t(cu_seqlens_kv_t)
    seqused_q_list = _tolist_t(seqused_q_t)
    seqused_kv_list = _tolist_t(seqused_kv_t)

    # actual_seq (有效长度)
    actual_seq_q = (
        list(seqused_q_list)
        if (seqused_q_list is not None and len(seqused_q_list) > 0)
        else _cu_seqlens_to_actual(cu_seqlens_q_list)
    )
    actual_seq_kv = (
        list(seqused_kv_list)
        if (seqused_kv_list is not None and len(seqused_kv_list) > 0)
        else _cu_seqlens_to_actual(cu_seqlens_kv_list)
    )

    # batch_size: CSV 原始值 (可为 -1), 透传给 metadata;
    # B: 从 cu_seqlens_q 推导的正整数, 供 shape assert 与 BNSD 逆转换。
    B = (
        max(1, len(cu_seqlens_q_list) - 1)
        if cu_seqlens_q_list and len(cu_seqlens_q_list) >= 2
        else 1
    )

    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    else:
        p_scale_value = float(p_scale_value)

    # 注入 golden 全局变量 (cpu_fp8_fullquant_golden 读 N_q/N_kv/SPARSE_MODE 等)
    _apply_golden_globals(
        {
            "B": B,
            "N_q": N_q,
            "N_kv": N_kv,
            "D": D,
            "cu_seqlens_q": cu_seqlens_q_list,
            "cu_seqlens_kv": cu_seqlens_kv_list,
            "seqused_q": seqused_q_list,
            "seqused_kv": seqused_kv_list,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
            "enable_pa": enable_pa,
            "kv_cache_layout": kv_cache_layout,
            "block_size": block_size,
            "mask_mode": mask_mode,
            "q_scale_layout": q_scale_layout,
            "enable_lse": enable_lse,
            "input_layout": input_layout,
            "is_contiguous": is_contiguous,
            "device_id": device_id,
            "graph_path": graph_path,
            "softmax_scale": softmax_scale,
            "p_scale_value": p_scale_value,
            "QUANT_MODE": 6,
            "SPARSE_MODE": mask_mode,
            "BLOCK_SIZE": block_size,
            "ENABLE_PA": enable_pa,
            "KV_CACHE_LAYOUT": kv_cache_layout,
        },
        quant_mode=6,
    )

    # Q: NTD [N,T,D] -> BNSD [B,N,max_sq,D]
    q_fp8_bnsd = fp8_golden_mod.ntd_to_bnsd_q_gqa(_to_torch_fp8(q), actual_seq_q)

    # deq_q: NT [N,T] -> BNSD [B,N,max_sq,1] FP32
    deq_q_bnsd = fp8_golden_mod.nt_to_bnsd_q_scale_gqa(
        dequant_scale_q.float(), actual_seq_q
    )

    # K/V: PA cache (含 scale rows) -> BNSD + 提取 deq_k
    bt = _to_torch_int32(block_table)
    if bt.numel() == 0 or bt.ndim < 2:
        raise ValueError(
            f"[GOLDEN GQA FP8] enable_pa=True but block_table invalid: shape={tuple(bt.shape)}"
        )
    k_pa = _to_torch_fp8(k)
    v_pa = _to_torch_fp8(v)
    k_fp8_bnsd, v_fp8_bnsd, deq_k_bnsd = fp8_golden_mod.pa_cache_to_bnsd_gqa(
        k_pa, v_pa, bt, actual_seq_kv, block_size
    )
    # deq_k_bnsd: [B,N_kv,max_skv,1] FP32 (从 K cache 提取)
    # deq_v: [N_kv] -> [1,N_kv,1,1] FP32 (cpu golden 内部按 GQA 广播)
    deq_v_bnsd = dequant_scale_v.float().view(1, N_kv, 1, 1).contiguous()

    # 调 cpu_fp8_fullquant_golden (descale 不做 group 维扩展, 内部按 GQA 广播)
    cpu_out, cpu_lse = fp8_golden_mod.cpu_fp8_fullquant_golden(
        q_fp8_bnsd,
        k_fp8_bnsd,
        v_fp8_bnsd,
        deq_q_bnsd,
        deq_k_bnsd,
        deq_v_bnsd,
        torch.tensor([p_scale_value], dtype=torch.float32),
        actual_seq_q,
        actual_seq_kv,
        softmax_scale=softmax_scale,
    )

    # golden 输出 BNSD -> TND (对齐 NPU layout_out=TND)
    cpu_out_aligned = fp8_golden_mod.convert_q_bnsd_to_layout(
        cpu_out,
        actual_seq_q,
        "TND",
        cu_seqlens=cu_seqlens_q_list,
    )

    if enable_lse:
        cpu_lse_aligned = fp8_golden_mod.convert_q_bnsd_to_layout(
            cpu_lse,
            actual_seq_q,
            "TND",
            cu_seqlens=cu_seqlens_q_list,
        )
        # TND padding 位置填 inf 以匹配 NPU 行为
        if cu_seqlens_q_list is not None:
            cpu_lse_aligned = fp8_golden_mod.fill_tnd_padding(
                cpu_lse_aligned,
                actual_seq_q,
                cu_seqlens_q_list,
                fill_value=float("inf"),
            )
        if cpu_lse_aligned.ndim == 3 and cpu_lse_aligned.shape[-1] == 1:
            cpu_lse_aligned = cpu_lse_aligned.squeeze(-1).contiguous()
        if cpu_lse_aligned.ndim == 2:
            cpu_lse_aligned = cpu_lse_aligned.permute(1, 0).contiguous()
        result = [cpu_out_aligned, cpu_lse_aligned]
    else:
        result = [cpu_out_aligned, None]

    return result


def cpu_qfa_hif8(
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
    """HIF8 CPU golden (quant_mode=0, per-tensor, TND, 无 PA)。

    入参 slot 由 inputs.py generate_qfa_hif8_inputs 写入:
      q/k/v: TND uint8 (hifloat8 编码)
      dequant_scale_q/k/v: (1,) float32 per-tensor descale
      p_scale: (1,) float32
    """
    import quant_flash_attn_hif8_golden as hif8_golden_mod

    hif8_golden_mod._csv_precision_tolerances = kwargs.get("precision_tolerances")
    hif8_golden_mod._csv_absolute_precision = kwargs.get("absolute_precision")
    input_layout = layout_q or input_layout

    # cu_seqlens/seqused 的真实值由 inputs.py in-place 写入 tensor slot (_t)。
    cu_seqlens_q = _tolist_t(cu_seqlens_q_t)
    cu_seqlens_kv = _tolist_t(cu_seqlens_kv_t)
    seqused_q = _tolist_t(seqused_q_t)
    seqused_kv = _tolist_t(seqused_kv_t)

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

    p_scale_value = (
        float(p_scale.item()) if isinstance(p_scale, torch.Tensor) else float(p_scale)
    )

    _apply_hif8_golden_globals(
        {
            "B": B,
            "N_q": N_q,
            "N_kv": N_kv,
            "D": D,
            "CU_SEQLENS_Q": cu_seqlens_q,
            "CU_SEQLENS_KV": cu_seqlens_kv,
            "SEQUSED_Q": list(seqused_q) if seqused_q is not None else None,
            "SEQUSED_KV": list(seqused_kv) if seqused_kv is not None else None,
            "MAX_SEQLEN_Q": max_seqlen_q,
            "MAX_SEQLEN_KV": max_seqlen_kv,
            "SPARSE_MODE": mask_mode,
            "Q_SCALE_LAYOUT": q_scale_layout,
            "INPUT_LAYOUT": input_layout,
            "P_SCALE": p_scale_value,
            "ENABLE_LSE": enable_lse,
            "GRAPH_PATH": graph_path,
            "SOFTMAX_SCALE": softmax_scale,
        }
    )

    deq_q = (
        dequant_scale_q
        if isinstance(dequant_scale_q, torch.Tensor)
        else torch.tensor([float(dequant_scale_q)], dtype=torch.float32)
    )
    deq_k = (
        dequant_scale_k
        if isinstance(dequant_scale_k, torch.Tensor)
        else torch.tensor([float(dequant_scale_k)], dtype=torch.float32)
    )
    deq_v = (
        dequant_scale_v
        if isinstance(dequant_scale_v, torch.Tensor)
        else torch.tensor([float(dequant_scale_v)], dtype=torch.float32)
    )
    p_scale_t = (
        p_scale
        if isinstance(p_scale, torch.Tensor)
        else torch.tensor([float(p_scale)], dtype=torch.float32)
    )

    q_cpu = q.detach().cpu() if isinstance(q, torch.Tensor) else torch.as_tensor(q)
    k_cpu = k.detach().cpu() if isinstance(k, torch.Tensor) else torch.as_tensor(k)
    v_cpu = v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v)

    # TTK slot dtype is float8_e4m3fn, hif8 golden expects uint8 (hifloat8 encoded)
    if str(q_cpu.dtype) == "torch.float8_e4m3fn":
        q_cpu = q_cpu.view(torch.uint8)
    if str(k_cpu.dtype) == "torch.float8_e4m3fn":
        k_cpu = k_cpu.view(torch.uint8)
    if str(v_cpu.dtype) == "torch.float8_e4m3fn":
        v_cpu = v_cpu.view(torch.uint8)

    # TTK slot is in input_layout format, cpu_hif8_golden expects BNSD 4D
    cu_seqlens_q_list = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv_list = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]

    def _tnd_to_bnsd(tensor_tnd, seq_lens, cu_seqlens=None):
        B = len(seq_lens)
        N = tensor_tnd.shape[1]
        D = tensor_tnd.shape[2]
        max_seq = max(seq_lens) if seq_lens else 0
        result = torch.zeros(
            (B, N, max_seq, D), dtype=tensor_tnd.dtype, device=tensor_tnd.device
        )
        if cu_seqlens is not None:
            for b in range(B):
                act_s = seq_lens[b]
                if act_s <= 0:
                    continue
                offset = cu_seqlens[b]
                result[b, :, :act_s, :] = tensor_tnd[
                    offset : offset + act_s, :, :
                ].permute(1, 0, 2)
        else:
            t = 0
            for b in range(B):
                act_s = seq_lens[b]
                result[b, :, :act_s, :] = tensor_tnd[t : t + act_s, :, :].permute(
                    1, 0, 2
                )
                t += act_s
        return result.contiguous()

    def _bsnd_to_bnsd(tensor_bsnd, seq_lens):
        B = tensor_bsnd.shape[0]
        N = tensor_bsnd.shape[2]
        D = tensor_bsnd.shape[3]
        max_seq = max(seq_lens) if seq_lens else 0
        result = torch.zeros(
            (B, N, max_seq, D), dtype=tensor_bsnd.dtype, device=tensor_bsnd.device
        )
        for b in range(B):
            act_s = seq_lens[b]
            if act_s <= 0:
                continue
            result[b, :, :act_s, :] = tensor_bsnd[b, :act_s, :, :].permute(1, 0, 2)
        return result.contiguous()

    def _bnsd_to_bnsd(tensor_bnsd, seq_lens):
        B = tensor_bnsd.shape[0]
        N = tensor_bnsd.shape[1]
        D = tensor_bnsd.shape[3]
        max_seq = max(seq_lens) if seq_lens else 0
        result = torch.zeros(
            (B, N, max_seq, D), dtype=tensor_bnsd.dtype, device=tensor_bnsd.device
        )
        for b in range(B):
            act_s = seq_lens[b]
            if act_s <= 0:
                continue
            result[b, :, :act_s, :] = tensor_bnsd[b, :, :act_s, :]
        return result.contiguous()

    if str(input_layout).upper() == "BSND":
        q_bnsd = _bsnd_to_bnsd(q_cpu, actual_seq_q)
        k_bnsd = _bsnd_to_bnsd(k_cpu, actual_seq_kv)
        v_bnsd = _bsnd_to_bnsd(v_cpu, actual_seq_kv)
    elif str(input_layout).upper() == "BNSD":
        q_bnsd = _bnsd_to_bnsd(q_cpu, actual_seq_q)
        k_bnsd = _bnsd_to_bnsd(k_cpu, actual_seq_kv)
        v_bnsd = _bnsd_to_bnsd(v_cpu, actual_seq_kv)
    else:
        q_bnsd = _tnd_to_bnsd(
            q_cpu,
            actual_seq_q,
            cu_seqlens_q_list if len(cu_seqlens_q_list) > 1 else None,
        )
        k_bnsd = _tnd_to_bnsd(
            k_cpu,
            actual_seq_kv,
            cu_seqlens_kv_list if len(cu_seqlens_kv_list) > 1 else None,
        )
        v_bnsd = _tnd_to_bnsd(
            v_cpu,
            actual_seq_kv,
            cu_seqlens_kv_list if len(cu_seqlens_kv_list) > 1 else None,
        )

    cpu_out, cpu_lse = hif8_golden_mod.cpu_hif8_golden(
        q_bnsd,
        k_bnsd,
        v_bnsd,
        deq_q,
        deq_k,
        deq_v,
        p_scale_t,
        actual_seq_q,
        actual_seq_kv,
        softmax_scale=softmax_scale,
    )

    compare_layout = input_layout
    cu_q_for_convert = (
        cu_seqlens_q
        if (str(input_layout).upper() == "TND" and len(cu_seqlens_q_list) > 1)
        else None
    )
    cpu_out_aligned = hif8_golden_mod.convert_q_bnsd_to_layout(
        cpu_out,
        actual_seq_q,
        compare_layout,
        cu_seqlens=cu_q_for_convert,
    )

    if enable_lse:
        lse_compare_layout = "TND" if str(compare_layout).upper() == "TND" else "BNSD"
        lse_cu_q = cu_q_for_convert if lse_compare_layout == "TND" else None
        cpu_lse_aligned = hif8_golden_mod.convert_q_bnsd_to_layout(
            cpu_lse,
            actual_seq_q,
            lse_compare_layout,
            cu_seqlens=lse_cu_q,
        )
        # TND padding 位置填 inf 以匹配 NPU 行为：NPU 对超出实际序列长度的 Q 位置输出 inf LSE
        if lse_compare_layout == "TND" and lse_cu_q is not None:
            cpu_lse_aligned = hif8_golden_mod.fill_tnd_padding(
                cpu_lse_aligned,
                actual_seq_q,
                list(lse_cu_q),
                fill_value=float("inf"),
            )
        while cpu_lse_aligned.ndim > 1 and cpu_lse_aligned.shape[-1] == 1:
            cpu_lse_aligned = cpu_lse_aligned.squeeze(-1).contiguous()
        if compare_layout == "TND" and cpu_lse_aligned.ndim == 2:
            cpu_lse_aligned = cpu_lse_aligned.permute(1, 0).contiguous()
        result = [cpu_out_aligned, cpu_lse_aligned]
    else:
        result = [cpu_out_aligned, None]

    return result


def _apply_hif8_golden_globals(attrs):
    """把 case 属性注入 hif8 golden 模块全局变量。"""
    import quant_flash_attn_hif8_golden as hif8_golden_mod

    for golden_key, val in attrs.items():
        setattr(hif8_golden_mod, golden_key, val)
