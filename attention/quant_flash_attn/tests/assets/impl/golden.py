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
import quant_flash_attn_hifp4_golden as hifp4_golden_mod

logger = logging.getLogger(__name__)


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


def _to_torch_bf16(t):
    """CSV slot → bf16 torch tensor (CPU)。numpy 侧 ml_dtypes.bfloat16 经 uint8
    视图位级还原 (numpy 原生不支持 bfloat16); 空/None → None。"""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        if t.numel() == 0:
            return None
        return t if t.dtype == torch.bfloat16 else t.to(torch.bfloat16)
    arr = numpy.asarray(t)
    if arr.size == 0:
        return None
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.view(numpy.uint8)).view(torch.bfloat16)
    if arr.dtype == numpy.uint16:
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    return torch.from_numpy(arr).to(torch.bfloat16)


def _reverse_vtail_to_bnsd(v_tail_bf16, kv_cache_layout):
    """v_tail slot (final layout, 与主KV cache同布局) → (B, N_kv, Bs, D) BNSD bf16。

    前向变换 (inputs.py): BNSD --reshape(B,N2,Bs,D/16,16)--permute(0,1,3,2,4)--> PA_NZ;
    逆变换: permute(0,1,3,2,4) (交换回 Bs/Df 轴) --reshape--> BNSD, D 序 = f*16+j 还原。
    """
    layout_up = str(kv_cache_layout or "").upper()
    if layout_up in ("BBSND", "PA_BBND"):
        return v_tail_bf16.transpose(1, 2).contiguous()  # (B,Bs,N2,D) → (B,N2,Bs,D)
    if layout_up in ("PA_NZ",):
        Bn, N2, Df, Bs, inner = v_tail_bf16.shape  # (B,N2,D/16,Bs,16)
        return (
            v_tail_bf16.permute(0, 1, 3, 2, 4)
            .reshape(Bn, N2, Bs, Df * inner)
            .contiguous()
        )
    # BNSD / BNNBSD / PA_BNBD (BnNBsD): 布局即 BNSD
    return v_tail_bf16.contiguous()


def _extract_vtail_bundle(
    v_tail, block_table_tail, seqused_v_tail, enable_pa, kv_cache_layout
):
    """从 v_tail 三件套 slot 提取 CPU 参考注入所需的 (v_tail_bnsd, tail_lens)。

    gate (与 NPU 侧 vtail_slots_to_npu_args 一致): 空 slot / 全零尾长 → (None, None)。
    非 PA + 非空 slot → 报错 (v_tail PA 专属, checker 契约)。
    """
    v_tail_bf16 = _to_torch_bf16(v_tail)
    if v_tail_bf16 is None:
        return None, None
    if not enable_pa:
        raise ValueError(
            "[GOLDEN] v_tail slot 非空但 enable_pa=False — v_tail 仅支持 PA 场景 (checker 契约)"
        )
    skt = _to_torch_int32(seqused_v_tail) if seqused_v_tail is not None else None
    if skt is None or skt.numel() == 0:
        raise ValueError(
            "[GOLDEN] v_tail slot 非空但 seqused_v_tail slot 为空 — CSV slot 不一致"
        )
    tail_lens = [int(x) for x in skt.flatten().tolist()]
    if all(t == 0 for t in tail_lens):
        return None, None
    v_tail_bnsd = _reverse_vtail_to_bnsd(v_tail_bf16, kv_cache_layout)
    return v_tail_bnsd, tail_lens


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


def _data_layout_to_bnsd(tensor_in, seq_lens, layout, cu_seqlens=None):
    """Q/K/V 数据按 input_layout 逆转换回 BNSD (TND/BNSD/BSND)"""
    layout = str(layout or "TND").upper()
    if layout == "BNSD":
        tensor = (
            tensor_in
            if isinstance(tensor_in, torch.Tensor)
            else torch.as_tensor(tensor_in)
        )
        # 保留 slot 完整 S 维 (padding 上界, 可大于 max(seq_lens), 超出 seqused
        # 的行在 cpu golden 中被 mask 输出 0); 正常用例 S == max_seq 等价。
        return tensor.contiguous()
    if layout == "BSND":
        # (B, S, N, D) -> (B, N, S, D)
        tensor = (
            tensor_in
            if isinstance(tensor_in, torch.Tensor)
            else torch.as_tensor(tensor_in)
        )
        return tensor.permute(0, 2, 1, 3).contiguous()
    return _tnd_to_bnsd_fixed(tensor_in, seq_lens, cu_seqlens=cu_seqlens)


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
    if layout == "BNSD":
        # (B, N, S, Dg//2, 2) -> (B, N, S, Dg); 保留完整 S 维 (slot S 为 padding
        # 上界, 可大于 max(seq_lens)), 正常用例 S == max_seq 等价。
        B, N, S, Dg_half, _pair = scale_tnd_packed.shape
        return scale_tnd_packed.reshape(B, N, -1, Dg_half * 2).contiguous()
    if layout == "BSND":
        # (B, S, N, Dg//2, 2) -> (B, N, S, Dg)
        B, S, N, Dg_half, _pair = scale_tnd_packed.shape
        bnsd_packed = scale_tnd_packed.permute(0, 2, 1, 3, 4)
        return bnsd_packed.reshape(B, N, -1, Dg_half * 2).contiguous()
    if layout == "BN2SGD":
        # (B, N2, S1, G, Dg//2, 2) -> (B, N_q, S1, Dg), 内存序 B-N2-S1-G-D
        if num_kv_heads is None:
            raise ValueError("num_kv_heads required for BN2SGD layout")
        B, N2, S1, G, Dg_half, _pair = scale_tnd_packed.shape
        bnsd_packed = scale_tnd_packed.permute(0, 1, 3, 2, 4, 5).reshape(
            B, N2 * G, S1, Dg_half, 2
        )
        return bnsd_packed.reshape(B, N2 * G, S1, Dg_half * 2).contiguous()
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


def _v_scale_layout_to_bnsd_grouped(scale_packed, seq_lens, layout, group_size=32):
    """V scale 按 input_layout 逆转换回 BNSD grouped (B, N, Sg, D)"""
    layout = str(layout or "TND").upper()
    if layout in ("BNSD", "BSND"):
        if layout == "BSND":
            # (B, S_out, N, D, 2) -> (B, N, S_out, D, 2)
            scale_packed = scale_packed.permute(0, 2, 1, 3, 4).contiguous()
        B, N, S_out, D, _pair = scale_packed.shape
        sg = S_out * 2
        recovered = torch.zeros(
            (B, N, sg, D), dtype=scale_packed.dtype, device=scale_packed.device
        )
        recovered[:, :, 0::2, :] = scale_packed[..., 0]
        recovered[:, :, 1::2, :] = scale_packed[..., 1]
        sg_max = max(math.ceil(s / group_size) for s in seq_lens) if seq_lens else sg
        return recovered[:, :, :sg_max, :].contiguous()
    return _tnd_v_scale_to_bnsd_grouped(scale_packed, seq_lens, group_size=group_size)


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


# ==============================================================================
# 三方 cross_check 支撑: fp64 bench / 分段参考缓存
# 仅 CLI 显式 --compare cross_check 时生成, 默认两方回归不额外计算。
# ==============================================================================


def _bench_enabled():
    """是否生成三方 bench: 仅 CLI 显式 --compare cross_check 时生成。"""
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
_REF_CACHE = {}
_LAST_BENCH = None


def peek_bench(testcase_name=None):
    """取当前用例的 bench 输出; 未生成时返回 None。"""
    if testcase_name is not None:
        return _BENCH_CACHE.get(str(testcase_name))
    return _LAST_BENCH


def peek_sectioned_ref(testcase_name=None):
    """取当前用例的分段量化参考; 未生成时返回 None。"""
    return _REF_CACHE.get(str(testcase_name))


def _convert_cross_check_output(
    convert_mod,
    out_bnsd,
    lse_bnsd,
    actual_seq_q,
    compare_layout,
    cu_out,
    enable_lse,
    to_bf16,
    lse_non_tnd_layout=None,
):
    """把 BNSD 结果转换到算子输出布局(与 golden 转换语义保持一致)。

    to_bf16=True 用于分段参考(对齐算子 bf16 输出); bench 保持 float32。
    """
    out = convert_mod.convert_q_bnsd_to_layout(
        out_bnsd, actual_seq_q, compare_layout, cu_seqlens=cu_out
    )
    if to_bf16:
        out = out.to(torch.bfloat16)

    lse = None
    if enable_lse and lse_bnsd is not None:
        if str(compare_layout).upper() == "TND":
            lse = convert_mod.convert_q_bnsd_to_layout(
                lse_bnsd, actual_seq_q, "TND", cu_seqlens=cu_out
            )
            if cu_out is not None:
                lse = convert_mod.fill_tnd_padding(
                    lse, actual_seq_q, cu_out, fill_value=float("inf")
                )
            if lse.ndim == 3 and lse.shape[-1] == 1:
                lse = lse.squeeze(-1).contiguous()
            if lse.ndim == 2:
                lse = lse.permute(1, 0).contiguous()
        elif lse_non_tnd_layout is None:
            lse = lse_bnsd
            if lse.ndim == 3 and lse.shape[-1] == 1:
                lse = lse.squeeze(-1).contiguous()
        else:
            lse = convert_mod.convert_q_bnsd_to_layout(
                lse_bnsd, actual_seq_q, lse_non_tnd_layout
            )
            if lse.ndim > 1 and lse.shape[-1] == 1:
                lse = lse.squeeze(-1).contiguous()
    return out, lse


def _cache_cross_check(
    testcase_name,
    convert_mod,
    bench_fn,
    ref_fn,
    golden_args,
    golden_kwargs,
    actual_seq_q,
    compare_layout,
    cu_out,
    enable_lse,
    lse_non_tnd_layout=None,
):
    """生成并缓存当前用例的 fp64 bench 与分段参考, 供 compare 侧三方比对读取。"""
    global _LAST_BENCH
    name = str(testcase_name)
    _BENCH_CACHE.pop(name, None)
    _REF_CACHE.pop(name, None)
    _LAST_BENCH = None
    if not _bench_enabled():
        return
    try:
        bench_out, bench_lse = bench_fn(*golden_args, **golden_kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("[%s] cross_check bench failed: %s", name, exc)
        return
    bench_res, bench_lse_c = _convert_cross_check_output(
        convert_mod,
        bench_out,
        bench_lse,
        actual_seq_q,
        compare_layout,
        cu_out,
        enable_lse,
        to_bf16=False,
        lse_non_tnd_layout=lse_non_tnd_layout,
    )
    _LAST_BENCH = [bench_res, bench_lse_c]
    _BENCH_CACHE[name] = _LAST_BENCH
    logger.info(
        "[%s] bench(fp64): result=%s lse=%s",
        name,
        tuple(bench_res.shape),
        None if bench_lse_c is None else tuple(bench_lse_c.shape),
    )

    if ref_fn is None:
        return
    try:
        ref_out, ref_lse = ref_fn(*golden_args, **golden_kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("[%s] sectioned reference failed: %s", name, exc)
        return
    ref_res, ref_lse_c = _convert_cross_check_output(
        convert_mod,
        ref_out,
        ref_lse,
        actual_seq_q,
        compare_layout,
        cu_out,
        enable_lse,
        to_bf16=True,
        lse_non_tnd_layout=lse_non_tnd_layout,
    )
    _REF_CACHE[name] = [ref_res, ref_lse_c]
    logger.info("[%s] sectioned ref: result=%s", name, tuple(ref_res.shape))


def cpu_qfa_mxfp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    quant_mode: int,
    block_table: torch.Tensor,
    p_scale: torch.Tensor,
    cu_seqlens_q_t: torch.Tensor,
    cu_seqlens_kv_t: torch.Tensor,
    seqused_q_t: torch.Tensor,
    seqused_kv_t: torch.Tensor,
    sinks_t: torch.Tensor,
    attn_mask_t: torch.Tensor,
    metadata_t: torch.Tensor,
    v_tail: torch.Tensor = None,
    block_table_tail: torch.Tensor = None,
    seqused_v_tail: torch.Tensor = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_q_descale: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    **kwargs,
):
    # —— op-schema 直调: 额外适配参数经 kwargs 传入 ——
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    D_v = kwargs.get("head_dim_v") or D
    enable_pa = bool(kwargs.get("enable_pa", False))
    kv_cache_layout = kwargs.get("kv_cache_layout") or layout_kv
    block_size = kwargs.get("block_size", 0)
    q_scale_layout = kwargs.get("q_scale_layout") or layout_q_descale
    # 数据布局解析: 优先 input_layout 属性 (pytest 直跑路径); CSV/TTK 路径无该
    # 属性, 按算子真实 layout 参数解析 — Q 用 layout_q, 非 PA 的 K/V 用 layout_kv
    # (PA 时 layout_kv 是 cache 布局 PA_*), 与 impl/inputs.py 生成侧保持一致。
    # 老 paramset CSV layout_q/layout_kv 恒为 "TND", 解析结果与原默认一致。
    _data_layouts = ("TND", "BNSD", "BSND")
    _il_attr = str(kwargs.get("input_layout") or "").upper()
    if _il_attr in _data_layouts:
        input_layout = q_data_layout = kv_data_layout = _il_attr
    else:
        _lq_attr = str(layout_q or "").upper()
        _lk_attr = str(layout_kv or "").upper()
        q_data_layout = _lq_attr if _lq_attr in _data_layouts else "TND"
        kv_data_layout = _lk_attr if _lk_attr in _data_layouts else "TND"
        input_layout = q_data_layout
    is_contiguous = kwargs.get("is_contiguous", True)
    device_id = kwargs.get("device_id", 0)
    graph_path = kwargs.get("graph_path", 0)
    enable_lse = return_softmax_lse
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
    # B: cu_seqlens_q (len-1) → seqused_q (len) → batch_size 属性 → 1。
    # 与 impl/inputs.py 一致: cu_seqlens 占位 [0] (BNSD/BSND 用例不提供) 时
    # 回退 seqused_q 长度; 非 varlen 的 BNSD/BSND decode 用例两者都不提供,
    # batch 只能由 batch_size 属性给出, 否则 B=1 与 NPU 实际 batch 不匹配。
    _bs_attr = kwargs.get("batch_size")
    if cu_seqlens_q_list and len(cu_seqlens_q_list) >= 2:
        B = len(cu_seqlens_q_list) - 1
    elif seqused_q_list is not None and len(seqused_q_list) > 0:
        B = len(seqused_q_list)
    elif _bs_attr is not None and int(_bs_attr) > 0:
        B = int(_bs_attr)
    else:
        B = 1

    # 非 varlen (BNSD/BSND decode) 场景 cu_seqlens/seqused 均为空, actual_seq
    # 按 max_seqlen 补齐到 B 长度 — 与 metadata 的 max_seqlen 口径一致。
    if not actual_seq_q:
        actual_seq_q = [max_seqlen_q] * B if max_seqlen_q > 0 else []
    if not actual_seq_kv:
        actual_seq_kv = [max_seqlen_kv] * B if max_seqlen_kv > 0 else []

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
            "head_dim_v": D_v,
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

    # actual_seq 推导 max_skv (用于 KV assertion 和 PA 逆转换的 scatter 范围);
    # Q 侧 S 维用 q 逆转换后的实际 Sq_full (含 padding 区), 见下方。
    max_skv = max(actual_seq_kv) if actual_seq_kv else D
    # D=64→2, D=72→4 D=128→4.
    _num_groups = math.ceil(D / group_size)
    Dg = _num_groups + (_num_groups % 2)

    # Q: 按解析的 Q 数据布局逆转换 -> BNSD (PA 不再固定 TND — 算子支持
    # PA + layout_q=BNSD/BSND, 见 layout redline; 老 CSV PA 用例 layout_q="TND"
    # 行为不变)
    q_fp8_tnd = _to_torch_fp8(q)
    q_fp8_bnsd = _data_layout_to_bnsd(
        q_fp8_tnd, actual_seq_q, q_data_layout, cu_seqlens=cu_seqlens_q_list
    )
    # Q 的完整 S 维 (含 padding 区, slot S 可大于 max(seqused_q)) — descale
    # 逆转换与输出 shape 均以此为准, 输出与 q slot S 一致。
    Sq_full = int(q_fp8_bnsd.shape[2])

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
    assert dq_bnsd_grouped.shape == (B, N_q, Sq_full, Dg), (
        f"Q scale grouped shape mismatch: got {tuple(dq_bnsd_grouped.shape)}, "
        f"expected {(B, N_q, Sq_full, Dg)}"
    )

    # K/V: PA paged / TND -> BNSD
    if enable_pa:
        bt = _to_torch_int32(block_table)
        if bt.numel() == 0 or bt.ndim < 2:
            raise ValueError(
                f"[GOLDEN] enable_pa=True but block_table invalid: shape={tuple(bt.shape)}"
            )

        k_pa = _to_torch_fp8(k)
        # 测试脚本不对 cache 形状做合法性校验: K cache 为空 (dim0=0) 时无法反查
        # block, K 主体用占位, 合法性由算子侧拦截 (空 K 属非法输入)。
        k_cache_empty = k_pa.ndim < 4 or int(k_pa.shape[0]) == 0
        if k_cache_empty:
            logger.warning(
                "[GOLDEN] PA K cache dim0=0 — 主体 K/scale 用占位, 合法性由算子侧"
                "校验 (k slot=%s)",
                tuple(k_pa.shape),
            )
            k_fp8_bnsd = torch.zeros((B, N_kv, max_skv, D), dtype=torch.float8_e4m3fn)
        else:
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
        # v_tail 场景下 V cache 可以为空: 每 batch 主体段长度 align_down(kvLen,64)
        # 为 0 时 (kvLen<64, 真实 V 值全部在 v_tail 里), cache slot dim0 合法为 0,
        # 此时没有可反查的物理 block, 主体段用同形占位 (内容会被尾段注入覆盖)。
        v_cache_empty = v_pa.ndim < 4 or int(v_pa.shape[0]) == 0
        if v_cache_empty or k_cache_empty:
            logger.warning(
                "[GOLDEN] PA V cache 用占位 (v_tail 主体段为 0 或 K cache 为空) — "
                "真实值由 v_tail 注入 (v slot=%s, k_cache_empty=%s)",
                tuple(v_pa.shape),
                k_cache_empty,
            )
            v_fp8_bnsd = torch.zeros_like(k_fp8_bnsd)
        else:
            v_cache = _reverse_layout_to_cache(
                v_pa, kv_cache_layout, is_scale=False, is_vscale=False
            )
            v_fp8_bnsd = mxfp8_golden_mod.pa_to_bnsd_data(
                v_cache, actual_seq_kv, block_size, bt, kv_layout=kv_cache_layout
            )
            if v_fp8_bnsd.shape[-1] > D_v:
                v_fp8_bnsd = v_fp8_bnsd[:, :, :, :D_v].contiguous()

        # K scale: e8m0 -> fp32 -> reverse layout -> out_cache (Bn,N,Bs,Dg//2,2) -> grouped BNSD
        if k_cache_empty:
            dk_bnsd_grouped = torch.ones((B, N_kv, max_skv, Dg), dtype=torch.float32)
        else:
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
        # V cache 为空 (v_tail 主体段为 0) 时没有 cache scale 可反查: slot 也可能
        # 缺失 (None) 或为 (0,), 直接用 scale=1 占位 — 尾段注入会把覆盖区间
        # 的 dequant_scale_v 置 1.0, 与 kernel 尾块 scale 恒 1 的语义一致。
        if v_cache_empty or k_cache_empty:
            Sg_max = math.ceil(max_skv / group_size) if max_skv > 0 else 0
            dv_bnsd_grouped = torch.ones((B, N_kv, Sg_max, D), dtype=torch.float32)
        else:
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
            assert dv_bnsd_grouped.shape[3] >= D_v, (
                f"V scale grouped D_v={dv_bnsd_grouped.shape[3]} < logical D_v={D_v}"
            )
            if dv_bnsd_grouped.shape[3] > D_v:
                dv_bnsd_grouped = dv_bnsd_grouped[:, :, :, :D_v].contiguous()
    else:
        # 非 PA: K/V 按解析的 KV 数据布局逆转换 -> BNSD
        k_fp8_tnd = _to_torch_fp8(k)
        k_fp8_bnsd = _data_layout_to_bnsd(
            k_fp8_tnd, actual_seq_kv, kv_data_layout, cu_seqlens=cu_seqlens_kv_list
        )
        if k_fp8_bnsd.shape[-1] > D:
            k_fp8_bnsd = k_fp8_bnsd[:, :, :, :D].contiguous()
        v_fp8_tnd = _to_torch_fp8(v)
        v_fp8_bnsd = _data_layout_to_bnsd(
            v_fp8_tnd, actual_seq_kv, kv_data_layout, cu_seqlens=cu_seqlens_kv_list
        )
        if v_fp8_bnsd.shape[-1] > D_v:
            v_fp8_bnsd = v_fp8_bnsd[:, :, :, :D_v].contiguous()

        # K scale: e8m0 -> fp32 -> 按 KV 数据布局 grouped -> BNSD grouped
        dk_e8m0 = _to_torch_e8m0(dequant_scale_k)
        _assert_no_e8m0_nan(dk_e8m0, "descale_k")
        dk_fp32 = mxfp8_golden_mod.e8m0_to_fp32(dk_e8m0)
        dk_bnsd_grouped = _tnd_qk_scale_to_bnsd_grouped(
            dk_fp32,
            actual_seq_kv,
            cu_seqlens=cu_seqlens_kv_list,
            q_scale_layout=kv_data_layout,
            num_kv_heads=N_kv,
        )
        assert dk_bnsd_grouped.shape == (B, N_kv, max_skv, Dg), (
            f"K scale grouped shape mismatch: got {tuple(dk_bnsd_grouped.shape)}, "
            f"expected {(B, N_kv, max_skv, Dg)}"
        )

        # V scale: e8m0 -> fp32 -> 按 KV 数据布局 v-scale grouped -> BNSD grouped
        dv_e8m0 = _to_torch_e8m0(dequant_scale_v)
        _assert_no_e8m0_nan(dv_e8m0, "descale_v")
        dv_fp32 = mxfp8_golden_mod.e8m0_to_fp32(dv_e8m0)
        dv_bnsd_grouped = _v_scale_layout_to_bnsd_grouped(
            dv_fp32, actual_seq_kv, kv_data_layout, group_size=group_size
        )
        Sg_max = math.ceil(max_skv / group_size) if max_skv > 0 else 0
        # TND V scale D 维可能因对齐 padding 而大于 logical D, 同 PA 分支处理.
        assert dv_bnsd_grouped.shape[:3] == (B, N_kv, Sg_max), (
            f"V scale grouped first 3 dims mismatch: got {tuple(dv_bnsd_grouped.shape[:3])}, "
            f"expected {(B, N_kv, Sg_max)}"
        )
        assert dv_bnsd_grouped.shape[3] >= D_v, (
            f"V scale grouped D_v={dv_bnsd_grouped.shape[3]} < logical D_v={D_v}"
        )
        if dv_bnsd_grouped.shape[3] > D_v:
            dv_bnsd_grouped = dv_bnsd_grouped[:, :, :, :D_v].contiguous()

    # ----- v_tail 尾块高精窗口: 从 slot 反算 BNSD bf16 + 尾长 (slot 驱动 gate) -----
    vtail_bnsd, vtail_tail_lens = _extract_vtail_bundle(
        v_tail, block_table_tail, seqused_v_tail, enable_pa, kv_cache_layout
    )

    # ----- 调 cpu_mxfp8_golden (BNSD fp8 + group 维 fp32 scale) -----
    # v_tail_bnsd: (B,N_kv,block_size,D) bf16 尾块原值, cpu_mxfp8_golden 内注入
    # 尾段V原值(scale=1) + 尾列P直存bf16 (与 kernel 纯尾task VF bf16 直出对齐)
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
        v_tail_bnsd=vtail_bnsd,
        tail_lens=vtail_tail_lens,
    )

    # 输出布局: 优先算子 layout_out 参数 (redline 用例 BNSD); 老 CSV layout_out
    # 恒为 "TND" 与原行为一致; 无有效值时回退旧逻辑 (PA→TND, 非PA→Q 布局)。
    _lo_attr = str(layout_out or "").upper()
    compare_layout = (
        _lo_attr
        if _lo_attr in _data_layouts
        else ("TND" if enable_pa else input_layout)
    )
    cpu_out_aligned = mxfp8_golden_mod.convert_q_bnsd_to_layout(
        cpu_out,
        actual_seq_q,
        compare_layout,
        cu_seqlens=cu_seqlens_q_list if enable_pa else None,
    )

    if enable_lse:
        # NPU LSE 输出不跟随数据布局: kernel 仅对 TND 做 N-major 重排,
        # BNSD/BSND (以及 layout_out 无效回退的非 TND 场景) 的 LSE 恒为
        # (B, N, S) — 与 Q 布局无关。仅 TND 时按 varlen 转换 + N-major 转置。
        if compare_layout == "TND":
            cpu_lse_aligned = mxfp8_golden_mod.convert_q_bnsd_to_layout(
                cpu_lse,
                actual_seq_q,
                "TND",
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
            if cpu_lse_aligned.ndim == 2:
                cpu_lse_aligned = cpu_lse_aligned.permute(1, 0).contiguous()
        else:
            # BNSD/BSND: LSE 保持 golden 内部 (B, N, S, 1), squeeze 尾轴即可
            cpu_lse_aligned = cpu_lse
            if cpu_lse_aligned.ndim == 3 and cpu_lse_aligned.shape[-1] == 1:
                cpu_lse_aligned = cpu_lse_aligned.squeeze(-1).contiguous()
        result = [cpu_out_aligned, cpu_lse_aligned]

    else:
        # 未开 lse 时不返回 None 占位: TTK 的 compare_with_hooks 要求
        # validate_outputs 全为 VALID 才会调用插件自定义 compare, 而 None golden
        # 会被判为 SUPPRESSED 从而跳过钩子(回落 TTK 原生 cross_check, 无 XPU 即 FAIL)。
        result = [cpu_out_aligned]

    # ----- 三方 cross_check: 缓存 fp64 bench 与固定分段参考 -----
    _cache_cross_check(
        kwargs.get("testcase_name", "unknown"),
        mxfp8_golden_mod,
        mxfp8_golden_mod.cpu_mxfp8_bench,
        mxfp8_golden_mod.cpu_mxfp8_sectioned_golden,
        (
            q_fp8_bnsd,
            k_fp8_bnsd,
            v_fp8_bnsd,
            dq_bnsd_grouped,
            dk_bnsd_grouped,
            dv_bnsd_grouped,
            p_scale_value,
            actual_seq_q,
            actual_seq_kv,
        ),
        {
            "softmax_scale": softmax_scale,
            "v_tail_bnsd": vtail_bnsd,
            "tail_lens": vtail_tail_lens,
        },
        actual_seq_q,
        compare_layout,
        cu_seqlens_q_list if enable_pa else None,
        enable_lse,
    )

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
    quant_mode: int,
    block_table: torch.Tensor,
    p_scale: torch.Tensor,
    cu_seqlens_q_t: torch.Tensor,
    cu_seqlens_kv_t: torch.Tensor,
    seqused_q_t: torch.Tensor,
    seqused_kv_t: torch.Tensor,
    sinks_t: torch.Tensor,
    attn_mask_t: torch.Tensor,
    metadata_t: torch.Tensor,
    v_tail: torch.Tensor = None,
    block_table_tail: torch.Tensor = None,
    seqused_v_tail: torch.Tensor = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_q_descale: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    **kwargs,
):
    # —— op-schema 直调: 额外适配参数经 kwargs 传入 ——
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    enable_pa = bool(kwargs.get("enable_pa", False))
    kv_cache_layout = kwargs.get("kv_cache_layout") or layout_kv
    block_size = kwargs.get("block_size", 0)
    q_scale_layout = kwargs.get("q_scale_layout") or layout_q_descale
    input_layout = kwargs.get("input_layout", "NTD")
    is_contiguous = kwargs.get("is_contiguous", True)
    device_id = kwargs.get("device_id", 0)
    graph_path = kwargs.get("graph_path", 0)
    enable_lse = return_softmax_lse
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
    返回: [cpu_out_aligned] 或 [cpu_out_aligned, cpu_lse_aligned] (仅启用 lse 时含第二项)
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
        # 未开 lse 时省略该输出(不返回 None), 保证 TTK 插件 compare 钩子可被调用
        result = [cpu_out_aligned]

    return result


def cpu_qfa_hif8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    quant_mode: int,
    block_table: torch.Tensor,
    p_scale: torch.Tensor,
    cu_seqlens_q_t: torch.Tensor,
    cu_seqlens_kv_t: torch.Tensor,
    seqused_q_t: torch.Tensor,
    seqused_kv_t: torch.Tensor,
    sinks_t: torch.Tensor,
    attn_mask_t: torch.Tensor,
    metadata_t: torch.Tensor,
    v_tail: torch.Tensor = None,
    block_table_tail: torch.Tensor = None,
    seqused_v_tail: torch.Tensor = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_q_descale: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    **kwargs,
):
    # —— op-schema 直调: 额外适配参数经 kwargs 传入 ——
    batch_size = kwargs.get("batch_size")
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    q_scale_layout = kwargs.get("q_scale_layout") or layout_q_descale
    input_layout = kwargs.get("input_layout", "TND")
    graph_path = kwargs.get("graph_path", 0)
    enable_lse = return_softmax_lse
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
        if compare_layout == "TND":
            # TND: golden LSE 转 TND 后为 (T,N,1), 剥附加尾维得 (T,N),
            # 再翻转成 NPU 的 N-major (N,T); 只剥一次, 避免 N=1 时误剥 N 维
            if cpu_lse_aligned.ndim > 1 and cpu_lse_aligned.shape[-1] == 1:
                cpu_lse_aligned = cpu_lse_aligned.squeeze(-1).contiguous()
            if cpu_lse_aligned.ndim == 2:
                cpu_lse_aligned = cpu_lse_aligned.permute(1, 0).contiguous()
        else:
            # BNSD/BSND: 只剥 golden 附加的尾维 1 ((B,N,S,1)->(B,N,S)),
            # S=1 时保留 (B,N,1) 与 NPU infershape 输出一致
            if cpu_lse_aligned.ndim > 1 and cpu_lse_aligned.shape[-1] == 1:
                cpu_lse_aligned = cpu_lse_aligned.squeeze(-1).contiguous()
        result = [cpu_out_aligned, cpu_lse_aligned]
    else:
        # 未开 lse 时省略该输出(不返回 None), 保证 TTK 插件 compare 钩子可被调用
        result = [cpu_out_aligned]

    # ----- 三方 cross_check: 缓存 fp64 bench + 分段量化参考 -----
    _cache_cross_check(
        kwargs.get("testcase_name", "unknown"),
        hif8_golden_mod,
        hif8_golden_mod.cpu_hif8_bench,
        hif8_golden_mod.cpu_hif8_sectioned_golden,
        (
            q_bnsd,
            k_bnsd,
            v_bnsd,
            deq_q,
            deq_k,
            deq_v,
            p_scale_t,
            actual_seq_q,
            actual_seq_kv,
        ),
        {"softmax_scale": softmax_scale},
        actual_seq_q,
        compare_layout,
        cu_q_for_convert,
        enable_lse,
        lse_non_tnd_layout="BNSD",
    )

    return result


def _apply_hif8_golden_globals(attrs):
    """把 case 属性注入 hif8 golden 模块全局变量。"""
    import quant_flash_attn_hif8_golden as hif8_golden_mod

    for golden_key, val in attrs.items():
        setattr(hif8_golden_mod, golden_key, val)


# ==============================================================================
# HiFP4 CPU Golden 插件
# ==============================================================================


def _apply_hifp4_golden_globals(attrs):
    """把 case 属性注入 hifp4 golden 模块全局变量 (HiFP4 路径)。"""
    mapping = {
        "B": "B",
        "N_q": "N_q",
        "N_kv": "N_kv",
        "D": "D",
        "enable_pa": "ENABLE_PA",
        "kv_cache_layout": "KV_CACHE_LAYOUT",
        "block_size": "BLOCK_SIZE",
        "mask_mode": "SPARSE_MODE",
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
    }
    for attr_key, golden_key in mapping.items():
        if attr_key in attrs:
            setattr(hifp4_golden_mod, golden_key, attrs[attr_key])
    setattr(hifp4_golden_mod, "HIFP4_GROUP_SIZE", 64)
    p_scale = attrs.get("p_scale_value", 1.0)
    if isinstance(p_scale, torch.Tensor):
        p_scale = float(p_scale.item())
    else:
        p_scale = float(p_scale)
    setattr(hifp4_golden_mod, "P_SCALE", p_scale)


def cpu_qfa_hifp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dequant_scale_q: torch.Tensor,
    dequant_scale_k: torch.Tensor,
    dequant_scale_v: torch.Tensor,
    quant_mode: int,
    block_table: torch.Tensor,
    p_scale: torch.Tensor,
    cu_seqlens_q_t: torch.Tensor,
    cu_seqlens_kv_t: torch.Tensor,
    seqused_q_t: torch.Tensor,
    seqused_kv_t: torch.Tensor,
    sinks_t: torch.Tensor,
    attn_mask_t: torch.Tensor,
    metadata_t: torch.Tensor,
    v_tail: torch.Tensor = None,
    block_table_tail: torch.Tensor = None,
    seqused_v_tail: torch.Tensor = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_q_descale: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    **kwargs,
):
    """HiFP4 CPU golden 插件 (quant_mode=10)。

    与 cpu_qfa_mxfp8 同签名 (op-schema 直调), hifp4 专用处理:
      - q/k/v: hifloat4 (unpacked) → packed uint8
      - descale: float32 (32-bit 三级量化 scale bitcast), 无 e8m0 解码
      - golden: hifp4_golden_mod.cpu_hifp4_golden (numpy FP4 E1M2 + 三级 scale)
      - layout 转换复用 mxfp8_golden_mod.convert_q_bnsd_to_layout
    """
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    D_v = kwargs.get("head_dim_v") or D
    enable_pa = bool(kwargs.get("enable_pa", False))
    kv_cache_layout = kwargs.get("kv_cache_layout") or layout_kv
    block_size = kwargs.get("block_size", 0)
    q_scale_layout = kwargs.get("q_scale_layout") or layout_q_descale
    input_layout = kwargs.get("input_layout", "BNSD")
    is_contiguous = kwargs.get("is_contiguous", True)
    device_id = kwargs.get("device_id", 0)
    graph_path = kwargs.get("graph_path", 0)
    enable_lse = return_softmax_lse
    # csv precision 暂存到 hifp4_golden_mod 供 compare 插件读取
    hifp4_golden_mod._csv_precision_tolerances = kwargs.get("precision_tolerances")
    hifp4_golden_mod._csv_absolute_precision = kwargs.get("absolute_precision")

    # cu_seqlens/seqused 的真实值由 inputs.py in-place 写入 tensor slot (_t)
    cu_seqlens_q_list = _tolist_t(cu_seqlens_q_t)
    cu_seqlens_kv_list = _tolist_t(cu_seqlens_kv_t)
    if isinstance(seqused_q_t, numpy.ndarray):
        seqused_q_t = torch.from_numpy(seqused_q_t)
    seqused_q_list = _tolist_t(seqused_q_t)
    if isinstance(seqused_kv_t, numpy.ndarray):
        seqused_kv_t = torch.from_numpy(seqused_kv_t)
    seqused_kv_list = _tolist_t(seqused_kv_t)

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

    # HiFP4 仅支持 BNSD, 不感知 cu_seqlens_q/kv
    # B 推导优先级: cu_seqlens_q (len-1) → seqused_q (len) → batch_size attr → q.shape[0]
    if cu_seqlens_q_list is not None and len(cu_seqlens_q_list) >= 2:
        B = max(1, len(cu_seqlens_q_list) - 1)
    elif seqused_q_list is not None and len(seqused_q_list) > 0:
        B = len(seqused_q_list)
    else:
        _bs_attr = kwargs.get("batch_size")
        if _bs_attr is not None and int(_bs_attr) > 0:
            B = int(_bs_attr)
        elif hasattr(q, "shape") and q.ndim >= 4:
            B = int(q.shape[0])
        else:
            B = 1

    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    else:
        p_scale_value = float(p_scale_value)

    _apply_hifp4_golden_globals(
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
        }
    )

    # hifloat4 (unpacked) → packed uint8
    q_np = q.numpy() if isinstance(q, torch.Tensor) else numpy.asarray(q)
    k_np = k.numpy() if isinstance(k, torch.Tensor) else numpy.asarray(k)
    v_np = v.numpy() if isinstance(v, torch.Tensor) else numpy.asarray(v)
    q_np = hifp4_golden_mod._pack_hifloat4_to_uint8(q_np)
    k_np = hifp4_golden_mod._pack_hifloat4_to_uint8(k_np)
    v_np = hifp4_golden_mod._pack_hifloat4_to_uint8(v_np)

    # descale: float32 bitcast (32-bit 三级量化 scale), 直接转 numpy
    dq_q_np = (
        dequant_scale_q.numpy()
        if isinstance(dequant_scale_q, torch.Tensor)
        else numpy.asarray(dequant_scale_q)
    )
    dq_k_np = (
        dequant_scale_k.numpy()
        if isinstance(dequant_scale_k, torch.Tensor)
        else numpy.asarray(dequant_scale_k)
    )
    dq_v_np = (
        dequant_scale_v.numpy()
        if isinstance(dequant_scale_v, torch.Tensor)
        else numpy.asarray(dequant_scale_v)
    )

    p_scale_torch = torch.tensor([p_scale_value], dtype=torch.float32)

    # ----- 调 cpu_hifp4_golden (BNSD fp4 + 32bit scale) -----
    cpu_out, cpu_lse = hifp4_golden_mod.cpu_hifp4_golden(
        q_np,
        k_np,
        v_np,
        dq_q_np,
        dq_k_np,
        dq_v_np,
        actual_seq_q,
        actual_seq_kv,
        p_scale=p_scale_torch,
    )

    # cpu_out 已是 torch.bfloat16 tensor; cpu_lse 是 numpy → 转 torch
    if cpu_lse is not None and not isinstance(cpu_lse, torch.Tensor):
        cpu_lse = torch.from_numpy(numpy.ascontiguousarray(cpu_lse)).to(torch.float32)

    compare_layout = "TND" if enable_pa else input_layout
    if compare_layout == "BNSD":
        result = [cpu_out]
        if enable_lse and cpu_lse is not None:
            result = [cpu_out, cpu_lse]
    else:
        cpu_out_aligned = mxfp8_golden_mod.convert_q_bnsd_to_layout(
            cpu_out,
            actual_seq_q,
            compare_layout,
            cu_seqlens=cu_seqlens_q_list if enable_pa else None,
        )
        if enable_lse and cpu_lse is not None:
            cpu_lse_aligned = mxfp8_golden_mod.convert_q_bnsd_to_layout(
                cpu_lse,
                actual_seq_q,
                compare_layout,
                cu_seqlens=cu_seqlens_q_list if enable_pa else None,
            )
            if cpu_lse_aligned.ndim == 3 and cpu_lse_aligned.shape[-1] == 1:
                cpu_lse_aligned = cpu_lse_aligned.squeeze(-1).contiguous()
            if compare_layout == "TND" and cpu_lse_aligned.ndim == 2:
                cpu_lse_aligned = cpu_lse_aligned.permute(1, 0).contiguous()
            result = [cpu_out_aligned, cpu_lse_aligned]
        else:
            # 未开 lse 时省略该输出(不返回 None), 保证 TTK 插件 compare 钩子可被调用
            result = [cpu_out_aligned]

    return result
