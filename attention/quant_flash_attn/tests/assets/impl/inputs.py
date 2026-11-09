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
import quant_flash_attn_hifp4_golden as hifp4_golden_mod

logger = logging.getLogger(__name__)

_SEED_MAP = {"q": 54, "k": 3, "v": 4}


def _empty_like_slot(slot, dtype):
    """按 CSV slot 形状生成同形空张量占位（slot 缺失时退化为长度 0）。

    用于 PA 空 cache 的透传: 测试脚本不做形状合法性校验，只保证脚本自身
    不下标越界，合法性/报错交给算子侧。
    """
    if slot is None:
        return torch.zeros(0, dtype=dtype)
    return torch.zeros(tuple(slot.shape), dtype=dtype)


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


def _inplace_write_bf16(slot, src_bf16, slot_name):
    """bf16 slot 覆写 (v_tail 专用)。

    torch slot: 直接 copy_ (dtype 对齐);
    numpy slot: ml_dtypes.bfloat16 / uint16 走 uint16 位级拷贝 (numpy 不支持
    torch bf16 → numpy 的直接转换, 位级拷贝保证 bit-exact)。
    空 slot (shape (0,)) 且期望非空 → 报错 (CSV 未分配 v_tail slot)。
    """
    expected_shape = tuple(src_bf16.shape)
    if isinstance(slot, torch.Tensor):
        if slot.numel() == 0:
            raise ValueError(
                f"[INPUTS] {slot_name} slot 为空 (0,) 但 enable_v_tail=True — "
                f"CSV tensor_view_shapes 未分配 v_tail slot, 检查 excel_to_csv 推导"
            )
        if tuple(slot.shape) != expected_shape:
            raise ValueError(
                f"[INPUTS] {slot_name} shape mismatch: CSV slot {tuple(slot.shape)} "
                f"!= computed {expected_shape}. "
                f"Check Excel shape vs v_tail layout derivation (B/N_kv/block_size/D)."
            )
        slot.copy_(src_bf16.to(slot.dtype))
        return
    arr = numpy.asarray(slot)
    if arr.size == 0:
        raise ValueError(
            f"[INPUTS] {slot_name} slot 为空 (0,) 但 enable_v_tail=True — "
            f"CSV tensor_view_shapes 未分配 v_tail slot, 检查 excel_to_csv 推导"
        )
    if tuple(arr.shape) != expected_shape:
        raise ValueError(
            f"[INPUTS] {slot_name} shape mismatch: CSV slot {tuple(arr.shape)} "
            f"!= computed {expected_shape}. "
            f"Check Excel shape vs v_tail layout derivation (B/N_kv/block_size/D)."
        )
    src_u16 = src_bf16.contiguous().view(torch.uint16).numpy()
    if arr.dtype == numpy.uint16:
        arr[...] = src_u16
    else:
        # ml_dtypes.bfloat16: 同 itemsize(2B) 的 uint16 视图位级拷贝
        arr.view(numpy.uint16)[...] = src_u16


def _write_causal_mask(slot, mask_mode, mask_shape):
    """mask_mode != 0 时生成上三角 causal mask 写入 attn_mask slot。

    直调 op 后主算子的 attn_mask 直接来自 CSV slot (ttk 随机生成), 需在这里用与
    golden _build_causal_mask 一致的方式重建: triu(ones, diagonal=1)。
    mask_mode == 0 或 slot 为 None 时跳过。
    """
    if mask_mode == 0 or slot is None:
        return
    shape = tuple(int(v) for v in mask_shape) if mask_shape else (2048, 2048)
    causal = torch.triu(torch.ones(shape, dtype=torch.int8), diagonal=1)
    if isinstance(slot, torch.Tensor):
        if tuple(slot.shape) != tuple(causal.shape):
            raise ValueError(
                f"[INPUTS] attn_mask shape mismatch: CSV slot {tuple(slot.shape)} "
                f"!= causal {tuple(causal.shape)}"
            )
        slot.copy_(causal.to(slot.dtype))
    else:
        arr = causal.numpy().astype(numpy.asarray(slot).dtype)
        if tuple(slot.shape) != tuple(arr.shape):
            raise ValueError(
                f"[INPUTS] attn_mask shape mismatch: CSV slot {tuple(slot.shape)} "
                f"!= causal {tuple(arr.shape)}"
            )
        slot[...] = arr


def generate_qfa_mxfp8_inputs(
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
    # —— op-schema 直调: 额外适配参数与真实值 sidecar 经 kwargs 传入 ——
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    D_v = kwargs.get("head_dim_v") or D
    enable_pa = bool(kwargs.get("enable_pa", False))
    kv_cache_layout = kwargs.get("kv_cache_layout") or layout_kv
    block_size = kwargs.get("block_size", 0)
    # PA 的 block_size 物理真值在 K cache slot 上 (按 kv_cache_layout 的 Bs 轴取:
    # PA_BBND/BnBsND → dim1, PA_NZ → dim3, PA_BNBD/BnNBsD → dim2)。CSV 属性与之
    # 不一致时以 slot 为准, 否则 ceil(kvLen/block_size) 会大于 block_table 列数,
    # 填块表时越界 (IndexError: index N is out of bounds for dimension 1)。
    if enable_pa:
        _pa_lay_up = str(kv_cache_layout or "").upper()
        _bs_axis = (
            1
            if _pa_lay_up in ("BNBSND", "PA_BBND", "BBSND")
            else (3 if _pa_lay_up == "PA_NZ" else 2)
        )
        if k is not None and k.ndim > _bs_axis and int(k.shape[_bs_axis]) > 0:
            _slot_bs = int(k.shape[_bs_axis])
            if int(block_size or 0) != _slot_bs:
                logger.warning(
                    "[INPUTS] PA block_size attr=%s != K cache slot dim%d=%d — "
                    "use slot value",
                    block_size,
                    _bs_axis,
                    _slot_bs,
                )
            block_size = _slot_bs
    q_scale_layout = kwargs.get("q_scale_layout") or layout_q_descale
    cu_seqlens_q = kwargs.get("cu_seqlens_q_values")
    cu_seqlens_kv = kwargs.get("cu_seqlens_kv_values")
    seqused_q = kwargs.get("seqused_q_values")
    seqused_kv = kwargs.get("seqused_kv_values")
    # 数据布局解析: 优先 input_layout 属性 (pytest 直跑路径); CSV/TTK 路径无该
    # 属性, 按算子真实 layout 参数解析 — Q 用 layout_q, 非 PA 的 K/V 用 layout_kv
    # (PA 时 layout_kv 是 cache 布局 PA_*, K/V 走 mxfp8_pa_preprocessing 分支)。
    # 老 paramset CSV layout_q/layout_kv 恒为 "TND", 解析结果与原默认一致。
    _data_layouts = ("TND", "BNSD", "BSND")
    _il_attr = str(kwargs.get("input_layout") or "").upper()
    _lq_attr = str(layout_q or "").upper()
    _lk_attr = str(layout_kv or "").upper()
    if _il_attr in _data_layouts:
        input_layout = _il_attr
        q_data_layout = _il_attr
        kv_data_layout = _il_attr
    else:
        q_data_layout = _lq_attr if _lq_attr in _data_layouts else "TND"
        kv_data_layout = _lk_attr if _lk_attr in _data_layouts else "TND"
        input_layout = q_data_layout

    # 异常用例容忍: N_q/N_kv/D 可能被注入非法值 (如 -1)。非法值不能用于数据合成
    # (torch.rand 负维度会裸崩, 掩盖真正要验证的异常属性), 按 CSV slot 形状推导
    # 回退值完成数据生成; 原始非法属性仍由 ttk 透传给 metadata/主算子, 由算子
    # host 侧校验拦截并给出明确报错 (如 "numHeadsQ must be greater than 0")。
    def _positive_or_derive(name, value, slot, head_axis):
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            ivalue = 0
        if ivalue > 0:
            return ivalue
        shape = tuple(int(dim) for dim in slot.shape)
        axis = head_axis if head_axis >= 0 else len(shape) + head_axis
        if not 0 <= axis < len(shape) or shape[axis] <= 0:
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
            shape[axis],
            shape,
        )
        return shape[axis]

    _q_head_axis = 2 if str(layout_q).upper() == "BSND" else 1
    _k_head_axis = 2 if str(layout_kv).upper() == "BSND" else 1
    N_q = _positive_or_derive("N_q", N_q, q, _q_head_axis)
    N_kv = _positive_or_derive("N_kv", N_kv, k, _k_head_axis)
    D = _positive_or_derive("D", D, q, -1)

    # cu_seqlens -> actual_seq (差分还原)
    cu_seqlens_q = list(cu_seqlens_q) if cu_seqlens_q is not None else [0]
    cu_seqlens_kv = list(cu_seqlens_kv) if cu_seqlens_kv is not None else [0]

    # batch_size: CSV 原始值 (可为 -1), 透传给 metadata;
    # B: cu_seqlens_q (len-1) → seqused_q (len) → batch_size 属性 → 1。
    # BNSD/BSND 用例 cu_seqlens 缺失 (占位 [0]) 时回退 seqused_q 长度 —
    # redline PA+BNSD 用例 seqused_q=[4,4,4,4] → B=4, 与 CSV slot 形状一致;
    # 非 varlen 的 BNSD/BSND decode 用例 (如 141B/8Q/3500KV) 两者都不提供,
    # batch 只能由 batch_size 属性给出, 否则 B=1 与 CSV slot 形状不符。
    _bs_attr = kwargs.get("batch_size")
    if len(cu_seqlens_q) >= 2:
        B = len(cu_seqlens_q) - 1
    elif seqused_q is not None and len(seqused_q) > 0:
        B = len(seqused_q)
    elif _bs_attr is not None and int(_bs_attr) > 0:
        B = int(_bs_attr)
    else:
        B = 1

    # Q 与 KV 一致: cu_seqlens 缺失 (BNSD/BSND 用例不提供) 时回退 seqused,
    # 再回退 max_seqlen*B (非 varlen decode), 不能回退 [0] — 否则 Q 生成长度
    # 为 0, 与 CSV slot 形状不匹配。
    if len(cu_seqlens_q) > 1:
        actual_seq_q = [
            cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(len(cu_seqlens_q) - 1)
        ]
    elif seqused_q is not None and len(seqused_q) > 0:
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
    elif seqused_kv is not None and len(seqused_kv) > 0:
        actual_seq_kv = list(seqused_kv)
    elif max_seqlen_kv is not None and max_seqlen_kv > 0:
        actual_seq_kv = [max_seqlen_kv] * B
    else:
        actual_seq_kv = [0]

    max_sq = max(actual_seq_q) if actual_seq_q else D
    # CSV q slot 的 S 维是 padding 权威上界: redline 用例 slot S 可大于
    # max(seqused_q) (如 000022 S=16426 > max=1024)。B/N/D 与属性一致而仅 S
    # 更大时, 按 slot S 生成 padded 数据 (超出 seqused 的行不参与有效计算,
    # golden/NPU 侧 mask 掉), 保证 in-place 写入 shape 匹配、输出 shape 与
    # q slot S 一致; 结构不一致的异常用例仍走 mismatch 跳过路径。
    try:
        _q_slot_shape = tuple(int(_d) for _d in q.shape)
    except (AttributeError, TypeError, ValueError):
        _q_slot_shape = ()
    if q_data_layout == "BNSD" and len(_q_slot_shape) == 4:
        _slot_ok = (
            _q_slot_shape[0] == B and _q_slot_shape[1] == N_q and _q_slot_shape[3] == D
        )
        _slot_sq = _q_slot_shape[2]
    elif q_data_layout == "BSND" and len(_q_slot_shape) == 4:
        _slot_ok = (
            _q_slot_shape[0] == B and _q_slot_shape[2] == N_q and _q_slot_shape[3] == D
        )
        _slot_sq = _q_slot_shape[1]
    else:
        _slot_ok = False
        _slot_sq = 0
    if _slot_ok and _slot_sq > max_sq:
        max_sq = _slot_sq
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
        torch.rand(B, N_kv, max_skv, D_v, dtype=torch.bfloat16) * (rv_max - rv_min)
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
    # 主体段(每batch前align_down(kvLen,64)行)才参与V的量化, 尾段置零不进cache
    # (真值只在v_tail, 模拟写入路径的cache尾块剥离); 尾长=kv%64。
    # v_tail PA专属(checker 契约); PA_NZ 要求 D%16==0 (bf16 32B分形内径16)
    # v_tail 是否生效以 CSV 里 v_tail slot 是否非空为准（Excel 列直接透传，不再有
    # enable_v_tail 属性）；兼容旧 CSV 仍带 enable_v_tail 的情形。
    _vt_shape = tuple(getattr(v_tail, "shape", ()) or ())
    enable_v_tail = bool(kwargs.get("enable_v_tail", False)) or (
        len(_vt_shape) > 0 and 0 not in _vt_shape
    )
    if enable_v_tail:
        if not enable_pa:
            raise ValueError(
                "[INPUTS] enable_v_tail=True 但 enable_pa=False — "
                "v_tail 仅支持 PA (paged KV cache) 场景 (checker 契约)"
            )
        _layout_up = str(kv_cache_layout or "").upper()
        if _layout_up in ("PA_NZ",) and D_v % 16 != 0:
            raise ValueError(
                f"[INPUTS] PA_NZ + D={D} + v_tail 结构性不成立: NZ 的16列分形要求 "
                f"D%16==0 (与 golden bundle / checker / kernel D/16 寻址一致)"
            )
    tail_lens = []
    v_quant_input = v_bf16
    if enable_v_tail:
        v_quant_input = v_bf16.clone()
        for b in range(B):
            main_len = (actual_seq_kv[b] // 64) * 64
            tail_len = actual_seq_kv[b] - main_len
            tail_lens.append(tail_len)
            if tail_len > 0:
                v_quant_input[b, :, main_len : actual_seq_kv[b], :] = 0
        logger.info(
            "[V_TAIL] enable_v_tail=True, actual_seq_kv=%s, tail_lens=%s",
            actual_seq_kv,
            tail_lens,
        )

    quant_scale_v_bnsd = mxfp8_golden_mod.get_mxfp8_per_channel_group_quant_scale(
        v_quant_input, fp8_dtype, group_size
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
            v_quant_input, quant_scale_v_bnsd, group_size
        )
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
    )

    # ----- Step 3: layout 转换 (BNSD -> final layout) -----
    # bf16尾块原值(pad到block_size) + 尾块表 + 每batch尾长。
    # v_tail与主KV cache同布局: PA_BNBD(BnNBsD)=(Bn,N2,Bs,D) /
    # PA_BBND(BnBsND)=(Bn,Bs,N2,D) / PA_NZ=(Bn,N2,D/16,Bs,16)(bf16的32B分形内径16)
    v_tail_final = None
    block_table_tail_final = None
    seqused_v_tail_final = None
    if enable_v_tail:
        tail_capacity = block_size
        v_tail_bnsd_list = []
        for b in range(B):
            main_len = actual_seq_kv[b] - tail_lens[b]
            seg = v_bf16[
                b, :, main_len : actual_seq_kv[b], :
            ]  # (N_kv, tail, D) bf16原值
            # data_range极端值(±fp32_max)时bf16表示溢出为NaN/Inf —
            # clamp到bf16安全范围(finfo), 避免NaN毒化golden(全NaN)与NPU输入
            if torch.isnan(seg.float()).any() or torch.isinf(seg.float()).any():
                seg = (
                    seg.float()
                    .clamp(
                        torch.finfo(torch.bfloat16).min, torch.finfo(torch.bfloat16).max
                    )
                    .to(torch.bfloat16)
                )
            seg_pad = torch.zeros(N_kv, tail_capacity, D_v, dtype=torch.bfloat16)
            if tail_lens[b] > 0:
                seg_pad[:, : tail_lens[b], :] = seg
            v_tail_bnsd_list.append(seg_pad)
        v_tail_bnsd = torch.stack(
            v_tail_bnsd_list, dim=0
        ).contiguous()  # (B,N_kv,Bs,D) bf16
        layout_up = str(kv_cache_layout or "").upper()
        if layout_up in ("BBSND", "PA_BBND"):
            v_tail_final = v_tail_bnsd.transpose(1, 2).contiguous()
        elif layout_up in ("PA_NZ",):
            Bn, N2c, Bsc, Dc = v_tail_bnsd.shape
            v_tail_final = (
                v_tail_bnsd.reshape(Bn, N2c, Bsc, Dc // 16, 16)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )
        else:  # BNSD / BNNBSD / PA_BNBD (BnNBsD)
            v_tail_final = v_tail_bnsd
        block_table_tail_final = torch.arange(B, dtype=torch.int32).view(B, 1)
        seqused_v_tail_final = torch.tensor(tail_lens, dtype=torch.int32)
        logger.info(
            "[V_TAIL] bundle ready: v_tail=%s btt=%s tail=%s",
            tuple(v_tail_final.shape),
            tuple(block_table_tail_final.shape),
            tail_lens,
        )

    # Q 数据布局按解析结果 (PA 不再强制 TND — 算子支持 PA + layout_q=BNSD/BSND,
    # 见 layout redline 用例; 老 CSV PA 用例 layout_q="TND" 行为不变)。
    # cu_seqlens 占位 [0] (CSV 未提供时) 不能传给 TND 转换 — T=cu[-1]=0 会
    # 产生空 tensor, len<2 视为未提供。
    _cu_q_real = cu_seqlens_q if len(cu_seqlens_q) > 1 else None
    q_fp8_final = mxfp8_golden_mod.convert_q_bnsd_to_layout(
        q_fp8_bnsd_f32, actual_seq_q, q_data_layout, cu_seqlens=_cu_q_real
    )
    q_scale_final_layout = mxfp8_golden_mod.convert_q_scale_bnsd_to_layout(
        quant_scale_q_bnsd, actual_seq_q, q_scale_layout, cu_seqlens=_cu_q_real
    )

    if enable_pa:
        # PA 物理 block 数 (Bn) 由 K/V cache 数据 slot 的 dim0 决定
        # (PA_BNBD/PA_BBND/PA_NZ 的 dim0 均为 Bn)。不能再用 descale slot 作为
        # 唯一来源: 该 slot 在 CSV 里可能缺失 (None) 或为空 ((0,)), 直接取
        # shape[0] 会分别触发 AttributeError 与 total_blocks=0 的除零。
        total_blocks_k = int(k.shape[0]) if k is not None and k.ndim >= 4 else 0
        total_blocks_v = int(v.shape[0]) if v is not None and v.ndim >= 4 else 0
        # 测试脚本不对 cache 形状做合法性校验 — 空 cache 直接透传给算子, 由 NPU 侧
        # 校验并报错。这里只按空处理避免脚本自身越界:
        #   - V cache 空是 v_tail 的合法形态: 每 batch 主体段长度 align_down(kvLen,64)
        #     为 0 时 (kvLen<64) 真实 V 全部在 v_tail 里;
        #   - K cache 空属于算子侧应拦截的非法输入。
        k_cache_empty = total_blocks_k <= 0
        v_cache_empty = total_blocks_v <= 0
        if k_cache_empty or v_cache_empty:
            logger.warning(
                "[INPUTS] PA cache slot dim0: k=%d v=%d (k slot=%s, v slot=%s) — "
                "空 cache 直接透传, 合法性由算子侧校验; 本脚本跳过对应 scatter",
                total_blocks_k,
                total_blocks_v,
                tuple(k.shape) if k is not None else None,
                tuple(v.shape) if v is not None else None,
            )
        total_blocks = (
            total_blocks_k if v_cache_empty else min(total_blocks_k, total_blocks_v)
        )

        if k_cache_empty:
            # K cache 为空时无法建立 block 映射: K/V 数据与 scale 全部按 slot 同形
            # 占位原样透传, 由算子侧报错。
            k_fp8_final = _empty_like_slot(k, torch.float8_e4m3fn)
            v_fp8_final = _empty_like_slot(v, torch.float8_e4m3fn)
            k_scale_final_layout = _empty_like_slot(dequant_scale_k, torch.float32)
            v_scale_final_layout = _empty_like_slot(dequant_scale_v, torch.float32)
        else:
            if block_table is None or block_table.size == 0:
                raise ValueError(
                    "[INPUTS] enable_pa=True but block_table CSV shape is (0,) — "
                    "Excel AI column empty for PA row"
                )
            max_blocks = block_table.shape[1] if block_table.ndim >= 2 else 0
            if max_blocks <= 0:
                raise ValueError(
                    f"[INPUTS] PA mode block_table max_blocks={max_blocks} invalid"
                )
            # block_table 列数是每 batch 能表达的最大 block 数 (CSV 口径应等于
            # ceil(max(actual_seq_kv)/block_size))。需求超出列数时无法表达该 batch
            # 的 KV 映射, 直接给出可定位的报错, 而不是填表时裸 IndexError。
            _need_blocks = max(math.ceil(a / block_size) for a in actual_seq_kv)
            if _need_blocks > max_blocks:
                raise ValueError(
                    f"[INPUTS] PA block_table too narrow: need {_need_blocks} blocks "
                    f"per batch (actual_seq_kv={list(actual_seq_kv)}, "
                    f"block_size={block_size}) but block_table slot is "
                    f"{tuple(block_table.shape)} (max_blocks={max_blocks}); "
                    f"check block_table shape / block_size / seqused_kv attrs. "
                    f"k slot={tuple(k.shape)}, v slot={tuple(v.shape)}"
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
            if v_cache_empty:
                # 空 V cache: 没有可 scatter 的物理 block, 直接给与 slot 同形的空张量
                # (V 主体真值全部在 v_tail / seqused_v_tail 里)。
                v_fp8_final = _empty_like_slot(v, torch.float8_e4m3fn)
                v_scale_final_layout = _empty_like_slot(dequant_scale_v, torch.float32)
            else:
                v_fp8_final = mxfp8_golden_mod.mxfp8_pa_preprocessing(
                    v_fp8_bnsd_f32,
                    actual_seq_kv,
                    block_size,
                    bt_real,
                    is_scale=False,
                    kv_layout=kv_cache_layout,
                    total_blocks=total_blocks_v,
                )

            k_scale_final_layout = mxfp8_golden_mod.mxfp8_pa_preprocessing(
                quant_scale_k_bnsd,
                actual_seq_kv,
                block_size,
                bt_real,
                is_scale=True,
                is_vscale=False,
                kv_layout=kv_cache_layout,
                total_blocks=total_blocks_k,
            )
            if not v_cache_empty:
                v_scale_final_layout = mxfp8_golden_mod.mxfp8_pa_preprocessing(
                    quant_scale_v_bnsd,
                    actual_seq_kv,
                    block_size,
                    bt_real,
                    is_scale=True,
                    is_vscale=True,
                    kv_layout=kv_cache_layout,
                    total_blocks=total_blocks_v,
                )
    else:
        # 非 PA 模式: k/v 按解析的 KV 数据布局转换 (TND/BNSD/BSND);
        # cu_seqlens 占位 [0] 视为未提供 (同 Q 侧)。
        _cu_kv_real = cu_seqlens_kv if len(cu_seqlens_kv) > 1 else None
        k_fp8_final = mxfp8_golden_mod.convert_kv_bnsd_to_layout(
            k_fp8_bnsd_f32,
            actual_seq_kv,
            kv_data_layout,
            cu_seqlens=_cu_kv_real,
        )
        v_fp8_final = mxfp8_golden_mod.convert_kv_bnsd_to_layout(
            v_fp8_bnsd_f32,
            actual_seq_kv,
            kv_data_layout,
            cu_seqlens=_cu_kv_real,
        )
        # k/v scale: convert 函数内部做 pack, 输入 UNPACKED 4D scale (B, N, S, D)
        k_scale_final_layout = mxfp8_golden_mod.convert_k_scale_bnsd_to_layout(
            quant_scale_k_bnsd,
            actual_seq_kv,
            kv_data_layout,
            cu_seqlens=_cu_kv_real,
        )
        v_scale_final_layout = mxfp8_golden_mod.convert_v_scale_bnsd_to_layout(
            quant_scale_v_bnsd, actual_seq_kv, kv_data_layout
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

    def _write_slot(slot, src_torch, slot_name):
        """形状一致时原位 bit-exact copy_; 不一致时 WARN 并跳过写入。

        异常用例的 CSV slot 形状常与按属性推导的计算结果不一致 (如 q_descale
        slot 为 N2TGD 形状 (2,32,4,2,2) 而 q_scale_layout='TND' 计算得
        (32,8,2,2), v_descale slot 首维为 0)。裸 copy_ 的 broadcast 报错
        ("The size of tensor a (4) must match the size of tensor b (8)") 会
        掩盖真正注入的异常属性, 且无法定位。跳过写入让非法输入按 CSV 声明的
        形状透传到算子, 由 host 侧 checker 给出明确拦截报错。
        """
        if slot is None:
            # CSV 未分配该 slot (省略 → 主算子收到 None): 不写, 也不崩溃。
            logger.warning(
                "[INPUTS] %s slot is None (CSV omitted) - skip in-place write "
                "(computed %s)",
                slot_name,
                tuple(src_torch.shape),
            )
            return
        if tuple(slot.shape) == tuple(src_torch.shape):
            slot.copy_(src_torch)
            return
        # 形状不一致: 跳过写入并 WARN, 让非法输入按 CSV 原样透传到算子, 由
        # host 侧 checker 拦截 (如 "v_descale has empty dimension at axis 0")。
        # 注: ttk 在算子调用前会从 numpy 底层存储重建 runtime tensor, 插件侧
        # resize_/set_ 改形状不会传导, 只有共享内存的原位数据写入有效, 因此
        # 这里不能也不应改写 slot 形状。
        logger.warning(
            "[INPUTS] %s shape mismatch: CSV slot %s != computed %s - skip "
            "in-place write (keep CSV slot data/shape), let op-side validation "
            "intercept the inconsistent case",
            slot_name,
            tuple(slot.shape),
            tuple(src_torch.shape),
        )

    _write_slot(q, q_fp8_final, "q (slot 0)")
    _write_slot(k, k_fp8_final, "k (slot 1)")
    _write_slot(v, v_fp8_final, "v (slot 2)")
    _write_slot(dequant_scale_q, q_scale_e8m0, "dequant_scale_q (slot 3)")
    _write_slot(dequant_scale_k, k_scale_e8m0, "dequant_scale_k (slot 4)")
    _write_slot(dequant_scale_v, v_scale_e8m0, "dequant_scale_v (slot 5)")

    # ----- p_scale: in-place 写 fp32 -----
    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    if p_scale is not None:
        p_scale[...] = p_scale_value

    if enable_pa:
        # 空 K cache 时没有 block 映射可填, block_table 保持 slot 原值透传,
        # 由算子侧校验报错。
        if not k_cache_empty:
            block_table[...] = bt_real
    elif block_table is not None:
        block_table[...] = 0

    # ----- 新增 slot 8-14: 用 attributes 真实值覆盖 ttk 随机生成的 cu_seqlens/seqused。
    # sinks/metadata (slot 12,14) 无 value，保持 None 语义（CSV shape (0,) → 空 tensor，
    # sink/golden 不消费）；attn_mask (slot 13) 不传值覆盖——mask 由 golden
    # _build_causal_mask() 按 attributes 里 attn_mask_shape 重建。 -----
    _write_int32_list(cu_seqlens_q_t, cu_seqlens_q, "cu_seqlens_q (slot 8)")
    _write_int32_list(cu_seqlens_kv_t, cu_seqlens_kv, "cu_seqlens_kv (slot 9)")
    _write_int32_list(seqused_q_t, seqused_q, "seqused_q (slot 10)")
    _write_int32_list(seqused_kv_t, seqused_kv, "seqused_kv (slot 11)")
    _write_causal_mask(attn_mask_t, mask_mode, kwargs.get("attn_mask_shape"))

    # ----- v_tail 三件套 slot 15-17: 尾块 bf16 原值(final layout) + 尾块表 + 尾长 -----
    if enable_v_tail:
        _inplace_write_bf16(v_tail, v_tail_final, "v_tail (slot 15)")
        _write_int32_list(
            block_table_tail,
            block_table_tail_final.tolist(),
            "block_table_tail (slot 16)",
        )
        _write_int32_list(seqused_v_tail, tail_lens, "seqused_v_tail (slot 17)")

    def _slot_shape(t):
        return tuple(t.shape) if t is not None and hasattr(t, "shape") else None

    logger.info(
        "[INPUTS] in-place wrote fp8 q/k/v (q=%s), e8m0 descale (dq=%s, dk=%s, dv=%s), "
        "fp32 p_scale, int32 block_table (pa=%s)",
        _slot_shape(q),
        _slot_shape(dequant_scale_q),
        _slot_shape(dequant_scale_k),
        _slot_shape(dequant_scale_v),
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
    # —— op-schema 直调: 额外适配参数与真实值 sidecar 经 kwargs 传入 ——
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    enable_pa = bool(kwargs.get("enable_pa", False))
    kv_cache_layout = kwargs.get("kv_cache_layout") or layout_kv
    block_size = kwargs.get("block_size", 0)
    q_scale_layout = kwargs.get("q_scale_layout") or layout_q_descale
    cu_seqlens_q = kwargs.get("cu_seqlens_q_values")
    cu_seqlens_kv = kwargs.get("cu_seqlens_kv_values")
    seqused_q = kwargs.get("seqused_q_values")
    seqused_kv = kwargs.get("seqused_kv_values")
    input_layout = kwargs.get("input_layout", "NTD")
    """GQA FP8 输入生成 (quant_mode=6, 仅 PA)

    输出 slot 约定 (in-place 写入 ttk 分配的 numpy slot):
      q slot:              NTD [N,T,D] FP8_E4M3
      k slot:              PA K cache [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8 (末 K_SCALE_ROWS 行存 FP32 deq_k)
      v slot:              PA V cache [Bn,N_kv,block_size+K_SCALE_ROWS,D] FP8
      dequant_scale_q slot: NT [N,T] FP32
      dequant_scale_k slot: PA 块状 [Bn,N_kv,K_SCALE_ROWS*D//4] FP32 (kernel 要求 dim2 为 scale
                            buffer 容量 128, 与 block_size 无关; NPU/golden 均从 K cache 提取 deq_k)
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
        ("LAYOUT_Q_DESCALE", layout_q_descale),
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
    # kernel 要求 k_descale dim[2] = K_SCALE_ROWS*D//4 (scale buffer 容量, D=128 时为 128),
    # 与 block_size 无关; 有效 scale 打包在区域前段 (bnsd_to_k_cache_gqa flat_scale[:, :valid])
    k_f32 = (
        k_fp8_final.view(torch.uint8)
        .view(k_fp8_final.shape[0], k_fp8_final.shape[1], -1)
        .view(torch.float32)
    )
    _kscale_capacity = fp8_golden_mod.K_SCALE_ROWS * D // 4
    deq_k_slot = k_f32[:, :, -_kscale_capacity:]

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
    _write_causal_mask(attn_mask_t, mask_mode, kwargs.get("attn_mask_shape"))

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
    # —— op-schema 直调: 额外适配参数与真实值 sidecar 经 kwargs 传入 ——
    batch_size = kwargs.get("batch_size")
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    q_scale_layout = kwargs.get("q_scale_layout") or layout_q_descale
    cu_seqlens_q = kwargs.get("cu_seqlens_q_values")
    cu_seqlens_kv = kwargs.get("cu_seqlens_kv_values")
    seqused_q = kwargs.get("seqused_q_values")
    seqused_kv = kwargs.get("seqused_kv_values")
    input_layout = kwargs.get("input_layout", "TND")
    is_contiguous = kwargs.get("is_contiguous", True)
    device_id = kwargs.get("device_id", 0)
    graph_path = kwargs.get("graph_path", 0)
    data_range_q = kwargs.get("data_range_q", 1.0)
    data_range_k = kwargs.get("data_range_k", 1.0)
    data_range_v = kwargs.get("data_range_v", 1.0)
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
    if block_table is not None:
        block_table.copy_(torch.zeros_like(block_table))

    # ----- slot 8-14: 用 attributes 真实值覆盖 ttk 随机生成的 cu_seqlens/seqused -----
    _write_int32_list(cu_seqlens_q_t, cu_seqlens_q, "cu_seqlens_q (slot 8)")
    _write_int32_list(cu_seqlens_kv_t, cu_seqlens_kv, "cu_seqlens_kv (slot 9)")
    _write_int32_list(seqused_q_t, seqused_q, "seqused_q (slot 10)")
    _write_int32_list(seqused_kv_t, seqused_kv, "seqused_kv (slot 11)")
    _write_causal_mask(attn_mask_t, mask_mode, kwargs.get("attn_mask_shape"))

    logger.info(
        "[INPUTS] HIF8 in-place wrote q/k/v (q=%s), fp32 descale (dq=%s, dk=%s, dv=%s), "
        "fp32 p_scale, int32 block_table (pa=False)",
        tuple(q.shape),
        tuple(dequant_scale_q.shape),
        tuple(dequant_scale_k.shape),
        tuple(dequant_scale_v.shape),
    )


# ==============================================================================
# HiFP4 输入生成插件
# ==============================================================================

try:
    from ml_dtypes import bfloat16 as np_bfloat16
except ImportError:
    np_bfloat16 = numpy.float32

_HIFP4_SEED_MAP = {"q": 54, "k": 3, "v": 4}
_HIFP4_GROUP_SIZE = 64


def _hifp4_to_bf16(x):
    x = numpy.asarray(x, dtype=numpy.float64)
    tmp = numpy.abs(x)
    e = numpy.floor(numpy.log2(tmp + 2.0 ** (-1000)))
    return numpy.round(x * 2.0 ** (-e + 7)) * 2.0 ** (e - 7)


def _hifp4_bf16_to_e6m2(x):
    x = numpy.asarray(x, dtype=numpy.float64)
    max_val = 2.0**15 * 1.5
    min_val = 2.0 ** (-48)
    x = numpy.where(x > max_val, max_val, x)
    x = numpy.where(x < min_val, min_val, x)
    e = numpy.floor(numpy.log2(x))
    return numpy.round(x * 2.0 ** (-e + 2)) * 2.0 ** (-2 + e)


def _hifp4_float_to_e6m2_bits(x):
    x = numpy.asarray(x, dtype=numpy.float64)
    max_val = 2.0**15 * 1.5
    min_val = 2.0 ** (-48)
    nan_val = 2.0**15 * 1.75
    xc = numpy.where(numpy.isnan(x), max_val, x)
    xc = numpy.where(xc > max_val, max_val, xc)
    xc = numpy.where(xc < min_val, min_val, xc)
    m, e = numpy.frexp(xc)
    f = m * 2.0
    exp = e - 1
    r = numpy.clip(numpy.round(f * 4.0).astype(numpy.int64), 4, 8)
    carry = r == 8
    exp = numpy.where(carry, exp + 1, exp)
    m_val = numpy.where(carry, 0, r - 4)
    code = ((exp + 48).astype(numpy.int64) << 2) | m_val
    code = code.astype(numpy.uint32)
    nan_mask = numpy.isnan(x) | numpy.isclose(x, nan_val, rtol=1e-12, atol=1e-12)
    return numpy.where(nan_mask, numpy.uint32(0xFF), code)


def _hifp4_combine_to_32bit(e6m2_bits, e1_8, e1_16):
    w8 = numpy.array([1 << (7 - i) for i in range(8)], dtype=numpy.uint32)
    w16 = numpy.array([1 << (15 - i) for i in range(16)], dtype=numpy.uint32)
    int8 = (e1_8.astype(numpy.uint32) * w8).sum(axis=-1, dtype=numpy.uint32)
    int16 = (e1_16.astype(numpy.uint32) * w16).sum(axis=-1, dtype=numpy.uint32)
    return (e6m2_bits.astype(numpy.uint32) | (int8 << 8) | (int16 << 16)).astype(
        numpy.uint32
    )


def _hifp4_three_level_quant(x_2d, group_size=64):
    x = numpy.asarray(x_2d, dtype=numpy.float64)
    Mi, Ni = x.shape
    g = group_size
    ncnt = Ni // g
    sign = numpy.where(x < 0, -1.0, 1.0).reshape(Mi, ncnt, g)
    abs_grp = numpy.abs(x).reshape(Mi, ncnt, g)
    v16 = abs_grp.reshape(Mi, ncnt, 16, g // 16).max(axis=-1)
    v8 = v16.reshape(Mi, ncnt, 8, 2).max(axis=-1)
    vmax = v8.max(axis=-1)
    vmax_safe = numpy.where(vmax == 0, 2.0 ** (-48), vmax)
    const_rec = _hifp4_to_bf16(1.0 / 7.0)
    sf = _hifp4_to_bf16(vmax_safe * const_rec)
    e6m2 = _hifp4_bf16_to_e6m2(sf)
    e6m2_bits = _hifp4_float_to_e6m2_bits(sf)
    rec_e6m2 = _hifp4_to_bf16(1.0 / e6m2)
    e1_8 = (v8 * rec_e6m2[..., None] >= 4.0).astype(numpy.uint8)
    e1_8x2 = numpy.repeat(e1_8, 2, axis=-1)
    e1_16 = (
        v16 * rec_e6m2[..., None] * numpy.power(2.0, -e1_8x2.astype(numpy.float64))
        >= 2.0
    ).astype(numpy.uint8)
    de16 = e1_16.astype(numpy.int64) + e1_8x2.astype(numpy.int64)
    de64 = numpy.repeat(de16, g // 16, axis=-1)
    ng = 2
    abs_bf16 = np_bfloat16(abs_grp)
    rec_bf16 = np_bfloat16(rec_e6m2[..., None])
    prod = abs_bf16 * rec_bf16 * numpy.power(2.0, -de64 + ng)
    in_grp = numpy.floor(prod + 0.5) * (2.0 ** (-ng))
    in_grp = numpy.where(in_grp >= 2.0, 2.0 - 2.0 ** (-ng), in_grp)
    x_hifp4 = (sign * in_grp).reshape(Mi, Ni)
    scale = _hifp4_combine_to_32bit(e6m2_bits, e1_8[:, :, ::-1], e1_16[:, :, ::-1])
    return x_hifp4, scale


def _hifp4_pack_fp4_e1m2(x_hifp4):
    x = numpy.asarray(x_hifp4, dtype=numpy.float64)
    sign = (x < 0).astype(numpy.uint8)
    abs_val = numpy.abs(x)
    code = numpy.clip(numpy.round(abs_val * 4.0).astype(numpy.int64), 0, 7).astype(
        numpy.uint8
    )
    fp4_val = (sign << 3) | code
    flat = fp4_val.reshape(-1)
    packed = (flat[1::2] << 4) | flat[0::2]
    out_shape = tuple(x_hifp4.shape[:-1]) + (x_hifp4.shape[-1] // 2,)
    return packed.reshape(out_shape).astype(numpy.uint8)


def _hifp4_quant_along_d(tensor_bnsd, group_size=64):
    B, N, S, D = tensor_bnsd.shape
    g = group_size
    if S == 0 or D == 0:
        return (
            numpy.zeros((B, N, S, D // 2), dtype=numpy.uint8),
            numpy.zeros((B, N, S, D // g), dtype=numpy.float32),
        )
    flat = tensor_bnsd.reshape(B * N * S, D).astype(numpy.float64)
    x_h, scale_h = _hifp4_three_level_quant(flat, g)
    x_h = x_h.reshape(B, N, S, D).astype(np_bfloat16)
    fp4 = _hifp4_pack_fp4_e1m2(x_h)
    scale = numpy.ascontiguousarray(scale_h.reshape(B, N, S, D // g)).view(
        numpy.float32
    )
    return fp4, scale


def _hifp4_quant_v_along_s(tensor_bnsd, group_size=64):
    B, N, S, D = tensor_bnsd.shape
    g = group_size
    n_groups = math.ceil(S / g) if S > 0 else 0
    if S == 0 or D == 0:
        return (
            numpy.zeros((B, N, S, D // 2), dtype=numpy.uint8),
            numpy.zeros((B, N, D, n_groups), dtype=numpy.float32),
        )
    s_pad = n_groups * g
    flat = tensor_bnsd.transpose(0, 1, 3, 2).reshape(B * N * D, S).astype(numpy.float64)
    if s_pad > S:
        flat_pad = numpy.zeros((B * N * D, s_pad), dtype=numpy.float64)
        flat_pad[:, :S] = flat
    else:
        flat_pad = flat
    x_h, scale_h = _hifp4_three_level_quant(flat_pad, g)
    x_h = x_h[:, :S].reshape(B, N, D, S).transpose(0, 1, 3, 2)
    x_h = numpy.ascontiguousarray(x_h).astype(np_bfloat16)
    fp4 = _hifp4_pack_fp4_e1m2(x_h)
    scale = numpy.ascontiguousarray(scale_h.reshape(B, N, D, n_groups)).view(
        numpy.float32
    )
    return fp4, scale


def _hifp4_inplace_write(dst, src_np, slot_name):
    """hifloat4 (unpacked 4-bit) storage 写入: 拆包 packed uint8 → 1 byte/fp4。"""
    dst_dtype_str = str(getattr(dst, "dtype", ""))
    if "float4" in dst_dtype_str or "hifloat4" in dst_dtype_str:
        _flags = getattr(dst, "flags", None)
        if _flags is not None:
            _c_contig = (
                _flags["C_CONTIGUOUS"]
                if not isinstance(_flags, dict)
                else _flags.get("C_CONTIGUOUS", True)
            )
            if not bool(_c_contig):
                raise ValueError(
                    f"[HIFP4 INPUTS] {slot_name} hifloat4 storage must be C-contiguous"
                )
        dst_u8 = numpy.ascontiguousarray(dst).view(numpy.uint8)
        dst_u8[..., 0::2] = src_np & 0x0F
        dst_u8[..., 1::2] = (src_np >> 4) & 0x0F
        return
    if isinstance(dst, torch.Tensor):
        dst_np = (
            dst.view(torch.uint8).numpy() if "float8" in str(dst.dtype) else dst.numpy()
        )
        if dst_np.dtype != src_np.dtype:
            src_np = src_np.astype(dst_np.dtype)
        if tuple(dst_np.shape) != tuple(src_np.shape):
            raise ValueError(
                f"[HIFP4 INPUTS] {slot_name} shape mismatch: CSV {tuple(dst_np.shape)} != {tuple(src_np.shape)}"
            )
        dst_np[...] = src_np
    else:
        if dst.dtype != src_np.dtype:
            src_np = src_np.astype(dst.dtype)
        if tuple(dst.shape) != tuple(src_np.shape):
            raise ValueError(
                f"[HIFP4 INPUTS] {slot_name} shape mismatch: CSV {tuple(dst.shape)} != {tuple(src_np.shape)}"
            )
        dst[...] = src_np


def generate_qfa_hifp4_inputs(
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
    """HiFP4 输入生成插件 (quant_mode=10)。

    与 generate_qfa_mxfp8_inputs 同签名 (op-schema 直调), hifp4 专用:
      - bf16 BNSD random → 三阶量化 (E6M2 + E1_8 + E1_16) → fp4 e1m2 (packed uint8)
      - scale: 32-bit 位打包, 以 fp32 存储 (bitcast)
      - hifloat4 storage: 拆包 packed uint8 → 1 byte/fp4 (code in low nibble)
    """
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    enable_pa = bool(kwargs.get("enable_pa", False))
    kv_cache_layout = kwargs.get("kv_cache_layout") or layout_kv
    block_size = kwargs.get("block_size", 0)
    q_scale_layout = kwargs.get("q_scale_layout") or layout_q_descale
    cu_seqlens_q = kwargs.get("cu_seqlens_q_values")
    cu_seqlens_kv = kwargs.get("cu_seqlens_kv_values")
    seqused_q = kwargs.get("seqused_q_values")
    seqused_kv = kwargs.get("seqused_kv_values")
    input_layout = kwargs.get("input_layout", "BNSD")

    # HiFP4 仅支持 BNSD, 不感知 cu_seqlens_q/kv
    # B 推导优先级: cu_seqlens_q (len-1) → seqused_q (len) → batch_size attr → q.shape[0]
    cu_seqlens_q_list = list(cu_seqlens_q) if cu_seqlens_q is not None else None
    seqused_q_list = (
        list(seqused_q) if (seqused_q is not None and len(seqused_q) > 0) else []
    )
    seqused_kv_list = (
        list(seqused_kv) if (seqused_kv is not None and len(seqused_kv) > 0) else []
    )

    if cu_seqlens_q_list is not None and len(cu_seqlens_q_list) >= 2:
        B = max(1, len(cu_seqlens_q_list) - 1)
    elif seqused_q_list:
        B = len(seqused_q_list)
    else:
        _bs_attr = kwargs.get("batch_size")
        if _bs_attr is not None and int(_bs_attr) > 0:
            B = int(_bs_attr)
        elif hasattr(q, "shape") and q.ndim >= 4:
            B = int(q.shape[0])
        else:
            B = 1

    # HiFP4 BNSD: S 维对所有 batch 统一
    if len(set(seqused_q_list)) > 1:
        raise ValueError(
            f"[HIFP4 INPUTS] BNSD requires uniform seqused_q across batches, got {seqused_q_list}"
        )
    if len(set(seqused_kv_list)) > 1:
        raise ValueError(
            f"[HIFP4 INPUTS] BNSD requires uniform seqused_kv across batches, got {seqused_kv_list}"
        )

    # max_sq/max_skv 优先级: seqused → max_seqlen → tensor S 维
    s1_size = int(q.shape[2]) if (hasattr(q, "ndim") and q.ndim >= 3) else 0
    s2_size = int(k.shape[2]) if (hasattr(k, "ndim") and k.ndim >= 3) else 0
    max_seqlen_q_val = -1 if max_seqlen_q is None else int(max_seqlen_q)
    max_seqlen_kv_val = -1 if max_seqlen_kv is None else int(max_seqlen_kv)

    if seqused_q_list:
        max_sq = max(seqused_q_list)
    elif max_seqlen_q_val >= 0:
        max_sq = max_seqlen_q_val
    else:
        max_sq = s1_size
    if seqused_kv_list:
        max_skv = max(seqused_kv_list)
    elif max_seqlen_kv_val >= 0:
        max_skv = max_seqlen_kv_val
    else:
        max_skv = s2_size
    if max_sq <= 0:
        max_sq = D
    if max_skv <= 0:
        max_skv = D

    for gkey, gval in [
        ("B", B),
        ("N_q", N_q),
        ("N_kv", N_kv),
        ("D", D),
        ("HIFP4_GROUP_SIZE", _HIFP4_GROUP_SIZE),
        ("CU_SEQLENS_Q", None),
        ("CU_SEQLENS_KV", None),
        ("SEQUSED_Q", seqused_q_list if seqused_q_list else None),
        ("SEQUSED_KV", seqused_kv_list if seqused_kv_list else None),
        ("MAX_SEQLEN_Q", max_sq),
        ("MAX_SEQLEN_KV", max_skv),
        ("ENABLE_PA", enable_pa),
        ("KV_CACHE_LAYOUT", kv_cache_layout),
        ("BLOCK_SIZE", block_size),
        ("SPARSE_MODE", mask_mode),
        ("Q_SCALE_LAYOUT", q_scale_layout),
        ("INPUT_LAYOUT", input_layout),
        ("SOFTMAX_SCALE", softmax_scale),
    ]:
        setattr(hifp4_golden_mod, gkey, gval)

    G = _HIFP4_GROUP_SIZE

    # csv input_data_ranges -> bf16 q/k/v 数据范围
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

    # ----- Step 1: 生成 bf16 BNSD random -----
    torch.manual_seed(_HIFP4_SEED_MAP["q"])
    q_bf16 = (
        torch.rand(B, N_q, max_sq, D, dtype=torch.bfloat16) * (rq_max - rq_min) + rq_min
    )
    torch.manual_seed(_HIFP4_SEED_MAP["k"])
    k_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rk_max - rk_min)
        + rk_min
    )
    torch.manual_seed(_HIFP4_SEED_MAP["v"])
    v_bf16 = (
        torch.rand(B, N_kv, max_skv, D, dtype=torch.bfloat16) * (rv_max - rv_min)
        + rv_min
    )

    q_bf16_np = q_bf16.to(torch.float32).numpy()
    k_bf16_np = k_bf16.to(torch.float32).numpy()
    v_bf16_np = v_bf16.to(torch.float32).numpy()

    # ----- Step 2: 三阶量化 bf16 -> fp4 e1m2 (packed uint8) + uint32 scale -----
    q_fp4, dequant_scale_q_np = _hifp4_quant_along_d(q_bf16_np, G)
    k_fp4, k_antiquant_scale_np = _hifp4_quant_along_d(k_bf16_np, G)
    v_fp4, v_antiquant_scale_np = _hifp4_quant_v_along_s(v_bf16_np, G)

    # ----- Step 3: in-place 写入 CSV tensor slots -----
    _hifp4_inplace_write(q, q_fp4, "q (slot 0)")
    _hifp4_inplace_write(k, k_fp4, "k (slot 1)")
    _hifp4_inplace_write(v, v_fp4, "v (slot 2)")
    _hifp4_inplace_write(dequant_scale_q, dequant_scale_q_np, "descale_q (slot 3)")
    _hifp4_inplace_write(dequant_scale_k, k_antiquant_scale_np, "descale_k (slot 4)")
    _hifp4_inplace_write(dequant_scale_v, v_antiquant_scale_np, "descale_v (slot 5)")

    # p_scale
    p_scale_value = kwargs.get("p_scale_value", 1.0)
    if isinstance(p_scale_value, torch.Tensor):
        p_scale_value = float(p_scale_value.item())
    p_scale[...] = float(p_scale_value)

    # block_table
    if block_table is not None:
        block_table[...] = 0

    # cu_seqlens/seqused: 用 attributes 真实值覆盖 ttk 随机生成
    _write_int32_list(cu_seqlens_q_t, cu_seqlens_q, "cu_seqlens_q (slot 8)")
    _write_int32_list(cu_seqlens_kv_t, cu_seqlens_kv, "cu_seqlens_kv (slot 9)")
    _write_int32_list(seqused_q_t, seqused_q, "seqused_q (slot 10)")
    _write_int32_list(seqused_kv_t, seqused_kv, "seqused_kv (slot 11)")
    _write_causal_mask(attn_mask_t, mask_mode, kwargs.get("attn_mask_shape"))

    logger.info(
        "[HIFP4 INPUTS] wrote fp4 q/k/v (q=%s), fp32 three-level scale (dq=%s, dk=%s, dv=%s)",
        tuple(q.shape),
        tuple(dequant_scale_q.shape),
        tuple(dequant_scale_k.shape),
        tuple(dequant_scale_v.shape),
    )
