#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
compressor_grad_kernel — Compressor 反向传播 NPU Kernel（统一入口）。

TilingKey:  Coff, Layout, DataType                  — 编译期常量
TilingData: batch_size, token_size, seq_size,       — 运行时参数
            cmp_ratio, hidden_size, head_dim（另有 23 个 host 派生的分核/workspace 字段）

三阶段流水线:
  Phase 1: Vec Scatter  — 每压缩块计算 d_kv, d_score, APE 累加；cast 后 insert L1 + x 搬运
  Phase 2: Cube Matmul  — d_kv @ wkv + d_score @ wgate → d_x partial；
                          d_kv @ x + d_score @ x → d_wkv / d_wgate partial
  Phase 3: Vec Reduce   — 跨核归约 d_x / d_wkv / d_wgate / d_ape（跨轮累加）
"""

from dataclasses import dataclass
import torch
import torch_npu
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField as tilingkey_field


# ================================================================
#  Compile-time constants
# ================================================================
FLOAT_REG_ELEMENTS = 64  # FP32 VF register holds 64 elements
D_BASE_SIZE = 128
H_L1_BASE_SIZE = 256  # mm1/mm2 hidden 维 L1 分块
BIT16_REG_ELEMENTS = 128  # FP16/BF16 VF 寄存器元素数
BASE_SIZE = 128  # L0/L1 通用基础维度
H_BASE_SIZE = 128  # hidden 维子块
M_BASE_SIZE = 128


# ================================================================
#  UB / L1 地址常量（tile_group 声明）
# ================================================================
UB_DC_F16_ADDR = 0x00000  # dc_f16 / temp
UB_DC_F32_ADDR = 0x04000  # dc_f32
UB_KV_APE_ADDR = 0x0C000  # kv / ape_local / reduce_acc / p3_cast
UB_SM_DSB_ADDR = 0x1C000  # softmax / dsb_cast / moveX0 / reduce_ld0
UB_DSB_NZ_ADDR = 0x24000  # dsb_nz / dsb_nz_md_prev
UB_DSB_NZ_MD_CUR_ADDR = 0x28000  # dsb_nz_md_cur
UB_DKV_ADDR = 0x2C000  # dkv / dkv_cast / moveX1 / reduce_ld1
UB_DKV_NZ_ADDR = 0x34000  # dkv_nz / dkv_nz_md_prev
UB_DKV_NZ_MD_CUR_ADDR = 0x38000  # dkv_nz_md_cur
L1_KV0_ADDR = 0x00000  # L1Kv0 (NZ/ZN)
L1_KV1_ADDR = 0x10000  # L1Kv1 (NZ/ZN)
L1_SB0_ADDR = 0x20000  # L1Score0 (NZ/ZN)
L1_SB1_ADDR = 0x30000  # L1Score1 (NZ/ZN)
L1_W_ADDR = 0x40000  # l1_w_db


# ================================================================
#  TilingKey
# ================================================================


class CompressorGradTilingKey:
    Coff = tilingkey_field(bits=2, values=[1, 2])
    Layout = tilingkey_field(bits=1, values=[0, 1])
    DataType = tilingkey_field(bits=2, values=[0, 1])

    def is_valid(self, key):
        coff, layout, dtype = key
        return coff in (1, 2) and layout in (0, 1) and dtype in (0, 1)


# ================================================================
#  TilingData
# ================================================================
@dataclass
class CompressorGradTiling:
    batch_size: int = 1
    token_size: int = 0
    seq_size: int = 0
    cmp_ratio: int = 4
    hidden_size: int = 512
    head_dim: int = 128
    # ── 以下为 host 计算的派生配置（kernel 运行期不变，单一来源）──
    cube_core_num: int = 0  # Cube 核数 = launch block_dim（替代 pl.get_block_num()）
    core_num: int = 0  # Vec 逻辑核数 = cubeCoreNum * subBlockNum(2)
    total_head_dim: int = 0  # Coff * headDim
    cmp_row_cnt: int = 0  # Coff * cmpRatio
    cmp_size: int = 0  # Coff * cmpRatio * headDim
    cmp_kv_batch_stride: int = 0  # ceil_div(seqSize, cmpRatio)
    cmp_kv_rows: int = 0  # dCmpKv/kv/sm 行数（= host outputRows）
    x_rows: int = 0  # x/dX 行数
    group_size: int = 1  # headDim // D_BASE_SIZE
    group_num: int = 0  # cubeCoreNum // groupSize
    group_deal_sc_num: int = 0  # cubeMBaseSize // cmpRatio
    deal_sc_num: int = 0  # M_BASE_SIZE // cmpRatio
    total_sc_num_per_round: int = 0  # groupNum * groupDealScNum
    db_row_cnt: int = 0  # groupNum * groupRowStride（xArrangeGm 每 db 区域行数）
    group_row_stride: int = (
        0  # 每 group 实际行数 = groupDealScNum*cmpRatio + (Coff-1)*cmpRatio
    )
    # ── 编译期派生（TilingKey 折叠值，host 同步传入；TileType 静态 shape 仍用内联算术）──
    coff_coef: int = 0  # 2 // Coff
    cube_m_base_size: int = 0  # M_BASE_SIZE * coffCoef
    d_deal_size: int = 0  # D_BASE_SIZE // Coff
    m_deal_size: int = 0  # M_BASE_SIZE * Coff
    # ── workspace 分区大小（FP32 元素数，dbRatio=2 双缓冲）──
    dape_ws_size: int = 0  # dbRatio * groupNum * cmpSize * coffCoef
    d_x_ws_size: int = 0  # dbRatio * cubeCoreNum * (M_BASE_SIZE*2) * hiddenSize
    d_w_weight_ws_size: int = (
        0  # groupNum * totalHeadDim * hiddenSize（单缓冲，无 dbRatio）
    )
    x_ws_size: int = 0  # dbRatio * groupNum * groupRowStride * hiddenSize
    d_x_cache_ws_size: int = 0  # dbRatio * cmpRatio * hiddenSize


def _ceil_div(dividend, divisor):
    return 0 if divisor == 0 else ((dividend) + (divisor) - 1) // (divisor)


def _align(num, rnd):
    return 0 if (rnd) == 0 else (((num) + (rnd) - 1) // (rnd) * (rnd))


def _trunc(num, rnd):
    return 0 if (rnd) == 0 else ((num) // (rnd) * (rnd))


# ================================================================
#  VF: 反向 scatter — d_kv = dC * softmax, d_score = softmax * (dsw - col_sum)
# ================================================================


@pl.vector_function
def _vf_scatter_backward(
    d_cmp_kv_tile,  # dC: (n_blocks, D) FP32 (pre-cast outside VF)
    kv_tile,  # kv_2d: (n_blocks*N, D) fp32 — kv（保持不动，供 (kv-w)）
    softmax_score_tile,  # softmax: (n_blocks*N, D) fp32 — sm → dsb 输出
    temp_tile,  # temp: (n_blocks*N, D) fp32 — dkv output
    sc_num: pl.DT_INT64,
    cmp_row_cnt: pl.DT_INT64,  # cmpRatio * coff
    d_deal_size: pl.DT_INT64,
):
    d_kv_tile = temp_tile
    d_score_tile = softmax_score_tile

    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

    halves = d_deal_size // FLOAT_REG_ELEMENTS
    cmp_size = cmp_row_cnt * d_deal_size

    for d_idx in pl.range(0, halves):
        d_offset = d_idx * FLOAT_REG_ELEMENTS

        for sc_idx in pl.range(0, sc_num):
            sc_offset = sc_idx * cmp_size
            d_cmp_kv_offset = sc_idx * d_deal_size + d_offset

            vreg_d_cmp_kv = vf.load_align(d_cmp_kv_tile, d_cmp_kv_offset)

            # ── 第一轮：d_kv = dC*sm；w = Kahan(Σ sm*kv) ──
            # 相消重构：d_score = sm*dC*(kv - w)（col = dC*w 代数恒等），把
            # (dC*kv − col) 的大幅相消拆成 O(1)×O(1) 的 (kv − w)。
            vreg_w = vf.full(0.0, dtype=pl.DT_FP32)
            vreg_wcomp = vf.full(0.0, dtype=pl.DT_FP32)
            for t_idx in pl.range(0, cmp_row_cnt):
                offset = sc_offset + t_idx * d_deal_size + d_offset
                vreg_softmax_score = vf.load_align(softmax_score_tile, offset)
                vreg_kv = vf.load_align(kv_tile, offset)
                vreg_d_kv = vf.mul(vreg_d_cmp_kv, vreg_softmax_score, mask)
                vf.store_align(d_kv_tile + offset, vreg_d_kv, mask)
                # Dekker 双乘积 (sm, kv) → (p_hi, p_lo)（12bit split，纯 fp32）
                vreg_t = vf.muls(vreg_softmax_score, 4097.0, mask)
                vreg_u = vf.sub(vreg_t, vreg_softmax_score, mask)
                vreg_sm_hi = vf.sub(vreg_t, vreg_u, mask)
                vreg_sm_lo = vf.sub(vreg_softmax_score, vreg_sm_hi, mask)
                vreg_t = vf.muls(vreg_kv, 4097.0, mask)
                vreg_u = vf.sub(vreg_t, vreg_kv, mask)
                vreg_kv_hi = vf.sub(vreg_t, vreg_u, mask)
                vreg_kv_lo = vf.sub(vreg_kv, vreg_kv_hi, mask)
                vreg_p_hi = vf.mul(vreg_softmax_score, vreg_kv, mask)
                vreg_p1 = vf.mul(vreg_sm_hi, vreg_kv_hi, mask)
                vreg_p2 = vf.mul(vreg_sm_lo, vreg_kv_hi, mask)
                vreg_p3 = vf.mul(vreg_sm_hi, vreg_kv_lo, mask)
                vreg_p4 = vf.mul(vreg_sm_lo, vreg_kv_lo, mask)
                vreg_tmp = vf.sub(vreg_p_hi, vreg_p1, mask)
                vreg_tmp = vf.sub(vreg_tmp, vreg_p2, mask)
                vreg_tmp = vf.sub(vreg_tmp, vreg_p3, mask)
                vreg_p_lo = vf.sub(vreg_p4, vreg_tmp, mask)
                # 单链 Kahan：w = Σ(sm*kv)，乘积残差 p_lo 注入补偿项
                vreg_y = vf.sub(vreg_p_hi, vreg_wcomp, mask)
                vreg_t = vf.add(vreg_w, vreg_y, mask)
                vreg_c = vf.sub(vreg_t, vreg_w, mask)
                vreg_c = vf.sub(vreg_c, vreg_y, mask)
                vreg_w = vf.move(vreg_t, mask)
                vreg_wcomp = vf.add(vreg_c, vreg_p_lo, mask)

            vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
            # ── 第二轮：d_score = sm * (dC*s + dC*e)，(kv-w) → Knuth TwoSum 精确对 (s,e) ──
            for t_idx in pl.range(0, cmp_row_cnt):
                offset = sc_offset + t_idx * d_deal_size + d_offset
                vreg_softmax_score = vf.load_align(softmax_score_tile, offset)
                vreg_kv = vf.load_align(kv_tile, offset)
                # TwoSum(kv, -w): s=fl(kv-w); z=fl(s-kv); e=(kv-(s-z)) - (w+z)
                vreg_s = vf.sub(vreg_kv, vreg_w, mask)
                vreg_z = vf.sub(vreg_s, vreg_kv, mask)
                vreg_a = vf.sub(vreg_s, vreg_z, mask)
                vreg_b = vf.sub(vreg_kv, vreg_a, mask)
                vreg_c = vf.add(vreg_w, vreg_z, mask)
                vreg_e = vf.sub(vreg_b, vreg_c, mask)
                # d_score = sm * (dC*s + dC*e)（两项乘积分别舍入，保留低半）
                vreg_p1 = vf.mul(vreg_d_cmp_kv, vreg_s, mask)
                vreg_p2 = vf.mul(vreg_d_cmp_kv, vreg_e, mask)
                vreg_ds = vf.add(vreg_p1, vreg_p2, mask)
                vreg_d_score = vf.mul(vreg_softmax_score, vreg_ds, mask)
                vf.store_align(d_score_tile + offset, vreg_d_score, mask)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 清零 tile
# ================================================================
@pl.vector_function
def _vf_tile_zero(
    tile,
    n_rows: pl.DT_INT64,
    n_cols: pl.DT_INT64,
):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    halves = n_cols // FLOAT_REG_ELEMENTS
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_FP32)
    for half in pl.range(0, halves):
        half_off = half * FLOAT_REG_ELEMENTS
        for m in pl.range(0, n_rows):
            vf.store_align(tile + m * row_stride + half_off, zero_reg, mask)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 清零 tile (FP16)
# ================================================================
@pl.vector_function
def _vf_tile_zero_f16(
    tile,
    n_rows: pl.DT_INT64,
    n_cols: pl.DT_INT64,
):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    halves = n_cols // BIT16_REG_ELEMENTS
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_FP16)
    for half in pl.range(0, halves):
        half_off = half * BIT16_REG_ELEMENTS
        for m in pl.range(0, n_rows):
            vf.store_align(tile + m * row_stride + half_off, zero_reg, mask)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 清零 tile (BF16)
# ================================================================
@pl.vector_function
def _vf_tile_zero_bf16(
    tile,
    n_rows: pl.DT_INT64,
    n_cols: pl.DT_INT64,
):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    halves = n_cols // BIT16_REG_ELEMENTS
    row_stride = n_cols
    zero_reg = vf.full(0.0, dtype=pl.DT_BF16)
    for half in pl.range(0, halves):
        half_off = half * BIT16_REG_ELEMENTS
        for m in pl.range(0, n_rows):
            vf.store_align(tile + m * row_stride + half_off, zero_reg, mask)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 掩码清零 — 将非有效 token 对应的 d_kv/d_score 行置零
# ================================================================
@pl.vector_function
def _vf_mask_cache_fill(
    d_score_tile,
    dkv_tile,
    row: pl.DT_INT64,
    d_deal_size: pl.DT_INT64,
    d_stride: pl.DT_INT64,
):
    if row > 0:
        mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        halves = d_deal_size // FLOAT_REG_ELEMENTS
        vreg_zero = vf.full(0.0, dtype=pl.DT_FP32)
        for h_idx in pl.range(0, halves):
            d_offset = h_idx * FLOAT_REG_ELEMENTS
            for r_idx in pl.range(0, row):
                vf.store_align(
                    d_score_tile + r_idx * d_stride + d_offset, vreg_zero, mask
                )
                vf.store_align(dkv_tile + r_idx * d_stride + d_offset, vreg_zero, mask)
        vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: 跨 block 归约 — dApe[m,:] += Σ_k d_score[k,m,:]（per-round 调用）
# ================================================================
@pl.vector_function
def _vf_reduce_dscore_to_ape(
    d_score_tile,
    ape_tile,
    n_blocks: pl.DT_INT64,
    n_rows_per_block: pl.DT_INT64,
    d_deal_size: pl.DT_INT64,
    d_stride: pl.DT_INT64,
):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    halves = d_deal_size // FLOAT_REG_ELEMENTS
    row_stride = d_stride
    block_stride = n_rows_per_block * row_stride
    ape_row_stride = d_stride
    for half in pl.range(0, halves):
        half_off = half * FLOAT_REG_ELEMENTS
        for m in pl.range(0, n_rows_per_block):
            ape_off = m * ape_row_stride + half_off
            # 累加起点 = ape_tile 当前值（round0 已清零 / round>0 已 load ws 旧值）
            ape_reg = vf.load_align(ape_tile + ape_off, 0)
            # Kahan 补偿累加：跨 block 的 d_score → ape 顺序累加在长序列/相消场景下
            # 舍入噪声被放大（d_ape 是纯 vec 求和，无 matmul 洗噪），需带补偿项
            ape_comp = vf.full(0.0, dtype=pl.DT_FP32)
            for k in pl.range(0, n_blocks):
                block_off = k * block_stride
                row_off = block_off + m * row_stride + half_off
                vreg_d_score = vf.load_align(d_score_tile + row_off, 0)
                # Kahan: y = x - c; t = s + y; c = (t - s) - y; s = t
                vreg_y = vf.sub(vreg_d_score, ape_comp, mask)
                vreg_t = vf.add(ape_reg, vreg_y, mask)
                vreg_c = vf.sub(vreg_t, ape_reg, mask)
                vreg_c = vf.sub(vreg_c, vreg_y, mask)
                ape_reg = vf.move(vreg_t, mask)
                ape_comp = vf.move(vreg_c, mask)
            vf.store_align(ape_tile + ape_off, ape_reg, mask)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  VF: Kahan 补偿累加 — acc_tile += ld_tile（comp_tile 跨行保持补偿项）
# ================================================================
@pl.vector_function
def _vf_kahan_accumulate(
    acc_tile,
    ld_tile,
    comp_tile,
    n: pl.DT_INT64,
):
    """逐元素 Kahan：y = x - c; t = s + y; c = (t - s) - y; s = t。
    用于 Phase 3 跨核/跨轮 partial 的 fp32 顺序累加——长序列、多项相消场景下
    消除累加舍入（误差从 O(N·eps·相消倍数) 降至仅剩每项 fp32 存储舍入）。"""
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    halves = n // FLOAT_REG_ELEMENTS
    for half in pl.range(0, halves):
        off = half * FLOAT_REG_ELEMENTS
        vreg_ld = vf.load_align(ld_tile, off)
        vreg_acc = vf.load_align(acc_tile, off)
        vreg_comp = vf.load_align(comp_tile, off)
        vreg_y = vf.sub(vreg_ld, vreg_comp, mask)
        vreg_t = vf.add(vreg_acc, vreg_y, mask)
        vreg_c = vf.sub(vreg_t, vreg_acc, mask)
        vreg_c = vf.sub(vreg_c, vreg_y, mask)
        vf.store_align(acc_tile + off, vreg_t, mask)
        vf.store_align(comp_tile + off, vreg_c, mask)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ================================================================
#  Helper: get start position / seq used / seq length
# ================================================================
def _get_start_pos(b_idx, start_pos, ctx):
    result = 0
    if start_pos is not None:
        result = pl.getval(
            pl.make_tensor(start_pos, [ctx.batch_size], [1], dtype=pl.DT_INT32), b_idx
        )
    return result


def _get_seq_length(b_idx, cu_seqlens, ctx):
    result = ctx.seq_size
    if ctx.layout == 1:  # TH (1=TH)
        result = pl.getval(
            pl.make_tensor(cu_seqlens, [ctx.batch_size + 1], [1], dtype=pl.DT_INT32),
            b_idx + 1,
        ) - pl.getval(
            pl.make_tensor(cu_seqlens, [ctx.batch_size + 1], [1], dtype=pl.DT_INT32),
            b_idx,
        )
    return result


def _get_seq_used(b_idx, seq_used, cu_seqlens, ctx):
    if seq_used is not None:
        result = pl.getval(
            pl.make_tensor(seq_used, [ctx.batch_size], [1], dtype=pl.DT_INT32), b_idx
        )
    else:
        result = _get_seq_length(b_idx, cu_seqlens, ctx)
    return result


# ================================================================
#  Helper: 将 (b, s) 映射为全局 token 索引 tIdx，BSH/TH 统一
# ================================================================
def _get_token_idx(b_idx, s_idx, cu_seqlens, ctx):
    result = b_idx * ctx.seq_size + s_idx
    if ctx.layout == 1:
        result = (
            pl.getval(
                pl.make_tensor(
                    cu_seqlens, [ctx.batch_size + 1], [1], dtype=pl.DT_INT32
                ),
                b_idx,
            )
            + s_idx
        )
    return result


# ================================================================
#  Helper: 计算 (bStart,sStart) 到 (bEnd,sEnd) 之间的行数，sEnd 左闭右开
# ================================================================
def _get_row_count(b_start, s_start, b_end, s_end, cu_seqlens, ctx):
    result = _get_token_idx(b_end, s_end, cu_seqlens, ctx) - _get_token_idx(
        b_start, s_start, cu_seqlens, ctx
    )
    return result


# ================================================================
#  Helper: 计算位置 (bIdx, sIdx) 之前的压缩块总数
# ================================================================
def _get_cmp_block_count(b_idx, s_idx, start_pos, seq_used, cu_seqlens, ctx):
    """位置 (bIdx, sIdx) 之前的压缩块总数（按对齐位置计数）。

    语义与 blkInBatch 的 _ceil_div(cmpLimit-startPos, cmpRatio) 等价：
    startPos%cmpRatio!=0 时首块（部分块）计 1 块，末尾不足一块不计。
    修改时须与 _skip_blocks/_process_scatter/_arrange_x 的 blkInBatch 保持一致。
    """
    result = 0
    for bb in pl.range(0, b_idx):
        b_start_pos = _get_start_pos(bb, start_pos, ctx)
        b_seq_length = _get_seq_used(bb, seq_used, cu_seqlens, ctx)
        cmp_end = (b_start_pos + b_seq_length) // ctx.cmp_ratio
        if cmp_end > b_start_pos // ctx.cmp_ratio:
            result += cmp_end - (b_start_pos // ctx.cmp_ratio)
    b_start_pos = _get_start_pos(b_idx, start_pos, ctx)
    last_s = (
        _trunc(
            _get_seq_used(b_idx, seq_used, cu_seqlens, ctx) + b_start_pos, ctx.cmp_ratio
        )
        - b_start_pos
    )
    result += (
        min(last_s, s_idx) + b_start_pos
    ) // ctx.cmp_ratio - b_start_pos // ctx.cmp_ratio
    return result


# ================================================================
#  Helper: 将全局 token 索引 tIdx 反向映射为 (bIdx, sIdx)
# ================================================================
def _get_pos_from_token_idx(t_idx, cu_seqlens, ctx):
    result_b = t_idx // ctx.seq_size
    result_s = t_idx % ctx.seq_size
    if ctx.layout == 1:
        cu_tensor = pl.make_tensor(
            cu_seqlens, [ctx.batch_size + 1], [1], dtype=pl.DT_INT32
        )
        result_b = 0
        while result_b < ctx.batch_size:
            if t_idx < pl.getval(cu_tensor, result_b + 1):
                result_s = t_idx - pl.getval(cu_tensor, result_b)
                break
            result_b += 1
    return result_b, result_s


# ================================================================
#  Helper: advance block traversal（scatter/x 搬运与 dX reduce 共用）
# ================================================================
def _skip_blocks(block_count, b_idx, sc_idx, start_pos, seq_used, cu_seqlens, ctx):
    skipped = 0
    while skipped < block_count:
        b_start_pos = _get_start_pos(b_idx, start_pos, ctx)
        sq_val = _get_seq_used(b_idx, seq_used, cu_seqlens, ctx)
        cmp_limit = (b_start_pos + sq_val) // ctx.cmp_ratio * ctx.cmp_ratio
        # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（_ceil_div）
        blk_in_batch = _ceil_div(cmp_limit - b_start_pos, ctx.cmp_ratio)
        remaining = blk_in_batch - sc_idx
        if remaining > 0:
            taken = min(block_count - skipped, remaining)
            skipped += taken
            sc_idx += taken
        if sc_idx >= blk_in_batch:
            sc_idx = 0
            b_idx += 1
        if b_idx >= ctx.batch_size:
            break
    return b_idx, sc_idx


# ================================================================
#  Unified Compressor Grad Kernel
# ================================================================


# ================================================================
#  Phase 2 matmul sub-functions（从 kernel 提取，coff=1/2 统一）
# ================================================================
def _mm1_nl0_block(
    matmul_ctx,
    n_tile_rows,
    d_coff_idx,
    dx_row_off,
    h,
    w_idx,
    n_l0,
    cv_l0b,
    cv_l0c,
    w_l1,
    l0a,
    d_x_result_gm,
):
    """mm1 nL0 内层：L0B/L0C 轮转 + matmul（w_idx=0 首次写，w_idx=1 累加）+ store dX partial"""
    l0b = cv_l0b.next()
    l0c = cv_l0c.next()
    pl.set_validshape(l0b, [D_BASE_SIZE, H_BASE_SIZE])
    pl.set_validshape(l0c, [n_tile_rows, H_BASE_SIZE])
    pl.move(l0b, w_l1, offset=[0, n_l0])
    if w_idx == 0:
        # 尾轮 nTileRows=0（coff=1 subVecIdx=1）时跳过 0 行 matmul（aMatrixRow 下限 1）
        if n_tile_rows > 0:
            pl.matmul(l0c, l0a, l0b)
    else:
        if n_tile_rows > 0:
            pl.matmul_acc(l0c, l0c, l0a, l0b)
        pl.store(d_x_result_gm, l0c, [d_coff_idx, dx_row_off, h + n_l0])


def _mm1_sub_tile(
    matmul_ctx,
    h,
    w_idx,
    sub_vec_idx,
    db_idx,
    cv_l0a,
    cv_l0b,
    cv_l0c,
    l1_w_db,
    l1_kv0,
    l1_kv1,
    l1_sb0,
    l1_sb1,
    wkv_t,
    wgate_t,
    d_x_result_gm,
    n_start,
    dx_row_base,
):
    """mm1 单个 (wIdx, subVecIdx) 块：L0A 搬移 + w 权重加载 + nL0 内层 matmul"""
    n_tile_rows = (
        matmul_ctx.n_tile_rows0 if sub_vec_idx == 0 else matmul_ctx.n_tile_rows1
    )
    # coff=1 M 轴子槽行距 = dealScNum*cmpRatio（cr|128 时 = D_BASE_SIZE）
    m_off = (
        sub_vec_idx * matmul_ctx.deal_sc_num * matmul_ctx.cmp_ratio if Coff == 1 else 0
    )
    d_coff_idx = 0 if Coff == 1 else sub_vec_idx
    l0a = cv_l0a.next()
    pl.set_validshape(l0a, [n_tile_rows, D_BASE_SIZE])
    w_l1 = l1_w_db.next() if (sub_vec_idx == 0 or Coff == 2) else l1_w_db.current()
    src_l1 = (
        (l1_kv0 if sub_vec_idx == 0 else l1_kv1)
        if w_idx == 0
        else (l1_sb0 if sub_vec_idx == 0 else l1_sb1)
    )
    pl.set_validshape(src_l1, [n_tile_rows, D_BASE_SIZE])
    pl.move(l0a, src_l1)
    pl.set_validshape(w_l1, [D_BASE_SIZE, H_L1_BASE_SIZE])
    if sub_vec_idx == 0 or Coff == 2:
        w_row_off = 0 if Coff == 1 else sub_vec_idx * matmul_ctx.head_dim
        if w_idx == 0:
            pl.load(w_l1, wkv_t, [n_start + w_row_off, h])
        else:
            pl.load(w_l1, wgate_t, [n_start + w_row_off, h])
    for n_l0 in pl.range(0, H_L1_BASE_SIZE, H_BASE_SIZE):
        _mm1_nl0_block(
            matmul_ctx,
            n_tile_rows,
            d_coff_idx,
            dx_row_base + m_off,
            h,
            w_idx,
            n_l0,
            cv_l0b,
            cv_l0c,
            w_l1,
            l0a,
            d_x_result_gm,
        )


def _compute_dx_partial(
    matmul_ctx,
    db_idx,
    cv_l0a,
    cv_l0b,
    cv_l0c,
    l1_w_db,
    l1_kv0,
    l1_kv1,
    l1_sb0,
    l1_sb1,
    wkv_t,
    wgate_t,
    d_x_result_gm,
):
    """Matmul #1: dkv@wkv + dsb@wgate → dX partial（h×wIdx×subVecIdx 三重循环骨架）"""
    cube_core_idx = pl.get_block_idx()
    group_size = matmul_ctx.group_size
    group_num = matmul_ctx.group_num
    group_idx = cube_core_idx // group_size
    intra_group_idx = cube_core_idx % group_size
    n_start = intra_group_idx * D_BASE_SIZE
    dx_row_base = (
        db_idx * matmul_ctx.cube_core_num + cube_core_idx
    ) * matmul_ctx.cube_m_base_size
    for h in pl.range(0, matmul_ctx.hidden_size, H_L1_BASE_SIZE):
        for w_idx in pl.range(0, 2):
            for sub_vec_idx in pl.range(0, 2):
                _mm1_sub_tile(
                    matmul_ctx,
                    h,
                    w_idx,
                    sub_vec_idx,
                    db_idx,
                    cv_l0a,
                    cv_l0b,
                    cv_l0c,
                    l1_w_db,
                    l1_kv0,
                    l1_kv1,
                    l1_sb0,
                    l1_sb1,
                    wkv_t,
                    wgate_t,
                    d_x_result_gm,
                    n_start,
                    dx_row_base,
                )


def _mm2_store(
    matmul_ctx,
    round_idx,
    w_idx,
    row_off,
    col_off,
    l0c,
    d_wkv_result_gm,
    d_w_gate_result_gm,
):
    """mm2 单个 (wIdx) 结果写回：round 0 覆盖写 / round 1+ 原子加（跨轮累加）"""
    if w_idx == 0:
        if round_idx == 0:
            pl.store(d_wkv_result_gm, l0c, [row_off, col_off])
        else:
            pl.store(
                d_wkv_result_gm, l0c, [row_off, col_off], atomic=pl.AtomicType.AtomicAdd
            )
    else:
        if round_idx == 0:
            pl.store(d_w_gate_result_gm, l0c, [row_off, col_off])
        else:
            pl.store(
                d_w_gate_result_gm,
                l0c,
                [row_off, col_off],
                atomic=pl.AtomicType.AtomicAdd,
            )


def _mm2_l0a_load(
    matmul_ctx,
    sub_vec_idx,
    w_idx,
    n_tile_rows,
    m_start,
    h,
    l0a,
    w_l1,
    l1_kv0_t,
    l1_kv1_t,
    l1_sb0_t,
    l1_sb1_t,
    x_arrange_gm,
):
    """mm2 nL0==0 时：L0A 搬移（dkv/dsb 半区）+ x 权重加载到 L1W（w_idx==0 时）"""
    src_l1 = (
        (l1_kv0_t if sub_vec_idx == 0 else l1_kv1_t)
        if w_idx == 0
        else (l1_sb0_t if sub_vec_idx == 0 else l1_sb1_t)
    )
    pl.set_validshape(src_l1, [D_BASE_SIZE, n_tile_rows])
    pl.move(l0a, src_l1)
    if w_idx == 0:
        x_off = (
            m_start + sub_vec_idx * matmul_ctx.deal_sc_num * matmul_ctx.cmp_ratio
            if Coff == 1
            else m_start
        )
        if Coff == 2 and sub_vec_idx == 0:
            x_off = x_off - matmul_ctx.cmp_ratio
        pl.set_validshape(w_l1, [n_tile_rows, H_L1_BASE_SIZE])
        pl.load(w_l1, x_arrange_gm, [matmul_ctx.intra_group_idx, x_off, h])


def _mm2_core(
    matmul_ctx,
    round_idx,
    sub_vec_idx,
    w_idx,
    n_l0,
    h,
    dw_row_base,
    m_start,
    cv_l0a,
    cv_l0b,
    cv_l0c,
    l1_w_db,
    l1_kv0_t,
    l1_kv1_t,
    l1_sb0_t,
    l1_sb1_t,
    x_arrange_gm,
    d_wkv_result_gm,
    d_w_gate_result_gm,
):
    """mm2 单个 (wIdx) 核心：L0A/L0B/L0C 轮转 + matmul + 结果写回"""
    n_tile_rows = (
        matmul_ctx.n_tile_rows0 if sub_vec_idx == 0 else matmul_ctx.n_tile_rows1
    )
    l0a = cv_l0a.next()
    pl.set_validshape(l0a, [D_BASE_SIZE, n_tile_rows])
    w_l1 = l1_w_db.next() if (n_l0 == 0 and w_idx == 0) else l1_w_db.current()
    if n_l0 == 0:
        _mm2_l0a_load(
            matmul_ctx,
            sub_vec_idx,
            w_idx,
            n_tile_rows,
            m_start,
            h,
            l0a,
            w_l1,
            l1_kv0_t,
            l1_kv1_t,
            l1_sb0_t,
            l1_sb1_t,
            x_arrange_gm,
        )
    l0b = cv_l0b.next() if w_idx == 0 else cv_l0b.current()
    l0c = cv_l0c.next()
    pl.set_validshape(l0b, [n_tile_rows, H_BASE_SIZE])
    pl.set_validshape(l0c, [D_BASE_SIZE, H_BASE_SIZE])
    pl.move(l0b, w_l1, offset=[0, n_l0])
    if Coff == 1 and sub_vec_idx == 1:
        if n_tile_rows > 0:
            pl.matmul_acc(l0c, l0c, l0a, l0b)
    else:
        pl.matmul(l0c, l0a, l0b)
    if sub_vec_idx == 1 or Coff == 2:
        dw_row_off = 0 if Coff == 1 else sub_vec_idx * matmul_ctx.head_dim
        _mm2_store(
            matmul_ctx,
            round_idx,
            w_idx,
            dw_row_base + dw_row_off,
            h + n_l0,
            l0c,
            d_wkv_result_gm,
            d_w_gate_result_gm,
        )


def _mm2_sub_block(
    matmul_ctx,
    round_idx,
    sub_vec_idx,
    n_l0,
    h,
    dw_row_base,
    m_start,
    cv_l0a,
    cv_l0b,
    cv_l0c,
    l1_w_db,
    l1_kv0_t,
    l1_kv1_t,
    l1_sb0_t,
    l1_sb1_t,
    x_arrange_gm,
    d_wkv_result_gm,
    d_w_gate_result_gm,
):
    """mm2 nL0 块：wIdx（wkv/wgate）遍历"""
    for w_idx in pl.range(0, 2):
        _mm2_core(
            matmul_ctx,
            round_idx,
            sub_vec_idx,
            w_idx,
            n_l0,
            h,
            dw_row_base,
            m_start,
            cv_l0a,
            cv_l0b,
            cv_l0c,
            l1_w_db,
            l1_kv0_t,
            l1_kv1_t,
            l1_sb0_t,
            l1_sb1_t,
            x_arrange_gm,
            d_wkv_result_gm,
            d_w_gate_result_gm,
        )


def _mm2_sub_vec(
    matmul_ctx,
    round_idx,
    sub_vec_idx,
    h,
    dw_row_base,
    m_start,
    cv_l0a,
    cv_l0b,
    cv_l0c,
    l1_w_db,
    l1_kv0_t,
    l1_kv1_t,
    l1_sb0_t,
    l1_sb1_t,
    x_arrange_gm,
    d_wkv_result_gm,
    d_w_gate_result_gm,
):
    """mm2 subVecIdx 块：nL0 遍历"""
    for n_l0 in pl.range(0, H_L1_BASE_SIZE, H_BASE_SIZE):
        _mm2_sub_block(
            matmul_ctx,
            round_idx,
            sub_vec_idx,
            n_l0,
            h,
            dw_row_base,
            m_start,
            cv_l0a,
            cv_l0b,
            cv_l0c,
            l1_w_db,
            l1_kv0_t,
            l1_kv1_t,
            l1_sb0_t,
            l1_sb1_t,
            x_arrange_gm,
            d_wkv_result_gm,
            d_w_gate_result_gm,
        )


def _compute_dw_partial(
    matmul_ctx,
    round_idx,
    db_idx,
    cv_l0a,
    cv_l0b,
    cv_l0c,
    l1_w_db,
    l1_kv0_t,
    l1_kv1_t,
    l1_sb0_t,
    l1_sb1_t,
    x_arrange_gm,
    d_wkv_result_gm,
    d_w_gate_result_gm,
):
    """Matmul #2: dkv@x + dsb@x → dWkv/dWgate partial（h×subVecIdx 循环骨架）

    末轮归约方案：round 0 覆盖写初始化固定槽，round 1+ 原子加（跨轮累加），
    ws 单缓冲（无 dbIdx），末轮由 Phase 3 一次跨核归约。
    """
    cube_core_idx = pl.get_block_idx()
    group_size = matmul_ctx.group_size
    group_num = matmul_ctx.group_num
    group_idx = cube_core_idx // group_size
    intra_group_idx = cube_core_idx % group_size
    total_head_dim = matmul_ctx.total_head_dim
    dw_row_base = group_idx * total_head_dim + intra_group_idx * D_BASE_SIZE
    db_row_cnt = matmul_ctx.db_row_cnt
    # group 区域步长用实际值 groupRowStride（与 _arrange_x 写方逐字节对齐）
    m_start = (
        db_idx * db_row_cnt
        + group_idx * matmul_ctx.group_row_stride
        + (Coff - 1) * matmul_ctx.cmp_ratio
    )
    for h in pl.range(0, matmul_ctx.hidden_size, H_L1_BASE_SIZE):
        for sub_vec_idx in pl.range(0, 2):
            _mm2_sub_vec(
                matmul_ctx,
                round_idx,
                sub_vec_idx,
                h,
                dw_row_base,
                m_start,
                cv_l0a,
                cv_l0b,
                cv_l0c,
                l1_w_db,
                l1_kv0_t,
                l1_kv1_t,
                l1_sb0_t,
                l1_sb1_t,
                x_arrange_gm,
                d_wkv_result_gm,
                d_w_gate_result_gm,
            )


# ================================================================
#  Phase 3 reduce sub-functions（从 kernel 提取）
# ================================================================
def _reduce_ape(
    vec_ctx,
    core_idx,
    group_idx,
    db_idx,
    round_blocks,
    reduce_acc,
    reduce_comp,
    reduce_ld_group,
    tensor_d_ape_flat,
    tensor_d_ape_ws_flat,
):
    """Phase 3: dApe 跨核归约（每核处理 rgStart..rgEnd 的 chunk，累加器从零开始）"""
    core_num = vec_ctx.core_num
    group_num = vec_ctx.group_num
    flat_row_base = db_idx * group_num * (2 // Coff)
    rg_chunk_num = vec_ctx.cmp_size // D_BASE_SIZE
    rg_per_core = rg_chunk_num // core_num
    rg_remain = rg_chunk_num % core_num
    rg_start = core_idx * rg_per_core + (
        core_idx if core_idx < rg_remain else rg_remain
    )
    rg_end = rg_start + rg_per_core + (1 if core_idx < rg_remain else 0)
    rg_start = rg_start * D_BASE_SIZE
    rg_end = rg_end * D_BASE_SIZE
    # validRows：实际写入 tensorApe 的行数（只读被写过的行，与 workspace 初始内容无关）
    # coff=1：每轮每个 vec 子核处理 dealScNum 块，写行号 = groupIdx*2+sub，
    #   写 iff groupIdx*gds + sub*ds < roundBlocks ⟺ 行号 < roundBlocks/ds，
    #   即写行数 = min(ceil(roundBlocks/ds), 2*groupNum)；
    #   原 ceil(roundBlocks/gds)*2 在 roundBlocks 非 ds 整数倍时高估（读入未写
    #   槽位→垃圾）；min(roundBlocks, 2*groupNum) 仅 ds==1 时正确
    # coff=2：每 group 仅 1 行（coffCoef=1），ceil(roundBlocks/gds) 精确
    if Coff == 1:
        valid_rows = min(
            (round_blocks + vec_ctx.deal_sc_num - 1) // vec_ctx.deal_sc_num,
            group_num * 2,
        )
    else:
        valid_rows = (
            round_blocks + vec_ctx.group_deal_sc_num - 1
        ) // vec_ctx.group_deal_sc_num
    pl.set_validshape(reduce_acc, [1, rg_end - rg_start])
    pl.set_validshape(reduce_comp, [1, rg_end - rg_start])
    # 末轮一次性归约：累加器从零开始（输出 tensor 不保证清零，不能读入做 RMW）
    # 跨核/跨轮 partial 的顺序累加在长序列（多项相消 ~1e4）下会放大累加舍入，
    # 超过两方 golden 的块化归约精度 → 用 Kahan 补偿累加消除累加舍入
    _vf_tile_zero(reduce_acc, 1, (rg_end - rg_start))
    _vf_tile_zero(reduce_comp, 1, (rg_end - rg_start))
    for r in pl.range(0, valid_rows):
        reduce_ld = reduce_ld_group.next()
        pl.set_validshape(reduce_ld, [1, rg_end - rg_start])
        pl.load(reduce_ld, tensor_d_ape_ws_flat, [flat_row_base + r, rg_start])
        _vf_kahan_accumulate(reduce_acc, reduce_ld, reduce_comp, rg_end - rg_start)
    pl.store(tensor_d_ape_flat, reduce_acc, [0, rg_start])


def _reduce_d_weight(
    vec_ctx,
    core_idx,
    group_idx,
    db_idx,
    round_blocks,
    reduce_acc,
    reduce_ld_group,
    p3_cast,
    tensor_dw_kv_flat,
    tensor_dw_kv_ws_flat,
    tensor_dw_gate_flat,
    tensor_dw_gate_ws_flat,
):
    """Phase 3: dWkv/dWgate 跨核归约（每核处理 rgStart..rgEnd 的 chunk，累加器从零开始）"""
    core_num = vec_ctx.core_num
    group_num = vec_ctx.group_num
    total_head_dim = vec_ctx.total_head_dim
    flat_row_base = db_idx * group_num
    rg_chunk_num = (total_head_dim * vec_ctx.hidden_size) // D_BASE_SIZE
    rg_per_core = rg_chunk_num // core_num
    rg_remain = rg_chunk_num % core_num
    rg_start = core_idx * rg_per_core + (
        core_idx if core_idx < rg_remain else rg_remain
    )
    rg_end = rg_start + rg_per_core + (1 if core_idx < rg_remain else 0)
    rg_start = rg_start * D_BASE_SIZE
    rg_end = rg_end * D_BASE_SIZE
    valid_rows = (
        round_blocks + vec_ctx.group_deal_sc_num - 1
    ) // vec_ctx.group_deal_sc_num
    for cur_rg_start in pl.range(rg_start, rg_end, (BASE_SIZE * BASE_SIZE)):
        cur_rg_num = min(BASE_SIZE * BASE_SIZE, rg_end - cur_rg_start)
        pl.set_validshape(reduce_acc, [1, cur_rg_num])
        # dWkv：末轮一次性归约，累加器从零开始（输出 tensor 不保证清零，不能读入做 RMW）
        _vf_tile_zero(reduce_acc, 1, cur_rg_num)
        for r in pl.range(0, valid_rows):
            reduce_ld = reduce_ld_group.next()
            pl.set_validshape(reduce_ld, [1, cur_rg_num])
            pl.load(reduce_ld, tensor_dw_kv_ws_flat, [flat_row_base + r, cur_rg_start])
            pl.add(reduce_acc, reduce_acc, reduce_ld)
        pl.set_validshape(p3_cast, [1, cur_rg_num])
        pl.cast(p3_cast, reduce_acc, mode=pl.RoundMode.CAST_ROUND)
        pl.store(tensor_dw_kv_flat, p3_cast, [0, cur_rg_start])
        # dWgate：同上，从零开始
        _vf_tile_zero(reduce_acc, 1, cur_rg_num)
        for r in pl.range(0, valid_rows):
            reduce_ld = reduce_ld_group.next()
            pl.set_validshape(reduce_ld, [1, cur_rg_num])
            pl.load(
                reduce_ld, tensor_dw_gate_ws_flat, [flat_row_base + r, cur_rg_start]
            )
            pl.add(reduce_acc, reduce_acc, reduce_ld)
        pl.set_validshape(p3_cast, [1, cur_rg_num])
        pl.cast(p3_cast, reduce_acc, mode=pl.RoundMode.CAST_ROUND)
        pl.store(tensor_dw_gate_flat, p3_cast, [0, cur_rg_start])


# ================================================================
#  无压缩块：四输出显式刷 0（totalValid=0 / compressedCnt=0）
# ================================================================
def _zero_outputs(
    core_idx,
    core_num,
    x_rows,
    hidden_size,
    total_head_dim,
    cmp_size,
    d_x,
    d_wkv,
    d_wgate,
    d_ape,
    io_d_type,
    reduce_acc,
    zero_cast,
):
    """无压缩块时对四个输出 GM 显式刷 0，不依赖归约路径的隐式行为。

    各核按 D_BASE_SIZE 块/行分片（与 _reduce_ape/_reduce_d_weight/_reduce_dx
    的分核模式一致）；零 tile 仅初始化一次，循环复用 store 到 GM。
    """
    d_x_out_gm = pl.make_tensor(
        d_x, [x_rows, hidden_size], [hidden_size, 1], dtype=io_d_type
    )
    tensor_dw_kv_flat = pl.make_tensor(
        d_wkv,
        [1, total_head_dim * hidden_size],
        [total_head_dim * hidden_size, 1],
        dtype=io_d_type,
    )
    tensor_dw_gate_flat = pl.make_tensor(
        d_wgate,
        [1, total_head_dim * hidden_size],
        [total_head_dim * hidden_size, 1],
        dtype=io_d_type,
    )
    tensor_d_ape_flat = pl.make_tensor(
        d_ape, [1, cmp_size], [cmp_size, 1], dtype=pl.DT_FP32
    )
    # ── 零 tile 初始化（一次）：清满 tile 物理空间（BASE_SIZE*BASE_SIZE=16384 元素），
    #    保证后续 d_w/d_ape 任意 cur_w_num/cur_rg_num（≤16384）的 store 读到的都是零 ──
    pl.set_validshape(zero_cast, [1, BASE_SIZE * BASE_SIZE])
    if DataType == 0:  # BF16
        _vf_tile_zero_bf16(zero_cast, 1, BASE_SIZE * BASE_SIZE)
    else:  # FP16
        _vf_tile_zero_f16(zero_cast, 1, BASE_SIZE * BASE_SIZE)
    pl.set_validshape(reduce_acc, [1, BASE_SIZE * BASE_SIZE])
    _vf_tile_zero(reduce_acc, 1, BASE_SIZE * BASE_SIZE)  # FP32 全零（d_ape 用）

    # ── d_x [xRows, hiddenSize]：按行分片（_reduce_dx 同款）──
    rows_per_core = x_rows // core_num
    rows_remain = x_rows % core_num
    r_s = core_idx * rows_per_core + (
        core_idx if core_idx < rows_remain else rows_remain
    )
    r_e = r_s + rows_per_core + (1 if core_idx < rows_remain else 0)
    for r in pl.range(r_s, r_e):
        pl.store(d_x_out_gm, zero_cast, [r, 0])

    # ── d_wkv / d_wgate [totalHeadDim * hiddenSize]：按 D_BASE_SIZE 块分片 ──
    #    （与 _reduce_d_weight 同款：扁平 view + 块分片）
    dw_chunk_num = (total_head_dim * hidden_size) // D_BASE_SIZE
    dw_per_core = dw_chunk_num // core_num
    dw_remain = dw_chunk_num % core_num
    w_s = core_idx * dw_per_core + (core_idx if core_idx < dw_remain else dw_remain)
    w_e = w_s + dw_per_core + (1 if core_idx < dw_remain else 0)
    w_s = w_s * D_BASE_SIZE
    w_e = w_e * D_BASE_SIZE
    for cur_w_start in pl.range(w_s, w_e, BASE_SIZE * BASE_SIZE):
        cur_w_num = min(BASE_SIZE * BASE_SIZE, w_e - cur_w_start)
        pl.set_validshape(zero_cast, [1, cur_w_num])
        pl.store(tensor_dw_kv_flat, zero_cast, [0, cur_w_start])
        pl.store(tensor_dw_gate_flat, zero_cast, [0, cur_w_start])

    # ── d_ape [1, cmpSize] FP32：按 D_BASE_SIZE 块分片（_reduce_ape 同款）──
    rg_chunk_num = cmp_size // D_BASE_SIZE
    rg_per_core = rg_chunk_num // core_num
    rg_remain = rg_chunk_num % core_num
    rg_start = core_idx * rg_per_core + (
        core_idx if core_idx < rg_remain else rg_remain
    )
    rg_end = rg_start + rg_per_core + (1 if core_idx < rg_remain else 0)
    rg_start = rg_start * D_BASE_SIZE
    rg_end = rg_end * D_BASE_SIZE
    for cur_rg_start in pl.range(rg_start, rg_end, BASE_SIZE * BASE_SIZE):
        cur_rg_num = min(BASE_SIZE * BASE_SIZE, rg_end - cur_rg_start)
        pl.set_validshape(reduce_acc, [1, cur_rg_num])
        pl.store(tensor_d_ape_flat, reduce_acc, [0, cur_rg_start])


def _cast_to_l1_channel(
    vec_ctx,
    n_tile_rows,
    deal_tc_size,
    tile_cast,
    tile_nz,
    tile_cast_md,
    tile_nz_md_prev,
    tile_nz_md_cur,
    src_tile,
    temp_tile_group,
):
    """单通路（dkv/dsb 对称共用）：FP32→FP16 cast + ND→NZ + prev/cur 拆分（coff=1/2 统一）"""
    d_deal_size = vec_ctx.d_deal_size
    cmp_ratio = vec_ctx.cmp_ratio
    pl.set_validshape(tile_cast, [n_tile_rows, d_deal_size])
    pl.set_validshape(src_tile, [n_tile_rows, d_deal_size])
    pl.cast(tile_cast, src_tile, mode=pl.RoundMode.CAST_ROUND)
    if Coff == 1:
        pl.set_validshape(tile_nz, [n_tile_rows, d_deal_size])
        pl.move(tile_nz, tile_cast)
    else:
        temp_tile = temp_tile_group.next()
        pl.set_validshape(tile_cast_md, [deal_tc_size * cmp_ratio, d_deal_size])
        pl.set_validshape(tile_nz_md_prev, [deal_tc_size * cmp_ratio, d_deal_size])
        pl.set_validshape(temp_tile, [deal_tc_size * cmp_ratio, d_deal_size])
        pl.move(temp_tile, tile_cast_md, offset=[0, 0])
        pl.move(tile_nz_md_prev, temp_tile)
        temp_tile = temp_tile_group.next()
        pl.set_validshape(
            tile_cast_md[:, d_deal_size:], [deal_tc_size * cmp_ratio, d_deal_size]
        )
        pl.set_validshape(tile_nz_md_cur, [deal_tc_size * cmp_ratio, d_deal_size])
        pl.set_validshape(temp_tile, [deal_tc_size * cmp_ratio, d_deal_size])
        pl.move(temp_tile, tile_cast_md, offset=[0, d_deal_size])
        pl.move(tile_nz_md_cur, temp_tile)


def _cast_dkv_dsb_to_l1(
    vec_ctx,
    n_tile_rows,
    deal_tc_size,
    sub_idx,
    tile_temp,
    tile_dkv_cast,
    tile_dkv_cast_md,
    tile_dkv_nz,
    tile_dkv_nz_md_prev,
    tile_dkv_nz_md_cur,
    tile_dsb_cast,
    tile_dsb_cast_md,
    tile_dsb_nz,
    tile_softmax,
    tile_dsb_nz_md_prev,
    tile_dsb_nz_md_cur,
    temp_tile_group,
    l1_kv0_g,
    l1_kv1_g,
    l1_sb0_g,
    l1_sb1_g,
):
    """Phase 1: dkv/dsb FP32→FP16 cast + ND→NZ + insert L1（coff=1/2 统一）"""
    d_deal_size = vec_ctx.d_deal_size
    cmp_ratio = vec_ctx.cmp_ratio
    # dkv 通路
    _cast_to_l1_channel(
        vec_ctx,
        n_tile_rows,
        deal_tc_size,
        tile_dkv_cast,
        tile_dkv_nz,
        tile_dkv_cast_md,
        tile_dkv_nz_md_prev,
        tile_dkv_nz_md_cur,
        tile_temp,
        temp_tile_group,
    )
    # dsb 通路
    _cast_to_l1_channel(
        vec_ctx,
        n_tile_rows,
        deal_tc_size,
        tile_dsb_cast,
        tile_dsb_nz,
        tile_dsb_cast_md,
        tile_dsb_nz_md_prev,
        tile_dsb_nz_md_cur,
        tile_softmax,
        temp_tile_group,
    )
    # insert L1
    if Coff == 1:
        l1_kv = l1_kv0_g.next() if sub_idx == 0 else l1_kv1_g.next()
        l1_sb = l1_sb0_g.next() if sub_idx == 0 else l1_sb1_g.next()
        pl.set_validshape(l1_kv, [n_tile_rows, d_deal_size])
        pl.set_validshape(l1_sb, [n_tile_rows, d_deal_size])
        pl.insert(l1_kv, tile_dkv_nz, [0, 0])
        pl.insert(l1_sb, tile_dsb_nz, [0, 0])
    else:
        l1_kv0 = l1_kv0_g.next()
        l1_kv1 = l1_kv1_g.next()
        l1_sb0 = l1_sb0_g.next()
        l1_sb1 = l1_sb1_g.next()
        col_offset = 0 if sub_idx == 0 else d_deal_size
        pl.set_validshape(l1_kv0, [deal_tc_size * cmp_ratio, D_BASE_SIZE])
        pl.set_validshape(l1_kv1, [deal_tc_size * cmp_ratio, D_BASE_SIZE])
        pl.set_validshape(l1_sb0, [deal_tc_size * cmp_ratio, D_BASE_SIZE])
        pl.set_validshape(l1_sb1, [deal_tc_size * cmp_ratio, D_BASE_SIZE])
        pl.insert(l1_kv0, tile_dkv_nz_md_prev, [0, col_offset])
        pl.insert(l1_kv1, tile_dkv_nz_md_cur, [0, col_offset])
        pl.insert(l1_sb0, tile_dsb_nz_md_prev, [0, col_offset])
        pl.insert(l1_sb1, tile_dsb_nz_md_cur, [0, col_offset])


def _reduce_dx_ws_offset(
    db_idx,
    cube_core_num,
    cube_m_base_size,
    group_size,
    group_deal_sc_num,
    cmp_ratio,
    p3l_sc_idx,
    prev_sc_idx,
    sc_inner_idx,
):
    """dX partial workspace 槽位偏移（相对块号 → group/块内序号映射）"""
    rel_sc = p3l_sc_idx - prev_sc_idx
    return (
        db_idx * cube_core_num * cube_m_base_size
        + (rel_sc // group_deal_sc_num) * group_size * cube_m_base_size
        + (rel_sc % group_deal_sc_num) * cmp_ratio
        + sc_inner_idx
    )


def _reduce_dx_setup(
    vec_ctx,
    core_idx,
    round_idx,
    b_idx_start,
    sc_idx_start,
    b_idx_end,
    sc_idx_end,
    start_pos,
    seq_used,
    cu_seqlens,
    seq_ctx,
):
    """行分片与轮节点计算：返回 7 元组（myRowS/myRowE/startTidx/roundEndScIdx/bStart/sStart/prevScIdx）"""
    core_num = vec_ctx.core_num
    round_cnt = vec_ctx.round_cnt
    cmp_ratio = vec_ctx.cmp_ratio
    batch_size = vec_ctx.batch_size
    # start 节点局部副本（避免别名修改传入的 scIdxStart）
    start_sc_idx = sc_idx_start
    prev_sc_idx = 0
    if Coff == 2 and start_sc_idx != 0:
        start_sc_idx -= 1
        prev_sc_idx += 1
    b_start = b_idx_start
    b_end = b_idx_end - 1 if sc_idx_end == 0 else b_idx_end
    s_start = (
        0
        if start_sc_idx == 0
        else start_sc_idx * cmp_ratio
        - (_get_start_pos(b_start, start_pos, seq_ctx) % cmp_ratio)
    )
    s_end = (
        _get_seq_length(b_end, cu_seqlens, seq_ctx)
        if sc_idx_end == 0
        else sc_idx_end * cmp_ratio
        - (_get_start_pos(b_end, start_pos, seq_ctx) % cmp_ratio)
    )
    if round_idx == round_cnt - 1:
        b_end = batch_size - 1
        s_end = _get_seq_length(b_end, cu_seqlens, seq_ctx)
    total_rows = _get_row_count(b_start, s_start, b_end, s_end, cu_seqlens, seq_ctx)
    rows_per_core = total_rows // core_num
    rows_remain = total_rows % core_num
    my_row_s = core_idx * rows_per_core + (
        core_idx if core_idx < rows_remain else rows_remain
    )
    my_row_e = my_row_s + rows_per_core + (1 if core_idx < rows_remain else 0)
    round_end_sc_idx = (
        _get_cmp_block_count(b_end, s_end, start_pos, seq_used, cu_seqlens, seq_ctx)
        - _get_cmp_block_count(
            b_start, s_start, start_pos, seq_used, cu_seqlens, seq_ctx
        )
        - 1
    )
    start_tidx = _get_token_idx(b_start, s_start, cu_seqlens, seq_ctx)
    return (
        my_row_s,
        my_row_e,
        start_tidx,
        round_end_sc_idx,
        b_start,
        s_start,
        prev_sc_idx,
    )


def _reduce_dx_row(
    vec_ctx,
    db_idx,
    prev_sc_idx,
    p3l_sc_idx,
    round_end_sc_idx,
    batch_end_sc_idx,
    sc_inner_idx,
    reduce_acc,
    reduce_ld_group,
    d_x_cache_gm,
    d_x_result_gm,
):
    """单行压缩区累加：跨轮 cache 起算或 result 槽位起算 + 组间累加 + coff=2 下一块累加"""
    cube_core_num = vec_ctx.cube_core_num
    hidden_size = vec_ctx.hidden_size
    group_size = vec_ctx.group_size
    if p3l_sc_idx < prev_sc_idx:
        ws_offset = db_idx * cube_core_num * vec_ctx.cube_m_base_size + sc_inner_idx
        pl.set_validshape(reduce_acc, [1, hidden_size])
        pl.load(
            reduce_acc, d_x_cache_gm, [db_idx * vec_ctx.cmp_ratio + sc_inner_idx, 0]
        )
        for r in pl.range(0, group_size):
            reduce_ld = reduce_ld_group.next()
            pl.set_validshape(reduce_ld, [1, hidden_size])
            pl.load(
                reduce_ld,
                d_x_result_gm,
                [0, ws_offset + r * vec_ctx.cube_m_base_size, 0],
            )
            pl.add(reduce_acc, reduce_acc, reduce_ld)
    else:
        ws_offset = _reduce_dx_ws_offset(
            db_idx,
            cube_core_num,
            vec_ctx.cube_m_base_size,
            group_size,
            vec_ctx.group_deal_sc_num,
            vec_ctx.cmp_ratio,
            p3l_sc_idx,
            prev_sc_idx,
            sc_inner_idx,
        )
        pl.set_validshape(reduce_acc, [1, hidden_size])
        pl.load(reduce_acc, d_x_result_gm, [Coff - 1, ws_offset, 0])
        if group_size > 1:
            for r in pl.range(1, group_size):
                reduce_ld = reduce_ld_group.next()
                pl.set_validshape(reduce_ld, [1, hidden_size])
                pl.load(
                    reduce_ld,
                    d_x_result_gm,
                    [Coff - 1, ws_offset + r * vec_ctx.cube_m_base_size, 0],
                )
                pl.add(reduce_acc, reduce_acc, reduce_ld)
        if (
            Coff == 2
            and p3l_sc_idx != round_end_sc_idx
            and p3l_sc_idx != batch_end_sc_idx
        ):
            pl.set_validshape(reduce_acc, [1, hidden_size])
            next_ws_offset = _reduce_dx_ws_offset(
                db_idx,
                cube_core_num,
                vec_ctx.cube_m_base_size,
                group_size,
                vec_ctx.group_deal_sc_num,
                vec_ctx.cmp_ratio,
                p3l_sc_idx + 1,
                prev_sc_idx,
                sc_inner_idx,
            )
            for r in pl.range(0, group_size):
                reduce_ld = reduce_ld_group.next()
                pl.set_validshape(reduce_ld, [1, hidden_size])
                pl.load(
                    reduce_ld,
                    d_x_result_gm,
                    [0, next_ws_offset + r * vec_ctx.cube_m_base_size, 0],
                )
                pl.add(reduce_acc, reduce_acc, reduce_ld)


def _reduce_dx_store(
    vec_ctx,
    db_idx,
    p3l_sc_idx,
    round_end_sc_idx,
    batch_end_sc_idx,
    sc_inner_idx,
    t_idx,
    reduce_acc,
    p3_cast,
    d_x_cache_gm,
    d_x_out_gm,
):
    """行结果写回：coff=2 轮末块存 dXCache（下轮跨轮累加），否则 cast+store d_x"""
    cmp_ratio = vec_ctx.cmp_ratio
    hidden_size = vec_ctx.hidden_size
    db_ratio = vec_ctx.db_ratio
    if Coff == 2 and p3l_sc_idx == round_end_sc_idx and p3l_sc_idx != batch_end_sc_idx:
        pl.store(
            d_x_cache_gm,
            reduce_acc,
            [(db_idx + 1) % db_ratio * cmp_ratio + sc_inner_idx, 0],
        )
    else:
        pl.set_validshape(p3_cast, [1, hidden_size])
        pl.cast(p3_cast, reduce_acc, mode=pl.RoundMode.CAST_ROUND)
        pl.store(d_x_out_gm, p3_cast, [t_idx, 0])


def _reduce_dx_zero_row(p3_cast, hidden_size, d_x_out_gm, t_idx):
    """非压缩区行：输出显式刷 0（不依赖归约路径隐式行为）"""
    pl.set_validshape(p3_cast, [1, hidden_size])
    if DataType == 0:
        _vf_tile_zero_bf16(p3_cast, 1, hidden_size)
    else:
        _vf_tile_zero_f16(p3_cast, 1, hidden_size)
    pl.store(d_x_out_gm, p3_cast, [t_idx, 0])


def _reduce_dx_advance_batch(
    vec_ctx,
    b,
    local_s,
    p3l_sc_idx,
    cmp_limit,
    b_start,
    s_start,
    batch_size,
    start_pos,
    seq_used,
    cu_seqlens,
    seq_ctx,
):
    """跨 batch 转移：前序批有压缩块（cmpLimit>0）→ 块索引 +1；0 块 → 不变。
    返回 8 元组（b/localS/p3lScIdx/bStartPos/bSeqUsed/bSeqLength/cmpLimit/batchEndScIdx）；
    b 越界（==batchSize）时返回哨兵值，由调用方 break。⚠️ local_s 必须重置 0。"""
    cmp_ratio = vec_ctx.cmp_ratio
    if cmp_limit > 0:
        p3l_sc_idx += 1
    local_s = 0
    b += 1
    b_start_pos = 0
    b_seq_used = 0
    b_seq_length = 0
    cmp_limit = 0
    batch_end_sc_idx = 0
    while b < batch_size:
        b_start_pos = _get_start_pos(b, start_pos, seq_ctx)
        b_seq_used = _get_seq_used(b, seq_used, cu_seqlens, seq_ctx)
        b_seq_length = _get_seq_length(b, cu_seqlens, seq_ctx)
        cmp_limit = (b_start_pos + b_seq_used) // cmp_ratio * cmp_ratio - b_start_pos
        batch_end_sc_idx = (
            _get_cmp_block_count(
                b, b_seq_used, start_pos, seq_used, cu_seqlens, seq_ctx
            )
            - _get_cmp_block_count(
                b_start, s_start, start_pos, seq_used, cu_seqlens, seq_ctx
            )
            - 1
        )
        if b_seq_length != 0:
            break
        # 空 batch（seq_len==0）不占行、无压缩块：块索引不推进，直接跳到下一批
        b += 1
    return (
        b,
        local_s,
        p3l_sc_idx,
        b_start_pos,
        b_seq_used,
        b_seq_length,
        cmp_limit,
        batch_end_sc_idx,
    )


def _reduce_dx_scan(
    vec_ctx,
    start_tidx,
    my_row_s,
    b_start,
    s_start,
    start_pos,
    seq_used,
    cu_seqlens,
    seq_ctx,
):
    """行起点状态扫描：返回 10 元组（b/localS/tIdx/bStartPos/bSeqUsed/bSeqLength/cmpLimit/p3lScIdx/batchEndScIdx/scInnerIdx）"""
    cmp_ratio = vec_ctx.cmp_ratio
    b, local_s = _get_pos_from_token_idx(start_tidx + my_row_s, cu_seqlens, seq_ctx)
    t_idx = start_tidx + my_row_s
    b_start_pos = _get_start_pos(b, start_pos, seq_ctx)
    b_seq_used = _get_seq_used(b, seq_used, cu_seqlens, seq_ctx)
    b_seq_length = _get_seq_length(b, cu_seqlens, seq_ctx)
    cmp_limit = (b_start_pos + b_seq_used) // cmp_ratio * cmp_ratio - b_start_pos
    p3l_sc_idx = _get_cmp_block_count(
        b, local_s, start_pos, seq_used, cu_seqlens, seq_ctx
    ) - _get_cmp_block_count(b_start, s_start, start_pos, seq_used, cu_seqlens, seq_ctx)
    if local_s >= cmp_limit and cmp_limit > 0:
        p3l_sc_idx -= 1
    batch_end_sc_idx = (
        _get_cmp_block_count(b, b_seq_used, start_pos, seq_used, cu_seqlens, seq_ctx)
        - _get_cmp_block_count(
            b_start, s_start, start_pos, seq_used, cu_seqlens, seq_ctx
        )
        - 1
    )
    sc_inner_idx = (local_s + b_start_pos) % cmp_ratio
    return (
        b,
        local_s,
        t_idx,
        b_start_pos,
        b_seq_used,
        b_seq_length,
        cmp_limit,
        p3l_sc_idx,
        batch_end_sc_idx,
        sc_inner_idx,
    )


def _reduce_dx(
    vec_ctx,
    core_idx,
    db_idx,
    round_idx,
    b_idx_start,
    sc_idx_start,
    b_idx_end,
    sc_idx_end,
    reduce_acc,
    reduce_ld_group,
    p3_cast,
    d_x_cache_gm,
    d_x_result_gm,
    d_x_out_gm,
    start_pos,
    seq_used,
    cu_seqlens,
    seq_ctx,
):
    """Phase 3: dX 跨核归约（token 行遍历 + coff=2 dXCache 缓存逻辑）"""
    hidden_size = vec_ctx.hidden_size
    cmp_ratio = vec_ctx.cmp_ratio
    batch_size = vec_ctx.batch_size
    (
        my_row_s,
        my_row_e,
        start_tidx,
        round_end_sc_idx,
        b_start,
        s_start,
        prev_sc_idx,
    ) = _reduce_dx_setup(
        vec_ctx,
        core_idx,
        round_idx,
        b_idx_start,
        sc_idx_start,
        b_idx_end,
        sc_idx_end,
        start_pos,
        seq_used,
        cu_seqlens,
        seq_ctx,
    )
    if my_row_s < my_row_e:
        (
            b,
            local_s,
            t_idx,
            b_start_pos,
            b_seq_used,
            b_seq_length,
            cmp_limit,
            p3l_sc_idx,
            batch_end_sc_idx,
            sc_inner_idx,
        ) = _reduce_dx_scan(
            vec_ctx,
            start_tidx,
            my_row_s,
            b_start,
            s_start,
            start_pos,
            seq_used,
            cu_seqlens,
            seq_ctx,
        )
        for _ in pl.range(my_row_s, my_row_e):
            in_compress = local_s < cmp_limit
            if in_compress:
                _reduce_dx_row(
                    vec_ctx,
                    db_idx,
                    prev_sc_idx,
                    p3l_sc_idx,
                    round_end_sc_idx,
                    batch_end_sc_idx,
                    sc_inner_idx,
                    reduce_acc,
                    reduce_ld_group,
                    d_x_cache_gm,
                    d_x_result_gm,
                )
                _reduce_dx_store(
                    vec_ctx,
                    db_idx,
                    p3l_sc_idx,
                    round_end_sc_idx,
                    batch_end_sc_idx,
                    sc_inner_idx,
                    t_idx,
                    reduce_acc,
                    p3_cast,
                    d_x_cache_gm,
                    d_x_out_gm,
                )
            else:
                if p3l_sc_idx < prev_sc_idx:
                    continue
                _reduce_dx_zero_row(p3_cast, hidden_size, d_x_out_gm, t_idx)
            t_idx += 1
            local_s += 1
            if local_s >= b_seq_length:
                # ── 跨 batch 转移：压缩块索引全局连续，新批首块是前序批末块的后继。
                # 前序批有压缩块（cmpLimit>0，此刻 cmpLimit 仍是前序批的值）→ +1；
                # 前序批 0 块 → 索引不变（块索引跨批不连续，不能无条件 +1）。
                (
                    b,
                    local_s,
                    p3l_sc_idx,
                    b_start_pos,
                    b_seq_used,
                    b_seq_length,
                    cmp_limit,
                    batch_end_sc_idx,
                ) = _reduce_dx_advance_batch(
                    vec_ctx,
                    b,
                    local_s,
                    p3l_sc_idx,
                    cmp_limit,
                    b_start,
                    s_start,
                    batch_size,
                    start_pos,
                    seq_used,
                    cu_seqlens,
                    seq_ctx,
                )
                if b >= batch_size:
                    break
            else:
                # ── 批内块边界（localS 为 cr 倍数，进入下一块）→ +1
                # else 分支 localS≥1；(localS+bStartPos)%cmpRatio==0 即块边界；
                # localS<cmpLimit 保证仍在压缩区内（尾区不增量）。
                if (local_s < cmp_limit) and ((local_s + b_start_pos) % cmp_ratio == 0):
                    p3l_sc_idx += 1
            sc_inner_idx = (local_s + b_start_pos) % cmp_ratio


def _process_mask_fill(
    sm_tile, dkv_tile, b_start_pos, sc_idx, cmp_ratio, d_deal_size, taken
):
    """scatter 后按 coff/batch 首块规则掩码清零非有效 cache 行（coff=1/2 统一）"""
    if Coff == 1:
        cache_fill_rows = b_start_pos % cmp_ratio if sc_idx == 0 else 0
        _vf_mask_cache_fill(
            sm_tile, dkv_tile, cache_fill_rows, d_deal_size, D_BASE_SIZE
        )
    else:
        if sc_idx == 0:
            cache_fill_rows = (
                b_start_pos % cmp_ratio + cmp_ratio if taken >= 2 else cmp_ratio
            )
        elif sc_idx == 1:
            cache_fill_rows = b_start_pos % cmp_ratio
        else:
            cache_fill_rows = 0
        _vf_mask_cache_fill(
            sm_tile, dkv_tile, cache_fill_rows, d_deal_size, D_BASE_SIZE
        )
        cache_fill_rows = b_start_pos % cmp_ratio if sc_idx == 0 else 0
        _vf_mask_cache_fill(
            sm_tile[1:, :], dkv_tile[1:, :], cache_fill_rows, d_deal_size, D_BASE_SIZE
        )


def _process_scatter_slice(
    vec_ctx,
    round_idx,
    total_sc_num_per_round,
    pre_deal_tc_size,
    processed,
    b_idx,
    sc_idx,
    taken,
    n_start,
    b_start_pos,
    tile_dc_f16,
    tile_dc_f32,
    tile_kv,
    tile_softmax,
    tile_temp,
    d_cmp_kv_gm,
    kv_gm,
    softmax_score_gm,
):
    """单个 taken 块：load dc/kv/sm → cast → VF 反向 → mask cache fill"""
    cmp_ratio = vec_ctx.cmp_ratio
    cmp_row_cnt = vec_ctx.cmp_row_cnt
    d_deal_size = vec_ctx.d_deal_size
    if Layout == 1:
        global_blk_idx = (
            round_idx * total_sc_num_per_round + pre_deal_tc_size + processed
        )
    else:
        global_blk_idx = b_idx * vec_ctx.cmp_kv_batch_stride + sc_idx
    n_tile_rows = taken * cmp_row_cnt
    dc_tile = tile_dc_f16[processed:, :]
    dc_f32 = tile_dc_f32[processed:, :]
    kv_tile = tile_kv[processed * cmp_row_cnt :, :]
    sm_tile = tile_softmax[processed * cmp_row_cnt :, :]
    dkv_tile = tile_temp[processed * cmp_row_cnt :, :]
    pl.set_validshape(dc_tile, [taken, d_deal_size])
    pl.set_validshape(kv_tile, [n_tile_rows, d_deal_size])
    pl.set_validshape(sm_tile, [n_tile_rows, d_deal_size])
    pl.load(dc_tile, d_cmp_kv_gm, [global_blk_idx, n_start])
    pl.load(kv_tile, kv_gm, [global_blk_idx, 0, n_start])
    pl.load(sm_tile, softmax_score_gm, [global_blk_idx, 0, n_start])
    pl.set_validshape(dc_f32, [taken, d_deal_size])
    pl.cast(dc_f32, dc_tile, mode=pl.RoundMode.CAST_ROUND)
    _vf_scatter_backward(
        dc_f32,
        kv_tile,
        sm_tile,
        dkv_tile,
        taken,
        cmp_row_cnt,
        d_deal_size,
    )
    _process_mask_fill(
        sm_tile, dkv_tile, b_start_pos, sc_idx, cmp_ratio, d_deal_size, taken
    )


def _process_scatter(
    vec_ctx,
    round_idx,
    total_sc_num_per_round,
    pre_deal_tc_size,
    deal_tc_size,
    b_idx,
    sc_idx,
    tile_dc_f16,
    tile_dc_f32,
    tile_kv,
    tile_softmax,
    tile_temp,
    d_cmp_kv_gm,
    kv_gm,
    softmax_score_gm,
    n_start,
    start_pos,
    seq_used,
    cu_seqlens,
    seq_ctx,
):
    """Phase 1: scatter（按 batch 边界推进块索引，逐 taken 块 load/cast/VF）"""
    cmp_ratio = vec_ctx.cmp_ratio
    d_deal_size = vec_ctx.d_deal_size
    batch_size = vec_ctx.batch_size
    processed = 0
    while processed < deal_tc_size:
        b_start_pos = _get_start_pos(b_idx, start_pos, seq_ctx)
        sq_val = _get_seq_used(b_idx, seq_used, cu_seqlens, seq_ctx)
        cmp_limit = (b_start_pos + sq_val) // cmp_ratio * cmp_ratio
        # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（_ceil_div）
        blk_in_batch = _ceil_div(cmp_limit - b_start_pos, cmp_ratio)
        remaining = blk_in_batch - sc_idx
        if remaining > 0:
            taken = min(deal_tc_size - processed, remaining)
            _process_scatter_slice(
                vec_ctx,
                round_idx,
                total_sc_num_per_round,
                pre_deal_tc_size,
                processed,
                b_idx,
                sc_idx,
                taken,
                n_start,
                b_start_pos,
                tile_dc_f16,
                tile_dc_f32,
                tile_kv,
                tile_softmax,
                tile_temp,
                d_cmp_kv_gm,
                kv_gm,
                softmax_score_gm,
            )
            processed += taken
            sc_idx += taken
        b_idx, sc_idx, ended = _advance_block_boundary(
            b_idx, sc_idx, blk_in_batch, batch_size
        )
        if ended:
            break


def _advance_block_boundary(b_idx, sc_idx, blk_in_batch, batch_size):
    """块遍历记账推进：批内 sc 耗尽则换批；越界返回 ended 哨兵供调用方 break。
    返回 (b_idx, sc_idx, ended)。"""
    if sc_idx >= blk_in_batch:
        sc_idx = 0
        b_idx += 1
    ended = b_idx >= batch_size
    return (b_idx, sc_idx, ended)


def _arrange_x_zero_fill(
    move_x_tiles, x_arrange_gm, intra_group_idx, rows, dst_row, hidden_size
):
    """xArrangeGm 指定区域显式清零（防 mm2 读到未初始化 workspace 的 NaN 位模式）"""
    for hi in pl.range(0, hidden_size, D_BASE_SIZE * 2):
        mx_tie = move_x_tiles.next()
        pl.set_validshape(mx_tie, [rows, D_BASE_SIZE * 2])
        if DataType == 0:
            _vf_tile_zero_bf16(mx_tie, rows, D_BASE_SIZE * 2)
        else:
            _vf_tile_zero_f16(mx_tie, rows, D_BASE_SIZE * 2)
        pl.store(x_arrange_gm, mx_tie, [intra_group_idx, dst_row, hi])


def _arrange_x_copy_zone(
    move_x_tiles, src_gm, dst_gm, intra_group_idx, rows, src_row, dst_row, hidden_size
):
    """xArrangeGm 区域拷贝（load src_gm → store dst_gm，源可为 x_gm 或 xArrangeGm）"""
    for hi in pl.range(0, hidden_size, D_BASE_SIZE * 2):
        mx_tie = move_x_tiles.next()
        pl.set_validshape(mx_tie, [rows, D_BASE_SIZE * 2])
        pl.load(mx_tie, src_gm, [src_row, hi])
        pl.store(dst_gm, mx_tie, [intra_group_idx, dst_row, hi])


def _arrange_x_copy_zone_self(
    move_x_tiles, src_gm, dst_gm, intra_group_idx, rows, src_row, dst_row, hidden_size
):
    """xArrangeGm 区域拷贝（load src_gm → store dst_gm，源可为 x_gm 或 xArrangeGm）"""
    for hi in pl.range(0, hidden_size, D_BASE_SIZE * 2):
        mx_tie = move_x_tiles.next()
        pl.set_validshape(mx_tie, [rows, D_BASE_SIZE * 2])
        pl.load(mx_tie, src_gm, [intra_group_idx, src_row, hi])
        pl.store(dst_gm, mx_tie, [intra_group_idx, dst_row, hi])


def _arrange_x_prev_head(
    vec_ctx,
    intra_group_idx,
    db_idx,
    group_idx,
    src_idx,
    move_x_tiles,
    x_gm,
    x_arrange_gm,
):
    """coff=2 prev 头部补写：前 cr-srcIdx 行清零（cache 历史），后 srcIdx 行拷贝上一块 cur"""
    cmp_ratio = vec_ctx.cmp_ratio
    hidden_size = vec_ctx.hidden_size
    head_dst = db_idx * vec_ctx.db_row_cnt + group_idx * vec_ctx.group_row_stride
    head_zero = cmp_ratio - src_idx if src_idx < cmp_ratio else 0
    if head_zero > 0:
        _arrange_x_zero_fill(
            move_x_tiles,
            x_arrange_gm,
            intra_group_idx,
            head_zero,
            head_dst,
            hidden_size,
        )
    head_src = max(src_idx - cmp_ratio, 0)
    head_rows = cmp_ratio - head_zero
    if head_rows > 0:
        _arrange_x_copy_zone(
            move_x_tiles,
            x_gm,
            x_arrange_gm,
            intra_group_idx,
            head_rows,
            head_src,
            head_dst + head_zero,
            hidden_size,
        )


def _arrange_x_batch_slice(
    vec_ctx,
    intra_group_idx,
    db_idx,
    group_idx,
    db_row_cnt,
    b_idx,
    sc_idx,
    sub_slot,
    processed,
    taken,
    move_x_tiles,
    x_gm,
    x_arrange_gm,
    start_pos,
    seq_used,
    cu_seqlens,
    seq_ctx,
):
    """单个 taken 块：槽位计算 + 首块部分块清零 + coff=2 prev 头部 + 主拷贝"""
    cmp_ratio = vec_ctx.cmp_ratio
    hidden_size = vec_ctx.hidden_size
    b_start_pos = _get_start_pos(b_idx, start_pos, seq_ctx)
    # 块槽位相对化：group 区域步长用实际值 groupRowStride（= gs*cr + coff=2 头部 cr），
    # 与 mm2 mStart 逐字节对齐；槽位 = subSlot(coff=1 子核偏移) + processed(块序)
    dst_idx = (
        db_idx * db_row_cnt
        + group_idx * vec_ctx.group_row_stride
        + (sub_slot + processed) * cmp_ratio
        + (Coff - 1) * cmp_ratio
    )
    mxs_idx = 0 if sc_idx == 0 else sc_idx * cmp_ratio - (b_start_pos % cmp_ratio)
    src_idx = _get_token_idx(b_idx, mxs_idx, cu_seqlens, seq_ctx)
    mx_rows = taken * cmp_ratio
    if mxs_idx == 0:
        mx_rows -= b_start_pos % cmp_ratio
        dst_idx += b_start_pos % cmp_ratio
    if mxs_idx == 0 and b_start_pos % cmp_ratio > 0:
        # 首块部分块：cur 区域前 bStartPos%cr 行 = cache 历史（对应 dkv 已被
        # scatter mask 为 0，matmul 贡献恒 0），显式清零——防 mm2 读到未初始化
        # workspace（FP32 随机位按 FP16 视图解释 → NaN/Inf → 0×NaN 传染
        # d_wkv/d_wgate）
        zero_rows = b_start_pos % cmp_ratio
        cur_base = dst_idx - zero_rows
        _arrange_x_zero_fill(
            move_x_tiles,
            x_arrange_gm,
            intra_group_idx,
            zero_rows,
            cur_base,
            hidden_size,
        )
    if Coff == 2 and processed == 0 and group_idx > 0:
        _arrange_x_prev_head(
            vec_ctx,
            intra_group_idx,
            db_idx,
            group_idx,
            src_idx,
            move_x_tiles,
            x_gm,
            x_arrange_gm,
        )
    _arrange_x_copy_zone(
        move_x_tiles,
        x_gm,
        x_arrange_gm,
        intra_group_idx,
        mx_rows,
        src_idx,
        dst_idx,
        hidden_size,
    )


def _arrange_x(
    vec_ctx,
    intra_group_idx,
    db_idx,
    group_idx,
    db_row_cnt,
    deal_tc_size,
    b_idx,
    sc_idx,
    sub_slot,
    move_x_tiles,
    x_gm,
    x_arrange_gm,
    start_pos,
    seq_used,
    cu_seqlens,
    seq_ctx,
):
    """Phase 1: x 搬运（cache copy + 块排列 + coff=2 头部补写）"""
    cmp_ratio = vec_ctx.cmp_ratio
    hidden_size = vec_ctx.hidden_size
    batch_size = vec_ctx.batch_size
    db_ratio = vec_ctx.db_ratio
    # cache copy（coff=2, group 0 跨 db 头部）：源 = 上一 db 区域最后一个块的 cr 行。
    # 紧凑布局下区域排满，末块起点恰好 = dbRowCnt − cmpRatio。
    if Coff == 2 and group_idx == 0 and sc_idx != 0:
        first_src_idx = (
            ((db_idx + db_ratio - 1) % db_ratio) * db_row_cnt + db_row_cnt - cmp_ratio
        )
        first_dst_idx = db_idx * db_row_cnt
        _arrange_x_copy_zone_self(
            move_x_tiles,
            x_arrange_gm,
            x_arrange_gm,
            intra_group_idx,
            cmp_ratio,
            first_src_idx,
            first_dst_idx,
            hidden_size,
        )
    elif Coff == 2 and group_idx == 0 and sc_idx == 0:
        # 轮首/批次首块：prev 头部无有效数据（对应 dkv prev 半区已被 scatter mask
        # 为 0，matmul 贡献恒 0），显式清零——防止 mm2 读到未初始化 workspace
        # （FP32 随机位模式按 FP16 视图解释 → NaN/Inf → 0×NaN 传染 d_wkv/d_wgate）
        _arrange_x_zero_fill(
            move_x_tiles,
            x_arrange_gm,
            intra_group_idx,
            cmp_ratio,
            db_idx * db_row_cnt,
            hidden_size,
        )
    processed = 0
    while processed < deal_tc_size:
        b_start_pos = _get_start_pos(b_idx, start_pos, seq_ctx)
        b_seq_used = _get_seq_used(b_idx, seq_used, cu_seqlens, seq_ctx)
        cmp_limit = (b_start_pos + b_seq_used) // cmp_ratio * cmp_ratio
        # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（_ceil_div）
        blk_in_batch = _ceil_div(cmp_limit - b_start_pos, cmp_ratio)
        remaining = blk_in_batch - sc_idx
        if remaining > 0:
            taken = min(deal_tc_size - processed, remaining)
            _arrange_x_batch_slice(
                vec_ctx,
                intra_group_idx,
                db_idx,
                group_idx,
                db_row_cnt,
                b_idx,
                sc_idx,
                sub_slot,
                processed,
                taken,
                move_x_tiles,
                x_gm,
                x_arrange_gm,
                start_pos,
                seq_used,
                cu_seqlens,
                seq_ctx,
            )
            processed += taken
            sc_idx += taken
        b_idx, sc_idx, ended = _advance_block_boundary(
            b_idx, sc_idx, blk_in_batch, batch_size
        )
        if ended:
            break


def _accumulate_ape(
    vec_ctx,
    round_idx,
    core_idx,
    group_idx,
    n_start,
    tile_ape_local,
    tile_softmax,
    tensor_ape,
    deal_tc_size,
    cmp_row_cnt,
    d_deal_size,
    coff_coef,
):
    """APE 局部累加 + 写 workspace（round0 清零 / round>0 读旧值，VF 内累加，一次 MTE3）"""
    local_slot = group_idx * coff_coef * cmp_row_cnt
    if Coff == 1 and core_idx % 2 == 1:
        local_slot += cmp_row_cnt
    pl.set_validshape(tile_ape_local, [cmp_row_cnt, d_deal_size])
    if round_idx == 0:
        _vf_tile_zero(tile_ape_local, cmp_row_cnt, d_deal_size)
    else:
        pl.load(tile_ape_local, tensor_ape, [local_slot, n_start])
    _vf_reduce_dscore_to_ape(
        tile_softmax,
        tile_ape_local,
        deal_tc_size,
        cmp_row_cnt,
        d_deal_size,
        d_deal_size,
    )
    pl.store(tensor_ape, tile_ape_local, [local_slot, n_start])


# ================================================================
#  Phase 1/2/3 循环骨架函数（从 kernel 主函数 round 循环提取）
# ================================================================
def _phase1_scatter(
    vec_ctx,
    round_idx,
    pre_deal_tc_size,
    deal_tc_size,
    b_idx,
    sc_idx,
    core_idx,
    sub_core_idx,
    group_idx,
    n_start,
    db_idx,
    cmp_ratio,
    cmp_row_cnt,
    d_deal_size,
    coff_coef,
    db_row_cnt,
    deal_sc_num,
    total_sc_num_per_round,
    tile_dc_f16,
    tile_dc_f32,
    tile_kv,
    tile_softmax,
    tile_temp,
    d_cmp_kv,
    kv,
    softmax_score,
    x,
    io_d_type,
    cmp_kv_rows,
    head_dim,
    cmp_size,
    token_size,
    work_space_ptr,
    group_num,
    intra_group_idx,
    tile_ape_local,
    tile_dkv_cast,
    tile_dkv_cast_md,
    tile_dkv_nz,
    tile_dkv_nz_md_prev,
    tile_dkv_nz_md_cur,
    tile_dsb_cast,
    tile_dsb_cast_md,
    tile_dsb_nz,
    tile_dsb_nz_md_prev,
    tile_dsb_nz_md_cur,
    temp_tile_group,
    l1_kv0_tile_group_nz,
    l1_kv1_tile_group_nz,
    l1_score0_tile_group_nz,
    l1_score1_tile_group_nz,
    move_x_tile_group,
    x_arrange_gm,
    start_pos,
    seq_used,
    cu_seqlens,
    seq_ctx,
):
    """Phase 1: scatter + APE 累加 + cast L1 + x 搬运（vec 侧，before → after 窗口）"""
    hidden_size = vec_ctx.hidden_size
    d_cmp_kv_gm = pl.make_tensor(
        d_cmp_kv, [cmp_kv_rows, head_dim], [head_dim, 1], dtype=io_d_type
    )
    kv_gm = pl.make_tensor(
        kv,
        [cmp_kv_rows, cmp_row_cnt, head_dim],
        [cmp_size, head_dim, 1],
        dtype=pl.DT_FP32,
    )
    softmax_score_gm = pl.make_tensor(
        softmax_score,
        [cmp_kv_rows, cmp_row_cnt, head_dim],
        [cmp_size, head_dim, 1],
        dtype=pl.DT_FP32,
    )
    x_gm = pl.make_tensor(
        x, [token_size, hidden_size], [hidden_size, 1], dtype=io_d_type
    )
    tensor_ape = pl.make_tensor(
        work_space_ptr,
        [group_num * coff_coef * cmp_row_cnt, head_dim],
        [head_dim, 1],
        dtype=pl.DT_FP32,
    )
    # scatter：before → after 窗口内逐 taken 块处理（b_idx/sc_idx 传值局部推进）
    _process_scatter(
        vec_ctx,
        round_idx,
        total_sc_num_per_round,
        pre_deal_tc_size,
        deal_tc_size,
        b_idx,
        sc_idx,
        tile_dc_f16,
        tile_dc_f32,
        tile_kv,
        tile_softmax,
        tile_temp,
        d_cmp_kv_gm,
        kv_gm,
        softmax_score_gm,
        n_start,
        start_pos,
        seq_used,
        cu_seqlens,
        seq_ctx,
    )

    n_tile_rows = deal_tc_size * cmp_row_cnt

    # APE: local_ape 累加 + 写 workspace（round0 清零 / round>0 读旧值，VF 内累加，一次 MTE3）
    _accumulate_ape(
        vec_ctx,
        round_idx,
        core_idx,
        group_idx,
        n_start,
        tile_ape_local,
        tile_softmax,
        tensor_ape,
        deal_tc_size,
        cmp_row_cnt,
        d_deal_size,
        coff_coef,
    )
    sub_idx = core_idx % 2
    # ── Cast dkv/dsb FP32→BF16/FP16 + ND→NZ + insert L1（子函数）──
    _cast_dkv_dsb_to_l1(
        vec_ctx,
        n_tile_rows,
        deal_tc_size,
        sub_idx,
        tile_temp,
        tile_dkv_cast,
        tile_dkv_cast_md,
        tile_dkv_nz,
        tile_dkv_nz_md_prev,
        tile_dkv_nz_md_cur,
        tile_dsb_cast,
        tile_dsb_cast_md,
        tile_dsb_nz,
        tile_softmax,
        tile_dsb_nz_md_prev,
        tile_dsb_nz_md_cur,
        temp_tile_group,
        l1_kv0_tile_group_nz,
        l1_kv1_tile_group_nz,
        l1_score0_tile_group_nz,
        l1_score1_tile_group_nz,
    )

    # 搬运x（subSlot: coff=1 的 M 轴子核槽位偏移；coff=2 恒 0）
    sub_slot = sub_core_idx * deal_sc_num if Coff == 1 else 0
    if Coff == 1 or sub_idx == 0:
        _arrange_x(
            vec_ctx,
            intra_group_idx,
            db_idx,
            group_idx,
            db_row_cnt,
            deal_tc_size,
            b_idx,
            sc_idx,
            sub_slot,
            move_x_tile_group,
            x_gm,
            x_arrange_gm,
            start_pos,
            seq_used,
            cu_seqlens,
            seq_ctx,
        )


def _phase2_matmul(
    round_idx,
    db_idx,
    deal_tc_size,
    hidden_size,
    head_dim,
    deal_sc_num,
    cube_m_base_size,
    cmp_ratio,
    cube_core_num,
    group_size,
    group_num,
    total_head_dim,
    db_row_cnt,
    group_row_stride,
    intra_group_idx,
    cv_l0a,
    cv_l0b,
    cv_l0c,
    l1_w_db,
    l1_kv0_tile_group_nz,
    l1_kv1_tile_group_nz,
    l1_score0_tile_group_nz,
    l1_score1_tile_group_nz,
    l1_kv0_tile_group_zn,
    l1_kv1_tile_group_zn,
    l1_score0_tile_group_zn,
    l1_score1_tile_group_zn,
    wkv,
    wgate,
    io_d_type,
    d_x_result_gm,
    x_arrange_gm,
    d_wkv_result_gm,
    d_w_gate_result_gm,
):
    """Phase 2: dkv@wkv + dsb@wgate → dX partial；dkv@x + dsb@x → dW partial"""
    wkv_t = pl.make_tensor(
        wkv, [total_head_dim, hidden_size], [hidden_size, 1], dtype=io_d_type
    )
    wgate_t = pl.make_tensor(
        wgate, [total_head_dim, hidden_size], [hidden_size, 1], dtype=io_d_type
    )
    l1_kv0 = l1_kv0_tile_group_nz.next()
    l1_kv1 = l1_kv1_tile_group_nz.next()
    l1_sb0 = l1_score0_tile_group_nz.next()
    l1_sb1 = l1_score1_tile_group_nz.next()

    l1_kv0_t = l1_kv0_tile_group_zn.next()
    l1_kv1_t = l1_kv1_tile_group_zn.next()
    l1_sb0_t = l1_score0_tile_group_zn.next()
    l1_sb1_t = l1_score1_tile_group_zn.next()

    n_tile_rows0 = (
        min(deal_tc_size, deal_sc_num) * cmp_ratio
        if Coff == 1
        else deal_tc_size * cmp_ratio
    )
    n_tile_rows1 = (
        max(deal_tc_size - deal_sc_num, 0) * cmp_ratio
        if Coff == 1
        else deal_tc_size * cmp_ratio
    )

    matmul_ctx = pl.struct(
        "MatmulCtx",
        hidden_size=hidden_size,
        n_tile_rows0=n_tile_rows0,
        n_tile_rows1=n_tile_rows1,
        head_dim=head_dim,
        deal_sc_num=deal_sc_num,
        cube_m_base_size=cube_m_base_size,
        cmp_ratio=cmp_ratio,
        cube_core_num=cube_core_num,
        group_size=group_size,
        group_num=group_num,
        total_head_dim=total_head_dim,
        db_row_cnt=db_row_cnt,
        group_row_stride=group_row_stride,
        intra_group_idx=intra_group_idx,
    )

    # ── Matmul #1/#2：提取子函数（coff=1/2 统一）──
    _compute_dx_partial(
        matmul_ctx,
        db_idx,
        cv_l0a,
        cv_l0b,
        cv_l0c,
        l1_w_db,
        l1_kv0,
        l1_kv1,
        l1_sb0,
        l1_sb1,
        wkv_t,
        wgate_t,
        d_x_result_gm,
    )
    _compute_dw_partial(
        matmul_ctx,
        round_idx,
        db_idx,
        cv_l0a,
        cv_l0b,
        cv_l0c,
        l1_w_db,
        l1_kv0_t,
        l1_kv1_t,
        l1_sb0_t,
        l1_sb1_t,
        x_arrange_gm,
        d_wkv_result_gm,
        d_w_gate_result_gm,
    )


def _phase3_reduce(
    vec_ctx,
    core_idx,
    group_idx,
    round_idx,
    round_cnt,
    max_round_blocks,
    prev_db_idx,
    prev_b_idx_start,
    prev_sc_idx_start,
    prev_b_idx_end,
    prev_sc_idx_end,
    reduce_acc,
    reduce_comp,
    reduce_load_tile_group,
    p3_cast,
    d_x_cache_gm,
    d_x_result_gm,
    d_x,
    d_wkv,
    d_wgate,
    d_ape,
    io_d_type,
    x_rows,
    total_head_dim,
    hidden_size,
    cmp_size,
    d_weight_work_space_size,
    work_space_ptr,
    d_w_kv_work_space_ptr,
    d_w_gate_work_space_ptr,
    start_pos,
    seq_used,
    cu_seqlens,
    seq_ctx,
):
    """Phase 3: dX reduce（上一轮 start/end 节点）+ 末轮 dApe/dW 一次性跨核归约"""
    d_x_out_gm = pl.make_tensor(
        d_x, [x_rows, hidden_size], [hidden_size, 1], dtype=io_d_type
    )
    tensor_d_ape_flat = pl.make_tensor(
        d_ape, [1, cmp_size], [cmp_size, 1], dtype=pl.DT_FP32
    )
    tensor_d_ape_ws_flat = pl.make_tensor(
        work_space_ptr,
        [vec_ctx.group_num * vec_ctx.coff_coef, cmp_size],
        [cmp_size, 1],
        dtype=pl.DT_FP32,
    )
    tensor_dw_kv_ws_flat = pl.make_tensor(
        d_w_kv_work_space_ptr,
        [
            d_weight_work_space_size // (total_head_dim * hidden_size),
            total_head_dim * hidden_size,
        ],
        [total_head_dim * hidden_size, 1],
        dtype=pl.DT_FP32,
    )
    tensor_dw_gate_ws_flat = pl.make_tensor(
        d_w_gate_work_space_ptr,
        [
            d_weight_work_space_size // (total_head_dim * hidden_size),
            total_head_dim * hidden_size,
        ],
        [total_head_dim * hidden_size, 1],
        dtype=pl.DT_FP32,
    )
    # dW 输出为 FP16/BF16（与 op 注册一致），归约在 FP32 完成、写回前 cast
    tensor_dw_kv_flat = pl.make_tensor(
        d_wkv,
        [1, total_head_dim * hidden_size],
        [total_head_dim * hidden_size, 1],
        dtype=io_d_type,
    )
    tensor_dw_gate_flat = pl.make_tensor(
        d_wgate,
        [1, total_head_dim * hidden_size],
        [total_head_dim * hidden_size, 1],
        dtype=io_d_type,
    )
    _reduce_dx(
        vec_ctx,
        core_idx,
        prev_db_idx,
        round_idx - 1,
        prev_b_idx_start,
        prev_sc_idx_start,
        prev_b_idx_end,
        prev_sc_idx_end,
        reduce_acc,
        reduce_load_tile_group,
        p3_cast,
        d_x_cache_gm,
        d_x_result_gm,
        d_x_out_gm,
        start_pos,
        seq_used,
        cu_seqlens,
        seq_ctx,
    )

    # ── 末轮：dApe/dW 一次性跨核归约（生产者侧已按轮累加至 ws 固定槽）──
    # roundBlocks 传 maxRoundBlocks：validRows 只覆盖写过数据的行
    # （不满轮时 groupIdx >= ceil(maxRoundBlocks/groupDealScNum) 的核未执行
    #  Phase 1/2 写入，其 ws 区域是初始值，不能参与归约）
    if round_idx == round_cnt:
        _reduce_ape(
            vec_ctx,
            core_idx,
            group_idx,
            0,
            max_round_blocks,
            reduce_acc,
            reduce_comp,
            reduce_load_tile_group,
            tensor_d_ape_flat,
            tensor_d_ape_ws_flat,
        )
        _reduce_d_weight(
            vec_ctx,
            core_idx,
            group_idx,
            0,
            max_round_blocks,
            reduce_acc,
            reduce_load_tile_group,
            p3_cast,
            tensor_dw_kv_flat,
            tensor_dw_kv_ws_flat,
            tensor_dw_gate_flat,
            tensor_dw_gate_ws_flat,
        )


def _init_scalars(batch_size, seq_size, cmp_ratio, start_pos, seq_used, cu_seqlens):
    """Init: seq_ctx struct + compressed_cnt（依赖 startPos/seqUsed 数据内容）"""

    # ── Sequence context ──
    seq_ctx = pl.struct(
        "SeqCtx",
        batch_size=batch_size,
        cmp_ratio=cmp_ratio,
        layout=Layout,
        seq_size=seq_size,
    )

    # ── Compressed rows / block counts ──
    # compressedCnt 依赖 startPos/seqUsed 数据内容（op_host tiling 阶段读不到），kernel 内算

    compressed_cnt = 0
    for b_idx in pl.range(0, batch_size):
        b_start_pos = _get_start_pos(b_idx, start_pos, seq_ctx)
        b_seq_used = _get_seq_used(b_idx, seq_used, cu_seqlens, seq_ctx)
        compress_seq_idx = _trunc(b_start_pos + b_seq_used, cmp_ratio)
        if compress_seq_idx > b_start_pos:
            # startPos%cmpRatio!=0 时首块为部分块但仍计 1 块 → 向上取整（_ceil_div）
            compressed_cnt = compressed_cnt + _ceil_div(
                compress_seq_idx - b_start_pos, cmp_ratio
            )
    return (seq_ctx, compressed_cnt)


def _init_round_params(
    compressed_cnt,
    batch_size,
    cmp_ratio,
    head_dim,
    hidden_size,
    total_head_dim,
    cmp_row_cnt,
    cmp_size,
    d_deal_size,
    m_deal_size,
    db_ratio,
    coff_coef,
    cube_m_base_size,
    group_deal_sc_num,
    deal_sc_num,
    cmp_kv_batch_stride,
    cube_core_num,
    core_num,
    group_size,
    group_num,
    db_row_cnt,
    group_row_stride,
    total_sc_num_per_round,
):
    """Init: round 循环参数 + Vec 子函数共享 ctx"""
    # ── 共享 round 参数（Phase 1 + Phase 2 都需要）──

    round_cnt = _ceil_div(compressed_cnt, total_sc_num_per_round)
    # 单轮实际块数上限 = 首轮块数（各轮最大值）：归约侧有效行数依据
    # （不满轮时空闲 group 的 ws 区域未写，归约只读写过数据的行）
    max_round_blocks = min(total_sc_num_per_round, compressed_cnt)

    # ── Vec 子函数共享 ctx（int64 稳定值，派生配置经 tiling 传入）──
    vec_ctx = pl.struct(
        "VecCtx",
        hidden_size=hidden_size,
        batch_size=batch_size,
        cmp_ratio=cmp_ratio,
        head_dim=head_dim,
        cmp_row_cnt=cmp_row_cnt,
        cmp_size=cmp_size,
        d_deal_size=d_deal_size,
        m_deal_size=m_deal_size,
        db_ratio=db_ratio,
        coff_coef=coff_coef,
        cube_m_base_size=cube_m_base_size,
        group_deal_sc_num=group_deal_sc_num,
        deal_sc_num=deal_sc_num,
        cmp_kv_batch_stride=cmp_kv_batch_stride,
        cube_core_num=cube_core_num,
        core_num=core_num,
        group_size=group_size,
        group_num=group_num,
        total_head_dim=total_head_dim,
        db_row_cnt=db_row_cnt,
        group_row_stride=group_row_stride,
        round_cnt=round_cnt,
    )
    return (round_cnt, max_round_blocks, vec_ctx)


def _init_l1_tile_groups(io_d_type):
    """Init: L1A CV 通道 4 组 [128,128] 双缓冲 tile_group（Mat 内存，vec/cube 共享）"""
    # ── L1A CV 通道: 4 个独立 tile_group [128,128], double-buffer ──
    l1_nz_tile_type = pl.TileType(
        shape=[M_BASE_SIZE, D_BASE_SIZE],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.NZ,
        compact=1,
    )
    l1_zn_tile_type = pl.TileType(
        shape=[M_BASE_SIZE, D_BASE_SIZE],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Mat,
        layout=pl.ZN,
        compact=1,
    )

    l1_kv0_tile_group_nz = pl.make_tile_group(
        type=l1_nz_tile_type, addrs=L1_KV0_ADDR, mutex_ids=[10, 11]
    )  # 128*128*2 = 0x8000（双缓冲 2 槽 = 0x10000）
    l1_kv1_tile_group_nz = pl.make_tile_group(
        type=l1_nz_tile_type, addrs=L1_KV1_ADDR, mutex_ids=[12, 13]
    )
    l1_score0_tile_group_nz = pl.make_tile_group(
        type=l1_nz_tile_type, addrs=L1_SB0_ADDR, mutex_ids=[14, 15]
    )
    l1_score1_tile_group_nz = pl.make_tile_group(
        type=l1_nz_tile_type, addrs=L1_SB1_ADDR, mutex_ids=[16, 17]
    )

    l1_kv0_tile_group_zn = pl.make_tile_group(
        type=l1_zn_tile_type, addrs=L1_KV0_ADDR, mutex_ids=[10, 11]
    )
    l1_kv1_tile_group_zn = pl.make_tile_group(
        type=l1_zn_tile_type, addrs=L1_KV1_ADDR, mutex_ids=[12, 13]
    )
    l1_score0_tile_group_zn = pl.make_tile_group(
        type=l1_zn_tile_type, addrs=L1_SB0_ADDR, mutex_ids=[14, 15]
    )
    l1_score1_tile_group_zn = pl.make_tile_group(
        type=l1_zn_tile_type, addrs=L1_SB1_ADDR, mutex_ids=[16, 17]
    )
    return (
        l1_kv0_tile_group_nz,
        l1_kv1_tile_group_nz,
        l1_score0_tile_group_nz,
        l1_score1_tile_group_nz,
        l1_kv0_tile_group_zn,
        l1_kv1_tile_group_zn,
        l1_score0_tile_group_zn,
        l1_score1_tile_group_zn,
    )


def _init_vec_tile_groups(io_d_type, group_size, d_deal_size):
    """Init: UB tile/tile_group 声明 + vec 身份值"""
    #  Vec tile declarations (once)
    # ════════════════════════════════════════════════════════════

    core_idx = pl.get_block_idx()
    sub_core_idx = pl.get_subblock_idx()

    # ── UB tile type declarations ──
    # TileType 静态 shape 需编译期立即量 → 用内联算术（值恒等于 tiling.dDealSize/mDealSize）
    d_cmp_kv_tile_type = pl.TileType(
        shape=[M_BASE_SIZE * Coff // 2, D_BASE_SIZE // Coff],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )

    d_cmp_kv_fp32_tile_type = pl.TileType(
        shape=[M_BASE_SIZE * Coff // 2, D_BASE_SIZE // Coff],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        pad=pl.TilePad.zero,
    )

    normel_tile_type = pl.TileType(
        shape=[M_BASE_SIZE * Coff, D_BASE_SIZE // Coff],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        pad=pl.TilePad.zero,
        compact=1,
    )

    cast_tile_type = pl.TileType(
        shape=[M_BASE_SIZE * Coff, D_BASE_SIZE // Coff],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        compact=1,
    )

    cast_md_tile_type = pl.TileType(
        shape=[M_BASE_SIZE, D_BASE_SIZE],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        compact=1,
    )

    nz_tile_type = pl.TileType(
        shape=[M_BASE_SIZE * Coff, D_BASE_SIZE // Coff],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        layout=pl.NZ,
        compact=1,
    )

    nz_md_tile_type = pl.TileType(
        shape=[M_BASE_SIZE // 2 * Coff, D_BASE_SIZE // Coff],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        layout=pl.NZ,
        compact=1,
    )

    move_x_tile_type = pl.TileType(
        shape=[M_BASE_SIZE, D_BASE_SIZE * 2],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        pad=pl.TilePad.zero,
    )

    flat_tile_type = pl.TileType(
        shape=[1, M_BASE_SIZE * D_BASE_SIZE],
        dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        pad=pl.TilePad.zero,
    )

    flat_cast_tile_type = pl.TileType(
        shape=[1, M_BASE_SIZE * D_BASE_SIZE],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )

    temp_tile_type = pl.TileType(
        shape=[M_BASE_SIZE, D_BASE_SIZE // 2],
        dtype=io_d_type,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        compact=1,
    )

    # ── UB tile_group allocation (auto_mutex for shared addresses) ──
    #          | group             | addr      | size（coff=2 上限）| mutex |
    #          | dc_f16            | 0x00000   | 16384  | [0]   |
    #          | dc_f32            | 0x04000   | 32768  | [1]   |
    #          | kv / dsw          | 0x0C000   | 65536  | [2]   |
    #          | softmax          | 0x1C000   | 65536  | [3]   |
    #          | dkv (temp)       | 0x2C000   | 65536  | [4]   |
    #          | dsb_cast         | 0x1C000   | 32768  | [3]   |
    #          | dsb_nz           | 0x24000   | 32768  | [5]   |
    #          | dkv_cast         | 0x2C000   | 32768  | [4]   |
    #          | dkv_nz           | 0x34000   | 32768  | [6]   |
    d_cmp_kv_tile_group = pl.make_tile_group(
        type=d_cmp_kv_tile_type, addrs=[UB_DC_F16_ADDR], mutex_ids=[0]
    )
    d_cmp_kv_fp32_tile_group = pl.make_tile_group(
        type=d_cmp_kv_fp32_tile_type, addrs=[UB_DC_F32_ADDR], mutex_ids=[1]
    )  # 16kB = 0x4000

    kv_tile_group = pl.make_tile_group(
        type=normel_tile_type, addrs=[UB_KV_APE_ADDR], mutex_ids=[2]
    )  # 32kB = 0x8000
    softmax_score_tile_group = pl.make_tile_group(
        type=normel_tile_type, addrs=[UB_SM_DSB_ADDR], mutex_ids=[3]
    )  # 64kb = 0x10000
    d_kv_tile_group = pl.make_tile_group(
        type=normel_tile_type, addrs=[UB_DKV_ADDR], mutex_ids=[4]
    )

    d_score_cast_tile_group = pl.make_tile_group(
        type=cast_tile_type, addrs=[UB_SM_DSB_ADDR], mutex_ids=[3]
    )
    d_kv_cast_tile_group = pl.make_tile_group(
        type=cast_tile_type, addrs=[UB_DKV_ADDR], mutex_ids=[4]
    )

    ape_tile_group = pl.make_tile_group(
        type=normel_tile_type, addrs=[UB_KV_APE_ADDR], mutex_ids=[2]
    )

    d_score_cast_md_tile_group = pl.make_tile_group(
        type=cast_md_tile_type, addrs=[UB_SM_DSB_ADDR], mutex_ids=[3]
    )
    d_kv_cast_md_tile_group = pl.make_tile_group(
        type=cast_md_tile_type, addrs=[UB_DKV_ADDR], mutex_ids=[4]
    )

    d_score_nz_tile_group = pl.make_tile_group(
        type=nz_tile_type, addrs=[UB_DSB_NZ_ADDR], mutex_ids=[3]
    )
    d_kv_nz_tile_group = pl.make_tile_group(
        type=nz_tile_type, addrs=[UB_DKV_NZ_ADDR], mutex_ids=[4]
    )

    d_score_nz_md_prev_tile_group = pl.make_tile_group(
        type=nz_md_tile_type, addrs=[UB_DSB_NZ_ADDR], mutex_ids=[3]
    )
    d_kv_nz_md_prev_tile_group = pl.make_tile_group(
        type=nz_md_tile_type, addrs=[UB_DKV_NZ_ADDR], mutex_ids=[4]
    )
    d_score_nz_md_cur_tile_group = pl.make_tile_group(
        type=nz_md_tile_type, addrs=[UB_DSB_NZ_MD_CUR_ADDR], mutex_ids=[3]
    )
    d_kv_nz_md_cur_tile_group = pl.make_tile_group(
        type=nz_md_tile_type, addrs=[UB_DKV_NZ_MD_CUR_ADDR], mutex_ids=[4]
    )

    temp_tile_group = pl.make_tile_group(
        type=temp_tile_type, addrs=UB_DC_F16_ADDR, mutex_ids=[0, 1]
    )  # 16kB = 0x4000

    move_x_tile_group = pl.make_tile_group(
        type=move_x_tile_type, addrs=[UB_SM_DSB_ADDR, UB_DKV_ADDR], mutex_ids=[3, 4]
    )

    # Phase 3 reduce: 复用 0x0C000/0x1C000/0x2C000 三个 64KB 区域
    reduce_acc_tile_group = pl.make_tile_group(
        type=flat_tile_type, addrs=[UB_KV_APE_ADDR], mutex_ids=[2]
    )
    # reduce Kahan 补偿项（Phase 3 归约累加器；复用 Phase 1 后已释放的 DC_F16 区域）
    reduce_comp_tile_group = pl.make_tile_group(
        type=flat_tile_type, addrs=[UB_DC_F16_ADDR], mutex_ids=[0]
    )
    reduce_load_tile_group = pl.make_tile_group(
        type=flat_tile_type, addrs=[UB_SM_DSB_ADDR, UB_DKV_ADDR], mutex_ids=[3, 4]
    )

    # Phase 3 dX reduce: acc/ld 复用 reduce_acc/reduce_ld_group，仅新增 cast
    reduce_result_tile_group = pl.make_tile_group(
        type=flat_cast_tile_type, addrs=[UB_KV_APE_ADDR], mutex_ids=[2]
    )

    tile_dc_f16 = d_cmp_kv_tile_group.next()
    tile_dc_f32 = d_cmp_kv_fp32_tile_group.next()
    tile_kv = kv_tile_group.next()
    tile_softmax = softmax_score_tile_group.next()
    tile_dsb_cast = d_score_cast_tile_group.next()
    tile_dsb_cast_md = d_score_cast_md_tile_group.next()
    tile_temp = d_kv_tile_group.next()
    tile_dkv_cast = d_kv_cast_tile_group.next()
    tile_dkv_cast_md = d_kv_cast_md_tile_group.next()
    tile_ape_local = ape_tile_group.next()
    tile_dkv_nz = d_kv_nz_tile_group.next()
    tile_dsb_nz = d_score_nz_tile_group.next()

    tile_dkv_nz_md_prev = d_kv_nz_md_prev_tile_group.next()
    tile_dsb_nz_md_prev = d_score_nz_md_prev_tile_group.next()
    tile_dkv_nz_md_cur = d_kv_nz_md_cur_tile_group.next()
    tile_dsb_nz_md_cur = d_score_nz_md_cur_tile_group.next()

    reduce_acc = reduce_acc_tile_group.next()
    reduce_comp = reduce_comp_tile_group.next()

    p3_cast = reduce_result_tile_group.next()

    # ── D group division ──
    cube_core_idx = core_idx // 2
    group_idx = cube_core_idx // group_size
    intra_group_idx = cube_core_idx % group_size

    n_start = intra_group_idx * D_BASE_SIZE + (sub_core_idx % Coff) * d_deal_size
    return (
        core_idx,
        sub_core_idx,
        cube_core_idx,
        group_idx,
        intra_group_idx,
        n_start,
        tile_dc_f16,
        tile_dc_f32,
        tile_kv,
        tile_softmax,
        tile_dsb_cast,
        tile_dsb_cast_md,
        tile_temp,
        tile_dkv_cast,
        tile_dkv_cast_md,
        tile_ape_local,
        tile_dkv_nz,
        tile_dsb_nz,
        tile_dkv_nz_md_prev,
        tile_dsb_nz_md_prev,
        tile_dkv_nz_md_cur,
        tile_dsb_nz_md_cur,
        reduce_acc,
        reduce_comp,
        p3_cast,
        move_x_tile_group,
        reduce_load_tile_group,
        reduce_acc_tile_group,
        reduce_comp_tile_group,
        reduce_result_tile_group,
        temp_tile_group,
    )


def _init_cube_tile_groups(io_d_type, group_size):
    """Init: L1W/L0 tile_group + cube 身份值（wkv/wgate 视图在 _phase2_matmul 内创建）"""
    #  Cube tile declarations (once)
    # ═════════════════════════════════════════════════════════════════════════

    cube_core_idx = pl.get_block_idx()
    group_idx = cube_core_idx // group_size
    intra_group_idx = cube_core_idx % group_size

    n_start = intra_group_idx * D_BASE_SIZE

    # L1A CV 数据: 复用主函数体 tile_group, 通过 .next() db

    # L1W: wkv/wgate chunk 加载（double-buffer, 0x40000-0x4FFFF, 64KB）
    l1_w_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[BASE_SIZE, H_L1_BASE_SIZE],
            dtype=io_d_type,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=L1_W_ADDR,
        mutex_ids=[20, 21],
    )

    # L0 tiles
    cv_l0a = pl.make_tile_group(
        type=pl.TileType(
            shape=[BASE_SIZE, BASE_SIZE],
            dtype=io_d_type,
            target_memory=pl.MemorySpace.Left,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=[0x00000, 0x08000],
        mutex_ids=[18, 19],
    )
    cv_l0b = pl.make_tile_group(
        type=pl.TileType(
            shape=[BASE_SIZE, H_BASE_SIZE],
            dtype=io_d_type,
            target_memory=pl.MemorySpace.Right,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=[0x00000, 0x08000],
        mutex_ids=[22, 23],
    )
    cv_l0c = pl.make_tile_group(
        type=pl.TileType(
            shape=[BASE_SIZE, H_BASE_SIZE],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            layout=pl.NZ,
            fractal=1024,
            compact=1,
        ),
        addrs=0x00000,
        mutex_ids=[24, 25, 26, 27],
    )

    # ═════════════════════════════════════════════════════════════════════════
    return (cube_core_idx, group_idx, intra_group_idx, l1_w_db, cv_l0a, cv_l0b, cv_l0c)


@pl.jit(tiling_key=CompressorGradTilingKey, auto_mutex=True, arch="a5")
def compressor_grad(
    # ── 输入（前向参数） ──
    x: pl.Ptr[pl.DT_UINT8],  # (T, hiddenSize)
    wkv: pl.Ptr[pl.DT_UINT8],  # (C, hiddenSize)
    wgate: pl.Ptr[pl.DT_UINT8],  # (C, hiddenSize)
    # ── 输入（前向中间变量 / 上游梯度） ──
    d_cmp_kv: pl.Ptr[pl.DT_UINT8],  # (cmpKvRows, headDim)
    softmax_score: pl.Ptr[pl.DT_UINT8],  # (cmpKvRows, cmpRowCnt, headDim) FP32
    kv: pl.Ptr[pl.DT_UINT8],  # (cmpKvRows, cmpRowCnt, headDim) FP32
    # ── 位置 / 序列参数 ──
    cu_seqlens: pl.Ptr[pl.DT_UINT8],
    seq_used: pl.Ptr[pl.DT_UINT8],
    start_pos: pl.Ptr[pl.DT_UINT8],
    # ── 输出（最终梯度） ──
    d_x: pl.Ptr[pl.DT_UINT8],  # (T, hiddenSize) FP16/BF16 输出
    d_wkv: pl.Ptr[pl.DT_UINT8],  # (C, hiddenSize) FP16/BF16 输出
    d_wgate: pl.Ptr[pl.DT_UINT8],  # (C, hiddenSize) FP16/BF16 输出
    d_ape: pl.Ptr[pl.DT_UINT8],  # (cmpRatio, coff*D) FP32 输出
    # ── 工作空间 ──
    workspace: pl.Ptr[
        pl.DT_UINT8
    ],  # 各 phase 中间结果（分区布局见 op_host CalcWorkSpace）
    # ── Tiling ──
    tiling: CompressorGradTiling,
):
    io_d_type = pl.DT_BF16 if DataType == 0 else pl.DT_FP16
    db_ratio = 2  # workspace 双缓冲
    group_row_stride = tiling.group_row_stride

    # ═══ Init：标量派生 / round 参数（普通函数返回元组）═══
    (seq_ctx, compressed_cnt) = _init_scalars(
        tiling.batch_size,
        tiling.seq_size,
        tiling.cmp_ratio,
        start_pos,
        seq_used,
        cu_seqlens,
    )
    (round_cnt, max_round_blocks, vec_ctx) = _init_round_params(
        compressed_cnt,
        tiling.batch_size,
        tiling.cmp_ratio,
        tiling.head_dim,
        tiling.hidden_size,
        tiling.total_head_dim,
        tiling.cmp_row_cnt,
        tiling.cmp_size,
        tiling.d_deal_size,
        tiling.m_deal_size,
        db_ratio,
        tiling.coff_coef,
        tiling.cube_m_base_size,
        tiling.group_deal_sc_num,
        tiling.deal_sc_num,
        tiling.cmp_kv_batch_stride,
        tiling.cube_core_num,
        tiling.core_num,
        tiling.group_size,
        tiling.group_num,
        tiling.db_row_cnt,
        group_row_stride,
        tiling.total_sc_num_per_round,
    )

    # ── Workspace 分块: APE 区域 + dX 区域──
    ape_work_space_size = tiling.dape_ws_size
    d_x_work_space_size = tiling.d_x_ws_size
    d_weight_work_space_size = tiling.d_w_weight_ws_size
    x_work_space_size = tiling.x_ws_size
    d_x_cache_work_space_size = tiling.d_x_cache_ws_size

    work_space_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    d_x_work_space_ptr = pl.addptr(work_space_ptr, ape_work_space_size)
    d_w_kv_work_space_ptr = pl.addptr(d_x_work_space_ptr, d_x_work_space_size)
    d_w_gate_work_space_ptr = pl.addptr(d_w_kv_work_space_ptr, d_weight_work_space_size)
    x_work_space_ptr = pl.addptr(d_w_gate_work_space_ptr, d_weight_work_space_size)
    d_x_cache_ptr = pl.addptr(x_work_space_ptr, x_work_space_size)

    # ── dX output + workspace views（跨 phase 共享，Phase 1/3 其余视图在函数内创建）──
    d_x_result_gm = pl.make_tensor(
        d_x_work_space_ptr,
        [Coff, d_x_work_space_size // tiling.hidden_size // Coff, tiling.hidden_size],
        [d_x_work_space_size // Coff, tiling.hidden_size, 1],
        dtype=pl.DT_FP32,
    )
    d_wkv_result_gm = pl.make_tensor(
        d_w_kv_work_space_ptr,
        [d_weight_work_space_size // tiling.hidden_size, tiling.hidden_size],
        [tiling.hidden_size, 1],
        dtype=pl.DT_FP32,
    )
    d_w_gate_result_gm = pl.make_tensor(
        d_w_gate_work_space_ptr,
        [d_weight_work_space_size // tiling.hidden_size, tiling.hidden_size],
        [tiling.hidden_size, 1],
        dtype=pl.DT_FP32,
    )
    x_arrange_gm = pl.make_tensor(
        x_work_space_ptr,
        [
            tiling.group_size,
            x_work_space_size // tiling.group_size // tiling.hidden_size,
            tiling.hidden_size,
        ],
        [x_work_space_size // tiling.group_size, tiling.hidden_size, 1],
        dtype=io_d_type,
    )
    d_x_cache_gm = pl.make_tensor(
        d_x_cache_ptr,
        [d_x_cache_work_space_size // tiling.hidden_size, tiling.hidden_size],
        [tiling.hidden_size, 1],
        dtype=pl.DT_FP32,
    )

    # ── L1A CV 通道（vec insert / cube 读共享）──
    (
        l1_kv0_tile_group_nz,
        l1_kv1_tile_group_nz,
        l1_score0_tile_group_nz,
        l1_score1_tile_group_nz,
        l1_kv0_tile_group_zn,
        l1_kv1_tile_group_zn,
        l1_score0_tile_group_zn,
        l1_score1_tile_group_zn,
    ) = _init_l1_tile_groups(io_d_type)

    # ── Per-round shared state (persists across rounds) ──
    b_idx, sc_idx = 0, 0
    prev_db_idx, prev_b_idx_start, prev_sc_idx_start = 0, 0, 0
    prev_b_idx_end, prev_sc_idx_end = 0, 0
    cur_db_idx, cur_b_idx_start, cur_sc_idx_start = 0, 0, 0
    cur_b_idx_end, cur_sc_idx_end = 0, 0

    # ════════════════════════════════════════════════════════════
    #  Vec tile declarations (once)
    # ════════════════════════════════════════════════════════════
    with pl.section_vector():
        (
            core_idx,
            sub_core_idx,
            cube_core_idx,
            group_idx,
            intra_group_idx,
            n_start,
            tile_dc_f16,
            tile_dc_f32,
            tile_kv,
            tile_softmax,
            tile_dsb_cast,
            tile_dsb_cast_md,
            tile_temp,
            tile_dkv_cast,
            tile_dkv_cast_md,
            tile_ape_local,
            tile_dkv_nz,
            tile_dsb_nz,
            tile_dkv_nz_md_prev,
            tile_dsb_nz_md_prev,
            tile_dkv_nz_md_cur,
            tile_dsb_nz_md_cur,
            reduce_acc,
            reduce_comp,
            p3_cast,
            move_x_tile_group,
            reduce_load_tile_group,
            reduce_acc_tile_group,
            reduce_comp_tile_group,
            reduce_result_tile_group,
            temp_tile_group,
        ) = _init_vec_tile_groups(io_d_type, tiling.group_size, tiling.d_deal_size)

    # ════════════════════════════════════════════════════════════
    #  Cube tile declarations (once)
    # ════════════════════════════════════════════════════════════
    with pl.section_cube():
        (cube_core_idx, group_idx, intra_group_idx, l1_w_db, cv_l0a, cv_l0b, cv_l0c) = (
            _init_cube_tile_groups(io_d_type, tiling.group_size)
        )
        # wkv/wgate 视图在 _phase2_matmul 内创建（tensor 下沉）

    #  无压缩块（totalValid=0 / compressedCnt=0）特判：放所有 section 外部，
    #  主函数支持多个 return——vec 侧四输出刷 0 后返回，cube 侧无代码直接返回，
    #  均不进 round 循环（d_ape/d_wkv/d_wgate 归约在 validRows=0 时虽隐式写 0，
    #  d_x 的 _reduce_dx 用回绕节点不可依赖——统一在此显式处理）
    # ═════════════════════════════════════════════════════════════════════════
    if compressed_cnt == 0:
        with pl.section_vector():
            _zero_outputs(
                core_idx,
                vec_ctx.core_num,
                tiling.x_rows,
                tiling.hidden_size,
                tiling.total_head_dim,
                tiling.cmp_size,
                d_x,
                d_wkv,
                d_wgate,
                d_ape,
                io_d_type,
                reduce_acc,
                p3_cast,
            )
        return
    # ════════════════════════════════════════════════════════════
    #  Phase 1+3: Vec Scatter & Reduce (per-round compute)
    # ════════════════════════════════════════════════════════════
    """Round 循环：Phase1+3（vec）与 Phase2（cube）交替，节点规划/快照/跨核同步在主循环"""
    prev_db_idx, prev_b_idx_start, prev_sc_idx_start = 0, 0, 0
    prev_b_idx_end, prev_sc_idx_end = 0, 0
    cur_db_idx, cur_b_idx_start, cur_sc_idx_start = 0, 0, 0
    cur_b_idx_end, cur_sc_idx_end = 0, 0
    for round_idx in pl.range(0, round_cnt + 1):
        round_blocks = min(
            tiling.total_sc_num_per_round,
            compressed_cnt - round_idx * tiling.total_sc_num_per_round,
        )
        db_idx = round_idx % 2

        with pl.section_vector():
            if round_idx < round_cnt:
                pre_deal_tc_size = group_idx * tiling.group_deal_sc_num
                if Coff == 1 and core_idx % 2 == 1:
                    pre_deal_tc_size += tiling.deal_sc_num
                deal_tc_size = min(
                    tiling.deal_sc_num, max(0, round_blocks - pre_deal_tc_size)
                )
                blocks_after = max(0, round_blocks - pre_deal_tc_size - deal_tc_size)

                # ── 块遍历节点规划（所有核：scatter/x 用 before/after，dX reduce 用 start/end）──
                b_idx_start, sc_idx_start = b_idx, sc_idx
                b_idx_before, sc_idx_before = _skip_blocks(
                    pre_deal_tc_size,
                    b_idx_start,
                    sc_idx_start,
                    start_pos,
                    seq_used,
                    cu_seqlens,
                    seq_ctx,
                )
                b_idx_after, sc_idx_after = _skip_blocks(
                    deal_tc_size,
                    b_idx_before,
                    sc_idx_before,
                    start_pos,
                    seq_used,
                    cu_seqlens,
                    seq_ctx,
                )
                b_idx_end, sc_idx_end = _skip_blocks(
                    blocks_after,
                    b_idx_after,
                    sc_idx_after,
                    start_pos,
                    seq_used,
                    cu_seqlens,
                    seq_ctx,
                )

                if pre_deal_tc_size < round_blocks:
                    _phase1_scatter(
                        vec_ctx,
                        round_idx,
                        pre_deal_tc_size,
                        deal_tc_size,
                        b_idx_before,
                        sc_idx_before,
                        core_idx,
                        sub_core_idx,
                        group_idx,
                        n_start,
                        db_idx,
                        tiling.cmp_ratio,
                        tiling.cmp_row_cnt,
                        tiling.d_deal_size,
                        tiling.coff_coef,
                        tiling.db_row_cnt,
                        tiling.deal_sc_num,
                        tiling.total_sc_num_per_round,
                        tile_dc_f16,
                        tile_dc_f32,
                        tile_kv,
                        tile_softmax,
                        tile_temp,
                        d_cmp_kv,
                        kv,
                        softmax_score,
                        x,
                        io_d_type,
                        tiling.cmp_kv_rows,
                        tiling.head_dim,
                        tiling.cmp_size,
                        tiling.token_size,
                        work_space_ptr,
                        tiling.group_num,
                        intra_group_idx,
                        tile_ape_local,
                        tile_dkv_cast,
                        tile_dkv_cast_md,
                        tile_dkv_nz,
                        tile_dkv_nz_md_prev,
                        tile_dkv_nz_md_cur,
                        tile_dsb_cast,
                        tile_dsb_cast_md,
                        tile_dsb_nz,
                        tile_dsb_nz_md_prev,
                        tile_dsb_nz_md_cur,
                        temp_tile_group,
                        l1_kv0_tile_group_nz,
                        l1_kv1_tile_group_nz,
                        l1_score0_tile_group_nz,
                        l1_score1_tile_group_nz,
                        move_x_tile_group,
                        x_arrange_gm,
                        start_pos,
                        seq_used,
                        cu_seqlens,
                        seq_ctx,
                    )

                # ── CV 通路: 本轮 v1 完成即发信号（供 cube c1(i) 消费）──
                b_idx, sc_idx = b_idx_end, sc_idx_end
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

                # ── 本轮快照存入 cur（V2(i-1) 结束后才轮换为 prev，避免被本轮 V2 读到）──
                cur_db_idx, cur_b_idx_start, cur_sc_idx_start = (
                    db_idx,
                    b_idx_start,
                    sc_idx_start,
                )
                cur_b_idx_end, cur_sc_idx_end = b_idx_end, sc_idx_end
                if round_idx == 0:
                    prev_db_idx, prev_b_idx_start, prev_sc_idx_start = (
                        cur_db_idx,
                        cur_b_idx_start,
                        cur_sc_idx_start,
                    )
                    prev_b_idx_end, prev_sc_idx_end = cur_b_idx_end, cur_sc_idx_end

            if round_idx > 0:
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

                _phase3_reduce(
                    vec_ctx,
                    core_idx,
                    group_idx,
                    round_idx,
                    round_cnt,
                    max_round_blocks,
                    prev_db_idx,
                    prev_b_idx_start,
                    prev_sc_idx_start,
                    prev_b_idx_end,
                    prev_sc_idx_end,
                    reduce_acc,
                    reduce_comp,
                    reduce_load_tile_group,
                    p3_cast,
                    d_x_cache_gm,
                    d_x_result_gm,
                    d_x,
                    d_wkv,
                    d_wgate,
                    d_ape,
                    io_d_type,
                    tiling.x_rows,
                    tiling.total_head_dim,
                    tiling.hidden_size,
                    tiling.cmp_size,
                    d_weight_work_space_size,
                    work_space_ptr,
                    d_w_kv_work_space_ptr,
                    d_w_gate_work_space_ptr,
                    start_pos,
                    seq_used,
                    cu_seqlens,
                    seq_ctx,
                )

                # ── V2 结束后轮换：本轮快照 cur → prev（供下一轮 V2(i) 使用）──
                (
                    prev_db_idx,
                    prev_b_idx_start,
                    prev_sc_idx_start,
                    prev_b_idx_end,
                    prev_sc_idx_end,
                ) = (
                    cur_db_idx,
                    cur_b_idx_start,
                    cur_sc_idx_start,
                    cur_b_idx_end,
                    cur_sc_idx_end,
                )

        # ════════════════════════════════════════════════════════════
        #  Phase 2: Cube Matmul — dkv@wkv + dsb@wgate → d_x_partial
        # ════════════════════════════════════════════════════════════
        with pl.section_cube():
            if round_idx < round_cnt:
                pre_deal_tc_size = group_idx * tiling.group_deal_sc_num
                deal_tc_size = min(
                    tiling.group_deal_sc_num, max(0, round_blocks - pre_deal_tc_size)
                )
                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
                if pre_deal_tc_size < round_blocks:
                    _phase2_matmul(
                        round_idx,
                        db_idx,
                        deal_tc_size,
                        tiling.hidden_size,
                        tiling.head_dim,
                        tiling.deal_sc_num,
                        tiling.cube_m_base_size,
                        tiling.cmp_ratio,
                        tiling.cube_core_num,
                        tiling.group_size,
                        tiling.group_num,
                        tiling.total_head_dim,
                        tiling.db_row_cnt,
                        group_row_stride,
                        intra_group_idx,
                        cv_l0a,
                        cv_l0b,
                        cv_l0c,
                        l1_w_db,
                        l1_kv0_tile_group_nz,
                        l1_kv1_tile_group_nz,
                        l1_score0_tile_group_nz,
                        l1_score1_tile_group_nz,
                        l1_kv0_tile_group_zn,
                        l1_kv1_tile_group_zn,
                        l1_score0_tile_group_zn,
                        l1_score1_tile_group_zn,
                        wkv,
                        wgate,
                        io_d_type,
                        d_x_result_gm,
                        x_arrange_gm,
                        d_wkv_result_gm,
                        d_w_gate_result_gm,
                    )

                pl.system.sync_all(core_type=pl.SyncCoreType.MIX)
