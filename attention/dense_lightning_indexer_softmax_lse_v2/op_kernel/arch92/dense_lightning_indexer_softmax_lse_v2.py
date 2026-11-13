# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DenseLightningIndexLseV2 kernel.

Computes softmax LSE for dense lightning indexer.
A6(ascend960dt)适配版: 1:1架构, 单Vec核, 无sub_id, qk tile为[TN, TKV]整块.
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField

# ================================================================
# Tile dimensions and constants
# ================================================================
TS = 1
TKV = 128
TD = 128
TN = 128
TILINGDATA_N = 128

Q_F16 = TN * TD * 2
KT_F16 = TKV * TD * 2
QK_F32 = TKV * TS * 4

MA0 = 0
MA1 = Q_F16 * 2
LA0 = 0
RA0 = 0
CA0 = 0


def _align_up(value, align=1024):
    return ((value + align - 1) // align) * align


VB4_KV = TN * TKV * 4

VA0 = 0
VA1 = _align_up(VA0 + VB4_KV * 2, 32)
VA2 = _align_up(VA1 + TN * 4 * 2, 32)
VA3 = _align_up(VA2 + 1, 32)

# max_vec is [1, 64] fp32 = 256 B. The old "+1" only reserved 32 B, so
# reduce_sum_vec elements 0..55 aliased max slots 8..63 (tile-dump proven).
VA4 = _align_up(VA3 + 256, 32)
KV_CACHE_NUM = _align_up(248 * 1024 - VA4, 32) // 4

MAX_UB_S2_TILES = KV_CACHE_NUM // TKV

QK_READY_IDS = (0, 1)
QK_FREE_IDS = (2, 3)

WS_MAX_ELEMS = 72
WS_MAX_BYTES = 288

# Size-aware scheduling table offsets in the metadata buffer (int32 elements).
# Keep in sync with attention/.../op_kernel/dense_lightning_indexer_softmax_lse_v2_metadata.h
SCHED_FLAG_IDX = 4
SCHED_CNT_IDX = 8
SCHED_OFF_IDX = 72
SCHED_ITEM_IDX = 136
SCHED_ENTRY_STRIDE = 8
SCHED_ITEM_MAX_NUM = 65536
SCHED_METADATA_MAX_SIZE = SCHED_ITEM_IDX + SCHED_ENTRY_STRIDE * SCHED_ITEM_MAX_NUM


# ================================================================
# Tiling data
# ================================================================
@dataclass
class DenseLILseV2Tiling:
    b: int
    s1: int
    s2: int
    n1: int
    d: int
    cmp_ratio: int
    block_num: int


# ================================================================
# TilingKey
# ================================================================
class DenseLiLseTilingKey:
    is_long_s2 = TilingKeyField(bits=1, values=[0, 1])
    has_seq_used_q = TilingKeyField(bits=1, values=[0, 1])
    has_seq_used_k = TilingKeyField(bits=1, values=[0, 1])
    mask_mode = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    layout = TilingKeyField(bits=1, values=[0, 1])


# ================================================================
# VF helper functions
# ================================================================
@pl.vector_function
def init_data(max_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vreg_x = vf.full(-3.4028234663852886e38, dtype=pl.DT_FP32)
    vf.store_align(max_tile, vreg_x, preg_all)


@pl.vector_function
def write_scalar_vf(tile, val):
    preg_one = vf.update_mask(1, dtype=pl.DT_FP32)
    vreg_val = vf.full(val, dtype=pl.DT_FP32)
    vf.store_align(tile, vreg_val, preg_one)


@pl.vector_function
def write_neg_inf_vf(tile):
    preg_one = vf.update_mask(1, dtype=pl.DT_FP32)
    vreg_zero = vf.full(0, dtype=pl.DT_FP32)
    vreg_log = vf.log(vreg_zero, preg_one)
    vf.store_align(tile, vreg_log, preg_one)


@pl.vector_function
def process_reduce_max_vf(
    dst_tile, qk_tile, weight_tile, max_tile, n1_dim, s2_size, row_heads, max_off
):
    # lo/hi split retained (A6 64-lane VF); unroll-4 + tree for ILP
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    lo_size = pl.min(s2_size, 64)
    hi_size = pl.max(s2_size - 64, 0)
    preg_lo = vf.update_mask(lo_size, dtype=pl.DT_FP32)
    preg_hi = vf.update_mask(hi_size, dtype=pl.DT_FP32)

    vreg_sum_lo = vf.full(0, dtype=pl.DT_FP32)
    vreg_sum_hi = vf.full(0, dtype=pl.DT_FP32)

    unroll = 4
    n1_main = n1_dim // unroll * unroll

    for m in pl.range(0, n1_main, unroll):
        vreg_w0 = vf.load_align(
            weight_tile, row_heads + m + 0, dist=pl.LoadDist.BRC_B32
        )
        vreg_x0 = vf.load_align(qk_tile, (row_heads + m + 0) * TKV)
        vreg_relu0 = vf.relu(vreg_x0, preg_lo)
        vreg_mul0 = vf.mul(vreg_relu0, vreg_w0, preg_lo)
        vreg_x0h = vf.load_align(qk_tile, (row_heads + m + 0) * TKV + 64)
        vreg_relu0h = vf.relu(vreg_x0h, preg_hi)
        vreg_mul0h = vf.mul(vreg_relu0h, vreg_w0, preg_hi)
        vreg_w1 = vf.load_align(
            weight_tile, row_heads + m + 1, dist=pl.LoadDist.BRC_B32
        )
        vreg_x1 = vf.load_align(qk_tile, (row_heads + m + 1) * TKV)
        vreg_relu1 = vf.relu(vreg_x1, preg_lo)
        vreg_mul1 = vf.mul(vreg_relu1, vreg_w1, preg_lo)
        vreg_x1h = vf.load_align(qk_tile, (row_heads + m + 1) * TKV + 64)
        vreg_relu1h = vf.relu(vreg_x1h, preg_hi)
        vreg_mul1h = vf.mul(vreg_relu1h, vreg_w1, preg_hi)
        vreg_w2 = vf.load_align(
            weight_tile, row_heads + m + 2, dist=pl.LoadDist.BRC_B32
        )
        vreg_x2 = vf.load_align(qk_tile, (row_heads + m + 2) * TKV)
        vreg_relu2 = vf.relu(vreg_x2, preg_lo)
        vreg_mul2 = vf.mul(vreg_relu2, vreg_w2, preg_lo)
        vreg_x2h = vf.load_align(qk_tile, (row_heads + m + 2) * TKV + 64)
        vreg_relu2h = vf.relu(vreg_x2h, preg_hi)
        vreg_mul2h = vf.mul(vreg_relu2h, vreg_w2, preg_hi)
        vreg_w3 = vf.load_align(
            weight_tile, row_heads + m + 3, dist=pl.LoadDist.BRC_B32
        )
        vreg_x3 = vf.load_align(qk_tile, (row_heads + m + 3) * TKV)
        vreg_relu3 = vf.relu(vreg_x3, preg_lo)
        vreg_mul3 = vf.mul(vreg_relu3, vreg_w3, preg_lo)
        vreg_x3h = vf.load_align(qk_tile, (row_heads + m + 3) * TKV + 64)
        vreg_relu3h = vf.relu(vreg_x3h, preg_hi)
        vreg_mul3h = vf.mul(vreg_relu3h, vreg_w3, preg_hi)

        vreg_l03 = vf.add(vreg_mul0, vreg_mul3, preg_lo)
        vreg_l12 = vf.add(vreg_mul1, vreg_mul2, preg_lo)
        vreg_ltmp = vf.add(vreg_l03, vreg_l12, preg_lo)
        vreg_sum_lo = vf.add(vreg_sum_lo, vreg_ltmp, preg_lo)

        vreg_h03 = vf.add(vreg_mul0h, vreg_mul3h, preg_hi)
        vreg_h12 = vf.add(vreg_mul1h, vreg_mul2h, preg_hi)
        vreg_htmp = vf.add(vreg_h03, vreg_h12, preg_hi)
        vreg_sum_hi = vf.add(vreg_sum_hi, vreg_htmp, preg_hi)

    for m in pl.range(n1_main, n1_dim):
        vreg_w = vf.load_align(weight_tile, row_heads + m, dist=pl.LoadDist.BRC_B32)
        vreg_x = vf.load_align(qk_tile, (row_heads + m) * TKV)
        vreg_relu = vf.relu(vreg_x, preg_lo)
        vreg_mul = vf.mul(vreg_relu, vreg_w, preg_lo)
        vreg_sum_lo = vf.add(vreg_sum_lo, vreg_mul, preg_lo)
        vreg_xh = vf.load_align(qk_tile, (row_heads + m) * TKV + 64)
        vreg_reluh = vf.relu(vreg_xh, preg_hi)
        vreg_mulh = vf.mul(vreg_reluh, vreg_w, preg_hi)
        vreg_sum_hi = vf.add(vreg_sum_hi, vreg_mulh, preg_hi)

    preg_max = vf.update_mask(1, dtype=pl.DT_FP32)
    vreg_reduce_max = vf.reduce_max(
        vreg_sum_lo, preg_lo, merge_mode=pl.MergeMode.ZEROING
    )
    vreg_tmp_max = vf.load_align(max_tile, max_off)
    vreg_max = vf.max(vreg_tmp_max, vreg_reduce_max, preg_max)
    if hi_size > 0:
        vreg_reduce_max_hi = vf.reduce_max(
            vreg_sum_hi, preg_hi, merge_mode=pl.MergeMode.ZEROING
        )
        vreg_max = vf.max(vreg_max, vreg_reduce_max_hi, preg_max)
    vf.store_align(max_tile, vreg_max, preg_max, max_off)
    vf.store_align(dst_tile, vreg_sum_lo, preg_all, 0)
    vf.store_align(dst_tile, vreg_sum_hi, preg_all, 64)


@pl.vector_function
def process_reduce_sum_vf(dst_tile, reduce_sum_vec_tile, tmp_max, s2_tiles, tail_size):
    # split semantics retained; unroll-4 + tree
    vreg_max = vf.full(tmp_max, dtype=pl.DT_FP32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    lo_size = pl.min(tail_size, 64)
    hi_size = pl.max(tail_size - 64, 0)
    preg_lo = vf.update_mask(lo_size, dtype=pl.DT_FP32)
    preg_hi = vf.update_mask(hi_size, dtype=pl.DT_FP32)

    vreg_acc_full = vf.full(0, dtype=pl.DT_FP32)

    unroll = 4
    main_tiles = (s2_tiles - 1) // unroll * unroll

    for m in pl.range(0, main_tiles, unroll):
        vreg_x0l = vf.load_align(reduce_sum_vec_tile, (m + 0) * TKV)
        vreg_e0l = vf.exp_sub(vreg_x0l, vreg_max, preg_all)
        vreg_x0h = vf.load_align(reduce_sum_vec_tile, (m + 0) * TKV + 64)
        vreg_e0h = vf.exp_sub(vreg_x0h, vreg_max, preg_all)
        vreg_x1l = vf.load_align(reduce_sum_vec_tile, (m + 1) * TKV)
        vreg_e1l = vf.exp_sub(vreg_x1l, vreg_max, preg_all)
        vreg_x1h = vf.load_align(reduce_sum_vec_tile, (m + 1) * TKV + 64)
        vreg_e1h = vf.exp_sub(vreg_x1h, vreg_max, preg_all)
        vreg_x2l = vf.load_align(reduce_sum_vec_tile, (m + 2) * TKV)
        vreg_e2l = vf.exp_sub(vreg_x2l, vreg_max, preg_all)
        vreg_x2h = vf.load_align(reduce_sum_vec_tile, (m + 2) * TKV + 64)
        vreg_e2h = vf.exp_sub(vreg_x2h, vreg_max, preg_all)
        vreg_x3l = vf.load_align(reduce_sum_vec_tile, (m + 3) * TKV)
        vreg_e3l = vf.exp_sub(vreg_x3l, vreg_max, preg_all)
        vreg_x3h = vf.load_align(reduce_sum_vec_tile, (m + 3) * TKV + 64)
        vreg_e3h = vf.exp_sub(vreg_x3h, vreg_max, preg_all)

        vreg_t01 = vf.add(vreg_e0l, vreg_e1l, preg_all)
        vreg_t23 = vf.add(vreg_e2l, vreg_e3l, preg_all)
        vreg_c_lo = vf.add(vreg_t01, vreg_t23, preg_all)
        vreg_u01 = vf.add(vreg_e0h, vreg_e1h, preg_all)
        vreg_u23 = vf.add(vreg_e2h, vreg_e3h, preg_all)
        vreg_c_hi = vf.add(vreg_u01, vreg_u23, preg_all)

        vreg_acc_full = vf.add(vreg_acc_full, vreg_c_lo, preg_all)
        vreg_acc_full = vf.add(vreg_acc_full, vreg_c_hi, preg_all)

    for m in pl.range(main_tiles, s2_tiles - 1):
        vreg_xl = vf.load_align(reduce_sum_vec_tile, m * TKV)
        vreg_el = vf.exp_sub(vreg_xl, vreg_max, preg_all)
        vreg_acc_full = vf.add(vreg_acc_full, vreg_el, preg_all)
        vreg_xh = vf.load_align(reduce_sum_vec_tile, m * TKV + 64)
        vreg_eh = vf.exp_sub(vreg_xh, vreg_max, preg_all)
        vreg_acc_full = vf.add(vreg_acc_full, vreg_eh, preg_all)

    vreg_acc_last_lo = vf.full(0, dtype=pl.DT_FP32)
    vreg_xl = vf.load_align(reduce_sum_vec_tile, (s2_tiles - 1) * TKV)
    vreg_el = vf.exp_sub(vreg_xl, vreg_max, preg_lo)
    vreg_acc_last_lo = vf.add(vreg_acc_last_lo, vreg_el, preg_lo)

    vreg_acc_last_hi = vf.full(0, dtype=pl.DT_FP32)
    if hi_size > 0:
        vreg_xh = vf.load_align(reduce_sum_vec_tile, (s2_tiles - 1) * TKV + 64)
        vreg_eh = vf.exp_sub(vreg_xh, vreg_max, preg_hi)
        vreg_acc_last_hi = vf.add(vreg_acc_last_hi, vreg_eh, preg_hi)

    vreg_total = vf.add(vreg_acc_full, vreg_acc_last_lo, preg_all)
    vreg_total = vf.add(vreg_total, vreg_acc_last_hi, preg_all)

    red_count = pl.max(pl.max(lo_size, hi_size), pl.min((s2_tiles - 1) * 64, 64))
    preg_red = vf.update_mask(red_count, dtype=pl.DT_FP32)
    vreg_reduce_sum = vf.reduce_sum(
        vreg_total, preg_red, merge_mode=pl.MergeMode.ZEROING
    )
    preg_sum = vf.update_mask(1, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, vreg_reduce_sum, preg_sum)


@pl.vector_function
def process_lse_vf(lse_vec_tile, reduce_res, tmp_s2_max):
    vreg_max = vf.full(tmp_s2_max, dtype=pl.DT_FP32)
    vreg_sum = vf.full(reduce_res, dtype=pl.DT_FP32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_lse = vf.update_mask(1, dtype=pl.DT_FP32)
    vreg_log = vf.log(vreg_sum, preg_all)
    vreg_lse = vf.add(vreg_log, vreg_max, preg_all)
    vf.store_align(lse_vec_tile, vreg_lse, preg_lse)


def process_long_s2_reduce_max(
    qk_vec,
    weight_vec,
    reduce_sum_vec_tile,
    workspace,
    max_tile,
    n1_dim,
    core_id,
    prev_s2_tiles,
    prev_s2_vec_tail_size,
    j,
    work_count,
    task_id,
):
    """is_long_s2==1 分支1: 分chunk做 reduce_max + store到workspace。返回更新后的 task_id。"""
    num_chunks = (prev_s2_tiles + MAX_UB_S2_TILES - 1) // MAX_UB_S2_TILES
    for chunk_idx in pl.range(0, num_chunks):
        chunk_start = chunk_idx * MAX_UB_S2_TILES
        chunk_end = chunk_start + MAX_UB_S2_TILES
        if chunk_end > prev_s2_tiles:
            chunk_end = prev_s2_tiles
        chunk_tiles = chunk_end - chunk_start
        for ci in pl.range(0, chunk_tiles):
            qk_vec_tile = qk_vec.next()
            weight_tile_cur = weight_vec.previous()
            pl.system.wait_cross_core(
                pipe=pl.PipeType.V, event_id=QK_READY_IDS[task_id % 2]
            )
            src_tile = reduce_sum_vec_tile[0:1, ci * TKV :]
            cur_vec_size = (
                prev_s2_vec_tail_size
                if (chunk_start + ci) == prev_s2_tiles - 1
                else TKV
            )
            if cur_vec_size > 0:
                pl.set_validshape(src_tile, [1, cur_vec_size])
                process_reduce_max_vf(
                    src_tile,
                    qk_vec_tile,
                    weight_tile_cur,
                    max_tile,
                    n1_dim,
                    cur_vec_size,
                    0,
                    0,
                )
            pl.system.set_cross_core(
                pipe=pl.PipeType.V, event_id=QK_FREE_IDS[task_id % 2]
            )
            task_id = task_id + 1
        for ci in pl.range(0, chunk_tiles):
            ws_tile = reduce_sum_vec_tile[0:1, ci * TKV :]
            cur_vec_size = (
                prev_s2_vec_tail_size
                if (chunk_start + ci) == prev_s2_tiles - 1
                else TKV
            )
            if cur_vec_size > 0:
                pl.set_validshape(ws_tile, [1, cur_vec_size])
                pl.store(
                    workspace,
                    ws_tile,
                    [core_id, (chunk_start + ci) * TKV],
                )
    return task_id


def process_long_s2_reduce_sum(
    lse_vec_tile,
    reduce_sum_vec,
    workspace,
    tmp_s2_max,
    core_id,
    prev_s2_tiles,
    prev_s2_vec_tail_size,
):
    """is_long_s2==1 分支2: 从workspace分chunk load + reduce_sum 累加,写回 lse_vec_tile。"""
    ws_chunk_sum = 0.0
    num_chunks = (prev_s2_tiles + MAX_UB_S2_TILES - 1) // MAX_UB_S2_TILES
    for chunk_idx in pl.range(0, num_chunks):
        chunk_start = chunk_idx * MAX_UB_S2_TILES
        chunk_end = chunk_start + MAX_UB_S2_TILES
        if chunk_end > prev_s2_tiles:
            chunk_end = prev_s2_tiles
        chunk_tiles = chunk_end - chunk_start
        chunk_tile = reduce_sum_vec.next()
        for ci in pl.range(0, chunk_tiles):
            src = chunk_tile[0:1, ci * TKV :]
            pl.set_validshape(src, [1, TKV])
            pl.load(src, workspace, [core_id, (chunk_start + ci) * TKV])
        chunk_tail_sum = prev_s2_vec_tail_size if chunk_idx == num_chunks - 1 else TKV
        process_reduce_sum_vf(
            lse_vec_tile, chunk_tile, tmp_s2_max, chunk_tiles, chunk_tail_sum
        )
        ws_chunk_sum = ws_chunk_sum + pl.getval(lse_vec_tile, 0)
    write_scalar_vf(lse_vec_tile, ws_chunk_sum)


# ================================================================
# Inner kernel — receives typed tensor views + workspace
# ================================================================
def resolve_b_s1(
    work_id,
    layout,
    b_dim,
    s1_dim,
    s2_dim,
    cu_seq_lens_q,
    cu_seq_lens_k,
    has_seq_used_q,
    seqused_q,
):
    b_idx = 0
    s1_i = 0
    q_offset = 0
    k_offset_base = 0
    if has_seq_used_q == 1:
        acc = 0
        for b in pl.range(b_dim):
            cur_used = pl.getval(seqused_q, b)
            if work_id >= acc + cur_used:
                acc = acc + cur_used
                b_idx = b + 1
            else:
                b_idx = b
                break
        s1_i = work_id - acc
        if layout == 1:
            q_offset = pl.getval(cu_seq_lens_q, b_idx) + s1_i
            k_offset_base = pl.getval(cu_seq_lens_k, b_idx)
        else:
            q_offset = b_idx * s1_dim + s1_i
            k_offset_base = b_idx * s2_dim
    elif layout == 1:
        for b in pl.range(b_dim):
            cu_val = pl.getval(cu_seq_lens_q, b + 1)
            if work_id >= cu_val:
                b_idx = b + 1
        s1_i = work_id - pl.getval(cu_seq_lens_q, b_idx)
        q_offset = work_id
        k_offset_base = pl.getval(cu_seq_lens_k, b_idx)
    else:
        b_idx = work_id // s1_dim
        s1_i = work_id % s1_dim
        q_offset = b_idx * s1_dim + s1_i
        k_offset_base = b_idx * s2_dim
    return b_idx, s1_i, q_offset, k_offset_base


def dense_lightning_indexer_softmax_lse_v2_inner(
    query_index,
    query_index_flat,
    key_index,
    weight,
    lse_out,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    cu_seq_lens_q,
    cu_seq_lens_k,
    workspace_max,
    workspace_sum,
    workspace,
    tiling,
    metadata,
):
    s1_dim = tiling.s1
    s2_dim = tiling.s2
    n1_dim = tiling.n1
    b_dim = tiling.b
    cmp_ratio = tiling.cmp_ratio

    core_id = pl.get_block_idx()

    fore_core_num = pl.getval(metadata, 0)
    # pl.printf("--dedebug, get fore_core_num: %d---\n", fore_core_num)
    tail_core_num = pl.getval(metadata, 1)
    b_s1_per_core = pl.getval(metadata, 2)
    b_s1_per_tail_core = pl.getval(metadata, 3)
    total_cores = fore_core_num + tail_core_num
    # size-aware scheduling table (arch92): metadata[4] = 1 means per-core item
    # table at [SCHED_ITEM_IDX ..] is valid, balance done by the aicpu side (LPT)
    use_sched_table = pl.getval(metadata, SCHED_FLAG_IDX)
    sched_off = 0
    work_count = 0

    if use_sched_table == 1:
        sched_off = pl.getval(metadata, SCHED_OFF_IDX + core_id)
        work_count = 0 + pl.getval(metadata, SCHED_CNT_IDX + core_id)
    elif core_id < fore_core_num:
        work_count = 0 + b_s1_per_core
    elif core_id < total_cores:
        work_count = 0 + b_s1_per_tail_core

    # 【128，128】基本块用于qk的结果，这里UB开两块做db
    qk_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[TN, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=VA0,
        mutex_ids=[0, 1],
    )
    weight_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, TN], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=VA1,
        mutex_ids=[2, 3],
    )
    lse_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=VA2,
        mutex_ids=[4],
    )
    max_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=VA3,
        mutex_ids=[5],
    )
    reduce_sum_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, KV_CACHE_NUM], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=VA4,
        mutex_ids=[6],
    )

    with pl.section_cube():
        q_mat = pl.make_tile_group(
            type=pl.TileType(
                shape=[TN, TD],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                compact=1,
            ),
            addrs=MA0,
            mutex_ids=[7, 8],
        )
        k_mat = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TKV],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                compact=1,
            ),
            addrs=MA1,
            mutex_ids=[9, 10],
        )
        left = pl.make_tile_group(
            type=pl.TileType(
                shape=[TN, TD],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
                compact=1,
            ),
            addrs=LA0,
            mutex_ids=[11, 12],
        )
        right = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TKV],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
                compact=1,
            ),
            addrs=RA0,
            mutex_ids=[13, 14],
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[TN, TKV],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                compact=1,
            ),
            addrs=CA0,
            mutex_ids=[15, 16],
        )

        task_id = 0
        rows = 1
        m_eff = 128
        for j in pl.range(0, work_count):
            b_idx = 0
            s1_i = 0
            q_offset = 0
            k_offset_base = 0
            if use_sched_table == 1:
                # 8-int entry (arch92 row-block aicpu builder):
                # (q_offset, k_base, rows, vk_max, vk_param, q_head_offset, 0, 0)
                item_idx = SCHED_ITEM_IDX + SCHED_ENTRY_STRIDE * (sched_off + j)
                q_offset = pl.getval(metadata, item_idx)
                k_offset_base = pl.getval(metadata, item_idx + 1)
                rows = 0 + pl.getval(metadata, item_idx + 2)
                valid_k_size = 0 + pl.getval(metadata, item_idx + 3)
                is_s1_valid = valid_k_size > 0
            else:
                work_id = core_id + j * total_cores
                b_idx, s1_i, q_offset, k_offset_base = resolve_b_s1(
                    work_id,
                    layout,
                    b_dim,
                    s1_dim,
                    s2_dim,
                    cu_seq_lens_q,
                    cu_seq_lens_k,
                    has_seq_used_q,
                    seqused_q,
                )
                if has_seq_used_q == 1:
                    q_used_size = pl.getval(seqused_q, b_idx)
                elif layout == 1:
                    q_used_size = pl.getval(cu_seq_lens_q, b_idx + 1) - pl.getval(
                        cu_seq_lens_q, b_idx
                    )
                else:
                    q_used_size = s1_dim
                if has_seq_used_k == 1:
                    k_used_size = pl.getval(seqused_k, b_idx)
                elif layout == 1:
                    k_used_size = pl.getval(cu_seq_lens_k, b_idx + 1) - pl.getval(
                        cu_seq_lens_k, b_idx
                    )
                else:
                    k_used_size = s2_dim
                valid_s1_start = 0
                valid_k_size = 0 + k_used_size
                if mask_mode == 3:
                    cur_residual_k = 0
                    if cmp_ratio > 1:
                        cur_residual_k = 0 + pl.getval(cmp_residual_k, b_idx)
                    ori_k_size = k_used_size * cmp_ratio + cur_residual_k
                    qk_residual = q_used_size - ori_k_size
                    valid_s1_start = pl.max(qk_residual, 0)
                    valid_k_size = pl.max((s1_i + 1 - qk_residual) // cmp_ratio, 0)
                is_s1_valid = s1_i >= valid_s1_start and valid_k_size > 0
            if is_s1_valid:
                s2_tiles = (valid_k_size + TKV - 1) // TKV
                s2_tail_size = TKV + valid_k_size - s2_tiles * TKV
                # A6 HW: fixpipe/mm M,N must be multiples of 16 (C0 granularity).
                # Align cube-side tail shapes up. GM loads keep actual sizes;
                # the vector side masks reads to actual sizes.
                s2_tail_align = ((s2_tail_size + 15) // 16) * 16
                n1_align = ((n1_dim + 15) // 16) * 16
                # sched row-block: one fused load covers up to r_load rows
                # (aicpu builder caps rows at 128 // n1). M is loop-invariant
                # so the MTE2 descriptor keeps the static preamble-spill form;
                # unused rows are never consumed by the vector side.
                r_load = 128 // n1_dim
                m_load = r_load * n1_dim
                m_ub = ((m_load + 15) // 16) * 16

                q_tile = q_mat.next()
                if use_sched_table == 1:
                    q_head_offset = q_offset * n1_dim
                    # load only the entry's actual rows: a static r_load-row
                    # load over-reads q by (r_load - rows) rows and page-faults
                    # when the last block ends exactly at the tensor end
                    m_need = rows * n1_dim
                    m_eff = ((m_need + 15) // 16) * 16
                    pl.set_validshape(q_tile, [m_need, TD])
                    pl.load(q_tile, query_index_flat, [q_head_offset, 0])
                else:
                    pl.set_validshape(q_tile, [n1_dim, TD])
                    pl.load(q_tile, query_index, [q_offset, 0, 0, 0])
                for s2_i in pl.range(0, s2_tiles + 1):
                    k_tile_nxt = k_mat.next()  # 0-a 1-b 2-a
                    if s2_i < s2_tiles:
                        if s2_i == s2_tiles - 1 and s2_tail_size != TKV:
                            pl.set_validshape(k_tile_nxt, [TD, s2_tail_size])
                        else:
                            pl.set_validshape(k_tile_nxt, [TD, TKV])
                        pl.load(
                            k_tile_nxt,
                            key_index,
                            [k_offset_base + s2_i * TKV, 0, 0, 0],
                            order=[1, 0],
                        )
                    if s2_i > 0:
                        k_tile_cur = k_mat.previous()  # 1-a 2-b
                        left_tile = left.next()
                        right_tile = right.next()
                        acc_tile = acc.next()
                        if use_sched_table == 1:
                            pl.set_validshape(left_tile, [m_eff, TD])
                        else:
                            pl.set_validshape(left_tile, [n1_align, TD])
                        if s2_i == s2_tiles and s2_tail_size != TKV:
                            pl.set_validshape(right_tile, [TD, s2_tail_align])
                            pl.set_validshape(k_tile_cur, [TD, s2_tail_align])
                            if use_sched_table == 1:
                                pl.set_validshape(acc_tile, [m_eff, s2_tail_align])
                            else:
                                pl.set_validshape(acc_tile, [n1_align, s2_tail_align])
                        else:
                            pl.set_validshape(right_tile, [TD, TKV])
                            pl.set_validshape(k_tile_cur, [TD, TKV])
                            if use_sched_table == 1:
                                pl.set_validshape(acc_tile, [m_eff, TKV])
                            else:
                                pl.set_validshape(acc_tile, [n1_align, TKV])
                        pl.move(left_tile, q_tile)
                        pl.move(right_tile, k_tile_cur)  # 1-a
                        pl.matmul(acc_tile, left_tile, right_tile)

                        qk_vec_tile = qk_vec.next()
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX, event_id=QK_FREE_IDS[task_id % 2]
                        )
                        pl.move(
                            qk_vec_tile,
                            acc_tile,
                            acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0,  # A6为1:1架构, 不支持A5(1:2)的DualModeSplitN
                        )
                        pl.system.bar_fix()
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[task_id % 2]
                        )
                        task_id = task_id + 1

        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_FREE_IDS[0])
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_FREE_IDS[1])

    with pl.section_vector():
        task_id = 0
        prev_s2_tiles = 0
        prev_is_s1_valid = False
        prev_valid_k_size = 0
        prev_s2_tail_size = TKV
        prev_q_offset = 0
        prev_rows = 1
        prev_vk_param = 0
        s2_tiles = 0
        s2_tail_size = 0
        valid_k_size = 0
        rows = 1
        vk_param = 0
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_FREE_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_FREE_IDS[1])

        for j in pl.range(0, work_count + 1):
            b_idx = 0
            s1_i = 0
            q_offset = 0
            if use_sched_table != 1:
                work_id = core_id + j * total_cores
                b_idx, s1_i, q_offset, _ = resolve_b_s1(
                    work_id,
                    layout,
                    b_dim,
                    s1_dim,
                    s2_dim,
                    cu_seq_lens_q,
                    cu_seq_lens_k,
                    has_seq_used_q,
                    seqused_q,
                )

            max_tile = max_vec.next()
            init_data(max_tile)
            weight_vec_nxt = weight_vec.next()
            is_s1_valid = False
            if j < work_count:
                if use_sched_table == 1:
                    # 8-int table entry (see cube section): blocked item setup
                    # is a handful of scalar loads.
                    item_idx = SCHED_ITEM_IDX + SCHED_ENTRY_STRIDE * (sched_off + j)
                    q_offset = pl.getval(metadata, item_idx)
                    rows = 0 + pl.getval(metadata, item_idx + 2)
                    valid_k_size = 0 + pl.getval(metadata, item_idx + 3)
                    vk_param = 0 + pl.getval(metadata, item_idx + 4)
                    is_s1_valid = True
                else:
                    if has_seq_used_q == 1:
                        q_used_size = pl.getval(seqused_q, b_idx)
                    elif layout == 1:
                        q_used_size = pl.getval(cu_seq_lens_q, b_idx + 1) - pl.getval(
                            cu_seq_lens_q, b_idx
                        )
                    else:
                        q_used_size = s1_dim
                    if has_seq_used_k == 1:
                        k_used_size = pl.getval(seqused_k, b_idx)
                    elif layout == 1:
                        k_used_size = pl.getval(cu_seq_lens_k, b_idx + 1) - pl.getval(
                            cu_seq_lens_k, b_idx
                        )
                    else:
                        k_used_size = s2_dim
                    valid_s1_start = 0
                    valid_k_size = 0 + k_used_size
                    if mask_mode == 3:
                        cur_residual_k = 0
                        if cmp_ratio > 1:
                            cur_residual_k = 0 + pl.getval(cmp_residual_k, b_idx)
                        ori_k_size = k_used_size * cmp_ratio + cur_residual_k
                        qk_residual = q_used_size - ori_k_size
                        valid_s1_start = pl.max(qk_residual, 0)
                        valid_k_size = pl.max((s1_i + 1 - qk_residual) // cmp_ratio, 0)
                    is_s1_valid = s1_i >= valid_s1_start and valid_k_size > 0
                    rows = 1
                if is_s1_valid:
                    s2_tiles = (valid_k_size + TKV - 1) // TKV
                    s2_tail_size = TKV + valid_k_size - s2_tiles * TKV
                    if use_sched_table == 1:
                        # blocked: one flat load covers all R rows' weights.
                        # [T1, 1, n1] is row-major contiguous, so elements
                        # [q_offset*n1, (q_offset+r_load)*n1) are exactly the
                        # block's rows; consumption stays 4B-granular
                        # vf.load_align(weight_tile, r*n1 + m), the same form
                        # the strided path already uses.
                        m_load_v = rows * n1_dim
                        pl.set_validshape(weight_vec_nxt, [1, m_load_v])
                        pl.load(weight_vec_nxt, weight, [q_offset, 0, 0])
                    else:
                        pl.set_validshape(weight_vec_nxt, [1, n1_dim])
                        pl.load(weight_vec_nxt, weight, [q_offset, 0, 0])

            if j > 0 and prev_is_s1_valid:
                reduce_sum_vec_tile = reduce_sum_vec.next()
                lse_vec_tile = lse_vec.next()

                prev_s2_vec_tail_size = prev_s2_tail_size
                prev_row_stride = prev_s2_tiles * TKV

                if is_long_s2 == 0:
                    for s2_i in pl.range(0, prev_s2_tiles):
                        qk_vec_tile = qk_vec.next()
                        weight_tile_cur = weight_vec.previous()
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.V, event_id=QK_READY_IDS[task_id % 2]
                        )
                        for r in pl.range(0, prev_rows):
                            if use_sched_table == 1:
                                if mask_mode == 3:
                                    vk_r = pl.max((prev_vk_param + r) // cmp_ratio, 0)
                                else:
                                    vk_r = prev_vk_param
                            else:
                                vk_r = prev_valid_k_size
                            tiles_r = (vk_r + TKV - 1) // TKV
                            if s2_i < tiles_r:
                                tail_r = TKV + vk_r - tiles_r * TKV
                                cur_vec_size = tail_r if s2_i == tiles_r - 1 else TKV
                                if cur_vec_size > 0:
                                    src_tile = reduce_sum_vec_tile[
                                        0:1, r * prev_row_stride + s2_i * TKV :
                                    ]
                                    pl.set_validshape(src_tile, [1, cur_vec_size])
                                    process_reduce_max_vf(
                                        src_tile,
                                        qk_vec_tile,
                                        weight_tile_cur,
                                        max_tile,
                                        n1_dim,
                                        cur_vec_size,
                                        r * n1_dim,
                                        r * 8,
                                    )
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.V, event_id=QK_FREE_IDS[task_id % 2]
                        )
                        task_id = task_id + 1
                else:
                    task_id = process_long_s2_reduce_max(
                        qk_vec,
                        weight_vec,
                        reduce_sum_vec_tile,
                        workspace,
                        max_tile,
                        n1_dim,
                        core_id,
                        prev_s2_tiles,
                        prev_s2_vec_tail_size,
                        j,
                        work_count,
                        task_id,
                    )

                for r in pl.range(0, prev_rows):
                    if use_sched_table == 1:
                        if mask_mode == 3:
                            vk_r = pl.max((prev_vk_param + r) // cmp_ratio, 0)
                        else:
                            vk_r = prev_vk_param
                    else:
                        vk_r = prev_valid_k_size
                    tiles_r = (vk_r + TKV - 1) // TKV
                    if tiles_r > 0:
                        tail_r = TKV + vk_r - tiles_r * TKV
                        tmp_s2_max = pl.getval(max_tile, r * 8)
                        lse_slice = lse_vec_tile[0:1, 0:]

                        if is_long_s2 == 0:
                            region_r = reduce_sum_vec_tile[0:1, r * prev_row_stride :]

                            process_reduce_sum_vf(
                                lse_slice,
                                region_r,
                                tmp_s2_max,
                                tiles_r,
                                tail_r,
                            )
                        else:
                            process_long_s2_reduce_sum(
                                lse_slice,
                                reduce_sum_vec,
                                workspace,
                                tmp_s2_max,
                                core_id,
                                prev_s2_tiles,
                                prev_s2_vec_tail_size,
                            )

                        reduce_res = pl.getval(lse_vec_tile, 0)
                        # drain pipes between work items: without this the simulator's
                        # cross-function pipelining feeds stale/in-flight register
                        # lanes into the next item's reduce ops (traps on inf/nan)
                        pl.system.bar_all()

                        process_lse_vf(lse_slice, reduce_res, tmp_s2_max)
                        pl.set_validshape(lse_slice, [1, 1])
                        pl.store(lse_out, lse_slice, [prev_q_offset + r, 0])
                        pl.system.bar_all()

            prev_s2_tiles = s2_tiles
            prev_is_s1_valid = is_s1_valid
            prev_q_offset = q_offset
            prev_rows = rows
            prev_vk_param = vk_param
            if is_s1_valid:
                prev_valid_k_size = valid_k_size
                prev_s2_tail_size = s2_tail_size
            else:
                prev_valid_k_size = 0
                prev_s2_tail_size = TKV
    return


# ================================================================
# Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# ================================================================
@pl.jit(
    auto_mutex=True,
    tiling_key=DenseLiLseTilingKey,
    datatype={
        "query_index": "input_dtype",
    },
)
def dense_lightning_indexer_softmax_lse_v2(
    query_index: pl.Ptr[pl.DT_UINT8],
    key_index: pl.Ptr[pl.DT_UINT8],
    weight: pl.Ptr[pl.DT_UINT8],
    cu_seq_lens_q: pl.Ptr[pl.DT_UINT8],
    cu_seq_lens_k: pl.Ptr[pl.DT_UINT8],
    seq_used_q: pl.Ptr[pl.DT_UINT8],
    seq_used_k: pl.Ptr[pl.DT_UINT8],
    cmp_residual_k: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: DenseLILseV2Tiling,
):
    tensor_seqused_q = pl.make_tensor(
        seq_used_q,
        [tiling.b, 1],
        [1, 1],
        dtype=pl.DT_INT32,
    )
    tensor_seqused_k = pl.make_tensor(
        seq_used_k,
        [tiling.b, 1],
        [1, 1],
        dtype=pl.DT_INT32,
    )
    tensor_cmp_residual_k = pl.make_tensor(
        cmp_residual_k,
        [tiling.b, 1],
        [1, 1],
        dtype=pl.DT_INT32,
    )
    tensor_cu_seq_lens_q = pl.make_tensor(
        cu_seq_lens_q,
        [tiling.b + 1, 1],
        [1, 1],
        dtype=pl.DT_INT32,
    )
    tensor_cu_seq_lens_k = pl.make_tensor(
        cu_seq_lens_k,
        [tiling.b + 1, 1],
        [1, 1],
        dtype=pl.DT_INT32,
    )
    tensor_metadata = pl.make_tensor(
        metadata,
        [1, SCHED_METADATA_MAX_SIZE],
        [SCHED_METADATA_MAX_SIZE, 1],
        dtype=pl.DT_INT32,
    )
    tensor_workspace_max = pl.make_tensor(
        workspace,
        [1, WS_MAX_ELEMS],
        [WS_MAX_ELEMS, 1],
        dtype=pl.DT_FP32,
    )
    tensor_workspace_sum = pl.make_tensor(
        workspace + WS_MAX_BYTES,
        [1, WS_MAX_ELEMS],
        [WS_MAX_ELEMS, 1],
        dtype=pl.DT_FP32,
    )
    s2_size_aligned = (tiling.s2 + TKV - 1) // TKV * TKV
    tensor_workspace = pl.make_tensor(
        workspace + WS_MAX_BYTES * 2,
        [tiling.block_num, s2_size_aligned],
        [s2_size_aligned, 1],
        dtype=pl.DT_FP32,
    )

    if layout == 1:
        tensor_query_index = pl.make_tensor(
            query_index,
            [tiling.s1, 1, tiling.n1, tiling.d],
            [tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )

        # flat 2D view over the packed (row, head) space: a [rows*n1, D] load
        # from q_head_offset covers a whole sched-table row block
        tensor_query_index_flat = pl.make_tensor(
            query_index,
            [tiling.s1 * tiling.n1, tiling.d],
            [tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_key_index = pl.make_tensor(
            key_index,
            [tiling.s2, 1, 1, tiling.d],
            [tiling.d, tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_weight = pl.make_tensor(
            weight,
            [tiling.s1, 1, tiling.n1],
            [tiling.n1, tiling.n1, 1],
            dtype=pl.DT_FP32,
        )
        tensor_lse_out = pl.make_tensor(
            softmax_lse,
            [tiling.s1, 1],
            [1, 1],
            dtype=pl.DT_FP32,
        )
        dense_lightning_indexer_softmax_lse_v2_inner(
            tensor_query_index,
            tensor_query_index_flat,
            tensor_key_index,
            tensor_weight,
            tensor_lse_out,
            tensor_seqused_q,
            tensor_seqused_k,
            tensor_cmp_residual_k,
            tensor_cu_seq_lens_q,
            tensor_cu_seq_lens_k,
            tensor_workspace_max,
            tensor_workspace_sum,
            tensor_workspace,
            tiling,
            tensor_metadata,
        )
    else:
        total_s1 = tiling.b * tiling.s1
        total_s2 = tiling.b * tiling.s2
        tensor_query_index = pl.make_tensor(
            query_index,
            [total_s1, 1, tiling.n1, tiling.d],
            [tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_query_index_flat = pl.make_tensor(
            query_index,
            [total_s1 * tiling.n1, tiling.d],
            [tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_key_index = pl.make_tensor(
            key_index,
            [total_s2, 1, 1, tiling.d],
            [tiling.d, tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_weight = pl.make_tensor(
            weight,
            [total_s1, 1, tiling.n1],
            [tiling.n1, tiling.n1, 1],
            dtype=pl.DT_FP32,
        )
        tensor_lse_out = pl.make_tensor(
            softmax_lse,
            [total_s1, 1],
            [1, 1],
            dtype=pl.DT_FP32,
        )
        dense_lightning_indexer_softmax_lse_v2_inner(
            tensor_query_index,
            tensor_query_index_flat,
            tensor_key_index,
            tensor_weight,
            tensor_lse_out,
            tensor_seqused_q,
            tensor_seqused_k,
            tensor_cmp_residual_k,
            tensor_cu_seq_lens_q,
            tensor_cu_seq_lens_k,
            tensor_workspace_max,
            tensor_workspace_sum,
            tensor_workspace,
            tiling,
            tensor_metadata,
        )
