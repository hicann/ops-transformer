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
"""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 1
TKV = 128
TD = 128
TN = 128
TILINGDATA_N = 128
TKV_HALF = TKV // 2

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


VB4_KV = TN * TKV_HALF * 4

VA0 = 0
VA1 = _align_up(VA0 + VB4_KV * 2, 32)
VA2 = _align_up(VA1 + TN * 4 * 2, 32)
VA3 = _align_up(VA2 + 1, 32)
VA4 = _align_up(VA3 + 1, 32)
KV_CACHE_NUM = _align_up(248 * 1024 - VA4, 32) // 4

MAX_UB_S2_TILES = KV_CACHE_NUM // TKV_HALF

QK_READY_IDS = (0, 1)
QK_FREE_IDS = (2, 3)

WS_MAX_ELEMS = 72
WS_MAX_BYTES = 288
WS_MAX_CORES = 36


# ================================================================
#  Tiling data
# ================================================================
@dataclass
class DenseLILseV2Tiling:
    b: int
    s1: int
    s2: int
    n1: int
    d: int
    cmp_ratio: int


# ================================================================
#  TilingKey
# ================================================================
class DenseLiLseTilingKey:
    is_long_s2 = TilingKeyField(bits=1, values=[0, 1])
    has_seq_used_q = TilingKeyField(bits=1, values=[0, 1])
    has_seq_used_k = TilingKeyField(bits=1, values=[0, 1])
    mask_mode = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    layout = TilingKeyField(bits=1, values=[0, 1])


# ================================================================
#  VF helper functions
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
def process_reduce_max_vf(dst_tile, qk_tile, weight_tile, max_tile, n1_dim, s2_size):
    vreg_reduce_sum = vf.full(0, dtype=pl.DT_FP32)
    preg_s2 = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_max = vf.update_mask(1, dtype=pl.DT_FP32)

    unroll = 8
    n1_main = n1_dim // unroll * unroll
    n1_tail = n1_dim % unroll

    for m in pl.range(0, n1_main, unroll):
        vreg_x0 = vf.load_align(qk_tile, (m + 0) * TKV_HALF)
        vreg_w0 = vf.load_align(weight_tile, m + 0, dist=pl.LoadDist.BRC_B32)
        vreg_relu0 = vf.relu(vreg_x0, preg_s2)
        vreg_mul0 = vf.mul(vreg_relu0, vreg_w0, preg_s2)

        vreg_x1 = vf.load_align(qk_tile, (m + 1) * TKV_HALF)
        vreg_w1 = vf.load_align(weight_tile, m + 1, dist=pl.LoadDist.BRC_B32)
        vreg_relu1 = vf.relu(vreg_x1, preg_s2)
        vreg_mul1 = vf.mul(vreg_relu1, vreg_w1, preg_s2)

        vreg_x2 = vf.load_align(qk_tile, (m + 2) * TKV_HALF)
        vreg_w2 = vf.load_align(weight_tile, m + 2, dist=pl.LoadDist.BRC_B32)
        vreg_relu2 = vf.relu(vreg_x2, preg_s2)
        vreg_mul2 = vf.mul(vreg_relu2, vreg_w2, preg_s2)

        vreg_x3 = vf.load_align(qk_tile, (m + 3) * TKV_HALF)
        vreg_w3 = vf.load_align(weight_tile, m + 3, dist=pl.LoadDist.BRC_B32)
        vreg_relu3 = vf.relu(vreg_x3, preg_s2)
        vreg_mul3 = vf.mul(vreg_relu3, vreg_w3, preg_s2)

        vreg_x4 = vf.load_align(qk_tile, (m + 4) * TKV_HALF)
        vreg_w4 = vf.load_align(weight_tile, m + 4, dist=pl.LoadDist.BRC_B32)
        vreg_relu4 = vf.relu(vreg_x4, preg_s2)
        vreg_mul4 = vf.mul(vreg_relu4, vreg_w4, preg_s2)

        vreg_x5 = vf.load_align(qk_tile, (m + 5) * TKV_HALF)
        vreg_w5 = vf.load_align(weight_tile, m + 5, dist=pl.LoadDist.BRC_B32)
        vreg_relu5 = vf.relu(vreg_x5, preg_s2)
        vreg_mul5 = vf.mul(vreg_relu5, vreg_w5, preg_s2)

        vreg_x6 = vf.load_align(qk_tile, (m + 6) * TKV_HALF)
        vreg_w6 = vf.load_align(weight_tile, m + 6, dist=pl.LoadDist.BRC_B32)
        vreg_relu6 = vf.relu(vreg_x6, preg_s2)
        vreg_mul6 = vf.mul(vreg_relu6, vreg_w6, preg_s2)

        vreg_x7 = vf.load_align(qk_tile, (m + 7) * TKV_HALF)
        vreg_w7 = vf.load_align(weight_tile, m + 7, dist=pl.LoadDist.BRC_B32)
        vreg_relu7 = vf.relu(vreg_x7, preg_s2)
        vreg_mul7 = vf.mul(vreg_relu7, vreg_w7, preg_s2)

        vreg_sum07 = vf.add(vreg_mul0, vreg_mul7, preg_s2)
        vreg_sum16 = vf.add(vreg_mul1, vreg_mul6, preg_s2)
        vreg_sum25 = vf.add(vreg_mul2, vreg_mul5, preg_s2)
        vreg_sum34 = vf.add(vreg_mul3, vreg_mul4, preg_s2)
        vreg_sum0716 = vf.add(vreg_sum07, vreg_sum16, preg_s2)
        vreg_sum2534 = vf.add(vreg_sum25, vreg_sum34, preg_s2)
        vreg_sum_tmp = vf.add(vreg_sum0716, vreg_sum2534, preg_s2)
        vreg_reduce_sum = vf.add(vreg_reduce_sum, vreg_sum_tmp, preg_s2)

    for m in pl.range(n1_main, n1_dim):
        vreg_x = vf.load_align(qk_tile, m * TKV_HALF)
        vreg_w = vf.load_align(weight_tile, m, dist=pl.LoadDist.BRC_B32)
        vreg_relu = vf.relu(vreg_x, preg_s2)
        vreg_mul = vf.mul(vreg_relu, vreg_w, preg_s2)
        vreg_reduce_sum = vf.add(vreg_reduce_sum, vreg_mul, preg_s2)

    vreg_reduce_max = vf.reduce_max(
        vreg_reduce_sum, preg_s2, merge_mode=pl.MergeMode.ZEROING
    )
    vreg_tmp_max = vf.load_align(max_tile, 0)
    vreg_max = vf.max(vreg_tmp_max, vreg_reduce_max, preg_max)
    vf.store_align(dst_tile, vreg_reduce_sum, preg_s2)
    vf.store_align(max_tile, vreg_max, preg_max)


@pl.vector_function
def process_reduce_sum_vf(dst_tile, reduce_sum_vec_tile, tmp_max, s2_tiles, tail_size):
    vreg_max = vf.full(tmp_max, dtype=pl.DT_FP32)
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(tail_size, dtype=pl.DT_FP32)
    preg_sum = vf.update_mask(1, dtype=pl.DT_FP32)

    unroll = 8
    main_tiles = (s2_tiles - 1) // unroll * unroll

    vreg_all_sum = vf.full(0, dtype=pl.DT_FP32)

    for m in pl.range(0, main_tiles, unroll):
        vreg_x0 = vf.load_align(reduce_sum_vec_tile, (m + 0) * TKV_HALF)
        vreg_exp0 = vf.exp_sub(vreg_x0, vreg_max, preg_all)

        vreg_x1 = vf.load_align(reduce_sum_vec_tile, (m + 1) * TKV_HALF)
        vreg_exp1 = vf.exp_sub(vreg_x1, vreg_max, preg_all)

        vreg_x2 = vf.load_align(reduce_sum_vec_tile, (m + 2) * TKV_HALF)
        vreg_exp2 = vf.exp_sub(vreg_x2, vreg_max, preg_all)

        vreg_x3 = vf.load_align(reduce_sum_vec_tile, (m + 3) * TKV_HALF)
        vreg_exp3 = vf.exp_sub(vreg_x3, vreg_max, preg_all)

        vreg_x4 = vf.load_align(reduce_sum_vec_tile, (m + 4) * TKV_HALF)
        vreg_exp4 = vf.exp_sub(vreg_x4, vreg_max, preg_all)

        vreg_x5 = vf.load_align(reduce_sum_vec_tile, (m + 5) * TKV_HALF)
        vreg_exp5 = vf.exp_sub(vreg_x5, vreg_max, preg_all)

        vreg_x6 = vf.load_align(reduce_sum_vec_tile, (m + 6) * TKV_HALF)
        vreg_exp6 = vf.exp_sub(vreg_x6, vreg_max, preg_all)

        vreg_x7 = vf.load_align(reduce_sum_vec_tile, (m + 7) * TKV_HALF)
        vreg_exp7 = vf.exp_sub(vreg_x7, vreg_max, preg_all)

        vreg_sum01 = vf.add(vreg_exp0, vreg_exp1, preg_all)
        vreg_sum23 = vf.add(vreg_exp2, vreg_exp3, preg_all)
        vreg_sum45 = vf.add(vreg_exp4, vreg_exp5, preg_all)
        vreg_sum67 = vf.add(vreg_exp6, vreg_exp7, preg_all)
        vreg_sum0123 = vf.add(vreg_sum01, vreg_sum23, preg_all)
        vreg_sum4567 = vf.add(vreg_sum45, vreg_sum67, preg_all)
        vreg_chunk_sum = vf.add(vreg_sum0123, vreg_sum4567, preg_all)

        vreg_all_sum = vf.add(vreg_all_sum, vreg_chunk_sum, preg_all)

    for m in pl.range(main_tiles, s2_tiles - 1):
        vreg_x = vf.load_align(reduce_sum_vec_tile, m * TKV_HALF)
        vreg_exp = vf.exp_sub(vreg_x, vreg_max, preg_all)
        vreg_all_sum = vf.add(vreg_all_sum, vreg_exp, preg_all)

    vreg_x = vf.load_align(reduce_sum_vec_tile, (s2_tiles - 1) * TKV_HALF)
    vreg_exp = vf.exp_sub(vreg_x, vreg_max, preg_tail)
    vreg_all_sum_tail = vf.full(0, dtype=pl.DT_FP32)
    vreg_all_sum_tail = vf.add(vreg_all_sum_tail, vreg_exp, preg_tail)
    vreg_all_sum = vf.add(vreg_all_sum, vreg_all_sum_tail, preg_all)
    vreg_reduce_sum = vf.reduce_sum(
        vreg_all_sum, preg_all, merge_mode=pl.MergeMode.ZEROING
    )
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
    sub_id,
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
            src_tile = reduce_sum_vec_tile[0:1, ci * TKV_HALF :]
            cur_vec_size = (
                prev_s2_vec_tail_size
                if (chunk_start + ci) == prev_s2_tiles - 1
                else TKV_HALF
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
                )
            pl.system.set_cross_core(
                pipe=pl.PipeType.V, event_id=QK_FREE_IDS[task_id % 2]
            )
            task_id = task_id + 1
        for ci in pl.range(0, chunk_tiles):
            ws_tile = reduce_sum_vec_tile[0:1, ci * TKV_HALF :]
            cur_vec_size = (
                prev_s2_vec_tail_size
                if (chunk_start + ci) == prev_s2_tiles - 1
                else TKV_HALF
            )
            if cur_vec_size > 0:
                pl.set_validshape(ws_tile, [1, cur_vec_size])
                pl.store(
                    workspace,
                    ws_tile,
                    [core_id, (chunk_start + ci) * TKV + TKV_HALF * sub_id],
                )
    return task_id


def process_long_s2_reduce_sum(
    lse_vec_tile,
    reduce_sum_vec,
    workspace,
    tmp_s2_max,
    core_id,
    sub_id,
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
            src = chunk_tile[0:1, ci * TKV_HALF :]
            pl.set_validshape(src, [1, TKV_HALF])
            pl.load(
                src, workspace, [core_id, (chunk_start + ci) * TKV + TKV_HALF * sub_id]
            )
        chunk_tail_sum = (
            prev_s2_vec_tail_size if chunk_idx == num_chunks - 1 else TKV_HALF
        )
        process_reduce_sum_vf(
            lse_vec_tile, chunk_tile, tmp_s2_max, chunk_tiles, chunk_tail_sum
        )
        ws_chunk_sum = ws_chunk_sum + pl.getval(lse_vec_tile, 0)
    write_scalar_vf(lse_vec_tile, ws_chunk_sum)

# ================================================================
#  Inner kernel — receives typed tensor views + workspace
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

    core_id = pl.get_block_idx() // pl.get_subblock_num()
    sub_id = pl.get_subblock_idx()

    fore_core_num = pl.getval(metadata, 0)
    tail_core_num = pl.getval(metadata, 1)
    b_s1_per_core = pl.getval(metadata, 2)
    b_s1_per_tail_core = pl.getval(metadata, 3)
    total_cores = fore_core_num + tail_core_num
    work_count = 0
    message = pl.struct("Message", max_v0=0.0, sum_v0=0.0, max_v1=0.0, sum_v1=0.0)

    if core_id < fore_core_num:
        work_count = b_s1_per_core
    elif core_id < total_cores:
        work_count = b_s1_per_tail_core

    # 【128，64】基本块用于qk的结果，这里UB开两块做db
    qk_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[TN, TKV_HALF], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
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
            shape=[1, 8], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
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
        for j in pl.range(0, work_count):
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
            valid_k_size = k_used_size
            if mask_mode == 3:
                cur_residual_k = 0
                if cmp_ratio > 1:
                    cur_residual_k = pl.getval(cmp_residual_k, b_idx)
                ori_k_size = k_used_size * cmp_ratio + cur_residual_k
                qk_residual = q_used_size - ori_k_size
                valid_s1_start = pl.max(qk_residual, 0)
                valid_k_size = pl.max((s1_i + 1 - qk_residual) // cmp_ratio, 0)
            is_s1_valid = s1_i >= valid_s1_start and valid_k_size > 0
            if is_s1_valid:
                s2_tiles = (valid_k_size + TKV - 1) // TKV
                s2_tail_size = TKV + valid_k_size - s2_tiles * TKV

                q_tile = q_mat.next()
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
                        pl.set_validshape(left_tile, [n1_dim, TD])
                        if s2_i == s2_tiles and s2_tail_size != TKV:
                            pl.set_validshape(right_tile, [TD, s2_tail_size])
                            pl.set_validshape(k_tile_cur, [TD, s2_tail_size])
                            pl.set_validshape(acc_tile, [n1_dim, s2_tail_size])
                        else:
                            pl.set_validshape(right_tile, [TD, TKV])
                            pl.set_validshape(k_tile_cur, [TD, TKV])
                            pl.set_validshape(acc_tile, [n1_dim, TKV])
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
                            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN,
                        )
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.FIX, event_id=QK_READY_IDS[task_id % 2]
                        )
                        task_id = task_id + 1

                # C侧汇聚V0/V1的max值
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.ssbuf_load(message, 0)
                max0 = message.max_v0
                pl.ssbuf_load(message, 32)
                max1 = message.max_v1
                message.max_v0 = max0
                message.max_v1 = max1
                tmp_max = pl.max(max0, max1)
                message.max_v1 = tmp_max
                pl.ssbuf_store(message, 64)
                pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)

                # C侧汇聚V0/V1的sum值
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.ssbuf_load(message, 0)
                sum0 = message.sum_v0
                pl.ssbuf_load(message, 32)
                sum1 = message.sum_v1
                tmp_sum = sum0 + sum1
                message.sum_v1 = tmp_sum
                pl.ssbuf_store(message, 64)
                pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
   
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_FREE_IDS[0])
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_FREE_IDS[1])
    with pl.section_vector():
        task_id = 0
        prev_s2_tiles = 0
        prev_is_s1_valid = False
        prev_valid_k_size = 0
        prev_s2_tail_size = TKV
        prev_q_offset = 0
        s2_tiles = 0
        s2_tail_size = 0
        valid_k_size = 0
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_FREE_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_FREE_IDS[1])

        for j in pl.range(0, work_count + 1):
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
                valid_k_size = k_used_size
                if mask_mode == 3:
                    cur_residual_k = 0
                    if cmp_ratio > 1:
                        cur_residual_k = pl.getval(cmp_residual_k, b_idx)
                    ori_k_size = k_used_size * cmp_ratio + cur_residual_k
                    qk_residual = q_used_size - ori_k_size
                    valid_s1_start = pl.max(qk_residual, 0)
                    valid_k_size = pl.max((s1_i + 1 - qk_residual) // cmp_ratio, 0)
                is_s1_valid = s1_i >= valid_s1_start and valid_k_size > 0
                if is_s1_valid:
                    s2_tiles = (valid_k_size + TKV - 1) // TKV
                    s2_tail_size = TKV + valid_k_size - s2_tiles * TKV
                    pl.load(weight_vec_nxt, weight, [q_offset, 0, 0])

            if j > 0 and prev_is_s1_valid:
                reduce_sum_vec_tile = reduce_sum_vec.next()
                lse_vec_tile = lse_vec.next()

                prev_s2_vec_tail_size = TKV_HALF
                if prev_valid_k_size % TKV != 0:
                    s2_tail_align = ((prev_s2_tail_size + 31) // 32) * 32
                    split_half = s2_tail_align // 2
                    if sub_id == 0:
                        prev_s2_vec_tail_size = pl.min(prev_s2_tail_size, split_half)
                    else:
                        prev_s2_vec_tail_size = pl.max(
                            prev_s2_tail_size - split_half, 0
                        )

                if is_long_s2 == 0:
                    for s2_i in pl.range(0, prev_s2_tiles):
                        qk_vec_tile = qk_vec.next()
                        weight_tile_cur = weight_vec.previous()
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.V, event_id=QK_READY_IDS[task_id % 2]
                        )
                        src_tile = reduce_sum_vec_tile[0:1, s2_i * TKV_HALF :]
                        cur_vec_size = (
                            prev_s2_vec_tail_size
                            if s2_i == prev_s2_tiles - 1
                            else TKV_HALF
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
                        sub_id,
                        prev_s2_tiles,
                        prev_s2_vec_tail_size,
                        j,
                        work_count,
                        task_id,
                    )

                if sub_id == 0:
                    message.max_v0 = pl.getval(max_tile, 0)
                    pl.ssbuf_store(message, 0)
                else:
                    message.max_v1 = pl.getval(max_tile, 0)
                    pl.ssbuf_store(message, 32)

                pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=4, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=5, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.ssbuf_load(message, 64)
                tmp_s2_max = message.max_v1

                if is_long_s2 == 0:
                    process_reduce_sum_vf(
                        lse_vec_tile,
                        reduce_sum_vec_tile,
                        tmp_s2_max,
                        prev_s2_tiles,
                        prev_s2_vec_tail_size,
                    )
                else:
                    process_long_s2_reduce_sum(
                        lse_vec_tile,
                        reduce_sum_vec,
                        workspace,
                        tmp_s2_max,
                        core_id,
                        sub_id,
                        prev_s2_tiles,
                        prev_s2_vec_tail_size,
                    )

                if sub_id == 0:
                    message.sum_v0 = pl.getval(lse_vec_tile, 0)
                    pl.ssbuf_store(message, 0)
                else:
                    message.sum_v1 = pl.getval(lse_vec_tile, 0)
                    pl.ssbuf_store(message, 32)
                pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=6, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=7, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
                pl.ssbuf_load(message, 64)
                reduce_res = message.sum_v1

                process_lse_vf(lse_vec_tile, reduce_res, tmp_s2_max)
                pl.set_validshape(lse_vec_tile, [1, 1])
                pl.store(lse_out, lse_vec_tile, [prev_q_offset, 0])

            prev_s2_tiles = s2_tiles
            prev_is_s1_valid = is_s1_valid
            prev_q_offset = q_offset
            if is_s1_valid:
                prev_valid_k_size = valid_k_size
                prev_s2_tail_size = s2_tail_size
            else:
                prev_valid_k_size = 0
                prev_s2_tail_size = TKV
    return


# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
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
        [1, 64],
        [64, 1],
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
        [WS_MAX_CORES, s2_size_aligned],
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
