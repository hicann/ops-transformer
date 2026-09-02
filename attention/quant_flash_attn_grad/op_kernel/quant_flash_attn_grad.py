# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""QuantFlashAttnGrad kernel (pypto-pro)."""

from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


# ================================================================
#  Tile dimensions and constants
# ================================================================
D_SIZE = 128
HALF_D_SIZE = 64
TS = 128
TS_HALF = 64
TKV = 128
TG = 1
TD = 128
CUBE_BASEM = 512
CUBE_BASEN = 512
VEC_CHUNK = 128

M_16 = 16
N_4096 = 4096

BASE_K = 128
BASE_N = 128
K_SIZE = 256
QUANT_S1_BASE_COUNT = 64
QUANT_S2_BASE_COUNT = 8
BLOCK_SIZE = 32
# ---- VEC addresses ----
CUBE_BUFFER_NUM = 2
VEC_BUFFER_NUM = 2
VB4_KV = TS_HALF * TS * 4 * VEC_BUFFER_NUM  # [TS_HALF, TS] FP32 = 64KB
VB_Q = TS * TS * 4
VB1_Y = TS * VEC_CHUNK * 2 * VEC_BUFFER_NUM
VB2_DX = TS * VEC_CHUNK * VEC_BUFFER_NUM
VB_P = TS * TS * 4 * VEC_BUFFER_NUM
# ================================================================
#  sync flag ids
# ================================================================
SYNC_COMPUTE_DKV_FLAG = 2
SYNC_TRANSFER_DKV_FLAG = 3
SYNC_TRANSFER_DQ_FLAG = 4
SYNC_UB2L1_P_FLAG = 9
SYNC_UB2L1_DS_FLAG = 10
SYNC_PDS_TO_DKV_FLAG = 8
SYNC_PDS_TO_DQ_FLAG = 7
SYNC_DETER_FLAG = 11
SYNC_PDS_TO_DKV_FLAG_TAIL = 12


def _align_up(value, align=1024):
    return ((value + align - 1) // align) * align


def ceil(a, b):
    return (a + b - 1) // b


def _gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


VAQ_PRE = 0
VA0_PRE = _align_up(VAQ_PRE + VB_Q)
VA1_PRE = _align_up(VA0_PRE + VB1_Y)
VA2_PRE = _align_up(VA1_PRE + VB2_DX)

VA0 = 0
VA1 = _align_up(VA0 + VB4_KV)
VA2 = _align_up(VA1 + VB4_KV)
VA3 = _align_up(VA2 + VB4_KV)
VA4 = VA3 + TS_HALF * 4
VAP_0 = 0
VAP_1 = _align_up(VAP_0 + VB_P)
MA0 = 0
MA1 = MA0 + TS * TS * 16
MA2 = MA1 + TS * TS * 4
MA3 = MA2 + TS * TS * 4
MA4 = MA3 + TS * TS * 4

LA0 = 0
RA0 = 0
CA0 = 0
CA1 = CA0 + TS * TS * 4
CA2 = CA1 + TS * TS * 4


# ================================================================
#  Tiling data
# ================================================================
@dataclass
class QuantFlashAttnGradTiling:
    b: int
    s1: int
    s2: int
    n1: int
    n2: int
    g: int
    d: int
    t1: int
    t2: int
    softmax_scale: float
    s1_outer: int
    s2_outer: int
    s1_tail: int
    s2_tail: int
    has_seq_used_q: bool
    has_seq_used_k: bool
    metadata_len: int

    dq_work_space_offset: int
    dk_work_space_offset: int
    dv_work_space_offset: int
    sfmg_work_space_offset: int
    q_pre_block_factor: int
    q_pre_block_total: int
    q_pre_block_tail: int
    k_pre_block_factor: int
    k_pre_block_total: int
    k_pre_block_tail: int
    v_pre_block_factor: int
    v_pre_block_total: int
    v_pre_block_tail: int

    sfmg_used_core_num: int
    sfmg_dy_buffer_len: int
    sfmg_y_buffer_len: int
    sfmg_output_buffer_len: int
    single_loop_nburst_num: int
    normal_core_loop_times: int
    tail_core_loop_times: int
    normal_core_last_loop_nburst_num: int
    tail_core_last_loop_nburst_num: int
    normal_core_nburst_nums: int
    tail_core_nburst_nums: int
    normal_axis_size: int

    q_post_block_factor: int
    q_post_block_total: int
    q_post_base_num: int
    q_post_tail_num: int
    k_post_block_factor: int
    k_post_block_total: int
    k_post_base_num: int
    k_post_tail_num: int
    v_post_block_factor: int
    v_post_block_total: int
    v_post_base_num: int
    v_post_tail_num: int


class QuantFlashAttnGradTilingKey:
    has_attn_mask = TilingKeyField(bits=1, values=[0])
    has_sink = TilingKeyField(bits=1, values=[0])
    s1_template_num = TilingKeyField(bits=1, values=[512])
    s2_template_num = TilingKeyField(bits=1, values=[512])
    d_template_num = TilingKeyField(bits=1, values=[128])
    is_n_equal = TilingKeyField(bits=1, values=[1])
    # 0: BSND 1: BNSD 2: TND
    layout = TilingKeyField(bits=2, values=[0, 1])


def l1_ds(i, j, task_mod2):
    return i + j * 4 if task_mod2 == 0 else i * 4 + j


def cal_deter_max_loop_num(const_info):
    b = const_info.b_size * const_info.n2_size
    m = const_info.s1_outer
    n = const_info.s2_outer
    k = pl.get_block_num()
    res = 0
    if n == 1:
        res = max(ceil(m * b, k), m)
    else:
        res = ceil(n * b, min(k, m * b)) * m
    return res


def init_index(sfmg_output_offset, const_info):
    start_idx = sfmg_output_offset * const_info.d_size
    b_idx = start_idx // (const_info.n1_size * const_info.s1_size * const_info.d_size)
    b_tail = start_idx % (const_info.n1_size * const_info.s1_size * const_info.d_size)
    n_idx = b_tail // (const_info.s1_size * const_info.d_size)
    n_tail = b_tail % (const_info.s1_size * const_info.d_size)
    s1_idx = n_tail // const_info.d_size
    return b_idx, n_idx, s1_idx


@pl.vector_function
def anti_quant_softmax_grad_front_cast_hif8_vf(
    src_m, deq_scale_do_value, y_vec, dx_vec, out_vec
):
    for m in pl.range(0, src_m):
        preg_all_8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
        preg_all_16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
        preg_all_32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        vreg_dx = vf.load_align(dx_vec, m * 128)
        vreg_y = vf.load_align(y_vec, m * 128)
        vreg_dx1 = vf.astype(
            vreg_dx, preg_all_8, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vreg_dx2 = vf.astype(
            vreg_dx, preg_all_8, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vreg_dx3 = vf.astype(
            vreg_dx, preg_all_8, dtype=pl.DT_FP32, layout=pl.CastLayout.TWO
        )
        vreg_dx4 = vf.astype(
            vreg_dx, preg_all_8, dtype=pl.DT_FP32, layout=pl.CastLayout.THREE
        )
        vreg_y1 = vf.astype(
            vreg_y, preg_all_16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vreg_y2 = vf.astype(
            vreg_y, preg_all_16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vreg_dx1, vreg_dx3 = vf.interleave(vreg_dx1, vreg_dx3)
        vreg_dx2, vreg_dx4 = vf.interleave(vreg_dx2, vreg_dx4)
        vreg_dx1 = vf.muls(vreg_dx1, deq_scale_do_value, preg_all_32)
        vreg_dx2 = vf.muls(vreg_dx2, deq_scale_do_value, preg_all_32)
        vreg_res1 = vf.mul(vreg_dx1, vreg_y1, preg_all_32)
        vreg_res2 = vf.mul(vreg_dx2, vreg_y2, preg_all_32)
        vreg_res1 = vf.add(vreg_res1, vreg_res2, preg_all_32)
        vreg_res = vf.reduce_sum(vreg_res1, preg_all_32)
        vf.store(out_vec, vreg_res, 1, post_update=True)


def init_dq_workspace(const_info, tensor_info):
    if const_info.core_id_vec < const_info.sfmg_used_core_num:
        dq_tt = pl.TileType(
            shape=[1, TS * TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        )
        dq_tile = pl.make_tile_group(type=dq_tt, addrs=VAQ_PRE, mutex_ids=[7])
        pl.expands(dq_tile.current(), 0)
        init_dq_size = (
            const_info.q_pre_block_tail
            if const_info.core_id_vec == const_info.q_pre_block_factor - 1
            else const_info.q_pre_block_factor
        )
        dq_offset = const_info.core_id_vec * const_info.q_pre_block_factor
        for i in pl.range(0, init_dq_size, 128 * 128):
            size = TS * TS if init_dq_size - i > TS * TS else init_dq_size - i
            pl.set_validshape(dq_tile, [1, size])
            pl.store(
                tensor_info.tensor_workspace_dq_flat,
                dq_tile.current(),
                offsets=[0, dq_offset + i],
            )


def presfmg_quant_inner_hif8(const_info, tensor_info):
    with pl.section_vector():
        init_dq_workspace(const_info, tensor_info)
        n_burst = const_info.single_loop_nburst_num
        num_of_128_in_s1 = (const_info.s1_size + n_burst - 1) // n_burst
        s1_residual = (
            const_info.s1_size % n_burst
            if const_info.s1_size % n_burst != 0
            else n_burst
        )

        num_d_chunks = (const_info.d_size + VEC_CHUNK - 1) // VEC_CHUNK
        tail_d_chunk = (
            const_info.d_size % VEC_CHUNK
            if const_info.d_size % VEC_CHUNK != 0
            else VEC_CHUNK
        )

        deq_scale_do_value = pl.getval(tensor_info.tensor_do_descale, 0)
        y_tt = pl.TileType(
            shape=[TS, D_SIZE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
        )
        dx_tt = pl.TileType(
            shape=[TS, D_SIZE], dtype=pl.DT_HF8, target_memory=pl.MemorySpace.Vec
        )
        out_tt = pl.TileType(
            shape=[1, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        )

        y_tile = pl.make_tile_group(type=y_tt, addrs=VA0_PRE, mutex_ids=[0, 1])
        dx_tile = pl.make_tile_group(type=dx_tt, addrs=VA1_PRE, mutex_ids=[2, 3])
        out_tile = pl.make_tile_group(type=out_tt, addrs=VA2_PRE, mutex_ids=[4, 5])

        data_block_idx = const_info.core_id_vec - const_info.sfmg_used_core_num
        while True:
            data_block_idx = data_block_idx + const_info.sfmg_used_core_num
            sfmg_output_offset = data_block_idx * n_burst - (
                data_block_idx // num_of_128_in_s1
            ) * (n_burst - s1_residual)
            if sfmg_output_offset >= const_info.normal_axis_size:
                break
            y_vec = y_tile.next()
            dx_vec = dx_tile.next()
            out_vec = out_tile.next()

            cur_n_burst = (
                s1_residual if (data_block_idx + 1) % num_of_128_in_s1 == 0 else n_burst
            )
            b_idx, n_idx, s1_idx = init_index(sfmg_output_offset, const_info)
            pl.set_validshape(y_vec, [cur_n_burst, D_SIZE])
            pl.set_validshape(dx_vec, [cur_n_burst, D_SIZE])
            if layout == 0:  # BSND
                pl.load(
                    y_vec,
                    tensor_info.tensor_attn_out,
                    [b_idx, s1_idx, n_idx, 0],
                    order=[1, 3],
                )
                pl.load(
                    dx_vec,
                    tensor_info.tensor_do,
                    [b_idx, s1_idx, n_idx, 0],
                    order=[1, 3],
                )
            if layout == 1:  # BNSD
                pl.load(
                    y_vec,
                    tensor_info.tensor_attn_out,
                    [b_idx, n_idx, s1_idx, 0],
                    order=[2, 3],
                )
                pl.load(
                    dx_vec,
                    tensor_info.tensor_do,
                    [b_idx, n_idx, s1_idx, 0],
                    order=[2, 3],
                )

            anti_quant_softmax_grad_front_cast_hif8_vf(
                cur_n_burst, deq_scale_do_value, y_vec, dx_vec, out_vec
            )
            pl.set_validshape(out_vec, [1, cur_n_burst])
            pl.store(
                tensor_info.tensor_workspace_sfmg,
                out_vec,
                [b_idx, n_idx, s1_idx],
                order=[1, 2],
            )


def init_coordinate_info(const_info, m_offset, n_offset, coordinate_info):
    coordinate_info.s1_outer = const_info.s1_outer
    coordinate_info.s2_outer = const_info.s2_outer
    coordinate_info.m_offset = m_offset
    coordinate_info.n_offset = n_offset


def alloc_event_id():
    with pl.section_vector():
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=0,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=1,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=SYNC_TRANSFER_DKV_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=SYNC_TRANSFER_DQ_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=SYNC_DETER_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )

    with pl.section_cube():
        for i in pl.range(2):
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1,
                event_id=SYNC_UB2L1_P_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
        for i in pl.range(8):
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1,
                event_id=SYNC_UB2L1_DS_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )


def free_event_id():
    with pl.section_cube():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=0, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=1, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_TRANSFER_DKV_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_TRANSFER_DQ_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )

    with pl.section_vector():
        for i in pl.range(2):
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_UB2L1_P_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
        for i in pl.range(8):
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_UB2L1_DS_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=SYNC_DETER_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )


def cal_dense_index_for_single_n(k, m, b, j, r, R, coordinate):
    n = 1
    id_ = (j - 1) * R + r
    num = m * n

    delta1 = (id_ - 1) // num + 1
    delta = id_ % num
    delta = num if delta == 0 else delta
    g = _gcd(m, R)
    t1 = R // g
    t2 = m // g

    x = (delta - 1) % m + 1
    y = (delta - 1) // m + 1

    if t1 < n:
        n1 = n % t1
        n1 = t1 if n1 == 0 else n1
        if y <= n - n1:
            delta_adj = ceil(y, t1)
            delta += delta_adj
            if delta > delta_adj * t2 * R:
                delta -= t2 * R
            x = (delta - 1) % m + 1
            y = (delta - 1) // m + 1

    coordinate.batch_id = delta1
    coordinate.s1_idx = x
    coordinate.s2_idx = y


def cal_dense_index(k, m, n, b, j, r, coordinate):
    k = min(k, b * m)
    if j > k:
        coordinate.batch_id = -1
        return

    p = (ceil(r, m) - 1) * k + j

    w = p % b
    w = w if w != 0 else b
    y = ceil(p, b)

    y1 = y % m
    y1 = y1 if y1 != 0 else m
    r1 = r % m
    r1 = r1 if r1 != 0 else m

    x = y1 + r1 - 1
    if x > m:
        x -= m

    if (1 <= w and w <= b) and (1 <= x and x <= m) and (1 <= y and y <= n):
        coordinate.batch_id = w
        coordinate.s1_idx = x
        coordinate.s2_idx = y
    else:
        coordinate.batch_id = -1


def cal_dense_deter_index(round_id, max_loop_num, coordinate_info, const_info, flag):
    j = const_info.core_id_cube + 1
    if flag == True:
        j += 1
    r = round_id + 1
    k = const_info.core_num
    res = -1
    n1 = 0
    if j > k:
        return -1

    b = const_info.b_size * const_info.n2_size
    if const_info.s2_outer == 1:
        cal_dense_index_for_single_n(
            k, const_info.s1_outer, b, j, r, max_loop_num, coordinate_info
        )
    else:
        cal_dense_index(
            k, const_info.s1_outer, const_info.s2_outer, b, j, r, coordinate_info
        )

    w = coordinate_info.batch_id
    n1 = const_info.g_size * const_info.n2_size
    coordinate_info.batch_id = ceil(w, n1) - 1
    n1_idx = w - coordinate_info.batch_id * n1 - 1
    coordinate_info.n2_idx = n1_idx // const_info.g_size
    coordinate_info.g_idx = n1_idx % const_info.g_size
    coordinate_info.s1_idx -= 1
    coordinate_info.s2_idx -= 1

    if not (
        w > 0
        and coordinate_info.batch_id < const_info.b_size
        and coordinate_info.n2_idx < const_info.n2_size
        and coordinate_info.g_idx < const_info.g_size
        and coordinate_info.s1_idx >= 0
        and coordinate_info.s1_idx < coordinate_info.s1_outer
        and coordinate_info.s2_idx >= 0
        and coordinate_info.s2_idx < coordinate_info.s2_outer
    ):
        return -1
    res = (
        (
            coordinate_info.batch_id * n1
            + coordinate_info.n2_idx * const_info.g_size
            + coordinate_info.g_idx
        )
        * const_info.s1_outer
        * const_info.s2_outer
        + coordinate_info.s2_idx * const_info.s1_outer
        + coordinate_info.s1_idx
    )
    return res


def cal_deter_index(round_id, max_loop_num, coordinate_info, const_info, flag=False):
    next_valid_round_id = max_loop_num
    next_valid_index = -1
    for current_round_id in pl.range(round_id, max_loop_num, 1):
        next_valid_index = cal_dense_deter_index(
            current_round_id, max_loop_num, coordinate_info, const_info, flag
        )
        if next_valid_index >= 0:
            next_valid_round_id = current_round_id
            return next_valid_round_id, next_valid_index
    coordinate_info.batch_id = -1
    next_valid_index = -1
    next_valid_round_id = max_loop_num
    return next_valid_round_id, next_valid_index


def set_run_info(
    run_info,
    last_run_info,
    task_id,
    coordinate_info,
    next_coordinate_info,
    next_core_first_block_coordinate_info,
    const_info,
    tensor_info,
    tmp_info,
):
    run_info.is_key_reuse = (
        (next_coordinate_info.batch_id == coordinate_info.batch_id)
        and (next_coordinate_info.n2_idx == coordinate_info.n2_idx)
        and (next_coordinate_info.s2_idx == coordinate_info.s2_idx)
        or (next_coordinate_info.batch_id == -1)
    )
    run_info.is_last_process_block = next_coordinate_info.batch_id == -1
    run_info.is_first_process_block = task_id == 0
    last_run_info.is_next_key_reuse = run_info.is_key_reuse
    run_info.is_value_reuse = (
        tmp_info.last_s2_idx == coordinate_info.s2_idx
        and tmp_info.last_batch_idx == coordinate_info.batch_id
        and tmp_info.last_n2_idx == coordinate_info.n2_idx
    )
    tmp_info.last_batch_idx = coordinate_info.batch_id
    tmp_info.last_n2_idx = coordinate_info.n2_idx
    tmp_info.last_s2_idx = coordinate_info.s2_idx

    run_info.bo_idx = coordinate_info.batch_id
    run_info.n2o_idx = coordinate_info.n2_idx
    run_info.n1o_idx = run_info.n2o_idx * const_info.g_size
    run_info.go_idx = coordinate_info.g_idx
    run_info.s2o_idx = coordinate_info.s2_idx
    run_info.s1o_idx = coordinate_info.s1_idx
    run_info.s2_cv_begin = run_info.s2o_idx * CUBE_BASEN
    run_info.batch_id = run_info.bo_idx

    run_info.task_id = task_id
    run_info.task_id_mod2 = task_id % 2
    run_info.s1_real_size = (
        const_info.s1_tail
        if run_info.s1o_idx == const_info.s1_outer - 1
        else CUBE_BASEM
    )
    run_info.s2_real_size = (
        const_info.s2_tail
        if run_info.s2o_idx == const_info.s2_outer - 1
        else CUBE_BASEN
    )

    run_info.inner_s1_loop_num = ceil(run_info.s1_real_size, 128)
    run_info.inner_s2_loop_num = ceil(run_info.s2_real_size, 128)
    with pl.section_vector():
        run_info.deq_scale_q_value = pl.getval(tensor_info.tensor_q_descale, 0)
        run_info.deq_scale_k_value = pl.getval(tensor_info.tensor_k_descale, 0)
        run_info.deq_scale_v_value = pl.getval(tensor_info.tensor_v_descale, 0)
        run_info.deq_scale_do_value = pl.getval(tensor_info.tensor_do_descale, 0)

        run_info.kv_need_atomic = (
            const_info.s2_outer != 1 and run_info.is_value_reuse
        ) or (
            const_info.s2_outer == 1
            and (
                not run_info.is_first_process_block
                and (
                    (
                        coordinate_info.batch_id
                        == next_core_first_block_coordinate_info.batch_id
                        and coordinate_info.n2_idx
                        == next_core_first_block_coordinate_info.n2_idx
                    )
                    or run_info.is_value_reuse
                )
            )
        )

    inner_s1_tail_size = (
        128 if run_info.s1_real_size % 128 == 0 else run_info.s1_real_size % 128
    )
    inner_s2_tail_size = (
        128 if run_info.s2_real_size % 128 == 0 else run_info.s2_real_size % 128
    )

    for i in pl.range(4):
        run_info.inner_s1_real_size[i] = (
            inner_s1_tail_size if i == run_info.inner_s1_loop_num - 1 else 128
        )
        run_info.inner_s2_real_size[i] = (
            inner_s2_tail_size if i == run_info.inner_s2_loop_num - 1 else 128
        )


def set_quant_run_info(run_info, s1_idx, s2_idx):
    run_info.s1_idx = s1_idx
    run_info.s2_idx = s2_idx


def iterate_mm_ds_p(mm1_res, mm2_res, const_info, run_info, tensor_info):
    real_m = run_info.inner_s2_real_size[run_info.s2_idx]
    real_n = run_info.inner_s1_real_size[run_info.s1_idx]

    q_l1 = tensor_info.common_l1.next()
    do_l1 = tensor_info.common_l1.next()
    tensor_info.common_l1_db.next()
    k_l1 = tensor_info.k_l1[run_info.s2_idx]
    v_l1 = tensor_info.v_l1[run_info.s2_idx]

    left_first = tensor_info.left_four.next()
    left_second = tensor_info.left_four.next()
    tensor_info.left_db.next()

    right_first = tensor_info.right_four.next()
    right_second = tensor_info.right_four.next()
    tensor_info.right_db.next()

    mm1_acc = tensor_info.acc_mm1.current()
    mm2_acc = tensor_info.acc_mm2.current()

    pl.set_validshape(q_l1, [TD, real_n])
    pl.set_validshape(do_l1, [TD, real_n])
    if layout == 0:  # BSND
        pl.load(
            q_l1,
            tensor_info.tensor_q,
            [
                run_info.batch_id,
                run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                run_info.n1o_idx,
                0,
            ],
            order=[3, 1],
        )
        pl.load(
            do_l1,
            tensor_info.tensor_do,
            [
                run_info.batch_id,
                run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                run_info.n1o_idx,
                0,
            ],
            order=[3, 1],
        )
    elif layout == 1:  # BNSD
        pl.load(
            q_l1,
            tensor_info.tensor_q,
            [
                run_info.batch_id,
                run_info.n1o_idx,
                run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                0,
            ],
            order=[3, 2],
        )
        pl.load(
            do_l1,
            tensor_info.tensor_do,
            [
                run_info.batch_id,
                run_info.n1o_idx,
                run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                0,
            ],
            order=[3, 2],
        )
    if not run_info.is_value_reuse and run_info.s1_idx == 0:
        pl.set_validshape(k_l1, [real_m, TD])
        pl.set_validshape(v_l1, [real_m, TD])
        if layout == 0:  # BSND
            pl.load(
                k_l1,
                tensor_info.tensor_k,
                [
                    run_info.batch_id,
                    run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                    run_info.n2o_idx,
                    0,
                ],
                order=[1, 3],
            )
            pl.load(
                v_l1,
                tensor_info.tensor_v,
                [
                    run_info.batch_id,
                    run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                    run_info.n2o_idx,
                    0,
                ],
                order=[1, 3],
            )
        elif layout == 1:  # BNSD
            pl.load(
                k_l1,
                tensor_info.tensor_k,
                [
                    run_info.batch_id,
                    run_info.n2o_idx,
                    run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                    0,
                ],
                order=[2, 3],
            )
            pl.load(
                v_l1,
                tensor_info.tensor_v,
                [
                    run_info.batch_id,
                    run_info.n2o_idx,
                    run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                    0,
                ],
                order=[2, 3],
            )

    pl.set_validshape(left_first, [real_m, TD])
    pl.set_validshape(left_second, [real_m, TD])
    pl.move(left_first, k_l1, [0, 0])
    pl.move(left_second, v_l1, [0, 0])

    pl.set_validshape(right_first, [TD, real_n])
    pl.set_validshape(right_second, [TD, real_n])
    pl.move(right_first, q_l1, [0, 0])
    pl.move(right_second, do_l1, [0, 0])
    pl.set_validshape(mm1_acc, [real_m, real_n])
    pl.set_validshape(mm2_acc, [real_m, real_n])
    pl.matmul(mm1_acc, left_first, right_first)
    pl.matmul(mm2_acc, left_second, right_second)

    pl.set_validshape(mm1_acc, [real_m, _align_up(real_n, 64)])
    pl.set_validshape(mm2_acc, [real_m, _align_up(real_n, 64)])
    pl.move(mm1_res, mm1_acc, [0, 0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    pl.move(mm2_res, mm2_acc, [0, 0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)


@pl.vector_function
def compute_p_ds_vf(
    sp_tile,
    dpds_tile,
    sp_tile_bit8,
    dpds_tile_bit8,
    perm_tile,
    lse_tile,
    d_tile,
    src_m,
    src_n,
    ss,
    dps,
    deq_p_scale,
    ds_scale,
):
    dscale_neg = -1.0 * ds_scale
    un_roll_num = 8
    preg_all = vf.update_mask(src_n, dtype=pl.DT_FP32)
    preg_all_8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    preg_all_16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)

    vreg_d = vf.load_align(d_tile, 0)
    vreg_perm = vf.load_align(perm_tile, 0)
    vreg_lse = vf.load_align(lse_tile, 0)

    vreg_ps = vf.full(deq_p_scale, preg_all, dtype=pl.DT_FP32)
    vreg_ps = vf.log(vreg_ps, preg_all)

    vreg_lse = vf.add(vreg_lse, vreg_ps, preg_all)
    vreg_d = vf.muls(vreg_d, dscale_neg, preg_all)
    vreg_dps = vf.full(dps, preg_all, dtype=pl.DT_FP32)
    vreg_dps = vf.muls(vreg_dps, ds_scale, preg_all)

    for i in pl.range(0, src_m, un_roll_num):
        vreg_sp1 = vf.load_align(sp_tile, i * 128)
        vreg_sp2 = vf.load_align(sp_tile, (i + 1) * 128)
        vreg_sp3 = vf.load_align(sp_tile, (i + 2) * 128)
        vreg_sp4 = vf.load_align(sp_tile, (i + 3) * 128)

        vreg_sp1 = vf.muls(vreg_sp1, ss, preg_all)
        vreg_sp2 = vf.muls(vreg_sp2, ss, preg_all)
        vreg_sp3 = vf.muls(vreg_sp3, ss, preg_all)
        vreg_sp4 = vf.muls(vreg_sp4, ss, preg_all)

        vreg_sp1 = vf.exp_sub(vreg_sp1, vreg_lse, preg_all)
        vreg_sp2 = vf.exp_sub(vreg_sp2, vreg_lse, preg_all)
        vreg_sp3 = vf.exp_sub(vreg_sp3, vreg_lse, preg_all)
        vreg_sp4 = vf.exp_sub(vreg_sp4, vreg_lse, preg_all)

        vreg_p1 = vf.astype(
            vreg_sp1,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_p2 = vf.astype(
            vreg_sp2,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_p3 = vf.astype(
            vreg_sp3,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.TWO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_p4 = vf.astype(
            vreg_sp4,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.TWO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )

        vreg_p13 = vf.or_(
            vf.bit_cast(vreg_p1, dtype=pl.DT_INT8),
            vf.bit_cast(vreg_p3, dtype=pl.DT_INT8),
            preg_all_8,
        )
        vreg_p24 = vf.or_(
            vf.bit_cast(vreg_p2, dtype=pl.DT_INT8),
            vf.bit_cast(vreg_p4, dtype=pl.DT_INT8),
            preg_all_8,
        )
        vreg_p_res1, vreg_p_res2 = vf.interleave(
            vf.bit_cast(vreg_p13, dtype=pl.DT_FP16),
            vf.bit_cast(vreg_p24, dtype=pl.DT_FP16),
        )

        vf.scatter(
            sp_tile_bit8 + ((i + 0) * 512),
            vf.bit_cast(vreg_p_res1, dtype=pl.DT_INT8),
            vreg_perm,
            preg_all_8,
        )
        vf.scatter(
            sp_tile_bit8 + ((i + 1) * 512),
            vf.bit_cast(vreg_p_res2, dtype=pl.DT_INT8),
            vreg_perm,
            preg_all_8,
        )

        vreg_dps1 = vf.load_align(dpds_tile, i * 128)
        vreg_dps2 = vf.load_align(dpds_tile, (i + 1) * 128)
        vreg_dps3 = vf.load_align(dpds_tile, (i + 2) * 128)
        vreg_dps4 = vf.load_align(dpds_tile, (i + 3) * 128)

        vreg_dps1 = vf.mul_dst_add(vreg_dps, vreg_d, preg_all)
        vreg_dps2 = vf.mul_dst_add(vreg_dps, vreg_d, preg_all)
        vreg_dps3 = vf.mul_dst_add(vreg_dps, vreg_d, preg_all)
        vreg_dps4 = vf.mul_dst_add(vreg_dps, vreg_d, preg_all)

        vreg_dps1 = vf.mul(vreg_dps1, vreg_sp1, preg_all)
        vreg_dps2 = vf.mul(vreg_dps2, vreg_sp2, preg_all)
        vreg_dps3 = vf.mul(vreg_dps3, vreg_sp3, preg_all)
        vreg_dps4 = vf.mul(vreg_dps4, vreg_sp4, preg_all)

        vreg_ds1 = vf.astype(
            vreg_dps1,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_ds2 = vf.astype(
            vreg_dps2,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_ds3 = vf.astype(
            vreg_dps3,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.TWO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_ds4 = vf.astype(
            vreg_dps4,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.TWO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )

        vreg_ds13 = vf.or_(
            vf.bit_cast(vreg_ds1, dtype=pl.DT_INT8),
            vf.bit_cast(vreg_ds3, dtype=pl.DT_INT8),
            preg_all_8,
        )
        vreg_ds24 = vf.or_(
            vf.bit_cast(vreg_ds2, dtype=pl.DT_INT8),
            vf.bit_cast(vreg_ds4, dtype=pl.DT_INT8),
            preg_all_8,
        )
        vreg_ds_res1, vreg_ds_res2 = vf.interleave(
            vf.bit_cast(vreg_ds13, dtype=pl.DT_FP16),
            vf.bit_cast(vreg_ds24, dtype=pl.DT_FP16),
        )

        vf.scatter(
            dpds_tile_bit8 + ((i + 0) * 512),
            vf.bit_cast(vreg_ds_res1, dtype=pl.DT_INT8),
            vreg_perm,
            preg_all_8,
        )
        vf.scatter(
            dpds_tile_bit8 + ((i + 1) * 512),
            vf.bit_cast(vreg_ds_res2, dtype=pl.DT_INT8),
            vreg_perm,
            preg_all_8,
        )

    for i in pl.range(0, src_m, un_roll_num):
        vreg_sp1 = vf.load_align(sp_tile, (i + 4) * 128)
        vreg_sp2 = vf.load_align(sp_tile, (i + 5) * 128)
        vreg_sp3 = vf.load_align(sp_tile, (i + 6) * 128)
        vreg_sp4 = vf.load_align(sp_tile, (i + 7) * 128)

        vreg_sp1 = vf.muls(vreg_sp1, ss, preg_all)
        vreg_sp2 = vf.muls(vreg_sp2, ss, preg_all)
        vreg_sp3 = vf.muls(vreg_sp3, ss, preg_all)
        vreg_sp4 = vf.muls(vreg_sp4, ss, preg_all)

        vreg_sp1 = vf.exp_sub(vreg_sp1, vreg_lse, preg_all)
        vreg_sp2 = vf.exp_sub(vreg_sp2, vreg_lse, preg_all)
        vreg_sp3 = vf.exp_sub(vreg_sp3, vreg_lse, preg_all)
        vreg_sp4 = vf.exp_sub(vreg_sp4, vreg_lse, preg_all)

        vreg_p1 = vf.astype(
            vreg_sp1,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_p2 = vf.astype(
            vreg_sp2,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_p3 = vf.astype(
            vreg_sp3,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.TWO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_p4 = vf.astype(
            vreg_sp4,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.TWO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )

        vreg_p13 = vf.or_(
            vf.bit_cast(vreg_p1, dtype=pl.DT_INT8),
            vf.bit_cast(vreg_p3, dtype=pl.DT_INT8),
            preg_all_8,
        )
        vreg_p24 = vf.or_(
            vf.bit_cast(vreg_p2, dtype=pl.DT_INT8),
            vf.bit_cast(vreg_p4, dtype=pl.DT_INT8),
            preg_all_8,
        )
        vreg_p_res1, vreg_p_res2 = vf.interleave(
            vf.bit_cast(vreg_p13, dtype=pl.DT_FP16),
            vf.bit_cast(vreg_p24, dtype=pl.DT_FP16),
        )

        vf.scatter(
            sp_tile_bit8 + ((i + 0) * 512 + 128),
            vf.bit_cast(vreg_p_res1, dtype=pl.DT_INT8),
            vreg_perm,
            preg_all_8,
        )
        vf.scatter(
            sp_tile_bit8 + ((i + 1) * 512 + 128),
            vf.bit_cast(vreg_p_res2, dtype=pl.DT_INT8),
            vreg_perm,
            preg_all_8,
        )

        vreg_dps1 = vf.load_align(dpds_tile, (i + 4) * 128)
        vreg_dps2 = vf.load_align(dpds_tile, (i + 5) * 128)
        vreg_dps3 = vf.load_align(dpds_tile, (i + 6) * 128)
        vreg_dps4 = vf.load_align(dpds_tile, (i + 7) * 128)

        vreg_dps1 = vf.mul_dst_add(vreg_dps, vreg_d, preg_all)
        vreg_dps2 = vf.mul_dst_add(vreg_dps, vreg_d, preg_all)
        vreg_dps3 = vf.mul_dst_add(vreg_dps, vreg_d, preg_all)
        vreg_dps4 = vf.mul_dst_add(vreg_dps, vreg_d, preg_all)

        vreg_dps1 = vf.mul(vreg_dps1, vreg_sp1, preg_all)
        vreg_dps2 = vf.mul(vreg_dps2, vreg_sp2, preg_all)
        vreg_dps3 = vf.mul(vreg_dps3, vreg_sp3, preg_all)
        vreg_dps4 = vf.mul(vreg_dps4, vreg_sp4, preg_all)

        vreg_ds1 = vf.astype(
            vreg_dps1,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_ds2 = vf.astype(
            vreg_dps2,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.ZERO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_ds3 = vf.astype(
            vreg_dps3,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.TWO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )
        vreg_ds4 = vf.astype(
            vreg_dps4,
            preg_all,
            dtype=pl.DT_HF8,
            layout=pl.CastLayout.TWO,
            round_mode=pl.VFRoundMode.CAST_ROUND,
        )

        vreg_ds13 = vf.or_(
            vf.bit_cast(vreg_ds1, dtype=pl.DT_INT8),
            vf.bit_cast(vreg_ds3, dtype=pl.DT_INT8),
            preg_all_8,
        )
        vreg_ds24 = vf.or_(
            vf.bit_cast(vreg_ds2, dtype=pl.DT_INT8),
            vf.bit_cast(vreg_ds4, dtype=pl.DT_INT8),
            preg_all_8,
        )
        vreg_ds_res1, vreg_ds_res2 = vf.interleave(
            vf.bit_cast(vreg_ds13, dtype=pl.DT_FP16),
            vf.bit_cast(vreg_ds24, dtype=pl.DT_FP16),
        )

        vf.scatter(
            dpds_tile_bit8 + ((i + 0) * 512 + 128),
            vf.bit_cast(vreg_ds_res1, dtype=pl.DT_INT8),
            vreg_perm,
            preg_all_8,
        )
        vf.scatter(
            dpds_tile_bit8 + ((i + 1) * 512 + 128),
            vf.bit_cast(vreg_ds_res2, dtype=pl.DT_INT8),
            vreg_perm,
            preg_all_8,
        )


def compute_p_ds(
    sp_tile,
    dpds_tile,
    sp_tile_bit8,
    dpds_tile_bit8,
    perm_tile,
    lse_tile,
    d_tile,
    zero_tile,
    src_m,
    src_n,
    ss,
    dps,
    deq_p_scale,
    ds_scale,
):
    for i in pl.range(src_m, ceil(src_m, 8) * 8):
        pl.insert(sp_tile, zero_tile, offset=[i, 0])
        pl.move(dpds_tile, zero_tile, offset=[i, 0])
    compute_p_ds_vf(
        sp_tile,
        dpds_tile,
        sp_tile_bit8,
        dpds_tile_bit8,
        perm_tile,
        lse_tile,
        d_tile,
        src_m,
        src_n,
        ss,
        dps,
        deq_p_scale,
        ds_scale,
    )


def iterate_p_ds(p_l1_bit8, ds_l1_bit8, sdp_id, const_info, run_info, tensor_info):
    real_s1 = run_info.inner_s1_real_size[run_info.s1_idx]
    real_s2 = run_info.inner_s2_real_size[run_info.s2_idx]
    first_half_s1 = ceil(real_s1, QUANT_S1_BASE_COUNT) * QUANT_S1_BASE_COUNT // 2
    current_real_s1 = (
        first_half_s1 if const_info.sub_id == 0 else real_s1 - first_half_s1
    )
    current_real_s2 = ceil(real_s2, QUANT_S2_BASE_COUNT) * QUANT_S2_BASE_COUNT

    lse_vec = tensor_info.lse_vec.next()
    d_vec = tensor_info.d_vec.next()

    if current_real_s1 <= 0:
        return

    perm_vec = tensor_info.perm_vec.current()
    zero_vec = tensor_info.zero_vec.current()
    pl.set_validshape(lse_vec, [1, current_real_s1])
    pl.set_validshape(d_vec, [1, current_real_s1])
    if layout == 0 or layout == 1:
        pl.load(
            lse_vec,
            tensor_info.tensor_softmax_lse,
            [
                run_info.batch_id,
                run_info.n1o_idx,
                run_info.s1o_idx * CUBE_BASEM
                + run_info.s1_idx * TS
                + const_info.sub_id * first_half_s1,
            ],
            order=[1, 2],
        )
        pl.load(
            d_vec,
            tensor_info.tensor_workspace_sfmg,
            [
                run_info.batch_id,
                run_info.n1o_idx,
                run_info.s1o_idx * CUBE_BASEM
                + run_info.s1_idx * TS
                + const_info.sub_id * first_half_s1,
            ],
            order=[1, 2],
        )

    sp_vec = tensor_info.sp_vec[sdp_id]
    sp_vec_bit8 = tensor_info.sp_vec_bit8[sdp_id]
    dpds_vec = tensor_info.dpds_vec[sdp_id]
    dpds_vec_bit8 = tensor_info.dpds_vec_bit8[sdp_id]
    compute_p_ds(
        sp_vec,
        dpds_vec,
        sp_vec_bit8,
        dpds_vec_bit8,
        perm_vec,
        lse_vec,
        d_vec,
        zero_vec,
        current_real_s2,
        current_real_s1,
        run_info.deq_scale_q_value
        * run_info.deq_scale_k_value
        * const_info.softmax_scale,
        run_info.deq_scale_v_value * run_info.deq_scale_do_value,
        const_info.deq_scale_p_value,
        const_info.deq_scale_p_value * const_info.scale_ds,
    )
    if first_half_s1 > BLOCK_SIZE:
        pl.insert(p_l1_bit8, sp_vec_bit8[:, :256], [const_info.sub_id * 32, 0])
        pl.insert(p_l1_bit8, sp_vec_bit8[:, 512:768], [const_info.sub_id * 32 + 16, 0])

        pl.insert(ds_l1_bit8, dpds_vec_bit8[:, :256], [const_info.sub_id * 32, 0])
        pl.insert(
            ds_l1_bit8, dpds_vec_bit8[:, 512:768], [const_info.sub_id * 32 + 16, 0]
        )
    else:
        pl.insert(p_l1_bit8, sp_vec_bit8[:, :256], [const_info.sub_id * 16, 0])
        pl.insert(ds_l1_bit8, dpds_vec_bit8[:, :256], [const_info.sub_id * 16, 0])


def iterate_mm_ds_k(
    ds_l1_tensor0, ds_l1_tensor1, const_info, run_info, tensor_info, tmp_info
):
    is_tail_k = (
        run_info.s2_idx * BASE_K < run_info.s2_real_size
        and (run_info.s2_idx + 2) * BASE_K > run_info.s2_real_size
    )
    real_m = run_info.inner_s1_real_size[run_info.s1_idx]
    real_k = (run_info.s2_real_size % K_SIZE) if is_tail_k else K_SIZE
    k_size_first = BASE_K
    k_size_second = real_k - BASE_K
    if real_k < BASE_K:
        k_size_first = real_k
        k_size_second = 0
    left_first = tensor_info.left_four.next()
    left_second = tensor_info.left_four.next()
    left = tensor_info.left_db.next()
    right_first = tensor_info.right_four.next()
    right_second = tensor_info.right_four.next()
    right = tensor_info.right_db.next()

    mm_acc = tensor_info.acc_db[tmp_info.l0c_buffer_id]

    pl.set_validshape(left_first, [128, 128])
    pl.set_validshape(left_second, [128, 128])
    pl.move(left_first, ds_l1_tensor0, [0, 0])
    pl.move(left_second, ds_l1_tensor1, [0, 0])
    if run_info.is_key_reuse:
        k_l1_first = tensor_info.k_l1[run_info.s2_idx]
        k_l1_second = tensor_info.k_l1[run_info.s2_idx + 1]
    else:
        k_l1_first = tensor_info.common_l1.next()
        k_l1_second = tensor_info.common_l1.next()
        tensor_info.common_l1_db.next()

        pl.set_validshape(k_l1_first, [k_size_first, D_SIZE])
        pl.set_validshape(k_l1_second, [k_size_second, D_SIZE])
        if layout == 0:  # BSND
            pl.load(
                k_l1_first,
                tensor_info.tensor_k,
                [
                    run_info.batch_id,
                    run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                    run_info.n2o_idx,
                    0,
                ],
                order=[1, 3],
            )
            pl.load(
                k_l1_second,
                tensor_info.tensor_k,
                [
                    run_info.batch_id,
                    run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS + TS,
                    run_info.n2o_idx,
                    0,
                ],
                order=[1, 3],
            )
        elif layout == 1:  # BNSD
            pl.load(
                k_l1_first,
                tensor_info.tensor_k,
                [
                    run_info.batch_id,
                    run_info.n2o_idx,
                    run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                    0,
                ],
                order=[2, 3],
            )
            pl.load(
                k_l1_second,
                tensor_info.tensor_k,
                [
                    run_info.batch_id,
                    run_info.n2o_idx,
                    run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS + TS,
                    0,
                ],
                order=[2, 3],
            )
    pl.set_validshape(right_first, [k_size_first, D_SIZE])
    pl.set_validshape(right_second, [k_size_second, D_SIZE])
    pl.move(right_first, k_l1_first, [0, 0])
    pl.move(right_second, k_l1_second, [0, 0])
    pl.set_validshape(left, [128, real_k])
    pl.set_validshape(right, [real_k, D_SIZE])
    if run_info.is_dq_fix_out:
        pl.matmul_acc(mm_acc, mm_acc, left, right)
    else:
        pl.matmul(mm_acc, left, right)


def iterate_mm_p_dy(
    p_l1_tensor0, p_l1_tensor1, const_info, run_info, tensor_info, tmp_info
):
    is_tail_k = (run_info.s1_idx * BASE_K < run_info.s1_real_size) and (
        (run_info.s1_idx + 2) * BASE_K > run_info.s1_real_size
    )
    real_m = run_info.inner_s2_real_size[run_info.s2_idx]
    real_k = (run_info.s1_real_size % K_SIZE) if is_tail_k else K_SIZE

    do_l1 = tensor_info.common_l1_db.next()
    tensor_info.common_l1.next()
    tensor_info.common_l1.next()

    left_first = tensor_info.left_four.next()
    left_second = tensor_info.left_four.next()
    left = tensor_info.left_db.next()

    right_first = tensor_info.right_four.next()
    right_second = tensor_info.right_four.next()
    right = tensor_info.right_db.next()

    acc = tensor_info.acc_db[tmp_info.l0c_buffer_id]
    pl.set_validshape(left_first, [128, 128])  #
    pl.set_validshape(left_second, [128, 128])  #
    pl.move(left_first, p_l1_tensor0, [0, 0])
    pl.move(left_second, p_l1_tensor1, [0, 0])
    pl.set_validshape(do_l1, [real_k, D_SIZE])
    if layout == 0:  # BSND
        pl.load(
            do_l1,
            tensor_info.tensor_do_p_dy,
            [
                run_info.batch_id,
                run_info.s1o_idx * CUBE_BASEN + run_info.s1_idx * TS,
                run_info.n1o_idx,
                0,
            ],
            order=[1, 3],
        )
    elif layout == 1:  # BNSD
        pl.load(
            do_l1,
            tensor_info.tensor_do_p_dy,
            [
                run_info.batch_id,
                run_info.n1o_idx,
                run_info.s1o_idx * CUBE_BASEN + run_info.s1_idx * TS,
                0,
            ],
            order=[2, 3],
        )

    pl.set_validshape(right, [real_k, D_SIZE])
    pl.move(right, do_l1, [0, 0])

    pl.set_validshape(left, [128, real_k])
    pl.set_validshape(right, [real_k, D_SIZE])
    if run_info.is_dv_fix_out:
        pl.matmul_acc(acc, acc, left, right)
    else:
        pl.matmul(acc, left, right)


def iterate_mm_ds_q(
    ds_l1_tensor0, ds_l1_tensor1, const_info, run_info, tensor_info, tmp_info
):
    is_tail_k = (run_info.s1_idx * BASE_K < run_info.s1_real_size) and (
        (run_info.s1_idx + 2) * BASE_K > run_info.s1_real_size
    )
    real_m = run_info.inner_s2_real_size[run_info.s2_idx]
    real_k = (run_info.s1_real_size % K_SIZE) if is_tail_k else K_SIZE
    acc_id = (tmp_info.l0c_buffer_id + 1) % 2

    q_l1 = tensor_info.common_l1_db.next()
    tensor_info.common_l1.next()
    tensor_info.common_l1.next()

    left_first = tensor_info.left_four.next()
    left_second = tensor_info.left_four.next()
    left = tensor_info.left_db.next()

    right = tensor_info.right_db.next()
    right_first = tensor_info.right_four.next()
    right_second = tensor_info.right_four.next()

    acc = tensor_info.acc_db[acc_id]
    pl.set_validshape(left_first, [128, 128])
    pl.set_validshape(left_second, [128, 128])
    pl.move(left_first, ds_l1_tensor0, [0, 0])
    pl.move(left_second, ds_l1_tensor1, [0, 0])
    pl.set_validshape(q_l1, [real_k, D_SIZE])
    if layout == 0:  # BSND
        pl.load(
            q_l1,
            tensor_info.tensor_q_ds,
            [
                run_info.batch_id,
                run_info.s1o_idx * CUBE_BASEN + run_info.s1_idx * TS,
                run_info.n1o_idx,
                0,
            ],
            order=[1, 3],
        )
    elif layout == 1:  # BNSD
        pl.load(
            q_l1,
            tensor_info.tensor_q_ds,
            [
                run_info.batch_id,
                run_info.n1o_idx,
                run_info.s1o_idx * CUBE_BASEN + run_info.s1_idx * TS,
                0,
            ],
            order=[2, 3],
        )

    pl.set_validshape(right, [real_k, D_SIZE])
    pl.move(right, q_l1, [0, 0])

    pl.set_validshape(left, [128, real_k])
    pl.set_validshape(right, [real_k, D_SIZE])
    if run_info.is_dk_fix_out:
        pl.matmul_acc(acc, acc, left, right)
    else:
        pl.matmul(acc, left, right)


def copy_out_dq_result(tensor_info, const_info, tmp_info):
    acc_dq = tensor_info.acc_db[tmp_info.l0c_buffer_id]
    dq_vec = tensor_info.dq_vec.current()

    pl.move(dq_vec, acc_dq, [0, 0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    tmp_info.l0c_buffer_id = (tmp_info.l0c_buffer_id + 1) % 2


def copy_out_dkv_result(tensor_info, const_info, tmp_info):
    acc_dv = tensor_info.acc_db[tmp_info.l0c_buffer_id]
    acc_dk = tensor_info.acc_db[1 - tmp_info.l0c_buffer_id]

    dv_vec = tensor_info.dkv_vec[0]
    dk_vec = tensor_info.dkv_vec[1]
    pl.move(dv_vec, acc_dv, [0, 0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)
    pl.move(dk_vec, acc_dk, [0, 0], acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN)


def copy_dqkv2gm(is_kv, run_info, const_info, tensor_info):
    if is_kv:
        block_count = run_info.inner_s2_real_size[run_info.s2_idx]
        workspace_dk = tensor_info.tensor_workspace_dk
        workspace_dv = tensor_info.tensor_workspace_dv
        dk_vec = tensor_info.dkv_vec[1]
        dv_vec = tensor_info.dkv_vec[0]
        pl.set_validshape(dk_vec, [block_count, HALF_D_SIZE])
        pl.set_validshape(dv_vec, [block_count, HALF_D_SIZE])
        if layout == 0:  # BSND
            if run_info.kv_need_atomic:
                pl.store(
                    workspace_dk,
                    dk_vec,
                    [
                        run_info.batch_id,
                        run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                        run_info.n2o_idx,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                )
                pl.store(
                    workspace_dv,
                    dv_vec,
                    [
                        run_info.batch_id,
                        run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                        run_info.n2o_idx,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                )
            else:
                pl.store(
                    workspace_dk,
                    dk_vec,
                    [
                        run_info.batch_id,
                        run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                        run_info.n2o_idx,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[1, 3],
                )
                pl.store(
                    workspace_dv,
                    dv_vec,
                    [
                        run_info.batch_id,
                        run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                        run_info.n2o_idx,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[1, 3],
                )
        if layout == 1:  # BNSD
            if run_info.kv_need_atomic:
                pl.store(
                    workspace_dk,
                    dk_vec,
                    [
                        run_info.batch_id,
                        run_info.n2o_idx,
                        run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[2, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                )
                pl.store(
                    workspace_dv,
                    dv_vec,
                    [
                        run_info.batch_id,
                        run_info.n2o_idx,
                        run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[2, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                )
            else:
                pl.store(
                    workspace_dk,
                    dk_vec,
                    [
                        run_info.batch_id,
                        run_info.n2o_idx,
                        run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[2, 3],
                )
                pl.store(
                    workspace_dv,
                    dv_vec,
                    [
                        run_info.batch_id,
                        run_info.n2o_idx,
                        run_info.s2o_idx * CUBE_BASEN + run_info.s2_idx * TS,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[2, 3],
                )
    else:
        block_count = run_info.inner_s1_real_size[run_info.s1_idx]
        workspace_dq = tensor_info.tensor_workspace_dq
        dq_vec = tensor_info.dq_vec.current()

        pl.set_validshape(dq_vec, [block_count, HALF_D_SIZE])

        if layout == 0:  # BSND
            if const_info.s2_outer != 1:
                pl.store(
                    workspace_dq,
                    dq_vec,
                    [
                        run_info.batch_id,
                        run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                        run_info.n1o_idx,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                )
            else:
                pl.store(
                    workspace_dq,
                    dq_vec,
                    [
                        run_info.batch_id,
                        run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                        run_info.n1o_idx,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[1, 3],
                )
        if layout == 1:  # BNSD
            if const_info.s2_outer != 1:
                pl.store(
                    workspace_dq,
                    dq_vec,
                    [
                        run_info.batch_id,
                        run_info.n1o_idx,
                        run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[2, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                )
            else:
                pl.store(
                    workspace_dq,
                    dq_vec,
                    [
                        run_info.batch_id,
                        run_info.n1o_idx,
                        run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[2, 3],
                )


@pl.vector_function
def dequant_out(tensor, real_length, scale):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(real_length):
        vreg_x1 = vf.load_align(tensor, i * D_SIZE * 2)
        vreg_x2 = vf.load_align(tensor, i * D_SIZE * 2 + D_SIZE)

        vreg_x1 = vf.muls(vreg_x1, scale, preg_all)
        vreg_x2 = vf.muls(vreg_x2, scale, preg_all)

        vf.store_align(tensor + i * D_SIZE * 2, vreg_x1, preg_all)
        vf.store_align(tensor + i * D_SIZE * 2 + D_SIZE, vreg_x2, preg_all)


def dequant_dqkv(mode, tensor_1, tensor_2, const_info, run_info):
    if mode == 0:
        scale = run_info.deq_scale_k_value * const_info.deq_scale_ds_value
        real_length = run_info.inner_s1_real_size[run_info.s1_idx]
    else:
        scale = run_info.deq_scale_do_value * const_info.deq_scale_p_value
        real_length = run_info.inner_s2_real_size[run_info.s2_idx]
    dequant_out(tensor_1, ceil(real_length, 2), scale)
    if mode != 0:
        scale = run_info.deq_scale_q_value * const_info.deq_scale_ds_value
        dequant_out(tensor_2, ceil(real_length, 2), scale)


def process_sp(run_info, const_info, sdp_id, tensor_info):
    with pl.section_cube():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=sdp_id,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        if (
            run_info.s2_idx < run_info.inner_s2_loop_num
            and run_info.s1_idx < run_info.inner_s1_loop_num
        ):
            iterate_mm_ds_p(
                tensor_info.sp_vec[sdp_id],
                tensor_info.dpds_vec[sdp_id],
                const_info,
                run_info,
                tensor_info,
            )
        pl.system.set_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=sdp_id + 5,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )


def process_p_ds(run_info, const_info, sdp_id, tensor_info, tmp_info):
    with pl.section_vector():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.V,
            event_id=sdp_id + 5,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        if (sdp_id % 2) == 0:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_UB2L1_P_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_UB2L1_DS_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
        p_l1_bit8_tmp = tensor_info.p_l1_bit8
        p_l1_bit8 = p_l1_bit8_tmp.next()
        if (
            run_info.s2_idx < run_info.inner_s2_loop_num
            and run_info.s1_idx < run_info.inner_s1_loop_num
        ):
            ds_l1_bit8 = tensor_info.ds_l1_bit8[
                l1_ds(run_info.s1_idx, run_info.s2_idx, run_info.task_id_mod2)
            ]
            iterate_p_ds(
                p_l1_bit8, ds_l1_bit8, sdp_id, const_info, run_info, tensor_info
            )

        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=sdp_id,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        if sdp_id % 2 == 1:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_PDS_TO_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
        if run_info.s2_idx == 3:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_PDS_TO_DKV_FLAG_TAIL,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )


def process_dq(run_info, sdp_id, const_info, tensor_info, tmp_info):
    with pl.section_cube():
        if run_info.s2_idx == 2:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1,
                event_id=SYNC_PDS_TO_DKV_FLAG_TAIL,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )

        ds_l1_tensor0 = tensor_info.ds_l1_t[
            l1_ds(run_info.s1_idx, run_info.s2_idx, run_info.task_id_mod2)
        ]
        ds_l1_tensor1 = tensor_info.ds_l1_t[
            l1_ds(run_info.s1_idx, run_info.s2_idx + 1, run_info.task_id_mod2)
        ]

        run_info.is_dq_fix_out = sdp_id == 2

        if (
            run_info.s2_idx < run_info.inner_s2_loop_num
            and run_info.s1_idx < run_info.inner_s1_loop_num
        ):
            iterate_mm_ds_k(
                ds_l1_tensor0,
                ds_l1_tensor1,
                const_info,
                run_info,
                tensor_info,
                tmp_info,
            )

        if not (sdp_id == 2 and run_info.s1_idx == 0):
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1,
                event_id=SYNC_UB2L1_DS_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )

        if sdp_id == 2:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.FIX,
                event_id=SYNC_TRANSFER_DQ_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
            if run_info.s1_idx < run_info.inner_s1_loop_num:
                copy_out_dq_result(tensor_info, const_info, tmp_info)
            pl.system.set_cross_core(
                pipe=pl.PipeType.FIX,
                event_id=SYNC_COMPUTE_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )


def compute_dq(run_info, const_info, tensor_info):
    with pl.section_vector():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.V,
            event_id=SYNC_COMPUTE_DKV_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        if run_info.s1_idx < run_info.inner_s1_loop_num:
            dequant_dqkv(
                0,
                tensor_info.dq_vec.current(),
                tensor_info.dq_vec.current(),
                const_info,
                run_info,
            )


def process_dkv(is_dk, run_info, sdp_id, const_info, tensor_info, tmp_info):
    with pl.section_cube():
        if not is_dk:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1,
                event_id=SYNC_PDS_TO_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
            p_l1_tensor0 = tensor_info.p_l1.next()
            p_l1_tensor1 = tensor_info.p_l1.next()
            if (
                run_info.s2_idx < run_info.inner_s2_loop_num
                and run_info.s1_idx < run_info.inner_s1_loop_num
            ):
                run_info.is_dv_fix_out = sdp_id == 2
                iterate_mm_p_dy(
                    p_l1_tensor0,
                    p_l1_tensor1,
                    const_info,
                    run_info,
                    tensor_info,
                    tmp_info,
                )
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1,
                event_id=SYNC_UB2L1_P_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
        else:
            if (
                run_info.s2_idx < run_info.inner_s2_loop_num
                and run_info.s1_idx < run_info.inner_s1_loop_num
            ):
                run_info.is_dk_fix_out = sdp_id == 2
                ds_l1_tensor0 = tensor_info.ds_l1[
                    l1_ds(run_info.s1_idx, run_info.s2_idx, run_info.task_id_mod2)
                ]
                ds_l1_tensor1 = tensor_info.ds_l1[
                    l1_ds(run_info.s1_idx + 1, run_info.s2_idx, run_info.task_id_mod2)
                ]
                iterate_mm_ds_q(
                    ds_l1_tensor0,
                    ds_l1_tensor1,
                    const_info,
                    run_info,
                    tensor_info,
                    tmp_info,
                )

            if sdp_id == 0 and run_info.s2_idx == 3:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE1,
                    event_id=SYNC_UB2L1_DS_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                )

            if sdp_id == 2:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_TRANSFER_DKV_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                )
                if run_info.s2_idx < run_info.inner_s2_loop_num:
                    copy_out_dkv_result(tensor_info, const_info, tmp_info)

                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_COMPUTE_DKV_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                )
    with pl.section_vector():
        if sdp_id == 2 and is_dk:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.V,
                event_id=SYNC_COMPUTE_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
            if run_info.s2_idx < run_info.inner_s2_loop_num:
                dequant_dqkv(
                    1,
                    tensor_info.dkv_vec[0],
                    tensor_info.dkv_vec[1],
                    const_info,
                    run_info,
                )


def process_first_s2(
    const_info, run_info, last_run_info, kv_inner_id, tensor_info, tmp_info
):
    set_quant_run_info(run_info, 0, kv_inner_id)
    process_sp(run_info, const_info, 0, tensor_info)
    process_p_ds(run_info, const_info, 0, tensor_info, tmp_info)
    with pl.section_cube():
        if not last_run_info.is_dq_completed:
            set_quant_run_info(last_run_info, kv_inner_id, 0)
            process_dq(last_run_info, 0, const_info, tensor_info, tmp_info)

    set_quant_run_info(run_info, 1, kv_inner_id)
    process_sp(run_info, const_info, 1, tensor_info)
    process_p_ds(run_info, const_info, 1, tensor_info, tmp_info)
    with pl.section_cube():
        if not last_run_info.is_dq_completed:
            set_quant_run_info(last_run_info, kv_inner_id, 2)
            process_dq(last_run_info, 2, const_info, tensor_info, tmp_info)

    with pl.section_vector():
        if kv_inner_id > 1:
            set_quant_run_info(run_info, 0, kv_inner_id - 2)
            if run_info.s2_idx < run_info.inner_s2_loop_num:
                copy_dqkv2gm(True, run_info, const_info, tensor_info)
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_TRANSFER_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
        elif not last_run_info.is_dkv_completed:
            set_quant_run_info(last_run_info, 0, kv_inner_id + 2)
            if last_run_info.s2_idx < last_run_info.inner_s2_loop_num:
                copy_dqkv2gm(True, last_run_info, const_info, tensor_info)
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_TRANSFER_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
            last_run_info.is_dkv_completed = (kv_inner_id + 2) == 3
        if not last_run_info.is_dq_completed:
            set_quant_run_info(last_run_info, kv_inner_id, 2)
            compute_dq(last_run_info, const_info, tensor_info)

    if kv_inner_id > 0:
        set_quant_run_info(run_info, 0, kv_inner_id - 1)
        process_dkv(False, run_info, 0, const_info, tensor_info, tmp_info)
    elif not last_run_info.is_dkv_completed and kv_inner_id == 0:
        set_quant_run_info(last_run_info, 0, 3)
        process_dkv(False, last_run_info, 0, const_info, tensor_info, tmp_info)

    set_quant_run_info(run_info, 2, kv_inner_id)
    process_sp(run_info, const_info, 0, tensor_info)
    process_p_ds(run_info, const_info, 0, tensor_info, tmp_info)
    if kv_inner_id > 0:
        set_quant_run_info(run_info, 2, kv_inner_id - 1)
        process_dkv(False, run_info, 2, const_info, tensor_info, tmp_info)
    elif not last_run_info.is_dkv_completed and kv_inner_id == 0:
        set_quant_run_info(last_run_info, 2, 3)
        process_dkv(False, last_run_info, 2, const_info, tensor_info, tmp_info)

    set_quant_run_info(run_info, 3, kv_inner_id)
    process_sp(run_info, const_info, 1, tensor_info)
    process_p_ds(run_info, const_info, 1, tensor_info, tmp_info)

    with pl.section_vector():
        if not last_run_info.is_dq_completed:
            set_quant_run_info(last_run_info, kv_inner_id, 2)
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_DETER_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
            )

            if last_run_info.s1_idx < last_run_info.inner_s1_loop_num:
                copy_dqkv2gm(False, last_run_info, const_info, tensor_info)
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_DETER_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
            )
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_TRANSFER_DQ_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )
            last_run_info.is_dq_completed = kv_inner_id == 3

    if kv_inner_id > 0:
        set_quant_run_info(run_info, 0, kv_inner_id - 1)
        process_dkv(True, run_info, 0, const_info, tensor_info, tmp_info)
        set_quant_run_info(run_info, 2, kv_inner_id - 1)
        process_dkv(True, run_info, 2, const_info, tensor_info, tmp_info)
    elif not last_run_info.is_dkv_completed and kv_inner_id == 0:
        set_quant_run_info(last_run_info, 0, 3)
        process_dkv(True, last_run_info, 0, const_info, tensor_info, tmp_info)
        set_quant_run_info(last_run_info, 2, 3)
        process_dkv(True, last_run_info, 2, const_info, tensor_info, tmp_info)


def process_post_dqkv(const_info, tensor_info):
    with pl.section_vector():
        vec_blk = const_info.core_id_vec
        in_ping_tmp = tensor_info.post_in
        in_ping = in_ping_tmp.next()

        in_pong_tmp = tensor_info.post_in
        in_pong = in_pong_tmp.next()

        out_ping_tmp = tensor_info.post_out
        out_ping = out_ping_tmp.next()

        out_pong_tmp = tensor_info.post_out
        out_pong = out_pong_tmp.next()

        for qkv in pl.range(3):
            factor = const_info.q_post_block_factor
            total = const_info.q_post_block_total
            tail = const_info.q_post_tail_num

            if qkv == 1:
                factor = const_info.k_post_block_factor
                total = const_info.k_post_block_total
                tail = const_info.k_post_tail_num
            elif qkv == 2:
                factor = const_info.v_post_block_factor
                total = const_info.v_post_block_total
                tail = const_info.v_post_tail_num

            block_core = factor * TS * TS
            begin = vec_blk * block_core
            end = begin + block_core

            if end > total:
                end = total
            if begin < total:
                idx = begin
                while True:
                    if idx >= end:
                        break
                    pong_off = idx + TS * TS
                    ping_size = TS * TS if pong_off < total else tail
                    pl.set_validshape(in_ping, [1, ping_size])

                    if qkv == 0:
                        pl.load(in_ping, tensor_info.tensor_workspace_dq_flat, [0, idx])
                    elif qkv == 1:
                        pl.load(in_ping, tensor_info.tensor_workspace_dk_flat, [0, idx])
                    else:
                        pl.load(in_ping, tensor_info.tensor_workspace_dv_flat, [0, idx])

                    pl.set_validshape(out_ping, [1, ping_size])
                    if qkv < 2:
                        pl.mul(in_ping, in_ping, const_info.softmax_scale)
                    pl.cast(out_ping, in_ping, mode=pl.RoundMode.CAST_ROUND)

                    pl.set_validshape(out_ping, [1, ping_size])
                    if qkv == 0:
                        pl.store(tensor_info.tensor_dq_out_flat, out_ping, [0, idx])
                    elif qkv == 1:
                        pl.store(tensor_info.tensor_dk_out_flat, out_ping, [0, idx])
                    else:
                        pl.store(tensor_info.tensor_dv_out_flat, out_ping, [0, idx])
                    if pong_off < end:
                        pong_size = TS * TS if pong_off < total else tail
                        pl.set_validshape(in_pong, [1, pong_size])

                        if qkv == 0:
                            pl.load(
                                in_pong,
                                tensor_info.tensor_workspace_dq_flat,
                                [0, pong_off],
                            )
                        elif qkv == 1:
                            pl.load(
                                in_pong,
                                tensor_info.tensor_workspace_dk_flat,
                                [0, pong_off],
                            )
                        else:
                            pl.load(
                                in_pong,
                                tensor_info.tensor_workspace_dv_flat,
                                [0, pong_off],
                            )

                        pl.set_validshape(out_pong, [1, pong_size])
                        if qkv < 2:
                            pl.mul(in_pong, in_pong, const_info.softmax_scale)
                        pl.cast(out_pong, in_pong, mode=pl.RoundMode.CAST_ROUND)

                        pl.set_validshape(out_pong, [1, pong_size])
                        if qkv == 0:
                            pl.store(
                                tensor_info.tensor_dq_out_flat, out_pong, [0, pong_off]
                            )
                        elif qkv == 1:
                            pl.store(
                                tensor_info.tensor_dk_out_flat, out_pong, [0, pong_off]
                            )
                        else:
                            pl.store(
                                tensor_info.tensor_dv_out_flat, out_pong, [0, pong_off]
                            )
                    idx += TS * TS * 2


# ════════════════════════════════════════════════════════════════════════
#  KERNEL (PyPTO Pro SIMD, CV fusion)
# ════════════════════════════════════════════════════════════════════════


# ================================================================
#  Kernel — dynamic rank: inputs are raw pointers, shapes come from tiling
# ================================================================
@pl.jit(
    auto_mutex=True,
    tiling_key=QuantFlashAttnGradTilingKey,
    datatype={
        "q": "input_dtype",
        "attn_out": "output_dtype",
    },
)
def quant_flash_attn_grad(
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    q_descale: pl.Ptr[pl.DT_UINT8],
    k_descale: pl.Ptr[pl.DT_UINT8],
    v_descale: pl.Ptr[pl.DT_UINT8],
    do_descale: pl.Ptr[pl.DT_UINT8],
    p_scale: pl.Ptr[pl.DT_UINT8],
    ds_scale: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    dsink: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: QuantFlashAttnGradTiling,
):
    tensor_q_descale = pl.make_tensor(
        q_descale,
        [1],
        [1],
        dtype=pl.DT_FP32,
    )
    tensor_k_descale = pl.make_tensor(
        k_descale,
        [1],
        [1],
        dtype=pl.DT_FP32,
    )
    tensor_v_descale = pl.make_tensor(
        v_descale,
        [1],
        [1],
        dtype=pl.DT_FP32,
    )
    tensor_do_descale = pl.make_tensor(
        do_descale,
        [1],
        [1],
        dtype=pl.DT_FP32,
    )
    tensor_p_scale = pl.make_tensor(
        p_scale,
        [1],
        [1],
        dtype=pl.DT_FP32,
    )
    tensor_ds_scale = pl.make_tensor(
        ds_scale,
        [1],
        [1],
        dtype=pl.DT_FP32,
    )
    tensor_metadata = pl.make_tensor(
        metadata,
        [2, tiling.metadata_len],
        [tiling.metadata_len, 1],
        dtype=pl.DT_INT32,
    )
    tensor_workspace_sfmg = pl.make_tensor(
        workspace + tiling.sfmg_work_space_offset,
        [tiling.b, tiling.n1, tiling.s1],
        [tiling.n1 * tiling.s1, tiling.s1, 1],
        dtype=pl.DT_FP32,
    )
    if has_sink == 1:
        tensor_sinks = pl.make_tensor(
            sinks,
            [tiling.n1],
            [1],
            dtype=pl.DT_FP32,
        )

    if layout == 0:  # BSND
        tensor_q = pl.make_tensor(
            q,
            [tiling.b, tiling.s1, tiling.n1, tiling.d],
            [tiling.s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_q_ds = pl.make_tensor(
            q,
            [tiling.b, tiling.s1, tiling.n1, tiling.d],
            [tiling.s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_k = pl.make_tensor(
            k,
            [tiling.b, tiling.s2, tiling.n2, tiling.d],
            [tiling.s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_v = pl.make_tensor(
            v,
            [tiling.b, tiling.s2, tiling.n2, tiling.d],
            [tiling.s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_do = pl.make_tensor(
            dout,
            [tiling.b, tiling.s1, tiling.n1, tiling.d],
            [tiling.s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_do_p_dy = pl.make_tensor(
            dout,
            [tiling.b, tiling.s1, tiling.n1, tiling.d],
            [tiling.s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_attn_out = pl.make_tensor(
            attn_out,
            [tiling.b, tiling.s1, tiling.n1, tiling.d],
            [tiling.s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_softmax_lse = pl.make_tensor(
            softmax_lse,
            [tiling.b, tiling.n1, tiling.s1],
            [tiling.s1 * tiling.n1, tiling.s1, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dq = pl.make_tensor(
            workspace + tiling.dq_work_space_offset,
            [tiling.b, tiling.s1, tiling.n1, tiling.d],
            [tiling.s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dq_flat = pl.make_tensor(
            workspace + tiling.dq_work_space_offset,
            [1, tiling.b * tiling.s1 * tiling.n1 * tiling.d],
            [tiling.b * tiling.s1 * tiling.n1 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dq_out_flat = pl.make_tensor(
            dq,
            [1, tiling.b * tiling.s1 * tiling.n1 * tiling.d],
            [tiling.b * tiling.s1 * tiling.n1 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_workspace_dk = pl.make_tensor(
            workspace + tiling.dk_work_space_offset,
            [tiling.b, tiling.s2, tiling.n2, tiling.d],
            [tiling.s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dk_flat = pl.make_tensor(
            workspace + tiling.dk_work_space_offset,
            [1, tiling.b * tiling.s2 * tiling.n2 * tiling.d],
            [tiling.b * tiling.s2 * tiling.n2 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dk_out_flat = pl.make_tensor(
            dk,
            [1, tiling.b * tiling.s2 * tiling.n2 * tiling.d],
            [tiling.b * tiling.s2 * tiling.n2 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_workspace_dv = pl.make_tensor(
            workspace + tiling.dv_work_space_offset,
            [tiling.b, tiling.s2, tiling.n2, tiling.d],
            [tiling.s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dv_flat = pl.make_tensor(
            workspace + tiling.dv_work_space_offset,
            [1, tiling.b * tiling.s2 * tiling.n2 * tiling.d],
            [tiling.b * tiling.s2 * tiling.n2 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dv_out_flat = pl.make_tensor(
            dv,
            [1, tiling.b * tiling.s2 * tiling.n2 * tiling.d],
            [tiling.b * tiling.s2 * tiling.n2 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dq_out = pl.make_tensor(
            dq,
            [tiling.b, tiling.s1, tiling.n1, tiling.d],
            [tiling.s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dk_out = pl.make_tensor(
            dk,
            [tiling.b, tiling.s2, tiling.n2, tiling.d],
            [tiling.s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dv_out = pl.make_tensor(
            dv,
            [tiling.b, tiling.s2, tiling.n2, tiling.d],
            [tiling.s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
    elif layout == 1:  # BNSD
        tensor_q = pl.make_tensor(
            q,
            [tiling.b, tiling.n1, tiling.s1, tiling.d],
            [tiling.n1 * tiling.s1 * tiling.d, tiling.s1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_q_ds = pl.make_tensor(
            q,
            [tiling.b, tiling.n1, tiling.s1, tiling.d],
            [tiling.n1 * tiling.s1 * tiling.d, tiling.s1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_k = pl.make_tensor(
            k,
            [tiling.b, tiling.n2, tiling.s2, tiling.d],
            [tiling.n2 * tiling.s2 * tiling.d, tiling.s2 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_v = pl.make_tensor(
            v,
            [tiling.b, tiling.n2, tiling.s2, tiling.d],
            [tiling.n2 * tiling.s2 * tiling.d, tiling.s2 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_do = pl.make_tensor(
            dout,
            [tiling.b, tiling.n1, tiling.s1, tiling.d],
            [tiling.n1 * tiling.s1 * tiling.d, tiling.s1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_do_p_dy = pl.make_tensor(
            dout,
            [tiling.b, tiling.n1, tiling.s1, tiling.d],
            [tiling.n1 * tiling.s1 * tiling.d, tiling.s1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_attn_out = pl.make_tensor(
            attn_out,
            [tiling.b, tiling.n1, tiling.s1, tiling.d],
            [tiling.n1 * tiling.s1 * tiling.d, tiling.s1 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_softmax_lse = pl.make_tensor(
            softmax_lse,
            [tiling.b, tiling.n1, tiling.s1],
            [tiling.n1 * tiling.s1, tiling.s1, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dq = pl.make_tensor(
            workspace + tiling.dq_work_space_offset,
            [tiling.b, tiling.n1, tiling.s1, tiling.d],
            [tiling.n1 * tiling.s1 * tiling.d, tiling.s1 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dk = pl.make_tensor(
            workspace + tiling.dk_work_space_offset,
            [tiling.b, tiling.n2, tiling.s2, tiling.d],
            [tiling.n2 * tiling.s2 * tiling.d, tiling.s2 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dv = pl.make_tensor(
            workspace + tiling.dv_work_space_offset,
            [tiling.b, tiling.n2, tiling.s2, tiling.d],
            [tiling.n2 * tiling.s2 * tiling.d, tiling.s2 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dq_out = pl.make_tensor(
            dq,
            [tiling.b, tiling.n1, tiling.s1, tiling.d],
            [tiling.n1 * tiling.s1 * tiling.d, tiling.s1 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dk_out = pl.make_tensor(
            dk,
            [tiling.b, tiling.n2, tiling.s2, tiling.d],
            [tiling.n2 * tiling.s2 * tiling.d, tiling.s2 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dv_out = pl.make_tensor(
            dv,
            [tiling.b, tiling.n2, tiling.s2, tiling.d],
            [tiling.n2 * tiling.s2 * tiling.d, tiling.s2 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_workspace_dq_flat = pl.make_tensor(
            workspace + tiling.dq_work_space_offset,
            [1, tiling.b * tiling.n1 * tiling.s1 * tiling.d],
            [tiling.b * tiling.n1 * tiling.s1 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dk_flat = pl.make_tensor(
            workspace + tiling.dk_work_space_offset,
            [1, tiling.b * tiling.n2 * tiling.s2 * tiling.d],
            [tiling.b * tiling.n2 * tiling.s2 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dv_flat = pl.make_tensor(
            workspace + tiling.dv_work_space_offset,
            [1, tiling.b * tiling.n2 * tiling.s2 * tiling.d],
            [tiling.b * tiling.n2 * tiling.s2 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dq_out_flat = pl.make_tensor(
            dq,
            [1, tiling.b * tiling.n1 * tiling.s1 * tiling.d],
            [tiling.b * tiling.n1 * tiling.s1 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dk_out_flat = pl.make_tensor(
            dk,
            [1, tiling.b * tiling.n2 * tiling.s2 * tiling.d],
            [tiling.b * tiling.n2 * tiling.s2 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dv_out_flat = pl.make_tensor(
            dv,
            [1, tiling.b * tiling.n2 * tiling.s2 * tiling.d],
            [tiling.b * tiling.n2 * tiling.s2 * tiling.d, 1],
            dtype=output_dtype,
        )
    core_id_cube = pl.get_block_idx() // pl.get_subblock_num()
    sub_id = pl.get_subblock_idx()
    core_id_vec = core_id_cube * pl.get_subblock_num() + sub_id
    core_num = pl.get_block_num()
    scale_p_value = pl.getval(tensor_p_scale, 0)
    scale_ds_value = pl.getval(tensor_ds_scale, 0)
    const_info = pl.make_tuple(
        b_size=tiling.b,
        g_size=tiling.g,
        s1_size=tiling.s1,
        s2_size=tiling.s2,
        n1_size=tiling.n1,
        n2_size=tiling.n2,
        d_size=tiling.d,
        s1_outer=tiling.s1_outer,
        s2_outer=tiling.s2_outer,
        s1_tail=tiling.s1_tail,
        s2_tail=tiling.s2_tail,
        softmax_scale=tiling.softmax_scale,
        n2_g=tiling.n2 * tiling.g,
        n2_d=tiling.n2 * tiling.d,
        s1_d=tiling.s1 * tiling.d,
        s2_d=tiling.s2 * tiling.d,
        g_d=tiling.g * tiling.d,
        n2_g_d=tiling.n2 * tiling.g * tiling.d,
        n2_s2_d=tiling.n2 * tiling.s2 * tiling.d,
        g_s1_d=tiling.g * tiling.s1 * tiling.d,
        n2_g_s1_d=tiling.n2 * tiling.g * tiling.s1 * tiling.d,
        scale_p=scale_p_value,
        scale_ds=scale_ds_value,
        deq_scale_p_value=1.0 / scale_p_value,
        deq_scale_ds_value=1.0 / scale_ds_value,
        sfmg_used_core_num=tiling.sfmg_used_core_num,
        sfmg_dy_buffer_len=tiling.sfmg_dy_buffer_len,
        sfmg_y_buffer_len=tiling.sfmg_y_buffer_len,
        sfmg_output_buffer_len=tiling.sfmg_output_buffer_len,
        single_loop_nburst_num=tiling.single_loop_nburst_num,
        normal_core_loop_times=tiling.normal_core_loop_times,
        tail_core_loop_times=tiling.tail_core_loop_times,
        normal_core_last_loop_nburst_num=tiling.normal_core_last_loop_nburst_num,
        tail_core_last_loop_nburst_num=tiling.tail_core_last_loop_nburst_num,
        normal_core_nburst_nums=tiling.normal_core_nburst_nums,
        tail_core_nburst_nums=tiling.tail_core_nburst_nums,
        normal_axis_size=tiling.normal_axis_size,
        q_pre_block_factor=tiling.q_pre_block_factor,
        q_pre_block_total=tiling.q_pre_block_total,
        q_pre_block_tail=tiling.q_pre_block_tail,
        q_post_block_factor=tiling.q_post_block_factor,
        q_post_block_total=tiling.q_post_block_total,
        q_post_tail_num=tiling.q_post_tail_num,
        k_post_block_factor=tiling.k_post_block_factor,
        k_post_block_total=tiling.k_post_block_total,
        k_post_tail_num=tiling.k_post_tail_num,
        v_post_block_factor=tiling.v_post_block_factor,
        v_post_block_total=tiling.v_post_block_total,
        v_post_tail_num=tiling.v_post_tail_num,
        sub_id=sub_id,
        core_id_vec=core_id_vec,
        core_id_cube=core_id_cube,
        core_num=core_num,
    )
    tmp_info = pl.struct_array(
        1,
        "tmp_info",
        l0c_buffer_id=0,
        last_batch_idx=-1,
        last_n2_idx=-1,
        last_s2_idx=-1,
    )
    sp_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA0, VA0 + 256],
        mutex_ids=[0, 1],
    )
    sp_vec_bit8 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_16, N_4096], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA0, VA0 + 256],
        mutex_ids=[0, 1],
    )
    dpds_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA1, VA1 + 256],
        mutex_ids=[2, 3],
    )
    dpds_vec_bit8 = pl.make_tile_group(
        type=pl.TileType(
            shape=[M_16, N_4096], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA1, VA1 + 256],
        mutex_ids=[2, 3],
    )
    dkv_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA2, VA2 + 256],
        mutex_ids=[29, 30],
    )
    dq_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=VA3,
        mutex_ids=[31],
    )
    lse_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA4, VA4 + 512, VA4 + 512 * 2, VA4 + 512 * 3],
        mutex_ids=[9, 10, 11, 12],
    )
    d_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA4 + 512 * 4, VA4 + 512 * 5, VA4 + 512 * 6, VA4 + 512 * 7],
        mutex_ids=[13, 14, 15, 16],
    )
    perm_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 128], dtype=pl.DT_UINT16, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA4 + 512 * 8],
        mutex_ids=[17],
    )
    zero_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA4 + 512 * 9],
        mutex_ids=[18],
    )
    with pl.section_vector():
        pl.expands(zero_vec.current(), 0)
        perm_tile = perm_vec.current()
        for i in pl.range(0, BLOCK_SIZE):
            perm_tile[0, 4 * i + 0] = 0 * BLOCK_SIZE + i
            perm_tile[0, 4 * i + 1] = 1 * BLOCK_SIZE + i
            perm_tile[0, 4 * i + 2] = 2 * BLOCK_SIZE + i
            perm_tile[0, 4 * i + 3] = 3 * BLOCK_SIZE + i

    ds_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=MA0,
        mutex_ids=[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    )
    ds_l1_t = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.ZN,
            compact=1,
        ),
        addrs=MA0,
        mutex_ids=[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    )
    ds_l1_bit8 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS_HALF, TS * 2],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=MA0,
        mutex_ids=[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    )
    p_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=MA1,
        mutex_ids=[8, 8, 8, 8],
    )
    p_l1_bit8 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS_HALF, TS * 2],
            dtype=pl.DT_INT8,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=MA1,
        mutex_ids=[8, 8, 8, 8],
    )
    common_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=MA2,
        mutex_ids=[9, 10, 11, 12],
    )
    common_l1_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS * 2, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=MA2,
        mutex_ids=[[9, 10], [11, 12]],
    )
    k_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=MA3,
        mutex_ids=[13, 14, 15, 16],
    )
    v_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
            compact=1,
        ),
        addrs=MA4,
        mutex_ids=[17, 18, 19, 20],
    )
    left_four = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Left,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=LA0,
        mutex_ids=[21, 22, 23, 24],
    )
    left_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS * 2],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Left,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=LA0,
        mutex_ids=[[21, 22], [23, 24]],
    )
    right_four = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Right,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=LA0,
        mutex_ids=[25, 26, 27, 28],
    )
    right_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS * 2, TS],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Right,
            valid_shape=[-1, -1],
            compact=1,
        ),
        addrs=RA0,
        mutex_ids=[[25, 26], [27, 28]],
    )
    acc_mm1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=CA0,
        mutex_ids=[0],
    )
    acc_mm2 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=CA1,
        mutex_ids=[2],
    )
    acc_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=CA2,
        mutex_ids=[4, 6],
    )
    ## post
    post_in = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, TS * TS],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[VAP_0, VAP_0 + TS * TS * 4],
        mutex_ids=[0, 1],
    )
    post_out = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, TS * TS],
            dtype=output_dtype,
            target_memory=pl.MemorySpace.Vec,
        ),
        addrs=[VAP_1, VAP_1 + TS * TS * 2],
        mutex_ids=[2, 3],
    )
    tensor_info = pl.make_tuple(
        tensor_q_descale=tensor_q_descale,
        tensor_k_descale=tensor_k_descale,
        tensor_v_descale=tensor_v_descale,
        tensor_do_descale=tensor_do_descale,
        tensor_p_scale=tensor_p_scale,
        tensor_ds_scale=tensor_ds_scale,
        tensor_metadata=tensor_metadata,
        tensor_q=tensor_q,
        tensor_q_ds=tensor_q_ds,
        tensor_k=tensor_k,
        tensor_v=tensor_v,
        tensor_do=tensor_do,
        tensor_do_p_dy=tensor_do_p_dy,
        tensor_attn_out=tensor_attn_out,
        tensor_softmax_lse=tensor_softmax_lse,
        tensor_workspace_dq=tensor_workspace_dq,
        tensor_workspace_dk=tensor_workspace_dk,
        tensor_workspace_dv=tensor_workspace_dv,
        tensor_workspace_dq_flat=tensor_workspace_dq_flat,
        tensor_workspace_dk_flat=tensor_workspace_dk_flat,
        tensor_workspace_dv_flat=tensor_workspace_dv_flat,
        tensor_workspace_sfmg=tensor_workspace_sfmg,
        tensor_dq_out=tensor_dq_out,
        tensor_dk_out=tensor_dk_out,
        tensor_dv_out=tensor_dv_out,
        tensor_dq_out_flat=tensor_dq_out_flat,
        tensor_dk_out_flat=tensor_dk_out_flat,
        tensor_dv_out_flat=tensor_dv_out_flat,
        sp_vec=sp_vec,
        sp_vec_bit8=sp_vec_bit8,
        dpds_vec=dpds_vec,
        dpds_vec_bit8=dpds_vec_bit8,
        dkv_vec=dkv_vec,
        dq_vec=dq_vec,
        lse_vec=lse_vec,
        d_vec=d_vec,
        perm_vec=perm_vec,
        zero_vec=zero_vec,
        ds_l1=ds_l1,
        ds_l1_t=ds_l1_t,
        ds_l1_bit8=ds_l1_bit8,
        p_l1=p_l1,
        p_l1_bit8=p_l1_bit8,
        common_l1=common_l1,
        common_l1_db=common_l1_db,
        k_l1=k_l1,
        v_l1=v_l1,
        left_four=left_four,
        left_db=left_db,
        right_four=right_four,
        right_db=right_db,
        acc_mm1=acc_mm1,
        acc_mm2=acc_mm2,
        acc_db=acc_db,
        post_in=post_in,
        post_out=post_out,
    )

    run_infos = pl.struct_array(
        2,
        "run_info",
        batch_id=0,
        s1_idx=0,
        s2_idx=0,
        bo_idx=0,
        n2o_idx=0,
        n1o_idx=0,
        go_idx=0,
        s2o_idx=0,
        s1o_idx=0,
        s2_cv_begin=0,
        s1_real_size=0,
        s2_real_size=0,
        inner_s1_loop_num=0,
        inner_s2_loop_num=0,
        inner_s1_real_size=[0, 0, 0, 0],
        inner_s2_real_size=[0, 0, 0, 0],
        maxsum_offset=0,
        deq_scale_q_value=0.0,
        deq_scale_k_value=0.0,
        deq_scale_v_value=0.0,
        deq_scale_do_value=0.0,
        kv_need_atomic=False,
        is_key_reuse=False,
        is_first_process_block=False,
        is_last_process_block=False,
        is_next_key_reuse=False,
        is_value_reuse=False,
        is_first_block=True,
        is_dkv_completed=True,
        is_dq_completed=True,
        is_dq_fix_out=False,
        is_dk_fix_out=False,
        is_dv_fix_out=False,
        task_id=0,
        task_id_mod2=0,
    )
    coordinate_infos = pl.struct_array(
        3,
        "coordinate_info",
        batch_id=0,
        s1_idx=0,
        s2_idx=0,
        n2_idx=0,
        g_idx=0,
        s1_outer=0,
        s2_outer=0,
        m_offset=0,
        n_offset=0,
    )
    next_core_first_block_coordinate_info = coordinate_infos[2]

    # process presfmg
    presfmg_quant_inner_hif8(const_info, tensor_info)
    pl.system.sync_all()

    # process kernel
    alloc_event_id()
    loop_max = 0
    loop_max = loop_max + tensor_metadata[1, 0]
    task_id = 0

    init_coordinate_info(const_info, 0, 0, coordinate_infos[0])
    init_coordinate_info(const_info, 0, 0, coordinate_infos[1])
    with pl.section_vector():
        if const_info.s2_outer == 1:
            init_coordinate_info(
                const_info, 0, 0, next_core_first_block_coordinate_info
            )
            next_valid_loop_idx, next_block_idx = cal_deter_index(
                0, loop_max, next_core_first_block_coordinate_info, const_info, True
            )
    next_valid_loop_idx, next_block_idx = cal_deter_index(
        0, loop_max, coordinate_infos[task_id % 2], const_info
    )
    for loop_idx in pl.range(loop_max):
        block_inner_idx = next_block_idx
        if loop_idx >= next_valid_loop_idx:
            block_inner_idx = next_block_idx
            next_valid_loop_idx, next_block_idx = cal_deter_index(
                loop_idx + 1, loop_max, coordinate_infos[(task_id + 1) % 2], const_info
            )
            run_infos[task_id % 2].is_dkv_completed = False
            run_infos[task_id % 2].is_dq_completed = False
        else:
            block_inner_idx = -1
        is_valid_block = block_inner_idx >= 0
        if is_valid_block:
            set_run_info(
                run_infos[task_id % 2],
                run_infos[(task_id + 1) % 2],
                task_id,
                coordinate_infos[task_id % 2],
                coordinate_infos[(task_id + 1) % 2],
                next_core_first_block_coordinate_info,
                const_info,
                tensor_info,
                tmp_info[0],
            )
            run_info = run_infos[task_id % 2]
            for kv_inner_id in pl.range(4):
                process_first_s2(
                    const_info,
                    run_infos[task_id % 2],
                    run_infos[(task_id + 1) % 2],
                    kv_inner_id,
                    tensor_info,
                    tmp_info[0],
                )
        else:
            with pl.section_vector():
                if loop_idx > 0:
                    for j in pl.range(4):
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE3,
                            event_id=SYNC_DETER_FLAG,
                            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                        )
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.MTE3,
                            event_id=SYNC_DETER_FLAG,
                            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                        )
        if is_valid_block:
            task_id += 1

    if not run_infos[(task_id + 1) % 2].is_dkv_completed:
        kv_iib = 3
        with pl.section_vector():
            set_quant_run_info(run_infos[(task_id + 1) % 2], 0, kv_iib - 1)
            if (
                run_infos[(task_id + 1) % 2].s2_idx
                < run_infos[(task_id + 1) % 2].inner_s2_loop_num
            ):
                copy_dqkv2gm(
                    True, run_infos[(task_id + 1) % 2], const_info, tensor_info
                )
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_TRANSFER_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )

        set_quant_run_info(run_infos[(task_id + 1) % 2], 0, kv_iib)
        process_dkv(
            False, run_infos[(task_id + 1) % 2], 0, const_info, tensor_info, tmp_info[0]
        )
        set_quant_run_info(run_infos[(task_id + 1) % 2], 2, kv_iib)
        process_dkv(
            False, run_infos[(task_id + 1) % 2], 2, const_info, tensor_info, tmp_info[0]
        )

        set_quant_run_info(run_infos[(task_id + 1) % 2], 0, kv_iib)
        process_dkv(
            True, run_infos[(task_id + 1) % 2], 0, const_info, tensor_info, tmp_info[0]
        )
        set_quant_run_info(run_infos[(task_id + 1) % 2], 2, kv_iib)
        process_dkv(
            True, run_infos[(task_id + 1) % 2], 2, const_info, tensor_info, tmp_info[0]
        )

        with pl.section_vector():
            set_quant_run_info(run_infos[(task_id + 1) % 2], 0, kv_iib)
            if (
                run_infos[(task_id + 1) % 2].s2_idx
                < run_infos[(task_id + 1) % 2].inner_s2_loop_num
            ):
                copy_dqkv2gm(
                    True, run_infos[(task_id + 1) % 2], const_info, tensor_info
                )
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_TRANSFER_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )

    if not run_infos[(task_id + 1) % 2].is_dq_completed:
        for i in pl.range(4):
            set_quant_run_info(run_infos[(task_id + 1) % 2], i, 0)
            process_dq(
                run_infos[(task_id + 1) % 2], 0, const_info, tensor_info, tmp_info[0]
            )
            set_quant_run_info(run_infos[(task_id + 1) % 2], i, 2)
            process_dq(
                run_infos[(task_id + 1) % 2], 2, const_info, tensor_info, tmp_info[0]
            )
            compute_dq(run_infos[(task_id + 1) % 2], const_info, tensor_info)
            with pl.section_vector():
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=SYNC_DETER_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                if (
                    run_infos[(task_id + 1) % 2].s1_idx
                    < run_infos[(task_id + 1) % 2].inner_s1_loop_num
                ):
                    copy_dqkv2gm(
                        False, run_infos[(task_id + 1) % 2], const_info, tensor_info
                    )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=SYNC_DETER_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=SYNC_TRANSFER_DQ_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                )
    else:
        with pl.section_vector():
            for i in pl.range(4):
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=SYNC_DETER_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=SYNC_DETER_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
    free_event_id()
    pl.system.sync_all()

    # process post
    process_post_dqkv(const_info, tensor_info)
