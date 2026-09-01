# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SparseFlashMlaSoftmaxL1Norm kernel (pypto-pro)."""

import math
from dataclasses import dataclass

import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 1
TG = 128
TG_HALF = TG // 2
TKV = 128
TD = 128
D_TOTAL = 512
SCALE = 1.0 / math.sqrt(D_TOTAL)

# Cube tile byte sizes
Q_F16 = TG * D_TOTAL * 2  # [TG,  TD]  FP16 = 32KB
KT_F16 = TKV * TD * 2  # [TKV, TD]  FP16 = 32KB
QK_F32 = TKV * TS * 4  # [TKV, TS]  FP32 = 64KB

# ---- MAT (512KB) ----
MA0 = 0
MA1 = Q_F16 * 2

# DN: Left holds K [TKV, TD], Right holds Q^T [TD, TS].
LA0 = 0
LA1 = KT_F16
RA0 = 0
RA1 = Q_F16
CA0 = 0
CA1 = QK_F32

# ---- VEC addresses (248KB on a5) ----
CUBE_BUFFER_NUM = 2
VEC_BUFFER_NUM = 2
GATHER_ROW_NUM = 32
VB4_KV = TG_HALF * TKV * 4 * VEC_BUFFER_NUM  # [TG_HALF, TKV] FP32 = 64KB
VB4_GATHER = GATHER_ROW_NUM * D_TOTAL * 2 * VEC_BUFFER_NUM  # 64KB
VB4_LSE = TG * 4 * VEC_BUFFER_NUM  # 1k
VB4_OUT = TKV * 4 * VEC_BUFFER_NUM  # 1k
VB4_GATHER_NZ = GATHER_ROW_NUM * D_TOTAL * 2 * VEC_BUFFER_NUM  # 64KB

# Cross-core event IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BACKWARD_IDS = (2, 3)
GATHER_READY_FORWARD_IDS = (4, 5)
GATHER_READY_BACKWARD_IDS = (6, 7)


def _align_up(value, align=1024):
    return ((value + align - 1) // align) * align


def _ceil_div(value, align):
    return (value + align - 1) // align


VA0 = 0  # qk_vec  [TKV, TS_HALF] FP32
VA1 = _align_up(VA0 + VB4_KV)  # tmp_vec [TKV, TS_HALF] FP32
VA2 = _align_up(VA1 + VB4_GATHER)  # p_f16   [TKV, TS_HALF] FP16
VA3 = _align_up(VA2 + VB4_LSE)
VA4 = _align_up(VA3 + VB4_OUT)
VA5 = _align_up(VA4 + VB4_GATHER_NZ)

EVENT_IDS_01 = (0, 1)
EVENT_IDS_23 = (2, 3)
QK_READY_IDS = (0, 1)
P_READY_IDS = (2, 3)
PV_READY_IDS = (4, 5)
QK_MAX_EID = 2
P_MAX_EID = 4
PV_MAX_EID = 6


# ================================================================
#  Tiling data
# ================================================================
@dataclass
class OpTiling:
    b: int
    sq: int
    sk: int
    g: int
    d: int
    t1: int
    t2: int
    max_seqlen_k: int
    k_length: int
    cmp_ratio: int
    init_per_core_num: int
    init_total_num: int
    softmax_scale: float
    has_seqused_q: bool
    has_seqused_k: bool
    has_topk_length: bool


# ================================================================
#  TilingKey
# ================================================================
class SmlaTilingKey:
    SPARSE_MODE = TilingKeyField(bits=2, values=[0, 1, 2, 3])
    IS_TND = TilingKeyField(bits=1, values=[0, 1])
    IS_SPARSE = TilingKeyField(bits=1, values=[0, 1])


def get_sparse_offset(const_info, run_info, first_core_half_s2_base_size):
    offset = 0
    if IS_TND == 1:
        offset = (
            run_info.t1_offset + run_info.s1_idx
        ) * const_info.k_length + const_info.sub_id * first_core_half_s2_base_size
    else:
        offset = (
            run_info.batch_id * const_info.s1_size * const_info.k_length
            + run_info.s1_idx * const_info.k_length
            + const_info.sub_id * first_core_half_s2_base_size
        )
    return offset


def get_seqused(seq_dim, has_seqused, seqused_tensor, batch_id):
    seq_length = 0
    if has_seqused:
        seq_length = seqused_tensor[batch_id]
    else:
        seq_length = seq_dim
    return seq_length


@pl.pipeline.stage
def gather_k(
    k_l1_db, k, sparse_indices, gather_nd_db, gather_nz_db, const_info, run_info
):
    half_s2_base_size = _ceil_div(run_info.s2_base_size, 2)
    first_core_half_s2_base_size = half_s2_base_size
    current_core_half_s2_base_size = first_core_half_s2_base_size
    if const_info.sub_id == 1:
        current_core_half_s2_base_size = (
            run_info.s2_base_size - first_core_half_s2_base_size
        )

    loop_num = (current_core_half_s2_base_size + GATHER_ROW_NUM - 1) // GATHER_ROW_NUM
    key_l1_slot = k_l1_db.next()
    pl.set_validshape(key_l1_slot, [D_TOTAL, run_info.s2_base_size])

    sparse_indices_s1_offset = get_sparse_offset(
        const_info, run_info, first_core_half_s2_base_size
    )
    for loop_idx in pl.range(loop_num):
        sparse_indices_k_offset = run_info.ki + loop_idx * GATHER_ROW_NUM
        gather_k_slot = gather_nd_db.next()

        if loop_idx == loop_num - 1:
            actual_row_num = current_core_half_s2_base_size - loop_idx * GATHER_ROW_NUM
        else:
            actual_row_num = GATHER_ROW_NUM
        sparse_indices_s1k_offset = sparse_indices_s1_offset + sparse_indices_k_offset
        for i in pl.range(actual_row_num):
            k_index = pl.getval(sparse_indices, sparse_indices_s1k_offset + i)
            if k_index < 0:
                break
            gather_k_slot_single_row = gather_k_slot[i:, :]
            pl.set_validshape(gather_k_slot_single_row, [1, D_TOTAL])
            if IS_TND == 1:
                pl.load(
                    gather_k_slot_single_row,
                    k,
                    [run_info.t2_offset + k_index, 0, 0],
                    order=[0, 2],
                )
            else:
                pl.load(
                    gather_k_slot_single_row,
                    k,
                    [run_info.batch_id, k_index, 0, 0],
                    order=[2, 3],
                )

        gather_k_nz_slot = gather_nz_db.next()
        pl.set_validshape(gather_k_nz_slot, [actual_row_num, D_TOTAL])
        pl.set_validshape(gather_k_slot, [actual_row_num, D_TOTAL])
        pl.move(gather_k_nz_slot, gather_k_slot)  # ND → NZ
        pl.insert(
            key_l1_slot,
            gather_k_nz_slot,
            [
                const_info.sub_id * first_core_half_s2_base_size
                + loop_idx * GATHER_ROW_NUM,
                0,
            ],
        )  # UB → L1


@pl.pipeline.stage
def compute_qk(
    q,
    k,
    cur_q_slot,
    k_l1_db,
    left_db,
    right_db,
    acc_db,
    qk_vec_db,
    const_info,
    run_info,
):
    """QK matmul stage (Cube). Writes qk_vec."""
    qk_acc = acc_db.next()
    g_align2 = (const_info.g_size + 1) // 2 * 2
    pl.set_validshape(qk_acc, [g_align2, TKV])

    # q复用
    if run_info.ki == 0:
        if IS_TND == 1:
            pl.load(
                cur_q_slot,
                q,
                [run_info.t1_offset + run_info.s1_idx, 0, 0],
                order=[1, 2],
            )
        else:
            pl.load(
                cur_q_slot, q, [run_info.batch_id, run_info.s1_idx, 0, 0], order=[2, 3]
            )

    key_l1_slot = k_l1_db.next()
    pl.set_validshape(key_l1_slot, [D_TOTAL, run_info.s2_base_size])

    for d_offset in pl.range(0, D_TOTAL, TD):
        qk_left = left_db.next()
        pl.set_validshape(qk_left, [g_align2, TD])
        qk_right = right_db.next()
        pl.set_validshape(qk_right, [TD, run_info.s2_base_size])

        pl.move(qk_left, cur_q_slot, [0, d_offset])
        pl.move(qk_right, key_l1_slot, [d_offset, 0])
        if d_offset == 0:
            pl.matmul(qk_acc, qk_left, qk_right)
        else:
            pl.matmul_acc(qk_acc, qk_acc, qk_left, qk_right)

    qk_vec_slot = qk_vec_db.next()

    pl.move(qk_vec_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def compute_qk_dense(
    q,
    k,
    cur_q_slot,
    k_l1_db,
    left_db,
    right_db,
    acc_db,
    qk_vec_db,
    const_info,
    run_info,
):
    """QK matmul stage (Cube). Writes qk_vec."""
    qk_acc = acc_db.next()
    g_align2 = (const_info.g_size + 1) // 2 * 2
    pl.set_validshape(qk_acc, [g_align2, TKV])

    # q复用
    if run_info.ki == 0:
        if IS_TND == 1:
            pl.load(
                cur_q_slot,
                q,
                [run_info.t1_offset + run_info.s1_idx, 0, 0],
                order=[1, 2],
            )
        else:
            pl.load(
                cur_q_slot, q, [run_info.batch_id, run_info.s1_idx, 0, 0], order=[2, 3]
            )

    for d_offset in pl.range(0, D_TOTAL, TD):
        cur_k_slot = k_l1_db.next()
        pl.set_validshape(cur_k_slot, [TD, run_info.s2_base_size])

        if IS_TND == 1:
            pl.load(
                cur_k_slot,
                k,
                [run_info.t2_offset + run_info.ki, 0, d_offset],
                order=[2, 0],
            )
        else:
            pl.load(
                cur_k_slot,
                k,
                [run_info.batch_id, run_info.ki, 0, d_offset],
                order=[3, 1],
            )
        qk_left = left_db.next()
        pl.set_validshape(qk_left, [g_align2, TD])
        qk_right = right_db.next()
        pl.set_validshape(qk_right, [TD, run_info.s2_base_size])

        pl.move(qk_left, cur_q_slot, [0, d_offset])
        pl.move(qk_right, cur_k_slot)
        if d_offset == 0:
            pl.matmul(qk_acc, qk_left, qk_right)
        else:
            pl.matmul_acc(qk_acc, qk_acc, qk_left, qk_right)

    qk_vec_slot = qk_vec_db.next()

    pl.move(qk_vec_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


# Alternative implementation using multiple pre-configured tiles
@pl.vector_function
def _muls_sel_vf_pse_type1_inner(
    src_tiles,
    lse_tiles,
    softmax_l1_tiles,
    scale,
    src_m,
    repeat_times,
    tail_size,
    g_scalar,
):
    preg_full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(tail_size, dtype=pl.DT_FP32)

    vreg_reduce_sum_muls = vf.full(0.0, preg_full)
    vreg_reduce_sum_muls_tail = vf.full(0.0, preg_tail)
    vreg_g_scale = vf.full(g_scalar, preg_full)

    un_roll_num = 8
    n1_main = src_m // un_roll_num * un_roll_num
    tail_offset = repeat_times * 64
    for _ in pl.range(0, repeat_times):
        for m_idx in pl.range(0, n1_main, un_roll_num):
            vreg_lse0 = vf.load_align(lse_tiles, m_idx, dist=pl.LoadDist.BRC_B32)
            vreg_src0 = vf.load_align(src_tiles, m_idx * TKV)
            vreg_muls0 = vf.muls(vreg_src0, scale, preg_full)
            vreg_exp0 = vf.exp_sub(vreg_muls0, vreg_lse0, preg_full)

            vreg_lse1 = vf.load_align(lse_tiles, m_idx + 1, dist=pl.LoadDist.BRC_B32)
            vreg_src1 = vf.load_align(src_tiles, (m_idx + 1) * TKV)
            vreg_muls1 = vf.muls(vreg_src1, scale, preg_full)
            vreg_exp1 = vf.exp_sub(vreg_muls1, vreg_lse1, preg_full)

            vreg_lse2 = vf.load_align(lse_tiles, m_idx + 2, dist=pl.LoadDist.BRC_B32)
            vreg_src2 = vf.load_align(src_tiles, (m_idx + 2) * TKV)
            vreg_muls2 = vf.muls(vreg_src2, scale, preg_full)
            vreg_exp2 = vf.exp_sub(vreg_muls2, vreg_lse2, preg_full)

            vreg_lse3 = vf.load_align(lse_tiles, m_idx + 3, dist=pl.LoadDist.BRC_B32)
            vreg_src3 = vf.load_align(src_tiles, (m_idx + 3) * TKV)
            vreg_muls3 = vf.muls(vreg_src3, scale, preg_full)
            vreg_exp3 = vf.exp_sub(vreg_muls3, vreg_lse3, preg_full)

            vreg_lse4 = vf.load_align(lse_tiles, m_idx + 4, dist=pl.LoadDist.BRC_B32)
            vreg_src4 = vf.load_align(src_tiles, (m_idx + 4) * TKV)
            vreg_muls4 = vf.muls(vreg_src4, scale, preg_full)
            vreg_exp4 = vf.exp_sub(vreg_muls4, vreg_lse4, preg_full)

            vreg_lse5 = vf.load_align(lse_tiles, m_idx + 5, dist=pl.LoadDist.BRC_B32)
            vreg_src5 = vf.load_align(src_tiles, (m_idx + 5) * TKV)
            vreg_muls5 = vf.muls(vreg_src5, scale, preg_full)
            vreg_exp5 = vf.exp_sub(vreg_muls5, vreg_lse5, preg_full)

            vreg_lse6 = vf.load_align(lse_tiles, m_idx + 6, dist=pl.LoadDist.BRC_B32)
            vreg_src6 = vf.load_align(src_tiles, (m_idx + 6) * TKV)
            vreg_muls6 = vf.muls(vreg_src6, scale, preg_full)
            vreg_exp6 = vf.exp_sub(vreg_muls6, vreg_lse6, preg_full)

            vreg_lse7 = vf.load_align(lse_tiles, m_idx + 7, dist=pl.LoadDist.BRC_B32)
            vreg_src7 = vf.load_align(src_tiles, (m_idx + 7) * TKV)
            vreg_muls7 = vf.muls(vreg_src7, scale, preg_full)
            vreg_exp7 = vf.exp_sub(vreg_muls7, vreg_lse7, preg_full)

            vreg_sum01 = vf.add(vreg_exp0, vreg_exp1, preg_full)
            vreg_sum23 = vf.add(vreg_exp2, vreg_exp3, preg_full)
            vreg_sum45 = vf.add(vreg_exp4, vreg_exp5, preg_full)
            vreg_sum67 = vf.add(vreg_exp6, vreg_exp7, preg_full)
            vreg_sum0123 = vf.add(vreg_sum01, vreg_sum23, preg_full)
            vreg_sum4567 = vf.add(vreg_sum45, vreg_sum67, preg_full)
            vreg_sumall = vf.add(vreg_sum0123, vreg_sum4567, preg_full)
            vreg_reduce_sum_muls = vf.add(vreg_reduce_sum_muls, vreg_sumall, preg_full)

        for m_idx in pl.range(n1_main, src_m):
            vreg_lse = vf.load_align(lse_tiles, m_idx, dist=pl.LoadDist.BRC_B32)
            vreg_src = vf.load_align(src_tiles, m_idx * TKV)
            vreg_muls = vf.muls(vreg_src, scale, preg_full)
            vreg_exp = vf.exp_sub(vreg_muls, vreg_lse, preg_full)
            vreg_reduce_sum_muls = vf.add(vreg_reduce_sum_muls, vreg_exp, preg_full)

    for m_idx in pl.range(0, n1_main, un_roll_num):
        vreg_lse0 = vf.load_align(lse_tiles, m_idx, dist=pl.LoadDist.BRC_B32)
        vreg_src0 = vf.load_align(src_tiles, m_idx * TKV + tail_offset)
        vreg_muls0 = vf.muls(vreg_src0, scale, preg_tail)
        vreg_exp0 = vf.exp_sub(vreg_muls0, vreg_lse0, preg_tail)

        vreg_lse1 = vf.load_align(lse_tiles, m_idx + 1, dist=pl.LoadDist.BRC_B32)
        vreg_src1 = vf.load_align(src_tiles, (m_idx + 1) * TKV + tail_offset)
        vreg_muls1 = vf.muls(vreg_src1, scale, preg_tail)
        vreg_exp1 = vf.exp_sub(vreg_muls1, vreg_lse1, preg_tail)

        vreg_lse2 = vf.load_align(lse_tiles, m_idx + 2, dist=pl.LoadDist.BRC_B32)
        vreg_src2 = vf.load_align(src_tiles, (m_idx + 2) * TKV + tail_offset)
        vreg_muls2 = vf.muls(vreg_src2, scale, preg_tail)
        vreg_exp2 = vf.exp_sub(vreg_muls2, vreg_lse2, preg_tail)

        vreg_lse3 = vf.load_align(lse_tiles, m_idx + 3, dist=pl.LoadDist.BRC_B32)
        vreg_src3 = vf.load_align(src_tiles, (m_idx + 3) * TKV + tail_offset)
        vreg_muls3 = vf.muls(vreg_src3, scale, preg_tail)
        vreg_exp3 = vf.exp_sub(vreg_muls3, vreg_lse3, preg_tail)

        vreg_lse4 = vf.load_align(lse_tiles, m_idx + 4, dist=pl.LoadDist.BRC_B32)
        vreg_src4 = vf.load_align(src_tiles, (m_idx + 4) * TKV + tail_offset)
        vreg_muls4 = vf.muls(vreg_src4, scale, preg_tail)
        vreg_exp4 = vf.exp_sub(vreg_muls4, vreg_lse4, preg_tail)

        vreg_lse5 = vf.load_align(lse_tiles, m_idx + 5, dist=pl.LoadDist.BRC_B32)
        vreg_src5 = vf.load_align(src_tiles, (m_idx + 5) * TKV + tail_offset)
        vreg_muls5 = vf.muls(vreg_src5, scale, preg_tail)
        vreg_exp5 = vf.exp_sub(vreg_muls5, vreg_lse5, preg_tail)

        vreg_lse6 = vf.load_align(lse_tiles, m_idx + 6, dist=pl.LoadDist.BRC_B32)
        vreg_src6 = vf.load_align(src_tiles, (m_idx + 6) * TKV + tail_offset)
        vreg_muls6 = vf.muls(vreg_src6, scale, preg_tail)
        vreg_exp6 = vf.exp_sub(vreg_muls6, vreg_lse6, preg_tail)

        vreg_lse7 = vf.load_align(lse_tiles, m_idx + 7, dist=pl.LoadDist.BRC_B32)
        vreg_src7 = vf.load_align(src_tiles, (m_idx + 7) * TKV + tail_offset)
        vreg_muls7 = vf.muls(vreg_src7, scale, preg_tail)
        vreg_exp7 = vf.exp_sub(vreg_muls7, vreg_lse7, preg_tail)

        vreg_sum01 = vf.add(vreg_exp0, vreg_exp1, preg_tail)
        vreg_sum23 = vf.add(vreg_exp2, vreg_exp3, preg_tail)
        vreg_sum45 = vf.add(vreg_exp4, vreg_exp5, preg_tail)
        vreg_sum67 = vf.add(vreg_exp6, vreg_exp7, preg_tail)
        vreg_sum0123 = vf.add(vreg_sum01, vreg_sum23, preg_tail)
        vreg_sum4567 = vf.add(vreg_sum45, vreg_sum67, preg_tail)
        vreg_sumall = vf.add(vreg_sum0123, vreg_sum4567, preg_tail)
        vreg_reduce_sum_muls_tail = vf.add(
            vreg_reduce_sum_muls_tail, vreg_sumall, preg_tail
        )

    for m_idx in pl.range(n1_main, src_m):
        vreg_lse = vf.load_align(lse_tiles, m_idx, dist=pl.LoadDist.BRC_B32)
        vreg_src = vf.load_align(src_tiles, m_idx * TKV + tail_offset)
        vreg_muls = vf.muls(vreg_src, scale, preg_tail)
        vreg_exp = vf.exp_sub(vreg_muls, vreg_lse, preg_tail)
        vreg_reduce_sum_muls_tail = vf.add(
            vreg_reduce_sum_muls_tail, vreg_exp, preg_tail
        )

    for _ in pl.range(0, repeat_times):
        vreg_reduce_sum_muls = vf.div(vreg_reduce_sum_muls, vreg_g_scale, preg_full)
        vf.store_align(softmax_l1_tiles, vreg_reduce_sum_muls, preg_full)
    vreg_reduce_sum_muls_tail = vf.div(
        vreg_reduce_sum_muls_tail, vreg_g_scale, preg_tail
    )
    vf.store_align(softmax_l1_tiles + tail_offset, vreg_reduce_sum_muls_tail, preg_full)


def muls_sel_vf_pse_type1_multi_tile(
    src_tiles,
    lse_tiles,
    softmax_l1_tiles,
    scale,
    src_m,
    s2_real_size,
    g_scalar=1.0,
):
    """
    Alternative VF kernel using pre-configured tiles per block.
    """
    full_exe_size = 64
    repeat_times = 0
    tail_size = s2_real_size
    if s2_real_size > full_exe_size:
        repeat_times = 1
        tail_size = s2_real_size - full_exe_size

    _muls_sel_vf_pse_type1_inner(
        src_tiles,
        lse_tiles,
        softmax_l1_tiles,
        scale,
        src_m,
        repeat_times,
        tail_size,
        g_scalar,
    )


@pl.pipeline.stage
def compute_softmax_l1_norm(
    softmax_l1_norm, qk_vec_db, lse_vec_slot, out_vec_db, const_info, run_info
):
    qk_vec_slot = qk_vec_db.next()
    out_vec_slot = out_vec_db.next()
    if const_info.current_g_size > 0:
        muls_sel_vf_pse_type1_multi_tile(
            qk_vec_slot,
            lse_vec_slot,
            out_vec_slot,
            const_info.softmax_scale,
            const_info.current_g_size,
            run_info.s2_base_size,
            const_info.g_scale,
        )

        pl.set_validshape(out_vec_slot, [1, run_info.s2_base_size])

        if IS_TND == 1:
            pl.store(
                softmax_l1_norm,
                out_vec_slot,
                [run_info.t1_offset + run_info.s1_idx, 0, run_info.ki],
                order=[1, 2],
                atomic=pl.AtomicType.AtomicAdd,
            )
        else:
            pl.store(
                softmax_l1_norm,
                out_vec_slot,
                [run_info.batch_id, run_info.s1_idx, 0, run_info.ki],
                order=[2, 3],
                atomic=pl.AtomicType.AtomicAdd,
            )


def get_seq_length(seqused_q, seq_q, seq_k, const_info, run_info, used_t1_index):
    if const_info.has_seqused_q:
        for batch_id in pl.range(run_info.batch_id, const_info.b_size):
            if run_info.next_used_t1_offset > used_t1_index:
                break

            run_info.batch_id = batch_id + 1
            next_batch_used_t1 = seqused_q[batch_id + 1]
            run_info.used_t1_offset = run_info.next_used_t1_offset
            run_info.next_used_t1_offset = (
                run_info.next_used_t1_offset + next_batch_used_t1
            )
        run_info.s1_idx = used_t1_index - run_info.used_t1_offset
        if IS_TND == 1:
            run_info.t1_offset = seq_q[run_info.batch_id]
            run_info.t2_offset = seq_k[run_info.batch_id]
            run_info.next_t1_offset = seq_q[run_info.batch_id + 1]
            run_info.s1_size = run_info.next_t1_offset - run_info.t1_offset
            s2_pre = seq_k[run_info.batch_id]
            s2_next = seq_k[run_info.batch_id + 1]
            run_info.s2_size = s2_next - s2_pre
    else:
        if IS_TND == 1:
            for batch_id in pl.range(run_info.batch_id, const_info.b_size):
                if run_info.next_t1_offset > used_t1_index:
                    break

                run_info.batch_id = batch_id + 1
                next_batch_t1 = seq_q[run_info.batch_id + 1]
                run_info.t1_offset = run_info.next_t1_offset
                run_info.next_t1_offset = next_batch_t1
            run_info.t2_offset = seq_k[run_info.batch_id]
            run_info.s1_idx = used_t1_index - run_info.t1_offset
            run_info.s1_size = run_info.next_t1_offset - run_info.t1_offset
            s2_pre = seq_k[run_info.batch_id]
            s2_next = seq_k[run_info.batch_id + 1]

            run_info.s2_size = s2_next - s2_pre
        else:
            run_info.batch_id = used_t1_index // const_info.s1_size
            run_info.s1_idx = used_t1_index % const_info.s1_size


def set_preload_info(run_info, preload_info, ki):
    preload_info.ki = ki
    if ki + TKV > run_info.s2_real_size:
        preload_info.s2_base_size = run_info.s2_real_size - ki
    else:
        preload_info.s2_base_size = TKV

    preload_info.used_t1_index = run_info.used_t1_index
    preload_info.batch_id = run_info.batch_id
    preload_info.s1_idx = run_info.s1_idx
    if IS_TND == 1:
        preload_info.t1_offset = run_info.t1_offset
        preload_info.t2_offset = run_info.t2_offset


@pl.jit(
    auto_mutex=True,
    tiling_key=SmlaTilingKey,
    datatype={
        "q": "input_dtype",
    },
)
def sparse_flash_mla_softmax_l1_norm(
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    sparse_indices_optional: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_q_optional: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_k_optional: pl.Ptr[pl.DT_UINT8],
    seqused_q_optional: pl.Ptr[pl.DT_UINT8],
    seqused_k_optional: pl.Ptr[pl.DT_UINT8],
    cmp_residual_k_optional: pl.Ptr[pl.DT_UINT8],
    topk_length_optional: pl.Ptr[pl.DT_UINT8],
    metadata_optional: pl.Ptr[pl.DT_UINT8],
    softmax_l1_norm: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: OpTiling,
):
    if IS_TND == 1:
        tensor_q = pl.make_tensor(
            q,
            [tiling.t1, tiling.g, tiling.d],
            [tiling.g * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_k = pl.make_tensor(
            k,
            [tiling.t2, 1, tiling.d],
            [tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [1, tiling.t1, tiling.g],
            [tiling.g, tiling.g, 1],
            dtype=pl.DT_FP32,
        )
        if IS_SPARSE == 1:
            tensor_softmax_l1_norm = pl.make_tensor(
                softmax_l1_norm,
                [tiling.t1, 1, tiling.k_length],
                [tiling.k_length, tiling.k_length, 1],
                dtype=pl.DT_FP32,
            )
            tensor_sparse_indices = pl.make_tensor(
                sparse_indices_optional,
                [tiling.t1, 1, tiling.k_length],
                [tiling.k_length, tiling.k_length, 1],
                dtype=pl.DT_INT32,
            )
            tensor_topk_length = pl.make_tensor(
                topk_length_optional,
                [tiling.t1, 1],
                [1, 1],
                dtype=pl.DT_INT32,
            )
            tensor_softmax_l1_norm_init = pl.make_tensor(
                softmax_l1_norm,
                [1, tiling.t1 * tiling.k_length],
                [tiling.t1 * tiling.k_length, 1],
                dtype=pl.DT_FP32,
            )
        else:
            tensor_softmax_l1_norm = pl.make_tensor(
                softmax_l1_norm,
                [tiling.t1, 1, tiling.max_seqlen_k],
                [tiling.max_seqlen_k, tiling.max_seqlen_k, 1],
                dtype=pl.DT_FP32,
            )
            tensor_softmax_l1_norm_init = pl.make_tensor(
                softmax_l1_norm,
                [1, tiling.t1 * 1 * tiling.max_seqlen_k],
                [tiling.t1 * 1 * tiling.max_seqlen_k, 1],
                dtype=pl.DT_FP32,
            )
    else:
        tensor_q = pl.make_tensor(
            q,
            [tiling.b, tiling.sq, tiling.g, tiling.d],
            [tiling.sq * tiling.g * tiling.d, tiling.g * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_k = pl.make_tensor(
            k,
            [tiling.b, tiling.sk, 1, tiling.d],
            [tiling.sk * tiling.d, tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [tiling.b, tiling.sq, 1, tiling.g],
            [tiling.sq * tiling.g, tiling.g, tiling.g, 1],
            dtype=pl.DT_FP32,
        )
        if IS_SPARSE == 1:
            tensor_softmax_l1_norm = pl.make_tensor(
                softmax_l1_norm,
                [tiling.b, tiling.sq, 1, tiling.k_length],
                [tiling.sq * tiling.k_length, tiling.k_length, tiling.k_length, 1],
                dtype=pl.DT_FP32,
            )
            tensor_sparse_indices = pl.make_tensor(
                sparse_indices_optional,
                [tiling.b, tiling.sq, 1, tiling.k_length],
                [tiling.sq * tiling.k_length, tiling.k_length, tiling.k_length, 1],
                dtype=pl.DT_INT32,
            )
            tensor_topk_length = pl.make_tensor(
                topk_length_optional,
                [tiling.b, tiling.sq, 1],
                [tiling.sq, 1, 1],
                dtype=pl.DT_INT32,
            )
            tensor_softmax_l1_norm_init = pl.make_tensor(
                softmax_l1_norm,
                [1, tiling.b * tiling.sq * tiling.k_length],
                [tiling.b * tiling.sq * tiling.k_length, 1],
                dtype=pl.DT_FP32,
            )
        else:
            tensor_softmax_l1_norm = pl.make_tensor(
                softmax_l1_norm,
                [tiling.b, tiling.sq, 1, tiling.sk],
                [tiling.sq * tiling.sk, tiling.sk, tiling.sk, 1],
                dtype=pl.DT_FP32,
            )
            tensor_softmax_l1_norm_init = pl.make_tensor(
                softmax_l1_norm,
                [1, tiling.b * tiling.sq * tiling.sk],
                [tiling.b * tiling.sq * tiling.sk, 1],
                dtype=pl.DT_FP32,
            )
    tensor_cu_seqlens_q = pl.make_tensor(
        cu_seqlens_q_optional,
        [tiling.b + 1],
        [1],
        dtype=pl.DT_INT32,
    )
    tensor_cu_seqlens_k = pl.make_tensor(
        cu_seqlens_k_optional,
        [tiling.b + 1],
        [1],
        dtype=pl.DT_INT32,
    )
    tensor_seqused_q = pl.make_tensor(
        seqused_q_optional,
        [tiling.b],
        [1],
        dtype=pl.DT_INT32,
    )
    tensor_seqused_k = pl.make_tensor(
        seqused_k_optional,
        [tiling.b],
        [1],
        dtype=pl.DT_INT32,
    )
    tensor_cmp_residual_k = pl.make_tensor(
        cmp_residual_k_optional,
        [tiling.b],
        [1],
        dtype=pl.DT_INT32,
    )
    tensor_metadata = pl.make_tensor(
        metadata_optional,
        [64],
        [1],
        dtype=pl.DT_INT32,
    )

    total_num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    sub_id = pl.get_subblock_idx()

    has_seqused_q = tiling.has_seqused_q
    has_seqused_k = tiling.has_seqused_k
    has_topk_length = tiling.has_topk_length

    sq_tiles = tensor_metadata[0]
    num_cores = tensor_metadata[4]

    first_half_g = (tiling.g + 2 - 1) // 2
    second_half_g = tiling.g - first_half_g
    current_half_g = first_half_g
    if sub_id == 1:
        current_half_g = second_half_g

    const_info = pl.make_tuple(
        b_size=tiling.b,
        g_size=tiling.g,
        g_scale=tiling.g * 1.0,
        s1_size=tiling.sq,
        s2_size=tiling.sk,
        is_sparse=IS_SPARSE,
        k_length=tiling.k_length,
        mask_mode=SPARSE_MODE,
        cmp_ratio=tiling.cmp_ratio,
        is_tnd=IS_TND,
        has_seqused_q=tiling.has_seqused_q,
        has_seqused_k=tiling.has_seqused_k,
        first_g_size=first_half_g,
        current_g_size=current_half_g,
        softmax_scale=tiling.softmax_scale,
        softmax_l1_norm_scale=1.0 / tiling.g,
        sub_id=sub_id,
        core_id=core_id,
        num_cores=num_cores,
    )
    run_infos = pl.struct_array(
        1,
        "run_info",
        batch_id=0,
        s1_idx=0,
        s1_size=tiling.sq,
        s2_size=tiling.sk,
        used_t1_index=0,
        used_t1_offset=0,
        next_used_t1_offset=0,
        t1_offset=0,
        next_t1_offset=0,
        t2_offset=0,
        ki=0,
        s2_real_size=0,
        s2_base_size=TKV,
        task_id=0,
    )
    preload_infos = pl.struct_array(
        2,
        "PreloadRunInfo",
        batch_id=0,
        s1_idx=0,
        s1_size=tiling.sq,
        s2_size=tiling.sk,
        used_t1_index=0,
        used_t1_offset=0,
        next_used_t1_offset=0,
        t1_offset=0,
        next_t1_offset=0,
        t2_offset=0,
        ki=0,
        s2_real_size=0,
        s2_base_size=TKV,
        task_id=0,
    )
    run_info = run_infos[0]
    if has_seqused_q:
        run_info.next_used_t1_offset = tensor_seqused_q[0]
    if IS_TND:
        run_info.next_t1_offset = tensor_cu_seqlens_q[1]

    # common
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TG_HALF, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=VA0,
        mutex_ids=[0, 1],
        fwd_ids=QK_READY_FORWARD_IDS,
        bwd_ids=QK_READY_BACKWARD_IDS,
    )
    if IS_SPARSE == 1:
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[D_TOTAL, TKV],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=MA1,
            mutex_ids=[12, 13],
            fwd_ids=GATHER_READY_FORWARD_IDS,
            bwd_ids=GATHER_READY_BACKWARD_IDS,
        )
    with pl.section_cube():
        if IS_SPARSE == 0:
            k_l1_db = pl.make_tile_group(
                type=pl.TileType(
                    shape=[TD, TKV],
                    dtype=input_dtype,
                    target_memory=pl.MemorySpace.Mat,
                    layout=pl.ZN,
                    valid_shape=[-1, -1],
                    compact=1,
                ),
                addrs=MA1,
                mutex_ids=[12, 13],
            )
        q_mat_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TG, D_TOTAL],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=MA0,
            mutex_ids=[2, 3],
        )
        left_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TG, TD],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Left,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=LA0,
            mutex_ids=[6, 7],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TKV],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Right,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=RA0,
            mutex_ids=[8, 9],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TG, TKV],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=CA0,
            mutex_ids=[10, 11],
        )

    with pl.section_vector():
        # vec
        gather_nd_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[GATHER_ROW_NUM, D_TOTAL],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=VA1,
            mutex_ids=[2, 3],
        )
        gather_nz_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[GATHER_ROW_NUM, D_TOTAL],
                dtype=input_dtype,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=VA4,
            mutex_ids=[8, 9],
        )
        lse_vec_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, TG], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=VA2,
            mutex_ids=[4, 5],
        )
        out_vec_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, TKV], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=VA3,
            mutex_ids=[6, 7],
        )
    tick = 0
    q_count = 0
    with pl.section_vector():
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BACKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BACKWARD_IDS[1])
    with pl.section_cube():
        if IS_SPARSE == 1:
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1, event_id=GATHER_READY_BACKWARD_IDS[0]
            )
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE1, event_id=GATHER_READY_BACKWARD_IDS[1]
            )

    # init output start
    init_start_offset = tiling.init_per_core_num * (core_id * 2 + sub_id)
    init_end_offset = init_start_offset + tiling.init_per_core_num
    init_end_offset = (
        tiling.init_total_num
        if init_end_offset > tiling.init_total_num
        else init_end_offset
    )
    init_per_loop_num = 8 * 1024
    init_num = init_per_loop_num
    with pl.section_vector():
        # vec
        init_src_tile_type = pl.TileType(
            shape=[1, init_per_loop_num],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            pad=pl.TilePad.zero,
        )
        init_dst_tile_type = pl.TileType(
            shape=[1, init_per_loop_num],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            pad=pl.TilePad.zero,
        )
        init_ub_src = pl.make_tile_group(
            type=init_src_tile_type, addrs=VA1, mutex_ids=[2]
        )
        init_ub_dst = pl.make_tile_group(
            type=init_dst_tile_type, addrs=VA1, mutex_ids=[3]
        )
        init_ub_src_slot = init_ub_src.current()
        init_ub_dst_slot = init_ub_dst.current()
        pl.set_validshape(init_ub_src_slot, [0, 0])
        pl.fillpad(init_ub_dst_slot, init_ub_src_slot)
        for init_offset in pl.range(
            init_start_offset, init_end_offset, init_per_loop_num
        ):
            if init_offset + init_per_loop_num > init_end_offset:
                init_num = init_end_offset - init_offset
            pl.set_validshape(init_ub_dst_slot, [1, init_num])
            pl.store(tensor_softmax_l1_norm_init, init_ub_dst_slot, [0, init_offset])

        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)
    # init output end

    _pl_sync_id = 0
    for used_t1_index in pl.range(core_id, sq_tiles, num_cores):
        get_seq_length(
            tensor_seqused_q,
            tensor_cu_seqlens_q,
            tensor_cu_seqlens_k,
            const_info,
            run_info,
            used_t1_index,
        )

        ### 计算s2_real_size
        current_s1_size = get_seqused(
            run_info.s1_size, has_seqused_q, tensor_seqused_q, run_info.batch_id
        )
        current_s2_size = get_seqused(
            run_info.s2_size, has_seqused_k, tensor_seqused_k, run_info.batch_id
        )
        s1_idx = run_info.s1_idx
        if IS_SPARSE == 0:
            if SPARSE_MODE == 0:
                s2_real_size = current_s2_size
            else:
                if const_info.cmp_ratio > 1:
                    ori_s2_size = (
                        current_s2_size * const_info.cmp_ratio
                        + tensor_cmp_residual_k[run_info.batch_id]
                    )
                else:
                    ori_s2_size = current_s2_size * const_info.cmp_ratio
                s2_real_size = pl.max(
                    (ori_s2_size - current_s1_size + s1_idx + 1)
                    // const_info.cmp_ratio,
                    0,
                )
        else:
            cur_k_length = const_info.k_length
            if has_topk_length:
                if IS_TND == 1:
                    cur_k_length = min(
                        cur_k_length,
                        tensor_topk_length[run_info.t1_offset + run_info.s1_idx, 0],
                    )
                else:
                    cur_k_length = min(
                        cur_k_length,
                        tensor_topk_length[run_info.batch_id, run_info.s1_idx, 0],
                    )
            if SPARSE_MODE == 0:
                s2_real_size = pl.min(cur_k_length, current_s2_size)
            else:
                if const_info.cmp_ratio > 1:
                    ori_s2_size = (
                        current_s2_size * const_info.cmp_ratio
                        + tensor_cmp_residual_k[run_info.batch_id]
                    )
                else:
                    ori_s2_size = current_s2_size * const_info.cmp_ratio
                s2_valid_size = pl.max(
                    (ori_s2_size - current_s1_size + s1_idx + 1)
                    // const_info.cmp_ratio,
                    0,
                )
                s2_real_size = pl.min(cur_k_length, s2_valid_size)

        run_info.s2_real_size = s2_real_size
        if run_info.s2_real_size <= 0:
            continue
        run_info.used_t1_index = used_t1_index

        with pl.section_cube():
            cur_q_slot = q_mat_db.next()
            pl.set_validshape(cur_q_slot, [const_info.g_size, D_TOTAL])
        with pl.section_vector():
            lse_vec_slot = lse_vec_db.next()
            if current_half_g > 0 and run_info.s2_real_size > 0:
                pl.set_validshape(lse_vec_slot, [1, current_half_g])
                if IS_TND == 1:
                    pl.load(
                        lse_vec_slot,
                        tensor_lse,
                        [
                            0,
                            run_info.t1_offset + run_info.s1_idx,
                            const_info.sub_id * const_info.first_g_size,
                        ],
                        order=[1, 2],
                    )
                else:
                    pl.load(
                        lse_vec_slot,
                        tensor_lse,
                        [
                            run_info.batch_id,
                            0,
                            run_info.s1_idx,
                            const_info.sub_id * const_info.first_g_size,
                        ],
                        order=[1, 3],
                    )

        for ki in pl.range(0, run_info.s2_real_size, TKV):
            cur_loop_preload_info = preload_infos[tick % 2]
            last_loop_preload_info = preload_infos[(tick + 1) % 2]
            set_preload_info(run_info, cur_loop_preload_info, ki)
            if IS_SPARSE == 1:
                with pl.section_vector():
                    if IS_SPARSE == 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE3,
                            event_id=GATHER_READY_BACKWARD_IDS[_pl_sync_id % 2],
                        )
                        gather_k(
                            k_l1_db,
                            tensor_k,
                            tensor_sparse_indices,
                            gather_nd_db,
                            gather_nz_db,
                            const_info,
                            cur_loop_preload_info,
                        )
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.MTE3,
                            event_id=GATHER_READY_FORWARD_IDS[_pl_sync_id % 2],
                        )
            with pl.section_cube():
                if IS_SPARSE == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE1,
                        event_id=GATHER_READY_FORWARD_IDS[_pl_sync_id % 2],
                    )
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=QK_READY_BACKWARD_IDS[_pl_sync_id % 2],
                    )
                    compute_qk(
                        tensor_q,
                        tensor_k,
                        cur_q_slot,
                        k_l1_db,
                        left_db,
                        right_db,
                        acc_db,
                        qk_vec_db,
                        const_info,
                        cur_loop_preload_info,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.MTE1,
                        event_id=GATHER_READY_BACKWARD_IDS[_pl_sync_id % 2],
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=QK_READY_FORWARD_IDS[_pl_sync_id % 2],
                    )
                else:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=QK_READY_BACKWARD_IDS[_pl_sync_id % 2],
                    )
                    compute_qk_dense(
                        tensor_q,
                        tensor_k,
                        cur_q_slot,
                        k_l1_db,
                        left_db,
                        right_db,
                        acc_db,
                        qk_vec_db,
                        const_info,
                        cur_loop_preload_info,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=QK_READY_FORWARD_IDS[_pl_sync_id % 2],
                    )
            if tick > 0:
                with pl.section_vector():
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.V,
                        event_id=QK_READY_FORWARD_IDS[(_pl_sync_id + 1) % 2],
                    )
                    if last_loop_preload_info.used_t1_index == run_info.used_t1_index:
                        cur_lse_vec_slot = lse_vec_db.current()
                        compute_softmax_l1_norm(
                            tensor_softmax_l1_norm,
                            qk_vec_db,
                            cur_lse_vec_slot,
                            out_vec_db,
                            const_info,
                            last_loop_preload_info,
                        )
                    else:
                        cur_lse_vec_slot = lse_vec_db.previous()
                        compute_softmax_l1_norm(
                            tensor_softmax_l1_norm,
                            qk_vec_db,
                            cur_lse_vec_slot,
                            out_vec_db,
                            const_info,
                            last_loop_preload_info,
                        )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.V,
                        event_id=QK_READY_BACKWARD_IDS[(_pl_sync_id + 1) % 2],
                    )
            tick = tick + 1
            _pl_sync_id = _pl_sync_id + 1
        q_count = q_count + 1

    if tick > 0:
        cur_loop_preload_info = preload_infos[tick % 2]
        last_loop_preload_info = preload_infos[(tick + 1) % 2]
        with pl.section_vector():
            pl.system.wait_cross_core(
                pipe=pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[(_pl_sync_id + 1) % 2]
            )
            if last_loop_preload_info.used_t1_index == run_info.used_t1_index:
                cur_lse_vec_slot = lse_vec_db.current()
                compute_softmax_l1_norm(
                    tensor_softmax_l1_norm,
                    qk_vec_db,
                    cur_lse_vec_slot,
                    out_vec_db,
                    const_info,
                    last_loop_preload_info,
                )
            else:
                cur_lse_vec_slot = lse_vec_db.previous()
                compute_softmax_l1_norm(
                    tensor_softmax_l1_norm,
                    qk_vec_db,
                    cur_lse_vec_slot,
                    out_vec_db,
                    const_info,
                    last_loop_preload_info,
                )
            pl.system.set_cross_core(
                pipe=pl.PipeType.V,
                event_id=QK_READY_BACKWARD_IDS[(_pl_sync_id + 1) % 2],
            )

    with pl.section_cube():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=QK_READY_BACKWARD_IDS[0]
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=QK_READY_BACKWARD_IDS[1]
        )
    with pl.section_vector():
        if IS_SPARSE == 1:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3, event_id=GATHER_READY_BACKWARD_IDS[0]
            )
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE3, event_id=GATHER_READY_BACKWARD_IDS[1]
            )
