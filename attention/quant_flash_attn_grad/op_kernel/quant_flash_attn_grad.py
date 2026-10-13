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
ATTEN_MASK_COMPRESS_SIZE = 2048
ATTEN_MASK_MIN = -3.4028234663852886e38

BASE_K = 128
BASE_N = 128
K_SIZE = 256
QUANT_S1_BASE_COUNT = 64
QUANT_S2_BASE_COUNT = 8
BLOCK_SIZE = 32
# ---- VEC addresses ----
CUBE_BUFFER_NUM = 2
VEC_BUFFER_NUM = 2
VB4_KV = TS_HALF * TS * 4 * VEC_BUFFER_NUM
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

# ================================================================
#  metadata 第二维 (row 1) 槽位
#  必须与 quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata.h
#  中的 QUANT_FAG_*_INDEX 保持一致。
# ================================================================
META_DETER_MAX_NUM_IDX = 0
META_NEED_INIT_OUTPUT_IDX = 1
META_BATCH_SIZE_IDX = 2
META_SCHEDULE_VALID_IDX = 3
META_AIC_CORE_NUM_IDX = 4  # Must match the actual mixed-kernel AIC group count.
# 三个 per-batch 数组的起点由 metadata 算子按 batch 动态排布后写在这几个标量槽位里,
# kernel 直接读起点, 不重复推导排布公式(否则两侧公式一旦不同步就是静默错读)。
META_ROUND_PREFIX_OFFSET_IDX = 5
META_S1_OUTER_OFFSET_IDX = 6
META_S2_OUTER_OFFSET_IDX = 7
META_ROW_BEGIN_OFFSET_IDX = 8
META_COL_BEGIN_OFFSET_IDX = 9
META_SCHEDULE_ROWS_OFFSET_IDX = 10
META_SCHEDULE_COLS_OFFSET_IDX = 11
META_S1_TOKEN_OFFSET_IDX = 12
META_S2_TOKEN_OFFSET_IDX = 13
META_SCHEDULE_KIND_IDX = 14
# TND line swizzle 时槽8为mode；dense swizzle 时槽8为 ROW_BEGIN 偏移
META_SCHEDULE_MODE_IDX = 8
META_TND_LINE_MODE = 1
META_TND_LINE_M_OFFSET_IDX = 9
META_TND_LINE_N_OFFSET_IDX = 10
META_TND_LINE_P_OFFSET_IDX = 11
META_TND_LINE_Q_OFFSET_IDX = 12
META_TND_LINE_RUN_SIZE_OFFSET_IDX = 13
META_DENSE_SWIZZLE = 2
META_TND_LINE_SWIZZLE = 3

init_data = [1] * 16


def _align_up(value, align=1024):
    return ((value + align - 1) // align) * align


def _abs(value):
    return value if value > 0 else value * -1


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
VA4 = VA3 + TS * HALF_D_SIZE * 4
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
    metadata_len: int
    mask_mode: int
    s1_token: int
    s2_token: int
    deter_max_round: int

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
    has_attn_mask = TilingKeyField(bits=1, values=[0, 1])
    has_sink = TilingKeyField(bits=1, values=[0])
    s1_template_num = TilingKeyField(bits=1, values=[512])
    s2_template_num = TilingKeyField(bits=1, values=[512])
    d_template_num = TilingKeyField(bits=1, values=[128])
    is_n_equal = TilingKeyField(bits=1, values=[1])
    # 0: BSND 1: BNSD 2: TND
    layout = TilingKeyField(bits=2, values=[0, 1, 2])
    # DeterSparseType: 0: DENSE 3: CAUSAL 4: BAND
    sparse_type = TilingKeyField(bits=2, values=[0, 3, 4])
    # seqused_q / seqused_kv: 每 batch 实际使用的序列长度 (<= 定长 S)。
    has_seq_used_q = TilingKeyField(bits=1, values=[0, 1])
    has_seq_used_kv = TilingKeyField(bits=1, values=[0, 1])

    def is_valid(self, key):
        return True


def l1_ds(i, j, task_mod2):
    return i + j * 4 if task_mod2 == 0 else i * 4 + j


def is_var_len():
    return layout == 2 or has_seq_used_q == 1 or has_seq_used_kv == 1


def qk_s1_fuse():
    return False


def is_fused_s1_pair(run_info, s1_idx):
    return (
        s1_idx % 2 == 0
        and (s1_idx + 1) < run_info.inner_s1_loop_num
        and run_info.inner_s1_real_size[s1_idx] == 128
        and run_info.inner_s1_real_size[s1_idx + 1] == 128
    )


def meta_get(const_info, tensor_metadata, idx):
    return pl.getval(tensor_metadata, const_info.metadata_len + idx)


def get_actual_s1_len(const_info, tensor_seq_q, b_idx):
    if has_seq_used_q == 1:
        return pl.getval(tensor_seq_q, b_idx)
    if layout == 2:
        return pl.getval(tensor_seq_q, b_idx + 1) - pl.getval(tensor_seq_q, b_idx)
    return const_info.s1_size


def get_actual_s2_len(const_info, tensor_seq_kv, b_idx):
    if has_seq_used_kv == 1:
        return pl.getval(tensor_seq_kv, b_idx)
    if layout == 2:
        return pl.getval(tensor_seq_kv, b_idx + 1) - pl.getval(tensor_seq_kv, b_idx)
    return const_info.s2_size


def cal_deter_max_loop_num(const_info, band_info):
    if sparse_type == 4:
        return max(const_info.deter_max_round, band_info.rm2)
    if sparse_type == 3:
        return const_info.deter_max_round
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
    b_idx = start_idx // (
        const_info.n1_size * const_info.gm_s1_size * const_info.d_size
    )
    b_tail = start_idx % (
        const_info.n1_size * const_info.gm_s1_size * const_info.d_size
    )
    n_idx = b_tail // (const_info.gm_s1_size * const_info.d_size)
    n_tail = b_tail % (const_info.gm_s1_size * const_info.d_size)
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
        vf.store(out_vec, vreg_res, 1)


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
        if layout == 2:
            remaining = const_info.q_post_block_total - dq_offset
            init_dq_size = min(const_info.q_pre_block_factor, max(remaining, 0))
        for i in pl.range(0, init_dq_size, 128 * 128):
            size = TS * TS if init_dq_size - i > TS * TS else init_dq_size - i
            pl.set_validshape(dq_tile, [1, size])
            pl.store(
                tensor_info.tensor_workspace_dq_flat,
                dq_tile.current(),
                offsets=[0, dq_offset + i],
            )
        if is_var_len() or sparse_type == 3 or sparse_type == 4:
            # Sparse schedules can revisit a head/KV column after switching away.
            # Zero the full workspace before additive writeback; varlen also needs
            # zero padding for columns that are never visited.
            init_dk_size = (
                const_info.k_pre_block_tail
                if const_info.core_id_vec == const_info.k_pre_block_total - 1
                else const_info.k_pre_block_factor
            )
            dk_offset = const_info.core_id_vec * const_info.k_pre_block_factor
            if const_info.core_id_vec < const_info.k_pre_block_total:
                for i in pl.range(0, init_dk_size, 128 * 128):
                    size = TS * TS if init_dk_size - i > TS * TS else init_dk_size - i
                    pl.set_validshape(dq_tile, [1, size])
                    pl.store(
                        tensor_info.tensor_workspace_dk_flat,
                        dq_tile.current(),
                        offsets=[0, dk_offset + i],
                    )
            init_dv_size = (
                const_info.v_pre_block_tail
                if const_info.core_id_vec == const_info.v_pre_block_total - 1
                else const_info.v_pre_block_factor
            )
            dv_offset = const_info.core_id_vec * const_info.v_pre_block_factor
            if const_info.core_id_vec < const_info.v_pre_block_total:
                for i in pl.range(0, init_dv_size, 128 * 128):
                    size = TS * TS if init_dv_size - i > TS * TS else init_dv_size - i
                    pl.set_validshape(dq_tile, [1, size])
                    pl.store(
                        tensor_info.tensor_workspace_dv_flat,
                        dq_tile.current(),
                        offsets=[0, dv_offset + i],
                    )


def init_sfmg_workspace(const_info, tensor_info):
    with pl.section_vector():
        if const_info.core_id_vec < const_info.sfmg_used_core_num:
            zero_tt = pl.TileType(
                shape=[1, TS * TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
            )
            zero_tile = pl.make_tile_group(type=zero_tt, addrs=VAQ_PRE, mutex_ids=[7])
            pl.expands(zero_tile.current(), 0)
            sfmg_total = (
                const_info.gm_b_size * const_info.n1_size * const_info.gm_s1_size
            )
            sfmg_factor = ceil(sfmg_total, const_info.sfmg_used_core_num)
            sfmg_begin = const_info.core_id_vec * sfmg_factor
            sfmg_end = sfmg_begin + sfmg_factor
            if sfmg_end > sfmg_total:
                sfmg_end = sfmg_total
            if sfmg_begin < sfmg_total:
                for i in pl.range(sfmg_begin, sfmg_end, TS * TS):
                    size = TS * TS if sfmg_end - i > TS * TS else sfmg_end - i
                    pl.set_validshape(zero_tile, [1, size])
                    pl.store(
                        tensor_info.tensor_workspace_sfmg_flat,
                        zero_tile.current(),
                        offsets=[0, i],
                    )


def presfmg_quant_tnd_used_hif8(const_info, tensor_info):
    with pl.section_vector():
        init_dq_workspace(const_info, tensor_info)
        chunks_per_head = ceil(const_info.s1_size, TS)
        head_count = const_info.n1_size
        task_count = const_info.b_size * head_count * chunks_per_head
        core_id = const_info.core_id_vec
        core_count = const_info.sfmg_used_core_num
        cu_q = tensor_info.tensor_cu_q
        used_q = tensor_info.tensor_seq_q
        attn_out = tensor_info.tensor_attn_out
        dy = tensor_info.tensor_do
        sfmg = tensor_info.tensor_workspace_sfmg
        deq_scale_do_value = const_info.deq_scale_do_value
        y_tile = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, D_SIZE], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=VA0_PRE,
            mutex_ids=[0, 1],
        )
        dx_tile = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, D_SIZE], dtype=pl.DT_HF8, target_memory=pl.MemorySpace.Vec
            ),
            addrs=VA1_PRE,
            mutex_ids=[2, 3],
        )
        out_tile = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, TS], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=VA2_PRE,
            mutex_ids=[4, 5],
        )
        if core_id < core_count:
            for task in pl.range(core_id, task_count, core_count):
                local_q = (task % chunks_per_head) * TS
                batch_head = task // chunks_per_head
                batch_id = batch_head // head_count
                head_id = batch_head % head_count
                effective_q = pl.getval(used_q, batch_id)
                if local_q < effective_q:
                    rows = min(TS, effective_q - local_q)
                    global_q = pl.getval(cu_q, batch_id) + local_q
                    y_vec = y_tile.next()
                    dx_vec = dx_tile.next()
                    out_vec = out_tile.next()
                    pl.set_validshape(y_vec, [rows, D_SIZE])
                    pl.set_validshape(dx_vec, [rows, D_SIZE])
                    pl.load(y_vec, attn_out, [0, global_q, head_id, 0], order=[1, 3])
                    pl.load(dx_vec, dy, [0, global_q, head_id, 0], order=[1, 3])
                    anti_quant_softmax_grad_front_cast_hif8_vf(
                        rows, deq_scale_do_value, y_vec, dx_vec, out_vec
                    )
                    pl.set_validshape(out_vec, [1, rows])
                    pl.store(sfmg, out_vec, [0, head_id, global_q], order=[1, 2])


def presfmg_quant_inner_hif8(const_info, tensor_info):
    if layout == 2 and has_seq_used_q == 1:
        presfmg_quant_tnd_used_hif8(const_info, tensor_info)
        return
    with pl.section_vector():
        init_dq_workspace(const_info, tensor_info)
        n_burst = const_info.single_loop_nburst_num
        num_of_128_in_s1 = (const_info.gm_s1_size + n_burst - 1) // n_burst
        s1_residual = (
            const_info.gm_s1_size % n_burst
            if const_info.gm_s1_size % n_burst != 0
            else n_burst
        )

        num_d_chunks = (const_info.d_size + VEC_CHUNK - 1) // VEC_CHUNK
        tail_d_chunk = (
            const_info.d_size % VEC_CHUNK
            if const_info.d_size % VEC_CHUNK != 0
            else VEC_CHUNK
        )

        deq_scale_do_value = const_info.deq_scale_do_value
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
            is_row_valid = True
            if has_seq_used_q == 1:
                s1_remain = pl.getval(tensor_info.tensor_seq_q, b_idx) - s1_idx
                is_row_valid = s1_remain > 0
                if cur_n_burst > s1_remain:
                    cur_n_burst = s1_remain if s1_remain > 0 else 1
            pl.set_validshape(y_vec, [cur_n_burst, D_SIZE])
            pl.set_validshape(dx_vec, [cur_n_burst, D_SIZE])
            if layout == 0 or layout == 2:  # BSND
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
            if is_row_valid:
                pl.store(
                    tensor_info.tensor_workspace_sfmg,
                    out_vec,
                    [b_idx, n_idx, s1_idx],
                    order=[1, 2],
                )


def init_coordinate_info(const_info, m_offset, n_offset, coordinate_info):
    coordinate_info.s1_outer = const_info.s1_outer
    coordinate_info.s2_outer = const_info.s2_outer
    coordinate_info.s1_token = const_info.s1_token
    coordinate_info.s2_token = const_info.s2_token
    coordinate_info.actual_s1_len = const_info.s1_size
    coordinate_info.actual_s2_len = const_info.s2_size
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


def deter_empty_round_barrier():
    """Pass through one dQ uniqueness barrier without writing.

    Swizzle uniqueness is per outer 512 tile (s1o), not per inner 128 slice.
    FAG does one INTER_BLOCK wait+set per round; matching that keeps idle
    cores in lockstep without 4x bubbles on every 128 writeback.
    """
    with pl.section_vector():
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


def _isqrt(x):
    # floor(sqrt(x)) for nonnegative int64 x. Keep a single loop-carried
    # scalar to avoid PyPTO overwriting the old res/nxt state on the backedge.
    if x < 2:
        return x
    res = x // 2 + 1
    while res > x // res:
        res = (res + x // res) // 2
    return res


def _ceil_trunc(a, b):
    num = a + b - 1
    q = num // b
    if num % b != 0 and num < 0:
        return q + 1
    return q


def cal_causal_g2k_single_batch_deter_index(k, j, a, l1, offset, coordinate):
    if j % 2 == 1:
        if a <= l1 - j + 1:
            y = j + offset
            x = y + a - 1
        else:
            y = 2 * k + 1 - j + offset
            x = y + 2 * l1 - 2 * k + 1 - a
    else:
        if a >= l1 - 2 * k + 1 + j:
            y = j + offset
            x = y + 2 * l1 - 2 * k + 1 - a
        else:
            y = 2 * k + 1 - j + offset
            x = y + a - 1
    coordinate.batch_id = 0
    coordinate.s1_idx = x
    coordinate.s2_idx = y


def cal_causal_no_rec_single_batch_deter_index(k, m, n, j, r, coordinate):
    coordinate.batch_id = -1
    if j > (n // 2) + 1:
        return
    if j % 2 == 1:
        if r + j <= n + 1:
            x = r + j - 1
            y = j
        else:
            x = 2 * n + 2 - j - r
            y = n + 3 - j - (n % 2)
    else:
        if j <= r + 1 - (n % 2):
            x = n + j - r - 1 + n % 2
            y = j
        else:
            x = n + 2 + r - j - (n % 2)
            y = n + 3 - j - (n % 2)
    if y >= 1 and y <= m and y <= x and x <= m:
        coordinate.batch_id = 0
        coordinate.s1_idx = x
        coordinate.s2_idx = y


def cal_causal_rec_single_batch_deter_index(k, m, n, j, r, coordinate):
    if 2 * k < m + 1 and k < n:
        cal_causal_g2k_single_batch_deter_index(k, j, r, m, 0, coordinate)
        x = coordinate.s1_idx
        y = coordinate.s2_idx
    else:
        cal_causal_no_rec_single_batch_deter_index(k, m, m, j, r, coordinate)
        if coordinate.batch_id != -1:
            x = coordinate.s1_idx
            y = coordinate.s2_idx
        else:
            x = m + 1
            y = m + 1
    if y >= 1 and y <= n and y <= x and x <= m:
        coordinate.batch_id = 0
        coordinate.s1_idx = x
        coordinate.s2_idx = y
    else:
        coordinate.batch_id = -1


def cal_causal_single_batch_deter_index(k, m, n, j, r, coordinate):
    if k >= (n // 2) + 1:
        cal_causal_rec_single_batch_deter_index(k, m, n, j, r, coordinate)
        return
    ell = n % k
    t1 = n // (2 * k)
    t3 = (n // k) % 2
    bound1 = (2 * m + 1) * t1 - 2 * k * t1 * t1
    if r <= bound1:
        disc = (2 * m + 1) * (2 * m + 1) - 8 * k * r
        i = ceil(2 * m + 1 - _isqrt(disc), 4 * k)
        rm = (2 * m + 1) * (i - 1) - 2 * k * (i - 1) * (i - 1)
        offset = 2 * k * (i - 1)
        l1 = m - 2 * k * (i - 1)
        a = r - rm
        cal_causal_g2k_single_batch_deter_index(k, j, a, l1, offset, coordinate)
        return
    rm = bound1
    offset = 2 * k * t1
    rem = t3 * k + ell
    coordinate.batch_id = -1
    if rem > 0:
        l1 = m - n + rem
        a = r - rm
        if t3 == 0 and j <= rem:
            coordinate.batch_id = 0
            coordinate.s2_idx = offset + j
            coordinate.s1_idx = coordinate.s2_idx + a - 1
        elif t3 == 1:
            cal_causal_rec_single_batch_deter_index(k, l1, l1, j, a, coordinate)
            if coordinate.batch_id != -1:
                shift = 2 * k * t1
                coordinate.s1_idx += shift
                coordinate.s2_idx += shift
        return


def cal_causal_index(k, m, n, b, j, r, coordinate):
    b1 = b // k
    b2 = b % k
    delta = m - n
    size_tri = n * (m + delta + 1) // 2
    rm1 = b1 * size_tri
    if r <= rm1:
        a = r % size_tri
        a = a if a != 0 else size_tri
        w = ceil(r, size_tri) + b1 * (j - 1)
        n1_even = n // 2 * 2
        l_len = 2 * m - n1_even + 1
        rm_local = n1_even * l_len // 2
        if a <= rm_local:
            y = ceil(a, l_len)
            r_rem = a % l_len
            r_rem = r_rem if r_rem != 0 else l_len
            x = r_rem + y - 1
            if x > m:
                x = 2 * m + 1 - x
                y = n1_even + 1 - y
        else:
            a1 = a - rm_local
            y = n
            x = a1 - 1 + y
        w = ((w - 1) % b1) * k + (w - 1) // b1 + 1
        w = ((w - 1) // k) * k + ((y - 1 + (w - 1)) % k) + 1
        coordinate.batch_id = w
        coordinate.s1_idx = x
        coordinate.s2_idx = y
        return
    t = n // k
    ell = n % k
    a2 = r - rm1
    half_ = b2 // 2
    rm2 = (2 * m - t * k + 1) * t * half_
    if a2 >= 1 and a2 <= rm2:
        cal_dense_index(k, 2 * m - t * k + 1, t * k, half_, j, a2, coordinate)
        w_sub = coordinate.batch_id
        x_sub = coordinate.s1_idx
        y_sub = coordinate.s2_idx
        max_x = 2 * m - t * k + 1
        max_y = t * k
        max_w = half_
        if (
            x_sub >= 1
            and x_sub <= max_x
            and y_sub >= 1
            and y_sub <= max_y
            and w_sub >= 1
            and w_sub <= max_w
        ):
            if x_sub - y_sub <= m - t * k:
                w = 2 * w_sub - 1 + b1 * k
                x = m + 1 - x_sub
                y = t * k + 1 - y_sub
            else:
                w = 2 * w_sub + b1 * k
                x = x_sub - m + t * k - 1
                y = y_sub
            coordinate.batch_id = w
            coordinate.s1_idx = x
            coordinate.s2_idx = y
        else:
            coordinate.batch_id = -1
        return
    a3 = r - rm1 - rm2
    rm3 = 0
    if b2 % 2 == 1:
        t1 = n // (2 * k)
        t3 = t % 2
        if t3 == 1:
            m1 = m - t1 * 2 * k
            rm3 = (m + m1 + 1) * t1
            if ell == 0:
                rm3 += m1
            else:
                rm3 += max(m1, 2 * m1 - 2 * k + 1)
            if a3 >= 1 and a3 <= rm3:
                cal_causal_single_batch_deter_index(k, m, n, j, a3, coordinate)
                if coordinate.batch_id != -1:
                    coordinate.batch_id = b
                return
            b -= 1
            b2 -= 1
        else:
            rm3 = (2 * m - t * k + 1) * t // 2
            if a3 >= 1 and a3 <= rm3:
                cal_causal_single_batch_deter_index(k, m, t * k, j, a3, coordinate)
                if coordinate.batch_id != -1:
                    coordinate.batch_id = b
                return
    a4 = a3 - rm3
    p = ceil(ell, 2)
    ell1 = ell + 1 - (ell % 2)
    block = b2 * p // k
    res0 = b2 * p % k
    if a4 > block * (ell1 + 2 * delta) and res0 <= k // 2:
        offs = a4 - block * (ell1 + 2 * delta)
        if j >= 1 and j <= res0:
            limit = (res0 - j) // b2 + ceil(ell1, 2) + delta
            if offs <= limit:
                w = (k * ((a4 - 1) // (ell1 + 2 * delta)) + j) % b2
                w = w if w != 0 else b2
                y = p - (res0 - j) // b2
                x = y + offs - 1
                if y >= 1 and y <= ell and y <= x and x <= ell + delta:
                    coordinate.batch_id = b1 * k + w
                    coordinate.s1_idx = x + t * k
                    coordinate.s2_idx = y + t * k
                    return
        if k - res0 + 1 <= j and j <= k:
            idx = j - (k - res0 + 1)
            limit = (ell1 - 1) // 2 - idx // b2 + delta
            if offs <= limit:
                w = (k * ((a4 - 1) // (ell1 + 2 * delta)) + k + 1 - j) % b2
                w = w if w != 0 else b2
                y = p + 1 + idx // b2
                x = y + offs - 1
                if y >= 1 and y <= ell and y <= x and x <= ell + delta:
                    coordinate.batch_id = b1 * k + w
                    coordinate.s1_idx = x + t * k
                    coordinate.s2_idx = y + t * k
                    return
        coordinate.batch_id = -1
        return
    w = (k * ((a4 - 1) // (ell1 + 2 * delta)) + j) % b2
    w = w if w != 0 else b2
    g = ceil(k * ((a4 - 1) // (ell1 + 2 * delta)) + j, b2)
    if g >= 1 and g <= p:
        a5 = a4 % (ell1 + 2 * delta)
        a5 = a5 if a5 != 0 else ell1 + 2 * delta
        if g % 2 == 1:
            if a5 <= ell - g + 1 + delta:
                x0 = g + a5 - 1
                y0 = g
            else:
                x0 = 2 * ell + 2 * delta + 2 - g - a5
                y0 = ell + 1 + (ell % 2) - g
        else:
            if a5 >= g + 1 + delta - (ell % 2):
                x0 = g + ell + 2 * delta + 1 - (ell % 2) - a5
                y0 = g
            else:
                x0 = a5 + ell - g + (ell % 2)
                y0 = ell + 1 + (ell % 2) - g
        if y0 >= 1 and y0 <= ell and y0 <= x0 and x0 <= ell + delta:
            coordinate.batch_id = b1 * k + w
            coordinate.s1_idx = x0 + t * k
            coordinate.s2_idx = y0 + t * k
            return
    coordinate.batch_id = -1


def cal_causal_deter_index(round_id, max_loop_num, coordinate_info, const_info, flag):
    j = const_info.core_id_cube + 1
    if flag == True:
        j += 1
    r = round_id + 1
    k = const_info.core_num
    if j > k:
        return -1
    m_gap = 0
    b = const_info.b_size * const_info.n2_size
    m = const_info.s1_outer
    n = const_info.s2_outer
    if sparse_type == 3:
        if m > n:
            m_gap = (const_info.s1_size - const_info.s2_size) // CUBE_BASEM
            m -= m_gap
    tmp_res = 0
    cal_causal_index(k, m, n, b, j, r, coordinate_info)
    w = coordinate_info.batch_id
    n1 = const_info.g_size * const_info.n2_size
    coordinate_info.batch_id = ceil(w, n1) - 1
    n1_idx = w - coordinate_info.batch_id * n1 - 1
    coordinate_info.n2_idx = n1_idx // const_info.g_size
    coordinate_info.g_idx = n1_idx % const_info.g_size
    coordinate_info.s1_idx = coordinate_info.s1_idx + m_gap - 1
    coordinate_info.s2_idx = coordinate_info.s2_idx - 1
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


def cal_band_index(band_info, j, r, coordinate):
    coordinate.batch_id = -1
    k = band_info.k
    m = band_info.m
    n = band_info.n
    p = band_info.p
    q = band_info.q
    b = band_info.b
    b1 = band_info.b1
    b2 = band_info.b2
    l1 = band_info.l1
    l2 = band_info.l2
    l3 = band_info.l3
    n_seg = band_info.n_seg
    r1 = band_info.r1
    r2 = band_info.r2
    r3 = band_info.r3
    rm_batch = band_info.rm_batch
    rm = band_info.rm
    if p + q > m:
        if r <= rm:
            a = r % rm_batch
            w = ceil(r, rm_batch) + b1 * (j - 1)
            if a == 0:
                a = rm_batch
            if a <= r1:
                l1_even = l1 // 2 * 2
                l_len = 2 * p + l1_even - 1
                local_round = l1_even * l_len // 2
                if a <= local_round:
                    y = ceil(a, l_len)
                    r_rem = a % l_len
                    r_rem = r_rem if r_rem != 0 else l_len
                    x = p + y - r_rem
                    if x < 1:
                        y = l1_even + 1 - y
                        x = 1 - x
                else:
                    x = a - local_round
                    y = l1
            elif a <= r1 + r2:
                a2 = a - r1
                y = ceil(a2, m)
                x = a2 % m
                if x == 0:
                    x = m
                y = y + l1
            else:
                a3 = a - r1 - r2
                l3_even = l3 // 2 * 2
                l_len = 2 * m - l3_even - 1
                local_round = l3_even * l_len // 2
                if a3 <= local_round:
                    y = ceil(a3, l_len)
                    r_rem = a3 % l_len
                    r_rem = r_rem if r_rem != 0 else l_len
                    x = y + r_rem
                    if x > m:
                        y = l3_even + 1 - y
                        x = 2 * m + 1 - x
                else:
                    x = m - a3 + local_round + 1
                    y = l3
                y = y + l1 + l2
            w = ((w - 1) % b1) * k + (w - 1) // b1 + 1
            w = ((w - 1) // k) * k + ((y - 1 + (w - 1)) % k) + 1
            coordinate.batch_id = w
            coordinate.s1_idx = x
            coordinate.s2_idx = y
            return
        a = r - rm
        cal_dense_index(k, m, n, b2, j, a, coordinate)
        w = coordinate.batch_id
        if w != -1:
            x = coordinate.s1_idx
            y = coordinate.s2_idx
            if y <= m - p and x >= p + y:
                return
            if y > l1 + l2 and x <= y - l1 - l2:
                return
            if w > 0 and w < b2 + 1 and x > 0 and x < m + 1 and y > 0 and y < n + 1:
                coordinate.batch_id = b1 * k + w
                coordinate.s1_idx = x
                coordinate.s2_idx = y
                return
        return
    if l3 == 0:
        m = p + q + l2 - 2
    if r <= rm:
        a = r % rm_batch
        w = ceil(r, rm_batch) + b1 * (j - 1)
        if a == 0:
            a = rm_batch
        if a <= r1:
            l1_even = l1 // 2 * 2
            l_len = 2 * p + l1_even - 1
            local_round = l1_even * l_len // 2
            if a <= local_round:
                y = ceil(a, l_len)
                r_rem = a % l_len
                r_rem = r_rem if r_rem != 0 else l_len
                x = p + y - r_rem
                if x < 1:
                    y = l1_even + 1 - y
                    x = 1 - x
            else:
                x = a - local_round
                y = l1
        elif a <= r1 + r2:
            a2 = a - r1
            y = ceil(a2, p + q - 1)
            x = a2 % (p + q - 1) + (y - 1)
            if x == y - 1:
                x = p + q - 1 + y - 1
            y = y + l1
        else:
            a3 = a - r1 - r2
            l3_even = l3 // 2 * 2
            l_len = 2 * (p + q) - l3_even - 3
            local_round = l3_even * l_len // 2
            if a3 <= local_round:
                y = ceil(a3, l_len)
                r_rem = a3 % l_len
                r_rem = r_rem if r_rem != 0 else l_len
                x = y + r_rem + 1 + m - (p + q)
                if x > m:
                    y = l3_even + 1 - y
                    x = 2 * m + 1 - x
            else:
                x = m - a3 + local_round + 1
                y = l3
            y = y + l1 + l2
        coordinate.batch_id = w
        coordinate.s1_idx = x
        coordinate.s2_idx = y
        return
    if b2 == 0:
        coordinate.batch_id = -1
        return
    a = r - rm
    seg = p + q - 1
    a1 = ceil(a, seg)
    a2 = a % seg
    a2 = seg if a2 == 0 else a2
    if l3 == 0 or n_seg - m < 1:
        idx = (a1 - 1) * k + j
        w = ceil(idx, n_seg)
        y = idx % n_seg
        if y == 0:
            y = n_seg
        x = y + a2 - q
        if x >= 1 and x <= m:
            coordinate.batch_id = w + b1 * k
            coordinate.s1_idx = x
            coordinate.s2_idx = y
        return
    y = (a1 - 1) * k + j
    x = y + a2 - q
    if x < 1:
        coordinate.batch_id = b
        coordinate.s1_idx = x + m
        coordinate.s2_idx = y + m
        return
    w = ceil(x, m)
    x = x % m
    if x == 0:
        x = m
    y = x + q - a2
    if w == b2 and y > m:
        return
    if y >= 1 and y <= n:
        coordinate.batch_id = w + b1 * k
        coordinate.s1_idx = x
        coordinate.s2_idx = y
    return


def cal_dense_swizzle_deter_index(
    round_id,
    coordinate_info,
    const_info,
    tensor_metadata,
    tensor_seq_q,
    tensor_seq_kv,
    flag,
):
    coordinate_info.batch_id = -1
    j = const_info.core_id_cube
    if flag == True:
        j += 1
    k = const_info.core_num
    if j >= k:
        return -1
    n1 = const_info.n2_size * const_info.g_size
    meta_base = const_info.metadata_len
    b_size = const_info.b_size
    prefix_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_ROUND_PREFIX_OFFSET_IDX
    )
    s1_outer_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_S1_OUTER_OFFSET_IDX
    )
    s2_outer_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_S2_OUTER_OFFSET_IDX
    )
    found_b = -1
    delta = 0
    for b_idx in pl.range(0, b_size, 1):
        prefix_lo = pl.astype(
            pl.getval(tensor_metadata, prefix_base + b_idx), pl.DT_INT64
        )
        prefix_hi = pl.astype(
            pl.getval(tensor_metadata, prefix_base + b_idx + 1), pl.DT_INT64
        )
        if found_b < 0 and round_id < prefix_hi:
            found_b = b_idx
            delta = round_id - prefix_lo
    if found_b < 0:
        return -1
    m_b = pl.getval(tensor_metadata, s1_outer_base + found_b)
    n_b = pl.getval(tensor_metadata, s2_outer_base + found_b)
    schedule_rows = m_b
    schedule_cols = n_b
    row_begin = 0
    col_begin = 0
    s1_token = const_info.s1_token
    s2_token = const_info.s2_token
    if sparse_type != 0 or layout == 2:
        row_begin = pl.getval(
            tensor_metadata,
            meta_base
            + meta_get(const_info, tensor_metadata, META_ROW_BEGIN_OFFSET_IDX)
            + found_b,
        )
        col_begin = pl.getval(
            tensor_metadata,
            meta_base
            + meta_get(const_info, tensor_metadata, META_COL_BEGIN_OFFSET_IDX)
            + found_b,
        )
        schedule_rows = pl.getval(
            tensor_metadata,
            meta_base
            + meta_get(const_info, tensor_metadata, META_SCHEDULE_ROWS_OFFSET_IDX)
            + found_b,
        )
        schedule_cols = pl.getval(
            tensor_metadata,
            meta_base
            + meta_get(const_info, tensor_metadata, META_SCHEDULE_COLS_OFFSET_IDX)
            + found_b,
        )
        s1_token = pl.getval(
            tensor_metadata,
            meta_base
            + meta_get(const_info, tensor_metadata, META_S1_TOKEN_OFFSET_IDX)
            + found_b,
        )
        s2_token = pl.getval(
            tensor_metadata,
            meta_base
            + meta_get(const_info, tensor_metadata, META_S2_TOKEN_OFFSET_IDX)
            + found_b,
        )
    if schedule_rows <= 0 or schedule_cols <= 0:
        return -1
    linear_idx = delta // schedule_rows * k + j
    if linear_idx >= schedule_cols * n1:
        return -1
    n1_idx = linear_idx // schedule_cols
    local_col = linear_idx % schedule_cols
    s2_idx = col_begin + local_col
    s1_idx = row_begin + (local_col + delta) % schedule_rows
    actual_s1 = get_actual_s1_len(const_info, tensor_seq_q, found_b)
    actual_s2 = get_actual_s2_len(const_info, tensor_seq_kv, found_b)
    if s1_idx >= m_b or s2_idx >= n_b:
        return -1
    if sparse_type != 0:
        # Exact tile/band intersection. Boundary tiles still apply the element mask.
        q_lo = s1_idx * CUBE_BASEM
        q_hi = min(q_lo + CUBE_BASEM, actual_s1) - 1
        kv_lo = s2_idx * CUBE_BASEN
        kv_hi = min(kv_lo + CUBE_BASEN, actual_s2) - 1
        if kv_lo > q_hi + s2_token or kv_hi < q_lo - s1_token:
            return -1
    coordinate_info.batch_id = found_b
    coordinate_info.n2_idx = n1_idx // const_info.g_size
    coordinate_info.g_idx = n1_idx % const_info.g_size
    coordinate_info.s1_idx = s1_idx
    coordinate_info.s2_idx = s2_idx
    coordinate_info.s1_outer = m_b
    coordinate_info.s2_outer = n_b
    coordinate_info.actual_s1_len = actual_s1
    coordinate_info.actual_s2_len = actual_s2
    coordinate_info.s1_token = s1_token
    coordinate_info.s2_token = s2_token
    return (found_b * n1 + n1_idx) * m_b * n_b + s2_idx * m_b + s1_idx


def _tnd_line_commit(
    coordinate_info,
    const_info,
    tensor_metadata,
    tensor_seq_q,
    tensor_seq_kv,
    found_b,
    copy_id,
    x,
    y,
    n1,
    g_size,
    is_band,
    s1_outer_base,
    s2_outer_base,
    s1_token_base,
    s2_token_base,
):
    batch = found_b + (copy_id - 1) // n1
    n1_idx = (copy_id - 1) % n1
    s1o_real = pl.getval(tensor_metadata, s1_outer_base + batch)
    s2o_real = pl.getval(tensor_metadata, s2_outer_base + batch)
    s1_token = pl.getval(tensor_metadata, s1_token_base + batch)
    s2_token = pl.getval(tensor_metadata, s2_token_base + batch)
    if is_band:
        p0 = min(_ceil_trunc(s1_token, CUBE_BASEM) + 1, s1o_real)
        q0 = min(_ceil_trunc(s2_token, CUBE_BASEN) + 1, s2o_real)
        if p0 < 0:
            s1_idx = x - 1
            s2_idx = y - 1 + (0 - p0)
        elif q0 < 0:
            s1_idx = x - 1 + (0 - q0)
            s2_idx = y - 1
        else:
            s1_idx = x - 1
            s2_idx = y - 1
    else:
        s1_idx = x + max(0, s1o_real - s2o_real - 1) - 1
        s2_idx = y - 1
    if s1_idx >= s1o_real or s2_idx >= s2o_real or s1_idx < 0 or s2_idx < 0:
        return -1
    actual_s1 = get_actual_s1_len(const_info, tensor_seq_q, batch)
    actual_s2 = get_actual_s2_len(const_info, tensor_seq_kv, batch)
    coordinate_info.batch_id = batch
    coordinate_info.n2_idx = n1_idx // g_size
    coordinate_info.g_idx = n1_idx % g_size
    coordinate_info.s1_idx = s1_idx
    coordinate_info.s2_idx = s2_idx
    coordinate_info.s1_outer = s1o_real
    coordinate_info.s2_outer = s2o_real
    coordinate_info.actual_s1_len = actual_s1
    coordinate_info.actual_s2_len = actual_s2
    coordinate_info.s1_token = s1_token
    coordinate_info.s2_token = s2_token
    return (batch * n1 + n1_idx) * s1o_real * s2o_real + s2_idx * s1o_real + s1_idx


def cal_tnd_line_swizzle_deter_index(
    round_id,
    coordinate_info,
    const_info,
    tensor_metadata,
    tensor_seq_q,
    tensor_seq_kv,
    flag,
):
    coordinate_info.batch_id = -1
    j = const_info.core_id_cube
    if flag == True:
        j += 1
    k = const_info.core_num
    if j >= k:
        return -1
    n1 = const_info.n2_size * const_info.g_size
    g_size = const_info.g_size
    mask_mode = const_info.mask_mode
    is_band = mask_mode == 4
    meta_base = const_info.metadata_len
    b_size = const_info.b_size
    prefix_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_ROUND_PREFIX_OFFSET_IDX
    )
    s1_outer_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_S1_OUTER_OFFSET_IDX
    )
    s2_outer_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_S2_OUTER_OFFSET_IDX
    )
    line_m_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_TND_LINE_M_OFFSET_IDX
    )
    line_n_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_TND_LINE_N_OFFSET_IDX
    )
    line_p_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_TND_LINE_P_OFFSET_IDX
    )
    line_q_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_TND_LINE_Q_OFFSET_IDX
    )
    run_size_base = meta_base + pl.getval(
        tensor_metadata, meta_base + META_TND_LINE_RUN_SIZE_OFFSET_IDX
    )
    s1_token_base = run_size_base + b_size
    s2_token_base = s1_token_base + b_size
    found_b = -1
    delta = 0
    for b_idx in pl.range(0, b_size, 1):
        prefix_lo = pl.astype(
            pl.getval(tensor_metadata, prefix_base + b_idx), pl.DT_INT64
        )
        prefix_hi = pl.astype(
            pl.getval(tensor_metadata, prefix_base + b_idx + 1), pl.DT_INT64
        )
        if found_b < 0 and round_id < prefix_hi:
            found_b = b_idx
            delta = round_id - prefix_lo
    if found_b < 0:
        return -1
    line_m = pl.getval(tensor_metadata, line_m_base + found_b)
    line_n = pl.getval(tensor_metadata, line_n_base + found_b)
    line_p = pl.getval(tensor_metadata, line_p_base + found_b)
    line_q = pl.getval(tensor_metadata, line_q_base + found_b)
    run_size = pl.getval(tensor_metadata, run_size_base + found_b)
    if line_m <= 0 or line_n <= 0 or run_size <= 0 or n1 <= 0:
        return -1
    a = delta + 1
    j1 = j + 1
    m = line_m
    n = line_n
    p = line_p
    q = line_q
    if is_band:
        copies = n1 * run_size
        if p >= m:
            if j1 >= 1 and j1 <= k and a >= 1 and copies > 0 and m > 0 and n > 0:
                y1_d = ((a - 1) // m) * k + j1
                if y1_d <= n * copies:
                    w_d = (y1_d - 1) // n + 1
                    y_d = (y1_d - 1) % n + 1
                    x_d = (y_d + (a - 1) - 1) % m + 1
                    if (
                        w_d >= 1
                        and w_d <= copies
                        and x_d >= 1
                        and x_d <= m
                        and y_d >= 1
                        and y_d <= n
                        and y_d <= x_d + q - 1
                    ):
                        return _tnd_line_commit(
                            coordinate_info,
                            const_info,
                            tensor_metadata,
                            tensor_seq_q,
                            tensor_seq_kv,
                            found_b,
                            w_d,
                            x_d,
                            y_d,
                            n1,
                            g_size,
                            is_band,
                            s1_outer_base,
                            s2_outer_base,
                            s1_token_base,
                            s2_token_base,
                        )
            return -1
        if p + q > m:
            n_new = min(m - 1 + q, n)
            l1_w = m - p
            l2_w = p + q - m
            l3_w = n_new - l1_w - l2_w
            dense_n = n_new - max(0, l3_w - p + 1)
            if j1 >= 1 and j1 <= k and a >= 1 and copies > 0 and m > 0 and dense_n > 0:
                y1_w = ((a - 1) // m) * k + j1
                if y1_w <= dense_n * copies:
                    w_w = (y1_w - 1) // dense_n + 1
                    y_w = (y1_w - 1) % dense_n + 1
                    x_w = (y_w + (a - 1) - 1) % m + 1
                    l1_m = m - p
                    l2_m = p + q - m
                    l3_m = n - l1_m - l2_m
                    fold = max(0, l3_m - p + 1)
                    fold_hit = 0
                    if fold > 0 and y_w <= fold and x_w >= p + y_w:
                        fold_hit = 1
                    y_m = y_w + fold_hit * (n - fold)
                    miss = 0
                    if fold_hit == 0 and y_w <= l1_m and x_w >= p + y_w:
                        miss = 1
                    if y_m > l1_m + l2_m and x_w <= y_m - l1_m - l2_m:
                        miss = 1
                    if (
                        miss == 0
                        and w_w >= 1
                        and w_w <= copies
                        and x_w >= 1
                        and x_w <= m
                        and y_m >= 1
                        and y_m <= n
                    ):
                        return _tnd_line_commit(
                            coordinate_info,
                            const_info,
                            tensor_metadata,
                            tensor_seq_q,
                            tensor_seq_kv,
                            found_b,
                            w_w,
                            x_w,
                            y_m,
                            n1,
                            g_size,
                            is_band,
                            s1_outer_base,
                            s2_outer_base,
                            s1_token_base,
                            s2_token_base,
                        )
            return -1
        seg = p + q - 1
        if seg <= 0:
            return -1
        a1 = (a - 1) // seg + 1
        a2 = (a - 1) % seg + 1
        l3_n = max(0, min(p + n - m - 1, p + q - 2))
        if l3_n == 0 or n - m < 1:
            idx = (a1 - 1) * k + j1
            w_n = (idx - 1) // n + 1
            y_n = (idx - 1) % n + 1
            x_n = y_n + a2 - q
            if (
                w_n >= 1
                and w_n <= copies
                and x_n >= 1
                and x_n <= m
                and y_n >= 1
                and y_n <= n
            ):
                return _tnd_line_commit(
                    coordinate_info,
                    const_info,
                    tensor_metadata,
                    tensor_seq_q,
                    tensor_seq_kv,
                    found_b,
                    w_n,
                    x_n,
                    y_n,
                    n1,
                    g_size,
                    is_band,
                    s1_outer_base,
                    s2_outer_base,
                    s1_token_base,
                    s2_token_base,
                )
            return -1
        y_abs = (a1 - 1) * k + j1
        x_abs = y_abs + a2 - q
        if x_abs < 1:
            return -1
        w_n2 = (x_abs - 1) // m + 1
        x_n2 = (x_abs - 1) % m + 1
        y_n2 = x_n2 + q - a2
        if (
            w_n2 >= 1
            and w_n2 <= copies
            and x_n2 >= 1
            and x_n2 <= m
            and y_n2 >= 1
            and y_n2 <= n
        ):
            return _tnd_line_commit(
                coordinate_info,
                const_info,
                tensor_metadata,
                tensor_seq_q,
                tensor_seq_kv,
                found_b,
                w_n2,
                x_n2,
                y_n2,
                n1,
                g_size,
                is_band,
                s1_outer_base,
                s2_outer_base,
                s1_token_base,
                s2_token_base,
            )
        return -1
    total = n1 * run_size
    pairs = total // 2
    virt_m = m
    virt_n = 2 * n - m + 3
    pair_rounds = ceil(virt_n * pairs, k) * virt_m
    if a <= pair_rounds:
        if j1 >= 1 and j1 <= k and a >= 1 and pairs > 0 and virt_m > 0 and virt_n > 0:
            y1_c = ((a - 1) // virt_m) * k + j1
            if y1_c <= virt_n * pairs:
                w_c = (y1_c - 1) // virt_n + 1
                y_c = (y1_c - 1) % virt_n + 1
                x_c = (y_c + (a - 1) - 1) % virt_m + 1
                n_ext = n + 1
                flip = 0
                if y_c >= x_c + n_ext - m + 1:
                    flip = 1
                y_cm = y_c + flip * (2 * n_ext - m + 2 - 2 * y_c)
                x_cm = x_c + flip * (m + 1 - 2 * x_c)
                nid = 2 * w_c - 1 + flip
                if (
                    nid >= 1
                    and nid <= pairs * 2
                    and x_cm >= 1
                    and x_cm <= m
                    and y_cm >= 1
                    and y_cm <= n
                    and y_cm <= x_cm + n - m + 1
                ):
                    return _tnd_line_commit(
                        coordinate_info,
                        const_info,
                        tensor_metadata,
                        tensor_seq_q,
                        tensor_seq_kv,
                        found_b,
                        nid,
                        x_cm,
                        y_cm,
                        n1,
                        g_size,
                        is_band,
                        s1_outer_base,
                        s2_outer_base,
                        s1_token_base,
                        s2_token_base,
                    )
        return -1
    if (total % 2) == 1:
        single_round = a - pair_rounds
        if j1 >= 1 and j1 <= k and single_round >= 1 and m > 0 and n > 0:
            y1_s = ((single_round - 1) // m) * k + j1
            if y1_s <= n:
                w_s = (y1_s - 1) // n + 1
                y_s = (y1_s - 1) % n + 1
                x_s = (y_s + (single_round - 1) - 1) % m + 1
                if (
                    x_s >= 1
                    and x_s <= m
                    and y_s >= 1
                    and y_s <= n
                    and y_s <= x_s + n - m + 1
                ):
                    return _tnd_line_commit(
                        coordinate_info,
                        const_info,
                        tensor_metadata,
                        tensor_seq_q,
                        tensor_seq_kv,
                        found_b,
                        total,
                        x_s,
                        y_s,
                        n1,
                        g_size,
                        is_band,
                        s1_outer_base,
                        s2_outer_base,
                        s1_token_base,
                        s2_token_base,
                    )
        return -1
    return -1


def cal_band_deter_index(
    round_id, max_loop_num, coordinate_info, const_info, band_info, flag
):
    j = const_info.core_id_cube + 1
    if flag == True:
        j += 1
    r = round_id + 1
    k = const_info.core_num
    if j > k:
        return -1
    cal_band_index(band_info, j, r, coordinate_info)
    w = coordinate_info.batch_id
    n1 = const_info.g_size * const_info.n2_size
    coordinate_info.batch_id = ceil(w, n1) - 1
    n1_idx = w - coordinate_info.batch_id * n1 - 1
    coordinate_info.n2_idx = n1_idx // const_info.g_size
    coordinate_info.g_idx = n1_idx % const_info.g_size
    coordinate_info.s1_idx = coordinate_info.s1_idx - 1 + coordinate_info.m_offset
    coordinate_info.s2_idx = coordinate_info.s2_idx - 1 + coordinate_info.n_offset
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
    return (
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


def gen_band_info(const_info, band_info):
    k = const_info.core_num
    m = const_info.s1_outer
    n = const_info.s2_outer
    b = const_info.b_size * const_info.n2_size
    s1_token = const_info.s1_token
    s2_token = const_info.s2_token
    m_offset = 0
    n_offset = 0
    if const_info.mask_mode == 3 and const_info.s1_size > const_info.s2_size:
        skip_m = (const_info.s1_size - const_info.s2_size) // CUBE_BASEM
        if skip_m > 0 and skip_m < m:
            m = m - skip_m
            s2_token = s2_token + skip_m * CUBE_BASEM
            m_offset = skip_m
    p = _ceil_trunc(s1_token, CUBE_BASEM) + 1
    q = _ceil_trunc(s2_token, CUBE_BASEN) + 1
    p = m if p > m else p
    q = n if q > n else q
    if p < 0:
        actual_m = m
        actual_n = n + p
        actual_p = 1
        actual_q = p + q
        n_offset = -p
    elif q < 0:
        actual_m = m + q
        actual_n = n
        actual_p = p + q
        actual_q = 1
        m_offset = -q
    else:
        actual_m = m
        actual_n = n
        actual_p = p
        actual_q = q
    b1 = b // k
    b2 = b % k
    if actual_p + actual_q > actual_m:
        l1 = actual_m - actual_p
        l2 = actual_p + actual_q - actual_m
        l3 = min(actual_m - 1, actual_n - actual_q)
        n_seg = l1 + l2 + l3
        r1 = (actual_p + actual_m - 1) * l1 // 2
        r2 = actual_m * l2
        r3 = (2 * actual_m - 1 - l3) * l3 // 2
        rm_batch = r1 + r2 + r3
        rm = b1 * rm_batch
        rm2 = actual_m * ceil(actual_n * b, min(k, b * actual_m))
    else:
        l1 = actual_q - 1
        l2 = min(actual_n - actual_q + 1, actual_m + 2 - actual_p - actual_q)
        l3 = max(0, min(actual_p + actual_n - actual_m - 1, actual_p + actual_q - 2))
        n_seg = l1 + l2 + l3
        r1 = (2 * actual_p - 2 + actual_q) * l1 // 2
        r2 = (actual_p + actual_q - 1) * l2
        r3 = (actual_p + actual_q - 2) * l3 - l3 * (l3 - 1) // 2
        rm_batch = r1 + r2 + r3
        rm = b1 * rm_batch
        rm2 = ceil(actual_n * b, k) * (actual_p + actual_q - 1)
    band_info.k = k
    band_info.m = actual_m
    band_info.n = actual_n
    band_info.p = actual_p
    band_info.q = actual_q
    band_info.b = b
    band_info.b1 = b1
    band_info.b2 = b2
    band_info.l1 = l1
    band_info.l2 = l2
    band_info.l3 = l3
    band_info.n_seg = n_seg
    band_info.r1 = r1
    band_info.r2 = r2
    band_info.r3 = r3
    band_info.rm_batch = rm_batch
    band_info.rm = rm
    band_info.rm2 = rm2
    band_info.m_offset = m_offset
    band_info.n_offset = n_offset


def is_valid_for_deter(index, const_info):
    g_dim_tail = index % (const_info.s1_outer * const_info.s2_outer)
    s2o_dim_idx = g_dim_tail // const_info.s1_outer
    s1o_dim_idx = g_dim_tail % const_info.s1_outer
    s2_idx_left = s2o_dim_idx * CUBE_BASEN
    s2_idx_right = min((s2o_dim_idx + 1) * CUBE_BASEN, const_info.s2_size)
    if has_attn_mask:
        if const_info.mask_mode == 3:
            s2_ignored_end_len = const_info.s1_size - CUBE_BASEM * (s1o_dim_idx + 1)
            if const_info.s2_size > s2_ignored_end_len:
                s2_end_len = const_info.s2_size - s2_ignored_end_len
            else:
                s2_end_len = 0
            s2_end_len = min(s2_end_len, const_info.s2_size)
            return s2_idx_left < s2_end_len
        s2_sparse_left = CUBE_BASEM * s1o_dim_idx - const_info.s1_token
        if s2_sparse_left < 0:
            s2_sparse_left = 0
        s2_sparse_left = s2_sparse_left // 64 * 64
        s2_sparse_right = (
            (
                min(CUBE_BASEM * (s1o_dim_idx + 1), const_info.s1_size)
                + const_info.s2_token
                + 63
            )
            // 64
            * 64
        )
        s2_sparse_right = min(s2_sparse_right, const_info.s2_size)
        return s2_idx_left < s2_sparse_right and s2_idx_right > s2_sparse_left
    return True


def cal_deter_index(
    round_id,
    max_loop_num,
    coordinate_info,
    const_info,
    band_info,
    tensor_metadata,
    tensor_seq_q,
    tensor_seq_kv,
    flag=False,
):
    coordinate_info.mask_mode = const_info.mask_mode
    next_valid_round_id = max_loop_num
    next_valid_index = -1
    schedule_kind = 0
    if is_var_len():
        schedule_kind = meta_get(const_info, tensor_metadata, META_SCHEDULE_KIND_IDX)
    for current_round_id in pl.range(round_id, max_loop_num, 1):
        if is_var_len():
            # pl.range 上界必须是 tiling; 真实轮数只做提前退出, 避免扫完保守上界.
            deter_rounds = meta_get(const_info, tensor_metadata, META_DETER_MAX_NUM_IDX)
            if current_round_id >= deter_rounds:
                break
            if schedule_kind == META_TND_LINE_SWIZZLE:
                next_valid_index = cal_tnd_line_swizzle_deter_index(
                    current_round_id,
                    coordinate_info,
                    const_info,
                    tensor_metadata,
                    tensor_seq_q,
                    tensor_seq_kv,
                    flag,
                )
            else:
                next_valid_index = cal_dense_swizzle_deter_index(
                    current_round_id,
                    coordinate_info,
                    const_info,
                    tensor_metadata,
                    tensor_seq_q,
                    tensor_seq_kv,
                    flag,
                )
        elif sparse_type == 4:
            next_valid_index = cal_band_deter_index(
                current_round_id,
                max_loop_num,
                coordinate_info,
                const_info,
                band_info,
                flag,
            )
        elif sparse_type == 3:
            next_valid_index = cal_causal_deter_index(
                current_round_id, max_loop_num, coordinate_info, const_info, flag
            )
        else:
            next_valid_index = cal_dense_deter_index(
                current_round_id, max_loop_num, coordinate_info, const_info, flag
            )
        if next_valid_index >= 0:
            if is_var_len() or is_valid_for_deter(next_valid_index, const_info):
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
    run_info.gm_batch_id = run_info.batch_id
    run_info.q_start = 0
    run_info.kv_start = 0
    if layout == 2:
        run_info.gm_batch_id = 0
        run_info.q_start = pl.getval(tensor_info.tensor_cu_q, run_info.batch_id)
        run_info.kv_start = pl.getval(tensor_info.tensor_cu_kv, run_info.batch_id)

    run_info.task_id = task_id
    run_info.task_id_mod2 = task_id % 2
    run_info.s2_outer_cur = coordinate_info.s2_outer
    run_info.s1_token = coordinate_info.s1_token
    run_info.s2_token = coordinate_info.s2_token
    if has_seq_used_q == 1 or layout == 2:
        s1_remain = coordinate_info.actual_s1_len - run_info.s1o_idx * CUBE_BASEM
        run_info.s1_real_size = CUBE_BASEM if s1_remain > CUBE_BASEM else s1_remain
    else:
        run_info.s1_real_size = (
            const_info.s1_tail
            if run_info.s1o_idx == const_info.s1_outer - 1
            else CUBE_BASEM
        )
    if has_seq_used_kv == 1 or layout == 2:
        s2_remain = coordinate_info.actual_s2_len - run_info.s2o_idx * CUBE_BASEN
        run_info.s2_real_size = CUBE_BASEN if s2_remain > CUBE_BASEN else s2_remain
    else:
        run_info.s2_real_size = (
            const_info.s2_tail
            if run_info.s2o_idx == const_info.s2_outer - 1
            else CUBE_BASEN
        )

    run_info.inner_s1_loop_num = ceil(run_info.s1_real_size, 128)
    run_info.inner_s2_loop_num = ceil(run_info.s2_real_size, 128)
    with pl.section_vector():
        run_info.deq_scale_q_value = const_info.deq_scale_q_value
        run_info.deq_scale_k_value = const_info.deq_scale_k_value
        run_info.deq_scale_v_value = const_info.deq_scale_v_value
        run_info.deq_scale_do_value = const_info.deq_scale_do_value

        if not is_var_len() and (sparse_type == 3 or sparse_type == 4):
            run_info.kv_need_atomic = True
        elif is_var_len():
            # swizzle 下 (n1_idx, s2o) 与 linearIdx = (delta // m_b) * k + j 一一对应,
            # 同一列只会落在唯一一个核上, dk/dv 不存在跨核累加, 只需核内跨 s1 行累加。
            if sparse_type != 0:
                run_info.kv_need_atomic = True
            else:
                run_info.kv_need_atomic = run_info.is_value_reuse
        else:
            run_info.kv_need_atomic = (
                coordinate_info.s2_outer != 1 and run_info.is_value_reuse
            ) or (
                coordinate_info.s2_outer == 1
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
    delta = coordinate_info.actual_s1_len - coordinate_info.actual_s2_len
    is_outer_valid = (
        True
        if _abs((run_info.s1o_idx * CUBE_BASEM - delta) - run_info.s2o_idx * CUBE_BASEN)
        < 512
        else False
    )
    for i in pl.range(4):
        run_info.inner_s1_real_size[i] = (
            inner_s1_tail_size if i == run_info.inner_s1_loop_num - 1 else 128
        )
        run_info.inner_s2_real_size[i] = (
            inner_s2_tail_size if i == run_info.inner_s2_loop_num - 1 else 128
        )
        if has_attn_mask:
            if is_outer_valid:
                s1_offset = run_info.s1o_idx * CUBE_BASEM + i * TS - delta
                for j in pl.range(4):
                    s2_offset = run_info.s2o_idx * CUBE_BASEM + j * TS
                    run_info.is_valid_inner_block[i] = (
                        0
                        if (
                            (
                                i < run_info.inner_s1_loop_num
                                and j < run_info.inner_s2_loop_num
                            )
                            and (s1_offset - s2_offset < 128)
                        )
                        else 1
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
    if layout == 0 or layout == 2:  # BSND
        pl.load(
            q_l1,
            tensor_info.tensor_q,
            [
                run_info.gm_batch_id,
                run_info.q_start + run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                run_info.n1o_idx,
                0,
            ],
            order=[3, 1],
        )
        pl.load(
            do_l1,
            tensor_info.tensor_do,
            [
                run_info.gm_batch_id,
                run_info.q_start + run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
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
                run_info.gm_batch_id,
                run_info.n1o_idx,
                run_info.q_start + run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                0,
            ],
            order=[3, 2],
        )
        pl.load(
            do_l1,
            tensor_info.tensor_do,
            [
                run_info.gm_batch_id,
                run_info.n1o_idx,
                run_info.q_start + run_info.s1o_idx * CUBE_BASEM + run_info.s1_idx * TS,
                0,
            ],
            order=[3, 2],
        )
    if not run_info.is_value_reuse and run_info.s1_idx == 0:
        pl.set_validshape(k_l1, [real_m, TD])
        pl.set_validshape(v_l1, [real_m, TD])
        if layout == 0 or layout == 2:  # BSND
            pl.load(
                k_l1,
                tensor_info.tensor_k,
                [
                    run_info.gm_batch_id,
                    run_info.kv_start
                    + run_info.s2o_idx * CUBE_BASEN
                    + run_info.s2_idx * TS,
                    run_info.n2o_idx,
                    0,
                ],
                order=[1, 3],
            )
            pl.load(
                v_l1,
                tensor_info.tensor_v,
                [
                    run_info.gm_batch_id,
                    run_info.kv_start
                    + run_info.s2o_idx * CUBE_BASEN
                    + run_info.s2_idx * TS,
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
                    run_info.gm_batch_id,
                    run_info.n2o_idx,
                    run_info.kv_start
                    + run_info.s2o_idx * CUBE_BASEN
                    + run_info.s2_idx * TS,
                    0,
                ],
                order=[2, 3],
            )
            pl.load(
                v_l1,
                tensor_info.tensor_v,
                [
                    run_info.gm_batch_id,
                    run_info.n2o_idx,
                    run_info.kv_start
                    + run_info.s2o_idx * CUBE_BASEN
                    + run_info.s2_idx * TS,
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
def merge_band_attn_mask_vf(attn_tile, attn_pre_tile):
    preg_all_16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    preg_all_8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT8)
    for offset in pl.range(0, TS * TS_HALF, 256):
        next_bytes = vf.load_align(attn_tile, offset)
        pre_bytes = vf.load_align(attn_pre_tile, offset)
        next_words = vf.bit_cast(next_bytes, dtype=pl.DT_UINT16)
        pre_words = vf.bit_cast(pre_bytes, dtype=pl.DT_UINT16)
        pre_inverse = vf.not_(pre_words, preg_all_16)
        keep_words = vf.and_(next_words, pre_inverse, preg_all_16)
        vf.store_align(
            attn_pre_tile + offset,
            vf.bit_cast(keep_words, dtype=pl.DT_INT8),
            preg_all_8,
        )


@pl.vector_function
def compute_p_ds_vf(
    sp_tile,
    dpds_tile,
    sp_tile_bit8,
    dpds_tile_bit8,
    perm_tile,
    lse_tile,
    d_tile,
    attn_tile,
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
    preg_all_32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    if has_attn_mask:
        preg_mask1 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
        preg_mask2 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
        preg_mask3 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
        preg_mask4 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_INT32)
        vreg_min = vf.full(ATTEN_MASK_MIN, preg_all_32, dtype=pl.DT_FP32)

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
        if has_attn_mask:
            preg_mask1 = vf.load_align(attn_tile, i * 64, dist=pl.LoadDist.DS)
            preg_mask2 = vf.load_align(attn_tile, (i + 1) * 64, dist=pl.LoadDist.DS)
            preg_mask3 = vf.load_align(attn_tile, (i + 2) * 64, dist=pl.LoadDist.DS)
            preg_mask4 = vf.load_align(attn_tile, (i + 3) * 64, dist=pl.LoadDist.DS)

            vreg_sp1 = vf.select(vreg_sp1, vreg_min, preg_mask1)
            vreg_sp2 = vf.select(vreg_sp2, vreg_min, preg_mask2)
            vreg_sp3 = vf.select(vreg_sp3, vreg_min, preg_mask3)
            vreg_sp4 = vf.select(vreg_sp4, vreg_min, preg_mask4)

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

        if has_attn_mask:
            preg_mask1 = vf.load_align(attn_tile, (i + 4) * 64, dist=pl.LoadDist.DS)
            preg_mask2 = vf.load_align(attn_tile, (i + 5) * 64, dist=pl.LoadDist.DS)
            preg_mask3 = vf.load_align(attn_tile, (i + 6) * 64, dist=pl.LoadDist.DS)
            preg_mask4 = vf.load_align(attn_tile, (i + 7) * 64, dist=pl.LoadDist.DS)

            vreg_sp1 = vf.select(vreg_sp1, vreg_min, preg_mask1)
            vreg_sp2 = vf.select(vreg_sp2, vreg_min, preg_mask2)
            vreg_sp3 = vf.select(vreg_sp3, vreg_min, preg_mask3)
            vreg_sp4 = vf.select(vreg_sp4, vreg_min, preg_mask4)

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
    attn_tile,
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
        attn_tile,
        src_m,
        src_n,
        ss,
        dps,
        deq_p_scale,
        ds_scale,
    )


def cal_atten_mask_offset(run_info, const_info, s1_extra=0):
    # Mask coordinates are batch-local, including packed TND. q_start/kv_start
    # are only for GM addressing; metadata supplies the per-batch token.
    # Next boundary: s2 - s1 <= s2_token.
    real_s1 = run_info.inner_s1_real_size[run_info.s1_idx]
    if qk_s1_fuse() and is_fused_s1_pair(run_info, run_info.s1_idx):
        real_s1 = real_s1 + run_info.inner_s1_real_size[run_info.s1_idx + 1]
    first_half_s1 = ceil(real_s1, QUANT_S1_BASE_COUNT) * QUANT_S1_BASE_COUNT // 2
    s1_offset = (
        run_info.s1o_idx * CUBE_BASEM
        + run_info.s1_idx * TS
        + const_info.sub_id * first_half_s1
        + s1_extra
    )
    s2_offset = run_info.s2_cv_begin + run_info.s2_idx * TS
    delta = s1_offset - s2_offset + run_info.s2_token
    return delta


def cal_atten_mask_offset_pre(run_info, const_info):
    real_s1 = run_info.inner_s1_real_size[run_info.s1_idx]
    first_half_s1 = ceil(real_s1, QUANT_S1_BASE_COUNT) * QUANT_S1_BASE_COUNT // 2
    s1_offset = (
        run_info.s1o_idx * CUBE_BASEM
        + run_info.s1_idx * TS
        + const_info.sub_id * first_half_s1
    )
    s2_offset = run_info.s2_cv_begin + run_info.s2_idx * TS
    delta_pre = run_info.s1_token - (s1_offset - s2_offset)
    return delta_pre


def load_attn_mask(attn_vec, attn_pre_vec, run_info, const_info, tensor_info):
    if has_attn_mask:
        delta = cal_atten_mask_offset(run_info, const_info)
        if delta >= 0:
            s1_offset = pl.min(delta, 128) + 1
            s2_offset = 0
        else:
            s1_offset = 1
            s2_offset = pl.min(_abs(delta), 128)
        pl.load(
            attn_vec,
            tensor_info.tensor_attn_mask,
            [
                s2_offset,
                s1_offset,
            ],
            order=[0, 1],
        )
        if sparse_type == 4:
            delta_pre = cal_atten_mask_offset_pre(run_info, const_info)
            if delta_pre >= 0:
                s1_offset_pre = 0
                s2_offset_pre = pl.min(delta_pre, 128)
            else:
                s1_offset_pre = pl.min(_abs(delta_pre), 128)
                s2_offset_pre = 0
            pl.load(
                attn_pre_vec,
                tensor_info.tensor_attn_mask,
                [
                    s2_offset_pre,
                    s1_offset_pre,
                ],
                order=[0, 1],
            )


def preload_attn_mask(run_info, const_info, tensor_info):
    if not has_attn_mask:
        return
    if (
        run_info.s2_idx >= run_info.inner_s2_loop_num
        or run_info.s1_idx >= run_info.inner_s1_loop_num
    ):
        return
    real_s1 = run_info.inner_s1_real_size[run_info.s1_idx]
    first_half_s1 = ceil(real_s1, QUANT_S1_BASE_COUNT) * QUANT_S1_BASE_COUNT // 2
    current_real_s1 = min(first_half_s1, real_s1 - const_info.sub_id * first_half_s1)
    if current_real_s1 <= 0:
        return

    attn_vec = tensor_info.attn_mask_pool.next()
    attn_pre_vec = attn_vec
    if sparse_type == 4:
        attn_pre_vec = tensor_info.attn_mask_pool.next()
    load_attn_mask(attn_vec, attn_pre_vec, run_info, const_info, tensor_info)
    if sparse_type == 4:
        merge_band_attn_mask_vf(attn_vec, attn_pre_vec)


def iterate_p_ds(p_l1_bit8, ds_l1_bit8, sdp_id, const_info, run_info, tensor_info):
    real_s1 = run_info.inner_s1_real_size[run_info.s1_idx]
    real_s2 = run_info.inner_s2_real_size[run_info.s2_idx]
    first_half_s1 = ceil(real_s1, QUANT_S1_BASE_COUNT) * QUANT_S1_BASE_COUNT // 2
    current_real_s1 = min(first_half_s1, real_s1 - const_info.sub_id * first_half_s1)
    current_real_s2 = ceil(real_s2, QUANT_S2_BASE_COUNT) * QUANT_S2_BASE_COUNT

    lse_vec = tensor_info.lse_vec.next()
    d_vec = tensor_info.d_vec.next()

    if current_real_s1 <= 0:
        return

    perm_vec = tensor_info.perm_vec.current()
    zero_vec = tensor_info.zero_vec.current()
    pl.set_validshape(lse_vec, [1, current_real_s1])
    pl.set_validshape(d_vec, [1, current_real_s1])

    attn_vec = tensor_info.attn_mask_pool.current()
    if layout == 0 or layout == 1 or layout == 2:
        pl.load(
            lse_vec,
            tensor_info.tensor_softmax_lse,
            [
                run_info.gm_batch_id,
                run_info.n1o_idx,
                run_info.q_start
                + run_info.s1o_idx * CUBE_BASEM
                + run_info.s1_idx * TS
                + const_info.sub_id * first_half_s1,
            ],
            order=[1, 2],
        )
        pl.load(
            d_vec,
            tensor_info.tensor_workspace_sfmg,
            [
                run_info.gm_batch_id,
                run_info.n1o_idx,
                run_info.q_start
                + run_info.s1o_idx * CUBE_BASEM
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
        attn_vec,
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


def load_k_for_dq(
    right_first, right_second, k_size_first, k_size_second, run_info, tensor_info
):
    k_l1_first = tensor_info.common_l1.next()
    k_l1_second = tensor_info.common_l1.next()
    tensor_info.common_l1_db.next()
    pl.set_validshape(k_l1_first, [k_size_first, D_SIZE])
    pl.set_validshape(k_l1_second, [k_size_second, D_SIZE])
    if layout == 0 or layout == 2:  # BSND
        pl.load(
            k_l1_first,
            tensor_info.tensor_k,
            [
                run_info.gm_batch_id,
                run_info.kv_start
                + run_info.s2o_idx * CUBE_BASEN
                + run_info.s2_idx * TS,
                run_info.n2o_idx,
                0,
            ],
            order=[1, 3],
        )
        pl.load(
            k_l1_second,
            tensor_info.tensor_k,
            [
                run_info.gm_batch_id,
                run_info.kv_start
                + run_info.s2o_idx * CUBE_BASEN
                + run_info.s2_idx * TS
                + TS,
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
                run_info.gm_batch_id,
                run_info.n2o_idx,
                run_info.kv_start
                + run_info.s2o_idx * CUBE_BASEN
                + run_info.s2_idx * TS,
                0,
            ],
            order=[2, 3],
        )
        pl.load(
            k_l1_second,
            tensor_info.tensor_k,
            [
                run_info.gm_batch_id,
                run_info.n2o_idx,
                run_info.kv_start
                + run_info.s2o_idx * CUBE_BASEN
                + run_info.s2_idx * TS
                + TS,
                0,
            ],
            order=[2, 3],
        )
    pl.set_validshape(right_first, [k_size_first, D_SIZE])
    pl.set_validshape(right_second, [k_size_second, D_SIZE])
    pl.move(right_first, k_l1_first, [0, 0])
    pl.move(right_second, k_l1_second, [0, 0])


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
        pl.set_validshape(right_first, [k_size_first, D_SIZE])
        pl.set_validshape(right_second, [k_size_second, D_SIZE])
        pl.move(right_first, k_l1_first, [0, 0])
        pl.move(right_second, k_l1_second, [0, 0])
    else:
        load_k_for_dq(
            right_first,
            right_second,
            k_size_first,
            k_size_second,
            run_info,
            tensor_info,
        )
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
    if layout == 0 or layout == 2:  # BSND
        pl.load(
            do_l1,
            tensor_info.tensor_do_p_dy,
            [
                run_info.gm_batch_id,
                run_info.q_start + run_info.s1o_idx * CUBE_BASEN + run_info.s1_idx * TS,
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
                run_info.gm_batch_id,
                run_info.n1o_idx,
                run_info.q_start + run_info.s1o_idx * CUBE_BASEN + run_info.s1_idx * TS,
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
    if layout == 0 or layout == 2:  # BSND
        pl.load(
            q_l1,
            tensor_info.tensor_q_ds,
            [
                run_info.gm_batch_id,
                run_info.q_start + run_info.s1o_idx * CUBE_BASEN + run_info.s1_idx * TS,
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
                run_info.gm_batch_id,
                run_info.n1o_idx,
                run_info.q_start + run_info.s1o_idx * CUBE_BASEN + run_info.s1_idx * TS,
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
        if layout == 0 or layout == 2:  # BSND
            if run_info.kv_need_atomic:
                pl.store(
                    workspace_dk,
                    dk_vec,
                    [
                        run_info.gm_batch_id,
                        run_info.kv_start
                        + run_info.s2o_idx * CUBE_BASEN
                        + run_info.s2_idx * TS,
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
                        run_info.gm_batch_id,
                        run_info.kv_start
                        + run_info.s2o_idx * CUBE_BASEN
                        + run_info.s2_idx * TS,
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
                        run_info.gm_batch_id,
                        run_info.kv_start
                        + run_info.s2o_idx * CUBE_BASEN
                        + run_info.s2_idx * TS,
                        run_info.n2o_idx,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[1, 3],
                )
                pl.store(
                    workspace_dv,
                    dv_vec,
                    [
                        run_info.gm_batch_id,
                        run_info.kv_start
                        + run_info.s2o_idx * CUBE_BASEN
                        + run_info.s2_idx * TS,
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
                        run_info.gm_batch_id,
                        run_info.n2o_idx,
                        run_info.kv_start
                        + run_info.s2o_idx * CUBE_BASEN
                        + run_info.s2_idx * TS,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[2, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                )
                pl.store(
                    workspace_dv,
                    dv_vec,
                    [
                        run_info.gm_batch_id,
                        run_info.n2o_idx,
                        run_info.kv_start
                        + run_info.s2o_idx * CUBE_BASEN
                        + run_info.s2_idx * TS,
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
                        run_info.gm_batch_id,
                        run_info.n2o_idx,
                        run_info.kv_start
                        + run_info.s2o_idx * CUBE_BASEN
                        + run_info.s2_idx * TS,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[2, 3],
                )
                pl.store(
                    workspace_dv,
                    dv_vec,
                    [
                        run_info.gm_batch_id,
                        run_info.n2o_idx,
                        run_info.kv_start
                        + run_info.s2o_idx * CUBE_BASEN
                        + run_info.s2_idx * TS,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[2, 3],
                )
    else:
        block_count = run_info.inner_s1_real_size[run_info.s1_idx]
        workspace_dq = tensor_info.tensor_workspace_dq
        dq_vec = tensor_info.dq_vec.current()

        pl.set_validshape(dq_vec, [block_count, HALF_D_SIZE])

        if layout == 0 or layout == 2:  # BSND
            if run_info.s2_outer_cur != 1:
                pl.store(
                    workspace_dq,
                    dq_vec,
                    [
                        run_info.gm_batch_id,
                        run_info.q_start
                        + run_info.s1o_idx * CUBE_BASEM
                        + run_info.s1_idx * TS,
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
                        run_info.gm_batch_id,
                        run_info.q_start
                        + run_info.s1o_idx * CUBE_BASEM
                        + run_info.s1_idx * TS,
                        run_info.n1o_idx,
                        const_info.sub_id * HALF_D_SIZE,
                    ],
                    order=[1, 3],
                )
        if layout == 1:  # BNSD
            if run_info.s2_outer_cur != 1:
                pl.store(
                    workspace_dq,
                    dq_vec,
                    [
                        run_info.gm_batch_id,
                        run_info.n1o_idx,
                        run_info.q_start
                        + run_info.s1o_idx * CUBE_BASEM
                        + run_info.s1_idx * TS,
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
                        run_info.gm_batch_id,
                        run_info.n1o_idx,
                        run_info.q_start
                        + run_info.s1o_idx * CUBE_BASEM
                        + run_info.s1_idx * TS,
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


@pl.vector_function
def dequant_out_dq(tensor, real_length, scale):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(real_length):
        vreg_x1 = vf.load_align(tensor, i * D_SIZE)
        vreg_x2 = vf.load_align(tensor, i * D_SIZE + HALF_D_SIZE)

        vreg_x1 = vf.muls(vreg_x1, scale, preg_all)
        vreg_x2 = vf.muls(vreg_x2, scale, preg_all)

        vf.store_align(tensor + i * D_SIZE, vreg_x1, preg_all)
        vf.store_align(tensor + i * D_SIZE + HALF_D_SIZE, vreg_x2, preg_all)


def dequant_dqkv(mode, tensor_1, tensor_2, const_info, run_info):
    if mode == 0:
        scale = run_info.deq_scale_k_value * const_info.deq_scale_ds_value
        real_length = run_info.inner_s1_real_size[run_info.s1_idx]
        dequant_out_dq(tensor_1, ceil(real_length, 2), scale)
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
        preload_attn_mask(run_info, const_info, tensor_info)
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
        fused = qk_s1_fuse() and is_fused_s1_pair(run_info, run_info.s1_idx)
        if fused:
            p_l1_bit8_hi = p_l1_bit8_tmp.next()
            if const_info.sub_id == 1:
                p_l1_bit8 = p_l1_bit8_hi
        if (
            run_info.s2_idx < run_info.inner_s2_loop_num
            and run_info.s1_idx < run_info.inner_s1_loop_num
        ):
            ds_s1 = run_info.s1_idx
            if fused:
                ds_s1 = run_info.s1_idx + const_info.sub_id
            ds_l1_bit8 = tensor_info.ds_l1_bit8[
                l1_ds(ds_s1, run_info.s2_idx, run_info.task_id_mod2)
            ]
            iterate_p_ds(
                p_l1_bit8, ds_l1_bit8, sdp_id, const_info, run_info, tensor_info
            )

        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=sdp_id,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        if fused or sdp_id % 2 == 1:
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
            if fused:
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
    skip_s1_1 = qk_s1_fuse() and is_fused_s1_pair(run_info, 0)
    skip_s1_3 = qk_s1_fuse() and is_fused_s1_pair(run_info, 2)

    set_quant_run_info(run_info, 0, kv_inner_id)
    process_sp(run_info, const_info, 0, tensor_info)
    process_p_ds(run_info, const_info, 0, tensor_info, tmp_info)
    with pl.section_cube():
        if not last_run_info.is_dq_completed:
            set_quant_run_info(last_run_info, kv_inner_id, 0)
            process_dq(last_run_info, 0, const_info, tensor_info, tmp_info)

    if not skip_s1_1:
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

    if not skip_s1_3:
        set_quant_run_info(run_info, 3, kv_inner_id)
        process_sp(run_info, const_info, 1, tensor_info)
        process_p_ds(run_info, const_info, 1, tensor_info, tmp_info)

    with pl.section_vector():
        if not last_run_info.is_dq_completed:
            set_quant_run_info(last_run_info, kv_inner_id, 2)
            if kv_inner_id == 0:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=SYNC_DETER_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )

            if last_run_info.s1_idx < last_run_info.inner_s1_loop_num:
                copy_dqkv2gm(False, last_run_info, const_info, tensor_info)
            if kv_inner_id == CUBE_BASEM // TS - 1:
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


def drain_pending_task(run_info, const_info, tensor_info, tmp_info):
    if not run_info.is_dkv_completed:
        kv_iib = 3
        with pl.section_vector():
            set_quant_run_info(run_info, 0, kv_iib - 1)
            if run_info.s2_idx < run_info.inner_s2_loop_num:
                copy_dqkv2gm(True, run_info, const_info, tensor_info)
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_TRANSFER_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )

        set_quant_run_info(run_info, 0, kv_iib)
        process_dkv(False, run_info, 0, const_info, tensor_info, tmp_info)
        set_quant_run_info(run_info, 2, kv_iib)
        process_dkv(False, run_info, 2, const_info, tensor_info, tmp_info)

        set_quant_run_info(run_info, 0, kv_iib)
        process_dkv(True, run_info, 0, const_info, tensor_info, tmp_info)
        set_quant_run_info(run_info, 2, kv_iib)
        process_dkv(True, run_info, 2, const_info, tensor_info, tmp_info)

        with pl.section_vector():
            set_quant_run_info(run_info, 0, kv_iib)
            if run_info.s2_idx < run_info.inner_s2_loop_num:
                copy_dqkv2gm(True, run_info, const_info, tensor_info)
            pl.system.set_cross_core(
                pipe=pl.PipeType.MTE3,
                event_id=SYNC_TRANSFER_DKV_FLAG,
                sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
            )

    with pl.section_vector():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=SYNC_DETER_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    if not run_info.is_dq_completed:
        for i in pl.range(4):
            set_quant_run_info(run_info, i, 0)
            process_dq(run_info, 0, const_info, tensor_info, tmp_info)
            set_quant_run_info(run_info, i, 2)
            process_dq(run_info, 2, const_info, tensor_info, tmp_info)
            compute_dq(run_info, const_info, tensor_info)
            with pl.section_vector():
                if run_info.s1_idx < run_info.inner_s1_loop_num:
                    copy_dqkv2gm(False, run_info, const_info, tensor_info)
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3,
                    event_id=SYNC_TRANSFER_DQ_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
                )
    with pl.section_vector():
        pl.system.set_cross_core(
            pipe=pl.PipeType.MTE3,
            event_id=SYNC_DETER_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    run_info.is_dkv_completed = True
    run_info.is_dq_completed = True


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
                        pong_size = TS * TS if pong_off + TS * TS < total else tail
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
    if layout == 2:
        tensor_cu_q = pl.make_tensor(
            cu_seqlens_q, [tiling.b + 1], [1], dtype=pl.DT_INT32
        )
        tensor_cu_kv = pl.make_tensor(
            cu_seqlens_kv, [tiling.b + 1], [1], dtype=pl.DT_INT32
        )
    else:
        tensor_cu_q = tensor_metadata
        tensor_cu_kv = tensor_metadata
    if has_seq_used_q == 1:
        tensor_seq_q = pl.make_tensor(seqused_q, [tiling.b], [1], dtype=pl.DT_INT32)
    elif layout == 2:
        tensor_seq_q = tensor_cu_q
    else:
        tensor_seq_q = tensor_metadata
    if has_seq_used_kv == 1:
        tensor_seq_kv = pl.make_tensor(seqused_kv, [tiling.b], [1], dtype=pl.DT_INT32)
    elif layout == 2:
        tensor_seq_kv = tensor_cu_kv
    else:
        tensor_seq_kv = tensor_metadata
    tensor_attn_mask = pl.make_tensor(
        attn_mask,
        [2048, 2048],
        [2048, 1],
        dtype=pl.DT_INT8,
    )
    gm_b = 1 if layout == 2 else tiling.b
    gm_s1 = tiling.t1 if layout == 2 else tiling.s1
    gm_s2 = tiling.t2 if layout == 2 else tiling.s2
    tensor_workspace_sfmg = pl.make_tensor(
        workspace + tiling.sfmg_work_space_offset,
        [gm_b, tiling.n1, gm_s1],
        [tiling.n1 * gm_s1, gm_s1, 1],
        dtype=pl.DT_FP32,
    )
    tensor_workspace_sfmg_flat = pl.make_tensor(
        workspace + tiling.sfmg_work_space_offset,
        [1, gm_b * tiling.n1 * gm_s1],
        [gm_b * tiling.n1 * gm_s1, 1],
        dtype=pl.DT_FP32,
    )
    if has_sink == 1:
        tensor_sinks = pl.make_tensor(
            sinks,
            [tiling.n1],
            [1],
            dtype=pl.DT_FP32,
        )

    if layout == 0 or layout == 2:  # BSND
        tensor_q = pl.make_tensor(
            q,
            [gm_b, gm_s1, tiling.n1, tiling.d],
            [gm_s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_q_ds = pl.make_tensor(
            q,
            [gm_b, gm_s1, tiling.n1, tiling.d],
            [gm_s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_k = pl.make_tensor(
            k,
            [gm_b, gm_s2, tiling.n2, tiling.d],
            [gm_s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_v = pl.make_tensor(
            v,
            [gm_b, gm_s2, tiling.n2, tiling.d],
            [gm_s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_do = pl.make_tensor(
            dout,
            [gm_b, gm_s1, tiling.n1, tiling.d],
            [gm_s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_do_p_dy = pl.make_tensor(
            dout,
            [gm_b, gm_s1, tiling.n1, tiling.d],
            [gm_s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_attn_out = pl.make_tensor(
            attn_out,
            [gm_b, gm_s1, tiling.n1, tiling.d],
            [gm_s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_softmax_lse = pl.make_tensor(
            softmax_lse,
            [gm_b, tiling.n1, gm_s1],
            [gm_s1 * tiling.n1, gm_s1, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dq = pl.make_tensor(
            workspace + tiling.dq_work_space_offset,
            [gm_b, gm_s1, tiling.n1, tiling.d],
            [gm_s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dq_flat = pl.make_tensor(
            workspace + tiling.dq_work_space_offset,
            [1, gm_b * gm_s1 * tiling.n1 * tiling.d],
            [gm_b * gm_s1 * tiling.n1 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dq_out_flat = pl.make_tensor(
            dq,
            [1, gm_b * gm_s1 * tiling.n1 * tiling.d],
            [gm_b * gm_s1 * tiling.n1 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_workspace_dk = pl.make_tensor(
            workspace + tiling.dk_work_space_offset,
            [gm_b, gm_s2, tiling.n2, tiling.d],
            [gm_s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dk_flat = pl.make_tensor(
            workspace + tiling.dk_work_space_offset,
            [1, gm_b * gm_s2 * tiling.n2 * tiling.d],
            [gm_b * gm_s2 * tiling.n2 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dk_out_flat = pl.make_tensor(
            dk,
            [1, gm_b * gm_s2 * tiling.n2 * tiling.d],
            [gm_b * gm_s2 * tiling.n2 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_workspace_dv = pl.make_tensor(
            workspace + tiling.dv_work_space_offset,
            [gm_b, gm_s2, tiling.n2, tiling.d],
            [gm_s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dv_flat = pl.make_tensor(
            workspace + tiling.dv_work_space_offset,
            [1, gm_b * gm_s2 * tiling.n2 * tiling.d],
            [gm_b * gm_s2 * tiling.n2 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dv_out_flat = pl.make_tensor(
            dv,
            [1, gm_b * gm_s2 * tiling.n2 * tiling.d],
            [gm_b * gm_s2 * tiling.n2 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dq_out = pl.make_tensor(
            dq,
            [gm_b, gm_s1, tiling.n1, tiling.d],
            [gm_s1 * tiling.n1 * tiling.d, tiling.n1 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dk_out = pl.make_tensor(
            dk,
            [gm_b, gm_s2, tiling.n2, tiling.d],
            [gm_s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dv_out = pl.make_tensor(
            dv,
            [gm_b, gm_s2, tiling.n2, tiling.d],
            [gm_s2 * tiling.n2 * tiling.d, tiling.n2 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
    elif layout == 1:  # BNSD
        tensor_q = pl.make_tensor(
            q,
            [gm_b, tiling.n1, gm_s1, tiling.d],
            [tiling.n1 * gm_s1 * tiling.d, gm_s1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_q_ds = pl.make_tensor(
            q,
            [gm_b, tiling.n1, gm_s1, tiling.d],
            [tiling.n1 * gm_s1 * tiling.d, gm_s1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_k = pl.make_tensor(
            k,
            [gm_b, tiling.n2, gm_s2, tiling.d],
            [tiling.n2 * gm_s2 * tiling.d, gm_s2 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_v = pl.make_tensor(
            v,
            [gm_b, tiling.n2, gm_s2, tiling.d],
            [tiling.n2 * gm_s2 * tiling.d, gm_s2 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_do = pl.make_tensor(
            dout,
            [gm_b, tiling.n1, gm_s1, tiling.d],
            [tiling.n1 * gm_s1 * tiling.d, gm_s1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_do_p_dy = pl.make_tensor(
            dout,
            [gm_b, tiling.n1, gm_s1, tiling.d],
            [tiling.n1 * gm_s1 * tiling.d, gm_s1 * tiling.d, tiling.d, 1],
            dtype=input_dtype,
        )
        tensor_attn_out = pl.make_tensor(
            attn_out,
            [gm_b, tiling.n1, gm_s1, tiling.d],
            [tiling.n1 * gm_s1 * tiling.d, gm_s1 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_softmax_lse = pl.make_tensor(
            softmax_lse,
            [gm_b, tiling.n1, gm_s1],
            [tiling.n1 * gm_s1, gm_s1, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dq = pl.make_tensor(
            workspace + tiling.dq_work_space_offset,
            [gm_b, tiling.n1, gm_s1, tiling.d],
            [tiling.n1 * gm_s1 * tiling.d, gm_s1 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dk = pl.make_tensor(
            workspace + tiling.dk_work_space_offset,
            [gm_b, tiling.n2, gm_s2, tiling.d],
            [tiling.n2 * gm_s2 * tiling.d, gm_s2 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dv = pl.make_tensor(
            workspace + tiling.dv_work_space_offset,
            [gm_b, tiling.n2, gm_s2, tiling.d],
            [tiling.n2 * gm_s2 * tiling.d, gm_s2 * tiling.d, tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dq_out = pl.make_tensor(
            dq,
            [gm_b, tiling.n1, gm_s1, tiling.d],
            [tiling.n1 * gm_s1 * tiling.d, gm_s1 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dk_out = pl.make_tensor(
            dk,
            [gm_b, tiling.n2, gm_s2, tiling.d],
            [tiling.n2 * gm_s2 * tiling.d, gm_s2 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dv_out = pl.make_tensor(
            dv,
            [gm_b, tiling.n2, gm_s2, tiling.d],
            [tiling.n2 * gm_s2 * tiling.d, gm_s2 * tiling.d, tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_workspace_dq_flat = pl.make_tensor(
            workspace + tiling.dq_work_space_offset,
            [1, gm_b * tiling.n1 * gm_s1 * tiling.d],
            [gm_b * tiling.n1 * gm_s1 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dk_flat = pl.make_tensor(
            workspace + tiling.dk_work_space_offset,
            [1, gm_b * tiling.n2 * gm_s2 * tiling.d],
            [gm_b * tiling.n2 * gm_s2 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_workspace_dv_flat = pl.make_tensor(
            workspace + tiling.dv_work_space_offset,
            [1, gm_b * tiling.n2 * gm_s2 * tiling.d],
            [gm_b * tiling.n2 * gm_s2 * tiling.d, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dq_out_flat = pl.make_tensor(
            dq,
            [1, gm_b * tiling.n1 * gm_s1 * tiling.d],
            [gm_b * tiling.n1 * gm_s1 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dk_out_flat = pl.make_tensor(
            dk,
            [1, gm_b * tiling.n2 * gm_s2 * tiling.d],
            [gm_b * tiling.n2 * gm_s2 * tiling.d, 1],
            dtype=output_dtype,
        )
        tensor_dv_out_flat = pl.make_tensor(
            dv,
            [1, gm_b * tiling.n2 * gm_s2 * tiling.d],
            [gm_b * tiling.n2 * gm_s2 * tiling.d, 1],
            dtype=output_dtype,
        )
    core_id_cube = pl.get_block_idx() // pl.get_subblock_num()
    sub_id = pl.get_subblock_idx()
    core_id_vec = core_id_cube * pl.get_subblock_num() + sub_id
    core_num = pl.get_block_num()
    deq_scale_q_value = pl.getval(tensor_q_descale, 0)
    deq_scale_k_value = pl.getval(tensor_k_descale, 0)
    deq_scale_v_value = pl.getval(tensor_v_descale, 0)
    deq_scale_do_value = pl.getval(tensor_do_descale, 0)
    scale_p_value = pl.getval(tensor_p_scale, 0)
    scale_ds_value = pl.getval(tensor_ds_scale, 0)
    const_info = pl.make_tuple(
        b_size=tiling.b,
        gm_b_size=gm_b,
        gm_s1_size=gm_s1,
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
        mask_mode=tiling.mask_mode,
        s1_token=tiling.s1_token,
        s2_token=tiling.s2_token,
        deter_max_round=tiling.deter_max_round,
        scale_p=scale_p_value,
        scale_ds=scale_ds_value,
        deq_scale_p_value=1.0 / scale_p_value,
        deq_scale_ds_value=1.0 / scale_ds_value,
        deq_scale_q_value=deq_scale_q_value,
        deq_scale_k_value=deq_scale_k_value,
        deq_scale_v_value=deq_scale_v_value,
        deq_scale_do_value=deq_scale_do_value,
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
        metadata_len=tiling.metadata_len,
        q_pre_block_factor=tiling.q_pre_block_factor,
        q_pre_block_total=tiling.q_pre_block_total,
        q_pre_block_tail=tiling.q_pre_block_tail,
        k_pre_block_factor=tiling.k_pre_block_factor,
        k_pre_block_total=tiling.k_pre_block_total,
        k_pre_block_tail=tiling.k_pre_block_tail,
        v_pre_block_factor=tiling.v_pre_block_factor,
        v_pre_block_total=tiling.v_pre_block_total,
        v_pre_block_tail=tiling.v_pre_block_tail,
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
            shape=[TS, HALF_D_SIZE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=VA3,
        mutex_ids=[31],
    )
    lse_n = 128 if qk_s1_fuse() else 64
    lse_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, lse_n], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA4, VA4 + 512, VA4 + 512 * 2, VA4 + 512 * 3],
        mutex_ids=[9, 10, 11, 12],
    )
    d_vec = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, lse_n], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec
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
    attn_mask_pool = pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 64], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA4 + 512 * 10, VA4 + 512 * 26, VA4 + 512 * 42],
        mutex_ids=[19, 21, 20],
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
    acc_n = TS * 2 if qk_s1_fuse() else TS
    acc_db_addr = CA0 + TS * TS * 8 if qk_s1_fuse() else CA2
    acc_mm1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, acc_n],
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
        addrs=CA0 if qk_s1_fuse() else CA1,
        mutex_ids=[2],
    )
    acc_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS, TS],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc,
            compact=1,
        ),
        addrs=acc_db_addr,
        mutex_ids=[4, 6],
    )
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
        tensor_cu_q=tensor_cu_q,
        tensor_cu_kv=tensor_cu_kv,
        tensor_seq_q=tensor_seq_q,
        tensor_seq_kv=tensor_seq_kv,
        tensor_workspace_sfmg_flat=tensor_workspace_sfmg_flat,
        tensor_p_scale=tensor_p_scale,
        tensor_ds_scale=tensor_ds_scale,
        tensor_metadata=tensor_metadata,
        tensor_attn_mask=tensor_attn_mask,
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
        attn_mask_pool=attn_mask_pool,
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
        gm_batch_id=0,
        q_start=0,
        kv_start=0,
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
        s2_outer_cur=0,
        s1_token=0,
        s2_token=0,
        inner_s1_loop_num=0,
        inner_s2_loop_num=0,
        inner_s1_real_size=[0, 0, 0, 0],
        inner_s2_real_size=[0, 0, 0, 0],
        is_valid_inner_block=init_data,
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
        s1_token=0,
        s2_token=0,
        actual_s1_len=0,
        actual_s2_len=0,
        m_offset=0,
        n_offset=0,
        mask_mode=0,
    )
    next_core_first_block_coordinate_info = coordinate_infos[2]
    band_infos = pl.struct_array(
        1,
        "band_info",
        k=0,
        m=0,
        n=0,
        p=0,
        q=0,
        b=0,
        b1=0,
        b2=0,
        l1=0,
        l2=0,
        l3=0,
        n_seg=0,
        r1=0,
        r2=0,
        r3=0,
        rm_batch=0,
        rm=0,
        rm2=0,
        m_offset=0,
        n_offset=0,
    )
    band_info = band_infos[0]

    if is_var_len():
        init_sfmg_workspace(const_info, tensor_info)
        pl.system.sync_all()
    presfmg_quant_inner_hif8(const_info, tensor_info)
    pl.system.sync_all()

    alloc_event_id()
    if sparse_type == 4 and not is_var_len():
        gen_band_info(const_info, band_info)
    # pl.range / cal_deter_index 的上界必须来自 tiling (host 标量).
    # metadata 槽0 是 GM, 赋给 loop_max 会让 CCE 报「定义前使用」.
    loop_max = 0
    if is_var_len():
        loop_max = const_info.deter_max_round
        meta_rounds = meta_get(const_info, tensor_metadata, META_DETER_MAX_NUM_IDX)
    else:
        loop_max = cal_deter_max_loop_num(const_info, band_info)
    task_id = 0
    next_valid_loop_idx = 0
    next_block_idx = 0
    init_coordinate_info(const_info, 0, 0, coordinate_infos[0])
    init_coordinate_info(const_info, 0, 0, coordinate_infos[1])
    if sparse_type == 4 and not is_var_len():
        init_coordinate_info(
            const_info, band_info.m_offset, band_info.n_offset, coordinate_infos[0]
        )
        init_coordinate_info(
            const_info, band_info.m_offset, band_info.n_offset, coordinate_infos[1]
        )
    if not is_var_len():
        with pl.section_vector():
            if const_info.s2_outer == 1:
                init_coordinate_info(
                    const_info, 0, 0, next_core_first_block_coordinate_info
                )
                if sparse_type == 4:
                    init_coordinate_info(
                        const_info,
                        band_info.m_offset,
                        band_info.n_offset,
                        next_core_first_block_coordinate_info,
                    )
                next_valid_loop_idx, next_block_idx = cal_deter_index(
                    0,
                    loop_max,
                    next_core_first_block_coordinate_info,
                    const_info,
                    band_info,
                    tensor_metadata,
                    tensor_seq_q,
                    tensor_seq_kv,
                    True,
                )
    next_valid_loop_idx, next_block_idx = cal_deter_index(
        0,
        loop_max,
        coordinate_infos[task_id % 2],
        const_info,
        band_info,
        tensor_metadata,
        tensor_seq_q,
        tensor_seq_kv,
    )
    for loop_idx in pl.range(loop_max):
        if is_var_len():
            if loop_idx >= meta_rounds:
                break
        block_inner_idx = next_block_idx
        if loop_idx >= next_valid_loop_idx:
            block_inner_idx = next_block_idx
            next_valid_loop_idx, next_block_idx = cal_deter_index(
                loop_idx + 1,
                loop_max,
                coordinate_infos[(task_id + 1) % 2],
                const_info,
                band_info,
                tensor_metadata,
                tensor_seq_q,
                tensor_seq_kv,
            )
            run_infos[task_id % 2].is_dkv_completed = False
            run_infos[task_id % 2].is_dq_completed = False
        else:
            block_inner_idx = -1
        is_valid_block = block_inner_idx >= 0
        if is_valid_block:
            if not is_var_len():
                is_valid_block = is_valid_for_deter(block_inner_idx, const_info)
        if is_valid_block:
            # A gap may have already drained the previous task. Match the
            # single dQ uniqueness barrier even when this core has no pending dQ.
            if loop_idx > 0 and run_infos[(task_id + 1) % 2].is_dq_completed:
                deter_empty_round_barrier()
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
            if loop_idx > 0:
                drain_pending_task(
                    run_infos[(task_id + 1) % 2],
                    const_info,
                    tensor_info,
                    tmp_info[0],
                )
        if is_valid_block:
            task_id += 1

    drain_pending_task(
        run_infos[(task_id + 1) % 2], const_info, tensor_info, tmp_info[0]
    )
    free_event_id()
    pl.system.sync_all()

    # # process post
    process_post_dqkv(const_info, tensor_info)
