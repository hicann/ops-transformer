# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Forward-precision cache reconstruction inside the existing grad kernel.

BF16 operand packing and Cube contractions retain FP32 recurrent state.
All implementation lives here. The grad metadata and owner schedule are reused;
no second launch or separate forward schedule is created. Aqk/Akk and activation/norm caches stay read-only.
"""

from cannbotdsl import PIPE, MemLoc, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang import vf
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import convert, copy_options, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    update_mask,
    vabs,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdups,
    vexp,
    vexp_sub,
    vload,
    vload_broadcast,
    vload_unpack,
    vlog,
    vmadd,
    vmaxs,
    vmem_bar,
    vmul,
    vmuls,
    vpack,
    vreinterpret,
    vreinterpret_lanes,
    vstore,
    vstore_first,
    vstore_pack,
    vstore_strided,
    vsub,
)
from cannbotdsl.ops.reg import vselect as vselect_raw
from cannbotdsl.ops.sync import (
    cube_fill_l1_zero,
    cube_sync_all,
    cube_sync_pipe,
    vec_sync_all,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.tensor import reinterpret, tile_slice
from cannbotdsl import Constexpr
from cannbotdsl import make_layout, permute
from cannbotdsl.ring_buffer import RingBuffer

try:
    from . import flash_kda_grad as grad
except ImportError:
    import flash_kda_grad as grad

_cube_buffer = grad._cube_buffer
_cube_ring = grad._cube_ring

# Cross-core transfers use paired single-slot channels; same-core scratch
# uses plain buffers. Each chunk consumes its operands before recurrence.
C, D, HALF = 64, 128, 64
CHUNK_SIZE, SUPPORTED_HEAD_DIM, D_HALF = C, D, HALF
K_RESTORED_OPERAND_INDEX = 1
K_DECAYED_BETA_OPERAND_INDEX = 2
V_BETA_OPERAND_INDEX = 3
# qg/kg plus three BF16 components of weighted K and V.
RECOMPUTE_OPERAND_PLANES = 8


def cast_and_pack_bf16(value, lane_mask):
    value_bf16 = vcast(value, dtypes.bfloat16, mask=lane_mask)
    packed_value = vpack(vreinterpret_lanes(value_bf16, dtypes.uint32), dtypes.uint16)
    return vreinterpret(packed_value, dtypes.bfloat16)


@jit
def load_sequence_tile(
    dst, src, tile_idx, column_idx, sequence_length, has_tail, cache_policy=0
):
    """Load one ND tile; explicitly zero invalid rows of a partial sequence."""
    rows, cols = dst.shape
    source_tile = tile_slice(src, (rows, cols), (tile_idx, column_idx))
    if const_expr(has_tail):
        first_row = tile_idx * rows
        if first_row + rows <= sequence_length:
            mem_copy(dst, source_tile, options=copy_options(l2_cache_ctl=cache_policy))
        else:
            with vf(mode="simd"):
                for offset in range(0, rows * cols, D_HALF):
                    lane_mask, _ = update_mask(rows * cols - offset, elem_bits=32)
                    zero_fp32 = vdups(0.0, dtypes.float32, mask=lane_mask)
                    if const_expr(dst.dtype == dtypes.float32):
                        vstore(dst, offset, zero_fp32, lane_mask)
                    else:
                        zero_bf16 = vcast(zero_fp32, dtypes.bfloat16, mask=lane_mask)
                        vstore_pack(
                            dst,
                            offset,
                            zero_bf16,
                            lane_mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                vmem_bar("vst_vld")
            if first_row < sequence_length:
                valid_rows = sequence_length - first_row
                mem_copy(
                    dst[:valid_rows, None],
                    source_tile[:valid_rows, None],
                    options=copy_options(l2_cache_ctl=cache_policy),
                )
    else:
        mem_copy(dst, source_tile, options=copy_options(l2_cache_ctl=cache_policy))


@jit
def _zero_fp32(tensor, elements: Constexpr[int]):
    with vf(mode="simd"):
        mask, _ = update_mask(HALF, elem_bits=32)
        zero = vdups(0.0, dtypes.float32, mask=mask)
        for offset in range(0, elements, HALF):
            vstore(tensor, offset, zero, mask)
        vmem_bar("vst_vld")


@jit
def _clear_padded_caches(
    w, u, v_new, qg, kg, gamma, batch, head, length, capacity, part, low=None, high=None
):
    """Publish deterministic padding before the backward chunk consumers run."""
    if const_expr(low is None):
        low = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((C, HALF)))
        high = Buffer(MemLoc.UB, dtypes.float32, make_layout((C, HALF)))
    with vf(mode="simd"):
        mask, _ = update_mask(HALF, elem_bits=32)
        zero = vdups(0.0, dtypes.float32, mask=mask)
        zero_low = vcast(zero, dtypes.bfloat16, mask=mask)
        for offset in range(0, C * HALF, HALF):
            vstore(high, offset, zero, mask)
            vstore_pack(low, offset, zero_low, mask, pack_mode=PackMode.B32_TO_B16)
        vmem_bar("vst_vld")
    for start in range(length, capacity, C):
        rows = min(C, capacity - start)
        for destination in (w, u, v_new, qg, kg):
            window = destination[batch, head, start : start + rows, None]
            mem_copy(tile_slice(window, (C, HALF), (0, part)), low[:rows, None])
        window = gamma[batch, head, start : start + rows, None]
        mem_copy(tile_slice(window, (C, HALF), (0, part)), high[:rows, None])


class _Preprocess:
    def __init__(self, part, gate_dtype, beta_dtype, operands=None):
        self.subblock_idx = part
        self.ub_g_raw = Buffer(MemLoc.UB, gate_dtype, make_layout((C, HALF)))
        self.ub_v_raw = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((C, HALF)))
        self.ub_q_norm = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((C, HALF)))
        self.ub_k_norm = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((C, HALF)))
        self.ub_beta_raw = Buffer(MemLoc.UB, beta_dtype, make_layout((C, 1)))
        self.ub_beta = Buffer(MemLoc.UB, dtypes.float32, make_layout((C, 1)))
        self.ub_dt_bias = Buffer(MemLoc.UB, dtypes.float32, make_layout((HALF,)))
        self.ub_neg_exp_a = Buffer(MemLoc.UB, dtypes.float32, make_layout((1,)))
        self.ub_g_cumsum = Buffer(MemLoc.UB, dtypes.float32, make_layout((C, HALF)))
        self.ub_recurrent_operands_nz = Buffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((RECOMPUTE_OPERAND_PLANES * C, HALF)),
            data_format="nz",
            physical_layout=make_layout(
                (
                    ((RECOMPUTE_OPERAND_PLANES * C, HALF)[1] + 15) // 16,
                    ((RECOMPUTE_OPERAND_PLANES * C, HALF)[0] + 15) // 16,
                    16,
                    16,
                ),
                stride=(
                    ((RECOMPUTE_OPERAND_PLANES * C, HALF)[0] + 15) // 16 * 256 + 16,
                    256,
                    16,
                    1,
                ),
            ),
        )
        self.l1_recurrent_operands = (
            operands
            if operands is not None
            else RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((RECOMPUTE_OPERAND_PLANES * C, D)),
                depth=1,
            )
        )
        self.operand_tiles = None
        self.has_tail = True
        self.sequence_length = None
        self.has_dt_bias = None
        self.use_gate_in_kernel = None
        self.has_lower_bound = None
        self.fused_beta = None
        self.save_qg = 1
        self.cache_writer = None
        self.saved_qg_nd = None
        self.saved_kg_nd = self.ub_v_raw

    @jit
    def _vf_select_beta_mode(self, bound, rows, gate_mode, has_dt_bias):
        if self.fused_beta:
            self._vf_activate_gate_beta(bound, rows, gate_mode, True, has_dt_bias)
        else:
            self._vf_activate_gate_beta(bound, rows, gate_mode, False, has_dt_bias)

    @jit
    def prepare(self, q, k, v, g, beta, chunk, bound):
        # Q/K are already the saved normalized values. Load complete half-D
        # tiles instead of issuing four tiny DMA operations for every token.
        load_sequence_tile(
            self.ub_q_norm, q, chunk, self.subblock_idx, self.sequence_length, True, 4
        )
        load_sequence_tile(
            self.ub_k_norm, k, chunk, self.subblock_idx, self.sequence_length, True, 4
        )
        load_sequence_tile(
            self.ub_v_raw, v, chunk, self.subblock_idx, self.sequence_length, True, 4
        )
        load_sequence_tile(
            self.ub_g_raw, g, chunk, self.subblock_idx, self.sequence_length, True, 4
        )
        load_sequence_tile(
            self.ub_beta_raw, beta, chunk, 0, self.sequence_length, True, 4
        )
        vec_sync_notify(PIPE.MTE2, PIPE.V, 3)
        self.activate_gate_and_beta(bound, chunk)

    @jit
    def load_head(self, a_log, dt_bias, value_head_idx):
        # These plain UB buffers stay live through the preceding gate/operand
        # VFs. Unowned staged tasks can advance to the next head without a
        # Cube handoff; retire those reads before MTE2 replaces A_log/bias.
        vec_sync_all()
        mem_copy(self.ub_neg_exp_a, tile_slice(a_log, (1,), (value_head_idx,)))
        if self.has_dt_bias:
            mem_copy(
                self.ub_dt_bias,
                tile_slice(
                    dt_bias[value_head_idx, None], (D_HALF,), (self.subblock_idx,)
                ),
            )
        # A_log and bias must both be ready before their first VF consumer.
        vec_sync_all()
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            exp_a = vexp(vload_broadcast(self.ub_neg_exp_a, 0), mask=lane_mask)
            vstore_first(self.ub_neg_exp_a, 0, vmuls(exp_a, -1.0, mask=lane_mask))

    @jit
    def activate_gate_and_beta(self, bound, chunk_idx):
        vec_sync_wait(PIPE.MTE2, PIPE.V, 3)
        valid_rows = CHUNK_SIZE
        if const_expr(self.has_tail):
            valid_rows = self.sequence_length - chunk_idx * CHUNK_SIZE
            if valid_rows > CHUNK_SIZE:
                valid_rows = CHUNK_SIZE
        if self.use_gate_in_kernel != 0:
            if self.has_lower_bound != 0:
                self._vf_select_dt_bias_mode(bound, valid_rows, 2)
            else:
                self._vf_select_dt_bias_mode(bound, valid_rows, 1)
        else:
            self._vf_select_beta_mode(bound, valid_rows, 0, False)

    @jit
    def _vf_select_dt_bias_mode(self, bound, valid_rows, gate_mode):
        if self.has_dt_bias:
            self._vf_select_beta_mode(bound, valid_rows, gate_mode, True)
        else:
            self._vf_select_beta_mode(bound, valid_rows, gate_mode, False)

    @jit
    def _vf_activate_gate_beta(
        self, bound, valid_rows, gate_mode, beta_cached, has_dt_bias
    ):
        raw_gate = self.ub_g_raw
        raw_beta = self.ub_beta_raw
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            beta_mask, _ = update_mask(CHUNK_SIZE, elem_bits=32)
            negative_exp_a = vdups(0.0, dtypes.float32, mask=lane_mask)
            dt_bias = vdups(0.0, dtypes.float32, mask=lane_mask)
            if const_expr(gate_mode != 0):
                negative_exp_a = vload_broadcast(self.ub_neg_exp_a, 0)
                if const_expr(has_dt_bias):
                    dt_bias = vload(self.ub_dt_bias, 0)
            prefix = vdups(0.0, dtypes.float32, mask=lane_mask)
            if const_expr(raw_beta.dtype == dtypes.float32):
                beta_value = vload(raw_beta, 0)
            else:
                beta_value = vcast(
                    vload_unpack(raw_beta, 0, unpack_mode=UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=beta_mask,
                )
            # Raw and activated beta retain their public precision. Only the
            # stored cache tensors are BF16; Cube operand rounding is recovered
            # by a second product with the BF16 residual below.
            valid_beta_mask, _ = update_mask(valid_rows, elem_bits=32)
            beta_value = vselect_raw(
                beta_value,
                vdups(0.0, dtypes.float32, mask=beta_mask),
                cond_mask=valid_beta_mask,
            )
            vstore(self.ub_beta, 0, beta_value, beta_mask)
            for row in dsl_range(0, valid_rows, 1):
                if const_expr(raw_gate.dtype == dtypes.float32):
                    gate_value = vload(raw_gate, row * D_HALF)
                else:
                    gate_value = vcast(
                        vload_unpack(
                            raw_gate, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=lane_mask,
                    )
                gate = gate_value
                if const_expr(gate_mode != 0):
                    biased_gate = vadd(gate_value, dt_bias, mask=lane_mask)
                    if const_expr(gate_mode == 2):
                        logit = vmul(negative_exp_a, biased_gate, mask=lane_mask)
                        gate = vdiv(
                            vdups(bound, dtypes.float32, mask=lane_mask),
                            vadds(vexp(logit, mask=lane_mask), 1.0, mask=lane_mask),
                            mask=lane_mask,
                        )
                    else:
                        softplus_tail = vlog(
                            vadds(
                                vexp(
                                    vmuls(
                                        vabs(biased_gate, mask=lane_mask),
                                        -1.0,
                                        mask=lane_mask,
                                    ),
                                    mask=lane_mask,
                                ),
                                1.0,
                                mask=lane_mask,
                            ),
                            mask=lane_mask,
                        )
                        softplus = vadd(
                            vmaxs(biased_gate, 0.0, mask=lane_mask),
                            softplus_tail,
                            mask=lane_mask,
                        )
                        gate = vmul(negative_exp_a, softplus, mask=lane_mask)
                prefix = vadd(prefix, gate, mask=lane_mask)
                vstore(self.ub_g_cumsum, row * D_HALF, prefix, lane_mask)
            if const_expr(self.has_tail):
                for row in dsl_range(valid_rows, CHUNK_SIZE, 1):
                    vstore(self.ub_g_cumsum, row * D_HALF, prefix, lane_mask)
            vmem_bar("vst_vld")

    @jit
    def prepare_recurrent_operands(self, scale, chunk_decay, chunk_idx):
        if self.save_qg != 0:
            self._vf_pack_recurrent_operands(scale, chunk_decay, True)
        else:
            self._vf_pack_recurrent_operands(scale, chunk_decay, False)
        mem_copy(
            self.l1_recurrent_operands.next_write().partition(
                dim=1, part_id=self.subblock_idx
            ),
            self.ub_recurrent_operands_nz,
        )
        rows = self.cache_writer.valid_rows(chunk_idx)
        if self.save_qg != 0:
            mem_copy(
                tile_slice(
                    self.cache_writer.qg,
                    (CHUNK_SIZE, D_HALF),
                    (chunk_idx, self.subblock_idx),
                )[:rows, None],
                self.saved_qg_nd[:rows, None],
                options=copy_options(l2_cache_ctl=4),
            )
        if self.save_qg != 0:
            mem_copy(
                tile_slice(
                    self.cache_writer.kg,
                    (CHUNK_SIZE, D_HALF),
                    (chunk_idx, self.subblock_idx),
                )[:rows, None],
                self.saved_kg_nd[:rows, None],
                options=copy_options(l2_cache_ctl=4),
            )

    def acquire_recurrent_operands(self):
        operands_l1_read = self.l1_recurrent_operands.next_read()
        self.operand_tiles = [
            tile_slice(
                operands_l1_read, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (operand_idx, 0)
            )
            for operand_idx in range(RECOMPUTE_OPERAND_PLANES)
        ]

    @jit
    def _vf_pack_recurrent_operands(self, scale, chunk_decay, save_inputs):
        raw_value = self.ub_v_raw
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            low_mask, _ = update_mask(D_HALF, elem_bits=16)
            last_gate_cumsum = vload(self.ub_g_cumsum, (CHUNK_SIZE - 1) * D_HALF)
            vstore(chunk_decay, 0, vexp(last_gate_cumsum, mask=lane_mask), lane_mask)
            for row in dsl_range(0, CHUNK_SIZE, 1):
                element_offset = row * D_HALF
                gate_prefix_sum = vload(self.ub_g_cumsum, element_offset)
                decay_scale = vexp(gate_prefix_sum, mask=lane_mask)
                restore_scale = vexp_sub(
                    last_gate_cumsum, gate_prefix_sum, mask=lane_mask
                )
                q_value = vcast(
                    vload_unpack(
                        self.ub_q_norm,
                        element_offset,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=lane_mask,
                )
                k_value = vcast(
                    vload_unpack(
                        self.ub_k_norm,
                        element_offset,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=lane_mask,
                )
                v_value = vcast(
                    vload_unpack(
                        raw_value, element_offset, unpack_mode=UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=lane_mask,
                )
                raw_beta = vload_broadcast(self.ub_beta, row)
                qg_value = vmul(q_value, decay_scale, mask=lane_mask)
                kg_value = vmul(k_value, restore_scale, mask=lane_mask)
                if const_expr(save_inputs):
                    vstore_pack(
                        self.saved_qg_nd,
                        element_offset,
                        vcast(qg_value, dtypes.bfloat16, mask=lane_mask),
                        lane_mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                if const_expr(save_inputs):
                    vstore_pack(
                        self.saved_kg_nd,
                        element_offset,
                        vcast(kg_value, dtypes.bfloat16, mask=lane_mask),
                        lane_mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                operand_values = (
                    vmuls(qg_value, scale, mask=lane_mask),
                    kg_value,
                    vmul(
                        vmul(k_value, decay_scale, mask=lane_mask),
                        raw_beta,
                        mask=lane_mask,
                    ),
                    vmul(v_value, raw_beta, mask=lane_mask),
                )
                for operand_idx in (0, 1, 2, 3):
                    operand_bf16 = cast_and_pack_bf16(
                        operand_values[operand_idx], lane_mask
                    )
                    vstore_strided(
                        self.ub_recurrent_operands_nz,
                        (operand_idx * CHUNK_SIZE + row) * 16,
                        operand_bf16,
                        low_mask,
                        block_stride=self.ub_recurrent_operands_nz.physical_stride[0]
                        // 16,
                    )
                for operand_idx in (2, 3):
                    value = operand_values[operand_idx]
                    high = vcast(
                        vcast(value, dtypes.bfloat16, mask=lane_mask),
                        dtypes.float32,
                        mask=lane_mask,
                    )
                    # Two BF16 components retain about 16 mantissa bits. Keep
                    # the remaining FP32 bits until the W/U cache is rounded.
                    residual_value = vsub(value, high, mask=lane_mask)
                    low = vcast(residual_value, dtypes.bfloat16, mask=lane_mask)
                    residual = cast_and_pack_bf16(residual_value, lane_mask)
                    vstore_strided(
                        self.ub_recurrent_operands_nz,
                        ((operand_idx + 2) * CHUNK_SIZE + row) * 16,
                        residual,
                        low_mask,
                        block_stride=self.ub_recurrent_operands_nz.physical_stride[0]
                        // 16,
                    )
                    remainder = vsub(
                        residual_value,
                        vcast(low, dtypes.float32, mask=lane_mask),
                        mask=lane_mask,
                    )
                    vstore_strided(
                        self.ub_recurrent_operands_nz,
                        ((operand_idx + 4) * CHUNK_SIZE + row) * 16,
                        cast_and_pack_bf16(remainder, lane_mask),
                        low_mask,
                        block_stride=self.ub_recurrent_operands_nz.physical_stride[0]
                        // 16,
                    )


class _Matmul:
    def __init__(self):
        self.resident_inv_l1 = Buffer(MemLoc.L1, dtypes.bfloat16, make_layout((C, C)))
        self.l0a_recurrent = _cube_buffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((C, D))
        )
        self.l0b_recurrent = _cube_buffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((D, D))
        )
        self.l0c_w_product = _cube_buffer(
            MemLoc.L0C, dtypes.float32, make_layout((C, D))
        )
        self.l0c_w = _cube_buffer(
            MemLoc.L0C, dtypes.float32, make_layout((C, D)), _slot=1
        )
        self.l0c_state = _cube_buffer(MemLoc.L0C, dtypes.float32, make_layout((D, D)))
        self.w_l1 = Buffer(MemLoc.L1, dtypes.bfloat16, make_layout((C, D)))
        self.save_backward_inputs = 1
        self.saved_w = None
        self.saved_u = None
        self.saved_v_new = None
        self.chunk_idx = None
        self.valid_rows = None

    @jit
    def load_inverse(self, inverse, rows):
        # The retained triangular inverse is BF16. Clear invalid tail rows
        # before a full 64x64 Cube load; never read beyond the valid source.
        if rows < C:
            self.zero_fp16_l1(self.resident_inv_l1)
            # Finish the fill before DMA writes the retained inverse into L1.
            cube_sync_pipe(PIPE.MTE2)
        source = tile_slice(inverse, (C, C), (0, 0))
        mem_copy(self.resident_inv_l1[:rows, None], source[:rows, None])

    @jit
    def correct_values(self, state):
        state_l0b = self.l0b_recurrent
        mem_copy(state_l0b, state.state_l1.next_read())
        w_l0a = self.l0a_recurrent
        mem_copy(w_l0a, self.w_l1)
        # Forward publishes -W in BF16 and accumulates U + (-W) @ BF16(H).
        matmul(self.l0c_w, w_l0a, state_l0b, init=False)
        mem_copy(state.u_l1, self.l0c_w)
        mem_copy(
            tile_slice(self.saved_v_new, (C, D), (self.chunk_idx, 0))[
                : self.valid_rows, None
            ],
            self.l0c_w[: self.valid_rows, None],
            options=copy_options(l2_cache_ctl=4),
        )

    @jit
    def state_delta(self, preprocess, state):
        k_panel = reinterpret(
            self.l0b_recurrent, data_format="nz", layout=make_layout((D, HALF))
        )
        mem_copy(
            k_panel, permute(preprocess.operand_tiles[K_RESTORED_OPERAND_INDEX], (1, 0))
        )
        values = state.u_l1
        u_l0a = reinterpret(
            self.l0a_recurrent, data_format="nz", layout=make_layout((D, C))
        )
        mem_copy(u_l0a, permute(values, (1, 0)))
        matmul(self.l0c_state, u_l0a, k_panel, init=True)
        mem_copy(state.ub_state_delta.next_write(), self.l0c_state.partition(dim=1))

    @staticmethod
    def zero_fp16_l1(l1):
        cube_fill_l1_zero(l1, repeat=256, blk_num=1, dst_gap=0)

    @jit
    def compute_u_pre_and_w(self, preprocessing, recurrent_state):
        inverse_l0a = self._load_resident_inv_to_l0a()
        k_beta_l0b = self._load_beta_to_l0b(
            preprocessing.operand_tiles[K_DECAYED_BETA_OPERAND_INDEX]
        )
        matmul(
            self.l0c_w_product,
            reinterpret(
                inverse_l0a, offset=0, layout=make_layout((CHUNK_SIZE, CHUNK_SIZE))
            ),
            k_beta_l0b,
            init=True,
        )
        k_beta_low = self._load_beta_to_l0b(preprocessing.operand_tiles[4])
        matmul(
            self.l0c_w_product,
            reinterpret(
                inverse_l0a, offset=0, layout=make_layout((CHUNK_SIZE, CHUNK_SIZE))
            ),
            k_beta_low,
            init=False,
        )
        k_beta_remainder = self._load_beta_to_l0b(preprocessing.operand_tiles[6])
        matmul(
            self.l0c_w_product,
            reinterpret(inverse_l0a, layout=make_layout((C, C))),
            k_beta_remainder,
            init=False,
        )
        # Round W once at the cache boundary, after all three operand products.
        mem_copy(self.w_l1, convert(self.l0c_w_product, self.w_l1.dtype, scale=-1.0))
        v_beta_l0b = self._load_beta_to_l0b(
            preprocessing.operand_tiles[V_BETA_OPERAND_INDEX]
        )
        matmul(
            self.l0c_w,
            reinterpret(
                inverse_l0a, offset=0, layout=make_layout((CHUNK_SIZE, CHUNK_SIZE))
            ),
            v_beta_l0b,
            init=True,
        )
        v_beta_low = self._load_beta_to_l0b(preprocessing.operand_tiles[5])
        matmul(
            self.l0c_w,
            reinterpret(
                inverse_l0a, offset=0, layout=make_layout((CHUNK_SIZE, CHUNK_SIZE))
            ),
            v_beta_low,
            init=False,
        )
        v_beta_remainder = self._load_beta_to_l0b(preprocessing.operand_tiles[7])
        matmul(
            self.l0c_w,
            reinterpret(inverse_l0a, layout=make_layout((C, C))),
            v_beta_remainder,
            init=False,
        )
        if self.save_backward_inputs != 0:
            mem_copy(
                tile_slice(
                    self.saved_u, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (self.chunk_idx, 0)
                )[: self.valid_rows, None],
                self.l0c_w[: self.valid_rows, None],
                options=copy_options(l2_cache_ctl=4),
            )
        if self.save_backward_inputs != 0:
            mem_copy(
                tile_slice(
                    self.saved_w, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (self.chunk_idx, 0)
                )[: self.valid_rows, None],
                self.l0c_w_product[: self.valid_rows, None],
                options=copy_options(l2_cache_ctl=4),
            )

    def _load_resident_inv_to_l0a(self):
        l0a_write = self.l0a_recurrent
        mem_copy(
            reinterpret(
                l0a_write, offset=0, layout=make_layout((CHUNK_SIZE, CHUNK_SIZE))
            ),
            self.resident_inv_l1,
        )
        return l0a_write

    def _load_beta_to_l0b(self, beta_read):
        l0b_write = reinterpret(
            self.l0b_recurrent,
            data_format="nz",
            layout=make_layout((SUPPORTED_HEAD_DIM, D_HALF)),
        )
        mem_copy(l0b_write, permute(beta_read, (1, 0)))
        return l0b_write


class _State:
    """Retain FP32 state in UB and publish BF16 snapshots for Cube products."""

    def __init__(self, subblock_idx):
        self.subblock_idx = subblock_idx
        self.ub_state_fp32 = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((SUPPORTED_HEAD_DIM, D_HALF))
        )
        self.ub_state_delta = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((SUPPORTED_HEAD_DIM, D_HALF)),
            depth=1,
        )
        self.ub_state_nz = Buffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((SUPPORTED_HEAD_DIM, D_HALF)),
            data_format="nz",
            physical_layout=make_layout(
                (
                    ((SUPPORTED_HEAD_DIM, D_HALF)[1] + 15) // 16,
                    ((SUPPORTED_HEAD_DIM, D_HALF)[0] + 15) // 16,
                    16,
                    16,
                ),
                stride=(
                    ((SUPPORTED_HEAD_DIM, D_HALF)[0] + 15) // 16 * 256 + 16,
                    256,
                    16,
                    1,
                ),
            ),
        )
        self.state_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)),
            depth=1,
        )
        self.gamma_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((D_HALF,)))
        self.u_l1 = Buffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM))
        )

    def load(self, gm_state_rows):
        mem_copy(
            self.ub_state_fp32,
            tile_slice(
                gm_state_rows, (SUPPORTED_HEAD_DIM, D_HALF), (0, self.subblock_idx)
            ),
        )
        vec_sync_notify(PIPE.MTE2, PIPE.V, 7)
        vec_sync_wait(PIPE.MTE2, PIPE.V, 7)
        self.cast_state_snapshot_to_bf16_nz()

    @jit
    def cast_state_snapshot_to_bf16_nz(self):
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            low_mask, _ = update_mask(D_HALF, elem_bits=16)
            for row in dsl_range(0, SUPPORTED_HEAD_DIM, 1):
                state_bf16 = cast_and_pack_bf16(
                    vload(self.ub_state_fp32, row * D_HALF), lane_mask
                )
                vstore_strided(
                    self.ub_state_nz,
                    row * 16,
                    state_bf16,
                    low_mask,
                    block_stride=self.ub_state_nz.physical_stride[0] // 16,
                )

    def publish_state_snapshot(self):
        mem_copy(
            self.state_l1.next_write().partition(dim=1, part_id=self.subblock_idx),
            self.ub_state_nz,
        )

    @jit
    def update_recurrent_state(self):
        chunk_decay = self.gamma_ub
        delta = self.ub_state_delta.next_read()
        with vf(mode="simd"):
            lane_mask, _ = update_mask(D_HALF, elem_bits=32)
            gamma_c = vload(chunk_decay, 0)
            low_mask, _ = update_mask(D_HALF, elem_bits=16)
            for row in dsl_range(0, SUPPORTED_HEAD_DIM, 1):
                state_value = vload(self.ub_state_fp32, row * D_HALF)
                state_delta = vload(delta, row * D_HALF)
                state_next = vmadd(state_value, gamma_c, state_delta, mask=lane_mask)
                vstore(self.ub_state_fp32, row * D_HALF, state_next, lane_mask)
                state_bf16 = cast_and_pack_bf16(state_next, lane_mask)
                vstore_strided(
                    self.ub_state_nz,
                    row * 16,
                    state_bf16,
                    low_mask,
                    block_stride=self.ub_state_nz.physical_stride[0] // 16,
                )

    def store(self, gm_state_rows):
        vec_sync_notify(PIPE.V, PIPE.MTE3, 7)
        vec_sync_wait(PIPE.V, PIPE.MTE3, 7)
        mem_copy(
            tile_slice(
                gm_state_rows, (SUPPORTED_HEAD_DIM, D_HALF), (0, self.subblock_idx)
            ),
            self.ub_state_fp32,
        )


class _Caches:
    def __init__(self, subblock_idx):
        self.subblock_idx = subblock_idx
        self.ub_cast_bf16 = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((CHUNK_SIZE, D_HALF))
        )

        # Bound by the task dispatcher before the corresponding stage runs.
        self.save_backward_inputs = None
        self.save_intermediate_state = None
        self.sequence_length = None
        self.write_qk = None
        self.q_norm = None
        self.k_norm = None
        self.q_rstd = None
        self.k_rstd = None
        self.beta_activated = None
        self.qg = None
        self.kg = None
        self.g_cumsum = None
        self.v_new = None
        self.Aqk = None
        self.Akk = None
        self.h = None

    @jit
    def valid_rows(self, chunk_idx):
        rows = self.sequence_length - chunk_idx * CHUNK_SIZE
        if rows > CHUNK_SIZE:
            rows = CHUNK_SIZE
        return rows

    @jit
    def store_entry_state(self, recurrent_state, chunk_idx):
        if self.save_intermediate_state != 0:
            mem_copy(
                tile_slice(
                    self.h[0, 0, chunk_idx, None, None],
                    (SUPPORTED_HEAD_DIM, D_HALF),
                    (0, self.subblock_idx),
                ),
                recurrent_state.ub_state_fp32,
                options=copy_options(l2_cache_ctl=4),
            )


@jit
def recompute_tracks(
    q,
    k,
    v,
    g,
    beta,
    inverse,
    states,
    a_log,
    bias,
    metadata,
    w,
    u,
    v_new,
    qg,
    kg,
    gamma,
    initial_state,
    padded: dtypes.int64,
    fused_gate: dtypes.int64,
    has_bias: dtypes.int64,
    softplus: dtypes.int64,
    bound: dtypes.float32,
    fused_beta: dtypes.int64,
    has_initial_state: dtypes.int64,
    gm_owner_schedule,
):
    part = get_subblock_id()
    heads_per_key_head = v.shape[1] // q.shape[1]
    task_count = (metadata.shape[0] - grad.HEADER_WORDS) // grad.TASK_WORDS
    if (
        metadata[grad.HEADER_STATUS] != grad.STATUS_OK
        or metadata[grad.HEADER_MAGIC] != grad.MAGIC
        or metadata[grad.HEADER_ABI_VERSION] != grad.ABI_VERSION
    ):
        task_count = 0
    begin, end, step = grad._owner_loop_bounds(
        gm_owner_schedule, task_count // heads_per_key_head
    )
    for slot in range(begin, end, step):
        owner = grad._scheduled_owner(gm_owner_schedule, slot)
        for local_head in range(heads_per_key_head):
            # Restrict reconstruction resources to this owner. They must not
            # become loop-carried channel state in the later backward scan.
            prep = _Preprocess(part, g.dtype, beta.dtype)
            state = _State(part)
            cube = _Matmul()
            caches = _Caches(part)
            caches.save_intermediate_state = 1
            prep.cache_writer = caches
            prep.saved_qg_nd = caches.ub_cast_bf16
            prep.has_dt_bias = has_bias != 0
            prep.use_gate_in_kernel = fused_gate
            prep.has_lower_bound = softplus == 0
            prep.fused_beta = fused_beta
            record = (
                grad.HEADER_WORDS
                + (owner * heads_per_key_head + local_head) * grad.TASK_WORDS
            )
            sequence = metadata[record + grad.TASK_SEQUENCE]
            head = metadata[record + grad.TASK_VALUE_HEAD]
            length = metadata[record + grad.TASK_LENGTH]
            compact_start = metadata[record + grad.TASK_COMPACT_TOKEN_START]
            chunks = metadata[record + grad.TASK_CHUNKS]
            batch, origin, state_batch, state_base = (
                0,
                compact_start,
                0,
                compact_start // C + sequence,
            )
            if padded:
                batch, origin, state_batch, state_base = sequence, 0, sequence, 0
            stop = origin + length
            q_head = head // heads_per_key_head
            query = q[batch, q_head, origin:stop, None]
            key = k[batch, q_head, origin:stop, None]
            value = v[batch, head, origin:stop, None]
            gate = g[batch, head, origin:stop, None]
            activated_beta = beta[batch, head, origin:stop, None]
            prep.sequence_length = length
            caches.sequence_length = length
            caches.qg = qg[batch, head, origin:stop, None]
            caches.kg = kg[batch, head, origin:stop, None]
            caches.h = states[state_batch, head, state_base:, None, None]
            cube.saved_w = w[batch, head, origin:stop, None]
            cube.saved_u = u[batch, head, origin:stop, None]
            cube.saved_v_new = v_new[batch, head, origin:stop, None]
            if fused_gate:
                prep.load_head(a_log, bias, head)
            if chunks > 0:
                if has_initial_state:
                    state.load(initial_state[sequence, head, None, None])
                else:
                    _zero_fp32(state.ub_state_fp32, D * HALF)
                    state.cast_state_snapshot_to_bf16_nz()
                for chunk in range(chunks):
                    rows = min(C, length - chunk * C)
                    # Keep one publish/consume pair within each iteration.
                    # Carry the FP32/BF16 state values across chunks, rather
                    # than a conditionally advanced Channel cursor.
                    state.publish_state_snapshot()
                    caches.store_entry_state(state, chunk)
                    prep.prepare(query, key, value, gate, activated_beta, chunk, bound)
                    mem_copy(
                        tile_slice(
                            gamma[batch, head, origin:stop, None],
                            (C, HALF),
                            (chunk, part),
                        )[:rows, None],
                        prep.ub_g_cumsum[:rows, None],
                        options=copy_options(l2_cache_ctl=4),
                    )
                    # All four BF16 panels and their rounding boundaries come
                    # from the forward implementation, including kg and v_new.
                    prep.prepare_recurrent_operands(1.0, state.gamma_ub, chunk)
                    prep.acquire_recurrent_operands()
                    if padded:
                        inverse_rows = inverse[
                            batch, head, chunk * C : chunk * C + rows, None
                        ]
                        cube.load_inverse(
                            tile_slice(inverse_rows, (C, C), (0, 0)), rows
                        )
                    else:
                        cube.load_inverse(
                            inverse[state_base + chunk, head, None, None], rows
                        )
                    cube.chunk_idx = chunk
                    cube.valid_rows = rows
                    cube.compute_u_pre_and_w(prep, state)
                    cube.correct_values(state)
                    cube.state_delta(prep, state)
                    state.update_recurrent_state()
            if padded:
                _clear_padded_caches(
                    w, u, v_new, qg, kg, gamma, batch, head, length, v.shape[2], part
                )
    cube_sync_all()
    vec_sync_all()


class _StagedState:
    """Forward staged recurrence, sharded over independent value rows.

    Subtract the compensated W @ H product from the saved BF16 U in FP32,
    round v_new, then decay/add the FP32 state.
    The two AIVs split key columns for H and token rows for v_new.
    """

    def __init__(self, value_rows, operands, result, shared=None):
        self.value_rows = value_rows
        if shared is not None:
            self.state = reinterpret(
                shared.state, layout=make_layout((value_rows, HALF))
            )
            self.state_nz = reinterpret(
                shared.state_nz,
                data_format="nz",
                layout=make_layout((2 * value_rows, HALF)),
                physical_layout=make_layout(
                    (
                        ((2 * value_rows, HALF)[1] + 15) // 16,
                        ((2 * value_rows, HALF)[0] + 15) // 16,
                        16,
                        16,
                    ),
                    stride=(
                        ((2 * value_rows, HALF)[0] + 15) // 16 * 256 + 16,
                        256,
                        16,
                        1,
                    ),
                ),
            )
            self.gamma = shared.gamma
            self.u = reinterpret(shared.u, layout=make_layout((C // 2, value_rows)))
            self.new = reinterpret(shared.new, layout=make_layout((C // 2, value_rows)))
            self.new_nz = reinterpret(
                shared.new_nz,
                data_format="nz",
                layout=make_layout((C // 2, value_rows)),
                physical_layout=make_layout(
                    (
                        ((C // 2, value_rows)[1] + 15) // 16,
                        ((C // 2, value_rows)[0] + 15) // 16,
                        16,
                        16,
                    ),
                    stride=(
                        ((C // 2, value_rows)[0] + 15) // 16 * 256 + 16,
                        256,
                        16,
                        1,
                    ),
                ),
            )
            self.correction, self.delta = result, result
            self.state_l1, self.new_l1 = operands, operands
            self.w_l1, self.kg_l1 = shared.w_l1, shared.kg_l1
            self.w_l0a = shared.w_l0a
            self.new_l0a = reinterpret(
                shared.new_l0a, data_format="nz", layout=make_layout((value_rows, C))
            )
            self.state_l0b = reinterpret(
                shared.state_l0b, data_format="nz", layout=make_layout((value_rows, D))
            )
            self.kg_l0b = shared.kg_l0b
            self.correction_l0c = reinterpret(
                shared.correction_l0c,
                data_format="nz",
                layout=make_layout((C, value_rows)),
            )
            self.delta_l0c = reinterpret(
                shared.delta_l0c, data_format="nz", layout=make_layout((value_rows, D))
            )
            return
        self.state = Buffer(MemLoc.UB, dtypes.float32, make_layout((value_rows, HALF)))
        self.state_nz = Buffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((2 * value_rows, HALF)),
            data_format="nz",
            physical_layout=make_layout(
                (
                    ((2 * value_rows, HALF)[1] + 15) // 16,
                    ((2 * value_rows, HALF)[0] + 15) // 16,
                    16,
                    16,
                ),
                stride=(((2 * value_rows, HALF)[0] + 15) // 16 * 256 + 16, 256, 16, 1),
            ),
        )
        self.gamma = Buffer(MemLoc.UB, dtypes.float32, make_layout((HALF,)))
        self.u = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((C // 2, value_rows)))
        self.new = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((C // 2, value_rows)))
        self.new_nz = Buffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((C // 2, value_rows)),
            data_format="nz",
            physical_layout=make_layout(
                (
                    ((C // 2, value_rows)[1] + 15) // 16,
                    ((C // 2, value_rows)[0] + 15) // 16,
                    16,
                    16,
                ),
                stride=(((C // 2, value_rows)[0] + 15) // 16 * 256 + 16, 256, 16, 1),
            ),
        )
        self.correction = result
        self.delta = result
        self.state_l1 = operands
        self.new_l1 = operands
        self.w_l1 = Buffer(MemLoc.L1, dtypes.bfloat16, make_layout((C, D)))
        self.kg_l1 = Buffer(MemLoc.L1, dtypes.bfloat16, make_layout((C, D)))
        self.w_l0a = _cube_buffer(MemLoc.L0A, dtypes.bfloat16, make_layout((C, D)))
        self.new_l0a = _cube_buffer(
            MemLoc.L0A, dtypes.bfloat16, make_layout((value_rows, C))
        )
        self.state_l0b = _cube_buffer(
            MemLoc.L0B, dtypes.bfloat16, make_layout((value_rows, D))
        )
        self.kg_l0b = reinterpret(
            self.state_l0b, data_format="nz", layout=make_layout((D, C))
        )
        self.correction_l0c = _cube_buffer(
            MemLoc.L0C, dtypes.float32, make_layout((C, value_rows))
        )
        self.delta_l0c = _cube_buffer(
            MemLoc.L0C, dtypes.float32, make_layout((value_rows, D))
        )

    @jit
    def publish_state(self, part):
        with vf(mode="simd"):
            mask, _ = update_mask(HALF, elem_bits=32)
            low_mask, _ = update_mask(HALF, elem_bits=16)
            for row in range(self.value_rows):
                value = vload(self.state, row * HALF)
                high = vcast(value, dtypes.bfloat16, mask=mask)
                residual = vsub(
                    value, vcast(high, dtypes.float32, mask=mask), mask=mask
                )
                packed = cast_and_pack_bf16(value, mask)
                packed_low = cast_and_pack_bf16(residual, mask)
                vstore_strided(
                    self.state_nz,
                    row * 16,
                    packed,
                    low_mask,
                    block_stride=self.state_nz.physical_stride[0] // 16,
                )
                vstore_strided(
                    self.state_nz,
                    (self.value_rows + row) * 16,
                    packed_low,
                    low_mask,
                    block_stride=self.state_nz.physical_stride[0] // 16,
                )
        mem_copy(
            reinterpret(
                self.state_l1.next_write(),
                data_format="nz",
                layout=make_layout((2 * self.value_rows, D)),
            ).partition(dim=1, part_id=part),
            self.state_nz,
        )

    @jit
    def load_panel(self, destination, source, rows):
        if rows < C:
            cube_fill_l1_zero(destination, repeat=C * D // 16)
            cube_sync_pipe(PIPE.MTE2)
        source_tile = tile_slice(source, (C, D), (0, 0))
        mem_copy(destination[:rows, None], source_tile[:rows, None])

    @jit
    def correct_values(self, destination, rows, part):
        correction = reinterpret(
            self.correction.next_read(), layout=make_layout((C // 2, self.value_rows))
        )
        live_rows = min(C // 2, max(0, rows - part * (C // 2)))
        with vf(mode="simd"):
            for row in range(C // 2):
                for column in range(0, self.value_rows, HALF):
                    mask, _ = update_mask(self.value_rows - column, elem_bits=32)
                    offset = row * self.value_rows + column
                    value = vdups(0.0, dtypes.float32, mask=mask)
                    if row < live_rows:
                        raw = vcast(
                            vload_unpack(
                                self.u, offset, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                        value = vsub(raw, vload(correction, offset), mask=mask)
                    low = vcast(value, dtypes.bfloat16, mask=mask)
                    vstore_pack(
                        self.new, offset, low, mask, pack_mode=PackMode.B32_TO_B16
                    )
            vmem_bar("vst_vld")
        # Use the same ND->NZ layout as forward's staged v_new publication.
        with vf(mode="simd"):
            low_mask, _ = update_mask(self.value_rows, elem_bits=16)
            for row in range(C // 2):
                low = vload(self.new, row * self.value_rows)
                vstore_strided(
                    self.new_nz,
                    row * 16,
                    low,
                    low_mask,
                    block_stride=self.new_nz.physical_stride[0] // 16,
                )
        mem_copy(
            reinterpret(
                self.new_l1.next_write(),
                data_format="nz",
                layout=make_layout((C, self.value_rows)),
            ).partition(dim=0, part_id=part),
            self.new_nz,
        )
        if live_rows > 0:
            mem_copy(
                tile_slice(destination, (C // 2, self.value_rows), (part, 0))[
                    :live_rows, None
                ],
                self.new[:live_rows, None],
                options=copy_options(l2_cache_ctl=4),
            )

    @jit
    def update_state(self):
        delta = reinterpret(
            self.delta.next_read(), layout=make_layout((self.value_rows, HALF))
        )
        with vf(mode="simd"):
            mask, _ = update_mask(HALF, elem_bits=32)
            decay = vexp(vload(self.gamma, 0), mask=mask)
            for row in range(self.value_rows):
                offset = row * HALF
                scaled = vmul(vload(self.state, offset), decay, mask=mask)
                value = vadd(scaled, vload(delta, offset), mask=mask)
                vstore(self.state, offset, value, mask)
            vmem_bar("vst_vld")


@jit
def _recompute_staged_states(
    w,
    u,
    kg,
    gamma,
    states,
    v_new,
    initial_state,
    metadata,
    padded,
    has_initial_state,
    task_count,
    value_rows: Constexpr[int],
    operands,
    result,
    resources,
):
    part = get_subblock_id()
    begin, end, step = grad._metadata._staged_reverse_loop_bounds(
        task_count, (D // value_rows)
    )
    for owner in range(begin, end, step):
        recurrence = _StagedState(value_rows, operands, result, resources)
        task, value_part = grad._metadata._staged_reverse_owner(
            owner, (D // value_rows)
        )
        record = grad.HEADER_WORDS + task * grad.TASK_WORDS
        sequence = metadata[record + grad.TASK_SEQUENCE]
        head = metadata[record + grad.TASK_VALUE_HEAD]
        length = metadata[record + grad.TASK_LENGTH]
        chunks = metadata[record + grad.TASK_CHUNKS]
        origin = metadata[record + grad.TASK_COMPACT_TOKEN_START]
        batch, state_batch, state_base = 0, 0, origin // C + sequence
        if padded:
            batch, origin, state_batch, state_base = sequence, 0, sequence, 0
        if chunks > 0:
            if has_initial_state:
                mem_copy(
                    recurrence.state,
                    tile_slice(
                        initial_state[sequence, head, None, None],
                        (recurrence.value_rows, HALF),
                        (value_part, part),
                    ),
                )
            else:
                _zero_fp32(recurrence.state, recurrence.value_rows * HALF)
            for chunk in range(chunks):
                start = origin + chunk * C
                rows = min(C, length - chunk * C)
                state_tile = tile_slice(
                    states[state_batch, head, state_base + chunk, None, None],
                    (recurrence.value_rows, HALF),
                    (value_part, part),
                )
                mem_copy(
                    state_tile, recurrence.state, options=copy_options(l2_cache_ctl=4)
                )
                recurrence.publish_state(part)
                recurrence.load_panel(
                    recurrence.w_l1, w[batch, head, start : start + rows, None], rows
                )
                recurrence.load_panel(
                    recurrence.kg_l1, kg[batch, head, start : start + rows, None], rows
                )
                mem_copy(recurrence.w_l0a, recurrence.w_l1)
                state_panels = reinterpret(
                    recurrence.state_l1.next_read(),
                    data_format="nz",
                    layout=make_layout((2 * value_rows, D)),
                )
                mem_copy(
                    recurrence.state_l0b,
                    tile_slice(state_panels, (value_rows, D), (0, 0)),
                )
                matmul(
                    recurrence.correction_l0c,
                    recurrence.w_l0a,
                    recurrence.state_l0b,
                    init=True,
                )
                mem_copy(
                    recurrence.state_l0b,
                    tile_slice(state_panels, (value_rows, D), (1, 0)),
                )
                matmul(
                    recurrence.correction_l0c,
                    recurrence.w_l0a,
                    recurrence.state_l0b,
                    init=False,
                )
                mem_copy(
                    reinterpret(
                        recurrence.correction.next_write(),
                        layout=make_layout((C // 2, value_rows)),
                    ),
                    reinterpret(
                        recurrence.correction_l0c, layout=make_layout((C, value_rows))
                    ).partition(dim=0),
                )
                load_sequence_tile(
                    recurrence.u,
                    u[batch, head, origin : origin + length, None],
                    chunk * 2 + part,
                    value_part,
                    length,
                    True,
                )
                mem_copy(
                    recurrence.gamma,
                    tile_slice(
                        gamma[batch, head, start + rows - 1 : start + rows, None],
                        (1, HALF),
                        (0, part),
                    ).view(HALF),
                )
                destination = tile_slice(
                    v_new[batch, head, start : start + rows, None],
                    (C, recurrence.value_rows),
                    (0, value_part),
                )
                recurrence.correct_values(destination, rows, part)
                mem_copy(
                    recurrence.new_l0a,
                    permute(
                        reinterpret(
                            recurrence.new_l1.next_read(),
                            data_format="nz",
                            layout=make_layout((C, value_rows)),
                        ),
                        (1, 0),
                    ),
                )
                mem_copy(recurrence.kg_l0b, permute(recurrence.kg_l1, (1, 0)))
                matmul(
                    recurrence.delta_l0c,
                    recurrence.new_l0a,
                    recurrence.kg_l0b,
                    init=True,
                )
                mem_copy(
                    reinterpret(
                        recurrence.delta.next_write(),
                        layout=make_layout((value_rows, HALF)),
                    ),
                    reinterpret(
                        recurrence.delta_l0c, layout=make_layout((value_rows, D))
                    ).partition(dim=1),
                )
                recurrence.update_state()
    cube_sync_all()
    vec_sync_all()


@jit
def recompute_staged_tracks(
    q,
    k,
    v,
    g,
    beta,
    inverse,
    states,
    a_log,
    bias,
    metadata,
    w,
    u,
    v_new,
    qg,
    kg,
    gamma,
    initial_state,
    padded: dtypes.int64,
    fused_gate: dtypes.int64,
    has_bias: dtypes.int64,
    softplus: dtypes.int64,
    bound: dtypes.float32,
    fused_beta: dtypes.int64,
    has_initial_state: dtypes.int64,
    gm_owner_schedule,
):
    """Fwd's independent chunk products followed by value-sharded recurrence.

    All chunk caches already have full-size GM storage in grad, so stage 1
    can publish all chunks before stage 2; no fwd workspace rounds or second
    metadata ABI are needed. Scheduling reuses grad's staged metadata helpers.
    """
    part, block, cores = get_subblock_id(), get_block_idx(), get_block_num()
    heads_per_key_head = v.shape[1] // q.shape[1]
    task_count = (metadata.shape[0] - grad.HEADER_WORDS) // grad.TASK_WORDS
    if (
        metadata[grad.HEADER_STATUS] != grad.STATUS_OK
        or metadata[grad.HEADER_MAGIC] != grad.MAGIC
        or metadata[grad.HEADER_ABI_VERSION] != grad.ABI_VERSION
    ):
        task_count = 0
    # Keep reconstruction resources in a uniform metadata-valid region.
    if task_count > 0:
        # Chunk operands live only until the first phase handoff. Recurrence
        # allocates fresh channels after these local buffers have been retired.
        operands = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((RECOMPUTE_OPERAND_PLANES * C, D)),
            depth=1,
        )
        prep = _Preprocess(part, g.dtype, beta.dtype, operands)
        cube = _Matmul()
        caches = _Caches(part)
        decay = Buffer(MemLoc.UB, dtypes.float32, make_layout((HALF,)))
        for task in range(task_count):
            record = grad.HEADER_WORDS + task * grad.TASK_WORDS
            sequence = metadata[record + grad.TASK_SEQUENCE]
            head = metadata[record + grad.TASK_VALUE_HEAD]
            length = metadata[record + grad.TASK_LENGTH]
            chunks = metadata[record + grad.TASK_CHUNKS]
            origin = metadata[record + grad.TASK_COMPACT_TOKEN_START]
            batch, state_base = 0, origin // C + sequence
            if padded:
                batch, origin = sequence, 0
            stop = origin + length
            first = grad._metadata._staged_chunk_start(
                metadata, record, heads_per_key_head, block, cores
            )
            prep.sequence_length = length
            prep.cache_writer = caches
            prep.saved_qg_nd = caches.ub_cast_bf16
            prep.has_dt_bias = has_bias != 0
            prep.use_gate_in_kernel = fused_gate
            prep.has_lower_bound = softplus == 0
            prep.fused_beta = fused_beta
            caches.sequence_length = length
            caches.qg = qg[batch, head, origin:stop, None]
            caches.kg = kg[batch, head, origin:stop, None]
            cube.saved_w = w[batch, head, origin:stop, None]
            cube.saved_u = u[batch, head, origin:stop, None]
            # Unowned/empty tasks must not replace a live head's parameters.
            if fused_gate and first < chunks:
                prep.load_head(a_log, bias, head)
            for chunk in range(first, chunks, cores):
                rows = min(C, length - chunk * C)
                prep.prepare(
                    q[batch, head // heads_per_key_head, origin:stop, None],
                    k[batch, head // heads_per_key_head, origin:stop, None],
                    v[batch, head, origin:stop, None],
                    g[batch, head, origin:stop, None],
                    beta[batch, head, origin:stop, None],
                    chunk,
                    bound,
                )
                mem_copy(
                    tile_slice(
                        gamma[batch, head, origin:stop, None], (C, HALF), (chunk, part)
                    )[:rows, None],
                    prep.ub_g_cumsum[:rows, None],
                    options=copy_options(l2_cache_ctl=4),
                )
                prep.prepare_recurrent_operands(1.0, decay, chunk)
                prep.acquire_recurrent_operands()
                if padded:
                    cube.load_inverse(
                        inverse[batch, head, chunk * C : chunk * C + rows, None], rows
                    )
                else:
                    cube.load_inverse(
                        inverse[state_base + chunk, head, None, None], rows
                    )
                cube.chunk_idx, cube.valid_rows = chunk, rows
                cube.compute_u_pre_and_w(prep, None)
        if padded:
            for task in range(block, task_count, cores):
                record = grad.HEADER_WORDS + task * grad.TASK_WORDS
                _clear_padded_caches(
                    w,
                    u,
                    v_new,
                    qg,
                    kg,
                    gamma,
                    metadata[record + grad.TASK_SEQUENCE],
                    metadata[record + grad.TASK_VALUE_HEAD],
                    metadata[record + grad.TASK_LENGTH],
                    v.shape[2],
                    part,
                    caches.ub_cast_bf16,
                    prep.ub_g_cumsum,
                )
        # Stage one publishes only GM caches. Retire its channels and local
        # buffers before allocating the independent FP32 recurrence stage.
        grad._sync_cube_vector_handoff(1, grid=True)
        operands = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((2 * D, D)), depth=1
        )
        result = RingBuffer(MemLoc.UB, dtypes.float32, make_layout((D, D)), depth=1)
        resources = _StagedState(D, operands, result)
        if grad._metadata._staged_reverse_is_unsplit(task_count, cores):
            _recompute_staged_states(
                w,
                u,
                kg,
                gamma,
                states,
                v_new,
                initial_state,
                metadata,
                padded,
                has_initial_state,
                task_count,
                D,
                operands,
                result,
                resources,
            )
        else:
            _recompute_staged_states(
                w,
                u,
                kg,
                gamma,
                states,
                v_new,
                initial_state,
                metadata,
                padded,
                has_initial_state,
                task_count,
                grad._metadata.STAGED_SPLIT_VALUE_ROWS,
                operands,
                result,
                resources,
            )
