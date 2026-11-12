# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Persistent owner scheduling and its full-chunk/state pipelines."""

from __future__ import annotations

from cannbotdsl import (
    RoundingMode,
    from_tensor,
    Buffer,
    Constexpr,
    MemLoc,
    PIPE,
    const_expr,
    dtypes,
)
from cannbotdsl import make_layout
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.control_flow import range as drange
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import (
    create_mask,
    update_mask,
    vadd,
    vcast,
    vdups,
    vexp,
    vload,
    vmadd,
    vmem_bar,
    vmul,
    vmuls,
    vstore,
    vstore_strided,
    vpack,
    vreinterpret,
    vreinterpret_lanes,
    vsub,
)
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_subblock_dim,
    get_subblock_id,
)
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import (
    onchip_rewind,
    cube_sync_block_wait,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.tensor import tile_slice

# Shared arithmetic and metadata constants live in the public entry module.
if __package__:
    from .flash_kda_grad import (
        _cube_ring,
        ABI_VERSION,
        CHUNK_SIZE,
        CUBE_TILE_ROWS,
        D_HALF,
        HEADER_ABI_VERSION,
        HEADER_MAGIC,
        HEADER_STATUS,
        HEADER_WORDS,
        KEY_DIM,
        MAGIC,
        NZ_BLOCK_WIDTH,
        STATUS_OK,
        SUPPORTED_HEAD_DIM,
        StageOneProcess,
        _StageOneScratchScope,
        StageTwoMatmulResources,
        StageTwoOutputVector,
        StageTwoPipelineMatmul,
        StageTwoVector,
        TASK_CHUNKS,
        TASK_COMPACT_TOKEN_START,
        TASK_LENGTH,
        TASK_SEQUENCE,
        TASK_VALUE_HEAD,
        TASK_WORDS,
        VALUE_DIM,
        _finish_masked_track_chunk,
        _finish_track_chunk,
        _full_token_tile,
        _initialize_backward_gradients,
        _owner_loop_bounds,
        _prepare_track_chunk,
        _run_single_chunk_gate_sensitivity,
        _scheduled_owner,
        _sync_cube_vector_handoff,
        _tnd_chunk_index,
        _token_rows_tile,
    )
else:
    from flash_kda_grad import (
        _cube_ring,
        ABI_VERSION,
        CHUNK_SIZE,
        CUBE_TILE_ROWS,
        D_HALF,
        HEADER_ABI_VERSION,
        HEADER_MAGIC,
        HEADER_STATUS,
        HEADER_WORDS,
        KEY_DIM,
        MAGIC,
        NZ_BLOCK_WIDTH,
        STATUS_OK,
        SUPPORTED_HEAD_DIM,
        StageOneProcess,
        _StageOneScratchScope,
        StageTwoMatmulResources,
        StageTwoOutputVector,
        StageTwoPipelineMatmul,
        StageTwoVector,
        TASK_CHUNKS,
        TASK_COMPACT_TOKEN_START,
        TASK_LENGTH,
        TASK_SEQUENCE,
        TASK_VALUE_HEAD,
        TASK_WORDS,
        VALUE_DIM,
        _finish_masked_track_chunk,
        _finish_track_chunk,
        _full_token_tile,
        _initialize_backward_gradients,
        _owner_loop_bounds,
        _prepare_track_chunk,
        _run_single_chunk_gate_sensitivity,
        _scheduled_owner,
        _sync_cube_vector_handoff,
        _tnd_chunk_index,
        _token_rows_tile,
    )


class StageTwoTrack:
    """Full chunk Stage 2: keep operands in L1 and state columns in AIV UB."""

    def __init__(self, scale, dqg_cache=None, dr_cache=None, dw_cache=None):
        self.dqg_cache = dqg_cache
        self.dr_cache = dr_cache
        self.dw_cache = dw_cache
        self.scale = scale
        fp32_storage = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((2 * VALUE_DIM + 2, D_HALF)), depth=1
        ).next_write()
        self.state = fp32_storage[:VALUE_DIM, None]
        self.saved = fp32_storage[VALUE_DIM : 2 * VALUE_DIM, None]
        self.gate = fp32_storage[2 * VALUE_DIM : 2 * VALUE_DIM + 1, None]
        self.gate_product = fp32_storage[2 * VALUE_DIM + 1 : 2 * VALUE_DIM + 2, None]
        self.state_nz = Buffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((2 * VALUE_DIM, D_HALF)),
            data_format="nz",
            physical_layout=make_layout(
                (
                    ((2 * VALUE_DIM, D_HALF)[1] + 15) // 16,
                    ((2 * VALUE_DIM, D_HALF)[0] + 15) // 16,
                    16,
                    16,
                ),
                stride=(
                    ((2 * VALUE_DIM, D_HALF)[0] + 15) // 16 * 256 + NZ_BLOCK_WIDTH,
                    256,
                    16,
                    1,
                ),
            ),
        )
        self.saved_nz = Buffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((VALUE_DIM, D_HALF)),
            data_format="nz",
            physical_layout=make_layout(
                (
                    ((VALUE_DIM, D_HALF)[1] + 15) // 16,
                    ((VALUE_DIM, D_HALF)[0] + 15) // 16,
                    16,
                    16,
                ),
                stride=(
                    ((VALUE_DIM, D_HALF)[0] + 15) // 16 * 256 + NZ_BLOCK_WIDTH,
                    256,
                    16,
                    1,
                ),
            ),
        )
        self.state_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((2 * VALUE_DIM, KEY_DIM)), depth=1
        )
        self.saved_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((VALUE_DIM, KEY_DIM)), depth=1
        )
        result_width = D_HALF if dqg_cache is not None else KEY_DIM
        result_depth = 2 if dqg_cache is not None else 1
        self.result = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((VALUE_DIM, result_width)),
            depth=result_depth,
        )
        self.tokens = tuple(
            RingBuffer(
                MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, KEY_DIM)), depth=1
            )
            for _ in range(5)
        )
        self.aqk = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        )
        self.dr = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, VALUE_DIM)), depth=1
        )
        self.token_a = _cube_ring(
            MemLoc.L0A, dtypes.bfloat16, make_layout((CHUNK_SIZE, KEY_DIM)), depth=1
        )
        self.outer_a = _cube_ring(
            MemLoc.L0A, dtypes.bfloat16, make_layout((VALUE_DIM, CHUNK_SIZE)), depth=1
        )
        self.matrix_a = _cube_ring(
            MemLoc.L0A, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        )
        self.token_b = _cube_ring(
            MemLoc.L0B, dtypes.bfloat16, make_layout((CHUNK_SIZE, KEY_DIM)), depth=1
        )
        self.state_b = _cube_ring(
            MemLoc.L0B, dtypes.bfloat16, make_layout((VALUE_DIM, KEY_DIM)), depth=1
        )
        self.token_c = _cube_ring(
            MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, KEY_DIM)), depth=1
        )
        self.state_c = _cube_ring(
            MemLoc.L0C, dtypes.float32, make_layout((VALUE_DIM, KEY_DIM)), depth=1
        )
        self.matrix_c = _cube_ring(
            MemLoc.L0C, dtypes.float32, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        )
        self.output_vector = StageTwoOutputVector()

    @jit
    def _publish_state(self):
        # The AIVs own disjoint key columns. Preserve the exact BF16 high/low
        # split used by the dependent dKg product, and share high with dR.
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            packed, _ = update_mask(D_HALF, elem_bits=16)
            nz_stride = self.state_nz.physical_layout.stride[0]
            saved_stride = self.saved_nz.physical_layout.stride[0]
            total = vdups(0.0, dtypes.float32, mask=full)
            for row in range(VALUE_DIM):
                value = vload(self.state, row * D_HALF)
                saved_value = vload(self.saved, row * D_HALF)
                total = vmadd(value, saved_value, total, mask=full)
                saved_high = vcast(
                    saved_value, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN
                )
                saved_packed = vreinterpret(
                    vpack(vreinterpret_lanes(saved_high, dtypes.uint32), dtypes.uint16),
                    dtypes.bfloat16,
                )
                vstore_strided(
                    self.saved_nz,
                    row * NZ_BLOCK_WIDTH,
                    saved_packed,
                    packed,
                    block_stride=saved_stride // NZ_BLOCK_WIDTH,
                    repeat_stride=0,
                )
                high = vcast(value, dtypes.bfloat16, mask=full)
                low = vcast(
                    vsub(value, vcast(high, dtypes.float32, mask=full), mask=full),
                    dtypes.bfloat16,
                    mask=full,
                )
                high_packed = vreinterpret(
                    vpack(vreinterpret_lanes(high, dtypes.uint32), dtypes.uint16),
                    dtypes.bfloat16,
                )
                low_packed = vreinterpret(
                    vpack(vreinterpret_lanes(low, dtypes.uint32), dtypes.uint16),
                    dtypes.bfloat16,
                )
                vstore_strided(
                    self.state_nz,
                    row * NZ_BLOCK_WIDTH,
                    high_packed,
                    packed,
                    block_stride=nz_stride // NZ_BLOCK_WIDTH,
                    repeat_stride=0,
                )
                vstore_strided(
                    self.state_nz,
                    (row + VALUE_DIM) * NZ_BLOCK_WIDTH,
                    low_packed,
                    packed,
                    block_stride=nz_stride // NZ_BLOCK_WIDTH,
                    repeat_stride=0,
                )
            vstore(self.gate_product, 0, total, full)
            vmem_bar("vst_vld")
        mem_copy(
            self.state_l1.next_write()[
                : (2 * VALUE_DIM, KEY_DIM)[0], : (2 * VALUE_DIM, KEY_DIM)[1]
            ].partition(dim=1, part_id=get_subblock_id()),
            self.state_nz,
        )

    @jit
    def _prepare_state(self, qterm):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            decay = vexp(vload(self.gate, 0), mask=full)
            for row in range(VALUE_DIM):
                offset = row * D_HALF
                state = vmul(vload(self.state, offset), decay, mask=full)
                output = vmuls(vload(qterm, offset), self.scale, mask=full)
                vstore(self.state, offset, vadd(state, output, mask=full), full)
            vmem_bar("vst_vld")

    @jit
    def _finish_state(self, contribution):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for row in range(VALUE_DIM):
                offset = row * D_HALF
                vstore(
                    self.state,
                    offset,
                    vsub(
                        vload(self.state, offset),
                        vload(contribution, offset),
                        mask=full,
                    ),
                    full,
                )
            vmem_bar("vst_vld")

    @jit
    def _scale_result(self, result, scale: dtypes.float32):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for offset in range(0, (CHUNK_SIZE // 2) * KEY_DIM, D_HALF):
                vstore(
                    result, offset, vmuls(vload(result, offset), scale, mask=full), full
                )
            vmem_bar("vst_vld")

    @jit
    def _outer(self, left, right):
        mem_copy(self.outer_a.next_write(), left.permute((1, 0)))
        mem_copy(
            from_tensor(
                self.token_b.next_write(),
                data_format="nz",
                layout=make_layout((KEY_DIM, CHUNK_SIZE)),
            ),
            right.permute((1, 0)),
        )
        matmul(
            self.state_c.next_write(),
            self.outer_a.next_read(),
            from_tensor(
                self.token_b.next_read(),
                data_format="nz",
                layout=make_layout((KEY_DIM, CHUNK_SIZE)),
            ),
            init=True,
        )
        mem_copy(
            from_tensor(
                self.result.next_write(),
                data_format="nd",
                layout=make_layout((VALUE_DIM, D_HALF)),
            )
            if self.dqg_cache is not None
            else from_tensor(
                self.result.next_write(), layout=make_layout((VALUE_DIM, D_HALF))
            ),
            from_tensor(
                self.state_c.next_read(), layout=make_layout((VALUE_DIM, KEY_DIM))
            ).partition(dim=1),
        )

    @jit
    def _store_token_result(self, result, destination, resident=None):
        mem_copy(
            from_tensor(
                self.result.next_write(),
                data_format="nd",
                layout=make_layout((CHUNK_SIZE // 2, KEY_DIM)),
            )
            if self.dqg_cache is not None
            else from_tensor(
                self.result.next_write(), layout=make_layout((CHUNK_SIZE // 2, KEY_DIM))
            ),
            from_tensor(result, layout=make_layout((CHUNK_SIZE, KEY_DIM))).partition(
                dim=0
            ),
        )
        local_result = (
            from_tensor(
                self.result.next_read(),
                data_format="nd",
                layout=make_layout((CHUNK_SIZE // 2, KEY_DIM)),
            )
            if self.dqg_cache is not None
            else from_tensor(
                self.result.next_read(), layout=make_layout((CHUNK_SIZE // 2, KEY_DIM))
            )
        )
        if const_expr(resident is None):
            mem_copy(
                destination[
                    : (CHUNK_SIZE, KEY_DIM)[0], : (CHUNK_SIZE, KEY_DIM)[1]
                ].partition(dim=0, part_id=get_subblock_id()),
                local_result,
            )
        else:
            mem_copy(resident, local_result)

    @jit
    def _token_state(
        self, token, state, destination, scale: dtypes.float32, resident=None
    ):
        mem_copy(self.token_a.next_write(), token)
        mem_copy(self.state_b.next_write(), state.permute((1, 0)))
        matmul(
            self.token_c.next_write(),
            self.token_a.next_read(),
            self.state_b.next_read(),
            init=True,
        )
        mem_copy(
            from_tensor(
                self.result.next_write(),
                data_format="nd",
                layout=make_layout((CHUNK_SIZE // 2, KEY_DIM)),
            )
            if self.dqg_cache is not None
            else from_tensor(
                self.result.next_write(), layout=make_layout((CHUNK_SIZE // 2, KEY_DIM))
            ),
            from_tensor(
                self.token_c.next_read(), layout=make_layout((CHUNK_SIZE, KEY_DIM))
            ).partition(dim=0),
        )
        result = (
            from_tensor(
                self.result.next_read(),
                data_format="nd",
                layout=make_layout((CHUNK_SIZE // 2, KEY_DIM)),
            )
            if self.dqg_cache is not None
            else from_tensor(
                self.result.next_read(), layout=make_layout((CHUNK_SIZE // 2, KEY_DIM))
            )
        )
        self._scale_result(result, scale)
        if const_expr(resident is None):
            mem_copy(
                destination[
                    : (CHUNK_SIZE, KEY_DIM)[0], : (CHUNK_SIZE, KEY_DIM)[1]
                ].partition(dim=0, part_id=get_subblock_id()),
                result,
            )
        else:
            mem_copy(resident, result)

    @jit
    def run_step(
        self,
        stage_two_gm_Aqk,
        stage_two_gm_qg,
        stage_two_gm_kg,
        stage_two_gm_w,
        stage_two_gm_v_new,
        stage_two_gm_g_cumsum,
        stage_two_gm_chunk_states,
        stage_two_gm_output_gradient,
        stage_two_gm_final_state_gradient,
        stage_two_gm_task_metadata,
        stage_two_gm_d_v_new,
        stage_two_gm_d_Aqk,
        stage_two_gm_d_qg,
        stage_two_gm_d_kg,
        stage_two_gm_d_w,
        stage_two_gm_d_gate_last,
        stage_two_gm_dh0,
        padded_layout: dtypes.int64,
        task: dtypes.int64,
        chunk: dtypes.int64,
        scratch_qg,
        scratch_kg,
        scratch_w,
        scratch_v,
        scratch_do,
        scratch_aqk,
        tail_safe: Constexpr[bool] = False,
    ):
        record = HEADER_WORDS + task * TASK_WORDS
        valid_rows = min(
            CHUNK_SIZE,
            stage_two_gm_task_metadata[record + TASK_LENGTH] - chunk * CHUNK_SIZE,
        )
        sequence = stage_two_gm_task_metadata[record + TASK_SEQUENCE]
        head = stage_two_gm_task_metadata[record + TASK_VALUE_HEAD]
        compact_start = stage_two_gm_task_metadata[record + TASK_COMPACT_TOKEN_START]
        chunks = stage_two_gm_task_metadata[record + TASK_CHUNKS]
        batch = sequence
        row = chunk * CHUNK_SIZE
        if not padded_layout:
            batch = 0
            row = row + compact_start
        matrix_batch = dtypes.int64(batch)
        matrix_row = row
        if not padded_layout:
            matrix_batch = _tnd_chunk_index(compact_start, sequence, chunk)
            matrix_row = 0
        saved_batch = dtypes.int64(batch)
        saved_chunk = dtypes.int64(chunk)
        if not padded_layout:
            saved_batch = 0
            saved_chunk = _tnd_chunk_index(compact_start, sequence, chunk)
        destination_row = get_block_idx() * CHUNK_SIZE
        # Full chunks are the only callers; every view preserves original GM strides.
        gm_qg = _full_token_tile(stage_two_gm_qg[batch, head, None, None], row, KEY_DIM)
        gm_kg = _full_token_tile(stage_two_gm_kg[batch, head, None, None], row, KEY_DIM)
        gm_w = _full_token_tile(stage_two_gm_w[batch, head, None, None], row, KEY_DIM)
        gm_v = _full_token_tile(
            stage_two_gm_v_new[batch, head, None, None], row, VALUE_DIM
        )
        gm_do = _full_token_tile(
            stage_two_gm_output_gradient[batch, head, None, None], row, VALUE_DIM
        )
        gm_aqk = _full_token_tile(
            stage_two_gm_Aqk[matrix_batch, head, None, None], matrix_row, CHUNK_SIZE
        )
        gm_dr = _full_token_tile(
            stage_two_gm_d_v_new[0, 0, None, None], destination_row, VALUE_DIM
        )
        gm_dqg = _full_token_tile(
            stage_two_gm_d_qg[0, 0, None, None], destination_row, KEY_DIM
        )
        gm_dkg = _full_token_tile(
            stage_two_gm_d_kg[0, 0, None, None], destination_row, KEY_DIM
        )
        gm_dw = _full_token_tile(
            stage_two_gm_d_w[0, 0, None, None], destination_row, KEY_DIM
        )
        if chunk == chunks - 1:
            mem_copy(
                self.state,
                tile_slice(
                    stage_two_gm_final_state_gradient[sequence, head, None, None],
                    (VALUE_DIM, D_HALF),
                    (0, get_subblock_id()),
                ),
            )
        else:
            mem_copy(
                self.state,
                tile_slice(
                    stage_two_gm_dh0[sequence, head, None, None],
                    (VALUE_DIM, D_HALF),
                    (0, get_subblock_id()),
                ),
            )
        mem_copy(
            self.saved,
            tile_slice(
                stage_two_gm_chunk_states[saved_batch, head, saved_chunk, None, None],
                (VALUE_DIM, D_HALF),
                (0, get_subblock_id()),
            ),
        )
        mem_copy(
            self.gate,
            tile_slice(
                stage_two_gm_g_cumsum[batch, head, None, None],
                (1, D_HALF),
                (row + valid_rows - 1, get_subblock_id()),
            ),
        )
        if const_expr(tail_safe):
            if valid_rows < CHUNK_SIZE:
                vector = StageTwoVector(double_window=True)
                half = get_subblock_id() * (CHUNK_SIZE // 2)
                live = max(0, min(CHUNK_SIZE // 2, valid_rows - half))
                slot_row = get_block_idx() * CHUNK_SIZE + half
                vector.copy_window(
                    stage_two_gm_qg[batch, head, None, None],
                    row + half,
                    scratch_qg,
                    slot_row,
                    live,
                    KEY_DIM,
                    False,
                )
                vector.copy_window(
                    stage_two_gm_kg[batch, head, None, None],
                    row + half,
                    scratch_kg,
                    slot_row,
                    live,
                    KEY_DIM,
                    False,
                )
                vector.copy_window(
                    stage_two_gm_w[batch, head, None, None],
                    row + half,
                    scratch_w,
                    slot_row,
                    live,
                    KEY_DIM,
                    False,
                )
                vector.copy_window(
                    stage_two_gm_v_new[batch, head, None, None],
                    row + half,
                    scratch_v,
                    slot_row,
                    live,
                    VALUE_DIM,
                    False,
                )
                vector.copy_window(
                    stage_two_gm_output_gradient[batch, head, None, None],
                    row + half,
                    scratch_do,
                    slot_row,
                    live,
                    VALUE_DIM,
                    False,
                )
                vector.copy_window(
                    stage_two_gm_Aqk[matrix_batch, head, None, None],
                    matrix_row + half,
                    scratch_aqk,
                    slot_row,
                    live,
                    CHUNK_SIZE,
                    False,
                    valid_rows,
                )
                _sync_cube_vector_handoff(0)
                mem_copy(
                    self.tokens[0].next_write(),
                    tile_slice(scratch_qg, (CHUNK_SIZE, KEY_DIM), (get_block_idx(), 0)),
                )
                mem_copy(
                    self.tokens[1].next_write(),
                    tile_slice(scratch_kg, (CHUNK_SIZE, KEY_DIM), (get_block_idx(), 0)),
                )
                mem_copy(
                    self.tokens[2].next_write(),
                    tile_slice(scratch_w, (CHUNK_SIZE, KEY_DIM), (get_block_idx(), 0)),
                )
                mem_copy(
                    self.tokens[3].next_write(),
                    tile_slice(
                        scratch_v, (CHUNK_SIZE, VALUE_DIM), (get_block_idx(), 0)
                    ),
                )
                mem_copy(
                    self.tokens[4].next_write(),
                    tile_slice(
                        scratch_do, (CHUNK_SIZE, VALUE_DIM), (get_block_idx(), 0)
                    ),
                )
                mem_copy(
                    self.aqk.next_write(),
                    tile_slice(
                        scratch_aqk, (CHUNK_SIZE, CHUNK_SIZE), (get_block_idx(), 0)
                    ),
                )
            else:
                # Load each immutable token tile once for all its Cube consumers.
                mem_copy(self.tokens[0].next_write(), gm_qg)
                mem_copy(self.tokens[1].next_write(), gm_kg)
                mem_copy(self.tokens[2].next_write(), gm_w)
                mem_copy(self.tokens[3].next_write(), gm_v)
                mem_copy(self.tokens[4].next_write(), gm_do)
                mem_copy(self.aqk.next_write(), gm_aqk)
        else:
            # Load each immutable token tile once for all its Cube consumers.
            mem_copy(self.tokens[0].next_write(), gm_qg)
            mem_copy(self.tokens[1].next_write(), gm_kg)
            mem_copy(self.tokens[2].next_write(), gm_w)
            mem_copy(self.tokens[3].next_write(), gm_v)
            mem_copy(self.tokens[4].next_write(), gm_do)
            mem_copy(self.aqk.next_write(), gm_aqk)
        qg, kg, w, value, output = (
            self.tokens[0].next_read(),
            self.tokens[1].next_read(),
            self.tokens[2].next_read(),
            self.tokens[3].next_read(),
            self.tokens[4].next_read(),
        )
        aqk = self.aqk.next_read()
        self._publish_state()
        mem_copy(
            tile_slice(
                stage_two_gm_d_gate_last[0, 0, None, None],
                (1, D_HALF),
                (get_block_idx(), get_subblock_id()),
            ),
            self.gate_product,
        )
        mem_copy(
            self.saved_l1.next_write()[
                : (VALUE_DIM, KEY_DIM)[0], : (VALUE_DIM, KEY_DIM)[1]
            ].partition(dim=1, part_id=get_subblock_id()),
            self.saved_nz,
        )
        state = self.state_l1.next_read()
        high = tile_slice(state, (VALUE_DIM, KEY_DIM), (0, 0))
        low = tile_slice(state, (VALUE_DIM, KEY_DIM), (1, 0))
        saved = self.saved_l1.next_read()
        # dAqk has no state dependency. Masking overlaps the following products.
        mem_copy(self.token_a.next_write(), output)
        mem_copy(self.token_b.next_write(), value)
        matmul(
            self.matrix_c.next_write(),
            self.token_a.next_read(),
            self.token_b.next_read(),
            init=True,
        )
        mem_copy(
            self.output_vector.matrix_in.next_write(),
            self.matrix_c.next_read().partition(dim=0),
        )
        self.output_vector.mask_matrix(
            tile_slice(
                stage_two_gm_d_Aqk, (CHUNK_SIZE, CHUNK_SIZE), (get_block_idx(), 0)
            ),
            valid_rows,
            get_subblock_id(),
        )
        # high/low already capture dH_next, so the independent Q contribution
        # can update the private FP32 state while Cube starts computing dR.
        if const_expr(not tail_safe):
            self._outer(output, qg)
            self._prepare_state(
                (
                    from_tensor(
                        self.result.next_read(),
                        data_format="nd",
                        layout=make_layout((VALUE_DIM, D_HALF)),
                    )
                    if self.dqg_cache is not None
                    else from_tensor(
                        self.result.next_read(), layout=make_layout((VALUE_DIM, D_HALF))
                    )
                )
            )
        # Keep the accepted L0C accumulation order and FP32 public scratch.
        mem_copy(self.token_a.next_write(), kg)
        mem_copy(self.state_b.next_write(), high)
        total = self.token_c.next_write()
        matmul(total, self.token_a.next_read(), self.state_b.next_read(), init=True)
        mem_copy(self.matrix_a.next_write(), aqk.permute((1, 0)))
        mem_copy(
            from_tensor(
                self.token_b.next_write(),
                data_format="nz",
                layout=make_layout((KEY_DIM, CHUNK_SIZE)),
            ),
            output.permute((1, 0)),
        )
        matmul(
            total,
            self.matrix_a.next_read(),
            from_tensor(
                self.token_b.next_read(),
                data_format="nz",
                layout=make_layout((KEY_DIM, CHUNK_SIZE)),
            ),
            init=False,
        )
        dr_fp32 = self.token_c.next_read()
        self._store_token_result(dr_fp32, gm_dr, self.dr_cache)
        mem_copy(self.dr.next_write(), dr_fp32)
        dr = self.dr.next_read()
        if const_expr(tail_safe):
            self._outer(output, qg)
            self._prepare_state(
                (
                    from_tensor(
                        self.result.next_read(),
                        data_format="nd",
                        layout=make_layout((VALUE_DIM, D_HALF)),
                    )
                    if self.dqg_cache is not None
                    else from_tensor(
                        self.result.next_read(), layout=make_layout((VALUE_DIM, D_HALF))
                    )
                )
            )
        self._outer(dr, w)
        self._finish_state(
            (
                from_tensor(
                    self.result.next_read(),
                    data_format="nd",
                    layout=make_layout((VALUE_DIM, D_HALF)),
                )
                if self.dqg_cache is not None
                else from_tensor(
                    self.result.next_read(), layout=make_layout((VALUE_DIM, D_HALF))
                )
            )
        )
        mem_copy(
            tile_slice(
                stage_two_gm_dh0[sequence, head, None, None],
                (VALUE_DIM, D_HALF),
                (0, get_subblock_id()),
            ),
            self.state,
        )
        self._token_state(output, saved, gm_dqg, self.scale, self.dqg_cache)
        mem_copy(self.token_a.next_write(), value)
        left = self.token_a.next_read()
        mem_copy(self.state_b.next_write(), high.permute((1, 0)))
        dkg = self.token_c.next_write()
        matmul(dkg, left, self.state_b.next_read(), init=True)
        mem_copy(self.state_b.next_write(), low.permute((1, 0)))
        matmul(dkg, left, self.state_b.next_read(), init=False)
        self._store_token_result(self.token_c.next_read(), gm_dkg)
        self._token_state(dr, saved, gm_dw, -1.0, self.dw_cache)


class StageTwoPersistentState:
    """Update one head's FP32 state for the current reverse-scan chunk."""

    def __init__(self, scale):
        self.scale = scale
        value_shard_rows = VALUE_DIM
        self.live_rows = VALUE_DIM // 2
        self.cube_rows = max(CUBE_TILE_ROWS, value_shard_rows)
        self.aiv_rows = self.cube_rows // 2
        # State is sharded by value; token-major dR is sharded by token.
        # Cube/FIXP changes ownership at the two products without a VF transpose.
        self.value_rows = value_shard_rows
        resources = StageTwoMatmulResources(value_shard_rows)
        self.d_v_new_state_matmul = StageTwoPipelineMatmul(
            value_shard_rows, True, resources
        )
        self.d_state_value_matmul = StageTwoPipelineMatmul(
            value_shard_rows, False, resources
        )
        fp32_storage = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((2 * self.aiv_rows + 1, KEY_DIM)),
            depth=1,
        ).next_write()
        self.state_gradient = fp32_storage[: self.aiv_rows, None]
        self.output_state_gradient = fp32_storage[
            self.aiv_rows : 2 * self.aiv_rows, None
        ]
        self.gate_last = fp32_storage[2 * self.aiv_rows : 2 * self.aiv_rows + 1, None]

    @jit
    def prepare_state_update(self):
        """Decay and output-state addition need no dR.T @ W result."""
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            for key in range(0, KEY_DIM, D_HALF):
                decay = vexp(vload(self.gate_last, key), mask=mask)
                for value in drange(self.live_rows):
                    offset = value * self.state_gradient.physical_layout.stride[0] + key
                    state = vmul(vload(self.state_gradient, offset), decay, mask=mask)
                    output = vmuls(
                        vload(self.output_state_gradient, offset), self.scale, mask=mask
                    )
                    vstore(
                        self.state_gradient,
                        offset,
                        vadd(state, output, mask=mask),
                        mask,
                    )
            vmem_bar("vst_vld")

    @jit
    def finish_state_update(self, contribution):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            for offset in range(0, self.live_rows * KEY_DIM, D_HALF):
                value = vsub(
                    vload(self.state_gradient, offset),
                    vload(contribution, offset),
                    mask=mask,
                )
                vstore(self.state_gradient, offset, value, mask)
            vmem_bar("vst_vld")

    @jit
    def run(
        self,
        gm_kg,
        gm_w,
        gm_g_cumsum,
        gm_d_v_new,
        gm_final_state_gradient,
        gm_task_metadata,
        gm_state_gradients,
        gm_dh0,
        gm_Aqk,
        gm_output_gradient,
        gm_masked_aqk,
        gm_masked_output_gradient,
        gm_masked_kg,
        gm_masked_w,
        gm_value_scratch,
        padded_layout: dtypes.int64,
        selected_task: dtypes.int64,
        selected_chunk: dtypes.int64,
    ):
        vector = StageTwoVector()
        aqk_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        )
        do_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, self.value_rows)),
            depth=1,
        )
        aqk_l0a = _cube_ring(
            MemLoc.L0A, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        )
        do_l0b = _cube_ring(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((self.value_rows, CHUNK_SIZE)),
            depth=1,
        )
        value_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, self.value_rows)),
            depth=1,
        )

        # The previous chunk of this value head published FP32 dh0. Other
        # value heads in a GQA owner have separate dh0 matrices.
        task = selected_task
        value_tile = get_subblock_id()
        task_record = HEADER_WORDS + task * TASK_WORDS
        seq_idx = gm_task_metadata[task_record + TASK_SEQUENCE]
        compact_token_start = gm_task_metadata[task_record + TASK_COMPACT_TOKEN_START]
        chunks = gm_task_metadata[task_record + TASK_CHUNKS]
        seq_len = gm_task_metadata[task_record + TASK_LENGTH]
        value_head = gm_task_metadata[task_record + TASK_VALUE_HEAD]
        if selected_chunk + 1 < chunks:
            mem_copy(
                tile_slice(self.state_gradient, (self.live_rows, KEY_DIM), (0, 0)),
                tile_slice(
                    gm_dh0[seq_idx, value_head, None, None],
                    (self.live_rows, KEY_DIM),
                    (value_tile, 0),
                ),
            )
        else:
            mem_copy(
                tile_slice(self.state_gradient, (self.live_rows, KEY_DIM), (0, 0)),
                tile_slice(
                    gm_final_state_gradient[seq_idx, value_head, None, None],
                    (self.live_rows, KEY_DIM),
                    (value_tile, 0),
                ),
            )
        storage_batch = seq_idx
        source_token_start = 0
        if not padded_layout:
            storage_batch = 0
            source_token_start = compact_token_start
        kg_source = gm_kg[storage_batch, value_head, None, None]
        w_source = gm_w[storage_batch, value_head, None, None]
        gate_source = gm_g_cumsum[storage_batch, value_head, None, None]
        d_v_new_destination = gm_d_v_new[0, 0, None, None]
        initial_state_gradient = gm_dh0[seq_idx, value_head, None, None]
        output_state_gradient_rows = tile_slice(
            self.output_state_gradient, (self.live_rows, KEY_DIM), (0, 0)
        )
        if chunks > 0:
            chunk = selected_chunk
            valid_rows = seq_len - chunk * CHUNK_SIZE
            if valid_rows > CHUNK_SIZE:
                valid_rows = CHUNK_SIZE
            source_row_start = source_token_start + chunk * CHUNK_SIZE
            compact_row_start = get_block_idx() * CHUNK_SIZE
            current_state_tile = tile_slice(
                gm_state_gradients,
                (self.live_rows, KEY_DIM),
                (
                    (2 * get_block_idx() + 1) * (VALUE_DIM // self.live_rows)
                    + value_tile,
                    0,
                ),
            )
            mem_copy(
                current_state_tile,
                tile_slice(self.state_gradient, (self.live_rows, KEY_DIM), (0, 0)),
            )
            matrix_batch = dtypes.int64(storage_batch)
            matrix_row = source_row_start
            if not padded_layout:
                matrix_batch = _tnd_chunk_index(compact_token_start, seq_idx, chunk)
                matrix_row = 0
            matrix_source = gm_Aqk[matrix_batch, value_head, None, None]
            output_source = gm_output_gradient[storage_batch, value_head, None, None]
            direct = 0
            if (
                valid_rows == CHUNK_SIZE
                and kg_source.physical_layout.stride[-1] == 1
                and w_source.physical_layout.stride[-1] == 1
                and matrix_source.physical_layout.stride[-1] == 1
                and output_source.physical_layout.stride[-1] == 1
            ):
                direct = 1
            kg_tile = tile_slice(
                gm_masked_kg, (CHUNK_SIZE, KEY_DIM), (get_block_idx(), 0)
            )
            w_tile = tile_slice(
                gm_masked_w, (CHUNK_SIZE, KEY_DIM), (get_block_idx(), 0)
            )
            aqk_tile = tile_slice(
                gm_masked_aqk, (CHUNK_SIZE, CHUNK_SIZE), (get_block_idx(), 0)
            )
            do_tile = tile_slice(
                gm_masked_output_gradient, (CHUNK_SIZE, VALUE_DIM), (get_block_idx(), 0)
            )
            if direct == 0:
                row = get_subblock_id() * (CHUNK_SIZE // 2)
                valid = max(0, min(CHUNK_SIZE // 2, valid_rows - row))
                vector.copy_window(
                    kg_source,
                    source_row_start + row,
                    kg_tile,
                    row,
                    valid,
                    KEY_DIM,
                    False,
                )
                vector.copy_window(
                    w_source, source_row_start + row, w_tile, row, valid, KEY_DIM, False
                )
                vector.copy_window(
                    matrix_source,
                    matrix_row + row,
                    aqk_tile,
                    row,
                    valid,
                    CHUNK_SIZE,
                    False,
                )
                vector.copy_window(
                    output_source,
                    source_row_start + row,
                    do_tile,
                    row,
                    valid,
                    KEY_DIM,
                    False,
                )
                _sync_cube_vector_handoff(0)
            # QTerm/gate are ready after the phase handoff and independent
            # of dH packing. Launch their DMA before that VF.
            output_state_gradient_tile = tile_slice(
                gm_state_gradients,
                (self.live_rows, KEY_DIM),
                (
                    (2 * get_block_idx()) * (VALUE_DIM // self.live_rows) + value_tile,
                    0,
                ),
            )
            mem_copy(output_state_gradient_rows, output_state_gradient_tile)
            mem_copy(
                self.gate_last,
                tile_slice(
                    gate_source,
                    (1, KEY_DIM),
                    (source_row_start + valid_rows - 1, 0),
                ),
            )
            self.d_v_new_state_matmul._publish_left(self.state_gradient)
            product = self.d_v_new_state_matmul
            if direct == 1:
                mem_copy(
                    product.token_gm_l1.next_write(),
                    _full_token_tile(kg_source, source_row_start, KEY_DIM),
                )
            else:
                mem_copy(product.token_gm_l1.next_write(), kg_tile)
            mem_copy(product.left_l0a.next_write(), product.token_gm_l1.next_read())
            mem_copy(
                product.right_l0b.next_write(),
                tile_slice(
                    product.operand_l1.next_read(), product.operand_shape, (0, 0)
                ),
            )
            total = product.result_l0c.next_write()
            matmul(
                total,
                product.left_l0a.next_read(),
                product.right_l0b.next_read(),
                init=True,
            )
            if direct == 1:
                mem_copy(
                    aqk_l1.next_write(),
                    _full_token_tile(matrix_source, matrix_row, CHUNK_SIZE),
                )
            else:
                mem_copy(aqk_l1.next_write(), aqk_tile)
            if direct == 1:
                mem_copy(
                    do_l1.next_write(),
                    _token_rows_tile(
                        output_source, source_row_start, CHUNK_SIZE, self.value_rows, 0
                    ),
                )
            else:
                mem_copy(
                    do_l1.next_write(),
                    tile_slice(do_tile, (CHUNK_SIZE, self.value_rows), (0, 0)),
                )
            mem_copy(aqk_l0a.next_write(), aqk_l1.next_read().permute((1, 0)))
            mem_copy(do_l0b.next_write(), do_l1.next_read().permute((1, 0)))
            matmul(total, aqk_l0a.next_read(), do_l0b.next_read(), init=False)
            summed = product.result_l0c.next_read()
            # Preserve FP32 in GM; the Cube consumer still requires BF16 L1.
            mem_copy(value_l1.next_write(), summed)
            if direct == 1:
                mem_copy(
                    _token_rows_tile(
                        d_v_new_destination,
                        compact_row_start,
                        CHUNK_SIZE,
                        self.value_rows,
                        0,
                    ),
                    summed,
                )
            else:
                mem_copy(
                    tile_slice(
                        gm_value_scratch,
                        (CHUNK_SIZE, self.value_rows),
                        (get_block_idx(), 0),
                    ),
                    summed,
                )
            state_product = self.d_state_value_matmul
            if direct == 1:
                mem_copy(
                    state_product.token_gm_l1.next_write(),
                    _full_token_tile(w_source, source_row_start, KEY_DIM),
                )
            else:
                mem_copy(state_product.token_gm_l1.next_write(), w_tile)
            state_product._compute(
                value_l1.next_read(), state_product.token_gm_l1.next_read()
            )
            if direct == 0:
                _sync_cube_vector_handoff(1)
                for row in range(get_subblock_id(), valid_rows, get_subblock_dim()):
                    # Copy this value shard through the existing FP32 row channel.
                    mem_copy(
                        tile_slice(
                            vector.state_row.next_write(), (1, self.value_rows), (0, 0)
                        ),
                        tile_slice(
                            gm_value_scratch,
                            (1, self.value_rows),
                            (get_block_idx() * CHUNK_SIZE + row, 0),
                        ),
                    )
                    mem_copy(
                        tile_slice(
                            d_v_new_destination,
                            (1, self.value_rows),
                            (compact_row_start + row, 0),
                        ),
                        tile_slice(
                            vector.state_row.next_read(), (1, self.value_rows), (0, 0)
                        ),
                    )
                _sync_cube_vector_handoff(3)
            self.prepare_state_update()
            self.finish_state_update(self.d_state_value_matmul.next_read())
        mem_copy(
            tile_slice(
                initial_state_gradient, (self.live_rows, KEY_DIM), (value_tile, 0)
            ),
            tile_slice(self.state_gradient, (self.live_rows, KEY_DIM), (0, 0)),
        )


@jit
def _finish_full_track_chunk(
    gm_effective_query,
    gm_effective_key,
    gm_value,
    gm_activated_beta,
    gm_Akk,
    gm_saved_qg,
    gm_saved_kg,
    gm_g_cumsum,
    gm_stage_two_d_v_new,
    gm_stage_two_d_w,
    gm_stage_two_d_qg,
    gm_stage_two_d_kg,
    gm_stage_two_d_Aqk,
    gm_stage_two_d_gate_last,
    gm_task_metadata,
    gm_public_d_value,
    gm_public_d_beta,
    gm_raw_gate,
    gm_A_log,
    gm_dt_bias,
    gm_q_rstd,
    gm_k_rstd,
    gm_public_d_query,
    gm_public_d_key,
    gm_public_d_gate,
    gm_public_dA,
    gm_public_dBias,
    gm_inverse,
    gm_values,
    gm_keys,
    gm_gamma,
    gm_beta,
    gm_d_v_new_low,
    gm_d_w_low,
    gm_weighted_v,
    gm_weighted_k,
    gm_d_beta_v,
    gm_d_beta_k,
    gm_d_beta_wy,
    gm_d_lower_unmasked,
    gm_qg,
    gm_kg,
    gm_d_qg,
    gm_d_kg,
    gm_d_gate_last,
    gm_d_query_base,
    gm_d_key_base,
    gm_d_gate_base,
    gm_query,
    gm_d_Aqk,
    gm_d_beta_matrix,
    gm_d_query_pair,
    gm_d_key_pair,
    gm_d_gate_pair,
    gm_d_beta_pair,
    gm_d_value,
    gm_d_beta,
    scale: dtypes.float32,
    fused_beta: dtypes.int64,
    inverse_factor: dtypes.float32,
    has_bias: dtypes.int64,
    use_l2norm: dtypes.int64,
    lower_bound: dtypes.float32,
    beta_fp32: Constexpr[bool],
    padded_layout: dtypes.int64,
    selected_task: dtypes.int64,
    selected_chunk: dtypes.int64,
    aligned_only: Constexpr[bool],
    cube_causal: Constexpr[bool],
    suppress_single_chunk_dA: dtypes.int64,
    resident_dqg=None,
    resident_dr=None,
    resident_dw=None,
):
    gm_inverse = tile_slice(gm_inverse, (CHUNK_SIZE, CHUNK_SIZE), (get_block_idx(), 0))
    gm_values = tile_slice(
        gm_values, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_keys = tile_slice(
        gm_keys, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_gamma = tile_slice(
        gm_gamma, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_beta = tile_slice(gm_beta, (CHUNK_SIZE,), (get_block_idx(),))
    gm_d_v_new_low = tile_slice(
        gm_d_v_new_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_w_low = tile_slice(
        gm_d_w_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_weighted_v = tile_slice(
        gm_weighted_v, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_weighted_k = tile_slice(
        gm_weighted_k, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_beta_v = tile_slice(
        gm_d_beta_v, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_beta_k = tile_slice(
        gm_d_beta_k, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_beta_wy = tile_slice(gm_d_beta_wy, (CHUNK_SIZE,), (get_block_idx(),))
    gm_d_lower_unmasked = tile_slice(
        gm_d_lower_unmasked, (CHUNK_SIZE, CHUNK_SIZE), (get_block_idx(), 0)
    )
    gm_qg = tile_slice(gm_qg, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0))
    gm_kg = tile_slice(gm_kg, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0))
    gm_d_qg = tile_slice(
        gm_d_qg, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_kg = tile_slice(
        gm_d_kg, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_gate_last = tile_slice(
        gm_d_gate_last, (1, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_query_base = tile_slice(
        gm_d_query_base, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_key_base = tile_slice(
        gm_d_key_base, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_gate_base = tile_slice(
        gm_d_gate_base, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_query = tile_slice(
        gm_query, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_Aqk = tile_slice(gm_d_Aqk, (CHUNK_SIZE, CHUNK_SIZE), (get_block_idx(), 0))
    gm_d_beta_matrix = tile_slice(
        gm_d_beta_matrix, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_query_pair = tile_slice(
        gm_d_query_pair, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_key_pair = tile_slice(
        gm_d_key_pair, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_gate_pair = tile_slice(
        gm_d_gate_pair, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_beta_pair = tile_slice(gm_d_beta_pair, (CHUNK_SIZE,), (get_block_idx(),))
    gm_d_value = tile_slice(
        gm_d_value, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (get_block_idx(), 0)
    )
    gm_d_beta = tile_slice(gm_d_beta, (CHUNK_SIZE,), (get_block_idx(),))
    # The fused epilogue is specialized for aligned-only inputs. Keep the
    # mixed kernel unchanged to avoid regressions on short tail workloads.
    process = StageOneProcess(
        fuse_base=aligned_only,
        resident_dr=resident_dr,
        resident_dw=resident_dw,
        adjacent_decay=aligned_only,
        cube_causal=cube_causal,
    )
    vector = process.vector
    task = selected_task
    task_record = HEADER_WORDS + task * TASK_WORDS
    seq_idx = gm_task_metadata[task_record + TASK_SEQUENCE]
    compact_token_start = gm_task_metadata[task_record + TASK_COMPACT_TOKEN_START]
    seq_len = gm_task_metadata[task_record + TASK_LENGTH]
    value_head = gm_task_metadata[task_record + TASK_VALUE_HEAD]
    task_chunks = gm_task_metadata[task_record + TASK_CHUNKS]
    raw_batch = 0
    raw_token_start = compact_token_start
    if padded_layout:
        raw_batch = seq_idx
        raw_token_start = 0
    qk_head = value_head
    query_source = gm_effective_query[raw_batch, qk_head, None, None]
    key_source = gm_effective_key[raw_batch, qk_head, None, None]
    value_source = gm_value[raw_batch, value_head, None, None]
    qg_source = gm_saved_qg[raw_batch, value_head, None, None]
    kg_source = gm_saved_kg[raw_batch, value_head, None, None]
    gamma_source = gm_g_cumsum[raw_batch, value_head, None, None]
    d_v_new_source = gm_stage_two_d_v_new[0, 0, None, None]
    d_w_source = gm_stage_two_d_w[0, 0, None, None]
    d_qg_source = gm_stage_two_d_qg[0, 0, None, None]
    d_kg_source = gm_stage_two_d_kg[0, 0, None, None]
    d_gate_last_source = gm_stage_two_d_gate_last[0, 0, None, None]
    beta_source = gm_activated_beta[raw_batch, value_head, None, None]
    value_destination = gm_public_d_value[raw_batch, value_head, None, None]
    beta_destination = gm_public_d_beta[raw_batch, value_head, None, 0]
    raw_gate_source = gm_raw_gate[raw_batch, value_head, None, None]
    gate_destination = gm_public_d_gate[raw_batch, value_head, None, None]
    vector.begin_gate_task()
    # All scratch tensors above are already sliced to this core's slot.
    chunk = selected_chunk
    scratch_chunk = dtypes.int64(0)
    # The caller selected a complete chunk; later or earlier chunks may be masked.
    valid_rows = CHUNK_SIZE
    raw_row_start = raw_token_start + chunk * CHUNK_SIZE
    compact_row_start = get_block_idx() * CHUNK_SIZE
    inverse_batch = dtypes.int64(raw_batch)
    inverse_row_start = raw_row_start
    if not padded_layout:
        inverse_batch = _tnd_chunk_index(compact_token_start, seq_idx, chunk)
        inverse_row_start = 0
    inverse_source = gm_Akk[inverse_batch, value_head, None, None]
    d_Aqk_source = gm_stage_two_d_Aqk[get_block_idx(), 0, None, None]
    gm_query = _full_token_tile(query_source, raw_row_start, KEY_DIM)
    gm_keys = _full_token_tile(key_source, raw_row_start, KEY_DIM)
    gm_values = _full_token_tile(value_source, raw_row_start, VALUE_DIM)
    gm_gamma = _full_token_tile(gamma_source, raw_row_start, KEY_DIM)
    gm_d_qg = _full_token_tile(d_qg_source, compact_row_start, KEY_DIM)
    gm_d_kg = _full_token_tile(d_kg_source, compact_row_start, KEY_DIM)
    gm_d_Aqk = _full_token_tile(d_Aqk_source, 0, CHUNK_SIZE)
    # The full 64-row path reads these saved caches directly.
    gm_inverse = _full_token_tile(inverse_source, inverse_row_start, CHUNK_SIZE)
    gm_qg = _full_token_tile(qg_source, raw_row_start, KEY_DIM)
    gm_kg = _full_token_tile(kg_source, raw_row_start, KEY_DIM)
    gm_d_value = _full_token_tile(value_destination, raw_row_start, VALUE_DIM)
    if get_subblock_id() == 0:
        vector.load_fp32_row(
            d_gate_last_source,
            get_block_idx(),
            gm_d_gate_last,
            scratch_chunk,
            True,
            SUPPORTED_HEAD_DIM,
            6,
        )
    half_start = get_subblock_id() * (CHUNK_SIZE // 2)
    vector.load_full_chunk_rows(
        d_v_new_source,
        compact_row_start + half_start,
        gm_d_v_new_low,
        half_start,
        SUPPORTED_HEAD_DIM,
        True,
        0,
        resident=resident_dr,
    )
    vector.load_full_chunk_rows(
        d_w_source,
        compact_row_start + half_start,
        gm_d_w_low,
        half_start,
        SUPPORTED_HEAD_DIM,
        True,
        1,
        resident=resident_dw,
    )
    diagonal_ready = 1
    if get_subblock_id() == 0:
        if beta_source.physical_layout.stride[-2] == 1:
            vector.load_contiguous_beta_chunk(
                beta_source, raw_row_start, gm_beta, scratch_chunk
            )
        else:
            vector.load_strided_beta_chunk(
                beta_source, raw_row_start, gm_beta, scratch_chunk
            )
    _sync_cube_vector_handoff(0)
    process.process_chunk(
        gm_inverse,
        gm_values,
        gm_keys,
        gm_gamma,
        gm_beta,
        gm_d_v_new_low,
        gm_d_w_low,
        gm_weighted_v,
        gm_weighted_k,
        gm_d_beta_v,
        gm_d_beta_k,
        gm_d_beta_wy,
        gm_d_lower_unmasked,
        gm_qg,
        gm_kg,
        gm_d_qg,
        gm_d_kg,
        gm_d_gate_last,
        gm_d_query_base,
        gm_d_key_base,
        gm_d_gate_base,
        gm_query,
        gm_d_Aqk,
        gm_d_beta_matrix,
        gm_d_query_pair,
        gm_d_key_pair,
        gm_d_gate_pair,
        gm_d_beta_pair,
        gm_d_value,
        gm_d_beta,
        scale,
        fused_beta,
        inverse_factor,
        scratch_chunk,
        valid_rows,
        diagonal_ready,
    )
    if const_expr(aligned_only):
        # Exchange feature-owned direct outputs into the row ownership used by
        # gate and normalization epilogues, then add base terms in the same UB.
        row_shape = (CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)
        half_tile = scratch_chunk * 2 + get_subblock_id()
        if const_expr(not cube_causal):
            mem_copy(
                vector.batch_fp32[0],
                tile_slice(gm_d_query_pair, row_shape, (half_tile, 0)),
            )
            mem_copy(
                vector.batch_fp32[1],
                tile_slice(gm_d_key_pair, row_shape, (half_tile, 0)),
            )
            mem_copy(
                vector.batch_fp32[2],
                tile_slice(gm_d_gate_pair, row_shape, (half_tile, 0)),
            )
            mem_copy(
                vector.batch_bf16[0], tile_slice(gm_query, row_shape, (half_tile, 0))
            )
            mem_copy(
                vector.batch_bf16[1], tile_slice(gm_keys, row_shape, (half_tile, 0))
            )
        vector.fuse_base_chunk(
            gm_keys,
            gm_qg,
            gm_kg,
            gm_gamma,
            gm_d_beta_k,
            gm_d_qg,
            gm_d_kg,
            gm_d_gate_last,
            gm_d_gate_pair,
            scratch_chunk,
            resident_dqg,
        )
        if const_expr(cube_causal):
            _sync_cube_vector_handoff(0)
        else:
            # The gate epilogue consumes both AIVs' updated feature rows.
            # Cube has no work here; the final chunk handoff protects reuse.
            vec_sync_all()
            vec_sync_block_arrive(PIPE.MTE3, 0, mode=1)
            vec_sync_block_wait(PIPE.S, 0, mode=1)
            onchip_rewind(reset_sync_id=True)
    vector.begin_gate_chunk()
    alog = gm_A_log[value_head]
    first_half = get_subblock_id()
    half_stride = get_subblock_dim()
    for half in range(first_half, 2, half_stride):
        vector.begin_gate_half()
        vector.gate_full_chunk(
            gm_d_gate_base,
            gm_d_gate_pair,
            raw_gate_source,
            gm_dt_bias,
            gate_destination,
            scratch_chunk,
            raw_row_start,
            half,
            value_head,
            alog,
            has_bias,
            False,
            lower_bound,
        )
        vector.accumulate_gate_chunk(half)
    # ValueBetaEpilogue wrote the final dV tile directly.
    if get_subblock_id() == 0:
        beta_output_rows = valid_rows
        if (
            beta_output_rows == CHUNK_SIZE
            and beta_destination.physical_layout.stride[-2] == 1
        ):
            vector.store_contiguous_beta_chunk(
                gm_d_beta, scratch_chunk, beta_destination, raw_row_start
            )
            beta_output_rows = 0
        for row in range(beta_output_rows):
            scratch_row = scratch_chunk * CHUNK_SIZE + row
            raw_row = raw_row_start + row
            vector.store_beta_scalar(gm_d_beta, scratch_row, beta_destination, raw_row)
    # This value head is the only contributor to its Q/K head.
    # Consume the just-produced rows before the next chunk.
    vector.finish_single_head_chunk(
        gm_d_query_base,
        gm_d_query_pair,
        gm_d_key_base,
        gm_d_key_pair,
        gm_effective_query,
        gm_effective_key,
        gm_q_rstd,
        gm_k_rstd,
        gm_public_d_query,
        gm_public_d_key,
        scratch_chunk,
        raw_batch,
        qk_head,
        raw_row_start,
        raw_batch,
        raw_row_start,
        use_l2norm,
        aligned_only or cube_causal,
    )

    if suppress_single_chunk_dA and task_chunks == 1:
        # Short sequences replace only dA by the FP32 chunk JVP; dBias keeps
        # its Stage-1 publication, matching the masked-track kernel.
        vector.finish_gate_task_atomic(
            gm_public_dA, gm_public_dBias, value_head, has_bias, False
        )
    else:
        vector.finish_gate_task_atomic(
            gm_public_dA, gm_public_dBias, value_head, has_bias
        )

    if padded_layout and selected_chunk == 0:
        vector.zero_task_padding(
            gm_public_d_value,
            gm_public_d_beta,
            gm_public_d_gate,
            gm_public_d_query,
            gm_public_d_key,
            seq_idx,
            value_head,
            1,
            seq_len,
        )
    if const_expr(aligned_only):
        # process_chunk has retired the Cube consumers. Notify Cube when the
        # vector epilogues finish, before it reuses the shared UB storage.
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, 1, mode=2)
        cube_sync_block_wait(PIPE.S, 1, mode=2)
        onchip_rewind(reset_sync_id=True)
    else:
        _sync_cube_vector_handoff(1)


@jit
def run_persistent_flash_kda_grad(
    stage_two_gm_Aqk,
    stage_two_gm_qg,
    stage_two_gm_kg,
    stage_two_gm_w,
    stage_two_gm_v_new,
    stage_two_gm_g_cumsum,
    stage_two_gm_chunk_states,
    stage_two_gm_output_gradient,
    stage_two_gm_final_state_gradient,
    stage_two_gm_task_metadata,
    stage_two_gm_d_v_new,
    stage_two_gm_d_Aqk,
    stage_two_gm_d_qg,
    stage_two_gm_d_kg,
    stage_two_gm_d_w,
    stage_two_gm_d_gate_last,
    stage_two_gm_dh0,
    stage_two_gm_state_gradients,
    stage_two_gm_masked_aqk,
    stage_two_gm_masked_qg,
    stage_two_gm_masked_v_new,
    stage_two_gm_masked_output_gradient,
    stage_two_gm_d_v_new_output,
    stage_two_gm_d_kg_scratch,
    stage_two_gm_d_v_new_scratch,
    stage_two_gm_d_v_new_low,
    stage_two_gm_d_qg_scratch,
    stage_two_gm_d_w_scratch,
    stage_two_gm_state_high,
    stage_two_gm_state_residual,
    stage_two_gm_saved_state_high,
    stage_one_gm_effective_query,
    stage_one_gm_effective_key,
    stage_one_gm_value,
    stage_one_gm_activated_beta,
    stage_one_gm_Akk,
    stage_one_gm_saved_qg,
    stage_one_gm_saved_kg,
    stage_one_gm_g_cumsum,
    stage_one_gm_stage_two_d_v_new,
    stage_one_gm_stage_two_d_w,
    stage_one_gm_stage_two_d_qg,
    stage_one_gm_stage_two_d_kg,
    stage_one_gm_stage_two_d_Aqk,
    stage_one_gm_stage_two_d_gate_last,
    stage_one_gm_task_metadata,
    stage_one_gm_public_d_value,
    stage_one_gm_public_d_beta,
    stage_one_gm_raw_gate,
    stage_one_gm_A_log,
    stage_one_gm_dt_bias,
    stage_one_gm_q_rstd,
    stage_one_gm_k_rstd,
    stage_one_gm_public_d_query,
    stage_one_gm_public_d_key,
    stage_one_gm_public_d_gate,
    stage_one_gm_public_dA,
    stage_one_gm_public_dBias,
    stage_one_gm_inverse,
    stage_one_gm_values,
    stage_one_gm_keys,
    stage_one_gm_gamma,
    stage_one_gm_beta,
    stage_one_gm_d_v_new_low,
    stage_one_gm_d_w_low,
    stage_one_gm_weighted_v,
    stage_one_gm_weighted_k,
    stage_one_gm_d_beta_v,
    stage_one_gm_d_beta_k,
    stage_one_gm_d_beta_wy,
    stage_one_gm_d_lower_unmasked,
    stage_one_gm_qg,
    stage_one_gm_kg,
    stage_one_gm_d_qg,
    stage_one_gm_d_kg,
    stage_one_gm_d_gate_last,
    stage_one_gm_d_query_base,
    stage_one_gm_d_key_base,
    stage_one_gm_d_gate_base,
    stage_one_gm_query,
    stage_one_gm_d_Aqk,
    stage_one_gm_d_beta_matrix,
    stage_one_gm_d_query_pair,
    stage_one_gm_d_key_pair,
    stage_one_gm_d_gate_pair,
    stage_one_gm_d_beta_pair,
    stage_one_gm_d_value,
    stage_one_gm_d_beta,
    gm_owner_schedule,
    stage_one_fused_beta: dtypes.int64,
    stage_one_heads_per_key_head: dtypes.int64,
    stage_one_fused_gate: dtypes.int64,
    stage_one_has_bias: dtypes.int64,
    stage_one_use_l2norm: dtypes.int64,
    stage_one_softplus: dtypes.int64,
    stage_one_beta_fp32: Constexpr[bool],
    padded_layout: dtypes.int64,
    use_full_chunks: Constexpr[bool],
    needs_masked_chunks: Constexpr[bool],
    cube_causal: Constexpr[bool],
    stage_two_scale: dtypes.float32,
    stage_one_scale: dtypes.float32,
    stage_one_inverse_factor: dtypes.float32,
    stage_one_lower_bound: dtypes.float32,
    recomputed: Constexpr[bool],
):
    if const_expr(not recomputed):
        _initialize_backward_gradients(
            stage_one_gm_public_dA, stage_one_gm_public_dBias, stage_one_fused_gate
        )
    task_count = (stage_two_gm_task_metadata.shape[0] - HEADER_WORDS) // TASK_WORDS
    if (
        stage_two_gm_task_metadata[HEADER_STATUS] != STATUS_OK
        or stage_two_gm_task_metadata[HEADER_MAGIC] != MAGIC
        or stage_two_gm_task_metadata[HEADER_ABI_VERSION] != ABI_VERSION
    ):
        task_count = 0
    # One owner holds a sequence/QK-head group. Chunk specialization never
    # changes ownership, and dh0 bridges the FP32 reverse state between steps.
    groups = task_count // stage_one_heads_per_key_head
    resident_dqg = None
    resident_dr = None
    resident_dw = None
    if const_expr(use_full_chunks and not needs_masked_chunks):
        resident_dqg = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE // 2, KEY_DIM))
        )
        resident_dr = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE // 2, KEY_DIM))
        )
        resident_dw = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE // 2, KEY_DIM))
        )
    begin, end, step = _owner_loop_bounds(gm_owner_schedule, groups)
    for slot in range(begin, end, step):
        owner = _scheduled_owner(gm_owner_schedule, slot)
        first_task = owner * stage_one_heads_per_key_head
        record = HEADER_WORDS + first_task * TASK_WORDS
        chunks = stage_two_gm_task_metadata[record + TASK_CHUNKS]
        length = stage_two_gm_task_metadata[record + TASK_LENGTH]
        # Empty tasks still transmit dht and clear padded output rows.
        for reverse_index in range(max(chunks, 1)):
            chunk = max(chunks - reverse_index - 1, 0)
            for local_head in range(stage_one_heads_per_key_head):
                task = first_task + local_head
                full_chunk = dtypes.int64(0)
                if const_expr(use_full_chunks):
                    full_chunk = (chunk + 1) * CHUNK_SIZE <= length
                    if chunks > 0:
                        StageTwoTrack(
                            stage_two_scale, resident_dqg, resident_dr, resident_dw
                        ).run_step(
                            stage_two_gm_Aqk,
                            stage_two_gm_qg,
                            stage_two_gm_kg,
                            stage_two_gm_w,
                            stage_two_gm_v_new,
                            stage_two_gm_g_cumsum,
                            stage_two_gm_chunk_states,
                            stage_two_gm_output_gradient,
                            stage_two_gm_final_state_gradient,
                            stage_two_gm_task_metadata,
                            stage_two_gm_d_v_new,
                            stage_two_gm_d_Aqk,
                            stage_two_gm_d_qg,
                            stage_two_gm_d_kg,
                            stage_two_gm_d_w,
                            stage_two_gm_d_gate_last,
                            stage_two_gm_dh0,
                            padded_layout,
                            task,
                            chunk,
                            stage_one_gm_query,
                            stage_one_gm_keys,
                            stage_one_gm_values,
                            stage_one_gm_qg,
                            stage_one_gm_kg,
                            stage_one_gm_inverse,
                            tail_safe=needs_masked_chunks,
                        )
                        _sync_cube_vector_handoff(0, reset_sync_id=True)
                if const_expr(needs_masked_chunks):
                    if not use_full_chunks or not full_chunk:
                        if not use_full_chunks or chunks == 0:
                            _prepare_track_chunk(
                                stage_two_gm_Aqk,
                                stage_two_gm_qg,
                                stage_two_gm_v_new,
                                stage_two_gm_output_gradient,
                                stage_two_gm_task_metadata,
                                stage_two_gm_d_v_new,
                                stage_two_gm_d_Aqk,
                                stage_two_gm_state_gradients,
                                stage_two_gm_masked_qg,
                                stage_two_gm_masked_v_new,
                                stage_two_gm_masked_output_gradient,
                                padded_layout,
                                task,
                                chunk,
                            )
                            _sync_cube_vector_handoff(0)
                            StageTwoPersistentState(stage_two_scale).run(
                                stage_two_gm_kg,
                                stage_two_gm_w,
                                stage_two_gm_g_cumsum,
                                stage_two_gm_d_v_new,
                                stage_two_gm_final_state_gradient,
                                stage_two_gm_task_metadata,
                                stage_two_gm_state_gradients,
                                stage_two_gm_dh0,
                                stage_two_gm_Aqk,
                                stage_two_gm_output_gradient,
                                stage_two_gm_masked_aqk,
                                stage_two_gm_masked_output_gradient,
                                stage_two_gm_masked_qg,
                                stage_two_gm_masked_v_new,
                                stage_two_gm_d_v_new_output,
                                padded_layout,
                                task,
                                chunk,
                            )
                            _sync_cube_vector_handoff(0)
                            _finish_track_chunk(
                                stage_two_gm_v_new,
                                stage_two_gm_chunk_states,
                                stage_two_gm_output_gradient,
                                stage_two_gm_task_metadata,
                                stage_two_gm_d_v_new,
                                stage_two_gm_d_qg,
                                stage_two_gm_d_kg,
                                stage_two_gm_d_w,
                                stage_two_gm_d_gate_last,
                                stage_two_gm_state_gradients,
                                stage_two_gm_masked_v_new,
                                stage_two_gm_masked_output_gradient,
                                stage_two_gm_d_v_new_scratch,
                                stage_two_gm_d_v_new_low,
                                stage_two_gm_d_kg_scratch,
                                stage_two_gm_d_qg_scratch,
                                stage_two_gm_d_w_scratch,
                                stage_two_gm_state_high,
                                stage_two_gm_state_residual,
                                stage_two_gm_saved_state_high,
                                stage_two_scale,
                                padded_layout,
                                task,
                                chunk,
                            )
                            # Both stages use the same (sequence, head, chunk) owner; all Stage 1
                            # inputs produced here need only this block's Cube/AIV handoff.
                            _sync_cube_vector_handoff(0, reset_sync_id=True)

                # Join the full/tail Stage 2 lifetimes before reserving Stage 1 UB.
                _sync_cube_vector_handoff(2, reset_sync_id=True)
                with _StageOneScratchScope(
                    resident_dr=resident_dr,
                    resident_dw=resident_dw,
                    resident_dqg=resident_dqg,
                ) as stage_one_scratch:
                    if const_expr(use_full_chunks):
                        if full_chunk:
                            _finish_full_track_chunk(
                                stage_one_gm_effective_query,
                                stage_one_gm_effective_key,
                                stage_one_gm_value,
                                stage_one_gm_activated_beta,
                                stage_one_gm_Akk,
                                stage_one_gm_saved_qg,
                                stage_one_gm_saved_kg,
                                stage_one_gm_g_cumsum,
                                stage_one_gm_stage_two_d_v_new,
                                stage_one_gm_stage_two_d_w,
                                stage_one_gm_stage_two_d_qg,
                                stage_one_gm_stage_two_d_kg,
                                stage_one_gm_stage_two_d_Aqk,
                                stage_one_gm_stage_two_d_gate_last,
                                stage_one_gm_task_metadata,
                                stage_one_gm_public_d_value,
                                stage_one_gm_public_d_beta,
                                stage_one_gm_raw_gate,
                                stage_one_gm_A_log,
                                stage_one_gm_dt_bias,
                                stage_one_gm_q_rstd,
                                stage_one_gm_k_rstd,
                                stage_one_gm_public_d_query,
                                stage_one_gm_public_d_key,
                                stage_one_gm_public_d_gate,
                                stage_one_gm_public_dA,
                                stage_one_gm_public_dBias,
                                stage_one_gm_inverse,
                                stage_one_gm_values,
                                stage_one_gm_keys,
                                stage_one_gm_gamma,
                                stage_one_gm_beta,
                                stage_one_gm_d_v_new_low,
                                stage_one_gm_d_w_low,
                                stage_one_gm_weighted_v,
                                stage_one_gm_weighted_k,
                                stage_one_gm_d_beta_v,
                                stage_one_gm_d_beta_k,
                                stage_one_gm_d_beta_wy,
                                stage_one_gm_d_lower_unmasked,
                                stage_one_gm_qg,
                                stage_one_gm_kg,
                                stage_one_gm_d_qg,
                                stage_one_gm_d_kg,
                                stage_one_gm_d_gate_last,
                                stage_one_gm_d_query_base,
                                stage_one_gm_d_key_base,
                                stage_one_gm_d_gate_base,
                                stage_one_gm_query,
                                stage_one_gm_d_Aqk,
                                stage_one_gm_d_beta_matrix,
                                stage_one_gm_d_query_pair,
                                stage_one_gm_d_key_pair,
                                stage_one_gm_d_gate_pair,
                                stage_one_gm_d_beta_pair,
                                stage_one_gm_d_value,
                                stage_one_gm_d_beta,
                                stage_one_scale,
                                stage_one_fused_beta,
                                stage_one_inverse_factor,
                                stage_one_has_bias,
                                stage_one_use_l2norm,
                                stage_one_lower_bound,
                                stage_one_beta_fp32,
                                padded_layout,
                                task,
                                chunk,
                                not needs_masked_chunks,
                                cube_causal,
                                stage_one_fused_gate
                                # The FP32 single-chunk JVP below is the only
                                # writer of dA for a one-chunk task and it is
                                # launched for the softplus gate alone, so the
                                # atomic dA store may only be suppressed when
                                # that JVP will actually run. Mirrors the
                                # staged schedule's condition.
                                and stage_one_softplus
                                and stage_one_gm_effective_query.shape[2] > 0
                                and stage_one_gm_g_cumsum.shape[2] > 0
                                and stage_two_gm_chunk_states.shape[2] > 0,
                                stage_one_scratch.resident_dqg,
                                stage_one_scratch.resident_dr,
                                stage_one_scratch.resident_dw,
                            )
                    if const_expr(needs_masked_chunks):
                        if not use_full_chunks or not full_chunk:
                            _finish_masked_track_chunk(
                                stage_one_gm_effective_query,
                                stage_one_gm_effective_key,
                                stage_one_gm_value,
                                stage_one_gm_activated_beta,
                                stage_one_gm_Akk,
                                stage_one_gm_saved_qg,
                                stage_one_gm_saved_kg,
                                stage_one_gm_g_cumsum,
                                stage_one_gm_stage_two_d_v_new,
                                stage_one_gm_stage_two_d_w,
                                stage_one_gm_stage_two_d_qg,
                                stage_one_gm_stage_two_d_kg,
                                stage_one_gm_stage_two_d_Aqk,
                                stage_one_gm_stage_two_d_gate_last,
                                stage_one_gm_task_metadata,
                                stage_one_gm_public_d_value,
                                stage_one_gm_public_d_beta,
                                stage_one_gm_raw_gate,
                                stage_one_gm_A_log,
                                stage_one_gm_dt_bias,
                                stage_one_gm_q_rstd,
                                stage_one_gm_k_rstd,
                                stage_one_gm_public_d_query,
                                stage_one_gm_public_d_key,
                                stage_one_gm_public_d_gate,
                                stage_one_gm_public_dA,
                                stage_one_gm_public_dBias,
                                stage_one_gm_inverse,
                                stage_one_gm_values,
                                stage_one_gm_keys,
                                stage_one_gm_gamma,
                                stage_one_gm_beta,
                                stage_one_gm_d_v_new_low,
                                stage_one_gm_d_w_low,
                                stage_one_gm_weighted_v,
                                stage_one_gm_weighted_k,
                                stage_one_gm_d_beta_v,
                                stage_one_gm_d_beta_k,
                                stage_one_gm_d_beta_wy,
                                stage_one_gm_d_lower_unmasked,
                                stage_one_gm_qg,
                                stage_one_gm_kg,
                                stage_one_gm_d_qg,
                                stage_one_gm_d_kg,
                                stage_one_gm_d_gate_last,
                                stage_one_gm_d_query_base,
                                stage_one_gm_d_key_base,
                                stage_one_gm_d_gate_base,
                                stage_one_gm_query,
                                stage_one_gm_d_Aqk,
                                stage_one_gm_d_beta_matrix,
                                stage_one_gm_d_query_pair,
                                stage_one_gm_d_key_pair,
                                stage_one_gm_d_gate_pair,
                                stage_one_gm_d_beta_pair,
                                stage_one_gm_d_value,
                                stage_one_gm_d_beta,
                                stage_one_scale,
                                stage_one_fused_beta,
                                stage_one_inverse_factor,
                                stage_one_heads_per_key_head,
                                stage_one_fused_gate,
                                stage_one_has_bias,
                                stage_one_use_l2norm,
                                stage_one_softplus,
                                stage_one_lower_bound,
                                stage_one_beta_fp32,
                                padded_layout,
                                stage_one_fused_gate
                                and stage_one_softplus
                                and stage_one_gm_effective_query.shape[2] > 0
                                and stage_one_gm_g_cumsum.shape[2] > 0
                                and stage_two_gm_chunk_states.shape[2] > 0,
                                task,
                                chunk,
                            )
    # Full-chunk dispatch requires a bounded gate; softplus never selects it.
    if const_expr(not use_full_chunks):
        # The epilogue starts a new unconditional allocation epoch after the owner loop.
        _sync_cube_vector_handoff(3, reset_sync_id=True)
        if (
            stage_one_fused_gate
            and stage_one_softplus
            and stage_one_gm_effective_query.shape[2] > 0
            and stage_one_gm_g_cumsum.shape[2] > 0
            and stage_two_gm_chunk_states.shape[2] > 0
        ):
            _run_single_chunk_gate_sensitivity(
                stage_one_gm_effective_query,
                stage_one_gm_effective_key,
                stage_one_gm_value,
                stage_one_gm_g_cumsum,
                stage_one_gm_activated_beta,
                stage_two_gm_chunk_states,
                stage_two_gm_output_gradient,
                stage_two_gm_final_state_gradient,
                stage_one_gm_task_metadata,
                stage_one_gm_public_dA,
                stage_one_gm_public_d_beta,
                stage_one_gm_raw_gate,
                stage_one_gm_A_log,
                stage_one_gm_dt_bias,
                stage_one_lower_bound,
                stage_one_softplus,
                stage_one_has_bias,
                stage_one_scale,
                stage_one_heads_per_key_head,
                stage_one_beta_fp32,
                padded_layout,
                stage_one_fused_beta,
                stage_one_inverse_factor,
                True,
                gm_owner_schedule,
                stage_one_gm_d_beta_matrix,
            )
