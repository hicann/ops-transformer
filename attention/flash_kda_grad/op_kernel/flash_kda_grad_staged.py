# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Four-stage backward; fused recompute adds a preceding global forward phase."""

from __future__ import annotations

from cannbotdsl import (
    Buffer,
    Constexpr,
    MemLoc,
    const_expr,
    dtypes,
)
from cannbotdsl import make_layout, permute
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.control_flow import range as drange
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import (
    create_mask,
    vadd,
    vexp,
    vload,
    vmem_bar,
    vmul,
    vmuls,
    vstore,
    vsub,
)
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
)
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.sync import (
    onchip_rewind,
)
from cannbotdsl.tensor import tile_slice

try:
    from . import flash_kda_grad_metadata as _metadata
except ImportError:
    import flash_kda_grad_metadata as _metadata

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
        STATUS_OK,
        StageTwoMatmulResources,
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
        _gradient_slot,
        _initialize_backward_gradients,
        _prepare_track_chunk,
        _run_single_chunk_gate_sensitivity,
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
        STATUS_OK,
        StageTwoMatmulResources,
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
        _gradient_slot,
        _initialize_backward_gradients,
        _prepare_track_chunk,
        _run_single_chunk_gate_sensitivity,
        _sync_cube_vector_handoff,
        _tnd_chunk_index,
        _token_rows_tile,
    )


class StagedReverseState:
    """Keep a value shard of FP32 dH resident throughout the reverse scan."""

    def __init__(self, scale, value_shard_rows):
        self.scale = scale
        self.value_parts = VALUE_DIM // value_shard_rows
        self.live_rows = value_shard_rows // 2
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
        self.state_gradient = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.aiv_rows, KEY_DIM))
        )
        self.output_state_gradient = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.aiv_rows, KEY_DIM))
        )
        self.gate_last = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, KEY_DIM)))

    @jit
    def prepare_state_update(self):
        """Decay and output-state addition need no dR.T @ W result."""
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            for key in range(0, KEY_DIM, D_HALF):
                decay = vexp(vload(self.gate_last, key), mask=mask)
                for value in drange(self.live_rows):
                    offset = value * self.state_gradient.physical_stride[0] + key
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
        task_count: dtypes.int64,
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

        begin, end, step = _metadata._staged_reverse_loop_bounds(
            task_count, self.value_parts
        )
        for owner in range(begin, end, step):
            task, value_part = _metadata._staged_reverse_owner(owner, self.value_parts)
            value_tile = value_part * 2 + get_subblock_id()
            task_record = HEADER_WORDS + task * TASK_WORDS
            seq_idx = gm_task_metadata[task_record + TASK_SEQUENCE]
            compact_token_start = gm_task_metadata[
                task_record + TASK_COMPACT_TOKEN_START
            ]
            chunks = gm_task_metadata[task_record + TASK_CHUNKS]
            seq_len = gm_task_metadata[task_record + TASK_LENGTH]
            value_head = gm_task_metadata[task_record + TASK_VALUE_HEAD]
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
            for reverse_index in range(chunks):
                chunk = chunks - reverse_index - 1
                gradient_slot = _gradient_slot(gm_task_metadata, task, chunk, True)
                valid_rows = seq_len - chunk * CHUNK_SIZE
                if valid_rows > CHUNK_SIZE:
                    valid_rows = CHUNK_SIZE
                source_row_start = source_token_start + chunk * CHUNK_SIZE
                compact_row_start = gradient_slot * CHUNK_SIZE
                current_state_tile = tile_slice(
                    gm_state_gradients,
                    (self.live_rows, KEY_DIM),
                    (
                        (2 * gradient_slot + 1) * (VALUE_DIM // self.live_rows)
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
                output_source = gm_output_gradient[
                    storage_batch, value_head, None, None
                ]
                direct = 0
                if (
                    valid_rows == CHUNK_SIZE
                    and kg_source.physical_stride[-1] == 1
                    and w_source.physical_stride[-1] == 1
                    and matrix_source.physical_stride[-1] == 1
                    and output_source.physical_stride[-1] == 1
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
                    gm_masked_output_gradient,
                    (CHUNK_SIZE, VALUE_DIM),
                    (get_block_idx(), 0),
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
                        w_source,
                        source_row_start + row,
                        w_tile,
                        row,
                        valid,
                        KEY_DIM,
                        False,
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
                        (2 * gradient_slot) * (VALUE_DIM // self.live_rows)
                        + value_tile,
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
                            output_source,
                            source_row_start,
                            CHUNK_SIZE,
                            self.value_rows,
                            value_part,
                        ),
                    )
                else:
                    mem_copy(
                        do_l1.next_write(),
                        tile_slice(
                            do_tile, (CHUNK_SIZE, self.value_rows), (0, value_part)
                        ),
                    )
                mem_copy(aqk_l0a.next_write(), permute(aqk_l1.next_read(), (1, 0)))
                mem_copy(do_l0b.next_write(), permute(do_l1.next_read(), (1, 0)))
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
                            value_part,
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
                                vector.state_row.next_write(),
                                (1, self.value_rows),
                                (0, 0),
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
                                (compact_row_start + row, value_part),
                            ),
                            tile_slice(
                                vector.state_row.next_read(),
                                (1, self.value_rows),
                                (0, 0),
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
            _sync_cube_vector_handoff(3)


@jit
def run_staged_flash_kda_grad(
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
    block_idx = get_block_idx()
    block_num = get_block_num()
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
    # Stage 1/4: independent dAqk and QTerm, distributed by chunk.
    for task in range(task_count):
        record = HEADER_WORDS + task * TASK_WORDS
        chunks = stage_two_gm_task_metadata[record + TASK_CHUNKS]
        first_chunk = _metadata._staged_chunk_start(
            stage_two_gm_task_metadata,
            record,
            stage_one_heads_per_key_head,
            block_idx,
            block_num,
        )
        for chunk in range(first_chunk, max(chunks, 1), block_num):
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
                staged=True,
            )
    _sync_cube_vector_handoff(0, grid=True)
    # Stage 2/4: reverse recurrence. Small BN uses eight value shards/head,
    # exactly as profiler; each shard's state remains in UB across chunks.
    if _metadata._staged_reverse_is_unsplit(task_count, block_num):
        reverse = StagedReverseState(stage_two_scale, _metadata.STAGED_FULL_VALUE_ROWS)
        reverse.run(
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
            task_count,
        )
    else:
        onchip_rewind(reset_sync_id=True)
        reverse = StagedReverseState(stage_two_scale, _metadata.STAGED_SPLIT_VALUE_ROWS)
        reverse.run(
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
            task_count,
        )
    _sync_cube_vector_handoff(1, grid=True)
    # Stage 3/4: chunk-parallel gradients consuming the saved outgoing dH.
    for task in range(task_count):
        record = HEADER_WORDS + task * TASK_WORDS
        chunks = stage_two_gm_task_metadata[record + TASK_CHUNKS]
        first_chunk = _metadata._staged_chunk_start(
            stage_two_gm_task_metadata,
            record,
            stage_one_heads_per_key_head,
            block_idx,
            block_num,
        )
        for chunk in range(first_chunk, max(chunks, 1), block_num):
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
                staged=True,
            )
    # Stage 4 has the same chunk owners, including all heads of a GQA
    # group. Only a local handoff is needed; its scratch is reused per core.
    _sync_cube_vector_handoff(2, reset_sync_id=True)
    for group in range(task_count // stage_one_heads_per_key_head):
        first_task = group * stage_one_heads_per_key_head
        record = HEADER_WORDS + first_task * TASK_WORDS
        chunks = stage_two_gm_task_metadata[record + TASK_CHUNKS]
        first_chunk = _metadata._staged_chunk_start(
            stage_two_gm_task_metadata,
            record,
            stage_one_heads_per_key_head,
            block_idx,
            block_num,
        )
        for chunk in range(first_chunk, max(chunks, 1), block_num):
            for local_head in range(stage_one_heads_per_key_head):
                task = first_task + local_head
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
                    staged=True,
                )
    _sync_cube_vector_handoff(3, reset_sync_id=True)
    if stage_one_fused_gate and stage_one_softplus:
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
            stage_one_scale,
            stage_one_heads_per_key_head,
            stage_one_beta_fp32,
            padded_layout,
            stage_one_fused_beta,
            stage_one_inverse_factor,
            recomputed,
            gm_owner_schedule,
            stage_one_gm_d_beta_matrix,
        )
