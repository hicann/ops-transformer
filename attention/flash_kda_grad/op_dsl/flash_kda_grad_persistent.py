# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Persistent task scheduling over shared state and chunk implementations."""

from __future__ import annotations
from cannbotdsl import Buffer, Constexpr, MemLoc, const_expr, dtypes, make_layout
from cannbotdsl.lang.jit import jit

if __package__:
    from .flash_kda_grad_chunk import (
        _finish_full_track_chunk,
        _finish_masked_track_chunk,
    )
    from .flash_kda_grad_outputs import _run_single_chunk_gate_sensitivity
    from .flash_kda_grad_metadata import _owner_loop_bounds, _scheduled_owner
    from .flash_kda_grad_buffer import (
        CHUNK_SIZE,
        KEY_DIM,
        _ChunkScratchScope,
        _sync_cube_vector_handoff,
    )
    from .flash_kda_grad_metadata import (
        ABI_VERSION,
        HEADER_ABI_VERSION,
        HEADER_MAGIC,
        HEADER_STATUS,
        HEADER_WORDS,
        MAGIC,
        STATUS_OK,
        TASK_CHUNKS,
        TASK_LENGTH,
        TASK_WORDS,
    )
    from .flash_kda_grad_state import (
        PersistentStateBackward,
        FullChunkStateBackward,
        _finish_track_chunk,
        _prepare_track_chunk,
    )
else:
    from flash_kda_grad_chunk import (
        _finish_full_track_chunk,
        _finish_masked_track_chunk,
    )
    from flash_kda_grad_outputs import _run_single_chunk_gate_sensitivity
    from flash_kda_grad_metadata import _owner_loop_bounds, _scheduled_owner
    from flash_kda_grad_buffer import (
        CHUNK_SIZE,
        KEY_DIM,
        _ChunkScratchScope,
        _sync_cube_vector_handoff,
    )
    from flash_kda_grad_metadata import (
        ABI_VERSION,
        HEADER_ABI_VERSION,
        HEADER_MAGIC,
        HEADER_STATUS,
        HEADER_WORDS,
        MAGIC,
        STATUS_OK,
        TASK_CHUNKS,
        TASK_LENGTH,
        TASK_WORDS,
    )
    from flash_kda_grad_state import (
        PersistentStateBackward,
        FullChunkStateBackward,
        _finish_track_chunk,
        _prepare_track_chunk,
    )


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
                        FullChunkStateBackward(
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
                            PersistentStateBackward(stage_two_scale).run(
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
                with _ChunkScratchScope(
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
                                round_raw_gate=recomputed,
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
                                round_raw_gate=recomputed,
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
                round_raw_gate=recomputed,
            )
