# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Staged task scheduling over shared state and chunk implementations."""

from __future__ import annotations
from cannbotdsl import Constexpr, dtypes
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.sync import onchip_rewind

if __package__:
    from .flash_kda_grad_chunk import _finish_masked_track_chunk
    from .flash_kda_grad_outputs import _run_single_chunk_gate_sensitivity
    from .flash_kda_grad_buffer import (
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
        TASK_WORDS,
    )
    from .flash_kda_grad_state import (
        StagedStateBackward,
        _finish_track_chunk,
        _prepare_track_chunk,
    )
    from . import flash_kda_grad_metadata as _metadata
else:
    from flash_kda_grad_chunk import _finish_masked_track_chunk
    from flash_kda_grad_outputs import _run_single_chunk_gate_sensitivity
    from flash_kda_grad_buffer import (
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
        TASK_WORDS,
    )
    from flash_kda_grad_state import (
        StagedStateBackward,
        _finish_track_chunk,
        _prepare_track_chunk,
    )
    import flash_kda_grad_metadata as _metadata


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
        reverse = StagedStateBackward(stage_two_scale, _metadata.STAGED_FULL_VALUE_ROWS)
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
        reverse = StagedStateBackward(
            stage_two_scale, _metadata.STAGED_SPLIT_VALUE_ROWS
        )
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
                    round_raw_gate=recomputed,
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
            recomputed,
            gm_owner_schedule,
            stage_one_gm_d_beta_matrix,
            round_raw_gate=recomputed,
        )
