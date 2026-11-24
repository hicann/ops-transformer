# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Single device entry: initialization, optional reconstruction and pipeline dispatch."""

from __future__ import annotations
import cannbotdsl
from cannbotdsl import Constexpr, const_expr, dtypes
from cannbotdsl.lang.kernel import kernel

if __package__:
    from .flash_kda_grad_outputs import (
        _finish_deterministic_gate_gradients,
        _initialize_backward_gradients,
    )
    from .flash_kda_grad_buffer import (
        _SharedCubeScope,
        _sync_cube_vector_handoff,
    )
    from .flash_kda_grad_persistent import run_persistent_flash_kda_grad
    from .flash_kda_grad_staged import run_staged_flash_kda_grad
else:
    from flash_kda_grad_outputs import (
        _finish_deterministic_gate_gradients,
        _initialize_backward_gradients,
    )
    from flash_kda_grad_buffer import (
        _SharedCubeScope,
        _sync_cube_vector_handoff,
    )
    from flash_kda_grad_persistent import run_persistent_flash_kda_grad
    from flash_kda_grad_staged import run_staged_flash_kda_grad


def _backward_implementation(staged=False):
    return run_staged_flash_kda_grad if staged else run_persistent_flash_kda_grad


def _recompute_implementation(staged=False):
    try:
        from .flash_kda_grad_recompute import recompute_tracks, recompute_staged_tracks
    except ImportError:
        from flash_kda_grad_recompute import recompute_tracks, recompute_staged_tracks
    return recompute_staged_tracks if staged else recompute_tracks


_PROFILE_INPUTS = (
    "stage_two_gm_Aqk",
    "stage_two_gm_chunk_states",
    "stage_two_gm_output_gradient",
    "stage_two_gm_final_state_gradient",
    "stage_two_gm_task_metadata",
    "stage_one_gm_effective_query",
    "stage_one_gm_effective_key",
    "stage_one_gm_value",
)


_PROFILE_OUTPUTS = (
    "stage_one_gm_public_d_query",
    "stage_one_gm_public_d_key",
    "stage_one_gm_public_d_value",
    "stage_two_gm_dh0",
    "stage_one_gm_public_d_gate",
    "stage_one_gm_public_d_beta",
    "stage_one_gm_public_dA",
    "stage_one_gm_public_dBias",
)


BACKWARD_PROFILE = cannbotdsl.ProfileSpec(
    name="DSL_flash_kda_grad",
    op_type="FlashKDAGrad",
    inputs=_PROFILE_INPUTS,
    outputs=_PROFILE_OUTPUTS,
)


@kernel(profile=BACKWARD_PROFILE)
class FlashKDAGradKernel:
    """One device entry; cache mode and schedule specialize at compile time."""

    def __call__(
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
        initial_state,
        recomputed_u,
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
        has_initial_state: dtypes.int64,
        round_u_pre: dtypes.int64,
        stage_two_scale: dtypes.float32,
        stage_one_scale: dtypes.float32,
        stage_one_inverse_factor: dtypes.float32,
        stage_one_lower_bound: dtypes.float32,
        has_dht: dtypes.int64,
        staged: Constexpr[bool],
        recompute: Constexpr[bool],
    ):
        with _SharedCubeScope():
            _initialize_backward_gradients(
                stage_one_gm_public_dA,
                stage_one_gm_public_dBias,
                stage_two_gm_final_state_gradient,
                stage_one_fused_gate,
                has_dht,
            )
            if const_expr(recompute):
                rebuild = _recompute_implementation(
                    staged or needs_masked_chunks or (not use_full_chunks)
                )
                rebuild(
                    stage_one_gm_effective_query,
                    stage_one_gm_effective_key,
                    stage_one_gm_value,
                    stage_one_gm_raw_gate,
                    stage_one_gm_activated_beta,
                    stage_one_gm_Akk,
                    stage_two_gm_chunk_states,
                    stage_one_gm_A_log,
                    stage_one_gm_dt_bias,
                    stage_two_gm_task_metadata,
                    stage_two_gm_w,
                    recomputed_u,
                    stage_two_gm_v_new,
                    stage_two_gm_qg,
                    stage_two_gm_kg,
                    stage_two_gm_g_cumsum,
                    initial_state,
                    padded_layout,
                    stage_one_fused_gate,
                    stage_one_has_bias,
                    stage_one_softplus,
                    stage_one_lower_bound,
                    stage_one_fused_beta,
                    has_initial_state,
                    gm_owner_schedule,
                    round_u_pre,
                )
                if const_expr(
                    use_full_chunks
                    and needs_masked_chunks
                    and cube_causal
                    and (not staged)
                ):
                    _sync_cube_vector_handoff(2, grid=True, reset_sync_id=True)
                else:
                    _sync_cube_vector_handoff(2, grid=True)
            _backward_implementation(staged)(
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
                stage_one_fused_beta,
                stage_one_heads_per_key_head,
                stage_one_fused_gate,
                stage_one_has_bias,
                stage_one_use_l2norm,
                stage_one_softplus,
                stage_one_beta_fp32,
                padded_layout,
                use_full_chunks,
                needs_masked_chunks,
                cube_causal,
                stage_two_scale,
                stage_one_scale,
                stage_one_inverse_factor,
                stage_one_lower_bound,
                recompute,
            )

        _finish_deterministic_gate_gradients(
            stage_one_gm_public_dA,
            stage_one_gm_public_dBias,
            stage_one_fused_gate,
            stage_one_has_bias,
        )
