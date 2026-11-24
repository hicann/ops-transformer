# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Chunk-local backward orchestration and shared WY/inverse matrix contractions."""

from __future__ import annotations
from cannbotdsl import (
    Constexpr,
    MemLoc,
    PIPE,
    const_expr,
    dtypes,
    from_tensor,
    make_layout,
)
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_subblock_dim, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    create_mask,
    update_mask,
    vadd,
    vcast,
    vdups,
    vexp,
    vload,
    vload_broadcast,
    vload_unpack,
    vmem_bar,
    vmul,
    vreduce_sum,
    vstore_first,
    vstore_pack,
)
from cannbotdsl.ops.sync import (
    cube_sync_block_wait,
    onchip_rewind,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import tile_slice

if __package__:
    from .flash_kda_grad_causal import (
        CausalOperandVector,
        CausalResultVector,
        DirectCausalDeltaVector,
        InverseMaskVector,
    )
    from .flash_kda_grad_outputs import (
        BaseGradientVector,
        BetaCombineVector,
        ChunkGradientVector,
        ValueBetaEpilogueVector,
    )
    from .flash_kda_grad_buffer import (
        ChunkWorkspace,
        _gradient_slot,
        CHUNK_SIZE,
        KEY_DIM,
        LANES,
        SUPPORTED_HEAD_DIM,
        VALUE_DIM,
        _MatrixSlotView,
        _cube_ring,
        _full_token_tile,
        _sync_cube_vector_handoff,
    )
    from .flash_kda_grad_metadata import (
        HEADER_WORDS,
        TASK_CHUNKS,
        TASK_COMPACT_TOKEN_START,
        TASK_LENGTH,
        TASK_SEQUENCE,
        TASK_VALUE_HEAD,
        TASK_WORDS,
    )
else:
    from flash_kda_grad_causal import (
        CausalOperandVector,
        CausalResultVector,
        DirectCausalDeltaVector,
        InverseMaskVector,
    )
    from flash_kda_grad_outputs import (
        BaseGradientVector,
        BetaCombineVector,
        ChunkGradientVector,
        ValueBetaEpilogueVector,
    )
    from flash_kda_grad_buffer import (
        ChunkWorkspace,
        _gradient_slot,
        CHUNK_SIZE,
        KEY_DIM,
        LANES,
        SUPPORTED_HEAD_DIM,
        VALUE_DIM,
        _MatrixSlotView,
        _cube_ring,
        _full_token_tile,
        _sync_cube_vector_handoff,
    )
    from flash_kda_grad_metadata import (
        HEADER_WORDS,
        TASK_CHUNKS,
        TASK_COMPACT_TOKEN_START,
        TASK_LENGTH,
        TASK_SEQUENCE,
        TASK_VALUE_HEAD,
        TASK_WORDS,
    )


class ChunkBackward:
    """Run the dependent WY inverse and local-gradient stages per chunk."""

    def __init__(
        self,
        fuse_base=False,
        resident_dr=None,
        resident_dw=None,
        adjacent_decay=False,
        cube_causal=False,
        round_raw_gate=False,
    ):
        self.vector = ChunkGradientVector(
            batch_rows=True,
            fuse_base=fuse_base,
            resident_dr=resident_dr,
            resident_dw=resident_dw,
            direct_results=cube_causal,
            round_raw_gate=round_raw_gate,
        )
        self.cube_causal = cube_causal
        self.matmul = ChunkGradientMatmul(
            self.vector.causal_results,
            self.vector.causal_l1,
            self.vector.stacked_matrix,
        )
        self.vector.cache_causal_inputs = cube_causal
        if cube_causal:
            self.inverse_mask = InverseMaskVector(self.vector)
            self.causal_operands = CausalOperandVector(False, self.vector)
            self.causal_result = CausalResultVector(False, self.vector)
        self.wy_operands = WyOperandVector(self.vector)
        self.wy_result = WyResultVector(self.vector)
        self.base_gradient = BaseGradientVector(self.vector)
        self.value_beta = ValueBetaEpilogueVector(self.vector)
        self.beta_combine = BetaCombineVector(self.vector)
        self.direct_delta = DirectCausalDeltaVector(self.vector, adjacent_decay)

    @jit
    def process_chunk(
        self,
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
        chunk,
        valid_rows,
        diagonal_ready=0,
    ):
        # These two contractions consume only prepared StageTwo gradients
        # and Akk. They can execute while AIV constructs weighted V/K.
        inverse = tile_slice(gm_inverse, (CHUNK_SIZE, CHUNK_SIZE), (chunk, 0))
        self.matmul.compute_weighted_beta_gradient(
            inverse,
            tile_slice(gm_d_v_new_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk, 0)),
            tile_slice(gm_d_beta_v, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk, 0)),
        )
        self.matmul.compute_weighted_beta_gradient(
            inverse,
            tile_slice(gm_d_w_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk, 0)),
            tile_slice(gm_d_beta_k, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk, 0)),
        )
        self.wy_operands.process_chunk(
            gm_values, gm_keys, gm_gamma, gm_beta, gm_weighted_v, gm_weighted_k, chunk
        )
        _sync_cube_vector_handoff(0, reset_sync_id=True)
        self.matmul.compute_inverse_chain(
            inverse,
            tile_slice(gm_d_v_new_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk, 0)),
            tile_slice(gm_d_w_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk, 0)),
            tile_slice(gm_weighted_v, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk, 0)),
            tile_slice(gm_weighted_k, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (chunk, 0)),
            tile_slice(gm_d_lower_unmasked, (CHUNK_SIZE, CHUNK_SIZE), (chunk, 0)),
        )
        self.wy_result.process_chunk(
            gm_d_beta_v,
            gm_d_beta_k,
            gm_values,
            gm_keys,
            gm_gamma,
            gm_d_beta_wy,
            chunk,
            diagonal_ready,
        )
        if const_expr(not self.vector.fuse_base):
            self.base_gradient.process_chunk(
                gm_keys,
                gm_qg,
                gm_kg,
                gm_gamma,
                gm_beta,
                gm_d_beta_k,
                gm_d_qg,
                gm_d_kg,
                gm_d_gate_last,
                gm_d_query_base,
                gm_d_key_base,
                gm_d_gate_base,
                chunk,
                valid_rows,
            )
        _sync_cube_vector_handoff(1, reset_sync_id=True)
        if const_expr(self.cube_causal):
            # Preserve the validated full-chunk pipeline and resource ownership.
            # The legacy GM-only arguments are unused in the resident specialization.
            self.inverse_mask.process_chunk(
                gm_d_lower_unmasked, None, chunk, valid_rows
            )
            self.vector.load_causal_cache(gm_query, gm_keys, gm_gamma, chunk)
            _sync_cube_vector_handoff(3, reset_sync_id=True)
            self.causal_operands.process_chunk_pair(
                gm_query,  # gm_query
                gm_keys,  # gm_key
                gm_gamma,  # gm_gamma
                gm_beta,  # gm_beta
                gm_d_Aqk,  # gm_d_Aqk
                None,  # gm_d_lower
                None,  # gm_query_exp
                None,  # gm_key_exp
                None,  # gm_key_inverse_exp
                None,  # gm_query_exp_low
                None,  # gm_key_exp_low
                None,  # gm_key_inverse_exp_low
                None,  # gm_d_output_matrix
                None,  # gm_d_inverse_matrix
                None,  # gm_d_inverse_unweighted
                scale,  # scale
                chunk,  # chunk
                valid_rows,  # valid
                0,  # block_start
                CHUNK_SIZE,  # block_end
                0,  # midpoint
            )
            _sync_cube_vector_handoff(1, reset_sync_id=True)
            self.matmul.compute_causal(
                None,  # gm_output_matrix
                None,  # gm_inverse_matrix
                None,  # gm_inverse_unweighted
                None,  # gm_query_exp
                None,  # gm_key_exp
                None,  # gm_key_inverse_exp
                None,  # gm_d_query_exp
                None,  # gm_d_key_row_exp
                None,  # gm_d_key_col_output
                None,  # gm_d_key_col_inverse
                None,  # gm_d_beta_matrix
                None,  # gm_query_exp_low
                None,  # gm_key_exp_low
                None,  # gm_key_inverse_exp_low
            )
            # Retire FIXPIPE before shared channels are reused by the new schedule.
            _sync_cube_vector_handoff(2, reset_sync_id=True)
            self.value_beta.process_chunk(
                gm_beta, gm_d_beta_v, gm_d_value, chunk, valid_rows
            )
            if const_expr(self.vector.fuse_base):
                self.vector.prefetch_fused_base(
                    gm_qg, gm_kg, gm_gamma, gm_d_beta_k, gm_d_kg, gm_d_gate_last, chunk
                )
            self.causal_result.process_chunk(
                gm_query,  # gm_query
                gm_keys,  # gm_key
                gm_gamma,  # gm_gamma
                None,  # gm_d_query_exp
                None,  # gm_d_key_row_exp
                None,  # gm_d_key_col_output
                None,  # gm_d_key_col_inverse
                None,  # gm_d_beta_matrix
                gm_d_query_pair,  # gm_d_query
                gm_d_key_pair,  # gm_d_key
                gm_d_gate_pair,  # gm_d_gate
                gm_d_beta_pair,  # gm_d_beta
                chunk,  # chunk
                valid_rows,  # valid
                0,  # block_start
                CHUNK_SIZE,  # block_end
                0,  # midpoint
                True,  # retain_head
            )
            _sync_cube_vector_handoff(3, reset_sync_id=True)
        else:
            # Cube completion is already published. Mask and round coefficients
            # at their consumer instead of writing an intermediate through GM.
            self.direct_delta.process_chunk(
                gm_query,
                gm_keys,
                gm_gamma,
                gm_beta,
                gm_d_Aqk,
                gm_d_lower_unmasked,
                gm_d_query_pair,
                gm_d_key_pair,
                gm_d_gate_pair,
                gm_d_beta_pair,
                gm_d_beta_matrix,
                scale,
                chunk,
                valid_rows,
            )
            self.value_beta.process_chunk(
                gm_beta, gm_d_beta_v, gm_d_value, chunk, valid_rows
            )
            if const_expr(self.vector.fuse_base):
                self.vector.prefetch_fused_base(
                    gm_qg, gm_kg, gm_gamma, gm_d_beta_k, gm_d_kg, gm_d_gate_last, chunk
                )
            if const_expr(self.vector.fuse_base):
                # Direct's beta handoff already published both feature owners'
                # gradients and retired every Cube consumer. The full epilogue
                # only needs this AIV's local writes before its beta readback;
                # the final chunk handoff still protects shared-UB reuse.
                vec_sync_all()
                onchip_rewind(reset_sync_id=True)
            else:
                _sync_cube_vector_handoff(3, reset_sync_id=True)
        if get_subblock_id() == 0:
            # Both AIVs own disjoint token rows. Merge each row's WY and
            # causal terms once, then write in the requested gradient dtype.
            self.beta_combine.process_chunk_single_pair(
                gm_beta,
                gm_d_beta_wy,
                gm_d_beta_pair,
                gm_d_beta,
                chunk,
                valid_rows,
                fused_beta,
                inverse_factor,
            )
        if const_expr(not self.vector.fuse_base):
            _sync_cube_vector_handoff(0, reset_sync_id=True)


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
    round_raw_gate: Constexpr[bool] = False,
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
    process = ChunkBackward(
        fuse_base=aligned_only,
        resident_dr=resident_dr,
        resident_dw=resident_dw,
        adjacent_decay=aligned_only,
        cube_causal=cube_causal,
        round_raw_gate=round_raw_gate,
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
def _finish_masked_track_chunk(
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
    heads_per_key_head: dtypes.int64,
    fused_gate: dtypes.int64,
    has_bias: dtypes.int64,
    use_l2norm: dtypes.int64,
    softplus: dtypes.int64,
    lower_bound: dtypes.float32,
    beta_fp32: Constexpr[bool],
    padded_layout: dtypes.int64,
    recompute_single_chunk_alog: dtypes.int64,
    selected_task: dtypes.int64,
    selected_chunk: dtypes.int64,
    staged: Constexpr[bool] = False,
    round_raw_gate: Constexpr[bool] = False,
):
    process = ChunkBackward(round_raw_gate=round_raw_gate)
    vector = process.vector
    block_idx = get_block_idx()
    task = selected_task
    task_record = HEADER_WORDS + task * TASK_WORDS
    gradient_slot = _gradient_slot(gm_task_metadata, task, selected_chunk, staged)
    seq_idx = gm_task_metadata[task_record + TASK_SEQUENCE]
    compact_token_start = gm_task_metadata[task_record + TASK_COMPACT_TOKEN_START]
    chunks = gm_task_metadata[task_record + TASK_CHUNKS]
    seq_len = gm_task_metadata[task_record + TASK_LENGTH]
    value_head = gm_task_metadata[task_record + TASK_VALUE_HEAD]
    raw_batch = 0
    raw_token_start = compact_token_start
    if padded_layout:
        raw_batch = seq_idx
        raw_token_start = 0
    qk_head = value_head // heads_per_key_head
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
    chunk = selected_chunk
    if fused_gate:
        if chunk < chunks:
            vector.begin_gate_task()
    if gm_inverse.shape[0] > 0:
        if chunk < chunks:
            # Preserve this head's contributions until the group is complete.
            scratch_chunk = (
                block_idx * heads_per_key_head + value_head % heads_per_key_head
            )
            valid_rows = seq_len - chunk * CHUNK_SIZE
            if valid_rows > CHUNK_SIZE:
                valid_rows = CHUNK_SIZE
            raw_row_start = raw_token_start + chunk * CHUNK_SIZE
            compact_row_start = gradient_slot * CHUNK_SIZE
            inverse_batch = dtypes.int64(raw_batch)
            inverse_row_start = raw_row_start
            inverse_source = gm_Akk[inverse_batch, value_head, None, None]
            d_Aqk_source = gm_stage_two_d_Aqk[gradient_slot, 0, None, None]
            if get_subblock_id() == 0:
                vector.load_fp32_row(
                    d_gate_last_source,
                    gradient_slot,
                    gm_d_gate_last,
                    scratch_chunk,
                    True,
                    SUPPORTED_HEAD_DIM,
                    6,
                )
            diagonal_ready = 0
            half_start = get_subblock_id() * (CHUNK_SIZE // 2)
            destination_row = scratch_chunk * CHUNK_SIZE + half_start
            vector.load_full_chunk_rows(
                query_source,
                raw_row_start + half_start,
                gm_query,
                destination_row,
                SUPPORTED_HEAD_DIM,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                key_source,
                raw_row_start + half_start,
                gm_keys,
                destination_row,
                SUPPORTED_HEAD_DIM,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                value_source,
                raw_row_start + half_start,
                gm_values,
                destination_row,
                SUPPORTED_HEAD_DIM,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                inverse_source,
                inverse_row_start + half_start,
                gm_inverse,
                destination_row,
                CHUNK_SIZE,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                qg_source,
                raw_row_start + half_start,
                gm_qg,
                destination_row,
                SUPPORTED_HEAD_DIM,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                kg_source,
                raw_row_start + half_start,
                gm_kg,
                destination_row,
                SUPPORTED_HEAD_DIM,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                gamma_source,
                raw_row_start + half_start,
                gm_gamma,
                destination_row,
                SUPPORTED_HEAD_DIM,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                d_v_new_source,
                compact_row_start + half_start,
                gm_d_v_new_low,
                destination_row,
                SUPPORTED_HEAD_DIM,
                True,
                0,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                d_w_source,
                compact_row_start + half_start,
                gm_d_w_low,
                destination_row,
                SUPPORTED_HEAD_DIM,
                True,
                1,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                d_qg_source,
                compact_row_start + half_start,
                gm_d_qg,
                destination_row,
                SUPPORTED_HEAD_DIM,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                d_kg_source,
                compact_row_start + half_start,
                gm_d_kg,
                destination_row,
                SUPPORTED_HEAD_DIM,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            vector.load_full_chunk_rows(
                d_Aqk_source,
                half_start,
                gm_d_Aqk,
                destination_row,
                CHUNK_SIZE,
                valid_rows=max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            if const_expr(vector.correct_wy_diagonal):
                if valid_rows == CHUNK_SIZE:
                    diagonal_ready = 1
            if get_subblock_id() == 0:
                if valid_rows == CHUNK_SIZE:
                    if beta_source.physical_layout.stride[-2] == 1:
                        vector.load_contiguous_beta_chunk(
                            beta_source, raw_row_start, gm_beta, scratch_chunk
                        )
                    else:
                        vector.load_strided_beta_chunk(
                            beta_source, raw_row_start, gm_beta, scratch_chunk
                        )
                else:
                    vector.load_beta_window(
                        beta_source, raw_row_start, gm_beta, scratch_chunk, valid_rows
                    )
            _sync_cube_vector_handoff(0, reset_sync_id=True)
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
            _sync_cube_vector_handoff(1, reset_sync_id=True)
            vector.begin_gate_chunk()
            alog = 0.0
            if fused_gate:
                alog = gm_A_log[value_head]
            first_half = get_subblock_id()
            half_stride = get_subblock_dim()
            for half in range(first_half, 2, half_stride):
                vector.begin_gate_half()
                gate_rows = valid_rows
                if fused_gate:
                    if valid_rows == CHUNK_SIZE:
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
                            softplus,
                            lower_bound,
                        )
                        gate_rows = 0
                    else:
                        vector.gate_padded_chunk(
                            gm_d_gate_base,
                            gm_d_gate_pair,
                            raw_gate_source,
                            gm_dt_bias,
                            gate_destination,
                            scratch_chunk,
                            raw_row_start,
                            valid_rows,
                            half,
                            value_head,
                            alog,
                            has_bias,
                            softplus,
                            lower_bound,
                        )
                        gate_rows = 0
                for reverse_row in range(gate_rows):
                    row = valid_rows - reverse_row - 1
                    vector.gate_row(
                        gm_d_gate_base,
                        gm_d_gate_pair,
                        raw_gate_source,
                        gm_dt_bias,
                        gate_destination,
                        scratch_chunk * CHUNK_SIZE + row,
                        raw_row_start + row,
                        half,
                        value_head,
                        alog,
                        fused_gate,
                        has_bias,
                        softplus,
                        lower_bound,
                    )
                if fused_gate:
                    vector.accumulate_gate_chunk(half)
            half_start = get_subblock_id() * (CHUNK_SIZE // 2)
            vector.store_value_window(
                gm_d_value,
                scratch_chunk * CHUNK_SIZE + half_start,
                value_destination,
                raw_row_start + half_start,
                max(0, min(CHUNK_SIZE // 2, valid_rows - half_start)),
            )
            if get_subblock_id() == 0:
                beta_output_rows = valid_rows
                if recompute_single_chunk_alog:
                    # The FP32 helper owns valid beta rows for this
                    # task. Its block assignment differs from Stage 1.
                    if chunks == 1:
                        beta_output_rows = 0
                if (
                    beta_output_rows == CHUNK_SIZE
                    and beta_destination.physical_layout.stride[-2] == 1
                ):
                    vector.store_contiguous_beta_chunk(
                        gm_d_beta, scratch_chunk, beta_destination, raw_row_start
                    )
                    beta_output_rows = 0
                if beta_output_rows > 0:
                    vector.store_beta_window(
                        gm_d_beta,
                        scratch_chunk,
                        beta_destination,
                        raw_row_start,
                        beta_output_rows,
                    )
            if heads_per_key_head == 1:
                # This value head is the only contributor to its Q/K head.
                # Consume the just-produced rows before the next chunk.
                head_output_rows = valid_rows
                head_output_start = 0
                if valid_rows == CHUNK_SIZE:
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
                    )
                    head_output_rows = 0
                elif valid_rows >= vector.head_batch_rows:
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
                        valid_rows=valid_rows,
                    )
                    head_output_start = (
                        valid_rows // vector.head_batch_rows
                    ) * vector.head_batch_rows
                for row in range(
                    head_output_start + get_subblock_id(),
                    head_output_rows,
                    get_subblock_dim(),
                ):
                    scratch_row = scratch_chunk * CHUNK_SIZE + row
                    vector.begin_head_row()
                    vector.add_head_row(
                        gm_d_query_base,
                        gm_d_query_pair,
                        gm_d_key_base,
                        gm_d_key_pair,
                        scratch_row,
                    )
                    vector.finish_head_row(
                        gm_effective_query,
                        gm_effective_key,
                        gm_q_rstd,
                        gm_k_rstd,
                        gm_public_d_query,
                        gm_public_d_key,
                        raw_batch,
                        qk_head,
                        raw_row_start + row,
                        raw_batch,
                        raw_row_start + row,
                        use_l2norm,
                    )
            _sync_cube_vector_handoff(2, reset_sync_id=True)
    if fused_gate:
        if chunk < chunks:
            if recompute_single_chunk_alog:
                # Short sequences replace only dA by the FP32 chunk JVP.
                # Other sequences keep contributing through this atomic path.
                if chunks == 1:
                    vector.finish_gate_task_atomic(
                        gm_public_dA, gm_public_dBias, value_head, has_bias, False
                    )
                else:
                    vector.finish_gate_task_atomic(
                        gm_public_dA, gm_public_dBias, value_head, has_bias
                    )
            else:
                vector.finish_gate_task_atomic(
                    gm_public_dA, gm_public_dBias, value_head, has_bias
                )

    _sync_cube_vector_handoff(1, reset_sync_id=True)
    # Earlier value heads in this group left FP32 Q/K contributions in
    # this core's slots. Only the last head normalizes and writes the sum.
    if (
        gm_effective_query.shape[2] > 0
        and heads_per_key_head > 1
        and value_head % heads_per_key_head == heads_per_key_head - 1
    ):
        chunk_start = selected_chunk * CHUNK_SIZE
        valid = min(CHUNK_SIZE, max(seq_len - chunk_start, 0))
        for row in range(get_subblock_id(), valid, get_subblock_dim()):
            vector.begin_head_row()
            for local_head in range(heads_per_key_head):
                scratch_row = (
                    block_idx * heads_per_key_head + local_head
                ) * CHUNK_SIZE + row
                vector.add_head_row(
                    gm_d_query_base,
                    gm_d_query_pair,
                    gm_d_key_base,
                    gm_d_key_pair,
                    scratch_row,
                )
            vector.finish_head_row(
                gm_effective_query,
                gm_effective_key,
                gm_q_rstd,
                gm_k_rstd,
                gm_public_d_query,
                gm_public_d_key,
                raw_batch,
                qk_head,
                raw_token_start + chunk_start + row,
                raw_batch,
                raw_token_start + chunk_start + row,
                use_l2norm,
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
            heads_per_key_head,
            seq_len,
        )
    _sync_cube_vector_handoff(2, reset_sync_id=True)


class ChunkGradientMatmul:
    """Cube contractions for WY, inverse, and causal gradients."""

    def __init__(self, causal_results=None, causal_l1=None, stacked_matrix=None):
        self.stacked_matrix = stacked_matrix
        self.use_stacked = stacked_matrix is not None
        self.causal_l1 = causal_l1
        self.direct_results = causal_results is not None
        self.causal_results = causal_results
        # Each contraction consumes its L0 inputs before loading the next pair.
        self._operand_a_storage = _cube_ring(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((2 * CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        )
        self._operand_b_storage = _cube_ring(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout((2 * CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        )
        if self.use_stacked:
            self.stacked_a = _MatrixSlotView(
                self._operand_a_storage, (2 * CHUNK_SIZE, CHUNK_SIZE)
            )
        # WY, inverse and causal products consume each result sequentially.
        self.shared_c = _cube_ring(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((2 * CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        )
        # Stage 1 runs after Stage 2 in the fused kernel. These addresses
        # reuse its L0 range and stay disjoint within Stage 1.
        self.inverse_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        )
        self.gradient_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        )
        self.inverse_l0a = _MatrixSlotView(
            self._operand_a_storage, (CHUNK_SIZE, CHUNK_SIZE)
        )
        self.gradient_l0b = _MatrixSlotView(
            self._operand_b_storage, (SUPPORTED_HEAD_DIM, CHUNK_SIZE)
        )
        self.gradient_l0c = (
            self.shared_c
            if self.use_stacked
            else _MatrixSlotView(self.shared_c, (CHUNK_SIZE, SUPPORTED_HEAD_DIM))
        )
        self.pair_left_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        )
        self.pair_right_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        )
        self.pair_left_l0a = _MatrixSlotView(
            self._operand_a_storage, (CHUNK_SIZE, SUPPORTED_HEAD_DIM)
        )
        self.pair_right_l0b = _MatrixSlotView(
            self._operand_b_storage, (CHUNK_SIZE, SUPPORTED_HEAD_DIM)
        )
        self.pair_l0c = (
            self.shared_c
            if self.use_stacked
            else _MatrixSlotView(self.shared_c, (CHUNK_SIZE, CHUNK_SIZE))
        )

        self.left_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        )
        self.right_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        )
        self.left_l0a = _MatrixSlotView(
            self._operand_a_storage, (CHUNK_SIZE, CHUNK_SIZE)
        )
        self.right_l0b = _MatrixSlotView(
            self._operand_b_storage, (CHUNK_SIZE, CHUNK_SIZE)
        )
        self.inverse_result_l0c = (
            self.shared_c
            if self.use_stacked
            else _MatrixSlotView(self.shared_c, (CHUNK_SIZE, CHUNK_SIZE))
        )
        self.matrix_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, CHUNK_SIZE)), depth=1
        )
        self.feature_l1 = RingBuffer(
            MemLoc.L1,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            depth=1,
        )
        self.matrix_l0a = _MatrixSlotView(
            self._operand_a_storage, (CHUNK_SIZE, CHUNK_SIZE)
        )
        self.feature_l0b = _MatrixSlotView(
            self._operand_b_storage, (SUPPORTED_HEAD_DIM, CHUNK_SIZE)
        )
        self.causal_result_l0c = (
            self.shared_c
            if self.use_stacked
            else _MatrixSlotView(self.shared_c, (CHUNK_SIZE, SUPPORTED_HEAD_DIM))
        )

    def compute_weighted_beta_gradient(self, gm_inverse, gm_gradient, gm_result):
        mem_copy(self.inverse_l1.next_write(), gm_inverse)
        mem_copy(self.gradient_l1.next_write(), gm_gradient)
        mem_copy(
            self.inverse_l0a.next_write(), self.inverse_l1.next_read().permute((1, 0))
        )
        mem_copy(
            self.gradient_l0b.next_write(),
            self.gradient_l1.next_read().permute((1, 0)),
        )
        matmul(
            (
                from_tensor(
                    self.gradient_l0c.next_write(),
                    data_format="nz",
                    layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                )
                if self.use_stacked
                else self.gradient_l0c.next_write()
            ),
            self.inverse_l0a.next_read(),
            self.gradient_l0b.next_read(),
            init=True,
        )
        mem_copy(
            gm_result,
            (
                from_tensor(
                    self.gradient_l0c.next_read(),
                    data_format="nz",
                    layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                )
                if self.use_stacked
                else self.gradient_l0c.next_read()
            ),
        )

    def compute_inverse_chain(
        self,
        gm_inverse,
        gm_d_v_new_low,
        gm_d_w_low,
        gm_weighted_v,
        gm_weighted_k,
        gm_d_lower_unmasked,
    ):
        # FIXPIPE rounds each intermediate to BF16 directly in L1. Channel
        # dependencies order FIXPIPE -> MTE1 without GM or a Vector round-trip.
        pair_sum = (
            from_tensor(
                self.pair_l0c.next_write(),
                data_format="nz",
                layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
            )
            if self.use_stacked
            else self.pair_l0c.next_write()
        )
        mem_copy(self.pair_left_l1.next_write(), gm_d_v_new_low)
        mem_copy(self.pair_right_l1.next_write(), gm_weighted_v)
        mem_copy(self.pair_left_l0a.next_write(), self.pair_left_l1.next_read())
        mem_copy(self.pair_right_l0b.next_write(), self.pair_right_l1.next_read())
        matmul(
            pair_sum,
            self.pair_left_l0a.next_read(),
            self.pair_right_l0b.next_read(),
            init=True,
        )
        mem_copy(self.pair_left_l1.next_write(), gm_d_w_low)
        mem_copy(self.pair_right_l1.next_write(), gm_weighted_k)
        mem_copy(self.pair_left_l0a.next_write(), self.pair_left_l1.next_read())
        mem_copy(self.pair_right_l0b.next_write(), self.pair_right_l1.next_read())
        matmul(
            pair_sum,
            self.pair_left_l0a.next_read(),
            self.pair_right_l0b.next_read(),
            init=False,
        )
        mem_copy(
            self.right_l1.next_write(),
            (
                from_tensor(
                    self.pair_l0c.next_read(),
                    data_format="nz",
                    layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
                )
                if self.use_stacked
                else self.pair_l0c.next_read()
            ),
        )
        mem_copy(self.left_l1.next_write(), gm_inverse)
        mem_copy(self.left_l0a.next_write(), self.left_l1.next_read().permute((1, 0)))
        mem_copy(self.right_l0b.next_write(), self.right_l1.next_read().permute((1, 0)))
        matmul(
            (
                from_tensor(
                    self.inverse_result_l0c.next_write(),
                    data_format="nz",
                    layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
                )
                if self.use_stacked
                else self.inverse_result_l0c.next_write()
            ),
            self.left_l0a.next_read(),
            self.right_l0b.next_read(),
            init=True,
        )
        mem_copy(
            self.left_l1.next_write(),
            (
                from_tensor(
                    self.inverse_result_l0c.next_read(),
                    data_format="nz",
                    layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
                )
                if self.use_stacked
                else self.inverse_result_l0c.next_read()
            ),
        )
        mem_copy(self.right_l1.next_write(), gm_inverse)
        mem_copy(self.left_l0a.next_write(), self.left_l1.next_read())
        mem_copy(self.right_l0b.next_write(), self.right_l1.next_read())
        matmul(
            (
                from_tensor(
                    self.inverse_result_l0c.next_write(),
                    data_format="nz",
                    layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
                )
                if self.use_stacked
                else self.inverse_result_l0c.next_write()
            ),
            self.left_l0a.next_read(),
            self.right_l0b.next_read(),
            init=True,
        )
        if const_expr(self.direct_results):
            mem_copy(
                from_tensor(
                    self.causal_results[2],
                    layout=make_layout((CHUNK_SIZE // 2, CHUNK_SIZE)),
                ),
                from_tensor(
                    from_tensor(
                        self.inverse_result_l0c.next_read(),
                        data_format="nz",
                        layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
                    )
                    if self.use_stacked
                    else self.inverse_result_l0c.next_read(),
                    layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
                ).partition(dim=0),
            )
        else:
            mem_copy(
                gm_d_lower_unmasked,
                (
                    from_tensor(
                        self.inverse_result_l0c.next_read(),
                        data_format="nz",
                        layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
                    )
                    if self.use_stacked
                    else self.inverse_result_l0c.next_read()
                ),
            )

    def _load_causal_operands(self, gm_matrix, gm_feature, transpose_matrix: bool):
        if const_expr(self.direct_results):
            mem_copy(
                self.matrix_l0a.next_write(),
                gm_matrix.permute((1, 0)) if transpose_matrix else gm_matrix,
            )
            mem_copy(self.feature_l0b.next_write(), gm_feature.permute((1, 0)))
        else:
            mem_copy(self.matrix_l1.next_write(), gm_matrix)
            mem_copy(self.feature_l1.next_write(), gm_feature)
            mem_copy(
                self.matrix_l0a.next_write(),
                self.matrix_l1.next_read().permute((1, 0))
                if transpose_matrix
                else self.matrix_l1.next_read(),
            )
            mem_copy(
                self.feature_l0b.next_write(),
                self.feature_l1.next_read().permute((1, 0)),
            )

    def _accumulate_causal_feature_split(
        self,
        gm_matrix,
        gm_feature_high,
        gm_feature_low,
        result,
        transpose_matrix: bool,
        initialize: bool,
    ):
        # Both products use the identical coefficient matrix. Keep its L0A
        # view alive through high then low; only the feature operand changes.
        # No extra storage or change to the FP32 accumulation order is needed.
        self._load_causal_operands(gm_matrix, gm_feature_high, transpose_matrix)
        matrix = self.matrix_l0a.next_read()
        matmul(result, matrix, self.feature_l0b.next_read(), init=initialize)
        if const_expr(self.direct_results):
            mem_copy(self.feature_l0b.next_write(), gm_feature_low.permute((1, 0)))
        else:
            mem_copy(self.feature_l1.next_write(), gm_feature_low)
            mem_copy(
                self.feature_l0b.next_write(),
                self.feature_l1.next_read().permute((1, 0)),
            )
        matmul(result, matrix, self.feature_l0b.next_read(), init=False)

    def _compute_causal_feature_split(
        self,
        gm_matrix,
        gm_feature_high,
        gm_feature_low,
        gm_output,
        transpose_matrix: bool,
        output_plane: Constexpr[int] = 0,
    ):
        result = (
            from_tensor(
                self.causal_result_l0c.next_write(),
                data_format="nz",
                layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            )
            if self.use_stacked
            else self.causal_result_l0c.next_write()
        )
        self._accumulate_causal_feature_split(
            gm_matrix, gm_feature_high, gm_feature_low, result, transpose_matrix, True
        )
        if const_expr(self.direct_results):
            mem_copy(
                self.causal_results[output_plane],
                from_tensor(
                    from_tensor(
                        self.causal_result_l0c.next_read(),
                        data_format="nz",
                        layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                    )
                    if self.use_stacked
                    else self.causal_result_l0c.next_read(),
                    layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                ).partition(dim=0),
            )
        else:
            mem_copy(
                gm_output,
                (
                    from_tensor(
                        self.causal_result_l0c.next_read(),
                        data_format="nz",
                        layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                    )
                    if self.use_stacked
                    else self.causal_result_l0c.next_read()
                ),
            )

    def compute_causal(
        self,
        gm_output_matrix,
        gm_inverse_matrix,
        gm_inverse_unweighted,
        gm_query_exp,
        gm_key_exp,
        gm_key_inverse_exp,
        gm_d_query_exp,
        gm_d_key_row_exp,
        gm_d_key_col_output,
        gm_d_key_col_inverse,
        gm_d_beta_matrix,
        gm_query_exp_low,
        gm_key_exp_low,
        gm_key_inverse_exp_low,
    ):
        if const_expr(self.direct_results):
            gm_query_exp = self.causal_l1[0]
            gm_key_exp = self.causal_l1[1]
            gm_key_inverse_exp = self.causal_l1[2]
            gm_query_exp_low = self.causal_l1[3]
            gm_key_exp_low = self.causal_l1[4]
            gm_key_inverse_exp_low = self.causal_l1[5]
            gm_output_matrix = self.causal_l1[6]
            gm_inverse_unweighted = self.causal_l1[7]
        # The shared row path uses beta * BF16(dL). Use that same dL
        # for the transposed path, with beta carried by the split feature.
        # Independently rounding beta*dL breaks the row/column cancellation
        # of gate derivatives, which the reverse scan amplifies in dA.
        column_inverse = gm_inverse_matrix
        column_inverse = gm_inverse_unweighted
        if const_expr(self.use_stacked):
            self._stacked_query_beta(gm_key_inverse_exp, gm_key_inverse_exp_low)
        else:
            self._compute_causal_feature_split(
                gm_output_matrix,
                gm_key_inverse_exp,
                gm_key_inverse_exp_low,
                gm_d_query_exp,
                False,
            )
        column_sum = (
            from_tensor(
                self.causal_result_l0c.next_write(),
                data_format="nz",
                layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            )
            if self.use_stacked
            else self.causal_result_l0c.next_write()
        )
        self._accumulate_causal_feature_split(
            gm_output_matrix, gm_query_exp, gm_query_exp_low, column_sum, True, True
        )
        self._accumulate_causal_feature_split(
            column_inverse, gm_key_exp, gm_key_exp_low, column_sum, True, False
        )
        if const_expr(self.direct_results):
            mem_copy(
                self.causal_results[1],
                from_tensor(
                    from_tensor(
                        self.causal_result_l0c.next_read(),
                        data_format="nz",
                        layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                    )
                    if self.use_stacked
                    else self.causal_result_l0c.next_read(),
                    layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                ).partition(dim=0),
            )
        else:
            mem_copy(
                gm_d_key_col_output,
                (
                    from_tensor(
                        self.causal_result_l0c.next_read(),
                        data_format="nz",
                        layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                    )
                    if self.use_stacked
                    else self.causal_result_l0c.next_read()
                ),
            )
        # T = L @ K_inverse serves both beta and the row-key gradient.
        # The vector epilogue multiplies T by beta; no division by beta is
        # needed, including exact zero or saturated-sigmoid beta values.
        if const_expr(not self.use_stacked):
            self._compute_causal_feature_split(
                gm_inverse_unweighted,
                gm_key_inverse_exp,
                gm_key_inverse_exp_low,
                gm_d_beta_matrix,
                False,
                2,
            )

    def _stacked_query_beta(self, feature_high, feature_low):
        mem_copy(self.stacked_a.next_write(), self.stacked_matrix)
        a = self.stacked_a.next_read()
        mem_copy(self.feature_l0b.next_write(), feature_high.permute((1, 0)))
        result = self.shared_c.next_write()
        matmul(result, a, self.feature_l0b.next_read(), init=True)
        mem_copy(self.feature_l0b.next_write(), feature_low.permute((1, 0)))
        matmul(result, a, self.feature_l0b.next_read(), init=False)
        result_read = self.shared_c.next_read()
        # Each half keeps the 128-row NZ panel stride of the stacked result.
        # Reinterpreting it as a compact 64-row matrix would change that stride.
        mem_copy(
            self.causal_results[0],
            tile_slice(result_read, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (0, 0)).partition(
                dim=0
            ),
        )
        mem_copy(
            self.causal_results[2],
            tile_slice(result_read, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (1, 0)).partition(
                dim=0
            ),
        )


class WyOperandVector:
    """Form beta-weighted BF16 operands for the four WY Cube products."""

    def __init__(self, shared: ChunkWorkspace):
        self.batch_values, self.batch_keys = shared.batch_bf16
        self.batch_gamma = shared.batch_fp32[0]
        self.beta = shared.beta_values

    @jit
    def process_chunk(
        self,
        gm_values,
        gm_keys,
        gm_gamma,
        gm_beta,
        gm_weighted_values,
        gm_weighted_keys,
        chunk,
    ):
        shape = (CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)
        row_tile = chunk * 2 + get_subblock_id()
        mem_copy(self.beta.next_write(), tile_slice(gm_beta, (CHUNK_SIZE,), (chunk,)))
        mem_copy(self.batch_values, tile_slice(gm_values, shape, (row_tile, 0)))
        mem_copy(self.batch_keys, tile_slice(gm_keys, shape, (row_tile, 0)))
        mem_copy(self.batch_gamma, tile_slice(gm_gamma, shape, (row_tile, 0)))
        self._weight_chunk(
            self.batch_values,
            self.batch_keys,
            self.batch_gamma,
            self.beta.next_read(),
            get_subblock_id() * (CHUNK_SIZE // 2),
        )
        mem_copy(
            tile_slice(gm_weighted_values, shape, (row_tile, 0)), self.batch_values
        )
        mem_copy(tile_slice(gm_weighted_keys, shape, (row_tile, 0)), self.batch_keys)

    @jit
    def _weight_chunk(self, values, keys, gamma, beta, token_start):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for row in range(CHUNK_SIZE // 2):
                factor = vload_broadcast(beta, token_start + row)
                for half in range(2):
                    offset = row * SUPPORTED_HEAD_DIM + half * LANES
                    value = vcast(
                        vload_unpack(values, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    key = vcast(
                        vload_unpack(keys, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    gate = vexp(vload(gamma, offset), mask=full)
                    value_product = vmul(factor, value, mask=full)
                    key_product = vmul(factor, vmul(gate, key, mask=full), mask=full)
                    vstore_pack(
                        values,
                        offset,
                        vcast(value_product, dtypes.bfloat16, mask=full),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                    vstore_pack(
                        keys,
                        offset,
                        vcast(key_product, dtypes.bfloat16, mask=full),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
            vmem_bar("vst_vld")


class WyResultVector:
    """Reduce the two beta-gradient branches after the WY Cube products."""

    def __init__(self, shared: ChunkWorkspace):
        self.diagonal_value, self.diagonal_key = shared.wy_diagonal_residual
        self.block_fp32 = shared.batch_fp32
        self.block_bf16 = shared.batch_bf16
        self.block_beta = shared.batch_scalar

    @jit
    def process_chunk(
        self,
        gm_d_value,
        gm_d_key,
        gm_values,
        gm_keys,
        gm_gamma,
        gm_d_beta,
        chunk,
        diagonal_ready=0,
    ):
        """Reduce WY beta rows, optionally restoring the FP32 identity contribution."""
        half_rows = CHUNK_SIZE // 2
        row_tile = chunk * 2 + get_subblock_id()
        shape = (half_rows, SUPPORTED_HEAD_DIM)
        d_value, d_key, gamma = self.block_fp32
        values, keys = self.block_bf16
        mem_copy(d_value, tile_slice(gm_d_value, shape, (row_tile, 0)))
        mem_copy(d_key, tile_slice(gm_d_key, shape, (row_tile, 0)))
        mem_copy(values, tile_slice(gm_values, shape, (row_tile, 0)))
        mem_copy(keys, tile_slice(gm_keys, shape, (row_tile, 0)))
        mem_copy(gamma, tile_slice(gm_gamma, shape, (row_tile, 0)))
        self._reduce_beta_chunk(
            d_value, d_key, values, keys, gamma, self.block_beta, diagonal_ready
        )
        mem_copy(tile_slice(gm_d_beta, (half_rows,), (row_tile,)), self.block_beta)

    @jit
    def _reduce_beta_chunk(
        self, d_value, d_key, values, keys, gamma, out, diagonal_ready=0
    ):
        # Runtime switches are resolved on the scalar side of the VF boundary.
        if diagonal_ready:
            self._reduce_beta_chunk_constexpr(
                d_value,
                d_key,
                values,
                keys,
                gamma,
                out,
                True,
            )
        else:
            self._reduce_beta_chunk_constexpr(
                d_value,
                d_key,
                values,
                keys,
                gamma,
                out,
                False,
            )

    @jit
    def _reduce_beta_chunk_constexpr(
        self,
        d_value,
        d_key,
        values,
        keys,
        gamma,
        out,
        diagonal_ready: Constexpr[bool] = 0,
    ):
        dv_stride, dk_stride = (
            d_value.physical_layout.stride[0],
            d_key.physical_layout.stride[0],
        )
        v_stride, k_stride = (
            values.physical_layout.stride[0],
            keys.physical_layout.stride[0],
        )
        g_stride = gamma.physical_layout.stride[0]
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            one, _ = update_mask(1, elem_bits=32)
            for row in range(CHUNK_SIZE // 2):
                total = vdups(0.0, dtypes.float32, mask=one)
                for half in range(2):
                    offset = half * LANES
                    value = vcast(
                        vload_unpack(
                            values,
                            row * v_stride + offset,
                            unpack_mode=UnpackMode.B16_TO_B32,
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    key = vcast(
                        vload_unpack(
                            keys,
                            row * k_stride + offset,
                            unpack_mode=UnpackMode.B16_TO_B32,
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    gate = vexp(vload(gamma, row * g_stride + offset), mask=full)
                    d_value_row = vload(d_value, row * dv_stride + offset)
                    d_key_row = vload(d_key, row * dk_stride + offset)
                    if const_expr(diagonal_ready != 0):
                        d_value_row = vadd(
                            d_value_row,
                            vload(
                                self.diagonal_value,
                                row * SUPPORTED_HEAD_DIM + offset,
                            ),
                            mask=full,
                        )
                        d_key_row = vadd(
                            d_key_row,
                            vload(self.diagonal_key, row * SUPPORTED_HEAD_DIM + offset),
                            mask=full,
                        )
                    value_term = vmul(d_value_row, value, mask=full)
                    key_term = vmul(d_key_row, vmul(gate, key, mask=full), mask=full)
                    total = vadd(total, vreduce_sum(value_term, mask=full), mask=one)
                    total = vadd(total, vreduce_sum(key_term, mask=full), mask=one)
                vstore_first(out, row, total)
                vmem_bar("vst_vld")
