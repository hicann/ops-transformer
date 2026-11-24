# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Causal operand construction, contractions and local gate-gradient contributions."""

from __future__ import annotations
from cannbotdsl import (
    Buffer,
    Constexpr,
    MemLoc,
    PIPE,
    RoundingMode,
    const_expr,
    dtypes,
    from_tensor,
    make_layout,
)
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
)
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
    vges,
    vload,
    vload_broadcast,
    vload_unpack,
    vlts,
    vmem_bar,
    vmul,
    vmuls,
    vneg,
    vnes,
    vpack,
    vreduce_sum,
    vreinterpret,
    vreinterpret_lanes,
    vselect,
    vstore,
    vstore_first,
    vstore_pack,
    vstore_strided,
    vsub,
)
from cannbotdsl.ops.sync import (
    onchip_rewind,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import tile_slice

if __package__:
    from .flash_kda_grad_buffer import ChunkWorkspace
    from .flash_kda_grad_buffer import (
        BF16_ELEMENT_BYTES,
        CAUSAL_BATCH_ROWS,
        CHUNK_SIZE,
        D_HALF,
        FP32_ELEMENT_BYTES,
        SUPPORTED_HEAD_DIM,
        _sync_cube_vector_handoff,
    )
else:
    from flash_kda_grad_buffer import ChunkWorkspace
    from flash_kda_grad_buffer import (
        BF16_ELEMENT_BYTES,
        CAUSAL_BATCH_ROWS,
        CHUNK_SIZE,
        D_HALF,
        FP32_ELEMENT_BYTES,
        SUPPORTED_HEAD_DIM,
        _sync_cube_vector_handoff,
    )


class InverseMaskVector:
    """Apply the minus sign and strict lower triangle to the Cube product."""

    def __init__(self, shared: ChunkWorkspace | None = None):
        self.batch_rows = shared is not None and shared.batch_rows
        self.direct_result = shared is not None and shared.direct_results
        self.inverse_channel = shared.causal_results[2] if self.direct_result else None
        if self.batch_rows:
            self.block_source, self.block_result = shared.batch_fp32[:2]
            if self.direct_result:
                # WY and base-gradient readers have finished with this tile.
                self.block_result = shared.batch_fp32[2]
        if shared is None:
            self.source = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, CHUNK_SIZE)), depth=1
            )
            self.result = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, CHUNK_SIZE)), depth=1
            )
        else:
            self.source, self.result = shared.fp32_rows[:2]

    @jit
    def run(self, gm_product, gm_lengths, gm_d_lower):
        for chunk in range(get_block_idx(), gm_lengths.shape[0], get_block_num()):
            self.process_chunk(gm_product, gm_d_lower, chunk, gm_lengths[chunk])

    @jit
    def process_chunk(self, gm_product, gm_d_lower, chunk, valid):
        if const_expr(self.batch_rows):
            half_rows = CHUNK_SIZE // 2
            row_tile = chunk * 2 + get_subblock_id()
            source = from_tensor(
                self.block_source, layout=make_layout((half_rows, CHUNK_SIZE))
            )
            result = from_tensor(
                self.block_result, layout=make_layout((half_rows, CHUNK_SIZE))
            )
            if const_expr(self.direct_result):
                source = from_tensor(
                    self.inverse_channel, layout=make_layout((half_rows, CHUNK_SIZE))
                )
            else:
                mem_copy(
                    source,
                    tile_slice(gm_product, (half_rows, CHUNK_SIZE), (row_tile, 0)),
                )
            with vf(mode="simd"):
                full = create_mask(pattern="all")
                zero = vdups(0.0, dtypes.float32, mask=full)
                for local_row in range(half_rows):
                    token = get_subblock_id() * half_rows + local_row
                    value = zero
                    row_valid = vlts(
                        vdups(token, dtypes.int32, mask=full), valid, mask=full
                    )
                    row_valid_value = value
                    causal, _ = update_mask(token, elem_bits=32)
                    value = vselect(
                        vneg(vload(source, local_row * CHUNK_SIZE), mask=full),
                        zero,
                        cond_mask=causal,
                    )
                    value = vselect(value, row_valid_value, cond_mask=row_valid)
                    vstore(result, local_row * CHUNK_SIZE, value, full)
            if const_expr(not self.direct_result):
                mem_copy(
                    tile_slice(gm_d_lower, (half_rows, CHUNK_SIZE), (row_tile, 0)),
                    from_tensor(
                        self.block_result, layout=make_layout((half_rows, CHUNK_SIZE))
                    ),
                )
        else:
            self.process_rows(gm_product, gm_d_lower, chunk, valid)

    @jit
    def process_rows(self, gm_product, gm_d_lower, chunk, valid):
        for row in range(get_subblock_id(), CHUNK_SIZE, get_subblock_dim()):
            index = chunk * CHUNK_SIZE + row
            mem_copy(
                tile_slice(self.source, (1, CHUNK_SIZE), (0, 0)),
                tile_slice(gm_product, (1, CHUNK_SIZE), (index, 0)),
            )
            self._mask(self.source, self.result, row, valid)
            mem_copy(
                tile_slice(gm_d_lower, (1, CHUNK_SIZE), (index, 0)),
                tile_slice(self.result, (1, CHUNK_SIZE), (0, 0)),
            )

    @jit
    def _mask(self, source, result, row, valid):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            value = zero
            row_valid = vlts(vdups(row, dtypes.int32, mask=full), valid, mask=full)
            row_valid_value = value
            causal, _ = update_mask(row, elem_bits=32)
            value = vselect(vneg(vload(source, 0), mask=full), zero, cond_mask=causal)
            value = vselect(value, row_valid_value, cond_mask=row_valid)
            vstore(result, 0, value, full)


class CausalOperandVector:
    """Build row-scaled BF16 operands for the causal Cube products."""

    def __init__(self, fp32: bool, shared: ChunkWorkspace | None = None):
        input_dtype = dtypes.float32 if fp32 else dtypes.bfloat16
        self.cached_inputs = (
            shared is not None and not fp32 and shared.cache_causal_inputs
        )
        self.resident_inverse = shared is not None and shared.direct_results
        self.operand_batch_rows = 32 if self.resident_inverse else CAUSAL_BATCH_ROWS
        self.inverse_mask = shared.batch_fp32[2] if self.resident_inverse else None
        self.causal_l1 = shared.causal_l1 if self.resident_inverse else None

        if self.cached_inputs:
            self.cached_query, self.cached_key = shared.batch_bf16
            self.cached_gamma = shared.wy_diagonal_residual[0]
            self.cached_forward = shared.wy_diagonal_residual[1]
        self.batch_operands = (not fp32) and (shared is not None)
        if self.batch_operands:
            self.batch_float = shared.value_gradient
            self.batch_low = shared.value_output
        # Existing shared beta_tile is no longer modified after this block's
        # operands are published. Both AIVs refresh all 64 entries, so the
        # result epilogue never relies on the other AIV's private UB.
        self.beta_cache = (
            Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE,)))
            if shared is None
            else shared.beta_tile
        )
        if shared is None:
            self.query = RingBuffer(
                MemLoc.UB, input_dtype, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.key = RingBuffer(
                MemLoc.UB, input_dtype, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.gamma = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.reference = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.beta = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE,)), depth=1
            )
            self.d_Aqk = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, CHUNK_SIZE)), depth=1
            )
            self.d_lower = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, CHUNK_SIZE)), depth=1
            )
            self.query_exp = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((1, SUPPORTED_HEAD_DIM)),
                depth=1,
            )
            self.key_exp = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((1, SUPPORTED_HEAD_DIM)),
                depth=1,
            )
            self.key_inverse_exp = RingBuffer(
                MemLoc.UB,
                dtypes.bfloat16,
                make_layout((1, SUPPORTED_HEAD_DIM)),
                depth=1,
            )
            self.d_output_matrix = RingBuffer(
                MemLoc.UB, dtypes.bfloat16, make_layout((1, CHUNK_SIZE)), depth=1
            )
            self.d_inverse_matrix = RingBuffer(
                MemLoc.UB, dtypes.bfloat16, make_layout((1, CHUNK_SIZE)), depth=1
            )
            self.d_inverse_unweighted = RingBuffer(
                MemLoc.UB, dtypes.bfloat16, make_layout((1, CHUNK_SIZE)), depth=1
            )
        else:
            if fp32:
                self.query, self.key = shared.fp32_rows[4:6]
            else:
                self.query, self.key = shared.bf16_rows[:2]
            self.gamma, self.reference, self.d_Aqk, self.d_lower = shared.fp32_rows[:4]
            self.beta = shared.beta_values
            (
                self.query_exp,
                self.key_exp,
                self.key_inverse_exp,
                self.d_output_matrix,
                self.d_inverse_matrix,
                self.d_inverse_unweighted,
            ) = shared.bf16_rows[2:8]

    @jit
    def run(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_beta,
        gm_d_Aqk,
        gm_d_lower,
        gm_lengths,
        gm_query_exp,
        gm_key_exp,
        gm_key_inverse_exp,
        gm_d_output_matrix,
        gm_d_inverse_matrix,
        gm_d_inverse_unweighted,
        scale: dtypes.float32,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
    ):
        for chunk in range(get_block_idx(), gm_lengths.shape[0], get_block_num()):
            self.process_chunk(
                gm_query,
                gm_key,
                gm_gamma,
                gm_beta,
                gm_d_Aqk,
                gm_d_lower,
                gm_query_exp,
                gm_key_exp,
                gm_key_inverse_exp,
                gm_d_output_matrix,
                gm_d_inverse_matrix,
                gm_d_inverse_unweighted,
                scale,
                chunk,
                gm_lengths[chunk],
                block_start,
                block_end,
            )

    @jit
    def process_chunk(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_beta,
        gm_d_Aqk,
        gm_d_lower,
        gm_query_exp,
        gm_key_exp,
        gm_key_inverse_exp,
        gm_d_output_matrix,
        gm_d_inverse_matrix,
        gm_d_inverse_unweighted,
        scale: dtypes.float32,
        chunk,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        residual_features: Constexpr[bool] = False,
    ):
        mem_copy(self.beta.next_write(), tile_slice(gm_beta, (CHUNK_SIZE,), (chunk,)))
        beta = self.beta.next_read()
        mem_copy(self.beta_cache, beta)
        mem_copy(
            self.reference,
            tile_slice(
                gm_gamma, (1, SUPPORTED_HEAD_DIM), (chunk * CHUNK_SIZE + block_start, 0)
            ),
        )
        reference = self.reference
        if const_expr(self.batch_operands):
            self._process_chunk_batch(
                gm_query,
                gm_key,
                gm_gamma,
                gm_d_Aqk,
                gm_d_lower,
                gm_query_exp,
                gm_key_exp,
                gm_key_inverse_exp,
                gm_d_output_matrix,
                gm_d_inverse_matrix,
                gm_d_inverse_unweighted,
                beta,
                reference,
                scale,
                chunk,
                valid,
                block_start,
                block_end,
                residual_features,
            )
        else:
            self._process_chunk_rows(
                gm_query,
                gm_key,
                gm_gamma,
                gm_d_Aqk,
                gm_d_lower,
                gm_query_exp,
                gm_key_exp,
                gm_key_inverse_exp,
                gm_d_output_matrix,
                gm_d_inverse_matrix,
                gm_d_inverse_unweighted,
                beta,
                reference,
                scale,
                chunk,
                valid,
                block_start,
                block_end,
                residual_features,
            )

    @jit
    def process_chunk_pair(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_beta,
        gm_d_Aqk,
        gm_d_lower,
        gm_query_exp,
        gm_key_exp,
        gm_key_inverse_exp,
        gm_query_exp_low,
        gm_key_exp_low,
        gm_key_inverse_exp_low,
        gm_d_output_matrix,
        gm_d_inverse_matrix,
        gm_d_inverse_unweighted,
        scale: dtypes.float32,
        chunk,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        midpoint: dtypes.int64 = 0,
    ):
        reference_row = block_start
        if midpoint != 0:
            reference_row = block_start + (block_end - block_start) // 2
        if const_expr(self.batch_operands):
            mem_copy(
                self.beta.next_write(), tile_slice(gm_beta, (CHUNK_SIZE,), (chunk,))
            )
            beta = self.beta.next_read()
            mem_copy(self.beta_cache, beta)
            mem_copy(
                self.reference,
                tile_slice(
                    gm_gamma,
                    (1, SUPPORTED_HEAD_DIM),
                    (chunk * CHUNK_SIZE + reference_row, 0),
                ),
            )
            reference = self.reference
            self._process_chunk_batch_pair(
                gm_query,
                gm_key,
                gm_gamma,
                gm_d_Aqk,
                gm_d_lower,
                gm_query_exp,
                gm_key_exp,
                gm_key_inverse_exp,
                gm_query_exp_low,
                gm_key_exp_low,
                gm_key_inverse_exp_low,
                gm_d_output_matrix,
                gm_d_inverse_matrix,
                gm_d_inverse_unweighted,
                beta,
                reference,
                scale,
                chunk,
                valid,
                block_start,
                block_end,
            )
        else:
            self.process_chunk(
                gm_query,
                gm_key,
                gm_gamma,
                gm_beta,
                gm_d_Aqk,
                gm_d_lower,
                gm_query_exp,
                gm_key_exp,
                gm_key_inverse_exp,
                gm_d_output_matrix,
                gm_d_inverse_matrix,
                gm_d_inverse_unweighted,
                scale,
                chunk,
                valid,
                block_start,
                block_end,
            )
            self.process_chunk(
                gm_query,
                gm_key,
                gm_gamma,
                gm_beta,
                gm_d_Aqk,
                gm_d_lower,
                gm_query_exp_low,
                gm_key_exp_low,
                gm_key_inverse_exp_low,
                gm_d_output_matrix,
                gm_d_inverse_matrix,
                gm_d_inverse_unweighted,
                scale,
                chunk,
                valid,
                block_start,
                block_end,
                True,
            )

    @jit
    def _process_chunk_rows(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_d_Aqk,
        gm_d_lower,
        gm_query_exp,
        gm_key_exp,
        gm_key_inverse_exp,
        gm_d_output_matrix,
        gm_d_inverse_matrix,
        gm_d_inverse_unweighted,
        beta,
        reference,
        scale: dtypes.float32,
        chunk,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        residual_features: Constexpr[bool],
    ):
        for token in range(get_subblock_id(), CHUNK_SIZE, get_subblock_dim()):
            row = chunk * CHUNK_SIZE + token
            mem_copy(
                self.query,
                tile_slice(gm_query, (1, SUPPORTED_HEAD_DIM), (row, 0)),
            )
            mem_copy(
                self.key,
                tile_slice(gm_key, (1, SUPPORTED_HEAD_DIM), (row, 0)),
            )
            mem_copy(
                self.gamma,
                tile_slice(gm_gamma, (1, SUPPORTED_HEAD_DIM), (row, 0)),
            )
            self._gate_rows(
                self.query,
                self.key,
                self.gamma,
                reference,
                beta,
                token,
                valid,
                block_start,
                block_end,
                self.query_exp,
                self.key_exp,
                self.key_inverse_exp,
                residual_features,
            )
            mem_copy(
                tile_slice(gm_query_exp, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                self.query_exp,
            )
            mem_copy(
                tile_slice(gm_key_exp, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                self.key_exp,
            )
            mem_copy(
                tile_slice(gm_key_inverse_exp, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                self.key_inverse_exp,
            )
            mem_copy(
                tile_slice(self.d_Aqk, (1, CHUNK_SIZE), (0, 0)),
                tile_slice(gm_d_Aqk, (1, CHUNK_SIZE), (row, 0)),
            )
            mem_copy(
                tile_slice(self.d_lower, (1, CHUNK_SIZE), (0, 0)),
                tile_slice(gm_d_lower, (1, CHUNK_SIZE), (row, 0)),
            )
            self._mask_matrices(
                self.d_Aqk,
                self.d_lower,
                beta,
                token,
                valid,
                block_start,
                block_end,
                self.d_output_matrix,
                self.d_inverse_matrix,
                self.d_inverse_unweighted,
                scale,
            )
            mem_copy(
                tile_slice(gm_d_output_matrix, (1, CHUNK_SIZE), (row, 0)),
                tile_slice(self.d_output_matrix, (1, CHUNK_SIZE), (0, 0)),
            )
            mem_copy(
                tile_slice(gm_d_inverse_unweighted, (1, CHUNK_SIZE), (row, 0)),
                tile_slice(self.d_inverse_unweighted, (1, CHUNK_SIZE), (0, 0)),
            )

    @jit
    def _process_chunk_batch_pair(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_d_Aqk,
        gm_d_lower,
        gm_query_exp,
        gm_key_exp,
        gm_key_inverse_exp,
        gm_query_exp_low,
        gm_key_exp_low,
        gm_key_inverse_exp_low,
        gm_d_output_matrix,
        gm_d_inverse_matrix,
        gm_d_inverse_unweighted,
        beta,
        reference,
        scale: dtypes.float32,
        chunk,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
    ):
        if const_expr(self.resident_inverse):
            # Full chunks use cached Q/K/G. Compact 32-row NZ output planes
            # occupy exactly the existing 32 KiB float and low slots.
            token_start = get_subblock_id() * (CHUNK_SIZE // 2)
            feature_shape = (CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)
            matrix_shape = (CHUNK_SIZE // 2, CHUNK_SIZE)
            fp_write = self.batch_float.next_write()
            mem_copy(
                from_tensor(fp_write, layout=make_layout(matrix_shape)),
                tile_slice(gm_d_Aqk, matrix_shape, (chunk * 2 + get_subblock_id(), 0)),
            )
            low_write = self.batch_low.next_write()
            fp_read = self.batch_float.next_read()
            low_read = self.batch_low.next_read()
            d_aqk = from_tensor(fp_read, layout=make_layout(matrix_shape))
            d_lower = from_tensor(self.inverse_mask, layout=make_layout(matrix_shape))
            query_exp = from_tensor(
                low_read, offset=0, data_format="nz", layout=make_layout(feature_shape)
            )
            key_exp = from_tensor(
                low_read,
                offset=8192,
                data_format="nz",
                layout=make_layout(feature_shape),
            )
            key_inverse = from_tensor(
                low_read,
                offset=16384,
                data_format="nz",
                layout=make_layout(feature_shape),
            )
            d_output = from_tensor(
                low_read,
                offset=24576,
                data_format="nz",
                layout=make_layout(matrix_shape),
            )
            d_unweighted = from_tensor(
                low_read,
                offset=28672,
                data_format="nz",
                layout=make_layout(matrix_shape),
            )
            query_low = from_tensor(
                fp_read,
                dtype=dtypes.bfloat16,
                offset=8192,
                data_format="nz",
                layout=make_layout(feature_shape),
            )
            key_low = from_tensor(
                fp_read,
                dtype=dtypes.bfloat16,
                offset=16384,
                data_format="nz",
                layout=make_layout(feature_shape),
            )
            inverse_low = from_tensor(
                fp_read,
                dtype=dtypes.bfloat16,
                offset=24576,
                data_format="nz",
                layout=make_layout(feature_shape),
            )
            self._gate_rows_pair_batch(
                self.cached_query,
                self.cached_key,
                self.cached_gamma,
                reference,
                beta,
                token_start,
                valid,
                block_start,
                block_end,
                query_exp,
                key_exp,
                key_inverse,
                query_low,
                key_low,
                inverse_low,
                1,
            )
            self._mask_matrices_batch(
                d_aqk,
                d_lower,
                beta,
                token_start,
                valid,
                block_start,
                block_end,
                d_output,
                d_unweighted,
                d_unweighted,
                scale,
                1,
            )
            mem_copy(
                tile_slice(self.causal_l1[0], feature_shape, (get_subblock_id(), 0)),
                query_exp,
            )
            mem_copy(
                tile_slice(self.causal_l1[1], feature_shape, (get_subblock_id(), 0)),
                key_exp,
            )
            mem_copy(
                tile_slice(self.causal_l1[2], feature_shape, (get_subblock_id(), 0)),
                key_inverse,
            )
            mem_copy(
                tile_slice(self.causal_l1[3], feature_shape, (get_subblock_id(), 0)),
                query_low,
            )
            mem_copy(
                tile_slice(self.causal_l1[4], feature_shape, (get_subblock_id(), 0)),
                key_low,
            )
            mem_copy(
                tile_slice(self.causal_l1[5], feature_shape, (get_subblock_id(), 0)),
                inverse_low,
            )
            mem_copy(
                tile_slice(self.causal_l1[6], matrix_shape, (get_subblock_id(), 0)),
                d_output,
            )
            mem_copy(
                tile_slice(self.causal_l1[7], matrix_shape, (get_subblock_id(), 0)),
                d_unweighted,
            )
        else:
            # The input prefix occupies 8 KiB of the shared 48 KiB FP32 channel.
            # Place three BF16 residual planes in its unused tail; no new UB allocation.
            # Reference, beta and coefficient matrices are shared by both feature planes.
            # At eight rows: gamma[8,128] + dA[8,64] + dL[8,64] = 8 KiB.
            # BF16 Q/K + three feature outputs + three matrix outputs use 13 KiB.
            # All views retain the existing Channel's produce/consume ownership.
            for group in range(CHUNK_SIZE // (2 * self.operand_batch_rows)):
                token_start = (
                    get_subblock_id() * (CHUNK_SIZE // 2)
                    + group * self.operand_batch_rows
                )
                row_tile = (
                    chunk * (CHUNK_SIZE // self.operand_batch_rows)
                    + token_start // self.operand_batch_rows
                )
                feature_shape = (self.operand_batch_rows, SUPPORTED_HEAD_DIM)
                matrix_shape = (self.operand_batch_rows, CHUNK_SIZE)
                low_write = self.batch_low.next_write()
                if const_expr(not self.cached_inputs):
                    mem_copy(
                        tile_slice(low_write, feature_shape, (0, 0)),
                        tile_slice(gm_query, feature_shape, (row_tile, 0)),
                    )
                if const_expr(not self.cached_inputs):
                    mem_copy(
                        tile_slice(low_write, feature_shape, (1, 0)),
                        tile_slice(gm_key, feature_shape, (row_tile, 0)),
                    )
                fp_write = self.batch_float.next_write()
                if const_expr(not self.cached_inputs):
                    mem_copy(
                        tile_slice(fp_write, feature_shape, (0, 0)),
                        tile_slice(gm_gamma, feature_shape, (row_tile, 0)),
                    )
                mem_copy(
                    from_tensor(
                        fp_write,
                        offset=self.operand_batch_rows
                        * SUPPORTED_HEAD_DIM
                        * FP32_ELEMENT_BYTES,
                        layout=make_layout(matrix_shape),
                    ),
                    tile_slice(gm_d_Aqk, matrix_shape, (row_tile, 0)),
                )
                if const_expr(not self.resident_inverse):
                    mem_copy(
                        from_tensor(
                            fp_write,
                            offset=(
                                self.operand_batch_rows * SUPPORTED_HEAD_DIM
                                + self.operand_batch_rows * CHUNK_SIZE
                            )
                            * FP32_ELEMENT_BYTES,
                            layout=make_layout(matrix_shape),
                        ),
                        tile_slice(gm_d_lower, matrix_shape, (row_tile, 0)),
                    )

                low_read = self.batch_low.next_read()
                fp_read = self.batch_float.next_read()
                if const_expr(self.cached_inputs):
                    query = tile_slice(self.cached_query, feature_shape, (group, 0))
                    key = tile_slice(self.cached_key, feature_shape, (group, 0))
                    gamma = tile_slice(self.cached_gamma, feature_shape, (group, 0))
                else:
                    query = tile_slice(low_read, feature_shape, (0, 0))
                    key = tile_slice(low_read, feature_shape, (1, 0))
                    gamma = tile_slice(fp_read, feature_shape, (0, 0))
                d_aqk = from_tensor(
                    fp_read,
                    offset=self.operand_batch_rows
                    * SUPPORTED_HEAD_DIM
                    * FP32_ELEMENT_BYTES,
                    layout=make_layout(matrix_shape),
                )
                d_lower = from_tensor(
                    fp_read,
                    offset=(
                        self.operand_batch_rows * SUPPORTED_HEAD_DIM
                        + self.operand_batch_rows * CHUNK_SIZE
                    )
                    * FP32_ELEMENT_BYTES,
                    layout=make_layout(matrix_shape),
                )
                if const_expr(self.resident_inverse):
                    d_lower = tile_slice(
                        from_tensor(
                            self.inverse_mask,
                            layout=make_layout((CHUNK_SIZE // 2, CHUNK_SIZE)),
                        ),
                        matrix_shape,
                        (group, 0),
                    )
                query_exp = tile_slice(low_read, feature_shape, (2, 0))
                key_exp = tile_slice(low_read, feature_shape, (3, 0))
                key_inverse = tile_slice(low_read, feature_shape, (4, 0))
                # Matrix outputs are compact batch-by-64 views, not 128-stride slices.
                d_output = from_tensor(
                    low_read,
                    offset=5
                    * self.operand_batch_rows
                    * SUPPORTED_HEAD_DIM
                    * BF16_ELEMENT_BYTES,
                    layout=make_layout(matrix_shape),
                )
                d_inverse = from_tensor(
                    low_read,
                    offset=(
                        5 * self.operand_batch_rows * SUPPORTED_HEAD_DIM
                        + self.operand_batch_rows * CHUNK_SIZE
                    )
                    * BF16_ELEMENT_BYTES,
                    layout=make_layout(matrix_shape),
                )
                d_unweighted = from_tensor(
                    low_read,
                    offset=(
                        5 * self.operand_batch_rows * SUPPORTED_HEAD_DIM
                        + 2 * self.operand_batch_rows * CHUNK_SIZE
                    )
                    * BF16_ELEMENT_BYTES,
                    layout=make_layout(matrix_shape),
                )
                residual = from_tensor(
                    fp_read,
                    dtype=dtypes.bfloat16,
                    offset=(
                        self.operand_batch_rows * SUPPORTED_HEAD_DIM
                        + 2 * self.operand_batch_rows * CHUNK_SIZE
                    )
                    * FP32_ELEMENT_BYTES,
                    layout=make_layout(
                        (3 * self.operand_batch_rows, SUPPORTED_HEAD_DIM)
                    ),
                )
                query_low = tile_slice(residual, feature_shape, (0, 0))
                key_low = tile_slice(residual, feature_shape, (1, 0))
                inverse_low = tile_slice(residual, feature_shape, (2, 0))
                if const_expr(self.resident_inverse):
                    query_exp = from_tensor(
                        query_exp, data_format="nz", layout=make_layout(feature_shape)
                    )
                    key_exp = from_tensor(
                        key_exp, data_format="nz", layout=make_layout(feature_shape)
                    )
                    key_inverse = from_tensor(
                        key_inverse, data_format="nz", layout=make_layout(feature_shape)
                    )
                    query_low = from_tensor(
                        query_low, data_format="nz", layout=make_layout(feature_shape)
                    )
                    key_low = from_tensor(
                        key_low, data_format="nz", layout=make_layout(feature_shape)
                    )
                    inverse_low = from_tensor(
                        inverse_low, data_format="nz", layout=make_layout(feature_shape)
                    )
                    d_output = from_tensor(
                        d_output, data_format="nz", layout=make_layout(matrix_shape)
                    )
                    d_unweighted = from_tensor(
                        d_unweighted, data_format="nz", layout=make_layout(matrix_shape)
                    )
                if (
                    token_start >= block_start
                    and token_start + self.operand_batch_rows <= block_end
                    and token_start + self.operand_batch_rows <= valid
                ):
                    self._gate_rows_pair_batch(
                        query,
                        key,
                        gamma,
                        reference,
                        beta,
                        token_start,
                        valid,
                        block_start,
                        block_end,
                        query_exp,
                        key_exp,
                        key_inverse,
                        query_low,
                        key_low,
                        inverse_low,
                        1,
                    )
                    self._mask_matrices_batch(
                        d_aqk,
                        d_lower,
                        beta,
                        token_start,
                        valid,
                        block_start,
                        block_end,
                        d_output,
                        d_inverse,
                        d_unweighted,
                        scale,
                        1,
                    )
                elif (
                    token_start + self.operand_batch_rows <= block_start
                    and token_start + self.operand_batch_rows <= valid
                ):
                    self._gate_rows_pair_batch(
                        query,
                        key,
                        gamma,
                        reference,
                        beta,
                        token_start,
                        valid,
                        block_start,
                        block_end,
                        query_exp,
                        key_exp,
                        key_inverse,
                        query_low,
                        key_low,
                        inverse_low,
                        2,
                    )
                    self._mask_matrices_batch(
                        d_aqk,
                        d_lower,
                        beta,
                        token_start,
                        valid,
                        block_start,
                        block_end,
                        d_output,
                        d_inverse,
                        d_unweighted,
                        scale,
                        2,
                    )
                elif token_start >= block_end or token_start >= valid:
                    self._gate_rows_pair_batch(
                        query,
                        key,
                        gamma,
                        reference,
                        beta,
                        token_start,
                        valid,
                        block_start,
                        block_end,
                        query_exp,
                        key_exp,
                        key_inverse,
                        query_low,
                        key_low,
                        inverse_low,
                        3,
                    )
                    self._mask_matrices_batch(
                        d_aqk,
                        d_lower,
                        beta,
                        token_start,
                        valid,
                        block_start,
                        block_end,
                        d_output,
                        d_inverse,
                        d_unweighted,
                        scale,
                        3,
                    )
                else:
                    self._gate_rows_pair_batch(
                        query,
                        key,
                        gamma,
                        reference,
                        beta,
                        token_start,
                        valid,
                        block_start,
                        block_end,
                        query_exp,
                        key_exp,
                        key_inverse,
                        query_low,
                        key_low,
                        inverse_low,
                        0,
                    )
                    self._mask_matrices_batch(
                        d_aqk,
                        d_lower,
                        beta,
                        token_start,
                        valid,
                        block_start,
                        block_end,
                        d_output,
                        d_inverse,
                        d_unweighted,
                        scale,
                        0,
                    )
                if const_expr(self.resident_inverse):
                    mem_copy(
                        tile_slice(
                            self.causal_l1[0],
                            feature_shape,
                            (token_start // self.operand_batch_rows, 0),
                        ),
                        query_exp,
                    )
                    mem_copy(
                        tile_slice(
                            self.causal_l1[1],
                            feature_shape,
                            (token_start // self.operand_batch_rows, 0),
                        ),
                        key_exp,
                    )
                    mem_copy(
                        tile_slice(
                            self.causal_l1[2],
                            feature_shape,
                            (token_start // self.operand_batch_rows, 0),
                        ),
                        key_inverse,
                    )
                    mem_copy(
                        tile_slice(
                            self.causal_l1[3],
                            feature_shape,
                            (token_start // self.operand_batch_rows, 0),
                        ),
                        query_low,
                    )
                    mem_copy(
                        tile_slice(
                            self.causal_l1[4],
                            feature_shape,
                            (token_start // self.operand_batch_rows, 0),
                        ),
                        key_low,
                    )
                    mem_copy(
                        tile_slice(
                            self.causal_l1[5],
                            feature_shape,
                            (token_start // self.operand_batch_rows, 0),
                        ),
                        inverse_low,
                    )
                    mem_copy(
                        tile_slice(
                            self.causal_l1[6],
                            matrix_shape,
                            (token_start // self.operand_batch_rows, 0),
                        ),
                        d_output,
                    )
                    mem_copy(
                        tile_slice(
                            self.causal_l1[7],
                            matrix_shape,
                            (token_start // self.operand_batch_rows, 0),
                        ),
                        d_unweighted,
                    )
                else:
                    mem_copy(
                        tile_slice(gm_query_exp, feature_shape, (row_tile, 0)),
                        tile_slice(low_read, feature_shape, (2, 0)),
                    )
                    mem_copy(
                        tile_slice(gm_key_exp, feature_shape, (row_tile, 0)),
                        tile_slice(low_read, feature_shape, (3, 0)),
                    )
                    mem_copy(
                        tile_slice(gm_key_inverse_exp, feature_shape, (row_tile, 0)),
                        tile_slice(low_read, feature_shape, (4, 0)),
                    )
                    mem_copy(
                        tile_slice(gm_query_exp_low, feature_shape, (row_tile, 0)),
                        query_low,
                    )
                    mem_copy(
                        tile_slice(gm_key_exp_low, feature_shape, (row_tile, 0)),
                        key_low,
                    )
                    mem_copy(
                        tile_slice(
                            gm_key_inverse_exp_low, feature_shape, (row_tile, 0)
                        ),
                        inverse_low,
                    )
                    mem_copy(
                        tile_slice(gm_d_output_matrix, matrix_shape, (row_tile, 0)),
                        from_tensor(
                            low_read,
                            offset=5
                            * self.operand_batch_rows
                            * SUPPORTED_HEAD_DIM
                            * BF16_ELEMENT_BYTES,
                            layout=make_layout(matrix_shape),
                        ),
                    )
                    mem_copy(
                        tile_slice(
                            gm_d_inverse_unweighted, matrix_shape, (row_tile, 0)
                        ),
                        from_tensor(
                            low_read,
                            offset=(
                                5 * self.operand_batch_rows * SUPPORTED_HEAD_DIM
                                + 2 * self.operand_batch_rows * CHUNK_SIZE
                            )
                            * BF16_ELEMENT_BYTES,
                            layout=make_layout(matrix_shape),
                        ),
                    )

    @jit
    def _process_chunk_batch(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_d_Aqk,
        gm_d_lower,
        gm_query_exp,
        gm_key_exp,
        gm_key_inverse_exp,
        gm_d_output_matrix,
        gm_d_inverse_matrix,
        gm_d_inverse_unweighted,
        beta,
        reference,
        scale: dtypes.float32,
        chunk,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        residual_features: Constexpr[bool],
    ):
        # Reference and the full beta vector are loaded once by process_chunk.
        # At eight rows: gamma[8,128] + dA[8,64] + dL[8,64] = 8 KiB.
        # BF16 Q/K + three feature outputs + three matrix outputs use 13 KiB.
        # All views retain the existing Channel's produce/consume ownership.
        for group in range(CHUNK_SIZE // (2 * self.operand_batch_rows)):
            token_start = (
                get_subblock_id() * (CHUNK_SIZE // 2) + group * self.operand_batch_rows
            )
            row_tile = (
                chunk * (CHUNK_SIZE // self.operand_batch_rows)
                + token_start // self.operand_batch_rows
            )
            feature_shape = (self.operand_batch_rows, SUPPORTED_HEAD_DIM)
            matrix_shape = (self.operand_batch_rows, CHUNK_SIZE)
            low_write = self.batch_low.next_write()
            mem_copy(
                tile_slice(low_write, feature_shape, (0, 0)),
                tile_slice(gm_query, feature_shape, (row_tile, 0)),
            )
            mem_copy(
                tile_slice(low_write, feature_shape, (1, 0)),
                tile_slice(gm_key, feature_shape, (row_tile, 0)),
            )
            fp_write = self.batch_float.next_write()
            mem_copy(
                tile_slice(fp_write, feature_shape, (0, 0)),
                tile_slice(gm_gamma, feature_shape, (row_tile, 0)),
            )
            mem_copy(
                from_tensor(
                    fp_write,
                    offset=self.operand_batch_rows
                    * SUPPORTED_HEAD_DIM
                    * FP32_ELEMENT_BYTES,
                    layout=make_layout(matrix_shape),
                ),
                tile_slice(gm_d_Aqk, matrix_shape, (row_tile, 0)),
            )
            mem_copy(
                from_tensor(
                    fp_write,
                    offset=(
                        self.operand_batch_rows * SUPPORTED_HEAD_DIM
                        + self.operand_batch_rows * CHUNK_SIZE
                    )
                    * FP32_ELEMENT_BYTES,
                    layout=make_layout(matrix_shape),
                ),
                tile_slice(gm_d_lower, matrix_shape, (row_tile, 0)),
            )

            low_read = self.batch_low.next_read()
            fp_read = self.batch_float.next_read()
            query = tile_slice(low_read, feature_shape, (0, 0))
            key = tile_slice(low_read, feature_shape, (1, 0))
            gamma = tile_slice(fp_read, feature_shape, (0, 0))
            d_aqk = from_tensor(
                fp_read,
                offset=self.operand_batch_rows
                * SUPPORTED_HEAD_DIM
                * FP32_ELEMENT_BYTES,
                layout=make_layout(matrix_shape),
            )
            d_lower = from_tensor(
                fp_read,
                offset=(
                    self.operand_batch_rows * SUPPORTED_HEAD_DIM
                    + self.operand_batch_rows * CHUNK_SIZE
                )
                * FP32_ELEMENT_BYTES,
                layout=make_layout(matrix_shape),
            )
            query_exp = tile_slice(low_read, feature_shape, (2, 0))
            key_exp = tile_slice(low_read, feature_shape, (3, 0))
            key_inverse = tile_slice(low_read, feature_shape, (4, 0))
            # Matrix outputs are compact batch-by-64 views, not 128-stride slices.
            d_output = from_tensor(
                low_read,
                offset=5
                * self.operand_batch_rows
                * SUPPORTED_HEAD_DIM
                * BF16_ELEMENT_BYTES,
                layout=make_layout(matrix_shape),
            )
            d_inverse = from_tensor(
                low_read,
                offset=(
                    5 * self.operand_batch_rows * SUPPORTED_HEAD_DIM
                    + self.operand_batch_rows * CHUNK_SIZE
                )
                * BF16_ELEMENT_BYTES,
                layout=make_layout(matrix_shape),
            )
            d_unweighted = from_tensor(
                low_read,
                offset=(
                    5 * self.operand_batch_rows * SUPPORTED_HEAD_DIM
                    + 2 * self.operand_batch_rows * CHUNK_SIZE
                )
                * BF16_ELEMENT_BYTES,
                layout=make_layout(matrix_shape),
            )
            for row in range(self.operand_batch_rows):
                token = token_start + row
                self._gate_rows(
                    tile_slice(query, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                    tile_slice(key, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                    tile_slice(gamma, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                    reference,
                    beta,
                    token,
                    valid,
                    block_start,
                    block_end,
                    tile_slice(query_exp, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                    tile_slice(key_exp, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                    tile_slice(key_inverse, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                    residual_features,
                )
                self._mask_matrices(
                    tile_slice(d_aqk, (1, CHUNK_SIZE), (row, 0)),
                    tile_slice(d_lower, (1, CHUNK_SIZE), (row, 0)),
                    beta,
                    token,
                    valid,
                    block_start,
                    block_end,
                    tile_slice(d_output, (1, CHUNK_SIZE), (row, 0)),
                    tile_slice(d_inverse, (1, CHUNK_SIZE), (row, 0)),
                    tile_slice(d_unweighted, (1, CHUNK_SIZE), (row, 0)),
                    scale,
                )
            mem_copy(
                tile_slice(gm_query_exp, feature_shape, (row_tile, 0)),
                tile_slice(low_read, feature_shape, (2, 0)),
            )
            mem_copy(
                tile_slice(gm_key_exp, feature_shape, (row_tile, 0)),
                tile_slice(low_read, feature_shape, (3, 0)),
            )
            mem_copy(
                tile_slice(gm_key_inverse_exp, feature_shape, (row_tile, 0)),
                tile_slice(low_read, feature_shape, (4, 0)),
            )
            mem_copy(
                tile_slice(gm_d_output_matrix, matrix_shape, (row_tile, 0)),
                from_tensor(
                    low_read,
                    offset=5
                    * self.operand_batch_rows
                    * SUPPORTED_HEAD_DIM
                    * BF16_ELEMENT_BYTES,
                    layout=make_layout(matrix_shape),
                ),
            )
            mem_copy(
                tile_slice(gm_d_inverse_unweighted, matrix_shape, (row_tile, 0)),
                from_tensor(
                    low_read,
                    offset=(
                        5 * self.operand_batch_rows * SUPPORTED_HEAD_DIM
                        + 2 * self.operand_batch_rows * CHUNK_SIZE
                    )
                    * BF16_ELEMENT_BYTES,
                    layout=make_layout(matrix_shape),
                ),
            )

    @jit
    def _gate_rows_pair_batch(
        self,
        query,
        key,
        gamma,
        reference,
        beta,
        token_start,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        query_exp,
        key_exp,
        key_inverse_exp,
        query_exp_low,
        key_exp_low,
        key_inverse_exp_low,
        row_kind: Constexpr[int] = 0,
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            for row in range(self.operand_batch_rows):
                token = token_start + row
                for half in range(2):
                    offset = row * SUPPORTED_HEAD_DIM + half * D_HALF
                    q_out = zero
                    k_out = zero
                    ki_out = zero
                    if const_expr(row_kind == 1 or row_kind == 2):
                        if const_expr(query.dtype == dtypes.float32):
                            q = vload(query, offset)
                            k = vload(key, offset)
                        else:
                            q = vcast(
                                vload_unpack(
                                    query, offset, unpack_mode=UnpackMode.B16_TO_B32
                                ),
                                dtypes.float32,
                                mask=full,
                            )
                            k = vcast(
                                vload_unpack(
                                    key, offset, unpack_mode=UnpackMode.B16_TO_B32
                                ),
                                dtypes.float32,
                                mask=full,
                            )
                        g = vload(gamma, offset)
                        reference_g = vload(reference, half * D_HALF)
                        inverse = vexp(vsub(reference_g, g, mask=full), mask=full)
                        if const_expr(self.cached_inputs):
                            vstore(
                                self.cached_gamma,
                                (token % (CHUNK_SIZE // 2)) * SUPPORTED_HEAD_DIM
                                + half * D_HALF,
                                inverse,
                                full,
                            )
                        ki_out = vmul(k, inverse, mask=full)
                        if const_expr(row_kind == 1):
                            forward = vexp(vsub(g, reference_g, mask=full), mask=full)
                            if const_expr(self.cached_inputs):
                                vstore(
                                    self.cached_forward,
                                    (token % (CHUNK_SIZE // 2)) * SUPPORTED_HEAD_DIM
                                    + half * D_HALF,
                                    forward,
                                    full,
                                )
                            q_out = vmul(q, forward, mask=full)
                            k_out = vmul(k, forward, mask=full)
                            k_out = vmul(k_out, vload_broadcast(beta, token), mask=full)
                    elif const_expr(row_kind == 0):
                        column_valid = vlts(
                            vdups(block_start, dtypes.int32, mask=full),
                            valid,
                            mask=full,
                        )
                        column_valid = vlts(
                            vdups(token, dtypes.int32, mask=full),
                            valid,
                            mask=column_valid,
                        )
                        column_valid = vlts(
                            vdups(token, dtypes.int32, mask=full),
                            block_end,
                            mask=column_valid,
                        )
                        column_valid_q_out = q_out
                        column_valid_k_out = k_out
                        column_valid_ki_out = ki_out
                        if const_expr(query.dtype == dtypes.float32):
                            q = vload(query, offset)
                            k = vload(key, offset)
                        else:
                            q = vcast(
                                vload_unpack(
                                    query, offset, unpack_mode=UnpackMode.B16_TO_B32
                                ),
                                dtypes.float32,
                                mask=full,
                            )
                            k = vcast(
                                vload_unpack(
                                    key, offset, unpack_mode=UnpackMode.B16_TO_B32
                                ),
                                dtypes.float32,
                                mask=full,
                            )
                        g = vload(gamma, offset)
                        reference_g = vload(reference, half * D_HALF)
                        inverse = vexp(vsub(reference_g, g, mask=full), mask=full)
                        if const_expr(self.cached_inputs):
                            vstore(
                                self.cached_gamma,
                                (token % (CHUNK_SIZE // 2)) * SUPPORTED_HEAD_DIM
                                + half * D_HALF,
                                inverse,
                                column_valid,
                            )
                        ki_out = vmul(k, inverse, mask=full)
                        row_valid = vges(
                            vdups(token, dtypes.int32, mask=full),
                            block_start,
                            mask=column_valid,
                        )
                        row_valid_q_out = q_out
                        row_valid_k_out = k_out
                        forward = vexp(vsub(g, reference_g, mask=full), mask=full)
                        if const_expr(self.cached_inputs):
                            vstore(
                                self.cached_forward,
                                (token % (CHUNK_SIZE // 2)) * SUPPORTED_HEAD_DIM
                                + half * D_HALF,
                                forward,
                                row_valid,
                            )
                        q_out = vmul(q, forward, mask=full)
                        k_out = vmul(k, forward, mask=full)
                        k_out = vmul(k_out, vload_broadcast(beta, token), mask=full)
                        q_out = vselect(q_out, row_valid_q_out, cond_mask=row_valid)
                        k_out = vselect(k_out, row_valid_k_out, cond_mask=row_valid)
                        q_out = vselect(
                            q_out, column_valid_q_out, cond_mask=column_valid
                        )
                        k_out = vselect(
                            k_out, column_valid_k_out, cond_mask=column_valid
                        )
                        ki_out = vselect(
                            ki_out, column_valid_ki_out, cond_mask=column_valid
                        )
                    q_high = vcast(
                        q_out, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN
                    )
                    q_low = vcast(
                        vsub(
                            q_out, vcast(q_high, dtypes.float32, mask=full), mask=full
                        ),
                        dtypes.bfloat16,
                        mask=full,
                        rounding=RoundingMode.RN,
                    )
                    if const_expr(self.resident_inverse):
                        packed_mask, _ = update_mask(D_HALF, elem_bits=16)
                        packed_value = vreinterpret(
                            vpack(
                                vreinterpret_lanes(q_high, dtypes.uint32), dtypes.uint16
                            ),
                            dtypes.bfloat16,
                        )
                        vstore_strided(
                            query_exp,
                            row * 16 + half * 4 * query_exp.physical_layout.stride[0],
                            packed_value,
                            packed_mask,
                            block_stride=query_exp.physical_layout.stride[0] // 16,
                            repeat_stride=0,
                        )
                    else:
                        vstore_pack(
                            query_exp,
                            offset,
                            q_high,
                            full,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    if const_expr(self.resident_inverse):
                        packed_mask, _ = update_mask(D_HALF, elem_bits=16)
                        packed_value = vreinterpret(
                            vpack(
                                vreinterpret_lanes(q_low, dtypes.uint32), dtypes.uint16
                            ),
                            dtypes.bfloat16,
                        )
                        vstore_strided(
                            query_exp_low,
                            row * 16
                            + half * 4 * query_exp_low.physical_layout.stride[0],
                            packed_value,
                            packed_mask,
                            block_stride=query_exp_low.physical_layout.stride[0] // 16,
                            repeat_stride=0,
                        )
                    else:
                        vstore_pack(
                            query_exp_low,
                            offset,
                            q_low,
                            full,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    k_high = vcast(
                        k_out, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN
                    )
                    k_low = vcast(
                        vsub(
                            k_out, vcast(k_high, dtypes.float32, mask=full), mask=full
                        ),
                        dtypes.bfloat16,
                        mask=full,
                        rounding=RoundingMode.RN,
                    )
                    if const_expr(self.resident_inverse):
                        packed_mask, _ = update_mask(D_HALF, elem_bits=16)
                        packed_value = vreinterpret(
                            vpack(
                                vreinterpret_lanes(k_high, dtypes.uint32), dtypes.uint16
                            ),
                            dtypes.bfloat16,
                        )
                        vstore_strided(
                            key_exp,
                            row * 16 + half * 4 * key_exp.physical_layout.stride[0],
                            packed_value,
                            packed_mask,
                            block_stride=key_exp.physical_layout.stride[0] // 16,
                            repeat_stride=0,
                        )
                    else:
                        vstore_pack(
                            key_exp, offset, k_high, full, pack_mode=PackMode.B32_TO_B16
                        )
                    if const_expr(self.resident_inverse):
                        packed_mask, _ = update_mask(D_HALF, elem_bits=16)
                        packed_value = vreinterpret(
                            vpack(
                                vreinterpret_lanes(k_low, dtypes.uint32), dtypes.uint16
                            ),
                            dtypes.bfloat16,
                        )
                        vstore_strided(
                            key_exp_low,
                            row * 16 + half * 4 * key_exp_low.physical_layout.stride[0],
                            packed_value,
                            packed_mask,
                            block_stride=key_exp_low.physical_layout.stride[0] // 16,
                            repeat_stride=0,
                        )
                    else:
                        vstore_pack(
                            key_exp_low,
                            offset,
                            k_low,
                            full,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    ki_high = vcast(
                        ki_out, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN
                    )
                    ki_low = vcast(
                        vsub(
                            ki_out, vcast(ki_high, dtypes.float32, mask=full), mask=full
                        ),
                        dtypes.bfloat16,
                        mask=full,
                        rounding=RoundingMode.RN,
                    )
                    if const_expr(self.resident_inverse):
                        packed_mask, _ = update_mask(D_HALF, elem_bits=16)
                        packed_value = vreinterpret(
                            vpack(
                                vreinterpret_lanes(ki_high, dtypes.uint32),
                                dtypes.uint16,
                            ),
                            dtypes.bfloat16,
                        )
                        vstore_strided(
                            key_inverse_exp,
                            row * 16
                            + half * 4 * key_inverse_exp.physical_layout.stride[0],
                            packed_value,
                            packed_mask,
                            block_stride=key_inverse_exp.physical_layout.stride[0]
                            // 16,
                            repeat_stride=0,
                        )
                    else:
                        vstore_pack(
                            key_inverse_exp,
                            offset,
                            ki_high,
                            full,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    if const_expr(self.resident_inverse):
                        packed_mask, _ = update_mask(D_HALF, elem_bits=16)
                        packed_value = vreinterpret(
                            vpack(
                                vreinterpret_lanes(ki_low, dtypes.uint32), dtypes.uint16
                            ),
                            dtypes.bfloat16,
                        )
                        vstore_strided(
                            key_inverse_exp_low,
                            row * 16
                            + half * 4 * key_inverse_exp_low.physical_layout.stride[0],
                            packed_value,
                            packed_mask,
                            block_stride=key_inverse_exp_low.physical_layout.stride[0]
                            // 16,
                            repeat_stride=0,
                        )
                    else:
                        vstore_pack(
                            key_inverse_exp_low,
                            offset,
                            ki_low,
                            full,
                            pack_mode=PackMode.B32_TO_B16,
                        )

            if const_expr(self.cached_inputs):
                vmem_bar("vst_vld")

    @jit
    def _gate_rows(
        self,
        query,
        key,
        gamma,
        reference,
        beta,
        token,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        query_exp,
        key_exp,
        key_inverse_exp,
        residual_features: Constexpr[bool],
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            for half in range(2):
                offset = half * D_HALF
                q_out = zero
                k_out = zero
                ki_out = zero
                column_valid = vlts(
                    vdups(block_start, dtypes.int32, mask=full), valid, mask=full
                )
                column_valid = vlts(
                    vdups(token, dtypes.int32, mask=full), valid, mask=column_valid
                )
                column_valid = vlts(
                    vdups(token, dtypes.int32, mask=full), block_end, mask=column_valid
                )
                column_valid_q_out = q_out
                column_valid_k_out = k_out
                column_valid_ki_out = ki_out
                if const_expr(query.dtype == dtypes.float32):
                    q = vload(query, offset)
                    k = vload(key, offset)
                else:
                    q = vcast(
                        vload_unpack(query, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    k = vcast(
                        vload_unpack(key, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                g = vload(gamma, offset)
                reference_g = vload(reference, offset)
                inverse = vexp(vsub(reference_g, g, mask=full), mask=full)
                ki_out = vmul(k, inverse, mask=full)
                row_valid = vges(
                    vdups(token, dtypes.int32, mask=full),
                    block_start,
                    mask=column_valid,
                )
                row_valid_q_out = q_out
                row_valid_k_out = k_out
                forward = vexp(vsub(g, reference_g, mask=full), mask=full)
                q_out = vmul(q, forward, mask=full)
                k_out = vmul(k, forward, mask=full)
                k_out = vmul(k_out, vload_broadcast(beta, token), mask=full)
                q_out = vselect(q_out, row_valid_q_out, cond_mask=row_valid)
                k_out = vselect(k_out, row_valid_k_out, cond_mask=row_valid)
                q_out = vselect(q_out, column_valid_q_out, cond_mask=column_valid)
                k_out = vselect(k_out, column_valid_k_out, cond_mask=column_valid)
                ki_out = vselect(ki_out, column_valid_ki_out, cond_mask=column_valid)
                if const_expr(residual_features):
                    # The high pass uses the same RN casts. The second pass
                    # keeps the discarded exp-scaled feature bits; all three
                    # BF16 row Channels are reused after their first DMA.
                    q_high = vcast(
                        q_out, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN
                    )
                    k_high = vcast(
                        k_out, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN
                    )
                    ki_high = vcast(
                        ki_out, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN
                    )
                    q_out = vsub(
                        q_out, vcast(q_high, dtypes.float32, mask=full), mask=full
                    )
                    k_out = vsub(
                        k_out, vcast(k_high, dtypes.float32, mask=full), mask=full
                    )
                    ki_out = vsub(
                        ki_out, vcast(ki_high, dtypes.float32, mask=full), mask=full
                    )
                vstore_pack(
                    query_exp,
                    offset,
                    vcast(q_out, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    key_exp,
                    offset,
                    vcast(k_out, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    key_inverse_exp,
                    offset,
                    vcast(ki_out, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def _mask_matrices_batch(
        self,
        d_Aqk,
        d_lower,
        beta,
        token_start,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        d_output_matrix,
        d_inverse_matrix,
        d_inverse_unweighted,
        scale: dtypes.float32,
        row_kind: Constexpr[int] = 0,
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            for row in range(self.operand_batch_rows):
                token = token_start + row
                output = zero
                inverse = zero
                if const_expr(row_kind == 1):
                    output_mask, _ = update_mask(token + 1, elem_bits=32)
                    inverse_mask, _ = update_mask(token, elem_bits=32)
                    output = vselect(
                        vmuls(vload(d_Aqk, row * CHUNK_SIZE), scale, mask=full),
                        zero,
                        cond_mask=output_mask,
                    )
                    inverse = vselect(
                        vload(d_lower, row * CHUNK_SIZE), zero, cond_mask=inverse_mask
                    )
                elif const_expr(row_kind == 0):
                    row_valid = vlts(
                        vdups(token, dtypes.int32, mask=full), valid, mask=full
                    )
                    row_valid = vges(
                        vdups(token, dtypes.int32, mask=full),
                        block_start,
                        mask=row_valid,
                    )
                    row_valid = vlts(
                        vdups(token, dtypes.int32, mask=full), block_end, mask=row_valid
                    )
                    row_valid_output = output
                    row_valid_inverse = inverse
                    output_mask, _ = update_mask(token + 1, elem_bits=32)
                    inverse_mask, _ = update_mask(token, elem_bits=32)
                    output = vselect(
                        vmuls(vload(d_Aqk, row * CHUNK_SIZE), scale, mask=full),
                        zero,
                        cond_mask=output_mask,
                    )
                    inverse = vselect(
                        vload(d_lower, row * CHUNK_SIZE), zero, cond_mask=inverse_mask
                    )
                    output = vselect(output, row_valid_output, cond_mask=row_valid)
                    inverse = vselect(inverse, row_valid_inverse, cond_mask=row_valid)
                if const_expr(self.resident_inverse):
                    packed_mask, _ = update_mask(D_HALF, elem_bits=16)
                    packed_value = vreinterpret(
                        vpack(
                            vreinterpret_lanes(
                                vcast(output, dtypes.bfloat16, mask=full), dtypes.uint32
                            ),
                            dtypes.uint16,
                        ),
                        dtypes.bfloat16,
                    )
                    vstore_strided(
                        d_output_matrix,
                        row * 16,
                        packed_value,
                        packed_mask,
                        block_stride=d_output_matrix.physical_layout.stride[0] // 16,
                        repeat_stride=0,
                    )
                else:
                    vstore_pack(
                        d_output_matrix,
                        row * CHUNK_SIZE,
                        vcast(output, dtypes.bfloat16, mask=full),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                if const_expr(self.resident_inverse):
                    packed_mask, _ = update_mask(D_HALF, elem_bits=16)
                    packed_value = vreinterpret(
                        vpack(
                            vreinterpret_lanes(
                                vcast(inverse, dtypes.bfloat16, mask=full),
                                dtypes.uint32,
                            ),
                            dtypes.uint16,
                        ),
                        dtypes.bfloat16,
                    )
                    vstore_strided(
                        d_inverse_unweighted,
                        row * 16,
                        packed_value,
                        packed_mask,
                        block_stride=d_inverse_unweighted.physical_layout.stride[0]
                        // 16,
                        repeat_stride=0,
                    )
                else:
                    vstore_pack(
                        d_inverse_unweighted,
                        row * CHUNK_SIZE,
                        vcast(inverse, dtypes.bfloat16, mask=full),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )

    @jit
    def _mask_matrices(
        self,
        d_Aqk,
        d_lower,
        beta,
        token,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        d_output_matrix,
        d_inverse_matrix,
        d_inverse_unweighted,
        scale: dtypes.float32,
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            output = zero
            inverse = zero
            row_valid = vlts(vdups(token, dtypes.int32, mask=full), valid, mask=full)
            row_valid = vges(
                vdups(token, dtypes.int32, mask=full), block_start, mask=row_valid
            )
            row_valid = vlts(
                vdups(token, dtypes.int32, mask=full), block_end, mask=row_valid
            )
            row_valid_output = output
            row_valid_inverse = inverse
            output_mask, _ = update_mask(token + 1, elem_bits=32)
            inverse_mask, _ = update_mask(token, elem_bits=32)
            output = vselect(
                vmuls(vload(d_Aqk, 0), scale, mask=full),
                zero,
                cond_mask=output_mask,
            )
            inverse = vselect(vload(d_lower, 0), zero, cond_mask=inverse_mask)
            output = vselect(output, row_valid_output, cond_mask=row_valid)
            inverse = vselect(inverse, row_valid_inverse, cond_mask=row_valid)
            vstore_pack(
                d_output_matrix,
                0,
                vcast(output, dtypes.bfloat16, mask=full),
                full,
                pack_mode=PackMode.B32_TO_B16,
            )
            vstore_pack(
                d_inverse_unweighted,
                0,
                vcast(inverse, dtypes.bfloat16, mask=full),
                full,
                pack_mode=PackMode.B32_TO_B16,
            )


class CausalResultVector:
    """Accumulate stable block Q/K/G/beta derivatives from Cube products."""

    def __init__(self, fp32: bool, shared: ChunkWorkspace | None = None):
        self.direct_results = shared is not None and shared.direct_results
        self.fuse_base = shared is not None and shared.fuse_base
        self.causal_results = None if shared is None else shared.causal_results
        input_dtype = dtypes.float32 if fp32 else dtypes.bfloat16
        self.cached_inputs = (
            shared is not None and not fp32 and shared.cache_causal_inputs
        )
        self.result_batch_rows = (
            CHUNK_SIZE // 2
            if self.direct_results and self.cached_inputs
            else CAUSAL_BATCH_ROWS
        )
        if self.cached_inputs:
            self.cached_query, self.cached_key = shared.batch_bf16
            self.cached_gamma = shared.wy_diagonal_residual[0]
            self.cached_forward = shared.wy_diagonal_residual[1]
        self.batch_result = (not fp32) and (shared is not None)
        if self.batch_result:
            # The existing value-epilogue Channels are idle until this phase
            # finishes. At eight rows, 11 FP32 [8,128] planes reserve 44 KiB;
            # the shared pool is 48 KiB. BF16 Q/K use 4 KiB of 16 KiB.
            self.batch_float = shared.value_gradient
            self.batch_low = shared.value_output
        self.resident_gradients = self.batch_result and shared.batch_rows
        if self.resident_gradients:
            # WY reduction has finished before the causal rounds. Retain its
            # three private 32-row tiles until the last causal round publishes.
            self.resident_q, self.resident_k, self.resident_g = shared.batch_fp32
            self.resident_beta = shared.batch_scalar
        self.beta_cache = (
            Buffer(MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE,)))
            if shared is None
            else shared.beta_tile
        )
        if shared is None:
            self.query = RingBuffer(
                MemLoc.UB, input_dtype, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.key = RingBuffer(
                MemLoc.UB, input_dtype, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.gamma = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.reference = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.d_query_exp = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.d_key_row_exp = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.d_key_col_output = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.d_key_col_inverse = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.d_beta_matrix = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.d_query = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.d_key = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.d_gate = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.d_beta = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1,)), depth=1
            )
            self.prior_d_query = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.prior_d_key = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.prior_d_gate = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)), depth=1
            )
            self.prior_d_beta = RingBuffer(
                MemLoc.UB, dtypes.float32, make_layout((1,)), depth=1
            )
        else:
            if fp32:
                self.query, self.key = shared.fp32_rows[13:15]
            else:
                self.query, self.key = shared.bf16_rows[:2]
            (
                self.gamma,
                self.reference,
                self.d_query_exp,
                self.d_key_row_exp,
                self.d_key_col_output,
                self.d_key_col_inverse,
                self.d_beta_matrix,
                self.prior_d_query,
                self.prior_d_key,
                self.prior_d_gate,
                self.d_query,
                self.d_key,
                self.d_gate,
            ) = shared.fp32_rows[:13]
            self.prior_d_beta, self.d_beta = shared.fp32_scalars

    @jit
    def run(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_beta,
        gm_d_query_exp,
        gm_d_key_row_exp,
        gm_d_key_col_output,
        gm_d_key_col_inverse,
        gm_d_beta_matrix,
        gm_lengths,
        gm_d_query,
        gm_d_key,
        gm_d_gate,
        gm_d_beta,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
    ):
        for row in range(get_block_idx(), gm_query.shape[0], get_block_num()):
            mem_copy(
                self.beta_cache,
                tile_slice(gm_beta, (CHUNK_SIZE,), (row // CHUNK_SIZE,)),
            )
            self.process_row(
                gm_query,
                gm_key,
                gm_gamma,
                gm_d_query_exp,
                gm_d_key_row_exp,
                gm_d_key_col_output,
                gm_d_key_col_inverse,
                gm_d_beta_matrix,
                gm_d_query,
                gm_d_key,
                gm_d_gate,
                gm_d_beta,
                row,
                gm_lengths[row // CHUNK_SIZE],
                block_start,
                block_end,
            )

    @jit
    def _batch_plane(self, slot, plane: Constexpr[int]):
        return tile_slice(
            slot, (self.result_batch_rows, SUPPORTED_HEAD_DIM), (plane, 0)
        )

    @jit
    def _process_chunk_resident(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_d_query_exp,
        gm_d_key_row_exp,
        gm_d_key_col_output,
        gm_d_key_col_inverse,
        gm_d_beta_matrix,
        gm_d_query,
        gm_d_key,
        gm_d_gate,
        gm_d_beta,
        chunk,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        midpoint: dtypes.int64 = 0,
        retain_head: Constexpr[bool] = False,
    ):
        if const_expr(self.direct_results):
            direct_q = self.causal_results[0]
            direct_k = self.causal_results[1]
            direct_beta = self.causal_results[2]
        if const_expr(self.direct_results and self.cached_inputs):
            # The full-block dispatcher guarantees one causal round covering
            # every row. Inputs/results already live in UB; no staging channel,
            # reference-gamma DMA or initialization of overwritten outputs.
            self._combine_batch(
                self.cached_query,
                self.cached_key,
                self.cached_gamma,
                self.cached_gamma,
                direct_q,
                direct_beta,
                direct_k,
                direct_k,
                direct_beta,
                self.resident_q,
                self.resident_k,
                self.resident_g,
                self.resident_beta,
                self.resident_q,
                self.resident_k,
                self.resident_g,
                self.resident_beta,
                get_subblock_id() * (CHUNK_SIZE // 2),
                valid,
                block_start,
                block_end,
                1,
            )
        else:
            reference_row = block_start
            if midpoint != 0:
                reference_row = block_start + (block_end - block_start) // 2
            if block_start == 0:
                self._zero_resident()
            # Both AIVs own a contiguous 32-row interval. Eight FP32 beta
            # stores form one aligned 32-B block with exactly one owner.
            for group in range(CHUNK_SIZE // (2 * self.result_batch_rows)):
                token_start = (
                    get_subblock_id() * (CHUNK_SIZE // 2)
                    + group * self.result_batch_rows
                )
                row_tile = (
                    chunk * (CHUNK_SIZE // self.result_batch_rows)
                    + token_start // self.result_batch_rows
                )
                shape = (self.result_batch_rows, SUPPORTED_HEAD_DIM)
                if token_start < block_end:
                    low_write = self.batch_low.next_write()
                    if const_expr(not self.cached_inputs):
                        mem_copy(
                            self._batch_plane(low_write, 0),
                            tile_slice(gm_query, shape, (row_tile, 0)),
                        )
                    if const_expr(not self.cached_inputs):
                        mem_copy(
                            self._batch_plane(low_write, 1),
                            tile_slice(gm_key, shape, (row_tile, 0)),
                        )
                    fp_write = self.batch_float.next_write()
                    if const_expr(not self.cached_inputs):
                        mem_copy(
                            self._batch_plane(fp_write, 0),
                            tile_slice(gm_gamma, shape, (row_tile, 0)),
                        )
                    mem_copy(
                        tile_slice(
                            self._batch_plane(fp_write, 1),
                            (1, SUPPORTED_HEAD_DIM),
                            (0, 0),
                        ),
                        tile_slice(
                            gm_gamma,
                            (1, SUPPORTED_HEAD_DIM),
                            (chunk * CHUNK_SIZE + reference_row, 0),
                        ),
                    )
                    if const_expr(not self.direct_results):
                        mem_copy(
                            self._batch_plane(fp_write, 2),
                            tile_slice(gm_d_query_exp, shape, (row_tile, 0)),
                        )
                    if const_expr(not self.direct_results):
                        mem_copy(
                            self._batch_plane(fp_write, 4),
                            tile_slice(gm_d_key_col_output, shape, (row_tile, 0)),
                        )
                    if const_expr(not self.direct_results):
                        mem_copy(
                            self._batch_plane(fp_write, 6),
                            tile_slice(gm_d_beta_matrix, shape, (row_tile, 0)),
                        )

                    low_read = self.batch_low.next_read()
                    fp_read = self.batch_float.next_read()
                    if const_expr(self.cached_inputs):
                        q = tile_slice(self.cached_query, shape, (group, 0))
                        k = tile_slice(self.cached_key, shape, (group, 0))
                        gamma = tile_slice(self.cached_gamma, shape, (group, 0))
                    else:
                        q = self._batch_plane(low_read, 0)
                        k = self._batch_plane(low_read, 1)
                        gamma = self._batch_plane(fp_read, 0)
                    reference = tile_slice(
                        self._batch_plane(fp_read, 1), (1, SUPPORTED_HEAD_DIM), (0, 0)
                    )
                    dq_exp = self._batch_plane(fp_read, 2)
                    dk_row = self._batch_plane(fp_read, 6)
                    dk_col = self._batch_plane(fp_read, 4)
                    dk_inverse = dk_col  # unused by the compiled combine branch
                    db_matrix = self._batch_plane(fp_read, 6)
                    if const_expr(self.direct_results):
                        dq_exp = tile_slice(direct_q, shape, (group, 0))
                        dk_col = tile_slice(direct_k, shape, (group, 0))
                        dk_inverse = dk_col
                        db_matrix = tile_slice(direct_beta, shape, (group, 0))
                        dk_row = db_matrix
                    prior_q = tile_slice(self.resident_q, shape, (group, 0))
                    prior_k = tile_slice(self.resident_k, shape, (group, 0))
                    prior_g = tile_slice(self.resident_g, shape, (group, 0))
                    prior_beta = tile_slice(
                        self.resident_beta, (self.result_batch_rows,), (group,)
                    )
                    # Each row keeps its original two-half beta reduction.
                    # Initialization and accumulation share one VF for all rows.
                    if (
                        token_start >= block_start
                        and token_start + self.result_batch_rows <= block_end
                        and token_start + self.result_batch_rows <= valid
                    ):
                        self._combine_batch(
                            q,
                            k,
                            gamma,
                            reference,
                            dq_exp,
                            dk_row,
                            dk_col,
                            dk_inverse,
                            db_matrix,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            token_start,
                            valid,
                            block_start,
                            block_end,
                            1,
                        )
                    elif (
                        token_start + self.result_batch_rows <= block_start
                        and token_start + self.result_batch_rows <= valid
                    ):
                        self._combine_batch(
                            q,
                            k,
                            gamma,
                            reference,
                            dq_exp,
                            dk_row,
                            dk_col,
                            dk_inverse,
                            db_matrix,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            token_start,
                            valid,
                            block_start,
                            block_end,
                            2,
                        )
                    else:
                        self._combine_batch(
                            q,
                            k,
                            gamma,
                            reference,
                            dq_exp,
                            dk_row,
                            dk_col,
                            dk_inverse,
                            db_matrix,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            token_start,
                            valid,
                            block_start,
                            block_end,
                            0,
                        )
        if block_end == valid:
            half_shape = (CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)
            half_tile = chunk * 2 + get_subblock_id()
            if const_expr(not retain_head):
                mem_copy(
                    tile_slice(gm_d_query, half_shape, (half_tile, 0)), self.resident_q
                )
            if const_expr(not retain_head):
                mem_copy(
                    tile_slice(gm_d_key, half_shape, (half_tile, 0)), self.resident_k
                )
            if const_expr(not self.fuse_base):
                mem_copy(
                    tile_slice(gm_d_gate, half_shape, (half_tile, 0)), self.resident_g
                )
            mem_copy(
                tile_slice(gm_d_beta, (CHUNK_SIZE // 2,), (half_tile,)),
                self.resident_beta,
            )

    @jit
    def _zero_resident(self):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            for offset in range(0, (CHUNK_SIZE // 2) * SUPPORTED_HEAD_DIM, D_HALF):
                vstore(self.resident_q, offset, zero, full)
                vstore(self.resident_k, offset, zero, full)
                vstore(self.resident_g, offset, zero, full)
            mask, _ = update_mask(CHUNK_SIZE // 2, elem_bits=32)
            vstore(self.resident_beta, 0, zero, mask)
            vmem_bar("vst_vld")

    @jit
    def process_chunk(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_d_query_exp,
        gm_d_key_row_exp,
        gm_d_key_col_output,
        gm_d_key_col_inverse,
        gm_d_beta_matrix,
        gm_d_query,
        gm_d_key,
        gm_d_gate,
        gm_d_beta,
        chunk,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        midpoint: dtypes.int64 = 0,
        retain_head: Constexpr[bool] = False,
    ):
        reference_row = block_start
        if midpoint != 0:
            reference_row = block_start + (block_end - block_start) // 2
        if const_expr(self.resident_gradients):
            self._process_chunk_resident(
                gm_query,
                gm_key,
                gm_gamma,
                gm_d_query_exp,
                gm_d_key_row_exp,
                gm_d_key_col_output,
                gm_d_key_col_inverse,
                gm_d_beta_matrix,
                gm_d_query,
                gm_d_key,
                gm_d_gate,
                gm_d_beta,
                chunk,
                valid,
                block_start,
                block_end,
                midpoint,
                retain_head,
            )
        elif const_expr(self.batch_result):
            # Both AIVs own a contiguous 32-row interval. Eight FP32 beta
            # stores form one aligned 32-B block with exactly one owner.
            for group in range(CHUNK_SIZE // (2 * self.result_batch_rows)):
                token_start = (
                    get_subblock_id() * (CHUNK_SIZE // 2)
                    + group * self.result_batch_rows
                )
                row_tile = (
                    chunk * (CHUNK_SIZE // self.result_batch_rows)
                    + token_start // self.result_batch_rows
                )
                full_row = chunk * CHUNK_SIZE + token_start
                shape = (self.result_batch_rows, SUPPORTED_HEAD_DIM)
                if token_start < block_end:
                    low_write = self.batch_low.next_write()
                    mem_copy(
                        self._batch_plane(low_write, 0),
                        tile_slice(gm_query, shape, (row_tile, 0)),
                    )
                    mem_copy(
                        self._batch_plane(low_write, 1),
                        tile_slice(gm_key, shape, (row_tile, 0)),
                    )
                    fp_write = self.batch_float.next_write()
                    mem_copy(
                        self._batch_plane(fp_write, 0),
                        tile_slice(gm_gamma, shape, (row_tile, 0)),
                    )
                    mem_copy(
                        tile_slice(
                            self._batch_plane(fp_write, 1),
                            (1, SUPPORTED_HEAD_DIM),
                            (0, 0),
                        ),
                        tile_slice(
                            gm_gamma,
                            (1, SUPPORTED_HEAD_DIM),
                            (chunk * CHUNK_SIZE + reference_row, 0),
                        ),
                    )
                    mem_copy(
                        self._batch_plane(fp_write, 2),
                        tile_slice(gm_d_query_exp, shape, (row_tile, 0)),
                    )
                    mem_copy(
                        self._batch_plane(fp_write, 4),
                        tile_slice(gm_d_key_col_output, shape, (row_tile, 0)),
                    )
                    mem_copy(
                        self._batch_plane(fp_write, 6),
                        tile_slice(gm_d_beta_matrix, shape, (row_tile, 0)),
                    )
                    if block_start != 0:
                        mem_copy(
                            self._batch_plane(fp_write, 7),
                            tile_slice(gm_d_query, shape, (row_tile, 0)),
                        )
                        mem_copy(
                            self._batch_plane(fp_write, 8),
                            tile_slice(gm_d_key, shape, (row_tile, 0)),
                        )
                        mem_copy(
                            self._batch_plane(fp_write, 9),
                            tile_slice(gm_d_gate, shape, (row_tile, 0)),
                        )
                        mem_copy(
                            from_tensor(
                                fp_write,
                                offset=10
                                * self.result_batch_rows
                                * SUPPORTED_HEAD_DIM
                                * FP32_ELEMENT_BYTES,
                                layout=make_layout((self.result_batch_rows,)),
                            ),
                            tile_slice(
                                gm_d_beta,
                                (self.result_batch_rows,),
                                (full_row // self.result_batch_rows,),
                            ),
                        )

                    low_read = self.batch_low.next_read()
                    fp_read = self.batch_float.next_read()
                    q = self._batch_plane(low_read, 0)
                    k = self._batch_plane(low_read, 1)
                    gamma = self._batch_plane(fp_read, 0)
                    reference = tile_slice(
                        self._batch_plane(fp_read, 1), (1, SUPPORTED_HEAD_DIM), (0, 0)
                    )
                    dq_exp = self._batch_plane(fp_read, 2)
                    dk_row = self._batch_plane(fp_read, 6)
                    dk_col = self._batch_plane(fp_read, 4)
                    dk_inverse = dk_col  # unused by the compiled combine branch
                    db_matrix = self._batch_plane(fp_read, 6)
                    prior_q = self._batch_plane(fp_read, 7)
                    prior_k = self._batch_plane(fp_read, 8)
                    prior_g = self._batch_plane(fp_read, 9)
                    prior_beta = from_tensor(
                        fp_read,
                        offset=10
                        * self.result_batch_rows
                        * SUPPORTED_HEAD_DIM
                        * FP32_ELEMENT_BYTES,
                        layout=make_layout((self.result_batch_rows,)),
                    )
                    # Each row keeps its original two-half beta reduction.
                    # Initialization and accumulation share one VF for all rows.
                    if (
                        token_start >= block_start
                        and token_start + self.result_batch_rows <= block_end
                        and token_start + self.result_batch_rows <= valid
                    ):
                        self._combine_batch(
                            q,
                            k,
                            gamma,
                            reference,
                            dq_exp,
                            dk_row,
                            dk_col,
                            dk_inverse,
                            db_matrix,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            token_start,
                            valid,
                            block_start,
                            block_end,
                            1,
                        )
                    elif (
                        token_start + self.result_batch_rows <= block_start
                        and token_start + self.result_batch_rows <= valid
                    ):
                        self._combine_batch(
                            q,
                            k,
                            gamma,
                            reference,
                            dq_exp,
                            dk_row,
                            dk_col,
                            dk_inverse,
                            db_matrix,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            token_start,
                            valid,
                            block_start,
                            block_end,
                            2,
                        )
                    else:
                        self._combine_batch(
                            q,
                            k,
                            gamma,
                            reference,
                            dq_exp,
                            dk_row,
                            dk_col,
                            dk_inverse,
                            db_matrix,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            prior_q,
                            prior_k,
                            prior_g,
                            prior_beta,
                            token_start,
                            valid,
                            block_start,
                            block_end,
                            0,
                        )
                    mem_copy(
                        tile_slice(gm_d_query, shape, (row_tile, 0)),
                        self._batch_plane(fp_read, 7),
                    )
                    mem_copy(
                        tile_slice(gm_d_key, shape, (row_tile, 0)),
                        self._batch_plane(fp_read, 8),
                    )
                    mem_copy(
                        tile_slice(gm_d_gate, shape, (row_tile, 0)),
                        self._batch_plane(fp_read, 9),
                    )
                    mem_copy(
                        tile_slice(
                            gm_d_beta,
                            (self.result_batch_rows,),
                            (full_row // self.result_batch_rows,),
                        ),
                        from_tensor(
                            fp_read,
                            offset=10
                            * self.result_batch_rows
                            * SUPPORTED_HEAD_DIM
                            * FP32_ELEMENT_BYTES,
                            layout=make_layout((self.result_batch_rows,)),
                        ),
                    )
                elif block_start == 0:
                    # Future rows have no contribution in this causal block.
                    # Initialize their scratch once; later blocks preserve it
                    # until their prefix becomes active. No stale slot reads.
                    fp_write = self.batch_float.next_write()
                    prior_q = self._batch_plane(fp_write, 7)
                    prior_k = self._batch_plane(fp_write, 8)
                    prior_g = self._batch_plane(fp_write, 9)
                    prior_beta = from_tensor(
                        fp_write,
                        offset=10
                        * self.result_batch_rows
                        * SUPPORTED_HEAD_DIM
                        * FP32_ELEMENT_BYTES,
                        layout=make_layout((self.result_batch_rows,)),
                    )
                    for row in range(self.result_batch_rows):
                        self._zero_prior(
                            tile_slice(prior_q, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                            tile_slice(prior_k, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                            tile_slice(prior_g, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                            tile_slice(prior_beta, (1,), (row,)),
                        )
                    fp_read = self.batch_float.next_read()
                    mem_copy(
                        tile_slice(gm_d_query, shape, (row_tile, 0)),
                        self._batch_plane(fp_read, 7),
                    )
                    mem_copy(
                        tile_slice(gm_d_key, shape, (row_tile, 0)),
                        self._batch_plane(fp_read, 8),
                    )
                    mem_copy(
                        tile_slice(gm_d_gate, shape, (row_tile, 0)),
                        self._batch_plane(fp_read, 9),
                    )
                    mem_copy(
                        tile_slice(
                            gm_d_beta,
                            (self.result_batch_rows,),
                            (full_row // self.result_batch_rows,),
                        ),
                        from_tensor(
                            fp_read,
                            offset=10
                            * self.result_batch_rows
                            * SUPPORTED_HEAD_DIM
                            * FP32_ELEMENT_BYTES,
                            layout=make_layout((self.result_batch_rows,)),
                        ),
                    )
        else:
            for token in range(get_subblock_id(), CHUNK_SIZE, get_subblock_dim()):
                self.process_row(
                    gm_query,
                    gm_key,
                    gm_gamma,
                    gm_d_query_exp,
                    gm_d_key_row_exp,
                    gm_d_key_col_output,
                    gm_d_key_col_inverse,
                    gm_d_beta_matrix,
                    gm_d_query,
                    gm_d_key,
                    gm_d_gate,
                    gm_d_beta,
                    chunk * CHUNK_SIZE + token,
                    valid,
                    block_start,
                    block_end,
                )

    @jit
    def process_row(
        self,
        gm_query,
        gm_key,
        gm_gamma,
        gm_d_query_exp,
        gm_d_key_row_exp,
        gm_d_key_col_output,
        gm_d_key_col_inverse,
        gm_d_beta_matrix,
        gm_d_query,
        gm_d_key,
        gm_d_gate,
        gm_d_beta,
        row,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
    ):
        shape = (1, SUPPORTED_HEAD_DIM)
        token = row % CHUNK_SIZE
        mem_copy(self.query, tile_slice(gm_query, shape, (row, 0)))
        mem_copy(self.key, tile_slice(gm_key, shape, (row, 0)))
        mem_copy(self.gamma, tile_slice(gm_gamma, shape, (row, 0)))
        mem_copy(
            self.reference,
            tile_slice(
                gm_gamma, shape, ((row // CHUNK_SIZE) * CHUNK_SIZE + block_start, 0)
            ),
        )
        mem_copy(self.d_query_exp, tile_slice(gm_d_query_exp, shape, (row, 0)))
        mem_copy(
            self.d_key_col_output,
            tile_slice(gm_d_key_col_output, shape, (row, 0)),
        )
        mem_copy(self.d_beta_matrix, tile_slice(gm_d_beta_matrix, shape, (row, 0)))
        if block_start == 0:
            self._zero_prior(
                self.prior_d_query,
                self.prior_d_key,
                self.prior_d_gate,
                self.prior_d_beta.next_write(),
            )
        else:
            mem_copy(self.prior_d_query, tile_slice(gm_d_query, shape, (row, 0)))
            mem_copy(self.prior_d_key, tile_slice(gm_d_key, shape, (row, 0)))
            mem_copy(self.prior_d_gate, tile_slice(gm_d_gate, shape, (row, 0)))
            mem_copy(
                self.prior_d_beta.next_write(), tile_slice(gm_d_beta, (1,), (row,))
            )
        beta_matrix = self.d_beta_matrix
        row_key_matrix = beta_matrix
        column_output = self.d_key_col_output
        column_inverse = column_output  # unused in the compiled branch
        self._combine(
            self.query,
            self.key,
            self.gamma,
            self.reference,
            self.d_query_exp,
            row_key_matrix,
            column_output,
            column_inverse,
            beta_matrix,
            self.prior_d_query,
            self.prior_d_key,
            self.prior_d_gate,
            self.prior_d_beta.next_read(),
            self.d_query,
            self.d_key,
            self.d_gate,
            self.d_beta.next_write(),
            token,
            valid,
            block_start,
            block_end,
        )
        mem_copy(tile_slice(gm_d_query, shape, (row, 0)), self.d_query)
        mem_copy(tile_slice(gm_d_key, shape, (row, 0)), self.d_key)
        mem_copy(tile_slice(gm_d_gate, shape, (row, 0)), self.d_gate)
        mem_copy(tile_slice(gm_d_beta, (1,), (row,)), self.d_beta.next_read())

    @jit
    def _zero_prior(self, d_query, d_key, d_gate, d_beta):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=mask)
            for half in range(2):
                offset = half * D_HALF
                vstore(d_query, offset, zero, mask)
                vstore(d_key, offset, zero, mask)
                vstore(d_gate, offset, zero, mask)
            vstore_first(d_beta, 0, zero)
            if const_expr(self.batch_result):
                vmem_bar("vst_vld")

    @jit
    def _combine_batch(
        self,
        query,
        key,
        gamma,
        reference,
        d_query_exp,
        d_key_row_exp,
        d_key_col_output,
        d_key_col_inverse,
        d_beta_matrix,
        prior_d_query,
        prior_d_key,
        prior_d_gate,
        prior_d_beta,
        d_query,
        d_key,
        d_gate,
        d_beta,
        token_start,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
        row_kind: Constexpr[int] = 0,
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            one, _ = update_mask(1, elem_bits=32)
            for row in range(self.result_batch_rows):
                token = token_start + row
                beta_total = vdups(0.0, dtypes.float32, mask=one)
                for half in range(2):
                    offset = row * SUPPORTED_HEAD_DIM + half * D_HALF
                    if const_expr(query.dtype == dtypes.float32):
                        q = vload(query, offset)
                        k = vload(key, offset)
                    else:
                        q = vcast(
                            vload_unpack(
                                query, offset, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=full,
                        )
                        k = vcast(
                            vload_unpack(
                                key, offset, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=full,
                        )
                    g = vload(gamma, offset)
                    reference_g = vload(reference, half * D_HALF)
                    q_grad = vdups(0.0, dtypes.float32, mask=full)
                    k_row_grad = vdups(0.0, dtypes.float32, mask=full)
                    beta_term = vdups(0.0, dtypes.float32, mask=full)
                    if const_expr(row_kind == 1):
                        if const_expr(self.cached_inputs):
                            forward = vload(
                                self.cached_forward,
                                (token % (CHUNK_SIZE // 2)) * SUPPORTED_HEAD_DIM
                                + half * D_HALF,
                            )
                        else:
                            forward = vexp(vsub(g, reference_g, mask=full), mask=full)
                        q_grad = vmul(forward, vload(d_query_exp, offset), mask=full)
                        k_row_grad = vmul(
                            forward, vload(d_key_row_exp, offset), mask=full
                        )
                        k_row_grad = vmul(
                            k_row_grad,
                            vload_broadcast(self.beta_cache, token),
                            mask=full,
                        )
                        beta_term = vmul(
                            vmul(forward, k, mask=full),
                            vload(d_beta_matrix, offset),
                            mask=full,
                        )
                    elif const_expr(row_kind == 0):
                        row_valid = vlts(
                            vdups(block_start, dtypes.int32, mask=full),
                            valid,
                            mask=full,
                        )
                        row_valid = vges(
                            vdups(token, dtypes.int32, mask=full),
                            block_start,
                            mask=row_valid,
                        )
                        row_valid = vlts(
                            vdups(token, dtypes.int32, mask=full),
                            block_end,
                            mask=row_valid,
                        )
                        row_valid = vlts(
                            vdups(token, dtypes.int32, mask=full), valid, mask=row_valid
                        )
                        row_valid_q_grad = q_grad
                        row_valid_k_row_grad = k_row_grad
                        row_valid_beta_term = beta_term
                        if const_expr(self.cached_inputs):
                            forward = vload(
                                self.cached_forward,
                                (token % (CHUNK_SIZE // 2)) * SUPPORTED_HEAD_DIM
                                + half * D_HALF,
                            )
                        else:
                            forward = vexp(vsub(g, reference_g, mask=full), mask=full)
                        q_grad = vmul(forward, vload(d_query_exp, offset), mask=full)
                        k_row_grad = vmul(
                            forward, vload(d_key_row_exp, offset), mask=full
                        )
                        k_row_grad = vmul(
                            k_row_grad,
                            vload_broadcast(self.beta_cache, token),
                            mask=full,
                        )
                        beta_term = vmul(
                            vmul(forward, k, mask=full),
                            vload(d_beta_matrix, offset),
                            mask=full,
                        )
                        q_grad = vselect(q_grad, row_valid_q_grad, cond_mask=row_valid)
                        k_row_grad = vselect(
                            k_row_grad, row_valid_k_row_grad, cond_mask=row_valid
                        )
                        beta_term = vselect(
                            beta_term, row_valid_beta_term, cond_mask=row_valid
                        )
                    k_col_grad = vdups(0.0, dtypes.float32, mask=full)
                    if const_expr(row_kind != 0):
                        if const_expr(self.cached_inputs):
                            inverse = g
                        else:
                            inverse = vexp(vsub(reference_g, g, mask=full), mask=full)
                        column_sum = vload(d_key_col_output, offset)
                        k_col_grad = vmul(inverse, column_sum, mask=full)
                    else:
                        column_valid = vlts(
                            vdups(block_start, dtypes.int32, mask=full),
                            valid,
                            mask=full,
                        )
                        column_valid = vlts(
                            vdups(token, dtypes.int32, mask=full),
                            block_end,
                            mask=column_valid,
                        )
                        column_valid = vlts(
                            vdups(token, dtypes.int32, mask=full),
                            valid,
                            mask=column_valid,
                        )
                        column_valid_k_col_grad = k_col_grad
                        if const_expr(self.cached_inputs):
                            inverse = g
                        else:
                            inverse = vexp(vsub(reference_g, g, mask=full), mask=full)
                        column_sum = vload(d_key_col_output, offset)
                        k_col_grad = vmul(inverse, column_sum, mask=full)
                        k_col_grad = vselect(
                            k_col_grad, column_valid_k_col_grad, cond_mask=column_valid
                        )
                    k_grad = vadd(k_row_grad, k_col_grad, mask=full)
                    g_grad = vsub(
                        vadd(
                            vmul(q, q_grad, mask=full),
                            vmul(k, k_row_grad, mask=full),
                            mask=full,
                        ),
                        vmul(k, k_col_grad, mask=full),
                        mask=full,
                    )
                    beta_total = vadd(
                        beta_total, vreduce_sum(beta_term, mask=full), mask=one
                    )
                    prior_q = vdups(0.0, dtypes.float32, mask=full)
                    prior_k = prior_q
                    prior_g = prior_q
                    has_prior = vnes(
                        vdups(block_start, dtypes.int32, mask=full), 0, mask=full
                    )
                    has_prior_prior_q = prior_q
                    has_prior_prior_k = prior_k
                    has_prior_prior_g = prior_g
                    prior_q = vload(prior_d_query, offset)
                    prior_k = vload(prior_d_key, offset)
                    prior_g = vload(prior_d_gate, offset)
                    prior_q = vselect(prior_q, has_prior_prior_q, cond_mask=has_prior)
                    prior_k = vselect(prior_k, has_prior_prior_k, cond_mask=has_prior)
                    prior_g = vselect(prior_g, has_prior_prior_g, cond_mask=has_prior)
                    vstore(
                        d_query,
                        offset,
                        vadd(prior_q, q_grad, mask=full),
                        full,
                    )
                    vstore(
                        d_key,
                        offset,
                        vadd(prior_k, k_grad, mask=full),
                        full,
                    )
                    vstore(
                        d_gate,
                        offset,
                        vadd(prior_g, g_grad, mask=full),
                        full,
                    )
                prior_beta = vdups(0.0, dtypes.float32, mask=one)
                has_prior_beta = vnes(
                    vdups(block_start, dtypes.int32, mask=one), 0, mask=one
                )
                has_prior_beta_prior_beta = prior_beta
                prior_beta = vload_broadcast(prior_d_beta, row)
                prior_beta = vselect(
                    prior_beta, has_prior_beta_prior_beta, cond_mask=has_prior_beta
                )
                vstore_first(d_beta, row, vadd(prior_beta, beta_total, mask=one))
            vmem_bar("vst_vld")

    @jit
    def _combine(
        self,
        query,
        key,
        gamma,
        reference,
        d_query_exp,
        d_key_row_exp,
        d_key_col_output,
        d_key_col_inverse,
        d_beta_matrix,
        prior_d_query,
        prior_d_key,
        prior_d_gate,
        prior_d_beta,
        d_query,
        d_key,
        d_gate,
        d_beta,
        token,
        valid,
        block_start: dtypes.int32,
        block_end: dtypes.int32,
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            one, _ = update_mask(1, elem_bits=32)
            beta_total = vdups(0.0, dtypes.float32, mask=one)
            for half in range(2):
                offset = half * D_HALF
                if const_expr(query.dtype == dtypes.float32):
                    q = vload(query, offset)
                    k = vload(key, offset)
                else:
                    q = vcast(
                        vload_unpack(query, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                    k = vcast(
                        vload_unpack(key, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                g = vload(gamma, offset)
                reference_g = vload(reference, offset)
                q_grad = vdups(0.0, dtypes.float32, mask=full)
                k_row_grad = vdups(0.0, dtypes.float32, mask=full)
                beta_term = vdups(0.0, dtypes.float32, mask=full)
                row_valid = vlts(
                    vdups(block_start, dtypes.int32, mask=full), valid, mask=full
                )
                row_valid = vges(
                    vdups(token, dtypes.int32, mask=full), block_start, mask=row_valid
                )
                row_valid = vlts(
                    vdups(token, dtypes.int32, mask=full), block_end, mask=row_valid
                )
                row_valid = vlts(
                    vdups(token, dtypes.int32, mask=full), valid, mask=row_valid
                )
                row_valid_q_grad = q_grad
                row_valid_k_row_grad = k_row_grad
                row_valid_beta_term = beta_term
                forward = vexp(vsub(g, reference_g, mask=full), mask=full)
                q_grad = vmul(forward, vload(d_query_exp, offset), mask=full)
                k_row_grad = vmul(forward, vload(d_key_row_exp, offset), mask=full)
                k_row_grad = vmul(
                    k_row_grad,
                    vload_broadcast(self.beta_cache, token),
                    mask=full,
                )
                beta_term = vmul(
                    vmul(forward, k, mask=full),
                    vload(d_beta_matrix, offset),
                    mask=full,
                )
                q_grad = vselect(q_grad, row_valid_q_grad, cond_mask=row_valid)
                k_row_grad = vselect(
                    k_row_grad, row_valid_k_row_grad, cond_mask=row_valid
                )
                beta_term = vselect(beta_term, row_valid_beta_term, cond_mask=row_valid)
                k_col_grad = vdups(0.0, dtypes.float32, mask=full)
                column_valid = vlts(
                    vdups(block_start, dtypes.int32, mask=full), valid, mask=full
                )
                column_valid = vlts(
                    vdups(token, dtypes.int32, mask=full), block_end, mask=column_valid
                )
                column_valid = vlts(
                    vdups(token, dtypes.int32, mask=full), valid, mask=column_valid
                )
                column_valid_k_col_grad = k_col_grad
                inverse = vexp(vsub(reference_g, g, mask=full), mask=full)
                column_sum = vload(d_key_col_output, offset)
                k_col_grad = vmul(inverse, column_sum, mask=full)
                k_col_grad = vselect(
                    k_col_grad, column_valid_k_col_grad, cond_mask=column_valid
                )
                k_grad = vadd(k_row_grad, k_col_grad, mask=full)
                g_grad = vsub(
                    vadd(
                        vmul(q, q_grad, mask=full),
                        vmul(k, k_row_grad, mask=full),
                        mask=full,
                    ),
                    vmul(k, k_col_grad, mask=full),
                    mask=full,
                )
                beta_total = vadd(
                    beta_total, vreduce_sum(beta_term, mask=full), mask=one
                )
                vstore(
                    d_query,
                    offset,
                    vadd(vload(prior_d_query, offset), q_grad, mask=full),
                    full,
                )
                vstore(
                    d_key,
                    offset,
                    vadd(vload(prior_d_key, offset), k_grad, mask=full),
                    full,
                )
                vstore(
                    d_gate,
                    offset,
                    vadd(vload(prior_d_gate, offset), g_grad, mask=full),
                    full,
                )
            vstore_first(
                d_beta, 0, vadd(vload_broadcast(prior_d_beta, 0), beta_total, mask=one)
            )
            if const_expr(self.batch_result):
                vmem_bar("vst_vld")


class DirectCausalDeltaVector:
    """Relative-gate contractions; each AIV owns 64 feature lanes."""

    def __init__(self, shared, adjacent_decay=False):
        self.shared = shared
        self.adjacent_decay = adjacent_decay
        self.query = shared.batch_fp32[0]
        self.key = shared.batch_fp32[1]
        self.gamma = shared.batch_fp32[2]
        self.dq = shared.wy_diagonal_residual[0]
        self.dk = shared.wy_diagonal_residual[1]
        self.beta = shared.beta_tile
        self.partial_beta = shared.partial_beta

    @jit
    def process_chunk(
        self,
        gm_q,
        gm_k,
        gm_g,
        gm_b,
        gm_a,
        gm_l,
        gm_dq,
        gm_dk,
        gm_dg,
        gm_db,
        gm_partial,
        scale,
        chunk,
        valid,
    ):
        feature = get_subblock_id()
        q = from_tensor(self.query, layout=make_layout((CHUNK_SIZE, D_HALF)))
        k = from_tensor(self.key, layout=make_layout((CHUNK_SIZE, D_HALF)))
        g = from_tensor(self.gamma, layout=make_layout((CHUNK_SIZE, D_HALF)))
        if const_expr(gm_q.dtype == dtypes.float32):
            mem_copy(q, tile_slice(gm_q, (CHUNK_SIZE, D_HALF), (chunk, feature)))
            mem_copy(k, tile_slice(gm_k, (CHUNK_SIZE, D_HALF), (chunk, feature)))
        else:
            raw_q = from_tensor(
                self.shared.batch_bf16[0],
                dtype=dtypes.bfloat16,
                layout=make_layout((CHUNK_SIZE, D_HALF)),
            )
            raw_k = from_tensor(
                self.shared.batch_bf16[1],
                dtype=dtypes.bfloat16,
                layout=make_layout((CHUNK_SIZE, D_HALF)),
            )
            mem_copy(raw_q, tile_slice(gm_q, (CHUNK_SIZE, D_HALF), (chunk, feature)))
            mem_copy(raw_k, tile_slice(gm_k, (CHUNK_SIZE, D_HALF), (chunk, feature)))
            self._convert_qk(raw_q, raw_k, q, k)
        mem_copy(g, tile_slice(gm_g, (CHUNK_SIZE, D_HALF), (chunk, feature)))
        if const_expr(gm_b.dtype == dtypes.float32):
            mem_copy(self.beta, tile_slice(gm_b, (CHUNK_SIZE,), (chunk,)))
        else:
            mem_copy(
                self.shared.bf16_scalars.next_write(),
                tile_slice(gm_b, (CHUNK_SIZE,), (chunk,)),
            )
            self._convert_beta(self.shared.bf16_scalars.next_read())
        work = from_tensor(
            self.shared.value_gradient.next_write(),
            layout=make_layout((2 * CHUNK_SIZE, D_HALF)),
        )
        lower = tile_slice(work, (CHUNK_SIZE, D_HALF), (0, 0))
        dg = tile_slice(work, (CHUNK_SIZE, D_HALF), (1, 0))
        coefficients = from_tensor(
            self.shared.value_output.next_write(),
            dtype=dtypes.float32,
            layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
        )
        mem_copy(coefficients, tile_slice(gm_a, (CHUNK_SIZE, CHUNK_SIZE), (chunk, 0)))
        mem_copy(lower, tile_slice(gm_l, (CHUNK_SIZE, CHUNK_SIZE), (chunk, 0)))
        coefficients = from_tensor(
            self.shared.value_output.next_read(),
            dtype=dtypes.float32,
            layout=make_layout((CHUNK_SIZE, CHUNK_SIZE)),
        )
        self._prepare_coefficients(coefficients, lower, scale, valid)
        # Only the aligned-only caller enables this specialization. That path
        # already guarantees built-in nonpositive gates and 64 valid rows.
        # Masked/external-gate paths keep the original direct contraction,
        # without introducing a runtime branch or a short-row setup cost.
        self._compute(
            q,
            k,
            g,
            coefficients,
            lower,
            dg,
            self.partial_beta,
            valid,
            self.adjacent_decay,
        )
        mem_copy(
            tile_slice(gm_dq, (CHUNK_SIZE, D_HALF), (chunk, feature)),
            from_tensor(self.dq, layout=make_layout((CHUNK_SIZE, D_HALF))),
        )
        mem_copy(
            tile_slice(gm_dk, (CHUNK_SIZE, D_HALF), (chunk, feature)),
            from_tensor(self.dk, layout=make_layout((CHUNK_SIZE, D_HALF))),
        )
        mem_copy(
            tile_slice(gm_dg, (CHUNK_SIZE, D_HALF), (chunk, feature)),
            tile_slice(
                from_tensor(
                    self.shared.value_gradient.next_read(),
                    layout=make_layout((2 * CHUNK_SIZE, D_HALF)),
                ),
                (CHUNK_SIZE, D_HALF),
                (1, 0),
            ),
        )
        mem_copy(
            tile_slice(gm_partial, (1, CHUNK_SIZE), (chunk * CHUNK_SIZE, feature)),
            self.partial_beta,
        )
        if const_expr(self.shared.fuse_base):
            # Cube already retired at the WY join. Only the paired AIV's
            # feature partials are needed by this reduction and row exchange.
            vec_sync_all()
            vec_sync_block_arrive(PIPE.MTE3, 1, mode=1)
            vec_sync_block_wait(PIPE.S, 1, mode=1)
            onchip_rewind(reset_sync_id=True)
        else:
            _sync_cube_vector_handoff(1, reset_sync_id=True)
        if get_subblock_id() == 0:
            mem_copy(
                self.shared.fp32_rows[0],
                tile_slice(
                    gm_partial, (1, SUPPORTED_HEAD_DIM), (chunk * CHUNK_SIZE, 0)
                ),
            )
            self._sum_beta(
                self.shared.fp32_rows[0], self.shared.beta_gradient.next_write()
            )
            mem_copy(
                tile_slice(gm_db, (CHUNK_SIZE,), (chunk,)),
                self.shared.beta_gradient.next_read(),
            )

    @jit
    def _convert_qk(self, raw_q, raw_k, q, k):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for row in range(CHUNK_SIZE):
                qr = vcast(
                    vload_unpack(
                        raw_q, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                kr = vcast(
                    vload_unpack(
                        raw_k, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32
                    ),
                    dtypes.float32,
                    mask=full,
                )
                vstore(q, row * D_HALF, qr, full)
                vstore(k, row * D_HALF, kr, full)
            vmem_bar("vst_vld")

    @jit
    def _convert_beta(self, source):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            value = vcast(
                vload_unpack(source, 0, unpack_mode=UnpackMode.B16_TO_B32),
                dtypes.float32,
                mask=full,
            )
            vstore(self.beta, 0, value, full)
            vmem_bar("vst_vld")

    @jit
    def _prepare_coefficients(self, coefficients, lower, scale: dtypes.float32, valid):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for row in range(valid):
                a = vmuls(vload(coefficients, row * CHUNK_SIZE), scale, mask=full)
                causal, _ = update_mask(row, elem_bits=32)
                lower_value = vselect(
                    vneg(vload(lower, row * CHUNK_SIZE), mask=full),
                    vdups(0.0, dtypes.float32, mask=full),
                    cond_mask=causal,
                )
                vstore(coefficients, row * CHUNK_SIZE, a, full)
                vstore(lower, row * CHUNK_SIZE, lower_value, full)
            vmem_bar("vst_vld")

    @jit
    def _compute(
        self,
        query,
        key,
        gamma,
        coeff,
        lower,
        dgate,
        dbeta,
        valid,
        adjacent_decay: Constexpr[bool],
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            for row in range(valid, CHUNK_SIZE):
                vstore(self.dq, row * D_HALF, zero, full)
                vstore(self.dk, row * D_HALF, zero, full)
                vstore(dgate, row * D_HALF, zero, full)
                vstore_first(dbeta, row, zero)
            vmem_bar("vst_vld")
            if const_expr(adjacent_decay):
                # Private gamma storage expires here. Descending conversion
                # reads each original adjacent pair before either is reused.
                for index in range(1, valid):
                    row = valid - index
                    step = vexp(
                        vsub(
                            vload(gamma, row * D_HALF),
                            vload(gamma, (row - 1) * D_HALF),
                            mask=full,
                        ),
                        mask=full,
                    )
                    vstore(gamma, row * D_HALF, step, full)
                vmem_bar("vst_vld")
            for row in range(valid):
                if const_expr(query.dtype == dtypes.float32):
                    qi = vload(query, row * D_HALF)
                    ki = vload(key, row * D_HALF)
                else:
                    qi = vcast(
                        vload_unpack(
                            query, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                    ki = vcast(
                        vload_unpack(
                            key, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                gi = zero
                if const_expr(not adjacent_decay):
                    gi = vload(gamma, row * D_HALF)
                bi = vload_broadcast(self.beta, row)
                bki = vmul(bi, ki, mask=full)
                dq = zero
                kr = zero
                weight = vdups(1.0, dtypes.float32, mask=full)
                for index in range(row):
                    if const_expr(adjacent_decay):
                        col = row - 1 - index
                    else:
                        col = index
                    if const_expr(key.dtype == dtypes.float32):
                        kj = vload(key, col * D_HALF)
                    else:
                        kj = vcast(
                            vload_unpack(
                                key, col * D_HALF, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=full,
                        )
                    if const_expr(adjacent_decay):
                        # exp(G_i-G_j) = product(exp(G_t-G_(t-1)), t=j+1..i).
                        # Descending columns only multiply by bounded decays;
                        # no inverse exponential or gate-span split is needed.
                        weight = vmul(
                            weight, vload(gamma, (col + 1) * D_HALF), mask=full
                        )
                    else:
                        weight = vexp(
                            vsub(gi, vload(gamma, col * D_HALF), mask=full), mask=full
                        )
                    a = vload_broadcast(coeff, row * CHUNK_SIZE + col)
                    lower_value = vload_broadcast(lower, row * CHUNK_SIZE + col)
                    weighted_key = vmul(kj, weight, mask=full)
                    dq = vadd(dq, vmul(a, weighted_key, mask=full), mask=full)
                    kr = vadd(kr, vmul(lower_value, weighted_key, mask=full), mask=full)
                    column_term = vmul(
                        weight,
                        vadd(
                            vmul(a, qi, mask=full),
                            vmul(lower_value, bki, mask=full),
                            mask=full,
                        ),
                        mask=full,
                    )
                    kc = vadd(vload(self.dk, col * D_HALF), column_term, mask=full)
                    vstore(self.dk, col * D_HALF, kc, full)
                # Earlier rows cannot contribute to this diagonal key row.
                # Initialize it directly; the single end-of-row barrier also
                # publishes all earlier-column updates for the next row.
                diagonal = vload_broadcast(coeff, row * CHUNK_SIZE + row)
                dq = vadd(dq, vmul(diagonal, ki, mask=full), mask=full)
                diagonal_key = vmul(diagonal, qi, mask=full)
                db = vreduce_sum(vmul(ki, kr, mask=full), mask=full)
                krow = vmul(bi, kr, mask=full)
                dk = vadd(diagonal_key, krow, mask=full)
                vstore(self.dq, row * D_HALF, dq, full)
                vstore(self.dk, row * D_HALF, dk, full)
                # Defer column gate contributions until all later rows arrive.
                dg = vadd(
                    vmul(qi, dq, mask=full),
                    vmuls(vmul(ki, krow, mask=full), 2.0, mask=full),
                    mask=full,
                )
                vstore(dgate, row * D_HALF, dg, full)
                vstore_first(dbeta, row, db)
                vmem_bar("vst_vld")
            for row in range(valid):
                if const_expr(key.dtype == dtypes.float32):
                    ki = vload(key, row * D_HALF)
                else:
                    ki = vcast(
                        vload_unpack(
                            key, row * D_HALF, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=full,
                    )
                dg = vsub(
                    vload(dgate, row * D_HALF),
                    vmul(ki, vload(self.dk, row * D_HALF), mask=full),
                    mask=full,
                )
                vstore(dgate, row * D_HALF, dg, full)
            vmem_bar("vst_vld")

    @jit
    def _sum_beta(self, partial, output):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            vstore(
                output,
                0,
                vadd(vload(partial, 0), vload(partial, CHUNK_SIZE), mask=full),
                full,
            )
