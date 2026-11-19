# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MXFP8 QKV kernel, Host entry, cached factory and Native AOT export.

Chunk mapping is applied only to output, position and slot rows.
"""

from itertools import product

import cannbotdsl
from cannbotdsl import Dim, MemLoc, Tensor, TensorSpec, aot, dtypes
from cannbotdsl.aot import Constexpr
from cannbotdsl.lang import jit, kernel, vf
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, cv_valid_extent, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    full_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup,
    vdups,
    vload,
    vload_unpack,
    vmul,
    vmuls,
    vreduce_sum,
    vsqrt,
    vstore_pack,
    vsub,
    update_mask,
)
from cannbotdsl.ops.reg.cast import RoundingMode
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import from_tensor, make_layout, tile_slice
from cannbotdsl.types.dtypes import bfloat16 as BF16, float32 as F32

# ---- per-head and quantization constants ----
HEAD_DIM = 256
ROPE_DIM = 64
NON_ROPE_DIM = HEAD_DIM - ROPE_DIM
ROPE_HALF_DIM = ROPE_DIM // 2
MX_SCALE_GROUP_SIZE = 32

# ---- tiling constants ----
WORK_WINDOW_ROWS = 4  # ASW token tiles per window
HEAD_UNIT_COLS = 256  # exactly one head-unit per N tile (epilogue granularity)
L0_K_TILE = 128
SCALE_K_L1 = 2048  # cover full K of the network case in one scale window
L1_SCALE_COUNT = SCALE_K_L1 // MX_SCALE_GROUP_SIZE  # E8M0 scales per L1 window
L0_SCALE_BYTES = L0_K_TILE // MX_SCALE_GROUP_SIZE  # scale bytes per L0 sub-block
L1_PACKED_SCALE_PAIRS = SCALE_K_L1 // (
    2 * MX_SCALE_GROUP_SIZE
)  # paired scale groups per L1 window

# Host-side tiling profiles. The small-M profile raises core occupancy at the
# cost of reloading B tiles more often; the large-M profile uses a
# 256-row Cube tile with full L0C utilization.
TILING_PROFILES = {
    "base256": dict(
        base_m=256,
        l1_k_tile=256,
        l1_data_depth=2,
        l0c_depth=1,
        ub_depth=1,
    ),
    "base128": dict(
        base_m=128,
        l1_k_tile=512,
        l1_data_depth=2,
        l0c_depth=2,
        ub_depth=2,
    ),
}
SMALL_M_PROFILE_MAX_ROWS = 3072


VECTOR_LANES = 64  # vector register elements (2048b / 32b)
INV_HEAD_DIM = 1.0 / HEAD_DIM

E8M0 = dtypes.float8_e8m0
FP8_E4M3 = dtypes.float8_e4m3fn
FP8_E5M2 = dtypes.float8_e5m2
FP8_DTYPES = (FP8_E4M3, FP8_E5M2)


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="fused_quant_qkv_proj_norm_rope_cache",
        op_type="FusedQuantQkvProjNormRopeCache",
        inputs=(
            "gm_hidden_quant",
            "gm_hidden_scale",
            "gm_weight_quant",
            "gm_weight_scale",
            "gm_k_gamma",
            "gm_positions",
            "gm_cos_sin",
            "gm_slot_mapping",
        ),
        inouts=(
            "gm_q_out",
            "gm_k_out",
            "gm_v_out",
            "gm_mirror_k_out",
            "gm_mirror_v_out",
            "gm_k_cache",
            "gm_v_cache",
        ),
    )
)
class FusedQuantQkvProjNormRopeCacheKernel:
    def __init__(
        self,
        has_mirror: bool,
        positions_contiguous: bool,
        has_chunk: bool,
        base_m: int,
        l1_k_tile: int,
        l1_data_depth: int,
        l0c_depth: int,
        ub_depth: int,
    ):
        # Trace-time constants: fold the mirror block-store and the
        # cos/sin load mode away before IR emission.
        self.has_mirror = bool(has_mirror)
        self.positions_contiguous = bool(positions_contiguous)
        self.has_chunk = bool(has_chunk)
        self.base_m = base_m
        self.aiv_tile_rows = base_m // 2
        self.l1_k_tile = l1_k_tile
        self.l0_k_tiles_per_l1 = l1_k_tile // L0_K_TILE
        self.scale_k_tiles = SCALE_K_L1 // l1_k_tile
        self.l1_data_depth = l1_data_depth
        self.l0c_depth = l0c_depth
        self.ub_depth = ub_depth

    @jit
    def _apply_rope_to_q(self, input_buffer, output_buffer, cos_sin_buffer, row_count):
        """Q head unit: 192-col passthrough + tail-64 NeoX rotate-half.

        Identical to the bf16 kernel's epilogue: input is the fp32
        cross-core tile; bf16 RN rounding happens at store. A second load
        at +32 puts this row's right-half values in the low lanes, so every
        rotate-half pair is lane-aligned and computed with 32-lane masked ops.
        """
        with vf(mode="simd"):
            full_lane_mask = full_mask()
            rope_lane_mask = update_mask(ROPE_HALF_DIM, elem_bits=32)[0]

            for row in dsl_range(0, row_count, 1):
                row_offset = row * HEAD_DIM
                for column_block in range(3):
                    element_offset = row_offset + column_block * VECTOR_LANES
                    values = vload(input_buffer, element_offset)
                    vstore_pack(
                        output_buffer,
                        element_offset,
                        vcast(
                            values, BF16, mask=full_lane_mask, rounding=RoundingMode.RN
                        ),
                        full_lane_mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                # Round the fp32 MMAD result to bf16 before the RoPE input:
                # the network materializes qkv as bf16 in GM and the rope
                # kernel reads it back (bf16 -> fp32 compute).
                rope_left = vload(input_buffer, row_offset + NON_ROPE_DIM)
                rope_left = vcast(
                    vcast(
                        rope_left, BF16, mask=full_lane_mask, rounding=RoundingMode.RN
                    ),
                    F32,
                    mask=full_lane_mask,
                )
                rope_right = vload(
                    input_buffer, row_offset + NON_ROPE_DIM + ROPE_HALF_DIM
                )
                rope_right = vcast(
                    vcast(
                        rope_right, BF16, mask=full_lane_mask, rounding=RoundingMode.RN
                    ),
                    F32,
                    mask=full_lane_mask,
                )
                cos_values = vload(cos_sin_buffer, row * ROPE_DIM)  # [cos||sin]
                sin_values = vload(
                    cos_sin_buffer, row * ROPE_DIM + ROPE_HALF_DIM
                )  # [sin|..]
                left_times_cos = vmul(
                    rope_left, cos_values, mask=rope_lane_mask
                )  # rope_left * cos
                right_times_sin = vmul(
                    rope_right, sin_values, mask=rope_lane_mask
                )  # rope_right * sin
                rotated_left = vsub(
                    left_times_cos, right_times_sin, mask=rope_lane_mask
                )
                left_times_sin = vmul(
                    rope_left, sin_values, mask=rope_lane_mask
                )  # rope_left * sin
                right_times_cos = vmul(
                    rope_right, cos_values, mask=rope_lane_mask
                )  # rope_right * cos
                rotated_right = vadd(
                    left_times_sin, right_times_cos, mask=rope_lane_mask
                )
                vstore_pack(
                    output_buffer,
                    row_offset + NON_ROPE_DIM,
                    vcast(
                        rotated_left,
                        BF16,
                        mask=rope_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    rope_lane_mask,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    output_buffer,
                    row_offset + NON_ROPE_DIM + ROPE_HALF_DIM,
                    vcast(
                        rotated_right,
                        BF16,
                        mask=rope_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    rope_lane_mask,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def _apply_norm_rope_to_k(
        self,
        input_buffer,
        output_buffer,
        cos_sin_buffer,
        gamma_buffer,
        row_count,
        inv_head_dim,
        epsilon,
    ):
        """K head unit: per-head fp32 RMSNorm (precise 1/sqrt) + tail-64 RoPE."""
        with vf(mode="simd"):
            full_lane_mask = full_mask()
            rope_lane_mask = update_mask(ROPE_HALF_DIM, elem_bits=32)[0]
            one_f32 = vdups(1.0, F32, mask=full_lane_mask)

            gamma_unpacked_0 = vload_unpack(
                gamma_buffer, 0, unpack_mode=UnpackMode.B16_TO_B32
            )
            gamma_unpacked_1 = vload_unpack(
                gamma_buffer, VECTOR_LANES, unpack_mode=UnpackMode.B16_TO_B32
            )
            gamma_unpacked_2 = vload_unpack(
                gamma_buffer, 2 * VECTOR_LANES, unpack_mode=UnpackMode.B16_TO_B32
            )
            gamma_unpacked_3 = vload_unpack(
                gamma_buffer, NON_ROPE_DIM, unpack_mode=UnpackMode.B16_TO_B32
            )
            # gamma[224:256] re-unpacked at +32 so gamma_rope_right lands in the
            # low lanes (uses the 32-element tail slack on gamma_buffer).
            gamma_unpacked_rope_tail = vload_unpack(
                gamma_buffer,
                NON_ROPE_DIM + ROPE_HALF_DIM,
                unpack_mode=UnpackMode.B16_TO_B32,
            )
            gamma_0 = vcast(gamma_unpacked_0, F32, mask=full_lane_mask)
            gamma_1 = vcast(gamma_unpacked_1, F32, mask=full_lane_mask)
            gamma_2 = vcast(gamma_unpacked_2, F32, mask=full_lane_mask)
            gamma_rope_left = vcast(
                gamma_unpacked_3, F32, mask=full_lane_mask
            )  # lanes 0-31 = gamma[192:224]
            gamma_rope_right = vcast(
                gamma_unpacked_rope_tail, F32, mask=full_lane_mask
            )  # lanes 0-31 = gamma[224:256]

            for row in dsl_range(0, row_count, 1):
                row_offset = row * HEAD_DIM
                # bf16 round-trip before the RMSNorm input: the network
                # materializes qkv as bf16 in GM and the norm kernel reads
                # it back (bf16 -> fp32 accumulate).
                input_0 = vload(input_buffer, row_offset)
                input_0 = vcast(
                    vcast(input_0, BF16, mask=full_lane_mask, rounding=RoundingMode.RN),
                    F32,
                    mask=full_lane_mask,
                )
                input_1 = vload(input_buffer, row_offset + VECTOR_LANES)
                input_1 = vcast(
                    vcast(input_1, BF16, mask=full_lane_mask, rounding=RoundingMode.RN),
                    F32,
                    mask=full_lane_mask,
                )
                input_2 = vload(input_buffer, row_offset + 2 * VECTOR_LANES)
                input_2 = vcast(
                    vcast(input_2, BF16, mask=full_lane_mask, rounding=RoundingMode.RN),
                    F32,
                    mask=full_lane_mask,
                )
                input_rope_left = vload(input_buffer, row_offset + NON_ROPE_DIM)
                input_rope_left = vcast(
                    vcast(
                        input_rope_left,
                        BF16,
                        mask=full_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    F32,
                    mask=full_lane_mask,
                )
                input_rope_right = vload(
                    input_buffer, row_offset + NON_ROPE_DIM + ROPE_HALF_DIM
                )
                input_rope_right = vcast(
                    vcast(
                        input_rope_right,
                        BF16,
                        mask=full_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    F32,
                    mask=full_lane_mask,
                )

                sum_squares_0 = vreduce_sum(
                    vmul(input_0, input_0, mask=full_lane_mask), mask=full_lane_mask
                )
                sum_squares_1 = vreduce_sum(
                    vmul(input_1, input_1, mask=full_lane_mask), mask=full_lane_mask
                )
                sum_squares_2 = vreduce_sum(
                    vmul(input_2, input_2, mask=full_lane_mask), mask=full_lane_mask
                )
                sum_squares_3 = vreduce_sum(
                    vmul(input_rope_left, input_rope_left, mask=full_lane_mask),
                    mask=full_lane_mask,
                )
                sum_squares = vadd(
                    vadd(sum_squares_0, sum_squares_1, mask=full_lane_mask),
                    vadd(sum_squares_2, sum_squares_3, mask=full_lane_mask),
                    mask=full_lane_mask,
                )
                variance = vadds(
                    vmuls(sum_squares, inv_head_dim, mask=full_lane_mask),
                    epsilon,
                    mask=full_lane_mask,
                )
                reciprocal_std = vdiv(
                    one_f32,
                    vsqrt(variance, mask=full_lane_mask),
                    mask=full_lane_mask,
                    precision_mode=True,
                )
                reciprocal_std_broadcast = vdup(reciprocal_std, mask=full_lane_mask)

                normalized_0 = vmul(
                    vmul(input_0, reciprocal_std_broadcast, mask=full_lane_mask),
                    gamma_0,
                    mask=full_lane_mask,
                )
                normalized_1 = vmul(
                    vmul(input_1, reciprocal_std_broadcast, mask=full_lane_mask),
                    gamma_1,
                    mask=full_lane_mask,
                )
                normalized_2 = vmul(
                    vmul(input_2, reciprocal_std_broadcast, mask=full_lane_mask),
                    gamma_2,
                    mask=full_lane_mask,
                )
                vstore_pack(
                    output_buffer,
                    row_offset,
                    vcast(
                        normalized_0,
                        BF16,
                        mask=full_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    full_lane_mask,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    output_buffer,
                    row_offset + VECTOR_LANES,
                    vcast(
                        normalized_1,
                        BF16,
                        mask=full_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    full_lane_mask,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    output_buffer,
                    row_offset + 2 * VECTOR_LANES,
                    vcast(
                        normalized_2,
                        BF16,
                        mask=full_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    full_lane_mask,
                    pack_mode=PackMode.B32_TO_B16,
                )

                # Round both normalized RoPE halves to BF16 before rotation.
                normalized_rope_left = vmul(
                    vmul(
                        input_rope_left, reciprocal_std_broadcast, mask=rope_lane_mask
                    ),
                    gamma_rope_left,
                    mask=rope_lane_mask,
                )
                normalized_rope_right = vmul(
                    vmul(
                        input_rope_right, reciprocal_std_broadcast, mask=rope_lane_mask
                    ),
                    gamma_rope_right,
                    mask=rope_lane_mask,
                )
                normalized_rope_left = vcast(
                    vcast(
                        normalized_rope_left,
                        BF16,
                        mask=rope_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    F32,
                    mask=rope_lane_mask,
                )
                normalized_rope_right = vcast(
                    vcast(
                        normalized_rope_right,
                        BF16,
                        mask=rope_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    F32,
                    mask=rope_lane_mask,
                )
                cos_values = vload(cos_sin_buffer, row * ROPE_DIM)
                sin_values = vload(cos_sin_buffer, row * ROPE_DIM + ROPE_HALF_DIM)
                rotated_left = vsub(
                    vmul(normalized_rope_left, cos_values, mask=rope_lane_mask),
                    vmul(normalized_rope_right, sin_values, mask=rope_lane_mask),
                    mask=rope_lane_mask,
                )
                rotated_right = vadd(
                    vmul(normalized_rope_left, sin_values, mask=rope_lane_mask),
                    vmul(normalized_rope_right, cos_values, mask=rope_lane_mask),
                    mask=rope_lane_mask,
                )
                vstore_pack(
                    output_buffer,
                    row_offset + NON_ROPE_DIM,
                    vcast(
                        rotated_left,
                        BF16,
                        mask=rope_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    rope_lane_mask,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    output_buffer,
                    row_offset + NON_ROPE_DIM + ROPE_HALF_DIM,
                    vcast(
                        rotated_right,
                        BF16,
                        mask=rope_lane_mask,
                        rounding=RoundingMode.RN,
                    ),
                    rope_lane_mask,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def _cast_rows_to_bf16(self, input_buffer, output_buffer, row_count):
        """V / mirror units: fp32 tile -> bf16 RN passthrough."""
        with vf(mode="simd"):
            full_lane_mask = full_mask()
            for row in dsl_range(0, row_count, 1):
                row_offset = row * HEAD_DIM
                for column_block in range(4):
                    element_offset = row_offset + column_block * VECTOR_LANES
                    values = vload(input_buffer, element_offset)
                    vstore_pack(
                        output_buffer,
                        element_offset,
                        vcast(
                            values, BF16, mask=full_lane_mask, rounding=RoundingMode.RN
                        ),
                        full_lane_mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )

    @jit
    def _get_asw_tile_idx(self, work_item, token_tile_count, head_count):
        """Traverse M-first within four-row windows, reversing heads each window."""
        window_rows = min(WORK_WINDOW_ROWS, token_tile_count)
        window_index = work_item // (window_rows * head_count)
        window_start = window_index * window_rows
        rows_here = min(window_rows, token_tile_count - window_start)
        within_window = work_item - window_start * head_count
        token_tile = window_start + within_window % rows_here
        head_forward = within_window // rows_here
        head = head_forward + (window_index % 2) * (head_count - 1 - 2 * head_forward)
        return token_tile, head

    @jit
    def _map_local_to_output_row(
        self, local_row, tokens_per_rank, chunk_tokens_per_rank, chunk_token_offset
    ):
        output_row = local_row
        if const_expr(self.has_chunk):
            output_row = (local_row // chunk_tokens_per_rank) * tokens_per_rank
            output_row = (
                output_row + chunk_token_offset + local_row % chunk_tokens_per_rank
            )
        return output_row

    @jit
    def _store_chunked_output_rows(
        self,
        gm_output,
        ub_source,
        local_row_start,
        row_count,
        head_unit,
        tokens_per_rank,
        chunk_tokens_per_rank,
        chunk_token_offset,
    ):
        """Dense output store: bulk copy while this AIV stays inside one chunk."""
        output_row_start = self._map_local_to_output_row(
            local_row_start, tokens_per_rank, chunk_tokens_per_rank, chunk_token_offset
        )
        # A local range that does not cross a chunk boundary maps to
        # consecutive output rows, so one tile copy replaces row_count copies.
        if local_row_start % chunk_tokens_per_rank + row_count <= chunk_tokens_per_rank:
            output_row_span = gm_output[
                output_row_start : output_row_start + row_count, None
            ]
            mem_copy(
                tile_slice(output_row_span, (self.base_m, HEAD_DIM), (0, head_unit)),
                ub_source[0:row_count, 0:HEAD_DIM],
            )
        else:
            for row_index in range(row_count):
                output_row = self._map_local_to_output_row(
                    local_row_start + row_index,
                    tokens_per_rank,
                    chunk_tokens_per_rank,
                    chunk_token_offset,
                )
                mem_copy(
                    tile_slice(gm_output, (1, HEAD_DIM), (output_row, head_unit)),
                    tile_slice(ub_source, (1, HEAD_DIM), (row_index, 0)),
                )

    @jit
    def _store_dense_output_rows(
        self,
        gm_output,
        ub_source,
        token_pair,
        aiv_subblock_index,
        local_row_start,
        row_count,
        head_index,
        tokens_per_rank,
        chunk_tokens_per_rank,
        chunk_token_offset,
    ):
        """Store one head, preserving the split-M or chunk row mapping."""
        if const_expr(self.has_chunk):
            self._store_chunked_output_rows(
                gm_output,
                ub_source,
                local_row_start,
                row_count,
                head_index,
                tokens_per_rank,
                chunk_tokens_per_rank,
                chunk_token_offset,
            )
        else:
            output_rows = tile_slice(
                gm_output,
                (self.base_m, HEAD_DIM),
                (token_pair, head_index),
            ).partition(dim=0, part_id=aiv_subblock_index)
            mem_copy(output_rows, ub_source[0:row_count, 0:HEAD_DIM])

    @jit
    def _scatter_cache_rows(
        self,
        gm_cache,
        gm_slot_mapping,
        ub_source,
        local_row_start,
        row_count,
        head_index,
        tokens_per_rank,
        chunk_tokens_per_rank,
        chunk_token_offset,
    ):
        """Scatter valid slots using the same row mapping as the dense output."""
        for row_index in range(row_count):
            output_row = self._map_local_to_output_row(
                local_row_start + row_index,
                tokens_per_rank,
                chunk_tokens_per_rank,
                chunk_token_offset,
            )
            cache_slot = gm_slot_mapping[output_row]
            if cache_slot != -1:
                mem_copy(
                    tile_slice(gm_cache, (1, HEAD_DIM), (cache_slot, head_index)),
                    tile_slice(ub_source, (1, HEAD_DIM), (row_index, 0)),
                )

    def __call__(
        self,
        gm_hidden_quant: Tensor,
        gm_hidden_scale: Tensor,
        gm_weight_quant: Tensor,
        gm_weight_scale: Tensor,
        gm_k_gamma: Tensor,
        gm_positions: Tensor,
        gm_cos_sin: Tensor,
        gm_slot_mapping: Tensor,
        gm_q_out: Tensor,
        gm_k_out: Tensor,
        gm_v_out: Tensor,
        gm_mirror_k_out: Tensor,
        gm_mirror_v_out: Tensor,
        gm_k_cache: Tensor,
        gm_v_cache: Tensor,
        epsilon,
        tokens_per_rank: dtypes.int64,
        chunk_tokens_per_rank: dtypes.int64,
        chunk_token_offset: dtypes.int64,
    ):
        # The ABI carries the real FP8/E8M0 dtypes; keep the local names for
        # buffer construction and let each payload keep its own format.
        hidden_quant_fp8_dtype = gm_hidden_quant.dtype
        weight_quant_fp8_dtype = gm_weight_quant.dtype
        cube_block_index = get_block_idx()
        cube_block_count = get_block_num()
        aiv_subblock_index = get_subblock_id()

        input_token_rows = gm_hidden_quant.shape[0]
        token_pair_count = (input_token_rows + self.base_m - 1) // self.base_m
        # Each projection unit contains one head of 256 output columns.
        head_unit_count = gm_weight_quant.shape[0] // HEAD_UNIT_COLS
        q_head_count = gm_q_out.shape[1] // HEAD_DIM
        kv_head_count = gm_k_out.shape[1] // HEAD_DIM
        k_tile_count = (gm_hidden_quant.shape[1] + self.l1_k_tile - 1) // self.l1_k_tile

        # ---- cube-side channels ----
        # L1 data rings: a (256,256) fp8 + b (256,256) fp8 == 256KB at
        # depth 2; the paired e8m0 scale rings ride alongside in the ZN/NZ
        # layouts the l1->l0 MX binding consumes.
        l1_hidden_quant = RingBuffer(
            MemLoc.L1,
            hidden_quant_fp8_dtype,
            make_layout((self.base_m, self.l1_k_tile)),
            depth=self.l1_data_depth,
        )
        l1_weight_quant = RingBuffer(
            MemLoc.L1,
            weight_quant_fp8_dtype,
            make_layout((HEAD_UNIT_COLS, self.l1_k_tile)),
            depth=self.l1_data_depth,
        )
        # NZ (scale, M) storage is exposed as a ZN (M, scale) Tensor view.
        # Bind the actual token rows before transposing so tails keep compact pitches.
        l1_hidden_scale = RingBuffer(
            MemLoc.L1, E8M0, make_layout((L1_SCALE_COUNT, self.base_m)), depth=2
        )
        l1_weight_scale = RingBuffer(
            MemLoc.L1, E8M0, make_layout((L1_SCALE_COUNT, HEAD_UNIT_COLS)), depth=2
        )
        l0_hidden_quant = RingBuffer(
            MemLoc.L0A,
            hidden_quant_fp8_dtype,
            make_layout((self.base_m, L0_K_TILE)),
            depth=2,
        )
        l0_weight_quant = RingBuffer(
            MemLoc.L0B,
            weight_quant_fp8_dtype,
            make_layout((HEAD_UNIT_COLS, L0_K_TILE)),
            depth=2,
        )
        # 256x256 f32 == 256KB L0C: depth1.
        l0_result = RingBuffer(
            MemLoc.L0C,
            F32,
            make_layout((self.base_m, HEAD_UNIT_COLS)),
            depth=self.l0c_depth,
        )
        # ---- cross-core handoff (fp32 tile) ----
        fixpipe_ub = RingBuffer(
            MemLoc.UB,
            F32,
            make_layout((self.aiv_tile_rows + 1, HEAD_UNIT_COLS)),
            depth=self.ub_depth,
        )
        # ---- vec-side channels ----
        output_ub = RingBuffer(
            MemLoc.UB,
            BF16,
            make_layout((self.aiv_tile_rows, HEAD_DIM)),
            depth=self.ub_depth,
        )
        cos_sin_ub = RingBuffer(
            MemLoc.UB,
            F32,
            make_layout((self.aiv_tile_rows + 1, ROPE_DIM)),
            depth=self.ub_depth,
        )
        gamma_ub = RingBuffer(
            MemLoc.UB, BF16, make_layout((HEAD_DIM + ROPE_HALF_DIM,)), depth=1
        )

        # Resident K-RMSNorm weight: loaded once, consumed by every item's
        # K-unit epilogue on this AIV. Both GM and UB gamma are rank one;
        # keep the tail slack for the final vector unpack.
        gamma_write = gamma_ub.next_write()
        mem_copy(
            tile_slice(gamma_write, (HEAD_DIM,), (0,)),
            tile_slice(gm_k_gamma, (HEAD_DIM,), (0,)),
        )

        # Keep grid-stride core assignment, but visit (token tile, head) in
        # four-row ASW windows. Adjacent windows reverse the head direction.
        total_work_items = token_pair_count * head_unit_count
        for work_item in range(cube_block_index, total_work_items, cube_block_count):
            token_pair, head_unit = self._get_asw_tile_idx(
                work_item, token_pair_count, head_unit_count
            )

            valid_token_rows = input_token_rows - token_pair * self.base_m
            if valid_token_rows > self.base_m:
                valid_token_rows = self.base_m

            # Balanced split-M plan for this (possibly tail-clipped) pair
            # tile; a trailing odd tile clips to its real row count.
            probe_tile = tile_slice(
                gm_hidden_quant, (self.base_m, self.l1_k_tile), (token_pair, 0)
            )
            rows_for_this_aiv = cv_valid_extent(
                probe_tile, axis=0, part_id=aiv_subblock_index
            )
            aiv_row_offset = aiv_subblock_index * (valid_token_rows - rows_for_this_aiv)
            local_row_start = token_pair * self.base_m + aiv_row_offset
            # A trailing subblock may own zero rows (e.g. M == 1 splits
            # 1/0); clamp so the scalar position load stays in bounds even
            # though every per-row loop below is empty.
            if local_row_start >= input_token_rows:
                local_row_start = input_token_rows - 1

            # Only Q/K units read RoPE tables; V and mirror units bypass the DMA.
            cos_sin_write = cos_sin_ub.next_write()
            if head_unit < q_head_count + kv_head_count:
                cos_sin_table_rows = gm_cos_sin.shape[0]
                if const_expr(self.positions_contiguous):
                    # Prefill: one run of self.aiv_tile_rows consecutive table rows.
                    output_row_start = self._map_local_to_output_row(
                        local_row_start,
                        tokens_per_rank,
                        chunk_tokens_per_rank,
                        chunk_token_offset,
                    )
                    position_base = gm_positions[output_row_start]
                    if rows_for_this_aiv > 0:
                        if position_base + self.aiv_tile_rows <= cos_sin_table_rows:
                            # Fast path: one bulk (<=128, 64) DMA.
                            mem_copy(
                                tile_slice(
                                    cos_sin_write,
                                    (self.aiv_tile_rows, ROPE_DIM),
                                    (0, 0),
                                ),
                                gm_cos_sin[
                                    position_base : position_base + self.aiv_tile_rows,
                                    0:ROPE_DIM,
                                ],
                            )
                        else:
                            # Caller precondition: used positions are within
                            # [0, cos_sin_table_rows), and contiguous mode advances
                            # by one within this local range. Load only valid rows
                            # when the full AIV tile would exceed the table.
                            for row_index in range(rows_for_this_aiv):
                                mem_copy(
                                    tile_slice(
                                        cos_sin_write, (1, ROPE_DIM), (row_index, 0)
                                    ),
                                    tile_slice(
                                        gm_cos_sin,
                                        (1, ROPE_DIM),
                                        (position_base + row_index, 0),
                                    ),
                                )
                else:
                    # Decode: arbitrary per-token positions (each request's own
                    # last position). One 1x64 gather per token.
                    for row_index in range(rows_for_this_aiv):
                        output_row = self._map_local_to_output_row(
                            local_row_start + row_index,
                            tokens_per_rank,
                            chunk_tokens_per_rank,
                            chunk_token_offset,
                        )
                        position = gm_positions[output_row]
                        mem_copy(
                            tile_slice(cos_sin_write, (1, ROPE_DIM), (row_index, 0)),
                            tile_slice(gm_cos_sin, (1, ROPE_DIM), (position, 0)),
                        )

            # ---- cube: K-loop MXFP8 MMAD accumulation for this head unit ----
            # GM -> L1 uses ND2NZ payloads and paired E8M0 scales; the L0
            # views select per-32-group dequantization with FP32 accumulation.
            # Tensor-access dependency analysis orders DMA, FIXPIPE, VF and
            # output stores, including ring-slot reuse across work items.
            l0_result_accumulator = from_tensor(
                l0_result.next_write(),
                layout=make_layout((valid_token_rows, HEAD_UNIT_COLS)),
                data_format="nz",
            )
            scale_window_count = (
                k_tile_count + self.scale_k_tiles - 1
            ) // self.scale_k_tiles
            for scale_window_index in range(scale_window_count):
                scale_k_tiles_here = min(
                    self.scale_k_tiles,
                    k_tile_count - scale_window_index * self.scale_k_tiles,
                )
                scale_pair_offset = scale_window_index
                hidden_scale_tile = tile_slice(
                    gm_hidden_scale,
                    (self.base_m, L1_PACKED_SCALE_PAIRS, 2),
                    (token_pair, scale_pair_offset, 0),
                )
                weight_scale_tile = tile_slice(
                    gm_weight_scale,
                    (HEAD_UNIT_COLS, L1_PACKED_SCALE_PAIRS, 2),
                    (head_unit, scale_pair_offset, 0),
                )
                mem_copy(
                    from_tensor(
                        l1_hidden_scale.next_write(),
                        layout=make_layout((L1_SCALE_COUNT, valid_token_rows)),
                        data_format="nz",
                    ).permute(1, 0),
                    hidden_scale_tile.flatten(-2, -1),
                )
                mem_copy(
                    l1_weight_scale.next_write(),
                    weight_scale_tile.flatten(-2, -1).permute(1, 0),
                )
                l1_hidden_scale_slot = from_tensor(
                    l1_hidden_scale.next_read(),
                    layout=make_layout((L1_SCALE_COUNT, valid_token_rows)),
                    data_format="nz",
                ).permute(1, 0)
                l1_weight_scale_slot = l1_weight_scale.next_read()

                for l1_k_index_in_scale in range(scale_k_tiles_here):
                    l1_k_index = (
                        scale_window_index * self.scale_k_tiles + l1_k_index_in_scale
                    )
                    hidden_quant_tile = tile_slice(
                        gm_hidden_quant,
                        (self.base_m, self.l1_k_tile),
                        (token_pair, l1_k_index),
                    )
                    weight_quant_tile = tile_slice(
                        gm_weight_quant,
                        (HEAD_UNIT_COLS, self.l1_k_tile),
                        (head_unit, l1_k_index),
                    )
                    # l2_cache_ctl=1 (normal): W tiles stay resident in L2
                    # across token tiles.
                    mem_copy(
                        from_tensor(
                            l1_hidden_quant.next_write(),
                            layout=make_layout((valid_token_rows, self.l1_k_tile)),
                            data_format="nz",
                        ),
                        hidden_quant_tile,
                        options=copy_options(l2_cache_ctl=1),
                    )
                    mem_copy(
                        l1_weight_quant.next_write(),
                        weight_quant_tile,
                        options=copy_options(l2_cache_ctl=1),
                    )
                    l1_hidden_quant_slot = from_tensor(
                        l1_hidden_quant.next_read(),
                        layout=make_layout((valid_token_rows, self.l1_k_tile)),
                        data_format="nz",
                    )
                    l1_weight_quant_slot = l1_weight_quant.next_read()
                    scale_offset = l1_k_index_in_scale * self.l0_k_tiles_per_l1
                    remaining_k = gm_hidden_quant.shape[1] - l1_k_index * self.l1_k_tile
                    l0_k_tiles_here = min(
                        self.l0_k_tiles_per_l1,
                        (remaining_k + L0_K_TILE - 1) // L0_K_TILE,
                    )
                    for l0_k_index in range(l0_k_tiles_here):
                        mem_copy(
                            from_tensor(
                                l0_hidden_quant.next_write(),
                                layout=make_layout((valid_token_rows, L0_K_TILE)),
                                data_format="nz",
                            ),
                            tile_slice(
                                l1_hidden_quant_slot,
                                (self.base_m, L0_K_TILE),
                                (0, l0_k_index),
                            ),
                            options=copy_options(
                                mx_scale=tile_slice(
                                    l1_hidden_scale_slot,
                                    (self.base_m, L0_SCALE_BYTES),
                                    (0, scale_offset + l0_k_index),
                                )
                            ),
                        )
                        mem_copy(
                            l0_weight_quant.next_write(),
                            tile_slice(
                                l1_weight_quant_slot,
                                (HEAD_UNIT_COLS, L0_K_TILE),
                                (0, l0_k_index),
                            ),
                            options=copy_options(
                                mx_scale=tile_slice(
                                    l1_weight_scale_slot,
                                    (L0_SCALE_BYTES, HEAD_UNIT_COLS),
                                    (scale_offset + l0_k_index, 0),
                                )
                            ),
                        )
                        l0_hidden_quant_slot = from_tensor(
                            l0_hidden_quant.next_read(),
                            layout=make_layout((valid_token_rows, L0_K_TILE)),
                            data_format="nz",
                        )
                        l0_weight_quant_slot = l0_weight_quant.next_read()
                        global_k_tile_index = (
                            l1_k_index * self.l0_k_tiles_per_l1 + l0_k_index
                        )
                        l0_result_accumulator = matmul(
                            l0_result_accumulator,
                            l0_hidden_quant_slot,
                            l0_weight_quant_slot,
                            init=(global_k_tile_index == 0),
                            mx_mode=True,
                        )

            # ---- handoff: L0C -> per-AIV UB halves ----
            # Partition the actual L0C Tensor for dual-destination FIXPIPE:
            # each AIV receives its balanced half of the
            # tail-clipped (valid_token_rows, HEAD_UNIT_COLS) tile directly in its own UB.
            fixpipe_write = fixpipe_ub.next_write()
            mem_copy(
                tile_slice(fixpipe_write, (self.aiv_tile_rows, HEAD_UNIT_COLS), (0, 0)),
                from_tensor(
                    l0_result.next_read(),
                    layout=make_layout((valid_token_rows, HEAD_UNIT_COLS)),
                    data_format="nz",
                ).partition(dim=0),
            )

            # ---- vec epilogue, specialized per unit ----
            fixpipe_read = fixpipe_ub.next_read()
            cos_sin_read = cos_sin_ub.next_read()
            output_write = output_ub.next_write()
            if head_unit < q_head_count:
                self._apply_rope_to_q(
                    fixpipe_read, output_write, cos_sin_read, rows_for_this_aiv
                )
                output_read = output_ub.next_read()
                self._store_dense_output_rows(
                    gm_q_out,
                    output_read,
                    token_pair,
                    aiv_subblock_index,
                    local_row_start,
                    rows_for_this_aiv,
                    head_unit,
                    tokens_per_rank,
                    chunk_tokens_per_rank,
                    chunk_token_offset,
                )
            elif head_unit < q_head_count + kv_head_count:
                self._apply_norm_rope_to_k(
                    fixpipe_read,
                    output_write,
                    cos_sin_read,
                    gamma_write,
                    rows_for_this_aiv,
                    INV_HEAD_DIM,
                    epsilon,
                )
                output_read = output_ub.next_read()
                # The tile drain and the slot scatter are two readers of one
                # output_ub slot; both consume the same VF-written epoch.
                self._store_dense_output_rows(
                    gm_k_out,
                    output_read,
                    token_pair,
                    aiv_subblock_index,
                    local_row_start,
                    rows_for_this_aiv,
                    head_unit - q_head_count,
                    tokens_per_rank,
                    chunk_tokens_per_rank,
                    chunk_token_offset,
                )
                self._scatter_cache_rows(
                    gm_k_cache,
                    gm_slot_mapping,
                    output_read,
                    local_row_start,
                    rows_for_this_aiv,
                    head_unit - q_head_count,
                    tokens_per_rank,
                    chunk_tokens_per_rank,
                    chunk_token_offset,
                )
            elif head_unit < q_head_count + 2 * kv_head_count:
                self._cast_rows_to_bf16(fixpipe_read, output_write, rows_for_this_aiv)
                output_read = output_ub.next_read()
                self._store_dense_output_rows(
                    gm_v_out,
                    output_read,
                    token_pair,
                    aiv_subblock_index,
                    local_row_start,
                    rows_for_this_aiv,
                    head_unit - q_head_count - kv_head_count,
                    tokens_per_rank,
                    chunk_tokens_per_rank,
                    chunk_token_offset,
                )
                self._scatter_cache_rows(
                    gm_v_cache,
                    gm_slot_mapping,
                    output_read,
                    local_row_start,
                    rows_for_this_aiv,
                    head_unit - q_head_count - kv_head_count,
                    tokens_per_rank,
                    chunk_tokens_per_rank,
                    chunk_token_offset,
                )
            else:
                self._cast_rows_to_bf16(fixpipe_read, output_write, rows_for_this_aiv)
                output_read = output_ub.next_read()
                gm_mirror_output = (
                    gm_mirror_k_out
                    if head_unit < q_head_count + 3 * kv_head_count
                    else gm_mirror_v_out
                )
                mirror_head_index = (
                    head_unit - q_head_count - 2 * kv_head_count
                ) % kv_head_count
                if const_expr(self.has_mirror):
                    self._store_dense_output_rows(
                        gm_mirror_output,
                        output_read,
                        token_pair,
                        aiv_subblock_index,
                        local_row_start,
                        rows_for_this_aiv,
                        mirror_head_index,
                        tokens_per_rank,
                        chunk_tokens_per_rank,
                        chunk_token_offset,
                    )
                else:
                    # Plain-layer stand-ins have one row; the mirror units
                    # never execute at runtime, so per-row runtime coords
                    # (which pass static verification) suffice.
                    for row_index in range(rows_for_this_aiv):
                        mem_copy(
                            tile_slice(
                                gm_mirror_output,
                                (1, HEAD_DIM),
                                (local_row_start + row_index, 0),
                            ),
                            tile_slice(output_read, (1, HEAD_DIM), (row_index, 0)),
                        )


class FusedQuantQkvProjNormRopeCache:
    """One Host entry selects between two statically configured kernels."""

    def __init__(self, has_mirror, positions_contiguous, has_chunk):
        self.kernel_base128 = FusedQuantQkvProjNormRopeCacheKernel(
            has_mirror, positions_contiguous, has_chunk, **TILING_PROFILES["base128"]
        )
        self.kernel_base256 = FusedQuantQkvProjNormRopeCacheKernel(
            has_mirror, positions_contiguous, has_chunk, **TILING_PROFILES["base256"]
        )

    @jit
    def _compute_launch_block_count(
        self, input_rows, weight_rows, base_m, cube_core_count
    ):
        token_tiles = (input_rows + base_m - 1) // base_m
        return min(token_tiles * (weight_rows // HEAD_UNIT_COLS), cube_core_count)

    @host
    def run(
        self,
        hidden_quant,
        hidden_scale,
        weight_quant,
        weight_scale,
        k_gamma,
        positions,
        cos_sin,
        slot_mapping,
        q_out,
        k_out,
        v_out,
        mirror_k_out,
        mirror_v_out,
        k_cache,
        v_cache,
        epsilon: dtypes.float32,
        tokens_per_rank: dtypes.int64,
        chunk_tokens_per_rank: dtypes.int64,
        chunk_token_offset: dtypes.int64,
        cube_core_count: dtypes.int32,
        *,
        has_mirror: Constexpr[bool],
        positions_contiguous: Constexpr[bool],
        has_chunk: Constexpr[bool],
    ):
        kernel_args = (
            hidden_quant,
            hidden_scale,
            weight_quant,
            weight_scale,
            k_gamma,
            positions,
            cos_sin,
            slot_mapping,
            q_out,
            k_out,
            v_out,
            mirror_k_out,
            mirror_v_out,
            k_cache,
            v_cache,
            epsilon,
            tokens_per_rank,
            chunk_tokens_per_rank,
            chunk_token_offset,
        )
        if hidden_quant.shape[0] <= SMALL_M_PROFILE_MAX_ROWS:
            blocks = self._compute_launch_block_count(
                hidden_quant.shape[0],
                weight_quant.shape[0],
                TILING_PROFILES["base128"]["base_m"],
                cube_core_count,
            )
            self.kernel_base128[blocks](*kernel_args)
        else:
            blocks = self._compute_launch_block_count(
                hidden_quant.shape[0],
                weight_quant.shape[0],
                TILING_PROFILES["base256"]["base_m"],
                cube_core_count,
            )
            self.kernel_base256[blocks](*kernel_args)


def _compile_source(
    has_mirror,
    positions_contiguous,
    has_chunk,
    hidden_quant_dtype,
    weight_quant_dtype,
):
    """Input rows, output rows, cache capacity and table capacity stay dynamic in one ABI."""
    if any(
        type(value) is not bool
        for value in (has_mirror, positions_contiguous, has_chunk)
    ):
        raise TypeError("factory choices must be bool")
    if hidden_quant_dtype not in FP8_DTYPES or weight_quant_dtype not in FP8_DTYPES:
        raise TypeError(
            "hidden_quant/weight_quant dtype must be dtypes.float8_e4m3fn or dtypes.float8_e5m2"
        )
    input_rows = Dim("M", min=1)
    output_rows = Dim("T", min=1) if has_chunk else input_rows
    cache_rows = Dim("S", min=1)
    cos_sin_rows = Dim("P", min=1)
    weight_rows = Dim("N", min=768, multiple_of=256)
    hidden_size = Dim("K", min=256, multiple_of=256)
    q_width = Dim("QW", min=256, multiple_of=256)
    kv_width = Dim("KVW", min=256, multiple_of=256)
    mirror_rows = output_rows if has_mirror else 1
    bf16_dtype, float32_dtype, int64_dtype = (
        dtypes.bfloat16,
        dtypes.float32,
        dtypes.int64,
    )
    host_runner = FusedQuantQkvProjNormRopeCache(
        has_mirror, positions_contiguous, has_chunk
    )
    return cannbotdsl.compile(
        host_runner.run,
        TensorSpec((input_rows, hidden_size), hidden_quant_dtype),
        TensorSpec((input_rows, hidden_size // 64, 2), E8M0),
        TensorSpec((weight_rows, hidden_size), weight_quant_dtype),
        TensorSpec((weight_rows, hidden_size // 64, 2), E8M0),
        TensorSpec((256,), bf16_dtype),
        TensorSpec((output_rows,), int64_dtype),
        TensorSpec((cos_sin_rows, 64), float32_dtype),
        TensorSpec((output_rows,), int64_dtype),
        TensorSpec((output_rows, q_width), bf16_dtype),
        TensorSpec((output_rows, kv_width), bf16_dtype),
        TensorSpec((output_rows, kv_width), bf16_dtype),
        TensorSpec((mirror_rows, kv_width), bf16_dtype),
        TensorSpec((mirror_rows, kv_width), bf16_dtype),
        TensorSpec((cache_rows, kv_width), bf16_dtype),
        TensorSpec((cache_rows, kv_width), bf16_dtype),
        float32_dtype,
        int64_dtype,
        int64_dtype,
        int64_dtype,
        dtypes.int32,
        has_mirror=has_mirror,
        positions_contiguous=positions_contiguous,
        has_chunk=has_chunk,
    )


@aot.compile_cache
def compile_fused_quant_qkv(
    has_mirror,
    positions_contiguous,
    has_chunk,
    hidden_quant_dtype,
    weight_quant_dtype,
):
    return _compile_source(
        has_mirror,
        positions_contiguous,
        has_chunk,
        hidden_quant_dtype,
        weight_quant_dtype,
    )


@aot.export("fused_quant_qkv_proj_norm_rope_cache")
def export_fused_quant_qkv():
    for has_mirror, positions_contiguous, has_chunk in product((False, True), repeat=3):
        for hidden_quant_dtype, weight_quant_dtype in product(FP8_DTYPES, repeat=2):
            compile_fused_quant_qkv(
                has_mirror,
                positions_contiguous,
                has_chunk,
                hidden_quant_dtype,
                weight_quant_dtype,
            )
