# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Recurrent-state backward and its shared Cube/Vector implementations."""

from __future__ import annotations
from cannbotdsl import (
    Buffer,
    Constexpr,
    MemLoc,
    RoundingMode,
    const_expr,
    dtypes,
    from_tensor,
    make_layout,
)
from cannbotdsl.lang.control_flow import range as drange
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
    vload_unpack,
    vlts,
    vmadd,
    vmem_bar,
    vmul,
    vmuls,
    vpack,
    vreinterpret,
    vreinterpret_lanes,
    vselect,
    vstore,
    vstore_pack,
    vstore_strided,
    vsub,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import tile_slice

if __package__:
    from .flash_kda_grad_buffer import (
        StateMatmulResources,
        CHUNK_SIZE,
        CUBE_TILE_ROWS,
        D_HALF,
        KEY_DIM,
        LANES,
        NZ_BLOCK_WIDTH,
        SUPPORTED_HEAD_DIM,
        VALUE_DIM,
        _gradient_slot,
        _FixedScratchSlot,
        _MatrixSlotView,
        _cube_ring,
        _full_token_tile,
        _load_zero_padded,
        _sync_cube_vector_handoff,
        _tnd_chunk_index,
        _token_rows_tile,
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
    from . import flash_kda_grad_metadata as _metadata
else:
    from flash_kda_grad_buffer import (
        StateMatmulResources,
        CHUNK_SIZE,
        CUBE_TILE_ROWS,
        D_HALF,
        KEY_DIM,
        LANES,
        NZ_BLOCK_WIDTH,
        SUPPORTED_HEAD_DIM,
        VALUE_DIM,
        _gradient_slot,
        _FixedScratchSlot,
        _MatrixSlotView,
        _cube_ring,
        _full_token_tile,
        _load_zero_padded,
        _sync_cube_vector_handoff,
        _tnd_chunk_index,
        _token_rows_tile,
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
    import flash_kda_grad_metadata as _metadata


# Recurrent-state implementations used by scheduling templates
# ---------------------------------------------------------------------------


class BackwardCastVector:
    """Round FP32 matrix operands to BF16 before Cube products.

    rows=1 supports scalar row transfers. The batched specialization uses
    rows=16 to amortize DMA without changing casts.
    """

    def __init__(self, width: int, rows: int = 1):
        self.width = width
        self.rows = rows
        self.source = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((rows, width)), depth=1
        )
        low_storage = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((2 * rows, width)), depth=1
        ).next_write()
        self.destination = _FixedScratchSlot(low_storage[:rows, None])
        self.residual_destination = _FixedScratchSlot(
            low_storage[rows : 2 * rows, None]
        )

    @jit
    def run(self, gm_source, gm_destination):
        if const_expr(gm_source.dtype == dtypes.bfloat16):
            for row in range(
                get_subblock_id(), gm_source.shape[-2], get_subblock_dim()
            ):
                low = tile_slice(self.destination.next_write(), (1, self.width), (0, 0))
                mem_copy(low, tile_slice(gm_source, (1, self.width), (row, 0)))
                mem_copy(
                    tile_slice(gm_destination, (1, self.width), (row, 0)),
                    tile_slice(self.destination.next_read(), (1, self.width), (0, 0)),
                )
        elif self.rows > 1 and gm_source.shape[-2] % self.rows == 0:
            for group in range(
                get_subblock_id(), gm_source.shape[-2] // self.rows, get_subblock_dim()
            ):
                mem_copy(
                    self.source.next_write(),
                    tile_slice(gm_source, (self.rows, self.width), (group, 0)),
                )
                self._cast(self.source.next_read(), self.destination.next_write())
                mem_copy(
                    tile_slice(gm_destination, (self.rows, self.width), (group, 0)),
                    self.destination.next_read(),
                )
        else:
            for row in range(
                get_subblock_id(), gm_source.shape[-2], get_subblock_dim()
            ):
                mem_copy(
                    tile_slice(self.source.next_write(), (1, self.width), (0, 0)),
                    tile_slice(gm_source, (1, self.width), (row, 0)),
                )
                self._cast(
                    tile_slice(self.source.next_read(), (1, self.width), (0, 0)),
                    tile_slice(self.destination.next_write(), (1, self.width), (0, 0)),
                )
                mem_copy(
                    tile_slice(gm_destination, (1, self.width), (row, 0)),
                    tile_slice(self.destination.next_read(), (1, self.width), (0, 0)),
                )

    @jit
    def run_split(self, gm_source, gm_high, gm_residual):
        if self.rows > 1 and gm_source.shape[-2] % self.rows == 0:
            for group in range(
                get_subblock_id(), gm_source.shape[-2] // self.rows, get_subblock_dim()
            ):
                mem_copy(
                    self.source.next_write(),
                    tile_slice(gm_source, (self.rows, self.width), (group, 0)),
                )
                self._split(
                    self.source.next_read(),
                    self.destination.next_write(),
                    self.residual_destination.next_write(),
                )
                mem_copy(
                    tile_slice(gm_high, (self.rows, self.width), (group, 0)),
                    self.destination.next_read(),
                )
                mem_copy(
                    tile_slice(gm_residual, (self.rows, self.width), (group, 0)),
                    self.residual_destination.next_read(),
                )
        else:
            for row in range(
                get_subblock_id(), gm_source.shape[-2], get_subblock_dim()
            ):
                mem_copy(
                    tile_slice(self.source.next_write(), (1, self.width), (0, 0)),
                    tile_slice(gm_source, (1, self.width), (row, 0)),
                )
                self._split(
                    tile_slice(self.source.next_read(), (1, self.width), (0, 0)),
                    tile_slice(self.destination.next_write(), (1, self.width), (0, 0)),
                    tile_slice(
                        self.residual_destination.next_write(), (1, self.width), (0, 0)
                    ),
                )
                mem_copy(
                    tile_slice(gm_high, (1, self.width), (row, 0)),
                    tile_slice(self.destination.next_read(), (1, self.width), (0, 0)),
                )
                mem_copy(
                    tile_slice(gm_residual, (1, self.width), (row, 0)),
                    tile_slice(
                        self.residual_destination.next_read(), (1, self.width), (0, 0)
                    ),
                )

    @jit
    def _cast(self, source, destination):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for row in range(source.shape[-2]):
                for half in range(self.width // D_HALF):
                    source_offset = (
                        row * source.physical_layout.stride[0] + half * D_HALF
                    )
                    destination_offset = (
                        row * destination.physical_layout.stride[0] + half * D_HALF
                    )
                    fp32 = vload(source, source_offset)
                    bf16 = vcast(fp32, dtypes.bfloat16, mask=full)
                    vstore_pack(
                        destination,
                        destination_offset,
                        bf16,
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )

    @jit
    def _split(self, source, high_destination, residual_destination):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for row in range(source.shape[-2]):
                for half in range(self.width // D_HALF):
                    source_offset = (
                        row * source.physical_layout.stride[0] + half * D_HALF
                    )
                    high_offset = (
                        row * high_destination.physical_layout.stride[0] + half * D_HALF
                    )
                    residual_offset = (
                        row * residual_destination.physical_layout.stride[0]
                        + half * D_HALF
                    )
                    fp32 = vload(source, source_offset)
                    high = vcast(fp32, dtypes.bfloat16, mask=full)
                    restored = vcast(high, dtypes.float32, mask=full)
                    residual = vcast(
                        vsub(fp32, restored, mask=full), dtypes.bfloat16, mask=full
                    )
                    vstore_pack(
                        high_destination,
                        high_offset,
                        high,
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                    vstore_pack(
                        residual_destination,
                        residual_offset,
                        residual,
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )


class _StateUpdate:
    """Identical recurrent-state update used by both scheduling templates."""

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


class PersistentStateBackward(_StateUpdate):
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
        resources = StateMatmulResources(value_shard_rows)
        self.d_v_new_state_matmul = StatePipelineMatmul(
            value_shard_rows, True, resources
        )
        self.d_state_value_matmul = StatePipelineMatmul(
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
        vector = StateGradientVector()
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


class StagedStateBackward(_StateUpdate):
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
        resources = StateMatmulResources(value_shard_rows)
        self.d_v_new_state_matmul = StatePipelineMatmul(
            value_shard_rows, True, resources
        )
        self.d_state_value_matmul = StatePipelineMatmul(
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
        vector = StateGradientVector()
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
                matrix_source = gm_Aqk[matrix_batch, value_head, None, None]
                output_source = gm_output_gradient[
                    storage_batch, value_head, None, None
                ]
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


class FullChunkStateBackward:
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
        self.saved_low = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, D_HALF)), depth=1
        ).next_write()
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
        self.output_vector = StateOutputVector()

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
        saved_source = stage_two_gm_chunk_states[
            saved_batch, head, saved_chunk, None, None
        ]
        if const_expr(saved_source.dtype == dtypes.bfloat16):
            for value_row in range(VALUE_DIM):
                mem_copy(
                    self.saved_low,
                    tile_slice(
                        saved_source, (1, D_HALF), (value_row, get_subblock_id())
                    ),
                )
                _promote_state_row(
                    tile_slice(self.saved, (1, D_HALF), (value_row, 0)), self.saved_low
                )
        else:
            mem_copy(
                self.saved,
                tile_slice(saved_source, (VALUE_DIM, D_HALF), (0, get_subblock_id())),
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
                vector = StateGradientVector(double_window=True)
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


# Per-chunk state-gradient data flow
# ---------------------------------------------------------------------------


@jit
def _prepare_track_chunk(
    gm_Aqk,
    gm_qg,
    gm_v_new,
    gm_output_gradient,
    gm_task_metadata,
    gm_d_v_new,
    gm_d_Aqk,
    gm_state_gradients,
    gm_masked_qg,
    gm_masked_v_new,
    gm_masked_output_gradient,
    padded_layout: dtypes.int64,
    selected_task: dtypes.int64,
    selected_chunk: dtypes.int64,
    staged: Constexpr[bool] = False,
):
    """Prepare the current owner chunk before its recurrent state update.

    QTerm occupies the first of two state slots. The reverse scan consumes it
    and publishes the outgoing state gradient in the second slot. Persistent
    execution reuses these slots per core; staged retains them for every chunk.
    """
    matmul = StateGradientMatmul()
    vector = StateGradientVector()
    output_vector = StateOutputVector()
    block_idx = get_block_idx()
    task = selected_task
    task_record = HEADER_WORDS + task * TASK_WORDS
    gradient_slot = _gradient_slot(gm_task_metadata, task, selected_chunk, staged)
    seq_idx = gm_task_metadata[task_record + TASK_SEQUENCE]
    compact_token_start = gm_task_metadata[task_record + TASK_COMPACT_TOKEN_START]
    chunks = gm_task_metadata[task_record + TASK_CHUNKS]
    seq_len = gm_task_metadata[task_record + TASK_LENGTH]
    value_head = gm_task_metadata[task_record + TASK_VALUE_HEAD]
    storage_batch = seq_idx
    source_token_start = 0
    if not padded_layout:
        storage_batch = 0
        source_token_start = compact_token_start
    qg_source = gm_qg[storage_batch, value_head, None, None]
    v_new_source = gm_v_new[storage_batch, value_head, None, None]
    output_source = gm_output_gradient[storage_batch, value_head, None, None]
    d_v_new_destination = gm_d_v_new[0, 0, None, None]
    chunk = selected_chunk
    if chunk < chunks:
        valid_rows = seq_len - chunk * CHUNK_SIZE
        if valid_rows > CHUNK_SIZE:
            valid_rows = CHUNK_SIZE
        source_row_start = source_token_start + chunk * CHUNK_SIZE
        matrix_batch = dtypes.int64(storage_batch)
        matrix_source = gm_Aqk[matrix_batch, value_head, None, None]
        # Keep-index GM slices preserve singleton batch/head axes.
        # The last two dimensions are always token and feature.
        direct_gm = 0
        if (
            (qg_source.shape[-2] >= CHUNK_SIZE)
            and (qg_source.physical_layout.stride[-1] == 1)
            and (v_new_source.shape[-2] >= CHUNK_SIZE)
            and (v_new_source.physical_layout.stride[-1] == 1)
            and (output_source.shape[-2] >= CHUNK_SIZE)
            and (output_source.physical_layout.stride[-1] == 1)
            and (matrix_source.shape[-2] >= CHUNK_SIZE)
            and (matrix_source.physical_layout.stride[-1] == 1)
            and (d_v_new_destination.shape[-2] >= CHUNK_SIZE)
            and (d_v_new_destination.physical_layout.stride[-1] == 1)
        ):
            if valid_rows == CHUNK_SIZE:
                direct_gm = 1
        if direct_gm == 1:
            # Full tiles need no row masks. Cube can read the original
            # strided GM view, avoiding GM->UB->scratch-GM round trips.
            qg_tile = _full_token_tile(qg_source, source_row_start, KEY_DIM)
            value_tile = _full_token_tile(v_new_source, source_row_start, VALUE_DIM)
            output_tile = _full_token_tile(
                output_source, source_token_start + chunk * CHUNK_SIZE, VALUE_DIM
            )
            d_Aqk = tile_slice(
                gm_d_Aqk,
                (CHUNK_SIZE, CHUNK_SIZE),
                (gradient_slot, 0),
            )
            qterm = tile_slice(
                gm_state_gradients,
                (VALUE_DIM, KEY_DIM),
                (2 * gradient_slot, 0),
            )
            # Publish dAqk directly to the two AIVs. Its masking can run
            # while Cube computes QTerm and loads the next chunk.
            matmul.compute_d_Aqk(
                output_tile, value_tile, output_vector.matrix_in.next_write()
            )
            output_vector.mask_matrix(d_Aqk, valid_rows, get_subblock_id())
            matmul.compute_state_outer(output_tile, qg_tile, qterm)
        else:
            row = get_subblock_id() * (CHUNK_SIZE // 2)
            valid = max(0, min(CHUNK_SIZE // 2, valid_rows - row))
            vector.copy_window(
                qg_source,
                source_row_start + row,
                tile_slice(
                    gm_masked_qg,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                row,
                valid,
                KEY_DIM,
                False,
            )
            vector.copy_window(
                v_new_source,
                source_row_start + row,
                tile_slice(
                    gm_masked_v_new,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                row,
                valid,
                KEY_DIM,
                False,
            )
            vector.copy_window(
                output_source,
                source_token_start + chunk * CHUNK_SIZE + row,
                tile_slice(
                    gm_masked_output_gradient,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                row,
                valid,
                KEY_DIM,
                False,
            )
            _sync_cube_vector_handoff(0)
            d_Aqk = tile_slice(
                gm_d_Aqk,
                (CHUNK_SIZE, CHUNK_SIZE),
                (gradient_slot, 0),
            )
            qterm = tile_slice(
                gm_state_gradients,
                (VALUE_DIM, KEY_DIM),
                (2 * gradient_slot, 0),
            )
            matmul.compute_d_Aqk(
                tile_slice(
                    gm_masked_output_gradient,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_masked_v_new,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                output_vector.matrix_in.next_write(),
            )
            # Each AIV masks and writes its own 32 d_Aqk rows.
            output_vector.mask_matrix(d_Aqk, valid_rows, get_subblock_id())
            matmul.compute_state_outer(
                tile_slice(
                    gm_masked_output_gradient,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_masked_qg, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (block_idx, 0)
                ),
                qterm,
            )
            _sync_cube_vector_handoff(1)
            _sync_cube_vector_handoff(3)


@jit
def _finish_track_chunk(
    gm_v_new,
    gm_chunk_states,
    gm_output_gradient,
    gm_task_metadata,
    gm_d_v_new,
    gm_d_qg,
    gm_d_kg,
    gm_d_w,
    gm_d_gate_last,
    gm_state_gradients,
    gm_masked_v_new,
    gm_masked_output_gradient,
    gm_d_v_new_scratch,
    gm_d_v_new_low,
    gm_d_kg_scratch,
    gm_d_qg_scratch,
    gm_d_w_scratch,
    gm_state_high,
    gm_state_residual,
    gm_saved_state_high,
    scale: dtypes.float32,
    padded_layout: dtypes.int64,
    selected_task: dtypes.int64,
    selected_chunk: dtypes.int64,
    staged: Constexpr[bool] = False,
):
    """Finish the current chunk using the two live per-core state slots."""
    matmul = StateGradientMatmul()
    vector = StateGradientVector()
    pre_vector = StateReductionVector(rows=16)
    post_vector = StateReductionResultVector()
    state_cast = BackwardCastVector(KEY_DIM, rows=16)
    saved_state_cast = BackwardCastVector(KEY_DIM, rows=16)
    d_v_new_cast = BackwardCastVector(VALUE_DIM, rows=16)
    block_idx = get_block_idx()
    task = selected_task
    task_record = HEADER_WORDS + task * TASK_WORDS
    gradient_slot = _gradient_slot(gm_task_metadata, task, selected_chunk, staged)
    seq_idx = gm_task_metadata[task_record + TASK_SEQUENCE]
    compact_token_start = gm_task_metadata[task_record + TASK_COMPACT_TOKEN_START]
    chunks = gm_task_metadata[task_record + TASK_CHUNKS]
    seq_len = gm_task_metadata[task_record + TASK_LENGTH]
    value_head = gm_task_metadata[task_record + TASK_VALUE_HEAD]
    storage_batch = seq_idx
    source_token_start = 0
    if not padded_layout:
        storage_batch = 0
        source_token_start = compact_token_start
    v_new_source = gm_v_new[storage_batch, value_head, None, None]
    output_source = gm_output_gradient[storage_batch, value_head, None, None]
    d_v_new_source = gm_d_v_new[0, 0, None, None]
    d_qg_destination = gm_d_qg[0, 0, None, None]
    d_kg_destination = gm_d_kg[0, 0, None, None]
    d_w_destination = gm_d_w[0, 0, None, None]
    d_gate_destination = gm_d_gate_last[0, 0, None, None]
    chunk = selected_chunk
    if chunk < chunks:
        valid_rows = seq_len - chunk * CHUNK_SIZE
        if valid_rows > CHUNK_SIZE:
            valid_rows = CHUNK_SIZE
        source_row_start = source_token_start + chunk * CHUNK_SIZE
        compact_row_start = gradient_slot * CHUNK_SIZE
        # Keep-index GM slices preserve singleton batch/head axes.
        # The last two dimensions are always token and feature.
        direct_gm = 0
        if (
            (v_new_source.shape[-2] >= CHUNK_SIZE)
            and (v_new_source.physical_layout.stride[-1] == 1)
            and (output_source.shape[-2] >= CHUNK_SIZE)
            and (output_source.physical_layout.stride[-1] == 1)
            and (d_v_new_source.shape[-2] >= CHUNK_SIZE)
            and (d_v_new_source.physical_layout.stride[-1] == 1)
            and (d_qg_destination.shape[-2] >= CHUNK_SIZE)
            and (d_qg_destination.physical_layout.stride[-1] == 1)
            and (d_kg_destination.shape[-2] >= CHUNK_SIZE)
            and (d_kg_destination.physical_layout.stride[-1] == 1)
            and (d_w_destination.shape[-2] >= CHUNK_SIZE)
            and (d_w_destination.physical_layout.stride[-1] == 1)
        ):
            if valid_rows == CHUNK_SIZE:
                direct_gm = 1
        if direct_gm == 1:
            # Keep the FP32 state/high-low numerical path unchanged.
            # Only bypass row staging and copy completed gradients to
            # their final compact GM tiles directly.
            value_tile = _full_token_tile(v_new_source, source_row_start, VALUE_DIM)
            output_tile = _full_token_tile(
                output_source, source_token_start + chunk * CHUNK_SIZE, VALUE_DIM
            )
            d_value_tile = _full_token_tile(
                d_v_new_source, compact_row_start, VALUE_DIM
            )
            d_qg_tile = _full_token_tile(d_qg_destination, compact_row_start, KEY_DIM)
            d_kg_tile = _full_token_tile(d_kg_destination, compact_row_start, KEY_DIM)
            d_w_tile = _full_token_tile(d_w_destination, compact_row_start, KEY_DIM)
            current_state = tile_slice(
                gm_state_gradients,
                (VALUE_DIM, KEY_DIM),
                (2 * gradient_slot + 1, 0),
            )
            state_batch = dtypes.int64(storage_batch)
            state_chunk = dtypes.int64(chunk)
            if not padded_layout:
                state_batch = 0
                state_chunk = _tnd_chunk_index(compact_token_start, seq_idx, chunk)
            saved_state = gm_chunk_states[
                state_batch, value_head, state_chunk, None, None
            ]
            state_cast.run_split(
                current_state,
                tile_slice(
                    gm_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_state_residual,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
            )
            saved_state_cast.run(
                saved_state,
                tile_slice(
                    gm_saved_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
            )
            d_v_new_cast.run(
                d_value_tile,
                tile_slice(
                    gm_d_v_new_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (block_idx, 0)
                ),
            )
            d_gate_last = tile_slice(
                d_gate_destination, (1, KEY_DIM), (gradient_slot, 0)
            )
            _sync_cube_vector_handoff(0)
            pre_vector.reduce_gate_last(
                current_state, saved_state, d_gate_last, get_subblock_id()
            )
            matmul.compute_token_state(
                output_tile,
                tile_slice(
                    gm_saved_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                post_vector.result.next_write(),
                True,
            )
            post_vector.finish_token_gradient(d_qg_tile, scale)
            matmul.compute_token_state_split(
                value_tile,
                tile_slice(
                    gm_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_state_residual,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                d_kg_tile,
            )
            matmul.compute_token_state(
                tile_slice(
                    gm_d_v_new_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (block_idx, 0)
                ),
                tile_slice(
                    gm_saved_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                post_vector.result.next_write(),
                True,
            )
            post_vector.finish_token_gradient(d_w_tile, -1.0)
            _sync_cube_vector_handoff(3)
        else:
            row = get_subblock_id() * (CHUNK_SIZE // 2)
            valid = max(0, min(CHUNK_SIZE // 2, valid_rows - row))
            vector.copy_window(
                v_new_source,
                source_row_start + row,
                tile_slice(
                    gm_masked_v_new,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                row,
                valid,
                KEY_DIM,
                False,
            )
            vector.copy_window(
                output_source,
                source_token_start + chunk * CHUNK_SIZE + row,
                tile_slice(
                    gm_masked_output_gradient,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                row,
                valid,
                KEY_DIM,
                False,
            )
            vector.copy_window(
                d_v_new_source,
                compact_row_start + row,
                tile_slice(
                    gm_d_v_new_scratch,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                row,
                valid,
                KEY_DIM,
                True,
            )
            current_state = tile_slice(
                gm_state_gradients,
                (VALUE_DIM, KEY_DIM),
                (2 * gradient_slot + 1, 0),
            )
            state_batch = dtypes.int64(storage_batch)
            state_chunk = dtypes.int64(chunk)
            if not padded_layout:
                state_batch = 0
                state_chunk = _tnd_chunk_index(compact_token_start, seq_idx, chunk)
            saved_state = gm_chunk_states[
                state_batch, value_head, state_chunk, None, None
            ]
            state_cast.run_split(
                current_state,
                tile_slice(
                    gm_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_state_residual,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
            )
            saved_state_cast.run(
                saved_state,
                tile_slice(
                    gm_saved_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
            )
            d_v_new_cast.run(
                tile_slice(
                    gm_d_v_new_scratch,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_d_v_new_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (block_idx, 0)
                ),
            )
            d_gate_last = tile_slice(
                d_gate_destination, (1, KEY_DIM), (gradient_slot, 0)
            )
            _sync_cube_vector_handoff(0)
            pre_vector.reduce_gate_last(
                current_state, saved_state, d_gate_last, get_subblock_id()
            )
            matmul.compute_token_state(
                tile_slice(
                    gm_masked_output_gradient,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_saved_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                post_vector.result.next_write(),
                True,
            )
            post_vector.finish_token_gradient(
                tile_slice(gm_d_qg_scratch, (CHUNK_SIZE, KEY_DIM), (block_idx, 0)),
                scale,
            )
            matmul.compute_token_state_split(
                tile_slice(
                    gm_masked_v_new,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_state_residual,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                tile_slice(
                    gm_d_kg_scratch,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
            )
            matmul.compute_token_state(
                tile_slice(
                    gm_d_v_new_low, (CHUNK_SIZE, SUPPORTED_HEAD_DIM), (block_idx, 0)
                ),
                tile_slice(
                    gm_saved_state_high,
                    (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                post_vector.result.next_write(),
                True,
            )
            post_vector.finish_token_gradient(
                tile_slice(gm_d_w_scratch, (CHUNK_SIZE, KEY_DIM), (block_idx, 0)),
                -1.0,
            )
            _sync_cube_vector_handoff(2)
            row = get_subblock_id() * (CHUNK_SIZE // 2)
            valid = max(0, min(CHUNK_SIZE // 2, valid_rows - row))
            compact_row = compact_row_start + row
            vector.copy_window(
                tile_slice(
                    gm_d_qg_scratch,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                row,
                d_qg_destination,
                compact_row,
                valid,
                KEY_DIM,
                True,
            )
            vector.copy_window(
                tile_slice(
                    gm_d_kg_scratch,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                row,
                d_kg_destination,
                compact_row,
                valid,
                KEY_DIM,
                True,
            )
            vector.copy_window(
                tile_slice(
                    gm_d_w_scratch,
                    (CHUNK_SIZE, SUPPORTED_HEAD_DIM),
                    (block_idx, 0),
                ),
                row,
                d_w_destination,
                compact_row,
                valid,
                KEY_DIM,
                True,
            )
            _sync_cube_vector_handoff(3)


# Shared state-gradient Vector and Cube operations
# ---------------------------------------------------------------------------


class StateReductionVector:
    """Reduce each AIV's 64 key channels in the original value-row order."""

    def __init__(self, rows: int = 1):
        self.rows = rows
        storage = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((2 * rows + 2, D_HALF)), depth=1
        ).next_write()
        self.state = _FixedScratchSlot(storage[:rows, None])
        self.saved_state = _FixedScratchSlot(storage[rows : 2 * rows, None])
        self.saved_low = RingBuffer(
            MemLoc.UB, dtypes.bfloat16, make_layout((1, D_HALF)), depth=1
        ).next_write()
        self.gate_accumulator = storage[2 * rows : 2 * rows + 1, None]
        self.gate_result = _FixedScratchSlot(storage[2 * rows + 1 : 2 * rows + 2, None])

    @jit
    def reduce_gate_last(
        self, gm_d_state, gm_saved_state, gm_d_gate_last, subblock_idx
    ):
        self._clear_gate()
        # Groups and their inner rows are visited strictly from value 0 to
        # 127. Splitting ownership remains along key channels, never values.
        for group in range(VALUE_DIM // self.rows):
            mem_copy(
                self.state.next_write(),
                tile_slice(gm_d_state, (self.rows, D_HALF), (group, subblock_idx)),
            )
            if const_expr(gm_saved_state.dtype == dtypes.bfloat16):
                for row in range(self.rows):
                    mem_copy(
                        self.saved_low,
                        tile_slice(
                            gm_saved_state,
                            (1, D_HALF),
                            (group * self.rows + row, subblock_idx),
                        ),
                    )
                    dst = tile_slice(
                        self.saved_state.next_write(), (1, D_HALF), (row, 0)
                    )
                    _promote_state_row(dst, self.saved_low)
            else:
                mem_copy(
                    self.saved_state.next_write(),
                    tile_slice(
                        gm_saved_state, (self.rows, D_HALF), (group, subblock_idx)
                    ),
                )
            self._accumulate_gate(
                self.state.next_read(),
                self.saved_state.next_read(),
                gm_saved_state.dtype == dtypes.float32,
            )
        mem_copy(self.gate_result.next_write(), self.gate_accumulator)
        mem_copy(
            tile_slice(gm_d_gate_last, (1, D_HALF), (0, subblock_idx)),
            self.gate_result.next_read(),
        )

    @jit
    def _clear_gate(self):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            vstore(self.gate_accumulator, 0, zero, full)
            vmem_bar("vst_vld")

    @jit
    def _accumulate_gate(self, state, saved_state, round_saved: Constexpr[bool]):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            total = vload(self.gate_accumulator, 0)
            for row in range(self.rows):
                # Each vmadd still produces FP32 in exactly the original
                # row order; keeping that FP32 value in a register avoids
                # a redundant UB store/reload inside this DMA group.
                saved = vload(saved_state, row * saved_state.physical_layout.stride[0])
                if const_expr(round_saved):
                    saved = vcast(
                        vcast(saved, dtypes.bfloat16, mask=full),
                        dtypes.float32,
                        mask=full,
                    )
                total = vmadd(
                    vload(state, row * state.physical_layout.stride[0]),
                    saved,
                    total,
                    mask=full,
                )
            vstore(self.gate_accumulator, 0, total, full)
            vmem_bar("vst_vld")


class StateReductionResultVector:
    """Consume each Cube result as soon as it reaches the owning AIV."""

    def __init__(self):
        self.result = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE // 2, KEY_DIM)), depth=2
        )

    @jit
    def finish_token_gradient(self, gm_destination, scale: dtypes.float32):
        result = self.result.next_read()
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for half in range(CHUNK_SIZE):
                offset = half * D_HALF
                vstore(
                    result, offset, vmuls(vload(result, offset), scale, mask=full), full
                )
        mem_copy(
            tile_slice(
                gm_destination, (CHUNK_SIZE // 2, KEY_DIM), (get_subblock_id(), 0)
            ),
            result,
        )


class StateOutputVector:
    """Mask the AIV-owned half of the causal output matrix."""

    def __init__(self):
        self.matrix_in = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((CHUNK_SIZE // 2, CHUNK_SIZE)),
            depth=2,
        )
        self.matrix_out = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((CHUNK_SIZE // 2, CHUNK_SIZE)),
            depth=1,
        )

    def mask_matrix(self, gm_matrix, valid_rows, subblock_idx):
        # tile_slice offsets count 32-row tiles, not individual rows.
        owned_rows = tile_slice(
            gm_matrix, (CHUNK_SIZE // 2, CHUNK_SIZE), (subblock_idx, 0)
        )
        self._mask_matrix(
            self.matrix_in.next_read(),
            self.matrix_out.next_write(),
            valid_rows,
            subblock_idx,
        )
        mem_copy(owned_rows, self.matrix_out.next_read())

    @jit
    def _mask_matrix(self, source, destination, valid_rows, subblock_idx):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            for row in range(CHUNK_SIZE // 2):
                global_row = subblock_idx * (CHUNK_SIZE // 2) + row
                result = zero
                row_valid = vlts(
                    vdups(global_row, dtypes.int32, mask=full), valid_rows, mask=full
                )
                row_valid_result = result
                causal, _ = update_mask(global_row + 1, elem_bits=32)
                result = vselect(
                    vload(source, row * CHUNK_SIZE), zero, cond_mask=causal
                )
                result = vselect(result, row_valid_result, cond_mask=row_valid)
                vstore(destination, row * CHUNK_SIZE, result, full)


class StateGradientVector:
    """Load tail-safe cache rows and publish compact stage-two gradients."""

    def __init__(self, double_window=False):
        # The full-track tail benefits from overlapping window transfers.
        # Generic/empty tasks retain the original channel resources.
        self.low_window = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((CHUNK_SIZE // 2, KEY_DIM)),
            depth=2 if double_window else 1,
        )
        self.fp_window = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE // 2, KEY_DIM)), depth=1
        )
        self.state_row = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((1, KEY_DIM)), depth=1
        )

    @jit
    def copy_window(
        self,
        gm_source,
        source_row,
        gm_destination,
        destination_row,
        valid_rows,
        width: Constexpr[int],
        fp32: Constexpr[bool],
        column_limit: dtypes.int64 = CHUNK_SIZE,
    ):
        rows = CHUNK_SIZE // 2
        if const_expr(fp32):
            write = from_tensor(
                self.fp_window.next_write(), layout=make_layout((rows, width))
            )
        else:
            write = from_tensor(
                self.low_window.next_write(), layout=make_layout((rows, width))
            )
        if valid_rows > 0:
            if const_expr(gm_source.rank == 4):
                window = gm_source[
                    None, None, source_row : source_row + valid_rows, None
                ]
            else:
                window = gm_source[source_row : source_row + valid_rows, None]
            source = tile_slice(window, (rows, width), (0, 0))[:valid_rows, None]
            _load_zero_padded(write, source)
        else:
            with vf(mode="simd"):
                mask = create_mask(pattern="all")
                zero = vdups(0.0, dtypes.float32, mask=mask)
                for lane_block in range(rows * width // LANES):
                    if const_expr(fp32):
                        vstore(write, lane_block * LANES, zero, mask)
                    else:
                        vstore_pack(
                            write,
                            lane_block * LANES,
                            vcast(zero, dtypes.bfloat16, mask=mask),
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
        if const_expr(fp32):
            read = from_tensor(
                self.fp_window.next_read(), layout=make_layout((rows, width))
            )
        else:
            read = from_tensor(
                self.low_window.next_read(), layout=make_layout((rows, width))
            )
        if const_expr(not fp32 and width == CHUNK_SIZE):
            if column_limit < CHUNK_SIZE:
                # Padded cache columns are not public inputs. Mask them before
                # Cube products so even poisoned columns cannot enter dH.
                with vf(mode="simd"):
                    full = create_mask(pattern="all")
                    live, _ = update_mask(column_limit, elem_bits=32)
                    zero = vdups(0.0, dtypes.float32, mask=full)
                    for row in range(rows):
                        value = vcast(
                            vload_unpack(
                                read, row * width, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=full,
                        )
                        value = vselect(value, zero, cond_mask=live)
                        vstore_pack(
                            read,
                            row * width,
                            vcast(value, dtypes.bfloat16, mask=full),
                            full,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    vmem_bar("vst_vld")
        mem_copy(
            tile_slice(gm_destination, (rows, width), (destination_row // rows, 0)),
            read,
        )


class StateGradientMatmul:
    """Reuse L0 operands across the eight Stage 2 contractions."""

    def __init__(self):
        self.token_left_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, VALUE_DIM)), depth=1
        )
        self.token_right_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, VALUE_DIM)), depth=1
        )
        self.state_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((VALUE_DIM, KEY_DIM)), depth=1
        )
        self.token_l0a = _cube_ring(
            MemLoc.L0A, dtypes.bfloat16, make_layout((CHUNK_SIZE, VALUE_DIM)), depth=1
        )
        self.token_transposed_l0a = _cube_ring(
            MemLoc.L0A, dtypes.bfloat16, make_layout((VALUE_DIM, CHUNK_SIZE)), depth=1
        )
        self.token_l0b = _cube_ring(
            MemLoc.L0B, dtypes.bfloat16, make_layout((CHUNK_SIZE, VALUE_DIM)), depth=1
        )
        # Transposed L0 operands use explicit logical axes in DSL 0.8.
        self.token_transposed_l0b = _cube_ring(
            MemLoc.L0B, dtypes.bfloat16, make_layout((VALUE_DIM, CHUNK_SIZE)), depth=1
        )
        self.state_l0b = _cube_ring(
            MemLoc.L0B, dtypes.bfloat16, make_layout((VALUE_DIM, KEY_DIM)), depth=1
        )
        # Each method publishes its result before the next contraction starts.
        # Reusing one slot also keeps Cube buffer IDs within the DSL 0.8 limit.
        result_storage = _cube_ring(
            MemLoc.L0C, dtypes.float32, make_layout((VALUE_DIM, KEY_DIM)), depth=1
        )
        self.token_result_l0c = _MatrixSlotView(result_storage, (CHUNK_SIZE, VALUE_DIM))
        self.matrix_result_l0c = _MatrixSlotView(
            result_storage, (CHUNK_SIZE, CHUNK_SIZE)
        )
        self.state_result_l0c = _MatrixSlotView(result_storage, (VALUE_DIM, KEY_DIM))

    def compute_d_Aqk(self, gm_d_output, gm_v_new, gm_result):
        mem_copy(self.token_left_l1.next_write(), gm_d_output)
        mem_copy(self.token_right_l1.next_write(), gm_v_new)
        mem_copy(self.token_l0a.next_write(), self.token_left_l1.next_read())
        mem_copy(self.token_l0b.next_write(), self.token_right_l1.next_read())
        matmul(
            self.matrix_result_l0c.next_write(),
            self.token_l0a.next_read(),
            self.token_l0b.next_read(),
            init=True,
        )
        mem_copy(gm_result, self.matrix_result_l0c.next_read().partition(dim=0))

    def compute_token_state(
        self, gm_token, gm_state_high, gm_result, transpose_state: bool
    ):
        mem_copy(self.token_left_l1.next_write(), gm_token)
        mem_copy(self.state_l1.next_write(), gm_state_high)
        mem_copy(self.token_l0a.next_write(), self.token_left_l1.next_read())
        mem_copy(
            self.state_l0b.next_write(),
            self.state_l1.next_read().permute((1, 0))
            if transpose_state
            else self.state_l1.next_read(),
        )
        matmul(
            self.token_result_l0c.next_write(),
            self.token_l0a.next_read(),
            self.state_l0b.next_read(),
            init=True,
        )
        mem_copy(gm_result, self.token_result_l0c.next_read().partition(dim=0))

    def compute_token_state_split(
        self, gm_token, gm_state_high, gm_state_residual, gm_result
    ):
        """Accumulate BF16 high and residual state products in FP32 L0C."""
        result = self.token_result_l0c.next_write()
        mem_copy(self.token_left_l1.next_write(), gm_token)
        mem_copy(self.state_l1.next_write(), gm_state_high)
        mem_copy(self.token_l0a.next_write(), self.token_left_l1.next_read())
        mem_copy(self.state_l0b.next_write(), self.state_l1.next_read().permute((1, 0)))
        token_left = self.token_l0a.next_read()
        matmul(result, token_left, self.state_l0b.next_read(), init=True)
        mem_copy(self.state_l1.next_write(), gm_state_residual)
        mem_copy(self.state_l0b.next_write(), self.state_l1.next_read().permute((1, 0)))
        matmul(result, token_left, self.state_l0b.next_read(), init=False)
        mem_copy(gm_result, self.token_result_l0c.next_read())

    def compute_state_outer(self, gm_token_gradient, gm_saved_token, gm_result):
        mem_copy(self.token_left_l1.next_write(), gm_token_gradient)
        mem_copy(self.token_right_l1.next_write(), gm_saved_token)
        mem_copy(
            self.token_transposed_l0a.next_write(),
            self.token_left_l1.next_read().permute((1, 0)),
        )
        mem_copy(
            self.token_transposed_l0b.next_write(),
            self.token_right_l1.next_read().permute((1, 0)),
        )
        matmul(
            self.state_result_l0c.next_write(),
            self.token_transposed_l0a.next_read(),
            self.token_transposed_l0b.next_read(),
            init=True,
        )
        mem_copy(gm_result, self.state_result_l0c.next_read())


@jit
def _cast_state_operand_nz(dst, src, scratch, rows, cols):
    """Cast one FP32 operand to BF16 NZ, preserving its logical axes."""
    src_stride = src.physical_layout.stride[0]
    nz_stride = dst.physical_layout.stride[0]
    if const_expr(cols == NZ_BLOCK_WIDTH and src_stride == cols):
        # One NZ column block has the same element order as packed ND.
        # Cast complete consecutive rows together; do not exchange any axes.
        with vf(mode="simd"):
            for offset in range(0, rows * cols, D_HALF):
                mask, _ = update_mask(min(D_HALF, rows * cols - offset), elem_bits=32)
                value = vcast(
                    vload(src, offset),
                    dtypes.bfloat16,
                    mask=mask,
                    rounding=RoundingMode.RN,
                )
                vstore_pack(dst, offset, value, mask, pack_mode=PackMode.B32_TO_B16)
            vmem_bar("vst_vld")
    else:
        with vf(mode="simd"):
            for row in range(rows):
                for col in range(0, cols, D_HALF):
                    mask, _ = update_mask(min(D_HALF, cols - col), elem_bits=32)
                    value = vcast(
                        vload(src, row * src_stride + col),
                        dtypes.bfloat16,
                        mask=mask,
                        rounding=RoundingMode.RN,
                    )
                    vstore_pack(
                        scratch, col, value, mask, pack_mode=PackMode.B32_TO_B16
                    )
                vmem_bar("vst_vld")
                for col in range(0, cols, KEY_DIM):
                    mask16, _ = update_mask(min(KEY_DIM, cols - col), elem_bits=16)
                    vstore_strided(
                        dst,
                        row * NZ_BLOCK_WIDTH + (col // NZ_BLOCK_WIDTH) * nz_stride,
                        vload(scratch, col),
                        mask16,
                        block_stride=nz_stride // NZ_BLOCK_WIDTH,
                        repeat_stride=0,
                    )
                vmem_bar("vst_vld")


class StatePipelineMatmul:
    """Compute Kg @ G.T or dR.T @ W with token-major dR throughout UB."""

    def __init__(self, value_rows, value_product, resources):
        self.value_product = value_product
        self.operand_shape = (
            (value_rows, KEY_DIM) if value_product else (CHUNK_SIZE, value_rows)
        )
        self.result_shape = (
            (CHUNK_SIZE, value_rows) if value_product else (value_rows, KEY_DIM)
        )
        self.operand_l1 = resources.operand_l1
        self.token_gm_l1 = resources.token_gm_l1
        self.operand_nz = resources.operand_nz
        self.row_scratch = resources.row_scratch
        self.result_to_vector = resources.result_to_vector
        self.left_l0a = _cube_ring(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout(
                (CHUNK_SIZE, KEY_DIM) if value_product else (value_rows, CHUNK_SIZE)
            ),
            depth=1,
        )
        self.right_l0b = _cube_ring(
            MemLoc.L0B,
            dtypes.bfloat16,
            make_layout(self.operand_shape if value_product else (KEY_DIM, CHUNK_SIZE)),
            depth=1,
        )
        self.result_l0c = _cube_ring(
            MemLoc.L0C, dtypes.float32, make_layout(self.result_shape), depth=1
        )

    @jit
    def _publish_left(self, operand):
        _cast_state_operand_nz(
            self.operand_nz,
            operand,
            self.row_scratch,
            self.operand_shape[0] // 2,
            self.operand_shape[1],
        )
        mem_copy(
            self.operand_l1.next_write()[
                : self.operand_shape[0], : self.operand_shape[1]
            ].partition(dim=0, part_id=get_subblock_id()),
            tile_slice(
                self.operand_nz,
                (self.operand_shape[0] // 2, self.operand_shape[1]),
                (0, 0),
            ),
        )

    @jit
    def _compute(self, operand, token):
        if const_expr(self.value_product):
            # Token-major dR: [L,K] @ [V,K].T -> [L,V].
            mem_copy(self.left_l0a.next_write(), token)
            mem_copy(self.right_l0b.next_write(), operand)
        else:
            # Transpose dR at Cube load, never in Vector: [L,V].T @ [L,K].
            mem_copy(self.left_l0a.next_write(), operand.permute((1, 0)))
            mem_copy(self.right_l0b.next_write(), token.permute((1, 0)))
        matmul(
            self.result_l0c.next_write(),
            self.left_l0a.next_read(),
            self.right_l0b.next_read(),
            init=True,
        )
        mem_copy(
            from_tensor(
                self.result_to_vector.next_write(),
                layout=make_layout((self.result_shape[0] // 2, self.result_shape[1])),
            ),
            from_tensor(
                self.result_l0c.next_read(), layout=make_layout(self.result_shape)
            ).partition(dim=0),
        )

    @jit
    def next_read(self):
        return from_tensor(
            self.result_to_vector.next_read(),
            layout=make_layout((self.result_shape[0] // 2, self.result_shape[1])),
        )


@jit
def _promote_state_row(dst, src):
    with vf(mode="simd"):
        full = create_mask(pattern="all")
        for column in range(dst.shape[-1] // D_HALF):
            offset = column * D_HALF
            value = vload_unpack(src, offset, unpack_mode=UnpackMode.B16_TO_B32)
            vstore(dst, offset, vcast(value, dtypes.float32, mask=full), full)
