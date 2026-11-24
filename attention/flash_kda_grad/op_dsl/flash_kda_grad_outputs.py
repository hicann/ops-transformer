# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Input-gradient accumulation, activation backward and single-chunk output handling."""

from __future__ import annotations
from cannbotdsl import (
    Buffer,
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
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
)
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    create_mask,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup,
    vdups,
    veqs,
    vexp,
    vload,
    vload_broadcast,
    vload_unpack,
    vlog,
    vlts,
    vmaxs,
    vmem_bar,
    vmins,
    vmul,
    vmuls,
    vreduce_sum,
    vselect,
    vstore,
    vstore_first,
    vstore_pack,
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
    from .flash_kda_grad_state import _promote_state_row
    from .flash_kda_grad_metadata import _owner_loop_bounds, _scheduled_owner
    from .flash_kda_grad_buffer import (
        BF16_ELEMENT_BYTES,
        CAUSAL_BATCH_ROWS,
        CHUNK_SIZE,
        D_HALF,
        FP32_ELEMENT_BYTES,
        GATE_BATCH_ROWS,
        GATE_SCALAR_TRANSFER_WIDTH,
        LANES,
        SUPPORTED_HEAD_DIM,
        VALUE_DIM,
        ChunkWorkspace,
        _sync_cube_vector_handoff,
        _full_token_tile,
        _load_zero_padded,
        _tnd_chunk_index,
        _token_rows_tile,
    )
    from .flash_kda_grad_metadata import (
        ABI_VERSION,
        HEADER_ABI_VERSION,
        HEADER_MAGIC,
        HEADER_STATUS,
        HEADER_WORDS,
        MAGIC,
        MAX_AIC_CORES,
        STATUS_OK,
        TASK_COMPACT_TOKEN_START,
        TASK_LENGTH,
        TASK_SEQUENCE,
        TASK_VALUE_HEAD,
        TASK_WORDS,
    )
else:
    from flash_kda_grad_state import _promote_state_row
    from flash_kda_grad_metadata import _owner_loop_bounds, _scheduled_owner
    from flash_kda_grad_buffer import (
        BF16_ELEMENT_BYTES,
        CAUSAL_BATCH_ROWS,
        CHUNK_SIZE,
        D_HALF,
        FP32_ELEMENT_BYTES,
        GATE_BATCH_ROWS,
        GATE_SCALAR_TRANSFER_WIDTH,
        LANES,
        SUPPORTED_HEAD_DIM,
        VALUE_DIM,
        ChunkWorkspace,
        _sync_cube_vector_handoff,
        _full_token_tile,
        _load_zero_padded,
        _tnd_chunk_index,
        _token_rows_tile,
    )
    from flash_kda_grad_metadata import (
        ABI_VERSION,
        HEADER_ABI_VERSION,
        HEADER_MAGIC,
        HEADER_STATUS,
        HEADER_WORDS,
        MAGIC,
        MAX_AIC_CORES,
        STATUS_OK,
        TASK_COMPACT_TOKEN_START,
        TASK_LENGTH,
        TASK_SEQUENCE,
        TASK_VALUE_HEAD,
        TASK_WORDS,
    )


@jit
def _gate_parameter_row(dA, head):
    # Width 16 is the internal deterministic ABI marker; width 8 retains the
    # original ABI. Each AIV issues its atomic updates into its own row, so
    # there is a single producer per bin, including the sensitivity epilogue.
    if const_expr(dA.shape[1] == 2 * GATE_SCALAR_TRANSFER_WIDTH):
        heads = dA.shape[0] // (1 + 2 * get_block_num())
        return head + (1 + 2 * get_block_idx() + get_subblock_id()) * heads
    return head


@jit
def _reduce_deterministic_parameter_bins(partials, alog_partials, total, alog, writers):
    # Visit AIV bins in a fixed order, independently of core completion order.
    with vf(mode="simd"):
        full = create_mask(pattern="all")
        scalar, _ = update_mask(8, elem_bits=32)
        low_sum = vdups(0.0, dtypes.float32, mask=full)
        high_sum = vdups(0.0, dtypes.float32, mask=full)
        alog_sum = vdups(0.0, dtypes.float32, mask=scalar)
        for writer in range(writers):
            low_sum = vadd(low_sum, vload(partials, writer * 128), mask=full)
            high_sum = vadd(high_sum, vload(partials, writer * 128 + 64), mask=full)
            alog_sum = vadd(alog_sum, vload(alog_partials, writer * 8), mask=scalar)
        vstore(total, 0, low_sum, full)
        vstore(total, 64, high_sum, full)
        vstore(alog, 0, alog_sum, scalar)


@jit
def _finish_deterministic_gate_gradients(dA, dBias, fused_gate, has_bias):
    if const_expr(dA.shape[1] == 2 * GATE_SCALAR_TRANSFER_WIDTH):
        _sync_cube_vector_handoff(0, grid=True)
        onchip_rewind(reset_sync_id=True)
        if fused_gate:
            total = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 128)))
            partials = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((2 * MAX_AIC_CORES, 128))
            )
            alog = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, 8)))
            alog_partials = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((2 * MAX_AIC_CORES, 8))
            )
            heads = dtypes.int64(dA.shape[0] // (1 + 2 * get_block_num()))
            writers = dtypes.int64(2 * get_block_num())
            first = 2 * get_block_idx() + get_subblock_id()
            for head in range(first, heads, 2 * get_block_num()):
                for writer in range(writers):
                    row = (writer + 1) * heads + head
                    mem_copy(
                        tile_slice(alog_partials, (1, 8), (writer, 0)),
                        tile_slice(dA, (1, 8), (row, 0)),
                    )
                    mem_copy(
                        tile_slice(partials, (1, 128), (writer, 0)),
                        tile_slice(dBias, (1, 128), (row, 0)),
                    )
                _reduce_deterministic_parameter_bins(
                    partials, alog_partials, total, alog, writers
                )
                mem_copy(tile_slice(dA, (1, 8), (head, 0)), alog)
                if has_bias:
                    mem_copy(tile_slice(dBias, (1, 128), (head, 0)), total)


@jit
def _zero_parameter_tile(zeros):
    with vf(mode="simd"):
        full = create_mask(pattern="all")
        zero = vdups(0.0, dtypes.float32, mask=full)
        for row in range(16):
            vstore(zeros, row * 128, zero, full)
            vstore(zeros, row * 128 + 64, zero, full)


@jit
def _initialize_backward_gradients(
    dA, dBias, dht, fused_gate: dtypes.int64, has_dht: dtypes.int64
):
    if has_dht == 0:
        zeros = Buffer(MemLoc.UB, dtypes.float32, make_layout((16, SUPPORTED_HEAD_DIM)))
        _zero_parameter_tile(zeros)
        heads = dht.shape[1]
        first = get_block_idx() * get_subblock_dim() + get_subblock_id()
        step = get_block_num() * get_subblock_dim()
        for task in range(first, dht.shape[0] * heads, step):
            state = dht[task // heads, task % heads, None, None]
            for tile in range(SUPPORTED_HEAD_DIM // 16):
                mem_copy(tile_slice(state, (16, SUPPORTED_HEAD_DIM), (tile, 0)), zeros)
    # Clear the gate parameter gradients before accumulating them. Constructing ChunkGradientVector
    # here carries all of its channels through the following recurrent scan.
    # A single plain buffer also gives each head exactly one AIV writer.
    if fused_gate:
        if const_expr(dA.shape[1] == 2 * GATE_SCALAR_TRANSFER_WIDTH):
            zeros = Buffer(MemLoc.UB, dtypes.float32, make_layout((16, 128)))
            _zero_parameter_tile(zeros)
            first = (2 * get_block_idx() + get_subblock_id()) * 16
            step = 2 * get_block_num() * 16
            for row in range(first, dA.shape[0], step):
                valid = min(16, dA.shape[0] - row)
                mem_copy(dBias[row : row + valid, None], zeros[:valid, None])
                mem_copy(dA[row : row + valid, :8], zeros[:valid, :8])
        else:
            zeros = Buffer(
                MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM))
            )
            with vf(mode="simd"):
                mask, _ = update_mask(D_HALF, elem_bits=32)
                zero = vdups(0.0, dtypes.float32, mask=mask)
                vstore(zeros, 0, zero, mask)
                vstore(zeros, D_HALF, zero, mask)
            first = get_block_idx() * get_subblock_dim() + get_subblock_id()
            step = get_block_num() * get_subblock_dim()
            for value_head in range(first, dA.shape[0], step):
                mem_copy(
                    tile_slice(dBias, (1, SUPPORTED_HEAD_DIM), (value_head, 0)), zeros
                )
                mem_copy(
                    tile_slice(dA, (1, GATE_SCALAR_TRANSFER_WIDTH), (value_head, 0)),
                    tile_slice(zeros, (1, GATE_SCALAR_TRANSFER_WIDTH), (0, 0)),
                )
    # Publish the terminal state and parameter-gradient zeroing before use.
    _sync_cube_vector_handoff(0, grid=True)


class ChunkGradientVector(ChunkWorkspace):
    """Shared UB channels for dependent stage-one vector phases.

    A channel is reused only after its preceding producer and consumer have
    completed. The 16 channels stay under the per-vector-core depth limit.
    """

    @jit
    def load_causal_cache(self, gm_query, gm_key, gm_gamma, chunk):
        # WY reductions have finished. Reuse their private tiles until the
        # causal rounds finish; no allocation or arithmetic is added.
        shape = (CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)
        half_tile = chunk * 2 + get_subblock_id()
        mem_copy(self.batch_bf16[0], tile_slice(gm_query, shape, (half_tile, 0)))
        mem_copy(self.batch_bf16[1], tile_slice(gm_key, shape, (half_tile, 0)))
        mem_copy(
            self.wy_diagonal_residual[0], tile_slice(gm_gamma, shape, (half_tile, 0))
        )

    @jit
    def load_full_chunk_rows(
        self,
        gm_source,
        source_row,
        gm_destination,
        destination_row,
        width: Constexpr[int],
        cast_bf16: Constexpr[bool] = False,
        wy_residual: Constexpr[int] = -1,
        valid_rows: dtypes.int64 = CHUNK_SIZE // 2,
        resident=None,
    ):
        """Copy up to 32 live rows per AIV, preserving strides and zeroing the tail."""
        rows = CHUNK_SIZE // 2
        if const_expr(gm_source.dtype == dtypes.bfloat16):
            write = from_tensor(
                self.value_output.next_write(), layout=make_layout((rows, width))
            )
            self._load_chunk_input(write, gm_source, source_row, width, valid_rows)
            read = from_tensor(
                self.value_output.next_read(), layout=make_layout((rows, width))
            )
            mem_copy(
                tile_slice(gm_destination, (rows, width), (destination_row // rows, 0)),
                read,
            )
        else:
            if const_expr(resident is None):
                write = from_tensor(
                    self.value_gradient.next_write(), layout=make_layout((rows, width))
                )
                self._load_chunk_input(write, gm_source, source_row, width, valid_rows)
                read = from_tensor(
                    self.value_gradient.next_read(), layout=make_layout((rows, width))
                )
            else:
                read = resident
            if const_expr(cast_bf16):
                output = from_tensor(
                    self.value_output.next_write(), layout=make_layout((rows, width))
                )
                with vf(mode="simd"):
                    mask = create_mask(pattern="all")
                    for lane_block in range(rows * width // LANES):
                        offset = lane_block * LANES
                        value = vload(read, offset)
                        low = vcast(value, dtypes.bfloat16, mask=mask)
                        if const_expr(self.correct_wy_diagonal and wy_residual >= 0):
                            restored = vcast(low, dtypes.float32, mask=mask)
                            vstore(
                                self.wy_diagonal_residual[wy_residual],
                                offset,
                                vsub(value, restored, mask=mask),
                                mask,
                            )
                        vstore_pack(
                            output,
                            offset,
                            low,
                            mask,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    if const_expr(self.correct_wy_diagonal and wy_residual >= 0):
                        vmem_bar("vst_vld")
                result = from_tensor(
                    self.value_output.next_read(), layout=make_layout((rows, width))
                )
            else:
                result = read
            mem_copy(
                tile_slice(gm_destination, (rows, width), (destination_row // rows, 0)),
                result,
            )

    @jit
    def _load_chunk_input(
        self, write, gm_source, source_row, width: Constexpr[int], valid_rows
    ):
        rows = CHUNK_SIZE // 2
        if valid_rows > 0:
            if const_expr(gm_source.rank == 4):
                window = gm_source[
                    None, None, source_row : source_row + valid_rows, None
                ]
            else:
                window = gm_source[source_row : source_row + valid_rows, None]
            source = tile_slice(window, (rows, width), (0, 0))
            if valid_rows == rows:
                mem_copy(write, source)
            else:
                _load_zero_padded(write, source[:valid_rows, None])
        else:
            with vf(mode="simd"):
                full = create_mask(pattern="all")
                zero = vdups(0.0, dtypes.float32, mask=full)
                for lane_block in range(rows * width // LANES):
                    if const_expr(gm_source.dtype == dtypes.bfloat16):
                        low = vcast(zero, dtypes.bfloat16, mask=full)
                        vstore_pack(
                            write,
                            lane_block * LANES,
                            low,
                            full,
                            pack_mode=PackMode.B32_TO_B16,
                        )
                    else:
                        vstore(write, lane_block * LANES, zero, full)
                vmem_bar("vst_vld")

    @jit
    def load_beta_window(
        self, gm_source, source_row, gm_destination, chunk, valid_rows
    ):
        window = gm_source[None, None, source_row : source_row + valid_rows, None]
        source = tile_slice(window, (CHUNK_SIZE, 1), (0, 0))[:valid_rows, None]
        output = self.beta_values.next_write()
        if const_expr(gm_source.dtype == dtypes.float32):
            _load_zero_padded(output.view(1, CHUNK_SIZE), source.permute((1, 0)))
        else:
            _load_zero_padded(
                self.bf16_scalars.next_write().view(1, CHUNK_SIZE),
                source.permute((1, 0)),
            )
            low = self.bf16_scalars.next_read()
            with vf(mode="simd"):
                full = create_mask(pattern="all")
                value = vcast(
                    vload_unpack(low, 0, unpack_mode=UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=full,
                )
                vstore(output, 0, value, full)
        mem_copy(
            tile_slice(gm_destination, (CHUNK_SIZE,), (chunk,)),
            self.beta_values.next_read(),
        )

    @jit
    def store_beta_window(
        self, gm_source, chunk, gm_destination, destination_row, valid_rows
    ):
        if const_expr(gm_source.dtype == dtypes.float32):
            channel = self.beta_gradient
        else:
            channel = self.bf16_scalars
        mem_copy(channel.next_write(), tile_slice(gm_source, (CHUNK_SIZE,), (chunk,)))
        packed = channel.next_read()
        if gm_destination.physical_layout.stride[-2] == 1:
            destination = gm_destination.view(gm_destination.shape[-2])
            mem_copy(
                destination[destination_row : destination_row + valid_rows,],
                packed[:valid_rows,],
            )
        else:
            for row in range(valid_rows):
                self.store_beta_scalar(
                    gm_source,
                    chunk * CHUNK_SIZE + row,
                    gm_destination,
                    destination_row + row,
                )

    @jit
    def load_fp32_row(
        self,
        gm_source,
        source_row,
        gm_destination,
        destination_row,
        valid,
        width: Constexpr[int],
        channel_index: Constexpr[int] = 0,
    ):
        row = tile_slice(self.fp32_rows[channel_index], (1, width), (0, 0))
        if valid:
            mem_copy(row, tile_slice(gm_source, (1, width), (source_row, 0)))
        else:
            self._zero_fp32_row(row, width)
        mem_copy(
            tile_slice(gm_destination, (1, width), (destination_row, 0)),
            tile_slice(self.fp32_rows[channel_index], (1, width), (0, 0)),
        )

    @jit
    def zero_output_row(self, gm_destination, destination_row):
        if const_expr(gm_destination.dtype == dtypes.float32):
            channel = self.fp32_rows[0]
            self._zero_fp32_row(channel, SUPPORTED_HEAD_DIM)
        else:
            channel = self.bf16_rows[0]
            self._zero_bf16_row(channel, SUPPORTED_HEAD_DIM)
        mem_copy(
            tile_slice(gm_destination, (1, SUPPORTED_HEAD_DIM), (destination_row, 0)),
            channel,
        )

    @jit
    def store_contiguous_beta_chunk(
        self, gm_source, chunk, gm_destination, destination_row
    ):
        """Write 64 beta gradients in their original output dtype."""
        if const_expr(gm_source.dtype == dtypes.float32):
            channel = self.beta_gradient
        else:
            channel = self.bf16_scalars
        mem_copy(channel.next_write(), tile_slice(gm_source, (CHUNK_SIZE,), (chunk,)))
        mem_copy(
            tile_slice(
                gm_destination.view(gm_destination.shape[-2])[
                    destination_row : destination_row + CHUNK_SIZE,
                ],
                (CHUNK_SIZE,),
                (0,),
            ),
            channel.next_read(),
        )

    @jit
    def store_beta_scalar(self, gm_source, source_row, gm_destination, destination_row):
        if const_expr(gm_source.dtype == dtypes.float32):
            channel = self.beta_gradient
        else:
            channel = self.bf16_scalars
        scalar = tile_slice(channel.next_write(), (1,), (0,))
        mem_copy(scalar, tile_slice(gm_source, (1,), (source_row,)))
        # The destination retains [token, 1]. A rank-one tile addresses its
        # trailing singleton axis, not the token axis (whose stride is Nv).
        mem_copy(
            tile_slice(gm_destination, (1, 1), (destination_row, 0)),
            tile_slice(channel.next_read(), (1,), (0,)).view(1, 1),
        )

    @jit
    def zero_beta_scalar(self, gm_destination, destination_row):
        if const_expr(gm_destination.dtype == dtypes.float32):
            channel = self.beta_gradient
            self._zero_fp32_row(tile_slice(channel.next_write(), (1,), (0,)), 1)
        else:
            channel = self.bf16_scalars
            self._zero_bf16_row(tile_slice(channel.next_write(), (1,), (0,)), 1)
        mem_copy(
            tile_slice(gm_destination, (1, 1), (destination_row, 0)),
            tile_slice(channel.next_read(), (1,), (0,)).view(1, 1),
        )

    @jit
    def load_contiguous_beta_chunk(self, gm_source, source_row, gm_destination, chunk):
        """Load a complete contiguous beta vector without 64 scalar DMA calls."""
        # Keep-index slices retain leading singleton axes in the SDK.
        # A tile has exactly 64 contiguous scalars under the caller guard.
        source = _full_token_tile(gm_source, source_row, 1).view(CHUNK_SIZE)
        output = self.beta_values.next_write()
        if const_expr(gm_source.dtype == dtypes.float32):
            mem_copy(output, source)
        else:
            mem_copy(self.bf16_scalars.next_write(), source)
            low = self.bf16_scalars.next_read()
            with vf(mode="simd"):
                full = create_mask(pattern="all")
                value = vcast(
                    vload_unpack(low, 0, unpack_mode=UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=full,
                )
                vstore(output, 0, value, full)
        mem_copy(
            tile_slice(gm_destination, (CHUNK_SIZE,), (chunk,)),
            self.beta_values.next_read(),
        )

    @jit
    def load_strided_beta_chunk(self, gm_source, source_row, gm_destination, chunk):
        """Gather a strided beta column during DMA, then cast only if needed."""
        source = _full_token_tile(gm_source, source_row, 1)
        output = self.beta_values.next_write()
        if const_expr(gm_source.dtype == dtypes.float32):
            mem_copy(
                output.view(1, CHUNK_SIZE),
                source.permute((1, 0)),
                options=copy_options(engine="nddma"),
            )
        else:
            mem_copy(
                self.bf16_scalars.next_write().view(1, CHUNK_SIZE),
                source.permute((1, 0)),
                options=copy_options(engine="nddma"),
            )
            read = self.bf16_scalars.next_read()
            with vf(mode="simd"):
                mask = create_mask(pattern="all")
                value = vcast(
                    vload_unpack(read, 0, unpack_mode=UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=mask,
                )
                vstore(output, 0, value, mask)
        mem_copy(
            tile_slice(gm_destination, (CHUNK_SIZE,), (chunk,)),
            self.beta_values.next_read(),
        )

    @jit
    def _zero_bf16_row(self, row, width: Constexpr[int]):
        with vf(mode="simd"):
            mask, _ = update_mask(width, elem_bits=16)
            vstore(row, 0, vdups(0.0, dtypes.bfloat16, mask=mask), mask)

    @jit
    def _zero_fp32_row(self, row, width: Constexpr[int]):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=mask)
            for half in range(width // LANES):
                vstore(row, half * LANES, zero, mask)
            if const_expr(width % LANES):
                tail_mask, _ = update_mask(width % LANES, elem_bits=32)
                vstore(row, (width // LANES) * LANES, zero, tail_mask)

    @jit
    def begin_gate_chunk(self):
        with vf(mode="simd"):
            mask, _ = update_mask(GATE_SCALAR_TRANSFER_WIDTH, elem_bits=32)
            vstore(self.gate_alog_sum, 0, vdups(0.0, dtypes.float32, mask=mask), mask)

    @jit
    def begin_gate_task(self):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=mask)
            for half in range(2):
                vstore(self.gate_task_bias_sum, half * D_HALF, zero, mask)
            scalar_mask, _ = update_mask(GATE_SCALAR_TRANSFER_WIDTH, elem_bits=32)
            vstore(
                self.gate_task_alog_sum,
                0,
                vdups(0.0, dtypes.float32, mask=scalar_mask),
                scalar_mask,
            )

    @jit
    def accumulate_gate_chunk(self, half):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            offset = half * D_HALF
            vmem_bar("vst_vld")
            vstore(
                self.gate_task_bias_sum,
                offset,
                vadd(
                    vload(self.gate_task_bias_sum, offset),
                    vload(self.gate_bias_sum, 0),
                    mask=mask,
                ),
                mask,
            )
            scalar_mask, _ = update_mask(GATE_SCALAR_TRANSFER_WIDTH, elem_bits=32)
            vstore(
                self.gate_task_alog_sum,
                0,
                vadd(
                    vload(self.gate_task_alog_sum, 0),
                    vload(self.gate_alog_sum, 0),
                    mask=scalar_mask,
                ),
                scalar_mask,
            )

    @jit
    def finish_gate_task_atomic(
        self,
        gm_public_dA,
        gm_public_dBias,
        value_head,
        has_bias: dtypes.int64,
        store_alog: Constexpr[bool] = True,
    ):
        parameter_row = _gate_parameter_row(gm_public_dA, value_head)
        if has_bias:
            for half in range(get_subblock_id(), 2, get_subblock_dim()):
                mem_copy(
                    tile_slice(self.fp32_rows[0], (1, D_HALF), (0, 0)),
                    tile_slice(self.gate_task_bias_sum, (1, D_HALF), (0, half)),
                )
                mem_copy(
                    tile_slice(gm_public_dBias, (1, D_HALF), (parameter_row, half)),
                    tile_slice(self.fp32_rows[0], (1, D_HALF), (0, 0)),
                    options=copy_options(atomic_add=True),
                )
        if const_expr(store_alog):
            # The scalar slot is already DMA-aligned. Publish it directly
            # without reusing the staging row of the preceding bias store.
            mem_copy(
                tile_slice(
                    gm_public_dA, (1, GATE_SCALAR_TRANSFER_WIDTH), (parameter_row, 0)
                ),
                self.gate_task_alog_sum,
                options=copy_options(atomic_add=True),
            )

    @jit
    def begin_gate_half(self):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=mask)
            vstore(self.gate_running, 0, zero, mask)
            vstore(self.gate_bias_sum, 0, zero, mask)

    @jit
    def gate_row(
        self,
        gm_d_gate_base,
        gm_d_gate_pair,
        gm_raw_gate,
        gm_dt_bias,
        gm_public_d_gate,
        flat_row,
        raw_row,
        raw_half,
        raw_head,
        alog,
        fused: dtypes.int64,
        has_bias: dtypes.int64,
        softplus: dtypes.int64,
        lower_bound: dtypes.float32,
    ):
        mem_copy(
            tile_slice(self.fp32_rows[0], (1, D_HALF), (0, 0)),
            tile_slice(gm_d_gate_base, (1, D_HALF), (flat_row, raw_half)),
        )
        mem_copy(
            tile_slice(self.fp32_rows[1], (1, D_HALF), (0, 0)),
            tile_slice(gm_d_gate_pair, (1, D_HALF), (flat_row, raw_half)),
        )
        if const_expr(gm_raw_gate.dtype == dtypes.float32):
            if fused:
                mem_copy(
                    tile_slice(self.fp32_rows[3], (1, D_HALF), (0, 0)),
                    tile_slice(gm_raw_gate, (1, D_HALF), (raw_row, raw_half)),
                )
            else:
                self._zero_fp32_row(self.fp32_rows[3], D_HALF)
            raw = self.fp32_rows[3]
        else:
            if fused:
                mem_copy(
                    tile_slice(self.bf16_rows[0], (1, D_HALF), (0, 0)),
                    tile_slice(gm_raw_gate, (1, D_HALF), (raw_row, raw_half)),
                )
            else:
                self._zero_bf16_row(self.bf16_rows[0], D_HALF)
            raw = self.bf16_rows[0]
        if fused and has_bias:
            mem_copy(
                tile_slice(self.fp32_rows[2], (1, D_HALF), (0, 0)),
                tile_slice(gm_dt_bias, (1, D_HALF), (raw_head, raw_half)),
            )
        else:
            self._zero_fp32_row(self.fp32_rows[2], D_HALF)
        if const_expr(gm_public_d_gate.dtype == dtypes.float32):
            output_channel = self.fp32_rows[4]
        else:
            output_channel = self.bf16_rows[1]
        self._compute_gate_row(
            self.fp32_rows[0],
            self.fp32_rows[1],
            raw,
            self.fp32_rows[2],
            output_channel,
            alog,
            fused,
            softplus,
            lower_bound,
        )
        mem_copy(
            tile_slice(gm_public_d_gate, (1, D_HALF), (raw_row, raw_half)),
            tile_slice(output_channel, (1, D_HALF), (0, 0)),
        )

    @jit
    def gate_full_chunk(
        self,
        gm_d_gate_base,
        gm_d_gate_pair,
        gm_raw_gate,
        gm_dt_bias,
        gm_public_d_gate,
        flat_chunk,
        raw_row_start,
        raw_half,
        raw_head,
        alog,
        has_bias: dtypes.int64,
        softplus: dtypes.int64,
        lower_bound: dtypes.float32,
    ):
        """Batch gate rows without changing the reverse FP32 recurrence."""
        # The preceding WY/Causal/Value phases have released these Channels.
        # FP32 raw gates reuse the third 16 KiB plane of value_gradient;
        # BF16 gates keep the original value_output plane.
        # Bias is constant across the half-chunk and remains live for 64 rows.
        if has_bias:
            mem_copy(
                tile_slice(self.fp32_rows[2], (1, D_HALF), (0, 0)),
                tile_slice(gm_dt_bias, (1, D_HALF), (raw_head, raw_half)),
            )
        else:
            self._zero_fp32_row(self.fp32_rows[2], D_HALF)
        bias = self.fp32_rows[2]
        groups = CHUNK_SIZE // GATE_BATCH_ROWS
        shape = (GATE_BATCH_ROWS, D_HALF)
        for reverse_group in range(groups):
            group = groups - reverse_group - 1
            flat_tile = flat_chunk * groups + group
            raw_row = raw_row_start + group * GATE_BATCH_ROWS
            fp_write = self.value_gradient.next_write()
            if const_expr(not self.fuse_base):
                mem_copy(
                    from_tensor(fp_write, layout=make_layout(shape)),
                    tile_slice(gm_d_gate_base, shape, (flat_tile, raw_half)),
                )
            mem_copy(
                from_tensor(
                    fp_write,
                    offset=GATE_BATCH_ROWS * D_HALF * FP32_ELEMENT_BYTES,
                    layout=make_layout(shape),
                ),
                tile_slice(gm_d_gate_pair, shape, (flat_tile, raw_half)),
            )
            low_write = self.value_output.next_write()
            if const_expr(gm_raw_gate.dtype == dtypes.float32):
                mem_copy(
                    from_tensor(
                        fp_write,
                        offset=2 * GATE_BATCH_ROWS * D_HALF * FP32_ELEMENT_BYTES,
                        layout=make_layout(shape),
                    ),
                    _token_rows_tile(
                        gm_raw_gate, raw_row, GATE_BATCH_ROWS, D_HALF, raw_half
                    ),
                )
            else:
                mem_copy(
                    from_tensor(low_write, layout=make_layout(shape)),
                    _token_rows_tile(
                        gm_raw_gate, raw_row, GATE_BATCH_ROWS, D_HALF, raw_half
                    ),
                )
            fp_read = self.value_gradient.next_read()
            low_read = self.value_output.next_read()
            base = from_tensor(fp_read, layout=make_layout(shape))
            pair = from_tensor(
                fp_read,
                offset=GATE_BATCH_ROWS * D_HALF * FP32_ELEMENT_BYTES,
                layout=make_layout(shape),
            )
            if const_expr(gm_raw_gate.dtype == dtypes.float32):
                raw = from_tensor(
                    fp_read,
                    offset=2 * GATE_BATCH_ROWS * D_HALF * FP32_ELEMENT_BYTES,
                    layout=make_layout(shape),
                )
            else:
                raw = from_tensor(low_read, layout=make_layout(shape))
            if const_expr(gm_public_d_gate.dtype == dtypes.float32):
                # Each raw row is dead after its derivative is evaluated.
                output = raw
            else:
                output = from_tensor(
                    low_read,
                    offset=GATE_BATCH_ROWS * D_HALF * BF16_ELEMENT_BYTES,
                    layout=make_layout(shape),
                )
            # Preserve token 63..0 and per-row FP32 reductions, while keeping
            # the running gate and parameter sums in registers for this batch.
            self._compute_gate_batch(
                base, pair, raw, bias, output, alog, True, softplus, lower_bound
            )
            mem_copy(
                _token_rows_tile(
                    gm_public_d_gate, raw_row, GATE_BATCH_ROWS, D_HALF, raw_half
                ),
                output,
            )

    @jit
    def _compute_gate_batch(
        self,
        base,
        pair,
        raw,
        bias,
        out,
        alog,
        fused: dtypes.int64,
        softplus: dtypes.int64,
        lower_bound: dtypes.float32,
    ):
        # Runtime switches are resolved on the scalar side of the VF boundary.
        if fused:
            if softplus:
                self._compute_gate_batch_constexpr(
                    base,
                    pair,
                    raw,
                    bias,
                    out,
                    alog,
                    True,
                    True,
                    lower_bound,
                )
            else:
                self._compute_gate_batch_constexpr(
                    base,
                    pair,
                    raw,
                    bias,
                    out,
                    alog,
                    True,
                    False,
                    lower_bound,
                )
        else:
            if softplus:
                self._compute_gate_batch_constexpr(
                    base,
                    pair,
                    raw,
                    bias,
                    out,
                    alog,
                    False,
                    True,
                    lower_bound,
                )
            else:
                self._compute_gate_batch_constexpr(
                    base,
                    pair,
                    raw,
                    bias,
                    out,
                    alog,
                    False,
                    False,
                    lower_bound,
                )

    @jit
    def _compute_gate_batch_constexpr(
        self,
        base,
        pair,
        raw,
        bias,
        out,
        alog,
        fused: Constexpr[bool],
        softplus: Constexpr[bool],
        lower_bound: dtypes.float32,
    ):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            vmem_bar("vst_vld")
            running = vload(self.gate_running, 0)
            bias_sum = vload(self.gate_bias_sum, 0)
            scalar_mask, _ = update_mask(1, elem_bits=32)
            alog_sum = vload_broadcast(self.gate_alog_sum, 0)
            alpha = vexp(vdups(alog, dtypes.float32, mask=mask), mask=mask)
            for reverse_row in range(GATE_BATCH_ROWS):
                offset = (GATE_BATCH_ROWS - reverse_row - 1) * D_HALF
                if const_expr(self.fuse_base):
                    step = vadd(running, vload(pair, offset), mask=mask)
                else:
                    step = vadd(
                        running,
                        vadd(vload(base, offset), vload(pair, offset), mask=mask),
                        mask=mask,
                    )
                running = step
                grad = step
                if const_expr(fused):
                    if const_expr(raw.dtype == dtypes.float32):
                        raw32 = vload(raw, offset)
                        if const_expr(self.round_raw_gate):
                            raw32 = vcast(
                                vcast(raw32, dtypes.bfloat16, mask=mask),
                                dtypes.float32,
                                mask=mask,
                            )
                    else:
                        raw32 = vcast(
                            vload_unpack(
                                raw, offset, unpack_mode=UnpackMode.B16_TO_B32
                            ),
                            dtypes.float32,
                            mask=mask,
                        )
                    pre = vadd(raw32, vload(bias, 0), mask=mask)
                    x = vmul(alpha, pre, mask=mask)
                    if const_expr(softplus):
                        x = pre
                    negative = vmins(x, 0.0, mask=mask)
                    positive = vmaxs(x, 0.0, mask=mask)
                    tiny_exp = vexp(vsub(negative, positive, mask=mask), mask=mask)
                    sigmoid = vdiv(
                        vselect(
                            tiny_exp,
                            vdups(1.0, dtypes.float32, mask=mask),
                            cond_mask=vlts(x, 0.0, mask=mask),
                        ),
                        vadds(tiny_exp, 1.0, mask=mask),
                        mask=mask,
                    )
                    one_minus = vadds(vmuls(sigmoid, -1.0, mask=mask), 1.0, mask=mask)
                    dz = vmuls(
                        vmul(step, vmul(sigmoid, one_minus, mask=mask), mask=mask),
                        lower_bound,
                        mask=mask,
                    )
                    grad = vmul(alpha, dz, mask=mask)
                    dA_channel = vmul(dz, x, mask=mask)
                    if const_expr(softplus):
                        grad = vmuls(
                            vmul(step, vmul(alpha, sigmoid, mask=mask), mask=mask),
                            -1.0,
                            mask=mask,
                        )
                        softplus_value = vadd(
                            positive,
                            vlog(vadds(tiny_exp, 1.0, mask=mask), mask=mask),
                            mask=mask,
                        )
                        dA_channel = vmuls(
                            vmul(
                                step, vmul(alpha, softplus_value, mask=mask), mask=mask
                            ),
                            -1.0,
                            mask=mask,
                        )
                    bias_sum = vadd(bias_sum, grad, mask=mask)
                    scalar = vreduce_sum(dA_channel, mask=mask)
                    alog_sum = vadd(alog_sum, scalar, mask=scalar_mask)
                if const_expr(out.dtype == dtypes.float32):
                    vstore(out, offset, grad, mask)
                else:
                    vstore_pack(
                        out,
                        offset,
                        vcast(grad, dtypes.bfloat16, mask=mask),
                        mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
            vstore(self.gate_running, 0, running, mask)
            if const_expr(fused):
                vstore(self.gate_bias_sum, 0, bias_sum, mask)
                vstore_first(self.gate_alog_sum, 0, alog_sum)

    @jit
    def _compute_gate_row(
        self,
        base,
        pair,
        raw,
        bias,
        out,
        alog,
        fused: dtypes.int64,
        softplus: dtypes.int64,
        lower_bound: dtypes.float32,
    ):
        # Runtime switches are resolved on the scalar side of the VF boundary.
        if fused:
            if softplus:
                self._compute_gate_row_constexpr(
                    base,
                    pair,
                    raw,
                    bias,
                    out,
                    alog,
                    True,
                    True,
                    lower_bound,
                )
            else:
                self._compute_gate_row_constexpr(
                    base,
                    pair,
                    raw,
                    bias,
                    out,
                    alog,
                    True,
                    False,
                    lower_bound,
                )
        else:
            if softplus:
                self._compute_gate_row_constexpr(
                    base,
                    pair,
                    raw,
                    bias,
                    out,
                    alog,
                    False,
                    True,
                    lower_bound,
                )
            else:
                self._compute_gate_row_constexpr(
                    base,
                    pair,
                    raw,
                    bias,
                    out,
                    alog,
                    False,
                    False,
                    lower_bound,
                )

    @jit
    def _compute_gate_row_constexpr(
        self,
        base,
        pair,
        raw,
        bias,
        out,
        alog,
        fused: Constexpr[bool],
        softplus: Constexpr[bool],
        lower_bound: dtypes.float32,
    ):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            vmem_bar("vst_vld")
            step = vadd(
                vload(self.gate_running, 0),
                vadd(vload(base, 0), vload(pair, 0), mask=mask),
                mask=mask,
            )
            vstore(self.gate_running, 0, step, mask)
            grad = step
            if const_expr(fused):
                if const_expr(raw.dtype == dtypes.float32):
                    raw32 = vload(raw, 0)
                    if const_expr(self.round_raw_gate):
                        raw32 = vcast(
                            vcast(raw32, dtypes.bfloat16, mask=mask),
                            dtypes.float32,
                            mask=mask,
                        )
                else:
                    raw32 = vcast(
                        vload_unpack(raw, 0, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=mask,
                    )
                pre = vadd(raw32, vload(bias, 0), mask=mask)
                alpha = vexp(vdups(alog, dtypes.float32, mask=mask), mask=mask)
                x = vmul(alpha, pre, mask=mask)
                if const_expr(softplus):
                    x = pre
                negative = vmins(x, 0.0, mask=mask)
                positive = vmaxs(x, 0.0, mask=mask)
                tiny_exp = vexp(vsub(negative, positive, mask=mask), mask=mask)
                sigmoid = vdiv(
                    vselect(
                        tiny_exp,
                        vdups(1.0, dtypes.float32, mask=mask),
                        cond_mask=vlts(x, 0.0, mask=mask),
                    ),
                    vadds(tiny_exp, 1.0, mask=mask),
                    mask=mask,
                )
                one_minus = vadds(vmuls(sigmoid, -1.0, mask=mask), 1.0, mask=mask)
                dz = vmuls(
                    vmul(step, vmul(sigmoid, one_minus, mask=mask), mask=mask),
                    lower_bound,
                    mask=mask,
                )
                grad = vmul(alpha, dz, mask=mask)
                dA_channel = vmul(dz, x, mask=mask)
                if const_expr(softplus):
                    grad = vmuls(
                        vmul(step, vmul(alpha, sigmoid, mask=mask), mask=mask),
                        -1.0,
                        mask=mask,
                    )
                    softplus_value = vadd(
                        positive,
                        vlog(vadds(tiny_exp, 1.0, mask=mask), mask=mask),
                        mask=mask,
                    )
                    dA_channel = vmuls(
                        vmul(step, vmul(alpha, softplus_value, mask=mask), mask=mask),
                        -1.0,
                        mask=mask,
                    )
                vstore(
                    self.gate_bias_sum,
                    0,
                    vadd(vload(self.gate_bias_sum, 0), grad, mask=mask),
                    mask,
                )
                scalar = vreduce_sum(dA_channel, mask=mask)
                scalar_mask, _ = update_mask(1, elem_bits=32)
                vmem_bar("vst_vld")
                vstore_first(
                    self.gate_alog_sum,
                    0,
                    vadd(
                        vload_broadcast(self.gate_alog_sum, 0), scalar, mask=scalar_mask
                    ),
                )
            if const_expr(out.dtype == dtypes.float32):
                vstore(out, 0, grad, mask)
            else:
                vstore_pack(
                    out,
                    0,
                    vcast(grad, dtypes.bfloat16, mask=mask),
                    mask,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def finish_gate_head(self, gm_public_dA, gm_public_dBias, value_head, half):
        mem_copy(
            tile_slice(self.fp32_rows[0], (1, D_HALF), (0, 0)),
            self.gate_bias_sum,
        )
        mem_copy(
            tile_slice(gm_public_dBias, (1, D_HALF), (value_head, half)),
            tile_slice(self.fp32_rows[0], (1, D_HALF), (0, 0)),
        )
        if half == 1:
            mem_copy(
                tile_slice(self.fp32_rows[0], (1, GATE_SCALAR_TRANSFER_WIDTH), (0, 0)),
                self.gate_alog_sum,
            )
            mem_copy(
                tile_slice(
                    gm_public_dA, (1, GATE_SCALAR_TRANSFER_WIDTH), (value_head, 0)
                ),
                tile_slice(self.fp32_rows[0], (1, GATE_SCALAR_TRANSFER_WIDTH), (0, 0)),
            )

    @jit
    def begin_head_row(self):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=mask)
            for half in range(2):
                vstore(self.head_query_sum, half * D_HALF, zero, mask)
                vstore(self.head_key_sum, half * D_HALF, zero, mask)

    @jit
    def add_head_row(
        self, gm_d_query_base, gm_d_query_pair, gm_d_key_base, gm_d_key_pair, flat_row
    ):
        mem_copy(
            self.fp32_rows[0],
            tile_slice(gm_d_query_base, (1, SUPPORTED_HEAD_DIM), (flat_row, 0)),
        )
        mem_copy(
            self.fp32_rows[1],
            tile_slice(gm_d_query_pair, (1, SUPPORTED_HEAD_DIM), (flat_row, 0)),
        )
        mem_copy(
            self.fp32_rows[2],
            tile_slice(gm_d_key_base, (1, SUPPORTED_HEAD_DIM), (flat_row, 0)),
        )
        mem_copy(
            self.fp32_rows[3],
            tile_slice(gm_d_key_pair, (1, SUPPORTED_HEAD_DIM), (flat_row, 0)),
        )
        self._add_head_row(
            self.fp32_rows[0],
            self.fp32_rows[1],
            self.fp32_rows[2],
            self.fp32_rows[3],
        )

    @jit
    def _add_head_row(self, query_base, query_pair, key_base, key_pair):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            vmem_bar("vst_vld")
            for half in range(2):
                offset = half * D_HALF
                q = vadd(
                    vload(self.head_query_sum, offset),
                    vadd(
                        vload(query_base, offset), vload(query_pair, offset), mask=mask
                    ),
                    mask=mask,
                )
                k = vadd(
                    vload(self.head_key_sum, offset),
                    vadd(vload(key_base, offset), vload(key_pair, offset), mask=mask),
                    mask=mask,
                )
                vstore(self.head_query_sum, offset, q, mask)
                vstore(self.head_key_sum, offset, k, mask)

    @jit
    def finish_head_row(
        self,
        gm_effective_query,
        gm_effective_key,
        gm_q_rstd,
        gm_k_rstd,
        gm_public_d_query,
        gm_public_d_key,
        effective_batch,
        effective_head,
        effective_row,
        raw_batch,
        raw_row,
        use_l2norm: dtypes.int64,
    ):
        if use_l2norm:
            query_source = gm_effective_query[
                effective_batch, effective_head, None, None
            ]
            key_source = gm_effective_key[effective_batch, effective_head, None, None]
            mem_copy(
                self.bf16_rows[0],
                tile_slice(query_source, (1, SUPPORTED_HEAD_DIM), (effective_row, 0)),
            )
            mem_copy(
                self.bf16_rows[1],
                tile_slice(key_source, (1, SUPPORTED_HEAD_DIM), (effective_row, 0)),
            )
            query_rstd = gm_q_rstd[effective_batch, effective_head, effective_row, 0]
            key_rstd = gm_k_rstd[effective_batch, effective_head, effective_row, 0]
            self._finish_head_row_normalized(
                self.bf16_rows[0],
                self.bf16_rows[1],
                self.bf16_rows[2],
                self.bf16_rows[3],
                query_rstd,
                key_rstd,
            )
        else:
            self._finish_head_row_direct(self.bf16_rows[2], self.bf16_rows[3])
        query_destination = gm_public_d_query[raw_batch, effective_head, None, None]
        key_destination = gm_public_d_key[raw_batch, effective_head, None, None]
        mem_copy(
            tile_slice(query_destination, (1, SUPPORTED_HEAD_DIM), (raw_row, 0)),
            self.bf16_rows[2],
        )
        mem_copy(
            tile_slice(key_destination, (1, SUPPORTED_HEAD_DIM), (raw_row, 0)),
            self.bf16_rows[3],
        )

    @jit
    def _finish_head_row_direct(self, query_out, key_out):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            vmem_bar("vst_vld")
            for half in range(2):
                offset = half * D_HALF
                vstore_pack(
                    query_out,
                    offset,
                    vcast(
                        vload(self.head_query_sum, offset), dtypes.bfloat16, mask=mask
                    ),
                    mask,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    key_out,
                    offset,
                    vcast(vload(self.head_key_sum, offset), dtypes.bfloat16, mask=mask),
                    mask,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def _finish_single_head_batch(
        self,
        query_base,
        query_pair,
        key_base,
        key_pair,
        query_norm,
        key_norm,
        query_out,
        key_out,
        query_rstd,
        key_rstd,
        use_l2norm: dtypes.int64,
    ):
        # Runtime switches are resolved on the scalar side of the VF boundary.
        if use_l2norm:
            self._finish_single_head_batch_constexpr(
                query_base,
                query_pair,
                key_base,
                key_pair,
                query_norm,
                key_norm,
                query_out,
                key_out,
                query_rstd,
                key_rstd,
                True,
            )
        else:
            self._finish_single_head_batch_constexpr(
                query_base,
                query_pair,
                key_base,
                key_pair,
                query_norm,
                key_norm,
                query_out,
                key_out,
                query_rstd,
                key_rstd,
                False,
            )

    @jit
    def _finish_single_head_batch_constexpr(
        self,
        query_base,
        query_pair,
        key_base,
        key_pair,
        query_norm,
        key_norm,
        query_out,
        key_out,
        query_rstd,
        key_rstd,
        use_l2norm: Constexpr[bool],
    ):
        # A single contributor needs no UB accumulator round-trip. Keep the
        # explicit zero addition and the original scalar-rstd multiplication.
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=mask)
            for row in range(self.head_batch_rows):
                offset = row * SUPPORTED_HEAD_DIM
                if const_expr(self.fuse_base):
                    q0 = vadd(zero, vload(query_pair, offset), mask=mask)
                else:
                    q0 = vadd(
                        zero,
                        vadd(
                            vload(query_base, offset),
                            vload(query_pair, offset),
                            mask=mask,
                        ),
                        mask=mask,
                    )
                if const_expr(self.fuse_base):
                    q1 = vadd(zero, vload(query_pair, offset + D_HALF), mask=mask)
                else:
                    q1 = vadd(
                        zero,
                        vadd(
                            vload(query_base, offset + D_HALF),
                            vload(query_pair, offset + D_HALF),
                            mask=mask,
                        ),
                        mask=mask,
                    )
                if const_expr(self.fuse_base):
                    k0 = vadd(zero, vload(key_pair, offset), mask=mask)
                else:
                    k0 = vadd(
                        zero,
                        vadd(
                            vload(key_base, offset), vload(key_pair, offset), mask=mask
                        ),
                        mask=mask,
                    )
                if const_expr(self.fuse_base):
                    k1 = vadd(zero, vload(key_pair, offset + D_HALF), mask=mask)
                else:
                    k1 = vadd(
                        zero,
                        vadd(
                            vload(key_base, offset + D_HALF),
                            vload(key_pair, offset + D_HALF),
                            mask=mask,
                        ),
                        mask=mask,
                    )
                qout0 = q0
                qout1 = q1
                kout0 = k0
                kout1 = k1
                if const_expr(use_l2norm):
                    qscale = vload_broadcast(query_rstd, row * 8)
                    kscale = vload_broadcast(key_rstd, row * 8)
                    qn0 = vcast(
                        vload_unpack(
                            query_norm, offset, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=mask,
                    )
                    qn1 = vcast(
                        vload_unpack(
                            query_norm,
                            offset + D_HALF,
                            unpack_mode=UnpackMode.B16_TO_B32,
                        ),
                        dtypes.float32,
                        mask=mask,
                    )
                    kn0 = vcast(
                        vload_unpack(
                            key_norm, offset, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=mask,
                    )
                    kn1 = vcast(
                        vload_unpack(
                            key_norm, offset + D_HALF, unpack_mode=UnpackMode.B16_TO_B32
                        ),
                        dtypes.float32,
                        mask=mask,
                    )
                    qdot = vdup(
                        vadd(
                            vreduce_sum(vmul(qn0, q0, mask=mask), mask=mask),
                            vreduce_sum(vmul(qn1, q1, mask=mask), mask=mask),
                            mask=mask,
                        ),
                        mask=mask,
                    )
                    kdot = vdup(
                        vadd(
                            vreduce_sum(vmul(kn0, k0, mask=mask), mask=mask),
                            vreduce_sum(vmul(kn1, k1, mask=mask), mask=mask),
                            mask=mask,
                        ),
                        mask=mask,
                    )
                    qout0 = vmul(
                        vsub(q0, vmul(qn0, qdot, mask=mask), mask=mask),
                        qscale,
                        mask=mask,
                    )
                    qout1 = vmul(
                        vsub(q1, vmul(qn1, qdot, mask=mask), mask=mask),
                        qscale,
                        mask=mask,
                    )
                    kout0 = vmul(
                        vsub(k0, vmul(kn0, kdot, mask=mask), mask=mask),
                        kscale,
                        mask=mask,
                    )
                    kout1 = vmul(
                        vsub(k1, vmul(kn1, kdot, mask=mask), mask=mask),
                        kscale,
                        mask=mask,
                    )
                vstore_pack(
                    query_out,
                    offset,
                    vcast(qout0, dtypes.bfloat16, mask=mask),
                    mask,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    query_out,
                    offset + D_HALF,
                    vcast(qout1, dtypes.bfloat16, mask=mask),
                    mask,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    key_out,
                    offset,
                    vcast(kout0, dtypes.bfloat16, mask=mask),
                    mask,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    key_out,
                    offset + D_HALF,
                    vcast(kout1, dtypes.bfloat16, mask=mask),
                    mask,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def _finish_head_row_normalized(
        self, query_norm, key_norm, query_out, key_out, query_rstd, key_rstd
    ):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            vmem_bar("vst_vld")
            q0 = vload(self.head_query_sum, 0)
            q1 = vload(self.head_query_sum, D_HALF)
            k0 = vload(self.head_key_sum, 0)
            k1 = vload(self.head_key_sum, D_HALF)
            qn0 = vcast(
                vload_unpack(query_norm, 0, unpack_mode=UnpackMode.B16_TO_B32),
                dtypes.float32,
                mask=mask,
            )
            qn1 = vcast(
                vload_unpack(query_norm, D_HALF, unpack_mode=UnpackMode.B16_TO_B32),
                dtypes.float32,
                mask=mask,
            )
            kn0 = vcast(
                vload_unpack(key_norm, 0, unpack_mode=UnpackMode.B16_TO_B32),
                dtypes.float32,
                mask=mask,
            )
            kn1 = vcast(
                vload_unpack(key_norm, D_HALF, unpack_mode=UnpackMode.B16_TO_B32),
                dtypes.float32,
                mask=mask,
            )
            qdot = vdup(
                vadd(
                    vreduce_sum(vmul(qn0, q0, mask=mask), mask=mask),
                    vreduce_sum(vmul(qn1, q1, mask=mask), mask=mask),
                    mask=mask,
                ),
                mask=mask,
            )
            kdot = vdup(
                vadd(
                    vreduce_sum(vmul(kn0, k0, mask=mask), mask=mask),
                    vreduce_sum(vmul(kn1, k1, mask=mask), mask=mask),
                    mask=mask,
                ),
                mask=mask,
            )
            qout0 = vmuls(
                vsub(q0, vmul(qn0, qdot, mask=mask), mask=mask), query_rstd, mask=mask
            )
            qout1 = vmuls(
                vsub(q1, vmul(qn1, qdot, mask=mask), mask=mask), query_rstd, mask=mask
            )
            kout0 = vmuls(
                vsub(k0, vmul(kn0, kdot, mask=mask), mask=mask), key_rstd, mask=mask
            )
            kout1 = vmuls(
                vsub(k1, vmul(kn1, kdot, mask=mask), mask=mask), key_rstd, mask=mask
            )
            vstore_pack(
                query_out,
                0,
                vcast(qout0, dtypes.bfloat16, mask=mask),
                mask,
                pack_mode=PackMode.B32_TO_B16,
            )
            vstore_pack(
                query_out,
                D_HALF,
                vcast(qout1, dtypes.bfloat16, mask=mask),
                mask,
                pack_mode=PackMode.B32_TO_B16,
            )
            vstore_pack(
                key_out,
                0,
                vcast(kout0, dtypes.bfloat16, mask=mask),
                mask,
                pack_mode=PackMode.B32_TO_B16,
            )
            vstore_pack(
                key_out,
                D_HALF,
                vcast(kout1, dtypes.bfloat16, mask=mask),
                mask,
                pack_mode=PackMode.B32_TO_B16,
            )

    @jit
    def _head_batch_plane(self, slot, plane: Constexpr[int]):
        return tile_slice(slot, (self.head_batch_rows, SUPPORTED_HEAD_DIM), (plane, 0))

    @jit
    def finish_single_head_chunk(
        self,
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
        flat_chunk,
        effective_batch,
        effective_head,
        effective_row_start,
        raw_batch,
        raw_row_start,
        use_l2norm: dtypes.int64,
        resident_pair: Constexpr[bool] = False,
        valid_rows: dtypes.int64 = CHUNK_SIZE,
    ):
        # Process complete row groups for a sole Q/K contributor.
        # AIVs own disjoint consecutive 32-row ranges. GM views retain each
        # public layout's actual strides, including strided norm/rstd caches.
        query_destination = gm_public_d_query[raw_batch, effective_head, None, None]
        key_destination = gm_public_d_key[raw_batch, effective_head, None, None]
        query_source = gm_effective_query[effective_batch, effective_head, None, None]
        key_source = gm_effective_key[effective_batch, effective_head, None, None]
        # Resident Q/K pairs need only two FP32 input planes; 16 rows plus
        # padded rstd scalars fit the fast 32 KiB slot. Cached effective Q/K
        # are the exact normalization operands and remain live through here.
        fp_planes = 0 if self.fuse_base else (2 if resident_pair else 4)
        for group in range(CHUNK_SIZE // (2 * self.head_batch_rows)):
            first_row = (
                get_subblock_id() * (CHUNK_SIZE // 2) + group * self.head_batch_rows
            )
            if first_row + self.head_batch_rows <= valid_rows:
                flat_tile = (
                    flat_chunk * (CHUNK_SIZE // self.head_batch_rows)
                    + first_row // self.head_batch_rows
                )
                effective_row = effective_row_start + first_row
                raw_row = raw_row_start + first_row
                shape = (self.head_batch_rows, SUPPORTED_HEAD_DIM)
                fp_write = self.value_gradient.next_write()
                if const_expr(not self.fuse_base):
                    mem_copy(
                        self._head_batch_plane(fp_write, 0),
                        tile_slice(gm_d_query_base, shape, (flat_tile, 0)),
                    )
                if const_expr(not resident_pair):
                    mem_copy(
                        self._head_batch_plane(fp_write, 1),
                        tile_slice(gm_d_query_pair, shape, (flat_tile, 0)),
                    )
                if const_expr(not self.fuse_base):
                    mem_copy(
                        self._head_batch_plane(fp_write, 1 if resident_pair else 2),
                        tile_slice(gm_d_key_base, shape, (flat_tile, 0)),
                    )
                if const_expr(not resident_pair):
                    mem_copy(
                        self._head_batch_plane(fp_write, 3),
                        tile_slice(gm_d_key_pair, shape, (flat_tile, 0)),
                    )
                q_rstd_tile = tile_slice(
                    from_tensor(
                        fp_write,
                        offset=fp_planes
                        * self.head_batch_rows
                        * SUPPORTED_HEAD_DIM
                        * FP32_ELEMENT_BYTES,
                        layout=make_layout((self.head_batch_rows, 8)),
                    ),
                    (self.head_batch_rows, 1),
                    (0, 0),
                )
                k_rstd_tile = tile_slice(
                    from_tensor(
                        fp_write,
                        offset=(
                            fp_planes * self.head_batch_rows * SUPPORTED_HEAD_DIM
                            + self.head_batch_rows * 8
                        )
                        * FP32_ELEMENT_BYTES,
                        layout=make_layout((self.head_batch_rows, 8)),
                    ),
                    (self.head_batch_rows, 1),
                    (0, 0),
                )
                if use_l2norm:
                    mem_copy(
                        q_rstd_tile,
                        _token_rows_tile(
                            gm_q_rstd[effective_batch, effective_head, None, None],
                            effective_row,
                            self.head_batch_rows,
                            1,
                        ),
                    )
                    mem_copy(
                        k_rstd_tile,
                        _token_rows_tile(
                            gm_k_rstd[effective_batch, effective_head, None, None],
                            effective_row,
                            self.head_batch_rows,
                            1,
                        ),
                    )
                low_write = self.value_output.next_write()
                if use_l2norm and not resident_pair:
                    mem_copy(
                        self._head_batch_plane(low_write, 0),
                        _token_rows_tile(
                            query_source,
                            effective_row,
                            self.head_batch_rows,
                            SUPPORTED_HEAD_DIM,
                        ),
                    )
                    mem_copy(
                        self._head_batch_plane(low_write, 1),
                        _token_rows_tile(
                            key_source,
                            effective_row,
                            self.head_batch_rows,
                            SUPPORTED_HEAD_DIM,
                        ),
                    )
                fp_read = self.value_gradient.next_read()
                # Consume the output slot once before selecting the VF variant.
                # Both variants write the same planes; a conditional channel
                # cursor obscures their shared resource from sync verification.
                low_rows = self.value_output.next_read()
                query_base = self._head_batch_plane(fp_read, 0)
                query_pair = self._head_batch_plane(fp_read, 1)
                key_base = self._head_batch_plane(fp_read, 1 if resident_pair else 2)
                key_pair = self._head_batch_plane(fp_read, 3)
                query_out = self._head_batch_plane(low_rows, 2)
                key_out = self._head_batch_plane(low_rows, 3)
                query_norm = self._head_batch_plane(low_rows, 0)
                key_norm = self._head_batch_plane(low_rows, 1)
                if const_expr(resident_pair):
                    query_norm = tile_slice(self.batch_bf16[0], shape, (group, 0))
                    key_norm = tile_slice(self.batch_bf16[1], shape, (group, 0))
                query_rstd = from_tensor(
                    fp_read,
                    offset=fp_planes
                    * self.head_batch_rows
                    * SUPPORTED_HEAD_DIM
                    * FP32_ELEMENT_BYTES,
                    layout=make_layout((self.head_batch_rows, 8)),
                )
                key_rstd = from_tensor(
                    fp_read,
                    offset=(
                        fp_planes * self.head_batch_rows * SUPPORTED_HEAD_DIM
                        + self.head_batch_rows * 8
                    )
                    * FP32_ELEMENT_BYTES,
                    layout=make_layout((self.head_batch_rows, 8)),
                )
                if const_expr(self.batch_rows):
                    if const_expr(resident_pair):
                        self._finish_single_head_batch(
                            query_base,
                            tile_slice(self.batch_fp32[0], shape, (group, 0)),
                            key_base,
                            tile_slice(self.batch_fp32[1], shape, (group, 0)),
                            query_norm,
                            key_norm,
                            query_out,
                            key_out,
                            query_rstd,
                            key_rstd,
                            use_l2norm,
                        )
                    else:
                        self._finish_single_head_batch(
                            query_base,
                            query_pair,
                            key_base,
                            key_pair,
                            query_norm,
                            key_norm,
                            query_out,
                            key_out,
                            query_rstd,
                            key_rstd,
                            use_l2norm,
                        )
                else:
                    self._finish_single_head_batch(
                        query_base,
                        query_pair,
                        key_base,
                        key_pair,
                        query_norm,
                        key_norm,
                        query_out,
                        key_out,
                        query_rstd,
                        key_rstd,
                        use_l2norm,
                    )
                mem_copy(
                    _token_rows_tile(
                        query_destination,
                        raw_row,
                        self.head_batch_rows,
                        SUPPORTED_HEAD_DIM,
                    ),
                    self._head_batch_plane(low_rows, 2),
                )
                mem_copy(
                    _token_rows_tile(
                        key_destination,
                        raw_row,
                        self.head_batch_rows,
                        SUPPORTED_HEAD_DIM,
                    ),
                    self._head_batch_plane(low_rows, 3),
                )

    @jit
    def _load_gate_window(self, destination, source, row, valid_rows, half):
        if const_expr(source.rank == 4):
            window = source[
                None, None, row : row + valid_rows, half * D_HALF : (half + 1) * D_HALF
            ]
        else:
            window = source[row : row + valid_rows, half * D_HALF : (half + 1) * D_HALF]
        _load_zero_padded(
            destination,
            tile_slice(window, (CHUNK_SIZE, D_HALF), (0, 0))[:valid_rows, None],
        )

    @jit
    def gate_padded_chunk(
        self,
        gm_d_gate_base,
        gm_d_gate_pair,
        gm_raw_gate,
        gm_dt_bias,
        gm_public_d_gate,
        flat_chunk,
        raw_row_start,
        valid_rows,
        raw_half,
        raw_head,
        alog,
        has_bias: dtypes.int64,
        softplus: dtypes.int64,
        lower_bound: dtypes.float32,
    ):
        if has_bias:
            mem_copy(
                tile_slice(self.fp32_rows[2], (1, D_HALF), (0, 0)),
                tile_slice(gm_dt_bias, (1, D_HALF), (raw_head, raw_half)),
            )
        else:
            self._zero_fp32_row(self.fp32_rows[2], D_HALF)
        bias = self.fp32_rows[2]
        shape = (CHUNK_SIZE, D_HALF)
        fp_write = self.value_gradient.next_write()
        self._load_gate_window(
            from_tensor(fp_write, layout=make_layout(shape)),
            gm_d_gate_base,
            flat_chunk * CHUNK_SIZE,
            valid_rows,
            raw_half,
        )
        self._load_gate_window(
            from_tensor(
                fp_write,
                offset=CHUNK_SIZE * D_HALF * FP32_ELEMENT_BYTES,
                layout=make_layout(shape),
            ),
            gm_d_gate_pair,
            flat_chunk * CHUNK_SIZE,
            valid_rows,
            raw_half,
        )
        low_write = self.value_output.next_write()
        if const_expr(gm_raw_gate.dtype == dtypes.float32):
            self._load_gate_window(
                from_tensor(
                    fp_write,
                    offset=2 * CHUNK_SIZE * D_HALF * FP32_ELEMENT_BYTES,
                    layout=make_layout(shape),
                ),
                gm_raw_gate,
                raw_row_start,
                valid_rows,
                raw_half,
            )
        else:
            self._load_gate_window(
                from_tensor(low_write, layout=make_layout(shape)),
                gm_raw_gate,
                raw_row_start,
                valid_rows,
                raw_half,
            )
        fp_read = self.value_gradient.next_read()
        low_read = self.value_output.next_read()
        base = from_tensor(fp_read, layout=make_layout(shape))
        pair = from_tensor(
            fp_read,
            offset=CHUNK_SIZE * D_HALF * FP32_ELEMENT_BYTES,
            layout=make_layout(shape),
        )
        if const_expr(gm_raw_gate.dtype == dtypes.float32):
            raw = from_tensor(
                fp_read,
                offset=2 * CHUNK_SIZE * D_HALF * FP32_ELEMENT_BYTES,
                layout=make_layout(shape),
            )
        else:
            raw = from_tensor(low_read, layout=make_layout(shape))
        if const_expr(gm_public_d_gate.dtype == dtypes.float32):
            output = raw
        else:
            output = from_tensor(
                low_read,
                offset=CHUNK_SIZE * D_HALF * BF16_ELEMENT_BYTES,
                layout=make_layout(shape),
            )
        # Invalid rows are zero in both gradient planes. They contribute
        # exactly zero before the first valid reverse-scan row.
        self._compute_gate_batch(
            base, pair, raw, bias, output, alog, True, softplus, lower_bound
        )
        destination = gm_public_d_gate[
            None,
            None,
            raw_row_start : raw_row_start + valid_rows,
            raw_half * D_HALF : (raw_half + 1) * D_HALF,
        ]
        mem_copy(
            tile_slice(destination, (CHUNK_SIZE, D_HALF), (0, 0))[:valid_rows, None],
            output[:valid_rows, None],
        )

    @jit
    def store_value_window(
        self, source, source_row, destination, destination_row, valid_rows
    ):
        rows = CHUNK_SIZE // 2
        if valid_rows > 0:
            mem_copy(
                from_tensor(
                    self.value_output.next_write(),
                    layout=make_layout((rows, VALUE_DIM)),
                ),
                tile_slice(source, (rows, VALUE_DIM), (source_row // rows, 0)),
            )
            packed = from_tensor(
                self.value_output.next_read(), layout=make_layout((rows, VALUE_DIM))
            )
            window = destination[
                None, None, destination_row : destination_row + valid_rows, None
            ]
            mem_copy(
                tile_slice(window, (CHUNK_SIZE // 2, VALUE_DIM), (0, 0))[
                    :valid_rows, None
                ],
                packed[:valid_rows, None],
            )

    @jit
    def zero_task_padding(
        self,
        gm_d_value,
        gm_d_beta,
        gm_d_gate,
        gm_d_query,
        gm_d_key,
        sequence,
        value_head,
        heads_per_key_head,
        length,
    ):
        value_destination = gm_d_value[sequence, value_head, None, None]
        gate_destination = gm_d_gate[sequence, value_head, None, None]
        beta_destination = gm_d_beta[sequence, value_head, None, 0]
        for row in range(
            length + get_subblock_id(), gm_d_value.shape[2], get_subblock_dim()
        ):
            self.zero_output_row(value_destination, row)
            self.zero_output_row(gate_destination, row)
        if get_subblock_id() == 0:
            for row in range(length, gm_d_beta.shape[2]):
                self.zero_beta_scalar(beta_destination, row)
        if value_head % heads_per_key_head == heads_per_key_head - 1:
            qk_head = value_head // heads_per_key_head
            query_destination = gm_d_query[sequence, qk_head, None, None]
            key_destination = gm_d_key[sequence, qk_head, None, None]
            for row in range(
                length + get_subblock_id(), gm_d_query.shape[2], get_subblock_dim()
            ):
                self.zero_output_row(query_destination, row)
                self.zero_output_row(key_destination, row)

    @jit
    def prefetch_fused_base(
        self, gm_qg, gm_kg, gm_gamma, gm_dkbeta, gm_dkg, gm_dgamma, chunk
    ):
        # Causal operands and dV have released these depth-one UB slots.
        # The full cached causal combine does not read or write either slot.
        self._reset_fused_terminal()
        if get_subblock_id() == 1:
            shape = (CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)
            mem_copy(
                from_tensor(self.value_output.next_write(), layout=make_layout(shape)),
                tile_slice(gm_kg, shape, (chunk * 2, 0)),
            )
            mem_copy(
                from_tensor(
                    self.value_gradient.next_write(), layout=make_layout(shape)
                ),
                tile_slice(gm_dkg, shape, (chunk * 2, 0)),
            )
            self._fused_terminal_prefix(
                from_tensor(self.value_output.next_read(), layout=make_layout(shape)),
                from_tensor(self.value_gradient.next_read(), layout=make_layout(shape)),
            )
        shape = (CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)
        tile = chunk * 2 + get_subblock_id()
        fp = self.value_gradient.next_write()
        low = self.value_output.next_write()
        mem_copy(tile_slice(fp, shape, (0, 0)), tile_slice(gm_dkbeta, shape, (tile, 0)))
        mem_copy(tile_slice(fp, shape, (1, 0)), tile_slice(gm_dkg, shape, (tile, 0)))
        mem_copy(tile_slice(low, shape, (0, 0)), tile_slice(gm_qg, shape, (tile, 0)))
        mem_copy(tile_slice(low, shape, (1, 0)), tile_slice(gm_kg, shape, (tile, 0)))
        mem_copy(
            self.fp32_rows[0],
            tile_slice(
                gm_gamma,
                (1, SUPPORTED_HEAD_DIM),
                (chunk * CHUNK_SIZE + CHUNK_SIZE - 1, 0),
            ),
        )
        mem_copy(
            self.fp32_rows[1],
            tile_slice(gm_dgamma, (1, SUPPORTED_HEAD_DIM), (chunk, 0)),
        )

    @jit
    def fuse_base_chunk(
        self,
        gm_keys,
        gm_qg,
        gm_kg,
        gm_gamma,
        gm_dkbeta,
        gm_dqg,
        gm_dkg,
        gm_dgamma,
        gm_gate_pair,
        chunk,
        resident_dqg=None,
    ):
        # Prefetch is complete at the causal handoff. The causal cache has
        # overwritten gamma with inverse exponents; reload raw gamma now.
        shape = (CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)
        fp = self.value_gradient.next_read()
        low = self.value_output.next_read()
        mem_copy(
            self.wy_diagonal_residual[0],
            tile_slice(gm_gamma, shape, (chunk * 2 + get_subblock_id(), 0)),
        )
        self._fuse_base_batch(
            self.batch_bf16[1],
            tile_slice(low, shape, (0, 0)),
            tile_slice(low, shape, (1, 0)),
            self.wy_diagonal_residual[0],
            tile_slice(fp, shape, (0, 0)),
            resident_dqg,
            tile_slice(fp, shape, (1, 0)),
            self.fp32_rows[0],
            self.fp32_rows[1],
            get_subblock_id() * (CHUNK_SIZE // 2),
            self.batch_fp32[0],
            self.batch_fp32[1],
            self.batch_fp32[2],
        )
        mem_copy(
            tile_slice(gm_gate_pair, shape, (chunk * 2 + get_subblock_id(), 0)),
            self.batch_fp32[2],
        )

    @jit
    def _reset_fused_terminal(self):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=mask)
            vstore(self.fused_terminal, 0, zero, mask)
            vstore(self.fused_terminal, D_HALF, zero, mask)
            vmem_bar("vst_vld")

    @jit
    def _fused_terminal_prefix(self, kg, dkg):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            for half in range(2):
                terminal = vload(self.fused_terminal, half * D_HALF)
                for row in range(CHUNK_SIZE // 2):
                    offset = row * SUPPORTED_HEAD_DIM + half * D_HALF
                    kg_v = vcast(
                        vload_unpack(kg, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=mask,
                    )
                    terminal = vadd(
                        terminal, vmul(kg_v, vload(dkg, offset), mask=mask), mask=mask
                    )
                vstore(self.fused_terminal, half * D_HALF, terminal, mask)
            vmem_bar("vst_vld")

    @jit
    def _fuse_base_batch(
        self,
        key,
        qg,
        kg,
        gamma,
        dkbeta,
        dqg,
        dkg,
        last_gamma,
        dgamma,
        first,
        out_q,
        out_k,
        out_g,
    ):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            for half in range(2):
                terminal = vload(self.fused_terminal, half * D_HALF)
                last_g = vload(last_gamma, half * D_HALF)
                dg = vload(dgamma, half * D_HALF)
                for row in range(32):
                    offset = row * SUPPORTED_HEAD_DIM + half * D_HALF
                    token = first + row
                    key_v = vcast(
                        vload_unpack(key, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=mask,
                    )
                    qg_v = vcast(
                        vload_unpack(qg, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=mask,
                    )
                    kg_v = vcast(
                        vload_unpack(kg, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=mask,
                    )
                    g = vload(gamma, offset)
                    eg = vexp(g, mask=mask)
                    ktilde = vmul(eg, key_v, mask=mask)
                    dkt = vmul(
                        vload_broadcast(self.beta_tile, token),
                        vload(dkbeta, offset),
                        mask=mask,
                    )
                    dq = vload(dqg, offset)
                    dk = vload(dkg, offset)
                    kr_term = vmul(kg_v, dk, mask=mask)
                    rho = vexp(vsub(last_g, g, mask=mask), mask=mask)
                    qgrad = vmul(eg, dq, mask=mask)
                    kgrad = vadd(
                        vmul(eg, dkt, mask=mask), vmul(rho, dk, mask=mask), mask=mask
                    )
                    ggrad = vsub(
                        vadd(
                            vmul(qg_v, dq, mask=mask),
                            vmul(ktilde, dkt, mask=mask),
                            mask=mask,
                        ),
                        kr_term,
                        mask=mask,
                    )
                    terminal = vadd(terminal, kr_term, mask=mask)
                    last_row = veqs(
                        vdups(token, dtypes.int32, mask=mask), CHUNK_SIZE - 1, mask=mask
                    )
                    old_ggrad = ggrad
                    ggrad = vadd(ggrad, terminal, mask=mask)
                    ggrad = vadd(ggrad, vmul(eg, dg, mask=mask), mask=mask)
                    ggrad = vselect(ggrad, old_ggrad, cond_mask=last_row)
                    vstore(
                        out_q,
                        offset,
                        vadd(qgrad, vload(out_q, offset), mask=mask),
                        mask,
                    )
                    vstore(
                        out_k,
                        offset,
                        vadd(kgrad, vload(out_k, offset), mask=mask),
                        mask,
                    )
                    vstore(
                        out_g,
                        offset,
                        vadd(ggrad, vload(out_g, offset), mask=mask),
                        mask,
                    )
                vstore(self.fused_terminal, half * D_HALF, terminal, mask)
            vmem_bar("vst_vld")


class BaseGradientVector:
    """Own UB rows for the FP32 value, query, key, and gate gradients."""

    def __init__(self, shared: ChunkWorkspace):
        self.batch_rows = CAUSAL_BATCH_ROWS
        self.batch_float = shared.value_gradient
        self.batch_low = shared.value_output
        self.last_gamma, self.dgamma = shared.fp32_rows[4:6]
        self.beta = shared.beta_values
        self.terminal_sum = shared.terminal_sum

    @jit
    def process_chunk(
        self,
        gm_key,
        gm_qg,
        gm_kg,
        gm_gamma,
        gm_beta,
        gm_dkbeta,
        gm_dqg,
        gm_dkg,
        gm_dgamma,
        gm_out_q,
        gm_out_k,
        gm_out_g,
        chunk,
        length,
    ):
        row_base = chunk * CHUNK_SIZE
        mem_copy(self.beta.next_write(), tile_slice(gm_beta, (CHUNK_SIZE,), (chunk,)))
        beta = self.beta.next_read()
        for half in range(get_subblock_id(), 2, get_subblock_dim()):
            col = half
            mem_copy(
                tile_slice(self.last_gamma, (1, LANES), (0, 0)),
                tile_slice(gm_gamma, (1, LANES), (row_base + length - 1, col)),
            )
            mem_copy(
                tile_slice(self.dgamma, (1, LANES), (0, 0)),
                tile_slice(gm_dgamma, (1, LANES), (chunk, col)),
            )
            last_gamma = self.last_gamma
            dgamma = self.dgamma
            self._reset_terminal()
            self._process_chunk_batch(
                gm_key,
                gm_qg,
                gm_kg,
                gm_gamma,
                gm_dkbeta,
                gm_dqg,
                gm_dkg,
                gm_out_q,
                gm_out_k,
                gm_out_g,
                beta,
                last_gamma,
                dgamma,
                chunk,
                col,
                length,
            )

    @jit
    def _batch_plane(self, slot, plane: Constexpr[int], element_bytes: Constexpr[int]):
        # Compact batch-by-64 planes do not inherit the slot's 128-column stride.
        return from_tensor(
            slot,
            offset=plane * self.batch_rows * LANES * element_bytes,
            layout=make_layout((self.batch_rows, LANES)),
        )

    @jit
    def _process_chunk_batch(
        self,
        gm_key,
        gm_qg,
        gm_kg,
        gm_gamma,
        gm_dkbeta,
        gm_dqg,
        gm_dkg,
        gm_out_q,
        gm_out_k,
        gm_out_g,
        beta,
        last_gamma,
        dgamma,
        chunk,
        col,
        length,
    ):
        # Each AIV still owns one 64-feature half and visits tokens 0..63 in
        # order. Batching the DMA must not change terminal_sum's FP32 recurrence.
        # At eight rows, four input plus three output FP32 planes use 14 KiB;
        # three BF16 input planes use 3 KiB of the existing Channels.
        for group in range(CHUNK_SIZE // self.batch_rows):
            token_start = group * self.batch_rows
            row_tile = chunk * (CHUNK_SIZE // self.batch_rows) + group
            shape = (self.batch_rows, LANES)
            low_write = self.batch_low.next_write()
            mem_copy(
                self._batch_plane(low_write, 0, BF16_ELEMENT_BYTES),
                tile_slice(gm_key, shape, (row_tile, col)),
            )
            mem_copy(
                self._batch_plane(low_write, 1, BF16_ELEMENT_BYTES),
                tile_slice(gm_qg, shape, (row_tile, col)),
            )
            mem_copy(
                self._batch_plane(low_write, 2, BF16_ELEMENT_BYTES),
                tile_slice(gm_kg, shape, (row_tile, col)),
            )
            fp_write = self.batch_float.next_write()
            mem_copy(
                self._batch_plane(fp_write, 0, FP32_ELEMENT_BYTES),
                tile_slice(gm_gamma, shape, (row_tile, col)),
            )
            mem_copy(
                self._batch_plane(fp_write, 1, FP32_ELEMENT_BYTES),
                tile_slice(gm_dkbeta, shape, (row_tile, col)),
            )
            mem_copy(
                self._batch_plane(fp_write, 2, FP32_ELEMENT_BYTES),
                tile_slice(gm_dqg, shape, (row_tile, col)),
            )
            mem_copy(
                self._batch_plane(fp_write, 3, FP32_ELEMENT_BYTES),
                tile_slice(gm_dkg, shape, (row_tile, col)),
            )
            low_read = self.batch_low.next_read()
            fp_read = self.batch_float.next_read()
            key = self._batch_plane(low_read, 0, BF16_ELEMENT_BYTES)
            qg = self._batch_plane(low_read, 1, BF16_ELEMENT_BYTES)
            kg = self._batch_plane(low_read, 2, BF16_ELEMENT_BYTES)
            gamma = self._batch_plane(fp_read, 0, FP32_ELEMENT_BYTES)
            dkbeta = self._batch_plane(fp_read, 1, FP32_ELEMENT_BYTES)
            dqg = self._batch_plane(fp_read, 2, FP32_ELEMENT_BYTES)
            dkg = self._batch_plane(fp_read, 3, FP32_ELEMENT_BYTES)
            out_q = self._batch_plane(fp_read, 4, FP32_ELEMENT_BYTES)
            out_k = self._batch_plane(fp_read, 5, FP32_ELEMENT_BYTES)
            out_g = self._batch_plane(fp_read, 6, FP32_ELEMENT_BYTES)
            self._compute_batch(
                key,
                qg,
                kg,
                gamma,
                dkbeta,
                dqg,
                dkg,
                beta,
                last_gamma,
                dgamma,
                token_start,
                length,
                out_q,
                out_k,
                out_g,
            )
            mem_copy(
                tile_slice(gm_out_q, shape, (row_tile, col)),
                self._batch_plane(fp_read, 4, FP32_ELEMENT_BYTES),
            )
            mem_copy(
                tile_slice(gm_out_k, shape, (row_tile, col)),
                self._batch_plane(fp_read, 5, FP32_ELEMENT_BYTES),
            )
            mem_copy(
                tile_slice(gm_out_g, shape, (row_tile, col)),
                self._batch_plane(fp_read, 6, FP32_ELEMENT_BYTES),
            )

    @jit
    def _reset_terminal(self):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            vstore(self.terminal_sum, 0, vdups(0.0, dtypes.float32, mask=mask), mask)
            vmem_bar("vst_vld")

    @jit
    def _compute_batch(
        self,
        key,
        qg,
        kg,
        gamma,
        dkbeta,
        dqg,
        dkg,
        beta,
        last_gamma,
        dgamma,
        token_start,
        length,
        out_q,
        out_k,
        out_g,
    ):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            terminal = vload(self.terminal_sum, 0)
            for row in range(self.batch_rows):
                token = token_start + row
                offset = row * LANES
                qgrad = vdups(0.0, dtypes.float32, mask=mask)
                kgrad = vdups(0.0, dtypes.float32, mask=mask)
                ggrad = vdups(0.0, dtypes.float32, mask=mask)
                row_valid = vlts(
                    vdups(token, dtypes.int32, mask=mask), length, mask=mask
                )
                row_valid_qgrad = qgrad
                row_valid_kgrad = kgrad
                row_valid_ggrad = ggrad
                row_valid_terminal = terminal
                key_v = vcast(
                    vload_unpack(key, offset, unpack_mode=UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=mask,
                )
                qg_v = vcast(
                    vload_unpack(qg, offset, unpack_mode=UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=mask,
                )
                kg_v = vcast(
                    vload_unpack(kg, offset, unpack_mode=UnpackMode.B16_TO_B32),
                    dtypes.float32,
                    mask=mask,
                )
                g = vload(gamma, offset)
                beta_v = vload_broadcast(beta, token)
                dkb = vload(dkbeta, offset)
                dq = vload(dqg, offset)
                dk = vload(dkg, offset)
                eg = vexp(g, mask=mask)
                ktilde = vmul(eg, key_v, mask=mask)
                dkt = vmul(beta_v, dkb, mask=mask)
                kr_term = vmul(kg_v, dk, mask=mask)
                rho = vexp(vsub(vload(last_gamma, 0), g, mask=mask), mask=mask)
                qgrad = vmul(eg, dq, mask=mask)
                kgrad = vadd(
                    vmul(eg, dkt, mask=mask), vmul(rho, dk, mask=mask), mask=mask
                )
                ggrad = vsub(
                    vadd(
                        vmul(qg_v, dq, mask=mask),
                        vmul(ktilde, dkt, mask=mask),
                        mask=mask,
                    ),
                    kr_term,
                    mask=mask,
                )
                terminal = vadd(terminal, kr_term, mask=mask)
                last_row = veqs(
                    vdups(token, dtypes.int32, mask=mask), length - 1, mask=row_valid
                )
                last_row_ggrad = ggrad
                ggrad = vadd(ggrad, terminal, mask=mask)
                ggrad = vadd(ggrad, vmul(eg, vload(dgamma, 0), mask=mask), mask=mask)
                ggrad = vselect(ggrad, last_row_ggrad, cond_mask=last_row)
                qgrad = vselect(qgrad, row_valid_qgrad, cond_mask=row_valid)
                kgrad = vselect(kgrad, row_valid_kgrad, cond_mask=row_valid)
                ggrad = vselect(ggrad, row_valid_ggrad, cond_mask=row_valid)
                terminal = vselect(terminal, row_valid_terminal, cond_mask=row_valid)
                vstore(out_q, offset, qgrad, mask)
                vstore(out_k, offset, kgrad, mask)
                vstore(out_g, offset, ggrad, mask)
            vstore(self.terminal_sum, 0, terminal, mask)
            vmem_bar("vst_vld")


class ValueBetaEpilogueVector:
    """Own one AIV half of FP32 value gradients and BF16 output storage."""

    def __init__(self, shared: ChunkWorkspace):
        self.beta = shared.beta_values
        self.d_value = shared.value_gradient
        self.d_value_out = shared.value_output

    @jit
    def process_chunk(self, gm_beta, gm_d_value, gm_d_value_out, chunk, valid_rows):
        subblock_idx = get_subblock_id()
        row_tile = chunk * get_subblock_dim() + subblock_idx
        source = tile_slice(gm_d_value, (CHUNK_SIZE // 2, VALUE_DIM), (row_tile, 0))
        destination = tile_slice(
            gm_d_value_out, (CHUNK_SIZE // 2, VALUE_DIM), (row_tile, 0)
        )
        mem_copy(self.beta.next_write(), tile_slice(gm_beta, (CHUNK_SIZE,), (chunk,)))
        mem_copy(
            tile_slice(self.d_value.next_write(), (CHUNK_SIZE // 2, VALUE_DIM), (0, 0)),
            source,
        )
        self._compute(
            self.beta.next_read(),
            self.d_value.next_read(),
            self.d_value_out.next_write(),
            valid_rows,
            subblock_idx * (CHUNK_SIZE // 2),
        )
        mem_copy(
            destination,
            tile_slice(
                self.d_value_out.next_read(), (CHUNK_SIZE // 2, VALUE_DIM), (0, 0)
            ),
        )

    @jit
    def _compute(self, beta, d_value, output, valid_rows, row_start):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            zero = vdups(0.0, dtypes.float32, mask=full)
            for token in range(CHUNK_SIZE // 2):
                for half in range(2):
                    offset = token * VALUE_DIM + half * LANES
                    value = zero
                    row_valid = vlts(
                        vdups(row_start + token, dtypes.int32, mask=full),
                        valid_rows,
                        mask=full,
                    )
                    row_valid_value = value
                    value = vmul(
                        vload(d_value, offset),
                        vload_broadcast(beta, row_start + token),
                        mask=full,
                    )
                    value = vselect(value, row_valid_value, cond_mask=row_valid)
                    vstore_pack(
                        output,
                        offset,
                        vcast(value, dtypes.bfloat16, mask=full),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )


class BetaCombineVector:
    """Combine FP32 beta branches and round only for BF16 outputs."""

    def __init__(self, shared: ChunkWorkspace):
        self.beta = shared.beta_values
        self.d_beta_wy = shared.beta_gradient
        self.d_beta_out = shared.bf16_scalars
        self.beta_gradient_sum = shared.beta_gradient_sum

    @jit
    def process_chunk_single_pair(
        self,
        gm_beta,
        gm_d_beta_wy,
        gm_d_beta_pair,
        gm_d_beta_out,
        chunk,
        valid_rows,
        fused: dtypes.int64,
        inverse_factor: dtypes.float32,
    ):
        mem_copy(self.beta.next_write(), tile_slice(gm_beta, (CHUNK_SIZE,), (chunk,)))
        mem_copy(
            self.d_beta_wy.next_write(),
            tile_slice(gm_d_beta_wy, (CHUNK_SIZE,), (chunk,)),
        )
        self._save_wy_gradient(self.d_beta_wy.next_read())
        mem_copy(
            self.d_beta_wy.next_write(),
            tile_slice(gm_d_beta_pair, (CHUNK_SIZE,), (chunk,)),
        )
        if const_expr(gm_d_beta_out.dtype == dtypes.float32):
            output = self.beta_gradient_sum
        else:
            output = self.d_beta_out.next_write()
        self._combine_single_pair(
            self.beta.next_read(),
            self.d_beta_wy.next_read(),
            output,
            valid_rows,
            fused,
            inverse_factor,
        )
        if const_expr(gm_d_beta_out.dtype != dtypes.float32):
            output = self.d_beta_out.next_read()
        mem_copy(
            tile_slice(gm_d_beta_out, (CHUNK_SIZE,), (chunk,)),
            output,
        )

    @jit
    def _save_wy_gradient(self, d_beta_wy):
        with vf(mode="simd"):
            mask = create_mask(pattern="all")
            vstore(self.beta_gradient_sum, 0, vload(d_beta_wy, 0), mask)
            vmem_bar("vst_vld")

    @jit
    def _combine_single_pair(
        self,
        beta,
        d_beta_pair,
        output,
        valid_rows,
        fused: dtypes.int64,
        inverse_factor: dtypes.float32,
    ):
        # Runtime switches are resolved on the scalar side of the VF boundary.
        if fused:
            self._combine_single_pair_constexpr(
                beta,
                d_beta_pair,
                output,
                valid_rows,
                True,
                inverse_factor,
            )
        else:
            self._combine_single_pair_constexpr(
                beta,
                d_beta_pair,
                output,
                valid_rows,
                False,
                inverse_factor,
            )

    @jit
    def _combine_single_pair_constexpr(
        self,
        beta,
        d_beta_pair,
        output,
        valid_rows,
        fused: Constexpr[bool],
        inverse_factor: dtypes.float32,
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            total = vadd(
                vload(self.beta_gradient_sum, 0), vload(d_beta_pair, 0), mask=full
            )
            if const_expr(fused):
                activated = vload(beta, 0)
                derivative = vmul(
                    activated,
                    vadds(vmuls(activated, -inverse_factor, mask=full), 1.0, mask=full),
                    mask=full,
                )
                total = vmul(total, derivative, mask=full)
            valid, _ = update_mask(valid_rows, elem_bits=32)
            total = vselect(
                total, vdups(0.0, dtypes.float32, mask=full), cond_mask=valid
            )
            if const_expr(output.dtype == dtypes.float32):
                vstore(output, 0, total, full)
            else:
                vstore_pack(
                    output,
                    0,
                    vcast(total, dtypes.bfloat16, mask=full),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )


@jit
def _run_single_chunk_gate_sensitivity(
    gm_query,
    gm_key,
    gm_value,
    gm_gamma,
    gm_beta,
    gm_states,
    gm_do,
    gm_dht,
    gm_metadata,
    gm_dA,
    gm_d_beta,
    gm_raw_gate,
    gm_A_log,
    gm_dt_bias,
    lower_bound: dtypes.float32,
    softplus: dtypes.int64,
    has_bias: dtypes.int64,
    scale: dtypes.float32,
    heads_per_key_head: dtypes.int64,
    beta_fp32: Constexpr[bool],
    padded_layout: dtypes.int64,
    fused_beta: dtypes.int64,
    inverse_factor: dtypes.float32,
    same_owner: Constexpr[bool],
    gm_owner_schedule,
    gm_sensitivity_scratch,
    round_raw_gate: Constexpr[bool] = False,
):
    # Stage 1 suppresses both rounded-cache dA and valid-row dbeta stores
    # for these tasks. The fused path keeps the recompute track owner.
    # The initialization handoff publishes dA zeroing before either phase.
    sensitivity = SingleChunkGateSensitivity(round_raw_gate=round_raw_gate)
    task_count = (gm_metadata.shape[0] - HEADER_WORDS) // TASK_WORDS
    if (
        gm_metadata[HEADER_STATUS] != STATUS_OK
        or gm_metadata[HEADER_MAGIC] != MAGIC
        or gm_metadata[HEADER_ABI_VERSION] != ABI_VERSION
    ):
        task_count = 0
    tasks_per_owner = dtypes.int64(1)
    if const_expr(same_owner):
        tasks_per_owner = heads_per_key_head
    begin = dtypes.int64(get_block_idx())
    end = dtypes.int64(task_count // tasks_per_owner)
    step = dtypes.int64(get_block_num())
    if const_expr(same_owner):
        begin, end, step = _owner_loop_bounds(
            gm_owner_schedule, task_count // tasks_per_owner
        )
    for slot in range(begin, end, step):
        owner = dtypes.int64(slot)
        if const_expr(same_owner):
            owner = _scheduled_owner(gm_owner_schedule, slot)
        for local_head in range(tasks_per_owner):
            task = owner * tasks_per_owner + local_head
            record = HEADER_WORDS + task * TASK_WORDS
            sequence = gm_metadata[record + TASK_SEQUENCE]
            head = gm_metadata[record + TASK_VALUE_HEAD]
            length = gm_metadata[record + TASK_LENGTH]
            compact_start = gm_metadata[record + TASK_COMPACT_TOKEN_START]
            if length > 0 and length <= CHUNK_SIZE:
                qk_head = head // heads_per_key_head
                token_batch = 0
                token_start = compact_start
                if padded_layout:
                    token_batch = sequence
                    token_start = 0
                state_batch = dtypes.int64(sequence)
                state_chunk = dtypes.int64(0)
                if not padded_layout:
                    state_batch = 0
                    state_chunk = _tnd_chunk_index(compact_start, sequence, 0)
                state = gm_states[state_batch, head, state_chunk, None, None]
                sensitivity.process_task(
                    gm_query[token_batch, qk_head, None, None],
                    gm_key[token_batch, qk_head, None, None],
                    gm_gamma[token_batch, head, None, None],
                    gm_value[token_batch, head, None, None],
                    gm_do[token_batch, head, None, None],
                    state,
                    gm_dht[sequence, head, None, None],
                    gm_beta,
                    gm_dA,
                    gm_d_beta[token_batch, head, None, None],
                    gm_sensitivity_scratch,
                    token_start,
                    token_start,
                    token_batch,
                    head,
                    token_start,
                    length,
                    scale,
                    beta_fp32,
                    fused_beta,
                    inverse_factor,
                    gm_raw_gate[token_batch, head, None, None],
                    gm_dt_bias,
                    gm_A_log[head],
                    lower_bound,
                    softplus,
                    has_bias,
                )


class SingleChunkGateSensitivity:
    """Rebuild a short chunk's A_log and beta derivatives entirely in FP32.

    A dot denotes d/dA_log. Softplus satisfies G_dot = G, so cumulative
    gamma doubles as its own derivative; bounded gates instead accumulate
    gamma_dot = cumsum(x * B * sigmoid(x) * (1 - sigmoid(x))) from the raw
    gate. Both modes publish the same gamma_dot buffer, and every derivative
    position reads gamma_dot while value positions keep reading gamma.
    For each value column solve L*z = rhs, then L*zdot = rhsdot - Ldot*z,
    where L is the unit-lower chunk system and z is the corrected value.
    Differentiate the chunk output and final-state formulas using z/zdot;
    their inner products with do/dht give this value column's dA contribution.
    """

    def __init__(self, round_raw_gate=False):
        self.round_raw_gate = round_raw_gate
        # All DMA-facing slots are depth one and Vector-only. Keep one
        # logical Buffer ID across their disjoint, 32-byte-aligned byte windows.
        # The six windows total exactly 67,360 bytes; arithmetic scratch below
        # remains separate and total UB capacity is unchanged.
        io_storage = RingBuffer(
            MemLoc.UB, dtypes.uint8, make_layout((67872,)), depth=1
        ).next_write()
        self.qk = from_tensor(
            io_storage,
            dtype=dtypes.bfloat16,
            layout=make_layout((2 * CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            offset=0,
        )
        self.gamma = from_tensor(
            io_storage,
            dtype=dtypes.float32,
            layout=make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
            offset=32768,
        )
        # Probe: coefficients are independent of value channel. Cache in FP32.
        self.coefficients = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((4, CHUNK_SIZE, CHUNK_SIZE))
        )
        self.beta_partial = from_tensor(
            io_storage,
            dtype=dtypes.float32,
            layout=make_layout((1, CHUNK_SIZE)),
            offset=65536,
        )
        self.beta = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, CHUNK_SIZE)))
        self.corrected = Buffer(MemLoc.UB, dtypes.float32, make_layout((2, CHUNK_SIZE)))
        # U, dZ, and the cross-value beta sum. Zdot storage becomes the
        # triangular adjoint only after the last output has consumed it.
        self.beta_work = Buffer(MemLoc.UB, dtypes.float32, make_layout((3, CHUNK_SIZE)))
        self.low_io = from_tensor(
            io_storage,
            dtype=dtypes.bfloat16,
            layout=make_layout((2, SUPPORTED_HEAD_DIM)),
            offset=65792,
        )
        self.state_io = from_tensor(
            io_storage,
            dtype=dtypes.float32,
            layout=make_layout((2, SUPPORTED_HEAD_DIM)),
            offset=66304,
        )
        self.value_output = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((2, CHUNK_SIZE, D_HALF))
        )
        self.loss = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, GATE_SCALAR_TRANSFER_WIDTH))
        )
        # Kahan compensation for the long dA accumulation: thousands of
        # cancelling FP32 terms otherwise drift at the same order as the
        # peer's own error, which the cross_check ratio then amplifies.
        self.loss_error = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, GATE_SCALAR_TRANSFER_WIDTH))
        )
        self.publish = from_tensor(
            io_storage,
            dtype=dtypes.float32,
            layout=make_layout((1, GATE_SCALAR_TRANSFER_WIDTH)),
            offset=67328,
        )
        # FP32 staging for one raw gate or bias row during the bounded build.
        self.gate_io = from_tensor(
            io_storage,
            dtype=dtypes.float32,
            layout=make_layout((1, SUPPORTED_HEAD_DIM)),
            offset=67360,
        )
        # Cumulative d(gamma)/d(A_log); identical to gamma for softplus.
        # Plain scratch: filled by vector code after the gamma loads, so no
        # cross-pipe synchronization is needed.
        self.gate_dot = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((CHUNK_SIZE, SUPPORTED_HEAD_DIM))
        )
        self.gate_running = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM))
        )
        self.gate_bias = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM))
        )

    @jit
    def _copy_gate_dot(self, gamma, valid_rows):
        """Softplus: d(gamma)/d(A_log) equals gamma itself, row by row."""
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for row in range(valid_rows):
                for half in range(2):
                    offset = row * SUPPORTED_HEAD_DIM + half * D_HALF
                    vstore(
                        self.gate_dot,
                        offset,
                        vload(gamma, offset),
                        full,
                    )
            vmem_bar("vst_vld")

    @jit
    def _zero_gate_bias(self):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            vstore(
                self.gate_bias,
                0,
                vdups(0.0, dtypes.float32, mask=full),
                full,
            )
            vmem_bar("vst_vld")

    @jit
    def _load_gate_bias(self, bias_source, value_head):
        mem_copy(
            tile_slice(self.gate_io, (1, SUPPORTED_HEAD_DIM), (0, 0)),
            tile_slice(bias_source, (1, SUPPORTED_HEAD_DIM), (value_head, 0)),
        )
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for half in range(2):
                offset = half * D_HALF
                vstore(
                    self.gate_bias,
                    offset,
                    vload(self.gate_io, offset),
                    full,
                )
            vmem_bar("vst_vld")

    @jit
    def _build_gate_dot_bounded_rows(
        self,
        gate_source,
        alog,
        bound,
        token_start,
        valid_rows,
        fp32_gate: Constexpr[bool],
    ):
        """Stage one full raw gate row per step, extend gamma_dot."""
        for row in range(valid_rows):
            if const_expr(fp32_gate):
                mem_copy(
                    tile_slice(self.gate_io, (1, SUPPORTED_HEAD_DIM), (0, 0)),
                    tile_slice(
                        gate_source, (1, SUPPORTED_HEAD_DIM), (token_start + row, 0)
                    ),
                )
                staged = self.gate_io
            else:
                mem_copy(
                    tile_slice(self.low_io, (1, SUPPORTED_HEAD_DIM), (0, 0)),
                    tile_slice(
                        gate_source, (1, SUPPORTED_HEAD_DIM), (token_start + row, 0)
                    ),
                )
                staged = self.low_io
            self._gate_dot_row(staged, row, alog, bound, fp32_gate)

    @jit
    def _gate_dot_row(self, staged, row, alog, bound, fp32_gate: Constexpr[bool]):
        # Both halves on every AIV: the coefficient and solve stages read the
        # full gate_dot rows, so each AIV publishes a complete local copy.
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for half in range(2):
                offset = half * D_HALF
                if const_expr(fp32_gate):
                    raw = vload(staged, offset)
                    if const_expr(self.round_raw_gate):
                        raw = vcast(
                            vcast(raw, dtypes.bfloat16, mask=full),
                            dtypes.float32,
                            mask=full,
                        )
                else:
                    raw = vcast(
                        vload_unpack(staged, offset, unpack_mode=UnpackMode.B16_TO_B32),
                        dtypes.float32,
                        mask=full,
                    )
                bias = vload(self.gate_bias, offset)
                pre = vadd(raw, bias, mask=full)
                # Match the general dA path's logit convention: the bounded
                # gate is B * sigmoid(+alpha * (bias + g)), so x here equals
                # d(logit)/d(A_log) = alpha * (bias + g) with its own sign.
                alpha = vexp(vdups(alog, dtypes.float32, mask=full), mask=full)
                x = vmul(alpha, pre, mask=full)
                sigmoid = vdiv(
                    vdups(1.0, dtypes.float32, mask=full),
                    vadds(vexp(x, mask=full), 1.0, mask=full),
                    mask=full,
                )
                one_minus = vadds(vmuls(sigmoid, -1.0, mask=full), 1.0, mask=full)
                # G_dot = x * B * s * (1 - s); x carries the sign, so the
                # bound factor enters with its public negative value.
                step = vmuls(
                    vmul(vmul(x, sigmoid, mask=full), one_minus, mask=full),
                    bound,
                    mask=full,
                )
                running = vadd(vload(self.gate_running, offset), step, mask=full)
                vstore(self.gate_running, offset, running, full)
                vstore(self.gate_dot, row * SUPPORTED_HEAD_DIM + offset, running, full)
            vmem_bar("vst_vld")

    @jit
    def _build_gate_dot(
        self,
        gamma,
        gate_source,
        bias_source,
        alog,
        bound,
        softplus,
        has_bias,
        value_head,
        token_start,
        valid_rows,
    ):
        """Publish cumulative d(gamma)/d(A_log) into gate_dot for both modes."""
        if softplus:
            self._copy_gate_dot(gamma, valid_rows)
        else:
            if has_bias:
                self._load_gate_bias(bias_source, value_head)
            else:
                self._zero_gate_bias()
            if const_expr(gate_source.dtype == dtypes.float32):
                self._build_gate_dot_bounded_rows(
                    gate_source, alog, bound, token_start, valid_rows, True
                )
            else:
                self._build_gate_dot_bounded_rows(
                    gate_source, alog, bound, token_start, valid_rows, False
                )

    @jit
    def process_task(
        self,
        q_source,
        k_source,
        gamma_source,
        value_source,
        output_gradient_source,
        initial_state_source,
        final_gradient_source,
        gm_beta,
        gm_public_dA,
        beta_destination,
        gm_sensitivity_scratch,
        token_start,
        cache_start,
        beta_batch,
        value_head,
        beta_start,
        valid_rows,
        scale: dtypes.float32,
        beta_fp32: Constexpr[bool],
        fused_beta: dtypes.int64,
        inverse_factor: dtypes.float32,
        gate_source,
        bias_source,
        gate_alog: dtypes.float32,
        gate_bound: dtypes.float32,
        gate_softplus: dtypes.int64,
        gate_has_bias: dtypes.int64,
    ):
        """One sequence/value-head task; caller ensures 0 < length <= 64.

        Each AIV reconstructs 64 value columns. Both publish FP32 dA;
        AIV 0 combines FP32 beta partials before the public dtype conversion.
        Stage 1 suppresses its old dA/dbeta stores for exactly these tasks;
        other outputs and all padded-row zero stores retain their path.
        """
        qk = self.qk
        gamma = self.gamma
        for row in range(valid_rows):
            mem_copy(
                tile_slice(qk, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                tile_slice(q_source, (1, SUPPORTED_HEAD_DIM), (token_start + row, 0)),
            )
            mem_copy(
                tile_slice(qk, (1, SUPPORTED_HEAD_DIM), (CHUNK_SIZE + row, 0)),
                tile_slice(k_source, (1, SUPPORTED_HEAD_DIM), (token_start + row, 0)),
            )
            mem_copy(
                tile_slice(gamma, (1, SUPPORTED_HEAD_DIM), (row, 0)),
                tile_slice(
                    gamma_source, (1, SUPPORTED_HEAD_DIM), (cache_start + row, 0)
                ),
            )
            if const_expr(beta_fp32):
                beta = gm_beta[beta_batch, value_head, beta_start + row, 0]
                self._save_beta(row, beta)
            else:
                # Use the same rank-two DMA view as Stage 1. Reinterpreting
                # an all-singleton rank-four tile is unsupported by the SDK.
                beta_source = gm_beta[beta_batch, value_head, None, None]
                mem_copy(
                    tile_slice(self.low_io, (1, 1), (0, 0)),
                    tile_slice(beta_source, (1, 1), (beta_start + row, 0)),
                )
                self._save_beta_low(row, self.low_io)
            io = self.low_io
            mem_copy(
                tile_slice(io, (1, SUPPORTED_HEAD_DIM), (0, 0)),
                tile_slice(
                    value_source, (1, SUPPORTED_HEAD_DIM), (token_start + row, 0)
                ),
            )
            mem_copy(
                tile_slice(io, (1, SUPPORTED_HEAD_DIM), (1, 0)),
                tile_slice(
                    output_gradient_source,
                    (1, SUPPORTED_HEAD_DIM),
                    (token_start + row, 0),
                ),
            )
            self._convert_value_output(self.low_io, row)
        qk = self.qk
        gamma = self.gamma
        # Both gate modes publish cumulative d(gamma)/d(A_log) first; every
        # derivative position below reads gate_dot, value positions read gamma.
        self._build_gate_dot(
            gamma,
            gate_source,
            bias_source,
            gate_alog,
            gate_bound,
            gate_softplus,
            gate_has_bias,
            value_head,
            token_start,
            valid_rows,
        )
        self._build_coefficients(qk, gamma, valid_rows)
        self._zero_loss()
        for value in range(
            get_subblock_id() * D_HALF, (get_subblock_id() + 1) * D_HALF
        ):
            state = self.state_io
            state_row = tile_slice(state, (1, SUPPORTED_HEAD_DIM), (0, 0))
            source_row = tile_slice(
                initial_state_source, (1, SUPPORTED_HEAD_DIM), (value, 0)
            )
            if const_expr(initial_state_source.dtype == dtypes.bfloat16):
                low_row = tile_slice(self.low_io, (1, SUPPORTED_HEAD_DIM), (0, 0))
                mem_copy(low_row, source_row)
                _promote_state_row(state_row, low_row)
            else:
                mem_copy(state_row, source_row)
                with vf(mode="simd"):
                    mask = create_mask(pattern="all")
                    for offset in range(0, SUPPORTED_HEAD_DIM, D_HALF):
                        value_fp32 = vload(state_row, offset)
                        rounded = vcast(value_fp32, dtypes.bfloat16, mask=mask)
                        vstore(
                            state_row,
                            offset,
                            vcast(rounded, dtypes.float32, mask=mask),
                            mask,
                        )
                    vmem_bar("vst_vld")
            mem_copy(
                tile_slice(state, (1, SUPPORTED_HEAD_DIM), (1, 0)),
                tile_slice(final_gradient_source, (1, SUPPORTED_HEAD_DIM), (value, 0)),
            )
            state = self.state_io
            self._begin_value(
                state,
                gamma,
                valid_rows,
            )
            for row in range(valid_rows):
                self._solve_row(
                    qk,
                    gamma,
                    state,
                    row,
                    value,
                    valid_rows,
                    scale,
                )
            # The forward row sweep has now consumed every Zdot. Reuse
            # that storage for the reverse unit-upper adjoint solve.
            self._accumulate_beta_adjoint(qk, gamma, valid_rows)
        # Each AIV owns 64 value channels and contributes its FP32 dA sum.
        mem_copy(self.publish, self.loss)
        mem_copy(
            tile_slice(
                gm_public_dA,
                (1, GATE_SCALAR_TRANSFER_WIDTH),
                (_gate_parameter_row(gm_public_dA, value_head), 0),
            ),
            self.publish,
            options=copy_options(atomic_add=True),
        )
        # Reuse a retired Stage 1 FP32 matrix row for two disjoint beta partials.
        mem_copy(
            self.beta_partial,
            tile_slice(self.beta_work, (1, CHUNK_SIZE), (2, 0)),
        )
        mem_copy(
            tile_slice(
                gm_sensitivity_scratch,
                (1, CHUNK_SIZE),
                (
                    get_block_idx()
                    * (gm_sensitivity_scratch.shape[0] // get_block_num()),
                    get_subblock_id(),
                ),
            ),
            self.beta_partial,
        )
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, 2, mode=1)
        vec_sync_block_wait(PIPE.S, 2, mode=1)
        if get_subblock_id() == 0:
            mem_copy(
                self.beta_partial,
                tile_slice(
                    gm_sensitivity_scratch,
                    (1, CHUNK_SIZE),
                    (
                        get_block_idx()
                        * (gm_sensitivity_scratch.shape[0] // get_block_num()),
                        1,
                    ),
                ),
            )
            self._combine_beta_partials(self.beta_partial)
            # Only valid sequence rows are published. TND sequences and
            # GQA value heads own disjoint destinations; padded tails
            # remain exclusively owned by Stage 1's existing zero path.
            for row in range(valid_rows):
                if const_expr(beta_destination.dtype == dtypes.float32):
                    output_channel = self.publish
                else:
                    output_channel = self.low_io
                self._cast_beta_gradient(
                    output_channel, row, fused_beta, inverse_factor
                )
                mem_copy(
                    tile_slice(beta_destination, (1, 1), (token_start + row, 0)),
                    tile_slice(output_channel, (1, 1), (0, 0)),
                )
        # Protect partial storage before either AIV begins the next task.
        vec_sync_all()
        vec_sync_block_arrive(PIPE.MTE3, 3, mode=1)
        vec_sync_block_wait(PIPE.S, 3, mode=1)

    @jit
    def _combine_beta_partials(self, other):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            combined = vadd(
                vload(self.beta_work, 2 * CHUNK_SIZE), vload(other, 0), mask=full
            )
            vstore(self.beta_work, 2 * CHUNK_SIZE, combined, full)
            vmem_bar("vst_vld")

    @jit
    def _save_beta(self, row, beta):
        with vf(mode="simd"):
            scalar, _ = update_mask(1, elem_bits=32)
            vstore_first(
                self.beta,
                row,
                vdups(beta, dtypes.float32, mask=scalar),
            )
            # vstore_first updates a scalar inside a 32-byte storage unit.
            vmem_bar("vst_vld")

    @jit
    def _zero_loss(self):
        with vf(mode="simd"):
            scalar_storage, _ = update_mask(GATE_SCALAR_TRANSFER_WIDTH, elem_bits=32)
            vstore(
                self.loss,
                0,
                vdups(0.0, dtypes.float32, mask=scalar_storage),
                scalar_storage,
            )
            vstore(
                self.loss_error,
                0,
                vdups(0.0, dtypes.float32, mask=scalar_storage),
                scalar_storage,
            )
            full = create_mask(pattern="all")
            vstore(
                self.beta_work,
                2 * CHUNK_SIZE,
                vdups(0.0, dtypes.float32, mask=full),
                full,
            )
            vmem_bar("vst_vld")

    @jit
    def _save_beta_low(self, row, beta_source):
        with vf(mode="simd"):
            scalar, _ = update_mask(1, elem_bits=32)
            beta = vcast(
                vload_unpack(beta_source, 0, unpack_mode=UnpackMode.B16_TO_B32),
                dtypes.float32,
                mask=scalar,
            )
            vstore_first(self.beta, row, beta)
            vmem_bar("vst_vld")

    @jit
    def _convert_value_output(self, io, row):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            offset = get_subblock_id() * D_HALF
            value = vcast(
                vload_unpack(io, offset, unpack_mode=UnpackMode.B16_TO_B32),
                dtypes.float32,
                mask=full,
            )
            output = vcast(
                vload_unpack(
                    io, SUPPORTED_HEAD_DIM + offset, unpack_mode=UnpackMode.B16_TO_B32
                ),
                dtypes.float32,
                mask=full,
            )
            vstore(self.value_output, row * D_HALF, value, full)
            vstore(self.value_output, (CHUNK_SIZE + row) * D_HALF, output, full)
            vmem_bar("vst_vld")

    @jit
    def _begin_value(
        self,
        state,
        gamma,
        valid_rows,
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            scalar, _ = update_mask(1, elem_bits=32)
            zero = vdups(0.0, dtypes.float32, mask=full)
            vstore(self.corrected, 0, zero, full)
            vstore(self.corrected, CHUNK_SIZE, zero, full)
            vstore(self.beta_work, 0, zero, full)
            vstore(self.beta_work, CHUNK_SIZE, zero, full)
            loss = vload_broadcast(self.loss, 0)
            loss_error = vload_broadcast(self.loss_error, 0)
            for half in range(2):
                offset = half * D_HALF
                last = vload(gamma, (valid_rows - 1) * SUPPORTED_HEAD_DIM + offset)
                last_dot = vload(
                    self.gate_dot, (valid_rows - 1) * SUPPORTED_HEAD_DIM + offset
                )
                initial = vload(state, offset)
                seed = vload(state, SUPPORTED_HEAD_DIM + offset)
                direct = vmul(
                    vmul(
                        vmul(vexp(last, mask=full), last_dot, mask=full),
                        initial,
                        mask=full,
                    ),
                    seed,
                    mask=full,
                )
                increment = vsub(
                    vreduce_sum(direct, mask=full), loss_error, mask=scalar
                )
                updated = vadd(loss, increment, mask=scalar)
                loss_error = vsub(
                    vsub(updated, loss, mask=scalar), increment, mask=scalar
                )
                loss = updated
            vstore_first(self.loss, 0, loss)
            vstore_first(self.loss_error, 0, loss_error)
            vmem_bar("vst_vld")

    @jit
    def _build_coefficients(self, qk, gamma, valid_rows):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            scalar, _ = update_mask(1, elem_bits=32)
            for row in range(valid_rows):
                for col in range(row):
                    lower = vdups(0.0, dtypes.float32, mask=scalar)
                    lower_dot = vdups(0.0, dtypes.float32, mask=scalar)
                    output = vdups(0.0, dtypes.float32, mask=scalar)
                    output_dot = vdups(0.0, dtypes.float32, mask=scalar)
                    for half in range(2):
                        offset = half * D_HALF
                        qi = vcast(
                            vload_unpack(
                                qk,
                                row * SUPPORTED_HEAD_DIM + offset,
                                unpack_mode=UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=full,
                        )
                        ki = vcast(
                            vload_unpack(
                                qk,
                                (CHUNK_SIZE + row) * SUPPORTED_HEAD_DIM + offset,
                                unpack_mode=UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=full,
                        )
                        kj = vcast(
                            vload_unpack(
                                qk,
                                (CHUNK_SIZE + col) * SUPPORTED_HEAD_DIM + offset,
                                unpack_mode=UnpackMode.B16_TO_B32,
                            ),
                            dtypes.float32,
                            mask=full,
                        )
                        delta = vsub(
                            vload(gamma, row * SUPPORTED_HEAD_DIM + offset),
                            vload(gamma, col * SUPPORTED_HEAD_DIM + offset),
                            mask=full,
                        )
                        delta_dot = vsub(
                            vload(self.gate_dot, row * SUPPORTED_HEAD_DIM + offset),
                            vload(self.gate_dot, col * SUPPORTED_HEAD_DIM + offset),
                            mask=full,
                        )
                        key_ratio = vmul(kj, vexp(delta, mask=full), mask=full)
                        kk = vmul(ki, key_ratio, mask=full)
                        lower = vadd(lower, vreduce_sum(kk, mask=full), mask=scalar)
                        lower_dot = vadd(
                            lower_dot,
                            vreduce_sum(vmul(kk, delta_dot, mask=full), mask=full),
                            mask=scalar,
                        )
                        query_key = vmul(qi, key_ratio, mask=full)
                        output = vadd(
                            output, vreduce_sum(query_key, mask=full), mask=scalar
                        )
                        output_dot = vadd(
                            output_dot,
                            vreduce_sum(
                                vmul(query_key, delta_dot, mask=full), mask=full
                            ),
                            mask=scalar,
                        )
                    offset = row * CHUNK_SIZE + col
                    vstore_first(self.coefficients, offset, lower)
                    vstore_first(
                        self.coefficients, CHUNK_SIZE * CHUNK_SIZE + offset, lower_dot
                    )
                    vstore_first(
                        self.coefficients, 2 * CHUNK_SIZE * CHUNK_SIZE + offset, output
                    )
                    vstore_first(
                        self.coefficients,
                        3 * CHUNK_SIZE * CHUNK_SIZE + offset,
                        output_dot,
                    )
                    vmem_bar("vst_vld")

    @jit
    def _solve_row(
        self,
        qk,
        gamma,
        state,
        row,
        value,
        valid_rows,
        scale: dtypes.float32,
    ):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            scalar, _ = update_mask(1, elem_bits=32)
            beta = vload_broadcast(self.beta, row)
            # U is formed directly, never by Z / beta; beta may be zero.
            u = vload_broadcast(
                self.value_output, row * D_HALF + value - get_subblock_id() * D_HALF
            )
            z = vmul(beta, u, mask=scalar)
            zdot = vdups(0.0, dtypes.float32, mask=scalar)
            odot = vdups(0.0, dtypes.float32, mask=scalar)
            diagonal_m = vdups(0.0, dtypes.float32, mask=scalar)
            final_coefficient = vdups(0.0, dtypes.float32, mask=scalar)
            final_coefficient_dot = vdups(0.0, dtypes.float32, mask=scalar)
            # RHS and direct sensitivity from the supplied effective Q/K in FP32.
            for half in range(2):
                offset = half * D_HALF
                q = vcast(
                    vload_unpack(
                        qk,
                        row * SUPPORTED_HEAD_DIM + offset,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=full,
                )
                k = vcast(
                    vload_unpack(
                        qk,
                        (CHUNK_SIZE + row) * SUPPORTED_HEAD_DIM + offset,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    ),
                    dtypes.float32,
                    mask=full,
                )
                g = vload(gamma, row * SUPPORTED_HEAD_DIM + offset)
                g_dot = vload(self.gate_dot, row * SUPPORTED_HEAD_DIM + offset)
                e = vexp(g, mask=full)
                initial = vload(state, offset)
                key_initial = vmul(vmul(e, k, mask=full), initial, mask=full)
                u = vsub(u, vreduce_sum(key_initial, mask=full), mask=scalar)
                z = vsub(
                    z,
                    vmul(beta, vreduce_sum(key_initial, mask=full), mask=scalar),
                    mask=scalar,
                )
                zdot = vsub(
                    zdot,
                    vmul(
                        beta,
                        vreduce_sum(vmul(g_dot, key_initial, mask=full), mask=full),
                        mask=scalar,
                    ),
                    mask=scalar,
                )
                q_direct = vmul(
                    vmul(vmul(g_dot, e, mask=full), q, mask=full),
                    initial,
                    mask=full,
                )
                odot = vadd(
                    odot,
                    vmuls(vreduce_sum(q_direct, mask=full), scale, mask=scalar),
                    mask=scalar,
                )
                diagonal_m = vadd(
                    diagonal_m,
                    vmuls(
                        vreduce_sum(vmul(q, k, mask=full), mask=full),
                        scale,
                        mask=scalar,
                    ),
                    mask=scalar,
                )
                last = vload(gamma, (valid_rows - 1) * SUPPORTED_HEAD_DIM + offset)
                last_dot = vload(
                    self.gate_dot, (valid_rows - 1) * SUPPORTED_HEAD_DIM + offset
                )
                delta = vsub(last, g, mask=full)
                delta_dot = vsub(last_dot, g_dot, mask=full)
                kg = vmul(vexp(delta, mask=full), k, mask=full)
                seed = vload(state, SUPPORTED_HEAD_DIM + offset)
                kg_seed = vmul(kg, seed, mask=full)
                final_coefficient = vadd(
                    final_coefficient, vreduce_sum(kg_seed, mask=full), mask=scalar
                )
                final_coefficient_dot = vadd(
                    final_coefficient_dot,
                    vreduce_sum(vmul(delta_dot, kg_seed, mask=full), mask=full),
                    mask=scalar,
                )
            # Two unit-lower solves. j<i guarantees every exponential is of
            # a nonpositive relative gate; no exp(-G) or explicit inverse.
            for col in range(row):
                offset = row * CHUNK_SIZE + col
                lower = vload_broadcast(self.coefficients, offset)
                lower_dot = vload_broadcast(
                    self.coefficients, CHUNK_SIZE * CHUNK_SIZE + offset
                )
                output = vload_broadcast(
                    self.coefficients, 2 * CHUNK_SIZE * CHUNK_SIZE + offset
                )
                output_dot = vload_broadcast(
                    self.coefficients, 3 * CHUNK_SIZE * CHUNK_SIZE + offset
                )
                previous = vload_broadcast(self.corrected, col)
                previous_dot = vload_broadcast(self.corrected, CHUNK_SIZE + col)
                u = vsub(u, vmul(lower, previous, mask=scalar), mask=scalar)
                seed = vload_broadcast(
                    self.value_output,
                    (CHUNK_SIZE + row) * D_HALF + value - get_subblock_id() * D_HALF,
                )
                dz = vload_broadcast(self.beta_work, CHUNK_SIZE + col)
                dz = vadd(
                    dz,
                    vmuls(vmul(output, seed, mask=scalar), scale, mask=scalar),
                    mask=scalar,
                )
                vstore_first(self.beta_work, CHUNK_SIZE + col, dz)
                vmem_bar("vst_vld")
                lower = vmul(beta, lower, mask=scalar)
                lower_dot = vmul(beta, lower_dot, mask=scalar)
                z = vsub(z, vmul(lower, previous, mask=scalar), mask=scalar)
                zdot = vsub(
                    zdot,
                    vadd(
                        vmul(lower_dot, previous, mask=scalar),
                        vmul(lower, previous_dot, mask=scalar),
                        mask=scalar,
                    ),
                    mask=scalar,
                )
                pair = vadd(
                    vmul(output_dot, previous, mask=scalar),
                    vmul(output, previous_dot, mask=scalar),
                    mask=scalar,
                )
                odot = vadd(odot, vmuls(pair, scale, mask=scalar), mask=scalar)
            vstore_first(self.corrected, row, z)
            vstore_first(self.corrected, CHUNK_SIZE + row, zdot)
            vstore_first(self.beta_work, row, u)
            dz = final_coefficient
            seed = vload_broadcast(
                self.value_output,
                (CHUNK_SIZE + row) * D_HALF + value - get_subblock_id() * D_HALF,
            )
            dz = vadd(dz, vmul(diagonal_m, seed, mask=scalar), mask=scalar)
            vstore_first(self.beta_work, CHUNK_SIZE + row, dz)
            loss = vload_broadcast(self.loss, 0)
            odot = vadd(odot, vmul(diagonal_m, zdot, mask=scalar), mask=scalar)
            seed = vload_broadcast(
                self.value_output,
                (CHUNK_SIZE + row) * D_HALF + value - get_subblock_id() * D_HALF,
            )
            loss_error = vload_broadcast(self.loss_error, 0)
            increment = vsub(vmul(seed, odot, mask=scalar), loss_error, mask=scalar)
            updated = vadd(loss, increment, mask=scalar)
            loss_error = vsub(vsub(updated, loss, mask=scalar), increment, mask=scalar)
            loss = updated
            term = vadd(
                vmul(final_coefficient_dot, z, mask=scalar),
                vmul(final_coefficient, zdot, mask=scalar),
                mask=scalar,
            )
            increment = vsub(term, loss_error, mask=scalar)
            updated = vadd(loss, increment, mask=scalar)
            loss_error = vsub(vsub(updated, loss, mask=scalar), increment, mask=scalar)
            loss = updated
            vstore_first(self.loss, 0, loss)
            vstore_first(self.loss_error, 0, loss_error)
            vmem_bar("vst_vld")

    @jit
    def _accumulate_beta_adjoint(self, qk, gamma, valid_rows):
        """Solve L.T*a=dZ and accumulate a*U without rounded WY caches."""
        with vf(mode="simd"):
            scalar, _ = update_mask(1, elem_bits=32)
            for reverse_row in range(valid_rows):
                row = valid_rows - reverse_row - 1
                adjoint = vload_broadcast(self.beta_work, CHUNK_SIZE + row)
                for later in range(row + 1, valid_rows):
                    lower = vload_broadcast(self.coefficients, later * CHUNK_SIZE + row)
                    beta_later = vload_broadcast(self.beta, later)
                    adjoint_later = vload_broadcast(self.corrected, CHUNK_SIZE + later)
                    adjoint = vsub(
                        adjoint,
                        vmul(
                            vmul(beta_later, lower, mask=scalar),
                            adjoint_later,
                            mask=scalar,
                        ),
                        mask=scalar,
                    )
                vstore_first(self.corrected, CHUNK_SIZE + row, adjoint)
                residual = vload_broadcast(self.beta_work, row)
                total = vload_broadcast(self.beta_work, 2 * CHUNK_SIZE + row)
                total = vadd(total, vmul(adjoint, residual, mask=scalar), mask=scalar)
                vstore_first(self.beta_work, 2 * CHUNK_SIZE + row, total)
                # The next reverse row consumes this newly stored adjoint.
                vmem_bar("vst_vld")

    @jit
    def _cast_beta_gradient(
        self, output, row, fused_beta: dtypes.int64, inverse_factor: dtypes.float32
    ):
        # Runtime switches are resolved on the scalar side of the VF boundary.
        if fused_beta:
            self._cast_beta_gradient_constexpr(
                output,
                row,
                True,
                inverse_factor,
            )
        else:
            self._cast_beta_gradient_constexpr(
                output,
                row,
                False,
                inverse_factor,
            )

    @jit
    def _cast_beta_gradient_constexpr(
        self, output, row, fused_beta: Constexpr[bool], inverse_factor: dtypes.float32
    ):
        with vf(mode="simd"):
            scalar, _ = update_mask(1, elem_bits=32)
            total = vload_broadcast(self.beta_work, 2 * CHUNK_SIZE + row)
            if const_expr(fused_beta):
                beta = vload_broadcast(self.beta, row)
                derivative = vmul(
                    beta,
                    vadds(vmuls(beta, -inverse_factor, mask=scalar), 1.0, mask=scalar),
                    mask=scalar,
                )
                total = vmul(total, derivative, mask=scalar)
            if const_expr(output.dtype == dtypes.float32):
                vstore(output, 0, total, scalar)
            else:
                vstore_pack(
                    output,
                    0,
                    vcast(total, dtypes.bfloat16, mask=scalar),
                    scalar,
                    pack_mode=PackMode.B32_TO_B16,
                )
