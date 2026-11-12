# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashKDA CANNBotDSL training backward.

Persistent and four-stage schedules share one public interface. Saved states
and recurrent gradients remain FP32; backward Cube operands use BF16. Fused
cache reconstruction follows forward BF16 operands, FP32 accumulation and
FP32 recurrent state for every public gate/beta dtype. Shared backward
arithmetic, addressing and synchronization live here; the unified reconstruction
implementation is a sibling in the backward directory.
Metadata owns strategy selection and core routing; the strategy modules retain
their specialized pipelines and phase barriers.
"""

from __future__ import annotations

import atexit
from contextvars import ContextVar
import math
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from queue import Queue
from typing import Optional

import cannbotdsl
import torch

from cannbotdsl import (
    RoundingMode,
    from_tensor,
    Buffer,
    Constexpr,
    MemLoc,
    PIPE,
    const_expr,
    dtypes,
    get_platform_info,
)
from cannbotdsl import make_layout
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.vf import vf
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
    vges,
    vload,
    vload_broadcast,
    vload_unpack,
    vlog,
    vlts,
    vmadd,
    vmaxs,
    vmem_bar,
    vmins,
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
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
)
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.sync import (
    onchip_rewind,
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.tensor import tile_slice
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel

if __package__:
    from . import flash_kda_grad_metadata as _metadata
else:
    # Native-package collection also supports flattened sample modules.
    import flash_kda_grad_metadata as _metadata


HEADER_WORDS = _metadata.HEADER_WORDS
HEADER_MAGIC = _metadata.HEADER_MAGIC
HEADER_ABI_VERSION = _metadata.HEADER_ABI_VERSION
HEADER_STATUS = _metadata.HEADER_STATUS
HEADER_BATCH = _metadata.HEADER_BATCH
HEADER_VALUE_HEADS = _metadata.HEADER_VALUE_HEADS
HEADER_STORAGE_LENGTH = _metadata.HEADER_STORAGE_LENGTH
HEADER_CHUNK_COUNT = _metadata.HEADER_CHUNK_COUNT
HEADER_FLAGS = _metadata.HEADER_FLAGS
MAGIC = _metadata.MAGIC
ABI_VERSION = _metadata.ABI_VERSION
STATUS_OK = _metadata.STATUS_OK
TASK_WORDS = _metadata.TASK_WORDS
TASK_SEQUENCE = _metadata.TASK_SEQUENCE
TASK_COMPACT_TOKEN_START = _metadata.TASK_COMPACT_TOKEN_START
TASK_CHUNKS = _metadata.TASK_CHUNKS
TASK_LENGTH = _metadata.TASK_LENGTH
TASK_VALUE_HEAD = _metadata.TASK_VALUE_HEAD
metadata_capacity = _metadata.metadata_capacity
wait_metadata_ready = _metadata.wait_metadata_ready

CHUNK_SIZE = 64
SUPPORTED_HEAD_DIM = KEY_DIM = VALUE_DIM = 128
LANES = D_HALF = 64
CAUSAL_BATCH_ROWS = 8
# Gate epilogue can use a full chunk independently of causal batch sizing.
# Two FP32 planes use 32 KiB; BF16 raw/output planes use the 16 KiB slot.
GATE_BATCH_ROWS = 64
FP32_ELEMENT_BYTES = 4
BF16_ELEMENT_BYTES = 2
CUBE_TILE_ROWS = 16
NZ_BLOCK_WIDTH = 16
GATE_SCALAR_TRANSFER_WIDTH = 8
# A conservative margin below BF16 exp overflow (~88) for causal gate spans.
MAX_AIC_CORES = _metadata.MAX_AIC_CORES
TASK_FIRST_CHUNK = _metadata.TASK_FIRST_CHUNK
TASK_FLAT_CHUNK_START = _metadata.TASK_FLAT_CHUNK_START


# Inline routing policies are owned by metadata and shared by both templates.
_owner_loop_bounds = _metadata._owner_loop_bounds
_scheduled_owner = _metadata._scheduled_owner

# ---------------------------------------------------------------------------
# Shared Cube/Vector casts and recurrent-gradient primitives
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
        if self.rows > 1 and gm_source.shape[-2] % self.rows == 0:
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


class StageTwoReversePreVector:
    """Reduce each AIV's 64 key channels in the original value-row order."""

    def __init__(self, rows: int = 1):
        self.rows = rows
        storage = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((2 * rows + 2, D_HALF)), depth=1
        ).next_write()
        self.state = _FixedScratchSlot(storage[:rows, None])
        self.saved_state = _FixedScratchSlot(storage[rows : 2 * rows, None])
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
            mem_copy(
                self.saved_state.next_write(),
                tile_slice(gm_saved_state, (self.rows, D_HALF), (group, subblock_idx)),
            )
            self._accumulate_gate(self.state.next_read(), self.saved_state.next_read())
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
    def _accumulate_gate(self, state, saved_state):
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            total = vload(self.gate_accumulator, 0)
            for row in range(self.rows):
                # Each vmadd still produces FP32 in exactly the original
                # row order; keeping that FP32 value in a register avoids
                # a redundant UB store/reload inside this DMA group.
                total = vmadd(
                    vload(state, row * state.physical_layout.stride[0]),
                    vload(saved_state, row * saved_state.physical_layout.stride[0]),
                    total,
                    mask=full,
                )
            vstore(self.gate_accumulator, 0, total, full)
            vmem_bar("vst_vld")


class StageTwoReversePostVector:
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


class StageTwoOutputVector:
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


@jit
def _token_rows_tile(
    source, row, rows: Constexpr[int], width: Constexpr[int], column=0
):
    """Keep packed token offsets and physical strides for a full row window."""
    row_end = min(row + rows, source.shape[-2])
    window = source[None, None, row:row_end, None]
    return tile_slice(window, (rows, width), (0, column))


@jit
def _tnd_chunk_index(
    token_start: dtypes.int64,
    sequence: dtypes.int64,
    chunk: dtypes.int64,
):
    """Locate a chunk in the current TND forward's reserved storage."""
    return token_start // CHUNK_SIZE + sequence + chunk


@jit
def _full_token_tile(source, row, width: Constexpr[int]):
    """A full 64-token GM window at an arbitrary packed sequence origin."""
    row_end = min(row + CHUNK_SIZE, source.shape[-2])
    window = source[None, None, row:row_end, None]
    return tile_slice(window, (CHUNK_SIZE, width), (0, 0))


def _begin_cube_vector_handoff(flag_id: int, grid: bool = False):
    """Publish completed GM writes, leaving AIV free to prefetch before waiting.

    Callers use flag IDs 0..3. The three handoff directions reserve separate
    groups of four IDs (0..3, 4..7, 8..11) to prevent cross-pipe reuse.
    """
    cube_sync_all()
    vec_sync_all()
    vec_sync_block_arrive(PIPE.MTE3, flag_id, mode=2)
    cube_sync_block_wait(PIPE.S, flag_id, mode=2)
    if grid:
        cube_sync_block_arrive(PIPE.FIXPIPE, flag_id + 4, mode=0)
        cube_sync_block_wait(PIPE.S, flag_id + 4, mode=0)
    cube_sync_block_arrive(PIPE.MTE3, flag_id + 8, mode=2)


def _finish_cube_vector_handoff(flag_id: int, reset_sync_id: bool = False):
    vec_sync_block_wait(PIPE.S, flag_id + 8, mode=2)
    onchip_rewind(reset_sync_id=reset_sync_id)


def _sync_cube_vector_handoff(
    flag_id: int, grid: bool = False, reset_sync_id: bool = False
):
    _begin_cube_vector_handoff(flag_id, grid)
    _finish_cube_vector_handoff(flag_id, reset_sync_id=reset_sync_id)


@jit
def _load_zero_padded(destination, source):
    """Initialize a full ND tile, then load only its valid GM rectangle.

    Explicit VF fill avoids a dynamic NDDMA padding-width requirement. Keep
    the same selected RingBuffer slot for the fill, DMA and later consumer.
    """
    rows, columns = destination.shape
    row_stride = destination.physical_layout.stride[0]
    with vf(mode="simd"):
        for row in range(rows):
            for column in range(0, columns, LANES):
                mask, _ = update_mask(min(LANES, columns - column), elem_bits=32)
                zero = vdups(0.0, dtypes.float32, mask=mask)
                if const_expr(destination.dtype == dtypes.bfloat16):
                    low = vcast(zero, dtypes.bfloat16, mask=mask)
                    vstore_pack(
                        destination,
                        row * row_stride + column,
                        low,
                        mask,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                else:
                    vstore(destination, row * row_stride + column, zero, mask)
        vmem_bar("vst_vld")
    if source.shape[0] > 0 and source.shape[1] > 0:
        mem_copy(
            destination[: source.shape[0], : source.shape[1]],
            source,
            options=copy_options(engine="nddma"),
        )


class StageTwoVector:
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


# L0 operands/results are depth-one scratch. Share their storage and Buffer IDs
# over the kernel's sequential contractions instead of keeping one ID per helper.
_CUBE_SCRATCH_SCOPE = ContextVar("kdag_cube_scratch", default=None)
_STAGE_ONE_SCRATCH_SCOPE = ContextVar("kdag_stage_one_scratch", default=None)


class _SharedCubeScope:
    def __enter__(self):
        maximum = SUPPORTED_HEAD_DIM
        resources = {
            MemLoc.L0A: RingBuffer(
                MemLoc.L0A,
                dtypes.bfloat16,
                make_layout((maximum, maximum)),
                depth=1,
                data_format="nz",
                physical_layout=make_layout((maximum // 16, maximum // 16, 16, 16)),
            ).next_write(),
            MemLoc.L0B: RingBuffer(
                MemLoc.L0B,
                dtypes.bfloat16,
                make_layout((maximum, maximum)),
                depth=1,
                data_format="nz",
                physical_layout=make_layout((maximum // 16, maximum // 16, 16, 16)),
            ).next_write(),
            # W and U coexist during reconstruction. Separate column panels
            # keep their NZ storage disjoint, including for short matrices.
            MemLoc.L0C: RingBuffer(
                MemLoc.L0C,
                dtypes.float32,
                make_layout((maximum, 2 * maximum)),
                depth=1,
                data_format="nz",
                physical_layout=make_layout((2 * maximum // 16, maximum // 16, 16, 16)),
            ).next_write(),
        }
        self.token = _CUBE_SCRATCH_SCOPE.set(resources)

    def __exit__(self, *_):
        _CUBE_SCRATCH_SCOPE.reset(self.token)


class _FixedScratchSlot:
    def __init__(self, tensor):
        self.tensor = tensor

    def next_write(self):
        return self.tensor

    def next_read(self):
        return self.tensor


class _StageOneScratchScope:
    """Allocate shared full/tail temporaries after Stage 2 has retired."""

    def __init__(self, resident_dr=None, resident_dw=None, resident_dqg=None):
        self.has_resident_gradients = resident_dqg is not None
        self.resident_dqg = resident_dqg
        self.resident_dr = resident_dr
        self.resident_dw = resident_dw
        half = CHUNK_SIZE // 2
        self.fp32_rows = 15 + 3 * half
        self.fp32_rows += half if resident_dr is None else 0
        self.fp32_rows += half if resident_dw is None else 0

    def __enter__(self):
        half = CHUNK_SIZE // 2
        if self.has_resident_gradients:
            # The Stage 2 handoff rewinds address allocation, including live
            # plain Buffer reservations. Re-declare the identical three leading
            # slots in the new epoch before allocating any Stage 1 scratch.
            # These aliases retain the bytes at 0/16/32 KiB; no copy or cast is
            # needed. Keep their order identical to run_persistent_flash_kda_grad.
            self.resident_dqg, self.resident_dr, self.resident_dw = tuple(
                Buffer(MemLoc.UB, dtypes.float32, make_layout((half, KEY_DIM)))
                for _ in range(3)
            )
        fields = [
            ("fp32", dtypes.float32, (self.fp32_rows, SUPPORTED_HEAD_DIM)),
            ("bf16", dtypes.bfloat16, (8 + 2 * half, SUPPORTED_HEAD_DIM)),
            ("terminal", dtypes.float32, (1, SUPPORTED_HEAD_DIM)),
            ("batch_scalar", dtypes.float32, (1, half)),
            ("scalars", dtypes.float32, (1, 5 * CHUNK_SIZE + 16 + 592)),
            ("low_scalars", dtypes.bfloat16, (1, CHUNK_SIZE)),
            (
                "value_gradient",
                dtypes.float32,
                (max(3 * half, 12 * CAUSAL_BATCH_ROWS), VALUE_DIM),
            ),
            ("value_output", dtypes.bfloat16, (2 * CHUNK_SIZE, VALUE_DIM)),
        ]
        pool = {}
        for name, dtype, shape in fields:
            pool[(name, dtype, shape)] = RingBuffer(
                MemLoc.UB, dtype, make_layout(shape), depth=1
            ).next_write()
        self.token = _STAGE_ONE_SCRATCH_SCOPE.set(pool)
        return self

    def __exit__(self, *_):
        _STAGE_ONE_SCRATCH_SCOPE.reset(self.token)


def _stage_one_scratch(name, dtype, shape):
    """Reuse matching Stage 1 storage across mutually exclusive chunk branches."""
    key = (name, dtype, tuple(shape))
    pool = _STAGE_ONE_SCRATCH_SCOPE.get()
    if pool is not None and key in pool:
        return pool[key]
    tensor = RingBuffer(MemLoc.UB, dtype, make_layout(shape), depth=1).next_write()
    if pool is not None:
        pool[key] = tensor
    return tensor


def _cube_scratch_view(mem_loc, dtype, layout, physical_layout, data_format, slot):
    resources = _CUBE_SCRATCH_SCOPE.get()
    if resources is None:
        return None
    maximum = SUPPORTED_HEAD_DIM
    shape = layout.shape
    assert len(shape) == 2 and all(0 < width <= maximum for width in shape)
    assert dtype == (dtypes.float32 if mem_loc == MemLoc.L0C else dtypes.bfloat16)
    tensor = resources[mem_loc]
    if mem_loc == MemLoc.L0C:
        assert slot in (0, 1)
        tensor = tile_slice(tensor, (maximum, maximum), (0, slot))
    else:
        assert slot == 0
    return from_tensor(
        tensor,
        layout=layout,
        physical_layout=physical_layout
        or make_layout(((shape[1] + 15) // 16, (shape[0] + 15) // 16, 16, 16)),
        data_format=data_format or "nz",
    )


def _cube_buffer(
    mem_loc, dtype, layout, *, physical_layout=None, data_format=None, _slot=0
):
    tensor = _cube_scratch_view(
        mem_loc, dtype, layout, physical_layout, data_format, _slot
    )
    if tensor is not None:
        return tensor
    return Buffer(
        mem_loc, dtype, layout, physical_layout=physical_layout, data_format=data_format
    )


def _cube_ring(
    mem_loc, dtype, layout, *, depth, physical_layout=None, data_format=None, _slot=0
):
    tensor = _cube_scratch_view(
        mem_loc, dtype, layout, physical_layout, data_format, _slot
    )
    if tensor is not None:
        assert depth == 1, "Shared L0 scratch requires a depth-one producer/consumer"
        return _FixedScratchSlot(tensor)
    return RingBuffer(
        mem_loc,
        dtype,
        layout,
        depth=depth,
        physical_layout=physical_layout,
        data_format=data_format,
    )


class _MatrixSlotView:
    """Expose a compact matrix over shared sequential on-chip storage."""

    def __init__(self, storage, shape):
        self.storage = storage
        self.layout = make_layout(shape)
        self.physical_layout = make_layout(
            ((shape[1] + 15) // 16, (shape[0] + 15) // 16, 16, 16)
        )

    def _view(self, tensor):
        return from_tensor(
            tensor,
            layout=self.layout,
            physical_layout=self.physical_layout,
            data_format="nz",
        )

    def next_write(self):
        return self._view(self.storage.next_write())

    def next_read(self):
        return self._view(self.storage.next_read())


class StageTwoMatmul:
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
def _stage_two_cast_packed_nz(dst, src, scratch, rows, cols):
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


class StageTwoMatmulResources:
    """Share state publication and result handoff across reverse products."""

    def __init__(self, value_rows):
        capacity_rows = max(value_rows, CHUNK_SIZE)
        self.operand_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((capacity_rows, KEY_DIM)), depth=2
        )
        self.token_gm_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((CHUNK_SIZE, KEY_DIM)), depth=2
        )
        self.result_to_vector = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((capacity_rows // 2, KEY_DIM)),
            depth=2,
        )
        self.operand_nz = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((capacity_rows // 2, KEY_DIM)),
            depth=1,
            data_format="nz",
            physical_layout=make_layout(
                (
                    ((capacity_rows // 2, KEY_DIM)[1] + 15) // 16,
                    ((capacity_rows // 2, KEY_DIM)[0] + 15) // 16,
                    16,
                    16,
                ),
                stride=(
                    ((capacity_rows // 2, KEY_DIM)[0] + 15) // 16 * 256
                    + NZ_BLOCK_WIDTH,
                    256,
                    16,
                    1,
                ),
            ),
        ).next_write()
        self.row_scratch = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((KEY_DIM,)))


class StageTwoPipelineMatmul:
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
        _stage_two_cast_packed_nz(
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


# ---------------------------------------------------------------------------
# Shared chunk preparation and dependent gradients
# ---------------------------------------------------------------------------


@jit
def _gradient_slot(metadata, task, chunk, staged: Constexpr[bool]):
    """Per-core persistent storage or a disjoint global task/chunk slot."""
    slot = get_block_idx()
    if const_expr(staged):
        record = HEADER_WORDS + task * TASK_WORDS
        slot = metadata[record + TASK_FLAT_CHUNK_START] + task + chunk
    return slot


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
    matmul = StageTwoMatmul()
    vector = StageTwoVector()
    output_vector = StageTwoOutputVector()
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
        if not padded_layout:
            matrix_batch = _tnd_chunk_index(compact_token_start, seq_idx, chunk)
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
    matmul = StageTwoMatmul()
    vector = StageTwoVector()
    pre_vector = StageTwoReversePreVector(rows=16)
    post_vector = StageTwoReversePostVector()
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


# ---------------------------------------------------------------------------
# Shared WY, local gradients and output epilogues
# ---------------------------------------------------------------------------


class StageOneMatmul:
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


class StageOneVector:
    """Shared UB channels for dependent stage-one vector phases.

    A channel is reused only after its preceding producer and consumer have
    completed. The 16 channels stay under the per-vector-core depth limit.
    """

    def __init__(
        self,
        batch_rows: bool = False,
        direct_results=False,
        fuse_base=False,
        resident_dr=None,
        resident_dw=None,
    ):
        self.direct_results = direct_results
        self.fuse_base = fuse_base
        if fuse_base:
            self.fused_terminal = _stage_one_scratch(
                "terminal", dtypes.float32, (1, SUPPORTED_HEAD_DIM)
            )
        # 16-row batches fit the 32-KiB direct-result pool only when dQ/dK stay
        # resident (two feature planes). The non-resident direct epilogue needs
        # four planes, so it keeps the eight-row batch.
        self.head_batch_rows = (
            32 if fuse_base else (16 if direct_results else CAUSAL_BATCH_ROWS)
        )
        self.causal_results = None
        if direct_results:
            result_pool = RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((3 * CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM)),
                depth=1,
            ).next_write()
            self.causal_results = tuple(
                tile_slice(
                    result_pool, (CHUNK_SIZE // 2, SUPPORTED_HEAD_DIM), (plane, 0)
                )
                for plane in range(3)
            )
        self.causal_l1 = None
        self.stacked_matrix = None
        if direct_results:
            # Explicit chunk handoffs protect these persistent shared L1 tiles.
            # Each AIV publishes its own disjoint 32-row interval.
            operand_pool = RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((8 * CHUNK_SIZE, SUPPORTED_HEAD_DIM)),
                depth=1,
            ).next_write()
            self.causal_l1 = tuple(
                tile_slice(operand_pool, (CHUNK_SIZE, width), (index, 0))
                for index, width in enumerate((128, 128, 128, 128, 128, 128, 64, 64))
            )
            if fuse_base:
                self.stacked_matrix = tile_slice(
                    operand_pool, (2 * CHUNK_SIZE, CHUNK_SIZE), (3, 0)
                )
        self.batch_rows = batch_rows
        self.cache_causal_inputs = False
        self.correct_wy_diagonal = batch_rows
        batch_height = CHUNK_SIZE // 2
        fp32_capacity = 15 + (3 * batch_height if batch_rows else 0)
        if self.correct_wy_diagonal:
            fp32_capacity += batch_height if resident_dr is None else 0
            fp32_capacity += batch_height if resident_dw is None else 0
        bf16_capacity = 8 + (2 * batch_height if batch_rows else 0)
        fp32_storage = _stage_one_scratch(
            "fp32", dtypes.float32, (fp32_capacity, SUPPORTED_HEAD_DIM)
        )
        bf16_storage = _stage_one_scratch(
            "bf16", dtypes.bfloat16, (bf16_capacity, SUPPORTED_HEAD_DIM)
        )
        fp32_row_pool = fp32_storage[:15, None]
        bf16_row_pool = bf16_storage[:8, None]
        offset = 15
        if self.correct_wy_diagonal:
            residual_v = resident_dr
            if resident_dr is None:
                residual_v = fp32_storage[offset : offset + batch_height, None]
                offset += batch_height
            residual_w = resident_dw
            if resident_dw is None:
                residual_w = fp32_storage[offset : offset + batch_height, None]
                offset += batch_height
            self.wy_diagonal_residual = (residual_v, residual_w)
        if batch_rows:
            self.batch_fp32 = tuple(
                fp32_storage[
                    offset + i * batch_height : offset + (i + 1) * batch_height, None
                ]
                for i in range(3)
            )
            self.batch_bf16 = tuple(
                bf16_storage[8 + i * batch_height : 8 + (i + 1) * batch_height, None]
                for i in range(2)
            )
            self.batch_scalar = _stage_one_scratch(
                "batch_scalar", dtypes.float32, (1, batch_height)
            ).view(batch_height)
        self.bf16_rows = tuple(
            tile_slice(bf16_row_pool, (1, SUPPORTED_HEAD_DIM), (index, 0))
            for index in range(8)
        )
        self.fp32_rows = tuple(
            tile_slice(fp32_row_pool, (1, SUPPORTED_HEAD_DIM), (index, 0))
            for index in range(15)
        )
        scalar_storage = _stage_one_scratch(
            "scalars", dtypes.float32, (1, 5 * CHUNK_SIZE + 16 + 592)
        )
        beta_storage = scalar_storage[None, : 5 * CHUNK_SIZE + 16]
        self.beta_values = _FixedScratchSlot(
            beta_storage[None, :CHUNK_SIZE].view(CHUNK_SIZE)
        )
        self.beta_tile = beta_storage[None, 2 * CHUNK_SIZE : 3 * CHUNK_SIZE].view(
            CHUNK_SIZE
        )
        self.beta_gradient = _FixedScratchSlot(
            beta_storage[None, CHUNK_SIZE : 2 * CHUNK_SIZE].view(CHUNK_SIZE)
        )
        self.fp32_scalars = tuple(
            _FixedScratchSlot(
                beta_storage[
                    None, 3 * CHUNK_SIZE + 8 * i : 3 * CHUNK_SIZE + 8 * i + 1
                ].view(1)
            )
            for i in range(2)
        )
        self.beta_gradient_sum = beta_storage[
            None, 3 * CHUNK_SIZE + 16 : 4 * CHUNK_SIZE + 16
        ].view(CHUNK_SIZE)
        self.partial_beta = beta_storage[
            None, 4 * CHUNK_SIZE + 16 : 5 * CHUNK_SIZE + 16
        ]
        self.bf16_scalars = _FixedScratchSlot(
            _stage_one_scratch("low_scalars", dtypes.bfloat16, (1, CHUNK_SIZE)).view(
                CHUNK_SIZE
            )
        )
        # FP32 gate processing needs three 64x64 planes (48 KiB), including
        # the direct Cube result path. Causal batches can require more space.
        # Feature stride stays 128.
        # Each use completes its logical batch before the next slot selection.
        # One slot keeps Channel cursor analysis small; automatic buffer sync
        # orders DMA/VF reuse. Runtime overlap may differ from depth=2.
        self.value_gradient = _FixedScratchSlot(
            _stage_one_scratch(
                "value_gradient",
                dtypes.float32,
                (max(3 * CHUNK_SIZE // 2, 12 * CAUSAL_BATCH_ROWS), VALUE_DIM),
            )
        )
        output_rows = 2 * CHUNK_SIZE if direct_results or fuse_base else CHUNK_SIZE
        self.value_output = _FixedScratchSlot(
            _stage_one_scratch(
                "value_output", dtypes.bfloat16, (2 * CHUNK_SIZE, VALUE_DIM)
            )[:output_rows, None]
        )
        # Scalar accumulators are disjoint views of one fixed slot. This keeps
        # their full byte capacity while sharing one Vector Buffer ID.
        accumulator_fields = (
            ("terminal_sum", LANES),
            ("gate_running", D_HALF),
            ("gate_bias_sum", D_HALF),
            ("gate_alog_sum", GATE_SCALAR_TRANSFER_WIDTH),
            ("gate_task_bias_sum", SUPPORTED_HEAD_DIM),
            ("gate_task_alog_sum", GATE_SCALAR_TRANSFER_WIDTH),
            ("head_query_sum", SUPPORTED_HEAD_DIM),
            ("head_key_sum", SUPPORTED_HEAD_DIM),
        )
        accumulator_pool = scalar_storage[None, 5 * CHUNK_SIZE + 16 :]
        offset = 0
        for name, width in accumulator_fields:
            setattr(self, name, accumulator_pool[None, offset : offset + width])
            offset += width

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
        mem_copy(
            tile_slice(gm_destination, (1,), (destination_row,)),
            tile_slice(channel.next_read(), (1,), (0,)),
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
            tile_slice(gm_destination, (1,), (destination_row,)),
            tile_slice(channel.next_read(), (1,), (0,)),
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
        if has_bias:
            for half in range(get_subblock_id(), 2, get_subblock_dim()):
                mem_copy(
                    tile_slice(self.fp32_rows[0], (1, D_HALF), (0, 0)),
                    tile_slice(self.gate_task_bias_sum, (1, D_HALF), (0, half)),
                )
                mem_copy(
                    tile_slice(gm_public_dBias, (1, D_HALF), (value_head, half)),
                    tile_slice(self.fp32_rows[0], (1, D_HALF), (0, 0)),
                    options=copy_options(atomic_add=True),
                )
        if const_expr(store_alog):
            mem_copy(
                tile_slice(self.fp32_rows[0], (1, GATE_SCALAR_TRANSFER_WIDTH), (0, 0)),
                self.gate_task_alog_sum,
            )
            mem_copy(
                tile_slice(
                    gm_public_dA, (1, GATE_SCALAR_TRANSFER_WIDTH), (value_head, 0)
                ),
                tile_slice(self.fp32_rows[0], (1, GATE_SCALAR_TRANSFER_WIDTH), (0, 0)),
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


class StageOneWyOperandVector:
    """Form beta-weighted BF16 operands for the four WY Cube products."""

    def __init__(self, shared: StageOneVector):
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


class StageOneWyResultVector:
    """Reduce the two beta-gradient branches after the WY Cube products."""

    def __init__(self, shared: StageOneVector):
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


class BaseGradientVector:
    """Own UB rows for the FP32 value, query, key, and gate gradients."""

    def __init__(self, shared: StageOneVector):
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

    def __init__(self, shared: StageOneVector):
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

    def __init__(self, shared: StageOneVector):
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


class StageOneInverseMaskVector:
    """Apply the minus sign and strict lower triangle to the Cube product."""

    def __init__(self, shared: StageOneVector | None = None):
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


class StageOneCausalOperandVector:
    """Build row-scaled BF16 operands for the causal Cube products."""

    def __init__(self, fp32: bool, shared: StageOneVector | None = None):
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


class StageOneCausalResultVector:
    """Accumulate stable block Q/K/G/beta derivatives from Cube products."""

    def __init__(self, fp32: bool, shared: StageOneVector | None = None):
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


class StageOneProcess:
    """Run the dependent WY inverse and local-gradient stages per chunk."""

    def __init__(
        self,
        fuse_base=False,
        resident_dr=None,
        resident_dw=None,
        adjacent_decay=False,
        cube_causal=False,
    ):
        self.vector = StageOneVector(
            batch_rows=True,
            fuse_base=fuse_base,
            resident_dr=resident_dr,
            resident_dw=resident_dw,
            direct_results=cube_causal,
        )
        self.cube_causal = cube_causal
        self.matmul = StageOneMatmul(
            self.vector.causal_results,
            self.vector.causal_l1,
            self.vector.stacked_matrix,
        )
        self.vector.cache_causal_inputs = cube_causal
        if cube_causal:
            self.inverse_mask = StageOneInverseMaskVector(self.vector)
            self.causal_operands = StageOneCausalOperandVector(False, self.vector)
            self.causal_result = StageOneCausalResultVector(False, self.vector)
        self.wy_operands = StageOneWyOperandVector(self.vector)
        self.wy_result = StageOneWyResultVector(self.vector)
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
):
    process = StageOneProcess()
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
            if not padded_layout:
                inverse_batch = _tnd_chunk_index(compact_token_start, seq_idx, chunk)
                inverse_row_start = 0
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

    def __init__(self):
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
    def _gate_dot_row(
        self, staged, row, alog, bound, fp32_gate: Constexpr[bool]
    ):
        # Both halves on every AIV: the coefficient and solve stages read the
        # full gate_dot rows, so each AIV publishes a complete local copy.
        with vf(mode="simd"):
            full = create_mask(pattern="all")
            for half in range(2):
                offset = half * D_HALF
                if const_expr(fp32_gate):
                    raw = vload(staged, offset)
                else:
                    raw = vcast(
                        vload_unpack(
                            staged, offset, unpack_mode=UnpackMode.B16_TO_B32
                        ),
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
                vstore(
                    self.gate_dot, row * SUPPORTED_HEAD_DIM + offset, running, full
                )
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
            mem_copy(
                tile_slice(state, (1, SUPPORTED_HEAD_DIM), (0, 0)),
                tile_slice(initial_state_source, (1, SUPPORTED_HEAD_DIM), (value, 0)),
            )
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
            tile_slice(gm_public_dA, (1, GATE_SCALAR_TRANSFER_WIDTH), (value_head, 0)),
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
            increment = vsub(
                vmul(seed, odot, mask=scalar), loss_error, mask=scalar
            )
            updated = vadd(loss, increment, mask=scalar)
            loss_error = vsub(
                vsub(updated, loss, mask=scalar), increment, mask=scalar
            )
            loss = updated
            term = vadd(
                vmul(final_coefficient_dot, z, mask=scalar),
                vmul(final_coefficient, zdot, mask=scalar),
                mask=scalar,
            )
            increment = vsub(term, loss_error, mask=scalar)
            updated = vadd(loss, increment, mask=scalar)
            loss_error = vsub(
                vsub(updated, loss, mask=scalar), increment, mask=scalar
            )
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
):
    # Stage 1 suppresses both rounded-cache dA and valid-row dbeta stores
    # for these tasks. The fused path keeps the recompute track owner.
    # The initialization handoff publishes dA zeroing before either phase.
    sensitivity = SingleChunkGateSensitivity()
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


# ---------------------------------------------------------------------------
# Shared gradient initialization and forward-cache recomputation
# ---------------------------------------------------------------------------


@jit
def _initialize_backward_gradients(dA, dBias, fused_gate: dtypes.int64):
    # This phase only clears parameter gradients. Constructing StageOneVector
    # here carries all of its channels through the following recurrent scan.
    # A single plain buffer also gives each head exactly one AIV writer.
    if fused_gate:
        zeros = Buffer(MemLoc.UB, dtypes.float32, make_layout((1, SUPPORTED_HEAD_DIM)))
        with vf(mode="simd"):
            mask, _ = update_mask(D_HALF, elem_bits=32)
            zero = vdups(0.0, dtypes.float32, mask=mask)
            vstore(zeros, 0, zero, mask)
            vstore(zeros, D_HALF, zero, mask)
        first = get_block_idx() * get_subblock_dim() + get_subblock_id()
        step = get_block_num() * get_subblock_dim()
        for value_head in range(first, dA.shape[0], step):
            mem_copy(tile_slice(dBias, (1, SUPPORTED_HEAD_DIM), (value_head, 0)), zeros)
            mem_copy(
                tile_slice(dA, (1, GATE_SCALAR_TRANSFER_WIDTH), (value_head, 0)),
                tile_slice(zeros, (1, GATE_SCALAR_TRANSFER_WIDTH), (0, 0)),
            )
    # Publish parameter-gradient zeroing once, before any atomic adds.
    _sync_cube_vector_handoff(0, grid=True)


def _recompute_implementation(staged=False):
    try:
        from .flash_kda_grad_recompute import recompute_tracks, recompute_staged_tracks
    except ImportError:
        from flash_kda_grad_recompute import recompute_tracks, recompute_staged_tracks
    return recompute_staged_tracks if staged else recompute_tracks


@jit
def _recompute_tracks(
    q,
    k,
    v,
    g,
    beta,
    inverse,
    states,
    a_log,
    bias,
    metadata,
    w,
    u,
    v_new,
    qg,
    kg,
    gamma,
    initial_state,
    padded: dtypes.int64,
    fused_gate: dtypes.int64,
    has_bias: dtypes.int64,
    softplus: dtypes.int64,
    bound: dtypes.float32,
    fused_beta: dtypes.int64,
    has_initial_state: dtypes.int64,
    gm_owner_schedule,
    staged: Constexpr[bool] = False,
):
    _recompute_implementation(staged)(
        q,
        k,
        v,
        g,
        beta,
        inverse,
        states,
        a_log,
        bias,
        metadata,
        w,
        u,
        v_new,
        qg,
        kg,
        gamma,
        initial_state,
        padded,
        fused_gate,
        has_bias,
        softplus,
        bound,
        fused_beta,
        has_initial_state,
        gm_owner_schedule,
    )


def _backward_template(staged):
    """Resolve the constexpr strategy during tracing, after shared definitions load.

    Strategy modules import this module's common arithmetic. Deferring the
    reverse import keeps both package and flat Native imports independent of
    import order. No import or Python dispatch is emitted into device code.
    """
    if staged:
        if __package__:
            from .flash_kda_grad_staged import run_staged_flash_kda_grad
        else:
            from flash_kda_grad_staged import run_staged_flash_kda_grad
        return run_staged_flash_kda_grad
    if __package__:
        from .flash_kda_grad_persistent import run_persistent_flash_kda_grad
    else:
        from flash_kda_grad_persistent import run_persistent_flash_kda_grad
    return run_persistent_flash_kda_grad


# Host preparation, kernel launches and public API


@dataclass
class _WorkspaceEntry:
    device: object
    tensors: dict = field(default_factory=dict)
    event: object = None


class _WorkspaceLease:
    def __init__(self, pool, entry, stream):
        self.pool, self.entry, self.stream = pool, entry, stream
        self.position = 0

    def empty(self, shape, dtype):
        shape = tuple(shape)
        key = (self.position, shape, dtype)
        self.position += 1
        tensor = self.entry.tensors.get(key)
        if tensor is None:
            tensor = torch.empty(shape, dtype=dtype, device=self.entry.device)
            size = tensor.numel() * tensor.element_size()
            occupied = sum(
                value.numel() * value.element_size()
                for entry in self.pool.entries.values()
                if entry.device == self.entry.device
                for value in entry.tensors.values()
            )
            if size <= self.pool.item_limit and occupied + size <= self.pool.byte_limit:
                self.entry.tensors[key] = tensor
        # Raw DSL launches do not tell the allocator about their pointer uses.
        tensor.record_stream(self.stream)
        return tensor


class WorkspacePool:
    """Small per-device scratch pool with stream isolation and bounded retention.

    The lock covers preparation and launch, so another host thread cannot take
    an entry before its completion event is recorded. Each caller stream owns
    its scratch, so independent streams never wait for another compute call.
    """

    def __init__(self, byte_limit=32 * 1024 * 1024, item_limit=1024 * 1024, limit=2):
        self.entries = OrderedDict()
        self.lock = threading.RLock()
        self.byte_limit, self.item_limit, self.limit = byte_limit, item_limit, limit
        self.pending_events = []

    @contextmanager
    def acquire(self, ref, key):
        with torch.npu.device(ref.device), self.lock:
            stream = torch.npu.current_stream(torch.npu.current_device())
            key = (str(ref.device), int(stream.npu_stream), *key)
            entry = self.entries.get(key)
            if entry is None:
                # Evicted tensors have recorded their consumer stream; freeing
                # the host owners cannot recycle storage before that use ends.
                matching = [
                    k for k, value in self.entries.items() if value.device == ref.device
                ]
                if len(matching) >= self.limit:
                    del self.entries[matching[0]]
                entry = _WorkspaceEntry(ref.device)
                self.entries[key] = entry
            self.entries.move_to_end(key)
            if entry.event is not None:
                stream.wait_event(entry.event)
            finished = torch.npu.Event()
            try:
                yield _WorkspaceLease(self, entry, stream)
                finished.record(stream)
                entry.event = finished
                self.pending_events[:] = [
                    (device, event)
                    for device, event in self.pending_events
                    if not event.query()
                ]
                self.pending_events.append((ref.device, finished))
            except BaseException:
                # Retire possibly submitted raw-pointer work before reusing or
                # releasing storage after a failed launch.
                stream.synchronize()
                self.entries.pop(key, None)
                raise

    def clear(self):
        with self.lock:
            # Include evicted entries: their code may still be executing.
            for device, event in self.pending_events:
                with torch.npu.device(device):
                    event.synchronize()
            self.pending_events.clear()
            self.entries.clear()


def _device_block_num(ref: Optional[torch.Tensor] = None) -> int:
    """Use the resident Cube core count, capped for whole-grid barriers."""
    if ref is not None and ref.device.type not in {"npu", "privateuseone"}:
        return MAX_AIC_CORES
    npu = getattr(torch, "npu", None)
    if npu is None:
        return MAX_AIC_CORES
    device_index = ref.device.index if ref is not None else None
    if device_index is None:
        try:
            device_index = npu.current_device()
        except RuntimeError:
            return MAX_AIC_CORES
    platform_info = get_platform_info(stream=npu.current_stream(device_index))
    try:
        cube_core_num = int(platform_info.cube_core_num)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"NPU {device_index} has no valid cube_core_num") from exc
    if cube_core_num <= 0:
        raise RuntimeError(f"NPU {device_index} reports cube_core_num={cube_core_num}")
    return min(cube_core_num, MAX_AIC_CORES)


# ---------------------------------------------------------------------------
# Reverse state scan and output gradients
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageTwoResult:
    d_v_new: torch.Tensor
    d_Aqk: torch.Tensor
    d_qg: torch.Tensor
    d_kg: torch.Tensor
    d_w: torch.Tensor
    d_gate_last: torch.Tensor
    dh0: torch.Tensor


def _backward_tensor_spec(tensor, dtype):
    return cannbotdsl.TensorSpec(
        tuple(tensor.shape), dtype, stride=tuple(tensor.stride())
    )


def _prepare_stage_two(
    *,
    aqk: torch.Tensor,
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    v_new: torch.Tensor,
    g_cumsum: torch.Tensor,
    chunk_states: torch.Tensor,
    output_gradient: torch.Tensor,
    final_state_gradient: torch.Tensor,
    layout: str,
    metadata: torch.Tensor,
    scale: float,
    core_num: int = 1,
    _workspace=None,
    _needs_masked_chunks: bool = True,
    _staged: bool = False,
):
    """Prepare inputs and per-core or global-history backward workspaces."""
    if layout not in ("TND", "BNSD", "BSND"):
        raise ValueError("layout must be TND, BNSD, or BSND")
    is_tnd = layout == "TND"
    if is_tnd:
        if qg.shape[2] != SUPPORTED_HEAD_DIM:
            raise ValueError("TND qg width is invalid")
        heads = qg.shape[1]
    else:
        heads = qg.shape[1] if layout == "BNSD" else qg.shape[2]
        if qg.shape[-1] != SUPPORTED_HEAD_DIM or qg.ndim != 4:
            raise ValueError("qg physical cache shape is invalid")
    if (
        metadata.ndim != 1
        or metadata.dtype != torch.int32
        or metadata.device != qg.device
        or not metadata.is_contiguous()
        or (metadata.numel() - HEADER_WORDS) % (TASK_WORDS * heads)
    ):
        raise ValueError("backward metadata ABI is invalid")
    batch = (metadata.numel() - HEADER_WORDS) // (TASK_WORDS * heads)
    if batch < 1 or metadata.numel() != metadata_capacity(batch, heads):
        raise ValueError("backward metadata task count is invalid")
    if not is_tnd and qg.shape[0] != batch:
        raise ValueError("physical qg batch does not match metadata")
    physical_matrix_shape = (*qg.shape[:-1], CHUNK_SIZE)
    if not is_tnd and tuple(aqk.shape) != physical_matrix_shape:
        raise ValueError("padded Aqk view must match the token rows")
    if not is_tnd and not (
        chunk_states.ndim == 5
        and chunk_states.shape[0] == batch
        and chunk_states.shape[1] == heads
        and chunk_states.shape[3:] == (SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)
    ):
        raise ValueError("padded chunk_states must match the batch and heads")
    if any(t.shape != qg.shape for t in (kg, w, v_new, g_cumsum)):
        raise ValueError("token caches do not match qg")
    device = qg.device

    def logical_tokens(tensor):
        return tensor.unsqueeze(0).transpose(1, 2)

    def input_view(tensor):
        return (
            logical_tokens(tensor)
            if layout == "TND"
            else tensor
            if layout == "BNSD"
            else tensor.transpose(1, 2)
        )

    qg_view = input_view(qg)
    kg_view = input_view(kg)
    w_view = input_view(w)
    v_new_view = input_view(v_new)
    gate_view = input_view(g_cumsum)
    matrix_view = aqk if is_tnd else input_view(aqk)
    state_view = chunk_states.unsqueeze(0).transpose(1, 2) if is_tnd else chunk_states
    output_view = input_view(output_gradient)
    terminal_gradient = final_state_gradient

    # Staged owners exchange all chunks through GM. One additional slot per
    # task keeps empty sequences and TASK_FLAT_CHUNK_START + task collision-free.
    gradient_slots = _metadata.backward_gradient_slots(
        metadata, heads, core_num, _staged
    )
    gradient_shape = (gradient_slots * CHUNK_SIZE, 1, SUPPORTED_HEAD_DIM)
    d_v_new = torch.empty(gradient_shape, dtype=torch.float32, device=device)
    d_qg = torch.empty_like(d_v_new)
    d_kg = torch.empty_like(d_v_new)
    d_w = torch.empty_like(d_v_new)
    d_Aqk = torch.empty(
        (gradient_slots, 1, CHUNK_SIZE, CHUNK_SIZE), dtype=torch.float32, device=device
    )
    d_gate_last = torch.empty(
        (gradient_slots, 1, SUPPORTED_HEAD_DIM), dtype=torch.float32, device=device
    )
    dh0 = torch.empty(
        (batch, heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )

    # Persistent full chunks keep state in UB. Masked chunks use two FP32
    # slots per core; staged execution keeps both slots for every chunk.
    masked_cores = core_num if _needs_masked_chunks or _staged else 0
    state_slots = 2 * gradient_slots if _staged else 2 * masked_cores
    state_gradients = torch.empty(
        (state_slots * SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )

    def scratch(shape, dtype):
        if _workspace is not None:
            return _workspace.empty(shape, dtype)
        return torch.empty(shape, dtype=dtype, device=device)

    masked_aqk = scratch((masked_cores * CHUNK_SIZE, CHUNK_SIZE), torch.bfloat16)
    masked_qg, masked_v_new, masked_d_output = (
        scratch((masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.bfloat16)
        for _ in range(3)
    )
    d_v_new_output, d_kg_scratch = (
        scratch((masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.float32)
        for _ in range(2)
    )
    d_v_new_scratch = scratch(
        (masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.float32
    )
    d_v_new_low = scratch(
        (masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.bfloat16
    )
    d_qg_scratch = scratch(
        (masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.float32
    )
    d_w_scratch = scratch(
        (masked_cores * CHUNK_SIZE, SUPPORTED_HEAD_DIM), torch.float32
    )
    state_high = scratch(
        (masked_cores * SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), torch.bfloat16
    )
    state_residual = scratch(
        (masked_cores * SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), torch.bfloat16
    )
    saved_state_high = scratch(
        (masked_cores * SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), torch.bfloat16
    )

    arguments = (
        matrix_view,
        qg_view,
        kg_view,
        w_view,
        v_new_view,
        gate_view,
        state_view,
        output_view,
        terminal_gradient,
        metadata,
        logical_tokens(d_v_new),
        d_Aqk.view(-1, CHUNK_SIZE),
        logical_tokens(d_qg),
        logical_tokens(d_kg),
        logical_tokens(d_w),
        logical_tokens(d_gate_last),
        dh0,
        state_gradients,
        masked_aqk,
        masked_qg,
        masked_v_new,
        masked_d_output,
        d_v_new_output,
        d_kg_scratch,
        d_v_new_scratch,
        d_v_new_low,
        d_qg_scratch,
        d_w_scratch,
        state_high,
        state_residual,
        saved_state_high,
    )
    dtype_map = {
        torch.bfloat16: dtypes.bfloat16,
        torch.float32: dtypes.float32,
        torch.int32: dtypes.int32,
    }
    specs = tuple(
        _backward_tensor_spec(tensor, dtype_map[tensor.dtype]) for tensor in arguments
    )
    result = StageTwoResult(d_v_new, d_Aqk, d_qg, d_kg, d_w, d_gate_last, dh0)
    return result, arguments, specs


# ---------------------------------------------------------------------------
# WY and triangular inverse gradients
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Base query, key, and gate gradients
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Value and beta gradients
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full NPU backward orchestration
# ---------------------------------------------------------------------------


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="DSL_flash_kda_grad",
        op_type="FlashKDAGrad",
        inputs=(
            "stage_two_gm_Aqk",
            "stage_two_gm_chunk_states",
            "stage_two_gm_output_gradient",
            "stage_two_gm_final_state_gradient",
            "stage_two_gm_task_metadata",
            "stage_one_gm_effective_query",
            "stage_one_gm_effective_key",
            "stage_one_gm_value",
        ),
        outputs=(
            "stage_one_gm_public_d_query",
            "stage_one_gm_public_d_key",
            "stage_one_gm_public_d_value",
            "stage_two_gm_dh0",
            "stage_one_gm_public_d_gate",
            "stage_one_gm_public_d_beta",
            "stage_one_gm_public_dA",
            "stage_one_gm_public_dBias",
        ),
    )
)
class flash_kda_grad_kernel:
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
        staged: Constexpr[bool],
        cube_causal: Constexpr[bool],
        stage_two_scale: dtypes.float32,
        stage_one_scale: dtypes.float32,
        stage_one_inverse_factor: dtypes.float32,
        stage_one_lower_bound: dtypes.float32,
    ):
        with _SharedCubeScope():
            _backward_template(staged)(
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
                False,
            )


class FlashKDAGrad:
    @host
    def run(
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
        stage_two_scale: dtypes.float32,
        stage_one_scale: dtypes.float32,
        stage_one_fused_beta: dtypes.int64,
        stage_one_inverse_factor: dtypes.float32,
        stage_one_heads_per_key_head: dtypes.int64,
        stage_one_fused_gate: dtypes.int64,
        stage_one_has_bias: dtypes.int64,
        stage_one_use_l2norm: dtypes.int64,
        stage_one_softplus: dtypes.int64,
        stage_one_lower_bound: dtypes.float32,
        stage_one_beta_fp32: Constexpr[bool],
        padded_layout: dtypes.int64,
        use_full_chunks: Constexpr[bool],
        needs_masked_chunks: Constexpr[bool],
        staged: Constexpr[bool],
        cube_causal: Constexpr[bool],
        core_num: int,
    ):
        flash_kda_grad_kernel()[core_num](
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
            staged,
            cube_causal,
            stage_two_scale,
            stage_one_scale,
            stage_one_inverse_factor,
            stage_one_lower_bound,
        )


def _prepare_stage_one(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    values: torch.Tensor,
    raw_gate: torch.Tensor,
    activated_beta: torch.Tensor,
    saved_inverse: torch.Tensor,
    saved_qg: torch.Tensor,
    saved_kg: torch.Tensor,
    saved_gamma: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    query_rstd: torch.Tensor | None,
    key_rstd: torch.Tensor | None,
    stage_two: StageTwoResult,
    layout: str,
    metadata: torch.Tensor,
    scale: float,
    fused_beta: bool,
    fused_gate: bool,
    use_l2norm: bool,
    inverse_factor: float,
    lower_bound: float | None,
    beta_dtype: torch.dtype = torch.bfloat16,
    _workspace=None,
):
    """Prepare local chunk scratch and public gradient outputs without launching."""
    device = values.device
    value_heads = (
        values.shape[1]
        if layout == "TND"
        else (values.shape[1] if layout == "BNSD" else values.shape[2])
    )
    qk_heads = query.shape[1] if query.ndim == 3 or layout == "BNSD" else query.shape[2]
    if qk_heads < 1 or value_heads % qk_heads:
        raise ValueError("effective Q/K heads do not match value heads")
    heads_per_key_head = value_heads // qk_heads
    has_tokens = saved_qg.numel() > 0
    # GQA retains only the current chunk's Q/K contributions for this core.
    # All other intermediates are also reused when this owner advances.
    scratch_chunks = _device_block_num(values) * heads_per_key_head if has_tokens else 0
    rows = scratch_chunks * CHUNK_SIZE

    def as_bnsd_view(tensor, scalar=False):
        if scalar:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0).transpose(1, 2)
        if layout == "BSND":
            return tensor.transpose(1, 2)
        return tensor

    d_value = torch.empty_like(values)
    if layout == "TND":
        beta_rows = values.shape[0]
        beta_aligned_rows = beta_rows + beta_rows % 2
        d_beta_storage = torch.empty(
            (value_heads, beta_aligned_rows), dtype=beta_dtype, device=device
        )
        d_beta = d_beta_storage[:, :beta_rows].transpose(0, 1)
    else:
        beta_rows = values.shape[2] if layout == "BNSD" else values.shape[1]
        beta_aligned_rows = beta_rows + beta_rows % 2
        d_beta_storage = torch.empty(
            (values.shape[0], value_heads, beta_aligned_rows),
            dtype=beta_dtype,
            device=device,
        )
        beta_head_major = d_beta_storage[:, :, :beta_rows]
        d_beta = (
            beta_head_major if layout == "BNSD" else beta_head_major.transpose(1, 2)
        )
    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    d_gate = torch.empty_like(raw_gate)
    dA_padded = torch.empty(
        (value_heads, GATE_SCALAR_TRANSFER_WIDTH), dtype=torch.float32, device=device
    )
    dBias = torch.empty(
        (value_heads, SUPPORTED_HEAD_DIM), dtype=torch.float32, device=device
    )
    has_bias = dt_bias is not None
    if A_log is None:
        A_log = torch.empty((value_heads,), dtype=torch.float32, device=device)
    if dt_bias is None:
        dt_bias = torch.empty(
            (value_heads, SUPPORTED_HEAD_DIM), dtype=torch.float32, device=device
        )
    if query_rstd is None:
        query_rstd = torch.empty(query.shape[:-1], dtype=torch.float32, device=device)
    if key_rstd is None:
        key_rstd = torch.empty(key.shape[:-1], dtype=torch.float32, device=device)

    def tensor(shape, dtype):
        if _workspace is not None:
            return _workspace.empty(shape, dtype)
        return torch.empty(shape, dtype=dtype, device=device)

    def f32_rows():
        return tensor((rows, SUPPORTED_HEAD_DIM), torch.float32)

    def b16_rows():
        return tensor((rows, SUPPORTED_HEAD_DIM), torch.bfloat16)

    def f32_matrix():
        return tensor((rows, CHUNK_SIZE), torch.float32)

    def b16_matrix():
        return tensor((rows, CHUNK_SIZE), torch.bfloat16)

    def f32_scalar():
        return tensor((rows,), torch.float32)

    stage_one_scratch = {
        "inverse": b16_matrix(),
        "values": b16_rows(),
        "keys": b16_rows(),
        "gamma": f32_rows(),
        "beta": f32_scalar(),
        "d_v_new_low": b16_rows(),
        "d_w_low": b16_rows(),
        "weighted_v": b16_rows(),
        "weighted_k": b16_rows(),
        "d_beta_v": f32_rows(),
        "d_beta_k": f32_rows(),
        "d_beta_wy": f32_scalar(),
        "d_lower_unmasked": f32_matrix(),
        "qg": b16_rows(),
        "kg": b16_rows(),
        "d_qg": f32_rows(),
        "d_kg": f32_rows(),
        "d_gate_last": tensor((scratch_chunks, SUPPORTED_HEAD_DIM), torch.float32),
        "d_query_base": f32_rows(),
        "d_key_base": f32_rows(),
        "d_gate_base": f32_rows(),
        "query": b16_rows(),
        "d_Aqk": f32_matrix(),
        "d_beta_matrix": f32_rows(),
        "d_query_pair": f32_rows(),
        "d_key_pair": f32_rows(),
        "d_gate_pair": f32_rows(),
        "d_beta_pair": f32_scalar(),
        "d_value": b16_rows(),
        "d_beta": tensor((rows,), beta_dtype),
    }
    source_arguments = (
        as_bnsd_view(query),
        as_bnsd_view(key),
        as_bnsd_view(values),
        as_bnsd_view(activated_beta, scalar=True),
        (saved_inverse if layout == "TND" else as_bnsd_view(saved_inverse)),
        as_bnsd_view(saved_qg),
        as_bnsd_view(saved_kg),
        as_bnsd_view(saved_gamma),
        as_bnsd_view(stage_two.d_v_new),
        as_bnsd_view(stage_two.d_w),
        as_bnsd_view(stage_two.d_qg),
        as_bnsd_view(stage_two.d_kg),
        stage_two.d_Aqk,
        stage_two.d_gate_last.unsqueeze(0).transpose(1, 2),
        metadata,
        as_bnsd_view(d_value),
        as_bnsd_view(d_beta, scalar=True),
        as_bnsd_view(raw_gate),
        A_log,
        dt_bias,
        as_bnsd_view(query_rstd, scalar=True),
        as_bnsd_view(key_rstd, scalar=True),
        as_bnsd_view(d_query),
        as_bnsd_view(d_key),
        as_bnsd_view(d_gate),
        dA_padded,
        dBias,
    )
    tensor_arguments = source_arguments + tuple(stage_one_scratch.values())
    dtype_map = {
        torch.bfloat16: dtypes.bfloat16,
        torch.float32: dtypes.float32,
        torch.int32: dtypes.int32,
    }
    specs = tuple(
        _backward_tensor_spec(item, dtype_map[item.dtype]) for item in tensor_arguments
    )
    config = (
        float(scale),
        fused_beta,
        float(inverse_factor),
        heads_per_key_head,
        fused_gate,
        has_bias,
        use_l2norm,
        lower_bound is None,
        0.0 if lower_bound is None else float(lower_bound),
        activated_beta.dtype == torch.float32,
        layout != "TND",
    )
    result = (
        d_query,
        d_key,
        d_value,
        d_gate,
        d_beta,
        dA_padded[:, 0] if fused_gate else None,
        dBias if fused_gate and has_bias else None,
    )
    return result, tensor_arguments, specs, config


@dataclass
class _BackwardProgramEntry:
    launcher: object
    program: object
    pending_events: list


@dataclass(eq=False)
class _BackwardWorkspaceLease:
    event: object
    device: object
    arguments: Optional[tuple]
    stream: object = None
    event_recorded: bool = True
    error: Optional[Exception] = None


_BACKWARD_PROGRAMS = OrderedDict()
_BACKWARD_SCRATCH = WorkspacePool()
_BACKWARD_PROGRAM_LOCK = threading.RLock()
_BACKWARD_PROGRAM_LIMIT = 16
_BACKWARD_WORKSPACE_LEASES = OrderedDict()
_BACKWARD_RETIREMENT_QUEUE = None
_BACKWARD_RETIREMENT_THREAD = None


def _release_backward_workspace_lease(lease):
    # Queue entries keep this mutable owner, never another copy of its Tensor
    # tuple. Detach under the lock, then destroy on the owning device. Tensor
    # destruction can call the native allocator and must not hold the worker's
    # cache lock or fall back to a new thread's uninitialized device 0.
    with torch.npu.device(lease.device):
        with _BACKWARD_PROGRAM_LOCK:
            arguments = lease.arguments
            lease.arguments = None
            lease.error = None
            _BACKWARD_WORKSPACE_LEASES.pop(id(lease), None)
        del arguments


def _synchronize_backward_workspace_lease(lease):
    with torch.npu.device(lease.device):
        if lease.event_recorded:
            lease.event.synchronize()
        else:
            # A failed launch/event record has no reliable completion marker.
            # Explicit cache cleanup can retry its original stream boundary.
            lease.stream.synchronize()


def _retire_backward_workspace_lease(lease):
    try:
        _synchronize_backward_workspace_lease(lease)
    except Exception as error:
        # A failed wait does not prove the allocation is idle. Keep every
        # owner until explicit cleanup successfully synchronizes the call.
        with _BACKWARD_PROGRAM_LOCK:
            if lease.arguments is not None:
                lease.error = error
    else:
        _release_backward_workspace_lease(lease)


def _backward_workspace_retirement_worker(retirement_queue):
    while True:
        lease = retirement_queue.get()
        try:
            if lease is None:
                return
            # Device/stream state is thread local. Keep the worker bound after
            # the temporary event guard exits, including final lease/event
            # destruction below. A rank's device need not be logical device 0.
            torch.npu.set_device(lease.device)
            torch.npu.current_stream(lease.device)
            _retire_backward_workspace_lease(lease)
        except Exception as error:
            # Initialization failures must retain owners and reach the caller,
            # rather than silently terminating the retirement worker.
            with _BACKWARD_PROGRAM_LOCK:
                if lease.arguments is not None:
                    lease.error = error
        finally:
            retirement_queue.task_done()
            del lease


def _queue_backward_workspace_retirement(lease):
    global _BACKWARD_RETIREMENT_QUEUE, _BACKWARD_RETIREMENT_THREAD
    # Called under the cache lock after recording the completion event. No
    # thread or queue is created by importing this module or compiling code.
    if _BACKWARD_RETIREMENT_QUEUE is None:
        _BACKWARD_RETIREMENT_QUEUE = Queue()
    if (
        _BACKWARD_RETIREMENT_THREAD is None
        or not _BACKWARD_RETIREMENT_THREAD.is_alive()
    ):
        _BACKWARD_RETIREMENT_THREAD = threading.Thread(
            target=_backward_workspace_retirement_worker,
            args=(_BACKWARD_RETIREMENT_QUEUE,),
            name="kda_bwd_workspace_retire",
            daemon=True,
        )
        _BACKWARD_RETIREMENT_THREAD.start()
    _BACKWARD_RETIREMENT_QUEUE.put(lease)


def _stop_backward_workspace_retirement():
    global _BACKWARD_RETIREMENT_QUEUE, _BACKWARD_RETIREMENT_THREAD
    with _BACKWARD_PROGRAM_LOCK:
        worker = _BACKWARD_RETIREMENT_THREAD
        retirement_queue = _BACKWARD_RETIREMENT_QUEUE
        _BACKWARD_RETIREMENT_THREAD = None
        _BACKWARD_RETIREMENT_QUEUE = None
        if worker is not None:
            retirement_queue.put(None)
    # The worker needs the same lock to retire leases; never join under it.
    if worker is not None and worker is not threading.current_thread():
        worker.join()


def _reap_backward_workspaces():
    """Release completed owners before allocating another backward workspace."""
    with _BACKWARD_PROGRAM_LOCK:
        leases = tuple(_BACKWARD_WORKSPACE_LEASES.values())
    for lease in leases:
        if lease.arguments is None:
            continue
        if lease.error is not None:
            raise RuntimeError(
                "KDA backward workspace retirement failed; "
                "clear the backward program cache after resolving the NPU error"
            ) from lease.error
        if lease.event_recorded:
            with torch.npu.device(lease.device):
                completed = lease.event.query()
            if completed:
                _release_backward_workspace_lease(lease)


# Both saved and recompute ABIs place the raw gate at this tensor position.
_BACKWARD_RAW_GATE_INDEX = 48
_BACKWARD_BETA_GRAD_INDEX = 47


def _backward_scalar_specs(config):
    """Specialize beta storage, chunk paths and scheduling at compile time."""
    return tuple(
        value
        if index in (9, 11, 12, 13, 14)
        else dtypes.float32
        if index in (0, 2, 8)
        else dtypes.int64
        for index, value in enumerate(config)
    )


def _backward_runtime_config(config):
    return tuple(
        float(value) if index in (0, 2, 8) else int(value)
        for index, value in enumerate(config)
        if index not in (9, 11, 12, 13, 14)
    )


def backward_dynamic_compile_request(
    beta_fp32=True,
    use_full_chunks=False,
    needs_masked_chunks=True,
    gate_fp32=False,
    staged=False,
    beta_grad_fp32=False,
    cube_causal=False,
):
    """Shared AOT/runtime ABI, independent of concrete shapes and gate settings.

    Host preparation normalizes all public layouts to head-major views. The
    input, token-cache, matrix-cache and state-cache storage can have different
    capacities; keep those dimensions independent. Q/K inverse norms use the
    input layout after host preparation. Public/cache strides are runtime
    values, while internal scratch is contiguous.
    """
    dim = cannbotdsl.Dim
    fake = cannbotdsl.TensorSpec
    p, length = dim("P"), dim("L", min=0)
    batch, nq, nv = dim("B"), dim("NQK"), dim("NV")
    cache_p, cache_l = dim("CACHE_P"), dim("CACHE_L", min=0)
    beta_p, beta_l = dim("BETA_P"), dim("BETA_L", min=0)
    matrix_p, matrix_l = dim("MATRIX_P", min=0), dim("MATRIX_L", min=0)
    inverse_p, inverse_l = dim("INVERSE_P", min=0), dim("INVERSE_L", min=0)
    state_p, state_c = dim("STATE_P"), dim("STATE_C", min=0)
    output_p, output_l = dim("DO_P"), dim("DO_L", min=0)
    meta = dim("META", min=HEADER_WORDS)
    cores = dim("CORES", min=1, max=MAX_AIC_CORES)
    gradient_slots = dim("STAGED_SLOTS", min=1) if staged else cores
    core_rows = gradient_slots * CHUNK_SIZE
    masked_rows = dim(
        "MASKED_CORE_ROWS",
        min=0,
        max=MAX_AIC_CORES * CHUNK_SIZE,
        multiple_of=CHUNK_SIZE,
    )
    state_gradient_rows = (
        2 * gradient_slots * SUPPORTED_HEAD_DIM
        if staged
        else masked_rows * (2 * SUPPORTED_HEAD_DIM // CHUNK_SIZE)
    )
    scratch_chunks = dim("STAGE_ONE_SCRATCH_CHUNKS", min=0)
    rows = scratch_chunks * CHUNK_SIZE
    bf16, fp32 = dtypes.bfloat16, dtypes.float32

    def sequence(
        dtype, heads=nv, physical=cache_p, size=cache_l, width=SUPPORTED_HEAD_DIM
    ):
        return fake((physical, heads, size, width), dtype)

    cache = sequence(bf16)
    cache_f32 = sequence(fp32)
    compact_f32 = sequence(fp32, heads=1, physical=1, size=core_rows)
    gate_last = sequence(fp32, heads=1, physical=1, size=gradient_slots)
    raw_value = sequence(bf16, physical=p, size=length)
    qk = sequence(bf16, heads=nq, physical=p, size=length)
    rstd = sequence(fp32, heads=nq, physical=p, size=length, width=1)
    state = fake((batch, nv, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), fp32)
    metadata = fake((meta,), dtypes.int32)
    chunk_matrix = fake((gradient_slots, 1, CHUNK_SIZE, CHUNK_SIZE), fp32)
    per_core_bf16 = fake((masked_rows, SUPPORTED_HEAD_DIM), bf16)
    per_core_f32 = fake((masked_rows, SUPPORTED_HEAD_DIM), fp32)
    per_core_state = fake(
        (masked_rows * (SUPPORTED_HEAD_DIM // CHUNK_SIZE), SUPPORTED_HEAD_DIM), bf16
    )
    specs = [
        sequence(bf16, physical=matrix_p, size=matrix_l, width=CHUNK_SIZE),
        cache,
        cache,
        cache,
        cache,
        cache_f32,
        fake((state_p, nv, state_c, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), fp32),
        sequence(bf16, physical=output_p, size=output_l),
        state,
        metadata,
        compact_f32,
        fake((core_rows, CHUNK_SIZE), fp32),
        compact_f32,
        compact_f32,
        compact_f32,
        gate_last,
        state,
        fake((state_gradient_rows, SUPPORTED_HEAD_DIM), fp32),
        fake((masked_rows, CHUNK_SIZE), bf16),
        per_core_bf16,
        per_core_bf16,
        per_core_bf16,
        per_core_f32,
        per_core_f32,
        per_core_f32,
        per_core_bf16,
        per_core_f32,
        per_core_f32,
        per_core_state,
        per_core_state,
        per_core_state,
        qk,
        qk,
        raw_value,
        sequence(fp32 if beta_fp32 else bf16, physical=beta_p, size=beta_l, width=1),
        sequence(bf16, physical=inverse_p, size=inverse_l, width=CHUNK_SIZE),
        cache,
        cache,
        cache_f32,
        compact_f32,
        compact_f32,
        compact_f32,
        compact_f32,
        chunk_matrix,
        gate_last,
        metadata,
        raw_value,
        sequence(fp32 if beta_grad_fp32 else bf16, physical=p, size=length, width=1),
        sequence(fp32 if gate_fp32 else bf16, physical=p, size=length),
        fake((nv,), fp32),
        fake((nv, SUPPORTED_HEAD_DIM), fp32),
        rstd,
        rstd,
        qk,
        qk,
        sequence(fp32 if gate_fp32 else bf16, physical=p, size=length),
        fake((nv, GATE_SCALAR_TRANSFER_WIDTH), fp32),
        fake((nv, SUPPORTED_HEAD_DIM), fp32),
    ]
    # The order mirrors _prepare_stage_one's stage_one_scratch dictionary.
    scratch = (
        (bf16, CHUNK_SIZE),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, None),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, None),
        (fp32, CHUNK_SIZE),
        (bf16, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, "gate_last"),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32, CHUNK_SIZE),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, SUPPORTED_HEAD_DIM),
        (fp32, None),
        (bf16, SUPPORTED_HEAD_DIM),
        (fp32 if beta_grad_fp32 else bf16, None),
    )
    for dtype, width in scratch:
        shape = (
            (scratch_chunks, SUPPORTED_HEAD_DIM)
            if width == "gate_last"
            else (rows,)
            if width is None
            else (rows, width)
        )
        specs.append(fake(shape, dtype))
    specs.append(fake((dim("OWNER_SCHEDULE_WORDS", min=1),), dtypes.int32))
    # Per-argument strides accept BNSD, BSND, TND and independently strided
    # forward caches, including the padded beta-gradient allocation.
    for index, spec in enumerate(specs):
        if index < 58:
            if len(spec.shape) > 2:
                stride = tuple(
                    dim(f"ARG{index}_S{axis}", min=0)
                    for axis in range(len(spec.shape) - 1)
                ) + (1,)
                specs[index] = fake(spec.shape, spec.dtype, stride=stride)
    if cube_causal and (not use_full_chunks or staged):
        raise ValueError("Cube causal requires persistent full chunks")
    if not use_full_chunks and not needs_masked_chunks:
        raise ValueError("at least one backward chunk path must be enabled")
    config = (0,) * 9 + (
        bool(beta_fp32),
        0,
        bool(use_full_chunks),
        bool(needs_masked_chunks),
        bool(staged),
        bool(cube_causal),
    )
    return tuple(specs), (dtypes.float32, *_backward_scalar_specs(config), dtypes.int64)


def _cache_backward_program_abi(program):
    """Cache immutable ABI descriptions only on this backward program."""
    try:
        from cannbotdsl.core.compiler.function import ProviderCallable
        from cannbotdsl.core.compiler.parameter_lowering import ParameterLoweringPlan
    except ImportError:
        return program
    # This adaptation targets the inspected SDK plan layout. Other providers,
    # collection stubs and changed SDK layouts retain their original behavior.
    if type(program) is not ProviderCallable or program.closed or program._noop:
        return program
    plan = program._parameter_lowering
    fields = ("contract", "validation", "layout", "_host_ctypes")
    if (
        type(plan) is not ParameterLoweringPlan
        or tuple(getattr(ParameterLoweringPlan, "__slots__", ())) != fields
    ):
        return program

    class _BackwardCachedLowering(ParameterLoweringPlan):
        __slots__ = ("_cached_abi_arguments",)

        def __init__(self, original):
            for name in fields:
                object.__setattr__(self, name, getattr(original, name))
            object.__setattr__(self, "_cached_abi_arguments", original.abi_arguments)

        @property
        def abi_arguments(self):
            return self._cached_abi_arguments

    program._parameter_lowering = _BackwardCachedLowering(plan)
    return program


def _compile_backward_program(
    beta_fp32,
    use_full_chunks,
    needs_masked_chunks,
    gate_fp32=False,
    staged=False,
    beta_grad_fp32=False,
    cube_causal=False,
):
    """Use one compile factory for Native collection and runtime caching."""
    launcher = FlashKDAGrad()
    specs, scalar_specs = backward_dynamic_compile_request(
        beta_fp32,
        use_full_chunks,
        needs_masked_chunks,
        gate_fp32,
        staged,
        beta_grad_fp32,
        cube_causal,
    )
    program = cannbotdsl.compile(launcher.run, *specs, *scalar_specs)
    return _BackwardProgramEntry(launcher, _cache_backward_program_abi(program), [])


def _close_backward_program(entry):
    for event in entry.pending_events:
        event.synchronize()
    for lease in tuple(_BACKWARD_WORKSPACE_LEASES.values()):
        if any(lease.event is event for event in entry.pending_events):
            _release_backward_workspace_lease(lease)
    entry.pending_events.clear()
    entry.program.close()


def _clear_backward_program_cache():
    """Release compiled code only after its outstanding device calls finish."""
    with _BACKWARD_SCRATCH.lock:
        _BACKWARD_SCRATCH.clear()
        with _BACKWARD_PROGRAM_LOCK:
            for lease in tuple(_BACKWARD_WORKSPACE_LEASES.values()):
                _synchronize_backward_workspace_lease(lease)
                _release_backward_workspace_lease(lease)
            while _BACKWARD_PROGRAMS:
                key, entry = next(iter(_BACKWARD_PROGRAMS.items()))
                _close_backward_program(entry)
                del _BACKWARD_PROGRAMS[key]
    _stop_backward_workspace_retirement()


def _launch_backward_program(device, arguments, specs, config, scale, core_num):
    # Cache code by configuration. Tensor owners belong only to the pending
    # completion lease and are retired as soon as that call's event finishes.
    key = (
        cannbotdsl.compile,
        str(device),
        config[9],
        bool(config[11]),
        bool(config[12]),
        bool(config[13]),
        specs[_BACKWARD_RAW_GATE_INDEX].dtype == dtypes.float32,
        specs[_BACKWARD_BETA_GRAD_INDEX].dtype == dtypes.float32,
        bool(config[14]),
    )
    with torch.npu.device(device), _BACKWARD_PROGRAM_LOCK:
        entry = _BACKWARD_PROGRAMS.get(key)
        if entry is None:
            entry = _compile_backward_program(
                config[9], config[11], config[12], key[-3], config[13], key[-2], key[-1]
            )
            _BACKWARD_PROGRAMS[key] = entry
        _BACKWARD_PROGRAMS.move_to_end(key)
        stream = torch.npu.current_stream(torch.npu.current_device())
        for tensor in arguments:
            tensor.record_stream(stream)
        finished = torch.npu.Event()
        entry.pending_events[:] = [
            event for event in entry.pending_events if not event.query()
        ]
        lease = _BackwardWorkspaceLease(
            finished, device, arguments, stream, event_recorded=False
        )
        _BACKWARD_WORKSPACE_LEASES[id(lease)] = lease
        try:
            entry.program(
                *arguments, scale, *_backward_runtime_config(config), core_num
            )
            finished.record(stream)
            lease.event_recorded = True
            entry.pending_events.append(finished)
            _queue_backward_workspace_retirement(lease)
        except Exception:
            try:
                stream.synchronize()
            except Exception as error:
                lease.error = error
                raise
            _release_backward_workspace_lease(lease)
            raise
        while len(_BACKWARD_PROGRAMS) > _BACKWARD_PROGRAM_LIMIT:
            _, oldest = _BACKWARD_PROGRAMS.popitem(last=False)
            _close_backward_program(oldest)


atexit.register(_clear_backward_program_cache)


def _validate_backward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    chunk_states: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    v_new: torch.Tensor,
    qg: torch.Tensor,
    kg: torch.Tensor,
    g_cumsum: torch.Tensor,
    q_rtsd: Optional[torch.Tensor],
    k_rstd: Optional[torch.Tensor],
    beta_sigmoid: Optional[torch.Tensor],
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    metadata: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
    disable_recompute: bool = False,
    initial_state: Optional[torch.Tensor] = None,
):
    """Validate the saved-cache ABI before preparing any compute launch."""
    if not isinstance(do, torch.Tensor):
        raise ValueError(
            "do must be a Tensor; pass a zero tensor for an absent output gradient"
        )
    if not isinstance(dht, torch.Tensor):
        raise ValueError(
            "dht must be a Tensor; pass a zero tensor for an absent final-state gradient"
        )
    required = (q, k, v, g, beta, Aqk, Akk)
    if disable_recompute:
        required += (chunk_states, w, u, v_new, qg, kg, g_cumsum)
    if any(t is None for t in required):
        raise ValueError("backward requires all saved forward caches")
    if any(not isinstance(t, torch.Tensor) for t in required):
        raise ValueError("backward inputs and required saved caches must be Tensors")
    if q.device.type != "npu" or any(t.device != q.device for t in required):
        raise ValueError("inputs and caches must be on one NPU")
    if any(
        t.dtype != torch.bfloat16
        for t in (
            (q, k, v, Aqk, Akk, w, u, v_new, qg, kg)
            if disable_recompute
            else (q, k, v, Aqk, Akk)
        )
    ):
        raise ValueError(
            "the first specialization requires BF16 token and matrix planes"
        )
    if any(t.dtype not in (torch.bfloat16, torch.float32) for t in (g, beta)):
        raise ValueError("g and beta must be BF16 or FP32")
    if disable_recompute and (
        chunk_states.dtype != torch.float32 or g_cumsum.dtype != torch.float32
    ):
        raise ValueError("chunk_states and g_cumsum must be FP32")
    if layout not in ("TND", "BNSD", "BSND"):
        raise ValueError("layout must be TND, BNSD, or BSND")
    if (
        q.ndim != (3 if layout == "TND" else 4)
        or q.shape != k.shape
        or q.shape[-1] != SUPPORTED_HEAD_DIM
    ):
        raise ValueError("Q/K layout or width is invalid")
    if v.ndim != q.ndim:
        raise ValueError("V rank must match Q")
    value_heads = v.shape[1] if layout in ("TND", "BNSD") else v.shape[2]
    qk_heads = q.shape[1] if layout in ("TND", "BNSD") else q.shape[2]
    if value_heads < 1 or qk_heads < 1 or value_heads % qk_heads:
        raise ValueError("value heads must be divisible by QK heads")
    if layout == "TND":
        value_shape = (q.shape[0], value_heads, SUPPORTED_HEAD_DIM)
        storage_length = q.shape[0]
    elif layout == "BNSD":
        value_shape = (q.shape[0], value_heads, q.shape[2], SUPPORTED_HEAD_DIM)
        storage_length = q.shape[2]
    else:
        value_shape = (q.shape[0], q.shape[1], value_heads, SUPPORTED_HEAD_DIM)
        storage_length = q.shape[1]
    if (
        v.shape != value_shape
        or g.shape != value_shape
        or beta.shape != value_shape[:-1]
    ):
        raise ValueError("V/g/beta physical shapes do not match Q")
    if do.shape != v.shape or do.dtype != torch.bfloat16 or do.device != q.device:
        raise ValueError("do must match BF16 v")
    if (
        not isinstance(metadata, torch.Tensor)
        or metadata.ndim != 1
        or metadata.dtype != torch.int32
        or metadata.device != q.device
        or not metadata.is_contiguous()
        or (metadata.numel() - HEADER_WORDS) % (TASK_WORDS * value_heads)
    ):
        raise ValueError("metadata must use the backward int32 task ABI")
    batch = (metadata.numel() - HEADER_WORDS) // (TASK_WORDS * value_heads)
    if batch < 1 or metadata.numel() != metadata_capacity(batch, value_heads):
        raise ValueError("metadata task count is invalid")
    if layout != "TND" and batch != q.shape[0]:
        raise ValueError("metadata batch does not match padded inputs")
    wait_metadata_ready(metadata)
    host_info = _metadata.metadata_host_info(metadata)
    metadata_header = host_info.header
    if (
        metadata_header[HEADER_MAGIC] != MAGIC
        or metadata_header[HEADER_ABI_VERSION] != ABI_VERSION
    ):
        raise ValueError("backward metadata ABI does not match")
    if metadata_header[HEADER_STATUS] != STATUS_OK:
        raise ValueError(f"backward metadata status {metadata_header[HEADER_STATUS]}")
    metadata_flags = metadata_header[HEADER_FLAGS]
    if (
        metadata_header[HEADER_BATCH] != batch
        or metadata_header[HEADER_VALUE_HEADS] != value_heads
        or metadata_header[HEADER_STORAGE_LENGTH] != storage_length
        or metadata_flags not in (0, 1, 3)
        or bool(metadata_flags & 2) != (layout == "TND")
    ):
        raise ValueError("backward metadata input shape or layout does not match")
    chunk_count = metadata_header[HEADER_CHUNK_COUNT]
    if chunk_count < 0 or chunk_count > batch * (
        (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
    ):
        raise ValueError("backward metadata chunk count is invalid")
    if layout == "TND":
        cache_shape = (storage_length, value_heads, SUPPORTED_HEAD_DIM)
        qk_scalar_shape = (storage_length, qk_heads)
        capacity = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE + batch
        matrix_shape = (capacity, value_heads, CHUNK_SIZE, CHUNK_SIZE)
        state_shape = (capacity, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)
    else:
        cache_shape = (batch, value_heads, storage_length, SUPPORTED_HEAD_DIM)
        qk_scalar_shape = (batch, qk_heads, storage_length)
        chunks = (storage_length + CHUNK_SIZE - 1) // CHUNK_SIZE
        matrix_shape = (batch, value_heads, chunks, CHUNK_SIZE, CHUNK_SIZE)
        state_shape = (
            batch,
            value_heads,
            chunks,
            SUPPORTED_HEAD_DIM,
            SUPPORTED_HEAD_DIM,
        )
    if disable_recompute:
        if qg.shape != cache_shape:
            raise ValueError("qg must use the current forward cache shape")
        if any(t.shape != cache_shape for t in (kg, w, u, v_new, g_cumsum)):
            raise ValueError("saved token caches must use the current forward layout")
    if Aqk.shape != matrix_shape or Akk.shape != matrix_shape:
        raise ValueError("Aqk/Akk must use the current forward chunk cache shape")
    if disable_recompute and chunk_states.shape != state_shape:
        raise ValueError("FP32 chunk_states shape must match the current forward cache")
    if (
        dht.dtype != torch.float32
        or dht.device != q.device
        or dht.shape != (batch, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)
    ):
        raise ValueError(
            f"dht must be FP32 [sequence, value_head, {SUPPORTED_HEAD_DIM}, {SUPPORTED_HEAD_DIM}]"
        )

    if not disable_recompute and initial_state is not None:
        if (
            not isinstance(initial_state, torch.Tensor)
            or initial_state.dtype != torch.float32
            or initial_state.device != q.device
            or initial_state.shape
            != (batch, value_heads, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM)
            or initial_state.stride(-1) != 1
        ):
            raise ValueError(
                "initial_state must be FP32 [sequence, value_head, 128, 128] on the input device with contiguous rows"
            )

    if use_qk_l2norm_in_kernel:
        if (
            not isinstance(q_rtsd, torch.Tensor)
            or not isinstance(k_rstd, torch.Tensor)
            or q_rtsd.shape != qk_scalar_shape
            or k_rstd.shape != qk_scalar_shape
            or q_rtsd.dtype != torch.float32
            or k_rstd.dtype != torch.float32
            or any(t.device != q.device for t in (q_rtsd, k_rstd))
        ):
            raise ValueError(
                "use_qk_l2norm_in_kernel requires valid FP32 q_rtsd and k_rstd caches"
            )
    if use_beta_sigmoid_in_kernel:
        if (
            not isinstance(beta_sigmoid, torch.Tensor)
            or beta_sigmoid.shape != cache_shape[:-1]
            or beta_sigmoid.dtype != torch.float32
            or beta_sigmoid.device != q.device
        ):
            raise ValueError("activated beta cache must be FP32")
    if use_gate_in_kernel:
        if (
            not isinstance(A_log, torch.Tensor)
            or A_log.shape != (value_heads,)
            or A_log.dtype != torch.float32
            or A_log.device != q.device
        ):
            raise ValueError("A_log must be FP32 [value_heads]")
    if dt_bias is not None and (
        not isinstance(dt_bias, torch.Tensor)
        or dt_bias.shape != (value_heads, SUPPORTED_HEAD_DIM)
        or dt_bias.dtype != torch.float32
        or dt_bias.device != q.device
    ):
        raise ValueError(f"dt_bias must be FP32 [value_heads, {SUPPORTED_HEAD_DIM}]")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires activated beta")
    if lower_bound is not None and not (-5.0 <= lower_bound < 0.0):
        raise ValueError("lower_bound must be None or in [-5, 0)")


def _persistent_owner_plan(q, v, layout, metadata, core_num):
    axis = 2 if layout == "BSND" else 1
    return _metadata.backward_schedule(metadata, q.shape[axis], v.shape[axis], core_num)


def _owner_schedule_for_launch(q, v, layout, metadata, core_num, staged):
    axis = 2 if layout == "BSND" else 1
    return _metadata.owner_schedule_for_launch(
        metadata, q.shape[axis], v.shape[axis], core_num, staged
    )


def _select_backward_template(q, v, layout, metadata, core_num):
    return _persistent_owner_plan(q, v, layout, metadata, core_num).strategy


def _track_chunk_plan(q, v, layout, metadata, block_size, staged=False):
    axis = 2 if layout == "BSND" else 1
    return _metadata.backward_chunk_paths(
        metadata, q.shape[axis], v.shape[axis], block_size, staged
    )


def _resolve_backward_metadata(q, v, layout, metadata, cu_seqlens, seqused):
    """Require caller-supplied metadata and check any explicit lengths."""
    if metadata is None:
        raise ValueError(
            "metadata is required: passing None is not supported yet; "
            "call flash_kda_grad_metadata first"
        )
    if cu_seqlens is None and seqused is None:
        return metadata
    if cu_seqlens is not None and layout != "TND":
        raise ValueError(
            "bwd cu_seqlens is only supported for TND; BNSD/BSND use seqused"
        )
    if seqused is not None and layout not in ("BNSD", "BSND"):
        raise ValueError(
            "bwd seqused is only supported for BNSD/BSND; TND uses cu_seqlens"
        )
    if not isinstance(q, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise ValueError("q and v must be Tensors")
    if q.ndim != (3 if layout == "TND" else 4) or v.ndim != q.ndim:
        raise ValueError("q/v rank does not match layout")
    head_axis = 2 if layout == "BSND" else 1
    length_axis = 0 if layout == "TND" else 1 if layout == "BSND" else 2
    if (
        q.device.type != "npu"
        or v.device != q.device
        or any(
            t.device != q.device
            for t in (cu_seqlens, seqused)
            if isinstance(t, torch.Tensor)
        )
    ):
        raise ValueError("q/v and sequence lengths must be on the same NPU")
    if q.shape[length_axis] != v.shape[length_axis] or (
        layout != "TND" and q.shape[0] != v.shape[0]
    ):
        raise ValueError("q/v sequence dimensions do not match")
    if layout == "TND" and not isinstance(cu_seqlens, torch.Tensor):
        raise ValueError("cu_seqlens must be a Tensor")
    generated = _metadata.flash_kda_grad_metadata(
        cu_seqlens,
        seqused,
        batch_size=cu_seqlens.numel() - 1 if layout == "TND" else q.shape[0],
        n_heads_qk=q.shape[head_axis],
        n_heads_v=v.shape[head_axis],
        seqlens=q.shape[length_axis],
        headdim_qk=q.shape[-1],
        headdim_v=v.shape[-1],
        layout=layout,
    )
    # Reading the generated snapshot also checks device-side length validation.
    expected = _metadata.metadata_host_info(generated)
    if metadata is generated:
        return metadata
    if (
        not isinstance(metadata, torch.Tensor)
        or metadata.dtype != torch.int32
        or metadata.device != q.device
        or metadata.shape != generated.shape
        or not metadata.is_contiguous()
    ):
        raise ValueError("metadata must match the explicit sequence-length task ABI")
    wait_metadata_ready(metadata)
    actual = _metadata.metadata_host_info(metadata)
    # An all-full padded snapshot may use flags=0 instead of flags=1 while
    # describing exactly the same token ranges. Packed layout must still agree.
    if (
        actual.header[:HEADER_FLAGS] != expected.header[:HEADER_FLAGS]
        or (actual.header[HEADER_FLAGS] & 2) != (expected.header[HEADER_FLAGS] & 2)
        or actual.sequences != expected.sequences
    ):
        raise ValueError("metadata does not match cu_seqlens/seqused")
    return metadata


def _forward_token_view(cache, layout):
    """Forward padded caches are head-major even for BSND inputs."""
    return cache.transpose(1, 2) if layout == "BSND" and cache is not None else cache


def _forward_matrix_view(cache, values, layout):
    if layout == "TND":
        return cache
    length = values.shape[1] if layout == "BSND" else values.shape[2]
    return _forward_token_view(cache.flatten(2, 3)[:, :, :length], layout)


def _forward_qk_to_input_layout(cache, reference, layout):
    expected = (
        (reference.shape[0], reference.shape[2], reference.shape[1], reference.shape[3])
        if layout == "BSND"
        else reference.shape
    )
    if (
        not isinstance(cache, torch.Tensor)
        or cache.dtype != torch.bfloat16
        or cache.device != reference.device
        or cache.shape != expected
    ):
        raise ValueError("saved Q/K cache must use the current forward layout")
    return _forward_token_view(cache, layout)


# Recompute and backward share one kernel with a separate program cache.
# The saved-cache path never allocates or reconstructs these seven caches.


@kernel(
    profile=cannbotdsl.ProfileSpec(
        name="DSL_flash_kda_grad_recompute",
        op_type="FlashKDAGradRecompute",
        inputs=(
            "stage_two_gm_Aqk",
            "stage_two_gm_chunk_states",
            "stage_two_gm_output_gradient",
            "stage_two_gm_final_state_gradient",
            "stage_two_gm_task_metadata",
            "stage_one_gm_effective_query",
            "stage_one_gm_effective_key",
            "stage_one_gm_value",
        ),
        outputs=(
            "stage_one_gm_public_d_query",
            "stage_one_gm_public_d_key",
            "stage_one_gm_public_d_value",
            "stage_two_gm_dh0",
            "stage_one_gm_public_d_gate",
            "stage_one_gm_public_d_beta",
            "stage_one_gm_public_dA",
            "stage_one_gm_public_dBias",
        ),
    )
)
class _RecomputeBackwardKernel:
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
        staged: Constexpr[bool],
        cube_causal: Constexpr[bool],
        has_initial_state: dtypes.int64,
        stage_two_scale: dtypes.float32,
        stage_one_scale: dtypes.float32,
        stage_one_inverse_factor: dtypes.float32,
        stage_one_lower_bound: dtypes.float32,
    ):
        with _SharedCubeScope():
            _initialize_backward_gradients(
                stage_one_gm_public_dA, stage_one_gm_public_dBias, stage_one_fused_gate
            )
            _recompute_tracks(
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
                staged or needs_masked_chunks or not use_full_chunks,
            )
            # Full, unmasked persistent owners can reconstruct locally.
            # Publish every cache before either backward template changes owners.
            if const_expr(
                use_full_chunks and needs_masked_chunks and cube_causal and not staged
            ):
                # Reconstruction has drained and published its GM caches. Only
                # the persistent full/tail Cube path needs a fresh ID epoch.
                _sync_cube_vector_handoff(2, grid=True, reset_sync_id=True)
            else:
                _sync_cube_vector_handoff(2, grid=True)
            _backward_template(staged)(
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
                True,
            )


class _RecomputeBackward:
    @host
    def run(
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
        stage_two_scale: dtypes.float32,
        stage_one_scale: dtypes.float32,
        stage_one_fused_beta: dtypes.int64,
        stage_one_inverse_factor: dtypes.float32,
        stage_one_heads_per_key_head: dtypes.int64,
        stage_one_fused_gate: dtypes.int64,
        stage_one_has_bias: dtypes.int64,
        stage_one_use_l2norm: dtypes.int64,
        stage_one_softplus: dtypes.int64,
        stage_one_lower_bound: dtypes.float32,
        stage_one_beta_fp32: Constexpr[bool],
        padded_layout: dtypes.int64,
        use_full_chunks: Constexpr[bool],
        needs_masked_chunks: Constexpr[bool],
        staged: Constexpr[bool],
        cube_causal: Constexpr[bool],
        has_initial_state: dtypes.int64,
        core_num: int,
    ):
        _RecomputeBackwardKernel()[core_num](
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
            staged,
            cube_causal,
            has_initial_state,
            stage_two_scale,
            stage_one_scale,
            stage_one_inverse_factor,
            stage_one_lower_bound,
        )


def _recompute_dynamic_compile_request(
    beta_fp32,
    use_full_chunks=False,
    needs_masked_chunks=True,
    gate_fp32=False,
    staged=False,
    beta_grad_fp32=False,
    cube_causal=False,
):
    """Fused two-phase ABI; the saved-cache backward ABI stays separate."""
    specs, scalars = backward_dynamic_compile_request(
        beta_fp32,
        use_full_chunks,
        needs_masked_chunks,
        gate_fp32,
        staged,
        beta_grad_fp32,
        cube_causal,
    )
    # Initial state follows dht; the U output shares W's cache layout.
    return (*specs, specs[8], specs[3]), (*scalars[:-1], dtypes.int64, scalars[-1])


_RECOMPUTE_PROGRAMS = OrderedDict()
_RECOMPUTE_LOCK = threading.RLock()


def _clear_recompute_program_cache():
    with _RECOMPUTE_LOCK, _BACKWARD_PROGRAM_LOCK:
        while _RECOMPUTE_PROGRAMS:
            key, entry = next(iter(_RECOMPUTE_PROGRAMS.items()))
            for lease in tuple(_BACKWARD_WORKSPACE_LEASES.values()):
                if any(
                    lease.event is pending or lease.stream is pending
                    for pending in entry.pending_events
                ):
                    _synchronize_backward_workspace_lease(lease)
                    _release_backward_workspace_lease(lease)
            _close_backward_program(entry)
            del _RECOMPUTE_PROGRAMS[key]


atexit.register(_clear_recompute_program_cache)


def _launch_recompute_program(
    device, arguments, config, scale, core_num, has_initial_state
):
    beta_fp32, full, masked, staged = config[9], config[11], config[12], config[13]
    cube_causal = bool(config[14])
    gate_fp32 = arguments[_BACKWARD_RAW_GATE_INDEX].dtype == torch.float32
    beta_grad_fp32 = arguments[_BACKWARD_BETA_GRAD_INDEX].dtype == torch.float32
    key = (
        cannbotdsl.compile,
        str(device),
        beta_fp32,
        bool(full),
        bool(masked),
        gate_fp32,
        bool(staged),
        beta_grad_fp32,
        cube_causal,
    )
    with torch.npu.device(device), _RECOMPUTE_LOCK, _BACKWARD_PROGRAM_LOCK:
        entry = _RECOMPUTE_PROGRAMS.get(key)
        if entry is None:
            launcher = _RecomputeBackward()
            specs, scalars = _recompute_dynamic_compile_request(
                beta_fp32, full, masked, gate_fp32, staged, beta_grad_fp32, cube_causal
            )
            program = cannbotdsl.compile(launcher.run, *specs, *scalars)
            # 保留 Cube 编译变体，并缓存不可变 ABI 描述，避免重复下发时重新构造。
            entry = _BackwardProgramEntry(
                launcher, _cache_backward_program_abi(program), []
            )
            _RECOMPUTE_PROGRAMS[key] = entry
        stream = torch.npu.current_stream(device)
        for tensor in arguments:
            tensor.record_stream(stream)
        event = torch.npu.Event()
        lease = _BackwardWorkspaceLease(
            event, device, arguments, stream, event_recorded=False
        )
        _BACKWARD_WORKSPACE_LEASES[id(lease)] = lease
        try:
            entry.program(
                *arguments,
                float(scale),
                *_backward_runtime_config(config),
                int(has_initial_state),
                core_num,
            )
            event.record(stream)
            lease.event_recorded = True
            entry.pending_events[:] = [
                pending for pending in entry.pending_events if not pending.query()
            ]
            entry.pending_events.append(event)
            _queue_backward_workspace_retirement(lease)
        except Exception:
            try:
                stream.synchronize()
            except Exception as error:
                lease.error = error
                # No completion event can be trusted. Retain the program too;
                # cleanup must retry this stream before releasing device code.
                entry.pending_events.append(stream)
                raise
            _release_backward_workspace_lease(lease)
            raise


def _allocate_recompute_caches(values, final_state_gradient, layout):
    """Allocate seven outputs without launching a device operation."""
    caches = tuple(
        values.new_empty(
            values.shape, dtype=torch.float32 if index == 5 else torch.bfloat16
        )
        for index in range(6)
    )
    batch, heads = final_state_gradient.shape[:2]
    length = (
        values.shape[0]
        if layout == "TND"
        else values.shape[1 if layout == "BSND" else 2]
    )
    chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
    state_prefix = (
        (chunks + batch, heads) if layout == "TND" else (batch, heads, chunks)
    )
    states = values.new_empty(
        (*state_prefix, SUPPORTED_HEAD_DIM, SUPPORTED_HEAD_DIM), dtype=torch.float32
    )
    return (*caches, states)


def _backward_variants():
    for staged in (False, True):
        paths = (
            ((False, True),) if staged else ((False, True), (True, False), (True, True))
        )
        # Fused beta is always FP32, independently of the original beta dtype.
        for beta_fp32, beta_grad_fp32 in ((False, False), (True, False), (True, True)):
            for gate_fp32 in (False, True):
                for full, masked in paths:
                    for cube_causal in (False, True) if full else (False,):
                        yield (
                            beta_fp32,
                            full,
                            masked,
                            gate_fp32,
                            staged,
                            beta_grad_fp32,
                            cube_causal,
                        )


def _backward_export_shard():
    """Select an isolated Native build worker; ordinary exports remain complete."""
    import os

    value = os.environ.get("CANNBOTDSL_FLASH_KDA_GRAD_EXPORT_SHARD", "0/1")
    try:
        index, count = map(int, value.split("/"))
    except ValueError as error:
        raise ValueError("backward export shard must be INDEX/COUNT") from error
    if not 1 <= count <= 72 or not 0 <= index < count:
        raise ValueError("backward export shard requires 0 <= INDEX < COUNT <= 72")
    return index, count


def _export_recompute_programs(shard=(0, 1)):
    variants = tuple(_backward_variants())
    for index, variant in enumerate(variants, start=len(variants)):
        if index % shard[1] != shard[0]:
            continue
        specs, scalars = _recompute_dynamic_compile_request(*variant)
        program = cannbotdsl.compile(_RecomputeBackward().run, *specs, *scalars)
        program.close()


def _disable_grad_at_call(fn):
    """Run ``fn`` with grad recording disabled, entering ``no_grad`` per call.

    The build-time torch stub used by host layout discovery only exposes
    symbolic declarations and has no ``no_grad`` attribute, so the context
    manager must not be created while this module is imported.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)

    return wrapper


@_disable_grad_at_call
def flash_kda_grad_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    metadata: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qg: Optional[torch.Tensor] = None,
    kg: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
    q_rtsd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
):
    """Return (dq, dk, dv, dh0, dg, dbeta, dA, dBias).

    Positional arguments follow the backward-operator sheet in KDA.xls.
    Optional inputs and attributes also accept keyword arguments.
    disable_recompute=False rebuilds w/u/v_new/qg/kg/g_cumsum and
    FP32 chunk_states, ignoring those seven supplied caches. initial_state
    is the forward's FP32 [B,Nv,128,128] state; None means a zero state.
    False (default) rebuilds the caches to save device memory; True uses the
    separate saved-cache kernel and requires all seven caches.
    metadata is a required input today: it is declared optional in the schema so
    callers can omit it once metadata derivation lands, but passing None raises.
    Aqk/Akk are required Tensor arguments in both modes; conditional
    normalization/beta caches remain required when their fusion is enabled.
    g/beta independently accept BF16 or FP32. Reconstruction rounds raw g/beta
    to BF16 like forward, while retained FP32 beta activations are read directly.
    Backward vector epilogues write dg/dbeta
    directly in the corresponding original g/beta dtype, rounding only for BF16.
    With Q/K L2Norm enabled, q/k are the saved normalized forward values;
    q_rtsd/k_rstd are required FP32 inverse norms. The returned dq/dk are
    gradients of the original pre-normalization inputs.
    Host checks supplied sequence metadata, validates the cache ABI, allocates outputs
    and workspace, prepares forward-cache views, and
    launches one compute kernel. The host automatically selects persistent
    or staged scheduling. Recompute mode first rebuilds the forward
    tracks in that same kernel. Persistent keeps the same owners and uses a
    block-local handoff; staged waits for all cores before its four backward
    phases, giving five phases with recompute enabled.
    ``u`` remains in the forward-cache ABI; its gradient is represented by
    the Stage 2 ``d_v_new`` path and does not require the saved ``u`` values.
    Caches must use the current forward output layout. TND chunk storage
    starts at floor(token_start / 64) + sequence; padded caches are head-major
    and matrix caches retain their explicit chunk axis.
    Layout defaults to TND, fusion flags default to True, and omitted/None
    scale uses 1/sqrt(Dqk). With gate fusion enabled, lower_bound=None (default)
    uses -exp(A_log) * softplus(dt_bias + g); an explicit bound in [-5, 0)
    uses lower_bound * sigmoid(exp(A_log) * (dt_bias + g)).
    TND accepts int32 [B+1] ``cu_seqlens``; BNSD/BSND accept
    int32 [B] ``seqused``. metadata is a required int32 snapshot built by
    flash_kda_grad_metadata; any explicit lengths must agree with it.
    Both ``do`` and ``dht`` must be tensors; pass zeros for an absent gradient.
    """
    if scale is not None:
        try:
            scale = float(scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("scale must be a finite number or None") from error
        if not math.isfinite(scale):
            raise ValueError("scale must be a finite number or None")
    metadata = _resolve_backward_metadata(q, v, layout, metadata, cu_seqlens, seqused)
    _validate_backward_inputs(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        chunk_states,
        Aqk,
        Akk,
        w,
        u,
        v_new,
        qg,
        kg,
        g_cumsum,
        q_rtsd,
        k_rstd,
        beta_sigmoid,
        A_log,
        dt_bias,
        metadata,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        lower_bound,
        layout,
        disable_recompute,
        initial_state,
    )
    Aqk = _forward_matrix_view(Aqk, v, layout)
    Akk = _forward_matrix_view(Akk, v, layout)
    if use_beta_sigmoid_in_kernel:
        beta_sigmoid = _forward_token_view(beta_sigmoid, layout)
    if use_qk_l2norm_in_kernel:
        q_rtsd = _forward_token_view(q_rtsd, layout)
        k_rstd = _forward_token_view(k_rstd, layout)
    if not disable_recompute:
        w, u, v_new, qg, kg, g_cumsum, chunk_states = _allocate_recompute_caches(
            v, dht, layout
        )
    else:
        w, u, v_new, qg, kg, g_cumsum = (
            _forward_token_view(tensor, layout)
            for tensor in (w, u, v_new, qg, kg, g_cumsum)
        )
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    _reap_backward_workspaces()
    core_num = _device_block_num(q)
    selected_template = _select_backward_template(q, v, layout, metadata, core_num)
    staged = selected_template == "staged"
    # All local causal contractions use direct gate differences. This legacy
    # flag only selects the existing fused bounded-gate epilogue/storage path.
    causal_block_size = (
        CHUNK_SIZE if use_gate_in_kernel and lower_bound is not None else 1
    )
    use_full_chunks, needs_masked_chunks = _track_chunk_plan(
        q,
        v,
        layout,
        metadata,
        causal_block_size,
        staged,
    )
    if not disable_recompute and not staged and needs_masked_chunks:
        # A fused reconstruction followed by the persistent full/tail branches
        # gives channels conditional ownership across phase boundaries. Use the
        # staged backward schedule so every channel has one acquire/release path.
        selected_template = "staged"
        staged = True
        use_full_chunks, needs_masked_chunks = False, True
    with _BACKWARD_SCRATCH.acquire(
        q, (core_num, layout, tuple(q.shape), selected_template)
    ) as workspace:
        stage_two, stage_two_arguments, stage_two_specs = _prepare_stage_two(
            aqk=Aqk,
            qg=qg,
            kg=kg,
            w=w,
            v_new=v_new,
            g_cumsum=g_cumsum,
            chunk_states=chunk_states,
            output_gradient=do,
            final_state_gradient=dht,
            layout=layout,
            metadata=metadata,
            scale=scale,
            core_num=core_num,
            _workspace=workspace,
            _needs_masked_chunks=needs_masked_chunks,
            _staged=staged,
        )
        stage_one_result, stage_one_arguments, stage_one_specs, stage_one_config = (
            _prepare_stage_one(
                query=q,
                key=k,
                values=v,
                raw_gate=g,
                activated_beta=beta_sigmoid if use_beta_sigmoid_in_kernel else beta,
                saved_inverse=Akk,
                saved_qg=qg,
                saved_kg=kg,
                saved_gamma=g_cumsum,
                A_log=A_log,
                dt_bias=dt_bias,
                query_rstd=q_rtsd if use_qk_l2norm_in_kernel else None,
                key_rstd=k_rstd if use_qk_l2norm_in_kernel else None,
                stage_two=stage_two,
                layout=layout,
                metadata=metadata,
                scale=scale,
                fused_beta=use_beta_sigmoid_in_kernel,
                fused_gate=use_gate_in_kernel,
                use_l2norm=use_qk_l2norm_in_kernel,
                inverse_factor=0.5 if allow_neg_eigval else 1.0,
                lower_bound=lower_bound,
                beta_dtype=beta.dtype,
                _workspace=workspace,
            )
        )
        owner_schedule = _owner_schedule_for_launch(
            q, v, layout, metadata, core_num, staged
        )
        arguments = (*stage_two_arguments, *stage_one_arguments, owner_schedule)
        cube_causal = bool(
            use_full_chunks
            and use_gate_in_kernel
            and lower_bound is not None
            and abs(lower_bound) * (CHUNK_SIZE - 1) <= 70.0
            and not staged
        )
        config = (
            *stage_one_config,
            use_full_chunks,
            needs_masked_chunks,
            staged,
            cube_causal,
        )
        if disable_recompute:
            _launch_backward_program(
                q.device,
                arguments,
                (
                    *stage_two_specs,
                    *stage_one_specs,
                    _backward_tensor_spec(owner_schedule, dtypes.int32),
                ),
                config,
                float(scale),
                core_num,
            )
        else:
            u_view = (
                u.unsqueeze(0).transpose(1, 2)
                if layout == "TND"
                else _forward_token_view(u, layout)
            )
            _launch_recompute_program(
                q.device,
                (*arguments, dht if initial_state is None else initial_state, u_view),
                config,
                float(scale),
                core_num,
                initial_state is not None,
            )
        dq, dk, dv, dg, dbeta, dA, dBias = stage_one_result
        return dq, dk, dv, stage_two.dh0, dg, dbeta, dA, dBias


def flash_kda_grad_from_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    forward_outputs: tuple[torch.Tensor | None, ...],
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    metadata: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    disable_recompute: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states_fp32: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
):
    """Consume the 16 DSL training-forward outputs in their return order.

    The first two outputs are ``out`` and ``final_state``. ``do`` and ``dht``
    are their required upstream gradient tensors; use zeros for an absent
    gradient. Layout defaults to TND. Backward task metadata is required;
    optional explicit lengths are checked against the supplied snapshot.
    Optional ``chunk_states_fp32`` must use the same FP32 shape and storage
    layout as the current forward's state cache when recomputation is disabled.
    Recompute mode uses initial_state (None means zeros) instead. Other caches are passed
    through unchanged. With L2Norm enabled, saved normalized Q/K are viewed in
    the public input layout; matrix/token views are prepared by the main entry.
    lower_bound defaults to None (softplus gate); pass the forward's explicit
    bound when it used a bounded sigmoid gate.
    """
    if len(forward_outputs) != 16:
        raise ValueError("training forward must return 16 tensors/cache slots")
    metadata = _resolve_backward_metadata(q, v, layout, metadata, cu_seqlens, seqused)
    (
        _,
        _,
        q_norm,
        k_norm,
        q_rtsd,
        k_rstd,
        beta_activated,
        Aqk,
        Akk,
        w,
        u,
        qg,
        kg,
        v_new,
        g_cumsum,
        saved_chunk_states,
    ) = forward_outputs
    chunk_states = (
        chunk_states_fp32 if chunk_states_fp32 is not None else saved_chunk_states
    )
    if use_qk_l2norm_in_kernel:
        q = _forward_qk_to_input_layout(q_norm, q, layout)
        k = _forward_qk_to_input_layout(k_norm, k, layout)
    return flash_kda_grad_npu(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        Aqk,
        Akk,
        metadata,
        initial_state=initial_state,
        chunk_states=chunk_states,
        w=w,
        u=u,
        v_new=v_new,
        qg=qg,
        kg=kg,
        g_cumsum=g_cumsum,
        q_rtsd=q_rtsd,
        k_rstd=k_rstd,
        beta_sigmoid=beta_activated,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        lower_bound=lower_bound,
        layout=layout,
        disable_recompute=disable_recompute,
    )


from cannbotdsl import aot  # noqa: E402 - Keep the existing module initialization order.


def _compile_recompute_export_variant(*variant):
    """Compile one recompute AOT variant (invoked by the AOT variant pool)."""
    specs, scalars = _recompute_dynamic_compile_request(*variant)
    cannbotdsl.compile(_RecomputeBackward().run, *specs, *scalars)


def _compile_export_variant(kind, *variant):
    """Compile exactly one AOT variant; entry point of the parallel variant pool."""
    if kind == "backward":
        _compile_backward_program(*variant)
    elif kind == "recompute":
        _compile_recompute_export_variant(*variant)
    else:
        raise ValueError(f"unknown AOT variant kind: {kind!r}")


def _export_variant_calls():
    """Enumerate every AOT variant without compiling, so the build can batch them."""
    for variant in _backward_variants():
        yield ("backward", *variant), {}
    for variant in _backward_variants():
        yield ("recompute", *variant), {}


@aot.export("flash_kda_grad")
def export_flash_kda_grad():
    """Export both templates, cache modes and independent g/beta storage types."""
    shard = _backward_export_shard()
    entries = []
    try:
        for index, variant in enumerate(_backward_variants()):
            if index % shard[1] == shard[0]:
                entries.append(_compile_backward_program(*variant))
    finally:
        for entry in entries:
            _close_backward_program(entry)
    _export_recompute_programs(shard)


# Declare the variants so the build compiles them one call at a time in its worker
# pool instead of running the whole export body serially. Builds that do not
# understand this attribute keep calling ``export_flash_kda_grad`` above.
export_flash_kda_grad.__cannbot_aot_variants__ = (
    "_compile_export_variant",
    _export_variant_calls,
)


__all__ = ["flash_kda_grad_npu", "flash_kda_grad_from_forward"]
