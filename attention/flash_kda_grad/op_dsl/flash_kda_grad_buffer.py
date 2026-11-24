# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared GM/UB/L1/L0 buffers, layouts and synchronization primitives."""

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
)
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    update_mask,
    vcast,
    vdups,
    vmem_bar,
    vstore,
    vstore_pack,
)
from cannbotdsl.ops.sync import (
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    onchip_rewind,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import tile_slice
from contextvars import ContextVar

if __package__:
    from .flash_kda_grad_metadata import (
        CHUNK_SIZE,
        SUPPORTED_HEAD_DIM,
        HEADER_WORDS,
        TASK_WORDS,
        TASK_FLAT_CHUNK_START,
    )
else:
    from flash_kda_grad_metadata import (
        CHUNK_SIZE,
        SUPPORTED_HEAD_DIM,
        HEADER_WORDS,
        TASK_WORDS,
        TASK_FLAT_CHUNK_START,
    )

KEY_DIM = VALUE_DIM = SUPPORTED_HEAD_DIM
LANES = D_HALF = 64
CAUSAL_BATCH_ROWS = 8
GATE_BATCH_ROWS = 64


FP32_ELEMENT_BYTES = 4


BF16_ELEMENT_BYTES = 2
CUBE_TILE_ROWS = 16
NZ_BLOCK_WIDTH = 16
GATE_SCALAR_TRANSFER_WIDTH = 8


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


_CUBE_SCRATCH_SCOPE = ContextVar("kdag_cube_scratch", default=None)


_CHUNK_SCRATCH_SCOPE = ContextVar("kdag_stage_one_scratch", default=None)


def _shared_cube_slot(mem_loc, dtype, columns):
    """Allocate one shared NZ slot with the same row capacity and depth."""
    rows = SUPPORTED_HEAD_DIM
    return RingBuffer(
        mem_loc,
        dtype,
        make_layout((rows, columns)),
        depth=1,
        data_format="nz",
        physical_layout=make_layout((columns // 16, rows // 16, 16, 16)),
    ).next_write()


class _SharedCubeScope:
    def __enter__(self):
        maximum = SUPPORTED_HEAD_DIM
        resources = {
            MemLoc.L0A: _shared_cube_slot(MemLoc.L0A, dtypes.bfloat16, maximum),
            MemLoc.L0B: _shared_cube_slot(MemLoc.L0B, dtypes.bfloat16, maximum),
            # W and U retain disjoint NZ column panels during reconstruction.
            MemLoc.L0C: _shared_cube_slot(MemLoc.L0C, dtypes.float32, 2 * maximum),
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


class _ChunkScratchScope:
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
        windows = []
        capacity = 0
        for name, dtype, shape in fields:
            offset = (capacity + 31) // 32 * 32
            windows.append((name, dtype, shape, offset))
            capacity = offset + shape[0] * shape[1] * (
                4 if dtype == dtypes.float32 else 2
            )
        storage = RingBuffer(
            MemLoc.UB, dtypes.uint8, make_layout((capacity,)), depth=1
        ).next_write()
        pool = {
            (name, dtype, shape): from_tensor(
                storage, dtype=dtype, layout=make_layout(shape), offset=offset
            )
            for name, dtype, shape, offset in windows
        }
        self.token = _CHUNK_SCRATCH_SCOPE.set(pool)
        return self

    def __exit__(self, *_):
        _CHUNK_SCRATCH_SCOPE.reset(self.token)


def _chunk_scratch(name, dtype, shape):
    """Reuse matching Stage 1 storage across mutually exclusive chunk branches."""
    key = (name, dtype, tuple(shape))
    pool = _CHUNK_SCRATCH_SCOPE.get()
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


class StateMatmulResources:
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


class ChunkWorkspace:
    """Shared chunk buffers and compile-time modes; contains no gradient formulas."""

    def __init__(
        self,
        batch_rows: bool = False,
        direct_results=False,
        fuse_base=False,
        resident_dr=None,
        resident_dw=None,
        round_raw_gate=False,
    ):
        self.round_raw_gate = round_raw_gate
        self.direct_results = direct_results
        self.fuse_base = fuse_base
        if fuse_base:
            self.fused_terminal = _chunk_scratch(
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
        fp32_storage = _chunk_scratch(
            "fp32", dtypes.float32, (fp32_capacity, SUPPORTED_HEAD_DIM)
        )
        bf16_storage = _chunk_scratch(
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
            self.batch_scalar = _chunk_scratch(
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
        scalar_storage = _chunk_scratch(
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
            _chunk_scratch("low_scalars", dtypes.bfloat16, (1, CHUNK_SIZE)).view(
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
            _chunk_scratch(
                "value_gradient",
                dtypes.float32,
                (max(3 * CHUNK_SIZE // 2, 12 * CAUSAL_BATCH_ROWS), VALUE_DIM),
            )
        )
        output_rows = 2 * CHUNK_SIZE if direct_results or fuse_base else CHUNK_SIZE
        self.value_output = _FixedScratchSlot(
            _chunk_scratch(
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
def _gradient_slot(metadata, task, chunk, staged: Constexpr[bool]):
    """Per-core persistent storage or a disjoint global task/chunk slot."""
    slot = get_block_idx()
    if const_expr(staged):
        record = HEADER_WORDS + task * TASK_WORDS
        slot = metadata[record + TASK_FLAT_CHUNK_START] + task + chunk
    return slot
