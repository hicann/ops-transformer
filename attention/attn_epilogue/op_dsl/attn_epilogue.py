# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fused local attention output projection for NG=8/2/1 and ND/NZ weights."""

from dataclasses import dataclass
from functools import lru_cache
from math import gcd
import inspect

import cannbotdsl as dsl
from cannbotdsl import Buffer, Constexpr, RegLayout, dtypes, make_layout
from cannbotdsl.aot import compile_cache, export
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops import reg
from cannbotdsl.ops.arch import (
    get_block_idx,
    get_block_num,
    get_ctrl,
    get_subblock_dim,
    get_subblock_id,
    set_ctrl,
)
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl.ops.reg.cast import RoundingMode
from cannbotdsl.ops.sync import (
    PIPE,
    global_sync_all,
    cube_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
    cube_sync_block_wait,
)
from cannbotdsl.tensor import MemLoc, tile_slice


NG = 8
F = 4096
O_LORA = 1024
DIM = 5120

BASE_K = 128

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
NOPE_CHUNKS = NOPE_DIM // 64
QUANT_GROUP = 32
MX_DIVISOR = 64
FP8 = dtypes.float8_e4m3fn
E8M0 = dtypes.float8_e8m0
_L1_NZ_STRIPE_BYTES = 32768

# Cube tiling granularity and legal tile bounds (elements, not bytes).
BASE_M_ALIGN = 16
BASE_N_ALIGN = 32
MAX_BASE_M = 256
MAX_BASE_N = 256
BF16_BYTES = 2
FP32_BYTES = 4
L0C_DEPTH_MAX = 2
# Scale/bias row+column overhead used by the tiling cost model (bytes).
TILE_OVERHEAD_BYTES = 256

# Read the target architecture capacities once; tiling and storage views must
# share the same limits. These constants do not query the device on each call.
_MEMORY_BYTES = {
    name: int(dsl.get_mem_size(name)) for name in ("l1", "l0a", "l0b", "l0c", "ub")
}
L1_BYTES, L0A_BYTES, L0B_BYTES, L0C_BYTES, UB_BYTES = (
    int(_MEMORY_BYTES[name]) for name in ("l1", "l0a", "l0b", "l0c", "ub")
)
if min(L1_BYTES, L0A_BYTES, L0B_BYTES, L0C_BYTES, UB_BYTES) <= 0:
    raise RuntimeError("target architecture reports invalid buffer capacities")

# DSL 0.8 CI wheels and layout-based builds expose different constructor
# conventions. Resolve this once at import; staged branches are compile-time.
_RING_BUFFER_USES_LAYOUT = "layout" in inspect.signature(RingBuffer).parameters
_BUFFER_POSITIONAL_PARAMETERS = tuple(
    name
    for name, parameter in inspect.signature(Buffer).parameters.items()
    if parameter.kind
    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
)
_BUFFER_USES_LAYOUT = _BUFFER_POSITIONAL_PARAMETERS[1:3] == ("dtype", "layout")
# cannbotdsl 0.8.3 replaced the top-level reinterpret helper with from_tensor.
_REINTERPRET = getattr(dsl, "reinterpret", None) or dsl.from_tensor
_REINTERPRET_USES_LAYOUT = "layout" in inspect.signature(_REINTERPRET).parameters


@jit
def matrix_view(tensor, row_begin, row_end, col_begin, col_end):
    """View a matrix tile using exclusive row and column bounds."""
    return tensor[row_begin:row_end, col_begin:col_end]


@jit
def scale_view(tensor, row_begin, row_end, group_begin, group_end):
    """View packed MX scales while retaining the two scale bytes per group."""
    if const_expr(len(tensor.shape) == 3):
        return tensor[row_begin:row_end, group_begin:group_end, 0:2].reshape(
            row_end - row_begin, (group_end - group_begin) * 2
        )
    return tensor[row_begin:row_end, group_begin * 2 : group_end * 2]


def _ring_buffer(location, shape, dtype, *, depth, data_format=None):
    """Declare typed slots using the installed DSL 0.8 constructor convention."""
    if _RING_BUFFER_USES_LAYOUT:
        return RingBuffer(
            location, dtype, make_layout(shape), depth=depth, data_format=data_format
        )
    return RingBuffer(location, shape, dtype, depth=depth, data_format=data_format)


def _buffer(location, shape, dtype):
    """Allocate bounded vector scratch using the installed constructor convention."""
    if _BUFFER_USES_LAYOUT:
        return Buffer(location, dtype, make_layout(shape))
    return Buffer(location, shape, dtype)


@dataclass(frozen=True)
class ProjectionShape:
    """Logical sequence, group and projection dimensions."""

    t: int
    ng: int
    first_nz: bool = False
    second_nz: bool = False


@dataclass(frozen=True)
class ProjectionTiles:
    """Shared M tile and the N tile of each projection."""

    rows: int
    first_n: int
    second_n: int
    step: int = 0


@dataclass(frozen=True)
class RopeInputs:
    """BF16 activations and the sine/cosine rows applied to their head tails."""

    source: object
    cos: object
    sin: object


@dataclass(frozen=True)
class QuantOutput:
    """Quantized bytes and their per-group scales."""

    data: object
    scales: object


@dataclass(frozen=True)
class TileExtent:
    """Logical output origin and the valid tile extent."""

    row: object
    col: object
    rows: object
    cols: object


@dataclass(frozen=True)
class RopeBatch:
    """Rows and K slab processed by one vector lane."""

    row: object
    group: object
    chunk: object
    count: object


@dataclass(frozen=True)
class QuantBuffers:
    """Input, quantized output, scales and temporary vector storage."""

    source: object
    output: object
    scales: object
    reciprocal: object
    even: object
    odd: object


@jit
def _quant_row(buffers, num_col):
    """Quantize BF16 blocks into MXFP8 payload and one E8M0 scale per 32 values."""
    ch = num_col // 256
    with vf(mode="simd"):
        m16 = reg.create_mask("all", elem_bits=16)
        fm32 = reg.create_mask("all", elem_bits=32)
        vl16b = reg.create_mask("vl16", elem_bits=8)
        vl16u = reg.create_mask("vl16", elem_bits=16)
        _quant_scales(buffers, ch, m16, vl16b, vl16u)
        reg.vmem_bar(mode="vst_vld")
        _quant_values(buffers, ch, m16, fm32)
        reg.vmem_bar(mode="vst_vld")
        _quant_interleave(buffers, ch)


@jit
def _quant_scales(buffers, ch, m16, vl16b, vl16u):
    """Reduce BF16 exponents once to derive packed scales and reciprocal factors."""
    em = reg.vdups(0x7F80, dtypes.uint16, mask=m16)
    c8 = reg.vdups(8, dtypes.uint16, mask=m16)
    c262 = reg.vdups(262, dtypes.uint16, mask=m16)
    idx = reg.varange(0, dtypes.uint8)
    one = reg.vdups(1, dtypes.uint8, mask=vl16b)
    even_mask = reg.veqs(reg.vbitwise_and(idx, one, mask=vl16b), 0, mask=vl16b)
    # Scale bytes and reciprocals share the same group exponent reduction.
    ureg_s = reg.vsqueeze_and_storeunalign_init(buffers.scales)
    for chunk in dsl_range(0, ch, 1):
        x0, x1 = reg.vload_deinterleave(buffers.source, chunk * 256)
        a0 = reg.vbitwise_and(reg.vreinterpret(x0, dtypes.uint16), em, mask=m16)
        a1 = reg.vbitwise_and(reg.vreinterpret(x1, dtypes.uint16), em, mask=m16)
        m = reg.vmax(a0, a1, mask=m16)
        gm = reg.vreduce_max_datablock(m, mask=m16)
        e = reg.vmaxs(reg.vshr(gm, 7, mask=m16), 8, mask=m16)
        sb = reg.vsub(e, c8, mask=m16)
        sb8 = reg.vcast(sb, dtypes.uint8, mask=m16, reg_layout=RegLayout.ZERO)
        reg.vsqueeze_and_storeunalign(buffers.scales, 0, sb8, ureg_s, mask=even_mask)
        rev = reg.vshl(reg.vsub(c262, e, mask=m16), 7, mask=m16)
        rev = reg.vselect(
            reg.vdups(0, dtypes.uint16, mask=m16),
            rev,
            cond_mask=reg.veqs(e, 8, mask=m16),
        )
        dup, _ = reg.vinterleave(rev, rev)
        reg.vstore(buffers.reciprocal, chunk * 16, dup, vl16u)
    reg.vsqueeze_and_storeunalign_finalize(buffers.scales, 0, ureg_s)


@jit
def _quant_values(buffers, ch, m16, fm32):
    """Apply group reciprocals and saturating E4M3 casts while preserving lane order."""
    for chunk in dsl_range(0, ch, 1):
        for half in range(2):
            off = chunk * 256 + half * 128
            xv = reg.vload(buffers.source, off)
            rs = reg.vload_broadcast(
                buffers.reciprocal, chunk * 16 + half * 8, mode="datablock"
            )
            rbf = reg.vreinterpret(rs, dtypes.bfloat16)
            y = reg.vmul(xv, rbf, mask=m16)
            yz = reg.vcast(y, dtypes.float32, mask=m16, reg_layout=RegLayout.ZERO)
            yo = reg.vcast(y, dtypes.float32, mask=m16, reg_layout=RegLayout.ONE)
            qz = reg.vcast(
                yz,
                dtypes.float8_e4m3fn,
                mask=fm32,
                reg_layout=RegLayout.ZERO,
                saturate=True,
            )
            qo = reg.vcast(
                yo,
                dtypes.float8_e4m3fn,
                mask=fm32,
                reg_layout=RegLayout.ZERO,
                saturate=True,
            )
            reg.vstore_pack(
                buffers.even,
                chunk * 128 + half * 64,
                qz,
                fm32,
                pack_mode=reg.PackMode.B32_TO_B8,
            )
            reg.vstore_pack(
                buffers.odd,
                chunk * 128 + half * 64,
                qo,
                fm32,
                pack_mode=reg.PackMode.B32_TO_B8,
            )


@jit
def _quant_interleave(buffers, ch):
    """Interleave even/odd FP8 lanes into the byte layout used by Cube loads."""
    for chunk in dsl_range(0, ch, 1):
        for half in range(2):
            off = chunk * 256 + half * 128
            qz8 = reg.vload(buffers.even, chunk * 128 + half * 64)
            qo8 = reg.vload(buffers.odd, chunk * 128 + half * 64)
            reg.vstore_interleave(buffers.output, off, qz8, qo8, width="b8")


def _operand_channel(
    storage,
    location,
    shape,
    dtype,
    depth,
    *,
    offset=0,
    data_format=None,
    element_bytes=1,
    padding=0,
):
    """Build an owned channel or identically laid out views of shared storage."""
    if storage is None:
        return _ring_buffer(
            location, shape, dtype, depth=depth, data_format=data_format
        )
    buffers = []
    stride = shape[0] * shape[1] * element_bytes + padding
    for slot in range(depth):
        buffers.append(
            _REINTERPRET(
                storage,
                dtype=dtype,
                **(
                    {"layout": make_layout(shape)}
                    if _REINTERPRET_USES_LAYOUT
                    else {"shape": shape}
                ),
                offset=offset + slot * stride,
                data_format=data_format or "nz",
            )
        )
    return dsl.make_ring_buffer(buffers)


@dataclass(frozen=True)
class MatmulShape:
    """Grouped matrix dimensions including the padded input row extent."""

    groups: object
    n: object
    k: object
    m: object


@dataclass(frozen=True)
class MxOperands:
    """Projection operands and their packed MX scales."""

    a: object
    b: object
    a_scale: object
    b_scale: object


@dataclass(frozen=True)
class TileIndex:
    """Output tile coordinates and the state of its weight prefetch."""

    m: object
    n: object
    k: object
    prefetched: object = 0


@dataclass(frozen=True)
class MxAccumulator:
    """Accumulator, B scales and live output extents for one output tile."""

    output: object
    b_scale: object
    m: object
    n: object


@dataclass(frozen=True)
class KTileIndex:
    """K tile positions in the projection and in the current scale slab."""

    global_tile: object
    local_tile: object
    k: object
    prefetched: object


@dataclass(frozen=True)
class L1Operands:
    """Live L1 data and A scales consumed by one Cube stage."""

    a: object
    b: object
    a_scale: object


class MatmulMx:
    """One (base_m, base_n) output tile of an MXFP8 GEMM, K streamed in base_k."""

    def __init__(
        self,
        k,
        base_m,
        base_n,
        base_k=BASE_K,
        step_k=1,
        reuse_n=0,
        l0c_depth=2,
        prefetch_tiles=1,
        prefetch_scales=False,
        stream_k=0,
        weight_nz=False,
    ):
        """Bind double-buffered Cube operands, MX scales, and the selected ND/NZ copy engines."""
        self.weight_nz = bool(weight_nz)
        self.n_axis = 1 if self.weight_nz else 0
        self.k = int(k)
        self.stream_k = int(stream_k)
        self.scale_k = self.stream_k or self.k
        self.base_m = int(base_m)
        self.base_n = int(base_n)
        self.base_k = int(base_k)
        self.step_k = int(step_k)  # base_k blocks staged per L1 slot (native stepKb)
        self.k_l1 = self.base_k * self.step_k
        self.prefetch_scales = bool(prefetch_scales)
        self.prefetch_tiles = min(int(prefetch_tiles), 2, self.k // self.k_l1)
        self.sk_l0 = self.base_k // 32  # packed scale length per base_k
        self.sk_full = (self.k // 64) * 2  # packed scale length for the full K
        # Reuse Cube storage only after the previous Cube phase has drained.
        reuse_n = int(reuse_n)
        if reuse_n and reuse_n < self.base_n:
            raise ValueError("reuse_n must cover the physical N tile")
        self._init_l1_channels(reuse_n)
        self._init_l0_channels(reuse_n, l0c_depth)

    @jit
    def prefetch(self, b, n0, sb=None):
        """Stage the first weight tiles and scales while Vector prepares activations."""
        for kk in range(self.prefetch_tiles):
            mem_copy(
                self.l1_b.next_write(),
                self._weight_tile(
                    b,
                    n0 * self.base_n,
                    min((n0 + 1) * self.base_n, b.shape[self.n_axis]),
                    kk * self.k_l1,
                    (kk + 1) * self.k_l1,
                ),
            )
        if const_expr(self.prefetch_scales):
            mem_copy(
                self.l1_sb.next_write(),
                self._weight_scale_tile(
                    sb,
                    n0 * self.base_n,
                    min((n0 + 1) * self.base_n, b.shape[self.n_axis]),
                    0,
                    b.shape[1 - self.n_axis] // MX_DIVISOR,
                ),
            )

    @jit
    def compute(self, operands, index):
        """Accumulate one valid output tile in FP32 without intermediate BF16 rounding."""
        m_end = min((index.m + 1) * self.base_m, operands.a.shape[0])
        n_end = min((index.n + 1) * self.base_n, operands.b.shape[self.n_axis])
        actual_m = m_end - index.m * self.base_m
        actual_n = n_end - index.n * self.base_n
        b_scales = self._load_scales(operands, index, actual_n)
        accumulator = self.l0c.next_write()
        state = MxAccumulator(accumulator, b_scales, actual_m, actual_n)
        scale_k = self.stream_k or index.k
        for chunk in range(index.k // scale_k):
            self._compute_chunk(operands, index, state, chunk)
        return accumulator

    @jit
    def tile(self, operands, out, index):
        """Compute an MM2 tile and write its valid rows to the BF16 output through Fixpipe."""
        b = operands.b
        m0, n0 = index.m, index.n
        accumulator = self.compute(operands, index)
        mem_copy(
            matrix_view(
                out,
                m0 * self.base_m,
                min((m0 + 1) * self.base_m, operands.a.shape[0]),
                n0 * self.base_n,
                min((n0 + 1) * self.base_n, b.shape[self.n_axis]),
            ),
            accumulator,
            options=copy_options(unit_flag=3),
        )

    @jit
    def _load_scales(self, operands, index, actual_n):
        """Load matching A/B scale tiles, reusing prefetched B scales for the first tile."""
        g = index.k // MX_DIVISOR
        m_begin = index.m * self.base_m
        n_begin = index.n * self.base_n
        m_end = min(m_begin + self.base_m, operands.a_scale.shape[0])
        n_end = min(n_begin + self.base_n, operands.b.shape[self.n_axis])
        if const_expr(not self.stream_k):
            if const_expr(_RING_BUFFER_USES_LAYOUT):
                scale_dst = self.l1_sa.next_write().permute(1, 0)
            else:
                scale_dst = self.l1_sa.next_write()
            mem_copy(
                scale_dst,
                scale_view(operands.a_scale, m_begin, m_end, 0, g),
            )
        if not self.prefetch_scales or index.prefetched == 0:
            mem_copy(
                self.l1_sb.next_write(),
                self._weight_scale_tile(operands.b_scale, n_begin, n_end, 0, g),
            )
        return self.l1_sb.next_read()[0 : 2 * g, 0 : self.base_n]

    @jit
    def _compute_chunk(self, operands, index, state, chunk):
        """Wait for a ready activation slab and consume all its L1 K windows."""
        scale_k = self.stream_k or index.k
        if const_expr(self.stream_k):
            if index.prefetched != 0:
                cube_sync_block_wait(PIPE.MTE2, 3 + chunk)
        if const_expr(self.stream_k):
            m_begin = index.m * self.base_m
            m_end = min(m_begin + self.base_m, operands.a_scale.shape[0])
            g_begin = chunk * (scale_k // 64)
            g_end = (chunk + 1) * (scale_k // 64)
            if const_expr(_RING_BUFFER_USES_LAYOUT):
                scale_dst = self.l1_sa.next_write().permute(1, 0)
            else:
                scale_dst = self.l1_sa.next_write()
            mem_copy(
                scale_dst,
                scale_view(operands.a_scale, m_begin, m_end, g_begin, g_end),
            )
        if const_expr(_RING_BUFFER_USES_LAYOUT):
            a_scales = self.l1_sa.next_read().permute(1, 0)[
                0 : self.base_m, 0 : scale_k // 32
            ]
        else:
            a_scales = self.l1_sa.next_read()[0 : self.base_m, 0 : scale_k // 32]
        for local_kk in range(scale_k // self.k_l1):
            kk = chunk * (scale_k // self.k_l1) + local_kk
            position = KTileIndex(kk, local_kk, index.k, index.prefetched)
            self._compute_l1(operands, index, state, a_scales, position)

    @jit
    def _compute_l1(self, operands, index, state, a_scales, position):
        """Stage one A/B K window in L1 and pass its scales to the L0 pipeline."""
        m_begin = index.m * self.base_m
        n_begin = index.n * self.base_n
        m_end = min(m_begin + self.base_m, operands.a.shape[0])
        n_end = min(n_begin + self.base_n, operands.b.shape[self.n_axis])
        k_begin = position.global_tile * self.k_l1
        k_end = k_begin + self.k_l1
        mem_copy(
            self.l1_a.next_write(),
            matrix_view(operands.a, m_begin, m_end, k_begin, k_end),
        )
        if position.global_tile >= self.prefetch_tiles or position.prefetched == 0:
            mem_copy(
                self.l1_b.next_write(),
                self._weight_tile(operands.b, n_begin, n_end, k_begin, k_end),
            )
        a_tile = self.l1_a.next_read()
        b_tile = self.l1_b.next_read()
        tiles = L1Operands(a_tile, b_tile, a_scales)
        self._compute_l0(tiles, state, position)

    @jit
    def _compute_l0(self, tiles, state, position):
        """Feed base-K operands and MX scales to MMA; transpose private NZ B at L0 load."""
        kk, local_kk = position.global_tile, position.local_tile
        steps = position.k // self.k_l1
        for j in range(self.step_k):
            kstep = kk * self.step_k + j
            a_operand = self.l0a.next_write()
            b_operand = self.l0b.next_write()
            a_slice = tile_slice(tiles.a, (self.base_m, self.base_k), (0, j))
            if const_expr(self.weight_nz):
                b_slice = tile_slice(
                    tiles.b, (self.base_k, self.base_n), (j, 0)
                ).permute(1, 0)
            else:
                b_slice = tile_slice(tiles.b, (self.base_n, self.base_k), (0, j))
            a_scale = tile_slice(
                tiles.a_scale,
                (self.base_m, self.sk_l0),
                (0, local_kk * self.step_k + j),
            )
            b_scale = tile_slice(state.b_scale, (self.sk_l0, self.base_n), (kstep, 0))
            mem_copy(a_operand, a_slice, options=copy_options(mx_scale=a_scale))
            mem_copy(b_operand, b_slice, options=copy_options(mx_scale=b_scale))
            matmul(
                state.output,
                a_operand,
                b_operand,
                init=(kk == 0 and j == 0),
                unit_flag=3 if (kk == steps - 1 and j == self.step_k - 1) else 2,
                mx_mode=True,
            )

    def _init_l1_channels(self, reuse_n):
        """Lay out two operand slots and scale storage inside the reserved L1 capacity."""
        storage = dsl.L1.view(L1_BYTES) if reuse_n else None
        a_shape = (self.base_m, self.k_l1)
        b_shape = (
            (self.k_l1, self.base_n) if self.weight_nz else (self.base_n, self.k_l1)
        )
        if _RING_BUFFER_USES_LAYOUT:
            sa_shape, sa_format = (self.scale_k // 32, self.base_m), "nz"
        else:
            sa_shape, sa_format = (self.base_m, self.scale_k // 32), "zn"
        sb_shape = (self.sk_full, self.base_n)
        # The 0.8 NZ GM->L1 raw copy zero-fills one 32 KiB stripe past the
        # destination view. Reserve that stripe per slot so the fill cannot
        # overwrite the adjacent stage's scales.
        b_padding = _L1_NZ_STRIPE_BYTES if self.weight_nz and reuse_n else 0
        b_addr = self.base_m * self.k_l1 * 2
        sa_addr = (self.base_m + reuse_n) * self.k_l1 * 2 + 2 * b_padding
        sb_addr = sa_addr + self.base_m * self.sk_full
        self.l1_a = _operand_channel(
            storage, MemLoc.L1, a_shape, FP8, 2, data_format="nz"
        )
        self.l1_b = _operand_channel(
            storage,
            MemLoc.L1,
            b_shape,
            FP8,
            2,
            offset=b_addr,
            data_format="nz",
            padding=b_padding,
        )
        self.l1_sa = _operand_channel(
            storage, MemLoc.L1, sa_shape, E8M0, 1, offset=sa_addr, data_format=sa_format
        )
        self.l1_sb = _operand_channel(
            storage, MemLoc.L1, sb_shape, E8M0, 1, offset=sb_addr, data_format="nz"
        )

    def _init_l0_channels(self, reuse_n, depth):
        """Bind double-buffered L0A/L0B and one or two FP32 accumulator slots."""
        a_storage = dsl.L0A.view(L0A_BYTES) if reuse_n else None
        b_storage = dsl.L0B.view(L0B_BYTES) if reuse_n else None
        c_storage = dsl.L0C.view(L0C_BYTES) if reuse_n else None
        a_shape = (self.base_m, self.base_k)
        b_shape = (self.base_n, self.base_k)
        c_shape = (self.base_m, self.base_n)
        self.l0a = _operand_channel(a_storage, MemLoc.L0A, a_shape, FP8, 2)
        self.l0b = _operand_channel(b_storage, MemLoc.L0B, b_shape, FP8, 2)
        self.l0c = _operand_channel(
            c_storage, MemLoc.L0C, c_shape, dtypes.float32, depth, element_bytes=4
        )

    @jit
    def _weight_tile(self, tensor, n_begin, n_end, k_begin, k_end):
        """Slice private NZ weights by physical tiles or ND weights by matrix bounds."""
        if const_expr(self.weight_nz):
            return tile_slice(
                tensor,
                (self.k_l1, self.base_n),
                (k_begin // self.k_l1, n_begin // self.base_n),
            )
        else:
            return matrix_view(tensor, n_begin, n_end, k_begin, k_end)

    @jit
    def _weight_scale_tile(self, tensor, n_begin, n_end, g_begin, g_end):
        """Select ND scale axes matching the weight logical (K,N) or (N,K) order."""
        if const_expr(self.weight_nz):
            return (
                tensor[g_begin:g_end, n_begin:n_end, 0:2]
                .permute(0, 2, 1)
                .reshape((g_end - g_begin) * 2, n_end - n_begin)
            )
        else:
            return scale_view(tensor, n_begin, n_end, g_begin, g_end).permute(1, 0)


class QuantTile:
    """Quantize a full Cube tile using bounded Vec temporary storage."""

    def __init__(self, base_m, base_n, reserved_ub=65536):
        """Fit a full Fixpipe receive tile and reusable quantization chunks within UB."""
        self.rows = int(base_m) // 2
        self.cols = int(base_n)
        # The caller reserves RoPE/QuantA storage, including input buffers.
        # The full FP32 tile stays live while bounded scratch is reused.
        choices = []
        row_alignment = 256 // gcd(self.cols, 256)
        row_limit = (self.rows + row_alignment - 1) // row_alignment * row_alignment
        for rows in range(1, row_limit + 1):
            count = rows * self.cols
            if count % 256:
                continue
            chunks = (self.rows + rows - 1) // rows
            padded = chunks * rows
            # Scratch per element: BF16 (2), packed FP8 (2), two byte
            # temporaries (1), reciprocal (1/8), and MX scale (1/32).
            # Padded scales need 32 bytes per row; leave alignment headroom.
            ub_bytes = (
                reserved_ub
                + 4 * padded * self.cols
                + count * 165 // 32
                + rows * 32
                + 512
            )
            if ub_bytes <= UB_BYTES:
                choices.append((chunks, padded - self.rows, -rows))
        if not choices:
            raise ValueError("no quant chunk fits UB")
        self.chunks, _, negative_rows = min(choices)
        self.chunk_rows = -negative_rows
        self.padded_rows = self.chunks * self.chunk_rows
        self.count = self.chunk_rows * self.cols
        count = self.count
        self.x = _ring_buffer(
            MemLoc.UB,
            (self.padded_rows, self.cols),
            dtypes.float32,
            depth=1,
        )
        self.bf = _buffer(MemLoc.UB, (1, count), dtypes.bfloat16)
        self.q = _buffer(MemLoc.UB, (1, 2 * count), dtypes.uint8)
        self.s = _buffer(MemLoc.UB, (1, count // 32 + 256), dtypes.uint8)
        self.spad = _buffer(MemLoc.UB, (self.chunk_rows, 32), dtypes.uint8)
        self.recip = _buffer(MemLoc.UB, (count // 16,), dtypes.uint16)
        self.qz = _buffer(MemLoc.UB, (count // 2,), dtypes.uint8)
        self.qo = _buffer(MemLoc.UB, (count // 2,), dtypes.uint8)

    @jit
    def run(self, accumulator, output, tile):
        """Receive half of an MM1 tile per AIV and quantize only its active rows."""
        actual_m = tile.rows
        if const_expr(self.padded_rows == self.rows):
            mem_copy(
                self.x.next_write(),
                accumulator.partition(dim=0),
                options=copy_options(unit_flag=3),
            )
        else:
            mem_copy(
                matrix_view(self.x.next_write(), 0, self.rows, 0, self.cols),
                accumulator.partition(dim=0),
                options=copy_options(unit_flag=3),
            )
        src = self.x.next_read()
        valid_rows = max(0, min(self.rows, actual_m - get_subblock_id() * self.rows))
        if const_expr(self.chunks == 1):
            self._chunk(src, output, tile, 0, valid_rows)
        else:
            for rb in dsl_range(0, max(valid_rows, 1), self.chunk_rows):
                valid = max(0, min(self.chunk_rows, valid_rows - rb))
                self._chunk(src, output, tile, rb, valid)

    @jit
    def _chunk(self, src, output, tile, rb, valid):
        """Round FP32 to BF16, quantize, and store valid FP8 payload/scale byte pairs."""
        gm_q, gm_s = output.data, output.scales
        row, col = tile.row, tile.col
        actual_n = tile.cols
        # An idle AIV still reads its received slot to complete RingBuffer
        # ownership; only nonempty row windows are written back to GM.
        active_count = (max(valid, 1) * self.cols + 255) // 256 * 256
        with vf(mode="simd"):
            mask = reg.full_mask()
            for offset in dsl_range(0, active_count, 64):
                value = reg.vload(src, rb * self.cols + offset)
                bf = reg.vcast(
                    value, dtypes.bfloat16, mask=mask, rounding=RoundingMode.RN
                )
                reg.vstore_pack(
                    self.bf, offset, bf, mask, pack_mode=reg.PackMode.B32_TO_B16
                )
        _quant_row(
            QuantBuffers(self.bf, self.q, self.s, self.recip, self.qz, self.qo),
            active_count,
        )
        with vf(mode="simd"):
            smask, _ = reg.update_mask(self.cols // 32, elem_bits=8)
            for row_index in dsl_range(0, valid, 1):
                offset = row_index * (self.cols // 32)
                reg.vload_unalign_init(self.s, offset)
                sv = reg.vload_unalign(self.s, offset)
                reg.vstore(self.spad, row_index * 32, sv, smask)
        begin = row + get_subblock_id() * self.rows + rb
        q = self.q[0:1, 0 : self.count].reshape(self.chunk_rows, self.cols)
        if valid > 0:
            mem_copy(
                matrix_view(gm_q, begin, begin + valid, col, col + actual_n),
                q[0:valid, 0:actual_n],
            )
            mem_copy(
                matrix_view(
                    gm_s, begin, begin + valid, col // 32, (col + actual_n) // 32
                ),
                matrix_view(self.spad, 0, valid, 0, actual_n // 32),
            )


class RopeQuant:
    """inverse RoPE + OCP MX quant for T tokens -> aq (g*T+t, F) / asc."""

    def __init__(self, ng=NG, f=F, stream_k=0, batch_rows=1):
        """Allocate bounded RoPE/quantization buffers and optional streaming input slots."""
        self.ng = int(ng)
        self.rows = int(batch_rows)
        self.full_f = int(f)
        self.stream_k = int(stream_k)
        self.f = self.stream_k or self.full_f
        self.num_heads = self.f // HEAD_DIM
        self.ngroups = self.f // QUANT_GROUP
        self.count = self.rows * self.f
        self.input_depth = 2 if self.stream_k else 1
        self.extra_ub = (self.input_depth - 1) * (
            2 * self.count + 8 * self.rows * ROPE_DIM
        )
        self.in_ch = _ring_buffer(
            MemLoc.UB,
            shape=(self.rows, self.f),
            dtype=dtypes.bfloat16,
            depth=self.input_depth,
        )
        self.out_ch = _ring_buffer(
            MemLoc.UB, shape=(self.rows, self.f), dtype=dtypes.bfloat16, depth=1
        )
        # q_ch is 2x: reg.vstore_interleave(b8) writes ~2x the logical bytes.
        self.q_ch = _ring_buffer(
            MemLoc.UB, shape=(1, 2 * self.count), dtype=dtypes.uint8, depth=1
        )
        self.s_ch = _ring_buffer(
            MemLoc.UB, shape=(self.rows, self.ngroups), dtype=dtypes.uint8, depth=1
        )
        self.cos_ch = _ring_buffer(
            MemLoc.UB,
            shape=(self.rows, ROPE_DIM),
            dtype=dtypes.float32,
            depth=self.input_depth,
        )
        self.sin_ch = _ring_buffer(
            MemLoc.UB,
            shape=(self.rows, ROPE_DIM),
            dtype=dtypes.float32,
            depth=self.input_depth,
        )
        self.rope_buf = _buffer(
            MemLoc.UB, (self.rows * self.num_heads, ROPE_DIM), dtypes.float32
        )
        nchunk = self.count // 256
        self.recip_f = _buffer(MemLoc.UB, (nchunk * 16,), dtypes.uint16)
        self.scr_qz = _buffer(MemLoc.UB, (nchunk * 128,), dtypes.uint8)
        self.scr_qo = _buffer(MemLoc.UB, (nchunk * 128,), dtypes.uint8)

    @jit
    def run(self, inputs, output, num_token, m_pad):
        """Choose complete-input production or the two-slab QuantA/MM1 pipeline."""
        if const_expr(self.stream_k):
            self._run_stream(inputs, output, num_token, m_pad)
        else:
            self._run_full(inputs, output, num_token, m_pad)

    @jit
    def _run_full(self, inputs, output, num_token, m_pad):
        """Distribute token/group rows across AIVs and write group-major quantized activations."""
        gm_x, gm_cos, gm_sin = inputs.source, inputs.cos, inputs.sin
        gm_aq, gm_asc = output.data, output.scales
        ng = self.ng
        f = self.f
        ngroups = self.ngroups
        aiv = get_block_idx() * get_subblock_dim() + get_subblock_id()
        naiv = get_block_num() * get_subblock_dim()
        total = num_token * ng
        tpb = (total + naiv - 1) // naiv
        for it in dsl_range(0, tpb, 1):
            u = aiv + it * naiv
            if u < total:
                t = u // ng
                g = u % ng
                mem_copy(
                    self.in_ch.next_write(), tile_slice(gm_x, (1, f), (t * ng + g, 0))
                )
                mem_copy(
                    self.cos_ch.next_write(), tile_slice(gm_cos, (1, ROPE_DIM), (t, 0))
                )
                mem_copy(
                    self.sin_ch.next_write(), tile_slice(gm_sin, (1, ROPE_DIM), (t, 0))
                )
                self.apply_inverse_rope()
                self.quant_row_fast()
                mem_copy(
                    tile_slice(gm_aq, (1, f), (g * m_pad + t, 0)),
                    tile_slice(self.q_ch.next_read(), (1, f), (0, 0)),
                )
                mem_copy(
                    tile_slice(gm_asc, (1, ngroups), (g * m_pad + t, 0)),
                    tile_slice(self.s_ch.next_read(), (1, ngroups), (0, 0)),
                )

    @jit
    def _run_stream(self, inputs, output, num_token, m_pad):
        """Produce K slabs and publish readiness only after all Vector stores complete."""
        aiv = get_block_idx() * get_subblock_dim() + get_subblock_id()
        naiv = get_block_num() * get_subblock_dim()
        lanes = naiv // self.ng
        group, lane = aiv % self.ng, aiv // self.ng
        source = inputs.source.view(num_token, self.ng * self.full_f)
        views = RopeInputs(source, inputs.cos, inputs.sin)
        for chunk in range(self.full_f // self.f):
            for t in dsl_range(lane * self.rows, num_token, lanes * self.rows):
                actual = min(self.rows, num_token - t)
                batch = RopeBatch(t, group, chunk, actual)
                self._stream_batch(views, output, batch, m_pad)
            # Publish immutable slab data only after all vector stores complete.
            vec_sync_block_arrive(PIPE.MTE3, 11 + chunk, mode=0)
            vec_sync_block_wait(PIPE.MTE3, 11 + chunk, mode=0)
            vec_sync_block_arrive(PIPE.MTE3, 3 + chunk)

    @jit
    def _stream_batch(self, inputs, output, batch, m_pad):
        """Load a bounded token batch, apply inverse RoPE, and quantize the current K slab."""
        t, actual, f = batch.row, batch.count, self.f
        row_end = t + actual
        col_begin = batch.group * self.full_f + batch.chunk * f
        col_end = col_begin + f
        mem_copy(
            self.in_ch.next_write()[0:actual, 0:f],
            matrix_view(inputs.source, t, row_end, col_begin, col_end),
        )
        mem_copy(
            self.cos_ch.next_write()[0:actual, 0:ROPE_DIM],
            matrix_view(inputs.cos, t, row_end, 0, ROPE_DIM),
        )
        mem_copy(
            self.sin_ch.next_write()[0:actual, 0:ROPE_DIM],
            matrix_view(inputs.sin, t, row_end, 0, ROPE_DIM),
        )
        self.apply_inverse_rope()
        self.quant_row_fast()
        self._store_stream(output, batch, m_pad)

    @jit
    def _store_stream(self, output, batch, m_pad):
        """Store valid slab payload/scales at their group-major GM addresses."""
        actual, f, ngroups = batch.count, self.f, self.ngroups
        begin = batch.group * m_pad + batch.row
        end = begin + actual
        q_begin, q_end = batch.chunk * f, (batch.chunk + 1) * f
        s_begin = batch.chunk * ngroups
        s_end = (batch.chunk + 1) * ngroups
        q = self.q_ch.next_read()[0:1, 0 : self.count].reshape(self.rows, f)
        mem_copy(matrix_view(output.data, begin, end, q_begin, q_end), q[0:actual, 0:f])
        mem_copy(
            matrix_view(output.scales, begin, end, s_begin, s_end),
            self.s_ch.next_read()[0:actual, 0:ngroups],
        )

    @jit
    def quant_row_fast(self):
        """Quantize the BF16 RoPE result using shared scale/reciprocal reductions."""
        _quant_row(
            QuantBuffers(
                self.out_ch.next_read(),
                self.q_ch.next_write(),
                self.s_ch.next_write(),
                self.recip_f,
                self.scr_qz,
                self.scr_qo,
            ),
            self.count,
        )

    @jit
    def apply_inverse_rope(self):
        """Preserve non-rotary BF16 lanes and rotate each head tail using FP32 arithmetic."""
        num_heads = self.num_heads
        src = self.in_ch.next_read()
        cos = self.cos_ch.next_read()
        sin = self.sin_ch.next_read()
        dst = self.out_ch.next_write()
        with vf(mode="simd"):
            full = reg.full_mask()
            if const_expr(self.stream_k):
                # Copy BF16 directly, then overwrite the rotated head tails.
                # The following VST->VLD fence also completes these stores.
                copy_mask = reg.create_mask("all", elem_bits=16)
                for offset in dsl_range(0, self.count, 128):
                    reg.vstore(dst, offset, reg.vload(src, offset), copy_mask)
            for h in dsl_range(0, self.rows * num_heads, 1):
                xu = reg.vload_unpack(
                    src, h * HEAD_DIM + NOPE_DIM, unpack_mode=reg.UnpackMode.B16_TO_B32
                )
                reg.vstore(
                    self.rope_buf,
                    h * ROPE_DIM,
                    reg.vcast(xu, dtypes.float32, mask=full),
                    full,
                )
            reg.vmem_bar(mode="vst_vld")
            cos_lanes, _ = reg.vload_deinterleave(cos, 0)
            sin_lanes, _ = reg.vload_deinterleave(sin, 0)
            for h in dsl_range(0, self.rows * num_heads, 1):
                if const_expr(self.stream_k):
                    cos_lanes, _ = reg.vload_deinterleave(
                        cos, (h // num_heads) * ROPE_DIM
                    )
                    sin_lanes, _ = reg.vload_deinterleave(
                        sin, (h // num_heads) * ROPE_DIM
                    )
                base = h * HEAD_DIM
                if const_expr(not self.stream_k):
                    self._copy_nope(src, dst, base, full)
                self._rotate_tail(dst, h, cos_lanes, sin_lanes, full)

    @jit
    def _copy_nope(self, src, dst, base, full):
        """Copy the first 448 head dimensions without changing their BF16 bit patterns."""
        for c in dsl_range(0, NOPE_CHUNKS, 1):
            reg.vstore_pack(
                dst,
                base + c * 64,
                reg.vload_unpack(
                    src,
                    base + c * 64,
                    unpack_mode=reg.UnpackMode.B16_TO_B32,
                ),
                full,
                pack_mode=reg.PackMode.B32_TO_B16,
            )

    @jit
    def _rotate_tail(self, dst, h, cos_lanes, sin_lanes, full):
        """Apply interleaved inverse RoPE and round its 64 FP32 tail values to BF16."""
        even, odd = reg.vload_deinterleave(self.rope_buf, h * ROPE_DIM)
        new_even = reg.vadd(
            reg.vmul(even, cos_lanes, mask=full),
            reg.vmul(odd, sin_lanes, mask=full),
            mask=full,
        )
        new_odd = reg.vsub(
            reg.vmul(odd, cos_lanes, mask=full),
            reg.vmul(even, sin_lanes, mask=full),
            mask=full,
        )
        merged, _ = reg.vinterleave(new_even, new_odd)
        reg.vstore_pack(
            dst,
            h * HEAD_DIM + NOPE_DIM,
            reg.vcast(merged, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN),
            full,
            pack_mode=reg.PackMode.B32_TO_B16,
        )


def core_budget():
    """Return the MIX launch quota of the current stream without caching it."""
    info = dsl.get_platform_info()
    cube, vector = int(info.cube_core_num), int(info.vector_core_num)
    if cube <= 0 or vector < 2:
        raise RuntimeError(f"Invalid MIX core quota: AIC={cube}, AIV={vector}")
    # A MIX block uses one AIC and two AIV on the supported architecture.
    return min(cube, vector // 2)


def _tile_fits(base_m, base_n, k):
    """Both stages must fit the architecture buffers, independent of core count."""
    # L0A/L0B hold one BF16 tile each; L0C holds the FP32 accumulator.
    if (
        base_m * BASE_K * BF16_BYTES > L0A_BYTES
        or base_n * BASE_K * BF16_BYTES > L0B_BYTES
    ):
        return False
    if base_m * base_n * FP32_BYTES > L0C_BYTES:
        return False
    # L1 holds both BF16 tiles plus one 4-byte scale per 32 K elements.
    return (base_m + base_n) * (BASE_K * FP32_BYTES + k // QUANT_GROUP) <= L1_BYTES


def _nz_resources(base_m, base_n1, base_n2, k_max, k_min):
    """Pick the largest K window whose padded NZ slots still fit L1.

    The private NZ B slots reserve two 32 KiB copy stripes.  Keep the M/N
    tiles when possible by shrinking the K window before falling back to a
    smaller M tile.
    """
    step_k, depth1, depth2 = _matmul_resources(base_m, base_n1, base_n2, k_max, k_min)
    reuse_n = max(base_n1, base_n2)
    stripes = 2 * _L1_NZ_STRIPE_BYTES if base_n1 != base_n2 else 0
    scale_full = (k_max // 64) * 2
    while step_k > 0:
        k_l1 = BASE_K * step_k
        total = (
            2 * base_m * k_l1
            + 2 * reuse_n * k_l1
            + stripes
            + base_m * scale_full
            + reuse_n * scale_full
        )
        if total <= L1_BYTES:
            return step_k, depth1, depth2
        step_k //= 2
    return None


@lru_cache(maxsize=1024)
def tiling_for_cores(block_num, t=72, m_pad=None, ng=NG, nz=False):
    """Select M/N tiles from the stream core quota and Cube buffer limits."""
    block_num, t = int(block_num), int(t)
    if block_num <= 0 or t <= 0:
        raise ValueError("core count and T must be positive")
    aligned_t = (t + BASE_M_ALIGN - 1) // BASE_M_ALIGN * BASE_M_ALIGN
    if m_pad is not None and (int(m_pad) < t or int(m_pad) % BASE_M_ALIGN):
        raise ValueError(f"m_pad must cover T and be a multiple of {BASE_M_ALIGN}")
    candidates = []
    if ng not in (1, 2, 8):
        raise ValueError("local groups must be 1, 2 or 8")
    k_max = max(F, ng * O_LORA)
    # Cube N tiles are multiples of 32 from 32 to 256 elements.
    widths = range(BASE_N_ALIGN, MAX_BASE_N + 1, BASE_N_ALIGN)
    for base_m in range(BASE_M_ALIGN, min(aligned_t, MAX_BASE_M) + 1, BASE_M_ALIGN):
        legal_n = [n for n in widths if _tile_fits(base_m, n, k_max)]
        if not legal_n:
            continue
        m_tiles = (t + base_m - 1) // base_m
        score, selected = 0, []
        for groups, n_size, k, n_align in (
            # QuantY writes a packed pair of E8M0 bytes per 64 output values.
            (ng, O_LORA, F, MX_DIVISOR),
            (1, DIM, ng * O_LORA, QUANT_GROUP),
        ):
            stage = []
            for n in legal_n:
                if n % n_align or n_size % n:
                    continue
                tiles = groups * m_tiles * ((n_size + n - 1) // n)
                waves = (tiles + block_num - 1) // block_num
                # Approximate L1 traffic: BF16 tile bytes plus scale/bias overhead.
                cost = (
                    waves
                    * k
                    * (BF16_BYTES * base_m * n + TILE_OVERHEAD_BYTES * (base_m + n))
                )
                stage.append((cost, waves, n))
            best = min(stage)
            score += best[0]
            selected.append(best[2])
        step = 0
        if nz:
            resources = _nz_resources(
                base_m, selected[0], selected[1], k_max, min(F // 2, ng * O_LORA)
            )
            if resources is None:
                continue
            step = resources[0]
        candidates.append((score, -base_m, selected[0], selected[1], step))
    if not candidates:
        raise ValueError("no projection tile fits the configured Cube buffers")
    _, negative_m, n1, n2, step = min(candidates)
    return -negative_m, n1, n2, step


def _matmul_resources(base_m, base_n1, base_n2, k=NG * O_LORA, k_min=None):
    """Use the largest legal K window shared by the two projection stages."""
    width = max(base_n1, base_n2)
    k_min = min(F // 2, k) if k_min is None else int(k_min)
    steps = []
    step = 1
    while step * BASE_K <= k_min:
        if k_min % (step * BASE_K) == 0 and (
            (base_m + width) * (BASE_K * step * BF16_BYTES + k // QUANT_GROUP)
            <= L1_BYTES
        ):
            steps.append(step)
        step *= 2
    depths = [
        min(L0C_DEPTH_MAX, L0C_BYTES // (base_m * n * FP32_BYTES))
        for n in (base_n1, base_n2)
    ]
    if not steps or min(depths) < 1:
        raise ValueError("projection matmul tile exceeds Cube buffers")
    return max(steps), depths[0], depths[1]


@kernel
class AttnEpilogueKernel:
    def __init__(self, shape, m_pad, block_num, tiling):
        """Choose projection tiles, legal buffers, and a quota-compatible streaming schedule."""
        self.block_num = int(block_num)
        self.base_n1 = int(tiling.first_n)
        self.base_n2 = int(tiling.second_n)
        self.first_nz, self.second_nz = shape.first_nz, shape.second_nz
        self.split_tiles = (
            self.base_n1 != self.base_n2 or self.first_nz != self.second_nz
        )
        self.reuse_n = max(self.base_n1, self.base_n2)
        self.t = int(shape.t)
        self.ng = int(shape.ng)
        self.f = F
        self.o_lora = O_LORA
        self.dim = DIM
        self.max_k = max(self.f, self.ng * self.o_lora)
        self.m_pad = int(m_pad)
        # Fixpipe gives one half to each paired AIV. Only this even extent is
        # computed; external storage keeps the established 16-row alignment.
        self.m_work = (self.t + 1) // 2 * 2
        self.base_m = int(tiling.rows) if tiling.rows else self.m_pad
        self.step_k, self.depth1, self.depth2 = _matmul_resources(
            self.base_m,
            self.base_n1,
            self.base_n2,
            self.max_k,
            k_min=min(self.f // 2, self.ng * self.o_lora),
        )
        if tiling.step:
            self.step_k = int(tiling.step)
        mm1_tiles = (
            self.ng
            * ((self.t + self.base_m - 1) // self.base_m)
            * ((self.o_lora + self.base_n1 - 1) // self.base_n1)
        )
        # Every launched Cube must own an MM1 tile before consuming slab
        # notifications; Vector lanes must split evenly across local groups.
        self.stream_k = (
            2048
            if self.split_tiles
            and mm1_tiles >= self.block_num
            and (2 * self.block_num) % self.ng == 0
            else 0
        )
        lanes = self.block_num * 2 // self.ng
        rows_per_lane = (self.t + lanes - 1) // lanes if self.stream_k else 1
        batches = (rows_per_lane + 3) // 4
        self.batch_rows = (rows_per_lane + batches - 1) // batches
        if not self.split_tiles:
            self.mm = MatmulMx(
                self.max_k,
                self.base_m,
                self.base_n1,
                BASE_K,
                step_k=self.step_k,
                prefetch_tiles=1,
                prefetch_scales=True,
                l0c_depth=self.depth1,
                weight_nz=self.first_nz,
            )

    def __call__(
        self,
        gm_x,
        gm_cos,
        gm_sin,
        woaq,
        dwa,
        wobq,
        dwb,
        gm_aq,
        gm_asc,
        gm_yq,
        gm_ysc,
        gm_out,
    ):
        """Fuse RoPE/QuantA, grouped MM1/QuantY, and MM2 with explicit producer barriers."""
        saved_ctrl = get_ctrl()
        fp8_saturation_mask = 1 << 48
        set_ctrl(
            saved_ctrl - (saved_ctrl // fp8_saturation_mask % 2) * fp8_saturation_mask
        )
        t = self.t
        ng = self.ng
        m_pad = self.m_pad
        k2 = ng * self.o_lora

        rq = RopeQuant(ng, stream_k=self.stream_k, batch_rows=self.batch_rows)
        rq.run(
            RopeInputs(gm_x, gm_cos, gm_sin),
            QuantOutput(gm_aq, gm_asc),
            t,
            m_pad,
        )
        mm1 = self._first_matmul()
        self._prefetch_weight(mm1, woaq, dwa, ng, self.o_lora)
        if const_expr(not self.stream_k):
            global_sync_all()
        qy = QuantTile(self.base_m, self.base_n1, 65536 + rq.extra_ub)
        self._tile_mm1_quant(
            mm1,
            qy,
            MxOperands(gm_aq, woaq, gm_asc, dwa),
            QuantOutput(gm_yq, gm_ysc),
        )
        # Protect aliased Cube buffers without waiting for other blocks or
        # their Vec quantization; Wb prefetch can overlap that work.
        cube_sync_all()
        mm2 = self._second_matmul()
        self._prefetch_weight(mm2, wobq, dwb, 1, self.dim)
        if const_expr(self.stream_k):
            # All QuantY data/scales must be in GM before any MM2 read.
            # MTE3 orders the stores; the reader waits on its MTE2 pipe.
            vec_sync_block_arrive(PIPE.MTE3, 13, mode=0)
            vec_sync_block_wait(PIPE.MTE3, 13, mode=0)
            vec_sync_block_arrive(PIPE.MTE3, 7)
            cube_sync_block_wait(PIPE.MTE2, 7)
        else:
            global_sync_all()
        self._tile_mm(
            mm2,
            MxOperands(gm_yq, wobq, gm_ysc, dwb),
            gm_out,
            MatmulShape(1, self.dim, k2, m_pad),
        )
        set_ctrl(saved_ctrl)

    @jit
    def _first_matmul(self):
        """Create MM1 on aliased Cube storage or reuse the common equal-tile instance."""
        if const_expr(self.split_tiles):
            mm = MatmulMx(
                self.max_k,
                self.base_m,
                self.base_n1,
                BASE_K,
                step_k=self.step_k,
                prefetch_tiles=2,
                prefetch_scales=True,
                reuse_n=self.reuse_n,
                stream_k=self.stream_k,
                l0c_depth=self.depth1,
                weight_nz=self.first_nz,
            )
        else:
            mm = self.mm
        return mm

    @jit
    def _second_matmul(self):
        """Create MM2 only after the prior Cube pipeline has drained shared storage."""
        if const_expr(self.split_tiles):
            mm = MatmulMx(
                self.max_k,
                self.base_m,
                self.base_n2,
                BASE_K,
                step_k=self.step_k,
                prefetch_tiles=1,
                prefetch_scales=True,
                reuse_n=self.reuse_n,
                l0c_depth=self.depth2,
                weight_nz=self.second_nz,
            )
        else:
            mm = self.mm
        return mm

    @jit
    def _tile_mm(self, mm, operands, out, shape):
        """Distribute final projection tiles across Cube blocks and write valid output rows."""
        a, b = operands.a, operands.b
        sa, sb = operands.a_scale, operands.b_scale
        gpr, n, k, m = shape.groups, shape.n, shape.k, shape.m
        gk = k // MX_DIVISOR
        sa3 = sa.view(sa.shape[0], gk, 2)
        bi = get_block_idx()
        nb = self.block_num
        n_tiles = (n + mm.base_n - 1) // mm.base_n
        m_tiles = (self.t + mm.base_m - 1) // mm.base_m
        for tile in dsl_range(bi, gpr * m_tiles * n_tiles, nb):
            gl = tile // (m_tiles * n_tiles)
            m0 = tile // n_tiles % m_tiles
            n0 = tile % n_tiles
            a_g = matrix_view(a, gl * m, gl * m + self.m_work, 0, k)
            b_g = self._weight_group(b, gl, n, k, mm.weight_nz)
            sa_g = scale_view(sa3, gl * m, gl * m + self.m_work, 0, gk)
            sb_g = self._scale_group(sb, gl, n, k, mm.weight_nz)
            out_g = matrix_view(out, 0, m, gl * n, (gl + 1) * n)
            mm.tile(
                MxOperands(a_g, b_g, sa_g, sb_g),
                out_g,
                TileIndex(m0, n0, k, tile < nb),
            )

    @jit
    def _tile_mm1_quant(self, mm, quant, operands, output):
        """Deliver each FP32 MM1 tile directly to paired AIV quantization without a GM Y buffer."""
        a, b = operands.a, operands.b
        sa, sb = operands.a_scale, operands.b_scale
        q, scales = output.data, output.scales
        sa3 = sa.view(sa.shape[0], self.f // MX_DIVISOR, 2)
        bi = get_block_idx()
        n_tiles = (self.o_lora + mm.base_n - 1) // mm.base_n
        m_tiles = (self.t + mm.base_m - 1) // mm.base_m
        for tile in dsl_range(bi, self.ng * m_tiles * n_tiles, self.block_num):
            gl = tile // (m_tiles * n_tiles)
            m0 = tile // n_tiles % m_tiles
            n0 = tile % n_tiles
            a_g = matrix_view(
                a, gl * self.m_pad, gl * self.m_pad + self.m_work, 0, self.f
            )
            b_g = self._weight_group(b, gl, self.o_lora, self.f, mm.weight_nz)
            sa_g = scale_view(
                sa3,
                gl * self.m_pad,
                gl * self.m_pad + self.m_work,
                0,
                self.f // MX_DIVISOR,
            )
            sb_g = self._scale_group(sb, gl, self.o_lora, self.f, mm.weight_nz)
            acc = mm.compute(
                MxOperands(a_g, b_g, sa_g, sb_g),
                TileIndex(m0, n0, self.f, tile < self.block_num),
            )
            actual_m = min(mm.base_m, self.m_work - m0 * mm.base_m)
            actual_n = min(mm.base_n, self.o_lora - n0 * mm.base_n)
            quant.run(
                acc,
                QuantOutput(q, scales),
                TileExtent(
                    m0 * mm.base_m,
                    gl * self.o_lora + n0 * mm.base_n,
                    actual_m,
                    actual_n,
                ),
            )

    @jit
    def _weight_group(self, b, group, n, k, nz):
        """Select one logical weight group without reinterpreting private NZ storage."""
        if const_expr(nz):
            if const_expr(len(b.shape) == 3):
                return tile_slice(
                    tile_slice(b, (1, k, n), (group, 0, 0)),
                    (k, n),
                    (0, 0),
                )
            else:
                return b
        else:
            return matrix_view(b, group * n, (group + 1) * n, 0, k)

    @jit
    def _scale_group(self, sb, group, n, k, nz):
        """Select the ND MX scale group in the logical axes required by its weight format."""
        if const_expr(nz):
            if const_expr(len(sb.shape) == 4):
                return sb[group : group + 1, 0 : k // MX_DIVISOR, 0:n, 0:2].view(
                    k // MX_DIVISOR, n, 2
                )
            else:
                return sb
        else:
            return scale_view(sb, group * n, (group + 1) * n, 0, k // MX_DIVISOR)

    @jit
    def _prefetch_weight(self, mm, b, sb, groups, n):
        """Prefetch the first owned B tile on blocks that have projection work."""
        bi = get_block_idx()
        nn = (n + mm.base_n - 1) // mm.base_n
        mn = (self.t + mm.base_m - 1) // mm.base_m
        k = b.shape[-2] if mm.weight_nz else b.shape[-1]
        if bi < groups * mn * nn:
            gl = bi // (mn * nn)
            mm.prefetch(
                self._weight_group(b, gl, n, k, mm.weight_nz),
                bi % nn,
                self._scale_group(sb, gl, n, k, mm.weight_nz),
            )


class AttnEpilogue:
    def __init__(self, shape, m_pad, tiling):
        """Store the static shape and launch plan used by the host compiler."""
        self.shape = shape
        self.m_pad = int(m_pad)
        self.tiling = tiling

    @host
    def run(
        self,
        gm_x,
        gm_cos,
        gm_sin,
        woaq,
        dwa,
        wobq,
        dwb,
        gm_aq,
        gm_asc,
        gm_yq,
        gm_ysc,
        gm_out,
        block_num: Constexpr[int],
    ):
        """Launch one MIX kernel with the current stream Cube quota and four workspaces."""
        op = AttnEpilogueKernel(self.shape, self.m_pad, block_num, self.tiling)
        op[block_num](
            gm_x,
            gm_cos,
            gm_sin,
            woaq,
            dwa,
            wobq,
            dwb,
            gm_aq,
            gm_asc,
            gm_yq,
            gm_ysc,
            gm_out,
        )


@compile_cache
def _get_compiled(t, ng, first_nz, second_nz, m_pad, block_num):
    """Share compilation, caching and AOT export with the Torch runtime entry."""
    shape = ProjectionShape(t, ng, first_nz, second_nz)
    base_m, base_n1, base_n2, step = tiling_for_cores(
        block_num, t=t, ng=ng, nz=first_nz or second_nz
    )
    fp8_dtype = dtypes.float8_e4m3fn
    k2 = ng * O_LORA
    return dsl.compile(
        AttnEpilogue(
            shape,
            m_pad,
            ProjectionTiles(base_m, base_n1, base_n2, step),
        ).run,
        dsl.TensorSpec((t * ng, F), dtypes.bfloat16),
        dsl.TensorSpec((t, ROPE_DIM), dtypes.float32),
        dsl.TensorSpec((t, ROPE_DIM), dtypes.float32),
        dsl.TensorSpec(
            (ng, F, O_LORA) if first_nz else (ng * O_LORA, F),
            fp8_dtype,
            storage_format="nz" if first_nz else "nd",
        ),
        dsl.TensorSpec(
            (ng, F // MX_DIVISOR, O_LORA, 2)
            if first_nz
            else (ng * O_LORA, F // MX_DIVISOR, 2),
            dtypes.uint8,
        ),
        dsl.TensorSpec(
            (k2, DIM) if second_nz else (DIM, k2),
            fp8_dtype,
            storage_format="nz" if second_nz else "nd",
        ),
        dsl.TensorSpec(
            (k2 // MX_DIVISOR, DIM, 2) if second_nz else (DIM, k2 // MX_DIVISOR, 2),
            dtypes.uint8,
        ),
        dsl.TensorSpec((ng * m_pad, F), dtypes.uint8),
        dsl.TensorSpec((ng * m_pad, F // QUANT_GROUP), dtypes.uint8),
        dsl.TensorSpec((m_pad, k2), dtypes.uint8),
        dsl.TensorSpec((m_pad, k2 // QUANT_GROUP), dtypes.uint8),
        dsl.TensorSpec((m_pad, DIM), dtypes.bfloat16),
        block_num,
    )


@export("attn_epilogue")
def export_attn_epilogue():
    """Compile the default NG=8/2/1 profiles without NPU execution.

    ``T`` enters the compiled program (tiling candidates, operand shapes), so
    the catalogue enumerates representative token counts for both supported
    MIX launch quotas. ``CANNBOTDSL_DS41_PROFILES`` may override the list with
    an ``attn_epilogue`` array of entries with ``block_num``, ``tokens``,
    ``ng`` and ``weight_format`` (``nd``/``nz``). Optional ``first_nz`` and
    ``second_nz`` select mixed layouts. Unexported profiles still use JIT.
    """
    import json
    import os

    default_tokens = (1, 72, 128, 256, 512, 1024, 2048, 4096, 8192)
    profiles = [
        (quota, t, ng, nz, nz)
        for quota in (28, 32)
        for ng in (8, 2, 1)
        for nz in (False, True)
        for t in (default_tokens if ng == 8 else (1, 16))
    ]
    profile_path = os.environ.get("CANNBOTDSL_DS41_PROFILES")
    if profile_path:
        with open(profile_path, encoding="utf-8") as stream:
            entries = json.load(stream).get("attn_epilogue")
        if entries is None:
            raise ValueError(
                "CANNBOTDSL_DS41_PROFILES must define an 'attn_epilogue' list"
            )
        profiles = []
        for entry in entries:
            ng = int(entry.get("ng", NG))
            layout = entry.get("weight_format", "nd")
            if ng not in (1, 2, 8) or layout not in ("nd", "nz"):
                raise ValueError("export ng must be 1/2/8 and weight_format nd/nz")
            for t in entry.get("tokens", default_tokens):
                profiles.append(
                    (
                        int(entry.get("block_num", 32)),
                        int(t),
                        ng,
                        bool(entry.get("first_nz", layout == "nz")),
                        bool(entry.get("second_nz", layout == "nz")),
                    )
                )
    for block_num, t, ng, first_nz, second_nz in dict.fromkeys(profiles):
        _get_compiled(t, ng, first_nz, second_nz, (t + 15) // 16 * 16, block_num)
