# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import os
import threading
from itertools import product
from types import SimpleNamespace
import cannbotdsl
import torch
from cannbotdsl import aot

from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.ops.arch import get_subblock_id, get_block_idx, get_subblock_dim
from cannbotdsl import dtypes
from cannbotdsl import Constexpr, Dim, ProfileSpec, TensorSpec, host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.cube import set_mmad_direction
from cannbotdsl.ops.memcpy import copy_options, mem_copy
from cannbotdsl import (
    MemLoc,
    PIPE,
    Tensor,
    is_aic,
    CArray,
    datastruct,
    from_tensor,
    make_layout,
)
from cannbotdsl.tensor import ceil_div, tile_slice
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.sync import (
    cube_sync_block_arrive,
    cube_sync_intra_wait,
    global_sync_all,
    vec_sync_block_wait,
    vec_sync_all,
    vec_sync_intra_arrive,
)
from cannbotdsl.ops.sync import onchip_rewind
from cannbotdsl.ops import reg as rr

# Consume the existing library metadata layout (smla_metadata_common.h).
# The serialized capacity is independent of the 36-core DSL launch limit.
AIC_CORE_MAX_NUM = 36
MQSMLA_METADATA_TOTAL_SIZE = 2048
FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8
FD_METADATA_BASE = 80 * FA_METADATA_SIZE

FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_S2_START_INDEX = 3
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5
FA_S2_END_INDEX = 6
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7

FD_CORE_ENABLE_INDEX = 0
FD_BN2_IDX_INDEX = 1
FD_M_IDX_INDEX = 2
FD_WORKSPACE_IDX_INDEX = 3
FD_WORKSPACE_NUM_INDEX = 4
FD_M_START_INDEX = 5
FD_M_NUM_INDEX = 6

_SPAN_CACHE = {}
_SPAN_CACHE_LOCK = threading.Lock()


def _get_cube_core_num(device=None):
    """Return the cube-core count of the active NPU (same source as the consumer)."""
    import torch

    return int(torch.npu.get_device_properties(device).cube_core_num)


def _bsnd_query_spans(batch, seq, device):
    """Reuse immutable shape constants; never cache a graph-owned allocation.

    The blocking CPU-to-device copy completes before publishing the tensor, so
    another stream can safely use a cache hit without an event or device work.
    Constants remain alive for the process lifetime: a captured graph can keep
    using them after arbitrary other shapes have run. Cold capture instead
    retains the previous graph-local construction.
    """
    import torch

    device = torch.device(device)
    if device.type == "npu" and device.index is None:
        device = torch.device("npu", torch.npu.current_device())
    key = (device, batch, seq)
    with _SPAN_CACHE_LOCK:
        spans = _SPAN_CACHE.get(key)
        if spans is not None:
            return spans
        if device.type == "npu" and torch.npu.is_current_stream_capturing():
            return torch.arange(batch + 1, dtype=torch.int32, device=device) * seq
        spans = (torch.arange(batch + 1, dtype=torch.int32) * seq).to(
            device, non_blocking=False
        )
        _SPAN_CACHE[key] = spans
        return spans


def _flatten_bsnd_lengths(lengths, seqused_q):
    """Flatten BSND lengths while clearing padded queries."""
    import torch

    if lengths is None:
        return None
    batch, seq, heads = lengths.shape
    if seqused_q is not None:
        valid = (
            torch.arange(seq, dtype=torch.int32, device=lengths.device)[None, :]
            < seqused_q[:, None]
        )
        lengths = torch.where(valid[:, :, None], lengths, torch.zeros_like(lengths))
    return lengths.view(batch * seq, heads)


def _validate_api_modes(
    quant_mode,
    rope_head_dim,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    cmp_residual_kv,
):
    """The public signature follows ops-transformer; math stays mixed FP8/FP4."""
    if quant_mode != 4:
        raise NotImplementedError("This mixed FP8/FP4 DSL supports quant_mode=4 only")
    if rope_head_dim not in (None, 64):
        raise NotImplementedError("This DSL supports rope_head_dim=64 only")
    if (
        cmp_ratio not in (None, 1)
        or ori_mask_mode not in (None, 0)
        or cmp_mask_mode not in (None, 0)
        or ori_win_left not in (None, -1)
        or ori_win_right not in (None, -1)
        or cmp_residual_kv is not None
    ):
        raise NotImplementedError(
            "This DSL currently supports explicit sparse indices with no mask only"
        )


# Address precomputation preserves logical sparse-column order in both UB paths.
# Dense power-of-two pages use vector addressing; other layouts use scalar addressing.
_ADDR_VEC_OVERRIDE = None


# NZ face stride in BF16 elements; UB padding is skipped by copy-out.
_NZ_ROW = 16
_NZ_PAD_ROWS = 17
_NZ_CHUNK = _NZ_PAD_ROWS * _NZ_ROW

# Address-prepass buffers are reclaimed after the block handshake.
# Main-loop and FD reduction buffers reuse UB at separate phases.
# Conservative address-prepass arena capacity. On this dav-3510 platform,
# SIMD has 248 KiB usable UB; SIMT reserves another 32 KiB for DCache.
# Keep the prepass arena at 216 KiB while the SIMD main phase uses up to 248 KiB.
_ADDR_UB_BYTES = 216 * 1024
# Separate small-pool owners retain MTE2/MTE3 overlap between prepass rows.
_ADDR_COMPACT_BT = 2048
_ADDR_COMPACT_K = 1024

# Compile-time shape contract.
D = 512  # Logical head dimension: nope[448] followed by rope[64].
N1 = 64  # Maximum Q heads; also the fixed FD workspace slot stride.
N2 = 1  # KV heads; Q supports every head count in 1..64.
TILE_N = 128  # KV tokens per S2 task.

# Per-token physical layout: nope[448], rope[64], then inline BF16 scales.
# FP8 E4M3 uses one scale per 32 values: 512 data bytes + 32 scale bytes.
# FP4 E2M1 uses one scale per 16 values: 256 packed bytes + 64 scale bytes.
# The even FP4 value occupies the low nibble; inputs use uint8 byte views.
KV_ROW_BYTES_ORI = 544
KV_ROW_BYTES_CMP = 320
# Empirical threshold on total dense KV payload, not the hardware L2 capacity.
_KV_NOALLOC_THRESHOLD_BYTES = 32 * 1024 * 1024
# Empirical prefill reuse threshold per batch; cache hints do not affect math.
_KV_L2_PREFILL_MIN_QUERIES = 128
ROW_BF16_ORI = KV_ROW_BYTES_ORI // 2  # ORI row width in BF16 elements.
SCALE_BF16_OFF_ORI = 256  # ORI scale offset in BF16 elements.
SCALE_BF16_OFF_CMP = 128  # CMP scale offset in BF16 elements.

# FD workspace: normalized O slots followed by per-slot max/sum.
# Each core stages at most two partial rows: a leading continuation and a trailing split.
FD_SLOTS_PER_CORE = 2
FD_O_ELEMS = N1 * D  # FP32 output elements per slot.
FD_MS_ELEMS = 2 * N1  # FP32 max[64] and sum[64] per slot.
FD_SLOT_ELEMS = FD_O_ELEMS + FD_MS_ELEMS  # 32896 fp32 = 131584 B
FD_BYTES_PER_CORE = FD_SLOTS_PER_CORE * FD_SLOT_ELEMS * 4  # 263168
WS_BYTES_PER_CORE = FD_BYTES_PER_CORE  # 263168

# FD is selected solely by runtime metadata. Partial O is written directly
# from res_o; reduction storage reuses UB after the global barrier.
_FD_NEG_INF = -1.0e38
_FD_MP_SLOTS = AIC_CORE_MAX_NUM  # One row may span every launched Cube core.

# Dequantized KV is written directly from UB to the shared L1 ring.

# Default ORI offsets; _vec0_body selects the offsets for each pool.
_ROW_BF16 = ROW_BF16_ORI  # Both input pools use the ORI 544-byte UB row stride.
_SCALE_BF16_OFF = SCALE_BF16_OFF_ORI  # ORI scale offset in the shared input layout.

_VL = 128  # BF16 vector width for dequantization.
_TILE_ROWS = 16  # Rows per dequantization subtile and per NZ face.
# Sentinel beyond every valid query index; prevents reads past the final cu_q span.
_ADDR_HALF = 64  # Row indices processed by each address vector.
_ADDR_M_INF = 1 << 40


TILE = 64
D_TILES = D // TILE  # Eight 64-column chunks: QK reduction / PV output.

# L1 per AIC: KV 384 KiB + Q 112 KiB + P 16 KiB = 512 KiB.
# KV slots remain live through QK and PV; Q slots remain live for a query.
_KV_RING = 3  # KV slots shared by QK and PV.

_Q_RING = 2  # Q[64,384] 双槽，加 Q[64,128] tail 单槽，共 112 KiB.

_P_L1_DEPTH = 1  # Cross-core P slots from AIV store_p to AIC PV.

# Cross-core RingBuffer slots occupy sync ids below 9. The address-table
# publication barrier uses id 9; AIV1's intra-core token is offset by 16.
_AIV1_ID_OFFSET = 16

_KV_NUM_PER_LOOP = 128
_TILE_SIZE = 64
_COMBINE_DIM = _ROW_BF16 * 2

# Rank contracts and hardware limits (not tensor data values).
_TENSOR_RANK_1D = 1
_TENSOR_RANK_2D = 2
_TENSOR_RANK_3D = 3
_TENSOR_RANK_4D = 4
_MAX_PA_BLOCK_SIZE = 1024
# UB/GM paired-copy source stride is encoded in an unsigned 39-bit field.
_DMA_SOURCE_STRIDE_MAX_BYTES = (1 << 39) - 1

# Byte offsets in the vector address table must remain positive INT32 values.
_PA_FAST_MAX_ORI_PAGES = ((1 << 31) - 1) // (128 * KV_ROW_BYTES_ORI)
_PA_FAST_MAX_CMP_PAGES = ((1 << 31) - 1) // (128 * KV_ROW_BYTES_CMP)
_PA_FAST_OVERRIDE = False

_COMPILE_LOCK = threading.Lock()


def _addr_vec_mode():
    return _ADDR_VEC_OVERRIDE or "off"


def _pa_fast_mode():
    return _PA_FAST_OVERRIDE


def _pa_fast_eligible(ori, bt, idx, cmp, btc, idxc, layout_kv):
    """Specialize common PA shapes without inspecting device tensor values.

    Larger pools, padded pages, other widths and other layouts keep the generic
    path. The page bounds also form part of the compiled TensorSpec contract.
    """
    if layout_kv != "PA_BBND" or not _addr_vec_eligible(ori, bt, idx, cmp, btc, idxc):
        return False
    return (
        ori.shape[1] == 128
        and idx.shape[2] == 128
        and ori.shape[0] <= _PA_FAST_MAX_ORI_PAGES
        and (
            cmp is None
            or (
                cmp.shape[1] == 128
                and idxc.shape[2] == 512
                and cmp.shape[0] <= _PA_FAST_MAX_CMP_PAGES
            )
        )
    )


def _addr_vec_eligible(ori_kv, ori_bt, ori_idx, cmp_kv, cmp_bt, cmp_idx):
    """Select vector addressing only when each pool's aligned staging fits UB.

    ORI/CMP run sequentially and reuse the same arena. Main-loop UB is allocated
    after the prepass publication barrier and rewind, so it is not additive.
    """

    def pool_supports_vector_addressing(kv, bt, idx, row_bytes):
        if bt is None:
            return False
        block_size, table_width, index_width = kv.shape[1], bt.shape[1], idx.shape[2]
        if not all(isinstance(n, int) for n in (block_size, table_width, index_width)):
            return False
        return (
            block_size >= 16
            and block_size & (block_size - 1) == 0
            and kv.stride(0) == block_size * row_bytes
            and kv.shape[0] * block_size <= 1 << 31
            and _addr_ub_bytes(table_width, index_width) <= _ADDR_UB_BYTES
        )

    return pool_supports_vector_addressing(
        ori_kv, ori_bt, ori_idx, KV_ROW_BYTES_ORI
    ) and (
        cmp_kv is None
        or pool_supports_vector_addressing(cmp_kv, cmp_bt, cmp_idx, KV_ROW_BYTES_CMP)
    )


def _addr_ub_bytes(table_width, index_width):
    """Include DMA alignment and the output's padded 16-column consumer tail."""
    return 4 * (_addr_tab_w(table_width) + 2 * _addr_tab_w(index_width))


def _addr_tab_w(k):
    """Round K up to 16 columns; the prepass clamps padded columns to the last valid index."""
    return ((k + _TILE_ROWS - 1) // _TILE_ROWS) * _TILE_ROWS


def _copy_addr_to_gm(gm, ch, elem_off, n_elems):
    mem_copy(
        gm.view(gm.shape[0] * gm.shape[1])[elem_off : elem_off + n_elems,].view(
            1, n_elems
        ),
        from_tensor(ch, layout=make_layout((1, n_elems))),
    )


def _copy_addr_from_gm(ch, gm, elem_off, count):
    mem_copy(
        from_tensor(ch, layout=make_layout((1, count))),
        gm.view(gm.shape[0] * gm.shape[1])[elem_off : elem_off + count,].view(1, count),
    )


def _i64(gm, *idx):
    raw = gm[idx] if len(idx) > 1 else gm[idx[0]]
    # Extend int32 GM indices to int64 before physical-address multiplication.
    return dtypes.int64(raw)


class Kvcache:
    """Resolve PA_BBND block-table indices to physical pages and row offsets.

    Block-table values are global physical page indices; batch selects the table.
    Runtime page sizes need not be powers of two.
    """

    @jit
    def pa_blk_off(self, gm_block_table, s2_idx, bo_idx, bs):
        # Common page size avoids runtime division; all other sizes remain valid.
        blk = dtypes.int64(0)
        off = dtypes.int64(0)
        if bs == 128:
            blk = s2_idx // 128
            off = s2_idx % 128
        else:
            blk = s2_idx // bs
            off = s2_idx % bs
        return _i64(gm_block_table, bo_idx, blk), off


class Matmul:
    """Load Q/KV into L1, compute QK and PV, and send results through FIXPIPE.

    Q splits lower/upper heads into two 384-column slots and one 128-column tail.
    Lower/upper KV each use three [64,512] slots, live through QK and PV.
    C1/C2 share four L0A and four L0B slots. QK has one [64,128]
    accumulator; PV has eight independent [64,64] accumulators.
    """

    def __init__(self, tile_cube_m, tile_vec_m, tile_n, full_heads=True):
        self.full_heads = full_heads
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D

        self.q_lower_heads_l1 = [
            Buffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((tile_vec_m, width)),
                data_format="nz",
            )
            for width in (384, 384, 128)
        ]
        self.kv_channels = [
            RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((tile_n // 2 * D,)),
                data_format="nd",
                depth=_KV_RING,
            )
        ]
        self.p_l1 = RingBuffer(
            MemLoc.L1, dtypes.bfloat16, make_layout((tile_cube_m, tile_n)), depth=1
        )
        self.kv_channels.append(
            RingBuffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((tile_n // 2 * D,)),
                data_format="nd",
                depth=_KV_RING,
            )
        )
        self.q_upper_heads_l1 = [
            Buffer(
                MemLoc.L1,
                dtypes.bfloat16,
                make_layout((tile_vec_m, width)),
                data_format="nz",
            )
            for width in (384, 384, 128)
        ]
        self.l0a = RingBuffer(
            MemLoc.L0A,
            dtypes.bfloat16,
            make_layout((tile_cube_m, 128)),
            depth=4,
            data_format="nz",
            physical_layout=make_layout(
                (8, tile_cube_m // 16, 16, 16), stride=(tile_cube_m * 16, 256, 16, 1)
            ),
        )
        self.l0b = [
            Buffer(
                MemLoc.L0B,
                dtypes.bfloat16,
                make_layout((tile_n, 64)),
                data_format="nz",
                physical_layout=make_layout(
                    (4, tile_n // 16, 16, 16), stride=(tile_n * 16, 256, 16, 1)
                ),
            )
            for _ in range(4)
        ]
        self.l0c_pv = [
            Buffer(MemLoc.L0C, dtypes.float32, make_layout((tile_cube_m, TILE)))
            for _ in range(8)
        ]
        self.l0c_qk = Buffer(
            MemLoc.L0C, dtypes.float32, make_layout((tile_cube_m, tile_n))
        )

    @jit
    def load_q_wide(self, q_tile_gm, job, load_d_tail):
        q = q_tile_gm.view(q_tile_gm.shape[1], self.tile_d)
        q_lo, q_hi = self.q_lower_heads_l1[0], self.q_upper_heads_l1[0]
        start, width = 0, 384
        if const_expr(load_d_tail):
            q_lo, q_hi = self.q_lower_heads_l1[2], self.q_upper_heads_l1[2]
            start, width = 384, 128
        elif job % 2 != 0:
            q_lo, q_hi = self.q_lower_heads_l1[1], self.q_upper_heads_l1[1]
        if const_expr(self.full_heads):
            mem_copy(
                q_lo,
                q[0 : self.tile_vec_m, start : start + width],
                copy_options(l2_cache_ctl=4),
            )
            mem_copy(
                q_hi,
                q[self.tile_vec_m : self.tile_cube_m, start : start + width],
                copy_options(l2_cache_ctl=4),
            )
        else:
            # The NZ slots retain their aligned M stride. Only real Q rows
            # are fetched; unused rows are never written to public outputs.
            lower = (q.shape[0] + 1) // 2
            # Even an inactive half must publish its MTE2 producer event.
            # Repeat one valid row for that half; it has no public output.
            upper = max(q.shape[0] // 2, 1)
            upper_start = min(lower, q.shape[0] - 1)
            mem_copy(
                q_lo[0:lower, None],
                q[0:lower, start : start + width],
                copy_options(l2_cache_ctl=4),
            )
            mem_copy(
                q_hi[0:upper, None],
                q[upper_start : upper_start + upper, start : start + width],
                copy_options(l2_cache_ctl=4),
            )

    @jit
    def bmm1_fanout_q(self, job, kv_lower_rows_l1, kv_upper_rows_l1):
        set_mmad_direction("m")
        q_lo, q_hi = self.q_lower_heads_l1[0], self.q_upper_heads_l1[0]
        for col in range_constexpr(8):
            a_raw = self.l0a.next_write()
            a_slot = from_tensor(
                a_raw, layout=make_layout((self.tile_cube_m, 64)), data_format="nz"
            )
            if const_expr(col < 6):
                q_col = col
                if job % 2 == 0:
                    q_lo, q_hi = self.q_lower_heads_l1[0], self.q_upper_heads_l1[0]
                else:
                    q_lo, q_hi = self.q_lower_heads_l1[1], self.q_upper_heads_l1[1]
            else:
                q_lo, q_hi = self.q_lower_heads_l1[2], self.q_upper_heads_l1[2]
                q_col = col - 6
            # Both halves share the full tile's NZ pitch across K fractals.
            a_half = make_layout((self.tile_vec_m, 64))
            a_physical = make_layout(
                (4, self.tile_vec_m // 16, 16, 16),
                stride=(self.tile_cube_m * 16, 256, 16, 1),
            )
            mem_copy(
                from_tensor(a_raw, layout=a_half, physical_layout=a_physical),
                tile_slice(q_lo, (self.tile_vec_m, 64), (0, q_col)),
            )
            mem_copy(
                from_tensor(
                    a_raw,
                    layout=a_half,
                    physical_layout=a_physical,
                    offset=self.tile_vec_m * 16 * 2,
                ),
                tile_slice(q_hi, (self.tile_vec_m, 64), (0, q_col)),
            )

            b_slot = from_tensor(
                self.l0b[col % 4],
                layout=make_layout((self.tile_n, 64)),
                data_format="nz",
            )
            b_half = make_layout((self.tile_n // 2, 64))
            b_physical = make_layout(
                (4, self.tile_n // 32, 16, 16), stride=(self.tile_n * 16, 256, 16, 1)
            )
            kv_layout = make_layout((self.tile_n // 2, D))
            kv_lo = from_tensor(kv_lower_rows_l1, layout=kv_layout, data_format="nz")
            kv_hi = from_tensor(kv_upper_rows_l1, layout=kv_layout, data_format="nz")
            mem_copy(
                from_tensor(b_slot, layout=b_half, physical_layout=b_physical),
                tile_slice(kv_lo, (self.tile_n // 2, 64), (0, col)),
            )
            mem_copy(
                from_tensor(
                    b_slot,
                    layout=b_half,
                    physical_layout=b_physical,
                    offset=self.tile_n // 2 * 16 * 2,
                ),
                tile_slice(kv_hi, (self.tile_n // 2, 64), (0, col)),
            )
            matmul(
                self.l0c_qk,
                a_slot,
                b_slot,
                init=(col == 0),
                unit_flag=3 if col == 7 else 2,
            )

    def store_s(self, buf_score):
        mem_copy(
            buf_score, self.l0c_qk.partition(dim=0), options=copy_options(unit_flag=3)
        )

    @jit
    def compute_pv_fanout(
        self, p_l1, results, kv_lower_rows_l1, kv_upper_rows_l1, is_first
    ):
        set_mmad_direction("n")
        a_slot = self.l0a.next_write()
        for col in range_constexpr(8):
            b_slot = from_tensor(
                self.l0b[col % 4],
                layout=make_layout((64, self.tile_n)),
                data_format="nz",
            )
            b_half = make_layout((64, self.tile_n // 2))
            kv_layout = make_layout((self.tile_n // 2, D))
            kv_lo = from_tensor(kv_lower_rows_l1, layout=kv_layout, data_format="nz")
            kv_hi = from_tensor(kv_upper_rows_l1, layout=kv_layout, data_format="nz")
            mem_copy(
                from_tensor(b_slot, layout=b_half, data_format="nz"),
                tile_slice(kv_lo, (self.tile_n // 2, 64), (0, col)).permute(1, 0),
            )
            mem_copy(
                from_tensor(
                    b_slot,
                    layout=b_half,
                    data_format="nz",
                    offset=64 * (self.tile_n // 2) * 2,
                ),
                tile_slice(kv_hi, (self.tile_n // 2, 64), (0, col)).permute(1, 0),
            )
            if const_expr(col == 0):
                mem_copy(a_slot, p_l1)
            matmul(self.l0c_pv[col], a_slot, b_slot, init=True, unit_flag=0)
            acc_target, pv_target = results[(col // 4) * 2], results[(col // 4) * 2 + 1]
            if is_first == 1:
                mem_copy(
                    tile_slice(acc_target, (self.tile_vec_m, 64), (0, col % 4)),
                    self.l0c_pv[col].partition(dim=0),
                    options=copy_options(unit_flag=0),
                )
            else:
                mem_copy(
                    tile_slice(pv_target, (self.tile_vec_m, 64), (0, col % 4)),
                    self.l0c_pv[col].partition(dim=0),
                    options=copy_options(unit_flag=0),
                )


class Vector:
    """Compute online softmax and output accumulation for up to 16/32 heads per AIV.

    For old state (m,l,o) and scaled scores s:
      m_new = max(m, max(s)); alpha = exp(m-m_new); p = exp(s-m_new).
      l_new = alpha*l + sum(p); o_new = alpha*o + BF16(p) @ KV.
    Only valid columns contribute to max, exponent/P and denominator.
    Sinks seed (m=sink,l=1,o=0) without a value contribution. Final output is
    BF16(o/l), with LSE=log(l)+m. A zero sink still contributes exp(0)=1.
    Double-buffer max/sum by nonempty query and alpha by tick. The two output
    halves serve consecutive queries, with copy completion before reuse.
    """

    def __init__(
        self,
        tile_vec_m,
        tile_n,
        tile_d,
        subblock_idx,
        sinks_span,
        full_heads=True,
        num_heads=64,
        has_sinks=True,
    ):
        self.has_sinks = has_sinks
        self.full_heads = full_heads
        self.num_heads = num_heads
        self.sinks_span = sinks_span
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx

        # Double-buffer max/sum by query and exp by task tick.
        self.sm_max_tb = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 2)))
            for _ in range(2)
        ]
        self.sm_sum_tb = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 2)))
            for _ in range(2)
        ]
        self.sm_exp_tb = [
            Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 2)))
            for _ in range(2)
        ]
        # Partial weights are needed only after the main-loop rewind.
        self.fd_mp_tb = []

        self.lse_ub = Buffer(MemLoc.UB, dtypes.float32, make_layout((tile_vec_m, 1)))
        # Main-loop output views are bound independently for each 256-column half.

        self.sinks_ub = (
            Buffer(MemLoc.UB, dtypes.float32, make_layout((1, self.sinks_span)))
            if has_sinks
            else None
        )

    def bind_result(self, results):
        self.result_views = results
        self.out_halves = [
            from_tensor(
                results[half * 2 + 1],
                layout=make_layout((self.tile_vec_m, 256), stride=(512, 1)),
                dtype=dtypes.bfloat16,
            )
            for half in range(2)
        ]

    @jit
    def softmax_rest_full(
        self,
        job,
        tile,
        scale,
        first,
        all_valid,
        buf_score,
        buf_p,
        valid_tile_kv_rows,
        has_sink,
    ):
        sm_max = self.sm_max_tb[job % 2]
        sm_sum = self.sm_sum_tb[job % 2]
        sm_exp = self.sm_exp_tb[tile % 2]
        head_rows = self.tile_vec_m
        if const_expr(not self.full_heads):
            head_rows = max(self.head_count(), 1)
        # VF softmax: reduce QK scores, update online state and pack BF16 P in NZ order.
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=32)
            state_mask = mask
            if const_expr(self.tile_vec_m < 32):
                state_mask, _ = rr.update_mask(2 * self.tile_vec_m, elem_bits=32)
            mask16 = rr.full_mask(elem_bits=16)
            # Keep two FP32 copies per head for the interleaved online softmax state.
            two_lanes, _ = rr.update_mask(2, elem_bits=32)
            softmax_state_offset = 0
            if const_expr(first and has_sink):
                sink = rr.vload(self.sinks_ub)
                neg_inf = rr.veqs(
                    rr.vreinterpret(sink, dtypes.uint32), 0xFF800000, mask=mask
                )
                pos_inf = rr.veqs(
                    rr.vreinterpret(sink, dtypes.uint32), 0x7F800000, mask=mask
                )
                unscaled_sink_reg = rr.vmuls(sink, 1.0 / scale, mask=mask)
                unscaled_sink_reg = rr.vdups(
                    (-3.4028234663852886e38),
                    dtypes.float32,
                    mask=neg_inf,
                    mode="merging",
                    merge=unscaled_sink_reg,
                )
                unscaled_sink_reg = rr.vdups(
                    -(-3.4028234663852886e38),
                    dtypes.float32,
                    mask=pos_inf,
                    mode="merging",
                    merge=unscaled_sink_reg,
                )
                replicated, unused = rr.vinterleave(
                    unscaled_sink_reg, unscaled_sink_reg
                )
                rr.vstore(sm_max, softmax_state_offset, replicated, state_mask)
                initial_sum = rr.vdups(1.0, dtypes.float32)
                initial_sum = rr.vdups(
                    0.0, dtypes.float32, mask=neg_inf, mode="merging", merge=initial_sum
                )
                replicated, unused = rr.vinterleave(initial_sum, initial_sum)
                rr.vstore(sm_sum, softmax_state_offset, replicated, state_mask)
                # Order these state stores before the following state loads.
                rr.vmem_bar("vst_vld")
            if const_expr(first and not self.has_sinks):
                # No tensor or sink contribution: initialize the online state
                # directly in registers, in the scaled score domain.
                old_max = rr.vreinterpret(
                    rr.vdups(0xFF800000, dtypes.uint32), dtypes.float32
                )
                old_sum = rr.vdups(0.0, dtypes.float32)
            elif const_expr(not first or has_sink):
                old_max = rr.vload(sm_max, softmax_state_offset)
                rr.vmem_bar("vld_vst")
                old_sum = rr.vload(sm_sum, softmax_state_offset)
                old_max = rr.vmuls(old_max, scale, mask=mask)

            neg_inf_reg = rr.vreinterpret(
                rr.vdups(0xFF800000, dtypes.uint32), dtypes.float32
            )
            if const_expr(not all_valid):
                even_index_reg = rr.vmuls(rr.varange(0, dtypes.int32), 2, mask=mask)
                odd_index_reg = rr.vadds(even_index_reg, 1, mask=mask)
                invalid_even = rr.vges(
                    even_index_reg, dtypes.int32(valid_tile_kv_rows), mask=mask
                )
                invalid_odd = rr.vges(
                    odd_index_reg, dtypes.int32(valid_tile_kv_rows), mask=mask
                )
            max_cursor = rr.vstore_unalign_begin(sm_max)
            # Explicit int32 loop values select the VF scalar ABI.
            for query_head_row in cannbotdsl.range(
                dtypes.int32(0),
                dtypes.int32(head_rows),
                dtypes.int32(1),
                unroll=1,
            ):
                score_even_reg, score_odd_reg = rr.vload_deinterleave(
                    buf_score, query_head_row * TILE_N
                )
                if const_expr(not all_valid):
                    score_even_reg = rr.vselect(
                        neg_inf_reg, score_even_reg, cond_mask=invalid_even
                    )
                    score_odd_reg = rr.vselect(
                        neg_inf_reg, score_odd_reg, cond_mask=invalid_odd
                    )
                row_max_reg = rr.vmax(
                    rr.vreduce_max(score_even_reg, mask=mask),
                    rr.vreduce_max(score_odd_reg, mask=mask),
                    mask=mask,
                )
                row_max_reg = rr.vdup(row_max_reg, mask=mask)
                if const_expr(first and not has_sink):
                    if const_expr(self.has_sinks):
                        prior_max_reg = rr.vdups(
                            (-3.4028234663852886e38), dtypes.float32
                        )
                    else:
                        prior_max_reg = rr.vreinterpret(
                            rr.vdups(0xFF800000, dtypes.uint32), dtypes.float32
                        )
                else:
                    prior_max_reg = rr.vload_broadcast(
                        sm_max, softmax_state_offset + 2 * query_head_row
                    )
                row_max_reg = rr.vmax(row_max_reg, prior_max_reg, mask=mask)
                squeezed = rr.vsqueeze_and_storeunalign_init(
                    row_max_reg, mask=two_lanes
                )
                rr.vsqueeze_and_storeunalign(
                    sm_max, softmax_state_offset, squeezed, max_cursor
                )
            rr.vsqueeze_and_storeunalign_finalize(
                sm_max, softmax_state_offset, max_cursor
            )
            rr.vmem_bar("vst_vld")
            if const_expr(not first or has_sink or not self.has_sinks):
                row_max_reg = rr.vmuls(
                    rr.vload(sm_max, softmax_state_offset),
                    scale,
                    mask=mask,
                )
                rescale_factor_reg = rr.vexp_sub(old_max, row_max_reg, mask=mask)
                rr.vstore(
                    sm_exp,
                    0,
                    rescale_factor_reg,
                    state_mask,
                )
            sum_cursor = rr.vstore_unalign_begin(sm_sum)
            for query_head_row in cannbotdsl.range(
                dtypes.int32(0),
                dtypes.int32(head_rows),
                dtypes.int32(1),
                unroll=1,
            ):
                score_even_reg, score_odd_reg = rr.vload_deinterleave(
                    buf_score, query_head_row * TILE_N
                )
                if const_expr(not all_valid):
                    score_even_reg = rr.vselect(
                        neg_inf_reg, score_even_reg, cond_mask=invalid_even
                    )
                    score_odd_reg = rr.vselect(
                        neg_inf_reg, score_odd_reg, cond_mask=invalid_odd
                    )
                row_max_reg = rr.vmuls(
                    rr.vload_broadcast(
                        sm_max, softmax_state_offset + 2 * query_head_row
                    ),
                    scale,
                    mask=mask,
                )
                exp_even_reg = rr.vexp_sub(
                    rr.vmuls(score_even_reg, scale, mask=mask), row_max_reg, mask=mask
                )
                exp_odd_reg = rr.vexp_sub(
                    rr.vmuls(score_odd_reg, scale, mask=mask), row_max_reg, mask=mask
                )
                row_sum_reg = rr.vadd(
                    rr.vreduce_sum(exp_even_reg, mask=mask),
                    rr.vreduce_sum(exp_odd_reg, mask=mask),
                    mask=mask,
                )
                row_sum_reg = rr.vdup(row_sum_reg, mask=mask)
                squeezed = rr.vsqueeze_and_storeunalign_init(
                    row_sum_reg, mask=two_lanes
                )
                rr.vsqueeze_and_storeunalign(
                    sm_sum, softmax_state_offset, squeezed, sum_cursor
                )
                p_even_reg = rr.vcast(
                    exp_even_reg,
                    dtypes.bfloat16,
                    mask=mask,
                    rounding="rn",
                    reg_layout=rr.RegLayout.ZERO,
                )
                p_odd_reg = rr.vcast(
                    exp_odd_reg,
                    dtypes.bfloat16,
                    mask=mask,
                    rounding="rn",
                    reg_layout=rr.RegLayout.ONE,
                )
                packed_p_reg = rr.vadd(p_even_reg, p_odd_reg, mask=mask16)
                rr.vstore_strided(
                    buf_p,
                    query_head_row * _NZ_ROW,
                    packed_p_reg,
                    mask16,
                    block_stride=self.tile_vec_m + 1,
                    repeat_stride=0,
                )
            rr.vsqueeze_and_storeunalign_finalize(
                sm_sum, softmax_state_offset, sum_cursor
            )
            if const_expr(not first or has_sink or not self.has_sinks):
                rr.vmem_bar("vst_vld")
                row_sum_reg = rr.vmadd(
                    old_sum,
                    rescale_factor_reg,
                    rr.vload(sm_sum, softmax_state_offset),
                    mask=mask,
                )
                rr.vstore(sm_sum, softmax_state_offset, row_sum_reg, state_mask)

    @jit
    def finalize_softmax_state(self, sum_idx: int, scale):
        # Compact duplicated head state; retain scaled max/sum for FD and LSE.
        sm_max = self.sm_max_tb[sum_idx]
        sm_sum = self.sm_sum_tb[sum_idx]
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            maximum = rr.vload(sm_max, 0)
            total = rr.vload(sm_sum, 0)
            maximum, unused = rr.vdeinterleave(maximum, maximum)
            total, unused = rr.vdeinterleave(total, total)
            rr.vstore(sm_max, 0, rr.vmuls(maximum, scale, mask=mask), mask)
            rr.vstore(sm_sum, 0, total, mask)
            rr.vmem_bar("vst_vld")

    @jit
    def update_o(
        self, half, first, last, exp_idx: int, sum_idx: int, pack_output=False
    ):
        sm_exp = self.sm_exp_tb[exp_idx]
        sm_sum = self.sm_sum_tb[sum_idx]
        accumulator = self.result_views[half * 2]
        partial = self.result_views[half * 2 + 1]
        out = self.out_halves[half]
        head_rows = self.tile_vec_m
        if const_expr(not self.full_heads):
            head_rows = max(self.head_count(), 1)
        with vf(mode="simd"):
            mask = rr.full_mask(elem_bits=32)
            for row in range(head_rows):
                alpha = rr.vdups(1.0, dtypes.float32)
                denorm = rr.vdups(1.0, dtypes.float32)
                if const_expr(not first):
                    alpha = rr.vload_broadcast(sm_exp, row * 2)
                if const_expr(last):
                    denorm = rr.vdiv(
                        rr.vdups(1.0, dtypes.float32),
                        rr.vload_broadcast(sm_sum, row),
                        mask=mask,
                    )
                for col in range_constexpr(4):
                    offset = row * 256 + col * 64
                    if const_expr(first):
                        value = rr.vload(accumulator, offset)
                    else:
                        value = rr.vload(partial, offset)
                        old = rr.vmul(rr.vload(accumulator, offset), alpha, mask=mask)
                        value = rr.vadd(old, value, mask=mask)
                    if const_expr(last):
                        value = rr.vmul(value, denorm, mask=mask)
                    if const_expr(pack_output):
                        packed = rr.vcast(
                            value, dtypes.bfloat16, mask=mask, rounding="rn"
                        )
                        rr.vstore_pack(
                            out,
                            row * 512 + col * 64,
                            packed,
                            mask,
                            pack_mode=rr.PackMode.B32_TO_B16,
                        )
                    else:
                        rr.vstore(accumulator, offset, value, mask)

    def finalize_o(self, o_tile_gm, half, sum_idx: int, lse_gm=None, lse_base=None):
        # Copy each completed half immediately. The output RingBuffer owns the
        # V-to-MTE3 dependency and protects this alias from the next PV write.
        if const_expr(half == 0 and lse_gm is not None):
            self._finalize_lse_vf(lse_gm, lse_base, sum_idx)
        if const_expr(self.full_heads):
            heads = tile_slice(
                o_tile_gm, (self.tile_vec_m, self.tile_d), (self.subblock_idx, 0)
            )
            mem_copy(
                tile_slice(heads, (self.tile_vec_m, 256), (0, half)),
                self.out_halves[half],
                options=copy_options(l2_cache_ctl=4),
            )
        else:
            self._finalize_tail_o(o_tile_gm, half)

    @jit
    def head_count(self):
        return (
            (self.num_heads + 1) // 2 if self.subblock_idx == 0 else self.num_heads // 2
        )

    def head_start(self):
        return self.subblock_idx * ((self.num_heads + 1) // 2)

    @jit
    def load_sinks(self, sinks_gm):
        if const_expr(self.full_heads):
            mem_copy(
                self.sinks_ub,
                tile_slice(sinks_gm, (1, self.sinks_span), (0, self.subblock_idx)),
            )
        else:
            count = max(self.head_count(), 1)
            h0 = min(self.head_start(), self.num_heads - 1)
            mem_copy(self.sinks_ub[0:1, 0:count], sinks_gm[0:1, h0 : h0 + count])

    @jit
    def _finalize_tail_o(self, o_tile_gm, half):
        count = self.head_count()
        if count > 0:
            h0 = self.head_start()
            mem_copy(
                o_tile_gm[h0 : h0 + count, half * 256 : (half + 1) * 256],
                self.out_halves[half][0:count, 0:256],
                options=copy_options(l2_cache_ctl=4),
            )

    @jit
    def _store_lse(self, lse_gm, lse_base, source):
        if const_expr(self.full_heads):
            mem_copy(
                tile_slice(lse_gm, (self.tile_vec_m,), (lse_base,)),
                from_tensor(source, layout=make_layout((self.tile_vec_m,))),
            )
        else:
            count = self.head_count()
            if count > 0:
                mem_copy(
                    lse_gm[lse_base : lse_base + count,],
                    from_tensor(source, layout=make_layout((self.tile_vec_m,)))[
                        0:count,
                    ],
                )

    @jit
    def finalize_empty(
        self, o_tile_gm, sinks_gm, lse_gm=None, lse_base=None, zero_lse=False
    ):
        # Empty rows produce O=0 and LSE=sink through a separate zero buffer.
        count = self.tile_vec_m
        if const_expr(self.full_heads):
            half = tile_slice(
                o_tile_gm, (self.tile_vec_m, self.tile_d), (self.subblock_idx, 0)
            )
        else:
            count = self.head_count()
            h0 = self.head_start()
            half = o_tile_gm[h0 : h0 + count, 0 : self.tile_d]
        empty_out = Buffer(MemLoc.UB, dtypes.bfloat16, make_layout((1, self.tile_d)))
        with vf(mode="simd"):
            full, _ = rr.update_mask(128, elem_bits=16)
            zero = rr.vdups(0.0, dtypes.bfloat16, mask=full)
            for col in range(0, self.tile_d, 128):
                rr.vstore(empty_out, col, zero, full)
        for row in range(count):
            mem_copy(tile_slice(half, (1, self.tile_d), (row, 0)), empty_out)
        if const_expr(lse_gm is not None):
            # Keep lse_ub's producer on V, as in _finalize_lse_vf; using
            # MTE2 here would change the shared buffer's event protocol.
            if const_expr(zero_lse):
                with vf(mode="simd"):
                    mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
                    rr.vstore(self.lse_ub, 0, rr.vdups(0.0, dtypes.float32), mask)
            elif const_expr(self.has_sinks):
                self.load_sinks(sinks_gm)
                with vf(mode="simd"):
                    mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
                    rr.vstore(self.lse_ub, 0, rr.vload(self.sinks_ub, 0), mask)
            else:
                with vf(mode="simd"):
                    mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
                    empty_lse = rr.vreinterpret(
                        rr.vdups(0xFF800000, dtypes.uint32), dtypes.float32
                    )
                    rr.vstore(self.lse_ub, 0, empty_lse, mask)
            self._store_lse(lse_gm, lse_base, self.lse_ub)

    # Keep each FD VF in a separate JIT method: register/mask locals must not
    # become loop-carried scalar values in a surrounding runtime loop.

    @jit
    def stage_partial(self, fd_o, fd_ms, slot, sum_idx: int):
        # Stage the normalized FP32 partial O and its max/sum for this AIV.
        # The caller normalizes O with update_o before staging.
        # O is staged as two 32 KiB halves; max/sum reuse the 32-FP32 lse buffer.
        # Workspace layout: O[slots*64,512], followed by max/sum[slots*128].
        # Each statistics slot contains max[0:64] followed by sum[0:64].
        sub = self.subblock_idx
        for half in range_constexpr(2):
            mem_copy(
                tile_slice(
                    fd_o,
                    (self.tile_vec_m, 256),
                    (slot * (N1 // self.tile_vec_m) + sub, half),
                ),
                self.result_views[half * 2],
            )
        self._fd_ms_to_ub(0, sum_idx)
        mem_copy(
            tile_slice(
                fd_ms,
                (self.tile_vec_m,),
                (slot * (FD_MS_ELEMS // self.tile_vec_m) + sub,),
            ),
            from_tensor(self.lse_ub, layout=make_layout((self.tile_vec_m,))),
        )
        self._fd_ms_to_ub(1, sum_idx)
        mem_copy(
            tile_slice(
                fd_ms,
                (self.tile_vec_m,),
                (
                    slot * (FD_MS_ELEMS // self.tile_vec_m)
                    + (N1 // self.tile_vec_m)
                    + sub,
                ),
            ),
            from_tensor(self.lse_ub, layout=make_layout((self.tile_vec_m,))),
        )

    @jit
    def fd_reduce(
        self, fd_o, fd_ms, fd_ub, out_gm, lse_gm, row, slot0, k, sub, h0, count
    ):
        # Reduce one staged half and copy only the library-assigned head range.
        # M = max(m_p); t_p = exp(m_p-M)*s_p; G = sum(t_p).
        # O = sum((t_p/G)*O_p); LSE = M + log(G).
        # Accumulate in metadata slot order for deterministic reassociation.
        # After the barrier, reuse main-loop UB for max/sum and the output accumulator.
        # weights remain in registers; res_o accumulates, fd_ub receives O parts.
        local_start = h0 - sub * ((self.num_heads + 1) // 2)
        self._fd_init_acc()
        ms_offset = slot0 * FD_MS_ELEMS
        ms_count = k * FD_MS_ELEMS
        ms_source = fd_ms[ms_offset : ms_offset + ms_count,].view(k, FD_MS_ELEMS)
        mem_copy(
            self.fd_ms_ub.next_write(),
            tile_slice(ms_source, (_FD_MP_SLOTS, FD_MS_ELEMS), (0, 0)),
        )
        ms_buf = self.fd_ms_ub.next_read()
        for p in range(k):
            self._fd_acc_max_save(p, ms_buf, sub)
        for p in range(k):
            self._fd_acc_sum(p, ms_buf, sub)
        # Optionally store LSE = M + log(G).
        if const_expr(lse_gm is not None):
            self._fd_lse_to_ub(self.lse_ub.next_write(), local_start, count)
            lse_base = row * self.num_heads + h0
            mem_copy(
                lse_gm[lse_base : lse_base + count,],
                from_tensor(
                    self.lse_ub.next_read(), layout=make_layout((self.tile_vec_m,))
                )[0:count,],
            )
        # Accumulate normalized partials using cached weights t_p/G.
        mem_copy(
            fd_ub.next_write(),
            tile_slice(
                fd_o,
                (self.tile_vec_m, self.tile_d),
                (slot0 * (N1 // self.tile_vec_m) + sub, 0),
            ),
        )
        self._fd_acc_full(fd_ub.next_read(), 0, True)
        for p in range(1, k):
            mem_copy(
                fd_ub.next_write(),
                tile_slice(
                    fd_o,
                    (self.tile_vec_m, self.tile_d),
                    ((slot0 + p) * (N1 // self.tile_vec_m) + sub, 0),
                ),
            )
            self._fd_acc_full(fd_ub.next_read(), p, False)
        # Cast the normalized result in place and write its 32-head tile.
        # The pre-reduction barrier drained old MTE3 operations; the following
        # V-to-MTE3 dependency still protects this newly computed output.
        self._cast_output()
        output_start = row * self.num_heads + h0
        mem_copy(
            out_gm[output_start : output_start + count, 0 : self.tile_d],
            self.out_view[local_start : local_start + count, 0 : self.tile_d],
        )

    @jit
    def _finalize_lse_vf(self, lse_gm, lse_base, sum_idx: int):
        # LSE = log(sum) + max; the framework tracks Buffer V-to-MTE3 accesses.
        sm_max_buf = self.sm_max_tb[sum_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            total = rr.vload(sm_sum_buf, 0)
            maximum = rr.vload(sm_max_buf, 0)
            rr.vstore(
                self.lse_ub,
                0,
                rr.vadd(rr.vlog(total, mask=mask), maximum, mask=mask),
                mask,
            )
        self._store_lse(lse_gm, lse_base, self.lse_ub)

    @jit
    def _cast_output(self):
        # Cast FP32 res_o in place to the first 32 KiB BF16 output view.
        # Each 64-element read precedes its packed write; that write ends before
        # the next unread FP32 chunk, preserving forward traversal safety.
        with vf(mode="simd"):
            full = rr.full_mask()
            for offset in range(0, self.tile_vec_m * self.tile_d, 64):
                value = rr.vload(self.res_o, offset)
                result = rr.vcast(value, dtypes.bfloat16, mask=full)
                rr.vstore_pack(
                    self.out_view,
                    offset,
                    result,
                    full,
                    pack_mode=rr.PackMode.B32_TO_B16,
                )

    @jit
    def _fd_ms_to_ub(self, which: int, sum_idx: int):
        # Copy max (which=0) or sum (which=1) to the LSE staging buffer.
        if const_expr(which == 0):
            buf = self.sm_max_tb[sum_idx]
        else:
            buf = self.sm_sum_tb[sum_idx]
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            rr.vstore(self.lse_ub, 0, rr.vload(buf, 0), rowmask)

    @jit
    def _fd_init_acc(self):
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            if const_expr(self.has_sinks):
                rr.vstore(
                    self.sm_max_tb[0],
                    0,
                    rr.vdups(_FD_NEG_INF, dtypes.float32, mask=rowmask),
                    rowmask,
                )
            else:
                initial_max = rr.vreinterpret(
                    rr.vdups(0xFF800000, dtypes.uint32), dtypes.float32
                )
                rr.vstore(self.sm_max_tb[0], 0, initial_max, rowmask)
            rr.vstore(
                self.sm_sum_tb[0],
                0,
                rr.vdups(0.0, dtypes.float32, mask=rowmask),
                rowmask,
            )
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_max_save(self, p: int, lse_buf, sub):
        # Cache m_p and accumulate the global maximum; the next pass reuses
        # the same weight slot for t_p = s_p * exp(m_p - M).
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (p, 0))
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            mp = rr.vload(lse_buf, p * FD_MS_ELEMS + sub * self.tile_vec_m + 0)
            nm = rr.vmax(rr.vload(self.sm_max_tb[0], 0), mp, mask=rowmask)
            rr.vstore(self.sm_max_tb[0], 0, nm, rowmask)
            rr.vstore(mp_buf, 0, mp, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_sum(self, p: int, lse_buf, sub):
        # Bulk max/sum landing: t_p = s_p·exp(m_p−M); G += t_p. Cache t_p.
        # Reuse cached m_p for t_p, then form t_p/G during output accumulation.
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (p, 0))
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            e = rr.vexp_sub(
                rr.vload(mp_buf, 0), rr.vload(self.sm_max_tb[0], 0), mask=rowmask
            )
            t = rr.vmul(
                rr.vload(lse_buf, p * FD_MS_ELEMS + sub * self.tile_vec_m + 64),
                e,
                mask=rowmask,
            )
            acc = rr.vadd(rr.vload(self.sm_sum_tb[0], 0), t, mask=rowmask)
            rr.vstore(self.sm_sum_tb[0], 0, acc, rowmask)
            rr.vstore(mp_buf, 0, t, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_lse_to_ub(self, lse_buf, local_start, count):
        # lse_ub ← M + log(G)
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            lse = rr.vadd(
                rr.vlog(rr.vload(self.sm_sum_tb[0], 0), mask=rowmask),
                rr.vload(self.sm_max_tb[0], 0),
                mask=rowmask,
            )
            # MTE3 requires an aligned UB source, even when the library
            # assigns a single unaligned head. Gather to the buffer origin.
            selected = rr.vgather_reg(lse, rr.varange(local_start, dtypes.uint32))
            output_mask, _ = rr.update_mask(count, elem_bits=32)
            rr.vstore(lse_buf, 0, selected, output_mask)

    @jit
    def _fd_acc_full(self, fd_ub, p: int, first: cannbotdsl.Constexpr[bool]):
        # Compute each head weight in registers while consuming partial O.
        # Part zero initializes the accumulator; later parts retain slot order.
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (p, 0))
        with vf(mode="simd"):
            full = rr.full_mask()
            for r in range(self.tile_vec_m):
                wb = rr.vdiv(
                    rr.vload_broadcast(mp_buf, r),
                    rr.vload_broadcast(self.sm_sum_tb[0], r),
                    mask=full,
                )
                for col in range(0, self.tile_d, 2048 // 32):
                    off = r * self.tile_d + col
                    cur = rr.vload(fd_ub, off)
                    if const_expr(first):
                        value = rr.vmul(cur, wb, mask=full)
                    else:
                        pre = rr.vload(self.res_o, off)
                        value = rr.vmadd(cur, wb, pre, mask=full)
                    rr.vstore(self.res_o, off, value, full)
            rr.vmem_bar("vst_vld")

    def _setup_fd_reduce_buffers(self):
        # After the global barrier, reuse main-loop UB for one full partial O,
        # an FP32 accumulator, all max/sum rows and up to 36 partial weights.
        # Main-loop allocations and reduction allocations are not simultaneous.
        self.fd_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((self.tile_vec_m, self.tile_d)),
            depth=1,
        )
        self.fd_ms_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((_FD_MP_SLOTS, FD_MS_ELEMS)), depth=1
        )
        self.res_o = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, self.tile_d))
        )
        self.out_view = from_tensor(
            self.res_o,
            layout=make_layout((self.tile_vec_m, self.tile_d)),
            dtype=dtypes.bfloat16,
        )
        self.lse_ub = RingBuffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, 1)), depth=1
        )
        self.sm_max_tb[0] = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, 1))
        )
        self.sm_sum_tb[0] = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, 1))
        )
        self.tmp_new_max = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((self.tile_vec_m, 1))
        )
        self.fd_mp_tb = Buffer(
            MemLoc.UB, dtypes.float32, make_layout((_FD_MP_SLOTS * self.tile_vec_m, 1))
        )


@datastruct
class PipelineRingState:
    clock: CArray[dtypes.int64, 1]
    metadata: CArray[dtypes.int64, 64]


class SlotMajorDelayLineGroup:
    def __init__(self):
        self.state = PipelineRingState(clock=[0], metadata=[0] * 64)
        self.fields = ("m", "n", "t", "last", "first", "sum_slot", "valid", "ori_tiles")

    def push(self, **kwargs):
        metadata_slot_offset = self.state.clock[0] % 8 * 8
        for i, field in enumerate(self.fields):
            self.state.metadata[metadata_slot_offset + i] = dtypes.int64(kwargs[field])

    def tap(self, lag):
        metadata_slot_offset = (self.state.clock[0] + 8 - lag) % 8 * 8
        return SimpleNamespace(
            **{
                field: self.state.metadata[metadata_slot_offset + i]
                for i, field in enumerate(self.fields)
            }
        )

    def advance(self):
        self.state.clock[0] = (self.state.clock[0] + 1) % 8


@kernel(profile=ProfileSpec(name="mixed_quant_sparse_flash_mla"))
class MqsmlaKernel:
    """Device attention pipeline; tensor views and launch live in _run_mqsmla."""

    def __init__(
        self,
        tile_cube_m,
        tile_vec_m,
        tile_n,
        has_cmp=False,
        layout_kv="PA_BBND",
        full_heads=True,
    ):
        self.full_heads = full_heads
        self.layout_kv = layout_kv
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D
        self.has_cmp = has_cmp
        # Cross-core chains: QK (AIC to AIV, depth 1), P (AIV to AIC, depth 1),
        # and PV (AIC to AIV, depth 1). UB shapes cover one AIV; P covers both halves.

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()

        self.sinks_span = tile_vec_m

        self.matmul = Matmul(tile_cube_m, tile_vec_m, tile_n, full_heads=full_heads)
        self.p_l1 = self.matmul.p_l1
        self.kvcache = Kvcache()

    def __call__(
        self,
        out_gm: Tensor,
        q_gm: Tensor,
        sinks_gm: Tensor,
        pa_phys_gm: Tensor = None,
        pa_bt_gm: Tensor = None,
        ws_gm: Tensor = None,
        pa_phys_cmp_gm: Tensor = None,
        pa_bt_cmp_gm: Tensor = None,
        cmp_idx_gm: Tensor = None,
        ori_idx_gm: Tensor = None,
        cu_seqlens_q: Tensor = None,
        softmax_lse: Tensor = None,
        ori_topk_length: Tensor = None,
        cmp_topk_length: Tensor = None,
        metadata_gm: Tensor = None,
        softmax_scale: dtypes.float32 = 1.0,
        batch_consistency: dtypes.int32 = 0,
        addr_ws_gm: Tensor = None,
        cu_ori_kv: Tensor = None,
        cu_cmp_kv: Tensor = None,
        seqused_q: Tensor = None,
    ):
        # Workspace capacity determines the launch grid and FD slot count.
        self._b_rt = cu_seqlens_q.shape[1] - 1
        self._scale = softmax_scale
        self._batch_consistency = batch_consistency
        self._ori_page_stride = pa_phys_gm.stride[0]
        self._pa_bs_rt = pa_phys_gm.shape[1]
        if const_expr(self.has_cmp):
            self._cmp_page_stride = pa_phys_cmp_gm.stride[0]
            self._pa_cmp_bs_rt = pa_phys_cmp_gm.shape[1]
        self._ws = ws_gm
        self._q = q_gm
        self._sinks = sinks_gm
        if const_expr(sinks_gm is None):
            # The no-sink entry retains Q's head axis through the launch.
            # Shape information no longer depends on a placeholder tensor.
            self._num_heads = q_gm.shape[1]
            self._q = q_gm.view(q_gm.shape[0] * self._num_heads, D)
        self._out = out_gm
        self._lse = softmax_lse
        self._ori_topk_length = ori_topk_length
        self._cmp_topk_length = cmp_topk_length

        self._seqused_q = seqused_q
        self._cu_q = cu_seqlens_q
        self._cu_ori_kv, self._cu_cmp_kv = cu_ori_kv, cu_cmp_kv

        self._pa_phys, self._pa_bt = pa_phys_gm, pa_bt_gm
        self._ori_idx = ori_idx_gm
        if const_expr(self.has_cmp):
            self._pa_phys_cmp, self._pa_bt_cmp = pa_phys_cmp_gm, pa_bt_cmp_gm
            self._cmp_idx = cmp_idx_gm
        # Address-prepass storage exists only in the vector-addressing template.
        _avec = _addr_vec_mode()
        if const_expr(_avec != "off"):
            self._avec_bs_ori = pa_phys_gm.shape[1]
            self._avec_k1 = ori_idx_gm.shape[1]
            self._avec_k1_tab = _addr_tab_w(self._avec_k1)
            self._avec_k2_tab = 0
            if const_expr(self.has_cmp):
                self._avec_bs_cmp = pa_phys_cmp_gm.shape[1]
                self._avec_k2 = cmp_idx_gm.shape[1]
                self._avec_k2_tab = _addr_tab_w(self._avec_k2)
            if const_expr(_avec == "arena"):
                self._addr_ub = Buffer(
                    MemLoc.UB,
                    dtypes.int32,
                    make_layout((1, _ADDR_UB_BYTES // 4)),
                )
            else:
                # Distinct owners allow the next index load to overlap the
                # preceding address store. No large-pool capacity is lost:
                # the host dispatch selects the arena kernel outside this range.
                self._ch_addr_bt = Buffer(
                    MemLoc.UB,
                    dtypes.int32,
                    make_layout((1, _ADDR_COMPACT_BT)),
                )
                self._ch_addr_sp = Buffer(
                    MemLoc.UB,
                    dtypes.int32,
                    make_layout((1, _ADDR_COMPACT_K)),
                )
                self._ch_addr_out = Buffer(
                    MemLoc.UB,
                    dtypes.int32,
                    make_layout((1, _ADDR_COMPACT_K)),
                )
            tab_cols = self._avec_k1_tab + (self._avec_k2_tab if self.has_cmp else 0)
            self._addr_tab = addr_ws_gm.view(dtypes.int32).view(
                addr_ws_gm.shape[0] // 4 // tab_cols, tab_cols
            )
            self._addr_bar_flag = 9
            # Both AIVs share an intra-sync ID; AIV1 uses the fixed ID offset on the AIC side.
            self._addr_bar_sid = 9

        # FA metadata follows the AscendC field layout.
        # A core owns rows [m_start,m_end], excluding m_end when s2_end is zero.
        # Boundary rows use the recorded S2 tile limits; zero denotes a full boundary.
        # Consecutive cores partition one contiguous (row,tile) interval.
        # Disabled cores have zero rows and only participate in required barriers.
        fa = self.block_idx * FA_METADATA_SIZE
        bn2_start = _i64(metadata_gm, fa + FA_BN2_START_INDEX)
        m_start_g = _i64(metadata_gm, fa + FA_M_START_INDEX)
        bn2_end = _i64(metadata_gm, fa + FA_BN2_END_INDEX)
        m_end_g = _i64(metadata_gm, fa + FA_M_END_INDEX)
        self._m_start = _i64(self._cu_q, 0, bn2_start) + m_start_g
        self._s2_start = _i64(metadata_gm, fa + FA_S2_START_INDEX)
        m_end = _i64(self._cu_q, 0, bn2_end) + m_end_g
        s2_end = _i64(metadata_gm, fa + FA_S2_END_INDEX)
        self._first_fd = _i64(metadata_gm, fa + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX)
        row_count = m_end - self._m_start + (1 if s2_end > 0 else 0)
        self._first_split_row = (
            dtypes.int64(0) if self._s2_start > 0 else dtypes.int64(-1)
        )
        self._last_split_row = row_count - 1 if s2_end > 0 else dtypes.int64(-1)
        # Zero s2 boundaries encode whole-row execution. Compute tile bounds
        # directly; the global FD gate is read only after the main loop.
        if const_expr(_addr_vec_mode() != "off"):
            # Publish the address table before the main loop consumes it.
            self._addr_precompute(row_count)
            # AIC waits for both AIV writes before releasing the block.
            # Use MTE1 so the wait does not block Q prefetch on MTE2.
            cube_sync_intra_wait(PIPE.MTE1, self._addr_bar_sid)
            cube_sync_intra_wait(PIPE.MTE1, self._addr_bar_sid + _AIV1_ID_OFFSET)
            # Wait and release must use the same pipe to preserve ordering.
            cube_sync_block_arrive(PIPE.MTE1, self._addr_bar_flag, mode=2)

            # _addr_precompute's MTE3 arrive follows the last workspace store;
            # its S wait observes both AIV completions through the AIC relay.
            # All address Buffer reads/writes are therefore complete before any
            # main-loop UB use. Rewind reclaims storage, not synchronization.
            onchip_rewind(reset_sync_id=False)

        self._setup_main_ub()

        if is_aic():
            self.pipeline_cube(row_count, s2_end)
        else:
            self.pipeline_vector(row_count, s2_end)

        # Empty queries never enter the delay line. Write their outputs after
        # draining so the shared output buffer cannot disturb pending PVs.
        for m_seq in range(0, row_count):
            m_idx = self._m_idx(m_seq)
            if self._n_end_rt(m_idx) == 0:
                o_tile = self.output_view(m_idx)
                lse_base = (
                    m_idx * 2 + self.subblock_idx
                    if const_expr(self._lse is not None)
                    else None
                )
                if const_expr(self._lse is not None and not self.full_heads):
                    lse_base = m_idx * self.num_heads() + self.subblock_idx * (
                        (self.num_heads() + 1) // 2
                    )
                if const_expr(self._seqused_q is not None):
                    if self._query_valid(m_idx):
                        self.vector.finalize_empty(
                            o_tile, self._sinks, self._lse, lse_base
                        )
                    else:
                        self.vector.finalize_empty(
                            o_tile, self._sinks, self._lse, lse_base, True
                        )
                else:
                    self.vector.finalize_empty(o_tile, self._sinks, self._lse, lse_base)

        # The library packs active FD entries from index zero. Its first
        # enable bit is the uniform gate; metadata remains read-only.
        fd_any = _i64(metadata_gm, FD_METADATA_BASE + FD_CORE_ENABLE_INDEX)
        if fd_any != 0:
            global_sync_all()
            # After the barrier, reclaim main-loop UB for the full (32,512) FP32 reduction landing.
            onchip_rewind(reset_sync_id=False)
            self.vector._setup_fd_reduce_buffers()
            aiv_idx = self.block_idx * get_subblock_dim() + self.subblock_idx
            fd_metadata_base = FD_METADATA_BASE + aiv_idx * FD_METADATA_SIZE
            fd_enable = _i64(metadata_gm, fd_metadata_base + FD_CORE_ENABLE_INDEX)
            if fd_enable != 0:
                fd_bn2 = _i64(metadata_gm, fd_metadata_base + FD_BN2_IDX_INDEX)
                fd_m = _i64(metadata_gm, fd_metadata_base + FD_M_IDX_INDEX)
                fd_row = _i64(self._cu_q, 0, fd_bn2) + fd_m
                fd_ws = _i64(metadata_gm, fd_metadata_base + FD_WORKSPACE_IDX_INDEX)
                fd_k = _i64(metadata_gm, fd_metadata_base + FD_WORKSPACE_NUM_INDEX)
                fd_h0 = _i64(metadata_gm, fd_metadata_base + FD_M_START_INDEX)
                fd_o, fd_ms = self._fd_workspace_views()
                fd_heads = _i64(metadata_gm, fd_metadata_base + FD_M_NUM_INDEX)
                # Library FD scheduling may assign arbitrary head ranges,
                # including ranges that cross the two staged AIV halves.
                head = fd_h0
                head_end = fd_h0 + fd_heads
                half_heads = (self.num_heads() + 1) // 2
                while head < head_end:
                    sub = dtypes.int64(0) if head < half_heads else dtypes.int64(1)
                    chunk_end = min(
                        head_end, half_heads if sub == 0 else self.num_heads()
                    )
                    self.vector.fd_reduce(
                        fd_o,
                        fd_ms,
                        self.vector.fd_ub,
                        self._out,
                        self._lse,
                        fd_row,
                        fd_ws,
                        fd_k,
                        sub,
                        head,
                        chunk_end - head,
                    )
                    head = chunk_end

    @staticmethod
    def _scale_idx(grp_lane, base: int, m16):
        # Scale gather uses uint16 indices, so its arithmetic requires a 16-bit mask.
        return rr.vadd(grp_lane, rr.vdups(base, dtypes.uint16), mask=m16)

    @jit
    def _setup_main_ub(self):
        # Allocate only after address-prepass storage is dead. L1/L0 and their
        # cross-core protocols remain live across rewind; do not reset IDs.
        self._ori_l2_hint, self._cmp_l2_hint = self.kv_l2_hints()
        tile_vec_m, tile_n = self.tile_vec_m, self.tile_n
        # 两个 256 列半区分别交接，每个 channel 的前/后 32 行为累加区/部分结果。
        self.result_channels = [
            RingBuffer(
                MemLoc.UB,
                dtypes.float32,
                make_layout((2 * tile_vec_m, 256)),
                depth=1,
                data_format="nd",
            )
            for half in range(2)
        ]
        self.qk_ub = RingBuffer(
            MemLoc.UB,
            dtypes.float32,
            make_layout((tile_vec_m, tile_n)),
            depth=1,
            data_format="nd",
        )
        self.p_out = Buffer(
            MemLoc.UB, dtypes.bfloat16, make_layout(((tile_vec_m + 1) * tile_n,))
        )
        self.vector = Vector(
            tile_vec_m,
            tile_n,
            self.tile_d,
            self.subblock_idx,
            sinks_span=self.sinks_span,
            full_heads=self.full_heads,
            num_heads=self.num_heads(),
            has_sinks=(self._sinks is not None),
        )

        if const_expr(self._sinks is not None):
            self.vector.load_sinks(self._sinks)

        # The existing scalar PA path accepts runtime page sizes and does not
        # allocate an on-chip buffer proportional to the block-table capacity.
        ch_rows = _TILE_ROWS
        # gather 领先反量化一拍；先填当前 lower 两槽，再消费上一拍 lower，最多六槽在途。
        self.ch_batch_bytes = RingBuffer(
            MemLoc.UB, dtypes.uint8, make_layout((2 * ch_rows, _COMBINE_DIM)), depth=3
        )
        # DMA and VF share one byte RingBuffer; interpret scale bits in registers.
        # Do not create a second typed RingBuffer for the same slot.
        self._v0_out = RingBuffer(
            MemLoc.UB,
            dtypes.bfloat16,
            make_layout((D // 16, _NZ_PAD_ROWS, 16)),
            depth=2,
        )

    # m_seq is the core-local query index; m_idx is the global query index.
    # n spans ORI then CMP tiles; tick increases continuously across queries.
    # Each pool uses its valid prefix for both gather and softmax masking.
    # CMP tile coordinates subtract the ORI tile count.
    # S2 start values are sparse-index column offsets, not token values.

    def _m_idx(self, m_seq: int):
        # Map the local query index with the preloaded metadata start.
        return self._m_start + m_seq

    @jit
    def _batch_of(self, m_idx: int):
        # Dynamic upper-bound search over the B query spans.
        lo, hi = 0, self._b_rt
        while lo < hi:
            mid = (lo + hi) // 2
            take = _i64(self._cu_q, 0, mid + 1) <= m_idx
            lo = mid + 1 if take else lo
            hi = hi if take else mid
        return lo

    @jit
    def _query_valid(self, m_idx: int):
        batch = self._batch_of(m_idx)
        return m_idx - _i64(self._cu_q, 0, batch) < _i64(self._seqused_q, batch)

    @jit
    def _bound_after(self, batch: int, b_last: int):
        # Read the next batch boundary once. At the final span, return a sentinel
        # instead of reading beyond cu_q; this matches _batch_of clamping.
        nxt = batch + 1 if batch < b_last else b_last
        v = _i64(self._cu_q, 0, nxt)
        return v if batch < b_last else dtypes.int64(_ADDR_M_INF)

    def _cmp_tiles_rt(self, m_idx: int):
        w = self._csa_valid_w(m_idx)
        return (w + self.tile_n - 1) // self.tile_n

    @jit
    def _ori_valid_w(self, m_idx: int):
        # ORI width is min(K1, topk_length), or K1 when lengths are absent.
        # Promote it to runtime Int64 even without lengths so FD tile-bound
        # predicates remain SSA values rather than Python booleans.
        if const_expr(self._ori_topk_length is not None):
            return min(self._ori_idx.shape[1], _i64(self._ori_topk_length, m_idx))
        return dtypes.int64(self._ori_idx.shape[1])

    @jit
    def _cmp_topk_len(self, m_idx: int):
        if const_expr(self._cmp_topk_length is not None):
            return min(self._cmp_idx.shape[1], _i64(self._cmp_topk_length, m_idx))
        return dtypes.int64(self._cmp_idx.shape[1])

    def _ori_tiles_rt(self, m_idx: int):
        w = self._ori_valid_w(m_idx)
        return (w + self.tile_n - 1) // self.tile_n

    @jit
    def _padding_n_end_rt(self, m_idx: int):
        tiles = self._ori_tiles_rt(m_idx)
        if const_expr(self.has_cmp):
            tiles = tiles + self._cmp_tiles_rt(m_idx)
        return tiles if self._query_valid(m_idx) else dtypes.int64(0)

    def _n_end_rt(self, m_idx: int):
        if const_expr(self._seqused_q is not None):
            return self._padding_n_end_rt(m_idx)
        # Runtime query tile count: ORI tiles plus CMP tiles.
        ori_tiles = self._ori_tiles_rt(m_idx)
        if const_expr(not self.has_cmp):
            return ori_tiles
        return ori_tiles + self._cmp_tiles_rt(m_idx)

    @jit
    def _actual_n(self, m_idx: int, n):
        # Return the selected pool tile width for exponent and P masks.
        # The unselected conditional width does not describe an actual task.
        ori_w = self._ori_width(m_idx, n)
        if const_expr(not self.has_cmp):
            return ori_w
        ori_tiles = self._ori_tiles_rt(m_idx)
        cmp_tile_index = n - ori_tiles
        cmp_w = self._cmp_valid(m_idx, cmp_tile_index)
        return cmp_w if n >= ori_tiles else ori_w

    @jit
    def _ori_width(self, m_idx: int, n):
        w = self._ori_valid_w(m_idx)
        return min(self.tile_n, w - n * self.tile_n)

    def _csa_valid_w(self, m_idx: int):
        return self._cmp_topk_len(m_idx)

    @jit
    def _cmp_valid(self, m_idx: int, c):
        return min(self.tile_n, self._csa_valid_w(m_idx) - c * self.tile_n)

    @jit
    def _antiquant_vf_fp8_g32(self, kv_fp8, out):
        # ORI byte order is nope[448], rope[64], then 16 BF16 scales.
        # Dequantize four 128-value chunks through FP32 to BF16 and compact even lanes.
        # Each chunk uses four group-32 scales and writes matching NZ columns.
        subtile_rows = _TILE_ROWS
        scale_view = from_tensor(
            kv_fp8,
            layout=make_layout((_TILE_ROWS, _ROW_BF16)),
            dtype=dtypes.bfloat16,
            data_format="nd",
        )
        with vf(mode="simd"):
            m32, _ = rr.update_mask(_TILE_SIZE, elem_bits=32)
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            m8, _ = rr.update_mask(256, elem_bits=8)
            row_bf16, row_fp8 = _ROW_BF16, _COMBINE_DIM

            # 每个 token 只读取一次 inline scale，四个 D chunk 复用寄存器。
            for i in range(subtile_rows):
                scale_a = rr.vreinterpret_lanes(
                    rr.vload_broadcast(
                        scale_view,
                        _SCALE_BF16_OFF + i * row_bf16,
                        mode="elem2datablock",
                    ),
                    dtypes.bfloat16,
                )
                scale_b = rr.vreinterpret_lanes(
                    rr.vload_broadcast(
                        scale_view,
                        _SCALE_BF16_OFF + i * row_bf16 + 8,
                        mode="elem2datablock",
                    ),
                    dtypes.bfloat16,
                )
                s0, s1 = rr.vinterleave(scale_a, scale_a)
                s2, s3 = rr.vinterleave(scale_b, scale_b)
                scale_vectors = (s0, s1, s2, s3)
                for j in tuple(range(D // _KV_NUM_PER_LOOP)):
                    src_j = j * _KV_NUM_PER_LOOP
                    dst_j = src_j
                    s = i * row_fp8 + src_j
                    unpacked = rr.vload_unpack(kv_fp8, s, unpack_mode="b8_to_b16")
                    fp8 = rr.vreinterpret_lanes(unpacked, dtypes.float8_e4m3fn)
                    even = rr.vcast(
                        fp8, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO
                    )
                    odd = rr.vcast(
                        fp8, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.TWO
                    )
                    # 有限 FP8 精确扩展到 FP32 后低16位为零；合并相邻 BF16。
                    even_bits = rr.vshr(
                        rr.vreinterpret_lanes(even, dtypes.uint32), 16, mask=m32
                    )
                    packed_bits = rr.vbitwise_or(
                        even_bits, rr.vreinterpret_lanes(odd, dtypes.uint32), mask=m32
                    )
                    raw = rr.vreinterpret_lanes(packed_bits, dtypes.bfloat16)
                    svec = scale_vectors[j]
                    packed = rr.vmul(raw, svec, mask=m16)
                    self._v0_store(out, i, dst_j, packed, m16)

    @jit
    def _antiquant_vf_fp4_g16(self, kv_fp8, out):
        # CMP layout: 224 nope bytes, 32 rope bytes, and 64 scale bytes.
        # UNPACK4 + packed FP4 reinterpret + quarter-0 cast.
        # Packed FP4 consumes an 8-bit mask; each byte expands to two BF16 values.
        # Scales follow groups of 16 features, and _v0_store writes NZ17 faces.
        subtile_rows = _TILE_ROWS
        scale_view = from_tensor(
            kv_fp8,
            layout=make_layout((_TILE_ROWS, _ROW_BF16)),
            dtype=dtypes.bfloat16,
            data_format="nd",
        )
        with vf(mode="simd"):
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            row_fp8 = _COMBINE_DIM
            scale_off = SCALE_BF16_OFF_CMP

            # Gather each FP4 group scale from the byte buffer and interpret it as BF16.
            fm = rr.full_mask()
            fm.elem_bits = 8
            for i in range(subtile_rows):
                for p in tuple(range(D // _KV_NUM_PER_LOOP)):
                    src_p = p * _TILE_SIZE
                    dst_p = p * _KV_NUM_PER_LOOP
                    u = rr.vload_unpack(
                        kv_fp8, src_p + i * row_fp8, unpack_mode=rr.UnpackMode.UNPACK4
                    )
                    f4 = rr.vreinterpret_lanes(u, dtypes.fp4x2_e2m1)
                    vals = rr.vcast(
                        f4, dtypes.bfloat16, mask=fm, reg_layout=rr.RegLayout.ZERO
                    )
                    scales = rr.vreinterpret_lanes(
                        rr.vload_broadcast(
                            scale_view,
                            scale_off + i * _ROW_BF16 + 8 * p,
                            mode="elem2datablock",
                        ),
                        dtypes.bfloat16,
                    )
                    packed = rr.vmul(vals, scales, mask=m16)
                    self._v0_store(out, i, dst_p, packed, m16)

    def _v0_store(self, out, i, col, val, mask):
        # i is the token row and col the D column; offsets use BF16 elements.
        # NZ17 address = (col//16)*(17*16) + i*16 + col%16.
        # Each 16-column store advances by 17 blocks of 32 bytes; row 17 is padding.
        rr.vstore_strided(
            out,
            (col // 16) * _NZ_CHUNK + i * _NZ_ROW,
            val,
            mask,
            block_stride=_NZ_PAD_ROWS,
            repeat_stride=0,
        )

    @jit
    def kv_l2_hints(self):
        # Keep ORI cached; prefill queries can reuse the CMP pool.
        ori_hint, cmp_hint = dtypes.int32(0), dtypes.int32(0)
        if const_expr(self.has_cmp and _addr_vec_mode() != "off"):
            query_count = self._q.shape[0] // self.tile_cube_m
            if const_expr(not self.full_heads):
                query_count = self._ori_idx.shape[0]
            cmp_hint = (
                dtypes.int32(0)
                if query_count >= self._b_rt * _KV_L2_PREFILL_MIN_QUERIES
                else dtypes.int32(4)
            )
        return ori_hint, cmp_hint

    @jit
    def gather_batch(
        self,
        raw_half,
        half,
        kv_pool,
        blk_table,
        blk_size,
        s2_start: int,
        n_real,
        idx_tab=None,
        idx_row=None,
        batch=None,
        is_cmp=False,
    ):
        # Each AIV handles four 16-token subblocks of a 128-token tile.
        # Pair DMA sorts KV and inline scales together; otherwise copy rows individually.
        # Tail rows repeat the last valid index; masked P excludes them from PV.
        # Both pools use 544-byte UB rows; cmp reads only its 320-byte payload.
        # Reuse launch-level hints for both paired and fallback row loads.
        kv_l2_policy = self._ori_l2_hint
        if const_expr(is_cmp):
            kv_l2_policy = self._cmp_l2_hint
        subtile_rows = _TILE_ROWS
        if const_expr(is_cmp):
            row_bytes = KV_ROW_BYTES_CMP
            page_stride = self._cmp_page_stride
        else:
            row_bytes = KV_ROW_BYTES_ORI
            page_stride = self._ori_page_stride
        n_rt = n_real
        # Bind the physical byte span, including the gaps between KV pages.
        storage_bytes = (kv_pool.shape[0] - 1) * page_stride + blk_size * row_bytes
        kv_bytes = from_tensor(kv_pool, layout=make_layout((storage_bytes,)))

        _mode = _addr_vec_mode()
        _consumer_vec = _mode != "off"
        tab = None
        if const_expr(_mode != "off"):
            # Read the already-published PA row table directly in scalar gather.
            tab = self._addr_tab
            col_base = self._avec_k1_tab if const_expr(is_cmp) else 0
        if const_expr(self.layout_kv != "PA_BBND"):
            dense_batch = 0 if batch is None else batch
            if const_expr(self.layout_kv == "TND"):
                cu_kv = self._cu_cmp_kv if const_expr(is_cmp) else self._cu_ori_kv
                batch_base = _i64(cu_kv, dense_batch) * row_bytes
            else:
                batch_base = dense_batch * page_stride
        # Every enqueued tile has n_real > 0. Write all subtiles, including
        # padding, so both AIVs publish one complete KV slot. QK masks padding;
        # PV uses zero probabilities there. Never publish an empty alias of KV.
        for chunk in range_constexpr(2):
            # Both AIVs publish each KV channel: 32 rows per AIV per half.
            # This keeps the retained QK/PV handoff counters consistent in the DSL.
            sub_b = half * 4 + self.subblock_idx * 2 + chunk
            sub_row = sub_b * subtile_rows
            buf = tile_slice(raw_half, (_TILE_ROWS, _COMBINE_DIM), (chunk, 0))
            bo_sp = 0 if batch is None else batch
            # Clamp padded rows to the final token in the valid prefix before addressing.
            last_off = n_rt - 1 - sub_row
            addresses = []
            pair_first = []
            pair_gap = []
            all_pairs_ok = dtypes.int32(1)
            if const_expr(_consumer_vec):
                # Address rows are padded to 16 columns. Reuse the last valid
                # chunk for the remaining masked chunks in a partial KV tile.
                index_base = (
                    col_base
                    + s2_start
                    + min(sub_row, ((n_rt - 1) // subtile_rows) * subtile_rows)
                )
                for pair in range_constexpr(subtile_rows // 2):
                    pair_row_offset = pair * 2
                    first_index_col = index_base + pair_row_offset
                    second_index_col = first_index_col + 1
                    first_row_index = _i64(tab, idx_row, first_index_col)
                    second_row_index = _i64(tab, idx_row, second_index_col)
                    if const_expr(_pa_fast_mode()):
                        # The bounded PA fast path publishes byte offsets once
                        # in VF, eliminating per-token scalar address multiplies.
                        addr0, addr1 = first_row_index, second_row_index
                    else:
                        addr0 = first_row_index * row_bytes
                        addr1 = second_row_index * row_bytes
                    first = min(addr0, addr1)
                    gap = max(addr0, addr1) - first
                    addresses.append(addr0)
                    addresses.append(addr1)
                    pair_first.append(first)
                    pair_gap.append(gap)
                    all_pairs_ok = (
                        (
                            all_pairs_ok
                            if gap - row_bytes <= _DMA_SOURCE_STRIDE_MAX_BYTES
                            else dtypes.int32(0)
                        )
                        if gap >= row_bytes
                        else dtypes.int32(0)
                    )
            else:
                for pair in range_constexpr(subtile_rows // 2):
                    pair_row_offset = pair * 2
                    pos0 = s2_start + sub_row + min(pair_row_offset, last_off)
                    pos1 = s2_start + sub_row + min(pair_row_offset + 1, last_off)
                    tok0 = _i64(idx_tab, idx_row, pos0)
                    tok1 = _i64(idx_tab, idx_row, pos1)
                    if const_expr(self.layout_kv == "PA_BBND"):
                        blk0, off0 = self.kvcache.pa_blk_off(
                            blk_table, tok0, bo_sp, blk_size
                        )
                        blk1, off1 = self.kvcache.pa_blk_off(
                            blk_table, tok1, bo_sp, blk_size
                        )
                        addr0 = blk0 * page_stride + off0 * row_bytes
                        addr1 = blk1 * page_stride + off1 * row_bytes
                    else:
                        addr0 = batch_base + tok0 * row_bytes
                        addr1 = batch_base + tok1 * row_bytes
                    first = min(addr0, addr1)
                    gap = max(addr0, addr1) - first
                    addresses.append(addr0)
                    addresses.append(addr1)
                    pair_first.append(first)
                    pair_gap.append(gap)
                    all_pairs_ok = (
                        (
                            all_pairs_ok
                            if gap - row_bytes <= _DMA_SOURCE_STRIDE_MAX_BYTES
                            else dtypes.int32(0)
                        )
                        if gap >= row_bytes
                        else dtypes.int32(0)
                    )
            # Deterministic level 3 requires sparse-index order, so disable pair sorting.
            all_pairs_ok = (
                dtypes.int32(0) if self._batch_consistency != 0 else all_pairs_ok
            )
            # Fill all 16 rows before publishing the slot. Reorder KV with inline scales;
            # QK/PV share that order. Duplicate tail addresses use per-row copies.
            if all_pairs_ok != 0:
                for pair in range_constexpr(subtile_rows // 2):
                    first = pair_first[pair]
                    second = first + pair_gap[pair]
                    dst = tile_slice(buf, (2, row_bytes), (pair, 0))
                    mem_copy(
                        dst,
                        (
                            kv_bytes[first : first + row_bytes,].view(1, row_bytes),
                            kv_bytes[second : second + row_bytes,].view(1, row_bytes),
                        ),
                        copy_options(l2_cache_ctl=kv_l2_policy),
                    )
            else:
                for pair_row_offset in range_constexpr(subtile_rows):
                    addr = addresses[pair_row_offset]
                    row = kv_bytes[addr : addr + row_bytes,].view(1, row_bytes)
                    mem_copy(
                        tile_slice(buf, (1, row_bytes), (pair_row_offset, 0)),
                        row,
                        copy_options(l2_cache_ctl=kv_l2_policy),
                    )

    @jit
    def antiquant_copy_batch(
        self, raw_half, half, kv_lower_rows_l1, kv_upper_rows_l1, is_cmp=False
    ):
        # 消费上一拍 gather 的两个 16-token 槽；VF 写 NZ17，再由 MTE3 发布到 L1。
        subtile_rows = _TILE_ROWS
        for chunk in range_constexpr(2):
            v0_out = self._v0_out.next_write()
            rslot_bytes = tile_slice(raw_half, (_TILE_ROWS, _COMBINE_DIM), (chunk, 0))

            if const_expr(is_cmp):
                self._antiquant_vf_fp4_g16(rslot_bytes, v0_out)
            else:
                self._antiquant_vf_fp8_g32(rslot_bytes, v0_out)
            # NZ17 faces become one regular strided UB-to-L1 copy.
            src = from_tensor(
                v0_out,
                layout=make_layout((D // 16, subtile_rows * 16), stride=(_NZ_CHUNK, 1)),
            )
            lt = self.subblock_idx * 2 + chunk
            if half == 0:
                l1_nd = from_tensor(
                    kv_lower_rows_l1,
                    layout=make_layout((D // 16, self.tile_n // 2 * 16)),
                    data_format="nd",
                )
                dst = tile_slice(l1_nd, (D // 16, subtile_rows * 16), (0, lt))
                mem_copy(dst, src)
            else:
                l1_nd = from_tensor(
                    kv_upper_rows_l1,
                    layout=make_layout((D // 16, self.tile_n // 2 * 16)),
                    data_format="nd",
                )
                dst = tile_slice(l1_nd, (D // 16, subtile_rows * 16), (0, lt))
                mem_copy(dst, src)

    @jit
    def _addr_vector_chunk(self, chunk, ktab, w, bs, inv_bs, row_bytes):
        lanes = min(_ADDR_HALF, ktab - chunk * _ADDR_HALF)
        mk, _ = rr.update_mask(lanes, elem_bits=32)
        lane = rr.varange(0, dtypes.int32)
        col = rr.vadds(lane, chunk * _ADDR_HALF, mask=mk)
        col = rr.vmins(col, w - dtypes.int64(1), mask=mk)
        idx = rr.vgather(self._ch_addr_sp, rr.vreinterpret(col, dtypes.uint32), mask=mk)
        if const_expr(isinstance(bs, int)):
            blk = rr.vshr(idx, bs.bit_length() - 1, mask=mk)
        else:
            # Drop four irrelevant offset bits before FP32 conversion. With
            # BT bounded by this UB arena and BS<=1024, idx>>4 is below 2**22,
            # hence exact even when the original logical index exceeds 2**24.
            if const_expr(_addr_vec_mode() == "arena"):
                idx_f = rr.vcast(rr.vshr(idx, 4, mask=mk), dtypes.float32, mask=mk)
            else:
                idx_f = rr.vcast(idx, dtypes.float32, mask=mk)
            blk = rr.vcast(
                rr.vmuls(idx_f, inv_bs, mask=mk), dtypes.int32, mask=mk, rounding="rd"
            )
        off = rr.vsub(idx, rr.vmuls(blk, bs, mask=mk), mask=mk)
        p = rr.vgather(self._ch_addr_bt, rr.vreinterpret(blk, dtypes.uint32), mask=mk)
        row = rr.vadd(rr.vmuls(p, bs, mask=mk), off, mask=mk)
        if const_expr(_pa_fast_mode()):
            row = rr.vmuls(row, row_bytes, mask=mk)
        rr.vstore(self._ch_addr_out, chunk * _ADDR_HALF, row, mk)

    @jit
    def _addr_precompute(self, m_count):
        """Precompute clamped physical row numbers for ori/cmp, then publish them within the block."""
        tab_cols = self._avec_k1_tab + (self._avec_k2_tab if self.has_cmp else 0)
        # Query rows are consecutive: locate the first batch once, then advance at cu_q boundaries.
        b_last = dtypes.int64(self._b_rt)
        batch0 = self._batch_of(self._m_start)
        bound0 = self._bound_after(batch0, b_last)
        for side in range_constexpr(2 if self.has_cmp else 1):
            if const_expr(side == 0):
                idx_tab, bt_gm = self._ori_idx, self._pa_bt
                bs, kwidth, col0 = self._avec_bs_ori, self._avec_k1, 0
                ktab = self._avec_k1_tab
                row_bytes = KV_ROW_BYTES_ORI

                def w_fn(mi):
                    return self._ori_valid_w(mi)
            else:
                idx_tab, bt_gm = self._cmp_idx, self._pa_bt_cmp
                bs, kwidth, col0 = (self._avec_bs_cmp, self._avec_k2, self._avec_k1_tab)
                ktab = self._avec_k2_tab
                row_bytes = KV_ROW_BYTES_CMP

                def w_fn(mi):
                    return self._cmp_topk_len(mi)

            btp = bt_gm.shape[1]
            if const_expr(_addr_vec_mode() == "arena"):
                # Disjoint runtime windows share an owner; sides reuse the arena.
                bt_capacity = _addr_tab_w(btp)
                self._ch_addr_bt = from_tensor(
                    self._addr_ub, layout=make_layout((1, bt_capacity))
                )
                self._ch_addr_sp = from_tensor(
                    self._addr_ub, layout=make_layout((1, ktab)), offset=bt_capacity * 4
                )
                self._ch_addr_out = from_tensor(
                    self._addr_ub,
                    layout=make_layout((1, ktab)),
                    offset=(bt_capacity + ktab) * 4,
                )
            numerator = 16.0 if _addr_vec_mode() == "arena" else 1.0
            if const_expr(isinstance(bs, int)):
                inv_bs = numerator / bs
            else:
                inv_bs = dtypes.float32(numerator) / dtypes.float32(bs)
            # Round up to cover every 16-column consumer subblock; mask the final vector chunk.
            chunks = (ktab + _ADDR_HALF - 1) // _ADDR_HALF
            per = m_count // 2
            tail = m_count % 2
            core = dtypes.int64(self.subblock_idx)
            # For an odd row count, give the extra ori row to AIV0 and the extra cmp row to AIV1.
            if const_expr(side == 0):
                start = per * core + min(core, tail)
                count = per + (dtypes.int64(1) if core < tail else dtypes.int64(0))
            else:
                start = per * core
                count = per + (dtypes.int64(0) if core < dtypes.int64(1) else tail)
            bt_staged = dtypes.int64(-1)
            # Initialize loop-carried batch state outside the loop; the inner while may execute zero times.
            batch = batch0
            bound = bound0
            for m_seq in range(start, start + count):
                m_idx = self._m_idx(m_seq)
                while m_idx >= bound:
                    batch = batch + 1
                    bound = self._bound_after(batch, b_last)
                w = w_fn(m_idx)
                active = w > dtypes.int64(0)
                if const_expr(self._seqused_q is not None):
                    valid_end = _i64(self._cu_q, 0, batch) + _i64(
                        self._seqused_q, batch
                    )
                    active = active and m_idx < valid_end
                if active:
                    _copy_addr_from_gm(
                        self._ch_addr_sp, idx_tab, m_idx * kwidth, kwidth
                    )
                    if batch != bt_staged:
                        _copy_addr_from_gm(self._ch_addr_bt, bt_gm, batch * btp, btp)
                        bt_staged = batch
                    with vf(mode="simd"):
                        for chunk in range(chunks):
                            self._addr_vector_chunk(
                                chunk, ktab, w, bs, inv_bs, row_bytes
                            )
                    dst = m_idx * tab_cols + col0
                    # Write the padded ktab columns, clamped by w-1. Multiples of 16 columns
                    # keep each UB-to-GM transfer aligned to 32 bytes without padding support.
                    _copy_addr_to_gm(self._addr_tab, self._ch_addr_out, dst, ktab)
        # Publish both AIVs' address writes through the AIC relay before either AIV reads the table.
        vec_sync_intra_arrive(PIPE.MTE3, self._addr_bar_sid)
        # Both AIV GM writes must be visible before either reads the shared table.
        # Local RingBuffer locks cannot replace this AIC-relayed publication.
        vec_sync_block_wait(PIPE.S, self._addr_bar_flag, mode=2)

    @jit
    def gather_quantized(self, m_seq: int, n, half, actual_n, ori_tiles):
        # 分支外只选一次 32-token 槽，两种格式共享同一生产游标。
        raw_half = self.ch_batch_bytes.next_write()
        m_idx = self._m_idx(m_seq)
        # Only scalar addressing needs a per-tile batch lookup.
        batch = None
        if const_expr(_addr_vec_mode() == "off"):
            batch = self._batch_of(m_idx)

        if const_expr(not self.has_cmp):
            self.gather_batch(
                raw_half,
                half,
                self._pa_phys,
                self._pa_bt,
                self._pa_bs_rt,
                n * self.tile_n,
                actual_n,
                idx_tab=self._ori_idx,
                idx_row=m_idx,
                batch=batch,
                is_cmp=False,
            )
        else:
            ori_bound = ori_tiles
            if n >= ori_bound:
                cmp_tile_index = n - ori_bound
                self.gather_batch(
                    raw_half,
                    half,
                    self._pa_phys_cmp,
                    self._pa_bt_cmp,
                    self._pa_cmp_bs_rt,
                    cmp_tile_index * self.tile_n,
                    actual_n,
                    idx_tab=self._cmp_idx,
                    idx_row=m_idx,
                    batch=batch,
                    is_cmp=True,
                )
            else:
                self.gather_batch(
                    raw_half,
                    half,
                    self._pa_phys,
                    self._pa_bt,
                    self._pa_bs_rt,
                    n * self.tile_n,
                    actual_n,
                    idx_tab=self._ori_idx,
                    idx_row=m_idx,
                    batch=batch,
                    is_cmp=False,
                )

    @jit
    def stage_antiquant(self, n, ori_tiles, half, kv_lower_rows_l1, kv_upper_rows_l1):
        # 格式来自延迟 task；分支外选槽，保持每半块一次消费。
        raw_half = self.ch_batch_bytes.next_read()
        if const_expr(self.has_cmp):
            if n >= ori_tiles:
                self.antiquant_copy_batch(
                    raw_half, half, kv_lower_rows_l1, kv_upper_rows_l1, True
                )
            else:
                self.antiquant_copy_batch(
                    raw_half, half, kv_lower_rows_l1, kv_upper_rows_l1, False
                )
        else:
            self.antiquant_copy_batch(
                raw_half, half, kv_lower_rows_l1, kv_upper_rows_l1, False
            )

    @jit
    def stage_softmax(
        self,
        tick: int,
        m_seq: int,
        n,
        is_first: int,
        job: int,
        is_last: int,
        score,
        p_l1_write,
        actual_n,
    ):
        if const_expr(self._sinks is None):
            if actual_n == self.tile_n:
                if is_first == 1:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        True,
                        True,
                        score,
                        self.p_out,
                        actual_n,
                        False,
                    )
                else:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        False,
                        True,
                        score,
                        self.p_out,
                        actual_n,
                        False,
                    )
            else:
                if is_first == 1:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        True,
                        False,
                        score,
                        self.p_out,
                        actual_n,
                        False,
                    )
                else:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        False,
                        False,
                        score,
                        self.p_out,
                        actual_n,
                        False,
                    )
        else:
            continuation = (self._s2_start > 0) if m_seq == 0 else False
            if actual_n == self.tile_n:
                if is_first == 1:
                    if continuation:
                        self.vector.softmax_rest_full(
                            job,
                            tick,
                            self._scale,
                            True,
                            True,
                            score,
                            self.p_out,
                            actual_n,
                            False,
                        )
                    else:
                        self.vector.softmax_rest_full(
                            job,
                            tick,
                            self._scale,
                            True,
                            True,
                            score,
                            self.p_out,
                            actual_n,
                            True,
                        )
                else:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        False,
                        True,
                        score,
                        self.p_out,
                        actual_n,
                        True,
                    )
            else:
                if is_first == 1:
                    if continuation:
                        self.vector.softmax_rest_full(
                            job,
                            tick,
                            self._scale,
                            True,
                            False,
                            score,
                            self.p_out,
                            actual_n,
                            False,
                        )
                    else:
                        self.vector.softmax_rest_full(
                            job,
                            tick,
                            self._scale,
                            True,
                            False,
                            score,
                            self.p_out,
                            actual_n,
                            True,
                        )
                else:
                    self.vector.softmax_rest_full(
                        job,
                        tick,
                        self._scale,
                        False,
                        False,
                        score,
                        self.p_out,
                        actual_n,
                        True,
                    )
        source = self.p_out.view(self.tile_n // 16, (self.tile_vec_m + 1) * 16)[
            None, 0 : self.tile_vec_m * 16
        ]
        target = tile_slice(
            from_tensor(
                p_l1_write,
                layout=make_layout((self.tile_n // 16, self.tile_cube_m * 16)),
                data_format="nd",
            ),
            (self.tile_n // 16, self.tile_vec_m * 16),
            (0, self.subblock_idx),
        )
        mem_copy(target, source)
        if is_last == 1:
            self.vector.finalize_softmax_state(job, self._scale)

    def next_kv(self, write):
        return (
            (
                self.matmul.kv_channels[0].next_write(),
                self.matmul.kv_channels[1].next_write(),
            )
            if write
            else (
                self.matmul.kv_channels[0].next_read(),
                self.matmul.kv_channels[1].next_read(),
            )
        )

    def next_result(self, write):
        lo = (
            self.result_channels[0].next_write()
            if write
            else self.result_channels[0].next_read()
        )
        hi = (
            self.result_channels[1].next_write()
            if write
            else self.result_channels[1].next_read()
        )
        return (
            tile_slice(lo, (self.tile_vec_m, 256), (0, 0)),
            tile_slice(lo, (self.tile_vec_m, 256), (1, 0)),
            tile_slice(hi, (self.tile_vec_m, 256), (0, 0)),
            tile_slice(hi, (self.tile_vec_m, 256), (1, 0)),
        )

    @jit
    def matmul_qk(self, job, kv_lower_rows_l1, kv_upper_rows_l1):
        self.matmul.bmm1_fanout_q(job, kv_lower_rows_l1, kv_upper_rows_l1)
        self.matmul.store_s(self.qk_ub.next_write())

    @jit
    def cube_pv(self, kv_lower_rows_l1, kv_upper_rows_l1, m_seq: int, n, is_first: int):
        self.matmul.compute_pv_fanout(
            self.p_l1.next_read(),
            self.next_result(True),
            kv_lower_rows_l1,
            kv_upper_rows_l1,
            is_first,
        )

    @jit
    def query_tile_range(self, m_seq, row_count, s2_end):
        tiles = self._n_end_rt(self._m_idx(m_seq))
        begin = self._s2_start if m_seq == 0 else dtypes.int64(0)
        end = (s2_end if s2_end > 0 else tiles) if m_seq == row_count - 1 else tiles
        return begin, end

    @jit
    def next_query(self, m_seq, row_count, s2_end):
        stop = row_count
        while m_seq < stop:
            begin, end = self.query_tile_range(m_seq, row_count, s2_end)
            stop = m_seq if end > begin else stop
            m_seq = m_seq if end > begin else m_seq + 1
        return m_seq

    def num_heads(self):
        # Preserve the established shape-read sites for sink specializations.
        return self._sinks.shape[1] if self._sinks is not None else self._num_heads

    def query_view(self, m_seq):
        if const_expr(self.full_heads):
            return self._q.view(
                self._q.shape[0] // self.tile_cube_m, self.tile_cube_m, self.tile_d
            )[self._m_idx(m_seq), None, None]
        return self._q.view(self._ori_idx.shape[0], self.num_heads(), self.tile_d)[
            self._m_idx(m_seq), None, None
        ]

    def output_view(self, m_idx):
        if const_expr(self.full_heads):
            return tile_slice(self._out, (self.tile_cube_m, self.tile_d), (m_idx, 0))
        return self._out.view(self._ori_idx.shape[0], self.num_heads(), self.tile_d)[
            m_idx, None, None
        ].view(self.num_heads(), self.tile_d)

    @jit
    def pipeline_cube(self, row_count, s2_end):
        m_seq = self.next_query(dtypes.int64(0), row_count, s2_end)
        job = dtypes.int64(0)
        if m_seq < row_count:
            begin, end = self.query_tile_range(m_seq, row_count, s2_end)
            self.matmul.load_q_wide(self.query_view(m_seq), job, False)
            self.matmul.load_q_wide(self.query_view(m_seq), job, True)
            kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(False)
            self.matmul_qk(job, kv_lower_rows_l1, kv_upper_rows_l1)
            active_rows = dtypes.int64(0)
            for row in range(row_count):
                row_begin, row_end = self.query_tile_range(row, row_count, s2_end)
                active_rows += 1 if row_end > row_begin else 0
            for _ in range(active_rows):
                next_m = self.next_query(m_seq + 1, row_count, s2_end)
                if next_m < row_count:
                    self.matmul.load_q_wide(self.query_view(next_m), job + 1, False)
                for n in range(begin, end - 1):
                    previous_kv_lower_rows_l1, previous_kv_upper_rows_l1 = (
                        kv_lower_rows_l1,
                        kv_upper_rows_l1,
                    )
                    kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(False)
                    self.matmul_qk(job, kv_lower_rows_l1, kv_upper_rows_l1)
                    self.cube_pv(
                        previous_kv_lower_rows_l1,
                        previous_kv_upper_rows_l1,
                        m_seq,
                        n,
                        1 if n == begin else 0,
                    )
                previous_kv_lower_rows_l1, previous_kv_upper_rows_l1 = (
                    kv_lower_rows_l1,
                    kv_upper_rows_l1,
                )
                if next_m < row_count:
                    self.matmul.load_q_wide(self.query_view(next_m), job + 1, True)
                    kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(False)
                    self.matmul_qk(job + 1, kv_lower_rows_l1, kv_upper_rows_l1)
                self.cube_pv(
                    previous_kv_lower_rows_l1,
                    previous_kv_upper_rows_l1,
                    m_seq,
                    end - 1,
                    1 if end - begin == 1 else 0,
                )
                m_seq = next_m
                job = job + 1
                if m_seq < row_count:
                    begin, end = self.query_tile_range(m_seq, row_count, s2_end)

    @jit
    def run_vector_pipeline_step(
        self, delay, tick, draining, kv_lower_rows_l1, kv_upper_rows_l1
    ):
        # gather(t)、反量化(t-1)、softmax(t-3)、update(t-4) 分别保留独立生命周期。
        score = self.qk_ub.next_read()
        p_l1_write = self.p_l1.next_write()
        results = self.next_result(False)
        gather = delay.tap(0)
        antiquant = delay.tap(1)
        softmax = delay.tap(3)
        update = delay.tap(4)
        if not draining:
            self.gather_quantized(gather.m, gather.n, 0, gather.valid, gather.ori_tiles)
        if tick >= 1 and antiquant.n >= 0:
            self.stage_antiquant(
                antiquant.n, antiquant.ori_tiles, 0, kv_lower_rows_l1, kv_upper_rows_l1
            )
        if tick >= 3 and softmax.n >= 0:
            self.stage_softmax(
                softmax.t,
                softmax.m,
                softmax.n,
                softmax.first,
                softmax.sum_slot,
                softmax.last,
                score,
                p_l1_write,
                softmax.valid,
            )
        if not draining:
            self.gather_quantized(gather.m, gather.n, 1, gather.valid, gather.ori_tiles)
        if tick >= 1 and antiquant.n >= 0:
            self.stage_antiquant(
                antiquant.n, antiquant.ori_tiles, 1, kv_lower_rows_l1, kv_upper_rows_l1
            )
            kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(True)
        if tick >= 4 and update.n >= 0:
            self.vector.bind_result(results)
            if update.first == 1:
                for half in range_constexpr(2):
                    with vf(mode="simd"):
                        mask, _ = rr.update_mask(1, elem_bits=32)
                        value = rr.vload(results[half * 2], 0)
                        rr.vstore(results[half * 2], 0, value, mask)
            for half in range_constexpr(2):
                self.stage_update(
                    update.t, update.m, update.last, update.first, update.sum_slot, half
                )
        delay.advance()
        return kv_lower_rows_l1, kv_upper_rows_l1

    @jit
    def pipeline_vector(self, row_count, s2_end):
        delay = SlotMajorDelayLineGroup()
        tick = dtypes.int64(0)
        job = dtypes.int64(0)
        kv_lower_rows_l1, kv_upper_rows_l1 = self.next_kv(True)
        for m_seq in range(row_count):
            begin, end = self.query_tile_range(m_seq, row_count, s2_end)
            m_idx = self._m_idx(m_seq)
            ori_length = self._ori_valid_w(m_idx)
            ori_tiles = (ori_length + self.tile_n - 1) // self.tile_n
            cmp_length = dtypes.int64(0)
            if const_expr(self.has_cmp):
                cmp_length = self._cmp_topk_len(m_idx)
            for n in range(begin, end):
                valid = min(self.tile_n, ori_length - n * self.tile_n)
                if const_expr(self.has_cmp):
                    if n >= ori_tiles:
                        valid = min(
                            self.tile_n, cmp_length - (n - ori_tiles) * self.tile_n
                        )
                delay.push(
                    m=m_seq,
                    n=n,
                    t=tick,
                    last=1 if n == end - 1 else 0,
                    first=1 if n == begin else 0,
                    sum_slot=job % 2,
                    valid=valid,
                    ori_tiles=ori_tiles,
                )
                kv_lower_rows_l1, kv_upper_rows_l1 = self.run_vector_pipeline_step(
                    delay, tick, False, kv_lower_rows_l1, kv_upper_rows_l1
                )
                tick = tick + 1
            job = job + (1 if end > begin else 0)
        for _ in range(4):
            delay.push(
                m=0, n=-1, t=tick, last=0, first=0, sum_slot=0, valid=0, ori_tiles=0
            )
            kv_lower_rows_l1, kv_upper_rows_l1 = self.run_vector_pipeline_step(
                delay, tick, True, kv_lower_rows_l1, kv_upper_rows_l1
            )
            tick = tick + 1

    @jit
    def stage_update(
        self, tick: int, m_seq: int, is_last: int, is_first: int, sum_slot: int, half
    ):
        # At a part boundary, normalize O and either stage the split part or write final output.
        # Empty rows bypass the pipeline and are finalized after draining.
        exp_idx = tick % 2
        sum_idx = sum_slot

        m_idx = self._m_idx(m_seq)
        o_row_tile = m_idx
        lse_tile_base = (
            (o_row_tile * (self.tile_cube_m // self.tile_vec_m) + self.subblock_idx)
            if const_expr(self._lse is not None)
            else None
        )
        if const_expr(self._lse is not None and not self.full_heads):
            lse_tile_base = m_idx * self.num_heads() + self.subblock_idx * (
                (self.num_heads() + 1) // 2
            )
        # LSE coordinates count 32-element tiles: flat offset m_idx*64 + aiv_id*32.
        if is_last == 1:
            split_count = (
                dtypes.int64(1) if m_seq == self._first_split_row else dtypes.int64(0)
            )
            split_count += (
                dtypes.int64(1) if m_seq == self._last_split_row else dtypes.int64(0)
            )
            if split_count > 0:
                if is_first == 1:
                    self.vector.update_o(half, True, True, exp_idx, sum_idx, False)
                else:
                    self.vector.update_o(half, False, True, exp_idx, sum_idx, False)
                if const_expr(half == 1):
                    # A leading continuation uses slot 0. A trailing split uses slot 1 only
                    # when this core also starts with a continuation.
                    part_idx = (
                        dtypes.int64(0)
                        if m_seq == 0
                        else (
                            dtypes.int64(1) if self._s2_start > 0 else dtypes.int64(0)
                        )
                    )
                    slot = self._first_fd + part_idx
                    fd_o, fd_ms = self._fd_workspace_views()
                    self.vector.stage_partial(fd_o, fd_ms, slot, sum_idx)

            else:
                if is_first == 1:
                    self.vector.update_o(half, True, True, exp_idx, sum_idx, True)
                else:
                    self.vector.update_o(half, False, True, exp_idx, sum_idx, True)
                o_tile_gm = self.output_view(o_row_tile)
                self.vector.finalize_o(
                    o_tile_gm, half, sum_idx, lse_gm=self._lse, lse_base=lse_tile_base
                )
        else:
            if is_first == 1:
                pass
            else:
                self.vector.update_o(half, False, False, exp_idx, sum_idx, False)

    @jit
    def _fd_workspace_views(self):
        # FD views are needed only by split-row staging and final reduction.
        blocks = self._ws.shape[0] // WS_BYTES_PER_CORE
        slots = blocks * FD_SLOTS_PER_CORE
        fd_o0 = dtypes.int64(0)
        ws_f32 = self._ws.view(dtypes.float32)
        fd_o = ws_f32[fd_o0 : fd_o0 + slots * FD_O_ELEMS,].view(slots * N1, D)
        fd_ms0 = fd_o0 + slots * FD_O_ELEMS
        fd_ms = ws_f32[fd_ms0 : fd_ms0 + slots * FD_MS_ELEMS,].view(slots * FD_MS_ELEMS)
        return fd_o, fd_ms


def _batch_consistency_enabled():
    """Enable batch consistency at torch_npu deterministic level 3.

    This selects per-row sparse KV copies instead of paired-row DMA.
    """
    try:
        import torch_npu

        return int(torch_npu.npu._get_deterministic_level()) == 3
    except (AttributeError, ImportError):
        return False


def _validate_inputs(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    kv_batch_strides=None,
):
    """Validate tensor metadata only; never read tensor values on the host.

    Callers supply valid prefix sums, sequence lengths and active indices.
    cu_seqlens_q contains physical TND spans. seqused_q may shorten them;
    TND padding produces zero O/LSE. A valid empty sparse row retains its
    sink/-inf LSE. BSND keeps its existing empty-row padding rule.
    """
    if ori_kv is None:
        raise ValueError("Invalid input: ori_kv is required")
    if layout_q not in ("TND", "BSND") or layout_kv not in ("PA_BBND", "TND", "BSND"):
        raise ValueError(
            "Invalid input: layout_q must be TND or BSND; "
            "layout_kv must be PA_BBND, TND or BSND"
        )
    if int(quant_mode) != 4:
        raise ValueError(
            "Invalid input: mixed_quant_sparse_flash_mla supports only quant_mode=4"
        )

    if not isinstance(q, torch.Tensor):
        raise TypeError("Invalid input: q must be a torch.Tensor")
    q_rank = _TENSOR_RANK_3D if layout_q == "TND" else _TENSOR_RANK_4D
    if q.dtype != torch.bfloat16 or q.dim() != q_rank or int(q.shape[-1]) != D:
        raise ValueError(
            f"Invalid input: q must be a BF16 {layout_q} tensor "
            "with trailing shape [N1, 512], N1 in 1..64"
        )
    query_shape = tuple(q.shape[:-2])
    n1 = int(q.shape[-2])
    if not 1 <= n1 <= N1:
        raise ValueError(f"N1 must be in 1..{N1}, got {n1}")
    if any(size <= 0 for size in query_shape):
        raise ValueError("Invalid input: q must contain at least one token")

    if layout_q == "TND":
        _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q")
        b = cu_seqlens_q.numel() - 1
    else:
        b = q.shape[0]
        if cu_seqlens_q is not None:
            _validate_cu_seqlens(cu_seqlens_q, name="cu_seqlens_q", batch=b)

    if seqused_q is not None:
        if not isinstance(seqused_q, torch.Tensor):
            raise TypeError(
                "Invalid input: seqused_q must be a torch.Tensor when provided"
            )
        if seqused_q.dtype != torch.int32 or seqused_q.dim() != _TENSOR_RANK_1D:
            raise ValueError(
                "Invalid input: seqused_q must be a 1-D int32 tensor of shape (B,)"
            )
        if seqused_q.numel() != b:
            raise ValueError(
                "Invalid input: seqused_q must have one length per query batch"
            )

    for name, seqused in (
        ("seqused_ori_kv", seqused_ori_kv),
        ("seqused_cmp_kv", seqused_cmp_kv),
    ):
        if seqused is None:
            continue
        if not isinstance(seqused, torch.Tensor):
            raise TypeError(
                f"Invalid input: {name} must be a torch.Tensor when provided"
            )
        if seqused.dtype != torch.int32 or seqused.dim() != _TENSOR_RANK_1D:
            raise ValueError(
                f"Invalid input: {name} must be a 1-D int32 tensor of shape (B,)"
            )
        if int(seqused.numel()) != b:
            raise ValueError(
                f"Invalid input: {name} must have one entry per batch element (B={b})"
            )

    has_cmp = (
        cmp_kv is not None
        or cmp_sparse_indices is not None
        or cmp_block_table is not None
    )
    if has_cmp:
        if cmp_kv is None or cmp_sparse_indices is None:
            raise ValueError(
                "ORI_CMP_SPARSE requires cmp_kv and cmp_sparse_indices together"
            )
    elif (
        cmp_topk_length is not None
        or cu_seqlens_cmp_kv is not None
        or seqused_cmp_kv is not None
    ):
        raise ValueError("Invalid input: cmp inputs are not valid for ORI_SPARSE")

    ori_batch_stride = cmp_batch_stride = None
    if kv_batch_strides is not None:
        if (
            layout_kv != "TND"
            or not isinstance(kv_batch_strides, (tuple, list))
            or len(kv_batch_strides) != 2
        ):
            raise ValueError(
                "Invalid input: kv_batch_strides requires TND KV and (ori_tokens, cmp_tokens)"
            )
        ori_batch_stride, cmp_batch_stride = kv_batch_strides
        required = (
            (ori_batch_stride, cmp_batch_stride) if has_cmp else (ori_batch_stride,)
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in required
        ):
            raise ValueError(
                "Invalid input: kv_batch_strides entries must be positive Python integers"
            )
        if not has_cmp and cmp_batch_stride is not None:
            raise ValueError(
                "Invalid input: absent CMP requires kv_batch_strides=(ori_tokens, None)"
            )

    _validate_kv_side(
        ori_kv,
        ori_block_table,
        cu_seqlens_ori_kv,
        name="ori_kv",
        row_bytes=KV_ROW_BYTES_ORI,
        batch=b,
        layout=layout_kv,
        batch_stride=ori_batch_stride,
    )
    _validate_indices(
        ori_sparse_indices, name="ori_sparse_indices", query_shape=query_shape
    )
    _validate_lengths(ori_topk_length, name="ori_topk_length", query_shape=query_shape)

    if has_cmp:
        _validate_kv_side(
            cmp_kv,
            cmp_block_table,
            cu_seqlens_cmp_kv,
            name="cmp_kv",
            row_bytes=KV_ROW_BYTES_CMP,
            batch=b,
            layout=layout_kv,
            batch_stride=cmp_batch_stride,
        )
        _validate_indices(
            cmp_sparse_indices, name="cmp_sparse_indices", query_shape=query_shape
        )
        _validate_lengths(
            cmp_topk_length, name="cmp_topk_length", query_shape=query_shape
        )

    if sinks is not None and (
        not isinstance(sinks, torch.Tensor)
        or sinks.dtype != torch.float32
        or tuple(sinks.shape) != (n1,)
    ):
        raise ValueError(
            "Invalid input: sinks must be an FP32 tensor with shape [N1], shared by all batches"
        )

    if metadata is None:
        raise ValueError(
            "Invalid input: metadata is required and is never synthesized: generate it with "
            "mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, "
            "num_heads_q=q.shape[-2], num_heads_kv=1, head_dim=512, quant_mode=1, has_cmp_kv=...) and pass "
            "the result in (two-stage call)"
        )
    if not isinstance(metadata, torch.Tensor):
        raise TypeError("Invalid input: metadata must be a torch.Tensor")
    if (
        metadata.dtype != torch.int32
        or metadata.dim() != _TENSOR_RANK_1D
        or metadata.numel() != MQSMLA_METADATA_TOTAL_SIZE
    ):
        raise ValueError(
            f"Invalid input: metadata must be a 1-D int32 tensor with exactly "
            f"{MQSMLA_METADATA_TOTAL_SIZE} elements (fixed shape)"
        )

    for name, tensor in (
        ("q", q),
        ("ori_kv", ori_kv),
        ("cmp_kv", cmp_kv),
        ("ori_sparse_indices", ori_sparse_indices),
        ("cmp_sparse_indices", cmp_sparse_indices),
        ("ori_block_table", ori_block_table),
        ("cmp_block_table", cmp_block_table),
        ("ori_topk_length", ori_topk_length),
        ("cmp_topk_length", cmp_topk_length),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_ori_kv", cu_seqlens_ori_kv),
        ("cu_seqlens_cmp_kv", cu_seqlens_cmp_kv),
        ("seqused_q", seqused_q),
        ("seqused_ori_kv", seqused_ori_kv),
        ("seqused_cmp_kv", seqused_cmp_kv),
        ("sinks", sinks),
        ("metadata", metadata),
    ):
        if tensor is None:
            continue
        if tensor.device != q.device:
            raise ValueError(f"Invalid input: {name} must be on the same device as q")
        if not (layout_kv == "PA_BBND" and name in ("ori_kv", "cmp_kv")):
            if not tensor.is_contiguous():
                raise ValueError(f"Invalid input: {name} must be contiguous")


def _validate_cu_seqlens(cu, *, name, batch=None):
    if (
        not isinstance(cu, torch.Tensor)
        or cu.dtype != torch.int32
        or cu.dim() != 1
        or cu.numel() < 2
        or (batch is not None and cu.numel() != batch + 1)
    ):
        raise ValueError(
            f"Invalid input: {name} is required for TND and must be int32 [B+1]"
        )


def _validate_kv_side(
    kv, block_table, cu, *, name, row_bytes, batch, layout, batch_stride=None
):
    if layout == "PA_BBND":
        _validate_pa_side(kv, block_table, name=name, row_bytes=row_bytes)
        if block_table.shape[0] != batch:
            raise ValueError(f"Invalid input: {name}_block_table must have B rows")
    else:
        rank = 3 if layout == "TND" else 4
        if (
            not isinstance(kv, torch.Tensor)
            or kv.dtype != torch.uint8
            or kv.dim() != rank
            or tuple(kv.shape[-2:]) != (N2, row_bytes)
            or any(size <= 0 for size in kv.shape)
        ):
            raise ValueError(
                f"Invalid input: {name} must be a non-empty uint8 {layout} "
                f"tensor with trailing shape [1, {row_bytes}]"
            )
        if layout == "BSND" and kv.shape[0] != batch:
            raise ValueError(f"Invalid input: {name} must have B batches")
        if block_table is not None:
            raise ValueError(
                f"Invalid input: {name}_block_table is only supported for PA_BBND"
            )
    if layout == "TND":
        if batch_stride is not None:
            if kv.shape[0] != batch * batch_stride:
                raise ValueError(
                    f"Invalid input: {name} token count must equal B * declared batch stride"
                )
            if cu is not None:
                _validate_cu_seqlens(cu, name=f"cu_seqlens_{name}", batch=batch)
        else:
            _validate_cu_seqlens(cu, name=f"cu_seqlens_{name}", batch=batch)
    elif cu is not None:
        raise ValueError(
            f"Invalid input: cu_seqlens_{name} is only supported for TND KV"
        )


def _validate_pa_side(kv, block_table, *, name, row_bytes):
    if not isinstance(kv, torch.Tensor) or not isinstance(block_table, torch.Tensor):
        raise TypeError(
            f"Invalid input: {name} and {name}_block_table must be torch.Tensor"
        )
    if (
        kv.dtype != torch.uint8
        or kv.dim() != _TENSOR_RANK_4D
        or int(kv.shape[2]) != N2
        or int(kv.shape[-1]) != row_bytes
    ):
        side = "FP8 (ori)" if name != "cmp_kv" else "FP4 (cmp)"
        raise ValueError(
            f"Invalid input: {name} must be a uint8 [blocknum, blocksize, 1, {row_bytes}] "
            f"tensor ({side} byte view); got shape {tuple(kv.shape)}, "
            f"dtype {kv.dtype}"
        )
    if int(kv.shape[0]) <= 0 or int(kv.shape[1]) <= 0:
        raise ValueError(f"Invalid input: {name} must be non-empty")
    if (
        block_table.dtype != torch.int32
        or block_table.dim() != _TENSOR_RANK_2D
        or int(block_table.shape[0]) <= 0
        or int(block_table.shape[1]) <= 0
    ):
        raise ValueError(
            f"Invalid input: {name}_block_table must be a non-empty int32 "
            f"[batch, num_blocks] tensor"
        )
    bs = int(kv.shape[1])
    strides = tuple(kv.stride())
    page_stride = int(
        kv.stride(0)
    )  # The uint8 page stride is measured in bytes, independently of page length.
    if strides[1:] != (N2 * row_bytes, row_bytes, 1):
        raise ValueError(
            f"Invalid input: {name} only supports axis-0 non-contiguity; "
            f"page dimensions must be contiguous, got strides {strides}"
        )
    if page_stride < bs * N2 * row_bytes:
        raise ValueError(
            f"Invalid input: {name} page stride must be at least {bs * N2 * row_bytes} "
            "bytes (positive, non-overlapping pages)"
        )
    if bs > _MAX_PA_BLOCK_SIZE:
        raise ValueError("PA block size must be in 1..1024")


def _validate_indices(idx, *, name, query_shape):
    if not isinstance(idx, torch.Tensor) or idx.dtype != torch.int32:
        raise ValueError(f"Invalid input: {name} must be an int32 torch.Tensor")
    if idx.dim() != len(query_shape) + 2 or tuple(idx.shape[:-1]) != (*query_shape, N2):
        raise ValueError(
            f"Invalid input: {name} must be int32 with shape {(*query_shape, N2, 'K')}"
        )
    k = int(idx.shape[-1])
    if k <= 0:
        raise ValueError(f"Invalid input: {name} must have at least one index column")


def _validate_lengths(x, *, name, query_shape):
    if x is None:
        return
    if not isinstance(x, torch.Tensor) or x.dtype != torch.int32:
        raise ValueError(f"Invalid input: {name} must be an int32 torch.Tensor")
    if tuple(x.shape) != (*query_shape, N2):
        raise ValueError(
            f"Invalid input: {name} must be int32 with shape {(*query_shape, N2)}"
        )


def construct_workspace(metadata, addr_bytes=0):
    # Query the same device core count as metadata; workspace capacity determines the launch grid.
    blocks = _get_cube_core_num(metadata.device)
    if not 0 < blocks <= AIC_CORE_MAX_NUM:
        raise ValueError(
            f"Invalid input: core count must be in 1..{AIC_CORE_MAX_NUM}, got {blocks}"
        )
    ws_bytes = blocks * WS_BYTES_PER_CORE
    # Allocation only: avoid a zeros device op during ACLGraph capture.
    # FD slots are fully written by their staging core before publication.
    # Append the prepass row-index table to the same device allocation.
    # The kernel writes and reads the table without host staging or H2D copies.
    return torch.empty(ws_bytes + addr_bytes, dtype=torch.uint8, device=metadata.device)


@host
def _run_mqsmla(
    q_gm: Tensor,
    ori_kv_gm: Tensor,
    ori_idx_gm: Tensor,
    ori_bt_gm: Tensor,
    ori_len_gm: Tensor,
    out_gm: Tensor,
    sinks_gm: Tensor,
    cu_seqlens_q: Tensor,
    metadata_gm: Tensor,
    workspace_gm: Tensor,
    cmp_kv_gm: Tensor = None,
    cmp_idx_gm: Tensor = None,
    cmp_bt_gm: Tensor = None,
    cmp_len_gm: Tensor = None,
    lse_gm: Tensor = None,
    softmax_scale: dtypes.float32 = 1.0,
    batch_consistency: dtypes.int32 = 0,
    addr_ws_gm: Tensor = None,
    cu_ori_kv: Tensor = None,
    cu_cmp_kv: Tensor = None,
    seqused_q: Tensor = None,
    layout_kv: Constexpr[str] = "PA_BBND",
    head_bucket: Constexpr[int] = 64,
):
    """Adapt dynamic tensor views and launch the device kernel from a JIT entry."""
    # Adapt DSL views during tracing without torch data conversions.
    # Flatten Q/O to [T1*N1,512], indices to [T1,K], lengths to [T1],
    # cu_q to [1,B+1], sinks to [1,N1], and LSE to [T1*N1].
    # Keep KV byte-addressed so physical page padding cannot become logical tokens.
    m_rows = q_gm.shape[0] * q_gm.shape[1]
    q_flat = q_gm
    if const_expr(sinks_gm is not None):
        q_flat = q_gm.view(m_rows, D)
    out_flat = out_gm.view(m_rows, D)
    ori_pool = ori_kv_gm
    ori_idx_flat = ori_idx_gm.view(
        ori_idx_gm.shape[0] * ori_idx_gm.shape[1], ori_idx_gm.shape[2]
    )
    ori_len_flat = None
    if const_expr(ori_len_gm is not None):
        ori_len_flat = ori_len_gm.view(ori_len_gm.shape[0] * ori_len_gm.shape[1])
    cu_q = cu_seqlens_q.view(1, cu_seqlens_q.shape[0])
    sinks_k = None
    if const_expr(sinks_gm is not None):
        sinks_k = sinks_gm.view(1, q_gm.shape[1])
    lse_flat = None
    if const_expr(lse_gm is not None):
        lse_flat = lse_gm.view(lse_gm.shape[1] * lse_gm.shape[2])
    cmp_pool = None
    cmp_idx_flat = None
    cmp_len_flat = None
    if const_expr(cmp_kv_gm is not None):
        cmp_pool = cmp_kv_gm
        cmp_idx_flat = cmp_idx_gm.view(
            cmp_idx_gm.shape[0] * cmp_idx_gm.shape[1], cmp_idx_gm.shape[2]
        )
        if const_expr(cmp_len_gm is not None):
            cmp_len_flat = cmp_len_gm.view(cmp_len_gm.shape[0] * cmp_len_gm.shape[1])
    tile_m = 32 if head_bucket == 32 else 64
    op = MqsmlaKernel(
        tile_cube_m=tile_m,
        tile_vec_m=tile_m // 2,
        tile_n=TILE_N,
        has_cmp=(cmp_kv_gm is not None),
        layout_kv=layout_kv,
        full_heads=(head_bucket == 64),
    )
    # Each core has two FD staging slots; workspace capacity determines the dynamic launch grid.
    blocks = workspace_gm.shape[0] // WS_BYTES_PER_CORE
    op[blocks](
        out_flat,
        q_flat,
        sinks_k,
        ori_pool,
        ori_bt_gm,
        workspace_gm,
        cmp_pool,
        cmp_bt_gm,
        cmp_idx_flat,
        ori_idx_flat,
        cu_q,
        lse_flat,
        ori_len_flat,
        cmp_len_flat,
        metadata_gm,
        softmax_scale,
        batch_consistency,
        addr_ws_gm,
        cu_ori_kv,
        cu_cmp_kv,
        seqused_q,
    )


def _select_addr_mode(ori_kv, ori_bt, ori_idx, cmp_kv, cmp_bt, cmp_idx):
    """Select staging and address units; K/page widths remain dynamic in AOT."""
    if not _addr_vec_eligible(ori_kv, ori_bt, ori_idx, cmp_kv, cmp_bt, cmp_idx):
        return "off"
    compact = (
        ori_bt.shape[1] <= _ADDR_COMPACT_BT
        and ori_idx.shape[2] <= _ADDR_COMPACT_K
        and (
            cmp_kv is None
            or (
                cmp_bt.shape[1] <= _ADDR_COMPACT_BT
                and cmp_idx.shape[2] <= _ADDR_COMPACT_K
            )
        )
    )
    return "pre" if compact else "arena"


def _get_compiled_kernel(
    ori_kv,
    ori_sparse_indices,
    ori_block_table,
    ori_topk_length,
    cmp_kv=None,
    cmp_sparse_indices=None,
    cmp_block_table=None,
    cmp_topk_length=None,
    lse=None,
    layout_kv="PA_BBND",
    num_heads=64,
    has_sinks=True,
    tnd_padding=False,
):
    """Cache dynamic executables by optional inputs and address representation."""
    addr_mode = _select_addr_mode(
        ori_kv,
        ori_block_table,
        ori_sparse_indices,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
    )
    return _get_compiled_variant(
        cmp_kv is not None,
        ori_topk_length is not None,
        cmp_topk_length is not None,
        lse is not None,
        addr_mode,
        layout_kv,
        _pa_fast_eligible(
            ori_kv,
            ori_block_table,
            ori_sparse_indices,
            cmp_kv,
            cmp_block_table,
            cmp_sparse_indices,
            layout_kv,
        ),
        _head_bucket(num_heads),
        has_sinks,
        tnd_padding,
    )


def _head_bucket(num_heads):
    # Full N1=64 retains its original static executable. Two dynamic variants
    # cover all smaller counts, without compiling separately for every N1.
    return 64 if num_heads == 64 else (32 if num_heads <= 32 else 63)


@aot.compile_cache
def _get_compiled_variant(
    has_cmp,
    has_ori_length,
    has_cmp_length,
    has_lse,
    addr_mode,
    layout_kv="PA_BBND",
    pa_fast=False,
    head_bucket=64,
    has_sinks=True,
    tnd_padding=False,
):
    """Share compilation, caching and address-mode lifetime with AOT export."""
    if addr_mode not in ("pre", "off", "arena"):
        raise ValueError(f"Unsupported address mode: {addr_mode}")
    if pa_fast and layout_kv != "PA_BBND":
        raise ValueError("PA specialization cannot be used for a dense KV layout")
    if head_bucket not in (32, 63, 64):
        raise ValueError(f"Unsupported head bucket: {head_bucket}")
    key = (
        has_cmp,
        has_ori_length,
        has_cmp_length,
        has_lse,
        addr_mode,
        layout_kv,
        pa_fast,
        head_bucket,
        has_sinks,
        tnd_padding,
    )
    with _COMPILE_LOCK:
        global _ADDR_VEC_OVERRIDE, _PA_FAST_OVERRIDE
        previous_mode, previous_fast = _ADDR_VEC_OVERRIDE, _PA_FAST_OVERRIDE
        _ADDR_VEC_OVERRIDE, _PA_FAST_OVERRIDE = addr_mode, pa_fast
        try:
            return _compile_mqsmla(*key)
        finally:
            _ADDR_VEC_OVERRIDE, _PA_FAST_OVERRIDE = previous_mode, previous_fast


def _compile_mqsmla(
    has_cmp,
    has_ori_length,
    has_cmp_length,
    has_lse,
    addr_mode,
    layout_kv="PA_BBND",
    pa_fast=False,
    head_bucket=64,
    has_sinks=True,
    tnd_padding=False,
):
    """Declare the dynamic signature; the caller holds the compilation lock."""
    rows = Dim("T1", min=1)
    heads = (
        N1
        if head_bucket == 64
        else Dim("N1", min=1 if head_bucket == 32 else 33, max=head_bucket)
    )
    batch = Dim("B", min=1)
    # TND KV has no tensor carrying B directly; bind its prefix-sum length
    # instead of leaving B present only through the derived expression B+1.
    query_boundaries = (
        Dim("B1", min=2) if layout_kv == "TND" and not tnd_padding else batch + 1
    )
    if layout_kv == "TND":
        ori_block_size = cmp_block_size = 1
    elif layout_kv == "PA_BBND":
        ori_block_size = (
            128 if pa_fast else Dim("ORI_BS", min=1, max=_MAX_PA_BLOCK_SIZE)
        )
        cmp_block_size = (
            128 if pa_fast else Dim("CMP_BS", min=1, max=_MAX_PA_BLOCK_SIZE)
        )
    else:
        ori_block_size = Dim("ORI_S2", min=1)
        cmp_block_size = Dim("CMP_S2", min=1)
    ori_table_width = Dim("ORI_BT", min=1)
    cmp_table_width = Dim("CMP_BT", min=1)
    ori_slots = 128 if pa_fast else Dim("K1", min=1)
    cmp_slots = 512 if pa_fast else Dim("K2", min=1)
    ori_stride = Dim("ORI_STRIDE", min=1)
    cmp_stride = Dim("CMP_STRIDE", min=1)
    # Divisibility lets the kernel recover its runtime launch grid.
    workspace_bytes = Dim(
        "WORKSPACE_BYTES", min=WS_BYTES_PER_CORE, multiple_of=WS_BYTES_PER_CORE
    )
    i32 = dtypes.int32
    # Argument order matches _run_mqsmla, including absent optional tensors.
    specs = (
        TensorSpec((rows, heads, D), dtypes.bfloat16),
        TensorSpec(
            (
                batch
                if layout_kv == "BSND"
                else Dim(
                    "ORI_PAGES", min=1, max=_PA_FAST_MAX_ORI_PAGES if pa_fast else None
                ),
                ori_block_size,
                N2,
                KV_ROW_BYTES_ORI,
            ),
            dtypes.uint8,
            stride=(ori_stride, KV_ROW_BYTES_ORI, KV_ROW_BYTES_ORI, 1),
        ),
        TensorSpec((rows, N2, ori_slots), i32),
        TensorSpec((batch, ori_table_width), i32) if layout_kv == "PA_BBND" else None,
        TensorSpec((rows, N2), i32) if has_ori_length else None,
        TensorSpec((rows, heads, D), dtypes.bfloat16),
        TensorSpec((heads,), dtypes.float32) if has_sinks else None,
        TensorSpec((query_boundaries,), i32),
        TensorSpec((MQSMLA_METADATA_TOTAL_SIZE,), i32),
        TensorSpec((workspace_bytes,), dtypes.uint8),
        TensorSpec(
            (
                batch
                if layout_kv == "BSND"
                else Dim(
                    "CMP_PAGES", min=1, max=_PA_FAST_MAX_CMP_PAGES if pa_fast else None
                ),
                cmp_block_size,
                N2,
                KV_ROW_BYTES_CMP,
            ),
            dtypes.uint8,
            stride=(cmp_stride, KV_ROW_BYTES_CMP, KV_ROW_BYTES_CMP, 1),
        )
        if has_cmp
        else None,
        TensorSpec((rows, N2, cmp_slots), i32) if has_cmp else None,
        TensorSpec((batch, cmp_table_width), i32)
        if has_cmp and layout_kv == "PA_BBND"
        else None,
        TensorSpec((rows, N2), i32) if has_cmp_length else None,
        TensorSpec((N2, rows, heads), dtypes.float32) if has_lse else None,
        dtypes.float32,
        i32,
        # The AOT signature must distinguish the local and pipelined prepasses.
        TensorSpec((Dim(f"ADDR_WS_BYTES_{addr_mode.upper()}", min=1),), dtypes.uint8)
        if addr_mode != "off"
        else None,
        TensorSpec((query_boundaries,), i32) if layout_kv == "TND" else None,
        TensorSpec((query_boundaries,), i32)
        if layout_kv == "TND" and has_cmp
        else None,
        TensorSpec((batch,), i32) if tnd_padding else None,
        layout_kv,
        head_bucket,
    )
    return cannbotdsl.compile(_run_mqsmla, *specs)


def _mixed_quant_sparse_flash_mla_out(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    kv_batch_strides=None,
    out,
    lse,
):
    """Run MQSMLA with zero-copy query views and direct KV layout addressing."""
    layout_q, layout_kv = str(layout_q).upper(), str(layout_kv).upper()
    if layout_kv == "PA":
        layout_kv = "PA_BBND"
    _validate_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        kv_batch_strides=kv_batch_strides,
    )
    n1 = q.shape[-2]
    for name, tensor, shape, dtype in (
        ("out", out, tuple(q.shape), torch.bfloat16),
        (
            "lse",
            lse,
            (
                (N2, q.shape[0], n1)
                if layout_q == "TND"
                else (q.shape[0], N2, q.shape[1], n1)
            ),
            torch.float32,
        ),
    ):
        if name == "lse" and not return_softmax_lse:
            shape = (0,)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Invalid input: {name} must be a caller-allocated tensor")
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
            or tensor.device != q.device
        ):
            raise ValueError(
                f"Invalid input: {name} must be contiguous {dtype} {shape}"
            )
    # BSND storage is already in flattened query order. Views do not copy Q/O,
    # indices or LSE. Mask padding on device, using the same rule as metadata.
    if layout_q == "BSND":
        batch, seq = q.shape[:2]
        if cu_seqlens_q is None:
            cu_seqlens_q = _bsnd_query_spans(batch, seq, q.device)
        for_cmp = cmp_kv is not None
        if seqused_q is not None:
            if ori_topk_length is None:
                ori_topk_length = torch.full(
                    (batch, seq, N2),
                    ori_sparse_indices.shape[-1],
                    dtype=torch.int32,
                    device=q.device,
                )
            if for_cmp and cmp_topk_length is None:
                cmp_topk_length = torch.full(
                    (batch, seq, N2),
                    cmp_sparse_indices.shape[-1],
                    dtype=torch.int32,
                    device=q.device,
                )
        ori_topk_length = _flatten_bsnd_lengths(ori_topk_length, seqused_q)
        cmp_topk_length = _flatten_bsnd_lengths(cmp_topk_length, seqused_q)
        q, out = q.view(batch * seq, n1, D), out.view(batch * seq, n1, D)
        ori_sparse_indices = ori_sparse_indices.view(batch * seq, N2, -1)
        if for_cmp:
            cmp_sparse_indices = cmp_sparse_indices.view(batch * seq, N2, -1)
        if return_softmax_lse:
            lse = lse.view(N2, batch * seq, n1)
    kernel_layout_kv = layout_kv
    if layout_kv == "TND" and kv_batch_strides is not None:
        # Uniform physical batch spans are a zero-copy BSND view. No prefix
        # tensor, page table or device operation is needed for KV addressing.
        batch = cu_seqlens_q.numel() - 1
        ori_kv = ori_kv.view(batch, kv_batch_strides[0], N2, KV_ROW_BYTES_ORI)
        if cmp_kv is not None:
            cmp_kv = cmp_kv.view(batch, kv_batch_strides[1], N2, KV_ROW_BYTES_CMP)
        kernel_layout_kv = "BSND"
    elif layout_kv == "TND":
        ori_kv = ori_kv.view(ori_kv.shape[0], 1, N2, KV_ROW_BYTES_ORI)
        if cmp_kv is not None:
            cmp_kv = cmp_kv.view(cmp_kv.shape[0], 1, N2, KV_ROW_BYTES_CMP)
    # An absent length tensor activates all sparse-index columns; the kernel
    # reads K from the index shape without allocating a replacement tensor.
    addr_ws = None
    addr_bytes = 0
    addr_mode = _select_addr_mode(
        ori_kv,
        ori_block_table,
        ori_sparse_indices,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
    )
    if addr_mode != "off":
        columns = _addr_tab_w(ori_sparse_indices.shape[2])
        if cmp_kv is not None:
            columns += _addr_tab_w(cmp_sparse_indices.shape[2])
        addr_bytes = q.shape[0] * columns * 4
    # Allocate FD staging and the address table together.
    workspace_full = construct_workspace(metadata, addr_bytes)
    if addr_bytes:
        workspace = workspace_full[:-addr_bytes]
        addr_ws = workspace_full[-addr_bytes:]
    else:
        workspace = workspace_full
    tensors = (
        q,
        ori_kv,
        ori_sparse_indices,
        ori_block_table,
        ori_topk_length,
        out,
        sinks,
        cu_seqlens_q,
        metadata,
        workspace,
        cmp_kv,
        cmp_sparse_indices,
        cmp_block_table,
        cmp_topk_length,
        lse if return_softmax_lse else None,
    )
    compiled = _get_compiled_kernel(
        ori_kv,
        ori_sparse_indices,
        ori_block_table,
        ori_topk_length,
        cmp_kv,
        cmp_sparse_indices,
        cmp_block_table,
        cmp_topk_length,
        lse if return_softmax_lse else None,
        kernel_layout_kv,
        n1,
        sinks is not None,
        layout_q == "TND" and seqused_q is not None,
    )
    # AOT fixes absent Tensor parameters at compile time and removes them from
    # the runtime signature. Remaining Tensor arguments keep their order.
    scale = (
        float(softmax_scale)
        if softmax_scale is not None
        else 1.0 / math.sqrt(q.shape[-1])
    )
    batch_consistency = 1 if _batch_consistency_enabled() else 0
    compiled(
        *(tensor for tensor in tensors if tensor is not None),
        scale,
        batch_consistency,
        *([addr_ws] if addr_ws is not None else []),
        *(
            [cu_seqlens_ori_kv] + ([cu_seqlens_cmp_kv] if cmp_kv is not None else [])
            if kernel_layout_kv == "TND"
            else []
        ),
        *([seqused_q] if layout_q == "TND" and seqused_q is not None else []),
    )


def mixed_quant_sparse_flash_mla(
    q,
    quant_mode,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    rope_head_dim=None,
    softmax_scale=None,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
    key_dtype=None,
    value_dtype=None,
):
    """ops-transformer parameter/return interface, retaining mixed FP8/FP4 KV.

    KV encoding is mixed FP8/FP4; metadata comes from the existing library operator.
    Non-default masks and non-sparse modes are rejected, not silently ignored.
    """
    _validate_api_modes(
        quant_mode,
        rope_head_dim,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        cmp_residual_kv,
    )
    if key_dtype is not None or value_dtype is not None:
        raise NotImplementedError(
            "Explicit KV dtype overrides are not supported for the fixed mixed FP8/FP4 encoding"
        )
    if topk_value_mode != 1:
        raise NotImplementedError("Only topk_value_mode=1 is supported")
    if not isinstance(q, torch.Tensor):
        raise ValueError("q must be a Tensor")
    if not isinstance(metadata, torch.Tensor) or metadata.numel() != 2048:
        raise ValueError(
            "metadata must be the int32[2048] result of the library metadata operator"
        )
    layout_q, layout_kv = str(layout_q).upper(), str(layout_kv).upper()
    if layout_q not in ("BSND", "TND"):
        raise ValueError("layout_q must be BSND or TND")
    out = torch.empty_like(q)
    lse_shape = (
        (1, q.shape[0], q.shape[-2])
        if layout_q == "TND"
        else (q.shape[0], 1, q.shape[1], q.shape[-2])
    )
    lse = torch.empty(
        lse_shape if return_softmax_lse else (0,), dtype=torch.float32, device=q.device
    )
    _mixed_quant_sparse_flash_mla_out(
        q,
        quant_mode=quant_mode,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=None if layout_q == "BSND" else cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata[:MQSMLA_METADATA_TOTAL_SIZE],
        softmax_scale=softmax_scale,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=return_softmax_lse,
        out=out,
        lse=lse,
    )
    return out, lse


def _variant_calls():
    """Enumerate pickleable configurations for AOT variant workers."""
    for tnd_padding, has_sinks, head_bucket, has_cmp in product(
        (False, True), (True, False), (64, 32, 63), (True, False)
    ):
        cmp_lengths = (True, False) if has_cmp else (False,)
        for has_ori_length, has_cmp_length, has_lse in product(
            (True, False), cmp_lengths, (True, False)
        ):
            args = (has_cmp, has_ori_length, has_cmp_length, has_lse)
            options = dict(
                head_bucket=head_bucket,
                has_sinks=has_sinks,
                tnd_padding=tnd_padding,
            )
            for addr_mode in ("pre", "off", "arena"):
                yield (*args, addr_mode), dict(layout_kv="PA_BBND", **options)
                if addr_mode != "off":
                    yield (
                        (*args, addr_mode),
                        dict(layout_kv="PA_BBND", pa_fast=True, **options),
                    )
            for layout_kv in ("TND", "BSND"):
                yield (*args, "off"), dict(layout_kv=layout_kv, **options)


@aot.export("mixed_quant_sparse_flash_mla_dsl")
def export_mqsmla():
    """Register the operator for OpKit AOT variant workers."""
    raise RuntimeError(
        "MQSMLA AOT export requires OpKit variant workers; "
        "build with opkit.build(jobs=N), N >= 2, and available jobserver slots"
    )


export_mqsmla.__cannbot_aot_variants__ = ("_get_compiled_variant", _variant_calls)
