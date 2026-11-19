# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""WeLMv4 QKV 投影、后处理与 KV cache 散写融合算子。

单个 MIX_AIC_1_2 复合核完成以下数据流（每 TP rank，2560 列）：

    qkv(M, 2560) = hidden(M, 2048) @ weight(2560, 2048)^T      (Cube, FP32 累加)
    ├── [0:1536]     6 个 Q head：尾 64 维 RoPE         -> q(M, 1536)
    ├── [1536:1792]  K head：RMSNorm -> 尾 64 维 RoPE    -> k(M, 256) / k_cache[slot]
    ├── [1792:2048]  V head：FP32 -> BF16                -> v_cache[slot]
    ├── [2048:2304]  mirror_k：直通                     -> mirror_k(M, 256)
    └── [2304:2560]  mirror_v：直通                     -> mirror_v(M, 256)

L0C 的 FP32 结果经 FIXPIPE 按行折半直达两个 AIV 的 UB，向量段就地完成
RoPE/RMSNorm/量化与散写，QKV 中间张量不再落 GM。权重 (2560, 2048) 按
head 单元切成 10 个 N tile，与 Q/K/V/mirror 输出一一对应。

数据流参照 sglang WeLMv4 的 fused_qkv_proj_norm_rope_cache。
"""

import cannbotdsl
import torch
from cannbotdsl import Constexpr, MemLoc, Tensor, TensorSpec, aot, dtypes
from cannbotdsl.lang import jit, kernel, vf
from cannbotdsl.lang import range as dsl_range
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import copy_options, cv_valid_extent, mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    full_mask,
    update_mask,
    vadd,
    vadds,
    vcast,
    vdiv,
    vdup,
    vdups,
    vgts,
    vload,
    vload_unpack,
    vmaxs,
    vmul,
    vmuls,
    vreduce_sum,
    vselect,
    vsqrt,
    vstore_pack,
    vsub,
)
from cannbotdsl.ops.reg.cast import RoundingMode
from cannbotdsl.ops.sync import vec_sync_all
from cannbotdsl.ring_buffer import RingBuffer
from cannbotdsl.tensor import make_layout, tile_slice
from cannbotdsl.types.dtypes import bfloat16 as BF16
from cannbotdsl.types.dtypes import float32 as F32

# ---- model structure constants (per TP rank, WeLMv4 NPU path) ----
HIDDEN = 2048  # K of the projection
HEAD_DIM = 256
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE_DIM = ROPE_DIM // 2
NUM_Q_HEADS = 6
NUM_UNITS = 10  # 6 Q + K + V + mirror_k + mirror_v
QKV_WIDTH = NUM_UNITS * HEAD_DIM  # 2560
UNIT_K = NUM_Q_HEADS
UNIT_V = NUM_Q_HEADS + 1
UNIT_MK = NUM_Q_HEADS + 2

# ---- tiling constants (arch35: L1 512K, L0A/B 64K, L0C 256K, AIC 32) ----
# Cube tiles are PAIR_M=256 rows: two consecutive BASE_M=128 m-tiles form ONE
# contiguous (256, K) GM region of hidden, so a single A load serves both and
# the B (weight) slice is streamed once per pair instead of once per tile —
# L1 traffic drops from (0.5+1)MB x n_units x 129 tiles = 1.95GB to
# (1+1)MB x n_units x 65 pairs = 1.30GB.
PAIR_M = 256
BASE_N = 256  # exactly one head-unit per N tile (epilogue granularity)
BASE_K = 64  # L0B (256,64) bf16 depth2 == 64KB; depth2 pipelining beats halved issue count (measured: depth1@128 is 31% slower)
K_L1 = 256
K_L1_TILES = HIDDEN // K_L1  # 8
K_L0_PER_L1 = K_L1 // BASE_K  # 4
TILE_VEC_M = PAIR_M // 2  # per-AIV rows after split-M
AIC_NUM = 32

VL = 64  # vector register elements (2048b / 32b)
AVG = 1.0 / HEAD_DIM
_F32_MAX = float.fromhex("0x1.fffffep+127")


@kernel
class FusedQkvProjNormRopeCacheKernel:
    def __init__(
        self,
        return_v: bool = False,
        has_mirror: bool = True,
        positions_contiguous: bool = True,
    ):
        # Trace-time constants: fold the optional v materialization, the
        # mirror block-store, and the cos/sin load mode away before IR
        # emission. ``has_mirror`` must be a host-side fact (plain layers
        # pass (1, 256) mirror stand-ins whose shape cannot satisfy the
        # block store's static partition check). ``positions_contiguous``
        # selects the prefill (bulk cos/sin DMA on a runtime-contiguous
        # position run) vs decode (per-token gather for arbitrary
        # positions) cos/sin preload; both artifacts share the identical
        # epilogue math.
        self.return_v = bool(return_v)
        self.has_mirror = bool(has_mirror)
        self.positions_contiguous = bool(positions_contiguous)

    @jit
    def _rope_q(self, in_ch, out_ch, cs_ch, cur_rows):
        """Q head unit: 192-col passthrough + tail-64 NeoX rotate-half.

        Input is the fp32 cross-core tile; bf16 RN rounding happens at store.
        A second load at +32 puts this row's x2 in the low lanes, so every
        rotate-half pair is lane-aligned and computed with 32-lane masked
        ops -- no UB scratch round-trip, no vmem_bar, no vgather. The +32
        load of the last row spans into cv_ub's pad row (never written; its
        lanes are masked out).
        """
        with vf(mode="simd"):
            full = full_mask()
            m_lo = update_mask(HALF_ROPE_DIM, elem_bits=32)[0]

            for r in dsl_range(0, cur_rows, 1):
                base = r * HEAD_DIM
                for j in range(3):
                    off = base + j * VL
                    x = vload(in_ch, off)
                    vstore_pack(
                        out_ch,
                        off,
                        vcast(x, BF16, mask=full, rounding=RoundingMode.RN),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                # Round the fp32 MMAD result to bf16 before the RoPE input:
                # the production path materializes qkv as bf16 in GM and the
                # rope kernel reads it back (bf16 -> fp32 compute).
                x3 = vload(in_ch, base + NOPE_DIM)
                x3 = vcast(
                    vcast(x3, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                # Lanes 0-31 of the +32 load hold this row's x2; lanes 32-63
                # hold the next row's x1 (or the pad row's stale data on the
                # last row) and are masked out below.
                x3n = vload(in_ch, base + NOPE_DIM + HALF_ROPE_DIM)
                x3n = vcast(
                    vcast(x3n, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                c = vload(cs_ch, r * ROPE_DIM)  # [cos||sin]
                s = vload(cs_ch, r * ROPE_DIM + HALF_ROPE_DIM)  # [sin|..]
                t1 = vmul(x3, c, mask=m_lo)  # x1*cos
                t2 = vmul(x3n, s, mask=m_lo)  # x2*sin
                o_lo = vsub(t1, t2, mask=m_lo)
                t3 = vmul(x3, s, mask=m_lo)  # x1*sin
                t4 = vmul(x3n, c, mask=m_lo)  # x2*cos
                o_hi = vadd(t3, t4, mask=m_lo)
                vstore_pack(
                    out_ch,
                    base + NOPE_DIM,
                    vcast(o_lo, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    m_lo,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    out_ch,
                    base + NOPE_DIM + HALF_ROPE_DIM,
                    vcast(o_hi, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    m_lo,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def _norm_rope_k(self, in_ch, out_ch, cs_ch, gamma_ch, cur_rows, avg, epsilon):
        """K head unit: per-head fp32 RMSNorm (Newton rsqrt) + tail-64 RoPE.

        RoPE uses the same dual-load low-lane scheme as _rope_q: the norm is
        applied to both halves separately (y1 from the base load, y2 from the
        +32 load), keeping all rotate-half pairs lane-aligned without
        gather/vmem_bar.
        """
        with vf(mode="simd"):
            full = full_mask()
            m_lo = update_mask(HALF_ROPE_DIM, elem_bits=32)[0]
            one = vdups(1.0, F32, mask=full)
            half = vdups(0.5, F32, mask=full)
            one_half = vdups(1.5, F32, mask=full)
            zero = vdups(0.0, F32, mask=full)
            fmax = vdups(_F32_MAX, F32, mask=full)

            gu0 = vload_unpack(gamma_ch, 0, unpack_mode=UnpackMode.B16_TO_B32)
            gu1 = vload_unpack(gamma_ch, VL, unpack_mode=UnpackMode.B16_TO_B32)
            gu2 = vload_unpack(gamma_ch, 2 * VL, unpack_mode=UnpackMode.B16_TO_B32)
            gu3 = vload_unpack(gamma_ch, NOPE_DIM, unpack_mode=UnpackMode.B16_TO_B32)
            # gamma[224:256] re-unpacked at +32 so g2 lands in the low lanes
            # (uses the 32-element tail slack on gamma_ch).
            gu3n = vload_unpack(
                gamma_ch, NOPE_DIM + HALF_ROPE_DIM, unpack_mode=UnpackMode.B16_TO_B32
            )
            g0 = vcast(gu0, F32, mask=full)
            g1 = vcast(gu1, F32, mask=full)
            g2 = vcast(gu2, F32, mask=full)
            g3 = vcast(gu3, F32, mask=full)  # lanes 0-31 = gamma[192:224]
            g3n = vcast(gu3n, F32, mask=full)  # lanes 0-31 = gamma[224:256]

            for r in dsl_range(0, cur_rows, 1):
                base = r * HEAD_DIM
                # bf16 round-trip before the RMSNorm input: the production
                # path materializes qkv as bf16 in GM and the norm kernel
                # reads it back (bf16 -> fp32 accumulate).
                x0 = vload(in_ch, base)
                x0 = vcast(
                    vcast(x0, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                x1 = vload(in_ch, base + VL)
                x1 = vcast(
                    vcast(x1, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                x2 = vload(in_ch, base + 2 * VL)
                x2 = vcast(
                    vcast(x2, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                x3 = vload(in_ch, base + NOPE_DIM)
                x3 = vcast(
                    vcast(x3, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )
                x3n = vload(in_ch, base + NOPE_DIM + HALF_ROPE_DIM)
                x3n = vcast(
                    vcast(x3n, BF16, mask=full, rounding=RoundingMode.RN),
                    F32,
                    mask=full,
                )

                t0 = vreduce_sum(vmul(x0, x0, mask=full), mask=full)
                t1 = vreduce_sum(vmul(x1, x1, mask=full), mask=full)
                t2 = vreduce_sum(vmul(x2, x2, mask=full), mask=full)
                t3 = vreduce_sum(vmul(x3, x3, mask=full), mask=full)
                total = vadd(
                    vadd(t0, t1, mask=full), vadd(t2, t3, mask=full), mask=full
                )
                var = vadds(vmuls(total, avg, mask=full), epsilon, mask=full)
                var = vmaxs(var, -99.99, mask=full)

                recip = vdiv(one, var, mask=full)
                y = vsqrt(recip, mask=full)
                t = vmuls(var, -0.5, mask=full)
                t = vmul(t, y, mask=full)
                t1n = vadd(one_half, vmul(t, y, mask=full), mask=full)
                rstd = vmul(y, t1n, mask=full)
                t3n = vmuls(var, -1.0, mask=full)
                s = vadd(one, vmul(t3n, recip, mask=full), mask=full)
                t4 = vmuls(rstd, -1.0, mask=full)
                recip = vadd(recip, vmul(t4, rstd, mask=full), mask=full)
                s = vadd(s, vmul(var, recip, mask=full), mask=full)
                s = vmul(s, rstd, mask=full)
                rstd = vadd(rstd, vmul(s, half, mask=full), mask=full)
                cmp_inf = vgts(var, _F32_MAX * 0.999, mask=full)
                rstd = vselect(zero, rstd, cond_mask=cmp_inf)
                cmp_pos = vgts(var, 0.0, mask=full)
                rstd = vselect(rstd, fmax, cond_mask=cmp_pos)
                rstd_b = vdup(rstd, mask=full)

                y0 = vmul(vmul(x0, rstd_b, mask=full), g0, mask=full)
                y1 = vmul(vmul(x1, rstd_b, mask=full), g1, mask=full)
                y2 = vmul(vmul(x2, rstd_b, mask=full), g2, mask=full)
                vstore_pack(
                    out_ch,
                    base,
                    vcast(y0, BF16, mask=full, rounding=RoundingMode.RN),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    out_ch,
                    base + VL,
                    vcast(y1, BF16, mask=full, rounding=RoundingMode.RN),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    out_ch,
                    base + 2 * VL,
                    vcast(y2, BF16, mask=full, rounding=RoundingMode.RN),
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )

                # Norm applied to both rope halves in the low lanes; round
                # each back to bf16 (original two-kernel GM round-trip).
                y1r = vmul(vmul(x3, rstd_b, mask=m_lo), g3, mask=m_lo)
                y2r = vmul(vmul(x3n, rstd_b, mask=m_lo), g3n, mask=m_lo)
                y1r = vcast(
                    vcast(y1r, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    F32,
                    mask=m_lo,
                )
                y2r = vcast(
                    vcast(y2r, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    F32,
                    mask=m_lo,
                )
                c = vload(cs_ch, r * ROPE_DIM)
                s = vload(cs_ch, r * ROPE_DIM + HALF_ROPE_DIM)
                o_lo = vsub(vmul(y1r, c, mask=m_lo), vmul(y2r, s, mask=m_lo), mask=m_lo)
                o_hi = vadd(vmul(y1r, s, mask=m_lo), vmul(y2r, c, mask=m_lo), mask=m_lo)
                vstore_pack(
                    out_ch,
                    base + NOPE_DIM,
                    vcast(o_lo, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    m_lo,
                    pack_mode=PackMode.B32_TO_B16,
                )
                vstore_pack(
                    out_ch,
                    base + NOPE_DIM + HALF_ROPE_DIM,
                    vcast(o_hi, BF16, mask=m_lo, rounding=RoundingMode.RN),
                    m_lo,
                    pack_mode=PackMode.B32_TO_B16,
                )

    @jit
    def _cast_rows(self, in_ch, out_ch, cur_rows):
        """V / mirror units: fp32 tile -> bf16 RN passthrough."""
        with vf(mode="simd"):
            full = full_mask()
            for r in dsl_range(0, cur_rows, 1):
                base = r * HEAD_DIM
                for j in range(4):
                    off = base + j * VL
                    x = vload(in_ch, off)
                    vstore_pack(
                        out_ch,
                        off,
                        vcast(x, BF16, mask=full, rounding=RoundingMode.RN),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )

    def __call__(
        self,
        gm_hidden: Tensor,
        gm_weight: Tensor,
        gm_gamma: Tensor,
        gm_positions: Tensor,
        gm_cos_sin: Tensor,
        gm_slot: Tensor,
        gm_q: Tensor,
        gm_k: Tensor,
        gm_v: Tensor,
        gm_mk: Tensor,
        gm_mv: Tensor,
        gm_k_cache: Tensor,
        gm_v_cache: Tensor,
        epsilon,
    ):
        """Original 14-argument device entry; helper is trace-time inlined."""
        self._run(
            gm_hidden,
            gm_weight,
            gm_gamma,
            gm_positions,
            gm_cos_sin,
            gm_slot,
            gm_q,
            gm_k,
            gm_v,
            gm_mk,
            gm_mv,
            gm_k_cache,
            gm_v_cache,
            epsilon,
        )

    @jit
    def _run(
        self,
        gm_hidden: Tensor,
        gm_weight: Tensor,
        gm_gamma: Tensor,
        gm_positions: Tensor,
        gm_cos_sin: Tensor,
        gm_slot: Tensor,
        gm_q: Tensor,
        gm_k: Tensor,
        gm_v: Tensor,
        gm_mk: Tensor,
        gm_mv: Tensor,
        gm_k_cache: Tensor,
        gm_v_cache: Tensor,
        epsilon,
    ):
        block_idx = get_block_idx()
        block_num = get_block_num()
        sub = get_subblock_id()

        m_rows = gm_hidden.shape[0]
        m_pairs = (m_rows + PAIR_M - 1) // PAIR_M
        # Unit count follows the projection width: 2560 = mirror-source
        # layer (10 units, Q/K/V/mirror_k/mirror_v), 2048 = plain layer
        # (8 units, Q/K/V). The per-unit dispatch below is width-agnostic;
        # mirror branches simply never fire for the 8-unit layout.
        n_units = gm_weight.shape[0] // BASE_N

        # ---- cube-side channels (L1 = 256+256 = 512KB exact; the arena
        # matmul sample runs the identical baseM=256 configuration) ----
        l1_a = RingBuffer(MemLoc.L1, BF16, make_layout((PAIR_M, K_L1)), depth=2)
        l1_b = RingBuffer(MemLoc.L1, BF16, make_layout((BASE_N, K_L1)), depth=2)
        l0a = RingBuffer(MemLoc.L0A, BF16, make_layout((PAIR_M, BASE_K)), depth=2)
        l0b = RingBuffer(MemLoc.L0B, BF16, make_layout((BASE_N, BASE_K)), depth=2)
        # 256x256 f32 == 256KB L0C: depth1 (the pair tile trades fixpipe
        # double-buffering for the doubled M; the next item's first MMAD
        # still overlaps this item's fixpipe drain).
        l0c = RingBuffer(MemLoc.L0C, F32, make_layout((PAIR_M, BASE_N)), depth=1)
        # ---- cross-core handoff (fp32: dual_dst_ctl forbids narrowing cast) ----
        # One extra pad row: the rope fast path re-loads x2 at +32, whose
        # 64-lane load on the last row spans into the next row. The pad row
        # is never written; its stale lanes are masked out by the 32-lane
        # rope math. Keeping the pad as a whole row preserves the (128, 256)
        # contiguous fixpipe destination layout.
        cv_ub = RingBuffer(
            MemLoc.UB, F32, make_layout((TILE_VEC_M + 1, BASE_N)), depth=1
        )
        # ---- vec-side channels ----
        out_ch = RingBuffer(
            MemLoc.UB, BF16, make_layout((TILE_VEC_M, HEAD_DIM)), depth=1
        )
        # Pad row for the rope fast path's +32 cos/sin load on the last row.
        cs_ch = RingBuffer(
            MemLoc.UB, F32, make_layout((TILE_VEC_M + 1, ROPE_DIM)), depth=1
        )
        # 32-element tail slack: _norm_rope_k re-unpacks gamma[224:256] at
        # offset 224 (64 bf16 lanes) so g2 lands in the low lanes; the slack
        # lanes are written never and read only as masked-out garbage.
        gamma_ch = RingBuffer(
            MemLoc.UB, BF16, make_layout((1, HEAD_DIM + HALF_ROPE_DIM)), depth=1
        )

        # Work-item scheduling: granularity is (pair, unit). Whole-tile
        # assignment gives 65 pairs / 32 cores = 2.03 waves — one core runs a
        # 3rd wave while the rest idle. Item granularity spreads
        # m_pairs*n_units items evenly (<=3.4% skew).
        total_items = m_pairs * n_units
        for item in range(block_idx, total_items, block_num):
            pair = item // n_units
            u = item - pair * n_units
            hidden_span = gm_hidden
            q_span = gm_q
            k_span = gm_k
            v_span = gm_v
            mk_span = gm_mk
            mv_span = gm_mv

            cur_tokens = min(m_rows - pair * PAIR_M, PAIR_M)

            # Balanced split-M plan for this (possibly tail-clipped) pair
            # tile; a trailing odd tile clips to its real row count.
            q_tile0 = tile_slice(q_span, (PAIR_M, HEAD_DIM), (pair, 0))
            rows_here = cv_valid_extent(q_tile0, axis=0, part_id=sub)
            sub_start = sub * (cur_tokens - rows_here)
            row0 = pair * PAIR_M + sub_start
            # A trailing subblock may own zero rows (e.g. M == 1 splits
            # 1/0); clamp so the scalar position load stays in bounds even
            # though every per-row loop below is empty.
            if row0 >= m_rows:
                row0 = m_rows - 1
            meta_row0 = row0

            # Preserve the existing per-item gamma preload.
            gamma_w = gamma_ch.next_write()
            mem_copy(
                tile_slice(gamma_w, (1, HEAD_DIM), (0, 0)),
                tile_slice(gm_gamma, (1, HEAD_DIM), (0, 0)),
            )

            # Vec-side preload for this item (this AIV's cos/sin rows);
            # item granularity re-issues it per unit (~16 KB DMA, noise).
            cs_w = cs_ch.next_write()
            # Table extent is a compile-time constant under every spec form
            # (JIT trace or AOT TensorSpec): only M is dynamic.
            table_rows = gm_cos_sin.shape[0]
            if const_expr(self.positions_contiguous):
                # Prefill: one run of TILE_VEC_M consecutive table rows.
                pos_base = gm_positions[meta_row0]
                if rows_here > 0:
                    if pos_base + TILE_VEC_M <= table_rows:
                        # Fast path: one bulk (<=128, 64) DMA.
                        mem_copy(
                            tile_slice(cs_w, (TILE_VEC_M, ROPE_DIM), (0, 0)),
                            gm_cos_sin[pos_base : pos_base + TILE_VEC_M, 0:ROPE_DIM],
                        )
                    else:
                        # Table-tail clamp: the bulk read would overrun the
                        # cos/sin table (e.g. a context filled to its last
                        # position). Fall back to exactly the valid rows --
                        # pos_base + rows_here <= table_rows holds because
                        # every position is range-checked.
                        for r in range(rows_here):
                            mem_copy(
                                tile_slice(cs_w, (1, ROPE_DIM), (r, 0)),
                                tile_slice(
                                    gm_cos_sin, (1, ROPE_DIM), (pos_base + r, 0)
                                ),
                            )
            else:
                # Decode: arbitrary per-token positions (each request's own
                # last position). One 1x64 gather per token -- the production
                # generic rope kernel's addressing, at decode-sized M.
                for r in range(rows_here):
                    pos = gm_positions[meta_row0 + r]
                    mem_copy(
                        tile_slice(cs_w, (1, ROPE_DIM), (r, 0)),
                        tile_slice(gm_cos_sin, (1, ROPE_DIM), (pos, 0)),
                    )

            # ---- cube: K-loop MMAD accumulation for unit u ----
            vec_sync_all()
            l0c_acc = l0c.next_write()
            for k_l1 in range(K_L1_TILES):
                a_tile = tile_slice(hidden_span, (PAIR_M, K_L1), (pair, k_l1))
                b_tile = tile_slice(gm_weight, (BASE_N, K_L1), (u, k_l1))
                # l2_cache_ctl=1 (normal): W tiles stay resident in L2
                # across m-tiles; the disabled default re-fetches the
                # 10.5 MB weight from HBM for every m-tile.
                mem_copy(
                    l1_a.next_write(),
                    a_tile,
                    options=copy_options(l2_cache_ctl=1),
                )
                mem_copy(
                    l1_b.next_write(),
                    b_tile,
                    options=copy_options(l2_cache_ctl=1),
                )
                l1_a_slot = l1_a.next_read()
                l1_b_slot = l1_b.next_read()
                for k_l0 in range(K_L0_PER_L1):
                    mem_copy(
                        l0a.next_write(),
                        tile_slice(l1_a_slot, (PAIR_M, BASE_K), (0, k_l0)),
                    )
                    mem_copy(
                        l0b.next_write(),
                        tile_slice(l1_b_slot, (BASE_N, BASE_K), (0, k_l0)),
                    )
                    l0a_slot = l0a.next_read()
                    l0b_slot = l0b.next_read()
                    gk = k_l1 * K_L0_PER_L1 + k_l0
                    matmul(l0c_acc, l0a_slot, l0b_slot, init=(gk == 0))

            # ---- handoff: L0C -> per-AIV UB halves ----
            # split_axis=0 on an L0C->UB copy is the dual-destination
            # FIXPIPE: each AIV receives its balanced half of the
            # tail-clipped (cur_tokens, N) tile directly in its own UB.
            cv_w = cv_ub.next_write()
            mem_copy(
                tile_slice(cv_w, (TILE_VEC_M, BASE_N), (0, 0)),
                l0c_acc[0:cur_tokens, None].partition(dim=0),
            )

            # ---- vec epilogue, specialized per unit ----
            # Dense outputs use one partitioned block copy per item (op①
            # CopyoutProto's blockCount=rows pattern) instead of per-row
            # 512B DMAs; only the slot scatter keeps row granularity.
            cv_r = cv_ub.next_read()
            cs_r = cs_ch.next_read()
            out_w = out_ch.next_write()
            if u < NUM_Q_HEADS:
                self._rope_q(cv_r, out_w, cs_r, rows_here)
                q_half = tile_slice(q_span, (PAIR_M, HEAD_DIM), (pair, u)).partition(
                    dim=0, part_id=sub
                )
                mem_copy(q_half, out_ch.next_read()[0:rows_here, None])
            elif u == UNIT_K:
                self._norm_rope_k(cv_r, out_w, cs_r, gamma_w, rows_here, AVG, epsilon)
                out_r = out_ch.next_read()
                k_half = tile_slice(k_span, (PAIR_M, HEAD_DIM), (pair, 0)).partition(
                    dim=0, part_id=sub
                )
                # The tile drain and the slot scatter are two readers of one
                # out_ch slot; both consume the same VF-written epoch.
                mem_copy(k_half, out_r[0:rows_here, None])
                for r in range(rows_here):
                    slot = gm_slot[meta_row0 + r]
                    if slot != -1:
                        mem_copy(
                            tile_slice(gm_k_cache, (1, HEAD_DIM), (slot, 0)),
                            tile_slice(out_r, (1, HEAD_DIM), (r, 0)),
                        )
            elif u == UNIT_V:
                self._cast_rows(cv_r, out_w, rows_here)
                out_r = out_ch.next_read()
                if const_expr(self.return_v):
                    # Transitional interface: materialize v like the
                    # production scatter's transient contiguous copy.
                    v_half = tile_slice(
                        v_span, (PAIR_M, HEAD_DIM), (pair, 0)
                    ).partition(dim=0, part_id=sub)
                    mem_copy(v_half, out_r[0:rows_here, None])
                for r in range(rows_here):
                    slot = gm_slot[meta_row0 + r]
                    if slot != -1:
                        mem_copy(
                            tile_slice(gm_v_cache, (1, HEAD_DIM), (slot, 0)),
                            tile_slice(out_r, (1, HEAD_DIM), (r, 0)),
                        )
            else:
                self._cast_rows(cv_r, out_w, rows_here)
                out_r = out_ch.next_read()
                gm_m = mk_span if u == UNIT_MK else mv_span
                if const_expr(self.has_mirror):
                    m_half = tile_slice(gm_m, (PAIR_M, HEAD_DIM), (pair, 0)).partition(
                        dim=0, part_id=sub
                    )
                    mem_copy(m_half, out_r[0:rows_here, None])
                else:
                    # Plain-layer stand-ins are (1, 256); the mirror units
                    # never execute at runtime, so per-row runtime coords
                    # (which pass static verification) suffice.
                    for r in range(rows_here):
                        mem_copy(
                            tile_slice(gm_m, (1, HEAD_DIM), (row0 + r, 0)),
                            tile_slice(out_r, (1, HEAD_DIM), (r, 0)),
                        )


def _core_budget(device=None):
    """Return (device, launchable MIX groups, AIV count) for the current stream.

    A MIX block uses one AIC and two AIV, so the launchable group count is
    ``min(cube_num, vector_num // 2)``.  The live stream quota is queried on
    every call and enters both the launch count and the compile cache key.
    """
    from cannbotdsl.ops.info import get_platform_info

    stream = None
    try:
        stream = torch.npu.current_stream(device)
    except (AttributeError, RuntimeError):
        stream = None
    try:
        info = (
            get_platform_info(stream=stream)
            if stream is not None
            else get_platform_info()
        )
    except (RuntimeError, TypeError):
        info = get_platform_info()
    if (
        info.available
        and int(info.cube_core_num) > 0
        and int(info.vector_core_num) >= 2
    ):
        cube = int(info.cube_core_num)
        vector = int(info.vector_core_num)
        return device, min(cube, vector // 2), vector
    return device, AIC_NUM, AIC_NUM * 2


_COMPILED = {}
_TORCH_TO_DSL = {
    torch.bfloat16: dtypes.bfloat16,
    torch.float32: dtypes.float32,
    torch.int32: dtypes.int32,
    torch.int64: dtypes.int64,
}


def _tensor_spec(tensor):
    return TensorSpec(
        tuple(int(v) for v in tensor.shape),
        _TORCH_TO_DSL[tensor.dtype],
        stride=tuple(int(v) for v in tensor.stride()),
        storage_format="nd",
    )


def _compile_program(op, specs, eps):
    """Compile the host entry once per specialization and reuse it.

    The effective MIX group count from the input stream is part of the key:
    lowering the stream quota must produce a program with a matching
    ``block_num`` instead of reusing an over-subscribed launch.
    """
    key = (
        op._return_v,
        op._has_mirror,
        op._positions_contiguous,
        int(op.block_num),
        float(eps),
        specs,
    )
    program = _COMPILED.get(key)
    if program is None:
        program = cannbotdsl.compile(
            op.run,
            *specs,
            eps=float(eps),
            block_num=int(op.block_num),
            return_v=bool(op._return_v),
            has_mirror=bool(op._has_mirror),
            positions_contiguous=bool(op._positions_contiguous),
        )
        _COMPILED[key] = program
    return program


def _compiled_run(op, tensors, eps):
    """Compile the host entry once per tensor contract and reuse it."""
    specs = tuple(_tensor_spec(tensor) for tensor in tensors)
    return _compile_program(op, specs, eps)


def clear_caches():
    """Close cached compiled programs (useful for tests and REPL sessions)."""
    for program in _COMPILED.values():
        program.close()
    _COMPILED.clear()


def _export_specs(tokens, has_mirror, return_v, positions_contiguous):
    """Build the TensorSpec tuple for one AOT profile."""
    units = NUM_UNITS if has_mirror else NUM_UNITS - 2
    n = units * HEAD_DIM
    specs = [
        TensorSpec((tokens, HIDDEN), dtypes.bfloat16),
        TensorSpec((n, HIDDEN), dtypes.bfloat16),
        TensorSpec((1, HEAD_DIM), dtypes.bfloat16),
        TensorSpec((tokens,), dtypes.int64),
        TensorSpec((8192, ROPE_DIM), dtypes.float32),
        TensorSpec((tokens,), dtypes.int64),
        TensorSpec((tokens, NUM_Q_HEADS * HEAD_DIM), dtypes.bfloat16),
        TensorSpec((tokens, HEAD_DIM), dtypes.bfloat16),
        TensorSpec((tokens, HEAD_DIM), dtypes.bfloat16),
        TensorSpec(
            (tokens, HEAD_DIM) if has_mirror else (1, HEAD_DIM), dtypes.bfloat16
        ),
        TensorSpec(
            (tokens, HEAD_DIM) if has_mirror else (1, HEAD_DIM), dtypes.bfloat16
        ),
        TensorSpec((8192, HEAD_DIM), dtypes.bfloat16),
        TensorSpec((8192, HEAD_DIM), dtypes.bfloat16),
    ]
    return tuple(specs)


@aot.export("fused_qkv_proj_norm_rope_cache")
def export_fused_qkv_proj_norm_rope_cache():
    """Compile representative prefill/decode specializations.

    The kv-mirror layer (N=2560) is the production path; the plain 8-unit
    layer keeps a single profile.  Other token counts and the remaining flag
    combinations are compiled on first use.
    """
    profiles = (
        (True, False, True, 1, 32),
        (True, False, True, 256, 32),
        (True, False, False, 1, 32),
        (True, False, False, 256, 32),
        (False, False, True, 72, 32),
        (False, False, False, 72, 32),
        (True, False, True, 256, 28),
        (True, False, False, 256, 28),
    )
    for has_mirror, return_v, contiguous, t, quota in profiles:
        op = FusedQkvProjNormRopeCache(
            return_v=return_v,
            has_mirror=has_mirror,
            positions_contiguous=contiguous,
            block_num=quota,
        )
        _compile_program(op, _export_specs(t, has_mirror, return_v, contiguous), 1e-6)


class FusedQkvProjNormRopeCache:
    """Host-facing wrapper: projection + post-processing + cache scatter in 1 kernel.

    ``return_v=False`` (default) matches the profiled NPU dataflow: the sink
    prefill attention reads K/V from the paged cache, so v is only scattered.
    ``return_v=True`` additionally materializes v (M, 256) for transitional
    callers that need the standard attn(q, k, v) interface.
    """

    def __init__(
        self,
        return_v: bool = False,
        has_mirror: bool = True,
        positions_contiguous: bool = True,
        block_num: int = AIC_NUM,
    ):
        self._return_v = bool(return_v)
        self._has_mirror = bool(has_mirror)
        self._positions_contiguous = bool(positions_contiguous)
        self.block_num = int(block_num)

    @host
    def run(
        self,
        gm_hidden,
        gm_weight,
        gm_gamma,
        gm_positions,
        gm_cos_sin,
        gm_slot,
        gm_q,
        gm_k,
        gm_v,
        gm_mk,
        gm_mv,
        gm_k_cache,
        gm_v_cache,
        eps: Constexpr[float],
        block_num: Constexpr[int],
        return_v: Constexpr[bool],
        has_mirror: Constexpr[bool],
        positions_contiguous: Constexpr[bool],
    ):
        m_rows = gm_hidden.shape[0]
        m_pairs = (m_rows + PAIR_M - 1) // PAIR_M
        launch = m_pairs * (gm_weight.shape[0] // BASE_N)
        block_dim = min(launch, block_num)
        op = FusedQkvProjNormRopeCacheKernel(return_v, has_mirror, positions_contiguous)
        op[block_dim](
            gm_hidden,
            gm_weight,
            gm_gamma,
            gm_positions,
            gm_cos_sin,
            gm_slot,
            gm_q,
            gm_k,
            gm_v,
            gm_mk,
            gm_mv,
            gm_k_cache,
            gm_v_cache,
            eps,
        )


def _is_stream_capturing(device):
    """True while the current NPU stream is being captured into an ACL Graph."""
    query = getattr(torch.npu, "is_current_stream_capturing", None)
    if query is None:
        return False
    try:
        with torch.npu.device(device):
            return bool(query())
    except (AttributeError, RuntimeError):
        return False


def fused_qkv_proj_norm_rope_cache(
    hidden,
    weight,
    k_gamma,
    positions,
    cos_sin_cache,
    slot_mapping,
    k_cache,
    v_cache,
    eps=1e-6,
    return_v=False,
    validate=True,
    positions_contiguous=None,
):
    """WeLMv4 QKV 投影 + 后处理全融合前向（对外公开入口）。

    Computes (per TP rank, kv-mirror layer):
        qkv(M,2560) = hidden(M,2048) @ weight(2560,2048)^T   (bf16 MMAD, fp32 acc)
        q   = rope(qkv[:, :1536])                              -> (M,1536) bf16
        k   = rope(rmsnorm(qkv[:, 1536:1792]))                 -> (M,256)  bf16
        v   = bf16(qkv[:, 1792:2048])                          -> v_cache[slot] only
        mk  = bf16(qkv[:, 2048:2304])                          -> (M,256)  bf16
        mv  = bf16(qkv[:, 2304:2560])                          -> (M,256)  bf16
        k_cache[slot] = k, v_cache[slot] = v                   (slot == -1 skipped)

    Args:
        hidden: (M, 2048) bf16 contiguous (QKV projection input)
        weight: (2560, 2048) bf16 contiguous, nn.Linear (N, K) layout
        k_gamma: (256,) bf16, K RMSNorm weight
        positions: (M,) int64, contiguous positions (validated when validate=True)
        cos_sin_cache: (max_pos, 64) fp32, first 32 cos / last 32 sin per row
        slot_mapping: (M,) int64 cache row indices, -1 skips the scatter
        k_cache / v_cache: paged KV cache, last dim 256, flattenable to (slots, 256)
        return_v: also materialize v (M, 256) for callers needing attn(q, k, v);
            the profiled NPU sink-prefill attention reads K/V from the cache,
            so the default (False) matches the production dataflow.
        validate: host-side checks (positions contiguity/range, slot range,
            cache device/shape consistency). Each device-value check is a
            small reduction + stream sync; disable on hot paths where the
            caller guarantees the invariants. Device-value checks are
            skipped automatically while the current stream is being captured
            into an ACL Graph (a sync there is illegal), so graph callers
            must keep positions/slot_mapping valid for both capture and
            replay.

    Returns:
        weight (2560, 2048) — kv-mirror source layer:
            return_v=False: (q, k, mirror_k, mirror_v) — v goes to cache only.
            return_v=True:  (q, k, v, mirror_k, mirror_v).
        weight (2048, 2048) — plain QKV layer (no mirror segments):
            return_v=False: (q, k).
            return_v=True:  (q, k, v).

    Note (scope): bf16 weights only. The production modelslim-mxfp8
    ``_kv_mirror_mxfp8_source_projection`` path (raw (N,K) weight + e8m0
    scales) is NOT covered by this operator.
    """
    assert hidden.dim() == 2 and hidden.shape[1] == HIDDEN
    assert hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
    assert weight.dim() == 2 and weight.shape[1] == HIDDEN
    # Two profiled widths: 2560 = kv-mirror source layer (5 head-units),
    # 2048 = plain QKV layer (q/k/v only, welmv4.py:1845-1847 branch).
    n_units = weight.shape[0] // HEAD_DIM
    assert n_units * HEAD_DIM == weight.shape[0] and n_units in (8, 10), (
        f"weight must be (2048, {HIDDEN}) or ({QKV_WIDTH}, {HIDDEN}), "
        f"got {tuple(weight.shape)}"
    )
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    assert k_gamma.dim() == 1 and k_gamma.shape[0] == HEAD_DIM
    assert k_gamma.dtype == torch.bfloat16 and k_gamma.is_contiguous()
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == ROPE_DIM
    assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.is_contiguous()

    num_tokens = hidden.shape[0]
    assert num_tokens > 0
    assert positions.shape[0] == num_tokens
    assert slot_mapping.shape[0] == num_tokens
    assert positions.device == hidden.device and slot_mapping.device == hidden.device

    if positions.dtype == torch.int32:
        positions = positions.to(torch.int64)
    assert positions.dtype == torch.int64
    if slot_mapping.dtype == torch.int32:
        slot_mapping = slot_mapping.to(torch.int64)
    assert slot_mapping.dtype == torch.int64

    def _flatten_cache(cache, name):
        assert cache.dtype == torch.bfloat16
        assert cache.shape[-1] == HEAD_DIM
        assert cache.is_contiguous()
        if cache.dim() > 2:
            cache = cache.view(-1, HEAD_DIM)
        assert cache.dim() == 2, f"{name} must be flattenable to (slots, 256)"
        return cache

    k_cache = _flatten_cache(k_cache, "k_cache")
    v_cache = _flatten_cache(v_cache, "v_cache")
    assert k_cache.device == hidden.device, "k_cache must live on hidden's device"
    assert v_cache.device == hidden.device, "v_cache must live on hidden's device"
    assert k_cache.shape == v_cache.shape, "k_cache/v_cache shape mismatch"

    # Every check below reads a device value back to the host, which
    # synchronizes the stream and is illegal inside ACL Graph capture.
    capturing = _is_stream_capturing(hidden.device)

    # Position mode: None auto-detects (one device reduction; decode
    # callers pass it explicitly to skip the sync). An explicit
    # positions_contiguous=True with validate=True is asserted, matching
    # the historical prefill-only contract. During capture, auto-detection
    # falls back to the decode path so the reduction is never emitted.
    if positions_contiguous is None:
        if capturing:
            positions_contiguous = False
        else:
            positions_contiguous = num_tokens == 1 or bool(
                (positions[1:] - positions[:-1] == 1).all()
            )
    elif validate and not capturing and positions_contiguous and num_tokens > 1:
        assert bool((positions[1:] - positions[:-1] == 1).all()), (
            "positions must be contiguous"
        )
    if validate and not capturing:
        pos_min, pos_max = torch.aminmax(positions)
        assert int(pos_min) >= 0, "negative position"
        assert int(pos_max) < cos_sin_cache.shape[0], (
            "position beyond cos_sin_cache rows"
        )
        slot_min, slot_max = torch.aminmax(slot_mapping)
        assert int(slot_min) >= -1, "slot must be -1 or non-negative"
        if int(slot_max) >= 0:
            assert int(slot_max) < k_cache.shape[0], "slot beyond cache rows"
            valid_slots = slot_mapping[slot_mapping != -1]
            assert valid_slots.numel() == valid_slots.unique().numel(), (
                "duplicate non-negative slots are not supported"
            )

    device = hidden.device
    q = torch.empty(
        (num_tokens, NUM_Q_HEADS * HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    k = torch.empty((num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device)
    if return_v:
        v = torch.empty((num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device)
        # real destination; kernel writes it in the return_v branch
        gm_v = v
    else:
        # shape-compatible stand-in; the v-write branch is folded away at
        # trace time so this tensor is never written
        gm_v = k
    if n_units == 10:
        mirror_k = torch.empty(
            (num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        mirror_v = torch.empty(
            (num_tokens, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    else:
        # Plain 8-unit layer has no mirror segments; the kernel never
        # touches gm_mk/gm_mv for u < 8, so 1-row stand-ins suffice.
        mirror_k = torch.zeros(1, HEAD_DIM, dtype=torch.bfloat16, device=device)
        mirror_v = mirror_k

    _, launch_groups, _ = _core_budget(hidden.device)
    m_pairs = (num_tokens + PAIR_M - 1) // PAIR_M
    block_num = min(m_pairs * n_units, launch_groups)
    op = FusedQkvProjNormRopeCache(
        return_v=return_v,
        has_mirror=(n_units == 10),
        positions_contiguous=positions_contiguous,
        block_num=block_num,
    )
    tensors = (
        hidden,
        weight,
        k_gamma.view(1, HEAD_DIM),
        positions,
        cos_sin_cache,
        slot_mapping,
        q,
        k,
        gm_v,
        mirror_k,
        mirror_v,
        k_cache,
        v_cache,
    )
    _compiled_run(op, tensors, eps)(*tensors)
    if n_units == 10:
        if return_v:
            return q, k, v, mirror_k, mirror_v
        return q, k, mirror_k, mirror_v
    if return_v:
        return q, k, v
    return q, k
