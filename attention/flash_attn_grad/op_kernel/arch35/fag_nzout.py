# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-S-tile FP32 NZ workspace and fused NZ-to-ND output conversion.

Each tile owns 128 physical rows and dAlign columns (the kernel writes whole
[128, dAlign] tiles; the route pins dAlign=128, and the host reserves the same
width). GM FP32 NZ uses
C0=8. TND batch gaps bound storage by N * (ceil(T / 128) + B - 1) tiles;
effective sequence lengths never change these physical addresses.
"""

import pypto_pro.language as pl

from fag_mem import POST_F16_ADDRS, POST_F32_ADDRS, MTX_UB_POST_F16, MTX_UB_POST_F32


def nz_workspace_tiles(total_tokens, batch, max_seq):
    if layout == 1:
        return (total_tokens + 127) // 128 + pl.max(batch - 1, 0)
    return batch * ((max_seq + 127) // 128)


def store_nz_partial(tensor_ws, acc, tile_index, cols):
    # A local typed DDR view supplies the NZ format; make_tensor alone is ND.
    ptr = pl.addptr(pl.make_ptr(tensor_ws, dtype=pl.DT_FP32), tile_index * 128 * cols)
    out_nz: pl.Tensor[
        [128, cols], pl.DT_FP32, pl.NZ, pl.MemRef(pl.MemorySpace.DDR, 0, 65536)
    ] = pl.make_tensor(ptr, [128, cols], [cols, 1], dtype=pl.DT_FP32)
    # Tail padding is private to this S tile and is discarded by the post stage.
    pl.set_validshape(acc, [128, cols])
    pl.store(out_nz, acc, [0, 0], atomic=pl.AtomicType.AtomicAdd)


def make_nz_post_f32_buffer():
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[1, 16384],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=POST_F32_ADDRS[d_align],
        mutex_ids=MTX_UB_POST_F32,
    )


def make_nz_post_f16_buffer():
    return pl.make_tile_group(
        type=pl.TileType(
            shape=[128, 128],
            dtype=input_dtype,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=0,
        ),
        addrs=POST_F16_ADDRS[d_align],
        mutex_ids=MTX_UB_POST_F16,
    )


@pl.vector_function
def nz_muls_cast_vf(dst, src, scale: pl.DT_FP32, rows: pl.DT_INT64, cols: pl.DT_INT64):
    """Gather eight 32-byte NZ blocks per register, then scale and cast a row."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    first = vf.update_mask(pl.min(cols, 64), dtype=pl.DT_FP32)
    last = vf.update_mask(pl.max(cols - 64, 0), dtype=pl.DT_FP32)
    # DATA_BLOCK_COPY stride is the physical distance between C0=8 fragments.
    for row in pl.range(0, rows):
        lo = vf.load_align(
            src + row * 8,
            first,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=128,
        )
        hi = vf.load_align(
            src + 8192 + row * 8,
            last,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=128,
        )
        lo_scaled = vf.muls(lo, scale, first)
        hi_scaled = vf.muls(hi, scale, last)
        lo_half = vf.astype(
            lo_scaled, first, dtype=input_dtype, layout=pl.CastLayout.ZERO
        )
        hi_half = vf.astype(
            hi_scaled, last, dtype=input_dtype, layout=pl.CastLayout.ZERO
        )
        vf.store_align(dst + row * 128, lo_half, full, dist=pl.StoreDist.PACK_B32)
        vf.store_align(dst + row * 128 + 64, hi_half, full, dist=pl.StoreDist.PACK_B32)


@pl.vector_function
def nz_zero_tail_vf(dst, valid_rows: pl.DT_INT64, rows: pl.DT_INT64):
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    zero = vf.full(0.0, full, dtype=input_dtype)
    for row in pl.range(valid_rows, rows):
        vf.store_align(dst + row * 128, zero, full)


def process_nz_post(
    ci,
    cu,
    used,
    has_used,
    tensor_ws,
    tensor_out,
    f32_group,
    f16_group,
    heads,
    max_seq,
    coef_b,
    scale,
):
    """Assign whole physical S tiles to AIVs, then scatter valid rows in output layout."""
    aiv = pl.get_block_idx()
    for batch in pl.range(0, ci.bSize):
        token_base = 0
        seq = max_seq
        batch_base = batch * heads * ((max_seq + 127) // 128)
        if layout == 1:
            token_base = pl.astype(pl.getval(cu, batch), pl.DT_INT64)
            seq = pl.astype(pl.getval(cu, batch + 1), pl.DT_INT64) - token_base
            batch_base = (token_base // 128 + batch) * heads
        used_seq = seq
        if has_used == 1:
            used_seq = pl.astype(pl.getval(used, batch), pl.DT_INT64)
        outer = (seq + 127) // 128
        for item in pl.range(aiv, heads * outer, ci.aivCount):
            head = item // outer
            block = item % outer
            rows = pl.min(128, seq - block * 128)
            valid_rows = pl.min(rows, pl.max(used_seq - block * 128, 0))
            ft = f32_group.next()
            ht = f16_group.next()
            pl.set_validshape(ft, [1, 128 * ci.dAlign])
            pl.load(ft, tensor_ws, [0, (batch_base + item) * 128 * ci.dAlign])
            nz_muls_cast_vf(ht, ft, scale, valid_rows, ci.dSize)
            # Expanded L0C stores may overwrite padding with unspecified values.
            # seqused tails remain part of the public tensor and must be zero.
            if valid_rows < rows:
                nz_zero_tail_vf(ht, valid_rows, rows)
            pl.set_validshape(ht, [rows, ci.dSize])
            i0 = batch * coef_b + head * ci.coefN0
            i2 = head * ci.coefN2
            if layout == 1:
                i0 = 0
                i2 = head
            pl.store(
                tensor_out, ht, [i0, token_base + block * 128, i2, 0], order=[1, 3]
            )
