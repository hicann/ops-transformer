#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side tiling checks for MinimaxSparseAttentionSplitKv.

Mirrors op_host/minimax_sparse_attention_split_kv_tiling.cpp:
  CheckTilingConstraints / CalculateWorkSpace / FillTilingData tilingKey.
No NPU required.
"""

from __future__ import annotations

import unittest

L0_TILE_M = 128
L0_TILE_N = 128
HEAD_SIZE = 128
MAX_BATCH_GROUPS = 8
UB_S_STAGES = 2
MAX_UB_S_ELEM = 16384
SM_ROW_MAX_ELEM = 64
UB_BLOCK = 32768
HIGH_PREC_S_BYTES = UB_S_STAGES * MAX_UB_S_ELEM * 4
HIGH_PREC_P_BYTES = UB_S_STAGES * MAX_UB_S_ELEM * 2
HIGH_PREC_TMP_BYTES = UB_BLOCK
HIGH_PREC_TMP_FLOATS = HIGH_PREC_TMP_BYTES // 4
UB_SIZE_950 = 256 * 1024

TILING_KEY_MIXED = 20001
TILING_KEY_INNER_LOW = 20002
TILING_KEY_INNER_HIGH = 20003


def ceil_div(n: int, d: int) -> int:
    if n == 0:
        return 0
    return (n + d - 1) // d if d else n


def align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align if align else value


def tiling_key(inner_precise: int) -> int:
    if inner_precise == 1:
        return TILING_KEY_INNER_LOW
    if inner_precise == 0:
        return TILING_KEY_INNER_HIGH
    return TILING_KEY_MIXED


def workspace_bytes(
    total_q: int,
    kv_heads: int,
    top_k: int,
    group_size: int,
    d: int,
    inner_precise: int,
    libapi: int = 0,
) -> int:
    slot_o = group_size * d
    slot_stat = group_size
    task_slots = total_q * kv_heads * top_k
    accum_out = task_slots * slot_o
    lse_stat = task_slots * slot_stat
    o_width = 2 if inner_precise == 1 else 4
    return libapi + accum_out * o_width + lse_stat * 2 * 4


def check_tiling(
    d: int,
    block_size: int,
    num_heads: int,
    kv_heads: int,
    inner_precise: int,
    ub_size: int = UB_SIZE_950,
) -> tuple[bool, str]:
    if inner_precise not in (0, 1, 4):
        return False, "innerPrecise"
    if d != HEAD_SIZE:
        return False, "D"
    if block_size == 0 or block_size > L0_TILE_N:
        return False, "blockSize"
    if kv_heads == 0 or num_heads % kv_heads != 0:
        return False, "kvHeads"
    group_size = num_heads // kv_heads
    if group_size == 0 or group_size > L0_TILE_M:
        return False, "groupSize"

    batch_groups_max = L0_TILE_M // group_size
    if batch_groups_max == 0 or batch_groups_max > MAX_BATCH_GROUPS:
        batch_groups_max = MAX_BATCH_GROUPS
    batch_m = min(batch_groups_max * group_size, L0_TILE_M)
    m_per_aiv = ceil_div(batch_groups_max, 2) * group_size
    m_per_aiv = min(m_per_aiv, batch_m)
    m_align = align_up(m_per_aiv, 16)
    n_align = align_up(block_size, 16)
    s_elem = m_align * n_align
    grp_stride = align_up(group_size, 8)
    stats_elem = ceil_div(batch_groups_max, 2) * grp_stride
    if stats_elem > SM_ROW_MAX_ELEM:
        return False, "stats"

    if inner_precise != 0:
        return True, "ok"

    ub_need = (
        HIGH_PREC_S_BYTES
        + HIGH_PREC_P_BYTES
        + HIGH_PREC_TMP_BYTES
        + 2 * SM_ROW_MAX_ELEM * 4 * UB_S_STAGES * 2
    )
    if ub_size > 0 and ub_need > ub_size:
        return False, "ubSize"
    # AIV0-only: shrink groups until the FULL tile fits tmp and stats.
    g_fit = batch_groups_max
    while g_fit > 0:
        bm = min(g_fit * group_size, L0_TILE_M)
        se = align_up(bm, 16) * n_align
        st = g_fit * grp_stride
        if se <= HIGH_PREC_TMP_FLOATS and se <= MAX_UB_S_ELEM and st <= SM_ROW_MAX_ELEM:
            break
        g_fit -= 1
    if g_fit == 0:
        return False, "tmp"
    return True, "ok"


class InnerPreciseTilingTest(unittest.TestCase):
    def test_tiling_key_dispatch(self):
        self.assertEqual(tiling_key(0), TILING_KEY_INNER_HIGH)
        self.assertEqual(tiling_key(1), TILING_KEY_INNER_LOW)
        self.assertEqual(tiling_key(4), TILING_KEY_MIXED)
        self.assertEqual(tiling_key(99), TILING_KEY_MIXED)

    def test_workspace_0_and_4_same_fp32_o(self):
        kwargs = dict(total_q=32, kv_heads=4, top_k=8, group_size=16, d=128)
        ws0 = workspace_bytes(**kwargs, inner_precise=0)
        ws4 = workspace_bytes(**kwargs, inner_precise=4)
        ws1 = workspace_bytes(**kwargs, inner_precise=1)
        self.assertEqual(ws0, ws4)
        self.assertLess(ws1, ws0)
        self.assertEqual(ws0 - ws1, 32 * 4 * 8 * 16 * 128 * 2)  # O_partial 4B vs 2B

    def test_production_group16_all_precise_ok(self):
        for ip in (0, 1, 4):
            ok, reason = check_tiling(128, 128, 64, 4, ip)
            self.assertTrue(ok, f"innerPrecise={ip} failed: {reason}")

    def test_high_prec_group32_and_64_ok(self):
        # AIV0-only shrinks to M=64: 64x128=8192 tmp, stats=64
        ok, reason = check_tiling(128, 128, 32, 1, 0)
        self.assertTrue(ok, reason)
        ok, reason = check_tiling(128, 128, 64, 1, 0)
        self.assertTrue(ok, reason)

    def test_high_prec_group128_rejected_by_tmp_and_stats(self):
        ok, reason = check_tiling(128, 128, 128, 1, 0)
        self.assertFalse(ok)
        self.assertEqual(reason, "stats")  # stats checked first (128 > 64)

    def test_high_prec_group128_rejected_by_tmp_and_stats(self):
        ok, reason = check_tiling(128, 128, 128, 1, 0)
        self.assertFalse(ok)
        self.assertEqual(reason, "stats")  # stats checked first (128 > 64)

    def test_mixed_group128_also_rejected_by_stats(self):
        ok, reason = check_tiling(128, 128, 128, 1, 4)
        self.assertFalse(ok)
        self.assertEqual(reason, "stats")

    def test_invalid_inner_precise(self):
        ok, reason = check_tiling(128, 128, 64, 4, 2)
        self.assertFalse(ok)
        self.assertEqual(reason, "innerPrecise")

    def test_d_and_block_size(self):
        self.assertEqual(check_tiling(64, 128, 64, 4, 0)[1], "D")
        self.assertEqual(check_tiling(128, 0, 64, 4, 0)[1], "blockSize")
        self.assertEqual(check_tiling(128, 256, 64, 4, 0)[1], "blockSize")

    def test_high_prec_ub_too_small(self):
        ok, reason = check_tiling(128, 128, 64, 4, 0, ub_size=128 * 1024)
        self.assertFalse(ok)
        self.assertEqual(reason, "ubSize")

    def test_high_prec_tmp_bound_is_8192(self):
        # AIV0-only: production groupSize=16 shrinks to 4 groups, 64x128=8192.
        ok, _ = check_tiling(128, 128, 64, 4, 0)
        self.assertTrue(ok)
        self.assertEqual(HIGH_PREC_TMP_FLOATS, 8192)
        self.assertEqual(HIGH_PREC_S_BYTES, 128 * 1024)
        self.assertEqual(HIGH_PREC_P_BYTES, 64 * 1024)

    def test_block_size_64_high_prec_ok(self):
        ok, reason = check_tiling(128, 64, 64, 4, 0)
        self.assertTrue(ok, reason)

    def test_ceil_div_zero_seqlen_is_zero_packed_rows(self):
        self.assertEqual(ceil_div(0, 128), 0)
        self.assertEqual(ceil_div(0, 32), 0)
        kv_lens = [0, 32, 0, 64]
        self.assertEqual(sum(ceil_div(n, 32) for n in kv_lens), 3)

    def test_packed_row_skips_zero_kv_len(self):
        # Mirrors InitPackedRowCoord / AdvancePackedRowCoord with padding requests.
        block_size = 32
        kv_lens = [0, 32, 0, 64]  # rows: 0, 1, 0, 2

        def rows(b):
            return ceil_div(kv_lens[b], block_size)

        batch = len(kv_lens)

        def skip_empty(batch_idx, kv_block_idx):
            while batch_idx < batch and kv_block_idx >= rows(batch_idx):
                batch_idx += 1
            return batch_idx

        def init():
            return skip_empty(0, 0), 0

        def advance(batch_idx, kv_block_idx):
            batch_idx += 1
            batch_idx = skip_empty(batch_idx, kv_block_idx)
            if batch_idx >= batch:
                kv_block_idx += 1
                batch_idx = skip_empty(0, kv_block_idx)
            return batch_idx, kv_block_idx

        coords = []
        b, blk = init()
        total = sum(rows(i) for i in range(batch))
        for _ in range(total):
            coords.append((b, blk))
            b, blk = advance(b, blk)
        self.assertEqual(coords, [(1, 0), (3, 0), (3, 1)])
        # Leading dummy: packedRow 0 is the first real request, not batch 0.
        self.assertEqual(coords[0], (1, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
