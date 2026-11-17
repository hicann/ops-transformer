# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-selected GQA BAND phases and overlap scheduling.

The common loader is the only place that knows the flat tiling ABI. Phase
geometry, round permutations and physical translation remain separate helpers.
"""

from schedule.deterministic.fag_deter_formulas import (
    pl,
    _unpack_wxy,
    cal_gqa_dense,
    cal_gqa_dense_no_tune,
    cal_gqa_causal,
)


def load_band_phase(ci, index, use_tail):
    kind = pl.astype(0, pl.DT_INT64)
    m = pl.astype(0, pl.DT_INT64)
    n = pl.astype(0, pl.DT_INT64)
    row = pl.astype(0, pl.DT_INT64)
    col = pl.astype(0, pl.DT_INT64)
    full = pl.astype(0, pl.DT_INT64)
    tail = pl.astype(0, pl.DT_INT64)
    if use_tail == 0:
        if index == 0:
            kind = pl.astype(ci.deterBandPhase0Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandPhase0M, pl.DT_INT64)
            n = pl.astype(ci.deterBandPhase0N, pl.DT_INT64)
            row = pl.astype(ci.deterBandPhase0RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandPhase0ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandPhase0FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandPhase0TailRounds, pl.DT_INT64)
        elif index == 1:
            kind = pl.astype(ci.deterBandPhase1Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandPhase1M, pl.DT_INT64)
            n = pl.astype(ci.deterBandPhase1N, pl.DT_INT64)
            row = pl.astype(ci.deterBandPhase1RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandPhase1ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandPhase1FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandPhase1TailRounds, pl.DT_INT64)
        elif index == 2:
            kind = pl.astype(ci.deterBandPhase2Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandPhase2M, pl.DT_INT64)
            n = pl.astype(ci.deterBandPhase2N, pl.DT_INT64)
            row = pl.astype(ci.deterBandPhase2RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandPhase2ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandPhase2FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandPhase2TailRounds, pl.DT_INT64)
        elif index == 3:
            kind = pl.astype(ci.deterBandPhase3Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandPhase3M, pl.DT_INT64)
            n = pl.astype(ci.deterBandPhase3N, pl.DT_INT64)
            row = pl.astype(ci.deterBandPhase3RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandPhase3ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandPhase3FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandPhase3TailRounds, pl.DT_INT64)
        elif index == 4:
            kind = pl.astype(ci.deterBandPhase4Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandPhase4M, pl.DT_INT64)
            n = pl.astype(ci.deterBandPhase4N, pl.DT_INT64)
            row = pl.astype(ci.deterBandPhase4RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandPhase4ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandPhase4FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandPhase4TailRounds, pl.DT_INT64)
    else:
        if index == 0:
            kind = pl.astype(ci.deterBandTailPhase0Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandTailPhase0M, pl.DT_INT64)
            n = pl.astype(ci.deterBandTailPhase0N, pl.DT_INT64)
            row = pl.astype(ci.deterBandTailPhase0RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandTailPhase0ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandTailPhase0FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandTailPhase0TailRounds, pl.DT_INT64)
        elif index == 1:
            kind = pl.astype(ci.deterBandTailPhase1Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandTailPhase1M, pl.DT_INT64)
            n = pl.astype(ci.deterBandTailPhase1N, pl.DT_INT64)
            row = pl.astype(ci.deterBandTailPhase1RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandTailPhase1ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandTailPhase1FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandTailPhase1TailRounds, pl.DT_INT64)
        elif index == 2:
            kind = pl.astype(ci.deterBandTailPhase2Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandTailPhase2M, pl.DT_INT64)
            n = pl.astype(ci.deterBandTailPhase2N, pl.DT_INT64)
            row = pl.astype(ci.deterBandTailPhase2RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandTailPhase2ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandTailPhase2FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandTailPhase2TailRounds, pl.DT_INT64)
        elif index == 3:
            kind = pl.astype(ci.deterBandTailPhase3Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandTailPhase3M, pl.DT_INT64)
            n = pl.astype(ci.deterBandTailPhase3N, pl.DT_INT64)
            row = pl.astype(ci.deterBandTailPhase3RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandTailPhase3ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandTailPhase3FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandTailPhase3TailRounds, pl.DT_INT64)
        elif index == 4:
            kind = pl.astype(ci.deterBandTailPhase4Kind, pl.DT_INT64)
            m = pl.astype(ci.deterBandTailPhase4M, pl.DT_INT64)
            n = pl.astype(ci.deterBandTailPhase4N, pl.DT_INT64)
            row = pl.astype(ci.deterBandTailPhase4RowOffset, pl.DT_INT64)
            col = pl.astype(ci.deterBandTailPhase4ColOffset, pl.DT_INT64)
            full = pl.astype(ci.deterBandTailPhase4FullRounds, pl.DT_INT64)
            tail = pl.astype(ci.deterBandTailPhase4TailRounds, pl.DT_INT64)
    return kind, m, n, row, col, full, tail


def band_head_inner_round(r, span, g, shift):
    out = pl.astype(r, pl.DT_INT64)
    if r > 0 and span > 0 and g > 0:
        period = span * g
        local = (r - 1) % period
        out = pl.astype(
            (r - 1) // period * period
            + (local % g) * span
            + (local // g + shift) % span
            + 1,
            pl.DT_INT64,
        )
    return out


def band_triangle_source_round(k, size, b, g, r):
    out = pl.astype(r, pl.DT_INT64)
    if k > 0 and size > 0 and b > 0 and g > 1 and r > 0:
        if b >= k or size >= k:
            triangle = size * (size + 1) // 2
            full = (b // k) * triangle * g
            if r <= full:
                out = band_head_inner_round(r, triangle, g, 0)
            else:
                tail = b % k
                t = size // k
                height = 2 * size - t * k + 1
                paired = height * t * (tail // 2) * g
                local = r - full
                if local <= paired:
                    shift = pl.astype(0, pl.DT_INT64)
                    if t == 1:
                        shift = pl.astype(size - k + 1, pl.DT_INT64)
                    out = full + band_head_inner_round(local, height, g, shift)
                else:
                    local = local - paired
                    odd = pl.astype(0, pl.DT_INT64)
                    if tail % 2 != 0:
                        if t % 2 != 0:
                            t1 = size // (2 * k)
                            m1 = size - t1 * 2 * k
                            odd = pl.astype((size + m1 + 1) * t1, pl.DT_INT64)
                            if size % k == 0:
                                odd = odd + m1
                            else:
                                odd = odd + pl.max(m1, 2 * m1 - 2 * k + 1)
                        else:
                            odd = pl.astype(height * t // 2, pl.DT_INT64)
                    if local <= odd * g:
                        out = full + paired + band_head_inner_round(local, odd, g, 0)
    return out


def band_edge_split(kind, m, n, row, col):
    split = pl.astype(-1, pl.DT_INT64)
    if kind == 4:
        split = pl.astype(n - 1, pl.DT_INT64)
    elif kind == 5:
        split = pl.astype(col - row - n, pl.DT_INT64)
    elif kind == 6:
        split = pl.astype(m + row - col - 1, pl.DT_INT64)
    return split


def band_edge_source_round(kind, m, n, row, col, k, groups, g, r):
    out = pl.astype(r, pl.DT_INT64)
    split = band_edge_split(kind, m, n, row, col)
    if (
        kind >= 4
        and kind <= 6
        and k > 0
        and groups > 0
        and g > 1
        and m > 0
        and n > 0
        and n <= m
    ):
        if (
            groups <= k // n
            and split >= 0
            and split < m - n
            and r > 0
            and (r - 1) // m < g
        ):
            out = band_head_inner_round(r, m, g, split + 1)
    return out


def cal_band_phase(k, g, groups, kind, m, n, row, col, rounds, r, j):
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    local_groups = pl.astype(groups, pl.DT_INT64)
    group_offset = pl.astype(0, pl.DT_INT64)
    triangle = kind == 1 or kind == 2
    edge = kind >= 4 and kind <= 6
    if triangle:
        source_round = band_triangle_source_round(k, m, groups, g, r)
        packed = cal_gqa_causal(k, m, n, groups, j, source_round, g, 1)
        w, x, y = _unpack_wxy(packed, m, n)
    elif kind == 8:
        column = (r - 1) // g * k + j
        if column <= groups * m:
            w = pl.astype((column - 1) // m * g + (r - 1) % g + 1, pl.DT_INT64)
            x = pl.astype((column - 1) % m + 1, pl.DT_INT64)
            y = x
    elif kind == 3:
        w, x, y = cal_gqa_dense_no_tune(k, m, n, groups, j, r, g)
    else:
        source_round = pl.astype(r, pl.DT_INT64)
        if edge:
            source_round = band_edge_source_round(kind, m, n, row, col, k, groups, g, r)
        elif kind == 7:
            rounds_per_group = rounds // groups
            group_offset = pl.astype((r - 1) // rounds_per_group, pl.DT_INT64)
            source_round = pl.astype((r - 1) % rounds_per_group + 1, pl.DT_INT64)
            local_groups = pl.astype(1, pl.DT_INT64)
        w, x, y = cal_gqa_dense(k, m, n, local_groups, j, source_round, g)
    if (
        w < 1
        or w > local_groups * g
        or x < 1
        or x > m
        or y < 1
        or y > n
        or (triangle and x < y)
    ):
        w = pl.astype(-1, pl.DT_INT64)
    else:
        if kind == 2:
            x = m + 1 - x
            y = n + 1 - y
        elif kind == 3:
            x = x + y - 1
        apply_offset = pl.astype(1, pl.DT_INT64)
        if edge:
            if x <= band_edge_split(kind, m, n, row, col) + y:
                apply_offset = pl.astype(0, pl.DT_INT64)
        x = x + row * apply_offset
        y = y + col * apply_offset
        w = w + group_offset * g
    return w, x, y


def cal_band_piecewise_plan(ci, j, r):
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    k = ci.deterBandK
    g = ci.deterBandG
    h = ci.deterBandGroupsPerChunk
    if (
        j >= 1
        and j <= k
        and r >= 1
        and r <= ci.deterBandTotalRounds
        and h > 0
        and ci.deterBandFullRounds > 0
    ):
        full_chunks = ci.deterBandB // h
        full_end = full_chunks * ci.deterBandFullRounds
        chunk = pl.astype((r - 1) // ci.deterBandFullRounds, pl.DT_INT64)
        groups = pl.astype(h, pl.DT_INT64)
        local_round = pl.astype((r - 1) % ci.deterBandFullRounds + 1, pl.DT_INT64)
        is_tail = pl.astype(0, pl.DT_INT64)
        use_tail = pl.astype(0, pl.DT_INT64)
        count = pl.astype(ci.deterBandPhaseCount, pl.DT_INT64)
        if r > full_end:
            is_tail = pl.astype(1, pl.DT_INT64)
            chunk = pl.astype(full_chunks, pl.DT_INT64)
            groups = pl.astype(ci.deterBandB % h, pl.DT_INT64)
            local_round = pl.astype(r - full_end, pl.DT_INT64)
            if ci.deterBandTailPhaseCount > 0:
                use_tail = pl.astype(1, pl.DT_INT64)
                count = pl.astype(ci.deterBandTailPhaseCount, pl.DT_INT64)
        index = pl.astype(0, pl.DT_INT64)
        found = pl.astype(0, pl.DT_INT64)
        while index < count and found == 0:
            kind, m, n, row, col, full, tail = load_band_phase(ci, index, use_tail)
            phase_rounds = pl.astype(full, pl.DT_INT64)
            if is_tail == 1:
                phase_rounds = pl.astype(tail, pl.DT_INT64)
            if local_round > phase_rounds:
                local_round = local_round - phase_rounds
            else:
                w, x, y = cal_band_phase(
                    k, g, groups, kind, m, n, row, col, phase_rounds, local_round, j
                )
                if w > 0:
                    w = w + chunk * h * g
                found = pl.astype(1, pl.DT_INT64)
            index = index + 1
    return w, x, y


def band_overlap_core(k, groups, m, n, core):
    group = pl.astype(-1, pl.DT_INT64)
    local = pl.astype(0, pl.DT_INT64)
    budget = pl.astype(0, pl.DT_INT64)
    extra = k % groups
    useful = m // 2 + pl.min(m, n)
    base = k // groups
    high = pl.min(base + 1, useful)
    low = pl.min(base, useful)
    high_cores = extra * high
    if core <= high_cores + (groups - extra) * low:
        if core <= high_cores:
            group = pl.astype((core - 1) // high, pl.DT_INT64)
            local = pl.astype((core - 1) % high + 1, pl.DT_INT64)
            budget = pl.astype(high, pl.DT_INT64)
        else:
            group = pl.astype(extra + (core - high_cores - 1) // low, pl.DT_INT64)
            local = pl.astype((core - high_cores - 1) % low + 1, pl.DT_INT64)
            budget = pl.astype(low, pl.DT_INT64)
    return group, local, budget


def cal_band_overlap_plan(ci, j, r):
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    h = ci.deterBandGroupsPerChunk
    k = ci.deterBandK
    g = ci.deterBandG
    if (
        h > 0
        and h <= ci.deterBandB
        and ci.deterBandFullRounds > 0
        and ci.deterBandPhaseCount == 1
        and ci.deterBandTailPhaseCount >= 0
        and ci.deterBandTailPhaseCount <= 1
        and r >= 1
        and r <= ci.deterBandTotalRounds
        and j >= 1
        and j <= k
    ):
        full_chunks = ci.deterBandB // h
        full_end = full_chunks * ci.deterBandFullRounds
        chunk = pl.astype((r - 1) // ci.deterBandFullRounds, pl.DT_INT64)
        groups = pl.astype(h, pl.DT_INT64)
        local_round = pl.astype(r - chunk * ci.deterBandFullRounds, pl.DT_INT64)
        is_tail = pl.astype(0, pl.DT_INT64)
        use_tail = pl.astype(0, pl.DT_INT64)
        if r > full_end:
            is_tail = pl.astype(1, pl.DT_INT64)
            chunk = pl.astype(full_chunks, pl.DT_INT64)
            groups = pl.astype(ci.deterBandB % h, pl.DT_INT64)
            local_round = pl.astype(r - full_end, pl.DT_INT64)
            if ci.deterBandTailPhaseCount == 1:
                use_tail = pl.astype(1, pl.DT_INT64)
        kind, m, n, row, col, full, tail = load_band_phase(ci, 0, use_tail)
        phase_rounds = pl.astype(full, pl.DT_INT64)
        if is_tail == 1:
            phase_rounds = pl.astype(tail, pl.DT_INT64)
        if (
            (kind == 15 or kind == 16)
            and local_round <= phase_rounds
            and k > 0
            and groups > 0
            and g > 1
            and m >= 2
            and n > 0
        ):
            if groups <= k // (m // 2 + 1):
                group, local_core, local_k = band_overlap_core(k, groups, m, n, j)
                if group >= 0:
                    triangle_cores = m // 2
                    stripe_columns = local_k - triangle_cores
                    prefix = m * g
                    if local_round <= prefix:
                        if local_core <= triangle_cores:
                            side = m - 1
                            if local_round <= (m - 1 + m % 2) * g:
                                source_round = band_triangle_source_round(
                                    triangle_cores, side, 1, g, local_round
                                )
                                packed = cal_gqa_causal(
                                    triangle_cores,
                                    side,
                                    side,
                                    1,
                                    local_core,
                                    source_round,
                                    g,
                                    1,
                                )
                                w, x, y = _unpack_wxy(packed, side, side)
                                if w > 0:
                                    x = x + 1
                                    y = y + n
                        else:
                            lane = local_core - triangle_cores - 1
                            w = pl.astype(local_round % g + 1, pl.DT_INT64)
                            x = pl.astype(
                                ((local_round - 1) // g + lane) % m + 1, pl.DT_INT64
                            )
                            y = pl.astype(lane + 1, pl.DT_INT64)
                    elif n > stripe_columns:
                        w, x, y = cal_gqa_dense(
                            local_k,
                            m,
                            n - stripe_columns,
                            1,
                            local_core,
                            local_round - prefix,
                            g,
                        )
                        if w > 0:
                            y = y + stripe_columns
                    if w > 0:
                        if kind == 16:
                            x = m + 1 - x
                            y = n + m - y
                        x = x + row
                        y = y + col
                        w = w + (chunk * h + group) * g
    return w, x, y
