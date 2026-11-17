# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Column-owner deterministic BAND maps, specialized by compile-time family.

Geometry and chunk boundaries come from the Host plan. Helpers implement only
coordinate permutations; tensor layouts and synchronization stay in the runner.
"""

from schedule.deterministic.fag_deter_formulas import (
    pl,
    _ceil,
)


def band_clipped_sum(first, step, count, limit):
    out = pl.astype(0, pl.DT_INT64)
    if count > 0:
        last = first + (count - 1) * step
        if last > 0:
            if first >= limit:
                out = pl.astype(count * limit, pl.DT_INT64)
            elif first >= 0 and last <= limit:
                out = pl.astype(count * (first + last) // 2, pl.DT_INT64)
            else:
                positive = pl.max(_ceil(1 - first, step), 0)
                saturated = pl.max(_ceil(limit - first, step), 0)
                begin = pl.min(positive, count)
                end = pl.min(saturated, count)
                middle = end - begin
                out = pl.astype(
                    middle * (2 * first + (begin + end - 1) * step) // 2
                    + (count - end) * limit,
                    pl.DT_INT64,
                )
    return out


def band_owner_sequence_prefix(m, n, p, q, first, step, pairs):
    out = pl.astype(0, pl.DT_INT64)
    if first <= n and pairs > 0:
        count = pl.min(pairs, (n - first) // step + 1)
        out = band_clipped_sum(first + p - 1, step, count, m) - band_clipped_sum(
            first - q, step, count, m
        )
    return out


def band_balanced_sequence_prefix(m, n, p, q, first, step, count):
    columns = pl.min(n, m + q - 1)
    out = pl.astype(0, pl.DT_INT64)
    if first < columns and count > 0:
        count = pl.min(count, (columns - 1 - first) // step + 1)
        peak = pl.min(pl.min(m, p + q - 1), n + p - 1)
        left = pl.max(1, peak - p + 1)
        right = pl.min(columns, m + q - peak)
        plateau = right - left + 1
        paired = pl.min(left - 1, columns - right)
        diff = first - plateau
        first_level = pl.astype(diff // 2 + 1, pl.DT_INT64)
        if diff < 0:
            first_level = pl.astype(-((-diff - 1) // 2), pl.DT_INT64)
        paired_drop = band_clipped_sum(first_level, step // 2, count, paired)
        single = columns - plateau - 2 * paired
        single_drop = band_clipped_sum(
            first - plateau - 2 * paired + 1, step, count, single
        )
        out = pl.astype(count * peak - paired_drop - single_drop, pl.DT_INT64)
    return out


def band_owner_pair_prefix_family(m, n, p, q, lanes, lane, pairs, family):
    if family == 8:
        out = band_balanced_sequence_prefix(m, n, p, q, lane, 2 * lanes, pairs)
        out = out + band_balanced_sequence_prefix(
            m, n, p, q, 2 * lanes - 1 - lane, 2 * lanes, pairs
        )
    else:
        out = band_owner_sequence_prefix(m, n, p, q, lane + 1, 2 * lanes, pairs)
        out = out + band_owner_sequence_prefix(
            m, n, p, q, 2 * lanes - lane, 2 * lanes, pairs
        )
    return out


def band_owner_pair_prefix(m, n, p, q, lanes, lane, pairs):
    return band_owner_pair_prefix_family(m, n, p, q, lanes, lane, pairs, deter_family)


def band_balanced_column(m, n, p, q, rank):
    columns = pl.min(n, m + q - 1)
    peak = pl.min(pl.min(m, p + q - 1), n + p - 1)
    left = pl.max(1, peak - p + 1)
    right = pl.min(columns, m + q - peak)
    plateau = right - left + 1
    paired = pl.min(left - 1, columns - right)
    y = pl.astype(left + rank, pl.DT_INT64)
    if rank >= plateau:
        rank = rank - plateau
        if rank < 2 * paired:
            offset = rank // 2 + 1
            if rank % 2 == 0:
                y = pl.astype(left - offset, pl.DT_INT64)
            else:
                y = pl.astype(right + offset, pl.DT_INT64)
        else:
            offset = rank - paired + 1
            if left - 1 > columns - right:
                y = pl.astype(left - offset, pl.DT_INT64)
            else:
                y = pl.astype(right + offset, pl.DT_INT64)
    return y


def band_owner_column_family(m, n, p, q, rank, family):
    if family == 8:
        return band_balanced_column(m, n, p, q, rank)
    return pl.astype(rank + 1, pl.DT_INT64)


def band_owner_column(m, n, p, q, rank):
    return band_owner_column_family(m, n, p, q, rank, deter_family)


def cal_band_column_owner_family(k, b, g, m, n, p, q, core, r, family):
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if (
        k > 0
        and b > 0
        and (b <= k)
        and (g > 1)
        and (m > 0)
        and (n > 0)
        and (p > 0)
        and (p <= m)
        and (q > 0)
        and (q <= n)
        and (core >= 1)
        and (core <= k)
        and (r > 0)
    ):
        lanes = pl.min(k // b, g)
        if core <= b * lanes:
            lane = (core - 1) % lanes
            group = (core - 1) // lanes
            row_slot = (r - 1) // g
            short = pl.astype(0, pl.DT_INT64)
            if family != 8:
                if n <= 2 * lanes:
                    short = pl.astype(1, pl.DT_INT64)
            if short == 1:
                local_row = row_slot
                y = pl.astype(lane + 1, pl.DT_INT64)
                first = pl.max(1, y - q + 1)
                height = pl.max(0, pl.min(y + p - 1, m) - first + 1)
                if y > n:
                    height = pl.astype(0, pl.DT_INT64)
                valid = pl.astype(1, pl.DT_INT64)
                if local_row >= height:
                    local_row = local_row - height
                    y = pl.astype(2 * lanes - lane, pl.DT_INT64)
                    first = pl.max(1, y - q + 1)
                    if y > n or local_row >= pl.max(
                        0, pl.min(y + p - 1, m) - first + 1
                    ):
                        valid = pl.astype(0, pl.DT_INT64)
                if valid == 1:
                    x = pl.astype(first + local_row, pl.DT_INT64)
                    if p == m and row_slot < height:
                        x = pl.astype(m - local_row, pl.DT_INT64)
                    w = pl.astype(group * g + (r - 1 + lane) % g + 1, pl.DT_INT64)
            else:
                columns = n
                if family == 8:
                    columns = pl.min(n, m + q - 1)
                pair_count = _ceil(columns, 2 * lanes)
                if row_slot < band_owner_pair_prefix_family(
                    m, n, p, q, lanes, lane, pair_count, family
                ):
                    low = pl.astype(0, pl.DT_INT64)
                    high = pl.astype(pair_count, pl.DT_INT64)
                    while low + 1 < high:
                        mid = low + (high - low) // 2
                        if (
                            band_owner_pair_prefix_family(
                                m, n, p, q, lanes, lane, mid, family
                            )
                            <= row_slot
                        ):
                            low = mid
                        else:
                            high = mid
                    local_row = row_slot - band_owner_pair_prefix_family(
                        m, n, p, q, lanes, lane, low, family
                    )
                    y = band_owner_column_family(
                        m, n, p, q, 2 * lanes * low + lane, family
                    )
                    height = pl.max(0, pl.min(y + p - 1, m) - pl.max(1, y - q + 1) + 1)
                    if y > n:
                        height = pl.astype(0, pl.DT_INT64)
                    if local_row >= height:
                        local_row = local_row - height
                        y = band_owner_column_family(
                            m, n, p, q, 2 * lanes * low + 2 * lanes - 1 - lane, family
                        )
                    x = pl.astype(pl.max(1, y - q + 1) + local_row, pl.DT_INT64)
                    w = pl.astype(group * g + (r - 1 + lane) % g + 1, pl.DT_INT64)
    return (w, x, y)


def cal_band_column_owner(k, b, g, m, n, p, q, core, r):
    return cal_band_column_owner_family(k, b, g, m, n, p, q, core, r, deter_family)


def cal_band_folded_owner(k, b, g, m, n, p, q, core, r):
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if (
        k > 0
        and b > 0
        and g > 1
        and m > 0
        and n > 0
        and p > 0
        and p <= m
        and q > 0
        and q <= n
        and core >= 1
        and core <= k
        and r > 0
    ):
        size = pl.min(m, n)
        width = p + q - 1
        column_batch = (r - 1) // (width * g)
        column = column_batch * k + core - 1
        if column < b * size:
            virtual_column = column % size + 1
            diagonal = (r - 1) // g % width + 1 - p
            remainder = (virtual_column - diagonal - 1) % m
            if remainder < 0:
                remainder = remainder + m
            x = pl.astype(remainder + 1, pl.DT_INT64)
            y = pl.astype(x + diagonal, pl.DT_INT64)
            if y >= 1 and y <= n:
                w = pl.astype((column // size) * g + (r - 1) % g + 1, pl.DT_INT64)
    return w, x, y


def cal_band_owner_plan_family(ci, j, r, family):
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    h = ci.deterBandGroupsPerChunk
    if (
        h > 0
        and ci.deterBandFullRounds > 0
        and (r >= 1)
        and (r <= ci.deterBandTotalRounds)
        and (j >= 1)
        and (j <= ci.deterBandK)
    ):
        full_chunks = ci.deterBandB // h
        full_end = full_chunks * ci.deterBandFullRounds
        chunk = pl.astype((r - 1) // ci.deterBandFullRounds, pl.DT_INT64)
        groups = pl.astype(h, pl.DT_INT64)
        local_round = pl.astype(r - chunk * ci.deterBandFullRounds, pl.DT_INT64)
        if r > full_end:
            chunk = pl.astype(full_chunks, pl.DT_INT64)
            groups = pl.astype(ci.deterBandB % h, pl.DT_INT64)
            local_round = pl.astype(r - full_end, pl.DT_INT64)
        m = ci.deterBandPhase0M
        n = ci.deterBandPhase0N
        p = ci.deterBandPhase0RowOffset
        q = ci.deterBandPhase0ColOffset
        row_origin = pl.astype(0, pl.DT_INT64)
        col_origin = pl.astype(0, pl.DT_INT64)
        if family == 10:
            kind = ci.deterBandPhase0Kind
            if kind == 13:
                col_origin = pl.astype(q, pl.DT_INT64)
                q = p
                p = pl.astype(1, pl.DT_INT64)
            elif kind == 14:
                row_origin = pl.astype(p, pl.DT_INT64)
                p = q
                q = pl.astype(1, pl.DT_INT64)
            w, x, y = cal_band_folded_owner(
                ci.deterBandK, groups, ci.deterBandG, m, n, p, q, j, local_round
            )
        elif family == 9:
            row_origin = pl.astype(p, pl.DT_INT64)
            col_origin = pl.astype(q, pl.DT_INT64)
            w, x, y = cal_band_column_owner_family(
                ci.deterBandK, groups, ci.deterBandG, m, n, m, 1, j, local_round, family
            )
        else:
            w, x, y = cal_band_column_owner_family(
                ci.deterBandK, groups, ci.deterBandG, m, n, p, q, j, local_round, family
            )
        if w > 0:
            w = w + chunk * h * ci.deterBandG
            x = x + row_origin
            y = y + col_origin
    return (w, x, y)


def cal_band_owner_plan(ci, j, r):
    return cal_band_owner_plan_family(ci, j, r, deter_family)
