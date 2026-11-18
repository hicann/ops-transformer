# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed-layout deterministic locality decoders.

Each public decoder returns the AscendC one-based (head, S1, S2) coordinate.
The compilation-time deter_family selects exactly one implementation at its
caller; plan parameters never cause unrelated families to execute.
"""

from schedule.deterministic.fag_deter_formulas import (
    pl,
    pack_fused,
    _ceil,
    _causal_single,
    _unpack_wxy,
    cal_dense,
    cal_gqa_dense,
    cal_gqa_dense_precomputed,
    gqa_dense_parameters,
    cal_gqa_causal,
    CAUSAL_BOUNDARY_EXACT_INTEGER,
    CAUSAL_BOUNDARY_DEVICE_INTEGER_SQRT,
)
from schedule.deterministic.fag_deter_baseline import (
    cal_source_mha_dense,
    cal_source_mha_causal,
    cal_source_mha_band,
    cal_source_gqa_band,
    source_baseline_valid,
)
from schedule.deterministic.fag_deter_band_owner import (
    cal_band_owner_plan,
)
from schedule.deterministic.fag_deter_band_plan import (
    cal_band_piecewise_plan,
    cal_band_overlap_plan,
)


def cal_dense_swizzle(k, m, n, b, j, r):
    """CalDenseSwizzleIndex, including its reference boundary convention."""
    kk = pl.min(k, b * m)
    safe_m = pl.max(m, 1)
    safe_n = pl.max(n, 1)
    jj = j - 1
    rr = r - 1
    linear = (rr // safe_m) * kk + jj
    w = linear // safe_n + 1
    y = linear % safe_n + 1
    x = ((y - 1) + rr) % safe_m + 1
    # The reference tests zero-based j > k rather than >=. Do not silently
    # change the decoder: the family admission guarantees full active cores.
    valid = kk > 0 and m > 0 and n > 0 and b > 0
    valid = valid and jj >= 0 and jj <= kk and r >= 1
    valid = valid and w >= 1 and w <= b and x >= 1 and x <= m and y >= 1 and y <= n
    out_w = pl.astype(-1, pl.DT_INT64)
    if valid:
        out_w = pl.astype(w, pl.DT_INT64)
    return out_w, pl.astype(x, pl.DT_INT64), pl.astype(y, pl.DT_INT64)


def cal_causal_swizzle(k, m, n, b, j, r):
    n_new = n + 1
    if m != n:
        n_new = n - m + 2 + n + 1
    w, x, y = cal_dense_swizzle(k, m, n_new, b // 2, j, r)
    if w != -1:
        threshold = x + 1
        if m != n:
            threshold = x + n + 2 - m
        if y >= threshold:
            y = 2 * n - m - y + 2
            if m != n:
                y = y + 2
            x = m + 1 - x
            w = 2 * w
        else:
            w = 2 * w - 1
        if w < 1 or w > b or x < 1 or x > m or y < 1 or y > n:
            w = pl.astype(-1, pl.DT_INT64)
    return w, x, y


def cal_gqa_causal_batch_major(k, m, n, b, j, r, g, rounds_per_batch):
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if (
        k > 0
        and n > 0
        and m >= n
        and b > 0
        and g > 0
        and j >= 1
        and j <= k
        and r >= 1
        and rounds_per_batch > 0
    ):
        batch = (r - 1) // rounds_per_batch
        if batch < b:
            local_round = (r - 1) % rounds_per_batch + 1
            packed = cal_gqa_causal(k, m, n, 1, j, local_round, g, 1)
            ww, xx, yy = _unpack_wxy(packed, m, n)
            if ww >= 1 and ww <= g and yy >= 1 and yy <= n and yy <= xx and xx <= m:
                w = pl.astype(ww + batch * g, pl.DT_INT64)
                x = pl.astype(xx, pl.DT_INT64)
                y = pl.astype(yy, pl.DT_INT64)
    return w, x, y


def cal_gqa_causal_plan(k, m, n, b, j, r, g, rounds_per_batch):
    """Select causal ownership first, then share the coordinate decoder body."""
    local_b = pl.astype(b, pl.DT_INT64)
    local_r = pl.astype(r, pl.DT_INT64)
    batch = pl.astype(0, pl.DT_INT64)
    boundary_mode = pl.astype(CAUSAL_BOUNDARY_DEVICE_INTEGER_SQRT, pl.DT_INT64)
    if rounds_per_batch > 0:
        local_b = pl.astype(1, pl.DT_INT64)
        batch = (r - 1) // rounds_per_batch
        local_r = (r - 1) % rounds_per_batch + 1
        boundary_mode = pl.astype(CAUSAL_BOUNDARY_EXACT_INTEGER, pl.DT_INT64)
    packed = cal_gqa_causal(k, m, n, local_b, j, local_r, g, boundary_mode)
    w, x, y = _unpack_wxy(packed, m, n)
    if rounds_per_batch > 0:
        valid = k > 0 and n > 0 and m >= n and b > 0 and g > 0
        valid = valid and j >= 1 and j <= k and r >= 1 and batch < b
        valid = valid and w >= 1 and w <= g and y >= 1 and y <= n and y <= x and x <= m
        if valid:
            w = pl.astype(w + batch * g, pl.DT_INT64)
        else:
            w = pl.astype(-1, pl.DT_INT64)
    return w, x, y


def cal_gqa_causal_grouped(k, m, n, b, j, r, g, groups, rounds_per_group, total_rounds):
    """Decode one group, including its independently sized final remainder."""
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if (
        k > 0
        and n > 0
        and m >= n
        and b > 0
        and g > 0
        and j >= 1
        and j <= k
        and groups > 0
        and groups <= b
        and r >= 1
        and r <= total_rounds
        and rounds_per_group > 0
    ):
        full_count = b // groups
        full_end = full_count * rounds_per_group
        group_idx = pl.astype(full_count, pl.DT_INT64)
        local_groups = b % groups
        local_round = r - full_end
        if r <= full_end:
            group_idx = (r - 1) // rounds_per_group
            local_groups = pl.astype(groups, pl.DT_INT64)
            local_round = (r - 1) % rounds_per_group + 1
        if local_groups > 0:
            packed = cal_gqa_causal(
                k, m, n, local_groups, j, local_round, g, CAUSAL_BOUNDARY_EXACT_INTEGER
            )
            ww, xx, yy = _unpack_wxy(packed, m, n)
            if (
                ww >= 1
                and ww <= local_groups * g
                and yy >= 1
                and yy <= n
                and yy <= xx
                and xx <= m
            ):
                w = pl.astype(ww + group_idx * groups * g, pl.DT_INT64)
                x = pl.astype(xx, pl.DT_INT64)
                y = pl.astype(yy, pl.DT_INT64)
    return w, x, y


def cal_mha_band_grouped(ci, j, r):
    """Rotate physical cores before filtering the inactive part of each group."""
    k = ci.deterBandK
    b = ci.deterBandB
    groups = ci.deterBandGroupsPerChunk
    m = ci.deterBandPhase0M
    n = ci.deterBandPhase0N
    full_rounds = ci.deterBandFullRounds
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if (
        k > 0
        and b > 0
        and ci.deterBandG == 1
        and groups > 0
        and groups <= b
        and m > 0
        and n > 0
        and full_rounds > 0
        and r >= 1
        and r <= ci.deterBandTotalRounds
        and j >= 1
        and j <= k
    ):
        rr = r - 1
        full_count = b // groups
        full_end = full_count * full_rounds
        chunk = pl.astype(full_count, pl.DT_INT64)
        local_round = rr - full_end
        chunk_rounds = pl.astype(ci.deterBandTailRounds, pl.DT_INT64)
        if rr < full_end:
            chunk = rr // full_rounds
            local_round = rr % full_rounds
            chunk_rounds = pl.astype(full_rounds, pl.DT_INT64)
        head_base = chunk * groups
        group_size = pl.min(groups, b - head_base)
        if group_size > 0 and local_round < chunk_rounds:
            logical_core = (j - 1 + head_base) % k
            active_k = pl.min(k, group_size * m)
            if logical_core < active_k:
                ww, xx, yy = cal_dense(
                    active_k, m, n, group_size, logical_core + 1, local_round + 1
                )
                if ww > 0:
                    w = pl.astype(ww + head_base, pl.DT_INT64)
                    x = pl.astype(xx, pl.DT_INT64)
                    y = pl.astype(yy, pl.DT_INT64)
    return w, x, y


def cal_band_causal(k, m, n, p, q, b, j, r, upper):
    shift = q - 1
    size = m + shift
    if upper == 1:
        shift = p - 1
        size = n + shift
    pair_round = _ceil((size + 1) * (b // 2), pl.min(k, size * (b // 2))) * size
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if b % 2 != 0 and r > pair_round:
        packed = _causal_single(k, size, size, j, r - pair_round, 1)
        w, x, y = _unpack_wxy(packed, size, size)
        if w != -1:
            w = pl.astype(b, pl.DT_INT64)
    else:
        w, x, y = cal_causal_swizzle(k, size, size, (b // 2) * 2, j, r)
    if w != -1:
        if upper == 1:
            x = size + 1 - x
            y = n + 1 - y
        else:
            x = x - shift
        if x < 1 or x > m or y < 1 or y > n:
            w = pl.astype(-1, pl.DT_INT64)
    return w, x, y


def cal_band_hybrid(k, m, n, p, q, b, j, r):
    l1 = m - p
    l2 = p + q - m
    l3 = pl.min(n - q, m - 1)
    if p + q <= m:
        l1 = q - 1
        l2 = pl.min(n - q + 1, m + 2 - p - q)
        l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
    pair_count = pl.max(0, pl.min(l1, l3 - p + 1))
    columns = l1 + l2 + l3 - pair_count
    slot = p + q - 1
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if j >= 1 and j <= k and r >= 1 and slot > 0 and columns > 0:
        global_col = ((r - 1) // slot) * k + j
        local_round = (r - 1) % slot + 1
        if global_col <= b * columns:
            candidate_w = (global_col - 1) // columns + 1
            local_col = (global_col - 1) % columns + 1
            if local_col <= pair_count:
                y = pl.astype(local_col, pl.DT_INT64)
                x = y + local_round - q
                if x < pl.max(1, y - q + 1) or x > pl.min(m, y + p - 1):
                    y = l1 + l2 + p + local_col - 1
            elif local_col <= pair_count + l2:
                y = pl.astype(l1 + local_col - pair_count, pl.DT_INT64)
            else:
                single = local_col - pair_count - l2
                if single <= l1 - pair_count:
                    y = pl.astype(pair_count + single, pl.DT_INT64)
                else:
                    seg3 = single - (l1 - pair_count)
                    if seg3 > pl.min(p - 1, l3):
                        seg3 = seg3 + pair_count
                    y = pl.astype(l1 + l2 + seg3, pl.DT_INT64)
            x = y + local_round - q
            if x >= pl.max(1, y - q + 1) and x <= pl.min(m, y + p - 1):
                w = pl.astype(candidate_w, pl.DT_INT64)
    return w, x, y


def cal_mha_band(k, m, n, p, q, b, j, r, mode):
    # Host owns translations. Hybrid clamp is exactly GenBandHybridInfo.
    if mode != 0:
        p = pl.min(p, m)
        q = pl.min(q, n)
        m = pl.min(m, n + p - 1)
        n = pl.min(n, m + q - 1)
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if mode == 2:
        w, x, y = cal_dense_swizzle(k, m, n, b, j, r)
    elif mode == 4:
        w, x, y = cal_band_causal(k, m, n, p, q, b, j, r, 1)
    elif mode == 1:
        w, x, y = cal_band_causal(k, m, n, p, q, b, j, r, 0)
    elif mode == 3:
        w, x, y = cal_band_hybrid(k, m, n, p, q, b, j, r)
    else:
        w, x, y = cal_causal_swizzle(k, m, n, b, j, r)
    return w, x, y


def deter_fixed_gidx(
    ci,
    core,
    cores,
    physical_round,
    dense_k=0,
    dense_round=0,
    dense_t1=0,
    dense_t2=0,
):
    """Compile one decoder family and apply the Host's single translation."""
    w = pl.astype(-1, pl.DT_INT64)
    x = pl.astype(1, pl.DT_INT64)
    y = pl.astype(1, pl.DT_INT64)
    if deter_family == 1:
        w, x, y = cal_dense_swizzle(
            cores,
            ci.deterM,
            ci.deterN,
            ci.bSize * ci.n2Size,
            core + 1,
            physical_round + 1,
        )
    elif deter_family == 2:
        w, x, y = cal_causal_swizzle(
            cores,
            ci.deterM,
            ci.deterN,
            ci.bSize * ci.n2Size,
            core + 1,
            physical_round + 1,
        )
    elif deter_family == 3:
        w, x, y = cal_mha_band(
            cores,
            ci.deterM,
            ci.deterN,
            ci.deterP,
            ci.deterQ,
            ci.bSize * ci.n2Size,
            core + 1,
            physical_round + 1,
            ci.deterBandMode,
        )
    elif deter_family == 4:
        active_k = dense_k
        selected_round = dense_round
        t1 = dense_t1
        t2 = dense_t2
        if selected_round <= 0:
            active_k, selected_round, t1, t2 = gqa_dense_parameters(
                cores,
                ci.deterM,
                ci.deterN,
                ci.bSize * ci.n2Size,
                ci.gSize,
                ci.deterDenseRound,
            )
        w, x, y = cal_gqa_dense_precomputed(
            active_k,
            ci.deterM,
            ci.deterN,
            ci.bSize * ci.n2Size,
            core + 1,
            physical_round + 1,
            ci.gSize,
            selected_round,
            t1,
            t2,
        )
    elif deter_family == 5:
        packed = cal_gqa_causal(
            cores,
            ci.deterM,
            ci.deterN,
            ci.bSize * ci.n2Size,
            core + 1,
            physical_round + 1,
            ci.gSize,
            CAUSAL_BOUNDARY_DEVICE_INTEGER_SQRT,
        )
        w, x, y = _unpack_wxy(packed, ci.deterM, ci.deterN)
    elif deter_family == 6:
        w, x, y = cal_band_piecewise_plan(ci, core + 1, physical_round + 1)
    elif deter_family >= 7 and deter_family <= 10:
        w, x, y = cal_band_owner_plan(ci, core + 1, physical_round + 1)
    elif deter_family == 11:
        w, x, y = cal_band_overlap_plan(ci, core + 1, physical_round + 1)
    elif deter_family == 21:
        w, x, y = cal_source_mha_dense(
            cores,
            ci.deterM,
            ci.deterN,
            ci.bSize * ci.n2Size,
            core + 1,
            physical_round + 1,
        )
    elif deter_family == 22:
        w, x, y = cal_source_mha_causal(
            cores,
            ci.deterM,
            ci.deterN,
            ci.bSize * ci.n2Size,
            core + 1,
            physical_round + 1,
        )
    elif deter_family == 23:
        w, x, y = cal_source_mha_band(
            cores,
            ci.deterM,
            ci.deterN,
            ci.deterP,
            ci.deterQ,
            ci.bSize * ci.n2Size,
            core + 1,
            physical_round + 1,
        )
    elif deter_family == 24:
        w, x, y = cal_source_gqa_band(
            ci.deterBandK,
            ci.deterM,
            ci.deterN,
            ci.deterP,
            ci.deterQ,
            ci.bSize * ci.n2Size,
            ci.gSize,
            core + 1,
            physical_round + 1,
        )
    elif deter_family == 25:
        w, x, y = cal_gqa_causal_grouped(
            cores,
            ci.deterM,
            ci.deterN,
            ci.bSize * ci.n2Size,
            core + 1,
            physical_round + 1,
            ci.gSize,
            ci.deterCausalGroups,
            ci.deterCausalRoundsPerGroup,
            ci.deterMaxRound,
        )
    elif deter_family == 26:
        w, x, y = cal_mha_band_grouped(ci, core + 1, physical_round + 1)
    fused = pack_fused(
        w,
        x,
        y,
        ci.bSize,
        ci.n2Size,
        ci.gSize,
        ci.s1Outer,
        ci.s2Outer,
        ci.deterRowOffset,
        ci.deterColOffset,
    )
    if (deter_family >= 21 and deter_family <= 24) or deter_family == 26:
        valid = source_baseline_valid(
            ci.s1Size,
            ci.s2Size,
            ci.s1Token,
            ci.s2Token,
            x - 1 + ci.deterRowOffset,
            y - 1 + ci.deterColOffset,
        )
        fused = fused * valid - (1 - valid)
    return pl.astype(fused, pl.DT_INT64)
