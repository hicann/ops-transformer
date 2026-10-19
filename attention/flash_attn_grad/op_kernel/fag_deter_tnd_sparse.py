# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TND non-swizzle newDeter: AscendC classic causal and folded-band phases.

Square right-down causal mirrors the classic TND MHA/GQA phase schedule,
including GQA coreDivide. Band and rectangular causal use clipped virtual
band rectangles, not the physical dense grid. Their round lower bound is
corrected to cover BOTH dimensions of the folded rectangle (with G in M),
which is required by the dense gcd permutation for conflict-free writeback.

Line-deter is a separate optional AscendC locality optimization, not selected
here.  No swizzle, oldDeter, private workspace or additional barrier is added.
The returned synthetic max-shape index is NOT the normal TND packed index;
physical GM offsets still come exclusively from cu_seqlens.
"""

from fag_deter_tnd_dense import (
    pl,
    tnd_deter_lengths,
    tnd_permute_delta,
    make_tnd_deter_schedule,
    reset_tnd_batch_cursor,
)


def tnd_sparse_ceil(a, b):
    return (a + pl.max(b, 1) - 1) // pl.max(b, 1)


def tnd_sparse_nzmod(a, b):
    return (a - 1) % pl.max(b, 1) + 1


def tnd_sparse_ceil_tz(a, b):
    """AscendC Ceil(a,b) uses signed C++ division toward zero."""
    num = a + b - 1
    out = num // b
    if num < 0:
        out = -((-num) // b)
    return out


def tnd_sparse_abs_ceil(a, b):
    """AscendC AbsCeil: signed ceil of the magnitude."""
    out = tnd_sparse_ceil(a, b)
    if a < 0:
        out = -tnd_sparse_ceil(-a, b)
    return out


def tnd_sparse_band_shape(ci, sq, sk):
    """AscendC UpdateMNPQ and virtual band shape, with a safe round bound.

    Undo the max-shape right-down token shift before applying this batch's
    effective lengths. Physical cu offsets do not participate in geometry.
    Negative windows remove leading blank rows/columns, retaining mo/no so
    the final coordinate can be restored. Partially masked tiles are kept.
    """
    raw_m = tnd_sparse_ceil(sq, 128)
    raw_n = tnd_sparse_ceil(sk, 128)
    actual_pre = ci.s1Token - ci.s1Size + ci.s2Size + sq - sk
    actual_next = ci.s2Token + ci.s1Size - ci.s2Size - sq + sk
    p = pl.min(tnd_sparse_ceil_tz(actual_pre, 128) + 1, raw_m)
    q = pl.min(tnd_sparse_ceil_tz(actual_next, 128) + 1, raw_n)
    m = raw_m
    n = raw_n
    mo = pl.astype(0, pl.DT_INT64)
    no = pl.astype(0, pl.DT_INT64)
    if p < 0:
        no = -p
        n = n + p
        q = p + q
        p = pl.astype(1, pl.DT_INT64)
    elif q < 0:
        mo = -q
        m = m + q
        p = p + q
        q = pl.astype(1, pl.DT_INT64)
    actual_m = m
    actual_n = n
    bound = pl.astype(0, pl.DT_INT64)
    if p + q <= m:
        l1 = q - 1
        l2 = pl.min(n - q + 1, m + 2 - p - q)
        l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
        if l3 == 0:
            actual_m = p + q + l2 - 2
        actual_n = l1 + l2 + l3
        bound = (p + q - 1) * ci.gSize
        if n > m and ci.gSize != 1:
            bound = pl.max(actual_m * ci.gSize, p + q - 1)
    else:
        actual_n = pl.min(m - 1 + q, n)
        if p + q <= n:
            bound = pl.max(actual_m * ci.gSize, p + q - 1)
        else:
            bound = pl.max(actual_m * ci.gSize, actual_n)
    vm = actual_m
    vn = actual_n
    if p + q <= actual_m:
        if actual_n > actual_m or (ci.gSize == 1 and actual_n == actual_m):
            vn = p + q - 1
        else:
            vm = p + q - 1
    elif p + q <= actual_n:
        vn = p + q - 1
    whole = actual_m * actual_n - (actual_m - p) * (actual_m - p + 1) // 2
    whole = whole - (actual_n - q) * (actual_n - q + 1) // 2
    # AscendC's W*G-only bound can omit the long virtual axis. The gcd
    # permutation needs R >= max(vm*G, vn); unfold preserves x/y residues,
    # so distinct virtual dQ/dKV targets remain distinct after unfolding.
    bound = pl.max(bound, pl.max(vm * ci.gSize, vn))
    if sq <= 0 or sk <= 0 or actual_m <= 0 or actual_n <= 0 or p + q < 2:
        vm = pl.astype(0, pl.DT_INT64)
        vn = pl.astype(0, pl.DT_INT64)
        whole = pl.astype(0, pl.DT_INT64)
        bound = pl.astype(0, pl.DT_INT64)
    return actual_m, actual_n, p, q, mo, no, vm, vn, whole, bound


def tnd_sparse_causal_shape(m, n, phase):
    """AscendC CalVirtualIndex; an empty batch owns no virtual area."""
    n = pl.min(m, n)
    vm = 2 * m - n + 1
    vn = n
    if phase == 1:
        vm = m - (n + 1) // 2 + 1
        vn = n // 2
    elif phase == 2:
        vm = m - n // 2
        vn = (n + 1) // 2
    if m <= 0 or n <= 0:
        vm = pl.astype(0, pl.DT_INT64)
        vn = pl.astype(0, pl.DT_INT64)
    return vm, vn


def load_tnd_sparse_batch(ci, plan, phase, cuq, cuk, usedq, usedk):
    """Cache one batch's geometry for a deterministic phase.

    Tags 1/2/3/4 are causal whole/paired/first-tail/second-tail; tags 5/6
    are band whole/folded. Tag 0 is reserved for the dense cursor. Phase
    counts are not interchangeable, even for the same physical batch.
    """
    sq, sk = tnd_deter_lengths(ci, plan.cursorBatch, cuq, cuk, usedq, usedk)
    plan.cursorSq = pl.astype(sq, pl.DT_INT64)
    plan.cursorSk = pl.astype(sk, pl.DT_INT64)
    if phase <= 4:
        m = tnd_sparse_ceil(sq, 128)
        n = pl.min(m, tnd_sparse_ceil(sk, 128))
        vm = m
        vn = n
        area = (2 * m - n + 1) * n // 2
        if phase != 1:
            vm, vn = tnd_sparse_causal_shape(m, n, phase - 2)
            area = vm * vn
        plan.cursorM = pl.astype(m, pl.DT_INT64)
        plan.cursorN = pl.astype(n, pl.DT_INT64)
        plan.cursorVm = pl.astype(vm, pl.DT_INT64)
        plan.cursorVn = pl.astype(vn, pl.DT_INT64)
        plan.cursorArea = pl.astype(area, pl.DT_INT64)
    else:
        m, n, p, q, mo, no, vm, vn, area, bound = tnd_sparse_band_shape(ci, sq, sk)
        if phase == 6:
            area = vm * vn
        plan.cursorM = pl.astype(m, pl.DT_INT64)
        plan.cursorN = pl.astype(n, pl.DT_INT64)
        plan.cursorP = pl.astype(p, pl.DT_INT64)
        plan.cursorQ = pl.astype(q, pl.DT_INT64)
        plan.cursorMo = pl.astype(mo, pl.DT_INT64)
        plan.cursorNo = pl.astype(no, pl.DT_INT64)
        plan.cursorVm = pl.astype(vm, pl.DT_INT64)
        plan.cursorVn = pl.astype(vn, pl.DT_INT64)
        plan.cursorArea = pl.astype(area, pl.DT_INT64)


def seek_tnd_sparse_batch(ci, ident, copies, phase, plan, cuq, cuk, usedq, usedk):
    """Move an in-kernel phase cursor; no per-task scan from batch zero.

    Only boundary crossings reload geometry. Forward and reverse movement
    both skip zero-area batches via the same (begin, end] range convention.
    The caller guards inactive/out-of-range physical rounds and bSize == 0.
    """
    reset_tnd_batch_cursor(plan, phase)
    if plan.cursorReady == 0:
        load_tnd_sparse_batch(ci, plan, phase, cuq, cuk, usedq, usedk)
        plan.cursorEnd = pl.astype(plan.cursorArea * copies, pl.DT_INT64)
        plan.cursorReady = pl.astype(1, pl.DT_INT64)
    while ident > plan.cursorEnd and plan.cursorBatch + 1 < ci.bSize:
        plan.cursorBegin = pl.astype(plan.cursorEnd, pl.DT_INT64)
        plan.cursorBatch = pl.astype(plan.cursorBatch + 1, pl.DT_INT64)
        load_tnd_sparse_batch(ci, plan, phase, cuq, cuk, usedq, usedk)
        plan.cursorEnd = pl.astype(
            plan.cursorBegin + plan.cursorArea * copies, pl.DT_INT64
        )
    while ident <= plan.cursorBegin and plan.cursorBatch > 0:
        plan.cursorEnd = pl.astype(plan.cursorBegin, pl.DT_INT64)
        plan.cursorBatch = pl.astype(plan.cursorBatch - 1, pl.DT_INT64)
        load_tnd_sparse_batch(ci, plan, phase, cuq, cuk, usedq, usedk)
        plan.cursorBegin = pl.astype(
            plan.cursorEnd - plan.cursorArea * copies, pl.DT_INT64
        )


def init_tnd_sparse_schedule(ci, k, cuq, cuk, usedq, usedk):
    """Build one common physical-round plan from effective batch lengths."""
    all_equal = pl.astype(1, pl.DT_INT64)
    band_area = pl.astype(0, pl.DT_INT64)
    band_whole = pl.astype(0, pl.DT_INT64)
    band_bound = pl.astype(0, pl.DT_INT64)
    sum0 = pl.astype(0, pl.DT_INT64)
    sum1 = pl.astype(0, pl.DT_INT64)
    sum2 = pl.astype(0, pl.DT_INT64)
    max0 = pl.astype(0, pl.DT_INT64)
    max1 = pl.astype(0, pl.DT_INT64)
    max2 = pl.astype(0, pl.DT_INT64)
    for b in pl.range(ci.bSize):
        sq, sk = tnd_deter_lengths(ci, b, cuq, cuk, usedq, usedk)
        if sq != sk:
            all_equal = pl.astype(0, pl.DT_INT64)
        m = tnd_sparse_ceil(sq, 128)
        raw_n = tnd_sparse_ceil(sk, 128)
        n = pl.min(m, raw_n)
        if m > 0 and n > 0:
            sum0 = sum0 + (2 * m - n + 1) * n
            sum1 = sum1 + (m - (n + 1) // 2 + 1) * (n // 2)
            sum2 = sum2 + (m - n // 2) * ((n + 1) // 2)
            max0 = pl.max(max0, ci.gSize * (2 * m - n + 1))
            if ci.gSize == 1 or n >= 2:
                max1 = pl.max(max1, ci.gSize * (m - (n + 1) // 2 + 1))
            max2 = pl.max(max2, ci.gSize * (m - n // 2))
        am, an, p, q, mo, no, vm, vn, whole, bound = tnd_sparse_band_shape(ci, sq, sk)
        band_area = band_area + vm * vn
        band_whole = band_whole + whole
        band_bound = pl.max(band_bound, bound)
    causal = pl.astype(0, pl.DT_INT64)
    if mask_mode == 3:
        causal = all_equal
    rwhole = pl.astype(0, pl.DT_INT64)
    r0 = pl.astype(0, pl.DT_INT64)
    r1 = pl.astype(0, pl.DT_INT64)
    r2 = pl.astype(0, pl.DT_INT64)
    divided = pl.astype(0, pl.DT_INT64)
    max_round = pl.astype(0, pl.DT_INT64)
    kk = pl.max(k, 1)
    if k > 0:
        if causal == 1:
            if ci.gSize == 1:
                nfull = ci.n2Size // kk
                npair = (ci.n2Size % kk) // 2
                nlast = ci.n2Size % kk % 2
                rwhole = (sum0 // 2) * nfull
                if npair > 0:
                    r0 = pl.max(tnd_sparse_ceil(sum0 * npair, kk), max0)
                if nlast > 0:
                    r1 = pl.max(tnd_sparse_ceil(sum1, kk), max1)
                    r2 = pl.max(tnd_sparse_ceil(sum2, kk), max2)
                max_round = rwhole + r0 + r1 + r2
            else:
                npair = ci.n2Size // 2
                nlast = ci.n2Size % 2
                if npair > 0:
                    r0 = pl.max(tnd_sparse_ceil(sum0 * ci.gSize * npair, kk), max0)
                if nlast > 0:
                    r1 = pl.max(tnd_sparse_ceil(sum1 * ci.gSize, kk), max1)
                    r2 = pl.max(tnd_sparse_ceil(sum2 * ci.gSize, kk), max2)
                max_round = r0 + r1 + r2
                if npair > 0 and nlast > 0:
                    k2 = tnd_sparse_ceil(kk, ci.n2Size)
                    k1 = kk - k2
                    nr0 = pl.max(tnd_sparse_ceil(sum0 * ci.gSize * npair, k1), max0)
                    nr1 = pl.max(tnd_sparse_ceil(sum1 * ci.gSize, k2), max1)
                    nr2 = pl.max(tnd_sparse_ceil(sum2 * ci.gSize, k2), max2)
                    candidate = pl.max(nr0, nr1 + nr2)
                    if k1 > 0 and candidate < max_round:
                        r0 = nr0
                        r1 = nr1
                        r2 = nr2
                        max_round = candidate
                        divided = pl.astype(1, pl.DT_INT64)
        else:
            remainder = ci.n2Size * ci.gSize
            if ci.gSize == 1:
                rwhole = band_whole * (ci.n2Size // kk)
                remainder = ci.n2Size % kk
            if remainder > 0 and band_area > 0:
                r1 = pl.max(tnd_sparse_ceil(band_area * remainder, kk), band_bound)
            max_round = rwhole + r1
    return make_tnd_deter_schedule(max_round, r0, r1, r2, divided, causal, rwhole)


def tnd_sparse_whole_causal(m, n, a):
    """AscendC CalCausalPosWholeBatch."""
    neven = n // 2 * 2
    length = 2 * m - neven + 1
    local = neven * length // 2
    x = a - local - 1 + n
    y = n
    if a <= local:
        y = tnd_sparse_ceil(a, length)
        x = tnd_sparse_nzmod(a, length) + y - 1
        if x > m:
            x = 2 * m + 1 - x
            y = neven + 1 - y
    return x, y


def tnd_causal_gidx(ci, c, k, round_id, plan, cuq, cuk, usedq, usedk):
    """CalTNDCausalIndex / CalTNDGQACausalIndex with a cached phase cursor."""
    j = c + 1
    r = round_id + 1
    kk = pl.max(k, 1)
    phase = pl.astype(0, pl.DT_INT64)
    phase_r = r
    phase_max = plan.r0
    phase_heads = ci.n2Size // 2
    core = j
    whole_phase = pl.astype(0, pl.DT_INT64)
    active = pl.astype(1, pl.DT_INT64)
    if ci.gSize == 1:
        phase_heads = (ci.n2Size % kk) // 2
        if r <= plan.rWhole:
            whole_phase = pl.astype(1, pl.DT_INT64)
        elif r <= plan.rWhole + plan.r0:
            phase_r = r - plan.rWhole
        elif r <= plan.rWhole + plan.r0 + plan.r1:
            phase = pl.astype(1, pl.DT_INT64)
            phase_r = r - plan.rWhole - plan.r0
            phase_max = plan.r1
            phase_heads = pl.astype(1, pl.DT_INT64)
        else:
            phase = pl.astype(2, pl.DT_INT64)
            phase_r = r - plan.rWhole - plan.r0 - plan.r1
            phase_max = plan.r2
            phase_heads = pl.astype(1, pl.DT_INT64)
    elif plan.coreDivide == 0:
        if r > plan.r0:
            phase_heads = pl.astype(1, pl.DT_INT64)
            if r <= plan.r0 + plan.r1:
                phase = pl.astype(1, pl.DT_INT64)
                phase_r = r - plan.r0
                phase_max = plan.r1
            else:
                phase = pl.astype(2, pl.DT_INT64)
                phase_r = r - plan.r0 - plan.r1
                phase_max = plan.r2
    else:
        k1 = kk - tnd_sparse_ceil(kk, ci.n2Size)
        if j <= k1:
            if r > plan.r0:
                active = pl.astype(0, pl.DT_INT64)
        else:
            core = j - k1
            phase_heads = pl.astype(1, pl.DT_INT64)
            if r <= plan.r1:
                phase = pl.astype(1, pl.DT_INT64)
                phase_max = plan.r1
            else:
                phase = pl.astype(2, pl.DT_INT64)
                phase_r = r - plan.r1
                phase_max = plan.r2
    out = pl.astype(-1, pl.DT_INT64)
    if r > 0 and r <= plan.maxRound and c >= 0 and c < k and ci.bSize > 0:
        if whole_phase == 1:
            seek_tnd_sparse_batch(
                ci, r, ci.n2Size // kk, 1, plan, cuq, cuk, usedq, usedk
            )
            if r > plan.cursorBegin and r <= plan.cursorEnd:
                m = plan.cursorM
                n = plan.cursorN
                area = plan.cursorArea
                a = r - plan.cursorBegin
                x, y = tnd_sparse_whole_causal(m, n, tnd_sparse_nzmod(a, area))
                head = ((a - 1) // area) * kk + j - 1
                head = (head // kk) * kk + (y - 1 + head) % kk
                out = (
                    (plan.cursorBatch * ci.n2Size + head) * ci.gSize * ci.s1oS2o
                    + (y - 1) * ci.s1Outer
                    + x
                    - 1
                )
        elif active == 1 and phase_r > 0 and phase_r <= phase_max:
            ident = (core - 1) * phase_max + phase_r
            seek_tnd_sparse_batch(
                ci,
                ident,
                phase_heads * ci.gSize,
                phase + 2,
                plan,
                cuq,
                cuk,
                usedq,
                usedk,
            )
            if ident > plan.cursorBegin and ident <= plan.cursorEnd:
                m = plan.cursorM
                n = plan.cursorN
                head, go, x, y = tnd_permute_delta(
                    ident - plan.cursorBegin,
                    plan.cursorVm,
                    plan.cursorVn,
                    ci.gSize,
                    phase_max,
                )
                if phase < 2:
                    if x >= y + m - n + 1:
                        x = x - (m - n + 1)
                        head = head * 2
                    else:
                        x = m + 1 - x
                        y = n + 1 - y
                        head = head * 2 + 1
                    if phase == 0 and ci.gSize == 1:
                        head = head + (ci.n2Size // kk) * kk
                else:
                    x = x + n // 2
                if phase > 0:
                    head = ci.n2Size - 1
                if head < ci.n2Size and x > 0 and x <= m and y > 0 and y <= n:
                    out = (
                        ((plan.cursorBatch * ci.n2Size + head) * ci.gSize + go)
                        * ci.s1oS2o
                        + (y - 1) * ci.s1Outer
                        + x
                        - 1
                    )
    return pl.astype(out, pl.DT_INT64)


def tnd_sparse_whole_band(m, n, p, q, a):
    """AscendC CalPosWholeBatch for MHA groups of K complete heads."""
    l1 = q - 1
    l2 = pl.min(n - q + 1, m + 2 - p - q)
    l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
    r1 = (2 * p - 2 + q) * l1 // 2
    r2 = (p + q - 1) * l2
    wide = pl.astype(0, pl.DT_INT64)
    if p + q > m:
        wide = pl.astype(1, pl.DT_INT64)
        l1 = m - p
        l2 = p + q - m
        l3 = pl.min(m - 1, n - q)
        r1 = (p + m - 1) * l1 // 2
        r2 = m * l2
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if a <= r1:
        even = l1 // 2 * 2
        length = 2 * p + even - 1
        local = even * length // 2
        if a <= local:
            y = tnd_sparse_ceil(a, length)
            rr = tnd_sparse_nzmod(a, length)
            x = p + y - rr
            if x < 1:
                y = even + 1 - y
                x = 1 - x
            elif wide == 0:
                x = p + y - x
        else:
            x = a - local
            y = l1
    elif a <= r1 + r2:
        aa = a - r1
        height = p + q - 1
        if wide == 1:
            height = m
        y = tnd_sparse_ceil(aa, height)
        x = tnd_sparse_nzmod(aa, height)
        if wide == 0:
            x = x + y - 1
        y = y + l1
    else:
        aa = a - r1 - r2
        even = l3 // 2 * 2
        length = 2 * (p + q) - even - 3
        if wide == 1:
            length = 2 * m - even - 1
        local = even * length // 2
        if aa <= local:
            y = tnd_sparse_ceil(aa, length)
            rr = tnd_sparse_nzmod(aa, length)
            x = y + rr
            if wide == 0:
                x = x + 1 + m - (p + q)
            if x > m:
                y = even + 1 - y
                if wide == 1:
                    x = x - m + y
                else:
                    x = x - (p + q - 1) + y
        else:
            x = aa - local + l3
            if wide == 0:
                x = x + m - (p + q - 1)
            y = l3
        y = y + l1 + l2
    return x, y


def tnd_sparse_unfold_band(m, n, p, q, g, x, y):
    """AscendC sparse unfold; restore the folded axis in multiples of W."""
    valid = pl.astype(1, pl.DT_INT64)
    if p + q <= m:
        if n > m or (g == 1 and n == m):
            if x < y - q + 1 or x > p + y - 1:
                y = tnd_sparse_abs_ceil(x - (p + y - 1), p + q - 1) * (p + q - 1) + y
        else:
            if y < x - p + 1 or y > x + q - 1:
                x = tnd_sparse_abs_ceil(y - (q + x - 1), p + q - 1) * (p + q - 1) + x
    elif p + q <= n:
        if y < x - p + 1 or y > x + q - 1:
            if y < x - p + 1 and y + p + q - 1 <= n:
                y = y + p + q - 1
            else:
                valid = pl.astype(0, pl.DT_INT64)
    elif y < x - p + 1 or y > x + q - 1:
        valid = pl.astype(0, pl.DT_INT64)
    return x, y, valid


def tnd_band_gidx(ci, c, k, round_id, plan, cuq, cuk, usedq, usedk):
    """CalTNDBandIndex / CalTNDGQABandIndex using the corrected phase R.

    Whole-head MHA phase and core-major folded rectangles keep AscendC's
    order. Invalid virtual slots stay empty physical rounds; no per-core
    compaction is allowed because that would change the FIX write order.
    """
    r = round_id + 1
    kk = pl.max(k, 1)
    out = pl.astype(-1, pl.DT_INT64)
    if r > 0 and r <= plan.maxRound and c >= 0 and c < k and ci.bSize > 0:
        if ci.gSize == 1 and r <= plan.rWhole:
            seek_tnd_sparse_batch(
                ci, r, ci.n2Size // kk, 5, plan, cuq, cuk, usedq, usedk
            )
            if r > plan.cursorBegin and r <= plan.cursorEnd:
                area = plan.cursorArea
                a = r - plan.cursorBegin
                x, y = tnd_sparse_whole_band(
                    plan.cursorM,
                    plan.cursorN,
                    plan.cursorP,
                    plan.cursorQ,
                    tnd_sparse_nzmod(a, area),
                )
                head = ((a - 1) // area) * kk + c
                x = x - 1 + plan.cursorMo
                y = y - 1 + plan.cursorNo
                if (
                    x >= 0
                    and y >= 0
                    and x < tnd_sparse_ceil(plan.cursorSq, 128)
                    and y < tnd_sparse_ceil(plan.cursorSk, 128)
                ):
                    out = (
                        (plan.cursorBatch * ci.n2Size + head) * ci.s1oS2o
                        + y * ci.s1Outer
                        + x
                    )
        else:
            phase_r = r - plan.rWhole
            heads = ci.n2Size
            if ci.gSize == 1:
                heads = ci.n2Size % kk
            ident = c * plan.r1 + phase_r
            if phase_r > 0 and phase_r <= plan.r1:
                seek_tnd_sparse_batch(
                    ci, ident, heads * ci.gSize, 6, plan, cuq, cuk, usedq, usedk
                )
                if ident > plan.cursorBegin and ident <= plan.cursorEnd:
                    head, go, x, y = tnd_permute_delta(
                        ident - plan.cursorBegin,
                        plan.cursorVm,
                        plan.cursorVn,
                        ci.gSize,
                        plan.r1,
                    )
                    x, y, valid = tnd_sparse_unfold_band(
                        plan.cursorM,
                        plan.cursorN,
                        plan.cursorP,
                        plan.cursorQ,
                        ci.gSize,
                        x,
                        y,
                    )
                    if ci.gSize == 1:
                        head = head + (ci.n2Size // kk) * kk
                    x = x - 1 + plan.cursorMo
                    y = y - 1 + plan.cursorNo
                    if (
                        valid == 1
                        and head < ci.n2Size
                        and x >= 0
                        and x < tnd_sparse_ceil(plan.cursorSq, 128)
                        and y >= 0
                        and y < tnd_sparse_ceil(plan.cursorSk, 128)
                    ):
                        out = (
                            ((plan.cursorBatch * ci.n2Size + head) * ci.gSize + go)
                            * ci.s1oS2o
                            + y * ci.s1Outer
                            + x
                        )
    return pl.astype(out, pl.DT_INT64)


def tnd_sparse_gidx(ci, c, k, round_id, plan, cuq, cuk, usedq, usedk):
    """The caller applies effective per-batch token/mask validation."""
    out = pl.astype(-1, pl.DT_INT64)
    if plan.isCausal == 1:
        out = tnd_causal_gidx(ci, c, k, round_id, plan, cuq, cuk, usedq, usedk)
    else:
        out = tnd_band_gidx(ci, c, k, round_id, plan, cuq, cuk, usedq, usedk)
    return out
