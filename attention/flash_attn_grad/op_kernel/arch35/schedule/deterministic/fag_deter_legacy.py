# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Legacy family 0 dispatch, round bounds and CPU parity entry.

Shared coordinate arithmetic lives in fag_deter_formulas.
"""

from schedule.deterministic.fag_deter_formulas import (
    CUBE_BASEM,
    _ceil,
    fused_valid_for_mask,
    _gen_band,
    _gen_gqa_band,
    _in_range,
    _lt,
    _unpack_wxy,
    band_pq,
    cal_band,
    cal_causal,
    cal_dense,
    cal_gqa_band,
    cal_gqa_causal,
    cal_gqa_dense_precomputed,
    gqa_dense_parameters,
    pack_fused,
    pl,
    select_gqa_dense_round,
)


def _fused_valid(constInfo, s1o, s2o):
    """Bind the legacy compile-time mask for device and CPU parity callers."""
    return fused_valid_for_mask(constInfo, s1o, s2o, mask_mode)


def _causal_max_round(m, n, k, b, g):
    """Exclusive upper bound from the mapper's actual segment lengths.

    cal_causal/cal_gqa_causal concatenate full batches, paired batches,
    an optional single batch, then a column-pair tail. The last tail wave
    splits its two columns across cores when at most half the cores are used;
    it then takes only the longest column, not the full reflected pair length.
    """
    out = 0
    if k > 0 and m > 0 and n > 0 and b > 0 and g > 0:
        n = pl.min(n, m)
        b_tail = b % k
        rounds = (b // k) * (n * m - n * (n - 1) // 2)
        t = n // k
        ell = n % k
        t1 = n // (2 * k)
        rounds = rounds + (2 * m - t * k + 1) * t * (b_tail // 2)
        if b_tail % 2 == 1:
            if t % 2 == 1:
                m1 = m - t1 * 2 * k
                rm3 = (m + m1 + 1) * t1
                if ell == 0:
                    rounds = rounds + rm3 + m1
                else:
                    rounds = rounds + rm3 + pl.max(m1, 2 * m1 - 2 * k + 1)
                b_tail = b_tail - 1
            else:
                rounds = rounds + (2 * m - t * k + 1) * t // 2
        pairs = ((ell + 1) // 2) * b_tail
        pair_len = 2 * (m - n) + ell + 1 - ell % 2
        rounds = rounds + (pairs // k) * pair_len
        remaining = pairs % k
        if remaining > 0:
            if remaining <= k // 2:
                rounds = (
                    rounds + m - n + ell // 2 + 1 + (remaining - 1) // pl.max(b_tail, 1)
                )
            else:
                rounds = rounds + pair_len
        out = pl.max(rounds * g, 0)
    return out


def _dense_lin_max_round(k, m, n, bb, g):
    """核上 dense/GQA 闭式的矩形轮次。"""
    g = pl.max(g, 1)
    out = 0
    if k > 0 and m > 0 and n > 0 and bb > 0:
        if g <= 1:
            out = _ceil(n * bb, pl.min(k, m * bb)) * m
        else:
            out = select_gqa_dense_round(k, m, n, bb, g) * m
    return out


def _band_lin_max_round(k, m, n, bb, g, s1_token, s2_token):
    """mask4 / 矩形 causal(S1<S2) 的 band 轮次。p/q==0 是合法边界，不拒绝。"""
    actual_m, actual_n, p, q, _mo, _no = band_pq(m, n, s1_token, s2_token)
    out = m * n
    if actual_m <= 0 or actual_n <= 0:
        out = m * n
    elif g <= 1:
        (
            _k,
            _m,
            _n,
            _p,
            _q,
            _b,
            _b1,
            _b2,
            _l1,
            _l2,
            _l3,
            _ns,
            _r1,
            _r2,
            _r3,
            _rm,
            rmb,
            rm2,
        ) = _gen_band(k, actual_m, actual_n, p, q, bb)
        out = pl.max(rmb, 0) + pl.max(rm2, 0) + m
    else:
        _k, _m, _n, _p, _q, _b, _g, _b1, _b2, _l1, _l2, _l3, _rm, rmb, _r1, _r2, rm2 = (
            _gen_gqa_band(k, actual_m, actual_n, p, q, bb, g)
        )
        out = pl.max(rmb, 0) + pl.max(rm2, 0) + m
    return out


def deter_lin_max_round(constInfo, coreNum):
    """非 swizzle 轮次上界，与对应的 AscendC 闭式分核配套。"""
    k = coreNum
    b = constInfo.bSize
    n2 = constInfo.n2Size
    g = constInfo.gSize
    g = pl.max(g, 1)
    m = constInfo.s1Outer
    n = constInfo.s2Outer
    bb = b * n2
    out = 0
    dense_grid = constInfo.sequsedQOptional + constInfo.sequsedKvOptional
    if mask_mode == 0:
        dense_grid = 1
    if k > 0 and m > 0 and n > 0 and bb > 0:
        dense_r = _dense_lin_max_round(k, m, n, bb, g)
        if dense_grid > 0:
            out = dense_r
        else:
            band_r = _band_lin_max_round(
                k, m, n, bb, g, constInfo.s1Token, constInfo.s2Token
            )
            if mask_mode == 3:
                causal_r = _causal_max_round(m, n, k, bb, g)
                use_band = _lt(constInfo.s1Size, constInfo.s2Size)
                out = causal_r * (1 - use_band) + band_r * use_band
            else:
                out = band_r
    return pl.astype(out, pl.DT_INT64)


def deter_lin_round_bounds(constInfo, coreNum):
    """Common [first, end) physical rounds; never compact per-core holes.

    A small rectangular causal schedule (B*N2 < k, S2Outer < k) consists
    solely of the causal column-pair tail. On its rising half x-y=a-1,
    and on its reflected half x-y=pair_len-a (coordinates are 1-based).
    Right-down masking requires x-y >= floor((S1-S2)/128). This gives a
    provably empty global prefix, and a suffix on a reflected final wave.
    Interior empty rounds and the original absolute round numbers remain.
    """
    first = pl.astype(0, pl.DT_INT64)
    end = deter_lin_max_round(constInfo, coreNum)
    if mask_mode == 3:
        m = constInfo.s1Outer
        n = constInfo.s2Outer
        bb = constInfo.bSize * constInfo.n2Size
        g = pl.max(constInfo.gSize, 1)
        if (
            coreNum > 0
            and bb > 0
            and bb < coreNum
            and n > 0
            and n < coreNum
            and constInfo.s1Size >= constInfo.s2Size
            and constInfo.s2Token == constInfo.s2Size - constInfo.s1Size
            and constInfo.sequsedQOptional + constInfo.sequsedKvOptional == 0
        ):
            offset = (constInfo.s1Size - constInfo.s2Size) // CUBE_BASEM
            first = pl.astype(offset * g, pl.DT_INT64)
            pairs = ((n + 1) // 2) * bb
            remaining = pairs % coreNum
            if remaining == 0 or remaining > coreNum // 2:
                end = pl.astype(end - offset * g, pl.DT_INT64)
    return first, end


def _deter_lin_gidx_host(
    constInfo,
    cBlockIdx,
    coreNum,
    roundId,
    dense_k=0,
    dense_round=0,
    dense_t1=0,
    dense_t2=0,
):
    """host 对拍：可用运行期 if/return。"""
    k = coreNum
    j = cBlockIdx + 1
    r = roundId + 1
    b = constInfo.bSize
    n2 = constInfo.n2Size
    g = constInfo.gSize
    if g < 1:
        g = 1
    s1o = constInfo.s1Outer
    s2o = constInfo.s2Outer
    bb = b * n2
    moff = 0
    noff = 0
    if k <= 0 or j < 1 or j > k or r < 1 or s1o <= 0 or s2o <= 0 or b <= 0 or n2 <= 0:
        return -1
    if mask_mode == 0 or constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
        if g <= 1:
            w, x, y = cal_dense(k, s1o, s2o, bb, j, r)
        else:
            active_k = dense_k
            selected_round = dense_round
            t1 = dense_t1
            t2 = dense_t2
            if selected_round <= 0:
                selected_round = select_gqa_dense_round(k, s1o, s2o, bb, g)
                active_k, selected_round, t1, t2 = gqa_dense_parameters(
                    k, s1o, s2o, bb, g, selected_round
                )
            w, x, y = cal_gqa_dense_precomputed(
                active_k, s1o, s2o, bb, j, r, g, selected_round, t1, t2
            )
    elif mask_mode == 3 and constInfo.s1Size >= constInfo.s2Size:
        m = s1o
        n = s2o if s2o <= s1o else s1o
        if g <= 1:
            packed = cal_causal(k, m, n, bb, j, r)
        else:
            packed = cal_gqa_causal(k, m, n, bb, j, r, g)
        w, x, y = _unpack_wxy(packed, m, n)
    else:
        # mask4，以及 mask3 且 S1<S2：矩形 right-down causal 走 band。
        actual_m, actual_n, p, q, moff, noff = band_pq(
            s1o, s2o, constInfo.s1Token, constInfo.s2Token
        )
        if actual_m <= 0 or actual_n <= 0:
            return -1
        if g <= 1:
            packed = cal_band(k, actual_m, actual_n, p, q, bb, j, r)
            w, x, y = _unpack_wxy(packed, actual_m, actual_n)
        else:
            packed = cal_gqa_band(k, actual_m, actual_n, p, q, bb, g, j, r)
            w, x, y = _unpack_wxy(packed, actual_m, actual_n)
    fused = pack_fused(w, x, y, b, n2, g, s1o, s2o, moff, noff)
    if fused < 0:
        return -1
    s1_idx = fused % s1o
    s2_idx = (fused % (s1o * s2o)) // s1o
    if constInfo.sequsedQOptional + constInfo.sequsedKvOptional == 0:
        if _fused_valid(constInfo, s1_idx, s2_idx) == 0:
            return -1
    return fused


def deter_lin_gidx(
    constInfo,
    cBlockIdx,
    coreNum,
    roundId,
    dense_k=0,
    dense_round=0,
    dense_t1=0,
    dense_t2=0,
):
    """(cBlockIdx, roundId) 0-based → fusedIdx，空槽 -1。

    无 seqused 时保留对应的 AscendC dense/causal/band 闭式。
    seqused 的 batch 几何在核上读取；先枚举完整物理 dense 网格，再由
    skip_empty_deter 按有效长度和 mask 过滤，避免静态稀疏几何提前漏块。
    """
    k = coreNum
    j = cBlockIdx + 1
    r = roundId + 1
    b = constInfo.bSize
    n2 = constInfo.n2Size
    g = pl.max(constInfo.gSize, 1)
    s1o = constInfo.s1Outer
    s2o = constInfo.s2Outer
    bb = b * n2
    moff = 0
    noff = 0
    dense_grid = constInfo.sequsedQOptional + constInfo.sequsedKvOptional
    if mask_mode == 0:
        dense_grid = 1
    if dense_grid > 0:
        if g <= 1:
            w, x, y = cal_dense(k, s1o, s2o, bb, j, r)
        else:
            active_k = dense_k
            selected_round = dense_round
            t1 = dense_t1
            t2 = dense_t2
            if selected_round <= 0:
                selected_round = select_gqa_dense_round(k, s1o, s2o, bb, g)
                active_k, selected_round, t1, t2 = gqa_dense_parameters(
                    k, s1o, s2o, bb, g, selected_round
                )
            w, x, y = cal_gqa_dense_precomputed(
                active_k, s1o, s2o, bb, j, r, g, selected_round, t1, t2
            )
    else:
        if mask_mode == 3:
            # S1>=S2 走三角 causal；S1<S2 的矩形 right-down 与 ASC 一样切 band。
            causal_n = pl.min(s2o, s1o)
            packed = cal_causal(k, s1o, causal_n, bb, j, r)
            packed_g = cal_gqa_causal(k, s1o, causal_n, bb, j, r, g)
            wc, xc, yc = _unpack_wxy(packed, s1o, causal_n)
            wcg, xcg, ycg = _unpack_wxy(packed_g, s1o, causal_n)
            actual_m, actual_n, p, q, moff_b, noff_b = band_pq(
                s1o, s2o, constInfo.s1Token, constInfo.s2Token
            )
            packed_b = cal_band(k, actual_m, actual_n, p, q, bb, j, r)
            packed_bg = cal_gqa_band(k, actual_m, actual_n, p, q, bb, g, j, r)
            wb, xb, yb = _unpack_wxy(packed_b, actual_m, actual_n)
            wbg, xbg, ybg = _unpack_wxy(packed_bg, actual_m, actual_n)
            use_band = _lt(constInfo.s1Size, constInfo.s2Size)
            use_causal = 1 - use_band
            w = wc * use_causal + wb * use_band
            x = xc * use_causal + xb * use_band
            y = yc * use_causal + yb * use_band
            wg = wcg * use_causal + wbg * use_band
            xg = xcg * use_causal + xbg * use_band
            yg = ycg * use_causal + ybg * use_band
            moff = moff_b * use_band
            noff = noff_b * use_band
        else:
            actual_m, actual_n, p, q, moff, noff = band_pq(
                s1o, s2o, constInfo.s1Token, constInfo.s2Token
            )
            packed = cal_band(k, actual_m, actual_n, p, q, bb, j, r)
            w, x, y = _unpack_wxy(packed, actual_m, actual_n)
            packed_g = cal_gqa_band(k, actual_m, actual_n, p, q, bb, g, j, r)
            wg, xg, yg = _unpack_wxy(packed_g, actual_m, actual_n)
        is_mha = _in_range(g, 1, 1)
        w = w * is_mha + wg * (1 - is_mha)
        x = x * is_mha + xg * (1 - is_mha)
        y = y * is_mha + yg * (1 - is_mha)
    fused = pack_fused(w, x, y, b, n2, g, s1o, s2o, moff, noff)
    s1s = pl.max(s1o, 1)
    s2s = pl.max(s2o, 1)
    s1_idx = fused % s1s
    s2_idx = (fused % (s1s * s2s)) // s1s
    hit = pl.min(pl.max(fused + 1, 0), 1)
    if constInfo.sequsedQOptional + constInfo.sequsedKvOptional == 0:
        hit = hit * _fused_valid(constInfo, s1_idx, s2_idx)
    return pl.astype(fused, pl.DT_INT64) * hit + pl.astype(-1, pl.DT_INT64) * (1 - hit)


_deter_lin_gidx_device = deter_lin_gidx

if type(pl).__name__ == "_Pl":
    deter_lin_gidx = _deter_lin_gidx_host
