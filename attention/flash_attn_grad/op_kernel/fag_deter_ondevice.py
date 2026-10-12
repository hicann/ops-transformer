# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""确定性非 swizzle 分核：核上闭式，不读 metadata。

(j, r) 1-based。mask_mode 是 tilingkey，只编当前 mask。gSize 运行期，
MHA/GQA 两路都编。host 单测可给本模块赋 ``mask_mode``。
"""

try:
    import pypto_pro.language as pl
except ImportError:  # host 对拍

    class _Pl:
        DT_INT64 = 0

        @staticmethod
        def min(a, b):
            return a if a < b else b

        @staticmethod
        def max(a, b):
            return a if a > b else b

        @staticmethod
        def astype(x, _t):
            return int(x)

        range = staticmethod(range)

    pl = _Pl()

CUBE_BASEM = 128
CUBE_BASEN = 128


def _ceil(a, b):
    bb = pl.max(b, 1)
    return ((a + bb - 1) // bb) * pl.min(pl.max(b, 0), 1)


def _ceil_tz(a, b):
    """C++ ``Ceil(a,b)=(a+b-1)/b``，有符号除法向零截断。只用于 band token→p/q。

    不改 ``_ceil()``：其它路径仍用 Python ``//`` 向下取整。
    """
    bb = pl.max(b, 1)
    num = a + bb - 1
    neg = pl.min(pl.max(-num, 0), 1)
    pos_q = num // bb
    neg_q = -((-num) // bb)
    return (pos_q * (1 - neg) + neg_q * neg) * pl.min(pl.max(b, 0), 1)


def _nz(v, m):
    eq0 = 1 - pl.min(pl.max(v, 0), 1)
    return v * (1 - eq0) + m * eq0


def _gcd(a, b):
    """欧几里得。主循环里不能套 while，固定 64 步。"""
    x = pl.max(a, 0)
    y = pl.max(b, 0)
    for _gcd_i in pl.range(64):
        alive = pl.min(pl.max(y, 0), 1)
        t = x % pl.max(y, 1)
        x = x * (1 - alive) + y * alive
        y = y * (1 - alive) + t * alive
    return x


def _isqrt(n):
    out = 0
    if n > 0:
        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        out = x
    return out


def _in_range(v, lo, hi):
    return pl.min(pl.max(v + 1 - lo, 0), 1) * pl.min(pl.max(hi + 1 - v, 0), 1)


def _le(lhs, rhs):
    """Return 1 when ``lhs <= rhs`` and 0 otherwise (Scalar friendly)."""
    return pl.min(pl.max(rhs - lhs + 1, 0), 1)


def _lt(lhs, rhs):
    """Return 1 when ``lhs < rhs`` and 0 otherwise (Scalar friendly)."""
    return pl.min(pl.max(rhs - lhs, 0), 1)


def select_gqa_dense_round(k, m, n, b, g):
    """Mirror AscendC ``SelectGQADenseSchedule`` on device."""
    gg = pl.max(g, 1)
    kk = pl.min(pl.min(k, b * gg * m), b * n)
    kk_safe = pl.max(kk, 1)
    base = pl.max(pl.max(_ceil(b * n * gg, kk_safe), _ceil(n, m)), gg)
    candidate = _ceil(base, gg) * gg
    batch_head = pl.max(b * gg, 1)
    base_period = batch_head // pl.max(_gcd(batch_head, base), 1)
    candidate_period = batch_head // pl.max(_gcd(batch_head, candidate), 1)
    candidate_invalid = kk * candidate - b * n * gg

    # Keep 100/25 in lockstep with the AscendC host selector.
    round_cost_ok = _le((candidate - base) * 100, base * 25)
    invalid_cost_ok = _le(candidate_invalid * 100, kk * candidate * 25)
    locality_better = _lt(candidate_period, base_period) * _lt(candidate_period, kk)
    row_offset_enough = _le(_ceil(kk, candidate_period), m)
    use_candidate = (
        round_cost_ok * invalid_cost_ok * locality_better * row_offset_enough
    )
    return base + (candidate - base) * use_candidate


def cal_dense(k, m, n, b, j, r):
    k = pl.min(k, b * m)
    mm = pl.max(m, 1)
    bb = pl.max(b, 1)
    p = (_ceil(r, mm) - 1) * k + j
    w = _nz(p % bb, bb)
    y = _ceil(p, bb)
    y1 = _nz(y % mm, mm)
    r1 = _nz(r % mm, mm)
    x = y1 + r1 - 1
    overx = pl.min(pl.max(x - m, 0), 1)
    x = x - m * overx
    ok = (
        _in_range(k, 1, k)
        * _in_range(m, 1, m)
        * _in_range(n, 1, n)
        * _in_range(b, 1, b)
        * _in_range(j, 1, k)
        * _in_range(w, 1, b)
        * _in_range(x, 1, m)
        * _in_range(y, 1, n)
    )
    w = w * ok + (-1) * (1 - ok)
    return w, x * ok, y * ok


def cal_gqa_dense(k, m, n, b, core_id, round_id, g, dense_round=0):
    k = pl.min(pl.min(k, b * g * m), b * n)
    rmax_def = pl.max(pl.max(_ceil(b * n * g, k), _ceil(n, m)), g)
    use_dr = pl.min(pl.max(dense_round, 0), 1)
    rmax = dense_round * use_dr + rmax_def * (1 - use_dr)
    ok = _in_range(core_id, 1, k) * _in_range(round_id, 1, rmax * m)
    idx = (core_id - 1) * rmax + _ceil(round_id, m)
    local_id = _nz(round_id % pl.max(m, 1), pl.max(m, 1))
    ok = ok * pl.min(pl.max(g * n * b + 1 - idx, 0), 1)
    nbg = pl.max(b * g, 1)
    gg = pl.max(g, 1)
    b_id = _ceil(_nz(idx % nbg, nbg), gg)
    y = _ceil(idx, nbg)
    w = _nz(idx % gg, gg)
    gcd = pl.max(_gcd(nbg, rmax), 1)
    t1 = rmax // gcd
    t2 = nbg // gcd
    t1_new = t1 * m
    t1s = pl.max(t1, 1)
    t1n = pl.max(t1_new, 1)
    y1 = _nz(y % t1n, t1n)
    offset = _ceil(y1, t1s)
    use = pl.min(pl.max(n - t1_new, 0), 1)
    n1 = _nz(n % t1n, t1n)
    use2 = use * pl.min(pl.max(n - n1 + 1 - y, 0), 1)
    delta = _ceil(y, t1n)
    idx2 = idx + delta
    lim = (delta - 1) * t2 * m * rmax + offset * t2 * rmax
    wrap = pl.min(pl.max(idx2 - lim, 0), 1)
    idx2 = idx2 - t2 * rmax * wrap
    b_id2 = _ceil(_nz(idx2 % nbg, nbg), gg)
    w2 = _nz(idx2 % gg, gg)
    y2 = _ceil(idx2, nbg)
    b_id = b_id * (1 - use2) + b_id2 * use2
    w = w * (1 - use2) + w2 * use2
    y = y * (1 - use2) + y2 * use2
    x = local_id + offset - 1
    overx = pl.min(pl.max(x - m, 0), 1)
    x = x - m * overx
    outw = (w + (b_id - 1) * g) * ok + (-1) * (1 - ok)
    return outw, x * ok, y * ok


def cal_gqa_dense_no_tune(k, m, n, b, core_id, round_id, g):
    k = pl.min(pl.min(k, b * g * m), b * n)
    rmax = pl.max(pl.max(_ceil(b * n * g, k), _ceil(n, m)), g)
    ok = _in_range(core_id, 1, k) * _in_range(round_id, 1, rmax * m)
    mm = pl.max(m, 1)
    idx = (core_id - 1) * rmax + _ceil(round_id, mm)
    local_id = _nz(round_id % mm, mm)
    ok = ok * pl.min(pl.max(g * n * b + 1 - idx, 0), 1)
    nbg = pl.max(b * g, 1)
    gg = pl.max(g, 1)
    b_id = _ceil(_nz(idx % nbg, nbg), gg)
    w = _nz(idx % gg, gg) + (b_id - 1) * gg
    outw = w * ok + (-1) * (1 - ok)
    return outw, local_id * ok, _ceil(idx, nbg) * ok


def _causal_g2k(k, j, a, l1, offset):
    odd = j % 2
    y_near = j + offset
    y_far = 2 * k + 1 - j + offset

    odd_near = _le(a, l1 - j + 1)
    x_odd_near = y_near + a - 1
    x_odd_far = y_far + 2 * l1 - 2 * k + 1 - a
    x_odd = x_odd_near * odd_near + x_odd_far * (1 - odd_near)
    y_odd = y_near * odd_near + y_far * (1 - odd_near)

    even_near = _le(l1 - 2 * k + 1 + j, a)
    x_even_near = y_near + 2 * l1 - 2 * k + 1 - a
    x_even_far = y_far + a - 1
    x_even = x_even_near * even_near + x_even_far * (1 - even_near)
    y_even = y_near * even_near + y_far * (1 - even_near)

    x = x_odd * odd + x_even * (1 - odd)
    y = y_odd * odd + y_even * (1 - odd)
    return 0, x, y


def _causal_no_rec(k, m, n, j, r):
    nmod = n % 2
    odd = j % 2
    odd_near = _le(r + j, n + 1)
    x_odd_near = r + j - 1
    x_odd_far = 2 * n + 2 - j - r
    y_far = n + 3 - j - nmod
    x_odd = x_odd_near * odd_near + x_odd_far * (1 - odd_near)
    y_odd = j * odd_near + y_far * (1 - odd_near)

    even_near = _le(j, r + 1 - nmod)
    x_even_near = n + j - r - 1 + nmod
    x_even_far = n + 2 + r - j - nmod
    x_even = x_even_near * even_near + x_even_far * (1 - even_near)
    y_even = j * even_near + y_far * (1 - even_near)

    x = x_odd * odd + x_even * (1 - odd)
    y = y_odd * odd + y_even * (1 - odd)
    valid = _le(j, (n // 2) + 1) * _in_range(y, 1, m) * _le(y, x) * _le(x, m)
    return -(1 - valid), x * valid, y * valid


def _causal_rec(k, m, n, j, r):
    use_g2k = _lt(2 * k, m + 1) * _lt(k, n)
    _wg, xg, yg = _causal_g2k(k, j, r, m, 0)
    _wn, xn, yn = _causal_no_rec(k, m, m, j, r)
    x = xg * use_g2k + xn * (1 - use_g2k)
    y = yg * use_g2k + yn * (1 - use_g2k)
    valid = _in_range(y, 1, n) * _le(y, x) * _le(x, m)
    return -(1 - valid), x * valid, y * valid


def _causal_single(k, m, n, j, r):
    invalid = _pack_wxy(-1, 0, 0, m, n)
    out = invalid
    if k >= (n // 2) + 1:
        w, x, y = _causal_rec(k, m, n, j, r)
        out = _pack_wxy(w, x, y, m, n)
    else:
        ell = n % k
        t1 = n // (2 * k)
        t3 = (n // k) % 2
        bound1 = (2 * m + 1) * t1 - 2 * k * t1 * t1
        if r <= bound1:
            disc = (2 * m + 1) * (2 * m + 1) - 8 * k * r
            i = _ceil((2 * m + 1) - _isqrt(disc), 4 * k)
            rm = (2 * m + 1) * (i - 1) - 2 * k * (i - 1) * (i - 1)
            offset = 2 * k * (i - 1)
            w, x, y = _causal_g2k(k, j, r - rm, m - 2 * k * (i - 1), offset)
            out = _pack_wxy(w, x, y, m, n)
        else:
            rem = t3 * k + ell
            if rem > 0:
                l1 = m - n + rem
                a = r - bound1
                if t3 == 0 and j <= rem:
                    y = 2 * k * t1 + j
                    out = _pack_wxy(0, y + a - 1, y, m, n)
                elif t3 == 1:
                    w, x, y = _causal_rec(k, l1, l1, j, a)
                    if w != -1:
                        sh = 2 * k * t1
                        out = _pack_wxy(0, x + sh, y + sh, m, n)
                    else:
                        out = invalid
                else:
                    out = invalid
            else:
                out = invalid
    return out


def _pack_wxy(w, x, y, m, n):
    """Encode a dynamic ``(w, x, y)`` result as one scalar.

    PyPTO cannot tuple-unpack helpers with several control-flow returns.  A
    zero packed value represents ``(-1, 0, 0)``; valid coordinates are
    losslessly encoded with a base larger than either outer dimension.
    """
    base = pl.max(pl.max(m, n), 0) + 1
    packed = ((w + 1) * base + x) * base + y
    # Formulas fed by PyPTO INDEX values and DT_INT64 values both render as
    # ``int64_t`` but are distinct IR scalar types and cannot share a phi.
    return pl.astype(packed, pl.DT_INT64)


def _unpack_wxy(packed, m, n):
    base = pl.max(pl.max(m, n), 0) + 1
    y = packed % base
    x = (packed // base) % base
    w = packed // (base * base) - 1
    return w, x, y


def _causal_tail_packed(k, m, n, b1, b2, j, a4, t, ell, delta):
    """Map the causal remainder with a fully typed single return slot."""
    invalid = _pack_wxy(-1, 0, 0, m, n)
    p = _ceil(ell, 2)
    ell1 = ell + 1 - (ell % 2)
    block = (b2 * p) // k
    res0 = (b2 * p) % k
    den = ell1 + 2 * delta
    if a4 > block * den and res0 <= k // 2:
        offs = a4 - block * den
        if j >= 1 and j <= res0:
            limit = (res0 - j) // b2 + _ceil(ell1, 2) + delta
            if offs <= limit:
                w = _nz((k * ((a4 - 1) // den) + j) % b2, b2)
                y = p - (res0 - j) // b2
                x = y + offs - 1
                if y >= 1 and y <= ell and y <= x and x <= (ell + delta):
                    out = _pack_wxy(b1 * k + w, x + t * k, y + t * k, m, n)
                else:
                    out = invalid
            else:
                out = invalid
        elif k - res0 + 1 <= j and j <= k:
            idx = j - (k - res0 + 1)
            limit = (ell1 - 1) // 2 - idx // b2 + delta
            if offs <= limit:
                w = _nz((k * ((a4 - 1) // den) + k + 1 - j) % b2, b2)
                y = p + 1 + idx // b2
                x = y + offs - 1
                if y >= 1 and y <= ell and y <= x and x <= (ell + delta):
                    out = _pack_wxy(b1 * k + w, x + t * k, y + t * k, m, n)
                else:
                    out = invalid
            else:
                out = invalid
        else:
            out = invalid
    else:
        w = _nz((k * ((a4 - 1) // den) + j) % b2, b2)
        gg = _ceil(k * ((a4 - 1) // den) + j, b2)
        if gg >= 1 and gg <= p:
            a5 = _nz(a4 % den, den)
            odd = gg % 2
            y_far = ell + 1 + (ell % 2) - gg
            odd_near = _le(a5, ell - gg + 1 + delta)
            x_odd_near = gg + a5 - 1
            x_odd_far = 2 * ell + 2 * delta + 2 - gg - a5
            x_odd = x_odd_near * odd_near + x_odd_far * (1 - odd_near)
            y_odd = gg * odd_near + y_far * (1 - odd_near)
            even_near = _le(gg + 1 + delta - (ell % 2), a5)
            x_even_near = gg + ell + 2 * delta + 1 - (ell % 2) - a5
            x_even_far = a5 + ell - gg + (ell % 2)
            x_even = x_even_near * even_near + x_even_far * (1 - even_near)
            y_even = gg * even_near + y_far * (1 - even_near)
            x0 = x_odd * odd + x_even * (1 - odd)
            y0 = y_odd * odd + y_even * (1 - odd)
            if y0 >= 1 and y0 <= ell and y0 <= x0 and x0 <= ell + delta:
                out = _pack_wxy(b1 * k + w, x0 + t * k, y0 + t * k, m, n)
            else:
                out = invalid
        else:
            out = invalid
    return out


def cal_causal(k, m, n, b, j, r):
    invalid = _pack_wxy(-1, 0, 0, m, n)
    out = invalid
    b1 = b // k
    b2 = b % k
    delta = m - n
    size_tri = n * (m + delta + 1) // 2
    rm1 = b1 * size_tri
    if r <= rm1:
        a = _nz(r % size_tri, size_tri)
        w = _ceil(r, size_tri) + b1 * (j - 1)
        n1 = n // 2 * 2
        leng = 2 * m - n1 + 1
        rm_local = (n1 * leng) // 2
        in_main = _le(a, rm_local)
        y_main = _ceil(a, leng)
        r1 = _nz(a % leng, leng)
        x_main = r1 + y_main - 1
        reflect = _lt(m, x_main)
        x_main = x_main * (1 - reflect) + (2 * m + 1 - x_main) * reflect
        y_main = y_main * (1 - reflect) + (n1 + 1 - y_main) * reflect
        y = y_main * in_main + n * (1 - in_main)
        x = x_main * in_main + (a - rm_local - 1 + n) * (1 - in_main)
        w = ((w - 1) % b1) * k + (w - 1) // b1 + 1
        w = ((w - 1) // k) * k + ((y - 1 + (w - 1)) % k) + 1
        out = _pack_wxy(w, x, y, m, n)
    else:
        t = n // k
        ell = n % k
        a2 = r - rm1
        half = b2 // 2
        rm2 = (2 * m - t * k + 1) * t * half
        if a2 >= 1 and a2 <= rm2:
            wsub, xsub, ysub = cal_dense(k, 2 * m - t * k + 1, t * k, half, j, a2)
            max_x = 2 * m - t * k + 1
            if (
                xsub >= 1
                and xsub <= max_x
                and ysub >= 1
                and ysub <= t * k
                and wsub >= 1
                and wsub <= half
            ):
                upper = _le(xsub - ysub, m - t * k)
                wout = (2 * wsub - 1 + b1 * k) * upper + (2 * wsub + b1 * k) * (
                    1 - upper
                )
                xout = (m + 1 - xsub) * upper + (xsub - m + t * k - 1) * (1 - upper)
                yout = (t * k + 1 - ysub) * upper + ysub * (1 - upper)
                out = _pack_wxy(wout, xout, yout, m, n)
            else:
                out = invalid
        else:
            a3 = r - rm1 - rm2
            if b2 % 2 == 1:
                t1 = n // (2 * k)
                t3 = t % 2
                if t3 == 1:
                    m1 = m - t1 * 2 * k
                    rm3 = (m + m1 + 1) * t1
                    if ell == 0:
                        rm3 = rm3 + m1
                    else:
                        rm3 = rm3 + pl.max(m1, 2 * m1 - 2 * k + 1)
                    if a3 >= 1 and a3 <= rm3:
                        single = _causal_single(k, m, n, j, a3)
                        w, x, y = _unpack_wxy(single, m, n)
                        if w != -1:
                            out = _pack_wxy(b, x, y, m, n)
                        else:
                            out = invalid
                    else:
                        out = _causal_tail_packed(
                            k, m, n, b1, b2 - 1, j, a3 - rm3, t, ell, delta
                        )
                else:
                    rm3 = (2 * m - t * k + 1) * t // 2
                    if a3 >= 1 and a3 <= rm3:
                        single = _causal_single(k, m, t * k, j, a3)
                        w, x, y = _unpack_wxy(single, m, t * k)
                        if w != -1:
                            out = _pack_wxy(b, x, y, m, n)
                        else:
                            out = invalid
                    else:
                        out = _causal_tail_packed(
                            k, m, n, b1, b2, j, a3 - rm3, t, ell, delta
                        )
            else:
                out = _causal_tail_packed(k, m, n, b1, b2, j, a3, t, ell, delta)
    return out


def _gqa_causal_tail_packed(k, m, n, b1, b2, j, a4, t, ell, delta, g):
    """GQA causal remainder with one typed return slot for PyPTO."""
    invalid = _pack_wxy(-1, 0, 0, m, n)
    out = invalid
    n1_id = _nz(a4 % g, g)
    a4 = _ceil(a4, g)
    p = _ceil(ell, 2)
    ell1 = ell + 1 - (ell % 2)
    block = (b2 * p) // k
    res0 = (b2 * p) % k
    den = ell1 + 2 * delta
    if a4 > block * den and res0 <= k // 2:
        offs = a4 - block * den
        if j >= 1 and j <= res0:
            limit = (res0 - j) // b2 + _ceil(ell1, 2) + delta
            if offs <= limit:
                w = _nz((k * ((a4 - 1) // den) + j) % b2, b2)
                y = p - (res0 - j) // b2
                x = y + offs - 1
                if y >= 1 and y <= ell and y <= x and x <= (ell + delta):
                    out = _pack_wxy(
                        (w - 1) * g + n1_id + b1 * k * g, x + t * k, y + t * k, m, n
                    )
                else:
                    out = invalid
            else:
                out = invalid
        elif k - res0 + 1 <= j and j <= k:
            idx = j - (k - res0 + 1)
            limit = (ell1 - 1) // 2 - idx // b2 + delta
            if offs <= limit:
                w = _nz((k * ((a4 - 1) // den) + k + 1 - j) % b2, b2)
                y = p + 1 + idx // b2
                x = y + offs - 1
                if y >= 1 and y <= ell and y <= x and x <= (ell + delta):
                    out = _pack_wxy(
                        (w - 1) * g + n1_id + b1 * k * g, x + t * k, y + t * k, m, n
                    )
                else:
                    out = invalid
            else:
                out = invalid
        else:
            out = invalid
    else:
        w = _nz((k * ((a4 - 1) // den) + j) % b2, b2)
        gtail = _ceil(k * ((a4 - 1) // den) + j, b2)
        if gtail >= 1 and gtail <= p:
            a5 = _nz(a4 % den, den)
            odd = gtail % 2
            y_far = ell + 1 + (ell % 2) - gtail
            odd_near = _le(a5, ell - gtail + 1 + delta)
            x_odd_near = gtail + a5 - 1
            x_odd_far = 2 * ell + 2 * delta + 2 - gtail - a5
            x_odd = x_odd_near * odd_near + x_odd_far * (1 - odd_near)
            y_odd = gtail * odd_near + y_far * (1 - odd_near)
            even_near = _le(gtail + 1 + delta - (ell % 2), a5)
            x_even_near = gtail + ell + 2 * delta + 1 - (ell % 2) - a5
            x_even_far = a5 + ell - gtail + (ell % 2)
            x_even = x_even_near * even_near + x_even_far * (1 - even_near)
            y_even = gtail * even_near + y_far * (1 - even_near)
            x0 = x_odd * odd + x_even * (1 - odd)
            y0 = y_odd * odd + y_even * (1 - odd)
            if y0 >= 1 and y0 <= ell and y0 <= x0 and x0 <= ell + delta:
                out = _pack_wxy(
                    (w - 1) * g + n1_id + b1 * k * g, x0 + t * k, y0 + t * k, m, n
                )
            else:
                out = invalid
        else:
            out = invalid
    return out


def cal_gqa_causal(k, m, n, b, j, r, g):
    invalid = _pack_wxy(-1, 0, 0, m, n)
    out = invalid
    b1 = b // k
    b2 = b % k
    delta = m - n
    size_tri = n * (m + delta + 1) // 2
    size_g = size_tri * g
    rm1 = b1 * size_g
    if r <= rm1:
        a_group = _nz(r % size_g, size_g)
        n1_id = _ceil(a_group, size_tri)
        a_tri = _nz(r % size_tri, size_tri)
        b_id = k * (_ceil(r, size_g) - 1) + j
        n1 = n // 2 * 2
        leng = 2 * m - n1 + 1
        rm_local = (n1 * leng) // 2
        in_main = _le(a_tri, rm_local)
        y_main = _ceil(a_tri, leng)
        r1 = _nz(a_tri % leng, leng)
        x_main = r1 + y_main - 1
        reflect = _lt(m, x_main)
        x_main = x_main * (1 - reflect) + (2 * m + 1 - x_main) * reflect
        y_main = y_main * (1 - reflect) + (n1 + 1 - y_main) * reflect
        y = y_main * in_main + n * (1 - in_main)
        x = x_main * in_main + (a_tri - rm_local - 1 + n) * (1 - in_main)
        b_id = ((b_id - 1) // k) * k + ((y - 1 + (b_id - 1)) % k) + 1
        out = _pack_wxy((b_id - 1) * g + n1_id, x, y, m, n)
    else:
        t = n // k
        ell = n % k
        a2 = r - rm1
        half = b2 // 2
        rm2 = (2 * m - t * k + 1) * t * half * g
        if a2 >= 1 and a2 <= rm2:
            wsub, xsub, ysub = cal_gqa_dense(
                k, 2 * m - t * k + 1, t * k, half, j, a2, g
            )
            max_x = 2 * m - t * k + 1
            if (
                xsub >= 1
                and xsub <= max_x
                and ysub >= 1
                and ysub <= t * k
                and wsub >= 1
                and wsub <= half * g
            ):
                b_id = _ceil(wsub, g)
                n1_id = _nz(wsub % g, g)
                upper = _le(xsub - ysub, m - t * k)
                b_id = 2 * b_id - upper
                x_upper = m + 1 - xsub
                y_upper = t * k + 1 - ysub
                x_lower = xsub - m + t * k - 1
                x = x_upper * upper + x_lower * (1 - upper)
                y = y_upper * upper + ysub * (1 - upper)
                out = _pack_wxy(b1 * k * g + (b_id - 1) * g + n1_id, x, y, m, n)
            else:
                out = invalid
        else:
            a3 = r - rm1 - rm2
            if b2 % 2 == 1:
                t1 = n // (2 * k)
                t3 = t % 2
                if t3 == 1:
                    m1 = m - t1 * 2 * k
                    rm3 = (m + m1 + 1) * t1
                    if ell == 0:
                        rm3 = rm3 + m1
                    else:
                        rm3 = rm3 + pl.max(m1, 2 * m1 - 2 * k + 1)
                    if a3 >= 1 and a3 <= rm3 * g:
                        n1_id = _ceil(a3, rm3)
                        a3r = _nz(a3 % rm3, rm3)
                        single = _causal_single(k, m, n, j, a3r)
                        w, x, y = _unpack_wxy(single, m, n)
                        if w != -1:
                            out = _pack_wxy((b - 1) * g + n1_id, x, y, m, n)
                        else:
                            out = invalid
                    else:
                        out = _gqa_causal_tail_packed(
                            k, m, n, b1, b2 - 1, j, a3 - rm3 * g, t, ell, delta, g
                        )
                else:
                    rm3 = (2 * m - t * k + 1) * t // 2
                    if a3 >= 1 and a3 <= rm3 * g:
                        n1_id = _ceil(a3, rm3)
                        a3r = _nz(a3 % rm3, rm3)
                        single = _causal_single(k, m, t * k, j, a3r)
                        w, x, y = _unpack_wxy(single, m, t * k)
                        if w != -1:
                            out = _pack_wxy((b - 1) * g + n1_id, x, y, m, n)
                        else:
                            out = invalid
                    else:
                        out = _gqa_causal_tail_packed(
                            k, m, n, b1, b2, j, a3 - rm3 * g, t, ell, delta, g
                        )
            else:
                out = _gqa_causal_tail_packed(k, m, n, b1, b2, j, a3, t, ell, delta, g)
    return out


def band_pq(s1_outer, s2_outer, s1_token, s2_token):
    m = s1_outer
    n = s2_outer
    p = _ceil_tz(s1_token, CUBE_BASEM) + 1
    q = _ceil_tz(s2_token, CUBE_BASEN) + 1
    if p > m:
        p = m
    if q > n:
        q = n
    out_m = m
    out_n = n
    out_p = p
    out_q = q
    moff = 0
    noff = 0
    if p < 0:
        out_n = n + p
        out_p = 1
        out_q = p + q
        noff = -p
    elif q < 0:
        out_m = m + q
        out_p = p + q
        out_q = 1
        moff = -q
    return out_m, out_n, out_p, out_q, moff, noff


def _gen_band(k, m, n, p, q, b):
    b1 = b // k
    b2 = b % k
    if p + q > m:
        l1 = m - p
        l2 = p + q - m
        l3 = pl.min(m - 1, n - q)
        nseg = l1 + l2 + l3
        r1 = (p + m - 1) * l1 // 2
        r2 = m * l2
        r3 = (2 * m - 1 - l3) * l3 // 2
        rm = r1 + r2 + r3
        rmb = b1 * rm
        rm2 = m * _ceil(n * b, pl.min(k, b * m))
    else:
        l1 = q - 1
        l2 = pl.min(n - q + 1, m + 2 - p - q)
        l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
        nseg = l1 + l2 + l3
        r1 = (2 * p - 2 + q) * l1 // 2
        r2 = (p + q - 1) * l2
        r3 = (p + q - 2) * l3 - (l3 * (l3 - 1)) // 2
        rm = r1 + r2 + r3
        rmb = b1 * rm
        rm2 = _ceil(n * b, k) * (p + q - 1)
    return k, m, n, p, q, b, b1, b2, l1, l2, l3, nseg, r1, r2, r3, rm, rmb, rm2


def _gen_gqa_band(k, m, n, p, q, b, g):
    b1 = b // k
    b2 = b % k
    r1 = 0
    r2 = 0
    if p + q > m:
        l1 = m - p
        l2 = p + q - m
        l3 = pl.min(m - 1, n - q)
        n = l1 + l2 + l3
        r1 = (p + m - 1) * l1 // 2
        r2 = m * l2
    else:
        l1 = q - 1
        l2 = pl.min(n - q + 1, m + 2 - p - q)
        l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
        if l3 == 0:
            m = p + q + l2 - 2
        n = l1 + l2 + l3
    rm = m * n - (m - p) * (m - p + 1) // 2 - (n - q) * (n - q + 1) // 2
    rmb = b1 * rm * g
    if p + q > m:
        if p + q <= n:
            n1 = p + q - 1
            rm2 = m * pl.max(pl.max(_ceil(b2 * n1 * g, k), _ceil(n1, m)), g)
        else:
            rm2 = m * pl.max(pl.max(_ceil(b2 * n * g, k), _ceil(n, m)), g)
    else:
        if n > m:
            r1 = (p + q - 1) * pl.max(
                pl.max(_ceil(b2 * n * g, k), _ceil(n, p + q - 1)), g
            )
            r2 = m * pl.max(
                pl.max(_ceil(b2 * (p + q - 1) * g, k), _ceil(p + q - 1, m)), g
            )
            rm2 = pl.min(r1, r2)
        else:
            m1 = p + q - 1
            rm2 = m1 * pl.max(pl.max(_ceil(b2 * n * g, k), _ceil(n, m1)), g)
    return k, m, n, p, q, b, g, b1, b2, l1, l2, l3, rm, rmb, r1, r2, rm2


def _pos_whole_batch(m, n, p, q, a):
    out = _pack_wxy(-1, 0, 0, m, n)
    if p + q > m:
        l1 = m - p
        l2 = p + q - m
        l3 = pl.min(m - 1, n - q)
        r1 = (p + m - 1) * l1 // 2
        r2 = m * l2
        if a <= r1:
            l11 = l1 // 2 * 2
            leng = 2 * p + l11 - 1
            local_r = l11 * leng // 2
            if a <= local_r:
                y0 = _ceil(a, leng)
                r1a = _nz(a % leng, leng)
                x0 = p + y0 - r1a
                reflect = _lt(x0, 1)
                y = y0 * (1 - reflect) + (l11 + 1 - y0) * reflect
                x = x0 * (1 - reflect) + (1 - x0) * reflect
                out = _pack_wxy(0, x, y, m, n)
            else:
                out = _pack_wxy(0, a - local_r, l1, m, n)
        elif a <= r1 + r2 and m != 0:
            a2 = a - r1
            y0 = _ceil(a2, m)
            x = _nz(a2 % m, m)
            out = _pack_wxy(0, x, y0 + l1, m, n)
        else:
            a3 = a - r1 - r2
            l31 = l3 // 2 * 2
            leng = 2 * m - l31 - 1
            local_r = l31 * leng // 2
            if a3 <= local_r:
                y0 = _ceil(a3, leng)
                r1a = _nz(a3 % leng, leng)
                x0 = y0 + r1a
                reflect = _lt(m, x0)
                y1 = y0 * (1 - reflect) + (l31 + 1 - y0) * reflect
                x = x0 * (1 - reflect) + (x0 - m + y1) * reflect
                out = _pack_wxy(0, x, y1 + l1 + l2, m, n)
            else:
                out = _pack_wxy(0, a3 - local_r + l3, l3 + l1 + l2, m, n)
    else:
        l1 = q - 1
        l2 = pl.min(n - q + 1, m + 2 - p - q)
        l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
        r1 = (2 * p - 2 + q) * l1 // 2
        r2 = (p + q - 1) * l2
        if a <= r1:
            l11 = l1 // 2 * 2
            leng = 2 * p + l11 - 1
            local_r = l11 * leng // 2
            if a <= local_r:
                y0 = _ceil(a, leng)
                r1a = _nz(a % leng, leng)
                x0 = p + y0 - r1a
                reflect = _lt(x0, 1)
                y = y0 * (1 - reflect) + (l11 + 1 - y0) * reflect
                x = (p + y0 - x0) * (1 - reflect) + (1 - x0) * reflect
                out = _pack_wxy(0, x, y, m, n)
            else:
                out = _pack_wxy(0, a - local_r, l1, m, n)
        elif a <= r1 + r2:
            a2 = a - r1
            seg = p + q - 1
            y0 = _ceil(a2, seg)
            x = _nz(a2 % seg, seg) + y0 - 1
            out = _pack_wxy(0, x, y0 + l1, m, n)
        else:
            a3 = a - r1 - r2
            l31 = l3 // 2 * 2
            seg = p + q - 1
            leng = 2 * (p + q) - l31 - 3
            local_r = l31 * leng // 2
            if a3 <= local_r:
                y0 = _ceil(a3, leng)
                r1a = _nz(a3 % leng, leng)
                x0 = y0 + r1a + m - seg
                reflect = _lt(m, x0)
                y1 = y0 * (1 - reflect) + (l31 + 1 - y0) * reflect
                x = x0 * (1 - reflect) + (x0 - seg + y1) * reflect
                out = _pack_wxy(0, x, y1 + l1 + l2, m, n)
            else:
                out = _pack_wxy(0, a3 - local_r + l3 + m - seg, l3 + l1 + l2, m, n)
    return out


def _pack_band_rotated(w, x, y, b1, k, m, n):
    """Pack one full-batch band coordinate without an x/y phi merge."""
    wout = ((w - 1) % b1) * k + (w - 1) // b1 + 1
    wout = ((wout - 1) // k) * k + ((y - 1 + (wout - 1)) % k) + 1
    return _pack_wxy(wout, x, y, m, n)


def cal_band(k, m, n, p, q, b, j, r):
    _k, m0, n0, p0, q0, _b, b1, b2, l1, l2, l3, nseg, r1, r2, _r3, rm, rmb, _rm2 = (
        _gen_band(k, m, n, p, q, b)
    )
    invalid = _pack_wxy(-1, 0, 0, m, n)
    out = invalid
    if p + q > m:
        if r <= rmb:
            a = _nz(r % rm, rm)
            w = _ceil(r, rm) + b1 * (j - 1)
            if a <= r1:
                l11 = l1 // 2 * 2
                leng = 2 * p + l11 - 1
                local_r = l11 * leng // 2
                if a <= local_r:
                    y0 = _ceil(a, leng)
                    r1a = _nz(a % leng, leng)
                    x0 = p + y0 - r1a
                    reflect = _lt(x0, 1)
                    y = y0 * (1 - reflect) + (l11 + 1 - y0) * reflect
                    x = x0 * (1 - reflect) + (1 - x0) * reflect
                    out = _pack_band_rotated(w, x, y, b1, k, m, n)
                else:
                    out = _pack_band_rotated(w, a - local_r, l1, b1, k, m, n)
            elif a <= r1 + r2:
                a2 = a - r1
                y = _ceil(a2, m) + l1
                x = _nz(a2 % m, m)
                out = _pack_band_rotated(w, x, y, b1, k, m, n)
            else:
                a3 = a - r1 - r2
                l31 = l3 // 2 * 2
                leng = 2 * m - l31 - 1
                local_r = l31 * leng // 2
                if a3 <= local_r:
                    y0 = _ceil(a3, leng)
                    r1a = _nz(a3 % leng, leng)
                    x0 = y0 + r1a
                    reflect = _lt(m, x0)
                    y = (y0 * (1 - reflect) + (l31 + 1 - y0) * reflect) + l1 + l2
                    x = x0 * (1 - reflect) + (2 * m + 1 - x0) * reflect
                    out = _pack_band_rotated(w, x, y, b1, k, m, n)
                else:
                    out = _pack_band_rotated(
                        w, m - a3 + local_r + 1, l3 + l1 + l2, b1, k, m, n
                    )
        else:
            w, x, y = cal_dense(k, m, n, b2, j, r - rmb)
            if w != -1:
                if y <= m - p and x >= p + y:
                    out = invalid
                elif y > l1 + l2 and x <= y - l1 - l2:
                    out = invalid
                elif (
                    w > 0 and w < b2 + 1 and x > 0 and x < m + 1 and y > 0 and y < n + 1
                ):
                    out = _pack_wxy(b1 * k + w, x, y, m, n)
    else:
        mm = m
        if l3 == 0:
            mm = p + q + l2 - 2
        if r <= rmb:
            a = _nz(r % rm, rm)
            w = _ceil(r, rm) + b1 * (j - 1)
            if a <= r1:
                l11 = l1 // 2 * 2
                leng = 2 * p + l11 - 1
                local_r = l11 * leng // 2
                if a <= local_r:
                    y0 = _ceil(a, leng)
                    r1a = _nz(a % leng, leng)
                    x0 = p + y0 - r1a
                    reflect = _lt(x0, 1)
                    y = y0 * (1 - reflect) + (l11 + 1 - y0) * reflect
                    x = x0 * (1 - reflect) + (1 - x0) * reflect
                    out = _pack_wxy(w, x, y, m, n)
                else:
                    out = _pack_wxy(w, a - local_r, l1, m, n)
            elif a <= r1 + r2:
                a2 = a - r1
                seg = p + q - 1
                y0 = _ceil(a2, seg)
                x = _nz(a2 % seg, seg) + y0 - 1
                out = _pack_wxy(w, x, y0 + l1, m, n)
            else:
                a3 = a - r1 - r2
                l31 = l3 // 2 * 2
                leng = 2 * (p + q) - l31 - 3
                local_r = l31 * leng // 2
                if a3 <= local_r:
                    y0 = _ceil(a3, leng)
                    r1a = _nz(a3 % leng, leng)
                    x0 = y0 + r1a + 1 + mm - (p + q)
                    reflect = _lt(mm, x0)
                    y = (y0 * (1 - reflect) + (l31 + 1 - y0) * reflect) + l1 + l2
                    x = x0 * (1 - reflect) + (2 * mm + 1 - x0) * reflect
                    out = _pack_wxy(w, x, y, m, n)
                else:
                    out = _pack_wxy(w, mm - a3 + local_r + 1, l3 + l1 + l2, m, n)
        elif b2 != 0:
            a = r - rmb
            seg = p + q - 1
            a1 = _ceil(a, seg)
            a2 = _nz(a % seg, seg)
            if l3 == 0 or nseg - m < 1:
                idx = (a1 - 1) * k + j
                w = _ceil(idx, nseg)
                y = _nz(idx % nseg, nseg)
                x = y + a2 - q
                if x >= 1 and x <= m:
                    out = _pack_wxy(w + b1 * k, x, y, m, n)
            else:
                y = (a1 - 1) * k + j
                x = y + a2 - q
                if x < 1:
                    out = _pack_wxy(b, x + m, y + m, m, n)
                else:
                    w = _ceil(x, m)
                    x = _nz(x % m, m)
                    y = x + q - a2
                    if w == b2 and y > m:
                        out = invalid
                    elif y >= 1 and y <= n:
                        out = _pack_wxy(w + b1 * k, x, y, m, n)
    return out


def _pack_gqa_band_symmetric(w, x, y, j, m, n, q, l3, base_m, base_n):
    even = 1 - (j % 2)
    yflip = n + 1 - y
    near = _le(yflip, q)
    xinc = x + 1
    xnear = xinc - m * _lt(m, xinc)
    xfar = l3 + yflip + x - n
    xflip = xnear * near + xfar * (1 - near)
    xout = x * (1 - even) + xflip * even
    yout = y * (1 - even) + yflip * even
    return _pack_wxy(w, xout, yout, base_m, base_n)


def _pack_gqa_band_first_segment(w, a, l1, p, j, m, n, q, l3, base_m, base_n):
    l11 = l1 // 2 * 2
    leng = 2 * p + l11 - 1
    local_r = l11 * leng // 2
    if a <= local_r:
        y0 = _ceil(a, leng)
        r1a = _nz(a % leng, leng)
        x0 = p + y0 - r1a
        reflect = _lt(x0, 1)
        y = y0 * (1 - reflect) + (l11 + 1 - y0) * reflect
        x = (p + y0 - x0) * (1 - reflect) + (1 - x0) * reflect
        out = _pack_gqa_band_symmetric(w, x, y, j, m, n, q, l3, base_m, base_n)
    else:
        out = _pack_gqa_band_symmetric(
            w, a - local_r, l1, j, m, n, q, l3, base_m, base_n
        )
    return out


def _pack_gqa_band_symmetric_tail(
    k, m, n, p, q, l1, l3, n1, b1, b_id, j, a, r1, r2, odd, base_m, base_n
):
    wbase = b1 * k * n1 + (b_id - 1) * n1
    out = _pack_wxy(-1, 0, 0, base_m, base_n)
    if a <= r1 * n1:
        if odd == 1:
            n1_id = _ceil(a - p * n1, r1 - p)
            apos = _nz((a - p * n1) % (r1 - p), r1 - p) + p
            out = _pack_gqa_band_first_segment(
                wbase + n1_id, apos, l1, p, j, m, n, q, l3, base_m, base_n
            )
        else:
            n1_id = _ceil(a, r1)
            apos = _nz(a % r1, r1)
            out = _pack_gqa_band_first_segment(
                wbase + n1_id, apos, l1, p, j, m, n, q, l3, base_m, base_n
            )
    elif a <= (r1 + r2 // 2) * n1:
        a2 = a - r1 * n1
        r_local = r2 // 2
        n1_id = _ceil(a2, r_local)
        apos = _nz(a2 % r_local, r_local)
        y = _ceil(apos, m) + l1
        x = _nz(apos % m, m)
        out = _pack_gqa_band_symmetric(
            wbase + n1_id, x, y, j, m, n, q, l3, base_m, base_n
        )
    return out


def cal_gqa_band(k, m, n, p, q, b, g, j, r):
    base_m = m
    base_n = n
    k, m, n, p, q, b, n1, b1, b2, l1, l2, l3, rm, rm1, r1, r2, _rm2 = _gen_gqa_band(
        k, m, n, p, q, b, g
    )
    invalid = _pack_wxy(-1, 0, 0, base_m, base_n)
    rm_g = rm * n1
    if r <= rm1:
        a = _nz(r % rm_g, rm_g)
        n1_id = _ceil(a, rm)
        a = _nz(a % rm, rm)
        b_id = k * (_ceil(r, rm_g) - 1) + j
        pos = _pos_whole_batch(m, n, p, q, a)
        _w0, x, y = _unpack_wxy(pos, m, n)
        b_id = ((b_id - 1) // k) * k + ((y - 1 + (b_id - 1)) % k) + 1
        out = _pack_wxy((b_id - 1) * n1 + n1_id, x, y, base_m, base_n)
    else:
        out = invalid
        a = r - rm1
        if b2 != 0:
            if p + q > m:
                if 2 * b2 == k and l1 == l3 and m > 1 and l1 > 0:
                    b_id = (j + 1) // 2
                    wbase = b1 * k * n1 + (b_id - 1) * n1
                    if l2 % 2 == 1:
                        r0 = pl.max(2 * p, m)
                        if a <= r0 * n1:
                            n1_id = _ceil(a, r0)
                            a0 = _nz(a % r0, r0)
                            if j % 2 == 1:
                                if n >= 3:
                                    if a0 <= p:
                                        out = _pack_wxy(
                                            wbase + n1_id, a0, 1, base_m, base_n
                                        )
                                    elif p < a0 and a0 <= 2 * p:
                                        out = _pack_wxy(
                                            wbase + n1_id,
                                            l3 + a0 - p,
                                            n,
                                            base_m,
                                            base_n,
                                        )
                            else:
                                y = (n + 1) // 2
                                if a0 <= p:
                                    out = _pack_wxy(
                                        wbase + n1_id, a0 + m - p, y, base_m, base_n
                                    )
                                else:
                                    x = a0 - p
                                    if x <= m - p:
                                        out = _pack_wxy(
                                            wbase + n1_id, x, y, base_m, base_n
                                        )
                        else:
                            out = _pack_gqa_band_symmetric_tail(
                                k,
                                m,
                                n,
                                p,
                                q,
                                l1,
                                l3,
                                n1,
                                b1,
                                b_id,
                                j,
                                a - (r0 - p) * n1,
                                r1,
                                r2 - m,
                                1,
                                base_m,
                                base_n,
                            )
                    else:
                        out = _pack_gqa_band_symmetric_tail(
                            k,
                            m,
                            n,
                            p,
                            q,
                            l1,
                            l3,
                            n1,
                            b1,
                            b_id,
                            j,
                            a,
                            r1,
                            r2,
                            0,
                            base_m,
                            base_n,
                        )
                elif p + q <= n:
                    w, x, y = cal_gqa_dense(k, m, p + q - 1, b2, j, a, n1)
                    if w != -1:
                        if x - p + 1 <= y and y <= x + q - 1:
                            out = _pack_wxy(b1 * k * n1 + w, x, y, base_m, base_n)
                        elif (
                            y < x - p + 1 and 1 <= y + p + q - 1 and y + p + q - 1 <= n
                        ):
                            out = _pack_wxy(
                                b1 * k * n1 + w, x, y + p + q - 1, base_m, base_n
                            )
                else:
                    w, x, y = cal_gqa_dense(k, m, n, b2, j, a, n1)
                    if w != -1 and x - p + 1 <= y and y <= x + q - 1:
                        out = _pack_wxy(b1 * k * n1 + w, x, y, base_m, base_n)
            elif n > m:
                if r2 <= r1:
                    w, x, y = cal_gqa_dense(k, m, p + q - 1, b2, j, a, n1)
                    if w != -1:
                        if y - q + 1 <= x and x <= p + y - 1:
                            out = _pack_wxy(b1 * k * n1 + w, x, y, base_m, base_n)
                        else:
                            y2 = _ceil(x - (p + y - 1), p + q - 1) * (p + q - 1) + y
                            if 1 <= y2 and y2 <= n:
                                out = _pack_wxy(b1 * k * n1 + w, x, y2, base_m, base_n)
                else:
                    w, x, y = cal_gqa_dense_no_tune(k, p + q - 1, n, b2, j, a, n1)
                    if w != -1:
                        x2 = x + y - q
                        if 1 <= x2 and x2 <= m:
                            out = _pack_wxy(b1 * k * n1 + w, x2, y, base_m, base_n)
            else:
                w, x, y = cal_gqa_dense(k, p + q - 1, n, b2, j, a, n1)
                if w != -1:
                    if x - p + 1 <= y and y <= x + q - 1:
                        out = _pack_wxy(b1 * k * n1 + w, x, y, base_m, base_n)
                    else:
                        x2 = _ceil(y - (q + x - 1), p + q - 1) * (p + q - 1) + x
                        if 1 <= x2 and x2 <= m:
                            out = _pack_wxy(b1 * k * n1 + w, x2, y, base_m, base_n)
    return out


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
        mha_round = _ceil(n * bb, pl.min(k, m * bb)) * m
        gqa_round = select_gqa_dense_round(k, m, n, bb, g) * m
        is_mha = _in_range(g, 1, 1)
        out = mha_round * is_mha + gqa_round * (1 - is_mha)
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
    if k > 0 and m > 0 and n > 0 and bb > 0:
        dense_r = _dense_lin_max_round(k, m, n, bb, g)
        if mask_mode == 0:
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
        ):
            offset = (constInfo.s1Size - constInfo.s2Size) // CUBE_BASEM
            first = pl.astype(offset * g, pl.DT_INT64)
            pairs = ((n + 1) // 2) * bb
            remaining = pairs % coreNum
            if remaining == 0 or remaining > coreNum // 2:
                end = pl.astype(end - offset * g, pl.DT_INT64)
    return first, end


def _fused_valid(constInfo, s1o, s2o):
    out = 1
    if mask_mode == 0:
        out = 1
    else:
        s2_left = s2o * CUBE_BASEN
        if mask_mode == 3:
            s2_ignored = constInfo.s1Size - CUBE_BASEM * (s1o + 1)
            s2_end = pl.min(pl.max(constInfo.s2Size - s2_ignored, 0), constInfo.s2Size)
            out = pl.min(pl.max(s2_end - s2_left, 0), 1)
        else:
            s2_right = pl.min((s2o + 1) * CUBE_BASEN, constInfo.s2Size)
            s2_sl = pl.max(CUBE_BASEM * s1o - constInfo.s1Token, 0)
            s2_sl = (s2_sl // 64) * 64
            row_end = pl.min(CUBE_BASEM * (s1o + 1), constInfo.s1Size)
            s2_sr = ((row_end + constInfo.s2Token + 63) // 64) * 64
            s2_sr = pl.min(s2_sr, constInfo.s2Size)
            a = pl.min(pl.max(s2_sr - s2_left, 0), 1)
            b = pl.min(pl.max(s2_right - s2_sl, 0), 1)
            out = a * b
    return out


def pack_fused(w, x, y, b, n2, g, s1_outer, s2_outer, moff, noff):
    n1 = pl.max(g * n2, 1)
    w_safe = pl.max(w, 1)
    batch = _ceil(w_safe, n1) - 1
    n1_idx = w_safe - batch * n1 - 1
    gg = pl.max(g, 1)
    n2_idx = n1_idx // gg
    g_idx = n1_idx % gg
    s1_idx = x - 1 + moff
    s2_idx = y - 1 + noff
    ok = (
        _in_range(w, 1, w)
        * _in_range(batch, 0, b - 1)
        * _in_range(n2_idx, 0, n2 - 1)
        * _in_range(g_idx, 0, g - 1)
        * _in_range(s1_idx, 0, s1_outer - 1)
        * _in_range(s2_idx, 0, s2_outer - 1)
    )
    fused = (
        ((batch * n2 + n2_idx) * g + g_idx) * s1_outer * s2_outer
        + s2_idx * s1_outer
        + s1_idx
    )
    return fused * ok + (-1) * (1 - ok)


def _deter_lin_gidx_host(constInfo, cBlockIdx, coreNum, roundId):
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
    if mask_mode == 0:
        if g <= 1:
            w, x, y = cal_dense(k, s1o, s2o, bb, j, r)
        else:
            dense_round = select_gqa_dense_round(k, s1o, s2o, bb, g)
            w, x, y = cal_gqa_dense(k, s1o, s2o, bb, j, r, g, dense_round)
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
    if _fused_valid(constInfo, s1_idx, s2_idx) == 0:
        return -1
    return fused


def deter_lin_gidx(constInfo, cBlockIdx, coreNum, roundId):
    """(cBlockIdx, roundId) 0-based → fusedIdx，空槽 -1。

    ``mask_mode`` 是 tiling key，dense/causal/band 分支会在编译时折叠。
    设备路径必须直接使用对应的 AscendC 闭式，不能用 dense 后置过滤代替。
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
    if mask_mode == 0:
        w, x, y = cal_dense(k, s1o, s2o, bb, j, r)
        dense_round = select_gqa_dense_round(k, s1o, s2o, bb, g)
        wg, xg, yg = cal_gqa_dense(k, s1o, s2o, bb, j, r, g, dense_round)
    elif mask_mode == 3:
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
    hit = pl.min(pl.max(fused + 1, 0), 1) * _fused_valid(constInfo, s1_idx, s2_idx)
    return pl.astype(fused, pl.DT_INT64) * hit + pl.astype(-1, pl.DT_INT64) * (1 - hit)


_deter_lin_gidx_device = deter_lin_gidx


if type(pl).__name__ == "_Pl":
    deter_lin_gidx = _deter_lin_gidx_host
