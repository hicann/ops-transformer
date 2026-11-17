# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ASC non-deterministic swizzle: valid-block groups and large-grid mappings.

本模块不 import fag_common，避免与 set_const_info 形成环。
"""

import pypto_pro.language as pl


def ceil_div_signed(x, y):
    """ceil(x/y)，y > 0，x 可正可负。

    右下对齐校正后 s1Token/s2Token 可以是负的（`s1Token = winLeft + s1 - s2`，
    winLeft >= 0 也能为负）。负半边不能写 `min(x,0)//y`：0909 的 `//` 对负数
    向下取整（floor），`(-1)//128 == -1`，而数学 ceil(-1/128) 是 0。向零截断
    的 C `/` 碰巧对负输入等于 ceil，本地 CInt 单测因此测不出来。

    只对非负数做 `//`（floor 与 trunc 同值），负输入改成 `-floor((-x)/y)`：
    ceil(x/y) = ceil(pos/y) - floor(neg/y)，pos=max(x,0)，neg=max(-x,0)。
    """
    pos = pl.max(x, 0)
    neg = pl.max(-x, 0)
    return (pos + y - 1) // y - neg // y


def _ramp_desc(lo, hi, a):
    """sum_{j=lo}^{hi} max(a - j, 0)。空区间（hi < lo）返回 0。"""
    top = pl.max(pl.min(hi, a), lo - 1)
    x = a - top
    y = a - lo
    return (x + y) * (y - x + 1) // 2


def _ramp_asc(lo, hi, q):
    """sum_{j=lo}^{hi} max(j - q, 0)。空区间（hi < lo）返回 0。"""
    bot = pl.min(pl.max(lo, q), hi + 1)
    x = bot - q
    y = hi - q
    return (x + y) * (y - x + 1) // 2


def col_prefix_body(m, p, q, c):
    """sum_{j<c} h(j)，h(j) = max(min(j+P, M-1) - max(j-Q, 0) + 1, 0)。O(1)。

    统一梯形，两种 mask 共用（mask3 取 P = M-1 即上界恒为 M-1）。
    **末列不能走这里**：它的上界被 s2 截断，要单独算，见 packedPerHead。
    调用方保证 c <= N-1。
    """
    # Cast before products, not merely on return: large grids exceed INT32.
    m = pl.astype(m, pl.DT_INT64)
    p = pl.astype(p, pl.DT_INT64)
    q = pl.astype(q, pl.DT_INT64)
    c = pl.astype(c, pl.DT_INT64)
    live = pl.min(pl.max(p + q + 1, 0), 1)
    lo = pl.max(-p, 0)
    hi = pl.min(m + q - 1, c - 1)
    cnt = pl.max(hi - lo + 1, 0)
    hi = lo + cnt - 1
    term1 = m * cnt - _ramp_desc(lo, hi, m - p - 1)
    term2 = _ramp_asc(lo, hi, q)
    return (term1 - term2) * live


def col_part_lo(total, nPart, idx):
    """把 [0,total) 均分成 nPart 份时第 idx 份的起点，即 total*idx//nPart。

    不能直接写 `total * idx // nPart`：total 可达 5e8、idx 达 288，乘积溢出
    int32。拆成 (total//nPart)*idx + ((total%nPart)*idx)//nPart。
    """
    return (total // nPart) * idx + ((total % nPart) * idx) // nPart


def col_total_blocks(constInfo):
    return (
        constInfo.packedPerHead * constInfo.bSize * constInfo.n2Size * constInfo.gSize
    )


def col_core_count(constInfo, cBlockIdx, coreNum):
    """ASC GetNextValidIdxForSwizzle: cyclic groups across all fused heads."""
    total = pl.astype(col_total_blocks(constInfo), pl.DT_INT64)
    contiguous = constInfo.normalContinuousBlocks
    stride = coreNum * contiguous
    return (total // stride) * contiguous + pl.min(
        pl.max(total % stride - cBlockIdx * contiguous, 0), contiguous
    )


def col_local_to_global(constInfo, cBlockIdx, coreNum, localIdx):
    """Do not reset the group at a head boundary (ASC continuousBlockNum)."""
    localIdx = pl.astype(localIdx, pl.DT_INT64)
    contiguous = constInfo.normalContinuousBlocks
    return (
        localIdx // contiguous * coreNum + cBlockIdx
    ) * contiguous + localIdx % contiguous


def col_invert(constInfo, t):
    """head 内 packed 下标 t -> (s1o, s2o)。收敛二分，自动跳过零高列。

    判据必须是「最小的 c 使 P(c+1) > t」。零高列满足 P(c) == P(c+1)，该判据
    自动跳过；若写成「P(c) <= t 的最大 c」会停在零高列上并算出越界 s1o。
    二分只会问到 c = mid+1 <= N-1，故这里用不上末列特例。
    """
    m = constInfo.s1Outer
    p = constInfo.colP
    q = constInfo.colQ
    t = pl.astype(t, pl.DT_INT64)
    lo = pl.astype(0, pl.DT_INT64)
    hi = pl.astype(constInfo.s2Outer - 1, pl.DT_INT64)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if col_prefix_body(m, p, q, mid + 1) > t:
            hi = mid
        else:
            lo = mid + 1
    s2o = lo
    s1o = pl.max(s2o - q, 0) + (t - col_prefix_body(m, p, q, s2o))
    return s1o, s2o


def col_packed_to_dense(constInfo, gidx):
    """全局 packed 下标 -> dense 全局块下标（与 swizzle_idx 同口径）。"""
    head = gidx // constInfo.packedPerHead
    t = gidx % constInfo.packedPerHead
    s1o, s2o = col_invert(constInfo, t)
    return head * constInfo.s1oS2o + s2o * constInfo.s1Outer + s1o


def causal_tail_prefix(m, residue, coreNum, columns):
    """Blocks in the first `columns` cyclic columns of the odd final head."""
    columns = pl.astype(columns, pl.DT_INT64)
    return columns * (m - residue) - coreNum * columns * (columns - 1) // 2


def causal_core_count(constInfo, cBlockIdx, coreNum):
    """ASC paired triangular heads; an odd final head owns whole columns."""
    m = constInfo.s1Outer
    heads = constInfo.bSize * constInfo.n2Size * constInfo.gSize
    pairColumns = (heads // 2) * (m + 1)
    ownedPairs = (pl.max(pairColumns - cBlockIdx, 0) + coreNum - 1) // coreNum
    residue = (cBlockIdx - pairColumns % coreNum + coreNum) % coreNum
    tailColumns = (pl.max(m - residue, 0) + coreNum - 1) // coreNum
    return ownedPairs * m + (heads % 2) * causal_tail_prefix(
        m, residue, coreNum, tailColumns
    )


def causal_swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx):
    """GetNextValidIdxForSwizzleCasual, with exact integer odd-head inversion."""
    m = constInfo.s1Outer
    heads = constInfo.bSize * constInfo.n2Size * constInfo.gSize
    pairColumns = (heads // 2) * (m + 1)
    pairTasks = ((pl.max(pairColumns - cBlockIdx, 0) + coreNum - 1) // coreNum) * m
    if localIdx < pairTasks:
        column = localIdx // m * coreNum + cBlockIdx
        pair = column // (m + 1)
        vc = column % (m + 1)
        vr = (vc + localIdx % m) % m
        if vc <= vr:
            head = pair * 2
            row = vr
            col = vc
        else:
            head = pair * 2 + 1
            row = m - 1 - vr
            col = m - vc
        return head * constInfo.s1oS2o + col * m + row
    tail = pl.astype(localIdx - pairTasks, pl.DT_INT64)
    residue = (cBlockIdx - pairColumns % coreNum + coreNum) % coreNum
    columns = (pl.max(m - residue, 0) + coreNum - 1) // coreNum
    lo = pl.astype(0, pl.DT_INT64)
    hi = pl.astype(columns - 1, pl.DT_INT64)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if causal_tail_prefix(m, residue, coreNum, mid + 1) > tail:
            hi = mid
        else:
            lo = mid + 1
    col = residue + lo * coreNum
    row = col + tail - causal_tail_prefix(m, residue, coreNum, lo)
    return (heads - 1) * constInfo.s1oS2o + col * m + row
