# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mask 3/4 过门分核：闭式列几何 + 定长二分反解 + HeadSliceSplit（L2 窗口）。

本模块不 import fag_common，避免与 set_const_info 形成环。
"""

import pypto_pro.language as pl

L2_CACHE_BYTES = 134217728  # 128MB，与 host info.l2CacheSize 在 9599 上同值

# s2Outer <= 1024（FAG_MAX_S2_OUTER），ceil(log2(1024)) = 10
COL_BISECT_STEPS = 10
# 轮数上限。localIdx 映射要一个 trace 期定长循环，故必须有界。
# 实测代表性 shape 的 K 恒为 1~2，超界只是 L2 略差，不影响正确性。
COL_MAX_ROUNDS = 8


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


# 邻域窗口：packed 列优先时，瘦头约覆盖 T=4K token 的 K → B≈(T/128)^2/2=512。
# 步长 ≈ B/36 ≈ 14，避免 B=V 时落到 229/913。
B_NEIGHBOR = 512
STRIDE_MIN = 8


def col_wave_tiles(constInfo, coreNum):
    """窗口 B = min(V, max(C*STRIDE_MIN, B_NEIGHBOR))。"""
    v = constInfo.packedPerHead
    b_min = pl.max(coreNum, coreNum * STRIDE_MIN)
    return pl.min(v, pl.max(b_min, B_NEIGHBOR))


def col_core_count(constInfo, cBlockIdx, coreNum):
    """本核块数 = tiles_per_head * heads。末 wave 按实际剩余，无空槽。"""
    v = constInfo.packedPerHead
    win = col_wave_tiles(constInfo, coreNum)
    start = col_part_lo(win, coreNum, cBlockIdx)
    count_c = col_part_lo(win, coreNum, cBlockIdx + 1) - start
    waves = (v + win - 1) // pl.max(win, 1)
    last_len = v - (waves - 1) * win
    last_count = pl.max(0, pl.min(count_c, last_len - start))
    tiles = (waves - 1) * count_c + last_count
    heads = constInfo.bSize * constInfo.n2Size * constInfo.gSize
    return tiles * heads


def col_local_to_global(constInfo, cBlockIdx, coreNum, localIdx):
    """HeadSliceSplit：先做完本 head 的所有 wave，再换下一个 head。"""
    v = constInfo.packedPerHead
    win = col_wave_tiles(constInfo, coreNum)
    start = col_part_lo(win, coreNum, cBlockIdx)
    count_c = col_part_lo(win, coreNum, cBlockIdx + 1) - start
    waves = (v + win - 1) // pl.max(win, 1)
    last_len = v - (waves - 1) * win
    last_count = pl.max(0, pl.min(count_c, last_len - start))
    tiles = (waves - 1) * count_c + last_count
    safe = pl.max(tiles, 1)
    head = localIdx // safe
    inner = localIdx % safe
    full = (waves - 1) * count_c
    in_full = pl.min(pl.max(full - inner, 0), 1)
    wave = in_full * (inner // pl.max(count_c, 1)) + (1 - in_full) * (waves - 1)
    off = in_full * (inner % pl.max(count_c, 1)) + (1 - in_full) * (inner - full)
    t = wave * win + start + off
    return head * v + t


def col_invert(constInfo, t):
    """head 内 packed 下标 t -> (s1o, s2o)。定长二分，自动跳过零高列。

    判据必须是「最小的 c 使 P(c+1) > t」。零高列满足 P(c) == P(c+1)，该判据
    自动跳过；若写成「P(c) <= t 的最大 c」会停在零高列上并算出越界 s1o。
    二分只会问到 c = mid+1 <= N-1，故这里用不上末列特例。
    """
    m = constInfo.s1Outer
    p = constInfo.colP
    q = constInfo.colQ
    lo = 0
    hi = constInfo.s2Outer - 1
    for _ in pl.range(0, COL_BISECT_STEPS):
        mid = (lo + hi) // 2
        gt = pl.min(pl.max(col_prefix_body(m, p, q, mid + 1) - t, 0), 1)
        act = pl.min(pl.max(hi - lo, 0), 1)
        hi = hi - act * gt * (hi - mid)
        lo = lo + act * (1 - gt) * (mid + 1 - lo)
    s2o = lo
    s1o = pl.max(s2o - q, 0) + (t - col_prefix_body(m, p, q, s2o))
    return s1o, s2o


def col_packed_to_dense(constInfo, gidx):
    """全局 packed 下标 -> dense 全局块下标（与 swizzle_idx 同口径）。"""
    head = gidx // constInfo.packedPerHead
    t = gidx % constInfo.packedPerHead
    s1o, s2o = col_invert(constInfo, t)
    return head * constInfo.s1oS2o + s2o * constInfo.s1Outer + s1o
