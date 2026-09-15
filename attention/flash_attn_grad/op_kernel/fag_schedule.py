# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""分核与 RunInfo。

职责是把「本核该干哪些块」翻译成逐 task 的 :class:`FagRunInfo`：

1. **块级有效性**：``is_block_valid_i`` / ``skip_invalid``；
2. **分核映射**：线性支读 AICPU metadata，swizzle 支按列/按 packed 有效块
   跨核发放；
3. **逐 task 展开**：``set_run_info`` 把全局块下标拆成 (b, n2, g, s2o, s1o)
   与各张量的 GM 偏移，并算好 K/V 复用与 dK/dV flush 的开关。

cube 与 vector 两侧调用**同一套**函数、读**同一块** metadata —— 这是两侧
task 数一致的前提，也是 set/wait 能配对的前提。任何只改一侧的分核逻辑都会
直接死锁。
"""

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_common import CUBE_BASEM, CUBE_BASEN, VECTOR_BASEM
# --- end generated imports ---

# ==================================================================
#  块级有效性与「下一块」
# ==================================================================


def is_block_valid_i(constInfo, index):
    """块级 IsValid。mask_mode 是 tilingkey，只编一侧公式。

    3：CheckIsValidBlock（s2Left < s2EndLen）。
    4：band 区间相交。不要用 sparseType 抢因果分支。

    比较做成 0/1 算术：此函数会内联进 skip_invalid 的 while，运行期
    if/return 会在 while 里多出一个 IfStmt Yield，和 idx 的 loop-carried
    拼成 1 vs 2（0902 `Loop-carried write-back`）。
    """
    if mask_mode == 0:
        return 1
    gDimTail = index % constInfo.s1oS2o
    s2o = gDimTail // constInfo.s1Outer
    s1o = gDimTail % constInfo.s1Outer
    s2Left = s2o * CUBE_BASEN
    if mask_mode == 3:
        s2Ignored = constInfo.s1Size - CUBE_BASEM * (s1o + 1)
        s2EndLen = pl.min(pl.max(constInfo.s2Size - s2Ignored, 0), constInfo.s2Size)
        return pl.min(pl.max(s2EndLen - s2Left, 0), 1)
    s2Right = pl.min((s2o + 1) * CUBE_BASEN, constInfo.s2Size)
    s2SparseLeft = pl.max(CUBE_BASEM * s1o - constInfo.s1Token, 0)
    s2SparseLeft = (s2SparseLeft // 64) * 64
    rowEnd = pl.min(CUBE_BASEM * (s1o + 1), constInfo.s1Size)
    s2SparseRight = ((rowEnd + constInfo.s2Token + 63) // 64) * 64
    s2SparseRight = pl.min(s2SparseRight, constInfo.s2Size)
    a = pl.min(pl.max(s2SparseRight - s2Left, 0), 1)
    b = pl.min(pl.max(s2Right - s2SparseLeft, 0), 1)
    return a * b


def skip_invalid(constInfo, idx, blockEnd):
    """线性支：while 无效则 idx++，越界 -1。

    0902：while 里不要 `return`，也不要把 runtime if/phi 当循环最后一句。
    条件只用标量 `v==0`；越界折进 v，循环体只更新 idx 和 v。
    """
    v = is_block_valid_i(constInfo, idx)
    over = pl.min(pl.max(idx + 1 - blockEnd, 0), 1)
    v = pl.min(v + over, 1)
    while v == 0:
        idx = idx + 1
        cur = is_block_valid_i(constInfo, idx)
        over = pl.min(pl.max(idx + 1 - blockEnd, 0), 1)
        v = pl.min(cur + over, 1)
    if idx >= blockEnd:
        return pl.astype(-1, pl.DT_INT64)
    return pl.astype(idx, pl.DT_INT64)


def next_work_gidx(
    constInfo,
    cBlockIdx,
    coreNum,
    blockStart,
    blockEnd,
    localIdx,
    denseCursor,
    numBlocks,
):
    """下一块全局下标；本核没有活块时 -1。线性走 skip_invalid，swizzle 走公式。"""
    if swizzle == 0:
        return skip_invalid(constInfo, denseCursor, blockEnd)
    if localIdx >= numBlocks:
        return pl.astype(-1, pl.DT_INT64)
    return block_idx_of(constInfo, cBlockIdx, coreNum, blockStart, localIdx)


# ==================================================================
#  Swizzle 分核 —— 让同时在跑的核落在同一 head 内，命中 L2
#  判据在 host（op_host/plan/flash_attn_grad_tiling_swizzle.cpp），
#  这里只负责「已经决定开 swizzle」之后的下标映射。
# ==================================================================


def swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx):
    """把本核的局部块序号映射成全局块下标。

    局部序号 localIdx 先拆成「第几列 k」与「列内第几个 s1 块」，
    列号再按跨步还原成全局列 t = cBlockIdx + k * coreNum，
    最后由 t 解出 (head, s2Idx) 拼回全局下标。
    """
    k = localIdx // constInfo.s1Outer
    s1Idx = localIdx % constInfo.s1Outer
    t = cBlockIdx + k * coreNum
    head = t // constInfo.s2Outer
    s2Idx = t % constInfo.s2Outer
    return head * constInfo.s1oS2o + s2Idx * constInfo.s1Outer + s1Idx


# Packed 反函数按列扫描的上限。S=8192 -> s2Outer=64；本轮不做 S>32768。
PACKED_SCAN_MAX = 128


def packed_g_to_s1s2(constInfo, gDimTail):
    """packed 有效块下标 -> (s1o, s2o)。按列扫描，与 kernel 既有 packed 同序。

    暂不改成求根公式，按列扫描即可。
    mask_mode 静态分支只编一侧长度公式。
    """
    acc = 0
    s1Idx = 0
    s2Idx = 0
    found = 0
    m = constInfo.s1Outer
    n = constInfo.s2Outer
    for s2 in pl.range(0, PACKED_SCAN_MAX):
        if s2 < n:
            active = 1
        else:
            active = 0
        if mask_mode == 3:
            need = s2 * CUBE_BASEN + 1
            rhs = need - (constInfo.s2Size - constInfo.s1Size)
            if rhs < 1:
                s1Min = 0
            else:
                s1Min = pl.max((rhs + CUBE_BASEM - 1) // CUBE_BASEM - 1, 0)
            length = pl.max(m - s1Min, 0) * active
            s1Base = s1Min
        elif mask_mode == 4:
            xMin = pl.max(s2 - constInfo.bandQ, 0)
            xMax = pl.min(m - 1, s2 + constInfo.bandP)
            length = pl.max(xMax - xMin + 1, 0) * active
            s1Base = xMin
        else:
            length = m * active
            s1Base = 0
        if found == 0:
            if gDimTail >= acc:
                if gDimTail < acc + length:
                    s2Idx = s2
                    s1Idx = s1Base + (gDimTail - acc)
                    found = 1
        acc = acc + length
    return s1Idx, s2Idx


def packed_to_dense_idx(constInfo, packedIdx):
    total = constInfo.totalPerBatchNum
    if total < 1:
        total = constInfo.s1oS2o
    bIdx = packedIdx // total
    gDimTail = packedIdx % total
    s1Idx, s2Idx = packed_g_to_s1s2(constInfo, gDimTail)
    return bIdx * constInfo.s1oS2o + s2Idx * constInfo.s1Outer + s1Idx


# ==================================================================
#  AICPU metadata 布局 —— 与 flash_attn_grad_metadata_split.h 对齐
# ==================================================================

META_HEAD_FAG_START_INDEX = 7
META_FAG_BLOCK_STARTS_OFFSET = 8
META_FAG_BLOCK_ENDS_OFFSET = 44
META_FAG_PRE_AIV_NUM = 121
META_FAG_DQ_ROWS = 122
META_FAG_DKV_ROWS = 123
META_FAG_DQ_ROW_STARTS = 124
META_FAG_DQ_ROW_ENDS = 196
META_FAG_DKV_ROW_STARTS = 268
META_FAG_DKV_ROW_ENDS = 340
META_FAG_AIV_LIST_NUM = 72
META_VIEW_LEN = 8192


def block_range_of(constInfo, cBlockIdx, coreNum, blockStart, blockEnd):
    """本核负责的块数。线性支读 metadata；swizzle 仍走列跨核公式。

    mask+swizzle 按 packed 有效块跨核发，不再用整矩形列跨核。
    """
    numColTasks = (
        constInfo.bSize * constInfo.n2Size * constInfo.gSize * constInfo.s2Outer
    )
    rem = pl.max(numColTasks - cBlockIdx, 0)
    swzDense = ((rem + coreNum - 1) // coreNum) * constInfo.s1Outer
    fused = constInfo.bSize * constInfo.n2Size * constInfo.gSize
    tot = fused * constInfo.totalPerBatchNum
    remP = pl.max(tot - cBlockIdx, 0)
    swzPacked = (remP + coreNum - 1) // coreNum
    lin = pl.max(blockEnd - blockStart, 0)
    if constInfo.enableSwizzle:
        if mask_mode == 0:
            return pl.astype(swzDense, pl.DT_INT64)
        return pl.astype(swzPacked, pl.DT_INT64)
    return lin


def block_idx_of(constInfo, cBlockIdx, coreNum, blockStart, localIdx):
    """局部块序号 -> 全局块下标。线性支起点取自 metadata。"""
    swzDense = swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx)
    swzPacked = packed_to_dense_idx(constInfo, cBlockIdx + localIdx * coreNum)
    lin = blockStart + localIdx
    if constInfo.enableSwizzle:
        if mask_mode == 0:
            return pl.astype(swzDense, pl.DT_INT64)
        return pl.astype(swzPacked, pl.DT_INT64)
    return lin


# ==================================================================
#  K/V 复用与 dK/dV L0C 驻留的开关
# ==================================================================


def s2_group_of(constInfo, idx):
    """(b, n2, g, s2o) 的组合编号：同一编号内 dK/dV 可在 L0C 上持续累加。

    块网格是 b -> n2 -> g -> s2o -> s1o，一组恰为「连续 s1Outer 个块」，
    故组号就是 idx // s1Outer —— 已融合 (b,n2,g,s2o) 四维，无需再逐维还原。
    GQA 下跨 g 即换组(g 在 s2o 外侧)，dK/dV 会先 flush 再重新累加；G 个 Q
    head 对同一 KV head 的贡献靠 GM 上的 atomicAdd 合并。
    """
    return idx // constInfo.s1Outer


def need_load_kv(constInfo, gIdx, localIdx):
    """是否需要重新从 GM 载入 K/V(1=需要，0=沿用 L1 里的)。

    只有换组那一块要重载，组内其余块 K/V 相同。dkvAcc = 1 - loadKV，
    不必再算一遍。
    8192 场景 s1Outer=64，K/V 的 GM 读次数降到 1/64。
    """
    swz = 1 - pl.min(localIdx % constInfo.s1Outer, 1)
    isFirst = 1 - pl.min(localIdx, 1)
    gCur = s2_group_of(constInfo, gIdx)
    gPrev = s2_group_of(constInfo, pl.max(gIdx - 1, 0))
    lin = pl.max(isFirst, pl.min(gCur - gPrev, 1))
    if constInfo.enableSwizzle:
        return pl.astype(swz, pl.DT_INT32)
    return pl.astype(lin, pl.DT_INT32)


def next_s2_same(constInfo, gIdx, localIdx):
    """下一块是否仍落在同一个 s2 块内(1=是，0=否)。"""
    swz = pl.min((localIdx + 1) % constInfo.s1Outer, 1)
    lin = 1 - pl.min(s2_group_of(constInfo, gIdx + 1) - s2_group_of(constInfo, gIdx), 1)
    if constInfo.enableSwizzle:
        return pl.astype(swz, pl.DT_INT32)
    return pl.astype(lin, pl.DT_INT32)


# ==================================================================
#  逐 task 的 RunInfo
# ==================================================================


def set_run_info(constInfo, runInfo, taskId, index, subIdx, localIdx):
    # index -> (boIdx, n2oIdx, goIdx, s2oIdx, s1oIdx)，S1 为最快轴。
    # 维度顺序 b -> n2 -> g -> s2o -> s1o（g 在 n2 之内、s2o 之外）。
    bDimTail = index % constInfo.n2GS1oS2o
    n2DimTail = bDimTail % constInfo.gS1oS2o
    gDimTail = index % constInfo.s1oS2o
    runInfo.boIdx = index // constInfo.n2GS1oS2o
    runInfo.n2oIdx = bDimTail // constInfo.gS1oS2o
    runInfo.goIdx = n2DimTail // constInfo.s1oS2o
    runInfo.s2oIdx = gDimTail // constInfo.s1Outer
    runInfo.s1oIdx = gDimTail % constInfo.s1Outer

    # 尾块真实大小。
    s1Real = pl.min(CUBE_BASEM, constInfo.s1Size - runInfo.s1oIdx * CUBE_BASEM)
    s2Real = pl.min(CUBE_BASEN, constInfo.s2Size - runInfo.s2oIdx * CUBE_BASEN)
    runInfo.s1RealSize = s1Real
    runInfo.s2RealSize = s2Real

    # vector 侧两个子核各承担半个 S1：
    # subIdx=0 取 firstHalf，subIdx=1 取剩余部分
    # 两个 vector 子核的分界固定为 VECTOR_BASEM(64)，不能按 s1Real 折半：
    # mm1/mm2 结果由 fixpipe 以 DualModeSplitM 写出，硬件在第 64 行处切分，
    # 子核 0 恒拿块内 0..63 行、子核 1 恒拿 64..127 行。若分界跟着尾块变小，
    # UB 里的数据与 VF 的行号就错位(实测尾块场景 dQ/dK/dV 全错)。
    # 固定 64 同时天然满足 dS^T/P^T 的 NZ 分形 16 对齐要求。
    # 本子核真实行数：尾块可能不足，甚至为 0(s1Real <= 64 时子核 1 无事可做)
    runInfo.halfS1RealSize = pl.max(
        pl.min(s1Real - subIdx * VECTOR_BASEM, VECTOR_BASEM), 0
    )

    # BSND 布局的 GM 偏移
    runInfo.queryOffset = runInfo.s1oIdx * CUBE_BASEM
    runInfo.keyOffset = runInfo.s2oIdx * CUBE_BASEN

    # 统一四维视图上的两个非 S 轴索引，纯算术无分支。
    # Q 侧的 head 是 n2oIdx*gSize + goIdx（共 n1 个），KV 侧是 n2oIdx（共 n2 个）
    # Q 偏移含 g，KV 偏移不含。
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    runInfo.idx0Q = runInfo.boIdx * constInfo.coefB0Q + headQ * constInfo.coefN0
    runInfo.idx2Q = headQ * constInfo.coefN2
    runInfo.idx0KV = (
        runInfo.boIdx * constInfo.coefB0KV + runInfo.n2oIdx * constInfo.coefN0
    )
    runInfo.idx2KV = runInfo.n2oIdx * constInfo.coefN2
    rowBase = runInfo.queryOffset + subIdx * VECTOR_BASEM
    runInfo.loadBase = pl.max(pl.min(rowBase, constInfo.s1Size - VECTOR_BASEM), 0)
    runInfo.rowShift = rowBase - runInfo.loadBase

    runInfo.loadKV = need_load_kv(constInfo, index, localIdx)
    # 与 loadKV 互补：列首覆写 L0C，列中累加。消费点必须 == 1，不要用 == 0。
    runInfo.dkvAcc = 1 - runInfo.loadKV
    runInfo.flushDkv = 1 - next_s2_same(constInfo, index, localIdx)
    runInfo.gIdx = index
    runInfo.dqFirst = 1
    runInfo.dqLast = 1
    if is_bn2_multiblk == 1:
        check_s1_range_in_bn2(constInfo, runInfo)


def check_s1_range_in_bn2(constInfo, runInfo):
    """本 s1 列的首/末有效 s2 块。"""
    s2o = runInfo.s2oIdx
    s1o = runInfo.s1oIdx
    nextLeft = (s2o + 1) * CUBE_BASEN
    nextRight = (s2o + 2) * CUBE_BASEN
    if mask_mode == 0:
        if s2o == constInfo.s2Outer - 1:
            runInfo.dqLast = 1
        else:
            runInfo.dqLast = 0
        if s2o == 0:
            runInfo.dqFirst = 1
        else:
            runInfo.dqFirst = 0
        return
    if mask_mode == 3:
        s2Ignored = constInfo.s1Size - CUBE_BASEM * (s1o + 1)
        s2EndLen = pl.min(pl.max(constInfo.s2Size - s2Ignored, 0), constInfo.s2Size)
        if nextLeft < s2EndLen:
            runInfo.dqLast = 0
        else:
            runInfo.dqLast = 1
        if s2o == 0:
            runInfo.dqFirst = 1
        else:
            runInfo.dqFirst = 0
        return
    s2SparseLeft = pl.max(CUBE_BASEM * s1o - constInfo.s1Token, 0)
    s2SparseLeft = (s2SparseLeft // 64) * 64
    rowEnd = pl.min(CUBE_BASEM * (s1o + 1), constInfo.s1Size)
    s2SparseRight = ((rowEnd + constInfo.s2Token + 63) // 64) * 64
    s2SparseRight = pl.min(s2SparseRight, constInfo.s2Size)
    if nextLeft < s2SparseRight:
        if s2SparseLeft < nextRight:
            runInfo.dqLast = 0
        else:
            runInfo.dqLast = 1
    else:
        runInfo.dqLast = 1
    if s2o > 0:
        preLeft = (s2o - 1) * CUBE_BASEN
        preRight = s2o * CUBE_BASEN
        if preLeft < s2SparseRight:
            if s2SparseLeft < preRight:
                runInfo.dqFirst = 0
            else:
                runInfo.dqFirst = 1
        else:
            runInfo.dqFirst = 1
    else:
        runInfo.dqFirst = 1


def apply_s2_idx_no_change(runInfos, curInfo, taskId, ring):
    """与上一有效块比 (s2o, b, n2)，不含 g。"""
    prevR = runInfos[(taskId + ring - 1) % ring]
    if taskId == 0:
        curInfo.loadKV = 1
        curInfo.dkvAcc = 0
        curInfo.flushDkv = 1
        return
    if (
        prevR.s2oIdx == curInfo.s2oIdx
        and prevR.boIdx == curInfo.boIdx
        and prevR.n2oIdx == curInfo.n2oIdx
    ):
        curInfo.loadKV = 0
        curInfo.dkvAcc = 1
        prevR.flushDkv = 0
    else:
        curInfo.loadKV = 1
        curInfo.dkvAcc = 0
        prevR.flushDkv = 1
    curInfo.flushDkv = 1
