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
2. **分核映射**：线性支读 AICPU metadata；swizzle 支按列/按 packed 有效块
   跨核发放；deter+swizzle0 核上闭式算 fusedIdx 并跳过空槽；
3. **逐 task 展开**：``set_run_info`` 把全局块下标拆成 (b, n2, g, s2o, s1o)
   与各张量的 GM 偏移，并算好 K/V 复用与 dK/dV flush 的开关。

cube 与 vector 两侧调用**同一套**函数、读**同一块** metadata —— 这是两侧
task 数一致的前提，也是 set/wait 能配对的前提。任何只改一侧的分核逻辑都会
直接死锁。
"""

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_colsplit import col_core_count, col_local_to_global, col_packed_to_dense
from fag_common import CUBE_BASEM, CUBE_BASEN, VECTOR_BASEM
from fag_deter import (
    deter_lin_gidx,
    deter_lin_max_round,
    deter_lin_round_bounds,
)
from fag_deter_tnd_dense import (
    init_tnd_dense_schedule,
    make_tnd_deter_schedule,
    tnd_dense_gidx,
)
from fag_deter_tnd_sparse import init_tnd_sparse_schedule, tnd_sparse_gidx
# --- end generated imports ---


# ==================================================================
#  块级有效性与「下一块」
# ==================================================================


def is_block_valid_i(
    constInfo,
    index,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """块级 IsValid。mask_mode 是 tilingkey，只编一侧公式。

    TND（layout==1）走 _is_block_valid_tnd 变长几何解码。
    比较做成 0/1 算术：此函数会内联进 skip_invalid 的 while。0930 移植。
    """
    if layout == 1:
        return _is_block_valid_tnd(
            constInfo,
            index,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
            batch_cursor,
        )
    valid = 0
    if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
        physicalIndex = index
        if deter == 0:
            physicalIndex = seqused_to_padded_index(
                constInfo, index, used_q_tensor, used_kv_tensor
            )
        valid = seqused_block_valid(
            constInfo, physicalIndex, used_q_tensor, used_kv_tensor
        )
    else:
        valid = padded_block_valid(constInfo, index)
    return valid


def padded_block_valid(constInfo, index):
    """Original fixed-length mask predicate; callers without seqused stay here."""
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


def skip_invalid(
    constInfo,
    idx,
    blockEnd,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """线性支：while 无效则 idx++，越界 -1。TND 时用变长几何解码。0930 移植。"""
    # Fetch remains outside the pipeline's runtime condition. At end-of-range,
    # decode the preceding tile instead of a possibly empty trailing batch;
    # ``over`` still forces the exhausted fetch to return -1 below.
    checkIdx = idx
    if layout == 1:
        checkIdx = pl.min(idx, pl.max(blockEnd - 1, 0))
    v = is_block_valid_i(
        constInfo,
        checkIdx,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
        batch_cursor,
    )
    over = pl.min(pl.max(idx + 1 - blockEnd, 0), 1)
    v = pl.min(v + over, 1)
    while v == 0:
        idx = idx + 1
        checkIdx = idx
        if layout == 1:
            checkIdx = pl.min(idx, pl.max(blockEnd - 1, 0))
        cur = is_block_valid_i(
            constInfo,
            checkIdx,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
            batch_cursor,
        )
        over = pl.min(pl.max(idx + 1 - blockEnd, 0), 1)
        v = pl.min(cur + over, 1)
    if idx >= blockEnd:
        return pl.astype(-1, pl.DT_INT64)
    return pl.astype(idx, pl.DT_INT64)


def deter_max_round(constInfo, coreNum):
    """本核循环上界。非 swizzle 闭式估轮；swizzle 取 core0 的列数（lockstep）。"""
    if swizzle == 0:
        return deter_lin_max_round(constInfo, coreNum)
    return block_range_of(constInfo, 0, coreNum, 0, 0)


def init_deter_schedule(
    constInfo,
    coreNum,
    cu_q_tensor,
    cu_kv_tensor,
    used_q_tensor,
    used_kv_tensor,
):
    """Compute TND phase lengths once, identically on cube and vector cores.

    This is scalar kernel state, not an ABI extension or a private workspace.
    Fixed-layout kernels fold away the TND plan and retain their own schedule.
    """
    if layout == 1:
        if mask_mode == 0:
            return init_tnd_dense_schedule(
                constInfo,
                coreNum,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
        return init_tnd_sparse_schedule(
            constInfo,
            coreNum,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
    return make_tnd_deter_schedule(0, 0, 0, 0, 0, 0, 0)


def deter_round_bounds(constInfo, coreNum, tnd_plan=None):
    """All cores share one physical-round interval; swizzle stays unchanged."""
    if layout == 1:
        return pl.astype(0, pl.DT_INT64), tnd_plan.maxRound
    if swizzle == 0:
        return deter_lin_round_bounds(constInfo, coreNum)
    return pl.astype(0, pl.DT_INT64), deter_max_round(constInfo, coreNum)


def deter_swz_gidx(constInfo, cBlockIdx, coreNum, roundId):
    """deter+swizzle：与 ``block_idx_of`` 同映射，越界空槽 -1。"""
    if mask_mode == 0:
        n_col = constInfo.bSize * constInfo.n2Size * constInfo.gSize * constInfo.s2Outer
        t = cBlockIdx + (roundId // constInfo.s1Outer) * coreNum
        if t >= n_col:
            return pl.astype(-1, pl.DT_INT64)
        return pl.astype(
            swizzle_idx(constInfo, cBlockIdx, coreNum, roundId), pl.DT_INT64
        )
    packed = cBlockIdx + roundId * coreNum
    limit = (
        constInfo.bSize
        * constInfo.n2Size
        * constInfo.gSize
        * constInfo.totalPerBatchNum
    )
    if packed >= limit:
        return pl.astype(-1, pl.DT_INT64)
    return pl.astype(packed_to_dense_idx(constInfo, packed), pl.DT_INT64)


def skip_empty_deter(
    constInfo,
    cBlockIdx,
    coreNum,
    roundId,
    maxRound,
    used_q_tensor=None,
    used_kv_tensor=None,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    tnd_plan=None,
):
    """deter+swizzle0：只算当前物理轮，空槽不启动新 tile。

    已到期的跨轮流水仍需消费，且必须透传对应 FIX token，不能整轮 continue。
    当前轮解码与有效任务前瞻分开，不能用前瞻游标推进流水物理轮次。
    """
    dead = pl.min(pl.max(roundId + 1 - maxRound, 0), 1)
    if layout == 1:
        if mask_mode == 0:
            gIdx = tnd_dense_gidx(
                constInfo,
                cBlockIdx,
                coreNum,
                roundId,
                tnd_plan,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
        else:
            gIdx = tnd_sparse_gidx(
                constInfo,
                cBlockIdx,
                coreNum,
                roundId,
                tnd_plan,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
        valid = tnd_deter_block_valid(
            constInfo,
            gIdx,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
        gIdx = gIdx * valid + pl.astype(-1, pl.DT_INT64) * (1 - valid)
    else:
        gIdx = deter_lin_gidx(constInfo, cBlockIdx, coreNum, roundId)
        if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
            valid = seqused_block_valid(constInfo, gIdx, used_q_tensor, used_kv_tensor)
            gIdx = gIdx * valid + pl.astype(-1, pl.DT_INT64) * (1 - valid)
    gOut = gIdx * (1 - dead) + pl.astype(-1, pl.DT_INT64) * dead
    return pl.astype(gOut, pl.DT_INT64), pl.astype(roundId + 1, pl.DT_INT64)


def find_next_valid_deter(
    constInfo,
    cBlockIdx,
    coreNum,
    startRound,
    maxRound,
    used_q_tensor=None,
    used_kv_tensor=None,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    tnd_plan=None,
):
    """返回下一个有效任务及其原始物理轮次；无任务时返回 (-1, maxRound)。

    调用方缓存结果，只在消费有效任务后继续前瞻，各物理轮只解码一次。
    此游标不控制 readyRound/FIX；空轮仍由主循环逐轮透传同步。
    """
    searchRound = pl.astype(startRound, pl.DT_INT64)
    nextGIdx = pl.const(-1, pl.DT_INT64)
    nextRound = pl.astype(maxRound, pl.DT_INT64)
    while searchRound < maxRound:
        candidateGIdx, followingRound = skip_empty_deter(
            constInfo,
            cBlockIdx,
            coreNum,
            searchRound,
            maxRound,
            used_q_tensor,
            used_kv_tensor,
            cu_q_tensor,
            cu_kv_tensor,
            tnd_plan,
        )
        if candidateGIdx >= 0:
            nextGIdx = candidateGIdx
            nextRound = searchRound
            break
        searchRound = followingRound
    return nextGIdx, nextRound


def skip_deter_empty_pipe(gIdx, isLastLoop):
    """deter+swizzle0 空槽且未收尾：跳过 mm/doFix，避免 CV 事件二次 wait。

    空槽仍推进 denseCursor；taskId/L0C 只跟有效块。swizzle1 lockstep 不跳。
    """
    if deter == 0:
        return 0
    if swizzle == 1:
        return 0
    if gIdx >= 0:
        return 0
    if isLastLoop == 1:
        return 0
    return 1


def next_work_gidx(
    constInfo,
    cBlockIdx,
    coreNum,
    blockStart,
    blockEnd,
    localIdx,
    denseCursor,
    numBlocks,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """下一块全局下标；本核没有活块时 -1。TND 时 cu 张量透传给 skip_invalid。0930 移植。"""
    if swizzle == 0:
        return skip_invalid(
            constInfo,
            denseCursor,
            blockEnd,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
            batch_cursor,
        )
    if localIdx >= numBlocks:
        return pl.astype(-1, pl.DT_INT64)
    return block_idx_of(constInfo, cBlockIdx, coreNum, blockStart, localIdx)


def fetch_work_gidx(
    constInfo,
    meta,
    fagOff,
    cBlockIdx,
    coreNum,
    blockStart,
    blockEnd,
    localIdx,
    denseCursor,
    numBlocks,
    roundId,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """主循环唯一取块入口，返回 (gIdx, nextCursor)。

    deter+swizzle0：核上闭式跳空槽，nextCursor 是下一轮下标。
    deter+swizzle1：按轮 lockstep，空槽保持 -1，cursor 原样。
    非 deter：next_work_gidx；线性支 nextCursor = gIdx+1。
    """
    if deter == 1:
        if swizzle == 1:
            return (
                deter_swz_gidx(constInfo, cBlockIdx, coreNum, roundId),
                pl.astype(denseCursor, pl.DT_INT64),
            )
        return skip_empty_deter(
            constInfo,
            cBlockIdx,
            coreNum,
            denseCursor,
            deter_max_round(constInfo, coreNum),
        )
    gIdx = next_work_gidx(
        constInfo,
        cBlockIdx,
        coreNum,
        blockStart,
        blockEnd,
        localIdx,
        denseCursor,
        numBlocks,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
        batch_cursor,
    )
    if swizzle == 0:
        if gIdx >= 0:
            return pl.astype(gIdx, pl.DT_INT64), pl.astype(gIdx + 1, pl.DT_INT64)
        return pl.astype(gIdx, pl.DT_INT64), pl.astype(blockEnd, pl.DT_INT64)
    return pl.astype(gIdx, pl.DT_INT64), pl.astype(denseCursor, pl.DT_INT64)


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
# fagOff 随 FA 段（aic * stride * B * N2）变大，与 shape 表无关。
META_VIEW_LEN = 1048576


def block_range_of(constInfo, cBlockIdx, coreNum, blockStart, blockEnd):
    """本核负责的块数。线性支读 metadata；swizzle 仍走列跨核公式。

    mask+swizzle 按 packed 有效块跨核发，不再用整矩形列跨核。
    分核开关用 tilingkey 裸名 ``swizzle``，与 ``next_work_gidx`` 一样先判
    ``swizzle == 0``，死支公式不进线性核。
    """
    if swizzle == 0:
        return pl.astype(pl.max(blockEnd - blockStart, 0), pl.DT_INT64)
    if mask_mode == 0:
        numColTasks = (
            constInfo.bSize * constInfo.n2Size * constInfo.gSize * constInfo.s2Outer
        )
        rem = pl.max(numColTasks - cBlockIdx, 0)
        return pl.astype(
            ((rem + coreNum - 1) // coreNum) * constInfo.s1Outer, pl.DT_INT64
        )
    return pl.astype(col_core_count(constInfo, cBlockIdx, coreNum), pl.DT_INT64)


def block_idx_of(constInfo, cBlockIdx, coreNum, blockStart, localIdx):
    """局部块序号 -> 全局块下标。线性支起点取自 metadata。"""
    if swizzle == 0:
        return pl.astype(blockStart + localIdx, pl.DT_INT64)
    if mask_mode == 0:
        return pl.astype(
            swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx), pl.DT_INT64
        )
    gidx = col_local_to_global(constInfo, cBlockIdx, coreNum, localIdx)
    return pl.astype(col_packed_to_dense(constInfo, gidx), pl.DT_INT64)


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
    if swizzle == 0:
        isFirst = 1 - pl.min(localIdx, 1)
        gCur = s2_group_of(constInfo, gIdx)
        gPrev = s2_group_of(constInfo, pl.max(gIdx - 1, 0))
        return pl.max(isFirst, pl.min(gCur - gPrev, 1))
    return 1 - pl.min(localIdx % constInfo.s1Outer, 1)


def next_s2_same(constInfo, gIdx, localIdx):
    """下一块是否仍落在同一个 s2 块内(1=是，0=否)。"""
    if swizzle == 0:
        return 1 - pl.min(
            s2_group_of(constInfo, gIdx + 1) - s2_group_of(constInfo, gIdx), 1
        )
    return pl.min((localIdx + 1) % constInfo.s1Outer, 1)


# ==================================================================
#  逐 task 的 RunInfo
# ==================================================================


def set_run_info(
    constInfo,
    runInfo,
    taskId,
    index,
    subIdx,
    localIdx,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    # TND（layout==1）复用本核 batch cursor，再做 per-batch divmod。
    # cu 张量由 dispatcher 循环外创建后传入；非 TND 时为 None，不读。0930 移植。
    if layout == 1:
        if deter == 1:
            # Pass literal None: forwarding an omitted optional scalar alias
            # is unsupported by the 0930 inline-argument binder.
            _set_run_info_tnd_varlen(
                constInfo,
                runInfo,
                taskId,
                index,
                subIdx,
                localIdx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
                None,
            )
        else:
            _set_run_info_tnd_varlen(
                constInfo,
                runInfo,
                taskId,
                index,
                subIdx,
                localIdx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
                batch_cursor,
            )
        return
    if deter == 0:
        if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
            # Metadata packs effective grids; GM remains the original padded layout.
            index = seqused_to_padded_index(
                constInfo, index, used_q_tensor, used_kv_tensor
            )
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

    # Effective lengths only clip compute tiles; all tensor strides stay physical.
    sq = padded_sequence_length(
        used_q_tensor, runInfo.boIdx, constInfo.sequsedQOptional, constInfo.s1Size
    )
    sk = padded_sequence_length(
        used_kv_tensor, runInfo.boIdx, constInfo.sequsedKvOptional, constInfo.s2Size
    )
    runInfo.batchS1Len = sq
    runInfo.batchS2Len = sk
    s1Real = pl.min(CUBE_BASEM, sq - runInfo.s1oIdx * CUBE_BASEM)
    s2Real = pl.min(CUBE_BASEN, sk - runInfo.s2oIdx * CUBE_BASEN)
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
    # TND：lse 行地址 = boIdx * S1 + loadBase（全局 token 行，lse 是 2D
    # [N1, B*S1] token 主序视图）。BSND 时 loadBaseTnd == loadBase，
    # 统一用 loadBaseTnd 消除 kernel 侧 layout 分支。0930 bisect-sink 移植。
    runInfo.loadBaseTnd = runInfo.boIdx * constInfo.s1Size + runInfo.loadBase

    runInfo.loadKV = need_load_kv(constInfo, index, localIdx)
    # 与 loadKV 互补：列首覆写 L0C，列中累加。消费点必须 == 1，不要用 == 0。
    runInfo.dkvAcc = 1 - runInfo.loadKV
    runInfo.flushDkv = 1 - next_s2_same(constInfo, index, localIdx)
    # swizzle 表同轮可能撞 (b,n2,s2)，L0C 跨块累加不成立。非 swizzle ASC
    # 闭式本核独占 (b,n2,s2)，由 apply_s2_idx_no_change 开累加。
    if deter == 1:
        if swizzle == 1:
            runInfo.loadKV = 1
            runInfo.dkvAcc = 0
            runInfo.flushDkv = 1
    runInfo.gIdx = index
    runInfo.dqFirst = 1
    runInfo.dqLast = 1
    if is_bn2_multiblk == 1:
        check_s1_range_in_bn2(constInfo, runInfo)


def padded_sequence_length(used_tensor, batch, has_used, physical_length):
    """Read an optional effective prefix without changing the padded GM stride."""
    length = physical_length
    if has_used == 1:
        length = pl.astype(pl.getval(used_tensor, batch), pl.DT_INT64)
    return length


def seqused_to_padded_index(constInfo, index, used_q_tensor, used_kv_tensor):
    """Map normal metadata's compact effective grid to a fixed-layout tile index.

    Deterministic scheduling already emits physical-grid indices and must not
    call this converter. Empty batches consume no compact indices.
    """
    batch = 0
    prefix = 0
    for _t in pl.range(0, constInfo.bSize):
        sq = padded_sequence_length(
            used_q_tensor, batch, constInfo.sequsedQOptional, constInfo.s1Size
        )
        sk = padded_sequence_length(
            used_kv_tensor, batch, constInfo.sequsedKvOptional, constInfo.s2Size
        )
        m = (sq + CUBE_BASEM - 1) // CUBE_BASEM
        n = (sk + CUBE_BASEN - 1) // CUBE_BASEN
        count = constInfo.n2Size * constInfo.gSize * m * n
        advance = pl.min(pl.max(index - prefix - count + 1, 0), 1)
        advance = advance * pl.min(pl.max(constInfo.bSize - 1 - batch, 0), 1)
        batch = batch + advance
        prefix = prefix + advance * count
    sq = padded_sequence_length(
        used_q_tensor, batch, constInfo.sequsedQOptional, constInfo.s1Size
    )
    sk = padded_sequence_length(
        used_kv_tensor, batch, constInfo.sequsedKvOptional, constInfo.s2Size
    )
    m = (sq + CUBE_BASEM - 1) // CUBE_BASEM
    n = (sk + CUBE_BASEN - 1) // CUBE_BASEN
    mn = pl.max(m * n, 1)
    delta = index - prefix
    head = delta // mn
    spatial = delta % mn
    s2o = spatial // pl.max(m, 1)
    s1o = spatial % pl.max(m, 1)
    return (
        batch * constInfo.n2GS1oS2o
        + head * constInfo.s1oS2o
        + s2o * constInfo.s1Outer
        + s1o
    )


def seqused_block_valid(constInfo, index, used_q_tensor, used_kv_tensor):
    """Effective per-batch geometry on a physical padded tile grid (also drain-safe)."""
    safeIndex = pl.max(index, 0)
    batch = pl.min(safeIndex // pl.max(constInfo.n2GS1oS2o, 1), constInfo.bSize - 1)
    spatial = safeIndex % pl.max(constInfo.s1oS2o, 1)
    s2o = spatial // pl.max(constInfo.s1Outer, 1)
    s1o = spatial % pl.max(constInfo.s1Outer, 1)
    sq = padded_sequence_length(
        used_q_tensor, batch, constInfo.sequsedQOptional, constInfo.s1Size
    )
    sk = padded_sequence_length(
        used_kv_tensor, batch, constInfo.sequsedKvOptional, constInfo.s2Size
    )
    qLeft = s1o * CUBE_BASEM
    kLeft = s2o * CUBE_BASEN
    qRight = pl.min(qLeft + CUBE_BASEM, sq)
    kRight = pl.min(kLeft + CUBE_BASEN, sk)
    valid = pl.min(pl.max(index + 1, 0), 1)
    valid = valid * pl.min(pl.max(constInfo.bSize * constInfo.n2GS1oS2o - index, 0), 1)
    valid = valid * pl.min(pl.max(sq - qLeft, 0), 1) * pl.min(pl.max(sk - kLeft, 0), 1)
    if mask_mode == 3:
        valid = valid * pl.min(pl.max(qRight + sk - sq - kLeft, 0), 1)
    if mask_mode == 4:
        maxDelta = constInfo.s1Size - constInfo.s2Size
        pre = constInfo.s1Token - maxDelta + sq - sk
        following = constInfo.s2Token + maxDelta - sq + sk
        valid = valid * pl.min(pl.max(qRight + following - kLeft, 0), 1)
        valid = valid * pl.min(pl.max(kRight + pre - qLeft, 0), 1)
    return valid


def tnd_sequence_length(cu_tensor, used_tensor, batch, has_used):
    """Effective geometry comes from seqused; cu always retains physical offsets.

    The host validates optional presence and metadata validates length bounds.
    A null optional tensor is never read. Keep one return for loop-phi lowering.
    """
    begin = pl.astype(pl.getval(cu_tensor, batch), pl.DT_INT64)
    end = pl.astype(pl.getval(cu_tensor, batch + 1), pl.DT_INT64)
    length = pl.max(end - begin, 0)
    if has_used == 1:
        length = pl.astype(pl.getval(used_tensor, batch), pl.DT_INT64)
    return length


def tnd_deter_block_valid(
    constInfo,
    index,
    cu_q_tensor,
    cu_kv_tensor,
    used_q_tensor,
    used_kv_tensor,
):
    """Validate a synthetic newDeter coordinate against actual TND geometry.

    Sparse virtual rectangles may include masked tiles. Only the logical index
    uses max-S strides; input/output storage remains packed by physical cu.
    Clamp the lookup for empty/drain slots so no cu or seqused read is out of range.
    """
    safeIndex = pl.max(index, 0)
    batch = pl.min(safeIndex // pl.max(constInfo.n2GS1oS2o, 1), constInfo.bSize - 1)
    sq = tnd_sequence_length(
        cu_q_tensor, used_q_tensor, batch, constInfo.sequsedQOptional
    )
    sk = tnd_sequence_length(
        cu_kv_tensor, used_kv_tensor, batch, constInfo.sequsedKvOptional
    )
    tail = safeIndex % pl.max(constInfo.s1oS2o, 1)
    s1o = tail % pl.max(constInfo.s1Outer, 1)
    s2o = tail // pl.max(constInfo.s1Outer, 1)
    inGrid = pl.min(pl.max(index + 1, 0), 1)
    inGrid = inGrid * pl.min(pl.max(constInfo.totalBlockNum - index, 0), 1)
    inGrid = inGrid * pl.min(pl.max(sq - CUBE_BASEM * s1o, 0), 1)
    inGrid = inGrid * pl.min(pl.max(sk - CUBE_BASEN * s2o, 0), 1)
    if mask_mode == 0:
        return inGrid
    if mask_mode == 3:
        end = pl.min(pl.max(sk - sq + CUBE_BASEM * (s1o + 1), 0), sk)
        return inGrid * pl.min(pl.max(end - CUBE_BASEN * s2o, 0), 1)
    maxD = constInfo.s1Size - constInfo.s2Size
    calcS1 = constInfo.s1Token - maxD + sq - sk
    calcS2 = constInfo.s2Token + maxD - sq + sk
    colL = CUBE_BASEN * s2o
    colR = pl.min(colL + CUBE_BASEN, sk)
    leftIP = pl.max(colL - calcS2, 0)
    m = (sq + CUBE_BASEM - 1) // CUBE_BASEM
    overLeft = pl.min(pl.max(leftIP - sq, 0), 1)
    begin = (1 - overLeft) * (leftIP // CUBE_BASEM) + overLeft * m
    end = (pl.min(sq, pl.max(colR + calcS1, 0)) + CUBE_BASEM - 1) // CUBE_BASEM
    ge = pl.min(pl.max(s1o - begin + 1, 0), 1)
    lt = pl.min(pl.max(end - s1o, 0), 1)
    return inGrid * ge * lt


def init_tnd_batch_cursor():
    """Per-pipeline scalar cache; never stored in constInfo or shared workspace."""
    return pl.struct(
        "FagTndBatchCursor",
        ready=pl.astype(0, pl.DT_INT64),
        batch=pl.astype(0, pl.DT_INT64),
        begin=pl.astype(0, pl.DT_INT64),
        end=pl.astype(0, pl.DT_INT64),
        sq=pl.astype(0, pl.DT_INT64),
        sk=pl.astype(0, pl.DT_INT64),
        m=pl.astype(0, pl.DT_INT64),
        n=pl.astype(0, pl.DT_INT64),
        qBase=pl.astype(0, pl.DT_INT64),
        kvBase=pl.astype(0, pl.DT_INT64),
    )


def load_tnd_batch_cursor(ci, cursor, cuq, cuk, usedq, usedk):
    """Refresh only when entering a batch; geometry is effective, offsets physical."""
    cursor.qBase = pl.astype(pl.getval(cuq, cursor.batch), pl.DT_INT64)
    cursor.kvBase = pl.astype(pl.getval(cuk, cursor.batch), pl.DT_INT64)
    sq = pl.astype(pl.getval(cuq, cursor.batch + 1), pl.DT_INT64) - cursor.qBase
    sk = pl.astype(pl.getval(cuk, cursor.batch + 1), pl.DT_INT64) - cursor.kvBase
    if ci.sequsedQOptional == 1:
        sq = pl.astype(pl.getval(usedq, cursor.batch), pl.DT_INT64)
    if ci.sequsedKvOptional == 1:
        sk = pl.astype(pl.getval(usedk, cursor.batch), pl.DT_INT64)
    cursor.sq = pl.max(sq, 0)
    cursor.sk = pl.max(sk, 0)
    cursor.m = (cursor.sq + CUBE_BASEM - 1) // CUBE_BASEM
    cursor.n = (cursor.sk + CUBE_BASEN - 1) // CUBE_BASEN
    cursor.end = cursor.begin + ci.n2Size * ci.gSize * cursor.m * cursor.n
    cursor.ready = pl.astype(1, pl.DT_INT64)


def locate_tnd_batch(ci, index, cursor, cuq, cuk, usedq, usedk):
    """Locate the packed rectangular tile index without restarting at batch zero.

    Normal streams advance monotonically, so each crossed batch is loaded once.
    The backward correction also supports revisits, without assuming swizzle
    support or sorting/reordering tasks. Duplicate prefixes (empty batches) are
    skipped with a bounded batch index; the all-empty drain remains in range.
    """
    if cursor.ready == 0:
        load_tnd_batch_cursor(ci, cursor, cuq, cuk, usedq, usedk)
    while index < cursor.begin and cursor.batch > 0:
        previousEnd = cursor.begin
        cursor.batch = cursor.batch - 1
        load_tnd_batch_cursor(ci, cursor, cuq, cuk, usedq, usedk)
        count = cursor.end - cursor.begin
        cursor.begin = previousEnd - count
        cursor.end = previousEnd
    while index >= cursor.end and cursor.batch + 1 < ci.bSize:
        cursor.begin = cursor.end
        cursor.batch = cursor.batch + 1
        load_tnd_batch_cursor(ci, cursor, cuq, cuk, usedq, usedk)


def _is_block_valid_tnd(
    constInfo,
    index,
    cu_q_tensor,
    cu_kv_tensor,
    used_q_tensor,
    used_kv_tensor,
    batch_cursor,
):
    """TND 变长 + mask 的块有效性，与 AICPU DoFagTndSplit 的 BEGIN/END
    逐项对齐。0930 bisect-sink 定案移植。
    """
    if mask_mode == 0:
        return 1
    n2 = constInfo.n2Size
    g = constInfo.gSize
    locate_tnd_batch(
        constInfo,
        index,
        batch_cursor,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
    )
    sq = batch_cursor.sq
    sk = batch_cursor.sk
    m = batch_cursor.m
    n = batch_cursor.n
    delta = index - batch_cursor.begin
    # An all-empty grid can still be visited by the unconditional drain fetch.
    # Use nonzero divisors and an explicit range predicate, without adding a
    # runtime early return around cursor updates (PyPTO loop-phi constraint).
    inGrid = pl.min(pl.max(n2 * g * m * n - delta, 0), 1)
    inGrid = inGrid * pl.min(pl.max(delta + 1, 0), 1)
    mn = pl.max(m * n, 1)
    gmn = g * mn
    bTail = delta % gmn
    gTail = bTail % mn
    s2o = gTail // pl.max(m, 1)
    s1o = gTail % pl.max(m, 1)
    if mask_mode == 3:
        s2EndLen = pl.min(pl.max(sk - sq + CUBE_BASEM * (s1o + 1), 0), sk)
        return inGrid * pl.min(pl.max(s2EndLen - CUBE_BASEN * s2o, 0), 1)
    maxD = constInfo.s1Size - constInfo.s2Size
    calcS1 = constInfo.s1Token - maxD + sq - sk
    calcS2 = constInfo.s2Token + maxD - sq + sk
    colL = CUBE_BASEN * s2o
    colOk = pl.min(pl.max(sk - colL, 0), 1)
    colR = pl.min(colL + CUBE_BASEN, sk)
    leftIP = pl.max(colL - calcS2, 0)
    overLeft = pl.min(pl.max(leftIP - sq, 0), 1)
    begin = (1 - overLeft) * (leftIP // CUBE_BASEM) + overLeft * m
    end = (pl.min(sq, pl.max(colR + calcS1, 0)) + CUBE_BASEM - 1) // CUBE_BASEM
    ge = pl.min(pl.max(s1o - begin + 1, 0), 1)
    lt = pl.min(pl.max(end - s1o, 0), 1)
    return inGrid * colOk * ge * lt


def _set_run_info_tnd_varlen(
    constInfo,
    runInfo,
    taskId,
    index,
    subIdx,
    localIdx,
    cu_q_tensor,
    cu_kv_tensor,
    used_q_tensor,
    used_kv_tensor,
    batch_cursor,
):
    """变长 TND 解码：拼接空间 gIdx → (bo,n2,g,s2o,s1o) + GM token 基址。

    非确定性复用 packed 网格的核内 batch cursor；确定性
    调度已定位 batch，以 max-shape 网格编码坐标，直接解码而不重复扫描。
    非确定性再用有效 outer 做块内 divmod，仅跨 batch 时读取几何；
    确定性用 max-shape outer 解码。GM 基址始终来自 cu，不能以有效长度
    求和替代物理偏移。
    寻址约定（与 [1, T, n, d] 视图配套）：
      idx0Q = 0，queryOffset = tokenBaseQ + s1oIdx * 128（全局 token 行）。
    0930 bisect-sink 定案移植。
    """
    n2 = constInfo.n2Size
    g = constInfo.gSize
    w = 0
    prefix = 0
    if deter == 1:
        w = index // constInfo.n2GS1oS2o
        prefix = w * constInfo.n2GS1oS2o
        tokenBaseQ = pl.astype(pl.getval(cu_q_tensor, w), pl.DT_INT64)
        batchS1Len = tnd_sequence_length(
            cu_q_tensor, used_q_tensor, w, constInfo.sequsedQOptional
        )
        tokenBaseKV = pl.astype(pl.getval(cu_kv_tensor, w), pl.DT_INT64)
        batchS2Len = tnd_sequence_length(
            cu_kv_tensor, used_kv_tensor, w, constInfo.sequsedKvOptional
        )
        # Synthetic coordinates use max-S divisors, never packed-grid prefixes.
        m = constInfo.s1Outer
        n = constInfo.s2Outer
    else:
        locate_tnd_batch(
            constInfo,
            index,
            batch_cursor,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
        w = batch_cursor.batch
        prefix = batch_cursor.begin
        tokenBaseQ = batch_cursor.qBase
        tokenBaseKV = batch_cursor.kvBase
        batchS1Len = batch_cursor.sq
        batchS2Len = batch_cursor.sk
        m = batch_cursor.m
        n = batch_cursor.n
    delta = index - prefix
    mn = m * n
    gmn = g * mn
    runInfo.boIdx = w
    runInfo.n2oIdx = delta // gmn
    bTail = delta % gmn
    runInfo.goIdx = bTail // mn
    gTail = bTail % mn
    runInfo.s2oIdx = gTail // m
    runInfo.s1oIdx = gTail % m
    s1Real = pl.max(pl.min(CUBE_BASEM, batchS1Len - runInfo.s1oIdx * CUBE_BASEM), 0)
    s2Real = pl.max(pl.min(CUBE_BASEN, batchS2Len - runInfo.s2oIdx * CUBE_BASEN), 0)
    runInfo.s1RealSize = s1Real
    runInfo.s2RealSize = s2Real
    runInfo.halfS1RealSize = pl.max(
        pl.min(s1Real - subIdx * VECTOR_BASEM, VECTOR_BASEM), 0
    )
    runInfo.queryOffset = tokenBaseQ + runInfo.s1oIdx * CUBE_BASEM
    runInfo.keyOffset = tokenBaseKV + runInfo.s2oIdx * CUBE_BASEN
    headQ = runInfo.n2oIdx * g + runInfo.goIdx
    runInfo.idx0Q = 0
    runInfo.idx2Q = headQ
    runInfo.idx0KV = 0
    runInfo.idx2KV = runInfo.n2oIdx
    runInfo.tokenBaseQ = tokenBaseQ
    runInfo.tokenBaseKV = tokenBaseKV
    runInfo.batchS1Len = batchS1Len
    runInfo.batchS2Len = batchS2Len
    rowBase = runInfo.queryOffset + subIdx * VECTOR_BASEM
    batchEndQ = tokenBaseQ + batchS1Len
    batchLoadBase = pl.max(pl.min(rowBase, batchEndQ - VECTOR_BASEM), tokenBaseQ)
    # Keep the vector y/dy/lse window inside the physical packed tensor.
    # A short final batch can borrow preceding rows: rowShift and
    # halfS1RealSize exclude them from computation. If the entire tensor
    # is shorter than VECTOR_BASEM, the vector loads also shrink their
    # valid row count; moving the base alone cannot make a full window fit.
    tensorCapQ = pl.max(constInfo.t1Size - VECTOR_BASEM, 0)
    runInfo.loadBase = pl.min(batchLoadBase, tensorCapQ)
    runInfo.loadBaseTnd = runInfo.loadBase
    runInfo.rowShift = rowBase - runInfo.loadBase
    # The packed TND grid uses each batch's own s1Outer, not maxS1's grid.
    # apply_l0c_s2 refines these safe defaults from adjacent decoded RunInfo.
    runInfo.loadKV = 1
    runInfo.dkvAcc = 0
    runInfo.flushDkv = 1
    runInfo.gIdx = index
    runInfo.dqFirst = 1
    runInfo.dqLast = 1
    if is_bn2_multiblk == 1:
        # Host permits BN2 only when both Q and KV are uniformly sized.
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


def apply_l0c_s2(constInfo, runInfos, curInfo, taskId, ring):
    """TND、mask 或 deter+非 swizzle：按上一有效块的 (s2,b,n2) 开 L0C 累加。"""
    if layout == 1:
        apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)
        return
    if mask_mode != 0:
        apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)
        return
    if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
        apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)
        return
    if deter == 1:
        if swizzle == 0:
            apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)


def dkv_target_of(constInfo, index):
    """Pack the dK/dV owner ``(b, n2, s2o)``; G is intentionally excluded."""
    safeIndex = pl.max(index, 0)
    bTail = safeIndex % constInfo.n2GS1oS2o
    n2Tail = bTail % constInfo.gS1oS2o
    bIdx = safeIndex // constInfo.n2GS1oS2o
    n2Idx = bTail // constInfo.gS1oS2o
    s2Idx = (safeIndex % constInfo.s1oS2o) // constInfo.s1Outer
    return (bIdx * constInfo.n2Size + n2Idx) * constInfo.s2Outer + s2Idx


def same_dkv_target(constInfo, lhs, rhs):
    """Scalar 0/1 equality for two valid fused indices' dK/dV destinations."""
    lhsValid = pl.min(pl.max(lhs + 1, 0), 1)
    rhsValid = pl.min(pl.max(rhs + 1, 0), 1)
    diff = dkv_target_of(constInfo, lhs) - dkv_target_of(constInfo, rhs)
    different = pl.min(pl.min(pl.max(diff, 0), 1) + pl.min(pl.max(-diff, 0), 1), 1)
    return lhsValid * rhsValid * (1 - different)


def set_deter_l0c_neighbors(constInfo, runInfo, prevGIdx, curGIdx, nextGIdx):
    """Use adjacent valid tasks; an empty physical round preserves residency."""
    prevSame = same_dkv_target(constInfo, prevGIdx, curGIdx)
    nextSame = same_dkv_target(constInfo, curGIdx, nextGIdx)
    runInfo.loadKV = 1 - prevSame
    runInfo.dkvAcc = prevSame
    runInfo.flushDkv = 1 - nextSame


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
    if deter == 1:
        if swizzle == 1:
            curInfo.loadKV = 1
            curInfo.dkvAcc = 0
            curInfo.flushDkv = 1
            prevR.flushDkv = 1
