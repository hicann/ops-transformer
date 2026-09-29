# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from types import SimpleNamespace

AIC_CORE_MAX_NUM = 36
AIV_CORE_MAX_NUM = 72
MQSMLA_METADATA_TOTAL_SIZE = 1024

FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8
FD_METADATA_BASE = AIC_CORE_MAX_NUM * FA_METADATA_SIZE  # 324
# faMetadata(324) + fdMetadata(576) = 900 字之后的首个保留字：存放 fdUsedVecNum
# （= FD 归约 AIV 数）。消费侧据此统一决定是否执行 grid barrier + FD 归约阶段。
FD_USED_VEC_NUM_WORD = (
    AIC_CORE_MAX_NUM * FA_METADATA_SIZE + AIV_CORE_MAX_NUM * FD_METADATA_SIZE
)  # 900

FA_CORE_ENABLE_INDEX = 0
FA_BN2_START_INDEX = 1
FA_M_START_INDEX = 2
FA_S2_START_INDEX = 3
FA_BN2_END_INDEX = 4
FA_M_END_INDEX = 5
FA_S2_END_INDEX = 6
FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7
FA_S2_MAX_NUM = 8

FD_CORE_ENABLE_INDEX = 0
FD_BN2_IDX_INDEX = 1
FD_M_IDX_INDEX = 2
FD_WORKSPACE_IDX_INDEX = 3
FD_WORKSPACE_NUM_INDEX = 4
FD_M_START_INDEX = 5
FD_M_NUM_INDEX = 6

FA_TOLERANCE_RATIO = 2

_COST_FULL_TILE = 44
_COST_M_PART = 24


def _md_uniform(a):
    for i in range(0, FD_USED_VEC_NUM_WORD + 1):
        a.metadata[i] = 0
    if a.rows <= 0 or a.blocks <= 0:
        return 1
    rows_per_core = a.rows // a.blocks
    extra = a.rows % a.blocks
    start = 0
    batch1 = a.batch + 1
    for core in range(0, a.blocks):
        count = rows_per_core
        if core < extra:
            count += 1
        end = start + count
        if count > 0:
            bn2_s = 0
            for b in range(0, batch1):
                if a.cu_q[b] <= start:
                    bn2_s = b
            m_s = start - a.cu_q[bn2_s]
            bn2_e = 0
            for b in range(0, batch1):
                if a.cu_q[b] <= end:
                    bn2_e = b
            m_e = end - a.cu_q[bn2_e]
            fa = FA_METADATA_SIZE * core
            a.metadata[fa + FA_CORE_ENABLE_INDEX] = 1
            a.metadata[fa + FA_BN2_START_INDEX] = bn2_s
            a.metadata[fa + FA_M_START_INDEX] = m_s
            a.metadata[fa + FA_BN2_END_INDEX] = bn2_e
            a.metadata[fa + FA_M_END_INDEX] = m_e
        start = end
    return 0


def _md_row_tiles(w, wc):
    n = 0
    if w > 0:
        n = (w + 127) // 128
    if wc > 0:
        n = n + (wc + 127) // 128
    return n


def _md_row_cost(w, wc):
    cost = 0
    if w > 0:
        cost = _COST_FULL_TILE * (w // 128)
        t = w % 128
        if t != 0:
            cost = cost + _COST_M_PART + 10 * ((t + 63) // 64)
    if wc > 0:
        costc = _COST_FULL_TILE * (wc // 128)
        tc = wc % 128
        if tc != 0:
            costc = costc + _COST_M_PART + 10 * ((tc + 63) // 64)
        cost = cost + costc
    return cost


def _md_row_last(w, wc):
    # 行末 tile 代价（cmp 侧在后，优先）
    if wc > 0:
        t = wc % 128
        if t == 0:
            return _COST_FULL_TILE
        return _COST_M_PART + 10 * ((t + 63) // 64)
    if w > 0:
        t = w % 128
        if t == 0:
            return _COST_FULL_TILE
        return _COST_M_PART + 10 * ((t + 63) // 64)
    return 0


def _md_tile_cost(w, wc, t):
    # 行内第 t 个 tile 的代价（0 ≤ t < 行 tile 数）
    tw = 0
    if w > 0:
        tw = (w + 127) // 128
    if t < tw:
        wt = w % 128
        if wt != 0 and t == tw - 1:
            return _COST_M_PART + 10 * ((wt + 63) // 64)
        return _COST_FULL_TILE
    tc = 0
    if wc > 0:
        tc = (wc + 127) // 128
    if tw + tc == 0:
        return 0
    c = t - tw
    ct = wc % 128
    if ct != 0 and c == tc - 1:
        return _COST_M_PART + 10 * ((ct + 63) // 64)
    return _COST_FULL_TILE


def _md_selective_fd_72(a):
    # Exact 72 x 5-tile contiguous plan: eight 9-row groups on four cores
    # (11/11/11/12 tiles). Three local split rows per group.
    if a.has_cmp != 1:
        return _md_uniform(a)
    for row in range(0, 72):
        if a.win_len[row] != 128 or a.cmp_len[row] != 512:
            return _md_uniform(a)
    for i in range(0, FD_USED_VEC_NUM_WORD + 51):
        a.metadata[i] = 0
    for core in range(0, 32):
        group = core // 4
        local = core % 4
        base_tile = group * 45
        start_tile = base_tile
        end_tile = base_tile + 11
        first_fd = group * 6
        if local == 1:
            start_tile = base_tile + 11
            end_tile = base_tile + 22
            first_fd = first_fd + 1
        if local == 2:
            start_tile = base_tile + 22
            end_tile = base_tile + 33
            first_fd = first_fd + 3
        if local == 3:
            start_tile = base_tile + 33
            end_tile = base_tile + 45
            first_fd = first_fd + 5
        start = start_tile // 5
        end = end_tile // 5
        bs = 0
        be = 0
        for bi in range(0, a.batch + 1):
            if a.cu_q[bi] <= start:
                bs = bi
            if a.cu_q[bi] <= end:
                be = bi
        base = core * FA_METADATA_SIZE
        a.metadata[base + FA_CORE_ENABLE_INDEX] = 1
        a.metadata[base + FA_BN2_START_INDEX] = bs
        a.metadata[base + FA_M_START_INDEX] = start - a.cu_q[bs]
        a.metadata[base + FA_S2_START_INDEX] = start_tile % 5
        a.metadata[base + FA_BN2_END_INDEX] = be
        a.metadata[base + FA_M_END_INDEX] = end - a.cu_q[be]
        a.metadata[base + FA_S2_END_INDEX] = end_tile % 5
        a.metadata[base + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = first_fd
    for task in range(0, 24):
        group = task // 3
        which = task % 3
        row = group * 9 + 2 + 2 * which
        reducer_core = group * 4 + which
        batch = 0
        for bi in range(0, a.batch + 1):
            if a.cu_q[bi] <= row:
                batch = bi
        for half in range(0, 2):
            base = FD_METADATA_BASE + (2 * reducer_core + half) * FD_METADATA_SIZE
            a.metadata[base + FD_CORE_ENABLE_INDEX] = 1
            a.metadata[base + FD_BN2_IDX_INDEX] = batch
            a.metadata[base + FD_M_IDX_INDEX] = row - a.cu_q[batch]
            a.metadata[base + FD_WORKSPACE_IDX_INDEX] = 2 * task
            a.metadata[base + FD_WORKSPACE_NUM_INDEX] = 2
            a.metadata[base + FD_M_START_INDEX] = 32 * half
            a.metadata[base + FD_M_NUM_INDEX] = 32
    a.metadata[FD_USED_VEC_NUM_WORD] = 48
    a.metadata[FD_USED_VEC_NUM_WORD + 50] = 1
    return 0


def _plan(a):
    if a.rows <= 0 or a.blocks <= 0 or a.blocks > AIC_CORE_MAX_NUM:
        return 1
    if a.support_fd == 2:
        if a.rows == 72 and a.blocks == 32:
            return _md_selective_fd_72(a)
        return _md_uniform(a)
    if a.support_fd == 0:
        return _md_uniform(a)
    fa_m_start = [0] * AIC_CORE_MAX_NUM
    fa_s2_start = [0] * AIC_CORE_MAX_NUM
    fa_m_end = [0] * AIC_CORE_MAX_NUM
    fa_s2_end = [0] * AIC_CORE_MAX_NUM
    fa_first_fd = [0] * AIC_CORE_MAX_NUM
    fd_row = [0] * AIC_CORE_MAX_NUM
    fd_ws = [0] * AIC_CORE_MAX_NUM
    fd_num = [0] * AIC_CORE_MAX_NUM

    for i in range(0, MQSMLA_METADATA_TOTAL_SIZE):
        a.metadata[i] = 0

    # ---- pass 1：总代价 + 总 tile 数 + 最大行代价/行数（逐行读 length 值）----
    total_cost = 0
    total_tiles = 0
    max_row_cost = 0
    max_row_tiles = 0
    r = 0
    while r < a.rows:
        w = a.win_len[r]
        wc = 0
        if a.has_cmp == 1:
            wc = a.cmp_len[r]
        rc = _md_row_cost(w, wc)
        rt = _md_row_tiles(w, wc)
        total_cost = total_cost + rc
        total_tiles = total_tiles + rt
        if rc > max_row_cost:
            max_row_cost = rc
        if rt > max_row_tiles:
            max_row_tiles = rt
        r = r + 1

    # Conservative benefit model, in 128-token sparse tiles. The uniform
    # critical path is bounded by ceil(rows/blocks)*max_row_tiles; the ideal
    # split floor is ceil(total_tiles/blocks). At 32 cores the fixed decode
    # guards save 0/0/3 tiles, so all remain whole-row. A 33-row,9-tile workload
    # saves 8 and permits FD. The margin pays for staging/barrier/reduction;
    # it also excludes numerically fragile two-tile short-row splits.
    ceil_rows_blocks = (a.rows + a.blocks - 1) // a.blocks
    fd_floor = (total_tiles + a.blocks - 1) // a.blocks
    uniform_crit = ceil_rows_blocks * max_row_tiles
    support_fd = 0
    if total_tiles > a.blocks:
        if uniform_crit - fd_floor >= a.fd_min_saved_tiles:
            support_fd = 1

    if support_fd == 0:
        return _md_uniform(a)

    # ---- pass 2：贪心分核（= AscendC CalcSplitPlan/AssignBlocksToCore）----
    cur_row = 0
    cur_tile = 0
    finished = 0
    wv = a.win_len[0]
    wcv = 0
    if a.has_cmp == 1:
        wcv = a.cmp_len[0]
    tiles_c = _md_row_tiles(wv, wcv)
    rem_cost = _md_row_cost(wv, wcv)
    last_c = _md_row_last(wv, wcv)
    unassigned = total_cost
    kv_split = 1  # 当前挂起切分行的分部数（= AscendC curKvSplitPart）
    pre_fd = 0  # 已记录 FD 任务占用的 slot 数（= preFdDataNum）
    num_fd = 0
    used = 0
    core = 0
    while core < a.blocks:
        if finished == 1:
            break
        # 不按 unassigned<=0 提前退出：代价耗尽后游标必在行边界，剩余的空行
        # 仍需落入某个使能核的区间以写出 finalize_empty（全空 case 由此自然
        # 收敛到 core0，等价 AscendC 的 isKvSeqAllZero 短路）。
        fa_first_fd[core] = pre_fd + kv_split - 1
        fa_m_start[core] = cur_row
        fa_s2_start[core] = cur_tile
        limit = unassigned // (a.blocks - core)
        if support_fd == 0:
            # 整行分配：costLimit 抬到最大行代价，保证任一整行都能落入单核
            # （= AscendC !supportFd 的 max(avgCost, maxS1GCost)），永不切行。
            if max_row_cost > limit:
                limit = max_row_cost
        cost = 0
        blk = 0
        # 1) 按行吸收（空行零代价直接推进游标，= AscendC 跳零块行）
        while finished == 0:
            if tiles_c == 0:
                cur_row = cur_row + 1
                cur_tile = 0
                if cur_row >= a.rows:
                    finished = 1
                else:
                    wv = a.win_len[cur_row]
                    wcv = 0
                    if a.has_cmp == 1:
                        wcv = a.cmp_len[cur_row]
                    tiles_c = _md_row_tiles(wv, wcv)
                    rem_cost = _md_row_cost(wv, wcv)
                    last_c = _md_row_last(wv, wcv)
            elif limit + last_c // FA_TOLERANCE_RATIO >= cost + rem_cost:
                cost = cost + rem_cost
                blk = blk + (tiles_c - cur_tile)
                cur_row = cur_row + 1
                cur_tile = 0
                if cur_row >= a.rows:
                    finished = 1
                else:
                    wv = a.win_len[cur_row]
                    wcv = 0
                    if a.has_cmp == 1:
                        wcv = a.cmp_len[cur_row]
                    tiles_c = _md_row_tiles(wv, wcv)
                    rem_cost = _md_row_cost(wv, wcv)
                    last_c = _md_row_last(wv, wcv)
            else:
                break
        # 2) 按 tile 吸收（FD 切分粒度；容差 = 当前块代价/2 ⇒ 行末块永不
        #    被块级吸收，切分点必落在行中，与 AscendC 不变式一致）。
        #    仅 support_fd 时切行；否则整行分配，跳过块级吸收。
        while support_fd == 1 and finished == 0:
            tcost = _md_tile_cost(wv, wcv, cur_tile)
            if limit + tcost // FA_TOLERANCE_RATIO >= cost + tcost:
                cost = cost + tcost
                blk = blk + 1
                rem_cost = rem_cost - tcost
                cur_tile = cur_tile + 1
                if cur_tile >= tiles_c:
                    cur_row = cur_row + 1
                    cur_tile = 0
                    if cur_row >= a.rows:
                        finished = 1
                    else:
                        wv = a.win_len[cur_row]
                        wcv = 0
                        if a.has_cmp == 1:
                            wcv = a.cmp_len[cur_row]
                        tiles_c = _md_row_tiles(wv, wcv)
                        rem_cost = _md_row_cost(wv, wcv)
                        last_c = _md_row_last(wv, wcv)
            else:
                break
        # 3) 强制分配 1 块保证推进（= ForceAssign；仅 support_fd 切行时需要）
        if blk == 0 and support_fd == 1:
            if finished == 0:
                tcost = _md_tile_cost(wv, wcv, cur_tile)
                cost = cost + tcost
                blk = 1
                rem_cost = rem_cost - tcost
                cur_tile = cur_tile + 1
                if cur_tile >= tiles_c:
                    cur_row = cur_row + 1
                    cur_tile = 0
                    if cur_row >= a.rows:
                        finished = 1
                    else:
                        wv = a.win_len[cur_row]
                        wcv = 0
                        if a.has_cmp == 1:
                            wcv = a.cmp_len[cur_row]
                        tiles_c = _md_row_tiles(wv, wcv)
                        rem_cost = _md_row_cost(wv, wcv)
                        last_c = _md_row_last(wv, wcv)
        fa_m_end[core] = cur_row
        fa_s2_end[core] = cur_tile
        unassigned = unassigned - cost
        # 4) 归约信息滞后记录：切分行被后续核吃完（游标行越过上一核终点行）
        if core > 0:
            if kv_split > 1:
                if cur_row != fa_m_end[core - 1]:
                    fd_row[num_fd] = fa_m_end[core - 1]
                    fd_ws[num_fd] = pre_fd
                    fd_num[num_fd] = kv_split
                    num_fd = num_fd + 1
                    pre_fd = pre_fd + kv_split
                    kv_split = 1
        # 5) 本核尾切行 ⇒ 挂起分部 +1
        if cur_tile > 0:
            kv_split = kv_split + 1
        used = core + 1
        core = core + 1

    # ---- GenMetadata：序列化到固定 1024 布局 ----
    i = 0
    while i < a.blocks:
        base = i * FA_METADATA_SIZE
        if i < used:
            a.metadata[base + FA_CORE_ENABLE_INDEX] = 1
            bs = 0
            be = 0
            for bi in range(0, a.batch + 1):
                if a.cu_q[bi] <= fa_m_start[i]:
                    bs = bi
                if a.cu_q[bi] <= fa_m_end[i]:
                    be = bi
            a.metadata[base + FA_BN2_START_INDEX] = bs
            a.metadata[base + FA_BN2_END_INDEX] = be
            a.metadata[base + FA_M_START_INDEX] = fa_m_start[i] - a.cu_q[bs]
            a.metadata[base + FA_S2_START_INDEX] = fa_s2_start[i]
            a.metadata[base + FA_M_END_INDEX] = fa_m_end[i] - a.cu_q[be]
            a.metadata[base + FA_S2_END_INDEX] = fa_s2_end[i]
            a.metadata[base + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = fa_first_fd[i]
        i = i + 1
    # FD 任务 t → AIV 2t（head 0-31）与 AIV 2t+1（head 32-63）；
    # 任务数 ≤ used-1 ≤ blocks-1 ⇒ 2*任务数 ≤ 2*blocks-2 < AIV 数，恒可容纳。
    j = 0
    while j < 2 * num_fd:
        base = FD_METADATA_BASE + j * FD_METADATA_SIZE
        t = j // 2
        a.metadata[base + FD_CORE_ENABLE_INDEX] = 1
        fb = 0
        for bi in range(0, a.batch + 1):
            if a.cu_q[bi] <= fd_row[t]:
                fb = bi
        a.metadata[base + FD_BN2_IDX_INDEX] = fb
        a.metadata[base + FD_M_IDX_INDEX] = fd_row[t] - a.cu_q[fb]
        a.metadata[base + FD_WORKSPACE_IDX_INDEX] = fd_ws[t]
        a.metadata[base + FD_WORKSPACE_NUM_INDEX] = fd_num[t]
        a.metadata[base + FD_M_START_INDEX] = (j % 2) * 32
        a.metadata[base + FD_M_NUM_INDEX] = 32
        j = j + 1
    # 全局 FD 使能计数（统一 barrier 门控）：无切分时为 0，kernel 跳过 barrier+归约。
    a.metadata[FD_USED_VEC_NUM_WORD] = 2 * num_fd
    return 0


def build_metadata(
    ori_lengths,
    cmp_lengths,
    cu_seqlens_q,
    blocks,
    *,
    has_cmp=True,
    support_fd=False,
    selective_fd_72=True,
    fd_min_saved_tiles=4,
):
    """Return 1024 Python integers; inputs are flattened host integer sequences."""
    ori = [int(v) for v in ori_lengths]
    cmp = [int(v) for v in cmp_lengths]
    rows = len(ori)
    if not 0 < rows <= 2147483647 or not 0 < blocks <= AIC_CORE_MAX_NUM:
        raise ValueError("rows and core count must fit positive int32 metadata")
    if len(cmp) != rows or any(v < 0 or v > 2147483647 for v in ori + cmp):
        raise ValueError(
            "topk lengths must have equal sizes and nonnegative int32 values"
        )
    cu = [0, rows] if cu_seqlens_q is None else [int(v) for v in cu_seqlens_q]
    if (
        len(cu) < 2
        or cu[0] != 0
        or cu[-1] != rows
        or any(a > b for a, b in zip(cu, cu[1:]))
    ):
        raise ValueError("cu_seqlens_q must be nondecreasing and span [0, rows]")
    mode = (
        2
        if selective_fd_72 and rows == 72 and blocks == 32 and has_cmp
        else int(bool(support_fd))
    )
    args = SimpleNamespace(
        win_len=ori,
        cmp_len=cmp,
        rows=rows,
        blocks=blocks,
        cu_q=cu,
        batch=len(cu) - 1,
        has_cmp=int(has_cmp),
        support_fd=mode,
        fd_min_saved_tiles=fd_min_saved_tiles,
        metadata=[0] * MQSMLA_METADATA_TOTAL_SIZE,
    )
    if _plan(args) != 0:
        raise ValueError("invalid metadata planner arguments")
    return args.metadata
