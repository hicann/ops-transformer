# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""确定性主循环。

preload3 / preload2 GS1S2 deterministic。循环里不再判断 ``deter``。
``physicalRound`` / ``readyRound`` / drain 仍是循环本地标量。
"""

import inspect

import pypto_pro.language as pl

from compute import fag_block_cube
from compute import fag_block_vec
from memory import fag_buffers
import fag_common
from schedule.deterministic import fag_deter_formulas, fag_deter_legacy, fag_deter_state
from schedule.deterministic import fag_deter_fixed
from schedule.deterministic import fag_deter_baseline
from schedule.deterministic import fag_deter_band_owner
from schedule.deterministic import fag_deter_band_plan
from schedule.deterministic.tnd import fag_deter_tnd_dense
from schedule.deterministic.tnd import fag_deter_tnd_sparse
from schedule.deterministic.tnd import fag_deter_tnd_plan
from schedule.deterministic.tnd import fag_deter_tnd_normalized
from schedule import fag_schedule

# codegen 内联时读的是本模块 globals() 里的裸名，不是 from-import 清单。
# module.func() 会被当成 IR op，所以这里把 helper / 常量绑成裸名。
# 其它 list 不灌进来，避免和必须具名的 ping-pong flag 抢提升。
_PINGPONG_LISTS = frozenset(
    {
        "SYNC_C1_TO_V2_FLAG",
        "SYNC_C2_TO_V2_FLAG",
        "SYNC_V2_TO_C1_FLAG",
        "SYNC_V2_TO_C2_FLAG",
    }
)


def _bind_helpers():
    g = globals()
    for mod in (
        fag_block_cube,
        fag_block_vec,
        fag_buffers,
        fag_common,
        fag_deter_formulas,
        fag_deter_legacy,
        fag_deter_state,
        fag_deter_fixed,
        fag_deter_baseline,
        fag_deter_band_owner,
        fag_deter_band_plan,
        fag_deter_tnd_dense,
        fag_deter_tnd_sparse,
        fag_deter_tnd_plan,
        fag_deter_tnd_normalized,
        fag_schedule,
    ):
        for name, val in vars(mod).items():
            if name.startswith("_") or inspect.ismodule(val):
                continue
            if isinstance(val, list) and name not in _PINGPONG_LISTS:
                continue
            g[name] = val


_bind_helpers()


def _run_preload3_cube_deter(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
):
    """Non-swizzle GS prefetches the next valid Q/dO task with one-round CV lag.

    Physical rounds and the DQ/DK/DV ordering tokens remain unchanged. A next
    valid task can lie beyond empty rounds; only its Q/dO are loaded early.
    Three live Q/dO tasks (next/current/previous) fit the D192 L1 allocation.
    """
    cBlockIdx = sched.cBlockIdx
    metadata_tensor = sched.meta
    coreNum = sched.coreNum
    cu_q_tensor = sched.cuQ
    cu_kv_tensor = sched.cuKv
    used_q_tensor = sched.usedQ
    used_kv_tensor = sched.usedKv
    tensor_q = gm.tensor_q
    tensor_k = gm.tensor_k
    tensor_k_nt = gm.tensor_k_nt
    tensor_dy = gm.tensor_dy
    tensor_v = gm.tensor_v
    tensor_dq_ws = gm.tensor_dq_ws
    tensor_dk_ws = gm.tensor_dk_ws
    tensor_dv_ws = gm.tensor_dv_ws
    mm1_res = buf.mm1_res
    mm2_res = buf.mm2_res
    q_l1 = buf.q_l1
    k_l1 = buf.k_l1
    k_l1_n = buf.k_l1_n
    v_l1 = buf.v_l1
    do_l1 = buf.do_l1
    q_l1_dk = buf.q_l1_dk
    do_l1_dv = buf.do_l1_dv
    left = buf.left
    right = buf.right
    acc = buf.acc
    acc_n64 = acc
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            swizzle == 0
            or (swizzle == 1 and deter == 1 and layout == 1 and deter_family == 12)
        )
    ):
        acc_n64 = make_l0c_acc_n64()
    dk_acc0 = buf.dk_acc0
    dv_acc0 = buf.dv_acc0
    dk_acc1 = buf.dk_acc1
    dv_acc1 = buf.dv_acc1
    ds_l1 = buf.ds_l1
    ds_t_l1 = buf.ds_t_l1
    p_t_l1 = buf.p_t_l1
    pipelineDelay = 2
    if swizzle == 0:
        # Current-task cursors share the prefetch/late-reader storage.
        q_for_mm = make_q_l1()
        do_for_mm = make_do_l1()
        pipelineDelay = 1
    taskId = 0
    fixTaskId = 0
    localIdx = 0
    denseCursor = pl.const(0, pl.DT_INT64)
    prevValidGIdx = pl.const(-1, pl.DT_INT64)
    physicalRound = pl.const(0, pl.DT_INT64)
    isLastLoop = 0
    drainDone = 0
    nFetched = 0
    tndPlan = init_deter_schedule(
        constInfo,
        coreNum,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
        metadata_tensor,
    )
    syncDq, syncKv, dqGroup = init_deter_sync_policy(tndPlan)
    firstRound, maxRound = deter_round_bounds(constInfo, coreNum, tndPlan)
    denseCursor = firstRound
    nextValidGIdx, nextValidRound = find_next_valid_deter(
        constInfo,
        cBlockIdx,
        coreNum,
        firstRound,
        maxRound,
        used_q_tensor,
        used_kv_tensor,
        cu_q_tensor,
        cu_kv_tensor,
        tndPlan,
        metadata_tensor,
    )
    if swizzle == 0:
        if nextValidGIdx >= 0:
            firstInfo = runInfos[0]
            set_run_info(
                constInfo,
                firstInfo,
                0,
                nextValidGIdx,
                0,
                0,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
            q_first = q_l1.next()
            do_first = do_l1.next()
            prefetch_qdo_l1(
                constInfo, firstInfo, tensor_q, tensor_dy, q_first, do_first
            )
    if syncDq == 1:
        pl.system.set_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DQ_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    if syncKv == 1:
        pl.system.set_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DK_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    if syncKv == 1:
        pl.system.set_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DV_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    while True:
        gIdx = pl.const(-1, pl.DT_INT64)
        physicalRound = denseCursor
        denseCursor = physicalRound + 1
        dqWait, dqSet = deter_dq_token_edges(
            physicalRound, firstRound, maxRound, pipelineDelay, dqGroup, syncDq
        )
        if physicalRound == nextValidRound:
            gIdx = nextValidGIdx
            nextValidGIdx, nextValidRound = find_next_valid_deter(
                constInfo,
                cBlockIdx,
                coreNum,
                physicalRound + 1,
                maxRound,
                used_q_tensor,
                used_kv_tensor,
                cu_q_tensor,
                cu_kv_tensor,
                tndPlan,
                metadata_tensor,
            )
        if physicalRound >= maxRound:
            isLastLoop = 1
        doFix = 0
        if gIdx >= 0:
            mm2_cur = mm2_res.next()
            mm1_cur = mm1_res.next()
            curInfo = runInfos[taskId % PRELOAD_TIMES]
            if swizzle == 0:
                # The lookup above advances only when this valid task is
                # consumed. Reserve/decode its successor once. Family13
                # D192/Dv128 issues its loads inside C1/C2 after current V/K.
                nextInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                if nextValidGIdx >= 0:
                    set_run_info(
                        constInfo,
                        nextInfo,
                        taskId + 1,
                        nextValidGIdx,
                        0,
                        localIdx + 1,
                        cu_q_tensor,
                        cu_kv_tensor,
                        used_q_tensor,
                        used_kv_tensor,
                    )
                    q_next = q_l1.next()
                    do_next = do_l1.next()
                    if not (
                        template == Tmpl.BN2GS1S2
                        and d_align == 192
                        and dv_align == 128
                        and swizzle == 0
                        and deter == 1
                        and layout == 1
                        and deter_family == 13
                    ):
                        prefetch_qdo_l1(
                            constInfo, nextInfo, tensor_q, tensor_dy, q_next, do_next
                        )
                q_cur = q_for_mm.next()
                do_cur = do_for_mm.next()
            else:
                q_cur = q_l1.next()
                do_cur = do_l1.next()
                set_run_info(
                    constInfo,
                    curInfo,
                    taskId,
                    gIdx,
                    0,
                    localIdx,
                    cu_q_tensor,
                    cu_kv_tensor,
                    used_q_tensor,
                    used_kv_tensor,
                )
            nFetched = nFetched + 1
            curInfo.readyRound = physicalRound + pipelineDelay
            set_deter_l0c_neighbors(
                constInfo, curInfo, prevValidGIdx, gIdx, nextValidGIdx
            )
            prevValidGIdx = gIdx
            if swizzle == 0:
                if (
                    template == Tmpl.BN2GS1S2
                    and d_align == 192
                    and dv_align == 128
                    and swizzle == 0
                    and deter == 1
                    and layout == 1
                    and deter_family == 13
                ):
                    # Match the deterministic AscendC front: C1 then C2.
                    # Writer current() is a typed next-task slot even when
                    # copyNext is false; no extra cursor advance occurs here.
                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX,
                            event_id=SYNC_V2_TO_C1_FLAG[taskId % 2],
                        )
                    iterate_mm_dyv(
                        constInfo,
                        curInfo,
                        tensor_dy,
                        tensor_v,
                        do_cur,
                        v_l1,
                        left,
                        right,
                        acc,
                        mm1_cur,
                        curInfo.loadKV,
                        0,
                        nextValidGIdx >= 0,
                        nextInfo,
                        do_l1.current(),
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_C1_TO_V2_FLAG[taskId % 2],
                    )
                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX,
                            event_id=SYNC_V2_TO_C2_FLAG[taskId % 2],
                        )
                    iterate_mm_qk(
                        constInfo,
                        curInfo,
                        tensor_q,
                        tensor_k,
                        q_cur,
                        k_l1,
                        left,
                        right,
                        acc,
                        mm2_cur,
                        curInfo.loadKV,
                        0,
                        nextValidGIdx >= 0,
                        nextInfo,
                        q_l1.current(),
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_C2_TO_V2_FLAG[taskId % 2],
                    )
                else:
                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX,
                            event_id=SYNC_V2_TO_C2_FLAG[taskId % 2],
                        )
                    iterate_mm_qk(
                        constInfo,
                        curInfo,
                        tensor_q,
                        tensor_k,
                        q_cur,
                        k_l1,
                        left,
                        right,
                        acc,
                        mm2_cur,
                        curInfo.loadKV,
                        0,  # Non-swizzle Q is already prefetched.
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_C2_TO_V2_FLAG[taskId % 2],
                    )
                    if taskId > 1:
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.FIX,
                            event_id=SYNC_V2_TO_C1_FLAG[taskId % 2],
                        )
                    iterate_mm_dyv(
                        constInfo,
                        curInfo,
                        tensor_dy,
                        tensor_v,
                        do_cur,
                        v_l1,
                        left,
                        right,
                        acc,
                        mm1_cur,
                        curInfo.loadKV,
                        0,  # Non-swizzle dO is already prefetched.
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_C1_TO_V2_FLAG[taskId % 2],
                    )
            taskId = taskId + 1
            localIdx = localIdx + 1

        fixInfo = runInfos[0]
        q_fix = q_l1.current()
        do_fix = do_l1.current()
        mm1_cur = mm1_res.current()
        if fixTaskId < taskId:
            fixInfo = runInfos[fixTaskId % PRELOAD_TIMES]
            if fixInfo.readyRound <= physicalRound:
                doFix = 1
                q_fix = q_l1_dk.next()
                do_fix = do_l1_dv.next()
        if doFix == 1:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V3_TO_C3_FLAG
            )
            if dqWait == 1:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DQ_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            iterate_mm_dsk(
                constInfo,
                fixInfo,
                tensor_k_nt,
                tensor_dq_ws,
                fixInfo.loadKV,
                ds_l1.current(),
                k_l1_n,
                left,
                right,
                acc,
                mm1_cur,
                cBlockIdx,
                acc_n64,
            )
            if dqSet == 1:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DQ_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            if syncKv == 1:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DK_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            if d_align <= 128 or (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and (
                    swizzle == 0
                    or (
                        swizzle == 1
                        and deter == 1
                        and layout == 1
                        and deter_family == 12
                    )
                )
            ):
                # Narrow dK and GS D192/Dv128 keep an owner-local resident sum.
                iterate_mm_dsq(
                    constInfo,
                    fixInfo,
                    ds_t_l1.current(),
                    q_fix,
                    left,
                    right,
                    dk_acc0,
                    dk_acc1,
                    fixInfo.dkvAcc,
                )
            else:
                iterate_mm_dsq(
                    constInfo,
                    fixInfo,
                    ds_t_l1.current(),
                    q_fix,
                    left,
                    right,
                    acc,
                    acc,
                    fixInfo.dkvAcc,
                    tensor_dk_ws,
                    tensor_q,
                )
            needFlush = fixInfo.flushDkv
            if d_align <= 128 or (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and (
                    swizzle == 0
                    or (
                        swizzle == 1
                        and deter == 1
                        and layout == 1
                        and deter_family == 12
                    )
                )
            ):
                if needFlush == 1:
                    flush_dk(
                        constInfo,
                        fixInfo,
                        tensor_dk_ws,
                        dk_acc0,
                        dk_acc1,
                    )
            if syncKv == 1:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DK_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C4_TO_V3_FLAG)
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V4_TO_C5_FLAG
            )
            if syncKv == 1:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DV_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            iterate_mm_pdy(
                constInfo,
                fixInfo,
                p_t_l1.current(),
                do_fix,
                left,
                right,
                dv_acc0,
                dv_acc1,
                fixInfo.dkvAcc,
            )
            if needFlush == 1:
                flush_dv(
                    constInfo,
                    fixInfo,
                    tensor_dv_ws,
                    dv_acc0,
                    dv_acc1,
                )
            if syncKv == 1:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DV_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C5_TO_V4_FLAG)
            fixTaskId = fixTaskId + 1
        if doFix == 0:
            # AscendC relays tokens only for a back-half logical round.
            # The logical consumer round follows the selected pipeline lag;
            # startup rounds have no dQ/dK/dV token to relay.
            if physicalRound >= firstRound + pipelineDelay:
                if dqWait == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DQ_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if dqSet == 1:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DQ_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if syncKv == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DK_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if syncKv == 1:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DK_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if syncKv == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DV_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if syncKv == 1:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DV_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
        if swizzle != 0:
            if gIdx >= 0:
                # Preserve readyRound/lag2 and all token edges, while consuming
                # the old C3 before the current C2 reuses a shared K slot.
                # Late Q/dO views have independent cursors from these writers.
                producerTaskId = taskId - 1
                curInfo = runInfos[producerTaskId % PRELOAD_TIMES]
                q_cur = q_l1.current()
                do_cur = do_l1.current()
                mm2_cur = mm2_res.current()
                mm1_cur = mm1_res.current()
                if producerTaskId > 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_V2_TO_C2_FLAG[producerTaskId % 2],
                    )
                iterate_mm_qk(
                    constInfo,
                    curInfo,
                    tensor_q,
                    tensor_k,
                    q_cur,
                    k_l1,
                    left,
                    right,
                    acc,
                    mm2_cur,
                    curInfo.loadKV,
                    1,  # Swizzle keeps current-task Q loading.
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_C2_TO_V2_FLAG[producerTaskId % 2],
                )
                if producerTaskId > 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_V2_TO_C1_FLAG[producerTaskId % 2],
                    )
                iterate_mm_dyv(
                    constInfo,
                    curInfo,
                    tensor_dy,
                    tensor_v,
                    do_cur,
                    v_l1,
                    left,
                    right,
                    acc,
                    mm1_cur,
                    curInfo.loadKV,
                    1,  # Swizzle keeps current-task dO loading.
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_C1_TO_V2_FLAG[producerTaskId % 2],
                )
        if isLastLoop == 1:
            drainDone = drainDone + 1
            if drainDone >= pipelineDelay:
                break
    if nFetched > 0:
        lastV2 = (nFetched - 1) % 2
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=SYNC_V2_TO_C1_FLAG[lastV2]
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=SYNC_V2_TO_C2_FLAG[lastV2]
        )
    if nFetched > 1:
        penultV2 = (nFetched - 2) % 2
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=SYNC_V2_TO_C1_FLAG[penultV2]
        )
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX, event_id=SYNC_V2_TO_C2_FLAG[penultV2]
        )
    if syncDq == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DQ_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    if syncKv == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DK_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    if syncKv == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DV_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )


def _run_preload3_vector_deter(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
    sink,
):
    cBlockIdx = sched.cBlockIdx
    metadata_tensor = sched.meta
    coreNum = sched.coreNum
    cu_q_tensor = sched.cuQ
    cu_kv_tensor = sched.cuKv
    used_q_tensor = sched.usedQ
    used_kv_tensor = sched.usedKv
    subIdx = sched.subIdx
    v1w = sched.v1w
    tensor_y = gm.tensor_y
    tensor_dy = gm.tensor_dy
    tensor_lse = gm.tensor_lse
    tensor_mask = gm.tensor_mask
    tensor_sfmg_ws = gm.tensor_sfmg_ws
    mm1_res = buf.mm1_res
    mm2_res = buf.mm2_res
    y_ub = buf.y_ub
    dy_ub = buf.dy_ub
    ds_out_ub = buf.ds_out_ub
    p_out_ub = buf.p_out_ub
    sfmg_ub = buf.sfmg_ub
    lse_ub = buf.lse_ub
    mask_ub = buf.mask_ub
    mask_pre_ub = buf.mask_pre_ub
    tensor_sinks = gm.tensor_sinks
    tensor_dsink_ws = gm.tensor_dsink_ws
    sink_part_ub = buf.sink_part_ub
    aivIdx = pl.get_block_idx()
    ds_l1 = buf.ds_l1
    p_l1 = buf.p_l1
    pipelineDelay = 2
    if swizzle == 0:
        pipelineDelay = 1
    taskId = 0
    vecTaskId = 0
    localIdx = 0
    denseCursor = pl.const(0, pl.DT_INT64)
    prevValidGIdx = pl.const(-1, pl.DT_INT64)
    physicalRound = pl.const(0, pl.DT_INT64)
    isLastLoop = 0
    drainDone = 0
    nFetched = 0
    tndPlan = init_deter_schedule(
        constInfo,
        coreNum,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
        metadata_tensor,
    )
    firstRound, maxRound = deter_round_bounds(constInfo, coreNum, tndPlan)
    denseCursor = firstRound
    nextValidGIdx, nextValidRound = find_next_valid_deter(
        constInfo,
        cBlockIdx,
        coreNum,
        firstRound,
        maxRound,
        used_q_tensor,
        used_kv_tensor,
        cu_q_tensor,
        cu_kv_tensor,
        tndPlan,
        metadata_tensor,
    )
    while True:
        gIdx = pl.const(-1, pl.DT_INT64)
        physicalRound = denseCursor
        denseCursor = physicalRound + 1
        if physicalRound == nextValidRound:
            gIdx = nextValidGIdx
            nextValidGIdx, nextValidRound = find_next_valid_deter(
                constInfo,
                cBlockIdx,
                coreNum,
                physicalRound + 1,
                maxRound,
                used_q_tensor,
                used_kv_tensor,
                cu_q_tensor,
                cu_kv_tensor,
                tndPlan,
                metadata_tensor,
            )
        if physicalRound >= maxRound:
            isLastLoop = 1
        skipPipe = 0
        if skipPipe == 0:
            if gIdx >= 0:
                curInfo = runInfos[taskId % PRELOAD_TIMES]
                set_run_info(
                    constInfo,
                    curInfo,
                    taskId,
                    gIdx,
                    subIdx,
                    localIdx,
                    cu_q_tensor,
                    cu_kv_tensor,
                    used_q_tensor,
                    used_kv_tensor,
                )
                nFetched = nFetched + 1
                curInfo.readyRound = physicalRound + pipelineDelay
                set_deter_l0c_neighbors(
                    constInfo, curInfo, prevValidGIdx, gIdx, nextValidGIdx
                )
                prevValidGIdx = gIdx
                taskId = taskId + 1
                localIdx = localIdx + 1
            vecInfo = runInfos[0]
            lse_t = lse_ub.current()
            needSyncDkMM = 0
            setV2C = 1
            doVec = 0
            vecEventTaskId = taskId
            if vecTaskId < taskId:
                vecInfo = runInfos[vecTaskId % PRELOAD_TIMES]
                if vecInfo.readyRound <= physicalRound:
                    doVec = 1
                    vecEventTaskId = vecTaskId
                    if vecTaskId > 0:
                        needSyncDkMM = 1
            if doVec == 1:
                mm2_t = mm2_res.next()
                mm1_t = mm1_res.next()
                lse_t = lse_ub.next()
                process_vec1(
                    constInfo,
                    vecInfo,
                    subIdx,
                    tensor_y,
                    tensor_dy,
                    tensor_sfmg_ws,
                    y_ub,
                    dy_ub,
                    sfmg_ub.current(),
                    v1w,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V,
                    event_id=SYNC_C2_TO_V2_FLAG[vecEventTaskId % 2],
                )
                if mask_mode != 0:
                    process_vec2(
                        constInfo,
                        vecInfo,
                        subIdx,
                        tensor_lse,
                        mm2_t,
                        lse_t,
                        tensor_mask,
                        mask_ub.current(),
                        mask_pre_ub.current(),
                    )
                else:
                    process_vec2(
                        constInfo,
                        vecInfo,
                        subIdx,
                        tensor_lse,
                        mm2_t,
                        lse_t,
                        tensor_lse,
                        mask_ub.current(),
                        mask_pre_ub.current(),
                    )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V,
                    event_id=SYNC_C1_TO_V2_FLAG[vecEventTaskId % 2],
                )
                if needSyncDkMM == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C5_TO_V4_FLAG
                    )
                process_vec4(
                    constInfo,
                    vecInfo,
                    subIdx,
                    p_l1.current(),
                    mm2_t,
                    p_out_ub.current(),
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_V4_TO_C5_FLAG
                )
                if needSyncDkMM == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C4_TO_V3_FLAG
                    )
                process_vec3(
                    constInfo,
                    vecInfo,
                    subIdx,
                    ds_l1.current(),
                    mm1_t,
                    mm2_t,
                    sfmg_ub.current(),
                    ds_out_ub.current(),
                )
                # GS1S2 C3/C4 read L1 and write GM/L0C, not mm1/mm2 UB.
                # dsink is computed here before the set_cross_core, consistent with BN2.
                if sink == 1:
                    headQ = vecInfo.n2oIdx * constInfo.gSize + vecInfo.goIdx
                    sink_val = pl.getval(tensor_sinks, headQ)
                    process_sink_partial(
                        constInfo,
                        vecInfo,
                        tensor_dsink_ws,
                        aivIdx,
                        mm2_t,
                        mm1_t,
                        lse_t,
                        sink_part_ub.current(),
                        sink_val,
                    )
                if setV2C == 1:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.V,
                        event_id=SYNC_V2_TO_C1_FLAG[vecEventTaskId % 2],
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.V,
                        event_id=SYNC_V2_TO_C2_FLAG[vecEventTaskId % 2],
                    )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_V3_TO_C3_FLAG
                )
                vecTaskId = vecTaskId + 1
        if isLastLoop == 1:
            drainDone = drainDone + 1
            if drainDone >= pipelineDelay:
                if nFetched > 0:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C5_TO_V4_FLAG
                    )
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C4_TO_V3_FLAG
                    )
                break


def _run_preload2_cube_gs_deter(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
):
    cBlockIdx = sched.cBlockIdx
    metadata_tensor = sched.meta
    coreNum = sched.coreNum
    cu_q_tensor = sched.cuQ
    cu_kv_tensor = sched.cuKv
    used_q_tensor = sched.usedQ
    used_kv_tensor = sched.usedKv
    tensor_q = gm.tensor_q
    tensor_k = gm.tensor_k
    tensor_k_nt = gm.tensor_k_nt
    tensor_dy = gm.tensor_dy
    tensor_v = gm.tensor_v
    tensor_dq_ws = gm.tensor_dq_ws
    tensor_dk_ws = gm.tensor_dk_ws
    tensor_dv_ws = gm.tensor_dv_ws
    mm1_res = buf.mm1_res
    mm2_res = buf.mm2_res
    q_l1 = buf.q_l1
    k_l1 = buf.k_l1
    k_l1_n = buf.k_l1_n
    v_l1 = buf.v_l1
    do_l1 = buf.do_l1
    q_l1_dk = buf.q_l1_dk
    do_l1_dv = buf.do_l1_dv
    left = buf.left
    right = buf.right
    acc = buf.acc
    dv_acc0 = buf.dv_acc0
    dv_acc1 = buf.dv_acc1
    ds_l1 = buf.ds_l1
    ds_t_l1 = buf.ds_t_l1
    p_t_l1 = buf.p_t_l1
    taskId = 0
    fixTaskId = 0
    localIdx = 0
    denseCursor = pl.const(0, pl.DT_INT64)
    prevValidGIdx = pl.const(-1, pl.DT_INT64)
    physicalRound = pl.const(0, pl.DT_INT64)
    isLastLoop = 0
    drainDone = 0
    tndPlan = init_deter_schedule(
        constInfo,
        coreNum,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
        metadata_tensor,
    )
    syncDq, syncKv, dqGroup = init_deter_sync_policy(tndPlan)
    firstRound, maxRound = deter_round_bounds(constInfo, coreNum, tndPlan)
    denseCursor = firstRound
    nextValidGIdx, nextValidRound = find_next_valid_deter(
        constInfo,
        cBlockIdx,
        coreNum,
        firstRound,
        maxRound,
        used_q_tensor,
        used_kv_tensor,
        cu_q_tensor,
        cu_kv_tensor,
        tndPlan,
        metadata_tensor,
    )
    if syncDq == 1:
        pl.system.set_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DQ_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    if syncKv == 1:
        pl.system.set_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DK_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    if syncKv == 1:
        pl.system.set_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DV_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    while True:
        gIdx = pl.const(-1, pl.DT_INT64)
        physicalRound = denseCursor
        denseCursor = physicalRound + 1
        dqWait, dqSet = deter_dq_token_edges(
            physicalRound, firstRound, maxRound, 1, dqGroup, syncDq
        )
        if physicalRound == nextValidRound:
            gIdx = nextValidGIdx
            nextValidGIdx, nextValidRound = find_next_valid_deter(
                constInfo,
                cBlockIdx,
                coreNum,
                physicalRound + 1,
                maxRound,
                used_q_tensor,
                used_kv_tensor,
                cu_q_tensor,
                cu_kv_tensor,
                tndPlan,
                metadata_tensor,
            )
        if physicalRound >= maxRound:
            isLastLoop = 1
        doFix = 0
        mm2_cur = mm2_res.current()
        mm1_cur = mm1_res.current()
        q_cur = q_l1.current()
        do_cur = do_l1.current()
        if gIdx >= 0:
            mm2_cur = mm2_res.next()
            mm1_cur = mm1_res.next()
            q_cur = q_l1.next()
            do_cur = do_l1.next()
            curInfo = runInfos[0]
            curInfo = runInfos[taskId % 2]
            set_run_info(
                constInfo,
                curInfo,
                taskId,
                gIdx,
                0,
                localIdx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
            curInfo.readyRound = physicalRound + 1
            set_deter_l0c_neighbors(
                constInfo, curInfo, prevValidGIdx, gIdx, nextValidGIdx
            )
            prevValidGIdx = gIdx
            iterate_mm_qk(
                constInfo,
                curInfo,
                tensor_q,
                tensor_k,
                q_cur,
                k_l1,
                left,
                right,
                acc,
                mm2_cur,
                curInfo.loadKV,
            )
            pl.system.set_cross_core(
                pipe=pl.PipeType.FIX, event_id=SYNC_C2_TO_V2_FLAG[taskId % 2]
            )
            iterate_mm_dyv(
                constInfo,
                curInfo,
                tensor_dy,
                tensor_v,
                do_cur,
                v_l1,
                left,
                right,
                acc,
                mm1_cur,
                curInfo.loadKV,
            )
            pl.system.set_cross_core(
                pipe=pl.PipeType.FIX, event_id=SYNC_C1_TO_V2_FLAG[taskId % 2]
            )
            taskId = taskId + 1
            localIdx = localIdx + 1
        fixInfo = runInfos[0]
        q_fix = q_cur
        do_fix = do_cur
        if fixTaskId < taskId:
            fixInfo = runInfos[fixTaskId % 2]
            if fixInfo.readyRound <= physicalRound:
                doFix = 1
                q_fix = q_l1_dk.next()
                do_fix = do_l1_dv.next()
        if doFix == 1:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V3_TO_C3_FLAG
            )
            if dqWait == 1:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DQ_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            iterate_mm_dsk(
                constInfo,
                fixInfo,
                tensor_k_nt,
                tensor_dq_ws,
                fixInfo.loadKV,
                ds_l1.current(),
                k_l1_n,
                left,
                right,
                acc,
                mm1_cur,
                cBlockIdx,
            )
            if dqSet == 1:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DQ_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            if syncKv == 1:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DK_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            iterate_mm_dsq(
                constInfo,
                fixInfo,
                ds_t_l1.current(),
                q_fix,
                left,
                right,
                acc,
                acc,
                0,
                tensor_dk_ws,
                tensor_q,
            )
            if syncKv == 1:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DK_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C4_TO_V3_FLAG)
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V4_TO_C5_FLAG
            )
            if syncKv == 1:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DV_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            if d_align == 256 and swizzle == 0:
                iterate_mm_pdy_gs_dv256(
                    constInfo,
                    fixInfo,
                    p_t_l1.current(),
                    do_fix,
                    left,
                    right,
                    acc,
                    dv_acc0,
                    dv_acc1,
                    tensor_dv_ws,
                    tensor_dy,
                )
                if constInfo.s1Size > 64 and constInfo.s2Size > 64:
                    # Flush remains inside the optional owner-order FIX token.
                    # Empty physical rounds preserve the resident dV sum.
                    if fixInfo.flushDkv == 1:
                        flush_dv(constInfo, fixInfo, tensor_dv_ws, dv_acc0, dv_acc1)
            else:
                iterate_mm_pdy(
                    constInfo,
                    fixInfo,
                    p_t_l1.current(),
                    do_fix,
                    left,
                    right,
                    acc,
                    acc,
                    0,
                    tensor_dv_ws,
                    tensor_dy,
                )
            if syncKv == 1:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX,
                    event_id=SYNC_DV_DETER_FIX_FLAG,
                    sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C5_TO_V4_FLAG)
            fixTaskId = fixTaskId + 1
        if doFix == 0:
            if physicalRound > firstRound:
                if dqWait == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DQ_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if dqSet == 1:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DQ_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if syncKv == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DK_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if syncKv == 1:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DK_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if syncKv == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DV_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
                if syncKv == 1:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX,
                        event_id=SYNC_DV_DETER_FIX_FLAG,
                        sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
                    )
        if isLastLoop == 1:
            drainDone = drainDone + 1
            if drainDone >= 1:
                break
    if syncDq == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DQ_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    if syncKv == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DK_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )
    if syncKv == 1:
        pl.system.wait_cross_core(
            pipe=pl.PipeType.FIX,
            event_id=SYNC_DV_DETER_FIX_FLAG,
            sync_mode=pl.CrossCoreSyncMode.INTER_BLOCK,
        )


def _run_preload2_vector_gs_deter(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
    sink,
):
    cBlockIdx = sched.cBlockIdx
    metadata_tensor = sched.meta
    coreNum = sched.coreNum
    cu_q_tensor = sched.cuQ
    cu_kv_tensor = sched.cuKv
    used_q_tensor = sched.usedQ
    used_kv_tensor = sched.usedKv
    subIdx = sched.subIdx
    v1w = sched.v1w
    tensor_y = gm.tensor_y
    tensor_dy = gm.tensor_dy
    tensor_lse = gm.tensor_lse
    tensor_mask = gm.tensor_mask
    tensor_sfmg_ws = gm.tensor_sfmg_ws
    mm1_res = buf.mm1_res
    mm2_res = buf.mm2_res
    y_ub = buf.y_ub
    dy_ub = buf.dy_ub
    ds_out_ub = buf.ds_out_ub
    p_out_ub = buf.p_out_ub
    sfmg_ub = buf.sfmg_ub
    lse_ub = buf.lse_ub
    mask_ub = buf.mask_ub
    mask_pre_ub = buf.mask_pre_ub
    tensor_sinks = gm.tensor_sinks
    tensor_dsink_ws = gm.tensor_dsink_ws
    sink_part_ub = buf.sink_part_ub
    aivIdx = pl.get_block_idx()
    ds_l1 = buf.ds_l1
    p_l1 = buf.p_l1
    taskId = 0
    vecTaskId = 0
    localIdx = 0
    denseCursor = pl.const(0, pl.DT_INT64)
    prevValidGIdx = pl.const(-1, pl.DT_INT64)
    physicalRound = pl.const(0, pl.DT_INT64)
    isLastLoop = 0
    drainDone = 0
    nFetched = 0
    tndPlan = init_deter_schedule(
        constInfo,
        coreNum,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
        metadata_tensor,
    )
    firstRound, maxRound = deter_round_bounds(constInfo, coreNum, tndPlan)
    denseCursor = firstRound
    nextValidGIdx, nextValidRound = find_next_valid_deter(
        constInfo,
        cBlockIdx,
        coreNum,
        firstRound,
        maxRound,
        used_q_tensor,
        used_kv_tensor,
        cu_q_tensor,
        cu_kv_tensor,
        tndPlan,
        metadata_tensor,
    )
    while True:
        gIdx = pl.const(-1, pl.DT_INT64)
        physicalRound = denseCursor
        denseCursor = physicalRound + 1
        if physicalRound == nextValidRound:
            gIdx = nextValidGIdx
            nextValidGIdx, nextValidRound = find_next_valid_deter(
                constInfo,
                cBlockIdx,
                coreNum,
                physicalRound + 1,
                maxRound,
                used_q_tensor,
                used_kv_tensor,
                cu_q_tensor,
                cu_kv_tensor,
                tndPlan,
                metadata_tensor,
            )
        if physicalRound >= maxRound:
            isLastLoop = 1
        if gIdx >= 0:
            curInfo = runInfos[taskId % 2]
            set_run_info(
                constInfo,
                curInfo,
                taskId,
                gIdx,
                subIdx,
                localIdx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
            nFetched = nFetched + 1
            curInfo.readyRound = physicalRound + 1
            set_deter_l0c_neighbors(
                constInfo, curInfo, prevValidGIdx, gIdx, nextValidGIdx
            )
            prevValidGIdx = gIdx
            taskId = taskId + 1
            localIdx = localIdx + 1
        vecInfo = runInfos[0]
        mm1_t = mm1_res.current()
        mm2_t = mm2_res.current()
        lse_t = lse_ub.current()
        c2ev = 0
        needSyncDkMM = 0
        doVec = 0
        if vecTaskId < taskId:
            vecInfo = runInfos[vecTaskId % 2]
            if vecInfo.readyRound <= physicalRound:
                doVec = 1
                c2ev = vecTaskId % 2
                if vecTaskId > 0:
                    needSyncDkMM = 1
        if doVec == 1:
            mm1_t = mm1_res.next()
            mm2_t = mm2_res.next()
            lse_t = lse_ub.next()
            process_vec1(
                constInfo,
                vecInfo,
                subIdx,
                tensor_y,
                tensor_dy,
                tensor_sfmg_ws,
                y_ub,
                dy_ub,
                sfmg_ub.current(),
                v1w,
            )
            pl.system.wait_cross_core(
                pipe=pl.PipeType.V, event_id=SYNC_C2_TO_V2_FLAG[c2ev]
            )
            if mask_mode != 0:
                process_vec2(
                    constInfo,
                    vecInfo,
                    subIdx,
                    tensor_lse,
                    mm2_t,
                    lse_t,
                    tensor_mask,
                    mask_ub.current(),
                    mask_pre_ub.current(),
                )
            else:
                process_vec2(
                    constInfo,
                    vecInfo,
                    subIdx,
                    tensor_lse,
                    mm2_t,
                    lse_t,
                    tensor_lse,
                    mask_ub.current(),
                    mask_pre_ub.current(),
                )
            pl.system.wait_cross_core(
                pipe=pl.PipeType.V, event_id=SYNC_C1_TO_V2_FLAG[c2ev]
            )
            if needSyncDkMM == 1:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_C5_TO_V4_FLAG
                )
            process_vec4(
                constInfo,
                vecInfo,
                subIdx,
                p_l1.current(),
                mm2_t,
                p_out_ub.current(),
            )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=SYNC_V4_TO_C5_FLAG)
            if needSyncDkMM == 1:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_C4_TO_V3_FLAG
                )
            process_vec3(
                constInfo,
                vecInfo,
                subIdx,
                ds_l1.current(),
                mm1_t,
                mm2_t,
                sfmg_ub.current(),
                ds_out_ub.current(),
            )
            # GS1S2 C3/C4 read L1 and write GM/L0C, not mm1/mm2 UB.
            # dsink is computed here before the set_cross_core, consistent with BN2.
            if sink == 1:
                headQ = vecInfo.n2oIdx * constInfo.gSize + vecInfo.goIdx
                sink_val = pl.getval(tensor_sinks, headQ)
                process_sink_partial(
                    constInfo,
                    vecInfo,
                    tensor_dsink_ws,
                    aivIdx,
                    mm2_t,
                    mm1_t,
                    lse_t,
                    sink_part_ub.current(),
                    sink_val,
                )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=SYNC_V3_TO_C3_FLAG)
            vecTaskId = vecTaskId + 1
        if isLastLoop == 1:
            drainDone = drainDone + 1
            if drainDone >= 1:
                if nFetched > 0:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C5_TO_V4_FLAG
                    )
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C4_TO_V3_FLAG
                    )
                break
