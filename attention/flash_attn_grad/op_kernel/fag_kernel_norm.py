# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""非确定性主循环。

preload3 / preload2 GS1S2 normal，以及隔离后的 BN2。循环里不再判断 ``deter``。
``taskId`` / ``denseCursor`` / drain 仍是循环本地标量。
"""

import inspect

import pypto_pro.language as pl

import fag_block_cube
import fag_block_vec
import fag_buffers
import fag_common
import fag_schedule

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
        fag_schedule,
    ):
        for name, val in vars(mod).items():
            if name.startswith("_") or inspect.ismodule(val):
                continue
            if isinstance(val, list) and name not in _PINGPONG_LISTS:
                continue
            g[name] = val


_bind_helpers()


def _run_preload3_cube_normal(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
):
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
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
    dk_acc0 = buf.dk_acc0
    dv_acc0 = buf.dv_acc0
    dk_acc1 = buf.dk_acc1
    dv_acc1 = buf.dv_acc1
    ds_l1 = buf.ds_l1
    ds_t_l1 = buf.ds_t_l1
    p_t_l1 = buf.p_t_l1
    taskId = 0
    localIdx = 0
    denseCursor = blockStart
    isLastLoop = 0
    drainDone = 0
    nFetched = 0
    while True:
        gIdx = pl.const(-1, pl.DT_INT64)
        # Drain pending work without rescanning the trailing invalid tiles.
        if isLastLoop == 0:
            gIdx = next_work_gidx(
                constInfo,
                cBlockIdx,
                coreNum,
                blockStart,
                blockEnd,
                localIdx,
                denseCursor,
                numBlocks,
            )
            if gIdx < 0:
                isLastLoop = 1
            else:
                if swizzle == 0:
                    denseCursor = gIdx + 1
        doFix = 0
        mm2_cur = mm2_res.next()
        mm1_cur = mm1_res.next()
        q_cur = q_l1.next()
        do_cur = do_l1.next()
        if gIdx >= 0:
            curInfo = runInfos[taskId % PRELOAD_TIMES]
            set_run_info(constInfo, curInfo, taskId, gIdx, 0, localIdx)
            nFetched = nFetched + 1
            apply_l0c_s2(runInfos, curInfo, taskId, PRELOAD_TIMES)
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
            )
            pl.system.set_cross_core(
                pipe=pl.PipeType.FIX,
                event_id=SYNC_C1_TO_V2_FLAG[taskId % 2],
            )
        fixInfo = runInfos[0]
        q_fix = q_l1.current()
        do_fix = do_l1.current()
        mm1_cur = mm1_res.current()
        if taskId > 1:
            doFix = 1
            fixInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
            q_fix = q_l1_dk.next()
            do_fix = do_l1_dv.next()
        if doFix == 1:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V4_TO_C5_FLAG
            )
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V3_TO_C3_FLAG
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
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C5_TO_V4_FLAG)
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
            needFlush = fixInfo.flushDkv
            if taskId - 2 >= nFetched - 1:
                needFlush = 1
            if needFlush == 1:
                flush_dkv(
                    constInfo,
                    fixInfo,
                    tensor_dk_ws,
                    tensor_dv_ws,
                    dk_acc0,
                    dk_acc1,
                    dv_acc0,
                    dv_acc1,
                )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C4_TO_V3_FLAG)
        if isLastLoop == 1:
            drainDone = drainDone + 1
            if drainDone >= 2:
                break
        if gIdx >= 0:
            taskId = taskId + 1
            localIdx = localIdx + 1
        else:
            if isLastLoop == 1:
                taskId = taskId + 1
                localIdx = localIdx + 1


def _run_preload3_vector_normal(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
):
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
    subIdx = sched.subIdx
    v1w = sched.v1w
    tensor_y = gm.tensor_y
    tensor_dy = gm.tensor_dy
    tensor_lse = gm.tensor_lse
    tensor_mask = gm.tensor_mask
    mm1_res = buf.mm1_res
    mm2_res = buf.mm2_res
    y_ub = buf.y_ub
    dy_ub = buf.dy_ub
    nzp_ub = buf.nzp_ub
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
    localIdx = 0
    denseCursor = blockStart
    isLastLoop = 0
    drainDone = 0
    nFetched = 0
    while True:
        mm2_res.next()
        mm1_res.next()
        # Start V1 for task N-2 before scalar scheduling for task N.
        if taskId > 1:
            prevInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
            process_vec1(
                constInfo,
                prevInfo,
                subIdx,
                tensor_y,
                tensor_dy,
                y_ub.current(),
                dy_ub.current(),
                sfmg_ub.current(),
                v1w,
            )
        gIdx = pl.const(-1, pl.DT_INT64)
        if isLastLoop == 0:
            gIdx = next_work_gidx(
                constInfo,
                cBlockIdx,
                coreNum,
                blockStart,
                blockEnd,
                localIdx,
                denseCursor,
                numBlocks,
            )
            if gIdx < 0:
                isLastLoop = 1
            else:
                if swizzle == 0:
                    denseCursor = gIdx + 1
        skipPipe = 0
        if skipPipe == 0:
            if gIdx >= 0:
                curInfo = runInfos[taskId % PRELOAD_TIMES]
                set_run_info(constInfo, curInfo, taskId, gIdx, subIdx, localIdx)
                nFetched = nFetched + 1
                apply_l0c_s2(runInfos, curInfo, taskId, PRELOAD_TIMES)
                lse_ub.next()
            vecInfo = runInfos[0]
            lse_t = lse_ub.current()
            needSyncDkMM = 0
            setV2C = 1
            doVec = 0
            vecEventTaskId = taskId
            if taskId > 1:
                doVec = 1
                vecInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
                if taskId > 2:
                    needSyncDkMM = 1
                if isLastLoop == 1:
                    setV2C = 0
            if doVec == 1:
                mm2_t = mm2_res.current()
                mm1_t = mm1_res.current()
                lse_t = lse_ub.previous()
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
                    nzp_ub.current(),
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
                    nzp_ub.current(),
                )
                # GS1S2 C3/C4 read L1 and write GM/L0C, not mm1/mm2 UB.
                # Let Cube overlap dsink; release UB only after its last read.
                pl.system.set_cross_core(
                    pipe=pl.PipeType.MTE3, event_id=SYNC_V3_TO_C3_FLAG
                )
                if constInfo.sinkOptional == 1:
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
        if isLastLoop == 1:
            drainDone = drainDone + 1
            if drainDone >= 2:
                if nFetched > 0:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C5_TO_V4_FLAG
                    )
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=SYNC_C4_TO_V3_FLAG
                    )
                break
        if gIdx >= 0:
            taskId = taskId + 1
            localIdx = localIdx + 1
        else:
            if isLastLoop == 1:
                taskId = taskId + 1
                localIdx = localIdx + 1


def _run_preload2_cube_bn2(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
):
    meta = sched.meta
    fagOff = sched.fagOff
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
    tensor_q = gm.tensor_q
    tensor_k = gm.tensor_k
    tensor_k_nt = gm.tensor_k_nt
    tensor_dy = gm.tensor_dy
    tensor_v = gm.tensor_v
    tensor_dq_ws = gm.tensor_dq_ws
    tensor_dv_out = gm.tensor_dv_out
    mm1_res = buf.mm1_res
    mm2_res = buf.mm2_res
    q_l1 = buf.q_l1
    k_l1 = buf.k_l1
    k_l1_n = buf.k_l1_n
    v_l1 = buf.v_l1
    do_l1 = buf.do_l1
    left = buf.left
    right = buf.right
    acc = buf.acc
    dk_acc = buf.dk_acc
    dv_acc = buf.dv_acc
    ds_l1 = buf.ds_l1
    ds_t_l1 = buf.ds_t_l1
    p_t_l1 = buf.p_t_l1
    taskId = 0
    localIdx = 0
    denseCursor = blockStart
    isLastLoop = 0
    drainDone = 0
    roundId = pl.const(0, pl.DT_INT64)
    while True:
        gIdx = pl.const(-1, pl.DT_INT64)
        fetchedGIdx, fetchedDenseCursor = fetch_work_gidx(
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
        )
        if isLastLoop == 0:
            gIdx = fetchedGIdx
            denseCursor = fetchedDenseCursor
            if gIdx < 0:
                isLastLoop = 1
        doFix = 0
        mm2_cur = mm2_res.next()
        mm1_cur = mm1_res.next()
        q_cur = q_l1.next()
        do_cur = do_l1.next()
        curInfo = runInfos[0]
        if gIdx >= 0:
            curInfo = runInfos[taskId % 2]
            set_run_info(constInfo, curInfo, taskId, gIdx, 0, localIdx)
            apply_l0c_s2(runInfos, curInfo, taskId, 2)
            if taskId > 1:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.FIX, event_id=SYNC_DETER_FIX_FLAG
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
        fixInfo = runInfos[0]
        q_fix = q_cur
        do_fix = do_cur
        if taskId > 0:
            doFix = 1
            fixInfo = runInfos[(taskId + 1) % 2]
            q_fix = q_l1.previous()
            do_fix = do_l1.previous()
        if doFix == 1:
            mm1_prev = mm1_res.previous()
            mm2_prev = mm2_res.previous()
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V3_TO_C3_FLAG
            )
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V4_TO_C5_FLAG
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
                mm1_prev,
                cBlockIdx,
            )
            if is_bn2_multiblk == 1:
                if fixInfo.dqLast == 1:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_C3_TO_V5_FLAG
                    )
                iterate_mm_dsq(
                    constInfo,
                    fixInfo,
                    ds_t_l1.current(),
                    q_fix,
                    left,
                    right,
                    dk_acc,
                    dk_acc,
                    fixInfo.dkvAcc,
                )
                if fixInfo.flushDkv == 1:
                    pl.set_validshape(mm2_prev, [VECTOR_BASEM, VECTOR_BASEN])
                    pl.move(
                        mm2_prev,
                        dk_acc.current(),
                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
                    )
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_C4_TO_V6_FLAG
                    )
                iterate_mm_pdy(
                    constInfo,
                    fixInfo,
                    p_t_l1.current(),
                    do_fix,
                    left,
                    right,
                    dv_acc,
                    dv_acc,
                    fixInfo.dkvAcc,
                )
                if fixInfo.flushDkv == 1:
                    pl.store(
                        tensor_dv_out,
                        dv_acc.current(),
                        [
                            fixInfo.idx0KV,
                            fixInfo.keyOffset,
                            fixInfo.idx2KV,
                            0,
                        ],
                        order=[1, 3],
                        atomic=pl.AtomicType.AtomicNone,
                    )
            else:
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX, event_id=SYNC_C3_TO_V5_FLAG
                )
                iterate_mm_dsq(
                    constInfo,
                    fixInfo,
                    ds_t_l1.current(),
                    q_fix,
                    left,
                    right,
                    dk_acc,
                    dk_acc,
                    0,
                )
                pl.set_validshape(mm2_prev, [VECTOR_BASEM, VECTOR_BASEN])
                pl.move(
                    mm2_prev,
                    dk_acc.current(),
                    acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.FIX, event_id=SYNC_C4_TO_V6_FLAG
                )
                iterate_mm_pdy(
                    constInfo,
                    fixInfo,
                    p_t_l1.current(),
                    do_fix,
                    left,
                    right,
                    dv_acc,
                    dv_acc,
                    0,
                )
                pl.store(
                    tensor_dv_out,
                    dv_acc.current(),
                    [fixInfo.idx0KV, fixInfo.keyOffset, fixInfo.idx2KV, 0],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicNone,
                )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C5_TO_V4_FLAG)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C4_TO_V3_FLAG)
        if isLastLoop == 1:
            drainDone = drainDone + 1
            if drainDone >= 1:
                break
        if gIdx >= 0:
            taskId = taskId + 1
            localIdx = localIdx + 1
        else:
            if isLastLoop == 1:
                taskId = taskId + 1
                localIdx = localIdx + 1


def _run_preload2_cube_gs_normal(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
):
    meta = sched.meta
    fagOff = sched.fagOff
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
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
    left = buf.left
    right = buf.right
    acc = buf.acc
    ds_l1 = buf.ds_l1
    ds_t_l1 = buf.ds_t_l1
    p_t_l1 = buf.p_t_l1
    taskId = 0
    localIdx = 0
    denseCursor = blockStart
    isLastLoop = 0
    drainDone = 0
    roundId = pl.const(0, pl.DT_INT64)
    while True:
        gIdx = pl.const(-1, pl.DT_INT64)
        fetchedGIdx, fetchedDenseCursor = fetch_work_gidx(
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
        )
        if isLastLoop == 0:
            gIdx = fetchedGIdx
            denseCursor = fetchedDenseCursor
            if gIdx < 0:
                isLastLoop = 1
        doFix = 0
        mm2_cur = mm2_res.next()
        mm1_cur = mm1_res.next()
        q_cur = q_l1.next()
        do_cur = do_l1.next()
        curInfo = runInfos[0]
        if gIdx >= 0:
            curInfo = runInfos[taskId % 2]
            set_run_info(constInfo, curInfo, taskId, gIdx, 0, localIdx)
            apply_l0c_s2(runInfos, curInfo, taskId, 2)
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
        fixInfo = runInfos[0]
        q_fix = q_cur
        do_fix = do_cur
        if taskId > 0:
            doFix = 1
            fixInfo = runInfos[(taskId + 1) % 2]
            q_fix = q_l1.previous()
            do_fix = do_l1.previous()
        if doFix == 1:
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V4_TO_C5_FLAG
            )
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V3_TO_C3_FLAG
            )
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
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C5_TO_V4_FLAG)
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
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C4_TO_V3_FLAG)
        if isLastLoop == 1:
            drainDone = drainDone + 1
            if drainDone >= 1:
                break
        if gIdx >= 0:
            taskId = taskId + 1
            localIdx = localIdx + 1
        else:
            if isLastLoop == 1:
                taskId = taskId + 1
                localIdx = localIdx + 1


def _run_preload2_vector_bn2(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
):
    meta = sched.meta
    fagOff = sched.fagOff
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
    subIdx = sched.subIdx
    dv_align = sched.dv_align
    tensor_y = gm.tensor_y
    tensor_dy = gm.tensor_dy
    tensor_lse = gm.tensor_lse
    tensor_mask = gm.tensor_mask
    tensor_dq_out = gm.tensor_dq_out
    tensor_dk_out = gm.tensor_dk_out
    tensor_dq_ws = gm.tensor_dq_ws
    mm1_res = buf.mm1_res
    mm2_res = buf.mm2_res
    y_ub = buf.y_ub
    dy_ub = buf.dy_ub
    nzp_ub = buf.nzp_ub
    sfmg_ub = buf.sfmg_ub
    lse_ub = buf.lse_ub
    mask_ub = buf.mask_ub
    mask_pre_ub = buf.mask_pre_ub
    tensor_sinks = gm.tensor_sinks
    tensor_dsink_ws = gm.tensor_dsink_ws
    sink_part_ub = buf.sink_part_ub
    aivIdx = pl.get_block_idx()
    cast_ub = buf.cast_ub
    ds_l1 = buf.ds_l1
    p_l1 = buf.p_l1
    taskId = 0
    localIdx = 0
    denseCursor = blockStart
    isLastLoop = 0
    drainDone = 0
    nFetched = 0
    roundId = pl.const(0, pl.DT_INT64)
    while True:
        mm2_res.next()
        mm1_res.next()
        lse_ub.next()
        # Start V1 for task N-1 before scalar scheduling for task N.
        # V1 only reads row geometry; select vecInfo again after scheduling
        # so V2-V6 see any updated L0C flush boundary on the previous task.
        if taskId > 0:
            prevInfo = runInfos[(taskId + 1) % 2]
            process_vec1_bn2(
                constInfo,
                prevInfo,
                subIdx,
                tensor_y,
                tensor_dy,
                y_ub.current(),
                dy_ub.current(),
                sfmg_ub.current(),
                dv_align,
            )
        gIdx = pl.const(-1, pl.DT_INT64)
        fetchedGIdx, fetchedDenseCursor = fetch_work_gidx(
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
        )
        if isLastLoop == 0:
            gIdx = fetchedGIdx
            denseCursor = fetchedDenseCursor
            if gIdx < 0:
                isLastLoop = 1
        if gIdx >= 0:
            curInfo = runInfos[taskId % 2]
            set_run_info(constInfo, curInfo, taskId, gIdx, subIdx, localIdx)
            nFetched = nFetched + 1
            apply_l0c_s2(runInfos, curInfo, taskId, 2)
        vecInfo = runInfos[0]
        mm1_t = mm1_res.current()
        mm2_t = mm2_res.current()
        lse_t = lse_ub.current()
        c2ev = 0
        needSyncDkMM = 0
        doVec = 0
        if taskId > 0:
            doVec = 1
            vecInfo = runInfos[(taskId + 1) % 2]
            mm1_t = mm1_res.previous()
            mm2_t = mm2_res.previous()
            lse_t = lse_ub.previous()
            c2ev = (taskId + 1) % 2
            needSyncDkMM = 0
            if taskId > 1:
                needSyncDkMM = 1
        if doVec == 1:
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
                    pipe=pl.PipeType.MTE3, event_id=SYNC_C4_TO_V3_FLAG
                )
            # C3/C4 reuse mm1/mm2 for dQ/dK in BN2. Finish reading the
            # original dP/P before V3's V->MTE3 copy and either ready event.
            if constInfo.sinkOptional == 1:
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
            process_vec3_bn2(
                constInfo,
                vecInfo,
                subIdx,
                ds_l1.current(),
                mm1_t,
                mm2_t,
                sfmg_ub.current(),
                nzp_ub.current(),
            )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=SYNC_V3_TO_C3_FLAG)
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
                nzp_ub.current(),
            )
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=SYNC_V4_TO_C5_FLAG)
            if is_bn2_multiblk == 1:
                if vecInfo.dqLast == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE2, event_id=SYNC_C3_TO_V5_FLAG
                    )
                    process_dq_multiblk(
                        tensor_dq_out,
                        tensor_dq_ws,
                        mm1_t,
                        cast_ub.current(),
                        constInfo,
                        vecInfo,
                        subIdx,
                        cBlockIdx,
                    )
                if vecInfo.flushDkv == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.V, event_id=SYNC_C4_TO_V6_FLAG
                    )
                    process_muls_cast_dk_bn2(
                        tensor_dk_out,
                        mm2_t,
                        cast_ub.current(),
                        constInfo,
                        vecInfo,
                        subIdx,
                    )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_DETER_FIX_FLAG
                )
            else:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C3_TO_V5_FLAG
                )
                process_muls_cast_dq_bn2(
                    tensor_dq_out,
                    mm1_t,
                    cast_ub.current(),
                    constInfo,
                    vecInfo,
                    subIdx,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C4_TO_V6_FLAG
                )
                process_muls_cast_dk_bn2(
                    tensor_dk_out,
                    mm2_t,
                    cast_ub.current(),
                    constInfo,
                    vecInfo,
                    subIdx,
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_DETER_FIX_FLAG
                )
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
        if gIdx >= 0:
            taskId = taskId + 1
            localIdx = localIdx + 1
        else:
            if isLastLoop == 1:
                taskId = taskId + 1
                localIdx = localIdx + 1


def _run_preload2_vector_gs_normal(
    constInfo,
    sched,
    gm,
    buf,
    runInfos,
):
    meta = sched.meta
    fagOff = sched.fagOff
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
    subIdx = sched.subIdx
    v1w = sched.v1w
    tensor_y = gm.tensor_y
    tensor_dy = gm.tensor_dy
    tensor_lse = gm.tensor_lse
    tensor_mask = gm.tensor_mask
    mm1_res = buf.mm1_res
    mm2_res = buf.mm2_res
    y_ub = buf.y_ub
    dy_ub = buf.dy_ub
    nzp_ub = buf.nzp_ub
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
    localIdx = 0
    denseCursor = blockStart
    isLastLoop = 0
    drainDone = 0
    nFetched = 0
    roundId = pl.const(0, pl.DT_INT64)
    while True:
        gIdx = pl.const(-1, pl.DT_INT64)
        fetchedGIdx, fetchedDenseCursor = fetch_work_gidx(
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
        )
        if isLastLoop == 0:
            gIdx = fetchedGIdx
            denseCursor = fetchedDenseCursor
            if gIdx < 0:
                isLastLoop = 1
        mm2_res.next()
        mm1_res.next()
        if gIdx >= 0:
            curInfo = runInfos[taskId % 2]
            set_run_info(constInfo, curInfo, taskId, gIdx, subIdx, localIdx)
            nFetched = nFetched + 1
            apply_l0c_s2(runInfos, curInfo, taskId, 2)
            lse_ub.next()
        vecInfo = runInfos[0]
        mm1_t = mm1_res.current()
        mm2_t = mm2_res.current()
        lse_t = lse_ub.current()
        c2ev = 0
        needSyncDkMM = 0
        doVec = 0
        if taskId > 0:
            doVec = 1
            vecInfo = runInfos[(taskId + 1) % 2]
            mm1_t = mm1_res.previous()
            mm2_t = mm2_res.previous()
            lse_t = lse_ub.previous()
            c2ev = (taskId + 1) % 2
            needSyncDkMM = 0
            if taskId > 1:
                needSyncDkMM = 1
        if doVec == 1:
            process_vec1(
                constInfo,
                vecInfo,
                subIdx,
                tensor_y,
                tensor_dy,
                y_ub.current(),
                dy_ub.current(),
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
                nzp_ub.current(),
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
                nzp_ub.current(),
            )
            if constInfo.sinkOptional == 1:
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
        if gIdx >= 0:
            taskId = taskId + 1
            localIdx = localIdx + 1
        else:
            if isLastLoop == 1:
                taskId = taskId + 1
                localIdx = localIdx + 1
