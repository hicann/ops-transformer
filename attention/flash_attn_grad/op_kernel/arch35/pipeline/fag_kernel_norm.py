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
import os

import pypto_pro.language as pl

from compute import fag_block_cube
from compute import fag_block_vec
from memory import fag_buffers
import fag_common
from schedule import fag_schedule

# Compile-time diagnostics. Keep disabled for precision/performance packages;
# a diagnostic build must include this environment value in its cache identity.
FAG_CORE_BALANCE_DEBUG = os.environ.get("FAG_CORE_BALANCE_DEBUG", "0") == "1"
FAG_CORE_INDEX_DEBUG = os.environ.get("FAG_CORE_INDEX_DEBUG", "0") == "1"

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
    """GS 非 swizzle 及普通 D192/Dv128 SW1 causal：预取 N+1，计算 N / N-1。

    D192 仅有三个 Q/dO 槽，分别容纳 N+1/N/N-1；上一轮的后半段已消费
    N-2，才允许预取 N+1 复用它的槽。C1/C2 前不增加等待 Vector 的后半段。
    普通 D192/Dv128 SW1 causal 的 Vector 同样延后一任务；其它路径保留原计数。
    该路径由 dS ready 的 MTE1 依赖保护 UB 槽复用，不另发 V2 反向 FIX token。
    """
    if swizzle != 0 and not (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and swizzle == 1
        and deter == 0
        and layout == 0
        and mask_mode == 3
        and sink == 0
    ):
        _run_preload3_cube_swizzle(constInfo, sched, gm, buf, runInfos)
        return
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
    cuQ = sched.cuQ
    cuKv = sched.cuKv
    usedQ = sched.usedQ
    usedKv = sched.usedKv
    batchCursor = init_tnd_batch_cursor()
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
            or (
                swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            )
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
    # 与预取 writer 及 C4/C5 lag reader 同址同 mutex，游标独立。
    # writer 领先当前 C1/C2 一个有效任务，不能用其 current() 读取当前任务。
    q_for_mm = make_q_l1()
    do_for_mm = make_do_l1()
    # Non-target inline calls may forward a numeric sentinel, but not a
    # None-valued local. Only the selected path constructs or uses the state.
    slotState = 0
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and swizzle == 1
        and deter == 0
        and layout == 0
        and mask_mode == 3
        and sink == 0
    ):
        slotState = make_cube_manual_slot_state()
    taskId = 0
    nFetched = 0
    denseCursor = blockStart
    if FAG_CORE_BALANCE_DEBUG:
        debugTasks = pl.const(0, pl.DT_INT64)
        debugLateTasks = pl.const(0, pl.DT_INT64)
        debugSumMN = pl.const(0, pl.DT_INT64)
        debugSumM = pl.const(0, pl.DT_INT64)
        debugSumN = pl.const(0, pl.DT_INT64)
        debugKvLoads = pl.const(0, pl.DT_INT64)
        debugFlushes = pl.const(0, pl.DT_INT64)
        debugFirst = pl.const(-1, pl.DT_INT64)
        debugLast = pl.const(-1, pl.DT_INT64)
        debugStart = pl.get_cycle()
        if FAG_CORE_INDEX_DEBUG:
            debugIndices = pl.struct(
                "FagCoreIndexBatch",
                values=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            )
    gIdx = next_work_gidx(
        constInfo,
        cBlockIdx,
        coreNum,
        blockStart,
        blockEnd,
        0,
        denseCursor,
        numBlocks,
        cuQ,
        cuKv,
        usedQ,
        usedKv,
        batchCursor,
    )
    # The aligned path loads task0's Q/dO inside C2/C1 respectively, just
    # before the current K/V. Other paths keep the paired startup preload.
    if gIdx >= 0:
        firstInfo = runInfos[0]
        set_run_info(
            constInfo,
            firstInfo,
            0,
            gIdx,
            0,
            0,
            cuQ,
            cuKv,
            usedQ,
            usedKv,
            batchCursor,
        )
        apply_l0c_s2(constInfo, runInfos, firstInfo, 0, PRELOAD_TIMES)
        if (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and swizzle == 1
            and deter == 0
            and layout == 0
            and mask_mode == 3
            and sink == 0
        ):
            q_first = q_l1[0]
        else:
            q_first = q_l1.next()
        do_first = do_l1.next()
        if not (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and swizzle == 1
            and deter == 0
            and layout == 0
            and mask_mode == 3
            and sink == 0
        ):
            prefetch_qdo_l1(
                constInfo, firstInfo, tensor_q, tensor_dy, q_first, do_first
            )
        nFetched = 1
        denseCursor = gIdx + 1
    else:
        denseCursor = blockEnd

    while True:
        # 唯一 lookahead 调度游标：每个有效任务仅 fetch/decode/load 一次。
        # 无任务后 cursor 固定 blockEnd，drain 不重扫 mask/变长序列的空尾。
        nextGIdx = next_work_gidx(
            constInfo,
            cBlockIdx,
            coreNum,
            blockStart,
            blockEnd,
            taskId + 1,
            denseCursor,
            numBlocks,
            cuQ,
            cuKv,
            usedQ,
            usedKv,
            batchCursor,
        )
        nextInfo = runInfos[(taskId + 1) % PRELOAD_TIMES]
        if nextGIdx >= 0:
            set_run_info(
                constInfo,
                nextInfo,
                taskId + 1,
                nextGIdx,
                0,
                taskId + 1,
                cuQ,
                cuKv,
                usedQ,
                usedKv,
                batchCursor,
            )
            apply_l0c_s2(constInfo, runInfos, nextInfo, taskId + 1, PRELOAD_TIMES)
            if (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
                q_next = q_l1[(taskId + 1) % PRELOAD_TIMES]
            else:
                q_next = q_l1.next()
            do_next = do_l1.next()
            if not (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
                prefetch_qdo_l1(
                    constInfo, nextInfo, tensor_q, tensor_dy, q_next, do_next
                )
            nFetched = nFetched + 1
            denseCursor = nextGIdx + 1
        else:
            denseCursor = blockEnd

        if gIdx >= 0:
            curInfo = runInfos[taskId % PRELOAD_TIMES]
            if FAG_CORE_BALANCE_DEBUG and (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
                # Compare semantic coordinates, not each implementation's
                # encoded work index. S2 is the fastest canonical axis.
                debugIndex = (
                    (
                        (curInfo.boIdx * constInfo.n2Size + curInfo.n2oIdx)
                        * constInfo.gSize
                        + curInfo.goIdx
                    )
                    * constInfo.s1Outer
                    + curInfo.s1oIdx
                ) * constInfo.s2Outer + curInfo.s2oIdx
                if debugTasks == 0:
                    debugFirst = debugIndex
                debugLast = debugIndex
                if FAG_CORE_INDEX_DEBUG:
                    debugIndices.values[debugTasks % 16] = debugIndex
                    if debugTasks % 16 == 15:
                        pl.printf(
                            "FAGIDX side=PTO core=%d first=%d count=%d idx=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                            cBlockIdx,
                            debugTasks - 15,
                            16,
                            debugIndices.values[0],
                            debugIndices.values[1],
                            debugIndices.values[2],
                            debugIndices.values[3],
                            debugIndices.values[4],
                            debugIndices.values[5],
                            debugIndices.values[6],
                            debugIndices.values[7],
                            debugIndices.values[8],
                            debugIndices.values[9],
                            debugIndices.values[10],
                            debugIndices.values[11],
                            debugIndices.values[12],
                            debugIndices.values[13],
                            debugIndices.values[14],
                            debugIndices.values[15],
                        )
                debugTasks = debugTasks + 1
                debugSumMN = debugSumMN + curInfo.s1RealSize * curInfo.s2RealSize
                debugSumM = debugSumM + curInfo.s1RealSize
                debugSumN = debugSumN + curInfo.s2RealSize
                debugKvLoads = debugKvLoads + curInfo.loadKV
            currentKSlot = 0
            if (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
                q_cur = q_for_mm[taskId % PRELOAD_TIMES]
                # Save even when loadKV==0: the late consumer must use this
                # task's owner slot, not the writer's potentially newer slot.
                if curInfo.loadKV == 1:
                    slotState.k_writer_slot = (slotState.k_writer_slot + 1) % 2
                currentKSlot = slotState.k_writer_slot
                slotState.k_slots[taskId % PRELOAD_TIMES] = currentKSlot
            else:
                q_cur = q_for_mm.next()
            do_cur = do_for_mm.next()
            mm2_cur = mm2_res.next()
            mm1_cur = mm1_res.next()
            loadCurrentQDo = 0
            if (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
                loadCurrentQDo = taskId == 0
            if not (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
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
                loadCurrentQDo,
                nextGIdx >= 0,
                nextInfo,
                q_l1[(taskId + 1) % PRELOAD_TIMES]
                if (
                    template == Tmpl.BN2GS1S2
                    and d_align == 192
                    and dv_align == 128
                    and swizzle == 1
                    and deter == 0
                    and layout == 0
                    and mask_mode == 3
                    and sink == 0
                )
                else q_l1.current(),
                slotState,
                taskId % PRELOAD_TIMES,
                currentKSlot,
            )
            pl.system.set_cross_core(
                pipe=pl.PipeType.FIX,
                event_id=SYNC_C2_TO_V2_FLAG[taskId % 2],
            )
            if not (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
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
                loadCurrentQDo,
                nextGIdx >= 0,
                nextInfo,
                do_l1.current(),
                slotState,
                taskId % PRELOAD_TIMES,
            )
            pl.system.set_cross_core(
                pipe=pl.PipeType.FIX,
                event_id=SYNC_C1_TO_V2_FLAG[taskId % 2],
            )
        # 后半段滞后一任务；先 P->C5，再等 dS->C3/C4。
        # 与原两任务延后相比，D192 释放槽的时刻允许下一轮提前载入 Q/dO。
        if taskId > 0:
            fixInfo = runInfos[(taskId + PRELOAD_TIMES - 1) % PRELOAD_TIMES]
            fixTaskId = taskId - 1
            if FAG_CORE_BALANCE_DEBUG and (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
                debugLateTasks = debugLateTasks + 1
            lateKSlot = 0
            if (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
                q_fix = q_l1_dk[fixTaskId % PRELOAD_TIMES]
                lateKSlot = slotState.k_slots[fixTaskId % PRELOAD_TIMES]
            else:
                q_fix = q_l1_dk.next()
            do_fix = do_l1_dv.next()
            if (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
                # Resolve Final before either resident C5/C4 MMAD. A later
                # needFlush override cannot complete a Partial unitflag.
                if fixTaskId >= nFetched - 1:
                    fixInfo.flushDkv = 1
            dsReadyFlag = SYNC_V3_TO_C3_FLAG
            pReadyFlag = SYNC_V4_TO_C5_FLAG
            dsFreeFlag = SYNC_C4_TO_V3_FLAG
            pFreeFlag = SYNC_C5_TO_V4_FLAG
            if d_align <= 128 and swizzle == 0:
                # Consume the actual late task N-1, whose parity differs from N.
                # Each view owns a cursor and advances once per valid task.
                ds_l1.next()
                ds_t_l1.next()
                p_t_l1.next()
                dsReadyFlag = SYNC_V3_TO_C3_FLAG + (fixTaskId % 2) * (
                    SYNC_V3_TO_C3_DB1_FLAG - SYNC_V3_TO_C3_FLAG
                )
                pReadyFlag = SYNC_V4_TO_C5_FLAG + (fixTaskId % 2) * (
                    SYNC_V4_TO_C5_DB1_FLAG - SYNC_V4_TO_C5_FLAG
                )
                dsFreeFlag = SYNC_C4_TO_V3_FLAG + (fixTaskId % 2) * (
                    SYNC_C4_TO_V3_DB1_FLAG - SYNC_C4_TO_V3_FLAG
                )
                pFreeFlag = SYNC_C5_TO_V4_FLAG + (fixTaskId % 2) * (
                    SYNC_C5_TO_V4_DB1_FLAG - SYNC_C5_TO_V4_FLAG
                )
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=pReadyFlag)
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
                None,
                None,
                slotState,
            )
            if (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and swizzle == 1
                and deter == 0
                and layout == 0
                and mask_mode == 3
                and sink == 0
            ):
                # AscendC issues dV FIX inside C5, before releasing P and
                # waiting for dS. The terminal-task Final was set above.
                if fixInfo.flushDkv == 1:
                    flush_dv(constInfo, fixInfo, tensor_dv_ws, dv_acc0, dv_acc1)
                    if FAG_CORE_BALANCE_DEBUG:
                        debugFlushes = debugFlushes + 1
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=pFreeFlag)
            # dV uses P/dO only; dS becomes a dependency at dQ.
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=dsReadyFlag)
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
                mm1_res.current(),
                cBlockIdx,
                acc_n64,
                slotState,
                lateKSlot,
            )
            if d_align <= 128 or (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and (
                    swizzle == 0
                    or (
                        swizzle == 1
                        and deter == 0
                        and layout == 0
                        and mask_mode == 3
                        and sink == 0
                    )
                )
            ):
                # 64/128 and D192/Dv128 retain dK across the actual KV-owner
                # segment, then flush it together with dV to FP32 workspace.
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
                    None,
                    None,
                    slotState,
                    fixTaskId % PRELOAD_TIMES,
                )
            else:
                # Other wide-D paths use acc as non-resident dK scratch.
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
            if fixTaskId >= nFetched - 1:
                needFlush = 1
            if needFlush == 1:
                if (
                    template == Tmpl.BN2GS1S2
                    and d_align == 192
                    and dv_align == 128
                    and swizzle == 1
                    and deter == 0
                    and layout == 0
                    and mask_mode == 3
                    and sink == 0
                ):
                    # dV was already issued immediately after C5.
                    flush_dk(constInfo, fixInfo, tensor_dk_ws, dk_acc0, dk_acc1)
                elif d_align <= 128 or (
                    template == Tmpl.BN2GS1S2
                    and d_align == 192
                    and dv_align == 128
                    and (
                        swizzle == 0
                        or (
                            swizzle == 1
                            and deter == 0
                            and layout == 0
                            and mask_mode == 3
                            and sink == 0
                        )
                    )
                ):
                    # Both resident outputs complete the same owner segment.
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
                else:
                    # Other wide-D paths already store dK per task.
                    flush_dv(constInfo, fixInfo, tensor_dv_ws, dv_acc0, dv_acc1)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=dsFreeFlag)
        # 第一个 gIdx<0 的轮次只消费最后一个后半段；空核无需任何 wait。
        if gIdx < 0:
            break
        gIdx = nextGIdx
        taskId = taskId + 1

    if FAG_CORE_BALANCE_DEBUG and (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and swizzle == 1
        and deter == 0
        and layout == 0
        and mask_mode == 3
        and sink == 0
    ):
        if FAG_CORE_INDEX_DEBUG:
            debugRemainder = debugTasks % 16
            if debugRemainder > 0:
                pl.printf(
                    "FAGIDX side=PTO core=%d first=%d count=%d idx=%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                    cBlockIdx,
                    debugTasks - debugRemainder,
                    debugRemainder,
                    debugIndices.values[0],
                    debugIndices.values[1],
                    debugIndices.values[2],
                    debugIndices.values[3],
                    debugIndices.values[4],
                    debugIndices.values[5],
                    debugIndices.values[6],
                    debugIndices.values[7],
                    debugIndices.values[8],
                    debugIndices.values[9],
                    debugIndices.values[10],
                    debugIndices.values[11],
                    debugIndices.values[12],
                    debugIndices.values[13],
                    debugIndices.values[14],
                    debugIndices.values[15],
                )
        # Drain this AIC only; do not make fast cores wait at a cross-core
        # barrier before reading the timestamp. Printed timings are diagnostic.
        pl.system.bar_all()
        debugElapsed = pl.get_cycle() - debugStart
        pl.printf(
            "FAGCORE side=PTO core=%d tasks=%d late=%d sum_mn=%d sum_m=%d sum_n=%d kv_load=%d flush=%d first=%d last=%d begin=%d end=%d ticks=%d\n",
            cBlockIdx,
            debugTasks,
            debugLateTasks,
            debugSumMN,
            debugSumM,
            debugSumN,
            debugKvLoads,
            debugFlushes,
            debugFirst,
            debugLast,
            blockStart,
            blockEnd,
            debugElapsed,
        )


def _run_preload3_cube_swizzle(
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
    cuQ = sched.cuQ
    cuKv = sched.cuKv
    usedQ = sched.usedQ
    usedKv = sched.usedKv
    batchCursor = init_tnd_batch_cursor()
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
        and swizzle == 1
        and deter == 0
        and not (layout == 0 and mask_mode == 3 and sink == 0)
    ):
        # The fixed-layout causal specialization stays in cube_normal.
        # Other normal swizzle keys retain this two-round pipeline and its
        # automatic L0A/L1 synchronization, with a resident dK/dV allocation.
        acc_n64 = make_l0c_acc_n64()
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
        # 取块调用放在运行期条件外：swizzle 分支内部含运行期 if/return，嵌在
        # ``if isLastLoop == 0`` 内会触发 loop-carried phi 断裂（de5ddd541
        # 重构后 GS1S2+swizzle 上板卡死，见 0902 教训）；条件内只保留简单
        # 变量拷贝——与重构前 fetch_work_gidx 模式同构的最小防护形态。
        fetchedGIdx = next_work_gidx(
            constInfo,
            cBlockIdx,
            coreNum,
            blockStart,
            blockEnd,
            localIdx,
            denseCursor,
            numBlocks,
            cuQ,
            cuKv,
            usedQ,
            usedKv,
            batchCursor,
        )
        gIdx = pl.const(-1, pl.DT_INT64)
        if isLastLoop == 0:
            gIdx = fetchedGIdx
            if gIdx < 0:
                isLastLoop = 1
                # Keep the unconditional fetch, but do not rescan sparse tails
                # in the two remaining preload3 drain iterations.
                if swizzle == 0 or constInfo.normalSwizzleEnabled == 0:
                    denseCursor = blockEnd
            else:
                if swizzle == 0 or constInfo.normalSwizzleEnabled == 0:
                    denseCursor = gIdx + 1
        doFix = 0
        mm2_cur = mm2_res.next()
        mm1_cur = mm1_res.next()
        q_cur = q_l1.next()
        do_cur = do_l1.next()
        if gIdx >= 0:
            curInfo = runInfos[taskId % PRELOAD_TIMES]
            set_run_info(
                constInfo,
                curInfo,
                taskId,
                gIdx,
                0,
                localIdx,
                cuQ,
                cuKv,
                usedQ,
                usedKv,
                batchCursor,
            )
            nFetched = nFetched + 1
            apply_l0c_s2(constInfo, runInfos, curInfo, taskId, PRELOAD_TIMES)
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
            dsReadyFlag = SYNC_V3_TO_C3_FLAG
            pReadyFlag = SYNC_V4_TO_C5_FLAG
            dsFreeFlag = SYNC_C4_TO_V3_FLAG
            pFreeFlag = SYNC_C5_TO_V4_FLAG
            if d_align <= 128 and swizzle == 0:
                # This stage consumes taskId-2; its parity equals taskId's.
                # Each view owns a cursor and advances once per valid task.
                ds_l1.next()
                ds_t_l1.next()
                p_t_l1.next()
                dsReadyFlag = SYNC_V3_TO_C3_FLAG + (taskId % 2) * (
                    SYNC_V3_TO_C3_DB1_FLAG - SYNC_V3_TO_C3_FLAG
                )
                pReadyFlag = SYNC_V4_TO_C5_FLAG + (taskId % 2) * (
                    SYNC_V4_TO_C5_DB1_FLAG - SYNC_V4_TO_C5_FLAG
                )
                dsFreeFlag = SYNC_C4_TO_V3_FLAG + (taskId % 2) * (
                    SYNC_C4_TO_V3_DB1_FLAG - SYNC_C4_TO_V3_FLAG
                )
                pFreeFlag = SYNC_C5_TO_V4_FLAG + (taskId % 2) * (
                    SYNC_C5_TO_V4_DB1_FLAG - SYNC_C5_TO_V4_FLAG
                )
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=pReadyFlag)
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
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=pFreeFlag)
            # dV uses P/dO only; dS becomes a dependency at dQ.
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=dsReadyFlag)
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
            if d_align <= 128 or (
                template == Tmpl.BN2GS1S2
                and d_align == 192
                and dv_align == 128
                and (
                    swizzle == 0
                    or (
                        swizzle == 1
                        and deter == 0
                        and not (layout == 0 and mask_mode == 3 and sink == 0)
                    )
                )
            ):
                # 窄档（64/128）：dK 常驻 L0C（单 D 分块一个槽），跨 s1 累加，
                # 到 flush 时由 flush_dkv 一并写回 fp32 workspace
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
                # 192 档 dK 不驻留、复用 acc 的两个槽作 mm4 scratch
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
            if taskId - 2 >= nFetched - 1:
                needFlush = 1
            if needFlush == 1:
                if d_align <= 128 or (
                    template == Tmpl.BN2GS1S2
                    and d_align == 192
                    and dv_align == 128
                    and (
                        swizzle == 0
                        or (
                            swizzle == 1
                            and deter == 0
                            and not (layout == 0 and mask_mode == 3 and sink == 0)
                        )
                    )
                ):
                    # 窄档：dK/dV 都常驻 L0C，这里一起累加回 ws
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
                else:
                    # 192 档：dK 已每块 atomic 进 ws、不驻留 L0C，这里只有 dV 还常驻，单独累加回 ws。
                    flush_dv(constInfo, fixInfo, tensor_dv_ws, dv_acc0, dv_acc1)
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=dsFreeFlag)
        # Keep the two-round preload3 schedule, but release C3(N-2)'s K
        # before C2(N) can reuse its slot. Vector only needs the old task's
        # already-issued C1/C2 results, so no current-task dependency is added.
        if gIdx >= 0:
            curInfo = runInfos[taskId % PRELOAD_TIMES]
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
    sink,
):
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
    cuQ = sched.cuQ
    cuKv = sched.cuKv
    usedQ = sched.usedQ
    usedKv = sched.usedKv
    batchCursor = init_tnd_batch_cursor()
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
    localIdx = 0
    denseCursor = blockStart
    isLastLoop = 0
    drainDone = 0
    nFetched = 0
    vectorDelay = 2
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and swizzle == 1
        and deter == 0
        and layout == 0
        and mask_mode == 3
        and sink == 0
    ):
        vectorDelay = 1
        sfmg_prefetch = make_sfmg_prefetch_ub(v1w)
    while True:
        if vectorDelay == 2:
            mm2_res.next()
            mm1_res.next()
        # Inline V1 stays at its original point. With preSfmg enabled, the
        # selected one-round pipeline prepares SFMG with the current task's
        # LSE below; V2/V3 consume the previous input slot.
        if taskId >= vectorDelay:
            prevInfo = runInfos[(taskId + PRELOAD_TIMES - vectorDelay) % PRELOAD_TIMES]
            if vectorDelay == 1:
                if constInfo.enablePreSfmg == 0:
                    process_vec1(
                        constInfo,
                        prevInfo,
                        subIdx,
                        tensor_y,
                        tensor_dy,
                        tensor_sfmg_ws,
                        y_ub,
                        dy_ub,
                        sfmg_ub.current(),
                        v1w,
                    )
            else:
                process_vec1(
                    constInfo,
                    prevInfo,
                    subIdx,
                    tensor_y,
                    tensor_dy,
                    tensor_sfmg_ws,
                    y_ub,
                    dy_ub,
                    sfmg_ub.current(),
                    v1w,
                )
        # 取块调用条件外化（与 cube 侧同因）：复杂调用嵌运行期条件触发
        # loop-carried phi 断裂；条件内仅简单变量拷贝。
        fetchedGIdx = next_work_gidx(
            constInfo,
            cBlockIdx,
            coreNum,
            blockStart,
            blockEnd,
            localIdx,
            denseCursor,
            numBlocks,
            cuQ,
            cuKv,
            usedQ,
            usedKv,
            batchCursor,
        )
        gIdx = pl.const(-1, pl.DT_INT64)
        if isLastLoop == 0:
            gIdx = fetchedGIdx
            if gIdx < 0:
                isLastLoop = 1
                if swizzle == 0 or constInfo.normalSwizzleEnabled == 0:
                    denseCursor = blockEnd
            else:
                if swizzle == 0 or constInfo.normalSwizzleEnabled == 0:
                    denseCursor = gIdx + 1
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
                    cuQ,
                    cuKv,
                    usedQ,
                    usedKv,
                    batchCursor,
                )
                nFetched = nFetched + 1
                apply_l0c_s2(constInfo, runInfos, curInfo, taskId, PRELOAD_TIMES)
                if vectorDelay == 2:
                    lse_ub.next()
                else:
                    prefetch_vec_lse(constInfo, curInfo, tensor_lse, lse_ub.next())
                    if constInfo.enablePreSfmg == 1:
                        process_vec1(
                            constInfo,
                            curInfo,
                            subIdx,
                            tensor_y,
                            tensor_dy,
                            tensor_sfmg_ws,
                            y_ub,
                            dy_ub,
                            sfmg_prefetch.next(),
                            v1w,
                        )
            vecInfo = runInfos[0]
            lse_t = lse_ub.current()
            needSyncDkMM = 0
            setV2C = 1
            if vectorDelay == 1:
                setV2C = 0
            doVec = 0
            vecEventTaskId = taskId
            if taskId >= vectorDelay:
                doVec = 1
                vecInfo = runInfos[
                    (taskId + PRELOAD_TIMES - vectorDelay) % PRELOAD_TIMES
                ]
                if vectorDelay == 1:
                    vecEventTaskId = taskId - 1
                if d_align <= 128 and swizzle == 0:
                    # Slots 0/1 have no previous consumer on their first use.
                    if taskId > 3:
                        needSyncDkMM = 1
                else:
                    if taskId > vectorDelay:
                        needSyncDkMM = 1
                if isLastLoop == 1:
                    setV2C = 0
            if doVec == 1:
                dsReadyFlag = SYNC_V3_TO_C3_FLAG
                pReadyFlag = SYNC_V4_TO_C5_FLAG
                dsFreeFlag = SYNC_C4_TO_V3_FLAG
                pFreeFlag = SYNC_C5_TO_V4_FLAG
                if d_align <= 128 and swizzle == 0:
                    ds_l1.next()
                    p_l1.next()
                    dsReadyFlag = SYNC_V3_TO_C3_FLAG + (taskId % 2) * (
                        SYNC_V3_TO_C3_DB1_FLAG - SYNC_V3_TO_C3_FLAG
                    )
                    pReadyFlag = SYNC_V4_TO_C5_FLAG + (taskId % 2) * (
                        SYNC_V4_TO_C5_DB1_FLAG - SYNC_V4_TO_C5_FLAG
                    )
                    dsFreeFlag = SYNC_C4_TO_V3_FLAG + (taskId % 2) * (
                        SYNC_C4_TO_V3_DB1_FLAG - SYNC_C4_TO_V3_FLAG
                    )
                    pFreeFlag = SYNC_C5_TO_V4_FLAG + (taskId % 2) * (
                        SYNC_C5_TO_V4_DB1_FLAG - SYNC_C5_TO_V4_FLAG
                    )
                if vectorDelay == 1:
                    mm2_t = mm2_res.next()
                    mm1_t = mm1_res.next()
                    # A valid current task has just advanced the input writers.
                    # The single drain has no new input and consumes the last
                    # current slot instead of the preceding task's slot.
                    lse_t = lse_ub.current()
                    if gIdx >= 0:
                        lse_t = lse_ub.previous()
                else:
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
                        vectorDelay == 1,
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
                        vectorDelay == 1,
                    )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V,
                    event_id=SYNC_C1_TO_V2_FLAG[vecEventTaskId % 2],
                )
                if needSyncDkMM == 1:
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=pFreeFlag)
                process_vec4(
                    constInfo,
                    vecInfo,
                    subIdx,
                    p_l1.current(),
                    mm2_t,
                    p_out_ub.current(),
                )
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=pReadyFlag)
                if needSyncDkMM == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=dsFreeFlag
                    )
                sfmg_t = sfmg_ub.current()
                if vectorDelay == 1:
                    if constInfo.enablePreSfmg == 1:
                        sfmg_t = sfmg_prefetch.current()
                        if gIdx >= 0:
                            sfmg_t = sfmg_prefetch.previous()
                process_vec3(
                    constInfo,
                    vecInfo,
                    subIdx,
                    ds_l1.current(),
                    mm1_t,
                    mm2_t,
                    sfmg_t,
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
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=dsReadyFlag)
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
            if drainDone >= vectorDelay:
                if nFetched > 0:
                    dsFreeFlag = SYNC_C4_TO_V3_FLAG
                    pFreeFlag = SYNC_C5_TO_V4_FLAG
                    if d_align <= 128 and swizzle == 0:
                        dsFreeFlag = SYNC_C4_TO_V3_FLAG + ((nFetched - 1) % 2) * (
                            SYNC_C4_TO_V3_DB1_FLAG - SYNC_C4_TO_V3_FLAG
                        )
                        pFreeFlag = SYNC_C5_TO_V4_FLAG + ((nFetched - 1) % 2) * (
                            SYNC_C5_TO_V4_DB1_FLAG - SYNC_C5_TO_V4_FLAG
                        )
                    pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=pFreeFlag)
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE3, event_id=dsFreeFlag
                    )
                if d_align <= 128 and swizzle == 0:
                    if nFetched > 1:
                        # Both slots retain a final consumer token; consume
                        # the penultimate task's slot before leaving the loop.
                        dsFreeFlag = SYNC_C4_TO_V3_FLAG + ((nFetched - 2) % 2) * (
                            SYNC_C4_TO_V3_DB1_FLAG - SYNC_C4_TO_V3_FLAG
                        )
                        pFreeFlag = SYNC_C5_TO_V4_FLAG + ((nFetched - 2) % 2) * (
                            SYNC_C5_TO_V4_DB1_FLAG - SYNC_C5_TO_V4_FLAG
                        )
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE3, event_id=pFreeFlag
                        )
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE3, event_id=dsFreeFlag
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
    cuQ = sched.cuQ
    cuKv = sched.cuKv
    usedQ = sched.usedQ
    usedKv = sched.usedKv
    batchCursor = init_tnd_batch_cursor()
    tensor_q = gm.tensor_q
    tensor_k = gm.tensor_k
    tensor_k_nt = gm.tensor_k_nt
    tensor_dy = gm.tensor_dy
    tensor_v = gm.tensor_v
    tensor_dq_ws = gm.tensor_dq_ws
    tensor_dv_out = gm.tensor_dv_out
    if d_align > 128:
        # BN2 wide：dK 256/768、dV 768 走 4D fp32 workspace（192 的 dK/dV、
        # 256 的 dV 驻留 L0C）。
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
    # dK/dV 每个 D 分块各一个槽：c0 是第 0 块、c1 是第 1 块（D_NCHUNKS==1 的
    # 档位只有 c0，c1 别名到 c0、不被引用）。
    dk_c0 = buf.dk_c0
    dk_c1 = buf.dk_c1
    dv_c0 = buf.dv_c0
    dv_c1 = buf.dv_c1
    l0c_load_ub = buf.l0c_load_ub
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
            cuQ,
            cuKv,
            usedQ,
            usedKv,
            batchCursor,
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
            set_run_info(
                constInfo,
                curInfo,
                taskId,
                gIdx,
                0,
                localIdx,
                cuQ,
                cuKv,
                usedQ,
                usedKv,
                batchCursor,
            )
            apply_l0c_s2(constInfo, runInfos, curInfo, taskId, 2)
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
            if d_align <= 128 and is_bn2_multiblk == 1:
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
                    dk_c0,
                    dk_c0,
                    fixInfo.dkvAcc,
                )
                if fixInfo.flushDkv == 1:
                    pl.set_validshape(mm2_prev, [VECTOR_BASEM, VECTOR_BASEN])
                    pl.move(
                        mm2_prev,
                        dk_c0.current(),
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
                    dv_c0,
                    dv_c0,
                    fixInfo.dkvAcc,
                )
                if fixInfo.flushDkv == 1:
                    pl.store(
                        tensor_dv_out,
                        dv_c0.current(),
                        [
                            fixInfo.idx0KV,
                            fixInfo.keyOffset,
                            fixInfo.idx2KV,
                            0,
                        ],
                        order=[1, 3],
                        atomic=pl.AtomicType.AtomicNone,
                    )
            elif d_align <= 128:
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
                    dk_c0,
                    dk_c0,
                    0,
                )
                pl.set_validshape(mm2_prev, [VECTOR_BASEM, VECTOR_BASEN])
                pl.move(
                    mm2_prev,
                    dk_c0.current(),
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
                    dv_c0,
                    dv_c0,
                    0,
                )
                pl.store(
                    tensor_dv_out,
                    dv_c0.current(),
                    [fixInfo.idx0KV, fixInfo.keyOffset, fixInfo.idx2KV, 0],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicNone,
                )
            else:
                # BN2 wide：
                # （1）dQ 192/256/768档 走workspace
                # （2）dK 256/768档 走workspace，192档走L0C
                # （3）dV 768档 走workspace，192/256 常驻 L0C
                if fixInfo.dqLast == 1:
                    pl.system.set_cross_core(
                        pipe=pl.PipeType.FIX, event_id=SYNC_C3_TO_V5_FLAG
                    )
                if d_align == 192:
                    iterate_mm_dsq(
                        constInfo,
                        fixInfo,
                        ds_t_l1.current(),
                        q_fix,
                        left,
                        right,
                        dk_c0,
                        dk_c1,
                        fixInfo.dkvAcc,
                    )
                    iterate_mm_pdy(
                        constInfo,
                        fixInfo,
                        p_t_l1.current(),
                        do_fix,
                        left,
                        right,
                        dv_c0,
                        dv_c1,
                        fixInfo.dkvAcc,
                    )
                    if fixInfo.flushDkv == 1:
                        flush_dv_l0c_to_out(
                            constInfo, fixInfo, tensor_dv_out, dv_c0, dv_c1
                        )
                        flush_dk_l0c_to_ub(
                            constInfo,
                            fixInfo,
                            dk_c0,
                            dk_c1,
                            mm2_prev,
                            l0c_load_ub.current(),
                        )
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.FIX, event_id=SYNC_C4_TO_V6_FLAG
                        )
                elif d_align == 256:
                    iterate_mm_dsq(
                        constInfo,
                        fixInfo,
                        ds_t_l1.current(),
                        q_fix,
                        left,
                        right,
                        dk_c0,
                        dk_c0,
                        fixInfo.dkvAcc,
                        tensor_dk_ws,
                        tensor_q,
                    )
                    iterate_mm_pdy(
                        constInfo,
                        fixInfo,
                        p_t_l1.current(),
                        do_fix,
                        left,
                        right,
                        dv_c0,
                        dv_c1,
                        fixInfo.dkvAcc,
                    )
                    if fixInfo.flushDkv == 1:
                        flush_dv_l0c_to_out(
                            constInfo, fixInfo, tensor_dv_out, dv_c0, dv_c1
                        )
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.FIX, event_id=SYNC_C4_TO_V6_FLAG
                        )
                else:
                    iterate_mm_dsq(
                        constInfo,
                        fixInfo,
                        ds_t_l1.current(),
                        q_fix,
                        left,
                        right,
                        dk_c0,
                        dk_c0,
                        fixInfo.dkvAcc,
                        tensor_dk_ws,
                        tensor_q,
                    )
                    iterate_mm_pdy(
                        constInfo,
                        fixInfo,
                        p_t_l1.current(),
                        do_fix,
                        left,
                        right,
                        dv_c0,
                        dv_c0,
                        fixInfo.dkvAcc,
                        tensor_dv_ws,
                        tensor_dy,
                    )
                    if fixInfo.flushDkv == 1:
                        # dK 与 dV 在同一个 flush 边界就绪，vector 收到本事件后
                        # 同时 cast 两者。
                        pl.system.set_cross_core(
                            pipe=pl.PipeType.FIX, event_id=SYNC_C4_TO_V6_FLAG
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
    cuQ = sched.cuQ
    cuKv = sched.cuKv
    usedQ = sched.usedQ
    usedKv = sched.usedKv
    batchCursor = init_tnd_batch_cursor()
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
    dv_acc0 = buf.dv_acc0
    dv_acc1 = buf.dv_acc1
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
            cuQ,
            cuKv,
            usedQ,
            usedKv,
            batchCursor,
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
            set_run_info(
                constInfo,
                curInfo,
                taskId,
                gIdx,
                0,
                localIdx,
                cuQ,
                cuKv,
                usedQ,
                usedKv,
                batchCursor,
            )
            apply_l0c_s2(constInfo, runInfos, curInfo, taskId, 2)
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
            # D256 dV storage and owner flags are valid for both scheduling orders.
            if d_align == 256:
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
                    # A core may end in the middle of a KV column. The last
                    # preload2 drain must flush even if its next global task
                    # would target the same column on another core.
                    if fixInfo.flushDkv == 1 or isLastLoop == 1:
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
            pl.system.set_cross_core(pipe=pl.PipeType.MTE1, event_id=SYNC_C5_TO_V4_FLAG)
            # dV uses P/dO only. Wait for dS immediately before its first consumer.
            pl.system.wait_cross_core(
                pipe=pl.PipeType.MTE1, event_id=SYNC_V3_TO_C3_FLAG
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
    sink,
):
    meta = sched.meta
    fagOff = sched.fagOff
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
    cuQ = sched.cuQ
    cuKv = sched.cuKv
    usedQ = sched.usedQ
    usedKv = sched.usedKv
    batchCursor = init_tnd_batch_cursor()
    subIdx = sched.subIdx
    dv_align = sched.dv_align
    tensor_y = gm.tensor_y
    tensor_dy = gm.tensor_dy
    tensor_lse = gm.tensor_lse
    tensor_mask = gm.tensor_mask
    tensor_dq_out = gm.tensor_dq_out
    tensor_dk_out = gm.tensor_dk_out
    tensor_dq_ws = gm.tensor_dq_ws
    # multiblk+dv_align>=128 时 process_vec1 读前移缓存（见其前移分支）；
    # single/64 档此参数为 lse 占位、读分支不触发、永不被解引用。
    tensor_sfmg_ws = gm.tensor_sfmg_ws
    if d_align > 128:
        # BN2 wide：V1~V4 走 GS1S2 的 D 档编排，v5/v6 从 4D workspace 内联
        # cast（192 的 dK 读 cube 搬进 UB 的结果；192/256 的 dV 不经过 vector）。
        v1w = sched.v1w
        tensor_dv_out = gm.tensor_dv_out
        tensor_dk_ws = gm.tensor_dk_ws
        tensor_dv_ws = gm.tensor_dv_ws
        ws_load_ub = buf.ws_load_ub
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
    ws_cast_ub = buf.ws_cast_ub
    l0c_cast_ub = buf.l0c_cast_ub
    l0c_load_ub = buf.l0c_load_ub
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
            if d_align > 128:
                process_vec1(
                    constInfo,
                    prevInfo,
                    subIdx,
                    tensor_y,
                    tensor_dy,
                    # wide+multiblk 读前移缓存（前移分支与 preload2 写入方
                    # 同口径）；wide+single 走内联分支，此参数不被解引用。
                    tensor_sfmg_ws,
                    y_ub,
                    dy_ub,
                    sfmg_ub.current(),
                    v1w,
                )
            else:
                # BN2 的 host 门（Bn2FamilyFeasible）保证 d==dv，且 dv_align<=128
                # 落在 process_vec1 的同一分支，直接复用。
                # narrow 的 multiblk+dv_align=128 读前移缓存；single/64 档走
                # 内联分支，tensor_sfmg_ws 为 lse 占位、永不被解引用。
                process_vec1(
                    constInfo,
                    prevInfo,
                    subIdx,
                    tensor_y,
                    tensor_dy,
                    tensor_sfmg_ws,
                    y_ub,
                    dy_ub,
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
            cuQ,
            cuKv,
            usedQ,
            usedKv,
            batchCursor,
        )
        if isLastLoop == 0:
            gIdx = fetchedGIdx
            denseCursor = fetchedDenseCursor
            if gIdx < 0:
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
                cuQ,
                cuKv,
                usedQ,
                usedKv,
                batchCursor,
            )
            nFetched = nFetched + 1
            apply_l0c_s2(constInfo, runInfos, curInfo, taskId, 2)
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
            if d_align <= 128 and is_bn2_multiblk == 1:
                if vecInfo.dqLast == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE2, event_id=SYNC_C3_TO_V5_FLAG
                    )
                    process_dq_multiblk(
                        tensor_dq_out,
                        tensor_dq_ws,
                        mm1_t,
                        l0c_cast_ub.current(),
                        constInfo,
                        vecInfo,
                        subIdx,
                        cBlockIdx,
                    )
                if vecInfo.flushDkv == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.V, event_id=SYNC_C4_TO_V6_FLAG
                    )
                    first = vecInfo.firstHalfS2RealSize
                    if subIdx == 0:
                        rows = first
                    else:
                        rows = pl.max(vecInfo.s2RealSize - first, 0)
                    muls_cast_ub_out(
                        tensor_dk_out,
                        mm2_t,
                        l0c_cast_ub.current(),
                        vecInfo.idx0KV,
                        vecInfo.keyOffset + first * subIdx,
                        vecInfo.idx2KV,
                        rows,
                        constInfo.scaleValue,
                        constInfo.dSize,
                    )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_DETER_FIX_FLAG
                )
            elif d_align <= 128:
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C3_TO_V5_FLAG
                )
                muls_cast_ub_out(
                    tensor_dq_out,
                    mm1_t,
                    l0c_cast_ub.current(),
                    vecInfo.idx0Q,
                    vecInfo.queryOffset + subIdx * VECTOR_BASEM,
                    vecInfo.idx2Q,
                    vecInfo.halfS1RealSize,
                    constInfo.scaleValue,
                    constInfo.dSize,
                )
                pl.system.wait_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_C4_TO_V6_FLAG
                )
                first = vecInfo.firstHalfS2RealSize
                if subIdx == 0:
                    rows = first
                else:
                    rows = pl.max(vecInfo.s2RealSize - first, 0)
                muls_cast_ub_out(
                    tensor_dk_out,
                    mm2_t,
                    l0c_cast_ub.current(),
                    vecInfo.idx0KV,
                    vecInfo.keyOffset + first * subIdx,
                    vecInfo.idx2KV,
                    rows,
                    constInfo.scaleValue,
                    constInfo.dSize,
                )
                pl.system.set_cross_core(
                    pipe=pl.PipeType.V, event_id=SYNC_DETER_FIX_FLAG
                )
            else:
                # BN2 wide：
                # （1）dQ 192/258/768、dK 256/768、dV 768 从 workspace 读回、Muls/Cast、落 out
                # （2）dK 192 从 L0C 搬进；dV 192/256 由 cube 直出，vector 不参与。
                if vecInfo.dqLast == 1:
                    pl.system.wait_cross_core(
                        pipe=pl.PipeType.MTE2, event_id=SYNC_C3_TO_V5_FLAG
                    )
                    muls_cast_ws_out(
                        tensor_dq_out,
                        tensor_dq_ws,
                        ws_load_ub.current(),
                        ws_cast_ub.current(),
                        vecInfo.idx0Q,
                        vecInfo.queryOffset + subIdx * VECTOR_BASEM,
                        vecInfo.idx2Q,
                        vecInfo.halfS1RealSize,
                        constInfo.scaleValue,
                        constInfo.dSize,
                    )
                if vecInfo.flushDkv == 1:
                    if d_align == 192:
                        # P2（BN2 192）：读的是 cube 刚搬进 UB 的 dK（不是 GM ws）
                        # → 等 V；chunk0 复用 mm2 的上一槽，chunk1 用 l0c_load_ub 槽。
                        # dV 已由 cube fixpipe 直出，vector 不再参与。
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.V, event_id=SYNC_C4_TO_V6_FLAG
                        )
                        first = vecInfo.firstHalfS2RealSize
                        if subIdx == 0:
                            rows = first
                        else:
                            rows = pl.max(vecInfo.s2RealSize - first, 0)
                        muls_cast_ub_out(
                            tensor_dk_out,
                            mm2_t,
                            l0c_cast_ub.current(),
                            vecInfo.idx0KV,
                            vecInfo.keyOffset + first * subIdx,
                            vecInfo.idx2KV,
                            rows,
                            constInfo.scaleValue,
                            chunk_valid_d(constInfo.dSize, 0),
                            D_CHUNK_OFF0[d_align],
                        )
                        if (
                            D_NCHUNKS[d_align] > 1
                            and chunk_valid_d(constInfo.dSize, 1) > 0
                        ):
                            muls_cast_ub_out(
                                tensor_dk_out,
                                l0c_load_ub.current(),
                                l0c_cast_ub.current(),
                                vecInfo.idx0KV,
                                vecInfo.keyOffset + first * subIdx,
                                vecInfo.idx2KV,
                                rows,
                                constInfo.scaleValue,
                                chunk_valid_d(constInfo.dSize, 1),
                                D_CHUNK_OFF1[d_align],
                            )
                    elif d_align == 256:
                        # 方案 A：dV 已由 cube 在 flush 时 fixpipe 直出，不再经过
                        # workspace；vector 只负责 dK 的 Muls(scale)+Cast，从
                        # fp32 workspace 读回。
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE2, event_id=SYNC_C4_TO_V6_FLAG
                        )
                        muls_cast_ws_out(
                            tensor_dk_out,
                            tensor_dk_ws,
                            ws_load_ub.current(),
                            ws_cast_ub.current(),
                            vecInfo.idx0KV,
                            vecInfo.keyOffset + subIdx * VECTOR_BASEM,
                            vecInfo.idx2KV,
                            pl.max(
                                pl.min(
                                    vecInfo.s2RealSize - subIdx * VECTOR_BASEM,
                                    VECTOR_BASEM,
                                ),
                                0,
                            ),
                            constInfo.scaleValue,
                            constInfo.dSize,
                        )
                    else:
                        # 768：首读是 workspace 的 MTE2 load，等待必须阻塞 MTE2；
                        # dK/dV 都在同一个 flush 边界就绪，一起 cast。
                        pl.system.wait_cross_core(
                            pipe=pl.PipeType.MTE2, event_id=SYNC_C4_TO_V6_FLAG
                        )
                        muls_cast_ws_out(
                            tensor_dk_out,
                            tensor_dk_ws,
                            ws_load_ub.current(),
                            ws_cast_ub.current(),
                            vecInfo.idx0KV,
                            vecInfo.keyOffset + subIdx * VECTOR_BASEM,
                            vecInfo.idx2KV,
                            pl.max(
                                pl.min(
                                    vecInfo.s2RealSize - subIdx * VECTOR_BASEM,
                                    VECTOR_BASEM,
                                ),
                                0,
                            ),
                            constInfo.scaleValue,
                            constInfo.dSize,
                        )
                        muls_cast_ws_out(
                            tensor_dv_out,
                            tensor_dv_ws,
                            ws_load_ub.current(),
                            ws_cast_ub.current(),
                            vecInfo.idx0KV,
                            vecInfo.keyOffset + subIdx * VECTOR_BASEM,
                            vecInfo.idx2KV,
                            pl.max(
                                pl.min(
                                    vecInfo.s2RealSize - subIdx * VECTOR_BASEM,
                                    VECTOR_BASEM,
                                ),
                                0,
                            ),
                            1.0,
                            constInfo.dvSize,
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
    sink,
):
    meta = sched.meta
    fagOff = sched.fagOff
    cBlockIdx = sched.cBlockIdx
    coreNum = sched.coreNum
    blockStart = sched.blockStart
    blockEnd = sched.blockEnd
    numBlocks = sched.numBlocks
    cuQ = sched.cuQ
    cuKv = sched.cuKv
    usedQ = sched.usedQ
    usedKv = sched.usedKv
    batchCursor = init_tnd_batch_cursor()
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
            cuQ,
            cuKv,
            usedQ,
            usedKv,
            batchCursor,
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
            set_run_info(
                constInfo,
                curInfo,
                taskId,
                gIdx,
                subIdx,
                localIdx,
                cuQ,
                cuKv,
                usedQ,
                usedKv,
                batchCursor,
            )
            nFetched = nFetched + 1
            apply_l0c_s2(constInfo, runInfos, curInfo, taskId, 2)
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
