/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_grad_kernel.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_GRAD_KERNEL_H
#define SPARSE_FLASH_ATTENTION_GRAD_KERNEL_H

#include "sparse_flash_attention_grad_common.h"
#include "sparse_flash_attention_grad_kernel_base.h"
#include "sparse_flash_attention_grad_tiling_data_regbase.h"

namespace SfagBaseApi {

template <typename CubeBlockType, typename VecBlockType>

class FlashAttentionScoreGradKernel
    : public FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                               CubeBlockType, VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseClass = FlashAttentionScoreGradKernelBase<FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>,
                                                        CubeBlockType, VecBlockType>;
    __aicore__ inline void SetUniqueRunInfo(FagRunInfo &runInfo);
    __aicore__ inline void SetUniqueConstInfo(FagConstInfo &constInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessDeter();
    __aicore__ inline void ProcessUnDeter();

private:
    __aicore__ inline void ProcessGather(FagRunInfo &runInfo, bool fusion = false);
    __aicore__ inline void ProcessMm12(FagRunInfo &runInfo);
    __aicore__ inline void ProcessSoftmax(FagRunInfo &runInfo);
    __aicore__ inline void ProcessMm345(FagRunInfo &runInfo);
    __aicore__ inline void ProcessScatter(FagRunInfo &runInfo);
    __aicore__ inline void DrainUnDeter(FagRunInfo runInfos[3], int64_t taskId, int64_t pipeStart);
    __aicore__ inline void FlushPendingDeter(FagRunInfo &runInfo);
    __aicore__ inline void InitFirstDeterEpoch();
    __aicore__ inline void LaunchDeterScatter(FagRunInfo &scatterRunInfo);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueRunInfo(FagRunInfo &runInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueConstInfo(
    FagConstInfo &constInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessGather(FagRunInfo &runInfo,
                                                                                                 bool fusion)
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        bool gather_scatter_fusion = false;
        if (this->constInfo.isHeadNLe64 && fusion) {
            gather_scatter_fusion = true;
        }
        if (gather_scatter_fusion == false) {
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }
    this->vecBlock.GatherKV(this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    this->vecBlock.CopyMaxSum(this->constInfo, runInfo, runInfo.commonRunInfo.taskId);
    if ASCEND_IS_AIV {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessMm12(FagRunInfo &runInfo)
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    this->IterateMmDyV(mm1ResTensor, this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }

    LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    this->IterateMmQK(mm2ResTensor, this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIC {
        // N<=64: K already in L1 after mm12 CopyKV, release GM ping-pong here.
        if (!IS_DETER && this->constInfo.isHeadNLe64) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    runInfo.taskStep = TASK_C1C2;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessSoftmax(FagRunInfo &runInfo)
{
    if (runInfo.taskStep != TASK_C1C2) {
        return;
    }
    if ASCEND_IS_AIV {
        this->vecBlock.ProcessVec1(this->constInfo, runInfo);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        LocalTensor<CALC_TYPE> mm1ResTensor =
            this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> mm2ResTensor =
            this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
        this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo, runInfo);

        Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
        Buffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
        this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo, runInfo);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG);
        this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, runInfo);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessMm345(FagRunInfo &runInfo)
{
    if (runInfo.taskStep != TASK_C1C2) {
        return;
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);

        this->template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
            this->dqWorkSpaceGm, this->selectedKWorkSpaceGm, this->dSL1Buf, this->constInfo, runInfo);

        // N>64 / deter: mm3 still reads selectedK GM; release only after mm3.
        if (IS_DETER || !this->constInfo.isHeadNLe64) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }

        if constexpr (!IS_DETER) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }

        this->template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(this->mm4ResWorkSpaceGm, this->dSL1Buf,
                                                                          this->constInfo, runInfo);

        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);

        this->template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(this->mm5ResWorkSpaceGm, this->pL1Buf,
                                                                          this->constInfo, runInfo);

        if constexpr (!IS_DETER) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C5_TO_V5_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C5_TO_V5_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
    }
    runInfo.taskStep = TASK_C3C4C5;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessScatter(FagRunInfo &runInfo)
{
    if constexpr (!IS_DETER) {
        if (runInfo.taskStep != TASK_C3C4C5) {
            if ASCEND_IS_AIV {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            }
            return;
        }
        // NLe64: ScatterAddHead64 uses dSOutQue/pOutQue, not mm1/mm2.
        // N>64 2-stage: Scatter(t-1) overlaps Mm12(t) -> this task's own mm1/mm2 slot
        // (softmax just finished with it; opposite slot is the live mm12).
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C5_TO_V5_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
        if (!IS_DETER && this->constInfo.isHeadNLe64) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            this->vecBlock.ScatterAddHead64(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dkWorkSpaceGm,
                                            this->dvWorkSpaceGm, this->constInfo, runInfo);
        } else {
            uint8_t scatterBufIdx = runInfo.commonRunInfo.taskIdMod2;
            LocalTensor<CALC_TYPE> dkInTensor = this->mm1ResBuf[scatterBufIdx].template Get<CALC_TYPE>();
            LocalTensor<CALC_TYPE> dvInTensor = this->mm2ResBuf[scatterBufIdx].template Get<CALC_TYPE>();
            this->vecBlock.ScatterAdd(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dkWorkSpaceGm,
                                      this->dvWorkSpaceGm, dkInTensor, dvInTensor, this->constInfo, runInfo);
        }
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
        runInfo.taskStep = TASK_SCATTERADD;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::Process()
{
    if constexpr (IS_DETER) {
        ProcessDeter();
    } else {
        ProcessUnDeter();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::FlushPendingDeter(
    FagRunInfo &runInfo)
{
    // 幂等：ProcessSoftmax / ProcessMm345 内部已按 TASK_C1C2 短路。
    ProcessSoftmax(runInfo);
    ProcessMm345(runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::InitFirstDeterEpoch()
{
    // 第一笔 deter epoch：只建立 flag13 launch 许可，不消费 flag14，不打 flag11。
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
    } else {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::LaunchDeterScatter(
    FagRunInfo &scatterRunInfo)
{
    // 正常/flush/补偿共用：先收上一 epoch 的 flag14，再开 flag11/13。
    // Wait 在 launch scatter 前，不在 compute 入口，compute(N+1) 仍可与 scatter(N) overlap。
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SCATTER_TO_FIX_SYNC_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_TO_FIX_SYNC_FLAG);
        CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
        CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
    } else {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
        this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dkWorkSpaceGm,
                                       this->dvWorkSpaceGm, this->constInfo, scatterRunInfo);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SCATTER_TO_FIX_SYNC_FLAG);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessDeter()
{
    this->AllocEventID();
    int64_t taskId = 0;
    int64_t sTaskId = 0;
    int64_t deterTaskId = 0;
    FagRunInfo runInfos[2];
    FagRunInfo deterRunInfos[2];
    const int64_t maxEpochs = this->GetMaxDeterEpochs();

    // 所有核走同一条 deter epoch 序列。没有 token / sel=0 / TND padding 只跳过 compute，不删除 epoch。
    for (int32_t i = 0; i < maxEpochs; i++) {
        const int64_t taskIdBeforeEpoch = taskId;
        this->t1Index = this->cBlockIdx + this->usedCoreNum * i;
        const bool hasCompute = (this->cBlockIdx < this->usedCoreNum) && (this->t1Index < this->GetValidTndT1Size());
        if (hasCompute) {
            this->GetTndSeqLen(this->t1Index, this->bIndex);
            for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; this->n2Index++) {
                this->GetActualSelCount(this->t1Index, this->n2Index, this->actualSelectedBlockCount);
                for (this->blkCntOffset = 0; this->blkCntOffset < this->actualSelectedBlockCount;
                     this->blkCntOffset += this->constInfo.selectedCountOffset) {
                    this->SetRunInfo(runInfos[taskId & 1], taskId, sTaskId, deterTaskId);
                    ProcessGather(runInfos[taskId & 1]);
                    ProcessMm12(runInfos[taskId & 1]);
                    if (taskId > 0) {
                        FlushPendingDeter(runInfos[(taskId + 1) & 1]);
                    }
                    taskId++;
                }
            }
        }

        // 本 epoch 没有产生新 task 时，pipeline 不会靠下一次 Gather/Mm12 自然推进。
        // 下面的 scatter 消费的是上一 epoch 的 mm4/mm5，必须先把 pending C3/C4/C5 补齐。
        if (taskId == taskIdBeforeEpoch) {
            FlushPendingDeter(runInfos[(taskId + 1) & 1]);
        }

        this->SetDeterRunInfo(deterRunInfos[deterTaskId & 1], sTaskId, deterTaskId);
        if (deterTaskId == 0) {
            InitFirstDeterEpoch();
        } else {
            LaunchDeterScatter(deterRunInfos[(deterTaskId + 1) & 1]);
        }
        deterTaskId++;
        sTaskId++;
    }

    if (runInfos[(taskId + 1) & 1].taskStep == TASK_C1C2) {
        FlushPendingDeter(runInfos[(taskId + 1) & 1]);
    }
    if (maxEpochs > 0) {
        LaunchDeterScatter(deterRunInfos[(deterTaskId + 1) & 1]);
        deterTaskId++;
    }

    if ASCEND_IS_AIV {
        this->vecBlock.FinalizeDSinkAcc(this->constInfo);
    }

    // 不能依赖 CrossCore flag(11-14)：David 上非全核 barrier / set-wait 计数错配，
    // 单 writer 会提前读到未写完的 slot（d_sinks 跨核 reduce 存在竞态）。
    // SyncAll<false> 已在 entry 复用且验证可靠，此处复用同一硬 barrier
    if constexpr (IS_SINKS) {
        this->SyncALLCores();
        if ASCEND_IS_AIV {
            LocalTensor<CALC_TYPE> dSinkScratch = this->mm1ResBuf[0].template Get<CALC_TYPE>();
            LocalTensor<CALC_TYPE> dSinkReduceOut = this->mm1ResBuf[1].template Get<CALC_TYPE>();
            this->vecBlock.ReduceDSink(dSinkScratch, dSinkReduceOut, this->constInfo);
        }
    }
    this->FreeEventID();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::DrainUnDeter(FagRunInfo runInfos[3],
                                                                                                int64_t taskId,
                                                                                                int64_t pipeStart)
{
    int64_t n = taskId - pipeStart;
    if (n <= 0) {
        return;
    }
    // Match the original ProcessUnDeter pre-drain wait: in-loop ScatterAddHead64
    // SetFlag MTE3_MTE2 without waiting; softmax of the last task needs MTE2.
    if (n > 2) {
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }
    if (n == 1) {
        ProcessSoftmax(runInfos[pipeStart % 3]);
        ProcessMm345(runInfos[pipeStart % 3]);
        ProcessScatter(runInfos[pipeStart % 3]);
        if ASCEND_IS_AIV {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        return;
    }
    // n >= 2: remaining mm345/softmax/scatter of the last two tasks in this pipeline
    ProcessSoftmax(runInfos[(taskId - 1) % 3]);
    ProcessMm345(runInfos[(taskId - 1) % 3]);
    ProcessScatter(runInfos[(taskId - 2) % 3]);
    if ASCEND_IS_AIV {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    ProcessScatter(runInfos[(taskId - 1) % 3]);
    if ASCEND_IS_AIV {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessUnDeter()
{
    this->AllocEventID();
    int64_t taskId = 0;
    int64_t pipeStart = 0;
    FagRunInfo runInfos[4];
    for (int32_t i = 0; i < this->processBS1ByCore; i++) {
        this->t1Index = this->cBlockIdx + this->usedCoreNum * i;
        if (!this->IsValidTndT1Index(this->t1Index)) {
            break;
        }
        this->GetTndSeqLen(this->t1Index, this->bIndex);
        for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; this->n2Index++) {
            this->GetActualSelCount(this->t1Index, this->n2Index, this->actualSelectedBlockCount);
            for (this->blkCntOffset = 0; this->blkCntOffset < this->actualSelectedBlockCount;
                 this->blkCntOffset += this->constInfo.selectedCountOffset) {
                FagRunInfo &cur = runInfos[taskId % 3];
                this->SetRunInfo(cur, taskId, i, 0);
                if (this->constInfo.isHeadNLe64) {
                    // NLe64: keep 3-stage within the same S1. When S1 changes, drain
                    // so Scatter(t-2) never overlaps a new S1's Gather/Mm12.
                    if (!cur.isS1IdxNoChange && taskId > pipeStart) {
                        DrainUnDeter(runInfos, taskId, pipeStart);
                        pipeStart = taskId;
                    }
                    int64_t relId = taskId - pipeStart;
                    if (unlikely(relId == 0)) {
                        ProcessGather(cur);
                        ProcessMm12(cur);
                    } else if (unlikely(relId == 1)) {
                        ProcessGather(cur);
                        ProcessMm12(cur);
                        ProcessSoftmax(runInfos[(taskId - 1) % 3]);
                        ProcessMm345(runInfos[(taskId - 1) % 3]);
                    } else {
                        ProcessGather(cur, relId > 2);
                        ProcessMm12(cur);
                        ProcessSoftmax(runInfos[(taskId - 1) % 3]);
                        ProcessMm345(runInfos[(taskId - 1) % 3]);
                        ProcessScatter(runInfos[(taskId - 2) % 3]);
                    }
                } else {
                    // N>64: 2-stage only. mm3 still reads selectedK GM so C3_TO_V0 is
                    // posted after mm3, and that flag id is shared with V5_TO_C4 {7,9}.
                    // 3-stage Gather(t)/Scatter(t-2) would Wait+Set the same id in one
                    // AIV beat while cube Set+Wait it in Mm345, which deadlocks.
                    ProcessGather(cur);
                    ProcessMm12(cur);
                    if (taskId > 0) {
                        ProcessSoftmax(runInfos[(taskId - 1) % 3]);
                        ProcessMm345(runInfos[(taskId - 1) % 3]);
                        ProcessScatter(runInfos[(taskId - 1) % 3]);
                    }
                }
                taskId++;
            }
        }
    }

    if (this->constInfo.isHeadNLe64) {
        DrainUnDeter(runInfos, taskId, pipeStart);
    } else if (taskId > 0) {
        ProcessSoftmax(runInfos[(taskId - 1) % 3]);
        ProcessMm345(runInfos[(taskId - 1) % 3]);
        ProcessScatter(runInfos[(taskId - 1) % 3]);
    }
    if ASCEND_IS_AIV {
        this->vecBlock.FinalizeDSinkAcc(this->constInfo);
    }
    this->FreeEventID();
}
} // namespace SfagBaseApi
#endif
