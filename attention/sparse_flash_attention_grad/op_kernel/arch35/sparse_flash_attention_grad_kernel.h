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
    __aicore__ inline void ProcessGather(FagRunInfo &runInfo);
    __aicore__ inline void ProcessMm12(FagRunInfo &runInfo);
    __aicore__ inline void ProcessSoftmax(FagRunInfo &runInfo);
    __aicore__ inline void ProcessMm345(FagRunInfo &runInfo);
    __aicore__ inline void ProcessScatter(FagRunInfo &runInfo);
    __aicore__ inline void DrainUnDeter(FagRunInfo runInfos[3], int64_t taskId);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueRunInfo(FagRunInfo &runInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::SetUniqueConstInfo(
    FagConstInfo &constInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessGather(FagRunInfo &runInfo)
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    this->vecBlock.GatherKV(this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    this->vecBlock.CopyMaxSum(this->constInfo, runInfo, runInfo.commonRunInfo.taskId);
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
        // Use the opposite mm1/mm2 ping-pong slot so scatter does not clobber
        // the current task's mm12 result that softmax still needs.
        uint8_t scatterBufIdx = (runInfo.commonRunInfo.taskIdMod2 + 1) & 1;
        LocalTensor<CALC_TYPE> dkInTensor = this->mm1ResBuf[scatterBufIdx].template Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> dvInTensor = this->mm2ResBuf[scatterBufIdx].template Get<CALC_TYPE>();
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C5_TO_V5_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
        this->vecBlock.ScatterAdd(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dkWorkSpaceGm,
                                  this->dvWorkSpaceGm, dkInTensor, dvInTensor, this->constInfo, runInfo);
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
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessDeter()
{
    this->AllocEventID();
    int64_t taskId = 0;
    int64_t sTaskId = 0;
    FagRunInfo runInfos[2]; // for cv ping pong
    FagRunInfo deterRunInfos[2];
    for (int32_t i = 0; i < this->processBS1ByCore; i++) {
        this->t1Index = this->cBlockIdx + this->usedCoreNum * i;
        this->GetTndSeqLen(this->t1Index, this->bIndex);
        this->SetDeterRunInfo(deterRunInfos[sTaskId & 1], sTaskId);
        for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; this->n2Index++) {
            this->GetActualSelCount(this->t1Index, this->n2Index, this->actualSelectedBlockCount);
            for (this->blkCntOffset = 0; this->blkCntOffset < this->actualSelectedBlockCount;
                 this->blkCntOffset += this->constInfo.selectedCountOffset) {
                if (taskId >= 0) {
                    this->SetRunInfo(runInfos[taskId & 1], taskId, sTaskId);
                    ProcessGather(runInfos[taskId & 1]);
                    ProcessMm12(runInfos[taskId & 1]);
                }

                if (taskId > 0) {
                    ProcessSoftmax(runInfos[(taskId + 1) & 1]);
                    ProcessMm345(runInfos[(taskId + 1) & 1]);
                    ProcessScatter(runInfos[(taskId + 1) & 1]);
                }
                taskId++;
            }
        }

        if (sTaskId == 0) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            }
        }

        if (sTaskId > 0) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
            } else {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dkWorkSpaceGm,
                                               this->dvWorkSpaceGm, this->constInfo, deterRunInfos[(sTaskId + 1) & 1]);
            }
        }
        sTaskId++;
    }

    if (runInfos[(taskId + 1) & 1].taskStep == TASK_C1C2) {
        ProcessSoftmax(runInfos[(taskId + 1) & 1]);
        ProcessMm345(runInfos[(taskId + 1) & 1]);
        ProcessScatter(runInfos[(taskId + 1) & 1]);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dkWorkSpaceGm,
                                           this->dvWorkSpaceGm, this->constInfo, deterRunInfos[(sTaskId + 1) & 1]);
        }
    }

    // 分核不均匀时部分核会少一个C核的全核同步
    if (this->processBS1ByCore < this->tilingData->baseParams.formerCoreProcessNNum) {
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            this->SetDeterRunInfo(deterRunInfos[sTaskId & 1], sTaskId);
            this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dkWorkSpaceGm,
                                           this->dvWorkSpaceGm, this->constInfo, deterRunInfos[sTaskId & 1]);
        }
    }
    this->FreeEventID();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::DrainUnDeter(FagRunInfo runInfos[3],
                                                                                                int64_t taskId)
{
    if (taskId <= 0) {
        return;
    }
    if (taskId == 1) {
        ProcessSoftmax(runInfos[0]);
        ProcessMm345(runInfos[0]);
        ProcessScatter(runInfos[0]);
        return;
    }
    // taskId >= 2: remaining mm345/softmax/scatter of the last two tasks
    ProcessMm345(runInfos[(taskId - 2) % 3]);
    ProcessSoftmax(runInfos[(taskId - 1) % 3]);
    ProcessScatter(runInfos[(taskId - 2) % 3]);
    ProcessMm345(runInfos[(taskId - 1) % 3]);
    ProcessScatter(runInfos[(taskId - 1) % 3]);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreGradKernel<CubeBlockType, VecBlockType>::ProcessUnDeter()
{
    this->AllocEventID();
    int64_t taskId = 0;
    FagRunInfo runInfos[3];
    for (int32_t i = 0; i < this->processBS1ByCore; i++) {
        this->t1Index = this->cBlockIdx + this->usedCoreNum * i;
        this->GetTndSeqLen(this->t1Index, this->bIndex);
        for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; this->n2Index++) {
            this->GetActualSelCount(this->t1Index, this->n2Index, this->actualSelectedBlockCount);
            for (this->blkCntOffset = 0; this->blkCntOffset < this->actualSelectedBlockCount;
                 this->blkCntOffset += this->constInfo.selectedCountOffset) {
                FagRunInfo &cur = runInfos[taskId % 3];
                this->SetRunInfo(cur, taskId, i);
                if (unlikely(taskId == 0)) {
                    ProcessGather(cur);
                    ProcessMm12(cur);
                } else if (unlikely(taskId == 1)) {
                    ProcessGather(cur);
                    ProcessMm12(cur);
                    ProcessSoftmax(runInfos[0]);
                } else {
                    ProcessMm345(runInfos[(taskId - 2) % 3]);
                    ProcessGather(cur);
                    ProcessMm12(cur);
                    ProcessSoftmax(runInfos[(taskId - 1) % 3]);
                    ProcessScatter(runInfos[(taskId - 2) % 3]);
                }
                taskId++;
            }
        }
    }
    DrainUnDeter(runInfos, taskId);
    this->FreeEventID();
}
} // namespace SfagBaseApi

#endif
