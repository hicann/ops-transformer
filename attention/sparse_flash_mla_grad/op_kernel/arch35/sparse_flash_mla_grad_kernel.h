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
 * \file sparse_flash_mla_grad_kernel.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_GRAD_KERNEL_H
#define SPARSE_FLASH_MLA_GRAD_KERNEL_H

#include "sparse_flash_mla_grad_common.h"
#include "sparse_flash_mla_grad_kernel_base.h"
#include "sparse_flash_mla_grad_tiling_data_regbase.h"

namespace SfagBaseApi {

template <typename CubeBlockType, typename VecBlockType>

class SparseFlashMlaGradKernel
    : public SparseFlashMlaGradKernelBase<SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>, CubeBlockType,
                                          VecBlockType> {
public:
    ARGS_TRAITS;
    using BaseClass = SparseFlashMlaGradKernelBase<SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>, CubeBlockType,
                                                   VecBlockType>;
    __aicore__ inline void SetUniqueRunInfo(FagRunInfo &runInfo);
    __aicore__ inline void SetUniqueConstInfo(FagConstInfo &constInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessDeter();
    __aicore__ inline void ProcessUnDeter();

private:
    __aicore__ inline void ProcessPreload(FagRunInfo &runInfo, int64_t taskId, bool &isFirstBlock);
    __aicore__ inline void ProcessNotFirst(FagRunInfo &runInfo);
    __aicore__ inline void ScatterAddUnDeter(FagRunInfo &runInfo);
    __aicore__ inline void CopyLse(FagRunInfo &runInfo);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::SetUniqueRunInfo(FagRunInfo &runInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::SetUniqueConstInfo(
    FagConstInfo &constInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessPreload(FagRunInfo &runInfo,
                                                                                             int64_t taskId,
                                                                                             bool &isFirstBlock)
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    int32_t s2RealSize = this->vecBlock.GatherKV(this->selectedKWorkSpaceGm, this->constInfo, runInfo);

    if ASCEND_IS_AIV {
        this->vecBlock.InitCubeVecSharedParams(runInfo, s2RealSize);

        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(
            SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]); // dqk must wait ds copy completely
    } else {
        this->cubeBlock.GetCubeVecSharedParams(runInfo);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmDyV(mm1ResTensor, this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }

    LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    this->cubeBlock.IterateMmQK(mm2ResTensor, this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    if (isFirstBlock) {
        this->vecBlock.CopyLse(this->constInfo, runInfo); // copy in lse double buffer
        if (this->constInfo.isSink) {
            this->vecBlock.ProcessSoftmaxSink(this->constInfo, runInfo);
        }
        isFirstBlock = false;
    }
    runInfo.taskStep = TASK_C1C2;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessNotFirst(FagRunInfo &runInfo)
{
    if (runInfo.taskStep != TASK_C1C2) {
        return;
    }
    this->vecBlock.ProcessVec1(this->constInfo, runInfo);
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(
            SYNC_C1_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]); // dqk must wait ds copy completely
        CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(
            SYNC_C2_TO_V2_FLAG[runInfo.commonRunInfo.taskIdMod2]); // dqk must wait ds copy completely
    }

    LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();
    LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[runInfo.commonRunInfo.taskIdMod2].template Get<CALC_TYPE>();

    this->vecBlock.ProcessVec2(mm2ResTensor, this->constInfo, runInfo); // v2: pse + attenMask + simpleSoftmax
    if (this->constInfo.isSink) {
        this->vecBlock.ProcessVecSink(mm1ResTensor, mm2ResTensor, this->constInfo, runInfo);
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C4_TO_V3_FLAG);
    }
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = this->dSL1Buf.Get();
    Buffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = this->pL1Buf.Get();
    this->vecBlock.ProcessVec3(dSL1Buffer, mm1ResTensor, mm2ResTensor, this->constInfo,
                               runInfo); // v3: dropout + cast + nd2nz
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V3_TO_C3_FLAG); // dqk must wait ds copy completely
    }

    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C5_TO_V4_FLAG);
    }
    this->vecBlock.ProcessVec4(pL1Buffer, mm2ResTensor, this->constInfo, runInfo); // v4: cast + nd2nz
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V4_TO_C5_FLAG); // dv must wait ds copy completely
    }
    if ASCEND_IS_AIC {
        // wait ds in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V3_TO_C3_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V3_TO_C3_FLAG);
    }
    // compute dq
    this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
        this->dqWorkSpaceGm, this->selectedKWorkSpaceGm, this->dSL1Buf, this->constInfo, runInfo); // c3
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        if constexpr (!IsDETER) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
    }

    // compute dk
    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(this->mm4ResWorkSpaceGm, this->dSL1Buf,
                                                                                this->constInfo, runInfo); // c4
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C4_TO_V3_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C4_TO_V3_FLAG);
    }

    if ASCEND_IS_AIC {
        // wait p in ub copy to l1
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V4_TO_C5_FLAG);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V4_TO_C5_FLAG);
    }
    // compute dv
    this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(this->mm5ResWorkSpaceGm, this->pL1Buf,
                                                                                this->constInfo, runInfo); // c5
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_C5_TO_V4_FLAG);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_C5_TO_V4_FLAG);
        if constexpr (!IsDETER) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C5_TO_V5_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C5_TO_V5_FLAG);
        }
    }
    runInfo.taskStep = TASK_C3C4C5;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ScatterAddUnDeter(FagRunInfo &runInfo)
{
    if constexpr (!IsDETER) {
        int8_t mode2 = (runInfo.commonRunInfo.taskIdMod2 + 1) % 2;
        LocalTensor<CALC_TYPE> mm1ResTensor = this->mm1ResBuf[mode2].template Get<CALC_TYPE>();
        LocalTensor<CALC_TYPE> mm2ResTensor = this->mm2ResBuf[mode2].template Get<CALC_TYPE>();
        if (runInfo.taskStep != TASK_C3C4C5) {
            if ASCEND_IS_AIV {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            }
            return;
        }
        if ASCEND_IS_AIV {
            // wait mm4 mm5 result
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C5_TO_V5_FLAG);
        }

        GlobalTensor<CALC_TYPE> outWorkspace = runInfo.isOriKV ? this->dOriKVWorkSpaceGm : this->dCmpKVWorkSpaceGm;
        bool isSparse = runInfo.isOriKV ? IsOriKVSparse : IsCmpKVSparse;
        if (isSparse) {
            this->vecBlock.template ScatterAdd<true>(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, outWorkspace,
                                                     mm1ResTensor, mm2ResTensor, this->constInfo, runInfo);
        } else {
            this->vecBlock.template ScatterAdd<false>(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, outWorkspace,
                                                      mm1ResTensor, mm2ResTensor, this->constInfo, runInfo);
        }

        if ASCEND_IS_AIV {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
        runInfo.taskStep = TASK_SCATTERADD;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::Process()
{
    if constexpr (IsDETER) {
        ProcessDeter();
    } else {
        ProcessUnDeter();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessDeter()
{
    this->AllocEventID();
    if (this->constInfo.isSink) {
        this->vecBlock.CopySink(this->constInfo);
    }
    int64_t taskId = 0;
    int64_t sTaskId = 0;
    int64_t deterTaskId = 0;
    FagRunInfo runInfos[2]; // for cv ping pong
    FagRunInfo deterRunInfos[2];
    for (int32_t i = 0; i < this->processBS1ByCore; i++) {
        bool isFirstBlock = true;
        this->usedT1Index = this->cBlockIdx + this->usedCoreNum * i;
        this->GetTndSeqLen(this->usedT1Index, this->bIndex);
        for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; this->n2Index++) {
            this->GetActualSelCount(this->usedT1Index, this->n2Index, this->actualSelectedBlockCount);
            if constexpr (IsOriKVExist) {
                for (this->oriKvOffset = 0; this->oriKvOffset < this->maxOriSeqKV;
                     this->oriKvOffset += PROCESS_KV_SIZE) {
                    int64_t blockSize = this->oriKvOffset;
                    int64_t tailSize = this->actualOriSelectedBlockCount - this->oriKvOffset > PROCESS_KV_SIZE ?
                                           PROCESS_KV_SIZE :
                                           this->actualOriSelectedBlockCount - this->oriKvOffset;
                    tailSize = Max(tailSize, 0);
                    for (this->blkCntOffset = blockSize; this->blkCntOffset < blockSize + tailSize;
                         this->blkCntOffset += this->constInfo.selectedCountOffset) {
                        if (taskId >= 0) {
                            this->template SetRunInfo<true>(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], taskId,
                                                            sTaskId, deterTaskId, this->oriKvOffset);
                            ProcessPreload(runInfos[taskId & 1], taskId, isFirstBlock);
                        }
                        int64_t curTaskS2RealSize = runInfos[taskId & 1].commonRunInfo.s2RealSize;
                        int64_t lastTaskS2RealSize = runInfos[(taskId + 1) & 1].commonRunInfo.s2RealSize;
                        if (taskId > 0) {
                            ProcessNotFirst(runInfos[(taskId + 1) & 1]);
                        }
                        taskId++;
                        if (curTaskS2RealSize < this->constInfo.selectedCountOffset) {
                            break;
                        }
                    }
                    if (this->actualOriSelectedBlockCount <= 0 || tailSize == 0) {
                        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
                    }
                    if (deterTaskId == 0) {
                        if ASCEND_IS_AIC {
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
                        } else {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                        }
                    }
                    this->template SetDeterRunInfo<true>(deterRunInfos[deterTaskId & 1], sTaskId, deterTaskId,
                                                         this->oriKvOffset);
                    if (deterTaskId > 0) {
                        if ASCEND_IS_AIC {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SCATTER_TO_FIX_SYNC_FLAG);
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_TO_FIX_SYNC_FLAG);
                            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
                        } else {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                            this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm,
                                                           this->dOriKVWorkSpaceGm, this->dCmpKVWorkSpaceGm,
                                                           this->constInfo, deterRunInfos[(deterTaskId + 1) & 1]);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SCATTER_TO_FIX_SYNC_FLAG);
                        }
                    }
                    deterTaskId++;
                }
            }

            if constexpr (IsCmpKVExist) {
                for (this->cmpKvOffset = 0; this->cmpKvOffset < this->maxCmpSeqKV;
                     this->cmpKvOffset += PROCESS_KV_SIZE) {
                    int64_t blockSize = this->actualOriSelectedBlockCount + this->cmpKvOffset;
                    int64_t tailSize = this->actualCmpSelectedBlockCount - this->cmpKvOffset > PROCESS_KV_SIZE ?
                                           PROCESS_KV_SIZE :
                                           this->actualCmpSelectedBlockCount - this->cmpKvOffset;
                    tailSize = Max(tailSize, 0);
                    for (this->blkCntOffset = blockSize; this->blkCntOffset < blockSize + tailSize;
                         this->blkCntOffset += this->constInfo.selectedCountOffset) {
                        if (taskId >= 0) {
                            this->template SetRunInfo<false>(runInfos[taskId & 1], runInfos[(taskId + 1) & 1], taskId,
                                                             sTaskId, deterTaskId, this->cmpKvOffset);
                            ProcessPreload(runInfos[taskId & 1], taskId, isFirstBlock);
                        }
                        int64_t curTaskS2RealSize = runInfos[taskId & 1].commonRunInfo.s2RealSize;
                        int64_t lastTaskS2RealSize = runInfos[(taskId + 1) & 1].commonRunInfo.s2RealSize;
                        if (taskId > 0) {
                            ProcessNotFirst(runInfos[(taskId + 1) & 1]);
                        }
                        taskId++;
                        if (curTaskS2RealSize < this->constInfo.selectedCountOffset) {
                            break;
                        }
                    }
                    if (this->actualOriSelectedBlockCount >= this->actualSelectedBlockCount || tailSize == 0) {
                        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
                    }
                    if (deterTaskId == 0) {
                        if ASCEND_IS_AIC {
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
                        } else {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                        }
                    }
                    this->template SetDeterRunInfo<false>(deterRunInfos[deterTaskId & 1], sTaskId, deterTaskId,
                                                          this->cmpKvOffset);
                    if (deterTaskId > 0) {
                        if ASCEND_IS_AIC {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SCATTER_TO_FIX_SYNC_FLAG);
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_TO_FIX_SYNC_FLAG);
                            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
                        } else {
                            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                            this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm,
                                                           this->dOriKVWorkSpaceGm, this->dCmpKVWorkSpaceGm,
                                                           this->constInfo, deterRunInfos[(deterTaskId + 1) & 1]);
                            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SCATTER_TO_FIX_SYNC_FLAG);
                        }
                    }
                    deterTaskId++;
                }
            }
        }
        sTaskId++;
    }

    if (runInfos[(taskId + 1) & 1].taskStep == TASK_C1C2) {
        ProcessNotFirst(runInfos[(taskId + 1) & 1]);
    }
    if (this->processBS1ByCore > 0) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SCATTER_TO_FIX_SYNC_FLAG);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_TO_FIX_SYNC_FLAG);
            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
            this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm, this->dOriKVWorkSpaceGm,
                                           this->dCmpKVWorkSpaceGm, this->constInfo,
                                           deterRunInfos[(deterTaskId + 1) & 1]);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SCATTER_TO_FIX_SYNC_FLAG);
        }
        deterTaskId++;
    }

    if (this->processBS1ByCore == 0) {
        deterTaskId++;
    }

    if constexpr (IsOriKVExist) {
        if (this->processBS1ByCore < this->formerCoreProcessS1Num) {
            for (this->oriKvOffset = 0; this->oriKvOffset < this->maxOriSeqKV; this->oriKvOffset += PROCESS_KV_SIZE) {
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SCATTER_TO_FIX_SYNC_FLAG);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_TO_FIX_SYNC_FLAG);
                    CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                    CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                    this->template SetDeterRunInfo<true>(deterRunInfos[(deterTaskId + 1) & 1], sTaskId, deterTaskId + 1,
                                                         this->oriKvOffset);
                    this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm,
                                                   this->dOriKVWorkSpaceGm, this->dCmpKVWorkSpaceGm, this->constInfo,
                                                   deterRunInfos[(deterTaskId + 1) & 1]);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SCATTER_TO_FIX_SYNC_FLAG);
                }
                deterTaskId++;
            }
        }
    }
    if constexpr (IsCmpKVExist) {
        if (this->processBS1ByCore < this->formerCoreProcessS1Num) {
            for (this->cmpKvOffset = 0; this->cmpKvOffset < this->maxCmpSeqKV; this->cmpKvOffset += PROCESS_KV_SIZE) {
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SCATTER_TO_FIX_SYNC_FLAG);
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_TO_FIX_SYNC_FLAG);
                    CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                    CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SCATTER_SYNC_FLAG);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SCATTER_SYNC_FLAG);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SCATTER_SYNC_FLAG);
                    this->template SetDeterRunInfo<false>(deterRunInfos[(deterTaskId + 1) & 1], sTaskId,
                                                          deterTaskId + 1, this->cmpKvOffset);
                    this->vecBlock.ScatterAddDeter(this->mm4ResWorkSpaceGm, this->mm5ResWorkSpaceGm,
                                                   this->dOriKVWorkSpaceGm, this->dCmpKVWorkSpaceGm, this->constInfo,
                                                   deterRunInfos[(deterTaskId + 1) & 1]);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SCATTER_TO_FIX_SYNC_FLAG);
                }
                deterTaskId++;
            }
        }
    }
    this->FreeEventID();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessUnDeter()
{
    this->AllocEventID();
    if (this->constInfo.isSink) {
        this->vecBlock.CopySink(this->constInfo);
    }
    int64_t taskId = 0;
    int64_t sTaskId = 0;
    FagRunInfo runInfos[3]; // for cv ping pong
    for (int32_t i = 0; i < this->processBS1ByCore; i++) {
        bool isFirstBlock = true;
        this->usedT1Index = this->cBlockIdx + this->usedCoreNum * i;
        this->GetTndSeqLen(this->usedT1Index, this->bIndex);
        for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; this->n2Index++) {
            this->GetActualSelCount(this->usedT1Index, this->n2Index, this->actualSelectedBlockCount);
            if constexpr (IsOriKVExist) {
                for (this->blkCntOffset = 0; this->blkCntOffset < this->actualOriSelectedBlockCount;
                     this->blkCntOffset += this->constInfo.selectedCountOffset) {
                    if (taskId >= 0) {
                        this->template SetRunInfo<true>(runInfos[taskId % 3], runInfos[(taskId + 2) % 3], taskId,
                                                        sTaskId, 0, 0);
                        ProcessPreload(runInfos[taskId % 3], taskId, isFirstBlock);
                    }
                    int64_t curTaskS2RealSize = runInfos[taskId % 3].commonRunInfo.s2RealSize;
                    int64_t lastTaskS2RealSize = runInfos[(taskId + 1) % 3].commonRunInfo.s2RealSize;
                    if (taskId > 0) {
                        ProcessNotFirst(runInfos[(taskId + 2) % 3]);
                    }
                    if (taskId > 1) {
                        ScatterAddUnDeter(runInfos[(taskId + 1) % 3]);
                    }

                    taskId++;
                    if (curTaskS2RealSize < this->constInfo.selectedCountOffset) {
                        break;
                    }
                }
            }

            if constexpr (IsCmpKVExist) {
                for (this->blkCntOffset = this->actualOriSelectedBlockCount;
                     this->blkCntOffset < this->actualSelectedBlockCount;
                     this->blkCntOffset += this->constInfo.selectedCountOffset) {
                    if (taskId >= 0) {
                        this->template SetRunInfo<false>(runInfos[taskId % 3], runInfos[(taskId + 2) % 3], taskId,
                                                         sTaskId, 0, 0);
                        ProcessPreload(runInfos[taskId % 3], taskId, isFirstBlock);
                    }
                    int64_t curTaskS2RealSize = runInfos[taskId % 3].commonRunInfo.s2RealSize;
                    int64_t lastTaskS2RealSize = runInfos[(taskId + 1) % 3].commonRunInfo.s2RealSize;
                    if (taskId > 0) {
                        ProcessNotFirst(runInfos[(taskId + 2) % 3]);
                    }
                    if (taskId > 1) {
                        ScatterAddUnDeter(runInfos[(taskId + 1) % 3]);
                    }
                    taskId++;
                    if (curTaskS2RealSize < this->constInfo.selectedCountOffset) {
                        break;
                    }
                }
            }
        }
        if (this->actualSelectedBlockCount > 0) {
            sTaskId++;
        }
    }

    if (runInfos[(taskId + 2) % 3].taskStep == TASK_C1C2) {
        ProcessNotFirst(runInfos[(taskId + 2) % 3]);
    }
    if (runInfos[(taskId + 1) % 3].taskStep == TASK_C3C4C5) {
        ScatterAddUnDeter(runInfos[(taskId + 1) % 3]);
    }
    if (runInfos[(taskId + 2) % 3].taskStep == TASK_C3C4C5) {
        ScatterAddUnDeter(runInfos[(taskId + 2) % 3]);
    }
    this->FreeEventID();
}
} // namespace SfagBaseApi

#endif
