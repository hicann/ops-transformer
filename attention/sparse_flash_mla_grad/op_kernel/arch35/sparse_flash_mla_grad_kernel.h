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
    __aicore__ inline void SetUniqueRunInfo(FagRunInfo& runInfo);
    __aicore__ inline void SetUniqueConstInfo(FagConstInfo& constInfo);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessDeter();
    __aicore__ inline void ProcessUnDeter();
    __aicore__ inline void ProcessUnDeterKvPrefetch();

private:
    __aicore__ inline void ProcessPreload(FagRunInfo& runInfo, int64_t taskId, bool& isFirstBlock);
    __aicore__ inline void ProcessNotFirst(FagRunInfo& runInfo);
    __aicore__ inline void ProcessKVPrefetch(FagRunInfo& runInfo);
    __aicore__ inline void ProcessCube12(FagRunInfo& runInfo);
    __aicore__ inline void AdvanceKvPrefetch(FagRunInfo* runInfos, int64_t taskId);
    __aicore__ inline void ScatterAddUnDeter(FagRunInfo& runInfo);
    __aicore__ inline void CopyLse(FagRunInfo& runInfo);
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::SetUniqueRunInfo(FagRunInfo& runInfo)
{}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::SetUniqueConstInfo(
    FagConstInfo& constInfo)
{
    if constexpr (SplitG) {
        // Physical strides were computed with G=128 in SetConstInfo. Only the
        // matrix/Vector work size becomes 64; each AIV owns 32 of these heads.
        constInfo.commonConstInfo.gSize = 64;
        constInfo.halfGRealSize = 32;
        constInfo.firstHalfGRealSize = 32;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessPreload(FagRunInfo& runInfo,
                                                                                             int64_t taskId,
                                                                                             bool& isFirstBlock)
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
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessKVPrefetch(FagRunInfo& runInfo)
{
    runInfo.kvL1Slot = runInfo.commonRunInfo.taskId % 3;
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    int32_t s2RealSize = this->vecBlock.GatherKV(this->selectedKWorkSpaceGm, this->constInfo, runInfo);
    if ASCEND_IS_AIV {
        this->vecBlock.InitCubeVecSharedParams(runInfo, s2RealSize);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    } else {
        this->cubeBlock.GetCubeVecSharedParams(runInfo);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_V0_TO_C1_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        this->cubeBlock.PrefetchKV(this->selectedKWorkSpaceGm, this->constInfo, runInfo);
        // The GM gather slot is reusable after MTE2 has copied it to L1.
        // L1 has separate three-slot ownership and stays live through dQ.
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessCube12(FagRunInfo& runInfo)
{
    // Prefetch-ready only protects GM input. C1/C2 writes share UB with an
    // older scatter task, so preserve a distinct destination-free handshake.
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V_UB_TO_C12_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    } else {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V_UB_TO_C12_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V_UB_TO_C12_FLAG[runInfo.commonRunInfo.taskIdMod2]);
    }
    bool isFirstBlock = !runInfo.isS1IdxNoChange;
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
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::AdvanceKvPrefetch(FagRunInfo* runInfos,
                                                                                                int64_t taskId)
{
    ProcessKVPrefetch(runInfos[taskId & 3]);
    if (taskId > 0) {
        ProcessCube12(runInfos[(taskId + 3) & 3]);
    }
    if (taskId > 1) {
        ProcessNotFirst(runInfos[(taskId + 2) & 3]);
    }
    if (taskId > 2) {
        ScatterAddUnDeter(runInfos[(taskId + 1) & 3]);
    }
    if ASCEND_IS_AIC {
        if (!runInfos[taskId & 3].isS1IdxNoChange) {
            // With one-task lookahead, the query two slots behind may still
            // own Q/dY at entry. Its final dK/dV above must be queued first.
            // The prefetch API addresses contiguous G64 rows. For SplitG this
            // logical row is physical_query * 2 + head_half, not a KV index.
            this->cubeBlock.PrefetchQueryDy(runInfos[taskId & 3].commonRunInfo.queryOffset / (64 * 512),
                                            runInfos[taskId & 3].sTaskIdMod2);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessNotFirst(FagRunInfo& runInfo)
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
    if constexpr (IsDETER && BaseClass::CUBE_BASEN == 64) {
        if ASCEND_IS_AIC {
            if (runInfo.isSparse && runInfo.commonRunInfo.s2RealSize > 0) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SOFTMAX_L1_SYNC_FLAG);
                CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SOFTMAX_L1_SYNC_FLAG);
            }
        }
    }
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
    if constexpr (IsDETER) {
        this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
            this->dqWorkSpaceGm, this->selectedKWorkSpaceGm, this->dSL1Buf, this->constInfo, runInfo);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
    }
    if ASCEND_IS_AIC {
        if constexpr (!IsDETER) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_V5_TO_C4_FLAG[runInfo.commonRunInfo.taskIdMod2]);
        }
    }
    // compute dk
    GlobalTensor<CALC_TYPE> dkOutTensor = this->mm4ResWorkSpaceGm;
    if constexpr (!IsDETER) {
        if (!runInfo.isSparse) {
            dkOutTensor = runInfo.isOriKV ? this->dOriKVWorkSpaceGm : this->dCmpKVWorkSpaceGm;
        }
    }
    this->cubeBlock.template IterateMmDsQ<CALC_TYPE, BaseClass::IS_DK_WRITE_UB>(dkOutTensor, this->dSL1Buf,
                                                                                this->constInfo, runInfo); // c4
    if constexpr (!IsDETER) {
        // compute dq
        if (this->directDqOutput) {
            this->cubeBlock.template IterateMmDsK<INPUT_TYPE, false>(this->dqDirectGm, this->selectedKWorkSpaceGm,
                                                                     this->dSL1Buf, this->constInfo, runInfo); // c3
        } else {
            this->cubeBlock.template IterateMmDsK<CALC_TYPE, BaseClass::IS_DQ_WRITE_UB>(
                this->dqWorkSpaceGm, this->selectedKWorkSpaceGm, this->dSL1Buf, this->constInfo, runInfo); // c3
        }
        if constexpr (BaseClass::CUBE_BASEN != 64) {
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(16 + SYNC_C3_TO_V0_FLAG[runInfo.commonRunInfo.taskIdMod2]);
            }
        }
    }
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
    GlobalTensor<CALC_TYPE> dvOutTensor = this->mm5ResWorkSpaceGm;
    if constexpr (!IsDETER) {
        if (!runInfo.isSparse) {
            dvOutTensor = runInfo.isOriKV ? this->dOriVWorkSpaceGm : this->dCmpVWorkSpaceGm;
        }
    }
    this->cubeBlock.template IterateMmPDy<CALC_TYPE, BaseClass::IS_DV_WRITE_UB>(dvOutTensor, this->pL1Buf,
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
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ScatterAddUnDeter(FagRunInfo& runInfo)
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
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessUnDeterKvPrefetch()
{
    this->AllocEventID();
    if (this->constInfo.isSink) {
        this->vecBlock.CopySink(this->constInfo);
    }
    int64_t taskId = 0;
    int64_t sTaskId = 0;
    FagRunInfo runInfos[4];
    for (int32_t i = 0; i < this->processBS1ByCore; ++i) {
        this->usedT1Index = this->cBlockIdx + this->usedCoreNum * i;
        this->GetTndSeqLen(this->usedT1Index, this->bIndex);
        for (this->n2Index = 0; this->n2Index < this->tilingData->baseParams.n2; ++this->n2Index) {
            this->GetActualSelCount(this->usedT1Index, this->n2Index, this->actualSelectedBlockCount);
            // Complete all KV tiles for one G64 group before changing groups:
            // only one dQ accumulator is resident in L0C. Q/dY ping-pong slots
            // keep the next half live while the preceding half drains.
            constexpr uint32_t headGroups = SplitG ? 2 : 1;
            for (uint32_t headGroup = 0; headGroup < headGroups; ++headGroup) {
                this->currentGOffset = headGroup * 64;
                if constexpr (IsOriKVExist) {
                    for (this->blkCntOffset = 0; this->blkCntOffset < this->actualOriSelectedBlockCount;
                         this->blkCntOffset += this->constInfo.selectedCountOffset) {
                        this->template SetRunInfo<true>(runInfos[taskId & 3], runInfos[(taskId + 3) & 3], taskId,
                                                        sTaskId, 0, 0);
                        AdvanceKvPrefetch(runInfos, taskId);
                        int64_t realSize = runInfos[taskId & 3].commonRunInfo.s2RealSize;
                        ++taskId;
                        if (realSize < this->constInfo.selectedCountOffset) {
                            break;
                        }
                    }
                }
                if constexpr (IsCmpKVExist) {
                    for (this->blkCntOffset = this->actualOriSelectedBlockCount;
                         this->blkCntOffset < this->actualSelectedBlockCount;
                         this->blkCntOffset += this->constInfo.selectedCountOffset) {
                        this->template SetRunInfo<false>(runInfos[taskId & 3], runInfos[(taskId + 3) & 3], taskId,
                                                         sTaskId, 0, 0);
                        AdvanceKvPrefetch(runInfos, taskId);
                        int64_t realSize = runInfos[taskId & 3].commonRunInfo.s2RealSize;
                        ++taskId;
                        if (realSize < this->constInfo.selectedCountOffset) {
                            break;
                        }
                    }
                }
                if (this->actualSelectedBlockCount > 0) {
                    ++sTaskId;
                }
            }
        }
    }
    // Drain exactly the outstanding stages, including cores with only one task.
    if (taskId > 0) {
        ProcessCube12(runInfos[(taskId - 1) & 3]);
    }
    if (taskId > 1) {
        ProcessNotFirst(runInfos[(taskId - 2) & 3]);
    }
    if (taskId > 2) {
        ScatterAddUnDeter(runInfos[(taskId - 3) & 3]);
    }
    if (taskId > 0) {
        ProcessNotFirst(runInfos[(taskId - 1) & 3]);
    }
    if (taskId > 1) {
        ScatterAddUnDeter(runInfos[(taskId - 2) & 3]);
    }
    if (taskId > 0) {
        ScatterAddUnDeter(runInfos[(taskId - 1) & 3]);
    }
    this->FreeEventID();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaGradKernel<CubeBlockType, VecBlockType>::ProcessUnDeter()
{
    if constexpr (BaseClass::CUBE_BASEN == 64) {
        ProcessUnDeterKvPrefetch();
        return;
    }
    this->AllocEventID();
    if (this->constInfo.isSink) {
        this->vecBlock.CopySink(this->constInfo);
    }
    int64_t taskId = 0;
    int64_t sTaskId = 0;
    FagRunInfo runInfos[3]{}; // for cv ping pong
    if constexpr (BaseClass::CUBE_BASEN == 64) {
        if ASCEND_IS_AIC {
            if (this->processBS1ByCore > 0) {
                this->cubeBlock.PrefetchQueryDy(this->cBlockIdx, 0);
            }
        }
    }

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
        if constexpr (BaseClass::CUBE_BASEN == 64) {
            if ASCEND_IS_AIC {
                if (i + 1 < this->processBS1ByCore) {
                    // Current KV loads are already queued. Overlap next Q/dY
                    // with outstanding gradient work and next-query metadata.
                    this->cubeBlock.PrefetchQueryDy(this->usedT1Index + this->usedCoreNum, sTaskId & 1);
                }
            }
        }
    }

    if (taskId > 0 && runInfos[(taskId + 2) % 3].taskStep == TASK_C1C2) {
        ProcessNotFirst(runInfos[(taskId + 2) % 3]);
    }
    if (taskId > 1 && runInfos[(taskId + 1) % 3].taskStep == TASK_C3C4C5) {
        ScatterAddUnDeter(runInfos[(taskId + 1) % 3]);
    }
    if (taskId > 0 && runInfos[(taskId + 2) % 3].taskStep == TASK_C3C4C5) {
        ScatterAddUnDeter(runInfos[(taskId + 2) % 3]);
    }
    this->FreeEventID();
}
} // namespace SfagBaseApi

#endif
