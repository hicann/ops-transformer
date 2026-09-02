/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file sparse_lightning_indexer_kl_loss_grad_kernel_base.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_KERNEL_BASE_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_KERNEL_BASE_H

#include "sparse_lightning_indexer_kl_loss_grad_tiling_regbase.h"
#include "sparse_lightning_indexer_kl_loss_grad_regbase_common.h"
#include "sparse_lightning_indexer_kl_loss_grad_cube_block.h"
#include "sparse_lightning_indexer_kl_loss_grad_vector_block.h"

namespace Slig {
template <typename CubeBlockType, typename VecBlockType>
class SparseLightningIndexerKLLossGradKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline SparseLightningIndexerKLLossGradKernelBase(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR weight, GM_ADDR sparseIndices,
                                GM_ADDR attnSoftmaxL1, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ,
                                GM_ADDR sequsedK, GM_ADDR cmpResidualK, GM_ADDR metadata, GM_ADDR dQuery, GM_ADDR dKey,
                                GM_ADDR dWeight, GM_ADDR softmaxOut, GM_ADDR workspace,
                                const optiling::SparseLightningIndexerGradRegBaseTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessDeter();
    __aicore__ inline void ProcessNormal();
    __aicore__ inline void ProcessSY(SLIGradRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx);
    __aicore__ inline void ProcessP(SLIGradRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx);
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void FreeBuf();
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline int32_t GetS2SparseLen(int32_t s1Idx, int32_t actualSeqLensQ, int32_t actualSeqLensK,
                                             int32_t cmpResidualK, int64_t cmpRatio, SLISparseMode sparseMode);
    __aicore__ inline void SetRunInfo(SLIGradRunInfo &runInfo, int64_t taskId);
    __aicore__ inline void SetRunInfoP(SLIGradRunInfo &runInfo);
    __aicore__ inline void SetSYRunInfo(SLIGradKRunInfo &kRunInfo, SLIGradRunInfo &runInfo, int64_t s2TaskId);
    __aicore__ inline void SetPRunInfo(SLIGradKRunInfo &kRunInfo, SLIGradRunInfo &runInfo, int64_t s2TaskId);
    __aicore__ inline void SetSYSingleRunInfo(SLIGradKRunInfo &kRunInfo, SLIGradRunInfo &runInfo, int64_t s2TaskId,
                                              int64_t taskId);
    __aicore__ inline void SetPSingleRunInfo(SLIGradKRunInfo &kRunInfo, SLIGradRunInfo &runInfo, int64_t s2TaskId,
                                             int64_t taskId);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline int32_t GetCmpResidualK(int32_t bIdx, int32_t defaultLens);
    __aicore__ inline void GetTndSeqLen(const int64_t t1Idx, int64_t &bIdx);

    TPipe *pipe = nullptr;
    const optiling::SparseLightningIndexerGradRegBaseTilingData *__restrict tilingData = nullptr;

    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GM_ADDR usedSeqQlenAddr;
    GM_ADDR usedSeqKvlenAddr;
    GM_ADDR cmpResidualKAddr;
    GM_ADDR metadataGm;

    // workspace
    GlobalTensor<T> reduceSumRes;
    GlobalTensor<T> reluRes;
    GlobalTensor<INPUT_T> gatherSYRes;
    GlobalTensor<T> scatterAddResGm;
    GlobalTensor<T> scatterAddResPong;
    GlobalTensor<T> softmaxOutGm;
    GlobalTensor<T> dWeightGm;
    GlobalTensor<OUT_T> dqGm;
    GlobalTensor<T> bmm3Res;
    // CV间共享buffer
    TBuf<> mm12ResBuf[3]; // UB上3buffer
    TBuf<> mm3ResBuf;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;

    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm3Buffer;
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> sYL1Buf;
    BuffersPolicySingleBuffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> reluGradResL1Buf;

    SLIGradConstInfo constInfo;
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    int64_t coreNum;

    int64_t curS1 = 0;
    int64_t curS2 = 0;
    int64_t usedS1 = 0;
    int64_t usedS2 = 0;

    int64_t processBS1ByCore = 0;
    int64_t usedCoreNum = 0;
    uint32_t cBlockIdx = 0;
    uint32_t vBlockIdx = 0;
    int64_t usedT1Index = 0; // 用于负载均衡
    int64_t usedT1Offset = 0;
    int64_t nextUsedT1Offset = 0;
    int64_t bIndex = HasSequsedQ ? -1 : 0;
    int64_t s1Index = 0;
    int64_t t1Index = 0;
    int64_t t2Index = 0;

    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t weightPreBlockFactor;
    uint32_t weightPreBlockTotal;
    uint32_t weightPreBlockTail;
    uint32_t softmaxOutPreBlockFactor;
    uint32_t softmaxOutPreBlockTotal;
    uint32_t softmaxOutPreBlockTail;
    uint32_t initdqSize;
    uint64_t dqOffset;
    uint32_t initdweightSize;
    uint64_t dweightOffset;
    uint32_t initsoftMaxSize;
    uint64_t softMaxOffset;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR attnSoftmaxL1, GM_ADDR cuSeqlensQ,
    GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK, GM_ADDR metadata, GM_ADDR dQuery,
    GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR softmaxOut, GM_ADDR workspace,
    const optiling::SparseLightningIndexerGradRegBaseTilingData *__restrict tiling, TPipe *tPipe)
{
    pipe = tPipe;
    tilingData = tiling;
    metadataGm = metadata;
    SetConstInfo();
    actualSeqQlenAddr = cuSeqlensQ;
    actualSeqKvlenAddr = cuSeqlensK;
    usedSeqQlenAddr = sequsedQ;
    usedSeqKvlenAddr = sequsedK;
    cmpResidualKAddr = cmpResidualK;
    softmaxOutGm.SetGlobalBuffer((__gm__ T *)softmaxOut);
    dWeightGm.SetGlobalBuffer((__gm__ T *)dWeight);
    dqGm.SetGlobalBuffer((__gm__ OUT_T *)dQuery);

    InitMMResBuf();
    InitWorkspace(workspace);

    // init vec block
    if ASCEND_IS_AIV {
        // InitVecOP
        vecBlock.InitParams(constInfo, tilingData);
        vecBlock.InitGlobalBuffer(key, weight, sparseIndices, attnSoftmaxL1, dKey, dWeight, softmaxOut, gatherSYRes,
                                  reluRes, reduceSumRes, scatterAddResGm, scatterAddResPong, bmm3Res, cuSeqlensQ,
                                  cuSeqlensK, sequsedQ, sequsedK, cmpResidualK);
        vecBlock.InitBuffers(pipe);
    } else if ASCEND_IS_AIC {
        // initCubeOP
        cubeBlock.SetCubeBlockParams(tPipe, &l1BufferManager);
        cubeBlock.InitCubeBuffers();
        cubeBlock.InitGlobalBuffer(query, dQuery, reluRes, gatherSYRes, bmm3Res);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::ProcessSY(
    SLIGradRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx)
{
    bool notLastLoop = true;
    SLIGradKRunInfo kRunInfos[MODE_NUM_2];
    if ASCEND_IS_AIV {
        vecBlock.CopyInWeight(runInfo);
    }
    for (int32_t s2TaskId = 0; s2TaskId < runInfo.s2LoopTimes + 1; ++s2TaskId) {
        SLIGradKRunInfo &kRunInfoNeg0 = kRunInfos[s2TaskId % MODE_NUM_2];
        SLIGradKRunInfo &kRunInfoNeg1 = kRunInfos[(s2TaskId + 1) % MODE_NUM_2];
        if (s2TaskId == runInfo.s2LoopTimes) {
            notLastLoop = false;
        }
        if (s2TaskId >= 0 && notLastLoop) {
            SetSYRunInfo(kRunInfoNeg0, runInfo, s2TaskId);
            if ASCEND_IS_AIV {
                for (int32_t s2InnerIdx = 0; s2InnerIdx < kRunInfoNeg0.s2SingleLoopTimes; ++s2InnerIdx) {
                    SetSYSingleRunInfo(kRunInfoNeg0, runInfo, s2TaskId, s2InnerIdx);
                    vecBlock.ProcessVector0(this->sYL1Buf.Get(), runInfo, kRunInfoNeg0);
                }
            }
            if ASCEND_IS_AIC {
                cubeBlock.ComputeMmSy(this->bmm1Buffers.Get(), sYL1Buf, runInfo, constInfo, kRunInfoNeg0);
            }
        }

        if (s2TaskId >= 1) {
            if ASCEND_IS_AIV {
                Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Res = this->bmm1Buffers.Get();
                for (int32_t s2InnerIdx = 0; s2InnerIdx < kRunInfoNeg1.s2SingleLoopTimes; ++s2InnerIdx) {
                    SetSYSingleRunInfo(kRunInfoNeg1, runInfo, s2TaskId - 1, s2InnerIdx);
                    vecBlock.ProcessVector1(bmm1Res, runInfo, kRunInfoNeg1);
                }
            }
        }
    }
    if ASCEND_IS_AIV {
        vecBlock.ProcessVector2(this->bmm1Buffers.GetPre(), runInfo);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::ProcessP(
    SLIGradRunInfo &runInfo, int64_t taskId, int64_t bIdx, int64_t s1Idx)
{
    bool notLastLoop = true;
    bool notLastSecondLoop = true;
    bool notSecondLast = true;
    SLIGradKRunInfo kRunInfos[MODE_NUM_3];
    bool isFixOut = false;
    for (int32_t s2TaskId = 0; s2TaskId < runInfo.s2LoopTimes + 2; ++s2TaskId) {
        SLIGradKRunInfo &kRunInfoNeg0 = kRunInfos[s2TaskId % MODE_NUM_3];
        SLIGradKRunInfo &kRunInfoNeg2 = kRunInfos[(s2TaskId + 1) % MODE_NUM_3];
        SLIGradKRunInfo &kRunInfoNeg1 = kRunInfos[(s2TaskId + 2) % MODE_NUM_3];
        if (s2TaskId == runInfo.s2LoopTimes + 1) {
            notLastLoop = false;
        } else if (s2TaskId == runInfo.s2LoopTimes) {
            notSecondLast = false;
        }
        notLastSecondLoop = notLastLoop && notSecondLast;
        if (s2TaskId >= 0 && notLastSecondLoop) {
            SetPRunInfo(kRunInfoNeg0, runInfo, s2TaskId);
            if ASCEND_IS_AIV {
                for (int32_t s2InnerIdx = 0; s2InnerIdx < kRunInfoNeg0.s2SingleLoopTimes; ++s2InnerIdx) {
                    SetPSingleRunInfo(kRunInfoNeg0, runInfo, s2TaskId, s2InnerIdx);
                }
            }
        }
        if (s2TaskId >= 1 && notLastLoop) {
            if ASCEND_IS_AIV {
                for (int32_t s2InnerIdx = 0; s2InnerIdx < kRunInfoNeg1.s2SingleLoopTimes; ++s2InnerIdx) {
                    SetPSingleRunInfo(kRunInfoNeg1, runInfo, s2TaskId - 1, s2InnerIdx);
                    vecBlock.ProcessVector6(this->reluGradResL1Buf.Get(), runInfo, kRunInfoNeg1);
                }
            }
        }
        if (s2TaskId >= 2) {
            if ASCEND_IS_AIC {
                isFixOut = !notLastLoop;
                cubeBlock.ComputeMm3(bmm3Buffer, this->reluGradResL1Buf.Get(), runInfo, constInfo, kRunInfoNeg2);
                cubeBlock.ComputeMm4(reluGradResL1Buf.Get(), runInfo, constInfo, kRunInfoNeg2, isFixOut);
            }
            if ASCEND_IS_AIV {
                if constexpr (!Deterministic) {
                    for (int32_t s2InnerIdx = 0; s2InnerIdx < kRunInfoNeg2.s2SingleLoopTimes; ++s2InnerIdx) {
                        SetPSingleRunInfo(kRunInfoNeg2, runInfo, s2TaskId - 2, s2InnerIdx);
                        vecBlock.ProcessVector7(this->bmm3Buffer.Get(), runInfo, kRunInfoNeg2);
                    }
                }
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::GetTndSeqLen(
    const int64_t t1Idx, int64_t &bIdx)
{
    int64_t t1Offset = 0;
    int64_t t2Offset = 0;

    if constexpr (!HasSequsedQ) {
        if constexpr (Layout_Q == SLILayout::TND) {
            int64_t curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            while (usedT1Index >= curT1) {
                curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[++bIndex];
            }
            bIndex = bIndex - 1;
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            s1Index = usedT1Index - t1Offset;
        } else {
            t1Offset = usedT1Index;
            bIndex = usedT1Index / curS1;
            s1Index = usedT1Index % constInfo.s1Size;
        }
        t1Index = usedT1Index;
    } else {
        while (usedT1Index >= nextUsedT1Offset) {
            usedT1Offset = nextUsedT1Offset;
            nextUsedT1Offset = ((__gm__ int32_t *)usedSeqQlenAddr)[++bIndex] + usedT1Offset;
        }
        s1Index = usedT1Index - usedT1Offset;
        if constexpr (Layout_Q == SLILayout::TND) {
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
        } else {
            t1Offset = bIndex * constInfo.s1Size;
        }
        t1Offset += s1Index;
        t1Index = t1Offset;
    }

    if constexpr (Layout_Q == SLILayout::TND) {
        if (unlikely(bIndex == 0)) {
            t2Offset = 0;
            curS1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex + 1];
            curS2 = ((__gm__ int32_t *)actualSeqKvlenAddr)[bIndex + 1];
        } else {
            curS1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex + 1] - ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            curS2 = ((__gm__ int32_t *)actualSeqKvlenAddr)[bIndex + 1] - ((__gm__ int32_t *)actualSeqKvlenAddr)[bIndex];
            t2Offset = ((__gm__ int32_t *)actualSeqKvlenAddr)[bIndex];
        }
    } else {
        t2Offset = bIndex * curS2;
    }

    if constexpr (HasSequsedQ) {
        usedS1 = ((__gm__ int32_t *)usedSeqQlenAddr)[bIndex];
    } else {
        usedS1 = curS1;
    }
    if constexpr (HasSequsedK) {
        usedS2 = ((__gm__ int32_t *)usedSeqKvlenAddr)[bIndex];
    } else {
        usedS2 = curS2;
    }
    t2Index = t2Offset;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::Process()
{
    if constexpr (Deterministic) {
        ProcessDeter();
    } else {
        ProcessNormal();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::ProcessNormal()
{
    int64_t taskId = 0;
    SLIGradRunInfo runInfo;
    for (int32_t i = 0; i < processBS1ByCore; i++) {
        this->usedT1Index = this->cBlockIdx + this->usedCoreNum * i;
        GetTndSeqLen(usedT1Index, bIndex);
        SetRunInfo(runInfo, taskId);
        if (runInfo.s2LoopTimes == 0) {
            taskId++;
            continue;
        }
        ProcessSY(runInfo, taskId, bIndex, s1Index);
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_RELU_TO_V6_FLAG);
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_ST_TO_P_FLAG);
        } else {
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_RELU_TO_V6_FLAG);
            CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_ST_TO_P_FLAG);
        }
        SetRunInfoP(runInfo);
        ProcessP(runInfo, taskId, bIndex, s1Index);
        taskId++;
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        vecBlock.ReInitBuffers(pipe);
        vecBlock.ProcessVector8();
    }
    FreeBuf();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::ProcessDeter()
{
    int64_t taskId = 0;
    SLIGradRunInfo runInfo;
    for (int32_t i = 0; i < processBS1ByCore; i++) {
        this->usedT1Index = this->cBlockIdx + this->usedCoreNum * i;
        GetTndSeqLen(usedT1Index, bIndex);
        SetRunInfo(runInfo, taskId);
        if (runInfo.s2LoopTimes == 0) {
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_V7_TO_C3_FLAG[runInfo.taskIdMod2]);
                CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C3_TO_V7_FLAG[runInfo.taskIdMod2]);
            }
            vecBlock.ProcessVector7Deter(runInfo, this->bmm3Buffer.Get());
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V7_TO_C3_FLAG[runInfo.taskIdMod2]);
            taskId++;
            continue;
        }
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V6_TO_RELU_FLAG);
        }
        ProcessSY(runInfo, taskId, bIndex, s1Index);
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_RELU_TO_V6_FLAG);
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_ST_TO_P_FLAG);
        } else {
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_RELU_TO_V6_FLAG);
            CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_ST_TO_P_FLAG);
        }
        SetRunInfoP(runInfo);
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_V7_TO_C3_FLAG[runInfo.taskIdMod2]);
        }
        ProcessP(runInfo, taskId, bIndex, s1Index);
        if ASCEND_IS_AIC {
            CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_V6_TO_RELU_FLAG);
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C3_TO_V7_FLAG[runInfo.taskIdMod2]);
        }
        if ASCEND_IS_AIV {
            vecBlock.ProcessVector7Deter(runInfo, this->bmm3Buffer.Get());
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V7_TO_C3_FLAG[runInfo.taskIdMod2]);
        }
        taskId++;
    }
    if (processBS1ByCore < constInfo.formerCoreProcessNNum) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_V7_TO_C3_FLAG[runInfo.taskIdMod2]);
            CrossCoreSetFlag<MODE_NUM_2, PIPE_FIX>(SYNC_C3_TO_V7_FLAG[taskId & 1]); // 额外空同步确保每个核处理+1轮
        }
        if ASCEND_IS_AIV {
            SLIGradRunInfo runInfoExtra;
            runInfoExtra.taskId = taskId;
            runInfoExtra.taskIdMod2 = taskId & 1;
            vecBlock.ProcessVector7Deter(runInfoExtra, this->bmm3Buffer.Get());
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V7_TO_C3_FLAG[runInfo.taskIdMod2]);
        }
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        vecBlock.ReInitBuffers(pipe);
        vecBlock.ProcessVector8();
    }
    FreeBuf();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int32_t SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::GetS2SparseLen(
    int32_t s1Idx, int32_t actualSeqLensQ, int32_t actualSeqLensK, int32_t cmpResidualK, int64_t cmpRatio,
    SLISparseMode sparseMode)
{
    if (sparseMode == SLISparseMode::RightDown) {
        if (cmpRatio == 1) {
            return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
        }
        return Max(((actualSeqLensK * cmpRatio + cmpResidualK) - actualSeqLensQ + s1Idx + 1) / cmpRatio, 0);
    } else if (sparseMode == SLISparseMode::DefaultMask) {
        return actualSeqLensK;
    } else {
        return 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetRunInfo(
    SLIGradRunInfo &runInfo, int64_t taskId)
{
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.bIdx = bIndex;
    runInfo.t1Index = t1Index;
    runInfo.t2Index = t2Index;
    runInfo.s1Idx = s1Index;
    runInfo.actS1Size = usedS1;
    runInfo.actS2Size = usedS2;
    runInfo.accumS1Idx = runInfo.t1Index;
    runInfo.accumS2Idx = runInfo.t2Index;

    int32_t cmpResidualK = GetCmpResidualK(runInfo.bIdx, 0);
    runInfo.cmpResidualK = cmpResidualK;
    runInfo.kBaseSize = 2048;
    runInfo.s2SparseLen = GetS2SparseLen(runInfo.s1Idx, runInfo.actS1Size, runInfo.actS2Size, runInfo.cmpResidualK,
                                         constInfo.cmpRatio, constInfo.sparseMode);
    runInfo.s2RealSize = Min(constInfo.kSize, runInfo.s2SparseLen);
    runInfo.kRealSize = runInfo.s2RealSize;
    runInfo.kRealSizeAlign8 = (runInfo.kRealSize + 7) >> 3 << 3;
    runInfo.s2LoopTimes = CeilDiv(runInfo.s2RealSize, constInfo.syKBaseSize);
    int64_t tt = (runInfo.s2RealSize + constInfo.syKBaseSize - 1) / constInfo.syKBaseSize;
    runInfo.s2TailSize = (runInfo.s2RealSize % constInfo.syKBaseSize == 0) ?
                             constInfo.syKBaseSize :
                             (runInfo.s2RealSize % constInfo.syKBaseSize);
    runInfo.s2BaseSize = VEC_SY_BASESIZE;

    runInfo.kLoopTimes = CeilDiv(runInfo.kRealSize, runInfo.kBaseSize);
    runInfo.kTailSize =
        (runInfo.kRealSize % runInfo.kBaseSize == 0) ? runInfo.kBaseSize : (runInfo.kRealSize % runInfo.kBaseSize);

    runInfo.queryIndexTensorOffset = runInfo.t1Index * constInfo.gSizeQueryIndex * constInfo.dSizeQueryIndex;
    runInfo.firstNIndexSize = AlignTo(constInfo.gSizeQueryIndex, ALIGN_NUM_2) / 2;
    if (constInfo.subBlockIdx == 0) {
        runInfo.nIndexSize = runInfo.firstNIndexSize;
        runInfo.nVecSize = AlignTo(constInfo.gSizeQuery, ALIGN_NUM_2) / 2;
    } else {
        runInfo.nIndexSize = constInfo.gSizeQueryIndex - runInfo.firstNIndexSize;
        runInfo.nVecSize = constInfo.gSizeQuery - AlignTo(constInfo.gSizeQuery, ALIGN_NUM_2) / 2;
    }

    runInfo.topkGmBaseOffset = runInfo.t1Index * constInfo.kSize;
    runInfo.weightOffset =
        runInfo.t1Index * constInfo.gSizeQueryIndex + runInfo.firstNIndexSize * constInfo.subBlockIdx;
    runInfo.softmaxInputOffset =
        runInfo.t1Index * constInfo.gSizeQuery + AlignTo(constInfo.gSizeQuery, ALIGN_NUM_2) * constInfo.subBlockIdx / 2;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetRunInfoP(
    SLIGradRunInfo &runInfo)
{
    runInfo.s2LoopTimes = CeilDiv(runInfo.s2RealSize, constInfo.pKBaseSize);
    runInfo.s2TailSize = (runInfo.s2RealSize % constInfo.pKBaseSize == 0) ? constInfo.pKBaseSize :
                                                                            (runInfo.s2RealSize % constInfo.pKBaseSize);
    runInfo.s2BaseSize = VEC_P_BASESIZE;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetSYRunInfo(
    SLIGradKRunInfo &kRunInfo, SLIGradRunInfo &runInfo, int64_t s2TaskId)
{
    runInfo.s2TailSize =
        (runInfo.s2RealSize % VEC_SY_BASESIZE == 0) ? VEC_SY_BASESIZE : (runInfo.s2RealSize % VEC_SY_BASESIZE);
    kRunInfo.kTaskId = s2TaskId;
    kRunInfo.kTaskIdMod2 = kRunInfo.kTaskId & 1;
    kRunInfo.kProcessSize = (kRunInfo.kTaskId == runInfo.s2LoopTimes - 1) ?
                                (runInfo.s2RealSize - (runInfo.s2LoopTimes - 1) * constInfo.syKBaseSize) :
                                constInfo.syKBaseSize;
    kRunInfo.s2SingleLoopTimes = CeilDiv(kRunInfo.kProcessSize, VEC_SY_BASESIZE);
    kRunInfo.dValue = constInfo.dSizeQueryIndex;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetPRunInfo(
    SLIGradKRunInfo &kRunInfo, SLIGradRunInfo &runInfo, int64_t s2TaskId)
{
    runInfo.s2TailSize =
        (runInfo.s2RealSize % VEC_P_BASESIZE == 0) ? VEC_P_BASESIZE : (runInfo.s2RealSize % VEC_P_BASESIZE);
    kRunInfo.kTaskId = s2TaskId;
    kRunInfo.kTaskIdMod2 = kRunInfo.kTaskId & 1;
    kRunInfo.kProcessSize = (kRunInfo.kTaskId == runInfo.s2LoopTimes - 1) ?
                                (runInfo.s2RealSize - (runInfo.s2LoopTimes - 1) * constInfo.pKBaseSize) :
                                constInfo.pKBaseSize;
    kRunInfo.s2SingleLoopTimes = CeilDiv(kRunInfo.kProcessSize, VEC_P_BASESIZE);
    kRunInfo.dValue = constInfo.dSizeQuery;
    kRunInfo.dRopeValue = constInfo.dSizeRope;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetSYSingleRunInfo(
    SLIGradKRunInfo &kRunInfo, SLIGradRunInfo &runInfo, int64_t s2TaskId, int64_t taskId)
{
    kRunInfo.s2SingleIdx = taskId;
    kRunInfo.s2Idx = s2TaskId * constInfo.syKBaseSize / VEC_SY_BASESIZE + kRunInfo.s2SingleIdx;
    kRunInfo.isS2end = kRunInfo.s2SingleIdx >= kRunInfo.s2SingleLoopTimes - 1 && s2TaskId >= runInfo.s2LoopTimes - 1;
    kRunInfo.s2RealBaseSize = kRunInfo.isS2end ? runInfo.s2TailSize : VEC_SY_BASESIZE;
    runInfo.s2CurSize = s2TaskId * constInfo.syKBaseSize + kRunInfo.s2SingleIdx * VEC_SY_BASESIZE;
    kRunInfo.s2SingleCurSize = kRunInfo.s2SingleIdx * VEC_SY_BASESIZE;
    kRunInfo.isAlign64 = (!kRunInfo.isS2end) || (kRunInfo.isS2end && runInfo.s2TailSize % 64 == 0);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetPSingleRunInfo(
    SLIGradKRunInfo &kRunInfo, SLIGradRunInfo &runInfo, int64_t s2TaskId, int64_t taskId)
{
    kRunInfo.s2SingleIdx = taskId;
    kRunInfo.s2Idx = s2TaskId * constInfo.pKBaseSize / VEC_P_BASESIZE + kRunInfo.s2SingleIdx;
    kRunInfo.isS2end = kRunInfo.s2SingleIdx >= kRunInfo.s2SingleLoopTimes - 1 && s2TaskId >= runInfo.s2LoopTimes - 1;
    kRunInfo.s2RealBaseSize = kRunInfo.isS2end ? runInfo.s2TailSize : VEC_P_BASESIZE;
    runInfo.s2CurSize = s2TaskId * constInfo.pKBaseSize + kRunInfo.s2SingleIdx * VEC_P_BASESIZE;
    kRunInfo.s2SingleCurSize = kRunInfo.s2SingleIdx * VEC_P_BASESIZE;
    kRunInfo.isAlign64 = (!kRunInfo.isS2end) || (kRunInfo.isS2end && runInfo.s2TailSize % 64 == 0);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetConstInfo()
{
    if ASCEND_IS_AIV {
        constInfo.aivIdx = GetBlockIdx();
        constInfo.aicIdx = constInfo.aivIdx / 2;
        constInfo.subBlockIdx = constInfo.aivIdx % MODE_NUM_2;
        cBlockIdx = constInfo.aicIdx;
        vBlockIdx = constInfo.aivIdx;
    } else {
        constInfo.aicIdx = GetBlockIdx();
        cBlockIdx = constInfo.aicIdx;
    }
    coreNum = static_cast<int64_t>(GetBlockNum());
    ;
    constInfo.totalSize = tilingData->multiCoreParams.totalSize;
    auto &baseInfo = tilingData->baseParams;
    constInfo.bSize = baseInfo.bSize;
    constInfo.n2Size = baseInfo.n2Size;
    constInfo.gSizeQuery = 128;
    constInfo.gSizeQueryIndex = baseInfo.gSizeQueryIndex;
    constInfo.s1Size = baseInfo.s1Size;
    constInfo.s2Size = baseInfo.s2Size;
    constInfo.kSize = baseInfo.kSize;
    constInfo.kSizeAlign128 = (constInfo.kSize + 127) >> 7 << 7;
    constInfo.dSizeQuery = 512;
    constInfo.dSizeQueryIndex = baseInfo.dSizeQueryIndex;
    constInfo.dSizeRope = 64;
    constInfo.sparseMode = static_cast<SLISparseMode>(baseInfo.sparseMode);
    constInfo.cmpRatio = baseInfo.cmpRatio;
    constInfo.sparseBlockSize = 1;
    constInfo.gatherKBaseSize = 128;
    constInfo.syKBaseSize = 16384 / (constInfo.n2Size * constInfo.gSizeQueryIndex);
    constInfo.pKBaseSize = 16384 / (constInfo.n2Size * constInfo.gSizeQueryIndex);
    constInfo.syKBaseSize = Min(1024, constInfo.syKBaseSize);
    constInfo.pKBaseSize = Min(1024, constInfo.pKBaseSize);
    constInfo.pKBaseSize = 128;
    constInfo.syKBaseSize = 128;
    constInfo.pScaler = 1.0f / static_cast<float>(static_cast<int64_t>(constInfo.gSizeQuery));
    constInfo.totalNum = ((__gm__ uint32_t *)metadataGm)[0];
    constInfo.formerCoreProcessNNum = ((__gm__ uint32_t *)metadataGm)[1];
    constInfo.remainCoreProcessNNum = ((__gm__ uint32_t *)metadataGm)[2];
    constInfo.remainCoreNum = ((__gm__ uint32_t *)metadataGm)[3];
    constInfo.usedCoreNum = ((__gm__ uint32_t *)metadataGm)[4];
    usedCoreNum = constInfo.usedCoreNum;
    int64_t totalCoreNum = static_cast<int64_t>(GetBlockNum());
    if (cBlockIdx >= constInfo.usedCoreNum) {
        processBS1ByCore = 0;
    } else if (cBlockIdx < totalCoreNum - constInfo.remainCoreNum) {
        processBS1ByCore = constInfo.formerCoreProcessNNum;
    } else {
        processBS1ByCore = constInfo.remainCoreProcessNNum;
    }

    if constexpr (Layout_Q == SLILayout::BSND) {
        curS1 = constInfo.s1Size;
        curS2 = constInfo.s2Size;
    }
    if constexpr (Deterministic) {
        constInfo.bmm3BaseOffset = constInfo.aicIdx * constInfo.kSize * constInfo.dSizeQueryIndex * MODE_NUM_2;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::InitMMResBuf()
{
    l1BufferManager.Init(pipe, L1_MAX_SIZE);
    sYL1Buf.Init(l1BufferManager, 128 * 256 * sizeof(INPUT_T));
    reluGradResL1Buf.Init(l1BufferManager, 64 * 256 * sizeof(INPUT_T));
    ubBufferManager.Init(pipe, UB_MAX_SIZE);
    bmm1Buffers.Init(ubBufferManager, 32 * 256 * sizeof(T));
    bmm3Buffer.Init(ubBufferManager, 64 * 128 * sizeof(T));
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[0]);
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[1]);
        if constexpr (!Deterministic) {
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[0]);
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[1]);
        } else {
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V7_TO_C3_FLAG[0]);
            CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V7_TO_C3_FLAG[1]);
        }
    } else if ASCEND_IS_AIC {
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_GATHER_TO_MM12_FLAG[0]);
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_GATHER_TO_MM12_FLAG[1]);
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_V6_TO_C3_FLAG);
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_V6_TO_RELU_FLAG);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::FreeBuf()
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[0]);
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[1]);
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V6_TO_RELU_FLAG);
    } else if ASCEND_IS_AIC {
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[0]);
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[1]);
        if constexpr (!Deterministic) {
            CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_C3_TO_V7_FLAG[0]);
            CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_C3_TO_V7_FLAG[1]);
        } else {
            CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_V7_TO_C3_FLAG[0]);
            CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_V7_TO_C3_FLAG[1]);
        }
    }
}
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::InitWorkspace(
    __gm__ uint8_t *workspace)
{
    int64_t reduceSumOffset = 2 * constInfo.kSizeAlign128 * sizeof(float);
    int64_t reluOffset = constInfo.gSizeQueryIndex * constInfo.kSizeAlign128 * sizeof(float);
    int64_t gatherSYOffset = constInfo.kSizeAlign128 * constInfo.dSizeQueryIndex * sizeof(INPUT_T);
    int64_t bmm3Offset = constInfo.kSize * constInfo.dSizeQueryIndex * sizeof(float) * 2;
    int64_t coreTotalOffset = constInfo.aicIdx * (reduceSumOffset + reluOffset + gatherSYOffset);
    int64_t totalOffset = GetBlockNum() * (reduceSumOffset + reluOffset + gatherSYOffset);
    totalOffset = GetBlockNum() * (reduceSumOffset + reluOffset + gatherSYOffset);
    uint64_t offset = 0;

    if constexpr (Layout_Q == SLILayout::TND) {
        constInfo.totalCost = ((__gm__ int32_t *)actualSeqKvlenAddr)[constInfo.bSize];
    } else if constexpr (Layout_Q == SLILayout::BSND) {
        constInfo.totalCost = constInfo.bSize * constInfo.s2Size;
    }
    int64_t scatterAddResOffset = constInfo.totalCost * constInfo.dSizeQueryIndex * sizeof(float);

    reduceSumRes.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += reduceSumOffset;
    reluRes.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += reluOffset;
    gatherSYRes.SetGlobalBuffer((__gm__ INPUT_T *)(workspace + offset + coreTotalOffset));
    offset += gatherSYOffset;
    scatterAddResGm.SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));
    if constexpr (Deterministic) {
        totalOffset += scatterAddResOffset;
        scatterAddResPong.SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));
        totalOffset += scatterAddResOffset;
        bmm3Res.SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));
    }

    if ASCEND_IS_AIV {
        int64_t totalCost = constInfo.totalCost;
        int64_t totalCoreNum = GetBlockNum() * GetTaskRation();
        int64_t coreNum_ = GetBlockNum();
        int64_t taskRation = GetTaskRation();
        int64_t avgCost = CeilDiv(totalCost, totalCoreNum);
        int32_t t2Start = Min(constInfo.aivIdx * avgCost, totalCost);
        int32_t t2End = Min(t2Start + avgCost, totalCost);
        GlobalTensor<T> sccatterAddTensor = scatterAddResGm[t2Start * constInfo.dSizeQueryIndex];
        AscendC::Fill(sccatterAddTensor, constInfo.dSizeQueryIndex * (t2End - t2Start), static_cast<T>(0));
        if constexpr (Deterministic) {
            GlobalTensor<T> sccatterAddTensorPong = scatterAddResPong[t2Start * constInfo.dSizeQueryIndex];
            AscendC::Fill(sccatterAddTensorPong, constInfo.dSizeQueryIndex * (t2End - t2Start), static_cast<T>(0));
        }

        qPreBlockFactor = tilingData->multiCoreParams.qPreBlockFactor;
        qPreBlockTotal = tilingData->multiCoreParams.qPreBlockTotal;
        qPreBlockTail = tilingData->multiCoreParams.qPreBlockTail;
        weightPreBlockFactor = tilingData->multiCoreParams.weightPreBlockFactor;
        weightPreBlockTotal = tilingData->multiCoreParams.weightPreBlockTotal;
        weightPreBlockTail = tilingData->multiCoreParams.weightPreBlockTail;
        softmaxOutPreBlockFactor = tilingData->multiCoreParams.softmaxOutPreBlockFactor;
        softmaxOutPreBlockTotal = tilingData->multiCoreParams.softmaxOutPreBlockTotal;
        softmaxOutPreBlockTail = tilingData->multiCoreParams.softmaxOutPreBlockTail;

        initdqSize = vBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = (uint64_t)vBlockIdx * qPreBlockFactor;
        initdweightSize = vBlockIdx == weightPreBlockTotal - 1 ? weightPreBlockTail : weightPreBlockFactor;
        dweightOffset = (uint64_t)vBlockIdx * weightPreBlockFactor;
        initsoftMaxSize = vBlockIdx == softmaxOutPreBlockTotal - 1 ? softmaxOutPreBlockTail : softmaxOutPreBlockFactor;
        softMaxOffset = (uint64_t)vBlockIdx * softmaxOutPreBlockFactor;
        InitOutput<float>(softmaxOutGm[softMaxOffset], initsoftMaxSize, 0);
        InitOutput<float>(dWeightGm[dweightOffset], initdweightSize, 0);
        InitOutput<OUT_T>(dqGm[dqOffset], initdqSize, 0);
    }
    SyncAll<false>();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int32_t SparseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::GetCmpResidualK(
    int32_t bIdx, int32_t defaultLens)
{
    if constexpr (HasCmpResidualK) {
        return ((__gm__ uint32_t *)cmpResidualKAddr)[bIdx];
    } else {
        return defaultLens;
    }
}
} // namespace Slig

#endif
