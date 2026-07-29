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
 * \file dense_lightning_indexer_kl_loss_grad_kernel_base.h
 * \brief
 */

#ifndef DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_KERNEL_BASE_H
#define DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_KERNEL_BASE_H

#include "dense_lightning_indexer_kl_loss_grad_tiling_regbase.h"
#include "dense_lightning_indexer_kl_loss_grad_regbase_common.h"
#include "dense_lightning_indexer_kl_loss_grad_cube_block.h"
#include "dense_lightning_indexer_kl_loss_grad_vector_block.h"

namespace Dlikg {
template <typename CubeBlockType, typename VecBlockType>
class DenseLightningIndexerKLLossGradKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline DenseLightningIndexerKLLossGradKernelBase(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR weight, GM_ADDR attnSoftmaxL1, GM_ADDR softmaxLse,
                                GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK,
                                GM_ADDR cmpResidualK, GM_ADDR metadata, GM_ADDR dQuery, GM_ADDR dKey, GM_ADDR dWeight,
                                GM_ADDR softmaxOut, GM_ADDR workspace,
                                const optiling::DenseLightningIndexerGradRegBaseTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessDeter();
    __aicore__ inline void ProcessNormal();
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void FreeBuf();
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline int32_t GetS2SparseLen(int32_t s1Idx, int32_t actualSeqLensQ, int32_t actualSeqLensK,
                                             int32_t cmpResidualK, int64_t cmpRatio, DLISparseMode sparseMode);
    __aicore__ inline void SetRunInfo(DLIKGradRunInfo &runInfo, int64_t taskId, int64_t sTaskId, int64_t deterTaskId);
    __aicore__ inline void SetDeterRunInfo(DLIKGradRunInfo &runInfo, int64_t taskId, int64_t sTaskId,
                                           int64_t deterTaskId, int64_t s2BlockOffset);
    __aicore__ inline void SetConstInfo();
    __aicore__ inline int32_t GetCmpResidualK(int32_t bIdx, int32_t defaultLens);
    __aicore__ inline void GetTndSeqLen(const int64_t t1Idx, int64_t &bIdx);

    TPipe *pipe = nullptr;
    const optiling::DenseLightningIndexerGradRegBaseTilingData *__restrict tilingData = nullptr;

    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GM_ADDR usedSeqQlenAddr;
    GM_ADDR usedSeqKvlenAddr;
    GM_ADDR cmpResidualKAddr;
    GM_ADDR metadataGm;

    // workspace
    GlobalTensor<T> reduceSumRes;
    GlobalTensor<T> softmaxOutGm;
    GlobalTensor<T> dWeightGm;
    GlobalTensor<OUT_T> dqGm;
    GlobalTensor<OUT_T> dkGm;
    GlobalTensor<T> bmm3Res;
    GlobalTensor<T> deterRes;
    // CV间共享buffer
    TBuf<> mm12ResBuf[3];
    TBuf<> mm3ResBuf;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;

    BuffersPolicy3buff<BufferType::UB, SyncType::NO_SYNC> bmm1Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::NO_SYNC> reluGradResL1Buf;

    DLIKGradConstInfo constInfo;
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;

    int64_t coreNum;

    int64_t curS1 = 0;
    int64_t curS2 = 0;
    int64_t usedS1 = 0;
    int64_t usedS2 = 0;
    int64_t currentS2Size = 0;
    int64_t currentCmpResidualK = 0;
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
    int64_t maxSeqKV = 0;
    int64_t s2BlockOffset;

    uint32_t qPreBlockFactor;
    uint32_t qPreBlockTotal;
    uint32_t qPreBlockTail;
    uint32_t kPreBlockFactor;
    uint32_t kPreBlockTotal;
    uint32_t kPreBlockTail;
    uint32_t weightPreBlockFactor;
    uint32_t weightPreBlockTotal;
    uint32_t weightPreBlockTail;
    uint32_t softmaxOutPreBlockFactor;
    uint32_t softmaxOutPreBlockTotal;
    uint32_t softmaxOutPreBlockTail;
    uint32_t initdqSize;
    uint32_t initdkSize;
    uint64_t dqOffset;
    uint64_t dkOffset;
    uint32_t initdweightSize;
    uint64_t dweightOffset;
    uint32_t initsoftMaxSize;
    uint64_t softMaxOffset;
    uint64_t s2Offset;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR weight, GM_ADDR attnSoftmaxL1, GM_ADDR softmaxLse, GM_ADDR cuSeqlensQ,
    GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK, GM_ADDR metadata, GM_ADDR dQuery,
    GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR softmaxOut, GM_ADDR workspace,
    const optiling::DenseLightningIndexerGradRegBaseTilingData *__restrict tiling, TPipe *tPipe)
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
    dkGm.SetGlobalBuffer((__gm__ OUT_T *)dKey);

    InitMMResBuf();
    InitWorkspace(workspace);

    if ASCEND_IS_AIV {
        vecBlock.InitParams(constInfo, tilingData);
        vecBlock.InitGlobalBuffer(key, weight, softmaxLse, attnSoftmaxL1, dKey, dWeight, softmaxOut, reduceSumRes,
                                  cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK, deterRes, bmm3Res);
        vecBlock.InitBuffers(pipe);
    } else if ASCEND_IS_AIC {
        cubeBlock.SetCubeBlockParams(tPipe, &l1BufferManager);
        cubeBlock.InitCubeBuffers();
        cubeBlock.InitGlobalBuffer(query, dQuery, key, dKey, bmm3Res, deterRes);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::GetTndSeqLen(
    const int64_t t1Idx, int64_t &bIdx)
{
    int64_t t1Offset = 0;
    int64_t t2Offset = 0;

    if constexpr (!HasSequsedQ) {
        if constexpr (Layout_Q == DLILayout::TND) {
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
        if constexpr (Layout_Q == DLILayout::TND) {
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
        } else {
            t1Offset = bIndex * constInfo.s1Size;
        }
        t1Offset += s1Index;
        t1Index = t1Offset;
    }

    if constexpr (Layout_Q == DLILayout::TND) {
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

    currentCmpResidualK = GetCmpResidualK(bIndex, 0);
    if constexpr (SparseMode == DLISparseMode::RightDown) {
        currentS2Size =
            Max(((usedS2 * constInfo.cmpRatio + currentCmpResidualK) - usedS1 + s1Index + 1) / constInfo.cmpRatio, 0);
    } else {
        currentS2Size = usedS2;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::Process()
{
    if constexpr (Deterministic) {
        ProcessDeter();
    } else {
        ProcessNormal();
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::ProcessNormal()
{
    int64_t sTaskId = 0;
    int64_t s2TaskId = 0;
    DLIKGradRunInfo runInfo[MODE_NUM_3];
    for (int32_t i = 0; i < processBS1ByCore; i++) {
        this->usedT1Index = this->cBlockIdx + this->usedCoreNum * i;
        GetTndSeqLen(usedT1Index, bIndex);
        if (currentS2Size <= 0) {
            continue;
        }
        for (this->s2Offset = 0; this->s2Offset < currentS2Size; this->s2Offset += VEC_SY_BASESIZE) {
            DLIKGradRunInfo &runInfoNeg0 = runInfo[s2TaskId % MODE_NUM_3];
            DLIKGradRunInfo &runInfoNeg2 = runInfo[(s2TaskId + 1) % MODE_NUM_3];
            DLIKGradRunInfo &runInfoNeg1 = runInfo[(s2TaskId + 2) % MODE_NUM_3];
            SetRunInfo(runInfoNeg0, s2TaskId, sTaskId, 0);
            if (s2Offset == 0) {
                if ASCEND_IS_AIV {
                    vecBlock.CopyInWeight(runInfoNeg0);
                    vecBlock.ProcessSoftmaxReduceSum(runInfoNeg0);
                }
            }
            if (s2TaskId >= 0) {
                if ASCEND_IS_AIC {
                    cubeBlock.ComputeMmSy(this->bmm1Buffers.Get(), runInfoNeg0, constInfo);
                }
                runInfoNeg0.taskStep = TASK_C1;
            }
            if (s2TaskId >= 1) {
                if ASCEND_IS_AIV {
                    Buffer<BufferType::UB, SyncType::NO_SYNC> bmm1Res = this->bmm1Buffers.Get();
                    vecBlock.ProcessVector1(bmm1Res, runInfoNeg1);
                    vecBlock.ProcessVector2(bmm1Res, this->reluGradResL1Buf.Get(), runInfoNeg1);
                }
                runInfoNeg1.taskStep = TASK_V1;
            }
            if (s2TaskId >= 2) {
                if ASCEND_IS_AIC {
                    cubeBlock.ComputeMmDk(this->reluGradResL1Buf.Get(), runInfoNeg2, constInfo);
                    cubeBlock.ComputeMmDq(reluGradResL1Buf.GetPre(), runInfoNeg2, constInfo, runInfoNeg2.isLastBlock);
                }
            }
            s2TaskId++;
        }
        sTaskId++;
    }
    if (runInfo[(s2TaskId + 2) % MODE_NUM_3].taskStep == TASK_C1) {
        if ASCEND_IS_AIV {
            Buffer<BufferType::UB, SyncType::NO_SYNC> bmm1Res = this->bmm1Buffers.Get();
            vecBlock.ProcessVector1(bmm1Res, runInfo[(s2TaskId + 2) % MODE_NUM_3]);
            vecBlock.ProcessVector2(bmm1Res, this->reluGradResL1Buf.Get(), runInfo[(s2TaskId + 2) % MODE_NUM_3]);
        }
        runInfo[(s2TaskId + 2) % MODE_NUM_3].taskStep = TASK_V1;
    }
    if (runInfo[(s2TaskId + 1) % MODE_NUM_3].taskStep == TASK_V1) {
        if ASCEND_IS_AIC {
            cubeBlock.ComputeMmDk(this->reluGradResL1Buf.Get(), runInfo[(s2TaskId + 1) % MODE_NUM_3], constInfo);
            cubeBlock.ComputeMmDq(reluGradResL1Buf.GetPre(), runInfo[(s2TaskId + 1) % MODE_NUM_3], constInfo,
                                  runInfo[(s2TaskId + 1) % MODE_NUM_3].isLastBlock);
        }
    }
    s2TaskId++;
    if (runInfo[(s2TaskId + 1) % MODE_NUM_3].taskStep == TASK_V1) {
        if ASCEND_IS_AIC {
            cubeBlock.ComputeMmDk(this->reluGradResL1Buf.Get(), runInfo[(s2TaskId + 1) % MODE_NUM_3], constInfo);
            cubeBlock.ComputeMmDq(reluGradResL1Buf.GetPre(), runInfo[(s2TaskId + 1) % MODE_NUM_3], constInfo,
                                  runInfo[(s2TaskId + 1) % MODE_NUM_3].isLastBlock);
        }
    }
    SyncAll<false>();
    FreeBuf();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::ProcessDeter()
{
    int64_t sTaskId = 0;
    int64_t s2TaskId = 0;
    int64_t deterTaskId = 0;
    DLIKGradRunInfo runInfo[MODE_NUM_3];
    DLIKGradRunInfo deterRunInfo[MODE_NUM_3];
    DLIKGradRunInfo deterInfo;
    DLIKGradRunInfo tailRunInfo;
    for (int32_t i = 0; i < processBS1ByCore; i++) {
        this->usedT1Index = this->cBlockIdx + this->usedCoreNum * i;
        GetTndSeqLen(usedT1Index, bIndex);
        for (this->s2BlockOffset = 0; this->s2BlockOffset < constInfo.maxSeqKV;
             this->s2BlockOffset += PROCESS_KV_SIZE) {
            int64_t blockSize = this->s2BlockOffset;
            int64_t processSize = this->currentS2Size - this->s2BlockOffset > PROCESS_KV_SIZE ?
                                      PROCESS_KV_SIZE :
                                      this->currentS2Size - this->s2BlockOffset;
            processSize = Max(processSize, 0);
            for (this->s2Offset = blockSize; this->s2Offset < blockSize + processSize;
                 this->s2Offset += VEC_SY_BASESIZE) {
                DLIKGradRunInfo &runInfoNeg0 = runInfo[s2TaskId % MODE_NUM_3];
                DLIKGradRunInfo &runInfoNeg2 = runInfo[(s2TaskId + 1) % MODE_NUM_3];
                DLIKGradRunInfo &runInfoNeg1 = runInfo[(s2TaskId + 2) % MODE_NUM_3];
                SetRunInfo(runInfoNeg0, s2TaskId, sTaskId, deterTaskId);
                if (s2Offset == 0) {
                    if ASCEND_IS_AIV {
                        vecBlock.CopyInWeight(runInfoNeg0);
                        vecBlock.ProcessSoftmaxReduceSum(runInfoNeg0);
                    }
                }
                if (s2TaskId >= 0) {
                    if ASCEND_IS_AIC {
                        cubeBlock.ComputeMmSy(this->bmm1Buffers.Get(), runInfoNeg0, constInfo);
                    }
                    runInfoNeg0.taskStep = TASK_C1;
                }
                if (s2TaskId >= 1 && runInfoNeg1.taskStep == TASK_C1) {
                    if ASCEND_IS_AIV {
                        Buffer<BufferType::UB, SyncType::NO_SYNC> bmm1Res = this->bmm1Buffers.Get();
                        vecBlock.ProcessVector1(bmm1Res, runInfoNeg1);
                        vecBlock.ProcessVector2(bmm1Res, this->reluGradResL1Buf.Get(), runInfoNeg1);
                    }
                    runInfoNeg1.taskStep = TASK_V1;
                }
                if (s2TaskId >= 2 && runInfoNeg2.taskStep == TASK_V1) {
                    if ASCEND_IS_AIC {
                        cubeBlock.ComputeMmDk(this->reluGradResL1Buf.Get(), runInfoNeg2, constInfo);
                        cubeBlock.ComputeMmDq(reluGradResL1Buf.GetPre(), runInfoNeg2, constInfo,
                                              runInfoNeg2.isLastBlock);
                    }
                    runInfoNeg2.taskStep = TASK_C2;
                }
                s2TaskId++;
            }
            if (processSize == 0) {
                bool processFlag = false;
                if (runInfo[(s2TaskId + 2) % MODE_NUM_3].taskStep == TASK_C1) {
                    if ASCEND_IS_AIV {
                        Buffer<BufferType::UB, SyncType::NO_SYNC> bmm1Res = this->bmm1Buffers.Get();
                        vecBlock.ProcessVector1(bmm1Res, runInfo[(s2TaskId + 2) % MODE_NUM_3]);
                        vecBlock.ProcessVector2(bmm1Res, this->reluGradResL1Buf.Get(),
                                                runInfo[(s2TaskId + 2) % MODE_NUM_3]);
                    }
                    processFlag = true;
                    runInfo[(s2TaskId + 2) % MODE_NUM_3].taskStep = TASK_V1;
                }
                if (runInfo[(s2TaskId + 1) % MODE_NUM_3].taskStep == TASK_V1) {
                    if ASCEND_IS_AIC {
                        cubeBlock.ComputeMmDk(this->reluGradResL1Buf.Get(), runInfo[(s2TaskId + 1) % MODE_NUM_3],
                                              constInfo);
                        cubeBlock.ComputeMmDq(reluGradResL1Buf.GetPre(), runInfo[(s2TaskId + 1) % MODE_NUM_3],
                                              constInfo, runInfo[(s2TaskId + 1) % MODE_NUM_3].isLastBlock);
                    }
                    processFlag = true;
                    runInfo[(s2TaskId + 1) % MODE_NUM_3].taskStep = TASK_C2;
                }
                if (processFlag) {
                    s2TaskId++;
                }
            }
            if (deterTaskId < 2) {
                if ASCEND_IS_AIC {
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C3_TO_V3_FLAG);
                } else {
                    CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C3_TO_V3_FLAG);
                }
            }
            SetDeterRunInfo(deterRunInfo[deterTaskId % 3], s2TaskId, sTaskId, deterTaskId, this->s2BlockOffset);
            if (deterTaskId >= 2) {
                deterInfo = deterRunInfo[(deterTaskId + 1) % 3];
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<2, PIPE_FIX>(DETER_TO_FIX_SYNC_FLAG);
                    CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                    CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                    CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C3_TO_V3_FLAG);
                } else {
                    CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C3_TO_V3_FLAG);
                    vecBlock.ProcessVector3Deter(deterInfo);
                    CrossCoreSetFlag<2, PIPE_MTE3>(DETER_TO_FIX_SYNC_FLAG);
                }
            }
            deterTaskId++;
        }
        sTaskId++;
    }

    if (runInfo[(s2TaskId + 2) % MODE_NUM_3].taskStep == TASK_C1) {
        if ASCEND_IS_AIV {
            Buffer<BufferType::UB, SyncType::NO_SYNC> bmm1Res = this->bmm1Buffers.Get();
            vecBlock.ProcessVector1(bmm1Res, runInfo[(s2TaskId + 2) % MODE_NUM_3]);
            vecBlock.ProcessVector2(bmm1Res, this->reluGradResL1Buf.Get(), runInfo[(s2TaskId + 2) % MODE_NUM_3]);
        }
        runInfo[(s2TaskId + 2) % MODE_NUM_3].taskStep = TASK_V1;
        if (runInfo[(s2TaskId + 1) % MODE_NUM_3].taskStep == TASK_V1) {
            if ASCEND_IS_AIC {
                cubeBlock.ComputeMmDk(this->reluGradResL1Buf.Get(), runInfo[(s2TaskId + 1) % MODE_NUM_3], constInfo);
                cubeBlock.ComputeMmDq(reluGradResL1Buf.GetPre(), runInfo[(s2TaskId + 1) % MODE_NUM_3], constInfo,
                                      runInfo[(s2TaskId + 1) % MODE_NUM_3].isLastBlock);
            }
            runInfo[(s2TaskId + 1) % MODE_NUM_3].taskStep = TASK_C2;
        }
        s2TaskId++;
    }

    if (runInfo[(s2TaskId + 1) % MODE_NUM_3].taskStep == TASK_V1) {
        if ASCEND_IS_AIC {
            cubeBlock.ComputeMmDk(this->reluGradResL1Buf.Get(), runInfo[(s2TaskId + 1) % MODE_NUM_3], constInfo);
            cubeBlock.ComputeMmDq(reluGradResL1Buf.GetPre(), runInfo[(s2TaskId + 1) % MODE_NUM_3], constInfo,
                                  runInfo[(s2TaskId + 1) % MODE_NUM_3].isLastBlock);
        }
        runInfo[(s2TaskId + 1) % MODE_NUM_3].taskStep = TASK_C2;
    }

    if (deterTaskId >= 2) {
        deterInfo = deterRunInfo[(deterTaskId + 1) % 3];
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<2, PIPE_FIX>(DETER_TO_FIX_SYNC_FLAG);
            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C3_TO_V3_FLAG);
        } else {
            CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C3_TO_V3_FLAG);
            vecBlock.ProcessVector3Deter(deterInfo);
            CrossCoreSetFlag<2, PIPE_MTE3>(DETER_TO_FIX_SYNC_FLAG);
        }
    }
    deterTaskId++;
    if (deterTaskId >= 2) {
        deterInfo = deterRunInfo[(deterTaskId + 1) % 3];
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<2, PIPE_FIX>(DETER_TO_FIX_SYNC_FLAG);
            CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
            CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C3_TO_V3_FLAG);
        } else {
            CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C3_TO_V3_FLAG);
            vecBlock.ProcessVector3Deter(deterInfo);
            CrossCoreSetFlag<2, PIPE_MTE3>(DETER_TO_FIX_SYNC_FLAG);
        }
    }
    deterTaskId++;
    if (this->processBS1ByCore < constInfo.formerCoreProcessNNum) {
        deterTaskId = deterTaskId - 2;
        for (this->s2BlockOffset = 0; this->s2BlockOffset < constInfo.maxSeqKV;
             this->s2BlockOffset += PROCESS_KV_SIZE) {
            SetDeterRunInfo(deterRunInfo[(deterTaskId + 1) % 3], s2TaskId, sTaskId, deterTaskId, this->s2BlockOffset);
            deterInfo = deterRunInfo[(deterTaskId + 1) % 3];
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<2, PIPE_FIX>(DETER_TO_FIX_SYNC_FLAG);
                CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
                CrossCoreSetFlag<2, PIPE_FIX>(SYNC_C3_TO_V3_FLAG);
            } else {
                CrossCoreWaitFlag<2, PIPE_MTE2>(SYNC_C3_TO_V3_FLAG);
                vecBlock.ProcessVector3Deter(deterInfo);
                CrossCoreSetFlag<2, PIPE_MTE3>(DETER_TO_FIX_SYNC_FLAG);
            }
            deterTaskId++;
        }
    }
    SyncAll<false>();
    FreeBuf();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int32_t DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::GetS2SparseLen(
    int32_t s1Idx, int32_t actualSeqLensQ, int32_t actualSeqLensK, int32_t cmpResidualK, int64_t cmpRatio,
    DLISparseMode sparseMode)
{
    if (sparseMode == DLISparseMode::RightDown) {
        if (cmpRatio == 1) {
            return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
        }
        return Max(((actualSeqLensK * cmpRatio + cmpResidualK) - actualSeqLensQ + s1Idx + 1) / cmpRatio, 0);
    } else if (sparseMode == DLISparseMode::DefaultMask) {
        return actualSeqLensK;
    } else {
        return 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetRunInfo(
    DLIKGradRunInfo &runInfo, int64_t taskId, int64_t sTaskId, int64_t deterTaskId)
{
    runInfo.sTaskId = sTaskId;
    runInfo.sTaskIdMod3 = sTaskId % 3;
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.bIdx = bIndex;
    runInfo.t1Index = t1Index;
    runInfo.t2Index = t2Index;
    runInfo.s1Idx = s1Index;
    runInfo.actS1Size = usedS1;
    runInfo.actS2Size = usedS2;
    runInfo.accumS1Idx = runInfo.t1Index;
    runInfo.accumS2Idx = runInfo.t2Index;
    runInfo.taskStep = TASK_INIT;
    runInfo.kBaseSize = 2048;
    runInfo.s2SparseLen = currentS2Size;
    runInfo.s2RealSize = currentS2Size;
    runInfo.kRealSize = currentS2Size;
    runInfo.s2Offset = s2Offset;
    runInfo.isLastBlock = s2Offset + VEC_SY_BASESIZE >= currentS2Size;
    runInfo.kProcessSize = runInfo.s2RealSize - runInfo.s2Offset > 128 ? 128 : runInfo.s2RealSize - runInfo.s2Offset;
    runInfo.isAlign64 = runInfo.kProcessSize % 64 == 0;
    runInfo.kRealSizeAlign8 = (runInfo.kRealSize + 7) >> 3 << 3;
    runInfo.s2LoopTimes = CeilDiv(runInfo.s2RealSize, constInfo.syKBaseSize);
    runInfo.s2TailSize = (runInfo.s2RealSize % constInfo.syKBaseSize == 0) ?
                             constInfo.syKBaseSize :
                             (runInfo.s2RealSize % constInfo.syKBaseSize);
    runInfo.s2BaseSize = VEC_SY_BASESIZE;

    if constexpr (Deterministic) {
        runInfo.deterTaskId = deterTaskId;
        runInfo.deterTaskIdMod3 = deterTaskId % 3;
        runInfo.s2BlockOffset = s2BlockOffset;
        runInfo.deterResOffset = runInfo.deterTaskIdMod3 * PROCESS_KV_SIZE * constInfo.dSizeQueryIndex * coreNum +
                                 constInfo.aicIdx * PROCESS_KV_SIZE * constInfo.dSizeQueryIndex +
                                 (s2Offset - s2BlockOffset) * constInfo.dSizeQueryIndex;
    }
    runInfo.keyIndexTensorOffset =
        runInfo.t2Index * constInfo.n2Size * constInfo.dSizeQueryIndex + runInfo.s2Offset * constInfo.dSizeQueryIndex;
    runInfo.queryIndexTensorOffset = runInfo.t1Index * constInfo.gSizeQueryIndex * constInfo.dSizeQueryIndex;
    runInfo.firstNIndexSize = AlignTo(constInfo.gSizeQueryIndex, ALIGN_NUM_2) / 2;
    if (constInfo.subBlockIdx == 0) {
        runInfo.nIndexSize = runInfo.firstNIndexSize;
    } else {
        runInfo.nIndexSize = constInfo.gSizeQueryIndex - runInfo.firstNIndexSize;
    }
    runInfo.weightOffset =
        runInfo.t1Index * constInfo.gSizeQueryIndex + runInfo.firstNIndexSize * constInfo.subBlockIdx;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetDeterRunInfo(
    DLIKGradRunInfo &runInfo, int64_t taskId, int64_t sTaskId, int64_t deterTaskId, int64_t s2BlockOffset)
{
    runInfo.s2BlockOffset = s2BlockOffset;
    runInfo.sTaskId = sTaskId;
    runInfo.sTaskIdMod3 = sTaskId % 3;
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.deterTaskId = deterTaskId;
    runInfo.deterTaskIdMod3 = deterTaskId % 3;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::SetConstInfo()
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
    constInfo.totalSize = tilingData->multiCoreParams.totalSize;
    auto &baseInfo = tilingData->baseParams;
    constInfo.bSize = baseInfo.bSize;
    constInfo.n2Size = baseInfo.n2Size;
    constInfo.gSizeQueryIndex = baseInfo.gSizeQueryIndex;
    constInfo.s1Size = baseInfo.s1Size;
    constInfo.s2Size = baseInfo.s2Size;
    constInfo.dSizeQueryIndex = baseInfo.dSizeQueryIndex;
    constInfo.sparseMode = static_cast<DLISparseMode>(baseInfo.sparseMode);
    constInfo.cmpRatio = baseInfo.cmpRatio;
    constInfo.maxSeqlenK = baseInfo.maxSeqlenK;
    constInfo.sparseBlockSize = 1;
    constInfo.pKBaseSize = 128;
    constInfo.syKBaseSize = 128;
    constInfo.totalNum = ((__gm__ uint32_t *)metadataGm)[0];
    constInfo.formerCoreProcessNNum = ((__gm__ uint32_t *)metadataGm)[1];
    constInfo.remainCoreProcessNNum = ((__gm__ uint32_t *)metadataGm)[2];
    constInfo.remainCoreNum = ((__gm__ uint32_t *)metadataGm)[3];
    constInfo.usedCoreNum = ((__gm__ uint32_t *)metadataGm)[4];
    constInfo.maxSeqKV = constInfo.maxSeqlenK;
    usedCoreNum = constInfo.usedCoreNum;
    int64_t totalCoreNum = static_cast<int64_t>(GetBlockNum());
    if (cBlockIdx >= constInfo.usedCoreNum) {
        processBS1ByCore = 0;
    } else if (cBlockIdx < totalCoreNum - constInfo.remainCoreNum) {
        processBS1ByCore = constInfo.formerCoreProcessNNum;
    } else {
        processBS1ByCore = constInfo.remainCoreProcessNNum;
    }

    if constexpr (Layout_Q == DLILayout::BSND) {
        curS1 = constInfo.s1Size;
        curS2 = constInfo.s2Size;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::InitMMResBuf()
{
    l1BufferManager.Init(pipe, L1_MAX_SIZE);
    reluGradResL1Buf.Init(l1BufferManager, 128 * 128 * sizeof(INPUT_T));
    ubBufferManager.Init(pipe, UB_MAX_SIZE);
    bmm1Buffers.Init(ubBufferManager, 64 * 128 * sizeof(T));
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[0]);
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[1]);
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[2]);
        if constexpr (Deterministic) {
            CrossCoreSetFlag<2, PIPE_MTE3>(DETER_TO_FIX_SYNC_FLAG);
        }
    } else if ASCEND_IS_AIC {
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_V2_TO_C3_FLAG[0]);
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_V2_TO_C3_FLAG[1]);
        CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_V2_TO_C3_FLAG[2]);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::FreeBuf()
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V2_TO_C3_FLAG[0]);
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V2_TO_C3_FLAG[1]);
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V2_TO_C3_FLAG[2]);
    } else if ASCEND_IS_AIC {
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[0]);
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[1]);
        CrossCoreWaitFlag<2, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[2]);
        if constexpr (Deterministic) {
            CrossCoreWaitFlag<2, PIPE_FIX>(DETER_TO_FIX_SYNC_FLAG);
        }
    }
}
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::InitWorkspace(
    __gm__ uint8_t *workspace)
{
    int64_t reduceSumOffset = 3 * VEC_SY_BASESIZE * sizeof(float) * 2;
    int64_t deterResOffset = 0;
    if constexpr (Deterministic) {
        deterResOffset = 3 * PROCESS_KV_SIZE * constInfo.dSizeQueryIndex * sizeof(float);
    }
    int64_t coreTotalOffset = constInfo.aicIdx * reduceSumOffset;
    int64_t totalOffset = GetBlockNum() * reduceSumOffset;
    uint64_t offset = 0;

    if constexpr (Layout_Q == DLILayout::TND) {
        constInfo.totalCost = ((__gm__ int32_t *)actualSeqKvlenAddr)[constInfo.bSize];
    } else if constexpr (Layout_Q == DLILayout::BSND) {
        constInfo.totalCost = constInfo.bSize * constInfo.s2Size;
    }
    reduceSumRes.SetGlobalBuffer((__gm__ T *)(workspace + offset + coreTotalOffset));
    offset += reduceSumOffset;
    if constexpr (Deterministic) {
        deterRes.SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));
        totalOffset += GetBlockNum() * deterResOffset;
    }
    bmm3Res.SetGlobalBuffer((__gm__ T *)(workspace + totalOffset));

    if ASCEND_IS_AIV {
        qPreBlockFactor = tilingData->multiCoreParams.qPreBlockFactor;
        qPreBlockTotal = tilingData->multiCoreParams.qPreBlockTotal;
        qPreBlockTail = tilingData->multiCoreParams.qPreBlockTail;
        kPreBlockFactor = tilingData->multiCoreParams.kPreBlockFactor;
        kPreBlockTotal = tilingData->multiCoreParams.kPreBlockTotal;
        kPreBlockTail = tilingData->multiCoreParams.kPreBlockTail;
        weightPreBlockFactor = tilingData->multiCoreParams.weightPreBlockFactor;
        weightPreBlockTotal = tilingData->multiCoreParams.weightPreBlockTotal;
        weightPreBlockTail = tilingData->multiCoreParams.weightPreBlockTail;
        softmaxOutPreBlockFactor = tilingData->multiCoreParams.softmaxOutPreBlockFactor;
        softmaxOutPreBlockTotal = tilingData->multiCoreParams.softmaxOutPreBlockTotal;
        softmaxOutPreBlockTail = tilingData->multiCoreParams.softmaxOutPreBlockTail;

        initdqSize = vBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        initdkSize = vBlockIdx == kPreBlockTotal - 1 ? kPreBlockTail : kPreBlockFactor;
        dqOffset = (uint64_t)vBlockIdx * qPreBlockFactor;
        dkOffset = (uint64_t)vBlockIdx * kPreBlockFactor;
        initdweightSize = vBlockIdx == weightPreBlockTotal - 1 ? weightPreBlockTail : weightPreBlockFactor;
        dweightOffset = (uint64_t)vBlockIdx * weightPreBlockFactor;
        initsoftMaxSize = vBlockIdx == softmaxOutPreBlockTotal - 1 ? softmaxOutPreBlockTail : softmaxOutPreBlockFactor;
        softMaxOffset = (uint64_t)vBlockIdx * softmaxOutPreBlockFactor;
        InitOutput<float>(softmaxOutGm[softMaxOffset], initsoftMaxSize, 0);
        InitOutput<float>(dWeightGm[dweightOffset], initdweightSize, 0);
        InitOutput<OUT_T>(dqGm[dqOffset], initdqSize, 0);
        InitOutput<OUT_T>(dkGm[dkOffset], initdkSize, 0);
        InitOutput<T>(bmm3Res[dkOffset], initdkSize, 0);
    }
    SyncAll<false>();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int32_t DenseLightningIndexerKLLossGradKernelBase<CubeBlockType, VecBlockType>::GetCmpResidualK(
    int32_t bIdx, int32_t defaultLens)
{
    if constexpr (HasCmpResidualK) {
        return ((__gm__ uint32_t *)cmpResidualKAddr)[bIdx];
    } else {
        return defaultLens;
    }
}
} // namespace Dlikg

#endif
