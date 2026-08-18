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
 * \file dense_lightning_indexer_kl_loss_grad_vector_block.h
 * \brief
 */

#ifndef DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_BLOCK_H
#define DENSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_BLOCK_H
#include "dense_lightning_indexer_kl_loss_grad_regbase_common.h"
#include "vf/vf_process_vec1_dense_kl_loss_grad.h"
#include "vf/vf_process_vec2_dense_kl_loss_grad.h"
#include "vf/vf_cast_dup_dense_kl_loss_grad.h"
#include "vf/vf_reduce_sum_dense_kl_loss_grad.h"

namespace Dlikg {
TEMPLATES_DEF
class DlikgBlockVec {
public:
    __aicore__ inline DlikgBlockVec(){};
    __aicore__ inline void InitParams(const DLIKGradConstInfo &vecConstInfo,
                                      const optiling::DenseLightningIndexerGradRegBaseTilingData *tiling);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR weight, GM_ADDR softmaxLse, GM_ADDR attnSoftmaxL1,
                                            GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR softmaxOut,
                                            GlobalTensor<T> &reduceSumRes, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK,
                                            GM_ADDR sequsedQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK,
                                            GlobalTensor<T> &deterRes, GlobalTensor<T> &bmm3Res);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void CopyInWeight(DLIKGradRunInfo &runInfo);
    __aicore__ inline void ProcessVector1(Buffer<BufferType::UB, SyncType::NO_SYNC> &bmm1ResBuf,
                                          DLIKGradRunInfo &runInfo);
    __aicore__ inline void ProcessSoftmaxReduceSum(DLIKGradRunInfo &runInfo);
    __aicore__ inline void ProcessVector2(Buffer<BufferType::UB, SyncType::NO_SYNC> &bmm1ResBuf,
                                          Buffer<BufferType::L1, SyncType::NO_SYNC> &outputBuf,
                                          DLIKGradRunInfo &runInfo);
    __aicore__ inline void ProcessVector3Deter(DLIKGradRunInfo &runInfo);
    // gm addr
    GlobalTensor<INPUT_T> keyIndexGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> softmaxLseGm;
    GlobalTensor<int32_t> topKGm;
    GlobalTensor<T> attnSoftmaxL1Gm;

    // workspace
    GlobalTensor<INPUT_T> gatherSYResGm;
    GlobalTensor<T> reduceSumResGm;
    GlobalTensor<T> reluGm;
    GlobalTensor<T> scatterAddResGm;
    GlobalTensor<T> scatterAddResGmPong;
    GlobalTensor<T> deterRes;
    GlobalTensor<T> bmm3Res;

    GlobalTensor<T> softmaxOutGm;
    GlobalTensor<T> dWeightGm;
    GlobalTensor<OUT_T> dKeyIndexGm;

    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GM_ADDR usedSeqQlenAddr;
    GM_ADDR usedSeqKvlenAddr;
    GM_ADDR cmpResidualKAddr;

    const optiling::DenseLightningIndexerGradRegBaseTilingData *tilingData;
    TPipe *pipe;
    MutexID mteBufMutexId;

private:
    static constexpr int64_t UB_ROW_SIZE = 8;
    static constexpr event_t EVENT_ID2 = (event_t)7;

    __aicore__ inline bool GetBS1Index(int64_t usedT1Index, int64_t &bIdx, int64_t &s1Idx, int64_t &bS1Index,
                                       int64_t &accumS1Len, int64_t &accumS2Len, int32_t &actualSeqLensQ,
                                       int32_t &actualSeqLensK, int64_t taskId, int64_t &currentS2Size);

    DLIKGradConstInfo constInfo;
    TBuf<> gatherTbuf;
    TQue<QuePosition::VECIN, 1> attnSoftmaxL1Que;
    TQue<QuePosition::VECIN, 1> dkQue;
    TQue<QuePosition::VECOUT, 1> reduceSumOutQue[3];
    TQue<QuePosition::VECOUT, 1> reduceSumOtherOutQue[3];
    TQue<QuePosition::VECOUT, 1> softmaxReduceSumOutQue[3];
    TQue<QuePosition::VECIN, 1> softmaxLseQue[3];
    TQue<QuePosition::VECIN, 1> weightInQue[3];
    TQue<QuePosition::VECOUT, 1> gatherKeyQue[2];
    TQue<QuePosition::VECOUT, 1> gatherKeyIndexQue[2];
    TQue<QuePosition::VECIN, 1> maxInQue;
    TQue<QuePosition::VECIN, 1> sumInQue;
    TQue<QuePosition::VECIN, 1> lseInQue;
    TQue<QuePosition::VECOUT, 1> mulsResQue;
    TQue<QuePosition::VECIN, 1> mulsInQue;
    TQue<QuePosition::VECOUT, 1> lossSumQue;
    TQue<QuePosition::VECIN, 1> reluQue;
    TQue<QuePosition::VECOUT, 1> reluGradQue[3];
    TQue<QuePosition::VECOUT, 1> dwQue[3];
    TQue<QuePosition::VECOUT, 1> dwOutQue;
    TQue<QuePosition::VECOUT, 1> gatherOutQue;
    TQue<QuePosition::VECIN, 1> scatterAddInQue[2];
    TQue<QuePosition::VECIN, 1> scatterAddInQue2[2];
    TQue<QuePosition::VECOUT, 1> dKeyOutQue[2];
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockVec<TEMPLATE_ARGS>::InitParams(
    const DLIKGradConstInfo &vecConstInfo, const optiling::DenseLightningIndexerGradRegBaseTilingData *tiling)
{
    this->constInfo = vecConstInfo;
    tilingData = tiling;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR key, GM_ADDR weight, GM_ADDR softmaxLse, GM_ADDR attnSoftmaxL1, GM_ADDR dKey, GM_ADDR dWeight,
    GM_ADDR softmaxOut, GlobalTensor<T> &reduceSumRes, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ,
    GM_ADDR sequsedK, GM_ADDR cmpResidualK, GlobalTensor<T> &deterRes, GlobalTensor<T> &bmm3Res)
{
    keyIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    weightGm.SetGlobalBuffer((__gm__ T *)weight);
    softmaxLseGm.SetGlobalBuffer((__gm__ T *)softmaxLse);
    attnSoftmaxL1Gm.SetGlobalBuffer((__gm__ T *)attnSoftmaxL1);

    dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKey);
    dWeightGm.SetGlobalBuffer((__gm__ T *)dWeight);
    softmaxOutGm.SetGlobalBuffer((__gm__ T *)softmaxOut);
    this->reduceSumResGm = reduceSumRes;
    this->deterRes = deterRes;
    this->bmm3Res = bmm3Res;

    actualSeqQlenAddr = cuSeqlensQ;
    actualSeqKvlenAddr = cuSeqlensK;
    usedSeqQlenAddr = sequsedQ;
    usedSeqKvlenAddr = sequsedK;
    cmpResidualKAddr = cmpResidualK;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockVec<TEMPLATE_ARGS>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(this->attnSoftmaxL1Que, 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->dkQue, 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_64K);
    pipe->InitBuffer(this->reduceSumOutQue[0], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_512);
    pipe->InitBuffer(this->reduceSumOutQue[1], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_512);
    pipe->InitBuffer(this->reduceSumOutQue[2], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_512);
    pipe->InitBuffer(this->reduceSumOtherOutQue[0], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_512);
    pipe->InitBuffer(this->reduceSumOtherOutQue[1], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_512);
    pipe->InitBuffer(this->reduceSumOtherOutQue[2], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_512);
    pipe->InitBuffer(this->weightInQue[0], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->weightInQue[1], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->weightInQue[2], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->reluGradQue[0], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->reluGradQue[1], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->reluGradQue[2], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->dwQue[0], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->dwQue[1], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->dwQue[2], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->dwOutQue, 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->softmaxReduceSumOutQue[0], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_32);
    pipe->InitBuffer(this->softmaxReduceSumOutQue[1], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_32);
    pipe->InitBuffer(this->softmaxReduceSumOutQue[2], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_32);
    pipe->InitBuffer(this->softmaxLseQue[0], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_32);
    pipe->InitBuffer(this->softmaxLseQue[1], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_32);
    pipe->InitBuffer(this->softmaxLseQue[2], 1, DLIKGradConstInfo::BUFFER_SIZE_BYTE_32);
    auto dwUb0 = this->dwQue[0].template AllocTensor<T>();
    auto dwUb1 = this->dwQue[1].template AllocTensor<T>();
    auto dwUb2 = this->dwQue[2].template AllocTensor<T>();
    auto softmaxReduceSumUb0 = this->softmaxReduceSumOutQue[0].template AllocTensor<T>();
    auto softmaxReduceSumUb1 = this->softmaxReduceSumOutQue[1].template AllocTensor<T>();
    auto softmaxReduceSumUb2 = this->softmaxReduceSumOutQue[2].template AllocTensor<T>();

    Duplicate(dwUb0, 0.0f, 64);
    Duplicate(dwUb1, 0.0f, 64);
    Duplicate(dwUb2, 0.0f, 64);
    Duplicate(softmaxReduceSumUb0, 0.0f, 8);
    Duplicate(softmaxReduceSumUb1, 0.0f, 8);
    Duplicate(softmaxReduceSumUb2, 0.0f, 8);

    this->dwQue[0].template FreeTensor(dwUb0);
    this->dwQue[1].template FreeTensor(dwUb1);
    this->dwQue[2].template FreeTensor(dwUb2);
    this->softmaxReduceSumOutQue[0].template FreeTensor(softmaxReduceSumUb0);
    this->softmaxReduceSumOutQue[1].template FreeTensor(softmaxReduceSumUb1);
    this->softmaxReduceSumOutQue[2].template FreeTensor(softmaxReduceSumUb2);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool DlikgBlockVec<TEMPLATE_ARGS>::GetBS1Index(int64_t usedT1Index, int64_t &bIdx, int64_t &s1Idx,
                                                                 int64_t &bS1Index, int64_t &accumS1Len,
                                                                 int64_t &accumS2Len, int32_t &actualSeqLensQ,
                                                                 int32_t &actualSeqLensK, int64_t taskId,
                                                                 int64_t &currentS2Size)
{
    int64_t t1Offset = 0;
    int64_t t2Offset = 0;
    int64_t bIndex = HasSequsedQ ? -1 : 0;
    int64_t s1Index = 0;
    int64_t nextUsedT1Offset = 0;
    int64_t usedT1Offset = 0;
    int64_t curT1 = 0;
    int64_t curT2 = 0;
    int64_t curS1 = constInfo.s1Size;
    int64_t curS2 = constInfo.s2Size;
    int64_t usedS1 = 0;
    int64_t usedS2 = 0;
    int64_t currentCmpResidualK = 0;
    if (usedT1Index >= constInfo.totalNum) {
        return false;
    }
    if constexpr (!HasSequsedQ) {
        if constexpr (Layout_Q == DLILayout::TND) {
            curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            while (usedT1Index >= curT1) {
                curT1 = ((__gm__ int32_t *)actualSeqQlenAddr)[++bIndex];
            }
            bIndex = bIndex - 1;
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
            s1Index = usedT1Index - t1Offset;
        } else {
            bIndex = usedT1Index / constInfo.s1Size;
            s1Index = usedT1Index % constInfo.s1Size;
            t1Offset = usedT1Index - s1Index;
        }
        bS1Index = usedT1Index;
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
        bS1Index = t1Offset + s1Index;
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
        t2Offset = bIndex * constInfo.s2Size;
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

    if constexpr (HasCmpResidualK) {
        currentCmpResidualK = ((__gm__ uint32_t *)cmpResidualKAddr)[bIndex];
    } else {
        currentCmpResidualK = 0;
    }
    if constexpr (SparseMode == DLISparseMode::RightDown) {
        currentS2Size =
            Max(((usedS2 * constInfo.cmpRatio + currentCmpResidualK) - usedS1 + s1Index + 1) / constInfo.cmpRatio, 0);
    } else {
        currentS2Size = usedS2;
    }
    bIdx = bIndex;
    s1Idx = s1Index;
    actualSeqLensQ = usedS1;
    actualSeqLensK = usedS2;
    accumS1Len = t1Offset;
    accumS2Len = t2Offset;
    return true;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockVec<TEMPLATE_ARGS>::CopyInWeight(DLIKGradRunInfo &runInfo)
{
    LocalTensor<float> weightInUb = weightInQue[runInfo.sTaskIdMod3].template AllocTensor<float>();
    LocalTensor<float> lseInUb = softmaxLseQue[runInfo.sTaskIdMod3].template AllocTensor<float>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = runInfo.nIndexSize * sizeof(float);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<float> padParams;
    DataCopyPad(weightInUb, weightGm[runInfo.weightOffset], dataCopyParams, padParams);
    dataCopyParams.blockLen = sizeof(float);
    DataCopyPad(lseInUb, softmaxLseGm[runInfo.t1Index], dataCopyParams, padParams);

    this->weightInQue[runInfo.sTaskIdMod3].template FreeTensor(weightInUb);
    this->softmaxLseQue[runInfo.sTaskIdMod3].template FreeTensor(lseInUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockVec<TEMPLATE_ARGS>::ProcessVector1(
    Buffer<BufferType::UB, SyncType::NO_SYNC> &bmm1ResBuf, DLIKGradRunInfo &runInfo)
{
    CrossCoreWaitFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[runInfo.taskIdMod3]);
    LocalTensor<float> weightInUb = weightInQue[runInfo.sTaskIdMod3].template AllocTensor<float>();
    auto reduceSumTensor = this->reduceSumOutQue[runInfo.taskIdMod3].template AllocTensor<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    ProcessVec1Vf<T, float>(reduceSumTensor, mmRes, weightInUb, runInfo.nIndexSize, constInfo.syKBaseSize);
    this->reduceSumOutQue[runInfo.taskIdMod3].template EnQue(reduceSumTensor);
    this->reduceSumOutQue[runInfo.taskIdMod3].template DeQue<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = runInfo.kProcessSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    uint64_t reduceSumOffset = (constInfo.subBlockIdx * 3 + runInfo.taskIdMod3) * VEC_SY_BASESIZE;
    DataCopyPad(reduceSumResGm[reduceSumOffset], reduceSumTensor, dataCopyParams);
    CrossCoreSetFlag<1, PIPE_MTE3>(EVENT_ID2);
    this->weightInQue[runInfo.sTaskIdMod3].template FreeTensor(weightInUb);
    this->reduceSumOutQue[runInfo.taskIdMod3].template FreeTensor(reduceSumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockVec<TEMPLATE_ARGS>::ProcessSoftmaxReduceSum(DLIKGradRunInfo &runInfo)
{
    auto softmaxTensor = this->attnSoftmaxL1Que.template AllocTensor<T>();
    auto softmaxReduceSumTensor = softmaxReduceSumOutQue[runInfo.sTaskIdMod3].template AllocTensor<T>();
    Duplicate(softmaxReduceSumTensor, 0.0f, 8);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    for (int64_t s2Offset = 0; s2Offset < runInfo.s2RealSize; s2Offset += ATTN_SOFTMAX_REDUCE_BASESIZE) {
        int64_t tmpProcessSize = (runInfo.s2RealSize - s2Offset > ATTN_SOFTMAX_REDUCE_BASESIZE) ?
                                     ATTN_SOFTMAX_REDUCE_BASESIZE :
                                     (runInfo.s2RealSize - s2Offset);
        DataCopyExtParams dataCopyParams;
        DataCopyPadExtParams<T> padParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = tmpProcessSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        DataCopyPad(softmaxTensor, attnSoftmaxL1Gm[runInfo.t1Index * constInfo.maxSeqlenK + s2Offset], dataCopyParams,
                    padParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        ReduceSumVf(softmaxReduceSumTensor, softmaxTensor, tmpProcessSize);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    this->attnSoftmaxL1Que.template FreeTensor(softmaxTensor);
    this->softmaxReduceSumOutQue[runInfo.sTaskIdMod3].template FreeTensor(softmaxReduceSumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockVec<TEMPLATE_ARGS>::ProcessVector2(
    Buffer<BufferType::UB, SyncType::NO_SYNC> &bmm1ResBuf, Buffer<BufferType::L1, SyncType::NO_SYNC> &outputBuf,
    DLIKGradRunInfo &runInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V2_TO_C3_FLAG[runInfo.taskIdMod3]);
    auto reduceSumTensor = this->reduceSumOutQue[runInfo.taskIdMod3].template AllocTensor<T>();
    LocalTensor<T> reduceOtherSumTensor = this->reduceSumOtherOutQue[runInfo.taskIdMod3].template AllocTensor<T>();
    LocalTensor<T> reluUb = bmm1ResBuf.template GetTensor<T>();
    auto reluGradUb = reluGradQue[runInfo.taskIdMod3].template AllocTensor<INPUT_T>();
    LocalTensor<T> weightInUb = weightInQue[runInfo.sTaskIdMod3].template AllocTensor<T>();
    LocalTensor<T> lseInUb = softmaxLseQue[runInfo.sTaskIdMod3].template AllocTensor<T>();
    auto dwTensor = this->dwQue[runInfo.sTaskIdMod3].template AllocTensor<T>();
    auto softmaxReduceSumTensor = softmaxReduceSumOutQue[runInfo.sTaskIdMod3].template AllocTensor<T>();
    LocalTensor<T> attnSoftmaxL1Tensor = this->attnSoftmaxL1Que.template AllocTensor<T>();
    LocalTensor<T> softmaxTensor = attnSoftmaxL1Tensor[runInfo.taskIdMod3 * VEC_SY_BASESIZE];

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = VEC_SY_BASESIZE * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    CrossCoreWaitFlag<1, PIPE_MTE2>(EVENT_ID2);
    DataCopyPad(reduceOtherSumTensor,
                reduceSumResGm[((1 - constInfo.subBlockIdx) * 3 + runInfo.taskIdMod3) * VEC_SY_BASESIZE],
                dataCopyParams, padParams);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = runInfo.kProcessSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(softmaxTensor, attnSoftmaxL1Gm[runInfo.t1Index * constInfo.maxSeqlenK + runInfo.s2Offset],
                dataCopyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (runInfo.isAlign64) {
        ProcessVec2Vf<T, INPUT_T, true>(reluGradUb, dwTensor, softmaxTensor, softmaxReduceSumTensor, weightInUb,
                                        reduceSumTensor, reduceOtherSumTensor, reluUb, lseInUb, runInfo.kProcessSize,
                                        runInfo.nIndexSize, VEC_P_BASESIZE);
    } else {
        ProcessVec2Vf<T, INPUT_T, false>(reluGradUb, dwTensor, softmaxTensor, softmaxReduceSumTensor, weightInUb,
                                         reduceSumTensor, reduceOtherSumTensor, reluUb, lseInUb, runInfo.kProcessSize,
                                         runInfo.nIndexSize, VEC_P_BASESIZE);
    }

    this->reduceSumOtherOutQue[runInfo.taskIdMod3].template EnQue(reduceOtherSumTensor);
    this->reduceSumOtherOutQue[runInfo.taskIdMod3].template DeQue<T>();
    uint64_t softmaxOutOffset = runInfo.t1Index * constInfo.maxSeqlenK + runInfo.s2Offset;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = runInfo.kProcessSize * sizeof(T);
    dataCopyParams.srcStride = 1;
    dataCopyParams.dstStride = 1;
    DataCopyPad(softmaxOutGm[softmaxOutOffset], reduceOtherSumTensor, dataCopyParams);

    CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[runInfo.taskIdMod3]);
    this->reluGradQue[runInfo.taskIdMod3].template EnQue(reluGradUb);
    this->reluGradQue[runInfo.taskIdMod3].template DeQue<INPUT_T>();
    LocalTensor<INPUT_T> mm3AL1Tensor = outputBuf.GetTensor<INPUT_T>();
    uint32_t gRealSizeAlignTo16 = (constInfo.gSizeQueryIndex + 15) / 16 * 16;
    DataCopy(mm3AL1Tensor[constInfo.subBlockIdx * runInfo.firstNIndexSize * BLOCK_SINGLE_LEN], reluGradUb,
             {static_cast<uint16_t>(constInfo.pKBaseSize / BLOCK_SINGLE_LEN), static_cast<uint16_t>(runInfo.nIndexSize),
              0, static_cast<uint16_t>(gRealSizeAlignTo16 - runInfo.nIndexSize)});
    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V2_TO_C3_FLAG[runInfo.taskIdMod3]);
    this->dwQue[runInfo.sTaskIdMod3].template EnQue(dwTensor);
    this->dwQue[runInfo.sTaskIdMod3].template DeQue<T>();
    if (runInfo.isLastBlock) {
        LocalTensor<T> dwOutTensor = dwOutQue.template AllocTensor<T>();
        CastDupVf<T, T>(dwOutTensor, dwTensor, runInfo.nIndexSize);
        this->dwOutQue.template EnQue(dwOutTensor);
        this->dwOutQue.template DeQue<T>();
        DataCopyExtParams dataCopyParams1;
        dataCopyParams1.blockCount = 1;
        dataCopyParams1.blockLen = runInfo.nIndexSize * sizeof(T);
        dataCopyParams1.srcStride = 0;
        dataCopyParams1.dstStride = 0;
        DataCopyPad(dWeightGm[runInfo.weightOffset], dwOutTensor, dataCopyParams1);
        this->dwOutQue.template FreeTensor(dwOutTensor);
    }

    this->reduceSumOutQue[runInfo.taskIdMod3].template FreeTensor(reduceSumTensor);
    this->reduceSumOtherOutQue[runInfo.taskIdMod3].template FreeTensor(reduceOtherSumTensor);
    this->reluGradQue[runInfo.taskIdMod3].template FreeTensor(reluGradUb);
    this->weightInQue[runInfo.sTaskIdMod3].template FreeTensor(weightInUb);
    this->softmaxLseQue[runInfo.sTaskIdMod3].template FreeTensor(lseInUb);
    this->dwQue[runInfo.sTaskIdMod3].template FreeTensor(dwTensor);
    this->softmaxReduceSumOutQue[runInfo.sTaskIdMod3].template FreeTensor(softmaxReduceSumTensor);
    this->attnSoftmaxL1Que.template FreeTensor(attnSoftmaxL1Tensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void DlikgBlockVec<TEMPLATE_ARGS>::ProcessVector3Deter(DLIKGradRunInfo &runInfo)
{
    CrossCoreSetFlag<0, PIPE_MTE2>(SYNC_C3_TO_V3_DETER_MTE2_FLAG);
    CrossCoreWaitFlag<0, PIPE_MTE2>(SYNC_C3_TO_V3_DETER_MTE2_FLAG);
    CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V3_DETER_SA_FLAG);
    LocalTensor<T> dkTensor = this->dkQue.template AllocTensor<T>();
    int32_t coreNum = GetBlockNum();
    int64_t preBIdx = -1;
    int64_t bIdx;
    int64_t s1Idx;
    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    int32_t actualSeqLensQ = 0;
    int32_t actualSeqLensK = 0;
    int64_t deterOffset = runInfo.deterTaskIdMod3 * PROCESS_KV_SIZE * constInfo.dSizeQueryIndex * coreNum;
    GlobalTensor<T> mm3DeterResGm = deterRes[deterOffset];
    int64_t usedBS1Index = -1;
    int64_t bS1Index = -1;
    int64_t usedBS1IndexEnd = constInfo.totalNum - 1;
    int64_t currentS2Size = 0;
    event_t mte3WaitMte2EventId = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
    event_t mte2WaitMte3EventId = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
    SetAtomicAdd<T>();
    for (int32_t idx = 0; idx < coreNum; idx++) {
        usedBS1Index = runInfo.sTaskId * coreNum + idx;
        // 重新获取b和S1的值
        if (!GetBS1Index(usedBS1Index, bIdx, s1Idx, bS1Index, accumS1Len, accumS2Len, actualSeqLensQ, actualSeqLensK,
                         runInfo.taskId, currentS2Size)) {
            continue;
        }
        int64_t processSize;
        int32_t v0RealKSize;
        int32_t v1RealKSize;
        int32_t vRealKSize;
        int32_t perCoreKSize;
        int32_t tailCoreKSize;
        int32_t curCoreKSize;
        int64_t coreKOffset;
        int64_t vCoreKOffset;
        processSize = currentS2Size - runInfo.s2BlockOffset > PROCESS_KV_SIZE ? PROCESS_KV_SIZE :
                                                                                currentS2Size - runInfo.s2BlockOffset;
        processSize = Max(processSize, 0);
        perCoreKSize = CeilDiv(processSize, coreNum);
        int32_t usedCoreNum = CeilDiv(processSize, perCoreKSize);
        if (constInfo.aicIdx >= usedCoreNum) {
            CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V3_DETER_SA_FLAG);
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V3_DETER_SA_FLAG);
            continue;
        }
        tailCoreKSize = processSize - (usedCoreNum - 1) * perCoreKSize;
        curCoreKSize = constInfo.aicIdx == usedCoreNum - 1 ? tailCoreKSize : perCoreKSize;
        v0RealKSize = CeilDiv(curCoreKSize, MODE_NUM_2);
        v0RealKSize = Min(AlignTo(v0RealKSize, static_cast<int32_t>(MODE_NUM_2)), curCoreKSize);
        v1RealKSize = curCoreKSize - v0RealKSize;
        coreKOffset = constInfo.aicIdx * perCoreKSize;
        if (constInfo.subBlockIdx == 0) {
            vRealKSize = v0RealKSize;
            vCoreKOffset = coreKOffset;
        } else {
            vRealKSize = v1RealKSize;
            vCoreKOffset = coreKOffset + v0RealKSize;
        }
        if (vRealKSize <= 0) {
            CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V3_DETER_SA_FLAG);
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V3_DETER_SA_FLAG);
            continue;
        }
        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V3_DETER_SA_FLAG);
        int64_t mm3GmBaseOffset =
            idx * PROCESS_KV_SIZE * constInfo.dSizeQueryIndex + vCoreKOffset * constInfo.dSizeQueryIndex;
        int64_t bmm3Offset = (accumS2Len + runInfo.s2BlockOffset + vCoreKOffset) * constInfo.dSizeQueryIndex;
        GlobalTensor<T> srcGm = mm3DeterResGm[mm3GmBaseOffset];
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        DataCopy(dkTensor, srcGm, vRealKSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
        DataCopy(bmm3Res[bmm3Offset], dkTensor, vRealKSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
        CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V3_DETER_SA_FLAG);
    }
    SetAtomicNone();
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(mte3WaitMte2EventId);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3EventId);
    this->dkQue.template FreeTensor(dkTensor);
    CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V3_DETER_SA_FLAG);
}

TEMPLATES_DEF
class DlikgBlockVecDummy {
public:
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR weight, GM_ADDR softmaxLse, GM_ADDR attnSoftmaxL1,
                                            GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR softmaxOut,
                                            GlobalTensor<T> &reduceSumRes, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK,
                                            GM_ADDR sequsedQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK,
                                            GlobalTensor<T> &deterRes, GlobalTensor<T> &bmm3Res) {};
    __aicore__ inline void InitBuffers(TPipe *pipe) {};
    __aicore__ inline void CopyInWeight(DLIKGradRunInfo &runInfo) {};
    __aicore__ inline void ProcessVector1(Buffer<BufferType::UB, SyncType::NO_SYNC> &bmm1ResBuf,
                                          DLIKGradRunInfo &runInfo) {};
    __aicore__ inline void ProcessSoftmaxReduceSum(DLIKGradRunInfo &runInfo);
    __aicore__ inline void ProcessVector2(Buffer<BufferType::UB, SyncType::NO_SYNC> &bmm1ResBuf,
                                          Buffer<BufferType::L1, SyncType::NO_SYNC> &outputBuf,
                                          DLIKGradRunInfo &runInfo) {};
    __aicore__ inline void ProcessVector3Deter(DLIKGradRunInfo &runInfo) {};
};
} // namespace Dlikg
#endif
