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
 * \file sparse_lightning_indexer_kl_loss_grad_vector_block.h
 * \brief
 */

#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_BLOCK_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_VECTOR_BLOCK_H
#include "sparse_lightning_indexer_kl_loss_grad_regbase_common.h"
#include "vf/vf_process_vec0_kl_loss_grad.h"
#include "vf/vf_process_vec1_kl_loss_grad.h"
#include "vf/vf_process_vec2_kl_loss_grad.h"
#include "vf/vf_process_vec4_kl_loss_grad.h"
#include "vf/vf_process_vec5_kl_loss_grad.h"
#include "vf/vf_process_vec6_kl_loss_grad.h"
#include "vf/vf_cast_dup_kl_loss_grad.h"
#include "vf/vf_reduce_sum_kl_loss_grad.h"

namespace Slig {
TEMPLATES_DEF
class SligBlockVec {
public:
    __aicore__ inline SligBlockVec(){};
    __aicore__ inline void InitParams(const SLIGradConstInfo &vecConstInfo,
                                      const optiling::SparseLightningIndexerGradRegBaseTilingData *tiling);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR attnSoftmaxL1,
                                            GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR softmaxOut,
                                            GlobalTensor<INPUT_T> &gatherSYRes, GlobalTensor<T> &relu,
                                            GlobalTensor<T> &reduceSumRes, GlobalTensor<T> &scatterAddRes,
                                            GlobalTensor<T> &scatterAddResPong, GlobalTensor<T> &bmm3Res,
                                            GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK,
                                            GM_ADDR cmpResidualK);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void ProcessVector0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                          SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo);
    __aicore__ inline void CopyInWeight(SLIGradRunInfo &runInfo);
    __aicore__ inline void ProcessVector1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                          SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo);
    __aicore__ inline void ProcessVector2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                          SLIGradRunInfo &runInfo);
    __aicore__ inline void ProcessVector6(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                          SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo);
    __aicore__ inline void ProcessVector7(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm3ResBuf,
                                          SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo);
    __aicore__ inline void DeterScatterAdd(int32_t vRealKSize, GlobalTensor<T> &srcGm,
                                           Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &tmpBuf,
                                           int64_t coreKOffset, SLIGradRunInfo &runInfo, int32_t idx,
                                           GlobalTensor<T> &scatterAddGm);
    __aicore__ inline void ProcessVector7Deter(SLIGradRunInfo &runInfo,
                                               Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &tmpBuf);
    __aicore__ inline void ReInitBuffers(TPipe *pipe);
    __aicore__ inline void ProcessVector8();

    // gm addr
    GlobalTensor<INPUT_T> keyIndexGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<int32_t> topKGm;
    GlobalTensor<T> attnSoftmaxL1Gm;

    // workspace
    GlobalTensor<INPUT_T> gatherSYResGm;
    GlobalTensor<T> reduceSumResGm;
    GlobalTensor<T> reluGm;
    GlobalTensor<T> scatterAddResGm;
    GlobalTensor<T> scatterAddResGmPong;
    GlobalTensor<T> mm3DeterResGm;

    GlobalTensor<T> softmaxOutGm;
    GlobalTensor<T> dWeightGm;
    GlobalTensor<OUT_T> dKeyIndexGm;

    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GM_ADDR usedSeqQlenAddr;
    GM_ADDR usedSeqKvlenAddr;
    GM_ADDR cmpResidualKAddr;

    const optiling::SparseLightningIndexerGradRegBaseTilingData *tilingData;
    TPipe *pipe;

private:
    static constexpr int64_t UB_ROW_SIZE = 8;
    static constexpr event_t EVENT_ID0 = (event_t)4;
    static constexpr event_t EVENT_ID2 = (event_t)7;

    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo,
                                    GlobalTensor<INPUT_T> srcTensor);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const SLIGradRunInfo &runInfo,
                                          SLIGradKRunInfo &kRunInfo, GlobalTensor<INPUT_T> srcTensor);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const SLIGradRunInfo &runInfo, int64_t s2IdLimit);
    template <event_t IdStart, bool syGmEn>
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2GmStartOffset,
                                             int64_t mergeMte3Idx, const SLIGradRunInfo &runInfo,
                                             SLIGradKRunInfo &kRunInfo);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, const SLIGradRunInfo &runInfo,
                                        SLIGradKRunInfo &kRunInfo);
    template <event_t IdStart, bool syGmEn>
    __aicore__ inline void MergeKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                   const SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo,
                                   GlobalTensor<INPUT_T> srcTensor);
    __aicore__ inline bool GetBS1Index(int64_t usedT1Index, int64_t &bIdx, int64_t &s1Idx, int64_t &bS1Index,
                                       int64_t &accumS1Len, int64_t &accumS2Len, int32_t &actualSeqLensQ,
                                       int32_t &actualSeqLensK, int64_t taskId);
    __aicore__ inline void GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx, SLIGradRunInfo &runInfo,
                                      int64_t accumS1Len, int64_t accumS2Len, int32_t actualSeqLensQ,
                                      int32_t actualSeqLensK);
    __aicore__ inline int32_t GetS2SparseLen(int32_t bIdx, int32_t s1Idx, int32_t actualSeqLensQ,
                                             int32_t actualSeqLensK);
    SLIGradConstInfo constInfo;
    TBuf<> gatherTbuf;
    TQue<QuePosition::VECOUT, 1> reduceSumOutQue;
    TQue<QuePosition::VECOUT, 1> softmaxReduceSumOutQue;
    TQue<QuePosition::VECIN, 1> weightInQue;
    TQue<QuePosition::VECOUT, 1> gatherKeyQue[2];
    TQue<QuePosition::VECOUT, 1> gatherKeyIndexQue[2];
    TQue<QuePosition::VECIN, 1> maxInQue;
    TQue<QuePosition::VECIN, 1> sumInQue;
    TQue<QuePosition::VECIN, 1> lseInQue;
    TQue<QuePosition::VECOUT, 1> mulsResQue;
    TQue<QuePosition::VECIN, 1> mulsInQue;
    TQue<QuePosition::VECOUT, 1> lossSumQue;
    TQue<QuePosition::VECIN, 1> reluQue;
    TQue<QuePosition::VECOUT, 1> reluGradQue;
    TQue<QuePosition::VECOUT, 1> dwQue;
    TQue<QuePosition::VECOUT, 1> dwOutQue;
    TQue<QuePosition::VECOUT, 1> gatherOutQue;
    TQue<QuePosition::VECIN, 1> scatterAddInQue[2];
    TQue<QuePosition::VECIN, 1> scatterAddInQue2[2];
    TQue<QuePosition::VECOUT, 1> dKeyOutQue[2];

    LocalTensor<INPUT_T> gatherKeyUb;
    LocalTensor<INPUT_T> gatherKeyIndexUb;

    LocalTensor<INPUT_T> kvMergUb_;
    LocalTensor<INPUT_T> mm1AL1Tensor;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::InitParams(
    const SLIGradConstInfo &vecConstInfo, const optiling::SparseLightningIndexerGradRegBaseTilingData *tiling)
{
    this->constInfo = vecConstInfo;
    tilingData = tiling;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    GM_ADDR key, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR attnSoftmaxL1, GM_ADDR dKey, GM_ADDR dWeight,
    GM_ADDR softmaxOut, GlobalTensor<INPUT_T> &gatherSYRes, GlobalTensor<T> &relu, GlobalTensor<T> &reduceSumRes,
    GlobalTensor<T> &scatterAddRes, GlobalTensor<T> &scatterAddResPong, GlobalTensor<T> &bmm3Res, GM_ADDR cuSeqlensQ,
    GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK)
{
    keyIndexGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
    weightGm.SetGlobalBuffer((__gm__ T *)weight);
    topKGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    attnSoftmaxL1Gm.SetGlobalBuffer((__gm__ T *)attnSoftmaxL1);

    dKeyIndexGm.SetGlobalBuffer((__gm__ OUT_T *)dKey);
    dWeightGm.SetGlobalBuffer((__gm__ T *)dWeight);
    softmaxOutGm.SetGlobalBuffer((__gm__ T *)softmaxOut);

    this->gatherSYResGm = gatherSYRes;
    this->reluGm = relu;
    this->reduceSumResGm = reduceSumRes;
    this->scatterAddResGm = scatterAddRes;

    actualSeqQlenAddr = cuSeqlensQ;
    actualSeqKvlenAddr = cuSeqlensK;
    usedSeqQlenAddr = sequsedQ;
    usedSeqKvlenAddr = sequsedK;
    cmpResidualKAddr = cmpResidualK;

    if constexpr (Deterministic) {
        this->mm3DeterResGm = bmm3Res;
        this->scatterAddResGmPong = scatterAddResPong;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(this->gatherTbuf, SLIGradConstInfo::BUFFER_SIZE_BYTE_9K * 2);
    pipe->InitBuffer(this->gatherOutQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_9K * 2);
    pipe->InitBuffer(this->reduceSumOutQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->weightInQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->reluQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->reluGradQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->dwQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->dwOutQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_256);
    pipe->InitBuffer(this->softmaxReduceSumOutQue, 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_64);
    gatherKeyUb = gatherTbuf.Get<INPUT_T>();
    gatherKeyIndexUb = gatherKeyUb;
    auto dwUb = this->dwQue.template AllocTensor<T>();
    Duplicate(dwUb, 0.0f, 64);
    this->dwQue.template FreeTensor(dwUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradRunInfo &runInfo,
    SLIGradKRunInfo &kRunInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[kRunInfo.kTaskIdMod2]);
    this->mm1AL1Tensor = outputBuf.GetTensor<INPUT_T>();
    this->kvMergUb_ = gatherKeyIndexUb;

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);
    MergeKv<EVENT_ID0, true>(outputBuf, runInfo, kRunInfo, keyIndexGm);
    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_GATHER_TO_MM12_FLAG[kRunInfo.kTaskIdMod2]);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0 + 1);
}

TEMPLATES_DEF_NO_DEFAULT
// template <bool gatherRope>
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::CopyInSingleKv(
    int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx, int64_t keyBNBOffset,
    int64_t s2IdLimit, const SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo, GlobalTensor<INPUT_T> srcTensor)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * kRunInfo.dValue * sizeof(INPUT_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<INPUT_T> padParams;
    DataCopyPad(kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue + (mte2Size - mte3Size) * kRunInfo.dValue],
                srcTensor[keyBNBOffset * kRunInfo.dValue], intriParams, padParams);
    mte2Size += validS2Count;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SligBlockVec<TEMPLATE_ARGS>::GetKeyGmOffset(int64_t realS2Idx, const SLIGradRunInfo &runInfo,
                                                                      int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyGmOffset = runInfo.accumS2Idx + realS2Idx;
    return realKeyGmOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                             int64_t realS2Idx1, int64_t realS2Idx2,
                                                             const SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo,
                                                             GlobalTensor<INPUT_T> srcTensor)
{
    int64_t s2IdLimit = runInfo.s2SparseLen;
    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                    constInfo.sparseBlockSize) *
                   kRunInfo.dValue * sizeof(INPUT_T);

    bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);
    bool strideInvalid = ((keySrcStride >= INT32_MAX) || (keySrcStride < 0));
    bool copyOutOfRange =
        (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit || realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);

    if (key1LessThankey2 || strideInvalid || copyOutOfRange) {
        // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, runInfo, kRunInfo,
                       srcTensor);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, runInfo, kRunInfo,
                       srcTensor);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = constInfo.sparseBlockSize * kRunInfo.dValue * sizeof(INPUT_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<INPUT_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue + (mte2Size - mte3Size) * kRunInfo.dValue],
                    srcTensor[startGmOffset * kRunInfo.dValue], intriParams, padParams);
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1));
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <event_t IdStart, bool syGmEn>
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                      int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                      const SLIGradRunInfo &runInfo,
                                                                      SLIGradKRunInfo &kRunInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<HardEvent::MTE2_MTE3>(mergeMte3Idx + IdStart);
    WaitFlag<HardEvent::MTE2_MTE3>(mergeMte3Idx + IdStart);
    uint32_t dstNzC0Stride = VEC_SY_BASESIZE;
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    LocalTensor<INPUT_T> kvMergUbCp_ = this->gatherOutQue.template AllocTensor<INPUT_T>();
    Ub2UbNd2Nz<INPUT_T>(kvMergUbCp_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue],
                        kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue], UB_ROW_SIZE, kRunInfo.dValue);
    this->gatherOutQue.template EnQue(kvMergUbCp_);
    this->gatherOutQue.template DeQue<INPUT_T>();
    uint16_t valid_row = (mte2Size % UB_ROW_SIZE == 0) ? UB_ROW_SIZE : (mte2Size % UB_ROW_SIZE);
    DataCopy(mm1AL1Tensor[(s2GmStartOffset + mte3Size) << 4], kvMergUbCp_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue],
             {static_cast<uint16_t>(kRunInfo.dValue >> 4), valid_row, static_cast<uint16_t>(UB_ROW_SIZE - valid_row),
              static_cast<uint16_t>(dstNzC0Stride - valid_row)});
    this->gatherOutQue.template FreeTensor(kvMergUbCp_);
    if constexpr (syGmEn) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mte2Size - mte3Size;
        dataCopyParams.blockLen = kRunInfo.dValue * sizeof(INPUT_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        int64_t gmStartOffset = kRunInfo.s2Idx * runInfo.s2BaseSize * kRunInfo.dValue;
        DataCopyPad(gatherSYResGm[gmStartOffset + (s2GmStartOffset + mte3Size) * kRunInfo.dValue],
                    kvMergUb_[mergeMte3Idx * UB_ROW_SIZE * kRunInfo.dValue], dataCopyParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                                 const SLIGradRunInfo &runInfo,
                                                                 SLIGradKRunInfo &kRunInfo)
{
    int64_t topkGmIdx = (s2GmOffset + kRunInfo.s2Idx * runInfo.s2BaseSize);
    realS2Idx = topKGm.GetValue(runInfo.topkGmBaseOffset + topkGmIdx);
}

TEMPLATES_DEF_NO_DEFAULT
template <event_t IdStart, bool syGmEn>
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::MergeKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, const SLIGradRunInfo &runInfo,
    SLIGradKRunInfo &kRunInfo, GlobalTensor<INPUT_T> srcTensor)
{
    int64_t s2ProcessSize = kRunInfo.s2RealBaseSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L);
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; s2GmOffsetArray += 2) {
        if (needWaitMte3ToMte2) {
            WaitFlag<HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, runInfo, kRunInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult<IdStart, syGmEn>(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo, kRunInfo);
            SetFlag<HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, runInfo, kRunInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo, kRunInfo, srcTensor);
        if ((mte2Size - mte3Size + 2 > UB_ROW_SIZE) || s2GmOffsetArray + 2 >= s2GmLimit) {
            CopyOutMrgeResult<IdStart, syGmEn>(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo, kRunInfo);
            mte3Size = mte2Size;
            SetFlag<HardEvent::MTE3_MTE2>(mergeMte3Idx + IdStart);
            mergeMte3Idx = 1 - mergeMte3Idx;
            needWaitMte3ToMte2 = true;
        }
    }
    return;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool SligBlockVec<TEMPLATE_ARGS>::GetBS1Index(int64_t usedT1Index, int64_t &bIdx, int64_t &s1Idx,
                                                                int64_t &bS1Index, int64_t &accumS1Len,
                                                                int64_t &accumS2Len, int32_t &actualSeqLensQ,
                                                                int32_t &actualSeqLensK, int64_t taskId)
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
    if (usedT1Index >= constInfo.totalNum) {
        return false;
    }
    if constexpr (!HasSequsedQ) {
        if constexpr (Layout_Q == SLILayout::TND) {
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
        if constexpr (Layout_Q == SLILayout::TND) {
            t1Offset = ((__gm__ int32_t *)actualSeqQlenAddr)[bIndex];
        } else {
            t1Offset = bIndex * constInfo.s1Size;
        }
        bS1Index = t1Offset + s1Index;
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
    bIdx = bIndex;
    s1Idx = s1Index;
    actualSeqLensQ = usedS1;
    actualSeqLensK = usedS2;
    accumS1Len = t1Offset;
    accumS2Len = t2Offset;
    return true;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::GetRunInfo(int64_t taskId, int64_t bIdx, int64_t s1Idx,
                                                               SLIGradRunInfo &runInfo, int64_t accumS1Len,
                                                               int64_t accumS2Len, int32_t actualSeqLensQ,
                                                               int32_t actualSeqLensK)
{
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;

    runInfo.bIdx = bIdx;
    runInfo.s1Idx = s1Idx;
    runInfo.actS1Size = actualSeqLensQ;
    runInfo.actS2Size = actualSeqLensK;
    runInfo.accumS1Idx = accumS1Len + s1Idx;
    runInfo.accumS2Idx = accumS2Len;
    runInfo.s2SparseLen = GetS2SparseLen(runInfo.bIdx, runInfo.s1Idx, runInfo.actS1Size, runInfo.actS2Size);
    runInfo.s2RealSize = Min(constInfo.kSize, runInfo.s2SparseLen);

    runInfo.kRealSize = runInfo.s2RealSize;

    runInfo.topkGmBaseOffset = runInfo.accumS1Idx * constInfo.kSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t SligBlockVec<TEMPLATE_ARGS>::GetS2SparseLen(int32_t bIdx, int32_t s1Idx,
                                                                      int32_t actualSeqLensQ, int32_t actualSeqLensK)
{
    if (constInfo.sparseMode == SLISparseMode::RightDown) {
        if (constInfo.cmpRatio == 1) {
            return Max(actualSeqLensK - actualSeqLensQ + s1Idx + 1, 0);
        }
        int32_t cmpResidualK = 0;
        if constexpr (HasCmpResidualK) {
            cmpResidualK = ((__gm__ uint32_t *)cmpResidualKAddr)[bIdx];
        }
        return Max(
            ((actualSeqLensK * constInfo.cmpRatio + cmpResidualK) - actualSeqLensQ + s1Idx + 1) / constInfo.cmpRatio,
            0);
    } else if (constInfo.sparseMode == SLISparseMode::DefaultMask) {
        return actualSeqLensK;
    } else {
        return 0;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::CopyInWeight(SLIGradRunInfo &runInfo)
{
    LocalTensor<float> weightInUb = weightInQue.template AllocTensor<float>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = runInfo.nIndexSize * sizeof(float);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<float> padParams;
    DataCopyPad(weightInUb, weightGm[runInfo.weightOffset], dataCopyParams, padParams);
    this->weightInQue.template EnQue(weightInUb);
    this->weightInQue.template DeQue<float>();
    this->weightInQue.template FreeTensor(weightInUb);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, SLIGradRunInfo &runInfo,
    SLIGradKRunInfo &kRunInfo)
{
    if (kRunInfo.s2SingleIdx == 0) {
        CrossCoreWaitFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
    }
    LocalTensor<float> weightInUb = weightInQue.template AllocTensor<float>();
    auto reduceSumTensor = this->reduceSumOutQue.template AllocTensor<T>();

    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    ProcessVec1Vf<T, float>(reduceSumTensor[runInfo.s2CurSize], mmRes[kRunInfo.s2SingleCurSize], weightInUb,
                            runInfo.nIndexSize, constInfo.syKBaseSize);
    if (kRunInfo.s2SingleIdx >= kRunInfo.s2SingleLoopTimes - 1) {
        CrossCoreSetFlag<2, PIPE_V>(SYNC_MM2_TO_V1_FLAG[kRunInfo.kTaskIdMod2]);
    }
    if (kRunInfo.isS2end) {
        this->reduceSumOutQue.template EnQue(reduceSumTensor);
        this->reduceSumOutQue.template DeQue<T>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = runInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(reduceSumResGm[constInfo.subBlockIdx * constInfo.kSizeAlign128], reduceSumTensor, dataCopyParams);
        CrossCoreSetFlag<1, PIPE_MTE3>(EVENT_ID2);
    }
    this->weightInQue.template FreeTensor(weightInUb);
    this->reduceSumOutQue.template FreeTensor(reduceSumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, SLIGradRunInfo &runInfo)
{
    auto reduceSumTensor = this->reduceSumOutQue.template AllocTensor<T>();
    LocalTensor<T> reduceOtherSumTensor = bmm1ResBuf.template GetTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = runInfo.s2RealSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    CrossCoreWaitFlag<1, PIPE_MTE2>(EVENT_ID2);
    DataCopyPad(reduceOtherSumTensor, reduceSumResGm[(1 - constInfo.subBlockIdx) * constInfo.kSizeAlign128],
                dataCopyParams, padParams);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    if (runInfo.s2TailSize == VEC_SY_BASESIZE) {
        ProcessVec2Vf<T, true>(reduceSumTensor, reduceSumTensor, reduceOtherSumTensor, runInfo.s2RealSize);
    } else {
        ProcessVec2Vf<T, false>(reduceSumTensor, reduceSumTensor, reduceOtherSumTensor, runInfo.s2RealSize);
    }
    this->reduceSumOutQue.template EnQue(reduceSumTensor);
    this->reduceSumOutQue.template DeQue<T>();
    int64_t softmaxOutOffset = runInfo.t1Index * constInfo.kSize;
    dataCopyParams.dstStride = constInfo.kSize;
    DataCopyPad(softmaxOutGm[softmaxOutOffset], reduceSumTensor, dataCopyParams);
    this->reduceSumOutQue.template FreeTensor(reduceSumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector6(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, SLIGradRunInfo &runInfo,
    SLIGradKRunInfo &kRunInfo)
{
    if (kRunInfo.s2SingleIdx == 0) {
        CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);
    }
    auto reluUb = this->reluQue.template AllocTensor<T>();
    auto reluGradUb = this->reluGradQue.template AllocTensor<INPUT_T>();
    LocalTensor<T> weightInUb = weightInQue.template AllocTensor<T>();
    auto reduceSumTensor = this->reduceSumOutQue.template AllocTensor<T>();
    auto dwTensor = this->dwQue.template AllocTensor<T>();
    auto softmaxReduceSumTensor = this->softmaxReduceSumOutQue.template AllocTensor<T>();
    LocalTensor<T> softmaxTensor = gatherTbuf.Get<T>();
    DataCopyExtParams dataCopyParams;
    DataCopyPadExtParams<T> padParams;
    if (kRunInfo.kTaskId == 0) {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = constInfo.kSize * sizeof(T);
        dataCopyParams.srcStride = (2048 - constInfo.kSize) * sizeof(T);
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxTensor, attnSoftmaxL1Gm[runInfo.t1Index * constInfo.kSize], dataCopyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        ReduceSumVf(softmaxReduceSumTensor, softmaxTensor, constInfo.kSize);
    }
    dataCopyParams.blockCount = runInfo.nIndexSize;
    dataCopyParams.blockLen = VEC_P_BASESIZE * sizeof(T);
    dataCopyParams.srcStride = (constInfo.kSizeAlign128 - VEC_P_BASESIZE) * sizeof(T);
    dataCopyParams.dstStride = 0;
    DataCopyPad(reluUb,
                reluGm[runInfo.s2CurSize + runInfo.firstNIndexSize * constInfo.subBlockIdx * constInfo.kSizeAlign128],
                dataCopyParams, padParams);
    this->reluQue.template EnQue(reluUb);
    this->reluQue.template DeQue<T>();

    if (kRunInfo.isAlign64) {
        ProcessVec6Vf<T, INPUT_T, true>(reluGradUb[kRunInfo.s2SingleCurSize * runInfo.nIndexSize], dwTensor,
                                        softmaxTensor[runInfo.s2CurSize], softmaxReduceSumTensor, weightInUb,
                                        reduceSumTensor[runInfo.s2CurSize], reluUb, kRunInfo.s2RealBaseSize,
                                        runInfo.nIndexSize, VEC_P_BASESIZE);
    } else {
        ProcessVec6Vf<T, INPUT_T, false>(reluGradUb[kRunInfo.s2SingleCurSize * runInfo.nIndexSize], dwTensor,
                                         softmaxTensor[runInfo.s2CurSize], softmaxReduceSumTensor, weightInUb,
                                         reduceSumTensor[runInfo.s2CurSize], reluUb, kRunInfo.s2RealBaseSize,
                                         runInfo.nIndexSize, VEC_P_BASESIZE);
    }
    if (kRunInfo.s2SingleIdx >= kRunInfo.s2SingleLoopTimes - 1) {
        this->reluGradQue.template EnQue(reluGradUb);
        this->reluGradQue.template DeQue<INPUT_T>();
        LocalTensor<INPUT_T> mm3AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        uint32_t gRealSizeAlignTo16 = (constInfo.gSizeQueryIndex + 15) / 16 * 16;
        DataCopy(
            mm3AL1Tensor[constInfo.subBlockIdx * runInfo.firstNIndexSize * BLOCK_SINGLE_LEN], reluGradUb,
            {static_cast<uint16_t>(constInfo.pKBaseSize / BLOCK_SINGLE_LEN), static_cast<uint16_t>(runInfo.nIndexSize),
             0, static_cast<uint16_t>(gRealSizeAlignTo16 - runInfo.nIndexSize)});
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_V6_TO_C3_FLAG);
    }
    if (kRunInfo.isS2end) {
        this->dwQue.template EnQue(dwTensor);
        this->dwQue.template DeQue<T>();
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
    this->reluGradQue.template FreeTensor(reluGradUb);
    this->reluQue.template FreeTensor(reluUb);
    this->weightInQue.template FreeTensor(weightInUb);
    this->dwQue.template FreeTensor(dwTensor);
    this->reduceSumOutQue.template FreeTensor(reduceSumTensor);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector7(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm3ResBuf, SLIGradRunInfo &runInfo,
    SLIGradKRunInfo &kRunInfo)
{
    CrossCoreWaitFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
    int32_t vRealKSize;
    int64_t coreKOffset;
    if (constInfo.subBlockIdx == 0) {
        vRealKSize = AlignTo(kRunInfo.s2RealBaseSize, MODE_NUM_2) / 2;
        coreKOffset = 0;
    } else {
        vRealKSize = kRunInfo.s2RealBaseSize - AlignTo(kRunInfo.s2RealBaseSize, MODE_NUM_2) / 2;
        coreKOffset = AlignTo(kRunInfo.s2RealBaseSize, MODE_NUM_2) / 2;
    }
    if (vRealKSize <= 0) {
        CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
        return;
    }

    LocalTensor<T> scatterAddTmpUb = bmm3ResBuf.template GetTensor<T>();
    int64_t realS2Idx1;
    int64_t realS2Idx2;
    int64_t s2GmOffset;
    SetAtomicAdd<T>();
    constexpr int32_t topKSplitSize = 2;
    int32_t topKTailSize = (vRealKSize % topKSplitSize == 0) ? topKSplitSize : (vRealKSize % topKSplitSize);
    int32_t topKProcessSize = topKSplitSize;
    int32_t topKLoopTimes = CeilDiv(vRealKSize, topKSplitSize);
    for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
        if (topKIdx >= topKLoopTimes - 1) {
            topKProcessSize = topKTailSize;
        }
        s2GmOffset = coreKOffset + topKIdx * topKSplitSize;
        GetRealS2Idx(s2GmOffset, realS2Idx1, runInfo, kRunInfo);
        GetRealS2Idx(s2GmOffset + 1, realS2Idx2, runInfo, kRunInfo);
        if (topKProcessSize % topKSplitSize != 0) {
            realS2Idx2 = -1;
        }

        int64_t s2IdLimit = runInfo.s2SparseLen;
        int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
        int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit);
        if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
            break;
        }

        int64_t keySrcStride = 0;
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                        constInfo.sparseBlockSize) *
                       constInfo.dSizeQueryIndex * sizeof(T);

        bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0);
        bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                               realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);
        bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);

        int64_t ub1Offset = topKIdx * topKSplitSize * constInfo.dSizeQueryIndex;
        int64_t ub2Offset = ub1Offset + constInfo.dSizeQueryIndex;
        if (strideInvalid || copyOutOfRange || key1LessThankey2) {
            // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
            if (keyOffset1 < 0) {
                break;
            }
            LocalTensor<T> srcTmpUb1 = scatterAddTmpUb[ub1Offset].template ReinterpretCast<T>();
            DataCopy(scatterAddResGm[keyOffset1 * constInfo.dSizeQueryIndex], srcTmpUb1, constInfo.dSizeQueryIndex);
            if (keyOffset2 < 0) {
                break;
            }
            LocalTensor<T> srcTmpUb2 = scatterAddTmpUb[ub2Offset].template ReinterpretCast<T>();
            DataCopy(scatterAddResGm[keyOffset2 * constInfo.dSizeQueryIndex], srcTmpUb2, constInfo.dSizeQueryIndex);
        } else {
            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockLen = constInfo.dSizeQueryIndex * sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
            dataCopyParams.dstStride = keySrcStride;
            int64_t keyStartOffset = (keyOffset1 >= 0) ? keyOffset1 : keyOffset2;
            int64_t ubStartOffset = (keyOffset1 >= 0) ? ub1Offset : ub2Offset;
            LocalTensor<T> srcTmpUb = scatterAddTmpUb.template ReinterpretCast<T>();
            DataCopyPad(scatterAddResGm[keyStartOffset * constInfo.dSizeQueryIndex], srcTmpUb[ubStartOffset],
                        dataCopyParams);
        }
    }
    SetAtomicNone();
    CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_C3_TO_V7_FLAG[kRunInfo.kTaskIdMod2]);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector7Deter(
    SLIGradRunInfo &runInfo, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &tmpBuf)
{
    CrossCoreWaitFlag<MODE_NUM_2, PIPE_MTE2>(SYNC_C3_TO_V7_FLAG[runInfo.taskIdMod2]);
    CrossCoreSetFlag<0, PIPE_MTE2>(SYNC_C3_TO_V7_DETER_MTE2_FLAG[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<0, PIPE_MTE2>(SYNC_C3_TO_V7_DETER_MTE2_FLAG[runInfo.taskIdMod2]);

    CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
    int32_t coreNum = GetBlockNum();
    int64_t preBIdx = -1;
    int64_t bIdx;
    int64_t s1Idx;
    int64_t accumS1Len = 0;
    int64_t accumS2Len = 0;
    int32_t actualSeqLensQ = 0;
    int32_t actualSeqLensK = 0;
    GlobalTensor<T> scatterAddGm[2] = {scatterAddResGm, scatterAddResGmPong};
    int64_t usedBS1Index = -1;
    int64_t bS1Index = -1;
    int64_t usedBS1IndexEnd = constInfo.totalNum - 1;

    for (int32_t idx = 0; idx < coreNum; idx++) {
        usedBS1Index = runInfo.taskId * coreNum + idx;
        // 重新获取b和S1的值
        if (!GetBS1Index(usedBS1Index, bIdx, s1Idx, bS1Index, accumS1Len, accumS2Len, actualSeqLensQ, actualSeqLensK,
                         runInfo.taskId)) {
            continue;
        }

        GetRunInfo(runInfo.taskId, bIdx, s1Idx, runInfo, accumS1Len, accumS2Len, actualSeqLensQ, actualSeqLensK);
        int32_t v0RealKSize;
        int32_t v1RealKSize;
        int32_t vRealKSize;
        int32_t perCoreKSize;
        int32_t tailCoreKSize;
        int32_t curCoreKSize;
        int64_t coreKOffset;
        int64_t vCoreKOffset;
        perCoreKSize = CeilDiv(runInfo.kRealSize, coreNum);
        int32_t usedCoreNum = CeilDiv(runInfo.kRealSize, perCoreKSize);
        if (constInfo.aicIdx >= usedCoreNum) {
            if (idx % MODE_NUM_2 == 0) {
                CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
            }
            if (idx % MODE_NUM_2 == 1 || usedBS1Index == usedBS1IndexEnd) {
                CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
            }
            continue;
        }
        tailCoreKSize = runInfo.kRealSize - (usedCoreNum - 1) * perCoreKSize;
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
            if (idx % MODE_NUM_2 == 0) {
                CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
            }
            if (idx % MODE_NUM_2 == 1 || usedBS1Index == usedBS1IndexEnd) {
                CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
            }
            continue;
        }

        int64_t mm3GmBaseOffset = idx * constInfo.kSize * constInfo.dSizeQueryIndex * MODE_NUM_2;
        GlobalTensor<T> srcGm = mm3DeterResGm[mm3GmBaseOffset + (runInfo.taskIdMod2 * constInfo.kSize + vCoreKOffset) *
                                                                    constInfo.dSizeQueryIndex];
        DeterScatterAdd(vRealKSize, srcGm, tmpBuf, vCoreKOffset, runInfo, idx, scatterAddGm[idx % MODE_NUM_2]);
        if (idx % MODE_NUM_2 == 1 || usedBS1Index == usedBS1IndexEnd) {
            CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
        }
    }
    CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::DeterScatterAdd(
    int32_t vRealKSize, GlobalTensor<T> &srcGm, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &tmpBuf,
    int64_t coreKOffset, SLIGradRunInfo &runInfo, int32_t idx, GlobalTensor<T> &scatterAddGm)
{
    constexpr int32_t topKSplitSize = MODE_NUM_2;
    LocalTensor<T> scatterAddTmpUb = tmpBuf.template GetTensor<T>();
    int32_t scatterAddPingpong = 0;
    int32_t kSplitSize = SLIGradConstInfo::BUFFER_SIZE_BYTE_32K / (MODE_NUM_2 * sizeof(T) * constInfo.dSizeQueryIndex);
    int32_t tailSize = vRealKSize % kSplitSize;
    int32_t kTailSize = (!tailSize) ? kSplitSize : tailSize;
    int32_t kProcessSize = kSplitSize;
    int32_t kLoopTimes = CeilDiv(vRealKSize, kSplitSize);
    int64_t realS2Idx1;
    int64_t realS2Idx2;
    int64_t s2GmOffset;
    event_t eventIdPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    event_t eventIdPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    event_t eventIdArr[MODE_NUM_2] = {eventIdPing, eventIdPong};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdPing);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdPong);
    int64_t s2IdxOffset = 0;
    if (idx % MODE_NUM_2 == 0) {
        CrossCoreWaitFlag<0, PIPE_MTE3>(SYNC_C3_TO_V7_DETER_SA_FLAG);
    }
    SetAtomicAdd<T>();
    for (int32_t kLoopIdx = 0; kLoopIdx < kLoopTimes; ++kLoopIdx) {
        if (kLoopIdx == kLoopTimes - 1) {
            kProcessSize = kTailSize;
        }
        int32_t tailSize = kProcessSize % topKSplitSize;
        int32_t topKTailSize = (!tailSize) ? topKSplitSize : tailSize;
        int32_t topKProcessSize = topKSplitSize;
        int32_t topKLoopTimes = CeilDiv(kProcessSize, topKSplitSize);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        LocalTensor<T> scatterUb = scatterAddTmpUb[scatterAddPingpong * kSplitSize * constInfo.dSizeQueryIndex];
        DataCopy(scatterUb, srcGm[kLoopIdx * kSplitSize * constInfo.dSizeQueryIndex],
                 kProcessSize * constInfo.dSizeQueryIndex);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIdArr[scatterAddPingpong]);
        for (int32_t topKIdx = 0; topKIdx < topKLoopTimes; ++topKIdx) {
            if (topKIdx >= topKLoopTimes - 1) {
                topKProcessSize = topKTailSize;
            }
            s2GmOffset = coreKOffset + kLoopIdx * kSplitSize + topKIdx * topKSplitSize;
            realS2Idx1 = topKGm.GetValue(runInfo.topkGmBaseOffset + s2GmOffset);
            realS2Idx2 = topKGm.GetValue(runInfo.topkGmBaseOffset + s2GmOffset + 1);
            if (topKProcessSize % topKSplitSize != 0) {
                realS2Idx2 = -1;
            }

            int64_t s2IdLimit = runInfo.s2SparseLen;
            int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
            int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit);
            bool keyOffset1Pass = (keyOffset1 >= 0);
            bool keyOffset2Pass = (keyOffset2 >= 0);
            if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
                break;
            }

            int64_t keySrcStride = 0;
            keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                            constInfo.sparseBlockSize) *
                           constInfo.dSizeQueryIndex * sizeof(T);

            bool strideInvalid = (keySrcStride >= INT32_MAX) || (keySrcStride < 0);
            bool copyOutOfRange = (realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                                   realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit);
            bool key1LessThankey2 = (realS2Idx1 > realS2Idx2);

            int64_t ub1Offset = topKIdx * topKSplitSize * constInfo.dSizeQueryIndex;
            int64_t ub2Offset = ub1Offset + constInfo.dSizeQueryIndex;

            if (strideInvalid || copyOutOfRange || key1LessThankey2) {
                // stride溢出、stride为负数、s2超长、topK降序等场景，还原成2条搬运指令
                if (keyOffset1 < 0) {
                    break;
                }
                LocalTensor<T> srcTmpUb1 = scatterUb[ub1Offset].template ReinterpretCast<T>();
                DataCopy(scatterAddGm[keyOffset1 * constInfo.dSizeQueryIndex], srcTmpUb1, constInfo.dSizeQueryIndex);
                if (keyOffset2 < 0) {
                    break;
                }
                LocalTensor<T> srcTmpUb2 = scatterUb[ub2Offset].template ReinterpretCast<T>();
                DataCopy(scatterAddGm[keyOffset2 * constInfo.dSizeQueryIndex], srcTmpUb2, constInfo.dSizeQueryIndex);
            } else {
                DataCopyExtParams dataCopyParams;
                dataCopyParams.blockLen = constInfo.dSizeQueryIndex * sizeof(T);
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockCount = keyOffset1Pass + keyOffset2Pass;
                dataCopyParams.dstStride = keySrcStride;
                int64_t keyStartOffset = keyOffset1Pass ? keyOffset1 : keyOffset2;
                int64_t ubStartOffset = keyOffset1Pass ? ub1Offset : ub2Offset;
                LocalTensor<T> srcTmpUb = scatterUb.template ReinterpretCast<T>();
                DataCopyPad(scatterAddGm[keyStartOffset * constInfo.dSizeQueryIndex], srcTmpUb[ubStartOffset],
                            dataCopyParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdArr[scatterAddPingpong]);
        scatterAddPingpong = 1 - scatterAddPingpong;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdPing);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdPong);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(eventIdPing);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(eventIdPong);
    SetAtomicNone();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ReInitBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(this->scatterAddInQue[0], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->scatterAddInQue[1], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->scatterAddInQue2[0], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->scatterAddInQue2[1], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(this->dKeyOutQue[0], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(this->dKeyOutQue[1], 1, SLIGradConstInfo::BUFFER_SIZE_BYTE_16K);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockVec<TEMPLATE_ARGS>::ProcessVector8()
{
    int64_t avgCost = CeilDiv(constInfo.totalCost, GetBlockNum() * GetTaskRation());
    int32_t t2Start = Min(constInfo.aivIdx * avgCost, constInfo.totalCost);
    int32_t t2End = Min(t2Start + avgCost, constInfo.totalCost);

    int32_t t2ProcessSize = UB_ROW_SIZE;
    int32_t t2TailSize = ((t2End - t2Start) % UB_ROW_SIZE == 0) ? UB_ROW_SIZE : ((t2End - t2Start) % UB_ROW_SIZE);
    int32_t pingPongIdx = 0;

    LocalTensor<T> copyInUb;
    LocalTensor<T> copyInUbPong;
    LocalTensor<OUT_T> copyOutUb;
    for (int32_t t2Idx = t2Start; t2Idx < t2End; t2Idx += UB_ROW_SIZE) {
        if (t2Idx + UB_ROW_SIZE >= t2End) {
            t2ProcessSize = t2TailSize;
        }
        if constexpr (Deterministic) {
            copyInUb = this->scatterAddInQue[pingPongIdx].template AllocTensor<T>();
            copyInUbPong = this->scatterAddInQue2[pingPongIdx].template AllocTensor<T>();
            DataCopy(copyInUb, scatterAddResGm[t2Idx * constInfo.dSizeQueryIndex],
                     t2ProcessSize * constInfo.dSizeQueryIndex);
            DataCopy(copyInUbPong, scatterAddResGmPong[t2Idx * constInfo.dSizeQueryIndex],
                     t2ProcessSize * constInfo.dSizeQueryIndex);
            this->scatterAddInQue[pingPongIdx].template EnQue(copyInUb);
            this->scatterAddInQue[pingPongIdx].template DeQue<T>();
            this->scatterAddInQue2[pingPongIdx].template EnQue(copyInUbPong);
            this->scatterAddInQue2[pingPongIdx].template DeQue<T>();
            Add(copyInUb, copyInUb, copyInUbPong, t2ProcessSize * constInfo.dSizeQueryIndex);
            this->scatterAddInQue2[pingPongIdx].template FreeTensor(copyInUbPong);
        } else {
            copyInUb = this->scatterAddInQue[pingPongIdx].template AllocTensor<T>();
            DataCopy(copyInUb, scatterAddResGm[t2Idx * constInfo.dSizeQueryIndex],
                     t2ProcessSize * constInfo.dSizeQueryIndex);
            this->scatterAddInQue[pingPongIdx].template EnQue(copyInUb);
            this->scatterAddInQue[pingPongIdx].template DeQue<T>();
        }
        copyOutUb = this->dKeyOutQue[pingPongIdx].template AllocTensor<OUT_T>();
        Cast(copyOutUb, copyInUb, RoundMode::CAST_ROUND, t2ProcessSize * constInfo.dSizeQueryIndex);
        this->dKeyOutQue[pingPongIdx].template EnQue(copyOutUb);
        this->dKeyOutQue[pingPongIdx].template DeQue<OUT_T>();
        DataCopy(dKeyIndexGm[t2Idx * constInfo.dSizeQueryIndex], copyOutUb, t2ProcessSize * constInfo.dSizeQueryIndex);
        this->scatterAddInQue[pingPongIdx].template FreeTensor(copyInUb);
        this->dKeyOutQue[pingPongIdx].template FreeTensor(copyOutUb);
        pingPongIdx = 1 - pingPongIdx;
    }
}

TEMPLATES_DEF
class SligBlockVecDummy {
public:
    __aicore__ inline void InitGlobalBuffer(GM_ADDR key, GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR attnSoftmaxL1,
                                            GM_ADDR dKey, GM_ADDR dWeight, GM_ADDR softmaxOut,
                                            GlobalTensor<INPUT_T> &gatherSYRes, GlobalTensor<T> &relu,
                                            GlobalTensor<T> &reduceSumRes, GlobalTensor<T> &scatterAddRes,
                                            GlobalTensor<T> &scatterAddResPong, GlobalTensor<T> &bmm3Res,
                                            GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK,
                                            GM_ADDR cmpResidualK) {};
    __aicore__ inline void InitBuffers(TPipe *pipe) {};
    __aicore__ inline void ProcessVector0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                          SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo) {};
    __aicore__ inline void CopyInWeight(SLIGradRunInfo &runInfo) {};
    __aicore__ inline void ProcessVector1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                          SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo) {};
    __aicore__ inline void ProcessVector2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                          SLIGradRunInfo &runInfo) {};
    __aicore__ inline void ProcessVector6(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                          SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo) {};
    __aicore__ inline void ProcessVector7(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm3ResBuf,
                                          SLIGradRunInfo &runInfo, SLIGradKRunInfo &kRunInfo) {};
    __aicore__ inline void DeterScatterAdd(int32_t vRealKSize, GlobalTensor<T> &srcGm,
                                           Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &tmpBuf,
                                           int64_t coreKOffset, SLIGradRunInfo &runInfo, int32_t idx,
                                           GlobalTensor<T> &scatterAddGm) {};
    __aicore__ inline void ProcessVector7Deter(SLIGradRunInfo &runInfo,
                                               Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &tmpBuf) {};
    __aicore__ inline void ReInitBuffers(TPipe *pipe) {};
    __aicore__ inline void ProcessVector8() {};
};
} // namespace Slig
#endif
