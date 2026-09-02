/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dense_lightning_indexer_softmax_lse_service_vector_arch35.h
 * \brief A5架构适配的Vector服务头文件
 */

#ifndef __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_SERVICE_VECTOR_H___ARCH35
#define __DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_SERVICE_VECTOR_H___ARCH35

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../dense_lightning_indexer_softmax_lse_common.h"
#include "dense_lightning_indexer_softmax_lse_vector_arch35.h"

namespace DenseLISoftmaxLseKernel {
using namespace DenseLISoftmaxLseCommon;
using namespace DenseLISoftmaxLseServiceVec;
using namespace AscendC;

template <typename T>
class DenseLISoftmaxLseVector {
public:
    using W_T = typename T::weightType;
    using OUT_T = typename T::outputType;
    static constexpr LAYOUT LAYOUT_T = T::layout;

    using MM1_OUT_T = float;

    static constexpr OUT_T SOFTMAX_MIN_NUM = -2e38;

    __aicore__ inline DenseLISoftmaxLseVector(){};
    __aicore__ inline void ProcessVec(const DenseLISoftmaxLseCommon::RunInfo &info);
    __aicore__ inline void ProcessSoftmax(const DenseLISoftmaxLseCommon::RunInfo &info);
    __aicore__ inline void ProcessReduceMax(uint32_t pingPongFlag);
    __aicore__ inline void ProcessReduceSum(uint32_t pingPongFlag);
    __aicore__ inline void ProcessOutput(uint64_t outOffset, uint64_t coreDealSize);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct DenseLISoftmaxLseCommon::ConstInfo &constInfo,
                                      const DenseLISoftmaxLseTilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<W_T> weightsGm, GlobalTensor<OUT_T> softmaxMaxGm,
                                                GlobalTensor<OUT_T> softmaxSumGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<OUT_T> softmaxMaxGm;
    GlobalTensor<OUT_T> softmaxSumGm;

private:
    constexpr static uint32_t DOUBLE_NUM = 2;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);
    constexpr static uint32_t REDUCE_BASE_BLOCK_SIZE = 64;
    constexpr static uint32_t REST_ZERO_BASE_BLOCK_SIZE = 64;
    constexpr static uint32_t AIV_RATIO = 2;

    TQue<TPosition::VECIN, 1> scaleBuf_;
    TQue<TPosition::VECIN, 1> reduceCacheBuf_;
    TQue<TPosition::VECOUT, 1> reduceOutBuf_;
    TQue<TPosition::VECIN, 1> brcBuf_;

    TQue<TPosition::VECIN, 1> reduceMaxSrc0Queue_;
    TQue<TPosition::VECIN, 1> reduceSumSrc0Queue_;

    TQue<TPosition::VECIN, 1> resetZeroQueue_;

    int32_t blockId_ = -1;
    int32_t groupInner_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    uint32_t reduceMaxBaseSize_ = 0;

    uint32_t maskedActS2Size_;
    uint64_t reduceOutGmOffset_;
    uint64_t reduceSrcGmOffset_;
    uint64_t kSeqSizeAlignS2BaseSize_;

    struct DenseLISoftmaxLseCommon::ConstInfo constInfo_;
    DenseLISoftmaxLseCommon::SplitCoreInfo splitCoreInfo{};

    float maxValue_;
};

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(
        scaleBuf_, DOUBLE_NUM,
        groupInner_ * s2BaseSize_ * sizeof(float) + s2BaseSize_ * sizeof(float) + groupInner_ * 8 * sizeof(float));

    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    pipe->InitBuffer(reduceCacheBuf_, DOUBLE_NUM, reduceCacheSize);

    pipe->InitBuffer(reduceOutBuf_, DOUBLE_NUM, s2BaseSize_ * sizeof(float));

    pipe->InitBuffer(reduceMaxSrc0Queue_, DOUBLE_NUM, 4 * REDUCE_BASE_BLOCK_SIZE * sizeof(float));
    pipe->InitBuffer(reduceSumSrc0Queue_, DOUBLE_NUM, 5 * REDUCE_BASE_BLOCK_SIZE * sizeof(float));

    pipe->InitBuffer(resetZeroQueue_, DOUBLE_NUM, REST_ZERO_BASE_BLOCK_SIZE * sizeof(float));
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::InitParams(
    const struct DenseLISoftmaxLseCommon::ConstInfo &constInfo,
    const DenseLISoftmaxLseTilingData *__restrict tilingData)
{
    constInfo_ = constInfo;
    gSize_ = constInfo.gSize;
    kHeadNum_ = constInfo.kHeadNum;
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;

    groupInner_ = 16;

    blockId_ = GetBlockIdx();

    kSeqSizeAlignS2BaseSize_ = DenseLISoftmaxLseCommon::Align(constInfo.kSeqSize, (uint64_t)s2BaseSize_);
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                        GlobalTensor<float> vec1ResGm,
                                                                        GlobalTensor<W_T> weightsGm,
                                                                        GlobalTensor<OUT_T> softmaxMaxGm,
                                                                        GlobalTensor<OUT_T> softmaxSumGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->weightsGm = weightsGm;
    this->softmaxMaxGm = softmaxMaxGm;
    this->softmaxSumGm = softmaxSumGm;
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::AllocEventID()
{}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::FreeEventID()
{}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::ProcessVec(const DenseLISoftmaxLseCommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    int64_t mmGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_) * s2BaseSize_);
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;
    int64_t reduceSumGmOffset = info.reduceSumGmOutOffset;

    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    int32_t perAivOffset = (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);
    cuS1BeginIdxPerAiv += perAivOffset;
    weightGmOffset += perAivOffset * kHeadNum_ * gSize_;
    mmGmOffset += perAivOffset * gSize_ * info.actualSingleProcessSInnerSizeAlign;
    reduceSumGmOffset += perAivOffset * kSeqSizeAlignS2BaseSize_;

    int32_t outerG = CeilDiv(gSize_, groupInner_);
    int32_t cuRealAcSeq = info.actS2Size;

    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;

            LocalTensor<float> reduceOutUb = reduceOutBuf_.AllocTensor<float>();
            LocalTensor<float> reduceCacheUb = reduceCacheBuf_.AllocTensor<float>();

            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;

                LocalTensor<float> mmInUb = scaleBuf_.AllocTensor<float>();
                LocalTensor<float> weightsInUb = mmInUb[procGnum * s2BaseSize_];
                LocalTensor<W_T> weightsInTUb = weightsInUb.template ReinterpretCast<W_T>();
                if constexpr (!IsSameType<W_T, float>::value) {
                    weightsInTUb = weightsInTUb[groupInner_];
                }

                int64_t curMm1ResGmOffset = mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                            outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign;
                int64_t curWeightGmOffset = weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_;
                DenseLISoftmaxLseServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm, curMm1ResGmOffset,
                                                    curWeightGmOffset, procGnum,
                                                    info.actualSingleProcessSInnerSizeAlign, mmUbStride);
                scaleBuf_.EnQue<float>(mmInUb);
                mmInUb = scaleBuf_.DeQue<float>();
                weightsInUb = mmInUb[procGnum * s2BaseSize_];
                DoScaleVfWrapper<W_T>(reduceCacheUb[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb, weightsInTUb,
                                      procGnum, s2BaseSize_, outerGidx);
                scaleBuf_.FreeTensor(mmInUb);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            DenseLISoftmaxLseServiceVec::DoReduce(reduceCacheUb[REDUCE_BANK_CONFLICT_NUM], reduceOutUb, gRedCnt,
                                                  s2BaseSize_);
            reduceOutBuf_.EnQue<float>(reduceOutUb);
            reduceOutUb = reduceOutBuf_.DeQue<float>();
            int64_t curReduceSumGmOffset = reduceSumGmOffset + innerS1Idx * kSeqSizeAlignS2BaseSize_;
            DenseLISoftmaxLseServiceVec::CopyOut(vec1ResGm[curReduceSumGmOffset], reduceOutUb,
                                                 info.actualSingleProcessSInnerSize);
            reduceCacheBuf_.FreeTensor(reduceCacheUb);
            reduceOutBuf_.FreeTensor(reduceOutUb);
        }
    }
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::ProcessSoftmax(const DenseLISoftmaxLseCommon::RunInfo &info)
{
    uint32_t loop = info.s1BlockInnerLoop;
    uint32_t actDealS1BaseSize =
        info.gS1Idx * s1BaseSize_ + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    uint32_t aiv0DealS1Size = DenseLISoftmaxLseCommon::CeilDiv(actDealS1BaseSize, (uint32_t)AIV_RATIO);
    if (blockId_ % AIV_RATIO == 0 && loop >= aiv0DealS1Size) {
        return;
    }

    if (blockId_ % AIV_RATIO != 0 && loop < aiv0DealS1Size) {
        return;
    }

    uint32_t s1Idx = info.s1Idx;
    uint32_t actS1Size = info.actS1Size;
    uint32_t actS2Size = info.actS2Size;

    maskedActS2Size_ =
        (actS2Size - actS1Size + s1Idx + 1) > actS2Size ? actS2Size : (actS2Size - actS1Size + s1Idx + 1);
    reduceOutGmOffset_ = info.reduceMaxOutGmOffset + s1Idx;
    reduceSrcGmOffset_ = loop * kSeqSizeAlignS2BaseSize_;

    uint32_t pingPongFlag = loop % DOUBLE_NUM;
    ProcessReduceMax(pingPongFlag);
    ProcessReduceSum(pingPongFlag);
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::ProcessReduceMax(uint32_t pingPongFlag)
{
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t blockNum = DenseLISoftmaxLseCommon::CeilDiv(maskedActS2Size_, (uint32_t)REDUCE_BASE_BLOCK_SIZE);
    LocalTensor<float> reduceMaxSrc0 = reduceMaxSrc0Queue_.AllocTensor<float>();
    LocalTensor<float> reduceMaxSrc1 = reduceMaxSrc0[REDUCE_BASE_BLOCK_SIZE];
    LocalTensor<float> reduceMaxDstTensor = reduceMaxSrc0[3 * REDUCE_BASE_BLOCK_SIZE];
    LocalTensor<float> sharedTmpTensor = reduceMaxSrc0[2 * REDUCE_BASE_BLOCK_SIZE];
    for (uint32_t blockIdx = 0; blockIdx <= blockNum - 1; blockIdx++) {
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        uint32_t curBlockSize =
            blockIdx != blockNum - 1 ? REDUCE_BASE_BLOCK_SIZE : maskedActS2Size_ - REDUCE_BASE_BLOCK_SIZE * blockIdx;
        if (blockIdx > 0) {
            uint64_t mask = 64;
            uint8_t repeatTime = 1;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Copy(reduceMaxSrc1, reduceMaxDstTensor, mask, repeatTime, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            Duplicate(reduceMaxSrc1, SOFTMAX_MIN_NUM, REDUCE_BASE_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(curBlockSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{
            true, 0, static_cast<uint8_t>(DenseLISoftmaxLseCommon::Align(curBlockSize, uint32_t(8)) - curBlockSize),
            SOFTMAX_MIN_NUM};
        AscendC::DataCopyPad(reduceMaxSrc0, vec1ResGm[reduceSrcGmOffset_ + REDUCE_BASE_BLOCK_SIZE * blockIdx],
                             dataCopyParams, padParams);
        reduceMaxSrc0Queue_.EnQue<float>(reduceMaxSrc0);
        reduceMaxSrc0 = reduceMaxSrc0Queue_.DeQue<float>();

        if (curBlockSize < REDUCE_BASE_BLOCK_SIZE) {
            Duplicate(reduceMaxSrc0[DenseLISoftmaxLseCommon::Align(curBlockSize, uint32_t(8))], SOFTMAX_MIN_NUM,
                      REDUCE_BASE_BLOCK_SIZE - DenseLISoftmaxLseCommon::Align(curBlockSize, uint32_t(8)));
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Max(reduceMaxDstTensor, reduceMaxSrc0, reduceMaxSrc1, REDUCE_BASE_BLOCK_SIZE);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
    }
    WaitFlag<HardEvent::V_S>(EVENT_ID0);

    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax<float>(reduceMaxDstTensor, reduceMaxDstTensor, sharedTmpTensor, REDUCE_BASE_BLOCK_SIZE, false);
    reduceMaxSrc0Queue_.EnQue<float>(reduceMaxSrc0);
    reduceMaxSrc0 = reduceMaxSrc0Queue_.DeQue<float>();

    SetFlag<HardEvent::V_S>(EVENT_ID2 + pingPongFlag);
    WaitFlag<HardEvent::V_S>(EVENT_ID2 + pingPongFlag);

    DataCopyExtParams copyOutParams{1, 1 * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(softmaxMaxGm[reduceOutGmOffset_], reduceMaxDstTensor, copyOutParams);
    maxValue_ = reduceMaxDstTensor.GetValue(0);
    reduceMaxSrc0Queue_.FreeTensor(reduceMaxSrc0);
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::ProcessReduceSum(uint32_t pingPongFlag)
{
    uint32_t blockNum = DenseLISoftmaxLseCommon::CeilDiv(maskedActS2Size_, (uint32_t)REDUCE_BASE_BLOCK_SIZE);
    LocalTensor<float> reduceSumSrc0 = reduceSumSrc0Queue_.AllocTensor<float>();
    LocalTensor<float> reduceSumSrc1 = reduceSumSrc0[REDUCE_BASE_BLOCK_SIZE];
    LocalTensor<float> reduceSumDstTensor = reduceSumSrc0[3 * REDUCE_BASE_BLOCK_SIZE];
    LocalTensor<float> sharedTmpTensor = reduceSumSrc0[2 * REDUCE_BASE_BLOCK_SIZE];

    LocalTensor<float> subMaxTensor = reduceSumSrc0[4 * REDUCE_BASE_BLOCK_SIZE];
    Duplicate(subMaxTensor, maxValue_, REDUCE_BASE_BLOCK_SIZE);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_S>(EVENT_ID4);
    for (uint32_t blockIdx = 0; blockIdx <= blockNum - 1; blockIdx++) {
        WaitFlag<HardEvent::V_S>(EVENT_ID4);
        uint32_t curBlockSize =
            blockIdx != blockNum - 1 ? REDUCE_BASE_BLOCK_SIZE : maskedActS2Size_ - REDUCE_BASE_BLOCK_SIZE * blockIdx;
        if (blockIdx > 0) {
            uint64_t mask = 64;
            uint8_t repeatTime = 1;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Copy(reduceSumSrc1, reduceSumDstTensor, mask, repeatTime, {1, 1, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            Duplicate(reduceSumSrc1, 0.0f, REDUCE_BASE_BLOCK_SIZE);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(curBlockSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{
            true, 0, static_cast<uint8_t>(DenseLISoftmaxLseCommon::Align(curBlockSize, uint32_t(8)) - curBlockSize),
            SOFTMAX_MIN_NUM};
        AscendC::DataCopyPad(reduceSumSrc0, vec1ResGm[reduceSrcGmOffset_ + REDUCE_BASE_BLOCK_SIZE * blockIdx],
                             dataCopyParams, padParams);
        reduceSumSrc0Queue_.EnQue<float>(reduceSumSrc0);
        reduceSumSrc0 = reduceSumSrc0Queue_.DeQue<float>();

        if (curBlockSize < REDUCE_BASE_BLOCK_SIZE) {
            Duplicate(reduceSumSrc0[DenseLISoftmaxLseCommon::Align(curBlockSize, uint32_t(8))], SOFTMAX_MIN_NUM,
                      REDUCE_BASE_BLOCK_SIZE - DenseLISoftmaxLseCommon::Align(curBlockSize, uint32_t(8)));
            AscendC::PipeBarrier<PIPE_V>();
        }
        DoReduceSumBlockVfWrapper(reduceSumDstTensor, reduceSumSrc0, subMaxTensor, reduceSumSrc1,
                                  REDUCE_BASE_BLOCK_SIZE);
        SetFlag<HardEvent::V_S>(EVENT_ID4);
    }
    WaitFlag<HardEvent::V_S>(EVENT_ID4);

    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceSum<float>(reduceSumDstTensor, reduceSumDstTensor, sharedTmpTensor, REDUCE_BASE_BLOCK_SIZE);
    reduceSumSrc0Queue_.EnQue<float>(reduceSumSrc0);
    reduceSumSrc0 = reduceSumSrc0Queue_.DeQue<float>();

    DataCopyExtParams copyOutParams{1, 1 * sizeof(float), 0, 0, 0};
    AscendC::DataCopyPad(softmaxSumGm[reduceOutGmOffset_], reduceSumDstTensor, copyOutParams);
    reduceSumSrc0Queue_.FreeTensor(reduceSumSrc0);
}

template <typename T>
__aicore__ inline void DenseLISoftmaxLseVector<T>::ProcessOutput(uint64_t outOffset, uint64_t coreDealSize)
{
    uint64_t aivCoreDealSize = coreDealSize / AIV_RATIO == 0 ? coreDealSize : coreDealSize / AIV_RATIO;
    uint64_t aivOutOffset = outOffset;
    if (blockId_ % AIV_RATIO == 1) {
        aivOutOffset += aivCoreDealSize;
        aivCoreDealSize = coreDealSize - aivCoreDealSize;
    }
    if (aivCoreDealSize <= 0) {
        return;
    }
    uint32_t blockNum = DenseLISoftmaxLseCommon::CeilDiv(aivCoreDealSize, (uint64_t)REST_ZERO_BASE_BLOCK_SIZE);
    LocalTensor<float> resetZeroTensor = resetZeroQueue_.AllocTensor<float>();
    Duplicate(resetZeroTensor, 0.0f, REST_ZERO_BASE_BLOCK_SIZE);
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    for (uint32_t blockIdx = 0; blockIdx <= blockNum - 1; blockIdx++) {
        uint32_t curBlockSize = blockIdx != blockNum - 1 ? REST_ZERO_BASE_BLOCK_SIZE :
                                                           aivCoreDealSize - REST_ZERO_BASE_BLOCK_SIZE * blockIdx;
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(curBlockSize * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(softmaxMaxGm[aivOutOffset + REST_ZERO_BASE_BLOCK_SIZE * blockIdx], resetZeroTensor,
                             copyOutParams);
        AscendC::DataCopyPad(softmaxSumGm[aivOutOffset + REST_ZERO_BASE_BLOCK_SIZE * blockIdx], resetZeroTensor,
                             copyOutParams);
    }
    resetZeroQueue_.FreeTensor(resetZeroTensor);
    GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vToMte3);
}
} // namespace DenseLISoftmaxLseKernel
#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_SERVICE_VECTOR_H
