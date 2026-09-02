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
 * \file lightning_indexer_service_vector_arch35.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_SERVICE_VECTOR_ARCH35_H
#define LIGHTNING_INDEXER_SERVICE_VECTOR_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../lightning_indexer_common.h"
#include "../arch35/vf/lightning_indexer_vector1.h"
#include "../arch35/vf/lightning_indexer_topk.h"

namespace LIKernel {
using namespace LICommon;
constexpr uint32_t TRUNK_LEN_8K = 8192;
constexpr uint32_t TRUNK_LEN_4K = 4096;
constexpr uint32_t TRUNK_LEN_2K = 2048;
constexpr uint32_t TOPK_LEN_7K = 7168;
constexpr uint32_t TOPK_LEN_5K = 5120;
constexpr uint32_t DUPSIZE = 256; // duplicate所需额外地址空间

template <typename Q_T, typename W_T = void>
struct LightningIndexerTypeTraits {
    using weightsType = Q_T; // 默认：weightsType绑定Q_T
};

template <typename Q_T>
struct LightningIndexerTypeTraits<Q_T, float> {
    using weightsType = float; // W_T=float时，强制weightsType为float
};
template <typename LIT>
class LightningIndexerServiceVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using SCORE_T = uint32_t;
    using W_T =
        typename LightningIndexerTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;

    __aicore__ inline LightningIndexerServiceVector(){};
    __aicore__ inline void ProcessVec1(const LICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const LICommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LICommon::ConstInfo &constInfo,
                                      const LITilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<K_T> valueOutGm, GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<K_T> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<float> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;

    // tmp buff for returnValue K_T
    TBuf<TPosition::VECCALC> valueOutBuf_;
    LocalTensor<K_T> valueOutLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;
    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t trunkLen_ = 0;
    bool returnValueFlag = false;

    struct LICommon::ConstInfo constInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(float));
    resMm1UB_ = resMm1Buf_.Get<float>();

    pipe->InitBuffer(weightBuf_,
                     2 * CeilDiv(s1BaseSize_, 2) * LICommon::Align((uint64_t)gSize_, (uint64_t)16) * sizeof(W_T));
    weightUB_ = weightBuf_.Get<W_T>();
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    vec1OutUB_ = outBuf_.Get<SCORE_T>(); // out

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    // returnvalue
    if (topkCount_ <= 2048) {
        pipe->InitBuffer(valueOutBuf_, topkCountAlign256_ * sizeof(K_T) + DUPSIZE);
        valueOutLocal_ = valueOutBuf_.Get<K_T>();
    } else {                                      // sparseCount > 2k时，复用return value相关UB
        valueOutLocal_ = mrgValueBuf_.Get<K_T>(); // returnValue float
    }

    // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    pipe->InitBuffer(indicesOutBuf_, topkCountAlign256_ * sizeof(uint32_t) + DUPSIZE);
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T) + DUPSIZE);
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_, indicesOutLocal_);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitParams(const struct LICommon::ConstInfo &constInfo,
                                                                      const LITilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize; // 4
    s2BaseSize_ = constInfo.s2BaseSize; // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    returnValueFlag = constInfo.returnValueFlag;
    blockId_ = GetBlockIdx();
    trunkLen_ = constInfo.sparseCount > TOPK_LEN_5K ?
                    (constInfo.sparseCount > TOPK_LEN_7K ? TRUNK_LEN_2K : TRUNK_LEN_4K) :
                    TRUNK_LEN_8K;
    topkCount_ = constInfo.sparseCount;
    topkOp_.Init(topkCount_, trunkLen_);
    topkCountAlign256_ = LICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm,
                                                                              GlobalTensor<int32_t> indiceOutGm,
                                                                              GlobalTensor<K_T> valueOutGm,
                                                                              GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    uint64_t dealSize = constInfo_.sparseCount;
    GlobalTensor<int32_t> indexOutput = indiceOutGm[invalidS1Offset];
    AscendC::InitGlobalMemory(indexOutput, dealSize, constInfo_.INVALID_IDX);
    if (returnValueFlag) {
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>(), constInfo_.INVALID_VAL, constInfo_.sparseCount);

        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

        AscendC::DataCopyParams copyOutValueParams;
        copyOutValueParams.blockCount = 1;
        copyOutValueParams.blockLen = constInfo_.sparseCount * sizeof(K_T);
        copyOutValueParams.srcStride = 0;
        copyOutValueParams.dstStride = 0;
        AscendC::DataCopyPad(valueOutGm[invalidS1Offset], valueOutLocal_, copyOutValueParams);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::ProcessVec1(const LICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    // CV同步, V核等C核计算完mm1，mm1Res已搬运到UB
    CrossCoreWaitFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LICommon::ConstInfo::CROSS_CV_EVENT + pingpong);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    if (curAivS1ProcNum == 0) {
        // V核处理完，通知C核可以把mm1Res搬运到UB
        CrossCoreSetFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LICommon::ConstInfo::CROSS_VC_EVENT + pingpong);
        return;
    }
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    // weightsGm --> weightUB_
    uint64_t gSizeAlign16 = LICommon::Align((uint64_t)gSize_, (uint64_t)16);
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    DataCopyPadExtParams<W_T> padWeightsParams{true, 0, 0, 0};
    DataCopyExtParams qwDataCopyExtParams;
    qwDataCopyExtParams.blockCount = curAivS1ProcNum;
    qwDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
    qwDataCopyExtParams.srcStride = 0;
    qwDataCopyExtParams.dstStride = (gSizeAlign16 - gSize_) * sizeof(W_T) / 32;
    DataCopyPad(weightUB_[pingpong * CeilDiv(s1BaseSize_, 2) * gSizeAlign16], weightsGm[weightGmOffset],
                qwDataCopyExtParams, padWeightsParams);

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    for (int64_t s1IdxTmp = 0; s1IdxTmp < curAivS1ProcNum; s1IdxTmp++) {
        vector1::MulWeightAndReduceSum(
            vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ + s1IdxTmp * s2BaseSize_],
            resMm1UB_[pingpong * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ + s1IdxTmp * gSize_ * s2BaseSize_],
            weightUB_[pingpong * CeilDiv(s1BaseSize_, 2) * gSizeAlign16 + s1IdxTmp * gSizeAlign16], gSize_);
    }
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    // outUB_ --->  scoreGm
    uint64_t kSeqSizeAlign = LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_);
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx : CeilDiv(s1BaseSize_, 2) * kSeqSizeAlign + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = (kSeqSizeAlign - s2BaseSize_) * sizeof(SCORE_T);

    DataCopyPad(scoreGm[vec1OutGmOffset], vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_], copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    // V核处理完，通知C核可以把mm1Res搬运到UB
    CrossCoreSetFlag<LICommon::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LICommon::ConstInfo::CROSS_VC_EVENT + pingpong);
}

template <typename LIT>
__aicore__ inline void LightningIndexerServiceVector<LIT>::ProcessTopK(const LICommon::RunInfo &info)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;

        SCORE_T zero = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = (int32_t)i + cuRealAcSeq;
        }
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (returnValueFlag) {
                WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>(), constInfo_.INVALID_VAL, topkCount_);

                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = topkCount_ * sizeof(K_T);
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_], valueOutLocal_,
                                     copyOutValueParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            }
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            bool useSingleLoop =
                (s2LoopNum == 1) || ((topkCount_ > trunkLen_) && (validS2Len <= (uint32_t)topkCountAlign256_));
            if (useSingleLoop) {
                uint32_t validS2LenAlign = LICommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(
                    mrgValueLocal_,
                    scoreGm[vecOffset * LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1, returnValueFlag);
            } else {
                uint32_t actS2LoopNum = 0;
                if (topkCount_ > trunkLen_) {
                    actS2LoopNum = 1 + (validS2Len - topkCountAlign256_ + trunkLen_ - 1) / trunkLen_;
                } else {
                    actS2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
                }
                for (uint32_t loopIdx = 0; loopIdx < actS2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        if (topkCount_ > trunkLen_) {
                            copyInParams.blockLen = topkCountAlign256_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(scoreOutLocal_,
                                                 scoreGm[vecOffset * LICommon::Align((uint64_t)constInfo_.kSeqSize,
                                                                                     (uint64_t)s2BaseSize_)],
                                                 copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero,
                                                    topkCountAlign256_);
                            AscendC::CreateVecIndex(topkSharedTmpLocal_.ReinterpretCast<int32_t>(), (int32_t)zero,
                                                    topkCountAlign256_);
                        } else {
                            copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(mrgValueLocal_,
                                                 scoreGm[vecOffset * LICommon::Align((uint64_t)constInfo_.kSeqSize,
                                                                                     (uint64_t)s2BaseSize_)],
                                                 copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx, actS2LoopNum,
                                    returnValueFlag);
                        }
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = 0;
                    uint32_t offset = 0;
                    if (topkCount_ > trunkLen_) {
                        validTrunkLen = (topkCountAlign256_ + (loopIdx - 1) * trunkLen_ + trunkLen_) > validS2Len ?
                                            (validS2Len - topkCountAlign256_) % trunkLen_ :
                                            trunkLen_;
                        offset = vecOffset * LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 topkCountAlign256_ + (loopIdx - 1) * trunkLen_;
                    } else {
                        validTrunkLen =
                            (loopIdx * trunkLen_ + trunkLen_) > validS2Len ? validS2Len % trunkLen_ : trunkLen_;
                        offset = vecOffset * LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 loopIdx * trunkLen_;
                    }
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    // 如果是tok > trunklen, 第一轮每调用topk，是直接拷贝的，所以不需要刷零
                    bool isZeroPadding = (topkCount_ > trunkLen_) ? (loopIdx > 1) : true;
                    if (topkCountAlign256_ != topkCount_ && isZeroPadding) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero,
                                  topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero,
                                  LICommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                            LICommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx, actS2LoopNum,
                            returnValueFlag);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
            if (returnValueFlag) {
                copyInParams.blockLen = LICommon::Align(validS2Len, (int32_t)32) * sizeof(SCORE_T);
                AscendC::DataCopyPad(
                    scoreOutLocal_,
                    scoreGm[vecOffset * LICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
        }

        if (validS2Len < topkCount_) {
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << (validS2Len % 8);
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
        }

        if (validS2Len / 8 * 8 + 64 < topkCount_) {
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg,
                      topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                             indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);

        // // 是否返回Value值
        if (returnValueFlag) {
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            // uint16_t -> bfloat16
            if (std::is_same_v<K_T, bfloat16_t>) {
                vector1::UIntToFloatReturnValue(valueOutLocal_.template ReinterpretCast<bfloat16_t>(), scoreOutLocal_,
                                                topkCountAlign256_);
            } else {
                vector1::UIntToFloatReturnValue(valueOutLocal_.template ReinterpretCast<half>(), scoreOutLocal_,
                                                topkCountAlign256_);
            }

            if (validS2Len < topkCount_) {
                uint64_t mask[2];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % 16);
                mask[1] = ~0;
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>()[validS2Len / 16 * 16],
                          constInfo_.INVALID_VAL, mask, 1, 1, 0);
            }
            if (validS2Len / 16 * 16 + 128 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>()[validS2Len / 16 * 16 + 128],
                          constInfo_.INVALID_VAL, topkCount_ - (validS2Len / 16 * 16 + 128));
            }
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyParams copyOutValueParams;
            copyOutValueParams.blockCount = 1;
            copyOutValueParams.blockLen = topkCount_ * sizeof(K_T); // bytes
            copyOutValueParams.srcStride = 0;
            copyOutValueParams.dstStride = 0;
            // 搬运到GM
            AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_], valueOutLocal_,
                                 copyOutValueParams);
        }
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
}
} // namespace LIKernel
#endif
