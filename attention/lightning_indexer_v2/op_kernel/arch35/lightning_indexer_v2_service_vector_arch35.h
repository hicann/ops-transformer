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
 * \file lightning_indexer_v2_service_vector_arch35.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_SERVICE_VECTOR_ARCH35_H
#define LIGHTNING_INDEXER_V2_SERVICE_VECTOR_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_arch35.h"
#include "../arch35/vf/lightning_indexer_v2_vector1.h"
#include "../arch35/vf/lightning_indexer_v2_topk.h"

#include "common/lightning_indexer_v2_service_vector_base_arch35.h"

namespace LIV2Kernel {
using namespace LIV2Common;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_8K = 8192;
constexpr uint32_t TRUNK_LEN_4K = 4096;
constexpr uint32_t TRUNK_LEN_2K = 2048;
constexpr uint32_t TRUNK_LEN_1K = 1024;
constexpr uint32_t TOPK_LEN_7K = 7168;
constexpr uint32_t TOPK_LEN_5K = 5120;

template <typename Liv2ServiceTraits>
class LightningIndexerV2ServiceVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_V2_LAYOUT LAYOUT_T = Liv2ServiceTraits::layout;
    static constexpr LI_V2_LAYOUT K_LAYOUT_T = Liv2ServiceTraits::keyLayout;
    static constexpr bool PAGE_ATTENTION = Liv2ServiceTraits::pageAttention;
    static constexpr bool DT_W_FLAG = true;
    using Q_T = typename Liv2ServiceTraits::queryType;
    using K_T = typename Liv2ServiceTraits::keyType;
    using QK_T = typename Liv2ServiceTraits::queryKeyType;
    using SCORE_T = typename Liv2ServiceTraits::scoreType;
    using W_T = float;

    __aicore__ inline LightningIndexerV2ServiceVector(){};
    __aicore__ inline void ProcessVec1(const LIV2Common::RunInfo &info);
    __aicore__ inline void ProcessTopK(const LIV2Common::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LIV2Common::ConstInfo &constInfo,
                                      const LIV2Common::LdSplitCoreInfo &ldInfo,
                                      const LIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitLDBuffers(TPipe *pipe, const LIV2Common::LdSplitCoreInfo &ldInfo);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> liv2ScoreGm,
                                                  GlobalTensor<SCORE_T> liv2LdScoreGm,
                                                  GlobalTensor<int32_t> liv2LdIndexGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> liv2WeightsGm, GlobalTensor<int32_t> liv2IndiceOutGm,
                                              GlobalTensor<float> liv2ValueOutGm,
                                              GlobalTensor<int32_t> liv2BlockTableGm,
                                              GlobalTensor<int32_t> liv2OutputIdxOffsetGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void DoTndPadding(const LIV2Common::RunInfo &runInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<SCORE_T> liv2ScoreGm;
    GlobalTensor<SCORE_T> liv2LdScoreGm;
    GlobalTensor<int32_t> liv2LdIndexGm;
    GlobalTensor<W_T> liv2WeightsGm;
    GlobalTensor<int32_t> liv2IndiceOutGm;
    GlobalTensor<float> liv2ValueOutGm;
    GlobalTensor<int32_t> liv2BlockTableGm;
    GlobalTensor<int32_t> liv2OutputIdxOffsetGm;
    // =================================常量区=================================
    static constexpr uint32_t LIV2_VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t LIV2_VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t LIV2_VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t LIV2_VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;
    static constexpr uint32_t MTE3_V_EVENT = EVENT_ID6;
    static constexpr uint32_t M_BASIC_BLOCK = 128;

private:
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> liv2ResMm1Buf_;
    LocalTensor<float> liv2ResMm1Ub_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> liv2WeightBuf_;
    LocalTensor<W_T> liv2WeightUb_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> liv2OutBuf_;
    LocalTensor<SCORE_T> liv2Vec1OutUb_;

    // tmp buff for returnValue K_T
    TBuf<TPosition::VECCALC> liv2ValueOutBuf_;
    LocalTensor<float> liv2ValueOutLocal_;
    // tmp buff for topk
    TBuf<TPosition::VECCALC> liv2MrgValueBuf_;
    LocalTensor<SCORE_T> liv2MrgValueLocal_;

    TBuf<TPosition::VECCALC> liv2IndicesOutBuf_;
    LocalTensor<uint32_t> liv2IndicesOutLocal_;

    TBuf<TPosition::VECCALC> liv2ScoreOutBuf_;
    LocalTensor<SCORE_T> liv2ScoreOutLocal_;

    TBuf<TPosition::VECCALC> liv2TopkSharedTmpBuf_;
    LocalTensor<uint32_t> liv2TopkSharedTmpLocal_;
    LocalTensor<uint32_t> liv2LdIndexLocal_;

    int32_t liv2BlockId_ = -1;
    // para for vector
    int32_t liv2GroupInner_ = 0;
    int32_t liv2GlobalTopkNum_ = 0;
    int64_t liv2BlockS2StartIdx_ = 0;
    int32_t liv2GSize_ = 0;
    int32_t liv2KSeqSize_ = 0;
    int32_t liv2KHeadNum_ = 0;
    int32_t liv2QHeadNum_ = 0;
    int32_t liv2S1BaseSize_ = 0;
    int32_t liv2S2BaseSize_ = 0;
    int32_t liv2KCacheBlockSize_ = 0;
    int32_t liv2MaxBlockNumPerBatch_ = 0;
    uint32_t liv2TopkCount_ = 0;
    uint32_t liv2TopkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t liv2TopkCountAlign16_ = 0;
    uint32_t liv2TrunkLen_ = 0;
    bool liv2ReturnValueFlag = false;

    struct LIV2Common::ConstInfo liv2ConstInfo_;
    LIV2Common::LdSplitCoreInfo liv2LdInfo_;
    liV2Topk::LIV2Topk<SCORE_T> liv2TopkOp_;
};

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(liv2ResMm1Buf_, 2 * CeilDiv(M_BASIC_BLOCK, 2) * liv2S2BaseSize_ * sizeof(float));
    liv2ResMm1Ub_ = liv2ResMm1Buf_.Get<float>();
    pipe->InitBuffer(liv2WeightBuf_, 2 * CeilDiv(liv2S1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    liv2WeightUb_ = liv2WeightBuf_.Get<W_T>();
    // 大小：2(开dB) * 2 * 128 * 4 = 2KB
    pipe->InitBuffer(liv2OutBuf_, 2 * CeilDiv(liv2S1BaseSize_, 2) * liv2S2BaseSize_ * sizeof(SCORE_T));
    // out
    // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    liv2Vec1OutUb_ = liv2OutBuf_.Get<SCORE_T>();
    pipe->InitBuffer(liv2MrgValueBuf_, (liv2TopkCountAlign256_ + liv2TrunkLen_) * sizeof(SCORE_T));
    liv2MrgValueLocal_ = liv2MrgValueBuf_.Get<SCORE_T>();
    // returnvalue
    // 大小：topK * sizeof(float) 64:duplicate刷-1需要额外空间
    if (liv2TopkCount_ <= 2048) {
        pipe->InitBuffer(liv2ValueOutBuf_, (liv2TopkCountAlign256_ + 64) * sizeof(float));
        liv2ValueOutLocal_ = liv2ValueOutBuf_.Get<float>();
    } else {
        liv2ValueOutLocal_ = liv2MrgValueBuf_.Get<float>();
    }
    // Topk
    // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    // 大小：topkCountAlign256_ * sizeof(SCORE_T)
    pipe->InitBuffer(liv2IndicesOutBuf_, (liv2TopkCountAlign256_ + 64) * sizeof(uint32_t));
    liv2IndicesOutLocal_ = liv2IndicesOutBuf_.Get<uint32_t>();
    pipe->InitBuffer(liv2ScoreOutBuf_, (liv2TopkCountAlign256_ + 64) * sizeof(SCORE_T));
    liv2ScoreOutLocal_ = liv2ScoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = liv2TopkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(liv2TopkSharedTmpBuf_, topkSharedTmpSize);
    liv2TopkSharedTmpLocal_ = liv2TopkSharedTmpBuf_.Get<uint32_t>();
    liv2TopkOp_.InitBuffers(liv2TopkSharedTmpLocal_, liv2IndicesOutLocal_);
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::InitParams(
    const struct LIV2Common::ConstInfo &constInfo, const LIV2Common::LdSplitCoreInfo &ldInfo,
    const LIV2TilingData *__restrict tilingData)
{
    this->liv2ConstInfo_ = constInfo;
    this->liv2LdInfo_ = ldInfo;
    liv2BlockS2StartIdx_ = 0;
    liv2GSize_ = constInfo.gSize;
    liv2KSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    liv2QHeadNum_ = constInfo.qHeadNum;
    liv2KHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    liv2S1BaseSize_ = constInfo.s1BaseSize; // 4
    liv2S2BaseSize_ = constInfo.s2BaseSize; // 128
    liv2KCacheBlockSize_ = constInfo.kCacheBlockSize;
    liv2MaxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    liv2ReturnValueFlag = constInfo.returnValueFlag;
    liv2BlockId_ = GetBlockIdx();
    liv2TrunkLen_ =
        constInfo.topk > TOPK_LEN_5K ? (constInfo.topk > TOPK_LEN_7K ? TRUNK_LEN_1K : TRUNK_LEN_4K) : TRUNK_LEN_8K;
    liv2TopkCount_ = constInfo.topk;
    liv2TopkOp_.Init(liv2TopkCount_, liv2TrunkLen_);
    liv2TopkCountAlign256_ = LIV2Common::Align(constInfo.topk, (uint64_t)256); // topkCount对齐到256
    liv2TopkCountAlign16_ = LIV2Common::Align(constInfo.topk, (uint64_t)16);
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::InitLDBuffers(
    TPipe *pipe, const LIV2Common::LdSplitCoreInfo &ldInfo)
{
    liv2LdIndexLocal_ = liv2ResMm1Ub_.template ReinterpretCast<uint32_t>();
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::InitVecInputTensor(
    GlobalTensor<W_T> liv2WeightsGm, GlobalTensor<int32_t> liv2IndiceOutGm, GlobalTensor<float> liv2ValueOutGm,
    GlobalTensor<int32_t> liv2BlockTableGm, GlobalTensor<int32_t> liv2OutputIdxOffsetGm)
{
    this->liv2WeightsGm = liv2WeightsGm;
    this->liv2IndiceOutGm = liv2IndiceOutGm;
    this->liv2ValueOutGm = liv2ValueOutGm;
    this->liv2BlockTableGm = liv2BlockTableGm;
    this->liv2OutputIdxOffsetGm = liv2OutputIdxOffsetGm;
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::InitVecWorkspaceTensor(
    GlobalTensor<SCORE_T> liv2ScoreGm, GlobalTensor<SCORE_T> liv2LdScoreGm, GlobalTensor<int32_t> liv2LdIndexGm)
{
    this->liv2ScoreGm = liv2ScoreGm; // resucesum*k
    this->liv2LdScoreGm = liv2LdScoreGm;
    this->liv2LdIndexGm = liv2LdIndexGm;
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(LIV2_VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(LIV2_VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(LIV2_VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(LIV2_VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(LIV2_VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(LIV2_VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(LIV2_VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(LIV2_VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    uint64_t dealSize = liv2ConstInfo_.topk;
    GlobalTensor<int32_t> indexOutput = liv2IndiceOutGm[invalidS1Offset];
    AscendC::InitGlobalMemory(indexOutput, dealSize, liv2ConstInfo_.INVALID_IDX);

    if (liv2ReturnValueFlag) {
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        Duplicate(liv2ValueOutLocal_.template ReinterpretCast<uint32_t>(), liv2ConstInfo_.NEG_INF_FLOAT,
                  liv2ConstInfo_.topk);

        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

        AscendC::DataCopyParams copyOutValueParams;
        copyOutValueParams.blockCount = 1;
        copyOutValueParams.blockLen = liv2ConstInfo_.topk * sizeof(float);
        copyOutValueParams.srcStride = 0;
        copyOutValueParams.dstStride = 0;
        AscendC::DataCopyPad(liv2ValueOutGm[invalidS1Offset], liv2ValueOutLocal_, copyOutValueParams);
    }
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::DoTndPadding(
    const LIV2Common::RunInfo &runInfo)
{
    LIV2Common::DoTndPadding(runInfo, liv2ConstInfo_.topk, liv2ConstInfo_.returnValueFlag, liv2ConstInfo_.NEG_INF_FLOAT,
                             liv2ConstInfo_.kHeadNum, liv2ConstInfo_.INVALID_IDX, MTE3_V_EVENT, liv2IndiceOutGm,
                             liv2ValueOutGm);
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::ProcessVec1(const LIV2Common::RunInfo &info)
{
    auto pingpong = info.loop % 2;
    // CV同步
    CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LIV2Common::ConstInfo::CROSS_CV_EVENT +
                                                                    pingpong); // V核等C核计算完mm1，mm1Res已搬运到UB

    int64_t curS1Idx = info.gS1Idx * liv2S1BaseSize_;
    int64_t curS2Idx = info.s2Idx * liv2S2BaseSize_;
    int64_t curS1ProcNum =
        curS1Idx + liv2S1BaseSize_ > info.actS1Size ? info.actS1Size % liv2S1BaseSize_ : liv2S1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (liv2BlockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (liv2BlockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    if (curAivS1ProcNum == 0) {
        // V核处理完，通知C核可以把mm1Res搬运到UB
        CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LIV2Common::ConstInfo::CROSS_VC_EVENT +
                                                                       pingpong);
        return;
    }
    // weightsGm --> weightUB_
    WaitFlag<HardEvent::V_MTE2>(LIV2_VEC1_V_MTE2_EVENT + pingpong);
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * liv2KHeadNum_ * liv2GSize_;
    DataCopyPadExtParams<W_T> padWeightsParams{true, 0, 0, 0};
    DataCopyExtParams qwDataCopyExtParams;
    qwDataCopyExtParams.blockCount = curAivS1ProcNum;
    qwDataCopyExtParams.blockLen = liv2GSize_ * sizeof(W_T);
    qwDataCopyExtParams.srcStride = 0;
    qwDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qwDataCopyExtParams.blockLen) / 32;
    DataCopyPad(liv2WeightUb_[pingpong * (UB_BANK_STRIDE / sizeof(W_T))], liv2WeightsGm[weightGmOffset],
                qwDataCopyExtParams, padWeightsParams);

    SetFlag<HardEvent::MTE2_V>(LIV2_VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(LIV2_VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE3_V>(LIV2_VEC1_MTE3_V_EVENT + pingpong);
    auto qkVLStride = (UB_BANK_DEPTH_STRIDE / sizeof(QK_T)) / 2 * M_BASIC_BLOCK;
    auto outBase = liv2Vec1OutUb_[(pingpong)*CeilDiv(liv2S1BaseSize_, 2) * liv2S2BaseSize_];
    auto qkBase = liv2ResMm1Ub_[(pingpong) * (UB_BANK_STRIDE / sizeof(QK_T))];
    auto weightBase = liv2WeightUb_[(pingpong) * (UB_BANK_STRIDE / sizeof(float))];
    liV2Vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T), qkBase, qkVLStride,
                                            (uint32_t)(liv2GSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float)), weightBase,
                                            UB_BANK_DEPTH_STRIDE / sizeof(W_T), liv2GSize_, curAivS1ProcNum);

    // outUB_ --->  scoreGm
    SetFlag<HardEvent::V_MTE2>(LIV2_VEC1_V_MTE2_EVENT + pingpong);
    SetFlag<HardEvent::V_MTE3>(LIV2_VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(LIV2_VEC1_V_MTE3_EVENT + pingpong);
    int64_t vec1OutGmOffset = liv2BlockId_ % 2 == 0 ?
                                  curS2Idx :
                                  CeilDiv(liv2S1BaseSize_, 2) * LIV2Common::Align((uint64_t)liv2ConstInfo_.kSeqSize,
                                                                                  (uint64_t)liv2S2BaseSize_) +
                                      curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = liv2S2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride =
        (LIV2Common::Align((uint64_t)liv2ConstInfo_.kSeqSize, (uint64_t)liv2S2BaseSize_) - liv2S2BaseSize_) *
        sizeof(SCORE_T);

    DataCopyPad(liv2ScoreGm[vec1OutGmOffset], liv2Vec1OutUb_[pingpong * CeilDiv(liv2S1BaseSize_, 2) * liv2S2BaseSize_],
                copyOutParams);
    SetFlag<HardEvent::MTE3_V>(LIV2_VEC1_MTE3_V_EVENT + pingpong);
    // V核处理完，通知C核可以把mm1Res搬运到UB
    CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LIV2Common::ConstInfo::CROSS_VC_EVENT + pingpong);
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::ProcessTopK(const LIV2Common::RunInfo &info)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * liv2S1BaseSize_;
    int64_t curS2EndIdx = (info.s2LoopEnd + 1) * liv2S2BaseSize_;
    int64_t curS2StartIdx = info.s2Start * liv2S2BaseSize_;
    int64_t curS1ProcNum =
        curS1Idx + liv2S1BaseSize_ > info.actS1Size ? info.actS1Size % liv2S1BaseSize_ : liv2S1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (liv2BlockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (liv2BlockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    // LD 需要搬运到的起始位置 32字节
    int64_t liV2LdDstOffset = (info.isNeedLD) ? curS2StartIdx : 0;
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = liv2TopkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyParams ldCopyIndicesOutParams;
    ldCopyIndicesOutParams.blockCount = 1;
    ldCopyIndicesOutParams.blockLen = liv2TopkCountAlign16_ * sizeof(uint32_t); // bytes
    ldCopyIndicesOutParams.srcStride = 0;
    ldCopyIndicesOutParams.dstStride = 0;

    AscendC::DataCopyParams ldCopyScoreOutParams;
    ldCopyScoreOutParams.blockCount = 1;
    ldCopyScoreOutParams.blockLen = liv2TopkCountAlign16_ * sizeof(SCORE_T); // bytes
    ldCopyScoreOutParams.srcStride = 0;
    ldCopyScoreOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (liv2ConstInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validAllS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = liv2BlockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = liv2BlockId_ % 2 * CeilDiv(liv2S1BaseSize_, 2) + i;
        int64_t outputIdxOffset = 0;
        if (info.isOutputIdxOffsetValid) {
            outputIdxOffset = liv2OutputIdxOffsetGm.GetValue(info.outputIdxCoreOffset + rowIdx * liv2KHeadNum_);
        }

        SCORE_T liV2Zero = 0;
        int32_t liV2Neg = -1;
        uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
        if (liv2ConstInfo_.attenMaskFlag) {
            validAllS2Len = ((int32_t)i + cuRealAcSeq) / static_cast<int32_t>(liv2ConstInfo_.cmpRatio);
        }
        int32_t validS2Len = validAllS2Len;
        if (info.isNeedLD) {
            // 当前核处理的s2长度validS2Len
            validS2Len = Min((info.s2LoopEnd + 1) * liv2S2BaseSize_, validAllS2Len) - curS2StartIdx;
        }
        uint64_t ldOffset =
            info.saveWorkSpaceIdx * liv2S1BaseSize_ * liv2TopkCountAlign16_ + rowIdx * liv2TopkCountAlign16_;
        if (validS2Len <= 0 && !info.isNeedLD) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>(), liV2Neg, liv2TopkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(liv2IndiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * liv2TopkCount_],
                                 liv2IndicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (liv2ReturnValueFlag) {
                WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

                Duplicate(liv2ValueOutLocal_.template ReinterpretCast<uint32_t>(), liv2ConstInfo_.NEG_INF_FLOAT,
                          liv2TopkCount_);

                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams liV2CopyOutValueParams;
                liV2CopyOutValueParams.blockCount = 1;
                liV2CopyOutValueParams.blockLen = liv2TopkCount_ * sizeof(float);
                liV2CopyOutValueParams.srcStride = 0;
                liV2CopyOutValueParams.dstStride = 0;
                AscendC::DataCopyPad(liv2ValueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * liv2TopkCount_],
                                     liv2ValueOutLocal_, liV2CopyOutValueParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            }
            continue;
        } else if (validS2Len <= 0 && info.isNeedLD) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>(), liV2Neg, liv2TopkCountAlign16_);
            Duplicate(liv2ScoreOutLocal_, liV2Zero, liv2TopkCountAlign16_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(liv2LdScoreGm[ldOffset], liv2ScoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(liv2LdIndexGm[ldOffset], liv2IndicesOutLocal_.ReinterpretCast<int32_t>(),
                                 ldCopyIndicesOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= liv2TopkCount_) {
            uint32_t liV2S2LoopNum = (validS2Len + liv2TrunkLen_ - 1) / liv2TrunkLen_;
            bool useSingleLoop = (liV2S2LoopNum == 1) ||
                                 ((liv2TopkCount_ > liv2TrunkLen_) && (validS2Len <= (uint32_t)liv2TopkCountAlign256_));
            if (useSingleLoop) {
                uint32_t validS2LenAlign = LIV2Common::Align(validS2Len, (int32_t)256);
                Duplicate(liv2MrgValueLocal_[validS2Len / 256 * 256], liV2Zero,
                          validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(liv2MrgValueLocal_,
                                     liv2ScoreGm[vecOffset * LIV2Common::Align((uint64_t)liv2ConstInfo_.kSeqSize,
                                                                               (uint64_t)liv2S2BaseSize_) +
                                                 liV2LdDstOffset],
                                     copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                liv2TopkOp_(liv2MrgValueLocal_, liv2IndicesOutLocal_, liv2ScoreOutLocal_, validS2LenAlign, 0, 1,
                            info.isNeedLD, liv2ReturnValueFlag, outputIdxOffset);
            } else {
                uint32_t liV2ActS2LoopNum = 0;
                uint32_t outputIdxOffsetTmp = 0;
                if (liv2TopkCount_ > liv2TrunkLen_) {
                    liV2ActS2LoopNum = 1 + (validS2Len - liv2TopkCountAlign256_ + liv2TrunkLen_ - 1) / liv2TrunkLen_;
                } else {
                    liV2ActS2LoopNum = (validS2Len + liv2TrunkLen_ - 1) / liv2TrunkLen_;
                }
                for (uint32_t loopIdx = 0; loopIdx < liV2ActS2LoopNum; loopIdx++) {
                    if (loopIdx == liV2ActS2LoopNum - 1) {
                        outputIdxOffsetTmp = outputIdxOffset;
                    }
                    if (loopIdx == 0) {
                        if (liv2TopkCount_ > liv2TrunkLen_) {
                            copyInParams.blockLen = liv2TopkCountAlign256_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(
                                liv2ScoreOutLocal_,
                                liv2ScoreGm[vecOffset * LIV2Common::Align((uint64_t)liv2ConstInfo_.kSeqSize,
                                                                          (uint64_t)liv2S2BaseSize_) +
                                            liV2LdDstOffset],
                                copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            AscendC::CreateVecIndex(liv2IndicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)liV2Zero,
                                                    liv2TopkCountAlign256_);
                            AscendC::CreateVecIndex(liv2TopkSharedTmpLocal_.ReinterpretCast<int32_t>(),
                                                    (int32_t)liV2Zero, liv2TopkCountAlign256_);
                        } else {
                            copyInParams.blockLen = liv2TrunkLen_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(
                                liv2MrgValueLocal_,
                                liv2ScoreGm[vecOffset * LIV2Common::Align((uint64_t)liv2ConstInfo_.kSeqSize,
                                                                          (uint64_t)liv2S2BaseSize_) +
                                            liV2LdDstOffset],
                                copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            liv2TopkOp_(liv2MrgValueLocal_, liv2IndicesOutLocal_, liv2ScoreOutLocal_, liv2TrunkLen_,
                                        loopIdx, liV2ActS2LoopNum, info.isNeedLD, liv2ReturnValueFlag,
                                        outputIdxOffsetTmp);
                        }

                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t liV2ValidTrunkLen = 0U;
                    uint32_t offset = 0;
                    if (liv2TopkCount_ > liv2TrunkLen_) {
                        liV2ValidTrunkLen =
                            (liv2TopkCountAlign256_ + (loopIdx - 1) * liv2TrunkLen_ + liv2TrunkLen_) > validS2Len ?
                                (validS2Len - liv2TopkCountAlign256_) % liv2TrunkLen_ :
                                liv2TrunkLen_;
                        offset = vecOffset *
                                     LIV2Common::Align((uint64_t)liv2ConstInfo_.kSeqSize, (uint64_t)liv2S2BaseSize_) +
                                 liv2TopkCountAlign256_ + (loopIdx - 1) * liv2TrunkLen_ + liV2LdDstOffset;
                    } else {
                        liV2ValidTrunkLen = (loopIdx * liv2TrunkLen_ + liv2TrunkLen_) > validS2Len ?
                                                validS2Len % liv2TrunkLen_ :
                                                liv2TrunkLen_;
                        offset = vecOffset *
                                     LIV2Common::Align((uint64_t)liv2ConstInfo_.kSeqSize, (uint64_t)liv2S2BaseSize_) +
                                 loopIdx * liv2TrunkLen_ + liV2LdDstOffset;
                    }
                    AscendC::DataCopy(liv2MrgValueLocal_, liv2ScoreOutLocal_, liv2TopkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    // 如果topk > trunkLen，第一轮调用topk是直接拷贝的，不需要刷0
                    bool isZeroPadding = (liv2TopkCount_ > liv2TrunkLen_) ? (loopIdx > 1) : true;
                    if (liv2TopkCountAlign256_ != liv2TopkCount_ && isZeroPadding) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (liv2TopkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(liv2MrgValueLocal_[liv2TopkCount_ / 64 * 64], liV2Zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(liv2MrgValueLocal_[liv2TopkCount_ / 64 * 64 + 64], liV2Zero,
                                  liv2TopkCountAlign256_ - (liv2TopkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = liV2ValidTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((liv2TopkCountAlign256_ + liV2ValidTrunkLen) % 256 != 0) {
                        Duplicate(liv2MrgValueLocal_[liv2TopkCountAlign256_ + liV2ValidTrunkLen / 256 * 256], liV2Zero,
                                  LIV2Common::Align(liV2ValidTrunkLen, (uint32_t)256) - liV2ValidTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(liv2MrgValueLocal_[liv2TopkCountAlign256_], liv2ScoreGm[offset], copyInParams,
                                         padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    liv2TopkOp_(liv2MrgValueLocal_, liv2IndicesOutLocal_, liv2ScoreOutLocal_,
                                LIV2Common::Align(liv2TopkCountAlign256_ + liV2ValidTrunkLen, (uint32_t)256), loopIdx,
                                liV2ActS2LoopNum, info.isNeedLD, liv2ReturnValueFlag, outputIdxOffsetTmp);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(liv2IndicesOutLocal_.ReinterpretCast<int32_t>(),
                                    (int32_t)(liV2Zero + outputIdxOffset), validS2Len);

            // 如果需要LD 则将需要保存Value值
            if (info.isNeedLD || liv2ReturnValueFlag) {
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                copyInParams.blockLen = LIV2Common::Align(validS2Len, (int32_t)32) * sizeof(SCORE_T);
                AscendC::DataCopyPad(liv2ScoreOutLocal_,
                                     liv2ScoreGm[vecOffset * LIV2Common::Align((uint64_t)liv2ConstInfo_.kSeqSize,
                                                                               (uint64_t)liv2S2BaseSize_) +
                                                 liV2LdDstOffset],
                                     copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
        }
        if (!info.isNeedLD) {
            if (validS2Len < liv2TopkCount_) {
                uint64_t liV2Mask[1];
                liV2Mask[0] = ~0;
                liV2Mask[0] = liV2Mask[0] << (validS2Len % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], liV2Neg, liV2Mask, 1, 1,
                          0);
            }

            if (validS2Len / 8 * 8 + 64 < liv2TopkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], liV2Neg,
                          liv2TopkCount_ - (validS2Len / 8 * 8 + 64));
            }

            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(liv2IndiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * liv2TopkCount_],
                                 liv2IndicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);

            // 是否返回Value值
            if (liv2ReturnValueFlag) {
                WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                liV2Vector1::UIntToFloatReturnValue(liv2ValueOutLocal_, liv2ScoreOutLocal_, liv2TopkCountAlign256_,
                                                    liv2ConstInfo_.NEG_INF_FLOAT);
                // 无效值刷0
                if (validS2Len < liv2TopkCount_) {
                    uint64_t mask[1];
                    mask[0] = ~0;
                    mask[0] = mask[0] << (validS2Len % 8);
                    PipeBarrier<PIPE_V>();
                    Duplicate(liv2ValueOutLocal_.template ReinterpretCast<uint32_t>()[validS2Len / 8 * 8],
                              liv2ConstInfo_.NEG_INF_FLOAT, mask, 1, 1, 0);
                }

                if (validS2Len / 8 * 8 + 64 < liv2TopkCount_) {
                    PipeBarrier<PIPE_V>();
                    Duplicate(liv2ValueOutLocal_.template ReinterpretCast<uint32_t>()[validS2Len / 8 * 8 + 64],
                              liv2ConstInfo_.NEG_INF_FLOAT, liv2TopkCount_ - (validS2Len / 8 * 8 + 64));
                }
                SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = liv2TopkCount_ * sizeof(float); // bytes
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                // 搬运到GM
                AscendC::DataCopyPad(liv2ValueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * liv2TopkCount_],
                                     liv2ValueOutLocal_, copyOutValueParams);
            }
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        } else {
            PipeBarrier<PIPE_V>();
            AscendC::Adds(liv2IndicesOutLocal_.ReinterpretCast<int32_t>(),
                          liv2IndicesOutLocal_.ReinterpretCast<int32_t>(), static_cast<int32_t>(curS2StartIdx),
                          liv2TopkCountAlign16_);
            if (validS2Len < liv2TopkCount_) {
                uint64_t mask[1];
                uint64_t maskI[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(liv2ScoreOutLocal_[validS2Len / scoreAlign * scoreAlign], liV2Zero, mask, 1, 1, 0);
                maskI[0] = ~0;
                maskI[0] = maskI[0] << (validS2Len % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], liV2Neg, maskI, 1, 1, 0);
            }
            if (validS2Len / scoreAlign * scoreAlign + 64 < liv2TopkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(liv2ScoreOutLocal_[validS2Len / scoreAlign * scoreAlign + 64], liV2Zero,
                          liv2TopkCount_ - (validS2Len / scoreAlign * scoreAlign + 64));
            }
            if (validS2Len / 8 * 8 + 64 < liv2TopkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], liV2Neg,
                          liv2TopkCount_ - (validS2Len / 8 * 8 + 64));
            }
            if (liv2TopkCountAlign16_ != liv2TopkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (liv2TopkCount_ % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(liv2ScoreOutLocal_[liv2TopkCount_ / scoreAlign * scoreAlign], liV2Zero, mask, 1, 1, 0);
                uint64_t maskIndices[1];
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (liv2TopkCount_ % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>()[liv2TopkCount_ / 8 * 8], liV2Neg, maskIndices,
                          1, 1, 0);
            }

            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(liv2LdScoreGm[ldOffset], liv2ScoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(liv2LdIndexGm[ldOffset], liv2IndicesOutLocal_.ReinterpretCast<int32_t>(),
                                 ldCopyIndicesOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        }
    }
}

template <typename Liv2ServiceTraits>
__aicore__ inline void LightningIndexerV2ServiceVector<Liv2ServiceTraits>::ProcessLD()
{
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = liv2TopkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyParams copyOutValueParams;
    copyOutValueParams.blockCount = 1;
    copyOutValueParams.blockLen = liv2TopkCount_ * sizeof(float); // bytes
    copyOutValueParams.srcStride = 0;
    copyOutValueParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<SCORE_T> scorePadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams ldScoreParams;
    ldScoreParams.blockLen = liv2TopkCountAlign16_ * sizeof(SCORE_T); // bytes
    // s1Basesize-1 两个基本块之间的距离
    ldScoreParams.srcStride = (liv2S1BaseSize_ - 1) * liv2TopkCountAlign16_ * sizeof(SCORE_T);
    ldScoreParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<uint32_t> indexPadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams ldIndexParams;
    ldIndexParams.blockLen = liv2TopkCountAlign16_ * sizeof(uint32_t); // bytes
    ldIndexParams.srcStride = (liv2S1BaseSize_ - 1) * liv2TopkCountAlign16_ * sizeof(uint32_t);
    ldIndexParams.dstStride = 0;

    uint64_t ldProcessLen = liv2LdInfo_.workspaceNum * liv2TopkCountAlign16_;
    if (ldProcessLen <= 0) {
        return;
    }

    SCORE_T liV2Zero = 0;
    int32_t liV2Neg = -1;
    uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
    if (liv2TopkCountAlign16_ > liv2TrunkLen_) {
        AscendC::DataCopyExtParams ldScoreSliceParams;
        ldScoreSliceParams.blockCount = 1;
        ldScoreSliceParams.srcStride = 0;
        ldScoreSliceParams.dstStride = 0;

        AscendC::DataCopyExtParams ldIndexSliceParams;
        ldIndexSliceParams.blockCount = 1;
        ldIndexSliceParams.srcStride = 0;
        ldIndexSliceParams.dstStride = 0;

        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        for (uint32_t j = 0; j < liv2LdInfo_.mNum; j++) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(liv2MrgValueLocal_, liV2Zero, liv2TopkCountAlign256_ + liv2TrunkLen_);
            Duplicate(liv2LdIndexLocal_.ReinterpretCast<int32_t>(), liV2Neg, liv2TopkCountAlign256_ + liv2TrunkLen_);

            uint64_t baseLdGmOffset = liv2LdInfo_.workspaceIdx * liv2S1BaseSize_ * liv2TopkCountAlign16_ +
                                      liv2TopkCountAlign16_ * (liv2LdInfo_.mStart + j);

            ldScoreSliceParams.blockLen = liv2TopkCountAlign16_ * sizeof(SCORE_T);
            ldIndexSliceParams.blockLen = liv2TopkCountAlign16_ * sizeof(uint32_t);

            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            AscendC::DataCopyPad(liv2MrgValueLocal_, liv2LdScoreGm[baseLdGmOffset], ldScoreSliceParams, scorePadParams);
            AscendC::DataCopyPad(liv2LdIndexLocal_, liv2LdIndexGm[baseLdGmOffset].ReinterpretCast<uint32_t>(),
                                 ldIndexSliceParams, indexPadParams);

            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

            if (liv2TopkCountAlign16_ < liv2TopkCountAlign256_) {
                Duplicate(liv2MrgValueLocal_[liv2TopkCountAlign16_], liV2Zero,
                          liv2TopkCountAlign256_ - liv2TopkCountAlign16_);
                Duplicate(liv2LdIndexLocal_.ReinterpretCast<int32_t>()[liv2TopkCountAlign16_], liV2Neg,
                          liv2TopkCountAlign256_ - liv2TopkCountAlign16_);
                PipeBarrier<PIPE_V>();
            }

            AscendC::DataCopy(liv2IndicesOutLocal_, liv2LdIndexLocal_, liv2TopkCountAlign256_);
            AscendC::DataCopy(liv2ScoreOutLocal_, liv2MrgValueLocal_, liv2TopkCountAlign256_);

            for (uint32_t workspaceOffset = 1; workspaceOffset < liv2LdInfo_.workspaceNum; workspaceOffset++) {
                for (uint32_t chunkOffset = 0; chunkOffset < liv2TopkCountAlign16_; chunkOffset += liv2TrunkLen_) {
                    uint32_t remainLen = liv2TopkCountAlign16_ - chunkOffset;
                    uint32_t chunkLen = remainLen > liv2TrunkLen_ ? liv2TrunkLen_ : remainLen;
                    uint64_t liV2LDGmOffset =
                        (liv2LdInfo_.workspaceIdx + workspaceOffset) * liv2S1BaseSize_ * liv2TopkCountAlign16_ +
                        liv2TopkCountAlign16_ * (liv2LdInfo_.mStart + j) + chunkOffset;

                    PipeBarrier<PIPE_V>();
                    Duplicate(liv2MrgValueLocal_[liv2TopkCountAlign256_], liV2Zero, liv2TrunkLen_);
                    Duplicate(liv2LdIndexLocal_.ReinterpretCast<int32_t>()[liv2TopkCountAlign256_], liV2Neg,
                              liv2TrunkLen_);

                    ldScoreSliceParams.blockLen = chunkLen * sizeof(SCORE_T);
                    ldIndexSliceParams.blockLen = chunkLen * sizeof(uint32_t);
                    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                    AscendC::DataCopyPad(liv2MrgValueLocal_[liv2TopkCountAlign256_], liv2LdScoreGm[liV2LDGmOffset],
                                         ldScoreSliceParams, scorePadParams);
                    AscendC::DataCopyPad(liv2LdIndexLocal_[liv2TopkCountAlign256_],
                                         liv2LdIndexGm[liV2LDGmOffset].ReinterpretCast<uint32_t>(), ldIndexSliceParams,
                                         indexPadParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

                    uint32_t s2Len = liv2TopkCountAlign256_ + chunkLen;
                    uint32_t s2LenAlign = LIV2Common::Align(s2Len, (uint32_t)256);

                    liv2TopkOp_.LdTopK(liv2MrgValueLocal_, liv2LdIndexLocal_, liv2IndicesOutLocal_, liv2ScoreOutLocal_,
                                       s2LenAlign, j, liv2LdInfo_.workspaceNum);
                    if (liv2TopkCountAlign256_ != liv2TopkCount_) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (liv2TopkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        Duplicate(liv2ScoreOutLocal_[liv2TopkCount_ / 64 * 64], liV2Zero, mask, 1, 1, 0);
                        uint64_t maskIndices[1];
                        maskIndices[0] = ~0;
                        maskIndices[0] = maskIndices[0] << (liv2TopkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>()[liv2TopkCount_ / 64 * 64], liV2Neg,
                                  maskIndices, 1, 1, 0);
                    }
                    if (liv2TopkCount_ / 64 * 64 + 64 < liv2TopkCountAlign256_) {
                        PipeBarrier<PIPE_V>();
                        Duplicate(liv2ScoreOutLocal_[liv2TopkCount_ / 64 * 64 + 64], liV2Zero,
                                  liv2TopkCountAlign256_ - (liv2TopkCount_ / 64 * 64 + 64));
                        PipeBarrier<PIPE_V>();
                        Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>()[liv2TopkCount_ / 64 * 64 + 64],
                                  liV2Neg, liv2TopkCountAlign256_ - (liv2TopkCount_ / 64 * 64 + 64));
                    }
                    PipeBarrier<PIPE_V>();
                    AscendC::DataCopy(liv2LdIndexLocal_, liv2IndicesOutLocal_, liv2TopkCountAlign256_);
                    AscendC::DataCopy(liv2MrgValueLocal_, liv2ScoreOutLocal_, liv2TopkCountAlign256_);
                }
            }

            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            uint64_t indiceOutGmOffset =
                liv2LdInfo_.indiceOutCoreOffset + (liv2LdInfo_.mStart + j) * liv2ConstInfo_.kHeadNum * liv2TopkCount_;
            AscendC::DataCopyPad(liv2IndiceOutGm[indiceOutGmOffset], liv2IndicesOutLocal_.ReinterpretCast<int32_t>(),
                                 copyOutParams);

            if (liv2ReturnValueFlag) {
                PipeBarrier<PIPE_V>();
                liV2Vector1::UIntToFloatReturnValue(liv2ValueOutLocal_, liv2ScoreOutLocal_, liv2TopkCountAlign256_,
                                                    liv2ConstInfo_.NEG_INF_FLOAT);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                AscendC::DataCopyPad(liv2ValueOutGm[indiceOutGmOffset], liv2ValueOutLocal_, copyOutValueParams);
            }
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        }
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        return;
    }
    uint64_t mrgValueLen = liv2TrunkLen_ + liv2TopkCountAlign256_;
    uint32_t ldWorkspaceNum =
        (ldProcessLen > mrgValueLen) ? (liv2TrunkLen_ / liv2TopkCountAlign16_) : liv2LdInfo_.workspaceNum;
    // 搬运次数，一次搬运ldworkspaceNum块
    uint32_t ldProcessNum = CeilDiv(liv2LdInfo_.workspaceNum, ldWorkspaceNum);
    uint32_t liV2LdProWorkspaceNum = liv2LdInfo_.workspaceNum;
    uint32_t liV2LdProcessOffset = 0;
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    for (uint32_t j = 0; j < liv2LdInfo_.mNum; j++) {
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // 读取数据段过长
            if (ldProcessNum > 1) {
                liV2LdProWorkspaceNum = (i == ldProcessNum - 1) ? liv2LdInfo_.workspaceNum - i * ldWorkspaceNum :
                                                                  ldWorkspaceNum; // 当前搬运块数
            }
            liV2LdProcessOffset = (i != 0) ? liv2TopkCountAlign256_ : 0;
            PipeBarrier<PIPE_V>();
            // 索引全部刷-1 value全部刷0
            Duplicate(liv2MrgValueLocal_[liV2LdProcessOffset], liV2Zero,
                      liv2TopkCountAlign256_ + liv2TrunkLen_ - liV2LdProcessOffset);
            Duplicate(liv2LdIndexLocal_.ReinterpretCast<int32_t>()[liV2LdProcessOffset], liV2Neg,
                      liv2TopkCountAlign256_ + liv2TrunkLen_ - liV2LdProcessOffset);

            ldScoreParams.blockCount = liV2LdProWorkspaceNum;
            ldIndexParams.blockCount = liV2LdProWorkspaceNum;

            int32_t s2Len = liv2TopkCountAlign16_ * liV2LdProWorkspaceNum + liV2LdProcessOffset;
            uint32_t s2LenAlign = LIV2Common::Align(s2Len, (int32_t)256); // 寄存器需要256对齐
            uint64_t liV2LDGmOffset =
                LIV2Common::GetLdGmOffset(liv2LdInfo_, liv2S1BaseSize_, liv2TopkCountAlign16_, j, i,
                                          ldWorkspaceNum); // 加入LD处理偏移
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            AscendC::DataCopyPad(liv2MrgValueLocal_[liV2LdProcessOffset], liv2LdScoreGm[liV2LDGmOffset], ldScoreParams,
                                 scorePadParams);
            AscendC::DataCopyPad(liv2LdIndexLocal_[liV2LdProcessOffset],
                                 liv2LdIndexGm[liV2LDGmOffset].ReinterpretCast<uint32_t>(), ldIndexParams,
                                 indexPadParams);

            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

            // 对非对齐索引和值刷0 -1
            if (s2LenAlign != s2Len) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (s2Len % 64);
                Duplicate(liv2MrgValueLocal_[s2Len / 64 * 64], liV2Zero, mask, 1, 1, 0);
                // 把s2Len对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                Duplicate(liv2LdIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64], liV2Neg, mask, 1, 1, 0);
                PipeBarrier<PIPE_V>();
            }
            if (s2Len / 64 * 64 + 64 < s2LenAlign) {
                PipeBarrier<PIPE_V>();
                Duplicate(liv2MrgValueLocal_[s2Len / 64 * 64 + 64], liV2Zero, s2LenAlign - (s2Len / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(liv2LdIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64 + 64], liV2Neg,
                          s2LenAlign - (s2Len / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
            }

            liv2TopkOp_.LdTopK(liv2MrgValueLocal_, liv2LdIndexLocal_, liv2IndicesOutLocal_, liv2ScoreOutLocal_,
                               s2LenAlign, j, ldProcessNum);
            if (liv2TopkCountAlign256_ != liv2TopkCount_) {
                uint64_t mask[1];
                uint64_t maskIndices[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (liv2TopkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(liv2ScoreOutLocal_[liv2TopkCount_ / 64 * 64], liV2Zero, mask, 1, 1, 0);
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (liv2TopkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>()[liv2TopkCount_ / 64 * 64], liV2Neg,
                          maskIndices, 1, 1, 0);
            }
            if (liv2TopkCount_ / 64 * 64 + 64 < liv2TopkCountAlign256_) {
                PipeBarrier<PIPE_V>();
                Duplicate(liv2ScoreOutLocal_[liv2TopkCount_ / 64 * 64 + 64], liV2Zero,
                          liv2TopkCountAlign256_ - (liv2TopkCount_ / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(liv2IndicesOutLocal_.ReinterpretCast<int32_t>()[liv2TopkCount_ / 64 * 64 + 64], liV2Neg,
                          liv2TopkCountAlign256_ - (liv2TopkCount_ / 64 * 64 + 64));
            }

            PipeBarrier<PIPE_V>();
            uint32_t copyLen = liv2TopkCountAlign256_;
            AscendC::DataCopy(liv2LdIndexLocal_, liv2IndicesOutLocal_, copyLen);
            AscendC::DataCopy(liv2MrgValueLocal_, liv2ScoreOutLocal_, copyLen);
        }
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        uint64_t indiceOutGmOffset =
            liv2LdInfo_.indiceOutCoreOffset + (liv2LdInfo_.mStart + j) * liv2ConstInfo_.kHeadNum * liv2TopkCount_;
        AscendC::DataCopyPad(liv2IndiceOutGm[indiceOutGmOffset], liv2IndicesOutLocal_.ReinterpretCast<int32_t>(),
                             copyOutParams);

        if (liv2ReturnValueFlag) {
            PipeBarrier<PIPE_V>();
            liV2Vector1::UIntToFloatReturnValue(liv2ValueOutLocal_, liv2ScoreOutLocal_, liv2TopkCountAlign256_,
                                                liv2ConstInfo_.NEG_INF_FLOAT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(liv2ValueOutGm[indiceOutGmOffset], liv2ValueOutLocal_, copyOutValueParams);
        }
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}
} // namespace LIV2Kernel
#endif
