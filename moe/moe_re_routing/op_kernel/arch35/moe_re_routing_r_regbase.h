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
 * \file moe_re_routing_r_regbase.h
 * \brief
 */

#ifndef MOE_RE_ROUTING_R_REGBASE_H
#define MOE_RE_ROUTING_R_REGBASE_H

#include "kernel_operator.h"
#include "platform.h"
#include "op_kernel/math_util.h"
#include "moe_re_routing_common.h"

namespace MoeReRouting {
using namespace AscendC;

template <typename T, typename TIndex, typename TScale, bool hasScales>
class MoeReRoutingRRegbase {
public:
    __aicore__ inline MoeReRoutingRRegbase(TPipe *pipe, const MoeReRoutingRTilingData *tiling)
        : pipe_(pipe),
          tilingData_(tiling)
    {}
    __aicore__ inline void Init(GM_ADDR tokens, GM_ADDR expertTokenNumPerRank, GM_ADDR perTokenScales,
                                GM_ADDR expertTopkWeight, GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales,
                                GM_ADDR permuteTokenIdx, GM_ADDR expertTokenNum, GM_ADDR permuteTopkWeight);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void ProcessTokenScale(const int64_t currTokenNum);
    __aicore__ inline void UbOverLoad(const int64_t currTokenNum);
    __aicore__ inline void UbFullLoad(const int64_t currTokenNum, const int64_t tokSclSize);
    __aicore__ inline void CalcExpertTokenNum();
    __aicore__ inline int64_t CalTokenOffset(const int64_t tokenOffset) const;
    __aicore__ inline void CopyIn(const int64_t currFactor, const int64_t offset);
    __aicore__ inline void CopyOut(const int64_t currFactor, const int64_t offset);
    __aicore__ inline void CopyInToken(const int64_t blockLen, const int64_t offset);
    __aicore__ inline void CopyOutToken(const int64_t blockLen, const int64_t offset);
    __aicore__ inline void CopyInScale(const int64_t blockLen, const int64_t offset);
    __aicore__ inline void CopyOutScale(const int64_t blockLen, const int64_t offset);
    __aicore__ inline void CopyOutIndex(const int64_t rows, const int64_t srcOffset, const int64_t dstOffset);
    __aicore__ inline void CopyTopkWeight(const int64_t rows, const int64_t srcOffset, const int64_t dstOffset);

    TPipe *pipe_ = nullptr;
    const MoeReRoutingRTilingData *tilingData_;
    constexpr static int64_t BLOCK_SIZE = static_cast<int64_t>(platform::GetUbBlockSize());
    GlobalTensor<T> srcTokenGm_;
    GlobalTensor<T> dstTokenGm_;
    GlobalTensor<TScale> srcScaleGm_;
    GlobalTensor<TScale> dstScaleGm_;
    GlobalTensor<TIndex> srcRankTokenGm_;
    GlobalTensor<TIndex> dstExpertTokenGm_;
    GlobalTensor<int32_t> tokenIdxGm_;
    GlobalTensor<float> srcTopkWeightGm_;
    GlobalTensor<float> dstTopkWeightGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind_;
    TQue<QuePosition::VECOUT, 1> idxOutQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> topkWeightQue_;
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t INDEX_UB_SIZE = 256;
    int64_t blockIdx_ = 0;
    int64_t tokensSrc_ = 0;
    int64_t tokensDst_ = 0;
    int64_t tokenSize_ = 0;
    int64_t tokenSizeOrigin_ = 0;
    int64_t scaleSize_ = 0;
};

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::Init(
    GM_ADDR tokens, GM_ADDR expertTokenNumPerRank, GM_ADDR perTokenScales, GM_ADDR expertTopkWeight,
    GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales, GM_ADDR permuteTokenIdx, GM_ADDR expertTokenNum,
    GM_ADDR permuteTopkWeight)
{
    blockIdx_ = GetBlockIdx();
    tokenSize_ = tilingData_->tokenSize;
    tokenSizeOrigin_ = tilingData_->tokenSizeOrigin;
    scaleSize_ = tilingData_->scaleSize;
    srcTokenGm_.SetGlobalBuffer((__gm__ T *)tokens);
    dstTokenGm_.SetGlobalBuffer((__gm__ T *)permuteTokens);
    srcRankTokenGm_.SetGlobalBuffer((__gm__ TIndex *)expertTokenNumPerRank);
    tokenIdxGm_.SetGlobalBuffer((__gm__ int32_t *)permuteTokenIdx);
    dstExpertTokenGm_.SetGlobalBuffer((__gm__ TIndex *)expertTokenNum);
    if constexpr (hasScales) {
        srcScaleGm_.SetGlobalBuffer((__gm__ TScale *)perTokenScales);
        dstScaleGm_.SetGlobalBuffer((__gm__ TScale *)permutePerTokenScales);
    }
    if (tilingData_->hasTopkWeight == HAS_TOPK_WEIGHT) {
        srcTopkWeightGm_.SetGlobalBuffer((__gm__ float *)expertTopkWeight);
        dstTopkWeightGm_.SetGlobalBuffer((__gm__ float *)permuteTopkWeight);
    }
    this->pipe_->InitBuffer(queBind_, DOUBLE_BUFFER,
                            Ops::Base::CeilAlign(tilingData_->ubFactor, static_cast<int64_t>(BLOCK_SIZE / sizeof(T))));
    this->pipe_->InitBuffer(idxOutQue_, DOUBLE_BUFFER,
                            Ops::Base::CeilAlign(INDEX_UB_SIZE * sizeof(TIndex), BLOCK_SIZE / sizeof(TIndex)));
    if (tilingData_->hasTopkWeight == HAS_TOPK_WEIGHT) {
        this->pipe_->InitBuffer(topkWeightQue_, DOUBLE_BUFFER,
                                Ops::Base::CeilAlign(INDEX_UB_SIZE * sizeof(float), BLOCK_SIZE / sizeof(float)));
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline int64_t MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CalTokenOffset(
    const int64_t tokenOffset) const
{
    if constexpr (IsSameType<T, int8_t>::value) {
        if (tokenSizeOrigin_ > 0) {
            // GM中 int8_t 采用了 2:1 压缩存储
            return (tokenOffset * tokenSizeOrigin_) / 2;
        }
    }
    return tokenOffset * tokenSize_;
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::Process()
{
    int64_t rankLoopNum = tilingData_->rankNum / tilingData_->coreNum;
    if (blockIdx_ < tilingData_->rankNum % tilingData_->coreNum) {
        rankLoopNum += 1;
    }
    for (int64_t rankLoopIdx = 0; rankLoopIdx < rankLoopNum; rankLoopIdx++) {
        int64_t currRankId = rankLoopIdx * tilingData_->coreNum + blockIdx_;
        tokensSrc_ = 0;
        tokensDst_ = 0;
        // 计算当前rankid之前的expertId的src偏移
        for (int64_t rankId = 0; rankId < currRankId; rankId++) {
            for (int64_t expertIdx = 0; expertIdx < tilingData_->expertNum; expertIdx++) {
                tokensSrc_ += srcRankTokenGm_(rankId * tilingData_->expertNum + expertIdx);
            }
        }
        // 计算当前rankId之前的dst偏移
        for (int64_t expertIdx = 0; expertIdx < tilingData_->expertNum; expertIdx++) {
            int64_t currTokenNum = srcRankTokenGm_(expertIdx + currRankId * tilingData_->expertNum);
            for (int64_t rankIdx = 0; rankIdx < currRankId; rankIdx++) {
                tokensDst_ += srcRankTokenGm_(rankIdx * tilingData_->expertNum + expertIdx);
            }
            ProcessTokenScale(currTokenNum);
            if (tilingData_->hasTopkWeight == HAS_TOPK_WEIGHT) {
                CopyTopkWeight(currTokenNum, tokensSrc_, tokensDst_);
            }
            tokensSrc_ += currTokenNum;
            // 计算当前rank之后的dst偏移
            for (int64_t rankIdx = currRankId; rankIdx < tilingData_->rankNum; rankIdx++) {
                tokensDst_ += srcRankTokenGm_(rankIdx * tilingData_->expertNum + expertIdx);
            }
        }
    }
    CalcExpertTokenNum();
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::UbOverLoad(const int64_t currTokenNum)
{
    int64_t ubFactor = tilingData_->ubFactor;
    for (int64_t tIdx = 0; tIdx < currTokenNum; tIdx++) {
        int64_t ubLoopCnt = Ops::Base::CeilDiv(tokenSize_, ubFactor);
        int64_t subTokSize = ubFactor;
        for (int64_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
            if (loopIdx == ubLoopCnt - 1 && tokenSize_ % ubFactor != 0) {
                subTokSize = tokenSize_ % ubFactor;
            }
            int64_t currTokenSrc = CalTokenOffset(tokensSrc_ + tIdx) + loopIdx * ubFactor;
            int64_t currTokenDst = CalTokenOffset(tokensDst_ + tIdx) + loopIdx * ubFactor;
            CopyInToken(subTokSize, currTokenSrc);
            CopyOutToken(subTokSize, currTokenDst);
        }
        if constexpr (hasScales) {
            int64_t ubLoopCnt = Ops::Base::CeilDiv(scaleSize_, ubFactor);
            int64_t subScaleSize = ubFactor;
            for (int64_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
                if (loopIdx == ubLoopCnt - 1 && scaleSize_ % ubFactor != 0) {
                    subScaleSize = scaleSize_ % ubFactor;
                }
                int64_t currScaleSrc = (tokensSrc_ + tIdx) * scaleSize_ + loopIdx * ubFactor;
                int64_t currScaleDst = (tokensDst_ + tIdx) * scaleSize_ + loopIdx * ubFactor;
                CopyInScale(subScaleSize, currScaleSrc);
                CopyOutScale(subScaleSize, currScaleDst);
            }
        }
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::UbFullLoad(const int64_t currTokenNum,
                                                                                      const int64_t tokSclSize)
{
    int64_t ubFactor = tilingData_->ubFactor;
    int64_t tokSclFactor = Ops::Base::FloorDiv(ubFactor, tokSclSize);
    if constexpr (IsSameType<T, int8_t>::value) {
        if (tokenSizeOrigin_ > 0 && tokenSizeOrigin_ % 2 != 0) {
            tokSclFactor = 1;
        }
    }
    int64_t ubLoopCnt = Ops::Base::CeilDiv(currTokenNum, tokSclFactor);
    int64_t currTokSclFactor = tokSclFactor;
    for (int64_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
        if (loopIdx == ubLoopCnt - 1 && currTokenNum % tokSclFactor != 0) {
            currTokSclFactor = currTokenNum % tokSclFactor;
        }
        int64_t currTokenSrc = (tokensSrc_ + loopIdx * tokSclFactor);
        int64_t currTokenDst = (tokensDst_ + loopIdx * tokSclFactor);
        CopyIn(currTokSclFactor, currTokenSrc);
        CopyOutIndex(currTokSclFactor, currTokenSrc, currTokenDst);
        CopyOut(currTokSclFactor, currTokenDst);
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::ProcessTokenScale(const int64_t currTokenNum)
{
    int64_t tokSclSize = Ops::Base::CeilAlign(static_cast<int64_t>(tokenSize_ * sizeof(T)), BLOCK_SIZE);
    if constexpr (hasScales) {
        tokSclSize += Ops::Base::CeilAlign(static_cast<int64_t>(scaleSize_ * sizeof(TScale)), BLOCK_SIZE);
    }
    if (tilingData_->ubFactor < tokSclSize) {
        CopyOutIndex(currTokenNum, tokensSrc_, tokensDst_);
        UbOverLoad(currTokenNum);
    } else {
        UbFullLoad(currTokenNum, tokSclSize);
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CalcExpertTokenNum()
{
    for (int64_t eIdx = blockIdx_; eIdx < tilingData_->expertNum; eIdx += tilingData_->coreNum) {
        TIndex res = 0;
        for (int64_t rIdx = 0; rIdx < tilingData_->rankNum; rIdx++) {
            res += srcRankTokenGm_.GetValue(rIdx * tilingData_->expertNum + eIdx);
        }
        LocalTensor<TIndex> expertTokenTensor = idxOutQue_.AllocTensor<TIndex>();
        Duplicate<TIndex>(expertTokenTensor, res, 1);
        idxOutQue_.EnQue(expertTokenTensor);
        expertTokenTensor = idxOutQue_.DeQue<TIndex>();
        DataCopyExtParams copyParams(1, sizeof(TIndex), 0, 0, 0);
        DataCopyPad(dstExpertTokenGm_[eIdx], expertTokenTensor, copyParams);
        idxOutQue_.FreeTensor(expertTokenTensor);
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CopyIn(const int64_t currFactor,
                                                                                  const int64_t offset)
{
    LocalTensor<T> srcLocal = queBind_.AllocTensor<T>();
    DataCopyExtParams copyParams(1, currFactor * tokenSize_ * sizeof(T), 0, 0, 0);
    DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
    DataCopyPad(srcLocal, srcTokenGm_[CalTokenOffset(offset)], copyParams, padParams);
    if constexpr (hasScales) {
        int64_t shiftSize = Ops::Base::CeilAlign(static_cast<int64_t>(currFactor * tokenSize_ * sizeof(T)), BLOCK_SIZE);
        LocalTensor<T> tmpLocal = srcLocal[shiftSize / sizeof(T)];
        LocalTensor<TScale> srcScale = tmpLocal.template ReinterpretCast<TScale>();
        copyParams.blockLen = currFactor * scaleSize_ * sizeof(TScale);
        DataCopyPadExtParams<TScale> padParams(false, 0, 0, 0);
        DataCopyPad(srcScale, srcScaleGm_[offset * scaleSize_], copyParams, padParams);
    }
    queBind_.EnQue(srcLocal);
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CopyOut(const int64_t currFactor,
                                                                                   const int64_t offset)
{
    LocalTensor<T> dstLocal = queBind_.DeQue<T>();
    DataCopyExtParams copyParams(1, currFactor * tokenSize_ * sizeof(T), 0, 0, 0);
    DataCopyPad(dstTokenGm_[CalTokenOffset(offset)], dstLocal, copyParams);
    if constexpr (hasScales) {
        int64_t shiftSize = Ops::Base::CeilAlign(static_cast<int64_t>(currFactor * tokenSize_ * sizeof(T)), BLOCK_SIZE);
        LocalTensor<T> tmpLocal = dstLocal[shiftSize / sizeof(T)];
        LocalTensor<TScale> dstScale = tmpLocal.template ReinterpretCast<TScale>();
        copyParams.blockLen = currFactor * scaleSize_ * sizeof(TScale);
        DataCopyPad(dstScaleGm_[offset * scaleSize_], dstScale, copyParams);
    }
    queBind_.FreeTensor(dstLocal);
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CopyInToken(const int64_t blockLen,
                                                                                       const int64_t offset)
{
    LocalTensor<T> srcLocal = queBind_.AllocTensor<T>();
    DataCopyExtParams copyParams(1, blockLen * sizeof(T), 0, 0, 0);
    DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
    DataCopyPad(srcLocal, srcTokenGm_[offset], copyParams, padParams);
    queBind_.EnQue(srcLocal);
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CopyOutToken(const int64_t blockLen,
                                                                                        const int64_t offset)
{
    LocalTensor<T> dstLocal = queBind_.DeQue<T>();
    DataCopyExtParams copyParams(1, blockLen * sizeof(T), 0, 0, 0);
    DataCopyPad(dstTokenGm_[offset], dstLocal, copyParams);
    queBind_.FreeTensor(dstLocal);
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CopyInScale(const int64_t blockLen,
                                                                                       const int64_t offset)
{
    LocalTensor<TScale> srcLocal = queBind_.AllocTensor<TScale>();
    DataCopyExtParams copyParams(1, blockLen * sizeof(TScale), 0, 0, 0);
    DataCopyPadExtParams<TScale> padParams(false, 0, 0, 0);
    DataCopyPad(srcLocal, srcScaleGm_[offset], copyParams, padParams);
    queBind_.EnQue(srcLocal);
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CopyOutScale(const int64_t blockLen,
                                                                                        const int64_t offset)
{
    LocalTensor<TScale> dstLocal = queBind_.DeQue<TScale>();
    DataCopyExtParams copyParams(1, blockLen * sizeof(TScale), 0, 0, 0);
    DataCopyPad(dstScaleGm_[offset], dstLocal, copyParams);
    queBind_.FreeTensor(dstLocal);
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CopyOutIndex(const int64_t rows,
                                                                                        const int64_t srcOffset,
                                                                                        const int64_t dstOffset)
{
    int64_t head = srcOffset;
    int64_t offset = dstOffset;
    if (tilingData_->idxType == 1) {
        head = dstOffset;
        offset = srcOffset;
    }
    int64_t loopCnt = Ops::Base::CeilDiv(rows, INDEX_UB_SIZE);
    int64_t subRows = INDEX_UB_SIZE;
    for (int64_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        if (loopIdx == loopCnt - 1 && rows % INDEX_UB_SIZE != 0) {
            subRows = rows % INDEX_UB_SIZE;
        }
        LocalTensor indexLocal = idxOutQue_.AllocTensor<int32_t>();
        CreateVecIndex(indexLocal, static_cast<int32_t>(head + loopIdx * INDEX_UB_SIZE), subRows);
        idxOutQue_.EnQue(indexLocal);
        indexLocal = idxOutQue_.DeQue<int32_t>();
        DataCopyExtParams copyParams(1, subRows * sizeof(int32_t), 0, 0, 0);
        DataCopyPad(tokenIdxGm_[offset + loopIdx * INDEX_UB_SIZE], indexLocal, copyParams);
        idxOutQue_.FreeTensor(indexLocal);
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingRRegbase<T, TIndex, TScale, hasScales>::CopyTopkWeight(const int64_t rows,
                                                                                          const int64_t srcOffset,
                                                                                          const int64_t dstOffset)
{
    int64_t loopCnt = Ops::Base::CeilDiv(rows, INDEX_UB_SIZE);
    int64_t subRows = INDEX_UB_SIZE;
    for (int64_t loopIdx = 0; loopIdx < loopCnt; loopIdx++) {
        if (loopIdx == loopCnt - 1 && rows % INDEX_UB_SIZE != 0) {
            subRows = rows % INDEX_UB_SIZE;
        }
        LocalTensor<float> topkLocal = topkWeightQue_.AllocTensor<float>();
        DataCopyExtParams copyParams(1, subRows * sizeof(float), 0, 0, 0);
        DataCopyPadExtParams<float> padParams(false, 0, 0, 0);
        DataCopyPad(topkLocal, srcTopkWeightGm_[srcOffset + loopIdx * INDEX_UB_SIZE], copyParams, padParams);
        topkWeightQue_.EnQue(topkLocal);
        topkLocal = topkWeightQue_.DeQue<float>();
        DataCopyPad(dstTopkWeightGm_[dstOffset + loopIdx * INDEX_UB_SIZE], topkLocal, copyParams);
        topkWeightQue_.FreeTensor(topkLocal);
    }
}

} // end namespace MoeReRouting

#endif // MOE_RE_ROUTING_R_REGBASE_H
