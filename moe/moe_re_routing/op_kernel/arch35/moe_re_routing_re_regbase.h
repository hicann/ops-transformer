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
 * \file moe_re_routing_re_regbase.h
 * \brief
 */

#ifndef MOE_RE_ROUTING_RE_REGBASE_H
#define MOE_RE_ROUTING_RE_REGBASE_H

#include "kernel_operator.h"
#include "platform.h"
#include "op_kernel/math_util.h"
#include "moe_re_routing_common.h"

namespace MoeReRouting {
using namespace AscendC;

template <typename T, typename TIndex, typename TScale, bool hasScales>
class MoeReRoutingReRegbase {
public:
    __aicore__ inline MoeReRoutingReRegbase(TPipe *pipe, const MoeReRoutingReTilingData *tiling)
        : pipe_(pipe),
          tilingData_(tiling)
    {}
    __aicore__ inline void Init(GM_ADDR tokens, GM_ADDR expertTokenNumPerRank, GM_ADDR perTokenScales,
                                GM_ADDR expertTopkWeight, GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales,
                                GM_ADDR permuteTokenIdx, GM_ADDR expertTokenNum, GM_ADDR permuteTopkWeight);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void PrepareParams();
    __aicore__ inline void CalSrcOffSet(const int64_t rIdx, const int64_t eIdx);
    __aicore__ inline void CalDstOffSet(const int64_t rIdx, const int64_t eIdx);
    __aicore__ inline int64_t CalTokenOffset(const int64_t tokenOffset) const;
    __aicore__ inline void ProcessTokenScale(const int64_t currTokenNum, const int64_t tokSclSize, const bool isScale);
    __aicore__ inline void CalcExpertTokenNum();
    __aicore__ inline void CopyIn(const int64_t blockLen, const int64_t offset, const bool isScale);
    __aicore__ inline void CopyOut(const int64_t blockLen, const int64_t offset, const bool isScale);
    __aicore__ inline void CopyOutIndex(const int64_t rows, const int64_t srcOffset, const int64_t dstOffset);
    __aicore__ inline void CopyTopkWeight(const int64_t rows, const int64_t srcOffset, const int64_t dstOffset);

    TPipe *pipe_ = nullptr;
    const MoeReRoutingReTilingData *tilingData_;
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
    int64_t rIdx_ = 0;
    int64_t eIdx_ = 0;
    int64_t rNum_ = 0; // 当前核处理多少个rank
    int64_t eNum_ = 0; // 当前核处理多少个expert
    int64_t tokenSize_ = 0;
    int64_t tokenSizeOrigin_ = 0;
    int64_t scaleSize_ = 0;
    int64_t rankNum_ = 0;
    int64_t expertNum_ = 0;
    int64_t tokensSrc_ = 0;
    int64_t tokensDst_ = 0;
};

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::Init(
    GM_ADDR tokens, GM_ADDR expertTokenNumPerRank, GM_ADDR perTokenScales, GM_ADDR expertTopkWeight,
    GM_ADDR permuteTokens, GM_ADDR permutePerTokenScales, GM_ADDR permuteTokenIdx, GM_ADDR expertTokenNum,
    GM_ADDR permuteTopkWeight)
{
    blockIdx_ = GetBlockIdx();
    tokenSize_ = tilingData_->tokenSize;
    tokenSizeOrigin_ = tilingData_->tokenSizeOrigin;
    scaleSize_ = tilingData_->scaleSize;
    rankNum_ = tilingData_->rankNum;
    expertNum_ = tilingData_->expertNum;
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
__aicore__ inline int64_t MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::CalTokenOffset(
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
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::PrepareParams()
{
    rIdx_ = blockIdx_ % tilingData_->blockNumR;
    eIdx_ = blockIdx_ / tilingData_->blockNumR;
    rNum_ = tilingData_->blockFactorR;
    eNum_ = tilingData_->blockFactorE;
    if (rIdx_ == tilingData_->blockNumR - 1 && rankNum_ % tilingData_->blockFactorR != 0) {
        rNum_ = rankNum_ % tilingData_->blockFactorR;
    }
    if (eIdx_ == tilingData_->blockNumE - 1 && expertNum_ % tilingData_->blockFactorE != 0) {
        eNum_ = expertNum_ % tilingData_->blockFactorE;
    }
    return;
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::CalSrcOffSet(const int64_t rIdx,
                                                                                         const int64_t eIdx)
{
    tokensSrc_ = 0;
    for (int64_t r = 0; r < rIdx; r++) {
        for (int64_t e = 0; e < expertNum_; e++) {
            tokensSrc_ += srcRankTokenGm_.GetValue(r * expertNum_ + e);
        }
    }
    for (int64_t e = 0; e < eIdx; e++) {
        tokensSrc_ += srcRankTokenGm_.GetValue(rIdx * expertNum_ + e);
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::CalDstOffSet(const int64_t rIdx,
                                                                                         const int64_t eIdx)
{
    tokensDst_ = 0;
    for (int64_t e = 0; e < eIdx; e++) {
        for (int64_t r = 0; r < rankNum_; r++) {
            tokensDst_ += srcRankTokenGm_.GetValue(r * expertNum_ + e);
        }
    }
    for (int64_t r = 0; r < rIdx; r++) {
        tokensDst_ += srcRankTokenGm_.GetValue(r * expertNum_ + eIdx);
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::Process()
{
    PrepareParams();
    int64_t rIdxStart = rIdx_ * tilingData_->blockFactorR;
    int64_t eIdxStart = eIdx_ * tilingData_->blockFactorE;
    for (int64_t rIdx = rIdxStart; rIdx < rIdxStart + rNum_; rIdx++) {
        CalSrcOffSet(rIdx, eIdxStart);
        for (int64_t eIdx = eIdxStart; eIdx < eIdxStart + eNum_; eIdx++) {
            int64_t currTokenNum = srcRankTokenGm_.GetValue(rIdx * expertNum_ + eIdx);
            if (currTokenNum == 0) {
                continue;
            }
            CalDstOffSet(rIdx, eIdx);
            ProcessTokenScale(currTokenNum, tokenSize_, false);
            if constexpr (hasScales) {
                ProcessTokenScale(currTokenNum, scaleSize_, true);
            }
            if (tilingData_->hasTopkWeight == HAS_TOPK_WEIGHT) {
                CopyTopkWeight(currTokenNum, tokensSrc_, tokensDst_);
            }
            tokensSrc_ += currTokenNum;
        }
    }
    CalcExpertTokenNum();
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::ProcessTokenScale(
    const int64_t currTokenNum, const int64_t tokSclSize, const bool isScale)
{
    int64_t ubFactor = 0;
    if (isScale) {
        ubFactor = Ops::Base::CeilAlign(tilingData_->ubFactor / sizeof(TScale), BLOCK_SIZE / sizeof(TScale));
    } else {
        ubFactor = Ops::Base::CeilAlign(tilingData_->ubFactor / sizeof(T), BLOCK_SIZE / sizeof(T));
    }
    if (ubFactor < tokSclSize) {
        if (!isScale) {
            CopyOutIndex(currTokenNum, tokensSrc_, tokensDst_);
        }
        for (int64_t tIdx = 0; tIdx < currTokenNum; tIdx++) {
            int64_t ubLoopCnt = Ops::Base::CeilDiv(tokSclSize, ubFactor);
            int64_t subTokSclSize = ubFactor;
            for (int64_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
                if (loopIdx == ubLoopCnt - 1 && tokSclSize % ubFactor != 0) {
                    subTokSclSize = tokSclSize % ubFactor;
                }
                int64_t currTokenSrc =
                    (isScale ? (tokensSrc_ + tIdx) * tokSclSize : CalTokenOffset(tokensSrc_ + tIdx)) +
                    loopIdx * ubFactor;
                int64_t currTokenDst =
                    (isScale ? (tokensDst_ + tIdx) * tokSclSize : CalTokenOffset(tokensDst_ + tIdx)) +
                    loopIdx * ubFactor;
                CopyIn(subTokSclSize, currTokenSrc, isScale);
                CopyOut(subTokSclSize, currTokenDst, isScale);
            }
        }
    } else {
        int64_t tokenFactor = Ops::Base::FloorDiv(ubFactor, tokSclSize);
        if constexpr (IsSameType<T, int8_t>::value) {
            if (!isScale && tokenSizeOrigin_ > 0 && tokenSizeOrigin_ % 2 != 0) {
                tokenFactor = 1;
            }
        }
        int64_t ubLoopCnt = Ops::Base::CeilDiv(currTokenNum, tokenFactor);
        int64_t currTokenFactor = tokenFactor;
        for (int64_t loopIdx = 0; loopIdx < ubLoopCnt; loopIdx++) {
            if (loopIdx == ubLoopCnt - 1 && currTokenNum % tokenFactor != 0) {
                currTokenFactor = currTokenNum % tokenFactor;
            }
            int64_t currTokenSrc = isScale ? (tokensSrc_ + loopIdx * tokenFactor) * tokSclSize :
                                             CalTokenOffset(tokensSrc_ + loopIdx * tokenFactor);
            int64_t currTokenDst = isScale ? (tokensDst_ + loopIdx * tokenFactor) * tokSclSize :
                                             CalTokenOffset(tokensDst_ + loopIdx * tokenFactor);
            CopyIn(currTokenFactor * tokSclSize, currTokenSrc, isScale);
            if (!isScale) {
                CopyOutIndex(currTokenFactor, tokensSrc_ + loopIdx * tokenFactor, tokensDst_ + loopIdx * tokenFactor);
            }
            CopyOut(currTokenFactor * tokSclSize, currTokenDst, isScale);
        }
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::CalcExpertTokenNum()
{
    for (int64_t eIdx = blockIdx_; eIdx < expertNum_; eIdx += tilingData_->coreNum) {
        TIndex res = 0;
        for (int64_t rIdx = 0; rIdx < rankNum_; rIdx++) {
            res += srcRankTokenGm_.GetValue(rIdx * expertNum_ + eIdx);
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
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::CopyIn(const int64_t blockLen,
                                                                                   const int64_t offset,
                                                                                   const bool isScale)
{
    if (isScale) {
        LocalTensor<TScale> srcLocal = queBind_.AllocTensor<TScale>();
        DataCopyExtParams copyParams(1, blockLen * sizeof(TScale), 0, 0, 0);
        DataCopyPadExtParams<TScale> padParams(false, 0, 0, 0);
        DataCopyPad(srcLocal, srcScaleGm_[offset], copyParams, padParams);
        queBind_.EnQue(srcLocal);
    } else {
        LocalTensor<T> srcLocal = queBind_.AllocTensor<T>();
        DataCopyExtParams copyParams(1, blockLen * sizeof(T), 0, 0, 0);
        DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
        DataCopyPad(srcLocal, srcTokenGm_[offset], copyParams, padParams);
        queBind_.EnQue(srcLocal);
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::CopyOut(const int64_t blockLen,
                                                                                    const int64_t offset,
                                                                                    const bool isScale)
{
    if (isScale) {
        LocalTensor<TScale> dstLocal = queBind_.DeQue<TScale>();
        DataCopyExtParams copyParams(1, blockLen * sizeof(TScale), 0, 0, 0);
        DataCopyPad(dstScaleGm_[offset], dstLocal, copyParams);
        queBind_.FreeTensor(dstLocal);
    } else {
        LocalTensor<T> dstLocal = queBind_.DeQue<T>();
        DataCopyExtParams copyParams(1, blockLen * sizeof(T), 0, 0, 0);
        DataCopyPad(dstTokenGm_[offset], dstLocal, copyParams);
        queBind_.FreeTensor(dstLocal);
    }
}

template <typename T, typename TIndex, typename TScale, bool hasScales>
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::CopyOutIndex(const int64_t rows,
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
__aicore__ inline void MoeReRoutingReRegbase<T, TIndex, TScale, hasScales>::CopyTopkWeight(const int64_t rows,
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

#endif // MOE_RE_ROUTING_RE_REGBASE_H
