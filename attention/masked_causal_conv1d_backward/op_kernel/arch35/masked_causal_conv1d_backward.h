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
 * \file masked_causal_conv1d_backward.h
 * \brief Arch35 AICore kernel for masked_causal_conv1d_backward
 */

#ifndef MASKED_CAUSAL_CONV1D_BACKWARD_ARCH35_H
#define MASKED_CAUSAL_CONV1D_BACKWARD_ARCH35_H

#include "kernel_operator.h"
#include "vf/compute.h"
#include "masked_causal_conv1d_backward_struct.h"

namespace MaskedCausalConv1dBackwardKernelNS {
using namespace AscendC;
using MaskedCausalConv1dBackwardArch35Tiling::MaskedCausalConv1dBackwardTilingDataV35;

template <typename DT>
class MaskedCausalConv1dBackward {
public:
    static constexpr int kW = 3;
    static constexpr int kBufferNum = 2;
    static constexpr int NUM_2 = 2;
    static constexpr int kAlignBytes = 32;

    __aicore__ inline MaskedCausalConv1dBackward();

    __aicore__ inline void Init(GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR mask, GM_ADDR grad_x,
                                GM_ADDR grad_weight, const MaskedCausalConv1dBackwardTilingDataV35 *td, TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInGradY(int64_t bTile, int64_t sTile, int64_t hTile, int64_t sEff, int64_t hLenThis);
    __aicore__ inline void CopyInX(int64_t bTile, int64_t sTile, int64_t hTile, int64_t sLen, int64_t hLenThis);
    __aicore__ inline void CopyInWeight(int64_t hTile, int64_t hLenThis);
    __aicore__ inline void CopyInMask(int64_t bTile, int64_t sTile, int64_t sLen);
    __aicore__ inline void ApplyMaskToGradY(LocalTensor<DT> &goLocal, int64_t sEff, int64_t hLenThis);
    __aicore__ inline void ComputeGradXWeight(LocalTensor<DT> &goLocal, LocalTensor<DT> &inLocal,
                                              LocalTensor<DT> &wLocal, int64_t sLen, int64_t hLenThis);
    __aicore__ inline void CopyOutGradX(int64_t bTile, int64_t sTile, int64_t hTile, int64_t sLen, int64_t hLenThis);
    __aicore__ inline void ReduceSTile(int64_t sTile, int64_t sLen, int64_t hLenThis);
    __aicore__ inline void ReduceWsTile(int64_t bTile, int64_t hLenThis);
    __aicore__ inline void ReduceBTileAndCopyOut(int64_t hTile, int64_t hLenThis);
    __aicore__ inline void AccumulateLevels(uint32_t kIdx, int64_t hLenThis, uint32_t wsBase, uint32_t wsStrideBytes,
                                            LocalTensor<float> &sumUb);

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

private:
    TPipe *pipe_;
    const MaskedCausalConv1dBackwardTilingDataV35 *td_;
    TQue<QuePosition::VECIN, kBufferNum> gradOutQ_;
    TQue<QuePosition::VECIN, kBufferNum> xQ_;
    TQue<QuePosition::VECIN, kBufferNum> weightQ_;
    TQue<QuePosition::VECIN, kBufferNum> maskQ_;
    TQue<QuePosition::VECOUT, kBufferNum> gradInQ_;
    TQue<QuePosition::VECOUT, kBufferNum> gradWeightQ_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;

    GlobalTensor<DT> gradYGm_;
    GlobalTensor<DT> xGm_;
    GlobalTensor<DT> weightGm_;
    GlobalTensor<DT> gradInGm_;
    GlobalTensor<DT> gradWeightGm_;
    GlobalTensor<bool> maskGm_;
    LocalTensor<float> gwAccF32_[kW];

    int64_t hMainCoreCnt_{0}; // h维度主核核数
    int64_t hTailCoreCnt_{0}; // h维度尾核核数
    int64_t hMainSize_{0};    // h维度主核处理的大小
    int64_t hTailSize_{0};    // h维度尾核处理的大小
    int64_t hasMask_{0};      // 1，有mask；0，无mask
    int64_t S_{0};            // S维度大小
    int64_t B_{0};            // B维度大小
    int64_t H_{0};            // H维度大小
    int64_t hStart_{0}, hLen_{0};

    // loop counts and UB factors
    int64_t hLoopCntCur_{0};
    int64_t sLoopCntCur_{0};
    int64_t ubMainHCur_{0};
    int64_t ubTailHCur_{0};
    int64_t ubMainSCur_{0};
    int64_t ubTailSCur_{0};
    int64_t sMainCntCur_{0};
    int64_t sTailCntCur_{0};
    bool isMainCore_{true};

    // Binary reduction cache for S
    int32_t sCacheLevels_{0};
};

template <typename DT>
__aicore__ inline MaskedCausalConv1dBackward<DT>::MaskedCausalConv1dBackward()
{}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::Init(GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR mask,
                                                            GM_ADDR grad_x, GM_ADDR grad_weight,
                                                            const MaskedCausalConv1dBackwardTilingDataV35 *td,
                                                            TPipe *pipe)
{
    pipe_ = pipe;
    td_ = td;

    hMainCoreCnt_ = td->hMainCoreCnt;
    hTailCoreCnt_ = td->hTailCoreCnt;
    hMainSize_ = td->hMainSize;
    hTailSize_ = td->hTailSize;
    hasMask_ = td->hasMask;
    S_ = td->S;
    B_ = td->B;
    H_ = td->H;

    int64_t blkIdx = GetBlockIdx();
    isMainCore_ = (blkIdx < hMainCoreCnt_);
    if (isMainCore_) {
        hStart_ = blkIdx * hMainSize_;
        hLen_ = hMainSize_;
    } else {
        int64_t tailIdx = blkIdx - hMainCoreCnt_;
        hStart_ = hMainCoreCnt_ * hMainSize_ + tailIdx * hTailSize_;
        hLen_ = hTailSize_;
    }

    hLoopCntCur_ = isMainCore_ ? td->hLoopCnt : td->tailHLoopCnt;
    sLoopCntCur_ = isMainCore_ ? td->sLoopCnt : td->tailSLoopCnt;
    ubMainHCur_ = isMainCore_ ? td->ubMainFactorH : td->tailCoreUbMainFactorH;
    ubTailHCur_ = isMainCore_ ? td->ubTailFactorH : td->tailCoreUbTailFactorH;
    ubMainSCur_ = isMainCore_ ? td->ubMainFactorS : td->tailCoreUbMainFactorS;
    ubTailSCur_ = isMainCore_ ? td->ubTailFactorS : td->tailCoreUbTailFactorS;
    sMainCntCur_ = td->ubMainFactorSCount;
    sTailCntCur_ = td->ubTailFactorSCount;

    gradYGm_.SetGlobalBuffer((__gm__ DT *)grad_y, S_ * B_ * H_);
    xGm_.SetGlobalBuffer((__gm__ DT *)x, S_ * B_ * H_);
    weightGm_.SetGlobalBuffer((__gm__ DT *)weight, kW * H_);
    gradInGm_.SetGlobalBuffer((__gm__ DT *)grad_x, S_ * B_ * H_);
    gradWeightGm_.SetGlobalBuffer((__gm__ DT *)grad_weight, kW * H_);
    if (hasMask_) {
        maskGm_.SetGlobalBuffer((__gm__ bool *)mask, B_ * S_);
    }

    pipe_->InitBuffer(gradOutQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * (ubMainSCur_ + NUM_2) * sizeof(DT)));
    pipe_->InitBuffer(xQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * ubMainSCur_ * sizeof(DT)));
    pipe_->InitBuffer(weightQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * kW * sizeof(DT)));
    if (hasMask_) {
        uint32_t sEffMax = static_cast<uint32_t>(ubMainSCur_ + NUM_2);
        uint32_t vecBytes =
            static_cast<uint32_t>(((sEffMax * sizeof(DT) + (kAlignBytes - 1)) / kAlignBytes) * kAlignBytes);
        uint32_t planeBytes = static_cast<uint32_t>(sEffMax * ubMainHCur_ * sizeof(DT));
        pipe_->InitBuffer(maskBuf_, vecBytes + planeBytes);
        uint32_t maskBytes = static_cast<uint32_t>(ubMainSCur_ + NUM_2);
        uint32_t maskBufSize = (maskBytes + (kAlignBytes - 1)) / kAlignBytes * kAlignBytes;
        pipe_->InitBuffer(maskQ_, kBufferNum, maskBufSize);
    }
    pipe_->InitBuffer(gradInQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * ubMainSCur_ * sizeof(DT)));
    pipe_->InitBuffer(gradWeightQ_, kBufferNum, static_cast<uint32_t>(ubMainHCur_ * kW * sizeof(DT)));

    // Layout in bytes: [accBytes | p2Plane | p1Plane | p0Plane | wsBytes | sumBytes]
    uint32_t accBytes = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float));
    uint32_t planeBytesCap = static_cast<uint32_t>(kW * ubMainSCur_ * ubMainHCur_ * sizeof(float));
    sCacheLevels_ = 1; // Binary reduction cache levels: floor(log2(sLoopCntCur_)) + 1
    {
        int64_t t = sLoopCntCur_;
        while (t > 1) {
            ++sCacheLevels_;
            t >>= 1;
        }
    }
    uint32_t wsBytes = static_cast<uint32_t>(kW * static_cast<uint32_t>(sCacheLevels_) * ubMainHCur_ * sizeof(float));
    uint32_t sumBytes = static_cast<uint32_t>(ubMainHCur_ * sizeof(float));
    uint32_t totalBytes = accBytes + planeBytesCap + wsBytes + sumBytes;
    pipe_->InitBuffer(tmpBuf_, totalBytes);

    // Map gwAccF32_ to the accumulator region within tmpBuf_
    uint32_t stride = static_cast<uint32_t>(ubMainHCur_ * B_ * sizeof(float));
    gwAccF32_[0] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(ubMainHCur_ * B_), 0);
    gwAccF32_[1] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(ubMainHCur_ * B_), stride);
    gwAccF32_[2] = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(ubMainHCur_ * B_), 2U * stride);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::Process()
{
    // Accumulate grad_weight in fp32 across all (b,s) tiles per h-tile, then cast+store once per h-tile
    for (int64_t hTile = 0; hTile < hLoopCntCur_; ++hTile) {
        int64_t hLenThis = (hTile == hLoopCntCur_ - 1 && ubTailHCur_ > 0) ? ubTailHCur_ : ubMainHCur_;

        Duplicate(gwAccF32_[0], 0.0f, static_cast<uint32_t>(ubMainHCur_ * B_));
        Duplicate(gwAccF32_[1], 0.0f, static_cast<uint32_t>(ubMainHCur_ * B_));
        Duplicate(gwAccF32_[2], 0.0f, static_cast<uint32_t>(ubMainHCur_ * B_));
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        CopyInWeight(hTile, hLenThis);
        LocalTensor<DT> wLocal = weightQ_.DeQue<DT>();

        for (int64_t bTile = 0; bTile < B_; ++bTile) {
            for (int64_t sTile = 0; sTile < sLoopCntCur_; ++sTile) {
                int64_t sLen = (sTile < sMainCntCur_) ? ubMainSCur_ : ubTailSCur_;
                int64_t sEff = (sTile == sLoopCntCur_ - 1) ? sLen : (sLen + 2);
                CopyInGradY(bTile, sTile, hTile, sEff, hLenThis);
                CopyInX(bTile, sTile, hTile, sLen, hLenThis);
                LocalTensor<DT> goLocal = gradOutQ_.DeQue<DT>();
                LocalTensor<DT> inLocal = xQ_.DeQue<DT>();
                if (hasMask_) {
                    CopyInMask(bTile, sTile, sEff);
                    ApplyMaskToGradY(goLocal, sEff, hLenThis);
                }
                ComputeGradXWeight(goLocal, inLocal, wLocal, sLen, hLenThis);
                ReduceSTile(sTile, sLen, hLenThis);
                gradOutQ_.FreeTensor(goLocal);
                xQ_.FreeTensor(inLocal);
                CopyOutGradX(bTile, sTile, hTile, sLen, hLenThis);
            }
            ReduceWsTile(bTile, hLenThis);
        }
        weightQ_.FreeTensor(wLocal);
        ReduceBTileAndCopyOut(hTile, hLenThis);
    }
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::ComputeGradXWeight(LocalTensor<DT> &goLocal,
                                                                          LocalTensor<DT> &inLocal,
                                                                          LocalTensor<DT> &wLocal, int64_t sLen,
                                                                          int64_t hLenThis)
{
    LocalTensor<DT> giLocal = gradInQ_.AllocTensor<DT>();

    uint32_t planeBytesCap = static_cast<uint32_t>(ubMainSCur_ * ubMainHCur_ * sizeof(float));
    uint32_t baseOff = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float)); // accBytes region size
    LocalTensor<float> p2Ub = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff);
    LocalTensor<float> p1Ub =
        tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff + planeBytesCap);
    LocalTensor<float> p0Ub =
        tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff + 2U * planeBytesCap);

    MaskedCausalConv1dBackwardVF::DoGradXWeight<DT>(goLocal, wLocal, inLocal, giLocal, p0Ub, p1Ub, p2Ub,
                                                    static_cast<uint32_t>(sLen), static_cast<uint32_t>(hLenThis));
    gradInQ_.EnQue<DT>(giLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::ReduceSTile(int64_t sTile, int64_t sLen, int64_t hLenThis)
{
    uint32_t planeBytesCap = static_cast<uint32_t>(ubMainSCur_ * ubMainHCur_ * sizeof(float));
    uint32_t baseOff = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float)); // accBytes region size
    uint32_t wsBase = baseOff + kW * planeBytesCap;                                  // start of Ws region
    uint32_t wsStrideBytes = static_cast<uint32_t>(hLenThis * sizeof(float));
    uint32_t wsBytesThisHTile =
        static_cast<uint32_t>(kW * static_cast<uint32_t>(sCacheLevels_) * hLenThis * sizeof(float));
    uint32_t sumOff = wsBase + wsBytesThisHTile; // sum buffer placed after Ws region
    LocalTensor<float> sumUb = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hLenThis), sumOff);

    uint32_t srcShape[2] = {static_cast<uint32_t>(sLen), static_cast<uint32_t>(hLenThis)};
    LocalTensor<float> p2Ub = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff);
    LocalTensor<float> p1Ub =
        tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff + planeBytesCap);
    LocalTensor<float> p0Ub =
        tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(sLen * hLenThis), baseOff + 2U * planeBytesCap);

    // k=2
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, p2Ub, srcShape, false);
    {
        int64_t level = 0;
        while (level < sCacheLevels_ - 1 && (static_cast<int64_t>(sTile) & (1ULL << level))) {
            LocalTensor<float> ws2_lvl = tmpBuf_.GetWithOffset<float>(
                static_cast<uint32_t>(hLenThis),
                static_cast<uint32_t>(wsBase +
                                      (0U * static_cast<uint32_t>(sCacheLevels_) + static_cast<uint32_t>(level)) *
                                          wsStrideBytes));
            Add(sumUb, sumUb, ws2_lvl, static_cast<uint32_t>(hLenThis));
            ++level;
        }
        LocalTensor<float> ws2_lvl_dst = tmpBuf_.GetWithOffset<float>(
            static_cast<uint32_t>(hLenThis),
            static_cast<uint32_t>(wsBase + (0U * static_cast<uint32_t>(sCacheLevels_) + static_cast<uint32_t>(level)) *
                                               wsStrideBytes));
        DataCopy(ws2_lvl_dst, sumUb, static_cast<uint32_t>(hLenThis));
    }

    // k=1
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, p1Ub, srcShape, false);
    {
        int64_t level = 0;
        while (level < sCacheLevels_ - 1 && (static_cast<int64_t>(sTile) & (1ULL << level))) {
            LocalTensor<float> ws1_lvl = tmpBuf_.GetWithOffset<float>(
                static_cast<uint32_t>(hLenThis),
                static_cast<uint32_t>(wsBase +
                                      (1U * static_cast<uint32_t>(sCacheLevels_) + static_cast<uint32_t>(level)) *
                                          wsStrideBytes));
            Add(sumUb, sumUb, ws1_lvl, static_cast<uint32_t>(hLenThis));
            ++level;
        }
        LocalTensor<float> ws1_lvl_dst = tmpBuf_.GetWithOffset<float>(
            static_cast<uint32_t>(hLenThis),
            static_cast<uint32_t>(wsBase + (1U * static_cast<uint32_t>(sCacheLevels_) + static_cast<uint32_t>(level)) *
                                               wsStrideBytes));
        DataCopy(ws1_lvl_dst, sumUb, static_cast<uint32_t>(hLenThis));
    }

    // k=0
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, p0Ub, srcShape, false);
    {
        int64_t level = 0;
        while (level < sCacheLevels_ - 1 && (static_cast<int64_t>(sTile) & (1ULL << level))) {
            LocalTensor<float> ws0_lvl = tmpBuf_.GetWithOffset<float>(
                static_cast<uint32_t>(hLenThis),
                static_cast<uint32_t>(wsBase +
                                      (2U * static_cast<uint32_t>(sCacheLevels_) + static_cast<uint32_t>(level)) *
                                          wsStrideBytes));
            Add(sumUb, sumUb, ws0_lvl, static_cast<uint32_t>(hLenThis));
            ++level;
        }
        LocalTensor<float> ws0_lvl_dst = tmpBuf_.GetWithOffset<float>(
            static_cast<uint32_t>(hLenThis),
            static_cast<uint32_t>(wsBase + (2U * static_cast<uint32_t>(sCacheLevels_) + static_cast<uint32_t>(level)) *
                                               wsStrideBytes));
        DataCopy(ws0_lvl_dst, sumUb, static_cast<uint32_t>(hLenThis));
    }
    PipeBarrier<PIPE_V>();
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::ReduceWsTile(int64_t bTile, int64_t hLenThis)
{
    uint32_t planeBytesCap = static_cast<uint32_t>(ubMainSCur_ * ubMainHCur_ * sizeof(float));
    uint32_t baseOff = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float));
    uint32_t wsBase = baseOff + kW * planeBytesCap;
    uint32_t wsStrideBytes = static_cast<uint32_t>(hLenThis * sizeof(float));
    uint32_t wsBytesThisHTile =
        static_cast<uint32_t>(kW * static_cast<uint32_t>(sCacheLevels_) * hLenThis * sizeof(float));
    uint32_t sumOff = wsBase + wsBytesThisHTile;
    LocalTensor<float> sumUb = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hLenThis), sumOff);

    // Accumulate from low level to high level using AccumulateLevels()
    // k=2
    AccumulateLevels(0U, hLenThis, wsBase, wsStrideBytes, sumUb);
    LocalTensor<float> acc2Slice = gwAccF32_[2][static_cast<uint32_t>(bTile * ubMainHCur_)];
    DataCopy(acc2Slice, sumUb, static_cast<uint32_t>(hLenThis));
    // k=1
    AccumulateLevels(1U, hLenThis, wsBase, wsStrideBytes, sumUb);
    LocalTensor<float> acc1Slice = gwAccF32_[1][static_cast<uint32_t>(bTile * ubMainHCur_)];
    DataCopy(acc1Slice, sumUb, static_cast<uint32_t>(hLenThis));
    // k=0
    AccumulateLevels(2U, hLenThis, wsBase, wsStrideBytes, sumUb);
    LocalTensor<float> acc0Slice = gwAccF32_[0][static_cast<uint32_t>(bTile * ubMainHCur_)];
    DataCopy(acc0Slice, sumUb, static_cast<uint32_t>(hLenThis));
    PipeBarrier<PIPE_V>();
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::AccumulateLevels(uint32_t kIdx, int64_t hLenThis,
                                                                        uint32_t wsBase, uint32_t wsStrideBytes,
                                                                        LocalTensor<float> &sumUb)
{
    Duplicate(sumUb, 0.0f, static_cast<uint32_t>(hLenThis));
    PipeBarrier<PIPE_V>();
    for (uint32_t lvl = 0U; lvl < static_cast<uint32_t>(sCacheLevels_); ++lvl) {
        if (((static_cast<uint32_t>(sLoopCntCur_) >> lvl) & 1U) == 0U) {
            continue;
        }
        LocalTensor<float> wsLvl = tmpBuf_.GetWithOffset<float>(
            static_cast<uint32_t>(hLenThis),
            static_cast<uint32_t>(wsBase + (kIdx * static_cast<uint32_t>(sCacheLevels_) + lvl) * wsStrideBytes));
        Add(sumUb, sumUb, wsLvl, static_cast<uint32_t>(hLenThis));
        PipeBarrier<PIPE_V>();
    }
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::ApplyMaskToGradY(LocalTensor<DT> &goLocal, int64_t sEff,
                                                                        int64_t hLenThis)
{
    LocalTensor<bool> mRawBool = maskQ_.DeQue<bool>();
    LocalTensor<uint8_t> mRaw = mRawBool.template ReinterpretCast<uint8_t>();
    uint32_t sEffU = static_cast<uint32_t>(sEff);
    uint32_t hU = static_cast<uint32_t>(hLenThis);
    uint32_t vecBytesAligned =
        static_cast<uint32_t>(((sEffU * sizeof(DT) + (kAlignBytes - 1)) / kAlignBytes) * kAlignBytes);

    // Regions in maskBuf_:
    // [0, vecBytesAligned): vector (DT)
    // [vecBytesAligned, vecBytesAligned + planeBytes): plane (DT)
    LocalTensor<DT> maskVecDT = maskBuf_.Get<DT>();
    LocalTensor<DT> maskPlaneDT = maskBuf_.GetWithOffset<DT>(static_cast<uint32_t>(sEffU * hU), vecBytesAligned);

    // Step1: u8 -> half
    LocalTensor<half> maskHalfVec = maskBuf_.Get<half>();
    Cast(maskHalfVec, mRaw, RoundMode::CAST_NONE, sEffU);
    PipeBarrier<PIPE_V>();

    // Step2: half -> DT (bf16 prioritized), then Broadcast
    if constexpr (std::is_same<DT, half>::value) {
        // half -> half copy via Adds
        Adds(maskVecDT.template ReinterpretCast<half>(), maskHalfVec, static_cast<half>(0), sEffU);
        PipeBarrier<PIPE_V>();
        uint32_t srcShape[2] = {sEffU, 1U};
        uint32_t dstShape[2] = {sEffU, hU};
        Broadcast<half, 2, 1>(maskPlaneDT.template ReinterpretCast<half>(), maskVecDT.template ReinterpretCast<half>(),
                              dstShape, srcShape);
    } else {
        // bf16 path
        Cast(maskVecDT, maskHalfVec, RoundMode::CAST_RINT, sEffU);
        PipeBarrier<PIPE_V>();
        uint32_t srcShape[2] = {sEffU, 1U};
        uint32_t dstShape[2] = {sEffU, hU};
        Broadcast<DT, 2, 1>(maskPlaneDT, maskVecDT, dstShape, srcShape);
    }
    PipeBarrier<PIPE_V>();

    Mul(goLocal, goLocal, maskPlaneDT, static_cast<uint32_t>(sEffU * hU));
    PipeBarrier<PIPE_V>();

    maskQ_.FreeTensor(mRawBool);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::ReduceBTileAndCopyOut(int64_t hTile, int64_t hLenThis)
{
    // Reduce across B in fp32 using ReduceSum, cast to DT, then write to GM
    int64_t hOff = hTile * ubMainHCur_;
    int64_t baseH = hStart_ + hOff;

    LocalTensor<DT> gwCast = gradWeightQ_.AllocTensor<DT>();
    LocalTensor<DT> out0 = gwCast;                                         // k=0 slice at offset 0
    LocalTensor<DT> out1 = gwCast[static_cast<uint32_t>(ubMainHCur_)];     // k=1 slice
    LocalTensor<DT> out2 = gwCast[static_cast<uint32_t>(2 * ubMainHCur_)]; // k=2 slice

    uint32_t accBytes = static_cast<uint32_t>(kW * ubMainHCur_ * B_ * sizeof(float));
    uint32_t planeBytesCap = static_cast<uint32_t>(ubMainSCur_ * ubMainHCur_ * sizeof(float));
    uint32_t sumOff = accBytes + kW * planeBytesCap;
    LocalTensor<float> sumUb = tmpBuf_.GetWithOffset<float>(static_cast<uint32_t>(hLenThis), sumOff);
    uint32_t srcShape[2] = {static_cast<uint32_t>(B_), static_cast<uint32_t>(hLenThis)};
    // k=0
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, gwAccF32_[0], srcShape, false);
    PipeBarrier<PIPE_V>();
    MaskedCausalConv1dBackwardVF::DoCastToDT<DT>(sumUb, static_cast<uint32_t>(hLenThis), out0);
    // k=1
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, gwAccF32_[1], srcShape, false);
    PipeBarrier<PIPE_V>();
    MaskedCausalConv1dBackwardVF::DoCastToDT<DT>(sumUb, static_cast<uint32_t>(hLenThis), out1);
    // k=2
    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumUb, gwAccF32_[2], srcShape, false);
    PipeBarrier<PIPE_V>();
    MaskedCausalConv1dBackwardVF::DoCastToDT<DT>(sumUb, static_cast<uint32_t>(hLenThis), out2);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    DataCopyExtParams outParams{static_cast<uint16_t>(kW), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                                static_cast<uint32_t>((ubMainHCur_ - hLenThis) * sizeof(DT)),
                                static_cast<uint32_t>((H_ - hLenThis) * sizeof(DT)), 0};
    DataCopyPad(gradWeightGm_[baseH], gwCast, outParams);

    gradWeightQ_.FreeTensor(gwCast);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::CopyInGradY(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                   int64_t sEff, int64_t hLenThis)
{
    LocalTensor<DT> goLocal = gradOutQ_.AllocTensor<DT>();

    int64_t sStart = (sTile < sMainCntCur_) ?
                         (sTile * ubMainSCur_) :
                         (sMainCntCur_ * ubMainSCur_ + (sTile - sMainCntCur_) * ubTailSCur_); // piecewise start
    int64_t bStart = bTile;
    int64_t hOff = hTile * ubMainHCur_;
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    LocalTensor<DT> dstBlock = goLocal[0];
    DataCopyExtParams inParams{static_cast<uint16_t>(sEff), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ * B_ - hLenThis) * sizeof(DT)), 0, 0};

    DataCopyPad(dstBlock, gradYGm_[base], inParams, padParams);

    // For last S-tile, zero the remaining 2 halo rows in buffer (batch clear)
    if (sTile == sLoopCntCur_ - 1) {
        uint32_t tailOff = static_cast<uint32_t>(sEff * hLenThis);
        LocalTensor<DT> tailRows = goLocal[tailOff];
        Duplicate(tailRows, static_cast<DT>(0), static_cast<uint32_t>(2 * hLenThis));
        PipeBarrier<PIPE_V>();
    }

    gradOutQ_.EnQue<DT>(goLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::CopyInX(int64_t bTile, int64_t sTile, int64_t hTile,
                                                               int64_t sLen, int64_t hLenThis)
{
    LocalTensor<DT> inLocal = xQ_.AllocTensor<DT>();

    int64_t sStart = (sTile < sMainCntCur_) ? (sTile * ubMainSCur_) :
                                              (sMainCntCur_ * ubMainSCur_ + (sTile - sMainCntCur_) * ubTailSCur_);
    int64_t bStart = bTile;
    int64_t hOff = hTile * ubMainHCur_;
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    LocalTensor<DT> dstBlock = inLocal[0];
    DataCopyExtParams inParams{static_cast<uint16_t>(sLen), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ * B_ - hLenThis) * sizeof(DT)), 0, 0};
    DataCopyPad(dstBlock, xGm_[base], inParams, padParams);

    xQ_.EnQue<DT>(inLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::CopyInWeight(int64_t hTile, int64_t hLenThis)
{
    LocalTensor<DT> wLocal = weightQ_.AllocTensor<DT>();
    DataCopyExtParams inParams{static_cast<uint16_t>(kW), static_cast<uint32_t>(hLenThis * sizeof(DT)),
                               static_cast<uint32_t>((H_ - hLenThis) * sizeof(DT)), 0, 0};
    int64_t hOff = hTile * ubMainHCur_;
    int64_t base = (hStart_ + hOff);
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
    DataCopyPad(wLocal, weightGm_[base], inParams, padParams);
    weightQ_.EnQue<DT>(wLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::CopyInMask(int64_t bTile, int64_t sTile, int64_t sEff)
{
    LocalTensor<bool> mLocal = maskQ_.AllocTensor<bool>();
    DataCopyExtParams inParams{static_cast<uint16_t>(1), static_cast<uint32_t>(sEff), static_cast<uint32_t>(S_ - sEff),
                               0, 0};
    int64_t sStart = (sTile < sMainCntCur_) ? (sTile * ubMainSCur_) :
                                              (sMainCntCur_ * ubMainSCur_ + (sTile - sMainCntCur_) * ubTailSCur_);
    int64_t bStart = bTile;
    int64_t base = bStart * S_ + sStart;
    DataCopyPadExtParams<bool> padParams{false, 0, 0, 0};
    DataCopyPad(mLocal, maskGm_[base], inParams, padParams);
    maskQ_.EnQue<bool>(mLocal);
}

template <typename DT>
__aicore__ inline void MaskedCausalConv1dBackward<DT>::CopyOutGradX(int64_t bTile, int64_t sTile, int64_t hTile,
                                                                    int64_t sLen, int64_t hLenThis)
{
    LocalTensor<DT> giLocal = gradInQ_.DeQue<DT>();

    int64_t sStart = (sTile < sMainCntCur_) ? (sTile * ubMainSCur_) :
                                              (sMainCntCur_ * ubMainSCur_ + (sTile - sMainCntCur_) * ubTailSCur_);
    int64_t bStart = bTile;
    int64_t hOff = hTile * ubMainHCur_;

    int64_t base = ((sStart * B_ + bStart) * H_) + (hStart_ + hOff);
    LocalTensor<DT> srcBlock = giLocal[0];
    DataCopyExtParams outParams{static_cast<uint16_t>(sLen), static_cast<uint32_t>(hLenThis * sizeof(DT)), 0,
                                static_cast<uint32_t>((H_ * B_ - hLenThis) * sizeof(DT)), 0};
    DataCopyPad(gradInGm_[base], srcBlock, outParams);
    gradInQ_.FreeTensor(giLocal);
}

} // namespace MaskedCausalConv1dBackwardKernelNS

#endif // MASKED_CAUSAL_CONV1D_BACKWARD_ARCH35_H
