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
 * \file block_attention_residuals_grad_regbase.h
 * \brief block_attention_residuals_grad A5 (ascend950) regbase kernel.
 *
 * Algorithm follows the membase implementation: per token, compute softmax
 * backward scores over N+1 rows, then write grad_v rows and accumulate the
 * grad score-weight partial sum with Kahan summation. The partial sums are
 * reduced across cores in the workspace and core 0 writes grad_proj_weight /
 * grad_norm_weight.
 */
#ifndef BLOCK_ATTENTION_RESIDUALS_GRAD_REGBASE_H
#define BLOCK_ATTENTION_RESIDUALS_GRAD_REGBASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../block_attention_residuals_grad_tiling_data.h"
#include "block_attention_residuals_grad_regbase_common.h"

namespace NsBlockAttentionResidualsGrad {
using namespace AscendC;

constexpr uint32_t GRAD_BUFFER_NUM = 2;
constexpr uint32_t GRAD_ELEM_PER_BLK_BF16 = 16;
constexpr uint32_t GRAD_SCALAR_SLOTS = 6;

template <typename D_IN>
class BlockAttentionResidualsGradRegbase {
public:
    __aicore__ inline BlockAttentionResidualsGradRegbase(TPipe *pipe,
                                                         const BlockAttentionResidualsGradTilingData *tiling)
        : pipe_(pipe),
          tiling_(tiling)
    {}

    __aicore__ inline void Init(GM_ADDR prefixSum, GM_ADDR blockResidual, GM_ADDR projWeight, GM_ADDR normWeight,
                                GM_ADDR gradOutput, GM_ADDR invRms, GM_ADDR probs, GM_ADDR gradPrefixSum,
                                GM_ADDR gradBlockResidual, GM_ADDR gradProjWeight, GM_ADDR gradNormWeight,
                                GM_ADDR workspace);

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitBuffers();
    __aicore__ inline void LoadScoreWeight();
    __aicore__ inline void CopyInRowEnqueue(const GlobalTensor<D_IN> &srcGm, uint64_t srcOffset);
    __aicore__ inline LocalTensor<D_IN> CopyInRowSync(const GlobalTensor<D_IN> &srcGm, uint64_t srcOffset);
    __aicore__ inline const GlobalTensor<D_IN> &GetLogicRowGm(uint32_t n) const;
    __aicore__ inline uint64_t GetLogicRowOffset(uint32_t tokenIdx, uint32_t n) const;
    __aicore__ inline void LoadToken(uint32_t tokenIdx);
    __aicore__ inline void ComputeGradProbs(uint32_t tokenIdx);
    __aicore__ inline void SoftmaxBackward();
    __aicore__ inline void ComputeBlockGrad(uint32_t blk, uint32_t tokenIdx);
    __aicore__ inline void ProcessOneToken(uint32_t tokenIdx);
    __aicore__ inline void WriteGradScoreWeight();
    __aicore__ inline void ReduceAndWriteWeights();

    TPipe *pipe_{nullptr};
    const BlockAttentionResidualsGradTilingData *tiling_{nullptr};

    GlobalTensor<D_IN> prefixSumGm_;
    GlobalTensor<D_IN> blockResidualGm_;
    GlobalTensor<D_IN> projWeightGm_;
    GlobalTensor<D_IN> normWeightGm_;
    GlobalTensor<D_IN> gradOutGm_;
    GlobalTensor<float> invRmsGm_;
    GlobalTensor<float> probsGm_;
    GlobalTensor<D_IN> gradPrefixSumGm_;
    GlobalTensor<D_IN> gradBlockResGm_;
    GlobalTensor<D_IN> gradProjWeightGm_;
    GlobalTensor<D_IN> gradNormWeightGm_;
    GlobalTensor<float> gswWkspGm_;
    GlobalTensor<float> wkspAllGm_;

    TQue<QuePosition::VECIN, GRAD_BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TQue<QuePosition::VECIN, 1> gradOutQue_;
    TQue<QuePosition::VECIN, 1> probsInQue_;
    TQue<QuePosition::VECIN, 1> invRmsInQue_;
    TBuf<TPosition::VECCALC> scoreWeightBuf_;
    TBuf<TPosition::VECCALC> vRowBuf_;
    TBuf<TPosition::VECCALC> outFp32Buf_;
    TBuf<TPosition::VECCALC> gradOutFp32Buf_;
    TBuf<TPosition::VECCALC> gswAccBuf_;
    TBuf<TPosition::VECCALC> gswCmpBuf_;
    TBuf<TPosition::VECCALC> gradProbBuf_;
    TBuf<TPosition::VECCALC> gradScoreBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;

    LocalTensor<float> scoreWeight_;
    LocalTensor<float> vRow_;
    LocalTensor<float> outFp32_;
    LocalTensor<float> gradOutFp32_;
    LocalTensor<float> gswAcc_;
    LocalTensor<float> gswCmp_;
    LocalTensor<float> gradProb_;
    LocalTensor<float> gradScore_;
    LocalTensor<float> scalarLocal_;
    LocalTensor<float> probs_;
    LocalTensor<float> invRms_;
    int32_t gswKahanPos_{0};

    uint32_t batchSize_{0};
    uint32_t numBlocks_{0};
    uint32_t hiddenSize_{0};
    uint32_t hiddenSizeAlignBf16_{0};
    uint32_t hiddenSizeAlignFp32_{0};
    uint32_t blockCount_{0};
    uint32_t metaAlign_{0};
    uint32_t coreNum_{0};
    uint32_t tokenStart_{0};
    uint32_t tokenNum_{0};
    uint32_t blockIdx_{0};
    uint64_t perCoreWkspBytes_{0};
    float invH_{0.0f};
    event_t eventMTE3S_;
    event_t eventMTE2V_;
    event_t eventVMTE2_;
    event_t eventSV_;
    event_t eventVS_;
};

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::InitBuffers()
{
    pipe_->InitBuffer(inQue_, GRAD_BUFFER_NUM, hiddenSizeAlignBf16_ * sizeof(D_IN));
    pipe_->InitBuffer(outQue_, 1, hiddenSizeAlignBf16_ * sizeof(D_IN));
    pipe_->InitBuffer(gradOutQue_, 1, hiddenSizeAlignBf16_ * sizeof(D_IN));
    pipe_->InitBuffer(probsInQue_, 1, metaAlign_ * sizeof(float));
    pipe_->InitBuffer(invRmsInQue_, 1, metaAlign_ * sizeof(float));
    pipe_->InitBuffer(scoreWeightBuf_, hiddenSizeAlignFp32_ * sizeof(float));
    pipe_->InitBuffer(vRowBuf_, hiddenSizeAlignFp32_ * sizeof(float));
    pipe_->InitBuffer(outFp32Buf_, hiddenSizeAlignFp32_ * sizeof(float));
    pipe_->InitBuffer(gradOutFp32Buf_, hiddenSizeAlignFp32_ * sizeof(float));
    pipe_->InitBuffer(gswAccBuf_, hiddenSizeAlignFp32_ * sizeof(float));
    pipe_->InitBuffer(gswCmpBuf_, hiddenSizeAlignFp32_ * sizeof(float));
    pipe_->InitBuffer(gradProbBuf_, metaAlign_ * sizeof(float));
    pipe_->InitBuffer(gradScoreBuf_, metaAlign_ * sizeof(float));
    pipe_->InitBuffer(scalarBuf_, GRAD_SCALAR_SLOTS * RegBase::GRAD_SCALAR_LOCAL_ELEMS * sizeof(float));

    scoreWeight_ = scoreWeightBuf_.Get<float>();
    vRow_ = vRowBuf_.Get<float>();
    outFp32_ = outFp32Buf_.Get<float>();
    gradOutFp32_ = gradOutFp32Buf_.Get<float>();
    gswAcc_ = gswAccBuf_.Get<float>();
    gswCmp_ = gswCmpBuf_.Get<float>();
    gradProb_ = gradProbBuf_.Get<float>();
    gradScore_ = gradScoreBuf_.Get<float>();
    scalarLocal_ = scalarBuf_.Get<float>();
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::Init(GM_ADDR prefixSum, GM_ADDR blockResidual,
                                                                      GM_ADDR projWeight, GM_ADDR normWeight,
                                                                      GM_ADDR gradOutput, GM_ADDR invRms, GM_ADDR probs,
                                                                      GM_ADDR gradPrefixSum, GM_ADDR gradBlockResidual,
                                                                      GM_ADDR gradProjWeight, GM_ADDR gradNormWeight,
                                                                      GM_ADDR workspace)
{
    batchSize_ = static_cast<uint32_t>(tiling_->batchSize);
    numBlocks_ = static_cast<uint32_t>(tiling_->numBlocks);
    blockCount_ = static_cast<uint32_t>(tiling_->totalBlocks);
    hiddenSize_ = static_cast<uint32_t>(tiling_->hiddenSize);
    coreNum_ = static_cast<uint32_t>(tiling_->coreNum);
    perCoreWkspBytes_ = tiling_->perCoreWkspBytes;
    invH_ = 1.0f / static_cast<float>(hiddenSize_);
    blockIdx_ = GetBlockIdx();
    if (coreNum_ == 0U) {
        return;
    }

    uint32_t perCore = batchSize_ / coreNum_;
    uint32_t rem = batchSize_ % coreNum_;
    if (blockIdx_ < rem) {
        tokenStart_ = blockIdx_ * (perCore + 1U);
        tokenNum_ = perCore + 1U;
    } else {
        tokenStart_ = rem * (perCore + 1U) + (blockIdx_ - rem) * perCore;
        tokenNum_ = perCore;
    }

    // hiddenSize_向上对齐到16的倍数
    hiddenSizeAlignBf16_ =
        (hiddenSize_ + GRAD_ELEM_PER_BLK_BF16 - 1U) / GRAD_ELEM_PER_BLK_BF16 * GRAD_ELEM_PER_BLK_BF16;
    hiddenSizeAlignFp32_ = RegBase::CeilDivU32(hiddenSize_, RegBase::kVlFp32) * RegBase::kVlFp32;
    metaAlign_ = RegBase::CeilDivU32(blockCount_, RegBase::kVlFp32) * RegBase::kVlFp32;

    prefixSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(prefixSum));
    blockResidualGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(blockResidual));
    projWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(projWeight));
    normWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(normWeight));
    gradOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(gradOutput));
    invRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(invRms));
    probsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(probs));
    gradPrefixSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(gradPrefixSum));
    gradBlockResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(gradBlockResidual));
    gradProjWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(gradProjWeight));
    gradNormWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(gradNormWeight));

    __gm__ float *userWksp = reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace));
    uint64_t wkspOff = perCoreWkspBytes_ * static_cast<uint64_t>(blockIdx_);
    gswWkspGm_.SetGlobalBuffer(userWksp + wkspOff / sizeof(float), hiddenSize_);
    wkspAllGm_.SetGlobalBuffer(userWksp, perCoreWkspBytes_ * static_cast<uint64_t>(coreNum_) / sizeof(float));

    InitBuffers();
    LoadScoreWeight();
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::LoadScoreWeight()
{
    LocalTensor<D_IN> inLocal = CopyInRowSync(projWeightGm_, 0);
    if constexpr (IsSameType<D_IN, float>::value) {
        AscendC::Adds(vRow_, inLocal, 0.0f, hiddenSize_);
    } else {
        AscendC::Cast(vRow_, inLocal, RoundMode::CAST_NONE, hiddenSize_);
    }
    inQue_.FreeTensor(inLocal);
    PipeBarrier<PIPE_V>();

    inLocal = CopyInRowSync(normWeightGm_, 0);
    if constexpr (IsSameType<D_IN, float>::value) {
        AscendC::Adds(outFp32_, inLocal, 0.0f, hiddenSize_);
    } else {
        AscendC::Cast(outFp32_, inLocal, RoundMode::CAST_NONE, hiddenSize_);
    }
    inQue_.FreeTensor(inLocal);
    PipeBarrier<PIPE_V>();

    AscendC::Mul(scoreWeight_, vRow_, outFp32_, hiddenSize_);
    PipeBarrier<PIPE_V>();
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::CopyInRowEnqueue(const GlobalTensor<D_IN> &srcGm,
                                                                                  uint64_t srcOffset)
{
    LocalTensor<D_IN> inLocal = inQue_.AllocTensor<D_IN>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
    DataCopyPadExtParams<D_IN> padParams{false, 0, 0, 0};
    DataCopyPad(inLocal, srcGm[srcOffset], copyParams, padParams);
    inQue_.EnQue(inLocal);
}

template <typename D_IN>
__aicore__ inline LocalTensor<D_IN> BlockAttentionResidualsGradRegbase<D_IN>::CopyInRowSync(
    const GlobalTensor<D_IN> &srcGm, uint64_t srcOffset)
{
    LocalTensor<D_IN> inLocal = inQue_.AllocTensor<D_IN>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
    DataCopyPadExtParams<D_IN> padParams{false, 0, 0, 0};
    DataCopyPad(inLocal, srcGm[srcOffset], copyParams, padParams);
    inQue_.EnQue(inLocal);
    return inQue_.DeQue<D_IN>();
}

template <typename D_IN>
__aicore__ inline const GlobalTensor<D_IN> &BlockAttentionResidualsGradRegbase<D_IN>::GetLogicRowGm(uint32_t n) const
{
    return (n < numBlocks_) ? blockResidualGm_ : prefixSumGm_;
}

template <typename D_IN>
__aicore__ inline uint64_t BlockAttentionResidualsGradRegbase<D_IN>::GetLogicRowOffset(uint32_t tokenIdx,
                                                                                       uint32_t n) const
{
    if (n < numBlocks_) {
        const uint64_t blockBase = static_cast<uint64_t>(tokenIdx) * numBlocks_ * hiddenSize_;
        return blockBase + static_cast<uint64_t>(n) * hiddenSize_;
    }
    return static_cast<uint64_t>(tokenIdx) * hiddenSize_;
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::LoadToken(uint32_t tokenIdx)
{
    LocalTensor<D_IN> goLocal = gradOutQue_.AllocTensor<D_IN>();
    DataCopyExtParams goParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
    DataCopyPadExtParams<D_IN> padParams{false, 0, 0, 0};
    DataCopyPad(goLocal, gradOutGm_[static_cast<uint64_t>(tokenIdx) * hiddenSize_], goParams, padParams);
    gradOutQue_.EnQue(goLocal);
    goLocal = gradOutQue_.DeQue<D_IN>();
    if constexpr (IsSameType<D_IN, float>::value) {
        AscendC::Adds(gradOutFp32_, goLocal, 0.0f, hiddenSize_);
    } else {
        AscendC::Cast(gradOutFp32_, goLocal, RoundMode::CAST_NONE, hiddenSize_);
    }
    gradOutQue_.FreeTensor(goLocal);

    DataCopyExtParams n1Params{1, static_cast<uint32_t>(blockCount_ * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padRight{true, 0, static_cast<uint8_t>(RegBase::RoundUpFp32(blockCount_) - blockCount_),
                                         0};
    LocalTensor<float> pLocal = probsInQue_.AllocTensor<float>();
    DataCopyPad(pLocal, probsGm_[static_cast<uint64_t>(tokenIdx) * blockCount_], n1Params, padRight);
    probsInQue_.EnQue(pLocal);
    probs_ = probsInQue_.DeQue<float>();

    LocalTensor<float> irLocal = invRmsInQue_.AllocTensor<float>();
    DataCopyPad(irLocal, invRmsGm_[static_cast<uint64_t>(tokenIdx) * blockCount_], n1Params, padRight);
    invRmsInQue_.EnQue(irLocal);
    invRms_ = invRmsInQue_.DeQue<float>();
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::ComputeGradProbs(uint32_t tokenIdx)
{
    Duplicate(gradProb_, 0.0f, metaAlign_);
    PipeBarrier<PIPE_V>();
    if (blockCount_ == 0U) {
        return;
    }
    CopyInRowEnqueue(GetLogicRowGm(0), GetLogicRowOffset(tokenIdx, 0));
    for (uint32_t n = 0; n < blockCount_; ++n) {
        LocalTensor<D_IN> inLocal = inQue_.DeQue<D_IN>();
        if constexpr (IsSameType<D_IN, float>::value) {
            AscendC::Adds(vRow_, inLocal, 0.0f, hiddenSize_);
        } else {
            AscendC::Cast(vRow_, inLocal, RoundMode::CAST_NONE, hiddenSize_);
        }
        inQue_.FreeTensor(inLocal);
        if (n + 1U < blockCount_) {
            CopyInRowEnqueue(GetLogicRowGm(n + 1U), GetLogicRowOffset(tokenIdx, n + 1U));
        }
        PipeBarrier<PIPE_V>();
        AscendC::Mul(outFp32_, vRow_, gradOutFp32_, hiddenSize_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(eventVS_);
        WaitFlag<HardEvent::V_S>(eventVS_);
        RegBase::ReduceMulSumTree(scalarLocal_, outFp32_, hiddenSize_, eventSV_, eventVS_);
        PipeBarrier<PIPE_V>();
        RegBase::CopyScalarToDense(gradProb_[n], scalarLocal_);
        PipeBarrier<PIPE_V>();
    }
    // gradProb 现在由标量单元写入，SoftmaxBackward 的向量 Mul 读取前必须等标量写完成。
    SetFlag<HardEvent::S_V>(eventSV_);
    WaitFlag<HardEvent::S_V>(eventSV_);
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::SoftmaxBackward()
{
    AscendC::Mul(gradScore_, probs_, gradProb_, blockCount_);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(eventVS_);
    WaitFlag<HardEvent::V_S>(eventVS_);
    RegBase::ReduceMulSumTree(scalarLocal_, gradScore_, blockCount_, eventSV_, eventVS_);
    PipeBarrier<PIPE_V>();
    RegBase::SoftmaxBackwardVf(gradScore_, probs_, gradProb_, scalarLocal_, blockCount_);
    PipeBarrier<PIPE_V>();
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::ComputeBlockGrad(uint32_t blk, uint32_t tokenIdx)
{
    LocalTensor<D_IN> inLocal = CopyInRowSync(GetLogicRowGm(blk), GetLogicRowOffset(tokenIdx, blk));
    if constexpr (IsSameType<D_IN, float>::value) {
        AscendC::Adds(vRow_, inLocal, 0.0f, hiddenSize_);
    } else {
        AscendC::Cast(vRow_, inLocal, RoundMode::CAST_NONE, hiddenSize_);
    }
    inQue_.FreeTensor(inLocal);

    SetFlag<HardEvent::V_S>(eventVS_);
    WaitFlag<HardEvent::V_S>(eventVS_);

    // 将标量值保存到 scalarLocal 槽位
    RegBase::CopyScalarToDense(scalarLocal_[1U * RegBase::GRAD_ELEM_PER_BLK_FP32], probs_[blk]);
    RegBase::CopyScalarToDense(scalarLocal_[2U * RegBase::GRAD_ELEM_PER_BLK_FP32], gradScore_[blk]);
    RegBase::CopyScalarToDense(scalarLocal_[3U * RegBase::GRAD_ELEM_PER_BLK_FP32], invRms_[blk]);
    PipeBarrier<PIPE_V>();

    AscendC::Mul(outFp32_, scoreWeight_, vRow_, hiddenSize_);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(eventVS_);
    WaitFlag<HardEvent::V_S>(eventVS_);
    RegBase::ReduceMulSumTree(scalarLocal_, outFp32_, hiddenSize_, eventSV_, eventVS_);
    PipeBarrier<PIPE_V>();

    // sc = gs * inv_rms
    Mul(scalarLocal_[4U * RegBase::GRAD_ELEM_PER_BLK_FP32], scalarLocal_[2U * RegBase::GRAD_ELEM_PER_BLK_FP32],
        scalarLocal_[3U * RegBase::GRAD_ELEM_PER_BLK_FP32], 1);
    // grad_inv_rms = gs * sum(score_weight * v)
    Mul(scalarLocal_[2U * RegBase::GRAD_ELEM_PER_BLK_FP32], scalarLocal_[2U * RegBase::GRAD_ELEM_PER_BLK_FP32],
        scalarLocal_, 1);
    // varScale = grad_inv_rms * (-invH) * inv_rms^3
    Mul(scalarLocal_[5U * RegBase::GRAD_ELEM_PER_BLK_FP32], scalarLocal_[3U * RegBase::GRAD_ELEM_PER_BLK_FP32],
        scalarLocal_[3U * RegBase::GRAD_ELEM_PER_BLK_FP32], 1);
    Mul(scalarLocal_[5U * RegBase::GRAD_ELEM_PER_BLK_FP32], scalarLocal_[5U * RegBase::GRAD_ELEM_PER_BLK_FP32],
        scalarLocal_[3U * RegBase::GRAD_ELEM_PER_BLK_FP32], 1);
    Mul(scalarLocal_[5U * RegBase::GRAD_ELEM_PER_BLK_FP32], scalarLocal_[2U * RegBase::GRAD_ELEM_PER_BLK_FP32],
        scalarLocal_[5U * RegBase::GRAD_ELEM_PER_BLK_FP32], 1);
    Muls(scalarLocal_[5U * RegBase::GRAD_ELEM_PER_BLK_FP32], scalarLocal_[5U * RegBase::GRAD_ELEM_PER_BLK_FP32], -invH_,
         1);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::S_V>(eventSV_);
    WaitFlag<HardEvent::S_V>(eventSV_);

    // 融合计算 gradV：out + k + var
    RegBase::FusedGradVF(outFp32_, gradOutFp32_, vRow_, scoreWeight_,
                         scalarLocal_[1U * RegBase::GRAD_ELEM_PER_BLK_FP32], hiddenSize_);
    PipeBarrier<PIPE_V>();

    // gradScoreWeight += (v * invRms) * gradScore，与 arch22/参考同序；
    // Kahan 保持 arch22 的 role-swap KahanSumUpdate。
    AscendC::Muls(vRow_, vRow_, invRms_.GetValue(blk), hiddenSize_);
    AscendC::Muls(vRow_, vRow_, gradScore_.GetValue(blk), hiddenSize_);
    LocalTensor<float> gswSumCmp[2] = {gswAcc_, gswCmp_};
    RegBase::KahanSumUpdate(vRow_, gswSumCmp, static_cast<int32_t>(hiddenSize_), gswKahanPos_);
    PipeBarrier<PIPE_V>();

    LocalTensor<D_IN> outLocal = outQue_.AllocTensor<D_IN>();
    if constexpr (IsSameType<D_IN, half>::value) {
        AscendC::Cast(outLocal, outFp32_, RoundMode::CAST_RINT, hiddenSize_);
    } else if constexpr (IsSameType<D_IN, float>::value) {
        AscendC::Adds(outLocal, outFp32_, 0.0f, hiddenSize_);
    } else {
        AscendC::Cast(outLocal, outFp32_, RoundMode::CAST_RINT, hiddenSize_);
    }
    outQue_.EnQue(outLocal);
    outLocal = outQue_.DeQue<D_IN>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
    if (blk < numBlocks_) {
        const uint64_t gmOffset =
            static_cast<uint64_t>(tokenIdx) * numBlocks_ * hiddenSize_ + static_cast<uint64_t>(blk) * hiddenSize_;
        DataCopyPad(gradBlockResGm_[gmOffset], outLocal, copyParams);
    } else {
        DataCopyPad(gradPrefixSumGm_[static_cast<uint64_t>(tokenIdx) * hiddenSize_], outLocal, copyParams);
    }
    outQue_.FreeTensor(outLocal);
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::ProcessOneToken(uint32_t tokenIdx)
{
    LoadToken(tokenIdx);
    ComputeGradProbs(tokenIdx);
    SoftmaxBackward();
    for (uint32_t blk = 0; blk < blockCount_; ++blk) {
        ComputeBlockGrad(blk, tokenIdx);
    }
    probsInQue_.FreeTensor(probs_);
    invRmsInQue_.FreeTensor(invRms_);
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::WriteGradScoreWeight()
{
    // role-swap Kahan 的最后"和"可能落在 gswAcc_ 或 gswCmp_，按 gswKahanPos_ 写 Workspace。
    PipeBarrier<PIPE_V>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(float)), 0, 0, 0};
    LocalTensor<float> gswResult = gswAcc_;
    if (gswKahanPos_ != 0) {
        gswResult = gswCmp_;
    }
    DataCopyPad(gswWkspGm_, gswResult, copyParams);
    SetFlag<HardEvent::MTE3_S>(eventMTE3S_);
    WaitFlag<HardEvent::MTE3_S>(eventMTE3S_);
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::ReduceAndWriteWeights()
{
    if (blockIdx_ != 0U) {
        return;
    }
    LocalTensor<float> total = gswAcc_;
    LocalTensor<float> totalCmp = gswCmp_;
    Duplicate(total, 0.0f, hiddenSizeAlignFp32_);
    Duplicate(totalCmp, 0.0f, hiddenSizeAlignFp32_);
    PipeBarrier<PIPE_V>();
    int32_t kahanPos = 0;
    LocalTensor<float> sumCmp[2] = {total, totalCmp};
    const uint64_t strideFloats = perCoreWkspBytes_ / sizeof(float);
    DataCopyExtParams cpF{1, static_cast<uint32_t>(hiddenSize_ * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    // 只归约真正分到 token 的核：A5 核数可能大于 batch，空闲核会把全 0 部分和写进
    // Workspace，而 KahanSumUpdate 对全 0 输入不是 no-op，会把补偿折进和并清零补偿，
    // 使 output_2/3 在抵消点差几个 ulp。
    const uint32_t reduceCores = (coreNum_ < batchSize_) ? coreNum_ : batchSize_;
    for (uint32_t c = 0; c < reduceCores; ++c) {
        DataCopyPad(outFp32_, wkspAllGm_[static_cast<uint64_t>(c) * strideFloats], cpF, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
        // 跨核逐核 Kahan 补偿，与 arch22 的 KahanSumUpdate 保持一致。
        RegBase::KahanSumUpdate(outFp32_, sumCmp, static_cast<int32_t>(hiddenSize_), kahanPos);
        SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);
    }
    // 偶数次更新后最终和仍在 total，奇数次在 totalCmp。
    if (kahanPos != 0) {
        Adds(total, totalCmp, 0.0f, hiddenSize_);
    }
    PipeBarrier<PIPE_V>();

    DataCopyExtParams cpT{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
    LocalTensor<D_IN> wLocal = CopyInRowSync(projWeightGm_, 0);
    if constexpr (IsSameType<D_IN, float>::value) {
        AscendC::Adds(vRow_, wLocal, 0.0f, hiddenSize_);
    } else {
        AscendC::Cast(vRow_, wLocal, RoundMode::CAST_NONE, hiddenSize_);
    }
    inQue_.FreeTensor(wLocal);
    PipeBarrier<PIPE_V>();
    AscendC::Mul(outFp32_, total, vRow_, hiddenSize_);
    PipeBarrier<PIPE_V>();
    LocalTensor<D_IN> outLocal = outQue_.AllocTensor<D_IN>();
    if constexpr (IsSameType<D_IN, half>::value) {
        AscendC::Cast(outLocal, outFp32_, RoundMode::CAST_RINT, hiddenSize_);
    } else if constexpr (IsSameType<D_IN, float>::value) {
        AscendC::Adds(outLocal, outFp32_, 0.0f, hiddenSize_);
    } else {
        AscendC::Cast(outLocal, outFp32_, RoundMode::CAST_RINT, hiddenSize_);
    }
    outQue_.EnQue(outLocal);
    outLocal = outQue_.DeQue<D_IN>();
    DataCopyPad(gradNormWeightGm_, outLocal, cpT);
    outQue_.FreeTensor(outLocal);

    wLocal = CopyInRowSync(normWeightGm_, 0);
    if constexpr (IsSameType<D_IN, float>::value) {
        AscendC::Adds(vRow_, wLocal, 0.0f, hiddenSize_);
    } else {
        AscendC::Cast(vRow_, wLocal, RoundMode::CAST_NONE, hiddenSize_);
    }
    inQue_.FreeTensor(wLocal);
    PipeBarrier<PIPE_V>();
    AscendC::Mul(outFp32_, total, vRow_, hiddenSize_);
    PipeBarrier<PIPE_V>();
    outLocal = outQue_.AllocTensor<D_IN>();
    if constexpr (IsSameType<D_IN, half>::value) {
        AscendC::Cast(outLocal, outFp32_, RoundMode::CAST_RINT, hiddenSize_);
    } else if constexpr (IsSameType<D_IN, float>::value) {
        AscendC::Adds(outLocal, outFp32_, 0.0f, hiddenSize_);
    } else {
        AscendC::Cast(outLocal, outFp32_, RoundMode::CAST_RINT, hiddenSize_);
    }
    outQue_.EnQue(outLocal);
    outLocal = outQue_.DeQue<D_IN>();
    DataCopyPad(gradProjWeightGm_, outLocal, cpT);
    outQue_.FreeTensor(outLocal);
}

template <typename D_IN>
__aicore__ inline void BlockAttentionResidualsGradRegbase<D_IN>::Process()
{
    if (hiddenSize_ == 0U || coreNum_ == 0U) {
        return;
    }
    eventMTE3S_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    eventMTE2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    eventVMTE2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    eventSV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    eventVS_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    Duplicate(gswAcc_, 0.0f, hiddenSizeAlignFp32_);
    Duplicate(gswCmp_, 0.0f, hiddenSizeAlignFp32_);
    gswKahanPos_ = 0;
    PipeBarrier<PIPE_V>();

    for (uint32_t localIdx = 0; localIdx < tokenNum_; ++localIdx) {
        ProcessOneToken(tokenStart_ + localIdx);
    }
    WriteGradScoreWeight();
    SyncAll();
    ReduceAndWriteWeights();
}

} // namespace NsBlockAttentionResidualsGrad

#endif // BLOCK_ATTENTION_RESIDUALS_GRAD_REGBASE_H
