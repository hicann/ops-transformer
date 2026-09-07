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
 * \file block_attention_residuals_reload.h
 * \brief TK-RELOAD: Softmax 前每行 GM 读 1 次（同趟 sumSq→invNorm→score）；
 *        Softmax 后重搬 BF16→Cast（第 2 次）；合计每行 v 搬入 2 次；无 FP32-v Workspace
 *
 * 行序（golden）：n=0..N-1 ← block_res[n]；n=N ← partial_block
 */
#ifndef ATTN_RES_FWD_RELOAD_H
#define ATTN_RES_FWD_RELOAD_H

#include "kernel_operator.h"
#include "../block_attention_residuals_tiling_data.h"
#include "reduce_common.h"
#include "block_attention_residuals_regbase_common.h"

namespace BlockAttentionResiduals {

using namespace AscendC;

constexpr uint32_t BUFFER_NUM_RELOAD = 2;
constexpr uint32_t ELEM_PER_BLK_BF16 = 16;

struct BlockAttentionResidualsInitParams {
    GM_ADDR partialBlock;
    GM_ADDR blockRes;
    GM_ADDR projWeight;
    GM_ADDR normWeight;
    GM_ADDR hiddenStates;
    GM_ADDR invNorm;
    GM_ADDR probs;
};

template <typename D_IN>
class BlockAttentionResidualsReload {
public:
    __aicore__ inline BlockAttentionResidualsReload(TPipe *pipe, const BlockAttentionResidualsTilingData *tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
        numTokens_ = tiling_->numTokens;
        numBlocks_ = tiling_->numBlocks;
        hiddenSize_ = tiling_->hiddenSize;
        blockCount_ = tiling_->blockCount;
        normEps_ = tiling_->normEps;
        invHiddenSize_ = tiling_->invHiddenSize;
        tokensPerCore_ = tiling_->tokensPerCore;
        needBackward_ = tiling_->needBackward != 0;
        hiddenSizeU32_ = static_cast<uint32_t>(hiddenSize_);
        blockCountU32_ = static_cast<uint32_t>(blockCount_);
        hiddenSizeAlignBf16_ = (hiddenSizeU32_ + ELEM_PER_BLK_BF16 - 1U) / ELEM_PER_BLK_BF16 * ELEM_PER_BLK_BF16;
        hiddenSizeAlignFp32_ = (hiddenSizeU32_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
    }

    __aicore__ inline void Init(const BlockAttentionResidualsInitParams &params)
    {
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        if (blockIdx_ >= blockNum_) {
            return;
        }

        tokenStart_ = static_cast<int64_t>(blockIdx_) * tokensPerCore_;
        if (tokenStart_ >= numTokens_) {
            tokenNum_ = 0;
            return;
        }
        tokenNum_ = tokensPerCore_;
        if (tokenStart_ + tokenNum_ > numTokens_) {
            tokenNum_ = numTokens_ - tokenStart_;
        }

        partialBlockGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.partialBlock));
        blockResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.blockRes));
        projWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.projWeight));
        normWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.normWeight));
        hiddenStatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ D_IN *>(params.hiddenStates));
        if (needBackward_) {
            invNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.invNorm));
            probsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.probs));
        }

        InitBuffers();
        LoadScoreWeight();
    }

    __aicore__ inline void Process()
    {
        if (tokenNum_ == 0 || hiddenSize_ == 0) {
            return;
        }
        for (int64_t localIdx = 0; localIdx < tokenNum_; ++localIdx) {
            ProcessOneToken(tokenStart_ + localIdx);
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        pipe_->InitBuffer(inQue_, BUFFER_NUM_RELOAD, hiddenSizeAlignBf16_ * sizeof(D_IN));
        pipe_->InitBuffer(outQue_, 1, hiddenSizeAlignBf16_ * sizeof(D_IN));
        pipe_->InitBuffer(scoreWeightBuf_, hiddenSizeAlignFp32_ * sizeof(float));
        pipe_->InitBuffer(vRowBuf_, hiddenSizeAlignFp32_ * sizeof(float));
        pipe_->InitBuffer(outFp32Buf_, hiddenSizeAlignFp32_ * sizeof(float));
        // 与 arch22 一致：meta 按 32B block（8 fp32）对齐
        metaAlign_ = (blockCountU32_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
        pipe_->InitBuffer(vecMetaBuf_, metaAlign_ * sizeof(float));
        pipe_->InitBuffer(metaSoftmaxBuf_, metaAlign_ * sizeof(float));
        // Softmax 期间作 fold/Brcb scratch；Softmax 后 Brcb 成 metaBrc[n*8] 供 Weighted
        pipe_->InitBuffer(metaBrcBuf_, metaAlign_ * ELEM_PER_BLK_FP32 * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, SCALAR_LOCAL_ELEMS * sizeof(float));

        scoreWeight_ = scoreWeightBuf_.Get<float>();
        vRow_ = vRowBuf_.Get<float>();
        outFp32_ = outFp32Buf_.Get<float>();
        vecMeta_ = vecMetaBuf_.Get<float>();
        metaSoftmax_ = metaSoftmaxBuf_.Get<float>();
        metaBrc_ = metaBrcBuf_.Get<float>();
        scalarLocal_ = scalarBuf_.Get<float>();
        Duplicate(scalarLocal_, 0.0f, SCALAR_LOCAL_ELEMS);
        PipeBarrier<PIPE_V>();

        if (needBackward_) {
            // inv：每行 1 标量经 invQue_；probs：Softmax 后每 token 一次搬 B 个经 probsQue_
            pipe_->InitBuffer(invQue_, 1, ELEM_PER_BLK_FP32 * sizeof(float));
            pipe_->InitBuffer(probsQue_, 1, metaAlign_ * sizeof(float));
        }
    }

    __aicore__ inline void CopyInRowEnqueue(const GlobalTensor<D_IN> &srcGm, int64_t srcOffset)
    {
        LocalTensor<D_IN> inLocal = inQue_.AllocTensor<D_IN>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
        DataCopyPadExtParams<D_IN> padParams{false, 0, 0, 0};
        DataCopyPad(inLocal, srcGm[srcOffset], copyParams, padParams);
        inQue_.EnQue(inLocal);
    }

    __aicore__ inline LocalTensor<D_IN> CopyInRowDeQue()
    {
        return inQue_.DeQue<D_IN>();
    }

    __aicore__ inline LocalTensor<D_IN> CopyInRowSync(const GlobalTensor<D_IN> &srcGm, int64_t srcOffset)
    {
        CopyInRowEnqueue(srcGm, srcOffset);
        return CopyInRowDeQue();
    }

    __aicore__ inline void FreeInRow(LocalTensor<D_IN> &inLocal)
    {
        inQue_.FreeTensor(inLocal);
    }

    /*! golden：n < N → residual；n == N → partial_block */
    __aicore__ inline const GlobalTensor<D_IN> &GetLogicRowGm(uint32_t n) const
    {
        return (n < numBlocks_) ? blockResGm_ : partialBlockGm_;
    }

    __aicore__ inline int64_t GetLogicRowOffset(int64_t tokenIdx, int64_t n) const
    {
        if (n < numBlocks_) {
            return tokenIdx * numBlocks_ * hiddenSize_ + n * hiddenSize_;
        }
        return tokenIdx * hiddenSize_;
    }

    /*! Softmax 前：RegBase Cast + 融合 Reduce(Square/Mul) + InvRms → vecMeta[metaIdx] */
    __aicore__ inline void ProcessRowScore(const LocalTensor<D_IN> &inLocal, uint32_t metaIdx, int64_t tokenIdx)
    {
        RegBase::CastB16ToFp32Dual(vRow_, inLocal, hiddenSizeU32_);
        // 融合 sum(v²)，不写 outFp32 中间行
        RegBase::ReduceSquareSum(scalarLocal_, vRow_, hiddenSizeU32_);
        RegBase::InvRmsScalar(scalarLocal_, invHiddenSize_, normEps_);
        if (needBackward_) {
            LocalTensor<float> invUb = invQue_.AllocTensor<float>();
            CopyMetaScalarToLocal(invUb, scalarLocal_);
            invQue_.EnQue(invUb);
            invUb = invQue_.DeQue<float>();
            DataCopyExtParams scalarParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            const int64_t gmOffset = tokenIdx * blockCount_ + static_cast<int64_t>(metaIdx);
            DataCopyPad(invNormGm_[gmOffset], invUb, scalarParams);
            invQue_.FreeTensor(invUb);
        }
        RegBase::BroadcastScalarMulDual(vRow_, vRow_, scalarLocal_, hiddenSizeU32_);
        // 融合 sum(v * scoreWeight) → vecMeta[n]
        RegBase::ReduceMulSum(vecMeta_[metaIdx], vRow_, scoreWeight_, hiddenSizeU32_);
    }

    /*! Softmax 后加权累加：RegBase VF Cast + DIST_BRC(metaBrc[n*8]) MulAddDst */
    __aicore__ inline void ProcessRowWeighted(const LocalTensor<D_IN> &inLocal, uint32_t metaIdx)
    {
        RegBase::WeightedMulAddFromB16(outFp32_, inLocal, metaBrc_[metaIdx * ELEM_PER_BLK_FP32], hiddenSizeU32_);
    }

    __aicore__ inline void LoadScoreWeight()
    {
        LocalTensor<D_IN> inLocal = CopyInRowSync(projWeightGm_, 0);
        RegBase::CastB16ToFp32Dual(vRow_, inLocal, hiddenSizeU32_);
        FreeInRow(inLocal);

        inLocal = CopyInRowSync(normWeightGm_, 0);
        RegBase::CastB16ToFp32Dual(outFp32_, inLocal, hiddenSizeU32_);
        FreeInRow(inLocal);

        RegBase::MulDual(scoreWeight_, vRow_, outFp32_, hiddenSizeU32_);
    }

    __aicore__ inline void ComputePreSoftmaxScores(int64_t tokenIdx)
    {
        if (blockCount_ == 0) {
            return;
        }
        CopyInRowEnqueue(GetLogicRowGm(0), GetLogicRowOffset(tokenIdx, 0));
        for (uint32_t n = 0; n < blockCountU32_; ++n) {
            LocalTensor<D_IN> inLocal = CopyInRowDeQue();
            if (n + 1U < blockCountU32_) {
                CopyInRowEnqueue(GetLogicRowGm(n + 1U), GetLogicRowOffset(tokenIdx, n + 1U));
            }
            ProcessRowScore(inLocal, n, tokenIdx);
            FreeInRow(inLocal);
        }
    }

    __aicore__ inline void SoftmaxSmall()
    {
        // probs 写在 vecMeta_；Brcb → metaBrc[n*8] 供 Weighted（n<B 才用）
        RegBase::SoftmaxSmallRegBase(vecMeta_, blockCountU32_, metaAlign_, scalarLocal_, metaSoftmax_, metaBrc_);
        const uint8_t brcRepeat = static_cast<uint8_t>((blockCountU32_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32);
        Brcb(metaBrc_, vecMeta_, brcRepeat, {1, MOV_8});
        PipeBarrier<PIPE_V>();
    }

    /*! Softmax 后：每 token 一次搬出 B 个 probs（Que 同步） */
    __aicore__ inline void WriteProbsQue(int64_t tokenIdx)
    {
        LocalTensor<float> probsUb = probsQue_.AllocTensor<float>();
        CopyCompactFloatsUb(probsUb, vecMeta_, blockCountU32_);
        probsQue_.EnQue(probsUb);
        probsUb = probsQue_.DeQue<float>();
        DataCopyExtParams probsParams{1, static_cast<uint32_t>(blockCount_ * sizeof(float)), 0, 0, 0};
        const int64_t gmOffset = tokenIdx * blockCount_;
        DataCopyPad(probsGm_[gmOffset], probsUb, probsParams);
        probsQue_.FreeTensor(probsUb);
    }

    __aicore__ inline void WeightedOutputReload(int64_t tokenIdx)
    {
        Duplicate(outFp32_, 0.0f, hiddenSizeU32_);
        PipeBarrier<PIPE_V>();
        if (blockCount_ == 0) {
            return;
        }
        CopyInRowEnqueue(GetLogicRowGm(0), GetLogicRowOffset(tokenIdx, 0));
        for (uint32_t n = 0; n < blockCountU32_; ++n) {
            LocalTensor<D_IN> inLocal = CopyInRowDeQue();
            if (n + 1U < blockCountU32_) {
                CopyInRowEnqueue(GetLogicRowGm(n + 1U), GetLogicRowOffset(tokenIdx, n + 1U));
            }
            ProcessRowWeighted(inLocal, n);
            FreeInRow(inLocal);
        }
    }

    __aicore__ inline void WriteHiddenStates(int64_t tokenIdx)
    {
        LocalTensor<D_IN> outLocal = outQue_.AllocTensor<D_IN>();
        RegBase::CastFp32ToB16Dual(outLocal, outFp32_, hiddenSizeU32_);
        outQue_.EnQue(outLocal);
        outLocal = outQue_.DeQue<D_IN>();
        const int64_t outOffset = tokenIdx * hiddenSize_;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
        DataCopyPad(hiddenStatesGm_[outOffset], outLocal, copyParams);
        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void ProcessOneToken(int64_t tokenIdx)
    {
        ComputePreSoftmaxScores(tokenIdx);
        SoftmaxSmall();
        if (needBackward_) {
            WriteProbsQue(tokenIdx);
        }
        WeightedOutputReload(tokenIdx);
        WriteHiddenStates(tokenIdx);
    }

private:
    TPipe *pipe_{nullptr};
    const BlockAttentionResidualsTilingData *tiling_{nullptr};

    GlobalTensor<D_IN> partialBlockGm_;
    GlobalTensor<D_IN> blockResGm_;
    GlobalTensor<D_IN> projWeightGm_;
    GlobalTensor<D_IN> normWeightGm_;
    GlobalTensor<D_IN> hiddenStatesGm_;
    GlobalTensor<float> invNormGm_;
    GlobalTensor<float> probsGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM_RELOAD> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TQue<QuePosition::VECOUT, 1> invQue_;
    TQue<QuePosition::VECOUT, 1> probsQue_;
    TBuf<TPosition::VECCALC> scoreWeightBuf_;
    TBuf<TPosition::VECCALC> vRowBuf_;
    TBuf<TPosition::VECCALC> outFp32Buf_;
    TBuf<TPosition::VECCALC> vecMetaBuf_;
    TBuf<TPosition::VECCALC> metaSoftmaxBuf_;
    TBuf<TPosition::VECCALC> metaBrcBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;

    LocalTensor<float> scoreWeight_;
    LocalTensor<float> vRow_;
    LocalTensor<float> outFp32_;
    LocalTensor<float> vecMeta_;
    LocalTensor<float> metaSoftmax_;
    LocalTensor<float> metaBrc_;
    LocalTensor<float> scalarLocal_;

    int64_t numTokens_{0};
    int64_t numBlocks_{0};
    int64_t hiddenSize_{0};
    uint32_t hiddenSizeU32_{0};
    uint32_t hiddenSizeAlignBf16_{0};
    uint32_t hiddenSizeAlignFp32_{0};
    int64_t blockCount_{0};
    uint32_t blockCountU32_{0};
    uint32_t metaAlign_{0};
    int64_t tokensPerCore_{0};
    int64_t tokenStart_{0};
    int64_t tokenNum_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{0};
    float normEps_{1e-6F};
    float invHiddenSize_{0.0f};
    bool needBackward_{false};
};

} // namespace BlockAttentionResiduals

#endif // ATTN_RES_FWD_RELOAD_H
