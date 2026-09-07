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
 * \file block_attention_residuals_resident.h
 * \brief TK-RESIDENT: Softmax 前单次搬入驻留 vBf16[B,H]；Softmax 后不重搬 GM
 *
 * 行序（golden）：n=0..N-1 ← block_res[n]；n=N ← partial_block
 */
#ifndef ATTN_RES_FWD_RESIDENT_H
#define ATTN_RES_FWD_RESIDENT_H

#include "kernel_operator.h"
#include "../block_attention_residuals_tiling_data.h"
#include "reduce_common.h"
#include "block_attention_residuals_regbase_common.h"
#include "block_attention_residuals_reload.h" // BlockAttentionResidualsInitParams

namespace BlockAttentionResiduals {

using namespace AscendC;

constexpr uint32_t BUFFER_NUM_RESIDENT = 1;

template <typename D_IN>
class BlockAttentionResidualsResident {
public:
    __aicore__ inline BlockAttentionResidualsResident(TPipe *pipe, const BlockAttentionResidualsTilingData *tilingData)
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
        pipe_->InitBuffer(inQue_, BUFFER_NUM_RESIDENT, hiddenSizeAlignBf16_ * sizeof(D_IN));
        pipe_->InitBuffer(outQue_, BUFFER_NUM_RESIDENT, hiddenSizeAlignBf16_ * sizeof(D_IN));
        pipe_->InitBuffer(vBf16Buf_, blockCountU32_ * hiddenSizeAlignBf16_ * sizeof(D_IN));
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

        vBf16_ = vBf16Buf_.Get<D_IN>();
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
            pipe_->InitBuffer(invQue_, 1, ELEM_PER_BLK_FP32 * sizeof(float));
            pipe_->InitBuffer(probsQue_, 1, metaAlign_ * sizeof(float));
        }
    }

    __aicore__ inline LocalTensor<D_IN> CopyInRow(const GlobalTensor<D_IN> &srcGm, int64_t srcOffset)
    {
        LocalTensor<D_IN> inLocal = inQue_.AllocTensor<D_IN>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(hiddenSize_ * sizeof(D_IN)), 0, 0, 0};
        DataCopyPadExtParams<D_IN> padParams{false, 0, 0, 0};
        DataCopyPad(inLocal, srcGm[srcOffset], copyParams, padParams);
        inQue_.EnQue(inLocal);
        inLocal = inQue_.DeQue<D_IN>();
        return inLocal;
    }

    __aicore__ inline void FreeInRow(LocalTensor<D_IN> &inLocal)
    {
        inQue_.FreeTensor(inLocal);
    }

    __aicore__ inline void LoadScoreWeight()
    {
        LocalTensor<D_IN> inLocal = CopyInRow(projWeightGm_, 0);
        RegBase::CastB16ToFp32Dual(vRow_, inLocal, hiddenSizeU32_);
        FreeInRow(inLocal);

        inLocal = CopyInRow(normWeightGm_, 0);
        RegBase::CastB16ToFp32Dual(outFp32_, inLocal, hiddenSizeU32_);
        FreeInRow(inLocal);

        RegBase::MulDual(scoreWeight_, vRow_, outFp32_, hiddenSizeU32_);
    }

    __aicore__ inline LocalTensor<D_IN> GetResidentRow(uint32_t n)
    {
        return vBf16_[n * hiddenSizeAlignBf16_];
    }

    /*! Phase-1~3：搬入驻留 + 同趟 sumSq→invNorm→score（向量路径，无 Get/Set）
     *  行序（golden）：n < N → residual[n]；n == N → partial_block
     */
    __aicore__ inline void LoadResidentAndScores(int64_t tokenIdx)
    {
        for (uint32_t n = 0; n < blockCountU32_; ++n) {
            LocalTensor<D_IN> inLocal;
            if (n < numBlocks_) {
                const int64_t blockBase = tokenIdx * numBlocks_ * hiddenSize_;
                inLocal = CopyInRow(blockResGm_, blockBase + static_cast<int64_t>(n) * hiddenSize_);
            } else {
                inLocal = CopyInRow(partialBlockGm_, tokenIdx * hiddenSize_);
            }
            LocalTensor<D_IN> dstRow = GetResidentRow(n);
            // 单条 Level0 Copy 受 VL 上限约束；RegBase 按 VL 分段并只拷有效 H。
            RegBase::CopyLocalExact(dstRow, inLocal, hiddenSizeU32_);
            RegBase::CastB16ToFp32Dual(vRow_, inLocal, hiddenSizeU32_);
            FreeInRow(inLocal);

            RegBase::ReduceSquareSum(scalarLocal_, vRow_, hiddenSizeU32_);
            RegBase::InvRmsScalar(scalarLocal_, invHiddenSize_, normEps_);
            if (needBackward_) {
                LocalTensor<float> invUb = invQue_.AllocTensor<float>();
                CopyMetaScalarToLocal(invUb, scalarLocal_);
                invQue_.EnQue(invUb);
                invUb = invQue_.DeQue<float>();
                DataCopyExtParams scalarParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                const int64_t gmOffset = tokenIdx * blockCount_ + static_cast<int64_t>(n);
                DataCopyPad(invNormGm_[gmOffset], invUb, scalarParams);
                invQue_.FreeTensor(invUb);
            }
            RegBase::BroadcastScalarMulDual(vRow_, vRow_, scalarLocal_, hiddenSizeU32_);
            RegBase::ReduceMulSum(vecMeta_[n], vRow_, scoreWeight_, hiddenSizeU32_);
        }
    }

    /*! Phase-4：小 B Softmax；随后 Brcb 到 metaBrc[n*8] 供 Weighted */
    __aicore__ inline void SoftmaxSmall()
    {
        RegBase::SoftmaxSmallRegBase(vecMeta_, blockCountU32_, metaAlign_, scalarLocal_, metaSoftmax_, metaBrc_);
        const uint8_t brcRepeat = static_cast<uint8_t>((blockCountU32_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32);
        Brcb(metaBrc_, vecMeta_, brcRepeat, {1, MOV_8});
        PipeBarrier<PIPE_V>();
    }

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

    __aicore__ inline void WeightedOutputResident()
    {
        Duplicate(outFp32_, 0.0f, hiddenSizeU32_);
        PipeBarrier<PIPE_V>();
        for (uint32_t n = 0; n < blockCountU32_; ++n) {
            LocalTensor<D_IN> row = GetResidentRow(n);
            RegBase::WeightedMulAddFromB16(outFp32_, row, metaBrc_[n * ELEM_PER_BLK_FP32], hiddenSizeU32_);
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
        LoadResidentAndScores(tokenIdx); // Phase-1~3：搬入 + 同趟 score
        SoftmaxSmall();                  // Phase-4
        if (needBackward_) {
            WriteProbsQue(tokenIdx);
        }
        WeightedOutputResident();    // Phase-5：不重搬 GM
        WriteHiddenStates(tokenIdx); // Phase-6
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

    TQue<QuePosition::VECIN, BUFFER_NUM_RESIDENT> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_RESIDENT> outQue_;
    TQue<QuePosition::VECOUT, 1> invQue_;
    TQue<QuePosition::VECOUT, 1> probsQue_;
    TBuf<TPosition::VECCALC> vBf16Buf_;
    TBuf<TPosition::VECCALC> scoreWeightBuf_;
    TBuf<TPosition::VECCALC> vRowBuf_;
    TBuf<TPosition::VECCALC> outFp32Buf_;
    TBuf<TPosition::VECCALC> vecMetaBuf_;
    TBuf<TPosition::VECCALC> metaSoftmaxBuf_;
    TBuf<TPosition::VECCALC> metaBrcBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;

    LocalTensor<D_IN> vBf16_;
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

#endif // ATTN_RES_FWD_RESIDENT_H
