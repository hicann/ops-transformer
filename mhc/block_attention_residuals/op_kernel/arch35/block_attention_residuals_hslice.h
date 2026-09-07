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
 * \file block_attention_residuals_hslice.h
 * \brief arch35 TK-HSLICE：以 TK-RELOAD 为基础，对 H 切块（hiddenSizeChunk）逐块重复搬运 v，
 *        支持任意 H（H 上限仅受 GM 容量约束）。计算使用 RegBase VF，偶数长度双路执行。
 *
 * 计算流程（每 token）：
 *   Phase A（score pass，按 h 块）：
 *     每块先搬 projWeight/normWeight 对应切片 → scoreWeight 块；
 *     再逐行搬 v[n, hc] → Cast → 本块行内归约 sumSq[n]/score[n]（vecMeta/metaSoftmax 标量槽）；
 *     块结束时用 Kahan 求和累加进 sumSqAcc/scoreAcc。
 *   Finalize：逐元素 invNorm = 1/sqrt(sumSq*invH+eps)、score *= invNorm（全向量）。
 *   SoftmaxSmall：小 B 向量 Softmax → probs → Brcb 展开到 metaBrc[n*8]。
 *   Phase B（weighted pass，按 h 块）：
 *     每块 Counter 外置 Cast+MulAdd 累加 outFp32，块结束时 Cast → 写 hiddenStates 对应切片。
 *
 * 行序（golden）：n=0..N-1 ← block_res[n]；n=N ← partial_block
 */
#ifndef ATTN_RES_FWD_HSLICE_H
#define ATTN_RES_FWD_HSLICE_H

#include "kernel_operator.h"
#include "../block_attention_residuals_tiling_data.h"
#include "reduce_common.h"
#include "block_attention_residuals_reload.h" // BlockAttentionResidualsInitParams / BUFFER_NUM_RELOAD
#include "block_attention_residuals_regbase_common.h"

namespace BlockAttentionResiduals {

using namespace AscendC;

template <typename D_IN>
class BlockAttentionResidualsHSlice {
public:
    __aicore__ inline BlockAttentionResidualsHSlice(TPipe *pipe, const BlockAttentionResidualsTilingData *tilingData)
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
        hiddenSizeChunk_ = tiling_->hiddenSizeChunk;
        if (hiddenSizeChunk_ == 0 || hiddenSizeChunk_ > hiddenSize_) {
            hiddenSizeChunk_ = hiddenSize_;
        }
        hChunks_ = (hiddenSizeChunk_ == 0) ? 0 : (hiddenSize_ + hiddenSizeChunk_ - 1) / hiddenSizeChunk_;
        lastChunkLen_ = hiddenSize_ - (hChunks_ - 1) * hiddenSizeChunk_;
        hiddenSizeChunkU32_ = static_cast<uint32_t>(hiddenSizeChunk_);
        blockCountU32_ = static_cast<uint32_t>(blockCount_);
        hChunkAlignBf16_ = (hiddenSizeChunkU32_ + ELEM_PER_BLK_BF16 - 1U) / ELEM_PER_BLK_BF16 * ELEM_PER_BLK_BF16;
        hChunkAlignFp32_ = (hiddenSizeChunkU32_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
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
        // scoreWeight 切块在每 token 每 h 块内现算，无需整行驻留
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
    __aicore__ inline void KahanSumUpdate(LocalTensor<float> &input, LocalTensor<float> sumList[2], int32_t &sumPos)
    {
        LocalTensor<float> sum = sumList[sumPos];
        LocalTensor<float> compensation = sumList[1 - sumPos];
        Sub(input, input, compensation, blockCountU32_); // y = x - c
        PipeBarrier<PIPE_V>();
        Add(compensation, input, sum, blockCountU32_); // t = sum + y
        PipeBarrier<PIPE_V>();
        Sub(sum, compensation, sum, blockCountU32_);
        PipeBarrier<PIPE_V>();
        Sub(sum, sum, input, blockCountU32_); // c = (t - sum) - y
        PipeBarrier<PIPE_V>();
        sumPos = 1 - sumPos;
    }

    __aicore__ inline LocalTensor<float> GetSumSqAcc()
    {
        return sumSqKahanPos_ == 0 ? sumSqAcc_ : sumSqComp_;
    }

    __aicore__ inline LocalTensor<float> GetScoreAcc()
    {
        return scoreKahanPos_ == 0 ? scoreAcc_ : scoreComp_;
    }

    __aicore__ inline void InitBuffers()
    {
        pipe_->InitBuffer(inQue_, BUFFER_NUM_RELOAD, hChunkAlignBf16_ * sizeof(D_IN));
        pipe_->InitBuffer(outQue_, 1, hChunkAlignBf16_ * sizeof(D_IN));
        pipe_->InitBuffer(scoreWeightBuf_, hChunkAlignFp32_ * sizeof(float));
        pipe_->InitBuffer(vRowBuf_, hChunkAlignFp32_ * sizeof(float));
        pipe_->InitBuffer(outFp32Buf_, hChunkAlignFp32_ * sizeof(float));
        metaAlign_ = (blockCountU32_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32 * ELEM_PER_BLK_FP32;
        pipe_->InitBuffer(vecMetaBuf_, metaAlign_ * sizeof(float));
        pipe_->InitBuffer(metaSoftmaxBuf_, metaAlign_ * sizeof(float));
        pipe_->InitBuffer(metaBrcBuf_, metaAlign_ * ELEM_PER_BLK_FP32 * sizeof(float));
        pipe_->InitBuffer(scalarBuf_, SCALAR_LOCAL_ELEMS * sizeof(float));
        pipe_->InitBuffer(sumSqBuf_, metaAlign_ * sizeof(float));
        pipe_->InitBuffer(sumSqCompBuf_, metaAlign_ * sizeof(float));
        pipe_->InitBuffer(scoreBuf_, metaAlign_ * sizeof(float));
        pipe_->InitBuffer(scoreCompBuf_, metaAlign_ * sizeof(float));

        scoreWeight_ = scoreWeightBuf_.Get<float>();
        vRow_ = vRowBuf_.Get<float>();
        outFp32_ = outFp32Buf_.Get<float>();
        vecMeta_ = vecMetaBuf_.Get<float>();
        metaSoftmax_ = metaSoftmaxBuf_.Get<float>();
        metaBrc_ = metaBrcBuf_.Get<float>();
        scalarLocal_ = scalarBuf_.Get<float>();
        Duplicate(scalarLocal_, 0.0f, SCALAR_LOCAL_ELEMS);
        PipeBarrier<PIPE_V>();
        sumSqAcc_ = sumSqBuf_.Get<float>();
        sumSqComp_ = sumSqCompBuf_.Get<float>();
        scoreAcc_ = scoreBuf_.Get<float>();
        scoreComp_ = scoreCompBuf_.Get<float>();

        if (needBackward_) {
            // inv/probs 均整块（metaAlign float）经 Que 直写 GM
            pipe_->InitBuffer(invQue_, 1, metaAlign_ * sizeof(float));
            pipe_->InitBuffer(probsQue_, 1, metaAlign_ * sizeof(float));
        }
    }

    /*! 按 len 切片的 GM→UB 搬入（DataCopyPad，len 可为非对齐尾部） */
    __aicore__ inline void CopyInRowEnqueueOff(const GlobalTensor<D_IN> &srcGm, int64_t srcOffset, uint32_t len)
    {
        LocalTensor<D_IN> inLocal = inQue_.AllocTensor<D_IN>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(D_IN)), 0, 0, 0};
        DataCopyPadExtParams<D_IN> padParams{false, 0, 0, 0};
        DataCopyPad(inLocal, srcGm[srcOffset], copyParams, padParams);
        inQue_.EnQue(inLocal);
    }

    __aicore__ inline LocalTensor<D_IN> CopyInRowDeQue()
    {
        return inQue_.DeQue<D_IN>();
    }

    __aicore__ inline LocalTensor<D_IN> CopyInRowSyncOff(const GlobalTensor<D_IN> &srcGm, int64_t srcOffset,
                                                         uint32_t len)
    {
        CopyInRowEnqueueOff(srcGm, srcOffset, len);
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

    /*! 搬入第 hc 块 scoreWeight（projWeight × normWeight 切片） */
    __aicore__ inline void LoadScoreWeightChunk(int64_t hcOffset, uint32_t hcLen)
    {
        LocalTensor<D_IN> inLocal = CopyInRowSyncOff(projWeightGm_, hcOffset, hcLen);
        RegBase::CastB16ToFp32Dual(vRow_, inLocal, hcLen);
        FreeInRow(inLocal);

        inLocal = CopyInRowSyncOff(normWeightGm_, hcOffset, hcLen);
        RegBase::CastB16ToFp32Dual(outFp32_, inLocal, hcLen);
        FreeInRow(inLocal);

        RegBase::MulDual(scoreWeight_, vRow_, outFp32_, hcLen);
    }

    /*! 本块行归约：vecMeta[n]=Σv²；metaSoftmax[n]=Σ(v*scoreWeight)。metaIdx 标量槽（WholeReduce 标量 sink）。 */
    __aicore__ inline void ProcessRowScoreChunk(const LocalTensor<D_IN> &inLocal, uint32_t metaIdx, uint32_t hcLen)
    {
        RegBase::CastB16ToFp32Dual(vRow_, inLocal, hcLen);
        RegBase::ReduceSquareSum(vecMeta_[metaIdx], vRow_, hcLen);
        RegBase::ReduceMulSum(metaSoftmax_[metaIdx], vRow_, scoreWeight_, hcLen);
    }

    /*! Phase A：按 h 块用 Kahan 求和累计 sumSqAcc/scoreAcc */
    __aicore__ inline void ComputePreSoftmaxScores(int64_t tokenIdx)
    {
        Duplicate(sumSqAcc_, 0.0f, metaAlign_);
        Duplicate(sumSqComp_, 0.0f, metaAlign_);
        Duplicate(scoreAcc_, 0.0f, metaAlign_);
        Duplicate(scoreComp_, 0.0f, metaAlign_);
        PipeBarrier<PIPE_V>();
        sumSqKahanPos_ = 0;
        scoreKahanPos_ = 0;
        for (int64_t hc = 0; hc < hChunks_; ++hc) {
            const uint32_t hcLen = static_cast<uint32_t>((hc + 1 == hChunks_) ? lastChunkLen_ : hiddenSizeChunk_);
            const int64_t hcOffset = hc * hiddenSizeChunk_;
            LoadScoreWeightChunk(hcOffset, hcLen);
            CopyInRowEnqueueOff(GetLogicRowGm(0), GetLogicRowOffset(tokenIdx, 0) + hcOffset, hcLen);
            for (uint32_t n = 0; n < blockCountU32_; ++n) {
                LocalTensor<D_IN> inLocal = CopyInRowDeQue();
                if (n + 1U < blockCountU32_) {
                    CopyInRowEnqueueOff(GetLogicRowGm(n + 1U), GetLogicRowOffset(tokenIdx, n + 1U) + hcOffset, hcLen);
                }
                ProcessRowScoreChunk(inLocal, n, hcLen);
                FreeInRow(inLocal);
            }
            LocalTensor<float> sumSqList[2] = {sumSqAcc_, sumSqComp_};
            LocalTensor<float> scoreList[2] = {scoreAcc_, scoreComp_};
            KahanSumUpdate(vecMeta_, sumSqList, sumSqKahanPos_);
            KahanSumUpdate(metaSoftmax_, scoreList, scoreKahanPos_);
        }
    }

    /*! invNorm=1/sqrt(sumSq*invH+eps)（逐元素）；score *= invNorm；整块写 invNorm GM */
    __aicore__ inline void FinalizeScores(int64_t tokenIdx)
    {
        LocalTensor<float> sumSqAcc = GetSumSqAcc();
        LocalTensor<float> scoreAcc = GetScoreAcc();
        Muls(sumSqAcc, sumSqAcc, invHiddenSize_, blockCountU32_);
        PipeBarrier<PIPE_V>();
        Adds(sumSqAcc, sumSqAcc, normEps_, blockCountU32_);
        PipeBarrier<PIPE_V>();
        Sqrt(sumSqAcc, sumSqAcc, blockCountU32_);
        PipeBarrier<PIPE_V>();
        Duplicate(metaSoftmax_, 1.0f, metaAlign_);
        PipeBarrier<PIPE_V>();
        Div(sumSqAcc, metaSoftmax_, sumSqAcc, blockCountU32_); // sumSqAcc ← invNorm
        PipeBarrier<PIPE_V>();
        Mul(scoreAcc, scoreAcc, sumSqAcc, blockCountU32_);
        PipeBarrier<PIPE_V>();
        if (needBackward_) {
            LocalTensor<float> invUb = invQue_.AllocTensor<float>();
            CopyCompactFloatsUb(invUb, sumSqAcc, blockCountU32_);
            invQue_.EnQue(invUb);
            invUb = invQue_.DeQue<float>();
            DataCopyExtParams invParams{1, static_cast<uint32_t>(blockCount_ * sizeof(float)), 0, 0, 0};
            const int64_t gmOffset = tokenIdx * blockCount_;
            DataCopyPad(invNormGm_[gmOffset], invUb, invParams);
            invQue_.FreeTensor(invUb);
        }
    }

    __aicore__ inline void SoftmaxSmall()
    {
        LocalTensor<float> scoreAcc = GetScoreAcc();
        RegBase::SoftmaxSmallRegBase(scoreAcc, blockCountU32_, metaAlign_, scalarLocal_, metaSoftmax_, metaBrc_);
        // 紧凑 prob → metaBrc[n*8]：零填充 staging 后单次 Brcb（repeat=ceil(B/8)）
        Duplicate(metaSoftmax_, 0.0f, metaAlign_);
        PipeBarrier<PIPE_V>();
        CopyCompactFloatsUb(metaSoftmax_, scoreAcc, blockCountU32_);
        const uint8_t brcRepeat = static_cast<uint8_t>((blockCountU32_ + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32);
        Brcb(metaBrc_, metaSoftmax_, brcRepeat, {1, MOV_8});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteProbsQue(int64_t tokenIdx)
    {
        LocalTensor<float> scoreAcc = GetScoreAcc();
        LocalTensor<float> probsUb = probsQue_.AllocTensor<float>();
        CopyCompactFloatsUb(probsUb, scoreAcc, blockCountU32_);
        probsQue_.EnQue(probsUb);
        probsUb = probsQue_.DeQue<float>();
        DataCopyExtParams probsParams{1, static_cast<uint32_t>(blockCount_ * sizeof(float)), 0, 0, 0};
        const int64_t gmOffset = tokenIdx * blockCount_;
        DataCopyPad(probsGm_[gmOffset], probsUb, probsParams);
        probsQue_.FreeTensor(probsUb);
    }

    /*! Softmax 后加权累加（Counter 外置）：outFp32 += v[n,hc] * prob[n] */
    __aicore__ inline void ProcessRowWeighted(const LocalTensor<D_IN> &inLocal, uint32_t metaIdx, uint32_t hcLen)
    {
        RegBase::WeightedMulAddFromB16(outFp32_, inLocal, metaBrc_[metaIdx * ELEM_PER_BLK_FP32], hcLen);
    }

    /*! Phase B：按 h 块加权输出，逐块 Cast → 写 hiddenStates 切片 */
    __aicore__ inline void WeightedOutputSliced(int64_t tokenIdx)
    {
        for (int64_t hc = 0; hc < hChunks_; ++hc) {
            const uint32_t hcLen = static_cast<uint32_t>((hc + 1 == hChunks_) ? lastChunkLen_ : hiddenSizeChunk_);
            const int64_t hcOffset = hc * hiddenSizeChunk_;
            Duplicate(outFp32_, 0.0f, hcLen);
            PipeBarrier<PIPE_V>();
            CopyInRowEnqueueOff(GetLogicRowGm(0), GetLogicRowOffset(tokenIdx, 0) + hcOffset, hcLen);
            for (uint32_t n = 0; n < blockCountU32_; ++n) {
                LocalTensor<D_IN> inLocal = CopyInRowDeQue();
                if (n + 1U < blockCountU32_) {
                    CopyInRowEnqueueOff(GetLogicRowGm(n + 1U), GetLogicRowOffset(tokenIdx, n + 1U) + hcOffset, hcLen);
                }
                ProcessRowWeighted(inLocal, n, hcLen);
                FreeInRow(inLocal);
            }
            WriteHiddenStatesChunk(tokenIdx, hcOffset, hcLen);
        }
    }

    __aicore__ inline void WriteHiddenStatesChunk(int64_t tokenIdx, int64_t hcOffset, uint32_t hcLen)
    {
        LocalTensor<D_IN> outLocal = outQue_.AllocTensor<D_IN>();
        RegBase::CastFp32ToB16Dual(outLocal, outFp32_, hcLen);
        outQue_.EnQue(outLocal);
        outLocal = outQue_.DeQue<D_IN>();
        const int64_t outOffset = tokenIdx * hiddenSize_ + hcOffset;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(hcLen * sizeof(D_IN)), 0, 0, 0};
        DataCopyPad(hiddenStatesGm_[outOffset], outLocal, copyParams);
        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void ProcessOneToken(int64_t tokenIdx)
    {
        ComputePreSoftmaxScores(tokenIdx); // Phase A：切 h 累计 sumSq/score
        FinalizeScores(tokenIdx);          // invNorm + score 缩放
        SoftmaxSmall();                    // softmax → probs → Brcb
        if (needBackward_) {
            WriteProbsQue(tokenIdx);
        }
        WeightedOutputSliced(tokenIdx); // Phase B：切 h 加权输出
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
    TBuf<TPosition::VECCALC> sumSqBuf_;
    TBuf<TPosition::VECCALC> sumSqCompBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> scoreCompBuf_;

    LocalTensor<float> scoreWeight_;
    LocalTensor<float> vRow_;
    LocalTensor<float> outFp32_;
    LocalTensor<float> vecMeta_;
    LocalTensor<float> metaSoftmax_;
    LocalTensor<float> metaBrc_;
    LocalTensor<float> scalarLocal_;
    LocalTensor<float> sumSqAcc_;
    LocalTensor<float> sumSqComp_;
    LocalTensor<float> scoreAcc_;
    LocalTensor<float> scoreComp_;

    int64_t numTokens_{0};
    int64_t numBlocks_{0};
    int64_t hiddenSize_{0};
    int64_t hiddenSizeChunk_{0};
    uint32_t hiddenSizeChunkU32_{0};
    uint32_t hChunkAlignBf16_{0};
    uint32_t hChunkAlignFp32_{0};
    int64_t hChunks_{0};
    int64_t lastChunkLen_{0};
    int64_t blockCount_{0};
    uint32_t blockCountU32_{0};
    uint32_t metaAlign_{0};
    int64_t tokensPerCore_{0};
    int64_t tokenStart_{0};
    int64_t tokenNum_{0};
    uint32_t blockIdx_{0};
    uint32_t blockNum_{0};
    int32_t sumSqKahanPos_{0};
    int32_t scoreKahanPos_{0};
    float normEps_{1e-6F};
    float invHiddenSize_{0.0f};
    bool needBackward_{false};
};

} // namespace BlockAttentionResiduals

#endif // ATTN_RES_FWD_HSLICE_H
