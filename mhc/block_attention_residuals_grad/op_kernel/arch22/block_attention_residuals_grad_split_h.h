/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file arch22/block_attention_residuals_grad_split_h.h
 *  \brief A2/A3 H-split single-pass-H backup implementation; not included by the default build.
 */

#ifndef BLOCK_ATTENTION_RESIDUALS_GRAD_SPLIT_H_SINGLE_PASS_BACKUP_H
#define BLOCK_ATTENTION_RESIDUALS_GRAD_SPLIT_H_SINGLE_PASS_BACKUP_H

#include "block_attention_residuals_grad.h"

namespace NsBlockAttentionResidualsGrad {

// SPLIT_H 仅在单核 UB 内按 H 轴分块，多核任务仍按 B 轴分配。
// 由于 gradScore 依赖完整 H 轴上的点积，Kernel 分为三个阶段：
// 1. 分块遍历 H，计算并保存 gradScore[B, N+1] 和 varianceScale[B, N+1]；
// 2. 再次分块遍历 H，计算 gradV，并生成每个核的权重梯度部分和；
// 3. 全核同步后，由核 0 归约各核部分和并写出最终权重梯度。
template <typename T>
class BlockAttentionResidualsGradSplitH {
public:
    __aicore__ inline void Init(GM_ADDR partialBlock, GM_ADDR blockRes, GM_ADDR projWeight, GM_ADDR normWeight,
                                GM_ADDR gradHiddenState, GM_ADDR invNorm, GM_ADDR probs, GM_ADDR gradPartialBlock,
                                GM_ADDR gradBlockRes, GM_ADDR gradProjWeight, GM_ADDR gradNormWeight, GM_ADDR workspace,
                                const BlockAttentionResidualsGradTilingData *td);
    __aicore__ inline void Process();

private:
    __aicore__ inline GlobalTensor<T> GetV(int64_t block, int64_t batch) const;
    __aicore__ inline void LoadTensorTile(const GlobalTensor<T> &src, uint64_t offset, uint32_t length,
                                          const LocalTensor<float> &dst);
    __aicore__ inline void LoadMeta(const GlobalTensor<float> &src, uint64_t offset, uint32_t length,
                                    const LocalTensor<float> &dst);
    __aicore__ inline void StoreGradV(int64_t block, int64_t batch, uint32_t hStart, uint32_t length,
                                      const LocalTensor<float> &src);
    __aicore__ inline void ComputeAndSaveGradMeta();
    __aicore__ inline void InitScoreWeightTile(uint32_t hStart, uint32_t length);
    __aicore__ inline void ComputeGradVAndWeightPartials();
    __aicore__ inline void ReduceAndWriteWeights();

    TPipe pipe_;

    GlobalTensor<T> partialBlockGm_, blockResGm_, projWeightGm_, normWeightGm_;
    GlobalTensor<T> gradHiddenStateGm_;
    GlobalTensor<float> invNormGm_, probsGm_;
    GlobalTensor<T> gradPartialBlockGm_, gradBlockResGm_;
    GlobalTensor<T> gradProjWeightGm_, gradNormWeightGm_;
    GlobalTensor<float> coreWeightWkspGm_, allWeightWkspGm_, gradScoresWkspGm_, varianceScaleWkspGm_;

    TBuf<TPosition::VECCALC> scoreWeightBuf_, inputBuf_;
    TBuf<TPosition::VECCALC> floatBuf0_, floatBuf1_, floatBuf2_;
    TBuf<TPosition::VECCALC> weightAccBuf_, weightCmpBuf_;
    TBuf<TPosition::VECCALC> probsBuf_, invNormBuf_, gradProbBuf_, gradScoreBuf_;
    TBuf<TPosition::VECCALC> gradProbCmpBuf_, weightedVDotCmpBuf_, reduceDstBuf_;

    int64_t batchSize_{0};
    int64_t numBlocks_{0};
    int64_t totalBlocks_{0};
    int64_t hiddenSize_{0};
    int64_t hiddenTileSize_{0};
    int64_t coreNum_{0};
    int64_t coreBatchStart_{0};
    int64_t coreBatchEnd_{0};
    uint64_t perCoreWkspBytes_{0};
    float invH_{0.0f};

    event_t eventMte2V_;
    event_t eventVMte2_;
    event_t eventVMte3_;
    event_t eventMte3Mte2_;
    event_t eventVS_;
    event_t eventSV_;
};

template <typename T>
__aicore__ inline GlobalTensor<T> BlockAttentionResidualsGradSplitH<T>::GetV(int64_t block, int64_t batch) const
{
    // V = concat(blockRes, partialBlock)：前 N 个 block 来自 blockRes，最后一个来自 partialBlock。
    if (block < numBlocks_) {
        uint64_t offset =
            (static_cast<uint64_t>(batch) * static_cast<uint64_t>(numBlocks_) + static_cast<uint64_t>(block)) *
            static_cast<uint64_t>(hiddenSize_);
        return blockResGm_[offset];
    }
    return partialBlockGm_[static_cast<uint64_t>(batch) * static_cast<uint64_t>(hiddenSize_)];
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradSplitH<T>::LoadTensorTile(const GlobalTensor<T> &src, uint64_t offset,
                                                                            uint32_t length,
                                                                            const LocalTensor<float> &dst)
{
    // 每次只搬入一个连续 H tile，并统一转换为 FP32 参与后续计算。
    LocalTensor<T> input = inputBuf_.Get<T>();
    DataCopyExtParams copyParams(1, length * sizeof(T), 0, 0, 0);
    DataCopyPad(input, src[offset], copyParams, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMte2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V_);
    if constexpr (std::is_same<T, float>::value) {
        // FP32 输入无需类型转换。使用矢量逐元素复制，支持最后一个非 32B 对齐的 H 尾块。
        Adds(dst, input, 0.0f, length);
    } else {
        Cast(dst, input, RoundMode::CAST_NONE, length);
    }
    SetFlag<HardEvent::V_MTE2>(eventVMte2_);
    WaitFlag<HardEvent::V_MTE2>(eventVMte2_);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradSplitH<T>::LoadMeta(const GlobalTensor<float> &src, uint64_t offset,
                                                                      uint32_t length, const LocalTensor<float> &dst)
{
    // K=N+1 轴元数据保持完整搬入 UB，尾部通过 DataCopyPad 补齐到 32 字节。
    DataCopyExtParams copyParams(1, length * sizeof(float), 0, 0, 0);
    uint32_t alignedBytes = (length * sizeof(float) + META_ALIGN_BYTES - 1U) / META_ALIGN_BYTES * META_ALIGN_BYTES;
    uint32_t rightPad = (alignedBytes - length * sizeof(float)) / sizeof(float);
    DataCopyPad(dst, src[offset], copyParams, {true, 0U, static_cast<uint8_t>(rightPad), 0U});
    SetFlag<HardEvent::MTE2_V>(eventMte2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V_);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradSplitH<T>::StoreGradV(int64_t block, int64_t batch, uint32_t hStart,
                                                                        uint32_t length, const LocalTensor<float> &src)
{
    // FP32 计算结果转换回输入类型，并按 V 的来源分别写入两个梯度输出。
    LocalTensor<T> output = inputBuf_.Get<T>();
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<T, float>::value) {
        // 使用矢量逐元素复制，支持最后一个非 32B 对齐的 H 尾块。
        Adds(output, src, 0.0f, length);
    } else if constexpr (std::is_same<T, half>::value) {
        Cast(output, src, RoundMode::CAST_NONE, length);
    } else {
        Cast(output, src, RoundMode::CAST_RINT, length);
    }
    SetFlag<HardEvent::V_MTE3>(eventVMte3_);
    WaitFlag<HardEvent::V_MTE3>(eventVMte3_);

    DataCopyExtParams copyParams(1, length * sizeof(T), 0, 0, 0);
    if (block < numBlocks_) {
        uint64_t offset =
            (static_cast<uint64_t>(batch) * static_cast<uint64_t>(numBlocks_) + static_cast<uint64_t>(block)) *
                static_cast<uint64_t>(hiddenSize_) +
            hStart;
        DataCopyPad(gradBlockResGm_[offset], output, copyParams);
    } else {
        uint64_t offset = static_cast<uint64_t>(batch) * static_cast<uint64_t>(hiddenSize_) + hStart;
        DataCopyPad(gradPartialBlockGm_[offset], output, copyParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradSplitH<T>::Init(
    GM_ADDR partialBlock, GM_ADDR blockRes, GM_ADDR projWeight, GM_ADDR normWeight, GM_ADDR gradHiddenState,
    GM_ADDR invNorm, GM_ADDR probs, GM_ADDR gradPartialBlock, GM_ADDR gradBlockRes, GM_ADDR gradProjWeight,
    GM_ADDR gradNormWeight, GM_ADDR workspace, const BlockAttentionResidualsGradTilingData *td)
{
    batchSize_ = td->batchSize;
    numBlocks_ = td->numBlocks;
    totalBlocks_ = td->totalBlocks;
    hiddenSize_ = td->hiddenSize;
    hiddenTileSize_ = td->hiddenTileSize;
    coreNum_ = td->coreNum;
    perCoreWkspBytes_ = td->perCoreWkspBytes;
    invH_ = 1.0f / static_cast<float>(hiddenSize_);

    // 多核沿 B 轴均分任务，余数 Token 依次分配给前 remainder 个核。
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t perCore = batchSize_ / coreNum_;
    int64_t remainder = batchSize_ % coreNum_;
    if (blockIdx < remainder) {
        coreBatchStart_ = blockIdx * (perCore + 1);
        coreBatchEnd_ = coreBatchStart_ + perCore + 1;
    } else {
        coreBatchStart_ = remainder * (perCore + 1) + (blockIdx - remainder) * perCore;
        coreBatchEnd_ = coreBatchStart_ + perCore;
    }

    uint64_t B = static_cast<uint64_t>(batchSize_);
    uint64_t N = static_cast<uint64_t>(numBlocks_);
    uint64_t K = static_cast<uint64_t>(totalBlocks_);
    uint64_t H = static_cast<uint64_t>(hiddenSize_);
    partialBlockGm_.SetGlobalBuffer((__gm__ T *)partialBlock, B * H);
    blockResGm_.SetGlobalBuffer((__gm__ T *)blockRes, B * N * H);
    projWeightGm_.SetGlobalBuffer((__gm__ T *)projWeight, H);
    normWeightGm_.SetGlobalBuffer((__gm__ T *)normWeight, H);
    gradHiddenStateGm_.SetGlobalBuffer((__gm__ T *)gradHiddenState, B * H);
    invNormGm_.SetGlobalBuffer((__gm__ float *)invNorm, B * K);
    probsGm_.SetGlobalBuffer((__gm__ float *)probs, B * K);
    gradPartialBlockGm_.SetGlobalBuffer((__gm__ T *)gradPartialBlock, B * H);
    gradBlockResGm_.SetGlobalBuffer((__gm__ T *)gradBlockRes, B * N * H);
    gradProjWeightGm_.SetGlobalBuffer((__gm__ T *)gradProjWeight, H);
    gradNormWeightGm_.SetGlobalBuffer((__gm__ T *)gradNormWeight, H);

    // Workspace 布局：
    // [coreNum, align512(H * sizeof(float))]：每核的 gradScoreWeight 部分和；
    // [B, K]：阶段一保存的 gradScore；
    // [B, K]：阶段一保存的 varianceScale。
    __gm__ float *userWorkspace = (__gm__ float *)GetUserWorkspace(workspace);
    uint64_t coreOffset = perCoreWkspBytes_ * static_cast<uint64_t>(blockIdx);
    coreWeightWkspGm_.SetGlobalBuffer(userWorkspace + coreOffset / sizeof(float), H);
    allWeightWkspGm_.SetGlobalBuffer(userWorkspace,
                                     perCoreWkspBytes_ * static_cast<uint64_t>(coreNum_) / sizeof(float));
    gradScoresWkspGm_.SetGlobalBuffer(userWorkspace + td->gradScoresWkspOff / sizeof(float), B * K);
    varianceScaleWkspGm_.SetGlobalBuffer(userWorkspace + td->varianceScaleWkspOff / sizeof(float), B * K);

    uint32_t tile = static_cast<uint32_t>(hiddenTileSize_);
    uint32_t metaBytes = (static_cast<uint32_t>(totalBlocks_) * sizeof(float) + META_ALIGN_BYTES - 1U) /
                         META_ALIGN_BYTES * META_ALIGN_BYTES;
    // H 相关 Buffer 按 hiddenTileSize_ 分配；K 相关 Buffer 始终完整驻留 UB。
    pipe_.InitBuffer(scoreWeightBuf_, tile * sizeof(float));
    pipe_.InitBuffer(inputBuf_, tile * sizeof(T));
    pipe_.InitBuffer(floatBuf0_, tile * sizeof(float));
    pipe_.InitBuffer(floatBuf1_, tile * sizeof(float));
    pipe_.InitBuffer(floatBuf2_, tile * sizeof(float));
    pipe_.InitBuffer(weightAccBuf_, tile * sizeof(float));
    pipe_.InitBuffer(weightCmpBuf_, tile * sizeof(float));
    pipe_.InitBuffer(probsBuf_, metaBytes);
    pipe_.InitBuffer(invNormBuf_, metaBytes);
    pipe_.InitBuffer(gradProbBuf_, metaBytes);
    pipe_.InitBuffer(gradScoreBuf_, metaBytes);
    pipe_.InitBuffer(gradProbCmpBuf_, metaBytes);
    pipe_.InitBuffer(weightedVDotCmpBuf_, metaBytes);
    pipe_.InitBuffer(reduceDstBuf_, 32U);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradSplitH<T>::ComputeAndSaveGradMeta()
{
    uint32_t K = static_cast<uint32_t>(totalBlocks_);
    LocalTensor<float> probs = probsBuf_.Get<float>();
    LocalTensor<float> gradProb = gradProbBuf_.Get<float>();
    LocalTensor<float> gradScore = gradScoreBuf_.Get<float>();
    LocalTensor<float> weightedVDot = invNormBuf_.Get<float>();
    LocalTensor<float> gradProbCmp = gradProbCmpBuf_.Get<float>();
    LocalTensor<float> weightedVDotCmp = weightedVDotCmpBuf_.Get<float>();
    LocalTensor<float> gradHidden = floatBuf0_.Get<float>();
    LocalTensor<float> v = floatBuf1_.Get<float>();
    LocalTensor<float> product = floatBuf2_.Get<float>();

    for (int64_t batch = coreBatchStart_; batch < coreBatchEnd_; ++batch) {
        uint64_t metaOffset = static_cast<uint64_t>(batch) * static_cast<uint64_t>(K);
        LoadMeta(probsGm_, metaOffset, K, probs);
        Duplicate(gradProb, 0.0f, K);
        Duplicate(weightedVDot, 0.0f, K);
        Duplicate(gradProbCmp, 0.0f, K);
        Duplicate(weightedVDotCmp, 0.0f, K);
        PipeBarrier<PIPE_V>();

        // 单次遍历 H：计算 gradProb，并暂存 sum(scoreWeight * V)。
        for (uint32_t hStart = 0; hStart < static_cast<uint32_t>(hiddenSize_);
             hStart += static_cast<uint32_t>(hiddenTileSize_)) {
            uint32_t length = static_cast<uint32_t>(hiddenSize_) - hStart;
            if (length > static_cast<uint32_t>(hiddenTileSize_)) {
                length = static_cast<uint32_t>(hiddenTileSize_);
            }
            InitScoreWeightTile(hStart, length);
            LoadTensorTile(gradHiddenStateGm_, static_cast<uint64_t>(batch) * hiddenSize_ + hStart, length, gradHidden);
            for (uint32_t block = 0; block < K; ++block) {
                LoadTensorTile(GetV(block, batch), hStart, length, v);

                Mul(product, gradHidden, v, length);
                ReduceSumHalfInterval(reduceDstBuf_.Get<float>(), product, static_cast<int32_t>(length));
                SetFlag<HardEvent::V_S>(eventVS_);
                WaitFlag<HardEvent::V_S>(eventVS_);
                float partial = reduceDstBuf_.Get<float>().GetValue(0);
                float sum = gradProb.GetValue(block);
                float corrected = partial - gradProbCmp.GetValue(block);
                float updated = sum + corrected;
                gradProbCmp.SetValue(block, (updated - sum) - corrected);
                gradProb.SetValue(block, updated);

                Mul(product, scoreWeightBuf_.Get<float>(), v, length);
                ReduceSumHalfInterval(reduceDstBuf_.Get<float>(), product, static_cast<int32_t>(length));
                SetFlag<HardEvent::V_S>(eventVS_);
                WaitFlag<HardEvent::V_S>(eventVS_);
                partial = reduceDstBuf_.Get<float>().GetValue(0);
                sum = weightedVDot.GetValue(block);
                corrected = partial - weightedVDotCmp.GetValue(block);
                updated = sum + corrected;
                weightedVDotCmp.SetValue(block, (updated - sum) - corrected);
                weightedVDot.SetValue(block, updated);
            }
        }

        SetFlag<HardEvent::S_V>(eventSV_);
        WaitFlag<HardEvent::S_V>(eventSV_);
        Mul(gradScore, probs, gradProb, K); // p * dp
        // ReduceSumHalfInterval 对 gradScore 原地折叠，K>64 时折叠后不能作为 p*dp 复用。
        ReduceSumHalfInterval(reduceDstBuf_.Get<float>(), gradScore, static_cast<int32_t>(K));
        SetFlag<HardEvent::V_S>(eventVS_);
        WaitFlag<HardEvent::V_S>(eventVS_);
        float dot = reduceDstBuf_.Get<float>().GetValue(0);
        // Match aten::_softmax_backward_data: p*dp - p*sum(p*dp).
        Mul(gradScore, probs, gradProb, K); // 重新计算 p*dp，避免使用被折叠的 gradScore
        Muls(probs, probs, dot, K);
        Sub(gradScore, gradScore, probs, K);

        // Keep the single-pass grad_inv_norm accumulation.
        Mul(weightedVDot, gradScore, weightedVDot, K);

        SetFlag<HardEvent::V_MTE2>(eventVMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventVMte2_);
        LoadMeta(invNormGm_, metaOffset, K, gradProb);
        Mul(probs, gradProb, gradProb, K);
        Mul(gradProb, probs, gradProb, K);
        Muls(weightedVDot, weightedVDot, -0.5f, K);
        Mul(weightedVDot, weightedVDot, gradProb, K);

        SetFlag<HardEvent::V_MTE3>(eventVMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3_);
        DataCopyExtParams copyParams(1, K * sizeof(float), 0, 0, 0);
        DataCopyPad(gradScoresWkspGm_[metaOffset], gradScore, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        DataCopyPad(varianceScaleWkspGm_[metaOffset], weightedVDot, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
    }
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradSplitH<T>::InitScoreWeightTile(uint32_t hStart, uint32_t length)
{
    // scoreWeight[h] = normWeight[h] * projWeight[h]。
    LocalTensor<float> scoreWeight = scoreWeightBuf_.Get<float>();
    LocalTensor<float> proj = floatBuf0_.Get<float>();
    LoadTensorTile(normWeightGm_, hStart, length, scoreWeight);
    LoadTensorTile(projWeightGm_, hStart, length, proj);
    Mul(scoreWeight, scoreWeight, proj, length);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradSplitH<T>::ComputeGradVAndWeightPartials()
{
    // 阶段二：以 H tile 为外层循环，计算当前核负责 Token 的 gradV，
    // 同时沿 B、K 累加当前核的 gradScoreWeight[h] 部分和。
    uint32_t K = static_cast<uint32_t>(totalBlocks_);
    LocalTensor<float> scoreWeight = scoreWeightBuf_.Get<float>();
    LocalTensor<float> gradHidden = floatBuf0_.Get<float>();
    LocalTensor<float> v = floatBuf1_.Get<float>();
    LocalTensor<float> gradV = floatBuf2_.Get<float>();
    LocalTensor<float> weightAcc = weightAccBuf_.Get<float>();
    LocalTensor<float> weightCmp = weightCmpBuf_.Get<float>();
    LocalTensor<float> probs = probsBuf_.Get<float>();
    LocalTensor<float> invNorm = invNormBuf_.Get<float>();
    LocalTensor<float> varianceScale = gradProbBuf_.Get<float>();
    LocalTensor<float> gradScore = gradScoreBuf_.Get<float>();

    for (uint32_t hStart = 0; hStart < static_cast<uint32_t>(hiddenSize_);
         hStart += static_cast<uint32_t>(hiddenTileSize_)) {
        uint32_t length = static_cast<uint32_t>(hiddenSize_) - hStart;
        if (length > static_cast<uint32_t>(hiddenTileSize_)) {
            length = static_cast<uint32_t>(hiddenTileSize_);
        }
        InitScoreWeightTile(hStart, length);
        Duplicate(weightAcc, 0.0f, length);
        Duplicate(weightCmp, 0.0f, length);
        PipeBarrier<PIPE_V>();
        int32_t kahanOutPos = 0;

        // 不同核的 batch 范围互不重叠，因此 gradV 可以直接写入最终输出，无需跨核归约。
        for (int64_t batch = coreBatchStart_; batch < coreBatchEnd_; ++batch) {
            uint64_t metaOffset = static_cast<uint64_t>(batch) * static_cast<uint64_t>(K);
            LoadMeta(probsGm_, metaOffset, K, probs);
            LoadMeta(invNormGm_, metaOffset, K, invNorm);
            LoadMeta(gradScoresWkspGm_, metaOffset, K, gradScore);
            LoadMeta(varianceScaleWkspGm_, metaOffset, K, varianceScale);

            for (uint32_t block = 0; block < K; ++block) {
                LoadTensorTile(gradHiddenStateGm_, static_cast<uint64_t>(batch) * hiddenSize_ + hStart, length,
                               gradHidden);
                LoadTensorTile(GetV(block, batch), hStart, length, v);

                SetFlag<HardEvent::V_S>(eventVS_);
                WaitFlag<HardEvent::V_S>(eventVS_);
                float probability = probs.GetValue(block);
                float scoreGrad = gradScore.GetValue(block);
                float inverseNorm = invNorm.GetValue(block);
                float scoreScale = scoreGrad * inverseNorm;
                float varianceScaleValue = varianceScale.GetValue(block);
                float meanScale = varianceScaleValue * invH_;

                // gradV 的前两部分：
                // probs * gradHiddenState + gradScore * invNorm * scoreWeight。

                // grad_out * probs    grad_v_from_out
                Muls(gradV, gradHidden, probability, length);
                // grad_v_from_k
                // Keep the reference operation order: (gradScore * scoreWeight) * invNorm.
                Muls(gradHidden, scoreWeight, scoreGrad, length);
                Muls(gradHidden, gradHidden, inverseNorm, length);
                Add(gradV, gradV, gradHidden, length);

                // grad_v_from_var = (grad_variance / H) * (2.0 * v)
                Muls(gradHidden, v, 2.0f, length);
                Muls(gradHidden, gradHidden, meanScale, length);
                Add(gradV, gradV, gradHidden, length);

                // gradScoreWeight[h] += gradScore * invNorm * V[h]，使用 Kahan 累加降低误差。
                Muls(gradHidden, v, scoreScale, length);
                LocalTensor<float> sumAndCmp[2] = {weightAcc, weightCmp};
                KahanSumUpdate(gradHidden, sumAndCmp, static_cast<int32_t>(length), kahanOutPos);
                StoreGradV(block, batch, hStart, length, gradV);
            }
        }

        LocalTensor<float> weightResult = weightAcc;
        if (kahanOutPos != 0) {
            weightResult = weightCmp;
        }
        // 每个核写入独立 Workspace 区域，等待阶段三进行跨核求和。
        SetFlag<HardEvent::V_MTE3>(eventVMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3_);
        DataCopyExtParams copyParams(1, length * sizeof(float), 0, 0, 0);
        DataCopyPad(coreWeightWkspGm_[hStart], weightResult, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
    }
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradSplitH<T>::ReduceAndWriteWeights()
{
    // 阶段三只由核 0 执行；其他核完成阶段二并参与 SyncAll 后直接返回。
    if (GetBlockIdx() != 0) {
        return;
    }
    LocalTensor<float> total = floatBuf0_.Get<float>();
    LocalTensor<float> temp = floatBuf1_.Get<float>();
    LocalTensor<float> output = floatBuf2_.Get<float>();
    LocalTensor<T> outputTyped = inputBuf_.Get<T>();

    for (uint32_t hStart = 0; hStart < static_cast<uint32_t>(hiddenSize_);
         hStart += static_cast<uint32_t>(hiddenTileSize_)) {
        uint32_t length = static_cast<uint32_t>(hiddenSize_) - hStart;
        if (length > static_cast<uint32_t>(hiddenTileSize_)) {
            length = static_cast<uint32_t>(hiddenTileSize_);
        }
        LocalTensor<float> totalCmp = weightCmpBuf_.Get<float>();
        Duplicate(total, 0.0f, length);
        Duplicate(totalCmp, 0.0f, length);
        PipeBarrier<PIPE_V>();
        int32_t kahanPos = 0;
        LocalTensor<float> sumCmp[2] = {total, totalCmp};
        uint64_t stride = perCoreWkspBytes_ / sizeof(float);
        // 只归约真正分到 token 的核；空闲核的全 0 部分和会破坏 Kahan 补偿。
        const int64_t reduceCores = (coreNum_ < batchSize_) ? coreNum_ : batchSize_;
        // 对所有核产生的 gradScoreWeight 部分执行 Kahan 补偿跨核归约。
        for (int64_t core = 0; core < reduceCores; ++core) {
            LoadMeta(allWeightWkspGm_, static_cast<uint64_t>(core) * stride + hStart, length, temp);
            KahanSumUpdate(temp, sumCmp, static_cast<int32_t>(length), kahanPos);
            SetFlag<HardEvent::V_MTE2>(eventVMte2_);
            WaitFlag<HardEvent::V_MTE2>(eventVMte2_);
        }
        // 偶数次更新后最终和仍在 total，奇数次在 totalCmp。
        if (kahanPos != 0) {
            Adds(total, totalCmp, 0.0f, length);
        }

        // gradNormWeight = gradScoreWeight * projWeight。
        LoadTensorTile(projWeightGm_, hStart, length, temp);
        Mul(output, total, temp, length);
        if constexpr (std::is_same<T, float>::value) {
            Adds(outputTyped, output, 0.0f, length);
        } else if constexpr (std::is_same<T, half>::value) {
            Cast(outputTyped, output, RoundMode::CAST_NONE, length);
        } else {
            Cast(outputTyped, output, RoundMode::CAST_RINT, length);
        }
        SetFlag<HardEvent::V_MTE3>(eventVMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3_);
        DataCopyExtParams typedCopy(1, length * sizeof(T), 0, 0, 0);
        DataCopyPad(gradNormWeightGm_[hStart], outputTyped, typedCopy);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);

        // gradProjWeight = gradScoreWeight * normWeight。
        LoadTensorTile(normWeightGm_, hStart, length, temp);
        Mul(output, total, temp, length);
        if constexpr (std::is_same<T, float>::value) {
            Adds(outputTyped, output, 0.0f, length);
        } else if constexpr (std::is_same<T, half>::value) {
            Cast(outputTyped, output, RoundMode::CAST_NONE, length);
        } else {
            Cast(outputTyped, output, RoundMode::CAST_RINT, length);
        }
        SetFlag<HardEvent::V_MTE3>(eventVMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3_);
        DataCopyPad(gradProjWeightGm_[hStart], outputTyped, typedCopy);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
    }
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradSplitH<T>::Process()
{
    eventMte2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    eventVMte2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    eventVMte3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    eventMte3Mte2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    eventVS_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    eventSV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    ComputeAndSaveGradMeta();        // 阶段一：完整 H 归约并保存 gradScore、varianceScale。
    ComputeGradVAndWeightPartials(); // 阶段二：计算 gradV 和每核权重梯度部分和。
    // 所有核必须完成 Workspace 写入后，核 0 才能读取并执行最终归约。
    SyncAll();
    ReduceAndWriteWeights(); // 阶段三：核 0 跨核归约并写出权重梯度。
}

} // namespace NsBlockAttentionResidualsGrad

#endif // BLOCK_ATTENTION_RESIDUALS_GRAD_SPLIT_H_SINGLE_PASS_BACKUP_H
