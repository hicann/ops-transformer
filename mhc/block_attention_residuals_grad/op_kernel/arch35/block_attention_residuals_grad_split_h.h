/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file arch35/block_attention_residuals_grad_split_h.h
 *  \brief A5 H-split implementation for shapes whose full-H RegBase working set does not fit in UB.
 */

#ifndef BLOCK_ATTENTION_RESIDUALS_GRAD_ARCH35_SPLIT_H_H
#define BLOCK_ATTENTION_RESIDUALS_GRAD_ARCH35_SPLIT_H_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../block_attention_residuals_grad_tiling_data.h"
#include "../block_attention_residuals_grad_tiling_key.h"
#include "block_attention_residuals_grad_regbase_common.h"

namespace NsBlockAttentionResidualsGrad {
using namespace AscendC;

// K 轴元数据按 32B 对齐搬入 UB。
constexpr uint32_t META_ALIGN_BYTES = 32U;

// SPLIT_H 仅在单核 UB 内按 H 轴分块，多核任务仍按 B 轴分配。
// 由于 gradScore 依赖完整 H 轴上的点积，Kernel 分为三个阶段：
// 1. 分块遍历 H，计算并保存 gradScore[B, N+1] 和 varianceScale[B, N+1]；
// 2. 再次分块遍历 H，计算 gradV，并生成每个核的权重梯度部分和；
// 3. 全核同步后，由核 0 归约各核部分和并写出最终权重梯度。
template <typename T>
class BlockAttentionResidualsGradA5SplitH {
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
    __aicore__ inline void LoadGradHiddenTile(uint64_t offset, uint32_t length, const LocalTensor<float> &dst);
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

    // A5 SPLIT_H 的 H tile Buffer 与 RegBase FULL_H 保持同构：
    // 4 个 dtype Buffer（inQue 双缓冲 + outQue + gradHiddenStateQue）；
    // 7 个 FP32 Buffer（6 个计算 Buffer + wkspOutQue）。
    TQue<QuePosition::VECIN, 2> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TQue<QuePosition::VECIN, 1> gradHiddenStateQue_;
    TQue<QuePosition::VECOUT, 1> wkspOutQue_;
    TBuf<TPosition::VECCALC> scoreWeightBuf_;
    TBuf<TPosition::VECCALC> floatBuf0_, floatBuf1_, floatBuf2_;
    TBuf<TPosition::VECCALC> weightAccBuf_, weightCmpBuf_;
    TBuf<TPosition::VECCALC> probsBuf_, invNormBuf_, gradProbBuf_, gradScoreBuf_;
    TBuf<TPosition::VECCALC> gradProbCmpBuf_, weightedVDotCmpBuf_, scalarBuf_;

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
__aicore__ inline GlobalTensor<T> BlockAttentionResidualsGradA5SplitH<T>::GetV(int64_t block, int64_t batch) const
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
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::LoadTensorTile(const GlobalTensor<T> &src,
                                                                              uint64_t offset, uint32_t length,
                                                                              const LocalTensor<float> &dst)
{
    // inQue 保留 RegBase FULL_H 的双缓冲布局，每次只搬入一个连续 H tile。
    // DeQue 已负责 MTE2->V 同步，FULL_H 的 CopyInRowSync 采用同一写法。
    LocalTensor<T> input = inQue_.AllocTensor<T>();
    DataCopyExtParams copyParams(1, length * sizeof(T), 0, 0, 0);
    DataCopyPad(input, src[offset], copyParams, {false, 0U, 0U, 0U});
    inQue_.EnQue(input);
    input = inQue_.DeQue<T>();
    if constexpr (IsSameType<T, float>::value) {
        // A5 fp32->fp32 Cast 不搬数据，改用矢量复制，与 arch22 保持一致。
        Adds(dst, input, 0.0f, length);
    } else {
        Cast(dst, input, RoundMode::CAST_NONE, length);
    }
    inQue_.FreeTensor(input);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::LoadGradHiddenTile(uint64_t offset, uint32_t length,
                                                                                  const LocalTensor<float> &dst)
{
    LocalTensor<T> input = gradHiddenStateQue_.AllocTensor<T>();
    DataCopyExtParams copyParams(1, length * sizeof(T), 0, 0, 0);
    DataCopyPad(input, gradHiddenStateGm_[offset], copyParams, {false, 0U, 0U, 0U});
    gradHiddenStateQue_.EnQue(input);
    input = gradHiddenStateQue_.DeQue<T>();
    if constexpr (IsSameType<T, float>::value) {
        Adds(dst, input, 0.0f, length);
    } else {
        Cast(dst, input, RoundMode::CAST_NONE, length);
    }
    gradHiddenStateQue_.FreeTensor(input);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::LoadMeta(const GlobalTensor<float> &src, uint64_t offset,
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
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::StoreGradV(int64_t block, int64_t batch, uint32_t hStart,
                                                                          uint32_t length,
                                                                          const LocalTensor<float> &src)
{
    // FP32 计算结果转换回输入类型，并按 V 的来源分别写入两个梯度输出。
    LocalTensor<T> output = outQue_.AllocTensor<T>();
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, float>::value) {
        Adds(output, src, 0.0f, length);
    } else {
        Cast(output, src, RoundMode::CAST_RINT, length);
    }
    outQue_.EnQue(output);
    output = outQue_.DeQue<T>();

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
    outQue_.FreeTensor(output);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::Init(
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
    constexpr uint32_t arch35UbAlignBytes = 256U;
    uint32_t metaBytes = (static_cast<uint32_t>(totalBlocks_) * sizeof(float) + arch35UbAlignBytes - 1U) /
                         arch35UbAlignBytes * arch35UbAlignBytes;
    // A5 H 相关 Buffer 按 hiddenTileSize_ 分配；K 相关 Buffer 完整驻留 UB，
    // 并与 Host 侧 CalcArch35HiddenTileSize 保持相同的 256B 对齐模型。
    pipe_.InitBuffer(inQue_, 2, tile * sizeof(T));
    pipe_.InitBuffer(outQue_, 1, tile * sizeof(T));
    pipe_.InitBuffer(gradHiddenStateQue_, 1, tile * sizeof(T));
    pipe_.InitBuffer(wkspOutQue_, 1, tile * sizeof(float));
    pipe_.InitBuffer(scoreWeightBuf_, tile * sizeof(float));
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
    pipe_.InitBuffer(scalarBuf_, 6U * RegBase::GRAD_SCALAR_LOCAL_ELEMS * sizeof(float));
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::ComputeAndSaveGradMeta()
{
    // 阶段一：先完成完整 H 轴归约，再执行 Softmax 反向并计算 varianceScale。
    // gradProb[b, k]  = sum_h(gradHiddenState[b, h] * V[b, k, h])
    // weightedVDot[b, k] = sum_h(scoreWeight[h] * V[b, k, h])
    // gradScore[b, k] = probs[b,k]*gradProb[b,k] - probs[b,k]*sum_j(probs[b,j]*gradProb[b,j])
    // varianceScale[b, k] = -gradScore[b, k] * weightedVDot[b, k] * invNorm[b, k]^3 / H
    uint32_t K = static_cast<uint32_t>(totalBlocks_);
    LocalTensor<float> probs = probsBuf_.Get<float>();
    LocalTensor<float> gradProb = gradProbBuf_.Get<float>();
    LocalTensor<float> gradScore = gradScoreBuf_.Get<float>();
    LocalTensor<float> weightedVDot = invNormBuf_.Get<float>();
    LocalTensor<float> gradProbCmp = gradProbCmpBuf_.Get<float>();
    LocalTensor<float> weightedVDotCmp = weightedVDotCmpBuf_.Get<float>();
    LocalTensor<float> gradHidden = floatBuf0_.Get<float>();
    LocalTensor<float> v = floatBuf1_.Get<float>();
    LocalTensor<float> scalar = scalarBuf_.Get<float>();

    for (int64_t batch = coreBatchStart_; batch < coreBatchEnd_; ++batch) {
        uint64_t metaOffset = static_cast<uint64_t>(batch) * static_cast<uint64_t>(K);
        LoadMeta(probsGm_, metaOffset, K, probs);
        Duplicate(gradProb, 0.0f, K);
        Duplicate(weightedVDot, 0.0f, K);
        Duplicate(gradProbCmp, 0.0f, K);
        Duplicate(weightedVDotCmp, 0.0f, K);
        PipeBarrier<PIPE_V>();

        // 各 H tile 的点积结果分别累加到 gradProb[K] 和 weightedVDot[K]，保证归约覆盖完整 H。
        for (uint32_t hStart = 0; hStart < static_cast<uint32_t>(hiddenSize_);
             hStart += static_cast<uint32_t>(hiddenTileSize_)) {
            uint32_t length = static_cast<uint32_t>(hiddenSize_) - hStart;
            if (length > static_cast<uint32_t>(hiddenTileSize_)) {
                length = static_cast<uint32_t>(hiddenTileSize_);
            }
            InitScoreWeightTile(hStart, length);
            LoadGradHiddenTile(static_cast<uint64_t>(batch) * hiddenSize_ + hStart, length, gradHidden);
            for (uint32_t block = 0; block < K; ++block) {
                LoadTensorTile(GetV(block, batch), hStart, length, v);
                Mul(floatBuf2_.Get<float>(), gradHidden, v, length);
                PipeBarrier<PIPE_V>();
                RegBase::ReduceMulSumTree(scalar, floatBuf2_.Get<float>(), length, eventSV_, eventVS_);
                SetFlag<HardEvent::V_S>(eventVS_);
                WaitFlag<HardEvent::V_S>(eventVS_);
                float partial = scalar.GetValue(0);
                // 跨 H tile 使用 Kahan 补偿求和，降低分块改变归约顺序带来的累计误差。
                float sum = gradProb.GetValue(block);
                float corrected = partial - gradProbCmp.GetValue(block);
                float updated = sum + corrected;
                gradProbCmp.SetValue(block, (updated - sum) - corrected);
                gradProb.SetValue(block, updated);

                Mul(floatBuf2_.Get<float>(), scoreWeightBuf_.Get<float>(), v, length);
                PipeBarrier<PIPE_V>();
                RegBase::ReduceMulSumTree(scalar, floatBuf2_.Get<float>(), length, eventSV_, eventVS_);
                SetFlag<HardEvent::V_S>(eventVS_);
                WaitFlag<HardEvent::V_S>(eventVS_);
                partial = scalar.GetValue(0);
                sum = weightedVDot.GetValue(block);
                corrected = partial - weightedVDotCmp.GetValue(block);
                updated = sum + corrected;
                weightedVDotCmp.SetValue(block, (updated - sum) - corrected);
                weightedVDot.SetValue(block, updated);
            }
        }

        SetFlag<HardEvent::S_V>(eventSV_);
        WaitFlag<HardEvent::S_V>(eventSV_);
        // A5 RegBase Softmax backward：gradScore = probs*gradProb - probs*sum(probs*gradProb)。
        Mul(gradScore, probs, gradProb, K);
        PipeBarrier<PIPE_V>();
        RegBase::ReduceMulSumTree(scalar, gradScore, K, eventSV_, eventVS_);
        PipeBarrier<PIPE_V>();
        RegBase::SoftmaxBackwardVf(gradScore, probs, gradProb, scalar, K);
        PipeBarrier<PIPE_V>();

        // gradProb 已完成使命，复用该 Buffer 加载 invNorm 并计算 invNorm^3。
        SetFlag<HardEvent::V_MTE2>(eventVMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventVMte2_);
        LoadMeta(invNormGm_, metaOffset, K, gradProb);
        Mul(probs, gradProb, gradProb, K);
        Mul(gradProb, probs, gradProb, K);
        // weightedVDot 原地转换为 varianceScale，避免增加新的 K 轴 UB Buffer。
        // 与标杆 FULL_H 保持相同计算顺序：gs*wv -> *invNorm^3 -> *(-invH)。
        Mul(weightedVDot, gradScore, weightedVDot, K);
        // 用probs的buffer存储了inv_rms
        Mul(weightedVDot, gradProb, weightedVDot, K);
        Muls(weightedVDot, weightedVDot, -invH_, K);

        // 阶段二处理任意 H tile 时都需要最终 gradScore 和 varianceScale，因此写入 GM Workspace。
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
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::InitScoreWeightTile(uint32_t hStart, uint32_t length)
{
    // scoreWeight[h] = normWeight[h] * projWeight[h]。
    LocalTensor<float> scoreWeight = scoreWeightBuf_.Get<float>();
    LocalTensor<float> proj = floatBuf0_.Get<float>();
    LoadTensorTile(normWeightGm_, hStart, length, scoreWeight);
    LoadTensorTile(projWeightGm_, hStart, length, proj);
    Mul(scoreWeight, scoreWeight, proj, length);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::ComputeGradVAndWeightPartials()
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
    LocalTensor<float> scalar = scalarBuf_.Get<float>();

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
        int32_t kahanPos = 0;
        LocalTensor<float> sumCmp[2] = {weightAcc, weightCmp};
        // 不同核的 batch 范围互不重叠，因此 gradV 可以直接写入最终输出，无需跨核归约。
        for (int64_t batch = coreBatchStart_; batch < coreBatchEnd_; ++batch) {
            uint64_t metaOffset = static_cast<uint64_t>(batch) * static_cast<uint64_t>(K);
            LoadMeta(probsGm_, metaOffset, K, probs);
            LoadMeta(invNormGm_, metaOffset, K, invNorm);
            LoadMeta(gradScoresWkspGm_, metaOffset, K, gradScore);
            LoadMeta(varianceScaleWkspGm_, metaOffset, K, varianceScale);

            for (uint32_t block = 0; block < K; ++block) {
                LoadGradHiddenTile(static_cast<uint64_t>(batch) * hiddenSize_ + hStart, length, gradHidden);
                LoadTensorTile(GetV(block, batch), hStart, length, v);

                SetFlag<HardEvent::V_S>(eventVS_);
                WaitFlag<HardEvent::V_S>(eventVS_);
                float probability = probs.GetValue(block);
                float scoreGrad = gradScore.GetValue(block);
                float inverseNorm = invNorm.GetValue(block);
                float scoreScale = scoreGrad * inverseNorm;
                float varianceScaleValue = varianceScale.GetValue(block);

                // RegBase scalar 布局与 FULL_H 保持一致：[0]=prob, [24]=scoreScale, [32]=varianceScale。
                scalar.SetValue(0, probability);
                scalar.SetValue(3U * RegBase::GRAD_ELEM_PER_BLK_FP32, scoreScale);
                scalar.SetValue(4U * RegBase::GRAD_ELEM_PER_BLK_FP32, varianceScaleValue);
                SetFlag<HardEvent::S_V>(eventSV_);
                WaitFlag<HardEvent::S_V>(eventSV_);

                // gradV = gradHidden * prob + scoreWeight * scoreScale + V * varianceScale。
                RegBase::FusedGradVF(gradV, gradHidden, v, scoreWeight, scalar, length);
                PipeBarrier<PIPE_V>();

                // gradScoreWeight += V * (gradScore * invNorm)，与 arch22 SPLIT_H 同序，
                // Kahan 使用 role-swap KahanSumUpdate。
                AscendC::Muls(v, v, scoreScale, length);
                RegBase::KahanSumUpdate(v, sumCmp, static_cast<int32_t>(length), kahanPos);
                PipeBarrier<PIPE_V>();
                StoreGradV(block, batch, hStart, length, gradV);
            }
        }

        // 每个核写入独立 Workspace 区域，等待阶段三进行跨核求和。
        LocalTensor<float> weightResult = wkspOutQue_.AllocTensor<float>();
        // role-swap Kahan 的最后"和"可能落在 weightAcc 或 weightCmp，按 kahanPos 选择。
        LocalTensor<float> gswSrc = weightAcc;
        if (kahanPos != 0) {
            gswSrc = weightCmp;
        }
        const uint32_t repeatTimes = (length + RegBase::kVlFp32 - 1U) / RegBase::kVlFp32;
        AscendC::Copy(weightResult, gswSrc, RegBase::kVlFp32, repeatTimes, {1, 1, 8, 8});
        wkspOutQue_.EnQue(weightResult);
        weightResult = wkspOutQue_.DeQue<float>();
        SetFlag<HardEvent::V_MTE3>(eventVMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3_);
        DataCopyExtParams copyParams(1, length * sizeof(float), 0, 0, 0);
        DataCopyPad(coreWeightWkspGm_[hStart], weightResult, copyParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        wkspOutQue_.FreeTensor(weightResult);
    }
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::ReduceAndWriteWeights()
{
    // 阶段三只由核 0 执行；其他核完成阶段二并参与 SyncAll 后直接返回。
    if (GetBlockIdx() != 0) {
        return;
    }
    LocalTensor<float> total = floatBuf0_.Get<float>();
    LocalTensor<float> temp = floatBuf1_.Get<float>();
    LocalTensor<float> output = floatBuf2_.Get<float>();

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
        // 对所有核产生的 gradScoreWeight 部分执行 Kahan 补偿跨核归约，与 arch22 一致。
        for (int64_t core = 0; core < reduceCores; ++core) {
            LoadMeta(allWeightWkspGm_, static_cast<uint64_t>(core) * stride + hStart, length, temp);
            RegBase::KahanSumUpdate(temp, sumCmp, static_cast<int32_t>(length), kahanPos);
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
        LocalTensor<T> outputTyped = outQue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            Adds(outputTyped, output, 0.0f, length);
        } else {
            Cast(outputTyped, output, RoundMode::CAST_RINT, length);
        }
        outQue_.EnQue(outputTyped);
        outputTyped = outQue_.DeQue<T>();
        SetFlag<HardEvent::V_MTE3>(eventVMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3_);
        DataCopyExtParams typedCopy(1, length * sizeof(T), 0, 0, 0);
        DataCopyPad(gradNormWeightGm_[hStart], outputTyped, typedCopy);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        outQue_.FreeTensor(outputTyped);

        // gradProjWeight = gradScoreWeight * normWeight。
        LoadTensorTile(normWeightGm_, hStart, length, temp);
        Mul(output, total, temp, length);
        outputTyped = outQue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            Adds(outputTyped, output, 0.0f, length);
        } else {
            Cast(outputTyped, output, RoundMode::CAST_RINT, length);
        }
        outQue_.EnQue(outputTyped);
        outputTyped = outQue_.DeQue<T>();
        SetFlag<HardEvent::V_MTE3>(eventVMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMte3_);
        DataCopyPad(gradProjWeightGm_[hStart], outputTyped, typedCopy);
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2_);
        outQue_.FreeTensor(outputTyped);
    }
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGradA5SplitH<T>::Process()
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

#endif // BLOCK_ATTENTION_RESIDUALS_GRAD_ARCH35_SPLIT_H_H
