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
 * \file arch22/block_attention_residuals_grad.h
 * \brief block_attention_residuals_grad kernel: B-axis split, outer B, inner N, H intact.
 *
 * Loop (per core):
 *   for batch in coreBatchStart_ .. coreBatchEnd_    (outer, B split)
 *       CopyInToken + PrecomputeGradProbs + SoftmaxBackwardN1
 *       for blk in 0 .. N+1  (inner, all blocks)
 *           ComputeGradV + CopyOutToken
 *   WriteGradScoreWeight (partial sum to workspace)
 *   SyncAll
 *   core 0: ReduceAndWriteWeights (cross-core sum, write grad_norm_weight / grad_proj_weight)
 */

#ifndef BLOCK_ATTENTION_RESIDUALS_GRAD_H
#define BLOCK_ATTENTION_RESIDUALS_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../block_attention_residuals_grad_tiling_data.h"
#include "../block_attention_residuals_grad_tiling_key.h"

namespace NsBlockAttentionResidualsGrad {
using namespace AscendC;

// ---- half-interval reduce，参照 attn_res_fwd/op_kernel/arch22/reduce_common.h ----
// src 就地 half-interval fold 后 Kahan 补偿求和归约到独立 dst[0]；无 sharedTmpBuffer。
constexpr uint32_t ARG_ELEM_PER_REP_FP32 = 64;
constexpr int32_t ARG_HALF_INTERVAL = 2;
constexpr int32_t ARG_INDEX_TWO = 2;
constexpr int32_t ARG_INDEX_FOUR = 4;
constexpr int32_t ARG_INDEX_EIGHT = 8;
constexpr int32_t ARG_INDEX_SIXTEEN = 16;
// K 轴元数据按 32B 对齐搬入 UB。
constexpr uint32_t META_ALIGN_BYTES = 32U;

__aicore__ inline int32_t ArgFindPowerTwo(int32_t n)
{
    n |= n >> 1;
    n |= n >> ARG_INDEX_TWO;
    n |= n >> ARG_INDEX_FOUR;
    n |= n >> ARG_INDEX_EIGHT;
    n |= n >> ARG_INDEX_SIXTEEN;
    return (n + 1) >> 1;
}

__aicore__ inline float ReduceSumKahan(const LocalTensor<float> &srcLocal, int32_t count);

__aicore__ inline void ReduceSumHalfInterval(const LocalTensor<float> &dstLocal, const LocalTensor<float> &srcLocal,
                                             int32_t count)
{
    if (likely(count > static_cast<int32_t>(ARG_ELEM_PER_REP_FP32))) {
        int32_t bodyCount = ArgFindPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(srcLocal, srcLocal, srcLocal[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > static_cast<int32_t>(ARG_ELEM_PER_REP_FP32)) {
            bodyCount = bodyCount / ARG_HALF_INTERVAL;
            Add(srcLocal, srcLocal, srcLocal[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }
    }
    // 最后对折半后的部分和做 Kahan 补偿求和，减少 FP32 归约舍入。
    const int32_t reduceCount =
        (count > static_cast<int32_t>(ARG_ELEM_PER_REP_FP32)) ? static_cast<int32_t>(ARG_ELEM_PER_REP_FP32) : count;
    PipeBarrier<PIPE_V>();
    dstLocal.SetValue(0, ReduceSumKahan(srcLocal, reduceCount));
}

// Kahan 补偿求和：H 较小（count<64）时逐标量累加，减少 FP32 归约的累积舍入。
// 调用方负责在向量写 src 之后、GetValue 之前做必要的同步。
__aicore__ inline float ReduceSumKahan(const LocalTensor<float> &srcLocal, int32_t count)
{
    float sum = 0.0f;
    float comp = 0.0f;
    for (int32_t i = 0; i < count; i++) {
        float y = srcLocal.GetValue(i) - comp;
        float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    return sum;
}

template <typename T>
__aicore__ inline void KahanSumUpdate(LocalTensor<T> &inputTensor, LocalTensor<T> sumTensorList[2], const int32_t len,
                                      int32_t &outPos)
{
    LocalTensor<T> sumTensor = sumTensorList[outPos];
    LocalTensor<T> eTensor = sumTensorList[1 - outPos];
    Sub(inputTensor, inputTensor, eTensor, len); // y = x - e
    Add(eTensor, inputTensor, sumTensor, len);   // t = y + s
    Sub(sumTensor, eTensor, sumTensor, len);     // e = t - s
    Sub(sumTensor, sumTensor, inputTensor, len); // e = (t - s) - y
    outPos = 1 - outPos;
}

template <typename T>
class BlockAttentionResidualsGrad {
public:
    __aicore__ inline BlockAttentionResidualsGrad(){};

    __aicore__ inline void Init(GM_ADDR prefixSum, GM_ADDR blockResidual, GM_ADDR projWeight, GM_ADDR normWeight,
                                GM_ADDR gradOutput, GM_ADDR invRms, GM_ADDR probs, GM_ADDR gradPrefixSum,
                                GM_ADDR gradBlockResidual, GM_ADDR gradProjWeight, GM_ADDR gradNormWeight,
                                GM_ADDR workspace, const BlockAttentionResidualsGradTilingData *td);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTiling(const BlockAttentionResidualsGradTilingData *td);
    __aicore__ inline GlobalTensor<T> VAddr(int64_t blk, int64_t batch) const;
    __aicore__ inline void InitScoreWeight();
    __aicore__ inline void CopyInToken(int64_t batch);
    __aicore__ inline void LoadGradOutAndV(int64_t blk);
    __aicore__ inline void PrecomputeGradProbs(int64_t batch);
    __aicore__ inline void SoftmaxBackwardN1();
    __aicore__ inline void ComputeGradV(int64_t blk);
    __aicore__ inline void CopyOutToken(int64_t blk, int64_t batch);
    __aicore__ inline void WriteGradScoreWeight();
    __aicore__ inline void ReduceAndWriteWeights();

    TPipe pipe;

    GlobalTensor<T> prefixSumGm, blockResGm, projWeightGm, normWeightGm;
    GlobalTensor<T> gradOutGm;
    GlobalTensor<float> invRmsGm, probsGm;
    GlobalTensor<T> gradPrefixSumGm, gradBlockResGm;
    GlobalTensor<T> gradProjWeightGm, gradNormWeightGm;
    GlobalTensor<float> gswWkspGm;
    GlobalTensor<float> wkspAllGm;

    TBuf<TPosition::VECCALC> scoreWeightBuf, halfBuf;
    TBuf<TPosition::VECCALC> floatBuf0, floatBuf1, floatBuf2, gswAccBuf;
    TBuf<TPosition::VECCALC> gswCmpBuf;
    TBuf<TPosition::VECCALC> probsFloatBuf, invrmsFloatBuf, gradProbBuf, gradScoreBuf;
    TBuf<TPosition::VECCALC> reduceDstBuf;

    int64_t batchSize_, numBlocks_, totalBlocks_, hiddenSize_;
    int64_t coreBatchStart_, coreBatchEnd_, coreBatchCount_;
    int64_t currentBatch_, coreNum_;
    int32_t kahanOutPos_;
    uint64_t perCoreWkspBytes_;
    float invH_;

    event_t eventMTE2V_;
    event_t eventVMTE2_;
    event_t eventMTE3MTE2_;
    event_t eventVMTE3_;
    event_t eventVS_;
    event_t eventSV_;
};

template <typename T>
__aicore__ inline GlobalTensor<T> BlockAttentionResidualsGrad<T>::VAddr(int64_t blk, int64_t batch) const
{
    if (blk < numBlocks_) {
        // 每个 block 占 hiddenSize_ 个元素，偏移需乘 H
        uint64_t blkOff =
            (static_cast<uint64_t>(batch) * static_cast<uint64_t>(numBlocks_) + static_cast<uint64_t>(blk)) *
            static_cast<uint64_t>(hiddenSize_);
        return blockResGm[blkOff];
    } else {
        // prefix_sum [B, H]，偏移 batch*H
        return prefixSumGm[static_cast<uint64_t>(batch) * static_cast<uint64_t>(hiddenSize_)];
    }
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::ParseTiling(const BlockAttentionResidualsGradTilingData *td)
{
    batchSize_ = td->batchSize;
    numBlocks_ = td->numBlocks;
    totalBlocks_ = td->totalBlocks;
    hiddenSize_ = td->hiddenSize;
    coreBatchStart_ = td->coreBatchStart;
    coreBatchEnd_ = td->coreBatchEnd;
    coreBatchCount_ = td->coreBatchCount;
    invH_ = 1.0f / static_cast<float>(hiddenSize_);
    currentBatch_ = 0;
    coreNum_ = td->coreNum;
    kahanOutPos_ = 0;
    perCoreWkspBytes_ = td->perCoreWkspBytes;
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::Init(GM_ADDR prefixSum, GM_ADDR blockResidual,
                                                            GM_ADDR projWeight, GM_ADDR normWeight, GM_ADDR gradOutput,
                                                            GM_ADDR invRms, GM_ADDR probs, GM_ADDR gradPrefixSum,
                                                            GM_ADDR gradBlockResidual, GM_ADDR gradProjWeight,
                                                            GM_ADDR gradNormWeight, GM_ADDR workspace,
                                                            const BlockAttentionResidualsGradTilingData *td)
{
    ParseTiling(td);

    __gm__ float *userWksp = (__gm__ float *)AscendC::GetUserWorkspace(workspace);

    // per-core batch range is computed in kernel, not host tiling
    int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t perCore = batchSize_ / coreNum_;
    int64_t rem = batchSize_ % coreNum_;
    if (blockIdx < rem) {
        coreBatchStart_ = blockIdx * (perCore + 1);
        coreBatchEnd_ = coreBatchStart_ + perCore + 1;
    } else {
        coreBatchStart_ = rem * (perCore + 1) + (blockIdx - rem) * perCore;
        coreBatchEnd_ = coreBatchStart_ + perCore;
    }
    coreBatchCount_ = coreBatchEnd_ - coreBatchStart_;

    uint64_t B = static_cast<uint64_t>(batchSize_);
    uint64_t N = static_cast<uint64_t>(numBlocks_);
    uint64_t H = static_cast<uint64_t>(hiddenSize_);
    uint64_t N1 = static_cast<uint64_t>(totalBlocks_);

    prefixSumGm.SetGlobalBuffer((__gm__ T *)prefixSum, B * H);
    blockResGm.SetGlobalBuffer((__gm__ T *)blockResidual, B * N * H);
    projWeightGm.SetGlobalBuffer((__gm__ T *)projWeight, H);
    normWeightGm.SetGlobalBuffer((__gm__ T *)normWeight, H);
    gradOutGm.SetGlobalBuffer((__gm__ T *)gradOutput, B * H);
    invRmsGm.SetGlobalBuffer((__gm__ float *)invRms, B * N1);
    probsGm.SetGlobalBuffer((__gm__ float *)probs, B * N1);
    gradPrefixSumGm.SetGlobalBuffer((__gm__ T *)gradPrefixSum, B * H);
    gradBlockResGm.SetGlobalBuffer((__gm__ T *)gradBlockResidual, B * N * H);
    gradProjWeightGm.SetGlobalBuffer((__gm__ T *)gradProjWeight, H);
    gradNormWeightGm.SetGlobalBuffer((__gm__ T *)gradNormWeight, H);
    uint64_t wkspOff = perCoreWkspBytes_ * static_cast<uint64_t>(blockIdx);
    gswWkspGm.SetGlobalBuffer(userWksp + (wkspOff / sizeof(float)), H);

    uint64_t totalWkspBytes = td->perCoreWkspBytes * static_cast<uint64_t>(td->coreNum);
    wkspAllGm.SetGlobalBuffer(userWksp, totalWkspBytes / sizeof(float));

    uint32_t HU = static_cast<uint32_t>(H);
    uint32_t N1U = static_cast<uint32_t>(N1);

    // H 轴 buffer 至少预留 64 个元素，保证 64-lane 的 WholeReduceSum 不越界。
    const uint32_t hBufElements = (HU > ARG_ELEM_PER_REP_FP32) ? HU : ARG_ELEM_PER_REP_FP32;
    uint32_t n1FloatBytes = (N1U * sizeof(float) + META_ALIGN_BYTES - 1U) / META_ALIGN_BYTES * META_ALIGN_BYTES;
    uint32_t n1AlignedFloats = n1FloatBytes / sizeof(float);
    // floatBuf2 在 SoftmaxBackwardN1 中按 N1 长度使用，容量不能小于 N1。
    const uint32_t maxBufFloats = (hBufElements > n1AlignedFloats) ? hBufElements : n1AlignedFloats;
    pipe.InitBuffer(scoreWeightBuf, hBufElements * sizeof(float));
    pipe.InitBuffer(halfBuf, hBufElements * sizeof(T));
    pipe.InitBuffer(floatBuf0, hBufElements * sizeof(float));
    pipe.InitBuffer(floatBuf1, hBufElements * sizeof(float));
    pipe.InitBuffer(floatBuf2, maxBufFloats * sizeof(float));
    pipe.InitBuffer(gswAccBuf, hBufElements * sizeof(float));
    pipe.InitBuffer(gswCmpBuf, hBufElements * sizeof(float));
    pipe.InitBuffer(probsFloatBuf, n1FloatBytes);
    pipe.InitBuffer(invrmsFloatBuf, n1FloatBytes);
    pipe.InitBuffer(gradProbBuf, n1FloatBytes);
    pipe.InitBuffer(gradScoreBuf, n1FloatBytes);
    pipe.InitBuffer(reduceDstBuf, 32U); // ReduceSumHalfInterval 的独立 dst（1 个 float）
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::InitScoreWeight()
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);

    LocalTensor<float> sw = scoreWeightBuf.Get<float>();
    LocalTensor<T> nh = halfBuf.Get<T>();

    DataCopyExtParams cp(1, H * sizeof(T), 0, 0, 0);
    DataCopyPad(nh, normWeightGm, cp, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    if constexpr (std::is_same<T, float>::value) {
        Adds(sw, nh, 0.0f, H);
    } else {
        Cast(sw, nh, RoundMode::CAST_NONE, H);
    }
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

    LocalTensor<T> ph = halfBuf.Get<T>();
    LocalTensor<float> pf = floatBuf0.Get<float>();

    DataCopyPad(ph, projWeightGm, cp, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    if constexpr (std::is_same<T, float>::value) {
        Adds(pf, ph, 0.0f, H);
    } else {
        Cast(pf, ph, RoundMode::CAST_NONE, H);
    }
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);
    Mul(sw, sw, pf, H);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::CopyInToken(int64_t batch)
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);
    uint32_t N1U = static_cast<uint32_t>(totalBlocks_);

    uint64_t gmOff = static_cast<uint64_t>(batch) * totalBlocks_;

    LocalTensor<T> hb = halfBuf.Get<T>();
    LocalTensor<float> pf = probsFloatBuf.Get<float>();
    LocalTensor<float> irf = invrmsFloatBuf.Get<float>();

    DataCopyExtParams cpH(1, H * sizeof(T), 0, 0, 0);
    DataCopyExtParams cpN1(1, N1U * sizeof(float), 0, 0, 0);
    uint32_t n1RightPad =
        ((N1U * sizeof(float) + META_ALIGN_BYTES - 1U) / META_ALIGN_BYTES * META_ALIGN_BYTES - N1U * sizeof(float)) /
        sizeof(float);

    DataCopyPad(hb, gradOutGm[static_cast<uint64_t>(batch) * hiddenSize_], cpH, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);

    // probs/inv_rms 输入为 FP32（正向算子输出），直接搬入 float buffer，不再经 BF16 量化。
    DataCopyPad(pf, probsGm[gmOff], cpN1, {true, 0U, static_cast<uint8_t>(n1RightPad), 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);

    DataCopyPad(irf, invRmsGm[gmOff], cpN1, {true, 0U, static_cast<uint8_t>(n1RightPad), 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::LoadGradOutAndV(int64_t blk)
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);

    LocalTensor<T> hb = halfBuf.Get<T>();
    LocalTensor<float> fb0 = floatBuf0.Get<float>();
    LocalTensor<float> fb1 = floatBuf1.Get<float>();

    DataCopyExtParams cp(1, H * sizeof(T), 0, 0, 0);

    DataCopyPad(hb, gradOutGm[static_cast<uint64_t>(currentBatch_) * hiddenSize_], cp, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    if constexpr (std::is_same<T, float>::value) {
        Adds(fb0, hb, 0.0f, H);
    } else {
        Cast(fb0, hb, RoundMode::CAST_NONE, H);
    }
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

    DataCopyPad(hb, VAddr(blk, currentBatch_), cp, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    if constexpr (std::is_same<T, float>::value) {
        Adds(fb1, hb, 0.0f, H);
    } else {
        Cast(fb1, hb, RoundMode::CAST_NONE, H);
    }
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::PrecomputeGradProbs(int64_t batch)
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);
    uint32_t N1U = static_cast<uint32_t>(totalBlocks_);

    LocalTensor<float> fb0 = floatBuf0.Get<float>();
    LocalTensor<float> fb1 = floatBuf1.Get<float>();
    LocalTensor<float> fb2 = floatBuf2.Get<float>();
    LocalTensor<T> hb = halfBuf.Get<T>();
    LocalTensor<float> gp = gradProbBuf.Get<float>();

    // grad_out was copied to hb in CopyInToken.
    uint32_t n1AlignedFloats =
        (N1U * sizeof(float) + META_ALIGN_BYTES - 1U) / META_ALIGN_BYTES * META_ALIGN_BYTES / sizeof(float);
    Duplicate(gp, 0.0f, static_cast<int32_t>(n1AlignedFloats));
    SetFlag<HardEvent::V_S>(eventVS_); // Duplicate -> scalar SetValue
    WaitFlag<HardEvent::V_S>(eventVS_);
    if constexpr (std::is_same<T, float>::value) {
        Adds(fb0, hb, 0.0f, H);
    } else {
        Cast(fb0, hb, RoundMode::CAST_NONE, H);
    }
    SetFlag<HardEvent::V_MTE2>(eventVMTE2_); // V has consumed hb; MTE2 may overwrite it.
    WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

    for (uint32_t bb = 0; bb < N1U; bb++) {
        DataCopyExtParams cp(1, H * sizeof(T), 0, 0, 0);
        DataCopyPad(hb, VAddr(static_cast<int64_t>(bb), batch), cp, {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
        if constexpr (std::is_same<T, float>::value) {
            Adds(fb1, hb, 0.0f, H);
        } else {
            Cast(fb1, hb, RoundMode::CAST_NONE, H);
        }
        SetFlag<HardEvent::V_MTE2>(eventVMTE2_); // V has consumed hb; MTE2 may overwrite it.
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

        Mul(fb2, fb1, fb0, H);
        SetFlag<HardEvent::V_S>(eventVS_); // vector -> scalar
        WaitFlag<HardEvent::V_S>(eventVS_);
        float gpVal;
        if (H < ARG_ELEM_PER_REP_FP32) {
            // 小 H：Kahan 补偿累加，减少 FP32 归约舍入。
            gpVal = ReduceSumKahan(fb2, static_cast<int32_t>(H));
        } else {
            // 大 H：沿用 half-interval + WholeReduceSum 二分归约。
            ReduceSumHalfInterval(reduceDstBuf.Get<float>(), fb2, static_cast<int32_t>(H));
            SetFlag<HardEvent::V_S>(eventVS_);
            WaitFlag<HardEvent::V_S>(eventVS_);
            gpVal = reduceDstBuf.Get<float>().GetValue(0);
        }
        gp.SetValue(bb, gpVal);
    }
    SetFlag<HardEvent::S_V>(eventSV_); // Scalar writes to gp are visible to vector.
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::SoftmaxBackwardN1()
{
    uint32_t N1 = static_cast<uint32_t>(totalBlocks_);

    LocalTensor<float> probsF = probsFloatBuf.Get<float>();
    LocalTensor<float> gp = gradProbBuf.Get<float>();
    LocalTensor<float> gs = gradScoreBuf.Get<float>();
    LocalTensor<float> tmp = floatBuf2.Get<float>();

    WaitFlag<HardEvent::S_V>(eventSV_); // Wait for scalar writes to gp.
    uint32_t n1AlignedFloats =
        (N1 * sizeof(float) + META_ALIGN_BYTES - 1U) / META_ALIGN_BYTES * META_ALIGN_BYTES / sizeof(float);
    Duplicate(gs, 0.0f, static_cast<int32_t>(n1AlignedFloats));
    Duplicate(tmp, 0.0f, static_cast<int32_t>(n1AlignedFloats));
    Mul(tmp, probsF, gp, N1);
    // ReduceSumHalfInterval 对 tmp 原地折叠，N1>64 时折叠后不能再作为 p*gp 复用。
    ReduceSumHalfInterval(reduceDstBuf.Get<float>(), tmp, static_cast<int32_t>(N1));
    SetFlag<HardEvent::V_S>(eventVS_); // ReduceSum -> GetValue
    WaitFlag<HardEvent::V_S>(eventVS_);
    float scalar = reduceDstBuf.Get<float>().GetValue(0);

    // Match aten::_softmax_backward_data: p*dp - p*sum(p*dp).
    Muls(gs, probsF, scalar, N1);
    Mul(tmp, probsF, gp, N1); // 重新计算 p*dp，避免使用被折叠的 tmp
    Sub(tmp, tmp, gs, N1);
    Adds(gs, tmp, 0.0f, N1);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::ComputeGradV(int64_t blk)
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);
    uint32_t blkU = static_cast<uint32_t>(blk);

    LocalTensor<float> sw = scoreWeightBuf.Get<float>();
    LocalTensor<float> fb0 = floatBuf0.Get<float>();
    LocalTensor<float> fb1 = floatBuf1.Get<float>();
    LocalTensor<float> fb2 = floatBuf2.Get<float>();
    LocalTensor<float> gswA = gswAccBuf.Get<float>();
    LocalTensor<float> gswC = gswCmpBuf.Get<float>();
    LocalTensor<float> probsF = probsFloatBuf.Get<float>();
    LocalTensor<float> invrF = invrmsFloatBuf.Get<float>();
    LocalTensor<float> gs = gradScoreBuf.Get<float>();

    // Load grad_output and v[batch, blk, :]
    LoadGradOutAndV(blk);

    // grad_v_from_out = grad_out * probs[blk]
    SetFlag<HardEvent::V_S>(eventVS_); // Vector writes are visible before scalar GetValue.
    WaitFlag<HardEvent::V_S>(eventVS_);
    float probBlk = probsF.GetValue(blkU);
    Muls(fb2, fb0, probBlk, H);

    float gsBlk = gs.GetValue(blkU);
    float irBlk = invrF.GetValue(blkU);

    // grad_v_from_k = (gradScore * inv_rms) * score_weight
    // Keep the reference operation order: (gradScore * scoreWeight) * invNorm.
    Muls(fb0, sw, gsBlk, H);
    Muls(fb0, fb0, irBlk, H);
    Add(fb2, fb2, fb0, H);
    // grad_inv_rms = sum((gradScore * score_weight) * v), matching the reference.
    Muls(fb0, sw, gsBlk, H);
    Mul(fb0, fb0, fb1, H);
    SetFlag<HardEvent::V_S>(eventVS_); // vector -> scalar
    WaitFlag<HardEvent::V_S>(eventVS_);
    float gradInvRms;
    if (H < ARG_ELEM_PER_REP_FP32) {
        // 小 H：Kahan 补偿累加，减少 FP32 归约舍入。
        gradInvRms = ReduceSumKahan(fb0, static_cast<int32_t>(H));
    } else {
        // 大 H：沿用 half-interval + WholeReduceSum 二分归约。
        ReduceSumHalfInterval(reduceDstBuf.Get<float>(), fb0, static_cast<int32_t>(H));
        SetFlag<HardEvent::V_S>(eventVS_);
        WaitFlag<HardEvent::V_S>(eventVS_);
        gradInvRms = reduceDstBuf.Get<float>().GetValue(0);
    }

    // grad_variance = grad_inv_rms * (-0.5) * inv_rms^3
    float r3 = irBlk * irBlk * irBlk;
    float gvar = gradInvRms * (-0.5f) * r3;

    float varScale = gvar / static_cast<float>(hiddenSize_);

    // grad_v_from_var = (grad_variance / H) * (2.0 * v)
    // 一次标量乘避免两次向量乘的舍入累积：v * ((2.0 * grad_variance) / H)
    float varScale2 = (2.0f * gvar) / static_cast<float>(hiddenSize_);
    Muls(fb0, fb1, varScale2, H);
    Add(fb2, fb2, fb0, H);

    // grad_score_weight partial accumulation = (v * invNorm) * gradScore (Kahan summation)
    Muls(fb0, fb1, irBlk, H);
    Muls(fb0, fb0, gsBlk, H);
    LocalTensor<float> sumCmp[2] = {gswA, gswC};
    KahanSumUpdate(fb0, sumCmp, static_cast<int32_t>(H), kahanOutPos_);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::CopyOutToken(int64_t blk, int64_t batch)
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);

    LocalTensor<float> fb2 = floatBuf2.Get<float>();
    LocalTensor<T> hb = halfBuf.Get<T>();

    PipeBarrier<PIPE_V>(); // Cast must not read fb2 before vector writes complete.
    if constexpr (std::is_same<T, half>::value) {
        Cast(hb, fb2, RoundMode::CAST_RINT, H);
    } else if constexpr (std::is_same<T, float>::value) {
        Adds(hb, fb2, 0.0f, H);
    } else {
        Cast(hb, fb2, RoundMode::CAST_RINT, H);
    }
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_); // Cast writes hb; MTE3 may read it.
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);

    DataCopyExtParams cp(1, H * sizeof(T), 0, 0, 0);
    if (blk < numBlocks_) {
        uint64_t gmOff = (static_cast<uint64_t>(batch) * numBlocks_ + static_cast<uint64_t>(blk)) * hiddenSize_;
        DataCopyPad(gradBlockResGm[gmOff], hb, cp);
    } else {
        uint64_t gmOff = static_cast<uint64_t>(batch) * hiddenSize_;
        DataCopyPad(gradPrefixSumGm[gmOff], hb, cp);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::WriteGradScoreWeight()
{
    uint32_t H = static_cast<uint32_t>(hiddenSize_);
    DataCopyExtParams cp(1, H * sizeof(float), 0, 0, 0);
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_); // Kahan accumulation is complete; MTE3 may read it.
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
    if (kahanOutPos_ == 0) {
        DataCopyPad(gswWkspGm, gswAccBuf.Get<float>(), cp);
    } else {
        DataCopyPad(gswWkspGm, gswCmpBuf.Get<float>(), cp);
    }
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::ReduceAndWriteWeights()
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    uint32_t H = static_cast<uint32_t>(hiddenSize_);

    LocalTensor<float> total = floatBuf0.Get<float>();
    LocalTensor<float> temp = floatBuf1.Get<float>();
    LocalTensor<float> out = floatBuf2.Get<float>();
    LocalTensor<T> hb = halfBuf.Get<T>();
    LocalTensor<float> gswA = gswAccBuf.Get<float>();
    LocalTensor<float> gswC = gswCmpBuf.Get<float>();

    DataCopyExtParams cpF(1, H * sizeof(float), 0, 0, 0);
    DataCopyExtParams cpT(1, H * sizeof(T), 0, 0, 0);

    Duplicate<float>(gswA, 0.0f, static_cast<int32_t>(H));
    Duplicate<float>(gswC, 0.0f, static_cast<int32_t>(H));
    int32_t kahanPos = 0;
    LocalTensor<float> sumCmp[2] = {gswA, gswC};

    // cross-core reduce: Kahan 补偿累加各核的 gradScoreWeight 部分和
    uint64_t strideFloats = perCoreWkspBytes_ / sizeof(float);
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0U};
    // 只归约真正分到 token 的核：核数大于 batch 时，空闲核的全 0 部分和会破坏
    // KahanSumUpdate 的补偿，导致权重梯度在抵消点差几个 ulp。
    const int64_t reduceCores = (coreNum_ < batchSize_) ? coreNum_ : batchSize_;
    for (int64_t c = 0; c < reduceCores; c++) {
        uint64_t cOff = static_cast<uint64_t>(c) * strideFloats;
        DataCopyPad(temp, wkspAllGm[cOff], cpF, padParams);
        SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
        KahanSumUpdate(temp, sumCmp, static_cast<int32_t>(H), kahanPos);
        SetFlag<HardEvent::V_MTE2>(eventVMTE2_); // Kahan has consumed temp; MTE2 may overwrite it.
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);
    }
    // 偶数次更新后最终和仍在 gswA，奇数次在 gswC
    if (kahanPos == 0) {
        Adds(total, gswA, 0.0f, H);
    } else {
        Adds(total, gswC, 0.0f, H);
    }

    // grad_norm_weight = total * proj_w
    LocalTensor<T> ph = halfBuf.Get<T>();
    DataCopyPad(ph, projWeightGm, cpT, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    if constexpr (std::is_same<T, float>::value) {
        Adds(temp, ph, 0.0f, H);
    } else {
        Cast(temp, ph, RoundMode::CAST_NONE, H);
    }
    Mul(out, total, temp, H);
    if constexpr (std::is_same<T, half>::value) {
        Cast(hb, out, RoundMode::CAST_RINT, H);
    } else if constexpr (std::is_same<T, float>::value) {
        Adds(hb, out, 0.0f, H);
    } else {
        Cast(hb, out, RoundMode::CAST_RINT, H);
    }
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_); // Cast writes hb; MTE3 may read it.
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
    DataCopyPad(gradNormWeightGm, hb, cpT);
    // 上一段 MTE3 读 hb 必须先完成，下一段 MTE2 才能覆写同一 halfBuf，否则输出被污染
    SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);

    // grad_proj_weight = total * norm_w
    DataCopyPad(ph, normWeightGm, cpT, {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
    if constexpr (std::is_same<T, float>::value) {
        Adds(temp, ph, 0.0f, H);
    } else {
        Cast(temp, ph, RoundMode::CAST_NONE, H);
    }
    Mul(out, total, temp, H);
    if constexpr (std::is_same<T, half>::value) {
        Cast(hb, out, RoundMode::CAST_RINT, H);
    } else if constexpr (std::is_same<T, float>::value) {
        Adds(hb, out, 0.0f, H);
    } else {
        Cast(hb, out, RoundMode::CAST_RINT, H);
    }
    SetFlag<HardEvent::V_MTE3>(eventVMTE3_); // Cast writes hb; MTE3 may read it.
    WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
    DataCopyPad(gradProjWeightGm, hb, cpT);
}

template <typename T>
__aicore__ inline void BlockAttentionResidualsGrad<T>::Process()
{
    eventMTE2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    eventVMTE2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    eventMTE3MTE2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    eventVMTE3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    eventVS_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    eventSV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    InitScoreWeight();
    {
        uint32_t H = static_cast<uint32_t>(hiddenSize_);
        Duplicate<float>(gswAccBuf.Get<float>(), 0.0f, static_cast<int32_t>(H));
        Duplicate<float>(gswCmpBuf.Get<float>(), 0.0f, static_cast<int32_t>(H));
    }

    int64_t N1 = totalBlocks_;
    for (int64_t batch = coreBatchStart_; batch < coreBatchEnd_; batch++) {
        currentBatch_ = batch;

        CopyInToken(batch);
        PrecomputeGradProbs(batch);
        SoftmaxBackwardN1();

        for (int64_t blk = 0; blk < N1; blk++) {
            ComputeGradV(blk);
            CopyOutToken(blk, batch);
        }
    }

    WriteGradScoreWeight();
    SyncAll();
    ReduceAndWriteWeights();
}

} // namespace NsBlockAttentionResidualsGrad
#endif // BLOCK_ATTENTION_RESIDUALS_GRAD_H
