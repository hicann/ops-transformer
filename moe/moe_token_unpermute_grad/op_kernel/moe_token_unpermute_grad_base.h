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
 * \file moe_token_unpermute_grad_base.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_GRAD_BASE_H
#define MOE_TOKEN_UNPERMUTE_GRAD_BASE_H
#include "kernel_operator.h"

namespace MoeTokenUnpermuteGrad {
using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int64_t BUFFER_32B_CALNUM = 32 / sizeof(float);
constexpr int64_t BLOCK_SIZE_512 = 512;
constexpr int64_t BLOCK_SIZE_256 = 256;
constexpr int64_t BLOCK_SIZE_32 = 32;
constexpr int64_t FP32_ONE_REPEAT = 64;
constexpr int64_t INDICES_PROBS_MAX_RESERVE_NUM = 512;

template <typename OriT, typename IdxT, typename ProbT = OriT>
class MoeTokenUnpermuteGradBase {
public:
    __aicore__ inline MoeTokenUnpermuteGradBase(){};
    __aicore__ inline void Init(GM_ADDR permuted_tokens, GM_ADDR unpermuted_tokens_grad, GM_ADDR sorted_indices,
                                GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
                                const MoeTokenUnpermuteGradTilingData &tilingData);
    __aicore__ inline int32_t AlignUp(int32_t a, int32_t b);
    __aicore__ inline int32_t CeilDiv(int32_t a, int32_t b);
    __aicore__ inline void BinaryAddFunc(LocalTensor<float> tmpBuffer, int32_t hiddensizeLen, int32_t threshold,
                                         int32_t offset);
    __aicore__ inline void ReduceSumFunc(LocalTensor<float> dstBuffer, LocalTensor<float> tmpBuffer,
                                         int32_t hiddensizeLen);

protected:
    TPipe pipe;
    GlobalTensor<OriT> permutedTokensGm;
    GlobalTensor<OriT> unpermutedOutputDGm;
    GlobalTensor<IdxT> sortedIndicesGm;
    GlobalTensor<OriT> permutedTokensGradGm;
    GlobalTensor<ProbT> probGm;
    GlobalTensor<ProbT> probGradGm;

    int64_t tokensNum;
    int64_t topK;
    int64_t hiddenSize;
    int64_t formerCoreNum;
    int64_t tailCoreNum;
    int64_t tokenNumEachCore;
    int64_t tokenNumTailCore;
    int64_t rowIdMapEachCore;
    int64_t rowIdMapTailCore;
    int64_t inputReserveNum;
    int64_t indicesReserveNum;
    int64_t indicesReserveNumAlign;
    int64_t coreIndex;
    int64_t rowIdMapStartOffset;
    int64_t hiddenSizeAlign;
    int64_t hiddenSizeLoopTimes;
    int64_t hiddenSizeTail;
    int64_t numOutTokens;
    int64_t unpermutedOutputDStartOffset;
    uint32_t inputTypeSize;
    uint32_t rowIdMapTypeSize;
    uint32_t probTypeSize;
    int64_t indicesReserveNumAlignFp32RepeatTimes;
    int64_t indicesReserveNumAlignFp32TailMask;
    int64_t indicesReserveNumAlignFp32TailOffset;
    int64_t hiddensizeAlignFp32RepeatTimes;
    int64_t hiddensizeAlignFp32TailMask;
    int64_t hiddensizeAlignFp32TailOffset;

    DataCopyPadExtParams<IdxT> rowIdMapPadParams{false, 0, 0, 0};
    DataCopyPadExtParams<OriT> inputPadParams{false, 0, 0, 0};
    DataCopyPadExtParams<ProbT> probPadParams{false, 0, 0, 0};
};

template <typename OriT, typename IdxT, typename ProbT>
__aicore__ inline int32_t MoeTokenUnpermuteGradBase<OriT, IdxT, ProbT>::AlignUp(int32_t a, int32_t b)
{
    if (unlikely(b == 0)) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename OriT, typename IdxT, typename ProbT>
__aicore__ inline int32_t MoeTokenUnpermuteGradBase<OriT, IdxT, ProbT>::CeilDiv(int32_t a, int32_t b)
{
    if (unlikely(b == 0)) {
        return a;
    }
    return (a + b - 1) / b;
}

// 二分累加到2 * offset以内，再累加成offset大小
template <typename OriT, typename IdxT, typename ProbT>
__aicore__ inline void MoeTokenUnpermuteGradBase<OriT, IdxT, ProbT>::BinaryAddFunc(LocalTensor<float> tmpBuffer,
                                                                                   int32_t hiddensizeLen,
                                                                                   int32_t threshold, int32_t offset)
{
    int32_t totalLen = hiddensizeLen;
    int32_t halfLen = this->AlignUp(this->CeilDiv(totalLen, 2), BUFFER_32B_CALNUM);
    while (totalLen > threshold) {
        Add(tmpBuffer, tmpBuffer, tmpBuffer[halfLen], totalLen - halfLen);
        totalLen = halfLen;
        halfLen = this->AlignUp(this->CeilDiv(totalLen, 2), BUFFER_32B_CALNUM);
    }
    Add(tmpBuffer, tmpBuffer, tmpBuffer[offset], totalLen - offset);
}

template <typename OriT, typename IdxT, typename ProbT>
__aicore__ inline void MoeTokenUnpermuteGradBase<OriT, IdxT, ProbT>::ReduceSumFunc(LocalTensor<float> dstBuffer,
                                                                                   LocalTensor<float> tmpBuffer,
                                                                                   int32_t hiddensizeLen)
{
    if (hiddensizeLen >= 4096) { // 二分累加到8192以内，再累加成4096大小，用blockreducesum
        this->BinaryAddFunc(tmpBuffer, hiddensizeLen, 8192, 4096); // 累加成4096大小
        // 用BlockReduceSum+WholeReduceSum 把4096 reduce成1个
        BlockReduceSum(tmpBuffer, tmpBuffer, 64, 64, 1, 1, 8); // 输出512
        BlockReduceSum(tmpBuffer, tmpBuffer, 8, 64, 1, 1, 8);  // 输出64
        BlockReduceSum(tmpBuffer, tmpBuffer, 1, 64, 1, 1, 8);  // 输出8
        WholeReduceSum(dstBuffer, tmpBuffer, 8, 1, 1, 1, 8);   // 输出1
    } else if (hiddensizeLen >= 512) { // 二分累加到1024以内，再累加成512大小，用blockreducesum
        this->BinaryAddFunc(tmpBuffer, hiddensizeLen, 1024, 512); // 累加成512大小
        // 用BlockReduceSum+WholeReduceSum 把512 reduce成1个
        BlockReduceSum(tmpBuffer, tmpBuffer, 8, 64, 1, 1, 8); // 输出64
        BlockReduceSum(tmpBuffer, tmpBuffer, 1, 64, 1, 1, 8); // 输出8
        WholeReduceSum(dstBuffer, tmpBuffer, 8, 1, 1, 1, 8);  // 输出1
    } else if (hiddensizeLen >= 64) { // 二分累加到128以内，再累加成64大小，用blockreducesum
        this->BinaryAddFunc(tmpBuffer, hiddensizeLen, 128, 64); // 累加成64大小
        // 用BlockReduceSum+WholeReduceSum 把64 reduce成1个
        BlockReduceSum(tmpBuffer, tmpBuffer, 1, 64, 1, 1, 8); // 输出8
        WholeReduceSum(dstBuffer, tmpBuffer, 8, 1, 1, 1, 8);  // 输出1
    } else {                                                  // 64个以内的元素直接WholeReduceSum成1个
        WholeReduceSum(dstBuffer, tmpBuffer, hiddensizeLen, 1, 1, 1, 8);
    }
}

template <typename OriT, typename IdxT, typename ProbT>
__aicore__ inline void MoeTokenUnpermuteGradBase<OriT, IdxT, ProbT>::Init(
    GM_ADDR permuted_tokens, GM_ADDR unpermuted_tokens_grad, GM_ADDR sorted_indices, GM_ADDR probs,
    GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, const MoeTokenUnpermuteGradTilingData &tilingData)
{
    tokensNum = tilingData.tokensNum;
    topK = tilingData.topK;
    hiddenSize = tilingData.hiddenSize;
    formerCoreNum = tilingData.formerCoreNum;
    tailCoreNum = tilingData.tailCoreNum;
    tokenNumEachCore = tilingData.tokenNumEachCore;
    tokenNumTailCore = tilingData.tokenNumTailCore;
    rowIdMapEachCore = tilingData.rowIdMapEachCore;
    rowIdMapTailCore = tilingData.rowIdMapTailCore;
    hiddenSizeAlign = tilingData.hiddenSizeAlign;
    hiddenSizeLoopTimes = tilingData.hiddenSizeLoopTimes;
    hiddenSizeTail = tilingData.hiddenSizeTail;
    inputReserveNum = tilingData.inputReserveNum;               // permutedTokenNumPerLoop
    indicesReserveNum = tilingData.indicesReserveNum;           // indicesNumPerLoop
    indicesReserveNumAlign = tilingData.indicesReserveNumAlign; // indicesNumPerLoopAlign
    numOutTokens = tilingData.numOutTokens;

    coreIndex = GetBlockIdx();
    inputTypeSize = sizeof(OriT);
    rowIdMapTypeSize = sizeof(IdxT);
    probTypeSize = sizeof(ProbT);
    indicesReserveNumAlignFp32RepeatTimes = this->indicesReserveNumAlign / FP32_ONE_REPEAT;
    indicesReserveNumAlignFp32TailMask = this->indicesReserveNumAlign % FP32_ONE_REPEAT;
    indicesReserveNumAlignFp32TailOffset = indicesReserveNumAlignFp32RepeatTimes * FP32_ONE_REPEAT;
    hiddensizeAlignFp32RepeatTimes = this->hiddenSizeAlign / FP32_ONE_REPEAT;
    hiddensizeAlignFp32TailMask = this->hiddenSizeAlign % FP32_ONE_REPEAT;
    hiddensizeAlignFp32TailOffset = hiddensizeAlignFp32RepeatTimes * FP32_ONE_REPEAT;

    permutedTokensGm.SetGlobalBuffer((__gm__ OriT *)permuted_tokens);
    unpermutedOutputDGm.SetGlobalBuffer((__gm__ OriT *)unpermuted_tokens_grad);
    sortedIndicesGm.SetGlobalBuffer((__gm__ IdxT *)sorted_indices);
    permutedTokensGradGm.SetGlobalBuffer((__gm__ OriT *)permuted_tokens_grad);
    probGm.SetGlobalBuffer((__gm__ ProbT *)probs);
    probGradGm.SetGlobalBuffer((__gm__ ProbT *)probs_grad);

    if (coreIndex < formerCoreNum) {
        rowIdMapStartOffset = coreIndex * rowIdMapEachCore;
        unpermutedOutputDStartOffset = coreIndex * tokenNumEachCore;
    } else {
        rowIdMapStartOffset = formerCoreNum * rowIdMapEachCore + (coreIndex - formerCoreNum) * rowIdMapTailCore;
        unpermutedOutputDStartOffset =
            formerCoreNum * tokenNumEachCore + (coreIndex - formerCoreNum) * tokenNumTailCore;
    }
}
} // namespace MoeTokenUnpermuteGrad
#endif // MOE_TOKEN_UNPERMUTE_GRAD_BASE_H
