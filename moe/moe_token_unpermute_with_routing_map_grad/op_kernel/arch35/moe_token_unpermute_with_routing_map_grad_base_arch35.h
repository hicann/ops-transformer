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
 * \file moe_token_unpermute_with_routing_map_grad_base_arch35.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_BASE_ARCH35_H
#define MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_BASE_ARCH35_H
#include "kernel_operator.h"
namespace MoeTokenUnpermuteWithRoutingMapGrad {
using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int64_t BUFFER_32B_CALNUM = 32 / sizeof(float);
constexpr int64_t SIZE_FLOAT = 4;
constexpr int64_t BLOCK_SIZE_512 = 512;
constexpr int64_t BLOCK_SIZE_256 = 256;
constexpr int64_t BLOCK_SIZE_32 = 32;
constexpr int64_t FP32_ONE_REPEAT = 64;
constexpr int64_t INDICES_PROBS_MAX_RESERVE_NUM = 512;

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
class MoeTokenUnpermuteWithRoutingMapGradBase {
public:
    __aicore__ inline MoeTokenUnpermuteWithRoutingMapGradBase(){};
    __aicore__ inline void Init(GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId,
                                GM_ADDR routing_map, GM_ADDR permuted_tokens, GM_ADDR probs,
                                GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
                                const MoeTokenUnpermuteWithRoutingMapGradTilingData &tiling_data);
    __aicore__ inline int32_t AlignUp(int32_t a, int32_t b);
    __aicore__ inline int32_t CeilDiv(int32_t a, int32_t b);
    __aicore__ inline int32_t SafeDiv(int32_t a, int32_t b);
    __aicore__ inline void BinaryAddFunc(LocalTensor<float> tmpBuffer, int32_t hiddensizeLen, int32_t threshold,
                                         int32_t offset);
    __aicore__ inline void ReduceSumFunc(LocalTensor<float> dstBuffer, LocalTensor<float> tmpBuffer,
                                         int32_t hiddensizeLen);

protected:
    TPipe pipe;
    GlobalTensor<PermutedTokenT> unpermutedTokensGradGm;
    GlobalTensor<IdxT> sortedTwiceIndexGm;
    GlobalTensor<IdxT> sortedTwiceIndicesGm;
    GlobalTensor<PermutedTokenT> permutedTokensGm;
    GlobalTensor<ProbsT> probGm;
    GlobalTensor<int8_t> routingMapGm;
    GlobalTensor<PermutedTokenT> permutedTokensGradGm;
    GlobalTensor<ProbsT> probGradGm;

    int64_t tokensNum;
    int64_t topK;
    int64_t capacity;
    int64_t numExpert;
    int64_t hiddenSize;
    int64_t formerCoreNum;
    int64_t tailCoreNum;
    int64_t tokenNumEachCore;
    int64_t tokenNumTailCore;
    int64_t inputReserveNum;
    int64_t indicesReserveNum;
    int64_t indicesReserveNumAlign;
    int64_t rowIdMapEachCore;
    int64_t rowIdMapTailCore;
    int64_t coreIndex;
    int64_t rowIdMapStartOffset;
    int64_t hiddenSizeAlign;
    int64_t hiddenSizeLoopTimes;
    int64_t hiddenSizeLoopTimesAlign;
    int64_t hiddenSizeTail;
    int64_t numOutTokens;
    int64_t unpermutedOutputDStartOffset;
    uint32_t inputTypeSize;
    uint32_t rowIdMapTypeSize;
    uint32_t probTypeSize;
    int64_t hiddensizeAlignFp32RepeatTimes;
    int64_t hiddensizeAlignFp32TailMask;
    int64_t hiddensizeAlignFp32TailOffset;
    int64_t indicesReserveNumAlignFp32RepeatTimes;
    int64_t indicesReserveNumAlignFp32TailMask;
    int64_t indicesReserveNumAlignFp32TailOffset;
    int64_t numExpertAlign;

    DataCopyPadExtParams<IdxT> rowIdMapPadParams{false, 0, 0, 0};
    DataCopyPadExtParams<PermutedTokenT> inputPadParams{false, 0, 0, 0};
    DataCopyPadExtParams<ProbsT> probPadParams{false, 0, 0, 0};
};

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline int32_t MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT>::AlignUp(int32_t a,
                                                                                                         int32_t b)
{
    if (unlikely(b == 0)) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline int32_t MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT>::CeilDiv(int32_t a,
                                                                                                         int32_t b)
{
    if (unlikely(b == 0)) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline int32_t MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT>::SafeDiv(int32_t a,
                                                                                                         int32_t b)
{
    if (unlikely(b == 0)) {
        return 0;
    }
    return a / b;
}

__aicore__ inline void SToMTE2Sync()
{
    event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
}

__aicore__ inline void VToMTE3Sync()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

__aicore__ inline void SToMTE3Sync()
{
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}

__aicore__ inline void MTE3ToSSync()
{
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

__aicore__ inline void MTE3ToVSync()
{
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
}

__aicore__ inline void VToSSync()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

__aicore__ inline void MTE2ToVSync()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

// 二分累加到2 * offset以内，再累加成offset大小
template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT>::BinaryAddFunc(
    LocalTensor<float> tmpBuffer, int32_t hiddensizeLen, int32_t threshold, int32_t offset)
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

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT>::ReduceSumFunc(
    LocalTensor<float> dstBuffer, LocalTensor<float> tmpBuffer, int32_t hiddensizeLen)
{
    if (hiddensizeLen > 4096) { // 二分累加到8192以内，再累加成4096大小，用blockreducesum
        this->BinaryAddFunc(tmpBuffer, hiddensizeLen, 8192, 4096); // 累加成4096大小
        // 用BlockReduceSum+WholeReduceSum 把4096 reduce成1个
        BlockReduceSum(tmpBuffer, tmpBuffer, 64, 64, 1, 1, 8); // 输出512
        BlockReduceSum(tmpBuffer, tmpBuffer, 8, 64, 1, 1, 8);  // 输出64
        BlockReduceSum(tmpBuffer, tmpBuffer, 1, 64, 1, 1, 8);  // 输出8
        WholeReduceSum(dstBuffer, tmpBuffer, 8, 1, 1, 1, 8);   // 输出1
    } else if (hiddensizeLen > 512) { // 二分累加到1024以内，再累加成512大小，用blockreducesum
        this->BinaryAddFunc(tmpBuffer, hiddensizeLen, 1024, 512); // 累加成512大小
        // 用BlockReduceSum+WholeReduceSum 把512 reduce成1个
        BlockReduceSum(tmpBuffer, tmpBuffer, 8, 64, 1, 1, 8); // 输出64
        BlockReduceSum(tmpBuffer, tmpBuffer, 1, 64, 1, 1, 8); // 输出8
        WholeReduceSum(dstBuffer, tmpBuffer, 8, 1, 1, 1, 8);  // 输出1
    } else if (hiddensizeLen > 64) { // 二分累加到128以内，再累加成64大小，用blockreducesum
        this->BinaryAddFunc(tmpBuffer, hiddensizeLen, 128, 64); // 累加成64大小
        // 用BlockReduceSum+WholeReduceSum 把64 reduce成1个
        BlockReduceSum(tmpBuffer, tmpBuffer, 1, 64, 1, 1, 8); // 输出8
        WholeReduceSum(dstBuffer, tmpBuffer, 8, 1, 1, 1, 8);  // 输出1
    } else {                                                  // 64个以内的元素直接WholeReduceSum成1个
        WholeReduceSum(dstBuffer, tmpBuffer, hiddensizeLen, 1, 1, 1, 8);
    }
}

template <typename PermutedTokenT, typename IdxT, typename ProbsT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradBase<PermutedTokenT, IdxT, ProbsT>::Init(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId, GM_ADDR routing_map,
    GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
    const MoeTokenUnpermuteWithRoutingMapGradTilingData &tiling_data)
{
    tokensNum = tiling_data.tokensNum;
    topK = tiling_data.topK;
    capacity = tiling_data.capacity;
    numExpert = tiling_data.numExpert;
    hiddenSize = tiling_data.hiddenSize;
    formerCoreNum = tiling_data.formerCoreNum;
    tailCoreNum = tiling_data.tailCoreNum;
    tokenNumEachCore = tiling_data.tokenNumEachCore;
    tokenNumTailCore = tiling_data.tokenNumTailCore;
    rowIdMapEachCore = tiling_data.rowIdMapEachCore;
    rowIdMapTailCore = tiling_data.rowIdMapTailCore;
    hiddenSizeAlign = tiling_data.hiddenSizeAlign;
    hiddenSizeLoopTimes = tiling_data.hiddenSizeLoopTimes;
    hiddenSizeLoopTimesAlign = tiling_data.hiddenSizeLoopTimesAlign;
    hiddenSizeTail = tiling_data.hiddenSizeTail;
    inputReserveNum = tiling_data.inputReserveNum;               // permutedTokenNumPerLoop
    indicesReserveNum = tiling_data.indicesReserveNum;           // indicesNumPerLoop
    indicesReserveNumAlign = tiling_data.indicesReserveNumAlign; // indicesNumPerLoopAlign
    numOutTokens = tiling_data.numOutTokens;
    numExpertAlign = tiling_data.numExpertAlign;
    coreIndex = GetBlockIdx();
    inputTypeSize = sizeof(PermutedTokenT);
    rowIdMapTypeSize = sizeof(IdxT);
    probTypeSize = sizeof(ProbsT);
    indicesReserveNumAlignFp32RepeatTimes = this->indicesReserveNumAlign / FP32_ONE_REPEAT;
    indicesReserveNumAlignFp32TailMask = this->indicesReserveNumAlign % FP32_ONE_REPEAT;
    indicesReserveNumAlignFp32TailOffset = indicesReserveNumAlignFp32RepeatTimes * FP32_ONE_REPEAT;
    hiddensizeAlignFp32RepeatTimes = this->hiddenSizeAlign / FP32_ONE_REPEAT;
    hiddensizeAlignFp32TailMask = this->hiddenSizeAlign % FP32_ONE_REPEAT;
    hiddensizeAlignFp32TailOffset = hiddensizeAlignFp32RepeatTimes * FP32_ONE_REPEAT;

    unpermutedTokensGradGm.SetGlobalBuffer((__gm__ PermutedTokenT *)unpermuted_tokens_grad);
    sortedTwiceIndexGm.SetGlobalBuffer((__gm__ IdxT *)outIndex);
    sortedTwiceIndicesGm.SetGlobalBuffer((__gm__ IdxT *)permuteTokenId);
    permutedTokensGm.SetGlobalBuffer((__gm__ PermutedTokenT *)permuted_tokens);
    probGm.SetGlobalBuffer((__gm__ ProbsT *)probs);
    routingMapGm.SetGlobalBuffer((__gm__ int8_t *)routing_map);

    permutedTokensGradGm.SetGlobalBuffer((__gm__ PermutedTokenT *)permuted_tokens_grad);
    probGradGm.SetGlobalBuffer((__gm__ ProbsT *)probs_grad);

    if (coreIndex < formerCoreNum) {
        rowIdMapStartOffset = coreIndex * rowIdMapEachCore;
        unpermutedOutputDStartOffset = coreIndex * tokenNumEachCore;
    } else {
        rowIdMapStartOffset = formerCoreNum * rowIdMapEachCore + (coreIndex - formerCoreNum) * rowIdMapTailCore;
        unpermutedOutputDStartOffset =
            formerCoreNum * tokenNumEachCore + (coreIndex - formerCoreNum) * tokenNumTailCore;
    }
} // namespace MoeTokenUnpermuteWithRoutingMapGrad
} // namespace MoeTokenUnpermuteWithRoutingMapGrad
#endif // MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_BASE_ARCH35_H
