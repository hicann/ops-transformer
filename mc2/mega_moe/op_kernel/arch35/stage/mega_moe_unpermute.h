/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_UNPERMUTE_H
#define MEGA_MOE_UNPERMUTE_H

#include "../common/mega_moe_mxfp8_utils.h"
#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

struct TokenUnpermuteConfig {
    MoeStageCommonConfig common;
    AivJobContext job;
    uint32_t quantTokenSizeBytes;
    uint32_t fullTokenChunkJobCount;
    MegaMoeUnpermuteBufferConfig fullTokenChunkConfig;
    MegaMoeUnpermuteBufferConfig tailTokenChunkConfig;
};

struct TokenUnpermuteScratch {
    LocalTensor<bfloat16_t> dataResTensor;
    LocalTensor<float> dataResFp32Tensor;
    LocalTensor<float> topKWeightsTensor;
    LocalTensor<float> fp32ScaleTensor;
    LocalTensor<bfloat16_t> bf16ScaleTensor;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor;
};

// 加载并转换一批 top-k 权重。
template <typename TopkWeightsType>
__aicore__ inline void LoadTokenUnpermuteWeights(const TokenUnpermuteConfig &context,
                                                 const Params &params,
                                                 TokenUnpermuteScratch &scratch, int32_t jobOffset,
                                                 int32_t batchTokenOffset, int32_t batchTokenCount)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.probsGmAddr));
        DataCopyPad(scratch.topKWeightsTensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * context.common.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * context.common.topK * sizeof(float)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topKWeightsGm;
        topKWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.probsGmAddr));
        DataCopyPad(scratch.topKWeightsBf16Tensor,
                    topKWeightsGm[static_cast<uint64_t>(jobOffset + batchTokenOffset) * context.common.topK],
                    {1U, static_cast<uint32_t>(batchTokenCount * context.common.topK * sizeof(bfloat16_t)), 0U, 0U, 0U},
                    {false, 0U, 0U, 0U});
        SyncFuncStatic<HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(scratch.topKWeightsTensor, scratch.topKWeightsBf16Tensor, RoundMode::CAST_NONE,
             batchTokenCount * context.common.topK);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
}

// 将一个 MoE 专家的输入搬入 UB，并按 Combine 模式完成 Cast 或 MXFP8 反量化。
template <uint8_t CombineMode>
__aicore__ inline void LoadMoeExpertInput(
    const TokenUnpermuteConfig &context, TokenUnpermuteScratch &scratch,
    const GlobalTensor<bfloat16_t> &expandedX, uint64_t expertInputIndex, TEventID event,
    LocalTensor<bfloat16_t> &dataInBf16, LocalTensor<float> &dataInFp32)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        DataCopy(dataInBf16, expandedX[expertInputIndex * context.common.tokenHiddenDim],
                 context.common.tokenHiddenDim);
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, context.common.tokenHiddenDim);
    } else {
        uint32_t quantTokenElementCount = context.quantTokenSizeBytes / sizeof(bfloat16_t);
        DataCopy(dataInBf16, expandedX[expertInputIndex * quantTokenElementCount], quantTokenElementCount);
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
        uint32_t nScale = Ops::Base::CeilDiv(context.common.tokenHiddenDim,
                                             static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT,
                                                  fp8_e4m3fn_t, fp8_e5m2_t>::type;
        Mxfp8::DeQuantMxFp8<Fp8Type, bfloat16_t>(
            dataInBf16, dataInFp32, scratch.bf16ScaleTensor, scratch.fp32ScaleTensor, nScale,
            context.common.tokenHiddenDim);
    }
}

// 累加一个 token 对应的全部 MoE 专家输入。
template <uint8_t CombineMode, bool TopkWeightsPrefetch>
__aicore__ inline void AccumulateMoeExpertsForToken(
    const TokenUnpermuteConfig &context, TokenUnpermuteScratch &scratch, int32_t tokenIdx,
    int32_t localIdx, const GlobalTensor<bfloat16_t> &expandedX,
    const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    for (int32_t expertIdx = 0; expertIdx < static_cast<int32_t>(context.common.topK); ++expertIdx) {
        int32_t accumulationItemIdx =
            localIdx * static_cast<int32_t>(context.common.topK + context.common.sharedExpertNum) + expertIdx;
        int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 =
            scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 =
            scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        uint64_t expertInputIndex = static_cast<uint64_t>(tokenIdx) * context.common.topK + expertIdx;
        LoadMoeExpertInput<CombineMode>(context, scratch, expandedX, expertInputIndex, event,
                                        dataInBf16, dataInFp32);
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
        PipeBarrier<PIPE_V>();
        if (expertIdx == 0) {
            if constexpr (TopkWeightsPrefetch) {
                DataCopy(scratch.dataResFp32Tensor, dataInFp32, context.common.tokenHiddenDim);
            } else {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * context.common.topK + expertIdx);
                Muls(scratch.dataResFp32Tensor, dataInFp32, expertScale, context.common.tokenHiddenDim);
            }
        } else {
            if constexpr (!TopkWeightsPrefetch) {
                float expertScale = scratch.topKWeightsTensor.GetValue(localIdx * context.common.topK + expertIdx);
                Muls(dataInFp32, dataInFp32, expertScale, context.common.tokenHiddenDim);
                PipeBarrier<PIPE_V>();
            }
            Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32,
                context.common.tokenHiddenDim);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE2>(event);
    }
}

// 累加一个已经就绪的共享专家输入。
__aicore__ inline void AccumulateSharedExpertForToken(
    const TokenUnpermuteConfig &context, const Params &params, TokenUnpermuteScratch &scratch,
    int32_t tokenIdx, int32_t localIdx, uint32_t sharedExpertIdx,
    const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(params.workspaceInfo.sharedExpertResultPtr));
    int32_t accumulationItemIdx =
        localIdx * static_cast<int32_t>(context.common.topK + context.common.sharedExpertNum) +
        static_cast<int32_t>(context.common.topK + sharedExpertIdx);
    int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
    TEventID event = static_cast<TEventID>(inputBufferIdx);
    LocalTensor<bfloat16_t> dataInBf16 =
        scratch.dataResTensor[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
    LocalTensor<float> dataInFp32 =
        scratch.dataResFp32Tensor[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopy(dataInBf16,
             sharedResult[(static_cast<uint64_t>(sharedExpertIdx) * context.common.tokenNum + tokenIdx) *
                          context.common.tokenHiddenDim],
             context.common.tokenHiddenDim);
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
    Cast(dataInFp32, dataInBf16, RoundMode::CAST_NONE, context.common.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    Add(scratch.dataResFp32Tensor, scratch.dataResFp32Tensor, dataInFp32, context.common.tokenHiddenDim);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event);
}

// 等待一个 Unpermute token 所依赖的共享专家 GMM2 tile group。
template <uint32_t Gmm1TileM>
__aicore__ inline void WaitSharedExpertGmm2Ready(const TokenUnpermuteConfig &context,
                                                 const Params &params, int32_t tokenIdx,
                                                 uint32_t sharedExpertIdx)
{
    uint32_t tokenGroupIndex = static_cast<uint32_t>(tokenIdx) / Gmm1TileM;
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(context.common.tokenNum, Gmm1TileM);
    uint64_t sharedExpertStride = static_cast<uint64_t>(tokenGroupCount) * INT_CACHELINE;
    __gm__ int32_t *counterAddr = GetCombineSyncCounterAddress(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.sharedExpertGmm2TileCounterPtr) +
            static_cast<uint64_t>(sharedExpertIdx) * sharedExpertStride,
        tokenGroupIndex);
    uint32_t gmm2NTilesPerGroup = Ops::Base::CeilDiv(context.common.tokenHiddenDim, L1_TILE_N);
    while (AscendC::ReadGmByPassDCache(counterAddr) !=
           static_cast<int32_t>(gmm2NTilesPerGroup)) {
        int64_t waitStartCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - waitStartCycle < 100) {
        }
    }
}

template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch,
          uint32_t Gmm1TileM>
__aicore__ inline void ProcessTokenUnpermuteBatch(
    const TokenUnpermuteConfig &context, const Params &params, TokenUnpermuteScratch &scratch,
    const MegaMoeUnpermuteBufferConfig &bufferConfig, const GlobalTensor<bfloat16_t> &expandedX,
    GlobalTensor<bfloat16_t> &output, int32_t jobOffset, int32_t batchTokenOffset,
    int32_t batchTokenCount, TEventID outputBufferEvent)
{
    if constexpr (!TopkWeightsPrefetch) {
        LoadTokenUnpermuteWeights<TopkWeightsType>(
            context, params, scratch, jobOffset, batchTokenOffset, batchTokenCount);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
    }
    for (int32_t localIdx = 0; localIdx < batchTokenCount; ++localIdx) {
        int32_t tokenIdx = jobOffset + batchTokenOffset + localIdx;
        AccumulateMoeExpertsForToken<CombineMode, TopkWeightsPrefetch>(
            context, scratch, tokenIdx, localIdx, expandedX, bufferConfig);
        for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < context.common.sharedExpertNum;
             ++sharedExpertIdx) {
            WaitSharedExpertGmm2Ready<Gmm1TileM>(context, params, tokenIdx, sharedExpertIdx);
            AccumulateSharedExpertForToken(
                context, params, scratch, tokenIdx, localIdx, sharedExpertIdx, bufferConfig);
        }
        WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
        Cast(scratch.dataResTensor, scratch.dataResFp32Tensor, RoundMode::CAST_RINT,
             context.common.tokenHiddenDim);
        SyncFuncStatic<HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[static_cast<uint64_t>(tokenIdx) * context.common.tokenHiddenDim],
                 scratch.dataResTensor, context.common.tokenHiddenDim);
        SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
    }
}

// 原型：MegaMoe::Unpermute。执行当前 token 分区的完整 Unpermute 流水。
template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch,
          uint32_t Gmm1TileM>
__aicore__ inline void UnpermuteTokens(const TokenUnpermuteConfig &context,
                                       const Params &params,
                                       TokenUnpermuteScratch &scratch,
                                       const MegaMoeUnpermuteBufferConfig &bufferConfig)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(context.common.tokenNum, context.job.jobIndex,
                                              context.job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }
    int32_t jobLen = static_cast<int32_t>(tokenRange.count);
    int32_t jobOffset = static_cast<int32_t>(tokenRange.start);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.peermemInfo.combineSendPtr));
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.y2GmAddr));

    constexpr TEventID outputBufferEvent = EVENT_ID0;
    SetFlag<HardEvent::MTE3_V>(outputBufferEvent);
    for (int32_t batchTokenOffset = 0; batchTokenOffset < jobLen;
         batchTokenOffset += bufferConfig.tokensPerBatch) {
        int32_t batchTokenCount =
            batchTokenOffset + bufferConfig.tokensPerBatch > jobLen ?
                jobLen - batchTokenOffset :
                bufferConfig.tokensPerBatch;
        ProcessTokenUnpermuteBatch<CombineMode, TopkWeightsType, TopkWeightsPrefetch, Gmm1TileM>(
            context, params, scratch, bufferConfig, expandedX, output, jobOffset,
            batchTokenOffset, batchTokenCount, outputBufferEvent);
    }
    WaitFlag<HardEvent::MTE3_V>(outputBufferEvent);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_UNPERMUTE_H
