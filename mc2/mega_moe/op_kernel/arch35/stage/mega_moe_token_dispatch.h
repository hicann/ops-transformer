/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TOKEN_DISPATCH_H
#define MEGA_MOE_TOKEN_DISPATCH_H

#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

// Token Dispatch uses a logical block job for work partitioning and a physical
// workspace slot for count/flag publication. Neither index is derived here.
struct TokenDispatchConfig {
    MoeStageCommonConfig common;
    BlockJobContext blockJob;
    BlockWorkspaceContext countWorkspace;
    uint64_t maxOutputSize;
    MegaMoeDispatchBufferConfig bufferConfig;
    uint32_t maskAlignSize;
    uint64_t quantWinOffset;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    int32_t dispatchFlagSlotsPerExpert;
};

template <typename ActivationType>
struct TokenDispatchScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
    GlobalTensor<int32_t> cumsumInfoGlobalTensor;
    LocalTensor<int32_t> topkIndexTensor;
    LocalTensor<int32_t> sendCntTensor;
    LocalTensor<uint8_t> maskBatchTensor;
    LocalTensor<uint32_t> maskBatchU32Tensor;
    LocalTensor<int32_t> expertTokenCntTensor;
    LocalTensor<int32_t> validTopkIndexTensor;
    LocalTensor<int32_t> cumsumInfoTensor;
    // Contiguous runtime-sized ring. A slot view is built from this UB base on demand.
    uint32_t copyTmpBaseAddr;
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<int32_t> expertTokenNumsOutTensor;
    int64_t revTokenElemCnt;
    int64_t revScaleElemCnt;
    uint64_t cumsumRevCntInRank;
};

template <typename ActivationType>
__aicore__ inline LocalTensor<ActivationType> GetDispatchCopyBuffer(
    const TokenDispatchConfig &context, const TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx)
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC,
        scratch.copyTmpBaseAddr + static_cast<uint32_t>(bufferIdx) * context.quantTokenScaleAlignBytes,
        context.quantTokenScaleAlignBytes / sizeof(ActivationType));
}

template <bool IsBufferReuse, bool TopkWeightsPrefetch, typename ActivationType>
__aicore__ inline void FetchDispatchTokenAndMetaInfo(const TokenDispatchConfig &context,
                                                     TokenDispatchScratch<ActivationType> &scratch,
                                                     int32_t bufferIdx, int32_t topkIndex, int32_t remoteRankIdx,
                                                     GlobalTensor<ActivationType> &remoteRankGlobalTensor)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    LocalTensor<ActivationType> copyTmpTensor = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    int32_t tokenIndex = topkIndex / static_cast<int32_t>(context.common.topK);
    uint64_t remoteCopyOffset =
        static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(context.quantTokenScaleAlignBytes);
    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
    DataCopy(copyTmpTensor, remoteRankGlobalTensor[remoteCopyOffset], context.quantTokenScaleAlignBytes);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOPK_INDEX,
                                                                topkIndex % static_cast<int32_t>(context.common.topK));
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDispatchTokenAndMetaInfo(
    const TokenDispatchConfig &context, TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx,
    int32_t dstIdx, GlobalTensor<ActivationType> &tokenRevGlobalTensor,
    GlobalTensor<QuantScaleType> &scaleRevGlobalTensor, GlobalTensor<int32_t> &metaInfoGlobalTensor,
    int32_t copyStartIdx, int32_t copyIdx)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScaleBuffer = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    LocalTensor<QuantScaleType> scaleBuffer =
        tokenScaleBuffer[context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
    if constexpr (TopkWeightsPrefetch) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = context.quantTokenAlignBytes + context.quantScaleAlignBytes;
        LocalTensor<int32_t> weightBitsTensor = tokenScaleBuffer[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + copyIdx);
        int32_t weightBits =
            weightBitsTensor.GetValue(static_cast<uint32_t>(topkIndex % static_cast<int32_t>(context.common.topK)));
        scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
    DataCopyPad(tokenRevGlobalTensor[dstIdx * scratch.revTokenElemCnt], tokenScaleBuffer,
                {1, static_cast<uint16_t>(scratch.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleRevGlobalTensor[dstIdx * scratch.revScaleElemCnt], scaleBuffer,
                {1, static_cast<uint16_t>(scratch.revScaleElemCnt * sizeof(QuantScaleType)), 0U, 0U, 0U});
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    DataCopy(metaInfoGlobalTensor[dstIdx * INT32_PER_256B], scratch.metaInfoTensor[bufferIdx * INT32_PER_256B],
             INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

// Prototype: MegaMoe::CopyGMToGMPerToken. Copies one dispatch shard and emits its metadata.
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyTokensAndMetaForDispatch(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, int32_t rowDstOffset, int32_t remoteRankIdx,
    int32_t copyStartIdx, int32_t copyNum)
{
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    int32_t bufferCount = bufferConfig.bufferCount;

    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params.workspaceInfo.dispatchRevDataPtr +
                                                  rowDstOffset * scratch.revTokenElemCnt));
    scaleRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType *>(params.workspaceInfo.dispatchRevScalePtr +
                                                  rowDstOffset * scratch.revScaleElemCnt));
    remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(winRankAddr[remoteRankIdx] + context.quantWinOffset));
    metaInfoGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr +
                                           rowDstOffset * INT32_PER_256B * sizeof(int32_t)));

    // The only caller enters this function for a non-empty match interval, so copyNum is at least one.
    // GatherMask's V-to-S synchronization covers the first scalar read, and the drain below covers
    // cross-call reuse of the copy and metadata rings.
    int32_t firstTopkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx);
    FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, scratch, 0, firstTopkIndex, remoteRankIdx,
                                                              remoteRankGlobalTensor);

    int32_t firstUseEnd = copyNum < bufferCount ? copyNum : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, scratch, issueIdx, topkIndex, remoteRankIdx,
                                                                  remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            copyStartIdx, copyIdx);
    }

    for (int32_t issueIdx = bufferCount; issueIdx < copyNum; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(context, scratch, issueBufferIdx, topkIndex,
                                                                 remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyBufferIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            copyStartIdx, copyIdx);
    }

    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, scratch, (copyNum - 1) % bufferCount, copyNum - 1, tokenRevGlobalTensor, scaleRevGlobalTensor,
        metaInfoGlobalTensor, copyStartIdx, copyNum - 1);

    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

struct DispatchShardRange {
    uint32_t remoteRankIdx;
    uint32_t rankSegmentDstRowBegin;
    int32_t matchOrdinalBegin;
    int32_t matchOrdinalEnd;
    int32_t dstRowBegin;
};

template <typename ActivationType>
__aicore__ inline DispatchShardRange CalculateDispatchShardRange(
    const TokenDispatchConfig &context, const TokenDispatchScratch<ActivationType> &scratch,
    uint32_t localExpertId, uint32_t dispatchShardIdx, uint32_t rankShardCount)
{
    uint32_t remoteRankIdx = dispatchShardIdx / rankShardCount;
    uint32_t rankShardIdx = dispatchShardIdx % rankShardCount;
    uint32_t rankSegmentDstRowBegin =
        (remoteRankIdx == 0U && localExpertId == 0U) ?
            0U :
            static_cast<uint32_t>(
                scratch.cumsumInfoTensor.GetValue(localExpertId * context.common.worldSize + remoteRankIdx - 1U));
    DispatchShardRange range{remoteRankIdx, rankSegmentDstRowBegin, 0, 0, 0};
    if (rankSegmentDstRowBegin >= context.maxOutputSize) {
        return range;
    }

    int32_t rankTokenCount =
        scratch.cumsumInfoTensor.GetValue(localExpertId * context.common.worldSize + remoteRankIdx) -
        static_cast<int32_t>(rankSegmentDstRowBegin);
    int32_t rankDispatchRowCount =
        rankSegmentDstRowBegin + rankTokenCount > context.maxOutputSize ?
            static_cast<int32_t>(context.maxOutputSize - rankSegmentDstRowBegin) :
            rankTokenCount;
    int32_t rowsPerRankShard =
        Ops::Base::CeilDiv(rankDispatchRowCount, static_cast<int32_t>(rankShardCount));
    int32_t rankShardRowBegin = static_cast<int32_t>(rankShardIdx) * rowsPerRankShard;
    range.dstRowBegin = static_cast<int32_t>(rankSegmentDstRowBegin) + rankShardRowBegin;
    int32_t dispatchRowCount =
        range.dstRowBegin + rowsPerRankShard > rankSegmentDstRowBegin + rankDispatchRowCount ?
            static_cast<int32_t>(rankSegmentDstRowBegin + rankDispatchRowCount - range.dstRowBegin) :
            rowsPerRankShard;
    if (dispatchRowCount > 0) {
        range.matchOrdinalBegin = rankShardRowBegin;
        range.matchOrdinalEnd = rankShardRowBegin + dispatchRowCount;
    }
    return range;
}

// Prototype: MegaMoe::MetaInfoCalAndDispatch. Scans route batches and dispatches one block's token shards.
template <typename ActivationType, typename QuantScaleType, bool EnableA8W4, uint32_t PipelineTileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokens(const TokenDispatchConfig &context, const Params &params,
                                            GM_ADDR *winRankAddr, TokenDispatchScratch<ActivationType> &scratch,
                                            uint32_t localExpertId)
{
    constexpr int32_t dispatchWaveRowCount = static_cast<int32_t>(PipelineTileM);
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    uint32_t maskSlotSize = context.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    int32_t expertGlobalRowBegin =
        localExpertId == 0 ? 0 : scratch.cumsumInfoTensor.GetValue(localExpertId * context.common.worldSize - 1);
    __gm__ int32_t *dispatchWaveReadyCount =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * context.dispatchFlagSlotsPerExpert;
    uint32_t rankShardCount = context.countWorkspace.blockNum / context.common.worldSize;
    rankShardCount = rankShardCount == 0U ? 1U : rankShardCount;

    for (uint32_t dispatchShardIdx = context.blockJob.jobIndex;
         dispatchShardIdx < context.common.worldSize * rankShardCount;
         dispatchShardIdx += context.blockJob.totalJobs) {
        DispatchShardRange shardRange =
            CalculateDispatchShardRange(context, scratch, localExpertId, dispatchShardIdx, rankShardCount);

        GlobalTensor<uint8_t> remoteRankMaskGlobal;
        int32_t matchedRouteCount = 0;
        int32_t dispatchedRowCount = 0;
        for (int32_t batchIdx = 0;
             batchIdx < bufferConfig.routeBatchCount && matchedRouteCount < shardRange.matchOrdinalEnd;
             ++batchIdx) {
            int32_t batchRouteBegin = batchIdx * bufferConfig.routeItemsPerBatch;
            bool isLastBatch = batchIdx == bufferConfig.routeBatchCount - 1;
            int32_t validRouteCount = bufferConfig.routeItemsPerBatch;
            int32_t maskSliceBytes = bufferConfig.routeItemsPerBatch / 8;
            if (isLastBatch) {
                uint64_t sendTotalNum = static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
                validRouteCount = static_cast<int32_t>(sendTotalNum - static_cast<uint64_t>(batchRouteBegin));
                if (batchRouteBegin / 8 + maskSliceBytes > static_cast<int32_t>(context.maskAlignSize)) {
                    maskSliceBytes = static_cast<int32_t>(context.maskAlignSize) - batchRouteBegin / 8;
                }
            }
            remoteRankMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
                params.peermemInfo.maskRecvPtr +
                (static_cast<uint64_t>(localExpertId) * context.common.worldSize + shardRange.remoteRankIdx) *
                    maskSlotSize +
                static_cast<uint64_t>(batchRouteBegin / 8)));
            DataCopy(scratch.maskBatchTensor, remoteRankMaskGlobal, static_cast<uint32_t>(maskSliceBytes));
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
            CreateVecIndex(scratch.topkIndexTensor, batchRouteBegin, bufferConfig.routeItemsPerBatch);
            uint64_t batchMatchedRouteCount = 0;
            GatherMask(scratch.validTopkIndexTensor, scratch.topkIndexTensor, scratch.maskBatchU32Tensor, true,
                       static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0}, batchMatchedRouteCount);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();

            int32_t batchMatchOrdinalBegin = matchedRouteCount;
            int32_t batchMatchOrdinalEnd = matchedRouteCount + static_cast<int32_t>(batchMatchedRouteCount);
            int32_t dispatchMatchOrdinalBegin =
                batchMatchOrdinalBegin > shardRange.matchOrdinalBegin ? batchMatchOrdinalBegin :
                                                                        shardRange.matchOrdinalBegin;
            int32_t dispatchMatchOrdinalEnd =
                batchMatchOrdinalEnd < shardRange.matchOrdinalEnd ? batchMatchOrdinalEnd : shardRange.matchOrdinalEnd;
            if (dispatchMatchOrdinalEnd > dispatchMatchOrdinalBegin) {
                int32_t batchLocalMatchBegin = dispatchMatchOrdinalBegin - batchMatchOrdinalBegin;
                int32_t batchDispatchRowCount = dispatchMatchOrdinalEnd - dispatchMatchOrdinalBegin;
                int32_t dispatchDstRowBegin =
                    static_cast<int32_t>(shardRange.rankSegmentDstRowBegin) + dispatchMatchOrdinalBegin;
                CopyTokensAndMetaForDispatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                    context, params, winRankAddr, scratch, dispatchDstRowBegin, shardRange.remoteRankIdx,
                    batchLocalMatchBegin,
                    batchDispatchRowCount);
                dispatchedRowCount += batchDispatchRowCount;
            }
            matchedRouteCount = batchMatchOrdinalEnd;
        }

        if (dispatchedRowCount > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            int32_t coreExpertRowBegin = shardRange.dstRowBegin - expertGlobalRowBegin;
            int32_t coreExpertRowEnd = coreExpertRowBegin + dispatchedRowCount;
            int32_t firstWaveIdx = coreExpertRowBegin / dispatchWaveRowCount;
            int32_t lastWaveIdx = (coreExpertRowEnd - 1) / dispatchWaveRowCount;
            for (int32_t waveIdx = firstWaveIdx; waveIdx <= lastWaveIdx; ++waveIdx) {
                int32_t waveExpertRowBegin = waveIdx * dispatchWaveRowCount;
                int32_t waveExpertRowEnd = waveExpertRowBegin + dispatchWaveRowCount;
                int32_t overlapRowBegin =
                    coreExpertRowBegin > waveExpertRowBegin ? coreExpertRowBegin : waveExpertRowBegin;
                int32_t overlapRowEnd = coreExpertRowEnd < waveExpertRowEnd ? coreExpertRowEnd : waveExpertRowEnd;
                AtomicAdd(dispatchWaveReadyCount + static_cast<int64_t>(waveIdx) * INT_CACHELINE,
                          overlapRowEnd - overlapRowBegin);
            }
        }
    }
}

// 计算一个专家的接收 token 数，并发布就绪 flag。
template <typename ActivationType, bool EnableA8W4, bool TopkWeightsPrefetch>
__aicore__ inline void ComputeExpertTokenCountAndNotify(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId, uint64_t &sendCnt)
{
    sendCnt = 0;
    uint32_t maskSlotSize = context.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    GlobalTensor<int32_t> countSrcGlobal;
    countSrcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * context.common.worldSize * maskSlotSize +
        context.maskAlignSize));
    DataCopyExtParams countCopyParams{static_cast<uint16_t>(context.common.worldSize),
                                      static_cast<uint32_t>(sizeof(int32_t)),
                                      static_cast<uint32_t>(maskSlotSize - sizeof(int32_t)), 0U, 0U};
    DataCopyPadExtParams<int32_t> countPad{true, 0U, 0U, 0U};
    DataCopyPad(scratch.sendCntTensor, countSrcGlobal, countCopyParams, countPad);

    if constexpr (EnableA8W4) {
        if (localExpertId != 0) {
            DataCopyPad(
                scratch.cumsumInfoTensor, scratch.cumsumInfoGlobalTensor,
                {1U, static_cast<uint32_t>(context.common.worldSize * localExpertId * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
        }
    }
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
    if constexpr (TopkWeightsPrefetch) {
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    }
    constexpr int32_t countStrideI32 = ALIGN_32 / sizeof(int32_t);
    for (uint32_t rankIdx = 0; rankIdx < context.common.worldSize; ++rankIdx) {
        int32_t rankCount = scratch.sendCntTensor.GetValue(rankIdx * countStrideI32);
        sendCnt += static_cast<uint64_t>(rankCount);
        scratch.cumsumRevCntInRank += static_cast<uint64_t>(rankCount);
        scratch.cumsumInfoTensor.SetValue(localExpertId * context.common.worldSize + rankIdx,
                                          static_cast<int32_t>(scratch.cumsumRevCntInRank));
    }

    scratch.expertTokenCntTensor.SetValue(0, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    uint64_t countOffset = localExpertId * INT32_PER_256B * context.countWorkspace.blockNum +
                           INT32_PER_256B * context.countWorkspace.blockIdx;
    DataCopy<int32_t>(scratch.expertRevNumsGlobalTensor[countOffset], scratch.expertTokenCntTensor, INT32_PER_256B);
    if constexpr (EnableA8W4) {
        DataCopyPad(
            scratch.cumsumInfoGlobalTensor, scratch.cumsumInfoTensor,
            {1U, static_cast<uint32_t>(context.common.worldSize * (localExpertId + 1) * sizeof(int32_t)), 0U, 0U, 0U});
    }
    PipeBarrier<PIPE_ALL>();

    __gm__ int32_t *sendCntFlag =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
        static_cast<uint64_t>(localExpertId) * context.countWorkspace.blockNum * INT_CACHELINE +
        static_cast<uint64_t>(context.countWorkspace.blockIdx) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

// 导出 Token Dispatch 生成的逐专家接收 token 数。
template <typename ActivationType, bool ReloadCumsumFromGm>
__aicore__ inline void ExportExpertTokenCounts(const TokenDispatchConfig &context,
                                               TokenDispatchScratch<ActivationType> &scratch,
                                               const Params &params)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if constexpr (ReloadCumsumFromGm) {
        DataCopyPad(scratch.cumsumInfoTensor, scratch.cumsumInfoGlobalTensor,
                    {1U,
                     static_cast<uint32_t>(
                         context.common.worldSize * context.common.moeExpertPerRank * sizeof(int32_t)),
                     0U, 0U, 0U},
                    {true, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    int32_t lastRankIdx = static_cast<int32_t>(context.common.worldSize - 1U);
    scratch.expertTokenNumsOutTensor.SetValue(0, scratch.cumsumInfoTensor.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < static_cast<int32_t>(context.common.moeExpertPerRank); ++expertIdx) {
        int32_t currentCount = scratch.cumsumInfoTensor.GetValue(
            expertIdx * static_cast<int32_t>(context.common.worldSize) + lastRankIdx);
        int32_t previousCount = scratch.cumsumInfoTensor.GetValue(
            (expertIdx - 1) * static_cast<int32_t>(context.common.worldSize) + lastRankIdx);
        scratch.expertTokenNumsOutTensor.SetValue(expertIdx, currentCount - previousCount);
    }
    SyncFuncStatic<HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    GlobalTensor<int32_t> expertTokenNumsOut;
    expertTokenNumsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertTokenNumsOutGmAddr));
    DataCopyPad(expertTokenNumsOut, scratch.expertTokenNumsOutTensor,
                {1U, static_cast<uint32_t>(context.common.moeExpertPerRank * sizeof(int32_t)), 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

struct ExpertTokenCountExportScratch {
    LocalTensor<int32_t> stridedTensor;
    LocalTensor<int32_t> compactTensor;
};

// 从稳定的物理 workspace 槽导出专家 token 数。
__aicore__ inline void ExportExpertTokenCountsFromWorkspace(
    const TokenDispatchConfig &context, const Params &params, ExpertTokenCountExportScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || context.blockJob.jobIndex != 0U) {
        return;
    }
    uint32_t expertPerRank = context.common.moeExpertPerRank;
    GlobalTensor<int32_t> expertRevTokenNums;
    expertRevTokenNums.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.expertRevTokenNumsPtr));
    uint32_t expertCountStrideBytes =
        context.countWorkspace.blockNum * static_cast<uint32_t>(INT32_PER_256B) * sizeof(int32_t);
    DataCopyPad(scratch.stridedTensor,
                expertRevTokenNums[context.countWorkspace.blockIdx * static_cast<uint32_t>(INT32_PER_256B)],
                {static_cast<uint16_t>(expertPerRank), static_cast<uint32_t>(sizeof(int32_t)),
                 expertCountStrideBytes - static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U},
                {true, 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE2_S, SYNC_EVENT_ID2>();
    for (uint32_t expertIdx = 0U; expertIdx < expertPerRank; ++expertIdx) {
        int32_t tokenCount =
            scratch.stridedTensor.GetValue(expertIdx * static_cast<uint32_t>(INT32_PER_256B));
        scratch.compactTensor.SetValue(expertIdx, tokenCount);
    }
    SyncFuncStatic<HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    GlobalTensor<int32_t> expertTokenNumsOut;
    expertTokenNumsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertTokenNumsOutGmAddr));
    DataCopyPad(expertTokenNumsOut, scratch.compactTensor,
                {1U, static_cast<uint32_t>(expertPerRank * sizeof(int32_t)), 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

// 执行一个 MoE 专家的 SendCount 和 Token Dispatch。
template <typename ActivationType, typename QuantScaleType, bool EnableA8W4, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void RunMoeExpertDispatchStage(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t expertIdx, uint64_t &sendCnt)
{
    sendCnt = 0U;
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() != 1U) {
            return;
        }
        ComputeExpertTokenCountAndNotify<ActivationType, EnableA8W4, TopkWeightsPrefetch>(
            context, params, scratch, expertIdx, sendCnt);
        if (sendCnt != 0U) {
            DispatchExpertTokens<ActivationType, QuantScaleType, EnableA8W4, Gmm1TileM, TopkWeightsPrefetch>(
                context, params, winRankAddr, scratch, expertIdx);
        }
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DISPATCH_H
