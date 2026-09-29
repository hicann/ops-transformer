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

#include "mega_moe_expert_count.h"
#include "../common/mega_moe_utils.h"
#include "mega_moe_token_quant.h"
#include "mega_moe_token_dedup.h"

namespace MegaMoeImpl {

using namespace AscendC;

// Token Dispatch 的阶段私有配置。任务分工(BlockJobContext)与 count 槽位(BlockWorkspaceContext)
// 与 GMM 阶段共用编排层的同一份实例，不在此重复持有。
struct TokenDispatchConfig {
    uint64_t maxOutputSize;
    MegaMoeDispatchBufferConfig bufferConfig;
    uint64_t routeIndexAlignSize;
    uint64_t quantWinOffset;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    DedupConfig dedup; // dedupMode==0 时所有去重字段为空，走基线路径
};

template <typename ActivationType, typename TopkIndexType>
struct TokenDispatchScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
    GlobalTensor<int32_t> cumsumInfoGlobalTensor;
    LocalTensor<TopkIndexType> validTopkIndexTensor;
    LocalTensor<int32_t> cumsumInfoTensor;
    uint32_t copyTmpBaseAddr;
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<int32_t> expertTokenNumsOutTensor;
    int64_t revTokenElemCnt;
    int64_t revScaleElemCnt;
    // 去重暂存区动态基址：紧跟 dispatch 固定 UB 区尾（host 已按 DEDUP_DISPATCH_UB_RESERVE_BYTES 预留）。
    uint32_t dedupUbBaseAddr = 0U;
};

// 当前物理 AIV1 在单个专家内负责的左闭右开 row 区间。
struct ExpertDispatchCoreRange {
    uint32_t expertGlobalRowBegin;
    uint32_t localExpertRowBegin;
    uint32_t localExpertRowEnd;
};

// 基线 dispatch 落地目标的 GM 视图集合（按段起始行 rowDstOffset 偏置）。
template <typename ActivationType, typename QuantScaleType>
struct DispatchCopyTargets {
    GlobalTensor<ActivationType> remoteRank;
    GlobalTensor<ActivationType> tokenRev;
    GlobalTensor<QuantScaleType> scaleRev;
    GlobalTensor<int32_t> metaInfo;
    GlobalTensor<int32_t> revWeights;
};

// 由 tiling/peermem/quant 配置装配 Token Dispatch 阶段配置（普通与 wave 编排模板共用）。
template <typename TopkIndexType>
__aicore__ inline TokenDispatchConfig CreateTokenDispatchConfig(const Params& params,
                                                                const QuantProcessConfig& quantProcessConfig)
{
    uint64_t quantWinOffset =
        static_cast<uint64_t>(params.peermemInfo.quantTokenScalePtr - params.peermemInfo.rankSyncInWorldPtr);
    return {.maxOutputSize = params.tilingData->maxOutputSize,
            .bufferConfig = params.tilingData->dispatchBufferConfig,
            .routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(params.tilingData)),
            .quantWinOffset = quantWinOffset,
            .quantTokenAlignBytes = quantProcessConfig.quantTokenAlignBytes,
            .quantScaleAlignBytes = quantProcessConfig.quantScaleAlignBytes,
            .quantTokenScaleAlignBytes = quantProcessConfig.quantTokenScaleAlignBytes,
            .dedup = CreateDedupConfig(params)};
}

template <typename ActivationType, typename TopkIndexType>
__aicore__ inline LocalTensor<ActivationType> GetDispatchCopyBuffer(
    const TokenDispatchConfig& context, const TokenDispatchScratch<ActivationType, TopkIndexType>& scratch,
    int32_t bufferIdx)
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC,
        scratch.copyTmpBaseAddr + static_cast<uint32_t>(bufferIdx) * context.quantTokenScaleAlignBytes,
        context.quantTokenScaleAlignBytes / sizeof(ActivationType));
}

template <bool IsBufferReuse, bool TopkWeightsPrefetch, typename ActivationType, typename TopkIndexType>
__aicore__ inline void FetchDispatchTokenAndMetaInfo(const TokenDispatchConfig& context, const Params& params,
                                                     TokenDispatchScratch<ActivationType, TopkIndexType>& scratch,
                                                     int32_t bufferIdx, int32_t topkIndex, int32_t remoteRankIdx,
                                                     GlobalTensor<ActivationType>& remoteRankGlobalTensor)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    LocalTensor<ActivationType> copyTmpTensor = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    int32_t tokenIndex = topkIndex / static_cast<int32_t>(params.tilingData->topK);
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
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(
        TOPK_INDEX, topkIndex % static_cast<int32_t>(params.tilingData->topK));
    // 事件分叉与 Store 侧的消费条件同式：记录带权重段（prefetch 或 tkw=0 combine 去重）时
    // Store 要标量读权重，走 MTE2_S；否则直接放行 S_MTE3。
    if (TopkWeightsPrefetch ||
        (params.tilingData->topkWeightsPrefetch == 1 || IsCombineDedupOn(params.tilingData->dedupMode))) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

// copyIdx 同时是段内目标行号：三个调用点的 ring 排空顺序保证写出行与匹配序恒一一对应。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDispatchTokenAndMetaInfo(const TokenDispatchConfig& context, const Params& params,
                                                     TokenDispatchScratch<ActivationType, TopkIndexType>& scratch,
                                                     int32_t bufferIdx,
                                                     GlobalTensor<ActivationType>& tokenRevGlobalTensor,
                                                     GlobalTensor<QuantScaleType>& scaleRevGlobalTensor,
                                                     GlobalTensor<int32_t>& metaInfoGlobalTensor,
                                                     GlobalTensor<int32_t>& revWeightsGlobalTensor,
                                                     int32_t copyStartIdx, int32_t copyIdx)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScaleBuffer = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    LocalTensor<QuantScaleType> scaleBuffer =
        tokenScaleBuffer[context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
    if (TopkWeightsPrefetch ||
        (params.tilingData->topkWeightsPrefetch == 1 || IsCombineDedupOn(params.tilingData->dedupMode))) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = context.quantTokenAlignBytes + context.quantScaleAlignBytes;
        LocalTensor<int32_t> weightBitsTensor = tokenScaleBuffer[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t topkIndex = static_cast<int32_t>(scratch.validTopkIndexTensor.GetValue(copyStartIdx + copyIdx));
        int32_t weightBits =
            weightBitsTensor.GetValue(static_cast<uint32_t>(topkIndex % static_cast<int32_t>(params.tilingData->topK)));
        scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
        // dc 单开（combine 去重 + dispatch 去重关）时本路径是唯一落地入口：每行把整 token
        // 权重段落到自己的 revWeights 槽，供 merge 在 LAST 行一次连续读回乘权。
        if constexpr (!TopkWeightsPrefetch) {
            if (IsCombineDedupOn(params.tilingData->dedupMode)) {
                const uint32_t topK = params.tilingData->topK;
                const uint32_t weightAlignInt32 =
                    Ops::Base::CeilAlign(topK * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32)) /
                    static_cast<uint32_t>(sizeof(int32_t));
                DataCopyPad(revWeightsGlobalTensor[static_cast<uint64_t>(copyIdx) * weightAlignInt32], weightBitsTensor,
                            {1U, topK * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U});
            }
        }
    }
    DataCopyPad(tokenRevGlobalTensor[copyIdx * scratch.revTokenElemCnt], tokenScaleBuffer,
                {1, static_cast<uint16_t>(scratch.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleRevGlobalTensor[copyIdx * scratch.revScaleElemCnt], scaleBuffer,
                {1, static_cast<uint16_t>(scratch.revScaleElemCnt * sizeof(QuantScaleType)), 0U, 0U, 0U});
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    DataCopy(metaInfoGlobalTensor[copyIdx * INT32_PER_256B], scratch.metaInfoTensor[bufferIdx * INT32_PER_256B],
             INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline DispatchCopyTargets<ActivationType, QuantScaleType> CreateDispatchCopyTargets(
    const TokenDispatchConfig& context, const Params& params, GM_ADDR* winRankAddr,
    const TokenDispatchScratch<ActivationType, TopkIndexType>& scratch, int32_t rowDstOffset, int32_t remoteRankIdx)
{
    DispatchCopyTargets<ActivationType, QuantScaleType> targets;
    targets.tokenRev.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(params.workspaceInfo.dispatchRevDataPtr +
                                                                              rowDstOffset * scratch.revTokenElemCnt));
    targets.scaleRev.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleType*>(params.workspaceInfo.dispatchRevScalePtr +
                                                                              rowDstOffset * scratch.revScaleElemCnt));
    targets.remoteRank.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType*>(winRankAddr[remoteRankIdx] + context.quantWinOffset));
    targets.metaInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
        params.workspaceInfo.metaInfoPtr + rowDstOffset * INT32_PER_256B * sizeof(int32_t)));
    // tkw=0 的 combine 去重且 dispatch 去重未开（dc 单开）：本基线路径是唯一落地入口，每行把
    // 整 token 权重段写到自己的 revWeights 槽（merge 在 LAST 行上读，LAST 行也经本路径写过）。
    // full 模式走 DispatchOwnedExpertRowsDedup/StoreDedupEntry，由那边写 LAST 槽，与此互斥。
    if constexpr (!TopkWeightsPrefetch) {
        if (IsCombineDedupOn(params.tilingData->dedupMode)) {
            uint32_t weightAlignInt32 = Ops::Base::CeilAlign(static_cast<uint32_t>(params.tilingData->topK) *
                                                                 static_cast<uint32_t>(sizeof(float)),
                                                             static_cast<uint32_t>(ALIGN_32)) /
                                        static_cast<uint32_t>(sizeof(int32_t));
            targets.revWeights.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                params.workspaceInfo.dispatchRevWeightsPtr +
                static_cast<uint64_t>(rowDstOffset) * weightAlignInt32 * sizeof(int32_t)));
        }
    }
    return targets;
}

template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyTokensAndMetaForDispatch(const TokenDispatchConfig& context, const Params& params,
                                                    GM_ADDR* winRankAddr,
                                                    TokenDispatchScratch<ActivationType, TopkIndexType>& scratch,
                                                    int32_t rowDstOffset, int32_t remoteRankIdx, int32_t copyStartIdx,
                                                    int32_t copyNum)
{
    int32_t bufferCount = context.bufferConfig.bufferCount;
    DispatchCopyTargets<ActivationType, QuantScaleType> targets =
        CreateDispatchCopyTargets<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, rowDstOffset, remoteRankIdx);
    GlobalTensor<ActivationType>& remoteRankGlobalTensor = targets.remoteRank;
    GlobalTensor<ActivationType>& tokenRevGlobalTensor = targets.tokenRev;
    GlobalTensor<QuantScaleType>& scaleRevGlobalTensor = targets.scaleRev;
    GlobalTensor<int32_t>& metaInfoGlobalTensor = targets.metaInfo;
    GlobalTensor<int32_t>& revWeightsGlobalTensor = targets.revWeights;

    int32_t firstTopkIndex = static_cast<int32_t>(scratch.validTopkIndexTensor.GetValue(copyStartIdx));
    FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, params, scratch, 0, firstTopkIndex,
                                                              remoteRankIdx, remoteRankGlobalTensor);

    int32_t firstUseEnd = copyNum < bufferCount ? copyNum : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = static_cast<int32_t>(scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx));
        FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, params, scratch, issueIdx, topkIndex,
                                                                  remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            revWeightsGlobalTensor, copyStartIdx, copyIdx);
    }

    for (int32_t issueIdx = bufferCount; issueIdx < copyNum; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = static_cast<int32_t>(scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx));
        FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(context, params, scratch, issueBufferIdx, topkIndex,
                                                                 remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, copyBufferIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            revWeightsGlobalTensor, copyStartIdx, copyIdx);
    }

    StoreDispatchTokenAndMetaInfo<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, params, scratch, (copyNum - 1) % bufferCount, tokenRevGlobalTensor, scaleRevGlobalTensor,
        metaInfoGlobalTensor, revWeightsGlobalTensor, copyStartIdx, copyNum - 1);

    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// 发布一个 source-rank 段覆盖到的 GMM1 tile ready 计数。
__aicore__ inline void PublishGmm1TileReady(const MoeSyncWorkspaceLayout& syncLayout, const Params& params,
                                            uint32_t localExpertId, int32_t gmm1TileRowCount,
                                            int32_t segmentExpertRowBegin, int32_t segmentExpertRowEnd)
{
    if (segmentExpertRowBegin >= segmentExpertRowEnd) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
    __gm__ int32_t* gmm1TileReadyCount =
        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * syncLayout.dispatchFlagSlotCountPerExpert;
    int32_t firstTileIdx = segmentExpertRowBegin / gmm1TileRowCount;
    int32_t lastTileIdx = (segmentExpertRowEnd - 1) / gmm1TileRowCount;
    for (int32_t tileIdx = firstTileIdx; tileIdx <= lastTileIdx; ++tileIdx) {
        int32_t tileExpertRowBegin = tileIdx * gmm1TileRowCount;
        int32_t tileExpertRowEnd = tileExpertRowBegin + gmm1TileRowCount;
        int32_t overlapRowBegin =
            segmentExpertRowBegin > tileExpertRowBegin ? segmentExpertRowBegin : tileExpertRowBegin;
        int32_t overlapRowEnd = segmentExpertRowEnd < tileExpertRowEnd ? segmentExpertRowEnd : tileExpertRowEnd;
        AtomicAdd(gmm1TileReadyCount + static_cast<int64_t>(tileIdx) * INT_CACHELINE, overlapRowEnd - overlapRowBegin);
    }
}

/*
 * 按单个 source rank 的 compact route-index ordinal 区间搬运 token 和 metadata。
 * compact index 已按 [expert][source rank] 分槽保存，因此这里只需按 ordinal 分批顺序读取，无需再次筛选 mask。
 */
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline int32_t DispatchRankTokens(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                                             const Params& params, GM_ADDR* winRankAddr,
                                             TokenDispatchScratch<ActivationType, TopkIndexType>& scratch,
                                             uint32_t localExpertId, int32_t expertGlobalRowBegin,
                                             uint32_t remoteRankIdx, int32_t rankSegmentRowBegin,
                                             int32_t segmentMatchOrdinalBegin, int32_t segmentMatchOrdinalEnd)
{
    const MegaMoeDispatchBufferConfig& bufferConfig = context.bufferConfig;
    int32_t processedRouteCount = 0;
    int32_t dispatchedRowCount = 0;
    GlobalTensor<TopkIndexType> remoteRouteIndexGlobal;
    while (processedRouteCount < segmentMatchOrdinalEnd - segmentMatchOrdinalBegin) {
        int32_t remainingRouteCount = segmentMatchOrdinalEnd - segmentMatchOrdinalBegin - processedRouteCount;
        int32_t batchDispatchRowCount = remainingRouteCount < bufferConfig.routeItemsPerBatch ?
                                            remainingRouteCount :
                                            bufferConfig.routeItemsPerBatch;
        int32_t routeOrdinal = segmentMatchOrdinalBegin + processedRouteCount;
        // 直接定位当前专家、source rank 的 route ordinal，不再扫描完整 topK mask。
        uint64_t slotOffset =
            (static_cast<uint64_t>(localExpertId) * common.worldSize + remoteRankIdx) * context.routeIndexAlignSize;
        remoteRouteIndexGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ TopkIndexType*>(
            params.peermemInfo.maskRecvPtr + slotOffset + static_cast<uint64_t>(routeOrdinal) * sizeof(TopkIndexType)));
        DataCopyExtParams routeCopyParams{1U, static_cast<uint32_t>(batchDispatchRowCount * sizeof(TopkIndexType)), 0U,
                                          0U, 0U};
        DataCopyPadExtParams<TopkIndexType> routeCopyPad{false, 0U, 0U, 0U};
        DataCopyPad(scratch.validTopkIndexTensor, remoteRouteIndexGlobal, routeCopyParams, routeCopyPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID4>();
        // route ordinal 同时决定专家输出行位置，保证分批搬运后仍保持专家内连续布局。
        int32_t dispatchDstRowBegin = rankSegmentRowBegin + expertGlobalRowBegin + routeOrdinal;
        CopyTokensAndMetaForDispatch<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, dispatchDstRowBegin, remoteRankIdx, 0, batchDispatchRowCount);
        processedRouteCount += batchDispatchRowCount;
        dispatchedRowCount += batchDispatchRowCount;
    }

    return dispatchedRowCount;
}

/*
 * 在当前专家的 source-rank 累计行数中二分查找 globalRowIdx 所属的首个 rank。
 * 返回值作为 DispatchOwnedExpertRows 的扫描起点，后续只访问与本核范围相交的连续 rank 段。
 */
template <typename ActivationType, typename TopkIndexType>
__aicore__ inline uint32_t FindDispatchSourceRank(const MoeStageCommonConfig& common,
                                                  const TokenDispatchScratch<ActivationType, TopkIndexType>& scratch,
                                                  uint32_t localExpertId, uint32_t globalRowIdx)
{
    uint32_t expertRankBegin = localExpertId * common.worldSize;
    uint32_t searchBegin = expertRankBegin;
    uint32_t searchEnd = expertRankBegin + common.worldSize;
    while (searchBegin < searchEnd) {
        uint32_t middle = searchBegin + (searchEnd - searchBegin) / 2U;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(middle));
        if (rankGlobalRowEnd <= globalRowIdx) {
            searchBegin = middle + 1U;
        } else {
            searchEnd = middle;
        }
    }
    return searchBegin - expertRankBegin;
}

/*
 * 搬运当前物理 AIV1 在一个专家内负责的连续 row。
 * 该范围先按 source-rank prefix 拆段，再从 compact index 槽搬运 token/metadata；全部 rank 段完成后，
 * 按实际覆盖行数累加 GMM1 tile ready。AtomicAdd 允许同一 tile 被多个 range 分段发布。
 */
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchOwnedExpertRows(const TokenDispatchConfig& context, const MoeStageCommonConfig& common,
                                               const MoeSyncWorkspaceLayout& syncLayout, const Params& params,
                                               GM_ADDR* winRankAddr,
                                               TokenDispatchScratch<ActivationType, TopkIndexType>& scratch,
                                               uint32_t localExpertId, const ExpertDispatchCoreRange& coreRange)
{
    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    uint32_t coreGlobalRowBegin = coreRange.expertGlobalRowBegin + coreRange.localExpertRowBegin;
    uint32_t coreGlobalRowEnd = coreRange.expertGlobalRowBegin + coreRange.localExpertRowEnd;
    // prefix 单调递增，二分找到与本核首行相交的第一个 source rank，避免从 rank 0 线性扫描。
    uint32_t sourceRankIdx = FindDispatchSourceRank(common, scratch, localExpertId, coreGlobalRowBegin);
    int32_t dispatchedRowCount = 0;

    while (sourceRankIdx < common.worldSize) {
        uint32_t prefixIndex = localExpertId * common.worldSize + sourceRankIdx;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex));
        uint32_t rankGlobalRowBegin =
            prefixIndex == 0U ? 0U : static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex - 1U));
        if (rankGlobalRowBegin >= coreGlobalRowEnd) {
            break;
        }

        uint32_t overlapGlobalRowBegin =
            coreGlobalRowBegin > rankGlobalRowBegin ? coreGlobalRowBegin : rankGlobalRowBegin;
        uint32_t overlapGlobalRowEnd = coreGlobalRowEnd < rankGlobalRowEnd ? coreGlobalRowEnd : rankGlobalRowEnd;
        if (overlapGlobalRowBegin < overlapGlobalRowEnd) {
            int32_t rankDispatchedRowCount =
                DispatchRankTokens<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                    context, common, params, winRankAddr, scratch, localExpertId,
                    static_cast<int32_t>(coreRange.expertGlobalRowBegin), sourceRankIdx,
                    static_cast<int32_t>(rankGlobalRowBegin - coreRange.expertGlobalRowBegin),
                    static_cast<int32_t>(overlapGlobalRowBegin - rankGlobalRowBegin),
                    static_cast<int32_t>(overlapGlobalRowEnd - rankGlobalRowBegin));
            dispatchedRowCount += rankDispatchedRowCount;
        }
        ++sourceRankIdx;
    }
    // 所有 source-rank 子段完成后统一发布本核贡献；GMM1 按 tile 累计到完整行数后即可启动。
    PublishGmm1TileReady(syncLayout, params, localExpertId, gmm1TileRowCount,
                         static_cast<int32_t>(coreRange.localExpertRowBegin),
                         static_cast<int32_t>(coreRange.localExpertRowBegin) + dispatchedRowCount);
}

/*
 * Dispatch 唯一执行入口：搬运调用方给定的左闭右开专家 token 范围，并发布 GMM1 tile ready。
 * 调用方负责提前规划范围、推进 WAVE/专家位置以及在必要时恢复 prefix；本函数只做范围裁剪、AIV1 分工、
 * 专家/source-rank 拆段和数据发送。A8W8 可传整 WAVE，W4 也可传单个专家 slice，执行路径完全一致。
 */
// 计算本核负责的全局行区间：先按 maxOutputSize（Dispatch workspace 硬边界）统一裁剪，再对整个
// 输入范围一次连续均分；按全局起点轮转首 owner，避免余数长期集中在低编号 AIV1。
// 返回 false = 本核无行可做。
__aicore__ inline bool CalcDispatchCoreRowRange(const TokenDispatchConfig& context, const BlockJobContext& blockJob,
                                                const ExpertTokenRange& range, uint32_t& coreGlobalRowBegin,
                                                uint32_t& coreGlobalRowEnd)
{
    uint32_t dispatchGlobalRowBegin = static_cast<uint32_t>(
        range.begin.globalTokenIndex < context.maxOutputSize ? range.begin.globalTokenIndex : context.maxOutputSize);
    uint32_t dispatchGlobalRowEnd = static_cast<uint32_t>(
        range.end.globalTokenIndex < context.maxOutputSize ? range.end.globalTokenIndex : context.maxOutputSize);
    if (dispatchGlobalRowEnd <= dispatchGlobalRowBegin) {
        return false;
    }
    uint32_t dispatchRowCount = dispatchGlobalRowEnd - dispatchGlobalRowBegin;
    WorkRange ownedRange =
        GetRotatedBalancedWorkRange(dispatchRowCount, blockJob.jobIndex, blockJob.totalJobs, dispatchGlobalRowBegin);
    if (ownedRange.count == 0U) {
        return false;
    }
    coreGlobalRowBegin = dispatchGlobalRowBegin + ownedRange.start;
    coreGlobalRowEnd = coreGlobalRowBegin + ownedRange.count;
    return true;
}

// dispatch 发布 gate：AIV0 参与建表档，发布任何 tile 前等全体 AIV0 建表完成。

/*
 * count 表的装载与本地计算段（从 PrepareMoeExpertTokenCountTable 提取，语义逐字不变）：
 * 装载原始 count → 逐槽 epoch 到达校验 → 前缀和。不含 GM 发布与 ready 标志。
 * AIV1 经 PrepareMoeExpertTokenCountTable 调用；AIV0 参与去重 prepass 建表时直接调用本函数，
 * 在自己的 UB 里得到一份本地前缀表（发布仍由 AIV1 独担；count 槽只被读，无并发写）。
 */
template <typename ActivationType, typename TopkIndexType>
__aicore__ inline void LoadAndComputeExpertCountTable(const MoeStageCommonConfig& common, const Params& params,
                                                      TokenDispatchScratch<ActivationType, TopkIndexType>& scratch)
{
    uint32_t rawCountElementCount = common.worldSize * common.moeExpertPerRank;

    /*
     * count 的本轮 epoch 代替入口跨卡握手：发送前已等待全部 MoE 量化和对应 index 写回。
     * 本核已在输入准备阶段更新同步计数，直接读取本核计数生成预期标签。
     * 整表搬入 UB 后批量检查本轮标签，包括零 count；失败退避重试，成功快照直接用于 cumsum。
     */
    auto* syncCountAddr = GetSyncCountAddress(params.peermemInfo.rankSyncInWorldPtr, GetBlockIdx());
    const int32_t syncCount = ReadGmBypassDCache(syncCountAddr);
    const uint32_t expectedSyncRoundTag = GetSyncRoundTag(syncCount);
    GlobalTensor<int32_t> expertCountGlobal;
    expertCountGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.peermemInfo.expertCountRecvPtr));
    // MoE 阶段入口已通过 MTE3_MTE2 完成这段 UB 的交接。
    // expertTokenNumsOutTensor 尚未保存输出，首项暂存整表检查结果，不增加 UB 分配。
    WaitForCountTable(expertCountGlobal, scratch.cumsumInfoTensor, scratch.expertTokenNumsOutTensor,
                      rawCountElementCount, expectedSyncRoundTag);
    ComputeExpertCountTables(scratch.cumsumInfoTensor, scratch.expertTokenNumsOutTensor, common.moeExpertPerRank,
                             common.worldSize, params.tilingData->maxOutputSize);
    // 后续只使用 UB 结果及 workspace 副本，不再读取 peermem 原始 count 表。
    CrossCoreSetFlag<ALL_AICORE_SYNC_MODE, PIPE_V>(COUNT_TABLE_READ_DONE_FLAG);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
}

// Wave 进入逐专家流水前一次准备完整 count 表；每个物理 block 只发布一次 ready。
// W4 Wave-ahead 路径同时将 prefix 持久化到 GM，避免后续 Activation 覆盖 UB 后丢失 Dispatch 状态。
template <bool NeedCumsumReload = false, typename ActivationType, typename TopkIndexType>
__aicore__ inline void PrepareMoeExpertTokenCountTable(const MoeStageCommonConfig& common,
                                                       const BlockWorkspaceContext& countWorkspace,
                                                       const Params& params,
                                                       TokenDispatchScratch<ActivationType, TopkIndexType>& scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        // AIV0 不读取 count，但 mode 0 需要全部 AIV 每轮各上报一次；参与去重建表的 AIV0
        // 经 PrepareDedupPrepassAiv0Local -> LoadAndComputeExpertCountTable 已上报，此处不得重复。
        // 条件必须与 PrepareDedupPrepassAiv0Local 的启用门严格互补（含 hiddenDim 档位），
        // 否则计算主导档的 AIV0 无人上报，ExportAndResetExpertCounts 等 read-done 挂死。
        if (params.tilingData->dedupMode == 0 || params.tilingData->hiddenDim > DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM) {
            CrossCoreSetFlag<ALL_AICORE_SYNC_MODE, PIPE_V>(COUNT_TABLE_READ_DONE_FLAG);
        }
        return;
    }
    LoadAndComputeExpertCountTable(common, params, scratch);
    uint32_t rawCountElementCount = common.worldSize * common.moeExpertPerRank;
    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    DataCopyPad(scratch.expertRevNumsGlobalTensor[countOffset], scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    if constexpr (NeedCumsumReload) {
        DataCopyPad(scratch.cumsumInfoGlobalTensor, scratch.cumsumInfoTensor,
                    {1U, rawCountElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID2>();

    __gm__ int32_t* countTableReady =
        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
        static_cast<uint64_t>(countWorkspace.blockIdx) * INT_CACHELINE;
    WriteGmBypassDCache(countTableReady, static_cast<int32_t>(1));
}

/*
 * W4 的 Activation/SwiGLU 会覆盖 AIV1 UB 0 开始的 prefix 表。调用方在执行 DispatchTokenRange 前，
 * 从当前物理 block 的 GM 备份恢复“首专家前缀 + 本次范围覆盖的专家前缀”。
 */
template <typename ActivationType, typename TopkIndexType>
__aicore__ inline void ReloadDispatchCumsumRange(const MoeStageCommonConfig& common,
                                                 TokenDispatchScratch<ActivationType, TopkIndexType>& scratch,
                                                 uint32_t firstExpertIdx, uint32_t lastExpertIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    if (firstExpertIdx >= common.moeExpertPerRank || firstExpertIdx > lastExpertIdx) {
        return;
    }
    if (lastExpertIdx >= common.moeExpertPerRank) {
        lastExpertIdx = common.moeExpertPerRank - 1U;
    }
    constexpr uint32_t INT32_PER_32B = ALIGN_32 / sizeof(int32_t);
    uint32_t requiredBeginIndex = firstExpertIdx == 0U ? 0U : firstExpertIdx * common.worldSize - 1U;
    uint32_t alignedBeginIndex = requiredBeginIndex / INT32_PER_32B * INT32_PER_32B;
    uint32_t requiredEndIndex = (lastExpertIdx + 1U) * common.worldSize;
    uint32_t reloadElementCount = requiredEndIndex - alignedBeginIndex;
    DataCopyPad(scratch.cumsumInfoTensor[alignedBeginIndex], scratch.cumsumInfoGlobalTensor[alignedBeginIndex],
                {1U, reloadElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
}

template <typename ActivationType, typename TopkIndexType>
__aicore__ inline void ExportExpertTokenCounts(const MoeStageCommonConfig& common,
                                               const BlockWorkspaceContext& countWorkspace, const Params& params,
                                               TokenDispatchScratch<ActivationType, TopkIndexType>& scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || countWorkspace.blockIdx != 0U) {
        return;
    }

    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    GlobalTensor<int32_t> expertRecvTokenCount;
    expertRecvTokenCount.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.expertRecvTokenCountPtr));
    DataCopyPad(scratch.expertTokenNumsOutTensor, expertRecvTokenCount[countOffset],
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE2_MTE3, SYNC_EVENT_ID2>();

    GlobalTensor<int32_t> expertTokenNumsOut;
    expertTokenNumsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.expertTokenNumsOutGmAddr));
    DataCopyPad(expertTokenNumsOut, scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE3_S, SYNC_EVENT_ID2>();
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DISPATCH_H
