/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SEND_MASK_H
#define MEGA_MOE_SEND_MASK_H

#include <type_traits>

#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

struct SendMaskConfig {
    uint64_t expertCountWinOffset;
    uint64_t routeIndexAlignSize;
    uint64_t routeIndexWinOffset;
    MegaMoeSendMaskBufferConfig bufferConfig;
};

template <typename TopkIndexType>
struct SendMaskScratch {
    GlobalTensor<int32_t> topkIdsGm;
    LocalTensor<int32_t> topkIdsTensor;
    // 当前批次元素在完整 topkIds 数组中的下标。
    LocalTensor<TopkIndexType> topkIdsIndexTensor;
    LocalTensor<uint8_t> routeRingTensor;
    LocalTensor<int32_t> sendCntAccTensor;
};

struct ExpertRouteInfo {
    int32_t ownedIdx;
    int32_t globalExpertId;
    int32_t bufferIdx;
};

// 装配 MTE 路径的 topK 有效下标发送配置。
template <typename TopkIndexType>
__aicore__ inline SendMaskConfig CreateSendMaskConfig(const Params &params, uint32_t aivCoreIdx)
{
    uint64_t routeIndexWinOffset =
        static_cast<uint64_t>(params.peermemInfo.maskRecvPtr - params.peermemInfo.rankSyncInWorldPtr);
    uint64_t expertCountWinOffset =
        static_cast<uint64_t>(params.peermemInfo.expertCountRecvPtr - params.peermemInfo.rankSyncInWorldPtr);
    const MegaMoeSendMaskBufferConfig &bufferConfig = aivCoreIdx < params.tilingData->sendMaskCoreCountWithExtraExpert ?
                                                          params.tilingData->sendMaskConfigForCoreWithExtraExpert :
                                                          params.tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    return {.expertCountWinOffset = expertCountWinOffset,
            .routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(params.tilingData)),
            .routeIndexWinOffset = routeIndexWinOffset,
            .bufferConfig = bufferConfig};
}

// 加载当前批次的 topkIds，并生成它们在完整数组中的下标。
template <typename TopkIndexType>
__aicore__ inline void PrepareTopkIdsForCurrentBatch(SendMaskScratch<TopkIndexType> &scratch, int32_t batchStart,
                                                     int32_t validLen)
{
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(scratch.topkIdsTensor, scratch.topkIdsGm[batchStart], loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    CreateVecIndex(scratch.topkIdsIndexTensor, static_cast<TopkIndexType>(batchStart), validLen);
}

/**
 * 筛选当前批次中匹配专家号的 topkIds 下标，写入当前 ring 槽并返回命中数量。
 * 调用方须先等待该槽上次搬出完成；返回前完成 V_S，调用方可读取命中数量。
 */
template <typename TopkIndexType>
__aicore__ inline uint64_t SelectTopkIdsIndexByExpert(const SendMaskConfig &config,
                                                      SendMaskScratch<TopkIndexType> &scratch,
                                                      const ExpertRouteInfo &routeInfo, int32_t validLen,
                                                      LocalTensor<TopkIndexType> &selectedTopkIdsIndexTensor)
{
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    const uint32_t compareMaskBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) / BITS_PER_BYTE;
    uint32_t slotOffset = routeInfo.bufferIdx * bufferConfig.bufferBytes;
    LocalTensor<uint8_t> compareMaskTensor = scratch.routeRingTensor[slotOffset];
    // 保存 topkIdsTensor 中值等于当前专家号的元素所对应的下标位置。
    selectedTopkIdsIndexTensor =
        scratch.routeRingTensor[slotOffset + compareMaskBytes].template ReinterpretCast<TopkIndexType>();

    uint64_t selectedTopkIdsIndexCount = 0U;
    if (validLen > 0) {
        CompareScalar(compareMaskTensor, scratch.topkIdsTensor, routeInfo.globalExpertId, AscendC::CMPMODE::EQ,
                      validLen);
        using CompareMaskPatternType =
            typename std::conditional<sizeof(TopkIndexType) == sizeof(int16_t), uint16_t, uint32_t>::type;
        LocalTensor<CompareMaskPatternType> compareMaskPatternTensor =
            compareMaskTensor.template ReinterpretCast<CompareMaskPatternType>();
        GatherMask(selectedTopkIdsIndexTensor, scratch.topkIdsIndexTensor, compareMaskPatternTensor, true,
                   static_cast<uint32_t>(validLen), {1, 1, 0, 0}, selectedTopkIdsIndexCount);
    }
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    return selectedTopkIdsIndexCount;
}

// 将筛选出的下标追加到专家所在卡，按剩余容量更新累计发送量。
// 调用方在发送后发出 ring 槽复用事件，保护下次筛选对该槽的覆盖。
template <typename TopkIndexType>
__aicore__ inline void SendSelectedTopkIdsIndex(const MoeStageCommonConfig &common, GM_ADDR *winRankAddr,
                                                const SendMaskConfig &config, SendMaskScratch<TopkIndexType> &scratch,
                                                const ExpertRouteInfo &routeInfo,
                                                const LocalTensor<TopkIndexType> &selectedTopkIdsIndexTensor,
                                                uint64_t selectedTopkIdsIndexCount)
{
    int32_t previousCount = scratch.sendCntAccTensor.GetValue(routeInfo.ownedIdx);
    int32_t remainingCapacity = static_cast<int32_t>(common.tokenNum) - previousCount;
    remainingCapacity = remainingCapacity > 0 ? remainingCapacity : 0;
    int32_t copiedCount = static_cast<int32_t>(selectedTopkIdsIndexCount);
    copiedCount = remainingCapacity > copiedCount ? copiedCount : remainingCapacity;
    scratch.sendCntAccTensor.SetValue(routeInfo.ownedIdx, previousCount + copiedCount);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

    if (copiedCount > 0) {
        int32_t expertPerRank = static_cast<int32_t>(common.moeExpertPerRank);
        int32_t dstRank = routeInfo.globalExpertId / expertPerRank;
        int32_t localExpertId = routeInfo.globalExpertId % expertPerRank;
        uint64_t dstOffset = config.routeIndexWinOffset +
                             static_cast<uint64_t>(localExpertId * static_cast<int32_t>(common.worldSize) +
                                                   static_cast<int32_t>(common.rankId)) *
                                 config.routeIndexAlignSize +
                             static_cast<uint64_t>(previousCount) * sizeof(TopkIndexType);
        GlobalTensor<TopkIndexType> dstRouteIndexGm;
        dstRouteIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ TopkIndexType *>(winRankAddr[dstRank] + dstOffset));
        DataCopyPad(dstRouteIndexGm, selectedTopkIdsIndexTensor,
                    {1U, static_cast<uint32_t>(copiedCount * sizeof(TopkIndexType)), 0U, 0U, 0U});
    }
}

// 发送到一张目的卡的连续专家 count；源起点不对齐时借用下标缓冲，搬出完成后才允许再次覆盖。
// 调用方已完成 count 编码及 S_MTE3，并在全部目的卡发送结束后统一等待 MTE3_S。
template <typename TopkIndexType>
__aicore__ inline void SendTopkIdsCountToRank(const WorkRange &sourceCountRange, int32_t localExpertBegin,
                                              GM_ADDR dstRankAddr, const MoeStageCommonConfig &common,
                                              const SendMaskConfig &config,
                                              const SendMaskScratch<TopkIndexType> &scratch)
{
    int32_t sourceRank = static_cast<int32_t>(common.rankId);
    int32_t worldSize = static_cast<int32_t>(common.worldSize);
    int32_t ownedOffset = static_cast<int32_t>(sourceCountRange.start);
    int32_t segmentExpertCount = static_cast<int32_t>(sourceCountRange.count);
    GlobalTensor<int32_t> dstCountGm;
    uint64_t dstOffset = config.expertCountWinOffset +
                         static_cast<uint64_t>(localExpertBegin * worldSize + sourceRank) * sizeof(int32_t);
    dstCountGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dstRankAddr + dstOffset));
    DataCopyExtParams countCopyParams{static_cast<uint16_t>(segmentExpertCount), static_cast<uint32_t>(sizeof(int32_t)),
                                      0, static_cast<int64_t>(worldSize - 1) * static_cast<int64_t>(sizeof(int32_t)),
                                      0U};

    if (ownedOffset % static_cast<int32_t>(INT32_PER_256B) == 0) {
        DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.sendCntAccTensor[ownedOffset], countCopyParams);
    } else {
        // topk index 已发送完成，复用 int32 topkIdsTensor 将非对齐 count 段重排到对齐起点。
        for (int32_t expertIdx = 0; expertIdx < segmentExpertCount; ++expertIdx) {
            scratch.topkIdsTensor.SetValue(expertIdx, scratch.sendCntAccTensor.GetValue(ownedOffset + expertIdx));
        }
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
        DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.topkIdsTensor, countCopyParams);
        SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
    }
}

// 发送各专家的 topkIds 下标数量及本轮 epoch，目标布局为 [localExpert][sourceRank]。
// 仅由 AIV 调用，专家范围非空；紧接同一范围的下标发送调用，期间不能覆盖 scratch 中的累计数量。
template <typename TopkIndexType>
__aicore__ inline void SendTopkIdsCountForExperts(const WorkRange &ownedExpertRange, const MoeStageCommonConfig &common,
                                                  __gm__ int32_t *launchCountSlot, GM_ADDR *winRankAddr,
                                                  const SendMaskConfig &config,
                                                  const SendMaskScratch<TopkIndexType> &scratch)
{
    int32_t ownedExpertBegin = static_cast<int32_t>(ownedExpertRange.start);
    int32_t ownedExpertNum = static_cast<int32_t>(ownedExpertRange.count);
    // 本 launch epoch = 计数槽值+1(跨卡同步在本阶段之后执行,槽值仍为上一 launch)。
    int32_t arrivalEpoch = ((ReadGmBypassDCache(launchCountSlot) + 1) & 0x7F) | 0x80;
    /*
     * 到达标记内嵌：count 与跨卡同步信号走不同源核/不同通道(MTE3 vs scalar)，互连不保证
     * 到达序，接收端凭同步放行后可能读到在途 count(旧序靠共享 GMM1 时间垫层掩盖)。
     * 判据必须内嵌在数据自身的单次写内：高 8 位写 launch epoch(值域[0x80,0xFF]恒非零,
     * 避开窗口零初值)，低 24 位为真实 count(上限 maxOutputSize 远小于 2^24)。
     * 接收端逐槽校验 epoch 后取低 24 位(见 token_dispatch.h PrepareMoeExpertTokenCountTable)。
     */
    constexpr uint32_t EXPERT_COUNT_EPOCH_SHIFT = 24U;
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
        int32_t rawCount = scratch.sendCntAccTensor.GetValue(ownedIdx);
        scratch.sendCntAccTensor.SetValue(ownedIdx, (arrivalEpoch << EXPERT_COUNT_EPOCH_SHIFT) | rawCount);
    }

    const int32_t expertPerRank = static_cast<int32_t>(common.moeExpertPerRank);
    const int32_t ownedExpertEnd = ownedExpertBegin + ownedExpertNum;
    const int32_t firstDstRank = ownedExpertBegin / expertPerRank;
    const int32_t lastDstRank = (ownedExpertEnd - 1) / expertPerRank;

    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    for (int32_t dstRank = firstDstRank; dstRank <= lastDstRank; ++dstRank) {
        const int32_t dstRankExpertBegin = dstRank * expertPerRank;
        const int32_t dstRankExpertEnd = dstRankExpertBegin + expertPerRank;
        // 仅首卡和末卡可能只发送部分专家，中间卡发送全部专家；End 为不包含在内的右边界。
        const int32_t sendExpertBegin = dstRank == firstDstRank ? ownedExpertBegin : dstRankExpertBegin;
        const int32_t sendExpertEnd = dstRank == lastDstRank ? ownedExpertEnd : dstRankExpertEnd;
        const WorkRange sourceCountRange{.start = static_cast<uint32_t>(sendExpertBegin - ownedExpertBegin),
                                         .count = static_cast<uint32_t>(sendExpertEnd - sendExpertBegin)};
        SendTopkIdsCountToRank(sourceCountRange, sendExpertBegin - dstRankExpertBegin, winRankAddr[dstRank], common,
                               config, scratch);
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
}

// 仅由 AIV 调用，为指定的非空连续专家范围发送 topkIds 下标，同时累计各专家的发送数量。
// 返回前等待所有 ring 槽搬出完成，随后由调用方发送累计数量。
template <typename TopkIndexType>
__aicore__ inline void SendTopkIdsIndexForExperts(const WorkRange &ownedExpertRange, const MoeStageCommonConfig &common,
                                                  GM_ADDR *winRankAddr, const SendMaskConfig &config,
                                                  SendMaskScratch<TopkIndexType> &scratch)
{
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    int32_t ownedExpertBegin = static_cast<int32_t>(ownedExpertRange.start);
    int32_t ownedExpertNum = static_cast<int32_t>(ownedExpertRange.count);

    Duplicate<int32_t>(scratch.sendCntAccTensor, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
    for (int32_t batchIdx = 0; batchIdx < bufferConfig.routeBatchCount; ++batchIdx) {
        const int32_t batchStart = batchIdx * bufferConfig.routeItemsPerBatch;
        const int32_t realSendTotalNum =
            static_cast<int32_t>(static_cast<uint64_t>(common.tokenNum) * static_cast<uint64_t>(common.topK));
        const int32_t realRemain = realSendTotalNum - batchStart;
        int32_t validLen = bufferConfig.routeItemsPerBatch;
        if (realRemain < validLen) {
            validLen = realRemain > 0 ? realRemain : 0;
        }

        if (validLen > 0) {
            SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
            PrepareTopkIdsForCurrentBatch(scratch, batchStart, validLen);
        }

        const int32_t batchRingBegin = batchIdx * ownedExpertNum;
        for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
            int32_t globalExpertId = ownedExpertBegin + ownedIdx;
            ExpertRouteInfo routeInfo{.ownedIdx = ownedIdx,
                                      .globalExpertId = globalExpertId,
                                      .bufferIdx = (batchRingBegin + ownedIdx) % bufferConfig.bufferCount};
            // 复用当前 ring 槽前，等待上次下标搬出完成。
            WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(routeInfo.bufferIdx));
            LocalTensor<TopkIndexType> selectedTopkIdsIndexTensor;
            uint64_t selectedTopkIdsIndexCount =
                SelectTopkIdsIndexByExpert(config, scratch, routeInfo, validLen, selectedTopkIdsIndexTensor);
            SendSelectedTopkIdsIndex(common, winRankAddr, config, scratch, routeInfo, selectedTopkIdsIndexTensor,
                                     selectedTopkIdsIndexCount);
            // 零命中时也归还当前槽，与下一次复用或尾部等待配对。
            SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(routeInfo.bufferIdx));
        }
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_SEND_MASK_H
