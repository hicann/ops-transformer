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

#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

struct SendMaskConfig {
    uint32_t maskAlignSize;
    // 单个 (expert, srcRank) win 槽位 = maskAlignSize(mask 位区) + 32B(count 区)，装配时算一次。
    uint32_t maskSlotSize;
    uint64_t maskWinOffset;
    uint64_t expertCountWinOffset;
    bool publishExpertCountTable;
    MegaMoeSendMaskBufferConfig bufferConfig;
};

/*
 * 装配 send-mask 阶段配置（普通与 wave 编排模板共用）。
 * 发送侧按 (专家, 目标卡) 往对端 peermem 窗口写 mask 槽：槽内先是按 32B 对齐的 mask 位图
 * （每条候选路由 1 bit），末尾是 32B 的 count 区。位图大小必须调用与窗口开设时同一个函数
 * （CalcDispatchMaskAlignSize）来算——两边各写一份公式的话，一旦改动不同步，
 * 发送写入的槽位就会与接收方读取的槽位错开。
 */
__aicore__ inline SendMaskConfig CreateSendMaskConfig(const Params &params, uint32_t aivCoreIdx,
                                                      bool publishExpertCountTable = false)
{
    uint32_t maskAlignSize = static_cast<uint32_t>(CalcDispatchMaskAlignSize(params.tilingData));
    uint64_t maskWinOffset =
        static_cast<uint64_t>(params.peermemInfo.maskRecvPtr - params.peermemInfo.rankSyncInWorldPtr);
    uint64_t expertCountWinOffset =
        static_cast<uint64_t>(params.peermemInfo.expertCountRecvPtr - params.peermemInfo.rankSyncInWorldPtr);
    const MegaMoeSendMaskBufferConfig &bufferConfig = aivCoreIdx < params.tilingData->sendMaskCoreCountWithExtraExpert ?
                                                          params.tilingData->sendMaskConfigForCoreWithExtraExpert :
                                                          params.tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    return {.maskAlignSize = maskAlignSize,
            .maskSlotSize = maskAlignSize + static_cast<uint32_t>(ALIGN_32),
            .maskWinOffset = maskWinOffset,
            .expertCountWinOffset = expertCountWinOffset,
            .publishExpertCountTable = publishExpertCountTable,
            .bufferConfig = bufferConfig};
}

struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor;
    // Contiguous runtime-sized ring. Slot i starts at i * bufferConfig.bufferBytes.
    LocalTensor<uint8_t> sendMaskTensor;
    LocalTensor<int32_t> sendGatherOutTensor;
    LocalTensor<int32_t> sendCntAccTensor;
};

__aicore__ inline void GatherAndSendExpertMaskBatch(const MoeStageCommonConfig &common, GM_ADDR *winRankAddr,
                                                    const SendMaskConfig &config, SendMaskScratch &scratch,
                                                    GlobalTensor<int32_t> &topkIdsGm, GlobalTensor<uint8_t> &dstMaskGm,
                                                    int32_t ownedExpertBegin, int32_t ownedExpertNum, int32_t batchIdx)
{
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    uint32_t maskSlotSize = config.maskSlotSize;
    int32_t batchStart = batchIdx * bufferConfig.routeItemsPerBatch;
    bool isLastBatch = batchIdx == bufferConfig.routeBatchCount - 1;
    // 批网格按全卡一致的上界(numMaxTokensPerRank*topK)划分, 本卡真实路由数是 tokenNum*topK,
    // 可能小于网格覆盖量, 装载长度逐批夹紧, 避免越界读本卡 topkIds。
    // validLen 之后的 UB 残留数据经 CompareScalar 可能产生无效 mask 位; 这些位排在有效位之后,
    // 且 GatherMask 只统计 validLen 内的匹配, 槽尾 count 不含无效位, 接收端按 count 做序数
    // 夹紧时会跳过它们(见 token_dispatch.h DispatchExpertTokens), 因此不需要额外清零。
    int32_t realSendTotalNum = static_cast<int32_t>(common.tokenNum * common.topK);
    int32_t realRemain = realSendTotalNum - batchStart;
    int32_t validLen = bufferConfig.routeItemsPerBatch;
    if (realRemain < validLen) {
        validLen = realRemain > 0 ? realRemain : 0;
    }
    int32_t sliceBytes = bufferConfig.routeItemsPerBatch / 8;
    int32_t pushBytes = sliceBytes;
    if (isLastBatch) {
        if (batchStart / 8 + sliceBytes > static_cast<int32_t>(config.maskAlignSize)) {
            sliceBytes = static_cast<int32_t>(config.maskAlignSize) - batchStart / 8;
        }
        // 非 layered Wave 路径从独立 count 表读取；mask 尾部 count 仅为 layered 兼容布局保留。
        pushBytes = sliceBytes + (config.publishExpertCountTable ? 0 : static_cast<int32_t>(sizeof(int32_t)));
    }

    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    if (validLen > 0) {
        DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
        DataCopyPad(scratch.topkIdsTensor, topkIdsGm[batchStart], loadParams, loadPad);
    }
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    int32_t batchRingBegin = batchIdx * ownedExpertNum;
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx) {
        int32_t globalExpertId = ownedExpertBegin + ownedIdx;
        int32_t dstRank = globalExpertId / static_cast<int32_t>(common.moeExpertPerRank);
        int32_t localExpertId = globalExpertId % static_cast<int32_t>(common.moeExpertPerRank);
        // 环形槽位按 (batch, ownedExpert) 的全局迭代序推进，与 MTE3_V 事件编号一一对应。
        int32_t bufferIdx = (batchRingBegin + ownedIdx) % bufferConfig.bufferCount;
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        LocalTensor<uint8_t> maskBuf = scratch.sendMaskTensor[bufferIdx * bufferConfig.bufferBytes];
        LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        CompareScalar(maskBuf, scratch.topkIdsTensor, globalExpertId, AscendC::CMPMODE::EQ,
                      bufferConfig.routeItemsPerBatch);
        uint64_t batchMatchedRouteCount = 0;
        // validLen==0 表示本批全在真实路由之外(上界网格的富余批), 没有匹配计数, 仍推送全 0 mask(末批带 count)
        if (validLen > 0) {
            GatherMask(scratch.sendGatherOutTensor, scratch.topkIdsTensor, maskBufU32, true,
                       static_cast<uint32_t>(validLen), {1, 1, 0, 0}, batchMatchedRouteCount);
        }
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

        int32_t expertMatchedRouteCount =
            scratch.sendCntAccTensor.GetValue(ownedIdx) + static_cast<int32_t>(batchMatchedRouteCount);
        scratch.sendCntAccTensor.SetValue(ownedIdx, expertMatchedRouteCount);
        if (isLastBatch && !config.publishExpertCountTable) {
            maskBuf.template ReinterpretCast<int32_t>().SetValue(sliceBytes / sizeof(int32_t), expertMatchedRouteCount);
        }
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

        uint64_t dstOffset = config.maskWinOffset +
                             static_cast<uint64_t>(localExpertId * static_cast<int32_t>(common.worldSize) +
                                                   static_cast<int32_t>(common.rankId)) *
                                 maskSlotSize +
                             static_cast<uint64_t>(batchStart / 8);
        dstMaskGm.SetGlobalBuffer((__gm__ uint8_t *)(winRankAddr[dstRank] + dstOffset));
        DataCopyPad(dstMaskGm, maskBuf, {1U, static_cast<uint32_t>(pushBytes), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
    }
}

// 将本核连续专家区间的 raw count 发布到各目标 rank 的 [localExpert][sourceRank] 表。
__aicore__ inline void PublishExpertCounts(const MoeStageCommonConfig &common, GM_ADDR *winRankAddr,
                                           const SendMaskConfig &config, const SendMaskScratch &scratch,
                                           int32_t ownedExpertBegin, int32_t ownedExpertNum)
{
    if (!config.publishExpertCountTable || ownedExpertNum <= 0) {
        return;
    }

    int32_t sourceRank = static_cast<int32_t>(common.rankId);
    int32_t expertPerRank = static_cast<int32_t>(common.moeExpertPerRank);
    int32_t worldSize = static_cast<int32_t>(common.worldSize);
    int32_t ownedOffset = 0;
    GlobalTensor<int32_t> dstCountGm;

    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    while (ownedOffset < ownedExpertNum) {
        int32_t globalExpertId = ownedExpertBegin + ownedOffset;
        int32_t dstRank = globalExpertId / expertPerRank;
        int32_t localExpertBegin = globalExpertId % expertPerRank;
        int32_t remainingInRank = expertPerRank - localExpertBegin;
        int32_t segmentExpertCount = ownedExpertNum - ownedOffset;
        segmentExpertCount = segmentExpertCount < remainingInRank ? segmentExpertCount : remainingInRank;

        uint64_t dstOffset = config.expertCountWinOffset +
                             static_cast<uint64_t>(localExpertBegin * worldSize + sourceRank) * sizeof(int32_t);
        dstCountGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(winRankAddr[dstRank] + dstOffset));
        DataCopyExtParams countCopyParams{
            static_cast<uint16_t>(segmentExpertCount), static_cast<uint32_t>(sizeof(int32_t)), 0,
            static_cast<int64_t>(worldSize - 1) * static_cast<int64_t>(sizeof(int32_t)), 0U};

        if (ownedOffset % static_cast<int32_t>(INT32_PER_256B) == 0) {
            DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.sendCntAccTensor[ownedOffset],
                                                       countCopyParams);
        } else {
            // 跨目标 rank 后源 UB 起点可能不满足 MTE 对齐，先重排到已对齐 scratch。
            for (int32_t expertIdx = 0; expertIdx < segmentExpertCount; ++expertIdx) {
                scratch.sendGatherOutTensor.SetValue(expertIdx,
                                                     scratch.sendCntAccTensor.GetValue(ownedOffset + expertIdx));
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
            DataCopyPad<int32_t, PaddingMode::Compact>(dstCountGm, scratch.sendGatherOutTensor, countCopyParams);
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
        }
        ownedOffset += segmentExpertCount;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
}

// Prototype: MegaMoe::SendMaskCal. Builds and sends per-expert masks for one logical AIV job using route batches.
__aicore__ inline void GatherAndSendExpertMasks(const AivJobContext &job, const MoeStageCommonConfig &common,
                                                const Params &params, GM_ADDR *winRankAddr,
                                                const SendMaskConfig &config, SendMaskScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs) {
        return;
    }

    int32_t totalExperts = static_cast<int32_t>(common.worldSize * common.moeExpertPerRank);
    int32_t jobIndex = static_cast<int32_t>(job.jobIndex);
    int32_t totalJobs = static_cast<int32_t>(job.totalJobs);
    int32_t expertsPerJob = totalExperts / totalJobs;
    int32_t jobCountWithExtraExpert = totalExperts % totalJobs;
    int32_t ownedExpertNum = expertsPerJob + (jobIndex < jobCountWithExtraExpert ? 1 : 0);
    int32_t ownedExpertBegin =
        jobIndex * expertsPerJob + (jobIndex < jobCountWithExtraExpert ? jobIndex : jobCountWithExtraExpert);
    if (ownedExpertNum <= 0) {
        return;
    }

    GlobalTensor<int32_t> topkIdsGm;
    topkIdsGm.SetGlobalBuffer((__gm__ int32_t *)params.expertIdxGmAddr);
    GlobalTensor<uint8_t> dstMaskGm;
    Duplicate<int32_t>(scratch.sendCntAccTensor, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    for (int32_t bufIdx = 0; bufIdx < bufferConfig.bufferCount; ++bufIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufIdx));
    }

    for (int32_t batchIdx = 0; batchIdx < bufferConfig.routeBatchCount; ++batchIdx) {
        GatherAndSendExpertMaskBatch(common, winRankAddr, config, scratch, topkIdsGm, dstMaskGm, ownedExpertBegin,
                                     ownedExpertNum, batchIdx);
    }

    for (int32_t bufIdx = 0; bufIdx < bufferConfig.bufferCount; ++bufIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufIdx));
    }
    PublishExpertCounts(common, winRankAddr, config, scratch, ownedExpertBegin, ownedExpertNum);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_SEND_MASK_H
