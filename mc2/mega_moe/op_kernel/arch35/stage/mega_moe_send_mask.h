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
    uint64_t maskWinOffset;
    MegaMoeSendMaskBufferConfig bufferConfig;
};

struct SendMaskScratch {
    LocalTensor<int32_t> topkIdsTensor;
    // Contiguous runtime-sized ring. Slot i starts at i * bufferConfig.bufferBytes.
    LocalTensor<uint8_t> sendMaskTensor;
    LocalTensor<int32_t> sendGatherOutTensor;
    LocalTensor<int32_t> sendCntAccTensor;
};

__aicore__ inline void GatherAndSendExpertMaskBatch(
    const DispatchPrepareConfig &context, GM_ADDR *winRankAddr, const SendMaskConfig &config,
    SendMaskScratch &scratch,
    GlobalTensor<int32_t> &topkIdsGm, GlobalTensor<uint8_t> &dstMaskGm, int32_t ownedExpertNum,
    int32_t batchIdx, int32_t &ringIteration)
{
    const AivJobContext &job = context.job;
    const MoeStageCommonConfig &common = context.common;
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    uint32_t maskSlotSize = config.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    int32_t batchStart = batchIdx * bufferConfig.routeItemsPerBatch;
    bool isLastBatch = batchIdx == bufferConfig.routeBatchCount - 1;
    int32_t validLen = bufferConfig.routeItemsPerBatch;
    int32_t sliceBytes = bufferConfig.routeItemsPerBatch / 8;
    int32_t pushBytes = sliceBytes;
    if (isLastBatch) {
        uint64_t sendTotalNum = static_cast<uint64_t>(common.tokenNum) * common.topK;
        validLen = static_cast<int32_t>(sendTotalNum - static_cast<uint64_t>(batchStart));
        if (batchStart / 8 + sliceBytes > static_cast<int32_t>(config.maskAlignSize)) {
            sliceBytes = static_cast<int32_t>(config.maskAlignSize) - batchStart / 8;
        }
        pushBytes = sliceBytes + static_cast<int32_t>(sizeof(int32_t));
    }

    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(scratch.topkIdsTensor, topkIdsGm[batchStart], loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    int32_t jobIndex = static_cast<int32_t>(job.jobIndex);
    int32_t totalJobs = static_cast<int32_t>(job.totalJobs);
    for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx, ++ringIteration) {
        int32_t globalExpertId = jobIndex + ownedIdx * totalJobs;
        int32_t dstRank = globalExpertId / static_cast<int32_t>(common.moeExpertPerRank);
        int32_t localExpertId = globalExpertId % static_cast<int32_t>(common.moeExpertPerRank);
        int32_t bufferIdx = ringIteration % bufferConfig.bufferCount;
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        LocalTensor<uint8_t> maskBuf = scratch.sendMaskTensor[bufferIdx * bufferConfig.bufferBytes];
        LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(eventId);
        CompareScalar(maskBuf, scratch.topkIdsTensor, globalExpertId, AscendC::CMPMODE::EQ,
                      bufferConfig.routeItemsPerBatch);
        uint64_t batchMatchedRouteCount = 0;
        GatherMask(scratch.sendGatherOutTensor, scratch.topkIdsTensor, maskBufU32, true,
                   static_cast<uint32_t>(validLen), {1, 1, 0, 0}, batchMatchedRouteCount);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

        int32_t expertMatchedRouteCount =
            scratch.sendCntAccTensor.GetValue(ownedIdx) + static_cast<int32_t>(batchMatchedRouteCount);
        scratch.sendCntAccTensor.SetValue(ownedIdx, expertMatchedRouteCount);
        if (isLastBatch) {
            maskBuf.template ReinterpretCast<int32_t>().SetValue(sliceBytes / sizeof(int32_t),
                                                                 expertMatchedRouteCount);
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

// Prototype: MegaMoe::SendMaskCal. Builds and sends per-expert masks for one logical AIV job using route batches.
__aicore__ inline void GatherAndSendExpertMasks(const DispatchPrepareConfig &context, const Params &params,
                                                GM_ADDR *winRankAddr, const SendMaskConfig &config,
                                                SendMaskScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const AivJobContext &job = context.job;
    const MegaMoeSendMaskBufferConfig &bufferConfig = config.bufferConfig;
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs) {
        return;
    }

    const MoeStageCommonConfig &common = context.common;
    int32_t totalExperts = static_cast<int32_t>(common.worldSize * common.moeExpertPerRank);
    int32_t jobIndex = static_cast<int32_t>(job.jobIndex);
    int32_t totalJobs = static_cast<int32_t>(job.totalJobs);
    int32_t ownedExpertNum = jobIndex < totalExperts ? Ops::Base::CeilDiv(totalExperts - jobIndex, totalJobs) : 0;
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

    int32_t ringIteration = 0;
    for (int32_t batchIdx = 0; batchIdx < bufferConfig.routeBatchCount; ++batchIdx) {
        GatherAndSendExpertMaskBatch(
            context, winRankAddr, config, scratch, topkIdsGm, dstMaskGm, ownedExpertNum, batchIdx, ringIteration);
    }

    for (int32_t bufIdx = 0; bufIdx < bufferConfig.bufferCount; ++bufIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufIdx));
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_SEND_MASK_H
