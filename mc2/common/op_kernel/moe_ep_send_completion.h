/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_EP_SEND_COMPLETION_H
#define MOE_EP_SEND_COMPLETION_H

#include "adv_api/hcomm/hcomm.h"
#include "mc2_moe_context.h"
#include "moe_shared_jetty_context.h"

namespace MoeEpCompletion {
using namespace AscendC;

constexpr uint32_t HCOMM_WORKSPACE_SIZE = 512U;

__aicore__ inline void DrainChannels(__gm__ Mc2Aclnn::MoeCommContext* context, uint32_t worldSize, TPipe* pipe)
{
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    TBuf<TPosition::VECCALC> commBuf;
    pipe->InitBuffer(commBuf, HCOMM_WORKSPACE_SIZE);
    Hcomm<COMM_PROTOCOL_UBC_CTP> comm;
    comm.Init(commBuf.Get<uint8_t>(), HCOMM_WORKSPACE_SIZE);
    uint32_t channelsPerRank = context->channelsPerRank == 0U ? 1U : context->channelsPerRank;
    // Assign one core per channel to consume CQEs. Unused channels have no pending CQEs.
    for (uint32_t index = GetBlockIdx(); index < worldSize * channelsPerRank; index += GetBlockNum()) {
        if (index / channelsPerRank == context->epRankId) {
            continue;
        }
        uint64_t channel = context->hcommHandle[index];
        int32_t ret = comm.Drain(channel);
        ASCENDC_ASSERT(ret == 0, { KERNEL_LOG(KERNEL_ERROR, "MoE send drain failed: %d", ret); });
    }
    GlobalTensor<uint32_t> contextTensor;
    contextTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(context));
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(contextTensor);
}

__aicore__ inline void DrainChannels(__gm__ Mc2Kernel::MoeSharedJettyContext* context, uint32_t worldSize, TPipe* pipe)
{
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    TBuf<TPosition::VECCALC> commBuf;
    pipe->InitBuffer(commBuf, HCOMM_WORKSPACE_SIZE);
    LocalTensor<uint8_t> scratch = commBuf.Get<uint8_t>();
    Duplicate<uint8_t>(scratch, 0U, HCOMM_WORKSPACE_SIZE);
    SyncFunc<HardEvent::V_S>();
    Hcomm<COMM_PROTOCOL_UBC_CTP> comm;
    for (uint32_t jetty = GetBlockIdx(); jetty < context->sharedJettyNum; jetty += GetBlockNum()) {
        auto batch = comm.MakeBatchHandle(static_cast<MultiChannelHandle>(context->multiChannelHandles[jetty]), scratch,
                                          HCOMM_WORKSPACE_SIZE);
        for (uint32_t peer = 0; peer < worldSize; ++peer) {
            if (peer == context->epRankId) {
                continue;
            }
            uint32_t peerIndex = peer < context->epRankId ? peer : peer - 1U;
            GM_ADDR remoteWindow = reinterpret_cast<GM_ADDR>(context->epHcclBuffer[peer]);
            auto& handle = comm.GetHandleRef(batch, peerIndex, remoteWindow);
            int32_t ret = comm.Drain(handle);
            ASCENDC_ASSERT(ret == 0, { KERNEL_LOG(KERNEL_ERROR, "MoE shared Jetty drain failed: %d", ret); });
        }
    }
    GlobalTensor<uint32_t> contextTensor;
    contextTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(context));
    DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(contextTensor);
}

} // namespace MoeEpCompletion
#endif
