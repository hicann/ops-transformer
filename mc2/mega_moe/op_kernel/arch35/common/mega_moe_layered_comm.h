/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_COMM_H
#define MEGA_MOE_LAYERED_COMM_H

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
#include "adv_api/hcomm/hcomm.h"
#include "../../../../common/op_kernel/mc2_moe_context.h"

namespace MegaMoeImpl {

using namespace AscendC;

// 封装 Layered 路径使用的 URMA 通信接口。
class LayeredUrmaComm {
public:
    using BatchHandle = AscendC::BatchHandle<AscendC::ChannelHandle>;

    __aicore__ inline explicit LayeredUrmaComm(__gm__ Mc2Aclnn::Mc2MoeContext* context)
        : context_(context),
          rankId_(context->epRankId)
    {}

    __aicore__ inline ChannelHandle GetChannel(uint32_t peer) const
    {
        const uint32_t index = peer > rankId_ ? peer - 1U : peer;
        return context_->hcommHandle_[index];
    }

    __aicore__ inline int32_t Init(const LocalTensor<uint8_t>& scratch, uint32_t bytes)
    {
        return hcomm_.Init(scratch, bytes);
    }

    __aicore__ inline int32_t Lock(ChannelHandle channel)
    {
        return hcomm_.Lock(channel);
    }
    __aicore__ inline int32_t Unlock(ChannelHandle channel)
    {
        return hcomm_.Unlock(channel);
    }

    __aicore__ inline BatchHandle MakeBatchHandle(ChannelHandle channel, const LocalTensor<uint8_t>& buffer,
                                                  uint32_t bytes, GM_ADDR remoteAddr)
    {
        return hcomm_.MakeBatchHandle(channel, buffer, bytes, remoteAddr);
    }

    // 使用 Hcomm 默认写配置，生成 CQE。
    __aicore__ inline int32_t WriteNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t bytes)
    {
        return hcomm_.WriteNbi(channel, dst, src, bytes);
    }

    __aicore__ inline int32_t WriteNbi(BatchHandle& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return hcomm_.WriteNbi(batch, dst, src, bytes);
    }

    __aicore__ inline int32_t ReadNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t bytes)
    {
        return hcomm_.ReadNbi(channel, dst, src, bytes);
    }

    __aicore__ inline int32_t ReadNbi(BatchHandle& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return hcomm_.ReadNbi(batch, dst, src, bytes);
    }

    // 弱执行保序写（ODR=5），配置完成保序，不生成 CQE。
    __aicore__ inline int32_t WriteRelaxedNbi(BatchHandle& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return hcomm_.WriteNbi<RELAXED_WRITE_WQE_CONFIG>(batch, dst, src, bytes);
    }

    // 强执行保序写（ODR=6），配置完成保序，不生成 CQE。
    __aicore__ inline int32_t WriteStrongNbi(BatchHandle& batch, GM_ADDR dst, GM_ADDR src, uint32_t bytes)
    {
        return hcomm_.WriteNbi<STRONG_WRITE_WQE_CONFIG>(batch, dst, src, bytes);
    }

    __aicore__ inline int32_t BatchCommit(BatchHandle& batch)
    {
        return hcomm_.BatchCommit(batch);
    }
    __aicore__ inline int32_t Drain(BatchHandle& batch)
    {
        return hcomm_.Drain(batch);
    }
    __aicore__ inline int32_t Drain(ChannelHandle channel)
    {
        return hcomm_.Drain(channel);
    }

private:
    static constexpr struct UrmaWqeEntry RELAXED_WRITE_WQE_CONFIG = {
        .odr = 5, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
    static constexpr struct UrmaWqeEntry STRONG_WRITE_WQE_CONFIG = {
        .odr = 6, .fence = 1, .se = 0, .cqe = 0, .inlineEn = 0};
    __gm__ Mc2Aclnn::Mc2MoeContext* const context_;
    const uint32_t rankId_;
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
};

} // namespace MegaMoeImpl
#endif
#endif
