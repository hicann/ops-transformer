/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attention_ffn_context.h
 * \brief
 */
#ifndef ATTENTION_FFN_CONTEXT_H
#define ATTENTION_FFN_CONTEXT_H

#include <cstddef>
#include "mc2_moe_context.h"

namespace Mc2Aclnn {
struct AttentionFFNContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr; // host kfc方案中，需要传递通信API所需的地址
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE]; // 支持ROCE或者URMA，按 rank*channelsPerRank+channel 布局
    uint32_t channelsPerRank;                  // 每个 rank 的 channel 数量
    bool enableGroupSend;
    uint64_t roceHcclBuffer_[HCCL_MAX_RANK_SIZE];
    uint64_t roceBufferSize; // 本地与各对端注册窗容量一致。
    uint64_t hostStatusTableAddr;
    uint64_t completedRounds;
    // ---- 持久 payload 区元数据（与 host 侧 CommContext 尾部字段二进制一致）----
    uint64_t payloadBufferAddr;
    uint64_t payloadBufferSize;
    uint64_t payloadSlotStride;
    uint32_t payloadSlotNum;
};
// 状态表字段必须紧跟channelsPerRank（含对齐填充），与host侧CommContext的二进制布局一致。
static_assert(offsetof(AttentionFFNContext, completedRounds) ==
                  offsetof(AttentionFFNContext, hostStatusTableAddr) + sizeof(uint64_t),
              "AttentionFFNContext keepalive tail fields must stay contiguous uint64 pairs");
static_assert(offsetof(AttentionFFNContext, payloadBufferAddr) ==
                  offsetof(AttentionFFNContext, completedRounds) + sizeof(uint64_t),
              "AttentionFFNContext payload fields must stay contiguous after completedRounds");
} // namespace Mc2Aclnn

#endif // ATTENTION_FFN_CONTEXT_H
