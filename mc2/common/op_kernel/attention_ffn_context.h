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

#include "mc2_moe_context.h"

namespace Mc2Aclnn {
struct AttentionFFNContext {
    uint32_t epRankId;
    uint32_t rankSizePerServer;
    uint64_t kfcContextAddr; // host kfc方案中，需要传递通信API所需的地址
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE]; // 支持ROCE或者URMA
};
} // namespace Mc2Aclnn

#endif // ATTENTION_FFN_CONTEXT_H
