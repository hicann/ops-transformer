/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_GRAD_UTILS_H
#define ENGRAM_FETCH_GRAD_UTILS_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Mc2Kernel {
constexpr uint32_t MAX_QP_SIZE = 1024U;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t TILE_BYTES = 64U * 1024U;
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t ALIGNED_LEN_256 = 256U;

constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t NUM_SLOTS = 8U;
constexpr uint32_t SENDER_CHANNEL_IDX = 0U;
constexpr uint32_t RECEIVER_CHANNEL_IDX = 1U;

constexpr int32_t ENGRAM_DT_BFLOAT16 = 27;
constexpr int32_t ENGRAM_DT_FLOAT16 = 1;
constexpr int32_t ENGRAM_DT_FLOAT = 0;

constexpr uint32_t ENTRY_BATCH_CAP = 1024U;
constexpr uint32_t GRAD_SUB_BATCH = 8U;
constexpr uint32_t SENDCOUNT_STRIDE_RATIO = 8U;

struct EngramCommContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[MAX_QP_SIZE];
    uint64_t hcommHandle[MAX_QP_SIZE * 2];
    uint32_t channelsPerRank;
};

} // namespace Mc2Kernel

#endif
