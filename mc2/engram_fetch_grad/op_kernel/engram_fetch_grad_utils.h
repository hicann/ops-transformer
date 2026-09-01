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

#include "engram_fetch_grad_tiling_data.h"

// 确定性失败兜底：用于不可恢复的容量/一致性校验（Kernel 侧无返回值路径的最后手段）
#ifndef RUNTIME_ABORT
#define RUNTIME_ABORT(fmt, ...) \
    do { \
        ascendc_assert(false, fmt, ##__VA_ARGS__); \
        while (true) { \
            (void)AscendC::GetSystemCycle(); \
        } \
    } while (0)
#endif

namespace Mc2Kernel {
// 共享布局常量已收敛至 engram_fetch_grad_tiling_data.h（单一权威定义），此处仅保留 Kernel 侧私有常量
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t ALIGNED_LEN_256 = 256U;

constexpr uint32_t NUM_SLOTS = 8U;
constexpr uint32_t SENDER_CHANNEL_IDX = 0U;
constexpr uint32_t RECEIVER_CHANNEL_IDX = 1U;

constexpr int32_t ENGRAM_DT_BFLOAT16 = 27;
constexpr int32_t ENGRAM_DT_FLOAT16 = 1;
constexpr int32_t ENGRAM_DT_FLOAT = 0;

constexpr uint32_t GRAD_SUB_BATCH = 8U;
constexpr uint32_t SENDCOUNT_STRIDE_RATIO = 8U;

constexpr uint32_t MAX_BLOCK_BYTES = 65504U; // 单次 DataCopy 块上限 65535
constexpr uint32_t MAX_PENDING_HANDLES = 8U;
constexpr uint32_t SORT_TMP_BUCKET_BYTES = 512U;
constexpr uint32_t SORT_TMP_BYTES_PER_ELEM = 7U;
constexpr uint32_t SORT_COUNT_ALIGN = 32U;

struct EngramCommContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[MAX_QP_SIZE];
    uint64_t hcommHandle[MAX_QP_SIZE * 2];
    uint32_t channelsPerRank;
};

} // namespace Mc2Kernel

#endif
