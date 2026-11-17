/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_A8_C8_KERNEL_TILING_H_
#define QUANT_FLASH_ATTN_A8_C8_KERNEL_TILING_H_

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include "quant_flash_attn_tiling_data_int8.h"

namespace optiling {

class QuantFlashAttnA8C8TilingData {
public:
    uint32_t batchSize = 0;
    uint32_t totalTokens = 0;
    uint32_t qHeadNum = 0;
    uint32_t kvHeadNum = 0;
    uint32_t groupSize = 0;
    uint32_t headDim = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNum = 0;
    uint32_t maxSeqQ = 0;
    uint32_t maxSeqKv = 0;
    uint32_t coreNum = 0;
    uint32_t kvLayout = 0;
    uint32_t maskMode = 0;
    float softmaxScale = 0.0F;
    uint64_t keyStrideBlock = 0UL;
    uint64_t keyStrideHead = 0UL;
    uint64_t keyStrideSlot = 0UL;
    uint64_t keyStrideDim = 0UL;
    uint64_t keyStrideDBlock = 0UL;
    uint64_t keyStrideInner = 0UL;
    uint64_t valueStrideBlock = 0UL;
    uint64_t valueStrideHead = 0UL;
    uint64_t valueStrideSlot = 0UL;
    uint64_t valueStrideDim = 0UL;
    uint64_t valueStrideDBlock = 0UL;
    uint64_t valueStrideInner = 0UL;
    uint64_t kDescaleStrideBlock = 0UL;
    uint64_t kDescaleStrideHead = 0UL;
    uint64_t kDescaleStrideSlot = 0UL;
    uint64_t vDescaleStrideHead = 0UL;

    uint32_t reservedNativeContext = 0U;
    uint32_t kScaleDenseStride = 0;
    uint64_t kScaleDenseOffset = 0UL;
    uint32_t shortQRoute = 0U;
    uint32_t smallDQtile = 0U;
    uint32_t kvTile = 512U;

    uint32_t get_batchSize() const
    {
        return batchSize;
    }
    uint32_t get_totalTokens() const
    {
        return totalTokens;
    }
    uint32_t get_qHeadNum() const
    {
        return qHeadNum;
    }
    uint32_t get_kvHeadNum() const
    {
        return kvHeadNum;
    }
    uint32_t get_groupSize() const
    {
        return groupSize;
    }
    uint32_t get_headDim() const
    {
        return headDim;
    }
    uint32_t get_blockSize() const
    {
        return blockSize;
    }
    uint32_t get_maxBlockNum() const
    {
        return maxBlockNum;
    }
    uint32_t get_maxSeqQ() const
    {
        return maxSeqQ;
    }
    uint32_t get_maxSeqKv() const
    {
        return maxSeqKv;
    }
    uint32_t get_coreNum() const
    {
        return coreNum;
    }
    uint32_t get_kvLayout() const
    {
        return kvLayout;
    }
    uint32_t get_maskMode() const
    {
        return maskMode;
    }
    float get_softmaxScale() const
    {
        return softmaxScale;
    }
};

struct QfaHostPlanTilingPayload {
    QuantFlashAttnA8C8TilingData config;
    QfaHostPlanData plan;
};
static_assert(std::is_standard_layout<QfaHostPlanTilingPayload>::value &&
                  std::is_trivially_copyable<QfaHostPlanTilingPayload>::value,
              "Paged INT8 tiling must have a stable binary layout");
static_assert(offsetof(QfaHostPlanTilingPayload, plan) == sizeof(QuantFlashAttnA8C8TilingData),
              "Paged INT8 plan must immediately follow its configuration");

constexpr uint32_t QFA_A8C8_KV_LAYOUT_PA_BBND = 0U;
constexpr uint32_t QFA_A8C8_KV_LAYOUT_PA_BNBD = 1U;
constexpr uint32_t QFA_A8C8_KV_LAYOUT_PA_NZ = 2U;
constexpr uint32_t QFA_A8C8_D_BLOCK = 32U;
constexpr uint32_t QFA_A8C8_CUBE_MAX_BATCH = 256U;
constexpr uint32_t QFA_A8C8_CUBE_SEQ_Q_BYTES = ((QFA_A8C8_CUBE_MAX_BATCH + 1U) * sizeof(uint64_t) + 255U) & ~255U;
constexpr uint32_t QFA_A8C8_CUBE_D_MAX = 256U;

constexpr uint32_t QFA_A8C8_CUBE_SLOT_NUM = 2U;

} // namespace optiling

#endif
