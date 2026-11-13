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
 * \file dense_lightning_indexer_softmax_lse_v2_metadata.h
 * \brief Runtime metadata consumed by dense_lightning_indexer_softmax_lse_v2.
 */
#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_H

#include <cstdint>

namespace optiling {

using DLI_METADATA_T = int32_t;

constexpr uint32_t DLI_METADATA_MAX_CORE_NUM = 36;
constexpr uint32_t DLI_METADATA_SIZE = 64;

constexpr uint32_t DLI_SCHED_CORE_NUM = 64;
constexpr uint32_t DLI_SCHED_ITEM_MAX_NUM = 65536;
constexpr uint32_t DLI_SCHED_FLAG_IDX = 4;
constexpr uint32_t DLI_SCHED_CNT_IDX = 8;
constexpr uint32_t DLI_SCHED_OFF_IDX = 72;
constexpr uint32_t DLI_SCHED_ITEM_IDX = 136;
constexpr uint32_t DLI_SCHED_ENTRY_STRIDE = 8;
constexpr uint32_t DLI_METADATA_MAX_SIZE = DLI_SCHED_ITEM_IDX + DLI_SCHED_ENTRY_STRIDE * DLI_SCHED_ITEM_MAX_NUM;

inline constexpr int64_t DliSchedMetadataSize(int64_t itemBound)
{
    if (itemBound <= 0) {
        return static_cast<int64_t>(DLI_METADATA_SIZE);
    }
    if (itemBound > static_cast<int64_t>(DLI_SCHED_ITEM_MAX_NUM)) {
        itemBound = static_cast<int64_t>(DLI_SCHED_ITEM_MAX_NUM);
    }
    return static_cast<int64_t>(DLI_SCHED_ITEM_IDX) + static_cast<int64_t>(DLI_SCHED_ENTRY_STRIDE) * itemBound;
}

namespace detail {
struct DenseLISoftmaxLseV2MetaData {
    int32_t forecore_num;
    int32_t tail_core_num;
    int32_t b_s1_per_core;
    int32_t b_s1_per_tail_core;
};
} // namespace detail

static_assert(DLI_METADATA_SIZE * sizeof(DLI_METADATA_T) >= sizeof(detail::DenseLISoftmaxLseV2MetaData));

} // namespace optiling

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_V2_METADATA_H
