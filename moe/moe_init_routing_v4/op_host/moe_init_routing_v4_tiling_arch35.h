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
 * \file moe_init_routing_v4_tiling_arch35.h
 * \brief
 */
#ifndef MOE_INIT_ROUTING_V4_TILING_ARCH35_H_
#define MOE_INIT_ROUTING_V4_TILING_ARCH35_H_

#include "../../moe_init_routing_v3/op_host/moe_init_routing_v3_tiling_arch35.h"

namespace optiling {
constexpr int64_t V4_INPUT_X_INDEX = 0LL;
constexpr int64_t V4_INPUT_EXPERT_IDX_INDEX = 1LL;
constexpr int64_t V4_INPUT_SCALE_INDEX = 2LL;
constexpr int64_t V4_INPUT_OFFSET_INDEX = 3LL;
constexpr int64_t V4_INPUT_ACTIVE_NUM_INDEX = 4LL;
constexpr int64_t V4_INPUT_TOPK_WEIGHT_INDEX = 5LL;

constexpr int64_t V4_ATTR_EXPERT_CAPACITY_INDEX = 0LL;
constexpr int64_t V4_ATTR_EXPERT_NUM_INDEX = 1LL;
constexpr int64_t V4_ATTR_DROP_PAD_MODE_INDEX = 2LL;
constexpr int64_t V4_ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX = 3LL;
constexpr int64_t V4_ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX = 4LL;
constexpr int64_t V4_ATTR_QUANT_MODE_INDEX = 5LL;
constexpr int64_t V4_ATTR_EXPERT_RANGE_INDEX = 6LL;
constexpr int64_t V4_ATTR_ROW_IDX_TYPE_INDEX = 7LL;

constexpr int64_t V4_OUTPUT_EXPANDED_X_INDEX = 0LL;
constexpr int64_t V4_OUTPUT_EXPANDED_ROW_IDX_INDEX = 1LL;
constexpr int64_t V4_OUTPUT_EXPERT_TOKENS_COUNT_INDEX = 2LL;
constexpr int64_t V4_OUTPUT_EXPANDED_SCALE_INDEX = 3LL;
constexpr int64_t V4_OUTPUT_EXPANDED_TOPK_WEIGHT_INDEX = 4LL;
} // namespace optiling

#endif
