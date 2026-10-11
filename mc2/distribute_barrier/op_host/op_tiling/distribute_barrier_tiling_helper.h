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
 * \file distribute_barrier_tiling_helper.h
 * \brief
 */

#ifndef DISTRIBUTE_BARRIER_TILING_HELPER_H
#define DISTRIBUTE_BARRIER_TILING_HELPER_H

#include <cstdint>
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint64_t INIT_TILINGKEY = 10000UL;

constexpr uint32_t INPUT_CONTEXT_EXTEND_INDEX = 0U;
constexpr uint32_t INPUT_X_INDEX = 0U;
constexpr uint32_t INPUT_X_EXTEND_INDEX = 1U;
constexpr uint32_t TIME_OUT_INDEX = 1U;
constexpr uint32_t TIME_OUT_EXTEND_INDEX = 2U;
constexpr uint32_t ELASTIC_INFO_INDEX = 2U;
constexpr uint32_t ELASTIC_INFO_EXTEND_INDEX = 3U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t ELASTIC_METAINFO_OFFSET = 4U;
constexpr uint32_t RANK_LIST_NUM = 2U;

constexpr uint32_t OUTPUT_X_INDEX = 0U;
constexpr uint32_t ATTR_GROUP_INDEX = 0U;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1U;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;

const int64_t MIN_WORLD_SIZE = 2;
const int64_t MAX_WORLD_SIZE_A3 = 384;
const int64_t MAX_WORLD_SIZE_A5 = 1024;

struct DistributeBarrierConfig {
    uint32_t contextIndex = INPUT_CONTEXT_EXTEND_INDEX; // 根据disributeBarrierExtend算子原型标志位初始化context索引
    uint32_t xRefIndex = INPUT_X_INDEX;             // 根据disributeBarrier算子原型标志位初始化xRef索引
    uint32_t timeOutIndex = TIME_OUT_INDEX;         // 根据disributeBarrier算子原型标志位初始化timeOut索引
    uint32_t elasticInfoIndex = ELASTIC_INFO_INDEX; // 根据disributeBarrier算子原型标志位初始化elasticInfo索引
    uint32_t xRefOutIndex = OUTPUT_X_INDEX;         // 根据disributeBarrier算子原型标志位设置xRefOut索引
    uint32_t attrGroupIndex = ATTR_GROUP_INDEX; // 根据disributeBarrier算子原型标志位初始化attrGroupIndex索引
    uint32_t attrWorldSizeIndex = ATTR_WORLD_SIZE_INDEX; // 根据disributeBarrier算子原型标志位设置attrWorldSizeIndex索引
    bool isMc2Context = false;
};

ge::graphStatus DistributeBarrierExtendTilingFuncNew(gert::TilingContext* context,
                                                     const DistributeBarrierConfig& config);

} // namespace optiling
#endif // DISTRIBUTE_BARRIER_TILING_HELPER_H
