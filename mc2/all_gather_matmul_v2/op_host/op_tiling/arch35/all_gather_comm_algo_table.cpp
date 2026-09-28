/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_comm_algo_table.h"
#include "mc2_tiling_utils.h"
#include "hccl/hccl_rank_graph.h"

namespace Mc2Tiling {

constexpr uint32_t MIN_LAYERS = 1;
constexpr uint32_t MAX_LAYERS_UBX = 1;
constexpr uint32_t MAX_LAYERS_SERVER = 2;
constexpr uint32_t MIN_DATA_BYTES_ZERO = 0;
constexpr uint64_t MAX_DATA_BYTES_UNLIMITED = ~0ULL;
constexpr uint32_t RANK_SIZE_2 = 2;
constexpr uint32_t MAX_RANK_SIZE = ~0U;

enum CommAlgoPriority : int32_t {
    DEFAULT = 0,
    MEDIUM = 2,
};

// UBX 场景 AllGather 通信算法表（算法名为 HCCL 新版算法规格串）
// 未命中场景（其他卡数、两层网络、拓扑不匹配等）由末两条 DEFAULT 兜底表项回退默认算法
static constexpr Mc2Hcom::CommAlgoEntry ALLGATHER_COMM_ALGO_TABLE[] = {
    // 全场景回退默认算法
    {mc2tiling::A5_AICPU_TS_ENGINE, Mc2Hcom::WILDCARD_TOPO_TYPE, MIN_LAYERS, MAX_LAYERS_SERVER, MIN_DATA_BYTES_ZERO,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_2, MAX_RANK_SIZE, DEFAULT, ALLGATHER_DEFAULT_ALGO_NAME},
    {mc2tiling::A5_CCU_ENGINE, Mc2Hcom::WILDCARD_TOPO_TYPE, MIN_LAYERS, MAX_LAYERS_SERVER, MIN_DATA_BYTES_ZERO,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_2, MAX_RANK_SIZE, DEFAULT, ALLGATHER_DEFAULT_ALGO_NAME},
};

const Mc2Hcom::CommAlgoEntry *GetAllGatherCommAlgoTable(uint32_t &count)
{
    count = sizeof(ALLGATHER_COMM_ALGO_TABLE) / sizeof(ALLGATHER_COMM_ALGO_TABLE[0]);
    return ALLGATHER_COMM_ALGO_TABLE;
}
} // namespace Mc2Tiling
