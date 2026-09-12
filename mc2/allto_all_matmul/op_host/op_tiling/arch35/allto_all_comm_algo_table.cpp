/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allto_all_comm_algo_table.h"
#include "hccl/hccl_rank_graph.h"

namespace Mc2Tiling {

constexpr uint8_t ENGINE_AICPU = 2;
constexpr uint8_t ENGINE_CCU = 6;
constexpr uint64_t MAX_DATA_BYTES = ~0ULL;
constexpr uint32_t MAX_RANK_SIZE = ~0U;

static constexpr Mc2Hcom::CommAlgoEntry ALLTOALL_COMM_ALGO_TABLE[] = {
    // AICPU + CUSTOM + 单层 + 卡数[2,MAX] → AicpuAllToAllSoleMeshUBX
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 2, "AicpuAllToAllSoleMeshUBX"},
    // AICPU + CUSTOM + 单层 + 卡数[2,MAX] → AicpuAllToAllSoleMeshConcurrent
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 1, "AicpuAllToAllSoleMeshConcurrent"},
    // AICPU + CUSTOM + 两层 + 卡数[2,MAX] → AicpuAllToAllSoleMeshSingleChannel
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 2,
     "AicpuAllToAllSoleMeshSingleChannel"},
    // AICPU + CUSTOM + 两层 + 卡数[2,MAX] → AicpuAllToAllSoleMesh
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 1, "AicpuAllToAllSoleMesh"},
    // AICPU + CUSTOM + 两层 + 卡数[2,4] → AicpuAllToAllSoleMeshConcurrent
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, 4, 3, "AicpuAllToAllSoleMeshConcurrent"},
    // CCU + CUSTOM + 单层 + 卡数[2,MAX] → CcuSchedAllToAllSoleMeshConcurrent
    {ENGINE_CCU, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 3, "CcuSchedAllToAllSoleMeshConcurrent"},
    // CCU + CUSTOM + 两层 + 卡数[2,8] → CcuSchedAllToAllSoleMesh
    {ENGINE_CCU, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, 8, 3, "CcuSchedAllToAllSoleMesh"},
};

const Mc2Hcom::CommAlgoEntry *GetAllToAllCommAlgoTable(uint32_t &count)
{
    count = sizeof(ALLTOALL_COMM_ALGO_TABLE) / sizeof(ALLTOALL_COMM_ALGO_TABLE[0]);
    return ALLTOALL_COMM_ALGO_TABLE;
}

} // namespace Mc2Tiling
