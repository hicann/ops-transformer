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
#include "mc2_log_compat.h"

namespace Mc2Tiling {

constexpr uint8_t ENGINE_AICPU = 2;
constexpr uint8_t ENGINE_CCU = 6;
constexpr uint64_t MAX_DATA_BYTES = ~0ULL;
constexpr uint32_t MAX_RANK_SIZE = ~0U;
constexpr uint64_t LARGE_DATA_BYTES = 256ULL * 1024ULL * 1024ULL;

static constexpr Mc2Hcom::CommAlgoEntry ALLTOALL_COMM_ALGO_TABLE[] = {
    // 高阶api具体算法未明确固定，当前暂时全走默认算法
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 3, ALLTOALL_DEFAULT_ALGO_NAME},
    // 高阶api具体算法未明确固定，当前暂时全走默认算法
    {ENGINE_CCU, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 3, ALLTOALL_DEFAULT_ALGO_NAME},
    // 暂不生效：被上方默认算法表项(priority=3)压制，当前对所有卡数均不生效。
    // 预留说明：若未来移除默认表项，本表项(AICPU + CUSTOM + 单层 + 卡数[8,16])
    // 中 rank=8 会被下方Concurrent(priority=2)以高优先级覆盖，假想生效范围为[9,16]
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 8, 16, 1, "AicpuAllToAllSoleMeshUBX"},
    // 暂不生效：被上方默认算法表项(priority=3)压制。AICPU + CUSTOM + 单层 + 卡数[2,8] → AicpuAllToAllSoleMeshConcurrent
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, 8, 2, "AicpuAllToAllSoleMeshConcurrent"},
    // 暂不生效：被上方默认算法表项(priority=3)压制。AICPU + CUSTOM + 两层 + 数据量[0,256MB) + 卡数[2,MAX] →
    // AicpuAllToAllSoleMeshSingleChannel
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 2, 2, 0, LARGE_DATA_BYTES - 1, 2, MAX_RANK_SIZE, 1,
     "AicpuAllToAllSoleMeshSingleChannel"},
    // 暂不生效：被上方默认算法表项(priority=3)压制。AICPU + CUSTOM + 两层 + 数据量[256MB,MAX] + 卡数[2,MAX] →
    // AicpuAllToAllSoleMesh
    {ENGINE_AICPU, COMM_TOPO_CUSTOM, 2, 2, LARGE_DATA_BYTES, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 1,
     "AicpuAllToAllSoleMesh"},
    // 暂不生效：被上方默认算法表项(priority=3)压制。CCU + CUSTOM + 单层 + 卡数[2,8] →
    // CcuSchedAllToAllSoleMeshConcurrent
    {ENGINE_CCU, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, 8, 1, "CcuSchedAllToAllSoleMeshConcurrent"},
    // 暂不生效：被上方默认算法表项(priority=3)压制。CCU + CUSTOM + 两层 + 卡数[2,8] → CcuSchedAllToAllSoleMesh
    {ENGINE_CCU, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, 8, 1, "CcuSchedAllToAllSoleMesh"},
};

const Mc2Hcom::CommAlgoEntry *GetAllToAllCommAlgoTable(uint32_t &count)
{
    count = sizeof(ALLTOALL_COMM_ALGO_TABLE) / sizeof(ALLTOALL_COMM_ALGO_TABLE[0]);
    return ALLTOALL_COMM_ALGO_TABLE;
}

std::string SelectAllToAllAlgoName(const std::string &opName, const std::string &group, uint8_t commEngine,
                                   uint32_t tileM, uint64_t dimSize, uint64_t dtypeSize, uint32_t rankDim)
{
    uint64_t commDataBytes = 0;
    if (dimSize != 0 && tileM > UINT64_MAX / dimSize) {
        OP_LOGW(opName.c_str(), "tileM * dimSize overflow, clamp to UINT64_MAX.");
        commDataBytes = UINT64_MAX;
    } else {
        commDataBytes = static_cast<uint64_t>(tileM) * dimSize;
    }
    if (dtypeSize != 0 && commDataBytes > UINT64_MAX / dtypeSize) {
        OP_LOGW(opName.c_str(), "commDataBytes overflow, clamp to UINT64_MAX.");
        commDataBytes = UINT64_MAX; // 钳位命中大数据量分支/默认算法，避免回绕成小值误入小数据算法
    } else {
        commDataBytes *= dtypeSize;
    }
    OP_LOGI(opName.c_str(), "[SetHcclTiling] commDataBytes=%llu, tileM=%u, dimSize=%llu, rankDim=%u, engine=%u",
            commDataBytes, tileM, dimSize, rankDim, static_cast<uint32_t>(commEngine));
    uint32_t algoCount = 0;
    const Mc2Hcom::CommAlgoEntry *algoEntries = GetAllToAllCommAlgoTable(algoCount);
    std::string algoName = Mc2Hcom::Mc2CommAlgoSelector::SelectAlgoName(
        opName, group.c_str(), commEngine, commDataBytes, rankDim, algoEntries, algoCount, ALLTOALL_DEFAULT_ALGO_NAME);
    OP_LOGI(opName.c_str(), "[SetHcclTiling] selected algoName=%s, group=%s", algoName.c_str(), group.c_str());
    return algoName;
}

} // namespace Mc2Tiling
