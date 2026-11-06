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
#include "mc2_comm_algo_selector.h"
#include "mc2_log_compat.h"
#include "mc2_tiling_utils.h"

namespace Mc2Tiling {

using Mc2Hcom::TOPO_FLAG_NONE;
using Mc2Hcom::TOPO_FLAG_SINGLE_LEVEL;
using Mc2Hcom::TOPO_FLAG_L0_MESH_1D_CLOS;

constexpr uint32_t MIN_DATA_BYTES_ZERO = 0;
// 数据量分档边界（闭区间，边界值同时命中相邻两档，按表项顺序先到先得归低一级档位）
constexpr uint64_t DATA_BYTES_2M = 2ULL * 1024ULL * 1024ULL;
constexpr uint64_t DATA_BYTES_4M = 4ULL * 1024ULL * 1024ULL;
constexpr uint64_t DATA_BYTES_8M = 8ULL * 1024ULL * 1024ULL;
constexpr uint64_t DATA_BYTES_16M = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t DATA_BYTES_32M = 32ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAX_DATA_BYTES_UNLIMITED = ~0ULL;
constexpr uint32_t RANK_SIZE_2 = 2;
constexpr uint32_t RANK_SIZE_4 = 4;
constexpr uint32_t RANK_SIZE_8 = 8;
constexpr uint32_t RANK_SIZE_16 = 16;
constexpr uint32_t MAX_RANK_SIZE = ~0U;

enum CommAlgoPriority : int32_t {
    DEFAULT = 0,
    MEDIUM = 2,
};

// Mesh+CLOS混联形态(MESH_1D_CLOS)的AllGather通信算法表（算法名为 HCCL 新版算法规格串）
// 具体算法表项仅Mesh+CLOS混联形态(MESH_1D_CLOS)支持，通过TOPO_FLAG_L0_MESH_1D_CLOS|TOPO_FLAG_SINGLE_LEVEL(单层Mesh+CLOS混联)约束；
// 未命中场景（其他卡数、多层组网、非Mesh+CLOS混联形态等）由末两条 DEFAULT 兜底表项回退默认算法
static constexpr Mc2Hcom::CommAlgoEntry ALLGATHER_COMM_ALGO_TABLE[] = {
    // AICPU通信算法
    // rank4 小数据量 → sole[mesh]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO,
     DATA_BYTES_2M, RANK_SIZE_4, RANK_SIZE_4, MEDIUM, "sole[mesh]"},
    // rank4 大数据量 → concur[mesh,nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_2M,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_4, RANK_SIZE_4, MEDIUM, "concur[mesh,nhr]"},
    // rank8 中数据量 → parallel[mesh,nhr.multi_channel]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_8M, DATA_BYTES_16M,
     RANK_SIZE_8, RANK_SIZE_8, MEDIUM, "parallel[mesh,nhr.multi_channel]"},
    // rank16 中大数据量 → parallel[mesh,nhr.multi_channel]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_16M, DATA_BYTES_32M,
     RANK_SIZE_16, RANK_SIZE_16, MEDIUM, "parallel[mesh,nhr.multi_channel]"},
    // rank8~16 其余数据量 → sole[nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_8, RANK_SIZE_16, MEDIUM, "sole[nhr]"},
    // CCU_SCHED通信算法
    // rank4 小数据量 → sole[mesh]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO, DATA_BYTES_8M,
     RANK_SIZE_4, RANK_SIZE_4, MEDIUM, "sole[mesh]"},
    // rank4 大数据量 → concur[mesh,nhr.multi_channel]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_8M,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_4, RANK_SIZE_4, MEDIUM, "concur[mesh,nhr.multi_channel]"},
    // rank8 小数据量 → sole[mesh]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO, DATA_BYTES_4M,
     RANK_SIZE_8, RANK_SIZE_8, MEDIUM, "sole[mesh]"},
    // rank8 大数据量 → parallel[mesh,nhr.multi_channel]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_4M,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_8, RANK_SIZE_8, MEDIUM, "parallel[mesh,nhr.multi_channel]"},
    // rank16 小数据量 → sole[mesh]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO, DATA_BYTES_8M,
     RANK_SIZE_16, RANK_SIZE_16, MEDIUM, "sole[mesh]"},
    // rank16 大数据量 → parallel[mesh,nhr.multi_channel]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_8M,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_16, RANK_SIZE_16, MEDIUM, "parallel[mesh,nhr.multi_channel]"},
    // 未命中场景（其他卡数如 rank 2/3/5/6/7/17+、多层组网、非Mesh+CLOS混联形态等）回退默认算法
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_2,
     MAX_RANK_SIZE, DEFAULT, ALLGATHER_DEFAULT_ALGO_NAME},
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_2,
     MAX_RANK_SIZE, DEFAULT, ALLGATHER_DEFAULT_ALGO_NAME},
};

const Mc2Hcom::CommAlgoEntry* GetAllGatherCommAlgoTable(uint32_t& count)
{
    count = sizeof(ALLGATHER_COMM_ALGO_TABLE) / sizeof(ALLGATHER_COMM_ALGO_TABLE[0]);
    return ALLGATHER_COMM_ALGO_TABLE;
}

std::string SelectAllGatherAlgoName(const std::string& opName, const std::string& group, uint8_t commEngine,
                                    uint64_t tileM, uint64_t kValue, uint64_t dtypeSize, uint32_t rankDim)
{
    uint64_t commDataBytes = Mc2Hcom::CalcCommDataBytes(opName.c_str(), tileM, kValue, dtypeSize);
    OP_LOGI(opName.c_str(),
            "[SetMc2Hcomm] commDataBytes=%llu, tileM=%llu, kValue=%llu, dtypeSize=%llu, rankDim=%u, commEngine=%u",
            commDataBytes, tileM, kValue, dtypeSize, rankDim, static_cast<uint32_t>(commEngine));
    uint32_t algoCount = 0;
    const Mc2Hcom::CommAlgoEntry* algoEntries = GetAllGatherCommAlgoTable(algoCount);
    // 组网事实单次查询后传入(SelectAlgoName纯匹配不自查)
    Mc2Hcom::Mc2TopoInfo topoInfo;
    uint64_t topoFlags = Mc2Hcom::TOPO_FLAG_NONE;
    Mc2Hcom::Mc2CommAlgoSelector::GetTopoFlags(group.c_str(), topoInfo, topoFlags);
    return Mc2Hcom::Mc2CommAlgoSelector::SelectAlgoName(opName, group.c_str(), commEngine, commDataBytes, rankDim,
                                                        algoEntries, algoCount, ALLGATHER_DEFAULT_ALGO_NAME, topoFlags);
}
} // namespace Mc2Tiling
