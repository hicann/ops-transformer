/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_comm_algo_table.h"
#include "mc2_comm_algo_selector.h"
#include "mc2_log_compat.h"
#include "mc2_tiling_utils.h"

namespace Mc2Tiling {

using Mc2Hcom::TOPO_FLAG_NONE;
using Mc2Hcom::TOPO_FLAG_SINGLE_LEVEL;
using Mc2Hcom::TOPO_FLAG_L0_1DMESH;
using Mc2Hcom::TOPO_FLAG_L0_1DMESH_COVER;
using Mc2Hcom::TOPO_FLAG_L0_MESH_1D_CLOS;
// 通信数据量
constexpr uint32_t MIN_DATA_BYTES_ZERO = 0ULL;
constexpr uint32_t DATA_BYTES_1M = 1ULL * 1024ULL * 1024ULL;
constexpr uint32_t DATA_BYTES_4M = 4ULL * 1024ULL * 1024ULL;
constexpr uint32_t DATA_BYTES_8M = 8ULL * 1024ULL * 1024ULL;
constexpr uint32_t DATA_BYTES_16M = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t DATA_BYTES_256M = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAX_DATA_BYTES_UNLIMITED = ~0ULL;
// 卡数
constexpr uint32_t MIN_RANK_SIZE = 0;
constexpr uint32_t RANK_SIZE_2 = 2;
constexpr uint32_t RANK_SIZE_4 = 4;
constexpr uint32_t RANK_SIZE_8 = 8;
constexpr uint32_t RANK_SIZE_16 = 16;
constexpr uint32_t MAX_RANK_SIZE = ~0U;
// 960dt CCU组网白名单：单层纯fullmesh 1D标志组合（按位与全命中才放行）
constexpr uint64_t CCU_WHITELIST_FLAGS = TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_1DMESH | TOPO_FLAG_L0_1DMESH_COVER;
// 960dt CCU mesh1D组网rank上限（uint16事件掩码口径）
constexpr uint32_t CCU_MESH1D_MAX_RANK_SIZE = 16U;
constexpr char A6_CCU_ALGORITHM[] = "sole[mesh]";

enum CommAlgoPriority : int32_t {
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
};

// ReduceScatter 通信算法表；选择器为硬过滤（任一维度清空候选集即无匹配，回退调用方传入的默认算法）。
// 通信算法统一由本表管理(组网/卡数/数据量/优先级)，调用方直接使用选出的算法名：
// -
// Mesh+CLOS混联形态具体表项(mesh/nhr类规格串)通过TOPO_FLAG_L0_MESH_1D_CLOS|TOPO_FLAG_SINGLE_LEVEL(单层Mesh+CLOS混联)约束，
//   置于同优先级非混联形态表项之前(同级优先级按表序取先)，Mesh+CLOS混联形态优先命中；
// - 16p非Mesh+CLOS混联形态(跨机)为通信api benchmark优选算法，不限组网；
// - CCU的4p/8p走All2All+VecReduce通路(AllToAll通信)，兜底为AlltoAll算法；
//   其余卡数(AICPU全部/CCU 2p与16p及以上)走ReduceScatter通路，兜底为ReduceScatter算法
static constexpr Mc2Hcom::CommAlgoEntry REDUCE_SCATTER_COMM_ALGO_TABLE[] = {
    // AICPU通信算法(ReduceScatter通路)
    // 兜底算法:ReduceScatter=level0:fullmesh
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, MIN_RANK_SIZE,
     MAX_RANK_SIZE, LOW, "ReduceScatter=level0:fullmesh"},
    // 单层Mesh+CLOS混联 + 4p/1M数据量 → sole[mesh]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO,
     DATA_BYTES_1M, MIN_RANK_SIZE, RANK_SIZE_4, MEDIUM, "sole[mesh]"},
    // 单层Mesh+CLOS混联 + 4p/1M-1024M数据量 → concur[mesh, nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_1M,
     MAX_DATA_BYTES_UNLIMITED, MIN_RANK_SIZE, RANK_SIZE_4, MEDIUM, "concur[mesh,nhr]"},
    // 单层Mesh+CLOS混联 + 8p/4M数据量 → sole[nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO,
     DATA_BYTES_4M, RANK_SIZE_4, RANK_SIZE_8, MEDIUM, "sole[nhr]"},
    // 单层Mesh+CLOS混联 + 8p/4M-1024M数据量 → parallel[mesh, nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_4M,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_4, RANK_SIZE_8, MEDIUM, "parallel[mesh,nhr]"},
    // 单层Mesh+CLOS混联 + 16p/8M数据量 → sole[nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO,
     DATA_BYTES_8M, RANK_SIZE_8, MAX_RANK_SIZE, MEDIUM, "sole[nhr]"},
    // 单层Mesh+CLOS混联 + 16p/8M-1024M数据量 → parallel[mesh, nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_8M,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_8, MAX_RANK_SIZE, MEDIUM, "parallel[mesh,nhr]"},
    // 16p非Mesh+CLOS混联形态(跨机)：通信api benchmark优选(置于混联形态表项之后，该形态优先命中上方具体表项)
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_16,
     RANK_SIZE_16, MEDIUM, "InsReduceScatterParallelMesh1DNHR"},

    // CCU通信算法
    // 2p兜底算法:ReduceScatter=level0:fullmesh
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, MIN_RANK_SIZE,
     RANK_SIZE_2, LOW, "ReduceScatter=level0:fullmesh"},
    // 4p/8p走All2All+VecReduce通路，兜底算法:AlltoAll=level0:fullmesh
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_4, RANK_SIZE_4,
     LOW, "AlltoAll=level0:fullmesh"},
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_8, RANK_SIZE_8,
     LOW, "AlltoAll=level0:fullmesh"},
    // 16p及以上ReduceScatter通路兜底算法:ReduceScatter=level0:fullmesh
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_16,
     MAX_RANK_SIZE, LOW, "ReduceScatter=level0:fullmesh"},
    // 单层Mesh+CLOS混联 + 4p/4M数据量 → sole[mesh.muti_channel]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO, DATA_BYTES_4M,
     RANK_SIZE_2, RANK_SIZE_4, MEDIUM, "sole[mesh.multi_channel]"},
    // 单层Mesh+CLOS混联 + 4p/4M-1024M数据量 → concur[mesh, mesh]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, DATA_BYTES_4M,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_2, RANK_SIZE_4, MEDIUM, "concur[mesh,mesh]"},
    // 单层Mesh+CLOS混联 + 8p/16p → sole[mesh.muti_channel]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, MIN_DATA_BYTES_ZERO,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_4, MAX_RANK_SIZE, MEDIUM, "sole[mesh.multi_channel]"},
};

const Mc2Hcom::CommAlgoEntry* GetReduceScatterCommAlgoTable(uint32_t& count)
{
    count = sizeof(REDUCE_SCATTER_COMM_ALGO_TABLE) / sizeof(REDUCE_SCATTER_COMM_ALGO_TABLE[0]);
    return REDUCE_SCATTER_COMM_ALGO_TABLE;
}

bool CheckA6CcuCommTopo(const char* opName, const std::string& group, NpuArch arch, uint8_t commEngine)
{
    // 机型直接判断（入参arch）：仅960dt（DAV_9201）走组网白名单，其余机型（950等）不干预
    if (arch != Ops::Base::DAV_9201) {
        return true;
    }
    // 引擎过滤前置：非CCU引擎（ai_cpu/aiv及缺省）全路径零干预
    if (commEngine != mc2tiling::A5_CCU_ENGINE) {
        return true;
    }
    Mc2Hcom::Mc2TopoInfo topoInfo;
    uint64_t topoFlags = Mc2Hcom::TOPO_FLAG_NONE;
    const bool topoValid = Mc2Hcom::Mc2CommAlgoSelector::GetTopoFlags(group.c_str(), topoInfo, topoFlags);
    // 组网查询失败：fail-closed拦截（根因为组网查询内部错误）
    if (!topoValid) {
        OP_LOGE(opName,
                "Failed to get group topology info for group=%s, CCU engine is not allowed on ascend960dt. "
                "Please use ai_cpu engine or check the group topology.",
                group.c_str());
        return false;
    }
    // 组网白名单判定：按位与匹配——纯fullmesh 1D为单层+1DMESH实例存在+1DMESH覆盖（混联/纯CLOS天然缺位不命中）
    if ((topoFlags & CCU_WHITELIST_FLAGS) != CCU_WHITELIST_FLAGS || topoInfo.userRankSize > CCU_MESH1D_MAX_RANK_SIZE) {
        std::string reason = "CCU engine on ascend960dt is not allowed (requires single-level pure fullmesh 1D "
                             "topology within rankSize <= 16), userRankSize=" +
                             std::to_string(topoInfo.userRankSize) +
                             ". Please use ai_cpu engine or check the group topology.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "comm_mode", "ccu", reason.c_str());
        return false;
    }
    return true;
}

std::string SelectReduceScatterAlgoName(const std::string& opName, const std::string& group, uint8_t commEngine,
                                        uint64_t orgM, uint64_t orgN, uint64_t dtypeSize, uint32_t rankDim,
                                        const std::string& defaultAlgoName, NpuArch npuArch)
{
    // A6(DAV_9201)的CCU通信不区分ReduceScatter/AlltoAll，统一使用sole[mesh]；组网仅允许单层纯fullmesh 1D。
    if (npuArch == Ops::Base::DAV_9201 && commEngine == mc2tiling::A5_CCU_ENGINE) {
        if (!CheckA6CcuCommTopo(opName.c_str(), group, npuArch, commEngine)) {
            return "";
        }
        OP_LOGI(opName.c_str(), "[SetMc2Hcomm] A6 CCU force algorithm to %s", A6_CCU_ALGORITHM);
        return A6_CCU_ALGORITHM;
    }
    uint64_t commDataBytes = Mc2Hcom::CalcCommDataBytes(opName.c_str(), orgM, orgN, dtypeSize);
    OP_LOGI(opName.c_str(),
            "[SetMc2Hcomm] commDataBytes=%llu, orgM=%llu, orgN=%llu, dtypeSize=%llu, rankDim=%u, commEngine=%u",
            commDataBytes, orgM, orgN, dtypeSize, rankDim, static_cast<uint32_t>(commEngine));
    uint32_t algoCount = 0;
    const Mc2Hcom::CommAlgoEntry* algoEntries = GetReduceScatterCommAlgoTable(algoCount);
    // 组网事实单次查询后传入(SelectAlgoName纯匹配不自查)
    Mc2Hcom::Mc2TopoInfo topoInfo;
    uint64_t topoFlags = Mc2Hcom::TOPO_FLAG_NONE;
    Mc2Hcom::Mc2CommAlgoSelector::GetTopoFlags(group.c_str(), topoInfo, topoFlags);
    return Mc2Hcom::Mc2CommAlgoSelector::SelectAlgoName(opName, group.c_str(), commEngine, commDataBytes, rankDim,
                                                        algoEntries, algoCount, defaultAlgoName, topoFlags);
}

} // namespace Mc2Tiling
