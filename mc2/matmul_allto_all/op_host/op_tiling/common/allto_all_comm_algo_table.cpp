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
#include "mc2_log_compat.h"
#include "mc2_tiling_utils.h"

namespace Mc2Tiling {

using Mc2Hcom::TOPO_FLAG_NONE;
using Mc2Hcom::TOPO_FLAG_SINGLE_LEVEL;
using Mc2Hcom::TOPO_FLAG_L0_MESH_1D_CLOS;

constexpr uint64_t MAX_DATA_BYTES = ~0ULL;
constexpr uint32_t MAX_RANK_SIZE = ~0U;

// 960dt CCU组网白名单：单层纯fullmesh 1D标志组合（按位与全命中才放行）
constexpr uint64_t CCU_WHITELIST_FLAGS =
    Mc2Hcom::TOPO_FLAG_SINGLE_LEVEL | Mc2Hcom::TOPO_FLAG_L0_1DMESH | Mc2Hcom::TOPO_FLAG_L0_1DMESH_COVER;
// 960dt CCU mesh1D组网rank上限（uint16事件掩码口径）
constexpr uint32_t CCU_MESH1D_MAX_RANK_SIZE = 16U;
// 960dt CCU强制算法名（不信任算法表结果，兜底legacy名与后续表演进一律覆盖）
constexpr char CCU_MESH1D_ALGO_NAME[] = "sole[mesh]";

// 选择器为硬过滤（任一维度清空候选集即无匹配，回退默认算法）；
// 各引擎配一条全区间低优先级兜底表项，承接非Mesh+CLOS混联形态/多层组网/未覆盖卡数等场景回退默认算法；
// 具体算法表项仅Mesh+CLOS混联形态(MESH_1D_CLOS)支持，通过TOPO_FLAG_L0_MESH_1D_CLOS|TOPO_FLAG_SINGLE_LEVEL(单层Mesh+CLOS混联)约束
static constexpr Mc2Hcom::CommAlgoEntry ALLTOALL_COMM_ALGO_TABLE[] = {
    // AICPU + 单层Mesh+CLOS混联 + 卡数[8,16] → sole[mesh]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, 0, MAX_DATA_BYTES, 8, 16, 1,
     "sole[mesh]"},
    // AICPU + 单层Mesh+CLOS混联 + 卡数[2,4] → concur[mesh,mesh]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, 0, MAX_DATA_BYTES, 2, 4, 1,
     "concur[mesh,mesh]"},
    // CCU + 单层Mesh+CLOS混联 + 卡数[2,4] → concur[mesh,mesh]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, 0, MAX_DATA_BYTES, 2, 4, 1,
     "concur[mesh,mesh]"},
    // CCU + 单层Mesh+CLOS混联 + 卡数[8,16] → sole[mesh.multi_channel]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, 0, MAX_DATA_BYTES, 8, 16, 1,
     "sole[mesh.multi_channel]"},
    // AICPU 兜底：非Mesh+CLOS混联形态/未覆盖场景低优先级回退默认算法
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_NONE, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 0, ALLTOALL_DEFAULT_ALGO_NAME},
    // CCU 兜底：非Mesh+CLOS混联形态/未覆盖场景低优先级回退默认算法
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 0, ALLTOALL_DEFAULT_ALGO_NAME},
};

const Mc2Hcom::CommAlgoEntry* GetAllToAllCommAlgoTable(uint32_t& count)
{
    count = sizeof(ALLTOALL_COMM_ALGO_TABLE) / sizeof(ALLTOALL_COMM_ALGO_TABLE[0]);
    return ALLTOALL_COMM_ALGO_TABLE;
}

std::string SelectAllToAllAlgoName(const std::string& opName, const std::string& group, uint8_t commEngine,
                                   uint32_t tileM, uint64_t dimSize, uint64_t dtypeSize, uint32_t rankDim, NpuArch arch)
{
    uint64_t commDataBytes = Mc2Hcom::CalcCommDataBytes(opName.c_str(), tileM, dimSize, dtypeSize);
    OP_LOGI(opName.c_str(), "[SetHcclTiling] commDataBytes=%llu, tileM=%u, dimSize=%llu, rankDim=%u, engine=%u",
            commDataBytes, tileM, dimSize, rankDim, static_cast<uint32_t>(commEngine));
    uint32_t algoCount = 0;
    const Mc2Hcom::CommAlgoEntry* algoEntries = GetAllToAllCommAlgoTable(algoCount);
    // 组网事实单次查询：先供SelectAlgoName匹配（纯匹配不自查），再传参复用给CheckCommTopoAndForceAlgoNameA6
    Mc2Hcom::Mc2TopoInfo topoInfo;
    uint64_t topoFlags = Mc2Hcom::TOPO_FLAG_NONE;
    const bool topoValid = Mc2Hcom::Mc2CommAlgoSelector::GetTopoFlags(group.c_str(), topoInfo, topoFlags);
    std::string algoName =
        Mc2Hcom::Mc2CommAlgoSelector::SelectAlgoName(opName, group.c_str(), commEngine, commDataBytes, rankDim,
                                                     algoEntries, algoCount, ALLTOALL_DEFAULT_ALGO_NAME, topoFlags);
    // 960dt（DAV_9201）+CCU：组网白名单校验与算法名强制（仅单层纯fullmesh 1D放行，见函数注释），
    // 拦截时返回空串（根因已在白名单内正式上报），由调用方按tiling失败处理
    if (!CheckCommTopoAndForceAlgoNameA6(opName.c_str(), arch, group, topoValid, topoInfo, topoFlags, commEngine,
                                         algoName)) {
        return "";
    }
    OP_LOGI(opName.c_str(), "[SetHcclTiling] selected algoName=%s, group=%s", algoName.c_str(), group.c_str());
    return algoName;
}

bool CheckCommTopoAndForceAlgoNameA6(const char* opName, NpuArch arch, const std::string& group, bool topoValid,
                                     const Mc2Hcom::Mc2TopoInfo& topoInfo, uint64_t topoFlags, uint8_t commMode,
                                     std::string& algoName)
{
    // 机型直接判断（入参arch）：仅960dt（DAV_9201）走组网白名单，其余机型（950等）不干预
    if (arch != Ops::Base::DAV_9201) {
        return true;
    }
    // 引擎过滤前置：非CCU引擎（ai_cpu/aiv及缺省）全路径零干预
    if (commMode != mc2tiling::A5_CCU_ENGINE) {
        return true;
    }
    // 组网查询失败：fail-closed拦截（根因为组网查询内部错误）
    if (!topoValid) {
        OP_LOGE(opName,
                "Failed to get group topology info for group=%s, CCU engine is not allowed on ascend960dt. "
                "Please use ai_cpu engine or check the group topology.",
                group.c_str());
        return false;
    }
    // 组网白名单判定：按位与匹配——纯fullmesh 1D为单层+1DMESH实例存在+1DMESH覆盖（混联/纯CLOS天然缺位不命中）；
    // 降级推导结果由flag拼接统一处理，无需区分主/降级路径
    if ((topoFlags & CCU_WHITELIST_FLAGS) != CCU_WHITELIST_FLAGS || topoInfo.userRankSize > CCU_MESH1D_MAX_RANK_SIZE) {
        std::string reason = "CCU engine on ascend960dt is not allowed (requires single-level pure fullmesh 1D "
                             "topology within rankSize <= 16), userRankSize=" +
                             std::to_string(topoInfo.userRankSize) +
                             ". Please use ai_cpu engine or check the group topology.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "comm_mode", "ccu", reason.c_str());
        return false;
    }
    // 白名单通过：强制算法名为sole[mesh]，不信任算法表结果（兜底legacy名与后续表演进一律覆盖）
    if (algoName != CCU_MESH1D_ALGO_NAME) {
        OP_LOGI(opName, "Force AlltoAll algoName from [%s] to [%s] for CCU engine on ascend960dt.", algoName.c_str(),
                CCU_MESH1D_ALGO_NAME);
    }
    algoName = CCU_MESH1D_ALGO_NAME;
    return true;
}

} // namespace Mc2Tiling
