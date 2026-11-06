/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_COMM_ALGO_TABLE_H
#define ALLTO_ALL_COMM_ALGO_TABLE_H

#include <cstdint>
#include <string>
#include "mc2_comm_algo_selector.h"
#include "platform/platform_ascendc.h"

namespace Mc2Tiling {

inline constexpr const char* ALLTOALL_DEFAULT_ALGO_NAME = "AlltoAll=level0:fullmesh;level1:pairwise";

const Mc2Hcom::CommAlgoEntry* GetAllToAllCommAlgoTable(uint32_t& count);

// 计算alltoall通信数据量（带回绕钳位），查表选择通信算法并返回算法名；
// 960dt（DAV_9201）+CCU时内嵌组网白名单校验与算法名强制（CheckCommTopoAndForceAlgoNameA6，仅单层纯
// fullmesh 1D且rankSize不超过16放行并强制sole[mesh]），拦截时返回空串（根因已在白名单内正式上报），
// 由调用方按tiling失败处理；非960dt+CCU路径返回查表结果，行为与主线一致
std::string SelectAllToAllAlgoName(const std::string& opName, const std::string& group, uint8_t commEngine,
                                   uint32_t tileM, uint64_t dimSize, uint64_t dtypeSize, uint32_t rankDim,
                                   NpuArch arch);

// 960dt（DAV_9201）+CCU：组网白名单校验与算法名强制（由SelectAllToAllAlgoName内部在查表后调用，
// 消费其传入的组网事实——与算法表SelectAlgoName同一事实源、单次查询不重复）做按位与匹配，
// 仅单层纯fullmesh 1D、rankSize不超过16时放行，并强制algoName为sole[mesh]；
// 返回false=拦截（白名单不匹配/组网查询失败），禁止静默降级；true=放行（含非CCU/非960dt零干预）
bool CheckCommTopoAndForceAlgoNameA6(const char* opName, NpuArch arch, const std::string& group, bool topoValid,
                                     const Mc2Hcom::Mc2TopoInfo& topoInfo, uint64_t topoFlags, uint8_t commMode,
                                     std::string& algoName);

} // namespace Mc2Tiling

#endif // ALLTO_ALL_COMM_ALGO_TABLE_H
