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
 * \file all_gather_comm_algo_table.h
 * \brief all_gather_matmul / all_gather_matmul_v2 共享的 AllGather 通信算法表。
 */

#ifndef ALL_GATHER_COMM_ALGO_TABLE_H
#define ALL_GATHER_COMM_ALGO_TABLE_H

#include <cstdint>
#include <string>
#include "mc2_comm_algo_selector.h"

namespace Mc2Tiling {

inline constexpr const char* ALLGATHER_DEFAULT_ALGO_NAME = "AllGather=level0:fullmesh";
// A6(ascend960dt) CCU 引擎 HCCL 仅支持的 AllGather algConfig（实测约束；整串替换，不带 AllGather=level0: 前缀）
inline constexpr const char* ALLGATHER_A6_CCU_ALGO_NAME = "sole[mesh]";

const Mc2Hcom::CommAlgoEntry* GetAllGatherCommAlgoTable(uint32_t& count);

// 按架构返回算法表：npuArch 取 Ops::Base::NpuArch 的整数值（如 static_cast<uint32_t>(Ops::Base::DAV_9201)）；
// A6(ascend960dt)返回专属算法表，其余架构返回A5标定表
const Mc2Hcom::CommAlgoEntry* GetAllGatherCommAlgoTable(uint32_t& count, uint32_t npuArch);

// 计算单轮AllGather通信数据量tileM×K×dtypeSize(带回绕钳位)，查表选择通信算法并返回算法名；
// 未命中表项时回退默认算法ALLGATHER_DEFAULT_ALGO_NAME
std::string SelectAllGatherAlgoName(const std::string& opName, const std::string& group, uint8_t commEngine,
                                    uint64_t tileM, uint64_t kValue, uint64_t dtypeSize, uint32_t rankDim);

// 架构感知的算法选择入口：npuArch 取 Ops::Base::NpuArch 的整数值（如 static_cast<uint32_t>(Ops::Base::DAV_9201)）。
// 与 SelectAllGatherAlgoName 走完全相同的选择流程（数据量计算 → 查表 → Mc2CommAlgoSelector），
// 仅按架构选取算法表，不修改原先选择逻辑：A6(ascend960dt)取专属表（CCU 全场景 sole[mesh] 实测约束、
// AICPU 默认算法），不复用A5的Mesh+CLOS标定表项；其余架构取A5表，行为与原先选择逻辑完全一致。
std::string SelectAllGatherAlgoNameByArch(const std::string& opName, const std::string& group, uint8_t commEngine,
                                          uint64_t tileM, uint64_t kValue, uint64_t dtypeSize, uint32_t rankDim,
                                          uint32_t npuArch);

} // namespace Mc2Tiling

#endif // ALL_GATHER_COMM_ALGO_TABLE_H
