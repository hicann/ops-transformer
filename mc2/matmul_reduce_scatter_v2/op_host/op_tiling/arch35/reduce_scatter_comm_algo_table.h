/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_COMM_ALGO_TABLE_H
#define REDUCE_SCATTER_COMM_ALGO_TABLE_H

#include <cstdint>
#include <string>
#include "mc2_comm_algo_selector.h"
#include "op_host/util/op_const_def.h"
namespace Mc2Tiling {

const Mc2Hcom::CommAlgoEntry* GetReduceScatterCommAlgoTable(uint32_t& count);

// 960dt（DAV_9201）+CCU组网白名单校验：仅单层纯fullmesh 1D且rankSize不超过16放行
// 返回false=拦截（白名单不匹配/组网查询失败），true=放行（含非CCU/非960dt零干预）
bool CheckA6CcuCommTopo(const char* opName, const std::string& group, NpuArch arch, uint8_t commEngine);

// 计算ReduceScatter通信数据量orgM×orgN×dtypeSize(带回绕钳位)，查表选择通信算法并返回算法名
// 未命中表项时回退defaultAlgoName
// 960dt（DAV_9201）+CCU：内嵌组网白名单校验并强制算法名为sole[mesh]，仅单层纯fullmesh 1D且
// rankSize不超过16放行；拦截时返回空串，由调用方按tiling失败处理
std::string SelectReduceScatterAlgoName(const std::string& opName, const std::string& group, uint8_t commEngine,
                                        uint64_t orgM, uint64_t orgN, uint64_t dtypeSize, uint32_t rankDim,
                                        const std::string& defaultAlgoName, NpuArch npuArch);

} // namespace Mc2Tiling

#endif // REDUCE_SCATTER_COMM_ALGO_TABLE_H
