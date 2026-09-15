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
#include "common/utils/mc2_comm_algo_selector.h"

namespace Mc2Tiling {

inline constexpr const char *ALLTOALL_DEFAULT_ALGO_NAME = "AlltoAll=level0:fullmesh;level1:pairwise";

const Mc2Hcom::CommAlgoEntry *GetAllToAllCommAlgoTable(uint32_t &count);

// 计算alltoall通信数据量（带回绕钳位），查表选择通信算法并返回算法名
std::string SelectAllToAllAlgoName(const std::string &opName, const std::string &group, uint8_t commEngine,
                                   uint32_t tileM, uint64_t dimSize, uint64_t dtypeSize, uint32_t rankDim);

} // namespace Mc2Tiling

#endif // ALLTO_ALL_COMM_ALGO_TABLE_H
