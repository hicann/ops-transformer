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
#include "common/utils/mc2_comm_algo_selector.h"
namespace Mc2Tiling {
inline constexpr const char *REDUCE_SCATTER_DEFAULT_ALGO_NAME = "ReduceScatter=level0:fullmesh";

const Mc2Hcom::CommAlgoEntry *GetReduceScatterCommAlgoTable(uint32_t &count);

} // namespace Mc2Tiling

#endif // REDUCE_SCATTER_COMM_ALGO_TABLE_H
