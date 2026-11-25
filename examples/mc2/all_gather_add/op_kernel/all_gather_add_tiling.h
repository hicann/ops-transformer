/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_add_tiling.h
 * \brief
 */

#ifndef ALL_GATHER_ADD_TILING_H
#define ALL_GATHER_ADD_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct AllGatherAddTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    uint32_t commTurn; // 通信轮次

    uint32_t totalElemNum;
    uint32_t blockElemNum;
    uint32_t tileNum;
    uint32_t addTileElemNum;
    uint32_t gatherTileElemNum;
    uint32_t addCoresPerRank;
    uint32_t rankSize; // 通信域总 rank 数（all_gather 的 rankDim）
};

#endif // ALL_GATHER_ADD_TILING_H
