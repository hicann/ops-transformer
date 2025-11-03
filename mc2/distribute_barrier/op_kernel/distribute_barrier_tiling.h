/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file distribute_barrier_tiling.h
 * \brief
 */
#ifndef DISTRIBUTE_BARRIER_TILING_H
#define DISTRIBUTE_BARRIER_TILING_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct DistributeBarrierInfo {
    uint32_t worldSize;
    uint32_t rankId;
    uint32_t aivNum;                     // aivNum
    uint64_t totalUbSize;
    uint64_t totalWinSize;
    bool isInputTimeOut;
    bool isInputElasticInfo;
};

struct DistributeBarrierTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    DistributeBarrierInfo distributeBarrierInfo;
};

#endif // DISTRIBUTE_BARRIER_TILING_H