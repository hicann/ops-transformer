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
 * \file moe_distribute_combine_teardown_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TEARDOWN_TILING_DATA_H
#define MOE_DISTRIBUTE_COMBINE_TEARDOWN_TILING_DATA_H

#include <cstdint>
#include <kernel_tiling/kernel_tiling.h>

struct MoeDistributeCombineTeardownInfo {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t aivNum;
    bool isActiveMask = false;     // input active mask or not
    bool hasSharedExpertX = false; // input shared expert x or not
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

struct MoeDistributeCombineTeardownTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeCombineTeardownInfo moeDistributeCombineTeardownInfo;
};

#endif // MOE_DISTRIBUTE_COMBINE_TEARDOWN_TILING_DATA_H
