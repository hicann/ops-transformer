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
 * \file moe_distribute_combine_setup_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef MOE_DISTRIBUTE_COMBINE_SETUP_TILING_DATA_H
#define MOE_DISTRIBUTE_COMBINE_SETUP_TILING_DATA_H

#include <cstdint>
#include <kernel_tiling/kernel_tiling.h>

// 910C 950
struct MoeDistributeCombineSetupInfo {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t commQuantMode;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t a;
    uint32_t aivNum;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

struct MoeDistributeCombineSetupTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeCombineSetupInfo moeDistributeCombineSetupInfo;
};

// 910C
struct BatchWriteItem {
    uint64_t type;
    uint32_t res1[5];
    uint32_t length;
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;
    uint32_t res2[4];
};

#endif // MOE_DISTRIBUTE_COMBINE_SETUP_TILING_DATA_H
