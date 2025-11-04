/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_buffer_reset_tiling.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_BUFFER_RESET_TILING_H
#define MOE_DISTRIBUTE_BUFFER_RESET_TILING_H
#include "kernel_tiling/kernel_tiling.h"

struct MoeDistributeBufferResetInfo {
    uint32_t worldSize;
    uint32_t needSync;
    uint32_t aivNum;                     // aivNum
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

struct MoeDistributeBufferResetTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeDistributeBufferResetInfo moeDistributeBufferReset;
};

#endif // MOE_DISTRIBUTE_BUFFER_RESET_TILING_H