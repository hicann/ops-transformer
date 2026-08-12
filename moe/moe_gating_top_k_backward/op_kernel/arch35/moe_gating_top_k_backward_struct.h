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
 * \file moe_gating_top_k_backward_struct.h
 * \brief Tiling data struct shared between host tiling and kernel
 */
#ifndef MOE_GATING_TOP_K_BACKWARD_STRUCT_H
#define MOE_GATING_TOP_K_BACKWARD_STRUCT_H

#include <cstdint>

struct MoeGatingTopKBackwardA5TilingData {
    int64_t needCoreNum;
    int64_t perCoreRows;
    int64_t baseRows;
    int64_t perLoopTimes;
    int64_t perTailRows;
    int64_t lastLoopTimes;
    int64_t lastTailRows;
    int64_t expertCount;
    int64_t k;
    int64_t gradYDtypeSize;
    int64_t renorm;
    int64_t normType;
    float routedScalingFactor;
    float eps;
};

#endif // MOE_GATING_TOP_K_BACKWARD_STRUCT_H
