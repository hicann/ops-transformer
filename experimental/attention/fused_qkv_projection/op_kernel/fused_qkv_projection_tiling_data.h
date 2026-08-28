/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_QKV_PROJECTION_TILING_DATA_H
#define FUSED_QKV_PROJECTION_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct FusedQkvProjectionTilingData {
    TCubeTiling cubeTiling;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t singleCoreM;
    int32_t singleCoreN;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t qDim;
    int32_t kDim;
    int32_t vDim;
    int64_t sysWsOffset;
    int64_t fusedOutOffset;
    bool hasBias;
    uint32_t blockDim;
    int32_t dtype; // ge::DT_FLOAT(0) or ge::DT_FLOAT16(1)
};

#endif
