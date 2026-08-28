/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_qkv_projection.h"
#include "fused_qkv_projection_tiling_key.h"

template <uint64_t schMode, uint64_t dtype>
__global__ __aicore__ void fused_qkv_projection(GM_ADDR hiddenStates, GM_ADDR weight, GM_ADDR bias, GM_ADDR query,
                                                GM_ADDR key, GM_ADDR value, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(FusedQkvProjectionTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedQkvProjectionTilingData, tilingData, tiling);

    if constexpr (dtype == TPL_DTYPE_FLOAT16) {
        FusedQkvProjection<half> op;
        op.Init(hiddenStates, weight, bias, query, key, value, workspace, &tilingData);
        op.Process();
    } else {
        FusedQkvProjection<float> op;
        op.Init(hiddenStates, weight, bias, query, key, value, workspace, &tilingData);
        op.Process();
    }
}
