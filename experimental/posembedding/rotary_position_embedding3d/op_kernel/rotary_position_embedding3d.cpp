/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rotary_position_embedding3d.h"

template <uint32_t schMode>
__global__ __aicore__ void rotary_position_embedding3d(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RotaryPositionEmbedding3dTilingData);
    GET_TILING_DATA_WITH_STRUCT(RotaryPositionEmbedding3dTilingData, td, tiling);
    if constexpr (schMode == 0) {
        NsRotaryPositionEmbedding3d::RotaryPositionEmbedding3d<float> op;
        op.Init(x, y, z, workspace, &td);
        op.Process();
    } else if constexpr (schMode == 1) {
        NsRotaryPositionEmbedding3d::RotaryPositionEmbedding3d<half> op;
        op.Init(x, y, z, workspace, &td);
        op.Process();
    } else if constexpr (schMode == 2) {
        NsRotaryPositionEmbedding3d::RotaryPositionEmbedding3d<bfloat16_t> op;
        op.Init(x, y, z, workspace, &td);
        op.Process();
    }
}
