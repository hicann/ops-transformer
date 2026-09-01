/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file gen_position_ids_from_mask.cpp
 */
#include "kernel_gen_position_ids_from_mask.h"

extern "C" __global__ __aicore__ void gen_position_ids_from_mask(GM_ADDR attention_mask, GM_ADDR position_ids,
                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // tiling key: 1=int32, 2=int64, 3=bool(int8). 输入 dtype 分三条入口, 内部统一转 int32.
    if (TILING_KEY_IS(1)) {
        KernelGenPositionIdsFromMask<int32_t> op;
        op.Init(attention_mask, position_ids, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGenPositionIdsFromMask<int64_t> op;
        op.Init(attention_mask, position_ids, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelGenPositionIdsFromMask<int8_t> op; // bool 以 int8 搬运
        op.Init(attention_mask, position_ids, &tilingData);
        op.Process();
    }
}
