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
 * \file moe_fused_topk.cpp
 * \brief
 */

#include "moe_fused_topk_kernel.h"

using namespace AscendC;

#define INVOKE_MOE_FUSED_ADD_TOPK_IMPL(templateClass, ...)                                                      \
    do {                                                                                                        \
        templateClass<__VA_ARGS__> op;                                                                          \
        op.InitTilingData(&tilingData, x, addNum, mappingNum, mappingTable, y, indices, userWsp);               \
        op.InitBuffer(&pipe);                                                                                   \
        op.Process();                                                                                           \
    } while (0)

extern "C" __global__ __aicore__ void moe_fused_topk(GM_ADDR x, GM_ADDR addNum,
                                                         GM_ADDR mappingNum, GM_ADDR mappingTable, GM_ADDR y,
                                                         GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    PRELOAD(8);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWsp = GetUserWorkspace(workspace);
    if (userWsp == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

#if (ORIG_DTYPE_X == DT_FLOAT)
    if (TILING_KEY_IS(0)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedTopk, float, float, 0);
        return;
    } else if (TILING_KEY_IS(1)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedTopk, float, float, 1);
        return;
    }
#endif
#if (ORIG_DTYPE_X == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedTopk, half, float, 0);
        return;
    } else if (TILING_KEY_IS(1)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedTopk, half, float, 1);
        return;
    }
#endif
#if (ORIG_DTYPE_X == DT_BF16)
    if (TILING_KEY_IS(0)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedTopk, bfloat16_t, float, 0);
        return;
    } else if (TILING_KEY_IS(1)) {
        INVOKE_MOE_FUSED_ADD_TOPK_IMPL(MoeFusedTopk, bfloat16_t, float, 1);
        return;
    }
#endif
}