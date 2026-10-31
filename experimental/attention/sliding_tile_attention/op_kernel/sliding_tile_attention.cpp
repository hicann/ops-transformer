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
 * \file sliding_tile_attention.cpp
 * \brief Kernel entry of SlidingTileAttention (cube + vector MIX, Atlas A2).
 */

#include "kernel_operator.h"
#include "sta_bsa_kernel.h"

extern "C" __global__ __aicore__ void sliding_tile_attention(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR windows,
                                                             GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    StaBsa::Infer<half>(query, key, value, windows, output, user, tiling);
#endif
#if (ORIG_DTYPE_QUERY == DT_BF16)
    StaBsa::Infer<bfloat16_t>(query, key, value, windows, output, user, tiling);
#endif
}
