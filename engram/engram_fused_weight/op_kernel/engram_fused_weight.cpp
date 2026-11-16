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
 * \file engram_fused_weight.cpp
 * \brief EngramFusedWeight kernel 入口
 */

#include "engram_fused_weight_kernel.h"

using namespace AscendC;

// dtype 宏缺省兜底: 正式编译由 binary.json 按 dtype 组合下发, UT 由 CMake 显式传入。
#ifndef DTYPE_WEIGHT_HIDDEN
#define DTYPE_WEIGHT_HIDDEN bfloat16_t
#endif
#ifndef DTYPE_WEIGHT_EMBED
#define DTYPE_WEIGHT_EMBED bfloat16_t
#endif
#ifndef DTYPE_WEIGHT_FUSED
#define DTYPE_WEIGHT_FUSED float
#endif

extern "C" __global__ __aicore__ void engram_fused_weight(GM_ADDR weight_hidden, GM_ADDR weight_embed,
                                                          GM_ADDR weight_fused, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA_WITH_STRUCT(EngramFusedWeightTilingData, tilingDataIn, tiling);
        const EngramFusedWeightTilingData* __restrict tilingData = &tilingDataIn;
        // 空输入 (totalElements == 0): 直接返回, 输出保持为空 tensor。
        if (tilingData->totalElements == 0) {
            return;
        }
        TPipe pipeIn;
        EngramFusedWeight::EngramFusedWeightRegbase<DTYPE_WEIGHT_HIDDEN, DTYPE_WEIGHT_FUSED> op(tilingData, &pipeIn);
        op.Init(weight_hidden, weight_embed, weight_fused);
        op.Process();
    }
}
