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
 * \file fused_gdn_gating.cpp
 * \brief AscendC kernel entry for FusedGdnGating.
 */

#include "kernel_operator.h"
#include "fused_gdn_gating_tiling_data.h"
#include "fused_gdn_gating_tiling_key.h"

#if __CCE_AICORE__ == 200
#include "fused_gdn_gating_310.h"
#else
#include "arch22/fused_gdn_gating_910.h"
#endif

using namespace AscendC;
using namespace FusedGdnGating;

extern "C" __global__ __aicore__ void fused_gdn_gating(GM_ADDR a_log, GM_ADDR a, GM_ADDR b, GM_ADDR dt_bias, GM_ADDR g,
                                                       GM_ADDR beta_output, GM_ADDR workspace, GM_ADDR tiling_gm)
{
    REGISTER_TILING_DEFAULT(FusedGdnGatingTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);

    auto tiling_ptr = reinterpret_cast<const FusedGdnGatingTilingData *>(&tilingData);

#if __CCE_AICORE__ == 200
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(TILING_KEY_FGG_310P_DEFAULT)) {
        optiling::KernelFusedGdnGating310 op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, tiling_ptr);
        op.Process();
    }

#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_FGG_BF16_FLOAT)) {
        KernelFusedGdnGating<bfloat16_t, float> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FGG_FP16_FLOAT)) {
        KernelFusedGdnGating<half, float> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FGG_BF16_BF16)) {
        KernelFusedGdnGating<bfloat16_t, bfloat16_t> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FGG_FP16_BF16)) {
        KernelFusedGdnGating<half, bfloat16_t> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FGG_BF16_FP16)) {
        KernelFusedGdnGating<bfloat16_t, half> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FGG_FP16_FP16)) {
        KernelFusedGdnGating<half, half> op;
        op.Init(a_log, a, b, dt_bias, g, beta_output, &tilingData, &pipe);
        op.Process();
    }
#endif
}
