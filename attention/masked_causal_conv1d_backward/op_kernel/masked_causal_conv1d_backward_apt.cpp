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
 * \file masked_causal_conv1d_backward_apt.cpp
 * \brief AICore entry for masked_causal_conv1d_backward
 */

#include "kernel_operator.h"
#include "arch35/masked_causal_conv1d_backward.h"
#include "arch35/masked_causal_conv1d_backward_struct.h"

#define TILING_KEY_MASKED_CAUSAL_CONV1D_BACKWARD_BF16 10000
#define TILING_KEY_MASKED_CAUSAL_CONV1D_BACKWARD_FP16 10001

using namespace AscendC;
using MaskedCausalConv1dBackwardArch35Tiling::MaskedCausalConv1dBackwardTilingDataV35;
using MaskedCausalConv1dBackwardKernelNS::MaskedCausalConv1dBackward;

extern "C" __global__ __aicore__ void masked_causal_conv1d_backward(GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight,
                                                                    GM_ADDR mask, GM_ADDR grad_x, GM_ADDR grad_weight,
                                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaskedCausalConv1dBackwardTilingDataV35);
    GET_TILING_DATA_WITH_STRUCT(MaskedCausalConv1dBackwardTilingDataV35, td, tiling);

    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_MASKED_CAUSAL_CONV1D_BACKWARD_BF16)) {
        MaskedCausalConv1dBackward<bfloat16_t> op;
        op.Init(grad_y, x, weight, mask, grad_x, grad_weight, &td, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_MASKED_CAUSAL_CONV1D_BACKWARD_FP16)) {
        MaskedCausalConv1dBackward<half> op;
        op.Init(grad_y, x, weight, mask, grad_x, grad_weight, &td, &pipe);
        op.Process();
    }
}
