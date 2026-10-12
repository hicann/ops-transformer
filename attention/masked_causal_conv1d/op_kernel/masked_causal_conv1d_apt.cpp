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
 * \file masked_causal_conv1d_apt.cpp
 * \brief MaskedCausalConv1d kernel entry point
 */

#include "./arch35/masked_causal_conv1d.h"

#define TILING_KEY_BF16 10000
#define TILING_KEY_FP16 10001

using namespace MaskedCausalConv1dNs;

extern "C" __global__ __aicore__ void masked_causal_conv1d(
    GM_ADDR x,      // input: [S, B, H], BF16/FP16 (may be non-contiguous)
    GM_ADDR weight, // input: [3, H], BF16/FP16
    GM_ADDR mask,   // input: [B, S], BOOL (uint8_t)
    GM_ADDR y,      // output: [S, B, H], BF16/FP16
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaskedCausalConv1dTilingData);

    if (TILING_KEY_IS(TILING_KEY_BF16)) {
        GET_TILING_DATA_WITH_STRUCT(MaskedCausalConv1dTilingData, tilingData, tiling);
        MaskedCausalConv1d<bfloat16_t> op;
        op.Init(x, weight, mask, y, &tilingData);
        op.Process();
    }

    if (TILING_KEY_IS(TILING_KEY_FP16)) {
        GET_TILING_DATA_WITH_STRUCT(MaskedCausalConv1dTilingData, tilingData, tiling);
        MaskedCausalConv1d<half> op;
        op.Init(x, weight, mask, y, &tilingData);
        op.Process();
    }
}
