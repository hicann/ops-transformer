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
 * \file interleave_rope.cpp
 * \brief
 */

#include "interleave_rope_fixed_bnsd_b11d.h"
#include "interleave_rope_b11d.h"
#include "interleave_rope_b1sd.h"
#include "interleave_rope_split_s.h"

using namespace InterleaveRope;

#define INTERLEAVE_ROPE_FIXED_BNSD_B11D 1000
#define INTERLEAVE_ROPE_B11D 2000
#define INTERLEAVE_ROPE_B1SD 3000
#define INTERLEAVE_ROPE_SPLIT_S 4000

extern "C" __global__ __aicore__ void interleave_rope(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR workspace,
                                                      GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(INTERLEAVE_ROPE_FIXED_BNSD_B11D)) {
        KernelInterleaveRopeFixBNSD<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(INTERLEAVE_ROPE_B11D)) {
        KernelInterleaveRopeB11D<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(INTERLEAVE_ROPE_B1SD)) {
        KernelInterleaveRopeB1SD<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(INTERLEAVE_ROPE_SPLIT_S)) {
        KernelInterleaveRopeSplitS<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    }
}
