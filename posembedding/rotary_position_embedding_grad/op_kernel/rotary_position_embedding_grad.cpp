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
 * \file rotary_position_embedding_grad.cpp
 * \brief
 */
#include "rotate_half_grad.h"
#include "rope_interleaved_grad_splits.h"

extern "C" __global__ __aicore__ void rotary_position_embedding_grad(GM_ADDR grad, GM_ADDR cos, GM_ADDR sin, GM_ADDR x,
                                                                     GM_ADDR xGrad, GM_ADDR cosGrad, GM_ADDR sinGrad,
                                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(0)) {
        RotateHalfGrad<half, 0, true, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1)) {
        RotateHalfGrad<float, 0, true, false, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(2)) {
        RotateHalfGrad<bfloat16_t, 0, true, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(10)) {
        RotateHalfGrad<half, 1, true, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(11)) {
        RotateHalfGrad<float, 1, true, false, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(12)) {
        RotateHalfGrad<bfloat16_t, 1, true, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(20)) {
        RotateHalfGrad<half, 2, true, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(21)) {
        RotateHalfGrad<float, 2, true, false, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(22)) {
        RotateHalfGrad<bfloat16_t, 2, true, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(100)) {
        RotateHalfGrad<half, 0, true, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(101)) {
        RotateHalfGrad<float, 0, true, false, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(102)) {
        RotateHalfGrad<bfloat16_t, 0, true, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(110)) {
        RotateHalfGrad<half, 1, true, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(111)) {
        RotateHalfGrad<float, 1, true, false, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(112)) {
        RotateHalfGrad<bfloat16_t, 1, true, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(120)) {
        RotateHalfGrad<half, 2, true, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(121)) {
        RotateHalfGrad<float, 2, true, false, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(122)) {
        RotateHalfGrad<bfloat16_t, 2, true, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1000)) {
        RotateHalfGrad<half, 0, false, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1001)) {
        RotateHalfGrad<float, 0, false, false, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1002)) {
        RotateHalfGrad<bfloat16_t, 0, false, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1010)) {
        RotateHalfGrad<half, 1, false, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1011)) {
        RotateHalfGrad<float, 1, false, false, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1012)) {
        RotateHalfGrad<bfloat16_t, 1, false, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1020)) {
        RotateHalfGrad<half, 2, false, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1021)) {
        RotateHalfGrad<float, 2, false, false, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1022)) {
        RotateHalfGrad<bfloat16_t, 2, false, true, true> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1100)) {
        RotateHalfGrad<half, 0, false, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1101)) {
        RotateHalfGrad<float, 0, false, false, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1102)) {
        RotateHalfGrad<bfloat16_t, 0, false, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1110)) {
        RotateHalfGrad<half, 1, false, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1111)) {
        RotateHalfGrad<float, 1, false, false, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1112)) {
        RotateHalfGrad<bfloat16_t, 1, false, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1120)) {
        RotateHalfGrad<half, 2, false, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1121)) {
        RotateHalfGrad<float, 2, false, false, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(1122)) {
        RotateHalfGrad<bfloat16_t, 2, false, true, false> rotateHalfGrad;
        rotateHalfGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, workspace, tilingData);
        rotateHalfGrad.Process();
    } else if (TILING_KEY_IS(20000)) {
        TPipe tpipe;
        RopeInterleavedGrad<half, false, false> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(21000)) {
        TPipe tpipe;
        RopeInterleavedGrad<half, false, true> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(20100)) {
        TPipe tpipe;
        RopeInterleavedGrad<half, true, false> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(21100)) {
        TPipe tpipe;
        RopeInterleavedGrad<half, true, true> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(20010)) {
        TPipe tpipe;
        RopeInterleavedGrad<bfloat16_t, false, false> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(21010)) {
        TPipe tpipe;
        RopeInterleavedGrad<bfloat16_t, false, true> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(20110)) {
        TPipe tpipe;
        RopeInterleavedGrad<bfloat16_t, true, false> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(21110)) {
        TPipe tpipe;
        RopeInterleavedGrad<bfloat16_t, true, true> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(20020)) {
        TPipe tpipe;
        RopeInterleavedGrad<float, false, false> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(21020)) {
        TPipe tpipe;
        RopeInterleavedGrad<float, false, true> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(20120)) {
        TPipe tpipe;
        RopeInterleavedGrad<float, true, false> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    } else if (TILING_KEY_IS(21120)) {
        TPipe tpipe;
        RopeInterleavedGrad<float, true, true> ropeInterleavedGrad;
        ropeInterleavedGrad.Init(grad, cos, sin, x, xGrad, cosGrad, sinGrad, tilingData, &tpipe);
        ropeInterleavedGrad.Process();
    }
}
