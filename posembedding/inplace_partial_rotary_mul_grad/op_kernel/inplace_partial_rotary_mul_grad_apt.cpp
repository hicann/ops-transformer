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
 * \file inplace_partial_rotary_mul_grad_apt.cpp
 * \brief InplacePartialRotaryMulGrad AscendC kernel entry for arch35 / ascend950.
 *
 * The (dy, cos, sin) dtype combination is selected at compile time via DTYPE_DY / DTYPE_COS macros
 * injected by the build system from the JSON binary config entries.
 * Note: sin always shares DTYPE_COS (cos and sin have identical dtypes per the op constraint).
 */

#include "kernel_operator.h"
#include "arch35/inplace_partial_rotary_mul_grad_common.h"
#include "arch35/inplace_partial_rotary_mul_grad_bab.h"
#include "arch35/inplace_partial_rotary_mul_grad_ab.h"
#include "arch35/inplace_partial_rotary_mul_grad_aba_and_ba.h"
#include "arch35/inplace_partial_rotary_mul_grad_a_and_b.h"

#define TILING_KEY_ABA 201
#define TILING_KEY_BA 202
#define TILING_KEY_BAB 203
#define TILING_KEY_AB 204
#define TILING_KEY_A 205
#define TILING_KEY_B 206
#define TILING_KEY_EMPTY 403

using namespace AscendC;
using namespace InplacePartialRotaryMulGrad;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_grad(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR dx,
                                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_EMPTY)) {
        return;
    } else if (TILING_KEY_IS(TILING_KEY_ABA)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradRegbaseTilingData, tilingDataAba, tiling);
        const InplacePartialRotaryMulGradRegbaseTilingData* __restrict tilingData = &tilingDataAba;
        InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, false> op(&pipe, tilingData);
        op.Init(dy, cos, sin, dx);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BA)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradRegbaseTilingData, tilingDataBa, tiling);
        const InplacePartialRotaryMulGradRegbaseTilingData* __restrict tilingData = &tilingDataBa;
        InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, true> op(&pipe, tilingData);
        op.Init(dy, cos, sin, dx);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BAB)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradRegbaseTilingData, tilingDataBab, tiling);
        const InplacePartialRotaryMulGradRegbaseTilingData* __restrict tilingData = &tilingDataBab;
        InplacePartialRotaryMulGradBAB<DTYPE_DY, DTYPE_COS> op(&pipe, tilingData);
        op.Init(dy, cos, sin, dx);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_AB)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradRegbaseTilingDataAb, tilingDataAb, tiling);
        const InplacePartialRotaryMulGradRegbaseTilingDataAb* __restrict tilingData = &tilingDataAb;
        InplacePartialRotaryMulGradAB<DTYPE_DY, DTYPE_COS> op(&pipe, tilingData);
        op.Init(dy, cos, sin, dx);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_A)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradRegbaseTilingData, tilingDataA, tiling);
        const InplacePartialRotaryMulGradRegbaseTilingData* __restrict tilingData = &tilingDataA;
        InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, false> op(&pipe, tilingData);
        op.Init(dy, cos, sin, dx);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradRegbaseTilingData, tilingDataB, tiling);
        const InplacePartialRotaryMulGradRegbaseTilingData* __restrict tilingData = &tilingDataB;
        InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, true> op(&pipe, tilingData);
        op.Init(dy, cos, sin, dx);
        op.Process();
    }
}
