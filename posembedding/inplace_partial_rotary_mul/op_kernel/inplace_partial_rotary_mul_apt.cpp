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
 * \file inplace_partial_rotary_mul_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/inplace_partial_rotary_mul_bab.h"
#include "arch35/inplace_partial_rotary_mul_ab.h"
#include "arch35/inplace_partial_rotary_mul_aba_and_ba.h"
#include "arch35/inplace_partial_rotary_mul_a_and_b.h"
#include "arch35/rotary_position_embedding_bab_mixed.h"
#include "arch35/rotary_position_embedding_aba_and_ba_mixed.h"
#include "arch35/rotary_position_embedding_a_and_b_mixed.h"
#include "arch35/rotary_position_embedding_ab_mixed.h"

#define TILING_KEY_ABA 20010
#define TILING_KEY_BA 20011
#define TILING_KEY_BAB 20020
#define TILING_KEY_AB 20030
#define TILING_KEY_A 20040
#define TILING_KEY_B 20041

#define TILING_KEY_ABA_BF16_FP32_MIXED 20110
#define TILING_KEY_ABA_FP16_FP32_MIXED 20210
#define TILING_KEY_BA_BF16_FP32_MIXED 20111
#define TILING_KEY_BA_FP16_FP32_MIXED 20211
#define TILING_KEY_BAB_BF16_FP32_MIXED 20120
#define TILING_KEY_BAB_FP16_FP32_MIXED 20220
#define TILING_KEY_AB_BF16_FP32_MIXED 20130
#define TILING_KEY_AB_FP16_FP32_MIXED 20230
#define TILING_KEY_A_BF16_FP32_MIXED 20140
#define TILING_KEY_A_FP16_FP32_MIXED 20240
#define TILING_KEY_B_BF16_FP32_MIXED 20141
#define TILING_KEY_B_FP16_FP32_MIXED 20241

using namespace AscendC;
using namespace InplacePartialRotaryMul;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    AscendC::TPipe pipe;
    // No-op: empty slice (sliceStart==sliceEnd), return directly
    {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, _noop, tiling);
        if (_noop.sliceLength == 0) {
            return;
        }
    }
    if (TILING_KEY_IS(TILING_KEY_ABA)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingABAAndBA<DTYPE_X, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BA)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingABAAndBA<DTYPE_X, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BAB)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingBAB<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_AB)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingAB<DTYPE_X> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_A)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingAAndB<DTYPE_X, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingAAndB<DTYPE_X, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    }
    // Mixed precision: x is half/bfloat16, cos/sin are float32
    else if (TILING_KEY_IS(TILING_KEY_BAB_FP16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingBABMixed<half> op(&pipe, tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BAB_BF16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingBABMixed<bfloat16_t> op(&pipe, tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    }
    // Mixed precision ABA/BA kernels
    else if (TILING_KEY_IS(TILING_KEY_ABA_FP16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingABAAndBAMixed<half, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_ABA_BF16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingABAAndBAMixed<bfloat16_t, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BA_FP16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingABAAndBAMixed<half, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BA_BF16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingABAAndBAMixed<bfloat16_t, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    }
    // Mixed precision AAndB kernels
    else if (TILING_KEY_IS(TILING_KEY_A_FP16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingAAndBMixed<half, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_A_BF16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingAAndBMixed<bfloat16_t, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B_FP16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingAAndBMixed<half, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B_BF16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingAAndBMixed<bfloat16_t, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    }
    // Mixed precision AB kernels
    else if (TILING_KEY_IS(TILING_KEY_AB_FP16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingABMixed<half> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_AB_BF16_FP32_MIXED)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRopeRegbaseTilingData, tiling_data_in, tiling);
        const InplacePartialRopeRegbaseTilingData *__restrict tilingData = &tiling_data_in;
        InplacePartialRotaryMul::RotaryPositionEmbeddingABMixed<bfloat16_t> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    }
}
