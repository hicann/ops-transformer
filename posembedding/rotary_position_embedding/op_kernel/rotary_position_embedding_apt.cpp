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
 * \file rotary_position_embedding_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/rotary_position_embedding_bab.h"
#include "arch35/rotary_position_embedding_ab.h"
#include "arch35/rotary_position_embedding_aba_and_ba.h"
#include "arch35/rotary_position_embedding_a_and_b.h"

#define TILING_KEY_ABA 20010
#define TILING_KEY_BA 20011
#define TILING_KEY_BAB 20020
#define TILING_KEY_AB 20030
#define TILING_KEY_A 20040
#define TILING_KEY_B 20041

using namespace AscendC;
using namespace RotaryPositionEmbedding;

extern "C" __global__ __aicore__ void rotary_position_embedding(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rotate,
                                                                GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_ABA)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingABAAndBA<DTYPE_X, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BA)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingABAAndBA<DTYPE_X, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BAB)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingBAB<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_AB)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseABTilingData, tiling_data_in, tiling);
        const RopeRegbaseABTilingData* __restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingAB<DTYPE_X> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_A)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingAAndB<DTYPE_X, false> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_B)) {
        GET_TILING_DATA_WITH_STRUCT(RopeRegbaseTilingData, tiling_data_in, tiling);
        const RopeRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        RotaryPositionEmbedding::RotaryPositionEmbeddingAAndB<DTYPE_X, true> op;
        op.Init(x, cos, sin, y, workspace, tilingData, &pipe);
        op.Process();
    }
}
