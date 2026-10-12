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
 * \file apply_rotary_pos_emb_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch35/apply_rotary_pos_emb_aba_and_ba.h"
#include "arch35/apply_rotary_pos_emb_bab.h"
#include "arch35/apply_rotary_pos_emb_ab.h"

#define TILING_KEY_ABA 20010
#define TILING_KEY_BA 20011
#define TILING_KEY_BAB 20020
#define TILING_KEY_AB 20030

using namespace ApplyRotaryPosEmb;

extern "C" __global__ __aicore__ void apply_rotary_pos_emb(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin,
                                                           GM_ADDR q_out, GM_ADDR k_out, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_AB)) {
        GET_TILING_DATA_WITH_STRUCT(ApplyRotaryPosEmbRegbaseABTilingData, tiling_data_in, tiling);
        const ApplyRotaryPosEmbRegbaseABTilingData* __restrict tilingData = &tiling_data_in;
        ApplyRotaryPosEmbAB<DTYPE_QUERY> op;
        op.Init(q, k, cos, sin, q_out, k_out, workspace, tilingData, &pipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TILING_KEY_BAB)) {
        GET_TILING_DATA_WITH_STRUCT(ApplyRotaryPosEmbRegbaseTilingData, tiling_data_in, tiling);
        const ApplyRotaryPosEmbRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        ApplyRotaryPosEmb::ApplyRotaryPosEmbBAB<DTYPE_QUERY> op(&pipe, tilingData);
        op.Init(q, k, cos, sin, q_out, k_out, workspace);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_ABA)) {
        GET_TILING_DATA_WITH_STRUCT(ApplyRotaryPosEmbRegbaseTilingData, tiling_data_in, tiling);
        const ApplyRotaryPosEmbRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        ApplyRotaryPosEmb::ApplyRotaryPosEmbABAAndBA<DTYPE_QUERY, false> op;
        op.Init(q, k, cos, sin, q_out, k_out, workspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BA)) {
        GET_TILING_DATA_WITH_STRUCT(ApplyRotaryPosEmbRegbaseTilingData, tiling_data_in, tiling);
        const ApplyRotaryPosEmbRegbaseTilingData* __restrict tilingData = &tiling_data_in;
        ApplyRotaryPosEmb::ApplyRotaryPosEmbABAAndBA<DTYPE_QUERY, true> op;
        op.Init(q, k, cos, sin, q_out, k_out, workspace, tilingData, &pipe);
        op.Process();
    }
}
