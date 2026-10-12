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
 * \file apply_rotary_pos_emb.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_small.h"
#include "apply_rotary_pos_emb_compute_ab.h"
#include "apply_rotary_pos_emb_compute_ab_cast.h"

using namespace ApplyRotaryPosEmb;

extern "C" __global__ __aicore__ void apply_rotary_pos_emb(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin,
                                                           GM_ADDR q_out, GM_ADDR k_out, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
#if ORIG_DTYPE_QUERY != DT_BF16
    if (TILING_KEY_IS(1)) {
        ApplyRotaryPosEmb::ARPESmall<DTYPE_QUERY, DTYPE_QUERY> op;
        op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
        op.Process(&tilingData);
    } else if (TILING_KEY_IS(3)) {
        ApplyRotaryPosEmb::ARPEComputeAB<DTYPE_QUERY, DTYPE_QUERY> op;
        op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
        op.Process(&tilingData);
    } else if (TILING_KEY_IS(4)) {
        ApplyRotaryPosEmb::ARPEComputeABCast<DTYPE_QUERY, DTYPE_QUERY> op;
        op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
        op.Process(&tilingData);
    } else {
        return;
    }
#else
    if (TILING_KEY_IS(1)) {
        ApplyRotaryPosEmb::ARPESmall<DTYPE_QUERY, float> op;
        op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
        op.Process(&tilingData);
    } else if (TILING_KEY_IS(4)) {
        ApplyRotaryPosEmb::ARPEComputeABCast<DTYPE_QUERY, float> op;
        op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
        op.Process(&tilingData);
    } else {
        return;
    }
#endif
}
