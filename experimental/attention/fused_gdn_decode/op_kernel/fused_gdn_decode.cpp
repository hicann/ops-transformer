/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_gdn_decode.h"
#include "fused_gdn_decode_tiling_data.h"

using namespace AscendC;
using namespace FusedGdnDecode;

extern "C" __global__ __aicore__ void fused_gdn_decode(GM_ADDR mixed_qkv, GM_ADDR a, GM_ADDR b, GM_ADDR a_log,
                                                       GM_ADDR dt_bias, GM_ADDR state, GM_ADDR ssm_state_indices,
                                                       GM_ADDR out, GM_ADDR state_out, GM_ADDR workspace,
                                                       GM_ADDR tiling_gm)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(FusedGdnDecodeTilingData);
    GET_TILING_DATA(tilingData, tiling_gm);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        KernelFusedGdnDecode<bfloat16_t, float, true> op;
        op.Init(mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices, out, state_out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelFusedGdnDecode<half, float, true> op;
        op.Init(mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices, out, state_out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelFusedGdnDecode<bfloat16_t, bfloat16_t, false> op;
        op.Init(mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices, out, state_out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelFusedGdnDecode<half, half, false> op;
        op.Init(mixed_qkv, a, b, a_log, dt_bias, state, ssm_state_indices, out, state_out, &tilingData, &pipe);
        op.Process();
    }
}
