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
 * \file moe_compute_expert_tokens.cpp
 * \brief
 */
#include "moe_compute_expert_tokens_int32_l.h"
#include "moe_compute_expert_tokens_int32_ss.h"
#include "moe_compute_expert_tokens_int32_m.h"

using namespace MoeCompute;

extern "C" __global__ __aicore__ void moe_compute_expert_tokens(GM_ADDR sortExperts, GM_ADDR out, GM_ADDR workspace,
                                                                GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1001)) {
        MoeCompute::MoeComputeExpertTokensInt32SS<int32_t> op;
        op.Init(sortExperts, out, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        MoeCompute::MoeComputeExpertTokensInt32M<int32_t> op;
        op.Init(sortExperts, out, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1003)) {
        MoeCompute::MoeComputeExpertTokensInt32L<int32_t> op;
        op.Init(sortExperts, out, userWS, &tilingData);
        op.Process();
    }
}
