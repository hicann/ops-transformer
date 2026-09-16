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
 * \file kda_input_proj.cpp
 * \brief
 */

#include "kernel_operator.h"
#if __CCE_AICORE__ == 310
#include "arch35/kda_input_proj_kernel.h"
#endif
#include "kda_input_proj_template_tiling_key.h"

template <bool TransWeightQkv, bool TransWeightBeta, bool TransWeightGate, bool TransWeightG>
__global__ __aicore__ void kda_input_proj(__gm__ uint8_t *x, __gm__ uint8_t *weightQkv, __gm__ uint8_t *weightBeta,
                                          __gm__ uint8_t *weightGate, __gm__ uint8_t *weightG,
                                          __gm__ uint8_t *weightQkvScale, __gm__ uint8_t *qkv, __gm__ uint8_t *beta,
                                          __gm__ uint8_t *gate, __gm__ uint8_t *g, __gm__ uint8_t *workspace,
                                          __gm__ uint8_t *tiling)
{
#if __CCE_AICORE__ == 310
    REGISTER_TILING_DEFAULT(optiling::KdaInputProjTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (ORIG_DTYPE_X == DT_BF16)
    GET_TILING_DATA_WITH_STRUCT(optiling::KdaInputProjTilingData, tilingDataIn, tiling);
    const optiling::KdaInputProjTilingData *__restrict tilingData = &tilingDataIn;
    __gm__ uint8_t *userWorkspace = AscendC::GetUserWorkspace(workspace);
    KdaInputProj::KdaInputProjKernel<
        KdaInputProj::KdaInputProjType<TransWeightQkv, TransWeightBeta, TransWeightGate, TransWeightG>>
        op(x, weightQkv, weightBeta, weightGate, weightG, weightQkvScale, qkv, beta, gate, g, userWorkspace,
           tilingData);
    op.Process();
#endif
#endif
}
