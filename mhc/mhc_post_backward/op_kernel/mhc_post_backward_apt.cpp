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
 * \file mhc_post_backward.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./arch35/mhc_post_backward_kernel.h"
#include "./arch35/mhc_post_backward_kernel_merge_batch.h"
#include "./arch35/mhc_post_backward_tiling_data_arch35.h"
#include "./arch35/mhc_post_backward_tiling_key_arch35.h"

using namespace AscendC;
using namespace MhcPostBackward;

template <int64_t MATMUL_IMPL, int64_t IS_HRES>
__global__ __aicore__ void mhc_post_backward(GM_ADDR gradOutput, GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
                                             GM_ADDR gradX, GM_ADDR gradHRes, GM_ADDR gradHOut, GM_ADDR gradHPost,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MhcPostBackwardTilingDataArch35);
    REGISTER_TILING_FOR_TILINGKEY(MHC_POST_BACKWARD_MERGE_BATCH_KEY_PATTERN, MhcPostBackwardOptimizedTilingDataArch35);
    TPipe pipe;
    if constexpr (MATMUL_IMPL == MHC_POST_BACKWARD_MERGE_BATCH_MATMUL) {
        GET_TILING_DATA_WITH_STRUCT(MhcPostBackwardOptimizedTilingDataArch35, tilingData, tiling);
        MhcPostBackwardKernelOptimized<DTYPE_GRAD_Y, IS_HRES, false, false> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, nullptr, &tilingData, &pipe);
        op.Process();
    } else {
        GET_TILING_DATA(tilingData, tiling);
        if (workspace == nullptr) {
            return;
        }
        GM_ADDR userWorkspace = GetUserWorkspace(workspace);
        if (userWorkspace == nullptr) {
            return;
        }
        MhcPostBackwardKernel<DTYPE_GRAD_Y, IS_HRES> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWorkspace, &tilingData,
                &pipe);
        op.Process();
    }
}
