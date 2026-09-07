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
 * \file all_gather_matmul_v2.cpp
 * \brief
 */

#include "../all_gather_matmul_v2_tiling_key.h"
#include "lib/matmul_intf.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../all_gather_matmul_aiv_mode.h"
#include "../all_gather_matmul_aiv_mode_tiling.h"

using namespace AllGatherMatmulAIVModeImpl;
using namespace AscendC;

template <bool TPL_ISBIAS, bool TPL_IS_TRANSPOSE_A, bool TPL_IS_TRANSPOSE_B>
__global__ __aicore__ void all_gather_matmul_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scaleInv1,
                                                GM_ADDR scaleInv2, GM_ADDR scale, GM_ADDR cGM, GM_ADDR gatherOut,
                                                GM_ADDR amax, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // aiv算子模板

#define INVOKE_ALLGATHERMATMUL_AIV_MODE_OP_IMPL(templateClass, ...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(AllGatherMatmulAIVModeTilingData, tilingData, tilingGM); \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_X2_SCALE, DTYPE_Y, __VA_ARGS__> op; \
        op.Init(aGM, bGM, biasGM, scaleInv1, scaleInv2, cGM, gatherOut, workspaceGM, tilingGM); \
        op.Process(); \
    } while (0)

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(AllGatherMatmulAIVModeTilingData);
    INVOKE_ALLGATHERMATMUL_AIV_MODE_OP_IMPL(AllGatherMatmulAIVMode, FORMAT_X2 == FORMAT_FRACTAL_NZ, false,
                                            TPL_IS_TRANSPOSE_B);
}
