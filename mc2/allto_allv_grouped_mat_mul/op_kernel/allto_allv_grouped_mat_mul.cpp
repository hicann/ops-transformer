/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "allto_allv_grouped_mat_mul_tiling.h"
#include "allto_allv_grouped_mat_mul_tiling_key.h"
#include "../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/mc2_templates.h"

using namespace AscendC;
using namespace MC2KernelTemplate;
using namespace Mc2GroupedMatmulTilingData;

template <bool TILINGKEY_GMM_WEIGHT_TRANSPOSE, bool TILINGKEY_MM_WEIGHT_TRANSPOSE, int TILINGKEY_COMM_MODE>
__global__ __aicore__ void allto_allv_grouped_mat_mul(GM_ADDR gmmxGM, GM_ADDR gmmweightGM,
                                                      GM_ADDR sendCountsTensorOptionalGM,
                                                      GM_ADDR recvCountsTensorOptionalGM, GM_ADDR mmxOptionalGM,
                                                      GM_ADDR mmweightOptionalGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
                                                      GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM,
                                                      GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspaceGM == nullptr) {
        return;
    }
    SetSysWorkspace(workspaceGM);
    GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
    if (userWorkspace == nullptr) {
        return;
    }
    TPipe pipe;

    REGISTER_TILING_DEFAULT(AlltoAllvGmmTilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    using ComputeOpType =
        QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, DTYPE_GMM_X, DTYPE_GMM_WEIGHT, float,
                           DTYPE_GMM_Y, CubeFormat::ND, false, TILINGKEY_GMM_WEIGHT_TRANSPOSE, false, true>;
    using LocalComputeOpType =
        QuantGroupedMatmul<AlltoAllvGmmTilingData, GMMQuantTilingData, DTYPE_MM_X, DTYPE_MM_WEIGHT, float, DTYPE_MM_Y,
                           CubeFormat::ND, false, TILINGKEY_MM_WEIGHT_TRANSPOSE, true, true>;
    A2avGmmScheduler<HcclA2avOp<DTYPE_GMM_X, true, TILINGKEY_COMM_MODE>, ComputeOpType, LocalComputeOpType,
                     AlltoAllvGmmTilingData, GMMQuantTilingData, MC2_TILING_TYPE>
        a2avGmmScheduler;

    GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, gmmQuantTilingData, gmmArray,
                                       gmmArrayAddr_, tilingGM);
    GET_NESTED_TILING_DATA_MEMBER_ADDR(AlltoAllvGmmTilingData, GMMQuantTilingData, mmQuantTilingData, gmmArray,
                                       mmArrayAddr_, tilingGM);

    a2avGmmScheduler.Init(gmmxGM, gmmweightGM, mmxOptionalGM, mmweightOptionalGM, nullptr, nullptr, nullptr, nullptr,
                          gmmyGM, mmyOptionalGM, permuteOutOptionalGM, userWorkspace, tilingGM, gmmArrayAddr_,
                          mmArrayAddr_, &pipe, true);
    a2avGmmScheduler.Process();
}
