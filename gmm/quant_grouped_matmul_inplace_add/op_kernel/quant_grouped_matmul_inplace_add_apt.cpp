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
 * \file quant_grouped_matmul_inplace_add_apt.cpp
 * \brief
 */
#include "arch35/qgmm_inplace_add_utils.h"
#include "arch35/quant_grouped_matmul_inplace_add_tiling_data.h"
#include "arch35/qgmm_inplace_add_tiling_key.h"
#if defined(V310_QGMM_QUANT_MX)
#include "arch35/qgmm_inplace_add_cube_on_the_fly.h"
#else
#include "arch35/qgmm_inplace_add_mix_online_dynamic.h"
#include "arch35/qgmm_inplace_add_cube_basic_api.h"
#endif

using namespace AscendC;
using namespace matmul;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
#endif
__global__ __aicore__ void quant_grouped_matmul_inplace_add(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2, GM_ADDR groupList,
                                                            GM_ADDR yIn, GM_ADDR scale1, GM_ADDR y, GM_ADDR workspace,
                                                            GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

#ifndef __CCE_KT_TEST__
#if defined(V310_QGMM_QUANT_MX) // mxfpx
    if constexpr (QUANT_B_TRANS == QGMM_INPLACE_ADD_NO_TRANS && QUANT_A_TRANS == QGMM_INPLACE_ADD_TRANS &&
                  KERNEL_TYPE == QGMM_INPLACE_ADD_DEQUANT_FIXP) { // transX = true, transW = false
        GET_TILING_DATA_WITH_STRUCT(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, tilingData, tiling);
        QGmmInplaceAddAswt<Cgmct::Gemm::layout::ColumnMajor, Cgmct::Gemm::layout::RowMajor>(x1, x2, scale2, groupList,
                                                                                            scale1, y, tiling);
    }
#else
    if constexpr (QUANT_B_TRANS == QGMM_INPLACE_ADD_NO_TRANS && QUANT_A_TRANS == QGMM_INPLACE_ADD_TRANS &&
                  KERNEL_TYPE == QGMM_INPLACE_ADD_DEQUANT_VECTOR) { // transX = true, transW = false, T-C
        GET_TILING_DATA_WITH_STRUCT(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddTilingDataParams, tilingData, tiling);
        QGmmInplaceAddMixAswt<Cgmct::Gemm::layout::ColumnMajor, Cgmct::Gemm::layout::RowMajor>(
            x1, x2, scale2, groupList, scale1, y, tiling);
    }
    if constexpr (QUANT_B_TRANS == QGMM_INPLACE_ADD_NO_TRANS && QUANT_A_TRANS == QGMM_INPLACE_ADD_TRANS &&
                  KERNEL_TYPE == QGMM_INPLACE_ADD_DEQUANT_FIXP) { // transX = true, transW = false, T-T
        GET_TILING_DATA_WITH_STRUCT(QuantGroupedMatmulInplaceAdd::QGmmInplaceAddBasicApiTilingData, tilingData, tiling);
        QGmmInplaceAddCubeBasicAPI<Cgmct::Gemm::layout::ColumnMajor, Cgmct::Gemm::layout::RowMajor>(
            x1, x2, scale2, groupList, scale1, y, tiling);
    }
#endif

#endif
}
