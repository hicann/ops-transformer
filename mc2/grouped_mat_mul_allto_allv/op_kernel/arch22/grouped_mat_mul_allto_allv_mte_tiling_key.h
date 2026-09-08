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
 * \file grouped_mat_mul_allto_allv_mte_tiling_key.h
 * \brief
 */
#ifndef __GROUPED_MAT_MUL_ALLTO_ALLV_MTE_TILING_KEY_H__
#define __GROUPED_MAT_MUL_ALLTO_ALLV_MTE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"
#include "grouped_mat_mul_allto_allv_mte_tiling.h"
#include "../../../quant_grouped_mat_mul_allto_allv/op_kernel/quant_grouped_mat_mul_allto_allv_tiling.h"

// 通信模式
#ifndef TILINGKEY_TPL_CCU
#define TILINGKEY_TPL_CCU 0
#endif
#ifndef TILINGKEY_TPL_AICPU
#define TILINGKEY_TPL_AICPU 1
#endif
#ifndef TILINGKEY_TPL_AIV
#define TILINGKEY_TPL_AIV 2
#endif

ASCENDC_TPL_ARGS_DECL(GroupedMatMulAlltoAllv, ASCENDC_TPL_BOOL_DECL(TILINGKEY_COMPUTE_MATMUL, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_GMM_WEIGHT_TRANS, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, 0, 1),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_COMM_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_TPL_CCU, TILINGKEY_TPL_AICPU, TILINGKEY_TPL_AIV), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, 0, 1),
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, 0, 1),
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, 0, 1),
        ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU,
                             TILINGKEY_TPL_AICPU),
        ASCENDC_TPL_TILING_STRUCT_SEL(QuantGmmA2avTilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_COMPUTE_MATMUL, 0, 1),
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_GMM_WEIGHT_TRANS, 0, 1),
        ASCENDC_TPL_BOOL_SEL(TILINGKEY_SHARED_MM_WEIGHT_TRANS, 0, 1),
        ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_AIV),
        ASCENDC_TPL_TILING_STRUCT_SEL(GroupedMatMulAlltoAllvMteTilingData)),
); // GroupedMatMulAlltoAllv

#endif // __GROUPED_MAT_MUL_ALLTO_ALLV_MTE_TILING_KEY_H__
