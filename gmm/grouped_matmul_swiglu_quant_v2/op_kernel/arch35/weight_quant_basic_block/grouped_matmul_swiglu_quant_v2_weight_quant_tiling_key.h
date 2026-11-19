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
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_tiling_key.h
 * \brief
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_KEY_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h"
#include "../grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"

#define GMM_SWIGLU_QUANT_NO_TRANS 0
#define GMM_SWIGLU_QUANT_TRANS 1
#define GMM_SWIGLU_QUANT_NOT_SINGLE_MULTI_SINGLE 0
#define GMM_SWIGLU_QUANT_SINGLE_MULTI_SINGLE 1
#define GMM_SWIGLU_QUANT_ORIGINAL_KERNEL_TYPE 0
#define GMM_SWIGLU_QUANT_TENSOR_LEVEL_KERNEL_TYPE 1
#define GMM_SWIGLU_HIF4_QUANT_KERNEL_TYPE 2
#define GMM_SWIGLU_QUANT_V3_A8W4_KERNEL_TYPE 3
#define GMM_SWIGLU_QUANT_V3_A8W4_NPU_ARCH 3510
#define GMM_SWIGLU_QUANT_V3_A8W4_TILING_KEY_MATCH "TILING_KEY_VAR == 49"

// UINT encodes list indices. Match the general declaration, including reserved value 2.

// 模板参数
ASCENDC_TPL_ARGS_DECL(GroupedMatmulSwigluQuantV2, // 算子OpType
                      ASCENDC_TPL_UINT_DECL(QUANT_B_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            GMM_SWIGLU_QUANT_NO_TRANS, GMM_SWIGLU_QUANT_TRANS),
                      ASCENDC_TPL_UINT_DECL(QUANT_A_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            GMM_SWIGLU_QUANT_NO_TRANS, GMM_SWIGLU_QUANT_TRANS),
                      ASCENDC_TPL_UINT_DECL(KERNEL_TYPE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            GMM_SWIGLU_QUANT_ORIGINAL_KERNEL_TYPE,
                                            GMM_SWIGLU_QUANT_TENSOR_LEVEL_KERNEL_TYPE,
                                            GMM_SWIGLU_HIF4_QUANT_KERNEL_TYPE, GMM_SWIGLU_QUANT_V3_A8W4_KERNEL_TYPE));

#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ == GMM_SWIGLU_QUANT_V3_A8W4_NPU_ARCH && defined(FORMAT_WEIGHT) && \
                               FORMAT_WEIGHT == FORMAT_FRACTAL_NZ && ORIG_DTYPE_Y == DT_FLOAT8_E4M3FN && \
                               ORIG_DTYPE_Y_SCALE == DT_FLOAT8_E8M0 && ORIG_DTYPE_WEIGHT_SCALE == DT_FLOAT8_E8M0)
#define GMM_SWIGLU_QUANT_V3_A8W4_ENABLED 1
#define GMMSQ_A8W4_V3_SELECTION \
    , ASCENDC_TPL_ARGS_SEL( \
          ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2), \
          ASCENDC_TPL_UINT_SEL(QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, GMM_SWIGLU_QUANT_TRANS), \
          ASCENDC_TPL_UINT_SEL(QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, GMM_SWIGLU_QUANT_NO_TRANS), \
          ASCENDC_TPL_UINT_SEL(KERNEL_TYPE, ASCENDC_TPL_UI_LIST, GMM_SWIGLU_QUANT_V3_A8W4_KERNEL_TYPE), \
          ASCENDC_TPL_TILING_STRUCT_SEL(GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData))
#else
#define GMM_SWIGLU_QUANT_V3_A8W4_ENABLED 0
#define GMMSQ_A8W4_V3_SELECTION
#endif

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ASCENDC_TPL_UINT_SEL(QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, GMM_SWIGLU_QUANT_TRANS),
    ASCENDC_TPL_UINT_SEL(QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, GMM_SWIGLU_QUANT_NO_TRANS),
    ASCENDC_TPL_UINT_SEL(KERNEL_TYPE, ASCENDC_TPL_UI_LIST, GMM_SWIGLU_QUANT_NOT_SINGLE_MULTI_SINGLE,
                         GMM_SWIGLU_QUANT_SINGLE_MULTI_SINGLE),
    ASCENDC_TPL_TILING_STRUCT_SEL(GMMSQArch35Tiling::GMMSQWeightQuantTilingData)) GMMSQ_A8W4_V3_SELECTION);
#undef GMMSQ_A8W4_V3_SELECTION
#endif // GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_KEY_H
