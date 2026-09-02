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
 * \file qgmm_inplace_add_tiling_key.h
 * \brief
 */

#ifndef __OP_KERNEL_QGMM_INPLACE_ADD_TILING_KEY_H__
#define __OP_KERNEL_QGMM_INPLACE_ADD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#if defined(ORIG_DTYPE_SCALE2) && defined(DT_FLOAT8_E8M0) && ORIG_DTYPE_SCALE2 == DT_FLOAT8_E8M0
#define V310_QGMM_QUANT_MX
#else
#define V310_QGMM_QUANT_MIX
#endif

#define QGMM_INPLACE_ADD_NO_TRANS 0
#define QGMM_INPLACE_ADD_TRANS 1

#define QGMM_INPLACE_ADD_DEQUANT_FIXP 0
#define QGMM_INPLACE_ADD_DEQUANT_VECTOR 1
#define QGMM_INPLACE_ADD_PERGROUP_PERBLOCK 2

// 模板参数
ASCENDC_TPL_ARGS_DECL(QuantGroupedMatmulInplaceAdd, // 算子OpType
                      ASCENDC_TPL_UINT_DECL(QUANT_B_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            QGMM_INPLACE_ADD_NO_TRANS, QGMM_INPLACE_ADD_TRANS),
                      ASCENDC_TPL_UINT_DECL(QUANT_A_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            QGMM_INPLACE_ADD_NO_TRANS, QGMM_INPLACE_ADD_TRANS),
                      ASCENDC_TPL_UINT_DECL(KERNEL_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            QGMM_INPLACE_ADD_DEQUANT_FIXP, QGMM_INPLACE_ADD_DEQUANT_VECTOR,
                                            QGMM_INPLACE_ADD_PERGROUP_PERBLOCK));

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
#if defined(V310_QGMM_QUANT_MX)
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
                         ASCENDC_TPL_UINT_SEL(QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, QGMM_INPLACE_ADD_NO_TRANS),
                         ASCENDC_TPL_UINT_SEL(QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, QGMM_INPLACE_ADD_TRANS),
                         ASCENDC_TPL_UINT_SEL(KERNEL_TYPE, ASCENDC_TPL_UI_LIST, QGMM_INPLACE_ADD_DEQUANT_FIXP)));
#else
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
                         ASCENDC_TPL_UINT_SEL(QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, QGMM_INPLACE_ADD_NO_TRANS),
                         ASCENDC_TPL_UINT_SEL(QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, QGMM_INPLACE_ADD_TRANS),
                         ASCENDC_TPL_UINT_SEL(KERNEL_TYPE, ASCENDC_TPL_UI_LIST, QGMM_INPLACE_ADD_DEQUANT_FIXP)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_UINT_SEL(QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, QGMM_INPLACE_ADD_NO_TRANS),
                         ASCENDC_TPL_UINT_SEL(QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, QGMM_INPLACE_ADD_TRANS),
                         ASCENDC_TPL_UINT_SEL(KERNEL_TYPE, ASCENDC_TPL_UI_LIST, QGMM_INPLACE_ADD_DEQUANT_VECTOR)));
#endif

#endif
