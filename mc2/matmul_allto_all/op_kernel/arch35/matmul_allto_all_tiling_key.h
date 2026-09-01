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
 * \file matmul_allto_all_tiling_key.h
 * \brief 定义tiling_key
 */
#ifndef MATMUL_ALLTO_ALL_TILING_KEY_H
#define MATMUL_ALLTO_ALL_TILING_KEY_H

#include <ascendc/host_api/tiling/template_argument.h>

// 量化组合模式
#define NON_QUANT_MODE 0
#define KC_QUANT_MODE 1
#define MX_QUANT_MODE 2

// bias的数据类型
#define DTYPE_BIAS_SAME_WITH_X 0
#define DTYPE_BIAS_FP32 1

// 通信方式
#define COMMMODE_CCU 0
#define COMMMODE_AICPU 1

// add_bias 分核模式 (仅非量化+有bias时生效, 0=关闭/无bias)
#define ADDBIAS_OFF 0
#define ADDBIAS_M_SPLIT 1
#define ADDBIAS_N_SPLIT 2
#define ADDBIAS_MN_SPLIT 3

// 模板参数范围声明
ASCENDC_TPL_ARGS_DECL(MatmulAlltoAll,
                      ASCENDC_TPL_UINT_DECL(QUANTMODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, NON_QUANT_MODE,
                                            KC_QUANT_MODE, MX_QUANT_MODE),
                      ASCENDC_TPL_BOOL_DECL(X2TRANSPOSE, 0, 1),
                      ASCENDC_TPL_UINT_DECL(DTYPEBIAS, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_SAME_WITH_X,
                                            DTYPE_BIAS_FP32),
                      ASCENDC_TPL_UINT_DECL(COMMMODE, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST, COMMMODE_CCU,
                                            COMMMODE_AICPU),
                      ASCENDC_TPL_UINT_DECL(ADDBIASSPLITMODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, ADDBIAS_OFF,
                                            ADDBIAS_M_SPLIT, ADDBIAS_N_SPLIT, ADDBIAS_MN_SPLIT), );

// 模板参数组合
// NON_QUANT_MODE: X2TRANSPOSE(0,1) x DTYPEBIAS(0,1) x COMMMODE(0,1) x ADDBIASSPLITMODE(0,1,2,3)
// KC/MX_QUANT_MODE: ADDBIASSPLITMODE 恒为 OFF
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QUANTMODE, ASCENDC_TPL_UI_LIST, NON_QUANT_MODE),
                                     ASCENDC_TPL_BOOL_SEL(X2TRANSPOSE, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(DTYPEBIAS, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_SAME_WITH_X,
                                                          DTYPE_BIAS_FP32),
                                     ASCENDC_TPL_UINT_SEL(COMMMODE, ASCENDC_TPL_UI_LIST, COMMMODE_CCU, COMMMODE_AICPU),
                                     ASCENDC_TPL_UINT_SEL(ADDBIASSPLITMODE, ASCENDC_TPL_UI_LIST, ADDBIAS_OFF,
                                                          ADDBIAS_M_SPLIT, ADDBIAS_N_SPLIT, ADDBIAS_MN_SPLIT)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QUANTMODE, ASCENDC_TPL_UI_LIST, KC_QUANT_MODE),
                                     ASCENDC_TPL_BOOL_SEL(X2TRANSPOSE, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(DTYPEBIAS, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_FP32),
                                     ASCENDC_TPL_UINT_SEL(COMMMODE, ASCENDC_TPL_UI_LIST, COMMMODE_CCU, COMMMODE_AICPU),
                                     ASCENDC_TPL_UINT_SEL(ADDBIASSPLITMODE, ASCENDC_TPL_UI_LIST, ADDBIAS_OFF)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QUANTMODE, ASCENDC_TPL_UI_LIST, MX_QUANT_MODE),
                                     ASCENDC_TPL_BOOL_SEL(X2TRANSPOSE, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(DTYPEBIAS, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_FP32),
                                     ASCENDC_TPL_UINT_SEL(COMMMODE, ASCENDC_TPL_UI_LIST, COMMMODE_CCU, COMMMODE_AICPU),
                                     ASCENDC_TPL_UINT_SEL(ADDBIASSPLITMODE, ASCENDC_TPL_UI_LIST, ADDBIAS_OFF)), );

#endif // MATMUL_ALLTO_ALL_TILING_KEY_H
