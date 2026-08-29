/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTO_ALL_MATMUL_APACE_TILING_KEY_H
#define ALLTO_ALL_MATMUL_APACE_TILING_KEY_H

#include <ascendc/host_api/tiling/template_argument.h>

#define NON_QUANT_MODE 0
#define MX_QUANT_MODE 6
#define DTYPE_BIAS_SAME_WITH_X 0
#define DTYPE_BIAS_FP32 1
#define ALL2ALL_COMM_TYPE_CCU 0
#define ALL2ALL_COMM_TYPE_AICPU 1
#define ALL2ALL_COMM_TYPE_UDMA 2

ASCENDC_TPL_ARGS_DECL(AlltoAllMatmulV2,
                      ASCENDC_TPL_UINT_DECL(QUANTMODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, NON_QUANT_MODE,
                                            MX_QUANT_MODE),
                      ASCENDC_TPL_BOOL_DECL(X2TRANSPOSE, 0, 1),
                      ASCENDC_TPL_UINT_DECL(DTYPEBIAS, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_SAME_WITH_X,
                                            DTYPE_BIAS_FP32),
                      ASCENDC_TPL_BOOL_DECL(ISSMALLK, 0, 1),
                      ASCENDC_TPL_UINT_DECL(COMMTYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, ALL2ALL_COMM_TYPE_CCU,
                                            ALL2ALL_COMM_TYPE_AICPU, ALL2ALL_COMM_TYPE_UDMA), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QUANTMODE, ASCENDC_TPL_UI_LIST, MX_QUANT_MODE),
                                     ASCENDC_TPL_BOOL_SEL(X2TRANSPOSE, 1),
                                     ASCENDC_TPL_UINT_SEL(DTYPEBIAS, ASCENDC_TPL_UI_LIST, DTYPE_BIAS_FP32),
                                     ASCENDC_TPL_BOOL_SEL(ISSMALLK, 0),
                                     ASCENDC_TPL_UINT_SEL(COMMTYPE, ASCENDC_TPL_UI_LIST, ALL2ALL_COMM_TYPE_UDMA)));

#endif
