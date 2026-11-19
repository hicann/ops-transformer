/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_TILING_KEY_H
#define BLOCK_ATTN_RES_UPDATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define BARU_NORM_WEIGHT_BF16 0UL
#define BARU_NORM_WEIGHT_FP32 1UL

ASCENDC_TPL_ARGS_DECL(BlockAttnResUpdate, ASCENDC_TPL_BOOL_DECL(SINGLE_TILE, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(APPLY_OUTPUT_NORM, 0, 1),
                      ASCENDC_TPL_UINT_DECL(NORM_WEIGHT_MODE, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_LIST,
                                            BARU_NORM_WEIGHT_BF16, BARU_NORM_WEIGHT_FP32));

// Disabled normalization uses mode 0 only; the weight type is ignored in that specialization.
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(SINGLE_TILE, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(APPLY_OUTPUT_NORM, 0),
                         ASCENDC_TPL_UINT_SEL(NORM_WEIGHT_MODE, ASCENDC_TPL_UI_LIST, BARU_NORM_WEIGHT_BF16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), ASCENDC_TPL_BOOL_SEL(SINGLE_TILE, 0, 1),
                         ASCENDC_TPL_BOOL_SEL(APPLY_OUTPUT_NORM, 1),
                         ASCENDC_TPL_UINT_SEL(NORM_WEIGHT_MODE, ASCENDC_TPL_UI_LIST, BARU_NORM_WEIGHT_BF16,
                                              BARU_NORM_WEIGHT_FP32)));

#endif // BLOCK_ATTN_RES_UPDATE_TILING_KEY_H
