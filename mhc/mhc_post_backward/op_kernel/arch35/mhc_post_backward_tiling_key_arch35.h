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
 * \file mhc_post_backward_tiling_key.h
 * \brief mhc_post_backward tiling key declare
 */

#ifndef __MHC_POST_BACKWARD_TILING_KEY_ARCH35_H__
#define __MHC_POST_BACKWARD_TILING_KEY_ARCH35_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define MHC_POST_BACKWARD_HIGH_LEVEL_MATMUL 1
#define MHC_POST_BACKWARD_MERGE_BATCH_MATMUL 2

// UINT values use list indices: high-level = 0, merge-batch = 1; IS_HRES occupies bit 2.
#define MHC_POST_BACKWARD_MERGE_BATCH_KEY_PATTERN "(TILING_KEY_VAR & 3) == 1"

ASCENDC_TPL_ARGS_DECL(MhcPostBackward,
                      ASCENDC_TPL_UINT_DECL(MATMUL_IMPL, 2, ASCENDC_TPL_UI_LIST, MHC_POST_BACKWARD_HIGH_LEVEL_MATMUL,
                                            MHC_POST_BACKWARD_MERGE_BATCH_MATMUL),
                      ASCENDC_TPL_BOOL_DECL(IS_HRES, 0, 1));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_UINT_SEL(MATMUL_IMPL, ASCENDC_TPL_UI_LIST, MHC_POST_BACKWARD_HIGH_LEVEL_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(IS_HRES, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MhcPostBackwardTilingDataArch35)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_UINT_SEL(MATMUL_IMPL, ASCENDC_TPL_UI_LIST, MHC_POST_BACKWARD_MERGE_BATCH_MATMUL),
                         ASCENDC_TPL_BOOL_SEL(IS_HRES, 0, 1),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MhcPostBackwardOptimizedTilingDataArch35)));

#endif
