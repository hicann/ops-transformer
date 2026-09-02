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
 * \file lightning_indexer_grad_template_tiling_key.h
 * \brief
 */
#ifndef TEMPLATE_TILING_KEY_LIG_H_
#define TEMPLATE_TILING_KEY_LIG_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define LIG_TPL_FP16 1
#define LIG_TPL_INT32 3
#define LIG_TPL_BF16 27

#define LIG_LAYOUT_BSND 0
#define LIG_LAYOUT_TND 1

#define ASCENDC_TPL_4_BW 4

ASCENDC_TPL_ARGS_DECL(LightningIndexerGrad, ASCENDC_TPL_DTYPE_DECL(DATATYPE, LIG_TPL_FP16, LIG_TPL_BF16),
                      ASCENDC_TPL_UINT_DECL(LAYOUT, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, LIG_LAYOUT_BSND,
                                            LIG_LAYOUT_TND));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DATATYPE, LIG_TPL_FP16, LIG_TPL_BF16),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT, ASCENDC_TPL_UI_LIST, LIG_LAYOUT_BSND, LIG_LAYOUT_TND),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(LIGTilingData)), );

#endif
