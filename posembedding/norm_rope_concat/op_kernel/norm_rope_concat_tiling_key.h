/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file norm_rope_concat_tiling_key.h
 * \brief
 */

#ifndef _NORM_ROPE_CONCAT_TILING_KEY_H_
#define _NORM_ROPE_CONCAT_TILING_KEY_H_
#include "ascendc/host_api/tiling/template_argument.h"
ASCENDC_TPL_ARGS_DECL(NormRopeConcat, ASCENDC_TPL_UINT_DECL(NORM_TYPE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                      ASCENDC_TPL_UINT_DECL(NORM_ADDED_TYPE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                      ASCENDC_TPL_UINT_DECL(ROPE_TYPE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 2),
                      ASCENDC_TPL_UINT_DECL(CONCAT_ORDER, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(IS_TRAINING, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(NORM_TYPE, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                                     ASCENDC_TPL_UINT_SEL(NORM_ADDED_TYPE, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                                     ASCENDC_TPL_UINT_SEL(ROPE_TYPE, ASCENDC_TPL_UI_RANGE, 1, 0, 2),
                                     ASCENDC_TPL_UINT_SEL(CONCAT_ORDER, ASCENDC_TPL_UI_RANGE, 1, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(IS_TRAINING, 0, 1)));
#endif // _NORM_ROPE_CONCAT_TILING_KEY_H_
