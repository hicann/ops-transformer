/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

#define RKDA_TPL_COMPAT 0
#define RKDA_TPL_V_FIRST 1
#define RKDA_TPL_K_FIRST 2
#define RKDA_TPL_V_FIRST_FUSED 3
#define RKDA_TPL_K_FIRST_FUSED 4

ASCENDC_TPL_ARGS_DECL(RecurrentKda,
                      ASCENDC_TPL_UINT_DECL(KERNEL_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, RKDA_TPL_V_FIRST,
                                            RKDA_TPL_K_FIRST, RKDA_TPL_V_FIRST_FUSED, RKDA_TPL_K_FIRST_FUSED));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(KERNEL_MODE, ASCENDC_TPL_UI_LIST, RKDA_TPL_V_FIRST,
                                                          RKDA_TPL_K_FIRST, RKDA_TPL_V_FIRST_FUSED,
                                                          RKDA_TPL_K_FIRST_FUSED)));
