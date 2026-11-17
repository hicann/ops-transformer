/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEMPLATE_TILING_KEY_QUANT_FLASH_MLA_ARCH22_H_
#define TEMPLATE_TILING_KEY_QUANT_FLASH_MLA_ARCH22_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quant_flash_mla_common_def_arch22.h"

ASCENDC_TPL_ARGS_DECL(QuantFlashMla, ASCENDC_TPL_UINT_DECL(QuantMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 15),
                      ASCENDC_TPL_UINT_DECL(InOutLayout, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 15),
                      ASCENDC_TPL_UINT_DECL(KvLayout, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 15),
                      ASCENDC_TPL_UINT_DECL(HasAttenMask, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 15));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, QFMLA_MLA_INT8_FULLQUANT),
                                     ASCENDC_TPL_UINT_SEL(InOutLayout, ASCENDC_TPL_UI_LIST, InOutLayoutType_BSH_BSH,
                                                          InOutLayoutType_TND_TND),
                                     ASCENDC_TPL_UINT_SEL(KvLayout, ASCENDC_TPL_UI_LIST, KvLayoutType_PA_NZ),
                                     ASCENDC_TPL_UINT_SEL(HasAttenMask, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(IncreFlashAttentionTilingDataMla)));

#endif
