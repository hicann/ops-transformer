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
 * \file mhc_pre_backward_tiling_key.h
 * \brief
 */

#ifndef __OP_KERNEL_MHC_PRE_BACKWARD_TILING_KEY_H__
#define __OP_KERNEL_MHC_PRE_BACKWARD_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define MHC_PRE_BACKWARD_DEFAULT 0

ASCENDC_TPL_ARGS_DECL(MhcPreBackward, ASCENDC_TPL_UINT_DECL(TILING_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                                            MHC_PRE_BACKWARD_DEFAULT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TILING_MODE, ASCENDC_TPL_UI_LIST, MHC_PRE_BACKWARD_DEFAULT)));

#endif
