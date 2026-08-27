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
 * \file mhc_pre_backward_key_arch22.h
 * \brief tiling key template arguments for arch22 (A2)
 */

#ifndef MHC_PRE_BACKWARD_KEY_ARCH22_H
#define MHC_PRE_BACKWARD_KEY_ARCH22_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_pre_backward_data_arch22.h"

ASCENDC_TPL_ARGS_DECL(MhcPreBackward, ASCENDC_TPL_BOOL_DECL(TILINGKEY, 0, 1), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_BOOL_SEL(TILINGKEY, 0, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(MhcPreBackwardArch22TilingData)), )

#endif // MHC_PRE_BACKWARD_KEY_ARCH22_H
