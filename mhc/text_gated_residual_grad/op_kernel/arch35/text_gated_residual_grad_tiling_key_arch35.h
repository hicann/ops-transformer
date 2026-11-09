/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file text_gated_residual_grad_tiling_key_arch35.h
 * \brief TextGatedResidualGrad template selection for input dtype and optional gamma.
 */

#ifndef TEXT_GATED_RESIDUAL_GRAD_TILING_KEY_ARCH35_H
#define TEXT_GATED_RESIDUAL_GRAD_TILING_KEY_ARCH35_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "text_gated_residual_grad_tiling_data_arch35.h"

// Template metadata consumes preprocessor values for the x input and gamma variants.
#define HC_GRAD_TPL_INPUT_X 1
#define HC_GRAD_TPL_GAMMA_ABSENT 0
#define HC_GRAD_TPL_GAMMA_PRESENT 1

ASCENDC_TPL_ARGS_DECL(TextGatedResidualGrad,
                      ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_BF16, C_DT_FLOAT16, ASCENDC_TPL_INPUT(HC_GRAD_TPL_INPUT_X)),
                      ASCENDC_TPL_BOOL_DECL(HAS_GAMMA, HC_GRAD_TPL_GAMMA_ABSENT, HC_GRAD_TPL_GAMMA_PRESENT));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2), ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_BF16),
                         ASCENDC_TPL_BOOL_SEL(HAS_GAMMA, HC_GRAD_TPL_GAMMA_ABSENT, HC_GRAD_TPL_GAMMA_PRESENT),
                         ASCENDC_TPL_TILING_STRUCT_SEL(TextGatedResidualGradTilingDataArch35)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_DATATYPE_SEL(D_T, C_DT_FLOAT16),
                         ASCENDC_TPL_BOOL_SEL(HAS_GAMMA, HC_GRAD_TPL_GAMMA_ABSENT, HC_GRAD_TPL_GAMMA_PRESENT),
                         ASCENDC_TPL_TILING_STRUCT_SEL(TextGatedResidualGradTilingDataArch35)), );

#endif // TEXT_GATED_RESIDUAL_GRAD_TILING_KEY_ARCH35_H
