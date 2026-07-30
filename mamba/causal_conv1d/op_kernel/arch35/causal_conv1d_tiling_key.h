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
 * \file causal_conv1d_tiling_key.h
 * \brief Tiling key constants — template parameter enumeration values.
 */

#ifndef __CAUSAL_CONV1D_TILING_KEY_H__
#define __CAUSAL_CONV1D_TILING_KEY_H__

#include "causal_conv1d_tiling_data.h"
#include "ascendc/host_api/tiling/template_argument.h"

#define CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN 0
#define CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH 1
#define CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN 2
#define CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH 3
#define CAUSAL_CONV1D_TPL_WIDTH_2 2
#define CAUSAL_CONV1D_TPL_WIDTH_3 3
#define CAUSAL_CONV1D_TPL_WIDTH_4 4
#define CAUSAL_CONV1D_TPL_HAS_BIAS_FALSE 0
#define CAUSAL_CONV1D_TPL_HAS_BIAS_TRUE 1
#define CAUSAL_CONV1D_TPL_ACTIVATION_NONE 0
#define CAUSAL_CONV1D_TPL_ACTIVATION_SILU 1
ASCENDC_TPL_ARGS_DECL(CausalConv1d,
                      ASCENDC_TPL_UINT_DECL(inputModeKey, 2, ASCENDC_TPL_UI_LIST,
                                            CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN,
                                            CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH,
                                            CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN,
                                            CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH),
                      ASCENDC_TPL_UINT_DECL(widthKey, 2, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_WIDTH_2,
                                            CAUSAL_CONV1D_TPL_WIDTH_3, CAUSAL_CONV1D_TPL_WIDTH_4),
                      ASCENDC_TPL_UINT_DECL(hasBiasKey, 1, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_HAS_BIAS_FALSE,
                                            CAUSAL_CONV1D_TPL_HAS_BIAS_TRUE),
                      ASCENDC_TPL_UINT_DECL(activationKey, 1, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_ACTIVATION_NONE,
                                            CAUSAL_CONV1D_TPL_ACTIVATION_SILU));

ASCENDC_TPL_SEL(
    // inputMode × width × hasBias × activation, full cross
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(inputModeKey, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN,
                                              CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH,
                                              CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN,
                                              CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH),
                         ASCENDC_TPL_UINT_SEL(widthKey, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_WIDTH_2,
                                              CAUSAL_CONV1D_TPL_WIDTH_3, CAUSAL_CONV1D_TPL_WIDTH_4),
                         ASCENDC_TPL_UINT_SEL(hasBiasKey, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_HAS_BIAS_FALSE,
                                              CAUSAL_CONV1D_TPL_HAS_BIAS_TRUE),
                         ASCENDC_TPL_UINT_SEL(activationKey, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_ACTIVATION_NONE,
                                              CAUSAL_CONV1D_TPL_ACTIVATION_SILU),
                         ASCENDC_TPL_TILING_STRUCT_SEL(CausalConv1dTilingData)));

#endif // __CAUSAL_CONV1D_TILING_KEY_H__
