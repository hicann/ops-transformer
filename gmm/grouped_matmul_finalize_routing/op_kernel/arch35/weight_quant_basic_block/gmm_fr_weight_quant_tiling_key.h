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
 * \file grouped_matmul_finalize_routing_tiling_key.h
 * \brief
 */

#ifndef ARCH35_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILINGKEY_H
#define ARCH35_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_TILINGKEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace GroupedMatmulFinalizeRoutingArch35WeightQuantTiling {

ASCENDC_TPL_ARGS_DECL(GroupedMatmulFinalizeRouting,
                      ASCENDC_TPL_UINT_DECL(ATRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0),
                      ASCENDC_TPL_UINT_DECL(BTRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(HASBIAS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(ATRANS, ASCENDC_TPL_UI_LIST, 0),
                                     ASCENDC_TPL_UINT_SEL(BTRANS, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(HASBIAS, ASCENDC_TPL_UI_LIST, 0, 1)));
} // namespace GroupedMatmulFinalizeRoutingArch35WeightQuantTiling
#endif
