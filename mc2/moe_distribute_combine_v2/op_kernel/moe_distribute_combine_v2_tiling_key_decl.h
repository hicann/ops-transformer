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
 * \file moe_distribute_combine_v2_tiling_key_decl.h
 * \brief common (arch-independent) tiling key macros and ARGS declaration.
 *        Arch-specific ASCENDC_TPL_SEL lives in arch35/ and arch22/ variants.
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_TILING_KEY_DECL_H
#define MOE_DISTRIBUTE_COMBINE_V2_TILING_KEY_DECL_H
#include "ascendc/host_api/tiling/template_argument.h"

namespace Mc2Tiling {
#define TILINGKEY_TPL_A2 0
#define TILINGKEY_TPL_A3 1
#define TILINGKEY_TPL_A5 2
#define TILINGKEY_TPL_A6 3
#define TILINGKEY_NO_QUANT 0
#define TILINGKEY_INT8_QUANT 2
#define TILINGKEY_MXFP8_E5M2_QUANT 3
#define TILINGKEY_MXFP8_E4M3_QUANT 4
#define TILINGKEY_TPL_MTE 0
#define TILINGKEY_TPL_AICPU 1
#define TILINGKEY_TPL_CCU 2
#define TILINGKEY_TPL_HIERARCHY 3
#define TILINGKEY_TPL_ROCE 4

ASCENDC_TPL_ARGS_DECL(MoeDistributeCombineV2,
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_QUANT_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_NO_QUANT, TILINGKEY_INT8_QUANT, TILINGKEY_MXFP8_E5M2_QUANT,
                                            TILINGKEY_MXFP8_E4M3_QUANT),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_LAYERED_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_TPL_MTE, TILINGKEY_TPL_AICPU, TILINGKEY_TPL_CCU,
                                            TILINGKEY_TPL_HIERARCHY, TILINGKEY_TPL_ROCE),
                      ASCENDC_TPL_UINT_DECL(ARCH_TAG, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A2,
                                            TILINGKEY_TPL_A3, TILINGKEY_TPL_A5, TILINGKEY_TPL_A6), );
} // namespace Mc2Tiling
#endif
