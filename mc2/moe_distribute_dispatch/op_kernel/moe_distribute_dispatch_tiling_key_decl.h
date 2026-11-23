/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* ! *_decl.h */
#ifndef M_DISPATCH_TILING_KEY_DECL_H
#define M_DISPATCH_TILING_KEY_DECL_H
#include "ascendc/host_api/tiling/template_argument.h"

namespace Mc2Tiling {
#define TILINGKEY_TPL_A2 0
#define TILINGKEY_TPL_A3 1
#define TILINGKEY_TPL_A5 2
#define TILINGKEY_NO_QUANT 0
#define TILINGKEY_STATIC_QUANT 1
#define TILINGKEY_PERTOKEN_QUANT 2
#define TILINGKEY_NO_FULLMESH 0
#define TILINGKEY_ENABLE_FULLMESH 1
#define TILINGKEY_TPL_MTE 0
#define TILINGKEY_TPL_AICPU 1
ASCENDC_TPL_ARGS_DECL(MoeDistributeDispatch,
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_NO_QUANT, TILINGKEY_STATIC_QUANT, TILINGKEY_PERTOKEN_QUANT),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_SCALE, 0, 1),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_FULLMESH, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_NO_FULLMESH, TILINGKEY_ENABLE_FULLMESH),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_COMM_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_TPL_MTE, TILINGKEY_TPL_AICPU),
                      ASCENDC_TPL_UINT_DECL(ARCH_TAG, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A2,
                                            TILINGKEY_TPL_A3, TILINGKEY_TPL_A5), );
} // namespace Mc2Tiling
#endif
