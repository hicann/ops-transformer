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
 * \file moe_ep_dispatch_tiling_key.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_TILING_KEY_H
#define MOE_EP_DISPATCH_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace Mc2Tiling {

#define TILINGKEY_DIRECT 0
#define TILINGKEY_HYBRID 1

ASCENDC_TPL_ARGS_DECL(MoeEpDispatch, ASCENDC_TPL_BOOL_DECL(TILINGKEY_DO_CPU_SYNC, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_IS_CACHED, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_IS_TOPK_WEIGHTS, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_IS_MXQUANT, 0, 1),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_NETWORK_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_DIRECT, TILINGKEY_HYBRID),
                      ASCENDC_TPL_BOOL_DECL(TILINGKEY_CLOS_ONLY, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_DO_CPU_SYNC, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(TILINGKEY_IS_CACHED, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(TILINGKEY_IS_TOPK_WEIGHTS, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(TILINGKEY_IS_MXQUANT, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(TILINGKEY_NETWORK_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_DIRECT,
                                                          TILINGKEY_HYBRID),
                                     ASCENDC_TPL_BOOL_SEL(TILINGKEY_CLOS_ONLY, 0, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(MoeEpDispatchTilingData)), );

} // namespace Mc2Tiling
#endif // MOE_EP_DISPATCH_TILING_KEY_H
