/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_apt_tiling_key.h
 * \brief A5 (arch35) tiling key selections.
 */
#ifndef MDISPATCH_APT_TILING_KEY_H
#define MDISPATCH_APT_TILING_KEY_H
#include "../moe_distribute_dispatch_tiling_key_decl.h"

namespace Mc2Tiling {
ASCENDC_TPL_SEL(
    // A5
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(TILINGKEY_QUANT_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_NO_QUANT),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SCALE, 0),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_FULLMESH, ASCENDC_TPL_UI_LIST, TILINGKEY_NO_FULLMESH),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_MTE),
                         ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MoeDistributeDispatchTilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(TILINGKEY_QUANT_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_PERTOKEN_QUANT),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SCALE, 0, 1),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_FULLMESH, ASCENDC_TPL_UI_LIST, TILINGKEY_NO_FULLMESH),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_MTE),
                         ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MoeDistributeDispatchTilingData)), );
} // namespace Mc2Tiling
#endif
