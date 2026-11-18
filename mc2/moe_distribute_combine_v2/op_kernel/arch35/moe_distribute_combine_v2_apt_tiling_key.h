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
 * \file moe_distribute_combine_v2_apt_tiling_key.h
 * \brief A5 (arch35) tiling key selections. Shared macros/DECL live in the outer
 *        moe_distribute_combine_v2_tiling_key_decl.h.
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_APT_TILING_KEY_H
#define MOE_DISTRIBUTE_COMBINE_V2_APT_TILING_KEY_H
#include "../moe_distribute_combine_v2_tiling_key_decl.h"

namespace Mc2Tiling {
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(TILINGKEY_QUANT_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_NO_QUANT),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_LAYERED_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_CCU),
                         ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MoeDistributeCombineV2TilingData)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TILINGKEY_QUANT_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_NO_QUANT, TILINGKEY_INT8_QUANT,
                             TILINGKEY_MXFP8_E5M2_QUANT, TILINGKEY_MXFP8_E4M3_QUANT),
        ASCENDC_TPL_UINT_SEL(TILINGKEY_LAYERED_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_MTE, TILINGKEY_TPL_ROCE),
        ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
        ASCENDC_TPL_TILING_STRUCT_SEL(MoeDistributeCombineV2TilingData)), );
} // namespace Mc2Tiling
#endif
