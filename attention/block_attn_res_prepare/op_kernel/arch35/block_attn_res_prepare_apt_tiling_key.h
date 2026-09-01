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
 * \file block_attn_res_prepare_apt_tiling_key.h
 * \brief BlockAttnResPrepare tiling key definition for Ascend 950.
 */
#ifndef BLOCK_ATTN_RES_PREPARE_APT_TILING_KEY_H
#define BLOCK_ATTN_RES_PREPARE_APT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"
#include "block_attn_res_prepare_tiling_data.h"

#define BLOCK_ATTN_RES_PREPARE_TPL_ONLY_VECTOR 0UL
#define BLOCK_ATTN_RES_PREPARE_TPL_MIX 1UL
#define BLOCK_ATTN_RES_PREPARE_TEMPLATE_MODE_BIT_WIDTH 1U

namespace BlockAttnResPrepareTilingKey {

static_assert(BLOCK_ATTN_RES_PREPARE_TPL_ONLY_VECTOR == optiling::BLOCK_ATTN_RES_PREPARE_VECTOR_KEY);
static_assert(BLOCK_ATTN_RES_PREPARE_TPL_MIX == optiling::BLOCK_ATTN_RES_PREPARE_MIX_KEY);

ASCENDC_TPL_ARGS_DECL(BlockAttnResPrepare,
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, BLOCK_ATTN_RES_PREPARE_TEMPLATE_MODE_BIT_WIDTH,
                                            ASCENDC_TPL_UI_LIST, BLOCK_ATTN_RES_PREPARE_TPL_ONLY_VECTOR,
                                            BLOCK_ATTN_RES_PREPARE_TPL_MIX));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST,
                                                          BLOCK_ATTN_RES_PREPARE_TPL_ONLY_VECTOR)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                                     ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST,
                                                          BLOCK_ATTN_RES_PREPARE_TPL_MIX)));

} // namespace BlockAttnResPrepareTilingKey

#endif // BLOCK_ATTN_RES_PREPARE_APT_TILING_KEY_H
