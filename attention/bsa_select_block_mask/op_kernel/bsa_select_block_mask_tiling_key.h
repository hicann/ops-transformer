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
 * \file bsa_select_block_mask_tiling_key.h
 * \brief
 */
#ifndef __BSA_SELECT_BLOCK_MASK_TILING_KEY_H__
#define __BSA_SELECT_BLOCK_MASK_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define ASCENDC_TPL_4_BW 4

ASCENDC_TPL_ARGS_DECL(BSASelectBlockMask, ASCENDC_TPL_UINT_DECL(LAYOUT_Q, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1),
                      ASCENDC_TPL_UINT_DECL(LAYOUT_KV, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(LAYOUT_Q, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(LAYOUT_KV, ASCENDC_TPL_UI_LIST, 0, 1),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(optiling::BSASelectBlockMaskTilingData)));

#endif
