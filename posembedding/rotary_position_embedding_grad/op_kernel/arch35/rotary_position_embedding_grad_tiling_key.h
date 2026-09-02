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
 * \file rotary_position_embedding_grad_tiling_key.h
 * \brief rotary_position_embedding_grad key
 */

#ifndef __ROTARY_POSITION_EMBEDDING_GRAD_TILING_KEY_H__
#define __ROTARY_POSITION_EMBEDDING_GRAD_TILING_KEY_H__

#include "atvoss/reduce/reduce_tiling_key_decl.h"

#define ROPE_GRAD_BIT_WIDTH 8

ASCENDC_TPL_ARGS_DECL(RotaryPositionEmbeddingGrad, REDUCE_TPL_KEY_DECL(),
                      ASCENDC_TPL_UINT_DECL(DxTilingKey, ROPE_GRAD_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 201, 206),
                      ASCENDC_TPL_UINT_DECL(DcosFlag, 1, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    // Empty
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY(),
                         ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_RANGE, 1, 201, 206),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    // A
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_RANGE, 1, 201, 206),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    // ARA
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_NORMAL(),
                         ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 201, 202, 203, 204, 206),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_GROUP(),
                         ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 201, 202, 203, 204, 206),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    // ARARARARA  RARA模版，Reduce会自动补齐维到ARARARARA
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_NORMAL(),
                         ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_GROUP(),
                         ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)));

#endif
