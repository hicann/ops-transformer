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
 * \file apply_rotary_pos_emb_grad_tiling_key.h
 * \brief
 */

#ifndef __APPLY_ROTARY_POS_EMB_GRAD_TILING_KEY_H__
#define __APPLY_ROTARY_POS_EMB_GRAD_TILING_KEY_H__

#include "atvoss/reduce/reduce_tiling_key_decl.h"

#define APPLY_ROPE_GRAD_DX_BIT_WIDTH 8

ASCENDC_TPL_ARGS_DECL(ApplyRotaryPosEmbGrad, REDUCE_TPL_KEY_DECL(),
                      ASCENDC_TPL_UINT_DECL(DxTilingKey, APPLY_ROPE_GRAD_DX_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 203,
                                            205),
                      ASCENDC_TPL_UINT_DECL(DcosFlag, 1, ASCENDC_TPL_UI_LIST, 0, 1));

ASCENDC_TPL_SEL(
    // ================================================================
    // EMPTY — DcosFlag=0 for all templates; DcosFlag=1 only for A (205) where no broadcast
    // ================================================================
    // BAB (203): BSND layout, cos 1S1D
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0)),
    // AB (204): BSND+cosB>1 or SBND layout
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0)),
    // A (205): NO_BROADCAST, DcosFlag={0,1}
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 205),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1)),

    // ================================================================
    // A reduce — DcosFlag=1, BAB+AB only
    // ================================================================
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1)),

    // ================================================================
    // ARA_NORMAL / ARA_GROUP — DcosFlag=1, BAB+AB only
    // ================================================================
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_NORMAL(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_NORMAL(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1)),

    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_GROUP(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_GROUP(), ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1)),

    // ================================================================
    // ARARARARA — only BSND+BAB, DcosFlag=1
    // ================================================================
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_NORMAL(),
                         ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_GROUP(),
                         ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
                         ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1)));

#endif
