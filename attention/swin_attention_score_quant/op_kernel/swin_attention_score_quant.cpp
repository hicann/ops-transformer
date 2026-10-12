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
 * \file swin_attention_score_quant.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "swin_attention_score_quant_int8.h"
#include "swin_attention_score_quant_int8_nomask.h"

#define INT8_NOMASK_MODE 0
#define INT8_MASK_MODE 1

extern "C" __global__ __aicore__ void swin_attention_score_quant(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR scale_quant, GM_ADDR scale_dequant1, GM_ADDR scale_dequant2,
    GM_ADDR bias_quant, GM_ADDR bias_dequant1, GM_ADDR bias_dequant2, GM_ADDR padding_mask1, GM_ADDR padding_mask2,
    GM_ADDR attention_score, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tPipe;
    AscendC::SetMaskNorm();
    AscendC::SetSysWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);

    __gm__ uint8_t* user = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(INT8_NOMASK_MODE)) {
        SwinAttentionScoreQuant<int8_t, half, uint64_t, int32_t, half, false> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkBmm, &tiling_data.qkBmmTilingData, op.pvBmm,
                          &tiling_data.pvBmmTilingData);
        op.Init(query, key, value, scale_quant, scale_dequant1, scale_dequant2, bias_quant, bias_dequant1,
                bias_dequant2, attention_score, user, &tiling_data, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(INT8_MASK_MODE)) {
        SwinAttentionScoreQuant<int8_t, half, uint64_t, int32_t, half, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.qkBmm, &tiling_data.qkBmmTilingData, op.pvBmm,
                          &tiling_data.pvBmmTilingData);
        op.Init(query, key, value, scale_quant, scale_dequant1, scale_dequant2, bias_quant, bias_dequant1,
                bias_dequant2, padding_mask1, attention_score, user, &tiling_data, &tPipe);
        op.Process();
    }
}
