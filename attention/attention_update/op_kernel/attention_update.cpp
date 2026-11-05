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
 * \file attention_update.cpp
 * \brief
 */
#include "decode_update.h"

#define TILING_KEY_GO_FP32_WITHOUT_MAX_OUT 10010
#define TILING_KEY_GO_FP16_WITHOUT_MAX_OUT 10020
#define TILING_KEY_GO_BF16_WITHOUT_MAX_OUT 10030
#define TILING_KEY_GO_FP32_ROW_STRIDED 11010
#define TILING_KEY_GO_FP16_ROW_STRIDED 11020
#define TILING_KEY_GO_BF16_ROW_STRIDED 11030

extern "C" __global__ __aicore__ void attention_update(GM_ADDR les, GM_ADDR in, GM_ADDR out, GM_ADDR lseOut,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILING_KEY_GO_FP32_ROW_STRIDED)) {
        GET_TILING_DATA_WITH_STRUCT(DecodeUpdateStridedTilingData, data, tiling);
        DecodeUpdateTilingData baseData;
        baseData.formerLength = data.formerLength;
        baseData.tailLength = data.tailLength;
        baseData.formerNum = data.formerNum;
        baseData.tailNum = data.tailNum;
        baseData.hDim = data.hDim;
        baseData.updateType = data.updateType;
        baseData.sp = data.sp;
        baseData.totalLength = data.totalLength;

        AttentionUpdate::DecodeUpdate<float, float, true> op;
        op.Init(les, in, out, lseOut, &baseData, data.rowStride);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GO_FP16_ROW_STRIDED)) {
        GET_TILING_DATA_WITH_STRUCT(DecodeUpdateStridedTilingData, data, tiling);
        DecodeUpdateTilingData baseData;
        baseData.formerLength = data.formerLength;
        baseData.tailLength = data.tailLength;
        baseData.formerNum = data.formerNum;
        baseData.tailNum = data.tailNum;
        baseData.hDim = data.hDim;
        baseData.updateType = data.updateType;
        baseData.sp = data.sp;
        baseData.totalLength = data.totalLength;

        AttentionUpdate::DecodeUpdate<float, half, true> op;
        op.Init(les, in, out, lseOut, &baseData, data.rowStride);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GO_BF16_ROW_STRIDED)) {
        GET_TILING_DATA_WITH_STRUCT(DecodeUpdateStridedTilingData, data, tiling);
        DecodeUpdateTilingData baseData;
        baseData.formerLength = data.formerLength;
        baseData.tailLength = data.tailLength;
        baseData.formerNum = data.formerNum;
        baseData.tailNum = data.tailNum;
        baseData.hDim = data.hDim;
        baseData.updateType = data.updateType;
        baseData.sp = data.sp;
        baseData.totalLength = data.totalLength;

        AttentionUpdate::DecodeUpdate<float, bfloat16_t, true> op;
        op.Init(les, in, out, lseOut, &baseData, data.rowStride);
        op.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(DecodeUpdateTilingData, tilingDataIn, tiling);
        const DecodeUpdateTilingData* __restrict__ tilingData = &tilingDataIn;
        if (TILING_KEY_IS(TILING_KEY_GO_FP32_WITHOUT_MAX_OUT)) {
            AttentionUpdate::DecodeUpdate<float, float> op;
            op.Init(les, in, out, lseOut, tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_GO_FP16_WITHOUT_MAX_OUT)) {
            AttentionUpdate::DecodeUpdate<float, half> op;
            op.Init(les, in, out, lseOut, tilingData);
            op.Process();
        } else if (TILING_KEY_IS(TILING_KEY_GO_BF16_WITHOUT_MAX_OUT)) {
            AttentionUpdate::DecodeUpdate<float, bfloat16_t> op;
            op.Init(les, in, out, lseOut, tilingData);
            op.Process();
        }
    }
}
