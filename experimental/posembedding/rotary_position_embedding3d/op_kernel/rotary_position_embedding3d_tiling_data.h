/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTARY_POSITION_EMBEDDING3D_TILING_DATA_H
#define ROTARY_POSITION_EMBEDDING3D_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

struct RotaryPositionEmbedding3dTilingData {
    int64_t totalLength;
    int64_t headDim;
    int64_t seqLen;
    int64_t T;     // video frames
    int64_t H;     // video height tokens
    int64_t W;     // video width tokens
    int64_t tBand; // full temporal band dim (d0)
    int64_t hBand; // full height band dim (d1)
    int64_t wBand; // full width band dim (d2)
    int64_t tileNum;
    int64_t tileLength;
    int64_t blockLength;
    float freqBase;
    float rT; // freqBase^(-2/tBand)
    float rH; // freqBase^(-2/hBand)
    float rW; // freqBase^(-2/wBand)
};
#endif
