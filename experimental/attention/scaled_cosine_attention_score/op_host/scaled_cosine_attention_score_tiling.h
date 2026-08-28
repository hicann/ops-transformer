/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_HOST_SCALED_COSINE_ATTENTION_SCORE_TILING_H_
#define OPS_OP_HOST_SCALED_COSINE_ATTENTION_SCORE_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint64_t SCAS_TILING_KEY_FP16 = 0;
constexpr uint64_t SCAS_TILING_KEY_BF16 = 1;
constexpr uint64_t SCAS_TILING_KEY_FP32 = 2;

BEGIN_TILING_DATA_DEF(ScaledCosineAttentionScoreTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, heads);
TILING_DATA_FIELD_DEF(uint32_t, seqLen);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, alignedHeadDim);
TILING_DATA_FIELD_DEF(uint32_t, keyTileRows);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
TILING_DATA_FIELD_DEF(uint64_t, totalQueryRows);
TILING_DATA_FIELD_DEF(float, clampMax);
TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScaledCosineAttentionScore, ScaledCosineAttentionScoreTilingData)

struct ScaledCosineAttentionScoreCompileInfo {
    int64_t aivCoreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling
#endif // OPS_OP_HOST_SCALED_COSINE_ATTENTION_SCORE_TILING_H_
