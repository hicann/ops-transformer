/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scaled_cosine_attention_score_tiling_def.h"
#include "scaled_cosine_attention_score_impl.hpp"

using namespace AscendC;

__aicore__ inline void CopyTilingData(optiling::ScaledCosineAttentionScoreTilingData &dst, GM_ADDR tiling)
{
    uint32_t *local = reinterpret_cast<uint32_t *>(&dst);
    auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0; i < sizeof(optiling::ScaledCosineAttentionScoreTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
}

extern "C" __global__ __aicore__ void scaled_cosine_attention_score(GM_ADDR query, GM_ADDR key, GM_ADDR scale,
                                                                    GM_ADDR attnScore, GM_ADDR workspace,
                                                                    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::ScaledCosineAttentionScoreTilingData);
    optiling::ScaledCosineAttentionScoreTilingData tilingDataIn;
    CopyTilingData(tilingDataIn, tiling);
    const auto *tilingData = &tilingDataIn;
    if (TILING_KEY_IS(0)) {
        NsScaledCosineAttentionScore::ScaledCosineAttentionScoreImpl<half> op;
        op.Init(query, key, scale, attnScore, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        NsScaledCosineAttentionScore::ScaledCosineAttentionScoreImpl<bfloat16_t> op;
        op.Init(query, key, scale, attnScore, tilingData);
        op.Process();
#endif
    } else if (TILING_KEY_IS(2)) {
        NsScaledCosineAttentionScore::ScaledCosineAttentionScoreImpl<float> op;
        op.Init(query, key, scale, attnScore, tilingData);
        op.Process();
    }
}
