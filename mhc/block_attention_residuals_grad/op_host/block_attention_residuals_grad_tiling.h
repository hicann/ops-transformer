/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _BLOCK_ATTENTION_RESIDUALS_GRAD_TILING_DATA_H_
#define _BLOCK_ATTENTION_RESIDUALS_GRAD_TILING_DATA_H_
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(BlockAttentionResidualsGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(int64_t, totalBlocks);
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
TILING_DATA_FIELD_DEF(int64_t, hiddenTileSize);
TILING_DATA_FIELD_DEF(int64_t, hiddenTileNum);
TILING_DATA_FIELD_DEF(int64_t, coreBatchStart);
TILING_DATA_FIELD_DEF(int64_t, coreBatchEnd);
TILING_DATA_FIELD_DEF(int64_t, coreBatchCount);
TILING_DATA_FIELD_DEF(uint64_t, gradScoreWeightWkspOff);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(uint64_t, perCoreWkspBytes);
TILING_DATA_FIELD_DEF(uint64_t, gradScoresWkspOff);
TILING_DATA_FIELD_DEF(uint64_t, varianceScaleWkspOff);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BlockAttentionResidualsGrad, BlockAttentionResidualsGradTilingData);
} // namespace optiling
struct BlockAttentionResidualsGradCompileInfo {
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
};
#endif
