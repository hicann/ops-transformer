/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_OP_SLIDING_TILE_ATTENTION_H_
#define OP_API_INC_LEVEL0_OP_SLIDING_TILE_ATTENTION_H_

#include "opdev/op_executor.h"

namespace l0op {

// Block-sparse sliding tile attention over 384-token (6x8x8) tiles. windows is an INT8 [1, N, 1, 3] tensor with the
// window of each head in tiles (T, H, W), already limited to the tile grid; tileGrid packs the image tiles as
// T | H << 16 | W << 32, and the image tokens are followed by at most one block of text slots. maxSelectNum is the
// largest window volume, plus one when textLength > 0. Key/value tokens at positions >= validKvSeqLen are ignored.
// When originalHeadDim is smaller than the tensors' head dim, query/key/value must be private zero-padded copies:
// the kernel rewrites their padding.
const aclTensor* SlidingTileAttention(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                      const aclTensor* windows, int64_t validKvSeqLen, double scaleValue,
                                      int64_t originalHeadDim, int64_t tileGrid, int64_t maxSelectNum,
                                      aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_OP_SLIDING_TILE_ATTENTION_H_
