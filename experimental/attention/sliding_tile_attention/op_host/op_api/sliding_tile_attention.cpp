/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "sliding_tile_attention.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SlidingTileAttention);

const aclTensor* SlidingTileAttention(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                      const aclTensor* windows, int64_t validKvSeqLen, double scaleValue,
                                      int64_t originalHeadDim, int64_t tileGrid, int64_t maxSelectNum,
                                      aclOpExecutor* executor)
{
    L0_DFX(SlidingTileAttention, query, key, value, windows, validKvSeqLen, scaleValue, originalHeadDim, tileGrid,
           maxSelectNum);

    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr");
        return nullptr;
    }
    if (query == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query must not be nullptr");
        return nullptr;
    }
    if (key == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "key must not be nullptr");
        return nullptr;
    }
    if (value == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "value must not be nullptr");
        return nullptr;
    }
    if (windows == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "windows must not be nullptr");
        return nullptr;
    }

    auto output = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "failed to allocate output tensor");
        return nullptr;
    }
    const float scale = static_cast<float>(scaleValue);
    auto ret = INFER_SHAPE(SlidingTileAttention, OP_INPUT(query, key, value, windows), OP_OUTPUT(output),
                           OP_ATTR(validKvSeqLen, scale, originalHeadDim, tileGrid, maxSelectNum));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SlidingTileAttention InferShape failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(SlidingTileAttention, OP_INPUT(query, key, value, windows), OP_OUTPUT(output),
                                OP_ATTR(validKvSeqLen, scale, originalHeadDim, tileGrid, maxSelectNum));
    return output;
}

} // namespace l0op
