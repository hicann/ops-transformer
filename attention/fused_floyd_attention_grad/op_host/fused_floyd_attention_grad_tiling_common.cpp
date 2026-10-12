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
 * \file fused_floyd_attention_grad_tiling_common.cpp
 * \brief
 */

#include "fused_floyd_attention_grad_tiling_common.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {
namespace FFAG {
bool CheckSameShape(const gert::StorageShape* aShape, const gert::StorageShape* bShape)
{
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr),
                OP_LOGW("fused_floyd_attention_grad_tiling_common", "aShape or bShape is nullptr."), return false);
    uint32_t dimNumA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimNumB = bShape->GetStorageShape().GetDimNum();
    if (dimNumA != dimNumB || dimNumA != SUPPORT_DIM_NUM) {
        return false;
    }

    for (uint32_t i = 0; i < dimNumA; i++) {
        auto dimA = aShape->GetStorageShape().GetDim(i);
        auto dimB = bShape->GetStorageShape().GetDim(i);
        if (dimA != dimB) {
            return false;
        }
    }
    return true;
}

ge::graphStatus CheckSupportShape(gert::TilingContext* context)
{
    const gert::StorageShape* queryShape = context->GetInputShape(QUERY);
    const gert::StorageShape* key1Shape = context->GetInputShape(KEY_1);
    const gert::StorageShape* value1Shape = context->GetInputShape(VALUE_1);
    const gert::StorageShape* key2Shape = context->GetInputShape(KEY_2);
    const gert::StorageShape* value2Shape = context->GetInputShape(VALUE_2);
    const gert::StorageShape* dyShape = context->GetInputShape(DY);
    const gert::StorageShape* attentionInShape = context->GetInputShape(ATTENTION_IN);

    OP_CHECK_IF(!CheckSameShape(key1Shape, value1Shape),
                OP_LOGE(context, "key1's shape and value1's shape is different"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckSameShape(key2Shape, value2Shape),
                OP_LOGE(context, "key2's shape and value2's shape is different"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckSameShape(queryShape, dyShape), OP_LOGE(context, "query's shape and dy's shape is different"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckSameShape(queryShape, attentionInShape),
                OP_LOGE(context, "query's shape and attentionIn's shape is different"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionMaskShape(gert::TilingContext* context, int64_t b, int64_t s1, int64_t s3)
{
    auto attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK);
    if (attentionMaskShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto attentionMaskShapeDim = attentionMaskShape->GetStorageShape().GetDimNum();
    if (attentionMaskShapeDim != SUPPORT_DIM_NUM) { // attentionMask only support 5 dimensions
        OP_LOGE(context, "The shape of atten_mask is invalid, got %lu dimensions", attentionMaskShapeDim);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = attentionMaskShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = attentionMaskShape->GetStorageShape().GetDim(1); // 1:1
    auto dim2 = attentionMaskShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = attentionMaskShape->GetStorageShape().GetDim(3); // 3:1
    auto dim4 = attentionMaskShape->GetStorageShape().GetDim(4); // 4:s3

    OP_CHECK_IF(
        (dim0 != b || dim1 != 1 || dim2 != s1 || dim3 != 1 || dim4 != s3),
        OP_LOGE(context, "The shape of atten_mask is invalid, got (%ld,%ld,%ld,%ld,%ld), should be (%ld,1,%ld,1,%ld)",
                dim0, dim1, dim2, dim3, dim4, b, s1, s3),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxMaxAndSumShape(gert::TilingContext* context, int64_t b, int64_t n, int64_t s1, int64_t s2,
                                           uint8_t inputIdx)
{
    auto softmaxMaxAndSumShape = context->GetOptionalInputShape(inputIdx);
    if (softmaxMaxAndSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxAndSumShapeDim = softmaxMaxAndSumShape->GetStorageShape().GetDimNum();
    if (softmaxMaxAndSumShapeDim != SUPPORT_DIM_NUM) { // softmaxMax/softmaxSum only support 5 dimensions
        OP_LOGE(context, "The shape of softmaxMax/softmaxSum is invalid, got %lu dimensions", softmaxMaxAndSumShapeDim);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxAndSumShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxMaxAndSumShape->GetStorageShape().GetDim(1); // 1:n
    auto dim2 = softmaxMaxAndSumShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxMaxAndSumShape->GetStorageShape().GetDim(3); // 3:s2
    auto dim4 = softmaxMaxAndSumShape->GetStorageShape().GetDim(4); // 4:8

    // softmaxMaxAndSum pad to 8
    OP_CHECK_IF(
        (dim0 != b || dim1 != n || dim2 != s1 || dim3 != s2 || dim4 != 8),
        OP_LOGE(
            context,
            "The shape of softmaxMax/softmaxSum is invalid, got (%ld,%ld,%ld,%ld,%ld), should be (%ld,%ld,%ld,%ld,8)",
            dim0, dim1, dim2, dim3, dim4, b, n, s1, s2),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputShapeValid(gert::TilingContext* context, int64_t b, int64_t n, int64_t s1, int64_t s2,
                                     int64_t s3, int64_t d)
{
    OP_CHECK_IF((b == 0 || n == 0 || s1 == 0 || s2 == 0 || s3 == 0 || d == 0),
                OP_LOGE(context, "input shape error, got 0 in bhmnkd(%ld,%ld,%ld,%ld,%ld,%ld)", b, n, s1, s2, s3, d),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((s1 % ALIGNED_NUM_16 != 0 || s2 % ALIGNED_NUM_128 != 0),
                OP_LOGE(context, "The third dimension of the query only supports 16-byte alignment, and the fourth "
                                 "dimension only supports 128-byte alignment"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(s3 % ALIGNED_NUM_128 != 0,
                OP_LOGE(context, "The fourth dimension of the key1 only supports 128-byte alignment"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((d != 32 && d != 64 && d != 128), OP_LOGE(context, "Headdim only supports 32/64/128."),
                return ge::GRAPH_FAILED);

    auto ret = CheckSupportShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionMaskShape(context, b, s1, s3);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckSoftmaxMaxAndSumShape(context, b, n, s1, s2, SOFTMAX_SUM);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckSoftmaxMaxAndSumShape(context, b, n, s1, s2, SOFTMAX_MAX);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace FFAG
} // namespace optiling
