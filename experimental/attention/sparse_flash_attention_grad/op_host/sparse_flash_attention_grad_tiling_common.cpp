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
 * \file sparse_flash_attention_grad_tiling_common.cpp
 * \brief
 */

#include "sparse_flash_attention_grad_tiling_common.h"
#include "err/ops_err.h"
#include "log/log.h"

namespace optiling {
namespace sfag {

ge::graphStatus CheckTndSoftmaxMaxShape(gert::TilingContext *context, int64_t t1, int64_t n2, int64_t g)
{
    auto softmaxMaxShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    if (softmaxMaxShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
    if (softmaxMaxShapeDim != 3) { // TND softmaxMax only support 3 dimensions
        OP_LOGE(context->GetNodeName(), "The shape of softmaxMax is invalid, got %lu dimensions", softmaxMaxShapeDim);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(0); // 0:n2
    auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(1); // 1:t1
    auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(2); // 2:g

    // softmaxMax shape: [N2, T1, G]
    OP_CHECK_IF((dim0 != n2 || dim1 != t1 || dim2 != g),
                OP_LOGE(context->GetNodeName(),
                        "The shape of softmaxMax is invalid, got (%ld,%ld,%ld), should be (%ld,%ld,%ld)", dim0, dim1,
                        dim2, n2, t1, g),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndSoftmaxSumShape(gert::TilingContext *context, int64_t t1, int64_t n2, int64_t g)
{
    auto softmaxSumShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    if (softmaxSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
    if (softmaxSumShapeDim != 3) { // TND softmaxSum only support 3 dimensions
        OP_LOGE(context->GetNodeName(), "The shape of softmaxSum is invalid, got %lu dimensions", softmaxSumShapeDim);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxSumShape->GetStorageShape().GetDim(0); // 0:n2
    auto dim1 = softmaxSumShape->GetStorageShape().GetDim(1); // 1:t1
    auto dim2 = softmaxSumShape->GetStorageShape().GetDim(2); // 2:g

    // softmaxMax shape: [N2, T1, G]
    OP_CHECK_IF((dim0 != n2 || dim1 != t1 || dim2 != g),
                OP_LOGE(context->GetNodeName(),
                        "The shape of softmaxSum is invalid, got (%ld,%ld,%ld), should be (%ld,%ld,%ld)", dim0, dim1,
                        dim2, n2, t1, g),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInShape(gert::TilingContext *context)
{
    auto attentionInShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::ATTENTION_OUT));
    if (attentionInShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    auto valueShape = context->GetInputShape(static_cast<size_t>(InputIndex::VALUE));
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    auto valueShapeDim = valueShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        OP_LOGE(context->GetNodeName(), "The dimnum of attentionIn %zu should be equal to query %zu",
                attentionInShapeDim, queryShapeDim);
        return ge::GRAPH_FAILED;
    }
    if (attentionInShapeDim != valueShapeDim) {
        OP_LOGE(context->GetNodeName(), "The dimnum of attentionIn %zu should be equal to value %zu",
                attentionInShapeDim, valueShapeDim);
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < queryShapeDim - 1; i++) {
        if (attentionInShape->GetStorageShape().GetDim(i) != queryShape->GetStorageShape().GetDim(i)) {
            OP_LOGE(context->GetNodeName(), "The dim %zu of attentionIn shape is invalid, got %ld, should be %ld", i,
                    attentionInShape->GetStorageShape().GetDim(i), queryShape->GetStorageShape().GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    // check head_dim, attention D == value D
    if (attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1) !=
        valueShape->GetStorageShape().GetDim(queryShapeDim - 1)) {
        OP_LOGE(context->GetNodeName(), "The head_dim of attentionIn shape is invalid, got %ld, should be %ld",
                attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1),
                valueShape->GetStorageShape().GetDim(queryShapeDim - 1));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxDtype(gert::TilingContext *context)
{
    auto softmaxMax = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    auto softmaxSum = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    OP_CHECK_IF(softmaxMax == nullptr || softmaxSum == nullptr,
                OP_LOGE(context->GetNodeName(), "softmax_max or softmax_sum is nullptr."), return ge::GRAPH_FAILED);

    auto softmaxMaxType = static_cast<uint32_t>(softmaxMax->GetDataType());
    auto softmaxSumType = static_cast<uint32_t>(softmaxSum->GetDataType());

    bool softmaxTypeCheck = (softmaxMaxType == softmaxSumType) && (softmaxMaxType == ge::DT_FLOAT);
    OP_CHECK_IF(softmaxTypeCheck != true,
                OP_LOGE(context->GetNodeName(), "softmaxMaxType should be DT_FLOAT and same with softmaxSumType"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInDtype(gert::TilingContext *context)
{
    auto query = context->GetInputDesc(static_cast<size_t>(InputIndex::QUERY));
    auto attentionIn = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::ATTENTION_OUT));
    OP_CHECK_IF(query == nullptr || attentionIn == nullptr,
                OP_LOGE(context->GetNodeName(), "query or attentionIn is nullptr."), return ge::GRAPH_FAILED);

    auto queryType = static_cast<uint32_t>(query->GetDataType());
    auto attentionInType = static_cast<uint32_t>(attentionIn->GetDataType());

    OP_CHECK_IF(queryType != attentionInType,
                OP_LOGE(context->GetNodeName(), "invalid attentionIn dtype should be same with query's dtype"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndShapeValid(gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d, int64_t d2,
                                   int64_t n2)
{
    if (context == nullptr) {
        OP_LOGE("SparseFlashAttentionGrad", "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto isShapeInValid = (t1 == 0 || n1 == 0 || d == 0 || d2 == 0 || n2 == 0);
    OP_CHECK_IF(isShapeInValid,
                OP_LOGE(context->GetNodeName(), "input shape error, got 0 in tnd(%ld,%ld,%ld) or tnd(%ld,%ld,%ld)", t1,
                        n1, d, t1, n1, d2),
                return ge::GRAPH_FAILED);

    auto g = n1 / n2;
    auto ret = CheckTndSoftmaxMaxShape(context, t1, n2, g);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckTndSoftmaxSumShape(context, t1, n2, g);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeValid(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("SparseFlashAttentionGrad", "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto ret = CheckSoftmaxDtype(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInDtype(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

bool IsSameShape(const gert::StorageShape *aShape, const gert::StorageShape *bShape)
{
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr),
                OP_LOGW("SparseFlashAttentionGrad", "aShape or bShape is nullptr."), return false);
    uint32_t dimSizeA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimSizeB = bShape->GetStorageShape().GetDimNum();
    if (dimSizeA != dimSizeB) {
        return false;
    }

    for (uint32_t i = 0; i < dimSizeA; i++) {
        auto dimA = aShape->GetStorageShape().GetDim(i);
        auto dimB = bShape->GetStorageShape().GetDim(i);
        if (dimA != dimB) {
            return false;
        }
    }
    return true;
}

} // namespace sfag
} // namespace optiling
