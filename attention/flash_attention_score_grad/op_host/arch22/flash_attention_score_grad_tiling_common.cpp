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
 * \file flash_attention_score_grad_tiling_common.cpp
 * \brief
 */

#include "flash_attention_score_grad_tiling_common.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {

ge::graphStatus CheckSoftmaxMaxShape(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1)
{
    auto softmaxMaxShape = context->GetOptionalInputShape(SOFTMAX_MAX);
    if (softmaxMaxShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
    if (softmaxMaxShapeDim != 4) { // softmaxMax only support 4 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIM("FlashAttentionScoreGrad", "softmaxMaxOptional",
                                     std::to_string(softmaxMaxShapeDim).c_str(), "4");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxMaxShape->GetStorageShape().GetDim(3); // 3:8

    // softmaxMax pad to 8
    OP_CHECK_IF(dim0 != b || dim1 != n1 || dim2 != s1 || dim3 != 8,
                OP_LOGE_FOR_INVALID_SHAPE("FlashAttentionScoreGrad", "softmaxMaxOptional",
                                          std::string("[") + std::to_string(dim0) + ", " + std::to_string(dim1) + ", " +
                                              std::to_string(dim2) + ", " + std::to_string(dim3) + "]",
                                          std::string("[") + std::to_string(b) + ", " + std::to_string(n1) + ", " +
                                              std::to_string(s1) + ", 8]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndSoftmaxMaxShape(gert::TilingContext *context, int64_t t1, int64_t n1)
{
    auto softmaxMaxShape = context->GetOptionalInputShape(SOFTMAX_MAX);
    if (softmaxMaxShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
    const char *tndSoftmaxIn = context->GetAttrs()->GetAttrNum() > static_cast<size_t>(TND_SOFTMAX_IN) ?
                                   context->GetAttrs()->GetAttrPointer<char>(TND_SOFTMAX_IN) :
                                   "";

    if (softmaxMaxShapeDim != 3) { // TND softmaxMax only support 3 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIM("FlashAttentionScoreGrad", "softmaxMaxOptional",
                                     std::to_string(softmaxMaxShapeDim).c_str(), "3");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(0); // 0:t1
    auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(2); // 2:8

    // softmaxMax pad to 8
    if (strcmp(tndSoftmaxIn, "same_as_input") == 0) {
        OP_CHECK_IF(
            dim0 != n1 || dim1 != t1 || dim2 != 8,
            OP_LOGE_FOR_INVALID_SHAPE("FlashAttentionScoreGrad", "softmaxMaxOptional",
                                      std::string("[") + std::to_string(dim0) + ", " + std::to_string(dim1) + ", " +
                                          std::to_string(dim2) + "]",
                                      std::string("[") + std::to_string(n1) + ", " + std::to_string(t1) + ", 8]"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            dim0 != t1 || dim1 != n1 || dim2 != 8,
            OP_LOGE_FOR_INVALID_SHAPE("FlashAttentionScoreGrad", "softmaxMaxOptional",
                                      std::string("[") + std::to_string(dim0) + ", " + std::to_string(dim1) + ", " +
                                          std::to_string(dim2) + "]",
                                      std::string("[") + std::to_string(t1) + ", " + std::to_string(n1) + ", 8]"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxSumShape(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1)
{
    auto softmaxSumShape = context->GetOptionalInputShape(SOFTMAX_SUM);
    if (softmaxSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
    if (softmaxSumShapeDim != 4) { // softmaxSum only support 4 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIM("FlashAttentionScoreGrad", "softmaxSumOptional",
                                     std::to_string(softmaxSumShapeDim).c_str(), "4");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxSumShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxSumShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxSumShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxSumShape->GetStorageShape().GetDim(3); // 3:8

    // softmaxSum pad to 8
    OP_CHECK_IF(dim0 != b || dim1 != n1 || dim2 != s1 || dim3 != 8,
                OP_LOGE_FOR_INVALID_SHAPE("FlashAttentionScoreGrad", "softmaxSumOptional",
                                          std::string("[") + std::to_string(dim0) + ", " + std::to_string(dim1) + ", " +
                                              std::to_string(dim2) + ", " + std::to_string(dim3) + "]",
                                          std::string("[") + std::to_string(b) + ", " + std::to_string(n1) + ", " +
                                              std::to_string(s1) + ", 8]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndSoftmaxSumShape(gert::TilingContext *context, int64_t t1, int64_t n1)
{
    auto softmaxSumShape = context->GetOptionalInputShape(SOFTMAX_SUM);
    if (softmaxSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
    const char *tndSoftmaxIn = context->GetAttrs()->GetAttrNum() > static_cast<size_t>(TND_SOFTMAX_IN) ?
                                   context->GetAttrs()->GetAttrPointer<char>(TND_SOFTMAX_IN) :
                                   "";

    if (softmaxSumShapeDim != 3) { // TND softmaxSum only support 3 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIM("FlashAttentionScoreGrad", "softmaxSumOptional",
                                     std::to_string(softmaxSumShapeDim).c_str(), "3");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxSumShape->GetStorageShape().GetDim(0); // 0:t1
    auto dim1 = softmaxSumShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxSumShape->GetStorageShape().GetDim(2); // 2:8

    // softmaxSum pad to 8
    if (strcmp(tndSoftmaxIn, "same_as_input") == 0) {
        OP_CHECK_IF(
            dim0 != n1 || dim1 != t1 || dim2 != 8,
            OP_LOGE_FOR_INVALID_SHAPE("FlashAttentionScoreGrad", "softmaxSumOptional",
                                      std::string("[") + std::to_string(dim0) + ", " + std::to_string(dim1) + ", " +
                                          std::to_string(dim2) + "]",
                                      std::string("[") + std::to_string(n1) + ", " + std::to_string(t1) + ", 8]"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            dim0 != t1 || dim1 != n1 || dim2 != 8,
            OP_LOGE_FOR_INVALID_SHAPE("FlashAttentionScoreGrad", "softmaxSumOptional",
                                      std::string("[") + std::to_string(dim0) + ", " + std::to_string(dim1) + ", " +
                                          std::to_string(dim2) + "]",
                                      std::string("[") + std::to_string(t1) + ", " + std::to_string(n1) + ", 8]"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInShape(gert::TilingContext *context)
{
    auto attentionInShape = context->GetOptionalInputShape(ATTENTION_IN);
    if (attentionInShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryShape = context->GetInputShape(QUERY);
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        std::string dimsMsg = "{" + std::to_string(attentionInShapeDim) + ", " + std::to_string(queryShapeDim) + "}";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            "FlashAttentionScoreGrad", "attentionInOptional, query", dimsMsg.c_str(),
            "The shape dimensions of attentionInOptional and query must be the same");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < queryShapeDim; i++) {
        // No matter TND, SBH, BSH, BSND, BNSD, the last dim could be different
        if (i == queryShapeDim - 1) {
            continue;
        } else if (attentionInShape->GetStorageShape().GetDim(i) != queryShape->GetStorageShape().GetDim(i)) {
            std::string shapesMsg = "{" + Ops::Base::ToString(attentionInShape->GetStorageShape()) + ", " +
                                    Ops::Base::ToString(queryShape->GetStorageShape()) + "}";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                "FlashAttentionScoreGrad", "attentionInOptional, query", shapesMsg.c_str(),
                "All axes except the last axis of attentionInOptional and query must be the same");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxDtype(gert::TilingContext *context)
{
    auto softmaxMax = context->GetOptionalInputDesc(SOFTMAX_MAX);
    auto softmaxSum = context->GetOptionalInputDesc(SOFTMAX_SUM);
    OP_CHECK_IF(softmaxMax == nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "softmaxMaxOptional"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxSum == nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "softmaxSumOptional"),
                return ge::GRAPH_FAILED);

    auto softmaxMaxType = static_cast<uint32_t>(softmaxMax->GetDataType());
    auto softmaxSumType = static_cast<uint32_t>(softmaxSum->GetDataType());

    bool softmaxTypeCheck = (softmaxMaxType == softmaxSumType) && (softmaxMaxType == ge::DT_FLOAT);
    OP_CHECK_IF(
        !softmaxTypeCheck,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "FlashAttentionScoreGrad", "softmaxMaxOptional, softmaxSumOptional",
            std::string("{") + ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(softmaxMaxType)) + ", " +
                ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(softmaxSumType)) + "}",
            "The dtypes of softmaxMaxOptional and softmaxSumOptional must both be FLOAT32"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInDtype(gert::TilingContext *context)
{
    auto query = context->GetInputDesc(QUERY);
    auto attentionIn = context->GetOptionalInputDesc(ATTENTION_IN);
    OP_CHECK_IF(query == nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attentionIn == nullptr, OP_LOGE_WITH_INVALID_INPUT("FlashAttentionScoreGrad", "attentionInOptional"),
                return ge::GRAPH_FAILED);

    auto queryType = static_cast<uint32_t>(query->GetDataType());
    auto attentionInType = static_cast<uint32_t>(attentionIn->GetDataType());

    OP_CHECK_IF(queryType != attentionInType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    "FlashAttentionScoreGrad", "query, attentionInOptional",
                    std::string("{") + ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(queryType)) +
                        ", " + ge::TypeUtils::DataTypeToSerialString(static_cast<ge::DataType>(attentionInType)) + "}",
                    "The dtypes of query and attentionInOptional must be the same"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeValid(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, int64_t d)
{
    auto isShapeInValid = (b == 0 || n1 == 0 || s1 == 0 || d == 0);
    OP_CHECK_IF(isShapeInValid,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("FlashAttentionScoreGrad", "query",
                                                      std::string("[") + std::to_string(b) + ", " + std::to_string(n1) +
                                                          ", " + std::to_string(s1) + ", " + std::to_string(d) + "]",
                                                      "All axes of query must be positive numbers"),
                return ge::GRAPH_FAILED);

    auto ret = CheckSoftmaxMaxShape(context, b, n1, s1);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckSoftmaxSumShape(context, b, n1, s1);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndShapeValid(gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d)
{
    if (context == nullptr) {
        OP_LOGE(context, "In op [FlashAttentionScoreGrad], context is null");
        return ge::GRAPH_FAILED;
    }

    auto isShapeInValid = (t1 == 0 || n1 == 0 || d == 0);
    OP_CHECK_IF(isShapeInValid,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    "FlashAttentionScoreGrad", "query",
                    std::string("[") + std::to_string(t1) + ", " + std::to_string(n1) + ", " + std::to_string(d) + "]",
                    "All axes of query must be positive numbers"),
                return ge::GRAPH_FAILED);

    auto ret = CheckTndSoftmaxMaxShape(context, t1, n1);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckTndSoftmaxSumShape(context, t1, n1);
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
        OP_LOGE(context, "In op [FlashAttentionScoreGrad], context is null");
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
                OP_LOGW("flash_attention_score_grad_tiling_common", "aShape or bShape is nullptr."), return false);
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

bool IsSameShapeButValueDLeEqD(const gert::StorageShape *aShape, const gert::StorageShape *bShape)
{
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr),
                OP_LOGW("flash_attention_score_grad_tiling_common", "aShape or bShape is nullptr."), return false);
    uint32_t dimSizeA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimSizeB = bShape->GetStorageShape().GetDimNum();
    if (dimSizeA != dimSizeB) {
        return false;
    }

    for (uint32_t i = 0; i < dimSizeA; i++) {
        if (i == dimSizeA - 1) {
            auto dimD = aShape->GetStorageShape().GetDim(i);
            auto dimValueD = bShape->GetStorageShape().GetDim(i);
            return dimValueD <= dimD;
        } else {
            auto dimA = aShape->GetStorageShape().GetDim(i);
            auto dimB = bShape->GetStorageShape().GetDim(i);
            if (dimA != dimB) {
                return false;
            }
        }
    }
    return true;
}

bool isTndSABHit(const gert::TilingContext *context)
{
    auto actualSeqQLenTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_LEN);
    auto actualSeqKVLenTensor = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_LEN);
    bool isTND = actualSeqQLenTensor != nullptr && actualSeqQLenTensor->GetShapeSize() > 0 &&
                 actualSeqKVLenTensor != nullptr && actualSeqKVLenTensor->GetShapeSize() > 0;

    if (isTND) {
        const int64_t *qTensor = actualSeqQLenTensor->GetData<int64_t>();
        const int64_t *kvTensor = actualSeqKVLenTensor->GetData<int64_t>();
        const size_t actualSeqQLen = static_cast<size_t>(actualSeqQLenTensor->GetShapeSize());

        uint64_t qSum = 0;
        uint64_t kvSum = 0;
        uint64_t len = 1;

        for (int64_t i = actualSeqQLen - 1; i >= 0; --i) {
            if (qTensor[i] != 0) {
                qSum = qTensor[i];
                kvSum = kvTensor[i];
                len = i + 1;
                break;
            }
        }

        if ((qSum / len >= SAB_TND_SIZE) && (kvSum / len >= SAB_TND_SIZE)) {
            return true;
        }
    }

    return false;
}

} // namespace optiling
