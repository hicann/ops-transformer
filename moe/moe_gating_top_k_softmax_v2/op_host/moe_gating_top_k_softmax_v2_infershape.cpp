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
 * \file moe_gating_top_k_softmax_v2_infer.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace ops {

static const int64_t SIZE_2 = 2;
static const int64_t SIZE_3 = 3;
static const int INDEX_0 = 0;
static const int INDEX_1 = 1;
static const int INDEX_2 = 2;
static const int MAX_K = 1024;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
static constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;
static inline bool IsUnknownShape(const gert::Shape* check_shape) {
  for (size_t i = 0; i < check_shape->GetDimNum(); i++) {
    if (check_shape->GetDim(i) == UNKNOWN_DIM_VALUE_) {
      return true;
    }
  }
  return false;
}

static void setOutShape(
    const gert::Shape* gatingShape, gert::Shape* outShape, gert::Shape* indicesShape, gert::Shape* softmaxResultShape,
    int64_t k, bool softmaxFlag)
{
    const int64_t gatingDimNum = gatingShape->GetDimNum();
    outShape->SetDimNum(gatingDimNum);
    indicesShape->SetDimNum(gatingDimNum);
    for (int64_t i = 0; i < gatingDimNum - 1; i++) {
        outShape->SetDim(i, gatingShape->GetDim(i));
        indicesShape->SetDim(i, gatingShape->GetDim(i));
    }
    outShape->SetDim(gatingDimNum - 1, k);
    indicesShape->SetDim(gatingDimNum - 1, k);

    if (softmaxFlag) {
        softmaxResultShape->SetDimNum(gatingDimNum);
        for (int64_t i = 0; i < gatingDimNum; i++) {
            softmaxResultShape->SetDim(i, gatingShape->GetDim(i));
        }
    } else {
        softmaxResultShape->SetDimNum(1);
        softmaxResultShape->SetDim(0, 0);
    }
}

static ge::graphStatus InferShapeMoeGatingTopKSoftmaxV2(gert::InferShapeContext* context)
{
    const gert::Shape* gatingShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gatingShape);

    gert::Shape* outShape = context->GetOutputShape(0);
    gert::Shape* indicesShape = context->GetOutputShape(1);
    gert::Shape* softmaxResultShape = context->GetOutputShape(2);

    if (gatingShape->GetDimNum() == 1 && gatingShape->GetDim(0) == -2LL) {
        *outShape = *gatingShape;
        *indicesShape = *gatingShape;
        *softmaxResultShape = *gatingShape;
        return ge::GRAPH_SUCCESS;
    }

    const int64_t gatingDimNum = gatingShape->GetDimNum();
    if (gatingDimNum != SIZE_2 && gatingDimNum != SIZE_3) {
        std::string gatingDimNumStr = std::to_string(gatingDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", gatingDimNumStr.c_str(), "2 or 3");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kPtr);
    const int64_t k = *kPtr;

    if (k <= 0 || k > MAX_K || (!IsUnknownShape(gatingShape) && k > gatingShape->GetDim(gatingDimNum - 1))) {
        std::string kStr = std::to_string(k);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "k", kStr.c_str(),
            "k value error");
        return ge::GRAPH_FAILED;
    }

    const int64_t* renormPtr = attrs->GetAttrPointer<int64_t>(INDEX_1);
    int64_t renorm = renormPtr == nullptr ? 0 : *renormPtr;
    if (renorm < 0 || renorm > 1) {
        std::string renormStr = std::to_string(renorm);
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "renorm", renormStr.c_str(), "0 or 1");
        return ge::GRAPH_FAILED;
    }

    const bool* softmaxPtr = attrs->GetAttrPointer<bool>(INDEX_2);
    bool softmaxFlag = (renorm == 0) && (softmaxPtr && *softmaxPtr);

    setOutShape(gatingShape, outShape, indicesShape, softmaxResultShape, k, softmaxFlag);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeGatingTopKSoftmaxV2(gert::InferDataTypeContext* context)
{
    auto xDtype = context->GetInputDataType(INDEX_0);
    context->SetOutputDataType(INDEX_0, xDtype);
    context->SetOutputDataType(INDEX_1, ge::DT_INT32);
    context->SetOutputDataType(INDEX_2, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End MoeGatingTopKSoftmaxV2InferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeGatingTopKSoftmaxV2)
    .InferShape(InferShapeMoeGatingTopKSoftmaxV2)
    .InferDataType(InferDataType4MoeGatingTopKSoftmaxV2);
} // namespace ops
