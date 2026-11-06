/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace {
constexpr int64_t HC_ACTIVATION_RANK = 3;
constexpr int64_t HC_WEIGHT_RANK = 2;
constexpr int64_t HC_FEATURE_DIM = 2;
constexpr int64_t HC_OUTPUT_UP = 4;
constexpr int64_t HC_OUTPUT_DOWN = 3;
constexpr int64_t HC_OUTPUT_INV = 2;
constexpr int64_t HC_OUTPUT_COUNT = 5;
} // namespace

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4TextGatedResidual(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* channelsPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, channelsPtr);

    gert::Shape* mixedShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, mixedShape);
    if (xShape->GetDimNum() != HC_ACTIVATION_RANK || *channelsPtr <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t d = xShape->GetDim(HC_FEATURE_DIM);
    if (d % *channelsPtr != 0) {
        return ge::GRAPH_FAILED;
    }
    const auto* wd = context->GetInputShape(1);
    const auto* wu = context->GetInputShape(2);
    if (d <= 0 || wd == nullptr || wu == nullptr || wd->GetDimNum() != HC_WEIGHT_RANK ||
        wu->GetDimNum() != HC_WEIGHT_RANK || wd->GetDim(0) <= 0 || wd->GetDim(1) != d || wu->GetDim(0) != d ||
        wu->GetDim(1) != wd->GetDim(0))
        return ge::GRAPH_FAILED;
    const int64_t hidden = d / *channelsPtr;
    mixedShape->SetDimNum(HC_ACTIVATION_RANK);
    mixedShape->SetDim(0, xShape->GetDim(0));
    mixedShape->SetDim(1, xShape->GetDim(1));
    mixedShape->SetDim(HC_FEATURE_DIM, hidden);

    const bool* needBackwardPtr = attrs->GetAttrPointer<bool>(2);
    const bool needBackward = needBackwardPtr != nullptr && *needBackwardPtr;
    const bool hasGate = context->GetOptionalInputShape(3) != nullptr;
    gert::Shape* injectShape = context->GetOutputShape(1);
    if (injectShape != nullptr) {
        injectShape->SetDimNum(HC_ACTIVATION_RANK);
        injectShape->SetDim(0, xShape->GetDim(0));
        injectShape->SetDim(1, xShape->GetDim(1));
        injectShape->SetDim(HC_FEATURE_DIM, *channelsPtr);
    }
    gert::Shape* invRmsShape = context->GetOutputShape(HC_OUTPUT_INV);
    if (invRmsShape != nullptr) {
        invRmsShape->SetDimNum(HC_ACTIVATION_RANK);
        invRmsShape->SetDim(0, xShape->GetDim(0));
        invRmsShape->SetDim(1, xShape->GetDim(1));
        invRmsShape->SetDim(HC_FEATURE_DIM, *channelsPtr);
    }
    const gert::Shape* wDownShape = context->GetInputShape(1);
    gert::Shape* downPreActShape = context->GetOutputShape(HC_OUTPUT_DOWN);
    if (downPreActShape != nullptr && wDownShape != nullptr && wDownShape->GetDimNum() >= 1) {
        downPreActShape->SetDimNum(HC_ACTIVATION_RANK);
        downPreActShape->SetDim(0, xShape->GetDim(0));
        downPreActShape->SetDim(1, xShape->GetDim(1));
        downPreActShape->SetDim(HC_FEATURE_DIM, wDownShape->GetDim(0));
    }
    auto* upPreActShape = context->GetOutputShape(HC_OUTPUT_UP);
    if (upPreActShape != nullptr) {
        *upPreActShape = *xShape;
    }
    for (size_t index = 1; index < HC_OUTPUT_COUNT; ++index) {
        if ((index == 1 && !hasGate) || (index >= HC_OUTPUT_INV && !needBackward)) {
            auto* disabledShape = context->GetOutputShape(index);
            if (disabledShape != nullptr) {
                disabledShape->SetDimNum(1);
                disabledShape->SetDim(0, 0);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4TextGatedResidual(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    context->SetOutputDataType(HC_OUTPUT_INV, ge::DT_FLOAT);
    context->SetOutputDataType(HC_OUTPUT_DOWN, context->GetInputDataType(0));
    context->SetOutputDataType(HC_OUTPUT_UP, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TextGatedResidual)
    .InferShape(InferShape4TextGatedResidual)
    .InferDataType(InferDataType4TextGatedResidual);
} // namespace ops
