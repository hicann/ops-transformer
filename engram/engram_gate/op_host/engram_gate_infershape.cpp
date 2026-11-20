/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_gate_infershape.cpp
 * \brief EngramGate shape/dtype 推导
 */
#include "util/shape_util.h"
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {

namespace {
constexpr size_t ATTR_SAVE_FOR_BACKWARD = 2;
constexpr int64_t EMPTY_DIM = 0;
} // namespace

graphStatus InferShape4EngramGate(gert::InferShapeContext* context)
{
    // output 与 hidden_states 同 shape；save_for_backward=true 时中间结果为 [num_tokens, hc_mult]，
    // 否则中间结果为空 Tensor（shape 为 [0]），由 kernel 跳过写出。
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    gert::Shape* dotShape = context->GetOutputShape(1);
    gert::Shape* gateShape = context->GetOutputShape(2);
    gert::Shape* rstdXShape = context->GetOutputShape(3);
    gert::Shape* rstdKShape = context->GetOutputShape(4);
    if (xShape == nullptr || outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const bool unknownRank = Ops::Base::IsUnknownRank(*xShape);
    if (!unknownRank && xShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    const bool* saveForBackwardAttr = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(ATTR_SAVE_FOR_BACKWARD);
    const bool saveForBackward = saveForBackwardAttr == nullptr ? true : *saveForBackwardAttr;

    // 中间结果输出在 OpDef 中为 OPTIONAL：save_for_backward=false 时允许为空。
    if (saveForBackward &&
        (dotShape == nullptr || gateShape == nullptr || rstdXShape == nullptr || rstdKShape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *xShape;
    if (saveForBackward) {
        *dotShape = unknownRank ? gert::Shape{-1, -1} : gert::Shape{xShape->GetDim(0), xShape->GetDim(1)};
        *gateShape = *dotShape;
        *rstdXShape = *dotShape;
        *rstdKShape = *dotShape;
    } else {
        if (dotShape != nullptr) {
            dotShape->SetDimNum(1);
            dotShape->SetDim(0, EMPTY_DIM);
        }
        if (gateShape != nullptr) {
            gateShape->SetDimNum(1);
            gateShape->SetDim(0, EMPTY_DIM);
        }
        if (rstdXShape != nullptr) {
            rstdXShape->SetDimNum(1);
            rstdXShape->SetDim(0, EMPTY_DIM);
        }
        if (rstdKShape != nullptr) {
            rstdKShape->SetDimNum(1);
            rstdKShape->SetDim(0, EMPTY_DIM);
        }
    }
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4EngramGate(gert::InferDataTypeContext* context)
{
    // 主输出沿用输入类型，中间结果固定为 FP32。
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    context->SetOutputDataType(4, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramGate).InferShape(InferShape4EngramGate).InferDataType(InferDtype4EngramGate);
} // namespace ops
