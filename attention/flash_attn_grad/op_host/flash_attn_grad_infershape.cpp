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

#include <string>

using namespace ge;

namespace ops {
namespace {
constexpr size_t ATTR_LAYOUT_Q_INDEX = 6;

// dsink shape 是 [N1]，N1 是 Q 的 head 数。
//   BSND [B, S1, N1, D] -> dim 2
//   BNSD [B, N1, S1, D] -> dim 1
//   TND  [T1, N1, D]    -> dim 1
int64_t InferHeadNumQ(gert::InferShapeContext* context, const gert::Shape& qShape)
{
    const auto attrs = context->GetAttrs();
    const char* layoutQ = nullptr;
    if (attrs != nullptr) {
        const auto layoutQPtr = attrs->GetAttrPointer<char>(ATTR_LAYOUT_Q_INDEX);
        if (layoutQPtr != nullptr) {
            layoutQ = layoutQPtr;
        }
    }
    if (layoutQ != nullptr && std::string(layoutQ) == "BNSD") {
        return qShape.GetDim(1);
    }
    return (qShape.GetDimNum() == 3U) ? qShape.GetDim(1) : qShape.GetDim(2);
}
} // namespace

static ge::graphStatus InferShapeFlashAttnGrad(gert::InferShapeContext* context)
{
    auto qShape = context->GetInputShape(0);
    auto kShape = context->GetInputShape(1);
    auto vShape = context->GetInputShape(2);

    auto dqShape = context->GetOutputShape(0);
    auto dkShape = context->GetOutputShape(1);
    auto dvShape = context->GetOutputShape(2);
    auto dsinkShape = context->GetOutputShape(3);

    *dqShape = *qShape;
    *dkShape = *kShape;
    *dvShape = *vShape;
    dsinkShape->SetDimNum(1U);
    dsinkShape->SetDim(0, InferHeadNumQ(context, *qShape));
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeFlashAttnGrad(gert::InferDataTypeContext* context)
{
    auto qDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, qDtype);
    context->SetOutputDataType(1, qDtype);
    context->SetOutputDataType(2, qDtype);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttnGrad).InferShape(InferShapeFlashAttnGrad).InferDataType(InferDtypeFlashAttnGrad);
} // namespace ops
