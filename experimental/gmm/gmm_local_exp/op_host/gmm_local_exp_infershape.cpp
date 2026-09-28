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
namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    const gert::Shape *weight_shape = context->GetInputShape(1);
    // check dim
    if (x_shape->GetDimNum() != 2 || weight_shape->GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "Dim of inputs not right");
        return ge::GRAPH_FAILED;
    }

    bool trans_b = *(context->GetAttrs()->GetAttrPointer<bool>(0));

    auto M = x_shape->GetDim(0);
    auto N = trans_b ? weight_shape->GetDim(1) : weight_shape->GetDim(2);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = gert::Shape({M, N});
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // auto dtype = context->GetInputDataType(0);
    // context->SetOutputDataType(0, dtype);
    // Get input and output data types from the graph
    auto input_dtype = context->GetInputDataType(0);
    auto output_dtype = context->GetOutputDataType(0);

    // If output dtype is not explicitly set, use input dtype
    if (output_dtype == ge::DT_UNDEFINED) {
        output_dtype = input_dtype;
    }

    context->SetOutputDataType(0, output_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GmmLocalExp).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ge
