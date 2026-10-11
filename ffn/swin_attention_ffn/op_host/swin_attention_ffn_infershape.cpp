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
 * \file swin_attention_ffn.cc
 * \brief
 */
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeSwinAttentionFFN(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter SwinAttentionFFN infershape impl.");

    const gert::Shape* x1_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);

    gert::Shape* y_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);

    *y_shape = *x1_shape;
    OP_LOGI(context->GetNodeName(), "SwinAttentionFFN infershape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSwinAttentionFFN(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter SwinAttentionFFN inferDataType impl.");

    const ge::DataType x1_data_type = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, x1_data_type);

    OP_LOGI(context->GetNodeName(), "SwinAttentionFFN inferDataType end.");
    return ret;
}

IMPL_OP_INFERSHAPE(SwinAttentionFFN)
    .InferShape(InferShapeSwinAttentionFFN)
    .InferDataType(InferDataTypeSwinAttentionFFN);
} // namespace ops
