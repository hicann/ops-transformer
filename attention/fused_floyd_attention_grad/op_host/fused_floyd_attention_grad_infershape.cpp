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
 * \file fused_floyd_attention_grad_infershape.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

static const uint64_t DIM_NUM_2 = 2;

ge::graphStatus InferShape4FusedFloydAttentionGrad(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGI(context, "Enter FusedFloydAttentionGrad runtime infershape impl.");
    const gert::Shape* queryShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape* key1Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, key1Shape);
    const gert::Shape* value1Shape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, value1Shape);
    const gert::Shape* key2Shape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, key2Shape);
    const gert::Shape* value2Shape = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, value2Shape);

    gert::Shape* dqShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dqShape);
    *dqShape = *queryShape;

    gert::Shape* dk1Shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dk1Shape);
    *dk1Shape = *key1Shape;

    gert::Shape* dv1Shape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, dv1Shape);
    *dv1Shape = *value1Shape;

    gert::Shape* dk2Shape = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, dk2Shape);
    *dk2Shape = *key2Shape;

    gert::Shape* dv2Shape = context->GetOutputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, dv2Shape);
    *dv2Shape = *value2Shape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4FusedFloydAttentionGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGI(context, "Enter FusedFloydAttentionGrad infer data type impl.");

    auto dtype = context->GetInputDataType(0);
    // dq, outidx:0
    context->SetOutputDataType(0, dtype);
    // dk1, outidx:1
    context->SetOutputDataType(1, dtype);
    // dv1, outidx:2
    context->SetOutputDataType(2, dtype);
    // dk2, outidx:3
    context->SetOutputDataType(3, dtype);
    // dv2, outidx:4
    context->SetOutputDataType(4, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP(FusedFloydAttentionGrad)
    .InferShape(InferShape4FusedFloydAttentionGrad)
    .InferDataType(InferDataType4FusedFloydAttentionGrad);

} // namespace ops
