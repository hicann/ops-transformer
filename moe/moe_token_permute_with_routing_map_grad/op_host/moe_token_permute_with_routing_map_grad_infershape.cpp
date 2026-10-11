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
 * \file moe_token_permute_with_routing_map_grad_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForMoeTokenPermuteWithRoutingMapGrad(gert::InferShapeContext* context)
{
    const gert::Shape* permuted_token_output_grad_shape = context->GetInputShape(0);
    const int64_t* tokens_num = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    int64_t tokensNum = *tokens_num;

    int64_t hidden_size = permuted_token_output_grad_shape->GetDim(1);
    int64_t topk_num = permuted_token_output_grad_shape->GetDim(0) / tokensNum;

    gert::Shape* tokens_grad_out_shape = context->GetOutputShape(0);
    gert::Shape* probs_grad_out_optional_shape = context->GetOutputShape(1);
    const int64_t out_dim_num = 2;
    tokens_grad_out_shape->SetDimNum(out_dim_num);
    tokens_grad_out_shape->SetDim(0, tokensNum);
    tokens_grad_out_shape->SetDim(1, hidden_size);
    probs_grad_out_optional_shape->SetDimNum(out_dim_num);
    probs_grad_out_optional_shape->SetDim(0, tokensNum);
    probs_grad_out_optional_shape->SetDim(1, topk_num);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMoeTokenPermuteWithRoutingMapGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenPermuteWithRoutingMapGrad)
    .InferShape(InferShapeForMoeTokenPermuteWithRoutingMapGrad)
    .InferDataType(InferDataTypeForMoeTokenPermuteWithRoutingMapGrad);
} // namespace ops
