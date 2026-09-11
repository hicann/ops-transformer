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
 * \file moe_token_permute_with_ep_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForMoeTokenPermuteWithEpGrad(gert::InferShapeContext *context)
{
    const gert::Shape *permuted_inputs_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuted_inputs_shape);
    const gert::Shape *indices_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
    const int64_t *top_k = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, top_k);
    OP_CHECK_IF(*top_k == 0, OP_LOGE(context->GetNodeName(), "num_topk cannot be 0."), return ge::GRAPH_FAILED);
    int64_t topk = *top_k;
    int64_t indices_dim0 = indices_shape->GetDim(0);
    int64_t tokens_num = (indices_dim0 == -1) ? -1 : indices_dim0 / topk;

    gert::Shape *out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    const int8_t out_dim_num = 2;
    out_shape->SetDimNum(out_dim_num);
    out_shape->SetDim(0, tokens_num);
    out_shape->SetDim(1, permuted_inputs_shape->GetDim(1));

    gert::Shape *probs_out_shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, probs_out_shape);
    probs_out_shape->SetDimNum(out_dim_num);
    probs_out_shape->SetDim(0, tokens_num);
    probs_out_shape->SetDim(1, topk);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMoeTokenPermuteWithEpGrad(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenPermuteWithEpGrad)
    .InferShape(InferShapeForMoeTokenPermuteWithEpGrad)
    .InferDataType(InferDataTypeForMoeTokenPermuteWithEpGrad);
} // namespace ops
