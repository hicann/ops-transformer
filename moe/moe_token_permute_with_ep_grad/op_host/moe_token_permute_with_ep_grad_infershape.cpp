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
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static constexpr size_t ATTR_INDEX_TOPK = 0;
static constexpr size_t ATTR_INDEX_RANGE = 1;

static ge::graphStatus InferShapeForMoeTokenPermuteWithEpGrad(gert::InferShapeContext* context)
{
    const gert::Shape* permuted_inputs_shape = context->GetInputShape(0);
    OP_CHECK_IF(permuted_inputs_shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "permuted_tokens_output_grad"),
                return ge::GRAPH_FAILED);
    const gert::Shape* indices_shape = context->GetInputShape(1);
    OP_CHECK_IF(indices_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "sorted_indices"),
                return ge::GRAPH_FAILED);
    const int64_t* top_k = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_INDEX_TOPK);
    OP_CHECK_IF(top_k == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "num_topk"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *top_k < 1,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "num_topk", "num_topk must be positive."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *top_k > 512,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "num_topk", "num_topk cannot exceed 512."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(indices_shape->GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "sorted_indices",
                                                         std::to_string(indices_shape->GetDimNum()),
                                                         "sorted_indices must be 1D."),
                return ge::GRAPH_FAILED);
    int64_t topk = *top_k;
    int64_t indices_dim0 = indices_shape->GetDim(0);
    OP_CHECK_IF(indices_dim0 != -1 && indices_dim0 % topk != 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "sorted_indices, num_topk",
                                                         "sorted_indices length must be a multiple of num_topk."),
                return ge::GRAPH_FAILED);
    int64_t tokens_num = (indices_dim0 == -1) ? -1 : indices_dim0 / topk;

    gert::Shape* out_shape = context->GetOutputShape(0);
    OP_CHECK_IF(out_shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "out_shape",
                    "Required out_shape metadata is null; context->GetOutputShape(0) must provide a valid object."),
                return ge::GRAPH_FAILED);
    const int8_t out_dim_num = 2;
    out_shape->SetDimNum(out_dim_num);
    out_shape->SetDim(0, tokens_num);
    out_shape->SetDim(1, permuted_inputs_shape->GetDim(1));

    // probs_grad 为可选输出：图调用省略该输出时 GetOutputShape(1) 返回 nullptr，跳过推导即可，不报错；
    // 输出存在时，仅当 range 提供且 probs 输入非空才推导 2D shape，否则为空 shape。
    gert::Shape* probs_out_shape = context->GetOutputShape(1);
    if (probs_out_shape != nullptr) {
        auto attrs = context->GetAttrs();
        const gert::ContinuousVector* rangePtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_RANGE);
        bool rangeProvided = (rangePtr != nullptr && rangePtr->GetSize() == 2);
        const gert::Shape* probs_input_shape = context->GetInputShape(2);
        bool hasProbs = rangeProvided && probs_input_shape != nullptr && probs_input_shape->GetDimNum() > 0;
        if (hasProbs) {
            probs_out_shape->SetDimNum(out_dim_num);
            probs_out_shape->SetDim(0, tokens_num);
            probs_out_shape->SetDim(1, topk);
        } else {
            // range 为空（回退）或 permuted_probs_output_grad 为空/缺失时，probs_grad 不输出（空 shape）
            probs_out_shape->SetDimNum(0);
        }
    }

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMoeTokenPermuteWithEpGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenPermuteWithEpGrad)
    .InferShape(InferShapeForMoeTokenPermuteWithEpGrad)
    .InferDataType(InferDataTypeForMoeTokenPermuteWithEpGrad);
} // namespace ops
