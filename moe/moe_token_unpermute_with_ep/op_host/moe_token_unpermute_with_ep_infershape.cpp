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
 * \file moe_token_unpermute_with_ep_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include <string>
using namespace ge;
namespace ops {

static constexpr int64_t UNPERMUTE_WITH_EP_INPUT_TOKENS = 0;
static constexpr int64_t UNPERMUTE_WITH_EP_INPUT_IDX = 1;
static constexpr int64_t UNPERMUTE_WITH_EP_INPUT_PROBS = 2;
static constexpr int64_t UNPERMUTE_WITH_EP_OUTPUT_TOKENS = 0;
static constexpr int64_t UNPERMUTE_WITH_EP_ARRT_TOPK = 0;
static constexpr int64_t UNKNOWN_DIM = -1;
static constexpr int64_t MAX_NUM_TOPK = 512;

static ge::graphStatus InferShapeForMoeTokenUnpermuteWithEp(gert::InferShapeContext* context)
{
    const char* nodeName = context->GetNodeName();
    const gert::Shape* permuted_inputs_shape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    const gert::Shape* sortedIndicesShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_IDX);
    const gert::Shape* probs_shape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_PROBS);
    const auto* attrs = context->GetAttrs();

    OP_CHECK_IF(
        permuted_inputs_shape == nullptr || sortedIndicesShape == nullptr || attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            nodeName, "context", "permuted_tokens/sorted_indices shape or attrs is nullptr during infer shape."),
        return GRAPH_FAILED);
    const int64_t* topk = attrs->GetAttrPointer<int64_t>(UNPERMUTE_WITH_EP_ARRT_TOPK);
    OP_CHECK_IF(topk == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "num_topk", "num_topk is nullptr."),
                return GRAPH_FAILED);
    const int64_t inputTopK = *topk;
    OP_CHECK_IF(inputTopK < 1,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "num_topk", "num_topk must be no less than 1."),
                return GRAPH_FAILED);
    OP_CHECK_IF(inputTopK > MAX_NUM_TOPK,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    nodeName, "num_topk", "num_topk must be no greater than " + std::to_string(MAX_NUM_TOPK) + "."),
                return GRAPH_FAILED);

    OP_CHECK_IF(permuted_inputs_shape->GetDimNum() != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "permuted_tokens",
                                                         std::to_string(permuted_inputs_shape->GetDimNum()),
                                                         "permuted_tokens's shape is not 2D."),
                return GRAPH_FAILED);
    OP_CHECK_IF(permuted_inputs_shape->GetDim(0) == 0 || permuted_inputs_shape->GetDim(1) == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "permuted_tokens",
                                                         "permuted_tokens does not support empty tensor."),
                return GRAPH_FAILED);

    OP_CHECK_IF(sortedIndicesShape->GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "sorted_indices",
                                                         std::to_string(sortedIndicesShape->GetDimNum()),
                                                         "sorted_indices's shape is not 1D."),
                return GRAPH_FAILED);
    const int64_t indicesDim0 = sortedIndicesShape->GetDim(0);
    OP_CHECK_IF(indicesDim0 == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "sorted_indices",
                                                         "sorted_indices does not support empty tensor."),
                return GRAPH_FAILED);

    int64_t tokens_num;
    if (probs_shape == nullptr) {
        if (indicesDim0 == UNKNOWN_DIM) {
            tokens_num = UNKNOWN_DIM;
        } else {
            OP_CHECK_IF(indicesDim0 % inputTopK != 0 || indicesDim0 / inputTopK < 1,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            nodeName, "sorted_indices",
                            "sorted_indices length(" + std::to_string(indicesDim0) +
                                ") must be a positive multiple of num_topk(" + std::to_string(inputTopK) + ")."),
                        return GRAPH_FAILED);
            tokens_num = indicesDim0 / inputTopK;
        }
    } else {
        OP_CHECK_IF(probs_shape->GetDimNum() != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "probs", std::to_string(probs_shape->GetDimNum()), "probs's shape is not 2D."),
                    return GRAPH_FAILED);
        OP_CHECK_IF(probs_shape->GetDim(0) == 0 || probs_shape->GetDim(1) == 0,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "probs", "probs does not support empty tensor."),
                    return GRAPH_FAILED);
        tokens_num = probs_shape->GetDim(0);
    }

    gert::Shape* out_shape = context->GetOutputShape(UNPERMUTE_WITH_EP_OUTPUT_TOKENS);
    OP_CHECK_IF(out_shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(nodeName, "unpermuted_tokens",
                                                         "output shape is nullptr during infer shape."),
                return GRAPH_FAILED);
    const int8_t out_dim_num = 2;
    out_shape->SetDimNum(out_dim_num);
    out_shape->SetDim(0, tokens_num);
    out_shape->SetDim(1, permuted_inputs_shape->GetDim(1));

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenUnpermuteWithEp).InferShape(InferShapeForMoeTokenUnpermuteWithEp);
} // namespace ops
