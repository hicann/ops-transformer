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
 * \file moe_token_unpermute_with_ep_grad_infershape.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "util/shape_util.h"

using namespace ge;
static constexpr size_t INPUT_PERMUTED_TOKENS_IDX = 2;
static constexpr size_t INPUT_UNPERMUTEDOUTPUTD_IDX = 0;
static constexpr size_t INPUT_ROWIDMAP_IDX = 1;
static constexpr size_t INPUT_PROB_IDX = 3;
static constexpr size_t OUTPUT_PERMUTEDTOKENSGRAD_IDX = 0;
static constexpr size_t OUTPUT_PROBGRAD_IDX = 1;
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_NUM_TWO = 2;
static constexpr size_t DIM_NUM_ONE = 1;
static constexpr size_t OUTPUT_INPUTGRAD_DIMNUM = 2;

namespace ops {
static graphStatus MoeTokenUnpermuteWithEpGrad(gert::InferShapeContext* context)
{
    constexpr int64_t UNKNOWN_RANK_DIM_VALUE_EP = -2;
    OP_LOGD(context->GetNodeName(), "Begin to do MoeTokenUnpermuteWithEpGrad");
    const gert::Shape* permutedTokensShape = context->GetInputShape(INPUT_PERMUTED_TOKENS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, permutedTokensShape);
    const gert::Shape* unpermutedOutputDShape = context->GetInputShape(INPUT_UNPERMUTEDOUTPUTD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, unpermutedOutputDShape);
    const gert::Shape* rowIdMapShape = context->GetInputShape(INPUT_ROWIDMAP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowIdMapShape);
    const gert::Shape* probShape = context->GetOptionalInputShape(INPUT_PROB_IDX);

    gert::Shape* permutedTokensGradShape = context->GetOutputShape(OUTPUT_PERMUTEDTOKENSGRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, permutedTokensGradShape);
    gert::Shape* probGradShape = context->GetOutputShape(OUTPUT_PROBGRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, probGradShape);

    // permutedTokensGrad
    if (Ops::Base::IsUnknownRank(*permutedTokensShape)) { // [-2]输入
        OP_LOGD(context->GetNodeName(), "Input shape is -2, set output shape to (-2)");
        permutedTokensGradShape->SetDim(DIM_0, UNKNOWN_RANK_DIM_VALUE_EP);
    } else {
        if (permutedTokensShape->GetDimNum() != DIM_NUM_TWO || unpermutedOutputDShape->GetDimNum() != DIM_NUM_TWO ||
            rowIdMapShape->GetDimNum() != DIM_NUM_ONE) {
            OP_LOGE(context->GetNodeName(),
                    "The dim number of input must be 2, indices must be 1, but got: input %zu, indices %zu",
                    permutedTokensShape->GetDimNum(), rowIdMapShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        uint32_t inputGradDim0 = rowIdMapShape->GetDim(DIM_0);
        uint32_t hiddenSize = unpermutedOutputDShape->GetDim(DIM_1);
        permutedTokensGradShape->SetDimNum(OUTPUT_INPUTGRAD_DIMNUM);
        permutedTokensGradShape->SetDim(DIM_0, inputGradDim0);
        permutedTokensGradShape->SetDim(DIM_1, hiddenSize);
    }
    // probGrad
    if (probShape != nullptr) {
        OP_LOGD(context->GetNodeName(), "MoeTokenUnpermuteWithEpGrad: probShape is not null");
        *probGradShape = *probShape;
    } else {
        OP_LOGD(context->GetNodeName(), "MoeTokenUnpermuteWithEpGrad: probShape is null");
        probGradShape = nullptr;
    }
    OP_LOGD(context->GetNodeName(), "End to do MoeTokenUnpermuteWithEpGrad");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeTokenUnpermuteWithEpGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4MoeTokenUnpermuteGrad");
    OP_LOGD(context->GetNodeName(), "input input dtype: %s",
            Ops::Base::ToString(context->GetInputDataType(INPUT_PERMUTED_TOKENS_IDX)).c_str());
    context->SetOutputDataType(OUTPUT_PERMUTEDTOKENSGRAD_IDX, context->GetInputDataType(INPUT_PERMUTED_TOKENS_IDX));
    context->SetOutputDataType(OUTPUT_PROBGRAD_IDX, context->GetInputDataType(INPUT_PERMUTED_TOKENS_IDX));
    // 混精场景PROBGRAD dtype与PROB一致，与PERMUTED_TOKENS不一致
    if (context->GetOptionalInputDataType(INPUT_PROB_IDX) != ge::DT_UNDEFINED) {
        context->SetOutputDataType(OUTPUT_PROBGRAD_IDX, context->GetInputDataType(INPUT_PROB_IDX));
    }
    OP_LOGD(context->GetNodeName(), "output input_grad dtype: %s",
            Ops::Base::ToString(context->GetOutputDataType(OUTPUT_PERMUTEDTOKENSGRAD_IDX)).c_str());
    OP_LOGD(context->GetNodeName(), "output prob_grad dtype: %s",
            Ops::Base::ToString(context->GetOutputDataType(OUTPUT_PROBGRAD_IDX)).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4MoeTokenUnpermuteGrad");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenUnpermuteWithEpGrad)
    .InferShape(MoeTokenUnpermuteWithEpGrad)
    .InferDataType(InferDataType4MoeTokenUnpermuteWithEpGrad);
} // namespace ops
