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
 * \file moe_re_routing_infershape.cpp
 * \brief
 */
#include <sstream>
#include <string>
#include <vector>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_TOKENS = 0;
constexpr size_t INPUT_IDX_EXPERT_TOKEN_NUM_PER_RANK = 1;
constexpr size_t INPUT_IDX_PER_TOKEN_SCALES = 2;
constexpr size_t OUTPUT_IDX_PERMUTE_TOKENS = 0;
constexpr size_t OUTPUT_IDX_PERMUTE_PER_TOKEN_SCALES = 1;
constexpr size_t OUTPUT_IDX_PERMUTE_TOKEN_IDX = 2;
constexpr size_t OUTPUT_IDX_EXPERT_TOKEN_NUM = 3;
constexpr size_t IDX_ZERO = 0;
constexpr size_t IDX_ONE = 1;
constexpr int64_t DIM_NUM = 2;

graphStatus InferShape4MoeReRouting(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4MoeReRouting.");

    const gert::Shape* tokensInputShape = context->GetInputShape(INPUT_IDX_TOKENS); // [A, H]
    OP_CHECK_NULL_WITH_CONTEXT(context, tokensInputShape);
    const gert::Shape* expertTokenNumPerRankInputShape = context->GetInputShape(INPUT_IDX_EXPERT_TOKEN_NUM_PER_RANK);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumPerRankInputShape);
    const gert::Shape* perTokenScaleShape = context->GetOptionalInputShape(INPUT_IDX_PER_TOKEN_SCALES);

    gert::Shape* permuteTokensShape = context->GetOutputShape(OUTPUT_IDX_PERMUTE_TOKENS);
    gert::Shape* permutePerTokenScalesShape = context->GetOutputShape(OUTPUT_IDX_PERMUTE_PER_TOKEN_SCALES);
    gert::Shape* permuteTokenIdxShape = context->GetOutputShape(OUTPUT_IDX_PERMUTE_TOKEN_IDX);
    gert::Shape* expertTokenNumShape = context->GetOutputShape(OUTPUT_IDX_EXPERT_TOKEN_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuteTokensShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, permutePerTokenScalesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuteTokenIdxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumShape);

    *permuteTokensShape = *tokensInputShape; // [A, H]
    size_t tokensDim = tokensInputShape->GetDimNum();
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*tokensInputShape) && tokensDim != DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "tokens", std::to_string(tokensDim).c_str(), "2D"),
                return ge::GRAPH_FAILED);
    const auto& tokensInputShapeDims = tokensInputShape->GetDim(IDX_ZERO); // 取A
    size_t expertDim = expertTokenNumPerRankInputShape->GetDimNum();
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*expertTokenNumPerRankInputShape) && expertDim != DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "expert_token_num_per_rank_dim",
                                             std::to_string(expertDim).c_str(), "2D"),
                return ge::GRAPH_FAILED);
    if (perTokenScaleShape) {
        *permutePerTokenScalesShape = *perTokenScaleShape;
    } else {
        permutePerTokenScalesShape->SetDimNum(IDX_ONE);
        permutePerTokenScalesShape->SetDim(IDX_ZERO, tokensInputShapeDims); // [A]
    }
    permuteTokenIdxShape->SetDimNum(IDX_ONE);
    permuteTokenIdxShape->SetDim(IDX_ZERO, tokensInputShapeDims); // [A]
    if (Ops::Base::IsUnknownRank(*expertTokenNumPerRankInputShape)) {
        *expertTokenNumShape = *expertTokenNumPerRankInputShape;
    } else {
        expertTokenNumShape->SetDimNum(IDX_ONE);
        const auto& expertTokenNumPerRankInputShapeDims = expertTokenNumPerRankInputShape->GetDim(IDX_ONE); // 取E
        expertTokenNumShape->SetDim(IDX_ZERO, expertTokenNumPerRankInputShapeDims);                         // [E,]
    }
    OP_LOGD(context, "End to do InferShape4MoeReRouting.");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4MoeReRouting(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDtype4MoeReRouting enter");

    auto tokens_input_dtype = context->GetInputDataType(INPUT_IDX_TOKENS);
    auto expt_token_num_per_rank_input_dtype = context->GetInputDataType(INPUT_IDX_EXPERT_TOKEN_NUM_PER_RANK);
    ge::DataType scalesDataType = context->GetOptionalInputDataType(INPUT_IDX_PER_TOKEN_SCALES);
    if (scalesDataType != ge::DT_UNDEFINED) {
        context->SetOutputDataType(OUTPUT_IDX_PERMUTE_PER_TOKEN_SCALES, scalesDataType);
    } else {
        context->SetOutputDataType(OUTPUT_IDX_PERMUTE_PER_TOKEN_SCALES, ge::DT_FLOAT);
    }

    context->SetOutputDataType(OUTPUT_IDX_PERMUTE_TOKENS, tokens_input_dtype);
    context->SetOutputDataType(OUTPUT_IDX_PERMUTE_TOKEN_IDX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_IDX_EXPERT_TOKEN_NUM, expt_token_num_per_rank_input_dtype);
    OP_LOGD(context, "InferDtype4MoeReRouting end");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeReRouting).InferShape(InferShape4MoeReRouting).InferDataType(InferDtype4MoeReRouting);
} // namespace ops
