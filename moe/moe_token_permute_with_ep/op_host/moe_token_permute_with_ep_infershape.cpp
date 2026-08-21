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
 * \file moe_token_permute_with_ep_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t PERMUTE_WITH_EP_INPUT_TOKENS = 0;
static constexpr int64_t PERMUTE_WITH_EP_INPUT_IDX = 1;
static constexpr int64_t PERMUTE_WITH_EP_INPUT_PROBS = 2;
static constexpr int64_t PERMUTE_WITH_EP_OUTPUT_TOKENS = 0;
static constexpr int64_t PERMUTE_WITH_EP_OUTPUT_IDX = 1;
static constexpr int64_t PERMUTE_WITH_EP_OUTPUT_PROBS = 2;
static constexpr int64_t PERMUTE_WITH_EP_ARRT_RANGE = 0;
static constexpr int64_t PERMUTE_WITH_EP_ARRT_NUM_OUT_TOKENS = 1;

static ge::graphStatus InferShape4MoeTokenPermuteWithEp(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeTokenPermuteWithEpInfershape.");
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(PERMUTE_WITH_EP_INPUT_TOKENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *indicesShape = context->GetInputShape(PERMUTE_WITH_EP_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

    gert::Shape *permuteXShape = context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_TOKENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuteXShape);
    gert::Shape *sortedIndices = context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortedIndices);
    gert::Shape *permuteProbsShape = context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_PROBS);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuteProbsShape);

    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto rangePtr = attrs->GetAttrPointer<gert::ContinuousVector>(PERMUTE_WITH_EP_ARRT_RANGE);
    const int64_t *numOutTokensPtr = attrs->GetAttrPointer<int64_t>(PERMUTE_WITH_EP_ARRT_NUM_OUT_TOKENS);

    int64_t sortedIndicesLen = NEG_ONE;
    sortedIndices->SetDimNum(DIM_ONE);
    if (sortedIndices->GetDim(0) < 0) {
        sortedIndices->SetDim(0U, NEG_ONE);
    } else {
        int64_t IndicesDimNnum = indicesShape->GetDimNum();
        if (IndicesDimNnum != DIM_TWO && IndicesDimNnum != DIM_ONE) {
            OP_LOGE(context->GetNodeName(), "The dim of indices should be 1 or 2, but got %ld.", IndicesDimNnum);
            return ge::GRAPH_FAILED;
        }
        int64_t topK = (IndicesDimNnum == 1) ? 1 : indicesShape->GetDim(1);
        int64_t N = indicesShape->GetDim(0);
        sortedIndicesLen = (topK * N > 0) ? topK * N : NEG_ONE;
        sortedIndices->SetDim(0U, sortedIndicesLen);
    }

    int64_t start;
    int64_t end;
    if (rangePtr != nullptr) {
        const int64_t *RangeList = reinterpret_cast<const int64_t *>(rangePtr->GetData());
        start = RangeList[0];
        end = RangeList[1];
    } else {
        start = 0;
        end = *numOutTokensPtr;
    }

    *permuteXShape = *xShape;
    if (sortedIndicesLen != NEG_ONE && !Ops::Base::IsUnknownRank(*xShape)) {
        int64_t outTokens = (start >= end) ? sortedIndicesLen + (end - start) : (end - start);
        outTokens = std::min(outTokens, sortedIndicesLen);
        outTokens = std::max(outTokens, (int64_t)0);
        permuteXShape->SetDim(0, outTokens);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeTokenPermuteWithEp(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeTokenPermuteWithEpInferDataType.");
    auto xDtype = context->GetInputDataType(PERMUTE_WITH_EP_INPUT_TOKENS);
    auto probsDtype = context->GetInputDataType(PERMUTE_WITH_EP_INPUT_PROBS);
    context->SetOutputDataType(PERMUTE_WITH_EP_OUTPUT_TOKENS, xDtype);
    context->SetOutputDataType(PERMUTE_WITH_EP_OUTPUT_IDX, ge::DT_INT32);
    context->SetOutputDataType(PERMUTE_WITH_EP_OUTPUT_PROBS, probsDtype);
    OP_LOGD(context->GetNodeName(), "End to do MoeTokenPermuteWithEpInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenPermuteWithEp)
    .InferShape(InferShape4MoeTokenPermuteWithEp)
    .InferDataType(InferDataType4MoeTokenPermuteWithEp);
} // namespace ops
