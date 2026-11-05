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
#include <cstdint>
#include <string>
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

static ge::graphStatus InferShape4MoeTokenPermuteWithEp(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeTokenPermuteWithEpInfershape.");
    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(PERMUTE_WITH_EP_INPUT_TOKENS);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tokens"),
                return ge::GRAPH_FAILED);
    const gert::Shape* indicesShape = context->GetInputShape(PERMUTE_WITH_EP_INPUT_IDX);
    OP_CHECK_IF(indicesShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "indices"),
                return ge::GRAPH_FAILED);

    gert::Shape* permuteXShape = context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_TOKENS);
    OP_CHECK_IF(permuteXShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "permuteXShape",
                    "Required permuteXShape metadata is null; "
                    "context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_TOKENS) must provide a valid object."),
                return ge::GRAPH_FAILED);
    gert::Shape* sortedIndices = context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_IDX);
    OP_CHECK_IF(sortedIndices == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "sortedIndices",
                    "Required sortedIndices metadata is null; "
                    "context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_IDX) must provide a valid object."),
                return ge::GRAPH_FAILED);
    gert::Shape* permuteProbsShape = context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_PROBS);
    OP_CHECK_IF(permuteProbsShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "permuteProbsShape",
                    "Required permuteProbsShape metadata is null; "
                    "context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_PROBS) must provide a valid object."),
                return ge::GRAPH_FAILED);

    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "attrs",
                    "Required attrs metadata is null; context->GetAttrs() must provide a valid object."),
                return ge::GRAPH_FAILED);
    auto rangePtr = attrs->GetAttrPointer<gert::ContinuousVector>(PERMUTE_WITH_EP_ARRT_RANGE);
    const int64_t* numOutTokensPtr = attrs->GetAttrPointer<int64_t>(PERMUTE_WITH_EP_ARRT_NUM_OUT_TOKENS);
    OP_CHECK_IF(numOutTokensPtr == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "num_out_tokens",
                    "Required attr num_out_tokens is null; it must provide a valid object."),
                return ge::GRAPH_FAILED);

    int64_t sortedIndicesLen = NEG_ONE;
    sortedIndices->SetDimNum(DIM_ONE);
    if (sortedIndices->GetDim(0) < 0) {
        sortedIndices->SetDim(0U, NEG_ONE);
    } else {
        int64_t IndicesDimNnum = indicesShape->GetDimNum();
        if (IndicesDimNnum != DIM_TWO && IndicesDimNnum != DIM_ONE) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                context->GetNodeName(), "indices",
                "The dim of indices should be 1 or 2, but got " + std::to_string(IndicesDimNnum) + ".");
            return ge::GRAPH_FAILED;
        }
        int64_t topK = (IndicesDimNnum == 1) ? 1 : indicesShape->GetDim(1);
        int64_t N = indicesShape->GetDim(0);
        // 任一维未知/为负或乘积可能溢出时传播未知维，避免负负得正或回绕成正数
        if (topK < 0 || N < 0) {
            sortedIndicesLen = NEG_ONE;
        } else if (topK != 0 && N > INT64_MAX / topK) {
            sortedIndicesLen = NEG_ONE;
        } else {
            sortedIndicesLen = topK * N;
        }
        sortedIndices->SetDim(0U, sortedIndicesLen);
    }

    int64_t start;
    int64_t end;
    if (rangePtr != nullptr) {
        if (rangePtr->GetSize() != DIM_TWO) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "range",
                                                     "the size of range only support 2");
            return ge::GRAPH_FAILED;
        }
        const int64_t* RangeList = reinterpret_cast<const int64_t*>(rangePtr->GetData());
        start = RangeList[0];
        end = RangeList[1];
    } else {
        start = 0;
        end = *numOutTokensPtr;
    }

    // 与 tiling 保持一致：shape 已知时校验 range/numOutTokens 取值域，越界直接失败
    if (sortedIndicesLen != NEG_ONE && (start < 0 || start > sortedIndicesLen || end < 0 || end > sortedIndicesLen)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "range, num_out_tokens",
                                                 "The start [" + std::to_string(start) + "] and end [" +
                                                     std::to_string(end) + "] should be in [0, " +
                                                     std::to_string(sortedIndicesLen) + "].");
        return ge::GRAPH_FAILED;
    }

    *permuteXShape = *xShape;
    permuteProbsShape->SetDimNum(DIM_ONE);
    permuteProbsShape->SetDim(0U, NEG_ONE);
    if (sortedIndicesLen != NEG_ONE && !Ops::Base::IsUnknownRank(*xShape)) {
        int64_t outTokens = (start >= end) ? sortedIndicesLen + (end - start) : (end - start);
        outTokens = std::min(outTokens, sortedIndicesLen);
        outTokens = std::max(outTokens, (int64_t)0);
        permuteXShape->SetDim(0, outTokens);
        permuteProbsShape->SetDim(0U, outTokens);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeTokenPermuteWithEp(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeTokenPermuteWithEpInferDataType.");
    auto xDtype = context->GetInputDataType(PERMUTE_WITH_EP_INPUT_TOKENS);
    auto probsDtype = context->GetInputDataType(PERMUTE_WITH_EP_INPUT_PROBS);
    if (probsDtype == ge::DT_UNDEFINED) {
        probsDtype = xDtype;
    }
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
