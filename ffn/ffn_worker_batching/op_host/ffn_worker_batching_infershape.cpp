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
 * \file ffn_worker_batching_infershape.cpp
 * \brief
 */

#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include <limits>
#include <string>

using namespace ge;
namespace ops {
static constexpr int64_t EXPERT_NUM_ATTR = 0;
static constexpr int64_t MAX_OUT_SHAPE_ATTR = 1;
static constexpr int64_t TOKEN_DTYPE_ATTR = 2;

static constexpr int64_t Y_OUT = 0;
static constexpr int64_t GROUP_LIST_OUT = 1;
static constexpr int64_t SESSION_IDS_OUT = 2;
static constexpr int64_t MICRO_BATCH_IDS_OUT = 3;
static constexpr int64_t TOKEN_IDS_OUT = 4;
static constexpr int64_t EXPERT_OFFSETS_OUT = 5;
static constexpr int64_t DYNAMIC_SCALE_OUT = 6;
static constexpr int64_t ACTUAL_TOKEN_NUM_OUT = 7;

static constexpr int64_t TOKEN_KIND_ZERO = 0;
static constexpr int64_t TOKEN_KIND_ONE = 1;
static constexpr int64_t TOKEN_KIND_TWO = 2;

static constexpr int64_t NUM_FOUR = 4;
static constexpr uint32_t INDEX_ZERO = 0;
static constexpr uint32_t INDEX_ONE = 1;
static constexpr uint32_t INDEX_TWO = 2;
static constexpr uint32_t INDEX_THREE = 3;

static constexpr int64_t EVEN_ALIGN = 2;
static constexpr int64_t TOKEN_DTYPE_E5M2 = 3;
static constexpr int64_t TOKEN_DTYPE_E4M3 = 4;
static constexpr int64_t TOKEN_DTYPE_E2M1 = 5;
static constexpr int64_t MXFP_SCALE_GROUP = 32;

static graphStatus InferShape4FfnWorkerBatching(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* expertNumPtr = attrs->GetAttrPointer<int64_t>(EXPERT_NUM_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertNumPtr);
    auto expertNum = *expertNumPtr;
    if (expertNum <= 0) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "expert_num", std::to_string(expertNum), "greater than 0");
        return ge::GRAPH_FAILED;
    }
    auto maxOutShapePtr = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(MAX_OUT_SHAPE_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxOutShapePtr);
    if (maxOutShapePtr->GetSize() != NUM_FOUR) {
        OP_LOGE_WITH_INVALID_ATTR_SIZE(context->GetNodeName(), "max_out_shape",
                                       std::to_string(maxOutShapePtr->GetSize()), std::to_string(NUM_FOUR));
        return ge::GRAPH_FAILED;
    }

    const int64_t* maxOutShapeArray = reinterpret_cast<const int64_t*>(maxOutShapePtr->GetData());
    const int64_t A = maxOutShapeArray[INDEX_ZERO];
    const int64_t BS = maxOutShapeArray[INDEX_ONE];
    const int64_t K = maxOutShapeArray[INDEX_TWO];
    const int64_t H = maxOutShapeArray[INDEX_THREE];
    if (A <= 0 || BS <= 0 || K <= 0 || H <= 0) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "max_out_shape",
                                  "[" + std::to_string(A) + ", " + std::to_string(BS) + ", " + std::to_string(K) +
                                      ", " + std::to_string(H) + "]",
                                  "all elements greater than 0");
        return ge::GRAPH_FAILED;
    }
    // The runtime context stores H as uint32. Reject unsupported values during
    // inference, before deriving output shapes or reaching the later tiling checks.
    OP_CHECK_IF(H > std::numeric_limits<uint32_t>::max(),
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "max_out_shape[3]", std::to_string(H),
                                          "at most " + std::to_string(std::numeric_limits<uint32_t>::max())),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(A > std::numeric_limits<int64_t>::max() / BS,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "max_out_shape[0], max_out_shape[1]",
                                                       std::to_string(A) + ", " + std::to_string(BS),
                                                       "A*BS must not overflow int64"),
                return ge::GRAPH_FAILED);
    const int64_t aBs = A * BS;
    OP_CHECK_IF(
        aBs > std::numeric_limits<int64_t>::max() / K,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "max_out_shape[0], max_out_shape[1], max_out_shape[2]",
            std::to_string(A) + ", " + std::to_string(BS) + ", " + std::to_string(K), "A*BS*K must not overflow int64"),
        return ge::GRAPH_FAILED);
    const int64_t Y = aBs * K;

    gert::Shape* yShape = context->GetOutputShape(Y_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* groupList = context->GetOutputShape(GROUP_LIST_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupList);

    gert::Shape* sessionIds = context->GetOutputShape(SESSION_IDS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, sessionIds);

    gert::Shape* microBatchIds = context->GetOutputShape(MICRO_BATCH_IDS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, microBatchIds);

    gert::Shape* tokenIds = context->GetOutputShape(TOKEN_IDS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenIds);

    gert::Shape* expertOffsets = context->GetOutputShape(EXPERT_OFFSETS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertOffsets);

    gert::Shape* dynamicScale = context->GetOutputShape(DYNAMIC_SCALE_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, dynamicScale);

    gert::Shape* actualTokenNum = context->GetOutputShape(ACTUAL_TOKEN_NUM_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, actualTokenNum);

    *yShape = {Y, H};
    *groupList = {expertNum, 2};
    *sessionIds = {Y};
    *microBatchIds = {Y};
    *tokenIds = {Y};
    *expertOffsets = {Y};
    const int64_t* tokenDtypePtr = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    const int64_t tokenDtype = tokenDtypePtr == nullptr ? TOKEN_KIND_ZERO : *tokenDtypePtr;
    if (tokenDtype == TOKEN_DTYPE_E2M1 && H % EVEN_ALIGN != 0) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "max_out_shape[3]", std::to_string(H),
                                  "even when token_dtype is 5 (float4_e2m1)");
        return ge::GRAPH_FAILED;
    }
    if (tokenDtype >= TOKEN_DTYPE_E5M2 && tokenDtype <= TOKEN_DTYPE_E2M1) {
        *dynamicScale = {Y, H / MXFP_SCALE_GROUP + (H % MXFP_SCALE_GROUP != 0)};
    } else {
        *dynamicScale = {Y};
    }
    *actualTokenNum = {1};

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4FfnWorkerBatching(gert::InferDataTypeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* tokenDtypePtr = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    int64_t tokenDtype = TOKEN_KIND_ZERO;
    if (tokenDtypePtr != nullptr) {
        tokenDtype = *tokenDtypePtr;
    }

    if (tokenDtype == TOKEN_KIND_ZERO) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT16);
    } else if (tokenDtype == TOKEN_KIND_ONE) {
        context->SetOutputDataType(Y_OUT, ge::DT_BF16);
    } else if (tokenDtype == TOKEN_DTYPE_E5M2) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT8_E5M2);
    } else if (tokenDtype == TOKEN_DTYPE_E4M3) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT8_E4M3FN);
    } else if (tokenDtype == TOKEN_DTYPE_E2M1) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT4_E2M1);
    } else {
        context->SetOutputDataType(Y_OUT, ge::DT_INT8);
    }

    context->SetOutputDataType(GROUP_LIST_OUT, ge::DT_INT64);
    context->SetOutputDataType(SESSION_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(MICRO_BATCH_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(TOKEN_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(EXPERT_OFFSETS_OUT, ge::DT_INT32);
    context->SetOutputDataType(DYNAMIC_SCALE_OUT, tokenDtype >= TOKEN_DTYPE_E5M2 && tokenDtype <= TOKEN_DTYPE_E2M1 ?
                                                      ge::DT_FLOAT8_E8M0 :
                                                      ge::DT_FLOAT);
    context->SetOutputDataType(ACTUAL_TOKEN_NUM_OUT, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FfnWorkerBatching)
    .InferShape(InferShape4FfnWorkerBatching)
    .InferDataType(InferDataType4FfnWorkerBatching);
} // namespace ops
