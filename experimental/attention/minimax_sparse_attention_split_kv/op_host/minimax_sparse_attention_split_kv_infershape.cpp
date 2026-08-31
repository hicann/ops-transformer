/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include <cstring>

using namespace ge;

namespace ops {

static constexpr uint32_t QUERY_INDEX = 0;
static constexpr uint32_t ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t SOFTMAX_LSE_INDEX = 1;
static constexpr uint32_t TND_DIM_NUM = 3;
static constexpr uint32_t RANK4_DIM_NUM = 4;
static constexpr uint32_t LSE_DIM_NUM = 3;
static constexpr uint32_t LSE_RANK4_DIM_NUM = 4;
static constexpr int32_t UNKNOWN_DIMS = -2;
static constexpr int64_t NUM_0 = 0;
static constexpr int64_t NUM_1 = 1;
// Matches op_def Attr order: numKeyValueHeads, scaleValue, blockSize, topK,
// innerPrecise, softmaxLseFlag, inputLayout.
static constexpr uint32_t ATTR_SOFTMAX_LSE_FLAG_INDEX = 5;
static constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 6;

static constexpr uint32_t LAYOUT_TND = 0;
static constexpr uint32_t LAYOUT_BNSD = 1;
static constexpr uint32_t LAYOUT_BSND = 2;

static bool ParseInputLayout(const char *layoutStr, uint32_t &layoutType)
{
    if (layoutStr == nullptr || layoutStr[0] == '\0') {
        layoutType = LAYOUT_TND;
        return true;
    }
    if (strcmp(layoutStr, "TND") == 0) {
        layoutType = LAYOUT_TND;
        return true;
    }
    if (strcmp(layoutStr, "BNSD") == 0) {
        layoutType = LAYOUT_BNSD;
        return true;
    }
    if (strcmp(layoutStr, "BSND") == 0) {
        layoutType = LAYOUT_BSND;
        return true;
    }
    return false;
}

static ge::graphStatus InferShapeMinimaxSparseAttentionSplitKv(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("MinimaxSparseAttentionSplitKv", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    gert::Shape *attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

    gert::Shape *softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);

    if (queryShape->GetDimNum() == 1 && queryShape->GetDim(0) == UNKNOWN_DIMS) {
        attentionOutShape->SetDimNum(1);
        (*attentionOutShape)[0] = UNKNOWN_DIMS;
        softmaxLseShape->SetDimNum(1);
        (*softmaxLseShape)[0] = UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    uint32_t layoutType = LAYOUT_TND;
    if (!ParseInputLayout(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX), layoutType)) {
        OP_LOGE(context->GetNodeName(), "inputLayout must be TND, BNSD or BSND.");
        return ge::GRAPH_FAILED;
    }

    const size_t qDimNum = queryShape->GetDimNum();
    if (layoutType == LAYOUT_TND) {
        if (qDimNum != TND_DIM_NUM) {
            OP_LOGE(context->GetNodeName(), "inputLayout TND requires query rank 3 [T, N, D], got %zu.", qDimNum);
            return ge::GRAPH_FAILED;
        }
    } else if (qDimNum != RANK4_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "inputLayout BNSD/BSND requires query rank 4, got %zu.", qDimNum);
        return ge::GRAPH_FAILED;
    }

    *attentionOutShape = *queryShape;

    const bool *softmaxLsePtr = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    bool softmaxLseFlag = (softmaxLsePtr != nullptr) ? *softmaxLsePtr : false;
    if (softmaxLseFlag) {
        if (layoutType == LAYOUT_BNSD) {
            // BNSD LSE [B, N, S, 1] fp32.
            softmaxLseShape->SetDimNum(LSE_RANK4_DIM_NUM);
            *softmaxLseShape = {queryShape->GetDim(0), queryShape->GetDim(1), queryShape->GetDim(2), NUM_1};
        } else if (layoutType == LAYOUT_BSND) {
            // BSND LSE [B, S, N, 1] fp32.
            softmaxLseShape->SetDimNum(LSE_RANK4_DIM_NUM);
            *softmaxLseShape = {queryShape->GetDim(0), queryShape->GetDim(1), queryShape->GetDim(2), NUM_1};
        } else {
            // TND LSE, same as FIA: [T, N, 1] fp32.
            softmaxLseShape->SetDimNum(LSE_DIM_NUM);
            *softmaxLseShape = {queryShape->GetDim(0), queryShape->GetDim(1), NUM_1};
        }
    } else {
        softmaxLseShape->SetDimNum(1);
        (*softmaxLseShape)[0] = NUM_0;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMinimaxSparseAttentionSplitKv(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(QUERY_INDEX);
    if (dtype == ge::DT_FLOAT8_E4M3FN) {
        context->SetOutputDataType(ATTENTION_OUT_INDEX, ge::DT_BF16);
    } else {
        context->SetOutputDataType(ATTENTION_OUT_INDEX, dtype);
    }
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MinimaxSparseAttentionSplitKv)
    .InferShape(InferShapeMinimaxSparseAttentionSplitKv)
    .InferDataType(InferDataTypeMinimaxSparseAttentionSplitKv);

} // namespace ops
