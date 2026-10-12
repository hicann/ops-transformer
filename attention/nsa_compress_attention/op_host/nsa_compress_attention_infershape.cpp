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
 * \file nsa_compress_attention_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr int SOFTMAXMAX_F32_DIM0SHAPE = 8;
static const uint64_t DIM_NUM_0 = 0;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;

ge::graphStatus InferShapeNsaCompressAttention(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Enter NsaCompressAttention runtime infershape impl.");

    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* queryShape = context->GetInputShape(0); // N2, T, G, D
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape* keyShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape* valueShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShape);
    auto attrs = context->GetAttrs();
    const auto* queryDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const char* inputLayout = attrs->GetAttrPointer<char>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayout);
    std::string inputLayoutStr = std::string(inputLayout);
    for (auto& c : inputLayoutStr) {
        c = toupper(c);
    }
    if (inputLayoutStr != "TND") {
        OP_LOGE(context, "The inputLayout should be TND(case-insensitive), but got %s.", inputLayoutStr.c_str());
        return GRAPH_FAILED;
    }

    int64_t shapeT = queryShape->GetDim(1);
    int64_t shapeN2 = queryShape->GetDim(0);
    int64_t shapeG = queryShape->GetDim(2);

    // softmaxMax, fp32: (B, N, S, 8)
    gert::Shape* softmaxMaxShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxMaxShape);
    // TN8 for fake
    softmaxMaxShape->SetDimNum(DIM_NUM_3);
    softmaxMaxShape->SetDim(DIM_NUM_0, shapeT);
    softmaxMaxShape->SetDim(DIM_NUM_1, shapeN2 * shapeG);
    softmaxMaxShape->SetDim(DIM_NUM_2, SOFTMAXMAX_F32_DIM0SHAPE);

    // softmaxSum, shape same as softmaxMax
    gert::Shape* softmaxSumShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxSumShape);
    *softmaxSumShape = *softmaxMaxShape;

    // attenOut: (N2, T, G, D2)
    gert::Shape* attentionOutShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;
    auto shapeD2 = valueShape->GetDim(2);
    attentionOutShape->SetDim(DIM_NUM_3, shapeD2);

    // topkIndicesOut: (N2, T, selectBlockCount)
    gert::Shape* topkIndicesOutShape = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIndicesOutShape);
    const int* selectBlockCountPtr = attrs->GetAttrPointer<int32_t>(7);
    auto selectBlockCount = *selectBlockCountPtr;
    topkIndicesOutShape->SetDimNum(DIM_NUM_3);
    topkIndicesOutShape->SetDim(DIM_NUM_0, shapeN2);
    topkIndicesOutShape->SetDim(DIM_NUM_1, shapeT);
    topkIndicesOutShape->SetDim(DIM_NUM_2, selectBlockCount);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeNsaCompressAttention(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = context->GetInputDataType(0);
    // softmax_max, outidx:0
    context->SetOutputDataType(0, DT_FLOAT);
    // softmax_sum, outidx:1
    context->SetOutputDataType(1, DT_FLOAT);
    // attention_out, outidx:2
    context->SetOutputDataType(2, dtype);
    // topk_indices_out, outidx:3
    context->SetOutputDataType(3, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaCompressAttention)
    .InferShape(InferShapeNsaCompressAttention)
    .InferDataType(InferDataTypeNsaCompressAttention);

} // namespace ops
