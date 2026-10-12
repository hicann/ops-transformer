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
 * \file ring_attention_update_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

static constexpr size_t INPUT_ATTN = 0;
static constexpr size_t INPUT_SOFTMAX_MAX = 1;
static constexpr size_t INPUT_SOFTMAX_SUM = 2;
static constexpr size_t OUTPUT_ATTN = 0;
static constexpr size_t OUTPUT_SOFTMAX_MAX = 1;
static constexpr size_t OUTPUT_SOFTMAX_SUM = 2;

static graphStatus InferShape4RingAttentionUpdate(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4RingAttentionUpdate");
    // get input shape
    const gert::Shape* inputAttnShape = context->GetInputShape(INPUT_ATTN);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputAttnShape);
    const gert::Shape* inputSoftmaxMax = context->GetInputShape(INPUT_SOFTMAX_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSoftmaxMax);
    const gert::Shape* inputSoftmaxSum = context->GetInputShape(INPUT_SOFTMAX_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSoftmaxSum);
    // get output shape
    gert::Shape* outputAttnShape = context->GetOutputShape(OUTPUT_ATTN);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputAttnShape);
    gert::Shape* outputSoftmaxMax = context->GetOutputShape(OUTPUT_SOFTMAX_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputSoftmaxMax);
    gert::Shape* outputSoftmaxSum = context->GetOutputShape(OUTPUT_SOFTMAX_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputSoftmaxSum);
    // infer shape
    *outputAttnShape = *inputAttnShape;
    *outputSoftmaxMax = *inputSoftmaxMax;
    *outputSoftmaxSum = *inputSoftmaxSum;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4RingAttentionUpdate");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4RingAttentionUpdate(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4RingAttentionUpdate");
    // infer shape
    context->SetOutputDataType(OUTPUT_ATTN, context->GetInputDataType(INPUT_ATTN));
    context->SetOutputDataType(OUTPUT_SOFTMAX_MAX, context->GetInputDataType(INPUT_SOFTMAX_MAX));
    context->SetOutputDataType(OUTPUT_SOFTMAX_SUM, context->GetInputDataType(INPUT_SOFTMAX_SUM));

    OP_LOGD(context->GetNodeName(), "End to do InferDataType4RingAttentionUpdate");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RingAttentionUpdate)
    .InferShape(InferShape4RingAttentionUpdate)
    .InferDataType(InferDataType4RingAttentionUpdate);
} // namespace ops
