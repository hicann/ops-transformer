/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_fused_causal_conv1d_infershape.cpp
 * \brief
 */

#include <exe_graph/runtime/infer_shape_context.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr int64_t INPUT_X_INDEX = 0;
constexpr int64_t INPUT_CONV_STATES_INDEX = 2;

constexpr int64_t OUTPUT_CONV_STATES_INDEX = 0;
constexpr int64_t OUTPUT_X_INDEX = 1;

ge::graphStatus InferShape4InplaceFusedCausalConv1d(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4InplaceFusedCausalConv1d");

    // inferShape x output: same shape as input x (inplace write-back)
    auto inputXShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXShape);
    auto outputXShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputXShape);
    *outputXShape = *inputXShape;

    // inferShape convStates output: same shape as input convStates
    auto inputConvStatesShape = context->GetInputShape(INPUT_CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputConvStatesShape);
    auto outputConvStatesShape = context->GetOutputShape(OUTPUT_CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputConvStatesShape);
    *outputConvStatesShape = *inputConvStatesShape;

    OP_LOGD(context, "End to do InferShape4InplaceFusedCausalConv1d");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4InplaceFusedCausalConv1d(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4InplaceFusedCausalConv1d");

    // output x dtype: same as input x
    context->SetOutputDataType(OUTPUT_X_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    // output convStates dtype: same as input convStates
    context->SetOutputDataType(OUTPUT_CONV_STATES_INDEX, context->GetInputDataType(INPUT_CONV_STATES_INDEX));

    OP_LOGD(context, "End to do InferDataType4InplaceFusedCausalConv1d");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InplaceFusedCausalConv1d)
    .InferShape(InferShape4InplaceFusedCausalConv1d)
    .InferDataType(InferDataType4InplaceFusedCausalConv1d);

} // namespace ops
