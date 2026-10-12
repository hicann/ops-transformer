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
 * \file masked_causal_conv1d_infershape.cpp
 */

#include <exe_graph/runtime/infer_shape_context.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr int64_t INPUT_X_INDEX = 0;
constexpr int64_t OUTPUT_Y_INDEX = 0;

ge::graphStatus InferShape4MaskedCausalConv1d(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin InferShape4MaskedCausalConv1d");
    auto inputXShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXShape);
    auto outputYShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYShape);
    *outputYShape = *inputXShape;
    OP_LOGD(context, "End InferShape4MaskedCausalConv1d");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4MaskedCausalConv1d(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaskedCausalConv1d)
    .InferShape(InferShape4MaskedCausalConv1d)
    .InferDataType(InferDataType4MaskedCausalConv1d);

} // namespace ops
