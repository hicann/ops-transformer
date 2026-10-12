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
 * \file inplace_partial_rotary_mul_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t OUTPUT_X_INDEX = 0;

namespace ops {
static ge::graphStatus InferShapeForInplacePartialRotaryMul(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForInplacePartialRotaryMul.");
    const gert::Shape* xShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    *yShape = *xShape;

    OP_LOGD(context, "End to do InferShapeForInplacePartialRotaryMul.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForInplacePartialRotaryMul(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_X_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InplacePartialRotaryMul)
    .InferShape(InferShapeForInplacePartialRotaryMul)
    .InferDataType(InferDataTypeForInplacePartialRotaryMul);
} // namespace ops
