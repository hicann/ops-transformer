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
 * \file causal_conv1d_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_CONV_STATES = 2;

static ge::graphStatus InferShapeCausalConv1d(gert::InferShapeContext *context)
{
    const gert::Shape *csShape = context->GetInputShape(IDX_CONV_STATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, csShape);
    gert::Shape *csOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, csOutShape);
    *csOutShape = *csShape;

    const gert::Shape *xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape *yShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeCausalConv1d(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(1, context->GetInputDataType(IDX_0));
    context->SetOutputDataType(0, context->GetInputDataType(IDX_CONV_STATES));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CausalConv1d).InferShape(InferShapeCausalConv1d).InferDataType(InferDataTypeCausalConv1d);
} // namespace ops
