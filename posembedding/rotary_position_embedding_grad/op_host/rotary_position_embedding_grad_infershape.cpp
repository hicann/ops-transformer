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
 * \file rotary_position_embedding_grad.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_GRAD_INDEX = 0;
static constexpr size_t INPUT_COS_INDEX = 1;
static constexpr size_t INPUT_SIN_INDEX = 2;
static constexpr size_t OUTPUT_DX_INDEX = 0;
static constexpr size_t OUTPUT_DCOS_INDEX = 1;
static constexpr size_t OUTPUT_DSIN_INDEX = 2;

static ge::graphStatus InferShapeForRotaryPositionEmbeddingGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForRotaryPositionEmbeddingGrad.");
    const gert::Shape* gradShape = context->GetInputShape(INPUT_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* cosShape = context->GetInputShape(INPUT_COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosShape);
    const gert::Shape* sinShape = context->GetInputShape(INPUT_SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinShape);
    gert::Shape* dxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    gert::Shape* dcosShape = context->GetOutputShape(OUTPUT_DCOS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dcosShape);
    gert::Shape* dsinShape = context->GetOutputShape(OUTPUT_DSIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dsinShape);

    *dxShape = *gradShape;
    *dcosShape = *cosShape;
    *dsinShape = *sinShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForRotaryPositionEmbeddingGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForRotaryPositionEmbeddingGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_DX_INDEX, context->GetInputDataType(INPUT_GRAD_INDEX));
    context->SetOutputDataType(OUTPUT_DCOS_INDEX, context->GetInputDataType(INPUT_COS_INDEX));
    context->SetOutputDataType(OUTPUT_DSIN_INDEX, context->GetInputDataType(INPUT_SIN_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotaryPositionEmbeddingGrad)
    .InferShape(InferShapeForRotaryPositionEmbeddingGrad)
    .InferDataType(InferDataTypeForRotaryPositionEmbeddingGrad);
} // namespace ops
