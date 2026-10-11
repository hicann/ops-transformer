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
 * \file distribute_barrier_infer.cc
 * \brief
 */
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
using namespace ge;
namespace ops {

static constexpr size_t DIM_ONE = 1UL;

static constexpr size_t BARRIER_INPUT_X_REF_INDEX = 0;
static constexpr size_t BARRIER_OUTPUT_X_REF_INDEX = 0;

static ge::graphStatus InferShapeDistributeBarrier(gert::InferShapeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferShapeDistributeBarrier.");
    // 获取输入shape
    const gert::Shape* xRefInputShape = context->GetInputShape(BARRIER_INPUT_X_REF_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xRefInputShape);
    gert::Shape* xRefOutputShape = context->GetOutputShape(BARRIER_OUTPUT_X_REF_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xRefOutputShape);

    // 这里要获取输入的dim，然后循环给输出赋值

    *xRefOutputShape = *xRefInputShape;
    OPS_LOG_D(context->GetNodeName(), "x_ref shape is :%s after infershape.",
              Ops::Base::ToString(*xRefOutputShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "End to do InferShapeDistributeBarrier.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDistributeBarrier(gert::InferDataTypeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferDataTypeDistributeBarrier.");
    auto xRefDtype = context->GetInputDataType(BARRIER_INPUT_X_REF_INDEX);
    context->SetOutputDataType(BARRIER_OUTPUT_X_REF_INDEX, xRefDtype);
    OPS_LOG_D(context->GetNodeName(), "End to do InferDataTypeDistributeBarrier.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DistributeBarrier)
    .InferShape(InferShapeDistributeBarrier)
    .InferDataType(InferDataTypeDistributeBarrier);
} // namespace ops
