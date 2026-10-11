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
 * \file engram_fetch_wait_infershape.cpp
 * \brief engram_fetch_wait算子infershape实现
 */

#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
using namespace ge;
namespace ops {

static constexpr size_t FETCH_WAIT_INPUT_FETCHED_INDEX = 1UL;
static constexpr size_t FETCH_WAIT_OUTPUT_FETCHED_INDEX = 0UL;

static ge::graphStatus InferShapeEngramFetchWait(gert::InferShapeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferShapeEngramFetchWait.");
    const gert::Shape* fetchedInputShape = context->GetInputShape(FETCH_WAIT_INPUT_FETCHED_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, fetchedInputShape);
    gert::Shape* fetchedOutputShape = context->GetOutputShape(FETCH_WAIT_OUTPUT_FETCHED_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, fetchedOutputShape);

    *fetchedOutputShape = *fetchedInputShape;
    OPS_LOG_D(context->GetNodeName(), "fetched shape is :%s after infershape.",
              Ops::Base::ToString(*fetchedOutputShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "End to do InferShapeEngramFetchWait.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeEngramFetchWait(gert::InferDataTypeContext* context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferDataTypeEngramFetchWait.");
    auto fetchedDtype = context->GetInputDataType(FETCH_WAIT_INPUT_FETCHED_INDEX);
    context->SetOutputDataType(FETCH_WAIT_OUTPUT_FETCHED_INDEX, fetchedDtype);
    OPS_LOG_D(context->GetNodeName(), "End to do InferDataTypeEngramFetchWait.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramFetchWait).InferShape(InferShapeEngramFetchWait).InferDataType(InferDataTypeEngramFetchWait);
} // namespace ops
