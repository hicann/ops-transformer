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
 * \file moe_distribute_dispatch_teardown_graph_infer.cpp
 * \brief InferDataType of MoeDistributeDispatchTeardown
 */
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr int64_t DYNAMIC_QUANT_MODE = 2;

static constexpr size_t DISPATCH_INPUT_Y_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_X_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX = 2;
static constexpr size_t DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX = 7;

static ge::graphStatus InferDataTypeMoeDistributeDispatchTeardown(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeDispatchTeardown.");
    auto yDtype = context->GetInputDataType(DISPATCH_INPUT_Y_INDEX);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    if (*quantMode == 0) {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, yDtype);
    } else if (*quantMode == DYNAMIC_QUANT_MODE) {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, ge::DT_INT8);
    } else {
        OPS_LOG_E(context->GetNodeName(), "Unsupported quantMode %ld.", *quantMode);
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(DISPATCH_OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX, ge::DT_INT64);
    OPS_LOG_D(context->GetNodeName(), "End to do InferDataTypeMoeDistributeDispatchTeardown.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeDispatchTeardown).InferDataType(InferDataTypeMoeDistributeDispatchTeardown);
} // namespace ops
