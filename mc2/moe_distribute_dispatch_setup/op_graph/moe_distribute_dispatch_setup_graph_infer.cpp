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
 * \file moe_distribute_dispatch_setup_graph_infer.cpp
 * \brief InferDataType of MoeDistributeDispatchSetup
 */
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr int64_t DYNAMIC_QUANT_MODE = 2;

static constexpr size_t DISPATCH_INPUT_X_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_Y_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_IDX_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_COMM_CMD_INFO_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX = 7;

static ge::graphStatus InferDataTypeMoeDistributeDispatchSetup(gert::InferDataTypeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeDispatchSetup.");
    auto xDtype = context->GetInputDataType(DISPATCH_INPUT_X_INDEX);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    if (*quantMode == 0) {
        context->SetOutputDataType(DISPATCH_OUTPUT_Y_INDEX, xDtype);
    } else if (*quantMode == DYNAMIC_QUANT_MODE) {
        context->SetOutputDataType(DISPATCH_OUTPUT_Y_INDEX, ge::DT_INT8);
    } else {
        OPS_LOG_E(context->GetNodeName(), "Unsupported quantMode %ld.", *quantMode);
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_IDX_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_COMM_CMD_INFO_INDEX, ge::DT_INT32);
    OPS_LOG_D(context->GetNodeName(), "End to do InferDataTypeMoeDistributeDispatchSetup.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MoeDistributeDispatchSetup).InferDataType(InferDataTypeMoeDistributeDispatchSetup);
} // namespace ops
