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
 * \file moe_distribute_combine_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "../../common/op_host/mc2_combine_infershape.h"
#include "platform/platform_info.h"
using namespace ge;
namespace ops {
static constexpr size_t COMBINE_INPUT_EXPAND_X_INDEX = 1;
static constexpr size_t COMBINE_INPUT_EXPERT_IDS_INDEX = 2;
static constexpr size_t COMBINE_OUTPUT_X_INDEX = 0;

static ge::graphStatus InferShapeMoeDistributeCombineV3(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeCombineV3.");
    if (Mc2InferShape::InferCombineOutputShape(context, COMBINE_INPUT_EXPAND_X_INDEX, COMBINE_INPUT_EXPERT_IDS_INDEX,
                                               COMBINE_OUTPUT_X_INDEX) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeCombineV3.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeCombineV3(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeCombineV3.");
    Mc2InferShape::InferCombineOutputXDataType(context, COMBINE_INPUT_EXPAND_X_INDEX, COMBINE_OUTPUT_X_INDEX);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeCombineV3.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeCombineV3)
    .InferShape(InferShapeMoeDistributeCombineV3)
    .InferDataType(InferDataTypeMoeDistributeCombineV3);
} // namespace ops
