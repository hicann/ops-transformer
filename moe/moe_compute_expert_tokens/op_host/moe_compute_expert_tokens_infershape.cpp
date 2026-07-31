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
 * \file moe_compute_expert_tokens_infer.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
// -------------------MoeComputeExpertTokens Ops START---------------------
static const size_t INDEX_IN_SORTED_EXPERTS_X1 = 0;
static constexpr size_t INDEX_out = 0;
constexpr size_t numExpertsAttrIdx = 0U;

static ge::graphStatus InfershapeForMoeComputeExpertTokens(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeComputeExpertTokensInfershape.");
    // 获取输入值shape
    const gert::Shape *inputShape = context->GetInputShape(INDEX_IN_SORTED_EXPERTS_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    // 获取属性值
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t numExperts = *(attrs->GetInt(numExpertsAttrIdx));
    std::string numExpertsStr = std::to_string(numExperts);
    OP_CHECK_IF(
        numExperts <= 0 || numExperts > 2048,
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "num_experts",
                                  numExpertsStr.c_str(), "greater than 0 and not greater than 2048"),
        return GRAPH_FAILED);

    // 获取输出值shape
    gert::Shape *output_y_shape = context->GetOutputShape(INDEX_out);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_y_shape);

    const size_t input_dim_num = 1;
    output_y_shape->SetDimNum(input_dim_num);
    output_y_shape->SetDim(0, numExperts);

    OP_LOGI(context->GetNodeName(), "output_y_shape = %s.", Ops::Base::ToString(*output_y_shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do MoeComputeExpertTokensInfershape.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeComputeExpertTokens).InferShape(InfershapeForMoeComputeExpertTokens);
// -------------------MoeComputeExpertTokens Ops END---------------------
} // namespace ops
