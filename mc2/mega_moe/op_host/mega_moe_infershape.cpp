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
 * \file mega_moe_infershape.cpp
 */

#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "op_host/util/op_const_def.h"

using namespace ge;
namespace ops {

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;

static constexpr size_t MEGA_MOE_INPUT_X_INDEX = 1;
static constexpr size_t MEGA_MOE_OUTPUT_Y_INDEX = 0;
static constexpr size_t MEGA_MOE_OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 1;
static constexpr size_t MEGA_MOE_ATTR_MOE_EXPERT_NUM_INDEX = 0;
static constexpr size_t MEGA_MOE_ATTR_EP_WORLD_SIZE_INDEX = 1;

static ge::graphStatus InferShapeMegaMoe(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMegaMoe.");

    const gert::Shape *xShape = context->GetInputShape(MEGA_MOE_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape *yShape = context->GetOutputShape(MEGA_MOE_OUTPUT_Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape *expertTokenNumsShape = context->GetOutputShape(MEGA_MOE_OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(MEGA_MOE_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(MEGA_MOE_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    OP_CHECK_IF(*epWorldSize <= 0,
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "ep_world_size", std::to_string(*epWorldSize).c_str(),
                                          "smaller than or equal to 0"),
                return ge::GRAPH_FAILED);

    int64_t bs = xShape->GetDim(0);
    int64_t h = xShape->GetDim(1);

    yShape->SetDimNum(DIM_TWO);
    yShape->SetDim(0U, bs);
    yShape->SetDim(1U, h);

    expertTokenNumsShape->SetDimNum(DIM_ONE);
    expertTokenNumsShape->SetDim(0U, *moeExpertNum / *epWorldSize);

    OP_LOGD(context->GetNodeName(), "y shape is [%ld, %ld] after infershape.", bs, h);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMegaMoe.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMegaMoe(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMegaMoe.");

    auto xDtype = context->GetInputDataType(MEGA_MOE_INPUT_X_INDEX);
    const bool isPreQuantizedX =
        xDtype == ge::DT_FLOAT8_E5M2 || xDtype == ge::DT_FLOAT8_E4M3FN || xDtype == ge::DT_FLOAT4_E2M1;
    // 950 预量化 token 输入由 MX GMM 消费，公共输出仍保持 BF16。
    // 对 FP16/BF16 输入保留历史 Arch22 行为。
    context->SetOutputDataType(MEGA_MOE_OUTPUT_Y_INDEX, isPreQuantizedX ? ge::DT_BF16 : xDtype);
    context->SetOutputDataType(MEGA_MOE_OUTPUT_EXPERT_TOKEN_NUMS_INDEX, ge::DT_INT32);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMegaMoe.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MegaMoe).InferShape(InferShapeMegaMoe).InferDataType(InferDataTypeMegaMoe);
} // namespace ops
