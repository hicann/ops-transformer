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
 * \file moe_distribute_combine_teardown_infershape.cpp
 * \brief
 */
#include "runtime_util.h"
#include "mc2_log.h"
#include "platform/platform_info.h"

namespace ops {

using namespace ge;

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;

static constexpr size_t COMBINE_INPUT_EXPAND_X_INDEX = 0;
static constexpr size_t COMBINE_INPUT_QUANT_EXPAND_X_INDEX = 1;
static constexpr size_t COMBINE_INPUT_EXPERT_IDS_INDEX = 2;
static constexpr size_t COMBINE_INPUT_EXPAND_IDX_INDEX = 3;
static constexpr size_t COMBINE_INPUT_EXPERT_SCALES_INDEX = 4;
static constexpr size_t COMBINE_INPUT_COMM_CMD_INFO_INDEX = 5;
static constexpr size_t COMBINE_OUTPUT_X_INDEX = 0;
static constexpr size_t COMBINE_INPUT_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t COMBINE_INPUT_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t COMBINE_INPUT_ATTR_MOE_EXPERT_NUM_INDEX = 3;
static constexpr size_t COMBINE_INPUT_ATTR_EXPERT_SHARED_TYPE_INDEX = 4;
static constexpr size_t COMBINE_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX = 5;
static constexpr size_t COMBINE_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
static constexpr size_t COMBINE_INPUT_ATTR_GLOBAL_BS_INDEX = 7;
static constexpr size_t COMBINE_INPUT_ATTR_COMM_QUANT_MODE_INDEX = 8;
static constexpr size_t COMBINE_INPUT_ATTR_QUANT_TYPE_INDEX = 9;

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static ge::graphStatus InferShapeMoeDistributeCombineTeardown(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeCombineTeardown.");
    // 获取输入shape
    const gert::Shape* expandXShape = context->GetInputShape(COMBINE_INPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);

    const gert::Shape* quantExpandXShape = context->GetInputShape(COMBINE_INPUT_QUANT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantExpandXShape);

    const gert::Shape* expertIdsShape = context->GetInputShape(COMBINE_INPUT_EXPERT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);

    const gert::Shape* expandIdxShape = context->GetInputShape(COMBINE_INPUT_EXPAND_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandIdxShape);

    const gert::Shape* expertScalesShape = context->GetInputShape(COMBINE_INPUT_EXPERT_SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertScalesShape);

    const gert::Shape* commCmdInfoShape = context->GetInputShape(COMBINE_INPUT_COMM_CMD_INFO_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commCmdInfoShape);

    gert::Shape* xShape = context->GetOutputShape(COMBINE_OUTPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    const auto epRankId = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);

    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);

    const auto expertSharedType = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_EXPERT_SHARED_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertSharedType);

    const auto sharedExpertNum = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertNum);

    const auto sharedExpertRankNum = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertRankNum);

    const auto globalBs = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, globalBs);

    const auto commQuantMode = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_COMM_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commQuantMode);

    const auto commType = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_QUANT_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commType);

    int64_t bs = ((expertIdsShape->GetDimNum() == 1U) ? NEG_ONE : expertIdsShape->GetDim(0));
    int64_t h = ((expandXShape->GetDimNum() == 1U) ? NEG_ONE : expandXShape->GetDim(1));

    xShape->SetDimNum(DIM_TWO);
    xShape->SetDim(0U, bs);
    xShape->SetDim(1U, h);
    OP_LOGD(context->GetNodeName(), "x shape is :%s after infershape.", Shape2String(*xShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeCombineTeardown.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeCombineTeardown(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeCombineTeardown.");
    auto expandXDtype = context->GetInputDataType(COMBINE_INPUT_EXPAND_X_INDEX);
    context->SetOutputDataType(COMBINE_OUTPUT_X_INDEX, expandXDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeCombineTeardown.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeCombineTeardown)
    .InferShape(InferShapeMoeDistributeCombineTeardown)
    .InferDataType(InferDataTypeMoeDistributeCombineTeardown);
} // namespace ops
