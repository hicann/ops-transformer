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
 * \file moe_distribute_combine_setup_infershape.cpp
 * \brief
 */
#include "runtime_util.h"
#include "mc2_log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t RANK_NUM_PER_NODE = 8;
static constexpr int64_t ASSIST_INFO_NUM_PER_A = 128;
static constexpr int64_t COMM_CMD_INFO_BASE = 16;
static constexpr int64_t ALIGN_8 = 8;
static constexpr int64_t ALIGN_32 = 32;
static constexpr int64_t ALIGN_512 = 512;

static constexpr size_t COMBINE_INPUT_EXPAND_X_INDEX = 0;
static constexpr size_t COMBINE_INPUT_EXPERT_IDS_INDEX = 1;
static constexpr size_t COMBINE_INPUT_ASSIST_INFO_FOR_COMBINE_INDEX = 2;
static constexpr size_t COMBINE_OUTPUT_QUANT_EXPAND_X_INDEX = 0;
static constexpr size_t COMBINE_OUTPUT_COMM_CMD_INFO_INDEX = 1;
static constexpr size_t COMBINE_INPUT_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t COMBINE_INPUT_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t COMBINE_INPUT_ATTR_MOE_EXPERT_NUM_INDEX = 3;
static constexpr size_t COMBINE_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX = 4;
static constexpr size_t COMBINE_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX = 5;
static constexpr size_t COMBINE_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
static constexpr size_t COMBINE_INPUT_ATTR_GLOBAL_BS_INDEX = 7;
static constexpr size_t COMBINE_INPUT_ATTR_COMM_QUANT_MODE_INDEX = 8;
static constexpr size_t COMBINE_INPUT_ATTR_COMM_TYPE_INDEX = 9;

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

inline int64_t Align(int64_t x, int64_t base)
{
    if (base == 0) {
        OP_LOGD("Align: base cannot be zero");
        return 0;
    }
    return ((x + base - 1) / base) * base;
}

static ge::graphStatus InferShapeMoeDistributeCombineSetup(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeCombineSetup.");
    // 获取输入shape
    const gert::Shape* expandXShape = context->GetInputShape(COMBINE_INPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);

    const gert::Shape* expertIdsShape = context->GetInputShape(COMBINE_INPUT_EXPERT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);

    const gert::Shape* assistInfoForCombineShape = context->GetInputShape(COMBINE_INPUT_ASSIST_INFO_FOR_COMBINE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, assistInfoForCombineShape);

    gert::Shape* quantExpandXShape = context->GetOutputShape(COMBINE_OUTPUT_QUANT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantExpandXShape);

    gert::Shape* commCmdInfoShape = context->GetOutputShape(COMBINE_OUTPUT_COMM_CMD_INFO_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commCmdInfoShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    const auto epRankId = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);

    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);

    const auto expertShardType = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertShardType);

    const auto sharedExpertNum = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertNum);

    const auto sharedExpertRankNum = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertRankNum);

    const auto globalBs = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, globalBs);

    const auto commQuantMode = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_COMM_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commQuantMode);

    const auto commType = attrs->GetAttrPointer<int64_t>(COMBINE_INPUT_ATTR_COMM_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commType);

    int64_t a = ((expandXShape->GetDimNum() == 1U) ? NEG_ONE : expandXShape->GetDim(0));
    int64_t h = ((expandXShape->GetDimNum() == 1U) ? NEG_ONE : expandXShape->GetDim(1));
    int64_t hs = Align(Align(h, ALIGN_32) + Align(h, ALIGN_8) / ALIGN_8 * sizeof(float), ALIGN_512);

    quantExpandXShape->SetDimNum(DIM_TWO);
    quantExpandXShape->SetDim(0U, a);
    quantExpandXShape->SetDim(1U, hs);
    OP_LOGD(context->GetNodeName(), "quantExpandX shape is :%s after infershape.",
            Shape2String(*quantExpandXShape).c_str());

    commCmdInfoShape->SetDimNum(DIM_ONE);
    commCmdInfoShape->SetDim(0U, (a + *epWorldSize) * COMM_CMD_INFO_BASE);
    OP_LOGD(context->GetNodeName(), "commCmdInfo shape is :%s after infershape.",
            Shape2String(*commCmdInfoShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeCombineSetup.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeCombineSetup(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeCombineSetup.");
    context->SetOutputDataType(COMBINE_OUTPUT_QUANT_EXPAND_X_INDEX, ge::DT_INT8);
    context->SetOutputDataType(COMBINE_OUTPUT_COMM_CMD_INFO_INDEX, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeCombineSetup.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeCombineSetup)
    .InferShape(InferShapeMoeDistributeCombineSetup)
    .InferDataType(InferDataTypeMoeDistributeCombineSetup);
} // namespace ops
