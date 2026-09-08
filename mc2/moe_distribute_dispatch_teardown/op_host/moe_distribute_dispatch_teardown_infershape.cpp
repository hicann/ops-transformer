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
 * \file moe_distribute_dispatch_teardown_infershape.cpp
 * \brief
 */
#include <algorithm>
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t ASSIST_INFO_NUM_PER_A = 128;
static constexpr int64_t DYNAMIC_QUANT_MODE = 2;

static constexpr size_t DISPATCH_INPUT_X_INDEX = 0;
static constexpr size_t DISPATCH_INPUT_Y_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_EXPERT_IDS_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_COMM_CMD_INFO_INDEX = 3;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_X_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX = 2;
static constexpr size_t DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX = 4;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX = 5;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX = 7;
static constexpr size_t DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX = 8;
static constexpr size_t DISPATCH_INPUT_ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX = 9;
static constexpr size_t DISPATCH_INPUT_ATTR_COMM_TYPE_INDEX = 10;

static ge::graphStatus InferShapeMoeDistributeDispatchTeardown(gert::InferShapeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferShapeMoeDistributeDispatchTeardown.");
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(DISPATCH_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape *yShape = context->GetInputShape(DISPATCH_INPUT_Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);

    const gert::Shape *expertIdsShape = context->GetInputShape(DISPATCH_INPUT_EXPERT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);

    const gert::Shape *commCmdInfoShape = context->GetInputShape(DISPATCH_INPUT_COMM_CMD_INFO_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commCmdInfoShape);

    gert::Shape *expandXShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);

    gert::Shape *dynamicScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dynamicScalesShape);

    gert::Shape *assistInfoForCombineShape = context->GetOutputShape(DISPATCH_OUTPUT_ASSIST_INFO_FOR_COMBINE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, assistInfoForCombineShape);

    gert::Shape *expertTokenNumsShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    const auto epRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);

    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);

    const auto expertShardType = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertShardType);

    const auto sharedExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertNum);

    const auto sharedExpertRankNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertRankNum);

    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);

    const auto globalBs = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, globalBs);

    const auto expertTokenNums = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EXPERT_TOKEN_NUMS_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokenNums);

    const auto commType = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_COMM_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commType);

    int64_t bs = ((xShape->GetDimNum() == 1U) ? NEG_ONE : xShape->GetDim(0));
    int64_t k = ((expertIdsShape->GetDimNum() == 1U) ? NEG_ONE : expertIdsShape->GetDim(1));
    int64_t h = ((xShape->GetDimNum() == 1U) ? NEG_ONE : xShape->GetDim(1));
    int64_t a;
    int64_t localExpertNum;
    int64_t globalBsReal = (*globalBs == 0) ? (bs * *epWorldSize) : *globalBs;
    if ((globalBsReal < 0) || ((*sharedExpertNum == 0) && (*sharedExpertRankNum != 0)) ||
        (*sharedExpertRankNum >= *epWorldSize)) {
        a = NEG_ONE;
        localExpertNum = NEG_ONE;
    } else {
        if (*epRankId < *sharedExpertRankNum) {
            localExpertNum = 1;
            a = bs * *epWorldSize * *sharedExpertNum / *sharedExpertRankNum;
        } else {
            localExpertNum = *moeExpertNum / (*epWorldSize - *sharedExpertRankNum);
            a = globalBsReal * std::min(localExpertNum, k);
        }
    }

    expandXShape->SetDimNum(DIM_TWO);
    expandXShape->SetDim(0U, a);
    expandXShape->SetDim(1U, h);
    OPS_LOG_D(context->GetNodeName(), "expandx shape is :%s after infershape.",
              Ops::Base::ToString(*expandXShape).c_str());

    dynamicScalesShape->SetDimNum(DIM_ONE);
    dynamicScalesShape->SetDim(0U, a);
    OPS_LOG_D(context->GetNodeName(), "dynamicScales shape is :%s after infershape.",
              Ops::Base::ToString(*dynamicScalesShape).c_str());

    assistInfoForCombineShape->SetDimNum(DIM_ONE);
    assistInfoForCombineShape->SetDim(0U, a * ASSIST_INFO_NUM_PER_A);
    OPS_LOG_D(context->GetNodeName(), "assistInfoForCombine shape is :%s after infershape.",
              Ops::Base::ToString(*assistInfoForCombineShape).c_str());

    expertTokenNumsShape->SetDimNum(DIM_ONE);
    expertTokenNumsShape->SetDim(0U, localExpertNum);
    OPS_LOG_D(context->GetNodeName(), "expertTokenNums shape is :%s after infershape.",
              Ops::Base::ToString(*expertTokenNumsShape).c_str());
    OPS_LOG_D(context->GetNodeName(), "End to do InferShapeMoeDistributeDispatchTeardown.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeDispatchTeardown).InferShape(InferShapeMoeDistributeDispatchTeardown);
} // namespace ops
