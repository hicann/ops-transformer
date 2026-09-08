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
 * \file moe_distribute_dispatch_setup_infershape.cpp
 * \brief
 */
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t RANK_NUM_PER_NODE = 8;
static constexpr int64_t ASSIST_INFO_NUM_PER_A = 128;
static constexpr int64_t COMM_CMD_INFO_BASE = 16;
static constexpr int64_t ALIGN_32 = 32;
static constexpr int64_t ALIGN_256 = 256;
static constexpr int64_t ALIGN_512 = 512;
static constexpr int64_t QUANT_ALIGN_OFFSET = 4;
static constexpr int64_t DYNAMIC_QUANT_MODE = 2;

static constexpr size_t DISPATCH_INPUT_X_INDEX = 0;
static constexpr size_t DISPATCH_INPUT_EXPERT_IDS_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_Y_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_IDX_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_COMM_CMD_INFO_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX = 4;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_NUM_INDEX = 5;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX = 7;
static constexpr size_t DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX = 8;
static constexpr size_t DISPATCH_INPUT_ATTR_COMM_TYPE_INDEX = 9;

inline int64_t Align(int64_t x, int64_t base)
{
    return ((x + base - 1) / base) * base;
}

static ge::graphStatus InferShapeMoeDistributeDispatchSetup(gert::InferShapeContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Begin to do InferShapeMoeDistributeDispatchSetup.");
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(DISPATCH_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape *expertIdsShape = context->GetInputShape(DISPATCH_INPUT_EXPERT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);

    gert::Shape *yShape = context->GetOutputShape(DISPATCH_OUTPUT_Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape *expandIdxShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandIdxShape);

    gert::Shape *commCmdInfoShape = context->GetOutputShape(DISPATCH_OUTPUT_COMM_CMD_INFO_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commCmdInfoShape);

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

    const auto commType = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_COMM_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, commType);

    int64_t bs = ((xShape->GetDimNum() == 1U) ? NEG_ONE : xShape->GetDim(0));
    int64_t h = ((xShape->GetDimNum() == 1U) ? NEG_ONE : xShape->GetDim(1));
    int64_t k = ((expertIdsShape->GetDimNum() == 1U) ? NEG_ONE : expertIdsShape->GetDim(1));
    int64_t hs;
    if (*quantMode == 0) {
        hs = Align(h, ALIGN_256);
    } else {
        hs = Align(Align(h, ALIGN_32) + QUANT_ALIGN_OFFSET, ALIGN_512);
    }
    int64_t localExpertNum;
    int64_t globalBsReal = (*globalBs == 0) ? (bs * *epWorldSize) : *globalBs;
    if ((globalBsReal < 0) || ((*sharedExpertNum == 0) && (*sharedExpertRankNum != 0)) ||
        (*sharedExpertRankNum >= *epWorldSize)) {
        localExpertNum = NEG_ONE;
    } else {
        if (*epRankId < *sharedExpertRankNum) {
            localExpertNum = 1;
        } else {
            localExpertNum = *moeExpertNum / (*epWorldSize - *sharedExpertRankNum);
        }
    }

    yShape->SetDimNum(DIM_TWO);
    yShape->SetDim(0U, bs * (k + *sharedExpertNum));
    yShape->SetDim(1U, hs);
    OPS_LOG_D(context->GetNodeName(), "y shape is :%s after infershape.", Ops::Base::ToString(*yShape).c_str());

    expandIdxShape->SetDimNum(DIM_ONE);
    expandIdxShape->SetDim(0U, bs * k);
    OPS_LOG_D(context->GetNodeName(), "expandIdx shape is :%s after infershape.",
              Ops::Base::ToString(*expandIdxShape).c_str());

    commCmdInfoShape->SetDimNum(DIM_ONE);
    commCmdInfoShape->SetDim(0U, (bs * (k + *sharedExpertNum) + *epWorldSize * localExpertNum) * COMM_CMD_INFO_BASE);
    OPS_LOG_D(context->GetNodeName(), "commCmdInfo shape is :%s after infershape.",
              Ops::Base::ToString(*commCmdInfoShape).c_str());

    OPS_LOG_D(context->GetNodeName(), "End to do InferShapeMoeDistributeDispatchSetup.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDistributeDispatchSetup).InferShape(InferShapeMoeDistributeDispatchSetup);
} // namespace ops
