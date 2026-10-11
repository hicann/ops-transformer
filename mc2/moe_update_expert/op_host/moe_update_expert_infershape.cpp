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
 * \file moe_update_expert_infershape.cpp
 * \brief
 */

#include "mc2_log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
using namespace ge;

namespace ops {
static constexpr size_t IDX_IN_EXPERT_IDS = 0;
static constexpr size_t IDX_IN_EPLB_TABLE = 1;

static constexpr size_t IDX_ATTR_LOCAL_RANK_ID = 0;
static constexpr size_t IDX_ATTR_WORLD_SIZE = 1;
static constexpr size_t IDX_ATTR_BALANCE_MODE = 2;

static constexpr size_t IDX_OUT_EXPERT_IDS = 0;
static constexpr size_t IDX_OUT_ACTIVE_MASK = 1;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;

static constexpr size_t DIM_NUM_2 = 2;

static ge::graphStatus CheckDims(const gert::InferShapeContext* context, const gert::Shape* expertIdsShape,
                                 const gert::Shape* eplbTableShape)
{
    auto result = ge::GRAPH_SUCCESS;

    auto expertIdsDimNum = expertIdsShape->GetDimNum();
    auto eplbTableDimNum = eplbTableShape->GetDimNum();
    if (expertIdsDimNum != DIM_NUM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "expert_ids", std::to_string(expertIdsDimNum).c_str(),
                                     "2");
        result = ge::GRAPH_FAILED;
    }
    if (eplbTableDimNum != DIM_NUM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "eplb_table", std::to_string(eplbTableDimNum).c_str(),
                                     "2");
        result = ge::GRAPH_FAILED;
    }
    return result;
}

static ge::graphStatus InferShapeMoeUpdateExpert(gert::InferShapeContext* context)
{
    auto* expertIdsShape = context->GetInputShape(IDX_IN_EXPERT_IDS);
    auto* eplbTableShape = context->GetInputShape(IDX_IN_EPLB_TABLE);
    auto* balancedExpertIdsShape = context->GetOutputShape(IDX_OUT_EXPERT_IDS);
    auto* balancedActiveMaskShape = context->GetOutputShape(IDX_OUT_ACTIVE_MASK);

    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, eplbTableShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, balancedExpertIdsShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, balancedActiveMaskShape);
    OP_CHECK_IF(CheckDims(context, expertIdsShape, eplbTableShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "%s", "checkDims failed"), return ge::GRAPH_FAILED);

    int64_t BS = expertIdsShape->GetDim(DIM_0);
    int64_t K = expertIdsShape->GetDim(DIM_1);
    balancedExpertIdsShape->SetDimNum(DIM_NUM_2);
    balancedExpertIdsShape->SetDim(DIM_0, BS);
    balancedExpertIdsShape->SetDim(DIM_1, K);
    balancedActiveMaskShape->SetDimNum(DIM_NUM_2);
    balancedActiveMaskShape->SetDim(DIM_0, BS);
    balancedActiveMaskShape->SetDim(DIM_1, K);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeUpdateExpert(gert::InferDataTypeContext* context)
{
    auto dType = context->GetInputDataType(IDX_IN_EXPERT_IDS);
    context->SetOutputDataType(IDX_OUT_EXPERT_IDS, dType);
    context->SetOutputDataType(IDX_OUT_ACTIVE_MASK, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeUpdateExpert).InferShape(InferShapeMoeUpdateExpert).InferDataType(InferDataTypeMoeUpdateExpert);
} // namespace ops
