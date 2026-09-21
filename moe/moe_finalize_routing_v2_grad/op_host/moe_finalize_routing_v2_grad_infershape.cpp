/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_finalize_routing_v2_grad_infer.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_0_IDX = 0;
static constexpr size_t INPUT_1_IDX = 1;
static constexpr size_t INPUT_3_IDX = 3;
static constexpr size_t OUTPUT_0_IDX = 0;
static constexpr size_t OUTPUT_1_IDX = 1;
static constexpr size_t ATTR_0_IDX = 0;
static constexpr size_t ATTR_1_IDX = 1;
static constexpr size_t ATTR_2_IDX = 2;
static constexpr size_t ATTR_3_IDX = 3;
static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_NUM_3 = 3;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE = -2;

inline bool IsUnknownRank(const gert::Shape *checkShape)
{
    return checkShape->GetDimNum() == 1 && checkShape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE;
}

static ge::graphStatus GetAttrs(const gert::InferShapeContext *context, int64_t &dropPadMode, int64_t &activeNum,
                                int64_t &expertNum, int64_t &expertCapacity)
{
    dropPadMode = 0;
    activeNum = 0;
    expertNum = 0;
    expertCapacity = 0;
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    if (attrs->GetAttrNum() > ATTR_0_IDX) {
        dropPadMode = *(attrs->GetAttrPointer<int64_t>(ATTR_0_IDX));
    }
    if (attrs->GetAttrNum() > ATTR_1_IDX) {
        activeNum = *(attrs->GetAttrPointer<int64_t>(ATTR_1_IDX));
    }
    if (dropPadMode == 1) {
        OP_CHECK_IF((attrs->GetAttrNum() <= ATTR_3_IDX),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "expert_num and expert_capacity",
                                                          "null", "expert_num and expert_capacity is required"),
                    return ge::GRAPH_FAILED);
        expertNum = *(attrs->GetAttrPointer<int64_t>(ATTR_2_IDX));
        expertCapacity = *(attrs->GetAttrPointer<int64_t>(ATTR_3_IDX));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetGradExpandedXOutputShape(gert::Shape *gradExpandedXShape,
                                                   const gert::Shape *expandedRowIdxShape,
                                                   const gert::Shape *gradYShape, int64_t dropPadMode,
                                                   int64_t activeNum, int64_t expertNum, int64_t expertCapacity)
{
    gradExpandedXShape->SetDimNum(DIM_NUM_2);
    gradExpandedXShape->SetDim(0, expandedRowIdxShape->GetDim(0));
    if (dropPadMode == 0 && activeNum > 0 && activeNum < gradExpandedXShape->GetDim(0)) {
        gradExpandedXShape->SetDim(0, activeNum);
    } else if (dropPadMode == 1) {
        gradExpandedXShape->SetDimNum(DIM_NUM_3);
        gradExpandedXShape->SetDim(0, expertNum);
        gradExpandedXShape->SetDim(1, expertCapacity);
    }
    gradExpandedXShape->SetDim(gradExpandedXShape->GetDimNum() - 1, gradYShape->GetDim(1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetGradScalesOutputShape(const gert::InferShapeContext *context, gert::Shape *gradScalesShape,
                                                const gert::Shape *gradYShape)
{
    gradScalesShape->SetDimNum(DIM_NUM_2);
    gradScalesShape->SetDim(0, gradYShape->GetDim(0));
    gradScalesShape->SetDim(1, 1);
    const gert::Shape *scalesShape = context->GetOptionalInputShape(INPUT_3_IDX);
    if (scalesShape != nullptr && !IsUnknownRank(scalesShape)) {
        OP_CHECK_IF((scalesShape->GetDimNum() != DIM_NUM_2),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "scales",
                                                 (std::to_string(scalesShape->GetDimNum()) + "D").c_str(), "2D"),
                    return ge::GRAPH_FAILED);
        gradScalesShape->SetDim(1, scalesShape->GetDim(1));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeFinalizeRoutingV2GradInferShape(gert::InferShapeContext *context)
{
    const gert::Shape *gradYShape = context->GetInputShape(INPUT_0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYShape);
    const gert::Shape *expandedRowIdxShape = context->GetInputShape(INPUT_1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdxShape);
    gert::Shape *gradExpandedXShape = context->GetOutputShape(OUTPUT_0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradExpandedXShape);
    gert::Shape *gradScalesShape = context->GetOutputShape(OUTPUT_1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradScalesShape);
    if (IsUnknownRank(gradYShape)) {
        *gradExpandedXShape = *gradYShape;
        *gradScalesShape = *gradYShape;
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownRank(expandedRowIdxShape)) {
        *gradExpandedXShape = *expandedRowIdxShape;
        *gradScalesShape = *expandedRowIdxShape;
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF((gradYShape->GetDimNum() != DIM_NUM_2 || expandedRowIdxShape->GetDimNum() != DIM_NUM_1),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "grad_y and expanded_row_idx",
                                                          (std::to_string(gradYShape->GetDimNum()) + "D and " +
                                                           std::to_string(expandedRowIdxShape->GetDimNum()) + "D")
                                                              .c_str(),
                                                          "grad_y must be 2D and expanded_row_idx must be 1D"),
                return ge::GRAPH_FAILED);

    int64_t dropPadMode = 0;
    int64_t activeNum = 0;
    int64_t expertNum = 0;
    int64_t expertCapacity = 0;
    OP_CHECK_IF((GetAttrs(context, dropPadMode, activeNum, expertNum, expertCapacity) != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "GetAttrs", "failed", "GetAttrs failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((SetGradExpandedXOutputShape(gradExpandedXShape, expandedRowIdxShape, gradYShape, dropPadMode,
                                             activeNum, expertNum, expertCapacity) != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SetGradExpandedXOutputShape", "failed",
                                                      "SetGradExpandedXOutputShape failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((SetGradScalesOutputShape(context, gradScalesShape, gradYShape) != ge::GRAPH_SUCCESS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SetGradScalesOutputShape", "failed",
                                                      "SetGradScalesOutputShape failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static graphStatus MoeFinalizeRoutingV2GradInferDtype(gert::InferDataTypeContext *context)
{
    auto gradYtype = context->GetInputDataType(INPUT_0_IDX);
    auto scaleType = context->GetOptionalInputDataType(INPUT_3_IDX);
    auto gradScalesType = (scaleType == ge::DT_UNDEFINED) ? gradYtype : scaleType;
    context->SetOutputDataType(OUTPUT_0_IDX, gradYtype);
    context->SetOutputDataType(OUTPUT_1_IDX, gradScalesType);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeFinalizeRoutingV2Grad)
    .InferShape(MoeFinalizeRoutingV2GradInferShape)
    .InferDataType(MoeFinalizeRoutingV2GradInferDtype);
} // namespace ops
