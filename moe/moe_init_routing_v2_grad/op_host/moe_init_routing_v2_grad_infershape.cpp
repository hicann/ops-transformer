/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v2_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t DROPLESS = 0;
static constexpr int64_t POSITION_DROP_AND_PAD_0 = 1;

template <typename T>
std::string Shape2String4Moe(const T& shape) {
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

static ge::graphStatus CheckParamShape(
    gert::InferShapeContext* context, const gert::Shape* checkShape, const size_t expectDimNum)
{
    if (checkShape->GetDimNum() == 1U) {
        if (expectDimNum != 1 && checkShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE(
                context->GetNodeName(), "The dynamic dim of input should be -2, current shape is %s.",
                Shape2String4Moe(*checkShape).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (checkShape->GetDimNum() != expectDimNum) {
        OP_LOGE(
            context->GetNodeName(), "The dim of input should be %zu or dynamic, current shape is %s.", expectDimNum,
            Shape2String4Moe(*checkShape).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParm(
    gert::InferShapeContext* context, const gert::Shape* gradExpandedXShape, const gert::Shape* expandedRowIdxShape,
    const int64_t dropPadMode, const int64_t topK, const int64_t activeNum)
{
    // attr drop_pad_mode
    if (dropPadMode != DROPLESS && dropPadMode != POSITION_DROP_AND_PAD_0) {
        OP_LOGE(context->GetNodeName(), "drop_pad_mode should be dropless or drop/pad.");
        return ge::GRAPH_FAILED;
    }

    // attr top_k
    if (topK <= 0) {
        OP_LOGE(context->GetNodeName(), "top_k must large than zero.");
        return ge::GRAPH_FAILED;
    }

    // attr active_num
    if (activeNum < 0) {
        OP_LOGE(context->GetNodeName(), "active_num must large than zero.");
        return ge::GRAPH_FAILED;
    }

    // grad_expanded_x
    size_t expectDimNum = (dropPadMode == DROPLESS) ? DIM_TWO : DIM_THREE;
    if (CheckParamShape(context, gradExpandedXShape, expectDimNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "grad_expanded_x shape dims invalid.");
        return ge::GRAPH_FAILED;
    }

    // expanded_row_idx
    if (CheckParamShape(context, expandedRowIdxShape, DIM_ONE) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "expanded_row_idx shape dims invalid.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4MoeInitRoutingV2Grad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRountingGradInfershape.");
    // 获取输入shape
    const gert::Shape* gradExpandedXShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradExpandedXShape);
    const gert::Shape* expandedRowIdxShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdxShape);
    gert::Shape* gradXShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradXShape);

    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* topKPtr = attrs->GetAttrPointer<int64_t>(0); // 0: top_k attr
    OP_CHECK_NULL_WITH_CONTEXT(context, topKPtr);
    const int64_t topK = *topKPtr;
    const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(1); // 1: drop_pad_mode attr
    const int64_t dropPadMode = (dropPadModePtr == nullptr) ? 0 : *dropPadModePtr;
    const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(2); // 2: active_num attr
    const int64_t activeNum = (activeNumPtr == nullptr) ? 0 : *activeNumPtr;

    // 参数校验
    if (CheckParm(context, gradExpandedXShape, expandedRowIdxShape, dropPadMode, topK, activeNum) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 获取入参shape轴信息，并比较入参轴关系
    int64_t gradExpandedXDim1 =
        (gradExpandedXShape->GetDimNum() == 1U) ?
            NEG_ONE :
            ((dropPadMode == 1) ? gradExpandedXShape->GetDim(2) : gradExpandedXShape->GetDim(1));
    int64_t expandedRowIdxDim0 =
        (expandedRowIdxShape->GetDim(0) == ge::UNKNOWN_DIM_NUM) ? NEG_ONE : expandedRowIdxShape->GetDim(0);

    // 设置输出shape
    int64_t gradXDim0 = (expandedRowIdxDim0 == NEG_ONE) ? NEG_ONE : expandedRowIdxDim0 / topK;
    int64_t gradXDim1 = gradExpandedXDim1;

    gradXShape->SetDimNum(DIM_TWO);
    gradXShape->SetDim(0U, gradXDim0);
    gradXShape->SetDim(1U, gradXDim1);

    OP_LOGD(context->GetNodeName(), "End to do MoeInitRountingGradInfershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingV2Grad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRountingGradInferDataType.");
    auto xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    OP_LOGD(context->GetNodeName(), "End to do MoeInitRountingGradInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingV2Grad)
    .InferShape(InferShape4MoeInitRoutingV2Grad)
    .InferDataType(InferDataType4MoeInitRoutingV2Grad);
} // namespace ops
