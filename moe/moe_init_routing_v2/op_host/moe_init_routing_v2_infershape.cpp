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
 * \file moe_init_routing_v2_proto.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr int64_t OTHER_SHAPE = -1L;
static constexpr int64_t INDEX_INPUT_X = 0;
static constexpr int64_t INDEX_INPUT_EXPERT_IDX = 1;
static constexpr int64_t OUTOUT_EXPANDED_X = 0;
static constexpr int64_t OUTOUT_EXPANDED_ROW_IDX = 1;
static constexpr int64_t OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2;
static constexpr int64_t OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3;
static constexpr int64_t ATTR_ACTIVE_ROWS = 0;
static constexpr int64_t ATTR_EXPERT_CAPACITY = 1;
static constexpr int64_t ATTR_EXPERT_NUM = 2;
static constexpr int64_t ATTR_DROP_PAD_MODE = 3;
static constexpr int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
static constexpr int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
static constexpr int64_t EXPERT_TOKENS_COUNT = 2;
static constexpr int64_t DROP_LESS = 0;
static constexpr int64_t DROP_PAD = 1;

static bool IsSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 == OTHER_SHAPE || dim2 == OTHER_SHAPE) {
        return true;
    }
    return dim1 == dim2;
}

std::string PrintShape(const gert::Shape& shape)
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

static ge::graphStatus CheckInputShape(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                       const gert::Shape* expertIdxShape)
{
    int64_t xN = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
    if (xN < OTHER_SHAPE || cols < OTHER_SHAPE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape).c_str(),
                                              "The dims of input x should be greater than or equal to -1");
        return ge::GRAPH_FAILED;
    }

    int64_t expertIdxN = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(0);
    int64_t expertIdxK = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);
    if (expertIdxN < OTHER_SHAPE || expertIdxK < OTHER_SHAPE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "expertIdx",
                                              Ops::Base::ToString(*expertIdxShape).c_str(),
                                              "The dims of input expertIdx should be greater than or equal to -1");
        return ge::GRAPH_FAILED;
    }

    if (!IsSameDim(xN, expertIdxN)) {
        std::string shapeMsg = Ops::Base::ToString(*xShape) + " and " + Ops::Base::ToString(*expertIdxShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and expertIdx", shapeMsg.c_str(),
                                               "The first dim of x and expertIdx should be equal");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShapes(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                        const gert::Shape* expertIdxShape)
{
    if (xShape->GetDimNum() == 1U) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape).c_str(),
                                                  "The dynamic dim of x should be -2");
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        std::string xDimNumStr = std::to_string(xShape->GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", xDimNumStr.c_str(), "2D or dynamic");
        return ge::GRAPH_FAILED;
    }

    if (expertIdxShape->GetDimNum() == 1U) {
        if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "expertIdx",
                                                  Ops::Base::ToString(*expertIdxShape).c_str(),
                                                  "The dynamic dim of expertIdx should be -2");
            return ge::GRAPH_FAILED;
        }
    } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
        std::string expertIdxDimNumStr = std::to_string(expertIdxShape->GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "expertIdx", expertIdxDimNumStr.c_str(), "2D or dynamic");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                       const int64_t activeNum, const int64_t expertCapacity, const int64_t expertNum,
                                       const int64_t dropPadMode, const int64_t expertTokensCountOrCumsumFlag)
{
    if (activeNum < 0) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "activeNum", std::to_string(activeNum).c_str(),
                                  "greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    if (expertCapacity < 0) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "expertCapacity", std::to_string(expertCapacity).c_str(),
                                  "greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    if (expertNum < 0) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "expertNum", std::to_string(expertNum).c_str(),
                                  "greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    if (dropPadMode < 0 || dropPadMode > 1) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "dropPadMode", std::to_string(dropPadMode).c_str(), "0 or 1");
        return ge::GRAPH_FAILED;
    }
    if (dropPadMode > 0 && (expertCapacity < 1 || expertNum < 1)) {
        std::string attrValue =
            "expertCapacity: " + std::to_string(expertCapacity) + ", expertNum: " + std::to_string(expertNum);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "expertCapacity and expertNum", attrValue.c_str(),
                                              "expertCapacity and expertNum should be greater than 0 when "
                                              "dropPadMode is 1");
        return ge::GRAPH_FAILED;
    }
    if (expertTokensCountOrCumsumFlag < 0 || expertTokensCountOrCumsumFlag > EXPERT_TOKENS_COUNT) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "expertTokensCountOrCumsumFlag",
                                  std::to_string(expertTokensCountOrCumsumFlag).c_str(), "0, 1 or 2");
        return ge::GRAPH_FAILED;
    }
    if (expertTokensCountOrCumsumFlag > 0 && expertNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "expertNum", std::to_string(expertNum).c_str(),
                                              "expertNum should be greater than 0 when "
                                              "expertTokensCountOrCumsumFlag is greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (dropPadMode > 0 && xShape->GetDim(0) > 0 && expertCapacity > xShape->GetDim(0)) {
        std::string shapeStr = Ops::Base::ToString(*xShape);
        std::string reasonMsg = "The first dim of x cannot be less than expertCapacity when dropPadMode is 1, "
                                "but expertCapacity is " +
                                std::to_string(expertCapacity);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParm(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                 const gert::Shape* expertIdxShape, const int64_t activeNum,
                                 const int64_t expertCapacity, const int64_t expertNum, const int64_t dropPadMode,
                                 const int64_t expertTokensCountOrCumsumFlag)
{
    auto ret = CheckInputShapes(context, xShape, expertIdxShape);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckAttrParams(context, xShape, activeNum, expertCapacity, expertNum, dropPadMode,
                          expertTokensCountOrCumsumFlag);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

static void InferOutputShape(const gert::Shape* xShape, const gert::Shape* expertIdxShape, gert::Shape* expandedXShape,
                             gert::Shape* expandedRowIdx, gert::Shape* expertTokensBeforeCapacityShape,
                             gert::Shape* expertTokensCountOrCumsumShape, const int64_t activeNum,
                             const int64_t expertNum, const int64_t expertCapacity, const int64_t dropPadMode,
                             bool expertTokensBeforeCapacityFlag, const int64_t expertTokensCountOrCumsumFlag)
{
    int64_t n = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
    int64_t k = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);
    int64_t outActiveNum = OTHER_SHAPE;
    int64_t expandedRowIdxNum = OTHER_SHAPE;

    if (n >= 0 && k >= 0) {
        outActiveNum = activeNum > 0 ? std::min(n * k, activeNum) : n * k;
        expandedRowIdxNum = n * k;
    }

    if (dropPadMode == DROP_PAD) {
        expandedXShape->SetDimNum(DIM_THREE);
        expandedXShape->SetDim(0U, expertNum);
        expandedXShape->SetDim(1U, expertCapacity);
        expandedXShape->SetDim(2U, cols < 0 ? OTHER_SHAPE : cols);
    }

    if (dropPadMode == DROP_LESS) {
        expandedXShape->SetDimNum(DIM_TWO);
        expandedXShape->SetDim(0U, outActiveNum);
        expandedXShape->SetDim(1U, cols < 0 ? OTHER_SHAPE : cols);
    }

    expandedRowIdx->SetDimNum(DIM_ONE);
    expandedRowIdx->SetDim(0U, expandedRowIdxNum);

    if (dropPadMode == DROP_PAD && expertTokensBeforeCapacityFlag) {
        expertTokensBeforeCapacityShape->SetDimNum(DIM_ONE);
        expertTokensBeforeCapacityShape->SetDim(0U, expertNum);
    }

    if (dropPadMode == DROP_LESS && expertTokensCountOrCumsumFlag > 0) {
        expertTokensCountOrCumsumShape->SetDimNum(DIM_ONE);
        expertTokensCountOrCumsumShape->SetDim(0U, expertNum);
    }
}

static ge::graphStatus InferShape4MoeInitRoutingV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRountingV2Infershape.");
    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_ROWS);
    const int64_t activeNum = (activeNumPtr == nullptr) ? 0 : *activeNumPtr;
    const int64_t* expertCapacityPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY);
    const int64_t expertCapacity = (expertCapacityPtr == nullptr) ? 0 : *expertCapacityPtr;
    const int64_t* expertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM);
    const int64_t expertNum = (expertNumPtr == nullptr) ? 0 : *expertNumPtr;
    const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
    const int64_t dropPadMode = (dropPadModePtr == nullptr) ? 0 : *dropPadModePtr;
    const int64_t* expertTokensCountOrCumsumFlagPtr =
        attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG);
    const int64_t expertTokensCountOrCumsumFlag =
        (expertTokensCountOrCumsumFlagPtr == nullptr) ? 0 : *expertTokensCountOrCumsumFlagPtr;
    const bool* expertTokensBeforeCapacityFlagPtr =
        attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG);
    const bool expertTokensBeforeCapacityFlag =
        (expertTokensBeforeCapacityFlagPtr == nullptr) ? false : *expertTokensBeforeCapacityFlagPtr;

    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* expertIdxShape = context->GetInputShape(INDEX_INPUT_EXPERT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);
    gert::Shape* expandedXShape = context->GetOutputShape(OUTOUT_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedXShape);
    gert::Shape* expandedRowIdx = context->GetOutputShape(OUTOUT_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdx);
    gert::Shape* expertTokensCountOrCumsumShape = context->GetOutputShape(OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM);
    if (dropPadMode == 0 && expertTokensCountOrCumsumFlag > 0) {
        OP_CHECK_NULL_WITH_CONTEXT(context, expertTokensCountOrCumsumShape);
    }
    gert::Shape* expertTokensBeforeCapacityShape = context->GetOutputShape(OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY);
    if (dropPadMode == 1 && expertTokensBeforeCapacityFlag) {
        OP_CHECK_NULL_WITH_CONTEXT(context, expertTokensBeforeCapacityShape);
    }

    if (CheckInputShape(context, xShape, expertIdxShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckParm(context, xShape, expertIdxShape, activeNum, expertCapacity, expertNum, dropPadMode,
                  expertTokensCountOrCumsumFlag) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InferOutputShape(xShape, expertIdxShape, expandedXShape, expandedRowIdx, expertTokensBeforeCapacityShape,
                     expertTokensCountOrCumsumShape, activeNum, expertNum, expertCapacity, dropPadMode,
                     expertTokensBeforeCapacityFlag, expertTokensCountOrCumsumFlag);
    OP_LOGD(context->GetNodeName(), "End to do MoeInitRountingV2Infershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRountingV2InferDataType.");
    auto xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(OUTOUT_EXPANDED_X, xDtype);
    context->SetOutputDataType(OUTOUT_EXPANDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM, ge::DT_INT32);
    context->SetOutputDataType(OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do MoeInitRountingV2InferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingV2)
    .InferShape(InferShape4MoeInitRoutingV2)
    .InferDataType(InferDataType4MoeInitRoutingV2);
} // namespace ops
