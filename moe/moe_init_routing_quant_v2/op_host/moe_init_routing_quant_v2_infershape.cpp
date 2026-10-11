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
 * \file moe_init_routing_quant_v2_infershape.cpp
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
static constexpr int64_t OTHER_SHAPE = -1;
static constexpr int64_t INDEX_INPUT_X = 0;
static constexpr int64_t INDEX_INPUT_EXPERT_IDX = 1;
static constexpr int64_t INDEX_INPUT_SCALE = 2;
static constexpr int64_t INDEX_INPUT_OFFSET = 3;
static constexpr int64_t OUTOUT_EXPANDED_X = 0;
static constexpr int64_t OUTOUT_EXPANDED_ROW_IDX = 1;
static constexpr int64_t OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2;
static constexpr int64_t OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3;
static constexpr int64_t OUTOUT_DYNAMIC_QUANT_SCALE = 4;
static constexpr int64_t ATTR_ACTIVE_ROWS = 0;
static constexpr int64_t ATTR_EXPERT_CAPACITY = 1;
static constexpr int64_t ATTR_EXPERT_NUM = 2;
static constexpr int64_t ATTR_DROP_PAD_MODE = 3;
static constexpr int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
static constexpr int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
static constexpr int64_t ATTR_QUANT_MODE = 6;
static constexpr int64_t EXPERT_TOKENS_COUNT = 2;
static constexpr int64_t ONE_BLOCK_NUM = 8;

struct MoeInitRoutingQuantV2InputParam {
    const gert::Shape* xShape;
    const gert::Shape* expertIdxShape;
    gert::Shape* expandedXShape;
    gert::Shape* expandedRowIdx;
    gert::Shape* expertTokensBeforeCapacityShape;
    gert::Shape* expertTokensCountOrCumsumShape;
    gert::Shape* dynamicQuantScaleShape;
    const int64_t activeNum;
    const int64_t expertNum;
    const int64_t expertCapacity;
    const int64_t dropPadMode;
    bool expertTokensBeforeCapacityFlag;
    const int64_t expertTokensCountOrCumsumFlag;
    const int64_t quantMode;
};

static bool isSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 == OTHER_SHAPE || dim2 == OTHER_SHAPE) {
        return true;
    }
    return dim1 == dim2;
}

static ge::graphStatus CheckInputShape(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                       const gert::Shape* expertIdxShape)
{
    int64_t x_n = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
    if (x_n < OTHER_SHAPE || cols < OTHER_SHAPE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape).c_str(),
                                              "The shape of x should not contain a dimension less than -1");
        return ge::GRAPH_FAILED;
    }

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);
    if (expert_idx_n < OTHER_SHAPE || expert_idx_k < OTHER_SHAPE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "expertIdx",
                                              Ops::Base::ToString(*expertIdxShape).c_str(),
                                              "The shape of expertIdx should not contain a dimension less than -1");
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(x_n, expert_idx_n)) {
        std::string shapeMsg = std::to_string(x_n) + " and " + std::to_string(expert_idx_n);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x and expertIdx", shapeMsg.c_str(),
                                               "The first dim of x and expertIdx should be equal");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckParm(const gert::InferShapeContext* context,
                                 MoeInitRoutingQuantV2InputParam moeInitRoutingQuantV2InputParam)
{
    const gert::Shape* xShape = moeInitRoutingQuantV2InputParam.xShape;
    const gert::Shape* expertIdxShape = moeInitRoutingQuantV2InputParam.expertIdxShape;
    const int64_t activeNum = moeInitRoutingQuantV2InputParam.activeNum;
    const int64_t expertNum = moeInitRoutingQuantV2InputParam.expertNum;
    const int64_t expertCapacity = moeInitRoutingQuantV2InputParam.expertCapacity;
    const int64_t dropPadMode = moeInitRoutingQuantV2InputParam.dropPadMode;
    const int64_t expertTokensCountOrCumsumFlag = moeInitRoutingQuantV2InputParam.expertTokensCountOrCumsumFlag;
    const int64_t quantMode = moeInitRoutingQuantV2InputParam.quantMode;
    if (xShape->GetDimNum() == 1U) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(*xShape).c_str(),
                                                  "The dynamic dim of x should be -2");
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        std::string dimStr = std::to_string(xShape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", dimStr.c_str(), "2D or dynamic");
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
        std::string dimStr = std::to_string(expertIdxShape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "expertIdx", dimStr.c_str(), "2D or dynamic");
        return ge::GRAPH_FAILED;
    }
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
    bool dropLessCheck = (dropPadMode > 0 && (expertCapacity < 1 || expertNum < 1));
    if (dropLessCheck) {
        std::string valueMsg = std::to_string(expertCapacity) + " and " + std::to_string(expertNum);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "expertCapacity and expertNum", valueMsg.c_str(),
            "The expertCapacity and expertNum should be greater than 0 when dropPadMode is 1");
        return ge::GRAPH_FAILED;
    }

    bool expertTokensCountOrCumsumFlagCheck =
        (expertTokensCountOrCumsumFlag < 0 || expertTokensCountOrCumsumFlag > EXPERT_TOKENS_COUNT);
    if (expertTokensCountOrCumsumFlagCheck) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "expertTokensCountOrCumsumFlag",
                                  std::to_string(expertTokensCountOrCumsumFlag).c_str(), "0, 1 or 2");
        return ge::GRAPH_FAILED;
    }
    bool expertNumCheck = (expertTokensCountOrCumsumFlag > 0 && expertNum <= 0);
    if (expertNumCheck) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "expertNum", std::to_string(expertNum).c_str(),
            "The expertNum should be greater than 0 when expertTokensCountOrCumsumFlag is greater than 0");
        return ge::GRAPH_FAILED;
    }
    bool tokenNumCheck = (dropPadMode > 0 && xShape->GetDim(0) > 0 && expertCapacity > xShape->GetDim(0));
    if (tokenNumCheck) {
        std::string shapeMsg = std::to_string(xShape->GetDim(0)) + " and " + std::to_string(expertCapacity);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x and expertCapacity", shapeMsg.c_str(),
            "The first dim of x cannot be less than expertCapacity when dropPadMode is 1");
        return ge::GRAPH_FAILED;
    }
    bool quantModeCheck = (quantMode < 0 || quantMode > 1);
    if (quantModeCheck) {
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "quantMode", std::to_string(quantMode).c_str(), "0 or 1");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleOffset(gert::InferShapeContext* context, const gert::Shape* shape, const char* tag)
{
    if (shape->GetDimNum() == 1U) {
        if (shape->GetDim(0) != ge::UNKNOWN_DIM_NUM && shape->GetDim(0) != OTHER_SHAPE && shape->GetDim(0) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), tag, Ops::Base::ToString(*shape).c_str(),
                                                  "The shape dim should be -2, -1 or 1");
            return ge::GRAPH_FAILED;
        }
    } else {
        std::string dimStr = std::to_string(shape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), tag, dimStr.c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleOffsetInput(gert::InferShapeContext* context, const int64_t quantMode,
                                             const int64_t expertNum, const int64_t cols)
{
    const gert::Shape* scaleShape = context->GetOptionalInputShape(INDEX_INPUT_SCALE);
    if (quantMode == 0) {
        OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
        if (CheckScaleOffset(context, scaleShape, "scale") == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }

        const gert::Shape* offsetShape = context->GetInputShape(INDEX_INPUT_OFFSET);
        OP_CHECK_NULL_WITH_CONTEXT(context, offsetShape);
        if (CheckScaleOffset(context, offsetShape, "offset") == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (nullptr == scaleShape) {
            return ge::GRAPH_SUCCESS;
        }
        if (scaleShape->GetDimNum() == 1U) {
            if (scaleShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "scale",
                                                      Ops::Base::ToString(*scaleShape).c_str(),
                                                      "The dynamic dim of scale should be -2");
                return ge::GRAPH_FAILED;
            }
        } else if (scaleShape->GetDimNum() != DIM_TWO) {
            std::string dimStr = std::to_string(scaleShape->GetDimNum()) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "scale", dimStr.c_str(), "2D or dynamic");
            return ge::GRAPH_FAILED;
        }
        int64_t fdm = scaleShape->GetDimNum() == 1U ? OTHER_SHAPE : scaleShape->GetDim(0);
        int64_t sdm = scaleShape->GetDimNum() == 1U ? OTHER_SHAPE : scaleShape->GetDim(1);
        if (fdm < OTHER_SHAPE || sdm < OTHER_SHAPE) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "scale",
                                                  Ops::Base::ToString(*scaleShape).c_str(),
                                                  "The shape of scale should not contain a dimension less than -1");
            return ge::GRAPH_FAILED;
        }
        if (fdm != OTHER_SHAPE && fdm != 1 && fdm != expertNum) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "scale",
                                                  Ops::Base::ToString(*scaleShape).c_str(),
                                                  "The first dim of scale should be -1, 1 or expertNum");
            return ge::GRAPH_FAILED;
        }
        if (sdm != OTHER_SHAPE && cols != OTHER_SHAPE && sdm != cols) {
            std::string shapeMsg = std::to_string(sdm) + " and " + std::to_string(cols);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "scale and x", shapeMsg.c_str(),
                                                   "The second dim of scale and x should be equal");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static void InferOutputShape(MoeInitRoutingQuantV2InputParam moeInitRoutingQuantV2InputParam)
{
    const gert::Shape* xShape = moeInitRoutingQuantV2InputParam.xShape;
    const gert::Shape* expertIdxShape = moeInitRoutingQuantV2InputParam.expertIdxShape;
    gert::Shape* expandedXShape = moeInitRoutingQuantV2InputParam.expandedXShape;
    gert::Shape* expandedRowIdx = moeInitRoutingQuantV2InputParam.expandedRowIdx;
    gert::Shape* expertTokensBeforeCapacityShape = moeInitRoutingQuantV2InputParam.expertTokensBeforeCapacityShape;
    gert::Shape* expertTokensCountOrCumsumShape = moeInitRoutingQuantV2InputParam.expertTokensCountOrCumsumShape;
    gert::Shape* dynamicQuantScaleShape = moeInitRoutingQuantV2InputParam.dynamicQuantScaleShape;
    const int64_t activeNum = moeInitRoutingQuantV2InputParam.activeNum;
    const int64_t expertNum = moeInitRoutingQuantV2InputParam.expertNum;
    const int64_t expertCapacity = moeInitRoutingQuantV2InputParam.expertCapacity;
    const int64_t dropPadMode = moeInitRoutingQuantV2InputParam.dropPadMode;
    bool expertTokensBeforeCapacityFlag = moeInitRoutingQuantV2InputParam.expertTokensBeforeCapacityFlag;
    const int64_t expertTokensCountOrCumsumFlag = moeInitRoutingQuantV2InputParam.expertTokensCountOrCumsumFlag;
    const int64_t quantMode = moeInitRoutingQuantV2InputParam.quantMode;
    int64_t n = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
    int64_t k = expertIdxShape->GetDimNum() == 1U ? OTHER_SHAPE : expertIdxShape->GetDim(1);
    int64_t outActiveNum = OTHER_SHAPE;
    int64_t expandedRowIdxNum = OTHER_SHAPE;

    if (n > 0 && k > 0) {
        expandedRowIdxNum = n * k;
        outActiveNum = activeNum > 0 ? std::min(n * k, activeNum) : n * k;
    }

    if (dropPadMode > 0) {
        expandedXShape->SetDimNum(DIM_THREE);
        expandedXShape->SetDim(0U, expertNum);
        expandedXShape->SetDim(1U, expertCapacity);
        expandedXShape->SetDim(2U, cols < 0 ? OTHER_SHAPE : cols);
    } else {
        expandedXShape->SetDimNum(DIM_TWO);
        expandedXShape->SetDim(0U, outActiveNum);
        expandedXShape->SetDim(1U, cols < 0 ? OTHER_SHAPE : cols);
    }

    expandedRowIdx->SetDimNum(DIM_ONE);
    expandedRowIdx->SetDim(0U, expandedRowIdxNum);

    if (dropPadMode == 1 && expertTokensBeforeCapacityFlag) {
        expertTokensBeforeCapacityShape->SetDimNum(DIM_ONE);
        expertTokensBeforeCapacityShape->SetDim(0U, expertNum);
    }

    if (dropPadMode == 0 && expertTokensCountOrCumsumFlag > 0) {
        expertTokensCountOrCumsumShape->SetDimNum(DIM_ONE);
        expertTokensCountOrCumsumShape->SetDim(0U, expertNum);
    }

    if (quantMode == 1) {
        dynamicQuantScaleShape->SetDimNum(DIM_ONE);
        int64_t fdim = dropPadMode == 0 ? outActiveNum : expertNum * expertCapacity;
        dynamicQuantScaleShape->SetDim(0U, fdim);
    }
}

static ge::graphStatus InferShape4MoeInitRoutingQuantV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRountingQuantV2Infershape.");
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
    const int64_t* quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE);
    const int64_t quantMode = (quantModePtr == nullptr) ? 0 : *quantModePtr;

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
    gert::Shape* dynamicQuantScaleShape = context->GetOutputShape(OUTOUT_DYNAMIC_QUANT_SCALE);
    if (quantMode == 1) {
        OP_CHECK_NULL_WITH_CONTEXT(context, dynamicQuantScaleShape);
    }

    if (CheckInputShape(context, xShape, expertIdxShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    MoeInitRoutingQuantV2InputParam moeInitRoutingQuantV2InputParam = {xShape,
                                                                       expertIdxShape,
                                                                       expandedXShape,
                                                                       expandedRowIdx,
                                                                       expertTokensBeforeCapacityShape,
                                                                       expertTokensCountOrCumsumShape,
                                                                       dynamicQuantScaleShape,
                                                                       activeNum,
                                                                       expertNum,
                                                                       expertCapacity,
                                                                       dropPadMode,
                                                                       expertTokensBeforeCapacityFlag,
                                                                       expertTokensCountOrCumsumFlag,
                                                                       quantMode};

    if (CheckParm(context, moeInitRoutingQuantV2InputParam) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t cols = xShape->GetDimNum() == 1U ? OTHER_SHAPE : xShape->GetDim(1);
    if (CheckScaleOffsetInput(context, quantMode, expertNum, cols) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    InferOutputShape(moeInitRoutingQuantV2InputParam);

    OP_LOGD(context->GetNodeName(), "End to do MoeInitRountingQuantV2Infershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRoutingQuantV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRountingQuantV2InferDataType.");
    context->SetOutputDataType(OUTOUT_EXPANDED_X, ge::DT_INT8);
    context->SetOutputDataType(OUTOUT_EXPANDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM, ge::DT_INT32);
    context->SetOutputDataType(OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY, ge::DT_INT32);
    context->SetOutputDataType(OUTOUT_DYNAMIC_QUANT_SCALE, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do MoeInitRountingQuantV2InferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRoutingQuantV2)
    .InferShape(InferShape4MoeInitRoutingQuantV2)
    .InferDataType(InferDataType4MoeInitRoutingQuantV2);
} // namespace ops
