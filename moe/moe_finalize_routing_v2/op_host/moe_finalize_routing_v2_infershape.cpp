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
 * \file moe_finalize_routing_v2_ops.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static const size_t INDEX_IN_EXPANDED_X = 0;
static const size_t INDEX_IN_EXPANDED_ROW_IDX = 1;
static const size_t INDEX_IN_SKIP1 = 2;
static const size_t INDEX_IN_SKIP2 = 3;
static const size_t INDEX_IN_BIAS = 4;
static const size_t INDEX_IN_SCALES = 5;
static const size_t INDEX_IN_EXPERT_IDX = 6;
static constexpr size_t INDEX_OUT = 0;
static constexpr size_t SHAPE_SIZE = 2;
static constexpr size_t INPUT_NUM = 7;
static constexpr size_t ATTR_DROP_PAD_MODE = 0;
static constexpr size_t ATTR_ZERO_EXPERT_RANGE_IDX = 1;
static constexpr size_t ATTR_COPY_EXPERT_RANGE_IDX = 2;
static constexpr size_t ATTR_CONSTANT_EXPERT_RANGE_IDX = 3;
static constexpr size_t ATTR_K = 4;
static constexpr int64_t VALUE_MODE_0 = 0;
static constexpr int64_t VALUE_MODE_1 = 1;
static constexpr int64_t VALUE_MODE_2 = 2;
static constexpr int64_t VALUE_MODE_3 = 3;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;

static inline bool IsValidType(const DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static ge::graphStatus InferDataTypeMoeFinalizeRoutingV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeFinalizeRoutingV2InferDataType.");
    OP_CHECK_IF(!IsValidType(context->GetInputDataType(INDEX_IN_EXPANDED_X)),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "expanded_x",
                                          Ops::Base::ToString(context->GetInputDataType(INDEX_IN_EXPANDED_X)).c_str(),
                                          "FLOAT, FLOAT16 or BF16"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context->GetInputDataType(INDEX_IN_EXPANDED_ROW_IDX) != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(
                    context->GetNodeName(), "expanded_row_idx",
                    Ops::Base::ToString(context->GetInputDataType(INDEX_IN_EXPANDED_ROW_IDX)).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    DataType parameterDtype = context->GetOptionalInputDataType(INDEX_IN_SKIP1);
    OP_CHECK_IF(parameterDtype != ge::DT_UNDEFINED && !IsValidType(parameterDtype),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "skip1", Ops::Base::ToString(parameterDtype).c_str(),
                                          "FLOAT, FLOAT16 or BF16"),
                return ge::GRAPH_FAILED);

    parameterDtype = context->GetOptionalInputDataType(INDEX_IN_SKIP2);
    OP_CHECK_IF(parameterDtype != ge::DT_UNDEFINED && !IsValidType(parameterDtype),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "skip2", Ops::Base::ToString(parameterDtype).c_str(),
                                          "FLOAT, FLOAT16 or BF16"),
                return ge::GRAPH_FAILED);

    parameterDtype = context->GetOptionalInputDataType(INDEX_IN_BIAS);
    OP_CHECK_IF(parameterDtype != ge::DT_UNDEFINED && !IsValidType(parameterDtype),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "bias", Ops::Base::ToString(parameterDtype).c_str(),
                                          "FLOAT, FLOAT16 or BF16"),
                return ge::GRAPH_FAILED);

    parameterDtype = context->GetOptionalInputDataType(INDEX_IN_SCALES);
    OP_CHECK_IF(parameterDtype != ge::DT_UNDEFINED && !IsValidType(parameterDtype),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "scales", Ops::Base::ToString(parameterDtype).c_str(),
                                          "FLOAT, FLOAT16 or BF16"),
                return ge::GRAPH_FAILED);

    parameterDtype = context->GetOptionalInputDataType(INDEX_IN_EXPERT_IDX);
    OP_CHECK_IF(parameterDtype != ge::DT_UNDEFINED && parameterDtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "expert_idx",
                                          Ops::Base::ToString(parameterDtype).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    context->SetOutputDataType(0, context->GetInputDataType(INDEX_IN_EXPANDED_X));
    return ge::GRAPH_SUCCESS;
}

static bool IsValidShape(const int64_t shape1, const int64_t shape2, const int64_t shape3, const int64_t shape4)
{
    std::vector<int64_t> validValue = {-1};
    std::vector<int64_t> currentValue = {shape1, shape2, shape3, shape4};
    for (auto value : currentValue) {
        if (value == -1) {
            continue;
        }
        if (validValue.size() == 1) {
            validValue.push_back(value);
        } else if (validValue[1] != value) {
            return false;
        }
    }
    return true;
}

static inline bool IsUnknownRank(const gert::Shape* check_shape)
{
    return check_shape->GetDimNum() == 1 && check_shape->GetDim(0) == UNKNOWN_RANK_DIM_VALUE_;
}

static inline bool IsUnknownShape(const gert::Shape* check_shape)
{
    for (size_t i = 0; i < check_shape->GetDimNum(); i++) {
        if (check_shape->GetDim(i) == UNKNOWN_DIM_VALUE_) {
            return true;
        }
    }
    return false;
}

inline ge::graphStatus SetAllUnknownDim(gert::Shape* outShape)
{
    outShape->SetDimNum(0);
    outShape->AppendDim(UNKNOWN_DIM_VALUE_);
    outShape->AppendDim(UNKNOWN_DIM_VALUE_);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus SetUnknownRank(gert::Shape* outShape)
{
    outShape->SetDimNum(0);
    outShape->AppendDim(UNKNOWN_RANK_DIM_VALUE_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpandedXShape(const gert::InferShapeContext* context, int64_t dropPadMode,
                                           gert::Shape* outShape, const gert::Shape** outExpandedXShape,
                                           int64_t* outLastDim, bool& earlyReturn)
{
    earlyReturn = false;
    const gert::Shape* expandedXShape = context->GetInputShape(INDEX_IN_EXPANDED_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedXShape);
    if (IsUnknownRank(expandedXShape)) {
        SetUnknownRank(outShape);
        earlyReturn = true;
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownShape(expandedXShape)) {
        SetAllUnknownDim(outShape);
        earlyReturn = true;
        return ge::GRAPH_SUCCESS;
    }
    const size_t expandedXShapeSize = expandedXShape->GetDimNum();
    *outLastDim = expandedXShape->GetDim(expandedXShapeSize - 1);
    *outExpandedXShape = expandedXShape;

    OP_CHECK_IF((dropPadMode == VALUE_MODE_0 || dropPadMode == VALUE_MODE_2) && expandedXShapeSize != SHAPE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "expanded_x", std::to_string(expandedXShapeSize).c_str(),
                    "The expanded_x of input should be 2D tensor when drop_pad_mode is 0 or 2."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((dropPadMode == VALUE_MODE_1 || dropPadMode == VALUE_MODE_3) && expandedXShapeSize != SHAPE_SIZE + 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "expanded_x", std::to_string(expandedXShapeSize).c_str(),
                    "The expanded_x of input should be 3D tensor when drop_pad_mode is 1 or 3."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpandedRowIdxShape(const gert::InferShapeContext* context, gert::Shape* outShape,
                                                const gert::Shape** outRowIdxShape, bool& earlyReturn)
{
    earlyReturn = false;
    const gert::Shape* expandedRowIdxShape = context->GetInputShape(INDEX_IN_EXPANDED_ROW_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdxShape);
    if (IsUnknownRank(expandedRowIdxShape)) {
        SetUnknownRank(outShape);
        earlyReturn = true;
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownShape(expandedRowIdxShape)) {
        SetAllUnknownDim(outShape);
        earlyReturn = true;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(expandedRowIdxShape->GetDimNum() != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "expanded_row_idx",
                                             std::to_string(expandedRowIdxShape->GetDimNum()).c_str(), "1D"),
                return ge::GRAPH_FAILED);
    *outRowIdxShape = expandedRowIdxShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOptionalInputShape(const gert::InferShapeContext* context, size_t inputIndex,
                                               const char* inputName, gert::Shape* outShape,
                                               const gert::Shape** outInputShape, bool& earlyReturn)
{
    earlyReturn = false;
    const gert::Tensor* tensor = context->GetOptionalInputTensor(inputIndex);
    *outInputShape = nullptr;
    if (tensor == nullptr || tensor->GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    *outInputShape = context->GetOptionalInputShape(inputIndex);
    if (IsUnknownRank(*outInputShape)) {
        SetUnknownRank(outShape);
        earlyReturn = true;
        return ge::GRAPH_SUCCESS;
    }
    if (IsUnknownShape(*outInputShape)) {
        SetAllUnknownDim(outShape);
        earlyReturn = true;
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((*outInputShape)->GetDimNum() != SHAPE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), inputName,
                                             std::to_string((*outInputShape)->GetDimNum()).c_str(), "2D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCrossInputConsistency(const gert::InferShapeContext* context, int64_t dropPadMode,
                                                  int64_t lastDimExpandedX, const gert::Shape* expandedXShape,
                                                  const gert::Shape* expandedRowIdxShape,
                                                  const gert::Shape* skip1InputShape,
                                                  const gert::Shape* skip2InputShape, const gert::Shape* biasInputShape,
                                                  const gert::Shape* scalesInputShape,
                                                  const gert::Shape* expertIdxShape)
{
    bool validColK = (scalesInputShape == nullptr || expertIdxShape == nullptr) ||
                     ((scalesInputShape != nullptr && expertIdxShape != nullptr) &&
                      (scalesInputShape->GetDim(1) == -1 || expertIdxShape->GetDim(1) == -1 ||
                       scalesInputShape->GetDim(1) == expertIdxShape->GetDim(1)));
    OP_CHECK_IF(!validColK,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "scales and expert_idx", "invalid",
                                                       "The dim 1 of scales and expert_idx should be same."),
                return ge::GRAPH_FAILED);

    int64_t skip2Row = skip2InputShape != nullptr ? skip2InputShape->GetDim(0) : -1;
    int64_t skip1Row = skip1InputShape != nullptr ? skip1InputShape->GetDim(0) : -1;
    int64_t scaleRow = scalesInputShape != nullptr ? scalesInputShape->GetDim(0) : -1;
    int64_t expertIdxRow = expertIdxShape != nullptr ? expertIdxShape->GetDim(0) : -1;
    OP_CHECK_IF(
        !IsValidShape(skip1Row, skip2Row, scaleRow, expertIdxRow),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "skip1, skip2, scales and expert_idx", "invalid",
                                               "The dim 0 of skip1, skip2, scales and expert_idx should be same."),
        return ge::GRAPH_FAILED);

    int64_t skip2Col = skip2InputShape != nullptr ? skip2InputShape->GetDim(1) : -1;
    int64_t skip1Col = skip1InputShape != nullptr ? skip1InputShape->GetDim(1) : -1;
    int64_t biasCol = biasInputShape != nullptr ? biasInputShape->GetDim(1) : -1;
    OP_CHECK_IF(!IsValidShape(skip1Col, skip2Col, lastDimExpandedX, biasCol),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "skip1, skip2, bias and expanded_x", "invalid",
                    "The dim 1 of skip1, skip2, bias and last dim of expanded_x should be same."),
                return ge::GRAPH_FAILED);

    if (dropPadMode == VALUE_MODE_0 || dropPadMode == VALUE_MODE_2) {
        bool validDim = expandedRowIdxShape->GetDim(0) == -1 || expandedXShape->GetDim(0) == -1 ||
                        (expandedRowIdxShape->GetDim(0) == expandedXShape->GetDim(0));
        OP_CHECK_IF(!validDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "expanded_x and expanded_row_idx", "invalid",
                        "The dim 0 of expanded_x and expanded_row_idx should be same when drop_pad_mode is 0."),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeCopyShapeInput2OutputWithIdx(gert::InferShapeContext* context, int64_t k)
{
    auto outShape = context->GetOutputShape(INDEX_OUT);
    outShape->SetDimNum(0);
    const gert::Shape* expandedXInputShape = context->GetInputShape(INDEX_IN_EXPANDED_X);
    const gert::Shape* expandedSrcToDstRowInputShape = context->GetInputShape(INDEX_IN_EXPANDED_ROW_IDX);
    const gert::Shape* scalesInputShape = context->GetOptionalInputShape(INDEX_IN_SCALES);
    int64_t valueDim0 = expandedSrcToDstRowInputShape->GetDim(0);
    if (scalesInputShape != nullptr) {
        valueDim0 = scalesInputShape->GetDim(0);
    } else if (k > 0) {
        valueDim0 = valueDim0 / k;
    }
    outShape->AppendDim(valueDim0);
    outShape->AppendDim(expandedXInputShape->GetDim(expandedXInputShape->GetDimNum() - 1));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4MoeFinalizeRoutingV2(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dropPadModePtr);
    const int64_t dropPadMode = *dropPadModePtr;

    auto outShape = context->GetOutputShape(INDEX_OUT);
    OP_CHECK_IF(dropPadMode < VALUE_MODE_0 || dropPadMode > VALUE_MODE_3,
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "drop_pad_mode", std::to_string(dropPadMode).c_str(),
                                          "[0,3]"),
                return ge::GRAPH_FAILED);

    bool earlyReturn = false;
    const gert::Shape* expandedXShape = nullptr;
    int64_t lastDimExpandedX = -1;
    ge::graphStatus ret =
        CheckExpandedXShape(context, dropPadMode, outShape, &expandedXShape, &lastDimExpandedX, earlyReturn);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }

    const gert::Shape* expandedRowIdxShape = nullptr;
    ret = CheckExpandedRowIdxShape(context, outShape, &expandedRowIdxShape, earlyReturn);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }

    const gert::Shape* skip1InputShape = nullptr;
    const gert::Shape* skip2InputShape = nullptr;
    const gert::Shape* biasInputShape = nullptr;
    const gert::Shape* scalesInputShape = nullptr;
    const gert::Shape* expertIdxShape = nullptr;

    ret = CheckOptionalInputShape(context, INDEX_IN_SKIP1, "skip1", outShape, &skip1InputShape, earlyReturn);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }
    ret = CheckOptionalInputShape(context, INDEX_IN_SKIP2, "skip2", outShape, &skip2InputShape, earlyReturn);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }
    ret = CheckOptionalInputShape(context, INDEX_IN_BIAS, "bias", outShape, &biasInputShape, earlyReturn);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }
    ret = CheckOptionalInputShape(context, INDEX_IN_SCALES, "scales", outShape, &scalesInputShape, earlyReturn);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }

    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(ATTR_K);
    OP_CHECK_NULL_WITH_CONTEXT(context, kPtr);
    int64_t k = *kPtr;
    if (scalesInputShape != nullptr && !IsUnknownRank(scalesInputShape)) {
        int64_t scalesDim1 = scalesInputShape->GetDim(1);
        // k > 1 means user explicitly specified k, need to verify consistency with scales dim1
        if (*kPtr > 1) {
            OP_CHECK_IF(scalesDim1 != UNKNOWN_DIM_VALUE_ && *kPtr != scalesDim1,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "k", std::to_string(*kPtr).c_str(), "k must equal scales dim1"),
                        return ge::GRAPH_FAILED);
        }
        k = scalesDim1;
    }

    ret = CheckOptionalInputShape(context, INDEX_IN_EXPERT_IDX, "expert_idx", outShape, &expertIdxShape, earlyReturn);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }

    OP_CHECK_IF(CheckCrossInputConsistency(context, dropPadMode, lastDimExpandedX, expandedXShape, expandedRowIdxShape,
                                           skip1InputShape, skip2InputShape, biasInputShape, scalesInputShape,
                                           expertIdxShape) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "infershape", "GRAPH_FAILED",
                                                      "Infershape4MoeFinalizeRoutingV2 failed!"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(MoeCopyShapeInput2OutputWithIdx(context, k) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "infershape", "GRAPH_FAILED",
                                                      "Infershape4MoeFinalizeRoutingV2 failed!"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeFinalizeRoutingV2)
    .InferShape(Infershape4MoeFinalizeRoutingV2)
    .InferDataType(InferDataTypeMoeFinalizeRoutingV2);
} // namespace ops
