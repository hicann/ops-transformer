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
#include <string>
#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "common/include/op_host/op_const_def_transformer.h"

using namespace ge;
namespace ops {
static const size_t INDEX_IN_EXPANDED_X = 0;
static const size_t INDEX_IN_EXPANDED_ROW_IDX = 1;
static const size_t INDEX_IN_SKIP1 = 2;
static const size_t INDEX_IN_SKIP2 = 3;
static const size_t INDEX_IN_BIAS = 4;
static const size_t INDEX_IN_SCALES = 5;
static const size_t INDEX_IN_EXPERT_IDX = 6;
static constexpr size_t INDEX_IN_X = 7;
static constexpr size_t INDEX_IN_V = 10;
static constexpr size_t INDEX_IN_EXPANDED_X_SCALE = 11;
static constexpr int64_t MXFP4_BLOCK_SIZE = 64;
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
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 ||
           dtype == Ops::Transformer::DT_HIFLOAT4;
}

static bool CheckOptionalAuxDtype(const gert::InferDataTypeContext* context, size_t index, const char* name,
                                  DataType expectedMx4Dtype, bool isMx4)
{
    const auto dtype = context->GetOptionalInputDataType(index);
    OP_CHECK_IF(
        dtype != ge::DT_UNDEFINED && (isMx4 ? dtype != expectedMx4Dtype : !IsValidType(dtype)),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), name, Ops::Base::ToString(dtype).c_str(),
                                  isMx4 ? Ops::Base::ToString(expectedMx4Dtype).c_str() : "FLOAT, FLOAT16 or BF16"),
        return false);
    return true;
}

static ge::graphStatus InferDataTypeMoeFinalizeRoutingV2(gert::InferDataTypeContext* context)
{
    const bool isMx4 = context->GetInputDataType(INDEX_IN_EXPANDED_X) == ge::DT_FLOAT4_E2M1;
    OP_LOGD(context->GetNodeName(), "Begin to do MoeFinalizeRoutingV2InferDataType.");
    OP_CHECK_IF(!isMx4 && !IsValidType(context->GetInputDataType(INDEX_IN_EXPANDED_X)),
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "expanded_x",
                                          Ops::Base::ToString(context->GetInputDataType(INDEX_IN_EXPANDED_X)).c_str(),
                                          "FLOAT, FLOAT16, BF16, HIFLOAT4 or FLOAT4_E2M1"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context->GetInputDataType(INDEX_IN_EXPANDED_ROW_IDX) != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(
                    context->GetNodeName(), "expanded_row_idx",
                    Ops::Base::ToString(context->GetInputDataType(INDEX_IN_EXPANDED_ROW_IDX)).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    if (!CheckOptionalAuxDtype(context, INDEX_IN_SKIP1, "skip1", ge::DT_BF16, isMx4) ||
        !CheckOptionalAuxDtype(context, INDEX_IN_SKIP2, "skip2", ge::DT_BF16, isMx4) ||
        !CheckOptionalAuxDtype(context, INDEX_IN_BIAS, "bias", ge::DT_BF16, isMx4) ||
        !CheckOptionalAuxDtype(context, INDEX_IN_SCALES, "scales", ge::DT_FLOAT, isMx4)) {
        return ge::GRAPH_FAILED;
    }
    if (isMx4) {
        const auto scaleDtype = context->GetOptionalInputDataType(INDEX_IN_EXPANDED_X_SCALE);
        OP_CHECK_IF(scaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "expanded_x_scale",
                                              Ops::Base::ToString(scaleDtype).c_str(), "FLOAT8_E8M0"),
                    return ge::GRAPH_FAILED);
    }

    const auto parameterDtype = context->GetOptionalInputDataType(INDEX_IN_EXPERT_IDX);
    OP_CHECK_IF(parameterDtype != ge::DT_UNDEFINED && parameterDtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "expert_idx",
                                          Ops::Base::ToString(parameterDtype).c_str(), "INT32"),
                return ge::GRAPH_FAILED);

    if (isMx4) {
        for (size_t i = INDEX_IN_X; i <= INDEX_IN_V; ++i) {
            OP_CHECK_IF(context->GetOptionalInputDataType(i) != ge::DT_UNDEFINED,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x, alpha1, alpha2, v",
                                                                 "MXFP4 requires empty special expert inputs."),
                        return ge::GRAPH_FAILED);
        }
    }
    if (isMx4 || context->GetInputDataType(INDEX_IN_EXPANDED_X) == Ops::Transformer::DT_HIFLOAT4) {
        context->SetOutputDataType(0, ge::DT_BF16);
    } else {
        context->SetOutputDataType(0, context->GetInputDataType(INDEX_IN_EXPANDED_X));
    }
    return ge::GRAPH_SUCCESS;
}

static bool IsValidShape(int64_t shape1, int64_t shape2, int64_t shape3 = UNKNOWN_DIM_VALUE_,
                         int64_t shape4 = UNKNOWN_DIM_VALUE_)
{
    int64_t knownDim = UNKNOWN_DIM_VALUE_;
    for (const auto dim : {shape1, shape2, shape3, shape4}) {
        if (dim == UNKNOWN_DIM_VALUE_) {
            continue;
        }
        if (knownDim != UNKNOWN_DIM_VALUE_ && knownDim != dim) {
            return false;
        }
        knownDim = dim;
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
                                           int64_t* outLastDim, bool& earlyReturn, bool isMx4)
{
    earlyReturn = false;
    const gert::Shape* expandedXShape = context->GetInputShape(INDEX_IN_EXPANDED_X);
    OP_CHECK_IF(expandedXShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "expanded_x"),
                return ge::GRAPH_FAILED);
    if (IsUnknownRank(expandedXShape)) {
        if (isMx4) {
            *outExpandedXShape = expandedXShape;
            return ge::GRAPH_SUCCESS;
        }
        SetUnknownRank(outShape);
        earlyReturn = true;
        return ge::GRAPH_SUCCESS;
    }
    if (!isMx4 && IsUnknownShape(expandedXShape)) {
        SetAllUnknownDim(outShape);
        earlyReturn = true;
        return ge::GRAPH_SUCCESS;
    }
    const size_t expandedXShapeSize = expandedXShape->GetDimNum();

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
    *outLastDim = expandedXShape->GetDim(expandedXShapeSize - 1);
    *outExpandedXShape = expandedXShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpandedRowIdxShape(const gert::InferShapeContext* context, gert::Shape* outShape,
                                                const gert::Shape** outRowIdxShape, bool& earlyReturn, bool isMx4)
{
    earlyReturn = false;
    const gert::Shape* expandedRowIdxShape = context->GetInputShape(INDEX_IN_EXPANDED_ROW_IDX);
    OP_CHECK_IF(expandedRowIdxShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "expanded_row_idx"),
                return ge::GRAPH_FAILED);
    if (IsUnknownRank(expandedRowIdxShape)) {
        if (isMx4) {
            *outRowIdxShape = expandedRowIdxShape;
            return ge::GRAPH_SUCCESS;
        }
        SetUnknownRank(outShape);
        earlyReturn = true;
        return ge::GRAPH_SUCCESS;
    }
    if (!isMx4 && IsUnknownShape(expandedRowIdxShape)) {
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

static ge::graphStatus InferMx4Shape(gert::InferShapeContext* context, const gert::Shape& expandedX,
                                     const gert::Shape& indices, int64_t h, gert::Shape* output)
{
    const auto k = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_K);
    OP_CHECK_NULL_WITH_CONTEXT(context, k);
    const auto scale = context->GetOptionalInputShape(INDEX_IN_EXPANDED_X_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);
    const int64_t indexCount = IsUnknownRank(&indices) ? UNKNOWN_DIM_VALUE_ : indices.GetDim(0);
    OP_CHECK_IF(*k <= 0 || indexCount < UNKNOWN_DIM_VALUE_ || (indexCount >= 0 && indexCount % *k != 0) ||
                    (h != UNKNOWN_DIM_VALUE_ && (h <= 0 || h % 2 != 0)),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "expanded_x, expanded_row_idx, k",
                                                         "Invalid MXFP4 width, index count or top-K."),
                return ge::GRAPH_FAILED);
    int64_t rows = UNKNOWN_DIM_VALUE_;
    if (!IsUnknownRank(&expandedX)) {
        rows = expandedX.GetDim(0);
        const int64_t cols = expandedX.GetDimNum() == SHAPE_SIZE + 1 ? expandedX.GetDim(1) : 1;
        OP_CHECK_IF(rows < UNKNOWN_DIM_VALUE_ || cols < UNKNOWN_DIM_VALUE_ ||
                        (rows >= 0 && cols >= 0 && __builtin_mul_overflow(rows, cols, &rows)) ||
                        (cols >= 0 && rows == 0 && indexCount > 0),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "expanded_x", "invalid",
                                                           "Invalid MXFP4 physical row count."),
                    return ge::GRAPH_FAILED);
        if (cols == UNKNOWN_DIM_VALUE_) {
            rows = UNKNOWN_DIM_VALUE_;
        }
    }
    if (!IsUnknownRank(scale)) {
        const int64_t blocks = h == UNKNOWN_DIM_VALUE_ ? h : h / MXFP4_BLOCK_SIZE + (h % MXFP4_BLOCK_SIZE != 0);
        OP_CHECK_IF(scale->GetDimNum() != SHAPE_SIZE + 1 || !IsValidShape(scale->GetDim(0), rows) ||
                        !IsValidShape(scale->GetDim(1), blocks) || !IsValidShape(scale->GetDim(2), SHAPE_SIZE) ||
                        scale->GetDim(0) < UNKNOWN_DIM_VALUE_ || scale->GetDim(1) < UNKNOWN_DIM_VALUE_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "expanded_x_scale", "invalid",
                                                           "Expected shape (R, ceil(H/64), 2)."),
                    return ge::GRAPH_FAILED);
    }
    *output = {indexCount == UNKNOWN_DIM_VALUE_ ? indexCount : indexCount / *k, h};
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDropPadModeAndOutShape(gert::InferShapeContext* context, int64_t& dropPadMode,
                                                 gert::Shape*& outShape)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "attrs",
                    "Required attrs metadata is null; context->GetAttrs() must provide a valid object."),
                return ge::GRAPH_FAILED);
    const int64_t* dropPadModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE);
    OP_CHECK_IF(dropPadModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "drop_pad_mode"),
                return ge::GRAPH_FAILED);
    dropPadMode = *dropPadModePtr;

    outShape = context->GetOutputShape(INDEX_OUT);
    OP_CHECK_IF(dropPadMode < VALUE_MODE_0 || dropPadMode > VALUE_MODE_3,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "drop_pad_mode",
                                                      std::to_string(dropPadMode).c_str(), "must be in [0, 3]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpandedShapes(gert::InferShapeContext* context, int64_t dropPadMode, gert::Shape* outShape,
                                           const gert::Shape*& expandedXShape, int64_t& lastDimExpandedX,
                                           const gert::Shape*& expandedRowIdxShape, bool& earlyReturn, bool isMx4)
{
    ge::graphStatus ret =
        CheckExpandedXShape(context, dropPadMode, outShape, &expandedXShape, &lastDimExpandedX, earlyReturn, isMx4);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }

    ret = CheckExpandedRowIdxShape(context, outShape, &expandedRowIdxShape, earlyReturn, isMx4);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOptionalShapesAndK(gert::InferShapeContext* context, gert::Shape* outShape,
                                               const gert::Shape*& skip1InputShape, const gert::Shape*& skip2InputShape,
                                               const gert::Shape*& biasInputShape, const gert::Shape*& scalesInputShape,
                                               const gert::Shape*& expertIdxShape, int64_t& k, bool& earlyReturn)
{
    ge::graphStatus ret =
        CheckOptionalInputShape(context, INDEX_IN_SKIP1, "skip1", outShape, &skip1InputShape, earlyReturn);
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

    auto attrs = context->GetAttrs();
    const int64_t* kPtr = attrs->GetAttrPointer<int64_t>(ATTR_K);
    OP_CHECK_IF(kPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "k"), return ge::GRAPH_FAILED);
    k = *kPtr;
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
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4MoeFinalizeRoutingV2(gert::InferShapeContext* context)
{
    int64_t dropPadMode = 0;
    gert::Shape* outShape = nullptr;
    if (GetDropPadModeAndOutShape(context, dropPadMode, outShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const auto desc = context->GetInputDesc(INDEX_IN_EXPANDED_X);
    const bool isMx4 = desc && desc->GetDataType() == ge::DT_FLOAT4_E2M1;

    const gert::Shape* expandedXShape = nullptr;
    const gert::Shape* expandedRowIdxShape = nullptr;
    const gert::Shape* skip1InputShape = nullptr;
    const gert::Shape* skip2InputShape = nullptr;
    const gert::Shape* biasInputShape = nullptr;
    const gert::Shape* scalesInputShape = nullptr;
    const gert::Shape* expertIdxShape = nullptr;
    int64_t lastDimExpandedX = -1;
    int64_t k = 0;
    bool earlyReturn = false;
    ge::graphStatus ret = CheckExpandedShapes(context, dropPadMode, outShape, expandedXShape, lastDimExpandedX,
                                              expandedRowIdxShape, earlyReturn, isMx4);
    if (ret != ge::GRAPH_SUCCESS || earlyReturn) {
        return ret;
    }

    if (isMx4) {
        return InferMx4Shape(context, *expandedXShape, *expandedRowIdxShape, lastDimExpandedX, outShape);
    }

    ret = CheckOptionalShapesAndK(context, outShape, skip1InputShape, skip2InputShape, biasInputShape, scalesInputShape,
                                  expertIdxShape, k, earlyReturn);
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
