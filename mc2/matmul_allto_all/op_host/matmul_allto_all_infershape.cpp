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
 * \file matmul_allto_all_infershape.cpp
 * \brief 图模式（动态图/静态图）走infershape
 */

#include <register/op_impl_registry.h>
#include <limits>
#include <utility>
#include "util/math_util.h"
#include "mc2_log.h"
#include "mc2_infer_failure_report.h"
#include "common/utils/op_mc2.h"
#include "op_host/mc2_common_infershape.h"

namespace ops {

using Ops::Base::CeilDiv;

namespace {

// input tensor index
constexpr size_t INDEX_IN_X1 = 0;
constexpr size_t INDEX_IN_X2 = 1;
constexpr size_t INDEX_IN_BIAS = 2;
constexpr size_t INDEX_IN_X1_SCALE = 3;
constexpr size_t INDEX_IN_X2_SCALE = 4;
// attr index
constexpr size_t INDEX_ATTR_GROUP = 0;
constexpr size_t INDEX_ATTR_WORLD_SIZE = 1;
constexpr size_t INDEX_ATTR_ALLTO_ALL_AXES = 2;
constexpr size_t INDEX_ATTR_Y_DTYPE = 3;
constexpr size_t INDEX_ATTR_X1_QUANT_MODE = 4;
constexpr size_t INDEX_ATTR_X2_QUANT_MODE = 5;
constexpr size_t INDEX_ATTR_TRANS_X1 = 8;
constexpr size_t INDEX_ATTR_TRANS_X2 = 9;
// output index
constexpr size_t INDEX_OUT = 0;
// 维度信息
constexpr uint64_t DIM_THREE = 3;
constexpr uint64_t DIM_TWO = 2;
constexpr uint64_t DIM_ONE = 1;
// 合法性校验
constexpr int64_t OUTPUT_INFER_SHAPE = 2;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t AXIS_K_UPPER_LIMIT = 65535;
constexpr int64_t NUM_MINUS_ONE = -1;
constexpr int64_t NUM_MINUS_TWO = -2;
const std::vector<int64_t> SUPPORT_RANK_NUM{2, 4, 8, 16};
// mxfp8量化模式
constexpr uint64_t X1_MXFP8_QUANT_NUM = 6;
constexpr uint64_t X2_MXFP8_QUANT_NUM = 6;
// kc量化模式
constexpr uint64_t X1_PERTOKEN_QUANT_NUM = 3;
constexpr uint64_t X2_PERCHANNEL_QUANT_NUM = 2;

static const char* INNER_DEBUG = "MC2: MatmulAlltoAll InferShape Debug";

struct MatmulAlltoAllShapeInfo {
    int64_t outputDim;
    int64_t rankNum;
    int64_t k1;
    int64_t k2;
    int64_t m;
    int64_t n;
};

} // namespace

static ge::graphStatus CheckAllToAllAxesShapeForMatmulAlltoAll(const gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Attribute metadata was already required at entry"),
              return ge::GRAPH_FAILED);
    const auto alltoAllAxesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_ALLTO_ALL_AXES);
    if (alltoAllAxesPtr != nullptr) {
        OPS_CHECK((alltoAllAxesPtr->GetSize() != DIM_TWO),
                  OP_LOGE_FOR_INVALID_LISTSIZE(context->GetNodeName(), "all2all_axes",
                                               std::to_string(alltoAllAxesPtr->GetSize()).c_str(), "2"),
                  return ge::GRAPH_FAILED);
        const auto alltoAllAxes = static_cast<const int64_t*>(alltoAllAxesPtr->GetData());
        const std::string axesVal =
            "[" + std::to_string(alltoAllAxes[0]) + ", " + std::to_string(alltoAllAxes[1]) + "]";
        OPS_CHECK((alltoAllAxes[0] != NUM_MINUS_ONE || alltoAllAxes[1] != NUM_MINUS_TWO),
                  OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "all2all_axes", axesVal.c_str(),
                                                        "The value of all2all_axes must be [-1, -2]"),
                  return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAxisKShapeForMatmulAlltoAll(const gert::InferShapeContext* context,
                                                        MatmulAlltoAllShapeInfo& shape)
{
    OPS_CHECK(
        shape.k1 > AXIS_K_UPPER_LIMIT || shape.k2 > AXIS_K_UPPER_LIMIT,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2",
                                               ("x1=" + Ops::Base::ToString(*context->GetInputShape(INDEX_IN_X1)) +
                                                ", x2=" + Ops::Base::ToString(*context->GetInputShape(INDEX_IN_X2)))
                                                   .c_str(),
                                               "The k dimensions of x1 and x2 must not exceed 65535"),
        return ge::GRAPH_FAILED);
    if (shape.k1 != shape.k2) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2",
                                               ("x1=" + Ops::Base::ToString(*context->GetInputShape(INDEX_IN_X1)) +
                                                ", x2=" + Ops::Base::ToString(*context->GetInputShape(INDEX_IN_X2)))
                                                   .c_str(),
                                               "The value of k of x1 must be equal to that of x2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetMatmulAxisInfoForMatmulAlltoAll(const gert::InferShapeContext* context,
                                                          MatmulAlltoAllShapeInfo& shape)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Attribute metadata was already required at entry"),
              return ge::GRAPH_FAILED);
    const char* groupStr = attrs->GetAttrPointer<char>(INDEX_ATTR_GROUP);
    OPS_CHECK(
        groupStr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "group", "Required attribute group is missing; provide the communication group"),
        return ge::GRAPH_FAILED);

    const bool* isTransX1 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X1);
    OPS_CHECK(isTransX1 == nullptr,
              Mc2Error::ReportInferFailure(
                  context->GetNodeName(),
                  "GetMatmulAxisInfo: transpose_x1 schema-defaulted metadata actual=null; expected bool false"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(*isTransX1, OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "transpose_x1", "true", "false"),
              return ge::GRAPH_FAILED);
    const bool* isTransX2 = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_X2);
    const bool transX2 = ((isTransX2 != nullptr) && (*isTransX2));

    const auto x1Shape = context->GetInputShape(INDEX_IN_X1);
    const auto x2Shape = context->GetInputShape(INDEX_IN_X2);
    shape.outputDim = x1Shape->GetDimNum();
    shape.m = x1Shape->GetDim(0U);
    shape.n = transX2 ? x2Shape->GetDim(0U) : x2Shape->GetDim(1U);
    shape.k1 = x1Shape->GetDim(1U);
    shape.k2 = transX2 ? x2Shape->GetDim(1U) : x2Shape->GetDim(0U);

    if (shape.m != NUM_MINUS_ONE) {
        OPS_CHECK(CheckAxisKShapeForMatmulAlltoAll(context, shape) != ge::GRAPH_SUCCESS,
                  OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Failed to check axis k for matmul allto all."),
                  return ge::GRAPH_FAILED);
    }

    OP_LOGD(INNER_DEBUG, "Matmul m is: %ld, n is: %ld, k1 is: %ld, k2 is: %ld. transX2 is: %d", shape.m, shape.n,
            shape.k1, shape.k2, transX2);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 获取，校验并记录卡数
 *
 * @param context
 * @param shape
 */
static ge::graphStatus CheckRankDimForMatmulAlltoAll(const gert::InferShapeContext* context,
                                                     MatmulAlltoAllShapeInfo& shape)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Attribute metadata was already required at entry"),
              return ge::GRAPH_FAILED);
    const int64_t* rankDim = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_WORLD_SIZE);
    OPS_CHECK(rankDim == nullptr,
              OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "world_size",
                                                       "Required attribute world_size is missing; provide world_size"),
              return ge::GRAPH_FAILED);
    OP_TILING_CHECK(std::find(SUPPORT_RANK_NUM.begin(), SUPPORT_RANK_NUM.end(), *rankDim) == SUPPORT_RANK_NUM.end(),
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "world_size", std::to_string(*rankDim).c_str(),
                                              VectorToString(SUPPORT_RANK_NUM).c_str()),
                    return ge::GRAPH_FAILED);
    shape.rankNum = *rankDim;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出shape
 *
 * @param context
 */
static ge::graphStatus InferShapeMatmulAlltoAll(gert::InferShapeContext* context)
{
    OPS_CHECK(context == nullptr,
              Mc2Error::ReportInferFailure("MatmulAlltoAll",
                                           "InferShape entry: context actual=null; expected materialized context"),
              return ge::GRAPH_FAILED);
    const auto x2Shape = context->GetInputShape(INDEX_IN_X2);
    OPS_CHECK(x2Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x2"), return ge::GRAPH_FAILED);
    const auto x1Shape = context->GetInputShape(INDEX_IN_X1);
    OPS_CHECK(x1Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x1"), return ge::GRAPH_FAILED);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              Mc2Error::ReportInferFailure(context->GetNodeName(),
                                           "InferShape entry: GetAttrs actual=null; expected schema metadata"),
              return ge::GRAPH_FAILED);

    for (const auto& input : {std::make_pair(x1Shape, "x1"), std::make_pair(x2Shape, "x2")}) {
        const bool unknownRank = input.first->GetDimNum() == 1 && input.first->GetDim(0) == NUM_MINUS_TWO;
        OPS_CHECK(unknownRank,
                  OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), input.second, "[-2]",
                                                        "Current MatmulAlltoAll inference requires a known rank of 2; "
                                                        "unknown-rank inference is not implemented"),
                  return ge::GRAPH_FAILED);
        OPS_CHECK(input.first->GetDimNum() != DIM_TWO,
                  OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), input.second,
                                               std::to_string(input.first->GetDimNum()).c_str(), "2"),
                  return ge::GRAPH_FAILED);
    }
    MatmulAlltoAllShapeInfo shape;
    OPS_CHECK(CheckRankDimForMatmulAlltoAll(context, shape) != ge::GRAPH_SUCCESS,
              OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Failed to check rank dim for matmul allto all."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(GetMatmulAxisInfoForMatmulAlltoAll(context, shape) != ge::GRAPH_SUCCESS,
              OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Failed to check shape for matmul allto all."),
              return ge::GRAPH_FAILED);
    OPS_CHECK(CheckAllToAllAxesShapeForMatmulAlltoAll(context) != ge::GRAPH_SUCCESS,
              OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Failed to check allto_all axes for matmul allto all."),
              return ge::GRAPH_FAILED);

    OPS_CHECK(shape.m < NUM_MINUS_ONE || shape.n < NUM_MINUS_ONE,
              OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                  context->GetNodeName(), "x1, x2",
                  ("x1=" + Ops::Base::ToString(*x1Shape) + ", x2=" + Ops::Base::ToString(*x2Shape)).c_str(),
                  "Dimensions M and N must be non-negative or dynamic -1"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(shape.m != NUM_MINUS_ONE && shape.m > std::numeric_limits<int64_t>::max() / shape.rankNum,
              OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                  context->GetNodeName(), "x1", Ops::Base::ToString(*x1Shape).c_str(),
                  ("M must not exceed " + std::to_string(std::numeric_limits<int64_t>::max() / shape.rankNum) +
                   " for world_size=" + std::to_string(shape.rankNum) + "; output M * world_size must fit int64")
                      .c_str()),
              return ge::GRAPH_FAILED);
    auto shapeOut = context->GetOutputShape(INDEX_OUT);
    OPS_CHECK(
        shapeOut == nullptr,
        Mc2Error::ReportInferFailure(context->GetNodeName(),
                                     "InferShape: output y shape holder actual=null; expected materialized holder"),
        return ge::GRAPH_FAILED);
    shapeOut->SetDimNum(OUTPUT_INFER_SHAPE);
    if (shape.m == NUM_MINUS_ONE) {
        shapeOut->SetDim(0U, shape.m);
        shapeOut->SetDim(1U, shape.n);
    } else {
        int64_t outFirstDim = shape.m * shape.rankNum;
        int64_t outSecondDim =
            shape.n == NUM_MINUS_ONE ? NUM_MINUS_ONE : shape.n / shape.rankNum + (shape.n % shape.rankNum != 0);
        shapeOut->SetDim(0U, outFirstDim);
        shapeOut->SetDim(1U, outSecondDim);
        OP_LOGI(INNER_DEBUG, "Matmul allto all output shape after infer shape, dim: %ld m: %ld n: %ld.",
                OUTPUT_INFER_SHAPE, outFirstDim, outSecondDim);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 推导输出datatype
 *
 * @param context
 */
static ge::graphStatus InferDataTypeMatmulAlltoAll(gert::InferDataTypeContext* context)
{
    OPS_CHECK(context == nullptr,
              Mc2Error::ReportDataTypeMetadataFailure(
                  "MatmulAlltoAll", "InferDataType entry: context actual=null; expected materialized context"),
              return ge::GRAPH_FAILED);
    OP_LOGD(INNER_DEBUG, "Start to infer datatype of matmul allto all.");

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              Mc2Error::ReportDataTypeMetadataFailure(context->GetNodeName(),
                                                      "InferDataType: GetAttrs actual=null; expected schema metadata"),
              return ge::GRAPH_FAILED);
    const int64_t* x1QuantMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_X1_QUANT_MODE);
    const int64_t* x2QuantMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_X2_QUANT_MODE);
    OPS_CHECK(x1QuantMode == nullptr,
              Mc2Error::ReportDataTypeMetadataFailure(
                  context->GetNodeName(),
                  "InferDataType: x1_quant_mode actual=null; expected materialized schema-defaulted attribute"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(x2QuantMode == nullptr,
              Mc2Error::ReportDataTypeMetadataFailure(
                  context->GetNodeName(),
                  "InferDataType: x2_quant_mode actual=null; expected materialized schema-defaulted attribute"),
              return ge::GRAPH_FAILED);
    OPS_CHECK(!(*x1QuantMode == 0 && *x2QuantMode == 0) &&
                  !(*x1QuantMode == X1_PERTOKEN_QUANT_NUM && *x2QuantMode == X2_PERCHANNEL_QUANT_NUM) &&
                  !(*x1QuantMode == X1_MXFP8_QUANT_NUM && *x2QuantMode == X2_MXFP8_QUANT_NUM),
              OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                  context->GetNodeName(), "x1_quant_mode, x2_quant_mode",
                  (std::to_string(*x1QuantMode) + "/" + std::to_string(*x2QuantMode)).c_str(),
                  "supported combinations are (0,0), (3,2), or (6,6)"),
              return ge::GRAPH_FAILED);

    auto yType = ge::DataType::DT_UNDEFINED;
    const int64_t* yDtypePtr = attrs->GetInt(INDEX_ATTR_Y_DTYPE);
    OPS_CHECK(yDtypePtr == nullptr,
              Mc2Error::ReportDataTypeMetadataFailure(
                  context->GetNodeName(),
                  "InferDataType: y_dtype actual=null; expected materialized schema-defaulted attribute"),
              return ge::GRAPH_FAILED);
    if (*yDtypePtr != static_cast<uint64_t>(ge::DataType::DT_UNDEFINED)) {
        OP_LOGI(INNER_DEBUG, "The yDtype value is: %ld", *yDtypePtr);
        yType = static_cast<ge::DataType>(*yDtypePtr);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "y_dtype", std::to_string(*yDtypePtr).c_str(),
                                              "DT_UNDEFINED is not a concrete output dtype; provide y_dtype");
        return ge::GRAPH_FAILED;
    }
    // 设置推导的datatype
    context->SetOutputDataType(0, yType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulAlltoAll).InferShape(InferShapeMatmulAlltoAll).InferDataType(InferDataTypeMatmulAlltoAll);
} // namespace ops
