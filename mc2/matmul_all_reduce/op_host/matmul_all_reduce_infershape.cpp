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
 * \file matmul_all_reduce_infershape.cpp
 * \brief
 */

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"

using Ops::Base::CeilDiv;

namespace ops {
constexpr size_t kMC2MinShapeSize = 2U;
constexpr size_t kMC2MaxShapeSize = 3U;
static const char *kInnerDebug = "MC2 Inner Debug";

struct MatmulShapeInfo {
    size_t output_dim;
    int64_t s = 1;
    int64_t m;
    int64_t n;
    int64_t k;
};

static ge::graphStatus CheckScaleShape(const gert::Shape *scale_shape, int64_t group_size, MatmulShapeInfo &shape,
                                       bool is_trans_b)
{
    if (scale_shape == nullptr || group_size == 0) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t k_value = static_cast<int64_t>(CeilDiv(shape.k, group_size));
    gert::Shape expect_scale;
    if (is_trans_b) {
        expect_scale = {shape.n, k_value};
    } else {
        expect_scale = {k_value, shape.n};
    }
    const std::string incorrectShape =
        "[" + std::to_string(scale_shape->GetDim(0U)) + "," + std::to_string(scale_shape->GetDim(1U)) + "]";
    const std::string correctShape =
        "[" + std::to_string(expect_scale[0]) + "," + std::to_string(expect_scale[1]) + "]";
    OPS_CHECK(expect_scale != *scale_shape,
              OP_LOGE_FOR_INVALID_SHAPE(kInnerDebug, "scale", incorrectShape.c_str(), correctShape.c_str()),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckMatmulKAndScale(const gert::InferShapeContext *context, const gert::Shape *shape_x2,
                                            MatmulShapeInfo &shape, bool is_trans_b, bool is_arn)
{
    const auto shapeX2KIndex = is_trans_b ? 1U : 0U;
    // shape range推导的最大范围是[1,-1]
    bool is_dynamic_shape =
        (shape.k == -1 || shape_x2->GetDim(shapeX2KIndex) == -1 || shape.k == 0 ||
         shape_x2->GetDim(shapeX2KIndex) == 0 || shape.k == 1 || shape_x2->GetDim(shapeX2KIndex) == 1);
    if (is_dynamic_shape) {
        return ge::GRAPH_SUCCESS;
    }
    OPS_CHECK((shape.k != shape_x2->GetDim(shapeX2KIndex)),
              OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                  context->GetNodeName(), "x1 and x2",
                  (std::to_string(shape.k) + " and " + std::to_string(shape_x2->GetDim(shapeX2KIndex))).c_str(),
                  "The k dimension of x1 must be equal to the k dimension of x2"),
              return ge::GRAPH_FAILED);
    const size_t scale_idx =
        is_arn ? static_cast<size_t>(MC2AddRmsNormInputIdx::K_SCALE) : static_cast<size_t>(MC2InputIdx::K_SCALE);
    const auto attrs = context->GetAttrs();
    const int64_t *p = attrs->GetInt(static_cast<size_t>(MmAllReduceAttrIdx::K_ANTIQUANT_GROUP_SIZE));
    const int64_t group_size = (p != nullptr ? *p : 0);
    OPS_CHECK(
        CheckScaleShape(context->GetOptionalInputShape(scale_idx), group_size, shape, is_trans_b) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Failed to check antiquant scale shape."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForMatmul(const gert::InferShapeContext *context, MatmulShapeInfo &shape, bool is_arn)
{
    const auto shape_x1 = context->GetInputShape(static_cast<size_t>(MC2InputIdx::K_X1));
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape_x1);
    const size_t dim_num_x1 = shape_x1->GetDimNum();
    OPS_CHECK(dim_num_x1 < kMC2MinShapeSize || dim_num_x1 > kMC2MaxShapeSize,
              OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x1", (std::to_string(dim_num_x1) + "D").c_str(),
                                           "2D or 3D"),
              return ge::GRAPH_FAILED);

    const auto shape_x2 = context->GetInputShape(static_cast<size_t>(MC2InputIdx::K_X2));
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape_x2);
    const size_t dim_num_x2 = shape_x2->GetDimNum();
    OPS_CHECK(
        dim_num_x2 != kMC2MinShapeSize,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x2", (std::to_string(dim_num_x2) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *trans_a = attrs->GetAttrPointer<bool>(static_cast<size_t>(MmAllReduceAttrIdx::K_TRANS_X1));
    OPS_CHECK(trans_a != nullptr && *trans_a,
              OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "trans_x1", "true", "false"), return ge::GRAPH_FAILED);
    const bool *trans_b = attrs->GetAttrPointer<bool>(static_cast<size_t>(MmAllReduceAttrIdx::K_TRANS_X2));
    const bool is_trans_b = ((trans_b != nullptr) && (*trans_b));
    if (dim_num_x1 == kMC2MaxShapeSize) {
        shape.s = shape_x1->GetDim(0U);
    }
    shape.m = shape_x1->GetDim(dim_num_x1 - 2U);
    shape.k = shape_x1->GetDim(dim_num_x1 - 1U);
    shape.n = is_trans_b ? shape_x2->GetDim(0U) : shape_x2->GetDim(1U);
    if (CheckMatmulKAndScale(context, shape_x2, shape, is_trans_b, is_arn) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    shape.output_dim = dim_num_x1;
    OP_LOGD(kInnerDebug, "Matmul x1 dim %zu s %ld m %ld n %ld k %ld.", dim_num_x1, shape.s, shape.m, shape.n, shape.k);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForMatmulAllReduce(gert::InferShapeContext *context)
{
    OPS_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(kInnerDebug, "context"), return ge::GRAPH_FAILED);
    OP_LOGD(kInnerDebug, "Start to infer shape of matmul all reduce.");
    auto shape_out = context->GetOutputShape(static_cast<size_t>(MC2OutputIdx::K_Y));
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape_out);
    MatmulShapeInfo shape;
    OPS_CHECK(InferShapeForMatmul(context, shape, false) != ge::GRAPH_SUCCESS,
              OP_LOGE(context->GetNodeName(), "Failed to infer shape for matmul all reduce."), return ge::GRAPH_FAILED);
    shape_out->SetDimNum(shape.output_dim);
    if (shape.output_dim == kMC2MaxShapeSize) {
        shape_out->SetDim(0U, shape.s);
        shape_out->SetDim(1U, shape.m);
        shape_out->SetDim(2U, shape.n);
    } else {
        shape_out->SetDim(0U, shape.m);
        shape_out->SetDim(1U, shape.n);
    }
    OP_LOGI(kInnerDebug, "MC2 output shape after infer shape, dim: %zu m: %ld n: %ld.", shape.output_dim, shape.m,
            shape.n);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMC2(gert::InferDataTypeContext *context)
{
    OPS_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(kInnerDebug, "context"), return ge::GRAPH_FAILED);
    OP_LOGD(kInnerDebug, "Start to infer datatype of MC2.");
    ge::DataType x1_type = context->GetInputDataType(static_cast<size_t>(MC2InputIdx::K_X1));
    ge::DataType x2_type = context->GetInputDataType(static_cast<size_t>(MC2InputIdx::K_X2));
    ge::DataType dequant_type = context->GetOptionalInputDataType(static_cast<size_t>(MC2InputIdx::K_DEQUANT));
    ge::DataType y_type = context->GetOutputDataType(0U);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *y_dtype_ptr = attrs->GetInt(static_cast<size_t>(MmAllReduceAttrIdx::K_DTYPE_Y));
    const int64_t y_dtype = (y_dtype_ptr != nullptr ? *y_dtype_ptr : static_cast<uint64_t>(ge::DataType::DT_UNDEFINED));

    if (y_dtype != static_cast<uint64_t>(ge::DataType::DT_UNDEFINED)) {
        y_type = static_cast<ge::DataType>(y_dtype);
    } else {
        if ((x1_type == ge::DT_INT8) && (x2_type == ge::DT_INT8) && (dequant_type == ge::DT_BF16)) {
            y_type = ge::DT_BF16;
        }
        if ((x1_type == ge::DT_INT8) && (x2_type == ge::DT_INT8) && (dequant_type != ge::DT_BF16)) {
            y_type = ge::DT_FLOAT16;
        }
    }
    return context->SetOutputDataType(0U, y_type);
}

IMPL_OP_INFERSHAPE(MatmulAllReduce).InferShape(InferShapeForMatmulAllReduce).InferDataType(InferDataTypeForMC2);
} // namespace ops
