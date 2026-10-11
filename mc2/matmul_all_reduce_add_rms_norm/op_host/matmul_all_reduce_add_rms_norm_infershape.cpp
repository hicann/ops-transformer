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
 * \file matmul_all_reduce_add_rms_norm_infershape.cpp
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
static const char* kInnerDebug = "MC2 Inner Debug";

struct MatmulShapeInfo {
    size_t output_dim;
    int64_t s = 1;
    int64_t m;
    int64_t n;
    int64_t k;
};

static ge::graphStatus CheckScaleShape(const gert::Shape* scale_shape, int64_t group_size, MatmulShapeInfo& shape,
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

static ge::graphStatus InferShapeForMatmul(const gert::InferShapeContext* context, MatmulShapeInfo& shape, bool is_arn)
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
    const bool* trans_a = attrs->GetAttrPointer<bool>(static_cast<size_t>(MmAllReduceAttrIdx::K_TRANS_X1));
    OPS_CHECK(trans_a != nullptr && *trans_a,
              OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "trans_x1", "true", "false"), return ge::GRAPH_FAILED);
    const bool* trans_b = attrs->GetAttrPointer<bool>(static_cast<size_t>(MmAllReduceAttrIdx::K_TRANS_X2));
    const bool is_trans_b = ((trans_b != nullptr) && (*trans_b));
    if (dim_num_x1 == kMC2MaxShapeSize) {
        shape.s = shape_x1->GetDim(0U);
    }
    shape.m = shape_x1->GetDim(dim_num_x1 - 2U);
    shape.k = shape_x1->GetDim(dim_num_x1 - 1U);
    shape.n = is_trans_b ? shape_x2->GetDim(0U) : shape_x2->GetDim(1U);
    const auto shapeX2KIndex = is_trans_b ? 1U : 0U;
    // shape range推导的最大范围是[1,-1]
    bool is_dynamic_shape =
        (shape.k == -1 || shape_x2->GetDim(shapeX2KIndex) == -1 || shape.k == 0 ||
         shape_x2->GetDim(shapeX2KIndex) == 0 || shape.k == 1 || shape_x2->GetDim(shapeX2KIndex) == 1);
    if (!is_dynamic_shape) {
        OPS_CHECK((shape.k != shape_x2->GetDim(shapeX2KIndex)),
                  OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                      context->GetNodeName(), "x1 and x2",
                      (std::to_string(shape.k) + " and " + std::to_string(shape_x2->GetDim(shapeX2KIndex))).c_str(),
                      "The k dimension of x1 must be equal to the k dimension of x2"),
                  return ge::GRAPH_FAILED);
        const size_t scale_idx =
            is_arn ? static_cast<size_t>(MC2AddRmsNormInputIdx::K_SCALE) : static_cast<size_t>(MC2InputIdx::K_SCALE);
        const int64_t* p = attrs->GetInt(static_cast<size_t>(MmAllReduceAttrIdx::K_ANTIQUANT_GROUP_SIZE));
        const int64_t group_size = (p != nullptr ? *p : 0);
        if (CheckScaleShape(context->GetOptionalInputShape(scale_idx), group_size, shape, is_trans_b) !=
            ge::GRAPH_SUCCESS) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "scale", "invalid",
                                                  "antiquant scale shape check failed");
            return ge::GRAPH_FAILED;
        }
    }
    shape.output_dim = dim_num_x1;
    OP_LOGD(kInnerDebug, "Matmul x1 dim %zu s %ld m %ld n %ld k %ld.", dim_num_x1, shape.s, shape.m, shape.n, shape.k);
    return ge::GRAPH_SUCCESS;
}

static bool InferAllZeroShape(const gert::InferShapeContext* context)
{
    const auto shape_x1 = context->GetInputShape(static_cast<size_t>(MC2AddRmsNormInputIdx::K_X1));
    if (shape_x1 == nullptr) {
        return false;
    }
    const size_t dim_num_x1 = shape_x1->GetDimNum();
    int64_t k = -1;
    if (dim_num_x1 == kMC2MaxShapeSize) {
        k = shape_x1->GetDim(2U);
    } else if (dim_num_x1 == kMC2MinShapeSize) {
        k = shape_x1->GetDim(1U);
    }
    if (k != 0) {
        return false;
    }
    OP_LOGI(context->GetNodeName(), "kValue of x1 is 0.");
    return true;
}

static ge::graphStatus InferShapeForMC2AddRmsNorm(gert::InferShapeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(kInnerDebug, "context"), return ge::GRAPH_FAILED);
    MatmulShapeInfo shape;
    if (InferShapeForMatmul(context, shape, true) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Failed to infer shape for matmul all reduce.");
        return ge::GRAPH_FAILED;
    }
    size_t residual_idx = static_cast<size_t>(MC2AddRmsNormInputIdx::K_RESIDUAL);
    if (context->GetComputeNodeInputNum() != static_cast<size_t>(MC2AddRmsNormInputIdx::K_MAX) &&
        context->GetOptionalInputShape(static_cast<size_t>(MC2AddRmsNormInputIdx::K_BIAS)) == nullptr) {
        residual_idx -= 1U;
    }
    OP_LOGI(context->GetNodeName(), "Start to infer shape, residual tensor idx %zu.", residual_idx);
    const auto res_shape = context->GetInputShape(residual_idx);
    auto shape_out_y = context->GetOutputShape(static_cast<size_t>(MC2AddRmsNormOutputIdx::K_Y));
    auto shape_out_norm = context->GetOutputShape(static_cast<size_t>(MC2AddRmsNormOutputIdx::K_NORM_OUT));
    OPS_CHECK_NULL_WITH_CONTEXT(context, res_shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape_out_y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape_out_norm);
    OPS_CHECK(
        InferAllZeroShape(context),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x1", "k=0", "The value of x1 k must not be 0"),
        return ge::GRAPH_FAILED);
    const size_t dim_num = res_shape->GetDimNum();
    OPS_CHECK(
        dim_num != 3U,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "residual", (std::to_string(dim_num) + "D").c_str(), "3D"),
        return ge::GRAPH_FAILED);
    shape_out_y->SetDimNum(dim_num);
    shape_out_norm->SetDimNum(dim_num);
    for (size_t i = 0U; i < dim_num; ++i) {
        const int64_t dim = res_shape->GetDim(i);
        shape_out_y->SetDim(i, dim);
        shape_out_norm->SetDim(i, dim);
        OP_LOGD(context->GetNodeName(), "Output shape dim idx %lu, dim value %ld.", i, dim);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeForMC2AddRmsNorm(gert::InferDataTypeContext* context)
{
    OPS_CHECK(context == nullptr, OP_LOGE_WITH_INVALID_INPUT(kInnerDebug, "context"), return ge::GRAPH_FAILED);
    const ge::DataType out_type = context->GetInputDataType(static_cast<size_t>(MC2AddRmsNormInputIdx::K_RESIDUAL));
    OP_LOGI(context->GetNodeName(), "Start to infer dtype, output dtype is %u.", out_type);
    ge::graphStatus ret = context->SetOutputDataType(static_cast<size_t>(MC2AddRmsNormOutputIdx::K_Y), out_type);
    OPS_CHECK(ret != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Failed to set dtype %d to y.", out_type),
              return ge::GRAPH_FAILED);
    ret = context->SetOutputDataType(static_cast<size_t>(MC2AddRmsNormOutputIdx::K_NORM_OUT), out_type);
    OPS_CHECK(ret != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Failed to set dtype %d to norm.", out_type),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulAllReduceAddRmsNorm)
    .InferShape(InferShapeForMC2AddRmsNorm)
    .InferDataType(InferDtypeForMC2AddRmsNorm);
} // namespace ops
