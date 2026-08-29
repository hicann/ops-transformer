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
 * \file all_gather_matmul_v3_infershape.cpp
 * \brief InferShape for AllGatherMatmulV3
 */

#include <register/op_impl_registry.h>

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "op_host/mc2_common_infershape.h"
#include "util/math_util.h"

namespace ops {

using Ops::Base::CeilDiv;

namespace {

constexpr size_t INDEX_IN_X1 = 1;
constexpr size_t INDEX_IN_X2 = 2;
constexpr size_t INDEX_IN_X1_SCALE = 4;
constexpr size_t INDEX_IN_X2_SCALE = 5;
constexpr size_t INDEX_OUT_Y = 0;
constexpr size_t INDEX_OUT_GATHER_OUT = 1;
constexpr size_t INDEX_ATTR_IS_TRANS_B = 3;
constexpr size_t INDEX_ATTR_RANK_SIZE = 4;
constexpr size_t INDEX_ATTR_Y_DTYPE = 6;

constexpr int64_t MX_SCALE_BLOCK_SIZE = 64;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t DIM_MINUS_ONE = -1;
constexpr int64_t K_MIN_VAL = 256;
constexpr int64_t K_MAX_VAL = 65535;
constexpr size_t SUPPORT_DIM_SIZE = 2;

// scale 各维度可能为动态未知(-1)，仅对已知维度做等值校验
bool IsDimMismatch(uint64_t actual, int64_t expected)
{
    return actual != static_cast<uint64_t>(DIM_MINUS_ONE) && actual != static_cast<uint64_t>(expected);
}

// rank_size 未设置(<=0)时拦截
ge::graphStatus ResolveRankSize(const gert::InferShapeContext *context, int64_t &rankSize)
{
    const auto nodeName = context->GetNodeName();
    const auto attrs = context->GetAttrs();
    const int64_t *rankSizeAttr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_RANK_SIZE);
    OPS_CHECK(rankSizeAttr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "rank_size"), return ge::GRAPH_FAILED);
    rankSize = *rankSizeAttr;
    OPS_CHECK(rankSize <= 0, OP_LOGE(nodeName, "rank_size attr must be set, got %ld", rankSize),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMatrixInputShapes(const gert::InferShapeContext *context)
{
    const auto x1Shape = context->GetInputShape(INDEX_IN_X1);
    const auto x2Shape = context->GetInputShape(INDEX_IN_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    if (x1Shape->GetDimNum() != SUPPORT_DIM_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x1", (std::to_string(x1Shape->GetDimNum()) + "D").c_str(),
                                     "2D");
        return ge::GRAPH_FAILED;
    }
    if (x2Shape->GetDimNum() != SUPPORT_DIM_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x2", (std::to_string(x2Shape->GetDimNum()) + "D").c_str(),
                                     "2D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckMatrixDtypes(const gert::InferShapeContext *context)
{
    const auto nodeName = context->GetNodeName();
    auto x1Dtype = context->GetInputDesc(INDEX_IN_X1)->GetDataType();
    auto x2Dtype = context->GetInputDesc(INDEX_IN_X2)->GetDataType();
    if ((x1Dtype != ge::DT_FLOAT8_E4M3FN && x1Dtype != ge::DT_FLOAT8_E5M2 && x1Dtype != ge::DT_FLOAT4_E2M1) ||
        (x2Dtype != ge::DT_FLOAT8_E4M3FN && x2Dtype != ge::DT_FLOAT8_E5M2 && x2Dtype != ge::DT_FLOAT4_E2M1)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            nodeName, "x1, x2",
            (std::to_string(static_cast<int>(x1Dtype)) + ", " + std::to_string(static_cast<int>(x2Dtype))).c_str(),
            "only FP8_E4M3FN/FP8_E5M2/FP4_E2M1 supported");
        return ge::GRAPH_FAILED;
    }
    if ((x1Dtype == ge::DT_FLOAT4_E2M1) != (x2Dtype == ge::DT_FLOAT4_E2M1)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            nodeName, "x1, x2",
            (std::to_string(static_cast<int>(x1Dtype)) + ", " + std::to_string(static_cast<int>(x2Dtype))).c_str(),
            "FP4 requires both x1 and x2 to be FP4_E2M1");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckKDimMatch(const gert::InferShapeContext *context, int64_t kX1, int64_t kX2)
{
    OPS_CHECK(kX1 != kX2,
              OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "x1.K, x2.K",
                                                     (std::to_string(kX1) + ", " + std::to_string(kX2)).c_str(),
                                                     "The values of x1.K and x2.K must be the same"),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckKConstraints(const gert::InferShapeContext *context, int64_t k)
{
    const auto nodeName = context->GetNodeName();
    OPS_CHECK(
        k < K_MIN_VAL || k >= K_MAX_VAL,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "K", std::to_string(k).c_str(),
            (std::string("in range [") + std::to_string(K_MIN_VAL) + ", " + std::to_string(K_MAX_VAL) + ")").c_str()),
        return ge::GRAPH_FAILED);

    // fp4 要求 K 为偶数
    auto x1Dtype = context->GetInputDesc(INDEX_IN_X1)->GetDataType();
    if (x1Dtype == ge::DT_FLOAT4_E2M1) {
        OPS_CHECK(k % 2 != 0,
                  OP_LOGE_FOR_INVALID_VALUE(nodeName, "K", std::to_string(k).c_str(), "must be even in mxfp4 scene"),
                  return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckOneScaleShape(const gert::InferShapeContext *context, const gert::Shape *scaleShape,
                                   const char *paramName, int64_t dim0, int64_t expectedScaleK)
{
    const auto nodeName = context->GetNodeName();
    OPS_CHECK(IsDimMismatch(scaleShape->GetDim(0U), dim0) || IsDimMismatch(scaleShape->GetDim(1U), expectedScaleK) ||
                  IsDimMismatch(scaleShape->GetDim(2U), SCALE_LAST_DIM),
              OP_LOGE_FOR_INVALID_SHAPE(
                  nodeName, paramName,
                  (std::string("[") + std::to_string(scaleShape->GetDim(0U)) + "," +
                   std::to_string(scaleShape->GetDim(1U)) + "," + std::to_string(scaleShape->GetDim(2U)) + "]")
                      .c_str(),
                  (std::string("[") + std::to_string(dim0) + "," + std::to_string(expectedScaleK) + "," +
                   std::to_string(SCALE_LAST_DIM) + "]")
                      .c_str()),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// x1_scale: [M, Ceil(K/64), 2]; x2_scale: [N, Ceil(K/64), 2]
ge::graphStatus CheckScaleShapes(const gert::InferShapeContext *context, int64_t m, int64_t n, int64_t k)
{
    auto x1ScaleShape = context->GetOptionalInputShape(INDEX_IN_X1_SCALE);
    auto x2ScaleShape = context->GetOptionalInputShape(INDEX_IN_X2_SCALE);
    if (x1ScaleShape == nullptr || x2ScaleShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t expectedScaleK = CeilDiv(k, MX_SCALE_BLOCK_SIZE);
    if (CheckOneScaleShape(context, x1ScaleShape, "x1_scale", m, expectedScaleK) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOneScaleShape(context, x2ScaleShape, "x2_scale", n, expectedScaleK) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DeriveOutputShapes(gert::InferShapeContext *context, int64_t m, int64_t k, int64_t n, int64_t worldSize)
{
    // y = [world_size * M_per_rank, N]
    auto outShape = context->GetOutputShape(INDEX_OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(SUPPORT_DIM_SIZE);
    if (m != DIM_MINUS_ONE && n != DIM_MINUS_ONE) {
        outShape->SetDim(0U, m * worldSize);
        outShape->SetDim(1U, n);
    } else {
        outShape->SetDim(0U, DIM_MINUS_ONE);
        outShape->SetDim(1U, DIM_MINUS_ONE);
    }

    // gather_out = [world_size * M_per_rank, K] (REQUIRED)
    auto gatherOutShape = context->GetOutputShape(INDEX_OUT_GATHER_OUT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gatherOutShape);
    gatherOutShape->SetDimNum(SUPPORT_DIM_SIZE);
    if (m != DIM_MINUS_ONE && k != DIM_MINUS_ONE) {
        gatherOutShape->SetDim(0U, m * worldSize);
        gatherOutShape->SetDim(1U, k);
    } else {
        gatherOutShape->SetDim(0U, DIM_MINUS_ONE);
        gatherOutShape->SetDim(1U, DIM_MINUS_ONE);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus AllGatherMatmulV3InferShape(gert::InferShapeContext *context)
{
    const auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "InferShape start");

    OP_LOGE_IF(CheckMatrixInputShapes(context) != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED, nodeName,
               "check matrix input shapes failed.");
    OP_LOGE_IF(CheckMatrixDtypes(context) != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED, nodeName,
               "check matrix dtypes failed.");

    // 提取维度：x1 [M, K]；x2 在 is_trans_b=true 时为 [N, K]
    const auto x1Shape = context->GetInputShape(INDEX_IN_X1);
    const auto x2Shape = context->GetInputShape(INDEX_IN_X2);
    int64_t m = x1Shape->GetDim(0U);
    int64_t k = x1Shape->GetDim(1U);
    const bool *isTransX2 = context->GetAttrs()->GetAttrPointer<bool>(INDEX_ATTR_IS_TRANS_B);
    const bool transX2 = ((isTransX2 != nullptr) && (*isTransX2));
    int64_t n = transX2 ? x2Shape->GetDim(0U) : x2Shape->GetDim(1U);
    int64_t kX2 = transX2 ? x2Shape->GetDim(1U) : x2Shape->GetDim(0U);

    // rank_size: 必须由外界显式传入（<=0 时 ResolveRankSize 拦截报错）
    int64_t worldSize = 0;
    OP_LOGE_IF(ResolveRankSize(context, worldSize) != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED, nodeName,
               "resolve rank_size failed.");

    OP_LOGD(nodeName, "InferShape: x1[%ld,%ld] x2 K=%ld N=%ld rankSize=%ld", m, k, kX2, n, worldSize);

    OP_LOGE_IF(CheckKDimMatch(context, k, kX2) != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED, nodeName,
               "check K dim match failed.");
    // 静态 shape 才做取值约束，动态 shape(-1) 留待 tiling 校验
    if (m != DIM_MINUS_ONE && k != DIM_MINUS_ONE && n != DIM_MINUS_ONE) {
        OP_LOGE_IF(CheckKConstraints(context, k) != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED, nodeName,
                   "check K constraints failed.");
        OP_LOGE_IF(CheckScaleShapes(context, m, n, k) != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED, nodeName,
                   "check scale shapes failed.");
    }

    OP_LOGE_IF(DeriveOutputShapes(context, m, k, n, worldSize) != ge::GRAPH_SUCCESS, ge::GRAPH_FAILED, nodeName,
               "derive output shapes failed.");

    OP_LOGD(nodeName, "InferShape success: M_per_rank=%ld K=%ld N=%ld world_size=%ld", m, k, n, worldSize);
    return ge::GRAPH_SUCCESS;
}

// y dtype 由 y_dtype attr 决定（BF16/FP16），gather_out 与 x1 同 dtype
ge::graphStatus AllGatherMatmulV3InferDataType(gert::InferDataTypeContext *context)
{
    const auto nodeName = context->GetNodeName();

    auto yDtypePtr = context->GetAttrs()->GetAttrPointer<int64_t>(INDEX_ATTR_Y_DTYPE);
    ge::DataType yDtype = (yDtypePtr != nullptr) ? static_cast<ge::DataType>(*yDtypePtr) : ge::DT_UNDEFINED;
    OPS_CHECK(
        yDtype != ge::DT_BF16 && yDtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "y_dtype", std::to_string(static_cast<int>(yDtype)).c_str(), "BF16/FP16"),
        return ge::GRAPH_FAILED);

    context->SetOutputDataType(INDEX_OUT_Y, yDtype);
    context->SetOutputDataType(INDEX_OUT_GATHER_OUT, context->GetInputDataType(INDEX_IN_X1));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AllGatherMatmulV3)
    .InferShape(AllGatherMatmulV3InferShape)
    .InferDataType(AllGatherMatmulV3InferDataType);

} // namespace ops
