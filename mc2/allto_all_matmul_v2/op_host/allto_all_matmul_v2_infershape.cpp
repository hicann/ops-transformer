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
 * \file allto_all_matmul_v2_infershape.cpp
 * \brief InferShape for AlltoAllMatmulV2
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
constexpr size_t INDEX_ATTR_WORLD_SIZE = 1;
constexpr size_t INDEX_ATTR_Y_DTYPE = 3;
constexpr size_t INDEX_ATTR_TRANSPOSE_X2 = 8;

constexpr int64_t MX_SCALE_BLOCK_SIZE = 64;
constexpr int64_t MX_SCALE_LAST_DIM = 2;
constexpr int64_t DIM_MINUS_ONE = -1;

bool ValidateInferDtypes(gert::InferShapeContext *context, const char *nodeName)
{
    auto *x1Desc = context->GetInputDesc(INDEX_IN_X1);
    auto *x2Desc = context->GetInputDesc(INDEX_IN_X2);
    if (x1Desc == nullptr || x2Desc == nullptr) {
        if (x1Desc == nullptr)
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "x1 desc");
        if (x2Desc == nullptr)
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "x2 desc");
        return false;
    }
    auto x1Dtype = x1Desc->GetDataType();
    auto x2Dtype = x2Desc->GetDataType();
    if ((x1Dtype != ge::DT_FLOAT8_E4M3FN && x1Dtype != ge::DT_FLOAT8_E5M2 && x1Dtype != ge::DT_FLOAT4_E2M1) ||
        (x2Dtype != ge::DT_FLOAT8_E4M3FN && x2Dtype != ge::DT_FLOAT8_E5M2 && x2Dtype != ge::DT_FLOAT4_E2M1)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            nodeName, "x1, x2", (Ops::Base::ToString(x1Dtype) + "," + Ops::Base::ToString(x2Dtype)).c_str(),
            "x1/x2 dtype must be FP8_E4M3/FP8_E5M2/FP4_E2M1");
        return false;
    }
    return true;
}

bool ValidateInferKCrossCheck(const char *nodeName, int64_t ka, int64_t kTotal, int64_t worldSize)
{
    if (kTotal != ka * worldSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "x2 K_total", std::to_string(kTotal).c_str(),
            ("K_total must equal x1.Ka * world_size = " + std::to_string(ka * worldSize) + ".").c_str());
        return false;
    }
    return true;
}

bool ValidateInferScaleShapes(gert::InferShapeContext *context, const char *nodeName, int64_t ka, int64_t kTotal,
                              int64_t m, int64_t n)
{
    if (m == DIM_MINUS_ONE || ka == DIM_MINUS_ONE || n == DIM_MINUS_ONE) {
        return true;
    }

    auto x1ScaleShape = context->GetOptionalInputShape(INDEX_IN_X1_SCALE);
    auto x2ScaleShape = context->GetOptionalInputShape(INDEX_IN_X2_SCALE);
    if (x1ScaleShape == nullptr || x2ScaleShape == nullptr) {
        return true;
    }
    if (x1ScaleShape->GetDimNum() < 3 || x2ScaleShape->GetDimNum() < 3) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            nodeName, "x1_scale, x2_scale",
            (std::to_string(x1ScaleShape->GetDimNum()) + "D," + std::to_string(x2ScaleShape->GetDimNum()) + "D")
                .c_str(),
            "x1_scale and x2_scale must be at least 3D");
        return false;
    }
    int64_t expectedScaleK1 = CeilDiv(ka, MX_SCALE_BLOCK_SIZE);
    int64_t expectedScaleK2 = CeilDiv(kTotal, MX_SCALE_BLOCK_SIZE);
    if (x1ScaleShape->GetDim(1U) != static_cast<uint64_t>(expectedScaleK1) ||
        x1ScaleShape->GetDim(2U) != static_cast<uint64_t>(MX_SCALE_LAST_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPE(
            nodeName, "x1_scale",
            ("[dim1=" + std::to_string(x1ScaleShape->GetDim(1U)) + ",dim2=" + std::to_string(x1ScaleShape->GetDim(2U)) +
             "]")
                .c_str(),
            ("[dim1=" + std::to_string(expectedScaleK1) + ",dim2=" + std::to_string(MX_SCALE_LAST_DIM) + "]").c_str());
        return false;
    }
    if (x2ScaleShape->GetDim(1U) != static_cast<uint64_t>(expectedScaleK2) ||
        x2ScaleShape->GetDim(2U) != static_cast<uint64_t>(MX_SCALE_LAST_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPE(
            nodeName, "x2_scale",
            ("[dim1=" + std::to_string(x2ScaleShape->GetDim(1U)) + ",dim2=" + std::to_string(x2ScaleShape->GetDim(2U)) +
             "]")
                .c_str(),
            ("[dim1=" + std::to_string(expectedScaleK2) + ",dim2=" + std::to_string(MX_SCALE_LAST_DIM) + "]").c_str());
        return false;
    }
    return true;
}

bool ValidateInferYDtype(gert::InferShapeContext *context, const char *nodeName)
{
    auto *yDesc = context->GetOutputDesc(INDEX_OUT_Y);
    if (yDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "y desc");
        return false;
    }
    auto outDtype = yDesc->GetDataType();
    const int64_t *yDtypePtr = context->GetAttrs()->GetAttrPointer<int64_t>(INDEX_ATTR_Y_DTYPE);
    if (yDtypePtr == nullptr) {
        return true;
    }
    if (*yDtypePtr != static_cast<int64_t>(ge::DT_BF16) && *yDtypePtr != static_cast<int64_t>(ge::DT_FLOAT16)) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "y_dtype", std::to_string(*yDtypePtr).c_str(), "BF16/FP16");
        return false;
    }
    if (outDtype != static_cast<ge::DataType>(*yDtypePtr)) {
        OP_LOGE_WITH_INVALID_ATTR(nodeName, "y_dtype", Ops::Base::ToString(outDtype).c_str(),
                                  Ops::Base::ToString(static_cast<ge::DataType>(*yDtypePtr)).c_str());
        return false;
    }
    return true;
}

} // namespace

ge::graphStatus AlltoAllMatmulV2InferShape(gert::InferShapeContext *context)
{
    const auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "InferShape start");

    // ---- Get input shapes -----------------------------------------------
    const auto x1Shape = context->GetInputShape(INDEX_IN_X1);
    const auto x2Shape = context->GetInputShape(INDEX_IN_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    if (x1Shape->GetDimNum() < 2 || x2Shape->GetDimNum() < 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            nodeName, "x1, x2",
            (std::to_string(x1Shape->GetDimNum()) + "D," + std::to_string(x2Shape->GetDimNum()) + "D").c_str(),
            "x1 and x2 must have at least 2D");
        return ge::GRAPH_FAILED;
    }

    int64_t m = x1Shape->GetDim(0U);
    int64_t ka = x1Shape->GetDim(1U); // per-rank K dim (Ka = K_total / world_size)
    // x2 shape: TransB=true → [N,K_total], TransB=false → [K_total,N]
    auto *tx2 = context->GetAttrs()->GetAttrPointer<bool>(INDEX_ATTR_TRANSPOSE_X2);
    bool transX2 = tx2 ? *tx2 : true;
    int64_t n = transX2 ? x2Shape->GetDim(0U) : x2Shape->GetDim(1U);
    int64_t kTotal = transX2 ? x2Shape->GetDim(1U) : x2Shape->GetDim(0U);

    OP_LOGD(nodeName, "InferShape: x1[%ld,%ld] x2[%ld,%ld]", m, ka, n, kTotal);

    // ---- Validate dtypes ------------------------------------------------
    if (!ValidateInferDtypes(context, nodeName)) {
        return ge::GRAPH_FAILED;
    }

    // ---- Get world_size attr --------------------------------------------
    const auto attrs = context->GetAttrs();
    const int64_t *wsPtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_WORLD_SIZE);
    if (wsPtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "world_size");
        return ge::GRAPH_FAILED;
    }
    int64_t worldSize = *wsPtr;
    if (worldSize == 0) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "world_size");
        return ge::GRAPH_FAILED;
    }

    // ---- Validate K dimension cross-check and scale constraints ---------
    if (!ValidateInferKCrossCheck(nodeName, ka, kTotal, worldSize) ||
        !ValidateInferScaleShapes(context, nodeName, ka, kTotal, m, n)) {
        return ge::GRAPH_FAILED;
    }

    // ---- Derive output shape: y = [M_total / world_size, N] ------------
    auto outShape = context->GetOutputShape(INDEX_OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outShape);

    outShape->SetDimNum(2);
    if (m != DIM_MINUS_ONE && n != DIM_MINUS_ONE) {
        outShape->SetDim(0U, m / worldSize); // per-rank output M
        outShape->SetDim(1U, n);
    } else {
        outShape->SetDim(0U, DIM_MINUS_ONE);
        outShape->SetDim(1U, DIM_MINUS_ONE);
    }

    // ---- Validate y_dtype consistency -----------------------------------
    if (!ValidateInferYDtype(context, nodeName)) {
        return ge::GRAPH_FAILED;
    }

    // Dtype and format are inferred from the proto definition

    OP_LOGD(nodeName, "InferShape success: M=%ld Ka=%ld K_total=%ld N=%ld world_size=%ld", m, ka, kTotal, n, worldSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllMatmulV2).InferShape(AlltoAllMatmulV2InferShape);

} // namespace ops
