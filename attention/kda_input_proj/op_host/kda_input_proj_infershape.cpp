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
 * \file kda_input_proj_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "err/ops_err.h"
#include "kda_input_proj_host_common.h"

using namespace ge;
using kda_input_proj::X_INDEX;
using kda_input_proj::WEIGHT_QKV_INDEX;
using kda_input_proj::WEIGHT_BETA_INDEX;
using kda_input_proj::WEIGHT_GATE_INDEX;
using kda_input_proj::WEIGHT_G_INDEX;
using kda_input_proj::WEIGHT_QKV_SCALE_INDEX;
using kda_input_proj::QKV_INDEX;
using kda_input_proj::BETA_INDEX;
using kda_input_proj::GATE_INDEX;
using kda_input_proj::G_INDEX;
using kda_input_proj::ATTR_TRANS_WEIGHT_QKV_INDEX;
using kda_input_proj::ATTR_TRANS_WEIGHT_BETA_INDEX;
using kda_input_proj::ATTR_TRANS_WEIGHT_GATE_INDEX;
using kda_input_proj::ATTR_TRANS_WEIGHT_G_INDEX;
using kda_input_proj::X_NAME;
using kda_input_proj::WEIGHT_QKV_NAME;
using kda_input_proj::WEIGHT_BETA_NAME;
using kda_input_proj::WEIGHT_GATE_NAME;
using kda_input_proj::WEIGHT_G_NAME;
using kda_input_proj::WEIGHT_QKV_SCALE_NAME;
using kda_input_proj::QKV_NAME;
using kda_input_proj::BETA_NAME;
using kda_input_proj::GATE_NAME;
using kda_input_proj::G_NAME;

namespace ops {
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;

constexpr int64_t WEIGHT_QKV_SCALE_PACK_NUM = 2;
constexpr int64_t MX_BLOCK_SIZE = 64;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;

struct KdaInputProjShapeParam {
    int64_t tSize = 0;
    int64_t hiddenSize = 0;
    int64_t qkvSize = 0;
    int64_t betaSize = 0;
    int64_t gateSize = 0;
    int64_t gSize = 0;
};

struct KdaInputProjTransAttrs {
    bool transWeightQkv = true;
    bool transWeightBeta = true;
    bool transWeightGate = true;
    bool transWeightG = true;
};

static bool IsKnownDim(int64_t dim)
{
    return dim > 0;
}

static bool IsValidDim(int64_t dim)
{
    return dim == UNKNOWN_DIM_VALUE || dim > 0;
}

static ge::graphStatus CheckRank(const gert::Shape *shape, size_t expectRank, const char *tensorName,
                                 const char *opName)
{
    OP_CHECK_IF(shape->GetDimNum() != expectRank,
                OP_LOGE(opName, "%s must be %zuD, but got %zuD.", tensorName, expectRank, shape->GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDimValue(int64_t dim, const char *tensorName, const char *dimName, const char *opName)
{
    OP_CHECK_IF(!IsValidDim(dim), OP_LOGE(opName, "%s.%s=%ld is invalid (expect >0 or -1).", tensorName, dimName, dim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShapes(const gert::Shape *xShape, const gert::Shape *weightQkvShape,
                                        const gert::Shape *weightBetaShape, const gert::Shape *weightGateShape,
                                        const gert::Shape *weightGShape, const char *opName)
{
    if (CheckRank(xShape, DIM_NUM_TWO, X_NAME, opName) != ge::GRAPH_SUCCESS ||
        CheckRank(weightQkvShape, DIM_NUM_TWO, WEIGHT_QKV_NAME, opName) != ge::GRAPH_SUCCESS ||
        CheckRank(weightBetaShape, DIM_NUM_TWO, WEIGHT_BETA_NAME, opName) != ge::GRAPH_SUCCESS ||
        CheckRank(weightGateShape, DIM_NUM_TWO, WEIGHT_GATE_NAME, opName) != ge::GRAPH_SUCCESS ||
        CheckRank(weightGShape, DIM_NUM_TWO, WEIGHT_G_NAME, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t xDim0 = xShape->GetDim(DIM_IDX_ZERO);
    const int64_t xDim1 = xShape->GetDim(DIM_IDX_ONE);
    if (CheckDimValue(xDim0, X_NAME, "dim0", opName) != ge::GRAPH_SUCCESS ||
        CheckDimValue(xDim1, X_NAME, "dim1", opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightQkvScaleShape(const gert::Shape *weightQkvScaleShape, int64_t hiddenSize,
                                                int64_t qkvSize, bool transWeightQkv, const char *opName)
{
    if (CheckRank(weightQkvScaleShape, DIM_NUM_THREE, WEIGHT_QKV_SCALE_NAME, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t dim0 = weightQkvScaleShape->GetDim(DIM_IDX_ZERO);
    const int64_t dim1 = weightQkvScaleShape->GetDim(DIM_IDX_ONE);
    const int64_t dim2 = weightQkvScaleShape->GetDim(DIM_IDX_TWO);
    if (CheckDimValue(dim0, WEIGHT_QKV_SCALE_NAME, "dim0", opName) != ge::GRAPH_SUCCESS ||
        CheckDimValue(dim1, WEIGHT_QKV_SCALE_NAME, "dim1", opName) != ge::GRAPH_SUCCESS ||
        CheckDimValue(dim2, WEIGHT_QKV_SCALE_NAME, "dim2", opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 物理排布：
    //   trans=false: [CeilDiv(x.dim1, 64), weight_qkv.out, 2]
    //   trans=true : [weight_qkv.out, CeilDiv(x.dim1, 64), 2]
    int64_t expectDim0 = UNKNOWN_DIM_VALUE;
    int64_t expectDim1 = UNKNOWN_DIM_VALUE;
    int64_t mxHiddenSize = UNKNOWN_DIM_VALUE;
    if (IsKnownDim(hiddenSize)) {
        mxHiddenSize = (hiddenSize + MX_BLOCK_SIZE - 1) / MX_BLOCK_SIZE;
    }
    if (transWeightQkv) {
        expectDim0 = qkvSize;
        expectDim1 = mxHiddenSize;
    } else {
        expectDim0 = mxHiddenSize;
        expectDim1 = qkvSize;
    }

    if (IsKnownDim(dim0) && IsKnownDim(expectDim0) && dim0 != expectDim0) {
        OP_LOGE(opName,
                "%s.dim0=%ld does not match expected=%ld "
                "(trans_weight_qkv=%d, %s=[%ld, %ld, %ld], expect=[%ld, %ld, %ld]).",
                WEIGHT_QKV_SCALE_NAME, dim0, expectDim0, static_cast<int32_t>(transWeightQkv), WEIGHT_QKV_SCALE_NAME,
                dim0, dim1, dim2, expectDim0, expectDim1, WEIGHT_QKV_SCALE_PACK_NUM);
        return ge::GRAPH_FAILED;
    }
    if (IsKnownDim(dim1) && IsKnownDim(expectDim1) && dim1 != expectDim1) {
        OP_LOGE(opName,
                "%s.dim1=%ld does not match expected=%ld "
                "(trans_weight_qkv=%d, %s=[%ld, %ld, %ld], expect=[%ld, %ld, %ld]).",
                WEIGHT_QKV_SCALE_NAME, dim1, expectDim1, static_cast<int32_t>(transWeightQkv), WEIGHT_QKV_SCALE_NAME,
                dim0, dim1, dim2, expectDim0, expectDim1, WEIGHT_QKV_SCALE_PACK_NUM);
        return ge::GRAPH_FAILED;
    }
    if (IsKnownDim(dim2) && dim2 != WEIGHT_QKV_SCALE_PACK_NUM) {
        OP_LOGE(opName, "%s.dim2=%ld does not match expected=%ld (%s=[%ld, %ld, %ld]).", WEIGHT_QKV_SCALE_NAME, dim2,
                WEIGHT_QKV_SCALE_PACK_NUM, WEIGHT_QKV_SCALE_NAME, dim0, dim1, dim2);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static bool GetOptionalBoolAttr(const gert::InferShapeContext *context, uint32_t attrIndex, bool defaultValue)
{
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return defaultValue;
    }
    const bool *attr = attrs->GetAttrPointer<bool>(attrIndex);
    return (attr != nullptr) ? *attr : defaultValue;
}

static void GetTransWeightAttrs(const gert::InferShapeContext *context, KdaInputProjTransAttrs &transAttrs)
{
    transAttrs.transWeightQkv = GetOptionalBoolAttr(context, ATTR_TRANS_WEIGHT_QKV_INDEX, true);
    transAttrs.transWeightBeta = GetOptionalBoolAttr(context, ATTR_TRANS_WEIGHT_BETA_INDEX, true);
    transAttrs.transWeightGate = GetOptionalBoolAttr(context, ATTR_TRANS_WEIGHT_GATE_INDEX, true);
    transAttrs.transWeightG = GetOptionalBoolAttr(context, ATTR_TRANS_WEIGHT_G_INDEX, true);
}

// trans=true: weight=[outFeatures, inFeatures]；trans=false: weight=[inFeatures, outFeatures]
static ge::graphStatus InferOutFeatures(const gert::Shape *weightShape, int64_t xDim1, bool transWeight,
                                        const char *weightName, const char *opName, int64_t &outFeatures)
{
    const int64_t dim0 = weightShape->GetDim(DIM_IDX_ZERO);
    const int64_t dim1 = weightShape->GetDim(DIM_IDX_ONE);
    if (CheckDimValue(dim0, weightName, "dim0", opName) != ge::GRAPH_SUCCESS ||
        CheckDimValue(dim1, weightName, "dim1", opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t inFeatures = transWeight ? dim1 : dim0;
    const uint32_t inDimIdx = transWeight ? DIM_IDX_ONE : DIM_IDX_ZERO;
    if (IsKnownDim(inFeatures) && IsKnownDim(xDim1) && inFeatures != xDim1) {
        OP_LOGE(opName, "%s.dim%u=%ld does not match x.dim1=%ld (trans_weight=%d, %s=[%ld, %ld], x.dim1=%ld).",
                weightName, inDimIdx, inFeatures, xDim1, static_cast<int32_t>(transWeight), weightName, dim0, dim1,
                xDim1);
        return ge::GRAPH_FAILED;
    }

    outFeatures = transWeight ? dim0 : dim1;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillShapeParam(const gert::Shape *xShape, const gert::Shape *weightQkvShape,
                                      const gert::Shape *weightBetaShape, const gert::Shape *weightGateShape,
                                      const gert::Shape *weightGShape, const KdaInputProjTransAttrs &transAttrs,
                                      const char *opName, KdaInputProjShapeParam &shapeParam)
{
    shapeParam.tSize = xShape->GetDim(DIM_IDX_ZERO);
    shapeParam.hiddenSize = xShape->GetDim(DIM_IDX_ONE);
    if (InferOutFeatures(weightQkvShape, shapeParam.hiddenSize, transAttrs.transWeightQkv, WEIGHT_QKV_NAME, opName,
                         shapeParam.qkvSize) != ge::GRAPH_SUCCESS ||
        InferOutFeatures(weightBetaShape, shapeParam.hiddenSize, transAttrs.transWeightBeta, WEIGHT_BETA_NAME, opName,
                         shapeParam.betaSize) != ge::GRAPH_SUCCESS ||
        InferOutFeatures(weightGateShape, shapeParam.hiddenSize, transAttrs.transWeightGate, WEIGHT_GATE_NAME, opName,
                         shapeParam.gateSize) != ge::GRAPH_SUCCESS ||
        InferOutFeatures(weightGShape, shapeParam.hiddenSize, transAttrs.transWeightG, WEIGHT_G_NAME, opName,
                         shapeParam.gSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void SetOutputShape2D(gert::Shape *outShape, int64_t tSize, int64_t outFeatures)
{
    outShape->SetDimNum(DIM_NUM_TWO);
    outShape->SetDim(DIM_IDX_ZERO, tSize);
    outShape->SetDim(DIM_IDX_ONE, outFeatures);
}

static ge::graphStatus SetOutputShapes(gert::InferShapeContext *context, const KdaInputProjShapeParam &shapeParam)
{
    gert::Shape *qkvShape = context->GetOutputShape(QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvShape);
    gert::Shape *betaShape = context->GetOutputShape(BETA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShape);
    gert::Shape *gateShape = context->GetOutputShape(GATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gateShape);
    gert::Shape *gShape = context->GetOutputShape(G_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gShape);

    SetOutputShape2D(qkvShape, shapeParam.tSize, shapeParam.qkvSize);
    SetOutputShape2D(betaShape, shapeParam.tSize, shapeParam.betaSize);
    SetOutputShape2D(gateShape, shapeParam.tSize, shapeParam.gateSize);
    SetOutputShape2D(gShape, shapeParam.tSize, shapeParam.gSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeKdaInputProj(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KdaInputProj", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);
    const char *opName = context->GetNodeName();

    const gert::Shape *xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *weightQkvShape = context->GetInputShape(WEIGHT_QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightQkvShape);
    const gert::Shape *weightBetaShape = context->GetInputShape(WEIGHT_BETA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightBetaShape);
    const gert::Shape *weightGateShape = context->GetInputShape(WEIGHT_GATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightGateShape);
    const gert::Shape *weightGShape = context->GetInputShape(WEIGHT_G_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightGShape);
    const gert::Shape *weightQkvScaleShape = context->GetInputShape(WEIGHT_QKV_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightQkvScaleShape);

    if (CheckInputShapes(xShape, weightQkvShape, weightBetaShape, weightGateShape, weightGShape, opName) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    KdaInputProjTransAttrs transAttrs;
    GetTransWeightAttrs(context, transAttrs);

    KdaInputProjShapeParam shapeParam;
    if (FillShapeParam(xShape, weightQkvShape, weightBetaShape, weightGateShape, weightGShape, transAttrs, opName,
                       shapeParam) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckWeightQkvScaleShape(weightQkvScaleShape, shapeParam.hiddenSize, shapeParam.qkvSize,
                                 transAttrs.transWeightQkv, opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (SetOutputShapes(context, shapeParam) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(opName, "KdaInputProj InferShape: %s[%ld, %ld], %s[%ld, %ld], %s[%ld, %ld], %s[%ld, %ld].", QKV_NAME,
            shapeParam.tSize, shapeParam.qkvSize, BETA_NAME, shapeParam.tSize, shapeParam.betaSize, GATE_NAME,
            shapeParam.tSize, shapeParam.gateSize, G_NAME, shapeParam.tSize, shapeParam.gSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckExpectDataType(ge::DataType actual, ge::DataType expect, const char *tensorName,
                                           const char *opName)
{
    OP_CHECK_IF(actual != expect,
                OP_LOGE(opName, "%s dtype %s does not match expected %s.", tensorName,
                        ge::TypeUtils::DataTypeToSerialString(actual).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(expect).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeKdaInputProj(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KdaInputProj", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const char *opName = context->GetNodeName();

    // 与 op_def / 设计说明对齐
    const ge::DataType xDtype = context->GetInputDataType(X_INDEX);
    const ge::DataType weightQkvDtype = context->GetInputDataType(WEIGHT_QKV_INDEX);
    const ge::DataType weightBetaDtype = context->GetInputDataType(WEIGHT_BETA_INDEX);
    const ge::DataType weightGateDtype = context->GetInputDataType(WEIGHT_GATE_INDEX);
    const ge::DataType weightGDtype = context->GetInputDataType(WEIGHT_G_INDEX);
    const ge::DataType weightQkvScaleDtype = context->GetInputDataType(WEIGHT_QKV_SCALE_INDEX);

    if (CheckExpectDataType(xDtype, ge::DT_BF16, X_NAME, opName) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightQkvDtype, ge::DT_FLOAT8_E4M3FN, WEIGHT_QKV_NAME, opName) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightBetaDtype, ge::DT_BF16, WEIGHT_BETA_NAME, opName) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightGateDtype, ge::DT_BF16, WEIGHT_GATE_NAME, opName) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightGDtype, ge::DT_BF16, WEIGHT_G_NAME, opName) != ge::GRAPH_SUCCESS ||
        CheckExpectDataType(weightQkvScaleDtype, ge::DT_FLOAT8_E8M0, WEIGHT_QKV_SCALE_NAME, opName) !=
            ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(QKV_INDEX, ge::DT_BF16);
    context->SetOutputDataType(BETA_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(GATE_INDEX, ge::DT_BF16);
    context->SetOutputDataType(G_INDEX, ge::DT_BF16);

    OP_LOGI(opName, "KdaInputProj InferDataType: %s/%s/%s=%s, %s=%s.", QKV_NAME, GATE_NAME, G_NAME,
            ge::TypeUtils::DataTypeToSerialString(ge::DT_BF16).c_str(), BETA_NAME,
            ge::TypeUtils::DataTypeToSerialString(ge::DT_FLOAT).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KdaInputProj).InferShape(InferShapeKdaInputProj).InferDataType(InferDataTypeKdaInputProj);
} // namespace ops
