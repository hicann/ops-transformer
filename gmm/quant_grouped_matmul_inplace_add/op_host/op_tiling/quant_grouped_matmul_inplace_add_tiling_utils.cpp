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
 * \file quant_grouped_matmul_inplace_add_tiling_utils.cpp
 * \brief
 */
#include "quant_grouped_matmul_inplace_add_tiling_utils.h"
#include "common/include/op_host/op_const_def_transformer.h"

#include <graph/utils/type_utils.h>

#include "../../op_kernel/arch35/qgmm_inplace_add_tiling_key.h"
#include "../quant_grouped_matmul_inplace_add_host_utils.h"
#include "log/log.h"

using namespace optiling;
using namespace optiling::GmmConstant;

namespace QuantGroupedMatmulInplaceAdd {

bool AnalyzeAttrsForInplaceAdd(const gert::TilingContext* context, GQmmInputInfo& inputParams)
{
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
        if (groupListTypePtr != nullptr) {
            OP_CHECK_IF(*groupListTypePtr != 0 && *groupListTypePtr != 1,
                        OP_LOGE_FOR_INVALID_VALUE(inputParams.opType, "groupListType",
                                                  std::to_string(*groupListTypePtr), "0 or 1"),
                        return false);
            inputParams.groupListType = static_cast<int8_t>(*groupListTypePtr);
        }
        const int64_t* groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_SIZE);
        if (groupSizePtr != nullptr) {
            OP_CHECK_IF(*groupSizePtr != 0,
                        OP_LOGE_FOR_INVALID_VALUE(inputParams.opType, "groupSize", std::to_string(*groupSizePtr), "0"),
                        return false);
        }
    }
    inputParams.transA = true;
    return true;
}

bool AnalyzeDtypeForInplaceAdd(const gert::TilingContext* context, GQmmInputInfo& inputParams)
{
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_IF(xDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opType, "x1", "nullptr", "xDesc cannot be nullptr"),
                return false);
    inputParams.aDtype = xDesc->GetDataType();
    inputParams.aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()));
    auto wDesc = context->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_IF(wDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opType, "x2", "nullptr", "wDesc cannot be nullptr"),
                return false);
    inputParams.bDtype = wDesc->GetDataType();
    inputParams.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    auto scaleDesc = context->GetInputDesc(SCALE_INDEX);
    OP_CHECK_IF(
        scaleDesc == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opType, "scale2", "nullptr", "scaleDesc cannot be nullptr"),
        return false);
    inputParams.scaleDtype = scaleDesc->GetDataType();
    auto pertokenScaleDesc = context->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams.perTokenScaleDtype;
    auto yDesc = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opType, "y", "nullptr", "yDesc cannot be nullptr"),
                return false);
    inputParams.cDtype = yDesc->GetDataType();
    return true;
}

bool IsHif4WeightForInplaceAdd(const GQmmInputInfo& inputParams)
{
    return inputParams.bDtype == Ops::Transformer::DT_HIFLOAT4;
}

bool CheckFormatForInplaceAdd(const GQmmInputInfo& inputParams)
{
    OP_CHECK_IF(inputParams.aFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT(inputParams.opType, "x1",
                                           ge::TypeUtils::FormatToSerialString(inputParams.aFormat), "FORMAT_ND"),
                return false);
    OP_CHECK_IF(inputParams.bFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT(inputParams.opType, "x2",
                                           ge::TypeUtils::FormatToSerialString(inputParams.bFormat), "FORMAT_ND"),
                return false);
    return true;
}

bool CheckShapeNotNegative(const gert::Shape& xShape, const gert::Shape& wShape, const GQmmInputInfo& inputParams)
{
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        OP_CHECK_IF(xShape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        inputParams.opType, "x1, x2", "x1=" + ShapeToString(xShape) + ", x2=" + ShapeToString(wShape),
                        BuildErrorMsgStr("dim[", i, "] of x1 must be non-negative, but the actual value is ",
                                         xShape.GetDim(i))),
                    return false);
    }
    for (size_t i = 0; i < wShape.GetDimNum(); ++i) {
        OP_CHECK_IF(wShape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        inputParams.opType, "x1, x2", "x1=" + ShapeToString(xShape) + ", x2=" + ShapeToString(wShape),
                        BuildErrorMsgStr("dim[", i, "] of x2 must be non-negative, but the actual value is ",
                                         wShape.GetDim(i))),
                    return false);
    }
    return true;
}

void NormalizeEmptyReductionForInplaceAdd(const gert::TilingContext* context, const gert::Shape& wShape,
                                          uint64_t groupNum, gert::Shape& xShape, gert::Shape& x1ScaleShape)
{
    if (xShape.GetDimNum() != DIM_NUM_2D || wShape.GetDimNum() != DIM_NUM_2D || wShape.GetDim(0) != 0 ||
        xShape.GetDim(1) != 0 || xShape.GetDim(0) <= 0) {
        return;
    }
    const auto* yStorageShape = context->GetOutputShape(Y_INDEX);
    if (yStorageShape == nullptr) {
        return;
    }
    const auto& yShape = yStorageShape->GetOriginShape();
    const int64_t m = xShape.GetDim(0);
    constexpr size_t Y_REF_RANK = 3;
    if (yShape.GetDimNum() != Y_REF_RANK || yShape.GetDim(0) != static_cast<int64_t>(groupNum) ||
        yShape.GetDim(1) != m || yShape.GetDim(2) != wShape.GetDim(1)) {
        return;
    }
    xShape.SetDim(0, 0);
    xShape.SetDim(1, m);
    // For K=0, MX has one scale row per group. An unfused [M, E, 2] becomes [E, M, 2].
    if (x1ScaleShape.GetDimNum() == MXFP_PER_TOKEN_SCALE_DIM_NUM && x1ScaleShape.GetDim(0) == m &&
        x1ScaleShape.GetDim(1) == static_cast<int64_t>(groupNum)) {
        x1ScaleShape.SetDim(0, static_cast<int64_t>(groupNum));
        x1ScaleShape.SetDim(1, m);
    }
}

bool CheckXAndWeightKDim(const gert::Shape& xShape, const gert::Shape& wShape, const GQmmInputInfo& inputParams)
{
    const size_t xDimNum = xShape.GetDimNum();
    const size_t wDimNum = wShape.GetDimNum();
    OP_CHECK_IF(xDimNum < DIM_NUM_2D || wDimNum < DIM_NUM_2D,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams.opType, "x1, x2", "x1=" + ShapeToString(xShape) + ", x2=" + ShapeToString(wShape),
                    "x1 and x2 must be at least 2D so that the K axis can be located"),
                return false);
    const int64_t xK = inputParams.transA ? xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX) :
                                            xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX);
    const int64_t wK = inputParams.transB ? wShape.GetDim(wDimNum - LAST_FIRST_DIM_INDEX) :
                                            wShape.GetDim(wDimNum - LAST_SECOND_DIM_INDEX);
    OP_CHECK_IF(xK != wK,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams.opType, "x1, x2", "x1=" + ShapeToString(xShape) + ", x2=" + ShapeToString(wShape),
                    BuildErrorMsgStr("axis K of x1[", xK, "] must equal axis K of x2[", wK, "]")),
                return false);
    return true;
}

bool CheckDtypeForInplaceAdd(const GQmmInputInfo& inputParams)
{
    OP_CHECK_IF(inputParams.cDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "yRef",
                                          ge::TypeUtils::DataTypeToSerialString(inputParams.cDtype), "DT_FLOAT"),
                return false);

    bool isHif8 = inputParams.aDtype == ge::DT_HIFLOAT8 && inputParams.bDtype == ge::DT_HIFLOAT8;
    bool isFp8 = (inputParams.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams.aDtype == ge::DT_FLOAT8_E5M2) &&
                 (inputParams.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams.bDtype == ge::DT_FLOAT8_E5M2);
    bool isHif4 =
        inputParams.aDtype == Ops::Transformer::DT_HIFLOAT4 && inputParams.bDtype == Ops::Transformer::DT_HIFLOAT4;
    bool isMxFp8Fp4 = inputParams.aDtype == ge::DT_FLOAT8_E4M3FN &&
                      (inputParams.bDtype == ge::DT_FLOAT4_E2M1 || inputParams.bDtype == Ops::Transformer::DT_HIFLOAT4);

    if (isMxFp8Fp4) {
        OP_CHECK_IF(inputParams.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale1",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype),
                                              "DT_FLOAT8_E8M0"),
                    return false);
        const bool isHif4Weight = IsHif4WeightForInplaceAdd(inputParams);
        const ge::DataType expectedScaleDtype = isHif4Weight ? Ops::Transformer::DT_HIFLOAT4_SCALE : ge::DT_FLOAT8_E8M0;
        OP_CHECK_IF(inputParams.scaleDtype != expectedScaleDtype,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale2",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype),
                                              ge::TypeUtils::DataTypeToSerialString(expectedScaleDtype)),
                    return false);
        return true;
    }

    // MXFP4：两路输入均为 FLOAT4_E2M1，块缩放均为 FLOAT8_E8M0，scale 约束与 MXFP8 一致。
    bool isMx4 = inputParams.aDtype == ge::DT_FLOAT4_E2M1 && inputParams.bDtype == ge::DT_FLOAT4_E2M1;

    if (isHif4) {
        OP_CHECK_IF(inputParams.scaleDtype != Ops::Transformer::DT_HIFLOAT4_SCALE,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale2",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype),
                                              "DT_HIFLOAT4_SCALE"),
                    return false);
        OP_CHECK_IF(inputParams.perTokenScaleDtype != Ops::Transformer::DT_HIFLOAT4_SCALE,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale1",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype),
                                              "DT_HIFLOAT4_SCALE"),
                    return false);
        return true;
    }

    if (isHif8) {
        OP_CHECK_IF(
            inputParams.scaleDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale2",
                                      ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype), "DT_FLOAT"),
            return false);
        OP_CHECK_IF(inputParams.perTokenScaleDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale1",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype),
                                              "DT_FLOAT"),
                    return false);
        return true;
    }

    if (isFp8 || isMx4) {
        OP_CHECK_IF(
            inputParams.scaleDtype != ge::DT_FLOAT8_E8M0,
            OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale2",
                                      ge::TypeUtils::DataTypeToSerialString(inputParams.scaleDtype), "DT_FLOAT8_E8M0"),
            return false);
        OP_CHECK_IF(inputParams.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams.opType, "scale1",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams.perTokenScaleDtype),
                                              "DT_FLOAT8_E8M0"),
                    return false);
        return true;
    }

    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        inputParams.opType, "x1, x2",
        ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams.aDtype),
                     ge::TypeUtils::DataTypeToSerialString(inputParams.bDtype)),
        "in quant case, the dtypes of x1 and x2 must be both DT_HIFLOAT4, both DT_HIFLOAT8, both "
        "DT_FLOAT8_E4M3FN/DT_FLOAT8_E5M2, both DT_FLOAT4_E2M1, or DT_FLOAT8_E4M3FN with DT_FLOAT4_E2M1/DT_HIFLOAT4");
    return false;
}

bool CheckMx4PackedShapeForInplaceAdd(const gert::Shape& xShape, const gert::Shape& wShape,
                                      const GQmmInputInfo& inputParams)
{
    // FP4 TN（左转置、右不转置）：原始张量为 x1 = [k, m]、x2 = [k, n]。
    // FLOAT4_E2M1 每字节打包两个元素，因此打包轴 m、n 必须为偶数。
    // mSize、nSize 已由 SetMKN 分别从 x1、x2 中提取。
    OP_CHECK_IF(xShape.GetDimNum() < DIM_NUM_2D || wShape.GetDimNum() < DIM_NUM_2D,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    inputParams.opType, "x1/x2",
                    std::to_string(xShape.GetDimNum()) + "/" + std::to_string(wShape.GetDimNum()), "at least 2"),
                return false);
    auto mSize = inputParams.mSize;
    auto nSize = inputParams.nSize;
    auto kSize = inputParams.kSize;
    // 2：FLOAT4_E2M1 沿内轴每字节打包两个元素。
    OP_CHECK_IF(
        mSize % 2 != 0 || nSize % 2 != 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            inputParams.opType, "x1/x2", "x1=[" + ShapeToString(xShape) + "], x2=[" + ShapeToString(wShape) + "]",
            "in FP4 TN case, the packed axes m[" + std::to_string(mSize) + "] and n[" + std::to_string(nSize) +
                "] of x1/x2 must both be even (FLOAT4_E2M1 packs 2 elements per byte)"),
        return false);

    OP_CHECK_IF(
        kSize == 1 && (mSize == 2 || nSize == 2),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams.opType, "x1/x2",
                                               "x1=[" + ShapeToString(xShape) + "], x2=[" + ShapeToString(wShape) + "]",
                                               "in mxfp4 case, if k is 1 then m and n must not be 2"),
        return false);
    return true;
}

bool CheckCoreNumForInplaceAdd(const gert::TilingContext* context, const GQmmInputInfo& inputParams)
{
    auto compileInfo = context->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(inputParams.opName, "compileInfo is nullptr."), return false);
    auto aicNum = compileInfo->aicNum;
    auto aivNum = compileInfo->aivNum;
    OP_CHECK_IF(aicNum == 0, OP_LOGE(inputParams.opName, "aicNum should be positive integer, actual is %u.", aicNum),
                return false);
    const uint32_t expectedAivRatio =
        (compileInfo->npuArch == NpuArch::DAV_9201 || compileInfo->npuArch == NpuArch::DAV_9202) ?
            1U :
            GmmConstant::CORE_RATIO;
    OP_CHECK_IF(aivNum != expectedAivRatio * aicNum,
                OP_LOGE(inputParams.opName, "aicNum:aivNum should be 1:%u, actual aicNum: %u, aivNum: %u.",
                        expectedAivRatio, aicNum, aivNum),
                return false);
    return true;
}

bool CheckShapeForHif8Quant(const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape,
                            const GQmmInputInfo& inputParams)
{
    // T-T / T-C 合并校验，与 aclnn CheckHif8QuantParamsShape 对齐：
    //   scale1 (x1Scale/perToken): 1D 或 2D；firstDim == groupNum；若 2D 则 lastDim == 1
    //   scale2 (x2Scale):          1D 或 2D；firstDim == groupNum；若 2D 则 lastDim ∈ {1, nSize}
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != 1 && x1ScaleDimNum != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale1", std::to_string(x1ScaleDimNum), "1 or 2"),
                return false);
    auto x1FirstDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    OP_CHECK_IF(x1FirstDim != inputParams.groupNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    inputParams.opType, "scale1", "[" + ShapeToString(x1ScaleShape) + "]",
                    BuildErrorMsgStr("in T-T/T-C mode, first dim of scale1 must be equal to groupNum[",
                                     inputParams.groupNum, "]")),
                return false);
    if (x1ScaleDimNum == QuantGroupedMatmulInplaceAdd::DIM_NUM_2D) {
        auto x1LastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(1));
        OP_CHECK_IF(
            x1LastDim != 1,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams.opType, "scale1", "[" + ShapeToString(x1ScaleShape) + "]",
                                                  "in T-T/T-C mode, last dim of scale1 must be equal to 1"),
            return false);
    }

    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != 1 && x2ScaleDimNum != 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale2", std::to_string(x2ScaleDimNum), "1 or 2"),
                return false);
    auto x2FirstDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    OP_CHECK_IF(
        x2FirstDim != inputParams.groupNum,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams.opType, "scale2", "[" + ShapeToString(x2ScaleShape) + "]",
                                              BuildErrorMsgStr("in T-T/T-C mode, first dim of scale2 must "
                                                               "be equal to groupNum[",
                                                               inputParams.groupNum, "]")),
        return false);
    if (x2ScaleDimNum == QuantGroupedMatmulInplaceAdd::DIM_NUM_2D) {
        auto x2LastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(1));
        OP_CHECK_IF(
            x2LastDim != 1 && x2LastDim != inputParams.nSize,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams.opType, "scale2", "[" + ShapeToString(x2ScaleShape) + "]",
                                                  BuildErrorMsgStr("in T-T/T-C mode, last dim of scale2 must be "
                                                                   "equal to 1 or n[",
                                                                   inputParams.nSize, "]")),
            return false);
    }
    return true;
}

bool CheckShapeForMxQuant(const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape,
                          const GQmmInputInfo& inputParams)
{
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x2ScaleDimNum != MXFP_TYPE_K_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale2", std::to_string(x2ScaleDimNum), "3"),
                return false);
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale1", std::to_string(x1ScaleDimNum), "3"),
                return false);
    auto xScaleLastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - 1));
    auto xScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto xScaleMDim = static_cast<uint64_t>(x1ScaleShape.GetDim(x1ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleLastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - 1));
    auto wScaleNDim = static_cast<uint64_t>(x2ScaleShape.GetDim(x2ScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleKDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    auto expectedKDimValue = inputParams.kSize / MXFP_BASEK_FACTOR + inputParams.groupNum;
    OP_CHECK_IF(
        xScaleLastDim != MXFP_MULTI_BASE_SIZE || xScaleKDim != expectedKDimValue || xScaleMDim != inputParams.mSize,
        OP_LOGE_FOR_INVALID_SHAPE(inputParams.opType, "scale1", ShapeToString(x1ScaleShape),
                                  ShapeDimsToString(expectedKDimValue, inputParams.mSize, MXFP_MULTI_BASE_SIZE)),
        return false);
    OP_CHECK_IF(
        wScaleLastDim != MXFP_MULTI_BASE_SIZE || wScaleKDim != expectedKDimValue || wScaleNDim != inputParams.nSize,
        OP_LOGE_FOR_INVALID_SHAPE(inputParams.opType, "scale2", ShapeToString(x2ScaleShape),
                                  ShapeDimsToString(expectedKDimValue, inputParams.nSize, MXFP_MULTI_BASE_SIZE)),
        return false);
    return true;
}

bool CheckShapeForHif4Quant(const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape,
                            const GQmmInputInfo& inputParams)
{
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != HIF4_TYPE_M_SCALE_DIM_NUM - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale1", std::to_string(x1ScaleDimNum), "3"),
                return false);
    OP_CHECK_IF(x2ScaleDimNum != HIF4_TYPE_K_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale2", std::to_string(x2ScaleDimNum), "3"),
                return false);
    auto xScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto xScaleMDim = static_cast<uint64_t>(x1ScaleShape.GetDim(1));
    auto xScaleLastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(2));
    auto wScaleKDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    auto wScaleNDim = static_cast<uint64_t>(x2ScaleShape.GetDim(1));
    auto wScaleLastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(2));
    auto expectedKDim = inputParams.kSize / HIF4_BASEK_FACTOR + inputParams.groupNum;
    OP_CHECK_IF(xScaleKDim != expectedKDim || xScaleMDim != inputParams.mSize || xScaleLastDim != HIF4_MULTI_BASE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams.opType, "scale1", ShapeToString(x1ScaleShape),
                                          ShapeDimsToString(expectedKDim, inputParams.mSize, HIF4_MULTI_BASE_SIZE)),
                return false);
    OP_CHECK_IF(wScaleKDim != expectedKDim || wScaleNDim != inputParams.nSize || wScaleLastDim != HIF4_MULTI_BASE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams.opType, "scale2", ShapeToString(x2ScaleShape),
                                          ShapeDimsToString(expectedKDim, inputParams.nSize, HIF4_MULTI_BASE_SIZE)),
                return false);
    return true;
}

bool CheckShapeForMxFp8Fp4Quant(const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape,
                                const GQmmInputInfo& inputParams)
{
    OP_CHECK_IF(inputParams.groupNum > GMM_MAX_GROUP_LIST_SIZE,
                OP_LOGE_FOR_INVALID_VALUE(inputParams.opType, "groupNum", std::to_string(inputParams.groupNum),
                                          "no greater than " + std::to_string(GMM_MAX_GROUP_LIST_SIZE)),
                return false);
    auto x1ScaleDimNum = x1ScaleShape.GetDimNum();
    auto x2ScaleDimNum = x2ScaleShape.GetDimNum();
    OP_CHECK_IF(x1ScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale1", std::to_string(x1ScaleDimNum), "3"),
                return false);
    OP_CHECK_IF(x2ScaleDimNum != MXFP_TYPE_K_SCALE_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams.opType, "scale2", std::to_string(x2ScaleDimNum), "3"),
                return false);
    auto xScaleKDim = static_cast<uint64_t>(x1ScaleShape.GetDim(0));
    auto xScaleMDim = static_cast<uint64_t>(x1ScaleShape.GetDim(1));
    auto xScaleLastDim = static_cast<uint64_t>(x1ScaleShape.GetDim(2));
    auto wScaleKDim = static_cast<uint64_t>(x2ScaleShape.GetDim(0));
    auto wScaleNDim = static_cast<uint64_t>(x2ScaleShape.GetDim(1));
    auto wScaleLastDim = static_cast<uint64_t>(x2ScaleShape.GetDim(2));
    auto expectedKDim = inputParams.kSize / MXFP_BASEK_FACTOR + inputParams.groupNum;

    uint64_t expectedWScaleLastDim =
        IsHif4WeightForInplaceAdd(inputParams) ? HIF4_MULTI_BASE_SIZE : MXFP_MULTI_BASE_SIZE;
    OP_CHECK_IF(xScaleKDim != expectedKDim || xScaleMDim != inputParams.mSize || xScaleLastDim != MXFP_MULTI_BASE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams.opType, "scale1", ShapeToString(x1ScaleShape),
                                          ShapeDimsToString(expectedKDim, inputParams.mSize, MXFP_MULTI_BASE_SIZE)),
                return false);
    OP_CHECK_IF(wScaleKDim != expectedKDim || wScaleNDim != inputParams.nSize || wScaleLastDim != expectedWScaleLastDim,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams.opType, "scale2", ShapeToString(x2ScaleShape),
                                          ShapeDimsToString(expectedKDim, inputParams.nSize, expectedWScaleLastDim)),
                return false);
    return true;
}

bool CheckMxFp8Fp4InputDim(const gert::Shape& xShape, const gert::Shape& wShape, const GQmmInputInfo& inputParams)
{
    OP_CHECK_IF(xShape.GetDimNum() != X_DIMS || wShape.GetDimNum() != X_DIMS,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    inputParams.opType, "x1, x2", "x1=" + ShapeToString(xShape) + ", x2=" + ShapeToString(wShape),
                    "in mxfp8+fp4 quantization mode, x1 and x2 must both be 2D because the scale capacity is "
                    "derived from the total K"),
                return false);
    const uint64_t nDim = static_cast<uint64_t>(wShape.GetDim(1));
    // N 为 fp4 内轴，mx_mmad 要求 64 对齐。
    OP_CHECK_IF(nDim % MXFP_BASEK_FACTOR != 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams.opType, "x1, x2",
                                                       "x1=" + ShapeToString(xShape) + ", x2=" + ShapeToString(wShape),
                                                       BuildErrorMsgStr("in mxfp8+fp4 quantization mode, N[", nDim,
                                                                        "] must be a multiple of ", MXFP_BASEK_FACTOR)),
                return false);
    return true;
}

uint64_t GetTilingKeyForInplaceAdd(const GQmmInputInfo& inputParams)
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams.transB), static_cast<uint64_t>(inputParams.transA),
                              static_cast<uint64_t>(inputParams.kernelType));
}

} // namespace QuantGroupedMatmulInplaceAdd
