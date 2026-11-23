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
 * \file quant_grouped_matmul_inplace_add_basic_api_tiling.cpp
 * \brief
 */
#include "quant_grouped_matmul_inplace_add_basic_api_tiling.h"

#include <alog_pub.h>

#include <algorithm>
#include <climits>

#include "../../op_kernel/arch35/qgmm_inplace_add_tiling_key.h"
#include "log/log.h"
#include "quant_grouped_matmul_inplace_add_tiling_utils.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"

using namespace Ops::Transformer::OpTiling;
using namespace QuantGroupedMatmulInplaceAdd;
using namespace optiling::GmmConstant;
namespace optiling {

QGMMInplaceAddBasicApiTiling::QGMMInplaceAddBasicApiTiling(gert::TilingContext* context)
    : GroupedQmmBasicApiTiling(context)
{}

bool QGMMInplaceAddBasicApiTiling::IsCapable()
{
    const auto* compileInfo = context_->GetCompileInfo<GMMCompileInfo>();
    if (compileInfo == nullptr) {
        return false;
    }
    const auto* xDesc = context_->GetInputDesc(X_INDEX);
    if (xDesc == nullptr) {
        return false;
    }
    const auto xDtype = xDesc->GetDataType();
    if (xDtype == ge::DT_FLOAT8_E4M3FN || xDtype == ge::DT_FLOAT8_E5M2 || xDtype == ge::DT_FLOAT4_E2M1) {
        if (compileInfo->npuArch != NpuArch::DAV_3510) {
            return false;
        }
        CalBasicBlock();
        // MX UB uses the CGMCT tiling data and is handled by the next template.
        return !IsMxUbCandidateForInplaceAdd(inputParams_, basicTiling_, aicoreParams_, npuArch_);
    }
    if (xDtype != ge::DT_HIFLOAT8) {
        return false;
    }
    const auto* scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    if (scaleStorageShape == nullptr) {
        return false;
    }
    const auto& wScaleShape = scaleStorageShape->GetOriginShape();
    const auto scaleDimNum = wScaleShape.GetDimNum();
    const bool isPerTensor =
        scaleDimNum == 1 || (scaleDimNum == QuantGroupedMatmulInplaceAdd::DIM_NUM_2D && wScaleShape.GetDim(1) == 1);
    if (isPerTensor) {
        CalBasicBlock();
    }
    return isPerTensor;
}

ge::graphStatus QGMMInplaceAddBasicApiTiling::GetShapeAttrsInfo()
{
    const auto* compileInfo = context_->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "compileInfo is nullptr."),
                return ge::GRAPH_FAILED);

    if (compileInfo->supportMmadF16S8) {
        return ge::GRAPH_PARAM_INVALID;
    }

    return GroupedQmmBasicApiTiling::GetShapeAttrsInfo();
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeAttrs()
{
    return AnalyzeAttrsForInplaceAdd(context_, inputParams_);
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeDtype()
{
    return AnalyzeDtypeForInplaceAdd(context_, inputParams_) && CheckDtypeForInplaceAdd(inputParams_) &&
           (!IsMicroScaling() || CheckFormatForInplaceAdd(inputParams_));
}

bool QGMMInplaceAddBasicApiTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_IF(
        xStorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x1", "nullptr", "xStorageShape cannot be nullptr"),
        return false);
    gert::Shape xShape = xStorageShape->GetOriginShape();

    auto wStorageShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_IF(
        wStorageShape == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "x2", "nullptr", "wStorageShape cannot be nullptr"),
        return false);
    const gert::Shape& wShape = wStorageShape->GetOriginShape();

    auto scaleStorageShape = context_->GetInputShape(SCALE_INDEX);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale2", "nullptr",
                                                      "scaleStorageShape cannot be nullptr"),
                return false);
    const gert::Shape& wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(
        scaleDimNum < 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "scale2", std::to_string(scaleDimNum), "positive integer"),
        return false);

    auto x1ScaleStorageShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(x1ScaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale1", "nullptr",
                                                      "x1ScaleStorageShape cannot be nullptr"),
                return false);
    gert::Shape x1ScaleShape = x1ScaleStorageShape->GetOriginShape();

    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    NormalizeEmptyReductionForInplaceAdd(context_, wShape, inputParams_.groupNum, xShape, x1ScaleShape);
    OP_CHECK_IF(!CheckShapeNotNegative(xShape, wShape, inputParams_),
                OP_LOGE(inputParams_.opName, "CheckShapeNotNegative failed."), return false);
    OP_CHECK_IF(!CheckXAndWeightKDim(xShape, wShape, inputParams_),
                OP_LOGE(inputParams_.opName, "CheckXAndWeightKDim failed."), return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);

    if (npuArch_ == NpuArch::DAV_3510 && IsMicroScaling()) {
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        if (inputParams_.aDtype == ge::DT_FLOAT4_E2M1) {
            OP_CHECK_IF(!CheckMx4PackedShapeForInplaceAdd(xShape, wShape, inputParams_),
                        OP_LOGE(inputParams_.opName, "CheckMx4PackedShapeForInplaceAdd failed."), return false);
        }
        OP_CHECK_IF(!CheckShapeForMxQuant(x1ScaleShape, wScaleShape, inputParams_),
                    OP_LOGE(inputParams_.opName, "CheckShapeForMxQuant failed."), return false);
    } else {
        // Basic API analyzes inputs before the framework calls IsCapable.
        // Leave the mixed MX scale shape check to the general tiling template.
        OP_CHECK_IF(
            !IsMxFp8Fp4() && !IsMicroScaling() && !CheckShapeForHif8Quant(x1ScaleShape, wScaleShape, inputParams_),
            OP_LOGE(inputParams_.opName, "CheckShapeForHif8Quant failed."), return false);
        inputParams_.aQuantMode = optiling::QuantMode::PERTENSOR_MODE;
        inputParams_.bQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    }
    SetKernelType();
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    return true;
}

bool QGMMInplaceAddBasicApiTiling::CheckCoreNum() const
{
    return CheckCoreNumForInplaceAdd(context_, inputParams_);
}

ge::graphStatus QGMMInplaceAddBasicApiTiling::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    OP_LOGD(inputParams_.opName, "%ld", LogQuantParams());
    return ge::GRAPH_SUCCESS;
}

int64_t QGMMInplaceAddBasicApiTiling::LogQuantParams() const
{
    std::ostringstream oss;
    oss << "QGmmInplaceAddBasicApiParams: groupNum = " << tilingData_.gmmQuantParams.groupNum
        << ", aQuantMode = " << tilingData_.gmmQuantParams.aQuantMode
        << ", bQuantMode = " << tilingData_.gmmQuantParams.bQuantMode
        << ", groupListType = " << static_cast<uint32_t>(tilingData_.gmmQuantParams.groupListType);
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
    return 0;
}

ge::graphStatus QGMMInplaceAddBasicApiTiling::DoLibApiTiling()
{
    tilingData_.mmTilingData.m = inputParams_.mSize;
    tilingData_.mmTilingData.n = inputParams_.nSize;
    tilingData_.mmTilingData.k = inputParams_.kSize;
    if (inputParams_.mSize == 0 || inputParams_.nSize == 0) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(GroupedQmmBasicApiTiling::CalL1Tiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"), return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.kAL1 = basicTiling_.stepKa * basicTiling_.baseK;
    tilingData_.mmTilingData.kBL1 = basicTiling_.stepKb * basicTiling_.baseK;
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        tilingData_.mmTilingData.scaleKAL1 = std::min(
            std::max(basicTiling_.scaleFactorA * basicTiling_.stepKa, basicTiling_.scaleFactorB * basicTiling_.stepKb) *
                basicTiling_.baseK,
            inputParams_.kSize);
        tilingData_.mmTilingData.scaleKBL1 = tilingData_.mmTilingData.scaleKAL1;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t QGMMInplaceAddBasicApiTiling::GetTilingKey() const
{
    return GetTilingKeyForInplaceAdd(inputParams_);
}

ge::graphStatus QGMMInplaceAddBasicApiTiling::PostTiling()
{
    return SaveTilingDataToContext(tilingData_);
}

} // namespace optiling
