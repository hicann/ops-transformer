/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_finalize_routing_weight_quant_checker.h"
#include "common/include/op_host/op_const_def_transformer.h"
#include "log/log.h"
#include "common/include/err/ops_err.h"
#include "../../../../grouped_matmul/op_host/grouped_matmul_host_util.h"

namespace optiling {
namespace GMMFinalizeRoutingWeightQuantChecker {

using namespace GMMFRTilingTools;
using namespace GMMFRTilingConstants;
using namespace GmmConstant;

ge::graphStatus GMMFinalizeRoutingWeightQuantChecker::CheckGroupedMatmulFinalizeRoutingWeightQuant(NpuArch npuArch,
                                                                                                   uint64_t aicNum,
                                                                                                   uint64_t aivNum)
{
    npuArch_ = npuArch;
    if (unlikely(!(IsMxFp8Fp4(inputParams_) || IsMxFp8Hif4(inputParams_)))) {
        std::string aDtypeStr = ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype);
        std::string bDtypeStr = ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype);
        std::string scaleDtypeStr = ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype);
        std::string incorrect = "x=" + aDtypeStr + ", weight=" + bDtypeStr + ", scale=" + scaleDtypeStr;
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "x, weight, scale", incorrect.c_str(),
                                              "the dtype combination of x, weight, scale is invalid");
        return ge::GRAPH_FAILED;
    }
    if (unlikely(!CheckAttrs())) {
        return ge::GRAPH_FAILED;
    }
    if (unlikely(!CheckDtypeValid())) {
        return ge::GRAPH_FAILED;
    }
    if (unlikely(!CheckWeightFormat())) {
        return ge::GRAPH_FAILED;
    }
    if (unlikely(!CheckShapeValid())) {
        return ge::GRAPH_FAILED;
    }
    if (unlikely(!CheckFinalizeRoutingInputsShape(inputParams_, frInfo_, context_))) {
        return ge::GRAPH_FAILED;
    }
    if (unlikely(!CheckSharedInput(inputParams_, frInfo_, context_))) {
        return ge::GRAPH_FAILED;
    }
    if (unlikely(!CheckCoreNum(aicNum, aivNum))) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckAttrs()
{
    if (unlikely(inputParams_.groupListType != 0 && inputParams_.groupListType != 1)) {
        std::string groupListTypeVal = std::to_string(inputParams_.groupListType);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "group_list_type", groupListTypeVal.c_str(),
                                              "the value of group_list_type must be 0 or 1");
        return false;
    }
    if (unlikely(frInfo_.outputDtype != 0)) {
        std::string outDtypeVal = std::to_string(frInfo_.outputDtype);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "dtype", outDtypeVal.c_str(),
                                              "the value of dtype must be 0 (float32)");
        return false;
    }
    if (unlikely(inputParams_.transA)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "transpose_x", "true", "the value of transpose_x must be false");
        return false;
    }
    if (unlikely(!inputParams_.transB)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "transpose_w", "false", "the value of transpose_w must be true");
        return false;
    }

    return true;
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckDtypeValid()
{
    if (unlikely(!CheckCommonInputsDtype(frInfo_, context_))) {
        return false;
    }
    return CheckInputsDtype();
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckInputsDtype()
{
    if (unlikely(inputParams_.biasDtype != ge::DT_BF16)) {
        std::string biasDtypeStr = ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "bias", biasDtypeStr.c_str(), "the dtype of bias must be BF16");
        return false;
    }
    if (unlikely(inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0)) {
        std::string ptsDtypeStr = ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(OP_NAME, "pertoken_scale", ptsDtypeStr.c_str(),
                                              "the dtype of pertoken_scale must be FLOAT8_E8M0");
        return false;
    }
    return true;
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckWeightFormat()
{
    if (unlikely(inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ)) {
        std::string wFmtStr = ge::TypeUtils::FormatToSerialString(inputParams_.bFormat);
        OP_LOGE_FOR_INVALID_FORMAT(OP_NAME, "weight", wFmtStr.c_str(), "FRACTAL_NZ");
        return false;
    }
    return true;
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckShapeValid()
{
    if (unlikely(!CheckKNAlign())) {
        return false;
    }
    if (unlikely(!CheckWeightShape())) {
        return false;
    }
    if (inputParams_.hasBias && unlikely(!CheckBiasShape())) {
        return false;
    }
    if (unlikely(!CheckMxScaleShape())) {
        return false;
    }
    if (unlikely(!CheckYShape())) {
        return false;
    }
    return true;
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckYShape()
{
    const gert::Shape& yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    return CheckInputTensorShape("y", yShape,
                                 {static_cast<int64_t>(frInfo_.outputBs), static_cast<int64_t>(inputParams_.nSize)});
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckBiasShape()
{
    const gert::Shape& biasShape = context_->GetOptionalInputShape(BIAS_INDEX)->GetStorageShape();
    return CheckInputTensorShape(
        "bias", biasShape, {static_cast<int64_t>(inputParams_.groupNum), static_cast<int64_t>(inputParams_.nSize)});
}

int64_t GMMFinalizeRoutingWeightQuantChecker::GetC0Size(ge::DataType dType) const
{
    if (dType == ge::DT_FLOAT4_E2M1 || dType == Ops::Transformer::DT_HIFLOAT4) {
        return C0_SIZE_64;
    }
    return C0_SIZE_32;
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckWeightShape()
{
    auto wShapePtr = context_->GetInputShape(W_INDEX);
    const gert::Shape& wShape = wShapePtr->GetOriginShape();
    std::vector<int64_t> expectedOriShape;
    if (inputParams_.transB) {
        expectedOriShape = {static_cast<int64_t>(inputParams_.groupNum), static_cast<int64_t>(inputParams_.nSize),
                            static_cast<int64_t>(inputParams_.kSize)};
    } else {
        expectedOriShape = {static_cast<int64_t>(inputParams_.groupNum), static_cast<int64_t>(inputParams_.kSize),
                            static_cast<int64_t>(inputParams_.nSize)};
    }
    if (!CheckInputTensorShape("weight", wShape, expectedOriShape)) {
        return false;
    }

    const gert::Shape& wStorageShape = wShapePtr->GetStorageShape();
    int64_t c0Size = GetC0Size(inputParams_.bDtype);
    // w storage (NZ, transB=true): (E, ceil(K/C0), ceil(N/16), 16, C0)
    // w storage (NZ, transB=false): (E, ceil(N/C0), ceil(K/16), 16, C0)
    std::vector<int64_t> expectedStorageShape;
    if (inputParams_.transB) {
        expectedStorageShape = {static_cast<int64_t>(inputParams_.groupNum),
                                GroupedMatmul::CeilDiv<int64_t>(static_cast<int64_t>(inputParams_.kSize), c0Size),
                                GroupedMatmul::CeilDiv<int64_t>(static_cast<int64_t>(inputParams_.nSize), CUBE_BLOCK),
                                CUBE_BLOCK, c0Size};
    } else {
        expectedStorageShape = {static_cast<int64_t>(inputParams_.groupNum),
                                GroupedMatmul::CeilDiv<int64_t>(static_cast<int64_t>(inputParams_.nSize), c0Size),
                                GroupedMatmul::CeilDiv<int64_t>(static_cast<int64_t>(inputParams_.kSize), CUBE_BLOCK),
                                CUBE_BLOCK, c0Size};
    }
    return CheckInputTensorShape("weight", wStorageShape, expectedStorageShape);
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckMxScaleShape()
{
    const gert::Shape& scaleShape = context_->GetInputShape(SCALE_INDEX)->GetOriginShape();
    int64_t scaleKSize = GroupedMatmul::CeilDiv<int64_t>(static_cast<int64_t>(inputParams_.kSize), MX_SCALE_BLOCK_SIZE);
    int64_t scaleLastDim = IsMxFp8Hif4(inputParams_) ? MX_HIF4_SCALE_LAST_DIM : MX_SCALE_LAST_DIM;
    std::vector<int64_t> expectedScaleShape = {static_cast<int64_t>(inputParams_.groupNum),
                                               static_cast<int64_t>(inputParams_.nSize), scaleKSize, scaleLastDim};
    if (!CheckInputTensorShape("scale", scaleShape, expectedScaleShape)) {
        return false;
    }

    const gert::Shape& pertokenScaleShape = context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX)->GetOriginShape();
    std::vector<int64_t> expectedPerTokenScaleShape = {static_cast<int64_t>(inputParams_.mSize), scaleKSize,
                                                       MX_SCALE_LAST_DIM};
    return CheckInputTensorShape("pertoken_scale", pertokenScaleShape, expectedPerTokenScaleShape);
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckKNAlign()
{
    if (unlikely(inputParams_.kSize % ALIGN_SIZE_K != 0)) {
        std::string shapeVal = "K=" + std::to_string(inputParams_.kSize);
        std::string reason = "the K dim of x must be aligned to " + std::to_string(ALIGN_SIZE_K);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "x", shapeVal.c_str(), reason.c_str());
        return false;
    }
    if (unlikely(inputParams_.nSize % ALIGN_SIZE_N != 0)) {
        std::string shapeVal = "N=" + std::to_string(inputParams_.nSize);
        std::string reason = "the N dim of weight must be aligned to " + std::to_string(ALIGN_SIZE_N);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(OP_NAME, "weight", shapeVal.c_str(), reason.c_str());
        return false;
    }
    return true;
}

bool GMMFinalizeRoutingWeightQuantChecker::CheckCoreNum(uint64_t aicNum, uint64_t aivNum)
{
    if (unlikely(aicNum == 0 || aivNum == 0)) {
        std::string aicVal = std::to_string(aicNum);
        std::string aivVal = std::to_string(aivNum);
        std::string aicAivVal = aicVal + ", " + aivVal;
        std::string reason = "aicNum and aivNum must be greater than 0";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "aicNum, aivNum", aicAivVal.c_str(), reason.c_str());
        return false;
    }
    uint64_t expectedRatio = npuArch_ == NpuArch::DAV_3510 ? AIC_AIV_CORE_RATIO_2 : AIC_AIV_CORE_RATIO_1;
    if (unlikely(aicNum * expectedRatio != aivNum)) {
        std::string aicVal = std::to_string(aicNum);
        std::string aivVal = std::to_string(aivNum);
        std::string aicAivVal = aicVal + ", " + aivVal;
        std::string reason = "aicNum : aivNum must be 1 : " + std::to_string(expectedRatio);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "aicNum, aivNum", aicAivVal.c_str(), reason.c_str());
        return false;
    }
    return true;
}

} // namespace GMMFinalizeRoutingWeightQuantChecker
} // namespace optiling
