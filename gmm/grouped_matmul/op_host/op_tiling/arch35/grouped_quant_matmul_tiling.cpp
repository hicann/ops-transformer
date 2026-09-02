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
 * \file grouped_quant_matmul_tiling.cpp
 * \brief
 */
#include "grouped_quant_matmul_tiling.h"
#include <sstream>

#include <alog_pub.h>

#include "grouped_quant_basic_api_matmul_tiling.h"
#include "grouped_quant_matmul_info_factory.h"

#include "log/log.h"
#include "log/error_code.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "../../../op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
using namespace Ops::Transformer::OpTiling;
using namespace GroupedMatmul;
using namespace optiling::GmmConstant;
using GMMQuantTilingData = GroupedMatmulTilingData::GMMQuantTilingData;
using GMMQuantParams = GroupedMatmulTilingData::GMMQuantParams;
namespace {
static optiling::GroupedQuantMatmulInfoFactory g_groupedQuantMatmulInfoFactory;

bool CheckGroupListType(const optiling::GQmmInputInfo &inputParams)
{
    OP_CHECK_IF(
        inputParams.groupListType != GROUPLIST_TYPE_CUMSUM && inputParams.groupListType != GROUPLIST_TYPE_COUNT &&
            inputParams.groupListType != GROUPLIST_TYPE_SPARSE_M,
        OP_LOGE_FOR_INVALID_VALUE(inputParams.opType, "groupListType", std::to_string(inputParams.groupListType),
                                  "0(cumsum), 1(count) or 2(sparse)"),
        return false);
    return true;
}
} // namespace
namespace optiling {
GroupedQmmTiling::GroupedQmmTiling(gert::TilingContext *context)
    : TilingBaseClass(context),
      inputParams_(*(g_groupedQuantMatmulInfoFactory.Get()))
{
    Reset();
}

bool GroupedQmmTiling::IsCapable()
{
    return true;
}

void GroupedQmmTiling::Reset()
{
    tilingData_ = GMMQuantTilingData();
}

ge::graphStatus GroupedQmmTiling::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<GMMCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfoPtr is null."),
                    return ge::GRAPH_FAILED);

        aicoreParams_.aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0aSize = compileInfoPtr->l0ASize;
        aicoreParams_.l0bSize = compileInfoPtr->l0BSize;
        aicoreParams_.l0cSize = compileInfoPtr->l0CSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }

    OP_LOGI(context_, "Platform info: aicNum(%lu) ubSize(%lu) l1Size(%lu) l0aSize(%lu) l0bSize(%lu) l0cSize(%lu).",
            aicoreParams_.aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0aSize,
            aicoreParams_.l0bSize, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

bool GroupedQmmTiling::IsMicroScaling() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0;
}

bool GroupedQmmTiling::IsMxfp4() const
{
    return (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) &&
           (inputParams_.bDtype == ge::DT_FLOAT4_E2M1 || inputParams_.bDtype == ge::DT_FLOAT4_E1M2);
}

bool GroupedQmmTiling::IsMultiTensorWeight() const
{
    auto weightShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    return weightShape != nullptr && weightShape->GetOriginShape().GetDimNum() == SPLIT_M_W_DIMS - 1;
}

bool GroupedQmmTiling::IsWeightNzMultiTensorLayout() const
{
    return inputParams_.groupType == SPLIT_M && inputParams_.isSingleX && inputParams_.isSingleY &&
           inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && IsMultiTensorWeight();
}

uint16_t GroupedQmmTiling::GetTensorListSize(uint32_t index) const
{
    uint16_t count = 0U;
    while (count < GroupedMatmul::MAX_TENSOR_CONT && context_->GetDynamicInputShape(index, count) != nullptr) {
        ++count;
    }
    return count;
}

bool GroupedQmmTiling::CheckWeightTensorListForWeightNz() const
{
    OP_CHECK_IF(IsWeightNzMultiTensorLayout() && inputParams_.hasBias,
                OP_LOGE(inputParams_.opName, "In WeightNz single-multi-single mode, current hasBias is true; "
                                             "expected hasBias to be false."),
                return false);
    if (!IsWeightNzMultiTensorLayout()) {
        return true;
    }
    const uint16_t weightTensorNum = GetTensorListSize(WEIGHT_INDEX);
    const uint16_t scaleTensorNum = GetTensorListSize(SCALE_INDEX);
    OP_CHECK_IF(weightTensorNum == 0,
                OP_LOGE_FOR_INVALID_LISTSIZE(inputParams_.opType, "weight", "0", "positive integer"), return false);
    OP_CHECK_IF(weightTensorNum != inputParams_.groupNum,
                OP_LOGE_FOR_INVALID_LISTSIZE(
                    inputParams_.opType, "weight, groupList",
                    std::to_string(weightTensorNum) + ", " + std::to_string(inputParams_.groupNum),
                    "in WeightNz single-multi-single mode, expected weight tensor count to equal group count"),
                return false);
    OP_CHECK_IF(scaleTensorNum != weightTensorNum,
                OP_LOGE_FOR_INVALID_LISTSIZE(
                    inputParams_.opType, "scale, weight",
                    std::to_string(scaleTensorNum) + ", " + std::to_string(weightTensorNum),
                    "in WeightNz single-multi-single mode, expected scale tensor count to equal weight tensor count"),
                return false);
    return true;
}

uint64_t GroupedQmmTiling::GetWeightLogicalNSize(const gert::Shape &wShape) const
{
    return static_cast<uint64_t>(inputParams_.transB ? wShape.GetDim(wShape.GetDimNum() - LAST_SECOND_DIM_INDEX) :
                                                       wShape.GetDim(wShape.GetDimNum() - LAST_FIRST_DIM_INDEX));
}

bool GroupedQmmTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    if (attrs) {
        OP_CHECK_IF(attrs->GetAttrNum() < ATTR_INDEX_ACT_TYPE + 1,
                    OP_LOGE(inputParams_.opName, "The num of attrs should be greater than %lu, actual is %zu",
                            ATTR_INDEX_ACT_TYPE + 1, attrs->GetAttrNum()),
                    return false);
        const int64_t *splitItemPtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_SPLIT_ITEM);
        const bool *transposeWeightPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
        const bool *transposeXPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
        const int64_t *groupTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPTYPE);
        const int64_t *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE); // 通路保证非负数
        const int64_t *actTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_ACT_TYPE);

        inputParams_.transB = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
        inputParams_.transA = transposeXPtr != nullptr ? *transposeXPtr : false;
        inputParams_.groupType = groupTypePtr != nullptr ? *groupTypePtr : inputParams_.groupType;
        inputParams_.splitItem = splitItemPtr != nullptr ? *splitItemPtr : inputParams_.splitItem;
        inputParams_.actType = actTypePtr != nullptr ? *actTypePtr : inputParams_.actType;
        inputParams_.groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : inputParams_.groupListType;
    }
    OP_CHECK_IF(!CheckGroupListType(inputParams_), OP_LOGE(inputParams_.opName, "Check groupListType failed."),
                return false);
    OP_CHECK_IF(
        inputParams_.groupType != SPLIT_M && inputParams_.groupType != SPLIT_K,
        OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "groupType", std::to_string(inputParams_.groupType), "0 or 2"),
        return false);
    OP_CHECK_IF(
        (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) &&
            inputParams_.groupType != SPLIT_M,
        OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "groupType", std::to_string(inputParams_.groupType), "0"),
        return false);
    if (inputParams_.groupType == SPLIT_M) {
        OP_CHECK_IF(inputParams_.transA,
                    OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "transpose_x1",
                                              inputParams_.transA ? "true" : "false", "false"),
                    return false);
    } else {
        OP_CHECK_IF(!CheckTransposeAndFormatByGroupType(),
                    OP_LOGE(inputParams_.opName, "CheckTransposeAndFormatByGroupType failed."), return false);
    }
    inputParams_.isSingleX = (context_->GetDynamicInputDesc(X_INDEX, 1) == nullptr);
    inputParams_.isSingleW = (context_->GetDynamicInputDesc(WEIGHT_INDEX, 1) == nullptr);
    // 2: when x is multi-tensor, y is single-tensor; 3: when x is single-tensor, y is single-tensor
    inputParams_.isSingleY = (inputParams_.splitItem == 2 || inputParams_.splitItem == 3);
    return true;
}

bool GroupedQmmTiling::CheckTransposeAndFormatByGroupType() const
{
    OP_CHECK_IF(
        !inputParams_.transA,
        OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "transpose_x1", inputParams_.transA ? "true" : "false", "true"),
        return false);
    OP_CHECK_IF(
        inputParams_.transB,
        OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "transpose_x2", inputParams_.transB ? "true" : "false", "false"),
        return false);
    OP_CHECK_IF(inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                    inputParams_.opType, "weight", ge::TypeUtils::FormatToSerialString(inputParams_.bFormat),
                    "when group type is 2, format of weight must be within the range ND"),
                return false);
    return true;
}

bool GroupedQmmTiling::CheckBiasDtype() const
{
    if (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) {
        OP_CHECK_IF(inputParams_.biasDtype != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opType, "bias",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype), "FLOAT"),
                    return false);
    } else if (inputParams_.aDtype == ge::DT_INT8) {
        if (inputParams_.cDtype == ge::DT_BF16) {
            OP_CHECK_IF(inputParams_.biasDtype != ge::DT_INT32 && inputParams_.biasDtype != ge::DT_BF16 &&
                            inputParams_.biasDtype != ge::DT_FLOAT,
                        OP_LOGE(inputParams_.opName,
                                "The dtype of bias should be INT32, BF16 or FLOAT when the dtype of x is INT8 and the \
dtype of output is BF16, actual is %s.",
                                ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
                        return false);
        } else if (inputParams_.cDtype == ge::DT_FLOAT16) {
            OP_CHECK_IF(inputParams_.biasDtype != ge::DT_INT32 && inputParams_.biasDtype != ge::DT_FLOAT16 &&
                            inputParams_.biasDtype != ge::DT_FLOAT,
                        OP_LOGE(inputParams_.opName,
                                "The dtype of bias should be INT32, FLOAT16 or FLOAT when the dtype of x is INT8 and \
the dtype of output is FLOAT16, actual is %s.",
                                ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
                        return false);
        } else if (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_INT32) {
            OP_CHECK_IF(
                inputParams_.biasDtype != ge::DT_INT32,
                OP_LOGE(inputParams_.opName,
                        "The dtype of bias should be INT32 when the dtype of x is INT8 and the dtype of output \
is INT8 or INT32, actual is %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str()),
                return false);
        } else {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opType, "y", ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype),
                "the dtype of y must be within the range BF16, FLOAT16, INT8 or INT32 when the dtype of x is INT8");
            return false;
        }
    } else {
        OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "bias", "nonnull", "nullptr");
        return false;
    }
    return true;
}

bool GroupedQmmTiling::CheckDtypeForWeightNz(bool isPertokenScaleNull) const
{
    bool isA8W8Int = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8;
    bool isA8W8Fp = inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN && inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN;
    auto isFp4Dtype = [](ge::DataType dt) { return dt == ge::DT_FLOAT4_E2M1 || dt == ge::DT_FLOAT4_E1M2; };
    bool isA4W4Fp = isFp4Dtype(inputParams_.aDtype) && isFp4Dtype(inputParams_.bDtype);
    bool isA4W4Int = inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4;
    OP_CHECK_IF(
        !(isA8W8Int || isA8W8Fp || isA4W4Fp || isA4W4Int),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opType, "x, weight",
            ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                         ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
            "when the format of weight is FRACTAL_NZ, the dtypes of x and weight must be within the range INT8, "
            "int4, FLOAT8_E4M3FN or FLOAT4 (E2M1/E1M2)"),
        return false);
    OP_CHECK_IF((isA8W8Int || isA8W8Fp || isA4W4Fp || isA4W4Int) && inputParams_.cDtype == ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opType, "y",
                                                      ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype),
                                                      "when the format of weight is FRACTAL_NZ, dtype of y cannot be "
                                                      "INT8"),
                return false);
    if (!isPertokenScaleNull) {
        OP_CHECK_IF(!CheckPertokenScaleDtypeForWeightNz(isA8W8Int, isA8W8Fp, isA4W4Fp, isA4W4Int),
                    OP_LOGE(inputParams_.opName, "CheckPertokenScaleDtypeForWeightNz failed."), return false);
    } else {
        static const std::vector<ge::DataType> legalScaleDtypes = {ge::DT_UINT64, ge::DT_INT64, ge::DT_FLOAT,
                                                                   ge::DT_BF16};
        OP_CHECK_IF(std::find(legalScaleDtypes.begin(), legalScaleDtypes.end(), inputParams_.scaleDtype) ==
                        legalScaleDtypes.end(),
                    OP_LOGE(inputParams_.opName,
                            "When the weight is in Nz format and the pertokenScale is None, the dtype of scale should \
be in {UINT64, INT64, FLOAT, BF16}, actual is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                    return false);
    }
    return true;
}

bool GroupedQmmTiling::CheckPertokenScaleDtypeForWeightNz(bool isA8W8Int, bool isA8W8Fp, bool isA4W4Fp,
                                                          bool isA4W4Int) const
{
    if (isA8W8Int) {
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT ||
                        (inputParams_.scaleDtype != ge::DT_BF16 && inputParams_.scaleDtype != ge::DT_FLOAT),
                    OP_LOGE(context_->GetNodeName(),
                            "When the weight is Nz format and x/weight's dtype are INT8 and pertokenScale exists, \
the dtype of pertokenScale should be FLOAT and the dtype of scale should be in {BF16, FLOAT}, actual are %s/%s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                    return false);
    } else if (isA4W4Int) {
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT || inputParams_.scaleDtype != ge::DT_UINT64,
                    OP_LOGE(context_->GetNodeName(),
                            "When the weight is Nz format and x/weight's dtype are INT4, "
                            "the dtype of pertokenScale must be FLOAT and scale must be UINT64, actual are %s/%s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                    return false);
    } else if (isA8W8Fp || isA4W4Fp) {
        OP_CHECK_IF(
            inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0 || inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0,
            OP_LOGE(context_->GetNodeName(),
                    "When the weight is Nz format and x/weight's dtype are FLOAT8_E4M3 or "
                    "FLOAT4 (E2M1/E1M2), the dtype "
                    "of pertokenScale and scale should be FLOAT8_E8M0, actual are %s/%s.",
                    ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                    ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
            return false);
    }
    return true;
}

bool GroupedQmmTiling::ValidateAAndWDtype(const std::vector<ge::DataType> &legalInputDtypes) const
{
    auto xDesc = context_->GetDynamicInputDesc(X_INDEX, 0);
    OP_CHECK_IF(xDesc == nullptr, OP_LOGE(context_->GetNodeName(), "xDesc is nullptr."), return false);
    inputParams_.aDtype = xDesc->GetDataType();
    OP_CHECK_IF(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.aDtype) == legalInputDtypes.end(),
        OP_LOGE(inputParams_.opName,
                "The dtype of x should be in {INT8, HIFLOAT8, FLOAT8_E4M3, FLOAT8_E5M2, "
                "FLOAT4_E2M1, FLOAT4_E1M2}, actual is %s.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str()),
        return false);
    auto wDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wDesc == nullptr, OP_LOGE(context_->GetNodeName(), "wDesc is nullptr."), return false);
    inputParams_.bDtype = wDesc->GetDataType();
    OP_CHECK_IF(
        std::find(legalInputDtypes.begin(), legalInputDtypes.end(), inputParams_.bDtype) == legalInputDtypes.end(),
        OP_LOGE(inputParams_.opName,
                "The dtype of weight should be in {INT8, HIFLOAT8, FLOAT8_E4M3, FLOAT8_E5M2, "
                "FLOAT4_E2M1, FLOAT4_E1M2}, actual is %s.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
        return false);
    return true;
}

bool GroupedQmmTiling::AnalyzeDtype()
{
    static const std::vector<ge::DataType> legalInputDtypes = {
        ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2};
    OP_CHECK_IF(!ValidateAAndWDtype(legalInputDtypes), OP_LOGE(inputParams_.opName, "ValidateAAndWDtype failed."),
                return false);
    auto wDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    if (wDesc != nullptr) {
        inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(wDesc->GetStorageFormat()));
    }
    if (inputParams_.aDtype == ge::DT_FLOAT4_E1M2 || inputParams_.bDtype == ge::DT_FLOAT4_E1M2) {
        OP_CHECK_IF(inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(inputParams_.opName,
                            "Format of weight only supports FORMAT_FRACTAL_NZ when xDType or weightDtype is "
                            "DT_FLOAT4_E1M2, but got xDType [%s], weightDtype [%s], weight format [%s].",
                            ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString(),
                            ge::TypeUtils::DataTypeToAscendString(inputParams_.bDtype).GetString(),
                            ge::TypeUtils::FormatToAscendString(inputParams_.bFormat).GetString()),
                    return false);
    }
    auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
    inputParams_.hasBias = !(biasStorageShape == nullptr || biasStorageShape->GetStorageShape().GetShapeSize() == 0);
    auto biasDesc = context_->GetDynamicInputDesc(BIAS_INDEX, 0);
    OP_CHECK_IF(inputParams_.hasBias && biasDesc == nullptr,
                OP_LOGE(inputParams_.opName, "Bias from tensor is not nullptr, but bias from desc is nullptr."),
                return false);
    inputParams_.biasDtype = inputParams_.hasBias ? biasDesc->GetDataType() : inputParams_.biasDtype;
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(context_->GetNodeName(), "yDesc is nullptr."), return false);
    inputParams_.cDtype = yDesc->GetDataType();
    auto scaleDesc = context_->GetDynamicInputDesc(SCALE_INDEX, 0);
    inputParams_.scaleDtype = scaleDesc != nullptr && inputParams_.cDtype != ge::DT_INT32 ? scaleDesc->GetDataType() :
                                                                                            inputParams_.scaleDtype;
    if (scaleDesc != nullptr && inputParams_.cDtype != ge::DT_INT32) {
        inputParams_.scaleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(scaleDesc->GetStorageFormat()));
        const bool scaleNzFormat = inputParams_.scaleFormat == ge::FORMAT_FRACTAL_NZ;
        auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
        const bool singleScaleNzStorage =
            scaleStorageShape != nullptr &&
            scaleStorageShape->GetOriginShape().GetDimNum() == MXFP_TYPE_M_SCALE_DIM_NUM &&
            scaleStorageShape->GetStorageShape().GetDimNum() == MXFP_SCALE_NZ_DIM_NUM;
        const bool listScaleNzStorage =
            scaleStorageShape != nullptr &&
            scaleStorageShape->GetOriginShape().GetDimNum() == MXFP_TYPE_M_SCALE_DIM_NUM - 1 &&
            scaleStorageShape->GetStorageShape().GetDimNum() == MXFP_SCALE_NZ_DIM_NUM - 1;
        inputParams_.scaleNzFlag = IsMicroScaling() && scaleNzFormat && inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ &&
                                   (singleScaleNzStorage || listScaleNzStorage);
    }
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PER_TOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_CHECK_IF(!CheckDtypeForWeightNz(nullptr == pertokenScaleDesc),
                    OP_LOGE(inputParams_.opName, "CheckDtypeForWeightNz failed."), return false);
    }
    if (inputParams_.hasBias) {
        OP_CHECK_IF(!CheckBiasDtype(), OP_LOGE(inputParams_.opName, "CheckBiasDtype failed."), return false);
    }
    return true;
}

bool GroupedQmmTiling::CheckQuantParamsForMXTypeM(const gert::Shape &xScaleShape,
                                                  const gert::StorageShape *wScaleStorageShape,
                                                  uint64_t expectedNSize) const
{
    OP_CHECK_IF(wScaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "scale", "nullptr", "nonnull"), return false);
    const gert::Shape &wScaleShape = wScaleStorageShape->GetOriginShape();
    const bool isScaleNzStorage = inputParams_.scaleNzFlag;
    if (isScaleNzStorage) {
        OP_CHECK_IF(!CheckScaleNzShape(wScaleStorageShape, expectedNSize),
                    OP_LOGE(inputParams_.opName, "CheckScaleNzShape failed."), return false);
    }
    return CheckQuantParamsForMXTypeMImpl(xScaleShape, wScaleShape, isScaleNzStorage, expectedNSize);
}

bool GroupedQmmTiling::CheckQuantParamsForMXTypeM(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape,
                                                  uint64_t expectedNSize) const
{
    return CheckQuantParamsForMXTypeMImpl(xScaleShape, wScaleShape, false, expectedNSize);
}

bool GroupedQmmTiling::CheckQuantParamsForMXTypeMImpl(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape,
                                                      bool isScaleNzStorage, uint64_t expectedNSize) const
{
    auto xScaleDimNum = xScaleShape.GetDimNum();
    auto wScaleDimNum = wScaleShape.GetDimNum();
    const size_t expectedWScaleDimNum =
        IsWeightNzMultiTensorLayout() ? (MXFP_TYPE_M_SCALE_DIM_NUM - 1) : MXFP_TYPE_M_SCALE_DIM_NUM;
    OP_CHECK_IF(
        wScaleDimNum != expectedWScaleDimNum,
        OP_LOGE(inputParams_.opName, "When split m, the dim num of scale should be %zu in mx quant mode, but actual \
is %zu",
                expectedWScaleDimNum, wScaleDimNum),
        return false);
    OP_CHECK_IF(
        xScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
        OP_LOGE(inputParams_.opName, "When split m, the dim num of pertokenScale should be 3 in mx quant mode, but \
actual is %zu",
                xScaleDimNum),
        return false);
    const bool isMultiTensorWeight = IsWeightNzMultiTensorLayout();
    auto expectedKDimValue = CeilDiv(inputParams_.kSize, MXFP_BASEK_FACTOR);
    const uint64_t logicalNSize = expectedNSize == 0UL ? inputParams_.nSize : expectedNSize;
    auto wScaleLastDim = static_cast<uint64_t>(wScaleShape.GetDim(wScaleDimNum - 1));
    if (isMultiTensorWeight) {
        auto wScaleFirstDim = static_cast<uint64_t>(wScaleShape.GetDim(0));
        auto wScaleSecondDim = static_cast<uint64_t>(wScaleShape.GetDim(1));
        const bool isNkLayout = wScaleFirstDim == logicalNSize && wScaleSecondDim == expectedKDimValue;
        const bool isKnLayout = wScaleFirstDim == expectedKDimValue && wScaleSecondDim == logicalNSize;
        const bool isBroadcastLayout = wScaleFirstDim == 1UL || wScaleSecondDim == 1UL;
        OP_CHECK_IF((!isNkLayout && !isKnLayout && !isBroadcastLayout) || wScaleLastDim != MXFP_MULTI_BASE_SIZE,
                    OP_LOGE_FOR_INVALID_SHAPE(
                        inputParams_.opType, "scale", ShapeToString(wScaleShape),
                        ShapeDimsToString(logicalNSize, expectedKDimValue, MXFP_MULTI_BASE_SIZE) + " or " +
                            ShapeDimsToString(expectedKDimValue, logicalNSize, MXFP_MULTI_BASE_SIZE)),
                    return false);

        auto xScaleMDim = static_cast<uint64_t>(inputParams_.transA ? xScaleShape.GetDim(1) : xScaleShape.GetDim(0));
        auto xScaleKDim = static_cast<uint64_t>(inputParams_.transA ? xScaleShape.GetDim(0) : xScaleShape.GetDim(1));
        auto xScaleLastDim = static_cast<uint64_t>(xScaleShape.GetDim(xScaleDimNum - 1));
        OP_CHECK_IF(
            xScaleMDim != inputParams_.mSize || xScaleKDim != expectedKDimValue ||
                xScaleLastDim != MXFP_MULTI_BASE_SIZE,
            OP_LOGE_FOR_INVALID_SHAPE(
                inputParams_.opType, "perTokenScale", ShapeToString(xScaleShape),
                inputParams_.transA ? ShapeDimsToString(expectedKDimValue, inputParams_.mSize, MXFP_MULTI_BASE_SIZE) :
                                      ShapeDimsToString(inputParams_.mSize, expectedKDimValue, MXFP_MULTI_BASE_SIZE)),
            return false);
        return true;
    }

    auto wScaleEDim = static_cast<uint64_t>(wScaleShape.GetDim(0));
    auto wScaleNDim =
        static_cast<uint64_t>(isScaleNzStorage ? wScaleShape.GetDim(2) :
                                                 (inputParams_.transB ? wScaleShape.GetDim(1) : wScaleShape.GetDim(2)));
    auto wScaleKDim =
        static_cast<uint64_t>(isScaleNzStorage ? wScaleShape.GetDim(1) :
                                                 (inputParams_.transB ? wScaleShape.GetDim(2) : wScaleShape.GetDim(1)));
    auto xScaleMDim = static_cast<uint64_t>(inputParams_.transA ? xScaleShape.GetDim(1) : xScaleShape.GetDim(0));
    auto xScaleKDim = static_cast<uint64_t>(inputParams_.transA ? xScaleShape.GetDim(0) : xScaleShape.GetDim(1));
    auto xScaleLastDim = static_cast<uint64_t>(xScaleShape.GetDim(xScaleDimNum - 1));
    if (wScaleKDim != 1 && wScaleNDim != 1) {
        OP_CHECK_IF(
            wScaleEDim != inputParams_.groupNum || wScaleKDim != expectedKDimValue || wScaleNDim != logicalNSize ||
                wScaleLastDim != MXFP_MULTI_BASE_SIZE,
            OP_LOGE_FOR_INVALID_SHAPE(
                inputParams_.opType, "scale", ShapeToString(wScaleShape),
                (inputParams_.transB && !isScaleNzStorage) ?
                    ShapeDimsToString(inputParams_.groupNum, logicalNSize, expectedKDimValue, MXFP_MULTI_BASE_SIZE) :
                    ShapeDimsToString(inputParams_.groupNum, expectedKDimValue, logicalNSize, MXFP_MULTI_BASE_SIZE)),
            return false);
    }
    OP_CHECK_IF(
        xScaleMDim != inputParams_.mSize || xScaleKDim != expectedKDimValue || xScaleLastDim != MXFP_MULTI_BASE_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "perTokenScale", ShapeToString(xScaleShape),
                                  inputParams_.transA ?
                                      ShapeDimsToString(expectedKDimValue, inputParams_.mSize, MXFP_MULTI_BASE_SIZE) :
                                      ShapeDimsToString(inputParams_.mSize, expectedKDimValue, MXFP_MULTI_BASE_SIZE)),
        return false);
    return true;
}

bool GroupedQmmTiling::CheckQuantParamsForMXTypeK(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape) const
{
    auto xScaleDimNum = xScaleShape.GetDimNum();
    auto wScaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(
        wScaleDimNum != MXFP_TYPE_K_SCALE_DIM_NUM,
        OP_LOGE(inputParams_.opName, "When split k, the dim num of scale should be 3 in mx quant mode, but actual \
is %zu",
                wScaleDimNum),
        return false);
    OP_CHECK_IF(
        xScaleDimNum != MXFP_PER_TOKEN_SCALE_DIM_NUM,
        OP_LOGE(inputParams_.opName, "When split k, the dim num of pertokenScale should be 3 in mx quant mode, but \
actual is %zu",
                xScaleDimNum),
        return false);
    auto xScaleLastDim = static_cast<uint64_t>(xScaleShape.GetDim(xScaleDimNum - 1));
    auto xScaleKDim = static_cast<uint64_t>(
        inputParams_.transA ? xScaleShape.GetDim(0) : xScaleShape.GetDim(xScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto xScaleMDim = static_cast<uint64_t>(
        inputParams_.transA ? xScaleShape.GetDim(xScaleDimNum - LAST_SECOND_DIM_INDEX) : xScaleShape.GetDim(0));
    auto wScaleLastDim = static_cast<uint64_t>(wScaleShape.GetDim(wScaleDimNum - 1));
    auto wScaleNDim = static_cast<uint64_t>(
        inputParams_.transB ? wScaleShape.GetDim(0) : wScaleShape.GetDim(wScaleDimNum - LAST_SECOND_DIM_INDEX));
    auto wScaleKDim = static_cast<uint64_t>(
        inputParams_.transB ? wScaleShape.GetDim(wScaleDimNum - LAST_SECOND_DIM_INDEX) : wScaleShape.GetDim(0));
    auto expectedKDimValue = inputParams_.kSize / MXFP_BASEK_FACTOR + inputParams_.groupNum;
    if (xScaleKDim != 1 && xScaleMDim != 1) {
        OP_CHECK_IF(
            !inputParams_.transA || inputParams_.transB,
            OP_LOGE(inputParams_.opName, "When split k in mx quant mode, the expected transpose attrs of x and \
    weight are true and false, but the actual transpose attrs of x and weight are %d and %d.",
                    inputParams_.transA, inputParams_.transB),
            return false);
        OP_CHECK_IF(
            xScaleLastDim != MXFP_MULTI_BASE_SIZE || xScaleKDim != expectedKDimValue ||
                xScaleMDim != inputParams_.mSize,
            OP_LOGE(inputParams_.opName, "When split k in mx quant mode, the expected shape of pertokenScale is \
(%lu,%lu,%lu), but the actual is (%lu,%lu,%lu).",
                    expectedKDimValue, inputParams_.mSize, MXFP_MULTI_BASE_SIZE, xScaleKDim, xScaleMDim, xScaleLastDim),
            return false);
    }
    OP_CHECK_IF(
        wScaleLastDim != MXFP_MULTI_BASE_SIZE || wScaleKDim != expectedKDimValue || wScaleNDim != inputParams_.nSize,
        OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "scale", ShapeToString(wScaleShape),
                                  inputParams_.transB ?
                                      ShapeDimsToString(inputParams_.nSize, expectedKDimValue, MXFP_MULTI_BASE_SIZE) :
                                      ShapeDimsToString(expectedKDimValue, inputParams_.nSize, MXFP_MULTI_BASE_SIZE)),
        return false);
    return true;
}

bool GroupedQmmTiling::CheckQuantParamsForMxQuantMode(const gert::StorageShape *xScaleStorageShape,
                                                      const gert::StorageShape *wScaleStorageShape,
                                                      uint64_t expectedNSize) const
{
    // 多数参数在CheckQuantParamsForMxQuantMode函数调用前已有非空校验
    OP_CHECK_IF(xScaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "perTokenScale", "nullptr", "nonnull"), return false);
    OP_CHECK_IF(wScaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "scale", "nullptr", "nonnull"), return false);
    auto &xScaleShape = xScaleStorageShape->GetStorageShape();
    const gert::Shape &wScaleShape = wScaleStorageShape->GetOriginShape();
    if (inputParams_.groupType == SPLIT_M) {
        OP_CHECK_IF(!CheckQuantParamsForMXTypeM(xScaleShape, wScaleStorageShape, expectedNSize),
                    OP_LOGE(inputParams_.opName, "CheckQuantParamsForMXTypeM failed."), return false);
    } else {
        OP_CHECK_IF(!CheckQuantParamsForMXTypeK(xScaleShape, wScaleShape),
                    OP_LOGE(inputParams_.opName, "CheckQuantParamsForMXTypeK failed."), return false);
    }
    return true;
}

bool GroupedQmmTiling::CheckQuantParamsForNonKGroupQuantMode(const gert::Shape &wScaleShape,
                                                             uint64_t expectedNSize) const
{
    auto wScaleDimNum = wScaleShape.GetDimNum();
    const uint64_t logicalNSize = expectedNSize == 0UL ? inputParams_.nSize : expectedNSize;
    // dim num 1 for the shape (g,), dim num 2 for the shape (g,1) or (g,n)
    OP_CHECK_IF(wScaleDimNum != 1 && wScaleDimNum != 2,
                OP_LOGE(inputParams_.opName, "In non k axis group quant mode, the dim num of scale \
should be 1 or 2, but the actual dim num is %zu.",
                        wScaleDimNum),
                return false);
    if (inputParams_.cDtype == ge::DT_INT8) {
        OP_CHECK_IF(wScaleDimNum == 1, OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "scale", "1", "not 1"),
                    return false);
        OP_CHECK_IF(wScaleShape.GetDim(0) != inputParams_.groupNum || wScaleShape.GetDim(1) != logicalNSize,
                    OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "scale", ShapeToString(wScaleShape),
                                              ShapeDimsToString(inputParams_.groupNum, logicalNSize)),
                    return false);
    }
    return true;
}

bool GroupedQmmTiling::CheckFp4Shape(const gert::Shape &xShape, const gert::Shape &wShape) const
{
    OP_CHECK_IF(inputParams_.kSize % EVEN_FACTOR != 0 || inputParams_.kSize <= EVEN_FACTOR,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opType, "x, weight",
                                                       ShapesToString({ShapeToString(xShape), ShapeToString(wShape)}),
                                                       "when the dtype of x is FLOAT4, k size must be even and greater "
                                                       "than 2"),
                return false);
    if (!inputParams_.transB) {
        OP_CHECK_IF(inputParams_.nSize % EVEN_FACTOR != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opType, "weight", ShapeToString(wShape),
                        "when the dtype of x is FLOAT4 and weight is not transposed, n size must be even number"),
                    return false);
    }
    return true;
}

bool GroupedQmmTiling::CheckBiasShape(const gert::StorageShape *biasStorageShape) const
{
    auto &biasShape = biasStorageShape->GetStorageShape();
    OP_CHECK_IF(biasStorageShape->GetStorageShape().GetDimNum() != BIAS_DIMS,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "bias",
                                             std::to_string(biasStorageShape->GetStorageShape().GetDimNum()), "2"),
                return false);
    auto biasEDim = static_cast<uint64_t>(biasShape.GetDim(0));
    auto biasNDim = static_cast<uint64_t>(biasShape.GetDim(1));
    OP_CHECK_IF(biasEDim != inputParams_.groupNum || biasNDim != inputParams_.nSize,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "bias", ShapeToString(biasShape),
                                          ShapeDimsToString(inputParams_.groupNum, inputParams_.nSize)),
                return false);
    return true;
}

bool GroupedQmmTiling::CheckQuantParams(const gert::StorageShape *xScaleStorageShape,
                                        const gert::StorageShape *wScaleStorageShape, uint64_t expectedNSize) const
{
    OP_CHECK_IF(wScaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "scale", "nullptr", "nonnull"), return false);
    const gert::Shape &wScaleShape = wScaleStorageShape->GetOriginShape();
    // 非k分组量化校验
    if (inputParams_.bQuantMode != optiling::QuantMode::MX_PERGROUP_MODE &&
        inputParams_.bQuantMode != optiling::QuantMode::PERGROUP_MODE &&
        inputParams_.bQuantMode != optiling::QuantMode::PERBLOCK_MODE) {
        OP_CHECK_IF(!CheckQuantParamsForNonKGroupQuantMode(wScaleShape, expectedNSize),
                    OP_LOGE(inputParams_.opName, "CheckQuantParamsForNonKGroupQuantMode failed."), return false);
    }
    // mx量化校验
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        OP_CHECK_IF(!CheckQuantParamsForMxQuantMode(xScaleStorageShape, wScaleStorageShape, expectedNSize),
                    OP_LOGE(inputParams_.opName, "CheckQuantParamsForMxQuantMode failed."), return false);
    }

    return true;
}

bool GroupedQmmTiling::CheckScaleNzShape(const gert::StorageShape *wScaleStorageShape, uint64_t expectedNSize) const
{
    OP_CHECK_IF(!IsMicroScaling(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opType, "scale",
                                                      ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype),
                                                      "scale nz only supports FLOAT8_E8M0 in mx quant mode"),
                return false);
    const bool isFp4Dtype = inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2 ||
                            inputParams_.bDtype == ge::DT_FLOAT4_E2M1 || inputParams_.bDtype == ge::DT_FLOAT4_E1M2;
    OP_CHECK_IF(
        isFp4Dtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(inputParams_.opType, "x, weight",
                                               ListToString(ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype),
                                                            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype)),
                                               "scale nz does not support FLOAT4_E2M1 or FLOAT4_E1M2 in this version"),
        return false);
    OP_CHECK_IF(inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(inputParams_.opType, "weight",
                                                        ge::TypeUtils::FormatToSerialString(inputParams_.bFormat),
                                                        "scale nz requires weight format to be FRACTAL_NZ"),
                return false);
    OP_CHECK_IF(inputParams_.transB,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "transpose_weight", std::to_string(inputParams_.transB),
                                          "false when scale is nz"),
                return false);
    OP_CHECK_IF(inputParams_.groupType != SPLIT_M,
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "groupType", std::to_string(inputParams_.groupType),
                                          "0 when scale is nz"),
                return false);

    const gert::Shape &originShape = wScaleStorageShape->GetOriginShape();
    const gert::Shape &storageShape = wScaleStorageShape->GetStorageShape();
    const bool isMultiTensorWeight = IsWeightNzMultiTensorLayout();
    const size_t expectedOriginDimNum =
        isMultiTensorWeight ? (MXFP_TYPE_M_SCALE_DIM_NUM - 1) : MXFP_TYPE_M_SCALE_DIM_NUM;
    const size_t expectedStorageDimNum = isMultiTensorWeight ? (MXFP_SCALE_NZ_DIM_NUM - 1) : MXFP_SCALE_NZ_DIM_NUM;
    OP_CHECK_IF(
        originShape.GetDimNum() != expectedOriginDimNum,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "scale origin", std::to_string(originShape.GetDimNum()),
                                     std::to_string(expectedOriginDimNum)),
        return false);
    OP_CHECK_IF(
        storageShape.GetDimNum() != expectedStorageDimNum,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "scale storage", std::to_string(storageShape.GetDimNum()),
                                     std::to_string(expectedStorageDimNum)),
        return false);

    const uint64_t logicalNSize = expectedNSize == 0UL ? inputParams_.nSize : expectedNSize;
    const uint64_t expectedKScale = CeilDiv(inputParams_.kSize, MXFP_BASEK_FACTOR);
    const uint64_t expectedN1 = CeilDiv(logicalNSize, CUBE_BLOCK);
    const size_t originOffset = isMultiTensorWeight ? 0UL : 1UL;
    const size_t storageOffset = isMultiTensorWeight ? 0UL : 1UL;

    if (!isMultiTensorWeight) {
        OP_CHECK_IF(static_cast<uint64_t>(originShape.GetDim(0)) != inputParams_.groupNum,
                    OP_LOGE_FOR_INVALID_SHAPE(
                        inputParams_.opType, "scale origin", ShapeToString(originShape),
                        ShapeDimsToString(inputParams_.groupNum, expectedKScale, logicalNSize, MXFP_MULTI_BASE_SIZE)),
                    return false);
        OP_CHECK_IF(static_cast<uint64_t>(storageShape.GetDim(0)) != inputParams_.groupNum,
                    OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "scale storage", ShapeToString(storageShape),
                                              ShapeDimsToString(inputParams_.groupNum, expectedN1, expectedKScale,
                                                                CUBE_BLOCK, MXFP_MULTI_BASE_SIZE)),
                    return false);
    }

    OP_CHECK_IF(static_cast<uint64_t>(originShape.GetDim(originOffset)) != expectedKScale ||
                    static_cast<uint64_t>(originShape.GetDim(originOffset + 1UL)) != logicalNSize ||
                    static_cast<uint64_t>(originShape.GetDim(originOffset + MXFP_SCALE_ORIGIN_PAIR_DIM_OFFSET)) !=
                        MXFP_MULTI_BASE_SIZE,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "scale origin", ShapeToString(originShape),
                                          ShapeDimsToString(expectedKScale, logicalNSize, MXFP_MULTI_BASE_SIZE)),
                return false);
    OP_CHECK_IF(
        static_cast<uint64_t>(storageShape.GetDim(storageOffset)) != expectedN1 ||
            static_cast<uint64_t>(storageShape.GetDim(storageOffset + 1UL)) != expectedKScale ||
            static_cast<uint64_t>(storageShape.GetDim(storageOffset + MXFP_SCALE_NZ_N0_DIM_OFFSET)) != CUBE_BLOCK ||
            static_cast<uint64_t>(storageShape.GetDim(storageOffset + MXFP_SCALE_NZ_C0_DIM_OFFSET)) !=
                MXFP_MULTI_BASE_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "scale storage", ShapeToString(storageShape),
                                  ShapeDimsToString(expectedN1, expectedKScale, CUBE_BLOCK, MXFP_MULTI_BASE_SIZE)),
        return false);
    return true;
}

bool GroupedQmmTiling::CheckShapeForWeightNz(const gert::Shape &wShape, uint64_t expectedNSize) const
{
    OP_CHECK_IF(!CheckWeightNzDimNum(wShape), OP_LOGE(inputParams_.opName, "CheckWeightNzDimNum failed."),
                return false);
    const size_t wDimNum = wShape.GetDimNum();
    const uint64_t logicalNSize = expectedNSize == 0UL ? inputParams_.nSize : expectedNSize;
    const bool isInt4 = inputParams_.bDtype == ge::DT_INT4;
    const bool isFp4 = inputParams_.bDtype == ge::DT_FLOAT4_E2M1 || inputParams_.bDtype == ge::DT_FLOAT4_E1M2;
    const bool isBit4 = isInt4 || isFp4;
    const uint32_t weightNzLastDim = isBit4 ? WEIGHTNZ_64 : WEIGHTNZ_32;
    const size_t nzDimOffset = wDimNum - (WEIGHTNZ_DIM_NUM - 1);
    const char *nzDtypeLabel = isInt4 ? "int4" : (isFp4 ? "fp4" : "fp8");
    OP_CHECK_IF(!CheckWeightNzLastDim(wShape, nzDimOffset, weightNzLastDim, nzDtypeLabel),
                OP_LOGE(inputParams_.opName, "CheckWeightNzLastDim failed."), return false);
    OP_CHECK_IF(!CheckWeightNzSecondLastDim(wShape, nzDimOffset),
                OP_LOGE(inputParams_.opName, "CheckWeightNzSecondLastDim failed."), return false);
    auto wShapeDimThird = static_cast<uint64_t>(wShape[nzDimOffset + WEIGHTNZ_SECOND_DIM]);
    auto wShapeDimSecond = static_cast<uint64_t>(wShape[nzDimOffset]);
    OP_CHECK_IF(!CheckWeightNzTransposedDims(wShape, wShapeDimThird, wShapeDimSecond, weightNzLastDim, logicalNSize),
                OP_LOGE(inputParams_.opName, "CheckWeightNzTransposedDims failed."), return false);
    // 逻辑上最后两根轴对应 N/K：Ascend950 MXFP4 weight NZ 文档要求 n、k 不能为 1；此处对所有 NZ weight 统一约束
    OP_CHECK_IF(1 == inputParams_.kSize || 1 == logicalNSize,
                OP_LOGE(context_->GetNodeName(),
                        "For an NZ weight, logical N and K must both be greater than 1; actual N=%lu, K=%lu.",
                        logicalNSize, inputParams_.kSize),
                return false);
    return true;
}

bool GroupedQmmTiling::CheckWeightNzDimNum(const gert::Shape &wShape) const
{
    const size_t dimNum = wShape.GetDimNum();
    const bool valid = dimNum == WEIGHTNZ_DIM_NUM || (IsWeightNzMultiTensorLayout() && dimNum == WEIGHTNZ_DIM_NUM - 1);
    const std::string expected = IsWeightNzMultiTensorLayout() ?
                                     std::to_string(WEIGHTNZ_DIM_NUM - 1) + " or " + std::to_string(WEIGHTNZ_DIM_NUM) :
                                     std::to_string(WEIGHTNZ_DIM_NUM);
    OP_CHECK_IF(!valid, OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "weight", std::to_string(dimNum), expected),
                return false);
    return true;
}

bool GroupedQmmTiling::CheckWeightNzLastDim(const gert::Shape &wShape, size_t nzDimOffset, uint32_t expectedValue,
                                            const char *dtypeLabel) const
{
    const uint64_t actual = static_cast<uint64_t>(wShape[nzDimOffset + WEIGHTNZ_FORTH_DIM]);
    OP_CHECK_IF(actual != static_cast<uint64_t>(expectedValue),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    inputParams_.opType, "weight", ShapeToString(wShape),
                    BuildErrorMsgStr("the last storage dimension of an NZ weight must be ", expectedValue, " for ",
                                     dtypeLabel, "; actual value is ", actual)),
                return false);
    return true;
}

bool GroupedQmmTiling::CheckWeightNzSecondLastDim(const gert::Shape &wShape, size_t nzDimOffset) const
{
    const int64_t actual = wShape[nzDimOffset + WEIGHTNZ_THIRD_DIM];
    OP_CHECK_IF(actual != static_cast<int64_t>(WEIGHTNZ_16),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    inputParams_.opType, "weight", ShapeToString(wShape),
                    BuildErrorMsgStr("the second-last storage dimension of an NZ weight must be ", WEIGHTNZ_16,
                                     "; actual value is ", actual)),
                return false);
    return true;
}

bool GroupedQmmTiling::CheckWeightNzTransposedDims(const gert::Shape &wShape, uint64_t wShapeDimThird,
                                                   uint64_t wShapeDimSecond, uint32_t weightNzLastDim,
                                                   uint64_t logicalNSize) const
{
    if (!inputParams_.transB) {
        OP_CHECK_IF(wShapeDimThird != CeilDiv(inputParams_.kSize, WEIGHTNZ_16),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opType, "weight", ShapeToString(wShape),
                        BuildErrorMsgStr("when the format of weight is FRACTAL_NZ, third dimension must be equal to "
                                         "ceil(kSize/16) = ",
                                         CeilDiv(inputParams_.kSize, WEIGHTNZ_16))),
                    return false);
        OP_CHECK_IF(wShapeDimSecond != CeilDiv(logicalNSize, weightNzLastDim),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opType, "weight", ShapeToString(wShape),
                        BuildErrorMsgStr(
                            "when the format of weight is FRACTAL_NZ, second dimension must be equal to ceil(nSize/",
                            weightNzLastDim, ") = ", CeilDiv(logicalNSize, weightNzLastDim))),
                    return false);
    } else {
        OP_CHECK_IF(wShapeDimThird != CeilDiv(logicalNSize, WEIGHTNZ_16),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opType, "weight", ShapeToString(wShape),
                        BuildErrorMsgStr("when the format of weight is FRACTAL_NZ, third dimension must be equal to "
                                         "ceil(nSize/16) = ",
                                         CeilDiv(logicalNSize, WEIGHTNZ_16))),
                    return false);
        OP_CHECK_IF(wShapeDimSecond != CeilDiv(inputParams_.kSize, weightNzLastDim),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opType, "weight", ShapeToString(wShape),
                        BuildErrorMsgStr(
                            "when the format of weight is FRACTAL_NZ, second dimension must be equal to ceil(kSize/",
                            weightNzLastDim, ") = ", CeilDiv(inputParams_.kSize, weightNzLastDim))),
                    return false);
    }
    return true;
}

bool GroupedQmmTiling::CheckMultiWeightNzInputs(const gert::StorageShape *xScaleStorageShape) const
{
    if (!IsWeightNzMultiTensorLayout()) {
        return true;
    }

    const uint16_t weightTensorNum = GetTensorListSize(WEIGHT_INDEX);
    const auto *firstWeightShape = context_->GetDynamicInputShape(WEIGHT_INDEX, FIRST_TENSOR_INDEX);
    OP_CHECK_IF(firstWeightShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "In WeightNz single-multi-single mode, current weight[0] is "
                                                 "nullptr; expected weight[0] to be non-null."),
                return false);
    const uint64_t firstNSize = GetWeightLogicalNSize(firstWeightShape->GetOriginShape());
    for (uint16_t i = SECOND_TENSOR_INDEX; i < weightTensorNum; ++i) {
        auto curWStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, i);
        auto curScaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, i);
        OP_CHECK_IF(curWStorageShape == nullptr || curScaleStorageShape == nullptr,
                    OP_LOGE(context_->GetNodeName(),
                            "In WeightNz single-multi-single mode at index %u, current weight is %s and scale is %s; "
                            "expected both tensors to be non-null.",
                            i, curWStorageShape == nullptr ? "nullptr" : "non-null",
                            curScaleStorageShape == nullptr ? "nullptr" : "non-null"),
                    return false);
        const gert::Shape &curWShape = curWStorageShape->GetOriginShape();
        const uint64_t curNSize = GetWeightLogicalNSize(curWShape);
        OP_CHECK_IF(curNSize != firstNSize,
                    OP_LOGE(inputParams_.opName,
                            "In WeightNz single-multi-single mode, current weight[%u] logical N is %lu; "
                            "expected logical N %lu from weight[0].",
                            i, curNSize, firstNSize),
                    return false);
        OP_CHECK_IF(!CheckQuantParams(xScaleStorageShape, curScaleStorageShape, curNSize),
                    OP_LOGE(inputParams_.opName, "CheckQuantParams failed for weight index %u.", i), return false);
        OP_CHECK_IF(!CheckShapeForWeightNz(curWStorageShape->GetStorageShape(), curNSize),
                    OP_LOGE(context_->GetNodeName(), "CheckShapeForWeightNz failed for index %u.", i), return false);
    }
    return true;
}

bool GroupedQmmTiling::CheckActiveModeDtype(const gert::StorageShape *xScaleStorageShape) const
{
    OP_CHECK_IF((inputParams_.aDtype != ge::DT_INT8) || (inputParams_.bDtype != ge::DT_INT8),
                OP_LOGE(context_->GetNodeName(),
                        "When the activation function is enabled, the dtype of x and weight should be DT_INT8, "
                        "actual is %s and %s.",
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str()),
                return false);
    if (inputParams_.cDtype == ge::DT_BF16) {
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_BF16 && inputParams_.scaleDtype != ge::DT_FLOAT,
                    OP_LOGE(inputParams_.opName,
                            "When the activation function is enabled and the dtype of output is DT_BF16, the dtype of "
                            "scale should be DT_BF16 or DT_FLOAT, actual is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                    return false);
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT && xScaleStorageShape != nullptr,
                    OP_LOGE(inputParams_.opName,
                            "When the activation function is enabled and the dtype of output is DT_BF16, the dtype of "
                            "perTokenScale should be DT_FLOAT, actual is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                    return false);
    } else if (inputParams_.cDtype == ge::DT_FLOAT16) {
        OP_CHECK_IF(inputParams_.scaleDtype != ge::DT_FLOAT,
                    OP_LOGE(inputParams_.opName,
                            "When the activation function is enabled and the dtype of output is DT_BF16, the dtype of "
                            "scale should be DT_FLOAT, actual is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str()),
                    return false);
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT && xScaleStorageShape != nullptr,
                    OP_LOGE(inputParams_.opName,
                            "When the activation function is enabled and the dtype of output is DT_BF16, the dtype of "
                            "perTokenScale should be DT_FLOAT, actual is %s.",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str()),
                    return false);
    } else {
        OP_LOGE(inputParams_.opName,
                "When the activation function is enabled, the dtype of output should be DT_BF16 or DT_FLOAT16, "
                "actual is %s.",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str());
        return false;
    }
    return true;
}

bool GroupedQmmTiling::CheckActiveMode(const gert::Shape &wScaleShape, const gert::StorageShape *xScaleStorageShape)
{
    OP_CHECK_IF(
        inputParams_.actType == static_cast<int8_t>(GMMActType::GMM_ACT_TYPE_GELU_ERR_FUNC),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "actType", std::to_string(inputParams_.actType),
                                              "activation function does not support GELU_ERR_FUNC now"),
        return false);
    OP_CHECK_IF(inputParams_.actType > static_cast<int8_t>(GMMActType::GMM_ACT_TYPE_SILU) ||
                    inputParams_.actType < static_cast<int8_t>(GMMActType::GMM_ACT_TYPE_NONE),
                OP_LOGE_FOR_INVALID_VALUE(inputParams_.opType, "actType", std::to_string(inputParams_.actType),
                                          "RELU/GELU_TANH/FASTGELU/SILU"),
                return false);
    OP_CHECK_IF(!CheckActiveModeDtype(xScaleStorageShape),
                OP_LOGE(context_->GetNodeName(), "CheckActiveModeDtype failed."), return false);
    auto wScaleDims = wScaleShape.GetDimNum();
    if (xScaleStorageShape != nullptr) {
        auto &xScaleShape = xScaleStorageShape->GetStorageShape();
        auto xScaleDims = xScaleShape.GetDimNum();
        OP_CHECK_IF(xScaleDims != 1,
                    OP_LOGE(context_->GetNodeName(), // 在启用激活函数情景下，perTorkenScale应该为1维
                            "When the activation function is enabled, the dim of perTokenScale should be 1 or nullptr, "
                            "actual is %d.",
                            xScaleDims),
                    return false);
        OP_CHECK_IF(static_cast<uint64_t>(xScaleShape[0]) != inputParams_.mSize,
                    OP_LOGE(context_->GetNodeName(),
                            "When the activation function is enabled and the dim of perTokenScale is 1, "
                            "the shape of perTokenScale should be (%d,), "
                            "actual is (%d,).",
                            inputParams_.mSize, static_cast<uint64_t>(xScaleShape[0])),
                    return false);
    }
    OP_CHECK_IF(wScaleDims != 2,
                OP_LOGE(context_->GetNodeName(), // 在启用激活函数情景下，Scale应该为2维
                        "When the activation function is enabled, the dim of Scale should be 2, "
                        "actual is %d.",
                        wScaleDims),
                return false);
    OP_CHECK_IF(static_cast<uint64_t>(wScaleShape[0]) != inputParams_.groupNum ||
                    static_cast<uint64_t>(wScaleShape[1]) != inputParams_.nSize,
                OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opType, "scale", ShapeToString(wScaleShape),
                                          ShapeDimsToString(inputParams_.groupNum, inputParams_.nSize)),
                return false);
    if (inputParams_.nSize == 1) {
        inputParams_.bQuantMode = optiling::QuantMode::PERCHANNEL_MODE;
    }
    return true;
}

bool GroupedQmmTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetDynamicInputShape(X_INDEX, 0);

    OP_CHECK_IF(xStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "xStorageShape is nullptr."), return false);
    const gert::Shape &xShape = xStorageShape->GetOriginShape();

    auto wStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    OP_CHECK_IF(wStorageShape == nullptr, OP_LOGE(context_->GetNodeName(), "wStorageShape is nullptr."), return false);
    const gert::Shape &wShape = wStorageShape->GetOriginShape();
    const gert::Shape &weightNzStorageShape = wStorageShape->GetStorageShape();

    OP_CHECK_IF(!SetGroupNum(GROUPLIST_INDEX), OP_LOGE(inputParams_.opName, "SetGroupNum failed."), return false);
    OP_CHECK_IF(!CheckWeightTensorListForWeightNz(),
                OP_LOGE(inputParams_.opName, "CheckWeightTensorListForWeightNz failed."), return false);
    OP_CHECK_IF(!SetMKN(xShape, wShape), OP_LOGE(inputParams_.opName, "SetMKN failed."), return false);

    if (inputParams_.cDtype == ge::DT_INT32) {
        return true;
    }

    auto scaleStorageShape = context_->GetDynamicInputShape(SCALE_INDEX, 0);
    OP_CHECK_IF(scaleStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "scale", "nullptr",
                                                      "scaleStorageShape cannot be nullptr when cDtype is not INT32"),
                return false);
    const gert::Shape &wScaleShape = scaleStorageShape->GetOriginShape();
    auto scaleDimNum = wScaleShape.GetDimNum();
    OP_CHECK_IF(
        scaleDimNum < 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "scale", std::to_string(scaleDimNum), "positive integer"),
        return false);
    auto xScaleStorageShape = context_->GetOptionalInputShape(PER_TOKEN_SCALE_INDEX);
    OP_CHECK_IF(!SetQuantMode(wScaleShape, xScaleStorageShape, wShape),
                OP_LOGE(inputParams_.opName, "SetQuantMode failed."), return false);
    OP_CHECK_IF(!CheckQuantParams(xScaleStorageShape, scaleStorageShape),
                OP_LOGE(inputParams_.opName, "CheckQuantParams failed."), return false);

    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_CHECK_IF(!CheckShapeForWeightNz(weightNzStorageShape),
                    OP_LOGE(context_->GetNodeName(), "CheckShapeForWeightNz failed."), return false);
    }
    OP_CHECK_IF(!CheckMultiWeightNzInputs(xScaleStorageShape),
                OP_LOGE(inputParams_.opName, "CheckMultiWeightNzInputs failed."), return false);
    if (inputParams_.actType != static_cast<int8_t>(GMMActType::GMM_ACT_TYPE_NONE)) {
        OP_CHECK_IF(!CheckActiveMode(wScaleShape, xScaleStorageShape),
                    OP_LOGE(context_->GetNodeName(), "CheckActiveMode failed."), return false);
    }
    if (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2) {
        OP_CHECK_IF(!CheckFp4Shape(xShape, wShape), OP_LOGE(inputParams_.opName, "CheckFp4Shape failed."),
                    return false);
        if (inputParams_.hasBias) {
            auto biasStorageShape = context_->GetDynamicInputShape(BIAS_INDEX, 0);
            OP_CHECK_IF(!CheckBiasShape(biasStorageShape), OP_LOGE(inputParams_.opName, "CheckBiasShape failed."),
                        return false);
        }
    }
    SetKernelType();
    OP_CHECK_IF(!CheckCoreNum(), OP_LOGE(inputParams_.opName, "CheckCoreNum failed."), return false);
    return true;
}

bool GroupedQmmTiling::CheckCoreNum() const
{
    auto compileInfo = context_->GetCompileInfo<GMMCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(inputParams_.opName, "compileInfo is nullptr."), return false);
    auto aicNum = compileInfo->aicNum;
    auto aivNum = compileInfo->aivNum;
    if (inputParams_.groupType == SPLIT_K) {
        OP_CHECK_IF(
            aivNum != GmmConstant::CORE_RATIO * aicNum,
            OP_LOGE(inputParams_.opName,
                    "When group type is 2 (mix template), aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.",
                    aicNum, aivNum),
            return false);
    }
    // kernelType, 1：dequant vec模板；2：perGroup-perBlock(GB)独立pertile模板
    if (inputParams_.kernelType == 1 || inputParams_.kernelType == 2) {
        OP_CHECK_IF(
            aivNum != GmmConstant::CORE_RATIO * aicNum,
            OP_LOGE(inputParams_.opName, "Current scene, aicNum:aivNum should be 1:2, actual aicNum: %u, aivNum: %u.",
                    aicNum, aivNum),
            return false);
    }
    return true;
}

bool GroupedQmmTiling::SetQuantMode(const gert::Shape &wScaleShape, const gert::StorageShape *xScaleStorageShape,
                                    const gert::Shape &wShape)
{
    auto wScaleDims = wScaleShape.GetDimNum();
    if (IsMicroScaling()) {
        inputParams_.bQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        inputParams_.aQuantMode = optiling::QuantMode::MX_PERGROUP_MODE;
        return true;
    }
    // scale pertensor: (g,1) 2维或（g,）1维, perchannel:（g, N), 2维
    if (wScaleDims == 2 && static_cast<uint64_t>(wScaleShape.GetDim(wScaleDims - 1)) == inputParams_.nSize &&
        inputParams_.nSize != 1UL) {
        inputParams_.bQuantMode = optiling::QuantMode::PERCHANNEL_MODE;
    } else if ((wScaleDims == 2 && wScaleShape[wScaleDims - 1] == 1) || // 2:（g,1) 2维
               (wScaleDims == 1 && static_cast<uint64_t>(wScaleShape[0]) == inputParams_.groupNum)) {
        inputParams_.bQuantMode = optiling::QuantMode::PERTENSOR_MODE;
    }
    if (xScaleStorageShape != nullptr) {
        // split_m: pertoken (M,), pertensor（g,1) 2维或（g,）1维;
        // split_k: pertoken (g, M), pertensor（g,1) 2维或（g,) 1维
        auto &xScaleShape = xScaleStorageShape->GetStorageShape();
        auto xScaleDims = xScaleShape.GetDimNum();
        if (inputParams_.aDtype != ge::DT_INT8 &&
            ((xScaleDims == 2 && xScaleShape[xScaleDims - 1] == 1) || // 2:（g,1) 2维
             (xScaleDims == 1 && static_cast<uint64_t>(xScaleShape[0]) == inputParams_.groupNum &&
              (inputParams_.groupType == SPLIT_K || inputParams_.groupNum != inputParams_.mSize)))) {
            inputParams_.aQuantMode = optiling::QuantMode::PERTENSOR_MODE;
        } else {
            inputParams_.aQuantMode = optiling::QuantMode::PERTOKEN_MODE;
        }
        SetPerGroupQuantMode(xScaleShape, wScaleShape, wShape);
    }
    return true;
}

void GroupedQmmTiling::SetPerGroupQuantMode(const gert::Shape &xScaleShape, const gert::Shape &wScaleShape,
                                            const gert::Shape &wShape)
{
    if (inputParams_.aDtype == ge::DT_INT8) {
        return;
    }
    auto xScaleDims = xScaleShape.GetDimNum();
    if (xScaleDims < X_DIMS) {
        return;
    }
    auto wScaleDims = wScaleShape.GetDimNum();
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    if (wDimNum != wScaleDims || (inputParams_.groupType == SPLIT_M && wScaleDims < SPLIT_M_W_DIMS) ||
        (inputParams_.groupType == SPLIT_K && wScaleDims < SPLIT_K_W_DIMS)) {
        return;
    }
    optiling::QuantMode aQuantMode = optiling::QuantMode::DEFAULT;
    optiling::QuantMode bQuantMode = optiling::QuantMode::DEFAULT;
    if (inputParams_.groupType == SPLIT_M) {
        for (uint64_t i = 1; i < wScaleDims; ++i) {
            if (wScaleShape.GetDim(i) != CeilDiv(wShape.GetDim(i), PER_BLOCK_GROUP_SIZE)) {
                return;
            }
        }
        bQuantMode = optiling::QuantMode::PERBLOCK_MODE;

        uint64_t scaleKPerBlock = CeilDiv(inputParams_.kSize, PER_BLOCK_GROUP_SIZE);
        if (static_cast<uint64_t>(xScaleShape.GetDim(xScaleDims - LAST_FIRST_DIM_INDEX)) == scaleKPerBlock &&
            static_cast<uint64_t>(xScaleShape.GetDim(xScaleDims - LAST_SECOND_DIM_INDEX)) == inputParams_.mSize) {
            aQuantMode = optiling::QuantMode::PERGROUP_MODE;
        }
    }
    if (inputParams_.groupType == SPLIT_K) {
        uint64_t scaleKPerBlock = inputParams_.kSize / PER_BLOCK_GROUP_SIZE + inputParams_.groupNum;
        uint64_t scaleNPerBlock = CeilDiv(inputParams_.nSize, PER_BLOCK_GROUP_SIZE);
        if (static_cast<uint64_t>(xScaleShape.GetDim(xScaleDims - LAST_SECOND_DIM_INDEX)) == scaleKPerBlock &&
            static_cast<uint64_t>(xScaleShape.GetDim(xScaleDims - LAST_FIRST_DIM_INDEX)) == inputParams_.mSize) {
            aQuantMode = optiling::QuantMode::PERGROUP_MODE;
        }
        if (static_cast<uint64_t>(wScaleShape.GetDim(wScaleDims - LAST_SECOND_DIM_INDEX)) == scaleKPerBlock &&
            static_cast<uint64_t>(wScaleShape.GetDim(wScaleDims - LAST_FIRST_DIM_INDEX)) == scaleNPerBlock) {
            bQuantMode = optiling::QuantMode::PERBLOCK_MODE;
        }
    }
    if (aQuantMode == optiling::QuantMode::PERGROUP_MODE && bQuantMode == optiling::QuantMode::PERBLOCK_MODE) {
        inputParams_.aQuantMode = optiling::QuantMode::PERGROUP_MODE;
        inputParams_.bQuantMode = optiling::QuantMode::PERBLOCK_MODE;
    }
}

bool GroupedQmmTiling::SetGroupNum(uint32_t groupListIndex)
{
    auto groupListStorageShape = context_->GetOptionalInputShape(groupListIndex);
    OP_CHECK_IF(groupListStorageShape == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "groupList", "nullptr",
                                                      "groupListStorageShape cannot be nullptr"),
                return false);
    const gert::Shape &groupListShape = groupListStorageShape->GetStorageShape();
    size_t groupListDimNum = groupListShape.GetDimNum();
    // groupListType 0(cumsum)/1(count): grouplist is 1D; groupListType 2(sparse_m): grouplist is 2D [E, 2]
    if (inputParams_.groupListType == GROUPLIST_TYPE_SPARSE_M) {
        OP_CHECK_IF(
            groupListDimNum != GROUP_LIST_SPARSE_DIMS,
            OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "groupList", std::to_string(groupListDimNum), "2"),
            return false);
        OP_CHECK_IF(groupListShape.GetDim(1) != GROUP_LIST_SPARSE_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opType, "groupList", ShapeToString(groupListShape),
                        "2nd dimension of groupList must be equal to 2 when groupListType is 2(sparse_m)"),
                    return false);
        inputParams_.groupNum = groupListShape.GetDim(0);
    } else {
        OP_CHECK_IF(
            groupListDimNum != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "groupList", std::to_string(groupListDimNum), "1"),
            return false);
        inputParams_.groupNum = groupListShape.GetDim(0);
    }
    OP_CHECK_IF(
        inputParams_.groupNum > GMM_MAX_GROUP_LIST_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opType, "groupNum", std::to_string(inputParams_.groupNum),
                                              "group number cannot be greater than 1024"),
        return false);
    return true;
}

bool GroupedQmmTiling::SetMKN(const gert::Shape &xShape, const gert::Shape &wShape)
{
    uint32_t wDimNum = static_cast<uint32_t>(wShape.GetDimNum());
    OP_CHECK_IF(wDimNum < MIN_ND_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "weight", std::to_string(wDimNum), "at least 2"),
                return false);
    uint32_t xDimNum = static_cast<uint32_t>(xShape.GetDimNum());
    OP_CHECK_IF(xDimNum < MIN_ND_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opType, "x", std::to_string(xDimNum), "at least 2"),
                return false);
    auto mSize = inputParams_.transA ? xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX) :
                                       xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX);
    auto kSize = inputParams_.transA ? xShape.GetDim(xDimNum - LAST_SECOND_DIM_INDEX) :
                                       xShape.GetDim(xDimNum - LAST_FIRST_DIM_INDEX);
    auto nSize = inputParams_.transB ? wShape.GetDim(wDimNum - LAST_SECOND_DIM_INDEX) :
                                       wShape.GetDim(wDimNum - LAST_FIRST_DIM_INDEX);
    // In static graph mode, when M or N is 0, the framework will skip tiling check and return empty tensors, but when
    // K = 0, tiling should be able to intercept.
    OP_CHECK_IF((xShape.GetShapeSize() == 0 || wShape.GetShapeSize() == 0) && inputParams_.groupType == SPLIT_M,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams_.opType, "x, weight",
                                                       ShapesToString({ShapeToString(xShape), ShapeToString(wShape)}),
                                                       "unsupported empty tensor here, please check k dimension"),
                return false);
    inputParams_.mSize = mSize;
    inputParams_.kSize = kSize;
    inputParams_.nSize = nSize;
    return true;
}

bool GroupedQmmTiling::SetMKNList()
{
    if (inputParams_.groupType == SPLIT_M) {
        mList_[0] = -1;
        if (IsWeightNzMultiTensorLayout()) {
            uint16_t weightTensorNum = GetTensorListSize(WEIGHT_INDEX);
            for (uint16_t i = 0; i < weightTensorNum; ++i) {
                auto weightShape = context_->GetDynamicInputShape(WEIGHT_INDEX, i);
                OP_CHECK_IF(weightShape == nullptr,
                            OP_LOGE(context_->GetNodeName(),
                                    "In WeightNz single-multi-single mode, current weight[%u] is nullptr; "
                                    "expected a non-null weight tensor.",
                                    i),
                            return false);
                const gert::Shape &wShape = weightShape->GetOriginShape();
                kList_[i] = static_cast<int32_t>(inputParams_.kSize);
                nList_[i] = static_cast<int32_t>(GetWeightLogicalNSize(wShape));
            }
        } else {
            kList_[0] = static_cast<int32_t>(inputParams_.kSize);
            nList_[0] = static_cast<int32_t>(inputParams_.nSize);
        }
    } else {
        mList_[0] = static_cast<int32_t>(inputParams_.mSize);
        kList_[0] = -1;
        nList_[0] = static_cast<int32_t>(inputParams_.nSize);
    }
    return true;
}

ge::graphStatus GroupedQmmTiling::GetShapeAttrsInfo()
{
    inputParams_.opName = context_->GetNodeName();
    inputParams_.opType = GetOpType();
    if (inputParams_.initFlag) {
        OP_LOGD(inputParams_.opName, "No need to get shape and attrs from tiling context again.");
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(!AnalyzeDtype() || !AnalyzeAttrs() || !AnalyzeInputs(),
                OP_LOGE(inputParams_.opName, "Failed to analyze context_ info."), return ge::GRAPH_FAILED);
    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedQmmTiling::DoOpTiling()
{
    tilingData_.gmmQuantParams.groupNum = inputParams_.groupNum;
    tilingData_.gmmQuantParams.activeType = inputParams_.actType;
    tilingData_.gmmQuantParams.aQuantMode = static_cast<uint32_t>(inputParams_.aQuantMode);
    tilingData_.gmmQuantParams.bQuantMode = static_cast<uint32_t>(inputParams_.bQuantMode);
    tilingData_.gmmQuantParams.singleX = static_cast<uint8_t>(inputParams_.isSingleX);
    tilingData_.gmmQuantParams.singleW = static_cast<uint8_t>(inputParams_.isSingleW);
    tilingData_.gmmQuantParams.singleY = static_cast<uint8_t>(inputParams_.isSingleY);
    tilingData_.gmmQuantParams.groupType = static_cast<int8_t>(inputParams_.groupType);
    tilingData_.gmmQuantParams.groupListType = static_cast<uint8_t>(inputParams_.groupListType);
    tilingData_.gmmQuantParams.hasBias = static_cast<uint8_t>(inputParams_.hasBias);
    OP_CHECK_IF(!SetMKNList(), OP_LOGE(inputParams_.opName, "SetMKNList failed."), return ge::GRAPH_FAILED);
    errno_t retM = memcpy_s(tilingData_.gmmArray.mList, sizeof(tilingData_.gmmArray.mList), mList_, sizeof(mList_));
    if (retM != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", retM);
        return ge::GRAPH_FAILED;
    }
    errno_t retK = memcpy_s(tilingData_.gmmArray.kList, sizeof(tilingData_.gmmArray.kList), kList_, sizeof(kList_));
    if (retK != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", retK);
        return ge::GRAPH_FAILED;
    }
    errno_t retN = memcpy_s(tilingData_.gmmArray.nList, sizeof(tilingData_.gmmArray.nList), nList_, sizeof(nList_));
    if (retN != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", retN);
        return ge::GRAPH_FAILED;
    }
    PrintQuantParams();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedQmmTiling::DoLibApiTiling()
{
    CalBasicBlock();
    OP_CHECK_IF(CalL1Tiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed"),
                return ge::GRAPH_FAILED);
    tilingData_.mmTilingData.M = inputParams_.mSize;
    tilingData_.mmTilingData.N = inputParams_.nSize;
    tilingData_.mmTilingData.Ka = inputParams_.kSize;
    tilingData_.mmTilingData.Kb = inputParams_.kSize;
    tilingData_.mmTilingData.usedCoreNum = aicoreParams_.aicNum;
    tilingData_.mmTilingData.baseM = basicTiling_.baseM;
    tilingData_.mmTilingData.baseN = basicTiling_.baseN;
    tilingData_.mmTilingData.baseK = basicTiling_.baseK;
    tilingData_.mmTilingData.singleCoreM = basicTiling_.singleCoreM;
    tilingData_.mmTilingData.singleCoreN = basicTiling_.singleCoreN;
    tilingData_.mmTilingData.singleCoreK = basicTiling_.singleCoreK;
    tilingData_.mmTilingData.depthA1 = basicTiling_.depthA1;
    tilingData_.mmTilingData.depthB1 = basicTiling_.depthB1;
    tilingData_.mmTilingData.stepM = basicTiling_.stepM;
    tilingData_.mmTilingData.stepN = basicTiling_.stepN;
    tilingData_.mmTilingData.stepKa = basicTiling_.stepKa;
    tilingData_.mmTilingData.stepKb = basicTiling_.stepKb;
    tilingData_.mmTilingData.isBias = inputParams_.hasBias ? 1 : 0;
    tilingData_.mmTilingData.iterateOrder = basicTiling_.iterateOrder;
    tilingData_.mmTilingData.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_.mmTilingData.dbL0C = basicTiling_.dbL0c;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.mmTilingData.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.mmTilingData.mxTypePara =
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_M_BIT) +
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void GroupedQmmTiling::SetKernelType()
{
    // 以选择主模板设置kernelType, 0: dequant fixp随路（包含K轴分组）；1：dequant vector计算；2：perGroup-perBlock
    inputParams_.kernelType = 0UL;
    // mx K轴分组当前是独立的模板，后续归一
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        return;
    }
    // perGroup-perBlock(GB)有独立pertile模板
    if (inputParams_.bQuantMode == optiling::QuantMode::PERBLOCK_MODE) {
        inputParams_.kernelType = 2UL;
        return;
    }
    // pertensor-pertensor且没有后处理的bias，都可以走dequant fixp随路
    bool isPertensorCube = inputParams_.aQuantMode <= optiling::QuantMode::PERTENSOR_MODE &&
                           inputParams_.bQuantMode == optiling::QuantMode::PERTENSOR_MODE;
    bool isBiasEpilogue =
        inputParams_.aDtype == ge::DT_INT8 && inputParams_.hasBias && inputParams_.biasDtype != ge::DT_INT32;
    // 如果bias bf16/fp16/fp32，需mix模板进行后处理
    if (isPertensorCube && !isBiasEpilogue) {
        return;
    }
    bool isScaleEpilogue = (inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64);
    // 后处理的bias和（scale非64bits && ！isPertensorCube）需要走dequant vec模板
    if (isBiasEpilogue || isScaleEpilogue) {
        inputParams_.kernelType = 1UL;
    }
}

uint64_t GroupedQmmTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA),
                              static_cast<uint64_t>(inputParams_.kernelType));
}

ge::graphStatus GroupedQmmTiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedQmmTiling::PostTiling()
{
    return SaveTilingDataToContext(tilingData_);
}

int64_t GroupedQmmTiling::LogQuantParams(const GMMQuantParams &params) const
{
    std::ostringstream oss;
    oss << "GMMQuantParams: groupNum = " << params.groupNum << ", activeType = " << params.activeType
        << ", aQuantMode = " << params.aQuantMode << ", bQuantMode = " << params.bQuantMode
        << ", singleX=" << static_cast<int32_t>(params.singleX)
        << ", singleW = " << static_cast<int32_t>(params.singleW)
        << ", singleY = " << static_cast<int32_t>(params.singleY)
        << ", groupType = " << static_cast<int32_t>(params.groupType)
        << ", groupListType = " << static_cast<uint32_t>(params.groupListType)
        << ", hasBias = " << static_cast<int32_t>(params.hasBias);
    OP_LOGD(inputParams_.opName, "%s", oss.str().c_str());
    return 0;
}

void GroupedQmmTiling::PrintQuantParams()
{
    OP_LOGD(inputParams_.opName, "%ld", LogQuantParams(tilingData_.gmmQuantParams));
}

void GroupedQmmTiling::CalBasicBlock()
{
    bool isGBQuantMode = inputParams_.aQuantMode == optiling::QuantMode::PERGROUP_MODE &&
                         inputParams_.bQuantMode == optiling::QuantMode::PERBLOCK_MODE;
    basicTiling_.baseM = std::min(inputParams_.mSize, static_cast<uint64_t>(GmmConstant::BASIC_BLOCK_SIZE_256));
    basicTiling_.baseM = !inputParams_.transA ?
                             CeilAlign(basicTiling_.baseM, CUBE_BLOCK) :
                             CeilAlign(basicTiling_.baseM, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype));
    if (isGBQuantMode) {
        // 不管M/K轴分组，单单单场景下，N不变，可以确定baseN
        if (inputParams_.nSize <= PER_BLOCK_GROUP_SIZE || basicTiling_.baseM > PER_BLOCK_GROUP_SIZE) {
            basicTiling_.baseN = PER_BLOCK_GROUP_SIZE;
        } else {
            basicTiling_.baseN = GmmConstant::BASIC_BLOCK_SIZE_256;
        }
        basicTiling_.baseK = PER_BLOCK_GROUP_SIZE;
        return;
    }
    basicTiling_.baseN = std::min(inputParams_.nSize, static_cast<uint64_t>(GmmConstant::BASIC_BLOCK_SIZE_256));
    basicTiling_.baseN = inputParams_.transB ?
                             CeilAlign(basicTiling_.baseN, CUBE_BLOCK) :
                             CeilAlign(basicTiling_.baseN, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype));
    basicTiling_.baseK = CeilAlign(
        std::min(GetShapeWithDataType(GmmConstant::BASIC_BLOCK_SIZE_128, inputParams_.aDtype), inputParams_.kSize),
        GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype));

    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        basicTiling_.baseK = CeilAlign(basicTiling_.baseK, MXFP_BASEK_FACTOR); // mx_mmad requires basek align to 64
        bool isFp4Input = inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2;
        if (isFp4Input && !inputParams_.transB) {
            // 64: mx_mmad requires the inner axis to align to 64
            basicTiling_.baseN = CeilAlign(basicTiling_.baseN, static_cast<uint64_t>(64));
        }
    }
}

bool GroupedQmmTiling::IsBiasInL1() const
{
    // Reserve L1 for bias cases consumed by quant tiling.
    if (!inputParams_.hasBias) {
        return false;
    }
    const bool isFp4Input = inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2;
    const bool isMxFp4Fp32Bias = inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE && isFp4Input &&
                                 inputParams_.biasDtype == ge::DT_FLOAT;
    return inputParams_.biasDtype == ge::DT_INT32 || isMxFp4Fp32Bias;
}

void GroupedQmmTiling::InitCommonL1TilingFields()
{
    basicTiling_.stepM = 1UL;
    basicTiling_.stepN = 1UL;
    basicTiling_.singleCoreM = std::min(inputParams_.mSize, basicTiling_.baseM);
    basicTiling_.singleCoreN = std::min(inputParams_.nSize, basicTiling_.baseN);
    basicTiling_.singleCoreK = inputParams_.kSize;
    basicTiling_.iterateOrder = 0U;
    basicTiling_.dbL0c =
        (basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= aicoreParams_.l0cSize) ? DB_SIZE : 1;
}

ge::graphStatus GroupedQmmTiling::CalcLeftL1Size(uint64_t &leftL1Size) const
{
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    uint64_t totalL1Size = aicoreParams_.l1Size;
    uint64_t singleCoreBiasSize = IsBiasInL1() ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t singleCoreScaleSize = inputParams_.bQuantMode == optiling::QuantMode::PERCHANNEL_MODE &&
                                           inputParams_.kernelType == 0 && inputParams_.cDtype != ge::DT_INT32 ?
                                       basicTiling_.baseN * scaleDtypeSize :
                                       0;
    uint64_t usedSize = singleCoreBiasSize + singleCoreScaleSize;
    OP_CHECK_IF(
        totalL1Size <= usedSize,
        OP_LOGE(context_->GetNodeName(), "L1 space overflow. L1Size: %lu, used space: %lu", totalL1Size, usedSize),
        return ge::GRAPH_FAILED);
    leftL1Size = totalL1Size - usedSize;
    return ge::GRAPH_SUCCESS;
}

void GroupedQmmTiling::CalcAlignedMxBaseScaleSize(uint64_t &baseScaleASize, uint64_t &baseScaleBSize) const
{
    baseScaleASize = GetSizeWithDataType(
        CeilAlign(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE) * basicTiling_.baseM,
        inputParams_.perTokenScaleDtype);
    baseScaleBSize = GetSizeWithDataType(
        CeilAlign(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE) * basicTiling_.baseN,
        inputParams_.scaleDtype);
}

ge::graphStatus GroupedQmmTiling::CalL1Tiling()
{
    InitCommonL1TilingFields();
    if (inputParams_.kSize == 0) {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t leftL1Size = 0;
    OP_CHECK_IF(CalcLeftL1Size(leftL1Size) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalcLeftL1Size failed"), return ge::GRAPH_FAILED);
    return CalL1Depth(leftL1Size);
}

ge::graphStatus GroupedQmmTiling::CalL1Depth(uint64_t leftL1Size)
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);

    uint64_t baseScaleASize = 0;
    uint64_t baseScaleBSize = 0;
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        if (inputParams_.groupType == SPLIT_M) {
            CalcAlignedMxBaseScaleSize(baseScaleASize, baseScaleBSize);
        } else {
            baseScaleASize = GetSizeWithDataType(
                (basicTiling_.baseK / (MX_GROUP_SIZE * MXFP_MULTI_BASE_SIZE) + inputParams_.groupNum) *
                    MXFP_MULTI_BASE_SIZE * basicTiling_.baseM, // 2 is dim value of last scale dim
                inputParams_.perTokenScaleDtype);
            baseScaleBSize = GetSizeWithDataType(
                (basicTiling_.baseK / (MX_GROUP_SIZE * MXFP_MULTI_BASE_SIZE) + inputParams_.groupNum) *
                    MXFP_MULTI_BASE_SIZE * basicTiling_.baseN, // 2 is dim value of last pertokenScale dim
                inputParams_.scaleDtype);
        }
    }
    uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
    OP_CHECK_IF(leftL1Size < baseL1Size,
                OP_LOGE(context_->GetNodeName(), "L1 space overflow. Free L1Size : %lu, used space: %lu", leftL1Size,
                        baseL1Size),
                return ge::GRAPH_FAILED);
    uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL);
    uint64_t leftL1SizeByDepthInit = leftL1Size - depthInit * (baseL1Size);
    uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
    uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
    basicTiling_.depthA1 = std::max(depthASec, depthBSec);
    basicTiling_.depthB1 = basicTiling_.depthA1;
    if (basicTiling_.depthA1 * baseL1Size > leftL1Size) {
        basicTiling_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
        basicTiling_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
    }
    CalStepKs();
    if (inputParams_.bQuantMode == optiling::QuantMode::MX_PERGROUP_MODE) {
        CalScaleFactors();
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedQmmTiling::GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
{
    if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE_V120) {
        return depthInit;
    }
    uint64_t depthScale = leftSize / perDepthSize;
    if (depthInit > 1UL) {
        uint64_t baseKSize = GetSizeWithDataType(basicTiling_.baseK, inputParams_.aDtype);
        while ((depthScale * baseKSize) % GmmConstant::BASIC_BLOCK_SIZE_512 != 0 &&
               (depthScale * baseKSize) > GmmConstant::BASIC_BLOCK_SIZE_512) {
            depthScale -= 1UL;
        }
        if ((depthScale * baseKSize) % GmmConstant::BASIC_BLOCK_SIZE_512 != 0 &&
            (depthScale * baseKSize) >= GmmConstant::BASIC_BLOCK_SIZE_256) {
            depthScale = GmmConstant::BASIC_BLOCK_SIZE_256 / baseKSize;
        }
        depthScale = std::max(depthScale, static_cast<uint64_t>(1));
    } else {
        constexpr uint64_t index = 2; // 2: depth的值是2的幂
        depthScale = 1UL;
        while (depthScale * (perDepthSize) < leftSize) {
            depthScale *= index;
        }
        depthScale = depthScale == 1UL ? depthScale : depthScale / index;
    }
    return depthInit * depthScale;
}

void GroupedQmmTiling::CalStepKs()
{
    // depthA,depthB 为1时，stepka, stepkb 只能是1.
    basicTiling_.stepKa = basicTiling_.depthA1 == 1UL ? 1UL : basicTiling_.depthA1 / DB_SIZE;
    basicTiling_.stepKb = basicTiling_.depthB1 == 1UL ? 1UL : basicTiling_.depthB1 / DB_SIZE;

    if (basicTiling_.stepKa * basicTiling_.baseK > inputParams_.kSize) {
        basicTiling_.stepKa = CeilDiv(inputParams_.kSize, basicTiling_.baseK);
    }

    if (basicTiling_.stepKb * basicTiling_.baseK >= inputParams_.kSize) {
        basicTiling_.stepKb = CeilDiv(inputParams_.kSize, basicTiling_.baseK);
    }
    // G-B量化场景下，限制stepK最大为4, 防止issue queue阻塞
    if (inputParams_.aQuantMode == optiling::QuantMode::PERGROUP_MODE &&
        inputParams_.bQuantMode == optiling::QuantMode::PERBLOCK_MODE) {
        basicTiling_.stepKa = std::min(basicTiling_.stepKa, static_cast<uint64_t>(4)); // 4: G-B最大stepk值
        basicTiling_.stepKb = std::min(basicTiling_.stepKb, static_cast<uint64_t>(4)); // 4: G-B最大stepk值
    }
    if (basicTiling_.stepKa >= basicTiling_.stepKb && basicTiling_.stepKa * basicTiling_.baseK < inputParams_.kSize) {
        basicTiling_.stepKa = basicTiling_.stepKa / basicTiling_.stepKb * basicTiling_.stepKb;
    }
    if (basicTiling_.stepKb > basicTiling_.stepKa && basicTiling_.stepKb * basicTiling_.baseK < inputParams_.kSize) {
        basicTiling_.stepKb = basicTiling_.stepKb / basicTiling_.stepKa * basicTiling_.stepKa;
    }

    basicTiling_.depthA1 = basicTiling_.stepKa * DB_SIZE;
    basicTiling_.depthB1 = basicTiling_.stepKb * DB_SIZE;
}

void GroupedQmmTiling::CalScaleFactors()
{
    uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
    uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);
    uint64_t baseScaleASize = GetSizeWithDataType(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseM,
                                                  inputParams_.perTokenScaleDtype);
    uint64_t baseScaleBSize =
        GetSizeWithDataType(CeilDiv(basicTiling_.baseK, MX_GROUP_SIZE) * basicTiling_.baseN, inputParams_.scaleDtype);
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t baseBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t leftL1Size =
        aicoreParams_.l1Size - (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit = static_cast<uint32_t>(
        leftL1Size / (basicTiling_.depthA1 * baseScaleASize + basicTiling_.depthB1 * baseScaleBSize));

    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    uint32_t scaleFactorAMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleASize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorBMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V120 / baseScaleBSize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorA = static_cast<uint32_t>(inputParams_.kSize / (basicTiling_.stepKa * basicTiling_.baseK));
    uint32_t scaleFactorB = static_cast<uint32_t>(inputParams_.kSize / (basicTiling_.stepKb * basicTiling_.baseK));
    basicTiling_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    basicTiling_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    basicTiling_.scaleFactorA = std::min(scaleFactorAMax, basicTiling_.scaleFactorA);
    basicTiling_.scaleFactorB = std::min(scaleFactorBMax, basicTiling_.scaleFactorB);

    // 来自L1 size 的约束
    if (basicTiling_.scaleFactorA <= scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1Size -= (basicTiling_.scaleFactorA * basicTiling_.depthA1 * baseScaleASize);
        basicTiling_.scaleFactorB = std::min(
            static_cast<uint32_t>(leftL1Size / (basicTiling_.depthB1 * baseScaleBSize)), basicTiling_.scaleFactorB);
    } else if (basicTiling_.scaleFactorB <= scaleInit && basicTiling_.scaleFactorA > scaleInit) {
        leftL1Size -= (basicTiling_.scaleFactorB * basicTiling_.depthB1 * baseScaleBSize);
        basicTiling_.scaleFactorA = std::min(
            static_cast<uint32_t>(leftL1Size / (basicTiling_.depthA1 * baseScaleASize)), basicTiling_.scaleFactorA);
    } else if (basicTiling_.scaleFactorA > scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1Size -=
            (scaleInit * basicTiling_.depthB1 * baseScaleBSize + scaleInit * basicTiling_.depthA1 * baseScaleASize);
        uint32_t scaleASec = std::min(static_cast<uint32_t>(leftL1Size / (basicTiling_.depthA1 * baseScaleASize)),
                                      basicTiling_.scaleFactorA - scaleInit);
        uint32_t scaleBSec = std::min(static_cast<uint32_t>(leftL1Size / (basicTiling_.depthB1 * baseScaleBSize)),
                                      basicTiling_.scaleFactorB - scaleInit);
        basicTiling_.scaleFactorA = scaleASec >= scaleBSec ? (scaleASec + scaleInit) : scaleInit;
        basicTiling_.scaleFactorB = scaleASec < scaleBSec ? (scaleBSec + scaleInit) : scaleInit;
    }
}

uint64_t GroupedQmmTiling::GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    // shapeSize应该是偶数
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        // 2: 判断是否是偶数
        OP_CHECK_IF(shapeSize % 2 != 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        inputParams_.opType, "matrix/array", std::to_string(shapeSize),
                        "to get size of matrix/array, the number of elements must be even when dtype is FLOAT4/INT4"),
                    return 0);
        // 1/2: 这几种数据类型的dsize=1/2
        return shapeSize / 2UL;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t GroupedQmmTiling::GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        return shapeSize + shapeSize;
    } else {
        return shapeSize / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

void GQmmInputInfo::Reset()
{
    mSize = 0UL;
    kSize = 0UL;
    nSize = 0UL;
    groupNum = 0UL;
    outDtype = 0L;
    kernelType = 0UL;
    aQuantMode = QuantMode::DEFAULT;
    bQuantMode = QuantMode::DEFAULT;
    groupType = GroupedMatmul::NO_SPLIT;
    groupListType = 0;
    splitItem = 0;
    actType = 0;
    opName = nullptr;
    opType = "GroupedMatmul";
    aDtype = ge::DT_INT8;
    bDtype = ge::DT_INT8;
    cDtype = ge::DT_FLOAT16;
    biasDtype = ge::DT_INT32;
    scaleDtype = ge::DT_UINT64;
    perTokenScaleDtype = ge::DT_FLOAT;
    outDataDtype = ge::DT_FLOAT16;
    outScaleDtype = ge::DT_FLOAT;
    aFormat = ge::FORMAT_ND;
    bFormat = ge::FORMAT_ND;
    scaleFormat = ge::FORMAT_ND;
    cFormat = ge::FORMAT_ND;
    transA = false;
    transB = false;
    scaleNzFlag = false;
    hasBias = false;
    isSingleX = false;
    isSingleW = false;
    isSingleY = false;
    initFlag = false; // 避免重复解析flag
}

REGISTER_OPS_TILING_TEMPLATE(GroupedMatmul, GroupedQmmBasicApiTiling, 0);
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmul, GroupedQmmTiling, 1);
} // namespace optiling
