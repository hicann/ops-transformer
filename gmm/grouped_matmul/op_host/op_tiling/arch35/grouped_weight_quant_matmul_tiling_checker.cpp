/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_weight_quant_matmul_tiling_checker.cpp
 * \brief
 */

#include "grouped_weight_quant_matmul_tiling_checker.h"
#include "gmm/common/op_host/log_format_util.h"

namespace GmmConstants = optiling::GMMWeightQuantConstants;
namespace GmmTilingTool = optiling::GMMWeightQuantTilingTool;
using GmmTilingTool::IsA16MxFp4;
using GmmTilingTool::IsA16MxHiF4;
using GmmTilingTool::IsMxA8HiF4;
using GmmTilingTool::IsMxA8W4;

namespace optiling {

std::string GMMWeightQuantTilingChecker::GetDataFlowString() const
{
    return "When x-weight dtype is [" + ge::TypeUtils::DataTypeToSerialString(inputParams_.xDtype) + "]-[" +
           ge::TypeUtils::DataTypeToSerialString(inputParams_.weightDtype) + "]";
}

bool GMMWeightQuantTilingChecker::CheckGroupedMatmulWeightQuant()
{
    OP_CHECK_IF(inputParams_.transX, OP_LOGE(inputParams_.opName, "Transposed x is not supported."), return false);
    OP_CHECK_IF(!CheckUnsupportedDataFlow(),
                OP_LOGE(inputParams_.opName, "Input data contains unsupported dtype or format."), return false);
    OP_CHECK_IF(!CheckGroupScenario(), OP_LOGE(inputParams_.opName, "CheckGroupScenario failed."), return false);
    OP_CHECK_IF(!CheckTensorListSize(), OP_LOGE(inputParams_.opName, "CheckTensorListSize failed."), return false);
    OP_CHECK_IF(!CheckEmptyTensor(), OP_LOGE(inputParams_.opName, "CheckEmptyTensor failed."), return false);
    OP_CHECK_IF(!CheckInputDtype(), OP_LOGE(inputParams_.opName, "CheckInputDtype failed."), return false);
    OP_CHECK_IF(!CheckEveryTensor(), OP_LOGE(inputParams_.opName, "CheckEveryTensor failed."), return false);
    OP_CHECK_IF(!CheckPerTokenScale(), OP_LOGE(inputParams_.opName, "CheckPerTokenScale failed."), return false);
    return true;
}

bool GMMWeightQuantTilingChecker::CheckUnsupportedDataFlow() const
{
    if (IsMxA8W4(inputParams_) || IsMxA8HiF4(inputParams_) || IsA16MxFp4(inputParams_) || IsA16MxHiF4(inputParams_)) {
        OP_CHECK_IF(!(inputParams_.isWeightNz && inputParams_.transWeight),
                    OP_LOGE(inputParams_.opName, "%s, only transposed weight with NZ format is supported.",
                            GetDataFlowString().c_str()),
                    return false);
        OP_CHECK_IF(inputParams_.hasAntiquantOffset,
                    OP_LOGE(inputParams_.opName, "%s, antiquantOffset is not supported.", GetDataFlowString().c_str()),
                    return false);
    } else {
        OP_LOGE(inputParams_.opName,
                "GroupedMatmul does not support xDtype [%s], weightDtype [%s] and antiquantScaleDtype [%s].",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.xDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(inputParams_.weightDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(inputParams_.antiquantScaleDtype).c_str());
        return false;
    }
    return true;
}

bool GMMWeightQuantTilingChecker::CheckGroupScenario() const
{
    if (IsMxA8W4(inputParams_) || IsMxA8HiF4(inputParams_) || IsA16MxFp4(inputParams_) || IsA16MxHiF4(inputParams_)) {
        OP_CHECK_IF(inputParams_.groupType != GmmConstants::SPLIT_M,
                    OP_LOGE(inputParams_.opName, "%s, groupType only supports 0, but the actual value is [%ld].",
                            GetDataFlowString().c_str(), inputParams_.groupType),
                    return false);
    }

    if (inputParams_.groupType == GmmConstants::NO_SPLIT) {
        OP_CHECK_IF(inputParams_.splitItem != 0 && inputParams_.splitItem != 1,
                    OP_LOGE(inputParams_.opName,
                            "When groupType is -1, splitItem can only be 0 or 1, but the actual value is [%ld].",
                            inputParams_.splitItem),
                    return false);
    } else {
        OP_CHECK_IF(inputParams_.splitItem != 2 && inputParams_.splitItem != 3,
                    OP_LOGE(inputParams_.opName,
                            "When groupType is 0, splitItem can only be 2 or 3, but the actual value is [%ld].",
                            inputParams_.splitItem),
                    return false);
        OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                    OP_LOGE(inputParams_.opName, "groupListType only supports 0 or 1, but the actual value is [%ld].",
                            inputParams_.groupListType),
                    return false);
    }
    return true;
}

bool GMMWeightQuantTilingChecker::CheckTensorListSize() const
{
    OP_CHECK_IF(inputNum_.numX >= GmmConstants::MAX_TENSOR_CONT,
                OP_LOGE(inputParams_.opName, "tensorlist's length cannot exceed [%hu], but the actual length is [%hu].",
                        GmmConstants::MAX_TENSOR_CONT, inputNum_.numX),
                return false);

    if (inputParams_.groupType == GmmConstants::NO_SPLIT) {
        OP_CHECK_IF(
            inputNum_.numX != inputNum_.numWeight,
            OP_LOGE(inputParams_.opName,
                    "When groupType is -1 (no split), the sizes of x and weight should be all the same, but the "
                    "actual sizes are [%hu] and [%hu].",
                    inputNum_.numX, inputNum_.numWeight),
            return false);
    } else {
        OP_CHECK_IF(inputNum_.numX != 1 || inputNum_.numWeight != 1,
                    OP_LOGE(inputParams_.opName,
                            "When groupType is 0 (split M), the sizes of x and weight should all be 1, but the actual "
                            "sizes are [%hu] and [%hu].",
                            inputNum_.numX, inputNum_.numWeight),
                    return false);
    }

    OP_CHECK_IF(inputNum_.numAntiquantScale != inputNum_.numWeight,
                OP_LOGE(inputParams_.opName,
                        "antiquantScale size should be equal to weight size, actual sizes are [%hu], [%hu].",
                        inputNum_.numAntiquantScale, inputNum_.numWeight),
                return false);

    if (inputParams_.hasAntiquantOffset) {
        OP_CHECK_IF(inputNum_.numAntiquantOffset != inputNum_.numWeight,
                    OP_LOGE(inputParams_.opName,
                            "antiquantOffset size should be equal to weight size, actual sizes are [%hu], [%hu].",
                            inputNum_.numAntiquantOffset, inputNum_.numWeight),
                    return false);
    }

    if (inputParams_.hasBias) {
        OP_CHECK_IF(
            inputNum_.numBias != inputNum_.numWeight,
            OP_LOGE(inputParams_.opName, "bias size should be equal to weight size, actual sizes are [%hu], [%hu].",
                    inputNum_.numBias, inputNum_.numWeight),
            return false);
    }

    return true;
}

bool GMMWeightQuantTilingChecker::CheckEmptyTensor() const
{
    // all M or N be zero, get true
    bool zeroM = true;
    bool zeroN = true;
    // exist one K be zero, get true
    bool zeroK = false;
    // current view_shape transpose is always false false
    for (size_t i = 0; i < inputNum_.numX; ++i) {
        auto xShapePtr = context_->GetDynamicInputShape(GmmConstants::X_INDEX, i);
        auto xShape = xShapePtr->GetStorageShape();
        size_t xDimNum = xShape.GetDimNum();
        OP_CHECK_IF(xDimNum < GmmConstants::MIN_X_DIM || xDimNum > GmmConstants::MAX_X_DIM,
                    OP_LOGE(inputParams_.opName, "Invalid x dimension, expect 2-6, actual [%zu].", xDimNum),
                    return false);
        int64_t m = inputParams_.transX ? xShape.GetDim(xDimNum - GmmConstants::LAST_FIRST_DIM_INDEX) :
                                          xShape.GetDim(xDimNum - GmmConstants::LAST_SECOND_DIM_INDEX);
        int64_t k = inputParams_.transX ? xShape.GetDim(xDimNum - GmmConstants::LAST_SECOND_DIM_INDEX) :
                                          xShape.GetDim(xDimNum - GmmConstants::LAST_FIRST_DIM_INDEX);
        for (size_t dimIdx = 0; dimIdx < xDimNum - GmmConstants::LAST_SECOND_DIM_INDEX; dimIdx++) {
            m *= xShape.GetDim(dimIdx);
        }
        zeroM = zeroM && m == 0;
        zeroK = zeroK || (k == 0);
    }
    for (size_t i = 0; i < inputNum_.numWeight; ++i) {
        auto wShapePtr = context_->GetDynamicInputShape(GmmConstants::WEIGHT_INDEX, i);
        auto wShape = wShapePtr->GetOriginShape();
        size_t wDimNum = wShape.GetDimNum();
        OP_CHECK_IF(wDimNum < GmmConstants::MIN_ND_DIM,
                    OP_LOGE(inputParams_.opName, "Invalid weight dimension, expect >= 2, actual %zu.", wDimNum),
                    return false);
        int64_t n = inputParams_.transWeight ? wShape.GetDim(wDimNum - GmmConstants::LAST_SECOND_DIM_INDEX) :
                                               wShape.GetDim(wDimNum - GmmConstants::LAST_FIRST_DIM_INDEX);
        zeroN = zeroN && (n == 0);
    }

    OP_CHECK_IF(!zeroM && !zeroN && zeroK,
                OP_LOGE(inputParams_.opName, " GroupedMatmul does not support input K being 0 unless all M/N is 0."),
                return false);
    return true;
}

bool GMMWeightQuantTilingChecker::CheckInputDtype() const
{
    if (IsMxA8W4(inputParams_) || IsMxA8HiF4(inputParams_)) {
        OP_CHECK_IF(inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(inputParams_.opName, "perTokenScale dtype [%s] does not match expected dtype [%s].",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(ge::DT_FLOAT8_E8M0).c_str()),
                    return false);
        if (inputParams_.hasBias) {
            OP_CHECK_IF(inputParams_.biasDtype != inputParams_.yDtype,
                        OP_LOGE(inputParams_.opName, "bias dtype [%s] does not match expected dtype [%s].",
                                ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                                ge::TypeUtils::DataTypeToSerialString(inputParams_.yDtype).c_str()),
                        return false);
        }
    } else if ((IsA16MxFp4(inputParams_) || IsA16MxHiF4(inputParams_)) && inputParams_.hasBias) {
        OP_CHECK_IF(inputParams_.biasDtype != inputParams_.xDtype,
                    OP_LOGE(inputParams_.opName, "bias dtype [%s] does not match expected dtype [%s].",
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(inputParams_.xDtype).c_str()),
                    return false);
    }
    return true;
}

bool GMMWeightQuantTilingChecker::CheckEveryTensor()
{
    for (size_t i = 0; i < inputNum_.numX; i++) {
        OP_CHECK_IF(!CheckNotNull(i), OP_LOGE(inputParams_.opName, "CheckNotNull failed."), return false);
        OP_CHECK_IF(!CheckEveryTensorDtype(i), OP_LOGE(inputParams_.opName, "CheckEveryTensorDtype failed."),
                    return false);
        OP_CHECK_IF(!CheckEveryTensorShape(i), OP_LOGE(inputParams_.opName, "CheckEveryTensorShape failed."),
                    return false);
    }
    return true;
}

bool GMMWeightQuantTilingChecker::CheckNotNull(size_t tensorIdx) const
{
    OP_CHECK_IF(!CheckNotNullDesc(GmmConstants::X_INDEX, tensorIdx), OP_LOGE(inputParams_.opName, "x Desc is nullptr."),
                return false);
    OP_CHECK_IF(!CheckNotNullDesc(GmmConstants::WEIGHT_INDEX, tensorIdx),
                OP_LOGE(inputParams_.opName, "weight Desc is nullptr."), return false);
    OP_CHECK_IF(!CheckNotNullDesc(GmmConstants::ANTIQUANT_SCALE_INDEX, tensorIdx),
                OP_LOGE(inputParams_.opName, "antiquantscale Desc is nullptr."), return false);

    if (inputParams_.hasBias) {
        OP_CHECK_IF(!CheckNotNullDesc(GmmConstants::BIAS_INDEX, tensorIdx),
                    OP_LOGE(inputParams_.opName, "bias Desc is nullptr."), return false);
    }

    if (inputParams_.hasAntiquantOffset) {
        OP_CHECK_IF(!CheckNotNullDesc(GmmConstants::ANTIQUANT_OFFSET_INDEX, tensorIdx),
                    OP_LOGE(inputParams_.opName, "antiquantoffset Desc is nullptr."), return false);
    }

    return true;
}

bool GMMWeightQuantTilingChecker::CheckNotNullDesc(size_t inputIdx, size_t tensorIdx) const
{
    auto tmpIDesc = context_->GetDynamicInputDesc(inputIdx, tensorIdx);
    OP_CHECK_IF(tmpIDesc == nullptr, OP_LOGE(inputParams_.opName, "Desc[%zu] is nullptr.", tensorIdx), return false);
    return true;
}

bool GMMWeightQuantTilingChecker::CheckEveryTensorDtype(size_t tensorIdx) const
{
    OP_CHECK_IF(!CheckTensorDtype(GmmConstants::X_INDEX, tensorIdx, inputParams_.xDtype, GmmConstants::X),
                OP_LOGE(inputParams_.opName, "Check Tensor x Dtype failed."), return false);
    OP_CHECK_IF(
        !CheckTensorDtype(GmmConstants::WEIGHT_INDEX, tensorIdx, inputParams_.weightDtype, GmmConstants::WEIGHT),
        OP_LOGE(inputParams_.opName, "Check Tensor weight Dtype failed."), return false);
    OP_CHECK_IF(!CheckTensorDtype(GmmConstants::ANTIQUANT_SCALE_INDEX, tensorIdx, inputParams_.antiquantScaleDtype,
                                  GmmConstants::ANTIQUANT_SCALE),
                OP_LOGE(inputParams_.opName, "Check Tensor antiquantScale Dtype failed."), return false);
    if (inputParams_.hasAntiquantOffset) {
        OP_CHECK_IF(!CheckTensorDtype(GmmConstants::ANTIQUANT_OFFSET_INDEX, tensorIdx,
                                      inputParams_.antiquantOffsetDtype, GmmConstants::ANTIQUANT_OFFSET),
                    OP_LOGE(inputParams_.opName, "Check Tensor antiquantOffset Dtype failed."), return false);
    }
    if (inputParams_.hasBias) {
        OP_CHECK_IF(!CheckTensorDtype(GmmConstants::BIAS_INDEX, tensorIdx, inputParams_.biasDtype, GmmConstants::BIAS),
                    OP_LOGE(inputParams_.opName, "Check Tensor bias Dtype failed."), return false);
    }
    return true;
}

bool GMMWeightQuantTilingChecker::CheckTensorDtype(size_t inputIdx, size_t tensorIdx, const ge::DataType expectedDtype,
                                                   const std::string& tensorType) const
{
    auto tmpDesc = context_->GetDynamicInputDesc(inputIdx, tensorIdx);
    auto tmpDtype = tmpDesc->GetDataType();
    OP_CHECK_IF(tmpDtype != expectedDtype,
                OP_LOGE(inputParams_.opName,
                        "The dtype of each tensor in %s tensor list must be consistent.%s[%zu]'s dtype [%s]"
                        "is different from the first tensor's dtype [%s].",
                        tensorType.c_str(), tensorType.c_str(), tensorIdx,
                        ge::TypeUtils::DataTypeToSerialString(tmpDtype).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(expectedDtype).c_str()),
                return false);
    return true;
}

bool GMMWeightQuantTilingChecker::CheckEveryTensorShape(size_t tensorIdx)
{
    OP_CHECK_IF(!CheckEveryTensorDim(tensorIdx), OP_LOGE(inputParams_.opName, "CheckEveryTensorDim failed."),
                return false);

    OP_CHECK_IF(!GetMKN(tensorIdx), OP_LOGE(inputParams_.opName, "GetMKN failed."), return false);

    OP_CHECK_IF(!CheckXAndWeightShape(tensorIdx), OP_LOGE(inputParams_.opName, "CheckXAndWeightShape failed."),
                return false);

    OP_CHECK_IF(!CheckTensorShape(GmmConstants::ANTIQUANT_SCALE_INDEX, tensorIdx, GmmConstants::ANTIQUANT_SCALE),
                OP_LOGE(inputParams_.opName, "Check antiquantScale tensor shape failed."), return false);

    if (inputParams_.hasAntiquantOffset) {
        OP_CHECK_IF(!CheckTensorShape(GmmConstants::ANTIQUANT_OFFSET_INDEX, tensorIdx, GmmConstants::ANTIQUANT_OFFSET),
                    OP_LOGE(inputParams_.opName, "Check antiquantOffset tensor shape failed."), return false);
    }

    if (inputParams_.hasBias) {
        OP_CHECK_IF(!CheckTensorShape(GmmConstants::BIAS_INDEX, tensorIdx, GmmConstants::BIAS),
                    OP_LOGE(inputParams_.opName, "Check bias tensor shape failed."), return false);
    }

    return true;
}

bool GMMWeightQuantTilingChecker::CheckEveryTensorDim(size_t tensorIdx) const
{
    if (inputParams_.groupType == GmmConstants::SPLIT_M) {
        OP_CHECK_IF(!CheckTensorDimSingleXSingleWeightSingleY(tensorIdx),
                    OP_LOGE(inputParams_.opName, "CheckTensorDimSingleXSingleWeightSingleY failed."), return false);
    }
    return true;
}

bool GMMWeightQuantTilingChecker::CheckTensorDimSingleXSingleWeightSingleY(size_t tensorIdx) const
{
    OP_CHECK_IF(!CheckTensorDim(GmmConstants::X_INDEX, tensorIdx, GmmConstants::DIM_NUM_2, GmmConstants::X),
                OP_LOGE(inputParams_.opName, "Invalid x dimension."), return false);

    size_t targetWeightDim = inputParams_.isWeightNz ?
                                 GmmConstants::DIM_NUM_5 :
                                 GmmConstants::DIM_NUM_3; // NZ (g, n1, k1, k0, n0) 5维 ND (g, k, n) 3维
    OP_CHECK_IF(!CheckTensorDim(GmmConstants::WEIGHT_INDEX, tensorIdx, targetWeightDim, GmmConstants::WEIGHT),
                OP_LOGE(inputParams_.opName, "Invalid weight dimension."), return false);

    size_t targetAntiquantParamsDim = GmmConstants::DIM_NUM_4;
    if (IsA16MxFp4(inputParams_) || IsA16MxHiF4(inputParams_)) {
        targetAntiquantParamsDim = GmmConstants::DIM_NUM_3;
    }
    OP_CHECK_IF(!CheckTensorDim(GmmConstants::ANTIQUANT_SCALE_INDEX, tensorIdx, targetAntiquantParamsDim,
                                GmmConstants::ANTIQUANT_SCALE),
                OP_LOGE(inputParams_.opName, "Invalid antiquantScale dimension."), return false);

    if (inputParams_.hasBias) {
        OP_CHECK_IF(!CheckTensorDim(GmmConstants::BIAS_INDEX, tensorIdx, GmmConstants::DIM_NUM_2, GmmConstants::BIAS),
                    OP_LOGE(inputParams_.opName, "Invalid bias dimension."), return false);
    }

    return true;
}

bool GMMWeightQuantTilingChecker::CheckTensorDim(size_t inputIdx, size_t tensorIdx, size_t targetDim,
                                                 const std::string& tensorType) const
{
    auto tmpShapePtr = context_->GetDynamicInputShape(inputIdx, tensorIdx);
    size_t tmpDimNum = tmpShapePtr->GetStorageShape().GetDimNum();
    OP_CHECK_IF(tmpDimNum != targetDim,
                OP_LOGE(inputParams_.opName, "The expected dimension of %s[%zu] is [%zu], but the actual dim is [%zu].",
                        tensorType.c_str(), tensorIdx, targetDim, tmpDimNum),
                return false);

    if (inputIdx == GmmConstants::WEIGHT_INDEX && inputParams_.isWeightNz) {
        auto oriDimNum = tmpShapePtr->GetOriginShape().GetDimNum();
        OP_CHECK_IF(oriDimNum != targetDim - GmmConstants::DIM_NUM_2,
                    OP_LOGE(inputParams_.opName,
                            "The expected origin dimension of %s[%zu] is [%zu], but the actual origin dim is [%zu].",
                            tensorType.c_str(), tensorIdx, targetDim - GmmConstants::DIM_NUM_2, oriDimNum),
                    return false);
    }
    return true;
}

bool GMMWeightQuantTilingChecker::GetMKN(size_t tensorIdx)
{
    auto xShape = context_->GetDynamicInputShape(GmmConstants::X_INDEX, tensorIdx)->GetStorageShape();
    auto xDimNum = xShape.GetDimNum();

    // 校验batch轴和M轴大于等于0，不考虑X转置的情况
    for (size_t dimIdx = 0; dimIdx < xDimNum - GmmConstants::LAST_FIRST_DIM_INDEX; dimIdx++) {
        int64_t xDimValue = xShape.GetDim(dimIdx);
        OP_CHECK_IF(
            xDimValue < 0,
            OP_LOGE(inputParams_.opName, "The dimension[%zu] of x[%zu] should be >= 0, but actual value is [%ld].",
                    dimIdx, tensorIdx, xDimValue),
            return false);
    }

    int64_t xKDimValue = xShape.GetDim(xDimNum - GmmConstants::LAST_FIRST_DIM_INDEX);
    OP_CHECK_IF(
        xKDimValue < 0,
        OP_LOGE(inputParams_.opName, "The k dimension of x[%zu] should be not be negative, but actual value is [%ld].",
                tensorIdx, xKDimValue),
        return false);

    auto wShape = context_->GetDynamicInputShape(GmmConstants::WEIGHT_INDEX, tensorIdx)->GetOriginShape();
    auto wDimNum = wShape.GetDimNum();
    int64_t weightKDimValue = inputParams_.transWeight ? wShape.GetDim(wDimNum - GmmConstants::LAST_FIRST_DIM_INDEX) :
                                                         wShape.GetDim(wDimNum - GmmConstants::LAST_SECOND_DIM_INDEX);
    int64_t weightNDimValue = inputParams_.transWeight ? wShape.GetDim(wDimNum - GmmConstants::LAST_SECOND_DIM_INDEX) :
                                                         wShape.GetDim(wDimNum - GmmConstants::LAST_FIRST_DIM_INDEX);

    OP_CHECK_IF(weightNDimValue < 0,
                OP_LOGE(inputParams_.opName,
                        "The n dimensions of the weight[%zu] should not be negative, but actual value is [%ld].",
                        tensorIdx, weightNDimValue),
                return false);

    dimInfo_.gDim = inputParams_.groupType == GmmConstants::SPLIT_M ? wShape.GetDim(0) : 0L;
    dimInfo_.mDim = xShape.GetDim(0); // 暂不考虑含batch的场景
    dimInfo_.xKDim = xKDimValue;
    dimInfo_.weightKDim = weightKDimValue;
    dimInfo_.nDim = weightNDimValue;

    return true;
}

bool GMMWeightQuantTilingChecker::CheckXAndWeightShape(size_t tensorIdx) const
{
    OP_CHECK_IF(
        dimInfo_.xKDim != dimInfo_.weightKDim,
        OP_LOGE(inputParams_.opName,
                "The k dimension of x[%zu] and weight[%zu] should be equal, but actual values are [%ld] and [%ld].",
                tensorIdx, tensorIdx, dimInfo_.xKDim, dimInfo_.weightKDim),
        return false);

    if (inputParams_.enableMxShapeGeneralization && IsMxA8W4(inputParams_)) {
        // Each byte packs 2 FP4 elements, so K=2 and N=1 make the transpose direction ambiguous.
        OP_CHECK_IF(dimInfo_.weightKDim % 2 != 0 || (dimInfo_.weightKDim == 2 && dimInfo_.nDim == 1),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "K, N",
                        Ops::Transformer::Gmm::FormatString("%ld, %ld", dimInfo_.weightKDim, dimInfo_.nDim),
                        "the value of K must be even, and N can not be 1 when K is 2 for 960DT MxA8W4 WeightNZ"),
                    return false);
    } else {
        OP_CHECK_IF(
            dimInfo_.nDim % GmmConstants::N_K_ALIGN_VALUE_4BIT != 0 ||
                dimInfo_.weightKDim % GmmConstants::N_K_ALIGN_VALUE_4BIT != 0,
            OP_LOGE(
                inputParams_.opName,
                "The value of dim n, k should be an integer multiple of [%ld], but actual values are [%ld] and [%ld].",
                GmmConstants::N_K_ALIGN_VALUE_4BIT, dimInfo_.nDim, dimInfo_.weightKDim),
            return false);
    }

    if (inputParams_.groupType == GmmConstants::SPLIT_M) {
        OP_CHECK_IF(dimInfo_.gDim != static_cast<int64_t>(inputParams_.groupNum),
                    OP_LOGE(inputParams_.opName,
                            "The length of groupList must match the value of first dimension of weight, but actual "
                            "values are [%lu] and [%ld].",
                            inputParams_.groupNum, dimInfo_.gDim),
                    return false);
    }

    if (inputParams_.isWeightNz) {
        auto wNzShape = context_->GetDynamicInputShape(GmmConstants::WEIGHT_INDEX, tensorIdx)->GetStorageShape();
        auto wNzDimNum = wNzShape.GetDimNum();
        if (inputParams_.enableMxShapeGeneralization && IsMxA8W4(inputParams_)) {
            OP_CHECK_IF(
                // The 4th axis from the end counts K blocks. Add 63 (64 - 1) to round up to blocks of 64.
                wNzShape.GetDim(wNzDimNum - 4) != (dimInfo_.weightKDim + 63L) / 64L ||
                    // The 3rd axis from the end counts N blocks. Add 15 (16 - 1) to round up to blocks of 16.
                    wNzShape.GetDim(wNzDimNum - 3) != (dimInfo_.nDim + 15L) / 16L ||
                    // The 2nd axis from the end is the N block size, fixed at 16.
                    wNzShape.GetDim(wNzDimNum - 2) != 16L,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    inputParams_.opName, "weight NZ K block count, weight NZ N block count, weight NZ N block size",
                    Ops::Transformer::Gmm::FormatString("%ld, %ld, %ld", wNzShape.GetDim(wNzDimNum - 4),
                                                        wNzShape.GetDim(wNzDimNum - 3), wNzShape.GetDim(wNzDimNum - 2)),
                    Ops::Transformer::Gmm::FormatString(
                        "the values must be %ld, %ld and 16 respectively for 960DT MxA8W4 WeightNZ",
                        (dimInfo_.weightKDim + 63L) / 64L, (dimInfo_.nDim + 15L) / 16L)),
                return false);
        }
        OP_CHECK_IF(wNzShape.GetDim(wNzDimNum - GmmConstants::LAST_FIRST_DIM_INDEX) != GmmConstants::C0_64,
                    OP_LOGE(inputParams_.opName,
                            "The C0 of weight with NZ format should be [%ld], but the actual value is [%ld].",
                            GmmConstants::C0_64, wNzShape.GetDim(wNzDimNum - GmmConstants::LAST_FIRST_DIM_INDEX)),
                    return false);
    }

    return true;
}

bool GMMWeightQuantTilingChecker::CheckTensorShape(size_t inputIdx, size_t tensorIdx,
                                                   const std::string& tensorType) const
{
    // 校验bias、antiquantScale和antiquantOffset的shape
    auto tensorShapePtr = context_->GetDynamicInputShape(inputIdx, tensorIdx);
    auto tensorShape = tensorShapePtr->GetStorageShape();
    size_t tensorDimNum = tensorShape.GetDimNum();

    if (inputParams_.groupType == GmmConstants::SPLIT_M) {
        // 校验bias、antiquantScale和antiquantOffset的G轴
        int64_t tensorGDim = tensorShape.GetDim(0);
        OP_CHECK_IF(tensorGDim != dimInfo_.gDim,
                    OP_LOGE(inputParams_.opName, "%s gDim [%ld] should be euqal to groupList length [%ld].",
                            tensorType.c_str(), tensorGDim, dimInfo_.gDim),
                    return false);
    }

    // 校验bias、antiquantScale和antiquantOffset的N轴
    // MxA8W4 antiquantScale的shape为(g, n, k/64, 2)，MxA8HiF4 antiquantScale的shape为(g, n, k/64, 1)
    // A16MxF4 antiquantScale的shape为(g, n, k/32)
    // 其他输入shape为(n,) (g, n)
    size_t tensorNDimIdx = tensorDimNum - GmmConstants::LAST_FIRST_DIM_INDEX;
    if (inputIdx == GmmConstants::ANTIQUANT_SCALE_INDEX) {
        if (IsMxA8W4(inputParams_) || IsMxA8HiF4(inputParams_)) {
            tensorNDimIdx = tensorDimNum - GmmConstants::LAST_THIRD_DIM_INDEX;
        } else if (IsA16MxFp4(inputParams_) || IsA16MxHiF4(inputParams_)) {
            tensorNDimIdx = tensorDimNum - GmmConstants::LAST_SECOND_DIM_INDEX;
        }
    }
    int64_t tensorNDimValue = tensorShape.GetDim(tensorNDimIdx);
    OP_CHECK_IF(tensorNDimValue != dimInfo_.nDim,
                OP_LOGE(inputParams_.opName,
                        "The NDim of %s[%zu] should be equal to the NDim of weight[%zu], but the actual values are "
                        "[%ld] and [%ld].",
                        tensorType.c_str(), tensorIdx, tensorIdx, tensorNDimValue, dimInfo_.nDim),
                return false);

    if ((IsMxA8W4(inputParams_) || IsMxA8HiF4(inputParams_) || IsA16MxFp4(inputParams_)) &&
        inputIdx == GmmConstants::ANTIQUANT_SCALE_INDEX) {
        OP_CHECK_IF(!CheckGroupSize(tensorShape, GmmConstants::ANTIQUANT_SCALE),
                    OP_LOGE(inputParams_.opName, "Check mx antiquantScale[%zu] groupSize failed.", tensorIdx),
                    return false);
    }

    return true;
}

bool GMMWeightQuantTilingChecker::CheckGroupSize(const gert::Shape& scaleShape, const std::string& tensorType) const
{
    if (inputParams_.enableMxShapeGeneralization && IsMxA8W4(inputParams_)) {
        const size_t dims = scaleShape.GetDimNum();
        // Each pair covers 64 K elements. Add 63 (64 - 1) to round up the pair count.
        const int64_t expectedPairs = (dimInfo_.weightKDim + 63L) / 64L;
        OP_CHECK_IF(
            // The 2nd axis from the end is the scale pair count.
            scaleShape.GetDim(dims - 2) != expectedPairs ||
                // The last axis is fixed at 2 scales per pair.
                scaleShape.GetDim(dims - 1) != 2L,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                inputParams_.opName, tensorType + " pair count, " + tensorType + " pair size",
                Ops::Transformer::Gmm::FormatString("%ld, %ld", scaleShape.GetDim(dims - 2),
                                                    scaleShape.GetDim(dims - 1)),
                Ops::Transformer::Gmm::FormatString(
                    "the values must be %ld and %ld respectively for 960DT MxA8W4 WeightNZ", expectedPairs, 2L)),
            return false);
        return true;
    }
    // MxA8W4: antiquantScale shape (g, n, k/64, 2); perTokenScale shape (m, k/64, 2)
    // MxA8HiF4: antiquantScale shape (g, n, k/64, 1); perTokenScale shape (m, k/64, 2)
    // A16MxF4: antiquantScale shape (g, n, k/32)
    // A16MxHiF4: antiquantScale shape (g, n, k/64, 1)
    size_t scaleDimNum = scaleShape.GetDimNum();
    int64_t expectedGroupSize = 1;
    int64_t scaleGroupNum = 1;

    if (IsMxA8W4(inputParams_) || IsMxA8HiF4(inputParams_)) {
        int64_t expectedMxScaleLastDim = GmmConstants::MX_SCALE_LAST_DIM_2;
        expectedGroupSize = GmmConstants::MX_GROUP_SIZE_32;
        if (IsMxA8HiF4(inputParams_) && tensorType == GmmConstants::ANTIQUANT_SCALE) {
            expectedMxScaleLastDim = GmmConstants::MX_SCALE_LAST_DIM_1;
            expectedGroupSize = GmmConstants::MX_GROUP_SIZE_64;
        }

        int64_t mxScaleLastDim = scaleShape.GetDim(scaleDimNum - GmmConstants::LAST_FIRST_DIM_INDEX);
        OP_CHECK_IF(
            mxScaleLastDim != expectedMxScaleLastDim,
            OP_LOGE(inputParams_.opName, "The last dim of mx %s should be [%ld], but the actual value is [%ld].",
                    tensorType.c_str(), expectedMxScaleLastDim, mxScaleLastDim),
            return false);

        scaleGroupNum = scaleShape.GetDim(scaleDimNum - GmmConstants::LAST_SECOND_DIM_INDEX) * mxScaleLastDim;
    } else if (IsA16MxFp4(inputParams_)) {
        expectedGroupSize = GmmConstants::MX_GROUP_SIZE_32;
        scaleGroupNum = scaleShape.GetDim(scaleDimNum - GmmConstants::LAST_FIRST_DIM_INDEX);
    } else if (IsA16MxHiF4(inputParams_)) {
        expectedGroupSize = GmmConstants::MX_GROUP_SIZE_64;
        scaleGroupNum = scaleShape.GetDim(scaleDimNum - GmmConstants::LAST_FIRST_DIM_INDEX);
    }

    OP_CHECK_IF(
        scaleGroupNum <= 0,
        OP_LOGE(inputParams_.opName, "%s perGroupNum should be greater than 0, but the actual perGroupNum is [%ld]. ",
                tensorType.c_str(), scaleGroupNum),
        return false);

    OP_CHECK_IF(dimInfo_.weightKDim % scaleGroupNum != 0,
                OP_LOGE(inputParams_.opName,
                        "The value of kSize must be divisible by %s perGroupNum, but the actual kSize is [%ld] and "
                        "perGroupNum is [%ld].",
                        tensorType.c_str(), dimInfo_.weightKDim, scaleGroupNum),
                return false);

    int64_t groupSize = dimInfo_.weightKDim / scaleGroupNum;
    OP_CHECK_IF(groupSize != expectedGroupSize,
                OP_LOGE(inputParams_.opName, "Mx %s groupSize should be [%ld], but the actual value is [%ld].",
                        tensorType.c_str(), expectedGroupSize, groupSize),
                return false);

    return true;
}

bool GMMWeightQuantTilingChecker::CheckPerTokenScale() const
{
    if (IsMxA8W4(inputParams_) || IsMxA8HiF4(inputParams_)) {
        auto perTokenScaleShapePtr = context_->GetOptionalInputShape(GmmConstants::PER_TOKEN_SCALE_INDEX);
        gert::Shape perTokenScaleShape = perTokenScaleShapePtr->GetStorageShape();
        size_t perTokenScaleDimNum = perTokenScaleShape.GetDimNum();
        OP_CHECK_IF(
            perTokenScaleDimNum != GmmConstants::DIM_NUM_3,
            OP_LOGE(inputParams_.opName, "The dimension of perTokenScale should be [%zu], but the actual dim is [%zu].",
                    GmmConstants::DIM_NUM_3, perTokenScaleDimNum),
            return false);
        OP_CHECK_IF(
            perTokenScaleShape.GetDim(0) != dimInfo_.mDim,
            OP_LOGE(
                inputParams_.opName,
                "The MDim of perTokenScale should be equal to that of x, but the actual values are [%ld] and [%ld].",
                perTokenScaleShape.GetDim(0), dimInfo_.mDim),
            return false);
        OP_CHECK_IF(!CheckGroupSize(perTokenScaleShape, GmmConstants::PER_TOKEN_SCALE),
                    OP_LOGE(inputParams_.opName, "Check mx perTokenScale groupSize failed. "), return false);
    }
    return true;
}

} // namespace optiling
