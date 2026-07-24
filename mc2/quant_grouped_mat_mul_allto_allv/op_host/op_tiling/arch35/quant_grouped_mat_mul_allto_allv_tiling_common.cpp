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
 * \file quant_grouped_mat_mul_allto_allv_tiling_common.cpp
 * \brief
 */

#include "op_mc2.h"
#include "mc2_log.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_common.h"
#include "quant_grouped_mat_mul_allto_allv_tiling_adapter.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <tiling/tiling_api.h>
#include "mc2_comm_utils.h"
#include <numeric>

using namespace Mc2Log;
using namespace AscendC;
using namespace Mc2Tiling;
using namespace Mc2Tiling::Mc2GroupedMatmul;

namespace Mc2Tiling {

const std::vector<uint32_t> QUANT_GMM_X_DTYPE_LIST = {
    ge::DT_HIFLOAT8,
};
const std::vector<uint32_t> QUANT_GMM_WEIGHT_DTYPE_LIST = {
    ge::DT_HIFLOAT8,
};
const std::vector<uint32_t> QUANT_GMM_X_SCALE_DTYPE_LIST = {
    ge::DT_FLOAT,
};
const std::vector<uint32_t> QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST = {
    ge::DT_FLOAT,
};
const std::vector<uint32_t> QUANT_GMM_Y_DTYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};

bool QuantGroupedMatmulAllToAllvTilingCommon::IsContains(const std::vector<uint32_t> &list, uint32_t value)
{
    return std::count(list.begin(), list.end(), value) > 0;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckShapeDimensions(const gert::StorageShape *shape,
                                                                              uint64_t dims, const char *shapeName)
{
    uint64_t dimNum = shape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK((dimNum != dims),
                    OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, shapeName, std::to_string(dimNum) + "D",
                                                 std::to_string(dims) + "D"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::GetShapeAttrsInfo()
{
    // base check required para
    auto status = QuantGroupedMatmulAllToAllvTilingBase::GetShapeAttrsInfo();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((opName_ == nullptr), OP_LOGE("QUANTGMMALLTOALLV", "The opName_ is null."),
                    return ge::GRAPH_FAILED);

    localParams_.opName = opName_;
    return ge::GRAPH_SUCCESS;
}

// not support param
ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckOpInputSingleParamsTensorNotSupport()
{
    auto sendCountsTensorShape = context_->GetOptionalInputShape(SEND_COUNTS_TENSOR_OPTIONAL_INDEX);
    auto recvCountsTensorShape = context_->GetOptionalInputShape(RECV_COUNTS_TENSOR_OPTIONAL_INDEX);
    bool sendCountsTensorShapeNotNull = sendCountsTensorShape != nullptr;
    bool recvCountsTensorShapeNotNull = recvCountsTensorShape != nullptr;
    OP_TILING_CHECK(sendCountsTensorShapeNotNull || recvCountsTensorShapeNotNull,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_,
                            "sendCountsTensor, recvCountsTensor",
                            "not null",
                            "The values of sendCountsTensor and recvCountsTensor must be nullptr."),
                    return ge::GRAPH_FAILED);
    auto commQuantScaleTensorShape = context_->GetOptionalInputShape(COMM_QUANT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(commQuantScaleTensorShape != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "commQuantScale",
                                                          "not null",
                                                          "The value of commQuantScale must be nullptr."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// quant must support param
ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckOpInputSingleParamsTensorSupport()
{
    auto gmmXScaleTensorShape = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX);
    auto gmmWeightScaleTensorShape = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX);
    bool gmmXScaleTensorShapeNull = gmmXScaleTensorShape == nullptr;
    bool gmmWeightScaleTensorShapeNull = gmmWeightScaleTensorShape == nullptr;
    OP_TILING_CHECK(gmmXScaleTensorShapeNull || gmmWeightScaleTensorShapeNull,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_,
                            "gmmXScale, gmmWeightScale",
                            "null",
                            "The values of gmmXScale and gmmWeightScale must not be nullptr."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckOpInputSingleParamsTensorMM()
{
    auto mmXTensorShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    auto mmWeightTensorShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    auto mmYShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);

    bool isMmXNull = (mmXTensorShape == nullptr);
    bool isMmWeightNull = (mmWeightTensorShape == nullptr);
    bool isMmYNull = (mmYShape == nullptr);
    if (!isMmYNull) {
        auto mmYDimNum = mmYShape->GetStorageShape().GetDimNum();
        isMmYNull = mmYDimNum == 0;
    }
    bool isAllSame = (isMmXNull == isMmWeightNull) && (isMmWeightNull == isMmYNull);
    OP_TILING_CHECK(!isAllSame,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_,
                            "mmXTensor, mmWeightTensor, mmYTensor",
                            (std::string(isMmXNull ? "null" : "not null") + ", " +
                             std::string(isMmWeightNull ? "null" : "not null") + ", " +
                             std::string(isMmYNull ? "null" : "not null")).c_str(),
                            "The values of mmXTensor, mmWeightTensor and mmYTensor must be the same in existence state"),
                    return ge::GRAPH_FAILED);
    if (!isMmXNull) {
        localParams_.hasSharedMm = true;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckOpInputSingleParamsTensor()
{
    auto status = CheckOpInputSingleParamsTensorNotSupport();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckOpInputSingleParamsTensorSupport();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckOpInputSingleParamsTensorMM();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckAndSetLocalParamsGmm()
{
    localParams_.gmmXDtype = context_->GetInputDesc(GMM_X_INDEX)->GetDataType();
    localParams_.gmmWeightDtype = context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_X_DTYPE_LIST, localParams_.gmmXDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmX",
                                              Ops::Base::ToString(localParams_.gmmXDtype).c_str(),
                                              "(DT_HIFLOAT8)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.gmmWeightDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmWeight",
                                              Ops::Base::ToString(localParams_.gmmWeightDtype).c_str(),
                                              "(DT_HIFLOAT8)"),
                    return ge::GRAPH_FAILED);
    localParams_.yDtype = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_Y_DTYPE_LIST, localParams_.yDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "y",
                                              Ops::Base::ToString(localParams_.yDtype).c_str(),
                                              "(DT_FLOAT16, DT_BF16)"),
                    return ge::GRAPH_FAILED);
    localParams_.gmmYDtype = localParams_.yDtype;

    const gert::StorageShape *gmmXStorageShape = context_->GetInputShape(GMM_X_INDEX);
    const gert::StorageShape *gmmWeightStorageShape = context_->GetInputShape(GMM_WEIGHT_INDEX);
    const gert::StorageShape *yStorageShape = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(gmmXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmX"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeight"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "y"), return ge::GRAPH_FAILED);
    auto status = CheckShapeDimensions(gmmXStorageShape, DIM_TWO, "gmmXShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(gmmWeightStorageShape, DIM_THREE, "gmmWeightShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(yStorageShape, DIM_TWO, "yShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    localParams_.A = gmmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H1 = gmmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.ep = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.gmmWeightDim1 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);
    localParams_.gmmWeightDim2 = gmmWeightStorageShape->GetStorageShape().GetDim(DIM_TWO);

    localParams_.BsK = yStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.N1 = yStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckAndSetLocalParamsMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    localParams_.mmXDtype = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX)->GetDataType();
    localParams_.mmWeightDtype = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_X_DTYPE_LIST, localParams_.mmXDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmX",
                                              Ops::Base::ToString(localParams_.mmXDtype).c_str(),
                                              "(DT_HIFLOAT8)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(QUANT_GMM_WEIGHT_DTYPE_LIST, localParams_.mmWeightDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmWeight",
                                              Ops::Base::ToString(localParams_.mmWeightDtype).c_str(),
                                              "(DT_HIFLOAT8)"),
                    return ge::GRAPH_FAILED);
    localParams_.mmYDtype = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_Y_DTYPE_LIST, localParams_.mmYDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmY",
                                              Ops::Base::ToString(localParams_.mmYDtype).c_str(),
                                              "(DT_FLOAT16, DT_BF16)"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *mmXStorageShape = context_->GetOptionalInputShape(MM_X_OPTIONAL_INDEX);
    const gert::StorageShape *mmWeightStorageShape = context_->GetOptionalInputShape(MM_WEIGHT_OPTIONAL_INDEX);
    const gert::StorageShape *mmYStorageShape = context_->GetOutputShape(OUTPUT_MM_Y_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmX"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeight"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmYStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmY"),
                    return ge::GRAPH_FAILED);
    auto status = CheckShapeDimensions(mmXStorageShape, DIM_TWO, "mmXShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(mmWeightStorageShape, DIM_TWO, "mmWeightShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckShapeDimensions(mmYStorageShape, DIM_TWO, "mmYShape");
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    localParams_.Bs = mmXStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.H2 = mmXStorageShape->GetStorageShape().GetDim(DIM_ONE);

    localParams_.mmWeightDim0 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    localParams_.mmWeightDim1 = mmWeightStorageShape->GetStorageShape().GetDim(DIM_ONE);

    uint64_t mmYDim0 = mmYStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(localParams_.Bs != mmYDim0,
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmX, mmY",
                                              (std::string("mmX.DIM0=") + std::to_string(localParams_.Bs) +
                                               ", mmY.DIM0=" + std::to_string(mmYDim0)).c_str(),
                                              "mmX DIM0 and mmY DIM0 should be equal"),
                    return ge::GRAPH_FAILED);

    localParams_.N2 = mmYStorageShape->GetStorageShape().GetDim(DIM_ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckAndSetLocalParamsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "attrs"), return ge::GRAPH_FAILED);

    auto gmmXQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_X_QUANT_MODE_INDEX);
    OP_TILING_CHECK(gmmXQuantModeptr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "gmmXQuantMode", "null", "not null"),
                    return ge::GRAPH_FAILED);
    localParams_.gmmXQuantMode = *gmmXQuantModeptr;
    auto gmmWeightQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_GMM_WEIGHT_QUANT_MODE_INDEX);
    OP_TILING_CHECK(gmmWeightQuantModeptr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "gmmWeightQuantMode", "null", "not null"),
                    return ge::GRAPH_FAILED);
    localParams_.gmmWeightQuantMode = *gmmWeightQuantModeptr;
    auto gmmTransWeightptr = attrs->GetAttrPointer<bool>(ATTR_TRANS_GMM_WEIGHT_INDEX);
    OP_TILING_CHECK(gmmTransWeightptr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "transGmmWeight", "null", "not null"),
                    return ge::GRAPH_FAILED);
    localParams_.isGmmWeightTrans = *gmmTransWeightptr;

    auto commQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_MODE_INDEX);
    OP_TILING_CHECK(commQuantModeptr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "commQuantMode", "null", "not null"),
                    return ge::GRAPH_FAILED);
    localParams_.commQuantMode = *commQuantModeptr;
    auto commQuantDtypeptr = attrs->GetAttrPointer<int64_t>(ATTR_COMM_QUANT_DTYPE_INDEX);
    OP_TILING_CHECK(commQuantDtypeptr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "commQuantDtype", "null", "not null"),
                    return ge::GRAPH_FAILED);
    localParams_.commQuantDtype = *commQuantDtypeptr;
    OP_TILING_CHECK(localParams_.commQuantMode != QUANT_NONE,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "commQuantMode",
                        std::to_string(localParams_.commQuantMode).c_str(), "commQuant is not supported yet; only 0 is allowed"),
                    return ge::GRAPH_FAILED);

    auto mmXQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_MM_X_QUANT_MODE_INDEX);
    OP_TILING_CHECK(mmXQuantModeptr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "mmXQuantMode", "null", "not null"),
                    return ge::GRAPH_FAILED);
    localParams_.mmXQuantMode = *mmXQuantModeptr;
    auto mmWeightQuantModeptr = attrs->GetAttrPointer<int64_t>(ATTR_MM_WEIGHT_QUANT_MODE_INDEX);
    OP_TILING_CHECK(mmWeightQuantModeptr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "mmWeightQuantMode", "null", "not null"),
                    return ge::GRAPH_FAILED);
    localParams_.mmWeightQuantMode = *mmWeightQuantModeptr;
    auto mmTransWeightptr = attrs->GetAttrPointer<bool>(ATTR_TRANS_MM_WEIGHT_INDEX);
    OP_TILING_CHECK(mmTransWeightptr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "transMmWeight", "null", "not null"),
                    return ge::GRAPH_FAILED);
    localParams_.isMmWeightTrans = *mmTransWeightptr;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckFormat()
{
    OP_LOGD(opName_, "start CheckFormat.");
    OP_TILING_CHECK(context_->GetInputDesc(GMM_X_INDEX)->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName_, "gmmX",
                                               Ops::Base::ToString(context_->GetInputDesc(GMM_X_INDEX)->GetStorageFormat()).c_str(),
                                               "FORMAT_ND"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName_, "gmmWeight",
                                               Ops::Base::ToString(context_->GetInputDesc(GMM_WEIGHT_INDEX)->GetStorageFormat()).c_str(),
                                               "FORMAT_ND"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX)->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName_, "gmmXScale",
                                               Ops::Base::ToString(context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX)->GetStorageFormat()).c_str(),
                                               "FORMAT_ND"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX)->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName_, "gmmWeightScale",
                                               Ops::Base::ToString(context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX)->GetStorageFormat()).c_str(),
                                               "FORMAT_ND"), return ge::GRAPH_FAILED);
    auto yDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "y"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(opName_, "y",
                                               Ops::Base::ToString(yDesc->GetStorageFormat()).c_str(),
                                               "FORMAT_ND"), return ge::GRAPH_FAILED);
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    // 即使传入nullptr，GetOptionalInputDesc接口也有可能拿到非nullptr的地址？？？
    auto mmXDesc = context_->GetOptionalInputDesc(MM_X_OPTIONAL_INDEX);
    if (mmXDesc != nullptr) {
        OP_TILING_CHECK(mmXDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(opName_, "mmX",
                                                   Ops::Base::ToString(mmXDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_ND"), return ge::GRAPH_FAILED);
    }
    auto mmWeightDesc = context_->GetOptionalInputDesc(MM_WEIGHT_OPTIONAL_INDEX);
    if (mmWeightDesc != nullptr) {
        OP_TILING_CHECK(mmWeightDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(opName_, "mmWeight",
                                                   Ops::Base::ToString(mmWeightDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_ND"), return ge::GRAPH_FAILED);
    }
    auto mmXScaleDesc = context_->GetOptionalInputDesc(MM_X_SCALE_OPTIONAL_INDEX);
    if (mmXScaleDesc != nullptr) {
        OP_TILING_CHECK(mmXScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(opName_, "mmXScale",
                                                   Ops::Base::ToString(mmXScaleDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_ND"), return ge::GRAPH_FAILED);
    }
    auto mmWeightScaleDesc = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    if (mmWeightScaleDesc != nullptr) {
        OP_TILING_CHECK(mmWeightScaleDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(opName_, "mmWeightScale",
                                                   Ops::Base::ToString(mmWeightScaleDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_ND"), return ge::GRAPH_FAILED);
    }
    auto mmYDesc = context_->GetOutputDesc(OUTPUT_MM_Y_OPTIONAL_INDEX);
    if (mmYDesc != nullptr) {
        OP_TILING_CHECK(mmYDesc->GetStorageFormat() != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(opName_, "mmY",
                                                   Ops::Base::ToString(mmYDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_ND"), return ge::GRAPH_FAILED);
    }
    OP_LOGD(opName_, "end CheckFormat.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckAndSetLocalParams()
{
    auto status = CheckAndSetLocalParamsGmm();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckAndSetLocalParamsMm();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckAndSetLocalParamsAttr();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckFormat();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckParamsRelationGmmTransShape()
{
    if (localParams_.isGmmWeightTrans) {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim2,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmWeight",
                                                  (std::string("dim2=") + std::to_string(localParams_.gmmWeightDim2)).c_str(),
                                                  (std::string("gmmX.dim1=") + std::to_string(localParams_.H1)).c_str()),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmWeight",
                                                  (std::string("dim1=") + std::to_string(localParams_.gmmWeightDim1)).c_str(),
                                                  (std::string("y.dim1=") + std::to_string(localParams_.N1)).c_str()),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H1 != localParams_.gmmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmWeight",
                                                  (std::string("dim1=") + std::to_string(localParams_.gmmWeightDim1)).c_str(),
                                                  (std::string("gmmX.dim1=") + std::to_string(localParams_.H1)).c_str()),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N1 != localParams_.gmmWeightDim2,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmWeight",
                                                  (std::string("dim2=") + std::to_string(localParams_.gmmWeightDim2)).c_str(),
                                                  (std::string("y.dim1=") + std::to_string(localParams_.N1)).c_str()),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckParamsRelationGmm()
{
    localParams_.gmmXScaleDtype = context_->GetRequiredInputDesc(GMM_X_SCALE_INDEX)->GetDataType();
    localParams_.gmmWeightScaleDtype = context_->GetRequiredInputDesc(GMM_WEIGHT_SCALE_INDEX)->GetDataType();
    OP_TILING_CHECK(!IsContains(QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.gmmXScaleDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmXScale",
                                              Ops::Base::ToString(localParams_.gmmXScaleDtype).c_str(),
                                              "(DT_FLOAT)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.gmmWeightScaleDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "gmmWeightScale",
                                              Ops::Base::ToString(localParams_.gmmWeightScaleDtype).c_str(),
                                              "(DT_FLOAT)"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *gmmXScaleStorageShape = context_->GetRequiredInputShape(GMM_X_SCALE_INDEX);
    const gert::StorageShape *gmmWeightScaleStorageShape = context_->GetRequiredInputShape(GMM_WEIGHT_SCALE_INDEX);
    OP_TILING_CHECK(gmmXScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmXScale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gmmWeightScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "gmmWeightScale"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        localParams_.gmmXQuantMode != QUANT_PERTENSOR,
        OP_LOGE_WITH_INVALID_ATTR(opName_, "gmmXQuantMode",
                                  std::to_string(localParams_.gmmXQuantMode),
                                  std::to_string(QUANT_PERTENSOR) + " (PERTENSOR)"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localParams_.gmmWeightQuantMode != QUANT_PERTENSOR,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "gmmWeightQuantMode",
                                              std::to_string(localParams_.gmmWeightQuantMode),
                                              std::to_string(QUANT_PERTENSOR) + " (PERTENSOR)"),
                    return ge::GRAPH_FAILED);
    localParams_.gmmQuantSuit = QUANT_PAIR_TT;

    ge::graphStatus status = CheckShapeDimensions(gmmXScaleStorageShape, DIM_ONE, "gmmXScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(gmmWeightScaleStorageShape, DIM_ONE, "gmmWeightScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    int64_t gmmXScaleDim0 = gmmXScaleStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(gmmXScaleDim0 != 1,
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmXScale",
                                              "[" + std::to_string(gmmXScaleDim0) + "]",
                                              "[1]"),
                    return ge::GRAPH_FAILED);
    int64_t gmmWeightScaleDim0 = gmmWeightScaleStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(gmmWeightScaleDim0 != 1,
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "gmmWeightScale",
                                              "[" + std::to_string(gmmWeightScaleDim0) + "]",
                                              "[1]"),
                    return ge::GRAPH_FAILED);

    status = CheckParamsRelationGmmTransShape();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckParamsRelationMmTransShape()
{
    if (localParams_.isMmWeightTrans) {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmWeight",
                                                  (std::string("dim1=") + std::to_string(localParams_.mmWeightDim1)).c_str(),
                                                  (std::string("mmX.dim1=") + std::to_string(localParams_.H2)).c_str()),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim0,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmWeight",
                                                  (std::string("dim0=") + std::to_string(localParams_.mmWeightDim0)).c_str(),
                                                  (std::string("mmY.dim1=") + std::to_string(localParams_.N2)).c_str()),
                        return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(localParams_.H2 != localParams_.mmWeightDim0,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmWeight",
                                                  (std::string("dim0=") + std::to_string(localParams_.mmWeightDim0)).c_str(),
                                                  (std::string("mmX.dim1=") + std::to_string(localParams_.H2)).c_str()),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(localParams_.N2 != localParams_.mmWeightDim1,
                        OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmWeight",
                                                  (std::string("dim1=") + std::to_string(localParams_.mmWeightDim1)).c_str(),
                                                  (std::string("mmY.dim1=") + std::to_string(localParams_.N2)).c_str()),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckParamsRelationMm()
{
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    localParams_.mmXScaleDtype = context_->GetOptionalInputDesc(MM_X_SCALE_OPTIONAL_INDEX)->GetDataType();
    localParams_.mmWeightScaleDtype = context_->GetOptionalInputDesc(MM_WEIGHT_SCALE_OPTIONAL_INDEX)->GetDataType();

    OP_TILING_CHECK(!IsContains(QUANT_GMM_X_SCALE_DTYPE_LIST, localParams_.mmXScaleDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmXScale",
                                              Ops::Base::ToString(localParams_.mmXScaleDtype).c_str(),
                                              "(DT_FLOAT)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!IsContains(QUANT_GMM_WEIGHT_SCALE_DTYPE_LIST, localParams_.mmWeightScaleDtype),
                    OP_LOGE_FOR_INVALID_DTYPE(opName_, "mmWeightScale",
                                              Ops::Base::ToString(localParams_.mmWeightScaleDtype).c_str(),
                                              "(DT_FLOAT)"),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *mmXScaleStorageShape = context_->GetOptionalInputShape(MM_X_SCALE_OPTIONAL_INDEX);
    const gert::StorageShape *mmWeightScaleStorageShape =
        context_->GetOptionalInputShape(MM_WEIGHT_SCALE_OPTIONAL_INDEX);
    OP_TILING_CHECK(mmXScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmXScale"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mmWeightScaleStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "mmWeightScale"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        localParams_.mmXQuantMode != QUANT_PERTENSOR,
        OP_LOGE_WITH_INVALID_ATTR(opName_, "mmXQuantMode",
                                  std::to_string(localParams_.mmXQuantMode),
                                  std::to_string(QUANT_PERTENSOR) + " (PERTENSOR)"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(localParams_.mmWeightQuantMode != QUANT_PERTENSOR,
                    OP_LOGE_WITH_INVALID_ATTR(opName_, "mmWeightQuantMode",
                                              std::to_string(localParams_.mmWeightQuantMode),
                                              std::to_string(QUANT_PERTENSOR) + " (PERTENSOR)"),
                    return ge::GRAPH_FAILED);

    localParams_.mmQuantSuit = QUANT_PAIR_TT;
    ge::graphStatus status = CheckShapeDimensions(mmXScaleStorageShape, DIM_ONE, "mmXScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);
    status = CheckShapeDimensions(mmWeightScaleStorageShape, DIM_ONE, "mmWeightScaleShape");
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    int64_t mmXScaleDim0 = mmXScaleStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(mmXScaleDim0 != 1,
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmXScale",
                                              "[" + std::to_string(mmXScaleDim0) + "]",
                                              "[1]"),
                    return ge::GRAPH_FAILED);
    int64_t mmWeightScaleDim0 = mmWeightScaleStorageShape->GetStorageShape().GetDim(DIM_ZERO);
    OP_TILING_CHECK(mmWeightScaleDim0 != 1,
                    OP_LOGE_FOR_INVALID_SHAPE(opName_, "mmWeightScale",
                                              "[" + std::to_string(mmWeightScaleDim0) + "]",
                                              "[1]"),
                    return ge::GRAPH_FAILED);

    status = CheckParamsRelationMmTransShape();
    OP_TILING_CHECK(status != ge::GRAPH_SUCCESS, "", return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckParamsAttrEpAndSetLocalParams()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    const char *group = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(group == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "group", "null", "not null"),
                    return ge::GRAPH_FAILED);
    int64_t rankDim = 0;
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "epWorldSize", "null", "not null"),
                    return ge::GRAPH_FAILED);
    if (*epWorldSizePtr == RANK_DEFAULT_NUM) {
        OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group, rankDim), OP_LOGE(opName_, "GetRankSize failed."),
                        return ge::GRAPH_FAILED);
    } else {
        rankDim = *epWorldSizePtr;
    }
    std::string supportRankSizeRange;
    for (const auto &v : SUPPORT_RANK_SIZE) {
        supportRankSizeRange += (std::to_string(v) + " ");
    }
    OP_TILING_CHECK(
        SUPPORT_RANK_SIZE.find(rankDim) == SUPPORT_RANK_SIZE.end(),
        OP_LOGE_WITH_INVALID_ATTR(opName_, "epWorldSize",
                                  std::to_string(rankDim), supportRankSizeRange.c_str()),
        return ge::GRAPH_FAILED);
    localParams_.epWorldSize = rankDim;

    bool isEpNumMatch = (localParams_.ep > 0) && (localParams_.ep < 33);
    if (isEpNumMatch) {
        uint64_t expertNum = localParams_.ep * rankDim;
        OP_TILING_CHECK(
            expertNum > MAX_EXPERT_NUM,
            OP_LOGE_FOR_INVALID_VALUE(opName_, "expertNum",
                                      std::to_string(expertNum),
                                      "<= " + std::to_string(MAX_EXPERT_NUM)),
            return ge::GRAPH_FAILED);
    } else {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "ep",
                                  std::to_string(localParams_.ep),
                                  "in range (0, " + std::to_string(MAX_EXPERT_NUM_PER_RANK) + "]");
        return ge::GRAPH_FAILED;
    }

    auto groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_SIZE_OPTIONAL_INDEX);
    OP_TILING_CHECK(groupSizePtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "groupSize", "null", "not null"),
                    return ge::GRAPH_FAILED);
    localParams_.groupSize = *groupSizePtr;
    OP_TILING_CHECK(
        localParams_.groupSize != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "groupSize",
                                               std::to_string(localParams_.groupSize),
                                                "The value of groupSize must be 0. Group quant mode is unavailable."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckAndSetSendRecvCountsAttr()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    uint64_t expertNum = localParams_.ep * localParams_.epWorldSize;
    auto sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SEND_COUNTS_INDEX);
    auto recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_RECV_COUNTS_INDEX);
    OP_TILING_CHECK(sendCountsPtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "sendCounts", "null", "not null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsPtr == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName_, "recvCounts", "null", "not null"),
                    return ge::GRAPH_FAILED);
    uint64_t sendCountsSize = sendCountsPtr->GetSize();
    uint64_t recvCountsSize = recvCountsPtr->GetSize();
    OP_TILING_CHECK(sendCountsSize != recvCountsSize,
                    OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "sendCounts, recvCounts",
                                                 std::to_string(sendCountsSize) + " vs " + std::to_string(recvCountsSize),
                                                 "must be equal"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsSize != expertNum,
                    OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "sendCounts",
                                                 std::to_string(sendCountsSize),
                                                 std::to_string(expertNum)),
                    return ge::GRAPH_FAILED);

    const int64_t *sendCounts = static_cast<const int64_t *>(sendCountsPtr->GetData());
    const int64_t *recvCounts = static_cast<const int64_t *>(recvCountsPtr->GetData());
    uint64_t sendCountsSum = std::accumulate(sendCounts, sendCounts + sendCountsSize, 0ULL);
    OP_TILING_CHECK(sendCountsSum != localParams_.A,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "sendCountsSum",
                                              std::to_string(sendCountsSum),
                                              std::to_string(localParams_.A)),
                    return ge::GRAPH_FAILED);
    uint64_t recvCountsSum = std::accumulate(recvCounts, recvCounts + recvCountsSize, 0ULL);
    OP_TILING_CHECK(recvCountsSum != localParams_.BsK,
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "recvCountsSum",
                                              std::to_string(recvCountsSum),
                                              std::to_string(localParams_.BsK)),
                    return ge::GRAPH_FAILED);

    auto gmmQTilingCommonInfoPtr = &localTilingData_.taskTilingInfo;
    uint64_t maxCountsSize = std::min<uint64_t>(expertNum, MAX_EXPERT_NUM);
    for (uint64_t i = 0; i < maxCountsSize; i++) {
        OP_TILING_CHECK(
            sendCounts[i] < 0 || sendCounts[i] > static_cast<int64_t>(localParams_.A),
            OP_LOGE_FOR_INVALID_VALUE(opName_,
                                      (std::string("sendCounts[") + std::to_string(i) + "]").c_str(),
                                      std::to_string(sendCounts[i]),
                                      "[0, " + std::to_string(localParams_.A) + "]"),
            return ge::GRAPH_FAILED);
        OP_TILING_CHECK(
            recvCounts[i] < 0 || recvCounts[i] > static_cast<int64_t>(localParams_.BsK),
            OP_LOGE_FOR_INVALID_VALUE(opName_,
                                      (std::string("recvCounts[") + std::to_string(i) + "]").c_str(),
                                      std::to_string(recvCounts[i]),
                                      "[0, " + std::to_string(localParams_.BsK) + "]"),
            return ge::GRAPH_FAILED);
        gmmQTilingCommonInfoPtr->sendCnt[i] = static_cast<int32_t>(sendCounts[i]);
        gmmQTilingCommonInfoPtr->recvCnt[i] = static_cast<int32_t>(recvCounts[i]);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckLocalParams()
{
    OP_TILING_CHECK((localParams_.H1 == 0) || (localParams_.H1 >= MAX_H1_VALUE),
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "H1",
                                              std::to_string(localParams_.H1),
                                              "(0, " + std::to_string(MAX_H1_VALUE) + ")"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.BsK == 0) || (localParams_.BsK >= MAX_BSK_VALUE),
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "BsK",
                                              std::to_string(localParams_.BsK),
                                              "(0, " + std::to_string(MAX_BSK_VALUE) + ")"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.N1 == 0) || (localParams_.N1 >= MAX_N1_VALUE),
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "N1",
                                              std::to_string(localParams_.N1),
                                              "(0, " + std::to_string(MAX_N1_VALUE) + ")"),
                    return ge::GRAPH_FAILED);

    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK((localParams_.Bs == 0) || (localParams_.BsK % localParams_.Bs != 0),
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "BSK mod BS",
                                              "BSK=" + std::to_string(localParams_.BsK) + " BS=" + std::to_string(localParams_.Bs),
                                              "BSK should be an integer multiple of BS"),
                    return ge::GRAPH_FAILED);
    uint64_t k = localParams_.BsK / localParams_.Bs;
    OP_TILING_CHECK((k < MIN_K_VALUE) || (k > MAX_K_VALUE),
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "K",
                                              std::to_string(k),
                                              "[" + std::to_string(MIN_K_VALUE) + ", " + std::to_string(MAX_K_VALUE) + "]"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (localParams_.H2 == 0) || (localParams_.H2 > MAX_SHARED_H_SHAPE_SIZE),
        OP_LOGE_FOR_INVALID_VALUE(opName_, "H2",
                                  std::to_string(localParams_.H2),
                                  "(0, " + std::to_string(MAX_SHARED_H_SHAPE_SIZE) + "]"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((localParams_.N2 == 0) || (localParams_.N2 >= MAX_N2_VALUE),
                    OP_LOGE_FOR_INVALID_VALUE(opName_, "N2",
                                              std::to_string(localParams_.N2),
                                              "(0, " + std::to_string(MAX_N2_VALUE) + ")"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckParamsRelationAndSetLocalParams()
{
    auto status = CheckParamsRelationGmm();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckParamsRelationMm();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckParamsAttrEpAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckAndSetSendRecvCountsAttr();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    status = CheckLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(opName_, "Failed to get platform info."), return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    libApiWorkSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    localParams_.aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    localParams_.aicCoreNum = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::CheckAndSetInputOutputInfo()
{
    auto status = CheckOpInputSingleParamsTensor();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    status = CheckParamsRelationAndSetLocalParams();
    if (status != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::SetTilingCommonInfo()
{
    auto gmmQTilingCommonInfoPtr = &localTilingData_.taskTilingInfo;
    gmmQTilingCommonInfoPtr->BSK = localParams_.BsK;
    gmmQTilingCommonInfoPtr->BS = localParams_.Bs;
    gmmQTilingCommonInfoPtr->H1 = localParams_.H1;
    gmmQTilingCommonInfoPtr->H2 = localParams_.H2;
    gmmQTilingCommonInfoPtr->A = localParams_.A;
    gmmQTilingCommonInfoPtr->N1 = localParams_.N1;
    gmmQTilingCommonInfoPtr->N2 = localParams_.N2;
    gmmQTilingCommonInfoPtr->epWorldSize = localParams_.epWorldSize;
    gmmQTilingCommonInfoPtr->e = localParams_.ep;
    gmmQTilingCommonInfoPtr->aivCoreNum = static_cast<uint64_t>(localParams_.aivCoreNum);

    // 动态计算 expertNum_，保证不同专家数量下通信和计算次数不超过4次
    // expertNum_ = CeilDiv(ep, 4)，使得 totalLoopCount = CeilDiv(ep, expertNum_) <= 4
    constexpr uint32_t TARGET_COMM_ROUNDS = 4U;
    expertNum_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(localParams_.ep, static_cast<uint64_t>(TARGET_COMM_ROUNDS)));
    if (expertNum_ == 0U) {
        expertNum_ = 1U;
    }
    OP_LOGD(opName_, "SetTilingCommonInfo: ep=%lu, expertNum_=%u, targetCommRounds=%u",
            localParams_.ep, expertNum_, TARGET_COMM_ROUNDS);
    gmmQTilingCommonInfoPtr->expertNum = expertNum_;
    gmmQTilingCommonInfoPtr->mainLoopExpertNum = expertNum_;
    gmmQTilingCommonInfoPtr->tailLoopExpertNum =
        (localParams_.ep % expertNum_ == 0) ? 0 : static_cast<uint32_t>(localParams_.ep % expertNum_);
    gmmQTilingCommonInfoPtr->totalLoopCount =
        static_cast<uint32_t>(Ops::Base::CeilDiv(localParams_.ep, static_cast<uint64_t>(expertNum_)));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::SetGmmA2avWorkspaceInfo()
{
    constexpr uint64_t alignAddrLen = 512;
    constexpr uint64_t tensorListSize = 512;
    auto gmmYDtypeSize = mc2tiling::GetDataTypeSize(opName_, localParams_.gmmYDtype);
    inferredInfo_.gmmResultLen = mc2tiling::AlignUp(localParams_.A * localParams_.N1 * gmmYDtypeSize, alignAddrLen);
    localTilingData_.workspaceInfo.wsGmmOutputSize = inferredInfo_.gmmResultLen;
    // GmmComputeOp workspace 内部布局:
    //   [groupList (ep * epWorldSize * 8B)] [cGroupOffsetTable (ep * epWorldSize * 8B)]
    //   [ptrTable (tensorListSize=512)] [TT scale repeat (sizeof(float) * ep * 2)]
    uint64_t groupListSize = localParams_.ep * localParams_.epWorldSize * sizeof(int64_t);
    uint64_t cGroupOffsetTableSize = localParams_.ep * localParams_.epWorldSize * sizeof(uint64_t);
    uint64_t ttScaleRepeatSize = sizeof(float) * localParams_.ep * 2;
    uint64_t gmmComputeWsSize = groupListSize + cGroupOffsetTableSize + tensorListSize + ttScaleRepeatSize;
    localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize = mc2tiling::AlignUp(gmmComputeWsSize, alignAddrLen);
    uint64_t sharedTtScaleRepeatSize = sizeof(float) * localParams_.ep * 2;
    uint64_t sharedGmmComputeWsSize = localParams_.ep * sizeof(int64_t) + tensorListSize + sharedTtScaleRepeatSize;
    localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize =
        mc2tiling::AlignUp(sharedGmmComputeWsSize, alignAddrLen);

    workSpaceSize_ = libApiWorkSpaceSize_ + inferredInfo_.gmmResultLen +
                     localTilingData_.workspaceInfo.wsGmmComputeWorkspaceSize +
                     localTilingData_.workspaceInfo.wsSharedGmmComputeWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::DoQuantGMMTiling()
{
    // 设置公共信息
    QuantGroupedMatmulAllToAllvAdapter gmmTile(context_);
    MC2_CHECK_LOG_RET(opName_, gmmTile.SetCommonInputParams(localParams_));
    // tokens最多的专家作为MM计算的M
    uint64_t mMaxSize = 0;
    uint64_t mSize = 0;
    for (uint64_t expertIdx = 0; expertIdx < localParams_.ep; expertIdx++) {
        mSize = 0;
        for (uint64_t i = 0; i < localParams_.epWorldSize; i++) {
            mSize += localTilingData_.taskTilingInfo.sendCnt[i * localParams_.ep + expertIdx];
        }
        mMaxSize = std::max(mSize, mMaxSize);
    }
    // check expertNum
    OP_TILING_CHECK(expertNum_ == 0,
        OP_LOGE_FOR_INVALID_VALUE(opName_, "expertNum_",
            std::to_string(expertNum_).c_str(), "positive value"),
        return ge::GRAPH_FAILED);
    expertNum_ = Ops::Base::CeilDiv(localParams_.ep, expertNum_) > MAX_HCCL_HANDLE_NUM ?
        Ops::Base::CeilDiv(localParams_.ep, static_cast<uint64_t>(MAX_HCCL_HANDLE_NUM)) : expertNum_;
    MC2_CHECK_LOG_RET(opName_, gmmTile.SetGroupExpertInputParameters(localParams_, mMaxSize,
        expertNum_));
    MC2_CHECK_LOG_RET(opName_, gmmTile.Process());
    localTilingData_.gmmBaseTiling = gmmTile.GetGmmQuantTilingAdapterData();

    // SharedMM
    if (!localParams_.hasSharedMm) {
        return ge::GRAPH_SUCCESS;
    }
    MC2_CHECK_LOG_RET(opName_, gmmTile.SetSharedExpertInputParameters(localParams_));
    MC2_CHECK_LOG_RET(opName_, gmmTile.Process());
    localTilingData_.sharedGmmTiling = gmmTile.GetGmmQuantTilingAdapterData();

    return ge::GRAPH_SUCCESS;
}

uint32_t QuantGroupedMatmulAllToAllvTilingCommon::GetCommModeIndex() const
{
    return GmmA2AvAttrIndex::ATTR_COMM_MODE_INDEX;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::QuantGetAndConvertCommMode(gert::TilingContext *context,
    uint8_t &commMode) const
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(context->GetNodeName(), "Failed to get attrs."), return ge::GRAPH_FAILED);
    const char *commModeStr = attrs->GetAttrPointer<char>(GetCommModeIndex());
    OP_TILING_CHECK(commModeStr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "comm_mode"), return ge::GRAPH_FAILED);
    const size_t maxLength = 7UL;
    if (strncmp(commModeStr, "ai_cpu", maxLength) == 0) {
        commMode = Mc2Comm::COMM_MODE_AICPU;
    } else if (strncmp(commModeStr, "ccu", maxLength) == 0) {
        commMode = Mc2Comm::COMM_MODE_CCU;
    } else {
        OP_LOGD(context->GetNodeName(),
                "Currently, commMode only support 'ccu', 'ai_cpu', but got %s.", commModeStr);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::SetHcclTiling()
{
    uint32_t alltoAllvCmd = 8U;
    std::string alltoAllvConfig = "AlltoAll=level0:fullmesh;level1:pairwise";

    auto attrs = context_->GetAttrs();
    auto groupEpPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);

    const uint32_t alltoAllvReduceType = 0u;
    auto outputDataType = context_->GetOutputDesc(OUTPUT_Y_INDEX)->GetDataType();
    OP_TILING_CHECK(
        mc2tiling::HCCL_DATA_TYPE.find(outputDataType) == mc2tiling::HCCL_DATA_TYPE.end(),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "outputDataType",
            Ops::Base::ToString(outputDataType).c_str(), "The dtype of outputDataType must be within the supported range."),
        return ge::GRAPH_FAILED);

    auto alltoAllvDstDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);
    auto alltoAllvSrcDataType = static_cast<uint8_t>(mc2tiling::HCCL_DATA_TYPE.find(outputDataType)->second);

    Mc2CcTilingConfig hcclCcTilingConfig(groupEpPtr, alltoAllvCmd, alltoAllvConfig, alltoAllvReduceType,
                                         alltoAllvDstDataType, alltoAllvSrcDataType);
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint8_t commMode = 0;
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        if (QuantGetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        commMode = Mc2Comm::COMM_MODE_AICPU;
    }
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_AICPU);
    } else if (commMode == Mc2Comm::COMM_MODE_CCU) {
        hcclCcTilingConfig.SetCommEngine(Mc2Comm::ENGINE_CCU_SCHED);
    }
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_.hcclA2avTiling.hcclInitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling hcclInitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclCcTilingConfig.GetTiling(localTilingData_.hcclA2avTiling.a2avCcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling alltoAllvCcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::DoOpTiling()
{
    // 输入参数的校验:Attrs,Dtype,Shape等
    MC2_CHECK_LOG_RET(opName_, CheckAndSetInputOutputInfo());

    MC2_CHECK_LOG_RET(opName_, SetTilingCommonInfo());
    // 调用量化Matmul的tiling方法进行切分
    MC2_CHECK_LOG_RET(opName_, DoQuantGMMTiling());
    // hccl的tiling参数赋值处理
    MC2_CHECK_LOG_RET(opName_, SetHcclTiling());
    MC2_CHECK_LOG_RET(opName_, SetGmmA2avWorkspaceInfo());
    return ge::GRAPH_SUCCESS;
}

void PrintGmmA2avWorkspaceInfo(const GmmA2avWorkspaceInfo &workspaceInfo, const char *opName_)
{
    std::stringstream ss;
    ss << "workspaceInfo: ";
    ss << "wsGmmOutputSize=" << workspaceInfo.wsGmmOutputSize
       << ", wsGmmComputeWorkspaceSize=" << workspaceInfo.wsGmmComputeWorkspaceSize
       << ", wsSharedGmmComputeWorkspaceSize=" << workspaceInfo.wsSharedGmmComputeWorkspaceSize;
    OP_LOGD(opName_, "%s", ss.str().c_str());
}

void PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo &taskTilingInfo,
                         const QuantGmmAlltoAllvParamsInfo &localParams, const char *opName_)
{
    std::stringstream ss;
    ss << "TaskTilingInfo: ";
    ss << "BSK=" << taskTilingInfo.BSK << ", BS=" << taskTilingInfo.BS << ", H1=" << taskTilingInfo.H1
       << ", H2=" << taskTilingInfo.H2 << ", A=" << taskTilingInfo.A << ", N1=" << taskTilingInfo.N1
       << ", N2=" << taskTilingInfo.N2;
    ss << ", epWorldSize=" << taskTilingInfo.epWorldSize << ", e=" << taskTilingInfo.e;
    ss << ", mainLoopExpertNum=" << taskTilingInfo.mainLoopExpertNum
       << ", tailLoopExpertNum=" << taskTilingInfo.tailLoopExpertNum
       << ", totalLoopCount=" << taskTilingInfo.totalLoopCount;
    ss << "\nSendCounts: ";
    for (int64_t i = 0; i < localParams.ep * localParams.epWorldSize; i++) {
        if (taskTilingInfo.sendCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.sendCnt[i];
        }
    }
    ss << "\nRecvCounts: ";
    for (int64_t i = 0; i < localParams.ep * localParams.epWorldSize; i++) {
        if (taskTilingInfo.recvCnt[i] != 0) {
            if (i != 0) {
                ss << " ,";
            }
            ss << taskTilingInfo.recvCnt[i];
        }
    }
    OP_LOGI(opName_, "%s", ss.str().c_str());
}

void PrintGMMQuantTilingData(const MC2KernelTemplate::GMMQuantTilingData &data, const char *opName_)
{
    const auto &mm = data.mmTilingData;
    const auto &quantParams = data.gmmQuantParams;
    const auto &gmmArray = data.gmmArray;

    std::stringstream ss;
    ss << "MM Tiling: M=" << mm.M << ", N=" << mm.N << ", K=" << mm.Ka << ", usedCoreNum=" << mm.usedCoreNum
       << ", baseM=" << mm.baseM << ", baseN=" << mm.baseN << ", baseK=" << mm.baseK
       << ", singleCoreM=" << mm.singleCoreM << ", singleCoreN=" << mm.singleCoreN << ", singleCoreK=" << mm.singleCoreK
       << ", dbL0C=" << mm.dbL0C << ", depthA1=" << mm.depthA1 << ", depthB1=" << mm.depthB1 << ", stepKa=" << mm.stepKa
       << ", stepKb=" << mm.stepKb << ", stepM=" << mm.stepM << ", stepN=" << mm.stepN
       << ", iterateOrder=" << mm.iterateOrder;

    ss << "\nQuant Params: groupNum=" << quantParams.groupNum << ", activeType=" << quantParams.activeType
       << ", aQuantMode=" << quantParams.aQuantMode << ", bQuantMode=" << quantParams.bQuantMode
       << ", singleX=" << static_cast<int32_t>(quantParams.singleX)
       << ", singleW=" << static_cast<int32_t>(quantParams.singleW)
       << ", singleY=" << static_cast<int32_t>(quantParams.singleY)
       << ", groupType=" << static_cast<int32_t>(quantParams.groupType)
       << ", groupListType=" << static_cast<uint32_t>(quantParams.groupListType)
       << ", hasBias=" << static_cast<int32_t>(quantParams.hasBias) << ", reserved=" << quantParams.reserved;

    ss << "\nArray: mList[0]=" << gmmArray.mList[0] << ", kList[0]=" << gmmArray.kList[0]
       << ", nList[0]=" << gmmArray.nList[0];

    OP_LOGI(opName_, "QuantGmmA2AvTiling TilingParams:\n%s", ss.str().c_str());
}

void QuantGroupedMatmulAllToAllvTilingCommon::PrintQuantGmmA2avTilingData(QuantGmmA2avTilingData &outTilingData)
{
    PrintGmmA2avWorkspaceInfo(outTilingData.workspaceInfo, opName_);
    PrintTaskTilingInfo(outTilingData.taskTilingInfo, localParams_, opName_);
    OP_LOGD(opName_, "------------- PrintGMMQuantTilingData -------------------");
    PrintGMMQuantTilingData(outTilingData.gmmBaseTiling, opName_);
    OP_LOGD(opName_, "------------- PrintGMMSharedQuantTilingData -------------");
    PrintGMMQuantTilingData(outTilingData.sharedGmmTiling, opName_);
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::PostTiling()
{
    PrintQuantGmmA2avTilingData(localTilingData_);
    context_->SetBlockDim(localParams_.aicCoreNum);
    QuantGmmA2avTilingData *outTilingData = context_->GetTilingData<QuantGmmA2avTilingData>();
    size_t tilingBufCap = context_->GetRawTilingData()->GetCapacity();
    OP_TILING_CHECK((outTilingData == nullptr), OP_LOGE(opName_, "Failed to get tiling data from context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (tilingBufCap < sizeof(localTilingData_)),
        OP_LOGE(opName_, "TilingBuffer too small, capacity = %zu, need = %zu.", tilingBufCap, sizeof(localTilingData_)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sizeof(localTilingData_) % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "Tiling data size[%zu] is not aligned to 8", sizeof(localTilingData_)),
                    return ge::GRAPH_FAILED);
    errno_t ret =
        memcpy_s(outTilingData, tilingBufCap, reinterpret_cast<void *>(&localTilingData_), sizeof(localTilingData_));
    if (ret != EOK) {
        OP_LOGE(opName_, "postTiling: memcpy_s failed with ret=%d.", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(sizeof(localTilingData_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantGroupedMatmulAllToAllvTilingCommon::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "Failed to get workspace."), return ge::GRAPH_FAILED);
    workspaces[0] = workSpaceSize_;
    OP_LOGD(opName_, "Workspaces[0] size=%ld", workspaces[0]);

    return ge::GRAPH_SUCCESS;
}

uint64_t QuantGroupedMatmulAllToAllvTilingCommon::GetTilingKey() const
{
    uint8_t commMode = 0;
    if (QuantGetAndConvertCommMode(context_, commMode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(localParams_.hasSharedMm, localParams_.isGmmWeightTrans,
                                                  localParams_.isMmWeightTrans, commMode);
    OP_LOGD(opName_, "GET_TPL_TILING_KEY: [%d,%d,%d,%d], TilingKey is [%lu].", localParams_.hasSharedMm,
            localParams_.isGmmWeightTrans, localParams_.isMmWeightTrans, commMode, tilingKey);
    return tilingKey;
}

} // namespace Mc2Tiling
