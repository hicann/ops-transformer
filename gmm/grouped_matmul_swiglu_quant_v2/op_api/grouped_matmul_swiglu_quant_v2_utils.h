/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V2_UTILS_H
#define OP_HOST_OP_API_GROUPED_MATMUL_SWIGLU_QUANT_V2_UTILS_H

#include "grouped_matmul_swiglu_quant_utils.h"
#include "util/math_util.h"
#include "log/log.h"

#include <sstream>
#include <vector>

namespace gmmSwigluQuantV2 {

using namespace gmm_dsq;

inline std::string ShapeToString(const op::Shape &shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape.GetDim(i);
    }
    oss << "]";
    return oss.str();
}

inline std::string ViewShapeToString(const aclTensor *tensor)
{
    return ShapeToString(tensor->GetViewShape());
}

inline std::string SupportListToString(const std::vector<DataType> &supportList)
{
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < supportList.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << op::ToString(supportList[i]).GetString();
    }
    oss << "}";
    return oss.str();
}

#define GMM_SWIGLU_CHECK_SHAPE(tensor, paramName, expectedShape, retExpr) \
    do { \
        if ((tensor)->GetViewShape() != (expectedShape)) { \
            OP_LOGE_FOR_INVALID_SHAPE(apiName_.c_str(), paramName, op::ToString((tensor)->GetViewShape()).GetString(), \
                                      op::ToString(expectedShape).GetString()); \
            retExpr; \
        } \
    } while (0)

#define GMM_SWIGLU_CHECK_DTYPE(tensor, paramName, supportList, retExpr) \
    do { \
        if (std::find((supportList).begin(), (supportList).end(), (tensor)->GetDataType()) == (supportList).end()) { \
            OP_LOGE_FOR_INVALID_DTYPE(apiName_.c_str(), paramName, op::ToString((tensor)->GetDataType()).GetString(), \
                                      SupportListToString(supportList).c_str()); \
            retExpr; \
        } \
    } while (0)

#define GMM_SWIGLU_CHECK_DIM(actualDim, expectedDim, paramName, retExpr) \
    do { \
        if ((actualDim) != (expectedDim)) { \
            OP_LOGE_FOR_INVALID_SHAPEDIM(apiName_.c_str(), paramName, std::to_string(actualDim), \
                                         std::to_string(expectedDim)); \
            retExpr; \
        } \
    } while (0)

#define GMM_SWIGLU_CHECK_TENSORNUM(actualNum, expectedNum, paramName, retExpr) \
    do { \
        if ((actualNum) != (expectedNum)) { \
            OP_LOGE_FOR_INVALID_TENSORNUM(apiName_.c_str(), paramName, actualNum, std::to_string(expectedNum)); \
            retExpr; \
        } \
    } while (0)

#define GMM_SWIGLU_CHECK_FORMAT(tensor, paramName, correctFormat, retExpr) \
    do { \
        if (op::IsPrivateFormat((tensor)->GetStorageFormat())) { \
            OP_LOGE_FOR_INVALID_FORMAT(apiName_.c_str(), paramName, \
                                       op::ToString((tensor)->GetStorageFormat()).GetString(), correctFormat); \
            retExpr; \
        } \
    } while (0)

constexpr int64_t OUTPUT_IDX_0 = 0L;
constexpr int64_t OUTPUT_IDX_1 = 1L;
constexpr size_t MX_SPLIT_K_PER_TOKEN_SCALE_DIM = 3UL;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_THIRD_DIM_INDEX = 3;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2L;
constexpr size_t MX_SPLIT_M_SCALE_DIM = 4UL;
constexpr size_t MX_SPLIT_M_MULTI_TENSOR_SCALE_DIM = 3UL;
constexpr size_t MX_X_DIM = 2UL;
constexpr size_t MX_X_SCALE_DIM = 3UL;
constexpr size_t MX_WEIGHT_DIM = 3UL;
constexpr size_t MX_WEIGHT_SCALE_DIM = 4UL;
constexpr size_t MX_MULTI_WEIGHT_DIM = 2UL;
constexpr size_t MX_MULTI_WEIGHT_SCALE_DIM = 3UL;
constexpr size_t MX_OUTPUT_DIM = 2UL;
constexpr size_t MX_OUTPUT_SCALE_DIM = 3UL;
constexpr size_t PERTOKEN_X_DIM = 2;
constexpr size_t PERTOKEN_X_SCALE_DIM = 1;
constexpr size_t PERTOKEN_WEIGHT_DIM = 3;
constexpr size_t PERTOKEN_WEIGHT_SCALE_DIM = 2;
constexpr size_t PERTOKEN_OUTPUT_DIM = 2;
constexpr size_t PERTOKEN_WEIGHT_NZ_STORAGE_DIM = 5;
constexpr int64_t NZ_INNER_SIZE = 16L;
constexpr int64_t B8_NZ_C0_SIZE = 32L;
constexpr size_t PERTOKEN_OUTPUT_SCALE_DIM = 1;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2L;
constexpr int64_t SWIGLU_SPLIT_SIZE = 64L;
constexpr int64_t MXFP4_K_CONSTRAINT = 2L;
constexpr int64_t SWIGLU_N_CONSTRAINT = 2L;
constexpr int64_t MXFP4_N_CONSTRAINT = 4L;
constexpr int64_t MXFP4_NZ_N_ALIGN = 128L;
constexpr int64_t MXFP8_NZ_N_ALIGN = 64L;
constexpr int64_t MXFP4_ND_N_ALIGN = 4L;
constexpr int64_t MXFP8_ND_N_ALIGN = 2L;
constexpr size_t SINGLE_TENSOR_SIZE = 1;
constexpr int64_t MAX_GROUP_LIST_SIZE = 1024L;
constexpr int64_t QUNAT_MODE_MX = 2;
constexpr int64_t QUNAT_MODE_PERTOKEN = 0;
constexpr int64_t MXA8W4_K_ALIGN = 32LL;
constexpr int64_t MXA8W4_K_MIN = 32LL;
constexpr int64_t MXA8W4_N_ALIGN = 128LL;
constexpr int64_t MXA8W4_N_MIN = 128LL;
constexpr int64_t MXA8W4_SCALE_BLOCK = 128LL;
const std::vector<DataType> X_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2};
const std::vector<DataType> X_DTYPE_SUPPORT_LIST_MXFP4_ND = {DataType::DT_FLOAT4_E2M1};
const std::vector<DataType> X_DTYPE_SUPPORT_LIST_MXFP4_NZ = {DataType::DT_FLOAT4_E2M1, DataType::DT_FLOAT4_E1M2};
const std::vector<DataType> XW_DTYPE_SUPPORT_LIST_PERTOKEN = {DataType::DT_INT8, DataType::DT_FLOAT8_E4M3FN,
                                                              DataType::DT_FLOAT8_E5M2, DataType::DT_HIFLOAT8};
const std::vector<DataType> WEIGHT_DTYPE_SUPPORT_LIST_PERTOKEN_NZ = {DataType::DT_INT8, DataType::DT_FLOAT8_E4M3FN,
                                                                     DataType::DT_HIFLOAT8};
const std::vector<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E5M2};
const std::vector<DataType> WEIGHT_DTYPE_SUPPORT_LIST_MXFP4_ND = {DataType::DT_FLOAT4_E2M1};
const std::vector<DataType> WEIGHT_DTYPE_SUPPORT_LIST_MXFP4_NZ = {DataType::DT_FLOAT4_E2M1, DataType::DT_FLOAT4_E1M2};
const std::vector<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::vector<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN_XINT8 = {DataType::DT_FLOAT16, DataType::DT_BF16,
                                                                              DataType::DT_FLOAT};
const std::vector<DataType> WEIGHT_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN_XFP8HIF8 = {DataType::DT_BF16, DataType::DT_FLOAT};
const std::vector<DataType> X_SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::vector<DataType> X_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN = {DataType::DT_FLOAT};
const std::vector<DataType> GROUP_LIST_DTYPE_SUPPORT_LIST = {DataType::DT_INT64};
const std::vector<DataType> QUANTOUT_DTYPE_SUPPORT_LIST_MXFP4_ND = {DataType::DT_FLOAT8_E4M3FN,
                                                                    DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT4_E2M1};
const std::vector<DataType> QUANTOUT_DTYPE_SUPPORT_LIST_MXFP4_NZ = {DataType::DT_FLOAT8_E4M3FN,
                                                                    DataType::DT_FLOAT4_E2M1, DataType::DT_FLOAT4_E1M2};
const std::vector<DataType> QUANTOUT_DTYPE_SUPPORT_LIST_PERTOKEN = {DataType::DT_INT8, DataType::DT_FLOAT8_E4M3FN,
                                                                    DataType::DT_FLOAT8_E5M2, DataType::DT_HIFLOAT8};
const std::vector<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT8_E8M0};
const std::vector<DataType> QUANTSCALEOUT_DTYPE_SUPPORT_LIST_PERTOKEN = {DataType::DT_FLOAT};
const std::vector<DataType> X_DTYPE_SUPPORT_LIST_MXA8W4 = {DataType::DT_FLOAT8_E4M3FN};
const std::vector<DataType> WEIGHT_DTYPE_SUPPORT_LIST_MXA8W4 = {DataType::DT_FLOAT4_E2M1};

inline bool IsFp8Dtype(DataType dtype)
{
    return dtype == DataType::DT_FLOAT8_E4M3FN || dtype == DataType::DT_FLOAT8_E5M2;
}

inline bool IsMxfp8WeightNzFormat(const aclTensor *weight)
{
    return weight != nullptr && weight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ &&
           IsFp8Dtype(weight->GetDataType());
}

inline bool IsMxfp4WeightNzFormat(const aclTensor *weight)
{
    return weight != nullptr && weight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ;
}

inline const std::vector<DataType> &GetXSupportListMxfp4(const aclTensor *weight)
{
    return IsMxfp4WeightNzFormat(weight) ? X_DTYPE_SUPPORT_LIST_MXFP4_NZ : X_DTYPE_SUPPORT_LIST_MXFP4_ND;
}

inline const std::vector<DataType> &GetWeightSupportListMxfp4(const aclTensor *weight)
{
    return IsMxfp4WeightNzFormat(weight) ? WEIGHT_DTYPE_SUPPORT_LIST_MXFP4_NZ : WEIGHT_DTYPE_SUPPORT_LIST_MXFP4_ND;
}

inline const std::vector<DataType> &GetOutputSupportListMxfp4(const aclTensor *weight)
{
    return IsMxfp4WeightNzFormat(weight) ? QUANTOUT_DTYPE_SUPPORT_LIST_MXFP4_NZ : QUANTOUT_DTYPE_SUPPORT_LIST_MXFP4_ND;
}

inline bool IsMxfp4Dtype(DataType dtype)
{
    return dtype == DataType::DT_FLOAT4_E2M1 || dtype == DataType::DT_FLOAT4_E1M2;
}

inline const char *GetGroupedMatmulSwigluQuantV2ScenarioName(const GroupedMatmulSwigluQuantParamsBase &params)
{
    if (params.x == nullptr || params.weight == nullptr || params.weight->Size() == 0 ||
        (*params.weight)[0] == nullptr) {
        return "unknown";
    }
    DataType xDtype = params.x->GetDataType();
    DataType weightDtype = (*params.weight)[0]->GetDataType();
    if (params.quantMode == QUNAT_MODE_MX) {
        if (IsMxfp4Dtype(xDtype) && IsMxfp4Dtype(weightDtype)) {
            return "MXFP4";
        }
        if ((xDtype == DataType::DT_FLOAT8_E4M3FN || xDtype == DataType::DT_FLOAT8_E5M2) &&
            (weightDtype == DataType::DT_FLOAT8_E4M3FN || weightDtype == DataType::DT_FLOAT8_E5M2)) {
            return "MX FP8";
        }
        return "MX quant";
    }
    if (params.quantMode == QUNAT_MODE_PERTOKEN) {
        if (xDtype == DataType::DT_INT8 && weightDtype == DataType::DT_INT8) {
            return "A8W8 pertoken";
        }
        if ((xDtype == DataType::DT_FLOAT8_E4M3FN || xDtype == DataType::DT_FLOAT8_E5M2) &&
            (weightDtype == DataType::DT_FLOAT8_E4M3FN || weightDtype == DataType::DT_FLOAT8_E5M2)) {
            return "FP8 pertoken";
        }
        if (xDtype == DataType::DT_HIFLOAT8 && weightDtype == DataType::DT_HIFLOAT8) {
            return "HIFLOAT8 pertoken";
        }
        return "pertoken quant";
    }
    return "unsupported quant mode";
}

class GroupedMatmulSwigluQuantBaseHandler : public GroupedMatmulSwigluQuantHandler {
protected:
    bool IsMultiTensorWeight() const
    {
        if (gmmDsqParams_.weight == nullptr || gmmDsqParams_.weight->Size() == 0 ||
            (*gmmDsqParams_.weight)[0] == nullptr) {
            return false;
        }
        auto wDimNum = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDimNum();
        return wDimNum == MX_MULTI_WEIGHT_DIM;
    }

    bool IsMxWeightNzMultiTensorSupported() const
    {
        return gmmDsqParams_.quantMode == QUNAT_MODE_MX && !gmmDsqParams_.isMxA8W4 && gmmDsqParams_.weight != nullptr &&
               gmmDsqParams_.weight->Size() > 0 && (*gmmDsqParams_.weight)[0] != nullptr &&
               op::IsPrivateFormat((*gmmDsqParams_.weight)[0]->GetStorageFormat()) &&
               ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDimNum() == MX_MULTI_WEIGHT_DIM;
    }

    bool IsTransposeForMxShape(const aclTensor *tensor) const
    {
        auto shape = tensor->GetViewShape();
        if (shape.GetDimNum() < MX_SPLIT_K_PER_TOKEN_SCALE_DIM) {
            return false;
        }
        const int64_t dimNum = static_cast<int64_t>(shape.GetDimNum());
        int64_t firstLastDim = dimNum - 1;
        int64_t secondLastDim = dimNum - static_cast<int64_t>(LAST_SECOND_DIM_INDEX);
        int64_t thirdLastDim = dimNum - static_cast<int64_t>(LAST_THIRD_DIM_INDEX);
        auto strides = tensor->GetViewStrides();
        if (strides[firstLastDim] == 1 && strides[thirdLastDim] == MXFP_MULTI_BASE_SIZE &&
            strides[secondLastDim] == shape.GetDim(thirdLastDim) * MXFP_MULTI_BASE_SIZE) {
            return true;
        }
        return false;
    }

    bool IsTransposeLastTwoDims(const aclTensor *tensor) const
    {
        auto shape = tensor->GetViewShape();
        const int64_t dimNum = static_cast<int64_t>(shape.GetDimNum());
        int64_t dim1 = dimNum - 1;
        int64_t dim2 = dimNum - static_cast<int64_t>(LAST_SECOND_DIM_INDEX);
        auto strides = tensor->GetViewStrides();
        if (strides[dim2] == 1 && strides[dim1] == shape.GetDim(dim2)) {
            int64_t tmpNxD = shape.GetDim(dim1) * shape.GetDim(dim2);
            for (int64_t batchDim = dimNum - static_cast<int64_t>(LAST_THIRD_DIM_INDEX); batchDim >= 0; batchDim--) {
                if (strides[batchDim] != tmpNxD) {
                    return false;
                }
                tmpNxD *= shape.GetDim(batchDim);
            }
            return true;
        }
        return false;
    }

    bool CreateContiguousTensorListForMXTypeMScale(const aclTensorList *tensorList,
                                                   std::vector<aclTensor *> &newTensorList,
                                                   aclOpExecutor *executor) const
    {
        op::Shape shape;
        for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
            const aclTensor *inputTensor = (*tensorList)[idx];
            op::Shape viewShape = inputTensor->GetViewShape();
            shape.SetScalar();
            if (viewShape.GetDimNum() < MX_SPLIT_M_MULTI_TENSOR_SCALE_DIM) {
                continue;
            }
            if (viewShape.GetDimNum() > MX_SPLIT_M_MULTI_TENSOR_SCALE_DIM) {
                shape.AppendDim(viewShape.GetDim(0));
            }
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_SECOND_DIM_INDEX));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - LAST_THIRD_DIM_INDEX));
            shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - 1));
            aclTensor *tensor =
                executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
            if (tensor == nullptr) {
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CreateView for contiguous MX scale failed.");
                return false;
            }
            tensor->SetStorageFormat(inputTensor->GetStorageFormat());
            newTensorList.emplace_back(tensor);
        }
        return true;
    }

    bool CreateContiguousTensorList(const aclTensorList *tensorList, std::vector<aclTensor *> &newTensorList,
                                    aclOpExecutor *executor) const
    {
        op::Shape shape;
        for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
            const aclTensor *inputTensor = (*tensorList)[idx];
            op::Shape viewShape = inputTensor->GetViewShape();
            uint32_t viewShapeDimsNum = viewShape.GetDimNum();
            auto storageShape = inputTensor->GetStorageShape();
            shape.SetScalar();
            // 2: the second last dimension; in for-loops, it indicates dimensions before the second last remain
            // unchanged.
            for (uint32_t i = 0; i < viewShapeDimsNum - LAST_SECOND_DIM_INDEX; ++i) {
                shape.AppendDim(viewShape.GetDim(i));
            }
            // viewShapeDimsNum - 1, the dim value of the last dim. viewShapeDimsNum - 2, the dim value of the second
            // last dim.
            shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
            shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - LAST_SECOND_DIM_INDEX));
            aclTensor *tensor =
                executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
            if (tensor == nullptr) {
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CreateView for contiguous tensor failed.");
                return false;
            }
            tensor->SetStorageFormat(inputTensor->GetStorageFormat());
            tensor->SetStorageShape(storageShape);
            newTensorList.emplace_back(tensor);
        }
        return true;
    }

    static void CheckOptionalTensorListEmpty(const aclTensorList *&tensorList)
    {
        if (tensorList != nullptr) {
            if (tensorList->Size() == 0) {
                tensorList = nullptr;
            } else if ((*tensorList)[0] == nullptr) {
                tensorList = nullptr;
            } else if (tensorList->Size() == 1) {
                op::Shape shape = (*tensorList)[0]->GetViewShape();
                if (shape.GetDimNum() == 1 && shape.GetDim(0) == 0) {
                    tensorList = nullptr;
                }
            }
        }
    }

    bool CheckAttrs()
    {
        CheckOptionalTensorListEmpty(gmmDsqParams_.weightAssistMatrix);
        if (gmmDsqParams_.tuningConfig != nullptr && gmmDsqParams_.tuningConfig->Size() == 0) {
            gmmDsqParams_.tuningConfig = nullptr;
        }
        if (gmmDsqParams_.weightAssistMatrix != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE(apiName_.c_str(), "weightAssistMatrix", "nonnull", "nullptr");
            return false;
        }
        if (gmmDsqParams_.bias != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE(apiName_.c_str(), "bias", "nonnull", "nullptr");
            return false;
        }
        if (gmmDsqParams_.smoothScale != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE(apiName_.c_str(), "smoothScale", "nonnull", "nullptr");
            return false;
        }
        if (gmmDsqParams_.tuningConfig != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE(apiName_.c_str(), "tuningConfig", "nonnull", "nullptr");
            return false;
        }
        if ((gmmDsqParams_.dequantMode != QUNAT_MODE_MX && gmmDsqParams_.dequantMode != QUNAT_MODE_PERTOKEN) ||
            (gmmDsqParams_.quantMode != QUNAT_MODE_MX && gmmDsqParams_.quantMode != QUNAT_MODE_PERTOKEN) ||
            (gmmDsqParams_.dequantMode != gmmDsqParams_.quantMode)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName_.c_str(), "dequantMode and quantMode",
                "dequantMode=" + std::to_string(gmmDsqParams_.dequantMode) +
                    ", quantMode=" + std::to_string(gmmDsqParams_.quantMode),
                "the values of dequantMode and quantMode must be 0 (pertoken) or 2 (mx), and they must be equal; "
                "FRACTAL_NZ weight supports pertoken mode for INT8, FLOAT8 or HIFLOAT8 x and weight");
            return false;
        }
        ge::DataType dequantDtype = static_cast<ge::DataType>(gmmDsqParams_.dequantDtype);
        if (gmmDsqParams_.quantMode == QUNAT_MODE_MX && dequantDtype != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName_.c_str(), "dequantDtype",
                                                  std::to_string(gmmDsqParams_.dequantDtype),
                                                  "when the quant mode is mx, dequantDtype must be 0");
            return false;
        }
        if (gmmDsqParams_.quantMode == QUNAT_MODE_PERTOKEN && dequantDtype != ge::DT_FLOAT &&
            dequantDtype != ge::DT_BF16 && dequantDtype != ge::DT_FLOAT16) {
            OP_LOGE_FOR_INVALID_VALUE(apiName_.c_str(), "dequantDtype", std::to_string(gmmDsqParams_.dequantDtype),
                                      "{0, 1, 27}");
            return false;
        }

        if (gmmDsqParams_.isMxA8W4 && gmmDsqParams_.dequantMode != QUNAT_MODE_MX) {
            OP_LOGE_FOR_INVALID_VALUE(apiName_.c_str(), "dequantMode", std::to_string(gmmDsqParams_.dequantMode),
                                      std::to_string(QUNAT_MODE_MX));
            return false;
        }
        return true;
    }

    bool CheckMxA8W4MultiWeightTranspose(bool &transposeWeight, bool &transposeWeightScale)
    {
        for (size_t i = 0; i < gmmDsqParams_.weight->Size(); i++) {
            bool wTrans = IsTransposeLastTwoDims((*gmmDsqParams_.weight)[i]);
            bool wsTrans = IsTransposeForMxShape((*gmmDsqParams_.weightScale)[i]);
            if (!wTrans) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName_.c_str(), "weight",
                                                      ViewShapeToString((*gmmDsqParams_.weight)[i]),
                                                      "MxA8W4: weight[" + std::to_string(i) + "] must be transposed");
                return false;
            }
            if (wsTrans != wTrans) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    apiName_.c_str(), "weightScale and weight",
                    "weightScale[" + std::to_string(i) + "]=" + ViewShapeToString((*gmmDsqParams_.weightScale)[i]) +
                        ", weight[" + std::to_string(i) + "]=" + ViewShapeToString((*gmmDsqParams_.weight)[i]),
                    "the transposition of weightScale must be equal to the transposition of weight");
                return false;
            }
        }
        transposeWeight = true;
        transposeWeightScale = true;
        return true;
    }

    bool CheckMXTranspose()
    {
        bool transposeWeightScale = IsTransposeForMxShape((*gmmDsqParams_.weightScale)[0]);
        bool transposeWeight = IsTransposeLastTwoDims((*gmmDsqParams_.weight)[0]);
        bool transposeX = IsTransposeLastTwoDims(gmmDsqParams_.x);
        bool transposeXScale = IsTransposeForMxShape(gmmDsqParams_.xScale);
        if (gmmDsqParams_.isMxA8W4) {
            if (IsMultiTensorWeight()) {
                if (!CheckMxA8W4MultiWeightTranspose(transposeWeight, transposeWeightScale)) {
                    return false;
                }
            } else {
                if (!transposeWeight) {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        apiName_.c_str(), "transposeWeight", "false",
                        "when the x-weight dtype is fp8-fp4, , the transposition of weight must be true");
                    return false;
                }
            }
            gmmDsqParams_.transposeWeight = true;
        }

        if (transposeWeightScale != transposeWeight) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                apiName_.c_str(), "weightScale and weight",
                "weightScale=" + ViewShapeToString((*gmmDsqParams_.weightScale)[0]) +
                    ", weight=" + ViewShapeToString((*gmmDsqParams_.weight)[0]),
                "the transposition of weightScale must be equal to the transposition of weight");
            return false;
        }

        if (transposeWeightScale && transposeWeight) {
            gmmDsqParams_.transposeWeight = true;
            if (!IsMxWeightNzMultiTensorSupported()) {
                auto uniqueExecutor = CREATE_EXECUTOR();
                CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
                aclOpExecutor *executorPtr = uniqueExecutor.get();
                CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
                std::vector<aclTensor *> scaleTensorList;
                std::vector<aclTensor *> weightTensorList;
                if (!CreateContiguousTensorListForMXTypeMScale(gmmDsqParams_.weightScale, scaleTensorList,
                                                               executorPtr)) {
                    return false;
                }
                gmmDsqParams_.weightScale =
                    executorPtr->AllocTensorList(scaleTensorList.data(), scaleTensorList.size());
                if (gmmDsqParams_.weightScale == nullptr) {
                    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensorList for contiguous weightScale failed.");
                    return false;
                }
                if (!CreateContiguousTensorList(gmmDsqParams_.weight, weightTensorList, executorPtr)) {
                    return false;
                }
                gmmDsqParams_.weight = executorPtr->AllocTensorList(weightTensorList.data(), weightTensorList.size());
                if (gmmDsqParams_.weight == nullptr) {
                    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensorList for contiguous weight failed.");
                    return false;
                }
                uniqueExecutor.ReleaseTo(executor_);
            }
        }

        if ((gmmDsqParams_.x->GetViewShape().GetDim(0) == 1 && gmmDsqParams_.x->GetViewShape().GetDim(1) == 1) ||
            (gmmDsqParams_.xScale->GetViewShape().GetDim(0) == 1 &&
             gmmDsqParams_.xScale->GetViewShape().GetDim(1) == 1)) {
            return true;
        }
        if (transposeX || transposeXScale) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                apiName_.c_str(), "x and xScale",
                "x=" + ViewShapeToString(gmmDsqParams_.x) + ", xScale=" + ViewShapeToString(gmmDsqParams_.xScale),
                "the transposition of x and xScale must be false");
            return false;
        }
        return true;
    }

    void SetTensorListOriginalShapes(const aclTensorList *tensorList,
                                     const std::vector<op::Shape> &originalShapes) const
    {
        if (tensorList == nullptr || tensorList->Size() != originalShapes.size()) {
            return;
        }
        for (size_t i = 0; i < tensorList->Size(); ++i) {
            auto *tensor = (*tensorList)[i];
            if (tensor != nullptr) {
                tensor->SetOriginalShape(originalShapes[i]);
            }
        }
    }

    void CreateMultiWeightScaleTensorListForTranspose(const aclTensorList *tensorList,
                                                      std::vector<aclTensor *> &newTensorList,
                                                      aclOpExecutor *executor) const
    {
        op::Shape shape;
        for (uint64_t idx = 0; idx < (*tensorList).Size(); idx++) {
            const aclTensor *inputTensor = (*tensorList)[idx];
            op::Shape viewShape = inputTensor->GetViewShape();
            if (viewShape.GetDimNum() != MX_MULTI_WEIGHT_SCALE_DIM) {
                continue;
            }
            auto storageShape = inputTensor->GetStorageShape();
            shape.SetScalar();
            shape.AppendDim(viewShape.GetDim(1));
            shape.AppendDim(viewShape.GetDim(0));
            shape.AppendDim(viewShape.GetDim(2));
            aclTensor *tensor = executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset());
            tensor->SetStorageFormat(inputTensor->GetStorageFormat());
            tensor->SetStorageShape(storageShape);
            newTensorList.emplace_back(tensor);
        }
    }

    void PrepareOriginalShapesBeforeContiguous() override
    {
        weightScaleOriginalShapes_.clear();
        if (!IsMxWeightNzMultiTensorSupported()) {
            GroupedMatmulSwigluQuantHandler::PrepareOriginalShapesBeforeContiguous();
            return;
        }
        if (!gmmDsqParams_.transposeWeight) {
            GroupedMatmulSwigluQuantHandler::PrepareOriginalShapesBeforeContiguous();
            return;
        }

        std::vector<aclTensor *> weightTensorList;
        std::vector<aclTensor *> weightScaleTensorList;
        CreateContiguousTensorList(gmmDsqParams_.weight, weightTensorList, l0Executor_);
        gmmDsqParams_.weight = l0Executor_->AllocTensorList(weightTensorList.data(), weightTensorList.size());
        CreateMultiWeightScaleTensorListForTranspose(gmmDsqParams_.weightScale, weightScaleTensorList, l0Executor_);
        gmmDsqParams_.weightScale =
            l0Executor_->AllocTensorList(weightScaleTensorList.data(), weightScaleTensorList.size());

        for (size_t i = 0; i < gmmDsqParams_.weight->Size(); ++i) {
            auto *w = (*gmmDsqParams_.weight)[i];
            if (w != nullptr && IsPrivateFormat(w->GetStorageFormat())) {
                w->SetOriginalShape(w->GetViewShape());
            }
        }
        for (size_t i = 0; i < gmmDsqParams_.weightScale->Size(); ++i) {
            weightScaleOriginalShapes_.emplace_back((*gmmDsqParams_.weightScale)[i]->GetViewShape());
        }
    }

    void RestoreOriginalShapesAfterContiguous() override
    {
        SetTensorListOriginalShapes(gmmDsqParams_.weightScale, weightScaleOriginalShapes_);
        weightScaleOriginalShapes_.clear();
    }

    bool CheckPertokenTranspose()
    {
        bool transposeWeight = IsTransposeLastTwoDims((*gmmDsqParams_.weight)[0]);
        bool transposeX = IsTransposeLastTwoDims(gmmDsqParams_.x);

        if (transposeWeight) {
            gmmDsqParams_.transposeWeight = true;
            auto uniqueExecutor = CREATE_EXECUTOR();
            CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            aclOpExecutor *executorPtr = uniqueExecutor.get();
            CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
            std::vector<aclTensor *> weightTensorList;
            if (!CreateContiguousTensorList(gmmDsqParams_.weight, weightTensorList, executorPtr)) {
                return false;
            }
            gmmDsqParams_.weight = executorPtr->AllocTensorList(weightTensorList.data(), weightTensorList.size());
            if (gmmDsqParams_.weight == nullptr) {
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensorList for contiguous weight failed.");
                return false;
            }
            uniqueExecutor.ReleaseTo(executor_);
        }
        if ((gmmDsqParams_.x->GetViewShape().GetDim(0) == 1 && gmmDsqParams_.x->GetViewShape().GetDim(1) == 1) ||
            (gmmDsqParams_.xScale->GetViewShape().GetDim(0) == 1 &&
             gmmDsqParams_.xScale->GetViewShape().GetDim(1) == 1)) {
            return true;
        }
        if (transposeX) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName_.c_str(), "x", ViewShapeToString(gmmDsqParams_.x),
                                                  "the transposition of x must be false");
            return false;
        }
        return true;
    }

    int64_t GetWeightK(const aclTensor *weight) const
    {
        auto dimNum = weight->GetViewShape().GetDimNum();
        return gmmDsqParams_.transposeWeight ? weight->GetViewShape().GetDim(dimNum - 1) :
                                               weight->GetViewShape().GetDim(dimNum - LAST_SECOND_DIM_INDEX);
    }

    int64_t GetWeightN(const aclTensor *weight) const
    {
        auto dimNum = weight->GetViewShape().GetDimNum();
        return gmmDsqParams_.transposeWeight ? weight->GetViewShape().GetDim(dimNum - LAST_SECOND_DIM_INDEX) :
                                               weight->GetViewShape().GetDim(dimNum - 1);
    }

    bool CheckMultiWeightKNConsistency()
    {
        int64_t k0 = GetWeightK((*gmmDsqParams_.weight)[0]);
        int64_t n0 = GetWeightN((*gmmDsqParams_.weight)[0]);
        for (size_t i = 1; i < gmmDsqParams_.weight->Size(); i++) {
            int64_t ki = GetWeightK((*gmmDsqParams_.weight)[i]);
            int64_t ni = GetWeightN((*gmmDsqParams_.weight)[i]);
            if (ki != k0 || ni != n0) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    apiName_.c_str(), "weight",
                    "weight[0]=(k=" + std::to_string(k0) + ", n=" + std::to_string(n0) + "), weight[" +
                        std::to_string(i) + "]=(k=" + std::to_string(ki) + ", n=" + std::to_string(ni) + ")",
                    "MxA8W4: all weight tensors must have the same k and n");
                return false;
            }
        }
        return true;
    }

    bool CheckMXExpectShape(int64_t m, int64_t k, int64_t n, int64_t e)
    {
        // x的shape期望为[M, K]
        op::Shape xExpectShape = {m, k};
        // xScale的shape期望为[M, CeilDiv(K, 64), 2]
        op::Shape xScaleExpectShape = {m, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};
        int64_t nAfterHalve = static_cast<int64_t>(n / SWIGLU_SPLIT_FACTOR);
        // output的shape期望为[M, N / 2]
        op::Shape outputExpectShape = {m, nAfterHalve};
        // outputScale的shape期望为[M, CeilDiv(N / 2, 64), 2]
        op::Shape outputScaleExpectShape = {m, Ops::Base::CeilDiv(nAfterHalve, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};
        const aclTensor *x = gmmDsqParams_.x;
        const aclTensor *xScale = gmmDsqParams_.xScale;
        const aclTensor *output = gmmDsqParams_.output;
        const aclTensor *outputScale = gmmDsqParams_.outputScale;
        const char *scenario = GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_);
        GMM_SWIGLU_CHECK_SHAPE(x, "x", xExpectShape, return false);
        GMM_SWIGLU_CHECK_SHAPE(xScale, "xScale", xScaleExpectShape, return false);
        GMM_SWIGLU_CHECK_SHAPE(output, "output", outputExpectShape, return false);
        GMM_SWIGLU_CHECK_SHAPE(outputScale, "outputScale", outputScaleExpectShape, return false);

        // 进行swiglu操作需满足n为偶数
        if (n % SWIGLU_N_CONSTRAINT != 0) {
            std::string gotStr = BuildLogValue("N", n);
            std::ostringstream constraint;
            constraint << "N must be even when " << scenario;
            std::string constraintStr = constraint.str();
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName_.c_str(), "weight",
                                                  ViewShapeToString((*gmmDsqParams_.weight)[0]),
                                                  "axis N of weight must be an even number");
            return false;
        }
        // groupList的长度应等于weight的专家数
        int64_t groupListLen = gmmDsqParams_.groupList->GetViewShape().GetDim(0);
        if (groupListLen != e) {
            std::ostringstream reason;
            reason << "groupList length should be equal to expert num " << e << " when " << scenario
                   << ", but actual is " << groupListLen;
            std::string reasonStr = reason.str();
            OP_LOGE_FOR_INVALID_LISTSIZE(apiName_.c_str(), "groupList", std::to_string(groupListLen),
                                         std::to_string(e));
            return false;
        }

        return true;
    }

    bool CheckMXMultiWeightShape()
    {
        // x shape为(m, k), weight shape为(n, k)
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0); // 从x的第0维获取m
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1); // 从x的第1维获取k
        int64_t n = GetWeightN((*gmmDsqParams_.weight)[0]);
        int64_t e = static_cast<int64_t>(gmmDsqParams_.weight->Size());

        if (!CheckMultiWeightKNConsistency()) {
            return false;
        }
        op::Shape weightTransExpectShape = {n, k};
        op::Shape weightExpectShape = {k, n};
        op::Shape weightScaleTransExpectShape = {n, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};
        op::Shape weightScaleExpectShape = {Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), n, SWIGLU_SPLIT_FACTOR};
        for (size_t i = 0; i < gmmDsqParams_.weight->Size(); i++) {
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            if (gmmDsqParams_.transposeWeight) {
                GMM_SWIGLU_CHECK_SHAPE(weight, "weight", weightTransExpectShape, return false);
                GMM_SWIGLU_CHECK_SHAPE(weightScale, "weightScale", weightScaleTransExpectShape, return false);
            } else {
                GMM_SWIGLU_CHECK_SHAPE(weight, "weight", weightExpectShape, return false);
                GMM_SWIGLU_CHECK_SHAPE(weightScale, "weightScale", weightScaleExpectShape, return false);
            }
        }

        return CheckMXExpectShape(m, k, n, e);
    }

    bool CheckMXWeightNzMultiWeightShape()
    {
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0);
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1);
        int64_t n = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1);
        int64_t e = static_cast<int64_t>(gmmDsqParams_.weight->Size());
        op::Shape weightExpectShape = {k, n};
        op::Shape weightScaleExpectShape = {Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), n, SWIGLU_SPLIT_FACTOR};
        for (size_t i = 0; i < gmmDsqParams_.weight->Size(); ++i) {
            GMM_SWIGLU_CHECK_SHAPE((*gmmDsqParams_.weight)[i], "weight", weightExpectShape, return false);
            GMM_SWIGLU_CHECK_SHAPE((*gmmDsqParams_.weightScale)[i], "weightScale", weightScaleExpectShape,
                                   return false);
        }
        return CheckMXExpectShape(m, k, n, e);
    }

    bool CheckMxSingleWeightShape()
    {
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0); // 从x的第0维获取m
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1); // 从x的第1维获取k
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t n = gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                                                    ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2);
        int64_t e = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0); // 从weight的第0维获取e

        // weight的shape期望为[E, K, N]
        op::Shape weightExpectShape = {e, k, n};
        // weightScale的shape期望为[E, CeilDiv(K, 64), N, 2]
        op::Shape weightScaleExpectShape = {e, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), n, SWIGLU_SPLIT_FACTOR};
        // weight转置的shape期望为[E, N, K]
        op::Shape weightTransExpectShape = {e, n, k};
        // weightScale转置的shape期望为[E, N, CeilDiv(K, 64), 2]
        op::Shape weightScaleTransExpectShape = {e, n, Ops::Base::CeilDiv(k, SWIGLU_SPLIT_SIZE), SWIGLU_SPLIT_FACTOR};
        const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[0];
        const aclTensor *weight = (*gmmDsqParams_.weight)[0];
        if (gmmDsqParams_.transposeWeight) {
            GMM_SWIGLU_CHECK_SHAPE(weightScale, "weightScale", weightScaleTransExpectShape, return false);
            GMM_SWIGLU_CHECK_SHAPE(weight, "weight", weightTransExpectShape, return false);
        } else {
            GMM_SWIGLU_CHECK_SHAPE(weightScale, "weightScale", weightScaleExpectShape, return false);
            GMM_SWIGLU_CHECK_SHAPE(weight, "weight", weightExpectShape, return false);
        }

        return CheckMXExpectShape(m, k, n, e);
    }

    bool CheckMXShape()
    {
        if (IsMxWeightNzMultiTensorSupported()) {
            return CheckMXWeightNzMultiWeightShape();
        } else if (IsMultiTensorWeight()) {
            return CheckMXMultiWeightShape();
        } else {
            return CheckMxSingleWeightShape();
        }
    }

    bool CheckPertokenShape()
    {
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0); // 从x的第0维获取m
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1); // 从x的第1维获取k
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t n = gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                                                    ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2);
        int64_t e = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0); // 从weight的第0维获取e

        // x的shape期望为[M, K]
        op::Shape xExpectShape = {m, k};
        // xScale的shape期望为[M]
        op::Shape xScaleExpectShape = {m};
        // weight的shape期望为根据转置的情况来具体确认[E, K, N] 或者[E, N, K]
        op::Shape weightExpectShape = gmmDsqParams_.transposeWeight ? op::Shape{e, n, k} : op::Shape{e, k, n};
        // weightScale的shape期望为[E, N]
        op::Shape weightScaleExpectShape = {e, n};
        int64_t nAfterHalve = static_cast<int64_t>(n / SWIGLU_SPLIT_FACTOR);
        // output的shape期望为[M, N / 2]
        op::Shape outputExpectShape = {m, nAfterHalve};
        // outputScale的shape期望为[M]
        op::Shape outputScaleExpectShape = {m};
        const aclTensor *x = gmmDsqParams_.x;
        const aclTensor *xScale = gmmDsqParams_.xScale;
        const aclTensor *weight = (*gmmDsqParams_.weight)[0];
        const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[0];
        const aclTensor *output = gmmDsqParams_.output;
        const aclTensor *outputScale = gmmDsqParams_.outputScale;
        GMM_SWIGLU_CHECK_SHAPE(x, "x", xExpectShape, return false);
        GMM_SWIGLU_CHECK_SHAPE(xScale, "xScale", xScaleExpectShape, return false);
        GMM_SWIGLU_CHECK_SHAPE(weight, "weight", weightExpectShape, return false);
        GMM_SWIGLU_CHECK_SHAPE(weightScale, "weightScale", weightScaleExpectShape, return false);
        GMM_SWIGLU_CHECK_SHAPE(output, "output", outputExpectShape, return false);
        GMM_SWIGLU_CHECK_SHAPE(outputScale, "outputScale", outputScaleExpectShape, return false);
        if (weight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
            const op::Shape &storageShape = weight->GetStorageShape();
            GMM_SWIGLU_CHECK_DIM(storageShape.GetDimNum(), PERTOKEN_WEIGHT_NZ_STORAGE_DIM, "weight storage shape",
                                 return false);
            if (n <= 0 || n % MXFP8_NZ_N_ALIGN != 0) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    apiName_.c_str(), "weight", ViewShapeToString(weight),
                    "in per-token B8 mode, N of FRACTAL_NZ weight must be positive and aligned to 64");
                return false;
            }
            int64_t groupListLen = gmmDsqParams_.groupList->GetViewShape().GetDim(0);
            if (groupListLen != e) {
                OP_LOGE_FOR_INVALID_LISTSIZE(apiName_.c_str(), "groupList", std::to_string(groupListLen),
                                             std::to_string(e));
                return false;
            }
            const int64_t n1 = gmmDsqParams_.transposeWeight ? Ops::Base::CeilDiv(k, B8_NZ_C0_SIZE) :
                                                               Ops::Base::CeilDiv(n, B8_NZ_C0_SIZE);
            const int64_t k1 = gmmDsqParams_.transposeWeight ? Ops::Base::CeilDiv(n, NZ_INNER_SIZE) :
                                                               Ops::Base::CeilDiv(k, NZ_INNER_SIZE);
            const op::Shape expectedStorageShape{e, n1, k1, NZ_INNER_SIZE, B8_NZ_C0_SIZE};
            if (storageShape != expectedStorageShape) {
                OP_LOGE_FOR_INVALID_SHAPE(apiName_.c_str(), "weight storage shape",
                                          op::ToString(storageShape).GetString(),
                                          op::ToString(expectedStorageShape).GetString());
                return false;
            }
        }
        return true;
    }

    bool CheckFp8DtypeValid(const aclTensor *x, const aclTensor *xScale, const aclTensor *groupList,
                            const aclTensor *output, const aclTensor *outputScale)
    {
        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < weightLength; i++) {
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            GMM_SWIGLU_CHECK_DTYPE(weight, "weight", WEIGHT_DTYPE_SUPPORT_LIST, return false);
            GMM_SWIGLU_CHECK_DTYPE(weightScale, "weightScale", WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
        }
        GMM_SWIGLU_CHECK_DTYPE(x, "x", X_DTYPE_SUPPORT_LIST, return false);
        GMM_SWIGLU_CHECK_DTYPE(xScale, "xScale", X_SCALE_DTYPE_SUPPORT_LIST, return false);
        GMM_SWIGLU_CHECK_DTYPE(groupList, "groupList", GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        GMM_SWIGLU_CHECK_DTYPE(outputScale, "outputScale", QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);
        DataType outputDtype = gmmDsqParams_.output->GetDataType();
        if (IsMxfp8WeightNzFormat((*gmmDsqParams_.weight)[0]) && outputDtype != DataType::DT_FLOAT8_E4M3FN) {
            OP_LOGE_FOR_INVALID_DTYPE(apiName_.c_str(), "output", op::ToString(outputDtype).GetString(),
                                      "{FLOAT8_E4M3FN}");
            return false;
        }
        if (outputDtype != DataType::DT_FLOAT8_E4M3FN && outputDtype != DataType::DT_FLOAT8_E5M2) {
            OP_LOGE_FOR_INVALID_DTYPE(apiName_.c_str(), "output", op::ToString(outputDtype).GetString(),
                                      "{FLOAT8_E4M3FN, FLOAT8_E5M2}");
            return false;
        }
        (void)output;
        return true;
    }

    bool CheckFp4DtypeValid(const aclTensor *x, const aclTensor *xScale, const aclTensor *groupList,
                            const aclTensor *output, const aclTensor *outputScale)
    {
        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < weightLength; i++) {
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            GMM_SWIGLU_CHECK_DTYPE(weight, "weight", GetWeightSupportListMxfp4(weight), return false);
            GMM_SWIGLU_CHECK_DTYPE(weightScale, "weightScale", WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
        }
        GMM_SWIGLU_CHECK_DTYPE(x, "x", GetXSupportListMxfp4((*gmmDsqParams_.weight)[0]), return false);
        GMM_SWIGLU_CHECK_DTYPE(xScale, "xScale", X_SCALE_DTYPE_SUPPORT_LIST, return false);
        GMM_SWIGLU_CHECK_DTYPE(groupList, "groupList", GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        GMM_SWIGLU_CHECK_DTYPE(output, "output", GetOutputSupportListMxfp4((*gmmDsqParams_.weight)[0]), return false);
        GMM_SWIGLU_CHECK_DTYPE(outputScale, "outputScale", QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);
        return true;
    }

    bool CheckMxA8W4DtypeValid(const aclTensor *x, const aclTensor *xScale, const aclTensor *groupList,
                               const aclTensor *output, const aclTensor *outputScale)
    {
        GMM_SWIGLU_CHECK_DTYPE(x, "x", X_DTYPE_SUPPORT_LIST_MXA8W4, return false);
        GMM_SWIGLU_CHECK_DTYPE(xScale, "xScale", X_SCALE_DTYPE_SUPPORT_LIST, return false);
        GMM_SWIGLU_CHECK_DTYPE(groupList, "groupList", GROUP_LIST_DTYPE_SUPPORT_LIST, return false);

        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < weightLength; i++) {
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            GMM_SWIGLU_CHECK_DTYPE(weight, "weight", WEIGHT_DTYPE_SUPPORT_LIST_MXA8W4, return false);
            GMM_SWIGLU_CHECK_DTYPE(weightScale, "weightScale", WEIGHT_SCALE_DTYPE_SUPPORT_LIST, return false);
        }

        DataType outputDtype = output->GetDataType();
        if (outputDtype != DataType::DT_FLOAT8_E4M3FN) {
            OP_LOGE_FOR_INVALID_DTYPE(apiName_.c_str(), "output", op::ToString(outputDtype).GetString(),
                                      "{FLOAT8_E4M3FN}");
            return false;
        }
        GMM_SWIGLU_CHECK_DTYPE(outputScale, "outputScale", QUANTSCALEOUT_DTYPE_SUPPORT_LIST, return false);
        return true;
    }

    bool CheckPertokenDtypeValid(const aclTensor *x, const aclTensor *xScale, const aclTensor *groupList,
                                 const aclTensor *output, const aclTensor *outputScale)
    {
        GMM_SWIGLU_CHECK_DTYPE(x, "x", XW_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        GMM_SWIGLU_CHECK_DTYPE(xScale, "xScale", X_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        GMM_SWIGLU_CHECK_DTYPE(groupList, "groupList", GROUP_LIST_DTYPE_SUPPORT_LIST, return false);
        GMM_SWIGLU_CHECK_DTYPE(output, "output", QUANTOUT_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        GMM_SWIGLU_CHECK_DTYPE(outputScale, "outputScale", QUANTSCALEOUT_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < weightLength; i++) {
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            if (weight->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
                GMM_SWIGLU_CHECK_DTYPE(weight, "weight", WEIGHT_DTYPE_SUPPORT_LIST_PERTOKEN_NZ, return false);
            } else {
                GMM_SWIGLU_CHECK_DTYPE(weight, "weight", XW_DTYPE_SUPPORT_LIST_PERTOKEN, return false);
            }
            DataType xDtype = gmmDsqParams_.x->GetDataType();
            if (xDtype == DataType::DT_INT8) {
                GMM_SWIGLU_CHECK_DTYPE(weightScale, "weightScale", WEIGHT_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN_XINT8,
                                       return false);
            } else {
                GMM_SWIGLU_CHECK_DTYPE(weightScale, "weightScale", WEIGHT_SCALE_DTYPE_SUPPORT_LIST_PERTOKEN_XFP8HIF8,
                                       return false);
            }
        }
        DataType xDtype = gmmDsqParams_.x->GetDataType();
        return IsDtypeCompatiblePertoken(xDtype, ((*gmmDsqParams_.weight)[0])->GetDataType());
    }

    bool IsDtypeCompatiblePertoken(const DataType a, const DataType b) const
    {
        if ((a == DataType::DT_FLOAT8_E4M3FN || a == DataType::DT_FLOAT8_E5M2) &&
            (b == DataType::DT_FLOAT8_E4M3FN || b == DataType::DT_FLOAT8_E5M2)) {
            return true;
        }
        return a == b;
    }

    bool checkMxfp4InputShape()
    {
        int64_t kValue = gmmDsqParams_.x->GetViewShape().GetDim(1);
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t nValue = IsMxWeightNzMultiTensorSupported() ?
                             ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                             (gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                                                              ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2));
        // mxfp4场景不支持k=2
        if (kValue == MXFP4_K_CONSTRAINT) {
            std::string gotStr = BuildLogValue("K", kValue);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                apiName_.c_str(), "x", ViewShapeToString(gmmDsqParams_.x),
                "when the dtypes of x and weight inputs are DT_FLOAT4, axis K of x must be greater than 2");
            return false;
        }

        // 1：检查K是否为偶数
        int64_t kModValue = kValue % MXFP4_K_CONSTRAINT;
        // 2：检查N是否为偶数
        int64_t nModValue = nValue % MXFP4_N_CONSTRAINT;
        if (kModValue != 0) {
            std::string gotStr = BuildLogValue("K", kValue);
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                apiName_.c_str(), "x", ViewShapeToString(gmmDsqParams_.x),
                "when the dtypes of x and weight inputs are DT_FLOAT4, axis K of x must be an even number");
            return false;
        }

        // mxfp4、NZ场景下，N需满足128对齐
        if (IsMxfp4WeightNzFormat((*gmmDsqParams_.weight)[0])) {
            if (nValue % MXFP4_NZ_N_ALIGN != 0 || nValue == 0) {
                std::string weightDtypeStr = op::ToString((*gmmDsqParams_.weight)[0]->GetDataType()).GetString();
                std::string reasonStr =
                    "when using the weightNZ format with " + weightDtypeStr +
                    " data type, axis N of weight must be greater than 0 and an integer multiple of 128";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName_.c_str(), "weight",
                                                      ViewShapeToString((*gmmDsqParams_.weight)[0]), reasonStr.c_str());
                return false;
            }
        } else {
            // mxfp4、ND场景下，N需满足4对齐
            if (nValue % MXFP4_ND_N_ALIGN != 0 || nValue == 0) {
                std::string weightDtypeStr = op::ToString((*gmmDsqParams_.weight)[0]->GetDataType()).GetString();
                std::string reasonStr =
                    "when using the ND format with " + weightDtypeStr +
                    " data type, axis N of weight must be greater than 0 and an integer multiple of 4";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName_.c_str(), "weight",
                                                      ViewShapeToString((*gmmDsqParams_.weight)[0]), reasonStr.c_str());
                return false;
            }
        }

        return true;
    }

    bool checkMxfp8InputShape()
    {
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t nValue = IsMxWeightNzMultiTensorSupported() ?
                             ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                             (gmmDsqParams_.transposeWeight ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1) :
                                                              ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(2));

        // mxfp8、NZ场景下，N需满足64对齐
        if (IsMxfp8WeightNzFormat((*gmmDsqParams_.weight)[0])) {
            if (nValue % MXFP8_NZ_N_ALIGN != 0 || nValue == 0) {
                std::string weightDtypeStr = op::ToString((*gmmDsqParams_.weight)[0]->GetDataType()).GetString();
                std::string reasonStr =
                    "when using the weightNZ format with " + weightDtypeStr +
                    " data type, axis N of weight must be greater than 0 and an integer multiple of 64";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName_.c_str(), "weight",
                                                      ViewShapeToString((*gmmDsqParams_.weight)[0]), reasonStr.c_str());
                return false;
            }
        } else {
            // mxfp8、ND场景下，N需满足2对齐
            if (nValue % MXFP8_ND_N_ALIGN != 0 || nValue == 0) {
                std::string weightDtypeStr = op::ToString((*gmmDsqParams_.weight)[0]->GetDataType()).GetString();
                std::string reasonStr =
                    "when using the ND format with " + weightDtypeStr +
                    " data type, axis N of weight must be greater than 0 and an integer multiple of 2";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName_.c_str(), "weight",
                                                      ViewShapeToString((*gmmDsqParams_.weight)[0]), reasonStr.c_str());
                return false;
            }
        }

        return true;
    }

    bool CheckMxA8W4InputShape()
    {
        int64_t m = gmmDsqParams_.x->GetViewShape().GetDim(0);
        int64_t k = gmmDsqParams_.x->GetViewShape().GetDim(1);
        // 多tensor下shape:(n,k), 单tensor下shape:(e, n, k)
        int64_t n = IsMultiTensorWeight() ? ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0) :
                                            ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1);
        int64_t e = IsMultiTensorWeight() ? static_cast<int64_t>(gmmDsqParams_.weight->Size()) :
                                            ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0);

        if (k % MXA8W4_K_ALIGN != 0 || k < MXA8W4_K_MIN) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName_.c_str(), "x", ViewShapeToString(gmmDsqParams_.x),
                                                  "when the quantization mode is MxA8W4, K(" + std::to_string(k) +
                                                      ") must align to " + std::to_string(MXA8W4_K_ALIGN) +
                                                      " and >= " + std::to_string(MXA8W4_K_MIN));
            return false;
        }
        if (n % MXA8W4_N_ALIGN != 0 || n < MXA8W4_N_MIN) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                apiName_.c_str(), "weight", ViewShapeToString((*gmmDsqParams_.weight)[0]),
                "when the quantization mode is MxA8W4, N(" + std::to_string(n) + ") must align to " +
                    std::to_string(MXA8W4_N_ALIGN) + " and >= " + std::to_string(MXA8W4_N_MIN));
            return false;
        }

        op::Shape groupListExpectShape = {e};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmDsqParams_.groupList, groupListExpectShape, return false);
        return true;
    }

    bool CheckEmptyTensor() override
    {
        auto xMDim = gmmDsqParams_.x->GetViewShape().GetDim(0);
        if (gmmDsqParams_.x->GetViewShape().GetDim(1) <= 0) {
            std::string gotStr = BuildLogValue("K", gmmDsqParams_.x->GetViewShape().GetDim(1));
            std::ostringstream constraint;
            constraint << "K must be positive when " << GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_);
            std::string constraintStr = constraint.str();
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(apiName_.c_str(), "x", ViewShapeToString(gmmDsqParams_.x),
                                                   "when the value of M is not 0 [M=" + std::to_string(xMDim) +
                                                       "], axis K of x must be a positive number");
            return false;
        }
        auto weightDimNum = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDimNum();
        auto weightKIndex = weightDimNum - LAST_SECOND_DIM_INDEX;
        auto weightNDim = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(weightDimNum - 1);
        if (((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(weightKIndex) <= 0) {
            std::string gotStr = BuildLogValue("K", ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(weightKIndex));
            std::ostringstream constraint;
            constraint << "K must be positive when " << GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_);
            std::string constraintStr = constraint.str();
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(apiName_.c_str(), "weight",
                                                   ViewShapeToString((*gmmDsqParams_.weight)[0]),
                                                   "when the value of N is not 0 [N=" + std::to_string(weightNDim) +
                                                       "], axis K of weight must be a positive number");
            return false;
        }
        return true;
    }

    bool CheckInputOutDims() override
    {
        if (!CheckAttrs()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In op [%s], when %s, CheckAttrs failed.", opName_.c_str(),
                    GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_));
            return false;
        }

        if (gmmDsqParams_.quantMode == QUNAT_MODE_MX) {
            return CheckInputOutDimsForMX();
        } else if (gmmDsqParams_.quantMode == QUNAT_MODE_PERTOKEN) {
            return CheckInputOutDimsForPertoken();
        } else {
            OP_LOGE_FOR_INVALID_VALUE(apiName_.c_str(), "quantMode", std::to_string(gmmDsqParams_.quantMode), "{0, 2}");
            return false;
        }
        return true;
    }

    bool CheckInputOutDimsForMX()
    {
        auto xDimNumber = gmmDsqParams_.x->GetViewShape().GetDimNum();
        auto xScaleDimNumber = gmmDsqParams_.xScale->GetViewShape().GetDimNum();
        auto outputDimNumber = gmmDsqParams_.output->GetViewShape().GetDimNum();
        auto outputScaleDimNumber = gmmDsqParams_.outputScale->GetViewShape().GetDimNum();
        GMM_SWIGLU_CHECK_DIM(xDimNumber, MX_X_DIM, "x", return false);
        GMM_SWIGLU_CHECK_DIM(xScaleDimNumber, MX_X_SCALE_DIM, "xScale", return false);
        GMM_SWIGLU_CHECK_DIM(outputDimNumber, MX_OUTPUT_DIM, "output", return false);
        GMM_SWIGLU_CHECK_DIM(outputScaleDimNumber, MX_OUTPUT_SCALE_DIM, "outputScale", return false);

        size_t wSize = gmmDsqParams_.weight->Size();
        size_t wsSize = gmmDsqParams_.weightScale->Size();
        if (wSize != wsSize) {
            OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                apiName_.c_str(), "weight and weightScale",
                "weight=" + std::to_string(wSize) + ", weightScale=" + std::to_string(wsSize),
                "weight and weightScale must have the same number of tensors");
            return false;
        }

        if (IsMultiTensorWeight()) {
            if (!gmmDsqParams_.isMxA8W4 && !IsMxWeightNzMultiTensorSupported()) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(apiName_.c_str(), "weight", std::to_string(MX_MULTI_WEIGHT_DIM),
                                             std::to_string(MX_WEIGHT_DIM));
                return false;
            }
            for (size_t i = 0; i < wSize; i++) {
                auto weightDimNumber = ((*gmmDsqParams_.weight)[i])->GetViewShape().GetDimNum();
                auto weightScaleDimNumber = ((*gmmDsqParams_.weightScale)[i])->GetViewShape().GetDimNum();
                GMM_SWIGLU_CHECK_DIM(weightDimNumber, MX_MULTI_WEIGHT_DIM, "weight", return false);
                GMM_SWIGLU_CHECK_DIM(weightScaleDimNumber, MX_MULTI_WEIGHT_SCALE_DIM, "weightScale", return false);
            }
        } else {
            GMM_SWIGLU_CHECK_TENSORNUM(wSize, SINGLE_TENSOR_SIZE, "weight", return false);
            GMM_SWIGLU_CHECK_TENSORNUM(wsSize, SINGLE_TENSOR_SIZE, "weightScale", return false);
            auto weightDimNumber = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDimNum();
            auto weightScaleDimNumber = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDimNum();
            GMM_SWIGLU_CHECK_DIM(weightScaleDimNumber, MX_WEIGHT_SCALE_DIM, "weightScale", return false);
            GMM_SWIGLU_CHECK_DIM(weightDimNumber, MX_WEIGHT_DIM, "weight", return false);
        }
        return true;
    }
    bool CheckInputOutDimsForPertoken()
    {
        auto xDimNumber = gmmDsqParams_.x->GetViewShape().GetDimNum();
        auto xScaleDimNumber = gmmDsqParams_.xScale->GetViewShape().GetDimNum();
        auto outputDimNumber = gmmDsqParams_.output->GetViewShape().GetDimNum();
        auto outputScaleDimNumber = gmmDsqParams_.outputScale->GetViewShape().GetDimNum();
        GMM_SWIGLU_CHECK_DIM(xDimNumber, PERTOKEN_X_DIM, "x", return false);
        GMM_SWIGLU_CHECK_DIM(xScaleDimNumber, PERTOKEN_X_SCALE_DIM, "xScale", return false);
        GMM_SWIGLU_CHECK_DIM(outputDimNumber, PERTOKEN_OUTPUT_DIM, "output", return false);
        GMM_SWIGLU_CHECK_DIM(outputScaleDimNumber, PERTOKEN_OUTPUT_SCALE_DIM, "outputScale", return false);
        GMM_SWIGLU_CHECK_TENSORNUM(gmmDsqParams_.weight->Size(), SINGLE_TENSOR_SIZE, "weight", return false);
        GMM_SWIGLU_CHECK_TENSORNUM(gmmDsqParams_.weightScale->Size(), SINGLE_TENSOR_SIZE, "weightScale", return false);
        auto weightDimNumber = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDimNum();
        auto weightScaleDimNumber = ((*gmmDsqParams_.weightScale)[0])->GetViewShape().GetDimNum();
        GMM_SWIGLU_CHECK_DIM(weightScaleDimNumber, PERTOKEN_WEIGHT_SCALE_DIM, "weightScale", return false);
        GMM_SWIGLU_CHECK_DIM(weightDimNumber, PERTOKEN_WEIGHT_DIM, "weight", return false);
        return true;
    }

    bool CheckInputOutShape() override
    {
        int64_t groupListLen = gmmDsqParams_.groupList->GetViewShape().GetDim(0);
        if (groupListLen > MAX_GROUP_LIST_SIZE) {
            std::string gotStr = BuildLogValue("length", groupListLen);
            std::ostringstream constraint;
            constraint << "length must be less than or equal to 1024 when "
                       << GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_);
            std::string constraintStr = constraint.str();
            OP_LOGE_FOR_INVALID_LISTSIZE(apiName_.c_str(), "groupList", std::to_string(groupListLen),
                                         "less than or equal to 1024");
            return false;
        }
        // 从x的第1维获取k
        int64_t kInX = gmmDsqParams_.x->GetViewShape().GetDim(1);
        int64_t kInWeight = 0;
        if (IsMxWeightNzMultiTensorSupported()) {
            kInWeight = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(0);
        } else if (gmmDsqParams_.isMxA8W4) {
            auto weightViewDimNum = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDimNum();
            kInWeight = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(weightViewDimNum - LAST_SECOND_DIM_INDEX);
        } else {
            kInWeight = ((*gmmDsqParams_.weight)[0])->GetViewShape().GetDim(1);
        }
        if (kInX != kInWeight) {
            std::ostringstream reason;
            reason << "K dim should be equal when " << GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_)
                   << ", but actual K dims are " << kInX << " and " << kInWeight;
            std::string reasonStr = reason.str();
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                apiName_.c_str(), "x and weight",
                "x=" + ViewShapeToString(gmmDsqParams_.x) + ", weight=" + ViewShapeToString((*gmmDsqParams_.weight)[0]),
                "axis K of x must be equal to axis K of weight");
            return false;
        }
        if (gmmDsqParams_.quantMode == QUNAT_MODE_MX) {
            return CheckInputOutShapeForMX();
        } else if (gmmDsqParams_.quantMode == QUNAT_MODE_PERTOKEN) {
            return CheckInputOutShapeForPertoken();
        } else {
            OP_LOGE_FOR_INVALID_VALUE(apiName_.c_str(), "quantMode", std::to_string(gmmDsqParams_.quantMode), "{0, 2}");
            return false;
        }

        return true;
    }

    bool CheckInputOutShapeForMX()
    {
        if (!CheckMXTranspose()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In op [%s], when %s, CheckMXTranspose failed.", opName_.c_str(),
                    GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_));
            return false;
        }
        if (!CheckMXShape()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In op [%s], when %s, CheckMXShape failed.", opName_.c_str(),
                    GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_));
            return false;
        }
        DataType xDtype = gmmDsqParams_.x->GetDataType();
        DataType weightDtype = ((*gmmDsqParams_.weight)[0])->GetDataType();
        bool isMxfp4 = (xDtype == DataType::DT_FLOAT4_E2M1 || xDtype == DataType::DT_FLOAT4_E1M2) &&
                       (weightDtype == DataType::DT_FLOAT4_E2M1 || weightDtype == DataType::DT_FLOAT4_E1M2);
        if (isMxfp4) {
            return checkMxfp4InputShape();
        }

        bool isMxfp8 = IsFp8Dtype(xDtype) && IsFp8Dtype(weightDtype) &&
                       (gmmDsqParams_.weightScale != nullptr && gmmDsqParams_.weightScale->Size() > 0 &&
                        (*gmmDsqParams_.weightScale)[0] != nullptr &&
                        (*gmmDsqParams_.weightScale)[0]->GetDataType() == DataType::DT_FLOAT8_E8M0);
        if (isMxfp8) {
            return checkMxfp8InputShape();
        }

        if (gmmDsqParams_.isMxA8W4) {
            return CheckMxA8W4InputShape();
        }

        return true;
    }

    bool CheckInputOutShapeForPertoken()
    {
        if (!CheckPertokenTranspose()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In op [%s], when %s, CheckPertokenTranspose failed.", opName_.c_str(),
                    GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_));
            return false;
        }
        if (!CheckPertokenShape()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In op [%s], when %s, CheckPertokenShape failed.", opName_.c_str(),
                    GetGroupedMatmulSwigluQuantV2ScenarioName(gmmDsqParams_));
            return false;
        }
        return true;
    }

    bool CheckWeightFormatConsistency() const
    {
        const auto firstFormat = (*gmmDsqParams_.weight)[0]->GetStorageFormat();
        size_t weightLength = gmmDsqParams_.weight->Size();
        for (size_t i = 1; i < weightLength; ++i) {
            const auto currentFormat = (*gmmDsqParams_.weight)[i]->GetStorageFormat();
            if (currentFormat != firstFormat) {
                const std::string firstFormatStr = op::ToString(firstFormat).GetString();
                OP_LOGE_FOR_INVALID_FORMAT(apiName_.c_str(), "weight", op::ToString(currentFormat).GetString(),
                                           firstFormatStr.c_str());
                return false;
            }
        }
        return true;
    }

    bool CheckDtypeValid() override
    {
        DataType xDtype = gmmDsqParams_.x->GetDataType();
        DataType weightDtype = ((*gmmDsqParams_.weight)[0])->GetDataType();
        const aclTensor *x = gmmDsqParams_.x;
        const aclTensor *weight = (*gmmDsqParams_.weight)[0];
        const aclTensor *xScale = gmmDsqParams_.xScale;
        const aclTensor *groupList = gmmDsqParams_.groupList;
        const aclTensor *output = gmmDsqParams_.output;
        const aclTensor *outputScale = gmmDsqParams_.outputScale;
        if (!CheckWeightFormatConsistency()) {
            return false;
        }
        if (weight->GetStorageFormat() == ge::FORMAT_ND &&
            (weightDtype == DataType::DT_FLOAT4_E1M2 || xDtype == DataType::DT_FLOAT4_E1M2 ||
             output->GetDataType() == DataType::DT_FLOAT4_E1M2)) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                apiName_.c_str(), "weight, x and output",
                "weight=" + std::string(op::ToString(weightDtype).GetString()) + ", x=" +
                    op::ToString(xDtype).GetString() + ", output=" + op::ToString(output->GetDataType()).GetString(),
                "when the format of weight is ND, the dtypes of weight, x and output can not be DT_FLOAT4_E1M2");
            return false;
        }
        const auto &xDtypeSupportListMxfp4 = GetXSupportListMxfp4(weight);
        const auto &weightDtypeSupportListMxfp4 = GetWeightSupportListMxfp4(weight);
        const char *dtypeSupportList = IsMxfp4WeightNzFormat(weight) ?
                                           "{INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1, FLOAT4_E1M2}" :
                                           "{INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1}";
        bool xDtypeNotSupported =
            std::find(X_DTYPE_SUPPORT_LIST.begin(), X_DTYPE_SUPPORT_LIST.end(), xDtype) == X_DTYPE_SUPPORT_LIST.end();
        bool xDtypeNotSupportedMxfp4 = std::find(xDtypeSupportListMxfp4.begin(), xDtypeSupportListMxfp4.end(),
                                                 xDtype) == xDtypeSupportListMxfp4.end();
        bool xDtypeNotSupportedPertoken =
            std::find(XW_DTYPE_SUPPORT_LIST_PERTOKEN.begin(), XW_DTYPE_SUPPORT_LIST_PERTOKEN.end(), xDtype) ==
            XW_DTYPE_SUPPORT_LIST_PERTOKEN.end();
        if (xDtypeNotSupported && xDtypeNotSupportedMxfp4 && xDtypeNotSupportedPertoken) {
            OP_LOGE_FOR_INVALID_DTYPE(apiName_.c_str(), "x", op::ToString(xDtype).GetString(), dtypeSupportList);
            return false;
        }
        bool weightDtypeNotSupported = std::find(WEIGHT_DTYPE_SUPPORT_LIST.begin(), WEIGHT_DTYPE_SUPPORT_LIST.end(),
                                                 weightDtype) == WEIGHT_DTYPE_SUPPORT_LIST.end();
        bool weightDtypeNotSupportedMxfp4 =
            std::find(weightDtypeSupportListMxfp4.begin(), weightDtypeSupportListMxfp4.end(), weightDtype) ==
            weightDtypeSupportListMxfp4.end();
        bool weightDtypeNotSupportedPertoken =
            std::find(XW_DTYPE_SUPPORT_LIST_PERTOKEN.begin(), XW_DTYPE_SUPPORT_LIST_PERTOKEN.end(), weightDtype) ==
            XW_DTYPE_SUPPORT_LIST_PERTOKEN.end();
        if (weightDtypeNotSupported && weightDtypeNotSupportedMxfp4 && weightDtypeNotSupportedPertoken) {
            OP_LOGE_FOR_INVALID_DTYPE(apiName_.c_str(), "weight", op::ToString(weightDtype).GetString(),
                                      dtypeSupportList);
            return false;
        }
        if (gmmDsqParams_.quantMode == QUNAT_MODE_MX &&
            (xDtype == DataType::DT_FLOAT8_E4M3FN || xDtype == DataType::DT_FLOAT8_E5M2) &&
            (weightDtype == DataType::DT_FLOAT8_E4M3FN || weightDtype == DataType::DT_FLOAT8_E5M2)) {
            return CheckFp8DtypeValid(x, xScale, groupList, output, outputScale);
        } else if (gmmDsqParams_.quantMode == QUNAT_MODE_MX &&
                   (xDtype == DataType::DT_FLOAT4_E2M1 || xDtype == DataType::DT_FLOAT4_E1M2) &&
                   (weightDtype == DataType::DT_FLOAT4_E2M1 || weightDtype == DataType::DT_FLOAT4_E1M2)) {
            return CheckFp4DtypeValid(x, xScale, groupList, output, outputScale);
        } else if (gmmDsqParams_.quantMode == QUNAT_MODE_MX && gmmDsqParams_.isMxA8W4) {
            return CheckMxA8W4DtypeValid(x, xScale, groupList, output, outputScale);
        } else if (gmmDsqParams_.quantMode == QUNAT_MODE_PERTOKEN &&
                   std::find(XW_DTYPE_SUPPORT_LIST_PERTOKEN.begin(), XW_DTYPE_SUPPORT_LIST_PERTOKEN.end(), xDtype) !=
                       XW_DTYPE_SUPPORT_LIST_PERTOKEN.end() &&
                   std::find(XW_DTYPE_SUPPORT_LIST_PERTOKEN.begin(), XW_DTYPE_SUPPORT_LIST_PERTOKEN.end(),
                             weightDtype) != XW_DTYPE_SUPPORT_LIST_PERTOKEN.end()) {
            return CheckPertokenDtypeValid(x, xScale, groupList, output, outputScale);
        } else {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                apiName_.c_str(), "x and weight",
                "x=" + std::string(op::ToString(xDtype).GetString()) +
                    ", weight=" + op::ToString(weightDtype).GetString(),
                "the dtypes of x and weight must be within the range quantMode 0: both INT8, both "
                "FLOAT8_E4M3FN/FLOAT8_E5M2, or both HIFLOAT8; quantMode 2: both FLOAT8_E4M3FN/FLOAT8_E5M2 "
                "or MXFP4 dtypes supported by the format of weight");
            return false;
        }
        return true;
    }

    bool CheckFormat() override
    {
        size_t wLength = gmmDsqParams_.weight->Size();
        for (size_t i = 0; i < wLength; i++) {
            const aclTensor *weightScale = (*gmmDsqParams_.weightScale)[i];
            const aclTensor *weight = (*gmmDsqParams_.weight)[i];
            if (gmmDsqParams_.isMxA8W4) {
                if (weight->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ &&
                    weight->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ_C0_32) {
                    OP_LOGE_FOR_INVALID_FORMAT(apiName_.c_str(), "weight",
                                               op::ToString(weight->GetStorageFormat()).GetString(),
                                               "{FORMAT_FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_32}");
                    return false;
                }
            }
            GMM_SWIGLU_CHECK_FORMAT(weightScale, "weightScale", "ND", return false);
        }

        GMM_SWIGLU_CHECK_FORMAT(gmmDsqParams_.x, "x", "ND", return false);
        GMM_SWIGLU_CHECK_FORMAT(gmmDsqParams_.xScale, "xScale", "ND", return false);
        GMM_SWIGLU_CHECK_FORMAT(gmmDsqParams_.groupList, "groupList", "ND", return false);
        GMM_SWIGLU_CHECK_FORMAT(gmmDsqParams_.output, "output", "ND", return false);
        GMM_SWIGLU_CHECK_FORMAT(gmmDsqParams_.outputScale, "outputScale", "ND", return false);
        return true;
    }

    std::vector<op::Shape> weightScaleOriginalShapes_;
};
#undef GMM_SWIGLU_CHECK_FORMAT
#undef GMM_SWIGLU_CHECK_TENSORNUM
#undef GMM_SWIGLU_CHECK_DIM
#undef GMM_SWIGLU_CHECK_DTYPE
#undef GMM_SWIGLU_CHECK_SHAPE
} // namespace gmmSwigluQuantV2
#endif
