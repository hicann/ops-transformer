// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>

#include <c10/core/DeviceGuard.h>

#include <cmath>
#include <string>
#include <vector>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t GE_DTYPE_FLOAT = 0L;
constexpr int64_t GE_DTYPE_FLOAT8_E5M2 = 35L;
constexpr int64_t GE_DTYPE_FLOAT8_E4M3FN = 36L;
constexpr int64_t GE_DTYPE_FLOAT8_E8M0 = 37L;
constexpr int64_t GE_DTYPE_FLOAT4_E2M1 = 40L;
constexpr int64_t GE_DTYPE_FLOAT4_E1M2 = 41L;
constexpr int64_t ACL_DTYPE_OFFSET = 256L;
constexpr int64_t DIM_0 = 0L;
constexpr int64_t DIM_2 = 2L;
constexpr int64_t MX_GROUP_SIZE = 64L;

bool IsKnownGeDtype(int64_t dtype)
{
    return dtype == GE_DTYPE_FLOAT || dtype == GE_DTYPE_FLOAT8_E5M2 || dtype == GE_DTYPE_FLOAT8_E4M3FN ||
           dtype == GE_DTYPE_FLOAT8_E8M0 || dtype == GE_DTYPE_FLOAT4_E2M1 || dtype == GE_DTYPE_FLOAT4_E1M2;
}

int64_t NormalizeGeDtype(int64_t dtype)
{
    if (IsKnownGeDtype(dtype)) {
        return dtype;
    }
    if (dtype >= ACL_DTYPE_OFFSET) {
        return dtype - ACL_DTYPE_OFFSET;
    }
    if (dtype == static_cast<int64_t>(at::ScalarType::Float)) {
        return GE_DTYPE_FLOAT;
    }
    if (dtype == static_cast<int64_t>(at::ScalarType::Float8_e5m2)) {
        return GE_DTYPE_FLOAT8_E5M2;
    }
    if (dtype == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn)) {
        return GE_DTYPE_FLOAT8_E4M3FN;
    }
    return dtype;
}

const char *GetDtypeName(int64_t dtype)
{
    switch (NormalizeGeDtype(dtype)) {
        case GE_DTYPE_FLOAT:
            return "FLOAT";
        case GE_DTYPE_FLOAT8_E5M2:
            return "FLOAT8_E5M2";
        case GE_DTYPE_FLOAT8_E4M3FN:
            return "FLOAT8_E4M3FN";
        case GE_DTYPE_FLOAT8_E8M0:
            return "FLOAT8_E8M0";
        case GE_DTYPE_FLOAT4_E2M1:
            return "FLOAT4_E2M1";
        case GE_DTYPE_FLOAT4_E1M2:
            return "FLOAT4_E1M2";
        default:
            return "UNKNOWN";
    }
}

aclDataType GetAclDataTypeFromValue(int64_t dtype)
{
    int64_t geDtype = NormalizeGeDtype(dtype);
    if (geDtype == GE_DTYPE_FLOAT4_E2M1) {
        return ACL_FLOAT4_E2M1;
    }
    if (geDtype == GE_DTYPE_FLOAT4_E1M2) {
        return ACL_FLOAT4_E1M2;
    }
    if (IsKnownGeDtype(geDtype)) {
        return static_cast<aclDataType>(geDtype);
    }
    return GetAclDataType(dtype);
}

aclDataType GetTensorAclDataType(const c10::optional<int64_t> &dtypeOptional, const at::Tensor &tensor)
{
    if (dtypeOptional.has_value()) {
        return GetAclDataTypeFromValue(dtypeOptional.value());
    }
    return ConvertToAclDataType(tensor.scalar_type());
}

at::ScalarType GetYScalarType(int64_t yDtype)
{
    int64_t geDtype = NormalizeGeDtype(yDtype);
    if (geDtype == GE_DTYPE_FLOAT8_E4M3FN) {
        return at::ScalarType::Float8_e4m3fn;
    }
    if (geDtype == GE_DTYPE_FLOAT8_E5M2) {
        return at::ScalarType::Float8_e5m2;
    }
    if (geDtype == GE_DTYPE_FLOAT4_E2M1 || geDtype == GE_DTYPE_FLOAT4_E1M2) {
        // Torch exposes the packed FP4 result as uint8; TensorWrapper restores the logical FP4 descriptor.
        return at::ScalarType::Byte;
    }
    TORCH_CHECK(false, "y_dtype only supports FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1 or FLOAT4_E1M2, but got ",
                GetDtypeName(yDtype), ".");
}

bool IsFp4GeDtype(int64_t dtype)
{
    const int64_t geDtype = NormalizeGeDtype(dtype);
    return geDtype == GE_DTYPE_FLOAT4_E2M1 || geDtype == GE_DTYPE_FLOAT4_E1M2;
}

int64_t ResolveYGeDtype(const c10::optional<int64_t> &yDtype, aclDataType xAclDtype)
{
    if (yDtype.has_value()) {
        return NormalizeGeDtype(yDtype.value());
    }
    if (xAclDtype == ACL_FLOAT8_E4M3FN) {
        return GE_DTYPE_FLOAT8_E4M3FN;
    }
    if (xAclDtype == ACL_FLOAT8_E5M2) {
        return GE_DTYPE_FLOAT8_E5M2;
    }
    if (xAclDtype == ACL_FLOAT4_E2M1 || xAclDtype == ACL_FLOAT4_E1M2) {
        return xAclDtype == ACL_FLOAT4_E2M1 ? GE_DTYPE_FLOAT4_E2M1 : GE_DTYPE_FLOAT4_E1M2;
    }
    TORCH_CHECK(false, "y_dtype is None only supports inferring from FP8 or FP4 x dtype.");
}

int64_t CeilDiv(int64_t value, int64_t factor)
{
    return (value + factor - 1) / factor;
}

int64_t InferNzLogicalN(const at::Tensor &weightScaleTensor)
{
    // Both the non-transposed scale and the view produced by
    // scaleSource.transpose(-3, -2) are [E, ceil(K / 64), N, 2].
    return weightScaleTensor.size(DIM_2);
}

void CheckTensorListIndexable(const std::vector<at::Tensor> &tensorList, const char *name)
{
    TORCH_CHECK(!tensorList.empty(), name, " must contain at least one tensor for torch output allocation.");
    TORCH_CHECK(tensorList[0].defined(), name, "[0] must be a defined tensor for torch output allocation.");
}
} // namespace

std::tuple<at::Tensor, at::Tensor> grouped_matmul_activation_quant(
    const at::Tensor &x, const at::Tensor &groupList, const std::vector<at::Tensor> &weight,
    const std::vector<at::Tensor> &weightScale, std::string activationType,
    const c10::optional<std::vector<at::Tensor>> &biasOptional, const c10::optional<at::Tensor> &xScaleOptional,
    int64_t groupListType, const c10::optional<std::vector<int64_t>> &tuningConfig,
    c10::optional<c10::string_view> quantModeOptional, c10::optional<int64_t> yDtype, std::string roundMode,
    int64_t scaleAlg, float dstTypeMax, c10::optional<int64_t> xDtype, c10::optional<int64_t> weightDtype,
    c10::optional<int64_t> weightScaleDtype, c10::optional<int64_t> xScaleDtype)
{
    CheckTensorListIndexable(weight, "weight");
    CheckTensorListIndexable(weightScale, "weight_scale");
    TORCH_CHECK(xScaleOptional.has_value() && xScaleOptional.value().defined(),
                "x_scale must be provided for torch wrapper tensor conversion.");

    const at::Tensor &weightTensor = weight[0];
    const at::Tensor &weightScaleTensor = weightScale[0];
    const at::Tensor &xScale = xScaleOptional.value();
    TORCH_CHECK(x.dim() > DIM_0, "x must have at least one dimension for torch output allocation.");
    TORCH_CHECK(weightScaleTensor.dim() > DIM_2,
                "weight_scale must have at least 3 dimensions for torch output allocation.");

    int64_t m = x.size(DIM_0);
    int64_t n = InferNzLogicalN(weightScaleTensor);

    aclDataType xAclDtype = GetTensorAclDataType(xDtype, x);
    aclDataType weightAclDtype = GetTensorAclDataType(weightDtype, weightTensor);
    aclDataType weightScaleAclDtype = GetTensorAclDataType(weightScaleDtype, weightScaleTensor);
    aclDataType xScaleAclDtype = GetTensorAclDataType(xScaleDtype, xScale);
    c10::string_view quantModeView = quantModeOptional.value_or("");
    std::string quantMode(quantModeView.data(), quantModeView.size());
    int64_t yDtypeValue = ResolveYGeDtype(yDtype, xAclDtype);

    at::Tensor y;
    at::Tensor yScale;
    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        const int64_t physicalN = IsFp4GeDtype(yDtypeValue) ? CeilDiv(n, 2L) : n;
        y = at::empty({m, physicalN}, x.options().dtype(GetYScalarType(yDtypeValue)));
        yScale = at::empty({m, CeilDiv(n, MX_GROUP_SIZE), 2}, x.options().dtype(at::ScalarType::Float8_e8m0fnu));
    }

    at::TensorList weightRef = weight;
    at::TensorList weightScaleRef = weightScale;
    const std::vector<at::Tensor> emptyBias;
    const std::vector<at::Tensor> &bias = biasOptional.has_value() ? biasOptional.value() : emptyBias;
    at::TensorList biasRef = bias;
    c10::optional<at::IntArrayRef> tuningConfigRef = c10::nullopt;
    if (tuningConfig.has_value()) {
        tuningConfigRef = at::IntArrayRef(tuningConfig.value());
    }

    TensorWrapper xWrapper = {x, xAclDtype};
    TensorWrapper xScaleWrapper = {xScale, xScaleAclDtype};
    TensorListWrapper weightWrapper = {weightRef, weightAclDtype};
    TensorListWrapper weightScaleWrapper = {weightScaleRef, weightScaleAclDtype};
    TensorWrapper yWrapper = {y, static_cast<aclDataType>(yDtypeValue)};
    TensorWrapper yScaleWrapper = {yScale, ACL_FLOAT8_E8M0};
    char *activationTypePtr = const_cast<char *>(activationType.c_str());
    char *quantModePtr = const_cast<char *>(quantMode.c_str());
    char *roundModePtr = const_cast<char *>(roundMode.c_str());

    ACLNN_CMD(aclnnGroupedMatmulActivationQuantWeightNz, xWrapper, groupList, weightWrapper, weightScaleWrapper,
              biasRef, xScaleWrapper, activationTypePtr, groupListType, tuningConfigRef, quantModePtr, roundModePtr,
              scaleAlg, dstTypeMax, yWrapper, yScaleWrapper);
    return std::tie(y, yScale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grouped_matmul_activation_quant", &grouped_matmul_activation_quant,
          "GroupedMatmulActivationQuant WeightNz torch wrapper");
}
} // namespace op_api
