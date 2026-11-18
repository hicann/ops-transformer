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

#include <tuple>
#include <vector>
#include <string>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t kDim0 = 0L;
constexpr int64_t kDim2 = 2L;
constexpr int64_t kMatrixRank = 2L;
constexpr int64_t kWeightRank = 3L;
constexpr int64_t kWeightScaleRank = 4L;
constexpr int64_t kSwigluSplitFactor = 2L;
constexpr int64_t kMxAlignment = 128L;
constexpr int64_t kMxDequantMode = 2L;
constexpr int64_t kFloatDequantDtype = 0L;
constexpr char kSwigluActType[] = "swiglu";

aclDataType ResolveInputDtype(const at::Tensor& tensor, const c10::optional<int64_t>& dtype, aclDataType expected,
                              const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined.");
    const aclDataType resolved = dtype.has_value() ? GetAclDataType(dtype.value()) : expected;
    TORCH_CHECK(resolved == expected, name, " specifies an unsupported logical dtype.");
    const bool isByteStorage =
        tensor.scalar_type() == at::ScalarType::Char || tensor.scalar_type() == at::ScalarType::Byte;
    TORCH_CHECK(isByteStorage || ConvertToAclDataType(tensor.scalar_type()) == expected, name,
                " must use byte storage or the matching native dtype.");
    return resolved;
}

void CheckSingleTensorList(const std::vector<at::Tensor>& tensors, const char* name)
{
    TORCH_CHECK(tensors.size() == 1UL && tensors[0].defined(), name, " must contain exactly one defined tensor.");
}
} // namespace

at::Tensor grouped_matmul_activation(const at::Tensor& x, const at::Tensor& xScale, const at::Tensor& groupList,
                                     const std::vector<at::Tensor>& weight, const std::vector<at::Tensor>& weightScale,
                                     const c10::optional<std::vector<at::Tensor>>& bias,
                                     const std::string& activationType, int64_t groupListType,
                                     const c10::optional<std::vector<int64_t>>& tuningConfig, int64_t swigluMode,
                                     double clampLimit, double gluAlpha, double gluBias,
                                     const c10::optional<int64_t>& xDtype, const c10::optional<int64_t>& weightDtype,
                                     const c10::optional<int64_t>& xScaleDtype,
                                     const c10::optional<int64_t>& weightScaleDtype)
{
    CheckSingleTensorList(weight, "weight");
    CheckSingleTensorList(weightScale, "weight_scale");
    const aclDataType xAclDtype = ResolveInputDtype(x, xDtype, ACL_FLOAT4_E2M1, "x");
    const aclDataType weightAclDtype = ResolveInputDtype(weight[0], weightDtype, ACL_FLOAT4_E2M1, "weight[0]");
    const aclDataType xScaleAclDtype = ResolveInputDtype(xScale, xScaleDtype, ACL_FLOAT8_E8M0, "x_scale");
    const aclDataType weightScaleAclDtype =
        ResolveInputDtype(weightScale[0], weightScaleDtype, ACL_FLOAT8_E8M0, "weight_scale[0]");
    TORCH_CHECK(groupList.scalar_type() == at::ScalarType::Long, "group_list must have torch.int64 dtype.");
    TORCH_CHECK(!bias.has_value() || bias->empty(),
                "bias must be None or an empty TensorList for the current A4W4 path.");
    TORCH_CHECK(x.dim() == kMatrixRank && weight[0].dim() == kWeightRank && weightScale[0].dim() == kWeightScaleRank,
                "x must be 2D, weight[0] must be 3D and weight_scale[0] must be 4D.");
    TORCH_CHECK(torch_npu::utils::is_npu(x), "x must be an NPU tensor.");
    TORCH_CHECK(
        xScale.device() == x.device() && weight[0].device() == x.device() && weightScale[0].device() == x.device(),
        "x_scale, weight[0] and weight_scale[0] must be on the same NPU device as x.");
    TORCH_CHECK(groupList.device().is_cpu() || groupList.device() == x.device(),
                "group_list must be on CPU or the same NPU device as x.");
    TORCH_CHECK(activationType == kSwigluActType, "only activation_type='swiglu' is supported.");

    const int64_t m = x.size(kDim0);
    const int64_t logicalN = weightScale[0].size(kDim2);
    TORCH_CHECK(logicalN > 0L && logicalN % kSwigluSplitFactor == 0L && logicalN % kMxAlignment == 0L,
                "A4W4 requires logical N to be divisible by 128.");
    at::Tensor groupListForOp = groupList;
    if (groupList.device().is_cpu()) {
        TORCH_CHECK(groupList.dim() == 1, "group_list must be 1D.");
        TORCH_CHECK(groupListType == 0 || groupListType == 1, "group_list_type must be 0 or 1.");
        auto contiguousGroup = groupList.contiguous();
        const int64_t* values = contiguousGroup.data_ptr<int64_t>();
        int64_t previous = 0;
        int64_t total = 0;
        for (int64_t index = 0; index < contiguousGroup.numel(); ++index) {
            const int64_t value = values[index];
            TORCH_CHECK(value >= 0, "group_list values must be non-negative.");
            if (groupListType == 0) {
                TORCH_CHECK(value >= previous && value <= m, "group_list must be non-decreasing and not exceed M.");
                previous = value;
            } else {
                TORCH_CHECK(value <= m - total, "group_list counts must not exceed M.");
                total += value;
            }
        }
        TORCH_CHECK((groupListType == 0 ? previous : total) == m, "group_list must cover exactly M rows.");
        groupListForOp = contiguousGroup.to(x.device());
    }
    at::Tensor y;
    {
        const c10::OptionalDeviceGuard deviceGuard(c10::Device(x.device()));
        y = at::empty({m, logicalN / kSwigluSplitFactor}, x.options().dtype(at::ScalarType::BFloat16));
    }

    at::TensorList weightRef = weight;
    at::TensorList weightScaleRef = weightScale;
    const std::vector<at::Tensor> emptyBias;
    const std::vector<at::Tensor>& biasValue = bias.has_value() ? bias.value() : emptyBias;
    at::TensorList biasRef = biasValue;
    c10::optional<at::IntArrayRef> tuningConfigRef = c10::nullopt;
    if (tuningConfig.has_value()) {
        tuningConfigRef = at::IntArrayRef(tuningConfig.value());
    }
    TensorWrapper xWrapper = {x, xAclDtype};
    TensorWrapper xScaleWrapper = {xScale, xScaleAclDtype};
    TensorListWrapper weightWrapper = {weightRef, weightAclDtype};
    TensorListWrapper weightScaleWrapper = {weightScaleRef, weightScaleAclDtype};
    TensorListWrapper biasWrapper = {biasRef, ACL_BF16};
    TensorWrapper yWrapper = {y, ACL_BF16};

    ACLNN_CMD(aclnnGroupedMatmulActivationWeightNz, xWrapper, xScaleWrapper, groupListForOp, weightWrapper,
              weightScaleWrapper, biasWrapper, kMxDequantMode, kFloatDequantDtype, activationType, groupListType,
              tuningConfigRef, swigluMode, clampLimit, gluAlpha, gluBias, yWrapper);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grouped_matmul_activation", &grouped_matmul_activation, "GroupedMatmulActivation A4W4 torch wrapper");
}
} // namespace op_api
