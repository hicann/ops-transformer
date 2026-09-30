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

#include <string>
#include <tuple>
#include <vector>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t kCeilDivAdjustment = 1;
constexpr int64_t kWeightScaleRank = 4;
constexpr int64_t kSwigluSplitFactor = 2;
constexpr int64_t kMxScalePairSize = 2;
constexpr int64_t kMxQuantMode = 2;
constexpr int64_t kDim0 = 0;
constexpr int64_t kDim2 = 2;
constexpr int64_t kMxGroupSize = 64;
constexpr int64_t kGeDtypeFloat8E4M3Fn = 36;
constexpr int64_t kGeDtypeFloat8E8M0 = 37;
constexpr int64_t kAclDtypeOffset = 256;
constexpr double kDefaultClampLimit = 7.0;
constexpr double kDefaultGluAlpha = 1.702;
constexpr double kDefaultGluBias = 1.0;

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - kCeilDivAdjustment) / divisor;
}

int64_t NormalizeGeDtype(int64_t dtype)
{
    if (dtype == kGeDtypeFloat8E4M3Fn || dtype == kGeDtypeFloat8E8M0) {
        return dtype;
    }
    if (dtype >= kAclDtypeOffset) {
        return dtype - kAclDtypeOffset;
    }
    if (dtype == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn)) {
        return kGeDtypeFloat8E4M3Fn;
    }
    return dtype;
}

aclDataType GetAclDtypeFromValue(int64_t dtype)
{
    const int64_t geDtype = NormalizeGeDtype(dtype);
    if (geDtype == kGeDtypeFloat8E4M3Fn || geDtype == kGeDtypeFloat8E8M0) {
        return static_cast<aclDataType>(geDtype);
    }
    return GetAclDataType(dtype);
}

aclDataType GetTensorAclDtype(const c10::optional<int64_t>& dtype, const at::Tensor& tensor, const char* name)
{
    const aclDataType storageDtype = ConvertToAclDataType(tensor.scalar_type());
    // TensorWrapper changes the descriptor, not the underlying storage. Do not
    // let a dtype override hide an incompatible tensor from the ACLNN checks.
    TORCH_CHECK(!dtype.has_value() || GetAclDtypeFromValue(dtype.value()) == storageDtype, name,
                "_dtype must match the actual tensor dtype.");
    return storageDtype;
}

aclDataType GetMxScaleAclDtype(const c10::optional<int64_t>& dtype, const at::Tensor& tensor, const char* name)
{
    // E8M0 is reinterpreted from its raw bytes, not converted numerically. Both
    // signed and unsigned byte storage preserve that bit pattern.
    TORCH_CHECK(tensor.defined(), name, " must be defined.");
    TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Char || tensor.scalar_type() == at::ScalarType::Byte, name,
                " must use torch.int8 or torch.uint8 storage for torch_npu.float8_e8m0fnu.");
    TORCH_CHECK(!dtype.has_value() || GetAclDtypeFromValue(dtype.value()) == ACL_FLOAT8_E8M0, name,
                "_dtype must describe the E8M0 byte storage.");
    return ACL_FLOAT8_E8M0;
}

void CheckListForIndexing(const std::vector<at::Tensor>& tensors, const char* name)
{
    // ACLNN validates the supported TensorList cardinality. The wrapper only
    // needs its first tensor to infer the descriptor and output shape safely.
    TORCH_CHECK(!tensors.empty(), name, " must not be empty.");
    TORCH_CHECK(tensors[0].defined(), name, "[0] must be defined.");
}

int64_t InferLogicalN(const at::Tensor& weightScale)
{
    TORCH_CHECK(weightScale.dim() == kWeightScaleRank, "weight_scale[0] must be 4D in the V3 MXFP8 scenario, but got ",
                weightScale.dim(), ".");
    // Both the non-transposed scale and the view produced by
    // weightScaleSource.transpose(-3, -2) are [E, ceil(K / 64), N, 2].
    return weightScale.size(kDim2);
}

} // namespace

std::tuple<at::Tensor, at::Tensor> grouped_matmul_swiglu_quant(
    const at::Tensor& x, const std::vector<at::Tensor>& weight, const std::vector<at::Tensor>& weightScale,
    const at::Tensor& xScale, const at::Tensor& groupList, const c10::optional<at::Tensor>& smoothScale,
    const c10::optional<std::vector<at::Tensor>>& weightAssistMatrix,
    const c10::optional<std::vector<at::Tensor>>& bias, c10::optional<int64_t> dequantMode,
    c10::optional<int64_t> dequantDtype, c10::optional<int64_t> quantMode, c10::optional<int64_t> quantDtype,
    c10::optional<int64_t> groupListType, const c10::optional<std::vector<int64_t>>& tuningConfig,
    c10::optional<int64_t> xDtype, c10::optional<int64_t> weightDtype, c10::optional<int64_t> weightScaleDtype,
    c10::optional<int64_t> xScaleDtype, c10::optional<int64_t> swigluMode, c10::optional<double> clampLimit,
    c10::optional<double> gluAlpha, c10::optional<double> gluBias, const std::string& roundMode, int64_t scaleAlg,
    double dstTypeMax)
{
    const double clampLimitValue = clampLimit.value_or(kDefaultClampLimit);
    const double gluAlphaValue = gluAlpha.value_or(kDefaultGluAlpha);
    const double gluBiasValue = gluBias.value_or(kDefaultGluBias);
    CheckListForIndexing(weight, "weight");
    CheckListForIndexing(weightScale, "weight_scale");
    TORCH_CHECK(x.defined() && x.dim() > kDim0, "x must be defined and have a dimension for output shape inference.");
    TORCH_CHECK(swigluMode.has_value(), "swiglu_mode must be specified; use torch_npu for legacy V2 calls.");
    // The Torch schema has a TensorList bias, while ACLNN takes one Tensor.
    // Keep the empty-list bridge explicit instead of silently discarding input.
    TORCH_CHECK(!bias.has_value() || bias->empty(), "The V3 wrapper requires an empty bias TensorList.");
    // ACLNN has no quant_dtype attribute: the caller provides the output tensor.
    // Its dtype must agree with the FP8 allocation below.
    TORCH_CHECK(!quantDtype.has_value() || NormalizeGeDtype(quantDtype.value()) == kGeDtypeFloat8E4M3Fn,
                "swiglu_mode=2 only supports torch.float8_e4m3fn output.");
    // ACLNN takes a C string: reject embedded NUL bytes instead of silently
    // validating only the prefix of the Python/std::string value.
    TORCH_CHECK(roundMode.find('\0') == std::string::npos, "round_mode must not contain embedded NUL bytes.");

    const int64_t m = x.size(kDim0);
    const int64_t n = InferLogicalN(weightScale[0]);
    TORCH_CHECK(n > 0 && n % kSwigluSplitFactor == 0, "The logical N dimension must be positive and even, but got ", n,
                ".");
    const aclDataType xAclDtype = GetTensorAclDtype(xDtype, x, "x");
    const aclDataType weightAclDtype = GetTensorAclDtype(weightDtype, weight[0], "weight");
    const aclDataType weightScaleAclDtype = GetMxScaleAclDtype(weightScaleDtype, weightScale[0], "weight_scale");
    const aclDataType xScaleAclDtype = GetMxScaleAclDtype(xScaleDtype, xScale, "x_scale");

    at::Tensor output;
    at::Tensor outputScale;
    {
        const c10::OptionalDeviceGuard deviceGuard(c10::Device(x.device()));
        output = at::empty({m, n / kSwigluSplitFactor}, x.options().dtype(at::ScalarType::Float8_e4m3fn));
        // The MX scale writer can leave padding/empty lanes untouched. Clear
        // the output tensor first so every externally visible lane is
        // deterministic.
        // ZerosLike does not support float8_e8m0fnu. Allocate byte storage,
        // initialize it, then reinterpret the one-byte elements as E8M0.
        outputScale = at::zeros({m, CeilDiv(n / kSwigluSplitFactor, kMxGroupSize), kMxScalePairSize},
                                x.options().dtype(at::ScalarType::Byte))
                          .view(at::ScalarType::Float8_e8m0fnu);
    }

    at::TensorList weightRef = weight;
    at::TensorList weightScaleRef = weightScale;
    const std::vector<at::Tensor> emptyAssist;
    const std::vector<at::Tensor>& assist = weightAssistMatrix.has_value() ? weightAssistMatrix.value() : emptyAssist;
    at::TensorList assistRef = assist;
    // V3 currently accepts only an empty bias TensorList. ACLNN V3 keeps its
    // optional single-tensor ABI, so pass an undefined tensor.
    const at::Tensor biasRef = at::Tensor();
    const at::Tensor smoothScaleRef = smoothScale.value_or(at::Tensor());
    c10::optional<at::IntArrayRef> tuningConfigRef = c10::nullopt;
    if (tuningConfig.has_value()) {
        tuningConfigRef = at::IntArrayRef(tuningConfig.value());
    }

    TensorWrapper xWrapper = {x, xAclDtype};
    TensorListWrapper weightWrapper = {weightRef, weightAclDtype};
    TensorListWrapper weightScaleWrapper = {weightScaleRef, weightScaleAclDtype};
    TensorWrapper xScaleWrapper = {xScale, xScaleAclDtype};
    TensorWrapper outputWrapper = {output, ACL_FLOAT8_E4M3FN};
    TensorWrapper outputScaleWrapper = {outputScale, ACL_FLOAT8_E8M0};
    const char* roundModePtr = roundMode.c_str();
    // ACLNN_CMD forwards every argument through the ACL type converter, whose
    // interface takes non-const lvalue references.  Keep the scalar attributes
    // in named variables instead of passing value_or() / temporary expressions.
    int64_t dequantModeValue = dequantMode.value_or(kMxQuantMode);
    // Preserve the legacy Torch FP32 enum alias, but forward every other value
    // unchanged so ACLNN, not this bridge, validates the dequantization mode.
    int64_t dequantDtypeValue = dequantDtype.value_or(static_cast<int64_t>(ACL_FLOAT));
    if (dequantDtypeValue == static_cast<int64_t>(at::ScalarType::Float)) {
        dequantDtypeValue = static_cast<int64_t>(ACL_FLOAT);
    }
    int64_t quantModeValue = quantMode.value_or(kMxQuantMode);
    int64_t groupListTypeValue = groupListType.value_or(0);
    int64_t swigluModeValue = swigluMode.value();
    double clampLimitArg = clampLimitValue;
    double gluAlphaArg = gluAlphaValue;
    double gluBiasArg = gluBiasValue;
    int64_t scaleAlgValue = scaleAlg;
    double dstTypeMaxArg = dstTypeMax;

    ACLNN_CMD(aclnnGroupedMatmulSwigluQuantWeightNzV3, xWrapper, weightWrapper, weightScaleWrapper, assistRef, biasRef,
              xScaleWrapper, smoothScaleRef, groupList, dequantModeValue, dequantDtypeValue, quantModeValue,
              groupListTypeValue, tuningConfigRef, swigluModeValue, clampLimitArg, gluAlphaArg, gluBiasArg,
              roundModePtr, scaleAlgValue, dstTypeMaxArg, outputWrapper, outputScaleWrapper);
    return std::tie(output, outputScale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grouped_matmul_swiglu_quant", &grouped_matmul_swiglu_quant,
          "GroupedMatmulSwigluQuant V3 MXFP8 WeightNz torch wrapper");
}
} // namespace op_api
