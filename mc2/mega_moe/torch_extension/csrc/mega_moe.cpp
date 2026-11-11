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
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include "aclnn_common.h"
#if __has_include("aclnnop/mega_moe_gm_layout.h")
#include "aclnnop/mega_moe_gm_layout.h"
#else
#include "../../op_kernel/arch35_92/common/mega_moe_gm_layout.h"
#endif

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;
const int64_t PACKED_FLOAT4_ELEMENTS_PER_BYTE = 2;
constexpr int64_t GE_DTYPE_UNDEFINED = 28;
constexpr int64_t INT4_PER_INT32 = 8;

struct MegaMoeWeightList {
    at::TensorList tensors;
    aclDataType dtype;
    bool packedInt4;
};

// Only arch22 packed INT32 weights need a different ACL descriptor. Keep the
// original storage; the last logical dimension contains eight INT4s per INT32.
aclTensorList* ConvertType(const MegaMoeWeightList& weights)
{
    if (!weights.packedInt4) {
        return ::ConvertType(TensorListWrapper{weights.tensors, weights.dtype});
    }
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensor == nullptr || aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor*> tensors;
    tensors.reserve(weights.tensors.size());
    for (const auto& tensor : weights.tensors) {
        TORCH_CHECK(tensor.scalar_type() == at::kInt && tensor.dim() > 0 && tensor.is_contiguous(),
                    "arch22 packed INT4 weights must be contiguous INT32 tensors.");
        TORCH_CHECK(at_npu::native::get_npu_format(tensor) == ACL_FORMAT_FRACTAL_NZ,
                    "arch22 packed INT4 weights must use FRACTAL_NZ.");
        auto shape = tensor.sizes().vec();
        shape.back() *= INT4_PER_INT32;
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size() - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * shape[i];
        }

        // Match the former CreateView: logical storage shape, contiguous strides,
        // and the original view offset, while keeping the same backing memory.
        auto* desc = aclCreateTensor(shape.data(), shape.size(), ACL_INT4, strides.data(), tensor.storage_offset(),
                                     ACL_FORMAT_FRACTAL_NZ, shape.data(), shape.size(),
                                     const_cast<void*>(tensor.storage().data()));
        TORCH_CHECK(desc != nullptr, "Failed to create packed INT4 weight descriptor.");
        tensors.push_back(desc);
    }
    return aclCreateTensorList(tensors.data(), tensors.size());
}

static MegaMoeWeightList MakeWeightList(at::TensorList tensors, aclDataType dtype, bool isArch22)
{
    bool packedInt4 = isArch22 && !tensors.empty() && tensors[0].scalar_type() == at::kInt &&
                      (dtype == ACL_INT32 || dtype == ACL_INT4);
    return {tensors, packedInt4 ? ACL_INT4 : dtype, packedInt4};
}

static void CheckNpuInput(const at::Tensor& tensor, const std::string& name, const char* socName)
{
    TORCH_CHECK(tensor.defined(), name, " must not be None on ", socName, ".");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU on ", socName, ", but got ", tensor.device());
}

static void CheckNpuInput(const std::vector<at::Tensor>& tensors, const std::string& name, const char* socName)
{
    for (size_t i = 0; i < tensors.size(); ++i) {
        CheckNpuInput(tensors[i], name + "[" + std::to_string(i) + "]", socName);
    }
}

template <typename T>
static void CheckNpuInput(const c10::optional<T>& input, const std::string& name, const char* socName)
{
    if (input.has_value()) {
        CheckNpuInput(input.value(), name, socName);
    }
}

template <typename T>
static void CheckInputAbsent(const c10::optional<T>& input, const std::string& name, const char* socName)
{
    TORCH_CHECK(!input.has_value(), name, " must be None on ", socName, ".");
}

static void CheckWeightType(const c10::optional<int64_t>& weightType, const char* name, const char* socName)
{
    if (!weightType.has_value()) {
        return;
    }
    const int64_t value = weightType.value();
    TORCH_CHECK(value == static_cast<int64_t>(at::ScalarType::Float8_e5m2) ||
                    value == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn) ||
                    value == static_cast<int64_t>(DType::FLOAT4_E2M1),
                name, " must be None, float8_e5m2 (", static_cast<int64_t>(at::ScalarType::Float8_e5m2),
                "), float8_e4m3fn (", static_cast<int64_t>(at::ScalarType::Float8_e4m3fn), ") or float4_e2m1 (",
                static_cast<int64_t>(DType::FLOAT4_E2M1), ") on ", socName, ", but got ", value, ".");
}

static void CheckWeightDtype(at::TensorList weights, aclDataType expectedDtype, const char* name, const char* socName)
{
    if (weights.empty()) {
        return;
    }
    const char* expectedName = nullptr;
    switch (expectedDtype) {
        case aclDataType::ACL_FLOAT8_E5M2:
            expectedName = "float8_e5m2";
            break;
        case aclDataType::ACL_FLOAT8_E4M3FN:
            expectedName = "float8_e4m3fn";
            break;
        case aclDataType::ACL_FLOAT4_E2M1:
            expectedName = "uint8 (packed FP4)";
            break;
        default:
            TORCH_CHECK(false, name, "[0] has unsupported dtype ", weights[0].scalar_type(), " on ", socName,
                        ". Without an explicit weight type, only float8_e5m2 and float8_e4m3fn are supported. "
                        "For uint8 packed FP4, specify the corresponding weight type as ",
                        static_cast<int64_t>(DType::FLOAT4_E2M1), ".");
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        const auto dtype = weights[i].scalar_type();
        const bool matches =
            (expectedDtype == aclDataType::ACL_FLOAT8_E5M2 && dtype == at::ScalarType::Float8_e5m2) ||
            (expectedDtype == aclDataType::ACL_FLOAT8_E4M3FN && dtype == at::ScalarType::Float8_e4m3fn) ||
            (expectedDtype == aclDataType::ACL_FLOAT4_E2M1 && dtype == at::ScalarType::Byte);
        TORCH_CHECK(matches, name, "[", i, "] dtype must be ", expectedName, " on ", socName, ", but got ", dtype, ".");
    }
}

struct MegaMoeTensorInputs {
    const at::Tensor& context;
    const at::Tensor& x;
    const at::Tensor& topkIds;
    const at::Tensor& topkWeights;
    const std::vector<at::Tensor>& weight1;
    const std::vector<at::Tensor>& weight2;
    const c10::optional<std::vector<at::Tensor>>& weightScales1;
    const c10::optional<std::vector<at::Tensor>>& weightScales2;
    const c10::optional<std::vector<at::Tensor>>& bias1;
    const c10::optional<std::vector<at::Tensor>>& bias2;
    const c10::optional<std::vector<at::Tensor>>& sharedWeight1;
    const c10::optional<std::vector<at::Tensor>>& sharedWeight2;
    const c10::optional<std::vector<at::Tensor>>& sharedWeightScales1;
    const c10::optional<std::vector<at::Tensor>>& sharedWeightScales2;
    const c10::optional<std::vector<at::Tensor>>& sharedBias1;
    const c10::optional<std::vector<at::Tensor>>& sharedBias2;
    const c10::optional<at::Tensor>& xActiveMask;
    const c10::optional<at::Tensor>& scales;
    const c10::optional<at::Tensor>& maskBuffer;
};

// MX 平台的存储类型校验；量化模式及类型组合仍由 tiling 校验。
static void CheckMegaMoeMxXInput(const at::Tensor& x, const char* socName)
{
    CheckNpuInput(x, "x", socName);
    const auto xScalarType = x.scalar_type();
    const bool isPreQuantizedStorage = xScalarType == at::ScalarType::Float8_e5m2 ||
                                       xScalarType == at::ScalarType::Float8_e4m3fn || xScalarType == at::kByte;
    const bool isSupportedX = xScalarType == at::kBFloat16 || xScalarType == at::kHalf || isPreQuantizedStorage;
    TORCH_CHECK(isSupportedX, "dtype of x should be bfloat16, float16, float8_e5m2, float8_e4m3fn or uint8 on ",
                socName, ", but got ", xScalarType, ".");
}

static void CheckMegaMoeMxInputs(const MegaMoeTensorInputs& inputs, int64_t epWorldSize, int64_t moeExpertNum,
                                 const char* socName)
{
    TORCH_CHECK(moeExpertNum >= epWorldSize && moeExpertNum <= 2048 && moeExpertNum % epWorldSize == 0,
                "num_experts must be in [ep_world_size, 2048] and divisible by ep_world_size on ", socName,
                ", but got num_experts=", moeExpertNum, " and ep_world_size=", epWorldSize);
    TORCH_CHECK(!inputs.weight1.empty(), "l1_weights must not be empty on ", socName, ".");
    TORCH_CHECK(!inputs.weight2.empty(), "l2_weights must not be empty on ", socName, ".");
    TORCH_CHECK(inputs.weightScales1.has_value() && !inputs.weightScales1->empty(),
                "l1_weights_sf must not be None or empty on ", socName, ".");
    TORCH_CHECK(inputs.weightScales2.has_value() && !inputs.weightScales2->empty(),
                "l2_weights_sf must not be None or empty on ", socName, ".");
    CheckInputAbsent(inputs.xActiveMask, "x_active_mask", socName);
    CheckInputAbsent(inputs.bias1, "l1_bias", socName);
    CheckInputAbsent(inputs.bias2, "l2_bias", socName);
    CheckInputAbsent(inputs.sharedBias1, "shared_l1_bias", socName);
    CheckInputAbsent(inputs.sharedBias2, "shared_l2_bias", socName);
    CheckMegaMoeMxXInput(inputs.x, socName);
    CheckNpuInput(inputs.context, "context", socName);
    CheckNpuInput(inputs.topkIds, "topk_ids", socName);
    CheckNpuInput(inputs.topkWeights, "topk_weights", socName);
    CheckNpuInput(inputs.weight1, "l1_weights", socName);
    CheckNpuInput(inputs.weight2, "l2_weights", socName);
    CheckNpuInput(inputs.weightScales1, "l1_weights_sf", socName);
    CheckNpuInput(inputs.weightScales2, "l2_weights_sf", socName);
    CheckNpuInput(inputs.sharedWeight1, "shared_l1_weights", socName);
    CheckNpuInput(inputs.sharedWeight2, "shared_l2_weights", socName);
    CheckNpuInput(inputs.sharedWeightScales1, "shared_l1_weights_sf", socName);
    CheckNpuInput(inputs.sharedWeightScales2, "shared_l2_weights_sf", socName);
    CheckNpuInput(inputs.scales, "scales", socName);
    CheckNpuInput(inputs.maskBuffer, "mask_buffer", socName);
    const auto checkScaleDtype = [socName](const c10::optional<std::vector<at::Tensor>>& scales, const char* name) {
        if (!scales.has_value()) {
            return;
        }
        for (size_t i = 0; i < scales->size(); ++i) {
            const auto dtype = (*scales)[i].scalar_type();
            TORCH_CHECK(dtype == at::ScalarType::Float8_e8m0fnu, name, "[", i, "] dtype must be float8_e8m0fnu on ",
                        socName, ", but got ", dtype, ".");
        }
    };
    checkScaleDtype(inputs.weightScales1, "l1_weights_sf");
    checkScaleDtype(inputs.weightScales2, "l2_weights_sf");
    checkScaleDtype(inputs.sharedWeightScales1, "shared_l1_weights_sf");
    checkScaleDtype(inputs.sharedWeightScales2, "shared_l2_weights_sf");
}

static void CheckMegaMoeInputs(const MegaMoeTensorInputs& inputs, int64_t epWorldSize, int64_t moeExpertNum,
                               const char* socName, bool isMxPlatform)
{
    TORCH_CHECK((epWorldSize > 0), "The ep_world_sizes should be greater than 0, current is: ", epWorldSize);
    if (isMxPlatform) {
        CheckMegaMoeMxInputs(inputs, epWorldSize, moeExpertNum, socName);
    }
    TORCH_CHECK((inputs.x.dim() == DIM_TWO) && (inputs.topkIds.dim() == DIM_TWO), "The x and topk_ids should be 2D");
    if (!isMxPlatform) {
        TORCH_CHECK(inputs.x.scalar_type() == at::kBFloat16 || inputs.x.scalar_type() == at::kHalf,
                    "dtype of x should be bfloat16 or float16.");
    }
    TORCH_CHECK(inputs.topkIds.scalar_type() == at::kInt, "dtype of topk_ids should be int.");
    if (inputs.maskBuffer.has_value()) {
        const at::Tensor& mask = inputs.maskBuffer.value();
        TORCH_CHECK(mask.scalar_type() == at::kInt, "mask_buffer dtype must be int32.");
        TORCH_CHECK(mask.dim() == 1 && mask.numel() == epWorldSize, "mask_buffer shape must be [ep_world_size].");
        TORCH_CHECK(mask.device() == inputs.x.device(), "mask_buffer must be on the same device as x.");
        TORCH_CHECK(mask.is_contiguous(), "mask_buffer must be contiguous.");
    }
}

std::tuple<at::Tensor, at::Tensor> NpuMegaMoe(
    const at::Tensor& context, const at::Tensor& x, const at::Tensor& topkIds, const at::Tensor& topkWeights,
    const std::vector<at::Tensor>& weight1, const std::vector<at::Tensor>& weight2, int64_t moeExpertNum,
    int64_t epWorldSize, int64_t cclBufferSize, const c10::optional<std::vector<at::Tensor>>& weightScales1,
    const c10::optional<std::vector<at::Tensor>>& weightScales2, const c10::optional<std::vector<at::Tensor>>& bias1,
    const c10::optional<std::vector<at::Tensor>>& bias2, const c10::optional<at::Tensor>& xActiveMask,
    const c10::optional<at::Tensor>& scales, const c10::optional<std::vector<at::Tensor>>& sharedWeight1,
    const c10::optional<std::vector<at::Tensor>>& sharedWeight2,
    const c10::optional<std::vector<at::Tensor>>& sharedWeightScales1,
    const c10::optional<std::vector<at::Tensor>>& sharedWeightScales2,
    const c10::optional<std::vector<at::Tensor>>& sharedBias1,
    const c10::optional<std::vector<at::Tensor>>& sharedBias2, const c10::optional<at::Tensor>& maskBuffer,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t combineQuantMode, std::string commAlg,
    int64_t numMaxTokensPerRank, std::string activation, std::vector<float> activationParams,
    c10::optional<int64_t> dispatchQuantOutDtype, c10::optional<int64_t> sharedExpertQuantOutDtype,
    c10::optional<int64_t> weight1Type, c10::optional<int64_t> weight2Type, c10::optional<int64_t> sharedWeight1Type,
    c10::optional<int64_t> sharedWeight2Type, c10::optional<int64_t> topoType, c10::optional<int64_t> rankNumPerServer,
    int64_t topkWeightsType, int64_t combineCommMode)
{
    const MegaMoeTensorInputs inputs{context,
                                     x,
                                     topkIds,
                                     topkWeights,
                                     weight1,
                                     weight2,
                                     weightScales1,
                                     weightScales2,
                                     bias1,
                                     bias2,
                                     sharedWeight1,
                                     sharedWeight2,
                                     sharedWeightScales1,
                                     sharedWeightScales2,
                                     sharedBias1,
                                     sharedBias2,
                                     xActiveMask,
                                     scales,
                                     maskBuffer};
    const c10::OptionalDeviceGuard deviceGuard(x.device());
    const char* socName = aclrtGetSocName();
    const bool isArch22 = socName != nullptr && (std::strstr(socName, "Ascend910B") != nullptr ||
                                                 std::strstr(socName, "Ascend910_93") != nullptr);
    const bool isMxPlatform = socName != nullptr && (std::strstr(socName, "Ascend950") != nullptr ||
                                                     std::strstr(socName, "Ascend960DT") != nullptr);
    CheckMegaMoeInputs(inputs, epWorldSize, moeExpertNum, socName, isMxPlatform);
    if (isMxPlatform) {
        CheckWeightType(weight1Type, "weight1_type", socName);
        CheckWeightType(weight2Type, "weight2_type", socName);
        CheckWeightType(sharedWeight1Type, "shared_weight1_type", socName);
        CheckWeightType(sharedWeight2Type, "shared_weight2_type", socName);
    }

    const auto xScalarType = x.scalar_type();
    const bool hasDispatchQuantOutDtype =
        dispatchQuantOutDtype.has_value() && dispatchQuantOutDtype.value() != GE_DTYPE_UNDEFINED;
    const bool hasSharedExpertQuantOutDtype =
        sharedExpertQuantOutDtype.has_value() && sharedExpertQuantOutDtype.value() != GE_DTYPE_UNDEFINED;
    aclDataType dispatchQuantAclDtype =
        hasDispatchQuantOutDtype ? GetAclDataType(dispatchQuantOutDtype.value()) : ACL_DT_UNDEFINED;
    // 当前 Torch C++ ScalarType 未统一提供 FP4 枚举；只有协议明确声明 FP4 时，uint8 才承载两个 E2M1 元素。
    const bool isPackedFp4X = dispatchQuantMode == 0 && dispatchQuantAclDtype == ACL_FLOAT4_E2M1;
    if (isPackedFp4X) {
        // TensorWrapper 会隐藏原始 Torch dtype，因此重解释前必须确认物理存储确实是 uint8。
        TORCH_CHECK(xScalarType == at::kByte, "FP4 pre-quantized x must use packed torch.uint8 storage.");
    }

    at::TensorList weight1Ref = weight1;
    at::TensorList weight2Ref = weight2;

    auto toTensorList = [](const c10::optional<std::vector<at::Tensor>>& opt) -> at::TensorList {
        return opt.has_value() ? at::TensorList(opt.value()) : at::TensorList();
    };
    at::TensorList weightScales1Ref = toTensorList(weightScales1);
    at::TensorList weightScales2Ref = toTensorList(weightScales2);
    at::TensorList bias1Ref = toTensorList(bias1);
    at::TensorList bias2Ref = toTensorList(bias2);

    aclDataType weight1RefDtype = weight1Type.has_value() ? GetAclDataType(weight1Type.value()) :
                                                            ConvertToAclDataType(weight1Ref[0].scalar_type());
    aclDataType weightScales1Dtype;
    if (weight1RefDtype == aclDataType::ACL_FLOAT8_E5M2 || weight1RefDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight1RefDtype == aclDataType::ACL_FLOAT4_E2M1) {
        weightScales1Dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weightScales1Dtype = aclDataType::ACL_UINT64;
    }

    aclDataType weight2RefDtype = weight2Type.has_value() ? GetAclDataType(weight2Type.value()) :
                                                            ConvertToAclDataType(weight2Ref[0].scalar_type());
    aclDataType weightScales2Dtype;
    if (weight2RefDtype == aclDataType::ACL_FLOAT8_E5M2 || weight2RefDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight2RefDtype == aclDataType::ACL_FLOAT4_E2M1) {
        weightScales2Dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weightScales2Dtype = aclDataType::ACL_UINT64;
    }

    auto xSize = x.sizes();
    int64_t bs = xSize[0];
    int64_t h = xSize[1];

    const bool dispatchQuantOutIsFp4 =
        dispatchQuantOutDtype.has_value() && dispatchQuantOutDtype.value() == static_cast<int64_t>(DType::FLOAT4_E2M1);
    const bool sharedQuantOutIsFp4 = hasSharedExpertQuantOutDtype ?
                                         sharedExpertQuantOutDtype.value() == static_cast<int64_t>(DType::FLOAT4_E2M1) :
                                         dispatchQuantOutIsFp4;
    if (dispatchQuantOutIsFp4 || sharedQuantOutIsFp4) {
        TORCH_CHECK(h % 2 == 0, "The last dim input shape must be divisible by 2 if "
                                "an expert quant output type is torch_npu.float4_e2m1");
    }

    int64_t localMoeExpertNum = 1;
    localMoeExpertNum = moeExpertNum / epWorldSize;
    at::Tensor expertTokenNums;
    expertTokenNums = at::empty({localMoeExpertNum}, x.options().dtype(at::kInt));

    std::string commAlgStr = std::string(commAlg);
    char* commAlgPtr = const_cast<char*>(commAlg.c_str());

    std::string activationStr = std::string(activation);
    char* activationPtr = const_cast<char*>(activationStr.c_str());

    int64_t topoTypeValue = topoType.value_or(0);
    int64_t rankNumPerServerValue = rankNumPerServer.value_or(2);

    // 这里传给 ACLNN 的是 GE dtype 属性；GE DT_UNDEFINED 为 28，与 ACL_DT_UNDEFINED=-1 不同。
    int64_t dispatchQuantResultType = hasDispatchQuantOutDtype ?
                                          static_cast<int64_t>(GetAclDataType(dispatchQuantOutDtype.value())) :
                                          GE_DTYPE_UNDEFINED;
    int64_t sharedExpertQuantResultType = hasSharedExpertQuantOutDtype ?
                                              static_cast<int64_t>(GetAclDataType(sharedExpertQuantOutDtype.value())) :
                                              dispatchQuantResultType;

    const bool isPreQuantizedDtype = dispatchQuantAclDtype == ACL_FLOAT8_E5M2 ||
                                     dispatchQuantAclDtype == ACL_FLOAT8_E4M3FN ||
                                     dispatchQuantAclDtype == ACL_FLOAT4_E2M1;
    const bool isPreQuantizedX = dispatchQuantMode == 0 && isPreQuantizedDtype;
    at::ScalarType outputScalarType = isPreQuantizedX ? at::kBFloat16 : xScalarType;
    if (isPackedFp4X) {
        h *= PACKED_FLOAT4_ELEMENTS_PER_BYTE;
    }
    at::Tensor y = at::empty({bs, h}, x.options().dtype(outputScalarType));

    aclDataType xAclDtype = isPackedFp4X ? ACL_FLOAT4_E2M1 : ConvertToAclDataType(xScalarType);
    TensorWrapper xWrapper = {x, xAclDtype};

    if (isMxPlatform) {
        CheckWeightDtype(weight1Ref, weight1RefDtype, "l1_weights", socName);
        CheckWeightDtype(weight2Ref, weight2RefDtype, "l2_weights", socName);
    }

    MegaMoeWeightList weight1Wrapper = MakeWeightList(weight1Ref, weight1RefDtype, isArch22);
    MegaMoeWeightList weight2Wrapper = MakeWeightList(weight2Ref, weight2RefDtype, isArch22);
    TensorListWrapper weightScales1Wrapper = {weightScales1Ref, weightScales1Dtype};
    TensorListWrapper weightScales2Wrapper = {weightScales2Ref, weightScales2Dtype};
    TensorListWrapper bias1Wrapper = {bias1Ref, aclDataType::ACL_FLOAT};
    TensorListWrapper bias2Wrapper = {bias2Ref, aclDataType::ACL_FLOAT};

    at::TensorList sharedWeight1Ref = toTensorList(sharedWeight1);
    at::TensorList sharedWeight2Ref = toTensorList(sharedWeight2);
    at::TensorList sharedWeightScales1Ref = toTensorList(sharedWeightScales1);
    at::TensorList sharedWeightScales2Ref = toTensorList(sharedWeightScales2);
    at::TensorList sharedBias1Ref = toTensorList(sharedBias1);
    at::TensorList sharedBias2Ref = toTensorList(sharedBias2);

    aclDataType sharedWeight1RefDtype =
        sharedWeight1Type.has_value() ? GetAclDataType(sharedWeight1Type.value()) : weight1RefDtype;
    aclDataType sharedWeight2RefDtype =
        sharedWeight2Type.has_value() ? GetAclDataType(sharedWeight2Type.value()) : weight2RefDtype;
    const auto getWeightScaleDtype = [](aclDataType weightDtype) {
        return weightDtype == aclDataType::ACL_FLOAT8_E5M2 || weightDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
                       weightDtype == aclDataType::ACL_FLOAT4_E2M1 ?
                   aclDataType::ACL_FLOAT8_E8M0 :
                   aclDataType::ACL_UINT64;
    };

    if (isMxPlatform) {
        CheckWeightDtype(sharedWeight1Ref, sharedWeight1RefDtype, "shared_l1_weights", socName);
        CheckWeightDtype(sharedWeight2Ref, sharedWeight2RefDtype, "shared_l2_weights", socName);
    }

    MegaMoeWeightList sharedWeight1Wrapper = MakeWeightList(sharedWeight1Ref, sharedWeight1RefDtype, isArch22);
    MegaMoeWeightList sharedWeight2Wrapper = MakeWeightList(sharedWeight2Ref, sharedWeight2RefDtype, isArch22);
    TensorListWrapper sharedWeightScales1Wrapper = {sharedWeightScales1Ref, getWeightScaleDtype(sharedWeight1RefDtype)};
    TensorListWrapper sharedWeightScales2Wrapper = {sharedWeightScales2Ref, getWeightScaleDtype(sharedWeight2RefDtype)};
    TensorListWrapper sharedBias1Wrapper = {sharedBias1Ref, aclDataType::ACL_FLOAT};
    TensorListWrapper sharedBias2Wrapper = {sharedBias2Ref, aclDataType::ACL_FLOAT};

    ACLNN_CMD(aclnnMegaMoe, context, xWrapper, topkIds, topkWeights, weight1Wrapper, weight2Wrapper,
              weightScales1Wrapper, weightScales2Wrapper, bias1Wrapper, bias2Wrapper, xActiveMask, scales,
              sharedWeight1Wrapper, sharedWeight2Wrapper, sharedWeightScales1Wrapper, sharedWeightScales2Wrapper,
              sharedBias1Wrapper, sharedBias2Wrapper, maskBuffer, moeExpertNum, epWorldSize, cclBufferSize,
              maxRecvTokenNum, dispatchQuantMode, dispatchQuantResultType, sharedExpertQuantResultType,
              combineQuantMode, commAlgPtr, numMaxTokensPerRank, activationPtr, activationParams, topoTypeValue,
              rankNumPerServerValue, topkWeightsType, combineCommMode, y, expertTokenNums);

    return std::tie(y, expertTokenNums);
}

namespace {
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int64_t MB_SIZE = 1024LL * 1024LL;
constexpr int64_t RESERVED_SPACE_SIZE = 10LL * 1024 * 1024;
constexpr int64_t MAX_EXPERTS_PER_RANK_A2A3 = 128LL;
constexpr int64_t SYNC_STATE_RESERVED_SIZE = 512LL * 1024;
constexpr int64_t MXFP_SCALE_GROUP_NUM = 32LL;
constexpr int64_t Y_DTYPE_SIZE = 2LL;
constexpr int64_t URMA_H_ALIGN = 64LL;
constexpr int64_t MTE_H_ALIGN = 64LL;
constexpr int64_t MOE_PERMUTE_CHUNK = 1024LL;
// 异常 Dump 区
constexpr int64_t EXCEPTION_DUMP_REGION_SIZE = 60LL * 1024LL;
// rankSyncInWorld 同步区
constexpr int64_t PEERMEM_DATA_OFFSET = 60LL * 1024LL;
constexpr int64_t PEERMEM_MTE_COUNT_REGION_SIZE = 512LL * 2048LL;
// 此 JIT 源文件随 wheel 搬移，包中不包含 op_kernel 头文件。下列架构和布局上界须与
// arch35_92/common/mega_moe_constants.h 及 arch35、arch92 各自的 Workspace 布局保持一致。
constexpr int64_t ARCH92_MAX_AIC_NUM = 64LL;
constexpr int64_t ARCH35_MAX_SHARED_EXPERT_NUM = 4LL;
constexpr int64_t ARCH35_INT_CACHELINE = 16LL;
constexpr int64_t ARCH35_MX_SCALE_DIVISOR = 64LL;
constexpr int64_t ARCH35_MX_MULTI_BASE_SIZE = 2LL;
constexpr int64_t ARCH35_TILE_M = 256LL;
constexpr int64_t ARCH35_TILE_N = 256LL;
constexpr int64_t ARCH35_DEDUP_PREPASS_MAX_CHUNKS = 128LL;
constexpr int64_t ARCH35_DEDUP_ROW_DESC_HEADER_INT32 = 2LL;
constexpr int64_t ARCH35_INT32_PER_32B = 8LL;

int64_t CeilAlign(int64_t val, int64_t align)
{
    return (val + align - 1) / align * align;
}

int64_t CeilDiv(int64_t val, int64_t divisor)
{
    return (val + divisor - 1) / divisor;
}

int64_t CalcLeastCclBufferSizeA2(int64_t maxRecvTokenNum, int64_t h, int64_t epWorldSize, bool isQuantRouting,
                                 int64_t bs, int64_t topK)
{
    int64_t offsetTokenPerExpert = epWorldSize * CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    int64_t offsetAAfterDispatch =
        maxRecvTokenNum * (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    int64_t offsetD = bs * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winInTensorSize = offsetAAfterDispatch + offsetD;

    // winOut A 区按单 chunk（不超过 bs）的 permute 输出分配，逐 chunk 复用，
    // 与 tiling_arch22.cpp CalcLeastCclBufferSize 的 A2 分支一致
    const int64_t chunkTokensA2 = std::min(bs, MOE_PERMUTE_CHUNK);
    int64_t offsetA =
        chunkTokensA2 * topK * (!isQuantRouting ? h * static_cast<int64_t>(sizeof(int16_t)) : (h + ALIGN_512));
    int64_t offsetC = maxRecvTokenNum * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winOutTensorSize = offsetA + offsetC;
    int64_t offsetTensor = std::max(winInTensorSize, winOutTensorSize);
    if (isQuantRouting) {
        offsetTensor += maxRecvTokenNum * static_cast<int64_t>(sizeof(float));
    }

    int64_t offsetFlag = epWorldSize * ALIGN_512;
    offsetFlag += epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 * 64LL;
    offsetFlag += epWorldSize * 64LL;

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

int64_t CalcLeastCclBufferSizeA3(int64_t h, int64_t epWorldSize, bool isQuantRouting, int64_t bs, int64_t topK)
{
    int64_t offsetTokenPerExpert = epWorldSize * CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    const int64_t chunkTokens = std::min(bs, MOE_PERMUTE_CHUNK);
    int64_t offsetAAfterDispatch =
        chunkTokens * topK * (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    int64_t offsetD = chunkTokens * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t offsetTensor = offsetAAfterDispatch + offsetD;
    if (isQuantRouting) {
        offsetTensor += chunkTokens * topK * static_cast<int64_t>(sizeof(float));
    }

    int64_t offsetFlag = std::max(epWorldSize * ALIGN_512, SYNC_STATE_RESERVED_SIZE);

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

// 零冗余（Dedup）布局下 winIn 最小需求，与 op_tiling/arch22/mega_moe_dedup_layout.cpp 的
// CalcLeastCclBufferSizeDedup 同源镜像（MB 向上取整）。仅 A3 + W4A8 路径。
constexpr int64_t DEDUP_PACK_SECTION_COUNT =
    2; // 打包区每 rank 段 [slots|probs] 等长两段（rankOrdinal 内联 slots 高 8bit，同源 mega_moe_dedup_layout.h）

int64_t CalcLeastCclBufferSizeDedupA3(int64_t h, int64_t epWorldSize, int64_t bs, int64_t topK)
{
    int64_t offsetTokenPerExpert = epWorldSize * CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    const int64_t chunkTokens = std::min(bs, MOE_PERMUTE_CHUNK);
    // A 区：原始序 chunkTokens 行 × (h+512)（不乘 topK——跨卡去重后同一 token 只发一次）
    int64_t offsetA = chunkTokens * (h + ALIGN_512);
    // 打包区：每 rank 段 [slots|probs] 各 chunkTokens·topK·i32，总 EP·chunkTokens·topK·8B
    int64_t offsetPack =
        epWorldSize * chunkTokens * topK * static_cast<int64_t>(sizeof(int32_t)) * DEDUP_PACK_SECTION_COUNT;
    // D 区（token 块）：(min(topK,EP)·chunkTokens + 1 行防御 pad)·h·2B；无
    // perTokenScale 保留区。D 基址 128B 对齐与 host 侧 dedup_layout 同口径
    int64_t offsetD = (std::min(topK, epWorldSize) * chunkTokens + 1) * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t offsetTensor = CeilAlign(offsetA + offsetPack, ALIGN_128) + offsetD;

    int64_t offsetFlag = std::max(epWorldSize * ALIGN_512, SYNC_STATE_RESERVED_SIZE);

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

int64_t CalcTokenScaleBytesA5(int64_t hidden, int64_t numTopk, int64_t topkWeightsType)
{
    int64_t mxScaleNum = (hidden + MXFP_SCALE_GROUP_NUM - 1) / MXFP_SCALE_GROUP_NUM;
    int64_t dataBytes = CeilAlign(hidden, ALIGN_256);
    int64_t tokenBytes = CeilAlign(dataBytes + mxScaleNum, ALIGN_32);
    if (topkWeightsType == 1) {
        int64_t weightBytes = CeilAlign(numTopk * static_cast<int64_t>(sizeof(float)), ALIGN_32);
        tokenBytes = CeilAlign(tokenBytes + weightBytes, ALIGN_32);
    }
    return tokenBytes;
}

int64_t CalcMteCclBufferSizeA5(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                               int64_t hidden, int64_t topkWeightsType)
{
    int64_t expertPerRank = moeExpertNum / epWorldSize;

    constexpr int64_t maxInt16RouteItems = 1LL << 15;
    bool useInt16RouteIndex =
        numTopk > 0 && numMaxTokensPerRank >= 0 && numMaxTokensPerRank * numTopk <= maxInt16RouteItems;
    int64_t routeIndexTypeBytes =
        useInt16RouteIndex ? static_cast<int64_t>(sizeof(int16_t)) : static_cast<int64_t>(sizeof(int32_t));
    int64_t routeIndexAlignSize = CeilAlign(numMaxTokensPerRank * routeIndexTypeBytes, ALIGN_32);
    int64_t routeRecvSize = CeilAlign(expertPerRank * epWorldSize * routeIndexAlignSize, ALIGN_512);

    // 与 kernel 的固定 MTE count 区容量保持一致。
    int64_t expertCountRecvSize = PEERMEM_MTE_COUNT_REGION_SIZE;

    int64_t tokenBytes = CalcTokenScaleBytesA5(hidden, numTopk, topkWeightsType);
    int64_t dispatchRecordAreaSize = CeilAlign(numMaxTokensPerRank * tokenBytes, ALIGN_512);

    int64_t combineSendSize = CeilAlign(numMaxTokensPerRank * numTopk * hidden * Y_DTYPE_SIZE, ALIGN_512);

    int64_t totalBytes = EXCEPTION_DUMP_REGION_SIZE + PEERMEM_DATA_OFFSET + routeRecvSize + expertCountRecvSize +
                         dispatchRecordAreaSize + combineSendSize;

    return totalBytes;
}

// 创建 SymmBuffer 时尚未传入具体权重列表，按 WorkspaceLayout 的分配顺序保守预留：
// A8W4、共享专家独立量化、对应架构的核数上界及 4 个共享专家。
// 根据已知通信模式排除 tiling 不会分配的区域。
struct CclDataBudget {
    int64_t epWorldSize;
    int64_t numMaxTokensPerRank;
    int64_t numTopk;
    int64_t hidden;
    int64_t intermediateHidden;
    int64_t topkWeightsType;
    int64_t serverNum;
    int64_t expertPerRank;
    int64_t aicNum;
    int64_t hiddenDim;
    int64_t maxOutput{0};
    int64_t wavesPerExpert{0};
    int64_t sharedTokenGroupCount{0};
    bool isUrma;
    bool reserveCombineDedup;
    bool reserveServerCombine;
    bool reserveDedup;
    bool needsGmm2Ready;

    CclDataBudget(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                  int64_t hidden, int64_t intermediateHidden, int64_t maxRecvTokenNum, int64_t topkWeightsType,
                  int64_t serverNum, int64_t combineQuantMode, int64_t combineCommMode, int64_t maxAicNum)
        : epWorldSize(epWorldSize),
          numMaxTokensPerRank(numMaxTokensPerRank),
          numTopk(numTopk),
          hidden(hidden),
          intermediateHidden(intermediateHidden),
          topkWeightsType(topkWeightsType),
          serverNum(serverNum),
          expertPerRank(moeExpertNum / epWorldSize),
          aicNum(maxAicNum),
          hiddenDim(intermediateHidden * 2LL),
          isUrma(serverNum > 0),
          reserveCombineDedup(!isUrma && combineCommMode == 1),
          reserveServerCombine(isUrma && combineCommMode == 2),
          reserveDedup(!isUrma && (numTopk > 1 || reserveCombineDedup)),
          needsGmm2Ready(combineQuantMode != 0 || reserveCombineDedup)
    {
        uint64_t defaultMaxOutputSize =
            static_cast<uint64_t>(numMaxTokensPerRank) * epWorldSize * std::min(numTopk, expertPerRank);
        uint64_t maxOutputSize = maxRecvTokenNum == 0 ? defaultMaxOutputSize : static_cast<uint64_t>(maxRecvTokenNum);
        maxOutput = static_cast<int64_t>(maxOutputSize);
        wavesPerExpert = CeilDiv(maxOutput, ARCH35_TILE_M);
        sharedTokenGroupCount = CeilDiv(numMaxTokensPerRank, ARCH35_TILE_M);
    }
};

void AccumulateDispatchSize(int64_t& cclDataSize, const CclDataBudget& budget)
{
    cclDataSize += CeilAlign(budget.maxOutput * budget.hidden, ALIGN_512);
    cclDataSize += CeilAlign(
        budget.maxOutput * CeilDiv(budget.hidden, ARCH35_MX_SCALE_DIVISOR) * ARCH35_MX_MULTI_BASE_SIZE, ALIGN_512);

    // 保留 URMA 的保守权重预算，包含未启用 topk 权重在激活量化阶段同步计算的路径。
    if (budget.isUrma || (budget.reserveCombineDedup && budget.topkWeightsType == 0)) {
        cclDataSize += CeilAlign(
            budget.maxOutput * CeilAlign(budget.numTopk * static_cast<int64_t>(sizeof(float)), ALIGN_32), ALIGN_512);
    }
}

void AccumulateActivationAndMetadataSize(int64_t& cclDataSize, const CclDataBudget& budget)
{
    cclDataSize += CeilAlign(budget.maxOutput * budget.intermediateHidden, ALIGN_512);
    cclDataSize += CeilAlign(
        budget.maxOutput * CeilDiv(budget.intermediateHidden, ARCH35_MX_SCALE_DIVISOR) * ARCH35_MX_MULTI_BASE_SIZE,
        ALIGN_512);
    int64_t expertRecvTokenCountBytes =
        CeilAlign(budget.expertPerRank, ARCH35_INT_CACHELINE) * static_cast<int64_t>(sizeof(int32_t)) * budget.aicNum;
    cclDataSize += CeilAlign(expertRecvTokenCountBytes, ALIGN_512);
    cclDataSize += CeilAlign(budget.maxOutput * ALIGN_32, ALIGN_512);
}

void AccumulateFlagSize(int64_t& cclDataSize, int64_t slotCount)
{
    cclDataSize += static_cast<int64_t>(sizeof(int32_t)) * slotCount * ARCH35_INT_CACHELINE;
}

void AccumulateSyncSize(int64_t& cclDataSize, const CclDataBudget& budget)
{
    // 保持 WorkspaceLayout 的标志位顺序，每个槽位独占一条 64B 缓存行。
    if (budget.isUrma) {
        AccumulateFlagSize(cclDataSize, budget.expertPerRank * budget.wavesPerExpert); // 激活到 GMM2 的通知。
        AccumulateFlagSize(cclDataSize, budget.expertPerRank * budget.wavesPerExpert); // 分发到 GMM1 的通知。
        AccumulateFlagSize(cclDataSize, budget.expertPerRank * budget.aicNum); // 接收数量元数据就绪通知。
        int64_t maxTokenGroups = CeilDiv(budget.numMaxTokensPerRank * budget.epWorldSize, ARCH35_TILE_M);
        int64_t combineSyncSlots = std::max(maxTokenGroups, budget.aicNum);
        AccumulateFlagSize(cclDataSize, combineSyncSlots * budget.expertPerRank); // GMM2 到分层合并的通知。
    } else {
        AccumulateFlagSize(cclDataSize, budget.expertPerRank * budget.wavesPerExpert); // 激活到 GMM2 的通知。
        // 共享专家激活到 GMM2 的通知。
        AccumulateFlagSize(cclDataSize, budget.sharedTokenGroupCount * ARCH35_MAX_SHARED_EXPERT_NUM);
        AccumulateFlagSize(cclDataSize, budget.expertPerRank * budget.wavesPerExpert); // 分发到 GMM1 的通知。
        AccumulateFlagSize(cclDataSize, budget.expertPerRank * budget.aicNum); // 接收数量元数据就绪通知。
        // topK=1 时收益门必定关闭分发去重；其他情况同时覆盖收益门开启和关闭的结果。
        if (budget.reserveDedup) {
            AccumulateFlagSize(cclDataSize, 1LL + budget.epWorldSize * ARCH35_DEDUP_PREPASS_MAX_CHUNKS);
        }
        if (budget.needsGmm2Ready) {
            AccumulateFlagSize(cclDataSize, budget.expertPerRank * budget.aicNum);
        }
        // 共享专家 GMM2 到最终输出累加的通知。
        AccumulateFlagSize(cclDataSize, budget.sharedTokenGroupCount * ARCH35_MAX_SHARED_EXPERT_NUM);
        AccumulateFlagSize(cclDataSize, budget.aicNum); // 保留已有 AIV1 topK 同步区预算。
    }
}

void AccumulateGmmSize(int64_t& cclDataSize, const CclDataBudget& budget)
{
    // 依次预留累计计数、A8W4 GMM1 结果和 GMM2 结果；A8W4 覆盖当前最大的路由专家布局。
    cclDataSize +=
        CeilAlign(static_cast<int64_t>(sizeof(int32_t)) * budget.expertPerRank * budget.epWorldSize, ALIGN_32) *
        budget.aicNum;
    cclDataSize += static_cast<int64_t>(sizeof(int16_t)) * budget.maxOutput * budget.hiddenDim;
    if (!budget.isUrma) {
        // 对齐累计偏移，因此必须包含前面标志位区域的大小。
        cclDataSize = CeilAlign(cclDataSize, ALIGN_512);
        cclDataSize += CeilAlign(static_cast<int64_t>(sizeof(int16_t)) * budget.maxOutput * budget.hidden, ALIGN_512);
    } else {
        cclDataSize += static_cast<int64_t>(sizeof(int16_t)) * budget.maxOutput * budget.hidden;
    }
}

void AccumulateRoutingAuxSize(int64_t& cclDataSize, const CclDataBudget& budget)
{
    if (budget.isUrma) {
        int64_t alignedRouteCount =
            CeilAlign(budget.numMaxTokensPerRank * budget.numTopk * static_cast<int64_t>(sizeof(int32_t)), ALIGN_256) /
            static_cast<int64_t>(sizeof(int32_t));
        int64_t maskAlignSize = CeilAlign(alignedRouteCount / 8LL, ALIGN_32);
        cclDataSize += CeilAlign(budget.expertPerRank * budget.epWorldSize * (maskAlignSize + ALIGN_32), ALIGN_512);
        cclDataSize += CeilAlign(budget.serverNum * (ALIGN_32 + budget.numMaxTokensPerRank * ALIGN_32), ALIGN_512);
        cclDataSize += budget.aicNum * CeilAlign(ARCH35_TILE_M * static_cast<int64_t>(sizeof(uint64_t)), ALIGN_512);
        if (budget.reserveServerCombine) {
            // 与 WorkspaceLayout::InitializeServerCombineBuffers 的独立状态、列表和位图区域一致。
            cclDataSize +=
                CeilAlign(budget.serverNum * ARCH35_INT_CACHELINE * static_cast<int64_t>(sizeof(int32_t)), ALIGN_512);
            const int64_t tokenListStride = (budget.numMaxTokensPerRank + 1LL) * static_cast<int64_t>(sizeof(int32_t));
            cclDataSize += CeilAlign(budget.serverNum * tokenListStride, ALIGN_512);
            cclDataSize += CeilAlign(
                budget.serverNum * budget.numMaxTokensPerRank * static_cast<int64_t>(sizeof(uint32_t)), ALIGN_512);
        }
    } else if (budget.reserveDedup) {
        int64_t rowDescStride =
            CeilAlign(ARCH35_DEDUP_ROW_DESC_HEADER_INT32 + 3LL * budget.numTopk, ARCH35_INT32_PER_32B);
        cclDataSize += CeilAlign(static_cast<int64_t>(sizeof(int32_t)) * rowDescStride * budget.maxOutput, ALIGN_512);
    }
}

void AccumulateSharedExpertSize(int64_t& cclDataSize, const CclDataBudget& budget)
{
    // 按 arch35 接口上限预留 4 个共享专家，独立量化需要预留 MTE 输入区域。
    cclDataSize += CeilAlign(static_cast<int64_t>(sizeof(int16_t)) * budget.numMaxTokensPerRank *
                                 ARCH35_MAX_SHARED_EXPERT_NUM * budget.hidden,
                             ALIGN_512);
    cclDataSize += CeilAlign(static_cast<int64_t>(sizeof(int16_t)) * budget.numMaxTokensPerRank *
                                 ARCH35_MAX_SHARED_EXPERT_NUM * budget.hiddenDim,
                             ALIGN_512);
    if (budget.isUrma) {
        cclDataSize += CeilAlign(budget.numMaxTokensPerRank * budget.hidden, ALIGN_512);
        cclDataSize += CeilAlign(
            budget.numMaxTokensPerRank * CeilDiv(budget.hidden, ALIGN_32) * ARCH35_MX_MULTI_BASE_SIZE, ALIGN_512);
    } else {
        int64_t sharedRecordBytes =
            CeilAlign(CeilAlign(budget.hidden, ALIGN_256) + CeilDiv(budget.hidden, ALIGN_32), ALIGN_32);
        cclDataSize += CeilAlign(budget.numMaxTokensPerRank * sharedRecordBytes, ALIGN_512);
    }
    cclDataSize +=
        CeilAlign(budget.numMaxTokensPerRank * ARCH35_MAX_SHARED_EXPERT_NUM * budget.intermediateHidden, ALIGN_512);
    cclDataSize +=
        CeilAlign(budget.numMaxTokensPerRank * ARCH35_MAX_SHARED_EXPERT_NUM *
                      CeilDiv(budget.intermediateHidden, ARCH35_MX_SCALE_DIVISOR) * ARCH35_MX_MULTI_BASE_SIZE,
                  ALIGN_512);
}

int64_t CalcCclDataSizeA5(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                          int64_t hidden, int64_t intermediateHidden, int64_t maxRecvTokenNum, int64_t topkWeightsType,
                          int64_t serverNum, int64_t combineQuantMode, int64_t combineCommMode, int64_t maxAicNum)
{
    const CclDataBudget budget(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden, intermediateHidden,
                               maxRecvTokenNum, topkWeightsType, serverNum, combineQuantMode, combineCommMode,
                               maxAicNum);
    int64_t cclDataSize = 0;
    AccumulateDispatchSize(cclDataSize, budget);
    AccumulateActivationAndMetadataSize(cclDataSize, budget);
    AccumulateSyncSize(cclDataSize, budget);
    AccumulateGmmSize(cclDataSize, budget);
    AccumulateRoutingAuxSize(cclDataSize, budget);
    AccumulateSharedExpertSize(cclDataSize, budget);
    return cclDataSize;
}

// Arch35 使用公共布局规划器，Arch92 保留上方主线全量分配预算。
// 容量查询尚未知核数、共享专家数及权重模式，未知特征在申请侧保留安全上界。
struct MemoryBudgetParams {
    MegaMoeImpl::CclCommLayoutParams comm;
    MegaMoeImpl::CclComputeLayoutParams compute;
};

MemoryBudgetParams BuildMemoryBudgetParams(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank,
                                           int64_t numTopk, int64_t h, int64_t intermediateHidden,
                                           int64_t maxRecvTokenNum, int64_t topkWeightsType, int64_t serverNum,
                                           int64_t combineQuantMode, int64_t combineCommMode,
                                           aclDataType dispatchQuantOutDtype)
{
    MemoryBudgetParams budget{};
    auto& params = budget.compute;
    params.topoType = serverNum > 0 ? MegaMoeImpl::TOPO_TYPE_URMA : MegaMoeImpl::TOPO_TYPE_MTE;
    params.maxOutputSize = maxRecvTokenNum == 0 ?
                               numMaxTokensPerRank * epWorldSize *
                                   (numTopk < moeExpertNum / epWorldSize ? numTopk : moeExpertNum / epWorldSize) :
                               maxRecvTokenNum;
    params.h = h;
    params.hiddenDim = intermediateHidden * 2;
    params.bs = numMaxTokensPerRank;
    params.moeExpertPerRank = moeExpertNum / epWorldSize;
    params.epWorldSize = epWorldSize;
    params.aicNum = 36;
    params.sharedExpertNum = 4;
    // 创建窗口时尚未知激活类型，按 GELU 的较窄 GMM1 输出取较大的 wave 行预算。
    // GMM1 结果宽度仍按门控激活的两倍 intermediateHidden 预留。
    params.mGroupsPerWave = MegaMoeImpl::CalcMemoryMGroupsPerWave(intermediateHidden, params.h, params.aicNum);
    const int64_t maxTokenGroups = MegaMoeImpl::MemoryLayoutDetail::CeilDiv(
        numMaxTokensPerRank * epWorldSize, MegaMoeImpl::MemoryLayoutDetail::L1_TILE_M_256);
    params.combineSyncSlotCountPerExpert = maxTokenGroups > params.aicNum ? maxTokenGroups : params.aicNum;
    params.serverNum = serverNum;
    params.numMaxTokensPerRank = numMaxTokensPerRank;
    params.topK = numTopk;
    params.dedupMode = params.topoType == MegaMoeImpl::TOPO_TYPE_MTE ?
                           (combineCommMode == 1 ? 1 : 0) :
                           (combineCommMode == 2 ? MegaMoeImpl::DEDUP_MODE_SERVER_COMBINE : 0);
    params.combineQuant = combineQuantMode != 0;
    params.topkWeightsPrefetch = topkWeightsType == 1;
    const bool isFp4 = dispatchQuantOutDtype == ACL_FLOAT4_E2M1;
    const bool isE5M2 = dispatchQuantOutDtype == ACL_FLOAT8_E5M2;
    params.dispatchFp4 = isFp4;
    // E5M2 仅支持 A8W8，FP4 仅支持 A4W4；两者的 GMM1 仅在权重前移时落 GM。
    params.reserveGmm1Output = params.topkWeightsPrefetch || (!isFp4 && !isE5M2);
    params.reserveCumsum = !isE5M2;
    // 共享专家的量化配置独立于 MoE，仍按 W4 与独立量化的上界预留。
    params.sharedInputIndependent = true;

    auto& comm = budget.comm;
    comm.numMaxTokensPerRank = numMaxTokensPerRank;
    comm.topK = numTopk;
    comm.h = h;
    comm.moeExpertPerRank = params.moeExpertPerRank;
    comm.epWorldSize = epWorldSize;
    comm.yDtypeSize = MegaMoeImpl::SIZE_BF_16;
    // 已知 FP4 时与 tiling 的实际记录宽度一致；未知类型继续保留 FP8 上界。
    comm.elemsPerByte = isFp4 ? 2U : 1U;
    comm.serverCombineReduce = MegaMoeImpl::IsServerCombineDedupOn(params.dedupMode);
    comm.topkWeightsPrefetch =
        params.topkWeightsPrefetch || MegaMoeImpl::IsCombineDedupOn(params.dedupMode) || comm.serverCombineReduce;
    comm.isQuantCombine = params.combineQuant;
    comm.topoType = params.topoType;
    comm.serverNum = serverNum;
    params.ring = MegaMoeImpl::CalcCclStageRingConfig(params);
    return budget;
}

int64_t CalcCclBufferSizeA5(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                            int64_t hidden, int64_t intermediateHidden, int64_t maxRecvTokenNum,
                            int64_t topkWeightsType, int64_t serverNum, int64_t combineQuantMode,
                            int64_t combineCommMode, aclDataType dispatchQuantOutDtype)
{
    // 创建 SymmBuffer 时尚未知权重和共享专家配置，统一构造安全上界。
    const auto budget = BuildMemoryBudgetParams(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden,
                                                intermediateHidden, maxRecvTokenNum, topkWeightsType, serverNum,
                                                combineQuantMode, combineCommMode, dispatchQuantOutDtype);
    int64_t dataBytes = MegaMoeImpl::WorkspaceLayout(budget.compute).workspaceSize;
    if (budget.compute.topoType == MegaMoeImpl::TOPO_TYPE_MTE) {
        // Dispatch 由 tiling 自动选择；分别计算开、关布局，覆盖数据区、去重描述符及标志区。
        auto dispatchDedupParams = budget.compute;
        dispatchDedupParams.dedupMode |= 2;
        dispatchDedupParams.ring = MegaMoeImpl::CalcCclStageRingConfig(dispatchDedupParams);
        dataBytes = std::max(dataBytes, MegaMoeImpl::WorkspaceLayout(dispatchDedupParams).workspaceSize);
    }
    return MegaMoeImpl::CalcCclComputeOffset(budget.comm) + dataBytes;
}
} // namespace

int64_t GetMegaMoeCclBufferSize(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                                int64_t hidden, int64_t intermediateHidden, int64_t maxRecvTokenNum,
                                int64_t dispatchQuantMode, c10::optional<int64_t> dispatchQuantOutDtype,
                                int64_t combineQuantMode, std::string commAlg, int64_t combineCommMode,
                                int64_t topkWeightsType, int64_t serverNum)
{
    TORCH_CHECK(serverNum >= 0, "server_num must be non-negative, but got ", serverNum);
    TORCH_CHECK(combineCommMode >= 0 && combineCommMode <= 2, "combine_comm_mode only supports 0, 1 or 2, but got ",
                combineCommMode);
    const char* socName = aclrtGetSocName();
    bool isA2 = (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr);
    bool isA3 = (socName != nullptr && std::strstr(socName, "Ascend910_93") != nullptr);
    const bool isArch92 = socName != nullptr && std::strstr(socName, "Ascend960DT") != nullptr;
    if (isA2 || isA3) {
        TORCH_CHECK(serverNum == 0, "server_num is only supported by the Ascend950 channel backend");
        TORCH_CHECK(epWorldSize == 2 || epWorldSize == 4 || epWorldSize == 8 || epWorldSize == 16 ||
                        epWorldSize == 32 || epWorldSize == 48 || epWorldSize == 64 || epWorldSize == 96 ||
                        epWorldSize == 128,
                    "ep_world_size only support {2, 4, 8, 16, 32, 48, 64, 96, 128} on A2/A3, but got ", epWorldSize);
        TORCH_CHECK(hidden >= 1024 && hidden <= 8192 && hidden % 512 == 0,
                    "hidden only support [1024, 8192] and hidden % 512 == 0 on A2/A3, but got ", hidden);
        TORCH_CHECK(numMaxTokensPerRank >= 1, "num_max_tokens_per_rank should be >= 1 on A2/A3, but got ",
                    numMaxTokensPerRank);
        TORCH_CHECK(moeExpertNum >= 1 && moeExpertNum <= 2048,
                    "moe_expert_num only support [1, 2048] on A2/A3, but got ", moeExpertNum);
        TORCH_CHECK(numTopk >= 1 && numTopk <= 16, "num_topk only support [1, 16] on A2/A3, but got ", numTopk);
        TORCH_CHECK(dispatchQuantMode == 0 || dispatchQuantMode == 2 || dispatchQuantMode == 4,
                    "dispatch_quant_mode only support {0, 2, 4} on A2/A3, but got ", dispatchQuantMode);

        // 建通信 context 前拒绝必败配置；权重 dtype 的 W4A8 精判由算子 tiling 把关
        TORCH_CHECK(combineCommMode == 0 || combineCommMode == 1,
                    "combine_comm_mode only support 0 or 1 on A2/A3, but got ", combineCommMode);
        TORCH_CHECK(!(isA2 && combineCommMode == 1),
                    "combine_comm_mode=1 (combine dedup) is not supported on Atlas A2, keep combine_comm_mode=0");
        TORCH_CHECK(!(isA3 && combineCommMode == 1 && dispatchQuantMode != 2),
                    "combine_comm_mode=1 (combine dedup) currently supports only the W4A8 (A8W4-INT) quant "
                    "path on Atlas A3, which requires dispatch_quant_mode=2; set combine_comm_mode=0 otherwise");

        bool isQuantRouting = (dispatchQuantMode == 4);
        // max_recv_token_num 为 0 时自动计算为 bs * epWorldSize * min(topK, expertPerRank)，
        if (maxRecvTokenNum == 0) {
            int64_t expertPerRank = moeExpertNum / epWorldSize;
            maxRecvTokenNum = numMaxTokensPerRank * epWorldSize * std::min(numTopk, expertPerRank);
        }
        if (isA3) {
            // 与 host tiling [DEDUP] 同源：combine_comm_mode=1 时 tiling 走 Dedup、被 G 收益门
            // 静默关回基线或直接报错三种结局，两公式大小关系随 EP/topK 翻转，取 max 以
            // 兼容可行性校验前的容量咨询；=0 或非量化 dispatch（无 Dedup 实例）→ 基线公式
            int64_t masterMb =
                CalcLeastCclBufferSizeA3(hidden, epWorldSize, isQuantRouting, numMaxTokensPerRank, numTopk);
            if (dispatchQuantMode != 0 && combineCommMode == 1) {
                return std::max(masterMb,
                                CalcLeastCclBufferSizeDedupA3(hidden, epWorldSize, numMaxTokensPerRank, numTopk));
            }
            return masterMb;
        }
        return CalcLeastCclBufferSizeA2(maxRecvTokenNum, hidden, epWorldSize, isQuantRouting, numMaxTokensPerRank,
                                        numTopk);
    }

    TORCH_CHECK(!isArch92 || serverNum == 0, "Ascend960DT MegaMoe only supports the MTE topology (server_num=0)");
    TORCH_CHECK(epWorldSize >= 2 && epWorldSize <= 1024, "ep_world_size only support in [2, 1024], but got ",
                epWorldSize);
    TORCH_CHECK(hidden >= 1024 && hidden <= 8192, "hidden only support in [1024, 8192], but got ", hidden);
    int64_t hiddenAlignment = serverNum > 0 ? URMA_H_ALIGN : MTE_H_ALIGN;
    TORCH_CHECK(hidden % hiddenAlignment == 0, "hidden must be a multiple of ", hiddenAlignment,
                " for the selected communication topology, but got ", hidden);
    TORCH_CHECK(
        numMaxTokensPerRank >= 1 && static_cast<uint64_t>(numMaxTokensPerRank) <= std::numeric_limits<uint32_t>::max(),
        "num_max_tokens_per_rank should be in [1, UINT32_MAX], but got ", numMaxTokensPerRank);
    TORCH_CHECK(maxRecvTokenNum >= 0, "max_recv_token_num should be non-negative, but got ", maxRecvTokenNum);
    TORCH_CHECK(moeExpertNum >= epWorldSize && moeExpertNum <= 2048 && moeExpertNum % epWorldSize == 0,
                "moe_expert_num should be in [ep_world_size, 2048] and divisible by ep_world_size, but got ",
                moeExpertNum, " and ep_world_size ", epWorldSize);
    TORCH_CHECK(numTopk >= 1 && numTopk <= 32, "num_topk only support in [1, 32], but got ", numTopk);
    TORCH_CHECK(topkWeightsType == 0 || topkWeightsType == 1, "topk_weights_type only support 0 or 1, but got ",
                topkWeightsType);
    TORCH_CHECK(combineQuantMode == 0 || combineQuantMode == 3 || combineQuantMode == 4,
                "combine_quant_mode only support 0, 3 or 4 on Ascend950, but got ", combineQuantMode);

    if (serverNum > 0) {
        TORCH_CHECK(serverNum > 1 && serverNum <= epWorldSize && epWorldSize % serverNum == 0,
                    "server_num should be in [2, ep_world_size] and divide ep_world_size, but got ", serverNum,
                    " and ep_world_size ", epWorldSize);
        int64_t routeCapacity = numMaxTokensPerRank * numTopk;
        TORCH_CHECK(routeCapacity <= std::numeric_limits<int32_t>::max(),
                    "num_max_tokens_per_rank * num_topk should be <= INT32_MAX for URMA, but got ", routeCapacity);
        int64_t expertPerRank = moeExpertNum / epWorldSize;
        int64_t maxOutputCapacity = numMaxTokensPerRank * epWorldSize * std::min(numTopk, expertPerRank);
        TORCH_CHECK(maxOutputCapacity <= std::numeric_limits<int32_t>::max(),
                    "maximum receive token capacity should be <= INT32_MAX for URMA, but got ", maxOutputCapacity);
        TORCH_CHECK(maxRecvTokenNum <= maxOutputCapacity,
                    "max_recv_token_num should not exceed the URMA maximum receive token capacity ", maxOutputCapacity,
                    ", but got ", maxRecvTokenNum);
    }
    if (isArch92) {
        // Arch92 沿用主线全量分配和 64 核容量上界，不接入 Arch35 的布局优化。
        const int64_t layoutTopkWeights = (topkWeightsType == 1 || combineCommMode == 1) ? 1 : 0;
        const int64_t communicationBufferSize =
            CalcMteCclBufferSizeA5(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden, layoutTopkWeights);
        return CeilAlign(communicationBufferSize, ALIGN_512) +
               CalcCclDataSizeA5(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden, intermediateHidden,
                                 maxRecvTokenNum, topkWeightsType, serverNum, combineQuantMode, combineCommMode,
                                 ARCH92_MAX_AIC_NUM);
    }
    const bool hasDispatchQuantOutDtype =
        dispatchQuantOutDtype.has_value() && dispatchQuantOutDtype.value() != GE_DTYPE_UNDEFINED;
    const aclDataType quantOutDtype =
        hasDispatchQuantOutDtype ? GetAclDataType(dispatchQuantOutDtype.value()) : ACL_DT_UNDEFINED;
    return CalcCclBufferSizeA5(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden, intermediateHidden,
                               maxRecvTokenNum, topkWeightsType, serverNum, combineQuantMode, combineCommMode,
                               quantOutDtype);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_mega_moe", &NpuMegaMoe, "npu_mega_moe");
    m.def("get_mega_moe_ccl_buffer_size", &GetMegaMoeCclBufferSize, "get_mega_moe_ccl_buffer_size",
          py::arg("ep_world_size"), py::arg("moe_expert_num"), py::arg("num_max_tokens_per_rank"), py::arg("num_topk"),
          py::arg("hidden"), py::arg("intermediate_hidden"), py::arg("max_recv_token_num"),
          py::arg("dispatch_quant_mode"), py::arg("dispatch_quant_out_dtype"), py::arg("combine_quant_mode"),
          py::arg("comm_alg"), py::arg("combine_comm_mode") = 0, py::arg("topk_weights_type"),
          py::arg("server_num") = 0);
}

} // namespace op_api
