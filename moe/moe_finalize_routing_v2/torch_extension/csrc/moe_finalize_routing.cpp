/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

// MXFP4 uses ND descriptors even for 3D tensors. The common converter labels
// 3D tensors NCL, and its singleton stride heuristic is ambiguous for packed FP4.
struct Mx4NdTensor {
    const at::Tensor& tensor;
    bool packed;
};

aclTensor* ConvertType(const Mx4NdTensor& input)
{
    static const auto createTensor = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(createTensor != nullptr, "aclCreateTensor was not found");
    TORCH_CHECK(IsOpInputBaseFormat(input.tensor), "MXFP4 requires a dense base-format tensor");
    auto shape = input.tensor.sizes().vec();
    auto strides = input.tensor.strides().vec();
    if (input.packed) {
        shape.back() *= FP4_IN_INT8;
        for (size_t i = 0; i + 1 < strides.size(); ++i) {
            strides[i] *= FP4_IN_INT8;
        }
    }
    auto dtype = ConvertToAclDataType(input.tensor.scalar_type());
    if (input.packed) {
        dtype = ACL_FLOAT4_E2M1;
    } else if (dtype == ACL_UINT8) {
        dtype = ACL_FLOAT8_E8M0;
    }
    return createTensor(shape.data(), shape.size(), dtype, strides.data(),
                        ConvertToAclStorageOffset(input.tensor, dtype), ACL_FORMAT_ND, shape.data(), shape.size(),
                        const_cast<void*>(input.tensor.storage().data()));
}

constexpr size_t INDEX_ONE = 1;
constexpr size_t INDEX_TWO = 2;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;

at::Tensor MoeFinalizeRouting(
    const at::Tensor& expandedX, const at::Tensor& expandedRowIdx, const c10::optional<at::Tensor>& x1Optional,
    const c10::optional<at::Tensor>& x2Optional, const c10::optional<at::Tensor>& biasOptional,
    const c10::optional<at::Tensor>& scalesOptional, const c10::optional<at::Tensor>& expertIdxOptional,
    const c10::optional<at::Tensor>& xOptional, const c10::optional<at::Tensor>& alpha1Optional,
    const c10::optional<at::Tensor>& alpha2Optional, const c10::optional<at::Tensor>& vOptional,
    c10::optional<int64_t> dropPadMode, const c10::optional<std::vector<int64_t>>& zeroExpertRange,
    const c10::optional<std::vector<int64_t>>& copyExpertRange,
    const c10::optional<std::vector<int64_t>>& constantExpertRange, c10::optional<int64_t> k,
    const c10::optional<at::Tensor>& expandedXScaleOptional)
{
    int64_t kAttr = c10::value_or_else(k, [] { return 1; });

    const bool isMx4 = expandedX.scalar_type() == at::ScalarType::Byte ||
                       ConvertToAclDataType(expandedX.scalar_type()) == ACL_FLOAT4_E2M1;
    const int64_t mode = dropPadMode.value_or(0);
    if (isMx4) {
        // Only guard accesses needed before ACLNN performs parameter validation.
        TORCH_CHECK(kAttr > 0, "MXFP4 k must be positive before allocating the output");
        TORCH_CHECK(expandedX.dim() == DIM_TWO || expandedX.dim() == DIM_THREE,
                    "MXFP4 expanded_x must be 2D or 3D before allocating the output");
        TORCH_CHECK(expandedXScaleOptional.has_value(), "MXFP4 expanded_x_scale is required");
    }
    int64_t dim0 = isMx4 ? expandedRowIdx.numel() : expandedRowIdx.size(0);
    if (!isMx4 && scalesOptional.has_value()) {
        dim0 = scalesOptional.value().size(0);
    } else if (kAttr > 0) {
        dim0 = dim0 / kAttr;
    }
    size_t dim1Index = INDEX_ONE;
    if (expandedX.dim() == DIM_THREE) {
        dim1Index = INDEX_TWO;
    }
    const int64_t width = expandedX.size(dim1Index) * (isMx4 ? FP4_IN_INT8 : 1);
    at::Tensor result =
        at::empty({dim0, width}, isMx4 ? expandedX.options().dtype(at::kBFloat16) : expandedX.options());
    std::vector<int64_t> zeroVec = zeroExpertRange.value_or(std::vector<int64_t>{});
    std::vector<int64_t> copyVec = copyExpertRange.value_or(std::vector<int64_t>{});
    std::vector<int64_t> constVec = constantExpertRange.value_or(std::vector<int64_t>{});
    at::IntArrayRef zeroRange(zeroVec);
    at::IntArrayRef copyRange(copyVec);
    at::IntArrayRef constRange(constVec);
    if (isMx4) {
        Mx4NdTensor mx4X{expandedX, true};
        Mx4NdTensor mx4Scales{*expandedXScaleOptional, false};
        ACLNN_CMD(aclnnMoeFinalizeRoutingV5, mx4X, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                  expertIdxOptional, xOptional, alpha1Optional, alpha2Optional, vOptional, mx4Scales, mode, zeroRange,
                  copyRange, constRange, kAttr, result);
        return result;
    }
    ACLNN_CMD(aclnnMoeFinalizeRoutingV4, expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional,
              scalesOptional, expertIdxOptional, xOptional, alpha1Optional, alpha2Optional, vOptional, mode, zeroRange,
              copyRange, constRange, kAttr, result);

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_finalize_routing", &MoeFinalizeRouting, "moe_finalize_routing");
}

} // namespace op_api
