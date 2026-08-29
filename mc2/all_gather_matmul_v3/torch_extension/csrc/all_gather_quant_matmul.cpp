// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <set>
#include <torch/extension.h>
#include <cstring>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;
static const int64_t GROUP_MAX = 65535;
static const size_t GROUP_DIM = 3;
static const size_t OFFSET_32_BITS = 32;
static const size_t OFFSET_16_BITS = 16;
const std::set<int> SUPPORT_RANK_SIZE_LIST{2, 4, 8, 16};

int64_t CheckAndGetGroupSizeAg(at::IntArrayRef groupSizeList)
{
    int64_t groups = 0;
    if (groupSizeList.empty()) {
        return groups;
    }
    size_t groupDim = groupSizeList.size();
    TORCH_CHECK(groupDim == GROUP_DIM, "group_sizes only support input with three elements, but got ", groupDim);

    int64_t groupM = static_cast<int64_t>(groupSizeList[0]);
    int64_t groupN = static_cast<int64_t>(groupSizeList[1]);
    int64_t groupK = static_cast<int64_t>(groupSizeList[2]);

    bool invalid_group_param = ((groupM <= GROUP_MAX && groupM >= 0) && (groupN <= GROUP_MAX && groupN >= 0) &&
                                (groupK <= GROUP_MAX && groupK >= 0));
    TORCH_CHECK(invalid_group_param, "group param value must conform to range [0, 65535]");

    groups = static_cast<int64_t>((static_cast<uint64_t>(groupM) << OFFSET_32_BITS) +
                                  (static_cast<uint64_t>(groupN) << OFFSET_16_BITS) + static_cast<uint64_t>(groupK));
    return groups;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> NpuAllGatherQuantMatmul(
    const at::Tensor &context, const at::Tensor &x1, const at::Tensor &x2, int64_t hcclBufferSize, std::string group,
    int64_t rankSize, const c10::optional<at::Tensor> &biasOptional, const c10::optional<at::Tensor> &x1ScaleOptional,
    const c10::optional<at::Tensor> &x2ScaleOptional, c10::IntArrayRef groupSizes, c10::optional<int64_t> x1Dtype,
    c10::optional<int64_t> x2Dtype, c10::optional<int64_t> x1ScaleDtype, c10::optional<int64_t> x2ScaleDtype,
    c10::optional<int64_t> yDtype, std::string commMode)
{
    TORCH_CHECK(x1.dim() == DIM_TWO,
                "The x1 input of allgatherquantmatmul is required to be 2D, but the actual x1 input is ", x1.dim(),
                "D.");
    TORCH_CHECK(x2.dim() == DIM_TWO,
                "The x2 input of allgatherquantmatmul is required to be 2D, but the actual x2 input is ", x2.dim(),
                "D.");
    // x2 为 .t() 转置 view: viewShape [K,N]，物理存储 [N,K]；
    // aclnn 层 IsTransposeLastTwoDims+TransX2Tensor 会翻回 [N,K]，K 维交叉校验在 aclnn CheckShape 做
    TORCH_CHECK(SUPPORT_RANK_SIZE_LIST.find(rankSize) != SUPPORT_RANK_SIZE_LIST.end(),
                "The rank_size should be in [2, 4, 8, 16], but the actual value is ", rankSize, ".");

    auto x1Size = x1.sizes();
    auto x2Size = x2.sizes();
    int64_t mPerRank = x1Size[0];
    int64_t n = x2Size[1]; // x2 传 .t() view [K, N]，dim1=N

    at::ScalarType outScalarType = at::kBFloat16;
    if (yDtype.has_value()) {
        aclDataType outAclDtype = GetAclDataType(yDtype.value());
        // V3 仅支持 bf16/fp16 输出，其余枚举前置拦截（之前非三者之一会静默 fallback 到 bf16 放行）
        TORCH_CHECK(outAclDtype == ACL_FLOAT16 || outAclDtype == ACL_BF16,
                    "y_dtype only supports fp16(5)/bf16(15), but got invalid enum value ", yDtype.value(), ".");
        outScalarType = (outAclDtype == ACL_FLOAT16) ? at::kHalf : at::kBFloat16;
    }

    at::Tensor y = at::empty({mPerRank * rankSize, n}, x1.options().dtype(outScalarType));

    std::string commModeStr = std::string(commMode);
    char *commModePtr = const_cast<char *>(commModeStr.c_str());

    std::string groupStr = std::string(group);
    char *groupPtr = const_cast<char *>(groupStr.c_str());

    aclDataType x1AclDtype =
        x1Dtype.has_value() ? GetAclDataType(x1Dtype.value()) : ConvertToAclDataType(x1.scalar_type());
    aclDataType x2AclDtype =
        x2Dtype.has_value() ? GetAclDataType(x2Dtype.value()) : ConvertToAclDataType(x2.scalar_type());

    // gather_out / amax_out 当前不使能，
    // torch 侧传 {0} 空占位 tensor 经 aclnn 透传 inner（aclnn/tiling/kernel 均不处理），并原样返回给 torch。
    at::Tensor gatherOut = at::empty({0}, x1.options().dtype(x1.scalar_type()));
    at::Tensor amaxOut = at::empty({0}, x1.options().dtype(at::kFloat));

    const at::Tensor &x1ScaleReal = x1ScaleOptional.value_or(at::Tensor());
    const at::Tensor &x2ScaleReal = x2ScaleOptional.value_or(at::Tensor());
    aclDataType x1ScaleAclDtype =
        x1ScaleDtype.has_value() ?
            GetAclDataType(x1ScaleDtype.value()) :
            (x1ScaleReal.defined() ? ConvertToAclDataType(x1ScaleReal.scalar_type()) : ACL_DT_UNDEFINED);
    aclDataType x2ScaleAclDtype =
        x2ScaleDtype.has_value() ?
            GetAclDataType(x2ScaleDtype.value()) :
            (x2ScaleReal.defined() ? ConvertToAclDataType(x2ScaleReal.scalar_type()) : ACL_DT_UNDEFINED);

    TensorWrapper x1Wrapper = {x1, x1AclDtype};
    TensorWrapper x2Wrapper = {x2, x2AclDtype};
    TensorWrapper x1ScaleWrapper = {x1ScaleReal, x1ScaleAclDtype};
    TensorWrapper x2ScaleWrapper = {x2ScaleReal, x2ScaleAclDtype};
    TensorWrapper gatherOutWrapper = {gatherOut, x1AclDtype};

    int64_t groupSize = CheckAndGetGroupSizeAg(groupSizes);

    ACLNN_CMD(aclnnAllGatherQuantMatmulV3, context, x1Wrapper, x2Wrapper, biasOptional, x1ScaleWrapper, x2ScaleWrapper,
              groupPtr, rankSize, hcclBufferSize, groupSize, commModePtr, y, gatherOutWrapper);

    return std::make_tuple(y, gatherOut, amaxOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_all_gather_quant_matmul", &NpuAllGatherQuantMatmul, "npu_all_gather_quant_matmul");
}

} // namespace op_api
