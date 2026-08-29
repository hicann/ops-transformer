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
static const int64_t DYN_PERTOKEN_QUANT_MODE = 7;
static const int64_t PERCHANNEL_QUANT_MODE = 2;
static const int64_t INT4_NUMS_IN_INT32 = 8;
static const int64_t GROUP_MAX = 65535;
static const size_t GROUP_DIM = 3;
static const size_t OFFSET_32_BITS = 32;
static const size_t OFFSET_16_BITS = 16;
const std::set<int> SUPPORT_WORLD_SIZE_LIST{2, 4, 8, 16};

int64_t CheckAndGetGroupSize(at::IntArrayRef groupSizeList)
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

static void CheckNpuAlltoAllQuantMatmulInputs(const at::Tensor &x1, const at::Tensor &x2, int64_t worldSize)
{
    TORCH_CHECK(x1.dim() == DIM_TWO,
                "The x1 input of alltoallquantmatmul is required to be 2D, but the actual x1 input is ", x1.dim(),
                "D.");
    TORCH_CHECK(x2.dim() == DIM_TWO,
                "The x2 input of alltoallquantmatmul is required to be 2D, but the actual x2 input is ", x2.dim(),
                "D.");
    TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(worldSize) != SUPPORT_WORLD_SIZE_LIST.end(),
                "The world_size should be in [2, 4, 8, 16], but the actual value is ", worldSize, ".");
}

static at::ScalarType GetOutputScalarType(c10::optional<int64_t> yDtype)
{
    if (!yDtype.has_value() || yDtype.value() == static_cast<int64_t>(-1)) {
        return at::kFloat;
    }
    aclDataType outAclDtype = GetAclDataType(yDtype.value());
    if (outAclDtype == ACL_FLOAT) {
        return at::kFloat;
    } else if (outAclDtype == ACL_FLOAT16) {
        return at::kHalf;
    } else if (outAclDtype == ACL_BF16) {
        return at::kBFloat16;
    }
    return at::kFloat;
}

static aclDataType GetTensorAclDtype(c10::optional<int64_t> dtype, const at::Tensor &tensor)
{
    if (dtype.has_value() && dtype.value() != -1) {
        return GetAclDataType(dtype.value());
    }
    return tensor.defined() ? ConvertToAclDataType(tensor.scalar_type()) : ACL_DT_UNDEFINED;
}

std::tuple<at::Tensor, at::Tensor> NpuAlltoAllQuantMatmul(
    const at::Tensor &context, const at::Tensor &x1, const at::Tensor &x2, int64_t hcclBufferSize, std::string group,
    int64_t worldSize, const c10::optional<at::Tensor> &biasOptional, const c10::optional<at::Tensor> &x1ScaleOptional,
    const c10::optional<at::Tensor> &x2ScaleOptional, c10::optional<int64_t> x1QuantMode,
    c10::optional<int64_t> x2QuantMode, c10::IntArrayRef groupSizes, c10::optional<int64_t> x1Dtype,
    c10::optional<int64_t> x2Dtype, c10::optional<int64_t> x1ScaleDtype, c10::optional<int64_t> x2ScaleDtype,
    c10::optional<int64_t> yDtype, std::string commMode, int64_t precisionMode)
{
    CheckNpuAlltoAllQuantMatmulInputs(x1, x2, worldSize);

    bool is_w4 = x2.dtype() == at::kInt;
    auto x1Size = x1.sizes();
    auto x2Size = x2.sizes();
    int64_t bs = x1Size[0];
    int64_t h = x1Size[1];
    int64_t localBs = bs / worldSize;
    int64_t n = is_w4 ? x2Size[1] * INT4_NUMS_IN_INT32 : x2Size[1];

    TORCH_CHECK(bs % worldSize == 0, "The first dim of x1 (", bs, ") should be divisible by world_size (", worldSize,
                ")");

    at::ScalarType outScalarType = GetOutputScalarType(yDtype);

    at::Tensor y = at::empty({localBs, n}, x1.options().dtype(outScalarType));
    at::Tensor alltoallOut = at::empty({localBs, h * worldSize}, x1.options().dtype(x1.scalar_type()));

    std::string commModeStr(commMode);
    std::string groupStr(group);
    char *commModePtr = const_cast<char *>(commModeStr.c_str());
    char *groupPtr = const_cast<char *>(groupStr.c_str());
    int64_t x1QuantModeVal = x1QuantMode.has_value() ? x1QuantMode.value() : DYN_PERTOKEN_QUANT_MODE;
    int64_t x2QuantModeVal = x2QuantMode.has_value() ? x2QuantMode.value() : PERCHANNEL_QUANT_MODE;

    aclDataType x1AclDtype = GetTensorAclDtype(x1Dtype, x1);
    aclDataType x2AclDtype = GetTensorAclDtype(x2Dtype, x2);
    const at::Tensor &x1ScaleReal = x1ScaleOptional.value_or(at::Tensor());
    const at::Tensor &x2ScaleReal = x2ScaleOptional.value_or(at::Tensor());
    aclDataType x1ScaleAclDtype = GetTensorAclDtype(x1ScaleDtype, x1ScaleReal);
    aclDataType x2ScaleAclDtype = GetTensorAclDtype(x2ScaleDtype, x2ScaleReal);

    TensorWrapper x1Wrapper = {x1, x1AclDtype};
    TensorWrapper x2Wrapper = {x2, x2AclDtype};
    TensorWrapper x1ScaleWrapper = {x1ScaleReal, x1ScaleAclDtype};
    TensorWrapper x2ScaleWrapper = {x2ScaleReal, x2ScaleAclDtype};
    TensorWrapper alltoallOutWrapper = {alltoallOut, x1AclDtype};

    int64_t groupSize = CheckAndGetGroupSize(groupSizes);

    ACLNN_CMD(AlltoAllMatmulV2, context, x1Wrapper, x2Wrapper, biasOptional, x1ScaleWrapper, x2ScaleWrapper, groupPtr,
              worldSize, hcclBufferSize, x1QuantModeVal, x2QuantModeVal, groupSize, commModePtr, precisionMode, y,
              alltoallOutWrapper);

    return std::make_tuple(y, alltoallOut);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_all_to_all_quant_matmul", &NpuAlltoAllQuantMatmul, "npu_all_to_all_quant_matmul");
}

} // namespace op_api
