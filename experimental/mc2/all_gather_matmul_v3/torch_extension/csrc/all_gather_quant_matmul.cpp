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
#include <cstdint>
#include <cstring>
#include <string>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnn_common.h"
#include "hccl_common.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/utils/comm_channel_builder.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;
const int DIM_THREE = 3;
const int DIM_FOUR = 4;
const int ACL_FORMAT_NZ_VALUE = 29; // ACL_FORMAT_FRACTAL_NZ
static const int64_t GROUP_MAX = 65535;
static const size_t GROUP_DIM = 3;
static const size_t OFFSET_32_BITS = 32;
static const size_t OFFSET_16_BITS = 16;
const std::set<int> SUPPORT_RANK_SIZE_LIST{2, 4, 8};

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
    const at::Tensor& context, const at::Tensor& x1, const at::Tensor& x2, int64_t hcclBufferSize, std::string group,
    int64_t rankSize, const c10::optional<at::Tensor>& biasOptional, const c10::optional<at::Tensor>& x1ScaleOptional,
    const c10::optional<at::Tensor>& x2ScaleOptional, c10::IntArrayRef groupSizes, c10::optional<int64_t> x1Dtype,
    c10::optional<int64_t> x2Dtype, c10::optional<int64_t> x1ScaleDtype, c10::optional<int64_t> x2ScaleDtype,
    c10::optional<int64_t> yDtype, std::string commMode)
{
    TORCH_CHECK(x1.dim() == DIM_TWO,
                "The x1 input of allgatherquantmatmul is required to be 2D, but the actual x1 input is ", x1.dim(),
                "D.");
    TORCH_CHECK(x2.dim() == DIM_TWO || x2.dim() == DIM_FOUR,
                "The x2 input of allgatherquantmatmul is required to be 2D (ND/NZ logical) or 4D "
                "(FRACTAL_NZ storage), but the actual x2 input is ",
                x2.dim(), "D.");
    TORCH_CHECK(SUPPORT_RANK_SIZE_LIST.find(rankSize) != SUPPORT_RANK_SIZE_LIST.end(),
                "The rank_size should be in [2, 4, 8], but the actual value is ", rankSize, ".");
    TORCH_CHECK(x1ScaleOptional.has_value() && x1ScaleOptional->dim() == DIM_THREE,
                "x1_scale is required for MX quant and must be 3D [M, ceil(K/64), 2].");
    TORCH_CHECK(x2ScaleOptional.has_value() && x2ScaleOptional->dim() == DIM_THREE,
                "x2_scale is required for MX quant and must be 3D "
                "([N, ceil(K/64), 2] for transposed B, [ceil(K/64), N, 2] for non-transposed B).");

    auto x1Size = x1.sizes();
    auto x2Size = x2.sizes();
    int64_t mPerRank = x1Size[0];
    const bool x2Nz = (at_npu::native::get_npu_format(x2) == ACL_FORMAT_NZ_VALUE);
    const bool x2IsFp4 = x2Dtype.has_value() && GetAclDataType(x2Dtype.value()) == ACL_FLOAT4_E2M1;
    // fp4 打包存储(2 元素/字节)时 x1 末维为 K/2, 方向判断用逻辑 K
    const int64_t k = x2IsFp4 ? x1Size[1] * 2 : x1Size[1];
    const int64_t kd = (k + 63) / 64;
    const int64_t sD0 = x2ScaleOptional->size(0);
    const int64_t sD1 = x2ScaleOptional->size(1);
    int64_t n = 0;
    if (sD0 == kd && sD1 != kd) {
        n = sD1; // 非转置 B: scale=[kd,n,2], N=scale.dim1
    } else if (sD1 == kd && sD0 != kd) {
        n = sD0; // 转置 B: scale=[n,kd,2], N=scale.dim0
    } else {
        n = (sD0 == kd) ? sD1 : sD0;
    }

    at::ScalarType outScalarType = at::kBFloat16;
    if (yDtype.has_value()) {
        aclDataType outAclDtype = GetAclDataType(yDtype.value());
        TORCH_CHECK(outAclDtype == ACL_BF16 || outAclDtype == ACL_FLOAT16,
                    "y_dtype only supports bf16(15) or fp16(5) in the ubmem path, but got invalid enum value ",
                    yDtype.value(), ".");
        outScalarType = (outAclDtype == ACL_FLOAT16) ? at::kHalf : at::kBFloat16;
    }

    at::Tensor y = at::empty({mPerRank * rankSize, n}, x1.options().dtype(outScalarType));

    std::string commModeStr = std::string(commMode);
    char* commModePtr = const_cast<char*>(commModeStr.c_str());

    std::string groupStr = std::string(group);
    char* groupPtr = const_cast<char*>(groupStr.c_str());

    // x1/x2 支持原生 fp8(e4m3/e5m2) 或 fp4(打包 uint8 + dtype 枚举 296)，
    auto isFp4Storage = [](const at::Tensor& t) { return t.scalar_type() == at::kByte; };
    for (const auto* item : {&x1, &x2}) {
        const auto st = item->scalar_type();
        TORCH_CHECK(st == at::ScalarType::Float8_e4m3fn || st == at::ScalarType::Float8_e5m2 || st == at::kByte,
                    "x1/x2 only support torch.float8_e4m3fn, torch.float8_e5m2 or uint8 "
                    "(packed fp4, pass x1_dtype/x2_dtype=296), but got tensor dtype ",
                    st, ".");
    }
    aclDataType x1AclDtype =
        x1Dtype.has_value() ? GetAclDataType(x1Dtype.value()) : ConvertToAclDataType(x1.scalar_type());
    aclDataType x2AclDtype =
        x2Dtype.has_value() ? GetAclDataType(x2Dtype.value()) : ConvertToAclDataType(x2.scalar_type());
    auto isSupportedX = [](aclDataType dt) {
        return dt == ACL_FLOAT8_E4M3FN || dt == ACL_FLOAT8_E5M2 || dt == ACL_FLOAT4_E2M1;
    };
    TORCH_CHECK(isSupportedX(x1AclDtype), "x1 dtype enum is not supported, got ", static_cast<int>(x1AclDtype),
                " (expected 24 e4m3 / 23 e5m2 / 296 fp4).");
    TORCH_CHECK(isSupportedX(x2AclDtype), "x2 dtype enum is not supported, got ", static_cast<int>(x2AclDtype),
                " (expected 24 e4m3 / 23 e5m2 / 296 fp4).");
    // fp4 必须用 uint8 打包存储; fp8 枚举允许原生 fp8 存储或 uint8 bitcast
    // (例如 e5m2 无法直接 npu_format_cast, 以 uint8 存储 + dtype 枚举传入)
    TORCH_CHECK((x1AclDtype != ACL_FLOAT4_E2M1) || isFp4Storage(x1),
                "x1 dtype enum 296 (fp4) requires uint8 packed storage.");
    TORCH_CHECK((x2AclDtype != ACL_FLOAT4_E2M1) || isFp4Storage(x2),
                "x2 dtype enum 296 (fp4) requires uint8 packed storage.");
    TORCH_CHECK((x1AclDtype == ACL_FLOAT4_E2M1) == (x2AclDtype == ACL_FLOAT4_E2M1),
                "FP4 requires both x1 and x2 to be FP4 (x1_dtype=x2_dtype=296).");
    if (x2.dim() == DIM_FOUR || at_npu::native::get_npu_format(x2) == ACL_FORMAT_NZ_VALUE) {
        TORCH_CHECK(x1AclDtype != ACL_FLOAT4_E2M1, "FP4 does not support FRACTAL_NZ x2.");
    }

    // gather_out / amax_out 当前不使能，
    // torch 侧传 {0} 空占位 tensor 经 aclnn 透传 inner（aclnn/tiling/kernel 均不处理），并原样返回给 torch。
    at::Tensor gatherOut = at::empty({0}, x1.options().dtype(x1.scalar_type()));
    at::Tensor amaxOut = at::empty({0}, x1.options().dtype(at::kFloat));

    const at::Tensor& x1ScaleReal = x1ScaleOptional.value_or(at::Tensor());
    const at::Tensor& x2ScaleReal = x2ScaleOptional.value_or(at::Tensor());

    // scale 原始 dtype 校验：仅支持 uint8(packed, 搭配 e8m0 枚举) 或原生 float8_e8m0fnu。
    if (x1ScaleReal.defined()) {
        auto scaleSt = x1ScaleReal.scalar_type();
        TORCH_CHECK(scaleSt == at::kByte || scaleSt == at::ScalarType::Float8_e8m0fnu,
                    "x1_scale only supports torch.uint8 (packed, pass x1_scale_dtype=293) or "
                    "torch.float8_e8m0fnu, but got tensor dtype ",
                    scaleSt, ". Dtype enum cannot override an unsupported storage dtype.");
    }
    if (x2ScaleReal.defined()) {
        auto scaleSt = x2ScaleReal.scalar_type();
        TORCH_CHECK(scaleSt == at::kByte || scaleSt == at::ScalarType::Float8_e8m0fnu,
                    "x2_scale only supports torch.uint8 (packed, pass x2_scale_dtype=293) or "
                    "torch.float8_e8m0fnu, but got tensor dtype ",
                    scaleSt, ". Dtype enum cannot override an unsupported storage dtype.");
    }

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

using ApaceCommUbmemContext = Apace::AivComm::CommUbmemContext;
using UbmemCommChannelBuilder = CommChannelBuilder<ChannelMode::UBMEM>; // apace内comm_channel_builder.h

/*!
 * \brief 创建 UBMEM 模式 device context 并包装为 at::Tensor 返回
 *
 * 内部流程：
 * 1. 动态加载 HCCL 函数指针，从 group name 获取 HcclComm handle
 * 2. CommChannelBuilder<UBMEM>::CreateDeviceContext：engine ctx 按 tag 幂等复用，新建时
 *    清零内置 buffer、建链并填充 rankId/rankSize/commBufferAddrs[]
 */
at::Tensor CreateUbmemContext(std::string group, std::string ctxTag)
{
    InitHcclEngineCtxFunctions();

    HcclComm comm = nullptr;
    auto hcclRet = HcomGetCommHandleByGroupFunc(group.c_str(), &comm); // group名-> HCCL comm句柄
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group, ", ret: ", hcclRet);

    UbmemCommChannelBuilder builder(comm);
    ApaceCommUbmemContext hostCtx = {};
    void* devCtx = builder.CreateDeviceContext(&hostCtx, sizeof(ApaceCommUbmemContext), ctxTag.c_str(), &hostCtx);
    TORCH_CHECK(devCtx != nullptr, "CreateDeviceContext failed, ctxTag: ", ctxTag);

    at::Tensor context =
        at::from_blob(devCtx, {static_cast<int64_t>(sizeof(ApaceCommUbmemContext) / sizeof(int32_t))}, at::kInt);
    return context;
}

/*!
 * \brief 获取 group 对应 HCCL 内置 buffer 大小（供 host 侧窗口容量预检）
 */
int64_t GetHcclBufferSizeByGroup(std::string group)
{
    InitHcclEngineCtxFunctions();

    HcclComm comm = nullptr;
    auto hcclRet = HcomGetCommHandleByGroupFunc(group.c_str(), &comm);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group, ", ret: ", hcclRet);

    void* buf = nullptr;
    uint64_t bufSize = 0;
    hcclRet = HcclGetHcclBufferFunc(comm, &buf, &bufSize);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS && buf != nullptr, "HcclGetHcclBuffer failed, ret: ", hcclRet);
    return static_cast<int64_t>(bufSize);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_all_gather_quant_matmul", &NpuAllGatherQuantMatmul, "npu_all_gather_quant_matmul");
    m.def("create_ubmem_context", &CreateUbmemContext, "create_ubmem_context", py::arg("group"), py::arg("ctx_tag"));
    m.def("get_hccl_buffer_size", &GetHcclBufferSizeByGroup, "get_hccl_buffer_size", py::arg("group"));
}

} // namespace op_api
