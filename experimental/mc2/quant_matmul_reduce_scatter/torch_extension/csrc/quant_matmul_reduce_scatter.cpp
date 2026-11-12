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
using ApaceCommUbmemContext = Apace::AivComm::CommUbmemContext;
using UbmemCommChannelBuilder = CommChannelBuilder<ChannelMode::UBMEM>;
const int DIM_TWO = 2;
const int DIM_FOUR = 4;
const size_t GROUP_DIM = 3;
const int64_t GROUP_MAX = 65535;
const uint64_t OFFSET_32_BITS = 32;
const uint64_t OFFSET_16_BITS = 16;
const std::set<int> SUPPORT_RANK_SIZE_LIST{2, 4, 8, 16};

/*!
 * \brief 创建 UBMEM 模式 device context 并包装为 at::Tensor 返回
 * 内部流程：
 * 1. 从 group name 获取 HcclComm handle
 * 2. 调用 CommChannelBuilder<UBMEM>::CreateDeviceContext(hostCtx, size, tag, hostCtx) 填充 CommUbmemContext
 * 3. 用 at::from_blob 将返回的 device 指针包装为 at::Tensor
 */
at::Tensor CreateUbmemContext(std::string group, std::string ctxTag)
{
    InitHcclEngineCtxFunctions();

    HcclComm comm = nullptr;
    auto hcclRet = HcomGetCommHandleByGroupFunc(group.c_str(), &comm);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group, ", ret: ", hcclRet);

    UbmemCommChannelBuilder builder(comm);
    ApaceCommUbmemContext hostCtx = {};
    void* devCtx = builder.CreateDeviceContext(&hostCtx, sizeof(ApaceCommUbmemContext), ctxTag.c_str(), &hostCtx);
    TORCH_CHECK(devCtx != nullptr, "CreateDeviceContext failed, ctxTag: ", ctxTag);

    at::Tensor context = at::from_blob(devCtx, {sizeof(ApaceCommUbmemContext) / sizeof(int32_t)}, at::kInt);
    return context;
}

// group_sizes [M, N, K] -> 位打包 int64：空返回 0（自动推断 [1,1,32]）
int64_t CheckAndGetGroupSizeAg(at::IntArrayRef groupSizeList)
{
    if (groupSizeList.empty()) {
        return 0;
    }
    size_t groupDim = groupSizeList.size();
    TORCH_CHECK(groupDim == GROUP_DIM, "group_sizes only supports input with three elements, but got ", groupDim);
    int64_t groupM = static_cast<int64_t>(groupSizeList[0]);
    int64_t groupN = static_cast<int64_t>(groupSizeList[1]);
    int64_t groupK = static_cast<int64_t>(groupSizeList[2]);
    bool invalid_group_param = ((groupM <= GROUP_MAX && groupM >= 0) && (groupN <= GROUP_MAX && groupN >= 0) &&
                                (groupK <= GROUP_MAX && groupK >= 0));
    TORCH_CHECK(invalid_group_param, "group param value must conform to range [0, 65535]");
    return static_cast<int64_t>((static_cast<uint64_t>(groupM) << OFFSET_32_BITS) +
                                (static_cast<uint64_t>(groupN) << OFFSET_16_BITS) + static_cast<uint64_t>(groupK));
}

/*!
 * \brief 获取 group 对应 HCCL 内置 buffer 大小（供 host 侧容量预检）
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

at::Tensor NpuQuantMatmulReduceScatter(const at::Tensor& context, const at::Tensor& x1, const at::Tensor& x2,
                                       const at::Tensor& x1Scale, const at::Tensor& x2Scale, int64_t hcclBufferSize,
                                       std::string group, int64_t rankSize, const c10::optional<at::Tensor>& bias,
                                       c10::IntArrayRef groupSizes, c10::optional<int64_t> x1Dtype,
                                       c10::optional<int64_t> x2Dtype, c10::optional<int64_t> x1ScaleDtype,
                                       c10::optional<int64_t> x2ScaleDtype, c10::optional<int64_t> yDtype,
                                       std::string commMode)
{
    TORCH_CHECK(x1.dim() == DIM_TWO,
                "The x1 input of quantmatmulreducescatter is required to be 2D, but the actual x1 input is ", x1.dim(),
                "D.");
    TORCH_CHECK(x2.dim() == DIM_TWO || x2.dim() == DIM_FOUR,
                "The x2 input of quantmatmulreducescatter is required to be 2D (ND) or 4D (FRACTAL_NZ), but the actual "
                "x2 input is ",
                x2.dim(), "D.");
    TORCH_CHECK(SUPPORT_RANK_SIZE_LIST.find(rankSize) != SUPPORT_RANK_SIZE_LIST.end(),
                "The rank_size should be in [2, 4, 8, 16], but the actual value is ", rankSize, ".");
    // bias 当前不支持：非空即报错
    TORCH_CHECK(!bias.has_value() || !bias->defined(),
                "bias not supported in ubmem path of quant_matmul_reduce_scatter");
    const bool groupSizesValid = groupSizes.empty() || (groupSizes.size() == 3 && groupSizes[0] == 1 &&
                                                        groupSizes[1] == 1 && groupSizes[2] == 32);
    TORCH_CHECK(
        groupSizesValid,
        "group_sizes only supports empty (auto-infer) or [1, 1, 32] (MX fixed group) in the ubmem path, but got ",
        groupSizes.size(), " elements.");
    TORCH_CHECK(commMode == "ubmem", "comm_mode only supports 'ubmem' in this experimental build, but got '", commMode,
                "'.");
    TORCH_CHECK(hcclBufferSize > 0, "hccl_buffer_size must be positive, but got ", hcclBufferSize, ".");
    const int64_t groupSize = CheckAndGetGroupSizeAg(groupSizes);

    auto x1Size = x1.sizes();
    int64_t m = x1Size[0];
    // N 推导统一以 x2_scale 形状为权威信号（显式构造，不受 fp4 打包 / .t() 视图干扰）：
    //   非转置 B -> scale=[kd,n,2]（sD0==kd）；转置 B -> scale=[n,kd,2]（sD1==kd）
    const bool x2IsFp4 = x2Dtype.has_value() && GetAclDataType(x2Dtype.value()) == ACL_FLOAT4_E2M1;
    // fp4 打包存储(2 元素/字节)时 x1 末维为 K/2, 方向判断用逻辑 K
    const int64_t k = x2IsFp4 ? x1Size[1] * 2 : x1Size[1];
    const int64_t kd = (k + 63) / 64;
    int64_t n = 0;
    // 2D / NZ 4D：N 统一由 x2_scale 形状权威推导（显式构造，精确且不受打包/视图干扰）
    const int64_t sD0 = x2Scale.size(0);
    const int64_t sD1 = x2Scale.size(1);
    if (sD0 == kd && sD1 != kd) {
        n = sD1; // 非转置 B：scale=[kd,n,2]，N=scale.dim1
    } else if (sD1 == kd && sD0 != kd) {
        n = sD0; // 转置 B：scale=[n,kd,2]，N=scale.dim0
    } else {
        n = (sD0 == kd) ? sD1 : sD0; // K==N 等歧义，仍按 scale 取 N
    }
    TORCH_CHECK(m % rankSize == 0, "The x1 dim0 should be divisible by rank_size, but got ", m, " % ", rankSize,
                " != 0.");

    at::ScalarType yScalar = at::kBFloat16;
    if (yDtype.has_value()) {
        aclDataType yAcl = GetAclDataType(yDtype.value());
        if (yAcl == ACL_BF16) {
            yScalar = at::kBFloat16;
        } else if (yAcl == ACL_FLOAT16) {
            yScalar = at::kHalf;
        } else {
            TORCH_CHECK(false, "y_dtype only supports 283 (BF16) or 257 (FP16), but got ", yDtype.value(), ".");
        }
    }
    at::Tensor y = at::empty({m / rankSize, n}, x1.options().dtype(yScalar));

    std::string groupStr = std::string(group);
    char* groupPtr = const_cast<char*>(groupStr.c_str());
    char* commModePtr = const_cast<char*>(commMode.c_str());

    // x1/x2 原始 dtype 校验：fp8 支持原生 e4m3fn/e5m2，或 uint8（bitcast，搭配 291=e5m2/292=e4m3 枚举）
    // fp4 以 uint8 打包存储（2 元素/字节，末维为 K/2），搭配 296=ACL_FLOAT4_E2M1 枚举
    auto isSupportedXStorageDtype = [](const at::Tensor& t) {
        auto st = t.scalar_type();
        return st == at::ScalarType::Float8_e4m3fn || st == at::ScalarType::Float8_e5m2 || st == at::kByte;
    };
    TORCH_CHECK(
        isSupportedXStorageDtype(x1),
        "x1 only supports torch.float8_e4m3fn, torch.float8_e5m2 or torch.uint8 "
        "(fp8 bitcast: pass x1_dtype=291/292; fp4 packed 2 elems/byte: pass x1_dtype=296), but got tensor dtype ",
        x1.scalar_type(), ". Dtype enum cannot override an unsupported storage dtype.");
    TORCH_CHECK(
        isSupportedXStorageDtype(x2),
        "x2 only supports torch.float8_e4m3fn, torch.float8_e5m2 or torch.uint8 "
        "(fp8 bitcast: pass x2_dtype=291/292; fp4 packed 2 elems/byte: pass x2_dtype=296), but got tensor dtype ",
        x2.scalar_type(), ". Dtype enum cannot override an unsupported storage dtype.");

    aclDataType x1AclDtype =
        x1Dtype.has_value() ? GetAclDataType(x1Dtype.value()) : ConvertToAclDataType(x1.scalar_type());
    aclDataType x2AclDtype =
        x2Dtype.has_value() ? GetAclDataType(x2Dtype.value()) : ConvertToAclDataType(x2.scalar_type());

    TORCH_CHECK((x1AclDtype == ACL_FLOAT4_E2M1) == (x2AclDtype == ACL_FLOAT4_E2M1),
                "fp4 requires both x1 and x2 to be fp4 (x1_dtype=296 and x2_dtype=296), "
                "fp8 requires both to be fp8: got x1 acl dtype ",
                static_cast<int>(x1AclDtype), ", x2 acl dtype ", static_cast<int>(x2AclDtype), ".");

    TORCH_CHECK(x1Scale.scalar_type() == at::ScalarType::Float8_e8m0fnu || x1Scale.scalar_type() == at::kByte,
                "x1_scale only supports torch.float8_e8m0fnu or torch.uint8 (packed, pass "
                "x1_scale_dtype=293), but got tensor dtype ",
                x1Scale.scalar_type(), ". Dtype enum cannot override an unsupported storage dtype.");
    TORCH_CHECK(x2Scale.scalar_type() == at::ScalarType::Float8_e8m0fnu || x2Scale.scalar_type() == at::kByte,
                "x2_scale only supports torch.float8_e8m0fnu or torch.uint8 (packed, pass "
                "x2_scale_dtype=293), but got tensor dtype ",
                x2Scale.scalar_type(), ". Dtype enum cannot override an unsupported storage dtype.");

    aclDataType x1ScaleAclDtype =
        x1ScaleDtype.has_value() ? GetAclDataType(x1ScaleDtype.value()) : ConvertToAclDataType(x1Scale.scalar_type());
    aclDataType x2ScaleAclDtype =
        x2ScaleDtype.has_value() ? GetAclDataType(x2ScaleDtype.value()) : ConvertToAclDataType(x2Scale.scalar_type());

    TensorWrapper x1Wrapper = {x1, x1AclDtype};
    TensorWrapper x2Wrapper = {x2, x2AclDtype};
    TensorWrapper x1ScaleWrapper = {x1Scale, x1ScaleAclDtype};
    TensorWrapper x2ScaleWrapper = {x2Scale, x2ScaleAclDtype};

    ACLNN_CMD(aclnnQuantMatmulReduceScatter, context, x1Wrapper, x2Wrapper, x1ScaleWrapper, x2ScaleWrapper, groupPtr,
              rankSize, hcclBufferSize, groupSize, commModePtr, y);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_quant_matmul_reduce_scatter", &NpuQuantMatmulReduceScatter, "npu_quant_matmul_reduce_scatter");
    m.def("create_ubmem_context", &CreateUbmemContext, "create_ubmem_context");
    m.def("get_hccl_buffer_size", &GetHcclBufferSizeByGroup, "get_hccl_buffer_size");
}

} // namespace op_api
