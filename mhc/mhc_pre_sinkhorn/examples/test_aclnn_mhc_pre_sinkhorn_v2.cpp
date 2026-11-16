/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_sinkhorn.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

void PrintTensorDataFloat(const std::vector<int64_t>& shape, void* device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(10), size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

void PrintTensorDataBfloat16(const std::vector<int64_t>& shape, void* device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_bf16(size);

    aclError ret = aclrtMemcpy(host_bf16.data(), size * sizeof(uint16_t), device_addr, size * sizeof(uint16_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements, BF16 hex): ");
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(10), size); ++i) {
        LOG_PRINT("0x%04x ", host_bf16[i]);
    }
    LOG_PRINT("\n");
}

int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed, error: %d\n", ret); return -1);

    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); return -1);

    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); return -1);

    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); return -1);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); return -1);

    return 0;
}

int CreateAclTensorFloat32(const std::vector<float>& host_data, const std::vector<int64_t>& shape, void*& device_addr,
                           aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorBfloat16(const std::vector<float>& host_data, const std::vector<int64_t>& shape, void*& device_addr,
                            aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_data_bf16(size);
    for (int64_t i = 0; i < size; ++i) {
        // fp32 -> bf16 标准舍入: 取高16位(符号1+指数8+尾数7), 低位四舍五入;
        // 不可用 x*32767 截断近似, 会改变数值幅值
        uint32_t bits = 0;
        std::memcpy(&bits, &host_data[i], sizeof(bits));
        uint32_t lsb = (bits >> 16) & 1U;
        bits = ((bits + (0x7FFFU + lsb)) >> 16) & 0xFFFFU;
        host_data_bf16[i] = static_cast<uint16_t>(bits);
    }

    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, byte_size, host_data_bf16.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorBfloat16Output(const std::vector<int64_t>& shape, void*& device_addr, aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorFloat32Output(const std::vector<int64_t>& shape, void*& device_addr, aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

struct Tensors {
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr;
    void* premix_addr = nullptr;
    void *hin_addr = nullptr, *h_post_addr = nullptr, *h_res_addr = nullptr;
    void *h_pre_addr = nullptr, *hc_before_norm_addr = nullptr, *inv_rms_addr = nullptr;
    void *sum_out_addr = nullptr, *norm_out_addr = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
    aclTensor* premix = nullptr;
    aclTensor *hin = nullptr, *h_post = nullptr, *h_res = nullptr;
    aclTensor *h_pre = nullptr, *hc_before_norm = nullptr, *inv_rms = nullptr;
    aclTensor *sum_out = nullptr, *norm_out = nullptr;
};

int CreateInputTensors(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& phi_shape,
                       const std::vector<int64_t>& alpha_shape, const std::vector<int64_t>& bias_shape,
                       const std::vector<int64_t>& premix_shape, Tensors& tensors)
{
    // 温和伪随机数据: 全 1.0 会使 x*phi 矩阵乘溢出 sigmoid 饱和, sinkhorn 输出 NaN。
    // 注意用有符号 int64_t 循环变量: size_t 下 (mod - 9) 会无符号下溢成极大值
    std::vector<float> x_host_data(GetShapeSize(x_shape), 0.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(x_host_data.size()); ++i) {
        x_host_data[i] = 0.5f * static_cast<float>((i * 37 % 19) - 9) / 9.0f;
    }
    std::vector<float> phi_host_data(GetShapeSize(phi_shape), 0.0f);
    float phi_scale = 16.0f * static_cast<float>(phi_shape[1]); // 16 * (N*C)
    for (int64_t i = 0; i < static_cast<int64_t>(phi_host_data.size()); ++i) {
        phi_host_data[i] = static_cast<float>((i * 91 % 17) - 8) / phi_scale;
    }
    std::vector<float> alpha_host_data = {0.3f, 0.3f, 0.3f};
    std::vector<float> bias_host_data(GetShapeSize(bias_shape), 0.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(bias_host_data.size()); ++i) {
        bias_host_data[i] = 0.05f * static_cast<float>((i * 13 % 7) - 3);
    }
    std::vector<float> premix_host_data(GetShapeSize(premix_shape), 0.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(premix_host_data.size()); ++i) {
        premix_host_data[i] = 0.5f + static_cast<float>(i * 11 % 10) / 20.0f; // [0.5, 0.95]
    }

    int ret = CreateAclTensorBfloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(premix_host_data, premix_shape, tensors.premix_addr, tensors.premix);
    CHECK_RET(ret == 0, LOG_PRINT("Create premix_tensor failed\n"); return -1);
    return 0;
}

int CreateOutputTensors(const std::vector<int64_t>& hin_shape, const std::vector<int64_t>& h_post_shape,
                        const std::vector<int64_t>& h_res_shape, const std::vector<int64_t>& h_pre_shape,
                        const std::vector<int64_t>& hc_before_norm_shape, const std::vector<int64_t>& inv_rms_shape,
                        const std::vector<int64_t>& sum_out_shape, const std::vector<int64_t>& norm_out_shape,
                        Tensors& tensors)
{
    int ret = CreateAclTensorBfloat16Output(hin_shape, tensors.hin_addr, tensors.hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create hin_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_post_shape, tensors.h_post_addr, tensors.h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_res_shape, tensors.h_res_addr, tensors.h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(hc_before_norm_shape, tensors.hc_before_norm_addr, tensors.hc_before_norm);
    CHECK_RET(ret == 0, LOG_PRINT("Create hc_before_norm_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(sum_out_shape, tensors.sum_out_addr, tensors.sum_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(norm_out_shape, tensors.norm_out_addr, tensors.norm_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_out_tensor failed\n"); return -1);
    return 0;
}

void DestroyTensors(Tensors& tensors)
{
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.premix);
    aclDestroyTensor(tensors.hin);
    aclDestroyTensor(tensors.h_post);
    aclDestroyTensor(tensors.h_res);
    aclDestroyTensor(tensors.h_pre);
    aclDestroyTensor(tensors.hc_before_norm);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.sum_out);
    aclDestroyTensor(tensors.norm_out);
}

void FreeDeviceMemory(Tensors& tensors)
{
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.premix_addr);
    aclrtFree(tensors.hin_addr);
    aclrtFree(tensors.h_post_addr);
    aclrtFree(tensors.h_res_addr);
    aclrtFree(tensors.h_pre_addr);
    aclrtFree(tensors.hc_before_norm_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.sum_out_addr);
    aclrtFree(tensors.norm_out_addr);
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    Tensors tensors;

    int64_t bs = 2, seq_len = 128, n = 4, c = 4096;
    int64_t hc_mult = n;
    int64_t hc_mix = n * n + 2 * n;
    int64_t num_iters = 20;
    float hc_eps = 1e-6f, norm_eps = 1e-6f;
    bool need_backward = true;
    bool enable_hf32 = true;

    std::vector<int64_t> x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> phi_shape = {hc_mix, n * c};
    std::vector<int64_t> alpha_shape = {3};
    std::vector<int64_t> bias_shape = {hc_mix};
    std::vector<int64_t> premix_shape = {bs, seq_len, n};

    std::vector<int64_t> hin_shape = {bs, seq_len, c};
    std::vector<int64_t> h_post_shape = {bs, seq_len, n};
    std::vector<int64_t> h_res_shape = {bs, seq_len, n * n};
    std::vector<int64_t> h_pre_shape = {bs, seq_len, n};
    std::vector<int64_t> hc_before_norm_shape = {bs, seq_len, hc_mix};
    std::vector<int64_t> inv_rms_shape = {bs, seq_len, 1};
    std::vector<int64_t> sum_out_shape = {num_iters * 2, bs, seq_len, n};
    std::vector<int64_t> norm_out_shape = {num_iters * 2, bs, seq_len, n, n};

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);

    ret = CreateInputTensors(x_shape, phi_shape, alpha_shape, bias_shape, premix_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    ret = CreateOutputTensors(hin_shape, h_post_shape, h_res_shape, h_pre_shape, hc_before_norm_shape, inv_rms_shape,
                              sum_out_shape, norm_out_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // V2 接口: 5 张量输入(含可选 premix) + 6 attr(含 enable_hf32), 8 输出。
    // premix 非空时 hin 使用 premix 加权(替换内部 sinkhorn 权重 h_pre);
    // premix 为 nullptr 时与 V1 行为一致(走 nullptr 即可, 本例展示 premix 用法)。
    aclnnStatus aclnn_ret = aclnnMhcPreSinkhornV2GetWorkspaceSize(
        tensors.x, tensors.phi, tensors.alpha, tensors.bias, tensors.premix, hc_mult, num_iters, hc_eps, norm_eps,
        need_backward, enable_hf32, tensors.hin, tensors.h_post, tensors.h_res, tensors.h_pre, tensors.hc_before_norm,
        tensors.inv_rms, tensors.sum_out, tensors.norm_out, &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcPreSinkhornV2GetWorkspaceSize failed, error: %d\n", aclnn_ret);
              FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);

    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        aclError acl_ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed, error: %d\n", acl_ret);
                  FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);
    }

    aclnn_ret = aclnnMhcPreSinkhornV2(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhornV2 failed, error: %d\n", aclnn_ret);
              FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);

    aclError acl_ret = aclrtSynchronizeStream(stream);
    CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", acl_ret);
              FreeDeviceMemory(tensors); DestroyTensors(tensors); return -1);

    PrintTensorDataBfloat16(hin_shape, tensors.hin_addr);
    PrintTensorDataFloat(h_post_shape, tensors.h_post_addr);
    PrintTensorDataFloat(h_res_shape, tensors.h_res_addr);
    PrintTensorDataFloat(h_pre_shape, tensors.h_pre_addr);

    if (workspace_size > 0) {
        (void)aclrtFree(workspace_addr);
    }
    FreeDeviceMemory(tensors);
    DestroyTensors(tensors);

    acl_ret = aclrtDestroyStream(stream);
    CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("destroy stream failed, error: %d\n", acl_ret); return -1);
    acl_ret = aclrtDestroyContext(context);
    CHECK_RET(acl_ret == ACL_SUCCESS, LOG_PRINT("destroy context failed, error: %d\n", acl_ret); return -1);
    (void)aclFinalize();

    return 0;
}
