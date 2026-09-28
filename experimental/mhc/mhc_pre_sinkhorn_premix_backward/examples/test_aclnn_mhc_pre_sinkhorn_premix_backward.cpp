/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_sinkhorn_premix_backward.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

void PrintTensorDataFloat(const std::vector<int64_t> &shape, void *device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    aclError ret = aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
}

void PrintTensorDataBfloat16(const std::vector<int64_t> &shape, void *device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_bf16(size);

    aclError ret = aclrtMemcpy(host_bf16.data(), size * sizeof(uint16_t), device_addr, size * sizeof(uint16_t),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    LOG_PRINT("Tensor data (first 10 elements, BF16 hex): ");
    for (int64_t i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("0x%04x ", host_bf16[i]);
    }
    LOG_PRINT("\n");
}

int InitAcl(int32_t device_id, aclrtContext &context, aclrtStream &stream)
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

int CreateAclTensorBfloat16(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                            aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_data_bf16(size);
    for (int64_t i = 0; i < size; ++i) {
        host_data_bf16[i] = static_cast<uint16_t>(static_cast<int16_t>(host_data[i] * 32767));
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

int CreateAclTensorFloat32(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                           aclTensor *&tensor)
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

int CreateAclTensorBfloat16Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
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

int CreateAclTensorFloat32Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
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
    void *grad_hin_addr = nullptr, *grad_h_post_addr = nullptr, *grad_h_res_addr = nullptr;
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr;
    void *h_pre_addr = nullptr, *hc_before_norm_addr = nullptr, *inv_rms_addr = nullptr;
    void *sum_out_addr = nullptr, *norm_out_addr = nullptr;
    void *grad_x_addr = nullptr, *grad_phi_addr = nullptr, *grad_alpha_addr = nullptr, *grad_bias_addr = nullptr;
    aclTensor *grad_hin = nullptr, *grad_h_post = nullptr, *grad_h_res = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
    aclTensor *h_pre = nullptr, *hc_before_norm = nullptr, *inv_rms = nullptr;
    aclTensor *sum_out = nullptr, *norm_out = nullptr;
    aclTensor *grad_x = nullptr, *grad_phi = nullptr, *grad_alpha = nullptr, *grad_bias = nullptr;
};

int CreateInputTensors(const std::vector<int64_t> &grad_hin_shape, const std::vector<int64_t> &grad_h_post_shape,
                       const std::vector<int64_t> &grad_h_res_shape, const std::vector<int64_t> &x_shape,
                       const std::vector<int64_t> &phi_shape, const std::vector<int64_t> &alpha_shape,
                       const std::vector<int64_t> &bias_shape, const std::vector<int64_t> &h_pre_shape,
                       const std::vector<int64_t> &hc_before_norm_shape, const std::vector<int64_t> &inv_rms_shape,
                       const std::vector<int64_t> &sum_out_shape, const std::vector<int64_t> &norm_out_shape,
                       Tensors &tensors)
{
    std::vector<float> grad_hin_host_data(GetShapeSize(grad_hin_shape), 1.0f);
    std::vector<float> grad_h_post_host_data(GetShapeSize(grad_h_post_shape), 1.0f);
    std::vector<float> grad_h_res_host_data(GetShapeSize(grad_h_res_shape), 1.0f);
    std::vector<float> x_host_data(GetShapeSize(x_shape), 1.0f);
    std::vector<float> phi_host_data(GetShapeSize(phi_shape), 1.0f);
    std::vector<float> alpha_host_data(GetShapeSize(alpha_shape), 1.0f);
    std::vector<float> bias_host_data(GetShapeSize(bias_shape), 1.0f);
    std::vector<float> h_pre_host_data(GetShapeSize(h_pre_shape), 1.0f);
    std::vector<float> hc_before_norm_host_data(GetShapeSize(hc_before_norm_shape), 1.0f);
    std::vector<float> inv_rms_host_data(GetShapeSize(inv_rms_shape), 1.0f);
    std::vector<float> sum_out_host_data(GetShapeSize(sum_out_shape), 1.0f);
    std::vector<float> norm_out_host_data(GetShapeSize(norm_out_shape), 1.0f);

    int ret = CreateAclTensorBfloat16(grad_hin_host_data, grad_hin_shape, tensors.grad_hin_addr, tensors.grad_hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_hin_tensor failed\n"); return -1);
    ret =
        CreateAclTensorFloat32(grad_h_post_host_data, grad_h_post_shape, tensors.grad_h_post_addr, tensors.grad_h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(grad_h_res_host_data, grad_h_res_shape, tensors.grad_h_res_addr, tensors.grad_h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorBfloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(h_pre_host_data, h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(hc_before_norm_host_data, hc_before_norm_shape, tensors.hc_before_norm_addr,
                                 tensors.hc_before_norm);
    CHECK_RET(ret == 0, LOG_PRINT("Create hc_before_norm_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(inv_rms_host_data, inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(sum_out_host_data, sum_out_shape, tensors.sum_out_addr, tensors.sum_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(norm_out_host_data, norm_out_shape, tensors.norm_out_addr, tensors.norm_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_out_tensor failed\n"); return -1);

    return 0;
}

int CreateOutputTensors(const std::vector<int64_t> &grad_x_shape, const std::vector<int64_t> &grad_phi_shape,
                        const std::vector<int64_t> &grad_alpha_shape, const std::vector<int64_t> &grad_bias_shape,
                        Tensors &tensors)
{
    int ret = CreateAclTensorBfloat16Output(grad_x_shape, tensors.grad_x_addr, tensors.grad_x);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_phi_shape, tensors.grad_phi_addr, tensors.grad_phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_alpha_shape, tensors.grad_alpha_addr, tensors.grad_alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_bias_shape, tensors.grad_bias_addr, tensors.grad_bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_bias_tensor failed\n"); return -1);

    return 0;
}

void DestroyTensors(Tensors &tensors)
{
    aclDestroyTensor(tensors.grad_hin);
    aclDestroyTensor(tensors.grad_h_post);
    aclDestroyTensor(tensors.grad_h_res);
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.h_pre);
    aclDestroyTensor(tensors.hc_before_norm);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.sum_out);
    aclDestroyTensor(tensors.norm_out);
    aclDestroyTensor(tensors.grad_x);
    aclDestroyTensor(tensors.grad_phi);
    aclDestroyTensor(tensors.grad_alpha);
    aclDestroyTensor(tensors.grad_bias);
}

void FreeDeviceMemory(Tensors &tensors)
{
    aclrtFree(tensors.grad_hin_addr);
    aclrtFree(tensors.grad_h_post_addr);
    aclrtFree(tensors.grad_h_res_addr);
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.h_pre_addr);
    aclrtFree(tensors.hc_before_norm_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.sum_out_addr);
    aclrtFree(tensors.norm_out_addr);
    aclrtFree(tensors.grad_x_addr);
    aclrtFree(tensors.grad_phi_addr);
    aclrtFree(tensors.grad_alpha_addr);
    aclrtFree(tensors.grad_bias_addr);
}

int main()
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    Tensors tensors;

    int64_t bs = 2, seq_len = 128, n = 4, c = 256;
    int64_t hc_mult = n;
    int64_t hc_mix = n * n + 2 * n;
    int64_t num_iters = 20;
    double hc_eps = 1e-6;

    std::vector<int64_t> grad_hin_shape = {bs, seq_len, c};
    std::vector<int64_t> grad_h_post_shape = {bs, seq_len, n};
    std::vector<int64_t> grad_h_res_shape = {bs, seq_len, n, n};
    std::vector<int64_t> x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> phi_shape = {hc_mix, n * c};
    std::vector<int64_t> alpha_shape = {3};
    std::vector<int64_t> bias_shape = {hc_mix};
    std::vector<int64_t> h_pre_shape = {bs, seq_len, n};
    std::vector<int64_t> hc_before_norm_shape = {bs, seq_len, hc_mix};
    std::vector<int64_t> inv_rms_shape = {bs, seq_len, 1};
    std::vector<int64_t> sum_out_shape = {num_iters * 2, bs, seq_len, n};
    std::vector<int64_t> norm_out_shape = {num_iters * 2, bs, seq_len, n, n};

    std::vector<int64_t> grad_x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> grad_phi_shape = {hc_mix, n * c};
    std::vector<int64_t> grad_alpha_shape = {3};
    std::vector<int64_t> grad_bias_shape = {hc_mix};

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);

    ret = CreateInputTensors(grad_hin_shape, grad_h_post_shape, grad_h_res_shape, x_shape, phi_shape, alpha_shape,
                             bias_shape, h_pre_shape, hc_before_norm_shape, inv_rms_shape, sum_out_shape,
                             norm_out_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    ret = CreateOutputTensors(grad_x_shape, grad_phi_shape, grad_alpha_shape, grad_bias_shape, tensors);
    CHECK_RET(ret == 0, return -1);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclnn_ret = aclnnMhcPreSinkhornPremixBackwardGetWorkspaceSize(
        tensors.grad_hin, tensors.grad_h_post, tensors.grad_h_res, tensors.x, tensors.phi, tensors.alpha, tensors.bias,
        tensors.h_pre, tensors.hc_before_norm, tensors.inv_rms, tensors.sum_out, tensors.norm_out, nullptr,
        nullptr, // pre_mix / grad_pre：v1 场景（无 pre_mix）传 nullptr
        hc_eps, tensors.grad_x, tensors.grad_phi, tensors.grad_alpha, tensors.grad_bias,
        nullptr, // grad_pre_mix：无 pre_mix 时不输出
        &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcPreSinkhornPremixBackwardGetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);

    void *workspace_addr = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }

    aclnn_ret = aclnnMhcPreSinkhornPremixBackward(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhornPremixBackward failed, error: %d\n", aclnn_ret);
              return -1);

    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed\n"); return -1);

    LOG_PRINT("MhcPreSinkhornPremixBackward compute success!\n");
    LOG_PRINT("Output tensor grad_x (first 10 elements):\n");
    PrintTensorDataBfloat16(grad_x_shape, tensors.grad_x_addr);

    DestroyTensors(tensors);
    FreeDeviceMemory(tensors);
    if (workspace_size > 0)
        aclrtFree(workspace_addr);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();
    LOG_PRINT("All resources released successfully!\n");
    return 0;
}
