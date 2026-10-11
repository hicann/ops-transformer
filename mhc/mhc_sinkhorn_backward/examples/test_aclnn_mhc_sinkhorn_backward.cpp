/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_mhc_sinkhorn_backward.cpp
 * \brief mhc_sinkhorn_backward算子example测试文件
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_sinkhorn_backward.h"

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

// 计算Tensor形状对应的总元素数
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

// 向上对齐到align的倍数
int64_t CeilAlign(int64_t value, int64_t align)
{
    return (value + align - 1) / align * align;
}

// 初始化AscendCL环境（Device/Context/Stream）
int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream)
{
    // 1. 初始化ACL
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed, error: %d\n", ret); return -1);

    // 2. 设置Device
    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); return -1);

    // 3. 创建Context
    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); return -1);

    // 4. 设置当前Context
    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); return -1);

    // 5. 创建Stream
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); return -1);

    return 0;
}

// 创建Device侧aclTensor（含数据拷贝）
int CreateAclTensor(const std::vector<float>& host_data, const std::vector<int64_t>& shape, void*& device_addr,
                    aclTensor*& tensor)
{
    // 1. 计算内存大小
    int64_t size = GetShapeSize(shape) * sizeof(float);

    // 2. 申请Device侧内存
    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    // 3. Host -> Device 数据拷贝
    ret = aclrtMemcpy(device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    // 4. 计算Tensor的strides（连续Tensor）
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // 5. 创建aclTensor
    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

// 创建输出Tensor（仅申请内存，无初始数据）
int CreateOutputTensor(const std::vector<int64_t>& shape, void*& device_addr, aclTensor*& tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc output failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor output failed\n"); return -1);

    return 0;
}

int main()
{
    // ========== 1. 初始化环境 ==========
    int32_t device_id = 0; // 根据实际Device ID调整
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);

    LOG_PRINT("ACL environment initialized successfully!\n");

    // ========== 2. 定义算子参数 ==========
    // grad_y形状可以是(T, N, N)或(B, S, N, N)
    // 这里使用T形状作为示例
    int64_t T = 128;        // total length
    int64_t N = 4;          // n size (N x N矩阵)
    int64_t num_iters = 20; // Sinkhorn迭代次数

    // 计算对齐后的N大小（对齐到8）
    int64_t align_unit = 8;
    int64_t n_align_size = CeilAlign(N, align_unit);

    LOG_PRINT("Parameters: T=%ld, N=%ld, num_iters=%ld, n_align_size=%ld\n", T, N, num_iters, n_align_size);

    // ========== 3. 构造输入/输出张量 ==========
    // 输入grad_y形状：(T, N, N)
    std::vector<int64_t> grad_y_shape = {T, N, N};
    int64_t grad_y_size = GetShapeSize(grad_y_shape);
    std::vector<float> grad_y_host_data(grad_y_size, 1.0f);
    for (int64_t i = 0; i < grad_y_size; ++i) {
        grad_y_host_data[i] = 0.1f * static_cast<float>(i % 100); // 填充测试数据
    }

    // 输入norm形状：[2 * num_iters * T * N * n_align_size]
    int64_t norm_size = 2 * num_iters * T * N * n_align_size;
    std::vector<int64_t> norm_shape = {norm_size};
    std::vector<float> norm_host_data(norm_size, 0.5f);
    for (int64_t i = 0; i < norm_size; ++i) {
        norm_host_data[i] = 1.0f / (1.0f + static_cast<float>(i % 1000) * 0.001f); // 填充测试数据
    }

    // 输入sum形状：[2 * num_iters * T * n_align_size]
    int64_t sum_size = 2 * num_iters * T * n_align_size;
    std::vector<int64_t> sum_shape = {sum_size};
    std::vector<float> sum_host_data(sum_size, 1.0f);
    for (int64_t i = 0; i < sum_size; ++i) {
        sum_host_data[i] = 1.0f + static_cast<float>(i % 500) * 0.01f; // 填充测试数据
    }

    // 输出grad_input形状与grad_y相同：(T, N, N)
    std::vector<int64_t> grad_input_shape = grad_y_shape;

    // 创建Device侧Tensor
    void* grad_y_device_addr = nullptr;
    aclTensor* grad_y_tensor = nullptr;
    ret = CreateAclTensor(grad_y_host_data, grad_y_shape, grad_y_device_addr, grad_y_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_y_tensor failed\n"); return -1);

    void* norm_device_addr = nullptr;
    aclTensor* norm_tensor = nullptr;
    ret = CreateAclTensor(norm_host_data, norm_shape, norm_device_addr, norm_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_tensor failed\n"); return -1);

    void* sum_device_addr = nullptr;
    aclTensor* sum_tensor = nullptr;
    ret = CreateAclTensor(sum_host_data, sum_shape, sum_device_addr, sum_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_tensor failed\n"); return -1);

    void* grad_input_device_addr = nullptr;
    aclTensor* grad_input_tensor = nullptr;
    ret = CreateOutputTensor(grad_input_shape, grad_input_device_addr, grad_input_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_input_tensor failed\n"); return -1);

    LOG_PRINT("All tensors created successfully!\n");
    LOG_PRINT("grad_y shape: [%ld, %ld, %ld]\n", grad_y_shape[0], grad_y_shape[1], grad_y_shape[2]);
    LOG_PRINT("norm shape: [%ld]\n", norm_shape[0]);
    LOG_PRINT("sum shape: [%ld]\n", sum_shape[0]);
    LOG_PRINT("grad_input shape: [%ld, %ld, %ld]\n", grad_input_shape[0], grad_input_shape[1], grad_input_shape[2]);

    // ========== 4. 调用第一段接口：获取Workspace大小 ==========
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclnn_ret = aclnnMhcSinkhornBackwardGetWorkspaceSize(grad_y_tensor, norm_tensor, sum_tensor,
                                                                     grad_input_tensor, &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcSinkhornBackwardGetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);

    LOG_PRINT("Workspace size: %lu bytes\n", workspace_size);

    // ========== 5. 申请Workspace内存 ==========
    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }

    // ========== 6. 调用第二段接口：执行MhcSinkhornBackward计算 ==========
    aclnn_ret = aclnnMhcSinkhornBackward(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcSinkhornBackward failed, error: %d\n", aclnn_ret);
              return -1);

    // ========== 7. 同步Stream ==========
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret); return -1);

    LOG_PRINT("MhcSinkhornBackward compute success!\n");

    // ========== 8. 释放资源 ==========
    // 销毁Tensor
    aclDestroyTensor(grad_y_tensor);
    aclDestroyTensor(norm_tensor);
    aclDestroyTensor(sum_tensor);
    aclDestroyTensor(grad_input_tensor);

    // 释放Device内存
    aclrtFree(grad_y_device_addr);
    aclrtFree(norm_device_addr);
    aclrtFree(sum_device_addr);
    aclrtFree(grad_input_device_addr);
    if (workspace_size > 0) {
        aclrtFree(workspace_addr);
    }

    // 销毁Stream/Context，重置Device
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();

    LOG_PRINT("All resources released successfully!\n");
    return 0;
}
