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
 * \file test_aclnn_mhc_sinkhorn.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_sinkhorn.h"

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

// 将Device侧Tensor数据拷贝到Host侧并打印
void PrintTensorData(const std::vector<int64_t>& shape, void* device_addr)
{
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    // Device -> Host 数据拷贝
    aclError ret = aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); return);

    // 打印前10个元素（示例）
    LOG_PRINT("Tensor data (first 10 elements): ");
    for (int i = 0; i < std::min((int64_t)10, size); ++i) {
        LOG_PRINT("%f ", host_data[i]);
    }
    LOG_PRINT("\n");
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

int main()
{
    // ========== 1. 初始化环境 ==========
    int32_t device_id = 0; // 根据实际Device ID调整
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed, error: %d\n", ret); return -1);

    // ========== 2. 构造输入/输出参数 ==========
    // 输入x的形状：B=1, S=1024, n=4 → (1024,4,4)（合并B*S为T=1024）
    std::vector<int64_t> x_shape = {1024, 4, 4};
    int64_t x_size = GetShapeSize(x_shape);
    std::vector<float> x_host_data(x_size, 1.0f); // 初始化输入数据为1.0

    // 输出output的形状与x一致
    std::vector<int64_t> output_shape = x_shape;
    void* output_device_addr = nullptr;
    aclTensor* output_tensor = nullptr;

    // 可选输出：norm_out（out_flag=1时有效）
    std::vector<int64_t> norm_out_shape = {40 * 1024 * 4 * 8}; // 2*20=40, T=1024, n=4, n_align=8
    void* norm_out_device_addr = nullptr;
    aclTensor* norm_out_tensor = nullptr;

    // 可选输出：sum_out（out_flag=1时有效）
    std::vector<int64_t> sum_out_shape = {40 * 1024 * 8}; // 2*20=40, n_align=8, T=1024
    void* sum_out_device_addr = nullptr;
    aclTensor* sum_out_tensor = nullptr;

    // 输入x的Device Tensor
    void* x_device_addr = nullptr;
    aclTensor* x_tensor = nullptr;
    ret = CreateAclTensor(x_host_data, x_shape, x_device_addr, x_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);

    // 输出output的Device Tensor（仅申请内存，无初始数据）
    ret = aclrtMalloc(&output_device_addr, GetShapeSize(output_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Malloc output failed, error: %d\n", ret); return -1);
    output_tensor = aclCreateTensor(output_shape.data(), output_shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                                    output_shape.data(), output_shape.size(), output_device_addr);

    // 可选输出norm_out/sum_out的Tensor（out_flag=1）
    ret = aclrtMalloc(&norm_out_device_addr, GetShapeSize(norm_out_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Malloc norm_out failed, error: %d\n", ret); return -1);
    norm_out_tensor =
        aclCreateTensor(norm_out_shape.data(), norm_out_shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                        norm_out_shape.data(), norm_out_shape.size(), norm_out_device_addr);

    ret = aclrtMalloc(&sum_out_device_addr, GetShapeSize(sum_out_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Malloc sum_out failed, error: %d\n", ret); return -1);
    sum_out_tensor = aclCreateTensor(sum_out_shape.data(), sum_out_shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                                     sum_out_shape.data(), sum_out_shape.size(), sum_out_device_addr);

    // MhcSinkhorn算子参数
    float eps = 1e-6f;      // 防除零参数
    int64_t num_iters = 20; // 迭代次数

    // ========== 3. 调用第一段接口：获取Workspace大小 ==========
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclnn_ret = aclnnMhcSinkhornGetWorkspaceSize(x_tensor, eps, num_iters, output_tensor, norm_out_tensor,
                                                             sum_out_tensor, &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcSinkhornGetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);

    // ========== 4. 申请Workspace内存 ==========
    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }

    // ========== 5. 调用第二段接口：执行MhcSinkhorn计算 ==========
    aclnn_ret = aclnnMhcSinkhorn(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcSinkhorn failed, error: %d\n", aclnn_ret); return -1);

    // ========== 6. 同步Stream并打印结果 ==========
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret); return -1);

    LOG_PRINT("MhcSinkhorn compute success!\n");
    LOG_PRINT("Output tensor data: ");
    PrintTensorData(output_shape, output_device_addr);

    // ========== 7. 释放资源 ==========
    // 销毁Tensor
    aclDestroyTensor(x_tensor);
    aclDestroyTensor(output_tensor);
    aclDestroyTensor(norm_out_tensor);
    aclDestroyTensor(sum_out_tensor);

    // 释放Device内存
    aclrtFree(x_device_addr);
    aclrtFree(output_device_addr);
    aclrtFree(norm_out_device_addr);
    aclrtFree(sum_out_device_addr);
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
