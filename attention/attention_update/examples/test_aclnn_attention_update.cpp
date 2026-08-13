/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_attention_update.cpp
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_attention_update.h"
#include <iostream>
#include <vector>

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的stride
    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    constexpr int64_t lse_dim = 32;
    constexpr int64_t local_out_batch = 32;
    constexpr int64_t local_out_feat = 16;
    constexpr int64_t update_type = 0;
    constexpr int32_t deviceId = 0;

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> lseShape = {lse_dim};
    std::vector<int64_t> localOutShape = {local_out_batch, local_out_feat};
    std::vector<int64_t> outShape = {local_out_batch, local_out_feat};

    void *lseDeviceAddr[2] = {nullptr, nullptr};
    void *localOutDeviceAddr[2] = {nullptr, nullptr};
    void *outDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;

    aclTensor *lse[2] = {nullptr, nullptr};
    aclTensor *localOut[2] = {nullptr, nullptr};
    aclTensor *out = nullptr;
    aclTensorList *lseList = nullptr;
    aclTensorList *localOutList = nullptr;

    std::vector<float> lse1HostData(GetShapeSize(lseShape), 1.0f);
    std::vector<float> lse2HostData(GetShapeSize(lseShape), 1.0f);
    std::vector<float> localOut1HostData(GetShapeSize(localOutShape), 1.0f);
    std::vector<float> localOut2HostData(GetShapeSize(localOutShape), 1.0f);
    std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

    ret = CreateAclTensor(lse1HostData, lseShape, &lseDeviceAddr[0], aclDataType::ACL_FLOAT, &lse[0]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create lse1 tensor failed. ERROR: %d\n", ret); return ret);
    ret = CreateAclTensor(lse2HostData, lseShape, &lseDeviceAddr[1], aclDataType::ACL_FLOAT, &lse[1]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create lse2 tensor failed. ERROR: %d\n", ret); return ret);
    lseList = aclCreateTensorList(lse, 2);
    CHECK_RET(lseList != nullptr, LOG_PRINT("Create lse TensorList failed\n"); return -1);

    ret = CreateAclTensor(localOut1HostData, localOutShape, &localOutDeviceAddr[0], aclDataType::ACL_FLOAT, &localOut[0]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create localOut1 tensor failed. ERROR: %d\n", ret); return ret);
    ret = CreateAclTensor(localOut2HostData, localOutShape, &localOutDeviceAddr[1], aclDataType::ACL_FLOAT, &localOut[1]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create localOut2 tensor failed. ERROR: %d\n", ret); return ret);
    localOutList = aclCreateTensorList(localOut, 2);
    CHECK_RET(localOutList != nullptr, LOG_PRINT("Create localOut TensorList failed\n"); return -1);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create out tensor failed. ERROR: %d\n", ret); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnAttentionUpdateGetWorkspaceSize(lseList, localOutList, update_type, out, nullptr,
                                               &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAttentionUpdate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdate failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto out_size = GetShapeSize(outShape);
    std::vector<float> outData(out_size, 0.0f);
    ret = aclrtMemcpy(outData.data(), out_size * sizeof(float), outDeviceAddr,
                      out_size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    constexpr int64_t printLen = 16;
    for (int64_t i = 0; i < printLen && i < out_size; i++) {
        LOG_PRINT("out[%ld]: %f\n", i, outData[i]);
    }

    if (lseList != nullptr) {
        aclDestroyTensorList(lseList);
    }
    if (localOutList != nullptr) {
        aclDestroyTensorList(localOutList);
    }
    aclDestroyTensor(out);

    aclrtFree(lseDeviceAddr[0]);
    aclrtFree(lseDeviceAddr[1]);
    aclrtFree(localOutDeviceAddr[0]);
    aclrtFree(localOutDeviceAddr[1]);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
