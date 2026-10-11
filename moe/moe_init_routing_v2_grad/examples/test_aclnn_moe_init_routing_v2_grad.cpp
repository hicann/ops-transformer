/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "aclnnop/aclnn_moe_init_routing_v2_grad.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> gradExpandedXShape = {4, 2};
    std::vector<int64_t> expandedRowIdxShape = {4};
    std::vector<int64_t> gradXShape = {2, 2};
    void* gradExpandedXDeviceAddr = nullptr;
    void* expandedRowIdxDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    aclTensor* gradExpandedX = nullptr;
    aclTensor* expandedRowIdx = nullptr;
    aclScalar* k = nullptr;
    aclScalar* dropPadMode = nullptr;
    aclScalar* activeNum = nullptr;
    aclTensor* expertIdx = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> gradExpandedXHostData = {0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4};
    std::vector<int32_t> expandedRowIdxHostData = {2, 0, 1, 3};
    std::vector<float> gradXOutHostData = {0, 0, 0, 0};
    int32_t kValue = 2;
    int32_t dropPadModeValue = 0;
    int32_t activeNumValue = 0;

    // 创建输入 aclTensor
    ret = CreateAclTensor(gradExpandedXHostData, gradExpandedXShape, &gradExpandedXDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradExpandedX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxDeviceAddr,
                          aclDataType::ACL_INT32, &expandedRowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    k = aclCreateScalar(&kValue, aclDataType::ACL_INT32);
    CHECK_RET(k != nullptr, return ret);
    dropPadMode = aclCreateScalar(&dropPadModeValue, aclDataType::ACL_INT32);
    CHECK_RET(dropPadMode != nullptr, return ret);
    activeNum = aclCreateScalar(&activeNumValue, aclDataType::ACL_INT32);
    CHECK_RET(activeNum != nullptr, return ret);
    // 创建输出 aclTensor
    ret = CreateAclTensor(gradXOutHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMoeInitRoutingV2Grad第一段接口
    ret = aclnnMoeInitRoutingV2GradGetWorkspaceSize(gradExpandedX, expandedRowIdx, kValue, dropPadModeValue,
                                                    activeNumValue, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2GradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnMoeInitRoutingV2Grad第二段接口
    ret = aclnnMoeInitRoutingV2Grad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2Grad failed. ERROR: %d\n", ret); return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto gradXSize = GetShapeSize(gradXShape);
    std::vector<float> gradXData(gradXSize, 0);
    ret = aclrtMemcpy(gradXData.data(), gradXData.size() * sizeof(gradXData[0]), gradXDeviceAddr,
                      gradXSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < gradXSize; i++) {
        LOG_PRINT("gradXData[%ld] is: %f\n", i, gradXData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradExpandedX);
    aclDestroyTensor(expandedRowIdx);
    aclDestroyScalar(k);
    aclDestroyScalar(dropPadMode);
    aclDestroyScalar(activeNum);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradExpandedXDeviceAddr);
    aclrtFree(expandedRowIdxDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
