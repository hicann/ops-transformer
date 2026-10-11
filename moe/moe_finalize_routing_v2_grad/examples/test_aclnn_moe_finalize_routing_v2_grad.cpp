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
#include "aclnnop/aclnn_moe_finalize_routing_v2_grad.h"
#include <iostream>

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
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
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradYShape = {2, 2};
    std::vector<int64_t> expandedRowIdxShape = {4};
    std::vector<int64_t> expandedXShape = {4, 2};
    std::vector<int64_t> scalesShape = {2, 2};
    std::vector<int64_t> expertIdxShape = {2, 2};
    std::vector<int64_t> biasShape = {2, 2};
    std::vector<int64_t> gradExpandedXShape = {4, 2};
    std::vector<int64_t> gradScalesShape = {2, 2};
    void* gradYDeviceAddr = nullptr;
    void* expandedRowIdxDeviceAddr = nullptr;
    void* expandedXDeviceAddr = nullptr;
    void* scalesDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* gradExpandedXDeviceAddr = nullptr;
    void* gradScalesDeviceAddr = nullptr;

    aclTensor* gradY = nullptr;
    aclTensor* expandedRowIdx = nullptr;
    aclTensor* expandedX = nullptr;
    aclTensor* scales = nullptr;
    aclTensor* expertIdx = nullptr;
    aclTensor* bias = nullptr;
    int64_t dropPadMode = 0;
    int64_t activeNum = 0;
    int64_t expertNum = 0;
    int64_t expertCapacity = 0;
    aclTensor* gradExpandedX = nullptr;
    aclTensor* gradScales = nullptr;

    std::vector<float> gradYHostData = {0.3816, 0.3939, 0.8474, 0.1652};
    std::vector<int> expandedRowIdxHostData = {1, 3, 0, 2};
    std::vector<float> expandedXHostData = {0.6049, 0.3315, 0.4954, 0.3284, 0.7060, 0.4359, 0.6514, 0.9476};
    std::vector<float> scalesHostData = {0.4708, 0.0656, 0.9652, 0.9512};
    std::vector<int> expertIdxHostData = {0, 1, 0, 1};
    std::vector<float> biasHostData = {0.6452, 0.1981, 0.4159, 0.9575};
    std::vector<float> gradExpandedXHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gradScalesHostData = {0, 0, 0, 0};

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxDeviceAddr,
                          aclDataType::ACL_INT32, &expandedRowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXDeviceAddr, aclDataType::ACL_FLOAT, &expandedX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradExpandedXHostData, gradExpandedXShape, &gradExpandedXDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradExpandedX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradScalesHostData, gradScalesShape, &gradScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnMoeFinalizeRoutingV2Grad第一段接口
    ret = aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(gradY, expandedRowIdx, expandedX, scales, expertIdx, bias,
                                                        dropPadMode, activeNum, expertNum, expertCapacity,
                                                        gradExpandedX, gradScales, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnMoeFinalizeRoutingV2Grad第二段接口
    ret = aclnnMoeFinalizeRoutingV2Grad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2Grad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    LOG_PRINT("gradExpandedX result is: \n");
    PrintOutResult(gradExpandedXShape, &gradExpandedXDeviceAddr);
    LOG_PRINT("gradScales result is: \n");
    PrintOutResult(gradScalesShape, &gradScalesDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradY);
    aclDestroyTensor(expandedRowIdx);
    aclDestroyTensor(expandedX);
    aclDestroyTensor(scales);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(bias);
    aclDestroyTensor(gradExpandedX);
    aclDestroyTensor(gradScales);

    // 7. 释放device资源
    aclrtFree(gradYDeviceAddr);
    aclrtFree(expandedRowIdxDeviceAddr);
    aclrtFree(expandedXDeviceAddr);
    aclrtFree(scalesDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(gradExpandedXDeviceAddr);
    aclrtFree(gradScalesDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
