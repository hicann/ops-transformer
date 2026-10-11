/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing.h"
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
    // 1. （固定写法）device/stream初始化, 参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    int64_t outSize = 12;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> expandedXShape = {3 * 2, 4};
    std::vector<int64_t> x1Shape = {3, 4};
    std::vector<int64_t> x2OptionalShape = {3, 4};
    std::vector<int64_t> biasShape = {2, 4};
    std::vector<int64_t> scalesShape = {3, 2};
    std::vector<int64_t> expandedExpertIdxShape = {3, 2};
    std::vector<int64_t> expandedRowIdxShape = {3 * 2};
    std::vector<int64_t> outShape = {3, 4};
    void* expandedXAddr = nullptr;
    void* x1Addr = nullptr;
    void* x2OptionalAddr = nullptr;
    void* biasAddr = nullptr;
    void* scalesDeviceAddr = nullptr;
    void* expandedExpertIdxAddr = nullptr;
    void* expandedRowIdxAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* expandedX = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2Optional = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* scales = nullptr;
    aclTensor* expandedExpertIdx = nullptr;
    aclTensor* expandedRowIdx = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> expandedXHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1,
                                            0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
    std::vector<float> x1HostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
    std::vector<float> x2OptionalHostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
    std::vector<float> biasHostData = {0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4};
    std::vector<float> scalesHostData = {1.3, 1.6, 1.2, 1.8, 1.2, 2.3};
    std::vector<int32_t> expandedExpertIdxHostData = {0, 1, 0, 1, 0, 1};
    std::vector<int32_t> expandedRowIdxHostData = {2, 1, 4, 3, 0, 5};
    std::vector<float> outHostData(outSize, 0.0f);
    // 创建expandedX aclTensor
    ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXAddr, aclDataType::ACL_FLOAT, &expandedX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1Addr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2Optional aclScalar
    ret = CreateAclTensor(x2OptionalHostData, x2OptionalShape, &x2OptionalAddr, aclDataType::ACL_FLOAT, &x2Optional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建expandedExpertIdx aclTensor
    ret = CreateAclTensor(expandedExpertIdxHostData, expandedExpertIdxShape, &expandedExpertIdxAddr,
                          aclDataType::ACL_INT32, &expandedExpertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建expandedRowIdx aclTensor
    ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxAddr, aclDataType::ACL_INT32,
                          &expandedRowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建Out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的算子接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnMoeFinalizeRouting第一段接口
    ret = aclnnMoeFinalizeRoutingGetWorkspaceSize(expandedX, x1, x2Optional, bias, scales, expandedRowIdx,
                                                  expandedExpertIdx, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeFinalizeRouting第二段接口
    ret = aclnnMoeFinalizeRouting(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRouting failed. ERROR: %d\n", ret); return ret);

    // 4.（ 固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(expandedX);
    aclDestroyTensor(x1);
    aclDestroyTensor(x2Optional);
    aclDestroyTensor(bias);
    aclDestroyTensor(scales);
    aclDestroyTensor(expandedExpertIdx);
    aclDestroyTensor(expandedRowIdx);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(expandedXAddr);
    aclrtFree(x1Addr);
    aclrtFree(x2OptionalAddr);
    aclrtFree(biasAddr);
    aclrtFree(scalesDeviceAddr);
    aclrtFree(expandedExpertIdxAddr);
    aclrtFree(expandedRowIdxAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
