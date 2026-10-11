/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_grouped_matmul_inplace_add.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_grouped_matmul_inplace_add.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            Finalize(deviceId, stream); \
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
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnQuantGroupedMatmulInplaceAddTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> x1Shape = {2, 2};
    std::vector<int64_t> x2Shape = {2, 2};
    std::vector<int64_t> scale2Shape = {2, 2};
    std::vector<int64_t> yShape = {2, 2, 2};
    std::vector<int64_t> scale1Shape = {2, 1};
    std::vector<int64_t> groupListShape = {2};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* scale2 = nullptr;
    aclTensor* yRef = nullptr;
    aclTensor* scale1 = nullptr;
    aclTensor* out = nullptr;
    int64_t groupListType = 0;
    int64_t groupSize = 0;
    std::vector<uint8_t> xData(GetShapeSize(x1Shape), 0X10); // hifloat8 2.0 转16进制 0X10
    std::vector<int64_t> groupListData = {1, 2};
    std::vector<float> scale2Data(GetShapeSize(scale2Shape), 1);
    std::vector<float> yData(GetShapeSize(yShape), 1);
    std::vector<float> scale1Data(GetShapeSize(scale1Shape), 1);

    // 创建x aclTensor
    ret = CreateAclTensor<uint8_t>(xData, x1Shape, &x1DeviceAddr, aclDataType::ACL_HIFLOAT8, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor<uint8_t>(xData, x2Shape, &x2DeviceAddr, aclDataType::ACL_HIFLOAT8, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale2 aclTensor
    ret = CreateAclTensor<float>(scale2Data, scale2Shape, &scale2DeviceAddr, aclDataType::ACL_FLOAT, &scale2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale2TensorPtr(scale2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scale2DeviceAddrPtr(scale2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensor
    ret = CreateAclTensor<float>(yData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &yRef);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(yRef, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建group_list aclTensor
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale1 aclTensor
    ret = CreateAclTensor<float>(scale1Data, scale1Shape, &scale1DeviceAddr, aclDataType::ACL_FLOAT, &scale1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale1TensorPtr(scale1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scale1DeviceAddrPtr(scale1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API
    // 调用aclnnQuantGroupedMatmulInplaceAdd第一段接口
    ret = aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(x1, x2, scale1, scale2, groupList, yRef, groupListType,
                                                            groupSize, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnQuantGroupedMatmulInplaceAdd第二段接口
    ret = aclnnQuantGroupedMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(uint32_t), yDeviceAddr, size * sizeof(uint32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnQuantGroupedMatmulInplaceAddTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAddTest failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
