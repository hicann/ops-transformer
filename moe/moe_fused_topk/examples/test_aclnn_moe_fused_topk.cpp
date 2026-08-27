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
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_fused_topk.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，初始化
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnMoeFusedTopkTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造

    int64_t num_token = 16;
    int64_t expert_num = 32;
    int64_t max_mapping_num = 16;

    uint32_t groupNum = 2;
    uint32_t groupTopk = 2;
    uint32_t topN = 2;
    uint32_t topK = 4;
    uint32_t activateType = 0;
    bool isNorm = false;
    float scale = 1.0;
    bool enableExpertMapping = true;

    std::vector<int64_t> xShape = {num_token, expert_num};
    std::vector<int64_t> addNumShape = {expert_num};
    std::vector<int64_t> mappingNumShape = {expert_num};
    std::vector<int64_t> mappingTableShape = {expert_num, max_mapping_num};

    std::vector<int64_t> yShape = {num_token, topK};
    std::vector<int64_t> indicesShape = {num_token, topK};

    void *xDeviceAddr = nullptr;
    void *addNumDeviceAddr = nullptr;
    void *mappingNumDeviceAddr = nullptr;
    void *mappingTableDeviceAddr = nullptr;

    void *yDeviceAddr = nullptr;
    void *indicesDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *addNum = nullptr;
    aclTensor *mappingNum = nullptr;
    aclTensor *mappingTable = nullptr;

    aclTensor *y = nullptr;
    aclTensor *indices = nullptr;

    std::vector<float> xHostData(GetShapeSize(xShape), 1);
    std::vector<float> addNumHostData(GetShapeSize(addNumShape), 1);
    std::vector<int32_t> mappingNumHostData(GetShapeSize(mappingNumShape), 1);
    std::vector<int32_t> mappingTableHostData(GetShapeSize(mappingTableShape), 1);

    std::vector<float> yHostData(GetShapeSize(yShape), 0);
    std::vector<int32_t> indicesHostData(GetShapeSize(indicesShape), 0);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);

    // 创建addNum aclTensor
    ret = CreateAclTensor(addNumHostData, addNumShape, &addNumDeviceAddr, aclDataType::ACL_FLOAT, &addNum);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> addNumTensorPtr(addNum, aclDestroyTensor);

    // 创建mappingNum aclTensor
    ret = CreateAclTensor(mappingNumHostData, mappingNumShape, &mappingNumDeviceAddr, aclDataType::ACL_INT32,
                          &mappingNum);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> mappingNumTensorPtr(mappingNum, aclDestroyTensor);

    // 创建mappingTable aclTensor
    ret = CreateAclTensor(mappingTableHostData, mappingTableShape, &mappingTableDeviceAddr, aclDataType::ACL_INT32,
                          &mappingTable);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> mappingTableTensorPtr(mappingTable,
                                                                                         aclDestroyTensor);

    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);

    // 创建indices aclTensor
    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> indicesTensorPtr(indices, aclDestroyTensor);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnMoeFusedTopk第一段接口
    ret = aclnnMoeFusedTopkGetWorkspaceSize(x, addNum, mappingNum, mappingTable, groupNum, groupTopk, topN, topK,
                                            activateType, isNorm, scale, enableExpertMapping, y, indices,
                                            &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopkGetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnMoeFusedTopk第二段接口
    ret = aclnnMoeFusedTopk(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopk failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> yData(size, 0);
    ret = aclrtMemcpy(yData.data(), yData.size() * sizeof(yData[0]), yDeviceAddr, size * sizeof(yData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out result[%ld] is: %f\n", i, yData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnMoeFusedTopkTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopkTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
