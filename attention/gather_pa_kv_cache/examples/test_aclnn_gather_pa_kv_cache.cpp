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
#include "aclnnop/aclnn_gather_pa_kv_cache.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
    // 1. (固定写法)device/stream初始化，参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> keyCacheShape = {2, 2, 32, 2};
    std::vector<int64_t> valueCacheShape = {2, 2, 32, 4};
    std::vector<int64_t> blockTablesShape = {4, 6};
    std::vector<int64_t> seqLensShape = {4};
    std::vector<int64_t> keyShape = {12, 32, 2};
    std::vector<int64_t> valueShape = {12, 32, 4};
    std::vector<int64_t> seqOffsetShape = {4};

    void *keyCacheDeviceAddr = nullptr;
    void *valueCacheDeviceAddr = nullptr;
    void *blockTablesDeviceAddr = nullptr;
    void *seqLensDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *seqOffsetAddr = nullptr;

    aclTensor *keyCache = nullptr;
    aclTensor *valueCache = nullptr;
    aclTensor *blockTables = nullptr;
    aclTensor *seqLens = nullptr;
    aclTensor *key = nullptr;
    aclTensor *value = nullptr;
    aclTensor *seqOffset = nullptr;

    std::vector<uint16_t> keyCacheHostData(256, 1);
    std::vector<uint16_t> valueCacheHostData(512, 1);
    std::vector<int32_t> blockTablesHostData(24, 1);
    std::vector<int32_t> seqLensHostData(4, 3);
    std::vector<uint16_t> keyHostData(768, 0);
    std::vector<uint16_t> valueHostData(1536, 0);
    std::vector<int32_t> seqOffsetHostData(4, 2);

    char cacheMode[] = "Norm";
    const bool isSeqLensCumsum = false;
    // 创建GatherPaKvCache的输入输出aclTensor
    ret = CreateAclTensor(keyCacheHostData, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT16, &keyCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(valueCacheHostData, valueCacheShape, &valueCacheDeviceAddr, aclDataType::ACL_FLOAT16,
                          &valueCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(blockTablesHostData, blockTablesShape, &blockTablesDeviceAddr, aclDataType::ACL_INT32,
                          &blockTables);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(seqLensHostData, seqLensShape, &seqLensDeviceAddr, aclDataType::ACL_INT32, &seqLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(seqOffsetHostData, seqOffsetShape, &seqOffsetAddr, aclDataType::ACL_INT32, &seqOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnGatherPaKvCache第一段接口
    ret = aclnnGatherPaKvCacheGetWorkspaceSize(keyCache, valueCache, blockTables, seqLens, key, value, seqOffset,
                                               cacheMode, isSeqLensCumsum, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCacheGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGatherPaKvCache第二段接口
    ret = aclnnGatherPaKvCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCache failed. ERROR: %d\n", ret); return ret);

    // 4. (固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(keyShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclIntArray，需要根据具体API的接口定义修改
    aclDestroyTensor(keyCache);
    aclDestroyTensor(valueCache);
    aclDestroyTensor(blockTables);
    aclDestroyTensor(seqLens);
    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(seqOffset);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(keyCacheDeviceAddr);
    aclrtFree(valueCacheDeviceAddr);
    aclrtFree(blockTablesDeviceAddr);
    aclrtFree(seqLensDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(seqOffsetAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
