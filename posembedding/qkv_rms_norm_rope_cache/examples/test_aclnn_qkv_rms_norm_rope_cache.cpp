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
#include "aclnnop/aclnn_qkv_rms_norm_rope_cache.h"

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
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
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
    // 1. 固定写法，device/stream初始化, 参考acl API
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> qkvShape = {48, 2304};
    std::vector<int64_t> qGammaShape = {128};
    std::vector<int64_t> kGammaShape = {128};
    std::vector<int64_t> cosShape = {48, 128};
    std::vector<int64_t> sinShape = {48, 128};
    std::vector<int64_t> indexShape = {48};
    std::vector<int64_t> qOutShape = {48, 2048};
    std::vector<int64_t> kCacheShape = {16, 4, 128, 32};
    std::vector<int64_t> vCacheShape = {16, 4, 128, 32};
    std::vector<int64_t> kScaleShape = {1, 128};
    std::vector<int64_t> vScaleShape = {1, 128};
    std::vector<int64_t> qkv_size_list = {16, 3, 18, 128};
    std::vector<int64_t> head_nums_list = {16, 1, 1};

    std::vector<int16_t> qkvHostData(48 * 2304, 0);
    std::vector<int16_t> qGammaHostData(128, 0);
    std::vector<int16_t> kGammaHostData(128, 0);
    std::vector<int16_t> cosHostData(48 * 128, 0);
    std::vector<int16_t> sinHostData(48 * 128, 0);
    // index的取值不能重复，且须落在[-1, BlockNum * BlockSize)内；此处取0..47递增
    std::vector<int64_t> indexHostData(48, 0);
    for (int64_t i = 0; i < static_cast<int64_t>(indexHostData.size()); i++) {
        indexHostData[i] = i;
    }
    std::vector<int16_t> qOutHostData(48 * 2048, 0);
    std::vector<int16_t> kCacheHostData(16 * 4 * 128 * 32, 0);
    std::vector<int16_t> vCacheHostData(16 * 4 * 128 * 32, 0);
    // kScale/vScale是对称量化的缩放因子，不能为0
    std::vector<float> kScaleHostData(1 * 128, 1.0f);
    std::vector<float> vScaleHostData(1 * 128, 1.0f);

    void* qkvDeviceAddr = nullptr;
    void* qGammaDeviceAddr = nullptr;
    void* kGammaDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* indexDeviceAddr = nullptr;
    void* qOutDeviceAddr = nullptr;
    void* kCacheDeviceAddr = nullptr;
    void* vCacheDeviceAddr = nullptr;
    void* kScaleDeviceAddr = nullptr;
    void* vScaleDeviceAddr = nullptr;

    aclTensor* qkv = nullptr;
    aclTensor* qGamma = nullptr;
    aclTensor* kGamma = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* index = nullptr;
    aclTensor* qOut = nullptr;
    aclTensor* kCache = nullptr;
    aclTensor* vCache = nullptr;
    aclTensor* kScale = nullptr;
    aclTensor* vScale = nullptr;

    aclIntArray* qkv_size = nullptr;
    aclIntArray* head_nums = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    double epsilon = 1e-6;
    char* cacheMode = "PA_NZ";

    // 失败路径统一走 cleanup:任何一步失败都要把已申请的张量、aclIntArray、设备内存与
    // 运行资源释放掉(逐条 return 的写法会把前面申请的资源泄漏在失败路径上)。
    auto destroyTensor = [](aclTensor*& tensor) {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
            tensor = nullptr;
        }
    };
    auto destroyIntArray = [](aclIntArray*& array) {
        if (array != nullptr) {
            aclDestroyIntArray(array);
            array = nullptr;
        }
    };
    auto freeDeviceAddr = [](void*& addr) {
        if (addr != nullptr) {
            aclrtFree(addr);
            addr = nullptr;
        }
    };
    auto cleanup = [&]() {
        // 6. 释放aclTensor与aclIntArray，需要根据具体API的接口定义修改
        destroyTensor(qkv);
        destroyTensor(qGamma);
        destroyTensor(kGamma);
        destroyTensor(cos);
        destroyTensor(sin);
        destroyTensor(index);
        destroyTensor(qOut);
        destroyTensor(kCache);
        destroyTensor(vCache);
        destroyTensor(kScale);
        destroyTensor(vScale);
        destroyIntArray(qkv_size);
        destroyIntArray(head_nums);

        // 7. 释放device 资源
        freeDeviceAddr(qkvDeviceAddr);
        freeDeviceAddr(qGammaDeviceAddr);
        freeDeviceAddr(kGammaDeviceAddr);
        freeDeviceAddr(cosDeviceAddr);
        freeDeviceAddr(sinDeviceAddr);
        freeDeviceAddr(indexDeviceAddr);
        freeDeviceAddr(qOutDeviceAddr);
        freeDeviceAddr(kCacheDeviceAddr);
        freeDeviceAddr(vCacheDeviceAddr);
        freeDeviceAddr(kScaleDeviceAddr);
        freeDeviceAddr(vScaleDeviceAddr);
        freeDeviceAddr(workspaceAddr);

        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    };

    qkv_size = aclCreateIntArray(qkv_size_list.data(), qkv_size_list.size());
    head_nums = aclCreateIntArray(head_nums_list.data(), head_nums_list.size());
    CHECK_RET(qkv_size != nullptr && head_nums != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); cleanup();
              return -1);

    ret = CreateAclTensor(qkvHostData, qkvShape, &qkvDeviceAddr, aclDataType::ACL_FLOAT16, &qkv);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(qGammaHostData, qGammaShape, &qGammaDeviceAddr, aclDataType::ACL_FLOAT16, &qGamma);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(kGammaHostData, kGammaShape, &kGammaDeviceAddr, aclDataType::ACL_FLOAT16, &kGamma);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(qOutHostData, qOutShape, &qOutDeviceAddr, aclDataType::ACL_FLOAT16, &qOut);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(kCacheHostData, kCacheShape, &kCacheDeviceAddr, aclDataType::ACL_INT8, &kCache);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(vCacheHostData, vCacheShape, &vCacheDeviceAddr, aclDataType::ACL_INT8, &vCache);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(kScaleHostData, kScaleShape, &kScaleDeviceAddr, aclDataType::ACL_FLOAT, &kScale);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(vScaleHostData, vScaleShape, &vScaleDeviceAddr, aclDataType::ACL_FLOAT, &vScale);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    // 调用aclnnQkvRmsNormRopeCache第一段接口
    ret = aclnnQkvRmsNormRopeCacheGetWorkspaceSize(qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache, kScale,
                                                   vScale, nullptr, nullptr, qkv_size, head_nums, epsilon, cacheMode,
                                                   nullptr, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret);
              cleanup(); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); cleanup(); return ret);
    }

    // 调用aclnnQkvRmsNormRopeCache第二段接口
    ret = aclnnQkvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCache failed. ERROR: %d\n", ret); cleanup();
              return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); cleanup(); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(kCacheShape, &kCacheDeviceAddr);
    PrintOutResult(vCacheShape, &vCacheDeviceAddr);

    // 6/7. 释放aclTensor、aclIntArray与device资源(与失败路径共用同一套清理)
    cleanup();

    return 0;
}
