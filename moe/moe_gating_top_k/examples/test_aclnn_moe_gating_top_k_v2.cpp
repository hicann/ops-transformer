/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
  * CANN Open Software License Agreement Version 2.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the root of the software repository for the full text of the License.
  */

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_v2.h"
#include <iostream>
#include <vector>
#include <random>

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

std::vector<float> GenerateRandomFloats(int64_t count)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    std::vector<float> result(count);
    for (auto& num : result) {
        num = dist(gen);
    }
    return result;
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
    // 1.（固定写法）device/stream初始化,参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {3, 256};
    std::vector<int64_t> biasShape = {256};
    std::vector<int64_t> inputIdsShape = {3};
    std::vector<int64_t> tid2eidShape = {3, 2};
    std::vector<int64_t> outShape = {3, 2};
    std::vector<int64_t> expertIdOutShape = {3, 2};
    std::vector<int64_t> normOutShape = {3, 256};

    void* inputAddr = nullptr;
    void* biasAddr = nullptr;
    void* inputIdsAddr = nullptr;
    void* tid2eidAddr = nullptr;
    void* outAddr = nullptr;
    void* expertIdOutAddr = nullptr;
    void* normOutAddr = nullptr;

    aclTensor* input = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* inputIds = nullptr;
    aclTensor* tid2eid = nullptr;
    aclTensor* out = nullptr;
    aclTensor* expertIdOut = nullptr;
    aclTensor* normOut = nullptr;

    std::vector<float> inputHostData = GenerateRandomFloats(GetShapeSize(inputShape));
    std::vector<float> biasHostData = GenerateRandomFloats(GetShapeSize(biasShape));
    std::vector<int32_t> inputIdsHostData = {0, 2, 1};
    std::vector<int32_t> tid2eidHostData = {10, 42, 33, 127, 5, 88};
    std::vector<float> outHostData(GetShapeSize(outShape));
    std::vector<int32_t> expertIdOutHostData(GetShapeSize(expertIdOutShape));
    std::vector<float> normOutHostData(GetShapeSize(normOutShape));

    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建inputIds aclTensor
    ret = CreateAclTensor(inputIdsHostData, inputIdsShape, &inputIdsAddr, aclDataType::ACL_INT32, &inputIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建tid2eid aclTensor
    ret = CreateAclTensor(tid2eidHostData, tid2eidShape, &tid2eidAddr, aclDataType::ACL_INT32, &tid2eid);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建expertIdOut aclTensor
    ret =
        CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建normOut aclTensor
    ret = CreateAclTensor(normOutHostData, normOutShape, &normOutAddr, aclDataType::ACL_FLOAT, &normOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的算子接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnMoeGatingTopKV2第一段接口
    ret = aclnnMoeGatingTopKV2GetWorkspaceSize(input, bias, inputIds, tid2eid, 2, 1, 1, 1, 0, 1, false, 1.0, 1e-20, out,
                                               expertIdOut, normOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeGatingTopKV2第二段接口
    ret = aclnnMoeGatingTopKV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKV2 failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(bias);
    aclDestroyTensor(inputIds);
    aclDestroyTensor(tid2eid);
    aclDestroyTensor(out);
    aclDestroyTensor(expertIdOut);
    aclDestroyTensor(normOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputAddr);
    aclrtFree(biasAddr);
    aclrtFree(inputIdsAddr);
    aclrtFree(tid2eidAddr);
    aclrtFree(outAddr);
    aclrtFree(normOutAddr);
    aclrtFree(expertIdOutAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
