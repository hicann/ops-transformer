/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnnop/aclnn_moe_token_permute_with_routing_map_grad.h"
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

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

template <typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T> &hostData)
{
    size_t fileSize = 1;
    for (int64_t i : shape) {
        fileSize *= i;
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    // 获取文件大小
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    hostData.reserve(fileSize);
    if (file.read(reinterpret_cast<char *>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << "Failed to read file" << std::endl;
        return 1;
    }
    file.close();
    return true;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T> &hostData)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        LOG_PRINT("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char *>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        LOG_PRINT("Write file Failed.");
        return false;
    }

    return true;
}
void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
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
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造

    int64_t num_token = 4096;
    int64_t hidden_size = 7168;
    int64_t num_expert = 256;
    int64_t num_capacity = 16;
    std::vector<float> permuted_output_grad_Data(num_expert * num_capacity * hidden_size, 0);
    std::vector<int64_t> permuted_output_grad_Shape = {num_expert * num_capacity, hidden_size};
    void *permuted_output_grad_Addr = nullptr;
    aclTensor *permuted_output_grad = nullptr;

    ret = CreateAclTensor(permuted_output_grad_Data, permuted_output_grad_Shape, &permuted_output_grad_Addr,
                          aclDataType::ACL_FLOAT, &permuted_output_grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> permutedProbsOutputGradOptional(num_expert * num_capacity, 0.1);
    std::vector<int64_t> permutedProbsOutputGradOptionalShape = {num_expert * num_capacity};
    void *permutedProbsOutputGrad_Addr = nullptr;
    aclTensor *ppermutedProbsOutputGrad = nullptr;
    ret = CreateAclTensor(permutedProbsOutputGradOptional, permutedProbsOutputGradOptionalShape,
                          &permutedProbsOutputGrad_Addr, aclDataType::ACL_FLOAT, &ppermutedProbsOutputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int> sortedIndicesData(num_expert * num_capacity, 0);
    std::vector<int64_t> sortedIndicesShape = {num_expert * num_capacity};
    void *sortedIndicesAddr = nullptr;
    aclTensor *sortedIndices = nullptr;
    ReadFile("./sortedIndices.bin", sortedIndicesShape, sortedIndicesData);
    ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr, aclDataType::ACL_INT32,
                          &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<char> routingMapOptionalData(num_token * num_expert, 1);
    std::vector<int64_t> routingMapOptionalShape = {num_token, num_expert};
    void *routingMapOptionalAddr = nullptr;
    aclTensor *proutingMapOptional = nullptr;

    ret = CreateAclTensor(routingMapOptionalData, routingMapOptionalShape, &routingMapOptionalAddr,
                          aclDataType::ACL_INT8, &proutingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData(num_token * hidden_size, 0.0f);
    std::vector<int64_t> outShape = {num_token, hidden_size};
    // std::vector<int64_t> outShape = {num_token};
    void *outAddr = nullptr;
    aclTensor *out = nullptr;

    ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outData2(num_token * num_expert, 0.0f);
    std::vector<int64_t> outShape2 = {num_token, num_expert};
    void *outAddr2 = nullptr;
    aclTensor *out2 = nullptr;

    ret = CreateAclTensor(outData2, outShape2, &outAddr2, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // 调用aclnnMoeTokenPermuteGrad第一段接口
    ret = aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(permuted_output_grad, ppermutedProbsOutputGrad,
                                                                 sortedIndices, proutingMapOptional, num_expert,
                                                                 num_token, true, out, out2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnMoeTokenPermuteWithRoutingMapGrad第二段接口
    ret = aclnnMoeTokenPermuteWithRoutingMapGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithRoutingMapGrad failed. ERROR: %d\n", ret);
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(outShape, &outAddr);
    PrintOutResult(outShape2, &outAddr2);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(permuted_output_grad);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(out);

    // 7. 释放device资源
    aclrtFree(permuted_output_grad_Addr);
    aclrtFree(sortedIndicesAddr);
    aclrtFree(outAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
