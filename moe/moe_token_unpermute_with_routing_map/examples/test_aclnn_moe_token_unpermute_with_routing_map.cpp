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
#include "aclnnop/aclnn_moe_token_unpermute_with_routing_map.h"

#include <cmath>
#include <cstdio>
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
    auto allocSize = size == 0 ? 1 : size;
    auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    if (size > 0) {
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CopyDeviceToHost(void* deviceAddr, std::vector<T>& hostData, const char* name)
{
    auto size = hostData.size() * sizeof(T);
    if (size == 0) {
        return ACL_SUCCESS;
    }
    auto ret = aclrtMemcpy(hostData.data(), size, deviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name, ret); return ret);
    return ACL_SUCCESS;
}

int RunMoeTokenUnpermute(aclTensor* permutedTokens, aclTensor* sortedIndices, aclTensor* routingMapOptional,
                         aclTensor* probs, bool padMode, aclIntArray* restoreShapeOptional, aclTensor* unpermutedTokens,
                         aclTensor* outIndex, aclTensor* permuteTokenId, aclTensor* permuteProbs, aclrtStream stream)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(
        permutedTokens, sortedIndices, routingMapOptional, probs, padMode, restoreShapeOptional, unpermutedTokens,
        outIndex, permuteTokenId, permuteProbs, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnMoeTokenUnpermuteWithRoutingMap(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        LOG_PRINT("aclnnMoeTokenUnpermuteWithRoutingMap failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

bool CheckFloatResult(const char* name, const std::vector<float>& actual, const std::vector<float>& expected)
{
    bool pass = actual.size() == expected.size();
    constexpr float tolerance = 1e-3f;
    for (size_t i = 0; i < actual.size() && i < expected.size(); ++i) {
        float diff = std::fabs(actual[i] - expected[i]);
        LOG_PRINT("%s[%zu] is: %f, expected: %f\n", name, i, actual[i], expected[i]);
        if (diff > tolerance) {
            pass = false;
        }
    }
    return pass;
}

int RunOriginalPadModeExample(int32_t deviceId)
{
    LOG_PRINT("Run original paddedMode=true example.\n");

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> permutedTokensShape = {2, 2};
    std::vector<int64_t> sortedIndicesShape = {2};
    std::vector<int64_t> routingMapOptionalShape = {2, 2};
    std::vector<int64_t> probsShape = {2, 2};
    std::vector<int64_t> unpermutedTokensShape = {2, 2};
    std::vector<int64_t> outIndexShape = {2};
    std::vector<int64_t> permuteTokenIdShape = {2};
    std::vector<int64_t> permuteProbsShape = {2};

    void* permutedTokensDeviceAddr = nullptr;
    void* sortedIndicesDeviceAddr = nullptr;
    void* routingMapOptionalDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* unpermutedTokensDeviceAddr = nullptr;
    void* outIndexDeviceAddr = nullptr;
    void* permuteTokenIdDeviceAddr = nullptr;
    void* permuteProbsDeviceAddr = nullptr;
    // in
    aclTensor* permutedTokens = nullptr;
    aclTensor* sortedIndices = nullptr;
    aclTensor* routingMapOptional = nullptr;
    aclTensor* probs = nullptr;
    aclTensor* unpermutedTokens = nullptr;
    aclTensor* outIndex = nullptr;
    aclTensor* permuteTokenId = nullptr;
    aclTensor* permuteProbs = nullptr;
    bool padMode = true;
    std::vector<int64_t> restoreShapeOptionalData = {2, 2};
    aclIntArray* restoreShapeOptional =
        aclCreateIntArray(restoreShapeOptionalData.data(), restoreShapeOptionalData.size());

    // 构造数据
    std::vector<float> permutedTokensHostData = {1.0, 1.0, 1.0, 1.0};
    std::vector<int> sortedIndicesHostData = {1, 1};
    std::vector<char> routingMapOptionalHostData = {1, 1, 1, 1};
    std::vector<float> probsHostData = {1, 1, 1, 1};

    std::vector<float> unpermutedTokensHostData = {0, 0, 0, 0};
    std::vector<int> outIndexHostData = {0, 0};
    std::vector<int> permuteTokenIdHostData = {0, 0};
    std::vector<float> permuteProbsHostData = {0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr,
                          aclDataType::ACL_FLOAT, &permutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr, aclDataType::ACL_INT32,
                          &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(routingMapOptionalHostData, routingMapOptionalShape, &routingMapOptionalDeviceAddr,
                          aclDataType::ACL_INT8, &routingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(unpermutedTokensHostData, unpermutedTokensShape, &unpermutedTokensDeviceAddr,
                          aclDataType::ACL_FLOAT, &unpermutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr,
                          aclDataType::ACL_INT32, &permuteTokenId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteProbsHostData, permuteProbsShape, &permuteProbsDeviceAddr, aclDataType::ACL_FLOAT,
                          &permuteProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = RunMoeTokenUnpermute(permutedTokens, sortedIndices, routingMapOptional, probs, padMode, restoreShapeOptional,
                               unpermutedTokens, outIndex, permuteTokenId, permuteProbs, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto unpermutedTokensSize = GetShapeSize(unpermutedTokensShape);
    std::vector<float> unpermutedTokensData(unpermutedTokensSize, 0);
    ret = CopyDeviceToHost(unpermutedTokensDeviceAddr, unpermutedTokensData, "unpermutedTokens");
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    for (int64_t i = 0; i < unpermutedTokensSize; i++) {
        LOG_PRINT("original unpermutedTokensData[%ld] is: %f\n", i, unpermutedTokensData[i]);
    }

    aclDestroyTensor(permutedTokens);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(routingMapOptional);
    aclDestroyTensor(probs);
    aclDestroyTensor(unpermutedTokens);
    aclDestroyTensor(outIndex);
    aclDestroyTensor(permuteTokenId);
    aclDestroyTensor(permuteProbs);
    aclDestroyIntArray(restoreShapeOptional);

    aclrtFree(permutedTokensDeviceAddr);
    aclrtFree(sortedIndicesDeviceAddr);
    aclrtFree(routingMapOptionalDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(unpermutedTokensDeviceAddr);
    aclrtFree(outIndexDeviceAddr);
    aclrtFree(permuteTokenIdDeviceAddr);
    aclrtFree(permuteProbsDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

int RunNotPadNoMinusOneRegressionExample(int32_t deviceId)
{
    LOG_PRINT("Run paddedMode=false no-minus-one regression example.\n");

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> permutedTokensShape = {4, 4};
    std::vector<int64_t> sortedIndicesShape = {4};
    std::vector<int64_t> unpermutedTokensShape = {2, 4};
    std::vector<int64_t> outIndexShape = {4};
    std::vector<int64_t> permuteTokenIdShape = {4};
    std::vector<int64_t> permuteProbsShape = {0};

    void* permutedTokensDeviceAddr = nullptr;
    void* sortedIndicesDeviceAddr = nullptr;
    void* unpermutedTokensDeviceAddr = nullptr;
    void* outIndexDeviceAddr = nullptr;
    void* permuteTokenIdDeviceAddr = nullptr;
    void* permuteProbsDeviceAddr = nullptr;

    aclTensor* permutedTokens = nullptr;
    aclTensor* sortedIndices = nullptr;
    aclTensor* routingMapOptional = nullptr;
    aclTensor* probs = nullptr;
    aclTensor* unpermutedTokens = nullptr;
    aclTensor* outIndex = nullptr;
    aclTensor* permuteTokenId = nullptr;
    aclTensor* permuteProbs = nullptr;

    bool padMode = false;
    std::vector<int64_t> restoreShapeOptionalData = {2, 4};
    aclIntArray* restoreShapeOptional =
        aclCreateIntArray(restoreShapeOptionalData.data(), restoreShapeOptionalData.size());

    std::vector<float> permutedTokensHostData = {
        1.0f,   2.0f,   3.0f,   4.0f,   10.0f,   20.0f,   30.0f,   40.0f,
        100.0f, 200.0f, 300.0f, 400.0f, 1000.0f, 2000.0f, 3000.0f, 4000.0f,
    };
    std::vector<int> sortedIndicesHostData = {0, 1, 2, 3};
    std::vector<float> unpermutedTokensHostData(GetShapeSize(unpermutedTokensShape), 0.0f);
    std::vector<int> outIndexHostData(GetShapeSize(outIndexShape), 0);
    std::vector<int> permuteTokenIdHostData(GetShapeSize(permuteTokenIdShape), 0);
    std::vector<float> permuteProbsHostData(GetShapeSize(permuteProbsShape), 0.0f);

    ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr,
                          aclDataType::ACL_FLOAT, &permutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr, aclDataType::ACL_INT32,
                          &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(unpermutedTokensHostData, unpermutedTokensShape, &unpermutedTokensDeviceAddr,
                          aclDataType::ACL_FLOAT, &unpermutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr,
                          aclDataType::ACL_INT32, &permuteTokenId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteProbsHostData, permuteProbsShape, &permuteProbsDeviceAddr, aclDataType::ACL_FLOAT,
                          &permuteProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = RunMoeTokenUnpermute(permutedTokens, sortedIndices, routingMapOptional, probs, padMode, restoreShapeOptional,
                               unpermutedTokens, outIndex, permuteTokenId, permuteProbs, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto unpermutedTokensSize = GetShapeSize(unpermutedTokensShape);
    std::vector<float> unpermutedTokensData(unpermutedTokensSize, 0.0f);
    ret = CopyDeviceToHost(unpermutedTokensDeviceAddr, unpermutedTokensData, "unpermutedTokens");
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> expected = {
        11.0f, 22.0f, 33.0f, 44.0f, 1100.0f, 2200.0f, 3300.0f, 4400.0f,
    };

    bool pass = CheckFloatResult("no-minus-one unpermutedTokensData", unpermutedTokensData, expected);
    LOG_PRINT("not-pad no-minus-one regression test %s\n", pass ? "PASSED" : "FAILED");

    aclDestroyTensor(permutedTokens);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(unpermutedTokens);
    aclDestroyTensor(outIndex);
    aclDestroyTensor(permuteTokenId);
    aclDestroyTensor(permuteProbs);
    aclDestroyIntArray(restoreShapeOptional);

    aclrtFree(permutedTokensDeviceAddr);
    aclrtFree(sortedIndicesDeviceAddr);
    aclrtFree(unpermutedTokensDeviceAddr);
    aclrtFree(outIndexDeviceAddr);
    aclrtFree(permuteTokenIdDeviceAddr);
    aclrtFree(permuteProbsDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return pass ? 0 : 1;
}

int RunMinusOneWithAlignedProbsExample(int32_t deviceId)
{
    LOG_PRINT("Run paddedMode=false minus-one with aligned probs example.\n");

    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> permutedTokensShape = {4, 4};
    std::vector<int64_t> sortedIndicesShape = {4};
    std::vector<int64_t> routingMapOptionalShape = {2, 2};
    std::vector<int64_t> probsShape = {2, 2};
    std::vector<int64_t> unpermutedTokensShape = {2, 4};
    std::vector<int64_t> outIndexShape = {4};
    std::vector<int64_t> permuteTokenIdShape = {4};
    std::vector<int64_t> permuteProbsShape = {4};

    void* permutedTokensDeviceAddr = nullptr;
    void* sortedIndicesDeviceAddr = nullptr;
    void* routingMapOptionalDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* unpermutedTokensDeviceAddr = nullptr;
    void* outIndexDeviceAddr = nullptr;
    void* permuteTokenIdDeviceAddr = nullptr;
    void* permuteProbsDeviceAddr = nullptr;

    aclTensor* permutedTokens = nullptr;
    aclTensor* sortedIndices = nullptr;
    aclTensor* routingMapOptional = nullptr;
    aclTensor* probs = nullptr;
    aclTensor* unpermutedTokens = nullptr;
    aclTensor* outIndex = nullptr;
    aclTensor* permuteTokenId = nullptr;
    aclTensor* permuteProbs = nullptr;

    bool padMode = false;
    std::vector<int64_t> restoreShapeOptionalData = {2, 4};
    aclIntArray* restoreShapeOptional =
        aclCreateIntArray(restoreShapeOptionalData.data(), restoreShapeOptionalData.size());

    std::vector<float> permutedTokensHostData = {
        2.0f, 4.0f, 6.0f, 8.0f, 4.0f, 8.0f, 12.0f, 16.0f, 8.0f, 16.0f, 24.0f, 32.0f, 100.0f, 200.0f, 300.0f, 400.0f,
    };
    std::vector<int> sortedIndicesHostData = {0, -1, 1, 2};

    // Assumption for the -1 feature: permute_probs is slot-aligned with sortedIndices.
    // The second slot corresponds to sortedIndices[1] == -1, so its prob is padded as 0.
    std::vector<char> routingMapOptionalHostData = {1, 1, 1, 1};
    std::vector<float> probsHostData = {0.5f, 0.0f, 0.25f, 0.75f};

    std::vector<float> unpermutedTokensHostData(GetShapeSize(unpermutedTokensShape), 0.0f);
    std::vector<int> outIndexHostData(GetShapeSize(outIndexShape), 0);
    std::vector<int> permuteTokenIdHostData(GetShapeSize(permuteTokenIdShape), 0);
    std::vector<float> permuteProbsHostData(GetShapeSize(permuteProbsShape), 0.0f);

    ret = CreateAclTensor(permutedTokensHostData, permutedTokensShape, &permutedTokensDeviceAddr,
                          aclDataType::ACL_FLOAT, &permutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesHostData, sortedIndicesShape, &sortedIndicesDeviceAddr, aclDataType::ACL_INT32,
                          &sortedIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(routingMapOptionalHostData, routingMapOptionalShape, &routingMapOptionalDeviceAddr,
                          aclDataType::ACL_INT8, &routingMapOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(unpermutedTokensHostData, unpermutedTokensShape, &unpermutedTokensDeviceAddr,
                          aclDataType::ACL_FLOAT, &unpermutedTokens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outIndexHostData, outIndexShape, &outIndexDeviceAddr, aclDataType::ACL_INT32, &outIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteTokenIdHostData, permuteTokenIdShape, &permuteTokenIdDeviceAddr,
                          aclDataType::ACL_INT32, &permuteTokenId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteProbsHostData, permuteProbsShape, &permuteProbsDeviceAddr, aclDataType::ACL_FLOAT,
                          &permuteProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = RunMoeTokenUnpermute(permutedTokens, sortedIndices, routingMapOptional, probs, padMode, restoreShapeOptional,
                               unpermutedTokens, outIndex, permuteTokenId, permuteProbs, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto unpermutedTokensSize = GetShapeSize(unpermutedTokensShape);
    std::vector<float> unpermutedTokensData(unpermutedTokensSize, 0.0f);
    ret = CopyDeviceToHost(unpermutedTokensDeviceAddr, unpermutedTokensData, "unpermutedTokens");
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto permuteProbsSize = GetShapeSize(permuteProbsShape);
    std::vector<float> permuteProbsData(permuteProbsSize, 0.0f);
    ret = CopyDeviceToHost(permuteProbsDeviceAddr, permuteProbsData, "permuteProbs");
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> expectedTokens = {
        1.0f, 2.0f, 3.0f, 4.0f, 7.0f, 14.0f, 21.0f, 28.0f,
    };
    std::vector<float> expectedPermuteProbs = {0.5f, 0.0f, 0.25f, 0.75f};

    bool pass = CheckFloatResult("minus-one-probs unpermutedTokensData", unpermutedTokensData, expectedTokens);
    pass = CheckFloatResult("minus-one-probs permuteProbsData", permuteProbsData, expectedPermuteProbs) && pass;
    LOG_PRINT("minus-one aligned-probs test %s\n", pass ? "PASSED" : "FAILED");

    aclDestroyTensor(permutedTokens);
    aclDestroyTensor(sortedIndices);
    aclDestroyTensor(routingMapOptional);
    aclDestroyTensor(probs);
    aclDestroyTensor(unpermutedTokens);
    aclDestroyTensor(outIndex);
    aclDestroyTensor(permuteTokenId);
    aclDestroyTensor(permuteProbs);
    aclDestroyIntArray(restoreShapeOptional);

    aclrtFree(permutedTokensDeviceAddr);
    aclrtFree(sortedIndicesDeviceAddr);
    aclrtFree(routingMapOptionalDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(unpermutedTokensDeviceAddr);
    aclrtFree(outIndexDeviceAddr);
    aclrtFree(permuteTokenIdDeviceAddr);
    aclrtFree(permuteProbsDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return pass ? 0 : 1;
}

int main()
{
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;

    auto ret = RunOriginalPadModeExample(deviceId);
    CHECK_RET(ret == 0, return ret);

    ret = RunNotPadNoMinusOneRegressionExample(deviceId);
    CHECK_RET(ret == 0, return ret);

    ret = RunMinusOneWithAlignedProbsExample(deviceId);
    CHECK_RET(ret == 0, return ret);

    return 0;
}
