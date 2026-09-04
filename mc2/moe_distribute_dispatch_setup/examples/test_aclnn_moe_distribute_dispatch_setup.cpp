/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_moe_distribute_dispatch_setup.cpp
 * \brief Minimal example for running dispatch setup and dispatch teardown together.
 */

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_moe_distribute_dispatch_setup.h"
#include "aclnnop/aclnn_moe_distribute_dispatch_teardown.h"

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

constexpr int32_t DEV_NUM = 2;
constexpr int64_t BS = 8;
constexpr int64_t H = 1024;
constexpr int64_t K = 1;
constexpr int64_t MOE_EXPERT_NUM = 2;
constexpr int64_t EP_WORLD_SIZE = DEV_NUM;
constexpr int64_t GLOBAL_BS = BS * EP_WORLD_SIZE;
constexpr int64_t LOCAL_EXPERT_NUM = MOE_EXPERT_NUM / EP_WORLD_SIZE;
constexpr int64_t LOCAL_TOKEN_NUM = GLOBAL_BS * ((LOCAL_EXPERT_NUM < K) ? LOCAL_EXPERT_NUM : K);
constexpr int64_t EXPERT_SHARD_TYPE = 0;
constexpr int64_t SHARED_EXPERT_NUM = 0;
constexpr int64_t SHARED_EXPERT_RANK_NUM = 0;
constexpr int64_t QUANT_MODE = 0;
constexpr int64_t EXPERT_TOKEN_NUMS_TYPE = 1;
constexpr int64_t COMM_TYPE = 2;
constexpr int64_t TIMEOUT = 100000000;

template <typename Func>
class Guard {
public:
    explicit Guard(Func &func)
        : func_(func)
    {}

    ~Guard()
    {
        func_();
    }

private:
    Func &func_;
};

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtContext context;
    aclrtStream setupStream;
    aclrtStream teardownStream;
};

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                    void **deviceAddr, aclTensor **tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    int ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret = %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("[ERROR] aclCreateTensor failed.\n"); return -1);
    return ACL_SUCCESS;
}

void DestroyTensor(aclTensor *tensor)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
}

void FreeDeviceAddr(void *deviceAddr)
{
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

int LaunchOneProcess(Args &args)
{
    std::cout << "[INFO] device_" << args.rankId << " worker start." << std::endl;
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] device_%u aclrtSetCurrentContext failed. ret = %d\n", args.rankId, ret);
              return ret);

    char groupEp[128] = {0};
    ret = HcclGetCommName(args.hcclComm, groupEp);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u HcclGetCommName failed. ret = %d\n", args.rankId, ret);
              return ret);

    auto runtimeCleanup = [&args]() {
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyStream(args.setupStream);
        aclrtDestroyStream(args.teardownStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
    };
    auto runtimeGuard = Guard<decltype(runtimeCleanup)>(runtimeCleanup);

    // setup input/output shapes
    const std::vector<int64_t> xShape{BS, H};
    const std::vector<int64_t> expertIdsShape{BS, K};
    const std::vector<int64_t> yShape{BS * K, H};
    const std::vector<int64_t> expandIdxShape{BS * K};
    const std::vector<int64_t> commCmdInfoShape{(BS * K + EP_WORLD_SIZE * LOCAL_EXPERT_NUM) * 16};

    // teardown output shapes
    const std::vector<int64_t> expandXShape{LOCAL_TOKEN_NUM, H};
    // The current teardown ACLNN wrapper requires a non-null tensor here even when quantMode is 0.
    const std::vector<int64_t> dynamicScalesShape{1};
    const std::vector<int64_t> assistInfoShape{LOCAL_TOKEN_NUM * 128};
    const std::vector<int64_t> expertTokenNumsShape{LOCAL_EXPERT_NUM};

    std::vector<int16_t> xHostData(GetShapeSize(xShape), 1);
    std::vector<int32_t> expertIdsHostData(GetShapeSize(expertIdsShape));
    for (int64_t i = 0; i < BS; ++i) {
        expertIdsHostData[i] = static_cast<int32_t>(i % MOE_EXPERT_NUM);
    }
    std::vector<int16_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<int32_t> expandIdxHostData(GetShapeSize(expandIdxShape), 0);
    std::vector<int32_t> commCmdInfoHostData(GetShapeSize(commCmdInfoShape), 0);
    std::vector<int16_t> expandXHostData(GetShapeSize(expandXShape), 0);
    std::vector<float> dynamicScalesHostData(GetShapeSize(dynamicScalesShape), 0);
    std::vector<int32_t> assistInfoHostData(GetShapeSize(assistInfoShape), 0);
    std::vector<int64_t> expertTokenNumsHostData(GetShapeSize(expertTokenNumsShape), 0);

    void *xAddr = nullptr;
    void *expertIdsAddr = nullptr;
    void *yAddr = nullptr;
    void *expandIdxAddr = nullptr;
    void *commCmdInfoAddr = nullptr;
    void *expandXAddr = nullptr;
    void *dynamicScalesAddr = nullptr;
    void *assistInfoAddr = nullptr;
    void *expertTokenNumsAddr = nullptr;
    void *setupWorkspace = nullptr;
    void *teardownWorkspace = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *y = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *commCmdInfo = nullptr;
    aclTensor *expandX = nullptr;
    aclTensor *dynamicScales = nullptr;
    aclTensor *assistInfo = nullptr;
    aclTensor *expertTokenNums = nullptr;

    auto tensorCleanup = [&]() {
        DestroyTensor(x);
        DestroyTensor(expertIds);
        DestroyTensor(y);
        DestroyTensor(expandIdx);
        DestroyTensor(commCmdInfo);
        DestroyTensor(expandX);
        DestroyTensor(dynamicScales);
        DestroyTensor(assistInfo);
        DestroyTensor(expertTokenNums);
        FreeDeviceAddr(xAddr);
        FreeDeviceAddr(expertIdsAddr);
        FreeDeviceAddr(yAddr);
        FreeDeviceAddr(expandIdxAddr);
        FreeDeviceAddr(commCmdInfoAddr);
        FreeDeviceAddr(expandXAddr);
        FreeDeviceAddr(dynamicScalesAddr);
        FreeDeviceAddr(assistInfoAddr);
        FreeDeviceAddr(expertTokenNumsAddr);
        FreeDeviceAddr(setupWorkspace);
        FreeDeviceAddr(teardownWorkspace);
    };
    auto tensorGuard = Guard<decltype(tensorCleanup)>(tensorCleanup);

    ret = CreateAclTensor(xHostData, xShape, aclDataType::ACL_FLOAT16, &xAddr, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, aclDataType::ACL_INT32, &expertIdsAddr, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, aclDataType::ACL_FLOAT16, &yAddr, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, aclDataType::ACL_INT32, &expandIdxAddr, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(commCmdInfoHostData, commCmdInfoShape, aclDataType::ACL_INT32, &commCmdInfoAddr, &commCmdInfo);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandXHostData, expandXShape, aclDataType::ACL_FLOAT16, &expandXAddr, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, aclDataType::ACL_FLOAT, &dynamicScalesAddr,
                          &dynamicScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(assistInfoHostData, assistInfoShape, aclDataType::ACL_INT32, &assistInfoAddr, &assistInfo);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, aclDataType::ACL_INT64, &expertTokenNumsAddr,
                          &expertTokenNums);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t setupWorkspaceSize = 0;
    aclOpExecutor *setupExecutor = nullptr;
    ret = aclnnMoeDistributeDispatchSetupGetWorkspaceSize(
        x, expertIds, nullptr, nullptr, groupEp, EP_WORLD_SIZE, args.rankId, MOE_EXPERT_NUM, EXPERT_SHARD_TYPE,
        SHARED_EXPERT_NUM, SHARED_EXPERT_RANK_NUM, QUANT_MODE, GLOBAL_BS, COMM_TYPE, nullptr, y, expandIdx, commCmdInfo,
        &setupWorkspaceSize, &setupExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] device_%u DispatchSetupGetWorkspaceSize failed. ret = %d\n", args.rankId, ret);
              return ret);

    if (setupWorkspaceSize > 0) {
        ret = aclrtMalloc(&setupWorkspace, setupWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] device_%u setup workspace malloc failed. ret = %d\n", args.rankId, ret);
                  return ret);
    }
    ret = aclnnMoeDistributeDispatchSetup(setupWorkspace, setupWorkspaceSize, setupExecutor, args.setupStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u DispatchSetup failed. ret = %d\n", args.rankId, ret);
              return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.setupStream, TIMEOUT);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u setup synchronize failed. ret = %d\n", args.rankId, ret);
              return ret);
    LOG_PRINT("[INFO] device_%u dispatch setup success.\n", args.rankId);

    uint64_t teardownWorkspaceSize = 0;
    aclOpExecutor *teardownExecutor = nullptr;
    ret = aclnnMoeDistributeDispatchTeardownGetWorkspaceSize(
        x, y, expertIds, commCmdInfo, groupEp, EP_WORLD_SIZE, args.rankId, MOE_EXPERT_NUM, EXPERT_SHARD_TYPE,
        SHARED_EXPERT_NUM, SHARED_EXPERT_RANK_NUM, QUANT_MODE, GLOBAL_BS, EXPERT_TOKEN_NUMS_TYPE, COMM_TYPE, nullptr,
        expandX, dynamicScales, assistInfo, expertTokenNums, &teardownWorkspaceSize, &teardownExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] device_%u DispatchTeardownGetWorkspaceSize failed. ret = %d\n", args.rankId, ret);
              return ret);

    if (teardownWorkspaceSize > 0) {
        ret = aclrtMalloc(&teardownWorkspace, teardownWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] device_%u teardown workspace malloc failed. ret = %d\n", args.rankId, ret);
                  return ret);
    }
    ret = aclnnMoeDistributeDispatchTeardown(teardownWorkspace, teardownWorkspaceSize, teardownExecutor,
                                             args.teardownStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] device_%u DispatchTeardown failed. ret = %d\n", args.rankId, ret);
              return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.teardownStream, TIMEOUT);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] device_%u teardown synchronize failed. ret = %d\n", args.rankId, ret);
              return ret);
    LOG_PRINT("[INFO] device_%u dispatch teardown success.\n", args.rankId);
    return ACL_SUCCESS;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

    aclrtContext contexts[DEV_NUM];
    aclrtStream setupStreams[DEV_NUM];
    aclrtStream teardownStreams[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; ++rankId) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&contexts[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&setupStreams[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&teardownStreams[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
    }

    int32_t devices[DEV_NUM] = {0, 1};
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d\n", ret); return ret);

    Args args[DEV_NUM];
    int results[DEV_NUM] = {ACL_SUCCESS, ACL_SUCCESS};
    std::thread threads[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; ++rankId) {
        args[rankId] = {rankId, comms[rankId], contexts[rankId], setupStreams[rankId], teardownStreams[rankId]};
        threads[rankId] =
            std::thread([&args, &results, rankId]() { results[rankId] = LaunchOneProcess(args[rankId]); });
    }
    for (auto &thread : threads) {
        thread.join();
    }

    ret = aclFinalize();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclFinalize failed. ret = %d\n", ret); return ret);
    for (uint32_t rankId = 0; rankId < DEV_NUM; ++rankId) {
        CHECK_RET(results[rankId] == ACL_SUCCESS,
                  LOG_PRINT("[ERROR] device_%u failed. ret = %d\n", rankId, results[rankId]);
                  return results[rankId]);
    }
    return ACL_SUCCESS;
}
