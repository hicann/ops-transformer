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
 * \file test_aclnn_moe_distribute_combine_setup.cpp
 * \brief aclnn直调测试脚本
 */

#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_moe_distribute_combine_setup.h"
#include "aclnnop/aclnn_moe_distribute_combine_teardown.h"

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

constexpr int DEV_NUM = 2;

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

int64_t AlignN(int64_t x, int64_t n)
{
    return (x + n - 1) / n * n;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    HcclComm hcclEpComm;
    aclrtStream combinesetupstream;
    aclrtStream combineteardownstream;
    aclrtContext context;
};

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
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
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return ret);

    auto destroyFunc = [&args]() {
        std::cout << "== begin to destroy " << std::endl;
        HcclCommDestroy(args.hcclEpComm);
        aclrtDestroyStream(args.combinesetupstream);
        aclrtDestroyStream(args.combineteardownstream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
    };
    auto guard = Guard<decltype(destroyFunc)>(destroyFunc);

    // 场景参数
    int64_t bs = 16;
    int64_t h = 4096;
    int64_t k = 6;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 32;
    int64_t commQuantMode = 0;
    int64_t tpWorldSize = 1;
    int64_t epWorldSize = DEV_NUM;
    int64_t commType = 2;
    int64_t timeOut = 100000000;

    int64_t globalBS = bs * epWorldSize;
    int64_t localExpertNum;
    int64_t localToken;

    if (args.epRankId < sharedExpertRankNum) {
        localExpertNum = 1;
        localToken = globalBS / sharedExpertRankNum;
    } else {
        localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
        localToken = globalBS * (localExpertNum < k ? localExpertNum : k);
    }

    uint64_t assistInfoForCombineOutSize = localToken * 128;

    // 定义 shape
    std::vector<int64_t> expandXShape{tpWorldSize * localToken, h};
    std::vector<int64_t> expertIdsShape{bs, k};
    std::vector<int64_t> expertScalesShape{bs, k};
    std::vector<int64_t> expandIdxShape{bs * k};
    std::vector<int64_t> assistInfoForCombineOutShape{assistInfoForCombineOutSize};
    std::vector<int64_t> quantExpandXOutShape{tpWorldSize * localToken,
                                              AlignN(AlignN(h, 32) + AlignN(h, 8) / 8 * sizeof(float), 512)};
    std::vector<int64_t> commCmdInfoOutShapeforcombine{(localToken + epWorldSize) * 16};
    std::vector<int64_t> xOutShape{bs, h};

    int64_t expandXShapeSize = GetShapeSize(expandXShape);
    int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
    int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
    int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
    int64_t assistInfoForCombineOutShapeSize = GetShapeSize(assistInfoForCombineOutShape);
    int64_t quantExpandXOutShapeSize = GetShapeSize(quantExpandXOutShape);
    int64_t commCmdInfoOutShapeSizeforcombine = GetShapeSize(commCmdInfoOutShapeforcombine);
    int64_t xOutShapeSize = GetShapeSize(xOutShape);

    // 构造 host 数据
    std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
        for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
            expertIdsHostData.push_back(k_id);
        }
    }
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int32_t> assistInfoForCombineOutHostData(assistInfoForCombineOutShapeSize, 0);
    std::vector<int8_t> quantExpandXOutHostData(quantExpandXOutShapeSize, 0);
    std::vector<int32_t> commCmdInfoOutforCombineHostData(commCmdInfoOutShapeSizeforcombine, 0);
    std::vector<float> xOutHostData(xOutShapeSize, 0);

    // 声明 device 地址和 tensor
    void *expandXDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *assistInfoForCombineOutDeviceAddr = nullptr;
    void *quantExpandXOutDeviceAddr = nullptr;
    void *commCmdInfoOutforCombineDeviceAddr = nullptr;
    void *xOutDeviceAddr = nullptr;

    aclTensor *expandX = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *expertScales = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *assistInfoForCombineOut = nullptr;
    aclTensor *quantExpandXOut = nullptr;
    aclTensor *commCmdInfoOutforCombine = nullptr;
    aclTensor *xOut = nullptr;

    // 创建 tensor
    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, aclDataType::ACL_FLOAT16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(assistInfoForCombineOutHostData, assistInfoForCombineOutShape,
                          &assistInfoForCombineOutDeviceAddr, aclDataType::ACL_INT32, &assistInfoForCombineOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(quantExpandXOutHostData, quantExpandXOutShape, &quantExpandXOutDeviceAddr,
                          aclDataType::ACL_INT8, &quantExpandXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(commCmdInfoOutforCombineHostData, commCmdInfoOutShapeforcombine,
                          &commCmdInfoOutforCombineDeviceAddr, aclDataType::ACL_INT32, &commCmdInfoOutforCombine);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_FLOAT16, &xOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 声明算子执行必需变量
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    /******************************调用combine_setup********************************************/
    ret = aclnnMoeDistributeCombineSetupGetWorkspaceSize(
        expandX, expertIds, assistInfoForCombineOut, hcomEpName, epWorldSize, args.epRankId, moeExpertNum,
        expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, commQuantMode, commType, nullptr,
        quantExpandXOut, commCmdInfoOutforCombine, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributeCombineSetupGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CombineSetup aclrtMalloc failed. ret = %d\n", ret);
                  return ret);
    }

    ret = aclnnMoeDistributeCombineSetup(workspaceAddr, workspaceSize, executor, args.combinesetupstream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineSetup failed. ret = %d \n", ret);
              return ret);

    ret = aclrtSynchronizeStreamWithTimeout(args.combinesetupstream, timeOut);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeDistributeCombineSetup execute successfully.\n", args.rankId);

    /******************************调用combine_teardown********************************************/
    ret = aclnnMoeDistributeCombineTeardownGetWorkspaceSize(
        expandX, quantExpandXOut, expertIds, expandIdx, expertScales, commCmdInfoOutforCombine, nullptr, nullptr,
        hcomEpName, epWorldSize, args.epRankId, moeExpertNum, expertShardType, sharedExpertNum, sharedExpertRankNum,
        globalBS, commQuantMode, commType, nullptr, xOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributeCombineTeardownGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] CombineTeardown aclrtMalloc failed. ret = %d\n", ret);
                  return ret);
    }

    ret = aclnnMoeDistributeCombineTeardown(workspaceAddr, workspaceSize, executor, args.combineteardownstream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineTeardown failed. ret = %d \n", ret);
              return ret);

    ret = aclrtSynchronizeStreamWithTimeout(args.combineteardownstream, timeOut);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeDistributeCombineTeardown execute successfully.\n", args.rankId);

    // 释放资源
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    DestroyTensor(expandX);
    DestroyTensor(expertIds);
    DestroyTensor(expertScales);
    DestroyTensor(expandIdx);
    DestroyTensor(assistInfoForCombineOut);
    DestroyTensor(quantExpandXOut);
    DestroyTensor(commCmdInfoOutforCombine);
    DestroyTensor(xOut);

    FreeDeviceAddr(expandXDeviceAddr);
    FreeDeviceAddr(expertIdsDeviceAddr);
    FreeDeviceAddr(expertScalesDeviceAddr);
    FreeDeviceAddr(expandIdxDeviceAddr);
    FreeDeviceAddr(assistInfoForCombineOutDeviceAddr);
    FreeDeviceAddr(quantExpandXOutDeviceAddr);
    FreeDeviceAddr(commCmdInfoOutforCombineDeviceAddr);
    FreeDeviceAddr(xOutDeviceAddr);

    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);

    aclrtStream combineSetupStream[DEV_NUM];
    aclrtStream combineTeardownStream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&combineSetupStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&combineTeardownStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }

    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].epRankId = rankId;
        args[rankId].hcclEpComm = comms[rankId];
        args[rankId].combinesetupstream = combineSetupStream[rankId];
        args[rankId].combineteardownstream = combineTeardownStream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneProcess, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
