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
 * \file test_aclnn_moe_distribute_dispatch_v2.cpp
 * \brief
 */

#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_moe_distribute_dispatch_v2.h"
#include "aclnnop/aclnn_moe_distribute_combine_v2.h"

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

struct Args {
    uint32_t rankId;
    uint32_t epRankId;
    HcclComm hcclEpComm;
    aclrtStream dispatchV2Stream;
    aclrtStream combineV2Stream;
    aclrtContext context;
};

const uint32_t MACHINE_NUM = 1;
const char *env_dev_num = std::getenv("ENV_DEV_NUM");

const uint32_t EP_WORLD_SIZE = 2;
const uint32_t DEV_NUM = EP_WORLD_SIZE;

const bool IS_TEST_A2 = false;
const bool IS_TEST_A3A5 = true;
const uint32_t EP_WORLD_SIZE_A2 = 8;
const uint32_t DEV_NUM_A2 = EP_WORLD_SIZE_A2;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

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

int launchOneThreadDispatchV2AndCombineV2_A3A5(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, context = %p\n",
              args.rankId, hcomEpName, args.dispatchV2Stream, args.combineV2Stream, args.context);

    // 设置场景
    int64_t BS = 8;
    int64_t H = 7168;
    int64_t K = 1;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = EP_WORLD_SIZE - sharedExpertRankNum;
    int64_t quantMode = 0;
    int64_t globalBS = BS * EP_WORLD_SIZE;
    int64_t expertTokenNumsType = 0;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupListType = 0;
    int64_t localExpertNum;
    int64_t A;
    if (args.epRankId < sharedExpertRankNum) {
        // 共享专家卡
        localExpertNum = 1;
        A = globalBS / sharedExpertRankNum;
    } else {
        // Moe专家卡
        localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
        A = globalBS * (localExpertNum < K ? localExpertNum : K);
    }
    std::string commAlg = "";

    /* 根据当前场景，构造device侧输入输出变量*/
    // 声明device侧输入输出变量
    void *xDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;
    void *expandXDeviceAddr = nullptr;
    void *dynamicScalesDeviceAddr = nullptr;
    void *expandIdxDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;
    void *epRecvCountsDeviceAddr = nullptr;
    void *tpRecvCountsDeviceAddr = nullptr;
    void *expandScalesDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *scales = nullptr;
    aclTensor *expertScales = nullptr;
    aclTensor *expandX = nullptr;
    aclTensor *dynamicScales = nullptr;
    aclTensor *expandIdx = nullptr;
    aclTensor *expertTokenNums = nullptr;
    aclTensor *epRecvCounts = nullptr;
    aclTensor *tpRecvCounts = nullptr;
    aclTensor *expandScales = nullptr;

    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{BS, H};
    std::vector<int64_t> expertIdsShape{BS, K};
    std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
    std::vector<int64_t> expertScalesShape{BS, K};
    std::vector<int64_t> expandXShape{A, H};
    std::vector<int64_t> dynamicScalesShape{A};
    std::vector<int64_t> expandIdxShape{A * 128};
    std::vector<int64_t> expertTokenNumsShape{localExpertNum};
    std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE};
    std::vector<int64_t> tpRecvCountsShape{1};
    std::vector<int64_t> expandScalesShape{A};

    long long xShapeSize = GetShapeSize(xShape);
    long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
    long long scalesShapeSize = GetShapeSize(scalesShape);
    long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
    long long expandXShapeSize = GetShapeSize(expandXShape);
    long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
    long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
    long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    long long epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
    long long tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
    long long expandScalesShapeSize = GetShapeSize(expandScalesShape);

    // 构造host侧变量
    std::vector<op::fp16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
        // 每个token发给moe专家{0, 1, ... k - 1}
        for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
            expertIdsHostData.push_back(k_id);
        }
    }
    std::vector<float> scalesHostData(scalesShapeSize, 0);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
    std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
    std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
    std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
    std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

    // 构造device侧变量
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr,
                          (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &dynamicScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr,
                          aclDataType::ACL_INT64, &expertTokenNums);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32,
                          &epRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32,
                          &tpRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expandScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    /* 声明算子执行必需变量 */
    uint64_t dispatchV2WorkspaceSize = 0;
    aclOpExecutor *dispatchV2Executor = nullptr;
    void *dispatchV2WorkspaceAddr = nullptr;

    uint64_t combineV2WorkspaceSize = 0;
    aclOpExecutor *combineV2Executor = nullptr;
    void *combineV2WorkspaceAddr = nullptr;

    /* 依次执行dispatchV2及combineV2算子 */
    // 调用dispatchV2算子第一阶段接口
    ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(
        x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr, expertScales, hcomEpName, EP_WORLD_SIZE,
        args.epRankId, moeExpertNum, nullptr, 0, 0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode,
        globalBS, expertTokenNumsType, commAlg.c_str(), expandX, dynamicScales, expandIdx, expertTokenNums,
        epRecvCounts, tpRecvCounts, expandScales, &dispatchV2WorkspaceSize, &dispatchV2Executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2GetWorkspaceSize failed. ret = %d\n", ret);
              return ret);
    // 根据dispatchV2算子第一阶段接口计算出的workspaceSize申请device内存
    if (dispatchV2WorkspaceSize > 0) {
        ret = aclrtMalloc(&dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }
    // 调用dispatchV2算子第二阶段接口
    ret = aclnnMoeDistributeDispatchV2(dispatchV2WorkspaceAddr, dispatchV2WorkspaceSize, dispatchV2Executor,
                                       args.dispatchV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributedispatchV2 failed. ret = %d\n", ret);
              return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret);
              return ret);

    // 调用combineV2算子第一阶段接口
    ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(
        expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, nullptr, 0, 0, expertShardType,
        sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode, groupListType, commAlg.c_str(), x,
        &combineV2WorkspaceSize, &combineV2Executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2GetWorkspaceSize failed. ret = %d\n", ret);
              return ret);
    // 根据combineV2算子第一阶段接口计算出的workspaceSize申请device内存
    if (combineV2WorkspaceSize > 0) {
        ret = aclrtMalloc(&combineV2WorkspaceAddr, combineV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
    }
    // 调用combineV2算子第二阶段接口
    ret = aclnnMoeDistributeCombineV2(combineV2WorkspaceAddr, combineV2WorkspaceSize, combineV2Executor,
                                      args.combineV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributecombineV2 failed. ret = %d\n", ret); return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret);
              return ret);

    LOG_PRINT("[INFO] device_%d aclnnMoeDistributedispatchV2 and aclnnMoeDistributecombineV2 execute successfully.\n",
              args.rankId);

    // 释放device资源
    if (dispatchV2WorkspaceSize > 0) {
        aclrtFree(dispatchV2WorkspaceAddr);
    }
    if (combineV2WorkspaceSize > 0) {
        aclrtFree(combineV2WorkspaceAddr);
    }
    DestroyTensor(x);
    DestroyTensor(expertIds);
    DestroyTensor(scales);
    DestroyTensor(expertScales);
    DestroyTensor(expandX);
    DestroyTensor(dynamicScales);
    DestroyTensor(expandIdx);
    DestroyTensor(expertTokenNums);
    DestroyTensor(epRecvCounts);
    DestroyTensor(tpRecvCounts);
    DestroyTensor(expandScales);

    FreeDeviceAddr(xDeviceAddr);
    FreeDeviceAddr(expertIdsDeviceAddr);
    FreeDeviceAddr(scalesDeviceAddr);
    FreeDeviceAddr(expertScalesDeviceAddr);
    FreeDeviceAddr(expandXDeviceAddr);
    FreeDeviceAddr(dynamicScalesDeviceAddr);
    FreeDeviceAddr(expandIdxDeviceAddr);
    FreeDeviceAddr(expertTokenNumsDeviceAddr);
    FreeDeviceAddr(epRecvCountsDeviceAddr);
    FreeDeviceAddr(tpRecvCountsDeviceAddr);
    FreeDeviceAddr(expandScalesDeviceAddr);

    HcclCommDestroy(args.hcclEpComm);
    aclrtDestroyStream(args.dispatchV2Stream);
    aclrtDestroyStream(args.combineV2Stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

int launchOneThreadDispatchV2AndCombineV2_A2(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);
    char hcomEpName[128] = {0};
    ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, dispatchV2Stream = %p, combineV2Stream = %p, \
                context = %p\n",
              args.rankId, hcomEpName, args.dispatchV2Stream, args.combineV2Stream, args.context);

    int64_t Bs = 32;
    int64_t H = 7168;
    int64_t K = 8;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t moeExpertNum = 256;
    int64_t quantMode = 0;
    int64_t globalBs = Bs * EP_WORLD_SIZE_A2;
    int64_t expertTokenNumsType = 1;
    int64_t outDtype = 0;
    int64_t commQuantMode = 0;
    int64_t groupList_type = 1;
    int64_t localExpertNum;
    int64_t A;
    std::string commAlg = "fullmesh";
    if (args.epRankId < sharedExpertRankNum) {
        localExpertNum = 1;
        A = globalBs / sharedExpertRankNum;
    } else {
        localExpertNum = moeExpertNum / (EP_WORLD_SIZE_A2 - sharedExpertRankNum);
        A = globalBs * (localExpertNum < K ? localExpertNum : K);
    }

    void *xDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;
    void *expertScalesDeviceAddr = nullptr;

    void *expandXDeviceAddr = nullptr;
    void *dynamicScalesDeviceAddr = nullptr;
    void *assistInfoForCombineDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;
    void *epRecvCountsDeviceAddr = nullptr;
    void *tpRecvCountsDeviceAddr = nullptr;
    void *expandScalesDeviceAddr = nullptr;

    void *xOutDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *scales = nullptr;
    aclTensor *xActiveMask = nullptr;
    aclTensor *expertScales = nullptr;

    aclTensor *elasticInfo = nullptr; // A3
    aclTensor *expandX = nullptr;
    aclTensor *dynamicScales = nullptr;
    aclTensor *assistInfoForCombine = nullptr; // expandIdx
    aclTensor *expertTokenNums = nullptr;
    aclTensor *epRecvCounts = nullptr;
    aclTensor *tpRecvCounts = nullptr;
    aclTensor *expandScales = nullptr;

    aclTensor *activationScale = nullptr; // 预留参数
    aclTensor *weightScale = nullptr;     // 预留参数
    aclTensor *groupList = nullptr;       // 预留参数

    aclTensor *sharedExpertX = nullptr; // A3

    aclTensor *xOut = nullptr;

    // 定义当前场景下各变量维度
    std::vector<int64_t> xShape{Bs, H};
    std::vector<int64_t> expertIdsShape{Bs, K};
    std::vector<int64_t> scalesShape{moeExpertNum + 1, H};
    std::vector<int64_t> expertScalesShape{Bs, K};

    std::vector<int64_t> expandXShape{A, H};
    std::vector<int64_t> dynamicScalesShape{A};
    std::vector<int64_t> assistInfoForCombineShape{A * 128};
    std::vector<int64_t> expertTokenNumsShape{localExpertNum};
    std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE_A2}; // 不分层
    std::vector<int64_t> tpRecvCountsShape{1};
    std::vector<int64_t> expandScalesShape{A};

    std::vector<int64_t> xOutShape{Bs, H};

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
    int64_t scalesShapeSize = GetShapeSize(scalesShape);
    int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);

    int64_t expandXShapeSize = GetShapeSize(expandXShape);
    int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
    int64_t assistInfoForCombineShapeSize = GetShapeSize(assistInfoForCombineShape);
    int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
    int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
    int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);

    int64_t xOutShapeSize = GetShapeSize(xOutShape);

    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
        for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
            expertIdsHostData.push_back(k_id);
        }
    }

    std::vector<float> scalesHostData(scalesShapeSize, 0.1);
    std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);

    std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
    std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
    std::vector<int32_t> assistInfoForCombineHostData(assistInfoForCombineShapeSize, 0);
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
    std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
    std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
    std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

    std::vector<int16_t> xOutHostData(xOutShapeSize, 0);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expertScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr,
                          (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &dynamicScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(assistInfoForCombineHostData, assistInfoForCombineShape, &assistInfoForCombineDeviceAddr,
                          aclDataType::ACL_INT32, &assistInfoForCombine);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr,
                          aclDataType::ACL_INT64, &expertTokenNums);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32,
                          &epRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32,
                          &tpRecvCounts);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT,
                          &expandScales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t dispatchWorkspaceSize = 0;
    aclOpExecutor *dispatchExecutor = nullptr;
    void *dispatchWorkspaceAddr = nullptr;

    uint64_t combineWorkspaceSize = 0;
    aclOpExecutor *combineExecutor = nullptr;
    void *combineWorkspaceAddr = nullptr;

    /**************************************** 调用dispatch ********************************************/
    // 调用第一阶段接口
    ret = aclnnMoeDistributeDispatchV2GetWorkspaceSize(
        x, expertIds, (quantMode > 0 ? scales : nullptr), xActiveMask, expertScales, hcomEpName, EP_WORLD_SIZE_A2,
        args.epRankId, moeExpertNum, "", 0, 0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode,
        globalBs, expertTokenNumsType, commAlg.c_str(), expandX, dynamicScales, assistInfoForCombine, expertTokenNums,
        epRecvCounts, tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2GetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    if (dispatchWorkspaceSize > 0) {
        ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // 调用第二阶段接口
    ret = aclnnMoeDistributeDispatchV2(dispatchWorkspaceAddr, dispatchWorkspaceSize, dispatchExecutor,
                                       args.dispatchV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV2 failed. ret = %d \n", ret);
              return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.dispatchV2Stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 execute successfully.\n", args.rankId);
    /**************************************** 调用combine ********************************************/
    // 调用第一阶段接口
    ret = aclnnMoeDistributeCombineV2GetWorkspaceSize(
        expandX, expertIds, assistInfoForCombine, epRecvCounts, expertScales, tpRecvCounts, xActiveMask,
        activationScale, weightScale, groupList, expandScales, sharedExpertX, hcomEpName, EP_WORLD_SIZE_A2,
        args.epRankId, moeExpertNum, "", 0, 0, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs,
        outDtype, commQuantMode, groupList_type, commAlg.c_str(), xOut, &combineWorkspaceSize, &combineExecutor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2GetWorkspaceSize failed. ret = %d \n", ret);
              return ret);
    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (combineWorkspaceSize > 0) {
        ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    // 调用第二阶段接口
    ret =
        aclnnMoeDistributeCombineV2(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineV2Stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineV2 failed. ret = %d \n", ret);
              return ret);
    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.combineV2Stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
              return ret);
    LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV2 and aclnnMoeDistributeCombineV2                      \
                execute successfully.\n",
              args.rankId);
    // 释放device资源
    if (dispatchWorkspaceSize > 0) {
        aclrtFree(dispatchWorkspaceAddr);
    }
    if (combineWorkspaceSize > 0) {
        aclrtFree(combineWorkspaceAddr);
    }
    DestroyTensor(x);
    DestroyTensor(expertIds);
    DestroyTensor(scales);
    DestroyTensor(xActiveMask);
    DestroyTensor(expertScales);
    DestroyTensor(elasticInfo);
    DestroyTensor(expandX);
    DestroyTensor(dynamicScales);
    DestroyTensor(assistInfoForCombine);
    DestroyTensor(expertTokenNums);
    DestroyTensor(epRecvCounts);
    DestroyTensor(tpRecvCounts);
    DestroyTensor(expandScales);
    DestroyTensor(activationScale);
    DestroyTensor(weightScale);
    DestroyTensor(groupList);
    DestroyTensor(sharedExpertX);
    DestroyTensor(xOut);

    FreeDeviceAddr(xDeviceAddr);
    FreeDeviceAddr(expertIdsDeviceAddr);
    FreeDeviceAddr(scalesDeviceAddr);
    FreeDeviceAddr(expertScalesDeviceAddr);
    FreeDeviceAddr(expandXDeviceAddr);
    FreeDeviceAddr(dynamicScalesDeviceAddr);
    FreeDeviceAddr(assistInfoForCombineDeviceAddr);
    FreeDeviceAddr(expertTokenNumsDeviceAddr);
    FreeDeviceAddr(epRecvCountsDeviceAddr);
    FreeDeviceAddr(tpRecvCountsDeviceAddr);
    FreeDeviceAddr(expandScalesDeviceAddr);
    FreeDeviceAddr(xOutDeviceAddr);

    HcclCommDestroy(args.hcclEpComm);
    aclrtDestroyStream(args.dispatchV2Stream);
    aclrtDestroyStream(args.combineV2Stream);
    aclrtDestroyContext(args.context);
    LOG_PRINT("[INFO] device_%d DeStroy.\n", args.rankId);
    aclrtResetDevice(args.rankId);
    LOG_PRINT("[INFO] device_%d Reset.\n", args.rankId);
    return 0;
}

int run_example_on_A2()
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

    aclrtStream dispatchV2Stream[DEV_NUM_A2];
    aclrtStream combineV2Stream[DEV_NUM_A2];
    aclrtContext context[DEV_NUM_A2];
    for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&dispatchV2Stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&combineV2Stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
    }

    int32_t devicesEp[EP_WORLD_SIZE_A2];
    for (int32_t epId = 0; epId < EP_WORLD_SIZE_A2; epId++) {
        devicesEp[epId] = epId;
    }

    HcclComm commsEp[EP_WORLD_SIZE_A2];
    ret = HcclCommInitAll(EP_WORLD_SIZE_A2, devicesEp, commsEp);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep failed, ret %d\n", ret); return ret);

    Args args[DEV_NUM_A2];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM_A2);
    for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
        uint32_t epRankId = rankId;

        args[rankId].rankId = rankId;
        args[rankId].epRankId = epRankId;
        args[rankId].hcclEpComm = commsEp[epRankId];
        args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
        args[rankId].combineV2Stream = combineV2Stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow)
                                  std::thread(&launchOneThreadDispatchV2AndCombineV2_A2, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM_A2; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");
    return 0;
}

int run_example_on_A3A5()
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

    aclrtStream dispatchV2Stream[DEV_NUM];
    aclrtStream combineV2Stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&dispatchV2Stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&combineV2Stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
    }

    int32_t devicesEp[EP_WORLD_SIZE];
    for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
        devicesEp[epId] = epId;
    }
    // 初始化ep通信域.
    HcclComm commsEp[EP_WORLD_SIZE];
    ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep failed. ret = %d\n", ret); return ret);

    Args args[DEV_NUM];
    // 各线程调用各卡执行算子
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        uint32_t epRankId = rankId;

        args[rankId].rankId = rankId;
        args[rankId].epRankId = epRankId;
        args[rankId].hcclEpComm = commsEp[epRankId];
        args[rankId].dispatchV2Stream = dispatchV2Stream[rankId];
        args[rankId].combineV2Stream = combineV2Stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new (std::nothrow)
                                  std::thread(&launchOneThreadDispatchV2AndCombineV2_A3A5, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");
    return 0;
}

int main(int argc, char *argv[])
{
    if (!env_dev_num) {
        LOG_PRINT("[WARNING] Please check whether environment variable ENV_DEV_NUM is set correctly.\n");
        return 0;
    }
    int actual_env_dev_num = std::stoi(std::string(env_dev_num));
    if (actual_env_dev_num < DEV_NUM) {
        LOG_PRINT("[INFO] ENV_DEV_NUM = %d is less than %d, currently not supported\n", actual_env_dev_num, DEV_NUM);
        return 0;
    }
    if (IS_TEST_A2) {
        LOG_PRINT("Example on <Atlas A2> will be executed!\n");
        int ret = run_example_on_A2();
    } else if (IS_TEST_A3A5) {
        LOG_PRINT("Example on <Atlas A3> or <Atlas A5> will be executed!\n");
        int ret = run_example_on_A3A5();
    }

    return 0;
}
