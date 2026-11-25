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
 * \file test_all_gather_add_small_shape.h
 * \brief 小 shape（功能正确性）测试共享：shape 常量 + 单轮执行 RunOneRound。
 *        由 BASIC / CONSECUTIVE / TIMEOUT 用例共同 include。
 */

#ifndef TEST_ALL_GATHER_ADD_SMALL_SHAPE_H
#define TEST_ALL_GATHER_ADD_SMALL_SHAPE_H

#include "test_all_gather_add_common.h"

// 小 shape：a=[240,256]，b=[480,256]
const std::vector<int64_t> aShape = {240, 256};
const std::vector<int64_t> bShape = {240 * RANK_DIM, 256};

const long long aShapeSize = GetShapeSize(aShape);
const long long bShapeSize = GetShapeSize(bShape);

// 执行一轮 all_gather_add：填输入 → 两阶段执行 → 同步 → 校验 gatherOut 和 c。
// a/b/out/gatherOut 及其 device 内存跨轮复用；aSize/bSize 为 shape 元素数（支持大 shape），roundName 仅用于日志。
static int RunOneRound(Args& args, const TestData& testData, const char* roundName, aclTensor* a, aclTensor* b,
                       aclTensor* out, aclTensor* gatherOut, void* aDeviceAddr, void* bDeviceAddr, void* outDeviceAddr,
                       void* gatherOutDeviceAddr, int64_t aSize, int64_t bSize)
{
    // 填 host 输入并拷贝到 device（复用已分配内存）
    std::vector<op::fp16_t> aHostData(aSize, 0);
    std::vector<op::fp16_t> bHostData(bSize, 0);
    if (args.rankId == 0) {
        std::copy(testData.rank0_a.begin(), testData.rank0_a.end(), aHostData.begin());
        std::copy(testData.rank0_b.begin(), testData.rank0_b.end(), bHostData.begin());
    } else {
        std::copy(testData.rank1_a.begin(), testData.rank1_a.end(), aHostData.begin());
        std::copy(testData.rank1_b.begin(), testData.rank1_b.end(), bHostData.begin());
    }
    int ret = aclrtMemcpy(aDeviceAddr, aSize * sizeof(op::fp16_t), aHostData.data(), aSize * sizeof(op::fp16_t),
                          ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(bDeviceAddr, bSize * sizeof(op::fp16_t), bHostData.data(), bSize * sizeof(op::fp16_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 两阶段执行 + 同步
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    ret = aclnnAllGatherAddGetWorkspaceSize(a, b, hcomName, RANK_DIM, out, gatherOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] %s GetWorkspaceSize failed. ret = %d \n", roundName, ret);
              return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    ret = aclnnAllGatherAdd(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] %s aclnnAllGatherAdd failed. ret = %d \n", roundName, ret);
              return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] %s sync failed. ret = %d \n", roundName, ret); return ret);

    // 校验 gatherOut
    std::vector<op::fp16_t> gatherOutData(bSize, 0);
    ret = aclrtMemcpy(gatherOutData.data(), bSize * sizeof(gatherOutData[0]), gatherOutDeviceAddr,
                      bSize * sizeof(gatherOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy gatherOut failed. ret = %d \n", ret); return ret);
    ret = CompareVector(gatherOutData, testData.gatherOut);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] %s gatherOut compare failed. ret = %d \n", roundName, ret);
        return ret;
    }

    // 校验 c
    std::vector<op::fp16_t> outputData(bSize, 0);
    ret = aclrtMemcpy(outputData.data(), bSize * sizeof(outputData[0]), outDeviceAddr, bSize * sizeof(outputData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy output failed. ret = %d \n", ret); return ret);
    const std::vector<op::fp16_t>& expectC = (args.rankId == 0) ? testData.rank0_c : testData.rank1_c;
    ret = CompareVector(outputData, expectC);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[ERROR] %s output compare failed. ret = %d \n", roundName, ret);
        return ret;
    }
    LOG_PRINT("[INFO] device_%d %s golden compare successfully.\n", args.rankId, roundName);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

#endif // TEST_ALL_GATHER_ADD_SMALL_SHAPE_H
