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
 * \file test_bsa_select_block_mask.cpp
 * \brief BSASelectBlockMask 算子 ST 测试，调用 aclnn API 在 NPU 上执行并验证 mask 输出
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <stdexcept>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_bsa_select_block_mask.h"

// 测试结果枚举
enum TestResult {
    TEST_PASS,
    TEST_FAIL,
    TEST_SKIP
};

// 测试统计
struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    int skipped = 0;
};

// 测试配置
struct TestConfig {
    int64_t batch;
    int64_t qSeqlen;
    int64_t kvSeqlen;
    int64_t numHeads;
    int64_t numKvHeads; // BSA 当前仅支持 MHA，numKvHeads == numHeads
    int64_t headDim;
    int64_t blockShapeX;
    int64_t blockShapeY;
    float sparsity;
    std::string testName;
};

// 全局变量
static aclrtStream g_stream = nullptr;
static TestStats g_stats;

// 辅助函数：运行单个测试
TestResult RunTest(const TestConfig& config)
{
    std::cout << "\n--- Test: " << config.testName << " ---" << std::endl;
    std::cout << "Config: batch=" << config.batch << ", qSeq=" << config.qSeqlen << ", kvSeq=" << config.kvSeqlen
              << ", heads=" << config.numHeads << ", blockShape=[" << config.blockShapeX << "," << config.blockShapeY
              << "]"
              << ", sparsity=" << config.sparsity << std::endl;

    int64_t totalQTokens = config.batch * config.qSeqlen;
    int64_t totalKvTokens = config.batch * config.kvSeqlen;
    int32_t qBlockNum = (config.qSeqlen + config.blockShapeX - 1) / config.blockShapeX;
    int32_t kvBlockNum = (config.kvSeqlen + config.blockShapeY - 1) / config.blockShapeY;

    // BSA 算子所需的 Tensors
    void *qDev = nullptr, *kDev = nullptr;
    void* maskDev = nullptr;

    // 声明 aclTensors 以便在 catch 中清理
    aclTensor *qTensor = nullptr, *kTensor = nullptr;
    aclTensor* maskTensor = nullptr;

    aclIntArray* blockShapeArray = nullptr;
    aclIntArray* qSeqLenArray = nullptr;
    aclIntArray* kvSeqLenArray = nullptr;
    void* workspace = nullptr;

    try {
        // 1. 分配内存大小
        size_t qSize = totalQTokens * config.numHeads * config.headDim * sizeof(uint16_t); // FP16
        size_t kvSize = totalKvTokens * config.numKvHeads * config.headDim * sizeof(uint16_t);

        aclrtMalloc(&qDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&kDev, kvSize, ACL_MEM_MALLOC_HUGE_FIRST);

        // 给 q/k 赋予基础小值 (FP16 0.1)，防止全0导致 NaN
        std::vector<uint16_t> initData(std::max(qSize, kvSize) / sizeof(uint16_t), 0x2E00); // 0.1 in FP16
        aclrtMemcpy(qDev, qSize, initData.data(), qSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(kDev, kvSize, initData.data(), kvSize, ACL_MEMCPY_HOST_TO_DEVICE);

        // mask 输出初始化为 0
        size_t maskSize = config.batch * config.numHeads * qBlockNum * kvBlockNum * sizeof(int8_t);
        aclrtMalloc(&maskDev, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(maskDev, maskSize, 0, maskSize);

        // 2. 构造 ACL Tensor (使用 BNSD Shape)
        std::vector<int64_t> qShape = {config.batch, config.numHeads, config.qSeqlen, config.headDim};
        std::vector<int64_t> kShape = {config.batch, config.numKvHeads, config.kvSeqlen, config.headDim};
        std::vector<int64_t> maskShape = {config.batch, config.numHeads, qBlockNum, kvBlockNum};

        qTensor = aclCreateTensor(qShape.data(), qShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, qShape.data(),
                                  qShape.size(), qDev);
        kTensor = aclCreateTensor(kShape.data(), kShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, kShape.data(),
                                  kShape.size(), kDev);
        maskTensor = aclCreateTensor(maskShape.data(), maskShape.size(), ACL_INT8, nullptr, 0, ACL_FORMAT_ND,
                                     maskShape.data(), maskShape.size(), maskDev);

        // 3. 构造 aclIntArray 属性参数
        int64_t blockShapeData[2] = {config.blockShapeX, config.blockShapeY};
        blockShapeArray = aclCreateIntArray(blockShapeData, 2);

        std::vector<int64_t> qSeqLenVec(config.batch, config.qSeqlen);
        std::vector<int64_t> kvSeqLenVec(config.batch, config.kvSeqlen);
        qSeqLenArray = aclCreateIntArray(qSeqLenVec.data(), config.batch);
        kvSeqLenArray = aclCreateIntArray(kvSeqLenVec.data(), config.batch);

        // 4. 调用算子
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        // --- 属性参数准备 ---
        char qLayout[] = "BNSD";
        char kvLayout[] = "BNSD";
        int64_t numKvHeads = config.numKvHeads;
        double scale = 1.0 / std::sqrt(static_cast<double>(config.headDim));
        double sparsity = config.sparsity;

        // --- 精确对位的算子调用 ---
        aclnnStatus ret =
            aclnnBSASelectBlockMaskGetWorkspaceSize(qTensor, kTensor,
                                                    blockShapeArray, // blockShapeOptional
                                                    nullptr, // postBlockShapeOptional (当前不支持，必须传 nullptr)
                                                    qSeqLenArray,  // actualSeqLengthsOptional
                                                    kvSeqLenArray, // actualSeqLengthsKvOptional
                                                    nullptr,       // actualBlockLenQueryOptional (完整压缩)
                                                    nullptr,       // actualBlockLenKeyOptional (完整压缩)
                                                    qLayout, kvLayout, numKvHeads, scale, sparsity,
                                                    maskTensor, // blockSparseMaskOut
                                                    &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("GetWorkspaceSize failed with code: " + std::to_string(ret));
        }

        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        std::cout << "DEBUG: workspaceSize = " << workspaceSize << std::endl;
        std::cout << "DEBUG: executor = " << executor << std::endl;

        ret = aclnnBSASelectBlockMask(workspace, workspaceSize, executor, g_stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("BSASelectBlockMask hardware execution failed");
        }

        aclrtSynchronizeStream(g_stream);

        // 5. 将 mask 结果拉回 Host 验证
        std::vector<int8_t> hostMask(maskSize / sizeof(int8_t));
        aclrtMemcpy(hostMask.data(), maskSize, maskDev, maskSize, ACL_MEMCPY_DEVICE_TO_HOST);

        // 统计 mask 中 1 的个数，验证 sparsity 大致正确
        int32_t selectedCount = 0;
        int32_t totalElements = static_cast<int32_t>(hostMask.size());
        for (int32_t i = 0; i < totalElements; ++i) {
            if (hostMask[i] == 1) {
                selectedCount++;
            }
        }
        double actualSparsity = static_cast<double>(selectedCount) / totalElements;
        std::cout << "--> Mask Output: total=" << totalElements << ", selected=" << selectedCount
                  << ", actual_sparsity=" << actualSparsity << std::endl;
        std::cout << "--> Mask Output (First 20 values): ";
        int32_t displayCount = std::min(20, totalElements);
        for (int32_t i = 0; i < displayCount; ++i) {
            std::cout << static_cast<int>(hostMask[i]) << " ";
        }
        std::cout << std::endl;

        // 验证 mask 值只有 0 或 1
        bool validMask = true;
        for (int32_t i = 0; i < totalElements; ++i) {
            if (hostMask[i] != 0 && hostMask[i] != 1) {
                validMask = false;
                break;
            }
        }
        if (!validMask) {
            throw std::runtime_error("Mask contains invalid values (not 0 or 1)");
        }

        // 验证 selectedCount > 0（sparsity > 0 时应选中部分块）
        if (config.sparsity > 0.0 && selectedCount == 0) {
            throw std::runtime_error("Mask is all-zero but sparsity > 0");
        }

        std::cout << "Result: PASS" << std::endl;

        // 6. 成功后的资源清理
        if (workspace)
            aclrtFree(workspace);
        aclDestroyTensor(qTensor);
        aclDestroyTensor(kTensor);
        aclDestroyTensor(maskTensor);
        aclDestroyIntArray(blockShapeArray);
        aclDestroyIntArray(qSeqLenArray);
        aclDestroyIntArray(kvSeqLenArray);

        aclrtFree(qDev);
        aclrtFree(kDev);
        aclrtFree(maskDev);

        return TEST_PASS;

    } catch (const std::exception& e) {
        std::cerr << "Result: FAIL - " << e.what() << std::endl;

        // 异常情况下的资源清理 (防止显存泄漏)
        if (workspace)
            aclrtFree(workspace);
        if (qTensor)
            aclDestroyTensor(qTensor);
        if (kTensor)
            aclDestroyTensor(kTensor);
        if (maskTensor)
            aclDestroyTensor(maskTensor);
        if (blockShapeArray)
            aclDestroyIntArray(blockShapeArray);
        if (qSeqLenArray)
            aclDestroyIntArray(qSeqLenArray);
        if (kvSeqLenArray)
            aclDestroyIntArray(kvSeqLenArray);

        if (qDev)
            aclrtFree(qDev);
        if (kDev)
            aclrtFree(kDev);
        if (maskDev)
            aclrtFree(maskDev);

        return TEST_FAIL;
    }
}

int main()
{
    std::cout << "=======================================================" << std::endl;
    std::cout << "  BSASelectBlockMask ST 测试" << std::endl;
    std::cout << "=======================================================\n" << std::endl;

    // 初始化环境
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtCreateStream(&g_stream);

    // 测试用例列表
    std::vector<TestConfig> testCases = {
        // 基本用例：最小规模 MHA
        {1, 256, 256, 1, 1, 128, 128, 128, 0.5f, "Basic MHA 256x256 sparsity=0.5"},
        // 多头场景
        {1, 256, 256, 4, 4, 128, 128, 128, 0.5f, "MHA N=4 256x256 sparsity=0.5"},
        // 不同 sparsity
        {1, 256, 256, 1, 1, 128, 128, 128, 0.25f, "sparsity=0.25"},
        {1, 256, 256, 1, 1, 128, 128, 128, 0.75f, "sparsity=0.75"},
        // 更大序列
        {1, 1024, 1024, 1, 1, 128, 128, 128, 0.5f, "1024x1024 sparsity=0.5"},
        // 不同 block shape
        {1, 256, 256, 1, 1, 128, 64, 64, 0.5f, "blockShape=64x64"},
        // batch > 1
        {2, 256, 256, 2, 2, 128, 128, 128, 0.5f, "batch=2 N=2"},
    };

    for (const auto& tc : testCases) {
        g_stats.total++;
        TestResult result = RunTest(tc);
        if (result == TEST_PASS) {
            g_stats.passed++;
        } else {
            g_stats.failed++;
        }
    }

    // 输出汇总
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "  Test Summary" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "  Total:   " << g_stats.total << std::endl;
    std::cout << "  Passed:  " << g_stats.passed << std::endl;
    std::cout << "  Failed:  " << g_stats.failed << std::endl;
    std::cout << "  Skipped: " << g_stats.skipped << std::endl;

    // 清理并退出
    aclrtDestroyStream(g_stream);
    aclrtResetDevice(0);
    aclFinalize();

    return (g_stats.failed == 0) ? 0 : 1;
}
