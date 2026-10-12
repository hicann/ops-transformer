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
#include <cmath>
#include <random>
#include <cassert>
#include "acl/acl.h"
#include "aclnn_rain_fusion_attention.h"

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
    int64_t numKvHeads;
    int64_t headDim;
    int64_t blockShapeX;
    int64_t blockShapeY;
    float sparsity;
    std::string testName;
};

// 全局变量
static aclrtStream g_stream = nullptr;
static TestStats g_stats;

// 辅助函数：生成稀疏模式
void GenerateSparsePattern(int32_t qBlockNum, int32_t kvBlockNum, int32_t numHeads, float sparsity,
                           std::vector<int32_t>& selectIdx, std::vector<int32_t>& selectNumIdx)
{
    std::random_device rd;
    std::mt19937 gen(42); // 固定种子以便复现
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int32_t maxKvBlockNum = kvBlockNum;
    selectIdx.resize(qBlockNum * numHeads * maxKvBlockNum, -1);
    selectNumIdx.resize(qBlockNum * numHeads, 0);

    for (int32_t qb = 0; qb < qBlockNum; ++qb) {
        for (int32_t h = 0; h < numHeads; ++h) {
            std::vector<int32_t> selected;
            for (int32_t kvb = 0; kvb < kvBlockNum; ++kvb) {
                if (dis(gen) < sparsity) {
                    selected.push_back(kvb);
                }
            }
            if (selected.empty())
                selected.push_back(0); // 至少选一个

            int32_t baseIdx = (qb * numHeads + h) * maxKvBlockNum;
            for (size_t i = 0; i < selected.size(); ++i) {
                selectIdx[baseIdx + i] = selected[i];
            }
            selectNumIdx[qb * numHeads + h] = static_cast<int32_t>(selected.size());
        }
    }
}

// 辅助函数：运行单个测试
TestResult RunTest(const TestConfig& config)
{
    std::cout << "\n--- Test: " << config.testName << " ---" << std::endl;
    std::cout << "Config: batch=" << config.batch << ", qSeq=" << config.qSeqlen << ", kvSeq=" << config.kvSeqlen
              << ", heads=" << config.numHeads << ", blockShape=[" << config.blockShapeX << "," << config.blockShapeY
              << "]" << std::endl;

    // 计算维度
    int64_t totalQTokens = config.batch * config.qSeqlen;
    int64_t totalKvTokens = config.batch * config.kvSeqlen;
    int32_t qBlockNum = (config.qSeqlen + config.blockShapeX - 1) / config.blockShapeX;
    int32_t kvBlockNum = (config.kvSeqlen + config.blockShapeY - 1) / config.blockShapeY;

    // 创建tensors（简化版，实际应该初始化数据）
    void *qDev = nullptr, *kDev = nullptr, *vDev = nullptr, *oDev = nullptr;
    void *selIdxDev = nullptr, *selNumDev = nullptr;

    try {
        // 分配内存
        size_t qkvSize = totalQTokens * config.numHeads * config.headDim * 2; // FP16
        size_t kvSize = totalKvTokens * config.numKvHeads * config.headDim * 2;

        aclrtMalloc(&qDev, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&kDev, kvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&vDev, kvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&oDev, qkvSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclrtMemset(qDev, qkvSize, 0, qkvSize);
        aclrtMemset(kDev, kvSize, 0, kvSize);
        aclrtMemset(vDev, kvSize, 0, kvSize);
        aclrtMemset(oDev, qkvSize, 0, qkvSize);

        // 生成稀疏索引
        std::vector<int32_t> selIdxHost, selNumHost;
        GenerateSparsePattern(qBlockNum, kvBlockNum, config.numHeads, config.sparsity, selIdxHost, selNumHost);

        size_t selIdxSize = selIdxHost.size() * sizeof(int32_t);
        size_t selNumSize = selNumHost.size() * sizeof(int32_t);

        aclrtMalloc(&selIdxDev, selIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&selNumDev, selNumSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclrtMemcpy(selIdxDev, selIdxSize, selIdxHost.data(), selIdxSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(selNumDev, selNumSize, selNumHost.data(), selNumSize, ACL_MEMCPY_HOST_TO_DEVICE);

        // 创建tensors
        std::vector<int64_t> qShape = {totalQTokens, config.numHeads, config.headDim};
        std::vector<int64_t> kvShape = {totalKvTokens, config.numKvHeads, config.headDim};
        std::vector<int64_t> selIdxShape = {qBlockNum * config.batch, config.numHeads, kvBlockNum};
        std::vector<int64_t> selNumShape = {qBlockNum * config.batch, config.numHeads};

        aclTensor* qTensor = aclCreateTensor(qShape.data(), qShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             qShape.data(), qShape.size(), qDev);
        aclTensor* kTensor = aclCreateTensor(kvShape.data(), kvShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             kvShape.data(), kvShape.size(), kDev);
        aclTensor* vTensor = aclCreateTensor(kvShape.data(), kvShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             kvShape.data(), kvShape.size(), vDev);
        aclTensor* oTensor = aclCreateTensor(qShape.data(), qShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             qShape.data(), qShape.size(), oDev);
        aclTensor* selIdxTensor = aclCreateTensor(selIdxShape.data(), selIdxShape.size(), ACL_INT32, nullptr, 0,
                                                  ACL_FORMAT_ND, selIdxShape.data(), selIdxShape.size(), selIdxDev);
        aclTensor* selNumTensor = aclCreateTensor(selNumShape.data(), selNumShape.size(), ACL_INT32, nullptr, 0,
                                                  ACL_FORMAT_ND, selNumShape.data(), selNumShape.size(), selNumDev);

        // 设置blockShape
        int64_t blockShapeData[2] = {config.blockShapeX, config.blockShapeY};
        aclIntArray* blockShape = aclCreateIntArray(blockShapeData, 2);

        // 调用算子
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        // 创建非const字符串（接口参数是char*）
        char qLayoutStr[] = "TND";
        char kvLayoutStr[] = "TND";

        aclnnStatus ret = aclnnRainFusionAttentionGetWorkspaceSize(
            qTensor, kTensor, vTensor, nullptr, nullptr, nullptr, nullptr, selIdxTensor, selNumTensor, qLayoutStr,
            kvLayoutStr, config.numKvHeads, 0, 1.0 / std::sqrt(static_cast<double>(config.headDim)),
            1, // innerPrecise (1=fp16 softmax for inference test)
            blockShape, oTensor, nullptr, &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("GetWorkspaceSize failed");
        }

        void* workspace = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        ret = aclnnRainFusionAttention(workspace, workspaceSize, executor, g_stream);
        if (ret != ACL_SUCCESS) {
            if (workspace)
                aclrtFree(workspace);
            throw std::runtime_error("RainFusionAttention failed");
        }

        aclrtSynchronizeStream(g_stream);

        // 清理
        if (workspace)
            aclrtFree(workspace);
        aclDestroyTensor(qTensor);
        aclDestroyTensor(kTensor);
        aclDestroyTensor(vTensor);
        aclDestroyTensor(oTensor);
        aclDestroyTensor(selIdxTensor);
        aclDestroyTensor(selNumTensor);
        aclDestroyIntArray(blockShape);

        aclrtFree(qDev);
        aclrtFree(kDev);
        aclrtFree(vDev);
        aclrtFree(oDev);
        aclrtFree(selIdxDev);
        aclrtFree(selNumDev);

        std::cout << "Result: PASS" << std::endl;
        return TEST_PASS;

    } catch (const std::exception& e) {
        std::cerr << "Result: FAIL - " << e.what() << std::endl;

        // 清理内存
        if (qDev)
            aclrtFree(qDev);
        if (kDev)
            aclrtFree(kDev);
        if (vDev)
            aclrtFree(vDev);
        if (oDev)
            aclrtFree(oDev);
        if (selIdxDev)
            aclrtFree(selIdxDev);
        if (selNumDev)
            aclrtFree(selNumDev);

        return TEST_FAIL;
    }
}

int main()
{
    std::cout << "======================================" << std::endl;
    std::cout << "  RainFusionAttention Test Suite" << std::endl;
    std::cout << "======================================\n" << std::endl;

    // 初始化
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtCreateStream(&g_stream);

    // 测试用例列表
    std::vector<TestConfig> tests = {
        // 测试1: 基础TND格式测试
        {1, 256, 512, 8, 8, 128, 64, 64, 0.3f, "Basic TND Format"},

        // 测试2: 不同blockShape - 小块
        {1, 256, 512, 8, 8, 128, 32, 32, 0.3f, "Small Block Shape (32x32)"},

        // 测试3: 不同blockShape - 大块
        {1, 512, 1024, 8, 8, 128, 128, 128, 0.25f, "Large Block Shape (128x128)"},

        // 测试4: 不同blockShape - 非对称
        {1, 256, 512, 8, 8, 128, 64, 128, 0.3f, "Asymmetric Block Shape (64x128)"},

        // 测试5: 边界条件 - 非对齐序列长度
        {1, 250, 500, 8, 8, 128, 64, 64, 0.3f, "Non-aligned Sequence Length"},

        // 测试6: 边界条件 - 小序列长度
        {1, 64, 128, 8, 8, 128, 32, 32, 0.5f, "Small Sequence Length"},

        // 测试7: 边界条件 - 单头
        {1, 256, 512, 1, 1, 128, 64, 64, 0.3f, "Single Head"},

        // 测试8: 边界条件 - 多batch
        {2, 128, 256, 8, 8, 128, 64, 64, 0.3f, "Multiple Batch (2)"},

        // 测试9: 高稀疏度
        {1, 256, 512, 8, 8, 128, 64, 64, 0.1f, "High Sparsity (10%)"},

        // 测试10: 低稀疏度
        {1, 256, 512, 8, 8, 128, 64, 64, 0.8f, "Low Sparsity (80%)"},
    };

    // 运行所有测试
    for (const auto& test : tests) {
        g_stats.total++;
        TestResult result = RunTest(test);

        switch (result) {
            case TEST_PASS:
                g_stats.passed++;
                break;
            case TEST_FAIL:
                g_stats.failed++;
                break;
            case TEST_SKIP:
                g_stats.skipped++;
                break;
        }
    }

    // 输出测试统计
    std::cout << "\n======================================" << std::endl;
    std::cout << "  Test Summary" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Total tests:   " << g_stats.total << std::endl;
    std::cout << "Passed:        " << g_stats.passed << " (" << (100.0 * g_stats.passed / g_stats.total) << "%)"
              << std::endl;
    std::cout << "Failed:        " << g_stats.failed << std::endl;
    std::cout << "Skipped:       " << g_stats.skipped << std::endl;
    std::cout << "======================================\n" << std::endl;

    // 清理
    aclrtDestroyStream(g_stream);
    aclrtResetDevice(0);
    aclFinalize();

    return (g_stats.failed == 0) ? 0 : 1;
}
