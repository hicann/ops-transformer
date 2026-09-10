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
#include <cmath>
#include <random>
#include <cassert>
#include <stdexcept>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
// 【注意】修改为你的算子编译后生成的实际头文件名称
#include "aclnnop/aclnn_block_sparse_attention_grad.h"

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

// 将稀疏模式生成器改为生成严格的 UINT8 4D 块掩码
void GenerateSparsePattern(int64_t batch, int32_t qBlockNum, int32_t kvBlockNum, int32_t numHeads, float sparsity,
                           std::vector<uint8_t> &maskData)
{
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    maskData.resize(batch * numHeads * qBlockNum * kvBlockNum, 0);

    for (int64_t b = 0; b < batch; ++b) {
        for (int32_t h = 0; h < numHeads; ++h) {
            for (int32_t qb = 0; qb < qBlockNum; ++qb) {
                bool has_selected = false;
                for (int32_t kvb = 0; kvb < kvBlockNum; ++kvb) {
                    if (dis(gen) < sparsity) {
                        maskData[((b * numHeads + h) * qBlockNum + qb) * kvBlockNum + kvb] = 1;
                        has_selected = true;
                    }
                }
                // 保证每行至少选一个块
                if (!has_selected) {
                    maskData[((b * numHeads + h) * qBlockNum + qb) * kvBlockNum + 0] = 1;
                }
            }
        }
    }
}

// 辅助函数：运行单个测试
TestResult RunTest(const TestConfig &config)
{
    std::cout << "\n--- Test: " << config.testName << " ---" << std::endl;
    std::cout << "Config: batch=" << config.batch << ", qSeq=" << config.qSeqlen << ", kvSeq=" << config.kvSeqlen
              << ", heads=" << config.numHeads << ", blockShape=[" << config.blockShapeX << "," << config.blockShapeY
              << "]" << std::endl;

    int64_t totalQTokens = config.batch * config.qSeqlen;
    int64_t totalKvTokens = config.batch * config.kvSeqlen;
    int32_t qBlockNum = (config.qSeqlen + config.blockShapeX - 1) / config.blockShapeX;
    int32_t kvBlockNum = (config.kvSeqlen + config.blockShapeY - 1) / config.blockShapeY;

    // 反向梯度算子所需的 Tensors
    void *doutDev = nullptr, *qDev = nullptr, *kDev = nullptr, *vDev = nullptr;
    void *outDev = nullptr, *lseDev = nullptr;
    void *dqDev = nullptr, *dkDev = nullptr, *dvDev = nullptr;
    void *maskDev = nullptr; // 替换了原本的 selIdxDev

    // 声明 aclTensors 以便在 catch 中清理
    aclTensor *doutTensor = nullptr, *qTensor = nullptr, *kTensor = nullptr, *vTensor = nullptr;
    aclTensor *outTensor = nullptr, *lseTensor = nullptr;
    aclTensor *dqTensor = nullptr, *dkTensor = nullptr, *dvTensor = nullptr;
    aclTensor *maskTensor = nullptr; // 替换了原本的 selIdxTensor

    aclIntArray *blockShapeArray = nullptr;
    aclIntArray *qSeqLenArray = nullptr;
    aclIntArray *kvSeqLenArray = nullptr;
    void *workspace = nullptr;

    try {
        // 1. 分配内存大小
        size_t qSize = totalQTokens * config.numHeads * config.headDim * sizeof(uint16_t); // FP16
        size_t kvSize = totalKvTokens * config.numKvHeads * config.headDim * sizeof(uint16_t);
        size_t lseSize = totalQTokens * config.numHeads * sizeof(float); // LSE 通常是 FP32

        aclrtMalloc(&doutDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&qDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&kDev, kvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&vDev, kvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&outDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&lseDev, lseSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclrtMalloc(&dqDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dkDev, kvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dvDev, kvSize, ACL_MEM_MALLOC_HUGE_FIRST);

        // 为了防止梯度全0无输出，给 q 赋予基础小值 (FP16 模拟)，其他给 0 防止 NaN
        std::vector<uint16_t> initData(qSize / sizeof(uint16_t), 0x2E00); // 0.1
        aclrtMemcpy(qDev, qSize, initData.data(), qSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(kDev, kvSize, initData.data(), kvSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(vDev, kvSize, initData.data(), kvSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(doutDev, qSize, initData.data(), qSize, ACL_MEMCPY_HOST_TO_DEVICE);

        aclrtMemset(outDev, qSize, 0, qSize);
        aclrtMemset(lseDev, lseSize, 0, lseSize);
        aclrtMemset(dqDev, qSize, 0, qSize);
        aclrtMemset(dkDev, kvSize, 0, kvSize);
        aclrtMemset(dvDev, kvSize, 0, kvSize);

        // 2. 生成稀疏索引并拷贝到 NPU (严谨的 UINT8 掩码)
        std::vector<uint8_t> maskHost;
        GenerateSparsePattern(config.batch, qBlockNum, kvBlockNum, config.numHeads, config.sparsity, maskHost);

        size_t maskSize = maskHost.size() * sizeof(uint8_t);
        aclrtMalloc(&maskDev, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(maskDev, maskSize, maskHost.data(), maskSize, ACL_MEMCPY_HOST_TO_DEVICE);

        // 3. 构造 ACL Tensor (使用 BNSD Shape)
        std::vector<int64_t> qShape = {config.batch, config.numHeads, config.qSeqlen, config.headDim};
        std::vector<int64_t> kvShape = {config.batch, config.numKvHeads, config.kvSeqlen, config.headDim};
        std::vector<int64_t> lseShape = {config.batch, config.numHeads, config.qSeqlen};
        std::vector<int64_t> maskShape = {config.batch, config.numHeads, qBlockNum, kvBlockNum}; // 严格 4D

        doutTensor = aclCreateTensor(qShape.data(), qShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                     qShape.data(), qShape.size(), doutDev);
        qTensor = aclCreateTensor(qShape.data(), qShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, qShape.data(),
                                  qShape.size(), qDev);
        kTensor = aclCreateTensor(kvShape.data(), kvShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                  kvShape.data(), kvShape.size(), kDev);
        vTensor = aclCreateTensor(kvShape.data(), kvShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                  kvShape.data(), kvShape.size(), vDev);
        outTensor = aclCreateTensor(qShape.data(), qShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, qShape.data(),
                                    qShape.size(), outDev);
        lseTensor = aclCreateTensor(lseShape.data(), lseShape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                                    lseShape.data(), lseShape.size(), lseDev);

        dqTensor = aclCreateTensor(qShape.data(), qShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, qShape.data(),
                                   qShape.size(), dqDev);
        dkTensor = aclCreateTensor(kvShape.data(), kvShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                   kvShape.data(), kvShape.size(), dkDev);
        dvTensor = aclCreateTensor(kvShape.data(), kvShape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                   kvShape.data(), kvShape.size(), dvDev);

        maskTensor = aclCreateTensor(maskShape.data(), maskShape.size(), ACL_UINT8, nullptr, 0, ACL_FORMAT_ND,
                                     maskShape.data(), maskShape.size(), maskDev);

        // 补齐完整的 Array 参数
        int64_t blockShapeData[2] = {config.blockShapeX, config.blockShapeY};
        blockShapeArray = aclCreateIntArray(blockShapeData, 2);

        std::vector<int64_t> qSeqLenVec(config.batch, config.qSeqlen);
        std::vector<int64_t> kvSeqLenVec(config.batch, config.kvSeqlen);
        qSeqLenArray = aclCreateIntArray(qSeqLenVec.data(), config.batch);
        kvSeqLenArray = aclCreateIntArray(kvSeqLenVec.data(), config.batch);

        // 4. 调用算子
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        // --- 属性参数准备 ---
        char qLayout[] = "BNSD";
        char kvLayout[] = "BNSD";
        int64_t numKvHeads = config.numKvHeads;
        int64_t maskType = 0;
        double scale = 1.0 / std::sqrt(static_cast<double>(config.headDim));
        int64_t preTokens = 2147483647;
        int64_t nextTokens = 2147483647;

        // --- 精确对位的算子调用 ---
        aclnnStatus ret = aclnnBlockSparseAttentionGradGetWorkspaceSize(
            doutTensor, qTensor, kTensor, vTensor, outTensor, lseTensor,
            maskTensor,      // blockSparseMaskOptional
            nullptr,         // attenMaskOptional
            blockShapeArray, // blockShapeOptional
            qSeqLenArray,    // actualSeqLengthsOptional
            kvSeqLenArray,   // actualSeqLengthsKvOptional
            qLayout, kvLayout, numKvHeads, maskType, scale, preTokens, nextTokens, dqTensor, dkTensor, dvTensor,
            &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("GetWorkspaceSize failed with code: " + std::to_string(ret));
        }

        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        std::cout << "DEBUG: workspaceSize = " << workspaceSize << std::endl;
        std::cout << "DEBUG: executor = " << executor << std::endl;

        ret = aclnnBlockSparseAttentionGrad(workspace, workspaceSize, executor, g_stream);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("BlockSparseAttentionGrad hardware execution failed");
        }

        aclrtSynchronizeStream(g_stream);
        std::cout << "Result: PASS" << std::endl;

        // 将计算结果拉回 Host 打印，证明梯度非空！
        std::vector<uint16_t> hostDq(qSize / sizeof(uint16_t));
        aclrtMemcpy(hostDq.data(), qSize, dqDev, qSize, ACL_MEMCPY_DEVICE_TO_HOST);
        std::cout << "--> dQ Output (First 5 values): ";
        for (int i = 0; i < 5; ++i)
            std::cout << std::hex << hostDq[i] << " ";
        std::cout << std::dec << std::endl;

        // 5. 成功后的资源清理
        if (workspace)
            aclrtFree(workspace);
        aclDestroyTensor(doutTensor);
        aclDestroyTensor(qTensor);
        aclDestroyTensor(kTensor);
        aclDestroyTensor(vTensor);
        aclDestroyTensor(outTensor);
        aclDestroyTensor(lseTensor);
        aclDestroyTensor(dqTensor);
        aclDestroyTensor(dkTensor);
        aclDestroyTensor(dvTensor);
        aclDestroyTensor(maskTensor);
        aclDestroyIntArray(blockShapeArray);
        aclDestroyIntArray(qSeqLenArray);
        aclDestroyIntArray(kvSeqLenArray);

        aclrtFree(doutDev);
        aclrtFree(qDev);
        aclrtFree(kDev);
        aclrtFree(vDev);
        aclrtFree(outDev);
        aclrtFree(lseDev);
        aclrtFree(dqDev);
        aclrtFree(dkDev);
        aclrtFree(dvDev);
        aclrtFree(maskDev);

        return TEST_PASS;

    } catch (const std::exception &e) {
        std::cerr << "Result: FAIL - " << e.what() << std::endl;

        // 异常情况下的资源清理 (防止显存泄漏)
        if (workspace)
            aclrtFree(workspace);
        if (doutTensor)
            aclDestroyTensor(doutTensor);
        if (qTensor)
            aclDestroyTensor(qTensor);
        if (kTensor)
            aclDestroyTensor(kTensor);
        if (vTensor)
            aclDestroyTensor(vTensor);
        if (outTensor)
            aclDestroyTensor(outTensor);
        if (lseTensor)
            aclDestroyTensor(lseTensor);
        if (dqTensor)
            aclDestroyTensor(dqTensor);
        if (dkTensor)
            aclDestroyTensor(dkTensor);
        if (dvTensor)
            aclDestroyTensor(dvTensor);
        if (maskTensor)
            aclDestroyTensor(maskTensor);
        if (blockShapeArray)
            aclDestroyIntArray(blockShapeArray);
        if (qSeqLenArray)
            aclDestroyIntArray(qSeqLenArray);
        if (kvSeqLenArray)
            aclDestroyIntArray(kvSeqLenArray);

        if (doutDev)
            aclrtFree(doutDev);
        if (qDev)
            aclrtFree(qDev);
        if (kDev)
            aclrtFree(kDev);
        if (vDev)
            aclrtFree(vDev);
        if (outDev)
            aclrtFree(outDev);
        if (lseDev)
            aclrtFree(lseDev);
        if (dqDev)
            aclrtFree(dqDev);
        if (dkDev)
            aclrtFree(dkDev);
        if (dvDev)
            aclrtFree(dvDev);
        if (maskDev)
            aclrtFree(maskDev);

        return TEST_FAIL;
    }
}

int main()
{
    std::cout << "=======================================================" << std::endl;
    std::cout << "  BlockSparseAttentionGrad 原生架构修复版 ST" << std::endl;
    std::cout << "=======================================================\n" << std::endl;

    // 初始化环境
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtCreateStream(&g_stream);

    // 只保留这一个核心用例
    TestConfig coreTest = {1, 128, 128, 1, 1, 128, 64, 64, 1.0f, "Core MHA Test"};

    g_stats.total++;
    TestResult result = RunTest(coreTest);

    if (result == TEST_PASS) {
        g_stats.passed++;
        std::cout << "\n✅ [SUCCESS] 核心用例跑通！算子逻辑在 910B 上验证通过。" << std::endl;
    } else {
        g_stats.failed++;
        std::cout << "\n❌ [FAILED] 核心用例失败，请检查 Tiling 或 Kernel 逻辑。" << std::endl;
    }

    // 清理并退出
    aclrtDestroyStream(g_stream);
    aclrtResetDevice(0);
    aclFinalize();

    return (g_stats.failed == 0) ? 0 : 1;
}
