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
 * \file test_chunk_gated_delta_rule.cpp
 * \brief Chunk Gated Delta Rule 算子单元测试
 *
 * 测试类别:
 *   - 基础正确性测试 (有/无Gamma)
 *   - 不同Shape组合测试 (batch, heads, dims)
 *   - 边界测试 (最小shape, 非对齐shape)
 *   - 可选参数测试 (scale, chunkSize, seqlens)
 */
#include <array>
#include <vector>
#include <gtest/gtest.h>
#include <cstring>
#include <iostream>
#include <string>
#include <cstdint>
#include <unistd.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;

struct alignas(8) ChunkGatedDeltaRuleTilingData {
    int64_t aiCoreNum;
    int64_t t;
    int64_t nk;
    int64_t dk;
    int64_t nv;
    int64_t dv;
    int64_t b;
    int64_t hasGamma;
    int64_t chunkSize;
    int64_t maxGroupLength;
    int64_t interWorkspaceSz;
    int64_t stageWorkspaceSz;
    float scale;
};

extern "C" __global__ __aicore__ void chunk_gated_delta_rule(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta,
                                                             GM_ADDR initialState, GM_ADDR seqlens, GM_ADDR gOptional,
                                                             GM_ADDR out, GM_ADDR finalState, GM_ADDR workspaceGM,
                                                             GM_ADDR tilingGM);

template <typename T>
T *GmAllocWrapper(size_t size)
{
    T *ptr = reinterpret_cast<T *>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr;
}

void InitTilingData(ChunkGatedDeltaRuleTilingData *tilingData, int64_t b, int64_t t, int64_t nk, int64_t nv, int64_t dk,
                    int64_t dv, int64_t hasGamma, int64_t chunkSize, int64_t maxGroupLength, float scale = 1.0f)
{
    tilingData->aiCoreNum = 8;
    tilingData->t = t;
    tilingData->nk = nk;
    tilingData->dk = dk;
    tilingData->nv = nv;
    tilingData->dv = dv;
    tilingData->b = b;
    tilingData->hasGamma = hasGamma;
    tilingData->chunkSize = chunkSize;
    tilingData->maxGroupLength = maxGroupLength;
    tilingData->scale = scale;

    // 计算中间workspace大小
    int64_t sizeLow = sizeof(bfloat16_t); // bf16
    int64_t sizeHigh = sizeof(float);     // float
    int64_t s = maxGroupLength;

    tilingData->interWorkspaceSz = 0;
    tilingData->interWorkspaceSz += sizeHigh * nv * s;             // gCumExp
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dk;        // kCumDecay (修正为float)
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dk;        // vInner
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dk;        // qPrime (修正为float)
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dv;        // attnInter
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dv;        // vNew
    tilingData->interWorkspaceSz += sizeHigh * nv * s * dk;        // kg
    tilingData->interWorkspaceSz += sizeHigh * nv * s * chunkSize; // qkt

    // stage workspace大小
    tilingData->stageWorkspaceSz = sizeHigh * chunkSize * (3 * chunkSize + dk + dv);
    tilingData->stageWorkspaceSz *= tilingData->aiCoreNum;
}

void InitInputDataWithSeqlens(uint8_t *queryGm, size_t shapeQ, uint8_t *keyGm, size_t shapeK, uint8_t *valueGm,
                              size_t shapeV, uint8_t *betaGm, size_t shapeBeta, uint8_t *stateGm, size_t shapeState,
                              uint8_t *gammaGm, size_t shapeGamma, uint8_t *seqlensGm,
                              const std::vector<int32_t> &seqlensVec, uint8_t *outGm, size_t shapeOut,
                              uint8_t *finalState, size_t shapeFs)
{
    memset_s(queryGm, shapeQ, 0, shapeQ);
    memset_s(keyGm, shapeK, 0, shapeK);
    memset_s(valueGm, shapeV, 0, shapeV);
    memset_s(betaGm, shapeBeta, 0, shapeBeta);
    memset_s(stateGm, shapeState, 0, shapeState);
    memset_s(gammaGm, shapeGamma, 0, shapeGamma);
    memset_s(outGm, shapeOut, 0, shapeOut);
    memset_s(finalState, shapeFs, 0, shapeFs);

    int32_t *seqlens = reinterpret_cast<int32_t *>(seqlensGm);
    for (size_t i = 0; i < seqlensVec.size(); ++i) {
        seqlens[i] = seqlensVec[i];
    }
}

void InitInputData(uint8_t *queryGm, size_t shapeQ, uint8_t *keyGm, size_t shapeK, uint8_t *valueGm, size_t shapeV,
                   uint8_t *betaGm, size_t shapeBeta, uint8_t *stateGm, size_t shapeState, uint8_t *gammaGm,
                   size_t shapeGamma, uint8_t *seqlensGm, size_t b, uint8_t *outGm, size_t shapeOut,
                   uint8_t *finalState, size_t shapeFs)
{
    std::vector<int32_t> seqlensVec(b, 64);
    InitInputDataWithSeqlens(queryGm, shapeQ, keyGm, shapeK, valueGm, shapeV, betaGm, shapeBeta, stateGm, shapeState,
                             gammaGm, shapeGamma, seqlensGm, seqlensVec, outGm, shapeOut, finalState, shapeFs);
}

// ============================================================================
// 测试参数结构体
// ============================================================================

struct CGDRTestParams {
    int64_t b;                    // batch size
    int64_t t;                    // total tokens
    int64_t nk;                   // key head num
    int64_t nv;                   // value head num
    int64_t dk;                   // key dimension
    int64_t dv;                   // value dimension
    int64_t hasGamma;             // 是否有gamma参数
    int64_t chunkSize;            // chunk size
    int64_t maxGroupLength;       // max group length
    float scale;                  // scale因子
    std::vector<int32_t> seqlens; // 每个batch的序列长度
    std::string description;      // 测试描述

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "b=" << b << ", t=" << t << ", nk=" << nk << ", nv=" << nv << ", dk=" << dk << ", dv=" << dv
           << ", hasGamma=" << hasGamma << ", chunkSize=" << chunkSize << ", maxGroupLength=" << maxGroupLength
           << ", scale=" << scale;
        return ss.str();
    }
};

// ============================================================================
// 通用测试类 - 支持参数化测试
// ============================================================================

class ChunkGatedDeltaRuleTest : public testing::TestWithParam<CGDRTestParams> {
protected:
    // 默认参数,将被GetParam()覆盖
    int64_t b = 2;
    int64_t nk = 4;
    int64_t nv = 4;
    int64_t dk = 64;
    int64_t dv = 64;
    int64_t t = 128;
    int64_t chunkSize = 64;
    int64_t maxGroupLength = 128;
    float scale = 1.0f;
    int64_t hasGamma = 1;
    std::vector<int32_t> seqlens = {64, 64};

    // Shape sizes (将在SetUp中计算)
    size_t shapeQuery = 0;
    size_t shapeKey = 0;
    size_t shapeValue = 0;
    size_t shapeBeta = 0;
    size_t shapeState = 0;
    size_t shapeFinalState = 0;
    size_t shapeGamma = 0;
    size_t shapeSeqlens = 0;
    size_t shapeOut = 0;
    size_t tilingSize = sizeof(ChunkGatedDeltaRuleTilingData);

    // GM pointers
    uint8_t *queryGm = nullptr;
    uint8_t *keyGm = nullptr;
    uint8_t *valueGm = nullptr;
    uint8_t *betaGm = nullptr;
    uint8_t *stateGm = nullptr;
    uint8_t *finalStateGm = nullptr;
    uint8_t *gammaGm = nullptr;
    uint8_t *seqlensGm = nullptr;
    uint8_t *outGm = nullptr;
    uint8_t *workspace = nullptr;
    uint8_t *tiling = nullptr;

    void SetUp() override
    {
        auto params = GetParam();
        b = params.b;
        t = params.t;
        nk = params.nk;
        nv = params.nv;
        dk = params.dk;
        dv = params.dv;
        hasGamma = params.hasGamma;
        chunkSize = params.chunkSize;
        maxGroupLength = params.maxGroupLength;
        scale = params.scale;
        seqlens = params.seqlens;

        // 计算shape大小
        shapeQuery = t * nk * dk * sizeof(bfloat16_t);
        shapeKey = t * nk * dk * sizeof(bfloat16_t);
        shapeValue = t * nv * dv * sizeof(bfloat16_t);
        shapeBeta = t * nv * sizeof(bfloat16_t);
        shapeState = b * nv * dv * dk * sizeof(bfloat16_t);
        shapeFinalState = b * nv * dv * dk * sizeof(bfloat16_t);
        shapeGamma = t * nv * sizeof(float);
        shapeSeqlens = b * sizeof(int32_t);
        shapeOut = t * nv * dv * sizeof(bfloat16_t);

        AscendC::SetKernelMode(KernelMode::AIV_MODE);

        // 分配内存
        queryGm = GmAllocWrapper<uint8_t>(shapeQuery);
        keyGm = GmAllocWrapper<uint8_t>(shapeKey);
        valueGm = GmAllocWrapper<uint8_t>(shapeValue);
        betaGm = GmAllocWrapper<uint8_t>(shapeBeta);
        stateGm = GmAllocWrapper<uint8_t>(shapeState);
        finalStateGm = GmAllocWrapper<uint8_t>(shapeFinalState);
        gammaGm = GmAllocWrapper<uint8_t>(shapeGamma);
        seqlensGm = GmAllocWrapper<uint8_t>(shapeSeqlens);
        outGm = GmAllocWrapper<uint8_t>(shapeOut);

        // 计算workspace大小
        ChunkGatedDeltaRuleTilingData tempTiling;
        InitTilingData(&tempTiling, b, t, nk, nv, dk, dv, hasGamma, chunkSize, maxGroupLength, scale);
        size_t workspaceSize = 16 * 1024 * 1024; // 16MB system workspace
        workspaceSize += tempTiling.interWorkspaceSz;
        workspaceSize += tempTiling.stageWorkspaceSz;
        workspace = GmAllocWrapper<uint8_t>(workspaceSize);
        tiling = GmAllocWrapper<uint8_t>(tilingSize);

        // 初始化输入数据
        InitInputDataWithSeqlens(queryGm, shapeQuery, keyGm, shapeKey, valueGm, shapeValue, betaGm, shapeBeta, stateGm,
                                 shapeState, gammaGm, shapeGamma, seqlensGm, seqlens, outGm, shapeOut, finalStateGm,
                                 shapeFinalState);

        // 初始化tiling数据
        ChunkGatedDeltaRuleTilingData *tilingData = reinterpret_cast<ChunkGatedDeltaRuleTilingData *>(tiling);
        InitTilingData(tilingData, b, t, nk, nv, dk, dv, hasGamma, chunkSize, maxGroupLength, scale);
    }

    void TearDown() override
    {
        AscendC::GmFree(queryGm);
        AscendC::GmFree(keyGm);
        AscendC::GmFree(valueGm);
        AscendC::GmFree(betaGm);
        AscendC::GmFree(stateGm);
        AscendC::GmFree(finalStateGm);
        AscendC::GmFree(gammaGm);
        AscendC::GmFree(seqlensGm);
        AscendC::GmFree(outGm);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

// ============================================================================
// 测试用例定义
// ============================================================================

// 1. 基础正确性测试 - 有/无Gamma
INSTANTIATE_TEST_SUITE_P(BasicCorrectness, ChunkGatedDeltaRuleTest,
                         testing::Values(
                             // 基础测试: 有Gamma
                             CGDRTestParams{2, 128, 4, 4, 64, 64, 1, 64, 128, 1.0f, {64, 64}, "basic_with_gamma"},
                             // 基础测试: 无Gamma
                             CGDRTestParams{2, 128, 4, 4, 64, 64, 0, 64, 128, 1.0f, {64, 64}, "basic_without_gamma"}));

// 2. 不同Shape组合测试
INSTANTIATE_TEST_SUITE_P(VariousShapes, ChunkGatedDeltaRuleTest,
                         testing::Values(
                             // 单batch
                             CGDRTestParams{1, 64, 4, 4, 64, 64, 1, 64, 64, 1.0f, {64}, "single_batch"},
                             // 多batch
                             CGDRTestParams{4, 256, 4, 4, 64, 64, 1, 64, 128, 1.0f, {64, 64, 64, 64}, "multi_batch_4"},
                             // 不同head数 (nk != nv)
                             CGDRTestParams{2, 128, 8, 4, 64, 64, 1, 64, 128, 1.0f, {64, 64}, "different_heads_gqa"},
                             // 更大head数
                             CGDRTestParams{2, 128, 16, 16, 64, 64, 1, 64, 128, 1.0f, {64, 64}, "more_heads"},
                             // 不同维度 dk=dv=128
                             CGDRTestParams{2, 128, 4, 4, 128, 128, 1, 64, 128, 1.0f, {64, 64}, "larger_dims_128"},
                             // 不同维度 dk=dv=32
                             CGDRTestParams{2, 128, 4, 4, 32, 32, 1, 64, 128, 1.0f, {64, 64}, "smaller_dims_32"}));

// 3. 边界测试
INSTANTIATE_TEST_SUITE_P(BoundaryTests, ChunkGatedDeltaRuleTest,
                         testing::Values(
                             // 最小shape (单batch, 单head, 最小chunk)
                             CGDRTestParams{1, 64, 1, 1, 64, 64, 1, 64, 64, 1.0f, {64}, "minimal_shape"},
                             // 序列长度等于chunk大小
                             CGDRTestParams{2, 64, 4, 4, 64, 64, 1, 64, 64, 1.0f, {64, 64}, "seqlen_equals_chunk"},
                             // 非对齐维度 (dk=48, dv=48)
                             CGDRTestParams{2, 128, 4, 4, 48, 48, 1, 64, 128, 1.0f, {64, 64}, "non_aligned_dims"},
                             // 非对齐维度 (dk=96, dv=96)
                             CGDRTestParams{2, 128, 4, 4, 96, 96, 1, 64, 128, 1.0f, {64, 64}, "non_aligned_dims_96"}));

// 4. Seqlens变化测试 - 不同batch序列长度
INSTANTIATE_TEST_SUITE_P(VariableSeqlens, ChunkGatedDeltaRuleTest,
                         testing::Values(
                             // 不同batch有不同seqlen
                             CGDRTestParams{2, 128, 4, 4, 64, 64, 1, 64, 128, 1.0f, {32, 96}, "different_seqlens"},
                             // 尾部chunk非对齐
                             CGDRTestParams{2, 128, 4, 4, 64, 64, 1, 64, 128, 1.0f, {100, 100}, "tail_chunk_unaligned"},
                             // 长序列测试
                             CGDRTestParams{2, 256, 4, 4, 64, 64, 1, 64, 128, 1.0f, {128, 200}, "long_sequence"},
                             // 短序列测试
                             CGDRTestParams{2, 64, 4, 4, 64, 64, 1, 64, 64, 1.0f, {32, 48}, "short_sequence"}));

// 5. Chunk Size变化测试
INSTANTIATE_TEST_SUITE_P(ChunkSizeVariations, ChunkGatedDeltaRuleTest,
                         testing::Values(
                             // chunkSize=32
                             CGDRTestParams{2, 128, 4, 4, 64, 64, 1, 32, 64, 1.0f, {64, 64}, "chunk_32"},
                             // chunkSize=128
                             CGDRTestParams{2, 256, 4, 4, 64, 64, 1, 128, 128, 1.0f, {128, 128}, "chunk_128"},
                             // 多chunk group (maxGroupLength < seqlen)
                             CGDRTestParams{2, 256, 4, 4, 64, 64, 1, 64, 128, 1.0f, {200, 200}, "multi_chunk_group"}));

// 6. Scale参数测试
INSTANTIATE_TEST_SUITE_P(ScaleVariations, ChunkGatedDeltaRuleTest,
                         testing::Values(
                             // scale=0.5
                             CGDRTestParams{2, 128, 4, 4, 64, 64, 1, 64, 128, 0.5f, {64, 64}, "scale_0_5"},
                             // scale=0.125 (1/sqrt(64))
                             CGDRTestParams{2, 128, 4, 4, 64, 64, 1, 64, 128, 0.125f, {64, 64}, "scale_inv_sqrt"},
                             // scale=2.0
                             CGDRTestParams{2, 128, 4, 4, 64, 64, 1, 64, 128, 2.0f, {64, 64}, "scale_2_0"}));

// 7. 组合边界测试
INSTANTIATE_TEST_SUITE_P(CombinedEdgeCases, ChunkGatedDeltaRuleTest,
                         testing::Values(
                             // 无Gamma + 不同seqlens
                             CGDRTestParams{2, 128, 4, 4, 64, 64, 0, 64, 128, 1.0f, {32, 96}, "no_gamma_diff_seqlens"},
                             // 无Gamma + 非对齐维度
                             CGDRTestParams{2, 128, 4, 4, 48, 48, 0, 64, 128, 1.0f, {64, 64}, "no_gamma_non_aligned"},
                             // 有Gamma + 小chunk + 长序列
                             CGDRTestParams{
                                 2, 256, 4, 4, 64, 64, 1, 32, 64, 1.0f, {200, 200}, "gamma_small_chunk_long"},
                             // GQA模式 (nk=8, nv=2)
                             CGDRTestParams{2, 128, 8, 2, 64, 64, 1, 64, 128, 1.0f, {64, 64}, "gqa_mode"}));

// ============================================================================
// 测试执行
// ============================================================================

TEST_P(ChunkGatedDeltaRuleTest, RunTest)
{
    auto params = GetParam();
    std::cout << "Test: " << params.description << std::endl;
    std::cout << "Config: " << params.ToString() << std::endl;
    std::cout << "Seqlens: [";
    for (size_t i = 0; i < params.seqlens.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << params.seqlens[i];
    }
    std::cout << "]" << std::endl;

    uint32_t blockDim = 8;
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(chunk_gated_delta_rule, blockDim, queryGm, keyGm, valueGm, betaGm, stateGm, seqlensGm,
                (hasGamma ? gammaGm : nullptr), outGm, finalStateGm, workspace, tiling);
}
