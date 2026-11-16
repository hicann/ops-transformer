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
 * \file test_engram_hash.cpp
 * \brief EngramHash op_kernel UT (arch35 / ascend950).
 *
 * 用 ICPU_RUN_KF 在 CPU 仿真上真正执行 kernel, 与 CPU 参考实现按 int32 位级比对:
 *   hash = int64(ids[t,0]) * mult[l,0];
 *   hash ^= int64(ids[t,k]) * mult[l,k]  (k = 1..last);
 *   r = hash % vocab[l,last-1,tb] (truncmod, 负数补正);  out = int32(r) + offs[l,c]。
 * tiling 字段按 host 侧 engram_hash_tiling_arch35.cpp 的同一策略构造
 * (镜像 tilelang 的 tileTokens 公式 + 块数多于核数时持久核跨块循环)。
 */

#include <cassert>
#include <vector>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void engram_hash(GM_ADDR ngram_token_ids, GM_ADDR multipliers, GM_ADDR vocab_sizes,
                                                  GM_ADDR offsets, GM_ADDR hash_indices, GM_ADDR workspace,
                                                  GM_ADDR tiling);

namespace {
constexpr size_t SYS_WORKSPACE = 16 * 1024 * 1024;
inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

inline size_t RoundUp32(size_t v)
{
    return (v + 31u) & ~static_cast<size_t>(31u);
}

template <typename T>
T* GmAllocWrapper(size_t size)
{
    T* ptr = reinterpret_cast<T*>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr;
}

// 简单确定性 LCG, 避免依赖 <random>。
struct Lcg {
    uint64_t state;
    explicit Lcg(uint64_t seed)
        : state(seed)
    {}
    uint32_t Next()
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>((state >> 33) & 0x7FFFFFFFU);
    }
    int32_t Uniform(int32_t lo, int32_t hi)
    {
        return lo + static_cast<int32_t>(Next() % static_cast<uint32_t>(hi - lo + 1));
    }
};
} // namespace

struct EhParams {
    int64_t numTokens;    // T
    int64_t maxNgramSize; // N
    int64_t layers;       // L
    int64_t tables;       // tables
    int64_t coreNum;      // tiling 侧假设的 AIV 核数 (决定分块与持久核路径)
    std::string desc;
};

class EngramHashKernelTest : public testing::TestWithParam<EhParams> {
protected:
    // 与 host EngramHashTilingArch35::DoOpTiling 逐步一致的 tiling 构造
    // (常量与分支均取自 engram_hash_tiling_arch35.h/.cpp)。
    void ComputeTiling(EngramHashTilingData& t, const EhParams& p, int64_t& usedCore)
    {
        constexpr int64_t kMaxSimtThreads = 1024;
        constexpr int64_t kUbBudgetBytes = 8 * 1024;

        auto AlignUp32 = [](int64_t v) { return (v + 31) & ~static_cast<int64_t>(31); };

        const int64_t T = p.numTokens;
        const int64_t N = p.maxNgramSize;
        const int64_t L = p.layers;
        const int64_t tables = p.tables;
        const int64_t numOutCols = (N - 1) * tables;
        t.numTokens = static_cast<int32_t>(T);
        t.maxNgramSize = static_cast<int32_t>(N);
        t.numNgramLayers = static_cast<int32_t>(L);
        t.numTables = static_cast<int32_t>(tables);
        t.numOutCols = static_cast<int32_t>(numOutCols);
        if (T == 0) {
            t.tileTokens = 0;
            t.colChunk = 0;
            t.numTilesPerLayer = 0;
            t.totalBlocks = 0;
            t.usedCoreNums = 1;
            usedCore = 1;
            return;
        }

        // tileTokens 初值 = max(1, 1024 // C): C <= 1024 时单轮 VF 覆盖整 tile 的一个列块。
        int64_t tileTokens = kMaxSimtThreads / numOutCols;
        tileTokens = std::max(tileTokens, static_cast<int64_t>(1));

        const int64_t coresPerLayerCap = std::max(static_cast<int64_t>(1), p.coreNum / L);
        const int64_t multBytesLocal = N * static_cast<int64_t>(sizeof(int64_t));
        const int64_t multAligned = AlignUp32(multBytesLocal);
        // 布局整体再留 32B 保护边距 (窗口实际大小约 8KB, 不顶满)。
        const int64_t windowBudget = kUbBudgetBytes - 32 - multAligned;
        const int64_t sliceBytesFull = 2 * AlignUp32(numOutCols * static_cast<int64_t>(sizeof(int32_t)));
        const int64_t idElemBytes = N * static_cast<int64_t>(sizeof(int32_t));
        int64_t ttCap = 1;
        int64_t ccCap = numOutCols;
        if (windowBudget - sliceBytesFull >= 32) {
            // 单列块方案: align32(tileTokens*N*4) <= 窗口 - mult - 2*align32(C*4)
            ttCap = std::max((windowBudget - sliceBytesFull - 31) / idElemBytes, static_cast<int64_t>(1));
        } else {
            // 多列块方案: 预算对半, 保证 align32 取整后仍不超窗
            const int64_t idsHalf = windowBudget / 2;
            ttCap = std::max((idsHalf - 31) / idElemBytes, static_cast<int64_t>(1));
            const int64_t sliceHalf = (windowBudget - AlignUp32(ttCap * idElemBytes)) / 2;
            ccCap = std::max((sliceHalf - 31) / static_cast<int64_t>(sizeof(int32_t)), static_cast<int64_t>(1));
        }
        int64_t tilesPerLayerTarget;
        const int64_t perCoreElems = (T * L * numOutCols + p.coreNum - 1) / p.coreNum;
        if (perCoreElems <= kMaxSimtThreads) {
            tilesPerLayerTarget = std::min(coresPerLayerCap, T);
        } else if (T <= coresPerLayerCap * ttCap) {
            // 单轮可容纳: 每层 tiles = coresPerLayer (blocks == coreNum, 每核恰 1 块)
            tilesPerLayerTarget = coresPerLayerCap;
        } else {
            // 多轮: 全局块数 L*minTiles 量化为 coreNum 整数倍, 每核轮次完全均衡
            const int64_t minTiles = (T + ttCap - 1) / ttCap;
            const int64_t rounds = (L * minTiles + p.coreNum - 1) / p.coreNum;
            tilesPerLayerTarget = rounds * p.coreNum / L;
        }
        const int64_t ttFit = (T + tilesPerLayerTarget - 1) / tilesPerLayerTarget;
        if (perCoreElems <= kMaxSimtThreads) {
            // 小/中 T: 允许 tileTokens < 1024//C, 多小块填满核
            tileTokens = std::max(std::min(ttFit, ttCap), static_cast<int64_t>(1));
        } else {
            tileTokens = std::max(tileTokens, std::min(ttFit, ttCap));
        }
        tileTokens = std::min(tileTokens, ttCap);

        const int64_t colChunk = std::min(std::max(ccCap, static_cast<int64_t>(1)), numOutCols);

        t.tileTokens = static_cast<int32_t>(tileTokens);
        t.colChunk = static_cast<int32_t>(colChunk);
        t.numTilesPerLayer = static_cast<int32_t>((T + tileTokens - 1) / tileTokens);
        t.totalBlocks = static_cast<int32_t>(L * t.numTilesPerLayer);
        t.usedCoreNums = static_cast<int32_t>(std::min(L * static_cast<int64_t>(t.numTilesPerLayer), p.coreNum));
        usedCore = t.usedCoreNums;
    }
};

TEST_P(EngramHashKernelTest, RunKernel)
{
    auto p = GetParam();
    std::cout << "[EngramHash] " << p.desc << " T=" << p.numTokens << " N=" << p.maxNgramSize << " L=" << p.layers
              << " tables=" << p.tables << " coreNum=" << p.coreNum << std::endl;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    const int64_t T = p.numTokens;
    const int64_t N = p.maxNgramSize;
    const int64_t L = p.layers;
    const int64_t tables = p.tables;
    const int64_t C = (N - 1) * tables;

    EngramHashTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));
    int64_t usedCore = 1;
    ComputeTiling(tilingData, p, usedCore);
    std::cout << "[EngramHash] tileTokens=" << tilingData.tileTokens
              << " numTilesPerLayer=" << tilingData.numTilesPerLayer << " totalBlocks=" << tilingData.totalBlocks
              << " usedCoreNums=" << tilingData.usedCoreNums << std::endl;

    // host 侧随机数据: ids/mult ∈ [0, 100000), vocab ∈ [100000, 1000000), offsets 为 vocab 前缀和。
    Lcg rng(20260521);
    std::vector<int32_t> idsHost(static_cast<size_t>(T * N));
    for (auto& v : idsHost) {
        v = rng.Uniform(0, 99999);
    }
    std::vector<int64_t> multHost(static_cast<size_t>(L * N));
    for (auto& v : multHost) {
        v = rng.Uniform(0, 99999);
    }
    std::vector<int32_t> vocabHost(static_cast<size_t>(L * (N - 1) * tables));
    for (auto& v : vocabHost) {
        v = rng.Uniform(100000, 999999);
    }
    std::vector<int32_t> offsHost(static_cast<size_t>(L * C), 0);
    for (int64_t l = 0; l < L; ++l) {
        int32_t prefix = 0;
        for (int64_t col = 0; col < C; ++col) {
            offsHost[l * C + col] = prefix;
            prefix += vocabHost[l * C + col];
        }
    }

    // CPU 参考实现 (与 tilelang engram_hash_kernel 位级一致)。
    std::vector<int32_t> expected(static_cast<size_t>(L * T * C), 0);
    for (int64_t l = 0; l < L; ++l) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t col = 0; col < C; ++col) {
                const int64_t last = col / tables + 1;
                const int64_t tb = col % tables;
                int64_t hashValue = static_cast<int64_t>(idsHost[t * N]) * multHost[l * N];
                for (int64_t k = 1; k <= last; ++k) {
                    hashValue ^= static_cast<int64_t>(idsHost[t * N + k]) * multHost[l * N + k];
                }
                const int64_t vocabSize = static_cast<int64_t>(vocabHost[(l * (N - 1) + (last - 1)) * tables + tb]);
                int64_t remainder = hashValue % vocabSize;
                if (remainder < 0) {
                    remainder += vocabSize;
                }
                expected[(l * T + t) * C + col] = static_cast<int32_t>(remainder) + offsHost[l * C + col];
            }
        }
    }

    // GM 分配与数据填充 (T=0 时给最小缓冲)。
    // CPU 仿真对 GM 访问按 32B 块校验: DataCopy 尾块会越界读到分配区外 (如 N=3 时
    // mult 单层 24B、最后一张 tile 的 12B 粒度偏移), 分配统一 32B 对齐并留 32B 余量。
    const size_t idsBytes = RoundUp32(static_cast<size_t>(T * N) * sizeof(int32_t)) + 32;
    const size_t multBytes = RoundUp32(static_cast<size_t>(L * N) * sizeof(int64_t)) + 32;
    const size_t vocabBytes = RoundUp32(static_cast<size_t>(L * (N - 1) * tables) * sizeof(int32_t)) + 32;
    const size_t offsBytes = RoundUp32(static_cast<size_t>(L * C) * sizeof(int32_t)) + 32;
    const size_t outBytes = RoundUp32(static_cast<size_t>(L * T * C) * sizeof(int32_t)) + 32;
    uint8_t* idsGm = GmAllocWrapper<uint8_t>(idsBytes > 32 ? idsBytes : 64);
    uint8_t* multGm = GmAllocWrapper<uint8_t>(multBytes);
    uint8_t* vocabGm = GmAllocWrapper<uint8_t>(vocabBytes);
    uint8_t* offsGm = GmAllocWrapper<uint8_t>(offsBytes);
    uint8_t* yGm = GmAllocWrapper<uint8_t>(outBytes > 32 ? outBytes : 64);
    uint8_t* workspace = GmAllocWrapper<uint8_t>(SYS_WORKSPACE);
    uint8_t* tilingGm = GmAllocWrapper<uint8_t>(sizeof(EngramHashTilingData));

    std::memcpy(multGm, multHost.data(), multBytes);
    std::memcpy(vocabGm, vocabHost.data(), vocabBytes);
    std::memcpy(offsGm, offsHost.data(), offsBytes);
    if (!idsHost.empty()) {
        // T=0 时 idsHost 为空, data() 返回 nullptr, 直接 memcpy(n>0) 会段错误;
        // 空输入 kernel 早退, GM 内容无意义。
        std::memcpy(idsGm, idsHost.data(), idsBytes);
    }
    std::memset(yGm, 0, outBytes > 0 ? outBytes : 32);
    std::memset(workspace, 0, SYS_WORKSPACE);
    std::memcpy(tilingGm, &tilingData, sizeof(EngramHashTilingData));

    // 单一 tiling key 0 (与 host GetTilingKey 一致)。
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(engram_hash, static_cast<uint32_t>(usedCore), idsGm, multGm, vocabGm, offsGm, yGm, workspace, tilingGm);

    // 逐元素位级比对 (T=0 时无元素, 仅验证 kernel 早退不崩溃)。
    const int32_t* y = reinterpret_cast<const int32_t*>(yGm);
    int64_t mismatches = 0;
    int64_t firstBad = -1;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (y[i] != expected[i]) {
            if (firstBad < 0) {
                firstBad = static_cast<int64_t>(i);
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "hash_indices mismatch: " << mismatches << "/" << expected.size()
                             << " elems; first bad idx=" << firstBad;

    AscendC::GmFree(idsGm);
    AscendC::GmFree(multGm);
    AscendC::GmFree(vocabGm);
    AscendC::GmFree(offsGm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
}

INSTANTIATE_TEST_SUITE_P(HashShapes, EngramHashKernelTest,
                         testing::Values(
                             // 模型默认形状: T=4096, N=3, L=2, tables=8 (C=16), 4 核持久核跨块
                             EhParams{4096, 3, 2, 8, 4, "model_shape"},
                             // 小规模: 尾部 tile 不满
                             EhParams{42, 3, 2, 8, 2, "small_tokens"},
                             // 单 token (idealTile=1)
                             EhParams{1, 3, 2, 8, 2, "single_token"},
                             // N=4, tables=8 即 C=24
                             EhParams{2048, 4, 2, 8, 4, "ngram4_c24"},
                             // N=2, L=1, tables=16
                             EhParams{128, 2, 1, 16, 2, "ngram2_l1"},
                             // N=7, L=5, tables=3 (C=18), 多层跨块复用参数
                             EhParams{100, 7, 5, 3, 2, "ngram7_l5"},
                             // 单块单核 (L * tiles <= coreNum, 一核一块)
                             EhParams{64, 3, 2, 8, 8, "one_block_per_core"},
                             // 空输入: T=0, kernel 早退
                             EhParams{0, 3, 2, 8, 4, "empty_tokens"}));

// 结构体字段 sanity: 字段可写、布局可用。
TEST(EngramHashTilingStruct, FieldsWritable)
{
    EngramHashTilingData t;
    std::memset(&t, 0, sizeof(t));
    t.numTokens = 4096;
    t.maxNgramSize = 3;
    t.numNgramLayers = 2;
    t.numTables = 8;
    t.numOutCols = 16;
    t.tileTokens = 147;
    t.numTilesPerLayer = 28;
    t.totalBlocks = 56;
    t.usedCoreNums = 56;
    EXPECT_EQ(t.numTokens, 4096);
    EXPECT_EQ(t.maxNgramSize, 3);
    EXPECT_EQ(t.numNgramLayers, 2);
    EXPECT_EQ(t.numTables, 8);
    EXPECT_EQ(t.numOutCols, 16);
    EXPECT_EQ(t.tileTokens, 147);
    EXPECT_EQ(t.numTilesPerLayer, 28);
    EXPECT_EQ(t.totalBlocks, 56);
    EXPECT_EQ(t.usedCoreNums, 56);
    EXPECT_GT(sizeof(EngramHashTilingData), 0u);
}
