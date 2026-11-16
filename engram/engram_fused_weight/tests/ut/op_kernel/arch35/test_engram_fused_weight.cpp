/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_fused_weight.cpp
 * \brief EngramFusedWeight op_kernel UT (arch35 / ascend950).
 *
 * 用 ICPU_RUN_KF 在 CPU 仿真上真正执行 kernel, 校验 y[i] == float(x1[i]) * float(x2[i])。
 * tiling 字段按 host 侧 engram_fused_weight_tiling_arch35.cpp 的同一策略构造
 * (blockSize 自适应 + 块数多于核数时持久核跨块循环)。
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

extern "C" __global__ __aicore__ void engram_fused_weight(GM_ADDR weight_hidden, GM_ADDR weight_embed,
                                                          GM_ADDR weight_fused, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr size_t SYS_WORKSPACE = 16 * 1024 * 1024;

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return b == 0 ? a : (a + b - 1) / b;
}

template <typename T>
T* GmAllocWrapper(size_t size)
{
    T* ptr = reinterpret_cast<T*>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr;
}

// host 侧 bf16 位模式 -> float
inline float Bf16ToFloat(uint16_t v)
{
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// 把 bf16 GM 用 1.0 (0x3F80) 与 -0.5 (0xBF00) 交替填充
void FillBf16(uint8_t* gm, int64_t numElem)
{
    uint16_t* p = reinterpret_cast<uint16_t*>(gm);
    for (int64_t i = 0; i < numElem; ++i) {
        p[i] = (i & 1) ? 0xBF00 /*-0.5*/ : 0x3F80 /*1.0*/;
    }
}
} // namespace

struct FwParams {
    int64_t m;
    int64_t n;
    int64_t blockDim;
    std::string desc;
};

class EngramFusedWeightKernelTest : public testing::TestWithParam<FwParams> {
protected:
    // 与 host EngramFusedWeightTilingArch35::DoOpTiling 完全一致的切分策略
    // (常量取自 engram_fused_weight_tiling_arch35.h; UB 按 CPU 模拟口径取 256KB)。
    void ComputeTiling(EngramFusedWeightTilingData& t, const FwParams& p, int64_t& usedCore)
    {
        constexpr int64_t kUbSize = 262144;
        constexpr int64_t kUbReservedByte = 256;
        constexpr int64_t kUbBytesPerElement = 16;
        constexpr int64_t kTotalElemSmall = 12288;
        constexpr int64_t kTotalElemMedium = 131072;
        constexpr int64_t kTotalElemLarge = 524288;
        constexpr int64_t kBlockSizeSmall = 512;
        constexpr int64_t kBlockSizeMedium = 2048;
        constexpr int64_t kBlockSizeLarge = 4096;
        constexpr int64_t kBlockSizeMax = 8192;

        int64_t total = p.m * p.n;
        t.totalElements = total;
        int64_t blockSize = (kUbSize - kUbReservedByte) / kUbBytesPerElement;
        if (total == 0) {
            // 空输入: blockDim 固定 1, 全部因子置 0, kernel 按 perCoreElem == 0 直接返回。
            t.blockSize = 0;
            t.perCoreElem = 0;
            t.blockFactor = 0;
            t.tailBlockFactor = 0;
            t.perCoreTailElem = 0;
            t.tailCoreTailElem = 0;
            t.usedCoreNums = 1;
            usedCore = 1;
            return;
        }
        int64_t perCoreElem = (total + p.blockDim - 1) / p.blockDim;
        if (total <= kTotalElemSmall) {
            perCoreElem = std::max(perCoreElem, kBlockSizeSmall);
        } else if (total <= kTotalElemMedium) {
            perCoreElem = std::max(perCoreElem, kBlockSizeMedium);
        } else if (total <= kTotalElemLarge) {
            perCoreElem = std::max(perCoreElem, kBlockSizeLarge);
        } else {
            perCoreElem = std::max(perCoreElem, kBlockSizeMax);
        }
        int64_t usedCoreNums = (total + perCoreElem - 1) / perCoreElem;
        int64_t blockFactor = (perCoreElem + blockSize - 1) / blockSize;
        int64_t tailCoreElem = total - (usedCoreNums - 1) * perCoreElem;
        int64_t tailBlockFactor = (tailCoreElem + blockSize - 1) / blockSize;
        int64_t perCoreTailElem = perCoreElem - (blockFactor - 1) * blockSize;
        int64_t tailCoreTailElem = tailCoreElem - (tailBlockFactor - 1) * blockSize;

        t.blockSize = blockSize;
        t.perCoreElem = perCoreElem;
        t.blockFactor = blockFactor;
        t.tailBlockFactor = tailBlockFactor;
        t.perCoreTailElem = perCoreTailElem;
        t.tailCoreTailElem = tailCoreTailElem;
        t.usedCoreNums = usedCoreNums;
        usedCore = usedCoreNums;
    }
};

TEST_P(EngramFusedWeightKernelTest, RunKernel)
{
    auto p = GetParam();
    std::cout << "[EngramFusedWeight] " << p.desc << " m=" << p.m << " n=" << p.n << " blockDim=" << p.blockDim
              << std::endl;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    int64_t total = p.m * p.n;
    EngramFusedWeightTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));
    int64_t usedCore = 1;
    ComputeTiling(tilingData, p, usedCore);
    std::cout << "[EngramFusedWeight] blockSize=" << tilingData.blockSize << " perCoreElem=" << tilingData.perCoreElem
              << " usedCoreNums=" << tilingData.usedCoreNums << std::endl;

    size_t shapeIn = static_cast<size_t>(total) * sizeof(uint16_t); // bf16
    size_t shapeOut = static_cast<size_t>(total) * sizeof(float);   // fp32
    uint8_t* x1Gm = GmAllocWrapper<uint8_t>(shapeIn);
    uint8_t* x2Gm = GmAllocWrapper<uint8_t>(shapeIn);
    uint8_t* yGm = GmAllocWrapper<uint8_t>(shapeOut);
    uint8_t* workspace = GmAllocWrapper<uint8_t>(SYS_WORKSPACE);
    uint8_t* tilingGm = GmAllocWrapper<uint8_t>(sizeof(EngramFusedWeightTilingData));

    FillBf16(x1Gm, total);
    FillBf16(x2Gm, total);
    std::memset(yGm, 0, shapeOut);
    std::memset(workspace, 0, SYS_WORKSPACE);
    std::memcpy(tilingGm, &tilingData, sizeof(EngramFusedWeightTilingData));

    // host PostTiling 无条件 SetTilingKey(0), kernel 仅 TILING_KEY_IS(0) 一个分支; UT 覆盖该通用路径
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(engram_fused_weight, static_cast<uint32_t>(usedCore), x1Gm, x2Gm, yGm, workspace, tilingGm);

    // 逐元素校验: y[i] == float(x1[i]) * float(x2[i]) (bf16 值的 fp32 乘积可精确表示)
    const uint16_t* x1 = reinterpret_cast<const uint16_t*>(x1Gm);
    const uint16_t* x2 = reinterpret_cast<const uint16_t*>(x2Gm);
    const float* y = reinterpret_cast<const float*>(yGm);
    int64_t mismatches = 0;
    int64_t firstBad = -1;
    for (int64_t i = 0; i < total; ++i) {
        float expect = Bf16ToFloat(x1[i]) * Bf16ToFloat(x2[i]);
        if (y[i] != expect) {
            if (firstBad < 0) {
                firstBad = i;
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0) << "y mismatch: " << mismatches << "/" << total << " elems; first bad idx=" << firstBad;

    AscendC::GmFree(x1Gm);
    AscendC::GmFree(x2Gm);
    AscendC::GmFree(yGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
}

INSTANTIATE_TEST_SUITE_P(FusedWeightShapes, EngramFusedWeightKernelTest,
                         testing::Values(
                             // 模型默认形状: hcMult=4, hiddenSize=5120 (total=20480, 单核 2 次 UB 循环)
                             FwParams{4, 5120, 1, "model_shape"},
                             // 最小合法输入 (total=256, 单核单循环)
                             FwParams{2, 128, 1, "min_total"},
                             // 单行 (total=256)
                             FwParams{1, 256, 1, "single_row"},
                             // 大规模 + 双核 (total=65536, 2 核各 2 次 UB 循环)
                             FwParams{8, 8192, 2, "large_2core"},
                             // 多块多核 (total=32768, 4 核各 1 块)
                             FwParams{64, 512, 4, "many_blocks_4core"},
                             // 每核下限抬高后核数少于 blockDim: 8 核请求 -> usedCoreNums=4
                             FwParams{4, 512, 8, "blocks_fewer_than_cores"}));

// 结构体字段 sanity: 字段可写、布局可用。
TEST(EngramFusedWeightTilingStruct, FieldsWritable)
{
    EngramFusedWeightTilingData t;
    std::memset(&t, 0, sizeof(t));
    t.totalElements = 20480;
    t.blockSize = 16384;
    t.perCoreElem = 20480;
    t.usedCoreNums = 10;
    EXPECT_EQ(t.totalElements, 20480);
    EXPECT_EQ(t.blockSize, 16384);
    EXPECT_EQ(t.perCoreElem, 20480);
    EXPECT_EQ(t.usedCoreNums, 10);
    EXPECT_GT(sizeof(EngramFusedWeightTilingData), 0u);
}
