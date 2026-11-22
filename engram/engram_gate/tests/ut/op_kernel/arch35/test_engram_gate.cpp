/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_gate.cpp
 * \brief EngramGate op_kernel UT for the arch35 CPU simulator.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

#include "../../../../op_host/engram_gate_pure_layout_arch35.h"
#include "../../../../op_kernel/arch35/engram_gate_tiling_data.h"

extern "C" __global__ __aicore__ void engram_gate(GM_ADDR hidden_states, GM_ADDR kv, GM_ADDR weight_fused,
                                                  GM_ADDR output, GM_ADDR dot, GM_ADDR gate_score, GM_ADDR rstd_x,
                                                  GM_ADDR rstd_k, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int64_t HEADS = 4;
constexpr uint64_t SIM_UB_BYTES = 48U * 1024U;
constexpr size_t SYS_WORKSPACE = 16U * 1024U * 1024U;
constexpr float EPS = 1e-6F;
constexpr float CLAMP_VALUE = 1e-6F;
constexpr uint32_t CHUNK_FIXED_BYTES = 32U + 3U * 64U * sizeof(float) + 4U * 32U;
constexpr uint32_t CHUNK_STREAM_BYTES_PER_ELEMENT = 22U;

template <typename T>
T* GmAlloc(size_t bytes)
{
    T* ptr = reinterpret_cast<T*>(AscendC::GmAlloc(bytes));
    EXPECT_NE(ptr, nullptr);
    return ptr;
}

void FillBf16(uint8_t* dst, int64_t count, uint16_t bits)
{
    auto* values = reinterpret_cast<uint16_t*>(dst);
    for (int64_t i = 0; i < count; ++i) {
        values[i] = bits;
    }
}

float Bf16ToFloat(uint16_t value)
{
    const uint32_t wide = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &wide, sizeof(result));
    return result;
}

struct Expected {
    float output;
    float dot;
    float gate;
    float rstd;
};

Expected MakeExpected(int64_t dim)
{
    const float d = static_cast<float>(dim);
    const float rstd = 1.0F / std::sqrt(1.0F + EPS);
    const float normalized = d * rstd * rstd / std::sqrt(d);
    const float gate = 1.0F / (1.0F + std::exp(-std::sqrt(normalized)));
    return {1.0F + gate, d, gate, rstd};
}

void FillBaseTiling(EngramGateTilingData& tiling, int64_t tokens, int64_t dim, bool save)
{
    tiling.heads = HEADS;
    tiling.dim = dim;
    tiling.rows = tokens * HEADS;
    tiling.saveStats = save ? 1U : 0U;
    tiling.eps = EPS;
    tiling.clampValue = CLAMP_VALUE;
    tiling.invDim = 1.0F / static_cast<float>(dim);
    tiling.invSqrtDim = 1.0F / std::sqrt(static_cast<float>(dim));
    tiling.usedCores = 1;
}

void CheckStats(const float* dot, const float* gate, const float* rstdX, const float* rstdK, int64_t count,
                const Expected& expected)
{
    for (int64_t i = 0; i < count; ++i) {
        EXPECT_NEAR(dot[i], expected.dot, 2e-5F) << "dot index=" << i;
        EXPECT_NEAR(gate[i], expected.gate, 2e-5F) << "gate index=" << i;
        EXPECT_NEAR(rstdX[i], expected.rstd, 2e-5F) << "rstd_x index=" << i;
        EXPECT_NEAR(rstdK[i], expected.rstd, 2e-5F) << "rstd_k index=" << i;
    }
}

void CheckOutput(const uint8_t* output, int64_t count, const Expected& expected)
{
    const auto* values = reinterpret_cast<const uint16_t*>(output);
    for (int64_t i = 0; i < count; ++i) {
        EXPECT_NEAR(Bf16ToFloat(values[i]), expected.output, 8e-3F) << "output index=" << i;
    }
}

struct DeviceBuffers {
    DeviceBuffers(int64_t xElems, int64_t kvElems, int64_t wElems, int64_t statElems, bool saveStats)
        : x(GmAlloc<uint8_t>(xElems * sizeof(uint16_t))),
          kv(GmAlloc<uint8_t>(kvElems * sizeof(uint16_t))),
          w(GmAlloc<float>(wElems * sizeof(float))),
          out(GmAlloc<uint8_t>(xElems * sizeof(uint16_t))),
          dot(saveStats ? GmAlloc<float>(statElems * sizeof(float)) : nullptr),
          gate(saveStats ? GmAlloc<float>(statElems * sizeof(float)) : nullptr),
          rstdX(saveStats ? GmAlloc<float>(statElems * sizeof(float)) : nullptr),
          rstdK(saveStats ? GmAlloc<float>(statElems * sizeof(float)) : nullptr),
          workspace(GmAlloc<uint8_t>(SYS_WORKSPACE)),
          tiling(GmAlloc<uint8_t>(sizeof(EngramGateTilingData))),
          save(saveStats)
    {}

    ~DeviceBuffers()
    {
        AscendC::GmFree(x);
        AscendC::GmFree(kv);
        AscendC::GmFree(w);
        AscendC::GmFree(out);
        if (save) {
            AscendC::GmFree(dot);
            AscendC::GmFree(gate);
            AscendC::GmFree(rstdX);
            AscendC::GmFree(rstdK);
        }
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }

    void Run(const EngramGateTilingData& tilingData, int64_t xElems, int64_t kvElems, int64_t wElems,
             const std::vector<uint16_t>* xBits = nullptr)
    {
        if (xBits == nullptr) {
            FillBf16(x, xElems, 0x3f80U);
        } else {
            ASSERT_EQ(static_cast<int64_t>(xBits->size()), xElems);
            std::memcpy(x, xBits->data(), xElems * sizeof(uint16_t));
        }
        FillBf16(kv, kvElems, 0x3f80U);
        std::fill(w, w + wElems, 1.0F);
        std::memset(out, 0, xElems * sizeof(uint16_t));
        std::memcpy(tiling, &tilingData, sizeof(tilingData));
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(0);
        ICPU_RUN_KF(engram_gate, tilingData.usedCores, x, kv, reinterpret_cast<uint8_t*>(w), out,
                    reinterpret_cast<uint8_t*>(dot), reinterpret_cast<uint8_t*>(gate),
                    reinterpret_cast<uint8_t*>(rstdX), reinterpret_cast<uint8_t*>(rstdK), workspace, tiling);
    }

    uint8_t* x;
    uint8_t* kv;
    float* w;
    uint8_t* out;
    float* dot;
    float* gate;
    float* rstdX;
    float* rstdK;
    uint8_t* workspace;
    uint8_t* tiling;
    bool save;
};

void SetupPureTiling(int64_t tokens, int64_t dim, bool save, EngramGateTilingData& tiling)
{
    EngramGatePure::Layout layout;
    ASSERT_TRUE(EngramGatePure::ComputeLayout(dim, save, SIM_UB_BYTES, layout, true));
    ASSERT_GE(layout.rowsPerSlot, HEADS);

    FillBaseTiling(tiling, tokens, dim, save);
    tiling.pureSlots = layout.slots;
    tiling.pureRowsPerSlot = layout.rowsPerSlot;
    tiling.pureUbBytes = layout.total;
    tiling.pureXOff = layout.xOff;
    tiling.pureKvOff = layout.kvOff;
    tiling.pureOutOff = layout.outOff;
    tiling.pureWOff = layout.wOff;
    tiling.pureStatsOff = layout.statsOff;
    const int64_t rows = tokens * HEADS;
    tiling.usedCores = static_cast<uint32_t>((rows + layout.rowsPerSlot - 1) / layout.rowsPerSlot);
    if (layout.rowsPerSlot == 1) {
        tiling.usedCores -= tiling.usedCores % HEADS;
    } else if (layout.rowsPerSlot == 2) {
        tiling.usedCores -= tiling.usedCores % 2U;
    }
    ASSERT_GT(tiling.usedCores, 0U);
}

void RunPure(int64_t tokens, int64_t dim, bool save)
{
    const int64_t rows = tokens * HEADS;
    const int64_t xElems = rows * dim;
    const int64_t kvElems = tokens * (HEADS + 1) * dim;
    const int64_t wElems = HEADS * dim;

    EngramGateTilingData tiling{};
    SetupPureTiling(tokens, dim, save, tiling);

    DeviceBuffers buffers(xElems, kvElems, wElems, rows, save);
    buffers.Run(tiling, xElems, kvElems, wElems);

    const auto expected = MakeExpected(dim);
    CheckOutput(buffers.out, xElems, expected);
    if (save) {
        CheckStats(buffers.dot, buffers.gate, buffers.rstdX, buffers.rstdK, rows, expected);
    }
}

void RunChunked(int64_t dim, bool save)
{
    constexpr int64_t tokens = 1;
    const int64_t rows = tokens * HEADS;
    const int64_t xElems = rows * dim;
    const int64_t kvElems = tokens * (HEADS + 1) * dim;
    const int64_t wElems = HEADS * dim;

    EngramGatePure::Layout layout;
    ASSERT_FALSE(EngramGatePure::ComputeLayout(dim, save, SIM_UB_BYTES, layout, true));

    EngramGateTilingData tiling{};
    FillBaseTiling(tiling, tokens, dim, save);
    tiling.chunkedEnabled = 1;
    tiling.ubLayoutBytes = static_cast<uint32_t>(SIM_UB_BYTES);
    tiling.usedCores = HEADS;
    const uint64_t streamCapacity = (SIM_UB_BYTES - CHUNK_FIXED_BYTES) / CHUNK_STREAM_BYTES_PER_ELEMENT;
    tiling.chunk = static_cast<uint32_t>(std::min<uint64_t>(dim, streamCapacity / 64U * 64U));
    tiling.weightResident = 0;
    ASSERT_GT(tiling.chunk, 0U);
    ASSERT_LT(tiling.chunk, dim);

    DeviceBuffers buffers(xElems, kvElems, wElems, rows, save);
    buffers.Run(tiling, xElems, kvElems, wElems);

    const auto expected = MakeExpected(dim);
    CheckOutput(buffers.out, xElems, expected);
    if (save) {
        CheckStats(buffers.dot, buffers.gate, buffers.rstdX, buffers.rstdK, rows, expected);
    }
}

// Opposite-sign cancellation across the four dot channels: merging the
// compensated accumulators must keep the rounding residual of every high-part
// addition, otherwise the small term between two large cancelling terms is
// lost. All-ones data cannot expose this.
void RunPureCancellation()
{
    constexpr int64_t tokens = 1;
    constexpr int64_t dim = 512;
    const int64_t rows = tokens * HEADS;
    const int64_t xElems = rows * dim;
    const int64_t kvElems = tokens * (HEADS + 1) * dim;
    const int64_t wElems = HEADS * dim;

    EngramGateTilingData tiling{};
    SetupPureTiling(tokens, dim, true, tiling);

    // Lane 0 of vectors 0..3 feeds dot channels A..D in order (VEC_LANES = 64).
    // big = 2^26 in bf16, so fl(big + 1) == big and the 1 survives only through
    // the merge residual channel.
    const uint16_t big = 0x4C80U;
    const uint16_t negBig = 0xCC80U;
    const uint16_t one = 0x3F80U;
    const uint16_t pattern[HEADS][4] = {
        {big, one, negBig, 0x0000U}, // pair merge A+B loses the 1
        {big, one, one, negBig},     // both pair merges lose their low term
        {big, 0x0000U, one, negBig}, // final merge A+C loses the 1
        {one, big, negBig, 0x0000U}, // |a| < |b| pair merge
    };
    std::vector<uint16_t> xBits(static_cast<size_t>(xElems), 0x0000U);
    for (int64_t h = 0; h < HEADS; ++h) {
        for (int64_t v = 0; v < 4; ++v) {
            xBits[static_cast<size_t>(h * dim + v * 64)] = pattern[h][v];
        }
    }

    DeviceBuffers buffers(xElems, kvElems, wElems, rows, true);
    buffers.Run(tiling, xElems, kvElems, wElems, &xBits);

    const auto* outValues = reinterpret_cast<const uint16_t*>(buffers.out);
    for (int64_t h = 0; h < HEADS; ++h) {
        double sumSq = 0.0;
        double dot = 0.0;
        for (int64_t v = 0; v < 4; ++v) {
            const double value = Bf16ToFloat(pattern[h][v]);
            sumSq += value * value;
            dot += value;
        }
        const double rstdX = 1.0 / std::sqrt(sumSq / static_cast<double>(dim) + EPS);
        const double rstdK = 1.0 / std::sqrt(1.0 + EPS);
        const double normalized = dot * rstdX * rstdK / std::sqrt(static_cast<double>(dim));
        const double magnitude = std::max(std::fabs(normalized), static_cast<double>(CLAMP_VALUE));
        const double gate = 1.0 / (1.0 + std::exp(-std::sqrt(magnitude)));

        EXPECT_NEAR(buffers.dot[h], dot, 2e-5F) << "dot head=" << h;
        EXPECT_NEAR(buffers.gate[h], gate, 2e-5F) << "gate head=" << h;
        EXPECT_NEAR(buffers.rstdX[h], rstdX, 2e-5F) << "rstd_x head=" << h;
        EXPECT_NEAR(buffers.rstdK[h], rstdK, 2e-5F) << "rstd_k head=" << h;
        for (int64_t i = 0; i < dim; ++i) {
            const double value = Bf16ToFloat(xBits[static_cast<size_t>(h * dim + i)]);
            const double out = value + gate; // v row is all ones
            const double tolerance = std::max(8e-3, std::fabs(out) / 128.0);
            EXPECT_NEAR(Bf16ToFloat(outValues[h * dim + i]), out, tolerance) << "output head=" << h << " i=" << i;
        }
    }
}

} // namespace

TEST(EngramGateKernel, PureSaveTrue)
{
    RunPure(1, 64, true);
}

TEST(EngramGateKernel, PureSaveFalse)
{
    RunPure(2, 512, false);
}

TEST(EngramGateKernel, PureCancellationSaveTrue)
{
    RunPureCancellation();
}

TEST(EngramGateKernel, ChunkedSaveTrue)
{
    RunChunked(4096, true);
}

TEST(EngramGateKernel, EmptyRowsReturn)
{
    EngramGateTilingData tiling{};
    FillBaseTiling(tiling, 0, 64, true);
    auto tilingGm = GmAlloc<uint8_t>(sizeof(tiling));
    std::memcpy(tilingGm, &tiling, sizeof(tiling));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(engram_gate, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                tilingGm);
    AscendC::GmFree(tilingGm);
}
