/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <cstdint>
#include <cstddef>

struct HcForwardMulSumTilingData {
    int64_t B;
    int64_t S;
    uint32_t M;
    uint32_t D;
    uint32_t R;
    uint32_t H;
    uint32_t C;
    uint32_t usedAic;
    uint32_t mTask;
    uint32_t taskNum;
    uint32_t L1M0;
    uint32_t L1N0;
    uint32_t L1M1;
    uint32_t L1N1;
    uint32_t szT;
    uint32_t useCombine;
    uint32_t hasGamma;
    uint32_t tilingKey;
    float eps;
    uint32_t smallN;
    uint32_t splitkFactor;
    uint32_t nConcat;
    uint32_t nWs;
    uint32_t nWsGate;
    uint32_t needBackward;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
};
static_assert(offsetof(HcForwardMulSumTilingData, B) == 0);
static_assert(offsetof(HcForwardMulSumTilingData, S) == 8);
static_assert(offsetof(HcForwardMulSumTilingData, M) == 16);
static_assert(offsetof(HcForwardMulSumTilingData, pad3) == 120);
static_assert(sizeof(HcForwardMulSumTilingData) == 128);

inline uint32_t HcAlignUp(uint32_t x, uint32_t align)
{
    if (align == 0) {
        return x;
    }
    return (x + align - 1u) / align * align;
}

inline uint32_t HcNConcat(uint32_t r, uint32_t channels, uint32_t useCombine)
{
    return useCombine != 0 ? (r + channels) : r;
}

inline uint32_t HcNWs(uint32_t nConcat)
{
    return HcAlignUp(nConcat, 16u);
}

constexpr uint32_t HC_SMALL_M_THRESH = 896;
constexpr uint32_t HC_MTASK_LARGE = 128;
constexpr uint32_t HC_MTASK_SMALL = 32;
constexpr uint32_t HC_L1M_SMALL = 32;
constexpr uint32_t HC_C0_L1M_SMALL = 16;
constexpr uint32_t HC_C0_L1N0_SMALL = 24;
constexpr uint32_t HC_L1N0_SMALL = HC_C0_L1N0_SMALL;
constexpr uint32_t HC_L1N0_LARGE = 320;
constexpr uint32_t HC_C1_L1N = 256;
constexpr uint32_t HC_L1N1 = HC_C1_L1N;
constexpr uint32_t HC_C0_L1K_SMALL_FP32 = 512;
constexpr uint32_t HC_C0_L1K_SMALL_F16 = 512;
constexpr uint32_t HC_C0_L0K_SMALL = 128;
constexpr uint32_t HC_L1K0_SMALL_FP32 = HC_C0_L1K_SMALL_FP32;
constexpr uint32_t HC_L1K0_SMALL_F16 = HC_C0_L1K_SMALL_F16;
constexpr uint32_t HC_L0K0_SMALL = HC_C0_L0K_SMALL;
constexpr uint32_t HC_SPLITK_FACTOR_SMALL = 5;
constexpr uint32_t HC_C0_USE_BASIC_SPLITK = 1;
constexpr uint32_t HC_C0_CROSS_SPLITK = 0;
constexpr uint32_t HC_C0_MM_LAYOUT = 0;
constexpr uint32_t HC_C0_L0C_STAGES = 2;
constexpr uint32_t HC_C1_L1K_F16 = 160;
constexpr uint32_t HC_C1_L0C_STAGES = 1;
static_assert(HC_C0_L1M_SMALL > 0 && (HC_MTASK_SMALL % HC_C0_L1M_SMALL == 0), "mTask must be a multiple of C0 L1.M");

// Workspace ping-pong storage uses two alternating slots.
constexpr int64_t HC_PING_PONG_SLOT_COUNT = 2;
constexpr uint32_t HC_TILING_DTYPE_MASK = 0x3u;
constexpr uint32_t HC_TILING_FLAG_MASK = 0x1u;
constexpr uint32_t HC_TILING_COMBINE_SHIFT = 2;
constexpr uint32_t HC_TILING_GAMMA_SHIFT = 3;
constexpr uint32_t HC_TILING_SWIZZLE_SHIFT = 4;
constexpr uint32_t HC_TILING_SMALL_N_SHIFT = 5;

inline uint32_t HcEncodeTilingKey(uint32_t dtypeId, uint32_t useCombine, uint32_t hasGamma, uint32_t c1Swizzle,
                                  uint32_t smallN = 0)
{
    return (dtypeId & HC_TILING_DTYPE_MASK) | ((useCombine & HC_TILING_FLAG_MASK) << HC_TILING_COMBINE_SHIFT) |
           ((hasGamma & HC_TILING_FLAG_MASK) << HC_TILING_GAMMA_SHIFT) |
           ((c1Swizzle & HC_TILING_FLAG_MASK) << HC_TILING_SWIZZLE_SHIFT) |
           ((smallN & HC_TILING_FLAG_MASK) << HC_TILING_SMALL_N_SHIFT);
}

inline uint32_t HcC0L1K(uint32_t szT)
{
    return szT == 4u ? HC_L1K0_SMALL_FP32 : HC_L1K0_SMALL_F16;
}

inline uint32_t HcGetSplitkFactor(uint32_t mTile, uint32_t r, uint32_t d, uint32_t l1M, uint32_t l1N, uint32_t l1K,
                                  uint32_t aicNum)
{
    (void)mTile;
    (void)r;
    (void)l1M;
    (void)l1N;
    (void)aicNum;
    if (HC_C0_USE_BASIC_SPLITK != 0 && HC_C0_CROSS_SPLITK == 0) {
        return 1;
    }
    uint32_t kTileNum = (d + l1K - 1) / l1K;
    uint32_t splitkFactor = HC_SPLITK_FACTOR_SMALL;
    if (kTileNum < splitkFactor) {
        splitkFactor = kTileNum;
    }
    if (splitkFactor == 0) {
        splitkFactor = 1;
    }
    return splitkFactor;
}

inline uint32_t HcTilingNWs(const HcForwardMulSumTilingData& t)
{
    if (t.nWs != 0) {
        return t.nWs;
    }
    uint32_t nConcat = t.nConcat != 0 ? t.nConcat : t.R;
    return HcNWs(nConcat);
}

inline uint32_t HcC0GateCols(const HcForwardMulSumTilingData& t)
{
    return (t.useCombine != 0 && t.C > 0) ? t.C : 0u;
}

// 每个 C0 slot：down 连续 [rows, R]，inject 紧跟其后，连续 [rows, C]。
inline int64_t HcC0SlotBytes(const HcForwardMulSumTilingData& t)
{
    const uint32_t rows = t.smallN != 0 ? t.mTask : t.L1M0;
    return static_cast<int64_t>(rows) * (static_cast<int64_t>(t.R) + HcC0GateCols(t)) *
           static_cast<int64_t>(sizeof(float));
}

inline int64_t HcC0PingBytes(const HcForwardMulSumTilingData& t)
{
    const int64_t slot = HcC0SlotBytes(t);
    if (t.smallN != 0) {
        return HC_PING_PONG_SLOT_COUNT * slot;
    }
    return static_cast<int64_t>(t.usedAic) * slot * HC_PING_PONG_SLOT_COUNT;
}

inline int64_t HcSplitkBytes(const HcForwardMulSumTilingData& t)
{
    if (t.smallN == 0) {
        return 0;
    }
    uint32_t factor = t.splitkFactor == 0 ? 1u : t.splitkFactor;
    if (factor <= 1u) {
        return 0;
    }
    return HC_PING_PONG_SLOT_COUNT * t.mTask * HcTilingNWs(t) * static_cast<int64_t>(sizeof(float)) * factor;
}

inline int64_t HcWorkspaceBytes(const HcForwardMulSumTilingData& t, bool nativeCubeOutput = false)
{
    const int64_t szF = static_cast<int64_t>(sizeof(float));
    // fused 大路径按核做环形槽：xn 3 拍、down 2 拍，与 M 无关；其余路径按全量 M。
    const bool fusedRing = t.smallN == 0;
    const int64_t s0 = fusedRing ? static_cast<int64_t>(t.usedAic) * 3u * t.mTask * t.D * t.szT :
                                   static_cast<int64_t>(t.M) * t.D * t.szT;
    const int64_t s1 = fusedRing ? static_cast<int64_t>(t.usedAic) * 2u * t.mTask * t.R * t.szT :
                                   static_cast<int64_t>(t.M) * t.R * t.szT;
    const int64_t s2 = HcC0PingBytes(t);
    const int64_t sSplitk = HcSplitkBytes(t);
    const int64_t s3 =
        nativeCubeOutput && t.needBackward ? 0u : static_cast<int64_t>(t.M) * t.D * (nativeCubeOutput ? t.szT : szF);
    return s0 + s1 + s2 + sSplitk + s3;
}

inline uint32_t HcNWsDown(uint32_t r)
{
    return HcAlignUp(r, 32u);
}
inline uint32_t HcNWsGate(uint32_t c, uint32_t inject)
{
    return inject ? HcAlignUp(c, 32u) : 0;
}

inline uint32_t HcTilingNWsDown(const HcForwardMulSumTilingData& t)
{
    if (t.nWs != 0) {
        return t.nWs;
    }
    return HcNWsDown(t.R);
}

inline uint32_t HcTilingNWsGate(const HcForwardMulSumTilingData& t)
{
    if (t.nWsGate != 0) {
        return t.nWsGate;
    }
    return HcNWsGate(t.C, t.useCombine);
}

inline uint32_t HcA5TilingNWs(const HcForwardMulSumTilingData& t)
{
    return HcTilingNWsDown(t) + HcTilingNWsGate(t);
}

inline int64_t HcA5C0PingBytes(const HcForwardMulSumTilingData& t)
{
    const int64_t szF = t.szT;
    const uint32_t nAll = HcA5TilingNWs(t);
    if (t.smallN != 0) {
        return HC_PING_PONG_SLOT_COUNT * t.mTask * nAll * szF;
    }
    return static_cast<int64_t>(t.usedAic) * t.L1M0 * nAll * szF * HC_PING_PONG_SLOT_COUNT;
}

inline int64_t HcA5SplitkBytes(const HcForwardMulSumTilingData& t)
{
    if (t.smallN == 0) {
        return 0;
    }
    uint32_t factor = t.splitkFactor == 0 ? 1u : t.splitkFactor;
    if (factor <= 1u) {
        return 0;
    }
    return HC_PING_PONG_SLOT_COUNT * t.mTask * HcA5TilingNWs(t) * static_cast<int64_t>(sizeof(float)) * factor;
}

inline int64_t HcA5WorkspaceBytes(const HcForwardMulSumTilingData& t)
{
    // fused 大路径按核私有环形槽：xn 3 拍（V0 写 → C0 读 → V2 读，同核串行依赖）、
    // down 2 拍（V1 写 → C1 读），与 M 无关；其余路径按全量 M。
    const bool fusedRing = t.smallN == 0;
    const int64_t s0 = fusedRing ? static_cast<int64_t>(t.usedAic) * 3u * t.mTask * t.D * t.szT :
                                   static_cast<int64_t>(t.M) * t.D * t.szT;
    const int64_t s1 = fusedRing ? static_cast<int64_t>(t.usedAic) * 2u * t.mTask * t.R * t.szT :
                                   static_cast<int64_t>(t.M) * t.R * t.szT;
    const int64_t s2 = HcA5C0PingBytes(t);
    const int64_t sSplitk = HcA5SplitkBytes(t);
    const int64_t s3 = t.needBackward ? 0u : static_cast<int64_t>(t.M) * t.D * t.szT;
    return s0 + s1 + s2 + sSplitk + s3;
}

// Dedicated full-M fused small workspace; independent of the large ring layout.
inline int64_t HcA5FusedSmallWorkspaceBytes(const HcForwardMulSumTilingData& t)
{
    const int64_t rawCols = static_cast<int64_t>(t.R) + t.nWsGate;
    return static_cast<int64_t>(t.M) * (static_cast<int64_t>(t.D) + t.R) * t.szT + int64_t{2} * t.M * rawCols * t.szT +
           int64_t{2} * t.M * rawCols * static_cast<int64_t>(sizeof(float)) * t.splitkFactor +
           (t.needBackward ? int64_t{0} : static_cast<int64_t>(t.M) * t.D * t.szT);
}
