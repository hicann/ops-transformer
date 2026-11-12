/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H

#include <cstdint>

namespace MhcPreSinkhornBackwardNs {
// hMixGrad 缓存模式；Phi 由 cachePhi 独立控制。
enum class HMixGradCacheMode : uint32_t {
    // 按需加载到普通 A1。
    NONE = 0U,
    // 单个 M 分块：全部 hMixGrad 常驻，供 C0/C1 的各列块共用。
    FULL,
    // 多个 M 分块：两个 hMixGrad 槽支持 C0 预加载与 C1 消费重叠。
    TILED,
};

// Host 按列分核和 L1 余量生成缓存计划，大小以 FP32 元素计，rows 按 NZ 对齐。
struct CubeCachePlan {
    HMixGradCacheMode hMixGradMode = HMixGradCacheMode::NONE;
    uint32_t cachePhi = 0;
    uint32_t validRows = 0;
    uint32_t rows = 0;
    uint32_t slotElements = 0;
    uint32_t slotCount = 0;
    uint32_t phiColumns = 0;
    uint32_t bufferElements = 0;
};
} // namespace MhcPreSinkhornBackwardNs

// Host 与 Kernel 共用的 Sinkhorn 布局计算。
namespace MhcPreSinkhornBackwardSinkhornNs {
constexpr uint32_t VECTOR_ELEMENTS = 64U;
constexpr uint32_t FLOAT_BLOCK_ELEMENTS = 8U;
constexpr uint32_t MAX_ITERATIONS = 20U;
constexpr uint32_t MAX_N = 8U;

#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline constexpr uint32_t
    AlignFloatBlock(uint32_t elements)
{
    return (elements + FLOAT_BLOCK_ELEMENTS - 1U) / FLOAT_BLOCK_ELEMENTS * FLOAT_BLOCK_ELEMENTS;
}

#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline constexpr uint32_t
    MatricesPerVector(uint32_t nSize)
{
    // N=1..8 时每个寄存器可容纳的完整矩阵数。
    return nSize == 1U ? 64U : nSize == 2U ? 16U : nSize == 3U ? 7U : nSize == 4U ? 4U : nSize == 5U ? 2U : 1U;
}

#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline constexpr uint32_t
    MatricesPerTile(uint32_t nSize, bool n4RowPacked = false)
{
    // 每个分块不超过一个 32 KiB 队列缓冲区，复用 Pre 的双缓冲。
    return n4RowPacked ? 16U : nSize == 6U ? 8U : nSize == 8U ? 4U : nSize > 5U ? 2U : MatricesPerVector(nSize);
}

#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline constexpr uint32_t
    ScratchElements(uint32_t nSize, uint32_t iterations, uint32_t tileMatrixCount = 0U)
{
    tileMatrixCount = tileMatrixCount != 0U ? tileMatrixCount : MatricesPerTile(nSize);
    const uint32_t sums = AlignFloatBlock(tileMatrixCount * nSize);
    const uint32_t norms = AlignFloatBlock(tileMatrixCount * nSize * nSize);
    // 各区域末尾预留整寄存器加载所需的冗余空间。
    const uint32_t gradients = norms + VECTOR_ELEMENTS;
    return gradients + iterations * (2U * sums + norms) + VECTOR_ELEMENTS;
}

#if defined(__CCE_AICORE__)
__aicore__
#endif
    inline constexpr uint32_t
    MaxScratchBytes()
{
    uint32_t bytes = 0;
    for (uint32_t nSize = 1; nSize <= MAX_N; ++nSize) {
        const uint32_t current = ScratchElements(nSize, MAX_ITERATIONS) * sizeof(float);
        bytes = bytes > current ? bytes : current;
    }
    return bytes;
}

// Host 选择主路径，Kernel 处理尾块。
enum class SinkhornPath : uint32_t {
    COMPACT = 0U,
    N4_COMPACT,
    N4_ROW_PACKED,
    N6_ROW_PACKED,
    N8_INTERLEAVED,
};

// Host 生成的 Sinkhorn 计划，偏移和跨度以 FP32 元素计。
struct SinkhornTilePlan {
    uint32_t nSize = 0;
    uint32_t matrixElements = 0;
    uint32_t matricesPerVector = 0;       // 紧凑布局下单个寄存器可容纳的矩阵数
    uint32_t matricesPerTile = 0;         // 每个 DMA 分块包含的矩阵数
    uint32_t sumElementsPerIteration = 0; // 单轮行和或列和的元素数，按 32 字节对齐
    uint32_t iterationDataOffset = 0;     // 输入梯度和保护区之后，各轮状态的起始偏移
    uint32_t iterationStride = 0;         // 单轮 [行和 | 列和 | 行 norm] 的跨度
    uint32_t logN = 0;
    uint32_t powerOfTwo = 0;
    uint32_t alignedVectors = 0;
    SinkhornPath path = SinkhornPath::COMPACT;
};
} // namespace MhcPreSinkhornBackwardSinkhornNs

// C0/C1 共用的 MatMul 配置
struct MhcPreSinkhornBackwardMMConfig {
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t l1K = 0;
    uint32_t depthA1 = 0;
    uint32_t depthB1 = 0;
    uint32_t dbL0A = 0;
    uint32_t dbL0B = 0;
    uint32_t dbL0C = 0;
};

// arch35 的 Host/Kernel 共享 Tiling 数据。
struct MhcPreSinkhornBackwardArch35TilingData {
    uint32_t coreNum = 0;
    uint32_t vecCoreNum = 0;
    uint64_t totalLength = 0;
    uint64_t nD = 0;
    uint64_t N = 0;
    uint64_t D = 0;
    uint64_t fusionSize = 0; // N ^ 2 + 2 * N
    float hcEps = 0.0f;
    uint32_t implMode = 0;

    MhcPreSinkhornBackwardMMConfig mmConfigC0;
    MhcPreSinkhornBackwardMMConfig mmConfigC1;

    uint32_t maxBufferDepth = 0;
    uint32_t l1UsedBytes = 0;
    uint32_t l1BOffsetElems = 0;
    uint32_t l1SingleBufferElems = 0;
    uint32_t l0ABUsedBytes = 0;
    uint32_t l0ABSingleBufferElems = 0;
    uint32_t l0CUsedBytes = 0;
    uint32_t l0CSingleBufferElems = 0;
    uint32_t c0MBlock = 0;
    uint32_t c0NBlock = 0;
    uint32_t c1KBlock = 0;
    uint32_t c0ToV2Rows = 0;
    uint32_t v2ToC1Rows = 0;

    // AIV 行分核与 AIC 列分核结果，由 Host TilingProcess 预先计算。
    uint64_t vecDealBSPeCore = 0;
    uint32_t usedVecCoreNum = 0;
    uint32_t cubeDealnDPeCore = 0;
    uint32_t v1UsedCubeCoreNum = 0;

    MhcPreSinkhornBackwardNs::CubeCachePlan cubeCachePlan;

    // Sinkhorn 迭代次数，sum_out/norm_out 各含 2 * iterTimes 个切片。
    uint64_t iterTimes = 0;
    MhcPreSinkhornBackwardSinkhornNs::SinkhornTilePlan sinkhornPlan;

    // Host 按 0 < M <= 512 下发，控制 Stage2 的输出通道和输入队列选择。
    uint32_t smallM = 0;
};

#endif // MHC_PRE_SINKHORN_BACKWARD_DATA_ARCH35_H
