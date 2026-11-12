/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_H
#define MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_data_arch35.h"

namespace optiling {
// 规划 hMixGrad 的 L1 缓存；limitElements 为可分配的 FP32 元素数，空计划表示不缓存。
inline MhcPreSinkhornBackwardNs::CubeCachePlan BuildHMixGradCachePlan(uint64_t rows, uint64_t fusionSize,
                                                                      uint32_t rowBlock, uint64_t limitElements)
{
    using namespace MhcPreSinkhornBackwardNs;
    if (rows == 0U || fusionSize == 0U || rowBlock == 0U) {
        return {};
    }
    constexpr uint64_t nzAlign = 16U;
    constexpr uint64_t maxClearBytes = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) * 32U;
    const auto align = [](uint64_t value) { return (value + nzAlign - 1U) / nzAlign * nzAlign; };
    const uint64_t validRows = std::min<uint64_t>(rows, rowBlock);
    const uint64_t cacheRows = align(validRows);
    const bool singleRowBlock = rows <= rowBlock;
    // 跨 M 分块预发 C0 时，需同时保留当前 C1 和下一 C0 的 hMixGrad。
    const uint64_t slotCount = singleRowBlock ? 1U : 2U;
    const uint64_t slotElements = cacheRows * align(fusionSize);
    const uint64_t elements = slotCount * slotElements;
    // 整块清零需满足 InitConstValue 的 uint16_t 块数上限，每块 32 字节。
    if (elements > limitElements || elements * sizeof(float) > maxClearBytes) {
        return {};
    }
    CubeCachePlan plan;
    plan.hMixGradMode = singleRowBlock ? HMixGradCacheMode::FULL : HMixGradCacheMode::TILED;
    plan.validRows = static_cast<uint32_t>(validRows);
    plan.rows = static_cast<uint32_t>(cacheRows);
    plan.slotElements = static_cast<uint32_t>(slotElements);
    plan.slotCount = static_cast<uint32_t>(slotCount);
    plan.bufferElements = static_cast<uint32_t>(elements);
    return plan;
}

inline MhcPreSinkhornBackwardNs::CubeCachePlan BuildPhiCachePlan(uint64_t rows, uint64_t fusionSize,
                                                                 uint32_t columnsPerCore, uint32_t rowBlock,
                                                                 uint64_t limitElements)
{
    using namespace MhcPreSinkhornBackwardNs;
    // 单个 M 分块无需跨块复用 Phi，使用普通 B1。
    if (rows <= rowBlock || fusionSize == 0U || columnsPerCore == 0U || rowBlock == 0U) {
        return {};
    }
    constexpr uint64_t nzAlign = 16U;
    constexpr uint64_t maxClearBytes = static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) * 32U;
    const auto align = [](uint64_t value) { return (value + nzAlign - 1U) / nzAlign * nzAlign; };
    const uint64_t phiColumns = align(columnsPerCore);
    const uint64_t alignedH = align(fusionSize);
    const uint64_t elements = phiColumns * alignedH;
    const bool needsPadding = phiColumns != columnsPerCore || alignedH != fusionSize;
    // 需要 NZ 补齐时，还需满足清零指令的块数上限。
    if (elements > limitElements || (needsPadding && elements * sizeof(float) > maxClearBytes)) {
        return {};
    }
    CubeCachePlan plan;
    plan.cachePhi = 1U;
    plan.phiColumns = static_cast<uint32_t>(phiColumns);
    plan.bufferElements = static_cast<uint32_t>(elements);
    return plan;
}

// columnsPerCore 为实际分核后的最大列数。
inline MhcPreSinkhornBackwardNs::CubeCachePlan BuildCubeCachePlan(uint64_t rows, uint64_t fusionSize,
                                                                  uint32_t columnsPerCore, uint32_t rowBlock,
                                                                  uint32_t columnBlock, uint64_t availableL1Bytes)
{
    if (rows == 0U || fusionSize == 0U || columnsPerCore == 0U || rowBlock == 0U || columnBlock == 0U) {
        return {};
    }
    const uint64_t limitElements =
        std::min<uint64_t>(availableL1Bytes / sizeof(float), std::numeric_limits<uint32_t>::max() / sizeof(float));
    auto gPlan = BuildHMixGradCachePlan(rows, fusionSize, rowBlock, limitElements);
    auto phiPlan = BuildPhiCachePlan(rows, fusionSize, columnsPerCore, rowBlock, limitElements);
    if (gPlan.bufferElements != 0U && phiPlan.bufferElements != 0U &&
        static_cast<uint64_t>(gPlan.bufferElements) + phiPlan.bufferElements > limitElements) {
        // 无法同时缓存时，比较节省的 GM -> L1 字节数，相同时选择 hMixGrad。
        // hMixGrad 从每列块读两次减为每行块一次；Phi 从每行块一次减为总共一次。
        // 使用 long double 避免估算乘法溢出。
        const uint64_t rowBlocks = 1U + (rows - 1U) / rowBlock;
        const uint64_t columnBlocks = 1U + (static_cast<uint64_t>(columnsPerCore) - 1U) / columnBlock;
        const long double gSavedBytes =
            (2U * columnBlocks - 1U) * static_cast<long double>(rows) * fusionSize * sizeof(float);
        const long double phiSavedBytes =
            (rowBlocks - 1U) * static_cast<long double>(columnsPerCore) * fusionSize * sizeof(float);
        if (gSavedBytes >= phiSavedBytes) {
            phiPlan = {};
        } else {
            gPlan = {};
        }
    }
    auto plan = gPlan;
    plan.cachePhi = phiPlan.cachePhi;
    plan.phiColumns = phiPlan.phiColumns;
    plan.bufferElements += phiPlan.bufferElements;
    return plan;
}

class MhcPreSinkhornBackwardArch35Tiling final : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcPreSinkhornBackwardArch35Tiling(gert::TilingContext* context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {}

    ~MhcPreSinkhornBackwardArch35Tiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    bool CheckTensorShape(uint32_t index, const char* name, const int64_t* dims, size_t rank,
                          bool logError = true) const;
    bool CheckOutputShape(uint32_t index, const char* name, const int64_t* dims, size_t rank,
                          bool logError = true) const;
    ge::graphStatus CheckDataTypes() const;
    void SetMmConfig();
    ge::graphStatus ValidateMmConfig();
    void FillTilingData();
    void SetCommonTilingParams();
    ge::graphStatus TilingProcess();
    void SetSinkhornTiling();
    void SetVecCoreSplit();
    void SetCubeCoreSplit();

    static uint64_t CalculateWorkspaceSize(uint64_t totalLength, uint64_t fusionSize, uint64_t cubeCoreNum,
                                           uint64_t vecCoreNum, uint64_t elementSize);

    MhcPreSinkhornBackwardArch35TilingData* tilingData_ = nullptr;
    uint64_t aicNum_ = 0;
    uint64_t aivNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t l2Size_ = 0;
    uint64_t l0ASize_ = 0;
    uint64_t l0BSize_ = 0;
    uint64_t l0CSize_ = 0;
    uint64_t systemWorkspaceBytes_ = 0;
    uint64_t workspaceSize_ = 0;
    int64_t xRank_ = 0;

    uint32_t blockDim_{0};   // AIC
    uint32_t vecCoreNum_{0}; // AIV
    uint64_t totalLength_{0};
    uint64_t N_{0};
    uint64_t D_{0};
    float hcEps_{0.0f};
    uint32_t implMode_{0U};
    uint64_t fusionSize_{0};
    uint32_t maxBufferDepth_{0};
    uint32_t l1UsedBytes_{0};
    uint32_t l1BOffsetElems_{0};
    uint32_t l1SingleBufferElems_{0};
    uint32_t l0ABUsedBytes_{0};
    uint32_t l0ABSingleBufferElems_{0};
    uint32_t l0CUsedBytes_{0};
    uint32_t l0CSingleBufferElems_{0};
};
} // namespace optiling

#endif // MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_H
