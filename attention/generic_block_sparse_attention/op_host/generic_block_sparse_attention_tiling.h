/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_TILING_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_TILING_H

#include <cstdint>
#include <string>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

namespace Mxfp8Tiling {
// Host-only budgets mirror the device components. Keep these inline so Host
// validation can exercise them without exporting symbols from the tiling library.
constexpr uint64_t UB_CAPACITY = 256U * 1024U;
constexpr uint64_t L1_DATA_CAPACITY = 512U * 1024U - 16U * 1024U - 3U * 4096U;
constexpr uint64_t CAPACITY = 128U * 1024U;
constexpr uint64_t DESCRIPTOR_RING_SLOTS = 4U;
constexpr uint32_t GROUP_TOKENS = 128U;
constexpr uint32_t GROUPED_TILE_TOKENS = 512U;
constexpr uint32_t GROUPS_PER_TILE = GROUPED_TILE_TOKENS / GROUP_TOKENS;
constexpr uint32_t GROUPS_PER_AIV = GROUPS_PER_TILE / 2U;
constexpr uint32_t RAW_STAGES = 2U;
constexpr uint32_t GROUPED_RAW_STAGES = 3U;
constexpr uint32_t NZ_STAGES = 2U;
constexpr uint32_t RAW_ROW_BYTES = 32U;
constexpr uint32_t UB_QUERY_ROW_ALIGNMENT = 16U;
constexpr uint32_t UB_STATS_ROWS = 64U;
constexpr uint32_t UB_COMPUTE_STAGES = 2U;
constexpr uint32_t UB_DELTA_STAGES = 3U;
constexpr uint32_t SCORE_ELEMENT_BYTES = sizeof(float);
constexpr uint32_t PROBABILITY_ELEMENT_BYTES = sizeof(uint8_t);

inline uint64_t Align(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

inline uint64_t BufferDim(uint64_t dim)
{
    // D192 and D256 share the D256 capacity tier. This is a reservation width,
    // never the active tensor width or a GM/scale address stride.
    return Align(dim, 128U);
}

inline uint32_t QueryTileRows(uint64_t group)
{
    return group <= 16U ? 16U : group <= 32U ? 32U : group <= 64U ? 64U : 128U;
}

inline uint32_t KvTileTokens(uint64_t group)
{
    return group <= 32U ? 512U : group <= 64U ? 256U : 128U;
}

inline uint32_t VectorRows(uint64_t qTile)
{
    // Each AIV reserves half the Q tile, rounded up to a 16-row capacity.
    return static_cast<uint32_t>(Align(qTile / 2U, UB_QUERY_ROW_ALIGNMENT));
}

inline uint64_t ScoreUbSlotBytes(uint64_t qTile, uint64_t tile)
{
    // All four supported M/KV tiers reserve 32 KiB per FP32 S slot per AIV.
    return VectorRows(qTile) * tile * SCORE_ELEMENT_BYTES;
}

inline uint32_t PvUbStages(uint64_t qTile)
{
    return qTile <= 64U ? UB_COMPUTE_STAGES : 1U;
}

inline uint64_t OutputUbSlotBytes(uint64_t dim, uint64_t qTile)
{
    return VectorRows(qTile) * BufferDim(dim) * sizeof(float);
}

inline uint64_t ValueScaleUbBytes(uint64_t dim)
{
    return BufferDim(dim) * sizeof(uint16_t);
}

inline uint64_t NzScaleUbSlotBytes(uint64_t dim)
{
    return Align(GROUP_TOKENS * (BufferDim(dim) / 32U), 256U);
}

inline uint32_t GroupsPerAiv(uint64_t tile)
{
    // A 128-token tile still gives AIV1 one empty notification to balance READY/FREE.
    return static_cast<uint32_t>((tile / GROUP_TOKENS + 1U) / 2U);
}

inline uint64_t RawTileTokens(uint64_t tile)
{
    return GroupsPerAiv(tile) * GROUP_TOKENS;
}

inline uint64_t DescriptorGroupWords()
{
    // Reserve an eight-word header and two words per possible token span.
    // Keep this conservative descriptor capacity independent of blockY.
    return 8U + 2U * GROUP_TOKENS;
}

inline uint64_t DescriptorTileWords(uint64_t tile)
{
    // Keep a header for AIV1's empty group in the single-group tier.
    return 2U * GroupsPerAiv(tile) * DescriptorGroupWords();
}

inline uint64_t PipelineL1End(uint64_t dim, uint64_t qTile, uint64_t kvTile)
{
    // Q, three K groups, three full P tiles, and two full V tiles (all FP8).
    const uint64_t capacityDim = BufferDim(dim);
    return qTile * capacityDim + 3U * GROUP_TOKENS * capacityDim + UB_DELTA_STAGES * qTile * kvTile +
           2U * kvTile * capacityDim;
}

inline uint32_t RawBlockRows(uint64_t blockY, bool headMajor)
{
    return headMajor ? static_cast<uint32_t>(blockY < 16U ? blockY : 16U) : 1U;
}

inline uint64_t RawCapacityBytes(uint64_t tokens)
{
    // Both layouts reserve 4 KiB per 128-token group. Each AIV tile slot
    // contains two groups for KV512 and one group for KV256/KV128.
    // BNSD keeps its compact payload at the beginning of each group region.
    return tokens * RAW_ROW_BYTES;
}

inline uint64_t StreamBytes(uint64_t dim, uint64_t tile, uint64_t blockY = 1U, bool headMajor = false)
{
    // BSND selects one head from each token and pads each row to 32 B.
    // BNSD keeps a contiguous microblock (at most 16 tokens) in one DMA block;
    // only that block's end is padded. This is the payload's padded footprint,
    // not the reserved capacity or the stride between groups/tile slots.
    const uint64_t rows = RawBlockRows(blockY, headMajor);
    return (tile + rows - 1U) / rows * Align(rows * (dim / 32U), RAW_ROW_BYTES);
}

inline bool StreamFits(uint64_t heads, uint64_t dim, uint64_t tile, uint64_t blockY = 1U, bool headMajor = false)
{
    const uint64_t capacity = RawCapacityBytes(tile);
    return heads != 0U && dim >= 32U && capacity <= CAPACITY && StreamBytes(dim, tile, blockY, headMajor) <= capacity;
}

inline uint64_t PipelineAuxBytes(uint64_t dim)
{
    return NZ_STAGES * NzScaleUbSlotBytes(dim);
}

inline uint64_t PipelineCacheOffset(uint64_t dim, uint64_t tile, uint64_t qTile = 16U)
{
    const uint64_t rows = VectorRows(qTile);
    const uint64_t scoreBytes = ScoreUbSlotBytes(qTile, tile) * UB_COMPUTE_STAGES;
    const uint64_t probabilityBytes = rows * tile * PROBABILITY_ELEMENT_BYTES * UB_COMPUTE_STAGES;
    // The largest head group has one PV result buffer and one FP32 accumulator.
    const uint64_t outputBytes = OutputUbSlotBytes(dim, qTile) * (PvUbStages(qTile) + 1U);
    // Local/global maxima and sums, plus three online correction buffers.
    const uint64_t statisticsBytes = (4U + UB_DELTA_STAGES) * UB_STATS_ROWS * sizeof(float);
    const uint64_t rowScratchBytes = UB_STATS_ROWS * 32U;
    const uint64_t valueScaleBytes = ValueScaleUbBytes(dim);
    return Align(scoreBytes + probabilityBytes + outputBytes + statisticsBytes + rowScratchBytes + valueScaleBytes,
                 256U);
}

inline uint64_t PipelineUbEndForStages(uint64_t dim, uint64_t tile, uint32_t rawStages, uint64_t qTile = 16U)
{
    // Unsupported tile sizes must not accidentally pass the Host capacity check.
    if ((qTile != 16U && qTile != 32U && qTile != 64U && qTile != 128U) || tile != KvTileTokens(qTile)) {
        return UB_CAPACITY + 1U;
    }
    const uint64_t cacheOffset = PipelineCacheOffset(dim, tile, qTile);
    return cacheOffset + rawStages * RawCapacityBytes(RawTileTokens(tile)) + PipelineAuxBytes(dim) +
           DescriptorTileWords(tile) * sizeof(uint64_t);
}

inline uint32_t PipelineRawStages(uint64_t dim, uint64_t tile, uint64_t qTile = 16U)
{
    // Retain the grouped three-slot preference and its bounded two-slot fallback.
    return PipelineUbEndForStages(dim, tile, GROUPED_RAW_STAGES, qTile) <= UB_CAPACITY ? GROUPED_RAW_STAGES :
                                                                                         RAW_STAGES;
}

inline uint64_t PipelineUbEnd(uint64_t dim, uint64_t tile, uint64_t qTile = 16U)
{
    return PipelineUbEndForStages(dim, tile, PipelineRawStages(dim, tile, qTile), qTile);
}
} // namespace Mxfp8Tiling

// Host tiling-key bases (BSA-style bitfields). Full keys live in op_kernel tilingkey.h.
constexpr uint64_t GBSA_OP_ARCH22_BASE = 9200000000000000ULL; // aicore220
constexpr uint64_t GBSA_OP_ARCH35_BASE = 9250000000000000ULL; // aicore310
constexpr uint64_t GBSA_LSE_OUT_OFFSET = 100000000ULL;

BEGIN_TILING_DATA_DEF(GbsaBaseTiling)
TILING_DATA_FIELD_DEF(uint32_t, qBaseTile);
TILING_DATA_FIELD_DEF(uint32_t, kvBaseTile);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GbsaBaseTilingOp, GbsaBaseTiling)

BEGIN_TILING_DATA_DEF(GbsaMmPhaseL1Tiling)
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileM);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileN);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileKLeft);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileKRight);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileM);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileN);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileKLeft);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileKRight);
TILING_DATA_FIELD_DEF(uint32_t, qL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, kL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, vL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, pL1BufNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GbsaMmPhaseL1TilingOp, GbsaMmPhaseL1Tiling)

BEGIN_TILING_DATA_DEF(GenericBlockSparseAttentionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, blockShapeX);
TILING_DATA_FIELD_DEF(uint32_t, blockShapeY);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(uint32_t, qBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerBatch);
TILING_DATA_FIELD_DEF(float, scaleValue);
// Workspace大小
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize);
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
TILING_DATA_FIELD_DEF(uint64_t, updateSize);
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, groupSize);
// PA_BBND page base stride in elements (dim0). Allows first-axis non-contiguous KV cache.
TILING_DATA_FIELD_DEF(uint64_t, kStride0);
TILING_DATA_FIELD_DEF(uint64_t, vStride0);
TILING_DATA_FIELD_DEF(uint32_t, fdStaticEnabled);
TILING_DATA_FIELD_DEF(uint32_t, fdLseSubStride);
TILING_DATA_FIELD_DEF(uint32_t, fdPartialCapacity);
TILING_DATA_FIELD_DEF(uint64_t, fdPartialLseOffset);
TILING_DATA_FIELD_DEF(uint64_t, fdPartialOOffset);
// 0: do not force-select residual KV block; 1: always include residual KV block.
TILING_DATA_FIELD_DEF(uint32_t, residualBlockMode);
// 0: per-Q-block sparseBlockCount; 1: every Q block uses the same topK.
TILING_DATA_FIELD_DEF(uint32_t, isConsistentTopk);
// Storage token capacities, not sums of seqused. PA KV counts all physical page tokens.
TILING_DATA_FIELD_DEF(uint64_t, totalQTokens);
TILING_DATA_FIELD_DEF(uint64_t, totalKvTokens);
TILING_DATA_FIELD_DEF_STRUCT(GbsaBaseTiling, baseTileInfo);
TILING_DATA_FIELD_DEF_STRUCT(GbsaMmPhaseL1Tiling, mmPhaseL1TileInfo);
// Nonquantized: descriptor base/words per tile. MXFP8 leaves both fields zero;
// its per-core descriptor rings start at workspace offset zero and use the kernel descriptor layout.
TILING_DATA_FIELD_DEF(uint64_t, kvOffsetInfoOffset);
TILING_DATA_FIELD_DEF(uint32_t, kvOffsetTileWords);
// Fixed per-batch storage capacities for BSND/BNSD. TND uses cuSeqLengths instead.
TILING_DATA_FIELD_DEF(uint32_t, qSeqCapacity);
TILING_DATA_FIELD_DEF(uint32_t, kvSeqCapacity);
// Mask mode consumed by arch22: 0 = none, 1 = built-in causal.
TILING_DATA_FIELD_DEF(uint32_t, maskType);
// Arch22 layout strides: 0=TND/PA_BBND, 1=BSND, 2=BNSD, 3=PA_BNBD.
TILING_DATA_FIELD_DEF(uint32_t, qLayout);
TILING_DATA_FIELD_DEF(uint32_t, kvLayout);
// Union packed O/LSE scratch (per AIC) when multi-token GQA rows are non-contiguous in GM.
TILING_DATA_FIELD_DEF(uint64_t, unionIoPackOffset);
TILING_DATA_FIELD_DEF(uint32_t, unionIoPackORowStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GenericBlockSparseAttention, GenericBlockSparseAttentionTilingData)

struct GenericBlockSparseAttentionCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;

    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion;
};

class GBSATiling {
public:
    GBSATiling() = default;
    ~GBSATiling() = default;

    ge::graphStatus GetTiling(gert::TilingContext* context, GenericBlockSparseAttentionTilingData& tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext* context, GenericBlockSparseAttentionTilingData& tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext* context);
    ge::graphStatus ParseAttrs(gert::TilingContext* context);
    ge::graphStatus ParseCapabilityAttrs(gert::TilingContext* context);
    ge::graphStatus CheckReservedAttrs(gert::TilingContext* context);
    ge::graphStatus GetInputLayout(gert::TilingContext* context);
    ge::graphStatus ParseInputTensors(gert::TilingContext* context);
    ge::graphStatus ParseQueryKeyShapes(gert::TilingContext* context);
    ge::graphStatus ParseSparseTensors(gert::TilingContext* context);
    ge::graphStatus ParseSequenceInputs(gert::TilingContext* context);
    ge::graphStatus ParseBlockTable(gert::TilingContext* context);
    ge::graphStatus ParseQkvDtype(gert::TilingContext* context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext* context);
    ge::graphStatus FillTilingData(gert::TilingContext* context);
    ge::graphStatus CheckAttentionOutDtype(gert::TilingContext* context);
    ge::graphStatus CheckSoftmaxPrecision(gert::TilingContext* context);
    ge::graphStatus CheckQuantConfig(gert::TilingContext* context);
    ge::graphStatus CheckMetadata(gert::TilingContext* context);
    ge::graphStatus CheckReservedOptionalInputs(gert::TilingContext* context);
    ge::graphStatus CheckMxfp8Scales(gert::TilingContext* context);
    ge::graphStatus ParseKvCacheStride0(gert::TilingContext* context);
    void CalcBaseTileTilingParams950();
    void CalcMatmulPhaseL1TileInfo950();
    bool IsMxfp8PackedPath() const;
    uint64_t GenerateTilingKey();

    uint32_t batch_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t embeddingSize_ = 0;
    // Non-union: blockShapeX=1. Union (maskType=3): blockShapeX carries unionSize
    // and currently only 8 is supported.
    uint32_t blockShapeX_ = 1;
    uint32_t blockShapeY_ = 128;
    uint32_t blockSize_ = 128;
    uint32_t topK_ = 16;
    uint32_t qBlockNum_ = 0;
    // Union mode (maskType=3): group count G; qBlockNum_ carries it and topK_
    // carries the union capacity U so existing addressing stays valid.
    uint32_t unionGroupNum_ = 0;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t groupSize_ = 0;
    float scaleValue_ = 0.0f;
    uint32_t softmaxPrecision_ = 0;
    int64_t residualBlockMode_ = 0;
    bool isConsistentTopk_ = false;
    int64_t softmaxLseFlag_ = 0;
    bool returnSoftmaxlse_ = false;
    bool isPaged_ = false;  // Derived only from blockTable input presence.
    bool isPaBnbd_ = false; // Paged KV laid out as [blockNum, Nkv, blockSize, D].

    uint64_t mm1OutSize_ = 0;
    uint64_t smOnlineOutSize_ = 0;
    uint64_t mm2OutSize_ = 0;
    uint64_t updateSize_ = 0;
    uint64_t workSpaceSize_ = 0;

    uint32_t blockDim_ = 0;
    uint32_t aicNum_ = 0;
    uint64_t libapiSize_ = 0;
    uint32_t socVer_ = 0;

    ge::DataType dataType_ = ge::DT_FLOAT16;
    ge::DataType attentionOutDtype_ = ge::DT_FLOAT16;

    std::string layoutQ_ = "TND";
    std::string layoutKv_ = "TND";
    int64_t layoutSparsePattern_ = 4;
    int64_t maskType_ = 1;
    int64_t quantType_ = 0;

    // Default = contiguous PA_BBND page size; overwritten when KV is a dim0-strided view.
    uint64_t kStride0_ = 0;
    uint64_t vStride0_ = 0;
    uint32_t qSeqCapacity_ = 0;
    uint32_t kvSeqCapacity_ = 0;
    uint64_t totalQTokens_ = 0;
    uint64_t totalKvTokens_ = 0;

    uint32_t qBaseTile_ = 128;
    uint32_t kvBaseTile_ = 128;
    uint32_t mm1L1TileM_ = 128;
    uint32_t mm1L1TileN_ = 128;
    uint32_t mm1L1TileKLeft_ = 128;
    uint32_t mm1L1TileKRight_ = 128;
    uint32_t mm2L1TileM_ = 128;
    uint32_t mm2L1TileN_ = 128;
    uint32_t mm2L1TileKLeft_ = 128;
    uint32_t mm2L1TileKRight_ = 128;
    uint32_t qL1BufNum_ = 1;
    uint32_t kL1BufNum_ = 1;
    uint32_t vL1BufNum_ = 1;
    uint32_t pL1BufNum_ = 3;

    bool fdStaticEnabled_ = false;
    uint32_t fdLseSubStride_ = 0;
    uint32_t fdPartialCapacity_ = 0;
    uint64_t fdPartialLseOffset_ = 0;
    uint64_t fdPartialOOffset_ = 0;
    uint64_t unionIoPackOffset_ = 0;
    uint32_t unionIoPackORowStride_ = 0;

    GenericBlockSparseAttentionTilingData* tilingData_ = nullptr;
};

} // namespace optiling

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_TILING_H
