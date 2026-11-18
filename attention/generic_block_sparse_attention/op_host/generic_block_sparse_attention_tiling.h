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
#include "../op_kernel/arch35/gbsa_layout_utils.hpp"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

constexpr int64_t GBSA_QUANT_TYPE_A4C4 = 4;
namespace Mxfp8Tiling {
// Host selects one complete head-group tile: Q32/KV512 or Q64/KV256.
using GbsaKernelArch35::Mxfp8Descriptor::QueryTile;
using GbsaKernelArch35::Mxfp8Descriptor::KvTile;
constexpr uint32_t MAX_GROUP_SIZE = GbsaKernelArch35::Mxfp8Descriptor::MAX_GROUP_SIZE;
constexpr uint32_t QUERY_TILE = 32U;
constexpr uint32_t KV_TILE = 512U;
constexpr uint32_t CAPACITY_DIM = 256U;
constexpr uint32_t DESCRIPTOR_WORKSPACE_SLOTS = GbsaKernelArch35::Mxfp8Descriptor::WORKSPACE_SLOTS;
constexpr uint32_t RAW_STAGES = 3U;
constexpr uint32_t KSCALE_STAGES = 3U;
constexpr uint32_t COMPUTE_STAGES = 2U;
constexpr uint32_t STATS_STAGES = 4U;
constexpr uint32_t P_STAGES = 3U;
constexpr uint64_t UB_CAPACITY = 256U * 1024U;
constexpr uint64_t L1_CAPACITY = 512U * 1024U;

inline uint32_t DescriptorRows(uint32_t dim, bool paged, uint32_t tileTokens = KV_TILE)
{
    return GbsaKernelArch35::Mxfp8Descriptor::Rows(dim, paged, tileTokens);
}
inline uint64_t DescriptorTileWords(uint32_t rows, uint32_t blockY, bool paged, uint32_t tileTokens = KV_TILE)
{
    return GbsaKernelArch35::Mxfp8Descriptor::TileWords(rows, blockY, paged, tileTokens);
}
inline uint64_t L1Bytes(uint32_t group = QUERY_TILE)
{
    const uint32_t q = QueryTile(group), kv = KvTile(group);
    return q * CAPACITY_DIM + kv * CAPACITY_DIM + P_STAGES * q * kv + 2U * kv * CAPACITY_DIM + q * CAPACITY_DIM / 32U +
           KSCALE_STAGES * kv * (CAPACITY_DIM / 32U);
}
inline uint64_t UbBytes(uint32_t group = QUERY_TILE)
{
    const uint32_t r = QueryTile(group) / 2U, kv = KvTile(group);
    // Raw retains 8 KiB per tile in both tiers; NZ/descriptors keep their capacity.
    return 2U * r * kv * sizeof(float) + 3U * r * CAPACITY_DIM * sizeof(float) + 2U * r * kv +
           (3U * STATS_STAGES + 3U) * 256U + 512U + 256U + r * 32U + RAW_STAGES * 256U * 32U +
           256U * (CAPACITY_DIM / 32U) + 2U * (8U + 128U * 4U) * sizeof(uint64_t);
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
// Descriptor workspace base and words per complete tile, for all arch35 paths.
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
// A8C4 paged BF16 scale strides in elements, independent of the K/V cache strides.
TILING_DATA_FIELD_DEF(uint64_t, kScaleStride0);
TILING_DATA_FIELD_DEF(uint64_t, vScaleStride0);
// CX parameters are derived once on host; keep existing tiling offsets unchanged.
TILING_DATA_FIELD_DEF(float, log2Cx);
TILING_DATA_FIELD_DEF(float, log2CxCeil);
TILING_DATA_FIELD_DEF(uint32_t, returnSoftmaxLse);
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
    ge::graphStatus CheckInt8Config(gert::TilingContext* context);
    ge::graphStatus CheckMxfp4Config(gert::TilingContext* context);
    ge::graphStatus CheckMetadata(gert::TilingContext* context);
    ge::graphStatus CheckReservedOptionalInputs(gert::TilingContext* context);
    ge::graphStatus CheckMxfp8Scales(gert::TilingContext* context);
    ge::graphStatus ParseKvCacheStride0(gert::TilingContext* context);
    void CalcBaseTileTilingParams950();
    void CalcMatmulPhaseL1TileInfo950();
    bool IsMxfp4() const
    {
        return quantType_ == GBSA_QUANT_TYPE_A4C4 && dataType_ == ge::DT_FLOAT4_E2M1;
    }
    enum class Mxfp8ConfigError {
        NONE,
        SOC_VERSION,
        QUANT_TYPE,
        BATCH,
        Q_LAYOUT,
        DENSE_KV_LAYOUT,
        PAGED_KV_LAYOUT,
        PAGE_SIZE,
        HEAD_DIM,
        BLOCK_X,
        BLOCK_Y,
        GROUP_SIZE,
        KV_CAPACITY,
        KV_HEADS,
        Q_TASK_CAPACITY,
        Q_HEAD_STRIDE,
        L1_CAPACITY,
        UB_CAPACITY,
    };
    Mxfp8ConfigError GetMxfp8ConfigError() const;
    void LogMxfp8ConfigError(gert::TilingContext* context, Mxfp8ConfigError error) const;
    bool IsMxfp8FullTilePath() const;
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
    float log2Cx_ = 0.0f;
    float log2CxCeil_ = 0.0f;
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
    uint64_t kScaleStride0_ = 0;
    uint64_t vScaleStride0_ = 0;
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
