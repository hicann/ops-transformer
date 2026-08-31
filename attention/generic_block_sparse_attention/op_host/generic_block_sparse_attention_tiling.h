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

// Host tiling-key bases (BSA-style bitfields). Full keys live in op_kernel tilingkey.h.
constexpr uint64_t GBSA_OP_ARCH22_BASE = 9200000000000000ULL; // aicore220
constexpr uint64_t GBSA_OP_ARCH35_BASE = 9250000000000000ULL; // aicore310
constexpr uint64_t GBSA_LSE_OUT_OFFSET = 100000000ULL;

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
TILING_DATA_FIELD_DEF(uint32_t, qBaseTile);
TILING_DATA_FIELD_DEF(uint32_t, kvBaseTile);
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
// PAGED_BBND page base stride in elements (dim0). Allows first-axis non-contiguous KV cache.
TILING_DATA_FIELD_DEF(uint64_t, kStride0);
TILING_DATA_FIELD_DEF(uint64_t, vStride0);
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

    ge::graphStatus GetTiling(gert::TilingContext *context, GenericBlockSparseAttentionTilingData &tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext *context, GenericBlockSparseAttentionTilingData &tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *context);
    ge::graphStatus ParseAttrs(gert::TilingContext *context);
    ge::graphStatus ParseCapabilityAttrs(gert::TilingContext *context);
    ge::graphStatus CheckReservedAttrs(gert::TilingContext *context);
    ge::graphStatus GetInputLayout(gert::TilingContext *context);
    ge::graphStatus ParseInputTensors(gert::TilingContext *context);
    ge::graphStatus ParseQueryKeyShapes(gert::TilingContext *context);
    ge::graphStatus ParseSparseTensors(gert::TilingContext *context);
    ge::graphStatus ParseBlockTable(gert::TilingContext *context);
    ge::graphStatus ParseQkvDtype(gert::TilingContext *context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context);
    ge::graphStatus FillTilingData(gert::TilingContext *context);
    ge::graphStatus CheckAttentionOutDtype(gert::TilingContext *context);
    ge::graphStatus CheckSoftmaxPrecision(gert::TilingContext *context);
    ge::graphStatus CheckQuantConfig(gert::TilingContext *context);
    ge::graphStatus CheckMetadata(gert::TilingContext *context);
    ge::graphStatus CheckReservedOptionalInputs(gert::TilingContext *context);
    ge::graphStatus CheckCuSeqLengths(gert::TilingContext *context);
    ge::graphStatus ParseKvCacheStride0(gert::TilingContext *context);
    uint64_t GenerateTilingKey();

    uint32_t batch_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t embeddingSize_ = 0;
    // Regular path only supports blockShapeX=1; default must match.
    uint32_t blockShapeX_ = 1;
    uint32_t blockShapeY_ = 128;
    uint32_t blockSize_ = 128;
    uint32_t topK_ = 16;
    uint32_t qBlockNum_ = 0;
    uint32_t maxBlocksPerBatch_ = 0;
    float scaleValue_ = 0.0f;
    uint32_t softmaxPrecision_ = 0;
    int64_t isPackedGQA_ = 1;
    int64_t softmaxLseFlag_ = 0;
    bool returnSoftmaxlse_ = false;
    bool blockTablePresent_ = false;

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
    int64_t maskType_ = 0;
    int64_t quantType_ = 0;

    // Default = contiguous PAGED_BBND page size; overwritten when KV is a dim0-strided view.
    uint64_t kStride0_ = 0;
    uint64_t vStride0_ = 0;

    GenericBlockSparseAttentionTilingData *tilingData_ = nullptr;
};

} // namespace optiling

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_TILING_H
