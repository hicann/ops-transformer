/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MINIMAX_SPARSE_ATTENTION_SPLIT_KV_TILING_H
#define MINIMAX_SPARSE_ATTENTION_SPLIT_KV_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

// Tiling Key 常量 (与 op_kernel/minimax_sparse_attention_split_kv_tilingkey.h 保持同步)
constexpr uint64_t MINIMAX_SA_SPLIT_KV_BASE_TILING = 20000;
constexpr uint64_t MINIMAX_SA_SPLIT_KV_BF16_D128_TILING = 20001;
constexpr uint64_t MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_LOW_TILING = 20002;
constexpr uint64_t MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_HIGH_TILING = 20003;
// FP8 Q/K/V + FP32 O_partial + BF16 out (same softmax template as 20001).
constexpr uint64_t MINIMAX_SA_SPLIT_KV_FP8_D128_BF16_TILING = 20004;
constexpr uint64_t MINIMAX_SA_SPLIT_KV_FP16_D128_TILING = 20007;
constexpr uint64_t MINIMAX_SA_SPLIT_KV_FP16_D128_INNER_LOW_TILING = 20008;
constexpr uint64_t MINIMAX_SA_SPLIT_KV_FP16_D128_HIGH_SOFTMAX_TILING = 20009;

BEGIN_TILING_DATA_DEF(MinimaxSparseAttentionSplitKvTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, groupSize);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint64_t, queryTokenStride);
TILING_DATA_FIELD_DEF(uint64_t, keyBlockStride);
TILING_DATA_FIELD_DEF(uint64_t, valueBlockStride);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(uint32_t, totalQTokens);
TILING_DATA_FIELD_DEF(uint32_t, numKvBlocks);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, k2qNnzUpperBound);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNumP1);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNumP2);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, innerPrecise);
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag);
TILING_DATA_FIELD_DEF(uint64_t, accumOutSize);
TILING_DATA_FIELD_DEF(uint64_t, lseStatSize);
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);
TILING_DATA_FIELD_DEF(uint64_t, gmSWorkspaceSize);
TILING_DATA_FIELD_DEF(uint64_t, gmPWorkspaceSize);
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, isKvContinuous);
TILING_DATA_FIELD_DEF(uint64_t, keyTokenStride);
TILING_DATA_FIELD_DEF(uint64_t, valueTokenStride);
TILING_DATA_FIELD_DEF(uint32_t, isPageAttention);
TILING_DATA_FIELD_DEF(uint32_t, layoutType);
TILING_DATA_FIELD_DEF(uint32_t, qSeqLen);
TILING_DATA_FIELD_DEF(uint32_t, kvSeqLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MinimaxSparseAttentionSplitKv, MinimaxSparseAttentionSplitKvTilingData)

struct MinimaxSparseAttentionSplitKvCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
};

// =========================================================================================
// A5 (ascend950) Tiling —— 独立实现, 与 A2 完全解耦
//
// 特点: L0C→UB 直接 Fixpipe, 无需 GM S 中转 workspace
// 支持: bf16 输入, D=128, innerPrecise 0/1/4
// 不支持: fp16 输入, stride, 连续 KV (blockSize=0)
// 实现位于 minimax_sparse_attention_split_kv_tiling_a5.cpp
// =========================================================================================
class MinimaxSaSplitKvTilingA5 {
public:
    ge::graphStatus GetTiling(gert::TilingContext *context, MinimaxSparseAttentionSplitKvTilingData &tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext *context, MinimaxSparseAttentionSplitKvTilingData &tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *context);
    ge::graphStatus ParseAttrs(gert::TilingContext *context);
    ge::graphStatus ParseInputTensors(gert::TilingContext *context);
    ge::graphStatus CheckTilingConstraints(gert::TilingContext *context);
    ge::graphStatus ParseSeqlens(gert::TilingContext *context);
    ge::graphStatus CalculateReverseIndexMeta(gert::TilingContext *context);
    ge::graphStatus CalculateTaskSplit(gert::TilingContext *context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context);
    ge::graphStatus FillTilingData(gert::TilingContext *context);

    uint32_t batch_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t embeddingSize_ = 0;
    uint32_t blockSize_ = 128;
    uint32_t topK_ = 8;
    uint32_t totalQTokens_ = 0;
    uint32_t numKvBlocks_ = 0;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t k2qNnzUpperBound_ = 0;
    float scaleValue_ = 0.0f;
    uint32_t innerPrecise_ = 4;
    uint32_t isPageAttention_ = 1;
    uint32_t softmaxLseFlag_ = 0;
    uint32_t layoutType_ = 0;
    uint32_t qSeqLen_ = 0;
    uint32_t kvSeqLen_ = 0;

    uint64_t workSpaceSize_ = 0;
    uint64_t accumOutSize_ = 0;
    uint64_t lseStatSize_ = 0;

    uint32_t blockDim_ = 20;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t libapiSize_ = 0;

    ge::DataType dataType_ = ge::DT_BF16;
    MinimaxSparseAttentionSplitKvTilingData *tilingData_ = nullptr;
};

// =========================================================================================
// A2 (ascend910b) Tiling —— 独立实现, 与 A5 完全解耦
//
// 特点: L0C→GM→UB 中转, 需额外 per-core GM S/P staging workspace
// 支持: bf16+fp16 输入, D=128/256, stride, 连续 KV (blockSize=0), innerPrecise 0/1/4
// blockDim = aicNum_ (SyncAll 需要所有核同时启动)
// 实现位于 minimax_sparse_attention_split_kv_tiling_a2.cpp
// =========================================================================================
class MinimaxSaSplitKvTilingA2 {
public:
    ge::graphStatus GetTiling(gert::TilingContext *context, MinimaxSparseAttentionSplitKvTilingData &tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext *context, MinimaxSparseAttentionSplitKvTilingData &tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *context);
    ge::graphStatus ParseAttrs(gert::TilingContext *context);
    ge::graphStatus ParseInputTensors(gert::TilingContext *context);
    ge::graphStatus CheckTilingConstraints(gert::TilingContext *context);
    ge::graphStatus ParseSeqlens(gert::TilingContext *context);
    ge::graphStatus CalculateReverseIndexMeta(gert::TilingContext *context);
    ge::graphStatus CalculateTaskSplit(gert::TilingContext *context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context);
    ge::graphStatus FillTilingData(gert::TilingContext *context);

    uint32_t batch_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t embeddingSize_ = 0;
    uint32_t blockSize_ = 128;
    uint64_t queryTokenStride_ = 0;
    uint64_t keyBlockStride_ = 0;
    uint64_t valueBlockStride_ = 0;
    uint32_t topK_ = 8;
    uint32_t totalQTokens_ = 0;
    uint32_t numKvBlocks_ = 0;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t k2qNnzUpperBound_ = 0;
    float scaleValue_ = 0.0f;
    uint32_t innerPrecise_ = 4;
    uint32_t softmaxLseFlag_ = 0;

    bool isKvContinuous_ = false;
    uint64_t keyTokenStride_ = 0;
    uint64_t valueTokenStride_ = 0;
    uint32_t isPageAttention_ = 1;
    uint32_t layoutType_ = 0;
    uint32_t qSeqLen_ = 0;
    uint32_t kvSeqLen_ = 0;

    uint64_t workSpaceSize_ = 0;
    uint64_t accumOutSize_ = 0;
    uint64_t lseStatSize_ = 0;
    uint64_t gmSWorkspaceSize_ = 0;
    uint64_t gmPWorkspaceSize_ = 0;

    uint32_t blockDim_ = 20;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t libapiSize_ = 0;

    ge::DataType dataType_ = ge::DT_BF16;
    MinimaxSparseAttentionSplitKvTilingData *tilingData_ = nullptr;
};

} // namespace optiling

#endif // MINIMAX_SPARSE_ATTENTION_SPLIT_KV_TILING_H
