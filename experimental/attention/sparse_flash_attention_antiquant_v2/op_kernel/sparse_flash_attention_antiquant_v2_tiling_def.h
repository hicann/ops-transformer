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
 * \file sparse_flash_attention_antiquant_v2_tiling_def.h
 * \brief SparseFlashAttentionAntiquantV2 kernel 侧 TilingData 结构定义
 */

#ifndef SPARSE_FLASH_ATTENTION_ANTIQUANT_V2_TILING_DEF_H
#define SPARSE_FLASH_ATTENTION_ANTIQUANT_V2_TILING_DEF_H

#include <cstddef>
#include <cstdint>

// 与 host 侧 TilingData 序列化保持一致：标量按自然对齐，嵌套结构体按 8 字节对齐。
struct alignas(8) SparseFlashAttentionAntiquantV2BaseParamsMla {
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    int64_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t actualLenDimsQ;
    uint32_t actualLenDimsKV;
    float scaleValue;
    uint32_t nNumOfQInOneGroup;
    uint32_t outputLayout;
    uint32_t sparseMode;
    int64_t sparseBlockSize;
    uint32_t sparseBlockCount;
    int64_t dSizeVInput;
    uint32_t isActualLenDimsNull;
    uint32_t isActualLenDimsKVNull;
    uint32_t keyStride0;
    uint32_t returnSoftmaxLse;
};

struct alignas(8) SparseFlashAttentionAntiquantV2SplitKVParamsMla {
    uint32_t s2;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};

struct alignas(8) SparseFlashAttentionAntiquantV2SingleCoreParamsMla {
    uint32_t usedCoreNum;
};

struct alignas(8) SparseFlashAttentionAntiquantV2SingleCoreTensorSizeMla {
    uint32_t mmResUbSize;
    uint32_t bmm2ResUbSize;
};

struct alignas(8) SparseFlashAttentionAntiquantV2InnerSplitParams {
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
};

struct alignas(8) SparseFlashAttentionAntiquantV2TilingDataMla {
    SparseFlashAttentionAntiquantV2BaseParamsMla baseParams;
    SparseFlashAttentionAntiquantV2SplitKVParamsMla splitKVParams;
    SparseFlashAttentionAntiquantV2SingleCoreParamsMla singleCoreParams;
    SparseFlashAttentionAntiquantV2SingleCoreTensorSizeMla singleCoreTensorSize;
    SparseFlashAttentionAntiquantV2InnerSplitParams innerSplitParams;
};

static_assert(sizeof(SparseFlashAttentionAntiquantV2BaseParamsMla) == 96,
              "baseParams size must match host tiling layout");
static_assert(offsetof(SparseFlashAttentionAntiquantV2TilingDataMla, splitKVParams) == 96,
              "splitKVParams offset must match host tiling layout");
static_assert(offsetof(SparseFlashAttentionAntiquantV2TilingDataMla, singleCoreParams) == 112,
              "singleCoreParams offset must match host tiling layout");
static_assert(offsetof(SparseFlashAttentionAntiquantV2TilingDataMla, singleCoreTensorSize) == 120,
              "singleCoreTensorSize offset must match host tiling layout");
static_assert(offsetof(SparseFlashAttentionAntiquantV2TilingDataMla, innerSplitParams) == 128,
              "innerSplitParams offset must match host tiling layout");
static_assert(sizeof(SparseFlashAttentionAntiquantV2TilingDataMla) == 136,
              "tiling data size must match host tiling layout");

#endif // SPARSE_FLASH_ATTENTION_ANTIQUANT_V2_TILING_DEF_H
