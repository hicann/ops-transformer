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
 * \file quant_compressor_tiling_data.h
 * \brief
 */

#ifndef QUANT_COMPRESSOR_TILING_DATA_H
#define QUANT_COMPRESSOR_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

const uint32_t CMP_MAX_AIC_CORE_NUM = 36;
constexpr uint32_t BATCH_CONSISTENCY = 3;

namespace optiling {
struct QuantCompressorSplitCoreParams {
    uint32_t mStart = 0;
    uint32_t mEnd = 0;
    uint32_t nStart = 0;
    uint32_t nEnd = 0;
    uint32_t kStart = 0;
    uint32_t kEnd = 0;
};

// 1. 基础参数结构体
struct QuantCompressorBaseParams {
    uint32_t batchSize = 0;            // bastch size（批大小）
    uint32_t seqSize = 0;              // sequence size（kvs大小）
    uint32_t hiddenSize = 0;           // hidden size（隐藏层大小）
    uint32_t tokenSize = 0;            // token size = batchSize * seqSize(token总数：批大小x序列1长度)
    uint32_t headDim = 0;              // head size of kv
    uint32_t cmpRatio = 4;             // Compress ratio
    uint32_t usedCoreNum = 0;          // 使用核数
    uint32_t nSize = 0;                // 控制v2积攒的轮数
    uint64_t stateCacheStrideDim0 = 0; // stateCache第0维的stride
    uint32_t kBaseNum = 0;
    uint32_t kBaseSize = 0;
    uint32_t coreGroupNum = 0;
    uint32_t mLoopNum = 0;
    uint32_t batchConsistency = 0;
    QuantCompressorSplitCoreParams splitCoreParam[CMP_MAX_AIC_CORE_NUM];
};

struct QuantCompressorPageAttentionParams {
    uint32_t blockNum = 0;
    uint32_t blockSize = 1;
    uint32_t maxBlockNumPerBatch = 1;
};

struct QuantCompressorInnerSplitParams {
    uint32_t mBaseSize = 0;
    uint32_t dBaseSize = 0;
};

struct QuantCompressorWorkspaceParams {
    uint32_t mm1KvResSize = 0;
    uint32_t mm1ScoreResSize = 0;
    uint32_t vec1TailCacheSize = 0;
    uint32_t dbWorkspaceRatio = 1;
};

struct QuantCompressorTilingData {
    QuantCompressorBaseParams baseParams;
    QuantCompressorPageAttentionParams pageAttentionParams;
    QuantCompressorInnerSplitParams innerSplitParams;
    QuantCompressorWorkspaceParams workspaceParams;
};
} // namespace optiling

#endif // QUANT_COMPRESSOR_TILING_DATA_H
