/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "minimax_sparse_attention_split_kv_tiling.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <vector>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "log/log.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int BLOCK_TABLE_INDEX = 3;
constexpr int K2Q_ROW_PTR_INDEX = 4;
constexpr int K2Q_Q_INDICES_INDEX = 5;
constexpr int K2Q_SLOT_INDICES_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 7;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 8;

// GetInputShape / GetInputDesc use INSTANCE index. Omitting optional blockTable
// (IR 3) compacts later REQUIRED inputs, so IR 8 is OOB and BNSD/contiguous TND
// tiling reports "actual_seq_lengths ... required" (EZ1008).
// GetRequiredInputShape maps IR prototype index -> instance when the API works;
// otherwise compact the instance index ourselves.
static int RequiredInstanceIndex(gert::TilingContext *context, int irIndex)
{
    if (irIndex <= BLOCK_TABLE_INDEX) {
        return irIndex;
    }
    if (context->GetOptionalInputShape(BLOCK_TABLE_INDEX) == nullptr) {
        return irIndex - 1;
    }
    return irIndex;
}

static const gert::StorageShape *GetIrRequiredShape(gert::TilingContext *context, int irIndex)
{
    const gert::StorageShape *shape = context->GetRequiredInputShape(static_cast<size_t>(irIndex));
    if (shape != nullptr) {
        return shape;
    }
    return context->GetInputShape(static_cast<size_t>(RequiredInstanceIndex(context, irIndex)));
}

static const gert::CompileTimeTensorDesc *GetIrRequiredDesc(gert::TilingContext *context, int irIndex)
{
    const gert::CompileTimeTensorDesc *desc = context->GetRequiredInputDesc(static_cast<size_t>(irIndex));
    if (desc != nullptr) {
        return desc;
    }
    return context->GetInputDesc(static_cast<size_t>(RequiredInstanceIndex(context, irIndex)));
}

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

constexpr int BNSD_DIM_B = 0;
constexpr int BNSD_DIM_N = 1;
constexpr int BNSD_DIM_S = 2;
constexpr int BNSD_DIM_D = 3;

constexpr int BSND_DIM_B = 0;
constexpr int BSND_DIM_S = 1;
constexpr int BSND_DIM_N = 2;
constexpr int BSND_DIM_D = 3;

constexpr uint32_t LAYOUT_TND = 0;
constexpr uint32_t LAYOUT_BNSD = 1;
constexpr uint32_t LAYOUT_BSND = 2;

constexpr int BLOCKED_KV_DIM_BLOCK_NUM = 0;
constexpr int BLOCKED_KV_DIM_BLOCK_SIZE = 1;
constexpr int BLOCKED_KV_DIM_KV_HEAD = 2;
constexpr int BLOCKED_KV_DIM_D = 3;

constexpr int ATTR_NUM_KV_HEADS_INDEX = 0;
constexpr int ATTR_SCALE_VALUE_INDEX = 1;
constexpr int ATTR_BLOCK_SIZE_INDEX = 2;
constexpr int ATTR_TOP_K_INDEX = 3;
constexpr int ATTR_INNER_PRECISE_INDEX = 4;
constexpr int ATTR_SOFTMAX_LSE_FLAG_INDEX = 5;
constexpr int ATTR_INPUT_LAYOUT_INDEX = 6;

constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

constexpr uint32_t INNER_PRECISE_ALL_HIGH = 0;
constexpr uint32_t INNER_PRECISE_ALL_LOW = 1;
constexpr uint32_t INNER_PRECISE_MIXED = 4;

// Keep in sync with kernel: BlockMmadQK L0 tile, softmax MAX_UB_S_ELEM_NUM, SM_UB offsets.
constexpr uint32_t KERNEL_L0_TILE_M = 128;
constexpr uint32_t KERNEL_L0_TILE_N = 128;
constexpr uint32_t KERNEL_HEAD_SIZE = 128;
constexpr uint32_t KERNEL_MAX_BATCH_GROUPS = 8;
constexpr uint32_t KERNEL_UB_S_STAGES = 2;
constexpr uint32_t KERNEL_MAX_UB_S_ELEM = 16384;
constexpr uint32_t KERNEL_SM_ROW_MAX_ELEM = 64;
constexpr uint32_t KERNEL_UB_BLOCK = 32768;
constexpr uint32_t KERNEL_HIGH_PREC_S_BYTES =
    KERNEL_UB_S_STAGES * KERNEL_MAX_UB_S_ELEM * static_cast<uint32_t>(sizeof(float));
constexpr uint32_t KERNEL_HIGH_PREC_P_BYTES =
    KERNEL_UB_S_STAGES * KERNEL_MAX_UB_S_ELEM * static_cast<uint32_t>(sizeof(uint16_t));
constexpr uint32_t KERNEL_HIGH_PREC_TMP_BYTES = KERNEL_UB_BLOCK;
constexpr uint32_t KERNEL_HIGH_PREC_TMP_FLOATS = KERNEL_HIGH_PREC_TMP_BYTES / static_cast<uint32_t>(sizeof(float));

namespace optiling {

static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0)
        return 0;
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (align != 0) ? ((value + align - 1) / align * align) : value;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    blockDim_ = aicNum_;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    kvHeads_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_NUM_KV_HEADS_INDEX));
    scaleValue_ = static_cast<float>(*attrs->GetFloat(ATTR_SCALE_VALUE_INDEX));
    blockSize_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_BLOCK_SIZE_INDEX));
    topK_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_TOP_K_INDEX));
    innerPrecise_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_INNER_PRECISE_INDEX));
    const bool *softmaxLsePtr = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    softmaxLseFlag_ = (softmaxLsePtr != nullptr && *softmaxLsePtr) ? 1U : 0U;
    const char *layoutStr = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    if (layoutStr == nullptr || layoutStr[0] == '\0' || strcmp(layoutStr, "TND") == 0) {
        layoutType_ = LAYOUT_TND;
    } else if (strcmp(layoutStr, "BNSD") == 0) {
        layoutType_ = LAYOUT_BNSD;
    } else if (strcmp(layoutStr, "BSND") == 0) {
        layoutType_ = LAYOUT_BSND;
    } else {
        OP_LOGE(context->GetNodeName(), "inputLayout must be TND, BNSD or BSND, got %s.", layoutStr);
        return ge::GRAPH_FAILED;
    }
    if (innerPrecise_ != INNER_PRECISE_ALL_HIGH && innerPrecise_ != INNER_PRECISE_ALL_LOW &&
        innerPrecise_ != INNER_PRECISE_MIXED) {
        OP_LOGE(context->GetNodeName(),
                "innerPrecise must be 0 (fp32 softmax), 1 (bf16 softmax + bf16 O_partial) "
                "or 4 (bf16 softmax + fp32 O_partial), got %u.",
                innerPrecise_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::ParseInputTensors(gert::TilingContext *context)
{
    auto qShape = GetIrRequiredShape(context, QUERY_INDEX);
    if (qShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "query shape is null.");
        return ge::GRAPH_FAILED;
    }
    const size_t qDimNum = qShape->GetStorageShape().GetDimNum();
    if (layoutType_ == LAYOUT_TND) {
        if (qDimNum != 3U) {
            OP_LOGE(context->GetNodeName(), "inputLayout TND requires query [T, N, D] rank 3, got dimNum=%zu.",
                    qDimNum);
            return ge::GRAPH_FAILED;
        }
        totalQTokens_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(TND_DIM_T));
        numHeads_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(TND_DIM_N));
        embeddingSize_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(TND_DIM_D));
        qSeqLen_ = 0U;
    } else if (layoutType_ == LAYOUT_BNSD) {
        if (qDimNum != 4U) {
            OP_LOGE(context->GetNodeName(), "inputLayout BNSD requires query [B, N, S, D] rank 4, got dimNum=%zu.",
                    qDimNum);
            return ge::GRAPH_FAILED;
        }
        batch_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(BNSD_DIM_B));
        numHeads_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(BNSD_DIM_N));
        qSeqLen_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(BNSD_DIM_S));
        embeddingSize_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(BNSD_DIM_D));
        if (batch_ == 0U || qSeqLen_ == 0U) {
            OP_LOGE(context->GetNodeName(), "BNSD query [B, N, S, D] requires B>0 and S>0, got B=%u S=%u.", batch_,
                    qSeqLen_);
            return ge::GRAPH_FAILED;
        }
        totalQTokens_ = batch_ * qSeqLen_;
    } else {
        // BSND [B, S, N, D]
        if (qDimNum != 4U) {
            OP_LOGE(context->GetNodeName(), "inputLayout BSND requires query [B, S, N, D] rank 4, got dimNum=%zu.",
                    qDimNum);
            return ge::GRAPH_FAILED;
        }
        batch_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(BSND_DIM_B));
        qSeqLen_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(BSND_DIM_S));
        numHeads_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(BSND_DIM_N));
        embeddingSize_ = static_cast<uint32_t>(qShape->GetStorageShape().GetDim(BSND_DIM_D));
        if (batch_ == 0U || qSeqLen_ == 0U) {
            OP_LOGE(context->GetNodeName(), "BSND query [B, S, N, D] requires B>0 and S>0, got B=%u S=%u.", batch_,
                    qSeqLen_);
            return ge::GRAPH_FAILED;
        }
        totalQTokens_ = batch_ * qSeqLen_;
    }

    auto kShape = GetIrRequiredShape(context, KEY_INDEX);
    auto vShape = GetIrRequiredShape(context, VALUE_INDEX);
    if (kShape == nullptr || vShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "key/value shape is null.");
        return ge::GRAPH_FAILED;
    }
    const size_t kDimNum = kShape->GetStorageShape().GetDimNum();
    const size_t vDimNum = vShape->GetStorageShape().GetDimNum();
    if (kDimNum != vDimNum) {
        OP_LOGE(context->GetNodeName(), "key dimNum (%zu) must equal value dimNum (%zu).", kDimNum, vDimNum);
        return ge::GRAPH_FAILED;
    }

    // block_table present => paged KV cache; absent => contiguous dense K/V.
    const gert::StorageShape *btShape = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    isPageAttention_ = (btShape != nullptr) ? 1U : 0U;

    if (isPageAttention_ == 1U) {
        if (layoutType_ != LAYOUT_TND) {
            OP_LOGE(context->GetNodeName(), "paged KV cache requires TND query; do not pass block_table "
                                            "with BNSD/BSND.");
            return ge::GRAPH_FAILED;
        }
        if (kDimNum != 4U) {
            OP_LOGE(context->GetNodeName(),
                    "paged KV cache requires key/value rank 4 "
                    "[numPhysicalBlocks, blockSize, kvHeads, D], got %zu.",
                    kDimNum);
            return ge::GRAPH_FAILED;
        }
        if (kvHeads_ == 0U) {
            kvHeads_ = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BLOCKED_KV_DIM_KV_HEAD));
        }
        if (btShape->GetStorageShape().GetDimNum() != 2U) {
            OP_LOGE(context->GetNodeName(), "block_table must be [batch, maxBlocksPerBatch], got dimNum=%zu.",
                    btShape->GetStorageShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
        batch_ = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(0));
        maxBlocksPerBatch_ = static_cast<uint32_t>(btShape->GetStorageShape().GetDim(1));
        kvSeqLen_ = 0U;
    } else if (layoutType_ == LAYOUT_BNSD || layoutType_ == LAYOUT_BSND) {
        // Training contiguous: key/value rank 4 matching query layout.
        if (kDimNum != 4U) {
            OP_LOGE(context->GetNodeName(), "%s query requires key/value rank 4, got rank %zu.",
                    (layoutType_ == LAYOUT_BNSD) ? "BNSD" : "BSND", kDimNum);
            return ge::GRAPH_FAILED;
        }
        uint32_t keyB;
        uint32_t keyHeads;
        uint32_t keyD;
        uint32_t valueB;
        uint32_t valueHeads;
        uint32_t valueS;
        uint32_t valueD;
        if (layoutType_ == LAYOUT_BNSD) {
            keyB = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BNSD_DIM_B));
            keyHeads = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BNSD_DIM_N));
            kvSeqLen_ = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BNSD_DIM_S));
            keyD = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BNSD_DIM_D));
            valueB = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(BNSD_DIM_B));
            valueHeads = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(BNSD_DIM_N));
            valueS = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(BNSD_DIM_S));
            valueD = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(BNSD_DIM_D));
        } else {
            keyB = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BSND_DIM_B));
            kvSeqLen_ = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BSND_DIM_S));
            keyHeads = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BSND_DIM_N));
            keyD = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(BSND_DIM_D));
            valueB = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(BSND_DIM_B));
            valueS = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(BSND_DIM_S));
            valueHeads = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(BSND_DIM_N));
            valueD = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(BSND_DIM_D));
        }
        if (kvHeads_ == 0U) {
            kvHeads_ = keyHeads;
        }
        const char *layoutName = (layoutType_ == LAYOUT_BNSD) ? "BNSD" : "BSND";
        if (keyB != batch_ || keyHeads != kvHeads_ || keyD != embeddingSize_ || kvSeqLen_ == 0U) {
            OP_LOGE(context->GetNodeName(),
                    "%s key must match query B=%u kvHeads=%u embeddingSize=%u with S>0, "
                    "got B=%u N=%u S=%u D=%u.",
                    layoutName, batch_, kvHeads_, embeddingSize_, keyB, keyHeads, kvSeqLen_, keyD);
            return ge::GRAPH_FAILED;
        }
        if (valueB != keyB || valueHeads != keyHeads || valueS != kvSeqLen_ || valueD != keyD) {
            OP_LOGE(context->GetNodeName(), "%s value must match key shape.", layoutName);
            return ge::GRAPH_FAILED;
        }
        auto seqQShape = GetIrRequiredShape(context, ACTUAL_SEQ_LENGTHS_INDEX);
        auto seqKvShape = GetIrRequiredShape(context, ACTUAL_SEQ_LENGTHS_KV_INDEX);
        if (seqQShape == nullptr || seqKvShape == nullptr) {
            OP_LOGE(context->GetNodeName(), "actual_seq_lengths and actual_seq_lengths_kv are required for %s.",
                    layoutName);
            return ge::GRAPH_FAILED;
        }
        uint32_t seqQBatch = static_cast<uint32_t>(seqQShape->GetStorageShape().GetShapeSize());
        uint32_t seqKvBatch = static_cast<uint32_t>(seqKvShape->GetStorageShape().GetShapeSize());
        if (seqQBatch != batch_ || seqKvBatch != batch_) {
            OP_LOGE(context->GetNodeName(), "%s actual_seq_lengths length (%u, %u) must equal query B=%u.", layoutName,
                    seqQBatch, seqKvBatch, batch_);
            return ge::GRAPH_FAILED;
        }
        maxBlocksPerBatch_ = 0U;
    } else {
        // Contiguous TND: key/value [T_kv, kvHeads, D].
        if (kDimNum != 3U) {
            OP_LOGE(context->GetNodeName(),
                    "TND contiguous input requires key/value [T, kvHeads, D] "
                    "(rank 3), got rank %zu. Pass block_table for paged KV cache, "
                    "or set inputLayout to BNSD/BSND for rank-4 contiguous K/V.",
                    kDimNum);
            return ge::GRAPH_FAILED;
        }
        if (kvHeads_ == 0U) {
            kvHeads_ = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(TND_DIM_N));
        }
        uint32_t keyHeads = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(TND_DIM_N));
        uint32_t keyD = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(TND_DIM_D));
        uint32_t valueHeads = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(TND_DIM_N));
        uint32_t valueD = static_cast<uint32_t>(vShape->GetStorageShape().GetDim(TND_DIM_D));
        if (keyHeads != kvHeads_ || keyD != embeddingSize_) {
            OP_LOGE(context->GetNodeName(), "TND key shape [T, %u, %u] must match kvHeads=%u embeddingSize=%u.",
                    keyHeads, keyD, kvHeads_, embeddingSize_);
            return ge::GRAPH_FAILED;
        }
        if (valueHeads != keyHeads || valueD != keyD) {
            OP_LOGE(context->GetNodeName(), "TND value shape [T, %u, %u] must match key [T, %u, %u].", valueHeads,
                    valueD, keyHeads, keyD);
            return ge::GRAPH_FAILED;
        }
        auto seqKvShape = GetIrRequiredShape(context, ACTUAL_SEQ_LENGTHS_KV_INDEX);
        if (seqKvShape == nullptr) {
            OP_LOGE(context->GetNodeName(), "actual_seq_lengths_kv is required for contiguous TND input.");
            return ge::GRAPH_FAILED;
        }
        batch_ = static_cast<uint32_t>(seqKvShape->GetStorageShape().GetShapeSize());
        maxBlocksPerBatch_ = 0U;
        kvSeqLen_ = 0U;
    }
    groupSize_ = (kvHeads_ > 0U) ? (numHeads_ / kvHeads_) : 1U;

    auto qTensor = GetIrRequiredDesc(context, QUERY_INDEX);
    if (qTensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "query desc is null.");
        return ge::GRAPH_FAILED;
    }
    dataType_ = qTensor->GetDataType();
    auto kTensor = GetIrRequiredDesc(context, KEY_INDEX);
    auto vTensor = GetIrRequiredDesc(context, VALUE_INDEX);
    if (kTensor == nullptr || vTensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "key/value desc is null.");
        return ge::GRAPH_FAILED;
    }
    if (kTensor->GetDataType() != dataType_ || vTensor->GetDataType() != dataType_) {
        OP_LOGE(context->GetNodeName(), "MinimaxSparseAttentionSplitKv Q/K/V dtype must be consistent.");
        return ge::GRAPH_FAILED;
    }
    if (dataType_ != ge::DT_BF16 && dataType_ != ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(context->GetNodeName(), "MinimaxSparseAttentionSplitKv Q/K/V only support BF16 or FLOAT8_E4M3FN.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::CheckTilingConstraints(gert::TilingContext *context)
{
    if (embeddingSize_ != KERNEL_HEAD_SIZE) {
        OP_LOGE(context->GetNodeName(), "Only D=%u is supported, got embeddingSize=%u.", KERNEL_HEAD_SIZE,
                embeddingSize_);
        return ge::GRAPH_FAILED;
    }
    if (blockSize_ == 0U || blockSize_ > KERNEL_L0_TILE_N) {
        OP_LOGE(context->GetNodeName(), "blockSize must be in (0, %u], got %u.", KERNEL_L0_TILE_N, blockSize_);
        return ge::GRAPH_FAILED;
    }
    if (kvHeads_ == 0U || numHeads_ % kvHeads_ != 0U) {
        OP_LOGE(context->GetNodeName(), "numHeads (%u) must be divisible by kvHeads (%u).", numHeads_, kvHeads_);
        return ge::GRAPH_FAILED;
    }
    if (groupSize_ == 0U || groupSize_ > KERNEL_L0_TILE_M) {
        OP_LOGE(context->GetNodeName(), "groupSize must be in (0, %u], got %u.", KERNEL_L0_TILE_M, groupSize_);
        return ge::GRAPH_FAILED;
    }
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        if (blockSize_ != KERNEL_HEAD_SIZE) {
            OP_LOGE(context->GetNodeName(), "MinimaxSparseAttentionSplitKv fp8 path requires blockSize=%u (got %u).",
                    KERNEL_HEAD_SIZE, blockSize_);
            return ge::GRAPH_FAILED;
        }
        if (innerPrecise_ != INNER_PRECISE_MIXED) {
            OP_LOGE(context->GetNodeName(),
                    "MinimaxSparseAttentionSplitKv fp8 path uses FP32 O_partial only "
                    "(innerPrecise=4); got %u. innerPrecise=0/1 are not implemented.",
                    innerPrecise_);
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t batchGroupsMax = KERNEL_L0_TILE_M / groupSize_;
    if (batchGroupsMax == 0U || batchGroupsMax > KERNEL_MAX_BATCH_GROUPS) {
        batchGroupsMax = KERNEL_MAX_BATCH_GROUPS;
    }
    uint32_t batchM = batchGroupsMax * groupSize_;
    if (batchM > KERNEL_L0_TILE_M) {
        batchM = KERNEL_L0_TILE_M;
    }
    // Matches kernel softmax / QK split: AIV0 owns ceil(groupCount/2) whole groups.
    uint32_t mPerAiv = CeilDiv(batchGroupsMax, 2U) * groupSize_;
    if (mPerAiv > batchM) {
        mPerAiv = batchM;
    }
    uint32_t mAlign = AlignUp(static_cast<uint64_t>(mPerAiv), 16ULL);
    uint32_t nAlign = AlignUp(static_cast<uint64_t>(blockSize_), 16ULL);
    uint64_t sElem = static_cast<uint64_t>(mAlign) * nAlign;
    uint32_t grpStride = static_cast<uint32_t>(AlignUp(static_cast<uint64_t>(groupSize_), 8ULL));
    uint32_t statsElem = CeilDiv(batchGroupsMax, 2U) * grpStride;
    if (statsElem > KERNEL_SM_ROW_MAX_ELEM) {
        OP_LOGE(context->GetNodeName(),
                "softmax stats UB overflow: per-AIV rowMax/rowSum need %u floats, cap is %u "
                "(groupSize=%u batchGroupsMax=%u).",
                statsElem, KERNEL_SM_ROW_MAX_ELEM, groupSize_, batchGroupsMax);
        return ge::GRAPH_FAILED;
    }

    if (innerPrecise_ != INNER_PRECISE_ALL_HIGH) {
        return ge::GRAPH_SUCCESS;
    }

    // innerPrecise=0: fp32 S is 2x bf16. Kernel UB map (256KB):
    //   S  2 * 16384 * 4B = 128KB at 0
    //   P  2 * 16384 * 2B =  64KB at 128KB
    //   tmp 32KB at 192KB (destructive row-max/sum copy; also ND P)
    //   stats at 224KB (7*32KB), must match kernel SM_UB_GM_OFFSET
    // AIV0-only softmax (NoQuant cannot split to AIV1): shrink batchGroupsMax until
    // the FULL tile fits tmp (8192 fp32) and stats (64).
    uint64_t ubNeed = static_cast<uint64_t>(KERNEL_HIGH_PREC_S_BYTES) + KERNEL_HIGH_PREC_P_BYTES +
                      KERNEL_HIGH_PREC_TMP_BYTES +
                      2ULL * KERNEL_SM_ROW_MAX_ELEM * sizeof(float) * KERNEL_UB_S_STAGES * 2ULL;
    if (ubSize_ > 0U && ubNeed > ubSize_) {
        OP_LOGE(context->GetNodeName(),
                "innerPrecise=0 needs %lluB UB (fp32 S ping-pong + P + tmp + stats), platform UB is %lluB.",
                static_cast<unsigned long long>(ubNeed), static_cast<unsigned long long>(ubSize_));
        return ge::GRAPH_FAILED;
    }
    uint32_t gFit = batchGroupsMax;
    while (gFit > 0U) {
        uint32_t bm = gFit * groupSize_;
        if (bm > KERNEL_L0_TILE_M) {
            bm = KERNEL_L0_TILE_M;
        }
        uint64_t se = static_cast<uint64_t>(AlignUp(static_cast<uint64_t>(bm), 16ULL)) * nAlign;
        uint32_t st = gFit * grpStride;
        if (se <= KERNEL_HIGH_PREC_TMP_FLOATS && se <= KERNEL_MAX_UB_S_ELEM && st <= KERNEL_SM_ROW_MAX_ELEM) {
            break;
        }
        gFit--;
    }
    if (gFit == 0U) {
        OP_LOGE(context->GetNodeName(),
                "innerPrecise=0 AIV0-only S tile cannot fit tmp=%u fp32 / stats=%u "
                "(groupSize=%u blockSize=%u).",
                KERNEL_HIGH_PREC_TMP_FLOATS, KERNEL_SM_ROW_MAX_ELEM, groupSize_, blockSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::ParseSeqlens(gert::TilingContext *context)
{
    // printf("[MinimaxSparseAttentionSplitKvTiling] Enter ParseSeqlens\n");
    // fflush(stdout);

    auto k2qRowPtrShape = GetIrRequiredShape(context, K2Q_ROW_PTR_INDEX);
    if (k2qRowPtrShape == nullptr) {
        printf("[MinimaxSparseAttentionSplitKvTiling] k2qRowPtr shape is null!\n");
        return ge::GRAPH_FAILED;
    }
    // k2qRowPtr is a plain int32 tensor Input (shape [kvHeads, totalRows+1] or flat 1D — either,
    // row-major so the linear layout is identical). flatSize = total element count; derive
    // totalRows = flatSize / kvHeads - 1. Shape-only: no host value-read.
    int64_t flatSize = k2qRowPtrShape->GetStorageShape().GetShapeSize();
    if (flatSize <= 0 || kvHeads_ == 0U || static_cast<uint64_t>(flatSize) % static_cast<uint64_t>(kvHeads_) != 0U) {
        printf("[MinimaxSparseAttentionSplitKvTiling] k2qRowPtr flat size %lld not divisible by kvHeads %u!\n",
               static_cast<long long>(flatSize), kvHeads_);
        return ge::GRAPH_FAILED;
    }
    uint64_t perHead = static_cast<uint64_t>(flatSize) / static_cast<uint64_t>(kvHeads_);
    if (perHead == 0U) {
        printf("[MinimaxSparseAttentionSplitKvTiling] k2qRowPtr per-head size 0 (flatSize/kvHeads < 1)!\n");
        return ge::GRAPH_FAILED;
    }
    numKvBlocks_ = static_cast<uint32_t>(perHead - 1U);
    // numKvBlocks_ is derived from the k2qRowPtr Input SHAPE only (flatSize/kvHeads - 1). The
    // strided row-outer partition in CalculateTaskSplit needs NO row_ptr values on host (shape-
    // only); the kernel reads csrStart/csrEnd from the runtime GM tensor via GetValue.
    // fflush(stdout);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::CalculateReverseIndexMeta(gert::TilingContext *context)
{
    // k2qQIndices / k2qSlotIndices: [kvHeads, totalQ * topK]
    auto qIndicesShape = GetIrRequiredShape(context, K2Q_Q_INDICES_INDEX);
    if (qIndicesShape != nullptr && qIndicesShape->GetStorageShape().GetDimNum() >= 2) {
        k2qNnzUpperBound_ = static_cast<uint32_t>(qIndicesShape->GetStorageShape().GetDim(1));
    } else {
        k2qNnzUpperBound_ = totalQTokens_ * topK_;
    }
    if (k2qNnzUpperBound_ == 0) {
        k2qNnzUpperBound_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::CalculateTaskSplit(gert::TilingContext *context)
{
    // STRIDED row-outer head-split partition (Phase1). taskIdx = packedRow*kvHeads + kvHeadIdx
    // (row-outer); core coreIdx handles taskIdx in {coreIdx, coreIdx+coreNum, ...}. A row's
    // kvHeads heads are co-hot -> they land on kvHeads DIFFERENT cores, balancing the head
    // dimension for free (residual imbalance is row-level only; sim 1.44x on step3510, under the
    // Phase2 Vec bottleneck). Shape-only: NO row_ptr value read on host (drops the GetData + nnz
    // walk overhead the IntArray/const-fold path added). The kernel derives coreNum from
    // GetBlockNum(); no per-core boundary arrays are serialized.
    (void)context;
    uint32_t totalTasks = numKvBlocks_ * kvHeads_;   // Phase1 row-outer task count
    uint32_t totalTaskP2 = totalQTokens_ * kvHeads_; // Phase2 strided task count
    // blockDim covers BOTH phases (strided): cores with no task simply skip the loop.
    uint32_t totalTaskMax = std::max(totalTasks, totalTaskP2);
    blockDim_ = (totalTaskMax == 0U) ? 1U : std::min(totalTaskMax, aicNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::CalculateWorkSpace(gert::TilingContext *context)
{
    (void)context;
    uint64_t slotOElems = static_cast<uint64_t>(groupSize_) * embeddingSize_;
    uint64_t slotStatElems = static_cast<uint64_t>(groupSize_);
    uint64_t taskSlots = static_cast<uint64_t>(totalQTokens_) * kvHeads_ * topK_;

    // accumOut | softmaxMax | softmaxSum (separate buffers).
    // O_partial: [totalQ, kvHeads, topK, groupSize, D] fp32, or bf16 when innerPrecise==1
    //   (F322BF16 fixpipe + Phase2 regbase cast). max/sum: [totalQ, kvHeads, topK, groupSize]
    //   fp32 (compact per slot) regardless.
    accumOutSize_ = taskSlots * slotOElems;
    lseStatSize_ = taskSlots * slotStatElems;
    // innerPrecise==1 halves the O_partial buffer (bf16=2B vs fp32=4B); 0 and 4 keep fp32.
    // FP8 only implements the FP32 O_partial template, so workspace stays fp32.
    uint64_t accumOutBytes =
        accumOutSize_ * ((dataType_ != ge::DT_FLOAT8_E4M3FN && innerPrecise_ == 1U) ? sizeof(uint16_t) : sizeof(float));
    uint64_t userWorkspaceSize = accumOutBytes + (lseStatSize_ * 2U) * sizeof(float);
    workSpaceSize_ = libapiSize_ + userWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::FillTilingData(gert::TilingContext *context)
{
    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_groupSize(groupSize_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_topK(topK_);
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_numKvBlocks(numKvBlocks_);
    tilingData_->set_maxBlocksPerBatch(maxBlocksPerBatch_);
    tilingData_->set_k2qNnzUpperBound(k2qNnzUpperBound_);
    tilingData_->set_totalTaskNumP1(numKvBlocks_ * kvHeads_);
    tilingData_->set_totalTaskNumP2(totalQTokens_ * kvHeads_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_innerPrecise(innerPrecise_);
    tilingData_->set_accumOutSize(accumOutSize_);
    tilingData_->set_lseStatSize(lseStatSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    uint64_t tilingKeyVal = MINIMAX_SA_SPLIT_KV_BF16_D128_TILING;
    if (dataType_ == ge::DT_FLOAT8_E4M3FN) {
        tilingKeyVal = MINIMAX_SA_SPLIT_KV_FP8_D128_BF16_TILING;
    } else if (innerPrecise_ == INNER_PRECISE_ALL_LOW) {
        tilingKeyVal = MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_LOW_TILING;
    } else if (innerPrecise_ == INNER_PRECISE_ALL_HIGH) {
        tilingKeyVal = MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_HIGH_TILING;
    }
    tilingData_->set_tilingKey(tilingKeyVal);
    tilingData_->set_isPageAttention(isPageAttention_);
    tilingData_->set_softmaxLseFlag(softmaxLseFlag_);
    tilingData_->set_layoutType(layoutType_);
    tilingData_->set_qSeqLen(qSeqLen_);
    tilingData_->set_kvSeqLen(kvSeqLen_);
    context->SetTilingKey(tilingKeyVal);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::GetTiling(gert::TilingContext *context,
                                                               MinimaxSparseAttentionSplitKvTilingData &tilingData)
{
    tilingData_ = &tilingData;

    auto ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    ret = ParseAttrs(context);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    ret = ParseInputTensors(context);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    ret = CheckTilingConstraints(context);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    ret = ParseSeqlens(context);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    ret = CalculateReverseIndexMeta(context);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS)
        return ret;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MinimaxSparseAttentionSplitKvTiling::SetTilingData(gert::TilingContext *context,
                                                                   MinimaxSparseAttentionSplitKvTilingData &tilingData)
{
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);

    // Set block dim computed from both Phase1 and Phase2 task counts.
    context->SetBlockDim(blockDim_);

    // Set workspace size
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(workSpaceSize_);

    // Set tiling data
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingMinimaxSparseAttentionSplitKv(gert::TilingContext *context)
{
    // printf("[MinimaxSparseAttentionSplitKvTiling] ===== TilingMinimaxSparseAttentionSplitKv ENTER =====\n");
    if (context == nullptr) {
        printf("[MinimaxSparseAttentionSplitKvTiling] context is null!\n");
        return ge::GRAPH_FAILED;
    }
    MinimaxSparseAttentionSplitKvTilingData tilingData;
    MinimaxSparseAttentionSplitKvTiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        // printf("[MinimaxSparseAttentionSplitKvTiling] GetTiling success, calling SetTilingData\n");
        tiling.SetTilingData(context, tilingData);
        // printf("[MinimaxSparseAttentionSplitKvTiling] ===== TilingMinimaxSparseAttentionSplitKv DONE =====\n");
        return ge::GRAPH_SUCCESS;
    }
    printf("[MinimaxSparseAttentionSplitKvTiling] GetTiling FAILED!\n");
    OP_LOGE(context->GetNodeName(), "GetTiling failed");
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForMinimaxSparseAttentionSplitKv(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MinimaxSparseAttentionSplitKv)
    .Tiling(TilingMinimaxSparseAttentionSplitKv)
    .TilingParse<MinimaxSparseAttentionSplitKvCompileInfo>(TilingPrepareForMinimaxSparseAttentionSplitKv);

} // namespace optiling
