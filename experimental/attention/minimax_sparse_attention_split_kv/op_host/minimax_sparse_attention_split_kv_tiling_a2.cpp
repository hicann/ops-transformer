/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * =========================================================================================
 * minimax_sparse_attention_split_kv -- A2 (ascend910b) Tiling
 * =========================================================================================
 *
 * Complete, independent A2 tiling implementation.  Does NOT share code with A5.
 *
 * A2 characteristics:
 *   - L0C->GM->UB staging (cannot Fixpipe directly to UB like A5)
 *   - Requires per-core GM S staging workspace (gmSWorkspaceSize > 0)
 *   - Requires per-core GM P staging workspace (gmPWorkspaceSize > 0)
 *   - bf16 + fp16 input supported
 *   - D=128 and D=256 supported
 *   - Supports continuous KV (blockSize=0 -> isKvContinuous_=true)
 *   - blockDim = aicNum_ (SyncAll requires all cores launched)
 *   - Parses stride from view tensors (queryTokenStride/keyBlockStride/etc.)
 *   - Tiling keys: 20001-20006 via SelectPrefillTilingKey (supports fp16 keys 20004-20006)
 *
 * A2 Workspace layout (kernel reads by offset):
 *   +---------------------------------------------------------------+
 *   | libapiWorkspace (libapiSize_)                                 |
 *   +---------------------------------------------------------------+
 *   | accumOut    [totalQ * kvHeads * topK * groupSize * D]         |
 *   |   fp32 (innerPrecise!=1) or bf16 (innerPrecise==1)            |
 *   +---------------------------------------------------------------+
 *   | softmaxMax  [totalQ * kvHeads * topK * roundUp(groupSize,8)]  |
 *   +---------------------------------------------------------------+
 *   | softmaxSum  [totalQ * kvHeads * topK * roundUp(groupSize,8)]  |
 *   +---------------------------------------------------------------+
 *   | GM S staging [blockDim * 2 * L0_TILE_M * blockSize * elemSz]  |
 *   |   per-core: 2 * 64 * 128 * 2 = 32768 bytes = 32KB (double buf)|
 *   +---------------------------------------------------------------+
 *   | GM P staging [blockDim * 3 * L0_TILE_M * blockSize * 2]        |
 *   |   per-core: 3 * 64 * 128 * 2 = 49152 bytes = 48KB (triple buf)|
 *   +---------------------------------------------------------------+
 * =========================================================================================
 */

#include "minimax_sparse_attention_split_kv_tiling.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"

// =================================================================================
// File-scope constants
// =================================================================================

// Input tensor indices (IR prototype order)
constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int BLOCK_TABLE_INDEX = 3;
constexpr int K2Q_ROW_PTR_INDEX = 4;
constexpr int K2Q_Q_INDICES_INDEX = 5;
constexpr int K2Q_SLOT_INDICES_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 7;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 8;

// TND dimension indices [T, N, D]
constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

// BNSD dimension indices [B, N, S, D]
constexpr int BNSD_DIM_B = 0;
constexpr int BNSD_DIM_N = 1;
constexpr int BNSD_DIM_S = 2;
constexpr int BNSD_DIM_D = 3;

// BSND dimension indices [B, S, N, D]
constexpr int BSND_DIM_B = 0;
constexpr int BSND_DIM_S = 1;
constexpr int BSND_DIM_N = 2;
constexpr int BSND_DIM_D = 3;

// Layout type constants
constexpr uint32_t LAYOUT_TND = 0;
constexpr uint32_t LAYOUT_BNSD = 1;
constexpr uint32_t LAYOUT_BSND = 2;

// Blocked KV dimension indices (paged KV cache [numBlocks, blockSize, kvHeads, D])
constexpr int BLOCKED_KV_DIM_BLOCK_NUM = 0;
constexpr int BLOCKED_KV_DIM_BLOCK_SIZE = 1;
constexpr int BLOCKED_KV_DIM_KV_HEAD = 2;
constexpr int BLOCKED_KV_DIM_D = 3;
constexpr uint32_t CONTINUOUS_KV_TILE_SIZE = 128U;

// Attribute indices
constexpr int ATTR_NUM_KV_HEADS_INDEX = 0;
constexpr int ATTR_SCALE_VALUE_INDEX = 1;
constexpr int ATTR_BLOCK_SIZE_INDEX = 2;
constexpr int ATTR_TOP_K_INDEX = 3;
constexpr int ATTR_INNER_PRECISE_INDEX = 4;
constexpr int ATTR_SOFTMAX_LSE_FLAG_INDEX = 5;
constexpr int ATTR_INPUT_LAYOUT_INDEX = 6;

// innerPrecise values
constexpr uint32_t INNER_PRECISE_ALL_HIGH = 0; // fp32 score softmax
constexpr uint32_t INNER_PRECISE_ALL_LOW = 1;  // bf16 O_partial
constexpr uint32_t INNER_PRECISE_MIXED = 4;    // bf16 softmax + fp32 O_partial

// A2: select tiling key by dtype + innerPrecise (supports fp16 keys 20004-20006)
static uint64_t SelectPrefillTilingKey(ge::DataType dataType, uint32_t innerPrecise)
{
    const bool isFp16 = dataType == ge::DT_FLOAT16;
    if (innerPrecise == 0U) {
        return isFp16 ? optiling::MINIMAX_SA_SPLIT_KV_FP16_D128_HIGH_SOFTMAX_TILING :
                        optiling::MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_HIGH_TILING;
    }
    if (innerPrecise == 1U) {
        return isFp16 ? optiling::MINIMAX_SA_SPLIT_KV_FP16_D128_INNER_LOW_TILING :
                        optiling::MINIMAX_SA_SPLIT_KV_BF16_D128_INNER_LOW_TILING;
    }
    return isFp16 ? optiling::MINIMAX_SA_SPLIT_KV_FP16_D128_TILING : optiling::MINIMAX_SA_SPLIT_KV_BF16_D128_TILING;
}

// Schedule mode
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

// A2 L0 tile M (smaller than A5's 128 due to GM staging double-buffer)
constexpr uint32_t A2_L0_TILE_M = 64U;

// =================================================================================
// Static helper functions (internal linkage -- no conflict with other .cpp files)
// =================================================================================

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

namespace optiling {

// =================================================================================
// MinimaxSaSplitKvTilingA2 -- A2 (ascend910b) tiling implementation
//
// A2: L0C->GM->UB staging, requires GM S/P staging workspace.
// Supports: bf16+fp16 input, D=128/256, continuous KV (blockSize=0).
// blockDim = aicNum_ (SyncAll needs all cores launched).
// =================================================================================

// ---------------------------------------------------------------------------------
// GetNpuInfo
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    blockDim_ = aicNum_; // A2: SyncAll requires all AIC cores launched
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// ParseAttrs
// A2-specific: supports blockSize=0 (continuous KV)
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    kvHeads_ = static_cast<uint32_t>(*attrs->GetInt(ATTR_NUM_KV_HEADS_INDEX));
    scaleValue_ = static_cast<float>(*attrs->GetFloat(ATTR_SCALE_VALUE_INDEX));
    int64_t blockSizeAttr = *attrs->GetInt(ATTR_BLOCK_SIZE_INDEX);
    if (blockSizeAttr < 0) {
        OP_LOGE(context->GetNodeName(), "blockSize must be non-negative.");
        return ge::GRAPH_FAILED;
    }
    // A2: blockSize=0 means continuous KV (not paged)
    isKvContinuous_ = (blockSizeAttr == 0);
    blockSize_ = isKvContinuous_ ? CONTINUOUS_KV_TILE_SIZE : static_cast<uint32_t>(blockSizeAttr);
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
                "innerPrecise must be 0 (fp32 softmax), 1 (bf16 O_partial) or 4 (bf16 softmax + fp32 O_partial), "
                "got %u.",
                innerPrecise_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// ParseInputTensors
// A2-specific: computes stride fields from view tensors
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::ParseInputTensors(gert::TilingContext *context)
{
    auto qShape = GetIrRequiredShape(context, QUERY_INDEX);
    if (qShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "query shape is null.");
        return ge::GRAPH_FAILED;
    }
    const size_t qDimNum = qShape->GetOriginShape().GetDimNum();
    if (layoutType_ == LAYOUT_TND) {
        if (qDimNum != 3U) {
            OP_LOGE(context->GetNodeName(), "inputLayout TND requires query [T, N, D] rank 3, got dimNum=%zu.",
                    qDimNum);
            return ge::GRAPH_FAILED;
        }
        totalQTokens_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(TND_DIM_T));
        numHeads_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(TND_DIM_N));
        embeddingSize_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(TND_DIM_D));
        qSeqLen_ = 0U;
    } else if (layoutType_ == LAYOUT_BNSD) {
        if (qDimNum != 4U) {
            OP_LOGE(context->GetNodeName(), "inputLayout BNSD requires query [B, N, S, D] rank 4, got dimNum=%zu.",
                    qDimNum);
            return ge::GRAPH_FAILED;
        }
        batch_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(BNSD_DIM_B));
        numHeads_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(BNSD_DIM_N));
        qSeqLen_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(BNSD_DIM_S));
        embeddingSize_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(BNSD_DIM_D));
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
        batch_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(BSND_DIM_B));
        qSeqLen_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(BSND_DIM_S));
        numHeads_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(BSND_DIM_N));
        embeddingSize_ = static_cast<uint32_t>(qShape->GetOriginShape().GetDim(BSND_DIM_D));
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
    const size_t kDimNum = kShape->GetOriginShape().GetDimNum();
    const size_t vDimNum = vShape->GetOriginShape().GetDimNum();
    if (kDimNum != vDimNum) {
        OP_LOGE(context->GetNodeName(), "key dimNum (%zu) must equal value dimNum (%zu).", kDimNum, vDimNum);
        return ge::GRAPH_FAILED;
    }

    // block_table present => paged KV cache; absent => contiguous dense K/V.
    // When blockSize attr is 0 (continuous KV), treat as non-paged regardless of block_table.
    const gert::StorageShape *btShape = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    isPageAttention_ = (btShape != nullptr && !isKvContinuous_) ? 1U : 0U;

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
            kvHeads_ = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(BLOCKED_KV_DIM_KV_HEAD));
        }
        if (btShape->GetOriginShape().GetDimNum() != 2U) {
            OP_LOGE(context->GetNodeName(), "block_table must be [batch, maxBlocksPerBatch], got dimNum=%zu.",
                    btShape->GetOriginShape().GetDimNum());
            return ge::GRAPH_FAILED;
        }
        batch_ = static_cast<uint32_t>(btShape->GetOriginShape().GetDim(0));
        maxBlocksPerBatch_ = static_cast<uint32_t>(btShape->GetOriginShape().GetDim(1));
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
            keyB = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(BNSD_DIM_B));
            keyHeads = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(BNSD_DIM_N));
            kvSeqLen_ = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(BNSD_DIM_S));
            keyD = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(BNSD_DIM_D));
            valueB = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(BNSD_DIM_B));
            valueHeads = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(BNSD_DIM_N));
            valueS = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(BNSD_DIM_S));
            valueD = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(BNSD_DIM_D));
        } else {
            keyB = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(BSND_DIM_B));
            kvSeqLen_ = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(BSND_DIM_S));
            keyHeads = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(BSND_DIM_N));
            keyD = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(BSND_DIM_D));
            valueB = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(BSND_DIM_B));
            valueS = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(BSND_DIM_S));
            valueHeads = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(BSND_DIM_N));
            valueD = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(BSND_DIM_D));
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
        uint32_t seqQBatch = static_cast<uint32_t>(seqQShape->GetOriginShape().GetShapeSize());
        uint32_t seqKvBatch = static_cast<uint32_t>(seqKvShape->GetOriginShape().GetShapeSize());
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
            kvHeads_ = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(TND_DIM_N));
        }
        uint32_t keyHeads = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(TND_DIM_N));
        uint32_t keyD = static_cast<uint32_t>(kShape->GetOriginShape().GetDim(TND_DIM_D));
        uint32_t valueHeads = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(TND_DIM_N));
        uint32_t valueD = static_cast<uint32_t>(vShape->GetOriginShape().GetDim(TND_DIM_D));
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
        batch_ = static_cast<uint32_t>(seqKvShape->GetOriginShape().GetShapeSize());
        maxBlocksPerBatch_ = 0U;
        kvSeqLen_ = 0U;
    }
    groupSize_ = (kvHeads_ > 0U) ? (numHeads_ / kvHeads_) : 1U;

    // Parse data type from query tensor desc
    auto qTensor = GetIrRequiredDesc(context, QUERY_INDEX);
    if (qTensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "query desc is null.");
        return ge::GRAPH_FAILED;
    }
    dataType_ = qTensor->GetDataType();

    // === A2-specific: compute stride fields from view tensors ===
    // A2 kernel reads strides from tiling data (unlike A5 which computes internally).
    // Default strides derived from shape dimensions:
    queryTokenStride_ = static_cast<uint64_t>(numHeads_) * embeddingSize_;
    keyTokenStride_ = static_cast<uint64_t>(kvHeads_) * embeddingSize_;
    valueTokenStride_ = keyTokenStride_;
    keyBlockStride_ = static_cast<uint64_t>(blockSize_) * keyTokenStride_;
    valueBlockStride_ = static_cast<uint64_t>(blockSize_) * valueTokenStride_;

    // Override strides from view tensors (non-contiguous views passed via CreateView in aclnn layer)
    if (context->InputIsView(QUERY_INDEX)) {
        auto *queryStride = context->GetInputStride(QUERY_INDEX);
        if (queryStride != nullptr && queryStride->GetDimNum() == 3U) {
            int64_t tokenStride = queryStride->GetStride(TND_DIM_T);
            if (tokenStride > 0) {
                queryTokenStride_ = static_cast<uint64_t>(tokenStride);
            }
        }
    }
    auto parseKvStride = [&](uint32_t inputIndex, bool isKey) {
        if (!context->InputIsView(inputIndex)) {
            return;
        }
        auto *stride = context->GetInputStride(inputIndex);
        if (stride == nullptr) {
            return;
        }
        int64_t axis0Stride = stride->GetStride(0);
        if (axis0Stride <= 0) {
            return;
        }
        if (isPageAttention_ == 1U) {
            // Paged: axis0 stride = block stride
            if (isKey) {
                keyBlockStride_ = static_cast<uint64_t>(axis0Stride);
            } else {
                valueBlockStride_ = static_cast<uint64_t>(axis0Stride);
            }
        } else {
            // Continuous: axis0 stride = token stride
            if (isKey) {
                keyTokenStride_ = static_cast<uint64_t>(axis0Stride);
            } else {
                valueTokenStride_ = static_cast<uint64_t>(axis0Stride);
            }
        }
    };
    parseKvStride(KEY_INDEX, true);
    parseKvStride(VALUE_INDEX, false);

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// CheckTilingConstraints
// A2-specific: D=128 or 256, bf16 or fp16, no UB overflow check
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::CheckTilingConstraints(gert::TilingContext *context)
{
    // A2: bf16 or fp16
    if (dataType_ != ge::DT_BF16 && dataType_ != ge::DT_FLOAT16) {
        OP_LOGE(context->GetNodeName(), "A2 supports bf16 or fp16 only, got dataType=%u.",
                static_cast<uint32_t>(dataType_));
        return ge::GRAPH_FAILED;
    }
    // A2: D=128 or 256
    if (embeddingSize_ != 128U && embeddingSize_ != 256U) {
        OP_LOGE(context->GetNodeName(), "A2 requires embeddingSize=128 or 256, got %u.", embeddingSize_);
        return ge::GRAPH_FAILED;
    }
    if (numHeads_ == 0U) {
        OP_LOGE(context->GetNodeName(), "numHeads must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (kvHeads_ == 0U) {
        OP_LOGE(context->GetNodeName(), "kvHeads must be positive.");
        return ge::GRAPH_FAILED;
    }
    if (numHeads_ % kvHeads_ != 0U) {
        OP_LOGE(context->GetNodeName(), "numHeads (%u) must be divisible by kvHeads (%u).", numHeads_, kvHeads_);
        return ge::GRAPH_FAILED;
    }
    // groupSize = numHeads / kvHeads, max 16
    if (groupSize_ == 0U || groupSize_ > 16U) {
        OP_LOGE(context->GetNodeName(), "groupSize must be in [1, 16], got %u.", groupSize_);
        return ge::GRAPH_FAILED;
    }
    if (topK_ == 0U) {
        OP_LOGE(context->GetNodeName(), "topK must be positive.");
        return ge::GRAPH_FAILED;
    }
    // For paged KV cache (non-continuous), blockSize must be a multiple of 16, <= 128
    if (!isKvContinuous_) {
        if (blockSize_ == 0U || blockSize_ > 128U) {
            OP_LOGE(context->GetNodeName(), "blockSize must be in (0, 128], got %u.", blockSize_);
            return ge::GRAPH_FAILED;
        }
        if (blockSize_ % 16U != 0U) {
            OP_LOGE(context->GetNodeName(), "blockSize must be a multiple of 16, got %u.", blockSize_);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// ParseSeqlens
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::ParseSeqlens(gert::TilingContext *context)
{
    auto k2qRowPtrShape = GetIrRequiredShape(context, K2Q_ROW_PTR_INDEX);
    if (k2qRowPtrShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "k2qRowPtr shape is null.");
        return ge::GRAPH_FAILED;
    }
    if (k2qRowPtrShape->GetOriginShape().GetDimNum() < 2) {
        OP_LOGE(context->GetNodeName(), "k2qRowPtr must be rank-2 [kvHeads, totalRows + 1].");
        return ge::GRAPH_FAILED;
    }
    int64_t flatSize = k2qRowPtrShape->GetOriginShape().GetShapeSize();
    if (flatSize <= 0 || kvHeads_ == 0U || static_cast<uint64_t>(flatSize) % static_cast<uint64_t>(kvHeads_) != 0U) {
        OP_LOGE(context->GetNodeName(), "k2qRowPtr flat size %lld not divisible by kvHeads %u.",
                static_cast<long long>(flatSize), kvHeads_);
        return ge::GRAPH_FAILED;
    }
    numKvBlocks_ = static_cast<uint32_t>(k2qRowPtrShape->GetOriginShape().GetDim(1)) - 1;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// CalculateReverseIndexMeta
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::CalculateReverseIndexMeta(gert::TilingContext *context)
{
    // k2qQIndices / k2qSlotIndices: [kvHeads, totalQ * topK]
    auto qIndicesShape = GetIrRequiredShape(context, K2Q_Q_INDICES_INDEX);
    if (qIndicesShape != nullptr && qIndicesShape->GetOriginShape().GetDimNum() >= 2) {
        k2qNnzUpperBound_ = static_cast<uint32_t>(qIndicesShape->GetOriginShape().GetDim(1));
    } else {
        k2qNnzUpperBound_ = totalQTokens_ * topK_;
    }
    if (k2qNnzUpperBound_ == 0U) {
        k2qNnzUpperBound_ = 1U;
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// CalculateTaskSplit
// A2: blockDim = aicNum_ (SyncAll requires all cores launched, even idle ones)
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::CalculateTaskSplit(gert::TilingContext *context)
{
    (void)context;
    uint32_t totalTaskP1 = numKvBlocks_ * kvHeads_;  // Phase1: KV block x KV head
    uint32_t totalTaskP2 = totalQTokens_ * kvHeads_; // Phase2: Q token x KV head
    // A2: SyncAll between QK, softmax, PV, and Phase2 requires all AIC cores launched.
    // Cores with no task still participate in SyncAll (must be launched).
    blockDim_ = aicNum_;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// CalculateWorkSpace
// A2-specific: includes GM S/P staging workspace (L0C->GM->UB)
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::CalculateWorkSpace(gert::TilingContext *context)
{
    (void)context;
    uint64_t slotOElems = static_cast<uint64_t>(groupSize_) * embeddingSize_; // per-slot O elements
    // A2 kernel uses slotStatElems = RoundUp(groupSize, 8) for 32-byte DMA alignment.
    // A5 uses groupSize_ directly (no alignment needed).
    uint64_t slotStatElems = static_cast<uint64_t>((groupSize_ + 7U) & ~7U);
    uint64_t taskSlots = static_cast<uint64_t>(totalQTokens_) * kvHeads_ * topK_;

    // === Common: accumOut + softmaxMax + softmaxSum ===
    // accumOut: [totalQ, kvHeads, topK, groupSize, D] fp32 or bf16
    //   innerPrecise==1: bf16 (PV fixpipe F322BF16 + Phase2 regbase cast)
    //   innerPrecise!=1: fp32 (default path)
    // softmaxMax/Sum: [totalQ, kvHeads, topK, roundUp(groupSize, 8)] fp32.
    accumOutSize_ = taskSlots * slotOElems;
    lseStatSize_ = taskSlots * slotStatElems;
    uint64_t accumOutBytes = accumOutSize_ * (innerPrecise_ == 1U ? sizeof(uint16_t) : sizeof(float));
    uint64_t userWorkspaceSize = accumOutBytes + (lseStatSize_ * 2U) * sizeof(float);
    // Align GM staging region to 32 bytes (Nd2Nz DataCopy GM->L1 requires 32-byte aligned src)
    userWorkspaceSize = (userWorkspaceSize + 31U) & ~static_cast<uint64_t>(31U);

    // === A2-specific: GM S staging workspace ===
    // A2's L0C->UB cannot Fixpipe directly (A5 supports it), needs GM staging:
    //   QK L0C -> Fixpipe -> GM S staging -> DataCopyPad -> UB
    // Per-core: 2 stages (double buffer) * L0_TILE_M * blockSize * scoreElementBytes
    //   = 2 * 64 * 128 * 2 = 32768 bytes = 32KB (blockSize=128, bf16)
    // Total = blockDim * perCoreBytes
    uint64_t gmSStageElems = static_cast<uint64_t>(A2_L0_TILE_M) * blockSize_;
    const uint32_t scoreElementBytes = innerPrecise_ == 0U ? sizeof(float) : sizeof(uint16_t);
    uint64_t gmSPerCoreBytes = 2U * gmSStageElems * scoreElementBytes;
    gmSWorkspaceSize_ = gmSPerCoreBytes;
    uint64_t totalGmSBytes = static_cast<uint64_t>(blockDim_) * gmSPerCoreBytes;
    userWorkspaceSize += totalGmSBytes;

    // === A2-specific: GM P staging workspace ===
    // VEC computes P=softmax(S) then writes to GM (UB->GM DataCopyPad), CUBE reads from GM (Nd2Nz).
    // A2 does not support VEC UB->L1 direct write (A5-only), must go through GM.
    // P is live for PRE_LAUNCH=2 batches, so 3 stages are needed.
    uint64_t gmPStageElems = static_cast<uint64_t>(A2_L0_TILE_M) * blockSize_;
    uint64_t gmPPerCoreBytes = 3U * gmPStageElems * sizeof(uint16_t); // ElementP = bf16
    gmPWorkspaceSize_ = gmPPerCoreBytes;
    uint64_t totalGmPBytes = static_cast<uint64_t>(blockDim_) * gmPPerCoreBytes;
    userWorkspaceSize += totalGmPBytes;

    workSpaceSize_ = libapiSize_ + userWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// FillTilingData
// A2-specific: uses SelectPrefillTilingKey (supports fp16 keys 20004-20006)
//              sets stride/gmS/gmP fields (non-zero)
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::FillTilingData(gert::TilingContext *context)
{
    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_groupSize(groupSize_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockSize(blockSize_);
    // A2: stride fields (non-zero, computed from view tensors)
    tilingData_->set_queryTokenStride(queryTokenStride_);
    tilingData_->set_keyBlockStride(keyBlockStride_);
    tilingData_->set_valueBlockStride(valueBlockStride_);
    tilingData_->set_isKvContinuous(static_cast<uint32_t>(isKvContinuous_));
    tilingData_->set_keyTokenStride(keyTokenStride_);
    tilingData_->set_valueTokenStride(valueTokenStride_);
    tilingData_->set_topK(topK_);
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_numKvBlocks(numKvBlocks_);
    tilingData_->set_maxBlocksPerBatch(maxBlocksPerBatch_);
    tilingData_->set_k2qNnzUpperBound(k2qNnzUpperBound_);
    tilingData_->set_totalTaskNumP1(numKvBlocks_ * kvHeads_);
    tilingData_->set_totalTaskNumP2(totalQTokens_ * kvHeads_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_innerPrecise(innerPrecise_);
    tilingData_->set_softmaxLseFlag(softmaxLseFlag_);
    tilingData_->set_accumOutSize(accumOutSize_);
    tilingData_->set_lseStatSize(lseStatSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    // A2: GM S/P staging workspace (non-zero)
    tilingData_->set_gmSWorkspaceSize(gmSWorkspaceSize_);
    tilingData_->set_gmPWorkspaceSize(gmPWorkspaceSize_);
    tilingData_->set_isPageAttention(isPageAttention_);
    tilingData_->set_layoutType(layoutType_);
    tilingData_->set_qSeqLen(qSeqLen_);
    tilingData_->set_kvSeqLen(kvSeqLen_);

    // A2 tiling keys: supports bf16 (20001-20003) and fp16 (20004-20006)
    const uint64_t tilingKeyVal = SelectPrefillTilingKey(dataType_, innerPrecise_);
    tilingData_->set_tilingKey(tilingKeyVal);
    context->SetTilingKey(tilingKeyVal);

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// GetTiling -- orchestrate all tiling steps
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::GetTiling(gert::TilingContext *context,
                                                    MinimaxSparseAttentionSplitKvTilingData &tilingData)
{
    tilingData_ = &tilingData;

    auto ret = GetNpuInfo(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ParseAttrs(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ParseInputTensors(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckTilingConstraints(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ParseSeqlens(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CalculateReverseIndexMeta(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CalculateTaskSplit(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CalculateWorkSpace(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = FillTilingData(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------------
// SetTilingData -- set schedule mode, blockDim, workspace, tiling buffer
// ---------------------------------------------------------------------------------
ge::graphStatus MinimaxSaSplitKvTilingA2::SetTilingData(gert::TilingContext *context,
                                                        MinimaxSparseAttentionSplitKvTilingData &tilingData)
{
    // A2 uses SyncAll: must set batchmode schedule (all cores launched simultaneously)
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    context->SetBlockDim(blockDim_);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(workSpaceSize_);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
