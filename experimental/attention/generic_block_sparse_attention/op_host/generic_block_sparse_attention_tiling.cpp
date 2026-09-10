/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "generic_block_sparse_attention_tiling.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include "log/log.h"
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int SPARSE_BLOCK_IDX_INDEX = 3;
constexpr int SPARSE_BLOCK_COUNT_INDEX = 4;
constexpr int METADATA_INDEX = 5;
constexpr int Q_DEQUANT_SCALE_INDEX = 6;
constexpr int K_DEQUANT_SCALE_INDEX = 7;
constexpr int V_DEQUANT_SCALE_INDEX = 8;
constexpr int BLOCK_TABLE_INDEX = 13;

// Keep in sync with sparse_attention_score_metadata.h METADATA_TOTAL_SIZE.
constexpr uint32_t GSA_METADATA_TOTAL_SIZE = 1024U;

constexpr int ATTENTION_OUT_INDEX = 0;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;

constexpr int PAGED_NZ_DIM_KV_HEAD = 1;
constexpr int PAGED_NZ_DIM_D1 = 2;
constexpr int PAGED_NZ_DIM_BLOCK_SIZE = 3;
constexpr int PAGED_NZ_DIM_D0 = 4;
constexpr int PAGED_NZ_DIM_NUM = 5;
constexpr uint32_t PAGED_NZ_INT8_D0 = 32U;

// TND + isPackedGQA=1 sparseBlockIdx 3D: [N_kv, totalQBlocks, topK]
constexpr int SPARSE_IDX_DIM_KV_HEAD = 0;
constexpr int SPARSE_IDX_DIM_Q_BLOCK = 1;
constexpr int SPARSE_IDX_DIM_KV_BLOCK = 2;
constexpr int SPARSE_IDX_DIM_NUM = 3;

// TND + isPackedGQA=1 sparseBlockCount 2D: [N_kv, totalQBlocks]
constexpr int SPARSE_COUNT_DIM_KV_HEAD = 0;
constexpr int SPARSE_COUNT_DIM_Q_BLOCK = 1;
constexpr int SPARSE_COUNT_DIM_NUM = 2;

constexpr int BLOCK_TABLE_DIM_BATCH = 0;
constexpr int BLOCK_TABLE_DIM_MAX_BLOCKS = 1;

constexpr int ATTR_BLOCK_SHAPE_INDEX = 0;
constexpr int ATTR_IS_PACKED_GQA_INDEX = 1;
constexpr int ATTR_Q_INPUT_LAYOUT_INDEX = 2;
constexpr int ATTR_KV_INPUT_LAYOUT_INDEX = 3;
constexpr int ATTR_SCALE_VALUE_INDEX = 4;
constexpr int ATTR_MASK_TYPE_INDEX = 5;
constexpr int ATTR_SOFTMAX_PRECISION_INDEX = 6;

constexpr uint32_t GSA_FD_MAX_SPLIT_NUM = 16U;
constexpr uint32_t GSA_FD_MAX_COMBINE_TASK_NUM = 32U;
constexpr uint64_t GSA_FD_WORKSPACE_ALIGNMENT = 512U;
constexpr uint32_t GBSA_MSD_DEFAULT = 2U;
constexpr uint32_t GBSA_MSD_HIGH_PRECISION = 3U;

namespace {
uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}
} // namespace

namespace optiling {

ge::graphStatus W8a8GSATiling::GetNpuInfo(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    // Task schedule is owned by AICPU metadata (saTotalTaskNum). Host only launches
    // all AIC cores; idle cores exit when taskIdx >= metadata saTotalTaskNum.
    blockDim_ = (aicNum_ == 0) ? 1U : aicNum_;
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus W8a8GSATiling::ParseAttrs(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "GetAttrs returned nullptr."),
                return ge::GRAPH_FAILED);

    const float *scalePtr = attrs->GetFloat(ATTR_SCALE_VALUE_INDEX);
    if (scalePtr != nullptr) {
        scaleValue_ = *scalePtr;
    }

    const gert::TypedContinuousVector<int64_t> *blockShapeArr = attrs->GetListInt(ATTR_BLOCK_SHAPE_INDEX);
    if (blockShapeArr != nullptr && blockShapeArr->GetSize() >= 2) {
        blockShapeX_ = static_cast<uint32_t>(blockShapeArr->GetData()[0]);
        blockShapeY_ = static_cast<uint32_t>(blockShapeArr->GetData()[1]);
    }

    const int64_t *softmaxPrecPtr = attrs->GetInt(ATTR_SOFTMAX_PRECISION_INDEX);
    if (softmaxPrecPtr != nullptr) {
        softmaxPrecision_ = static_cast<uint32_t>(*softmaxPrecPtr);
    }

    const char *layoutQPtr = attrs->GetStr(ATTR_Q_INPUT_LAYOUT_INDEX);
    if (layoutQPtr != nullptr) {
        layoutQ_ = std::string(layoutQPtr);
    }

    const char *layoutKvPtr = attrs->GetStr(ATTR_KV_INPUT_LAYOUT_INDEX);
    if (layoutKvPtr != nullptr) {
        layoutKv_ = std::string(layoutKvPtr);
    }

    const int64_t *maskTypePtr = attrs->GetInt(ATTR_MASK_TYPE_INDEX);
    if (maskTypePtr != nullptr) {
        maskType_ = *maskTypePtr;
    }

    // Kernel task decode and sparse layouts are packed-GQA only (task = T * Nkv).
    const int64_t *isPackedGqaPtr = attrs->GetInt(ATTR_IS_PACKED_GQA_INDEX);
    const int64_t isPackedGQA = (isPackedGqaPtr != nullptr) ? *isPackedGqaPtr : 1;
    if (isPackedGQA != 1) {
        OP_LOGE(context->GetNodeName(), "Unsupported isPackedGQA=%ld, only 1 (packed GQA) is supported.", isPackedGQA);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// PA_NZ has a fixed inner layout [kvHead, D1, blockSize, D0].  Only the
// page axis may carry a hole; all inner strides must remain contiguous.
static ge::graphStatus ValidatePagedNzInnerContig(gert::TilingContext *context, uint64_t inputIndex,
                                                  const gert::Shape &shape, const char *tensorName)
{
    if (shape.GetDimNum() != PAGED_NZ_DIM_NUM) {
        return ge::GRAPH_FAILED;
    }
    auto *stride = context->GetRequiredInputStride(inputIndex);
    if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t expectedStride = 1;
    for (size_t i = PAGED_NZ_DIM_D0; i >= PAGED_NZ_DIM_KV_HEAD; --i) {
        const uint64_t actualStride = static_cast<uint64_t>(stride->GetStride(i));
        if (actualStride != expectedStride) {
            OP_LOGE(context->GetNodeName(),
                    "Tensor %s dim%zu is non-contiguous for PAGED_NZ: actual stride=%llu, expected=%llu.", tensorName,
                    i, static_cast<unsigned long long>(actualStride), static_cast<unsigned long long>(expectedStride));
            return ge::GRAPH_FAILED;
        }
        expectedStride *= static_cast<uint64_t>(shape.GetDim(i));
        if (i == PAGED_NZ_DIM_KV_HEAD) {
            break;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus W8a8GSATiling::ParseKvCacheStride0(gert::TilingContext *context)
{
    const uint64_t pageElems =
        static_cast<uint64_t>(blockSize_) * static_cast<uint64_t>(kvHeads_) * static_cast<uint64_t>(embeddingSize_);

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    const gert::StorageShape *valueShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_IF(keyShape == nullptr || valueShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
                                            "key/value shape is nullptr when parsing KV stride0."),
                return ge::GRAPH_FAILED);

    if (ValidatePagedNzInnerContig(context, KEY_INDEX, keyShape->GetOriginShape(), "key") != ge::GRAPH_SUCCESS ||
        ValidatePagedNzInnerContig(context, VALUE_INDEX, valueShape->GetOriginShape(), "value") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto *keyStrides = context->GetRequiredInputStride(KEY_INDEX);
    kStride0_ = (keyStrides != nullptr && keyStrides->GetDimNum() > 0 && keyStrides->GetStride(0) > 0) ?
                    static_cast<uint64_t>(keyStrides->GetStride(0)) :
                    pageElems;
    auto *valueStrides = context->GetRequiredInputStride(VALUE_INDEX);
    vStride0_ = (valueStrides != nullptr && valueStrides->GetDimNum() > 0 && valueStrides->GetStride(0) > 0) ?
                    static_cast<uint64_t>(valueStrides->GetStride(0)) :
                    pageElems;

    const uint64_t rowElems = static_cast<uint64_t>(kvHeads_) * static_cast<uint64_t>(embeddingSize_);
    if (kStride0_ < pageElems || (rowElems > 0 && (kStride0_ % rowElems) != 0)) {
        OP_LOGE(context->GetNodeName(),
                "key dim0 stride (%llu) invalid for %s: expect >= pageElems=%llu and "
                "aligned to Nkv*D=%llu.",
                static_cast<unsigned long long>(kStride0_), layoutKv_.c_str(),
                static_cast<unsigned long long>(pageElems), static_cast<unsigned long long>(rowElems));
        return ge::GRAPH_FAILED;
    }
    if (vStride0_ < pageElems || (rowElems > 0 && (vStride0_ % rowElems) != 0)) {
        OP_LOGE(context->GetNodeName(),
                "value dim0 stride (%llu) invalid for %s: expect >= pageElems=%llu and "
                "aligned to Nkv*D=%llu.",
                static_cast<unsigned long long>(vStride0_), layoutKv_.c_str(),
                static_cast<unsigned long long>(pageElems), static_cast<unsigned long long>(rowElems));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus W8a8GSATiling::ParseInputTensors(gert::TilingContext *context)
{
    const gert::StorageShape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(queryShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Query shape is nullptr."),
                return ge::GRAPH_FAILED);

    totalQTokens_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_T));
    numHeads_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_N));
    embeddingSize_ = static_cast<uint32_t>(queryShape->GetStorageShape().GetDim(TND_DIM_D));

    const gert::StorageShape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Key shape is nullptr."),
                return ge::GRAPH_FAILED);

    // INT8 PA_NZ uses the physical format shape [blockNum, Nkv, D1, blockSize, D0].
    const gert::Shape &keyOrigin = keyShape->GetOriginShape();
    const gert::StorageShape *valueShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_IF(valueShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Value shape is nullptr."),
                return ge::GRAPH_FAILED);
    const gert::Shape &valueOrigin = valueShape->GetOriginShape();
    if (keyOrigin.GetDimNum() != PAGED_NZ_DIM_NUM || valueOrigin.GetDimNum() != PAGED_NZ_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "PAGED_NZ key/value origin shape must be 5D [page,kvHead,D1,blockSize,D0], "
                "got key=%zu value=%zu dims.",
                keyOrigin.GetDimNum(), valueOrigin.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    blockSize_ = static_cast<uint32_t>(keyOrigin.GetDim(PAGED_NZ_DIM_BLOCK_SIZE));
    if (keyOrigin.GetDim(PAGED_NZ_DIM_KV_HEAD) != valueOrigin.GetDim(PAGED_NZ_DIM_KV_HEAD) ||
        keyOrigin.GetDim(PAGED_NZ_DIM_D1) != valueOrigin.GetDim(PAGED_NZ_DIM_D1) ||
        keyOrigin.GetDim(PAGED_NZ_DIM_BLOCK_SIZE) != valueOrigin.GetDim(PAGED_NZ_DIM_BLOCK_SIZE) ||
        keyOrigin.GetDim(PAGED_NZ_DIM_D0) != valueOrigin.GetDim(PAGED_NZ_DIM_D0)) {
        OP_LOGE(context->GetNodeName(), "PAGED_NZ key/value physical dimensions must match.");
        return ge::GRAPH_FAILED;
    }

    // TND + isPackedGQA=1: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
    const gert::StorageShape *sparseIdxShape = context->GetInputShape(SPARSE_BLOCK_IDX_INDEX);
    OP_CHECK_IF(sparseIdxShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "sparseBlockIdx shape is nullptr."),
                return ge::GRAPH_FAILED);

    if (sparseIdxShape->GetStorageShape().GetDimNum() != SPARSE_IDX_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "sparseBlockIdx must be 3D [N_kv, totalQBlocks, topK] for TND, but got %zu dims.",
                sparseIdxShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    kvHeads_ = static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_HEAD));
    qBlockNum_ =
        static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_Q_BLOCK)); // totalQBlocks
    topK_ = static_cast<uint32_t>(sparseIdxShape->GetStorageShape().GetDim(SPARSE_IDX_DIM_KV_BLOCK));

    // sparseBlockCount 2D: [N_kv, totalQBlocks]
    const gert::StorageShape *sparseCountShape = context->GetInputShape(SPARSE_BLOCK_COUNT_INDEX);
    OP_CHECK_IF(sparseCountShape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "sparseBlockCount shape is nullptr."),
                return ge::GRAPH_FAILED);

    if (sparseCountShape->GetStorageShape().GetDimNum() != SPARSE_COUNT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "sparseBlockCount must be 2D [N_kv, totalQBlocks] for TND, but got %zu dims.",
                sparseCountShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const uint32_t sparseCountKvHeads =
        static_cast<uint32_t>(sparseCountShape->GetStorageShape().GetDim(SPARSE_COUNT_DIM_KV_HEAD));
    const uint32_t sparseCountQBlocks =
        static_cast<uint32_t>(sparseCountShape->GetStorageShape().GetDim(SPARSE_COUNT_DIM_Q_BLOCK));
    if (sparseCountKvHeads != kvHeads_ || sparseCountQBlocks != qBlockNum_) {
        OP_LOGE(context->GetNodeName(),
                "sparseBlockCount shape [%u,%u] must match sparseBlockIdx [N_kv,totalQBlocks]=[%u,%u].",
                sparseCountKvHeads, sparseCountQBlocks, kvHeads_, qBlockNum_);
        return ge::GRAPH_FAILED;
    }

    // blockTable is OPTIONAL in OpDef — must use GetOptionalInputShape (GetInputShape always nullptr).
    const gert::StorageShape *blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    if (blockTableShape != nullptr) {
        blockTablePresent_ = true;
        batch_ = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_BATCH));
        maxBlocksPerBatch_ =
            static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(BLOCK_TABLE_DIM_MAX_BLOCKS));
    } else {
        blockTablePresent_ = false;
        OP_LOGE(context->GetNodeName(), "W8A8 PAGED_NZ layout requires blockTable, but blockTableOptional is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto queryDesc = context->GetInputDesc(QUERY_INDEX);
    if (queryDesc != nullptr) {
        dataType_ = queryDesc->GetDataType();
    }

    auto keyDesc = context->GetInputDesc(KEY_INDEX);
    if (keyDesc != nullptr) {
        kvDataType_ = keyDesc->GetDataType();
    }

    headSplitFactor_ = 1U;

    // W8A8 pseudo-quantization is the only supported path: key/value must be INT8.
    OP_CHECK_IF(
        kvDataType_ != ge::DT_INT8,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "W8A8 pseudo-quantization requires INT8 key/value."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        kvHeads_ == 0 || numHeads_ == 0 || numHeads_ % kvHeads_ != 0,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
                                    "INT8 KV pseudo-quantization requires numHeads to be divisible by kvHeads."),
        return ge::GRAPH_FAILED);
    const uint32_t int8GroupSize = numHeads_ / kvHeads_;
    OP_CHECK_IF(embeddingSize_ == 0 || embeddingSize_ % 16 != 0,
                OPS_REPORT_VECTOR_INNER_ERR(
                    "GenericBlockSparseAttention",
                    "INT8 KV pseudo-quantization requires embeddingSize to be a positive multiple of 16."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHeads_ > 64U,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
                                            "INT8 KV pseudo-quantization supports numHeads up to 64."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(int8GroupSize > 16,
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
                                            "INT8 KV pseudo-quantization supports groupSize up to 16."),
                return ge::GRAPH_FAILED);

    const auto qScaleDesc = context->GetOptionalInputDesc(Q_DEQUANT_SCALE_INDEX);
    const auto kScaleDesc = context->GetOptionalInputDesc(K_DEQUANT_SCALE_INDEX);
    const auto vScaleDesc = context->GetOptionalInputDesc(V_DEQUANT_SCALE_INDEX);
    const bool hasCombinedScale = qScaleDesc != nullptr;
    const bool hasKScale = kScaleDesc != nullptr;
    const bool hasVScale = vScaleDesc != nullptr;
    OP_CHECK_IF(!hasCombinedScale && !(hasKScale && hasVScale),
                OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
                                            "INT8 KV requires qDequantScale or both kDequantScale and vDequantScale."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hasKScale != hasVScale,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention",
                                    "INT8 KV requires kDequantScale and vDequantScale to be provided together."),
        return ge::GRAPH_FAILED);

    // For the regular W8A8 path, keep one launched AIC block per base
    // task when the sparse batch is smaller than the physical AIC pool.
    // This keeps the fixed-count query handoff balanced and avoids idle
    // blocks entering a per-task rendezvous.
    const uint32_t antiquantTaskNum = totalQTokens_ * kvHeads_;
    if (aicNum_ != 0U && antiquantTaskNum != 0U) {
        // The qSeq=1/topK=4 target has one metadata task.  Split its
        // sixteen query heads into two independent groups so both AIC
        // blocks can preprocess and consume disjoint query rows.
        const bool exactQ1HeadSplit = batch_ == 1U && totalQTokens_ == 1U && qBlockNum_ == 1U && numHeads_ == 16U &&
                                      kvHeads_ == 1U && int8GroupSize == 16U && embeddingSize_ == 128U &&
                                      blockSize_ == 128U && topK_ == 4U && aicNum_ >= 2U;
        headSplitFactor_ = exactQ1HeadSplit ? 2U : 1U;
        // Keep the launch within the physical AIC pool.  Oversubscribed
        // task ranges use the task-scoped query handoff in the kernel.
        blockDim_ = std::min(aicNum_, antiquantTaskNum * headSplitFactor_);
        // A single TND token/head would otherwise launch one AIC block.
        // The Arch22 qS=1 split has an idle AIV sub-block; keep a second
        // block available so the task-scoped handoff follows the same
        // scheduling shape as the smallest multi-token launch.
        if (headSplitFactor_ == 1U && antiquantTaskNum == 1U && aicNum_ >= 2U) {
            blockDim_ = 2U;
        }
    }

    if (scaleValue_ < 1e-9f && scaleValue_ > -1e-9f && embeddingSize_ > 0) {
        scaleValue_ = 1.0f / std::sqrt(static_cast<float>(embeddingSize_));
    }

    // maxQSeqlen: upper bound from totalQBlocks (packed across batch)
    maxQSeqlen_ = qBlockNum_ * blockShapeX_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus W8a8GSATiling::CalculateWorkSpace(gert::TilingContext *context)
{
    constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
    constexpr uint32_t NUM3 = 3;
    // W8A8: BMM1/BMM2 outputs are int32 (same size as float), softmax output is int8.
    const uint32_t smElemSize = sizeof(int8_t);
    // Identity reserved after S/P/O buffers (must match kernel layout).
    mm1OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    smOnlineOutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * smElemSize * NUM3;
    mm2OutSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    updateSize_ = static_cast<uint64_t>(blockDim_) * WORKSPACE_BLOCK_SIZE_DB * sizeof(float) * NUM3;
    const uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
    // Match IFA int8 MSD policy: GBSA has no innerPrecise attr; use the
    // two-digit expansion by default.  The optional override is restricted
    // to the supported two- and three-digit modes; MSD1 is not valid.
    uint32_t antiquantMsdIterNum = GBSA_MSD_DEFAULT;
    const char *msdEnv = std::getenv("GBSA_ANTIQUANT_MSD_ITER");
    if (msdEnv != nullptr) {
        const long msdVal = strtol(msdEnv, nullptr, 10);
        if (msdVal == static_cast<long>(GBSA_MSD_HIGH_PRECISION)) {
            antiquantMsdIterNum = GBSA_MSD_HIGH_PRECISION;
        } else if (msdVal == static_cast<long>(GBSA_MSD_DEFAULT)) {
            antiquantMsdIterNum = GBSA_MSD_DEFAULT;
        }
    }
    const uint64_t queryPreProcessSize = static_cast<uint64_t>(totalQTokens_) * numHeads_ *
                                         AlignUp(embeddingSize_, 16U) * antiquantMsdIterNum * sizeof(int8_t);
    msdIterNum_ = antiquantMsdIterNum;
    queryPreProcessSize_ = queryPreProcessSize;
    const uint32_t groupSize = (kvHeads_ > 0) ? numHeads_ / kvHeads_ : 1;
    amaxQSize_ = static_cast<uint64_t>(blockDim_) * groupSize * 8 * sizeof(float);
    amaxPSize_ = static_cast<uint64_t>(blockDim_) * groupSize * 8 * sizeof(float) * NUM3;
    uint64_t pipelineWorkspaceSize = queryPreProcessSize + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_ + updateSize_ +
                                     identityIdxSize + amaxQSize_ + amaxPSize_;

    uint64_t userWorkspaceSize = pipelineWorkspaceSize;
    if (fdStaticEnabled_) {
        const uint32_t maxSparseBaseTasks =
            aicNum_ == 0U ? 0U : std::min(GSA_FD_MAX_COMBINE_TASK_NUM, (aicNum_ * 3U - 1U) / 10U);
        fdPartialCapacity_ = maxSparseBaseTasks * GSA_FD_MAX_SPLIT_NUM;
        fdLseSubStride_ = ((groupSize_ + 1U) / 2U + 7U) / 8U * 8U;
        fdPartialLseOffset_ = AlignUp(pipelineWorkspaceSize, GSA_FD_WORKSPACE_ALIGNMENT);
        const uint64_t partialLseSize =
            static_cast<uint64_t>(fdPartialCapacity_) * 2U * fdLseSubStride_ * sizeof(float);
        fdPartialOOffset_ = AlignUp(fdPartialLseOffset_ + partialLseSize, GSA_FD_WORKSPACE_ALIGNMENT);
        const uint64_t partialOSize =
            static_cast<uint64_t>(fdPartialCapacity_) * groupSize_ * embeddingSize_ * sizeof(float);
        if (fdPartialOOffset_ > std::numeric_limits<uint64_t>::max() - partialOSize) {
            OP_LOGE(context->GetNodeName(), "Flash Decoding workspace size overflow.");
            return ge::GRAPH_FAILED;
        }
        userWorkspaceSize = fdPartialOOffset_ + partialOSize;
    }
    if (userWorkspaceSize > std::numeric_limits<size_t>::max() - libapiSize_) {
        OP_LOGE(context->GetNodeName(), "GenericBlockSparseAttention workspace size overflow.");
        return ge::GRAPH_FAILED;
    }
    workSpaceSize_ = libapiSize_ + userWorkspaceSize;

    context->SetBlockDim(blockDim_);
    size_t *workspaceArray = context->GetWorkspaceSizes(1);
    if (workspaceArray != nullptr) {
        workspaceArray[0] = static_cast<size_t>(workSpaceSize_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus W8a8GSATiling::CheckMetadata(gert::TilingContext *context)
{
    // Align SMLA / FlashAttn: metadata is required shell — INT32, 1D, fixed size.
    // Content (magic / schedule tables) is produced by AICPU and not re-validated here.
    const gert::StorageShape *metadataShape = context->GetOptionalInputShape(METADATA_INDEX);
    if (metadataShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "metadata must be provided.");
        return ge::GRAPH_FAILED;
    }
    if (metadataShape->GetStorageShape().GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "metadata dim num must be 1, but got %zu.",
                metadataShape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }
    const int64_t metadataSize = metadataShape->GetStorageShape().GetDim(0);
    if (metadataSize != static_cast<int64_t>(GSA_METADATA_TOTAL_SIZE)) {
        OP_LOGE(context->GetNodeName(), "metadata dim 0 must be %u, but got %ld.", GSA_METADATA_TOTAL_SIZE,
                metadataSize);
        return ge::GRAPH_FAILED;
    }
    auto metadataDesc = context->GetOptionalInputDesc(METADATA_INDEX);
    if (metadataDesc == nullptr) {
        OP_LOGE(context->GetNodeName(), "metadata desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (metadataDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(context->GetNodeName(), "metadata dtype must be DT_INT32, but got %d.",
                static_cast<int32_t>(metadataDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus W8a8GSATiling::ValidateSupportedConfig(gert::TilingContext *context)
{
    // W8A8 path: query=TND, kv=PAGED_NZ, maskType=1, blockShapeX=1, isPackedGQA=1.
    // IMPORTANT: must fail hard — a wrong tiling key can launch the wrong
    // dtype kernel and destroy accuracy.
    if (layoutQ_ != "TND" || layoutKv_ != "PAGED_NZ" || maskType_ != 1 || blockShapeX_ != 1) {
        OP_LOGE(context->GetNodeName(),
                "Unsupported config: layoutQ=%s, layoutKv=%s, maskType=%ld, blockShapeX=%u. "
                "W8A8 path requires query=TND, kv=PAGED_NZ, maskType=1, blockShapeX=1.",
                layoutQ_.c_str(), layoutKv_.c_str(), maskType_, blockShapeX_);
        return ge::GRAPH_FAILED;
    }
    // The Arch22 W8A8 path is the generalized path: D is 128/256 and both
    // logical blockShapeY and the physical page size are 16-byte multiples.
    // The current sparse-page loader maps one logical sparse block to one
    // physical page, so reject mismatched sizes instead of silently using a
    // wrong page stride.
    const auto *keyDesc = context->GetInputDesc(KEY_INDEX);
    const auto *valueDesc = context->GetInputDesc(VALUE_INDEX);
    const ge::Format keyFormat =
        keyDesc == nullptr ? ge::FORMAT_RESERVED :
                             static_cast<ge::Format>(ge::GetPrimaryFormat(keyDesc->GetFormat().GetStorageFormat()));
    const ge::Format valueFormat =
        valueDesc == nullptr ? ge::FORMAT_RESERVED :
                               static_cast<ge::Format>(ge::GetPrimaryFormat(valueDesc->GetFormat().GetStorageFormat()));
    if (keyFormat != ge::FORMAT_FRACTAL_NZ || valueFormat != ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE(context->GetNodeName(), "INT8 KV pseudo-quantization requires layoutKv=PAGED_NZ and "
                                        "key/value storage format FRACTAL_NZ.");
        return ge::GRAPH_FAILED;
    }
    const auto *keyShape = context->GetInputShape(KEY_INDEX);
    const auto *valueShape = context->GetInputShape(VALUE_INDEX);
    const gert::Shape &keyOrigin = keyShape->GetOriginShape();
    const gert::Shape &valueOrigin = valueShape->GetOriginShape();
    if (keyOrigin.GetDimNum() != PAGED_NZ_DIM_NUM || valueOrigin.GetDimNum() != PAGED_NZ_DIM_NUM ||
        keyOrigin.GetDim(PAGED_NZ_DIM_KV_HEAD) != static_cast<int64_t>(kvHeads_) ||
        valueOrigin.GetDim(PAGED_NZ_DIM_KV_HEAD) != static_cast<int64_t>(kvHeads_) ||
        keyOrigin.GetDim(PAGED_NZ_DIM_D0) != PAGED_NZ_INT8_D0 ||
        valueOrigin.GetDim(PAGED_NZ_DIM_D0) != PAGED_NZ_INT8_D0 ||
        keyOrigin.GetDim(PAGED_NZ_DIM_D1) * keyOrigin.GetDim(PAGED_NZ_DIM_D0) != static_cast<int64_t>(embeddingSize_) ||
        valueOrigin.GetDim(PAGED_NZ_DIM_D1) * valueOrigin.GetDim(PAGED_NZ_DIM_D0) !=
            static_cast<int64_t>(embeddingSize_)) {
        OP_LOGE(context->GetNodeName(), "PAGED_NZ INT8 KV shape is incompatible with embedding size.");
        return ge::GRAPH_FAILED;
    }
    if (embeddingSize_ != 128U && embeddingSize_ != 256U) {
        OP_LOGE(context->GetNodeName(), "INT8 KV pseudo-quantization supports embeddingSize 128 or 256, got %u.",
                embeddingSize_);
        return ge::GRAPH_FAILED;
    }
    if (blockShapeY_ < 16U || blockShapeY_ > 128U || (blockShapeY_ % 16U) != 0U || blockSize_ < 16U ||
        blockSize_ > 128U || (blockSize_ % 16U) != 0U || blockSize_ != blockShapeY_) {
        OP_LOGE(context->GetNodeName(),
                "INT8 KV pseudo-quantization requires blockShapeY=blockSize in [16,128] "
                "with 16-byte alignment, got blockShapeY=%u blockSize=%u.",
                blockShapeY_, blockSize_);
        return ge::GRAPH_FAILED;
    }
    if (softmaxPrecision_ != 0) {
        OP_LOGE(context->GetNodeName(), "INT8 KV pseudo-quantization requires softmaxPrecision=0, but got %u.",
                softmaxPrecision_);
        return ge::GRAPH_FAILED;
    }
    if (kvHeads_ == 0 || numHeads_ % kvHeads_ != 0) {
        OP_LOGE(context->GetNodeName(), "numHeads=%u must be divisible by kvHeads=%u (and kvHeads > 0).", numHeads_,
                kvHeads_);
        return ge::GRAPH_FAILED;
    }
    groupSize_ = numHeads_ / kvHeads_;
    if (groupSize_ == 0 || groupSize_ > 128) {
        OP_LOGE(context->GetNodeName(), "Unsupported GQA group size %u, expect [1, 128].", groupSize_);
        return ge::GRAPH_FAILED;
    }
    if (topK_ == 0 || topK_ > GSA_FD_MAX_SPLIT_NUM) {
        OP_LOGE(context->GetNodeName(), "Unsupported topK=%u, current kernel capacity is [1, 16].", topK_);
        return ge::GRAPH_FAILED;
    }
    // Runtime FD is selected by metadata.
    // W8A8 antiquant FD verified: workspace S/P/OTmp/amax regions are indexed by
    // coreIdx (disjoint across split cores), ProcessPartial redirects only the
    // final float partial write, and FdCombine is dtype-agnostic.
    fdStaticEnabled_ = topK_ >= 12U;

    return ge::GRAPH_SUCCESS;
}

uint64_t W8a8GSATiling::GenerateTilingKey()
{
    // W8A8 pseudo-quant (arch22 only): INT8 KV routes all shapes through the
    // generic INT8 keys.  Key values are kept identical to the original
    // GenericBlockSparseAttention op for debugging parity (40006/40007).
    return (dataType_ == ge::DT_BF16) ? GSA_BF16_INT8_ARCH22_TILING : GSA_FP16_INT8_ARCH22_TILING;
}

ge::graphStatus W8a8GSATiling::FillTilingData(gert::TilingContext *context)
{
    tilingData_->set_batch(batch_);
    tilingData_->set_numHeads(numHeads_);
    tilingData_->set_kvHeads(kvHeads_);
    tilingData_->set_embeddingSize(embeddingSize_);
    tilingData_->set_blockShapeX(blockShapeX_);
    tilingData_->set_blockShapeY(blockShapeY_);
    tilingData_->set_blockSize(blockSize_);
    tilingData_->set_topK(topK_);
    tilingData_->set_qBlockNum(qBlockNum_);
    tilingData_->set_maxBlocksPerBatch(maxBlocksPerBatch_);
    tilingData_->set_totalQTokens(totalQTokens_);
    tilingData_->set_scaleValue(scaleValue_);
    tilingData_->set_softmaxPrecision(softmaxPrecision_);
    tilingData_->set_maxQSeqlen(maxQSeqlen_);
    tilingData_->set_mm1OutSize(mm1OutSize_);
    tilingData_->set_smOnlineOutSize(smOnlineOutSize_);
    tilingData_->set_mm2OutSize(mm2OutSize_);
    tilingData_->set_updateSize(updateSize_);
    tilingData_->set_workSpaceSize(workSpaceSize_);
    tilingData_->set_groupSize(groupSize_);
    uint64_t tilingKey = GenerateTilingKey();
    tilingData_->set_tilingKey(tilingKey);
    context->SetTilingKey(tilingKey);

    // BaseTileInfo
    uint32_t qBaseTile = (embeddingSize_ <= 128) ? 128 : 64;
    uint32_t kvBaseTile = blockShapeY_;
    tilingData_->set_qBaseTile(qBaseTile);
    tilingData_->set_kvBaseTile(kvBaseTile);

    // MmPhaseL1TileInfo: QK matmul L1 tile = [qBaseTile, kvBaseTile, embed]
    tilingData_->set_mm1L1TileM(qBaseTile);
    tilingData_->set_mm1L1TileN(kvBaseTile);
    tilingData_->set_mm1L1TileKLeft(embeddingSize_);
    tilingData_->set_mm1L1TileKRight(embeddingSize_);
    // PV matmul L1 tile = [qBaseTile, embed, kvBaseTile]
    tilingData_->set_mm2L1TileM(qBaseTile);
    tilingData_->set_mm2L1TileN(embeddingSize_);
    tilingData_->set_mm2L1TileKLeft(kvBaseTile);
    tilingData_->set_mm2L1TileKRight(kvBaseTile);
    // Buffer counts
    tilingData_->set_qL1BufNum(1);
    tilingData_->set_kL1BufNum(1);
    tilingData_->set_vL1BufNum(1);
    tilingData_->set_pL1BufNum(3); // PRE_LAUNCH + 1
    tilingData_->set_kStride0(kStride0_);
    tilingData_->set_vStride0(vStride0_);
    tilingData_->set_fdStaticEnabled(fdStaticEnabled_ ? 1U : 0U);
    tilingData_->set_fdLseSubStride(fdLseSubStride_);
    tilingData_->set_fdPartialCapacity(fdPartialCapacity_);
    tilingData_->set_fdPartialLseOffset(fdPartialLseOffset_);
    tilingData_->set_fdPartialOOffset(fdPartialOOffset_);
    // W8A8 pseudo-quantization metadata (always on for this op).
    tilingData_->set_antiquantFlag(1U);
    tilingData_->set_msdIterNum(msdIterNum_);
    tilingData_->set_queryPreProcessSize(queryPreProcessSize_);
    tilingData_->set_amaxQSize(amaxQSize_);
    tilingData_->set_amaxPSize(amaxPSize_);
    tilingData_->set_headSplitFactor(headSplitFactor_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus W8a8GSATiling::GetTiling(gert::TilingContext *context,
                                         GenericBlockSparseAttentionTilingData &tilingData)
{
    tilingData_ = &tilingData;

    ge::graphStatus ret = GetNpuInfo(context);
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

    ret = ParseKvCacheStride0(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckMetadata(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = ValidateSupportedConfig(context);
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

ge::graphStatus W8a8GSATiling::SetTilingData(gert::TilingContext *context,
                                             GenericBlockSparseAttentionTilingData &tilingData)
{
    OP_CHECK_IF(
        context->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingGenericBlockSparseAttention(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("GenericBlockSparseAttention", "Context is nullptr."),
                return ge::GRAPH_FAILED);
    GenericBlockSparseAttentionTilingData tilingData;
    W8a8GSATiling tiling;
    if (tiling.GetTiling(context, tilingData) == ge::GRAPH_SUCCESS) {
        tiling.SetTilingData(context, tilingData);
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(context->GetNodeName(), "GetTiling failed");
        return ge::GRAPH_FAILED;
    }
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGenericBlockSparseAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GenericBlockSparseAttention)
    .Tiling(TilingGenericBlockSparseAttention)
    .TilingParse<GenericBlockSparseAttentionCompileInfo>(TilingPrepareForGenericBlockSparseAttention);

} // namespace optiling
