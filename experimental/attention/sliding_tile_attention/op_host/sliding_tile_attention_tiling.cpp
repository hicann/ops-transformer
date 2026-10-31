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
 * \file sliding_tile_attention_tiling.cpp
 * \brief Host tiling of SlidingTileAttention (block-sparse MIX kernel, Atlas A2).
 */

#include "sliding_tile_attention_tiling.h"
#include <algorithm>
#include <cstring>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "log/log.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t WINDOWS_INDEX = 3;
constexpr uint32_t ATTR_VALID_KV_SEQ_LEN = 0;
constexpr uint32_t ATTR_SCALE_VALUE = 1;
constexpr uint32_t ATTR_ORIGINAL_HEAD_DIM = 2;
constexpr uint32_t ATTR_TILE_GRID = 3;
constexpr uint32_t ATTR_MAX_SELECT_NUM = 4;

constexpr uint32_t BNSD_DIM_NUM = 4;
constexpr uint64_t SPARSE_BLOCK = 384;                        // 6 x 8 x 8 tokens per tile
constexpr uint64_t KV_BLOCK_SIZE = 128;                       // KV tokens per stacked block inside the kernel
constexpr uint64_t Q_TASK_ROWS = 384;                         // query rows per kernel task (one tile), as in the kernel
constexpr uint64_t WORKSPACE_BLOCK_ELEMS = Q_TASK_ROWS * 512; // scores of one KV stack, as in the kernel
constexpr uint64_t WORKSPACE_STAGES = 4;                      // KV-stack slots of the kernel (STACK_SLOT_MAX)
// A 192 MiB L2 cache (Atlas A2 910B3) holds a fourth in-flight KV stack and the query read-ahead, and both shorten the
// kernel there; with a 96 MiB L2 (910B4) they slow it down, so the kernel keeps two stacks of delay and no read-ahead.
constexpr uint64_t LARGE_L2_BYTES = 192ULL * 1024 * 1024;
constexpr uint64_t STAGE2_DELAY_LARGE_L2 = 3;
constexpr uint64_t STAGE2_DELAY_SMALL_L2 = 2;
constexpr uint64_t IDX_PER_C0 = 8;      // int32 KV block indices per C0, as in the kernel
constexpr uint64_t WINDOW_DIMS = 3;     // window values per head (tiles along T, H and W)
constexpr uint64_t TILE_GRID_BITS = 16; // bits per axis of the tileGrid attribute (T | H << 16 | W << 32)
constexpr uint64_t TILE_GRID_MASK = (1ULL << TILE_GRID_BITS) - 1;
constexpr uint64_t SYNC_BYTES = sizeof(uint32_t) * 256;
constexpr uint32_t AIV_PER_AIC = 2;
constexpr uint64_t PREPASS_ROWS = 256;  // rows per query/value preparation step, as in the kernel
constexpr uint64_t SUM_COLUMN_PAD = 16; // padded columns needed for the row-sum column (one fp16 C0)

uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

bool IsSupportedHeadDim(uint64_t headDim)
{
    return headDim == 16 || headDim == 32 || headDim == 48 || headDim == 64;
}

// Query tasks of one head: one per complete 384-token block plus one for the tail rows.
uint64_t QueryTasksPerHead(uint64_t seqLen)
{
    const uint64_t tasksPerBlock = CeilDiv(SPARSE_BLOCK, Q_TASK_ROWS);
    const uint64_t fullBlocks = seqLen / SPARSE_BLOCK;
    const uint64_t tail = seqLen - fullBlocks * SPARSE_BLOCK;
    return tasksPerBlock * fullBlocks + CeilDiv(tail, Q_TASK_ROWS);
}

ge::graphStatus ParseShapes(gert::TilingContext* context, uint64_t& batch, uint64_t& heads, uint64_t& seqLen,
                            uint64_t& headDim, uint64_t& blocks)
{
    const gert::StorageShape* qShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(qShape == nullptr, OP_LOGE(context->GetNodeName(), "query shape is nullptr"), return ge::GRAPH_FAILED);
    const gert::StorageShape* kShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(kShape == nullptr, OP_LOGE(context->GetNodeName(), "key shape is nullptr"), return ge::GRAPH_FAILED);
    const gert::StorageShape* vShape = context->GetInputShape(VALUE_INDEX);
    OP_CHECK_IF(vShape == nullptr, OP_LOGE(context->GetNodeName(), "value shape is nullptr"), return ge::GRAPH_FAILED);
    const gert::StorageShape* windowsShape = context->GetInputShape(WINDOWS_INDEX);
    OP_CHECK_IF(windowsShape == nullptr, OP_LOGE(context->GetNodeName(), "windows shape is nullptr"),
                return ge::GRAPH_FAILED);

    const auto& qDims = qShape->GetStorageShape();
    if (qDims.GetDimNum() != BNSD_DIM_NUM) {
        OP_LOGE(context->GetNodeName(), "query must be a 4D BNSD tensor");
        return ge::GRAPH_FAILED;
    }
    if (kShape->GetStorageShape() != qDims) {
        OP_LOGE(context->GetNodeName(), "key shape must equal query shape");
        return ge::GRAPH_FAILED;
    }
    if (vShape->GetStorageShape() != qDims) {
        OP_LOGE(context->GetNodeName(), "value shape must equal query shape");
        return ge::GRAPH_FAILED;
    }
    batch = static_cast<uint64_t>(qDims.GetDim(0));
    heads = static_cast<uint64_t>(qDims.GetDim(1));
    seqLen = static_cast<uint64_t>(qDims.GetDim(2));
    headDim = static_cast<uint64_t>(qDims.GetDim(3));
    if (batch == 0 || heads == 0 || seqLen == 0) {
        OP_LOGE(context->GetNodeName(), "query dimensions must be positive");
        return ge::GRAPH_FAILED;
    }
    if (!IsSupportedHeadDim(headDim)) {
        OP_LOGE(context->GetNodeName(), "head dim must be 16, 32, 48 or 64, got %lu", headDim);
        return ge::GRAPH_FAILED;
    }

    blocks = CeilDiv(seqLen, SPARSE_BLOCK);
    const auto& windowsDims = windowsShape->GetStorageShape();
    if (windowsDims.GetDimNum() != BNSD_DIM_NUM || windowsDims.GetDim(0) != 1 ||
        static_cast<uint64_t>(windowsDims.GetDim(1)) != heads || windowsDims.GetDim(2) != 1 ||
        static_cast<uint64_t>(windowsDims.GetDim(3)) != WINDOW_DIMS) {
        OP_LOGE(context->GetNodeName(), "windows shape must be [1, %lu, 1, %lu]", heads, WINDOW_DIMS);
        return ge::GRAPH_FAILED;
    }

    const auto* qDesc = context->GetInputDesc(QUERY_INDEX);
    OP_CHECK_IF(qDesc == nullptr, OP_LOGE(context->GetNodeName(), "query descriptor is nullptr"),
                return ge::GRAPH_FAILED);
    const ge::DataType dtype = qDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "query dtype must be fp16 or bf16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

ASCENDC_EXTERN_C ge::graphStatus TilingSlidingTileAttention(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SlidingTileAttention", "context is nullptr"),
                return ge::GRAPH_FAILED);
    uint64_t batch = 0;
    uint64_t heads = 0;
    uint64_t seqLen = 0;
    uint64_t headDim = 0;
    uint64_t blocks = 0;
    if (ParseShapes(context, batch, heads, seqLen, headDim, blocks) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* validKvPtr = attrs->GetAttrPointer<int64_t>(ATTR_VALID_KV_SEQ_LEN);
    OP_CHECK_IF(validKvPtr == nullptr, OP_LOGE(context->GetNodeName(), "attr validKvSeqLen is nullptr"),
                return ge::GRAPH_FAILED);
    const float* scalePtr = attrs->GetAttrPointer<float>(ATTR_SCALE_VALUE);
    OP_CHECK_IF(scalePtr == nullptr, OP_LOGE(context->GetNodeName(), "attr scaleValue is nullptr"),
                return ge::GRAPH_FAILED);
    const int64_t* originalHeadDimPtr = attrs->GetAttrPointer<int64_t>(ATTR_ORIGINAL_HEAD_DIM);
    OP_CHECK_IF(originalHeadDimPtr == nullptr, OP_LOGE(context->GetNodeName(), "attr originalHeadDim is nullptr"),
                return ge::GRAPH_FAILED);
    const int64_t* tileGridPtr = attrs->GetAttrPointer<int64_t>(ATTR_TILE_GRID);
    OP_CHECK_IF(tileGridPtr == nullptr, OP_LOGE(context->GetNodeName(), "attr tileGrid is nullptr"),
                return ge::GRAPH_FAILED);
    const int64_t* maxSelectNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SELECT_NUM);
    OP_CHECK_IF(maxSelectNumPtr == nullptr, OP_LOGE(context->GetNodeName(), "attr maxSelectNum is nullptr"),
                return ge::GRAPH_FAILED);
    const int64_t validKvSeqLen = *validKvPtr;
    const int64_t originalHeadDim = *originalHeadDimPtr;
    if (originalHeadDim <= 0 || static_cast<uint64_t>(originalHeadDim) > headDim) {
        OP_LOGE(context->GetNodeName(), "originalHeadDim must be in (0, %lu], got %ld", headDim, originalHeadDim);
        return ge::GRAPH_FAILED;
    }
    if (validKvSeqLen <= 0 || static_cast<uint64_t>(validKvSeqLen) > seqLen) {
        OP_LOGE(context->GetNodeName(), "validKvSeqLen must be in (0, %lu], got %ld", seqLen, validKvSeqLen);
        return ge::GRAPH_FAILED;
    }
    const uint64_t tileGrid = static_cast<uint64_t>(*tileGridPtr);
    const uint64_t tilesT = tileGrid & TILE_GRID_MASK;
    const uint64_t tilesH = (tileGrid >> TILE_GRID_BITS) & TILE_GRID_MASK;
    const uint64_t tilesW = (tileGrid >> (2 * TILE_GRID_BITS)) & TILE_GRID_MASK;
    const uint64_t imageBlocks = tilesT * tilesH * tilesW;
    const uint64_t imageTokens = imageBlocks * SPARSE_BLOCK;
    // The image tokens are whole tiles; at most one block (the text slots) follows them.
    if (imageBlocks == 0 || imageTokens > static_cast<uint64_t>(validKvSeqLen) || blocks > imageBlocks + 1) {
        OP_LOGE(context->GetNodeName(), "tileGrid %lux%lux%lu does not match sequence %lu and validKvSeqLen %ld",
                tilesT, tilesH, tilesW, seqLen, validKvSeqLen);
        return ge::GRAPH_FAILED;
    }
    const int64_t maxSelectNum = *maxSelectNumPtr;
    if (maxSelectNum <= 0 || static_cast<uint64_t>(maxSelectNum) > blocks) {
        OP_LOGE(context->GetNodeName(), "maxSelectNum must be in (0, %lu], got %ld", blocks, maxSelectNum);
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "platformInfo is nullptr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint64_t aicNum = std::max<uint64_t>(1, ascendcPlatform.GetCoreNumAic());
    uint64_t l2Bytes = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Bytes);
    const bool largeL2 = l2Bytes >= LARGE_L2_BYTES;

    const uint64_t tasksPerBatch = QueryTasksPerHead(seqLen) * heads;
    const uint64_t totalTaskNum = batch * tasksPerBatch;
    const uint64_t blockDim = std::min(aicNum, totalTaskNum);
    // KV block lists: one slot of selectCap entries per (head, image block), then the text block's list.
    const uint64_t selectCap = CeilDiv(static_cast<uint64_t>(maxSelectNum), IDX_PER_C0) * IDX_PER_C0;
    const uint64_t listRowNum = heads * imageBlocks;
    const uint64_t textListEntries = (blocks > imageBlocks) ? CeilDiv(blocks, IDX_PER_C0) * IDX_PER_C0 : 0;
    const uint64_t selectIdxSize = (listRowNum * selectCap + textListEntries) * sizeof(int32_t);
    const uint64_t listRowsPerSubCore = CeilDiv(listRowNum, blockDim * AIV_PER_AIC);
    const uint64_t listSubCoreNum = CeilDiv(listRowNum, listRowsPerSubCore);
    const uint64_t mm1OutSize = blockDim * WORKSPACE_BLOCK_ELEMS * sizeof(float) * WORKSPACE_STAGES;
    const uint64_t smOnlineOutSize = blockDim * WORKSPACE_BLOCK_ELEMS * sizeof(uint16_t) * WORKSPACE_STAGES;
    const uint64_t mm2OutSize = blockDim * WORKSPACE_BLOCK_ELEMS * sizeof(float) * WORKSPACE_STAGES;
    const uint64_t updateSize = mm2OutSize;

    uint32_t scaleBits = 0;
    const float scaleValue = *scalePtr;
    std::memcpy(&scaleBits, &scaleValue, sizeof(scaleBits));

    // Zero-padded heads (private copies made by the aclnn layer): the kernel fills the padded value columns with
    // ones, so P*V also yields every row's probability sum and the vector cores skip the row sum. The ones start on
    // a whole C0, so the real head dim must be a multiple of 16. The score scale stays on the FLOAT32 scores:
    // scaling the queries by a power of two is only exact while the product stays normal, so queries near the
    // smallest normal FLOAT16 would lose the bits their scores differ in.
    const bool padded = static_cast<uint64_t>(originalHeadDim) + SUM_COLUMN_PAD <= headDim &&
                        static_cast<uint64_t>(originalHeadDim) % SUM_COLUMN_PAD == 0;
    const uint64_t totalRows = batch * heads * seqLen;
    const uint64_t prepassRowsPerSubCore =
        CeilDiv(CeilDiv(totalRows, blockDim * AIV_PER_AIC), PREPASS_ROWS) * PREPASS_ROWS;

    SlidingTileAttentionTilingData tilingData;
    tilingData.set_batch(batch);
    tilingData.set_numHeads(heads);
    tilingData.set_embeddingSize(headDim);
    tilingData.set_kvBlockSize(KV_BLOCK_SIZE);
    tilingData.set_firstBatchTaskNum(tasksPerBatch);
    tilingData.set_totalTaskNum(totalTaskNum);
    tilingData.set_blockShapeX(SPARSE_BLOCK);
    tilingData.set_blockShapeY(SPARSE_BLOCK);
    tilingData.set_seqLen(seqLen);
    tilingData.set_validKvSeqLen(static_cast<uint64_t>(validKvSeqLen));
    tilingData.set_mm1OutSize(mm1OutSize);
    tilingData.set_smOnlineOutSize(smOnlineOutSize);
    tilingData.set_mm2OutSize(mm2OutSize);
    tilingData.set_updateSize(updateSize);
    tilingData.set_selectIdxSize(selectIdxSize);
    tilingData.set_scaleValueBits(scaleBits);
    tilingData.set_sumColumn(padded ? static_cast<uint64_t>(originalHeadDim) : 0);
    tilingData.set_totalRows(totalRows);
    tilingData.set_prepassRowsPerSubCore(prepassRowsPerSubCore);
    tilingData.set_stage2Delay(largeL2 ? STAGE2_DELAY_LARGE_L2 : STAGE2_DELAY_SMALL_L2);
    tilingData.set_warmQueries(largeL2 ? 1 : 0);
    tilingData.set_tilesT(tilesT);
    tilingData.set_tilesH(tilesH);
    tilingData.set_tilesW(tilesW);
    tilingData.set_imageBlocks(imageBlocks);
    tilingData.set_blockNum(blocks);
    tilingData.set_textSelect(static_cast<uint64_t>(validKvSeqLen) > imageTokens ? 1 : 0);
    tilingData.set_selectCap(selectCap);
    tilingData.set_listRowNum(listRowNum);
    tilingData.set_listRowsPerSubCore(listRowsPerSubCore);
    tilingData.set_listSubCoreNum(listSubCoreNum);

    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr, OP_LOGE(context->GetNodeName(), "raw tiling data is nullptr"),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    context->SetTilingKey(0);
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "workspace is nullptr"),
                return ge::GRAPH_FAILED);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + mm1OutSize + smOnlineOutSize + mm2OutSize + updateSize +
                    selectIdxSize + SYNC_BYTES;
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForSlidingTileAttention(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SlidingTileAttention)
    .Tiling(TilingSlidingTileAttention)
    .TilingParse<SlidingTileAttentionCompileInfo>(TilingPrepareForSlidingTileAttention);
} // namespace optiling
