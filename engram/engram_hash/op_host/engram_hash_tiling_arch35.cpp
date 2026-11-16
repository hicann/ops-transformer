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
 * \file engram_hash_tiling_arch35.cpp
 * \brief EngramHash tiling 实现 (arch35 / ascend950)
 *
 * 输入范围约束:
 *   1. 4 个输入与输出 dtype 固定 int32/int64/int32/int32 -> int32, 且连续;
 *   2. ids.ndim == 2 且 N >= 2; vocab.ndim == 3 且 L >= 1, vocab.dim1 == N - 1, tables >= 1;
 *      mult.shape == (L, N); offsets.shape == (L, C);
 *   3. num_tokens == 0 允许, 直接返回空输出 [L, 0, C];
 *   4. 无 num_tokens 规模上限 (tileTokens 自适应);
 *   5. 层数与输出列数无额外限制 (L 不限核数, C > 1024 由 VF stride 循环覆盖)。
 */

#include <algorithm>
#include <cstdint>
#include <sstream>
#include "engram_hash_tiling_arch35.h"
#include "log/log.h"

using namespace ge;

namespace optiling {

namespace {
constexpr int64_t AlignUp32(int64_t value)
{
    return (value + 31) & ~static_cast<int64_t>(31);
}
} // namespace

ge::graphStatus EngramHashTilingArch35::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<EngramHashCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = ubSizePlatform;
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
    }
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context_->GetNodeName(), "invalid coreNum %lu", coreNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= 0, OP_LOGE(context_->GetNodeName(), "invalid ubSize %lu", ubSize_), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramHashTilingArch35::ValidateShapes(const gert::Shape& idsStorage, const gert::Shape& multStorage,
                                                       const gert::Shape& vocabStorage, const gert::Shape& offsStorage,
                                                       const gert::Shape& outStorage)
{
    const char* nodeName = context_->GetNodeName();

    // ids 须为 [num_tokens, max_ngram_size] 且 N >= 2。
    OP_CHECK_IF(idsStorage.GetDimNum() != IDS_DIM_NUM,
                OP_LOGE(nodeName, "ngram_token_ids must be 2D [num_tokens, max_ngram_size], got dimNum=%zu",
                        idsStorage.GetDimNum()),
                return ge::GRAPH_FAILED);
    numTokens_ = idsStorage.GetDim(0);
    maxNgramSize_ = idsStorage.GetDim(1);
    OP_CHECK_IF(maxNgramSize_ < MIN_NGRAM_SIZE,
                OP_LOGE(nodeName, "max_ngram_size must be at least 2, got %ld", maxNgramSize_),
                return ge::GRAPH_FAILED);

    // vocab 须为 [L, N - 1, tables] 且 L >= 1, tables >= 1。
    OP_CHECK_IF(vocabStorage.GetDimNum() != VOCAB_DIM_NUM,
                OP_LOGE(nodeName,
                        "vocab_sizes must be 3D [num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram], "
                        "got dimNum=%zu",
                        vocabStorage.GetDimNum()),
                return ge::GRAPH_FAILED);
    numNgramLayers_ = vocabStorage.GetDim(0);
    OP_CHECK_IF(numNgramLayers_ <= 0, OP_LOGE(nodeName, "num_ngram_layers must be positive, got %ld", numNgramLayers_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        vocabStorage.GetDim(1) != maxNgramSize_ - 1,
        OP_LOGE(nodeName, "vocab_sizes dimension 1 must be %ld, got %ld", maxNgramSize_ - 1, vocabStorage.GetDim(1)),
        return ge::GRAPH_FAILED);
    numTables_ = vocabStorage.GetDim(2);
    OP_CHECK_IF(numTables_ <= 0, OP_LOGE(nodeName, "num_embed_table_per_ngram must be positive, got %ld", numTables_),
                return ge::GRAPH_FAILED);

    numOutCols_ = (maxNgramSize_ - 1) * numTables_;

    // mult 须为 [L, N]。
    OP_CHECK_IF(multStorage.GetDimNum() != MULTIPLIERS_DIM_NUM || multStorage.GetDim(0) != numNgramLayers_ ||
                    multStorage.GetDim(1) != maxNgramSize_,
                OP_LOGE(nodeName, "multipliers must have shape [%ld, %ld], got dimNum=%zu", numNgramLayers_,
                        maxNgramSize_, multStorage.GetDimNum()),
                return ge::GRAPH_FAILED);

    // offsets 须为 [L, C]。
    OP_CHECK_IF(offsStorage.GetDimNum() != OFFSETS_DIM_NUM || offsStorage.GetDim(0) != numNgramLayers_ ||
                    offsStorage.GetDim(1) != numOutCols_,
                OP_LOGE(nodeName, "offsets must have shape [%ld, %ld], got dimNum=%zu", numNgramLayers_, numOutCols_,
                        offsStorage.GetDimNum()),
                return ge::GRAPH_FAILED);

    // 输出须为 [L, T, C]。
    OP_CHECK_IF(outStorage.GetDimNum() != 3 || outStorage.GetDim(0) != numNgramLayers_ ||
                    outStorage.GetDim(1) != numTokens_ || outStorage.GetDim(2) != numOutCols_,
                OP_LOGE(nodeName, "hash_indices must have shape [%ld, %ld, %ld], got dimNum=%zu", numNgramLayers_,
                        numTokens_, numOutCols_, outStorage.GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramHashTilingArch35::ValidateDtypes()
{
    const char* nodeName = context_->GetNodeName();

    auto idsDesc = context_->GetInputDesc(NGRAM_TOKEN_IDS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, idsDesc);
    OP_CHECK_IF(
        idsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "ngram_token_ids dtype only support INT32, got %d", static_cast<int>(idsDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    auto multDesc = context_->GetInputDesc(MULTIPLIERS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, multDesc);
    OP_CHECK_IF(
        multDesc->GetDataType() != ge::DT_INT64,
        OP_LOGE(nodeName, "multipliers dtype only support INT64, got %d", static_cast<int>(multDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    auto vocabDesc = context_->GetInputDesc(VOCAB_SIZES_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vocabDesc);
    OP_CHECK_IF(
        vocabDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "vocab_sizes dtype only support INT32, got %d", static_cast<int>(vocabDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    auto offsDesc = context_->GetInputDesc(OFFSETS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsDesc);
    OP_CHECK_IF(
        offsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "offsets dtype only support INT32, got %d", static_cast<int>(offsDesc->GetDataType())),
        return ge::GRAPH_FAILED);

    auto outDesc = context_->GetOutputDesc(HASH_INDICES_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);
    OP_CHECK_IF(
        outDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(nodeName, "hash_indices dtype only support INT32, got %d", static_cast<int>(outDesc->GetDataType())),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramHashTilingArch35::GetShapeAndDtypeInfo()
{
    // 注意: GetInputShape 返回 StorageShape, 须再 ->GetStorageShape() 得到 gert::Shape 后才能 GetDim。
    auto idsShape = context_->GetInputShape(NGRAM_TOKEN_IDS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, idsShape);
    auto multShape = context_->GetInputShape(MULTIPLIERS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, multShape);
    auto vocabShape = context_->GetInputShape(VOCAB_SIZES_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vocabShape);
    auto offsShape = context_->GetInputShape(OFFSETS_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsShape);
    auto outShape = context_->GetOutputShape(HASH_INDICES_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outShape);

    OP_CHECK_IF(ValidateShapes(idsShape->GetStorageShape(), multShape->GetStorageShape(), vocabShape->GetStorageShape(),
                               offsShape->GetStorageShape(), outShape->GetStorageShape()) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ValidateShapes failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtypes() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "ValidateDtypes failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramHashTilingArch35::DoOpTiling()
{
    // 空输入: T == 0 不启动 kernel, blockDim 固定 1, kernel 侧按 numTokens == 0 直接返回。
    if (numTokens_ == 0) {
        usedCoreNums_ = 1;
        tilingData_.set_numTokens(0);
        tilingData_.set_maxNgramSize(maxNgramSize_);
        tilingData_.set_numNgramLayers(numNgramLayers_);
        tilingData_.set_numTables(numTables_);
        tilingData_.set_numOutCols(numOutCols_);
        tilingData_.set_tileTokens(0);
        tilingData_.set_colChunk(0);
        tilingData_.set_numTilesPerLayer(0);
        tilingData_.set_totalBlocks(0);
        tilingData_.set_usedCoreNums(usedCoreNums_);
        return ge::GRAPH_SUCCESS;
    }

    // tileTokens = max(1, 1024 // C): C <= 1024 时单轮 VF 覆盖整 tile 的一个列块;
    // C > 1024 时 tileTokens = 1, 由 kernel 的 VF 内跨元素 stride 循环覆盖块内全部元素。
    int64_t tileTokens = MAX_SIMT_THREADS / numOutCols_;
    tileTokens = std::max(tileTokens, static_cast<int64_t>(1));

    const int64_t coresPerLayerCap =
        std::max(static_cast<int64_t>(1), static_cast<int64_t>(coreNum_ / numNgramLayers_));
    const int64_t multBytesLocal = maxNgramSize_ * static_cast<int64_t>(sizeof(int64_t));

    const int64_t multAligned = AlignUp32(multBytesLocal);
    // 布局整体再留 32B 保护边距 (窗口实际大小约 8KB, 不顶满)。
    const int64_t windowBudget = UB_BUDGET_BYTES - 32 - multAligned;
    const int64_t sliceBytesFull = 2 * AlignUp32(numOutCols_ * static_cast<int64_t>(sizeof(int32_t)));
    const int64_t idElemBytes = maxNgramSize_ * static_cast<int64_t>(sizeof(int32_t));
    int64_t ttCap = 1;
    int64_t ccCap = numOutCols_;
    if (windowBudget - sliceBytesFull >= 32) {
        // 单列块方案: align32(tileTokens*N*4) <= 窗口 - mult - 2*align32(C*4)
        ttCap = std::max((windowBudget - sliceBytesFull - 31) / idElemBytes, static_cast<int64_t>(1));
    } else {
        // 多列块方案: 预算对半, 保证 align32 取整后仍不超窗
        const int64_t idsHalf = windowBudget / 2;
        ttCap = std::max((idsHalf - 31) / idElemBytes, static_cast<int64_t>(1));
        const int64_t sliceHalf = (windowBudget - AlignUp32(ttCap * idElemBytes)) / 2;
        ccCap = std::max((sliceHalf - 31) / static_cast<int64_t>(sizeof(int32_t)), static_cast<int64_t>(1));
    }
    int64_t tilesPerLayerTarget;
    const int64_t perCoreElems = (numTokens_ * numNgramLayers_ * numOutCols_ + coreNum_ - 1) / coreNum_;
    if (perCoreElems <= MAX_SIMT_THREADS) {
        tilesPerLayerTarget = std::min(coresPerLayerCap, numTokens_);
    } else if (numTokens_ <= coresPerLayerCap * ttCap) {
        // 单轮可容纳: 每层 tiles = coresPerLayer (blocks == coreNum, 每核恰 1 块)
        tilesPerLayerTarget = coresPerLayerCap;
    } else {
        // 多轮: 全局块数 L*minTiles 量化为 coreNum 整数倍, 每核轮次完全均衡
        const int64_t minTiles = (numTokens_ + ttCap - 1) / ttCap;
        const int64_t rounds = (numNgramLayers_ * minTiles + coreNum_ - 1) / coreNum_;
        tilesPerLayerTarget = rounds * coreNum_ / numNgramLayers_;
    }
    const int64_t ttFit = (numTokens_ + tilesPerLayerTarget - 1) / tilesPerLayerTarget;
    if (perCoreElems <= MAX_SIMT_THREADS) {
        // 小/中 T: 允许 tileTokens < 1024//C, 多小块填满核 (每块 ~1 次 stride 迭代)
        tileTokens = std::max(std::min(ttFit, ttCap), static_cast<int64_t>(1));
    } else {
        tileTokens = std::max(tileTokens, std::min(ttFit, ttCap));
    }

    tileTokens = std::min(tileTokens, ttCap);

    const int64_t idsBytes = tileTokens * idElemBytes;
    const int64_t multBytes = multBytesLocal;
    int64_t colChunk = std::min(std::max(ccCap, static_cast<int64_t>(1)), numOutCols_);
    OP_CHECK_IF(
        AlignUp32(idsBytes) + AlignUp32(multBytes) + AlignUp32(colChunk * static_cast<int64_t>(sizeof(int32_t))) >=
            UB_BUDGET_BYTES,
        OP_LOGE(context_->GetNodeName(),
                "the n-gram configuration exceeds the VF call __ubuf__ argument encoding limit (<8KB): "
                "max_ngram_size=%ld, tileTokens=%ld, colChunk=%ld",
                maxNgramSize_, tileTokens, colChunk),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(idsBytes + multBytes > static_cast<int64_t>(ubSize_),
                OP_LOGE(context_->GetNodeName(),
                        "the n-gram configuration is too large for UB: max_ngram_size=%ld requires %ld bytes",
                        maxNgramSize_, idsBytes + multBytes),
                return ge::GRAPH_FAILED);

    const int64_t numTilesPerLayer = (numTokens_ + tileTokens - 1) / tileTokens;
    const int64_t totalBlocks = numNgramLayers_ * numTilesPerLayer;
    usedCoreNums_ = std::min(totalBlocks, static_cast<int64_t>(coreNum_));
    OP_CHECK_IF(usedCoreNums_ <= 0,
                OP_LOGE(context_->GetNodeName(), "usedCoreNums %ld must be positive", usedCoreNums_),
                return ge::GRAPH_FAILED);

    tilingData_.set_numTokens(numTokens_);
    tilingData_.set_maxNgramSize(maxNgramSize_);
    tilingData_.set_numNgramLayers(numNgramLayers_);
    tilingData_.set_numTables(numTables_);
    tilingData_.set_numOutCols(numOutCols_);
    tilingData_.set_tileTokens(tileTokens);
    tilingData_.set_colChunk(colChunk);
    tilingData_.set_numTilesPerLayer(numTilesPerLayer);
    tilingData_.set_totalBlocks(totalBlocks);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EngramHashTilingArch35::PostTiling()
{
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNums_));
    // 单一 tiling key: 输入输出 dtype 组合固定 (int32 x int64 x int32 x int32 -> int32)。
    context_->SetTilingKey(0);
    context_->SetDynUBufSize(static_cast<uint32_t>(ubSize_ - 32 * 1024));

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "get workspaces nullptr"),
                return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(DEFAULT_WORKSPACE_SIZE);

    OP_CHECK_IF(context_->GetRawTilingData() == nullptr, OP_LOGE(context_->GetNodeName(), "get tilingdata nullptr"),
                return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    DumpTilingInfo();
    return ge::GRAPH_SUCCESS;
}

void EngramHashTilingArch35::DumpTilingInfo()
{
    std::ostringstream info;
    info << "numTokens: " << tilingData_.get_numTokens();
    info << ", maxNgramSize: " << tilingData_.get_maxNgramSize();
    info << ", numNgramLayers: " << tilingData_.get_numNgramLayers();
    info << ", numTables: " << tilingData_.get_numTables();
    info << ", numOutCols: " << tilingData_.get_numOutCols();
    info << ", tileTokens: " << tilingData_.get_tileTokens();
    info << ", colChunk: " << tilingData_.get_colChunk();
    info << ", numTilesPerLayer: " << tilingData_.get_numTilesPerLayer();
    info << ", totalBlocks: " << tilingData_.get_totalBlocks();
    info << ", usedCoreNums: " << tilingData_.get_usedCoreNums();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus EngramHashTilingArch35::RunTiling()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("engram_hash_tiling", "failed to get tiling context."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAndDtypeInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "GetShapeAndDtypeInfo failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(DoOpTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "DoOpTiling failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(PostTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "PostTiling failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForEngramHash(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("EngramHash", "Tiling context is null"),
                return ge::GRAPH_FAILED);

    EngramHashTilingArch35 tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForEngramHash(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramHash).Tiling(TilingForEngramHash).TilingParse<EngramHashCompileInfo>(TilingPrepareForEngramHash);

} // namespace optiling
