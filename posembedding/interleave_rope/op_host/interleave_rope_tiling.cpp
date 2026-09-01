/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file interleave_rope_tiling.cpp
 * \brief
 */
#include "interleave_rope_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "log/log.h"

namespace optiling {

constexpr int64_t X_INDEX = 0;
constexpr int64_t COS_INDEX = 1;
constexpr int64_t SIN_INDEX = 2;
constexpr int64_t INPUT_NUM = 3;
constexpr int64_t OUTPUT_NUM = 1;
constexpr int64_t SHAPE_IDX_B = 0;
constexpr int64_t SHAPE_IDX_N = 1;
constexpr int64_t SHAPE_IDX_S = 2;
constexpr int64_t SHAPE_IDX_D = 3;
constexpr int64_t DIM_SIZE = 4;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;
constexpr int64_t DEFAULT_HIDDEN_DIM = 64;
constexpr int64_t DEFAULT_BATCH_SIZE = 32;
constexpr int64_t DEFAULT_NUM_HEAD = 32;
constexpr int64_t DEFAULT_BATCH_PER_BLOCK = 4;
constexpr uint64_t SPLIT_BATCH = 0;
constexpr uint64_t SPLIT_NS = 1;
constexpr uint64_t SPLIT_N = 2;
constexpr uint64_t SPLIT_S = 3;
constexpr uint64_t SPLIT_BNS = 4;

constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_FOUR = 4;
constexpr int64_t NUM_SIX = 6;
constexpr int64_t NUM_TEN = 10;
constexpr int64_t MIN_SEQUANCE_LEN = 512;

constexpr uint64_t FIXED_BNSD_B11D_TILINGKEY = 1000;
constexpr uint64_t B11D_TILINGKEY = 2000;
constexpr uint64_t B1SD_TILINGKEY = 3000;
constexpr uint64_t BN1D_TILINGKEY = 4000;
constexpr uint64_t BNSD_TILINGKEY = 5000;

static std::tuple<int64_t, int64_t, int64_t, int64_t> GetShapeTuple(
    const gert::TilingContext* context, const int64_t index = 0)
{
    const gert::StorageShape* shapePtr = context->GetInputShape(index);
    OP_CHECK_IF(
        shapePtr == nullptr, OP_LOGE(context, "Shape is nullptr."), return std::make_tuple(0, 0, 0, 0));

    // check shape length is DIM_SIZE
    OP_CHECK_IF(
        shapePtr->GetStorageShape().GetDimNum() != DIM_SIZE,
        OP_LOGE(context, "Shape must be (B,N,S,D)."), return std::make_tuple(0, 0, 0, 0));

    return std::make_tuple(
        shapePtr->GetStorageShape().GetDim(SHAPE_IDX_B), shapePtr->GetStorageShape().GetDim(SHAPE_IDX_N),
        shapePtr->GetStorageShape().GetDim(SHAPE_IDX_S), shapePtr->GetStorageShape().GetDim(SHAPE_IDX_D));
}

ge::graphStatus InterleaveRopeTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const InterleaveRopeCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_, "CompileInfo is nullptr."),
            return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize_ = ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InterleaveRopeTiling::GetShapeAttrsInfo()
{
    auto xShapeTuple = GetShapeTuple(context_, X_INDEX);
    auto cosShapeTuple = GetShapeTuple(context_, COS_INDEX);

    batchSize_ = std::get<SHAPE_IDX_B>(xShapeTuple);
    numHead_ = std::get<SHAPE_IDX_N>(xShapeTuple);
    seqLength_ = std::get<SHAPE_IDX_S>(xShapeTuple);
    hiddenDim_ = std::get<SHAPE_IDX_D>(xShapeTuple);
    OP_CHECK_IF(
        batchSize_ <= 0, OP_LOGE(context_, "batchSize should be greater than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        numHead_ <= 0, OP_LOGE(context_, "numHead should be greater than 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        seqLength_ <= 0, OP_LOGE(context_, "seqLength should be greater than 0."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        hiddenDim_ <= 0, OP_LOGE(context_, "hiddenDim should be greater than 0."),
        return ge::GRAPH_FAILED);

    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_numHead(numHead_);
    tilingData_.set_seqLength(seqLength_);
    tilingData_.set_hiddenDim(hiddenDim_);

    // Check hiddenDim must be DEFAULT_HIDDEN_DIM
    OP_CHECK_IF(
        hiddenDim_ != DEFAULT_HIDDEN_DIM, OP_LOGE(context_, "hiddenDim must be 64."),
        return ge::GRAPH_FAILED);
    // Check cos sin shape must be same
    OP_CHECK_IF(
        cosShapeTuple != GetShapeTuple(context_, SIN_INDEX),
        OP_LOGE(context_, "cos and sin shape must be the same."), return ge::GRAPH_FAILED);

    // Check cosB must be batchSize
    cosB_ = std::get<SHAPE_IDX_B>(cosShapeTuple);
    OP_CHECK_IF(
        cosB_ != batchSize_, OP_LOGE(context_, "cos or sin batchSize must be the same as x batchSize."),
        return ge::GRAPH_FAILED);
    // Check cosN must be 1
    cosN_ = std::get<SHAPE_IDX_N>(cosShapeTuple);
    OP_CHECK_IF(
        cosN_ != 1, OP_LOGE(context_, "cos or sin numHead must be 1."), return ge::GRAPH_FAILED);
    // Check cosS must 1
    cosS_ = std::get<SHAPE_IDX_S>(cosShapeTuple);
    OP_CHECK_IF(
        cosS_ != seqLength_ && cosS_ != 1,
        OP_LOGE(context_, "cos or sin seqLength must be 1 or the same as x seqLength."),
        return ge::GRAPH_FAILED);
    // Check cosD must be hiddenDim
    int64_t cosD = std::get<SHAPE_IDX_D>(cosShapeTuple);
    OP_CHECK_IF(
        cosD != hiddenDim_, OP_LOGE(context_, "cos or sin hiddenDim must be the same as x hiddenDim."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InterleaveRopeTiling::SplitBlockForFixBNS()
{
    int64_t batchsPerBlock = DEFAULT_BATCH_PER_BLOCK;
    tilingData_.set_batchsPerBlock(batchsPerBlock);
    tilingData_.set_batchsLastBlock(batchsPerBlock);
    tilingData_.set_numBlocks(batchSize_ / batchsPerBlock);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InterleaveRopeTiling::SplitBlockForBatch()
{
    int64_t batchsPerBlock = Ops::Base::CeilDiv(batchSize_, static_cast<int64_t>(coreNum_));
    int64_t needBlocks = Ops::Base::CeilDiv(batchSize_, batchsPerBlock);
    int64_t batchsLastBlock = batchSize_ - (needBlocks - 1) * batchsPerBlock;
    tilingData_.set_batchsPerBlock(batchsPerBlock);
    tilingData_.set_batchsLastBlock(batchsLastBlock);
    tilingData_.set_numBlocks(needBlocks);

    if (tilingKey_ == B11D_TILINGKEY) {
        int64_t NS = numHead_ * seqLength_;
        tilingData_.set_hiddenDimCountPerBlock(NS);
        tilingData_.set_hiddenDimCountLastBlock(NS);
        SplitHiddenDim();
        return ge::GRAPH_SUCCESS;
    }

    if (tilingKey_ == B1SD_TILINGKEY) {
        tilingData_.set_hiddenDimCountPerBlock(seqLength_);
        tilingData_.set_hiddenDimCountLastBlock(seqLength_);
        SplitHiddenDim();
        return ge::GRAPH_SUCCESS;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InterleaveRopeTiling::SplitBlockForNS()
{
    tilingData_.set_batchsPerBlock(batchSize_);
    tilingData_.set_batchsLastBlock(batchSize_);

    int64_t NS = numHead_ * seqLength_;
    int64_t NSPerBlock = Ops::Base::CeilDiv(NS, static_cast<int64_t>(coreNum_));
    int64_t needBlocks = Ops::Base::CeilDiv(NS, NSPerBlock);
    int64_t NSLastBlock = NS - (needBlocks - 1) * NSPerBlock;
    tilingData_.set_numBlocks(needBlocks);

    tilingData_.set_hiddenDimCountPerBlock(NSPerBlock);
    tilingData_.set_hiddenDimCountLastBlock(NSLastBlock);
    SplitHiddenDim();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InterleaveRopeTiling::SplitHiddenDim()
{
    int64_t hiddenDimCountPerBlock = tilingData_.get_hiddenDimCountPerBlock();
    int64_t hiddenDimCountLastBlock = tilingData_.get_hiddenDimCountLastBlock();

    int64_t hiddenDimLoopsPerBlock;
    int64_t hiddenDimCountPerLoopPerBlock;
    int64_t hiddenDimCountLastLoopPerBlock;
    SplitHiddenDimInblock(
        hiddenDimCountPerBlock, hiddenDimLoopsPerBlock, hiddenDimCountPerLoopPerBlock, hiddenDimCountLastLoopPerBlock);
    tilingData_.set_hiddenDimLoopsPerBlock(hiddenDimLoopsPerBlock);
    tilingData_.set_hiddenDimCountPerLoopPerBlock(hiddenDimCountPerLoopPerBlock);
    tilingData_.set_hiddenDimCountLastLoopPerBlock(hiddenDimCountLastLoopPerBlock);
    OP_LOGD(
        context_,
        "hiddenDimLoopsPerBlock is: %ld, hiddenDimCountPerLoopPerBlock is: %ld, hiddenDimCountLastLoopPerBlock is: "
        "%ld.",
        hiddenDimLoopsPerBlock, hiddenDimCountPerLoopPerBlock, hiddenDimCountLastLoopPerBlock);

    int64_t hiddenDimLoopsLastBlock;
    int64_t hiddenDimCountPerLoopLastBlock;
    int64_t hiddenDimCountLastLoopLastBlock;
    SplitHiddenDimInblock(
        hiddenDimCountLastBlock, hiddenDimLoopsLastBlock, hiddenDimCountPerLoopLastBlock,
        hiddenDimCountLastLoopLastBlock);
    tilingData_.set_hiddenDimLoopsLastBlock(hiddenDimLoopsLastBlock);
    tilingData_.set_hiddenDimCountPerLoopLastBlock(hiddenDimCountPerLoopLastBlock);
    tilingData_.set_hiddenDimCountLastLoopLastBlock(hiddenDimCountLastLoopLastBlock);
    OP_LOGD(
        context_,
        "hiddenDimLoopsLastBlock is: %ld, hiddenDimCountPerLoopLastBlock is: %ld, hiddenDimCountLastLoopLastBlock is: "
        "%ld.",
        hiddenDimLoopsLastBlock, hiddenDimCountPerLoopLastBlock, hiddenDimCountLastLoopLastBlock);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InterleaveRopeTiling::SplitHiddenDimInblock(
    int64_t hiddenDimCount, int64_t& hiddenDimLoops, int64_t& hiddenDimCountPerLoop, int64_t& hiddenDimCountLastLoop)
{
    if (tilingKey_ == B11D_TILINGKEY) {
        hiddenDimCountPerLoop =
            (ubSize_ - hiddenDim_ * sizeof(float) * NUM_FOUR) / (hiddenDim_ * sizeof(float) * NUM_SIX) / NUM_TWO;
    } else if (tilingKey_ == B1SD_TILINGKEY) {
        hiddenDimCountPerLoop = (ubSize_) / (hiddenDim_ * sizeof(float) * NUM_TEN) / NUM_TWO;
    }
    hiddenDimCountPerLoop = hiddenDimCount > hiddenDimCountPerLoop ? hiddenDimCountPerLoop : hiddenDimCount;
    hiddenDimLoops = Ops::Base::CeilDiv(hiddenDimCount, hiddenDimCountPerLoop);
    hiddenDimCountLastLoop = hiddenDimCount - (hiddenDimLoops - 1) * hiddenDimCountPerLoop;
    return ge::GRAPH_SUCCESS;
}

bool InterleaveRopeTiling::IsCapable()
{
    return true;
}

ge::graphStatus InterleaveRopeTiling::DoOpTiling()
{
    OP_LOGD(context_, "Start DoOpTiling.");

    if (seqLength_ >= MIN_SEQUANCE_LEN && cosS_ == seqLength_) {
        int64_t seqPerBlock = Ops::Base::CeilDiv(seqLength_, static_cast<int64_t>(coreNum_));
        int64_t needBlocks = Ops::Base::CeilDiv(seqLength_, seqPerBlock);
        tilingData_.set_numBlocks(needBlocks);
        tilingKey_ = BN1D_TILINGKEY;
        return ge::GRAPH_SUCCESS;
    }

    if (batchSize_ == DEFAULT_BATCH_SIZE && numHead_ == DEFAULT_NUM_HEAD && seqLength_ == 1 && cosN_ == 1 &&
        cosS_ == 1) {
        tilingKey_ = FIXED_BNSD_B11D_TILINGKEY;
        SplitBlockForFixBNS();
    } else if (cosN_ == 1 && cosS_ == 1) { // bnsd -> b11d
        tilingKey_ = B11D_TILINGKEY;
        if (batchSize_ >= static_cast<int64_t>(coreNum_) || batchSize_ >= numHead_ * seqLength_) {
            tilingData_.set_splitAxis(SPLIT_BATCH);
            SplitBlockForBatch();
        } else {
            tilingData_.set_splitAxis(SPLIT_NS);
            SplitBlockForNS();
        }
    } else if (cosN_ == 1) { // bnsd -> b1sd
        tilingKey_ = B1SD_TILINGKEY;
        tilingData_.set_splitAxis(SPLIT_BATCH);
        SplitBlockForBatch();
    } else {
        OP_LOGE(context_, "DoOpTiling failed, set tilingkey failed.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context_, "DoOpTiling success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InterleaveRopeTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InterleaveRopeTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t InterleaveRopeTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus InterleaveRopeTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(tilingData_.get_numBlocks());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
REGISTER_OPS_TILING_TEMPLATE(InterleaveRope, InterleaveRopeTiling, 1000);

ge::graphStatus Tiling4InterleaveRope(gert::TilingContext* context)
{
    OP_LOGD(context, "TilingForInterleaveRope running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4InterleaveRope(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4InterleaveRope running.");
    auto compileInfo = context->GetCompiledInfo<InterleaveRopeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        compileInfo->coreNum <= 0, OP_LOGE(context, "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    OP_CHECK_IF(
        compileInfo->ubSize <= 0, OP_LOGE(context, "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    OP_LOGD(context, "TilingPrepare4InterleaveRope success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InterleaveRope)
    .Tiling(Tiling4InterleaveRope)
    .TilingParse<InterleaveRopeCompileInfo>(TilingPrepare4InterleaveRope);

} // namespace optiling
