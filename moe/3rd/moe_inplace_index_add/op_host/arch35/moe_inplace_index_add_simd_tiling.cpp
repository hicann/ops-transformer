/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_inplace_index_add_simd_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "op_host/tiling_templates_registry.h"
#include "moe_inplace_index_add_tiling.h"
#include "moe_inplace_index_add_simd_tiling.h"

namespace optiling {
constexpr int64_t SIMD_DB_BUFFER = 2;
constexpr int64_t SIZE_TWO = 2;
constexpr int64_t ALIGN_SIZE = 32;
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t RESERVE_SIZE = 128;
constexpr int64_t AFTER_SPLIT_THRESHOLD = 128;
constexpr int64_t AFTER_HANDLE_THRESHOLD = 1024;
constexpr int64_t INDICES_MIN_BLOCK_SIZE = 8192;
constexpr int64_t GM_ALIGN = 512;

constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16L * 1024L * 1024L;

static const std::set<ge::DataType> VAR_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT64, ge::DT_INT32,
                                                 ge::DT_INT16, ge::DT_INT8,    ge::DT_UINT8, ge::DT_BOOL};

bool MoeInplaceIndexAddSimdTiling::IsCapable()
{
    if (isSimdNoSort_ == 1) {
        return true;
    }
    return false;
}

int64_t MoeInplaceIndexAddSimdTiling::GetIndicesAlignBlockSize(int64_t indicesFactor)
{
    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t occupy = Ops::Base::CeilAlign(indicesFactor * indicesTypeSize_, ubBlock);
    return occupy;
}

int64_t MoeInplaceIndexAddSimdTiling::GetAfterAlignBlockSize(int64_t indicesFactor, int64_t afterFactor)
{
    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t occupy = indicesFactor * Ops::Base::CeilAlign((varTypeSize_)*afterFactor, ubBlock);
    return occupy;
}

void MoeInplaceIndexAddSimdTiling::DoOpTilingSplitPre()
{
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t afterAxisAlign = Ops::Base::CeilAlign(afterAxis_, alignNum);
    int64_t halfUbSize = (ubSize_ - RESERVE_SIZE) / SIMD_DB_BUFFER;

    /* split preaxis */
    eachCorePreAxisCount_ = Ops::Base::CeilDiv(preAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(preAxis_, eachCorePreAxisCount_);
    tailCorePreAxisCount_ = preAxis_ - eachCorePreAxisCount_ * (usedCoreNumBefore_ - 1);
    auto transTypeSize = ge::GetSizeByDataType(ge::DT_INT32);
    int64_t oneBlockSize = indicesTypeSize_ + (varTypeSize_ + transTypeSize) * afterAxisAlign;

    if (halfUbSize < oneBlockSize) {
        ubIndexFactor_ = 1;
        afterAxisFactor_ = (halfUbSize - indicesTypeSize_) / (transTypeSize + varTypeSize_);
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
        ubIndexFactor_ = halfUbSize / oneBlockSize;
        int64_t restSize = static_cast<int64_t>(-1);
        ubIndexFactor_ += 1;
        while (restSize <= 0) {
            --ubIndexFactor_;
            restSize = halfUbSize - (GetIndicesAlignBlockSize(ubIndexFactor_) +
                                     GetAfterAlignBlockSize(ubIndexFactor_, afterAxisFactor_));
        }
        ubIndexFactor_ = ubIndexFactor_ == 0 ? 1 : ubIndexFactor_;
    }
    indicesLoopSize_ = Ops::Base::CeilDiv(indicesAxis_, ubIndexFactor_);
    indiceAxisTailNum_ = indicesAxis_ - (indicesLoopSize_ - 1) * ubIndexFactor_;

    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    isSplitPreAxis_ = 1;
}

void MoeInplaceIndexAddSimdTiling::DoOpTilingSplitIndices()
{
    auto ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t alignNum = ubBlock / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / SIMD_DB_BUFFER);

    /* split indices分核 */
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);

    int64_t indicesAlignSize = GetIndicesAlignBlockSize(ubIndexFactor_);
    int64_t afterAlignSize = GetAfterAlignBlockSize(ubIndexFactor_, afterAxis_);
    int64_t oneBlockSize = indicesAlignSize + afterAlignSize;

    if (oneBlockSize > halfUbSize) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, GetIndicesAlignBlockSize(indicesAxis_));
        ubIndexFactor_ = Ops::Base::CeilAlign(indicesUbSize, static_cast<int64_t>(ubBlock)) / indicesTypeSize_;
        ubIndexFactor_ = ubIndexFactor_ > eachCoreIndexCount_ ? eachCoreIndexCount_ : ubIndexFactor_;

        int64_t aviableSizeForUpdate = halfUbSize - GetIndicesAlignBlockSize(ubIndexFactor_);
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            afterAxisFactor_ -= alignNum;
            restSize = aviableSizeForUpdate - GetAfterAlignBlockSize(ubIndexFactor_, afterAxisFactor_);
        }
        afterAxisFactor_ = afterAxisFactor_ == 0 ? 1 : afterAxisFactor_;
    } else {
        afterAxisFactor_ = std::min(AFTER_HANDLE_THRESHOLD / varTypeSize_, afterAxis_);
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxisFactor_, alignNum);

        ubIndexFactor_ = halfUbSize / GetAfterAlignBlockSize(1, afterAxisFactor_);
        int64_t restSize = static_cast<int64_t>(-1);
        ubIndexFactor_ += 1;
        while (restSize <= 0) {
            --ubIndexFactor_;
            restSize = halfUbSize - (GetIndicesAlignBlockSize(ubIndexFactor_) +
                                     GetAfterAlignBlockSize(ubIndexFactor_, afterAxisFactor_));
        }
        ubIndexFactor_ = ubIndexFactor_ > eachCoreIndexCount_ ? eachCoreIndexCount_ : ubIndexFactor_;
    }

    /* 每个核分的update相同 */
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    mainCoreIndicesLoop_ = Ops::Base::CeilDiv(eachCoreIndexCount_, ubIndexFactor_);
    tailCoreIndicesLoop_ = Ops::Base::CeilDiv(tailCoreIndexCount_, ubIndexFactor_);

    mainCoreTailIndices_ = eachCoreIndexCount_ - (mainCoreIndicesLoop_ - 1) * ubIndexFactor_;
    tailCoreTailIndices_ = tailCoreIndexCount_ - (tailCoreIndicesLoop_ - 1) * ubIndexFactor_;

    isSplitIndicesAxis_ = 1;
}

void MoeInplaceIndexAddSimdTiling::DoOpTilingSplitAfter()
{
    int64_t halfUbSize = (ubSize_ - RESERVE_SIZE) / SIMD_DB_BUFFER;
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;

    eachCorePreAxisCount_ = preAxis_;
    tailCorePreAxisCount_ = preAxis_;
    /* split afterAxis */
    eachCoreAfterAxisCount_ = Ops::Base::CeilDiv(afterAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(afterAxis_, eachCoreAfterAxisCount_);
    tailCoreAfterAxisCount_ = afterAxis_ - eachCoreAfterAxisCount_ * (usedCoreNumBefore_ - 1);
    auto transTypeSize = ge::GetSizeByDataType(ge::DT_INT32);
    if (eachCoreAfterAxisCount_ * (varTypeSize_ + transTypeSize) > (halfUbSize - INDICES_MIN_BLOCK_SIZE)) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesTypeSize_);
        ubIndexFactor_ = Ops::Base::CeilAlign(indicesUbSize, ALIGN_SIZE) / indicesTypeSize_;
        afterAxisFactor_ =
            (halfUbSize - ubIndexFactor_ * indicesTypeSize_) / (ubIndexFactor_ * (varTypeSize_ + transTypeSize));
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(eachCoreAfterAxisCount_, alignNum);
        ubIndexFactor_ = halfUbSize / (afterAxisFactor_ * (varTypeSize_ + transTypeSize) + indicesTypeSize_);
        ubIndexFactor_ = std::max(ubIndexFactor_, 1L);
    }

    /* 每个核分的indices相同 */
    indicesLoopSize_ = Ops::Base::CeilDiv(indicesAxis_, ubIndexFactor_);
    indiceAxisTailNum_ = indicesAxis_ - (indicesLoopSize_ - 1) * ubIndexFactor_;

    /* 主核循环次数 */
    updateLoopSize_ = Ops::Base::CeilDiv(eachCoreAfterAxisCount_, afterAxisFactor_);
    /* 主核尾loop处理afterAxis大小 */
    updateTailNum_ = eachCoreAfterAxisCount_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    /* 尾核循环次数 */
    tailUpdateLoopSize_ = Ops::Base::CeilDiv(tailCoreAfterAxisCount_, afterAxisFactor_);
    /* 尾核尾loop处理afterAxis大小 */
    tailUpdateAxisNum_ = tailCoreAfterAxisCount_ - (tailUpdateLoopSize_ - 1) * afterAxisFactor_;
    isSplitAfterAxis_ = 1;
}

ge::graphStatus MoeInplaceIndexAddSimdTiling::DoOpTiling()
{
    usedCoreNum_ = totalCoreNum_;
    if (afterAxis_ * varTypeSize_ >= AFTER_SPLIT_THRESHOLD * usedCoreNum_) {
        DoOpTilingSplitAfter();
    } else if (indicesAxis_ > preAxis_) {
        DoOpTilingSplitIndices();
    } else {
        DoOpTilingSplitPre();
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddSimdTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInplaceIndexAddSimdTiling::GetTilingKey() const
{
    uint64_t tilingKey = 400000;
    return tilingKey;
}

ge::graphStatus MoeInplaceIndexAddSimdTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    if (castTypeSize_ != 0) {
        int64_t varWsSize = Ops::Base::CeilAlign(varAxis_ * castTypeSize_, GM_ALIGN);
        workspaceSize_ += varWsSize;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddSimdTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    context_->SetBlockDim(totalCoreNum_);
    context_->SetTilingKey(tilingKey_);
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_);
    tilingKey_ = GetTilingKey();
    workspaces[0] = workspaceSize_;
    MOE_OP_TILING_CHECK((res != ge::GRAPH_SUCCESS),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                            "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
                        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeInplaceIndexAddSimdTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "preAxis: " << tilingData_.get_preAxis();
    info << ", varInAxis: " << tilingData_.get_varInAxis();
    info << ", updatesInAxis: " << tilingData_.get_updatesInAxis();
    info << ", afterAxis: " << tilingData_.get_afterAxis();
    info << ", indicesStride: " << tilingData_.get_indicesStride();
    info << ", tilingKey: " << tilingKey_;
    info << ", usedCoreNum: " << usedCoreNum_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

void MoeInplaceIndexAddSimdTiling::SetTilingData()
{
    tilingData_.set_preAxis(preAxis_);
    tilingData_.set_varInAxis(varInAxis_);
    tilingData_.set_updatesInAxis(updatesInAxis_);
    tilingData_.set_afterAxis(afterAxis_);
    tilingData_.set_eachCorePreAxisCount(eachCorePreAxisCount_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_ubIndexFactor(ubIndexFactor_);
    tilingData_.set_usedCoreNumBefore(usedCoreNumBefore_);
    tilingData_.set_tailCorePreAxisCount(tailCorePreAxisCount_);
    tilingData_.set_eachCoreAfterAxisCount(eachCoreAfterAxisCount_);
    tilingData_.set_tailCoreAfterAxisCount(tailCoreAfterAxisCount_);
    tilingData_.set_indicesStride(indicesStride_);
    tilingData_.set_updateLoopSize(updateLoopSize_);
    tilingData_.set_eachCoreIndexCount(eachCoreIndexCount_);
    tilingData_.set_tailCoreIndexCount(tailCoreIndexCount_);
    tilingData_.set_mainCoreIndicesLoop(mainCoreIndicesLoop_);
    tilingData_.set_tailCoreIndicesLoop(tailCoreIndicesLoop_);
    tilingData_.set_mainCoreTailIndices(mainCoreTailIndices_);
    tilingData_.set_tailCoreTailIndices(tailCoreTailIndices_);
    tilingData_.set_isSplitIndicesAxis(isSplitIndicesAxis_);
    tilingData_.set_indicesLoopSize(indicesLoopSize_);
    tilingData_.set_updateTailNum(updateTailNum_);
    tilingData_.set_indiceAxisTailNum(indiceAxisTailNum_);
    tilingData_.set_isSplitPreAxis(isSplitPreAxis_);
    tilingData_.set_tailUpdateAxisNum(tailUpdateAxisNum_);
    tilingData_.set_isSplitAfterAxis(isSplitAfterAxis_);
    tilingData_.set_tailUpdateLoopSize(tailUpdateLoopSize_);
    tilingData_.set_isWithAlpha(withAlpha_);
    tilingData_.set_afterAxisFactor(afterAxisFactor_);
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeInplaceIndexAdd, MoeInplaceIndexAddSimdTiling, 2);
} // namespace optiling
