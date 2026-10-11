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
 * \file moe_inplace_index_add_tiling.cc
 * \brief
 */

#include "moe_inplace_index_add_tiling.h"
#include "moe_inplace_index_add_determinstic_tiling.h"
#include <cstdint>

namespace optiling {
constexpr int64_t SIMD_DB_BUFFER = 2L;
constexpr int64_t SIZE_TWO = 2L;

constexpr int64_t DETERMINSTIC_TILING_KEY = 200000L;
constexpr int64_t DETERMINSTIC_TILING_NOT_QUANT_KEY = 200001L;
constexpr int64_t RESERVE_SIZE = 192L;
constexpr int64_t MIN_BLOCK_SIZE = 1024L;
constexpr int64_t ALIGN_SIZE = 32L;
constexpr int64_t INDICES_MIN_BLOCK_SIZE = 1024L;
constexpr int64_t MIN_HANDLE_SIZE = 128L;
constexpr int64_t INT32_BYTES = 4L;
constexpr int64_t FP32_BYTES = 4L;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16L * 1024L * 1024L;

constexpr uint64_t DETERMINSTIC_FP32_TILING_KEY = 300000UL;
constexpr uint64_t DETERMINSTIC_FP16_TILING_KEY = 300001UL;
constexpr uint64_t DETERMINSTIC_BF16_TILING_KEY = 300002UL;

static constexpr float THRESHOLD_FACTOR = 0.1;

bool MoeInplaceIndexAddDeterminsticTiling::IsCapable()
{
    if (isDeterminstic_ == 1) {
        return true;
    }
    return false;
}

static inline int64_t safeDiv(const int64_t a, const int64_t b)
{
    return b == 0 ? 0 : a / b;
}

void MoeInplaceIndexAddDeterminsticTiling::DoOpTilingForDeterminsticSplitPre()
{
    int64_t alignNum = safeDiv(ALIGN_SIZE, varTypeSize_);
    int64_t afterAxisAlign = Ops::Base::CeilAlign(afterAxis_, alignNum);
    int64_t halfUbSize = (ubSize_ - RESERVE_SIZE) / SIMD_DB_BUFFER;

    /* split preaxis */
    eachCorePreAxisCount_ = Ops::Base::CeilDiv(preAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(preAxis_, eachCorePreAxisCount_);
    tailCorePreAxisCount_ = preAxis_ - eachCorePreAxisCount_ * (usedCoreNumBefore_ - 1);
    int64_t oneBlockSize = indicesTypeSize_ + varTypeSize_ * afterAxisAlign * SIZE_TWO;
    if (halfUbSize < oneBlockSize) {
        ubIndexFactor_ = 1;
        afterAxisFactor_ = (halfUbSize - indicesTypeSize_) / (SIZE_TWO * varTypeSize_);
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
        ubIndexFactor_ = halfUbSize / oneBlockSize;
        ubIndexFactor_ = Ops::Base::FloorAlign(ubIndexFactor_ * indicesTypeSize_, ALIGN_SIZE) / indicesTypeSize_;
    }
    /* 每个核分的update相同 */
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    /* 每个核处理的indices相同 */
    indicesLoopSize_ = Ops::Base::CeilDiv(indicesAxis_, ubIndexFactor_);
    indiceAxisTailNum_ = indicesAxis_ - (indicesLoopSize_ - 1) * ubIndexFactor_;
    isSplitPreAxis_ = 1;
}

void MoeInplaceIndexAddDeterminsticTiling::DoOpTilingForDeterminsticSplitAfter()
{
    int64_t halfUbSize = (ubSize_ - RESERVE_SIZE) / SIMD_DB_BUFFER;
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;

    eachCorePreAxisCount_ = preAxis_;
    tailCorePreAxisCount_ = preAxis_;
    /* split afterAxis */
    eachCoreAfterAxisCount_ = Ops::Base::CeilDiv(afterAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(afterAxis_, eachCoreAfterAxisCount_);
    tailCoreAfterAxisCount_ = afterAxis_ - eachCoreAfterAxisCount_ * (usedCoreNumBefore_ - 1);
    if (eachCoreAfterAxisCount_ * varTypeSize_ > (halfUbSize - MIN_BLOCK_SIZE)) {
        int64_t indicesUbSize = std::min(MIN_BLOCK_SIZE, indicesAxis_ * indicesTypeSize_);
        ubIndexFactor_ = Ops::Base::CeilAlign(indicesUbSize, ALIGN_SIZE) / indicesTypeSize_;
        afterAxisFactor_ =
            (halfUbSize - ubIndexFactor_ * indicesTypeSize_) / (ubIndexFactor_ * varTypeSize_ * SIZE_TWO);
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, alignNum);
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(eachCoreAfterAxisCount_, alignNum);
        ubIndexFactor_ = halfUbSize / (afterAxisFactor_ * varTypeSize_ * SIZE_TWO + indicesTypeSize_);
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

int64_t MoeInplaceIndexAddDeterminsticTiling::GetRestAvailableSize(int64_t sampleNum, int64_t valueTypeBytes,
                                                                   int64_t originalSize, int64_t postAxisSize,
                                                                   ge::DataType idType)
{
    auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t occupy = Ops::Base::CeilAlign(sampleNum * indicesTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * (indicesTypeSize_ + SIZE_TWO * ALIGN_SIZE), ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * INT32_BYTES, ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * (INT32_BYTES * SIZE_TWO), ubBlock) +
                     Ops::Base::CeilAlign(sampleNum * indicesTypeSize_, ubBlock) +
                     sampleNum * Ops::Base::CeilAlign((varTypeSize_)*postAxisSize, ubBlock) +
                     sampleNum * Ops::Base::CeilAlign((FP32_BYTES)*postAxisSize, ubBlock) +
                     sampleNum * Ops::Base::CeilAlign((FP32_BYTES)*postAxisSize, ubBlock) +
                     GetSortTmpSize(idType, sampleNum, false);
    return originalSize - occupy;
}

void MoeInplaceIndexAddDeterminsticTiling::DoOpTilingForDeterminstic()
{
    int64_t afterAxisAlignFp32 = Ops::Base::CeilAlign(afterAxis_ * FP32_BYTES, ALIGN_SIZE) / FP32_BYTES;
    int64_t halfUbSize = (ubSize_ - RESERVE_SIZE) / SIMD_DB_BUFFER;

    /* 优先分pre和after */
    int64_t splitCoreNumThresh = totalCoreNum_ / SIZE_TWO;
    if ((preAxis_ > splitCoreNumThresh) || (afterAxis_ > splitCoreNumThresh) || (indicesAxis_ <= splitCoreNumThresh)) {
        if (afterAxis_ >= preAxis_) {
            DoOpTilingForDeterminsticSplitAfter();
        } else {
            DoOpTilingForDeterminsticSplitPre();
        }
        return;
    }
    if (indicesAxis_ > splitCoreNumThresh) {
        /* split indices分核 */
        eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
        usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
        tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);

        /* varAxis分核，用于最后计算反量化 */
        eachCoreVarCount_ = Ops::Base::CeilDiv(preAxis_ * varInAxis_, totalCoreNum_);
        usedCoreNumAfter_ = Ops::Base::CeilDiv(preAxis_ * varInAxis_, eachCoreVarCount_);
        tailCoreVarCount_ = preAxis_ * varInAxis_ - eachCoreVarCount_ * (usedCoreNumAfter_ - 1);

        /* first step:搬入多少行indices,就搬入相同行数的updates:
         * ubIndexFactor_: indicesindicesQue + (sortIndicesQue + 2 * shiftOfset) + originIdxQue + (uniqueIdCntQue_ + 1)
         * + updateSumIdxQue_, ubIndexFactor_ * afterAxisAlign: updatesQue_ + updatesCastQue_ + updateSumQue_
         */
        int64_t indicesSize = indicesTypeSize_ + (indicesTypeSize_ + SIZE_TWO * ALIGN_SIZE) + INT32_BYTES +
                              (INT32_BYTES * SIZE_TWO) + indicesTypeSize_;
        int64_t oneBlockSize = indicesSize + (varTypeSize_ + FP32_BYTES + FP32_BYTES) * afterAxis_;
        ubIndexFactor_ = halfUbSize / oneBlockSize;

        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --ubIndexFactor_;
            restSize = GetRestAvailableSize(ubIndexFactor_, varTypeSize_, halfUbSize, afterAxis_, indicesDtype_);
        }
        if (ubIndexFactor_ > indicesAxis_) {
            ubIndexFactor_ = indicesAxis_;
        }

        /* second step */
        /* sumIdx, sumQue + rValueQue + sumQuantaQue */
        oneBlockSize = indicesTypeSize_ + (FP32_BYTES + FP32_BYTES + INT32_BYTES) * afterAxis_;
        ubQuantaIndxFactor_ = halfUbSize / oneBlockSize;

        restSize = static_cast<int64_t>(-1);
        auto ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
        while (restSize <= 0) {
            --ubQuantaIndxFactor_;
            int64_t occupy = Ops::Base::CeilAlign(ubQuantaIndxFactor_ * indicesTypeSize_, ubBlock) +
                             ubQuantaIndxFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock) +
                             ubQuantaIndxFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock) +
                             ubQuantaIndxFactor_ * Ops::Base::CeilAlign(INT32_BYTES * afterAxis_, ubBlock);
            restSize = halfUbSize - occupy;
        }
        if (ubQuantaIndxFactor_ > indicesAxis_) {
            ubQuantaIndxFactor_ = indicesAxis_;
        }

        /* third step */
        /* sumQuanToIntQue + rValueQue_ + invQuanDataQue_ */
        oneBlockSize = (INT32_BYTES + FP32_BYTES + FP32_BYTES) * afterAxisAlignFp32;
        ubVarFactor_ = halfUbSize / oneBlockSize;
        restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --ubVarFactor_;
            int64_t occupy = ubQuantaIndxFactor_ * Ops::Base::CeilAlign(INT32_BYTES * afterAxis_, ubBlock) +
                             ubQuantaIndxFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock) +
                             ubQuantaIndxFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock);
            restSize = halfUbSize - occupy;
        }
        if (ubVarFactor_ > eachCoreVarCount_) {
            ubVarFactor_ = eachCoreVarCount_;
        }

        float xValue = static_cast<float>(indicesAxis_) / varInAxis_;
        if (xValue < THRESHOLD_FACTOR) {
            isOpti_ = 1;
        }

        /* the optimized third step*/
        oneBlockSize = indicesTypeSize_ + (INT32_BYTES + FP32_BYTES + FP32_BYTES) * afterAxisAlignFp32;
        ubVarOptiFactor_ = halfUbSize / oneBlockSize;
        restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --ubVarOptiFactor_;
            int64_t occupy = Ops::Base::CeilAlign(ubQuantaIndxFactor_ * indicesTypeSize_, ubBlock) +
                             ubQuantaIndxFactor_ * Ops::Base::CeilAlign(INT32_BYTES * afterAxis_, ubBlock) +
                             ubQuantaIndxFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock) +
                             ubQuantaIndxFactor_ * Ops::Base::CeilAlign(FP32_BYTES * afterAxis_, ubBlock);
            restSize = halfUbSize - occupy;
        }
        if (ubVarOptiFactor_ > eachCoreVarCount_) {
            ubVarOptiFactor_ = eachCoreVarCount_;
        }
    }
}

void MoeInplaceIndexAddDeterminsticTiling::SetTilingData()
{
    tilingData_.set_preAxis(preAxis_);
    tilingData_.set_varInAxis(varInAxis_);
    tilingData_.set_updatesInAxis(updatesInAxis_);
    tilingData_.set_afterAxis(afterAxis_);
    tilingData_.set_usedCoreNumBefore(usedCoreNumBefore_);
    tilingData_.set_usedCoreNumAfter(usedCoreNumAfter_);
    tilingData_.set_eachCorePreAxisCount(eachCorePreAxisCount_);
    tilingData_.set_tailCorePreAxisCount(tailCorePreAxisCount_);
    tilingData_.set_eachCoreAfterAxisCount(eachCoreAfterAxisCount_);
    tilingData_.set_tailCoreAfterAxisCount(tailCoreAfterAxisCount_);

    tilingData_.set_updateLoopSize(updateLoopSize_);
    tilingData_.set_updateTailNum(updateTailNum_);
    tilingData_.set_indicesLoopSize(indicesLoopSize_);
    tilingData_.set_indiceAxisTailNum(indiceAxisTailNum_);
    tilingData_.set_isSplitPreAxis(isSplitPreAxis_);
    tilingData_.set_tailUpdateLoopSize(tailUpdateLoopSize_);
    tilingData_.set_tailUpdateAxisNum(tailUpdateAxisNum_);
    tilingData_.set_isSplitAfterAxis(isSplitAfterAxis_);
    tilingData_.set_eachCoreIndexCount(eachCoreIndexCount_);
    tilingData_.set_tailCoreIndexCount(tailCoreIndexCount_);
    tilingData_.set_eachCoreVarCount(eachCoreVarCount_);
    tilingData_.set_tailCoreVarCount(tailCoreVarCount_);
    tilingData_.set_ubIndexFactor(ubIndexFactor_);
    tilingData_.set_afterAxisFactor(afterAxisFactor_);
    tilingData_.set_ubQuantaIndxFactor(ubQuantaIndxFactor_);
    tilingData_.set_ubVarFactor(ubVarFactor_);
    tilingData_.set_isWithAlpha(withAlpha_);
    tilingData_.set_isDeterminstic(isDeterminstic_);
    tilingData_.set_ubVarOptiFactor(ubVarOptiFactor_);
    tilingData_.set_isOpti(isOpti_);
    return;
}

ge::graphStatus MoeInplaceIndexAddDeterminsticTiling::DoOpTiling()
{
    if (isDeterminstic_ == 1) {
        usedCoreNum_ = totalCoreNum_;
        DoOpTilingForDeterminstic();
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddDeterminsticTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInplaceIndexAddDeterminsticTiling::GetTilingKey() const
{
    uint64_t tilingKey = DETERMINSTIC_FP32_TILING_KEY;
    if (dtype_ == ge::DT_FLOAT16) {
        tilingKey = DETERMINSTIC_FP16_TILING_KEY;
    } else if (dtype_ == ge::DT_BF16) {
        tilingKey = DETERMINSTIC_BF16_TILING_KEY;
    }
    return tilingKey;
}

ge::graphStatus MoeInplaceIndexAddDeterminsticTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    /* 非量化不需要开workspace */
    if (isDeterminstic_ == 1 && isSplitPreAxis_ == 0 && isSplitAfterAxis_ == 0) {
        int64_t rCoutWsSize = preAxis_ * varInAxis_ * INT32_BYTES;
        int64_t rValueWsSize = varAxis_ * FP32_BYTES;
        int64_t sumQuantaIntWsSize = varAxis_ * INT32_BYTES;
        ;
        int64_t sumWsSize = totalCoreNum_ * eachCoreIndexCount_ * afterAxis_ * FP32_BYTES;
        int64_t sumIdxWsSize = totalCoreNum_ * eachCoreIndexCount_ * indicesTypeSize_;

        workspaceSize_ += (rCoutWsSize + rValueWsSize + sumQuantaIntWsSize + sumWsSize + sumIdxWsSize);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddDeterminsticTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingKey_ = GetTilingKey();
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_);
    MOE_OP_TILING_CHECK((res != ge::GRAPH_SUCCESS),
                        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                            "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
                        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeInplaceIndexAddDeterminsticTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "preAxis: " << tilingData_.get_preAxis();
    info << ", varInAxis: " << tilingData_.get_varInAxis();
    info << ", updatesInAxis: " << tilingData_.get_updatesInAxis();
    info << ", afterAxis: " << tilingData_.get_afterAxis();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", tilingKey: " << tilingKey_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(MoeInplaceIndexAdd, MoeInplaceIndexAddDeterminsticTiling, 100);
} // namespace optiling
