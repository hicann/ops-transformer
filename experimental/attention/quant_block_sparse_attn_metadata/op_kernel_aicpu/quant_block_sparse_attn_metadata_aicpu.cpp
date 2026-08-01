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
 * \file quant_block_sparse_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <utility>
#include "quant_block_sparse_attn_metadata_aicpu.h"

#define KERNEL_STATUS_OK 0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
uint32_t QuantBlockSparseAttnMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    std::vector<SectionInfo> sectionResults{};
    success = ScheduleSections(sectionResults) && GenMetadata(sectionResults);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantBlockSparseAttnMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    context_ = &ctx;
    // input
    sparseSeqLen_ = ctx.Input(static_cast<uint32_t>(ParamId::sparseSeqLen));
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs = GetAttrValue(ctx, "soc_version", socVersion_) &&
                         GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                         GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }
    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "num_heads_q", numHeadsQ_);
    GetAttrValueOpt(ctx, "num_heads_kv", numHeadsKv_);
    GetAttrValueOpt(ctx, "head_dim", headDim_);
    GetAttrValueOpt(ctx, "sparse_block_size_q", sparseBlockSizeQ_);
    GetAttrValueOpt(ctx, "sparse_block_size_k", sparseBlockSizeK_);
    GetAttrValueOpt(ctx, "quant_mode", quantMode_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_sparse_indices", layoutSparseIndices_);
    return (ParamsCheck() && ParamsInit());
}

bool QuantBlockSparseAttnMetadataCpuKernel::ParamsCheck()
{
    return CheckTensorData();
}

bool QuantBlockSparseAttnMetadataCpuKernel::CheckTensorData()
{
    // REQUIRED_INPUT tensor check
    if (sparseSeqLen_ == nullptr) {
        KERNEL_LOG_ERROR("sparseSeqLen is nullptr");
        return false;
    }
    if (sparseSeqLen_->GetTensorShape() == nullptr ||
        (sparseSeqLen_->GetTensorShape()->GetDims() != 2 && sparseSeqLen_->GetTensorShape()->GetDims() != 3)) {
        KERNEL_LOG_ERROR("sparseSeqLen must be 2D or 3D tensor");
        return false;
    }
    if (sparseSeqLen_->GetTensorShape()->GetDims() > 0 && sparseSeqLen_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("sparseSeqLen data is nullptr");
        return false;
    }
    // OUTPUT tensor check
    if (metadata_ == nullptr || metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    return true;
}

bool QuantBlockSparseAttnMetadataCpuKernel::ParamsInit()
{
    if (sparseSeqLen_ == nullptr || sparseSeqLen_->GetTensorShape() == nullptr || sparseSeqLen_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("sparseSeqLen must be a valid tensor");
        return false;
    }
    auto sparseSeqLenShape = sparseSeqLen_->GetTensorShape();
    sparseSeqLenDimNum_ = static_cast<uint32_t>(sparseSeqLenShape->GetDims());
    int64_t sparseSeqLenBatchSize = sparseSeqLenShape->GetDimSize(0);
    if (batchSize_ <= 0) {
        KERNEL_LOG_ERROR("invalid batch size, batchSize=%d", batchSize_);
        return false;
    }
    if (sparseSeqLenBatchSize != static_cast<int64_t>(batchSize_)) {
        KERNEL_LOG_ERROR("sparseSeqLen shape[0] must be batchSize, got shape[0]=%ld, batchSize=%d",
                         sparseSeqLenBatchSize, batchSize_);
        return false;
    }
    mBaseSize_ = static_cast<uint32_t>(std::max(sparseBlockSizeQ_, 1));
    s2BaseSize_ = static_cast<uint32_t>(std::max(sparseBlockSizeK_, 1));
    if (numHeadsKv_ == 0) {
        numHeadsKv_ = numHeadsQ_;
    }
    if (numHeadsKv_ <= 0 || numHeadsQ_ % numHeadsKv_ != 0) {
        KERNEL_LOG_ERROR("numHeadsQ must be divisible by numHeadsKv, numHeadsQ=%d, numHeadsKv=%d", numHeadsQ_,
                         numHeadsKv_);
        return false;
    }
    if (sparseSeqLenDimNum_ == 3U) {
        uint32_t sparseN1Size = static_cast<uint32_t>(sparseSeqLenShape->GetDimSize(1));
        Qbmax_ = static_cast<uint32_t>(sparseSeqLenShape->GetDimSize(2));
        if (sparseN1Size != static_cast<uint32_t>(numHeadsQ_)) {
            KERNEL_LOG_ERROR("sparseSeqLen shape must be [B,N1,Qb], got N1=%u, expected N1=%d", sparseN1Size,
                             numHeadsQ_);
            return false;
        }
    } else if (sparseSeqLenDimNum_ == 2U) {
        Qbmax_ = static_cast<uint32_t>(sparseSeqLenShape->GetDimSize(1));
    } else {
        KERNEL_LOG_ERROR("sparseSeqLen must be 3D [B,N1,Qb] or legacy 2D [B,Qb], but got %u dims", sparseSeqLenDimNum_);
        return false;
    }
    if (numHeadsQ_ <= 0 || aicCoreNum_ <= 0 || Qbmax_ == 0U) {
        KERNEL_LOG_ERROR("invalid split params, batchSize=%d, numHeadsQ=%d, numHeadsKv=%d, aicCoreNum=%d, Qbmax=%u",
                         batchSize_, numHeadsQ_, numHeadsKv_, aicCoreNum_, Qbmax_);
        return false;
    }
    InitBN1BlockInfo();
    return true;
}

uint32_t QuantBlockSparseAttnMetadataCpuKernel::GetRowBlockNum(uint32_t bIdx, uint32_t n1Idx, uint32_t s1Idx) const
{
    if (bIdx >= static_cast<uint32_t>(batchSize_) || n1Idx >= static_cast<uint32_t>(numHeadsQ_) || s1Idx >= Qbmax_ ||
        sparseSeqLen_ == nullptr || sparseSeqLen_->GetData() == nullptr) {
        return 0U;
    }
    const int32_t *sparseSeqLenData = static_cast<const int32_t *>(sparseSeqLen_->GetData());
    uint64_t idx = 0U;
    if (sparseSeqLenDimNum_ == 3U) {
        idx = (static_cast<uint64_t>(bIdx) * static_cast<uint32_t>(numHeadsQ_) + n1Idx) * Qbmax_ + s1Idx;
    } else {
        idx = static_cast<uint64_t>(bIdx) * Qbmax_ + s1Idx;
    }
    return sparseSeqLenData[idx] > 0 ? static_cast<uint32_t>(sparseSeqLenData[idx]) : 0U;
}

void QuantBlockSparseAttnMetadataCpuKernel::InitBN1BlockInfo()
{
    const uint32_t batchSize = static_cast<uint32_t>(batchSize_);
    const uint32_t numHeadsQ = static_cast<uint32_t>(numHeadsQ_);
    const uint32_t totalBN1 = batchSize * numHeadsQ;
    bN1BlockNum_.assign(totalBN1, 0U);
    bN1LastRowBlockNum_.assign(totalBN1, 0U);

    for (uint32_t bIdx = 0U; bIdx < batchSize; ++bIdx) {
        for (uint32_t n1Idx = 0U; n1Idx < numHeadsQ; ++n1Idx) {
            const uint32_t bn1Idx = bIdx * numHeadsQ + n1Idx;
            uint64_t totalBlock = 0U;
            uint32_t lastRowBlock = 0U;
            for (uint32_t s1Idx = 0U; s1Idx < Qbmax_; ++s1Idx) {
                const uint32_t rowBlock = GetRowBlockNum(bIdx, n1Idx, s1Idx);
                totalBlock += rowBlock;
                if (rowBlock > 0U) {
                    lastRowBlock = rowBlock;
                }
            }
            bN1BlockNum_[bn1Idx] = totalBlock;
            bN1LastRowBlockNum_[bn1Idx] = lastRowBlock;
        }
    }
}

uint64_t QuantBlockSparseAttnMetadataCpuKernel::GetBN1BlockNum(uint32_t bIdx, uint32_t n1Idx) const
{
    const uint32_t bn1Idx = bIdx * static_cast<uint32_t>(numHeadsQ_) + n1Idx;
    if (bn1Idx >= bN1BlockNum_.size()) {
        return 0U;
    }
    return bN1BlockNum_[bn1Idx];
}

uint32_t QuantBlockSparseAttnMetadataCpuKernel::CalcValidS1Rows(uint32_t bIdx, uint32_t n1Idx,
                                                                uint32_t &maxS2BlockNum) const
{
    uint32_t validRows = 0U;
    maxS2BlockNum = 0U;
    for (uint32_t s1Idx = 0U; s1Idx < Qbmax_; ++s1Idx) {
        const uint32_t rowBlock = GetRowBlockNum(bIdx, n1Idx, s1Idx);
        if (rowBlock > 0U) {
            validRows++;
            maxS2BlockNum = std::max(maxS2BlockNum, rowBlock);
        }
    }
    return validRows;
}

uint64_t QuantBlockSparseAttnMetadataCpuKernel::CalcBN1Cost(uint32_t bIdx, uint32_t n1Idx) const
{
    constexpr uint32_t BYTES_PER_ELEM_Q = 1U;
    constexpr uint32_t BYTES_PER_ELEM_KV = 1U;
    constexpr uint32_t COST_FACTOR = 2U;

    uint32_t maxS2BlockNum = 0U;
    const uint32_t validS1Rows = CalcValidS1Rows(bIdx, n1Idx, maxS2BlockNum);
    if (validS1Rows == 0U) {
        return 0U;
    }

    const uint32_t headDim = static_cast<uint32_t>(std::max(headDim_, 0));
    const uint64_t s1Cost = static_cast<uint64_t>(validS1Rows) * static_cast<uint32_t>(sparseBlockSizeQ_) *
        headDim * BYTES_PER_ELEM_Q * COST_FACTOR;
    const uint64_t s2Size = static_cast<uint64_t>(maxS2BlockNum) * static_cast<uint32_t>(sparseBlockSizeK_);
    const uint64_t s2Cost = s2Size * headDim * BYTES_PER_ELEM_KV * COST_FACTOR;
    return s1Cost + s2Cost;
}

bool QuantBlockSparseAttnMetadataCpuKernel::CalcSectionBoundaries(std::vector<SectionInfo> &sectionResults) const
{
    constexpr uint64_t L2_BYTE = 120U * 1024U * 1024U;
    const uint32_t totalBN1 = static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsQ_);
    uint32_t sectionStart = 0U;
    uint64_t sectionCost = 0U;
    uint64_t sectionBlockNum = 0U;

    for (uint32_t bn1Idx = 0U; bn1Idx < totalBN1; ++bn1Idx) {
        const uint32_t bIdx = bn1Idx / static_cast<uint32_t>(numHeadsQ_);
        const uint32_t n1Idx = bn1Idx % static_cast<uint32_t>(numHeadsQ_);
        const uint64_t curCost = CalcBN1Cost(bIdx, n1Idx);
        const uint64_t curBlocks = GetBN1BlockNum(bIdx, n1Idx);

        if (sectionCost != 0U && (sectionCost + curCost) > L2_BYTE) {
            SectionInfo section(static_cast<uint32_t>(aicCoreNum_));
            section.bn1Start = sectionStart;
            section.bn1End = bn1Idx;
            section.cost = sectionCost;
            section.blockNum = sectionBlockNum;
            sectionResults.emplace_back(std::move(section));

            sectionStart = bn1Idx;
            sectionCost = 0U;
            sectionBlockNum = 0U;
        }

        sectionCost += curCost;
        sectionBlockNum += curBlocks;
    }

    if (sectionBlockNum != 0U || sectionResults.empty()) {
        SectionInfo section(static_cast<uint32_t>(aicCoreNum_));
        section.bn1Start = sectionStart;
        section.bn1End = totalBN1;
        section.cost = sectionCost;
        section.blockNum = sectionBlockNum;
        sectionResults.emplace_back(std::move(section));
    }
    return true;
}

bool QuantBlockSparseAttnMetadataCpuKernel::AdvanceToValidRow(AssignContext &assignContext) const
{
    if (assignContext.isFinished) {
        return false;
    }

    while (assignContext.curBN1Idx < assignContext.bn1Limit) {
        assignContext.curBIdx = assignContext.curBN1Idx / static_cast<uint32_t>(numHeadsQ_);
        assignContext.curN1Idx = assignContext.curBN1Idx % static_cast<uint32_t>(numHeadsQ_);
        if (assignContext.curS1Idx == 0U && assignContext.bN1Block == 0U) {
            assignContext.bN1Block = GetBN1BlockNum(assignContext.curBIdx, assignContext.curN1Idx);
        }
        while (assignContext.curS1Idx < Qbmax_ &&
               GetRowBlockNum(assignContext.curBIdx, assignContext.curN1Idx, assignContext.curS1Idx) == 0U) {
            assignContext.curS1Idx++;
        }
        if (assignContext.curS1Idx < Qbmax_) {
            return true;
        }
        assignContext.curBN1Idx++;
        assignContext.curN1Idx = 0U;
        assignContext.curS1Idx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.bN1Block = 0U;
    }
    assignContext.isFinished = true;
    return false;
}

void QuantBlockSparseAttnMetadataCpuKernel::AssignByBatch(AssignContext &assignContext) const
{
    if (assignContext.isFinished) {
        return;
    }

    while (assignContext.curBN1Idx < assignContext.bn1Limit) {
        if (!AdvanceToValidRow(assignContext)) {
            return;
        }
        if (assignContext.bN1Block == 0U) {
            assignContext.curBN1Idx++;
            assignContext.curN1Idx = 0U;
            assignContext.curS1Idx = 0U;
            assignContext.curS2Idx = 0U;
            continue;
        }
        uint64_t tolerance = 0U;
        if (assignContext.curBN1Idx < bN1LastRowBlockNum_.size()) {
            tolerance = static_cast<uint64_t>(bN1LastRowBlockNum_[assignContext.curBN1Idx]) / BLOCK_TOLERANCE_RATIO;
        }
        if (assignContext.coreCache.block + assignContext.bN1Block > assignContext.coreCache.blockLimit + tolerance) {
            return;
        }

        assignContext.coreCache.block += assignContext.bN1Block;
        assignContext.curBN1Idx++;
        assignContext.curN1Idx = 0U;
        assignContext.curS1Idx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.bN1Block = 0U;
    }

    assignContext.isFinished = true;
}

void QuantBlockSparseAttnMetadataCpuKernel::AssignByRow(AssignContext &assignContext) const
{
    while (!assignContext.isFinished) {
        if (!AdvanceToValidRow(assignContext)) {
            break;
        }
        uint32_t rowBlock = GetRowBlockNum(assignContext.curBIdx, assignContext.curN1Idx, assignContext.curS1Idx);
        uint64_t tolerance = static_cast<uint64_t>(rowBlock) / BLOCK_TOLERANCE_RATIO;
        if (assignContext.coreCache.block > 0U &&
            assignContext.coreCache.block + rowBlock > assignContext.coreCache.blockLimit + tolerance) {
            return;
        }
        assignContext.coreCache.block += rowBlock;
        assignContext.bN1Block = assignContext.bN1Block > rowBlock ? assignContext.bN1Block - rowBlock : 0U;
        assignContext.curS1Idx++;
    }
}

void QuantBlockSparseAttnMetadataCpuKernel::ScheduleFa(uint32_t bn1Start, uint32_t bn1End, uint64_t blockNum,
                                                       SplitResult &result)
{
    AssignContext assignContext{};
    assignContext.unassignedBlock = blockNum;
    assignContext.curBN1Idx = bn1Start;
    assignContext.curBIdx = bn1Start / static_cast<uint32_t>(numHeadsQ_);
    assignContext.curN1Idx = bn1Start % static_cast<uint32_t>(numHeadsQ_);
    assignContext.curS1Idx = 0U;
    assignContext.curS2Idx = 0U;
    assignContext.bN1Block = 0U;
    assignContext.bn1Limit = bn1End;

    for (uint32_t i = 0; i < static_cast<uint32_t>(aicCoreNum_); i++) {
        if (assignContext.isFinished || assignContext.unassignedBlock == 0U) {
            break;
        }
        if (!AdvanceToValidRow(assignContext)) {
            break;
        }
        assignContext.curCoreIdx = i;

        assignContext.coreCache = {};
        uint32_t remainingCore = static_cast<uint32_t>(aicCoreNum_) - i;
        assignContext.coreCache.blockLimit = (assignContext.unassignedBlock + remainingCore - 1U) / remainingCore;
        uint32_t curRowBlock = GetRowBlockNum(assignContext.curBIdx, assignContext.curN1Idx, assignContext.curS1Idx);
        assignContext.coreCache.blockLimit =
            std::max(assignContext.coreCache.blockLimit, static_cast<uint64_t>(curRowBlock));

        AssignByBatch(assignContext);
        AssignByRow(assignContext);

        result.bN1End[i] = assignContext.curBN1Idx;
        result.s1End[i] = assignContext.curS1Idx;
        result.s2End[i] = 0U;
        result.firstFdDataWorkspaceIdx[i] = 0U;
        result.maxBlock = std::max(result.maxBlock, assignContext.coreCache.block);
        if (assignContext.unassignedBlock > assignContext.coreCache.block) {
            assignContext.unassignedBlock -= assignContext.coreCache.block;
        } else {
            assignContext.unassignedBlock = 0U;
        }
        result.usedCoreNum = i + 1U;
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U);
}

bool QuantBlockSparseAttnMetadataCpuKernel::ScheduleSections(std::vector<SectionInfo> &sectionResults)
{
    CalcSectionBoundaries(sectionResults);
    if (sectionResults.empty()) {
        SectionInfo emptySection(static_cast<uint32_t>(aicCoreNum_));
        emptySection.bn1Start = 0U;
        emptySection.bn1End = static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsQ_);
        emptySection.blockNum = 0U;
        emptySection.splitResult.usedCoreNum = 1U;
        emptySection.splitResult.bN1End[0] = emptySection.bn1End;
        sectionResults.emplace_back(std::move(emptySection));
        return true;
    }
    for (auto &section : sectionResults) {
        if (section.blockNum == 0U) {
            section.splitResult.usedCoreNum = 1U;
            section.splitResult.bN1End[0] = section.bn1End;
            section.splitResult.s1End[0] = 0U;
            section.splitResult.s2End[0] = 0U;
            section.splitResult.firstFdDataWorkspaceIdx[0] = 0U;
            continue;
        }
        ScheduleFa(section.bn1Start, section.bn1End, section.blockNum, section.splitResult);
    }
    return true;
}

bool QuantBlockSparseAttnMetadataCpuKernel::GenMetadata(const std::vector<SectionInfo> &sectionResults)
{
    if (metadata_ == nullptr || metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata is empty");
        return false;
    }
    const uint32_t sectionNum = static_cast<uint32_t>(sectionResults.size());
    optiling::detail::qbsaMetaData qbsaMetadata(metadata_->GetData(), sectionNum);
    qbsaMetadata.Clear();
    qbsaMetadata.SetHeadMetadata(optiling::QBSA_HEAD_SECTION_NUM_INDEX, sectionNum);

    for (uint32_t secIdx = 0; secIdx < sectionNum; ++secIdx) {
        const auto &section = sectionResults[secIdx];
        const auto &splitRes = section.splitResult;
        for (uint32_t i = 0; i < splitRes.usedCoreNum; i++) {
            qbsaMetadata.SetQbsaMetadata(secIdx, i, optiling::QBSA_CORE_ENABLE_INDEX, 1U);
            uint32_t bn1Start = (i == 0) ? section.bn1Start : splitRes.bN1End[i - 1];
            uint32_t bn1End = splitRes.bN1End[i];
            uint32_t s1Start = (i == 0U) ? 0U : splitRes.s1End[i - 1U];
            uint32_t s2Start = (i == 0U) ? 0U : splitRes.s2End[i - 1U];

            qbsaMetadata.SetQbsaMetadata(secIdx, i, optiling::QBSA_BN1_START_INDEX, bn1Start);
            qbsaMetadata.SetQbsaMetadata(secIdx, i, optiling::QBSA_M_START_INDEX, s1Start);
            qbsaMetadata.SetQbsaMetadata(secIdx, i, optiling::QBSA_S2_START_INDEX, s2Start);
            qbsaMetadata.SetQbsaMetadata(secIdx, i, optiling::QBSA_BN1_END_INDEX, bn1End);
            qbsaMetadata.SetQbsaMetadata(secIdx, i, optiling::QBSA_M_END_INDEX, splitRes.s1End[i]);
            qbsaMetadata.SetQbsaMetadata(secIdx, i, optiling::QBSA_S2_END_INDEX, splitRes.s2End[i]);
            qbsaMetadata.SetQbsaMetadata(secIdx, i, optiling::QBSA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                         splitRes.firstFdDataWorkspaceIdx[i]);
        }
    }
    return true;
}

namespace {
static const char *kernelType = "QuantBlockSparseAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, QuantBlockSparseAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
