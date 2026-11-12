/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lightning_indexer_v2_metadata_aicpu.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "status.h"

using namespace optiling;

namespace aicpu::liv2_metadata {
uint32_t LightningIndexerV2MetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const auto soc = ProcessSocVersion();
    if (soc == ValidSocVersion::ASCEND910B) {
        auto* metadata = static_cast<optiling::detail::LiV2MetadataArch22*>(metadata_->GetData());
        // A3 普通路径自行分核，Fast 不准入时输出禁用标志并正常返回
        metadata->fastMetadata = {};
        GenFastMetadataArch22();
        return KERNEL_STATUS_OK;
    }

    SplitResult splitRes{aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
    if (success && soc == ValidSocVersion::ASCEND950) {
        GenFastMetadata();
    }
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool LightningIndexerV2MetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
    cuSeqlensK_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenK));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedQ));
    sequsedK_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedK));
    cmpResidualK_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpResidualK));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs = GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                         GetAttrValue(ctx, "aiv_core_num", aivCoreNum_) &&
                         GetAttrValue(ctx, "soc_version", socVersion_) &&
                         GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_k", numHeadsK_) &&
                         GetAttrValue(ctx, "head_dim", headDim_) && GetAttrValue(ctx, "topk", topk_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_k", maxSeqlenK_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_k", layoutK_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);

    return (ParamsCheck() && ParamsInit());
}

bool LightningIndexerV2MetadataCpuKernel::ParamsCheck()
{
    if (metadata_ == nullptr) {
        KERNEL_LOG_ERROR("Output metadata is nullptr");
        return false;
    } else if (metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("Output metadata data is nullptr");
        return false;
    }
    const int32_t batchSize = GetQueryBatchSize();
    return CheckCuSeqlensQ(batchSize) && CheckCuSeqlensK(batchSize) && CheckSequsedQ(batchSize) &&
           CheckSequsedK(batchSize) && CheckCmpResidualK(batchSize);
}

bool LightningIndexerV2MetadataCpuKernel::CheckCuSeqlensQ(int32_t batchSize) const
{
    if (layoutQ_ != "TND" || cuSeqlensQ_ == nullptr || cuSeqlensQ_->GetData() == nullptr) {
        return true;
    }
    const int32_t* cuSeqlensQPtr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
    if (cuSeqlensQPtr[0] != 0) {
        KERNEL_LOG_ERROR("The first element of cu_seqlens_q should be 0, but got %d", cuSeqlensQPtr[0]);
        return false;
    }
    for (int32_t i = 1; i < batchSize + 1; ++i) {
        if (cuSeqlensQPtr[i - 1] > cuSeqlensQPtr[i]) {
            KERNEL_LOG_ERROR("The elements in cu_seqlens_q must be in ascending order, "
                             "but got cu_seqlens_q[%d] = %d, cu_seqlens_q[%d] = %d",
                             i - 1, cuSeqlensQPtr[i - 1], i, cuSeqlensQPtr[i]);
            return false;
        }
    }
    return true;
}

bool LightningIndexerV2MetadataCpuKernel::CheckCuSeqlensK(int32_t batchSize) const
{
    if (layoutK_ != "TND" || cuSeqlensK_ == nullptr || cuSeqlensK_->GetData() == nullptr) {
        return true;
    }
    const int32_t* cuSeqlensKPtr = static_cast<const int32_t*>(cuSeqlensK_->GetData());
    if (cuSeqlensKPtr[0] != 0) {
        KERNEL_LOG_ERROR("The first element of cu_seqlens_k should be 0, but got %d", cuSeqlensKPtr[0]);
        return false;
    }
    for (int32_t i = 1; i < batchSize + 1; ++i) {
        if (cuSeqlensKPtr[i - 1] > cuSeqlensKPtr[i]) {
            KERNEL_LOG_ERROR("The elements in cu_seqlens_k must be in ascending order, "
                             "but got cu_seqlens_k[%d] = %d, cu_seqlens_k[%d] = %d",
                             i - 1, cuSeqlensKPtr[i - 1], i, cuSeqlensKPtr[i]);
            return false;
        }
    }
    return true;
}

bool LightningIndexerV2MetadataCpuKernel::CheckSequsedQ(int32_t batchSize) const
{
    if (sequsedQ_ == nullptr || sequsedQ_->GetData() == nullptr) {
        return true;
    }
    const int32_t* sequsedQPtr = static_cast<const int32_t*>(sequsedQ_->GetData());
    const int32_t* cuSeqlensQPtr = (layoutQ_ == "TND" && cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) ?
                                       static_cast<const int32_t*>(cuSeqlensQ_->GetData()) :
                                       nullptr;
    for (int32_t i = 0; i < batchSize; ++i) {
        if (sequsedQPtr[i] < 0) {
            KERNEL_LOG_ERROR("The elements in seqused_q should be >= 0, but got seqused_q[%d] = %d", i, sequsedQPtr[i]);
            return false;
        }
        if (layoutQ_ == "BSND" && sequsedQPtr[i] > maxSeqlenQ_) {
            KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than max_seqlen_q %d, "
                             "but got seqused_q[%d] = %d",
                             maxSeqlenQ_, i, sequsedQPtr[i]);
            return false;
        }
        if (cuSeqlensQPtr != nullptr) {
            const int32_t seqLen = cuSeqlensQPtr[i + 1] - cuSeqlensQPtr[i];
            if (sequsedQPtr[i] > seqLen) {
                KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than the sequence length "
                                 "from cu_seqlens_q %d, but got seqused_q[%d] = %d",
                                 seqLen, i, sequsedQPtr[i]);
                return false;
            }
        }
    }
    return true;
}

bool LightningIndexerV2MetadataCpuKernel::CheckSequsedK(int32_t batchSize) const
{
    if (sequsedK_ == nullptr || sequsedK_->GetData() == nullptr) {
        return true;
    }
    const int32_t* sequsedKPtr = static_cast<const int32_t*>(sequsedK_->GetData());
    const int32_t* cuSeqlensKPtr = (layoutK_ == "TND" && cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) ?
                                       static_cast<const int32_t*>(cuSeqlensK_->GetData()) :
                                       nullptr;
    for (int32_t i = 0; i < batchSize; ++i) {
        if (sequsedKPtr[i] < 0) {
            KERNEL_LOG_ERROR("The elements in seqused_k should be >= 0, but got seqused_k[%d] = %d", i, sequsedKPtr[i]);
            return false;
        }
        if (layoutK_ == "BSND" && sequsedKPtr[i] > maxSeqlenK_) {
            KERNEL_LOG_ERROR("The elements in seqused_k should not be greater than max_seqlen_k %d, "
                             "but got seqused_k[%d] = %d",
                             maxSeqlenK_, i, sequsedKPtr[i]);
            return false;
        }
        if (cuSeqlensKPtr != nullptr) {
            const int32_t seqLen = cuSeqlensKPtr[i + 1] - cuSeqlensKPtr[i];
            if (sequsedKPtr[i] > seqLen) {
                KERNEL_LOG_ERROR("The elements in seqused_k should not be greater than the sequence length "
                                 "from cu_seqlens_k %d, but got seqused_k[%d] = %d",
                                 seqLen, i, sequsedKPtr[i]);
                return false;
            }
        }
    }
    return true;
}

bool LightningIndexerV2MetadataCpuKernel::CheckCmpResidualK(int32_t batchSize) const
{
    if (cmpResidualK_ == nullptr || cmpResidualK_->GetData() == nullptr) {
        return true;
    }
    const int32_t* cmpResidualKPtr = static_cast<const int32_t*>(cmpResidualK_->GetData());
    for (int32_t i = 0; i < batchSize; ++i) {
        if (cmpResidualKPtr[i] < 0 || cmpResidualKPtr[i] >= cmpRatio_) {
            KERNEL_LOG_ERROR("The elements in cmp_residual_k should be in [0, cmpRatio_(%d)), but got "
                             "cmp_residual_k[%d] = %d",
                             cmpRatio_, i, cmpResidualKPtr[i]);
            return false;
        }
    }
    return true;
}
int32_t LightningIndexerV2MetadataCpuKernel::GetQueryBatchSize()
{
    // 1. 如果sequsedQ_传了，使用sequsedQ_获取BatchSize
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        if (sequsedQ_->GetTensorShape() != nullptr) {
            return sequsedQ_->GetTensorShape()->GetDimSize(0);
        }
    }
    // 2. sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_获取BatchSize
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            if (cuSeqlensQ_->GetTensorShape() != nullptr) {
                return cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
            }
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用batchSize_
    return batchSize_;
}

ValidSocVersion LightningIndexerV2MetadataCpuKernel::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    const std::string ascend960 = "Ascend960DT";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    } else if (socVersion_.find(ascend960) != std::string::npos) {
        return ValidSocVersion::ASCEND960;
    } else {
        return ValidSocVersion::ASCEND910B;
    }
}

bool LightningIndexerV2MetadataCpuKernel::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    auto mode = static_cast<SparseMode>(maskMode_);
    if (mode == SparseMode::RIGHT_DOWN_CAUSAL) {
        attentionMode_ = 1;
        preToken_ = INT64_MAX;
    } else if (mode == SparseMode::DEFAULT_MASK) {
        attentionMode_ = 0;
    } else if (mode == SparseMode::BAND) {
        attentionMode_ = 1;
    }
    groupSize_ = static_cast<uint32_t>(numHeadsQ_) / static_cast<uint32_t>(numHeadsK_);
    ValidSocVersion validSocVersion = ProcessSocVersion();
    if (validSocVersion == ValidSocVersion::ASCEND910B) {
        mBaseSize_ = s1BaseSize_ * groupSize_;
        s2BaseSize_ = 2048U;
    } else if (validSocVersion == ValidSocVersion::ASCEND950 || validSocVersion == ValidSocVersion::ASCEND960) {
        if (groupSize_ > 32U || topk_ > 2048) {
            s1BaseSize_ = 2;
        }
        mBaseSize_ = s1BaseSize_ * groupSize_;
        s2BaseSize_ = 128U;
    } else {
        mBaseSize_ = s1BaseSize_ * groupSize_;
        s2BaseSize_ = 128U;
    }
    return true;
}

uint32_t LightningIndexerV2MetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    // 1. 如果 sequsedQ_ 传了，直接使用
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t* seqUsedPtr = static_cast<const int32_t*>(sequsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t* s1Ptr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用 querySeqSize_
    return static_cast<uint32_t>(maxSeqlenQ_);
}

uint32_t LightningIndexerV2MetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    // 如果 seqUsedKv_ 传了，直接使用
    if (sequsedK_ != nullptr && sequsedK_->GetData() != nullptr) {
        const int32_t* seqUsedPtr = static_cast<const int32_t*>(sequsedK_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // seqUsedKv_ 没传，判断 Layout
    if (layoutK_ == "TND") {
        // 如果是 TND，尝试使用 actSeqLenOriKv_
        if (cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) {
            const int32_t* s2Ptr = static_cast<const int32_t*>(cuSeqlensK_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 使用 max_seqlen_k
    return static_cast<uint32_t>(maxSeqlenK_);
}

uint64_t LightningIndexerV2MetadataCpuKernel::GetRevertS2Size(uint32_t bIdx)
{
    uint32_t cmpS2Size = GetS2SeqSize(bIdx);
    if (cmpResidualK_ != nullptr && cmpResidualK_->GetData() != nullptr) {
        const int32_t* residualPtr = static_cast<const int32_t*>(cmpResidualK_->GetData());
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_) +
               static_cast<uint64_t>(residualPtr[bIdx]);
    } else {
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_);
    }
}

void LightningIndexerV2MetadataCpuKernel::CalcSplitInfo(SplitContext& splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo& splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < static_cast<uint32_t>(batchSize_); bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        uint32_t s2Size = GetS2SeqSize(bIdx);
        splitInfo.s1GBaseNum[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_) % mBaseSize_;
        splitInfo.s2BaseNum[bIdx] = (s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        splitInfo.s2TailSize[bIdx] = s2Size % s2BaseSize_;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
    return;
}

int64_t LightningIndexerV2MetadataCpuKernel::CalcPreTokenLeftUp(uint32_t s1Size, uint64_t s2Size) const
{
    auto mode = static_cast<SparseMode>(maskMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t LightningIndexerV2MetadataCpuKernel::CalcNextTokenLeftUp(uint32_t s1Size, uint64_t s2Size) const
{
    auto mode = static_cast<SparseMode>(maskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return nextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken_;
        default:
            return nextToken_;
    }
}

int64_t LightningIndexerV2MetadataCpuKernel::CalcCost(uint32_t basicM, uint32_t basicS2) const
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(COST_WEIGHT_M * alignBasicM + COST_WEIGHT_S2 * alignBasicS2);
}

BlockCost<int64_t> LightningIndexerV2MetadataCpuKernel::CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
                                                                      uint32_t s1GTailSize, uint32_t s2TailSize) const
{
    BlockCost<int64_t> typeCost{};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<int64_t> LightningIndexerV2MetadataCpuKernel::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache& batchCache)
{
    // no mask
    if (!attentionMode_) {
        return std::make_pair(0, static_cast<int64_t>(batchCache.revertS2Size));
    }
    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
                                    static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) -
                           1;
    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (isS1G_) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
        s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in two or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    return std::make_pair(s2FirstToken, s2LastToken);
}

void LightningIndexerV2MetadataCpuKernel::CalcBatchCache(uint32_t bIdx, const SplitContext& splitContext,
                                                         BatchCache& batchCache)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;
    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.revertS2Size = GetRevertS2Size(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.revertS2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.revertS2Size);
}

void LightningIndexerV2MetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx, const SplitContext& splitContext,
                                                       const BatchCache& batchCache, S1GCache& s1GCache)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0 || splitInfo.s2BaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    auto s2TokenRange = CalcS2TokenRange(s1GIdx, batchCache);
    int64_t s2FirstToken = s2TokenRange.first;
    int64_t s2LastToken = s2TokenRange.second;
    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.revertS2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.revertS2Size - 1U));

    // get block start & end
    int64_t s2CmpLength = (s2LastToken - s2FirstToken + 1) / cmpRatio_;
    if (s2CmpLength <= 0) {
        s1GCache.s2Start = 0U;
        s1GCache.s2End = 0U;
    } else {
        s1GCache.s2Start = 0U;
        // end of block index, Right-open interval
        s1GCache.s2End = (static_cast<uint32_t>(s2CmpLength) + s2BaseSize_ - 1) / s2BaseSize_;
    }

    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    uint32_t s2TailSize = static_cast<uint32_t>(s2CmpLength) % s2BaseSize_;

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num = s2TailSize != 0 ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;

    BlockCost<int64_t> typeCost =
        CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[batchCache.bIdx], s2TailSize);

    if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost =
            typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num + typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost =
            curTailS2Num > 0U ? typeCost[TAIL_BLOCK][TAIL_BLOCK] : typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost =
            typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num + typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost =
            curTailS2Num > 0U ? typeCost[NORMAL_BLOCK][TAIL_BLOCK] : typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }
}

void LightningIndexerV2MetadataCpuKernel::CalcBatchCost(uint32_t bIdx, const SplitContext& splitContext,
                                                        CostInfo& costInfo)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx) == 0U || GetS2SeqSize(bIdx) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        // 更新最大S1G行开销
        if (s1GCache.s1GCost > costInfo.maxS1GCost) {
            costInfo.maxS1GCost = s1GCache.s1GCost;
        }

        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void LightningIndexerV2MetadataCpuKernel::CalcCostInfo(SplitContext& splitContext)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;
    CostInfo& costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < static_cast<uint32_t>(batchSize_); bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * numHeadsK_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * static_cast<uint32_t>(numHeadsK_);
    }
}

void LightningIndexerV2MetadataCpuKernel::UpdateCursor(const SplitContext& splitContext, AssignContext& assignContext)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;
    const CostInfo& costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) { // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx ==
        static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsK_)) { // 所有负载全部分配完
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsK_) != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsK_);
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    // Update Cache
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void LightningIndexerV2MetadataCpuKernel::AssignByBatch(const SplitContext& splitContext, AssignContext& assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo& costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsK_)) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsK_) != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / static_cast<uint32_t>(numHeadsK_);
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void LightningIndexerV2MetadataCpuKernel::AssignByRow(const SplitContext& splitContext, AssignContext& assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                    assignContext.bN2Cost - assignContext.s1GCache.s1GCost :
                                    0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                     assignContext.bN2Block - assignContext.s1GCache.s1GBlock :
                                     0U;
        // 计算新一行的信息
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void LightningIndexerV2MetadataCpuKernel::AssignByBlock(const SplitContext& splitContext, AssignContext& assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + curCost)) {
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

void LightningIndexerV2MetadataCpuKernel::ForceAssign(const SplitContext& splitContext, AssignContext& assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

bool LightningIndexerV2MetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext& assignContext,
                                                             const SplitResult& splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void LightningIndexerV2MetadataCpuKernel::RecordFDInfo(const SplitContext& splitContext,
                                                       const AssignContext& assignContext, SplitResult& result)
{
    const SplitInfo& splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / static_cast<uint32_t>(numHeadsK_);
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize =
        (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
            (static_cast<uint64_t>(s1Size) * groupSize_ - static_cast<uint64_t>(splitS1GIdx) * mBaseSize_) :
            mBaseSize_;
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.fdBN2Idx[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdMIdx[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.preFdDataNum;
    result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize / groupSize_;
    result.numOfFdHead++;
}

void LightningIndexerV2MetadataCpuKernel::AssignBlockToCore(const SplitContext& splitContext,
                                                            AssignContext& assignContext, SplitResult& result)
{
    const CostInfo& costInfo = splitContext.costInfo;
    result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
        assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
    assignContext.coreCache = {};
    assignContext.coreCache.costLimit = assignContext.unassignedCost / (aicCoreNum_ - assignContext.curCoreIdx);
    if (!supportFd_) {
        assignContext.coreCache.costLimit = costInfo.maxS1GCost > assignContext.coreCache.costLimit ?
                                                costInfo.maxS1GCost :
                                                assignContext.coreCache.costLimit;
    }
    // 1、按整batch分配
    AssignByBatch(splitContext, assignContext);
    // 2、按行分配
    AssignByRow(splitContext, assignContext);
    // 3、按块分配
    if (supportFd_) {
        AssignByBlock(splitContext, assignContext);
        // 4、强制分配
        if (assignContext.coreCache.block == 0) {
            ForceAssign(splitContext, assignContext);
        }
    }
    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
    result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
    assignContext.unassignedCost -= assignContext.coreCache.cost;
    // 对之前的归约信息进行记录并清理
    if (supportFd_ && IsNeedRecordFDInfo(assignContext, result)) {
        RecordFDInfo(splitContext, assignContext, result);
        assignContext.preFdDataNum += assignContext.curKvSplitPart;
        assignContext.curKvSplitPart = 1U;
    }
    // 更新S2切分信息
    if (supportFd_ && assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
        assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
        assignContext.curKvSplitPart++;
    }
}

void LightningIndexerV2MetadataCpuKernel::CalcSplitPlan(int64_t costLimit, const SplitContext& splitContext,
                                                        SplitResult& result)
{
    const CostInfo& costInfo = splitContext.costInfo;
    if (aicCoreNum_ == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;
    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlockToCore(splitContext, assignContext, result);
        // Keep trailing zero-cost batches in the last AIC range so their outputs are initialized.
        if (!assignContext.isFinished && assignContext.unassignedCost <= 0) {
            result.bN2End[i] = static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsK_);
            result.gS1End[i] = 0U;
            result.s2End[i] = 0U;
            assignContext.isFinished = true;
        }
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void LightningIndexerV2MetadataCpuKernel::SplitFD(SplitResult& splitRes)
{
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    // 计算当前最大冗余vec核数
    uint32_t emptyVectorNum = aivCoreNum_ - splitRes.numOfFdHead;
    // 计算每个核处理的load
    // 向上取整，避免核负载为0
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_;
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        // 冗余vec核数为0，此时规约任务无法进行更小的切分，只能1个vec核计算1个规约任务
        if (emptyVectorNum == 0U) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = 0U;
            splitRes.fdRes.fdMNum[curCoreIndex] = splitRes.fdRes.fdMSize[i];
            curCoreIndex++;
            continue;
        }
        // 计算当前归约任务所用核数，向下取整，避免使用核数超出总核数
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        curFDVectorNum = std::max(1U, curFDVectorNum);
        // 计算当前归约任务每个核的行数，向上取整，避免行数为0
        uint32_t curAveMSize = (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
        curFDVectorNum = (splitRes.fdRes.fdMSize[i] + curAveMSize - 1U) / curAveMSize;
        // 需要使用的vec核数与当前剩余可用vec核数取最小
        curFDVectorNum = std::min(curFDVectorNum, emptyVectorNum + 1U); // 1: Fd任务自身带一个核
        // FD负载分配
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
        // 更新冗余vec核数
        emptyVectorNum -= (curFDVectorNum - 1U); // 1: 空余核不包含FD自身的核，要-1
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

bool LightningIndexerV2MetadataCpuKernel::BalanceSchedule(SplitResult& splitRes)
{
    SplitContext splitContext(batchSize_);
    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = static_cast<uint32_t>(batchSize_) * static_cast<uint32_t>(numHeadsK_);
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);

    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;
    CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
    // 3、存在FD任务，对FD进行负载均衡分配
    if (supportFd_ && splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U); // 至少使用1个core
    return true;
}

bool LightningIndexerV2MetadataCpuKernel::GenMetadata(SplitResult& splitRes)
{
    optiling::detail::LiV2Metadata* metadataPtr = static_cast<optiling::detail::LiV2Metadata*>(metadata_->GetData());
    *metadataPtr = {};
    // LI Metadata Generate
    for (size_t i = 0; i < aicCoreNum_; ++i) {
        if (i >= splitRes.usedCoreNum) {
            metadataPtr->liV2Metadata[i][LI_V2_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        metadataPtr->liV2Metadata[i][LI_V2_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        metadataPtr->liV2Metadata[i][LI_V2_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
        metadataPtr->liV2Metadata[i][LI_V2_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
        metadataPtr->liV2Metadata[i][LI_V2_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
        // FA END
        metadataPtr->liV2Metadata[i][LI_V2_BN2_END_INDEX] = splitRes.bN2End[i];
        metadataPtr->liV2Metadata[i][LI_V2_M_END_INDEX] = splitRes.gS1End[i];
        metadataPtr->liV2Metadata[i][LI_V2_S2_END_INDEX] = splitRes.s2End[i];
        //
        metadataPtr->liV2Metadata[i][LI_V2_FIRST_LD_V2_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
    }

    // LD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metadataPtr->ldV2Metadata[i][LD_V2_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metadataPtr->ldV2Metadata[i][LD_V2_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metadataPtr->ldV2Metadata[i][LD_V2_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metadataPtr->ldV2Metadata[i][LD_V2_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metadataPtr->ldV2Metadata[i][LD_V2_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metadataPtr->ldV2Metadata[i][LD_V2_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metadataPtr->ldV2Metadata[i][LD_V2_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metadataPtr->ldV2Metadata[i][LD_V2_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}

struct LiV2FastConfig {
    static constexpr uint32_t CORE_COUNT = 32U;
    static constexpr uint32_t MAX_HEADS = 32U;
    static constexpr int32_t HEAD_DIM = 128;
    static constexpr int32_t TOPK_STATE_SIZE = 2048;
    static constexpr uint32_t QUERY_TILE = 8U;
    static constexpr uint32_t MAX_QUERY_LENGTH = 16384U;
    static constexpr int32_t CAUSAL_MASK_MODE = 3;
    // Work is counted in Q8*K256 blocks; each scan executes a full K512 pair.
    static constexpr uint32_t KEY_WORK_TILE = 256U;
    static constexpr uint32_t KEY_SCAN_TILE = 512U;
    static constexpr uint32_t WORK_BLOCKS_PER_SCAN = KEY_SCAN_TILE / KEY_WORK_TILE;
    static constexpr uint64_t MIN_QUERY_TOKENS = 1024U;
    static constexpr uint64_t MIN_KEY_LENGTH = 32768U;
    static constexpr uint64_t SHORT_KEY_LIMIT = 65536U;
    static constexpr uint64_t LONG_KEY_LIMIT = 262144U;
    // Smaller TopK retains a higher K floor, with a separate floor for G<=16.
    static constexpr uint32_t SMALL_HEAD_LIMIT = 16U;
    static constexpr uint64_t SMALL_TOPK_KEY_FLOOR = 131072U;
    // Utilization floors: 95% below 64K, 90% below 256K, otherwise 85%.
    static constexpr uint64_t SHORT_KEY_UTILIZATION = 95U;
    static constexpr uint64_t MEDIUM_KEY_UTILIZATION = 90U;
    static constexpr uint64_t LONG_KEY_UTILIZATION = 85U;
    static constexpr uint64_t PERCENT_BASE = 100U;
};

struct LiV2FastSequenceWork {
    uint32_t queryLength;
    uint32_t keyLength;
};

struct LiV2FastPartition {
    static constexpr uint32_t MAX_CORES = LiV2FastConfig::CORE_COUNT;
    std::array<uint32_t, MAX_CORES> tileStart{};
    std::array<uint32_t, MAX_CORES> tileEnd{};
    std::array<uint64_t, MAX_CORES> coreWork{};
    uint32_t totalTiles{0U};
    uint64_t totalWork{0U};
};

// Count executed Q8*K256 blocks. The cube/vector pipeline consumes a K512
// pair at a time, including the final partially masked pair in sparse mode 3.
inline uint32_t LiV2FastTileWork(const LiV2FastSequenceWork& sequence, uint32_t queryTile, uint32_t sparseMode)
{
    uint64_t keys = sequence.keyLength;
    if (sparseMode == static_cast<uint32_t>(LiV2FastConfig::CAUSAL_MASK_MODE)) {
        const uint64_t queryEnd =
            (static_cast<uint64_t>(queryTile) + 1U) * LiV2FastConfig::QUERY_TILE < sequence.queryLength ?
                (static_cast<uint64_t>(queryTile) + 1U) * LiV2FastConfig::QUERY_TILE :
                sequence.queryLength;
        const uint64_t maskedKeys = sequence.queryLength - queryEnd;
        keys = keys > maskedKeys ? keys - maskedKeys : 0U;
    }
    return static_cast<uint32_t>(((keys + (LiV2FastConfig::KEY_SCAN_TILE - 1U)) / LiV2FastConfig::KEY_SCAN_TILE) *
                                 LiV2FastConfig::WORK_BLOCKS_PER_SCAN);
}

// Count all Q tiles and their executed work before assigning core ranges.
inline bool LiV2CountFastWork(const std::vector<LiV2FastSequenceWork>& sequences, uint32_t sparseMode, uint32_t cores,
                              LiV2FastPartition& plan)
{
    uint64_t tiles = 0U;
    for (const auto& sequence : sequences) {
        if (sequence.queryLength == 0U || sequence.keyLength == 0U) {
            return false;
        }
        const uint32_t count = (static_cast<uint64_t>(sequence.queryLength) + LiV2FastConfig::QUERY_TILE - 1U) /
                               LiV2FastConfig::QUERY_TILE;
        tiles += count;
        if (tiles > UINT32_MAX) {
            return false;
        }
        for (uint32_t tile = 0U; tile < count; ++tile) {
            plan.totalWork += LiV2FastTileWork(sequence, tile, sparseMode);
        }
    }
    if (tiles < cores) {
        return false;
    }
    plan.totalTiles = static_cast<uint32_t>(tiles);
    return true;
}

// Traverse complete Q tiles in batch order. Close a core's contiguous range
// at the closer side of ceil(remainingWork / remainingCores), reserving at
// least one tile per remaining core. Never split a Q tile along K.
inline bool LiV2BuildFastPartition(const std::vector<LiV2FastSequenceWork>& sequences, uint32_t sparseMode,
                                   uint32_t cores, LiV2FastPartition& plan)
{
    plan = {};
    if (cores == 0U || cores > LiV2FastPartition::MAX_CORES) {
        return false;
    }
    if (!LiV2CountFastWork(sequences, sparseMode, cores, plan)) {
        return false;
    }
    uint64_t remainingWork = plan.totalWork;
    uint64_t target = (remainingWork + cores - 1U) / cores;
    uint32_t core = 0U;
    uint32_t cursor = 0U;
    for (const auto& sequence : sequences) {
        const uint32_t count = (static_cast<uint64_t>(sequence.queryLength) + LiV2FastConfig::QUERY_TILE - 1U) /
                               LiV2FastConfig::QUERY_TILE;
        for (uint32_t tile = 0U; tile < count; ++tile) {
            const uint64_t work = LiV2FastTileWork(sequence, tile, sparseMode);
            const uint64_t currentWork = plan.coreWork[core];
            // Compare stopping here with adding the pending whole Q tile.
            // On a tie, stop here to avoid an unnecessary heavy core.
            if (core + 1U < cores && cursor > plan.tileStart[core] && currentWork < target &&
                work >= target - currentWork && target - currentWork <= work - (target - currentWork)) {
                plan.tileEnd[core] = cursor;
                remainingWork -= currentWork;
                ++core;
                plan.tileStart[core] = cursor;
                const uint32_t activeCores = cores - core;
                target = (remainingWork + activeCores - 1U) / activeCores;
            }
            plan.coreWork[core] += work;
            ++cursor;
            const uint32_t remainingTiles = plan.totalTiles - cursor;
            const uint32_t remainingCores = cores - core - 1U;
            if (remainingCores != 0U && remainingTiles >= remainingCores &&
                (plan.coreWork[core] >= target || remainingTiles == remainingCores)) {
                plan.tileEnd[core] = cursor;
                remainingWork -= plan.coreWork[core];
                ++core;
                plan.tileStart[core] = cursor;
                target = (remainingWork + remainingCores - 1U) / remainingCores;
            }
        }
    }
    plan.tileEnd[core] = cursor;
    return true;
}

struct LiV2FastPerformanceShape {
    uint64_t queryTokens{0U};
    uint64_t usefulQueryKeyPairs{0U};
    uint64_t minKeyLength{UINT64_MAX};
    uint32_t topk{0U};
    uint32_t coreCount{0U};
    uint32_t heads{0U};
};

// Reject unsupported performance shapes before traversing and partitioning Q tiles.
inline bool LiV2FastPerformanceShapeSupported(const LiV2FastPerformanceShape& shape)
{
    if (shape.coreCount != LiV2FastConfig::CORE_COUNT || shape.queryTokens < LiV2FastConfig::MIN_QUERY_TOKENS ||
        shape.minKeyLength < LiV2FastConfig::MIN_KEY_LENGTH) {
        return false;
    }

    // TopK=2048 shares one K floor across G1-32. Preserve smaller-TopK admission.
    if (shape.topk != static_cast<uint32_t>(LiV2FastConfig::TOPK_STATE_SIZE)) {
        const uint64_t topkKeyFloor = shape.heads <= LiV2FastConfig::SMALL_HEAD_LIMIT ?
                                          LiV2FastConfig::LONG_KEY_LIMIT :
                                          LiV2FastConfig::SMALL_TOPK_KEY_FLOOR;
        if (shape.minKeyLength < topkKeyFloor) {
            return false;
        }
    }
    return true;
}

// Compare useful Q-K pairs with the capacity occupied until the heaviest core
// finishes. This accounts for Q/K padding, causal masking and core imbalance.
inline bool LiV2FastPerformanceSupported(const LiV2FastPerformanceShape& shape, const LiV2FastPartition& plan)
{
    if (plan.totalWork == 0U) {
        return false;
    }

    const uint64_t maxWork = *std::max_element(plan.coreWork.begin(), plan.coreWork.end());
    const uint64_t scheduledWork = maxWork * shape.coreCount;
    const uint64_t usefulWork =
        shape.usefulQueryKeyPairs / (LiV2FastConfig::QUERY_TILE * LiV2FastConfig::KEY_WORK_TILE);
    const uint64_t minUtilization =
        shape.minKeyLength < LiV2FastConfig::SHORT_KEY_LIMIT ?
            LiV2FastConfig::SHORT_KEY_UTILIZATION :
            (shape.minKeyLength < LiV2FastConfig::LONG_KEY_LIMIT ? LiV2FastConfig::MEDIUM_KEY_UTILIZATION :
                                                                   LiV2FastConfig::LONG_KEY_UTILIZATION);
    // Round the required capacity up without multiplying the full workload by
    // a percentage. Long K still needs enough utilization to amortize padding.
    const uint64_t requiredWork =
        scheduledWork / LiV2FastConfig::PERCENT_BASE * minUtilization +
        (scheduledWork % LiV2FastConfig::PERCENT_BASE * minUtilization + (LiV2FastConfig::PERCENT_BASE - 1U)) /
            LiV2FastConfig::PERCENT_BASE;
    // Require the useful-to-scheduled work ratio to meet the threshold using integer arithmetic.
    return usefulWork >= requiredWork;
}

bool LightningIndexerV2MetadataCpuKernel::IsFastMetadataCandidate()
{
    return ProcessSocVersion() == ValidSocVersion::ASCEND950 && aicCoreNum_ == LiV2FastConfig::CORE_COUNT &&
           numHeadsK_ == 1 && groupSize_ >= 1U && groupSize_ <= LiV2FastConfig::MAX_HEADS &&
           headDim_ == LiV2FastConfig::HEAD_DIM && topk_ >= 1 && topk_ <= LiV2FastConfig::TOPK_STATE_SIZE &&
           (maskMode_ == 0 || maskMode_ == LiV2FastConfig::CAUSAL_MASK_MODE) && cmpRatio_ == 1 &&
           (cmpResidualK_ == nullptr || cmpResidualK_->GetData() == nullptr);
}

bool LightningIndexerV2MetadataCpuKernel::InitFastSequenceInputs(FastSequenceInputs& inputs) const
{
    inputs.queryTnd = layoutQ_ == "TND";
    inputs.keyTnd = layoutK_ == "TND";
    const bool pageAttention = layoutK_ == "PA_BBND";
    inputs.cuQ = cuSeqlensQ_ != nullptr ? static_cast<const int32_t*>(cuSeqlensQ_->GetData()) : nullptr;
    inputs.cuK = cuSeqlensK_ != nullptr ? static_cast<const int32_t*>(cuSeqlensK_->GetData()) : nullptr;
    const auto* usedK = sequsedK_ != nullptr ? static_cast<const int32_t*>(sequsedK_->GetData()) : nullptr;
    return (!inputs.queryTnd || inputs.cuQ != nullptr) && (!inputs.keyTnd || inputs.cuK != nullptr) &&
           (!pageAttention || usedK != nullptr) && (pageAttention || layoutQ_ == layoutK_);
}

bool LightningIndexerV2MetadataCpuKernel::GetFastSequenceWork(uint32_t batch, const FastSequenceInputs& inputs,
                                                              uint32_t& q, uint32_t& k)
{
    const int64_t storedQ =
        inputs.queryTnd ? static_cast<int64_t>(inputs.cuQ[batch + 1]) - inputs.cuQ[batch] : maxSeqlenQ_;
    const int64_t storedK =
        inputs.keyTnd ? static_cast<int64_t>(inputs.cuK[batch + 1]) - inputs.cuK[batch] : maxSeqlenK_;
    q = GetS1SeqSize(batch);
    k = GetS2SeqSize(batch);
    if (q != storedQ || q < 1U || q > LiV2FastConfig::MAX_QUERY_LENGTH || k > storedK) {
        return false;
    }
    // Each causal row needs the complete initial TopK state.
    return maskMode_ != LiV2FastConfig::CAUSAL_MASK_MODE ||
           static_cast<uint64_t>(k) + 1U >= static_cast<uint64_t>(q) + LiV2FastConfig::TOPK_STATE_SIZE;
}

inline void LiV2PublishFastPartition(optiling::detail::LiV2Metadata& metadata, const LiV2FastPartition& plan,
                                     uint32_t cores)
{
    auto& fast = metadata.fastMetadata;
    fast.coreNum = cores;
    for (uint32_t core = 0; core < cores; ++core) {
        fast.tileStart[core] = plan.tileStart[core];
        fast.tileEnd[core] = plan.tileEnd[core];
    }
    fast.enable = 1U;
}

void LightningIndexerV2MetadataCpuKernel::GenFastMetadata()
{
    if (!IsFastMetadataCandidate()) {
        return;
    }
    FastSequenceInputs inputs{};
    if (!InitFastSequenceInputs(inputs)) {
        return;
    }
    LiV2FastPerformanceShape performance{};
    performance.topk = topk_;
    performance.heads = groupSize_;
    performance.coreCount = aicCoreNum_;
    std::vector<LiV2FastSequenceWork> sequences;
    for (int32_t batch = 0; batch < batchSize_; ++batch) {
        uint32_t q = 0U;
        uint32_t k = 0U;
        if (!GetFastSequenceWork(batch, inputs, q, k)) {
            return;
        }
        sequences.push_back({q, k});
        uint64_t usefulPairs = static_cast<uint64_t>(q) * k;
        if (maskMode_ == LiV2FastConfig::CAUSAL_MASK_MODE) {
            // Masked prefixes sum to 0 + ... + (q-1) = q*(q-1)/2.
            usefulPairs -= static_cast<uint64_t>(q) * (q - 1U) / 2U;
        }
        if (usefulPairs > UINT64_MAX - performance.usefulQueryKeyPairs) {
            return;
        }
        performance.usefulQueryKeyPairs += usefulPairs;
        performance.queryTokens += q;
        performance.minKeyLength = std::min(performance.minKeyLength, static_cast<uint64_t>(k));
    }
    if (!LiV2FastPerformanceShapeSupported(performance)) {
        return;
    }
    LiV2FastPartition plan;
    const uint32_t cores = aicCoreNum_;
    if (!LiV2BuildFastPartition(sequences, maskMode_, cores, plan) ||
        !LiV2FastPerformanceSupported(performance, plan)) {
        return;
    }
    auto* metadata = static_cast<optiling::detail::LiV2Metadata*>(metadata_->GetData());
    LiV2PublishFastPartition(*metadata, plan, cores);
}

namespace {
static const char* kernelType = "LightningIndexerV2Metadata";
REGISTER_CPU_KERNEL(kernelType, LightningIndexerV2MetadataCpuKernel);
} // namespace
}; // namespace aicpu::liv2_metadata
