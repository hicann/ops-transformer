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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch22.h
 * \brief AICPU kernel implementation for A3 (arch22)
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_ARCH22_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_ARCH22_H

#include "../sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include "../../../sparse_lightning_indexer_kl_loss_grad/op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_metadata_arch22.h"

using namespace optiling;

namespace aicpu {

enum class SliSparseMode : int32_t {
    NO_MASK = 0,
    RIGHT_DOWN_CAUSAL = 3,
};

enum class SliLayout {
    BSND = 0,
    TND = 1,
};

class SparseLightningIndexerKLLossGradMetadataCpuKernelArch22 : public CpuKernel {
public:
    SparseLightningIndexerKLLossGradMetadataCpuKernelArch22() = default;
    ~SparseLightningIndexerKLLossGradMetadataCpuKernelArch22() override = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool ParamsCheck();
    int64_t GetActualSeqLen(Tensor *tensor, int64_t bIdx) const;
    int64_t GetSeqUsedLen(Tensor *tensor, int64_t bIdx, int64_t defaultLen) const;
    int64_t GetS1SeqSize(int64_t bIdx) const;
    int64_t GetS2SeqSize(int64_t bIdx) const;
    int64_t CalcTotalSize() const;
    int64_t GetCmpResidualKey(int64_t bIdx) const;
    int64_t GetPreCompressS2Len(int64_t bIdx, int64_t s2Size) const;
    int64_t GetS2RealSize(int64_t bIdx, int64_t s1Size, int64_t s2Size, int64_t s1Idx) const;
    int64_t GetS1Load(int64_t s2RealSize) const;
    bool BuildSparseValidArray(std::vector<int64_t> &sparseValidArray);
    bool CanSplitWithMaxLoad(const std::vector<int64_t> &sparseValidArray, int64_t start, int64_t partNum,
                             int64_t maxLoad) const;
    int64_t FindMinMaxLoad(const std::vector<int64_t> &sparseValidArray, int64_t start, int64_t partNum) const;
    bool BuildBalancedSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                                     std::vector<int64_t> &sparseStartIdx) const;
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray);
    bool BuildMetadata();

    Tensor *cuSeqLensQuery_ = nullptr;
    Tensor *cuSeqLensKey_ = nullptr;
    Tensor *seqUsedQuery_ = nullptr;
    Tensor *seqUsedKey_ = nullptr;
    Tensor *cmpResidualKey_ = nullptr;
    Tensor *metadata_ = nullptr;

    int64_t bSize_ = 0;
    int64_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t kSize_ = 0;
    int64_t numHeadsQ_ = 0;
    int64_t numHeadsK_ = 0;
    int64_t headDim_ = 0;
    int64_t sparseMode_ = 0;
    int64_t cmpRatio_ = 1;
    int64_t aicCoreNum_ = 25;
    std::string layout_ = "BSND";
    std::string layoutK_ = "BSND";
    SliLayout layoutType_ = SliLayout::BSND;

    int64_t coreNum_ = 0;
    int64_t totalSize_ = 0;
    int64_t splitFactorSize_ = 0;
    std::vector<int64_t> bS1Index_;
};

inline uint32_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::Compute(CpuKernelContext &ctx)
{
    if (!Prepare(ctx)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return BuildMetadata() ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::Prepare(CpuKernelContext &ctx)
{
    cuSeqLensQuery_ = ctx.Input(0);
    cuSeqLensKey_ = ctx.Input(1);
    seqUsedQuery_ = ctx.Input(2);
    seqUsedKey_ = ctx.Input(3);
    cmpResidualKey_ = ctx.Input(4);
    metadata_ = ctx.Output(0);

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_k", numHeadsK_) &&
                         GetAttrValue(ctx, "head_dim", headDim_);
    if (!requiredAttrs) {
        return false;
    }

    GetAttrValueOpt(ctx, "batch_size", bSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", s1Size_);
    GetAttrValueOpt(ctx, "max_seqlen_k", s2Size_);
    GetAttrValueOpt(ctx, "topk", kSize_);
    GetAttrValueOpt(ctx, "layout_q", layout_);
    GetAttrValueOpt(ctx, "layout_k", layoutK_);
    GetAttrValueOpt(ctx, "mask_mode", sparseMode_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);

    if (layout_ == "BSND") {
        layoutType_ = SliLayout::BSND;
    } else if (layout_ == "TND") {
        layoutType_ = SliLayout::TND;
    } else {
        KERNEL_LOG_ERROR("layout_q must be BSND or TND, but got %s", layout_.c_str());
        return false;
    }

    if (bSize_ <= 0) {
        if (IsTensorValid(seqUsedQuery_)) {
            bSize_ = seqUsedQuery_->GetTensorShape()->GetDimSize(0);
        } else if (layoutType_ == SliLayout::TND && IsTensorValid(cuSeqLensQuery_)) {
            bSize_ = cuSeqLensQuery_->GetTensorShape()->GetDimSize(0) - 1;
        }
    }

    return ParamsCheck();
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::ParamsCheck()
{
    KERNEL_CHECK_NULLPTR(metadata_, false, "metadata output is null");
    KERNEL_CHECK_NULLPTR(metadata_->GetData(), false, "metadata data is null");
    KERNEL_CHECK_NULLPTR(metadata_->GetTensorShape(), false, "metadata shape is null");

    Tensor *int32VectorInputs[] = {cuSeqLensQuery_, cuSeqLensKey_, seqUsedQuery_, seqUsedKey_, cmpResidualKey_};
    const char *inputNames[] = {"cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k"};
    for (size_t idx = 0; idx < sizeof(int32VectorInputs) / sizeof(int32VectorInputs[0]); ++idx) {
        Tensor *tensor = int32VectorInputs[idx];
        if (!IsTensorValid(tensor)) {
            continue;
        }
        if (static_cast<DataType>(tensor->GetDataType()) != DT_INT32) {
            KERNEL_LOG_ERROR("%s dtype must be int32, but got %d.", inputNames[idx], tensor->GetDataType());
            return false;
        }
        if (tensor->GetTensorShape()->GetDims() != 1) {
            KERNEL_LOG_ERROR("%s must be a 1D tensor, but got rank %d.", inputNames[idx],
                             tensor->GetTensorShape()->GetDims());
            return false;
        }
    }

    if (aicCoreNum_ <= 0) {
        KERNEL_LOG_ERROR("aic_core_num must be positive, but got %ld", aicCoreNum_);
        return false;
    }
    if (numHeadsQ_ != 8 && numHeadsQ_ != 16 && numHeadsQ_ != 32 && numHeadsQ_ != 64) {
        KERNEL_LOG_ERROR("num_heads_q must be one of {8, 16, 32, 64}, but got %ld.", numHeadsQ_);
        return false;
    }
    if (numHeadsK_ != 1) {
        KERNEL_LOG_ERROR("num_heads_k must be 1, but got %ld.", numHeadsK_);
        return false;
    }
    if (headDim_ != 128) {
        KERNEL_LOG_ERROR("head_dim must be 128, but got %ld.", headDim_);
        return false;
    }
    if (bSize_ <= 0 || kSize_ <= 0) {
        KERNEL_LOG_ERROR("batch_size/topk must be positive, but got batch_size=%ld topk=%ld", bSize_, kSize_);
        return false;
    }
    if (kSize_ > 8192 || (kSize_ != 512 && kSize_ % 1024 != 0)) {
        KERNEL_LOG_ERROR("topk must be 512 or a multiple of 1024 in [1024, 8192], but got %ld.", kSize_);
        return false;
    }
    if (layoutType_ == SliLayout::BSND && (s1Size_ <= 0 || s2Size_ <= 0)) {
        KERNEL_LOG_ERROR("max_seqlen_q/max_seqlen_k must be positive for BSND, but got q=%ld k=%ld", s1Size_, s2Size_);
        return false;
    }
    if (s1Size_ < 0 || s2Size_ < 0) {
        KERNEL_LOG_ERROR("max_seqlen_q/max_seqlen_k must be non-negative, but got q=%ld k=%ld", s1Size_, s2Size_);
        return false;
    }
    if (numHeadsQ_ % numHeadsK_ != 0) {
        KERNEL_LOG_ERROR("num_heads_q must be divisible by num_heads_k, but got nq=%ld nk=%ld", numHeadsQ_, numHeadsK_);
        return false;
    }
    if (layoutK_ != layout_) {
        KERNEL_LOG_ERROR("layout_q and layout_k must be the same on arch22, but got %s and %s.", layout_.c_str(),
                         layoutK_.c_str());
        return false;
    }
    if (sparseMode_ != static_cast<int64_t>(SliSparseMode::NO_MASK) &&
        sparseMode_ != static_cast<int64_t>(SliSparseMode::RIGHT_DOWN_CAUSAL)) {
        KERNEL_LOG_ERROR("mask_mode only supports 0 or 3, but got %ld", sparseMode_);
        return false;
    }
    if (cmpRatio_ < 1 || cmpRatio_ > 128) {
        KERNEL_LOG_ERROR("cmp_ratio must be in [1, 128], but got %ld", cmpRatio_);
        return false;
    }
    if (sparseMode_ == static_cast<int64_t>(SliSparseMode::RIGHT_DOWN_CAUSAL) && cmpRatio_ != 1 &&
        !IsTensorValid(cmpResidualKey_)) {
        KERNEL_LOG_ERROR("cmp_residual_k is required when mask_mode is 3 and cmp_ratio is not 1.");
        return false;
    }
    if (layoutType_ == SliLayout::TND && (!IsTensorValid(cuSeqLensQuery_) || !IsTensorValid(cuSeqLensKey_))) {
        KERNEL_LOG_ERROR("cu_seqlens_q/cu_seqlens_k must be provided for TND metadata.");
        return false;
    }
    if (layoutType_ == SliLayout::TND && (cuSeqLensQuery_->GetTensorShape()->GetDimSize(0) < bSize_ + 1 ||
                                          cuSeqLensKey_->GetTensorShape()->GetDimSize(0) < bSize_ + 1)) {
        KERNEL_LOG_ERROR("cu_seqlens_q/cu_seqlens_k length must be at least batch_size + 1.");
        return false;
    }
    if (IsTensorValid(cmpResidualKey_) && cmpResidualKey_->GetTensorShape()->GetDimSize(0) < bSize_) {
        KERNEL_LOG_ERROR("cmp_residual_k length must be at least batch_size, but got %ld and batch_size=%ld.",
                         cmpResidualKey_->GetTensorShape()->GetDimSize(0), bSize_);
        return false;
    }
    if (IsTensorValid(seqUsedQuery_) && seqUsedQuery_->GetTensorShape()->GetDimSize(0) != bSize_) {
        KERNEL_LOG_ERROR("seqused_q length must equal batch_size, but got %ld and batch_size=%ld.",
                         seqUsedQuery_->GetTensorShape()->GetDimSize(0), bSize_);
        return false;
    }
    if (IsTensorValid(seqUsedKey_) && seqUsedKey_->GetTensorShape()->GetDimSize(0) != bSize_) {
        KERNEL_LOG_ERROR("seqused_k length must equal batch_size, but got %ld and batch_size=%ld.",
                         seqUsedKey_->GetTensorShape()->GetDimSize(0), bSize_);
        return false;
    }
    for (int64_t bIdx = 0; bIdx < bSize_; ++bIdx) {
        if (IsTensorValid(cmpResidualKey_) && GetCmpResidualKey(bIdx) < 0) {
            KERNEL_LOG_ERROR("cmp_residual_k[%ld] must be non-negative, but got %ld.", bIdx, GetCmpResidualKey(bIdx));
            return false;
        }
        int64_t allocS1 = layoutType_ == SliLayout::TND ? GetActualSeqLen(cuSeqLensQuery_, bIdx) : s1Size_;
        int64_t allocS2 = layoutType_ == SliLayout::TND ? GetActualSeqLen(cuSeqLensKey_, bIdx) : s2Size_;
        int64_t usedS1 = GetSeqUsedLen(seqUsedQuery_, bIdx, allocS1);
        int64_t usedS2 = GetSeqUsedLen(seqUsedKey_, bIdx, allocS2);
        if (usedS1 < 0 || usedS1 > allocS1) {
            KERNEL_LOG_ERROR("seqused_q[%ld] must be in [0, %ld], but got %ld.", bIdx, allocS1, usedS1);
            return false;
        }
        if (usedS2 < 0 || usedS2 > allocS2) {
            KERNEL_LOG_ERROR("seqused_k[%ld] must be in [0, %ld], but got %ld.", bIdx, allocS2, usedS2);
            return false;
        }
    }
    return true;
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::GetActualSeqLen(Tensor *tensor,
                                                                                        int64_t bIdx) const
{
    if (!IsTensorValid(tensor)) {
        return 0;
    }
    auto *data = reinterpret_cast<const int32_t *>(tensor->GetData());
    return static_cast<int64_t>(data[bIdx + 1]) - static_cast<int64_t>(data[bIdx]);
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::GetSeqUsedLen(Tensor *tensor, int64_t bIdx,
                                                                                      int64_t defaultLen) const
{
    if (!IsTensorValid(tensor)) {
        return defaultLen;
    }
    auto *data = reinterpret_cast<const int32_t *>(tensor->GetData());
    return static_cast<int64_t>(data[bIdx]);
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::GetS1SeqSize(int64_t bIdx) const
{
    int64_t defaultLen = layoutType_ == SliLayout::TND ? GetActualSeqLen(cuSeqLensQuery_, bIdx) : s1Size_;
    return GetSeqUsedLen(seqUsedQuery_, bIdx, defaultLen);
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::GetS2SeqSize(int64_t bIdx) const
{
    int64_t defaultLen = layoutType_ == SliLayout::TND ? GetActualSeqLen(cuSeqLensKey_, bIdx) : s2Size_;
    return GetSeqUsedLen(seqUsedKey_, bIdx, defaultLen);
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::CalcTotalSize() const
{
    if (IsTensorValid(seqUsedQuery_)) {
        int64_t total = 0;
        for (int64_t bIdx = 0; bIdx < bSize_; ++bIdx) {
            total += GetS1SeqSize(bIdx);
        }
        return total;
    }
    if (layoutType_ == SliLayout::TND) {
        auto *data = reinterpret_cast<const int32_t *>(cuSeqLensQuery_->GetData());
        return data[bSize_];
    }
    return bSize_ * s1Size_;
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::GetCmpResidualKey(int64_t bIdx) const
{
    if (!IsTensorValid(cmpResidualKey_)) {
        return 0;
    }
    auto *data = reinterpret_cast<const int32_t *>(cmpResidualKey_->GetData());
    return static_cast<int64_t>(data[bIdx]);
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::GetPreCompressS2Len(int64_t bIdx,
                                                                                            int64_t s2Size) const
{
    return std::max<int64_t>(0, s2Size * cmpRatio_ + GetCmpResidualKey(bIdx));
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::GetS2RealSize(int64_t bIdx, int64_t s1Size,
                                                                                      int64_t s2Size,
                                                                                      int64_t s1Idx) const
{
    int64_t s2SparseLen = 0;
    if (sparseMode_ == static_cast<int64_t>(SliSparseMode::NO_MASK)) {
        s2SparseLen = s2Size;
    } else if (sparseMode_ == static_cast<int64_t>(SliSparseMode::RIGHT_DOWN_CAUSAL)) {
        s2SparseLen = (GetPreCompressS2Len(bIdx, s2Size) - s1Size + s1Idx + 1) / cmpRatio_;
    }
    return std::max<int64_t>(0, std::min(s2SparseLen, kSize_));
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::GetS1Load(int64_t s2RealSize) const
{
    if (s2RealSize <= 0) {
        return 0;
    }
    int64_t fixedS1Load = std::min<int64_t>(256, std::min<int64_t>(kSize_, std::max<int64_t>(64, kSize_ / 4)));
    return s2RealSize + fixedS1Load;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::BuildSparseValidArray(
    std::vector<int64_t> &sparseValidArray)
{
    if (sparseValidArray.empty()) {
        KERNEL_LOG_ERROR("Sparse valid array size should be larger than 0.");
        return false;
    }
    if (layoutType_ == SliLayout::TND) {
        int64_t accumS1 = 0;
        for (int64_t bIdx = 0; bIdx < bSize_; ++bIdx) {
            int64_t actualS1 = GetS1SeqSize(bIdx);
            int64_t actualS2 = GetS2SeqSize(bIdx);
            for (int64_t s1Idx = 0; s1Idx < actualS1 && accumS1 < static_cast<int64_t>(sparseValidArray.size());
                 ++s1Idx, ++accumS1) {
                sparseValidArray[accumS1] = GetS1Load(GetS2RealSize(bIdx, actualS1, actualS2, s1Idx));
            }
        }
    } else {
        int64_t accum = 0;
        for (int64_t bIdx = 0; bIdx < bSize_; ++bIdx) {
            int64_t actualS1 = GetS1SeqSize(bIdx);
            int64_t actualS2 = GetS2SeqSize(bIdx);
            for (int64_t s1Idx = 0; s1Idx < actualS1 && accum < static_cast<int64_t>(sparseValidArray.size());
                 ++s1Idx, ++accum) {
                sparseValidArray[accum] = GetS1Load(GetS2RealSize(bIdx, actualS1, actualS2, s1Idx));
            }
        }
    }
    return true;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::CanSplitWithMaxLoad(
    const std::vector<int64_t> &sparseValidArray, int64_t start, int64_t partNum, int64_t maxLoad) const
{
    if (partNum == 0) {
        return start == totalSize_;
    }
    if (partNum < 0 || start < 0 || start > totalSize_ || totalSize_ - start < partNum) {
        return false;
    }

    int64_t usedPartNum = 1;
    int64_t curLoad = 0;
    for (int64_t idx = start; idx < totalSize_; ++idx) {
        int64_t load = sparseValidArray[idx];
        if (load > maxLoad) {
            return false;
        }
        if (curLoad + load > maxLoad) {
            ++usedPartNum;
            curLoad = load;
            if (usedPartNum > partNum) {
                return false;
            }
        } else {
            curLoad += load;
        }
    }
    return usedPartNum <= partNum;
}

inline int64_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::FindMinMaxLoad(
    const std::vector<int64_t> &sparseValidArray, int64_t start, int64_t partNum) const
{
    int64_t totalLoad = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.end(), 0LL);
    int64_t minMaxLoad = std::max(*std::max_element(sparseValidArray.begin() + start, sparseValidArray.end()),
                                  CeilDiv(totalLoad, partNum));
    int64_t maxLoad = totalLoad;
    while (minMaxLoad < maxLoad) {
        int64_t midLoad = minMaxLoad + (maxLoad - minMaxLoad) / 2;
        if (CanSplitWithMaxLoad(sparseValidArray, start, partNum, midLoad)) {
            maxLoad = midLoad;
        } else {
            minMaxLoad = midLoad + 1;
        }
    }
    return minMaxLoad;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::BuildBalancedSparseStartIdx(
    const std::vector<int64_t> &sparseValidArray, std::vector<int64_t> &sparseStartIdx) const
{
    sparseStartIdx.assign(SLI_METADATA_MAX_CORE_NUM, totalSize_);
    int64_t activeStart = 0;
    while (activeStart < totalSize_ && sparseValidArray[activeStart] <= 0) {
        ++activeStart;
    }
    if (activeStart >= totalSize_) {
        return true;
    }

    int64_t activeCoreNum = std::min(coreNum_, totalSize_ - activeStart);
    sparseStartIdx[0] = activeStart;
    if (activeCoreNum == 1) {
        return true;
    }

    std::vector<int64_t> prefixLoad(totalSize_ + 1, 0);
    for (int64_t idx = 0; idx < totalSize_; ++idx) {
        prefixLoad[idx + 1] = prefixLoad[idx] + sparseValidArray[idx];
    }

    int64_t maxLoad = FindMinMaxLoad(sparseValidArray, activeStart, activeCoreNum);
    int64_t prevStart = activeStart;
    for (int64_t coreIdx = 1; coreIdx < activeCoreNum; ++coreIdx) {
        int64_t remainingPartNum = activeCoreNum - coreIdx;
        int64_t minStart = prevStart + 1;
        int64_t maxStart = totalSize_ - remainingPartNum;

        auto maxLoadIter = std::upper_bound(prefixLoad.begin() + minStart, prefixLoad.begin() + maxStart + 1,
                                            prefixLoad[prevStart] + maxLoad);
        maxStart = std::min<int64_t>(maxStart, std::distance(prefixLoad.begin(), maxLoadIter) - 1);
        if (maxStart < minStart) {
            KERNEL_LOG_ERROR("Failed to build balanced metadata, invalid range [%ld, %ld].", minStart, maxStart);
            return false;
        }

        int64_t left = minStart;
        int64_t right = maxStart;
        while (left < right) {
            int64_t mid = left + (right - left) / 2;
            if (CanSplitWithMaxLoad(sparseValidArray, mid, remainingPartNum, maxLoad)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        int64_t lowerFeasibleStart = left;
        if (!CanSplitWithMaxLoad(sparseValidArray, lowerFeasibleStart, remainingPartNum, maxLoad)) {
            KERNEL_LOG_ERROR("Failed to build balanced metadata, suffix cannot be split.");
            return false;
        }

        int64_t remainingLoad = prefixLoad[totalSize_] - prefixLoad[prevStart];
        int64_t targetLoad = CeilDiv(remainingLoad, remainingPartNum + 1);
        int64_t targetPrefixLoad = prefixLoad[prevStart] + targetLoad;
        auto candidateIter = std::lower_bound(prefixLoad.begin() + lowerFeasibleStart,
                                              prefixLoad.begin() + maxStart + 1, targetPrefixLoad);
        int64_t candidate = std::min<int64_t>(std::distance(prefixLoad.begin(), candidateIter), maxStart);
        if (candidate > lowerFeasibleStart &&
            AbsDiff(prefixLoad[candidate - 1], targetPrefixLoad) <= AbsDiff(prefixLoad[candidate], targetPrefixLoad)) {
            --candidate;
        }

        sparseStartIdx[coreIdx] = candidate;
        prevStart = candidate;
    }
    return true;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::SetSparseStartIdx(
    const std::vector<int64_t> &sparseValidArray)
{
    if (!BuildBalancedSparseStartIdx(sparseValidArray, bS1Index_)) {
        return false;
    }

    for (int64_t idx = 1; idx < static_cast<int64_t>(SLI_METADATA_MAX_CORE_NUM); ++idx) {
        if (bS1Index_[idx] == 0) {
            bS1Index_[idx] = totalSize_;
        }
    }
    return true;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch22::BuildMetadata()
{
    totalSize_ = CalcTotalSize();
    if (totalSize_ <= 0) {
        KERNEL_LOG_ERROR("totalSize should be larger than 0, but got %ld", totalSize_);
        return false;
    }

    coreNum_ = std::min<int64_t>(std::min(totalSize_, aicCoreNum_), SLI_METADATA_MAX_CORE_NUM);
    splitFactorSize_ = CeilDiv(totalSize_, coreNum_);

    std::vector<int64_t> sparseValidArray(totalSize_, 0);
    if (!BuildSparseValidArray(sparseValidArray) || !SetSparseStartIdx(sparseValidArray)) {
        return false;
    }

    auto *metadataData = reinterpret_cast<SLI_METADATA_T *>(metadata_->GetData());
    std::fill_n(metadataData, SLI_METADATA_SIZE, static_cast<SLI_METADATA_T>(0));
    auto *metadata = reinterpret_cast<detail::SliGradKLLossMetaData *>(metadataData);
    metadata->coreNum = static_cast<int32_t>(coreNum_);
    metadata->totalSize = static_cast<int32_t>(totalSize_);
    metadata->splitFactorSize = static_cast<int32_t>(splitFactorSize_);
    for (uint32_t i = 0; i < SLI_METADATA_HEADER_SIZE - 3; ++i) {
        metadata->reserved[i] = 0;
    }
    for (uint32_t i = 0; i < SLI_METADATA_MAX_CORE_NUM; ++i) {
        metadata->bS1Index[i] = static_cast<int32_t>(bS1Index_[i]);
    }
    return true;
}

} // namespace aicpu

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_ARCH22_H
