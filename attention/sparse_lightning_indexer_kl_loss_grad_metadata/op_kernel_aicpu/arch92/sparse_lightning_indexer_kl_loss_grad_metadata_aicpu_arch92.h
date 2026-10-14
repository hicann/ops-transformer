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
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_aicpu_arch92.h
 * \brief AICPU kernel implementation for A6 (arch92)
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_ARCH92_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_ARCH92_H

#include "../sparse_lightning_indexer_kl_loss_grad_metadata_aicpu.h"
#include "../../../sparse_lightning_indexer_kl_loss_grad/op_kernel/arch92/sparse_lightning_indexer_kl_loss_grad_metadata_arch92.h"

namespace aicpu {

constexpr int64_t FA_TOLERANCE_RATIO_ARCH92 = 1;

enum BlockTypeArch92 : uint32_t {
    NORMAL_BLOCK_ARCH92 = 0,
    TAIL_BLOCK_ARCH92,
    BLOCK_MAX_TYPE_ARCH92
};

enum class SparseModeArch92 : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

enum class ValidSocVersionArch92 {
    ASCEND910 = 0,
    ASCEND950,
    ASCEND960,
    RESERVED_VERSION = 99999
};

template <class T>
using RangeArch92 = std::pair<T, T>;

template <class T>
using BlockCostArch92 =
    std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE_ARCH92)>, static_cast<size_t>(BLOCK_MAX_TYPE_ARCH92)>;

class SparseLightningIndexerKLLossGradMetadataCpuKernelArch92 : public CpuKernel {
public:
    SparseLightningIndexerKLLossGradMetadataCpuKernelArch92() = default;
    ~SparseLightningIndexerKLLossGradMetadataCpuKernelArch92() override = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    bool Prepare(CpuKernelContext& ctx);
    int32_t GetQueryBatchSize();
    bool ParamsInit();
    bool ParamsCheck();
    bool BalanceSchedule();
    bool GenMetadata();
    ValidSocVersionArch92 ProcessSocVersion();
    uint32_t GetS1SeqSize(uint32_t bIdx);
    uint32_t GetS2SeqSize(uint32_t bIdx);
    void CalcSplitInfo();

    CpuKernelContext* context_ = nullptr;
    Tensor* cuSeqlensQ_ = nullptr;
    Tensor* cuSeqlensK_ = nullptr;
    Tensor* sequsedQ_ = nullptr;
    Tensor* sequsedK_ = nullptr;
    Tensor* cmpResidualK_ = nullptr;
    Tensor* metadata_ = nullptr;

    std::string socVersion_ = "";
    bool supportFd_ = false;
    int32_t cmpRatio_ = 4;
    uint32_t aicCoreNum_ = 24U;
    uint32_t aivCoreNum_ = 48U;
    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = 0;
    int32_t maxSeqlenK_ = 0;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsK_ = 0;
    int32_t headDim_ = 0;
    int32_t topk_ = 0;
    std::string layoutQ_ = "BSND";
    std::string layoutK_ = "BSND";
    int32_t maskMode_ = 0;
    uint32_t attentionMode_ = 0;
    ValidSocVersionArch92 validSocVersion_ = ValidSocVersionArch92::ASCEND910;

    int64_t preToken_ = INT64_MAX;
    int64_t nextToken_ = INT64_MAX;
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = 256;
    uint32_t s2BaseSize_ = 0;
    bool isS1G_ = true;
    uint32_t totalNum = 0;

    enum class ParamId : uint32_t {
        actSeqLenQ = 0,
        actSeqLenK = 1,
        seqUsedQ = 2,
        seqUsedK = 3,
        cmpResidualK = 4,
        metadata = 0,
    };
};

inline uint32_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::Compute(CpuKernelContext& ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    success = BalanceSchedule() && GenMetadata();
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::Prepare(CpuKernelContext& ctx)
{
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
    cuSeqlensK_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenK));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedQ));
    sequsedK_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedK));
    cmpResidualK_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpResidualK));
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs =
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_) &&
        GetAttrValue(ctx, "soc_version", socVersion_) && GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
        GetAttrValue(ctx, "num_heads_k", numHeadsK_) && GetAttrValue(ctx, "head_dim", headDim_);
    if (!requiredAttrs) {
        return false;
    }

    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_k", maxSeqlenK_);
    GetAttrValueOpt(ctx, "topk", topk_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_k", layoutK_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);

    return (ParamsCheck() && ParamsInit());
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::ParamsCheck()
{
    // 校验输出 metadata 是否为空
    if (metadata_ == nullptr) {
        KERNEL_LOG_ERROR("Output metadata is nullptr");
        return false;
    } else if (metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("Output metadata data is nullptr");
        return false;
    }
    int32_t batchSize = GetQueryBatchSize();
    // 校验 cu_seqlens_q 元素
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t* cuSeqlensQPtr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
            // 校验 cu_seqlens_q 首元素为 0
            if (cuSeqlensQPtr[0] != 0) {
                KERNEL_LOG_ERROR("The first element of cu_seqlens_q should be 0, but got %d", cuSeqlensQPtr[0]);
                return false;
            }
            for (int i = 0; i < batchSize + 1; i++) {
                // 校验 cu_seqlens_q 元素递增
                if (i > 0 && cuSeqlensQPtr[i - 1] > cuSeqlensQPtr[i]) {
                    KERNEL_LOG_ERROR("The elements in cu_seqlens_q must be in ascending order, "
                                     "but got cu_seqlens_q[%d] = %d, cu_seqlens_q[%d] = %d",
                                     i - 1, cuSeqlensQPtr[i - 1], i, cuSeqlensQPtr[i]);
                    return false;
                }
            }
        }
    }
    // 校验 cu_seqlens_k 元素
    if (layoutK_ == "TND") {
        if (cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) {
            const int32_t* cuSeqlensKPtr = static_cast<const int32_t*>(cuSeqlensK_->GetData());
            // 校验 cu_seqlens_k 首元素为 0
            if (cuSeqlensKPtr[0] != 0) {
                KERNEL_LOG_ERROR("The first element of cu_seqlens_k should be 0, but got %d", cuSeqlensKPtr[0]);
                return false;
            }
            for (int i = 0; i < batchSize + 1; i++) {
                // 校验 cu_seqlens_k 元素递增
                if (i > 0 && cuSeqlensKPtr[i - 1] > cuSeqlensKPtr[i]) {
                    KERNEL_LOG_ERROR("The elements in cu_seqlens_k must be in ascending order, "
                                     "but got cu_seqlens_k[%d] = %d, cu_seqlens_k[%d] = %d",
                                     i - 1, cuSeqlensKPtr[i - 1], i, cuSeqlensKPtr[i]);
                    return false;
                }
            }
        }
    }
    // 校验 seqused_q 元素非负
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t* sequsedQPtr = static_cast<const int32_t*>(sequsedQ_->GetData());
        const int32_t* cuSeqlensQPtr =
            (layoutQ_ == "TND" && cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) ?
                static_cast<const int32_t*>(cuSeqlensQ_->GetData()) :
                nullptr;
        for (int i = 0; i < batchSize; i++) {
            if (sequsedQPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_q should be >= 0, but got seqused_q[%d] = %d", i,
                                 sequsedQPtr[i]);
                return false;
            }
            // 校验 seqused_q 元素不大于 max_seqlen_q (BSND) 或 cu_seqlens_q 序列长度 (TND)
            if (layoutQ_ == "BSND" && sequsedQPtr[i] > maxSeqlenQ_) {
                KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than max_seqlen_q %d, "
                                 "but got seqused_q[%d] = %d",
                                 maxSeqlenQ_, i, sequsedQPtr[i]);
                return false;
            }
            if (cuSeqlensQPtr != nullptr) {
                int32_t seqLen = cuSeqlensQPtr[i + 1] - cuSeqlensQPtr[i];
                if (sequsedQPtr[i] > seqLen) {
                    KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than the sequence length "
                                     "from cu_seqlens_q %d, but got seqused_q[%d] = %d",
                                     seqLen, i, sequsedQPtr[i]);
                    return false;
                }
            }
        }
    }
    // 校验 seqused_k 元素非负
    if (sequsedK_ != nullptr && sequsedK_->GetData() != nullptr) {
        const int32_t* sequsedKPtr = static_cast<const int32_t*>(sequsedK_->GetData());
        for (int i = 0; i < batchSize; i++) {
            if (sequsedKPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_k should be >= 0, but got seqused_k[%d] = %d", i,
                                 sequsedKPtr[i]);
                return false;
            }
        }
    }
    // 校验 cmp_residual_k 元素非负
    if (cmpResidualK_ != nullptr && cmpResidualK_->GetData() != nullptr) {
        const int32_t* cmpResidualKPtr = static_cast<const int32_t*>(cmpResidualK_->GetData());
        for (int i = 0; i < batchSize; i++) {
            if (cmpResidualKPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in cmp_residual_k should be >= 0, but got cmp_residual_k[%d] = %d", i,
                                 cmpResidualKPtr[i]);
                return false;
            }
        }
    }
    return true;
}

inline int32_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::GetQueryBatchSize()
{
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        if (sequsedQ_->GetTensorShape() != nullptr) {
            return sequsedQ_->GetTensorShape()->GetDimSize(0);
        }
    }
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            if (cuSeqlensQ_->GetTensorShape() != nullptr) {
                return cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
            }
        }
    }
    return batchSize_;
}

inline ValidSocVersionArch92 SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::ProcessSocVersion()
{
    if (socVersion_.find("Ascend910") != std::string::npos) {
        return ValidSocVersionArch92::ASCEND910;
    } else if (socVersion_.find("Ascend950") != std::string::npos) {
        return ValidSocVersionArch92::ASCEND950;
    } else {
        return ValidSocVersionArch92::ASCEND960;
    }
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    auto mode = static_cast<SparseModeArch92>(maskMode_);
    if (mode == SparseModeArch92::RIGHT_DOWN_CAUSAL) {
        attentionMode_ = 1;
        preToken_ = INT64_MAX;
    } else if (mode == SparseModeArch92::DEFAULT_MASK) {
        attentionMode_ = 0;
    } else if (mode == SparseModeArch92::BAND) {
        attentionMode_ = 1;
    }
    groupSize_ = numHeadsQ_ / numHeadsK_;
    validSocVersion_ = ProcessSocVersion();
    if (validSocVersion_ == ValidSocVersionArch92::ASCEND910) {
        s2BaseSize_ = 2048U;
    } else if (validSocVersion_ == ValidSocVersionArch92::ASCEND950) {
        s2BaseSize_ = 128U;
    } else {
        s2BaseSize_ = 128U;
    }
    return true;
}

inline uint32_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::GetS1SeqSize(uint32_t bIdx)
{
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t* seqUsedPtr = static_cast<const int32_t*>(sequsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t* s1Ptr = static_cast<const int32_t*>(cuSeqlensQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    return static_cast<uint32_t>(maxSeqlenQ_);
}

inline uint32_t SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::GetS2SeqSize(uint32_t bIdx)
{
    if (sequsedK_ != nullptr && sequsedK_->GetData() != nullptr) {
        const int32_t* seqUsedPtr = static_cast<const int32_t*>(sequsedK_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    if (layoutK_ == "TND") {
        if (cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) {
            const int32_t* s2Ptr = static_cast<const int32_t*>(cuSeqlensK_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    return static_cast<uint32_t>(maxSeqlenK_);
}

inline void SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::CalcSplitInfo()
{
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        totalNum += s1Size;
    }
    return;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::BalanceSchedule()
{
    CalcSplitInfo();
    return true;
}

inline bool SparseLightningIndexerKLLossGradMetadataCpuKernelArch92::GenMetadata()
{
    optiling::detail::SlikgMetadataArch92* gradMetadataPtr =
        static_cast<optiling::detail::SlikgMetadataArch92*>(metadata_->GetData());
    uint32_t formerCoreProcessNum = CeilDiv(totalNum, aicCoreNum_);
    uint32_t remainCoreProcessNum = formerCoreProcessNum - 1;
    uint32_t remainCoreNum = formerCoreProcessNum * aicCoreNum_ - totalNum;
    uint32_t usedCoreNum = totalNum < aicCoreNum_ ? totalNum : aicCoreNum_;

    gradMetadataPtr->gradMetadata[optiling::TOTAL_NUM_ARCH92] = totalNum;
    gradMetadataPtr->gradMetadata[optiling::FORMER_CORE_PROCESS_NUM_ARCH92] = formerCoreProcessNum;
    gradMetadataPtr->gradMetadata[optiling::REMAIN_CORE_PROCESS_NUM_ARCH92] = remainCoreProcessNum;
    gradMetadataPtr->gradMetadata[optiling::REMAIN_CORE_NUM_ARCH92] = remainCoreNum;
    gradMetadataPtr->gradMetadata[optiling::USED_CORE_NUM_ARCH92] = usedCoreNum;
    return true;
}

} // namespace aicpu

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_AICPU_ARCH92_H
