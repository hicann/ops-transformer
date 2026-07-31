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
 * \\file dense_lightning_indexer_softmax_lse_v2_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include "../../dense_lightning_indexer_softmax_lse_v2/op_kernel/dense_lightning_indexer_softmax_lse_v2_metadata.h"
#include "dense_lightning_indexer_softmax_lse_v2_metadata_aicpu.h"
#include "arch35/dense_lightning_indexer_softmax_lse_v2_metadata_aicpu_arch35.h"

using namespace optiling;

namespace aicpu {

uint32_t DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    if (!Prepare(ctx)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    ValidSocVersion socVersion = ProcessSocVersion();
    if (socVersion == ValidSocVersion::ASCEND950) {
        return BuildMetadata() ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_LOG_ERROR("Unsupported soc version: %s", socVersion_.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
}

bool DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel::Prepare(CpuKernelContext &ctx)
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
    GetAttrValueOpt(ctx, "layout_q", layout_);
    GetAttrValueOpt(ctx, "layout_k", layoutK_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
    GetAttrValueOpt(ctx, "soc_version", socVersion_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);

    if (layout_ == "BSND") {
        layoutType_ = DliLayout::BSND;
    } else if (layout_ == "TND") {
        layoutType_ = DliLayout::TND;
    } else {
        KERNEL_LOG_ERROR("layout_q must be BSND or TND, but got %s", layout_.c_str());
        return false;
    }

    if (bSize_ <= 0) {
        if (IsTensorValid(seqUsedQuery_)) {
            bSize_ = seqUsedQuery_->GetTensorShape()->GetDimSize(0);
        } else if (layoutType_ == DliLayout::TND && IsTensorValid(cuSeqLensQuery_)) {
            bSize_ = cuSeqLensQuery_->GetTensorShape()->GetDimSize(0) - 1;
        }
    }

    return ParamsCheck();
}

ValidSocVersion DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    }
    return ValidSocVersion::ASCEND910;
}

bool DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel::ParamsCheck()
{
    KERNEL_CHECK_NULLPTR(metadata_, false, "metadata output is null");
    KERNEL_CHECK_NULLPTR(metadata_->GetData(), false, "metadata data is null");
    KERNEL_CHECK_NULLPTR(metadata_->GetTensorShape(), false, "metadata shape is null");

    return CheckTensorValues();
}

bool DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel::CheckTensorValues()
{
    if (layoutType_ == DliLayout::TND && IsTensorValid(cuSeqLensQuery_)) {
        auto *cuSeqQ = reinterpret_cast<const int32_t *>(cuSeqLensQuery_->GetData());
        for (int64_t i = 0; i < bSize_ + 1; ++i) {
            if (static_cast<int64_t>(cuSeqQ[i]) < 0) {
                KERNEL_LOG_ERROR("The elements in cu_seqlens_q should be >= 0, but got cu_seqlens_q[%ld] = %d", i,
                                 cuSeqQ[i]);
                return false;
            }
            if (i > 0 && static_cast<int64_t>(cuSeqQ[i - 1]) > static_cast<int64_t>(cuSeqQ[i])) {
                KERNEL_LOG_ERROR("The elements in cu_seqlens_q must be in ascending order, "
                                 "but got cu_seqlens_q[%ld] = %d, cu_seqlens_q[%ld] = %d",
                                 i - 1, cuSeqQ[i - 1], i, cuSeqQ[i]);
                return false;
            }
        }
    }
    if (layoutType_ == DliLayout::TND && IsTensorValid(cuSeqLensKey_)) {
        auto *cuSeqK = reinterpret_cast<const int32_t *>(cuSeqLensKey_->GetData());
        for (int64_t i = 0; i < bSize_ + 1; ++i) {
            if (static_cast<int64_t>(cuSeqK[i]) < 0) {
                KERNEL_LOG_ERROR("The elements in cu_seqlens_k should be >= 0, but got cu_seqlens_k[%ld] = %d", i,
                                 cuSeqK[i]);
                return false;
            }
            if (i > 0 && static_cast<int64_t>(cuSeqK[i - 1]) > static_cast<int64_t>(cuSeqK[i])) {
                KERNEL_LOG_ERROR("The elements in cu_seqlens_k must be in ascending order, "
                                 "but got cu_seqlens_k[%ld] = %d, cu_seqlens_k[%ld] = %d",
                                 i - 1, cuSeqK[i - 1], i, cuSeqK[i]);
                return false;
            }
        }
    }
    if (IsTensorValid(seqUsedQuery_)) {
        auto *seqUsedQ = reinterpret_cast<const int32_t *>(seqUsedQuery_->GetData());
        for (int64_t i = 0; i < bSize_; ++i) {
            if (static_cast<int64_t>(seqUsedQ[i]) < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_q should be >= 0, but got seqused_q[%ld] = %d",
                                 i, seqUsedQ[i]);
                return false;
            }
        }
    }
    if (IsTensorValid(seqUsedKey_)) {
        auto *seqUsedK = reinterpret_cast<const int32_t *>(seqUsedKey_->GetData());
        for (int64_t i = 0; i < bSize_; ++i) {
            if (static_cast<int64_t>(seqUsedK[i]) < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_k should be >= 0, but got seqused_k[%ld] = %d",
                                 i, seqUsedK[i]);
                return false;
            }
        }
    }
    if (IsTensorValid(cmpResidualKey_)) {
        auto *data = reinterpret_cast<const int32_t *>(cmpResidualKey_->GetData());
        for (int64_t i = 0; i < bSize_; ++i) {
            if (static_cast<int64_t>(data[i]) < 0) {
                KERNEL_LOG_ERROR("The elements in cmp_residual_k should be >= 0, but got cmp_residual_k[%ld] = %d", i,
                                 data[i]);
                return false;
            }
            if (static_cast<int64_t>(data[i]) >= cmpRatio_) {
                KERNEL_LOG_ERROR("cmp_residual_k[%ld] must be less than cmp_ratio(%ld), but got %d.", i, cmpRatio_,
                                 data[i]);
                return false;
            }
        }
    }
    if (IsTensorValid(seqUsedQuery_)) {
        auto *seqUsedQ = reinterpret_cast<const int32_t *>(seqUsedQuery_->GetData());
        if (layoutType_ == DliLayout::TND && IsTensorValid(cuSeqLensQuery_)) {
            auto *cuSeqQ = reinterpret_cast<const int32_t *>(cuSeqLensQuery_->GetData());
            for (int64_t i = 0; i < bSize_; ++i) {
                int64_t seqlenQ = static_cast<int64_t>(cuSeqQ[i + 1]) - static_cast<int64_t>(cuSeqQ[i]);
                if (static_cast<int64_t>(seqUsedQ[i]) > seqlenQ) {
                    KERNEL_LOG_ERROR("seqused_q[%ld]=%d exceeds seqlens_q[%ld]=%ld", i, seqUsedQ[i], i, seqlenQ);
                    return false;
                }
            }
        } else {
            for (int64_t i = 0; i < bSize_; ++i) {
                if (static_cast<int64_t>(seqUsedQ[i]) > s1Size_) {
                    KERNEL_LOG_ERROR("seqused_q[%ld]=%d exceeds max_seqlen_q=%ld", i, seqUsedQ[i], s1Size_);
                    return false;
                }
            }
        }
    }
    if (IsTensorValid(seqUsedKey_)) {
        auto *seqUsedK = reinterpret_cast<const int32_t *>(seqUsedKey_->GetData());
        if (layoutType_ == DliLayout::TND && IsTensorValid(cuSeqLensKey_)) {
            auto *cuSeqK = reinterpret_cast<const int32_t *>(cuSeqLensKey_->GetData());
            for (int64_t i = 0; i < bSize_; ++i) {
                int64_t seqlenK = static_cast<int64_t>(cuSeqK[i + 1]) - static_cast<int64_t>(cuSeqK[i]);
                if (static_cast<int64_t>(seqUsedK[i]) > seqlenK) {
                    KERNEL_LOG_ERROR("seqused_k[%ld]=%d exceeds seqlens_k[%ld]=%ld", i, seqUsedK[i], i, seqlenK);
                    return false;
                }
            }
        } else {
            for (int64_t i = 0; i < bSize_; ++i) {
                if (static_cast<int64_t>(seqUsedK[i]) > s2Size_) {
                    KERNEL_LOG_ERROR("seqused_k[%ld]=%d exceeds max_seqlen_k=%ld", i, seqUsedK[i], s2Size_);
                    return false;
                }
            }
        }
    }
    return true;
}

static const char *kernelTypeDenseLiV2 = "DenseLightningIndexerSoftmaxLseV2Metadata";
REGISTER_CPU_KERNEL(kernelTypeDenseLiV2, DenseLightningIndexerSoftmaxLseV2MetadataCpuKernel);
} // namespace aicpu
