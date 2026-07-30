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
 * \file sparse_flash_mla_softmax_l1_norm_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include "../../sparse_flash_mla_softmax_l1_norm/op_kernel/sparse_flash_mla_softmax_l1_norm_metadata.h"
#include "sparse_flash_mla_softmax_l1_norm_metadata_aicpu.h"
#include "arch35/sparse_flash_mla_softmax_l1_norm_metadata_aicpu_arch35.h"

using namespace optiling;

namespace aicpu {

uint32_t SparseFlashMlaSoftmaxL1NormMetadataCpuKernel::Compute(CpuKernelContext &ctx)
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

bool SparseFlashMlaSoftmaxL1NormMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    cuSeqLensQuery_ = ctx.Input(0);
    cuSeqLensKey_ = ctx.Input(1);
    seqUsedQuery_ = ctx.Input(2);
    seqUsedKey_ = ctx.Input(3);
    cmpResidualKey_ = ctx.Input(4);
    topkLength_ = ctx.Input(5);
    metadata_ = ctx.Output(0);

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_k", numHeadsK_) &&
                         GetAttrValue(ctx, "head_dim", headDim_);
    if (!requiredAttrs) {
        return false;
    }

    GetAttrValueOpt(ctx, "batch_size", bSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", s1Size_);
    GetAttrValueOpt(ctx, "max_seqlen_k", s2Size_);
    GetAttrValueOpt(ctx, "topk", topk_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "layout_q", layout_);
    GetAttrValueOpt(ctx, "layout_k", layoutK_);
    GetAttrValueOpt(ctx, "soc_version", socVersion_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);

    if (layout_ == "BSND") {
        layoutType_ = SmlaLayout::BSND;
    } else if (layout_ == "TND") {
        layoutType_ = SmlaLayout::TND;
    } else {
        KERNEL_LOG_ERROR("layout_q must be BSND or TND, but got %s", layout_.c_str());
        return false;
    }

    if (bSize_ <= 0) {
        if (IsTensorValid(seqUsedQuery_)) {
            bSize_ = seqUsedQuery_->GetTensorShape()->GetDimSize(0);
        } else if (layoutType_ == SmlaLayout::TND && IsTensorValid(cuSeqLensQuery_)) {
            bSize_ = cuSeqLensQuery_->GetTensorShape()->GetDimSize(0) - 1;
        }
    }

    return ParamsCheck();
}

ValidSocVersion SparseFlashMlaSoftmaxL1NormMetadataCpuKernel::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    }
    return ValidSocVersion::ASCEND910;
}

bool SparseFlashMlaSoftmaxL1NormMetadataCpuKernel::ParamsCheck()
{
    KERNEL_CHECK_NULLPTR(metadata_, false, "metadata output is null");
    KERNEL_CHECK_NULLPTR(metadata_->GetData(), false, "metadata data is null");
    KERNEL_CHECK_NULLPTR(metadata_->GetTensorShape(), false, "metadata shape is null");

    if (aicCoreNum_ <= 0) {
        KERNEL_LOG_ERROR("aic_core_num must be positive, but got %ld", aicCoreNum_);
        return false;
    }
    if (numHeadsQ_ <= 0 || numHeadsK_ <= 0 || headDim_ <= 0) {
        KERNEL_LOG_ERROR("num_heads_q/num_heads_k/head_dim must be positive, but got nq=%ld nk=%ld d=%ld", numHeadsQ_,
                         numHeadsK_, headDim_);
        return false;
    }
    if (numHeadsQ_ % numHeadsK_ != 0) {
        KERNEL_LOG_ERROR("num_heads_q must be divisible by num_heads_k, but got nq=%ld nk=%ld", numHeadsQ_, numHeadsK_);
        return false;
    }
    if (bSize_ <= 0) {
        KERNEL_LOG_ERROR("batch_size must be positive, but got %ld", bSize_);
        return false;
    }
    if (layoutType_ == SmlaLayout::BSND && s1Size_ <= 0) {
        KERNEL_LOG_ERROR("max_seqlen_q must be positive for BSND, but got %ld", s1Size_);
        return false;
    }
    if (s1Size_ < 0 || s2Size_ < 0) {
        KERNEL_LOG_ERROR("max_seqlen_q/max_seqlen_k must be non-negative, but got q=%ld k=%ld", s1Size_, s2Size_);
        return false;
    }
    if (maskMode_ != static_cast<int64_t>(SmlaSparseMode::NO_MASK) &&
        maskMode_ != static_cast<int64_t>(SmlaSparseMode::RIGHT_DOWN_CAUSAL)) {
        KERNEL_LOG_ERROR("mask_mode only supports 0 or 3, but got %ld", maskMode_);
        return false;
    }
    if (cmpRatio_ < 1 || cmpRatio_ > 128) {
        KERNEL_LOG_ERROR("cmp_ratio must be in [1, 128], but got %ld", cmpRatio_);
        return false;
    }
    if (layoutType_ == SmlaLayout::TND && !IsTensorValid(cuSeqLensQuery_)) {
        KERNEL_LOG_ERROR("cu_seqlens_q must be provided for TND layout.");
        return false;
    }
    if (layoutType_ == SmlaLayout::TND && !IsTensorValid(cuSeqLensKey_)) {
        KERNEL_LOG_ERROR("cu_seqlens_k must be provided for TND layout.");
        return false;
    }
    if (IsTensorValid(cuSeqLensQuery_) && cuSeqLensQuery_->GetTensorShape()->GetDimSize(0) != bSize_ + 1) {
        KERNEL_LOG_ERROR("cu_seqlens_q length must be batch_size + 1, but got %ld and batch_size=%ld.",
                         cuSeqLensQuery_->GetTensorShape()->GetDimSize(0), bSize_);
        return false;
    }
    if (IsTensorValid(cuSeqLensKey_) && cuSeqLensKey_->GetTensorShape()->GetDimSize(0) != bSize_ + 1) {
        KERNEL_LOG_ERROR("cu_seqlens_k length must be batch_size + 1, but got %ld and batch_size=%ld.",
                         cuSeqLensKey_->GetTensorShape()->GetDimSize(0), bSize_);
        return false;
    }
    if (IsTensorValid(seqUsedQuery_) && seqUsedQuery_->GetTensorShape()->GetDimSize(0) != bSize_) {
        KERNEL_LOG_ERROR("seqused_q length must be batch_size, but got %ld and batch_size=%ld.",
                         seqUsedQuery_->GetTensorShape()->GetDimSize(0), bSize_);
        return false;
    }
    if (IsTensorValid(seqUsedKey_) && seqUsedKey_->GetTensorShape()->GetDimSize(0) != bSize_) {
        KERNEL_LOG_ERROR("seqused_k length must be batch_size, but got %ld and batch_size=%ld.",
                         seqUsedKey_->GetTensorShape()->GetDimSize(0), bSize_);
        return false;
    }
    if (IsTensorValid(cmpResidualKey_) && cmpResidualKey_->GetTensorShape()->GetDimSize(0) != bSize_) {
        KERNEL_LOG_ERROR("cmp_residual_k length must be batch_size, but got %ld and batch_size=%ld.",
                         cmpResidualKey_->GetTensorShape()->GetDimSize(0), bSize_);
        return false;
    }
    if (IsTensorValid(topkLength_) && topkLength_->GetTensorShape()->GetDims() < 1) {
        KERNEL_LOG_ERROR("topk_length must be at least 1-dimensional.");
        return false;
    }
    return true;
}

namespace {
static const char *kernelType = "SparseFlashMlaSoftmaxL1NormMetadata";
REGISTER_CPU_KERNEL(kernelType, SparseFlashMlaSoftmaxL1NormMetadataCpuKernel);
} // namespace
} // namespace aicpu
