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
 * \file sparse_flash_mla_grad_metadata_aicpu.cpp
 * \brief
 */
#include "sparse_flash_mla_grad_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
uint32_t SparseFlashMlaGradMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes{aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool SparseFlashMlaGradMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensOriKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensOriKv));
    cuSeqlensCmpKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensCmpKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedOriKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedOriKv));
    sequsedCmpKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedCmpKv));
    cmpResidualKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpResidualKv));
    oriTopkLength_ = ctx.Input(static_cast<uint32_t>(ParamId::oriTopkLength));
    cmpTopkLength_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpTopkLength));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                         GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) && GetAttrValue(ctx, "head_dim", headDim_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "soc_version", socVersion_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);
    GetAttrValueOpt(ctx, "aiv_core_num", aivCoreNum_);
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_ori_kv", maxSeqlenOriKv_);
    GetAttrValueOpt(ctx, "max_seqlen_cmp_kv", maxSeqlenCmpKv_);
    GetAttrValueOpt(ctx, "ori_topk", oriTopK_);
    GetAttrValueOpt(ctx, "cmp_topk", cmpTopK_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
    GetAttrValueOpt(ctx, "ori_mask_mode", oriMaskMode_);
    GetAttrValueOpt(ctx, "cmp_mask_mode", cmpMaskMode_);
    GetAttrValueOpt(ctx, "ori_win_left", oriWinLeft_);
    GetAttrValueOpt(ctx, "ori_win_right", oriWinRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "has_ori_kv", hasOriKv_);
    GetAttrValueOpt(ctx, "has_cmp_kv", hasCmpKv_);

    return (ParamsCheck() && ParamsInit());
}

bool SparseFlashMlaGradMetadataCpuKernel::ParamsCheck()
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
            const int32_t *cuSeqlensQPtr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
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
    // 校验 seqused_q 元素
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *sequsedQPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
        const int32_t *cuSeqlensQPtr = (layoutQ_ == "TND" && cuSeqlensQ_ != nullptr &&
                                        cuSeqlensQ_->GetData() != nullptr) ?
                                           static_cast<const int32_t *>(cuSeqlensQ_->GetData()) : nullptr;
        for (int i = 0; i < batchSize; i++) {
            // 校验 seqused_q 元素非负
            if (sequsedQPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_q should be >= 0, but got seqused_q[%d] = %d", i,
                                 sequsedQPtr[i]);
                return false;
            }
            // 校验 seqused_q 元素不大于 max_seqlen_q (BSND) 或 cu_seqlens_q 序列长度 (TND)
            if (layoutQ_ == "BSND" && sequsedQPtr[i] > maxSeqlenQ_) {
                KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than max_seqlen_q %d, "
                                 "but got seqused_q[%d] = %d", maxSeqlenQ_, i, sequsedQPtr[i]);
                return false;
            }
            if (cuSeqlensQPtr != nullptr) {
                int32_t seqLen = cuSeqlensQPtr[i + 1] - cuSeqlensQPtr[i];
                if (sequsedQPtr[i] > seqLen) {
                    KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than the sequence length "
                                     "from cu_seqlens_q %d, but got seqused_q[%d] = %d", seqLen, i, sequsedQPtr[i]);
                    return false;
                }
            }
        }
    }
    if (hasOriKv_) {
        // 校验 cu_seqlens_ori_kv 元素
        if (layoutKv_ == "TND") {
            if (cuSeqlensOriKv_ != nullptr && cuSeqlensOriKv_->GetData() != nullptr) {
                const int32_t *cuSeqlensOriKvPtr = static_cast<const int32_t *>(cuSeqlensOriKv_->GetData());
                // 校验 cu_seqlens_ori_kv 首元素为 0
                if (cuSeqlensOriKvPtr[0] != 0) {
                    KERNEL_LOG_ERROR("The first element of cu_seqlens_ori_kv should be 0, but got %d",
                                     cuSeqlensOriKvPtr[0]);
                    return false;
                }
                for (int i = 0; i < batchSize + 1; i++) {
                    // 校验 cu_seqlens_ori_kv 元素递增
                    if (i > 0 && cuSeqlensOriKvPtr[i - 1] > cuSeqlensOriKvPtr[i]) {
                        KERNEL_LOG_ERROR("The elements in cu_seqlens_ori_kv must be in ascending order, "
                                         "but got cu_seqlens_ori_kv[%d] = %d, cu_seqlens_ori_kv[%d] = %d",
                                         i - 1, cuSeqlensOriKvPtr[i - 1], i, cuSeqlensOriKvPtr[i]);
                        return false;
                    }
                }
            }
        }
        // 校验 seqused_ori_kv 元素
        if (sequsedOriKv_ != nullptr && sequsedOriKv_->GetData() != nullptr) {
            const int32_t *sequsedOriKvPtr = static_cast<const int32_t *>(sequsedOriKv_->GetData());
            const int32_t *cuSeqlensOriKvPtr = (layoutKv_ == "TND" && cuSeqlensOriKv_ != nullptr &&
                                                cuSeqlensOriKv_->GetData() != nullptr) ?
                                                   static_cast<const int32_t *>(cuSeqlensOriKv_->GetData()) :
                                                   nullptr;
            for (int i = 0; i < batchSize; i++) {
                // 校验 seqused_ori_kv 元素非负
                if (sequsedOriKvPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in seqused_ori_kv should be >= 0, but got seqused_ori_kv[%d] = %d",
                                     i, sequsedOriKvPtr[i]);
                    return false;
                }
                // 校验 seqused_ori_kv 元素不大于 max_seqlen_ori_kv (BSND) 或 cu_seqlens_ori_kv 序列长度 (TND)
                if (layoutKv_ == "BSND" && sequsedOriKvPtr[i] > maxSeqlenOriKv_) {
                    KERNEL_LOG_ERROR("The elements in seqused_ori_kv should not be greater than "
                                     "max_seqlen_ori_kv %d, but got seqused_ori_kv[%d] = %d",
                                     maxSeqlenOriKv_, i, sequsedOriKvPtr[i]);
                    return false;
                }
                if (cuSeqlensOriKvPtr != nullptr) {
                    int32_t seqLen = cuSeqlensOriKvPtr[i + 1] - cuSeqlensOriKvPtr[i];
                    if (sequsedOriKvPtr[i] > seqLen) {
                        KERNEL_LOG_ERROR("The elements in seqused_ori_kv should not be greater than the sequence "
                                         "length from cu_seqlens_ori_kv %d, but got seqused_ori_kv[%d] = %d",
                                         seqLen, i, sequsedOriKvPtr[i]);
                        return false;
                    }
                }
            }
        }
        // 校验 ori_topk_length 元素
        if (oriTopK_ != 0 && oriMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
            oriTopkLength_ != nullptr && oriTopkLength_->GetData() != nullptr) {
            // 校验 ori_topk_length 元素数量
            int32_t sumOfQuerySeq = GetSumOfQuerySeq();
            const int32_t *oriTopkLengthPtr = static_cast<const int32_t *>(oriTopkLength_->GetData());
            auto oriTopkLengthShape = oriTopkLength_->GetTensorShape();
            int32_t oriTopkLengthSize = layoutQ_ == "TND" ?
                                            oriTopkLengthShape->GetDimSize(0) * oriTopkLengthShape->GetDimSize(1) :
                                            oriTopkLengthShape->GetDimSize(0) * oriTopkLengthShape->GetDimSize(1) *
                                                oriTopkLengthShape->GetDimSize(2);
            if (oriTopkLengthSize < sumOfQuerySeq) {
                KERNEL_LOG_ERROR("The size of ori_topk_length %d should not be smaller than "
                                 "the sum of query sequence %d!",
                                 oriTopkLengthSize, sumOfQuerySeq);
                return false;
            }
            // 校验 ori_topk_length 元素非负
            for (int i = 0; i < oriTopkLengthSize; i++) {
                if (oriTopkLengthPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in ori_topk_length should be >= 0, but got ori_topk_length[%d] = %d",
                                     i, oriTopkLengthPtr[i]);
                    return false;
                }
            }
        }
    }
    if (hasCmpKv_) {
        if (layoutKv_ == "TND") {
            // 校验 cu_seqlens_cmp_kv 元素
            if (cuSeqlensCmpKv_ != nullptr && cuSeqlensCmpKv_->GetData() != nullptr) {
                const int32_t *cuSeqlensCmpKvPtr = static_cast<const int32_t *>(cuSeqlensCmpKv_->GetData());
                // 校验 cu_seqlens_cmp_kv 首元素为 0
                if (cuSeqlensCmpKvPtr[0] != 0) {
                    KERNEL_LOG_ERROR("The first element of cu_seqlens_cmp_kv should be 0, but got %d",
                                     cuSeqlensCmpKvPtr[0]);
                    return false;
                }
                for (int i = 0; i < batchSize + 1; i++) {
                    // 校验 cu_seqlens_cmp_kv 元素递增
                    if (i > 0 && cuSeqlensCmpKvPtr[i - 1] > cuSeqlensCmpKvPtr[i]) {
                        KERNEL_LOG_ERROR("The elements in cu_seqlens_cmp_kv must be in ascending order, "
                                         "but got cu_seqlens_cmp_kv[%d] = %d, cu_seqlens_cmp_kv[%d] = %d",
                                         i - 1, cuSeqlensCmpKvPtr[i - 1], i, cuSeqlensCmpKvPtr[i]);
                        return false;
                    }
                }
            }
        }
        // 校验 seqused_cmp_kv 元素
        if (sequsedCmpKv_ != nullptr && sequsedCmpKv_->GetData() != nullptr) {
            const int32_t *sequsedCmpKvPtr = static_cast<const int32_t *>(sequsedCmpKv_->GetData());
            const int32_t *cuSeqlensCmpKvPtr = (layoutKv_ == "TND" && cuSeqlensCmpKv_ != nullptr &&
                                                cuSeqlensCmpKv_->GetData() != nullptr) ?
                                                   static_cast<const int32_t *>(cuSeqlensCmpKv_->GetData()) :
                                                   nullptr;
            for (int i = 0; i < batchSize; i++) {
                // 校验 seqused_cmp_kv 元素非负
                if (sequsedCmpKvPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in seqused_cmp_kv should be >= 0, but got seqused_cmp_kv[%d] = %d",
                                     i, sequsedCmpKvPtr[i]);
                    return false;
                }
                // 校验 seqused_cmp_kv 元素不大于 max_seqlen_cmp_kv (BSND) 或 cu_seqlens_cmp_kv 序列长度 (TND)
                if (layoutKv_ == "BSND" && sequsedCmpKvPtr[i] > maxSeqlenCmpKv_) {
                    KERNEL_LOG_ERROR("The elements in seqused_cmp_kv should not be greater than "
                                     "max_seqlen_cmp_kv %d, but got seqused_cmp_kv[%d] = %d",
                                     maxSeqlenCmpKv_, i, sequsedCmpKvPtr[i]);
                    return false;
                }
                if (cuSeqlensCmpKvPtr != nullptr) {
                    int32_t seqLen = cuSeqlensCmpKvPtr[i + 1] - cuSeqlensCmpKvPtr[i];
                    if (sequsedCmpKvPtr[i] > seqLen) {
                        KERNEL_LOG_ERROR("The elements in seqused_cmp_kv should not be greater than the sequence "
                                         "length from cu_seqlens_cmp_kv %d, but got seqused_cmp_kv[%d] = %d",
                                         seqLen, i, sequsedCmpKvPtr[i]);
                        return false;
                    }
                }
            }
        }
        // 校验 cmp_residual_kv 元素
        if (cmpResidualKv_ != nullptr && cmpResidualKv_->GetData() != nullptr) {
            const int32_t *cmpResidualKvPtr = static_cast<const int32_t *>(cmpResidualKv_->GetData());
            for (int i = 0; i < batchSize; i++) {
                if (cmpResidualKvPtr[i] < 0 || cmpResidualKvPtr[i] >= cmpRatio_) {
                    KERNEL_LOG_ERROR("The elements in cmp_residual_kv should be in [0, cmpRatio_(%d)), but got "
                                     "cmp_residual_kv[%d] = %d", cmpRatio_,
                                     i, cmpResidualKvPtr[i]);
                    return false;
                }
            }
        }
        // 校验 cmp_topk_length 元素
        if (cmpTopK_ != 0 && cmpMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
            cmpTopkLength_ != nullptr && cmpTopkLength_->GetData() != nullptr) {
            // 校验 cmp_topk_length 元素数量
            int32_t sumOfQuerySeq = GetSumOfQuerySeq();
            const int32_t *cmpTopkLengthPtr = static_cast<const int32_t *>(cmpTopkLength_->GetData());
            auto cmpTopkLengthShape = cmpTopkLength_->GetTensorShape();
            int32_t cmpTopkLengthSize = layoutQ_ == "TND" ?
                                            cmpTopkLengthShape->GetDimSize(0) * cmpTopkLengthShape->GetDimSize(1) :
                                            cmpTopkLengthShape->GetDimSize(0) * cmpTopkLengthShape->GetDimSize(1) *
                                                cmpTopkLengthShape->GetDimSize(2);
            if (cmpTopkLengthSize < sumOfQuerySeq) {
                KERNEL_LOG_ERROR("The size of cmp_topk_length %d should not be smaller than "
                                 "the sum of query sequence %d!",
                                 cmpTopkLengthSize, sumOfQuerySeq);
                return false;
            }
            // 校验 cmp_topk_length 元素非负
            for (int i = 0; i < cmpTopkLengthSize; i++) {
                if (cmpTopkLengthPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in cmp_topk_length should be >= 0, but got cmp_topk_length[%d] = %d",
                                     i, cmpTopkLengthPtr[i]);
                    return false;
                }
            }
        }
    }
    return true;
}

int32_t SparseFlashMlaGradMetadataCpuKernel::GetSumOfQuerySeq()
{
    int32_t batchSize = GetQueryBatchSize();
    // 如果sequsedQ_ 传了，使用sequsedQ_获取 BsSize
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        if (sequsedQ_->GetTensorShape() != nullptr) {
            const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
            int32_t queryBsSize = 0;
            for (int i = 0; i < batchSize; i++) {
                queryBsSize += seqUsedPtr[i];
            }
            return queryBsSize;
        }
    }
    // sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_获取 BsSize
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            if (cuSeqlensQ_->GetTensorShape() != nullptr) {
                const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
                return s1Ptr[batchSize];
            }
        }
    }
    // 如果不是 TND，或者 cuSeqlensQ_ 为空，使用shape信息计算 BsSize
    return batchSize_ * maxSeqlenQ_;
}

int32_t SparseFlashMlaGradMetadataCpuKernel::GetQueryBatchSize()
{
    // 1. 如果sequsedQ_ 传了，使用sequsedQ_获取BatchSize
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

void SparseFlashMlaGradMetadataCpuKernel::CalcOriMaskMode()
{
    if (oriMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK)) {
        oriPreToken_ = INT64_MAX;
        oriNextToken_ = INT64_MAX;
        oriAttentionMode_ = NO_MASK;
    } else if (oriMaskMode_ == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        oriPreToken_ = INT64_MAX;
        oriNextToken_ = 0;
        oriAttentionMode_ = HAS_MASK;
    } else { // SparseMode = 4
        oriPreToken_ = (oriWinLeft_ > -1) ? oriWinLeft_ : INT64_MAX;
        oriNextToken_ = (oriWinRight_ > -1) ? oriWinRight_ : INT64_MAX;
        oriAttentionMode_ = HAS_MASK;
    }
}

void SparseFlashMlaGradMetadataCpuKernel::CalcCmpMaskMode()
{
    if (cmpMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK)) {
        cmpPreToken_ = INT64_MAX;
        cmpNextToken_ = INT64_MAX;
        cmpAttentionMode_ = NO_MASK;
    } else if (cmpMaskMode_ == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        cmpPreToken_ = INT64_MAX;
        cmpNextToken_ = 0;
        cmpAttentionMode_ = HAS_MASK;
    } else { // SparseMode = 4
        cmpPreToken_ = (oriWinLeft_ > -1) ? oriWinLeft_ : INT64_MAX;
        cmpNextToken_ = (oriWinRight_ > -1) ? oriWinRight_ : INT64_MAX;
        cmpAttentionMode_ = HAS_MASK;
    }
}

ValidSocVersion SparseFlashMlaGradMetadataCpuKernel::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    } else {
        return ValidSocVersion::ASCEND910;
    }
}

bool SparseFlashMlaGradMetadataCpuKernel::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    CalcOriMaskMode();
    CalcCmpMaskMode();
    isS1G_ = (layoutQ_ == "BSND" || layoutQ_ == "BSH" || layoutQ_ == "TND");
    groupSize_ = numHeadsQ_ / numHeadsKv_;
    if (hasOriKv_ && oriTopK_ != 0) {
        isSparseOriKv_ = true;
    }
    if (hasCmpKv_ && cmpTopK_ != 0) {
        isSparseCmpKv_ = true;
    }
    ValidSocVersion validSocVersion = ProcessSocVersion();
    if (validSocVersion == ValidSocVersion::ASCEND910) {
        mBaseSize_ = groupSize_;
        s2BaseSize_ = 512U;
    } else if (validSocVersion == ValidSocVersion::ASCEND950) {
        mBaseSize_ = groupSize_;
        s2BaseSize_ = 128U;
    } else {
        mBaseSize_ = groupSize_;
        s2BaseSize_ = 128U;
    }
    return true;
}

uint32_t SparseFlashMlaGradMetadataCpuKernel::GetS1Idx(uint32_t s1Size, uint32_t s1GIdx)
{
    uint32_t s1GToken = s1GIdx * mBaseSize_;
    uint32_t s1Idx = 0;
    if (isS1G_) {
        s1Idx = s1GToken / static_cast<int64_t>(groupSize_);
    } else {
        s1Idx = s1GToken % static_cast<int64_t>(s1Size);
    }
    return s1Idx;
}

uint32_t SparseFlashMlaGradMetadataCpuKernel::GetBsStride(uint32_t bIdx, uint32_t s1Idx)
{
    uint32_t bsStride = 0;
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
            bsStride = s1Ptr[bIdx] + s1Idx;
            return bsStride;
        }
    }
    bsStride = bIdx * static_cast<uint32_t>(maxSeqlenQ_) + s1Idx;
    return bsStride;
}

uint32_t SparseFlashMlaGradMetadataCpuKernel::GetOriTopkLength(uint32_t bsStride)
{
    // 尝试使用 oriTopkLength_
    if (oriTopK_ != 0 && oriMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
        oriTopkLength_ != nullptr && oriTopkLength_->GetData() != nullptr) {
        const int32_t *oriTopkPtr = static_cast<const int32_t *>(oriTopkLength_->GetData());
        return static_cast<uint32_t>(oriTopkPtr[bsStride]);
    }
    // 如果不是 DEFAULT_MASK，使用 oriTopK_
    return static_cast<uint32_t>(oriTopK_);
}

uint32_t SparseFlashMlaGradMetadataCpuKernel::GetCmpTopkLength(uint32_t bsStride)
{
    // 尝试使用 cmpTopkLength_
    if (cmpTopK_ != 0 && cmpMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK) &&
        cmpTopkLength_ != nullptr && cmpTopkLength_->GetData() != nullptr) {
        const int32_t *cmpTopkPtr = static_cast<const int32_t *>(cmpTopkLength_->GetData());
        return static_cast<uint32_t>(cmpTopkPtr[bsStride]);
    }
    // 如果不是 DEFAULT_MASK，使用 cmpTopK_
    return static_cast<uint32_t>(cmpTopK_);
}

uint32_t SparseFlashMlaGradMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    // 1. 如果 sequsedQ_ 传了，直接使用
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用 maxSeqlenQ_
    return static_cast<uint32_t>(maxSeqlenQ_);
}

uint32_t SparseFlashMlaGradMetadataCpuKernel::GetOriS2SeqSize(uint32_t bIdx)
{
    // 如果 sequsedOriKv_ 传了，直接使用
    if (sequsedOriKv_ != nullptr && sequsedOriKv_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedOriKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // sequsedOriKv_ 没传，判断 Layout
    if (layoutKv_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensOriKv_
        if (cuSeqlensOriKv_ != nullptr && cuSeqlensOriKv_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t *>(cuSeqlensOriKv_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 使用 max_seqlen_ori_kv
    return static_cast<uint32_t>(maxSeqlenOriKv_);
}

uint32_t SparseFlashMlaGradMetadataCpuKernel::GetCmpS2SeqSize(uint32_t bIdx)
{
    // 如果 sequsedCmpKv_ 传了，直接使用
    if (sequsedCmpKv_ != nullptr && sequsedCmpKv_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedCmpKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // sequsedCmpKv_ 没传，判断 Layout
    if (layoutKv_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensCmpKv_
        if (cuSeqlensCmpKv_ != nullptr && cuSeqlensCmpKv_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t *>(cuSeqlensCmpKv_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 使用 max_seqlen_cmp_kv
    return static_cast<uint32_t>(maxSeqlenCmpKv_);
}

uint64_t SparseFlashMlaGradMetadataCpuKernel::GetRevertS2Size(uint32_t bIdx)
{
    uint32_t cmpS2Size = GetCmpS2SeqSize(bIdx);
    if (cmpResidualKv_ != nullptr && cmpResidualKv_->GetData() != nullptr) {
        const int32_t *residualPtr = static_cast<const int32_t *>(cmpResidualKv_->GetData());
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_) + residualPtr[bIdx];
    } else {
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_);
    }
}

void SparseFlashMlaGradMetadataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        totalNum_ += s1Size;
        splitInfo.s1GBaseNum[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_) % mBaseSize_;
        if (hasOriKv_) {
            uint32_t curOriS2Size = GetOriS2SeqSize(bIdx);
            splitInfo.oriS2BaseNum[bIdx] = (static_cast<uint64_t>(curOriS2Size) + s2BaseSize_ - 1U) / s2BaseSize_;
        }
        if (hasCmpKv_) {
            uint32_t curCmpS2Size = GetCmpS2SeqSize(bIdx);
            splitInfo.cmpS2BaseNum[bIdx] = (static_cast<uint64_t>(curCmpS2Size) + s2BaseSize_ - 1U) / s2BaseSize_;
        }
        if (splitInfo.s1GBaseNum[bIdx] != 0U &&
            (splitInfo.oriS2BaseNum[bIdx] != 0U || splitInfo.cmpS2BaseNum[bIdx] != 0U)) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

int64_t SparseFlashMlaGradMetadataCpuKernel::CalcOriPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    if (mode == SparseMode::BAND) {
        return oriPreToken_ == INT64_MAX ? INT64_MAX :
                                           static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + oriPreToken_;
    }
    return oriPreToken_;
}

int64_t SparseFlashMlaGradMetadataCpuKernel::CalcOriNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return oriNextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return oriNextToken_ == INT64_MAX ?
                       INT64_MAX :
                       static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + oriNextToken_;
        default:
            return oriNextToken_;
    }
}

int64_t SparseFlashMlaGradMetadataCpuKernel::CalcCmpPreTokenLeftUp(uint32_t s1Size, uint64_t s2Size)
{
    auto mode = static_cast<SparseMode>(cmpMaskMode_);
    if (mode == SparseMode::BAND) {
        return cmpPreToken_ == INT64_MAX ? INT64_MAX :
                                           static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + cmpPreToken_;
    }
    return cmpPreToken_;
}

int64_t SparseFlashMlaGradMetadataCpuKernel::CalcCmpNextTokenLeftUp(uint32_t s1Size, uint64_t s2Size)
{
    auto mode = static_cast<SparseMode>(cmpMaskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return cmpNextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return cmpNextToken_ == INT64_MAX ?
                       INT64_MAX :
                       static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + cmpNextToken_;
        default:
            return cmpNextToken_;
    }
}

Range<int64_t> SparseFlashMlaGradMetadataCpuKernel::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache,
                                                                     bool isCmpKv)
{
    // actual seq == 0
    if (!isCmpKv) {
        if (batchCache.s1Size == 0U || batchCache.oriS2Size == 0U) {
            return std::make_pair(0, 0);
        }
    } else {
        if (batchCache.s1Size == 0U || batchCache.cmpRevertS2Size == 0U) {
            return std::make_pair(0, 0);
        }
    }

    // no mask
    uint32_t hasMask = 1;
    int64_t s2Size =
        isCmpKv ? static_cast<int64_t>(batchCache.cmpRevertS2Size) : static_cast<int64_t>(batchCache.oriS2Size);
    hasMask = isCmpKv ? cmpAttentionMode_ : oriAttentionMode_;
    if (!hasMask) {
        return std::make_pair(0, s2Size - 1);
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
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = 0;
    int64_t s2LastToken = 0;
    if (!isCmpKv) {
        s2FirstToken = s1FirstToken - batchCache.oriPreTokenLeftUp;
        s2LastToken =
            batchCache.oriNextTokenLeftUp == INT64_MAX ? INT64_MAX : s1LastToken + batchCache.oriNextTokenLeftUp;
    } else {
        s2FirstToken = s1FirstToken - batchCache.cmpPreTokenLeftUp;
        s2LastToken =
            batchCache.cmpNextTokenLeftUp == INT64_MAX ? INT64_MAX : s1LastToken + batchCache.cmpNextTokenLeftUp;
    }
    return std::make_pair(s2FirstToken, s2LastToken);
}

void SparseFlashMlaGradMetadataCpuKernel::CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext,
                                                         BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    if (hasOriKv_) {
        batchCache.oriS2Size = GetOriS2SeqSize(bIdx);
        batchCache.oriPreTokenLeftUp = CalcOriPreTokenLeftUp(batchCache.s1Size, batchCache.oriS2Size);
        batchCache.oriNextTokenLeftUp = CalcOriNextTokenLeftUp(batchCache.s1Size, batchCache.oriS2Size);
    }
    if (hasCmpKv_) {
        batchCache.cmpRevertS2Size = GetRevertS2Size(bIdx);
        batchCache.cmpPreTokenLeftUp = CalcCmpPreTokenLeftUp(batchCache.s1Size, batchCache.cmpRevertS2Size);
        batchCache.cmpNextTokenLeftUp = CalcCmpNextTokenLeftUp(batchCache.s1Size, batchCache.cmpRevertS2Size);
    }
}

void SparseFlashMlaGradMetadataCpuKernel::CalcOriBlockRange(const Range<int64_t> &oriS2TokenRange,
                                                            const BatchCache &batchCache, S1GCache &s1GCache)
{
    int64_t oriS2FirstToken = oriS2TokenRange.first;
    int64_t oriS2LastToken = oriS2TokenRange.second;
    s1GCache.oriS2Start = 0;
    // ori 部分 s2 起止和 tailSize
    if (oriS2FirstToken >= static_cast<int64_t>(batchCache.oriS2Size) || oriS2LastToken < 0 ||
        oriS2LastToken < oriS2FirstToken) {
        s1GCache.oriS2End = 0;
        s1GCache.oriS2TailSize = 0;
    } else {
        oriS2FirstToken =
            Clip(oriS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.oriS2Size - 1U));
        oriS2LastToken = Clip(oriS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.oriS2Size - 1U));
        // oriS2LastToken 与 topk 取最小
        uint32_t s1Idx = GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx);
        uint32_t bsStride = GetBsStride(s1GCache.bIdx, s1Idx);
        uint32_t oriTopkSize = GetOriTopkLength(bsStride);
        uint32_t actOriS2Size = isSparseOriKv_ ?
                                    std::min(static_cast<uint32_t>(oriS2LastToken - oriS2FirstToken + 1), oriTopkSize) :
                                    static_cast<uint32_t>(oriS2LastToken - oriS2FirstToken + 1);
        maxOriKvSize_ = std::max(maxOriKvSize_, actOriS2Size);
        s1GCache.oriS2End = actOriS2Size == 0 ? 0 : (actOriS2Size - 1) / s2BaseSize_ + 1U;
        s1GCache.oriS2TailSize = actOriS2Size % s2BaseSize_;
    }
}

void SparseFlashMlaGradMetadataCpuKernel::CalcCmpBlockRange(const Range<int64_t> &cmpRevertS2TokenRange,
                                                            const BatchCache &batchCache, S1GCache &s1GCache)
{
    int64_t cmpRevertS2FirstToken = cmpRevertS2TokenRange.first;
    int64_t cmpRevertS2LastToken = cmpRevertS2TokenRange.second;
    s1GCache.cmpS2Start = s1GCache.oriS2End;
    // cmp 部分 s2 起止和 tailSize
    if (cmpRevertS2FirstToken >= static_cast<int64_t>(batchCache.cmpRevertS2Size) || cmpRevertS2LastToken < 0 ||
        cmpRevertS2LastToken < cmpRevertS2FirstToken) {
        s1GCache.cmpS2End = s1GCache.cmpS2Start;
        s1GCache.cmpS2TailSize = 0;
    } else {
        cmpRevertS2FirstToken =
            Clip(cmpRevertS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.cmpRevertS2Size - 1U));
        cmpRevertS2LastToken =
            Clip(cmpRevertS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.cmpRevertS2Size - 1U));
        // 如果压缩后长度为0，则直接返回
        if ((cmpRevertS2LastToken + 1) / cmpRatio_ == 0) {
            s1GCache.cmpS2End = s1GCache.cmpS2Start;
            s1GCache.cmpS2TailSize = 0;
            return;
        }
        // 获取压缩后的 token 索引
        uint64_t cmpS2FirstToken =
            (cmpRevertS2FirstToken + 1) / cmpRatio_ == 0 ? 0 : (cmpRevertS2FirstToken + 1) / cmpRatio_ - 1U;
        uint64_t cmpS2LastToken = (cmpRevertS2LastToken + 1) / cmpRatio_ - 1U;
        // cmpS2LastToken 与 topk 取最小
        uint32_t s1Idx = GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx);
        uint32_t bsStride = GetBsStride(s1GCache.bIdx, s1Idx);
        uint32_t cmpTopkSize = GetCmpTopkLength(bsStride);
        uint32_t actCmpS2Size = isSparseCmpKv_ ?
                                    std::min(static_cast<uint32_t>(cmpS2LastToken - cmpS2FirstToken + 1), cmpTopkSize) :
                                    static_cast<uint32_t>(cmpS2LastToken - cmpS2FirstToken + 1);
        maxCmpKvSize_ = std::max(maxCmpKvSize_, actCmpS2Size);
        s1GCache.cmpS2End =
            actCmpS2Size == 0 ? s1GCache.cmpS2Start : s1GCache.cmpS2Start + (actCmpS2Size - 1) / s2BaseSize_ + 1U;
        s1GCache.cmpS2TailSize = actCmpS2Size % s2BaseSize_;
    }
}

void SparseFlashMlaGradMetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext,
                                                       const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 如果s1G是空行，则直接返回
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        return;
    }
    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;
    // 计算 ori_kv 有效负载起止
    if (hasOriKv_) {
        // 计算 ori_kv 的 s2Token 起止
        auto oriS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache, ORI_KV);
        // 计算 ori_kv 的 s2Block 起止
        CalcOriBlockRange(oriS2TokenRange, batchCache, s1GCache);
    } else {
        // ori_kv s2Token 起止初始化为0
        s1GCache.oriS2Start = 0;
        s1GCache.oriS2End = s1GCache.oriS2Start;
        s1GCache.oriS2TailSize = 0;
    }
    // 计算 cmp_kv 有效负载起止
    if (hasCmpKv_) {
        // 计算 cmp_kv 的 s2Token 起止
        auto cmpRevertS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache, CMP_KV);
        // 计算 cmp_kv 的 s2Block 起止
        CalcCmpBlockRange(cmpRevertS2TokenRange, batchCache, s1GCache);
    } else {
        // cmp_kv s2Token 起止初始化为0
        s1GCache.cmpS2Start = s1GCache.oriS2End;
        s1GCache.cmpS2End = s1GCache.cmpS2Start;
        s1GCache.cmpS2TailSize = 0;
    }
}

void SparseFlashMlaGradMetadataCpuKernel::CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    if (GetS1SeqSize(bIdx) == 0U) {
        return;
    }
    if (!hasOriKv_ && !hasCmpKv_) {
        return;
    } else if (!hasOriKv_) {
        if (GetCmpS2SeqSize(bIdx) == 0U) {
            return;
        }
    } else if (!hasCmpKv_) {
        if (GetOriS2SeqSize(bIdx) == 0U) {
            return;
        }
    } else {
        if ((GetOriS2SeqSize(bIdx) == 0U) && GetCmpS2SeqSize(bIdx) == 0U) {
            return;
        }
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
    }
}

void SparseFlashMlaGradMetadataCpuKernel::CalcCostInfo(SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    if (splitInfo.isKvSeqAllZero) {
        return;
    }
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        CalcBatchCost(bIdx, splitContext);
    }
}

bool SparseFlashMlaGradMetadataCpuKernel::BalanceSchedule(SplitResult &splitRes)
{
    SplitContext splitContext(batchSize_);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        return true;
    }
    CalcCostInfo(splitContext);
    return true;
}

bool SparseFlashMlaGradMetadataCpuKernel::GenMetadata(SplitResult &splitRes)
{
    optiling::detail::SmlagMetadata *gradMetadataPtr =
        static_cast<optiling::detail::SmlagMetadata *>(metadata_->GetData());
    // Grad Metadata Generate
    uint32_t formerCoreProcessNum = CeilDiv(totalNum_, aicCoreNum_);
    uint32_t remainCoreProcessNum = formerCoreProcessNum - 1;
    uint32_t remainCoreNum = formerCoreProcessNum * aicCoreNum_ - totalNum_;
    uint32_t usedCoreNum = totalNum_ < aicCoreNum_ ? totalNum_ : aicCoreNum_;

    gradMetadataPtr->gradMetadata[TOTAL_NUM] = totalNum_; // AIC enable
    gradMetadataPtr->gradMetadata[FORMER_CORE_PROCESS_NUM] = formerCoreProcessNum;
    gradMetadataPtr->gradMetadata[REMAIN_CORE_PROCESS_NUM] = remainCoreProcessNum;
    gradMetadataPtr->gradMetadata[REMAIN_CORE_NUM] = remainCoreNum;
    gradMetadataPtr->gradMetadata[USED_CORE_NUM] = usedCoreNum;
    gradMetadataPtr->gradMetadata[MAX_ORI_KV_SIZE] = maxOriKvSize_;
    gradMetadataPtr->gradMetadata[MAX_CMP_KV_SIZE] = maxCmpKvSize_;
    return true;
}
namespace {
static const char *kernelType = "SparseFlashMlaGradMetadata";
REGISTER_CPU_KERNEL(kernelType, SparseFlashMlaGradMetadataCpuKernel);
} // namespace

}; // namespace aicpu
