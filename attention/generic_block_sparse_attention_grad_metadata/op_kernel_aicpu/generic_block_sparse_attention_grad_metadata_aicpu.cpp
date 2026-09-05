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
 * \file generic_block_sparse_attention_grad_metadata_aicpu.cpp
 * \brief AICPU kernel: B→N2→J→G task list with [baseM,baseN] tile cost balancing.
 */

#include "generic_block_sparse_attention_grad_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
uint32_t GenericBlockSparseAttentionGradMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    success = BuildKvBlockGroups() && BalanceKvBlockGroups() && ExpandTaskList() && GenMetadata();
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

// 从 CpuKernelContext 取输入输出与属性，并进行参数检查和初始化
bool GenericBlockSparseAttentionGradMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    rsvdBlockIdx_ = ctx.Input(static_cast<uint32_t>(ParamId::rsvdBlockIdx));
    rsvdBlockCount_ = ctx.Input(static_cast<uint32_t>(ParamId::rsvdBlockCount));
    cuSeqLengthsQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqLengthsQ));
    cuSeqLengthsKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqLengthsKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs = GetAttrValue(ctx, "max_q_seqlen", maxQSeqlen_) &&
                         GetAttrValue(ctx, "max_kv_seqlen", maxKvSeqlen_) &&
                         GetAttrValue(ctx, "num_q_heads", numQHeads_) &&
                         GetAttrValue(ctx, "num_kv_heads", numKvHeads_) && GetAttrValue(ctx, "head_dim", headDim_);
    if (!requiredAttrs) {
        return false;
    }

    GetAttrValueOpt(ctx, "block_shape_x", blockShapeX_);
    GetAttrValueOpt(ctx, "block_shape_y", blockShapeY_);
    GetAttrValueOpt(ctx, "is_packed_gqa", isPackedGQA_);
    GetAttrValueOpt(ctx, "mask_type", maskType_);
    GetAttrValueOpt(ctx, "softmax_precision", softmaxPrecision_);
    GetAttrValueOpt(ctx, "window_size_left", winLeft_);
    GetAttrValueOpt(ctx, "window_size_right", winRight_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);
    GetAttrValueOpt(ctx, "aiv_core_num", aivCoreNum_);
    GetAttrValueOpt(ctx, "q_input_layout", layoutQ_);
    GetAttrValueOpt(ctx, "kv_input_layout", layoutKv_);
    GetAttrValueOpt(ctx, "soc_version", socVersion_);

    return ParamsCheck() && ParamsInit();
}

// 检查参数是否合法
bool GenericBlockSparseAttentionGradMetadataCpuKernel::ParamsCheck()
{
    if (metadata_ == nullptr || metadata_->GetData() == nullptr || metadata_->GetTensorShape() == nullptr) {
        KERNEL_LOG_ERROR("Output metadata is nullptr");
        return false;
    }
    if (metadata_->GetTensorShape()->GetDims() != 1) {
        KERNEL_LOG_ERROR("metadata must be a 1D tensor");
        return false;
    }
    if (rsvdBlockIdx_ == nullptr || rsvdBlockIdx_->GetData() == nullptr || rsvdBlockIdx_->GetTensorShape() == nullptr ||
        rsvdBlockIdx_->GetTensorShape()->GetDims() != 4) {
        KERNEL_LOG_ERROR("rsvd_block_idx must be a valid 4D tensor");
        return false;
    }
    if (rsvdBlockCount_ == nullptr || rsvdBlockCount_->GetData() == nullptr ||
        rsvdBlockCount_->GetTensorShape() == nullptr || rsvdBlockCount_->GetTensorShape()->GetDims() != 3) {
        KERNEL_LOG_ERROR("rsvd_block_count must be a valid 3D tensor");
        return false;
    }
    if (layoutQ_ == "TND") {
        if (cuSeqLengthsQ_ == nullptr || cuSeqLengthsQ_->GetData() == nullptr) {
            KERNEL_LOG_ERROR("cu_seq_lengths is required when layout_q is TND");
            return false;
        }
        if (cuSeqLengthsKv_ == nullptr || cuSeqLengthsKv_->GetData() == nullptr) {
            KERNEL_LOG_ERROR("cu_seq_lengths_kv is required when layout_kv is TND");
            return false;
        }
    }
    return true;
}

// 初始化参数
bool GenericBlockSparseAttentionGradMetadataCpuKernel::ParamsInit()
{
    auto idxShape = rsvdBlockIdx_->GetTensorShape();
    auto cntShape = rsvdBlockCount_->GetTensorShape();
    batchSize_ = static_cast<uint32_t>(idxShape->GetDimSize(0));
    n2Size_ = static_cast<uint32_t>(idxShape->GetDimSize(1));
    jSize_ = static_cast<uint32_t>(idxShape->GetDimSize(2));
    maxS1_ = static_cast<uint32_t>(idxShape->GetDimSize(3));

    if (static_cast<uint32_t>(cntShape->GetDimSize(0)) != batchSize_ ||
        static_cast<uint32_t>(cntShape->GetDimSize(1)) != n2Size_ ||
        static_cast<uint32_t>(cntShape->GetDimSize(2)) != jSize_) {
        KERNEL_LOG_ERROR("rsvd_block_count shape mismatch with rsvd_block_idx");
        return false;
    }
    if (numKvHeads_ <= 0 || numQHeads_ % numKvHeads_ != 0) {
        KERNEL_LOG_ERROR("invalid head config, num_q_heads=%d, num_kv_heads=%d", numQHeads_, numKvHeads_);
        return false;
    }
    groupSize_ = static_cast<uint32_t>(numQHeads_ / numKvHeads_);
    if (isPackedGQA_ != 1) {
        KERNEL_LOG_ERROR("only is_packed_gqa=1 is supported currently");
        return false;
    }

    metadataCapacity_ = static_cast<uint32_t>(metadata_->GetTensorShape()->GetDimSize(0));
    const uint64_t requiredElems = CalcGsagMetadataSize(batchSize_, static_cast<uint64_t>(numQHeads_), jSize_);
    if (static_cast<uint64_t>(metadataCapacity_) < requiredElems) {
        KERNEL_LOG_ERROR("metadata size %u < required %llu (B=%u N1=%d J=%u)", metadataCapacity_,
                         static_cast<unsigned long long>(requiredElems), batchSize_, numQHeads_, jSize_);
        return false;
    }
    if (CalcGsagMetadataMaxTasks(batchSize_, static_cast<uint64_t>(numQHeads_), jSize_) >
        GSAG_METADATA_ABSOLUTE_MAX_TASKS) {
        KERNEL_LOG_ERROR("task upper bound exceeds absolute max %llu",
                         static_cast<unsigned long long>(GSAG_METADATA_ABSOLUTE_MAX_TASKS));
        return false;
    }

    baseM_ = GSAG_DEFAULT_BASE_M;
    baseN_ = static_cast<uint32_t>(blockShapeY_ > 0 ? blockShapeY_ : GSAG_DEFAULT_BASE_N);
    coreGroupStart_.assign(aicCoreNum_, 0U);
    coreGroupEnd_.assign(aicCoreNum_, 0U);
    return true;
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernel::GetKvSeqLen(uint32_t bIdx) const
{
    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        const int32_t *sequsedPtr = static_cast<const int32_t *>(sequsedKv_->GetData());
        return static_cast<uint32_t>(sequsedPtr[bIdx]);
    }
    if (layoutKv_ == "TND" && cuSeqLengthsKv_ != nullptr && cuSeqLengthsKv_->GetData() != nullptr) {
        const int64_t *cuPtr = static_cast<const int64_t *>(cuSeqLengthsKv_->GetData());
        return static_cast<uint32_t>(cuPtr[bIdx + 1U] - cuPtr[bIdx]);
    }
    return static_cast<uint32_t>(maxKvSeqlen_);
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernel::GetKvBlockLen(uint32_t bIdx, uint32_t jIdx) const
{
    const uint32_t actS2 = GetKvSeqLen(bIdx);
    const uint32_t s2Start = jIdx * baseN_;
    if (s2Start >= actS2) {
        return 0U;
    }
    const uint32_t remain = actS2 - s2Start;
    return remain < baseN_ ? remain : baseN_;
}

uint64_t GenericBlockSparseAttentionGradMetadataCpuKernel::CalcGroupBlockCost(int32_t count, uint32_t kvBlockLen) const
{
    if (count <= 0 || kvBlockLen == 0U) {
        return 0U;
    }
    const uint64_t mTiles = CeilDiv(static_cast<uint32_t>(count), baseM_);
    const uint64_t nTiles = CeilDiv(kvBlockLen, baseN_);
    return mTiles * nTiles * static_cast<uint64_t>(groupSize_);
}

// 构建 KV 块组,遍历顺序 B → N2 → J。每个 count>0 的 (b,n2,j) 成一个 KV 块组，并记录总的任务数量（baseN=128）
bool GenericBlockSparseAttentionGradMetadataCpuKernel::BuildKvBlockGroups()
{
    const int32_t *countPtr = static_cast<const int32_t *>(rsvdBlockCount_->GetData());
    kvBlockGroups_.clear();
    totalBlockCost_ = 0U;
    maxTaskCount_ = 0;

    // Traverse B → N2 → J; each (b,n2,j) with count>0 is one schedulable KV block.
    for (uint32_t bIdx = 0U; bIdx < batchSize_; ++bIdx) {
        const uint32_t jLimit = CeilDiv(GetKvSeqLen(bIdx), baseN_);
        for (uint32_t n2Idx = 0U; n2Idx < n2Size_; ++n2Idx) {
            for (uint32_t jIdx = 0U; jIdx < jSize_; ++jIdx) {
                if (jIdx >= jLimit) {
                    continue;
                }
                const uint64_t cntOffset = (static_cast<uint64_t>(bIdx) * n2Size_ + n2Idx) * jSize_ + jIdx;
                const int32_t count = countPtr[cntOffset];
                if (count <= 0) {
                    continue;
                }
                if (count > static_cast<int32_t>(maxS1_)) {
                    KERNEL_LOG_ERROR("rsvd_block_count[%u,%u,%u]=%d exceeds maxS1=%u", bIdx, n2Idx, jIdx, count,
                                     maxS1_);
                    return false;
                }

                GsagKvBlockGroup group;
                group.b = bIdx;
                group.n2 = n2Idx;
                group.j = jIdx;
                group.count = count;
                group.kvBlockLen = GetKvBlockLen(bIdx, jIdx);
                group.blockCost = CalcGroupBlockCost(count, group.kvBlockLen);
                if (group.blockCost == 0U) {
                    continue;
                }
                maxTaskCount_ = std::max(maxTaskCount_, count);
                totalBlockCost_ += group.blockCost;
                kvBlockGroups_.push_back(group);
            }
        }
    }
    return true;
}

// 负载均衡,将 KV 块组分配到 AIC 核心上，按照任务数量进行分配，使得每个核心的块成本尽可能均衡，
// 且保证每个group在同一个核上进行处理，使KV L1 可复用
bool GenericBlockSparseAttentionGradMetadataCpuKernel::BalanceKvBlockGroups()
{
    const uint32_t groupNum = static_cast<uint32_t>(kvBlockGroups_.size());
    usedCoreNum_ = groupNum == 0U ? 1U : std::min(aicCoreNum_, groupNum);
    maxCoreBlockCost_ = 0U;

    for (uint32_t coreIdx = 0U; coreIdx < aicCoreNum_; ++coreIdx) {
        coreGroupStart_[coreIdx] = groupNum;
        coreGroupEnd_[coreIdx] = groupNum;
    }

    if (groupNum == 0U) {
        coreGroupStart_[0] = 0U;
        coreGroupEnd_[0] = 0U;
        return true;
    }

    uint32_t groupIdx = 0U;
    uint64_t unassignedCost = totalBlockCost_;
    for (uint32_t coreIdx = 0U; coreIdx < usedCoreNum_ && groupIdx < groupNum; ++coreIdx) {
        const bool isLastCore = (coreIdx + 1U == usedCoreNum_);
        const uint32_t remainingCores = usedCoreNum_ - coreIdx;
        uint64_t blockLimit = isLastCore ? unassignedCost : (unassignedCost + remainingCores - 1U) / remainingCores;
        blockLimit = std::max(blockLimit, kvBlockGroups_[groupIdx].blockCost);

        coreGroupStart_[coreIdx] = groupIdx;
        uint64_t coreCost = 0U;
        while (groupIdx < groupNum) {
            const uint64_t nextCost = kvBlockGroups_[groupIdx].blockCost;
            if (!isLastCore && coreCost > 0U && coreCost + nextCost > blockLimit &&
                groupIdx > coreGroupStart_[coreIdx]) {
                break;
            }
            coreCost += nextCost;
            ++groupIdx;
            if (!isLastCore && coreCost >= blockLimit) {
                break;
            }
        }
        coreGroupEnd_[coreIdx] = groupIdx;
        maxCoreBlockCost_ = std::max(maxCoreBlockCost_, coreCost);
        unassignedCost = unassignedCost > coreCost ? unassignedCost - coreCost : 0U;
    }

    // 处理剩余的 KV 块组，分配到最后一个核上
    if (groupIdx < groupNum && usedCoreNum_ > 0U) {
        const uint32_t lastCore = usedCoreNum_ - 1U;
        uint64_t tailCost = 0U;
        for (uint32_t idx = coreGroupStart_[lastCore]; idx < groupNum; ++idx) {
            tailCost += kvBlockGroups_[idx].blockCost;
        }
        coreGroupEnd_[lastCore] = groupNum;
        maxCoreBlockCost_ = std::max(maxCoreBlockCost_, tailCost);
    }
    return true;
}

// 展开任务列表，将每个 KV 块组中的任务展开，并记录到任务列表中
bool GenericBlockSparseAttentionGradMetadataCpuKernel::ExpandTaskList()
{
    taskList_.clear();
    for (const GsagKvBlockGroup &group : kvBlockGroups_) {
        for (uint32_t gIdx = 0U; gIdx < groupSize_; ++gIdx) {
            GsagTask task;
            task.b = group.b;
            task.n2 = group.n2;
            task.j = group.j;
            task.g = gIdx;
            task.count = group.count;
            taskList_.push_back(task);
        }
    }

    totalNum_ = static_cast<uint32_t>(taskList_.size());
    const uint32_t maxTasks = static_cast<uint32_t>(CalcGsagMetadataTaskCapacity(metadataCapacity_));
    if (totalNum_ > maxTasks) {
        KERNEL_LOG_ERROR("task num %u exceeds metadata capacity %u (elems=%u)", totalNum_, maxTasks, metadataCapacity_);
        return false;
    }
    return true;
}

// 生成元数据，将任务列表、块成本、核心数量等信息写入到 metadata 中
bool GenericBlockSparseAttentionGradMetadataCpuKernel::GenMetadata()
{
    int64_t *metadataPtr = static_cast<int64_t *>(metadata_->GetData());
    for (uint32_t i = 0U; i < metadataCapacity_; ++i) {
        metadataPtr[i] = 0;
    }

    metadataPtr[TOTAL_NUM] = static_cast<int64_t>(totalNum_);
    metadataPtr[TOTAL_BLOCK_COST] = static_cast<int64_t>(totalBlockCost_);
    metadataPtr[MAX_CORE_BLOCK_COST] = static_cast<int64_t>(maxCoreBlockCost_);
    metadataPtr[BASE_M] = static_cast<int64_t>(baseM_);
    metadataPtr[BASE_N] = static_cast<int64_t>(baseN_);
    metadataPtr[USED_CORE_NUM] = static_cast<int64_t>(usedCoreNum_);
    metadataPtr[GROUP_SIZE] = static_cast<int64_t>(groupSize_);
    metadataPtr[MAX_TASK_COUNT] = static_cast<int64_t>(maxTaskCount_);

    for (uint32_t coreIdx = 0U; coreIdx < aicCoreNum_; ++coreIdx) {
        metadataPtr[CORE_TASK_START_OFFSET + coreIdx] = 0;
        metadataPtr[CORE_TASK_END_OFFSET + coreIdx] = 0;
    }

    for (uint32_t coreIdx = 0U; coreIdx < usedCoreNum_; ++coreIdx) {
        const uint32_t groupStart = coreGroupStart_[coreIdx];
        const uint32_t groupEnd = coreGroupEnd_[coreIdx];
        const uint32_t taskStart = groupStart * groupSize_;
        const uint32_t taskEnd = groupEnd * groupSize_;
        metadataPtr[CORE_TASK_START_OFFSET + coreIdx] = static_cast<int64_t>(taskStart);
        metadataPtr[CORE_TASK_END_OFFSET + coreIdx] = static_cast<int64_t>(taskEnd);
    }

    for (uint32_t i = 0U; i < totalNum_; ++i) {
        const GsagTask &task = taskList_[i];
        const uint32_t base = TASK_LIST_OFFSET + i * TASK_ENTRY_SIZE;
        metadataPtr[base + TASK_B] = static_cast<int64_t>(task.b);
        metadataPtr[base + TASK_N2] = static_cast<int64_t>(task.n2);
        metadataPtr[base + TASK_J] = static_cast<int64_t>(task.j);
        metadataPtr[base + TASK_G] = static_cast<int64_t>(task.g);
    }

    (void)blockShapeX_;
    (void)headDim_;
    (void)maxQSeqlen_;
    (void)maskType_;
    (void)softmaxPrecision_;
    (void)winLeft_;
    (void)winRight_;
    (void)layoutQ_;
    (void)sequsedQ_;
    (void)cuSeqLengthsQ_;
    (void)rsvdBlockIdx_;
    (void)aivCoreNum_;
    (void)socVersion_;
    return true;
}

namespace {
static const char *kernelType = "GenericBlockSparseAttentionGradMetadata";
REGISTER_CPU_KERNEL(kernelType, GenericBlockSparseAttentionGradMetadataCpuKernel);
} // namespace

} // namespace aicpu
