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
 * \file generic_block_sparse_attention_grad_metadata_aicpu_arch22.h
 * \brief
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_AICPU_ARCH22_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_AICPU_ARCH22_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "generic_block_sparse_attention_grad_metadata_layout_arch22.h"

namespace aicpu {

/*
 * GenericBlockSparseAttentionGradMetadata 的 AICPU 实现：
 *   Compute            -> Prepare -> (BalanceSchedule -> GenMetadata)
 *   Prepare           读取 Host 已校验的输入 ABI，准备 per-batch 长度、前缀和 max 长度。
 *   BalanceSchedule   按 basic-block 粒度计算 K_OUT task 总数与分核。
 *   GenMetadata       写入 metadata header 和每个 AIC core 的 KTask。
 *
 * Metadata 输出 ABI 见 generic_block_sparse_attention_grad_metadata_layout.h。
 */
class GenericBlockSparseAttentionGradMetadataCpuKernelArch22 : public CpuKernel {
public:
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool BalanceSchedule();
    bool GenMetadata();

    // Read sequence values from the Host-validated tensors.
    uint32_t QSeqLen(uint32_t batchIdx) const;
    uint32_t KvSeqLen(uint32_t batchIdx) const;

    // Task partitioning uses the shared basic-block contract.
    static uint32_t GetBasicBlocks(uint32_t seqLen, uint32_t sparseBlockSize);
    static uint32_t DecodeBasicBlockStart(uint32_t basicBlockIdx, uint32_t sparseBlockSize);

    uint32_t GetOutputCapacityWords() const;
    bool FillKTask(uint32_t coreIdx, uint32_t startId, int32_t *out) const;

    // ---- input（device tensor，AICPU 侧 GetData 可直接读） ----
    Tensor *sparseBlockIdx_ = nullptr;
    Tensor *cuSeqQ_ = nullptr;
    Tensor *cuSeqKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedKv_ = nullptr;
    // ---- output ----
    Tensor *metadata_ = nullptr;

    // ---- attr ----
    int64_t maxQSeqlenAttr_ = 0;
    int64_t maxKvSeqlenAttr_ = 0;
    int64_t numQHeads_ = 0;
    int64_t numKvHeads_ = 0;
    int64_t headDim_ = 0;
    std::string layoutQ_ = "TND";
    uint32_t aicCoreNum_ = 1;

    // ---- 派生 ----
    uint32_t batch_ = 0;
    uint32_t blockShapeY_ = 128;
    uint32_t inputLayout_ = 0; // 0=TND 1=BNSD 2=BSND
    std::vector<uint32_t> qSeqlen_;
    std::vector<uint32_t> kvSeqlen_;

    std::vector<uint32_t> tasksInBatch_;
    uint32_t totalTaskNum_ = 0;
    uint32_t taskNumPerCore_ = 0;
    uint32_t tailTaskNum_ = 0;

    enum class ParamId : uint32_t {
        sparseBlockIdx = 0,
        sparseBlockCount = 1,
        cuSeqQ = 2,
        cuSeqKv = 3,
        sequsedQ = 4,
        sequsedKv = 5,
    };
};

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernelArch22::Compute(CpuKernelContext &ctx)
{
    if (!Prepare(ctx)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (!BalanceSchedule()) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (!GenMetadata()) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernelArch22::GetOutputCapacityWords() const
{
    if (metadata_ == nullptr || metadata_->GetDataSize() == 0) {
        return 0;
    }
    return static_cast<uint32_t>(metadata_->GetDataSize() / sizeof(int32_t));
}

bool GenericBlockSparseAttentionGradMetadataCpuKernelArch22::Prepare(CpuKernelContext &ctx)
{
    using PI = ParamId;
    sparseBlockIdx_ = ctx.Input(static_cast<uint32_t>(PI::sparseBlockIdx));
    cuSeqQ_ = ctx.Input(static_cast<uint32_t>(PI::cuSeqQ));
    cuSeqKv_ = ctx.Input(static_cast<uint32_t>(PI::cuSeqKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(PI::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(PI::sequsedKv));
    metadata_ = ctx.Output(0);

    // ---- required attrs ----
    bool ok = GetAttrValue(ctx, "max_q_seqlen", maxQSeqlenAttr_) &&
              GetAttrValue(ctx, "max_kv_seqlen", maxKvSeqlenAttr_) && GetAttrValue(ctx, "num_q_heads", numQHeads_) &&
              GetAttrValue(ctx, "num_kv_heads", numKvHeads_) && GetAttrValue(ctx, "head_dim", headDim_);
    if (!ok) {
        return false;
    }
    // ---- optional attrs ----
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);

    if (metadata_ == nullptr || metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("metadata output is null");
        return false;
    }
    if (sparseBlockIdx_ == nullptr || sparseBlockIdx_->GetData() == nullptr ||
        sparseBlockIdx_->GetTensorShape() == nullptr) {
        KERNEL_LOG_ERROR("sparseBlockIdx is null");
        return false;
    }

    // Host guarantees a supported layout; only decode the ABI value here.
    inputLayout_ = layoutQ_ == "TND"  ? gbsag_meta::LAYOUT_TND :
                   layoutQ_ == "BNSD" ? gbsag_meta::LAYOUT_BNSD :
                                        gbsag_meta::LAYOUT_BSND;

    // 契约：TND 场景 cuSeq 必传（int64 前缀和数组，长度 b+1）；
    // seqused 仅供 packed 布局（BNSD/BSND）作为可选长度来源。
    if (inputLayout_ == gbsag_meta::LAYOUT_TND && ((cuSeqQ_ == nullptr || cuSeqQ_->GetData() == nullptr) ||
                                                   (cuSeqKv_ == nullptr || cuSeqKv_->GetData() == nullptr))) {
        KERNEL_LOG_ERROR("TND layout requires both cuSeqQ and cuSeqKv inputs");
        return false;
    }

    // batch_ is used to size the derived arrays below, so the first dimension must exist.
    const auto sparseBlockIdxShape = sparseBlockIdx_->GetTensorShape();
    if (sparseBlockIdxShape == nullptr || sparseBlockIdxShape->GetDims() < 1) {
        KERNEL_LOG_ERROR("sparseBlockIdx shape has no batch dimension");
        return false;
    }
    batch_ = static_cast<uint32_t>(sparseBlockIdxShape->GetDimSize(0));

    // 契约对齐：aclnn 以标量 attr 传递 blockShape（block_shape_x / block_shape_y），
    // 不存在第 7 个输入张量；此处按 attr 读取，aclnn 接口保持不变。
    int64_t blockShapeYAttr = 0;
    if (!GetAttrValue(ctx, "block_shape_y", blockShapeYAttr) || blockShapeYAttr <= 0) {
        KERNEL_LOG_ERROR("get block_shape_y attr failed or invalid");
        return false;
    }
    blockShapeY_ = static_cast<uint32_t>(blockShapeYAttr);

    // Build per-batch lengths. 长度来源语义见 QSeqLen/KvSeqLen：
    //   TND 取 cuSeq 前缀和差分；BNSD/BSND 取 seqused（可选，缺省回退 attr 物理值，
    //   保持物理 stride 语义，不能取实际最大值否则 stride 错位）。
    qSeqlen_.assign(batch_, 0);
    kvSeqlen_.assign(batch_, 0);
    for (uint32_t b = 0; b < batch_; ++b) {
        qSeqlen_[b] = QSeqLen(b);
        kvSeqlen_[b] = KvSeqLen(b);
    }

    return true;
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernelArch22::QSeqLen(uint32_t b) const
{
    // TND 契约：cuSeq 必传（Prepare 已校验），长度取前缀和差分；
    // packed 布局（BNSD/BSND）seqused 可选，缺省回退物理 max。
    if (inputLayout_ == gbsag_meta::LAYOUT_TND) {
        const int64_t *cu = static_cast<const int64_t *>(cuSeqQ_->GetData());
        return static_cast<uint32_t>(cu[b + 1] - cu[b]);
    }
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *used = static_cast<const int32_t *>(sequsedQ_->GetData());
        return static_cast<uint32_t>(used[b]);
    }
    return static_cast<uint32_t>(maxQSeqlenAttr_);
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernelArch22::KvSeqLen(uint32_t b) const
{
    if (inputLayout_ == gbsag_meta::LAYOUT_TND) {
        const int64_t *cu = static_cast<const int64_t *>(cuSeqKv_->GetData());
        return static_cast<uint32_t>(cu[b + 1] - cu[b]);
    }
    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        const int32_t *used = static_cast<const int32_t *>(sequsedKv_->GetData());
        return static_cast<uint32_t>(used[b]);
    }
    return static_cast<uint32_t>(maxKvSeqlenAttr_);
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernelArch22::GetBasicBlocks(uint32_t seqLen,
                                                                                uint32_t sparseBlockSize)
{
    if (sparseBlockSize == 0) {
        return 0;
    }
    const uint32_t blocksPerSparse =
        (sparseBlockSize + gbsag_meta::BASIC_BLOCK_SIZE - 1) / gbsag_meta::BASIC_BLOCK_SIZE;
    const uint32_t fullSparseBlocks = seqLen / sparseBlockSize;
    const uint32_t tail = seqLen % sparseBlockSize;
    return fullSparseBlocks * blocksPerSparse +
           (tail + gbsag_meta::BASIC_BLOCK_SIZE - 1) / gbsag_meta::BASIC_BLOCK_SIZE;
}

uint32_t GenericBlockSparseAttentionGradMetadataCpuKernelArch22::DecodeBasicBlockStart(uint32_t basicBlockIdx,
                                                                                       uint32_t sparseBlockSize)
{
    const uint32_t blocksPerSparse =
        (sparseBlockSize + gbsag_meta::BASIC_BLOCK_SIZE - 1) / gbsag_meta::BASIC_BLOCK_SIZE;
    if (blocksPerSparse == 0) {
        return 0;
    }
    return (basicBlockIdx / blocksPerSparse) * sparseBlockSize +
           (basicBlockIdx % blocksPerSparse) * gbsag_meta::BASIC_BLOCK_SIZE;
}

bool GenericBlockSparseAttentionGradMetadataCpuKernelArch22::BalanceSchedule()
{
    const uint32_t kvHeads = static_cast<uint32_t>(numKvHeads_);
    if (kvHeads == 0) {
        KERNEL_LOG_ERROR("numKvHeads must be positive");
        return false;
    }

    totalTaskNum_ = 0;
    tasksInBatch_.assign(batch_, 0);
    for (uint32_t b = 0; b < batch_; ++b) {
        const uint32_t kBlocks = GetBasicBlocks(kvSeqlen_[b], blockShapeY_);
        tasksInBatch_[b] = kBlocks * kvHeads;
        totalTaskNum_ += tasksInBatch_[b];
    }
    if (totalTaskNum_ == 0) {
        KERNEL_LOG_ERROR("totalTaskNum is 0");
        return false;
    }

    const uint32_t coreNum = std::min<uint32_t>(
        std::min<uint32_t>(std::max<uint32_t>(aicCoreNum_, 1), gbsag_meta::MAX_AIC_CORE_NUM), totalTaskNum_);
    taskNumPerCore_ = totalTaskNum_ / coreNum;
    tailTaskNum_ = totalTaskNum_ % coreNum;

    // 输出容量防护：保证实际 totalLen ≤ 分配上界。
    const uint32_t actualLen = gbsag_meta::MetadataWords(coreNum);
    if (GetOutputCapacityWords() < actualLen) {
        KERNEL_LOG_ERROR("metadata capacity %u words < required %u words", GetOutputCapacityWords(), actualLen);
        return false;
    }
    return true;
}

bool GenericBlockSparseAttentionGradMetadataCpuKernelArch22::FillKTask(uint32_t coreIdx, uint32_t startId,
                                                                       int32_t *out) const
{
    const uint32_t count = taskNumPerCore_ + (coreIdx < tailTaskNum_ ? 1U : 0U);
    const uint32_t taskId = startId;
    if (count == 0) {
        return true;
    }
    uint32_t remaining = taskId;
    uint32_t b = 0;
    while (b + 1 < batch_ && remaining >= tasksInBatch_[b]) {
        remaining -= tasksInBatch_[b++];
    }
    uint32_t h = 0;
    uint32_t kBlk = 0;
    if (b < batch_) {
        if (inputLayout_ == gbsag_meta::LAYOUT_TND) {
            kBlk = remaining / static_cast<uint32_t>(numKvHeads_);
            h = remaining % static_cast<uint32_t>(numKvHeads_);
        } else {
            const uint32_t kBlocks = GetBasicBlocks(kvSeqlen_[b], blockShapeY_);
            if (kBlocks != 0) {
                h = remaining / kBlocks;
                kBlk = remaining % kBlocks;
            }
        }
    }
    auto *task = reinterpret_cast<gbsag_meta::KTask *>(out + gbsag_meta::HEADER_WORDS) + coreIdx;
    task->beginBatch = b;
    task->beginKvHead = h;
    task->beginKvSeqOffset = DecodeBasicBlockStart(kBlk, blockShapeY_);
    return true;
}

bool GenericBlockSparseAttentionGradMetadataCpuKernelArch22::GenMetadata()
{
    int32_t *out = reinterpret_cast<int32_t *>(metadata_->GetData());
    if (GetOutputCapacityWords() < gbsag_meta::MetadataWords(std::max<uint32_t>(aicCoreNum_, 1))) {
        return false;
    }
    std::memset(out, 0, metadata_->GetDataSize());
    const uint32_t coreNum = std::min<uint32_t>(
        std::min<uint32_t>(std::max<uint32_t>(aicCoreNum_, 1), gbsag_meta::MAX_AIC_CORE_NUM), totalTaskNum_);
    out[gbsag_meta::OFF_TOTAL_TASK_NUM] = totalTaskNum_;
    out[gbsag_meta::OFF_TASK_NUM_PER_CORE] = taskNumPerCore_;
    out[gbsag_meta::OFF_TAIL_TASK_NUM] = tailTaskNum_;
    out[gbsag_meta::OFF_TASK_TABLE_OFFSET] = gbsag_meta::HEADER_WORDS;
    out[gbsag_meta::OFF_CORE_NUM] = coreNum;
    out[gbsag_meta::OFF_TASK_WORDS] = gbsag_meta::KTASK_WORDS;
    uint32_t startId = 0;
    for (uint32_t core = 0; core < coreNum; ++core) {
        if (!FillKTask(core, startId, out)) {
            return false;
        }
        startId += taskNumPerCore_ + (core < tailTaskNum_ ? 1U : 0U);
    }
    return startId == totalTaskNum_;
}

} // namespace aicpu

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_AICPU_ARCH22_H
