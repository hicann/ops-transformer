/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Included inside the private section of MsaIndexScoreKernel.

__aicore__ inline void CopyNativeRows(Catlass::Arch::Resource<ArchTag>& resource,
                                      const AscendC::GlobalTensor<int8_t>& src,
                                      const AscendC::GlobalTensor<int8_t>& dst, uint32_t validRows, uint32_t totalRows,
                                      uint32_t srcStride)
{
    const uint32_t kp = tiling_->paddedHeadDim;
    auto ub = resource.ubBuf.template GetBufferByByte<int8_t>(Epilogue::UB_OFF_S);
    if (validRows != totalRows || kp != tiling_->headDim) {
        AscendC::Duplicate(ub.template ReinterpretCast<int16_t>(), static_cast<int16_t>(0), totalRows * kp / 2U);
    }
    // The padding scratch aliases the epilogue S buffer. This wait
    // orders reuse after prior Vector reads, including the full-copy case.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    if (validRows > 0U) {
        AscendC::DataCopyExtParams copy;
        copy.blockCount = validRows;
        copy.blockLen = tiling_->headDim;
        copy.srcStride = srcStride - tiling_->headDim;
        copy.dstStride = 0;
        AscendC::DataCopyPadExtParams<int8_t> pad;
        pad.isPad = true;
        pad.leftPadding = 0;
        pad.rightPadding = kp - tiling_->headDim;
        pad.paddingValue = 0;
        AscendC::DataCopyPad(ub, src, copy, pad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID2);
    AscendC::DataCopy(dst, ub, totalRows * kp);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
}

__aicore__ inline void PadNativeQuery(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task, uint32_t coreIdx,
                                      uint32_t subIdx, uint32_t subBlockNum)
{
    const uint32_t subM = MsaCeilDiv(task.mActual, subBlockNum);
    const uint32_t row = subIdx * subM;
    const uint32_t rows = row < task.mActual ? MsaMinU32(subM, task.mActual - row) : 0U;
    if (rows == 0U) {
        return;
    }
    const uint64_t qRow = static_cast<uint64_t>(task.cuQStart) * tiling_->numQHeads + task.mStart + row;
    const uint64_t dst = static_cast<uint64_t>(coreIdx) * tiling_->qScratchBytesPerCore +
                         static_cast<uint64_t>(row) * tiling_->paddedHeadDim;
    CopyNativeRows(resource, gQuery_[qRow * tiling_->headDim], gQueryScratch_[dst], rows, rows, tiling_->headDim);
}

__aicore__ inline void GatherNativeKey(Catlass::Arch::Resource<ArchTag>& resource, const MsaTask& task,
                                       uint32_t blkBase, uint64_t kScratchBase, uint32_t subIdx, uint32_t subBlockNum)
{
    for (uint32_t j = subIdx; j < MSA_BLOCKS_PER_STILE; j += subBlockNum) {
        const uint32_t blk = blkBase + j;
        if (blk >= task.visibleEndBlk) {
            break;
        }
        if (!KeyBlockNeedsScratch(task, blk)) {
            continue;
        }
        const uint64_t dst = kScratchBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE * MSA_K_TILE;
        CopyNativeRows(resource, gKeyInt8_[KeyBlockGmOffset(task, blk)], gKeyScratch_[dst],
                       KeyBlockValidTokens(task, blk), MSA_BLOCK_SIZE, tiling_->strideKvToken);
    }
}

__aicore__ inline void ComputeNativePairedSTile(BlockMmad& blockMmad, const MsaTask& task, uint32_t blkBase,
                                                uint64_t sBase, uint64_t kScratchBase, bool needLoadL1,
                                                const AscendC::GlobalTensor<ElementQ>& queryTile,
                                                const LayoutQ& layoutQ, const LayoutK& layoutKScratch,
                                                const LayoutK& layoutKGm, const LayoutS& layoutS,
                                                const Catlass::GemmCoord& shape)
{
    const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;
    uint64_t pairPageOff[MSA_BLOCKS_PER_STILE];
    uint8_t pairPageScratch[MSA_BLOCKS_PER_STILE];
    uint32_t pairPageCount = 0;
    for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
        const uint32_t blk = blkBase + j;
        if (blk >= task.visibleEndBlk) {
            break;
        }
        pairPageScratch[j] = KeyBlockNeedsScratch(task, blk) ? 1U : 0U;
        pairPageOff[j] = pairPageScratch[j] != 0U ? kScratchBase + static_cast<uint64_t>(j) * pageStride :
                                                    KeyBlockGmOffset(task, blk);
        ++pairPageCount;
    }
    if (pairPageCount == 0U) {
        return;
    }
    if (needLoadL1) {
        blockMmad.LoadQ(queryTile, layoutQ, shape);
    }
    const uint32_t pairDepth = pairPageCount < BlockMmad::STAGES ? pairPageCount : BlockMmad::STAGES;
    uint32_t pairIssued = 0;
    for (; pairIssued < pairDepth; ++pairIssued) {
        IssuePageK(blockMmad, pairPageScratch[pairIssued], pairPageOff[pairIssued], shape, layoutKScratch, layoutKGm);
    }
    uint32_t pairConsumed = 0;
    while (pairConsumed < pairPageCount) {
        uint32_t batch = 1U;
        if (pairConsumed + 1U < pairPageCount && blockMmad.CanConsumePair()) {
            batch = 2U;
            blockMmad.ConsumePair(gWorkspace_[sBase + static_cast<uint64_t>(pairConsumed) * MSA_BLOCK_SIZE], layoutS,
                                  shape);
        } else {
            blockMmad.ConsumeK(gWorkspace_[sBase + static_cast<uint64_t>(pairConsumed) * MSA_BLOCK_SIZE], layoutS,
                               shape);
        }
        pairConsumed += batch;
        for (uint32_t refill = 0; refill < batch && pairIssued < pairPageCount; ++refill, ++pairIssued) {
            IssuePageK(blockMmad, pairPageScratch[pairIssued], pairPageOff[pairIssued], shape, layoutKScratch,
                       layoutKGm);
        }
    }
    return;
}
