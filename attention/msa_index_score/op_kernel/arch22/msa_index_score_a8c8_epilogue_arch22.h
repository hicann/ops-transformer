/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Included inside the private section of MsaSegRowMaxEpilogue.

__aicore__ inline void LoadQueryScale(const MsaTask& task)
{
    scaleBeforeReduce_ = false;
    if (mSub_ == 0U) {
        return;
    }
    const uint64_t row = static_cast<uint64_t>(task.cuQStart) * numQHeads_ + task.mStart + mOff_;
    AscendC::DataCopyExtParams copy{1U, mSub_ * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<float> pad{};
    AscendC::DataCopyPad(fixedQuery_, gQueryScale_[row], copy, pad);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
    // A task holds at most 128 scales. Each repeat reduces up to 64
    // actual values, so DMA padding does not affect the minimum.
    constexpr int32_t repStride = MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK;
    constexpr auto order = AscendC::ReduceOrder::ORDER_ONLY_VALUE;
    AscendC::ReduceRepeat<AscendC::ReduceType::MIN, float, float>(
        ubRowMax_, fixedQuery_, MsaMinU32(mSub_, MSA_FP32_PER_REPEAT), 1, 1, 1, repStride, order);
    if (mSub_ > MSA_FP32_PER_REPEAT) {
        AscendC::ReduceRepeat<AscendC::ReduceType::MIN, float, float>(
            ubRowMax_[MSA_FP32_PER_BLOCK], fixedQuery_[MSA_FP32_PER_REPEAT], mSub_ - MSA_FP32_PER_REPEAT, 1, 1, 1,
            repStride, order);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    // Each query scale owns eight consecutive page maxima. A single
    // AIV handling more than 64 rows broadcasts one pass at a time.
    if (!MIX_ONE_WIDE_POLICY || mSub_ <= MSA_STAGE_ROWS) {
        AscendC::Brcb(fixedQueryBroadcast_, fixedQuery_, MsaCeilDiv(mSub_, 8U), {1, 8});
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    scaleBeforeReduce_ = ubRowMax_.GetValue(0) <= 0.0F;
    if (mSub_ > MSA_FP32_PER_REPEAT) {
        scaleBeforeReduce_ |= ubRowMax_.GetValue(MSA_FP32_PER_BLOCK) <= 0.0F;
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RestoreFixedScale(uint32_t rowOff, uint32_t rows)
{
    if (!scaleBeforeReduce_) {
        AscendC::Muls(ubRowMax_, ubRowMax_, 1024.0F, MSA_REDUCE_ROWS);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t scaleOffset = (rowOff - mOff_) * MSA_BLOCKS_PER_STILE;
        if constexpr (MIX_ONE_WIDE_POLICY) {
            if (mSub_ > MSA_STAGE_ROWS) {
                // Pass starts are multiples of sixteen rows. Brcb reads
                // at most sixteen source scales into this reusable slice.
                const uint32_t sourceRow = rowOff - mOff_;
                AscendC::Brcb(fixedQueryBroadcast_, fixedQuery_[sourceRow], MsaCeilDiv(rows, 8U), {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                scaleOffset = 0U;
            }
        }
        AscendC::Mul(ubRowMax_, ubRowMax_, fixedQueryBroadcast_[scaleOffset], rows * MSA_BLOCKS_PER_STILE);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Maxs(ubRowMax_, ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline int32_t ScaleEvent(uint32_t stage) const
{
    return static_cast<int32_t>(4U + stage);
}

__aicore__ inline void IssueNativeScales(const MsaTask& task, uint32_t blkBase, uint32_t stage)
{
    const int32_t event = ScaleEvent(stage);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
    auto dst = ubScalePing_[stage];
    const uint32_t pages = MsaMinU32(MSA_BLOCKS_PER_STILE, task.visibleEndBlk - blkBase);
    // Only a packed TND tail lacks physical scale elements. Clear that
    // cache before DMA so the vector broadcasts never read uninitialized UB.
    if (keyLayout_ == MSA_KEY_LAYOUT_TND &&
        static_cast<uint64_t>(blkBase + pages) * MSA_BLOCK_SIZE > static_cast<uint32_t>(task.kvLen)) {
        AscendC::Duplicate(dst, 0.0F, SCALE_STILE_ELEMS);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
    }
    for (uint32_t j = 0; j < pages; ++j) {
        const uint32_t blk = blkBase + j;
        uint64_t scaleOff;
        uint32_t count = MSA_BLOCK_SIZE;
        if (keyLayout_ == MSA_KEY_LAYOUT_TND) {
            scaleOff = static_cast<uint64_t>(task.cuKStart) + blk * MSA_BLOCK_SIZE;
            count = MsaMinU32(MSA_BLOCK_SIZE, static_cast<uint32_t>(task.kvLen) - blk * MSA_BLOCK_SIZE);
        } else {
            const uint32_t rowStride = tableWidth_ > 0U ? tableWidth_ : maxBlocksPerBatch_;
            const int32_t page = gBlockTable_.GetValue(task.batchIdx * rowStride + blk);
            scaleOff = static_cast<uint64_t>(page) * strideScalePage_;
        }
        AscendC::DataCopyExtParams copy;
        copy.blockCount = 1;
        copy.blockLen = count * sizeof(float);
        copy.srcStride = 0;
        copy.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> pad{};
        pad.isPad = false;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = 0;
        AscendC::DataCopyPad(dst[j * MSA_BLOCK_SIZE], gScale_[scaleOff], copy, pad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    nativeScaleBlk_[stage] = blkBase;
    nativeScalePending_[stage] = true;
}

__aicore__ inline void PrepareNativeScales(const MsaTask& task, uint32_t blkBase)
{
    nativeScaleStage_ = (blkBase / MSA_BLOCKS_PER_STILE) % SCALE_STAGES;
    if (nativeScaleBlk_[nativeScaleStage_] != blkBase) {
        IssueNativeScales(task, blkBase, nativeScaleStage_);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ScaleEvent(nativeScaleStage_));
    nativeScalePending_[nativeScaleStage_] = false;
}

__aicore__ inline void PrefetchNativeScales(const MsaTask& task, uint32_t blkBase)
{
    // Bound the lookahead by this task's actual KV chunk, including a
    // nonzero global stile start. No cache survives the task boundary.
    const uint32_t next = blkBase + MSA_BLOCKS_PER_STILE;
    if (next < task.visibleEndBlk && next / MSA_BLOCKS_PER_STILE < task.sStileEnd) {
        const uint32_t stage = nativeScaleStage_ ^ 1U;
        if (nativeScaleBlk_[stage] != next) {
            IssueNativeScales(task, next, stage);
        }
    }
}

__aicore__ inline void ReleaseNativeScales()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ScaleEvent(nativeScaleStage_));
    nativeScaleBlk_[nativeScaleStage_] = INVALID_SCALE_BLOCK;
}

__aicore__ inline void IssueCompactRows(const AscendC::GlobalTensor<ElementS>& gS, uint32_t rowOff, uint32_t rows,
                                        uint32_t stage, uint32_t columns)
{
    constexpr uint32_t halfPerBlock = MSA_DATABLOCK_BYTES / sizeof(half);
    const AscendC::DataCopyParams copy(static_cast<uint16_t>(rows), static_cast<uint16_t>(columns / halfPerBlock),
                                       static_cast<uint16_t>((STILE_WIDTH - columns) / halfPerBlock), 0U);
    const int32_t evt = static_cast<int32_t>(stage);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evt);
    AscendC::DataCopy(ubS16Ping_[stage], gS[static_cast<uint64_t>(rowOff) * STILE_WIDTH], copy);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evt);
}

__aicore__ inline void ReduceShortCausal(uint32_t pages, uint32_t rows)
{
    constexpr uint32_t halfPage = MSA_BLOCK_SIZE / 2U;
    constexpr uint8_t srcStride = MSA_BLOCK_SIZE / MSA_FP32_PER_BLOCK;
    constexpr uint8_t pairStride = halfPage / MSA_FP32_PER_BLOCK;
    const uint8_t groups = static_cast<uint8_t>(rows * pages);
    AscendC::Max<float>(ubT64_, ubS_, ubS_[halfPage], MSA_FP32_PER_REPEAT, groups,
                        AscendC::BinaryRepeatParams(1, 1, 1, pairStride, srcStride, srcStride));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, true>(ubT8_, ubT64_, groups, MSA_FP32_PER_REPEAT, 1, 1, pairStride);
    AscendC::PipeBarrier<PIPE_V>();
    // T8 is dense by page. Each repeat reads a compact row of p blocks
    // and writes the first p values of an eight-score row.
    AscendC::BlockReduceMax<float, true>(ubRowMax_, ubT8_, static_cast<uint8_t>(rows),
                                         static_cast<int32_t>(pages * MSA_FP32_PER_BLOCK), 1, 1,
                                         static_cast<uint8_t>(pages));
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ConsumeCompactRows(const AscendC::GlobalTensor<ElementS>& gS, const MsaTask& task,
                                          uint32_t blkBase, uint32_t rowOff, uint32_t rows, uint32_t stage,
                                          uint32_t pages, uint32_t columns)
{
    WaitS16(stage);
    AscendC::Cast(ubS_, ubS16_, AscendC::RoundMode::CAST_NONE, rows * columns);
    AscendC::PipeBarrier<PIPE_V>();
    ReleaseS16(stage);
    const auto scale = ubScalePing_[nativeScaleStage_];
    const uint8_t stride = static_cast<uint8_t>(columns / MSA_FP32_PER_BLOCK);
    const AscendC::BinaryRepeatParams broadcast(1, 1, 1, stride, stride, 0);
    AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(MSA_FP32_PER_REPEAT);
    for (uint32_t c = 0; c < columns; c += MSA_FP32_PER_REPEAT) {
        AscendC::Mul<float, false>(ubS_[c], ubS_[c], scale[c], MSA_FP32_PER_REPEAT, static_cast<uint8_t>(rows),
                                   broadcast);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ResetMask();
    ApplyQueryScaleBeforeReduce(rowOff, rows, columns, columns);
    ApplyFixedMaskByToken(task, blkBase, rowOff, rows, columns);
    ReduceShortCausal(pages, rows);
    ProcessPass<true>(gS, task, blkBase, rowOff, rows, true);
}

__aicore__ inline void ApplyQueryScaleBeforeReduce(uint32_t rowOff, uint32_t rows, uint32_t columns, uint32_t rowStride)
{
    if (!scaleBeforeReduce_) {
        return;
    }
    for (uint32_t r = 0; r < rows; ++r) {
        const float queryScale = fixedQuery_.GetValue(rowOff - mOff_ + r);
        const uint32_t base = r * rowStride;
        if (queryScale == 0.0F) {
            AscendC::Duplicate(ubS_[base], 0.0F, columns);
        } else {
            AscendC::Muls(ubS_[base], ubS_[base], 1024.0F, columns);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubS_[base], ubS_[base], queryScale, columns);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ __attribute__((noinline, aligned(64))) void ProcessUnifiedRows(const AscendC::GlobalTensor<ElementS>& gS,
                                                                          const MsaTask& task, uint32_t blkBase,
                                                                          uint32_t rowOff, uint32_t rows)
{
    const uint32_t pages = MsaMinU32(MSA_BLOCKS_PER_STILE, task.visibleEndBlk - blkBase);
    const uint32_t columns = pages * MSA_BLOCK_SIZE;
    uint32_t stage = 0U;
    uint32_t done = 0U;
    uint32_t current = MsaMinU32(MSA_ROWS_PER_PASS, rows);
    IssueCompactRows(gS, rowOff, current, stage, columns);
    while (done < rows) {
        const uint32_t nextOffset = done + current;
        const uint32_t nextRows = MsaMinU32(MSA_ROWS_PER_PASS, rows - nextOffset);
        if (nextRows > 0U) {
            IssueCompactRows(gS, rowOff + nextOffset, nextRows, stage ^ 1U, columns);
        }
        ConsumeCompactRows(gS, task, blkBase, rowOff + done, current, stage, pages, columns);
        stage ^= 1U;
        done = nextOffset;
        current = nextRows;
    }
}

__aicore__ inline void ConsumeFixedPass(const AscendC::GlobalTensor<ElementS>& gS, const MsaTask& task,
                                        uint32_t blkBase, uint32_t rowOff, uint32_t rows, uint32_t stage)
{
    WaitS16(stage);
    const uint32_t pages = MsaMinU32(MSA_BLOCKS_PER_STILE, task.visibleEndBlk - blkBase);
    const uint32_t columns = pages * MSA_BLOCK_SIZE;
    if (rows < MSA_ROWS_PER_PASS || columns < STILE_WIDTH) {
        AscendC::Duplicate(ubS_, MSA_FILL_VALUE, MSA_ROWS_PER_PASS * STILE_WIDTH);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const auto scale = ubScalePing_[nativeScaleStage_];
    if (columns == STILE_WIDTH) {
        // Both buffers have contiguous rows when every page is present.
        // The active S16 slot is selected by WaitS16(stage).
        AscendC::Cast(ubS_, ubS16_, AscendC::RoundMode::CAST_NONE, rows * STILE_WIDTH);
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::Cast(ubS_[r * STILE_WIDTH], ubS16_[r * STILE_WIDTH], AscendC::RoundMode::CAST_NONE, columns);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
    ReleaseS16(stage);
    AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(MSA_FP32_PER_REPEAT);
    const AscendC::BinaryRepeatParams broadcast(1, 1, 1, STILE_WIDTH / MSA_FP32_PER_BLOCK,
                                                STILE_WIDTH / MSA_FP32_PER_BLOCK, 0);
    for (uint32_t c = 0; c < columns; c += MSA_FP32_PER_REPEAT) {
        AscendC::Mul<float, false>(ubS_[c], ubS_[c], scale[c], MSA_FP32_PER_REPEAT, static_cast<uint8_t>(rows),
                                   broadcast);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ResetMask();
    ApplyQueryScaleBeforeReduce(rowOff, rows, columns, STILE_WIDTH);
    ProcessPass(gS, task, blkBase, rowOff, rows, true);
}

__aicore__ inline void ProcessFixedRows(const AscendC::GlobalTensor<ElementS>& gS, const MsaTask& task,
                                        uint32_t blkBase, uint32_t rowOff, uint32_t rows)
{
    uint32_t stage = 0;
    uint32_t previousOff = rowOff;
    uint32_t previousRows = MsaMinU32(MSA_ROWS_PER_PASS, rows);
    IssueS16(gS, rowOff, previousRows, stage);
    for (uint32_t p = previousRows; p < rows; p += MSA_ROWS_PER_PASS) {
        const uint32_t count = MsaMinU32(MSA_ROWS_PER_PASS, rows - p);
        IssueS16(gS, rowOff + p, count, stage ^ 1U);
        ConsumeFixedPass(gS, task, blkBase, previousOff, previousRows, stage);
        stage ^= 1U;
        previousOff = rowOff + p;
        previousRows = count;
    }
    ConsumeFixedPass(gS, task, blkBase, previousOff, previousRows, stage);
}

__aicore__ inline void ApplyFixedMaskByToken(const MsaTask& task, uint32_t blkBase, uint32_t rowOff, uint32_t rows,
                                             uint32_t columns = STILE_WIDTH)
{
    if (task.fullEndBlk >= task.visibleEndBlk) {
        return;
    }
    const uint32_t jLo = task.fullEndBlk > blkBase ? task.fullEndBlk - blkBase : 0U;
    if (jLo >= MSA_BLOCKS_PER_STILE) {
        return;
    }
    const uint32_t jHi =
        MsaMinU32(MSA_BLOCKS_PER_STILE, task.visibleEndBlk > blkBase ? task.visibleEndBlk - blkBase : 0U);
    const uint8_t rowStride = static_cast<uint8_t>(columns / MSA_FP32_PER_BLOCK);
    bool touched = false;
    for (uint32_t r = 0; r < rows;) {
        const uint32_t rowInReq = task.mStart + rowOff + r;
        const uint32_t token = rowInReq / numQHeads_;
        const uint32_t run = MsaMinU32(rows - r, numQHeads_ - rowInReq % numQHeads_);
        const int32_t visibleEnd = VisibleKeyEnd(task, static_cast<int32_t>(token));
        for (uint32_t j = jLo; j < jHi; ++j) {
            const int32_t count = visibleEnd - static_cast<int32_t>((blkBase + j) * MSA_BLOCK_SIZE);
            if (count >= static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                continue;
            }
            const uint32_t valid = count > 0 ? static_cast<uint32_t>(count) : 0U;
            const uint32_t base = r * columns + j * MSA_BLOCK_SIZE;
            // Each half-page is one FP32 repeat. A run is bounded by
            // MSA_ROWS_PER_PASS, including starts in the middle of a head group.
            for (uint32_t half = 0; half < MSA_BLOCK_SIZE; half += MSA_FP32_PER_REPEAT) {
                if (valid >= half + MSA_FP32_PER_REPEAT) {
                    continue;
                }
                const uint32_t lane = valid > half ? valid - half : 0U;
                uint64_t mask[MSA_VEC_MASK_WORDS] = {~0ULL << lane, 0};
                AscendC::Duplicate<float>(ubS_[base + half], MSA_FILL_VALUE, mask, static_cast<uint8_t>(run), 1,
                                          rowStride);
            }
            touched = true;
        }
        r += run;
    }
    if (touched) {
        AscendC::PipeBarrier<PIPE_V>();
    }
}
