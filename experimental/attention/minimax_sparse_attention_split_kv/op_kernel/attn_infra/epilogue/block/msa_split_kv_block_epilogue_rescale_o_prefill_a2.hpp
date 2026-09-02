/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_MSA_SPLIT_KV_BLOCK_EPILOGUE_RESCALE_O_PREFILL_A2_HPP
#define EPILOGUE_BLOCK_MSA_SPLIT_KV_BLOCK_EPILOGUE_RESCALE_O_PREFILL_A2_HPP

#include "../../../attn_infra/msa_split_kv_base_defs.hpp"
#include "../../../attn_infra/arch/msa_split_kv_resource.hpp"
#include "../../../attn_infra/epilogue/msa_split_kv_epilogue_dispatch_policy.hpp"
#include "kernel_operator.h"

#ifndef KERNEL_DUMP
#define KERNEL_DUMP 0
#endif

#ifndef KERNEL_DUMP_PHASE2_STATE
#define KERNEL_DUMP_PHASE2_STATE 0
#endif

using namespace AscendC;

namespace NpuArch::Epilogue::Block {

template <class InDtype, class Tws>
class BlockEpilogue<EpilogueRescaleOPrefillA2, InDtype, Tws> {
public:
    using DispatchPolicy = EpilogueRescaleOPrefillA2;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = InDtype;
    using T = float;
    using TwsO = Tws;

    static constexpr uint32_t FP32_ONE_BLOCK_SIZE = 8U;
    static constexpr uint32_t BYTE_BLOCK = 32U;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64U;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8U;
    static constexpr float NEG_INF_LSE = -3.4028235e38f;

    static constexpr uint32_t ACCUM_OUT_STAGES = 2U;
    static constexpr uint32_t ACCUM_OUT_FLAG_BASE = 3U;
    static constexpr uint32_t LSE_OUT_EVENT_ID = 2U;
    static constexpr float POS_INF_LSE = 3e+99;

    __aicore__ inline BlockEpilogue() {}

    // ---------------------------------------------------------------------------
    // InitFDBuffers: compact UB layout for A2 192KB
    // ---------------------------------------------------------------------------
    // Variable-sized slots (vs A5's fixed 32K-float slots) to fit A2 192KB UB:
    //   lseMaxBuf       [topK * groupSize * 8] floats  (max 2048 → 8KB)
    //   lseSumBuf       [topK * groupSize * 8] floats  (max 2048 → 8KB)
    //   dstBuf          [groupSize * headDimAlignFp32] TwsO (max 8192 → 32KB fp32 / 16KB bf16)
    //   broadCastTmpBuf [2*alignedLseCount + scratch] floats
    //   accumOutBuf[2]  [splitsPerBuf * groupSize * headDimAlignFp32] TwsO each
    // Total (worst case, fp32, splitsPerBuf=1): about 122KB -> fits 192KB
    // ---------------------------------------------------------------------------
    __aicore__ inline void InitFDBuffers(AscendC::LocalTensor<T> &ubBase, uint32_t D, uint32_t groupSize, uint32_t topK,
                                         uint32_t kvHeads)
    {
        D_ = D;
        groupSize_ = groupSize;
        topK_ = topK;
        kvHeads_ = kvHeads;
        headDimAlignFp32_ = RoundUp(D_, FP32_ONE_BLOCK_SIZE);

        // Slot sizes (in floats)
        uint32_t lseFloats = topK_ * groupSize_ * FP32_ONE_BLOCK_SIZE;
        uint32_t dstFloats = groupSize_ * headDimAlignFp32_;
        uint32_t alignedLseCount = RoundUp(topK_ * groupSize_, FP32_ONE_BLOCK_SIZE);
        uint32_t dealCount = groupSize_ * FP32_ONE_BLOCK_SIZE;
        // Keep the Broadcast work area separate from the scale/reduce scratch.
        // Broadcast receives broadCastTmpBuf_[2 * alignedLseCount] as its
        // temporary storage.  Reusing that storage for max/denominator/numerator
        // immediately afterwards let its final writeback clobber the first
        // numerator vector on A2.  The first split then acquired an invalid
        // weight while later splits remained correct.
        uint32_t scaleScratchFloats = (topK_ + 2U) * dealCount;
        if (scaleScratchFloats < 4U * FLOAT_VECTOR_SIZE) {
            scaleScratchFloats = 4U * FLOAT_VECTOR_SIZE;
        }
        uint32_t scratchFloats = lseFloats + scaleScratchFloats;
        uint32_t bcTmpFloats = 2U * alignedLseCount + scratchFloats;

        // Use the real remaining UB capacity for the O_partial ping-pong
        // buffers.  The previous fixed 512-float budget forced one split per
        // buffer for groupSize=16, D=128, so Phase2 issued 16 small GM reads
        // per task even though the A2 UB can hold eight splits per stage.
        constexpr uint32_t ubFloats = ArchTag::UB_SIZE / sizeof(float);
        const uint32_t castAccumFloats =
            (AscendC::IsSameType<TwsO, bfloat16_t>::value || AscendC::IsSameType<TwsO, half>::value) ? dstFloats : 0U;
        const uint32_t accumFloatsPerSplit = (dstFloats * sizeof(TwsO) + sizeof(float) - 1U) / sizeof(float);
        uint32_t fixedFloats = 2U * lseFloats + dstFloats + bcTmpFloats + castAccumFloats;
        uint32_t cap = 1U;
        if (fixedFloats < ubFloats && accumFloatsPerSplit > 0U) {
            cap = (ubFloats - fixedFloats) / (ACCUM_OUT_STAGES * accumFloatsPerSplit);
            if (cap == 0U) {
                cap = 1U;
            }
        }
        uint32_t half = (topK_ + 1U) / 2U;
        splitsPerBuf_ = (half < 1U) ? 1U : ((cap < half) ? cap : half);
        uint32_t accumOutFloats = splitsPerBuf_ * accumFloatsPerSplit;

        // Carve from ubBase (contiguous, no holes)
        uint32_t off = 0;
        lseMaxBuf_ = ubBase[off];
        off += lseFloats;
        lseSumBuf_ = ubBase[off];
        off += lseFloats;
        dstBuf_ = ubBase[off];
        off += dstFloats;
        broadCastTmpBuf_ = ubBase[off];
        off += bcTmpFloats;
        castAccumBuf_ = ubBase[off];
        off += castAccumFloats;
        for (uint32_t i = 0U; i < ACCUM_OUT_STAGES; ++i) {
            accumOutBuf[i] = ubBase[off + accumOutFloats * i].template ReinterpretCast<TwsO>();
        }
        off += accumOutFloats * ACCUM_OUT_STAGES;
    }

    // ---------------------------------------------------------------------------
    // FlashDecodeCompute: Phase2 main entry (same as A5)
    // ---------------------------------------------------------------------------
    __aicore__ inline void FlashDecodeCompute(uint32_t tmpBlockIdx, uint32_t totalTaskNumP2,
                                              AscendC::GlobalTensor<ElementO> &gmO,
                                              AscendC::GlobalTensor<T> &gmSoftmaxLse,
                                              AscendC::GlobalTensor<TwsO> &gmAccumOut,
                                              AscendC::GlobalTensor<T> &gmSoftmaxMax,
                                              AscendC::GlobalTensor<T> &gmSoftmaxSum, uint32_t numHeads,
                                              uint32_t kvHeads, bool softmaxLseFlag)
    {
        if (tmpBlockIdx >= totalTaskNumP2) {
            return;
        }
        uint32_t qToken = tmpBlockIdx / kvHeads;
        uint32_t kvHeadIdx = tmpBlockIdx % kvHeads;
        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t rowSplit = ((groupSize_ + 7U) / 8U * 8U) / 2U;
        const uint32_t firstRows = (rowSplit > groupSize_) ? groupSize_ : rowSplit;
        const uint32_t rowOffset = (subBlockNum > 1U && AscendC::GetSubBlockIdx() != 0U) ? firstRows : 0U;
        const uint32_t dealRowCount = (subBlockNum > 1U) ?
                                          ((AscendC::GetSubBlockIdx() == 0U) ? firstRows : (groupSize_ - firstRows)) :
                                          groupSize_;
        if (dealRowCount == 0U) {
            return;
        }
        uint64_t attenOutOffset = static_cast<uint64_t>(qToken) * kvHeads * groupSize_ * D_ +
                                  static_cast<uint64_t>(kvHeadIdx) * groupSize_ * D_ +
                                  static_cast<uint64_t>(rowOffset) * D_;
        uint64_t softmaxLseOffset =
            static_cast<uint64_t>(qToken) * numHeads + static_cast<uint64_t>(kvHeadIdx) * groupSize_ + rowOffset;

        actualCombineLoopSize_ = topK_;
        CombineSplitKVRes(attenOutOffset, qToken, kvHeadIdx, rowOffset, dealRowCount, gmO, gmSoftmaxLse, gmAccumOut,
                          gmSoftmaxMax, gmSoftmaxSum, softmaxLseOffset, softmaxLseFlag);
    }

private:
    AscendC::LocalTensor<T> lseMaxBuf_;
    AscendC::LocalTensor<T> lseSumBuf_;
    AscendC::LocalTensor<T> dstBuf_;
    AscendC::LocalTensor<TwsO> accumOutBuf[ACCUM_OUT_STAGES];
    AscendC::LocalTensor<T> broadCastTmpBuf_;
    AscendC::LocalTensor<T> castAccumBuf_;

    uint32_t D_ = 0;
    uint32_t groupSize_ = 0;
    uint32_t topK_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t headDimAlignFp32_ = 0;
    uint32_t actualCombineLoopSize_ = 0;
    uint32_t splitsPerBuf_ = 0;

    __aicore__ inline static uint32_t RoundUp(uint32_t a, uint32_t b)
    {
        return (a + b - 1U) / b * b;
    }

    __aicore__ inline static uint32_t CeilDiv(uint32_t a, uint32_t b)
    {
        return (a + b - 1U) / b;
    }

    __aicore__ inline static uint32_t RoundDown(uint32_t a, uint32_t b)
    {
        return a / b * b;
    }

    __aicore__ inline static void RowMulsFp32(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src,
                                              AscendC::LocalTensor<T> rowScale, uint32_t rowCount, uint32_t columnCount,
                                              uint32_t actualColumnCount)
    {
        constexpr uint32_t elemsPerRepeat = FLOAT_VECTOR_SIZE;
        constexpr uint32_t elemsPerBlock = FP32_ONE_BLOCK_SIZE;
        uint32_t fullRepeats = actualColumnCount / elemsPerRepeat;
        uint32_t remain = actualColumnCount % elemsPerRepeat;

        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1U;
        repeatParams.src1BlkStride = 0U;
        repeatParams.dstBlkStride = 1U;
        repeatParams.src0RepStride = columnCount / elemsPerBlock;
        repeatParams.src1RepStride = 1U;
        repeatParams.dstRepStride = columnCount / elemsPerBlock;

        if (fullRepeats <= rowCount) {
            for (uint32_t d = 0U; d < fullRepeats; ++d) {
                uint32_t offset = d * elemsPerRepeat;
                AscendC::Mul<float>(dst[offset], src[offset], rowScale, elemsPerRepeat, rowCount, repeatParams);
            }
        } else {
            AscendC::BinaryRepeatParams rowParams;
            rowParams.src0BlkStride = 1U;
            rowParams.src1BlkStride = 0U;
            rowParams.dstBlkStride = 1U;
            rowParams.src0RepStride = elemsPerRepeat / elemsPerBlock;
            rowParams.src1RepStride = 0U;
            rowParams.dstRepStride = elemsPerRepeat / elemsPerBlock;
            for (uint32_t row = 0U; row < rowCount; ++row) {
                uint32_t offset = row * columnCount;
                AscendC::Mul<float>(dst[offset], src[offset], rowScale[row * elemsPerBlock], elemsPerRepeat,
                                    fullRepeats, rowParams);
            }
        }
        if (remain > 0U) {
            AscendC::Mul<float>(dst[fullRepeats * elemsPerRepeat], src[fullRepeats * elemsPerRepeat], rowScale, remain,
                                rowCount, repeatParams);
        }
    }

    // Set vector mask for count elements (up to 128)
    __aicore__ inline void SetMask(uint32_t count)
    {
        uint64_t mask = 0U;
        uint64_t one = 1U;
        uint32_t tail = count % FLOAT_VECTOR_SIZE;
        for (uint32_t i = 0U; i < tail; ++i) {
            mask |= one << i;
        }
        if (count == 0U || count >= 2U * FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (count == FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (count >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ inline uint64_t TaskStatBase(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return (static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx) * topK_ * StatSlotStride();
    }

    __aicore__ inline uint32_t StatSlotStride() const
    {
        return RoundUp(groupSize_, FP32_ONE_BLOCK_SIZE);
    }

    __aicore__ inline uint64_t TaskAccumBase(uint32_t qToken, uint32_t kvHeadIdx) const
    {
        return (static_cast<uint64_t>(qToken) * kvHeads_ + kvHeadIdx) * topK_ * groupSize_ * D_;
    }

    __aicore__ inline bool ShouldDumpPhase2State(uint32_t qToken, uint32_t kvHeadIdx, uint32_t rowOffset) const
    {
        return qToken == 0U && kvHeadIdx == 0U && rowOffset == 0U;
    }

    // ---------------------------------------------------------------------------
    // CopyLseIn: same as A5 (standard DataCopyPad + Broadcast + Reciprocal)
    // ---------------------------------------------------------------------------
    __aicore__ inline void CopyLseIn(uint32_t qToken, uint32_t kvHeadIdx, uint32_t dealRowCount, uint32_t rowOffset,
                                     AscendC::GlobalTensor<T> &gmSoftmaxMax, AscendC::GlobalTensor<T> &gmSoftmaxSum,
                                     AscendC::LocalTensor<T> &lseMaxLocal, AscendC::LocalTensor<T> &lseSumLocal)
    {
        uint32_t ubRowStride = dealRowCount * FP32_ONE_BLOCK_SIZE;
        auto lseCount = dealRowCount * topK_, alignedLseCount = RoundUp(lseCount, FP32_ONE_BLOCK_SIZE);

        AscendC::LocalTensor<T> lseMaxTmpBuf = broadCastTmpBuf_;
        AscendC::LocalTensor<T> lseSumTmpBuf = broadCastTmpBuf_[alignedLseCount];
        uint64_t taskBase = TaskStatBase(qToken, kvHeadIdx);
        AscendC::DataCopyPadExtParams<float> copyPadParams(false, 0, 0, 0);
        // Initialize the aligned staging ranges from their aligned bases.
        // Starting a vector op at lseCount would be unaligned when lseCount
        // is not a multiple of the 32-byte float block.
        AscendC::Duplicate<float>(lseMaxTmpBuf, NEG_INF_LSE, static_cast<int32_t>(alignedLseCount));
        AscendC::Duplicate<float>(lseSumTmpBuf, 1.0f, static_cast<int32_t>(alignedLseCount));
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t statSlotStride = StatSlotStride();
        if (statSlotStride == dealRowCount) {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1U;
            copyParams.blockLen = static_cast<uint32_t>(lseCount) * sizeof(float);
            copyParams.srcStride = 0U;
            copyParams.dstStride = 0U;
            AscendC::DataCopyPad(lseMaxTmpBuf, gmSoftmaxMax[taskBase + rowOffset], copyParams, copyPadParams);
            AscendC::DataCopyPad(lseSumTmpBuf, gmSoftmaxSum[taskBase + rowOffset], copyParams, copyPadParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            // Keep every slot at its padded local offset.  When that gap is
            // 32B-aligned, one strided MTE2 transaction gathers all slots.
            const uint32_t statGap = statSlotStride - dealRowCount;
            if ((statGap % FP32_ONE_BLOCK_SIZE) == 0U) {
                AscendC::DataCopyExtParams slotCopyParams;
                slotCopyParams.blockCount = topK_;
                slotCopyParams.blockLen = dealRowCount * sizeof(float);
                slotCopyParams.srcStride = statGap * sizeof(float);
                slotCopyParams.dstStride = statGap / FP32_ONE_BLOCK_SIZE;
                AscendC::DataCopyPad(lseMaxLocal, gmSoftmaxMax[taskBase + rowOffset], slotCopyParams, copyPadParams);
                AscendC::DataCopyPad(lseSumLocal, gmSoftmaxSum[taskBase + rowOffset], slotCopyParams, copyPadParams);
            } else {
                // A non-32B UB stride cannot be expressed by DataCopyPad.
                AscendC::DataCopyExtParams slotCopyParams;
                slotCopyParams.blockCount = 1U;
                slotCopyParams.blockLen = dealRowCount * sizeof(float);
                slotCopyParams.srcStride = 0U;
                slotCopyParams.dstStride = 0U;
                for (uint32_t slot = 0U; slot < topK_; ++slot) {
                    uint32_t paddedOff = slot * statSlotStride;
                    AscendC::DataCopyPad(lseMaxLocal[paddedOff], gmSoftmaxMax[taskBase + paddedOff + rowOffset],
                                         slotCopyParams, copyPadParams);
                    AscendC::DataCopyPad(lseSumLocal[paddedOff], gmSoftmaxSum[taskBase + paddedOff + rowOffset],
                                         slotCopyParams, copyPadParams);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            for (uint32_t slot = 0U; slot < topK_; ++slot) {
                for (uint32_t row = 0U; row < dealRowCount; ++row) {
                    uint32_t compactOff = slot * dealRowCount + row;
                    uint32_t paddedOff = slot * statSlotStride + row;
                    lseMaxTmpBuf.SetValue(compactOff, lseMaxLocal.GetValue(paddedOff));
                    lseSumTmpBuf.SetValue(compactOff, lseSumLocal.GetValue(paddedOff));
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
#if KERNEL_DUMP_PHASE2_STATE
        if (ShouldDumpPhase2State(qToken, kvHeadIdx, rowOffset)) {
            // Direct GM scalar readback, followed by the compact MTE2 result.
            // This separates a bad Phase1 write from a bad Phase2 copy/compact.
            AscendC::LocalTensor<T> directMax = broadCastTmpBuf_[2U * alignedLseCount];
            AscendC::LocalTensor<T> directSum = broadCastTmpBuf_[3U * alignedLseCount];
            AscendC::Duplicate<float>(directMax, 0.0f, static_cast<int32_t>(alignedLseCount));
            AscendC::Duplicate<float>(directSum, 0.0f, static_cast<int32_t>(alignedLseCount));
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t slot = 0U; slot < topK_; ++slot) {
                for (uint32_t row = 0U; row < dealRowCount; ++row) {
                    const uint32_t compactOff = slot * dealRowCount + row;
                    const uint32_t paddedOff = slot * statSlotStride + rowOffset + row;
                    directMax.SetValue(compactOff, gmSoftmaxMax.GetValue(taskBase + paddedOff));
                    directSum.SetValue(compactOff, gmSoftmaxSum.GetValue(taskBase + paddedOff));
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DumpTensor(directMax, 964, alignedLseCount);
            AscendC::DumpTensor(directSum, 965, alignedLseCount);
            AscendC::DumpTensor(lseMaxTmpBuf, 966, alignedLseCount);
            AscendC::DumpTensor(lseSumTmpBuf, 967, alignedLseCount);
        }
#endif
        if (qToken < 2U) {
#if KERNEL_DUMP
            printf("[A2 lse] q=%u kv=%u max0=%f sum0=%f\\n", qToken, kvHeadIdx, lseMaxTmpBuf.GetValue(0),
                   lseSumTmpBuf.GetValue(0));
            AscendC::DumpTensor(lseMaxTmpBuf, 902, alignedLseCount);
            AscendC::DumpTensor(lseSumTmpBuf, 903, alignedLseCount);
#endif
        }
        // One Brcb repeat consumes eight scalars and writes eight 32B blocks.
        // Its destination is therefore rounded up to 64 float elements per
        // repeat. lse{Max,Sum}Local only reserve lseCount * 8 floats, so Brcb
        // is valid only when lseCount itself is a multiple of eight. In
        // particular, topK=3/groupSize=4 and topK=5/groupSize=4 otherwise
        // overrun their LSE buffers and contaminate the following invocation.
        if ((lseCount % FP32_ONE_BLOCK_SIZE) == 0U) {
            AscendC::Brcb(lseMaxLocal.ReinterpretCast<uint32_t>(), lseMaxTmpBuf.ReinterpretCast<uint32_t>(),
                          lseCount / FP32_ONE_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
            AscendC::Brcb(lseSumLocal.ReinterpretCast<uint32_t>(), lseSumTmpBuf.ReinterpretCast<uint32_t>(),
                          lseCount / FP32_ONE_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, 8));
        } else {
            for (uint32_t idx = 0U; idx < lseCount; ++idx) {
                const uint32_t dstOff = idx * FP32_ONE_BLOCK_SIZE;
                AscendC::Duplicate<float>(lseMaxLocal[dstOff], lseMaxTmpBuf.GetValue(idx), FP32_ONE_BLOCK_SIZE);
                AscendC::Duplicate<float>(lseSumLocal[dstOff], lseSumTmpBuf.GetValue(idx), FP32_ONE_BLOCK_SIZE);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
#if KERNEL_DUMP_PHASE2_STATE
        for (uint32_t idx = 0U; idx < lseCount; ++idx) {
            const float rowSum = lseSumTmpBuf.GetValue(idx);
            lseSumTmpBuf.SetValue(idx, rowSum > 0.0f ? 1.0f / rowSum : 0.0f);
        }
#endif
    }

    // ---------------------------------------------------------------------------
    // ComputeScaleValue: standard vector API replacement for ComputeScaleValue_VF
    // ---------------------------------------------------------------------------
    // Formula: scale[k] = rowSum[k] * exp(rowMax[k] - max_global) / Σ scale[j]
    //
    // Works on the broadcast layout [topK, dealRowCount * 8]:
    //   Step 1: maxGlobal = max_k(lseMaxLocal[k]) (element-wise max across topK)
    //   Step 2: For each k: lseSumLocal[k] = lseSumLocal[k] * exp(lseMaxLocal[k] - maxGlobal)
    //           Accumulate total = Σ lseSumLocal[k]
    //   Step 3: lseSumLocal[k] /= total
    // ---------------------------------------------------------------------------
    __aicore__ inline void ComputeScaleValue(AscendC::LocalTensor<T> &lseMaxLocal, AscendC::LocalTensor<T> &lseSumLocal,
                                             uint32_t dealRowCount, bool dumpState,
                                             AscendC::GlobalTensor<T> &gmSoftmaxLse, uint64_t softmaxLseOffset,
                                             bool softmaxLseFlag)
    {
        uint32_t dealCount = dealRowCount * FP32_ONE_BLOCK_SIZE;
        uint32_t vecRepeats = CeilDiv(dealCount, FLOAT_VECTOR_SIZE);

        // The Broadcast temporary range occupies one expanded LSE region after
        // the two compact LSE staging regions.  Start vector scratch after it.
        uint32_t alignedLseCount = RoundUp(dealRowCount * topK_, FP32_ONE_BLOCK_SIZE);
        uint32_t scratchBase = 2U * alignedLseCount + topK_ * dealCount;
        AscendC::LocalTensor<T> maxGlobalBuf = broadCastTmpBuf_[scratchBase];
        AscendC::LocalTensor<T> totalBuf = broadCastTmpBuf_[scratchBase + dealCount];
        // Keep every numerator live through the denominator reduction.  A2's
        // vector Mul must not write back to lseSumLocal in place: that buffer
        // is both the source rowSum for the remaining splits and the final
        // scale destination.
        AscendC::LocalTensor<T> numeratorBuf = broadCastTmpBuf_[scratchBase + 2U * dealCount];

        // Step 1: maxGlobal = max across topK (element-wise)
        SetMask(dealCount);
        AscendC::Duplicate<float>(maxGlobalBuf, NEG_INF_LSE, dealCount);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < actualCombineLoopSize_; ++k) {
            AscendC::Max<float, false>(maxGlobalBuf, maxGlobalBuf, lseMaxLocal[k * dealCount], (uint64_t)0, vecRepeats,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 2: calculate numerator[k] = rowSum[k] * exp(rowMax[k] - max).
        // The separate [topK, dealCount] storage matches the A5 VF path and
        // avoids the unsupported in-place lseSumLocal multiply on A2.
        SetMask(dealCount);
        AscendC::Duplicate<float>(totalBuf, 0.0f, dealCount);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < actualCombineLoopSize_; ++k) {
            AscendC::Sub<float, false>(numeratorBuf[k * dealCount], lseMaxLocal[k * dealCount], maxGlobalBuf,
                                       (uint64_t)0, vecRepeats, AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp<float, false>(numeratorBuf[k * dealCount], numeratorBuf[k * dealCount], (uint64_t)0,
                                       vecRepeats, AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul<float, false>(numeratorBuf[k * dealCount], numeratorBuf[k * dealCount],
                                       lseSumLocal[k * dealCount], (uint64_t)0, vecRepeats,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add<float, false>(totalBuf, totalBuf, numeratorBuf[k * dealCount], (uint64_t)0, vecRepeats,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (softmaxLseFlag) {
            // FusedInferAttentionScore/IFA contract: LSE = globalMax + log(globalSum),
            // laid out as [T, N, 1] fp32.  maxGlobalBuf/totalBuf are broadcast
            // [row, 8], so a 1-float strided copy compacts one value per head.
            AscendC::Ln<float, false>(numeratorBuf, totalBuf, (uint64_t)0, vecRepeats,
                                      AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add<float, false>(numeratorBuf, numeratorBuf, maxGlobalBuf, (uint64_t)0, vecRepeats,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t row = 0U; row < dealRowCount; ++row) {
                const uint32_t rowOff = row * FP32_ONE_BLOCK_SIZE;
                if (maxGlobalBuf.GetValue(rowOff) <= NEG_INF_LSE) {
                    AscendC::Duplicate<float>(numeratorBuf[rowOff], POS_INF_LSE, FP32_ONE_BLOCK_SIZE);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(LSE_OUT_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(LSE_OUT_EVENT_ID);
            AscendC::DataCopyExtParams lseCopyParams{};
            lseCopyParams.blockCount = dealRowCount;
            lseCopyParams.blockLen = sizeof(T);
            lseCopyParams.srcStride = 0;
            lseCopyParams.dstStride = 0;
            AscendC::DataCopyPad(gmSoftmaxLse[softmaxLseOffset], numeratorBuf, lseCopyParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(LSE_OUT_EVENT_ID);
            // numeratorBuf is reused immediately below to materialize the
            // per-split combine scales.  Drain this GM write before allowing
            // the vector pipe to overwrite its source, then re-prime the
            // event for the next Phase2 task (and ReleaseSyncFlags).
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(LSE_OUT_EVENT_ID);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(LSE_OUT_EVENT_ID);
        }

        // Step 3: recompute one exp(rowMax - globalMax) vector and materialize
        // the final combine weight before moving to the next split.  Phase1
        // stores unnormalized O_partial = P * V.  The original FlashDecode
        // formula uses:
        //   scale = rowSum * exp(rowMax - globalMax) / denominator
        //   O_norm = O_partial / rowSum
        // so the rowSum term cancels during the final multiply:
        //   combinedScale = exp(rowMax - globalMax) / denominator.
        // Store combinedScale in lseSumLocal directly; ReduceFinalRes can then
        // skip the per-row reciprocal load, duplicate, and vector multiply.
        //
        // Recompute one numerator and materialize its scale before
        // moving to the next split.  The denominator above proves that every
        // numerator was correct when accumulated, but A2 does not preserve the
        // first vector in the multi-split numerator array through the complete
        // loop.  Recomputing into a single live vector avoids consuming that
        // stale persistent value.  There are only topK * groupSize scalar
        // weights, so the extra exp is negligible compared with QK/PV.
#if KERNEL_DUMP_PHASE2_STATE
        if (dumpState) {
            AscendC::DumpTensor(totalBuf, 977, dealCount);
            AscendC::DumpTensor(numeratorBuf, 978, topK_ * dealCount);
        }
#endif
        for (uint32_t k = 0; k < actualCombineLoopSize_; ++k) {
            AscendC::Sub<float, false>(numeratorBuf, lseMaxLocal[k * dealCount], maxGlobalBuf, (uint64_t)0, vecRepeats,
                                       AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp<float, false>(numeratorBuf, numeratorBuf, (uint64_t)0, vecRepeats,
                                       AscendC::UnaryRepeatParams(1, 1, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
#if KERNEL_DUMP_PHASE2_STATE
            if (dumpState) {
                printf("[A2 p2scale] split=%u exp0=%f denominator0=%f\\n", k, numeratorBuf.GetValue(0),
                       totalBuf.GetValue(0));
            }
#endif
            // Keep the vector path for the block-aligned production layout.
            // For a non-aligned LSE count, retain the original guarded scalar
            // division: padded/invalid slots may carry a zero denominator and
            // must materialize a zero scale instead of evaluating 0 / 0.
            if (((topK_ * dealRowCount) % FP32_ONE_BLOCK_SIZE) == 0U) {
                AscendC::Div<float, false>(lseSumLocal[k * dealCount], numeratorBuf, totalBuf, (uint64_t)0, vecRepeats,
                                           AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            } else {
                for (uint32_t row = 0U; row < dealRowCount; ++row) {
                    const uint32_t rowOff = row * FP32_ONE_BLOCK_SIZE;
                    const float denominator = totalBuf.GetValue(rowOff);
                    const float expValue = numeratorBuf.GetValue(rowOff);
                    const float rowSum = lseSumLocal[k * dealCount + rowOff].GetValue(0);
                    const float scale = (denominator > 0.0f && rowSum > 0.0f) ? expValue / denominator : 0.0f;
                    AscendC::Duplicate<float>(lseSumLocal[k * dealCount + rowOff], scale, FP32_ONE_BLOCK_SIZE);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---------------------------------------------------------------------------
    // CopyAccumOutIn: batched contiguous read (same as A5)
    // ---------------------------------------------------------------------------
    __aicore__ inline void CopyAccumOutIn(uint32_t qToken, uint32_t kvHeadIdx, uint32_t jStart, uint32_t nSplits,
                                          uint32_t dealRowCount, uint32_t rowOffset,
                                          AscendC::GlobalTensor<TwsO> &gmAccumOut,
                                          AscendC::LocalTensor<TwsO> &accumOutLocal)
    {
        constexpr uint32_t blockElems = BYTE_BLOCK / sizeof(TwsO);
        AscendC::DataCopyPadExtParams<TwsO> copyInPadParams;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (headDimAlignFp32_ - D_) % blockElems;
        copyInPadParams.paddingValue = 0;

        // Each subblock's rows are contiguous within one split. Treat every
        // split as one block and skip the peer subblock rows with byte strides.
        // This layout is valid for both fp32 and bf16 O_partial.
        const uint32_t gapRows = groupSize_ - dealRowCount;
        const uint32_t dstGapBytes = gapRows * headDimAlignFp32_ * sizeof(TwsO);
        if ((nSplits > 1U) && (headDimAlignFp32_ == D_) && ((dstGapBytes % BYTE_BLOCK) == 0U)) {
            AscendC::DataCopyExtParams batchCopyParams;
            batchCopyParams.blockCount = nSplits;
            batchCopyParams.blockLen = dealRowCount * D_ * sizeof(TwsO);
            batchCopyParams.srcStride = gapRows * D_ * sizeof(TwsO);
            batchCopyParams.dstStride = dstGapBytes / BYTE_BLOCK;
            const uint64_t src = TaskAccumBase(qToken, kvHeadIdx) + static_cast<uint64_t>(jStart) * groupSize_ * D_ +
                                 static_cast<uint64_t>(rowOffset) * D_;
            AscendC::DataCopyPad(accumOutLocal, gmAccumOut[src], batchCopyParams, copyInPadParams);
            return;
        }

        // Generic path for one split or padded dimensions.
        for (uint32_t s = 0U; s < nSplits; ++s) {
            AscendC::DataCopyExtParams copyInParams;
            copyInParams.blockCount = dealRowCount;
            copyInParams.blockLen = static_cast<uint32_t>(D_) * sizeof(TwsO);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = (headDimAlignFp32_ - D_) / blockElems;
            uint64_t src = TaskAccumBase(qToken, kvHeadIdx) + static_cast<uint64_t>(jStart + s) * groupSize_ * D_ +
                           static_cast<uint64_t>(rowOffset) * D_;
            uint64_t dst = static_cast<uint64_t>(s) * groupSize_ * headDimAlignFp32_;
            AscendC::DataCopyPad(accumOutLocal[dst], gmAccumOut[src], copyInParams, copyInPadParams);
        }
    }

    // ---------------------------------------------------------------------------
    // ReduceFinalRes: standard vector API replacement for ReduceFinalRes_VF(_BF16)
    // ---------------------------------------------------------------------------
    // For each split j and row k:
    //   combined_scale = exp(rowMax - globalMax) / denominator
    //   if j == 0: dst[k] = combined_scale * accumOut[j][k]
    //   if j > 0:  dst[k] += combined_scale * accumOut[j][k]
    //
    // For a 16-bit path (TwsO=bfloat16_t or half):
    //   accumOut is cast to fp32 before Mul, dst stays fp32 across splits,
    //   Cast fp32→output happens in CopyFinalResOut.
    // ---------------------------------------------------------------------------
    __aicore__ inline void ReduceFinalRes(uint32_t qToken, uint32_t kvHeadIdx, AscendC::LocalTensor<T> &dst,
                                          AscendC::LocalTensor<T> &scaleLocal, uint32_t dealRowCount,
                                          uint32_t rowOffset, AscendC::GlobalTensor<TwsO> &gmAccumOut)
    {
        uint32_t lseCount = dealRowCount * topK_;
        uint32_t alignedLseCount = RoundUp(lseCount, FP32_ONE_BLOCK_SIZE);
        uint32_t scratchBase = 2U * alignedLseCount + lseCount * FP32_ONE_BLOCK_SIZE;
        uint32_t dLoops = headDimAlignFp32_ / FLOAT_VECTOR_SIZE;

        // Prime each stage's V_MTE2
        for (uint32_t s = 0U; s < ACCUM_OUT_STAGES; ++s) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + s);
        }

        uint32_t numGroups = (topK_ + splitsPerBuf_ - 1U) / splitsPerBuf_;
        bool combineDone = false;
        for (uint32_t g = 0U; (g < numGroups) && !combineDone; ++g) {
            uint32_t jStart = g * splitsPerBuf_;
            uint32_t nSplits = splitsPerBuf_;
            uint32_t remain = topK_ - jStart;
            if (nSplits > remain) {
                nSplits = remain;
            }
            uint32_t curId = g % ACCUM_OUT_STAGES;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + curId);
            CopyAccumOutIn(qToken, kvHeadIdx, jStart, nSplits, dealRowCount, rowOffset, gmAccumOut, accumOutBuf[curId]);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ACCUM_OUT_FLAG_BASE + curId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ACCUM_OUT_FLAG_BASE + curId);
            if (qToken < 2U) {
#if KERNEL_DUMP
                AscendC::DumpTensor(accumOutBuf[curId], 950 + jStart, nSplits * dealRowCount * headDimAlignFp32_);
                auto raw = accumOutBuf[curId].template ReinterpretCast<uint16_t>();
                printf("[A2 OpartialBits] block=%u sub=%u q=%u kv=%u %u %u %u %u\\n", AscendC::GetBlockIdx(),
                       AscendC::GetSubBlockIdx(), qToken, kvHeadIdx, raw.GetValue(0), raw.GetValue(1), raw.GetValue(2),
                       raw.GetValue(3));
#endif
            }

            for (uint32_t j = jStart; j < jStart + nSplits; ++j) {
                uint32_t maxOff = j * dealRowCount * FP32_ONE_BLOCK_SIZE;
                if (lseMaxBuf_[maxOff].GetValue(0) <= NEG_INF_LSE) {
                    combineDone = true;
                    break;
                }
                uint64_t localOff = static_cast<uint64_t>(j - jStart) * groupSize_ * headDimAlignFp32_;

                const uint32_t totalElems = dealRowCount * headDimAlignFp32_;
                const uint32_t addRepeats = totalElems / FLOAT_VECTOR_SIZE;
                AscendC::LocalTensor<T> splitFp32;
                if constexpr (AscendC::IsSameType<TwsO, bfloat16_t>::value || AscendC::IsSameType<TwsO, half>::value) {
                    // Keep the 16-bit O_partial compact in GM/UB, but cast the
                    // complete split once so all weighted accumulation remains fp32.
                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                    AscendC::Cast<float, TwsO, false>(castAccumBuf_, accumOutBuf[curId][localOff],
                                                      AscendC::RoundMode::CAST_NONE, (uint64_t)0, addRepeats,
                                                      AscendC::UnaryRepeatParams(1, 1, 8, 4));
                    AscendC::PipeBarrier<PIPE_V>();
                    splitFp32 = castAccumBuf_;
                } else {
                    splitFp32 = accumOutBuf[curId][localOff];
                }

                if (((totalElems % FLOAT_VECTOR_SIZE) == 0U) && (addRepeats <= 255U)) {
                    AscendC::LocalTensor<T> rowScale =
                        scaleLocal[static_cast<uint64_t>(j) * dealRowCount * FP32_ONE_BLOCK_SIZE];
                    if (j == 0U) {
                        RowMulsFp32(dst, splitFp32, rowScale, dealRowCount, headDimAlignFp32_, D_);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        // The scaled split is dead after this reduction, so
                        // scale it in place.  A full [dealRowCount, D] tmpMul
                        // can outgrow the LSE scratch area for supported
                        // D=128/256 shapes;
                        // placing it there aliases castAccumBuf_ on the BF16
                        // path and corrupts the source while RowMuls is still
                        // reading it.
                        RowMulsFp32(splitFp32, splitFp32, rowScale, dealRowCount, headDimAlignFp32_, D_);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add<float>(dst, dst, splitFp32, FLOAT_VECTOR_SIZE, addRepeats,
                                            AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    continue;
                }

                // Fallback for a non-64-element row layout.
                AscendC::LocalTensor<T> combinedScale = broadCastTmpBuf_[scratchBase];
                AscendC::LocalTensor<T> tmpMul = broadCastTmpBuf_[scratchBase + 2U * FLOAT_VECTOR_SIZE];
                for (uint32_t k = 0U; k < dealRowCount; ++k) {
                    const uint32_t scaleOff = j * dealRowCount * FP32_ONE_BLOCK_SIZE + k * FP32_ONE_BLOCK_SIZE;
                    const float scaleValue = scaleLocal[scaleOff].GetValue(0);
                    AscendC::Duplicate<float>(combinedScale, scaleValue, FLOAT_VECTOR_SIZE);
                    AscendC::PipeBarrier<PIPE_V>();
                    const uint64_t dstOff = static_cast<uint64_t>(k) * headDimAlignFp32_;
                    for (uint32_t z = 0U; z < dLoops; ++z) {
                        AscendC::Mul<float>(tmpMul, splitFp32[dstOff + z * FLOAT_VECTOR_SIZE], combinedScale,
                                            FLOAT_VECTOR_SIZE, 1U, AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 1));
                        AscendC::PipeBarrier<PIPE_V>();
                        if (j == 0U) {
                            AscendC::DataCopy(
                                dst[dstOff + z * FLOAT_VECTOR_SIZE], tmpMul,
                                AscendC::DataCopyParams(1, FLOAT_VECTOR_SIZE / FP32_ONE_BLOCK_SIZE, 0, 0));
                        } else {
                            AscendC::Add<float>(dst[dstOff + z * FLOAT_VECTOR_SIZE],
                                                dst[dstOff + z * FLOAT_VECTOR_SIZE], tmpMul, FLOAT_VECTOR_SIZE, 1U,
                                                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + curId);
        }
        // Trailing drain
        for (uint32_t s = 0U; s < ACCUM_OUT_STAGES; ++s) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ACCUM_OUT_FLAG_BASE + s);
        }
    }

    // ---------------------------------------------------------------------------
    // CopyFinalResOut: write final result from UB to GM (same as A5, simplified)
    // ---------------------------------------------------------------------------
    __aicore__ inline void CopyFinalResOut(uint64_t attenOutOffset, AscendC::GlobalTensor<ElementO> &gmO,
                                           AscendC::LocalTensor<T> &accumOutLocal, uint32_t dealRowCount)
    {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<ElementO> castBuf = broadCastTmpBuf_.template ReinterpretCast<ElementO>();
        uint32_t shapeArray[] = {dealRowCount, D_};
        castBuf.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));
        uint32_t totalElems = dealRowCount * headDimAlignFp32_;
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::Cast<ElementO, float, false>(castBuf, accumOutLocal, AscendC::RoundMode::CAST_ROUND, (uint64_t)0,
                                              (totalElems + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                              AscendC::UnaryRepeatParams(1, 1, 4, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = static_cast<uint32_t>(D_) * sizeof(ElementO);
        dataCopyParams.srcStride = (headDimAlignFp32_ - D_) / (BYTE_BLOCK / sizeof(ElementO));
        dataCopyParams.dstStride = 0;
        AscendC::DataCopyPad(gmO[attenOutOffset], castBuf, dataCopyParams);
        // The next Phase2 task handled by this AIV reuses dstBuf_.  Publish
        // MTE3 completion so that task waits until this copy has stopped
        // reading the shared UB storage.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    // ---------------------------------------------------------------------------
    // CombineSplitKVRes: top-level Phase2 combine flow (same as A5)
    // ---------------------------------------------------------------------------
    __aicore__ inline void CombineSplitKVRes(
        uint64_t attenOutOffset, uint32_t qToken, uint32_t kvHeadIdx, uint32_t rowOffset, uint32_t dealRowCount,
        AscendC::GlobalTensor<ElementO> &gmO, AscendC::GlobalTensor<T> &gmSoftmaxLse,
        AscendC::GlobalTensor<TwsO> &gmAccumOut, AscendC::GlobalTensor<T> &gmSoftmaxMax,
        AscendC::GlobalTensor<T> &gmSoftmaxSum, uint64_t softmaxLseOffset, bool softmaxLseFlag)
    {
        // EVENT_ID0 is primed by the kernel for the first task and set by
        // CopyFinalResOut for every subsequent task.  Wait before touching
        // either dstBuf_ (BF16 output source) or broadCastTmpBuf_ (FP32 cast
        // output source).
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        if (softmaxLseFlag) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(LSE_OUT_EVENT_ID);
        }
        CopyLseIn(qToken, kvHeadIdx, dealRowCount, rowOffset, gmSoftmaxMax, gmSoftmaxSum, lseMaxBuf_, lseSumBuf_);
#if KERNEL_DUMP_PHASE2_STATE
        if (ShouldDumpPhase2State(qToken, kvHeadIdx, rowOffset)) {
            AscendC::DumpTensor(lseSumBuf_, 969, topK_ * groupSize_ * FP32_ONE_BLOCK_SIZE);
        }
#endif
        ComputeScaleValue(lseMaxBuf_, lseSumBuf_, dealRowCount, ShouldDumpPhase2State(qToken, kvHeadIdx, rowOffset),
                          gmSoftmaxLse, softmaxLseOffset, softmaxLseFlag);
#if KERNEL_DUMP_PHASE2_STATE
        if (ShouldDumpPhase2State(qToken, kvHeadIdx, rowOffset)) {
            const uint32_t lseCount = topK_ * groupSize_;
            const uint32_t lseVecCount = lseCount * FP32_ONE_BLOCK_SIZE;
            const uint32_t alignedLseCount = RoundUp(lseCount, FP32_ONE_BLOCK_SIZE);
            const uint32_t dealCount = groupSize_ * FP32_ONE_BLOCK_SIZE;
            printf("[A2 p2state] q=%u kv=%u lseCount=%u\\n", qToken, kvHeadIdx, lseCount);
            AscendC::DumpTensor(lseMaxBuf_, 970, lseVecCount);
            AscendC::DumpTensor(lseSumBuf_, 971, lseVecCount);
            AscendC::DumpTensor(broadCastTmpBuf_[alignedLseCount], 972, alignedLseCount);
            const uint32_t scratchBase = 2U * alignedLseCount + lseVecCount;
            AscendC::DumpTensor(broadCastTmpBuf_[scratchBase], 974, dealCount);
            AscendC::DumpTensor(broadCastTmpBuf_[scratchBase + dealCount], 975, dealCount);
            AscendC::DumpTensor(broadCastTmpBuf_[scratchBase + 2U * dealCount], 976, topK_ * dealCount);
        }
#endif
        AscendC::LocalTensor<T> dst = dstBuf_;
        // Invalid or empty splits may be skipped in ReduceFinalRes; clear the
        // complete tile first so skipped rows cannot expose stale UB contents.
        AscendC::Duplicate<T>(dst, 0.0f, static_cast<int32_t>(dealRowCount * headDimAlignFp32_));
        AscendC::PipeBarrier<PIPE_V>();
        ReduceFinalRes(qToken, kvHeadIdx, dst, lseSumBuf_, dealRowCount, rowOffset, gmAccumOut);
#if KERNEL_DUMP_PHASE2_STATE
        if (ShouldDumpPhase2State(qToken, kvHeadIdx, rowOffset)) {
            AscendC::DumpTensor(dst, 973, dealRowCount * headDimAlignFp32_);
        }
#endif
#if KERNEL_DUMP
        if (qToken == 0U && kvHeadIdx == 0U) {
            AscendC::DumpTensor(dst, 960, dealRowCount * headDimAlignFp32_);
        }
#endif
        CopyFinalResOut(attenOutOffset, gmO, dst, dealRowCount);
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_MSA_SPLIT_KV_BLOCK_EPILOGUE_RESCALE_O_PREFILL_A2_HPP
