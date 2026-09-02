/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_grouped_matmul_dequant_phase_x.h
 * \brief Out-of-class definitions for Phase X (PhaseXDynamicQuant + PhaseXWriteXL1).
 *
 * Included from quant_grouped_matmul_dequant.h AFTER the class template
 * declaration. Reopens namespace AscendC for the member function bodies.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_PHASE_X_H
#define QUANT_GROUPED_MATMUL_DEQUANT_PHASE_X_H

namespace AscendC {

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_,
                                                 UseXZNStage_>::PhaseXDynamicQuant(uint64_t mStart, uint64_t mCount)
{
    // Per-row single-rounding DynamicQuant, streamed through phaseXChunk so
    // UB fits even when K > 4096. For row m:
    //   1. max-reduce pass: walk K in chunks, Abs+ReduceMax, keep running
    //      max in an S-register across chunks.
    //   2. quantize pass:  walk K in chunks, Cast triad per chunk, write
    //      int8ScratchLT_[m*kBytes + off] per chunk.
    // scale = 1/(max/127) stays in the S-register (never crosses UB<->GM).
    // Phase X writes only UB; the chunk loop flushes int8 to xL1 via
    // ScatterZZToL1 before Phase MM.

    const uint32_t K = K_;
    const uint32_t kBytes = tiling_->originKAligned32;
    const uint32_t chunkHalves = tiling_->phaseXChunk;

    // Previous iter's Phase X WriteXL1 must have finished MTE3 UB->L1 before
    // we clobber int8ScratchLT_ again. Primed in PrimeEvents().
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    // Drain previous iter's V writes to Phase D buffers before we MTE2
    // overwrite the aliased Phase X workspace. Pair with SetFlag at end
    // of ProcessGroup. Primed in PrimeEvents.
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);

    // Load smoothScale GM->UB once per call; buffer sized K halves, reused
    // across rows. The smooth-quant path needs it before any quantize step.
    if constexpr (HasSmooth) {
        DataCopy(smoothScaleLT_, smoothScaleGm_, K);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
    }

    // Per-row prime/drain of V_MTE2(0): each row is independently balanced
    // so no cross-row state carries a stale token.
    for (uint64_t m = 0; m < mCount; ++m) {
        const uint64_t rowGmOff = (mStart + m) * static_cast<uint64_t>(K);

        // Prime V_MTE2(0) for this row; drained by the WaitFlag at end-of-row.
        // Each chunk: MTE2->V (consume) -> V ops -> V_MTE2 -> next MTE2 waits.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);

        // ================ Pass 1: max-reduce over K chunks ================
        float running_max = 0.0f;
        for (uint32_t coff = 0; coff < kBytes; coff += chunkHalves) {
            const uint32_t thisChunk = (coff + chunkHalves > kBytes) ? (kBytes - coff) : chunkHalves;
            const uint32_t realInChunk = (coff >= K) ? 0u : ((K - coff < thisChunk) ? (K - coff) : thisChunk);
            // Wait for previous chunk's V ops to finish reading xRawPingLT_
            // before MTE2 clobbers it. Primed outside the loop / re-set after
            // each chunk's V body.
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            if (realInChunk > 0) {
                DataCopy(xRawPingLT_, xGm_[rowGmOff + coff], realInChunk);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            if (realInChunk < thisChunk) {
                Duplicate<half>(xRawPingLT_[realInChunk], static_cast<half>(0), thisChunk - realInChunk);
                PipeBarrier<PIPE_V>();
            }

            // Smooth path stays fp32 across the whole max-reduce (Mul, Abs,
            // ReduceMax) so the fp32 product precision is not lost:
            //   x'          = (x_fp16 -> fp32) * (smooth_fp16 -> fp32)
            //   |x'|        = Abs<float>(x')
            //   max         = ReduceMax<float>(|x'|)
            //   running_max = max across chunks (S-register, fp32)
            // No-smooth: fp16 max-reduce path.
            float part_f = 0.0f;
            if constexpr (HasSmooth) {
                Cast<float, half>(smoothFp32LT_, smoothScaleLT_[coff], RoundMode::CAST_NONE, thisChunk);
                Cast<float, half>(xCastFloatLT_, xRawPingLT_, RoundMode::CAST_NONE, thisChunk);
                PipeBarrier<PIPE_V>();
                Mul<float>(xScaledLT_, xCastFloatLT_, smoothFp32LT_, thisChunk);
                PipeBarrier<PIPE_V>();
                // |x'| -> reuse xCastFloatLT_ (x_fp32 no longer needed).
                Abs<float>(xCastFloatLT_, xScaledLT_, thisChunk);
                PipeBarrier<PIPE_V>();
                // ReduceMax<float>: borrow xInt32LT_ as fp32 work buffer
                // (sized chunkHalves * 4 bytes, same as input). maxLT_
                // (32 B) reinterprets to a fp32 lane.
                LocalTensor<float> reduceWorkFp32 = xInt32LT_.ReinterpretCast<float>();
                LocalTensor<float> maxFp32 = maxLT_.ReinterpretCast<float>();
                Duplicate<float>(maxFp32, 0.0f, 8);
                PipeBarrier<PIPE_V>();
                ReduceMax<float>(maxFp32, xCastFloatLT_, reduceWorkFp32, thisChunk, false);
                PipeBarrier<PIPE_V>();
                part_f = maxFp32.GetValue(0);
            } else {
                Abs<half>(absLT_, xRawPingLT_, thisChunk);
                PipeBarrier<PIPE_V>();
                Duplicate<half>(maxLT_, static_cast<half>(0), 16);
                PipeBarrier<PIPE_V>();
                ReduceMax<half>(maxLT_, absLT_, reduceWorkLT_, thisChunk, false);
                PipeBarrier<PIPE_V>();
                const half part_h = maxLT_.GetValue(0);
                part_f = static_cast<float>(part_h);
            }

            if (part_f > running_max)
                running_max = part_f;
            // xRawPingLT_ is now safe to overwrite.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        const float pts = running_max * FLOAT_RECIP_127;
        pertokenLT_.SetValue(static_cast<uint32_t>(m), pts);

        // Drain this row's leftover V_MTE2 token from Pass 1's last chunk.
        // Pairs with the SetFlag at top of this iteration.
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    /**
     * Pass 2: chunk-outer / row-inner
     * Per-chunk MTE3 scatter int8 -> xL1; bounds int8Scratch UB to
     * Mb*chunkHalves (not Mb*K) so tiling can pick larger Mb at K>=4096.
     *
     *   for coff in chunks:
     *     for m in mCount:
     *       MTE2 fp16 chunk -> xRawPingLT_
     *       V: cast triad with scale = 1/pts[m] -> int8ScratchLT_[m*chunk]
     *     (after all rows for this chunk filled int8Scratch)
     *     V -> MTE3 sync; ScatterZZToL1 chunk to xL1[coff..coff+thisChunk]
     *     (gate next chunk's V on MTE3 done)
     *
     * Sync flags (cross-chunk): V_MTE3 + MTE3_V on EVENT_ID0.
     */
    {
        const uint32_t kBytes2 = tiling_->originKAligned32;
        const uint32_t mAligned = ((static_cast<uint32_t>(mCount) + M_FRACTAL - 1) / M_FRACTAL) * M_FRACTAL;
        const uint32_t l1TurnStride = kBytes2 * M_FRACTAL;

        // Gate first chunk's MTE3 on previous iter's MTE1 having finished
        // reading xL1 (paired with SetFlag<MTE1_MTE3>(EVENT_ID0) at end of
        // ProcessGroup). Primed in PrimeEvents for the first iteration.
        WaitFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);

        // Per-row V_MTE2 prime (pairs with WaitFlag inside the row loop).
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);

        for (uint32_t coff = 0; coff < kBytes2; coff += chunkHalves) {
            const uint32_t thisChunk = (coff + chunkHalves > kBytes2) ? (kBytes2 - coff) : chunkHalves;
            const uint32_t realInChunk = (coff >= K) ? 0u : ((K - coff < thisChunk) ? (K - coff) : thisChunk);

            for (uint64_t m = 0; m < mCount; ++m) {
                const uint64_t rowGmOff = (mStart + m) * static_cast<uint64_t>(K);
                const float pts = pertokenLT_.GetValue(static_cast<uint32_t>(m));
                const float scale = (pts > DQ_MAX_EPS) ? (1.0f / pts) : 0.0f;

                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                if (realInChunk > 0) {
                    DataCopy(xRawPingLT_, xGm_[rowGmOff + coff], realInChunk);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
                if (realInChunk < thisChunk) {
                    Duplicate<half>(xRawPingLT_[realInChunk], static_cast<half>(0), thisChunk - realInChunk);
                    PipeBarrier<PIPE_V>();
                }

                // Cast triad: fp16 -> fp32 -> Muls(scale) -> int32 RINT ->
                // half ROUND -> int8 TRUNC.
                LocalTensor<float> fpAfterScale;
                if constexpr (HasSmooth) {
                    Cast<float, half>(smoothFp32LT_, smoothScaleLT_[coff], RoundMode::CAST_NONE, thisChunk);
                    Cast<float, half>(xCastFloatLT_, xRawPingLT_, RoundMode::CAST_NONE, thisChunk);
                    PipeBarrier<PIPE_V>();
                    Mul<float>(xScaledLT_, xCastFloatLT_, smoothFp32LT_, thisChunk);
                    PipeBarrier<PIPE_V>();
                    Muls<float>(xCastFloatLT_, xScaledLT_, scale, thisChunk);
                    fpAfterScale = xCastFloatLT_;
                } else {
                    Cast<float, half>(xCastFloatLT_, xRawPingLT_, RoundMode::CAST_NONE, thisChunk);
                    PipeBarrier<PIPE_V>();
                    Muls<float>(xScaledLT_, xCastFloatLT_, scale, thisChunk);
                    fpAfterScale = xScaledLT_;
                }
                PipeBarrier<PIPE_V>();
                Cast<int32_t, float>(xInt32LT_, fpAfterScale, RoundMode::CAST_RINT, thisChunk);
                SetDeqScale(static_cast<half>(1.0f));
                PipeBarrier<PIPE_V>();
                Cast<half, int32_t>(xHalfLT_, xInt32LT_, RoundMode::CAST_ROUND, thisChunk);
                PipeBarrier<PIPE_V>();
                // int8Scratch slot for this row uses STRIDE = thisChunk
                // (NOT chunkHalves) so the per-row data is packed tight
                // for ScatterZZToL1's row stride (which is chunkSize =
                // thisChunk). For the last chunk of a K not divisible by
                // chunkHalves, thisChunk < chunkHalves; using chunkHalves
                // would leave gaps that ScatterZZToL1 reads as garbage.
                Cast<int8_t, half>(int8ScratchLT_[m * thisChunk], xHalfLT_, RoundMode::CAST_TRUNC, thisChunk);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }

            // Zero int8Scratch padding rows (mCount..mAligned-1) so the
            // ScatterZZToL1 below (mValid=mAligned) overwrites stale xL1
            // padding; else those rows leak into L0A and contaminate output
            // rows 0..mCount-1. Reinterpret as half + chunked Duplicate
            // (<=128) since Duplicate<half>(count>128) silently truncates.
            if (mCount < mAligned) {
                LocalTensor<half> int8AsHalf = int8ScratchLT_.template ReinterpretCast<half>();
                const uint32_t halvesPerRow = thisChunk / 2; // 2 bytes int8 = 1 half
                for (uint64_t pad_m = static_cast<uint64_t>(mCount); pad_m < static_cast<uint64_t>(mAligned); ++pad_m) {
                    uint32_t written = 0;
                    const uint32_t base = static_cast<uint32_t>(pad_m) * (thisChunk / 2);
                    while (written < halvesPerRow) {
                        const uint32_t cnt = (halvesPerRow - written > 128U) ? 128U : (halvesPerRow - written);
                        Duplicate<half>(int8AsHalf[base + written], static_cast<half>(0), cnt);
                        written += cnt;
                    }
                }
                PipeBarrier<PIPE_V>();
            }

            // After all rows quantized for this chunk: V's last write to
            // int8Scratch is done; MTE3 can scatter to xL1.
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            // mValid=mAligned to scatter zero padding rows too; overwrites
            // xL1 stale padding from prev iter.
            ScatterZZToL1(xL1_, int8ScratchLT_, mAligned, coff, thisChunk, l1TurnStride,
                          static_cast<uint32_t>(mAligned));
            // Gate the NEXT chunk's V (which overwrites int8Scratch) on the
            // current chunk's MTE3 read finishing. Skip on the last chunk.
            if (coff + thisChunk < kBytes2) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }

        // Drain the V_MTE2 prime emitted before the chunk loop.
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
}

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_,
                                                 UseXZNStage_>::PhaseXWriteXL1(uint64_t mCount)
{
    // xL1 is populated by Pass 2's per-chunk scatter; this only emits the
    // cross-pipe SetFlags gating downstream work on Phase X finishing:
    //   MTE3_MTE1(0): xL1 ready for Mmad LoadData (consumed in the cb loop).
    //   MTE3_V(7):    prime for next iter's PhaseX wait at top.
    (void)mCount;

    SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
}

} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_PHASE_X_H
