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
 * \file quant_grouped_matmul_dequant_phase_mm.h
 * \brief Out-of-class definitions for Phase MM (ProcessGroup + ProcessColBatch).
 *
 * ProcessColBatch is the matmul-centric function: full Mmad K-pass loop plus
 * post-Mmad VDEQ16 + scatter + pertoken Mul + CopyOut. Included from
 * quant_grouped_matmul_dequant.h AFTER the class template declaration.
 * Reopens namespace AscendC for the member function bodies.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_PHASE_MM_H
#define QUANT_GROUPED_MATMUL_DEQUANT_PHASE_MM_H

namespace AscendC {

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_,
                                                 UseXZNStage_>::ProcessGroup(uint32_t groupIdx, uint64_t mStart,
                                                                             uint64_t mCount, uint32_t nCoreIdx,
                                                                             uint32_t nCoreNum)
{
    // Early-return when this core's cb-slice is empty: NCoreNum may exceed
    // numCbs, so cores with nCoreIdx >= cbRem (and cbPerNCore=0) have no cbs.
    // Running PhaseX + SetFlag<MTE1_MTE3>(0) without a matching cb-loop
    // WaitFlag<MTE3_MTE1>(0) leaves the event chain unbalanced -> hang.
    {
        const uint32_t totalColBatch = (N_ + Nb_ - 1) / Nb_;
        const uint32_t cbPerNCore = totalColBatch / nCoreNum;
        const uint32_t cbRem = totalColBatch % nCoreNum;
        const uint32_t myCbLen = cbPerNCore + (nCoreIdx < cbRem ? 1u : 0u);
        if (myCbLen == 0u) {
            return;
        }
    }
    PhaseXDynamicQuant(mStart, mCount);
    PhaseXWriteXL1(mCount);

    // Linear cb iteration 0..totalColBatch-1; L2 reuse comes from the
    // SetL2CacheHint at kernel entry. cb range split by (nCoreIdx, nCoreNum):
    // nCoreNum=1 walks the full range; nCoreNum>1 partitions cbs disjointly
    // across the cores sharing this mTile slice.
    const uint32_t totalColBatch = (N_ + Nb_ - 1) / Nb_;
    const uint32_t cbPerNCore = totalColBatch / nCoreNum;
    const uint32_t cbRem = totalColBatch % nCoreNum;
    const uint32_t cbStart = cbPerNCore * nCoreIdx + (nCoreIdx < cbRem ? nCoreIdx : cbRem);
    const uint32_t cbEnd = cbStart + cbPerNCore + (nCoreIdx < cbRem ? 1u : 0u);
    for (uint32_t cb = cbStart; cb < cbEnd; ++cb) {
        const uint32_t nStart = cb * Nb_;
        const uint32_t curColCount = (N_ - nStart < Nb_) ? (N_ - nStart) : Nb_;
        ProcessColBatch(groupIdx, cb, mStart, mCount, curColCount, cbEnd, cbStart);
    }
    // xL1_ read complete across all cbs; next iter's ScatterZZToL1 may reuse.
    SetFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);
    // Phase D UB region (dequantScratch / copyOut / finalOut) aliases the
    // PhaseX workspace (xRawPing/Pong / xCastFloat / xScaled). Gate next
    // iter's PhaseX MTE2 (writes xRawPing) on these V writes draining.
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
}
template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_,
                                                 UseXZNStage_>::ProcessColBatch(uint32_t groupIdx, uint32_t cbIdx,
                                                                                uint64_t mStart, uint64_t mCount,
                                                                                uint64_t curColCount, uint32_t numCbs,
                                                                                uint32_t cbStart)
{
    // First cb in this core's nCore slice: the first-cb gates (MTE3_MTE1
    // wait for xL1 ready, own-MTE2 path) fire once per slice on its FIRST
    // cb (cbStart), not on the global cb_0.
    const bool isFirstCb = (cbIdx == cbStart);
    (void)isFirstCb;
    // One column batch: compute Mmad(xL1, wL1) @ scale -> fp16 tile, apply
    // pertoken, then MTE3 UB->GM. Q=1 (single channel accumulate step).
    const uint32_t mAligned = ((static_cast<uint32_t>(mCount) + M_FRACTAL - 1) / M_FRACTAL) * M_FRACTAL;
    const uint32_t nSize = static_cast<uint32_t>(curColCount);
    const uint32_t kBytes = tiling_->originKAligned32;
    const uint32_t nStart = cbIdx * Nb_;
    const bool usePing = (cbIdx & 1u) == 0;

    // wL1 buffer selected by MTE2-sequence index (wBufIdx_), not
    // cb-parity. This lets pre-loop chunk prefetch use the OPPOSITE
    // buffer for in-cb chunk overlap. scaleU64/output stay cb-parity
    // (independent lifetimes).
    LocalTensor<int8_t> curWL1 = (wBufIdx_ == 0u) ? wL1PingLT_ : wL1PongLT_;
    LocalTensor<uint64_t> curScU64 = usePing ? scaleU64PingLT_ : scaleU64PongLT_;
    LocalTensor<half> finalOut = usePing ? outPingLT_ : outPongLT_;
    // cb-alternating intermediate: scatter writes here, pertoken Mul reads
    // here -> finalOut. With pertoken, scatterDst is the OPPOSITE of
    // finalOut; without, scatter writes directly to finalOut.
    LocalTensor<half> scatterDst = HasPertoken ? (usePing ? outPongLT_ : outPingLT_) : finalOut;

    // ---- wait the previous cb's MTE3 CopyOut to free finalOutput ----
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    // Previous cb's VDEQ16 must have finished reading this scale slot.
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    // First cb only: wait for this wL1 slot to be free. Later cbs skip it
    // because cb_{i-1}'s end-of-loop prefetch already claimed and filled it.
    if (cbIdx == cbStart) {
        WaitFlag<HardEvent::MTE1_MTE2>((wBufIdx_ == 0u) ? EVENT_ID0 : EVENT_ID1);
    }

    // ---- MTE2: scale + weight into L1/UB ----
    LoadScaleToUB(groupIdx, cbIdx, curColCount, usePing);

    // Expert stride in weight GM must match the ZN storage shape:
    // [E, K1, N1, 16, 32]. K/N tails are padded in the weight tensor, so
    // using the logical K*N would make groupIdx>0 start inside the previous
    // expert when K or N is not already fractal-aligned.
    const uint32_t weightBytesPerExpert = fracK_ * fracN_ * M_FRACTAL * K_FRACTAL_INT8_BYTES;
    const uint64_t expertBaseOff = static_cast<uint64_t>(groupIdx) * weightBytesPerExpert;
    // Chunked K-pass weight load: wL1 holds wChunkKPasses K-passes per MTE2
    // burst (2*wChunk*Kb*Nb per ping/pong half). The kp loop is outer-chunk
    // x inner-kp; each kp's LoadData(l0b) reads a distinct L1 offset.
    // wChunkKPasses=1 collapses to one kp per MTE2.
    const uint32_t Kb1_ = tiling_->Kb / K_FRACTAL_INT8_BYTES;
    const uint32_t K1_total = kBytes / K_FRACTAL_INT8_BYTES;
    const uint32_t kPasses_ = tiling_->kPasses;
    const uint32_t wChunkKPasses_ = tiling_->wChunkKPasses; // >=1, divides kPasses_
    const uint32_t k1PerChunk_ = wChunkKPasses_ * Kb1_;
    // First chunk's K-fractal count, clipped to K1_total when the chunk
    // extends past end (defensive; picker forbids non-dividing wChunk).
    const uint32_t kp0K1Count = (K1_total < k1PerChunk_) ? K1_total : k1PerChunk_;
    // First cb only: issue this cb's first-chunk weight MTE2 into curWL1.
    // Later cbs reuse cb_{i-1}'s prefetch which loaded curWL1 and set
    // MTE2_MTE1(1); the kp loop's wait below consumes that token.
    if (cbIdx == cbStart) {
        PrefetchWeightToL1<BlockedZN_>(curWL1, weightGm_, expertBaseOff,
                                       /* colBatch= */ cbIdx,
                                       /* nSize= */ nSize,
                                       /* baseNG= */ nStart / N_FRACTAL,
                                       /* N1= */ fracN_,
                                       /* K1q= */ kp0K1Count,
                                       /* K= */ K_,
                                       /* weightBlockN1= */ weightBlockN1_,
                                       /* k1Start= */ 0);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1); // weight L1[ZN] ready
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID4); // scale UB ready

    // ---- MTE1 + Cube K-pass loop ----
    // L0B capacity (64KB) forces K-splitting when K*Nb > L0B. Per pass we
    // load Kb (K-elements per pass) into L0A/L0B, Mmad with init=true on
    // kp=0 and init=false afterwards (L0C accumulates). wL1 is per-pass
    // sized; kp>0 issues its own MTE2 into curWL1 (same buffer, overwritten).
    LocalTensor<int8_t> l0a = l0aBuf_.Get<int8_t>();
    LocalTensor<int8_t> l0b = l0bBuf_.Get<int8_t>();
    LocalTensor<int32_t> l0c = l0cBuf_.Get<int32_t>();

    if (cbIdx == cbStart) {
        // xL1_ ready signal is produced once per Phase X; only first cb waits.
        WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    }
    const uint16_t nFracs = static_cast<uint16_t>(nSize / N_FRACTAL);
    const uint16_t mTurns = static_cast<uint16_t>(mAligned / M_FRACTAL);
    const uint32_t xL1TurnStride = kBytes * M_FRACTAL;                   // bytes per ZZ turn
    constexpr uint32_t FRACTAL_BYTES = M_FRACTAL * K_FRACTAL_INT8_BYTES; // 512

    // Outer: chunks (one MTE2 per chunk).  Inner: kp within chunk (one Mmad each).
    const uint32_t numChunks_ = kPasses_ / wChunkKPasses_; // picker ensures clean divide
    const uint32_t kbBytesPerKp = Kb1_ * static_cast<uint32_t>(nFracs) * FRACTAL_BYTES;
    bool nxFired = false; // tracks whether the next MTE2 was pre-fired
    for (uint32_t chunkIdx = 0; chunkIdx < numChunks_; ++chunkIdx) {
        const uint32_t kpBase = chunkIdx * wChunkKPasses_;
        const uint32_t k1BaseStart = kpBase * Kb1_;
        (void)k1BaseStart;

        // This chunk's MTE2 was fired by the previous iter's pre-loop (or
        // the first-cb gate for chunk 0). Wait MTE2_MTE1 before reading curWL1.
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

        // Pre-loop next-MTE2 (wChunkKPasses_ > 1 only, so the overlap window
        // can hide MTE2 setup ~422 cyc); wChunkKPasses_ == 1 falls through
        // to the post-loop fallback.
        nxFired = false;
        if (wChunkKPasses_ > 1) {
            // Tier 1: next chunk in this cb -> opposite buffer. Tier 2: first
            // chunk of next cb. wChunkKPasses==1 handled by post-loop fallback.
            const uint32_t nxBuf = 1u - wBufIdx_;
            LocalTensor<int8_t> nxWL1 = (nxBuf == 0u) ? wL1PingLT_ : wL1PongLT_;
            if (chunkIdx + 1 < numChunks_) {
                const uint32_t k1NxStart = (chunkIdx + 1) * k1PerChunk_;
                const uint32_t k1NxRemain = K1_total - k1NxStart;
                const uint32_t k1NxToLoad = (k1PerChunk_ < k1NxRemain) ? k1PerChunk_ : k1NxRemain;
                WaitFlag<HardEvent::MTE1_MTE2>((nxBuf == 0u) ? EVENT_ID0 : EVENT_ID1);
                PrefetchWeightToL1<BlockedZN_>(nxWL1, weightGm_, expertBaseOff,
                                               /* colBatch= */ cbIdx,
                                               /* nSize= */ nSize,
                                               /* baseNG= */ nStart / N_FRACTAL,
                                               /* N1= */ fracN_,
                                               /* K1q= */ k1NxToLoad,
                                               /* K= */ K_,
                                               /* weightBlockN1= */ weightBlockN1_,
                                               /* k1Start= */ k1NxStart);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
                nxFired = true;
            } else if (cbIdx + 1 < numCbs) {
                const uint32_t nxCb = cbIdx + 1;
                const uint32_t nxNStart = nxCb * Nb_;
                const uint32_t nxColCount = (N_ - nxNStart < Nb_) ? (N_ - nxNStart) : Nb_;
                WaitFlag<HardEvent::MTE1_MTE2>((nxBuf == 0u) ? EVENT_ID0 : EVENT_ID1);
                PrefetchWeightToL1<BlockedZN_>(nxWL1, weightGm_, expertBaseOff,
                                               /* colBatch= */ nxCb,
                                               /* nSize= */ nxColCount,
                                               /* baseNG= */ nxNStart / N_FRACTAL,
                                               /* N1= */ fracN_,
                                               /* K1q= */ kp0K1Count,
                                               /* K= */ K_,
                                               /* weightBlockN1= */ weightBlockN1_,
                                               /* k1Start= */ 0);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
                nxFired = true;
            }
        }

        for (uint32_t kpInChunk = 0; kpInChunk < wChunkKPasses_; ++kpInChunk) {
            const uint32_t kp = kpBase + kpInChunk;
            const uint32_t k1Start = kp * Kb1_;
            const uint32_t k1Count = (K1_total - k1Start < Kb1_) ? (K1_total - k1Start) : Kb1_;
            const uint32_t kThisBytes = k1Count * K_FRACTAL_INT8_BYTES;
            const uint32_t xL1KpOff = k1Start * FRACTAL_BYTES;
            const uint32_t wL1KpOff = kpInChunk * kbBytesPerKp;

            // L0A + L0B spatial ping-pong by kp parity. Kb is halved so per-kp
            // L0A and L0B both fit a 32 KB half; EVENT_ID0/1 keep the two
            // halves' M<->MTE1 chains independent, so kp+1's LoadData overlaps
            // kp's Mmad.
            const uint32_t kpPar = kp & 1u;
            const event_t l0Evt = kpPar ? EVENT_ID1 : EVENT_ID0;
            const uint32_t l0aOff = kpPar ? L0A_HALF_BYTES : 0u;
            const uint32_t l0bOff = kpPar ? L0B_HALF_BYTES : 0u;

            WaitFlag<HardEvent::M_MTE1>(l0Evt); // prime in PrimeEvents; tail SetFlag below

            for (uint16_t t = 0; t < mTurns; ++t) {
                LoadData(l0a[l0aOff + t * k1Count * FRACTAL_BYTES], xL1_[t * xL1TurnStride + xL1KpOff],
                         {0, static_cast<uint8_t>(k1Count), 1, 0, 0, false, 0});
            }
            LoadData(l0b[l0bOff], curWL1[wL1KpOff], {0, static_cast<uint8_t>(k1Count * nFracs), 1, 0, 0, false, 0});
            SetFlag<HardEvent::MTE1_M>(l0Evt);
            WaitFlag<HardEvent::MTE1_M>(l0Evt);

            if (kp == 0) {
                WaitFlag<HardEvent::V_M>(EVENT_ID0); // L0C free
            }
            PipeBarrier<PIPE_M>(); // serialize back-to-back Mmads on shared L0C
            Mmad(l0c, l0a[l0aOff], l0b[l0bOff],
                 {static_cast<uint16_t>(mAligned), static_cast<uint16_t>(nSize), static_cast<uint16_t>(kThisBytes), 0,
                  false, (kp == 0)});

            SetFlag<HardEvent::M_MTE1>(l0Evt);
        }

        // End-of-chunk signals the current buffer free; pairs with the
        // next MTE2's WaitFlag<MTE1_MTE2>(parity) when this slot is reused.
        SetFlag<HardEvent::MTE1_MTE2>((wBufIdx_ == 0u) ? EVENT_ID0 : EVENT_ID1);

        // Post-loop fallback fires only when the pre-loop did not
        // (wChunkKPasses_ == 1): covers next-chunk-in-this-cb and next-cb.
        if (!nxFired) {
            const uint32_t nxBuf = 1u - wBufIdx_;
            LocalTensor<int8_t> nxWL1 = (nxBuf == 0u) ? wL1PingLT_ : wL1PongLT_;
            if (chunkIdx + 1 < numChunks_) {
                const uint32_t k1NxStart = (chunkIdx + 1) * k1PerChunk_;
                const uint32_t k1NxRemain = K1_total - k1NxStart;
                const uint32_t k1NxToLoad = (k1PerChunk_ < k1NxRemain) ? k1PerChunk_ : k1NxRemain;
                WaitFlag<HardEvent::MTE1_MTE2>((nxBuf == 0u) ? EVENT_ID0 : EVENT_ID1);
                PrefetchWeightToL1<BlockedZN_>(nxWL1, weightGm_, expertBaseOff,
                                               /* colBatch= */ cbIdx,
                                               /* nSize= */ nSize,
                                               /* baseNG= */ nStart / N_FRACTAL,
                                               /* N1= */ fracN_,
                                               /* K1q= */ k1NxToLoad,
                                               /* K= */ K_,
                                               /* weightBlockN1= */ weightBlockN1_,
                                               /* k1Start= */ k1NxStart);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
                nxFired = true;
            } else if (cbIdx + 1 < numCbs) {
                const uint32_t nxCb = cbIdx + 1;
                const uint32_t nxNStart = nxCb * Nb_;
                const uint32_t nxColCount = (N_ - nxNStart < Nb_) ? (N_ - nxNStart) : Nb_;
                WaitFlag<HardEvent::MTE1_MTE2>((nxBuf == 0u) ? EVENT_ID0 : EVENT_ID1);
                PrefetchWeightToL1<BlockedZN_>(nxWL1, weightGm_, expertBaseOff,
                                               /* colBatch= */ nxCb,
                                               /* nSize= */ nxColCount,
                                               /* baseNG= */ nxNStart / N_FRACTAL,
                                               /* N1= */ fracN_,
                                               /* K1q= */ kp0K1Count,
                                               /* K= */ K_,
                                               /* weightBlockN1= */ weightBlockN1_,
                                               /* k1Start= */ 0);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
                nxFired = true;
            }
        }

        // Advance wBufIdx_ when a next MTE2 was fired. On the last chunk
        // of the last cb leave it; DrainEvents balances the residual.
        if (nxFired) {
            wBufIdx_ = 1u - wBufIdx_;
            curWL1 = (wBufIdx_ == 0u) ? wL1PingLT_ : wL1PongLT_;
        }
    }

    SetFlag<HardEvent::M_V>(EVENT_ID3);
    // xL1_ is shared across all cbs of the mTile; the "MTE1 done reading
    // xL1" signal is produced once after the last cb (see ProcessGroup).

    // ---- V: VDEQ16 L0C->UB (int32 * uint64 -> fp16 NZ) ----
    WaitFlag<HardEvent::M_V>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);

    DataCopyEnhancedParams vdeqP;
    vdeqP.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    vdeqP.deqScale = DeqScale::VDEQ16;
    vdeqP.deqTensorAddr = reinterpret_cast<uint64_t>(curScU64.GetPhyAddr());
    DataCopy(dequantScratchLT_, l0c, {static_cast<uint16_t>(nFracs), mTurns, 0, 0}, vdeqP);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2); // scale UB free
    SetFlag<HardEvent::V_M>(EVENT_ID0);    // L0C free

    PipeBarrier<PIPE_V>();

    // NZ->ND scatter via per-colGroup strided Muls(1.0h). Rewrites
    // NZ [ngCount, mAligned, 16] into ND row-major [mCount, nSize].
    // Per ng: mCount repeats of 16 halves; dst advances by nSize/16 blocks
    // per repeat (one ND row), src advances by 1 block (next NZ row).
    // Muls repeatTimes is uint8_t (max 255); chunk the scatter so each
    // repeat count stays <= 255 (mCount=256 would wrap to 0). At most two
    // sub-calls fire for MB_CANDIDATES (max 256).
    constexpr uint32_t MAX_VEC_REPEATS = 255U;
    const half scatterOne = static_cast<half>(1.0f);
    const uint8_t scatterDstRS = static_cast<uint8_t>(nSize / N_FRACTAL);
    const uint32_t srcNZStride = mAligned * N_FRACTAL;
    for (uint32_t ng = 0; ng < nFracs; ++ng) {
        uint32_t mRemaining = static_cast<uint32_t>(mCount);
        uint32_t mDone = 0U;
        while (mRemaining > 0U) {
            const uint8_t thisReps = (mRemaining > MAX_VEC_REPEATS) ? static_cast<uint8_t>(MAX_VEC_REPEATS) :
                                                                      static_cast<uint8_t>(mRemaining);
            Muls<half>(scatterDst[ng * N_FRACTAL + mDone * nSize],
                       dequantScratchLT_[ng * srcNZStride + mDone * N_FRACTAL], scatterOne, N_FRACTAL, thisReps,
                       {1, 1, scatterDstRS, 1});
            mDone += thisReps;
            mRemaining -= thisReps;
        }
    }
    PipeBarrier<PIPE_V>();

    // finalOut[m,n] = scatterDst[m,n] * pertoken[m].
    // Per-row fp32 intermediate multiply: dequantScratchLT_ is free after the
    // NZ->ND scatter above (VDEQ16 NZ data no longer needed). Reinterpret it
    // as float scratch for one row's Cast<float,half> + Muls<float> +
    // Cast<half,float> chain. This preserves fp32 precision for the pertoken
    // multiply, matching the pre-optimization reference kernel which applies
    // weight_scale and pertoken both in fp32 before the final half truncation.
    // The VDEQ16 int32*fp16_scale->fp16 step still quantizes to fp16, but the
    // pertoken factor is now applied at fp32, recovering one layer of fp16
    // precision loss.
    if constexpr (HasPertoken) {
        LocalTensor<float> rowFp32 = dequantScratchLT_.template ReinterpretCast<float>();
        for (uint32_t m = 0; m < mCount; ++m) {
            const float pts_f = pertokenLT_.GetValue(static_cast<uint32_t>(m));
            Cast<float, half>(rowFp32, scatterDst[m * nSize], RoundMode::CAST_NONE, nSize);
            PipeBarrier<PIPE_V>();
            Muls<float>(rowFp32, rowFp32, pts_f, nSize);
            PipeBarrier<PIPE_V>();
            Cast<half, float>(finalOut[m * nSize], rowFp32, RoundMode::CAST_NONE, nSize);
            PipeBarrier<PIPE_V>();
        }
    }
    // Non-pertoken: scatterDst == finalOut, scatter already wrote the result.

    // ---- MTE3: UB -> GM ----
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const uint64_t yRowStride = static_cast<uint64_t>(N_);
    DataCopy(yGm_[mStart * yRowStride + nStart], finalOut,
             {static_cast<uint16_t>(mCount), static_cast<uint16_t>(nSize / N_FRACTAL), 0,
              static_cast<uint16_t>((N_ - nSize) / N_FRACTAL)});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
    (void)curColCount;
}

} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_PHASE_MM_H
