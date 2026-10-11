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
 * \file quant_grouped_matmul_dequant_init.h
 * \brief Out-of-class definitions for Init / Init{Ub,L0,L1}Layout / Prime/DrainEvents.
 *
 * Included from quant_grouped_matmul_dequant.h AFTER the class template
 * declaration. Reopens namespace AscendC for the member function bodies.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_INIT_H
#define QUANT_GROUPED_MATMUL_DEQUANT_INIT_H

namespace AscendC {

// ============================================================
// Init - bind GM + tiling, carve UB + L1 layouts.
// ============================================================
template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void
QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_, UseXZNStage_>::Init(
    GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list, GM_ADDR bias, GM_ADDR x_scale,
    GM_ADDR x_offset, GM_ADDR smooth_scale, GM_ADDR y, GM_ADDR usr_workspace,
    const QuantMatmulDequantTilingData* __restrict tiling, TPipe* pipe)
{
    (void)bias;
    (void)x_offset;
    if constexpr (!UseXZNStage_) {
        (void)usr_workspace;
    }
    pipe_ = pipe;
    tiling_ = tiling;

    coreIdx_ = GetBlockIdx();
    coreNum_ = tiling->coreNum;
    MCoreNum_ = tiling->MCoreNum;
    NCoreNum_ = tiling->NCoreNum;
    if (MCoreNum_ == 0u)
        MCoreNum_ = 1u;
    if (NCoreNum_ == 0u)
        NCoreNum_ = 1u;
    M_ = tiling->M;
    K_ = tiling->K;
    N_ = tiling->N;
    E_ = tiling->E;
    Mb_ = tiling->Mb;
    Nb_ = tiling->Nb;
    Kb_ = tiling->Kb;
    fracN_ = tiling->fracN;
    fracK_ = tiling->fracK;
    weightBlockN1_ = tiling->weightBlockN1;
    perCoreIters_ = tiling->perCoreIters;
    remainderIters_ = tiling->remainderIters;
    nBlockSize_ = tiling->nBlockSize;
    if (nBlockSize_ == 0)
        nBlockSize_ = 1; // safety fallback
    ubKMask_ = tiling->ubKMask;
    hasXRawL1_ = GetFlag(tiling->flags, FLAG_BIT_HAS_X_RAW_L1);
    useL2Hint_ = GetFlag(tiling->flags, FLAG_BIT_USE_L2_HINT);
    l2WeightFits_ = GetFlag(tiling->flags, FLAG_BIT_L2_WEIGHT_FITS);
    dbL0c_ = GetFlag(tiling->flags, FLAG_BIT_DBL0C);
    // L0C half holds Mb*Nb int32 elements (used as the offset for ping-pong).
    l0cHalfElems_ = dbL0c_ ? static_cast<uint32_t>(Mb_) * Nb_ : 0u;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(quantized_weight));
    if constexpr (ScaleIsU64) {
        weightScaleU64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(weight_scale));
    } else {
        weightScaleFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight_scale));
    }
    groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));
    if constexpr (!HasPertoken) {
        // Static-quant path (x_scale provided by caller).
        xScaleFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x_scale));
        xScaleHalfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x_scale));
    } else {
        (void)x_scale;
    }
    if constexpr (HasSmooth) {
        smoothScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(smooth_scale));
    } else {
        (void)smooth_scale;
    }
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
    if constexpr (UseXZNStage_) {
        usrWs_ = usr_workspace;
    }

    // SetL2CacheHint on every GM tensor. Weight is int8 ZN, padded to
    // [K1, N1, 16, 32] per expert. PERSISTENT when all experts fit L2
    // together; else NORMAL (LRU). PERSISTENT thrashes if total > L2 budget.
    {
        const uint64_t totalIters = (static_cast<uint64_t>(M_) + tiling->Mb - 1) / tiling->Mb;
        // For grouped (E>1) the kernel walks every expert's padded ZN slab,
        // so total L2 footprint is the SUM across active experts.
        const uint64_t weightBytesPerExpert =
            static_cast<uint64_t>(fracK_) * static_cast<uint64_t>(fracN_) * M_FRACTAL * K_FRACTAL_INT8_BYTES;
        const uint64_t numExperts = (E_ > 0) ? static_cast<uint64_t>(E_) : 1ULL;
        const uint64_t weightBytesTotal = weightBytesPerExpert * numExperts;
        const CacheMode l2Mode = (weightBytesTotal <= AscendC::L2_HEADROOM_BYTES) ? CacheMode::CACHE_MODE_PERSISTENT :
                                                                                    CacheMode::CACHE_MODE_NORMAL;
        xGm_.SetL2CacheHint(l2Mode);
        weightGm_.SetL2CacheHint(l2Mode);
        yGm_.SetL2CacheHint(l2Mode);
        if constexpr (ScaleIsU64) {
            weightScaleU64Gm_.SetL2CacheHint(l2Mode);
        } else {
            weightScaleFp32Gm_.SetL2CacheHint(l2Mode);
        }
        if constexpr (HasSmooth) {
            smoothScaleGm_.SetL2CacheHint(l2Mode);
        }
    }

    InitUbLayout();
    InitL1Layout();
    InitL0Layout();
    if constexpr (UseXZNStage_) {
        InitSyncWs();
    }
}

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void
QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_, UseXZNStage_>::InitUbLayout()
{
    pipe_->InitBuffer(ubArena_, UB_BYTES);
    LocalTensor<uint8_t> arena = ubArena_.Get<uint8_t>();

    const uint32_t Mb = Mb_;
    const uint32_t Nb = Nb_;
    const uint32_t kBytes = tiling_->originKAligned32;
    const uint32_t kHalves = kBytes; // half elements per K-row (bytes == halves here since int8 view uses 1B; fp16 rows
                                     // use K_halves = originKAligned32 when K_halves padded to 32).
    // VDEQ16 / Mmad write mAligned rows (>= M_FRACTAL) into L0C; Phase D
    // buffers (dequantScratch, copyOut, pertokenBroadcast) sized for this
    // bound, not Mb alone. Mb < M_FRACTAL still issues a 16-row padded Mmad.
    const uint32_t MbAligned = (Mb < M_FRACTAL) ? M_FRACTAL : Mb;

    // Persistent slots (live across Phase X <-> Phase D).
    uint32_t off = 0;

    // SyncAll UB scratch - int32_t[coreNum * 8] = 32 bytes per core. Lives
    // at the bottom of the arena so it never aliases any Phase X / Phase D
    // buffer. Reinterpreted from arena offset 0 on-demand in InitSyncWs /
    // MySyncAllSwift (NOT stored as a class member - keeps the kernel
    // stack frame below the 16000-byte aicore limit).
    if constexpr (UseXZNStage_) {
        // Reserve the slot only; LocalTensor is re-derived where used.
        off += coreNum_ * 32u + UB_BANK_PAD; // 32 bytes per core
    }

    pertokenLT_ = arena[off].ReinterpretCast<float>();
    off += (((Mb * sizeof(float)) + 31) / 32) * 32 + UB_BANK_PAD;

    outPingLT_ = arena[off].ReinterpretCast<half>();
    off += Mb * Nb * sizeof(half) + UB_BANK_PAD;
    outPongLT_ = arena[off].ReinterpretCast<half>();
    off += Mb * Nb * sizeof(half) + UB_BANK_PAD;

    // Persistent pertoken broadcast buffer (Mb*Nb*2 bytes). Gated on E==1
    // (dense); for E>1 the extra UB cost forces a smaller Mb in the picker.
    if constexpr (HasPertoken) {
        if (E_ == 1u) {
            pertokenBcastLT_ = arena[off].ReinterpretCast<half>();
            off += Mb * Nb * sizeof(half) + UB_BANK_PAD;
        }
    }

    scaleU64PingLT_ = arena[off].ReinterpretCast<uint64_t>();
    off += Nb * sizeof(uint64_t) + UB_BANK_PAD;
    scaleU64PongLT_ = arena[off].ReinterpretCast<uint64_t>();
    off += Nb * sizeof(uint64_t) + UB_BANK_PAD;

    if constexpr (!ScaleIsU64) {
        scatterIdxLT_ = arena[off].ReinterpretCast<uint32_t>();
        off += Nb * sizeof(uint32_t) + UB_BANK_PAD;
        // Seed scatter table idx[i] = i * 8 (byte offset per uint64 slot).
        // Persistent across kernel instance; EncodeFp32ScaleToUint64 reads it.
        LocalTensor<int32_t> idxI32 = scatterIdxLT_.ReinterpretCast<int32_t>();
        CreateVecIndex<int32_t>(idxI32, static_cast<int32_t>(0), Nb);
        Muls<int32_t>(idxI32, idxI32, static_cast<int32_t>(8), Nb);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (HasSmooth) {
        smoothScaleLT_ = arena[off].ReinterpretCast<half>();
        off += K_ * sizeof(half) + UB_BANK_PAD;
        // Per-chunk fp32 cast of smooth_scale (phaseXChunk halves). Phase X
        // widens fp16 smooth -> fp32 so the Mul runs fp32*fp32 -> fp32: exact
        // at fp16 input precision (fp16 multiply accrues ~2^-11 rel err/chunk).
        smoothFp32LT_ = arena[off].ReinterpretCast<float>();
        off += tiling_->phaseXChunk * sizeof(float) + UB_BANK_PAD;
    }

    // Phase X workspace - aliased with the Phase D region. Per-chunk scratch
    // buffers sized by phaseXChunk (not full kBytes); int8Scratch holds the
    // full quantized row.
    const uint32_t phaseXStart = off;
    const uint32_t chunkHalves = tiling_->phaseXChunk;

    xRawPingLT_ = arena[off].ReinterpretCast<half>();
    off += chunkHalves * sizeof(half) + UB_BANK_PAD;
    xRawPongLT_ = arena[off].ReinterpretCast<half>();
    off += chunkHalves * sizeof(half) + UB_BANK_PAD;

    absLT_ = arena[off].ReinterpretCast<half>();
    off += chunkHalves * sizeof(half) + UB_BANK_PAD;
    reduceWorkLT_ = arena[off].ReinterpretCast<half>();
    off += 256 + UB_BANK_PAD;
    maxLT_ = arena[off].ReinterpretCast<half>();
    off += 32 + UB_BANK_PAD;

    xCastFloatLT_ = arena[off].ReinterpretCast<float>();
    off += chunkHalves * sizeof(float) + UB_BANK_PAD;
    xScaledLT_ = arena[off].ReinterpretCast<float>();
    off += chunkHalves * sizeof(float) + UB_BANK_PAD;
    xInt32LT_ = arena[off].ReinterpretCast<int32_t>();
    off += chunkHalves * sizeof(int32_t) + UB_BANK_PAD;
    xHalfLT_ = arena[off].ReinterpretCast<half>();
    off += chunkHalves * sizeof(half) + UB_BANK_PAD;

    // int8Scratch sized MbAligned*chunkHalves: Pass 2 scatters each chunk to
    // xL1 right after quantize, freeing it for the next chunk. Avoids an Mb*K
    // UB hog that would force Mb=16 on K>=4096 shapes.
    int8ScratchLT_ = arena[off].ReinterpretCast<int8_t>();
    off += MbAligned * chunkHalves + UB_BANK_PAD;

    (void)kHalves; // legacy reference; chunkHalves is the sizing axis now.
    (void)kBytes;  // legacy; int8Scratch sized by chunkHalves now.

    // Phase D aliases the Phase X region: single dequantScratch (NZ output of
    // VDEQ16). The cb-alternating output uses persistent outPing/Pong.
    uint32_t dOff = phaseXStart;
    dequantScratchLT_ = arena[dOff].ReinterpretCast<half>();
    dOff += MbAligned * Nb * sizeof(half) + UB_BANK_PAD;
    if constexpr (!ScaleIsU64) {
        scaleFp32StagingLT_ = arena[dOff].ReinterpretCast<float>();
        dOff += Nb * sizeof(float) + UB_BANK_PAD;
    }

    (void)phaseXStart;
    // ASCEND_ASSERT(max(off, dOff) <= UB_BYTES);  // budget enforced host-side.
}

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void
QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_, UseXZNStage_>::InitL0Layout()
{
    pipe_->InitBuffer(l0aBuf_, L0A_TOTAL_BYTES);
    pipe_->InitBuffer(l0bBuf_, L0B_TOTAL_BYTES);
    pipe_->InitBuffer(l0cBuf_, L0C_TOTAL_BYTES);
}

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void
QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_, UseXZNStage_>::InitL1Layout()
{
    // xL1 holds one mTile in ZZ fractal: mAligned rows * originKAligned32
    // bytes. SmallM (Mb < M_FRACTAL) still needs a full 16-row fractal turn
    // because ZZ packs rows across fractals; sizing by Mb alone overruns L1
    // on the second half of the K-fractals. Weight wL1 ping/pongs across
    // (cb, ch), sized to one (Kb, Nb) int8 tile.
    const uint32_t MbAligned = (Mb_ < static_cast<uint32_t>(M_FRACTAL)) ? static_cast<uint32_t>(M_FRACTAL) : Mb_;
    const uint32_t xL1Bytes = MbAligned * tiling_->originKAligned32;
    pipe_->InitBuffer(xL1Buf_, xL1Bytes);
    // Each ping/pong half holds wChunkKPasses * Kb * Nb bytes so one MTE2
    // burst loads multiple K-passes; falls back to single-kp (Kb*Nb) when 1.
    const uint32_t wL1HalfBytes = tiling_->wChunkKPasses * Kb_ * Nb_;
    pipe_->InitBuffer(wL1PingBuf_, wL1HalfBytes);
    pipe_->InitBuffer(wL1PongBuf_, wL1HalfBytes);

    xL1_ = xL1Buf_.Get<int8_t>();
    wL1PingLT_ = wL1PingBuf_.Get<int8_t>();
    wL1PongLT_ = wL1PongBuf_.Get<int8_t>();

    // xRawL1 within-mTile raw-X cache, allocated only when host tiling
    // verified it fits alongside xL1 + 2*wL1. Gated by hasXRawL1_.
    if (hasXRawL1_) {
        pipe_->InitBuffer(xRawL1PingBuf_, MbAligned * tiling_->originKAligned32 * static_cast<uint32_t>(sizeof(half)));
        xRawL1_ = xRawL1PingBuf_.Get<half>();
    }
}
template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void
QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_, UseXZNStage_>::PrimeEvents()
{
    // Seed every cross-iter / cross-cb ping/pong token so the first
    // consumer's WaitFlag<> finds it set.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID6);    // outPing/Pong free
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);    // int8Scratch free
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);    // scale UB free
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0); // wL1Ping free
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1); // wL1Pong free
    SetFlag<HardEvent::V_M>(EVENT_ID0);       // L0C free
    SetFlag<HardEvent::MTE1_MTE3>(EVENT_ID0); // xL1 free (first iter)
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5); // Phase D UB free before next-iter PhaseX MTE2 (cross-iter aliased-UB race)
    // L0A/L0B ping-pong tokens: ID0 = half 0, ID1 = half 1. First kp's
    // WaitFlag<M_MTE1> consumes these; the kp loop tail sustains the chain.
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
}

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void
QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_, UseXZNStage_>::DrainEvents()
{
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    // Drain the L0A/L0B ping-pong tokens (primed in PrimeEvents).
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
}

// InitSyncWs: zero-fill the syncGmWs_ flag region once per kernel launch.
// syncUbWs_ is a small UB scratch reinterpreted from a slot in ubArena_.
// Ordering: V duplicate -> MTE3 write -> next stage's MTE2.
template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void
QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_, UseXZNStage_>::InitSyncWs()
{
    constexpr uint64_t SYS_WS_BYTES = 2ULL * 1024 * 1024;
    const uint64_t alignedM = ((static_cast<uint64_t>(M_) + M_FRACTAL - 1) / M_FRACTAL) * M_FRACTAL;
    const uint64_t xZNBytes = alignedM * tiling_->originKAligned32;
    const uint32_t syncCount = coreNum_ * 8u; // int32 elements (= 32 B per core)
    GlobalTensor<int32_t> syncGm;
    syncGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(usrWs_ + SYS_WS_BYTES + xZNBytes), syncCount);
    LocalTensor<int32_t> syncUb = ubArena_.Get<uint8_t>().ReinterpretCast<int32_t>();
    Duplicate<int32_t>(syncUb, 0, syncCount);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
    DataCopy<int32_t>(syncGm, syncUb, syncCount);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    PipeBarrier<PIPE_ALL>();
}

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void
QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_, UseXZNStage_>::MySyncAllSwift()
{
    // Cross-core barrier: each core increments its slot in syncGm and spins
    // until all coreNum_ slots are advanced.
    constexpr uint64_t SYS_WS_BYTES = 2ULL * 1024 * 1024;
    const uint64_t alignedM = ((static_cast<uint64_t>(M_) + M_FRACTAL - 1) / M_FRACTAL) * M_FRACTAL;
    const uint64_t xZNBytes = alignedM * tiling_->originKAligned32;
    GlobalTensor<int32_t> syncGm;
    syncGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(usrWs_ + SYS_WS_BYTES + xZNBytes),
                           static_cast<uint64_t>(coreNum_) * 8u);
    LocalTensor<int32_t> syncUb = ubArena_.Get<uint8_t>().ReinterpretCast<int32_t>();
    AscendC::SyncAll<false>(syncGm, syncUb, coreNum_);
}

} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_INIT_H
