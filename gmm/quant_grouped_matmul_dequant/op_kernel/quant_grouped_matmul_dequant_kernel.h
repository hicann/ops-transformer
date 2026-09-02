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
 * \file quant_grouped_matmul_dequant_kernel.h
 * \brief Unified W8A8 MoE matmul+dequant kernel (Ascend 310P).
 *
 *   Y[m, n] = fp16( sum_q( sum_{k in K_q} x_q[m, k] * W[k, n] ) * scale[q, n] )
 *                   * pertoken[m]
 *   x_q = round(x * 127 / max_m), pertoken[m] = max_m / 127 (DynamicQuant).
 *
 *   Shapes:  x        [M, K]   half  ND   (quantized on-chip to int8)
 *            weight   [E, K, N] int8  ZN  (or blocked-ZN when N1 > 4080)
 *            scale    [E or 1, N] u64 (VDEQ16) or fp32 (encoded on-chip)
 *            group    [E]        int64 cumulative token counts
 *            y        [M, N]     half  ND
 *
 *   Tiling: M/N/K split into Mb/Nb/Kb tiles. Per-expert iters =
 *   ceil(tokens/Mb) aggregated into a global count and sliced across cores.
 *   cb loop walks N in Nb steps; Phase X dynamic-quantizes x; weight ping/pongs
 *   over (cb, ch). DynamicQuant scale stays in S-register (never UB<->GM).
 *
 *   Template parameters (compile-time dispatch):
 *     HasPertoken      -- pertoken quant mode (vs pertensor).
 *     HasSmoothScale   -- smooth_scale input applied in Phase X.
 *     ScaleIsInt64     -- weight_scale pre-encoded uint64 (skip V-pipe encode).
 *     BlockedZN        -- 6-D blocked-ZN weight storage.
 *     SmallM           -- Mb <= SMALL_M_THRESHOLD (collapses some vec work).
 *     UseXZNStage      -- staged PhaseX->GM->PhaseMM with cross-core barrier.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_KERNEL_H
#define QUANT_GROUPED_MATMUL_DEQUANT_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_grouped_matmul_dequant_config.h"
#include "quant_grouped_matmul_dequant_tiling_key.h"
#include "quant_grouped_matmul_dequant_utils.h"

namespace AscendC {

template <bool HasPertoken, bool HasSmoothScale, bool ScaleIsInt64, bool BlockedZN, bool SmallM,
          bool UseXZNStage // false: PhaseX writes xL1 (per-core, per-mTile); true: PhaseX writes xZNGm_, PhaseMM reads
                           // xZNGm_
          >
class QuantGroupedMatmulDequant {
public:
    __aicore__ inline QuantGroupedMatmulDequant() = default;

    // Bind GM tensors + tiling data. Must be called before Process().
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list,
                                GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale, GM_ADDR y,
                                GM_ADDR usr_workspace, const QuantMatmulDequantTilingData *__restrict tiling,
                                TPipe *pipe);

    // Main compute loop - iter-level distribution over experts.
    __aicore__ inline void Process();

private:
    // Phase-level helpers (bodies in _init.h and phase_*.h).
    __aicore__ inline void InitUbLayout();
    __aicore__ inline void InitL1Layout();
    __aicore__ inline void InitL0Layout();
    __aicore__ inline void PrimeEvents();
    __aicore__ inline void DrainEvents();
    __aicore__ inline void InitSyncWs();     // zero-init syncGmWs_ once at kernel start (UseXZNStage_ only)
    __aicore__ inline void MySyncAllSwift(); // cross-core barrier between PhaseX-out and PhaseMM-in
    __aicore__ inline void ProcessGroup(uint32_t groupIdx, uint64_t mStart, uint64_t mCount, uint32_t nCoreIdx,
                                        uint32_t nCoreNum);
    __aicore__ inline void PhaseXDynamicQuant(uint64_t mStart, uint64_t mCount);
    __aicore__ inline void PhaseXWriteXL1(uint64_t mCount);
    __aicore__ inline void ProcessColBatch(uint32_t groupIdx, uint32_t cbIdx, uint64_t mStart, uint64_t mCount,
                                           uint64_t curColCount, uint32_t numCbs, uint32_t cbStart);
    __aicore__ inline void LoadScaleToUB(uint32_t groupIdx, uint32_t cbIdx, uint64_t curColCount, bool usePing);
    __aicore__ inline void DequantAndScatter(uint64_t mCount, uint64_t curColCount, bool usePing);

    // -------------------------------------------------------------
    // Compile-time constants (derived from template params).
    // -------------------------------------------------------------
    static constexpr uint32_t UB_BYTES = UB_TOTAL_BYTES;
    static constexpr uint32_t L1_BYTES = L1_TOTAL_BYTES;

    // -------------------------------------------------------------
    // GM tensors (bound in Init()).
    // -------------------------------------------------------------
    GlobalTensor<half> xGm_;
    GlobalTensor<int8_t> weightGm_;
    GlobalTensor<float> weightScaleFp32Gm_;
    GlobalTensor<uint64_t> weightScaleU64Gm_;
    GlobalTensor<int64_t> groupListGm_;
    GlobalTensor<float> xScaleFp32Gm_;
    GlobalTensor<half> xScaleHalfGm_;
    GlobalTensor<half> smoothScaleGm_;
    GlobalTensor<half> yGm_;
    // Workspace base pointer. xZNGm_ / syncGmWs_ GlobalTensors are derived
    // on-demand inside the staged-PhaseX/PhaseMM code paths - keeping them
    // as class members exceeds the 16000-byte aicore stack frame limit.
    __gm__ uint8_t *usrWs_ = nullptr;

    TPipe *pipe_ = nullptr;
    const QuantMatmulDequantTilingData *__restrict tiling_ = nullptr;

    // -------------------------------------------------------------
    // UB arena - single TBuf carved into persistent + aliased slots.
    // Offsets computed in InitUbLayout(); handles below are views.
    // -------------------------------------------------------------
    TBuf<TPosition::VECCALC> ubArena_;
    // Persistent slots (live across Phase X <-> Phase D).
    LocalTensor<half> outPingLT_;
    LocalTensor<half> outPongLT_;
    LocalTensor<float> pertokenLT_;
    LocalTensor<uint64_t> scaleU64PingLT_;
    LocalTensor<uint64_t> scaleU64PongLT_;
    LocalTensor<uint32_t> scatterIdxLT_;
    LocalTensor<half> smoothScaleLT_;
    LocalTensor<float> smoothFp32LT_; // per-chunk fp32 cast of smooth_scale
    // Aliased Phase X workspace (int8 scratch + raw ping/pong).
    LocalTensor<half> xRawPingLT_;
    LocalTensor<half> xRawPongLT_;
    LocalTensor<int8_t> int8ScratchLT_;
    // Phase X per-row scratch (single-rounding chain scratch).
    LocalTensor<half> absLT_;
    LocalTensor<half> reduceWorkLT_;
    LocalTensor<half> maxLT_;
    LocalTensor<float> xCastFloatLT_;
    LocalTensor<float> xScaledLT_;
    LocalTensor<int32_t> xInt32LT_;
    LocalTensor<half> xHalfLT_;
    // Aliased Phase D workspace (reuses Phase X region after drain). Single
    // dequantScratch NZ buffer; the cb-alternating copyOut reuses outPing/Pong.
    LocalTensor<half> dequantScratchLT_;
    LocalTensor<float> scaleFp32StagingLT_;

    // Persistent pertoken row-broadcast: filled once per mTile (Duplicate of
    // pertokenLT_ across Nb cols), read by bulk Mul<half> per cb.
    LocalTensor<half> pertokenBcastLT_;

    // -------------------------------------------------------------
    // L1 buffers. xL1 single-buffered across the mTile; weight ping/pong
    // across (cb, ch); optional xRawL1 ping/pong for cross-iter pre-load.
    // -------------------------------------------------------------
    TBuf<TPosition::A1> xL1Buf_;
    TBuf<TPosition::A1> wL1PingBuf_;
    TBuf<TPosition::A1> wL1PongBuf_;
    TBuf<TPosition::A1> xRawL1PingBuf_;
    TBuf<TPosition::A1> xRawL1PongBuf_;
    // L1 handles (bound once in InitL1Layout; ScatterZZToL1 / MTE1 read these).
    LocalTensor<int8_t> xL1_;
    LocalTensor<int8_t> wL1PingLT_;
    LocalTensor<int8_t> wL1PongLT_;
    LocalTensor<half> xRawL1_; // within-mTile raw-X cache

    // L0 buffers (fixed 310P sizes - L0A/L0B 64 KB, L0C 256 KB).
    TBuf<TPosition::A2> l0aBuf_;
    TBuf<TPosition::B2> l0bBuf_;
    TBuf<QuePosition::CO1> l0cBuf_;

    // -------------------------------------------------------------
    // Cached tiling values (to avoid repeated indirections in hot loops).
    // -------------------------------------------------------------
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t MCoreNum_ = 1; // M-axis core count
    uint32_t NCoreNum_ = 1; // N-axis core count (cb-axis split)
    // wL1 buffer index toggles with each weight MTE2 (chunk or cb-cross
    // prefetch) so a prefetch can use the opposite buffer; persists across cbs.
    uint32_t wBufIdx_ = 0u;
    uint32_t M_ = 0;
    uint32_t K_ = 0;
    uint32_t N_ = 0;
    uint32_t E_ = 0;
    uint32_t Mb_ = 0;
    uint32_t Nb_ = 0;
    uint32_t Kb_ = 0;
    uint32_t fracN_ = 0;
    uint32_t fracK_ = 0;
    uint32_t weightBlockN1_ = 0;
    uint32_t perCoreIters_ = 0;
    uint32_t remainderIters_ = 0;
    uint32_t nBlockSize_ = 1; // cbs that fit L2 simultaneously
    uint64_t ubKMask_ = 0;
    bool hasXRawL1_ = false;
    bool useL2Hint_ = false;
    bool l2WeightFits_ = false;
    bool dbL0c_ = false;        // L0C ping-pong across cbs
    uint32_t l0cHalfElems_ = 0; // = Mb*Nb (int32 elems per L0C half when dbL0c_)
};

// ============================================================
// Process - iter-level work distribution across all experts.
// Aggregates ceil(tokens_per_expert / Mb) into a global iter count, then
// slices it across cores. Each core may process iters spanning multiple
// experts; a core never idles on sparse-MoE as long as it has >=1 iter.
// ============================================================
template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void
QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_, UseXZNStage_>::Process()
{
    PrimeEvents();

    // First pass: total iters across all experts (cumulative group_list).
    // prevCum is reused by the second pass; declared once at function scope.
    uint64_t totalIters = 0;
    int64_t prevCum = 0;
    int64_t endM;
    int64_t tokens;
    for (uint32_t g = 0; g < E_; ++g) {
        endM = groupListGm_.GetValue(g);
        tokens = endM - prevCum;
        prevCum = endM;
        if (tokens > 0) {
            totalIters += (static_cast<uint64_t>(tokens) + Mb_ - 1) / Mb_;
        }
    }

    // Cooperative-batch: cores split into M-groups (MCoreNum_) sharing an
    // mTile slice; cores within a group partition the cb range by nCoreIdx.
    // NCoreNum_=1 sends all cores over ALL cbs of their mTile so each cb's
    // weight is fetched by one core's MTE2 and the rest hit L2. Three modes:
    //   MN     : MCoreNum>=2 && NCoreNum>=2 - split iters across M-groups,
    //            share cb range within each group.
    //   N_ONLY : MCoreNum==1 && NCoreNum>=2 - all cores share single mTile,
    //            partition its cbs (avoids 1-core run when totalIters=1).
    //   CONTIGUOUS: otherwise - contiguous-iter slice across coreNum cores.
    enum class WorkMode {
        CONTIGUOUS,
        COOPBATCH_MN,
        COOPBATCH_N_ONLY
    };
    WorkMode mode;
    // N_ONLY only when cb-per-core >= threshold: below it, PhaseX re-quantizes
    // all rows of the mTile on every core, outweighing the cb-split parallelism.
    constexpr uint32_t COOP_N_ONLY_MIN_CBS_PER_CORE = 16u;
    const uint32_t totalColBatchProcess = (N_ + Nb_ - 1) / Nb_;
    const bool nOnlyWorthwhile =
        (NCoreNum_ > 0u) && ((totalColBatchProcess / NCoreNum_) >= COOP_N_ONLY_MIN_CBS_PER_CORE);
    if (MCoreNum_ >= 2u && NCoreNum_ >= 2u)
        mode = WorkMode::COOPBATCH_MN;
    else if (MCoreNum_ == 1u && NCoreNum_ >= 2u && nOnlyWorthwhile)
        mode = WorkMode::COOPBATCH_N_ONLY;
    else
        mode = WorkMode::CONTIGUOUS;
    const bool coopBatch = (mode != WorkMode::CONTIGUOUS);
    const uint32_t mCoreIdx = (mode == WorkMode::COOPBATCH_MN)     ? (coreIdx_ / NCoreNum_) :
                              (mode == WorkMode::COOPBATCH_N_ONLY) ? 0u :
                                                                     coreIdx_;
    const uint32_t nCoreIdx = (mode == WorkMode::COOPBATCH_MN)     ? (coreIdx_ % NCoreNum_) :
                              (mode == WorkMode::COOPBATCH_N_ONLY) ? coreIdx_ :
                                                                     0u;
    const uint32_t splitDen = (mode == WorkMode::COOPBATCH_MN)     ? MCoreNum_ :
                              (mode == WorkMode::COOPBATCH_N_ONLY) ? 1u :
                                                                     coreNum_;
    const uint64_t perCore = totalIters / splitDen;
    const uint64_t rem = totalIters % splitDen;
    const uint64_t myIters = perCore + (mCoreIdx < rem ? 1 : 0);
    const uint64_t startIter = perCore * mCoreIdx + (mCoreIdx < rem ? mCoreIdx : rem);
    const uint64_t endIter = startIter + myIters;

    // Second pass: walk experts and pick out iters in [startIter, endIter).
    // globalIterIdx is the cumulative iter counter across experts.
    if (myIters > 0) {
        prevCum = 0;
        uint64_t passedTokens = 0;
        uint64_t globalIterIdx = 0;
        for (uint32_t g = 0; g < E_; ++g) {
            endM = groupListGm_.GetValue(g);
            tokens = endM - prevCum;
            prevCum = endM;
            if (tokens <= 0)
                continue;
            const uint64_t expertTokens = static_cast<uint64_t>(tokens);
            const uint64_t itersThis = (expertTokens + Mb_ - 1) / Mb_;

            // Fast-skip experts entirely outside our slice.
            if (globalIterIdx + itersThis <= startIter) {
                globalIterIdx += itersThis;
                passedTokens += expertTokens;
                continue;
            }
            if (globalIterIdx >= endIter)
                break;

            // First iter in this expert that belongs to us.
            const uint64_t localStart = (startIter > globalIterIdx) ? (startIter - globalIterIdx) : 0;
            // Last iter in this expert that belongs to us (exclusive).
            const uint64_t localEnd = (endIter - globalIterIdx < itersThis) ? (endIter - globalIterIdx) : itersThis;

            for (uint64_t it = localStart; it < localEnd; ++it) {
                const uint64_t mOff = it * Mb_;
                const uint64_t mStart = passedTokens + mOff;
                const uint64_t mTail = expertTokens - mOff;
                const uint64_t mCount = (mTail < Mb_) ? mTail : Mb_;
                // nCoreNum passed to ProcessGroup depends on mode:
                //   MN     -> NCoreNum_ (cores in an M-group partition cbs)
                //   N_ONLY -> coreNum_  (all cores partition cbs of the mTile)
                //   CONTIG -> 1         (each core walks ALL cbs of its iters)
                const uint32_t pgNCoreNum = (mode == WorkMode::COOPBATCH_MN)     ? NCoreNum_ :
                                            (mode == WorkMode::COOPBATCH_N_ONLY) ? coreNum_ :
                                                                                   1u;
                ProcessGroup(g, mStart, mCount, nCoreIdx, pgNCoreNum);
            }

            globalIterIdx += itersThis;
            passedTokens += expertTokens;
        }
    }

    DrainEvents();
}

} // namespace AscendC

// Member function definitions for Phase X / MM / D and Init helpers live in
// chunked headers. Include them after the class declaration so out-of-class
// template member definitions are visible at instantiation time.
#include "quant_grouped_matmul_dequant_init.h"
#include "quant_grouped_matmul_dequant_phase_x.h"
#include "quant_grouped_matmul_dequant_phase_mm.h"
#include "quant_grouped_matmul_dequant_phase_d.h"

#endif // QUANT_GROUPED_MATMUL_DEQUANT_H
