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
 * \file quant_grouped_matmul_dequant_pertoken_wide_m.h
 * \brief Per-token dynamic grouped matmul dequant for tiling key 10000006.
 *
 * Phase X and Phase D share UB workspace. Weight prefetch uses two L1 slots,
 * with an optional third slot for cross-group prefetch.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_PERTOKEN_WIDE_M_H
#define QUANT_GROUPED_MATMUL_DEQUANT_PERTOKEN_WIDE_M_H

#include "quant_grouped_matmul_dequant_tiling_key.h"
#include "quant_grouped_matmul_dequant_kernel.h"

namespace AscendC {
template <bool HasSmooth, bool ScaleIsU64>
class QgmmdWideM {
public:
    __aicore__ inline QgmmdWideM() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list,
                                GM_ADDR bias, GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale, GM_ADDR y,
                                GM_ADDR usr_workspace, const QuantMatmulDequantTilingData* __restrict tiling,
                                TPipe* pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitUbLayout();
    __aicore__ inline void InitL1Layout();
    __aicore__ inline void InitL0Layout();
    __aicore__ inline void PrimeEvents();
    __aicore__ inline void DrainEvents();
    __aicore__ inline void ProcessGroup(uint32_t groupIdx, uint64_t mStart, uint64_t mCount, uint32_t nCoreIdx,
                                        uint32_t nCoreNum, uint32_t nextGroupIdx);
    __aicore__ inline void PhaseXDynamicQuant(uint64_t mStart, uint64_t mCount);
    __aicore__ inline void PhaseXWriteXL1(uint64_t mCount);
    __aicore__ inline void ProcessColBatch(uint32_t groupIdx, uint32_t cbIdx, uint64_t mStart, uint64_t mCount,
                                           uint64_t curColCount, uint32_t cbEnd, uint32_t cbStart);
    // Slot and event indices are 0/1, plus 2 when w3Enabled_ is set.
    __aicore__ inline LocalTensor<int8_t> WL1Slot(uint32_t idx);
    __aicore__ inline event_t WL1FreeEvt(uint32_t idx);
    __aicore__ inline uint32_t NextWBufIdx();
    __aicore__ inline void LoadScaleToUB(uint32_t groupIdx, uint32_t cbIdx, uint64_t curColCount, bool usePing);

    static constexpr uint32_t UB_BYTES = UB_TOTAL_BYTES;
    static constexpr uint32_t L1_BYTES = L1_TOTAL_BYTES;

    GlobalTensor<half> xGm_;
    GlobalTensor<int8_t> weightGm_;
    GlobalTensor<float> weightScaleFp32Gm_;
    GlobalTensor<uint64_t> weightScaleU64Gm_;
    GlobalTensor<int64_t> groupListGm_;
    GlobalTensor<half> smoothScaleGm_;
    GlobalTensor<half> yGm_;

    TPipe* pipe_ = nullptr;
    const QuantMatmulDequantTilingData* __restrict tiling_ = nullptr;

    // One UB arena provides persistent buffers and Phase X/D aliased workspace.
    TBuf<TPosition::VECCALC> ubArena_;
    LocalTensor<half> outPingLT_;
    LocalTensor<half> outPongLT_;
    LocalTensor<float> pertokenLT_;
    LocalTensor<uint64_t> wsRawPingLT_;
    LocalTensor<uint64_t> wsRawPongLT_;
    LocalTensor<half> smoothScaleLT_;
    LocalTensor<float> smoothFp32LT_;
    LocalTensor<half> xRawPingLT_;
    LocalTensor<int8_t> int8ScratchLT_;
    LocalTensor<half> absLT_;
    LocalTensor<half> reduceWorkLT_;
    LocalTensor<half> maxLT_;
    LocalTensor<float> xCastFloatLT_;
    LocalTensor<float> xScaledLT_;
    LocalTensor<int32_t> xInt32LT_;
    LocalTensor<half> xHalfLT_;
    // Phase D int32 NZ, fp32 NZ, and fp32 ND tiles reuse the Phase X region.
    LocalTensor<int32_t> int32NZLT_;
    LocalTensor<float> nzF32LT_;
    LocalTensor<float> fp32NDLT_;

    TBuf<TPosition::A1> xL1Buf_;
    TBuf<TPosition::A1> wL1PingBuf_;
    TBuf<TPosition::A1> wL1PongBuf_;
    TBuf<TPosition::A1> wL1ThirdBuf_;
    LocalTensor<int8_t> xL1_;
    LocalTensor<int8_t> wL1PingLT_;
    LocalTensor<int8_t> wL1PongLT_;
    LocalTensor<int8_t> wL1ThirdLT_; // Valid only when w3Enabled_

    // L0 buffers (fixed 310P sizes - L0A/L0B 64 KB, L0C 256 KB).
    TBuf<TPosition::A2> l0aBuf_;
    TBuf<TPosition::B2> l0bBuf_;
    TBuf<QuePosition::CO1> l0cBuf_;

    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t MCoreNum_ = 1; // M-axis core count
    uint32_t NCoreNum_ = 1; // N-axis core count (cb-axis split)
    // Current weight slot; advances after every successful prefetch.
    uint32_t wBufIdx_ = 0u;
    // E_ marks no next group for tail prefetch.
    bool w3Enabled_ = false;
    bool tailWInFlight_ = false;
    uint32_t nextGroupIdx_ = 0u;
    uint32_t K_ = 0;
    uint32_t N_ = 0;
    uint32_t E_ = 0;
    uint32_t Mb_ = 0;
    uint32_t Nb_ = 0;
    uint32_t Kb_ = 0;
    uint32_t fracN_ = 0;
    uint32_t fracK_ = 0;
    uint32_t weightBlockN1_ = 0;
    // ProcessGroup pre-copies row 0 when K fits one Phase X chunk.
    bool xRow0PreCopied_ = false;
};

// Distribute all non-empty expert mTiles across cores.
template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::Process()
{
    PrimeEvents();

    // Count mTiles using cumulative group_list boundaries.
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

    // Cooperative modes partition column batches within an mTile; the default
    // mode assigns contiguous mTiles to each core.
    enum class WorkMode {
        CONTIGUOUS,
        COOPBATCH_MN,
        COOPBATCH_N_ONLY
    };
    WorkMode mode;
    // Avoid re-quantizing the same rows on every core for small cb splits.
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

    // Select this core's mTiles.
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

            if (globalIterIdx + itersThis <= startIter) {
                globalIterIdx += itersThis;
                passedTokens += expertTokens;
                continue;
            }
            if (globalIterIdx >= endIter)
                break;

            const uint64_t localStart = (startIter > globalIterIdx) ? (startIter - globalIterIdx) : 0;
            const uint64_t localEnd = (endIter - globalIterIdx < itersThis) ? (endIter - globalIterIdx) : itersThis;

            for (uint64_t it = localStart; it < localEnd; ++it) {
                const uint64_t mOff = it * Mb_;
                const uint64_t mStart = passedTokens + mOff;
                const uint64_t mTail = expertTokens - mOff;
                const uint64_t mCount = (mTail < Mb_) ? mTail : Mb_;
                // E_ marks no successor for cross-group prefetch.
                uint32_t nextG = E_;
                if (it + 1 < localEnd) {
                    nextG = g;
                } else {
                    uint64_t gi2 = globalIterIdx + itersThis; // iters before expert g2
                    int64_t pc2 = prevCum;                    // tokens before expert g2
                    for (uint32_t g2 = g + 1; g2 < E_; ++g2) {
                        const int64_t endM2 = groupListGm_.GetValue(g2);
                        const int64_t tk2 = endM2 - pc2;
                        pc2 = endM2;
                        if (tk2 <= 0) {
                            continue;
                        }
                        const uint64_t it2 = (static_cast<uint64_t>(tk2) + Mb_ - 1) / Mb_;
                        if (gi2 + it2 <= startIter) {
                            gi2 += it2;
                            continue;
                        }
                        if (gi2 >= endIter) {
                            break;
                        }
                        nextG = g2;
                        break;
                    }
                }
                const uint32_t pgNCoreNum = (mode == WorkMode::COOPBATCH_MN)     ? NCoreNum_ :
                                            (mode == WorkMode::COOPBATCH_N_ONLY) ? coreNum_ :
                                                                                   1u;
                ProcessGroup(g, mStart, mCount, nCoreIdx, pgNCoreNum, nextG);
            }

            globalIterIdx += itersThis;
            passedTokens += expertTokens;
        }
    }

    DrainEvents();
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::Init(GM_ADDR x, GM_ADDR quantized_weight,
                                                               GM_ADDR weight_scale, GM_ADDR group_list, GM_ADDR bias,
                                                               GM_ADDR x_scale, GM_ADDR x_offset, GM_ADDR smooth_scale,
                                                               GM_ADDR y, GM_ADDR usr_workspace,
                                                               const QuantMatmulDequantTilingData* __restrict tiling,
                                                               TPipe* pipe)
{
    (void)bias;
    (void)x_offset;
    (void)usr_workspace;
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
    K_ = tiling->K;
    N_ = tiling->N;
    E_ = tiling->E;
    Mb_ = tiling->Mb;
    Nb_ = tiling->Nb;
    Kb_ = tiling->Kb;
    fracN_ = tiling->fracN;
    fracK_ = tiling->fracK;
    weightBlockN1_ = tiling->weightBlockN1;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
    weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(quantized_weight));
    if constexpr (ScaleIsU64) {
        weightScaleU64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(weight_scale));
    } else {
        weightScaleFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight_scale));
    }
    groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_list));
    (void)x_scale;
    if constexpr (HasSmooth) {
        smoothScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(smooth_scale));
    } else {
        (void)smooth_scale;
    }
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));

    // Use persistent L2 caching only when all padded weight slabs fit.
    {
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
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::InitUbLayout()
{
    pipe_->InitBuffer(ubArena_, UB_BYTES);
    LocalTensor<uint8_t> arena = ubArena_.Get<uint8_t>();

    const uint32_t Mb = Mb_;
    const uint32_t Nb = Nb_;
    // Mmad and Phase D operate on at least one full 16-row fractal.
    const uint32_t MbAligned = (Mb < M_FRACTAL) ? M_FRACTAL : Mb;

    uint32_t off = 0;

    pertokenLT_ = arena[off].ReinterpretCast<float>();
    off += (((Mb * sizeof(float)) + 31) / 32) * 32 + UB_BANK_PAD;

    outPingLT_ = arena[off].ReinterpretCast<half>();
    off += Mb * Nb * sizeof(half) + UB_BANK_PAD;
    outPongLT_ = arena[off].ReinterpretCast<half>();
    off += Mb * Nb * sizeof(half) + UB_BANK_PAD;

    wsRawPingLT_ = arena[off].ReinterpretCast<uint64_t>();
    off += Nb * sizeof(uint64_t) + UB_BANK_PAD;
    wsRawPongLT_ = arena[off].ReinterpretCast<uint64_t>();
    off += Nb * sizeof(uint64_t) + UB_BANK_PAD;

    if constexpr (HasSmooth) {
        // Include the K-alignment tail; PhaseXDynamicQuant zeroes it before
        // chunked fp32 conversion.
        const uint32_t smoothHalves = tiling_->originKAligned32;
        smoothScaleLT_ = arena[off].ReinterpretCast<half>();
        off += smoothHalves * sizeof(half) + UB_BANK_PAD;
        smoothFp32LT_ = arena[off].ReinterpretCast<float>();
        off += tiling_->phaseXChunk * sizeof(float) + UB_BANK_PAD;
    }

    // Phase X and Phase D workspace starts at the same offset.
    const uint32_t phaseXStart = off;
    const uint32_t chunkHalves = tiling_->phaseXChunk;

    xRawPingLT_ = arena[off].ReinterpretCast<half>();
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

    // One quantized chunk per padded row, scattered to L1 before reuse.
    int8ScratchLT_ = arena[off].ReinterpretCast<int8_t>();
    off += MbAligned * chunkHalves + UB_BANK_PAD;

    // Phase D aliases the Phase X region; output uses persistent outPing/Pong.
    uint32_t dOff = phaseXStart;
    int32NZLT_ = arena[dOff].ReinterpretCast<int32_t>();
    dOff += MbAligned * Nb * sizeof(int32_t) + UB_BANK_PAD;
    nzF32LT_ = arena[dOff].ReinterpretCast<float>();
    dOff += MbAligned * Nb * sizeof(float) + UB_BANK_PAD;
    fp32NDLT_ = arena[dOff].ReinterpretCast<float>();
    dOff += MbAligned * Nb * sizeof(float) + UB_BANK_PAD;
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::InitL0Layout()
{
    pipe_->InitBuffer(l0aBuf_, L0A_TOTAL_BYTES);
    pipe_->InitBuffer(l0bBuf_, L0B_TOTAL_BYTES);
    pipe_->InitBuffer(l0cBuf_, L0C_TOTAL_BYTES);
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::InitL1Layout()
{
    // xL1 always contains full 16-row ZZ fractals, including when Mb < 16.
    const uint32_t MbAligned = (Mb_ < static_cast<uint32_t>(M_FRACTAL)) ? static_cast<uint32_t>(M_FRACTAL) : Mb_;
    const uint32_t xL1Bytes = MbAligned * tiling_->originKAligned32;
    pipe_->InitBuffer(xL1Buf_, xL1Bytes);
    // Each weight slot holds one chunk, possibly containing multiple K-passes.
    const uint32_t wL1HalfBytes = tiling_->wChunkKPasses * Kb_ * Nb_;
    pipe_->InitBuffer(wL1PingBuf_, wL1HalfBytes);
    pipe_->InitBuffer(wL1PongBuf_, wL1HalfBytes);

    xL1_ = xL1Buf_.Get<int8_t>();
    wL1PingLT_ = wL1PingBuf_.Get<int8_t>();
    wL1PongLT_ = wL1PongBuf_.Get<int8_t>();

    // Allocate the cross-group prefetch slot only when L1 has room.
    {
        if (xL1Bytes + 2u * wL1HalfBytes + wL1HalfBytes <= L1_BYTES) {
            pipe_->InitBuffer(wL1ThirdBuf_, wL1HalfBytes);
            wL1ThirdLT_ = wL1ThirdBuf_.Get<int8_t>();
            w3Enabled_ = true;
        }
    }
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline LocalTensor<int8_t> QgmmdWideM<HasSmooth, ScaleIsU64>::WL1Slot(uint32_t idx)
{
    if (idx == 0u) {
        return wL1PingLT_;
    }
    if (idx == 1u) {
        return wL1PongLT_;
    }
    return wL1ThirdLT_;
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline event_t QgmmdWideM<HasSmooth, ScaleIsU64>::WL1FreeEvt(uint32_t idx)
{
    if (idx == 0u) {
        return EVENT_ID0;
    }
    if (idx == 1u) {
        return EVENT_ID1;
    }
    return EVENT_ID2;
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline uint32_t QgmmdWideM<HasSmooth, ScaleIsU64>::NextWBufIdx()
{
    return w3Enabled_ ? ((wBufIdx_ + 1u) % 3u) : (1u - wBufIdx_);
}
template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::PrimeEvents()
{
    // Prime buffer-free tokens consumed by the first iteration.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID6);    // outPing free
    SetFlag<HardEvent::MTE3_V>(EVENT_ID4);    // outPong free
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);    // int8Scratch free
    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);    // wsPing free
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);    // wsPong free
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0); // wL1Ping free
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1); // wL1Pong free
    if (w3Enabled_) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2); // wL1Third free
    }
    SetFlag<HardEvent::V_M>(EVENT_ID0);       // L0C free
    SetFlag<HardEvent::MTE1_MTE3>(EVENT_ID0); // xL1 free (first iter)
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5);    // Phase D alias free
    // L0A/L0B halves use ID0/ID1.
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::DrainEvents()
{
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    if (w3Enabled_) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }
    WaitFlag<HardEvent::V_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::PhaseXDynamicQuant(uint64_t mStart, uint64_t mCount)
{
    // Pass 1 stores each row's dequant scale in pertokenLT_; pass 2 quantizes
    // chunks and scatters them to xL1.

    const uint32_t K = K_;
    const uint32_t kBytes = tiling_->originKAligned32;
    const uint32_t chunkHalves = tiling_->phaseXChunk;

    if constexpr (HasSmooth) {
        DataCopy(smoothScaleLT_, smoothScaleGm_, K);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
        // Zero the K-alignment tail so padded x lanes produce zero products.
        if (kBytes > K) {
            Duplicate<half>(smoothScaleLT_[K], static_cast<half>(0), kBytes - K);
            PipeBarrier<PIPE_V>();
        }
    }

    for (uint64_t m = 0; m < mCount; ++m) {
        const uint64_t rowGmOff = (mStart + m) * static_cast<uint64_t>(K);

        // ProcessGroup already primes the token for a pre-copied first row.
        if (!(xRow0PreCopied_ && m == 0)) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        float running_max = 0.0f;
        for (uint32_t coff = 0; coff < kBytes; coff += chunkHalves) {
            const uint32_t thisChunk = (coff + chunkHalves > kBytes) ? (kBytes - coff) : chunkHalves;
            const uint32_t realInChunk = (coff >= K) ? 0u : ((K - coff < thisChunk) ? (K - coff) : thisChunk);
            // Reuse a pre-copied first chunk; otherwise wait for the previous
            // chunk's vector reads before overwriting xRawPingLT_.
            const bool p1PreCopied = (xRow0PreCopied_ && m == 0 && coff == 0);
            if (!p1PreCopied) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                if (realInChunk > 0) {
                    DataCopy(xRawPingLT_, xGm_[rowGmOff + coff], realInChunk);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                }
            } else {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            if (realInChunk < thisChunk) {
                Duplicate<half>(xRawPingLT_[realInChunk], static_cast<half>(0), thisChunk - realInChunk);
                PipeBarrier<PIPE_V>();
            }

            // The smooth path keeps multiplication, Abs, and ReduceMax in fp32.
            float part_f = 0.0f;
            if constexpr (HasSmooth) {
                Cast<float, half>(smoothFp32LT_, smoothScaleLT_[coff], RoundMode::CAST_NONE, thisChunk);
                Cast<float, half>(xCastFloatLT_, xRawPingLT_, RoundMode::CAST_NONE, thisChunk);
                PipeBarrier<PIPE_V>();
                Mul<float>(xScaledLT_, xCastFloatLT_, smoothFp32LT_, thisChunk);
                PipeBarrier<PIPE_V>();
                Abs<float>(xCastFloatLT_, xScaledLT_, thisChunk);
                PipeBarrier<PIPE_V>();
                // Reuse the chunk-sized int32 buffer as ReduceMax workspace.
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
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        const float pts = running_max * FLOAT_RECIP_127;
        pertokenLT_.SetValue(static_cast<uint32_t>(m), pts);

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    // Quantize chunk-outer/row-inner so int8Scratch holds only one K chunk
    // per row. Scatter each completed chunk before the next one reuses it.
    {
        const uint32_t kBytes2 = tiling_->originKAligned32;
        const uint32_t mAligned = ((static_cast<uint32_t>(mCount) + M_FRACTAL - 1) / M_FRACTAL) * M_FRACTAL;
        const uint32_t l1TurnStride = kBytes2 * M_FRACTAL;

        // Wait until the previous iteration's MTE1 has finished reading xL1.
        WaitFlag<HardEvent::MTE1_MTE3>(EVENT_ID0);

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);

        for (uint32_t coff = 0; coff < kBytes2; coff += chunkHalves) {
            const uint32_t thisChunk = (coff + chunkHalves > kBytes2) ? (kBytes2 - coff) : chunkHalves;
            const uint32_t realInChunk = (coff >= K) ? 0u : ((K - coff < thisChunk) ? (K - coff) : thisChunk);

            for (uint64_t m = 0; m < mCount; ++m) {
                const uint64_t rowGmOff = (mStart + m) * static_cast<uint64_t>(K);
                const float pts = pertokenLT_.GetValue(static_cast<uint32_t>(m));
                const float scale = (pts > DQ_MAX_EPS) ? (1.0f / pts) : 0.0f;

                // Pass 1 preserves a pre-copied single-row, single-chunk input.
                const bool p2Skip = (xRow0PreCopied_ && mCount == 1 && m == 0);
                if (!p2Skip) {
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
                }

                // fp16 -> fp32 scale -> int32 RINT -> half ROUND -> int8 TRUNC.
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
                // Pack rows with thisChunk stride to match ScatterZZToL1,
                // including a shorter final chunk.
                Cast<int8_t, half>(int8ScratchLT_[m * thisChunk], xHalfLT_, RoundMode::CAST_TRUNC, thisChunk);
                PipeBarrier<PIPE_V>();
                if (!p2Skip) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
            }

            // Scatter valid rows after all vector writes complete.
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            ScatterZZToL1(xL1_, int8ScratchLT_, mAligned, coff, thisChunk, l1TurnStride, static_cast<uint32_t>(mCount));
            // Do not overwrite int8Scratch until this MTE3 read completes.
            if (coff + thisChunk < kBytes2) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::PhaseXWriteXL1(uint64_t mCount)
{
    (void)mCount;

    SetFlag<HardEvent::MTE3_MTE1>(EVENT_ID0); // xL1 ready
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);    // int8Scratch free for the next iteration
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::ProcessGroup(uint32_t groupIdx, uint64_t mStart,
                                                                       uint64_t mCount, uint32_t nCoreIdx,
                                                                       uint32_t nCoreNum, uint32_t nextGroupIdx)
{
    nextGroupIdx_ = nextGroupIdx;
    // Skip empty cb slices before emitting unmatched Phase X events.
    {
        const uint32_t totalColBatch = (N_ + Nb_ - 1) / Nb_;
        const uint32_t cbPerNCore = totalColBatch / nCoreNum;
        const uint32_t cbRem = totalColBatch % nCoreNum;
        const uint32_t myCbLen = cbPerNCore + (nCoreIdx < cbRem ? 1u : 0u);
        if (myCbLen == 0u) {
            return;
        }
    }
    // Drain the previous scatter and Phase D alias before the row pre-copy.
    // The first weight copy then overlaps Phase X.
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    {
        const uint32_t kBytesPre = tiling_->originKAligned32;
        const uint32_t chunkHalvesPre = tiling_->phaseXChunk;
        xRow0PreCopied_ = (kBytesPre <= chunkHalvesPre);
        if (xRow0PreCopied_) {
            // Queue the short x copy before the larger weight copy.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            const uint64_t rowGmOff = mStart * static_cast<uint64_t>(K_);
            DataCopy(xRawPingLT_, xGm_[rowGmOff], K_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        // Consume a cross-group chunk-0 prefetch or issue a new one.
        {
            const uint32_t totalColBatchPre = (N_ + Nb_ - 1) / Nb_;
            const uint32_t cbPerNCorePre = totalColBatchPre / nCoreNum;
            const uint32_t cbRemPre = totalColBatchPre % nCoreNum;
            const uint32_t cbStartPre = cbPerNCorePre * nCoreIdx + (nCoreIdx < cbRemPre ? nCoreIdx : cbRemPre);
            const uint32_t nStartPre = cbStartPre * Nb_;
            const uint32_t nSizePre = (N_ - nStartPre < Nb_) ? (N_ - nStartPre) : Nb_;
            const uint32_t weightBytesPerExpertPre = fracK_ * fracN_ * M_FRACTAL * K_FRACTAL_INT8_BYTES;
            const uint64_t expertBaseOffPre = static_cast<uint64_t>(groupIdx) * weightBytesPerExpertPre;
            const uint32_t Kb1Pre = tiling_->Kb / K_FRACTAL_INT8_BYTES;
            const uint32_t K1TotalPre = kBytesPre / K_FRACTAL_INT8_BYTES;
            const uint32_t k1PerChunkPre = tiling_->wChunkKPasses * Kb1Pre;
            const uint32_t kp0K1CountPre = (K1TotalPre < k1PerChunkPre) ? K1TotalPre : k1PerChunkPre;
            if (tailWInFlight_) {
                tailWInFlight_ = false;
            } else {
                LocalTensor<int8_t> curWL1Pre = WL1Slot(wBufIdx_);
                WaitFlag<HardEvent::MTE1_MTE2>(WL1FreeEvt(wBufIdx_));
                PrefetchWeightToL1<false>(curWL1Pre, weightGm_, expertBaseOffPre,
                                          /* colBatch= */ cbStartPre,
                                          /* nSize= */ nSizePre,
                                          /* baseNG= */ nStartPre / N_FRACTAL,
                                          /* N1= */ fracN_,
                                          /* K1q= */ kp0K1CountPre,
                                          /* K= */ K_,
                                          /* weightBlockN1= */ weightBlockN1_,
                                          /* k1Start= */ 0);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1); // weight L1[ZN] ready
            }
        }
    }
    PhaseXDynamicQuant(mStart, mCount);
    PhaseXWriteXL1(mCount);

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
    SetFlag<HardEvent::MTE1_MTE3>(EVENT_ID0); // xL1 free for the next iteration
    // Phase D writes must drain before Phase X reuses its aliased UB region.
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
}
template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::ProcessColBatch(uint32_t groupIdx, uint32_t cbIdx,
                                                                          uint64_t mStart, uint64_t mCount,
                                                                          uint64_t curColCount, uint32_t cbEnd,
                                                                          uint32_t cbStart)
{
    const uint32_t mAligned = ((static_cast<uint32_t>(mCount) + M_FRACTAL - 1) / M_FRACTAL) * M_FRACTAL;
    const uint32_t nSize = static_cast<uint32_t>(curColCount);
    const uint32_t kBytes = tiling_->originKAligned32;
    const uint32_t nStart = cbIdx * Nb_;
    const bool usePing = (cbIdx & 1u) == 0;

    // Weight slots follow MTE2 issue order; scales and output follow cb parity.
    LocalTensor<int8_t> curWL1 = WL1Slot(wBufIdx_);
    LocalTensor<float> wsF32 = (usePing ? wsRawPingLT_ : wsRawPongLT_).ReinterpretCast<float>();
    LocalTensor<half> finalOut = usePing ? outPingLT_ : outPongLT_;

    // Wait for the same-parity predecessors to release output and scale slots.
    WaitFlag<HardEvent::MTE3_V>(usePing ? EVENT_ID6 : EVENT_ID4);
    WaitFlag<HardEvent::V_MTE2>(usePing ? EVENT_ID2 : EVENT_ID1);
    LoadScaleToUB(groupIdx, cbIdx, curColCount, usePing);

    // Weight GM uses padded [E, K1, N1, 16, 32] ZN storage.
    const uint32_t weightBytesPerExpert = fracK_ * fracN_ * M_FRACTAL * K_FRACTAL_INT8_BYTES;
    const uint64_t expertBaseOff = static_cast<uint64_t>(groupIdx) * weightBytesPerExpert;
    // Each weight chunk contains wChunkKPasses K-passes for one MTE2 copy.
    const uint32_t Kb1_ = tiling_->Kb / K_FRACTAL_INT8_BYTES;
    const uint32_t K1_total = kBytes / K_FRACTAL_INT8_BYTES;
    const uint32_t kPasses_ = tiling_->kPasses;
    const uint32_t wChunkKPasses_ = tiling_->wChunkKPasses; // >=1, divides kPasses_
    const uint32_t k1PerChunk_ = wChunkKPasses_ * Kb1_;
    const uint32_t kp0K1Count = (K1_total < k1PerChunk_) ? K1_total : k1PerChunk_;
    SetFlag<HardEvent::MTE2_V>(EVENT_ID4); // scale UB ready

    LocalTensor<int8_t> l0a = l0aBuf_.Get<int8_t>();
    LocalTensor<int8_t> l0b = l0bBuf_.Get<int8_t>();
    LocalTensor<int32_t> l0c = l0cBuf_.Get<int32_t>();

    if (cbIdx == cbStart) {
        // Phase X produces one xL1-ready token for all cbs.
        WaitFlag<HardEvent::MTE3_MTE1>(EVENT_ID0);
    }
    const uint16_t nFracs = static_cast<uint16_t>(nSize / N_FRACTAL);
    const uint16_t mTurns = static_cast<uint16_t>(mAligned / M_FRACTAL);
    const uint32_t xL1TurnStride = kBytes * M_FRACTAL;                   // bytes per ZZ turn
    constexpr uint32_t FRACTAL_BYTES = M_FRACTAL * K_FRACTAL_INT8_BYTES; // 512

    const uint32_t numChunks_ = kPasses_ / wChunkKPasses_;
    const uint32_t kbBytesPerKp = Kb1_ * static_cast<uint32_t>(nFracs) * FRACTAL_BYTES;
    bool nxFired = false;         // tracks whether the next MTE2 was pre-fired
    uint32_t nxBufIdx = wBufIdx_; // slot the fired prefetch targets
    for (uint32_t chunkIdx = 0; chunkIdx < numChunks_; ++chunkIdx) {
        const uint32_t kpBase = chunkIdx * wChunkKPasses_;

        // Consume the weight copy issued by the preceding chunk or pre-pack.
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

        nxFired = false;
        if (wChunkKPasses_ > 1) {
            // Prefetch the next chunk, next cb, or next group's first chunk.
            const uint32_t nxBuf = NextWBufIdx();
            nxBufIdx = nxBuf;
            LocalTensor<int8_t> nxWL1 = WL1Slot(nxBuf);
            if (chunkIdx + 1 < numChunks_) {
                const uint32_t k1NxStart = (chunkIdx + 1) * k1PerChunk_;
                const uint32_t k1NxRemain = K1_total - k1NxStart;
                const uint32_t k1NxToLoad = (k1PerChunk_ < k1NxRemain) ? k1PerChunk_ : k1NxRemain;
                WaitFlag<HardEvent::MTE1_MTE2>(WL1FreeEvt(nxBuf));
                PrefetchWeightToL1<false>(nxWL1, weightGm_, expertBaseOff,
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
            } else if (cbIdx + 1 < cbEnd) {
                const uint32_t nxCb = cbIdx + 1;
                const uint32_t nxNStart = nxCb * Nb_;
                const uint32_t nxColCount = (N_ - nxNStart < Nb_) ? (N_ - nxNStart) : Nb_;
                WaitFlag<HardEvent::MTE1_MTE2>(WL1FreeEvt(nxBuf));
                PrefetchWeightToL1<false>(nxWL1, weightGm_, expertBaseOff,
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
            } else if (w3Enabled_ && (nextGroupIdx_ < E_)) {
                // The three-slot pool keeps the next group's copy in flight across the tail.
                const uint32_t weightBytesPerExpertNx = fracK_ * fracN_ * M_FRACTAL * K_FRACTAL_INT8_BYTES;
                const uint64_t expertBaseOffNx = static_cast<uint64_t>(nextGroupIdx_) * weightBytesPerExpertNx;
                const uint32_t nxStartTail = cbStart * Nb_;
                const uint32_t nxSizeTail = (N_ - nxStartTail < Nb_) ? (N_ - nxStartTail) : Nb_;
                WaitFlag<HardEvent::MTE1_MTE2>(WL1FreeEvt(nxBuf));
                PrefetchWeightToL1<false>(nxWL1, weightGm_, expertBaseOffNx,
                                          /* colBatch= */ cbStart,
                                          /* nSize= */ nxSizeTail,
                                          /* baseNG= */ nxStartTail / N_FRACTAL,
                                          /* N1= */ fracN_,
                                          /* K1q= */ kp0K1Count,
                                          /* K= */ K_,
                                          /* weightBlockN1= */ weightBlockN1_,
                                          /* k1Start= */ 0);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
                tailWInFlight_ = true;
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

            // Alternate L0A/L0B halves so kp+1 LoadData overlaps kp Mmad.
            const uint32_t kpPar = kp & 1u;
            const event_t l0Evt = kpPar ? EVENT_ID1 : EVENT_ID0;
            const uint32_t l0aOff = kpPar ? L0A_HALF_BYTES : 0u;
            const uint32_t l0bOff = kpPar ? L0B_HALF_BYTES : 0u;

            WaitFlag<HardEvent::M_MTE1>(l0Evt);

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

        // All MTE1 reads of this weight slot have completed.
        SetFlag<HardEvent::MTE1_MTE2>(WL1FreeEvt(wBufIdx_));

        // If no successor was pre-issued, handle it after the compute loop.
        if (!nxFired) {
            const uint32_t nxBuf = NextWBufIdx();
            nxBufIdx = nxBuf;
            LocalTensor<int8_t> nxWL1 = WL1Slot(nxBuf);
            if (chunkIdx + 1 < numChunks_) {
                const uint32_t k1NxStart = (chunkIdx + 1) * k1PerChunk_;
                const uint32_t k1NxRemain = K1_total - k1NxStart;
                const uint32_t k1NxToLoad = (k1PerChunk_ < k1NxRemain) ? k1PerChunk_ : k1NxRemain;
                WaitFlag<HardEvent::MTE1_MTE2>(WL1FreeEvt(nxBuf));
                PrefetchWeightToL1<false>(nxWL1, weightGm_, expertBaseOff,
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
            } else if (cbIdx + 1 < cbEnd) {
                const uint32_t nxCb = cbIdx + 1;
                const uint32_t nxNStart = nxCb * Nb_;
                const uint32_t nxColCount = (N_ - nxNStart < Nb_) ? (N_ - nxNStart) : Nb_;
                WaitFlag<HardEvent::MTE1_MTE2>(WL1FreeEvt(nxBuf));
                PrefetchWeightToL1<false>(nxWL1, weightGm_, expertBaseOff,
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
            } else if (w3Enabled_ && (nextGroupIdx_ < E_)) {
                const uint32_t weightBytesPerExpertNx = fracK_ * fracN_ * M_FRACTAL * K_FRACTAL_INT8_BYTES;
                const uint64_t expertBaseOffNx = static_cast<uint64_t>(nextGroupIdx_) * weightBytesPerExpertNx;
                const uint32_t nxStartTail = cbStart * Nb_;
                const uint32_t nxSizeTail = (N_ - nxStartTail < Nb_) ? (N_ - nxStartTail) : Nb_;
                WaitFlag<HardEvent::MTE1_MTE2>(WL1FreeEvt(nxBuf));
                PrefetchWeightToL1<false>(nxWL1, weightGm_, expertBaseOffNx,
                                          /* colBatch= */ cbStart,
                                          /* nSize= */ nxSizeTail,
                                          /* baseNG= */ nxStartTail / N_FRACTAL,
                                          /* N1= */ fracN_,
                                          /* K1q= */ kp0K1Count,
                                          /* K= */ K_,
                                          /* weightBlockN1= */ weightBlockN1_,
                                          /* k1Start= */ 0);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
                tailWInFlight_ = true;
                nxFired = true;
            }
        }

        // Follow the slot used by the successful prefetch, including cross-group.
        if (nxFired) {
            wBufIdx_ = nxBufIdx;
            curWL1 = WL1Slot(wBufIdx_);
        }
    }

    SetFlag<HardEvent::M_V>(EVENT_ID3);

    // Keep dequant in fp32 and round only at the final fp16 store:
    // y = fp16(fp32(fp32(acc) * weightScale) * perTokenScale).
    WaitFlag<HardEvent::M_V>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);

    DataCopyEnhancedParams nzCopyP;
    nzCopyP.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(int32NZLT_, l0c, {static_cast<uint16_t>(nFracs), mTurns, 0, 0}, nzCopyP);
    SetFlag<HardEvent::V_M>(EVENT_ID0); // L0C free

    PipeBarrier<PIPE_V>();

    // Convert NZ to fp32, then scatter each 16-column fractal into ND while
    // applying its column scales. Repeat strides use 32-byte units: dst moves
    // one ND row, src0 one NZ row, and src1 broadcasts the same 16 scales.
    constexpr uint32_t MAX_VEC_REPEATS = 255U;
    const uint8_t ndRepStride = static_cast<uint8_t>(nSize / 8U);
    const uint32_t srcNZStride = mAligned * N_FRACTAL;
    const uint32_t castCount = static_cast<uint32_t>(mCount) * N_FRACTAL;
    for (uint32_t ng = 0; ng < nFracs; ++ng) {
        Cast<float, int32_t>(nzF32LT_[ng * srcNZStride], int32NZLT_[ng * srcNZStride], RoundMode::CAST_NONE, castCount);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t ng = 0; ng < nFracs; ++ng) {
        uint32_t mRemaining = static_cast<uint32_t>(mCount);
        uint32_t mDone = 0U;
        while (mRemaining > 0U) {
            const uint8_t thisReps = (mRemaining > MAX_VEC_REPEATS) ? static_cast<uint8_t>(MAX_VEC_REPEATS) :
                                                                      static_cast<uint8_t>(mRemaining);
            Mul<float>(fp32NDLT_[ng * N_FRACTAL + mDone * nSize], nzF32LT_[ng * srcNZStride + mDone * N_FRACTAL],
                       wsF32[ng * N_FRACTAL], N_FRACTAL, thisReps, {1, 1, 1, ndRepStride, 2, 0});
            mDone += thisReps;
            mRemaining -= thisReps;
        }
    }
    PipeBarrier<PIPE_V>();
    // The scatter Mul is the last reader of this scale slot.
    SetFlag<HardEvent::V_MTE2>(usePing ? EVENT_ID2 : EVENT_ID1);

    for (uint32_t m = 0; m < mCount; ++m) {
        const float pts_f = pertokenLT_.GetValue(static_cast<uint32_t>(m));
        Muls<float>(fp32NDLT_[m * nSize], fp32NDLT_[m * nSize], pts_f, nSize);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t m = 0; m < mCount; ++m) {
        Cast<half, float>(finalOut[m * nSize], fp32NDLT_[m * nSize], RoundMode::CAST_NONE, nSize);
    }
    PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    const uint64_t yRowStride = static_cast<uint64_t>(N_);
    DataCopy(yGm_[mStart * yRowStride + nStart], finalOut,
             {static_cast<uint16_t>(mCount), static_cast<uint16_t>(nSize / N_FRACTAL), 0,
              static_cast<uint16_t>((N_ - nSize) / N_FRACTAL)});
    SetFlag<HardEvent::MTE3_V>(usePing ? EVENT_ID6 : EVENT_ID4); // finalOut slot free
}

template <bool HasSmooth, bool ScaleIsU64>
__aicore__ inline void QgmmdWideM<HasSmooth, ScaleIsU64>::LoadScaleToUB(uint32_t groupIdx, uint32_t cbIdx,
                                                                        uint64_t curColCount, bool usePing)
{
    // Phase D consumes fp32 column scales. For uint64 input, GatherMask
    // compacts the low int32 lane of each word, which contains the fp32 bits.
    const uint32_t nSize = static_cast<uint32_t>(curColCount);
    const uint32_t nStart = cbIdx * Nb_;
    const uint64_t scaleOff = static_cast<uint64_t>(groupIdx) * N_ + nStart;
    LocalTensor<uint64_t> curWsRaw = usePing ? wsRawPingLT_ : wsRawPongLT_;

    if constexpr (ScaleIsU64) {
        DataCopy<uint64_t>(curWsRaw, weightScaleU64Gm_[scaleOff], nSize);
        // Wait for the local copy before GatherMask reads it.
        SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
        LocalTensor<int32_t> wsI32 = curWsRaw.ReinterpretCast<int32_t>();
        GatherMaskParams gatherMaskParams;
        gatherMaskParams.src0BlockStride = 1;
        gatherMaskParams.repeatTimes = (nSize * sizeof(uint64_t) + 255U) / 256U;
        gatherMaskParams.src0RepeatStride = 256U / 32U; // 32B blocks per 256B row
        gatherMaskParams.src1RepeatStride = 0;
        uint64_t rsvdCnt = nSize;
        AscendC::GatherMask(wsI32, wsI32, 1, false, 0, gatherMaskParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
    } else {
        DataCopy<float>(curWsRaw.ReinterpretCast<float>(), weightScaleFp32Gm_[scaleOff], nSize);
    }
}

// Only smooth-scale presence and weight-scale dtype vary at runtime.
#define QGMMD_WIDE_M_DISPATCH(hasSmooth, scaleIsU64) \
    do { \
        QgmmdWideM<hasSmooth, scaleIsU64> handle; \
        handle.Init(x, quantized_weight, weight_scale, group_list, bias, x_scale, x_offset, smooth_scale, y, \
                    usrworkspace, tilingData, &tPipe); \
        handle.Process(); \
    } while (0)

#define QGMMD_WIDE_M_DISPATCH_SMOOTH(scaleIsU64) \
    do { \
        if (hasSmooth) { \
            QGMMD_WIDE_M_DISPATCH(true, scaleIsU64); \
        } else { \
            QGMMD_WIDE_M_DISPATCH(false, scaleIsU64); \
        } \
    } while (0)

#define QGMMD_WIDE_M_DISPATCH_SCALE() \
    do { \
        if (scaleIsU64) { \
            QGMMD_WIDE_M_DISPATCH_SMOOTH(true); \
        } else { \
            QGMMD_WIDE_M_DISPATCH_SMOOTH(false); \
        } \
    } while (0)

__aicore__ inline __attribute__((noinline)) void RunPertokenDynamicWideM(
    GM_ADDR x, GM_ADDR quantized_weight, GM_ADDR weight_scale, GM_ADDR group_list, GM_ADDR bias, GM_ADDR x_scale,
    GM_ADDR x_offset, GM_ADDR smooth_scale, GM_ADDR y, GM_ADDR usrworkspace,
    QuantMatmulDequantTilingData* __restrict tilingData)
{
    TPipe tPipe;
    const bool hasSmooth = GetFlag(tilingData->flags, FLAG_BIT_HAS_SMOOTH);
    const bool scaleIsU64 = GetFlag(tilingData->flags, FLAG_BIT_SCALE_IS_INT64);
    QGMMD_WIDE_M_DISPATCH_SCALE();
}

} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_PERTOKEN_WIDE_M_H
