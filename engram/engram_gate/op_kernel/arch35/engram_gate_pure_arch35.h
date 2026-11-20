/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// EngramGate pure-path kernel — credit-based pipeline
// ============================================================================
// Pipeline structure (per core, persistent):
//
//   while (more batches to compute):
//     PREFETCH: issue MTE2 copies for future batches (up to `credits` ahead)
//     COMPUTE:  VF on current batch's input slot
//     OUTPUT:   MTE3 copy from current batch's output slot
//
// Input (x, kv) and output (out) live in separate UB regions, so their slot
// lifetimes are independent:
//   Input slot:  MTE2 writes → VF reads  → released immediately after VF
//   Output slot: VF writes  → MTE3 reads → released after MTE3 copy done
//
// This overlap means MTE2 for batch N+1 is already in flight while VF is
// still computing batch N — the copy latency is hidden behind compute.
//
// Slot assignment uses a local counter (not batch % S) to guarantee
// alternation regardless of usedCores % S.
// ============================================================================

#ifndef ENGRAM_GATE_PURE_ARCH35_H
#define ENGRAM_GATE_PURE_ARCH35_H

#include "kernel_operator.h"
#include "c_api/vector_datamove/vector_datamove.h"
#include "engram_gate_pure_vf_arch35.h"

namespace EngramGatePure {
namespace {
constexpr event_t INPUT_EVENT_BASE = static_cast<event_t>(0);
constexpr event_t OUTPUT_READY_EVENT = static_cast<event_t>(4);
constexpr event_t OUTPUT_DONE_EVENT_BASE = static_cast<event_t>(5);
} // namespace

template <typename T, bool SAVE, uint8_t STORE_CACHE_MODE>
class EngramGateArch35 {
    struct PipelineState {
        const EngramGateTilingData& t;
        int64_t d, H, totalRows, K, core, usedCores, numBatch, localBatchCount;
        uint32_t S, activeSlots, xBytesPerSlot, outBytesPerSlot, kvBytesPerSlot, rowBytes, kvTokenBytes, wBytes;
        bool singleBatch;
        __gm__ T* xGm;
        __gm__ T* kvGm;
        __gm__ float* wGm;
        __gm__ T* outGm;
        __gm__ float* dotGm;
        __gm__ float* gateGm;
        __gm__ float* rstdXGm;
        __gm__ float* rstdKGm;
        __ubuf__ T* xRegion;
        __ubuf__ T* kvRegion;
        __ubuf__ T* outRegion;
        __ubuf__ float* wBase;
        __ubuf__ float* statsBase;
    };

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR kv, GM_ADDR w, GM_ADDR out, GM_ADDR dot, GM_ADDR gate, GM_ADDR rx,
                                GM_ADDR rk, const EngramGateTilingData* tiling)
    {
        hiddenStates_ = x;
        kv_ = kv;
        weight_ = w;
        output_ = out;
        dot_ = dot;
        gate_ = gate;
        rstdX_ = rx;
        rstdK_ = rk;
        tilingData_ = tiling;
    }

    __aicore__ inline void Process()
    {
        const EngramGateTilingData& t = *tilingData_;
        if (t.rows == 0 || t.pureSlots == 0) {
            return;
        }
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> root;
        pipe.InitBuffer(root, t.pureUbBytes);
        auto buffer = root.Get<uint8_t>();
        __ubuf__ uint8_t* ub = reinterpret_cast<__ubuf__ uint8_t*>(buffer.GetPhyAddr());
        PipelineState state{t};
        InitializeState(state, ub);
        PreArm(state);
        RunPipeline(state);
        Drain(state);
    }

private:
    __aicore__ inline void InitializeState(PipelineState& s, __ubuf__ uint8_t* ub)
    {
        s.d = s.t.dim;
        s.H = s.t.heads;
        s.totalRows = s.t.rows;
        s.K = s.t.pureRowsPerSlot;
        s.S = s.t.pureSlots;
        s.core = AscendC::GetBlockIdx();
        s.usedCores = s.t.usedCores;
        s.numBatch = (s.totalRows + s.K - 1) / s.K;
        s.localBatchCount = (s.numBatch > s.core) ? ((s.numBatch - 1 - s.core) / s.usedCores + 1) : 0;
        s.activeSlots = static_cast<uint32_t>(s.localBatchCount < static_cast<int64_t>(s.S) ? s.localBatchCount : s.S);
        s.singleBatch = s.localBatchCount == 1;
        s.xGm = reinterpret_cast<__gm__ T*>(hiddenStates_);
        s.kvGm = reinterpret_cast<__gm__ T*>(kv_);
        s.wGm = reinterpret_cast<__gm__ float*>(weight_);
        s.outGm = reinterpret_cast<__gm__ T*>(output_);
        s.dotGm = reinterpret_cast<__gm__ float*>(dot_);
        s.gateGm = reinterpret_cast<__gm__ float*>(gate_);
        s.rstdXGm = reinterpret_cast<__gm__ float*>(rstdX_);
        s.rstdKGm = reinterpret_cast<__gm__ float*>(rstdK_);
        s.xBytesPerSlot = static_cast<uint32_t>(s.K * s.d * sizeof(T));
        s.outBytesPerSlot = s.xBytesPerSlot;
        const uint32_t kvRows = (s.K >= s.H) ? static_cast<uint32_t>(s.K / s.H * (s.H + 1)) : (s.K == 1 ? 2U : 3U);
        s.kvBytesPerSlot = kvRows * static_cast<uint32_t>(s.d) * sizeof(T);
        s.rowBytes = static_cast<uint32_t>(s.d * sizeof(T));
        s.kvTokenBytes = static_cast<uint32_t>((s.H + 1) * s.d * sizeof(T));
        s.xRegion = reinterpret_cast<__ubuf__ T*>(ub + s.t.pureXOff);
        s.kvRegion = reinterpret_cast<__ubuf__ T*>(ub + s.t.pureKvOff);
        s.outRegion = reinterpret_cast<__ubuf__ T*>(ub + s.t.pureOutOff);
        s.wBase = reinterpret_cast<__ubuf__ float*>(ub + s.t.pureWOff);
        s.statsBase = reinterpret_cast<__ubuf__ float*>(ub + s.t.pureStatsOff);
        const uint32_t weightRows = (s.K < s.H) ? static_cast<uint32_t>(s.K) : static_cast<uint32_t>(s.H);
        s.wBytes = static_cast<uint32_t>(weightRows * s.d * sizeof(float));
    }

    __aicore__ inline void PreArm(const PipelineState& s)
    {
        if (!s.singleBatch) {
            for (uint32_t slot = 0; slot < s.activeSlots; ++slot) {
                SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(INPUT_EVENT_BASE + slot));
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(OUTPUT_DONE_EVENT_BASE + slot));
            }
        }
    }

    __aicore__ inline void CopyKv(const PipelineState& s, __ubuf__ T* dst, int64_t firstRow, int64_t rows)
    {
        if (s.K >= s.H) {
            const uint16_t tokenBursts = static_cast<uint16_t>((rows + s.H - 1) / s.H);
            const int64_t firstToken = firstRow / s.H;
            AscendC::CopyGmToUbufAlignV2<T>(dst, s.kvGm + firstToken * (s.H + 1) * s.d, tokenBursts, s.kvTokenBytes, 0,
                                            0, 0, 0, true, 0, false);
        } else if (s.K == 1) {
            const int64_t token = firstRow / s.H;
            const int64_t head = firstRow % s.H;
            __gm__ T* key = s.kvGm + token * (s.H + 1) * s.d + head * s.d;
            if (head == s.H - 1) {
                AscendC::CopyGmToUbufAlignV2<T>(dst, key, 1, 2 * s.rowBytes, 0, 0, 0, 0, true, 0, false);
            } else {
                AscendC::CopyGmToUbufAlignV2<T>(dst, key, 1, s.rowBytes, 0, 0, 0, 0, true, 0, false);
                AscendC::CopyGmToUbufAlignV2<T>(dst + s.d, s.kvGm + token * (s.H + 1) * s.d + s.H * s.d, 1, s.rowBytes,
                                                0, 0, 0, 0, true, 0, false);
            }
        } else {
            const int64_t token = firstRow / s.H;
            const int64_t half = (firstRow % s.H) / 2;
            if (half == 1) {
                AscendC::CopyGmToUbufAlignV2<T>(dst, s.kvGm + token * (s.H + 1) * s.d + 2 * s.d, 1,
                                                static_cast<uint32_t>(3 * s.d * sizeof(T)), 0, 0, 0, 0, true, 0, false);
            } else {
                AscendC::CopyGmToUbufAlignV2<T>(dst, s.kvGm + token * (s.H + 1) * s.d, 1,
                                                static_cast<uint32_t>(2 * s.d * sizeof(T)), 0, 0, 0, 0, true, 0, false);
                AscendC::CopyGmToUbufAlignV2<T>(dst + 2 * s.d, s.kvGm + token * (s.H + 1) * s.d + s.H * s.d, 1,
                                                static_cast<uint32_t>(s.d * sizeof(T)), 0, 0, 0, 0, true, 0, false);
            }
        }
    }

    __aicore__ inline void Prefetch(PipelineState& s, int64_t batch, int64_t issued)
    {
        const uint32_t slot = static_cast<uint32_t>(issued % s.S);
        const int64_t firstRow = batch * s.K;
        const int64_t rows = (s.totalRows - firstRow < s.K) ? (s.totalRows - firstRow) : s.K;
        __ubuf__ T* x = s.xRegion + slot * (s.xBytesPerSlot / sizeof(T));
        __ubuf__ T* kv = s.kvRegion + slot * (s.kvBytesPerSlot / sizeof(T));
        if (!s.singleBatch) {
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(INPUT_EVENT_BASE + slot));
        }
        // Weight shares the first slot's MTE2 event; each core keeps its head or head pair.
        if (issued == 0) {
            const int64_t headBase = (s.K < s.H) ? ((s.core * s.K) % s.H) : 0;
            AscendC::CopyGmToUbufAlignV2<float>(s.wBase + headBase * s.d, s.wGm + headBase * s.d, 1, s.wBytes, 0, 0, 0,
                                                0, true, 0, false);
        }
        AscendC::CopyGmToUbufAlignV2<T>(x, s.xGm + firstRow * s.d, 1, static_cast<uint32_t>(rows * s.rowBytes), 0, 0, 0,
                                        0, true, 0, false);
        CopyKv(s, kv, firstRow, rows);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT_BASE + slot));
    }

    __aicore__ inline void ComputeRows(const PipelineState& s, __ubuf__ T* x, __ubuf__ T* kv, __ubuf__ T* out,
                                       __ubuf__ float* stats, uint32_t stride, int64_t rows)
    {
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t group = (rows - r < s.H) ? (rows - r) : s.H;
            const int64_t token = (s.K >= s.H) ? (r / s.H) : 0;
            const int64_t head = (s.K >= s.H) ? (r % s.H) : ((s.core * s.K + r) % s.H);
            __ubuf__ T* v = (s.K >= s.H) ? (kv + token * (s.H + 1) * s.d + s.H * s.d) : (kv + s.K * s.d);
            __ubuf__ T* key = (s.K >= s.H) ? (kv + token * (s.H + 1) * s.d + head * s.d) : (kv + r * s.d);
            __ubuf__ float* dot = stats + r;
            __ubuf__ float* gate = dot + stride;
            __ubuf__ float* rstdX = gate + stride;
            __ubuf__ float* rstdK = rstdX + stride;
            if (group == 1) {
                EngramGateOps::EngramGatePureFullRowVF<T, T, 1, SAVE>(
                    out + r * s.d, x + r * s.d, key, v, s.wBase + head * s.d, dot, gate, rstdX, rstdK,
                    static_cast<int32_t>(s.d), static_cast<int32_t>(s.d >> 6), s.t.eps, s.t.clampValue, s.t.invDim,
                    s.t.invSqrtDim);
            } else if (group == 2) {
                EngramGateOps::EngramGatePureFullRowVF<T, T, 2, SAVE>(
                    out + r * s.d, x + r * s.d, key, v, s.wBase + head * s.d, dot, gate, rstdX, rstdK,
                    static_cast<int32_t>(s.d), static_cast<int32_t>(s.d >> 6), s.t.eps, s.t.clampValue, s.t.invDim,
                    s.t.invSqrtDim);
            } else {
                EngramGateOps::EngramGatePureFullRowVF<T, T, 4, SAVE>(
                    out + r * s.d, x + r * s.d, key, v, s.wBase + head * s.d, dot, gate, rstdX, rstdK,
                    static_cast<int32_t>(s.d), static_cast<int32_t>(s.d >> 6), s.t.eps, s.t.clampValue, s.t.invDim,
                    s.t.invSqrtDim);
            }
            r += group - 1;
        }
    }

    __aicore__ inline void ComputeBatch(PipelineState& s, int64_t batch, int64_t consumed)
    {
        const int64_t firstRow = batch * s.K;
        const int64_t rows = (s.totalRows - firstRow < s.K) ? (s.totalRows - firstRow) : s.K;
        const uint32_t slot = static_cast<uint32_t>(consumed % s.S);
        __ubuf__ T* x = s.xRegion + slot * (s.xBytesPerSlot / sizeof(T));
        __ubuf__ T* kv = s.kvRegion + slot * (s.kvBytesPerSlot / sizeof(T));
        __ubuf__ T* out = s.outRegion + slot * (s.outBytesPerSlot / sizeof(T));
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT_BASE + slot));
        if (!s.singleBatch) {
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(OUTPUT_DONE_EVENT_BASE + slot));
        }
        // Four statistics windows prevent an in-flight MTE3 copy from aliasing a later VF write.
        const uint32_t stride = (static_cast<uint32_t>(s.K) + 7U) & ~7U;
        __ubuf__ float* stats = s.statsBase + static_cast<uint32_t>(consumed & 3U) * (4U * stride);
        ComputeRows(s, x, kv, out, stats, stride, rows);
        if (!s.singleBatch) {
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(INPUT_EVENT_BASE + slot));
        }
        SetFlag<HardEvent::V_MTE3>(OUTPUT_READY_EVENT);
        WaitFlag<HardEvent::V_MTE3>(OUTPUT_READY_EVENT);
        if constexpr (SAVE) {
            __gm__ float* targets[4] = {s.dotGm, s.gateGm, s.rstdXGm, s.rstdKGm};
            for (uint32_t cls = 0; cls < 4; ++cls) {
                AscendC::CopyUbufToGmAlignV2<float, AscendC::PaddingMode::Normal>(
                    targets[cls] + firstRow, stats + cls * stride, 1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0,
                    true, 0);
            }
        }
        AscendC::CopyUbufToGmAlignV2<T>(s.outGm + firstRow * s.d, out, 1, static_cast<uint32_t>(rows * s.rowBytes), 0,
                                        0, true, STORE_CACHE_MODE);
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(OUTPUT_DONE_EVENT_BASE + slot));
    }

    __aicore__ inline void RunPipeline(PipelineState& s)
    {
        int64_t issued = 0;
        int64_t consumed = 0;
        int64_t next = s.core;
        for (int64_t batch = s.core; batch < s.numBatch; batch += s.usedCores) {
            while (next < s.numBatch && issued < consumed + s.activeSlots) {
                Prefetch(s, next, issued);
                ++issued;
                next += s.usedCores;
            }
            ComputeBatch(s, batch, consumed);
            ++consumed;
        }
    }

    __aicore__ inline void Drain(const PipelineState& s)
    {
        for (uint32_t slot = 0; slot < s.activeSlots; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(OUTPUT_DONE_EVENT_BASE + slot));
            if (!s.singleBatch) {
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(INPUT_EVENT_BASE + slot));
            }
        }
    }

    const EngramGateTilingData* tilingData_ = nullptr;
    GM_ADDR hiddenStates_ = nullptr;
    GM_ADDR kv_ = nullptr;
    GM_ADDR weight_ = nullptr;
    GM_ADDR output_ = nullptr;
    GM_ADDR dot_ = nullptr;
    GM_ADDR gate_ = nullptr;
    GM_ADDR rstdX_ = nullptr;
    GM_ADDR rstdK_ = nullptr;
};

} // namespace EngramGatePure

#endif // ENGRAM_GATE_PURE_ARCH35_H
