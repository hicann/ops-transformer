/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_EPILOGUE_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_EPILOGUE_H

#include "compute/mega_mhc_pre.h"
#include "compute/rms_norm_common.h"
#include "mega_mhc_common.h"

namespace MegaMhc::Schedule {
using namespace AscendC;

constexpr uint64_t kMaxDmaGapBytes = 0xFFFFFFFFULL;
constexpr uint32_t kNzRowAlignment = 8U;
constexpr uint32_t kPackedHResPlaneOffset = 8U;
constexpr uint32_t kRmsGridElements = 8U;
constexpr uint32_t kNzRowRounding = 7U;
constexpr uint32_t kPartialBufferGuardElements = 64U;
constexpr uint32_t kAlphaElements = 3U;
constexpr uint32_t kAlphaBufferElements = 4U;
constexpr uint32_t kBiasBufferElements = 32U;
constexpr uint32_t kEpilogueStatsElements = 4U;
constexpr uint32_t kEpiloguePrefetchFull = 2U;
constexpr uint32_t kEpilogueRmsCarryIndex = 2U;

// Direction-specific event namespaces: projection uses MTE3_V 6/7, BS y uses 6/7,
// and tiled-row inputs use MTE2_V 2/3. These two channels remain independent.
constexpr event_t kShiftedPreInputEvent = EVENT_ID7;
constexpr event_t kCurrentPreOutputEvent = EVENT_ID2;

/** Keep external mixing coefficients separate from the current Pre/Post/Res planes.
 * External input DMA starts before projection reduction. Current pre output DMA is
 * drained only before its source plane is overwritten, or before the pipe is released. */
template <typename T>
class PreMixStorage {
public:
    __aicore__ inline void Init(TPipe* pipe, const MegaMhcTilingData& tiling, bool enabled)
    {
        enabled_ = enabled;
        outputPending_ = false;
        if (enabled_) {
            Kernel::InitUbBuffer(pipe, shiftedBuffer_, tiling.reduceRowsAligned * Kernel::kHeadCount * sizeof(float));
            shifted_ = shiftedBuffer_.template Get<float>();
        }
    }

    /** V_MTE2 protects external coefficients until all Vector readers from the previous batch finish. */
    __aicore__ inline void Prefetch(Kernel::Tensors<T>& tensors, uint64_t row, uint32_t rows)
    {
        if (enabled_) {
            SetFlag<HardEvent::V_MTE2>(kShiftedPreInputEvent);
            WaitFlag<HardEvent::V_MTE2>(kShiftedPreInputEvent);
            Kernel::EnqueueGlobalToUb(shifted_, tensors.shiftedPrevMix, row * Kernel::kHeadCount,
                                      rows * Kernel::kHeadCount);
            SetFlag<HardEvent::MTE2_V>(kShiftedPreInputEvent);
        }
    }

    /** The current pre plane may be regenerated only after its previous MTE3 reader completes. */
    __aicore__ inline void BeforeOverwrite()
    {
        if (outputPending_) {
            WaitFlag<HardEvent::MTE3_V>(kCurrentPreOutputEvent);
            outputPending_ = false;
        }
    }

    __aicore__ inline void StoreCurrent(Kernel::Tensors<T>& tensors, uint64_t row, uint32_t rows,
                                        const LocalTensor<float>& current)
    {
        if (enabled_) {
            SetFlag<HardEvent::V_MTE3>(kCurrentPreOutputEvent);
            WaitFlag<HardEvent::V_MTE3>(kCurrentPreOutputEvent);
            DataCopyExtParams copy{1U, static_cast<uint32_t>(rows * Kernel::kHeadCount * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(tensors.hPreNew[row * Kernel::kHeadCount], current, copy);
            SetFlag<HardEvent::MTE3_V>(kCurrentPreOutputEvent);
            outputPending_ = true;
        }
    }

    __aicore__ inline void WaitReady()
    {
        if (enabled_) {
            WaitFlag<HardEvent::MTE2_V>(kShiftedPreInputEvent);
        }
    }

    __aicore__ inline const LocalTensor<float>& Select(const LocalTensor<float>& current) const
    {
        return enabled_ ? shifted_ : current;
    }

    __aicore__ inline void End()
    {
        BeforeOverwrite();
    }

private:
    TBuf<TPosition::VECCALC> shiftedBuffer_;
    LocalTensor<float> shifted_;
    bool enabled_ = false;
    bool outputPending_ = false;
};

/** Load consecutive rows from each projection group; UB layout is [group, planned rows, stride]. */
template <typename T, bool WAIT_READY = true>
__aicore__ inline void LoadPrePartialRows(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, uint64_t row,
                                          uint32_t rows, const LocalTensor<float>& partialBuffer)
{
    const uint64_t gap = static_cast<uint64_t>(tiling.totalRows - rows) * tiling.nzN * sizeof(float);
    if (gap <= kMaxDmaGapBytes) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(tiling.partialGroupCount), static_cast<uint32_t>(rows * tiling.nzN * sizeof(float)),
            static_cast<uint32_t>(gap), (tiling.reduceRows - rows) * tiling.nzN / kNzRowAlignment, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(partialBuffer, tensors.partialMm[row * tiling.nzN], copyParams, padParams);
        if constexpr (WAIT_READY) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    } else {
        for (uint32_t group = 0; group < tiling.partialGroupCount; ++group) {
            Kernel::CopyGlobalToUb<float>(partialBuffer[group * tiling.reduceRows * tiling.nzN], tensors.partialMm,
                                          (static_cast<uint64_t>(group) * tiling.totalRows + row) * tiling.nzN,
                                          rows * tiling.nzN);
        }
    }
}

/** Load all RMS shards once per row batch; each shard retains its increasing-index ownership. */
template <typename T, bool ASYNC_COEFFICIENTS = false>
__aicore__ inline void PrepareEpilogueRows(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, uint64_t row,
                                           uint32_t rows, const LocalTensor<float>& partial,
                                           const LocalTensor<float>& rms, const LocalTensor<float>& coefficients,
                                           const LocalTensor<float>& alpha, const LocalTensor<float>& bias,
                                           PreMixStorage<T>& preMix)
{
    preMix.Prefetch(tensors, row, rows);
    LoadPrePartialRows<T, false>(tensors, tiling, row, rows, partial);
    const uint64_t gap = static_cast<uint64_t>(tiling.totalRows - rows) * sizeof(float);
    if (gap <= kMaxDmaGapBytes) {
        // One DMA covers all K shards; destination rows retain the host-planned aligned stride.
        const uint32_t paddedRows = (rows + kNzRowRounding) / kNzRowAlignment * kNzRowAlignment;
        DataCopyExtParams params{static_cast<uint16_t>(tiling.kGrid), static_cast<uint32_t>(rows * sizeof(float)),
                                 static_cast<uint32_t>(gap), (tiling.reduceRowsAligned - paddedRows) / kNzRowAlignment,
                                 0U};
        DataCopyPadExtParams<float> padding{true, 0U, static_cast<uint8_t>(paddedRows - rows), 0.0f};
        DataCopyPad(rms, tensors.partialRms[row], params, padding);
    } else {
        for (uint32_t shard = 0; shard < tiling.kGrid; ++shard) {
            Kernel::CopyGlobalToUb<float>(rms[shard * tiling.reduceRowsAligned], tensors.partialRms,
                                          static_cast<uint64_t>(shard) * tiling.totalRows + row, rows);
        }
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Compute::NormalizeProjectionRows(reinterpret_cast<__ubuf__ float*>(partial.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(rms.GetPhyAddr()), rows, tiling.nzN,
                                     tiling.reduceRows * tiling.nzN, tiling.partialGroupCount, tiling.reduceRowsAligned,
                                     tiling.kGrid, tiling.k, tiling.mhcPreNormEps);
    preMix.BeforeOverwrite();
    Compute::ComputeGateRows(
        reinterpret_cast<__ubuf__ float*>(partial.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(alpha.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(bias.GetPhyAddr()), tiling.hcEps, tiling.hcPostScale, rows, tiling.nzN,
        tiling.reduceRowsAligned, reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()));
    if constexpr (ASYNC_COEFFICIENTS) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        DataCopyExtParams postParams{1U, static_cast<uint32_t>(rows * Kernel::kHeadCount * sizeof(float)), 0U, 0U, 0U};
        DataCopyExtParams resParams{1U, static_cast<uint32_t>(rows * Kernel::kHeadMatrixElements * sizeof(float)), 0U,
                                    0U, 0U};
        DataCopyPad(tensors.hPostNew[row * Kernel::kHeadCount],
                    coefficients[tiling.reduceRowsAligned * Kernel::kHeadCount], postParams);
        DataCopyPad(tensors.hResNew[row * Kernel::kHeadMatrixElements],
                    coefficients[tiling.reduceRowsAligned * kPackedHResPlaneOffset], resParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
    } else {
        Kernel::CopyUbToGlobal<float>(tensors.hPostNew, row * Kernel::kHeadCount,
                                      coefficients[tiling.reduceRowsAligned * Kernel::kHeadCount],
                                      rows * Kernel::kHeadCount);
        Kernel::CopyUbToGlobal<float>(tensors.hResNew, row * Kernel::kHeadMatrixElements,
                                      coefficients[tiling.reduceRowsAligned * kPackedHResPlaneOffset],
                                      rows * Kernel::kHeadMatrixElements);
    }
    preMix.StoreCurrent(tensors, row, rows, coefficients);
    preMix.WaitReady();
}

/** Mix x2 using the selected internal or external pre coefficients, then round h_in and apply RMSNorm1.
 * Formula: output=RMSNorm1(Cast<T>(sum_n pre[n]*FP32(x2[n]))). */
template <typename T>
__aicore__ inline void FinalizeMhcRow(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, uint64_t row,
                                      const LocalTensor<T>& x2Row, const LocalTensor<T>& hIn,
                                      const LocalTensor<T>& hInPrime, const LocalTensor<float>& gamma1,
                                      const LocalTensor<float>& pre, const LocalTensor<float>& statsBuffer)
{
    __ubuf__ float* stats = reinterpret_cast<__ubuf__ float*>(statsBuffer.GetPhyAddr());
    Kernel::CopyGlobalToUb<T>(x2Row, tensors.x2Out, row * tiling.k, tiling.k);
    Compute::ComputeHInTile<T>(reinterpret_cast<__ubuf__ float*>(pre.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ T*>(x2Row.GetPhyAddr()), tiling.d, 0U, tiling.d,
                               reinterpret_cast<__ubuf__ T*>(hIn.GetPhyAddr()), stats);
    Compute::ComputeRmsNormFromSquares<T>(reinterpret_cast<__ubuf__ T*>(hIn.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(gamma1.GetPhyAddr()), tiling.d,
                                          tiling.normEps, stats, reinterpret_cast<__ubuf__ T*>(hInPrime.GetPhyAddr()));
    Kernel::CopyUbToGlobal<T>(tensors.hInPrime, row * tiling.d, hInPrime, tiling.d);
}

/** Issue one complete x2 row load into an available ping-pong slot; x2[r,n,d] stays in input dtype. */
template <typename T>
__aicore__ inline void PrefetchEpilogueRow(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, uint64_t row,
                                           const LocalTensor<T>& buffer, event_t event)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopyExtParams params{1, static_cast<uint32_t>(tiling.k * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, 0};
    DataCopyPad(buffer, tensors.x2Out[row * tiling.k], params, padding);
    SetFlag<HardEvent::MTE2_V>(event);
}

/** Finalize a selected-coefficient batch while overlapping next-row loads and previous-row output stores.
 * pre is shifted_prev_mix when present, otherwise the current sigmoid-plus-eps pre coefficients.
 * Formula: y[r]=RMSNorm1(Cast<T>(sum_n pre[r,n]*FP32(x2[r,n]))); stage roundings remain.
 * The host enables reuse of dead partial storage only when it fits a complete x2 row.
 */
template <typename T>
__aicore__ inline void FinalizeMhcRows(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, uint64_t start,
                                       uint32_t rows, const LocalTensor<T>& x2, const LocalTensor<T>& hIn,
                                       const LocalTensor<T>& hInPrime, const LocalTensor<float>& gamma,
                                       const LocalTensor<float>& partial, const LocalTensor<float>& coefficients,
                                       const LocalTensor<float>& statsBuffer)
{
    if (tiling.epiloguePrefetch == 0U) {
        for (uint32_t row = 0; row < rows; ++row) {
            FinalizeMhcRow(tensors, tiling, start + row, x2, hIn, hInPrime, gamma,
                           coefficients[row * Kernel::kHeadCount], statsBuffer);
        }
        return;
    }
    LocalTensor<T> alternate = partial.template ReinterpretCast<T>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
    PrefetchEpilogueRow(tensors, tiling, start, x2, EVENT_ID4);
    __ubuf__ float* stats = reinterpret_cast<__ubuf__ float*>(statsBuffer.GetPhyAddr());
    for (uint32_t row = 0; row < rows; ++row) {
        const event_t event = (row & 1U) == 0U ? EVENT_ID4 : EVENT_ID5;
        const event_t nextEvent = (row & 1U) == 0U ? EVENT_ID5 : EVENT_ID4;
        const LocalTensor<T> input = (row & 1U) == 0U ? x2 : alternate;
        const LocalTensor<T> next = (row & 1U) == 0U ? alternate : x2;
        const LocalTensor<T> output = (row & 1U) == 0U ? hIn : hInPrime;
        WaitFlag<HardEvent::MTE2_V>(event);
        if (row + 1U < rows) {
            PrefetchEpilogueRow(tensors, tiling, start + row + 1U, next, nextEvent);
        }
        WaitFlag<HardEvent::MTE3_V>(event);
        Compute::ComputeHInTile<T>(
            reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()) + row * Kernel::kHeadCount,
            reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()), tiling.d, 0U, tiling.d,
            reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), stats);
        SetFlag<HardEvent::V_MTE2>(event);
        Compute::ComputeRmsNormFromSquares<T>(
            reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()),
            tiling.d, tiling.normEps, stats, reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopyExtParams params{1, static_cast<uint32_t>(tiling.d * sizeof(T)), 0, 0, 0};
        DataCopyPad(tensors.hInPrime[(start + row) * tiling.d], output, params);
        SetFlag<HardEvent::MTE3_V>(event);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
}

/** Assign complete final rows to AIVs after all Cube and statistic partials are globally visible. */
template <typename T, bool BS_OWNERSHIP = false, bool MERGE_PARAMETER_LOADS = false, bool ASYNC_COEFFICIENTS = false,
          bool SKIP_IDLE_AIV = false>
__aicore__ inline void RunEpilogue(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
{
    if ASCEND_IS_AIV {
        if constexpr (SKIP_IDLE_AIV) {
            if (static_cast<uint64_t>(GetBlockIdx()) * tiling.reduceRows >= tiling.totalRows) {
                return;
            }
        }
        TBuf<TPosition::VECCALC> x2Buffer;
        TBuf<TPosition::VECCALC> hInBuffer;
        TBuf<TPosition::VECCALC> hInPrimeBuffer;
        TBuf<TPosition::VECCALC> gamma1Buffer;
        TBuf<TPosition::VECCALC> partialBuffer;
        TBuf<TPosition::VECCALC> rmsPartialBuffer;
        TBuf<TPosition::VECCALC> coefficientBuffer;
        TBuf<TPosition::VECCALC> alphaBuffer;
        TBuf<TPosition::VECCALC> biasBuffer;
        TBuf<TPosition::VECCALC> statsBuffer;
        PreMixStorage<T> preMix;
        Kernel::InitUbBuffer(pipe, x2Buffer, tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInPrimeBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, gamma1Buffer, tiling.d * sizeof(float));
        uint32_t partialBytes =
            (tiling.partialGroupCount * tiling.reduceRows * tiling.nzN + kPartialBufferGuardElements) * sizeof(float);
        if (tiling.epiloguePrefetch == kEpiloguePrefetchFull && partialBytes < tiling.k * sizeof(T)) {
            partialBytes = tiling.k * sizeof(T);
        }
        Kernel::InitUbBuffer(pipe, partialBuffer, partialBytes);
        Kernel::InitUbBuffer(pipe, rmsPartialBuffer, tiling.reduceRowsAligned * tiling.kGrid * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer,
                             tiling.reduceRowsAligned * Compute::kProjectionCoefficientElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, alphaBuffer, kAlphaBufferElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, biasBuffer, kBiasBufferElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, kEpilogueStatsElements * sizeof(float));
        preMix.Init(pipe, tiling, tensors.hasShiftedPrevMix);

        LocalTensor<T> x2 = x2Buffer.Get<T>();
        LocalTensor<T> hIn = hInBuffer.Get<T>();
        LocalTensor<T> hInPrime = hInPrimeBuffer.Get<T>();
        LocalTensor<float> gamma1 = gamma1Buffer.Get<float>();
        LocalTensor<float> partial = partialBuffer.Get<float>();
        LocalTensor<float> rmsPartials = rmsPartialBuffer.Get<float>();
        LocalTensor<float> coefficients = coefficientBuffer.Get<float>();
        LocalTensor<float> alpha = alphaBuffer.Get<float>();
        LocalTensor<float> bias = biasBuffer.Get<float>();
        LocalTensor<float> stats = statsBuffer.Get<float>();
        if constexpr (MERGE_PARAMETER_LOADS) {
            Kernel::EnqueueGlobalToUb(gamma1, tensors.gamma1, 0U, tiling.d);
            Kernel::EnqueueGlobalToUb(alpha, tensors.alpha, 0U, kAlphaElements);
            Kernel::EnqueueGlobalToUb(bias, tensors.bias, 0U, kProjectionColumns);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            Kernel::CopyGlobalToUb<float>(gamma1, tensors.gamma1, 0, tiling.d);
            Kernel::CopyGlobalToUb<float>(alpha, tensors.alpha, 0, kAlphaElements);
            Kernel::CopyGlobalToUb<float>(bias, tensors.bias, 0, kProjectionColumns);
        }

        const uint32_t aivIndex = GetBlockIdx();
        const uint32_t waves = BS_OWNERSHIP ? tiling.mWaveCount : 1U;
        for (uint32_t wave = 0U; wave < waves; ++wave) {
            // BS consumes exactly the rows produced/projected by this paired AIV.
            const uint64_t first = BS_OWNERSHIP ?
                                       static_cast<uint64_t>(wave) * tiling.rowsPerWave +
                                           static_cast<uint64_t>(aivIndex / Kernel::kAivPerAic) * tiling.mTile +
                                           static_cast<uint64_t>(GetSubBlockIdx()) * tiling.projectionRows :
                                       static_cast<uint64_t>(aivIndex) * tiling.reduceRows;
            const uint64_t ownedEnd = first + tiling.projectionRows;
            const uint64_t end = BS_OWNERSHIP && ownedEnd < tiling.totalRows ? ownedEnd : tiling.totalRows;
            const uint64_t stride =
                BS_OWNERSHIP ? tiling.reduceRows : static_cast<uint64_t>(tiling.activeAiv) * tiling.reduceRows;
            for (uint64_t start = first; start < end; start += stride) {
                const uint32_t rows =
                    end - start < tiling.reduceRows ? static_cast<uint32_t>(end - start) : tiling.reduceRows;
                PrepareEpilogueRows<T, BS_OWNERSHIP || ASYNC_COEFFICIENTS>(
                    tensors, tiling, start, rows, partial, rmsPartials, coefficients, alpha, bias, preMix);
                FinalizeMhcRows<T>(tensors, tiling, start, rows, x2, hIn, hInPrime, gamma1, partial,
                                   preMix.Select(coefficients), stats);
                if constexpr (BS_OWNERSHIP || ASYNC_COEFFICIENTS) {
                    // Finalize reads the separate Pre plane; drain Post/Res output reads before reusing coefficients.
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
                }
            }
        }
        preMix.End();
    }
}

template <typename T>
class WaveEpilogue {
public:
    /** Allocate the host-budgeted epilogue state and load immutable parameters once. */
    __aicore__ inline void Init(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& sourceTiling, TPipe* pipe)
    {
        plan_ = &sourceTiling;
        const auto& tiling = *plan_;
        Kernel::InitUbBuffer(pipe, x2Buffer, tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInPrimeBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, gamma1Buffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(
            pipe, partialBuffer,
            (tiling.partialGroupCount * tiling.reduceRows * tiling.nzN + kPartialBufferGuardElements) * sizeof(float));
        Kernel::InitUbBuffer(pipe, rmsPartialBuffer, tiling.reduceRowsAligned * tiling.kGrid * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer,
                             tiling.reduceRowsAligned * Compute::kProjectionCoefficientElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, alphaBuffer, kAlphaBufferElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, biasBuffer, kBiasBufferElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, kEpilogueStatsElements * sizeof(float));
        preMix_.Init(pipe, tiling, tensors.hasShiftedPrevMix);

        x2 = x2Buffer.Get<T>();
        hIn = hInBuffer.Get<T>();
        hInPrime = hInPrimeBuffer.Get<T>();
        gamma1 = gamma1Buffer.Get<float>();
        partial = partialBuffer.Get<float>();
        rmsPartials = rmsPartialBuffer.Get<float>();
        coefficients = coefficientBuffer.Get<float>();
        alpha = alphaBuffer.Get<float>();
        bias = biasBuffer.Get<float>();
        stats = statsBuffer.Get<float>();
        // Immutable parameters have disjoint destinations: issue all loads before one readiness wait.
        Kernel::EnqueueGlobalToUb(gamma1, tensors.gamma1, 0U, tiling.d);
        Kernel::EnqueueGlobalToUb(alpha, tensors.alpha, 0U, kAlphaElements);
        Kernel::EnqueueGlobalToUb(bias, tensors.bias, 0U, kProjectionColumns);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    /** Bind complete rows of a globally completed wave; partials are read in host-sized batches. */
    __aicore__ inline void Begin(uint32_t wave)
    {
        const auto& t = *plan_;
        const uint32_t member = GetBlockIdx();
        first_ = static_cast<uint64_t>(wave) * t.rowsPerWave +
                 static_cast<uint64_t>(member / t.waveGroupSize) * t.mTile +
                 (member % t.waveGroupSize) * t.waveProducerRows;
        count_ = wave + 1U != t.mWaveCount || member < t.lastWaveFullProducers ?
                     t.waveProducerRows :
                     (member == t.lastWaveFullProducers ? t.lastWaveProducerTailRows : 0U);
        done_ = 0U;
        batchStart_ = 0U;
        batchRows_ = 0U;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
    }

    /** Generate current gates, select external pre coefficients when present, then finish one row with RMSNorm1. */
    __aicore__ inline void Tick(Kernel::Tensors<T>& tensors)
    {
        if (done_ >= count_) {
            return;
        }
        if (done_ == batchStart_ + batchRows_) {
            batchStart_ = done_;
            batchRows_ = count_ - done_ < plan_->reduceRows ? count_ - done_ : plan_->reduceRows;
            PrepareEpilogueRows(tensors, *plan_, first_ + done_, batchRows_, partial, rmsPartials, coefficients, alpha,
                                bias, preMix_);
        }
        if (done_ == 0U) {
            PrefetchEpilogueRow(tensors, *plan_, first_, x2, EVENT_ID1);
        }
        const event_t outputEvent = (done_ & 1U) == 0U ? EVENT_ID4 : EVENT_ID5;
        const auto output = (done_ & 1U) == 0U ? hIn : hInPrime;
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(outputEvent);
        auto statsPtr = reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr());
        Compute::ComputeHInTile<T>(reinterpret_cast<__ubuf__ float*>(preMix_.Select(coefficients).GetPhyAddr()) +
                                       (done_ - batchStart_) * Kernel::kHeadCount,
                                   reinterpret_cast<__ubuf__ T*>(x2.GetPhyAddr()), plan_->d, 0U, plan_->d,
                                   reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), statsPtr);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        // The next x2 load overlaps RMS1 and the following projection work; input is no longer read by RMS1.
        if (done_ + 1U < count_) {
            PrefetchEpilogueRow(tensors, *plan_, first_ + done_ + 1U, x2, EVENT_ID1);
        }
        Compute::ComputeRmsNormFromSquares<T>(
            reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(gamma1.GetPhyAddr()),
            plan_->d, plan_->normEps, statsPtr, reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()));
        SetFlag<HardEvent::V_MTE3>(outputEvent);
        WaitFlag<HardEvent::V_MTE3>(outputEvent);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(plan_->d * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(tensors.hInPrime[(first_ + done_) * plan_->d], output, copy);
        SetFlag<HardEvent::MTE3_V>(outputEvent);
        ++done_;
    }

    /** Drain any remaining rows before reusing the epilogue state or leaving the final wave. */
    __aicore__ inline void Finish(Kernel::Tensors<T>& tensors)
    {
        while (done_ < count_) {
            Tick(tensors);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
        preMix_.End();
    }

private:
    PreMixStorage<T> preMix_;
    const MegaMhcTilingData* plan_ = nullptr;
    uint64_t first_;
    uint32_t count_, done_, batchStart_, batchRows_;
    TBuf<TPosition::VECCALC> x2Buffer, hInBuffer, hInPrimeBuffer, gamma1Buffer, partialBuffer;
    TBuf<TPosition::VECCALC> rmsPartialBuffer, coefficientBuffer, alphaBuffer, biasBuffer, statsBuffer;
    LocalTensor<T> x2, hIn, hInPrime;
    LocalTensor<float> gamma1, partial, rmsPartials, coefficients, alpha, bias, stats;
};

/** Generate ordered current coefficients, select the internal/external pre plane, then stream hIn through RMS1.
 * Formula: hIn=Cast<T>(sum_n pre[n]*FP32(x2[n,:])); y=RMSNorm1(hIn).
 * y temporarily holds the rounded hIn until its complete-D statistic is known; token ownership prevents races.
 */
template <typename T>
__aicore__ inline void RunTiledEpilogue(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
{
    if ASCEND_IS_AIV {
        Kernel::TiledRowBuffers<T> buffer;
        buffer.Init(pipe, tiling);
        TBuf<TPosition::VECCALC> partialBuffer, rmsBuffer, alphaBuffer, biasBuffer;
        Kernel::InitUbBuffer(pipe, partialBuffer,
                             (tiling.partialGroupCount * tiling.nzN + kPartialBufferGuardElements) * sizeof(float));
        Kernel::InitUbBuffer(pipe, rmsBuffer, kRmsGridElements * tiling.kGrid * sizeof(float));
        Kernel::InitUbBuffer(pipe, alphaBuffer, kAlphaBufferElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, biasBuffer, kBiasBufferElements * sizeof(float));
        auto partial = partialBuffer.Get<float>();
        auto rms = rmsBuffer.Get<float>();
        auto alpha = alphaBuffer.Get<float>();
        auto bias = biasBuffer.Get<float>();
        PreMixStorage<T> preMix;
        preMix.Init(pipe, tiling, tensors.hasShiftedPrevMix);
        auto stats = reinterpret_cast<__ubuf__ float*>(buffer.stats.GetPhyAddr());
        Kernel::CopyGlobalToUb(alpha, tensors.alpha, 0U, kAlphaElements);
        Kernel::CopyGlobalToUb(bias, tensors.bias, 0U, kProjectionColumns);
        for (uint64_t row = GetBlockIdx(); row < tiling.totalRows; row += tiling.activeAiv) {
            PrepareEpilogueRows(tensors, tiling, row, 1U, partial, rms, buffer.coefficients, alpha, bias, preMix);
            Kernel::ZeroFp32Values(stats, 1U);
            buffer.template PrefetchHeads<false>(tensors, row, 0U);
            for (uint32_t tile = 0U; tile < tiling.dTileCount; ++tile) {
                buffer.Wait(tile);
                if (tile + 1U < tiling.dTileCount) {
                    buffer.template PrefetchHeads<false>(tensors, row, tile + 1U);
                }
                auto input = buffer.Input(tile);
                auto heads = buffer.Heads(tile);
                Compute::ComputeHInTile<T>(
                    reinterpret_cast<__ubuf__ float*>(preMix.Select(buffer.coefficients).GetPhyAddr()),
                    reinterpret_cast<__ubuf__ T*>(heads.GetPhyAddr()), tiling.dTile, 0U, buffer.Count(tile),
                    reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()), stats + kEpilogueRmsCarryIndex);
                Compute::AddFp32Scalar(stats, stats + kEpilogueRmsCarryIndex);
                Kernel::CopyUbToGlobal(tensors.hInPrime, row * tiling.d + static_cast<uint64_t>(tile) * tiling.dTile,
                                       input, buffer.Count(tile));
                buffer.Release(tile);
            }
            Compute::FinalizeInverseRms(stats, tiling.d, tiling.normEps, stats + 1U);
            Kernel::ApplyTiledNormalization(buffer, tiling, tensors.hInPrime, tensors.hInPrime, row * tiling.d,
                                            tensors.gamma1, 0U, stats + 1U);
        }
        preMix.End();
        buffer.End();
    }
}

template <typename T>
__aicore__ inline void PrepareBsCompactEpilogueRows(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                                    uint64_t row, uint32_t rows, const LocalTensor<float>& partial,
                                                    const LocalTensor<float>& rms,
                                                    const LocalTensor<float>& coefficients,
                                                    const LocalTensor<float>& alpha, const LocalTensor<float>& bias,
                                                    PreMixStorage<T>& preMix)
{
    preMix.Prefetch(tensors, row, rows);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    Kernel::EnqueueGlobalToUb(partial, tensors.partialMm, row * tiling.nzN, rows * tiling.nzN);
    const uint64_t gap = static_cast<uint64_t>(tiling.totalRows - rows) * sizeof(float);
    if (gap <= kMaxDmaGapBytes) {
        // One DMA covers all K shards; destination rows retain the host-planned aligned stride.
        const uint32_t paddedRows = (rows + kNzRowRounding) / kNzRowAlignment * kNzRowAlignment;
        DataCopyExtParams params{static_cast<uint16_t>(tiling.kGrid), static_cast<uint32_t>(rows * sizeof(float)),
                                 static_cast<uint32_t>(gap), (tiling.reduceRowsAligned - paddedRows) / kNzRowAlignment,
                                 0U};
        DataCopyPadExtParams<float> padding{true, 0U, static_cast<uint8_t>(paddedRows - rows), 0.0f};
        DataCopyPad(rms, tensors.partialRms[row], params, padding);
    } else {
        for (uint32_t shard = 0; shard < tiling.kGrid; ++shard) {
            Kernel::CopyGlobalToUb<float>(rms[shard * tiling.reduceRowsAligned], tensors.partialRms,
                                          static_cast<uint64_t>(shard) * tiling.totalRows + row, rows);
        }
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Compute::NormalizeProjectionRows(reinterpret_cast<__ubuf__ float*>(partial.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(rms.GetPhyAddr()), rows, tiling.nzN,
                                     tiling.reduceRows * tiling.nzN, 1U, tiling.reduceRowsAligned, tiling.kGrid,
                                     tiling.k, tiling.mhcPreNormEps);
    preMix.BeforeOverwrite();
    Compute::ComputeGateRows(
        reinterpret_cast<__ubuf__ float*>(partial.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(alpha.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(bias.GetPhyAddr()), tiling.hcEps, tiling.hcPostScale, rows, tiling.nzN,
        tiling.reduceRowsAligned, reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()));
    // Both disjoint coefficient ranges share one Vector-to-MTE3 handoff and completion.
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams postParams{1U, static_cast<uint32_t>(rows * Kernel::kHeadCount * sizeof(float)), 0U, 0U, 0U};
    DataCopyExtParams resParams{1U, static_cast<uint32_t>(rows * Kernel::kHeadMatrixElements * sizeof(float)), 0U, 0U,
                                0U};
    DataCopyPad(tensors.hPostNew[row * Kernel::kHeadCount], coefficients[tiling.reduceRowsAligned * Kernel::kHeadCount],
                postParams);
    DataCopyPad(tensors.hResNew[row * Kernel::kHeadMatrixElements],
                coefficients[tiling.reduceRowsAligned * kPackedHResPlaneOffset], resParams);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    preMix.StoreCurrent(tensors, row, rows, coefficients);
    preMix.WaitReady();
}

/** Pipeline the single-input epilogue when the partial buffer cannot hold another x2 row.
 * Release x2 after HIn consumes it, prefetch the next row during RMS1, and ping-pong output stores.
 */
template <typename T>
__aicore__ inline void FinalizeBsMhcRows(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, uint64_t start,
                                         uint32_t rows, const LocalTensor<T>& x2, const LocalTensor<T>& hIn,
                                         const LocalTensor<T>& hInPrime, const LocalTensor<float>& gamma,
                                         const LocalTensor<float>& partial, const LocalTensor<float>& coefficients,
                                         const LocalTensor<float>& statsBuffer)
{
    if (tiling.epiloguePrefetch != 0U) {
        FinalizeMhcRows(tensors, tiling, start, rows, x2, hIn, hInPrime, gamma, partial, coefficients, statsBuffer);
        return;
    }
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID5);
    PrefetchEpilogueRow(tensors, tiling, start, x2, EVENT_ID1);
    auto stats = reinterpret_cast<__ubuf__ float*>(statsBuffer.GetPhyAddr());
    for (uint32_t row = 0U; row < rows; ++row) {
        const event_t event = (row & 1U) == 0U ? EVENT_ID4 : EVENT_ID5;
        const auto output = (row & 1U) == 0U ? hIn : hInPrime;
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(event);
        Compute::ComputeHInTile<T>(
            reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()) + row * Kernel::kHeadCount,
            reinterpret_cast<__ubuf__ T*>(x2.GetPhyAddr()), tiling.d, 0U, tiling.d,
            reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), stats);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        if (row + 1U < rows) {
            PrefetchEpilogueRow(tensors, tiling, start + row + 1U, x2, EVENT_ID1);
        }
        Compute::ComputeRmsNormFromSquares<T>(
            reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()),
            tiling.d, tiling.normEps, stats, reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(tiling.d * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(tensors.hInPrime[(start + row) * tiling.d], output, copy);
        SetFlag<HardEvent::MTE3_V>(event);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID5);
}

// Input and output credits are independent; Host owns the batch and UB capacity plan.
namespace BsHfEpilogue {
constexpr event_t kInput0 = EVENT_ID4;
constexpr event_t kInput1 = EVENT_ID5;
constexpr event_t kOutput0 = EVENT_ID6;
constexpr event_t kOutput1 = EVENT_ID7;
} // namespace BsHfEpilogue

/** Prefetch rows*K rounded x2 elements into a Host-sized input slot; MTE2_V publishes readiness. */
template <typename T>
__aicore__ inline void PrefetchBsHfEpilogueBatch(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& t, uint64_t row,
                                                 uint32_t rows, const LocalTensor<T>& input, event_t event)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopyExtParams copy{1U, static_cast<uint32_t>(rows * t.k * sizeof(T)), 0U, 0U, 0U};
    DataCopyPadExtParams<T> pad{false, 0U, 0U, 0};
    DataCopyPad(input, tensors.x2Out[row * t.k], copy, pad);
    SetFlag<HardEvent::MTE2_V>(event);
}

/** Ping-pong input batches independently of the two row-output buffers. */
template <typename T>
__aicore__ inline void FinalizeBsHfEpilogueBatches(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& t,
                                                   uint64_t start, uint32_t rows, uint32_t batch,
                                                   const LocalTensor<T>& input, const LocalTensor<T>& hIn,
                                                   const LocalTensor<T>& hInPrime, const LocalTensor<float>& gamma,
                                                   const LocalTensor<float>& coefficients,
                                                   const LocalTensor<float>& statsBuffer)
{
    SetFlag<HardEvent::V_MTE2>(BsHfEpilogue::kInput0);
    SetFlag<HardEvent::V_MTE2>(BsHfEpilogue::kInput1);
    SetFlag<HardEvent::MTE3_V>(BsHfEpilogue::kOutput0);
    SetFlag<HardEvent::MTE3_V>(BsHfEpilogue::kOutput1);
    const uint32_t firstCount = rows < batch ? rows : batch;
    PrefetchBsHfEpilogueBatch(tensors, t, start, firstCount, input, BsHfEpilogue::kInput0);
    auto stats = reinterpret_cast<__ubuf__ float*>(statsBuffer.GetPhyAddr());
    for (uint32_t done = 0U; done < rows; done += batch) {
        const uint32_t inputSlot = (done / batch) & 1U;
        const event_t inputEvent = inputSlot == 0U ? BsHfEpilogue::kInput0 : BsHfEpilogue::kInput1;
        const event_t nextInputEvent = inputSlot == 0U ? BsHfEpilogue::kInput1 : BsHfEpilogue::kInput0;
        const uint32_t count = rows - done < batch ? rows - done : batch;
        WaitFlag<HardEvent::MTE2_V>(inputEvent);
        if (done + batch < rows) {
            const uint32_t nextCount = rows - done - batch < batch ? rows - done - batch : batch;
            PrefetchBsHfEpilogueBatch(tensors, t, start + done + batch, nextCount,
                                      input[(inputSlot ^ 1U) * batch * t.k], nextInputEvent);
        }
        for (uint32_t batchRow = 0U; batchRow < count; ++batchRow) {
            const uint32_t row = done + batchRow;
            const event_t outputEvent = (row & 1U) == 0U ? BsHfEpilogue::kOutput0 : BsHfEpilogue::kOutput1;
            const auto output = (row & 1U) == 0U ? hIn : hInPrime;
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            Compute::ComputeHInTile<T>(
                reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()) + row * Kernel::kHeadCount,
                reinterpret_cast<__ubuf__ T*>(input[(inputSlot * batch + batchRow) * t.k].GetPhyAddr()), t.d, 0U, t.d,
                reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), stats);
            Compute::ComputeRmsNormFromSquares<T>(reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()), t.d, t.normEps,
                                                  stats, reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()));
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            DataCopyExtParams copy{1U, static_cast<uint32_t>(t.d * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(tensors.hInPrime[(start + row) * t.d], output, copy);
            SetFlag<HardEvent::MTE3_V>(outputEvent);
        }
        // All HIn reads from this batch are complete; RMS1 only reads the row-output buffers.
        SetFlag<HardEvent::V_MTE2>(inputEvent);
    }
    WaitFlag<HardEvent::V_MTE2>(BsHfEpilogue::kInput0);
    WaitFlag<HardEvent::V_MTE2>(BsHfEpilogue::kInput1);
    WaitFlag<HardEvent::MTE3_V>(BsHfEpilogue::kOutput0);
    WaitFlag<HardEvent::MTE3_V>(BsHfEpilogue::kOutput1);
}

/** Preserve projection ownership through the epilogue; full K requires only the paired Cube. */
template <typename T, bool COMPACT_PARTIALS = false, bool BATCHED_INPUT = false>
__aicore__ inline void RunBsOwnedEpilogue(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe,
                                          uint32_t batch = 0U)
{
    static_assert(!BATCHED_INPUT || COMPACT_PARTIALS, "Batched input requires the HF32 compact BS no_mid epilogue");
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> x2Buffer;
        TBuf<TPosition::VECCALC> hInBuffer;
        TBuf<TPosition::VECCALC> hInPrimeBuffer;
        TBuf<TPosition::VECCALC> gamma1Buffer;
        TBuf<TPosition::VECCALC> partialBuffer;
        TBuf<TPosition::VECCALC> rmsPartialBuffer;
        TBuf<TPosition::VECCALC> coefficientBuffer;
        TBuf<TPosition::VECCALC> alphaBuffer;
        TBuf<TPosition::VECCALC> biasBuffer;
        TBuf<TPosition::VECCALC> statsBuffer;
        PreMixStorage<T> preMix;
        Kernel::InitUbBuffer(pipe, x2Buffer, BATCHED_INPUT ? tiling.bsEpilogueInputBytes : tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInPrimeBuffer, tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, gamma1Buffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(
            pipe, partialBuffer,
            BATCHED_INPUT ? (tiling.reduceRows * tiling.nzN + kPartialBufferGuardElements) * sizeof(float) :
                            (tiling.partialGroupCount * tiling.reduceRows * tiling.nzN + kPartialBufferGuardElements) *
                                sizeof(float));
        Kernel::InitUbBuffer(pipe, rmsPartialBuffer, tiling.reduceRowsAligned * tiling.kGrid * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer,
                             tiling.reduceRowsAligned * Compute::kProjectionCoefficientElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, alphaBuffer, kAlphaBufferElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, biasBuffer, kBiasBufferElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, kEpilogueStatsElements * sizeof(float));
        preMix.Init(pipe, tiling, tensors.hasShiftedPrevMix);

        LocalTensor<T> x2 = x2Buffer.Get<T>();
        LocalTensor<T> hIn = hInBuffer.Get<T>();
        LocalTensor<T> hInPrime = hInPrimeBuffer.Get<T>();
        LocalTensor<float> gamma1 = gamma1Buffer.Get<float>();
        LocalTensor<float> partial = partialBuffer.Get<float>();
        LocalTensor<float> rmsPartials = rmsPartialBuffer.Get<float>();
        LocalTensor<float> coefficients = coefficientBuffer.Get<float>();
        LocalTensor<float> alpha = alphaBuffer.Get<float>();
        LocalTensor<float> bias = biasBuffer.Get<float>();
        LocalTensor<float> stats = statsBuffer.Get<float>();
        if constexpr (COMPACT_PARTIALS) {
            // Immutable parameters occupy independent fresh buffers; load them under one handoff.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            Kernel::EnqueueGlobalToUb<float>(gamma1, tensors.gamma1, 0, tiling.d);
            Kernel::EnqueueGlobalToUb<float>(alpha, tensors.alpha, 0, kAlphaElements);
            Kernel::EnqueueGlobalToUb<float>(bias, tensors.bias, 0, kProjectionColumns);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            Kernel::CopyGlobalToUb<float>(gamma1, tensors.gamma1, 0, tiling.d);
            Kernel::CopyGlobalToUb<float>(alpha, tensors.alpha, 0, kAlphaElements);
            Kernel::CopyGlobalToUb<float>(bias, tensors.bias, 0, kProjectionColumns);
        }

        const uint32_t pair = GetBlockIdx() / Kernel::kAivPerAic;
        const uint32_t rowOffset = GetSubBlockIdx() * tiling.projectionRows;
        for (uint32_t wave = 0U; wave < tiling.mWaveCount; ++wave) {
            const uint32_t mRows = Kernel::GetWaveRows(tiling, pair, wave);
            const uint32_t count =
                mRows > rowOffset ?
                    (mRows - rowOffset < tiling.projectionRows ? mRows - rowOffset : tiling.projectionRows) :
                    0U;
            const uint64_t first = static_cast<uint64_t>(wave) * tiling.rowsPerWave +
                                   static_cast<uint64_t>(pair) * tiling.mTile + rowOffset;
            for (uint32_t done = 0U; done < count; done += tiling.reduceRows) {
                const uint64_t start = first + done;
                const uint32_t rows = count - done < tiling.reduceRows ? count - done : tiling.reduceRows;
                if constexpr (COMPACT_PARTIALS) {
                    PrepareBsCompactEpilogueRows(tensors, tiling, start, rows, partial, rmsPartials, coefficients,
                                                 alpha, bias, preMix);
                } else {
                    PrepareEpilogueRows(tensors, tiling, start, rows, partial, rmsPartials, coefficients, alpha, bias,
                                        preMix);
                }
                if constexpr (BATCHED_INPUT) {
                    FinalizeBsHfEpilogueBatches(tensors, tiling, start, rows, batch, x2, hIn, hInPrime, gamma1,
                                                preMix.Select(coefficients), stats);
                } else {
                    FinalizeBsMhcRows(tensors, tiling, start, rows, x2, hIn, hInPrime, gamma1, partial,
                                      preMix.Select(coefficients), stats);
                }
            }
        }
        preMix.End();
    }
}

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_EPILOGUE_H
