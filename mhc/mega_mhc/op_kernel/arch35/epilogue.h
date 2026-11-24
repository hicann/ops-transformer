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
constexpr uint32_t kEpilogueRmsInverseIndex = 1U;
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
        tiling.reduceRowsAligned, reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()), tiling.sinkhornEps,
        tiling.numSinkhornIters);
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

/** Prefetch a contiguous row batch into one input bank; output completion has a separate credit. */
template <typename T>
__aicore__ inline void PrefetchQuantEpilogueRows(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                                 uint64_t row, uint32_t rows, const LocalTensor<T>& input,
                                                 event_t event)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopyExtParams copy{1U, static_cast<uint32_t>(rows * tiling.k * sizeof(T)), 0U, 0U, 0U};
    DataCopyPadExtParams<T> padding{false, 0U, 0U, 0};
    DataCopyPad(input, tensors.x2Out[row * tiling.k], copy, padding);
    SetFlag<HardEvent::MTE2_V>(event);
}

/** Output credits live across reduction batches. Input/partial buffers retire independently.
 * Slot selection is continuous even when a reduction batch has an odd row count.
 */
class EpilogueOutputSlots {
public:
    __aicore__ inline void Begin(bool hfBatch = false)
    {
        first_ = hfBatch ? EVENT_ID6 : EVENT_ID4;
        second_ = hfBatch ? EVENT_ID7 : EVENT_ID5;
        slot_ = 0U;
        SetFlag<HardEvent::MTE3_V>(first_);
        SetFlag<HardEvent::MTE3_V>(second_);
    }

    __aicore__ inline uint32_t Slot() const
    {
        return slot_;
    }
    __aicore__ inline event_t Event() const
    {
        return slot_ == 0U ? first_ : second_;
    }
    __aicore__ inline void Publish()
    {
        SetFlag<HardEvent::MTE3_V>(Event());
        slot_ ^= 1U;
    }
    __aicore__ inline void End()
    {
        WaitFlag<HardEvent::MTE3_V>(first_);
        WaitFlag<HardEvent::MTE3_V>(second_);
    }

private:
    event_t first_, second_;
    uint32_t slot_;
};

/** HIn and RMS1 each loop over rows inside one VF; the three quant VFs traverse the complete batch.
 * One B16 compute bank plus two FP8 banks use the original two B16 allocations.
 * Input retires after HIn; only quantization waits for the FP8/scale slot credit.
 */
template <typename T>
__aicore__ inline void FinalizeQuantEpilogueBatches(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                                    uint64_t start, uint32_t rows, const LocalTensor<T>& x2,
                                                    const LocalTensor<T>& hIn, const LocalTensor<T>& hInPrime,
                                                    const LocalTensor<float>& gamma, const LocalTensor<float>& partial,
                                                    const LocalTensor<float>& coefficients,
                                                    const LocalTensor<float>& statsBuffer, EpilogueOutputSlots& outputs)
{
    const uint32_t batch = tiling.quantBatchRows;
    const auto alternate = partial.template ReinterpretCast<T>();
    auto stats = reinterpret_cast<__ubuf__ float*>(statsBuffer.GetPhyAddr());
    SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
    PrefetchQuantEpilogueRows(tensors, tiling, start, rows < batch ? rows : batch, x2, EVENT_ID4);
    for (uint32_t done = 0U; done < rows; done += batch) {
        const uint32_t slot = (done / batch) & 1U;
        const event_t event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
        const event_t nextEvent = slot == 0U ? EVENT_ID5 : EVENT_ID4;
        const auto input = slot == 0U ? x2 : alternate;
        const auto next = slot == 0U ? alternate : x2;
        const auto output = hIn;
        const uint32_t outputSlot = outputs.Slot();
        const auto quant = hInPrime.template ReinterpretCast<uint8_t>()[outputSlot * batch * tiling.d];
        const uint32_t count = rows - done < batch ? rows - done : batch;
        WaitFlag<HardEvent::MTE2_V>(event);
        if (done + batch < rows) {
            const uint32_t nextCount = rows - done - batch < batch ? rows - done - batch : batch;
            PrefetchQuantEpilogueRows(tensors, tiling, start + done + batch, nextCount, next, nextEvent);
        }
        Compute::ComputeHInTile<T>(
            reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()) + done * Kernel::kHeadCount,
            reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()), tiling.d, 0U, tiling.d,
            reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), stats, count);
        SetFlag<HardEvent::V_MTE2>(event);
        Compute::ComputeRmsNormRowsFromSquares<T>(reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()), tiling.d,
                                                  count, tiling.normEps, stats);
        WaitFlag<HardEvent::MTE3_V>(outputs.Event());
        Kernel::WriteSeparateQuantOutput(tensors, tiling, start + done, output, quant, count * tiling.d, outputSlot);
        outputs.Publish();
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
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
                                       const LocalTensor<float>& statsBuffer, EpilogueOutputSlots& outputs)
{
    if (tiling.quantBatchRows > 1U) {
        FinalizeQuantEpilogueBatches(tensors, tiling, start, rows, x2, hIn, hInPrime, gamma, partial, coefficients,
                                     statsBuffer, outputs);
        return;
    }
    const bool doubleInput = tiling.epiloguePrefetch != 0U;
    LocalTensor<T> alternate = partial.template ReinterpretCast<T>();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
    if (doubleInput) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
    }
    PrefetchEpilogueRow(tensors, tiling, start, x2, EVENT_ID4);
    __ubuf__ float* stats = reinterpret_cast<__ubuf__ float*>(statsBuffer.GetPhyAddr());
    const bool separateQuant = tiling.reduceRows > QuantOutput::kOutputSlots && Kernel::CanSeparateQuantOutput(tiling);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t inputSlot = doubleInput ? row & 1U : 0U;
        const uint32_t outputSlot = outputs.Slot();
        const event_t event = inputSlot == 0U ? EVENT_ID4 : EVENT_ID5;
        const event_t outputEvent = outputs.Event();
        const event_t nextEvent = (row & 1U) == 0U ? EVENT_ID5 : EVENT_ID4;
        const LocalTensor<T> input = inputSlot == 0U ? x2 : alternate;
        const LocalTensor<T> next = (row & 1U) == 0U ? alternate : x2;
        const LocalTensor<T> output = separateQuant || outputSlot == 0U ? hIn : hInPrime;
        WaitFlag<HardEvent::MTE2_V>(event);
        if (doubleInput && row + 1U < rows) {
            PrefetchEpilogueRow(tensors, tiling, start + row + 1U, next, nextEvent);
        }
        if (!separateQuant) {
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
        }
        Compute::ComputeHInTile<T>(
            reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()) + row * Kernel::kHeadCount,
            reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()), tiling.d, 0U, tiling.d,
            reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), stats);
        SetFlag<HardEvent::V_MTE2>(event);
        if (!doubleInput && row + 1U < rows) {
            PrefetchEpilogueRow(tensors, tiling, start + row + 1U, x2, event);
        }
        Compute::ComputeRmsNormFromSquares<T>(
            reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()),
            tiling.d, tiling.normEps, stats, reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()));
        DataCopyExtParams params{1, static_cast<uint32_t>(tiling.d * sizeof(T)), 0, 0, 0};
        if (separateQuant) {
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            auto quant = hInPrime.template ReinterpretCast<uint8_t>()[outputSlot * tiling.d];
            Kernel::WriteSeparateQuantOutput(tensors, tiling, start + row, output, quant, tiling.d, outputSlot);
        } else if (tiling.outputMode == Kernel::kOutputModeMxFp8) {
            Kernel::WriteFinalOutput(tensors, tiling, start + row, 0U, output, tiling.d, true, outputSlot);
        } else {
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            DataCopyPad(tensors.hInPrime[(start + row) * tiling.d], output, params);
        }
        outputs.Publish();
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
    if (doubleInput) {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
    }
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
        const uint32_t batch = tiling.quantBatchRows;
        Kernel::InitUbBuffer(pipe, x2Buffer, batch * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInBuffer, batch * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, hInPrimeBuffer, batch * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, gamma1Buffer, tiling.d * sizeof(float));
        uint32_t partialBytes =
            (tiling.partialGroupCount * tiling.reduceRows * tiling.nzN + kPartialBufferGuardElements) * sizeof(float);
        if ((batch > 1U || tiling.epiloguePrefetch == kEpiloguePrefetchFull) &&
            partialBytes < batch * tiling.k * sizeof(T)) {
            partialBytes = batch * tiling.k * sizeof(T);
        }
        Kernel::InitUbBuffer(pipe, partialBuffer, partialBytes);
        Kernel::InitUbBuffer(pipe, rmsPartialBuffer, tiling.reduceRowsAligned * tiling.kGrid * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer,
                             tiling.reduceRowsAligned * Compute::kProjectionCoefficientElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, alphaBuffer, kAlphaBufferElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, biasBuffer, kBiasBufferElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, kEpilogueStatsElements * sizeof(float));
        preMix.Init(pipe, tiling, tensors.hasShiftedPrevMix);
        tensors.quantOutput.Init(pipe, tiling.outputMode == Kernel::kOutputModeMxFp8, batch * tiling.dTile,
                                 batch * tiling.dTileTail);

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

        EpilogueOutputSlots outputs;
        outputs.Begin();
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
                PrepareEpilogueRows<T, true>(tensors, tiling, start, rows, partial, rmsPartials, coefficients, alpha,
                                             bias, preMix);
                FinalizeMhcRows<T>(tensors, tiling, start, rows, x2, hIn, hInPrime, gamma1, partial,
                                   preMix.Select(coefficients), stats, outputs);
                // Coefficient storage is independent of the two row-output slots.
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
            }
        }
        outputs.End();
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
        tensors.quantOutput.Init(pipe, tiling.outputMode == Kernel::kOutputModeMxFp8, tiling.dTile, tiling.dTileTail);

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
            PrepareEpilogueRows<T, true>(tensors, *plan_, first_ + done_, batchRows_, partial, rmsPartials,
                                         coefficients, alpha, bias, preMix_);
        }
        if (done_ == 0U) {
            PrefetchEpilogueRow(tensors, *plan_, first_, x2, EVENT_ID1);
        }
        const event_t outputEvent = (done_ & QuantOutput::kOutputSlotMask) == 0U ? EVENT_ID4 : EVENT_ID5;
        const bool separateQuant = count_ > QuantOutput::kOutputSlots && Kernel::CanSeparateQuantOutput(*plan_);
        const auto output = separateQuant || (done_ & QuantOutput::kOutputSlotMask) == 0U ? hIn : hInPrime;
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        if (!separateQuant) {
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
        }
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
        DataCopyExtParams copy{1U, static_cast<uint32_t>(plan_->d * sizeof(T)), 0U, 0U, 0U};
        if (separateQuant) {
            WaitFlag<HardEvent::MTE3_V>(outputEvent);
            auto quant =
                hInPrime.template ReinterpretCast<uint8_t>()[(done_ & QuantOutput::kOutputSlotMask) * plan_->d];
            Kernel::WriteSeparateQuantOutput(tensors, *plan_, first_ + done_, output, quant, plan_->d,
                                             done_ & QuantOutput::kOutputSlotMask);
        } else if (plan_->outputMode == Kernel::kOutputModeMxFp8) {
            Kernel::WriteFinalOutput(tensors, *plan_, first_ + done_, 0U, output, plan_->d, true,
                                     done_ & QuantOutput::kOutputSlotMask);
        } else {
            SetFlag<HardEvent::V_MTE3>(outputEvent);
            WaitFlag<HardEvent::V_MTE3>(outputEvent);
            DataCopyPad(tensors.hInPrime[(first_ + done_) * plan_->d], output, copy);
        }
        SetFlag<HardEvent::MTE3_V>(outputEvent);
        ++done_;
        if (done_ == batchStart_ + batchRows_) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
        }
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
 * Mode 0 temporarily uses y for hIn; mode 1 uses an input-only workspace.
 * After the complete-D statistic is known, normalized tiles are quantized directly from UB.
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
        tensors.quantOutput.Init(pipe, tiling.outputMode == Kernel::kOutputModeMxFp8, tiling.dTile, tiling.dTileTail);
        const auto& rms1Input = tiling.outputMode == Kernel::kOutputModeMxFp8 ? tensors.rms1Input : tensors.hInPrime;
        auto stats = reinterpret_cast<__ubuf__ float*>(buffer.stats.GetPhyAddr());
        Kernel::CopyGlobalToUb(alpha, tensors.alpha, 0U, kAlphaElements);
        Kernel::CopyGlobalToUb(bias, tensors.bias, 0U, kProjectionColumns);
        for (uint64_t row = GetBlockIdx(); row < tiling.totalRows; row += tiling.activeAiv) {
            PrepareEpilogueRows<T, true>(tensors, tiling, row, 1U, partial, rms, buffer.coefficients, alpha, bias,
                                         preMix);
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
                buffer.Store(rms1Input, row * tiling.d + static_cast<uint64_t>(tile) * tiling.dTile, tile);
            }
            Compute::FinalizeInverseRms(stats, tiling.d, tiling.normEps, stats + kEpilogueRmsInverseIndex);
            if (tiling.outputMode == Kernel::kOutputModeUnquantized) {
                Kernel::ApplyTiledNormalization(buffer, tiling, rms1Input, tensors.hInPrime, row * tiling.d,
                                                tensors.gamma1, 0U, stats + kEpilogueRmsInverseIndex);
            } else {
                buffer.template PrefetchNorm<true>(rms1Input, row * tiling.d, tensors.gamma1, 0U, 0U);
                for (uint32_t tile = 0U; tile < tiling.dTileCount; ++tile) {
                    buffer.Wait(tile);
                    if (tile + 1U < tiling.dTileCount) {
                        buffer.template PrefetchNorm<true>(rms1Input, row * tiling.d, tensors.gamma1, 0U, tile + 1U);
                    }
                    auto normalized = buffer.Input(tile);
                    auto gamma = buffer.Gamma(tile);
                    Compute::ApplyRmsNorm<T>(reinterpret_cast<__ubuf__ T*>(normalized.GetPhyAddr()),
                                             reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()), buffer.Count(tile),
                                             stats + kEpilogueRmsInverseIndex,
                                             reinterpret_cast<__ubuf__ T*>(normalized.GetPhyAddr()));
                    Kernel::WriteFinalOutput(tensors, tiling, row, tile * tiling.dTile, normalized, buffer.Count(tile),
                                             true, tile & QuantOutput::kOutputSlotMask);
                    buffer.ReleaseOutput(tile);
                }
            }
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
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
        tiling.reduceRowsAligned, reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()), tiling.sinkhornEps,
        tiling.numSinkhornIters);
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
    SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
    preMix.StoreCurrent(tensors, row, rows, coefficients);
    preMix.WaitReady();
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
                                                   const LocalTensor<float>& statsBuffer, EpilogueOutputSlots& outputs)
{
    SetFlag<HardEvent::V_MTE2>(BsHfEpilogue::kInput0);
    SetFlag<HardEvent::V_MTE2>(BsHfEpilogue::kInput1);
    const uint32_t firstCount = rows < batch ? rows : batch;
    PrefetchBsHfEpilogueBatch(tensors, t, start, firstCount, input, BsHfEpilogue::kInput0);
    auto stats = reinterpret_cast<__ubuf__ float*>(statsBuffer.GetPhyAddr());
    const bool separateQuant = t.reduceRows > QuantOutput::kOutputSlots && Kernel::CanSeparateQuantOutput(t);
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
            const uint32_t outputSlot = outputs.Slot();
            const event_t outputEvent = outputs.Event();
            const auto output = separateQuant || outputSlot == 0U ? hIn : hInPrime;
            if (!separateQuant) {
                WaitFlag<HardEvent::MTE3_V>(outputEvent);
            }
            Compute::ComputeHInTile<T>(
                reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()) + row * Kernel::kHeadCount,
                reinterpret_cast<__ubuf__ T*>(input[(inputSlot * batch + batchRow) * t.k].GetPhyAddr()), t.d, 0U, t.d,
                reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), stats);
            Compute::ComputeRmsNormFromSquares<T>(reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()), t.d, t.normEps,
                                                  stats, reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()));
            DataCopyExtParams copy{1U, static_cast<uint32_t>(t.d * sizeof(T)), 0U, 0U, 0U};
            if (separateQuant) {
                WaitFlag<HardEvent::MTE3_V>(outputEvent);
                auto quant = hInPrime.template ReinterpretCast<uint8_t>()[outputSlot * t.d];
                Kernel::WriteSeparateQuantOutput(tensors, t, start + row, output, quant, t.d, outputSlot);
            } else if (t.outputMode == Kernel::kOutputModeMxFp8) {
                Kernel::WriteFinalOutput(tensors, t, start + row, 0U, output, t.d, true, outputSlot);
            } else {
                SetFlag<HardEvent::V_MTE3>(outputEvent);
                WaitFlag<HardEvent::V_MTE3>(outputEvent);
                DataCopyPad(tensors.hInPrime[(start + row) * t.d], output, copy);
            }
            outputs.Publish();
        }
        // All HIn reads from this batch are complete; RMS1 only reads the row-output buffers.
        SetFlag<HardEvent::V_MTE2>(inputEvent);
    }
    WaitFlag<HardEvent::V_MTE2>(BsHfEpilogue::kInput0);
    WaitFlag<HardEvent::V_MTE2>(BsHfEpilogue::kInput1);
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
        tensors.quantOutput.Init(pipe, tiling.outputMode == Kernel::kOutputModeMxFp8, tiling.dTile, tiling.dTileTail);

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

        EpilogueOutputSlots outputs;
        outputs.Begin(BATCHED_INPUT);
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
                    PrepareEpilogueRows<T, true>(tensors, tiling, start, rows, partial, rmsPartials, coefficients,
                                                 alpha, bias, preMix);
                }
                if constexpr (BATCHED_INPUT) {
                    FinalizeBsHfEpilogueBatches(tensors, tiling, start, rows, batch, x2, hIn, hInPrime, gamma1,
                                                preMix.Select(coefficients), stats, outputs);
                } else {
                    FinalizeMhcRows(tensors, tiling, start, rows, x2, hIn, hInPrime, gamma1, partial,
                                    preMix.Select(coefficients), stats, outputs);
                }
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
            }
        }
        outputs.End();
        preMix.End();
    }
}

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_EPILOGUE_H
