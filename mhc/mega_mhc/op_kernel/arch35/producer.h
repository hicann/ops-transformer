/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_PRODUCER_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_PRODUCER_H

#include "compute/mega_mhc_post.h"
#include "compute/rms_norm_0_mhc_post.h"
#include "mega_mhc_common.h"

namespace MegaMhc::Schedule {
using namespace AscendC;

// WITH_RMS0=false bypasses the RMS0 stage in every producer; Post then consumes hOut directly.
constexpr uint32_t kProducerCoefficientPlaneOffset = 64U;
constexpr uint32_t kProducerCoefficientSlotStride = 128U;
constexpr uint32_t kProducerCoefficientBufferElements = 128U;
constexpr uint32_t kProducerInputSlots = 2U;
constexpr uint32_t kProducerDoubleResidualSlots = 2U;
constexpr uint32_t kProducerResidualSlots = 3U;
constexpr uint32_t kProducerStatsElements = 2U;
constexpr uint32_t kProducerWaveReadyEventBase = 6U;
constexpr uint32_t kProducerRms0CarryIndex = 4U;
constexpr uint32_t kProducerMidSumIndex = 2U;
constexpr uint32_t kProducerMidInverseIndex = 3U;
constexpr uint32_t kProducerMidCarryIndex = 5U;

/** Enqueue producer inputs: hOut=x[row], optional residual=r[row], coefficients=H[row].
 * Wait for V_MTE2 input credit or MTE3_MTE2 aliased output credit; MTE2_V publishes the loaded inputs.
 */
template <typename T, bool OUTPUT_CREDIT = false, bool LOAD_RESIDUAL = true>
__aicore__ inline void PrefetchStagedProducer(Kernel::Tensors<T>& tensors, uint64_t row,
                                              const MegaMhcTilingData& tiling, const LocalTensor<T>& hOut,
                                              const LocalTensor<T>& residual, const LocalTensor<float>& coefficients,
                                              event_t event)
{
    if constexpr (OUTPUT_CREDIT) {
        WaitFlag<HardEvent::MTE3_MTE2>(event);
    } else {
        WaitFlag<HardEvent::V_MTE2>(event);
    }
    Kernel::EnqueueGlobalToUb(hOut, tensors.hOut, row * tiling.d, tiling.d);
    if constexpr (LOAD_RESIDUAL) {
        Kernel::EnqueueGlobalToUb(residual, tensors.residual, row * tiling.k, tiling.k);
    }
    Kernel::EnqueueGlobalToUb(coefficients, tensors.hPost, row * Kernel::kHeadCount, Kernel::kHeadCount);
    Kernel::EnqueueGlobalToUb(coefficients[kProducerCoefficientPlaneOffset], tensors.hRes,
                              row * Kernel::kHeadMatrixElements, Kernel::kHeadMatrixElements);
    SetFlag<HardEvent::MTE2_V>(event);
}

/** Prefetch no_mid inputs with independent input and aliased residual/output slot credits. */
template <typename T>
__aicore__ inline void PrefetchNoMidStagedProducer(Kernel::Tensors<T>& tensors, uint64_t row,
                                                   const MegaMhcTilingData& tiling, const LocalTensor<T>& hOut,
                                                   const LocalTensor<T>& residual,
                                                   const LocalTensor<float>& coefficients, event_t inputEvent,
                                                   event_t residualEvent)
{
    WaitFlag<HardEvent::V_MTE2>(inputEvent);
    Kernel::EnqueueGlobalToUb(hOut, tensors.hOut, row * tiling.d, tiling.d);
    Kernel::EnqueueGlobalToUb(coefficients, tensors.hPost, row * Kernel::kHeadCount, Kernel::kHeadCount);
    Kernel::EnqueueGlobalToUb(coefficients[kProducerCoefficientPlaneOffset], tensors.hRes,
                              row * Kernel::kHeadMatrixElements, Kernel::kHeadMatrixElements);
    WaitFlag<HardEvent::MTE3_MTE2>(residualEvent);
    Kernel::EnqueueGlobalToUb(residual, tensors.residual, row * tiling.k, tiling.k);
    SetFlag<HardEvent::MTE2_V>(residualEvent);
}

/** Compute one RMS0/Post row with the required x1 Cast, avoiding its UB round trip. */
template <typename T, bool WITH_RMS0>
__aicore__ inline void ComputeNoMidProducerRow(LocalTensor<T> input, LocalTensor<T> residual, LocalTensor<float> gamma,
                                               LocalTensor<float> coefficients, LocalTensor<float> stats,
                                               const MegaMhcTilingData& tiling)
{
    Compute::ComputeRmsNorm0PostRow<T, false, false, WITH_RMS0>(
        reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()),
        tiling.normEps, reinterpret_cast<__ubuf__ T*>(residual.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(coefficients.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(coefficients[kProducerCoefficientPlaneOffset].GetPhyAddr()),
        reinterpret_cast<__ubuf__ T*>(residual.GetPhyAddr()), tiling.d,
        reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr()));
}

/** AIVs with exactly one row need one input/output slot and no slot credits. */
template <typename T, bool WITH_RMS0>
__aicore__ inline void RunNoMidSingleRowProducer(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                                 TPipe* pipe, uint32_t row)
{
    TBuf<TPosition::VECCALC> inputBuffer, residualBuffer;
    TBuf<TPosition::VECCALC> gammaBuffer, coefficientBuffer, statsBuffer;
    Kernel::InitUbBuffer(pipe, inputBuffer, tiling.d * sizeof(T));
    Kernel::InitUbBuffer(pipe, residualBuffer, tiling.k * sizeof(T));
    Kernel::InitUbBuffer(pipe, gammaBuffer, tiling.d * sizeof(float));
    Kernel::InitUbBuffer(pipe, coefficientBuffer, kProducerCoefficientBufferElements * sizeof(float));
    Kernel::InitUbBuffer(pipe, statsBuffer, kProducerStatsElements * sizeof(float));
    auto input = inputBuffer.Get<T>();
    auto residual = residualBuffer.Get<T>();

    auto gamma = gammaBuffer.Get<float>();
    auto coefficients = coefficientBuffer.Get<float>();
    auto stats = statsBuffer.Get<float>();
    if constexpr (WITH_RMS0) {
        Kernel::CopyGlobalToUb(gamma, tensors.gamma0, 0U, tiling.d);
    }
    Kernel::EnqueueGlobalToUb(input, tensors.hOut, static_cast<uint64_t>(row) * tiling.d, tiling.d);
    Kernel::EnqueueGlobalToUb(coefficients, tensors.hPost, static_cast<uint64_t>(row) * Kernel::kHeadCount,
                              Kernel::kHeadCount);
    Kernel::EnqueueGlobalToUb(coefficients[kProducerCoefficientPlaneOffset], tensors.hRes,
                              static_cast<uint64_t>(row) * Kernel::kHeadMatrixElements, Kernel::kHeadMatrixElements);
    Kernel::EnqueueGlobalToUb(residual, tensors.residual, static_cast<uint64_t>(row) * tiling.k, tiling.k);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    ComputeNoMidProducerRow<T, WITH_RMS0>(input, residual, gamma, coefficients, stats, tiling);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams copy{1U, static_cast<uint32_t>(tiling.k * sizeof(T)), 0U, 0U, 0U};
    DataCopyPad(tensors.x2Out[static_cast<uint64_t>(row) * tiling.k], residual, copy);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

/** Produce x2[row]=Post(Cast<T>(RMS0(hOut[row])), residual[row]) without Mid.
 * Two input slots and three aliased residual/output slots overlap MTE2, Vector and MTE3.
 * Post output retains its input-type rounding before GM publication; the phase barrier remains.
 */
template <typename T, bool SKIP_IDLE_AIV, bool WITH_RMS0>
__aicore__ inline void RunNoMidStagedProducer(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
{
    if ASCEND_IS_AIV {
        const uint32_t firstRow = GetBlockIdx();
        if constexpr (SKIP_IDLE_AIV) {
            if (firstRow >= tiling.totalRows) {
                return;
            }
        }
        if (firstRow < tiling.totalRows && static_cast<uint64_t>(firstRow) + tiling.activeAiv >= tiling.totalRows) {
            RunNoMidSingleRowProducer<T, WITH_RMS0>(tensors, tiling, pipe, firstRow);
            return;
        }
        TBuf<TPosition::VECCALC> inputBuffer, residualBuffer;
        TBuf<TPosition::VECCALC> gammaBuffer, coefficientBuffer, statsBuffer;
        Kernel::InitUbBuffer(pipe, inputBuffer, kProducerInputSlots * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, residualBuffer, kProducerResidualSlots * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, gammaBuffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer, Kernel::kCoefficientTileElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, kProducerStatsElements * sizeof(float));
        auto input = inputBuffer.Get<T>();
        auto residual = residualBuffer.Get<T>();

        auto gamma = gammaBuffer.Get<float>();
        auto coefficients = coefficientBuffer.Get<float>();
        auto stats = statsBuffer.Get<float>();
        if constexpr (WITH_RMS0) {
            Kernel::CopyGlobalToUb(gamma, tensors.gamma0, 0U, tiling.d);
        }
        // Event IDs are scoped by HardEvent type: input credits use V_MTE2 2/3,
        // residual credits use MTE3_MTE2 4/5/6, and row readiness uses MTE2_V 4/5/6.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (firstRow < tiling.totalRows) {
            PrefetchNoMidStagedProducer(tensors, firstRow, tiling, input, residual, coefficients, EVENT_ID2, EVENT_ID4);
        }
        for (uint32_t iter = 0U;
             static_cast<uint64_t>(firstRow) + static_cast<uint64_t>(iter) * tiling.activeAiv < tiling.totalRows;
             ++iter) {
            const uint64_t row = static_cast<uint64_t>(firstRow) + static_cast<uint64_t>(iter) * tiling.activeAiv;
            const uint32_t inputSlot = iter % kProducerInputSlots;
            const uint32_t residualSlot = iter % kProducerResidualSlots;
            const event_t inputEvent = inputSlot == 0U ? EVENT_ID2 : EVENT_ID3;
            const event_t residualEvent = residualSlot == 0U ? EVENT_ID4 : residualSlot == 1U ? EVENT_ID5 : EVENT_ID6;
            WaitFlag<HardEvent::MTE2_V>(residualEvent);
            if (row + tiling.activeAiv < tiling.totalRows) {
                const uint32_t nextInputSlot = (iter + 1U) % kProducerInputSlots;
                const uint32_t nextResidualSlot = (iter + 1U) % kProducerResidualSlots;
                const event_t nextInputEvent = nextInputSlot == 0U ? EVENT_ID2 : EVENT_ID3;
                const event_t nextResidualEvent = nextResidualSlot == 0U ? EVENT_ID4 :
                                                  nextResidualSlot == 1U ? EVENT_ID5 :
                                                                           EVENT_ID6;
                PrefetchNoMidStagedProducer(tensors, row + tiling.activeAiv, tiling, input[nextInputSlot * tiling.d],
                                            residual[nextResidualSlot * tiling.k],
                                            coefficients[nextInputSlot * kProducerCoefficientSlotStride],
                                            nextInputEvent, nextResidualEvent);
            }
            ComputeNoMidProducerRow<T, WITH_RMS0>(input[inputSlot * tiling.d], residual[residualSlot * tiling.k], gamma,
                                                  coefficients[inputSlot * kProducerCoefficientSlotStride], stats,
                                                  tiling);
            SetFlag<HardEvent::V_MTE2>(inputEvent);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams copy{1U, static_cast<uint32_t>(tiling.k * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(tensors.x2Out[row * tiling.k], residual[residualSlot * tiling.k], copy);
            SetFlag<HardEvent::MTE3_MTE2>(residualEvent);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
    }
}

/** Produce x2[row]=Post(Cast<T>(RMS0(hOut[row])), residual[row]) followed by RMSMid, with both input-type stage
 * roundings. Input/coefficient prefetch remains double-buffered. A single residual slot can keep the full Mid gamma
 * resident; its next load waits until the current in-place Post/Mid output has reached GM.
 */
/** Map a with_mid producer to the same token half-tile as its paired BS projection.
 * MK retains their cyclic complete-row mapping. BS never redistributes K or tokens across pairs.
 */
template <bool BS_OWNERSHIP>
__aicore__ inline uint64_t WithMidProducerRow(const MegaMhcTilingData& tiling, uint64_t ordinal)
{
    if constexpr (BS_OWNERSHIP) {
        return (ordinal / tiling.projectionRows) * tiling.rowsPerWave +
               static_cast<uint64_t>(GetBlockIdx() / Kernel::kAivPerAic) * tiling.mTile +
               static_cast<uint64_t>(GetSubBlockIdx()) * tiling.projectionRows + ordinal % tiling.projectionRows;
    }
    return GetBlockIdx() + ordinal * tiling.activeAiv;
}

/** Input credits are independent of the single residual/output slot, so next hOut can prefetch during Post. */
template <typename T>
__aicore__ inline void PrefetchCachedGammaInput(Kernel::Tensors<T>& tensors, uint64_t row,
                                                const MegaMhcTilingData& tiling, const LocalTensor<T>& input,
                                                const LocalTensor<float>& coefficients, event_t event)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    Kernel::EnqueueGlobalToUb(input, tensors.hOut, row * tiling.d, tiling.d);
    Kernel::EnqueueGlobalToUb(coefficients, tensors.hPost, row * Kernel::kHeadCount, Kernel::kHeadCount);
    Kernel::EnqueueGlobalToUb(coefficients[kProducerCoefficientPlaneOffset], tensors.hRes,
                              row * Kernel::kHeadMatrixElements, Kernel::kHeadMatrixElements);
    SetFlag<HardEvent::MTE2_V>(event);
}

template <typename T>
__aicore__ inline void PrefetchCachedGammaResidual(Kernel::Tensors<T>& tensors, uint64_t row,
                                                   const MegaMhcTilingData& tiling, const LocalTensor<T>& residual,
                                                   event_t event = EVENT_ID1)
{
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    Kernel::EnqueueGlobalToUb(residual, tensors.residual, row * tiling.k, tiling.k);
    SetFlag<HardEvent::MTE2_V>(event);
}

/** Cache gamma0/gamma2 once with the Host-selected residual slot count.
 * In-place RMS0 avoids a separate x1 buffer; double residual slots overlap the next row load.
 */
template <typename T, bool DOUBLE_RESIDUAL, bool WITH_RMS0>
__aicore__ inline void RunCachedGammaWithMidProducer(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                                     TPipe* pipe)
{
    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> inputBuffer, residualBuffer, gammaBuffer, gamma2Buffer, coefficientBuffer, statsBuffer;
        Kernel::InitUbBuffer(pipe, inputBuffer, kProducerInputSlots * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, residualBuffer,
                             (DOUBLE_RESIDUAL ? kProducerDoubleResidualSlots : 1U) * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, gammaBuffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(pipe, gamma2Buffer, tiling.k * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer, Kernel::kCoefficientTileElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, kProducerStatsElements * sizeof(float));
        auto input = inputBuffer.Get<T>();
        auto residual = residualBuffer.Get<T>();
        auto gamma = gammaBuffer.Get<float>();
        auto gamma2 = gamma2Buffer.Get<float>();
        auto coefficients = coefficientBuffer.Get<float>();
        auto stats = statsBuffer.Get<float>();
        auto sum = reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr());
        if constexpr (WITH_RMS0) {
            Kernel::CopyGlobalToUb(gamma, tensors.gamma0, 0U, tiling.d);
        }
        Kernel::CopyGlobalToUb(gamma2, tensors.gamma2, 0U, tiling.k);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        if constexpr (DOUBLE_RESIDUAL) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
        const uint64_t firstRow = WithMidProducerRow<true>(tiling, 0U);
        if (firstRow < tiling.totalRows) {
            PrefetchCachedGammaInput(tensors, firstRow, tiling, input, coefficients, EVENT_ID4);
            PrefetchCachedGammaResidual(tensors, firstRow, tiling, residual);
        }
        uint32_t slot = 0U;
        for (uint64_t ordinal = 0U, row = firstRow; row < tiling.totalRows;
             ++ordinal, row = WithMidProducerRow<true>(tiling, ordinal)) {
            const event_t event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
            const event_t residualEvent = DOUBLE_RESIDUAL && slot == 1U ? EVENT_ID2 : EVENT_ID1;
            auto post = residual[(DOUBLE_RESIDUAL ? slot : 0U) * tiling.k];
            WaitFlag<HardEvent::MTE2_V>(event);
            const uint32_t next = slot ^ 1U;
            const uint64_t nextRow = WithMidProducerRow<true>(tiling, ordinal + 1U);
            if (nextRow < tiling.totalRows) {
                PrefetchCachedGammaInput(tensors, nextRow, tiling, input[next * tiling.d],
                                         coefficients[next * kProducerCoefficientSlotStride],
                                         next == 0U ? EVENT_ID4 : EVENT_ID5);
                if constexpr (DOUBLE_RESIDUAL) {
                    PrefetchCachedGammaResidual(tensors, nextRow, tiling, residual[next * tiling.k],
                                                next == 0U ? EVENT_ID1 : EVENT_ID2);
                }
            }
            auto x1 = reinterpret_cast<__ubuf__ T*>(input[slot * tiling.d].GetPhyAddr());
            if constexpr (WITH_RMS0) {
                Compute::AccumulateRowSquares<T>(x1, tiling.d, sum + 1U);
                Compute::ComputeRmsNormFromSquares<T>(x1, reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()),
                                                      tiling.d, tiling.normEps, sum + 1U, x1);
            }
            // Residual DMA can overlap RMS0; Post needs its completion before reading the aliased row.
            WaitFlag<HardEvent::MTE2_V>(residualEvent);
            Compute::ComputePostFourHeads<T, true>(
                x1, reinterpret_cast<__ubuf__ T*>(post.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(coefficients[slot * kProducerCoefficientSlotStride].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(
                    coefficients[slot * kProducerCoefficientSlotStride + kProducerCoefficientPlaneOffset].GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(post.GetPhyAddr()), 1U, tiling.d, 0U, tiling.d, sum);
            SetFlag<HardEvent::V_MTE2>(event);
            Compute::ComputeRmsNormFromSquares<T>(reinterpret_cast<__ubuf__ T*>(post.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ float*>(gamma2.GetPhyAddr()), tiling.k,
                                                  tiling.normEps, sum,
                                                  reinterpret_cast<__ubuf__ T*>(post.GetPhyAddr()));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams params{1U, static_cast<uint32_t>(tiling.k * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(tensors.x2Out[row * tiling.k], post, params);
            SetFlag<HardEvent::MTE3_MTE2>(residualEvent);
            if constexpr (!DOUBLE_RESIDUAL) {
                if (nextRow < tiling.totalRows) {
                    PrefetchCachedGammaResidual(tensors, nextRow, tiling, residual);
                }
            }
            slot = next;
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        if constexpr (DOUBLE_RESIDUAL) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
        SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
    }
}

template <typename T, bool BS_OWNERSHIP, bool WITH_RMS0>
__aicore__ inline void RunPrefetchedWithMidProducer(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                                    TPipe* pipe)
{
    if constexpr (BS_OWNERSHIP) {
        if (tiling.withMidProducerResidualSlots != 0U) {
            if (tiling.withMidProducerResidualSlots == kProducerDoubleResidualSlots) {
                RunCachedGammaWithMidProducer<T, true, WITH_RMS0>(tensors, tiling, pipe);
            } else {
                RunCachedGammaWithMidProducer<T, false, WITH_RMS0>(tensors, tiling, pipe);
            }
            return;
        }
    }
    if ASCEND_IS_AIV {
        const bool prefetchResidual = tiling.producerPrefetch != 0U;
        TBuf<TPosition::VECCALC> inputBuffer, residualBuffer, gamma2Buffer;
        TBuf<TPosition::VECCALC> gammaBuffer, coefficientBuffer, statsBuffer;
        Kernel::InitUbBuffer(pipe, inputBuffer, kProducerInputSlots * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, residualBuffer,
                             (prefetchResidual ? kProducerDoubleResidualSlots : 1U) * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, gamma2Buffer, tiling.midGammaTile * sizeof(float));
        Kernel::InitUbBuffer(pipe, gammaBuffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer, Kernel::kCoefficientTileElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, kProducerStatsElements * sizeof(float));
        auto input = inputBuffer.Get<T>();
        auto residual = residualBuffer.Get<T>();

        auto gamma2 = gamma2Buffer.Get<float>();
        auto gamma = gammaBuffer.Get<float>();
        auto coefficients = coefficientBuffer.Get<float>();
        auto stats = statsBuffer.Get<float>();
        if constexpr (WITH_RMS0) {
            Kernel::CopyGlobalToUb(gamma, tensors.gamma0, 0U, tiling.d);
        }
        if (tiling.midGammaTileCount == 1U) {
            Kernel::CopyGlobalToUb(gamma2, tensors.gamma2, 0U, tiling.k);
        }
        const uint64_t firstRow = WithMidProducerRow<BS_OWNERSHIP>(tiling, 0U);
        if (prefetchResidual) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        } else {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID5);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
        }
        if (firstRow < tiling.totalRows) {
            if (prefetchResidual) {
                PrefetchStagedProducer<T, true>(tensors, firstRow, tiling, input, residual, coefficients, EVENT_ID4);
            } else {
                PrefetchStagedProducer<T, false, false>(tensors, firstRow, tiling, input, residual, coefficients,
                                                        EVENT_ID4);
            }
        }
        uint32_t slot = 0U;
        for (uint64_t ordinal = 0U, row = firstRow; row < tiling.totalRows;
             ++ordinal, row = WithMidProducerRow<BS_OWNERSHIP>(tiling, ordinal)) {
            const event_t event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
            WaitFlag<HardEvent::MTE2_V>(event);
            const uint32_t next = slot ^ 1U;
            const uint64_t nextRow = WithMidProducerRow<BS_OWNERSHIP>(tiling, ordinal + 1U);
            if (prefetchResidual) {
                if (nextRow < tiling.totalRows) {
                    PrefetchStagedProducer<T, true>(
                        tensors, nextRow, tiling, input[next * tiling.d], residual[next * tiling.k],
                        coefficients[next * kProducerCoefficientSlotStride], next == 0U ? EVENT_ID4 : EVENT_ID5);
                }
            } else {
                // Residual aliases Post/Mid output. MTE3 must release it before the next row's MTE2 load.
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
                Kernel::EnqueueGlobalToUb(residual, tensors.residual, row * tiling.k, tiling.k);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
                if (nextRow < tiling.totalRows) {
                    PrefetchStagedProducer<T, false, false>(tensors, nextRow, tiling, input[next * tiling.d], residual,
                                                            coefficients[next * kProducerCoefficientSlotStride],
                                                            next == 0U ? EVENT_ID4 : EVENT_ID5);
                }
            }
            auto post = residual[(prefetchResidual ? slot : 0U) * tiling.k];
            // RMS0 statistics depend only on input, so keep them overlapped with residual DMA.
            if constexpr (WITH_RMS0) {
                Compute::AccumulateRowSquares<T>(reinterpret_cast<__ubuf__ T*>(input[slot * tiling.d].GetPhyAddr()),
                                                 tiling.d, reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr()));
            }
            if (!prefetchResidual) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
            }
            Compute::ComputeRmsNorm0PostRow<T, true, true, WITH_RMS0>(
                reinterpret_cast<__ubuf__ T*>(input[slot * tiling.d].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()), tiling.normEps,
                reinterpret_cast<__ubuf__ T*>(post.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(coefficients[slot * kProducerCoefficientSlotStride].GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(
                    coefficients[slot * kProducerCoefficientSlotStride + kProducerCoefficientPlaneOffset].GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(post.GetPhyAddr()), tiling.d,
                reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr()));
            if (tiling.midGammaTileCount == 1U) {
                Compute::ComputeRmsNormFromSquares<T>(reinterpret_cast<__ubuf__ T*>(post.GetPhyAddr()),
                                                      reinterpret_cast<__ubuf__ float*>(gamma2.GetPhyAddr()), tiling.k,
                                                      tiling.normEps,
                                                      reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr()),
                                                      reinterpret_cast<__ubuf__ T*>(post.GetPhyAddr()));
            } else {
                // invRms uses all K elements once; gamma chunks preserve the same rounding point.
                auto sum = reinterpret_cast<__ubuf__ float*>(stats.GetPhyAddr());
                Compute::FinalizeInverseRms(sum, tiling.k, tiling.normEps, sum + 1U);
                for (uint32_t chunk = 0U; chunk < tiling.midGammaTileCount; ++chunk) {
                    const uint32_t offset = chunk * tiling.midGammaTile;
                    const uint32_t count =
                        chunk + 1U == tiling.midGammaTileCount ? tiling.midGammaLastTile : tiling.midGammaTile;
                    Kernel::CopyGlobalToUb(gamma2, tensors.gamma2, offset, count);
                    auto rowPart = reinterpret_cast<__ubuf__ T*>(post[offset].GetPhyAddr());
                    Compute::ApplyRmsNorm<T>(rowPart, reinterpret_cast<__ubuf__ float*>(gamma2.GetPhyAddr()), count,
                                             sum + 1U, rowPart);
                }
            }
            // x2[row,k]=post[k]. Publish slot credit on MTE3; only reuse waits for completion.
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams params{1U, static_cast<uint32_t>(tiling.k * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(tensors.x2Out[row * tiling.k], post, params);
            SetFlag<HardEvent::MTE3_MTE2>(prefetchResidual ? event : EVENT_ID6);
            if (!prefetchResidual) {
                SetFlag<HardEvent::V_MTE2>(event);
            }
            slot = next;
        }
        if (prefetchResidual) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        } else {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        }
        // Drain both output reads before releasing UB, including unused/empty input slots.
        SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
    }
}

/** Materialize RMS0/Post and optional Mid with a bounded D working set.
 * Formula: x1=Cast<T>(x*rsqrt(sum_D(x*x)/D+eps)*gamma0), post=Cast<T>(postMix*x1+resMix*residual).
 * RMS0 and Mid accumulate statistics across every D tile before their apply pass; each token has one AIV owner.
 * Rounded Post lives temporarily in new_residual, then Mid overwrites it with its rounded result.
 */
template <typename T, bool WITH_MID, bool WITH_RMS0>
__aicore__ inline void RunTiledProducer(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
{
    if ASCEND_IS_AIV {
        Kernel::TiledRowBuffers<T> buffer;
        buffer.Init(pipe, tiling);
        auto stats = reinterpret_cast<__ubuf__ float*>(buffer.stats.GetPhyAddr());
        for (uint64_t row = GetBlockIdx(); row < tiling.totalRows; row += tiling.activeAiv) {
            Kernel::ZeroFp32Values(stats, Kernel::kRmsStatElements);
            if constexpr (WITH_RMS0) {
                buffer.template PrefetchNorm<false>(tensors.hOut, row * tiling.d, tensors.gamma0, 0U, 0U);
                for (uint32_t tile = 0U; tile < tiling.dTileCount; ++tile) {
                    buffer.Wait(tile);
                    if (tile + 1U < tiling.dTileCount) {
                        buffer.template PrefetchNorm<false>(tensors.hOut, row * tiling.d, tensors.gamma0, 0U,
                                                            tile + 1U);
                    }
                    auto input = buffer.Input(tile);
                    Compute::AccumulateCompensatedSquares<T>(reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()),
                                                             buffer.Count(tile), stats,
                                                             stats + kProducerRms0CarryIndex);
                    buffer.Release(tile);
                }
                Compute::FinalizeInverseRms<true>(stats, tiling.d, tiling.normEps, stats + 1U,
                                                  stats + kProducerRms0CarryIndex);
            }
            Kernel::CopyGlobalToUb(buffer.coefficients, tensors.hPost, row * Kernel::kHeadCount, Kernel::kHeadCount);
            Kernel::CopyGlobalToUb(buffer.coefficients[kProducerCoefficientPlaneOffset], tensors.hRes,
                                   row * Kernel::kHeadMatrixElements, Kernel::kHeadMatrixElements);
            buffer.template PrefetchHeads<true, WITH_RMS0>(tensors, row, 0U);
            for (uint32_t tile = 0U; tile < tiling.dTileCount; ++tile) {
                buffer.Wait(tile);
                if (tile + 1U < tiling.dTileCount) {
                    buffer.template PrefetchHeads<true, WITH_RMS0>(tensors, row, tile + 1U);
                }
                auto input = buffer.Input(tile);
                auto heads = buffer.Heads(tile);
                auto gamma = buffer.Gamma(tile);
                if constexpr (WITH_RMS0) {
                    Compute::ApplyRmsNorm<T>(reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()),
                                             reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()), buffer.Count(tile),
                                             stats + 1U, reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()));
                }
                Compute::ComputePostFourHeads<T>(reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()),
                                                 reinterpret_cast<__ubuf__ T*>(heads.GetPhyAddr()),
                                                 reinterpret_cast<__ubuf__ float*>(buffer.coefficients.GetPhyAddr()),
                                                 reinterpret_cast<__ubuf__ float*>(
                                                     buffer.coefficients[kProducerCoefficientPlaneOffset].GetPhyAddr()),
                                                 reinterpret_cast<__ubuf__ T*>(heads.GetPhyAddr()), 1U, tiling.dTile,
                                                 0U, buffer.Count(tile));
                for (uint32_t head = 0U; head < Kernel::kHeadCount; ++head) {
                    if constexpr (WITH_MID) {
                        Compute::AccumulateCompensatedSquares<T>(
                            reinterpret_cast<__ubuf__ T*>(heads[head * tiling.dTile].GetPhyAddr()), buffer.Count(tile),
                            stats + kProducerMidSumIndex, stats + kProducerMidCarryIndex);
                    }
                    Kernel::CopyUbToGlobal(tensors.x2Out,
                                           row * tiling.k + static_cast<uint64_t>(head) * tiling.d +
                                               static_cast<uint64_t>(tile) * tiling.dTile,
                                           heads[head * tiling.dTile], buffer.Count(tile));
                }
                buffer.Release(tile);
            }
            if constexpr (WITH_MID) {
                Compute::FinalizeInverseRms<true>(stats + kProducerMidSumIndex, tiling.k, tiling.normEps,
                                                  stats + kProducerMidInverseIndex, stats + kProducerMidCarryIndex);
                for (uint32_t head = 0U; head < Kernel::kHeadCount; ++head) {
                    Kernel::ApplyTiledNormalization(buffer, tiling, tensors.x2Out, tensors.x2Out,
                                                    row * tiling.k + static_cast<uint64_t>(head) * tiling.d,
                                                    tensors.gamma2, static_cast<uint64_t>(head) * tiling.d,
                                                    stats + kProducerMidInverseIndex);
                }
            }
        }
        buffer.End();
    }
}

/** Stateful row producer used only by the outer projection schedule.
 * x2=Cast<T>(Post(Cast<T>(RMS0(hOut)),residual)); arithmetic stays in shared RegBase functions.
 * Producer buffers coexist with projection buffers; host owns row counts and tick spacing.
 */
template <typename T, bool WITH_RMS0>
class WavePostProducer {
public:
    /** Allocate the host-budgeted two input slots and bind one host-planned AIV M group. */
    __aicore__ inline void Init(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
    {
        tensors_ = &tensors;
        tiling_ = &tiling;
        Kernel::InitUbBuffer(pipe, inputBuffer_, kProducerInputSlots * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, residualBuffer_, kProducerDoubleResidualSlots * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, gammaBuffer_, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientsBuffer_, Kernel::kCoefficientTileElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer_, kProducerStatsElements * sizeof(float));
        input_ = inputBuffer_.template Get<T>();
        residual_ = residualBuffer_.template Get<T>();
        gamma_ = gammaBuffer_.template Get<float>();
        coefficients_ = coefficientsBuffer_.template Get<float>();
        stats_ = statsBuffer_.template Get<float>();
        if constexpr (WITH_RMS0) {
            Kernel::CopyGlobalToUb(gamma_, tensors.gamma0, 0U, tiling.d);
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
    }

    /** Begin a producer batch: full members own R rows; Host supplies the final member count and tail.
     * Empty tail members skip data access but still publish completion and participate in group synchronization.
     */
    __aicore__ inline void Begin(uint32_t wave)
    {
        const auto& t = *tiling_;
        row_ = static_cast<uint64_t>(wave) * t.rowsPerWave +
               static_cast<uint64_t>(GetBlockIdx() / t.waveGroupSize) * t.mTile +
               (GetBlockIdx() % t.waveGroupSize) * t.waveProducerRows;
        produced_ = 0U;
        slot_ = 0U;
        nextTick_ = 1U;
        const uint32_t member = GetBlockIdx();
        rows_ = wave + 1U != t.mWaveCount || member < t.lastWaveFullProducers ?
                    t.waveProducerRows :
                    (member == t.lastWaveFullProducers ? t.lastWaveProducerTailRows : 0U);
        if (rows_ != 0U) {
            Prefetch(row_, 0U);
        }
    }

    /** Prefetch hOut, residual and coefficients for one row into a free input slot. */
    __aicore__ inline void Prefetch(uint64_t row, uint32_t slot)
    {
        const auto& t = *tiling_;
        const event_t event = slot == 0U ? EVENT_ID4 : EVENT_ID5;
        WaitFlag<HardEvent::MTE3_MTE2>(event);
        Kernel::EnqueueGlobalToUb(input_[slot * t.d], tensors_->hOut, row * t.d, t.d);
        Kernel::EnqueueGlobalToUb(residual_[slot * t.k], tensors_->residual, row * t.k, t.k);
        Kernel::EnqueueGlobalToUb(coefficients_[slot * kProducerCoefficientSlotStride], tensors_->hPost,
                                  row * Kernel::kHeadCount, Kernel::kHeadCount);
        Kernel::EnqueueGlobalToUb(
            coefficients_[slot * kProducerCoefficientSlotStride + kProducerCoefficientPlaneOffset], tensors_->hRes,
            row * Kernel::kHeadMatrixElements, Kernel::kHeadMatrixElements);
        SetFlag<HardEvent::MTE2_V>(event);
    }

    /** Compute one rounded RMS0/Post row and enqueue x2[row,:]; slot reuse waits for its store. */
    __aicore__ inline void ProduceOne()
    {
        const auto& t = *tiling_;
        const event_t event = slot_ == 0U ? EVENT_ID4 : EVENT_ID5;
        WaitFlag<HardEvent::MTE2_V>(event);
        const uint32_t next = slot_ ^ 1U;
        if (produced_ + 1U < rows_) {
            Prefetch(row_ + 1U, next);
        }
        ComputeNoMidProducerRow<T, WITH_RMS0>(input_[slot_ * t.d], residual_[slot_ * t.k], gamma_,
                                              coefficients_[slot_ * kProducerCoefficientSlotStride], stats_, t);
        // All four sources for each column are read before stores in the shared Post VF.
        // Return input-slot credit only after MTE3 finishes reading this aliased output.
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(t.k * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(tensors_->x2Out[row_ * t.k], residual_[slot_ * t.k], copy);
        SetFlag<HardEvent::MTE3_MTE2>(event);
        slot_ = next;
        ++row_;
        ++produced_;
    }

    /** Fill the prologue batch completely before the first projection tile can read it. */
    __aicore__ inline void Fill()
    {
        while (produced_ < rows_) {
            ProduceOne();
        }
    }

    /** Interleave a host-sized next-wave row batch after a tile is published to Cube.
     * The spacing is host planned; no tiling search or arithmetic VF branching occurs here.
     */
    __aicore__ inline void Tick(uint32_t tile)
    {
        if (tile >= nextTick_ && produced_ < rows_) {
            for (uint32_t row = 0U; row < tiling_->waveProducerRowsPerTick && produced_ < rows_; ++row) {
                ProduceOne();
            }
            nextTick_ += tiling_->waveProducerTileStride;
        }
    }

    /** Publish completed Post stores, then wait for all active AIV producers before reloading rows.
     * Alternating mode-0 notifications order each wave MTE3 store before projection MTE2 loads.
     */
    __aicore__ inline void WaitReady(uint32_t wave)
    {
        // Alternating hardware notifications; every AIV participates, including empty tail members.
        // MTE3 publishes completed Post stores; MTE2 gates all subsequent projection loads.
        const uint16_t ready = kProducerWaveReadyEventBase + (wave & 1U);
        CrossCoreSetFlag<0, PIPE_MTE3>(ready);
        CrossCoreWaitFlag<0, PIPE_MTE2>(ready);
    }

    /** Consume both input-slot credits after the final producer batch to close event lifetimes. */
    __aicore__ inline void End()
    {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
    }

private:
    Kernel::Tensors<T>* tensors_;
    const MegaMhcTilingData* tiling_;
    TBuf<TPosition::VECCALC> inputBuffer_, residualBuffer_;
    TBuf<TPosition::VECCALC> gammaBuffer_, coefficientsBuffer_, statsBuffer_;
    LocalTensor<T> input_, residual_;
    LocalTensor<float> gamma_, coefficients_, stats_;
    uint64_t row_;
    uint32_t produced_, slot_, nextTick_, rows_;
};

/** Prefetch two input slots independently of the three aliased residual/output slots.
 * BS retains full K. Each AIV produces the rows owned by its paired Cube, unlike MK cyclic ownership.
 */
template <typename T, bool WITH_RMS0>
__aicore__ inline void RunBsNoMidProducer(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
{
    if ASCEND_IS_AIV {
        constexpr uint32_t kInputSlots = 2U;
        constexpr uint32_t kResidualSlots = 3U;

        TBuf<TPosition::VECCALC> inputBuffer, residualBuffer;
        TBuf<TPosition::VECCALC> gammaBuffer, coefficientBuffer, statsBuffer;
        Kernel::InitUbBuffer(pipe, inputBuffer, kInputSlots * tiling.d * sizeof(T));
        Kernel::InitUbBuffer(pipe, residualBuffer, kResidualSlots * tiling.k * sizeof(T));
        Kernel::InitUbBuffer(pipe, gammaBuffer, tiling.d * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer, Kernel::kCoefficientTileElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer, kProducerStatsElements * sizeof(float));
        auto input = inputBuffer.Get<T>();
        auto residual = residualBuffer.Get<T>();

        auto gamma = gammaBuffer.Get<float>();
        auto coefficients = coefficientBuffer.Get<float>();
        auto stats = statsBuffer.Get<float>();
        if constexpr (WITH_RMS0) {
            Kernel::CopyGlobalToUb(gamma, tensors.gamma0, 0U, tiling.d);
        }
        // Seed independent input and residual/output credits; all slots start writable.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
        const uint32_t pair = GetBlockIdx() / Kernel::kAivPerAic;
        const uint32_t rowOffset = GetSubBlockIdx() * tiling.projectionRows;
        // Count locally produced rows across waves, independently of their global row indices.
        uint32_t ordinal = 0U;
        for (uint32_t wave = 0U; wave < tiling.mWaveCount; ++wave) {
            const uint32_t mRows = Kernel::GetWaveRows(tiling, pair, wave);
            const uint32_t count =
                mRows > rowOffset ?
                    (mRows - rowOffset < tiling.projectionRows ? mRows - rowOffset : tiling.projectionRows) :
                    0U;
            const uint64_t firstRow = static_cast<uint64_t>(wave) * tiling.rowsPerWave +
                                      static_cast<uint64_t>(pair) * tiling.mTile + rowOffset;
            const uint64_t endRow = firstRow + count;
            if (firstRow < endRow) {
                const uint32_t inputSlot = ordinal & 1U;
                const uint32_t residualSlot = ordinal % kResidualSlots;
                PrefetchNoMidStagedProducer(
                    tensors, firstRow, tiling, input[inputSlot * tiling.d], residual[residualSlot * tiling.k],
                    coefficients[inputSlot * kProducerCoefficientSlotStride], inputSlot == 0U ? EVENT_ID2 : EVENT_ID3,
                    residualSlot == 0U ? EVENT_ID4 :
                    residualSlot == 1U ? EVENT_ID5 :
                                         EVENT_ID6);
            }
            for (uint64_t row = firstRow; row < endRow; ++row, ++ordinal) {
                const uint32_t inputSlot = ordinal & 1U;
                const uint32_t residualSlot = ordinal % kResidualSlots;
                const event_t inputEvent = inputSlot == 0U ? EVENT_ID2 : EVENT_ID3;
                const event_t residualEvent = residualSlot == 0U ? EVENT_ID4 :
                                              residualSlot == 1U ? EVENT_ID5 :
                                                                   EVENT_ID6;
                WaitFlag<HardEvent::MTE2_V>(residualEvent);
                const uint32_t nextInputSlot = inputSlot ^ 1U;
                if (row + 1U < endRow) {
                    const uint32_t nextResidualSlot = (ordinal + 1U) % kResidualSlots;
                    PrefetchNoMidStagedProducer(tensors, row + 1U, tiling, input[nextInputSlot * tiling.d],
                                                residual[nextResidualSlot * tiling.k],
                                                coefficients[nextInputSlot * kProducerCoefficientSlotStride],
                                                nextInputSlot == 0U ? EVENT_ID2 : EVENT_ID3,
                                                nextResidualSlot == 0U ? EVENT_ID4 :
                                                nextResidualSlot == 1U ? EVENT_ID5 :
                                                                         EVENT_ID6);
                }
                ComputeNoMidProducerRow<T, WITH_RMS0>(input[inputSlot * tiling.d], residual[residualSlot * tiling.k],
                                                      gamma, coefficients[inputSlot * kProducerCoefficientSlotStride],
                                                      stats, tiling);
                // Input and coefficients are no longer read; their next load need not wait for MTE3.
                SetFlag<HardEvent::V_MTE2>(inputEvent);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyExtParams copy{1U, static_cast<uint32_t>(tiling.k * sizeof(T)), 0U, 0U, 0U};
                DataCopyPad(tensors.x2Out[row * tiling.k], residual[residualSlot * tiling.k], copy);
                SetFlag<HardEvent::MTE3_MTE2>(residualEvent);
            }
        }
        // Consume final or unused initial credits before the next phase reuses these event IDs.
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID6);
    }
}

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_PRODUCER_H
