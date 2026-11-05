/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_COMMON_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mega_mhc_tiling_key.h"
#include "compute/rms_norm_common.h"

namespace MegaMhc::Kernel {
constexpr uint32_t kUbAlignmentBytes = 256U;
constexpr uint32_t kUbAlignmentRounding = 255U;
constexpr uint32_t kFp32VectorElements = 64U;
constexpr uint32_t kFp32VectorRounding = 63U;
constexpr uint32_t kHeadCount = 4U;
constexpr uint32_t kHeadMatrixElements = 16U;
constexpr uint32_t kRowInputSlots = 2U;
constexpr uint32_t kRowHeadSlots = 8U;
constexpr uint32_t kRowGammaSlots = 2U;
constexpr uint32_t kCoefficientTileElements = 256U;
constexpr uint32_t kRmsStatElements = 8U;
constexpr uint32_t kAivPerAic = 2U;

using namespace AscendC;

constexpr uint32_t kRowsPerMSubblock = 16;
constexpr uint16_t kPairReadyFlag = 8;
constexpr uint16_t kPairConsumedFlag = 9;
constexpr uint16_t kSubblockFlagOffset = 16;
constexpr uint8_t kPairSyncMode = 4;

template <typename T>
struct Tensors {
    GlobalTensor<T> hOut;
    GlobalTensor<T> residual;
    GlobalTensor<float> hPost;
    GlobalTensor<float> hRes;
    GlobalTensor<float> phi;
    GlobalTensor<float> alpha;
    GlobalTensor<float> bias;
    GlobalTensor<float> gamma1;
    GlobalTensor<float> gamma0;
    GlobalTensor<float> gamma2;
    GlobalTensor<T> hInPrime;
    GlobalTensor<float> hPostNew;
    GlobalTensor<float> hResNew;
    GlobalTensor<T> x2Out;
    GlobalTensor<float> partialMm;
    GlobalTensor<float> partialRms;
    GlobalTensor<float> stage;
};

/** Allocate UB with 256B padding: each DMA base is aligned and small vectors can load a full register. */
__aicore__ inline void InitUbBuffer(TPipe* pipe, TBuf<TPosition::VECCALC>& buffer, uint32_t bytes)
{
    pipe->InitBuffer(buffer, (bytes + kUbAlignmentRounding) / kUbAlignmentBytes * kUbAlignmentBytes);
}

/** Enqueue GM -> UB; the caller manages reuse and completion events. */
template <typename T>
__aicore__ inline void EnqueueGlobalToUb(const LocalTensor<T>& destination, const GlobalTensor<T>& source,
                                         uint64_t sourceOffset, uint32_t elementCount)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(destination, source[sourceOffset], copyParams, padParams);
}

/** Copy GM -> UB and wait for MTE2_V readiness. */
template <typename T>
__aicore__ inline void CopyGlobalToUb(const LocalTensor<T>& destination, const GlobalTensor<T>& source,
                                      uint64_t sourceOffset, uint32_t elementCount)
{
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    EnqueueGlobalToUb(destination, source, sourceOffset, elementCount);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

/** Copy UB -> GM and wait for MTE3 completion. */
template <typename T>
__aicore__ inline void CopyUbToGlobal(const GlobalTensor<T>& destination, uint64_t destinationOffset,
                                      const LocalTensor<T>& source, uint32_t elementCount)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(elementCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(destination[destinationOffset], source, copyParams);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

/** Initialize one or more FP32 values before an ordered Vector reduction. */
__aicore__ inline void ZeroFp32Values(__ubuf__ float* values, uint32_t valueCount)
{
    const uint16_t repeats = static_cast<uint16_t>((valueCount + kFp32VectorRounding) / kFp32VectorElements);
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<float> zero;
        Reg::MaskReg fullMask = Compute::MaskForCount(kFp32VectorElements);
        Reg::Duplicate(zero, 0.0f, fullMask);
        uint32_t remaining = valueCount;
        for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
            mask = Reg::UpdateMask<float>(remaining);
            Reg::StoreAlign<float>(values + repeat * kFp32VectorElements, zero, mask);
        }
    }
}

/** Locate a shard from host-planned long/short lengths: offset=min(i,L)*long+max(i-L,0)*short. */
__aicore__ inline uint64_t PlannedKStart(const MegaMhcTilingData& tiling, uint32_t kIndex)
{
    return kIndex < tiling.largeKCoreCount ?
               static_cast<uint64_t>(kIndex) * tiling.splitK :
               static_cast<uint64_t>(tiling.largeKCoreCount) * tiling.splitK +
                   static_cast<uint64_t>(kIndex - tiling.largeKCoreCount) * tiling.shortSplitK;
}

/** Read the Host-planned L1 tile count for a K shard. */
__aicore__ inline uint32_t PlannedL1TileCount(const MegaMhcTilingData& tiling, uint32_t kIndex)
{
    const bool lastKCore = kIndex + 1U == tiling.kGrid;
    const uint32_t fullLoops =
        lastKCore ? tiling.lastKCoreL1FullLoops :
                    (kIndex < tiling.largeKCoreCount ? tiling.l1KFullLoops : tiling.shortKCoreL1FullLoops);
    const uint32_t tail = lastKCore ? tiling.lastKCoreL1Tail : tiling.l1KTail;
    return fullLoops + static_cast<uint32_t>(tail != 0U);
}

/** Read the Host-planned L1 tile K length. */
__aicore__ inline uint32_t PlannedL1TileKCount(const MegaMhcTilingData& tiling, uint32_t kIndex, uint32_t tileIndex)
{
    const bool lastKCore = kIndex + 1U == tiling.kGrid;
    const uint32_t fullLoops =
        lastKCore ? tiling.lastKCoreL1FullLoops :
                    (kIndex < tiling.largeKCoreCount ? tiling.l1KFullLoops : tiling.shortKCoreL1FullLoops);
    const uint32_t tail = lastKCore ? tiling.lastKCoreL1Tail : tiling.l1KTail;
    return tileIndex < fullLoops ? tiling.l1K : tail;
}

/** Select the precomputed K and L0 descriptor for one L1 transfer index. */
__aicore__ inline void GetPlannedL1Tile(const MegaMhcTilingData& tiling, uint32_t kIndex, uint32_t tileIndex,
                                        uint32_t& kCount, uint32_t& l0FullLoops, uint32_t& l0Tail,
                                        uint32_t& tailLoadKStep)
{
    const bool lastKCore = kIndex + 1U == tiling.kGrid;
    const uint32_t fullLoops =
        lastKCore ? tiling.lastKCoreL1FullLoops :
                    (kIndex < tiling.largeKCoreCount ? tiling.l1KFullLoops : tiling.shortKCoreL1FullLoops);
    const uint32_t tail = lastKCore ? tiling.lastKCoreL1Tail : tiling.l1KTail;
    if (tileIndex < fullLoops) {
        kCount = tiling.l1K;
        l0FullLoops = tiling.l0KFullLoops;
        l0Tail = tiling.l0KTail;
        tailLoadKStep = tiling.l0TailLoadKStep;
    } else {
        kCount = tail;
        if (lastKCore) {
            l0FullLoops = tiling.lastCoreTailL1L0FullLoops;
            l0Tail = tiling.lastCoreTailL1L0Tail;
            tailLoadKStep = tiling.lastCoreTailLoadKStep;
        } else {
            l0FullLoops = tiling.tailL1L0FullLoops;
            l0Tail = tiling.tailL1L0Tail;
            tailLoadKStep = tiling.tailL1TailLoadKStep;
        }
    }
}

/** Read the valid M rows, including the last-wave tail. */
__aicore__ inline uint32_t GetWaveRows(const MegaMhcTilingData& tiling, uint32_t mIndex, uint32_t wave)
{
    if (wave + 1U != tiling.mWaveCount) {
        return tiling.mTile;
    }
    if (mIndex >= tiling.lastWaveMGrid) {
        return 0U;
    }
    return mIndex + 1U == tiling.lastWaveMGrid ? tiling.lastMCoreRows : tiling.mTile;
}

/** Own two capacity-planned D slots shared by all row phases.
 * input[slot,D], heads[slot,4,D], gamma[slot,D] hold only one host-planned D tile.
 * MTE2 prefetches the alternate slot during VF work; reuse follows V and the synchronous output store.
 */
template <typename T>
class TiledRowBuffers {
public:
    /** Allocate exactly the host budget: 28*tile bytes plus padded coefficients and scalar statistics. */
    __aicore__ inline void Init(TPipe* pipe, const MegaMhcTilingData& tiling)
    {
        plan_ = &tiling;
        Kernel::InitUbBuffer(pipe, inputBuffer_, kRowInputSlots * tiling.dTile * sizeof(T));
        Kernel::InitUbBuffer(pipe, headsBuffer_, kRowHeadSlots * tiling.dTile * sizeof(T));
        Kernel::InitUbBuffer(pipe, gammaBuffer_, kRowGammaSlots * tiling.dTile * sizeof(float));
        Kernel::InitUbBuffer(pipe, coefficientBuffer_, kCoefficientTileElements * sizeof(float));
        Kernel::InitUbBuffer(pipe, statsBuffer_, kRmsStatElements * sizeof(float));
        coefficients = coefficientBuffer_.template Get<float>();
        stats = statsBuffer_.template Get<float>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
    }

    /** Return the valid tail length supplied by Host; all other tiles have the same physical stride. */
    __aicore__ inline uint32_t Count(uint32_t tile) const
    {
        return tile + 1U == plan_->dTileCount ? plan_->dTileTail : plan_->dTile;
    }

    /** Input/output share a ping-pong slot; input may hold rounded x1 or hIn. */
    __aicore__ inline LocalTensor<T> Input(uint32_t tile)
    {
        return inputBuffer_.template Get<T>()[(tile & 1U) * plan_->dTile];
    }

    /** Select the four-head slot, whose head stride stays dTile even in the final short tile. */
    __aicore__ inline LocalTensor<T> Heads(uint32_t tile)
    {
        return headsBuffer_.template Get<T>()[(tile & 1U) * kHeadCount * plan_->dTile];
    }

    /** Select the tile weight slot. */
    __aicore__ inline LocalTensor<float> Gamma(uint32_t tile)
    {
        return gammaBuffer_.template Get<float>()[(tile & 1U) * plan_->dTile];
    }

    /** Enqueue a normalization tile: input=source[base+offset], optionally gamma=weight[weightBase+offset]. */
    template <bool WITH_WEIGHT>
    __aicore__ inline void PrefetchNorm(const GlobalTensor<T>& source, uint64_t base, const GlobalTensor<float>& weight,
                                        uint64_t weightBase, uint32_t tile)
    {
        const event_t event = Event(tile);
        WaitFlag<HardEvent::V_MTE2>(event);
        const uint64_t offset = static_cast<uint64_t>(tile) * plan_->dTile;
        Kernel::EnqueueGlobalToUb(Input(tile), source, base + offset, Count(tile));
        if constexpr (WITH_WEIGHT) {
            Kernel::EnqueueGlobalToUb(Gamma(tile), weight, weightBase + offset, Count(tile));
        }
        SetFlag<HardEvent::MTE2_V>(event);
    }

    /** Enqueue all four source heads once; Post additionally loads the RMS0 source and weight tile. */
    template <bool POST, bool WITH_RMS0 = true>
    __aicore__ inline void PrefetchHeads(Kernel::Tensors<T>& tensors, uint64_t row, uint32_t tile)
    {
        const event_t event = Event(tile);
        WaitFlag<HardEvent::V_MTE2>(event);
        const uint64_t offset = static_cast<uint64_t>(tile) * plan_->dTile;
        if constexpr (POST) {
            Kernel::EnqueueGlobalToUb(Input(tile), tensors.hOut, row * plan_->d + offset, Count(tile));
            if constexpr (WITH_RMS0) {
                Kernel::EnqueueGlobalToUb(Gamma(tile), tensors.gamma0, offset, Count(tile));
            }
        }
        const auto& source = POST ? tensors.residual : tensors.x2Out;
        for (uint32_t head = 0U; head < kHeadCount; ++head) {
            Kernel::EnqueueGlobalToUb(Heads(tile)[head * plan_->dTile], source,
                                      row * plan_->k + static_cast<uint64_t>(head) * plan_->d + offset, Count(tile));
        }
        SetFlag<HardEvent::MTE2_V>(event);
    }

    /** Wait for the tile DMA. */
    __aicore__ inline void Wait(uint32_t tile)
    {
        WaitFlag<HardEvent::MTE2_V>(Event(tile));
    }

    /** Return the slot after its VF and any CopyUbToGlobal store have completed. */
    __aicore__ inline void Release(uint32_t tile)
    {
        SetFlag<HardEvent::V_MTE2>(Event(tile));
    }

    /** Consume both credits, including an unused slot when there is only one tile or no owned rows. */
    __aicore__ inline void End()
    {
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
    }

    LocalTensor<float> coefficients, stats;

private:
    /** Map the tile to its DMA/V event slot. */
    __aicore__ inline event_t Event(uint32_t tile) const
    {
        return (tile & 1U) == 0U ? EVENT_ID2 : EVENT_ID3;
    }
    const MegaMhcTilingData* plan_;
    TBuf<TPosition::VECCALC> inputBuffer_, headsBuffer_, gammaBuffer_, coefficientBuffer_, statsBuffer_;
};

/** Apply a completed row statistic to consecutive D tiles and preserve the Cast<T> boundary.
 * Formula: destination[i]=Cast<T>(FP32(source[i])*inverse*weight[i]). Source and destination may alias in GM.
 * Only alternate tile loads overlap computation; each store completes before its slot is released.
 */
template <typename T>
__aicore__ inline void ApplyTiledNormalization(TiledRowBuffers<T>& buffer, const MegaMhcTilingData& tiling,
                                               const GlobalTensor<T>& source, const GlobalTensor<T>& destination,
                                               uint64_t base, const GlobalTensor<float>& weight, uint64_t weightBase,
                                               __ubuf__ float* inverse)
{
    buffer.template PrefetchNorm<true>(source, base, weight, weightBase, 0U);
    for (uint32_t tile = 0U; tile < tiling.dTileCount; ++tile) {
        buffer.Wait(tile);
        if (tile + 1U < tiling.dTileCount) {
            buffer.template PrefetchNorm<true>(source, base, weight, weightBase, tile + 1U);
        }
        auto input = buffer.Input(tile);
        auto gamma = buffer.Gamma(tile);
        Compute::ApplyRmsNorm<T>(reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()), buffer.Count(tile), inverse,
                                 reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()));
        Kernel::CopyUbToGlobal(destination, base + static_cast<uint64_t>(tile) * tiling.dTile, input,
                               buffer.Count(tile));
        buffer.Release(tile);
    }
}

} // namespace MegaMhc::Kernel

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_COMMON_H
