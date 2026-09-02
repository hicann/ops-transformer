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
 * \file quant_grouped_matmul_dequant_phase_d.h
 * \brief Out-of-class definitions for Phase D helpers (LoadScaleToUB,
 *        DequantAndScatter).
 *
 * The bulk of Phase D work (VDEQ16 + scatter + pertoken Mul + CopyOut) lives
 * inline in ProcessColBatch (_phase_mm.h). Included from
 * quant_grouped_matmul_dequant.h AFTER the class template declaration.
 * Reopens namespace AscendC for the member function bodies.
 */
#ifndef QUANT_GROUPED_MATMUL_DEQUANT_PHASE_D_H
#define QUANT_GROUPED_MATMUL_DEQUANT_PHASE_D_H

namespace AscendC {

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_,
                                                 UseXZNStage_>::LoadScaleToUB(uint32_t groupIdx, uint32_t cbIdx,
                                                                              uint64_t curColCount, bool usePing)
{
    // MTE2 scale GM -> UB into the ping or pong uint64 slot.
    //   ScaleIsU64 = true  : direct uint64 copy; VDEQ16 reads it as-is.
    //   ScaleIsU64 = false : fp32 staging, V-pipe encode fills curScU64.
    const uint32_t nSize = static_cast<uint32_t>(curColCount);
    const uint32_t nStart = cbIdx * Nb_;
    const uint64_t scaleOff = static_cast<uint64_t>(groupIdx) * N_ + nStart;
    LocalTensor<uint64_t> curScU64 = usePing ? scaleU64PingLT_ : scaleU64PongLT_;

    if constexpr (ScaleIsU64) {
        // Plain DataCopy<uint64>. VDEQ16 reads curScU64 later via
        // vdeqP.deqTensorAddr; the cb-level MTE2_V(4) Set/Wait is the
        // MTE2->V sync, so no local Set/Wait is needed here.
        DataCopy<uint64_t>(curScU64, weightScaleU64Gm_[scaleOff], nSize);
    } else {
        DataCopy<float>(scaleFp32StagingLT_, weightScaleFp32Gm_[scaleOff], nSize);
        // V-pipe encode: fp32 staging -> uint64 VDEQ16 slot.
        SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
        EncodeFp32ScaleToUint64(curScU64, scaleFp32StagingLT_, scatterIdxLT_, nSize);
    }

    // Defensive zero-pad of the scale buffer tail for partial cbs
    // (nSize < Nb_) so stale tail bytes never pollute scale-strided access.
    // No-op for full cbs (the common case).
    if (nSize < Nb_) {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
        LocalTensor<half> curScHalf = curScU64.template ReinterpretCast<half>();
        Duplicate<half>(curScHalf[nSize * 4], static_cast<half>(0.0f), static_cast<int32_t>((Nb_ - nSize) * 4));
        SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
    }
}

template <bool HasPertoken, bool HasSmooth, bool ScaleIsU64, bool BlockedZN_, bool SmallM_, bool UseXZNStage_>
__aicore__ inline void QuantGroupedMatmulDequant<HasPertoken, HasSmooth, ScaleIsU64, BlockedZN_, SmallM_,
                                                 UseXZNStage_>::DequantAndScatter(uint64_t mCount, uint64_t curColCount,
                                                                                  bool usePing)
{
    (void)mCount;
    (void)curColCount;
    (void)usePing;
    // VDEQ16 L0C->UB, NZ->ND scatter, pertoken bulk Mul.
}

} // namespace AscendC

#endif // QUANT_GROUPED_MATMUL_DEQUANT_PHASE_D_H
