/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_MM_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_MM_ARCH35_H

#include "kernel_operator.h"
#include "mhc_pre_sinkhorn_backward_vector_arch35.h"

namespace MhcPreSinkhornBackwardNs {
using namespace AscendC;

constexpr uint32_t CUBE_M = 16;
constexpr uint32_t MM_TILE_M = ROWS_PER_AIV * 2;
constexpr uint32_t MM_K0 = 8;
// Number of elements in a 16x8 float fractal.
constexpr uint32_t FRACTAL_ELEMENTS = 128;
constexpr uint32_t MM1_L0_EVENT_BASE = 3;
constexpr uint32_t MM2_L0_EVENT_BASE = 5;
constexpr uint32_t MM1_MTE1_M_EVENT_BASE = 0;
constexpr uint32_t MM2_MTE1_M_EVENT_BASE = 2;
constexpr uint32_t MM2_C_EVENT = 2;
constexpr FixpipeConfig MHC_BWD_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

class MhcPreSinkhornBackwardMm {
    MhcBwdShape shape_;

public:
    __aicore__ inline void Init(const MhcPreSinkhornBackwardArch35TilingData* tiling)
    {
        shape_.Init(tiling->n);
        configMm1_ = tiling->mmConfigMm1;
        configMm2_ = tiling->mmConfigMm2;
        l1SingleBufferElems_ = tiling->l1SingleBufferElems;
        l0ABSingleBufferElems_ = tiling->l0ABSingleBufferElems;
        l0CSingleBufferElems_ = tiling->l0CSingleBufferElems;
        // Partition L0A/L0B into separate MM1/MM2 ping-pong pools, using slot sizes and offsets from tiling data.
        aMm1L0_ = LocalTensor<float>(TPosition::A2, 0, l0ABSingleBufferElems_ * configMm1_.dbL0A);
        bMm1L0_ = LocalTensor<float>(TPosition::B2, 0, l0ABSingleBufferElems_ * configMm1_.dbL0A);
        uint32_t mm1L0Bytes = l0ABSingleBufferElems_ * configMm1_.dbL0A * sizeof(float);
        aMm2L0_ = LocalTensor<float>(TPosition::A2, mm1L0Bytes, l0ABSingleBufferElems_ * configMm2_.dbL0A);
        bMm2L0_ = LocalTensor<float>(TPosition::B2, mm1L0Bytes, l0ABSingleBufferElems_ * configMm2_.dbL0A);
        // Partition L0C into MM1 ping-pong slots and MM2 accumulation slots for N chunks.
        cMm1L0_ = LocalTensor<float>(TPosition::CO1, 0, l0CSingleBufferElems_ * configMm1_.dbL0C);
        cMm2L0_ = LocalTensor<float>(TPosition::CO1, l0CSingleBufferElems_ * configMm1_.dbL0C * sizeof(float),
                                     l0CSingleBufferElems_ * MM2CSlots());

        // Initial tokens mark phi B1 slots as writable, L0A/B slots as loadable, and L0C slots as writable.
        for (uint32_t slot = 0; slot < configMm1_.depthB1; ++slot) {
            SetFlag<HardEvent::MTE1_MTE2>(slot);
        }
        for (uint32_t slot = 0; slot < configMm1_.dbL0A; ++slot) {
            SetFlag<HardEvent::M_MTE1>(MM1_L0_EVENT_BASE + slot);
        }
        for (uint32_t slot = 0; slot < configMm2_.dbL0A; ++slot) {
            SetFlag<HardEvent::M_MTE1>(MM2_L0_EVENT_BASE + slot);
        }
        for (uint32_t slot = 0; slot < configMm1_.dbL0C; ++slot) {
            SetFlag<HardEvent::FIX_M>(slot);
        }
        SetFlag<HardEvent::FIX_M>(MM2_C_EVENT);
        oldHf32Mode_ = GetCtrlSpr<HF32_MODE_BIT, HF32_MODE_BIT>();
        oldHf32TransMode_ = GetCtrlSpr<HF32_TRANS_MODE_BIT, HF32_TRANS_MODE_BIT>();
        SetHF32Mode(0);
        SetHF32TransMode(0);
    }

    // Preload phi into B1 ping-pong slots, overlapping computation of the current final wave.
    __aicore__ inline LocalTensor<float> PreloadPhi(const GlobalTensor<float>& phi, const LocalTensor<float>& b1Region,
                                                    uint32_t fullK, uint32_t kBegin, uint32_t kValid)
    {
        phiSlot_ = phiPingPong_ % configMm1_.depthB1;
        LocalTensor<float> dst = b1Region[phiSlot_ * l1SingleBufferElems_];
        // MTE2 may overwrite the slot only after MTE1 finishes reading the previous phi data.
        WaitFlag<HardEvent::MTE1_MTE2>(phiSlot_);
        // Zero-pad phi rows for NZ loading; MM1 computes only kMM1 elements along K.
        InitConstValueParams<float> zero(1, l1SingleBufferElems_ / FLOAT_C0, 0, 0.0f);
        Fill(dst, zero);
        PipeBarrier<PIPE_MTE2>(); // Wait for zero-padding to finish before writing to the same phi slot.
        Nd2NzParams params;
        params.ndNum = 1;
        params.srcNdMatrixStride = 1;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 1;
        params.nValue = shape_.hValue;
        params.dValue = kValid;
        params.srcDValue = fullK;
        params.dstNzC0Stride = shape_.hAlign;
        DataCopy(dst, phi[kBegin], params);
        SetFlag<HardEvent::MTE2_MTE1>(phiSlot_);
        ++phiPingPong_;
        return dst;
    }

    __aicore__ inline void ComputeMm1(const LocalTensor<float>& gqL1, const LocalTensor<float>& phiL1,
                                      const LocalTensor<float>& dst, uint32_t kValid, bool lastPhiUse = true,
                                      bool firstPhiUse = true)
    {
        const uint32_t redK = shape_.kMM1;
        const uint32_t baseK = (configMm1_.baseK == 0 || configMm1_.baseK >= redK) ? redK : configMm1_.baseK;
        const bool kChunked = baseK < redK;
        const uint32_t chunkCount = CeilDivU32(kValid, configMm1_.baseN);
        uint32_t l0Iter = 0;
        for (uint32_t offsetN = 0; offsetN < kValid; offsetN += configMm1_.baseN) {
            uint32_t currentN = kValid - offsetN < configMm1_.baseN ? kValid - offsetN : configMm1_.baseN;
            uint32_t chunkN = offsetN / configMm1_.baseN;
            uint32_t cSlot = chunkN % configMm1_.dbL0C;
            LocalTensor<float> cCur = cMm1L0_[cSlot * l0CSingleBufferElems_];
            const uint32_t nOff = NChunkOffset(shape_.hAlign, offsetN);
            // Accumulate all K chunks of a column chunk into one L0C slot; ensure the slot is free before use.
            WaitFlag<HardEvent::FIX_M>(cSlot);
            for (uint32_t offsetK = 0; offsetK < redK; offsetK += baseK) {
                uint32_t currentK = redK - offsetK < baseK ? redK - offsetK : baseK;
                uint32_t l0Slot = l0Iter % configMm1_.dbL0A;
                ++l0Iter;
                LocalTensor<float> a2 = aMm1L0_[l0Slot * l0ABSingleBufferElems_];
                LocalTensor<float> b2 = bMm1L0_[l0Slot * l0ABSingleBufferElems_];
                WaitFlag<HardEvent::M_MTE1>(MM1_L0_EVENT_BASE + l0Slot);
                LoadA(a2, gqL1[KOffA(offsetK)], MM_TILE_M, currentK, false);
                if (firstPhiUse && chunkN == 0 && offsetK == 0) {
                    // MTE1 must wait for phi B1 data; issue LoadA first to overlap the preload transfer.
                    WaitFlag<HardEvent::MTE2_MTE1>(phiSlot_);
                }
                // With K tiling, L0B holds one K slice and cannot reuse it across GQ waves; reload each slice from L1.
                if (kChunked || firstPhiUse || configMm1_.dbL0A < chunkCount) {
                    LoadB(b2, phiL1[nOff + KOffB(offsetK)], currentK, currentN, shape_.hAlign);
                }
                SetFlag<HardEvent::MTE1_M>(MM1_MTE1_M_EVENT_BASE + l0Slot);
                WaitFlag<HardEvent::MTE1_M>(MM1_MTE1_M_EVENT_BASE + l0Slot);
                Compute(cCur, a2, b2, MM_TILE_M, currentN, currentK, offsetK == 0);
                SetFlag<HardEvent::M_MTE1>(MM1_L0_EVENT_BASE + l0Slot);
            }
            SetFlag<HardEvent::M_FIX>(cSlot);
            WaitFlag<HardEvent::M_FIX>(cSlot);
            FixpipeMm1(dst, cCur, offsetN, currentN);
            SetFlag<HardEvent::FIX_M>(cSlot);
        }
        if (lastPhiUse) {
            // Release the phi B1 slot after MTE1 finishes reading it, allowing ping-pong prefetch of the next kTile.
            SetFlag<HardEvent::MTE1_MTE2>(phiSlot_);
        }
    }

    // kRows is the valid GQ row count aligned to 16; MM2 accumulates only these rows.
    __aicore__ inline void ComputeMm2(const LocalTensor<float>& gqL1, const LocalTensor<float>& xL1, uint32_t kValid,
                                      bool firstM = true, bool lastM = true, uint32_t kRows = MM_TILE_M)
    {
        if (firstM) {
            // Reinitialize N-chunk accumulation slots only after transferring the previous kTile's MM2 results.
            WaitFlag<HardEvent::FIX_M>(MM2_C_EVENT);
        }
        for (uint32_t offsetN = 0; offsetN < kValid; offsetN += configMm2_.baseN) {
            uint32_t currentN = kValid - offsetN < configMm2_.baseN ? kValid - offsetN : configMm2_.baseN;
            uint32_t chunk = offsetN / configMm2_.baseN;
            uint32_t l0Slot = chunk % configMm2_.dbL0A;
            LocalTensor<float> a2 = aMm2L0_[l0Slot * l0ABSingleBufferElems_];
            LocalTensor<float> b2 = bMm2L0_[l0Slot * l0ABSingleBufferElems_];
            WaitFlag<HardEvent::M_MTE1>(MM2_L0_EVENT_BASE + l0Slot);
            LoadA(a2, gqL1, MM_TILE_M, shape_.hValue, true);
            LoadB(b2, xL1[NChunkOffset(MM_TILE_M, offsetN)], MM_TILE_M, currentN, MM_TILE_M);
            SetFlag<HardEvent::MTE1_M>(MM2_MTE1_M_EVENT_BASE + l0Slot);
            WaitFlag<HardEvent::MTE1_M>(MM2_MTE1_M_EVENT_BASE + l0Slot);
            LocalTensor<float> cCur = cMm2L0_[chunk * l0CSingleBufferElems_];
            // M=MM2_M pads odd hcMix with one zero row so the two AIV subcores receive equal whole rows with
            // dualDstCtl=1.
            Compute(cCur, a2, b2, shape_.mm2M, currentN, kRows, firstM);
            SetFlag<HardEvent::M_MTE1>(MM2_L0_EVENT_BASE + l0Slot);
        }
        if (lastM) {
            SetFlag<HardEvent::M_FIX>(MM2_C_EVENT);
        }
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<float>& dst, const LocalTensor<float>& cCur, uint32_t offsetN,
                                      uint32_t currentN)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
        params.nSize = AlignU32(currentN, MM_K0);
        params.mSize = MM_TILE_M;
        params.dstStride = K_TILE_VALUE;
        params.srcStride = AlignU32(MM_TILE_M, CUBE_M);
        params.quantPre = QuantMode_t::NoQuant;
        params.dualDstCtl = 1;
        params.unitFlag = 0;
        params.params.ndNum = 1;
        params.params.srcNdStride = 1;
        params.params.dstNdStride = 1;
        Fixpipe<float, float, MHC_BWD_CFG_ROW_MAJOR_UB>(dst[offsetN], cCur, params);
    }

    __aicore__ inline void CopyMm2ToAiv(const LocalTensor<float>& dst, uint32_t kValid)
    {
        WaitFlag<HardEvent::M_FIX>(MM2_C_EVENT);
        for (uint32_t offsetN = 0; offsetN < kValid; offsetN += configMm2_.baseN) {
            uint32_t currentN = kValid - offsetN < configMm2_.baseN ? kValid - offsetN : configMm2_.baseN;
            uint32_t chunk = offsetN / configMm2_.baseN;
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> params;
            params.nSize = AlignU32(currentN, MM_K0);
            params.mSize = shape_.mm2M;
            params.dstStride = K_TILE_VALUE;
            params.srcStride = AlignU32(shape_.mm2M, CUBE_M);
            params.quantPre = QuantMode_t::NoQuant;
            params.dualDstCtl = 1;
            params.unitFlag = 0;
            params.params.ndNum = 1;
            params.params.srcNdStride = 1;
            params.params.dstNdStride = 1;
            Fixpipe<float, float, MHC_BWD_CFG_ROW_MAJOR_UB>(dst[offsetN], cMm2L0_[chunk * l0CSingleBufferElems_],
                                                            params);
        }
        SetFlag<HardEvent::FIX_M>(MM2_C_EVENT);
    }

    __aicore__ inline void CopyMm2ToGm(const GlobalTensor<float>& dst, uint32_t kValid, uint32_t dstStride)
    {
        WaitFlag<HardEvent::M_FIX>(MM2_C_EVENT);
        for (uint32_t offsetN = 0; offsetN < kValid; offsetN += configMm2_.baseN) {
            uint32_t currentN = kValid - offsetN < configMm2_.baseN ? kValid - offsetN : configMm2_.baseN;
            uint32_t chunk = offsetN / configMm2_.baseN;
            // Write only hcMix valid rows to GM; keep the final padded row for odd N in L0C.
            CopyToGm(dst[offsetN], cMm2L0_[chunk * l0CSingleBufferElems_], shape_.hValue, currentN, dstStride);
        }
        SetFlag<HardEvent::FIX_M>(MM2_C_EVENT);
    }

    __aicore__ inline void End()
    {
        // Reclaim all buffer tokens issued by Init.
        for (uint32_t slot = 0; slot < configMm1_.depthB1; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(slot);
        }
        for (uint32_t slot = 0; slot < configMm1_.dbL0A; ++slot) {
            WaitFlag<HardEvent::M_MTE1>(MM1_L0_EVENT_BASE + slot);
        }
        for (uint32_t slot = 0; slot < configMm2_.dbL0A; ++slot) {
            WaitFlag<HardEvent::M_MTE1>(MM2_L0_EVENT_BASE + slot);
        }
        for (uint32_t slot = 0; slot < configMm1_.dbL0C; ++slot) {
            WaitFlag<HardEvent::FIX_M>(slot);
        }
        WaitFlag<HardEvent::FIX_M>(MM2_C_EVENT);
        SetHF32TransMode(oldHf32TransMode_);
        SetHF32Mode(oldHf32Mode_);
    }

private:
    __aicore__ inline uint32_t MM2CSlots() const
    {
        return CeilDivU32(K_TILE_VALUE, configMm2_.baseN);
    }

    // L1 NZ uses 8-column N fractals and 16-row K fractals; the N-chunk offset is column-fractal index * mStep *
    // fractal size.
    __aicore__ inline uint32_t NChunkOffset(uint32_t kRows, uint32_t offsetN) const
    {
        return offsetN / MM_K0 * CeilDivU32(kRows, CUBE_M) * FRACTAL_ELEMENTS;
    }

    // gq L1 NZ layout is [k/8][M/16][16][8]; advance the reduction axis in 8-row fractals.
    __aicore__ inline uint32_t KOffA(uint32_t offsetK) const
    {
        return offsetK / MM_K0 * CeilDivU32(MM_TILE_M, CUBE_M) * FRACTAL_ELEMENTS;
    }

    // phi L1 NZ layout is [n/8][k/16][16][8]; the reduction axis is contiguous in 16-row chunks within each N fractal.
    __aicore__ inline uint32_t KOffB(uint32_t offsetK) const
    {
        return offsetK / CUBE_M * FRACTAL_ELEMENTS;
    }

    __aicore__ inline void LoadA(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t originalM,
                                 uint32_t originalK, bool transpose)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        if (transpose) {
            params.mStep = CeilDivU32(originalM, CUBE_M);
            params.kStep = AlignU32(CeilDivU32(originalK, MM_K0), 2);
            params.srcStride = CeilDivU32(originalM, CUBE_M);
            params.dstStride = CeilDivU32(originalK, CUBE_M);
            params.ifTranspose = true;
        } else {
            params.mStep = CeilDivU32(originalM, CUBE_M);
            params.kStep = CeilDivU32(originalK, MM_K0);
            params.srcStride = params.mStep;
            params.dstStride = params.mStep;
            params.ifTranspose = false;
        }
        LoadData(dst, src, params);
    }

    __aicore__ inline void LoadB(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t k, uint32_t n,
                                 uint32_t kL1)
    {
        LoadData2DParamsV2 params;
        params.mStartPosition = 0;
        params.kStartPosition = 0;
        params.mStep = CeilDivU32(k, CUBE_M);
        params.kStep = AlignU32(CeilDivU32(n, MM_K0), 2);
        // srcStride spans adjacent N fractals; use the full L1 K length (kL1), not the current K slice length.
        params.srcStride = CeilDivU32(kL1, CUBE_M);
        params.dstStride = CeilDivU32(n, CUBE_M);
        params.ifTranspose = true;
        LoadData(dst, src, params);
    }

    __aicore__ inline void Compute(const LocalTensor<float>& c, const LocalTensor<float>& a,
                                   const LocalTensor<float>& b, uint32_t m, uint32_t n, uint32_t k, bool initC)
    {
        MmadParams params;
        params.m = m;
        params.n = n;
        params.k = k;
        params.disableGemv = true;
        params.cmatrixInitVal = initC;
        params.cmatrixSource = false;
        // unitFlag=0: M_FIX and FIX_M events fully control L0C production and consumption.
        params.unitFlag = 0;
        Mmad(c, a, b, params);
    }

    __aicore__ inline void CopyToGm(const GlobalTensor<float>& dst, const LocalTensor<float>& src, uint32_t m,
                                    uint32_t n, uint32_t dstStride)
    {
        DataCopyCO12DstParams params;
        params.nSize = n;
        params.mSize = m;
        params.dstStride = dstStride;
        params.srcStride = AlignU32(shape_.mm2M, CUBE_M);
        params.quantPre = QuantMode_t::NoQuant;
        params.nz2ndEn = true;
        params.unitFlag = 0;
        SetFixpipeNz2ndFlag(1, 1, 1);
        DataCopy(dst, src, params);
    }

    MhcPreSinkhornBackwardMMConfig configMm1_;
    MhcPreSinkhornBackwardMMConfig configMm2_;
    uint32_t l1SingleBufferElems_ = 0;
    uint32_t l0ABSingleBufferElems_ = 0;
    uint32_t l0CSingleBufferElems_ = 0;
    uint32_t phiSlot_ = 0;
    uint32_t phiPingPong_ = 0;
    LocalTensor<float> aMm1L0_;
    LocalTensor<float> bMm1L0_;
    LocalTensor<float> aMm2L0_;
    LocalTensor<float> bMm2L0_;
    LocalTensor<float> cMm1L0_;
    LocalTensor<float> cMm2L0_;
    int64_t oldHf32Mode_ = 0;
    int64_t oldHf32TransMode_ = 0;
};
} // namespace MhcPreSinkhornBackwardNs

#endif // MHC_PRE_SINKHORN_BACKWARD_MM_ARCH35_H
