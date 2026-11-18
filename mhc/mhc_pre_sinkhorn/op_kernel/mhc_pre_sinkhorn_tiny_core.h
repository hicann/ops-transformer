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
 * \file mhc_pre_sinkhorn_tiny_core.h
 * \brief key=3 tiny kernel for small-batch (decode) inference shapes.
 *
 * The tiny route reuses the key=1 Part1/Part2 synchronization and numerical
 * protocol verbatim; only the launch scale and the cube-inner blocking change:
 *  - blockDim = 8 AIC + 16 AIV (key=1 launches 20 AIC + 40 AIV);
 *  - one outer K round per core (tiling sets cvLoopKSize == multCoreSplitKSize),
 *    so the single L1 A/B slot holds the whole slice;
 *  - inner MMAD block K_TINY_L0 = 256 (key=1 uses 32), cutting the per-core
 *    scalar/issue loop of the A2/AB matmuls;
 *  - Part2 is the key=1 original flow with the needGrad-only branches removed
 *    (the route only fires for needGrad=false); premix follows the upstream
 *    hin-fold consumption line by line.
 *
 * Workspace layout, reduction order and epilogue math are identical to key=1
 * for the same tiling field values; the only numerical deviation comes from the
 * different K-split length (kernel-count driven reduction order), reported as a
 * ULP-level difference during validation.
 */

#ifndef MHC_PRE_SINKHORN_TINY_CORE_H
#define MHC_PRE_SINKHORN_TINY_CORE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "mhc_pre_sinkhorn_base.h"
#include "mhc_pre_sinkhorn_cube_compute.h"
#include <cstdint>

namespace MhcPreSinkhorn {
using namespace AscendC;

// Inner MMAD block of the tiny cube path. 32KB per L0A/L0B slot bounds
// align16(rows or hcMix) * K_L0 * 4B <= 32KB, i.e. K_L0 <= 256 for hcMix=24,
// rows<=32; take the upper bound. The host L1/L0 gate checks the same budget.
constexpr uint64_t TINY_K_L0_SIZE = 256;
// L0 slots are 64KB each (L0A/L0B/l0c TBuf); with K_TINY_L0 the double-slot
// rotation peak equals 2 * align16(32) * 256 * 4B = 64KB.
constexpr uint64_t TINY_L0_BUF_BYTES = 64 * 1024;
// Float offsets inside the L0 TBufs, mirroring the key=1 slot arithmetic.
constexpr uint64_t TINY_L0AB_BUF_OFFSET = 32 * 256;
constexpr uint64_t TINY_L0C_BUF_OFFSET = 64 * 256;
constexpr uint64_t TINY_L0C_A2_BUF_OFFSET = 256 * 16;
constexpr uint16_t TINY_UNIT_FLAG_ENABLE = 2;
constexpr uint16_t TINY_UNIT_FLAG_ENABLE_AUTO_CLOSE = 3;

// Single-L1-slot / wide-L0 variant of HcCubeComputeSplitK. The interface and
// event protocol match HcCubeComputeSplitK so the upstream Part1 can drive it
// unchanged; single-round operation is guaranteed by the tiny tiling
// (cvLoopKSize == multCoreSplitKSize).
class MhcPreSinkhornTinyCubeCompute {
public:
    __aicore__ inline MhcPreSinkhornTinyCubeCompute() {}

    __aicore__ inline void Init(const GlobalTensor<float>& xGm, const GlobalTensor<float>& phiGm, TPipe* tpipe,
                                const MhcPreSinkhornTilingData* tilingData)
    {
        (void)xGm;
        phiGm_ = phiGm;
        // Slot sizes derive from the already-gated tiling: A NZ =
        // align16(mL1Size) * cvLoopKSize * 4B, B NZ = align16(hcMix) * cvLoopKSize * 4B.
        uint64_t l1aBytes = CeilAlign(tilingData->mL1Size, BLOCK_CUBE) * tilingData->cvLoopKSize * sizeof(float);
        uint64_t l1bBytes = CeilAlign(tilingData->hcMix, BLOCK_CUBE) * tilingData->cvLoopKSize * sizeof(float);
        TBuf<TPosition::A1> l1aBuffer;
        tpipe->InitBuffer(l1aBuffer, static_cast<uint32_t>(l1aBytes));
        l1a_ = l1aBuffer.Get<float>();

        TBuf<TPosition::B1> l1bBuffer;
        tpipe->InitBuffer(l1bBuffer, static_cast<uint32_t>(l1bBytes));
        l1b_ = l1bBuffer.Get<float>();

        TBuf<TPosition::A2> l0aBuffer;
        tpipe->InitBuffer(l0aBuffer, TINY_L0_BUF_BYTES);
        l0a_ = l0aBuffer.Get<float>();

        TBuf<TPosition::B2> l0bBuffer;
        tpipe->InitBuffer(l0bBuffer, TINY_L0_BUF_BYTES);
        l0b_ = l0bBuffer.Get<float>();

        TBuf<TPosition::CO1> l0cBuffer;
        tpipe->InitBuffer(l0cBuffer, TINY_L0_BUF_BYTES);
        l0c_ = l0cBuffer.Get<float>();

        // Pre-arm the L0 slots (double buffered) and the single L1 A/B slots.
        for (int i = 0; i < L0A_BUF_NUM; i++) {
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + i);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + i);
        }
        SetFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT);
        SetFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT);
    }

    __aicore__ inline void SetHf32Enable(bool enable)
    {
        isHf32_ = enable;
    }

    __aicore__ inline void WaitBL1Mte1ToMte2Flag()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT);
    }

    __aicore__ inline void CopyInB1(uint64_t mGmOffset, uint64_t kGmOffset, uint64_t kL1Size, const MmParams& mmParams)
    {
        (void)mGmOffset;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = mmParams.curNL1;
        nd2nzParams.dValue = kL1Size;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = mmParams.kGmSize;
        nd2nzParams.dstNzC0Stride = CeilAlign(mmParams.curNL1, BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        DataCopy(l1b_, phiGm_[kGmOffset], nd2nzParams);
        SetFlag<HardEvent::MTE2_MTE1>(B_MTE2_MTE1_EVENT);
    }

    __aicore__ inline void SetBL1Mte1ToMte2Flag()
    {
        SetFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT);
    }

    __aicore__ inline void ComputeDecode(const GlobalTensor<float>& xGm, const GlobalTensor<float>& nextXGm,
                                         const GlobalTensor<float>& workspaceGlobalA2,
                                         const GlobalTensor<float>& workspaceGlobalAB, const MmParams& mmParams,
                                         bool a1Preloaded, bool preloadNextTile, uint64_t nextB1Offset,
                                         uint64_t nextB1Size)
    {
        // Single-round by construction: the next-tile prefetch arguments are
        // never live (Part1's preloadNextB1 is false because the whole K slice
        // is covered by one round).
        (void)nextXGm;
        (void)preloadNextTile;
        (void)nextB1Offset;
        (void)nextB1Size;
        // B slice was issued by CopyInB1; wait for its MTE2 completion.
        WaitFlag<HardEvent::MTE2_MTE1>(B_MTE2_MTE1_EVENT);

        uint64_t curKL1Size = mmParams.curKL1 >= mmParams.singleCoreK ? mmParams.singleCoreK : mmParams.curKL1;
        if (!a1Preloaded) {
            WaitFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT);
            CopyInA1(curKL1Size, xGm, l1a_, mmParams);
            SetFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT);
        }
        WaitFlag<HardEvent::MTE2_MTE1>(MM1_MTE2_MTE1_EVENT);
        ComputeL0Pairs(0, curKL1Size, mmParams.isLastK, mmParams);
        SetFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT);
        if (mmParams.isLastK) {
            Fixp(workspaceGlobalA2, workspaceGlobalAB, mmParams, !preloadNextTile);
            l0cLoopIdx_++;
            if (!mmParams.useGmAccumulation) {
                CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG);
            }
        }
    }

    __aicore__ inline void End()
    {
        for (int i = 0; i < L0A_BUF_NUM; i++) {
            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + i);
            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + i);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(X_MTE1_MTE2_EVENT);
        WaitFlag<HardEvent::MTE1_MTE2>(B_MTE1_MTE2_EVENT);
    }

private:
    __aicore__ inline void CopyInA1(uint64_t kL1Size, const GlobalTensor<float>& aGlobal,
                                    const LocalTensor<float>& al1Local, const MmParams& mmParams)
    {
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = mmParams.curML1;
        nd2nzParams.dValue = kL1Size;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = mmParams.xWsKSize;
        nd2nzParams.dstNzC0Stride = CeilAlign(mmParams.curML1, BLOCK_CUBE);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void LoadAToL0A(uint64_t kL1Offset, uint64_t kL0Size, const MmParams& mmParams)
    {
        static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};
        LoadData3DParamsV2<float> loadData3DParams;
        loadData3DParams.l1H = CeilDiv(mmParams.curML1, BLOCK_CUBE);
        loadData3DParams.l1W = BLOCK_CUBE;
        loadData3DParams.channelSize = kL0Size;
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255;
        loadData3DParams.mExtension = CeilAlign(mmParams.curML1, BLOCK_CUBE);
        loadData3DParams.kExtension = kL0Size;
        loadData3DParams.mStartPt = 0;
        loadData3DParams.kStartPt = 0;
        loadData3DParams.strideW = 1;
        loadData3DParams.strideH = 1;
        loadData3DParams.filterW = 1;
        loadData3DParams.filterSizeW = (1 >> 8) & 255;
        loadData3DParams.filterH = 1;
        loadData3DParams.filterSizeH = (1 >> 8) & 255;
        loadData3DParams.dilationFilterW = 1;
        loadData3DParams.dilationFilterH = 1;
        loadData3DParams.enTranspose = 0;
        loadData3DParams.fMatrixCtrl = 0;

        LoadData<float, LOAD3DV2_CONFIG>(
            l0a_[(l0aLoopIdx_ % L0A_BUF_NUM) * TINY_L0AB_BUF_OFFSET],
            l1a_[CeilAlign(mmParams.curML1, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset], loadData3DParams);
    }

    __aicore__ inline void LoadAToL0B(uint64_t kL1Offset, uint64_t kL0Size, const MmParams& mmParams)
    {
        for (uint64_t mL0Offset = 0; mL0Offset < mmParams.curML1; mL0Offset += BLOCK_CUBE) {
            LoadData2DParams l1ToL0bParams;
            l1ToL0bParams.startIndex = 0;
            l1ToL0bParams.repeatTimes = CeilDiv(kL0Size, (uint64_t)BLOCK_CUBE >> 1);
            l1ToL0bParams.srcStride = CeilDiv(mmParams.curML1, (uint64_t)BLOCK_CUBE);
            l1ToL0bParams.dstGap = 0;
            LoadData(l0b_[(l0bLoopIdx_ % L0B_BUF_NUM) * TINY_L0AB_BUF_OFFSET +
                          mL0Offset * CeilAlign(kL0Size, (uint64_t)BLOCK_CUBE >> 1)],
                     l1a_[CeilAlign(mmParams.curML1, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset +
                          mL0Offset * (uint64_t)(BLOCK_CUBE >> 1)],
                     l1ToL0bParams);
        }
    }

    __aicore__ inline void LoadBToL0B(uint64_t kL1Offset, uint64_t kL0Size, const MmParams& mmParams)
    {
        LoadData2DParams l1ToL0bParams;
        l1ToL0bParams.startIndex = 0;
        l1ToL0bParams.repeatTimes =
            CeilDiv(mmParams.curNL1, (uint64_t)BLOCK_CUBE) * CeilDiv(kL0Size, (uint64_t)BLOCK_CUBE >> 1);
        l1ToL0bParams.srcStride = 1;
        l1ToL0bParams.dstGap = 0;
        LoadData(l0b_[(l0bLoopIdx_ % L0B_BUF_NUM) * TINY_L0AB_BUF_OFFSET],
                 l1b_[CeilAlign(mmParams.curNL1, static_cast<uint64_t>(BLOCK_CUBE)) * kL1Offset], l1ToL0bParams);
    }

    __aicore__ inline void MmadA2(uint64_t kGmOffset, uint64_t kL0Size, bool isLastK, const MmParams& mmParams)
    {
        uint64_t mte1MEvent = MTE1_M_EVENT + (l0bLoopIdx_ % L0B_BUF_NUM);
        SetFlag<HardEvent::MTE1_M>(mte1MEvent);
        WaitFlag<HardEvent::MTE1_M>(mte1MEvent);
        if (isHf32_) {
            SetHF32Mode(1);
            SetHF32TransMode(1);
        }
        for (uint64_t mL0Offset = 0; mL0Offset < mmParams.curML1; mL0Offset += BLOCK_CUBE) {
            MmadParams mmadParams;
            mmadParams.m = BLOCK_CUBE;
            mmadParams.n = BLOCK_CUBE;
            mmadParams.k = kL0Size;
            mmadParams.cmatrixInitVal = mmParams.isFirstK && kGmOffset == 0;
            mmadParams.cmatrixSource = false;
            mmadParams.unitFlag = isLastK ? TINY_UNIT_FLAG_ENABLE_AUTO_CLOSE : TINY_UNIT_FLAG_ENABLE;
            Mmad(l0c_[(l0cLoopIdx_ % L0C_BUF_NUM) * TINY_L0C_BUF_OFFSET + mL0Offset * BLOCK_CUBE],
                 l0a_[(l0aLoopIdx_ % L0A_BUF_NUM) * TINY_L0AB_BUF_OFFSET + mL0Offset * CeilAlign(kL0Size, 8)],
                 l0b_[(l0bLoopIdx_ % L0B_BUF_NUM) * TINY_L0AB_BUF_OFFSET + mL0Offset * CeilAlign(kL0Size, 8)],
                 mmadParams);
        }
        SetHF32Mode(0);
    }

    __aicore__ inline void MmadAB(uint64_t kGmOffset, uint64_t kL0Size, bool isLastK, const MmParams& mmParams)
    {
        uint64_t mte1MEvent = MTE1_M_EVENT + (l0bLoopIdx_ % L0B_BUF_NUM);
        SetFlag<HardEvent::MTE1_M>(mte1MEvent);
        WaitFlag<HardEvent::MTE1_M>(mte1MEvent);
        if (isHf32_) {
            SetHF32Mode(1);
            SetHF32TransMode(1);
        }
        MmadParams mmadParams;
        mmadParams.m = CeilAlign(mmParams.curML1, BLOCK_CUBE);
        mmadParams.n = mmParams.curNL1;
        mmadParams.k = kL0Size;
        mmadParams.cmatrixInitVal = mmParams.isFirstK && kGmOffset == 0;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = isLastK ? TINY_UNIT_FLAG_ENABLE_AUTO_CLOSE : TINY_UNIT_FLAG_ENABLE;
        Mmad(l0c_[(l0cLoopIdx_ % L0C_BUF_NUM) * TINY_L0C_BUF_OFFSET + TINY_L0C_A2_BUF_OFFSET],
             l0a_[(l0aLoopIdx_ % L0A_BUF_NUM) * TINY_L0AB_BUF_OFFSET],
             l0b_[(l0bLoopIdx_ % L0B_BUF_NUM) * TINY_L0AB_BUF_OFFSET], mmadParams);
        SetHF32Mode(0);
    }

    __aicore__ inline void ComputeL0Pairs(uint64_t l1BaseKOffset, uint64_t curKL1Size, bool closeAccumulator,
                                          const MmParams& mmParams)
    {
        for (uint64_t kL1Offset = 0; kL1Offset < curKL1Size; kL1Offset += TINY_K_L0_SIZE * 2) {
            uint64_t firstKOffset = l1BaseKOffset + kL1Offset;
            bool firstIsLast = closeAccumulator && kL1Offset + TINY_K_L0_SIZE >= curKL1Size;

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
            LoadAToL0A(kL1Offset, TINY_K_L0_SIZE, mmParams);

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            LoadAToL0B(kL1Offset, TINY_K_L0_SIZE, mmParams);
            MmadA2(firstKOffset, TINY_K_L0_SIZE, firstIsLast, mmParams);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            l0bLoopIdx_++;

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            LoadBToL0B(firstKOffset % mmParams.xWsKSize, TINY_K_L0_SIZE, mmParams);
            MmadAB(firstKOffset, TINY_K_L0_SIZE, firstIsLast, mmParams);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            l0bLoopIdx_++;
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
            l0aLoopIdx_++;

            uint64_t secondKL1Offset = kL1Offset + TINY_K_L0_SIZE;
            if (secondKL1Offset >= curKL1Size) {
                continue;
            }

            uint64_t secondKOffset = l1BaseKOffset + secondKL1Offset;
            bool secondIsLast = closeAccumulator && secondKL1Offset + TINY_K_L0_SIZE >= curKL1Size;
            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
            LoadAToL0A(secondKL1Offset, TINY_K_L0_SIZE, mmParams);

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            LoadAToL0B(secondKL1Offset, TINY_K_L0_SIZE, mmParams);
            MmadA2(secondKOffset, TINY_K_L0_SIZE, secondIsLast, mmParams);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            l0bLoopIdx_++;

            WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            LoadBToL0B(secondKOffset % mmParams.xWsKSize, TINY_K_L0_SIZE, mmParams);
            MmadAB(secondKOffset, TINY_K_L0_SIZE, secondIsLast, mmParams);
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0B + l0bLoopIdx_ % L0B_BUF_NUM);
            l0bLoopIdx_++;
            SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_L0A + l0aLoopIdx_ % L0A_BUF_NUM);
            l0aLoopIdx_++;
        }
    }

    __aicore__ inline void CopyOut(const GlobalTensor<float>& workspaceGlobal, const LocalTensor<float>& c1Local,
                                   uint64_t baseM, uint64_t baseN, bool enableNz2Nd, uint64_t N)
    {
        DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.nz2ndEn = enableNz2Nd;
        if (enableNz2Nd) {
            intriParams.dstStride = N;
            intriParams.srcStride = CeilAlign(baseM, BLOCK_CUBE);
            SetFixpipeNz2ndFlag(1, 1, 1);
        } else {
            intriParams.dstStride = CeilAlign(intriParams.nSize, BLOCK_CUBE);
            intriParams.srcStride = CeilAlign(baseM, BLOCK_CUBE);
        }
        intriParams.unitFlag = TINY_UNIT_FLAG_ENABLE_AUTO_CLOSE;
        DataCopy(workspaceGlobal, c1Local, intriParams);
    }

    __aicore__ inline void Fixp(const GlobalTensor<float>& workspaceGlobalA2,
                                const GlobalTensor<float>& workspaceGlobalAB, const MmParams& mmParams,
                                bool finalSegment)
    {
        if (mmParams.accumulateGm && !atomicEnabled_) {
            SetAtomicAdd<float>();
            atomicEnabled_ = true;
        }
        CopyOut(workspaceGlobalA2, l0c_[(l0cLoopIdx_ % L0C_BUF_NUM) * TINY_L0C_BUF_OFFSET], mmParams.curML1, BLOCK_CUBE,
                false, BLOCK_CUBE);
        CopyOut(workspaceGlobalAB, l0c_[(l0cLoopIdx_ % L0C_BUF_NUM) * TINY_L0C_BUF_OFFSET + TINY_L0C_A2_BUF_OFFSET],
                mmParams.curML1, mmParams.curNL1, true, mmParams.nOutSize);
        if (mmParams.useGmAccumulation && (!mmParams.accumulateGm || finalSegment)) {
            SetFlag<HardEvent::FIX_S>(0);
            WaitFlag<HardEvent::FIX_S>(0);
            if (finalSegment) {
                DisableDmaAtomic();
                atomicEnabled_ = false;
            }
        }
    }

    GlobalTensor<float> phiGm_;

    LocalTensor<float> l1a_;
    LocalTensor<float> l1b_;
    LocalTensor<float> l0a_;
    LocalTensor<float> l0b_;
    LocalTensor<float> l0c_;

    static constexpr uint64_t MM1_MTE2_MTE1_EVENT = 2;
    static constexpr uint64_t X_MTE1_MTE2_EVENT = 2;
    static constexpr uint64_t B_MTE2_MTE1_EVENT = 4;
    static constexpr uint64_t B_MTE1_MTE2_EVENT = 4;
    static constexpr uint64_t M_MTE1_EVENT_L0A = 3;
    static constexpr uint64_t M_MTE1_EVENT_L0B = 5;
    static constexpr uint64_t MTE1_M_EVENT = 2;
    static constexpr uint64_t SYNC_AIC_TO_AIV_FLAG = 9;

    static constexpr uint64_t L0A_BUF_NUM = 2;
    static constexpr uint64_t L0B_BUF_NUM = 2;
    static constexpr uint64_t L0C_BUF_NUM = 1;

    uint64_t l0aLoopIdx_ = 0;
    uint64_t l0bLoopIdx_ = 0;
    uint64_t l0cLoopIdx_ = 0;
    bool isHf32_ = false;
    bool atomicEnabled_ = false;
};

// Part2 for the tiny route: the key=1 ProcessOriginal flow with every
// needGrad-only branch removed. Merges (squareSum reduction / diagonal gather /
// mmOut cross-core reduction) and the epilogue math are line-by-line identical;
// premix follows the upstream hin-fold consumption (premix replaces the
// internal hPre source and is also written to hPreGm when present).
template <typename T>
class MhcPreSinkhornTinyPart2 {
public:
    __aicore__ inline MhcPreSinkhornTinyPart2() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR y, GM_ADDR post, GM_ADDR combFrag,
                                GM_ADDR premix, GM_ADDR hPre, GM_ADDR hcBeforeNorm, GM_ADDR invRms, GM_ADDR sumOut,
                                GM_ADDR normOut, GM_ADDR workspace, const MhcPreSinkhornTilingData* tilingDataPtr,
                                TPipe* pipePtr)
    {
        (void)hcBeforeNorm;
        (void)invRms;
        (void)sumOut;
        (void)normOut;
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float*)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float*)hcBase);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        postGm.SetGlobalBuffer((__gm__ float*)post);
        combFragGm.SetGlobalBuffer((__gm__ float*)combFrag);
        premixGm.SetGlobalBuffer((__gm__ float*)premix);
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);
        if (tilingData->hasPremix != 0) {
            hPreGm.SetGlobalBuffer((__gm__ float*)hPre);
        }
        if ASCEND_IS_AIC {
            return;
        }
        InitOriginalBuffers(tilingData->cubeBlockDimK);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            return;
        }
        ProcessOriginal();
    }

private:
    __aicore__ inline void InitCommonBuffers(int64_t stage1UsedCoreNum, int64_t rowFactor)
    {
        pipe->InitBuffer(squareSumQue, NUM_TWO, stage1UsedCoreNum * rowFactor * SQUARE_SUM_SIZE * sizeof(float));
        pipe->InitBuffer(squareReduceBuf, rowFactor * SQUARE_SUM_SIZE * sizeof(float));
        pipe->InitBuffer(rsqrtBuf, RoundUp<float>(rowFactor) * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(rowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(
            maskPatternBuf,
            RoundUp<uint32_t>((rowFactor + MASK_PATTERN_DIM_SIZE - 1) * MASK_PATTERN_REPEAT_SIZE) * sizeof(uint32_t));

        squareReduceLocal = squareReduceBuf.Get<float>();
        rsqrtLocal = rsqrtBuf.Get<float>();
        rowBrcbLocal0 = rowBrcbBuf0.Get<float>();
        maskPatternLocal = maskPatternBuf.Get<uint32_t>();
        SetGatherMaskPattern(maskPatternLocal, static_cast<uint16_t>(rowFactor + MASK_PATTERN_DIM_SIZE - 1));
    }

    __aicore__ inline void InitOriginalBuffers(int64_t stage1UsedCoreNum)
    {
        int64_t rowFactor = tilingData->stage2RowFactor;
        int64_t xQueSize = rowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor);
        InitCommonBuffers(stage1UsedCoreNum, rowFactor);

        pipe->InitBuffer(mixesQue01, NUM_TWO,
                         stage1UsedCoreNum * rowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(mixesQue2, NUM_TWO,
                         stage1UsedCoreNum * rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(xQue, NUM_TWO, xQueSize * sizeof(T));
        pipe->InitBuffer(yQue, NUM_TWO, rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, NUM_TWO, rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(combFragQue, NUM_TWO,
                         rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(rowBrcbBuf0, RoundUp<float>(rowFactor) * BLOCK_SIZE);
        pipe->InitBuffer(hcBrcbBuf1, RoundUp<float>(rowFactor * tilingData->hcMultAlign) * BLOCK_SIZE);
        pipe->InitBuffer(reduceBuf, rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mixes01ReduceBuf, rowFactor * tilingData->hcMultAlign * NUM_TWO * sizeof(float));
        pipe->InitBuffer(premixBuf, rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mixes02ReduceBuf, rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(xCastBuf, xQueSize * sizeof(float));
        pipe->InitBuffer(yCastBuf, rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        hcBrcbLocal1 = hcBrcbBuf1.Get<float>();
        reduceLocal = reduceBuf.Get<float>();
        mixes01ReduceLocal = mixes01ReduceBuf.Get<float>();
        premixLocal = premixBuf.Get<float>();
        mixes02ReduceLocal = mixes02ReduceBuf.Get<float>();
        xCastLocal = xCastBuf.Get<float>();
        yCastLocal = yCastBuf.Get<float>();
    }

    __aicore__ inline void CalcRsqrt(int64_t stage1UsedCoreNum, int64_t globalRowOffset, int64_t curRowFactor,
                                     int64_t workspaceSize1, int64_t workspaceSize2)
    {
        squareSumOutLocal = squareSumQue.AllocTensor<float>();
        CopyIn(workspaceGm[workspaceSize1 + workspaceSize2 + globalRowOffset * SQUARE_SUM_SIZE], squareSumOutLocal,
               stage1UsedCoreNum, curRowFactor * SQUARE_SUM_SIZE,
               CeilAlign(tilingData->bs, SQUARE_SUM_SIZE) * SQUARE_SUM_SIZE - curRowFactor * SQUARE_SUM_SIZE);
        squareSumQue.EnQue(squareSumOutLocal);
        squareSumOutLocal = squareSumQue.DeQue<float>();
        ReduceSumARAPerf(squareReduceLocal, squareSumOutLocal, 1, stage1UsedCoreNum, curRowFactor * SQUARE_SUM_SIZE);
        GatherMaskByDiagonal(rsqrtLocal, squareReduceLocal, maskPatternLocal, static_cast<uint16_t>(curRowFactor),
                             static_cast<uint16_t>(globalRowOffset % SQUARE_SUM_SIZE));
        float coeff = 1.0f / static_cast<float>(tilingData->k);
        Muls(rsqrtLocal, rsqrtLocal, coeff, curRowFactor);
        PipeBarrier<PIPE_V>();
        Adds(rsqrtLocal, rsqrtLocal, tilingData->normEps, curRowFactor);
        PipeBarrier<PIPE_V>();
        Sqrt(rsqrtLocal, rsqrtLocal, curRowFactor);
        Duplicate(rowBrcbLocal0, static_cast<float>(1.0f), curRowFactor);
        PipeBarrier<PIPE_V>();
        Div(rsqrtLocal, rowBrcbLocal0, rsqrtLocal, curRowFactor);
    }

    __aicore__ inline void ProcessOriginal()
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= tilingData->secondUsedCoreNum) {
            return;
        }

        int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
        int64_t mmLastAxisSize = CeilAlign(tilingData->hcMix, MM_CACHE_LINE_BYTES / sizeof(float));
        int64_t xCastFp32BufSize =
            tilingData->mL1Size * CeilAlign(tilingData->cvLoopKSize, MM_CACHE_LINE_BYTES / sizeof(float));
        int64_t workspaceSize1 = tilingData->cubeCoreNum * DOUBLE_BUFFER * xCastFp32BufSize;
        int64_t workspaceSize2 =
            CeilAlign(stage1UsedCoreNum * tilingData->bs * mmLastAxisSize * sizeof(float), WORKSPACE_ALIGN_SIZE) /
            sizeof(float);

        CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
        CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
        CopyIn(hcBaseGm[tilingData->hcMult * NUM_TWO], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        float hcScale0 = hcScaleGm.GetValue(0);
        float hcScale1 = hcScaleGm.GetValue(1);
        float hcScale2 = hcScaleGm.GetValue(NUM_TWO);

        int64_t rowOuterLoop = blockIdx == tilingData->secondUsedCoreNum - 1 ? tilingData->rowLoopOfTailBlock :
                                                                               tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = blockIdx == tilingData->secondUsedCoreNum - 1 ? tilingData->tailRowFactorOfTailBlock :
                                                                                tilingData->tailRowFactorOfFormerBlock;

        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = rowOuterIdx == rowOuterLoop - 1 ? tailRowFactor : tilingData->stage2RowFactor;
            int64_t globalRowOffset =
                blockIdx * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->stage2RowFactor;
            CalcRsqrt(stage1UsedCoreNum, globalRowOffset, curRowFactor, workspaceSize1, workspaceSize2);

            mixes01Local = mixesQue01.AllocTensor<float>();
            uint64_t mixBaseOffset = workspaceSize1 + globalRowOffset * mmLastAxisSize;
            CopyInWithOuterFor(workspaceGm[mixBaseOffset], mixes01Local, stage1UsedCoreNum, curRowFactor,
                               tilingData->hcMult, tilingData->bs, mmLastAxisSize);
            CopyInWithOuterFor(workspaceGm[mixBaseOffset + tilingData->hcMult],
                               mixes01Local[stage1UsedCoreNum * curRowFactor * tilingData->hcMultAlign],
                               stage1UsedCoreNum, curRowFactor, tilingData->hcMult, tilingData->bs, mmLastAxisSize);
            mixesQue01.EnQue(mixes01Local);
            mixes01Local = mixesQue01.DeQue<float>();
            ReduceSumARAPerf(mixes01ReduceLocal, mixes01Local, NUM_TWO, stage1UsedCoreNum,
                             curRowFactor * tilingData->hcMultAlign);

            ProcessPre(mixes01ReduceLocal, mixes01ReduceLocal, hcBase0Local, rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1,
                       hcScale0, tilingData->hcEps, curRowFactor, tilingData->hcMult);

            if (tilingData->hasPremix != 0) {
                // premixLocal 跨 rowLoop 复用：上一轮 ProcessY 仍在读，先等 V 流水读完再启动 MTE2 覆写。
                event_t eventIdPremixV_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(eventIdPremixV_MTE2);
                WaitFlag<HardEvent::V_MTE2>(eventIdPremixV_MTE2);
                CopyIn(premixGm[globalRowOffset * tilingData->hcMult], premixLocal, 1,
                       curRowFactor * tilingData->hcMult);
                event_t eventIdPremix = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdPremix);
                WaitFlag<HardEvent::MTE2_V>(eventIdPremix);
            }

            if (tilingData->hasPremix != 0) {
                VToMTE3Sync();
                CopyOut(mixes01ReduceLocal, hPreGm[globalRowOffset * tilingData->hcMult], curRowFactor,
                        tilingData->hcMult);
                MTE3ToVSync();
            }
            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor = dLoopIdx == tilingData->dLoop - 1 ? tilingData->tailDFactor : tilingData->dFactor;
                xLocal = xQue.template AllocTensor<T>();
                CopyIn(xGm[globalRowOffset * tilingData->hcMult * tilingData->d + dLoopIdx * tilingData->dFactor],
                       xLocal, curRowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                xQue.template EnQue(xLocal);
                xLocal = xQue.template DeQue<T>();
                yLocal = yQue.template AllocTensor<T>();

                ProcessY(yLocal, xLocal, tilingData->hasPremix != 0 ? premixLocal : mixes01ReduceLocal, hcBrcbLocal1,
                         xCastLocal, yCastLocal, curRowFactor, tilingData->hcMult, curDFactor);
                xQue.template FreeTensor(xLocal);
                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                CopyOut(yLocal, yGm[globalRowOffset * tilingData->d + dLoopIdx * tilingData->dFactor], curRowFactor,
                        curDFactor, tilingData->d - curDFactor);
                yQue.template FreeTensor(yLocal);
            }
            postLocal = postQue.template AllocTensor<float>();
            ProcessPost(postLocal, mixes01ReduceLocal[curRowFactor * tilingData->hcMultAlign], hcBase1Local, rsqrtLocal,
                        rowBrcbLocal0, hcBrcbLocal1, hcScale1, curRowFactor, tilingData->hcMult);
            mixesQue01.template FreeTensor(mixes01Local);
            postQue.EnQue(postLocal);
            postLocal = postQue.template DeQue<float>();
            CopyOut(postLocal, postGm[globalRowOffset * tilingData->hcMult], curRowFactor, tilingData->hcMult);
            postQue.FreeTensor(postLocal);

            mixes2Local = mixesQue2.AllocTensor<float>();
            for (int64_t kBlockIdx = 0; kBlockIdx < stage1UsedCoreNum; kBlockIdx++) {
                for (int64_t rowIdx = 0; rowIdx < curRowFactor; rowIdx++) {
                    int64_t gmOffset = workspaceSize1 + kBlockIdx * tilingData->bs * mmLastAxisSize +
                                       (globalRowOffset + rowIdx) * mmLastAxisSize + tilingData->hcMult * NUM_TWO;
                    int64_t ubOffset =
                        (kBlockIdx * curRowFactor + rowIdx) * tilingData->hcMult * tilingData->hcMultAlign;
                    CopyIn(workspaceGm[gmOffset], mixes2Local[ubOffset], tilingData->hcMult, tilingData->hcMult);
                }
            }
            mixesQue2.EnQue(mixes2Local);
            mixes2Local = mixesQue2.DeQue<float>();
            ReduceSumARAPerf(mixes02ReduceLocal, mixes2Local, 1, stage1UsedCoreNum,
                             curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);

            combFragLocal = combFragQue.AllocTensor<float>();

            MulABLastDimBrcInline<float, false>(mixes02ReduceLocal, mixes02ReduceLocal, rsqrtLocal, rowBrcbLocal0,
                                                curRowFactor, tilingData->hcMult * tilingData->hcMultAlign);
            Muls(mixes02ReduceLocal, mixes02ReduceLocal, hcScale2,
                 curRowFactor * tilingData->hcMult * tilingData->hcMultAlign);
            PipeBarrier<PIPE_V>();
            AddBAFirstDimBrcInline<float>(mixes02ReduceLocal, mixes02ReduceLocal, hcBase2Local, curRowFactor,
                                          tilingData->hcMult * tilingData->hcMultAlign);
            SoftmaxFP32Perf(mixes02ReduceLocal, mixes02ReduceLocal, reduceLocal, hcBrcbLocal1,
                            curRowFactor * tilingData->hcMult, tilingData->hcMult, tilingData->hcEps);

            ReduceSumARAPerf(reduceLocal, mixes02ReduceLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
            Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMultAlign);
            PipeBarrier<PIPE_V>();
            DivABABrcInline(combFragLocal, mixes02ReduceLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                            tilingData->hcMult);

            for (int64_t iter = 1; iter < tilingData->iterTimes; iter++) {
                LastDimReduceSumPerf(reduceLocal, combFragLocal, curRowFactor * tilingData->hcMult, tilingData->hcMult);
                Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMult);

                PipeBarrier<PIPE_V>();
                DivABLastDimBrcInline<float, true>(combFragLocal, combFragLocal, reduceLocal, hcBrcbLocal1,
                                                   curRowFactor * tilingData->hcMult, tilingData->hcMult);

                ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor, tilingData->hcMult, tilingData->hcMult);
                Adds(reduceLocal, reduceLocal, tilingData->hcEps, curRowFactor * tilingData->hcMultAlign);

                PipeBarrier<PIPE_V>();
                DivABABrcInline(combFragLocal, combFragLocal, reduceLocal, curRowFactor, tilingData->hcMult,
                                tilingData->hcMult);
            }
            mixesQue2.FreeTensor(mixes2Local);
            squareSumQue.template FreeTensor(squareSumOutLocal);

            combFragQue.EnQue(combFragLocal);
            combFragLocal = combFragQue.DeQue<float>();
            CopyOut(combFragLocal, combFragGm[globalRowOffset * tilingData->hcMult * tilingData->hcMult],
                    curRowFactor * tilingData->hcMult, tilingData->hcMult);
            combFragQue.FreeTensor(combFragLocal);
        }
    }

    TPipe* pipe;
    const MhcPreSinkhornTilingData* tilingData;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;
    GlobalTensor<float> premixGm;
    GlobalTensor<float> hPreGm;

    TQue<QuePosition::VECIN, 1> mixesQue01;
    TQue<QuePosition::VECIN, 1> mixesQue2;
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;
    TQue<QuePosition::VECIN, 1> squareSumQue;

    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;
    TBuf<QuePosition::VECCALC> rowBrcbBuf0;
    TBuf<QuePosition::VECCALC> hcBrcbBuf1;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TBuf<QuePosition::VECCALC> rsqrtBuf;
    TBuf<QuePosition::VECCALC> squareReduceBuf;
    TBuf<QuePosition::VECCALC> mixes01ReduceBuf;
    TBuf<QuePosition::VECCALC> premixBuf;
    TBuf<QuePosition::VECCALC> mixes02ReduceBuf;
    TBuf<QuePosition::VECCALC> xCastBuf;
    TBuf<QuePosition::VECCALC> yCastBuf;
    TBuf<QuePosition::VECCALC> maskPatternBuf;

    LocalTensor<float> mixes01Local;
    LocalTensor<float> mixes2Local;
    LocalTensor<float> rsqrtLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
    LocalTensor<float> rowBrcbLocal0;
    LocalTensor<float> hcBrcbLocal1;
    LocalTensor<float> reduceLocal;
    LocalTensor<float> squareReduceLocal;
    LocalTensor<float> mixes01ReduceLocal;
    LocalTensor<float> premixLocal;
    LocalTensor<float> mixes02ReduceLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> yCastLocal;
    LocalTensor<float> squareSumOutLocal;
    LocalTensor<uint32_t> maskPatternLocal;
};

} // namespace MhcPreSinkhorn

#endif // MHC_PRE_SINKHORN_TINY_CORE_H
