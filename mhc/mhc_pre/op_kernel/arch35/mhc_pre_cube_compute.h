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
 * \file mhc_pre_cube_compute.h
 * \brief Basic API Cube service for MHC Pre matmul
 */

#ifndef MHC_PRE_CUBE_COMPUTE_H
#define MHC_PRE_CUBE_COMPUTE_H

#include "mhc_pre_common.h"

namespace MhcPre {

// Mmad and Fixpipe unit-flag values.
constexpr uint32_t FINAL_ACCUMULATION = 3U;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2U;

// Fixpipe row-split policy.
constexpr bool MHC_PRE_BASIC_API_SPLIT_M = true;
constexpr uint64_t MHC_PRE_BASIC_API_SPLIT_M_ALIGN = 2U;

class MhcPreCubeCompute {
public:
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t k_{0};
    uint64_t baseM_{0};
    uint64_t baseN_{0};
    uint64_t baseK_{0};
    uint64_t kL1_{0};
    uint32_t implMode_{MHC_PRE_IMPL_MODE_FP32};
    uint8_t bL1BufferID_{0};
    uint8_t l0PingPongID_{0};
    uint8_t cl0PingPongID_{0};

    AscendC::LocalTensor<float> aL0Ping_;
    AscendC::LocalTensor<float> aL0Pong_;
    AscendC::LocalTensor<float> bL0Ping_;
    AscendC::LocalTensor<float> bL0Pong_;
    AscendC::LocalTensor<float> cL0Ping_;
    AscendC::LocalTensor<float> cL0Pong_;

    __aicore__ inline MhcPreCubeCompute() {}

    __aicore__ inline uint8_t GetBL1BufferId()
    {
        return bL1BufferID_;
    }

    __aicore__ inline void CopyInA1Nd2Nz(uint64_t m, uint64_t currentK, const AscendC::GlobalTensor<float> &aGlobal,
                                         const AscendC::LocalTensor<float> &al1Local)
    {
        CopyInA1Nd2Nz(m, currentK, currentK, aGlobal, al1Local);
    }

    __aicore__ inline void CopyInA1Nd2Nz(uint64_t m, uint64_t currentK, uint64_t srcDValue,
                                         const AscendC::GlobalTensor<float> &aGlobal,
                                         const AscendC::LocalTensor<float> &al1Local)
    {
        AscendC::Nd2NzParams nd2nzParam;
        nd2nzParam.ndNum = 1;
        nd2nzParam.nValue = m;
        nd2nzParam.dValue = currentK;
        nd2nzParam.srcNdMatrixStride = 1;
        nd2nzParam.srcDValue = srcDValue;
        nd2nzParam.dstNzC0Stride = BasicApiAlign(m, AscendC::BLOCK_CUBE);
        nd2nzParam.dstNzNStride = 1;
        nd2nzParam.dstNzMatrixStride = 1;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParam);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2);
    }

    __aicore__ inline void Init(uint32_t implMode)
    {
        implMode_ = implMode;
        uint32_t aL0OneBuffer = 256 * 32;
        uint32_t bL0OneBuffer = 256 * 32;
        uint32_t cL0OneBuffer = 256 * 128;

        aL0Ping_ = AscendC::LocalTensor<float>(AscendC::TPosition::A2, 0, aL0OneBuffer);
        aL0Pong_ = AscendC::LocalTensor<float>(AscendC::TPosition::A2, aL0OneBuffer * sizeof(float), aL0OneBuffer);
        bL0Ping_ = AscendC::LocalTensor<float>(AscendC::TPosition::B2, 0, bL0OneBuffer);
        bL0Pong_ = AscendC::LocalTensor<float>(AscendC::TPosition::B2, bL0OneBuffer * sizeof(float), bL0OneBuffer);
        cL0Ping_ = AscendC::LocalTensor<float>(AscendC::TPosition::CO1, 0, cL0OneBuffer);
        cL0Pong_ = AscendC::LocalTensor<float>(AscendC::TPosition::CO1, cL0OneBuffer * sizeof(float), cL0OneBuffer);

        // Ping-pong L0A/L0B/L0C so MTE1, Mmad, and Fixpipe can overlap.
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);

        // HF32 is a Cube mode. Restrict the mode write to AIC to avoid an invalid AIV-side operation.
        if ASCEND_IS_AIC {
            if (implMode == MHC_PRE_IMPL_MODE_HF32) {
                AscendC::SetHF32Mode(1);
                AscendC::SetHF32TransMode(1);
            }
        }
    }

    __aicore__ inline void CopyInB1Nd2Nz(uint64_t k, uint64_t currentK, uint64_t baseN,
                                         const AscendC::GlobalTensor<float> &bGlobal,
                                         const AscendC::LocalTensor<float> &bl1Local)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_);
        k_ = k;
        kL1_ = currentK;
        baseN_ = baseN;
        AscendC::Nd2NzParams nd2nzParam;
        nd2nzParam.ndNum = 1;
        nd2nzParam.nValue = baseN_;
        nd2nzParam.dValue = kL1_;
        nd2nzParam.srcNdMatrixStride = 1;
        nd2nzParam.srcDValue = k_;
        nd2nzParam.dstNzC0Stride = BasicApiAlign(baseN_, AscendC::BLOCK_CUBE);
        nd2nzParam.dstNzNStride = 1;
        nd2nzParam.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParam);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufferID_);
    }

    __aicore__ inline void Process(uint64_t m, uint64_t n, uint64_t baseM, uint64_t baseK, bool isFirstKL1,
                                   bool isLastKL1, const AscendC::LocalTensor<float> &al1Local,
                                   const AscendC::LocalTensor<float> &bl1Local)
    {
        m_ = m;
        n_ = n;
        baseM_ = baseM;
        baseK_ = baseK;
        if (isFirstKL1) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cl0PingPongID_);
        }
        uint64_t kL1Offset = 0;
        for (uint64_t kb = 0; kb < kL1_; kb += baseK_) {
            bool isLastKL0 = (kb + baseK_) >= kL1_;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + 3);
            CopyInA2(kb, kL1Offset, al1Local);
            CopyInB2(kb, kL1Offset, bl1Local);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPongID_);
            MmadBase(kb, isFirstKL1, isLastKL1, isLastKL0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPongID_ + 3);
            l0PingPongID_ ^= 1;
            kL1Offset += baseK_;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufferID_);
        bL1BufferID_ ^= 1;
    }

    // Own iter0 (K-L1 staging and cross-core handshakes); Process owns iter1 (K-L0 and Mmad).
    __aicore__ inline void ProcessKRange(uint64_t m, uint64_t n, uint64_t k, uint64_t baseM, uint64_t baseK,
                                         uint64_t kStart, uint64_t kEnd, uint64_t kL1Size, bool useSequentialPartials,
                                         uint64_t partialK, bool useGmStage, uint64_t stageSlotElements,
                                         uint64_t stageCoreOffset, const AscendC::GlobalTensor<float> &phiGlobal,
                                         const AscendC::GlobalTensor<float> &xStageGlobal,
                                         const AscendC::GlobalTensor<float> &dstGlobal, uint64_t dstGroupStride,
                                         const AscendC::LocalTensor<float> &al1Local,
                                         const AscendC::LocalTensor<float> &bl1Local)
    {
        uint8_t aL1BufferId = 0;
        for (uint64_t kOffset = kStart; kOffset < kEnd; kOffset += kL1Size) {
            uint64_t currentK = AscendC::Std::min(kL1Size, kEnd - kOffset);
            uint8_t bL1BufferId = GetBL1BufferId();
            AscendC::LocalTensor<float> currentAL1 = al1Local[aL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
            AscendC::LocalTensor<float> currentBL1 = bl1Local[bL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
            CopyInB1Nd2Nz(k, currentK, n, phiGlobal[kOffset], currentBL1);

            if (useGmStage) {
                AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_READY_FLAG);
                AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_READY_FLAG +
                                                                                    MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                uint64_t stageOffset = stageCoreOffset + aL1BufferId * stageSlotElements;
                CopyInA1Nd2Nz(m, currentK, kL1Size, xStageGlobal[stageOffset], currentAL1);
            } else {
                AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG);
                AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG +
                                                                                    MHC_PRE_SUBBLOCK_FLAG_OFFSET);
            }

            bool isFirstKL1 = useSequentialPartials ? kOffset % partialK == 0U : kOffset == kStart;
            bool isLastKL1 = useSequentialPartials ?
                                 ((kOffset + currentK) % partialK == 0U || kOffset + currentK >= kEnd) :
                                 kOffset + currentK >= kEnd;
            Process(m, n, baseM, baseK, isFirstKL1, isLastKL1, currentAL1, currentBL1);
            if (isLastKL1) {
                uint64_t partialIndex = useSequentialPartials ? (kOffset + currentK - 1U) / partialK : 0U;
                CopyOut(dstGlobal[partialIndex * dstGroupStride], n);
            }
            AscendC::CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
            AscendC::CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG +
                                                                               MHC_PRE_SUBBLOCK_FLAG_OFFSET);
            aL1BufferId ^= 1U;
        }
    }
    __aicore__ inline void CopyInA2(uint64_t kOffset, uint64_t kAL1Offset, const AscendC::LocalTensor<float> &al1Local)
    {
        uint64_t mAL1 = BasicApiAlign(baseM_, AscendC::BLOCK_CUBE);
        uint64_t offsetAL1 = BasicApiAlign(kAL1Offset, MHC_PRE_BASIC_API_C0_SIZE) * mAL1;
        AscendC::LoadData2DParamsV2 loadData2dParams;
        uint64_t currM = baseM_;
        uint64_t currK = AscendC::Std::min(baseK_, kL1_ - kOffset);
        loadData2dParams.mStartPosition = 0;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.mStep = BasicApiCeilDiv(currM, AscendC::BLOCK_CUBE);
        loadData2dParams.kStep = BasicApiCeilDiv(currK, MHC_PRE_BASIC_API_C0_SIZE);
        loadData2dParams.srcStride = BasicApiCeilDiv(currM, AscendC::BLOCK_CUBE);
        loadData2dParams.dstStride = loadData2dParams.mStep;
        loadData2dParams.ifTranspose = false;
        AscendC::LoadData(l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_, al1Local[offsetAL1], loadData2dParams);
    }

    __aicore__ inline void CopyInB2(uint64_t kOffset, uint64_t kBL1Offset, const AscendC::LocalTensor<float> &bl1Local)
    {
        uint64_t nBL1 = BasicApiAlign(baseN_, AscendC::BLOCK_CUBE);
        uint64_t offsetBL1 = BasicApiAlign(kBL1Offset, MHC_PRE_BASIC_API_C0_SIZE) * nBL1;
        AscendC::LoadData2DParamsV2 loadData2dParams;
        uint64_t currN = baseN_;
        uint64_t currK = AscendC::Std::min(baseK_, kL1_ - kOffset);
        loadData2dParams.mStartPosition = 0;
        loadData2dParams.kStartPosition = 0;
        loadData2dParams.mStep = BasicApiCeilDiv(currN, AscendC::BLOCK_CUBE);
        loadData2dParams.kStep = BasicApiCeilDiv(currK, MHC_PRE_BASIC_API_C0_SIZE);
        loadData2dParams.srcStride = BasicApiCeilDiv(currN, AscendC::BLOCK_CUBE);
        loadData2dParams.dstStride = loadData2dParams.mStep;
        loadData2dParams.ifTranspose = false;
        AscendC::LoadData(l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_, bl1Local[offsetBL1], loadData2dParams);
    }

    __aicore__ inline void MmadBase(uint64_t kOffset, bool isFirstKL1, bool isLastKL1, bool isLastKL0)
    {
        uint32_t mmK = AscendC::Std::min(baseK_, kL1_ - kOffset);
        // FP32 keeps K=32 accumulation order for precision; HF32 consumes the full baseK to reduce loop overhead.
        uint32_t mmKStep = implMode_ == MHC_PRE_IMPL_MODE_FP32 ? 32U : mmK;
        uint64_t mAlign = BasicApiAlign(baseM_, AscendC::BLOCK_CUBE);
        uint64_t nAlign = BasicApiAlign(baseN_, AscendC::BLOCK_CUBE);
        for (uint32_t innerK = 0; innerK < mmK; innerK += mmKStep) {
            uint32_t currentK = AscendC::Std::min(mmKStep, mmK - innerK);
            bool isLastInnerK = innerK + currentK >= mmK;
            AscendC::MmadParams params;
            params.m = baseM_;
            params.n = baseN_;
            params.k = currentK;
            params.disableGemv = true;
            params.cmatrixInitVal = (isFirstKL1 && kOffset == 0 && innerK == 0);
            params.cmatrixSource = false;
            params.unitFlag = (isLastKL1 && isLastKL0 && isLastInnerK) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            AscendC::Mmad(cl0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_,
                          (l0PingPongID_ == 0 ? aL0Ping_ : aL0Pong_)[innerK * mAlign],
                          (l0PingPongID_ == 0 ? bL0Ping_ : bL0Pong_)[innerK * nAlign], params);
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<float> &dstLocal)
    {
        // Direct L0C-to-UB removes the hMix GM write/read round trip when the caller has aligned UB space.
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cl0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(cl0PingPongID_);
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        uint64_t c0 = AscendC::AuxGetC0Size<float>();
        fixpipeParams.nSize = BasicApiAlign(baseN_, c0);
        fixpipeParams.mSize =
            MHC_PRE_BASIC_API_SPLIT_M ? BasicApiAlign(baseM_, MHC_PRE_BASIC_API_SPLIT_M_ALIGN) : baseM_;
        fixpipeParams.dstStride = fixpipeParams.nSize;
        fixpipeParams.srcStride = BasicApiAlign(baseM_, AscendC::BLOCK_CUBE);
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
        fixpipeParams.dualDstCtl =
            MHC_PRE_BASIC_API_SPLIT_M ? static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_M) : 0;
        fixpipeParams.unitFlag = FINAL_ACCUMULATION;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 1;
        fixpipeParams.params.dstNdStride = 1;
        AscendC::Fixpipe<float, float, AscendC::Impl::CFG_ROW_MAJOR_UB>(
            dstLocal, cl0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, fixpipeParams);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cl0PingPongID_);
        cl0PingPongID_ ^= 1;
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<float> &dstGlobal, uint32_t dstStride)
    {
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cl0PingPongID_);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(cl0PingPongID_);
        AscendC::DataCopyCO12DstParams copyParams;
        copyParams.nSize = baseN_;
        copyParams.mSize = baseM_;
        copyParams.dstStride = dstStride;
        copyParams.srcStride = BasicApiAlign(baseM_, AscendC::BLOCK_CUBE);
        copyParams.quantPre = QuantMode_t::NoQuant;
        copyParams.nz2ndEn = true;
        copyParams.unitFlag = FINAL_ACCUMULATION;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(dstGlobal, cl0PingPongID_ == 0 ? cL0Ping_ : cL0Pong_, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cl0PingPongID_);
        cl0PingPongID_ ^= 1;
    }

    __aicore__ inline void End(uint32_t implMode)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
        if ASCEND_IS_AIC {
            if (implMode == MHC_PRE_IMPL_MODE_HF32) {
                AscendC::SetHF32Mode(0);
            }
        }
    }
};

} // namespace MhcPre

#endif // MHC_PRE_CUBE_COMPUTE_H
