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
 * \file block_mmad_fixpipe_dequant.h
 * \brief
 */

#ifndef BLOCK_MMAD_FIXPIPE_DEQUANT_H
#define BLOCK_MMAD_FIXPIPE_DEQUANT_H
#include "../policy/cgmct_dispatch_policy.h"
#include "../utils/cgmct_common_utils.h"
#include "../utils/cgmct_layout_utils.h"
#include "../utils/tuple_utils.h"

namespace Cgmct {
namespace Gemm {
namespace Block {
namespace {
constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
constexpr static uint64_t BLOCK_REDUCE_CUBE = 32UL;
constexpr static uint64_t IDX_M_TILE_IDX = 0UL;
constexpr static uint64_t IDX_N_TILE_IDX = 1UL;
constexpr static uint64_t IDX_M_IDX = 0UL;
constexpr static uint64_t IDX_N_IDX = 1UL;
constexpr static uint64_t IDX_K_IDX = 2UL;
constexpr static uint64_t EVEN_FACTOR = 2UL;
constexpr static uint64_t B8_MIN_STEP = 2UL;
constexpr static uint16_t INPUT_BUFFER_FLAG_0 = 0;
constexpr static uint16_t INPUT_BUFFER_FLAG_1 = 1;
constexpr static uint16_t INPUT_BUFFER_FLAG_2 = 2;
constexpr static uint16_t INPUT_BUFFER_FLAG_3 = 3;

struct TileL1L0Param {
    uint64_t curM = 0;
    uint64_t curN = 0;
    uint64_t curAlignM = 0;
    uint64_t curAlignN = 0;
    uint64_t curGmAKL1 = 0;
    uint64_t curGmBKL1 = 0;
    uint64_t curPadAKL1 = 0;
    uint64_t curPadBKL1 = 0;
    uint64_t curKL0 = 0;
};

} // namespace

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_,
          class Enable = void>
class BlockMmadFixpipeDequant {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>, "Should not be here!");
};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_>
class BlockMmadFixpipeDequant<
    DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_,
    LayoutBias_, TileCopy_, AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<MatmulWithScale<>, DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using BiasType = BiasType_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kAL1;
        uint64_t kBL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline BlockMmadFixpipeDequant()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~BlockMmadFixpipeDequant()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out
    }

public:
    __aicore__ inline void Init(const TupleShape &problemShape, const BlockShape &l0TileShape, const L1Params &l1Params,
                                bool isBias, bool dbL0C)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        kAL1_ = l1Params.kAL1;
        kBL1_ = l1Params.kBL1;

        orderAL1BL1_ = l1Params.kAL1 >= l1Params.kBL1;
        minKL1_ = Min(l1Params.kAL1, l1Params.kBL1);
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        aL1OneBuffer_ = baseM_ * kAL1_;
        bL1OneBuffer_ = baseN_ * kBL1_;

        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
            l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
        }
    }

    __aicore__ inline void UpdateParamsForNextProblem(const TupleShape &problemShape)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
    }

    __aicore__ inline void CopyInA1(const AscendC::GlobalTensor<AType> &aGlobal,
                                    const AscendC::LocalTensor<AType> &al1Local, const TileL1L0Param &tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transA ? tileL1L0Param.curGmAKL1 : tileL1L0Param.curM;
        uint64_t dDim = transA ? tileL1L0Param.curM : tileL1L0Param.curGmAKL1;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transA ? m_ : k_;
        nd2nzParams.dstNzC0Stride = transA ? tileL1L0Param.curPadAKL1 : tileL1L0Param.curAlignM;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1(const AscendC::GlobalTensor<BType> &bGlobal,
                                    const AscendC::LocalTensor<BType> &bl1Local, const TileL1L0Param &tileL1L0Param)
    {
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        uint64_t nDim = transB ? tileL1L0Param.curN : tileL1L0Param.curGmBKL1;
        uint64_t dDim = transB ? tileL1L0Param.curGmBKL1 : tileL1L0Param.curN;

        nd2nzParams.nValue = nDim;
        nd2nzParams.dValue = dDim;
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transB ? k_ : n_;
        nd2nzParams.dstNzC0Stride = transB ? tileL1L0Param.curAlignN : tileL1L0Param.curPadBKL1;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInL0A(const AscendC::LocalTensor<AType> &l0aLocal,
                                     const AscendC::LocalTensor<AType> &al1Local, const TileL1L0Param &tileL1L0Param,
                                     uint64_t kL0L1Off)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        uint64_t m1 = Cgmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE);
        if constexpr (transA) {
            loadDataParams.mStartPosition = Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if ((m1 & 1) == 0) {
                loadDataParams.mStep = Cgmct::Gemm::CeilAlign(
                    Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), EVEN_FACTOR);
            } else {
                loadDataParams.mStep = B8_MIN_STEP;
            }
            loadDataParams.kStep = Cgmct::Gemm::CeilDiv(tileL1L0Param.curM, C0_SIZE);
            loadDataParams.srcStride = Cgmct::Gemm::CeilDiv(tileL1L0Param.curPadAKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = m1;
            loadDataParams.ifTranspose = true;
        }
        AscendC::LoadData(l0aLocal, al1Local, loadDataParams);
        if constexpr (transA) {
            if ((m1 & 1) != 0) {
                AscendC::PipeBarrier<PIPE_MTE1>();
                uint64_t loadTimes =
                    Cgmct::Gemm::CeilDiv(Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), B8_MIN_STEP);
                for (uint64_t i = 1; i < loadTimes; i++) {
                    loadDataParams.mStartPosition =
                        B8_MIN_STEP * i + Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
                    AscendC::LoadData(l0aLocal[i * m1 * AscendC::BLOCK_CUBE * C0_SIZE], al1Local, loadDataParams);
                    AscendC::PipeBarrier<PIPE_MTE1>();
                }
            }
        }
    }

    __aicore__ inline void CopyInL0B(const AscendC::LocalTensor<BType> &l0bLocal,
                                     const AscendC::LocalTensor<BType> &bl1Local, const TileL1L0Param &tileL1L0Param,
                                     uint64_t kL0L1Off)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        uint64_t n1 = Cgmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE);
        if constexpr (!transB) {
            loadDataParams.mStartPosition = Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if ((n1 & 1) == 0) {
                loadDataParams.mStep = Cgmct::Gemm::CeilAlign(
                    Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), EVEN_FACTOR);
            } else {
                loadDataParams.mStep = B8_MIN_STEP;
            }
            loadDataParams.kStep = Cgmct::Gemm::CeilDiv(tileL1L0Param.curN, C0_SIZE);
            loadDataParams.srcStride = Cgmct::Gemm::CeilDiv(tileL1L0Param.curPadBKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = n1;
            loadDataParams.ifTranspose = true;
        }
        AscendC::LoadData(l0bLocal, bl1Local, loadDataParams);
        if constexpr (!transB) {
            if ((n1 & 1) != 0) {
                AscendC::PipeBarrier<PIPE_MTE1>();
                uint64_t loadTimes =
                    Cgmct::Gemm::CeilDiv(Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), B8_MIN_STEP);
                for (uint64_t i = 1; i < loadTimes; i++) {
                    loadDataParams.mStartPosition =
                        B8_MIN_STEP * i + Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
                    AscendC::LoadData(l0bLocal[i * n1 * AscendC::BLOCK_CUBE * C0_SIZE], bl1Local, loadDataParams);
                    AscendC::PipeBarrier<PIPE_MTE1>();
                }
            }
        }
    }

    __aicore__ inline void CopyFixpipeScalar(const AscendC::GlobalTensor<CType> &cGlobal,
                                             AscendC::LocalTensor<float> &c1Local, MmadParams &mmadParams,
                                             uint64_t scalarX2Scale)
    {
        AscendC::FixpipeParamsC310 intriParams;
        intriParams.nSize = mmadParams.n;
        intriParams.mSize = mmadParams.m;
        intriParams.dstStride = n_;
        intriParams.srcStride = Cgmct::Gemm::Align(mmadParams.m, AscendC::BLOCK_CUBE);
        intriParams.quantPre = QuantMode_t::QF322F32_PRE;
        intriParams.deqScalar = scalarX2Scale;
        intriParams.unitFlag = FINAL_ACCUMULATION;
        intriParams.params = {1, 1, 1};
        AscendC::Fixpipe(cGlobal, c1Local, intriParams);
    }

    __aicore__ inline void UpdateKAL1(TileL1L0Param &tileL1L0Param, uint64_t offsetKAL1)
    {
        tileL1L0Param.curGmAKL1 = (offsetKAL1 + kAL1_ > k_) ? (k_ - offsetKAL1) : kAL1_;
        if constexpr (transA) {
            tileL1L0Param.curPadAKL1 = Cgmct::Gemm::CeilAlign(tileL1L0Param.curGmAKL1, AscendC::BLOCK_CUBE);
        } else {
            tileL1L0Param.curPadAKL1 = Cgmct::Gemm::CeilAlign(tileL1L0Param.curGmAKL1, C0_SIZE);
        }
    }

    __aicore__ inline void UpdateKBL1(TileL1L0Param &tileL1L0Param, uint64_t offsetKBL1)
    {
        tileL1L0Param.curGmBKL1 = (offsetKBL1 + kBL1_ > k_) ? (k_ - offsetKBL1) : kBL1_;
        if constexpr (transB) {
            tileL1L0Param.curPadBKL1 = Cgmct::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, C0_SIZE);
        } else {
            tileL1L0Param.curPadBKL1 = Cgmct::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, AscendC::BLOCK_CUBE);
        }
    }

    __aicore__ inline void UpdateKL0(TileL1L0Param &tileL1L0Param, uint64_t kL0Offset, uint64_t minGmKL1)
    {
        if (kL0Offset + baseK_ > minGmKL1) {
            tileL1L0Param.curKL0 = minGmKL1 - kL0Offset;
        } else {
            tileL1L0Param.curKL0 = baseK_;
        }
    }

    __aicore__ inline void GetAlignMN(TileL1L0Param &tileL1L0Param)
    {
        if constexpr (transA) {
            tileL1L0Param.curAlignM = Cgmct::Gemm::CeilAlign(tileL1L0Param.curM, C0_SIZE);
        } else {
            tileL1L0Param.curAlignM = Cgmct::Gemm::CeilAlign(tileL1L0Param.curM, AscendC::BLOCK_CUBE);
        }
        if constexpr (!transB) {
            tileL1L0Param.curAlignN = Cgmct::Gemm::CeilAlign(tileL1L0Param.curN, C0_SIZE);
        } else {
            tileL1L0Param.curAlignN = Cgmct::Gemm::CeilAlign(tileL1L0Param.curN, AscendC::BLOCK_CUBE);
        }
    }

    __aicore__ inline void CopyAInL1(const AscendC::GlobalTensor<AType> &aGlobal, const TileL1L0Param &tileL1L0Param,
                                     uint64_t l1BufId, uint64_t kL1Offset)
    {
        uint64_t offsetA = transA ? kL1Offset * m_ : kL1Offset;
        CopyInA1(aGlobal[offsetA], aL1Local_[l1BufferAOffset_[l1BufId]], tileL1L0Param);
    }

    __aicore__ inline void CopyBInL1(const AscendC::GlobalTensor<BType> &bGlobal, const TileL1L0Param &tileL1L0Param,
                                     uint64_t l1BufId, uint64_t kL1Offset)
    {
        uint64_t offsetB = transB ? kL1Offset : kL1Offset * n_;
        CopyInB1(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
    }

    __aicore__ inline void Iterate(TileL1L0Param &tileL1L0Param, AscendC::MmadParams &mmadParams, uint64_t kL1Offset,
                                   uint64_t al1BufId, uint64_t bl1BufId, uint64_t l0cOffset, uint64_t kaL1Offset,
                                   uint64_t kbL1Offset)
    {
        uint64_t minGmKL1 = Cgmct::Gemm::Min(tileL1L0Param.curGmBKL1, tileL1L0Param.curGmAKL1);
        for (uint64_t kL0Offset = kL1Offset; kL0Offset < Min(kL1Offset + minKL1_, k_); kL0Offset += baseK_) {
            UpdateKL0(tileL1L0Param, kL0Offset - kL1Offset, minGmKL1);
            uint64_t kL0L1Off = kL0Offset - kL1Offset;
            // Load data to L0 and open DB
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            CopyInL0A(l0aLocal_[l0Offset], aL1Local_[l1BufferAOffset_[al1BufId]], tileL1L0Param, kL0L1Off + kaL1Offset);
            CopyInL0B(l0bLocal_[l0Offset], bL1Local_[l1BufferBOffset_[bl1BufId]], tileL1L0Param, kL0L1Off + kbL1Offset);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            mmadParams.k = tileL1L0Param.curKL0;
            mmadParams.unitFlag = (kL0Offset + baseK_ >= k_) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            mmadParams.cmatrixInitVal = (kL0Offset == 0 && !isBias_);
            Mmad(mmadParams, l0cOffset, l0Offset, baseN_ * biasBufId_, NeedBias(kL0Offset));
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            l0PingPong_++;
        }
    }
    __aicore__ inline void IterBL1AL1(TileL1L0Param &tileL1L0Param, AscendC::MmadParams &mmadParams, uint64_t l0cOffset,
                                      const AscendC::GlobalTensor<AType> &aGlobal,
                                      const AscendC::GlobalTensor<BType> &bGlobal,
                                      const AscendC::GlobalTensor<BiasType> &biasGlobal)
    {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += kBL1_) {
            uint64_t bL1BufId = bL1LoopCnt_ & (l1BufNum_ - 1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
            UpdateKBL1(tileL1L0Param, kOuter);
            CopyBInL1(bGlobal, tileL1L0Param, bL1BufId, kOuter);

            for (uint64_t kInner = kOuter; kInner < Min(kOuter + kBL1_, k_); kInner += kAL1_) {
                uint64_t aL1BufId = aL1LoopCnt_ & (l1BufNum_ - 1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aL1BufId);
                UpdateKAL1(tileL1L0Param, kInner);
                CopyAInL1(aGlobal, tileL1L0Param, aL1BufId, kInner);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                Iterate(tileL1L0Param, mmadParams, kInner, aL1BufId, bL1BufId, l0cOffset, 0, kInner - kOuter);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aL1BufId);
                aL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
            bL1LoopCnt_++;
        }
    }

    __aicore__ inline void IterAL1BL1(TileL1L0Param &tileL1L0Param, AscendC::MmadParams &mmadParams, uint64_t l0cOffset,
                                      const AscendC::GlobalTensor<AType> &aGlobal,
                                      const AscendC::GlobalTensor<BType> &bGlobal,
                                      const AscendC::GlobalTensor<BiasType> &biasGlobal)
    {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += kAL1_) {
            uint64_t aL1BufId = aL1LoopCnt_ & (l1BufNum_ - 1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
            UpdateKAL1(tileL1L0Param, kOuter);
            CopyAInL1(aGlobal, tileL1L0Param, aL1BufId, kOuter);

            for (uint64_t kInner = kOuter; kInner < Min(kOuter + kAL1_, k_); kInner += kBL1_) {
                uint64_t bL1BufId = bL1LoopCnt_ & (l1BufNum_ - 1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bL1BufId);
                UpdateKBL1(tileL1L0Param, kInner);
                CopyBInL1(bGlobal, tileL1L0Param, bL1BufId, kInner);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                Iterate(tileL1L0Param, mmadParams, kInner, aL1BufId, bL1BufId, l0cOffset, kInner - kOuter, 0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bL1BufId);
                bL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
            aL1LoopCnt_++;
        }
    }

    __aicore__ inline void run(const AscendC::GlobalTensor<AType> &aGlobal, const AscendC::GlobalTensor<BType> &bGlobal,
                               uint64_t scalarScale, const AscendC::GlobalTensor<BiasType> &biasGlobal,
                               const AscendC::GlobalTensor<CType> &cGlobal, const BlockShape &singleShape)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = Get<IDX_M_TILE_IDX>(singleShape);
        tileL1L0Param.curN = Get<IDX_N_TILE_IDX>(singleShape);
        GetAlignMN(tileL1L0Param);
        AscendC::MmadParams mmadParams;
        mmadParams.m = tileL1L0Param.curM;
        mmadParams.n = tileL1L0Param.curN;
        mmadParams.disableGemv = true;
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        if (orderAL1BL1_) {
            IterAL1BL1(tileL1L0Param, mmadParams, l0cOffset, aGlobal, bGlobal, biasGlobal);
        } else {
            IterBL1AL1(tileL1L0Param, mmadParams, l0cOffset, aGlobal, bGlobal, biasGlobal);
        }
        // Copy out to GM
        AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
        CopyFixpipeScalar(cGlobal, c1Local, mmadParams, scalarScale);
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<AType> &aGlobal,
                                      const AscendC::GlobalTensor<BType> &bGlobal, uint64_t scalarScale,
                                      const AscendC::GlobalTensor<CType> &cGlobal, const BlockShape &singleShape)
    {
        AscendC::GlobalTensor<BiasType> biasGlobal;
        run(aGlobal, bGlobal, scalarScale, biasGlobal, cGlobal, singleShape);
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<AType> &aGlobal,
                                      const AscendC::GlobalTensor<BType> &bGlobal, uint64_t scalarScale,
                                      const AscendC::GlobalTensor<BiasType> &biasGlobal,
                                      const AscendC::GlobalTensor<CType> &cGlobal, const BlockShape &singleShape)
    {
        run(aGlobal, bGlobal, scalarScale, biasGlobal, cGlobal, singleShape);
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kL0Offset)
    {
        return isBias_ && kL0Offset == 0;
    }

    __aicore__ inline void Mmad(AscendC::MmadParams &mmadParams, uint64_t l0cOffset, uint64_t l0abOffset,
                                uint64_t biasOffset, bool needBias)
    {
        mmadParams.cmatrixSource = needBias;
        if (needBias) {
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], biasBt_[biasOffset],
                          mmadParams);
        } else {
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
        }
    }

private:
    uint16_t biasBufId_ = 0;
    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[2] = {0UL};    // default 2 buffer
    uint64_t l1BufferBOffset_[2] = {0UL};    // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL}; // default 2 buffer
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};
    uint64_t minKL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t aL1LoopCnt_{0};
    uint64_t bL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool orderAL1BL1_{false};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    AscendC::LocalTensor<AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<BiasType> biasBt_{AscendC::TPosition::C2, 0, 4096}; // 4096B: BT_SIZE
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BiasType> biasL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(BiasType)};
};
} // namespace Block
} // namespace Gemm
} // namespace Cgmct
#endif
