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
 * \file block_mmad_mx.h
 * \brief
 */

#ifndef MATMUL_BLOCK_MMAD_MX_H
#define MATMUL_BLOCK_MMAD_MX_H
#include "../policy/cgmct_dispatch_policy.h"
#include "../utils/cgmct_common_utils.h"
#include "../utils/cgmct_layout_utils.h"
#include "../utils/tuple_utils.h"

namespace Cgmct {
namespace Gemm {
namespace Block {
using AscendC::fp8_e8m0_t;
namespace {
constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);
constexpr static uint64_t BLOCK_REDUCE_CUBE = 32UL;
constexpr static uint64_t IDX_M_TILE_IDX = 0UL;
constexpr static uint64_t IDX_N_TILE_IDX = 1UL;
constexpr static uint64_t IDX_M_IDX = 0UL;
constexpr static uint64_t IDX_N_IDX = 1UL;
constexpr static uint64_t IDX_K_IDX = 2UL;
constexpr static uint64_t SCALE_BUFFER_NUM = 2UL;
constexpr static uint64_t EVEN_FACTOR = 2UL;
constexpr static uint64_t B8_MIN_STEP = 2UL;
constexpr static uint64_t B4_MIN_STEP = 4UL;
constexpr static uint16_t INPUT_BUFFER_FLAG_0 = 0;
constexpr static uint16_t INPUT_BUFFER_FLAG_1 = 1;
constexpr static uint16_t INPUT_BUFFER_FLAG_2 = 2;
constexpr static uint16_t INPUT_BUFFER_FLAG_3 = 3;
constexpr static uint16_t SCALE_BUFFER_FLAG_0 = 4;
constexpr static uint16_t SCALE_BUFFER_FLAG_1 = 5;

struct TileL1L0Param {
    uint64_t curM = 0;
    uint64_t curN = 0;
    uint64_t curAlignM = 0;
    uint64_t curAlignN = 0;
    uint64_t curGmAKL1 = 0;
    uint64_t curGmBKL1 = 0;
    uint64_t curPadAKL1 = 0; // pad to 64 align for mx
    uint64_t curPadBKL1 = 0; // pad to 64 align for mx
    uint64_t curKL0 = 0;
};

} // namespace

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_,
          class Enable = void>
class BlockMmadMx {
    static_assert(AscendC::Std::always_false_v<DispatchPolicy_>, "Should not be here!");
};

template <class DispatchPolicy_, class L1TileShape_, class L0TileShape_, class AType_, class LayoutA_, class BType_,
          class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_, class TileCopy_>
class BlockMmadMx<DispatchPolicy_, L1TileShape_, L0TileShape_, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_,
                  BiasType_, LayoutBias_, TileCopy_,
                  AscendC::Std::enable_if_t<
                      AscendC::Std::is_base_of_v<MatmulWithScale<>, DispatchPolicy_> ||
                      AscendC::Std::is_base_of_v<MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>,
                                                 DispatchPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using MxL0AType = typename AscendC::GetL0DataType<AType, true>::Type;
    using MxL0BType = typename AscendC::GetL0DataType<BType, true>::Type;
    using BiasType = BiasType_;
    using DispatchPolicy = DispatchPolicy_;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t>;
    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr bool IS_FP4 =
        AscendC::IsSameType<AType, fp4x2_e2m1_t>::value || AscendC::IsSameType<AType, fp4x2_e1m2_t>::value;
    static constexpr uint64_t MIN_STEP = IS_FP4 ? B4_MIN_STEP : B8_MIN_STEP;
    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static int32_t C0_SIZE = AscendC::AuxGetC0Size<AType>();
    constexpr static int32_t BIAS_C0 = AscendC::AuxGetC0Size<BiasType>();

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
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline BlockMmadMx()
    {
        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
            AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
        }
    }

    __aicore__ inline ~BlockMmadMx()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
            AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out
        }
    }

public:
    __aicore__ inline void Init(const TupleShape &problemShape, const BlockShape &l0TileShape, const L1Params &l1Params,
                                bool isBias, bool dbL0C)
    {
        m_ = Get<IDX_M_IDX>(problemShape);
        n_ = Get<IDX_N_IDX>(problemShape);
        k_ = Get<IDX_K_IDX>(problemShape);
        kAL1_ = l1Params.kAL1;
        kBL1_ = l1Params.kBL1;
        scaleKL1_ = l1Params.scaleKL1;
        baseM_ = Get<IDX_M_IDX>(l0TileShape);
        baseN_ = Get<IDX_N_IDX>(l0TileShape);
        baseK_ = Get<IDX_K_IDX>(l0TileShape);
        orderAL1BL1_ = l1Params.kAL1 >= l1Params.kBL1;
        minKL1_ = Min(l1Params.kAL1, l1Params.kBL1);
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        bL1OneBuffer_ = baseN_ * Cgmct::Gemm::Align(kBL1_, MXFP_DIVISOR_SIZE);
        mxScaleKL1B16_ = Cgmct::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE);
        auto mxScaleKL1 = mxScaleKL1B16_ * MXFP_MULTI_BASE_SIZE;
        scaleBL1OneBuffer_ = baseN_ * mxScaleKL1;
        if (isBias_) {
            biasL1OneBuffer_ = baseN_ * sizeof(BiasType);
        }
        aL1OneBuffer_ = baseM_ * Cgmct::Gemm::Align(kAL1_, MXFP_DIVISOR_SIZE);
        scaleAL1OneBuffer_ = baseM_ * mxScaleKL1;
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|bias0|...|A1|B1|AScale1|BScale1|bias1|...
            // l1Offset represents element count
            uint64_t l1Offset =
                IS_FP4 ? AscendC::TOTAL_L1_SIZE * (bufferId & 1) : (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
            l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
        }
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            // l1BufferScaleAOffset_[bufferId]: byte count when B8, element count when B4
            l1BufferScaleAOffset_[bufferId] =
                IS_FP4 ? ((l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1)) >> 1) :
                         (l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1));
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
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
        nd2nzParams.nValue = transA ? tileL1L0Param.curGmAKL1 : tileL1L0Param.curM;
        nd2nzParams.dValue = transA ? tileL1L0Param.curM : tileL1L0Param.curGmAKL1;
        if constexpr (IS_FP4) {
            nd2nzParams.dValue >>= 1;
        }
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transA ? m_ : k_;
        if constexpr (IS_FP4) {
            nd2nzParams.srcDValue >>= 1;
        }
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
        nd2nzParams.nValue = transB ? tileL1L0Param.curN : tileL1L0Param.curGmBKL1;
        nd2nzParams.dValue = transB ? tileL1L0Param.curGmBKL1 : tileL1L0Param.curN;
        if constexpr (IS_FP4) {
            nd2nzParams.dValue >>= 1;
        }
        nd2nzParams.srcNdMatrixStride = 1;
        nd2nzParams.srcDValue = transB ? k_ : n_;
        if constexpr (IS_FP4) {
            nd2nzParams.srcDValue >>= 1;
        }
        nd2nzParams.dstNzC0Stride = transB ? tileL1L0Param.curAlignN : tileL1L0Param.curPadBKL1;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 1;
        AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
    }

    __aicore__ inline void CopyInB1WeightNz(const AscendC::GlobalTensor<BType> &bGlobal,
                                            const AscendC::LocalTensor<BType> &bl1Local,
                                            const TileL1L0Param &tileL1L0Param)
    {
        AscendC::DataCopyExtParams dataCopyParams;
        AscendC::DataCopyPadExtParams<uint8_t> padParams;
        if constexpr (transB) {
            dataCopyParams.blockCount = Cgmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, C0_SIZE);
            dataCopyParams.blockLen = tileL1L0Param.curAlignN * C0_SIZE;
            if constexpr (IS_FP4) {
                dataCopyParams.blockLen >>= 1;
            }
            dataCopyParams.srcStride = (Align16(n_) - tileL1L0Param.curAlignN) * C0_SIZE;
            if constexpr (IS_FP4) {
                dataCopyParams.srcStride >>= 1;
            }
            dataCopyParams.dstStride = 0;
        } else {
            int64_t curGmBKL1NZ = Align16(tileL1L0Param.curGmBKL1);
            dataCopyParams.blockCount = tileL1L0Param.curAlignN / C0_SIZE;
            dataCopyParams.blockLen = curGmBKL1NZ * C0_SIZE;
            if constexpr (IS_FP4) {
                dataCopyParams.blockLen >>= 1;
            }
            dataCopyParams.srcStride = (Align16(k_) - curGmBKL1NZ) * C0_SIZE;
            if constexpr (IS_FP4) {
                dataCopyParams.srcStride >>= 1;
            }
            dataCopyParams.dstStride = tileL1L0Param.curPadBKL1 - curGmBKL1NZ;
        }
        AscendC::DataCopyPad(bl1Local.template ReinterpretCast<uint8_t>(), bGlobal.template ReinterpretCast<uint8_t>(),
                             dataCopyParams, padParams);
    }

    __aicore__ inline void InitA1(const AscendC::LocalTensor<AType> &al1Local, const TileL1L0Param &tileL1L0Param)
    {
        AscendC::InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        if constexpr (!transA) {
            // nd2nz pading to k 64 align, (m,k)->(k1,m1,m0,k0)
            if (tileL1L0Param.curPadAKL1 - tileL1L0Param.curGmAKL1 < BLOCK_REDUCE_CUBE) {
                return;
            }
            // pad m1, m0, 16 for half
            initConstValueParams.repeatTimes = 1;
            initConstValueParams.blockNum = tileL1L0Param.curAlignM;
            initConstValueParams.dstGap = 0;
            initConstValueParams.initValue = 0;
            uint64_t kAlign = Cgmct::Gemm::CeilDiv(tileL1L0Param.curGmAKL1, C0_SIZE) * AscendC::BLOCK_CUBE;
            offset = tileL1L0Param.curAlignM * kAlign;
        } else { // nd2nz pading to k 64 align, (k,m)->(m1,k1,k0,m0)
            if (tileL1L0Param.curPadAKL1 == tileL1L0Param.curGmAKL1) {
                return;
            }
            offset = tileL1L0Param.curGmAKL1 * AscendC::BLOCK_CUBE;
            initConstValueParams.repeatTimes = Cgmct::Gemm::CeilDiv(tileL1L0Param.curAlignM, C0_SIZE);
            initConstValueParams.blockNum = tileL1L0Param.curPadAKL1 - tileL1L0Param.curGmAKL1;
            initConstValueParams.dstGap = tileL1L0Param.curGmAKL1;
            initConstValueParams.initValue = 0;
        }
        AscendC::InitConstValue(al1Local.template ReinterpretCast<half>()[offset], initConstValueParams);
    }

    __aicore__ inline void InitB1(const AscendC::LocalTensor<BType> &bl1Local, const TileL1L0Param &tileL1L0Param)
    {
        AscendC::InitConstValueParams<half> initConstValueParams;
        uint64_t offset = 0;
        if constexpr (transB) {
            // nd2nz pading to 64 align, (n,k)->(k1,n1,n0,k0)
            if (tileL1L0Param.curPadBKL1 - tileL1L0Param.curGmBKL1 < BLOCK_REDUCE_CUBE) {
                return;
            }
            // pad n1, n0, 16 for half
            initConstValueParams.repeatTimes = 1;
            initConstValueParams.blockNum = tileL1L0Param.curAlignN;
            initConstValueParams.dstGap = 0;
            initConstValueParams.initValue = 0;
            uint64_t kAlign = Cgmct::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, C0_SIZE) * AscendC::BLOCK_CUBE;
            offset = tileL1L0Param.curAlignN * kAlign;
        } else { // nd2nz pading to k 64 align, (k,n)->(n1,k1,k0,n0)
            uint64_t movK = tileL1L0Param.curGmBKL1;
            if constexpr (formatB == CubeFormat::NZ) {
                movK = Align16(tileL1L0Param.curGmBKL1);
            }
            if (tileL1L0Param.curPadBKL1 == movK) {
                return;
            }
            offset = movK * AscendC::BLOCK_CUBE;
            initConstValueParams.repeatTimes = Cgmct::Gemm::CeilDiv(tileL1L0Param.curAlignN, C0_SIZE);
            initConstValueParams.blockNum = tileL1L0Param.curPadBKL1 - movK;
            initConstValueParams.dstGap = movK;
            initConstValueParams.initValue = 0;
        }
        AscendC::InitConstValue(bl1Local.template ReinterpretCast<half>()[offset], initConstValueParams);
    }

    __aicore__ inline void CopyInBias(const AscendC::GlobalTensor<BiasType> &biasGlobal,
                                      const AscendC::LocalTensor<BiasType> &cl1Local, uint64_t curNL1)
    {
        AscendC::DataCopyPadParams padParams;
        // 单位为Byte
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(curNL1 * sizeof(BiasType)), 0, 0};
        AscendC::DataCopyPad(cl1Local, biasGlobal, biasParam, padParams);
    }

    __aicore__ inline void CopyInScaleA(const AscendC::GlobalTensor<fp8_e8m0_t> &aScaleGlobal,
                                        const AscendC::LocalTensor<fp8_e8m0_t> &aScaleL1Local, uint64_t curML1,
                                        uint64_t kL1Offset)
    {
        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }

        AscendC::GlobalTensor<half> aScaleGlobalB16;
        aScaleGlobalB16.SetGlobalBuffer(((__gm__ half *)(aScaleGlobal.GetPhyAddr())));
        auto aScaleL1LocalImpl = aScaleL1Local.template ReinterpretCast<half>();

        if constexpr (!transA) {
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1;
            dn2nzParams.dValue = curML1;
            dn2nzParams.nValue = Cgmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
            dn2nzParams.srcDnMatrixStride = 0;
            dn2nzParams.srcDValue = Cgmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE);
            dn2nzParams.dstNzC0Stride = mxScaleKL1B16_;
            dn2nzParams.dstNzNStride = 1;
            dn2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[kL1Offset / MXFP_DIVISOR_SIZE], dn2nzParams);
        } else {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = Cgmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
            nd2nzParams.dValue = curML1;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = m_;
            nd2nzParams.dstNzC0Stride = mxScaleKL1B16_;
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(aScaleL1LocalImpl, aScaleGlobalB16[(kL1Offset / MXFP_DIVISOR_SIZE) * m_], nd2nzParams);
        }
    }

    __aicore__ inline void CopyInScaleB(const AscendC::GlobalTensor<fp8_e8m0_t> &bScaleGlobal,
                                        const AscendC::LocalTensor<fp8_e8m0_t> &bScaleL1Local, uint64_t curNL1,
                                        uint64_t kL1Offset)
    {
        uint64_t curScaleKL1 = scaleKL1_;
        if (kL1Offset + curScaleKL1 > k_) {
            curScaleKL1 = k_ - kL1Offset;
        }

        AscendC::GlobalTensor<half> bScaleGlobalB16;
        bScaleGlobalB16.SetGlobalBuffer(((__gm__ half *)(bScaleGlobal.GetPhyAddr())));
        auto bScaleL1LocalImpl = bScaleL1Local.template ReinterpretCast<half>();

        if constexpr (transB) {
            AscendC::Dn2NzParams dn2nzParams;
            dn2nzParams.dnNum = 1;
            dn2nzParams.dValue = curNL1;
            dn2nzParams.nValue = Cgmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
            dn2nzParams.srcDnMatrixStride = 0;
            dn2nzParams.srcDValue = Cgmct::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE);
            dn2nzParams.dstNzC0Stride = mxScaleKL1B16_;
            dn2nzParams.dstNzNStride = 1;
            dn2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[kL1Offset / MXFP_DIVISOR_SIZE], dn2nzParams);
        } else {
            AscendC::Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = Cgmct::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE);
            nd2nzParams.dValue = curNL1;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = n_;
            nd2nzParams.dstNzC0Stride = mxScaleKL1B16_;
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(bScaleL1LocalImpl, bScaleGlobalB16[(kL1Offset / MXFP_DIVISOR_SIZE) * n_], nd2nzParams);
        }
    }

    __aicore__ inline void CopyInC2(const AscendC::LocalTensor<BiasType> &biasL1Local,
                                    const AscendC::LocalTensor<float> &biasBt, uint64_t nl1Align, bool needBias)
    {
        if (!needBias) {
            return;
        }
        // s32场景要对齐到2 因此是align(nl1Align / 8, 2)
        uint64_t btAlign = AscendC::BLOCK_CUBE / BIAS_C0;
        uint16_t bustLenth = Cgmct::Gemm::Align(nl1Align / BIAS_C0, btAlign);
        AscendC::DataCopyParams biasParam{1, static_cast<uint16_t>(bustLenth), 0, 0};
        // 当dstlocal位于C2时，C2中至少为fp32*16
        AscendC::DataCopy(biasBt, biasL1Local, biasParam);
    }

    __aicore__ inline void CopyInL0A(const AscendC::LocalTensor<MxL0AType> &l0aLocal,
                                     const AscendC::LocalTensor<AType> &al1Local,
                                     const AscendC::LocalTensor<fp8_e8m0_t> &scaleAl1Local,
                                     const TileL1L0Param &tileL1L0Param, uint64_t kL0L1Off)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t m1 = Cgmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE);
        if constexpr (!transA) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cgmct::Gemm::CeilDiv(kL0L1Off, C0_SIZE);
            loadDataParams.mStep = m1;
            loadDataParams.kStep = Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadDataParams.srcStride = loadDataParams.mStep;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            loadDataParams.mStartPosition = Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if (((m1 & 1) == 0 && !(IS_FP4)) ||
                (IS_FP4 && (Cgmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE) & (B4_MIN_STEP - 1)) == 0)) {
                loadDataParams.mStep = Cgmct::Gemm::CeilAlign(
                    Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), MXFP_MULTI_BASE_SIZE);
            } else {
                loadDataParams.mStep = MIN_STEP;
            }
            loadDataParams.kStep = Cgmct::Gemm::CeilDiv(tileL1L0Param.curM, C0_SIZE);
            loadDataParams.srcStride = Cgmct::Gemm::CeilDiv(tileL1L0Param.curPadAKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = m1;
            loadDataParams.ifTranspose = true;
        }
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = 0;
        loadData2DMxParams.xStep = m1;
        loadData2DMxParams.yStep = Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = mxScaleKL1B16_;
        loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        AscendC::LoadData(l0aLocal, al1Local, scaleAl1Local, loadDataParams, loadData2DMxParams);
        if constexpr (transA) {
            if (((m1 & 1) != 0 && !(IS_FP4)) ||
                (IS_FP4 && (Cgmct::Gemm::CeilDiv(tileL1L0Param.curM, AscendC::BLOCK_CUBE) & (B4_MIN_STEP - 1)) != 0)) {
                AscendC::PipeBarrier<PIPE_MTE1>();
                AscendC::LocalTensor<AType> l0a = l0aLocal.template ReinterpretCast<AType>();
                uint64_t loadTimes =
                    Cgmct::Gemm::CeilDiv(Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), MIN_STEP);
                if constexpr (IS_FP4) {
                    loadData2DMxParams.xStartPosition = 0;
                    loadData2DMxParams.yStartPosition = 0;
                    loadData2DMxParams.xStep = 0;
                    loadData2DMxParams.yStep = 0;
                    loadData2DMxParams.srcStride = 0;
                    loadData2DMxParams.dstStride = 0;
                    for (uint64_t i = 1; i < loadTimes; i++) {
                        loadDataParams.mStartPosition =
                            MIN_STEP * i + Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
                        AscendC::LoadData(l0a[i * m1 * AscendC::BLOCK_CUBE * C0_SIZE], al1Local, scaleAl1Local,
                                          loadDataParams, loadData2DMxParams);
                        AscendC::PipeBarrier<PIPE_MTE1>();
                    }
                } else {
                    for (uint64_t i = 1; i < loadTimes; i++) {
                        loadDataParams.mStartPosition =
                            MIN_STEP * i + Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
                        AscendC::LoadData(l0a[i * m1 * AscendC::BLOCK_CUBE * C0_SIZE], al1Local, loadDataParams);
                        AscendC::PipeBarrier<PIPE_MTE1>();
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyInL0B(const AscendC::LocalTensor<MxL0BType> &l0bLocal,
                                     const AscendC::LocalTensor<BType> &bl1Local,
                                     const AscendC::LocalTensor<fp8_e8m0_t> &scaleBl1Local,
                                     const TileL1L0Param &tileL1L0Param, uint64_t kL0L1Off)
    {
        AscendC::LoadData2DParamsV2 loadDataParams;
        AscendC::LoadData2DMxParams loadData2DMxParams;
        uint64_t n1 = Cgmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE);
        if constexpr (transB) {
            loadDataParams.mStartPosition = 0;
            loadDataParams.kStartPosition = Cgmct::Gemm::CeilDiv(kL0L1Off, C0_SIZE);
            loadDataParams.mStep = n1;
            loadDataParams.kStep = Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, C0_SIZE);
            loadDataParams.srcStride = loadDataParams.mStep;
            loadDataParams.dstStride = loadDataParams.mStep;
            loadDataParams.ifTranspose = false;
        } else {
            loadDataParams.mStartPosition = Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
            loadDataParams.kStartPosition = 0;
            if (((n1 & 1) == 0 && !(IS_FP4)) ||
                (IS_FP4 && (Cgmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE) & (B4_MIN_STEP - 1)) == 0)) {
                loadDataParams.mStep = Cgmct::Gemm::CeilAlign(
                    Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), EVEN_FACTOR);
            } else {
                loadDataParams.mStep = MIN_STEP;
            }
            loadDataParams.kStep = Cgmct::Gemm::CeilDiv(tileL1L0Param.curN, C0_SIZE);
            loadDataParams.srcStride = Cgmct::Gemm::CeilDiv(tileL1L0Param.curPadBKL1, AscendC::BLOCK_CUBE);
            loadDataParams.dstStride = n1;
            loadDataParams.ifTranspose = true;
        }
        loadData2DMxParams.xStartPosition = 0;
        loadData2DMxParams.yStartPosition = 0;
        loadData2DMxParams.xStep = n1;
        loadData2DMxParams.yStep = Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, MXFP_DIVISOR_SIZE);
        loadData2DMxParams.srcStride = mxScaleKL1B16_;
        loadData2DMxParams.dstStride = loadData2DMxParams.yStep;
        AscendC::LoadData(l0bLocal, bl1Local, scaleBl1Local, loadDataParams, loadData2DMxParams);
        if constexpr (!transB) {
            if (((n1 & 1) != 0 && !(IS_FP4)) ||
                (IS_FP4 && (Cgmct::Gemm::CeilDiv(tileL1L0Param.curN, AscendC::BLOCK_CUBE) & (B4_MIN_STEP - 1)) != 0)) {
                AscendC::PipeBarrier<PIPE_MTE1>();
                AscendC::LocalTensor<BType> l0b = l0bLocal.template ReinterpretCast<BType>();
                uint64_t loadTimes =
                    Cgmct::Gemm::CeilDiv(Cgmct::Gemm::CeilDiv(tileL1L0Param.curKL0, AscendC::BLOCK_CUBE), MIN_STEP);
                if constexpr (IS_FP4) {
                    loadData2DMxParams.xStartPosition = 0;
                    loadData2DMxParams.yStartPosition = 0;
                    loadData2DMxParams.xStep = 0;
                    loadData2DMxParams.yStep = 0;
                    loadData2DMxParams.srcStride = 0;
                    loadData2DMxParams.dstStride = 0;
                    for (uint64_t i = 1; i < loadTimes; i++) {
                        loadDataParams.mStartPosition =
                            MIN_STEP * i + Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
                        AscendC::LoadData(l0b[i * n1 * AscendC::BLOCK_CUBE * C0_SIZE], bl1Local, scaleBl1Local,
                                          loadDataParams, loadData2DMxParams);
                        AscendC::PipeBarrier<PIPE_MTE1>();
                    }
                } else {
                    for (uint64_t i = 1; i < loadTimes; i++) {
                        loadDataParams.mStartPosition =
                            MIN_STEP * i + Cgmct::Gemm::CeilDiv(kL0L1Off, AscendC::BLOCK_CUBE);
                        AscendC::LoadData(l0b[i * n1 * AscendC::BLOCK_CUBE * C0_SIZE], bl1Local, loadDataParams);
                        AscendC::PipeBarrier<PIPE_MTE1>();
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<CType> &cGlobal,
                                   const AscendC::LocalTensor<float> &c1Local, uint64_t baseM, uint64_t baseN)
    {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.nSize = baseN;
        intriParams.mSize = baseM;
        intriParams.dstStride = n_;
        intriParams.srcStride = Align16(baseM);
        // set mode according to dtype
        if constexpr (AscendC::IsSameType<CType, bfloat16_t>::value) {
            intriParams.quantPre = QuantMode_t::F322BF16;
        } else if (AscendC::IsSameType<CType, half>::value) {
            intriParams.quantPre = QuantMode_t::F322F16;
        } else if (AscendC::IsSameType<CType, float>::value) {
            intriParams.quantPre = QuantMode_t::NoQuant;
        }
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = FINAL_ACCUMULATION; // 3 unitflag
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(cGlobal, c1Local, intriParams);
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<CType> &cLocal,
                                   const AscendC::LocalTensor<float> &c1Local, uint64_t baseM, uint64_t baseN)
    {
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.mSize = static_cast<uint16_t>(Cgmct::Gemm::Align(baseM, EVEN_FACTOR));
        fixpipeParams.nSize = static_cast<uint16_t>(baseN);
        fixpipeParams.srcStride = static_cast<uint16_t>(Cgmct::Gemm::Align(baseM, AscendC::BLOCK_CUBE));
        fixpipeParams.dstStride = baseN;
        fixpipeParams.dualDstCtl = static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_M);
        fixpipeParams.subBlockId = 0;
        fixpipeParams.unitFlag = FINAL_ACCUMULATION;
        AscendC::Fixpipe<CType, CType, AscendC::Impl::CFG_ROW_MAJOR_UB>(cLocal, c1Local, fixpipeParams);
    }

    __aicore__ inline void UpdateKAL1(TileL1L0Param &tileL1L0Param, uint64_t offsetKAL1)
    {
        tileL1L0Param.curGmAKL1 = Min(kAL1_, k_ - offsetKAL1);
        tileL1L0Param.curPadAKL1 = Align64(tileL1L0Param.curGmAKL1);
    }

    __aicore__ inline void UpdateKBL1(TileL1L0Param &tileL1L0Param, uint64_t offsetKBL1)
    {
        tileL1L0Param.curGmBKL1 = Min(kBL1_, k_ - offsetKBL1);
        tileL1L0Param.curPadBKL1 = Align64(tileL1L0Param.curGmBKL1);
    }

    __aicore__ inline void GetAlignMN(TileL1L0Param &tileL1L0Param)
    {
        if constexpr (transA) {
            tileL1L0Param.curAlignM = IS_FP4 ? Align64(tileL1L0Param.curM) : Align32(tileL1L0Param.curM);
        } else {
            tileL1L0Param.curAlignM = Align16(tileL1L0Param.curM);
        }
        if constexpr (!transB) {
            tileL1L0Param.curAlignN = IS_FP4 ? Align64(tileL1L0Param.curN) : Align32(tileL1L0Param.curN);
        } else {
            tileL1L0Param.curAlignN = Align16(tileL1L0Param.curN);
        }
    }

    __aicore__ inline void CopyScalesInL1(const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                          const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                                          const TileL1L0Param &tileL1L0Param, uint64_t kL1Offset, uint64_t scaleL1BufId)
    {
        if (kL1Offset % scaleKL1_ == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
            CopyInScaleA(scaleAGlobal, scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId]], tileL1L0Param.curM,
                         kL1Offset);
            CopyInScaleB(scaleBGlobal, scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId]], tileL1L0Param.curN,
                         kL1Offset);
        }
    }

    __aicore__ inline void CopyAInL1(const AscendC::GlobalTensor<AType> &aGlobal, const TileL1L0Param &tileL1L0Param,
                                     uint64_t l1BufId, uint64_t kL1Offset)
    {
        if constexpr (!(IS_FP4 && !transA)) {
            InitA1(aL1Local_[l1BufferAOffset_[l1BufId]], tileL1L0Param);
        }
        uint64_t offsetA = transA ? kL1Offset * m_ : kL1Offset;
        CopyInA1(aGlobal[offsetA], aL1Local_[l1BufferAOffset_[l1BufId]], tileL1L0Param);
    }

    __aicore__ inline void CopyBInL1(const AscendC::GlobalTensor<BType> &bGlobal, const TileL1L0Param &tileL1L0Param,
                                     uint64_t l1BufId, uint64_t kL1Offset)
    {
        if constexpr (!(IS_FP4 && transB)) {
            InitB1(bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
        }
        if constexpr (formatB == CubeFormat::NZ) {
            uint64_t offsetB = transB ? kL1Offset * Align16(n_) : kL1Offset * C0_SIZE;
            CopyInB1WeightNz(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
        } else {
            uint64_t offsetB = transB ? kL1Offset : kL1Offset * n_;
            CopyInB1(bGlobal[offsetB], bL1Local_[l1BufferBOffset_[l1BufId]], tileL1L0Param);
        }
    }

    __aicore__ inline void Iterate(TileL1L0Param &tileL1L0Param, AscendC::MmadParams &mmadParams, uint64_t kL1Offset,
                                   uint64_t al1BufId, uint64_t bl1BufId, uint64_t scaleL1BufId, uint64_t l0cOffset,
                                   uint64_t kaL1Offset, uint64_t kbL1Offset)
    {
        uint64_t minPadKL1 = Cgmct::Gemm::Min(tileL1L0Param.curPadBKL1, tileL1L0Param.curPadAKL1);
        for (uint64_t kL0Offset = kL1Offset; kL0Offset < Min(kL1Offset + minKL1_, k_); kL0Offset += baseK_) {
            tileL1L0Param.curKL0 = Min(baseK_, minPadKL1 + kL1Offset - kL0Offset);
            uint64_t kL0L1Off = kL0Offset - kL1Offset;
            // Load data to L0 and open DB
            // l0Offset represents element count
            uint64_t l0Offset = IS_FP4 ? (HALF_L0_SIZE << 1) * (l0PingPong_ & 0x1) : HALF_L0_SIZE * (l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingPong_ & 0x1);
            uint64_t offsetScaleL1 = AscendC::BLOCK_CUBE * ((kL0Offset % scaleKL1_) / MXFP_GROUP_SIZE);
            CopyInL0A(l0aLocal_[l0Offset], aL1Local_[l1BufferAOffset_[al1BufId]],
                      scaleAL1Local_[l1BufferScaleAOffset_[scaleL1BufId] + offsetScaleL1], tileL1L0Param,
                      kL0L1Off + kaL1Offset);
            // copy bias to bt
            CopyInC2(biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)], biasBt_[baseN_ * biasBufId_],
                     Align16(mmadParams.n), NeedBias(kL0Offset));
            CopyInL0B(l0bLocal_[l0Offset], bL1Local_[l1BufferBOffset_[bl1BufId]],
                      scaleBL1Local_[l1BufferScaleBOffset_[scaleL1BufId] + offsetScaleL1], tileL1L0Param,
                      kL0L1Off + kbL1Offset);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            mmadParams.k = Align64(tileL1L0Param.curKL0);
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
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                                      const AscendC::GlobalTensor<BiasType> &biasGlobal)
    {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += kBL1_) {
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t bL1BufId = bL1LoopCnt_ & (l1BufNum_ - 1);
            CopyScalesInL1(scaleAGlobal, scaleBGlobal, tileL1L0Param, kOuter, scaleL1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
            biasBufId_ = scaleL1BufId;
            UpdateKBL1(tileL1L0Param, kOuter);
            CopyBInL1(bGlobal, tileL1L0Param, bL1BufId, kOuter);
            if (isBias_ && kOuter == 0) {
                CopyInBias(biasGlobal, biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)],
                           tileL1L0Param.curN);
            }
            for (uint64_t kInner = kOuter; kInner < Min(kOuter + kBL1_, k_); kInner += kAL1_) {
                uint64_t aL1BufId = aL1LoopCnt_ & (l1BufNum_ - 1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aL1BufId);
                UpdateKAL1(tileL1L0Param, kInner);
                CopyAInL1(aGlobal, tileL1L0Param, aL1BufId, kInner);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aL1BufId);
                Iterate(tileL1L0Param, mmadParams, kInner, aL1BufId, bL1BufId, scaleL1BufId, l0cOffset, 0,
                        kInner - kOuter);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aL1BufId);
                aL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(bL1BufId);
            if ((kOuter + kBL1_) % scaleKL1_ == 0 || kOuter + kBL1_ >= k_) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                scaleLoopCnt_++;
            }
            bL1LoopCnt_++;
        }
    }

    __aicore__ inline void IterAL1BL1(TileL1L0Param &tileL1L0Param, AscendC::MmadParams &mmadParams, uint64_t l0cOffset,
                                      const AscendC::GlobalTensor<AType> &aGlobal,
                                      const AscendC::GlobalTensor<BType> &bGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                                      const AscendC::GlobalTensor<BiasType> &biasGlobal)
    {
        for (uint64_t kOuter = 0; kOuter < k_; kOuter += kAL1_) {
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
            uint64_t aL1BufId = aL1LoopCnt_ & (l1BufNum_ - 1);
            CopyScalesInL1(scaleAGlobal, scaleBGlobal, tileL1L0Param, kOuter, scaleL1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
            biasBufId_ = scaleL1BufId;
            UpdateKAL1(tileL1L0Param, kOuter);
            CopyAInL1(aGlobal, tileL1L0Param, aL1BufId, kOuter);
            if (isBias_ && kOuter == 0) {
                CopyInBias(biasGlobal, biasL1Local_[l1BufferBiasOffset_[biasBufId_] / sizeof(BiasType)],
                           tileL1L0Param.curN);
            }
            for (uint64_t kInner = kOuter; kInner < Min(kOuter + kAL1_, k_); kInner += kBL1_) {
                uint64_t bL1BufId = bL1LoopCnt_ & (l1BufNum_ - 1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bL1BufId);
                UpdateKBL1(tileL1L0Param, kInner);
                CopyBInL1(bGlobal, tileL1L0Param, bL1BufId, kInner);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(bL1BufId);
                Iterate(tileL1L0Param, mmadParams, kInner, aL1BufId, bL1BufId, scaleL1BufId, l0cOffset, kInner - kOuter,
                        0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bL1BufId);
                bL1LoopCnt_++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aL1BufId);
            if ((kOuter + kAL1_) % scaleKL1_ == 0 || kOuter + kAL1_ >= k_) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + (scaleL1BufId));
                scaleLoopCnt_++;
            }
            aL1LoopCnt_++;
        }
    }

    __aicore__ inline void run(const AscendC::GlobalTensor<AType> &aGlobal, const AscendC::GlobalTensor<BType> &bGlobal,
                               const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                               const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                               const AscendC::GlobalTensor<BiasType> &biasGlobal,
                               const AscendC::GlobalTensor<CType> &cGlobal, const BlockShape &singleShape)
    {
        AscendC::MmadParams mmParams;
        uint64_t l0cOffset = 0;
        RunMmadLoop(aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, biasGlobal, singleShape, mmParams, l0cOffset);
        // Copy out to GM
        CopyOut(cGlobal, c1Local_[l0cOffset], mmParams.m, mmParams.n);
        UpdateL0cPingPong();
    }

    __aicore__ inline void run(const AscendC::GlobalTensor<AType> &aGlobal, const AscendC::GlobalTensor<BType> &bGlobal,
                               const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                               const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                               const AscendC::GlobalTensor<BiasType> &biasGlobal,
                               const AscendC::LocalTensor<CType> &cLocal, const BlockShape &singleShape)
    {
        AscendC::MmadParams mmParams;
        uint64_t l0cOffset = 0;
        RunMmadLoop(aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, biasGlobal, singleShape, mmParams, l0cOffset);
        AscendC::LocalTensor<float> c1Local = c1Local_[l0cOffset];
        CopyOut(cLocal, c1Local, mmParams.m, mmParams.n);
        UpdateL0cPingPong();
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<AType> &aGlobal,
                                      const AscendC::GlobalTensor<BType> &bGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                                      const AscendC::GlobalTensor<CType> &cGlobal, const BlockShape &singleShape)
    {
        AscendC::GlobalTensor<BiasType> biasGlobal;
        run(aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, biasGlobal, cGlobal, singleShape);
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<AType> &aGlobal,
                                      const AscendC::GlobalTensor<BType> &bGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                                      const AscendC::GlobalTensor<BiasType> &biasGlobal,
                                      const AscendC::GlobalTensor<CType> &cGlobal, const BlockShape &singleShape)
    {
        run(aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, biasGlobal, cGlobal, singleShape);
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<AType> &aGlobal,
                                      const AscendC::GlobalTensor<BType> &bGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                                      const AscendC::LocalTensor<CType> &cLocal, const BlockShape &singleShape)
    {
        AscendC::GlobalTensor<BiasType> biasGlobal;
        run(aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, biasGlobal, cLocal, singleShape);
    }

    __aicore__ inline void operator()(const AscendC::GlobalTensor<AType> &aGlobal,
                                      const AscendC::GlobalTensor<BType> &bGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                      const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                                      const AscendC::GlobalTensor<BiasType> &biasGlobal,
                                      const AscendC::LocalTensor<CType> &cLocal, const BlockShape &singleShape)
    {
        run(aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, biasGlobal, cLocal, singleShape);
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kL0Offset)
    {
        return isBias_ && kL0Offset == 0;
    }

    __aicore__ inline void RunMmadLoop(const AscendC::GlobalTensor<AType> &aGlobal,
                                       const AscendC::GlobalTensor<BType> &bGlobal,
                                       const AscendC::GlobalTensor<fp8_e8m0_t> &scaleAGlobal,
                                       const AscendC::GlobalTensor<fp8_e8m0_t> &scaleBGlobal,
                                       const AscendC::GlobalTensor<BiasType> &biasGlobal, const BlockShape &singleShape,
                                       AscendC::MmadParams &mmParams, uint64_t &l0cOffset)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = Get<IDX_M_TILE_IDX>(singleShape);
        tileL1L0Param.curN = Get<IDX_N_TILE_IDX>(singleShape);
        GetAlignMN(tileL1L0Param);
        mmParams.m = tileL1L0Param.curM;
        mmParams.n = tileL1L0Param.curN;
        mmParams.disableGemv = true;
        l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        if (orderAL1BL1_) {
            IterAL1BL1(tileL1L0Param, mmParams, l0cOffset, aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, biasGlobal);
        } else {
            IterBL1AL1(tileL1L0Param, mmParams, l0cOffset, aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, biasGlobal);
        }
    }

    __aicore__ inline void UpdateL0cPingPong()
    {
        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

    __aicore__ inline void Mmad(AscendC::MmadParams &mmadParams, uint64_t l0cOffset, uint64_t l0abOffset,
                                uint64_t biasOffset, bool needBias)
    {
        mmadParams.cmatrixSource = needBias;
        if (needBias) {
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], biasBt_[biasOffset],
                          mmadParams);
        } else {
            mmadParams.cmatrixSource = false;
            AscendC::Mmad(c1Local_[l0cOffset], l0aLocal_[l0abOffset], l0bLocal_[l0abOffset], mmadParams);
        }
    }

private:
    uint16_t biasBufId_ = 0;
    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[2] = {0UL};      // default 2 buffer
    uint64_t l1BufferBOffset_[2] = {0UL};      // default 2 buffer
    uint64_t l1BufferScaleAOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferScaleBOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL};   // default 2 buffer
    uint64_t m_;
    uint64_t n_;
    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};
    uint64_t minKL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t mxScaleKL1B16_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t aL1LoopCnt_{0};
    uint64_t bL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool orderAL1BL1_{false};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    AscendC::LocalTensor<MxL0AType> l0aLocal_{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<MxL0BType> l0bLocal_{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> c1Local_{AscendC::TPosition::CO1, 0, AscendC::TOTAL_L0C_SIZE};
    AscendC::LocalTensor<float> biasBt_{AscendC::TPosition::C2, 0, 4096}; // 4096B: BT_SIZE
    AscendC::LocalTensor<AType> aL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BType> bL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<BiasType> biasL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE / sizeof(BiasType)};
    AscendC::LocalTensor<fp8_e8m0_t> scaleAL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
    AscendC::LocalTensor<fp8_e8m0_t> scaleBL1Local_{AscendC::TPosition::A1, 0, AscendC::TOTAL_L1_SIZE};
};
} // namespace Block
} // namespace Gemm
} // namespace Cgmct
#endif
