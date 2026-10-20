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
 * \file block_mx_low_level_mm_aic_to_aiv_builder.h
 * \brief MX low-level mmad AIC->AIV builder, uses BlockMmadMx + BlockMmadMxToUbAdapter.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../utils/cgmct_common_utils.h"
#include "../utils/cgmct_layout_utils.h"
#include "../utils/status_utils.h"
#include "../utils/tuple_utils.h"

#include "../policy/cgmct_dispatch_policy.h"
#include "block_mmad_mx.h"

namespace Cgmct {
namespace Gemm {
namespace Block {
namespace {
constexpr uint64_t MX_SCALE_FACTOR_BIT = 8UL;
constexpr uint64_t MX_SCALE_FACTOR_MASK = 0xFFUL;
constexpr uint64_t MX_L1_BUFFER_NUM = 2UL;
constexpr uint64_t K_DIM = 2UL;
} // namespace

template <class BlockMmadMxOp_>
class BlockMmadMxToUbAdapter {
public:
    using BlockMmadMxOp = BlockMmadMxOp_;
    using AType = typename BlockMmadMxOp::AType;
    using BType = typename BlockMmadMxOp::BType;
    using CType = typename BlockMmadMxOp::CType;
    using BiasType = typename BlockMmadMxOp::BiasType;
    using ScaleAType = typename BlockMmadMxOp::ScaleAType;
    using ScaleBType = typename BlockMmadMxOp::ScaleBType;
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t>;

    __aicore__ inline void Init(TCubeTiling* __restrict matmulTiling, AscendC::TPipe* tpipe = nullptr)
    {
        (void)tpipe;
        uint64_t scaleFactorA = matmulTiling->mxTypePara & MX_SCALE_FACTOR_MASK;
        uint64_t scaleFactorB = (matmulTiling->mxTypePara >> MX_SCALE_FACTOR_BIT) & MX_SCALE_FACTOR_MASK;
        uint64_t kAL1 = static_cast<uint64_t>(matmulTiling->stepKa) * static_cast<uint64_t>(matmulTiling->baseK);
        uint64_t kBL1 = static_cast<uint64_t>(matmulTiling->stepKb) * static_cast<uint64_t>(matmulTiling->baseK);
        uint64_t scaleKL1 = Cgmct::Gemm::Min(Cgmct::Gemm::Max(scaleFactorA * kAL1, scaleFactorB * kBL1),
                                             static_cast<uint64_t>(matmulTiling->Ka));
        typename BlockMmadMxOp::L1Params l1Params{kAL1, kBL1, scaleKL1, MX_L1_BUFFER_NUM};
        AscendC::Shape<int64_t, int64_t, int64_t> problemShape{matmulTiling->M, matmulTiling->N, matmulTiling->Ka};
        AscendC::Shape<int64_t, int64_t, int64_t> l0TileShape{matmulTiling->baseM, matmulTiling->baseN,
                                                              matmulTiling->baseK};
        if ASCEND_IS_AIC {
            mmadOp_.Init(problemShape, l0TileShape, l1Params, matmulTiling->isBias != 0, matmulTiling->dbL0C != 1);
        }
    }

    __aicore__ inline void UpdateParamsForNextProblem(const TupleShape& problemShape)
    {
        mmadOp_.UpdateParamsForNextProblem(problemShape);
    }

    template <typename SingleShape>
    __aicore__ inline void operator()(const AscendC::GlobalTensor<AType>& aGlobal,
                                      const AscendC::GlobalTensor<BType>& bGlobal,
                                      const AscendC::GlobalTensor<ScaleAType>& scaleAGlobal,
                                      const AscendC::GlobalTensor<ScaleBType>& scaleBGlobal,
                                      const AscendC::LocalTensor<CType>& cLocal, const SingleShape& singleShape,
                                      bool transA, bool transB)
    {
        (void)transA;
        (void)transB;
        AscendC::Shape<int64_t, int64_t, int64_t> blockShape{Get<0>(singleShape), Get<1>(singleShape),
                                                             Get<K_DIM>(singleShape)};
        mmadOp_(aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, cLocal, blockShape);
    }

    template <typename SingleShape>
    __aicore__ inline void operator()(const AscendC::GlobalTensor<AType>& aGlobal,
                                      const AscendC::GlobalTensor<BType>& bGlobal,
                                      const AscendC::GlobalTensor<ScaleAType>& scaleAGlobal,
                                      const AscendC::GlobalTensor<ScaleBType>& scaleBGlobal,
                                      const AscendC::GlobalTensor<BiasType>& biasGlobal,
                                      const AscendC::LocalTensor<CType>& cLocal, const SingleShape& singleShape,
                                      bool transA, bool transB)
    {
        (void)transA;
        (void)transB;
        AscendC::Shape<int64_t, int64_t, int64_t> blockShape{Get<0>(singleShape), Get<1>(singleShape),
                                                             Get<K_DIM>(singleShape)};
        mmadOp_(aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, biasGlobal, cLocal, blockShape);
    }

    template <typename SingleShape>
    __aicore__ inline void operator()(const AscendC::GlobalTensor<AType>& aGlobal,
                                      const AscendC::GlobalTensor<BType>& bGlobal,
                                      const AscendC::GlobalTensor<ScaleAType>& scaleAGlobal,
                                      const AscendC::GlobalTensor<ScaleBType>& scaleBGlobal,
                                      const AscendC::GlobalTensor<CType>& cGlobal, const SingleShape& singleShape,
                                      bool transA, bool transB)
    {
        (void)transA;
        (void)transB;
        AscendC::Shape<int64_t, int64_t, int64_t> blockShape{Get<0>(singleShape), Get<1>(singleShape),
                                                             Get<K_DIM>(singleShape)};
        mmadOp_(aGlobal, bGlobal, scaleAGlobal, scaleBGlobal, cGlobal, blockShape);
    }

private:
    BlockMmadMxOp mmadOp_;
};

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class BiasType_, class CType_, class LayoutC_,
          class L1TileShape_, class L0TileShape_, class BlockScheduler_, class BlockMatmulPolicy_ = MatmulWithScale<>,
          class TileCopyParam_ = void, typename Enable_ = void>
class BlockMxLowLevelMmAicToAivBuilder {
    static_assert(AscendC::Std::always_false_v<BlockMatmulPolicy_>,
                  "BlockMxLowLevelMmAicToAivBuilder is not implemented for this BlockMatmulPolicy");
};

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class BiasType_, class CType_, class LayoutC_,
          class L1TileShape_, class L0TileShape_, class BlockScheduler_, class BlockMatmulPolicy_, class TileCopyParam_>
class BlockMxLowLevelMmAicToAivBuilder<
    AType_, LayoutA_, BType_, LayoutB_, BiasType_, CType_, LayoutC_, L1TileShape_, L0TileShape_, BlockScheduler_,
    BlockMatmulPolicy_, TileCopyParam_,
    AscendC::Std::enable_if_t<AscendC::Std::is_base_of_v<MatmulWithScale<>, BlockMatmulPolicy_>>> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BlockMatmulPolicy = BlockMatmulPolicy_;
    using TileCopyParam = TileCopyParam_;

    static constexpr bool transA = TagToTrans<LayoutA>::value;
    static constexpr bool transB = TagToTrans<LayoutB>::value;
    static constexpr CubeFormat formatA = TagToFormat<LayoutA>::format;
    static constexpr CubeFormat formatB = TagToFormat<LayoutB>::format;
    static constexpr CubeFormat formatC = TagToFormat<LayoutC>::format;

    using BlockMmadMxOp = Block::BlockMmadMx<BlockMatmulPolicy, L1TileShape, L0TileShape, AType, LayoutA, BType,
                                             LayoutB, CType, LayoutC, BiasType, layout::RowMajor, TileCopyParam>;
    using BlockMmadOp = Block::BlockMmadMxToUbAdapter<BlockMmadMxOp>;

    static constexpr int64_t l1M = GetIntegralConstant<MNK_M, L1TileShape>();
    static constexpr int64_t l1N = GetIntegralConstant<MNK_N, L1TileShape>();
    static constexpr int64_t l1K = GetIntegralConstant<MNK_K, L1TileShape>();

    struct Arguments {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR groupListGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
    };

    using Params = Arguments;
};
} // namespace Block
} // namespace Gemm
} // namespace Cgmct
