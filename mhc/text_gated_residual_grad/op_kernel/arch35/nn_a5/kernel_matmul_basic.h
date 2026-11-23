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
 * \file kernel_matmul_basic.h
 * \brief
 */

#pragma once

#include "kernel_basic_intf.h"

#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_mmad_matmul_al1_full_load.h"
#include "kernel_universal_decl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

template <class ProblemShape_, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class HcGradNnGemm<ProblemShape_, BlockMmad_, BlockEpilogue_, BlockScheduler_,
                   AscendC::Std::enable_if_t<AscendC::Std::is_same_v<
                       KernelMmadMultiBlockBasic, typename BlockMmad_::DispatchPolicy::ScheduleType>>> {
public:
    __aicore__ inline HcGradNnGemm() {}
    __aicore__ inline ~HcGradNnGemm() {}

    using BlockMmad = BlockMmad_;
    using ProblemShape = ProblemShape_;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue = BlockEpilogue_;

    // mmad
    using BlockMmadParams = typename BlockMmad::Params;
    using BlockEpilogueParams = typename BlockEpilogue::Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<asc::te::c0_element<BType>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;
    using MakeLayoutBias = asc::te::frame_layout_format<LayoutBias, AscendC::Std::Int<asc::te::c0_element<BiasType>>>;
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockEpilogueParams epilogueParams;
        BlockSchedulerParams schParams;
        uint32_t fusionLda = 0;
        Params() = default;
    };

    __aicore__ inline void operator()(Params& params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        Init(params);

        // Initialize the block scheduler.
        BlockScheduler bs(params.problemShape, params.schParams);
        int64_t curBlockIdx = AscendC::GetBlockIdx();
        int64_t realCoreNums = bs.GetCoreNums(); // Actual number of cores required.
        if (curBlockIdx >= realCoreNums) {
            return;
        }

        Blaze::Gemm::SetHF32(params.schParams.isHf32);

        BlockMmad blockMmad;
        if constexpr (AscendC::Std::is_same_v<
                          MatmulMultiBlockBasicSplitK<0, 1, Blaze::Gemm::KernelMmadMultiBlockBasic, 0>,
                          typename BlockMmad_::DispatchPolicy>) {
            params.mmadParams.k = k_;
        }
        blockMmad.Init(params.mmadParams);

        if constexpr (NON_CONTIGUOUS_TYPE == static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_SLICE)) {
            MatmulSliceProcess(params, blockMmad, bs, curBlockIdx, params.schParams.usedCoreNum, bs.GetBlockNums());
        } else {
            MatmulProcess(params, blockMmad, bs, curBlockIdx, params.schParams.usedCoreNum, bs.GetBlockNums());
        }

        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

    template <uint32_t BLOCKED_B_ROWS, bool BYPASS_OUTPUT_L2>
    __aicore__ inline void RunTraffic(Params& params)
    {
        constexpr bool B16_TO_FLOAT =
            (AscendC::Std::is_same_v<AType, bfloat16_t> || AscendC::Std::is_same_v<AType, half>) &&
            AscendC::Std::is_same_v<CType, float>;
        constexpr bool HALF_DX = AscendC::Std::is_same_v<AType, half> &&
                                 (AscendC::Std::is_same_v<CType, half> || AscendC::Std::is_same_v<CType, float>);
        static_assert(B16_TO_FLOAT || (BYPASS_OUTPUT_L2 && HALF_DX));
        static_assert((BLOCKED_B_ROWS != 0U) != BYPASS_OUTPUT_L2);
        if ASCEND_IS_AIV {
            return;
        }
        Init(params);
        BlockScheduler bs(params.problemShape, params.schParams);
        const int64_t core = AscendC::GetBlockIdx();
        if (core >= bs.GetCoreNums()) {
            return;
        }
        Blaze::Gemm::SetHF32(params.schParams.isHf32);
        BlockMmad mm;
        mm.Init(params.mmadParams);
        MatmulProcessTraffic<BLOCKED_B_ROWS, BYPASS_OUTPUT_L2>(params, mm, bs, core, params.schParams.usedCoreNum,
                                                               bs.GetBlockNums());
        Blaze::Gemm::UnsetHF32(params.schParams.isHf32);
    }

private:
    // Keep the canonical Basic path and Params intact. Neither experimental
    // transport is instantiated by a route without its corresponding layout or hint.
    template <uint32_t BLOCKED_B_ROWS, bool BYPASS_OUTPUT_L2>
    __aicore__ inline void MatmulProcessTraffic(const Params& params, BlockMmad& mm, BlockScheduler& bs, int64_t core,
                                                int64_t cores, int64_t blocks)
    {
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), MakeLayoutAGm(params));
        auto gmB =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), MakeLayoutB{}(batch_, k_, n_));
        auto gmC =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), MakeLayoutC{}(batch_, m_, n_));
        auto gmBias =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), MakeLayoutBias{}(1L, n_));
        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);
        for (int64_t blockIdx = core; blockIdx < blocks; blockIdx += cores) {
            auto shape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx);
            const auto pos = bs.GetBlockCoord(blockIdx);
            const auto m = asc::te::get<MNK_M>(pos);
            const auto n = asc::te::get<MNK_N>(pos);
            const auto mTile = asc::te::get<MNK_M>(shape);
            const auto nTile = asc::te::get<MNK_N>(shape);
            const auto kTile = asc::te::get<MNK_K>(shape);
            curBatchIdx_ = static_cast<uint64_t>(asc::te::get<MNK_B>(pos));
            auto subTensorA = gmA.slice(asc::te::make_coord(curBatchIdx_, asc::te::make_coord(m, 0L)),
                                        asc::te::make_shape(1L, asc::te::make_shape(mTile, kTile)));
            auto a = asc::te::squeeze<0>(subTensorA);
            auto subTensorC = gmC.slice(asc::te::make_coord(curBatchIdx_, asc::te::make_coord(m, n)),
                                        asc::te::make_shape(1L, asc::te::make_shape(mTile, nTile)));
            auto c = asc::te::squeeze<0>(subTensorC);
            auto bias = gmBias.slice(asc::te::make_coord(0L, n), asc::te::make_shape(1L, nTile));
            if constexpr (BYPASS_OUTPUT_L2) {
                c.set_l2_cache_hint(asc::te::cache_mode::disable);
            }
            if constexpr (BLOCKED_B_ROWS != 0U) {
                // Fixed panel pitch in GM, tight row stride nTile within an
                // individual panel, including the 64-column last panel.
                auto b = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_ + static_cast<uint64_t>(n) * BLOCKED_B_ROWS),
                    MakeLayoutB{}(kTile, nTile));
                // The full-M G56 schedule consumes each xn column panel once.
                b.set_l2_cache_hint(asc::te::cache_mode::disable);
                mm(a, b, bias, c, shape);
            } else {
                auto subTensorB = gmB.slice(asc::te::make_coord(curBatchIdx_, asc::te::make_coord(0L, n)),
                                            asc::te::make_shape(1L, asc::te::make_shape(kTile, nTile)));
                auto b = asc::te::squeeze<0>(subTensorB);
                mm(a, b, bias, c, shape);
            }
        }
    }

    __aicore__ inline auto MakeLayoutAGm(Params const& params)
    {
        uint64_t lda = params.fusionLda;
        if constexpr (AscendC::Std::is_same_v<LayoutA, asc::te::zn_layout_ptn>) {
            using PreparedALayout =
                asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<asc::te::c0_element<AType>>>;
            return PreparedALayout{}(batch_, m_, k_);
        } else if constexpr (BlockMmad::TRANS_A) {
            return asc::te::make_pattern_layout<LayoutA, asc::te::layout_trait_default<>>(
                asc::te::make_shape(uint64_t(batch_),
                                    asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, uint64_t(m_)),
                                                        asc::te::make_shape(asc::te::_1{}, uint64_t(k_)))),
                asc::te::make_stride(uint64_t(k_) * lda,
                                     asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, asc::te::_1{}),
                                                          asc::te::make_stride(asc::te::_0{}, lda))));
        } else {
            return asc::te::make_pattern_layout<LayoutA, asc::te::layout_trait_default<>>(
                asc::te::make_shape(uint64_t(batch_),
                                    asc::te::make_shape(asc::te::make_shape(asc::te::_1{}, uint64_t(m_)),
                                                        asc::te::make_shape(asc::te::_1{}, uint64_t(k_)))),
                asc::te::make_stride(uint64_t(m_) * lda,
                                     asc::te::make_stride(asc::te::make_stride(asc::te::_0{}, lda),
                                                          asc::te::make_stride(asc::te::_0{}, asc::te::_1{}))));
        }
    }

    __aicore__ inline void MatmulProcess(Params const& params, BlockMmad& blockMmad, BlockScheduler& bs,
                                         int64_t curBlockIdx, int64_t coreNums, int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutAGm(params);
        auto layoutB = MakeLayoutB{}(batch_, k_, n_); // ND layout for B
        auto layoutC = MakeLayoutC{}(batch_, m_, n_); // ND layout for C
        auto layoutBias = MakeLayoutBias{}(1L, n_);   // ND layout for Bias
        // A,B,C Gm Tensor
        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);

        // Enable dual page tables.
        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);

        // Process tiles in ping-pong mode
        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx); // (m, n, k, b)
            auto blockCoord = bs.GetBlockCoord(blockIdx);                          // (m, n, k, b)
            auto coordM = asc::te::get<MNK_M>(blockCoord);
            auto coordN = asc::te::get<MNK_N>(blockCoord);
            auto shapeM = asc::te::get<MNK_M>(blockShape);
            auto shapeN = asc::te::get<MNK_N>(blockShape);
            auto shapeK = asc::te::get<MNK_K>(blockShape);
            curBatchIdx_ = static_cast<uint64_t>(asc::te::get<MNK_B>(blockCoord));

            // Block offset
            // A full-load shares one complete A matrix across batches; other modes select the current A tile.
            auto batchIdxA = IS_A_FULL_LOAD ? 0L : curBatchIdx_;
            auto coordMA = IS_A_FULL_LOAD ? 0L : coordM;
            auto shapeMA = IS_A_FULL_LOAD ? m_ : shapeM;
            auto subTensorA = gmA.slice(asc::te::make_coord(batchIdxA, asc::te::make_coord(coordMA, 0L)),
                                        asc::te::make_shape(1L, asc::te::make_shape(shapeMA, shapeK)));
            auto gmBlockA = asc::te::squeeze<0>(subTensorA);
            auto subTensorB = gmB.slice(asc::te::make_coord(curBatchIdx_, asc::te::make_coord(0L, coordN)),
                                        asc::te::make_shape(1L, asc::te::make_shape(shapeK, shapeN)));
            auto gmBlockB = asc::te::squeeze<0>(subTensorB);
            auto subTensorC = gmC.slice(asc::te::make_coord(curBatchIdx_, asc::te::make_coord(coordM, coordN)),
                                        asc::te::make_shape(1L, asc::te::make_shape(shapeM, shapeN)));
            auto gmBlockC = asc::te::squeeze<0>(subTensorC);
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, coordN), asc::te::make_shape(1L, shapeN));
            blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape);
        }
    }

    __aicore__ inline auto MakeLayoutA3DForSlice(Params const& params)
    {
        int64_t sliceM = static_cast<int64_t>(params.schParams.sliceM);
        int64_t sliceBatch = static_cast<int64_t>(m_) / sliceM;
        int64_t srcNdStride = static_cast<int64_t>(params.schParams.srcNdStride);

        return asc::te::make_pattern_layout<
            NDSliceLayoutPtn, asc::te::layout_trait<AType, AscendC::Std::Int<asc::te::c0_element<AType>>>>(
            asc::te::make_shape(sliceBatch, asc::te::make_shape(sliceM, k_)),
            asc::te::make_stride(srcNdStride, asc::te::make_stride(k_, 1L)));
    }

    __aicore__ inline void MatmulSliceProcess(Params const& params, BlockMmad& blockMmad, BlockScheduler& bs,
                                              int64_t curBlockIdx, int64_t coreNums, int64_t totalBlockNums)
    {
        auto layoutA = MakeLayoutA3DForSlice(params);
        auto layoutB = MakeLayoutB{}(k_, n_);
        auto layoutC = MakeLayoutC{}(m_, n_);
        auto layoutBias = MakeLayoutBias{}(1L, n_);

        auto gmA = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(aGmAddr_), layoutA);
        auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), layoutB);
        auto gmC = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(cGmAddr_), layoutC);
        auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_), layoutBias);
        SetL2Cache(gmA, gmB, params.schParams.l2CacheMode);

        for (int64_t blockIdx = curBlockIdx; blockIdx < totalBlockNums; blockIdx += coreNums) {
            auto blockShape = bs.template GetBlockShape<TRANS_B, BType>(blockIdx);
            auto blockCoord = bs.GetBlockCoord(blockIdx);
            auto coordM = asc::te::get<MNK_M>(blockCoord);
            auto coordN = asc::te::get<MNK_N>(blockCoord);
            auto shapeM = asc::te::get<MNK_M>(blockShape);
            auto shapeN = asc::te::get<MNK_N>(blockShape);
            auto shapeK = asc::te::get<MNK_K>(blockShape);

            int64_t sliceM = static_cast<int64_t>(params.schParams.sliceM);
            auto gmBlockA = gmA.slice(asc::te::make_coord(coordM / sliceM, asc::te::make_coord(0L, 0L)),
                                      asc::te::make_shape(shapeM / sliceM, asc::te::make_shape(sliceM, shapeK)));
            auto gmBlockB = gmB.slice(asc::te::make_coord(0L, coordN), asc::te::make_shape(shapeK, shapeN));
            auto gmBlockC = gmC.slice(asc::te::make_coord(coordM, coordN), asc::te::make_shape(shapeM, shapeN));
            auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, coordN), asc::te::make_shape(1L, shapeN));
            blockMmad(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, blockShape);
        }
    }

    __aicore__ inline void Init(Params const& params)
    {
        auto blockMmadParams = params.mmadParams;
        m_ = static_cast<uint64_t>(asc::te::get<MNK_M>(params.problemShape));
        n_ = static_cast<uint64_t>(asc::te::get<MNK_N>(params.problemShape));
        k_ = static_cast<uint64_t>(asc::te::get<MNK_K>(params.problemShape));
        batch_ = static_cast<uint64_t>(AscendC::Std::max(asc::te::get<MNK_B>(params.problemShape), 1L));
        aGmAddr_ = reinterpret_cast<__gm__ AType*>(blockMmadParams.aGmAddr);
        bGmAddr_ = reinterpret_cast<__gm__ BType*>(blockMmadParams.bGmAddr);
        cGmAddr_ = reinterpret_cast<__gm__ CType*>(blockMmadParams.cGmAddr);
        biasGmAddr_ = reinterpret_cast<__gm__ BiasType*>(blockMmadParams.biasGmAddr);
    }

    template <typename TensorA, typename TensorB>
    __aicore__ inline void SetL2Cache(TensorA& gmA, TensorB& gmB, uint32_t l2CacheMode)
    {
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == B_L2_CACHE_DISABLE) {
            gmB.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
        if (l2CacheMode == ALL_L2_CACHE_DISABLE || l2CacheMode == A_L2_CACHE_DISABLE) {
            gmA.set_l2_cache_hint(asc::te::cache_mode::disable);
        }
    }

private:
    static constexpr bool TRANS_A = BlockMmad::TRANS_A;
    static constexpr bool TRANS_B = BlockMmad::TRANS_B;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = BlockMmad::NON_CONTIGUOUS_TYPE;
    static constexpr bool IS_A_FULL_LOAD = BlockMmad::DispatchPolicy::FULL_LOAD_MODE == A_FULL_LOAD_MODE;
    __gm__ AType* aGmAddr_;
    __gm__ BType* bGmAddr_;
    __gm__ CType* cGmAddr_;
    __gm__ BiasType* biasGmAddr_ = nullptr; // Initialize the optional bias input here.

    uint64_t curBatchIdx_ = {0};
    uint64_t m_{1};
    uint64_t n_{1};
    uint64_t k_{1};
    uint64_t batch_{1};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
