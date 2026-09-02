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
 * \file matmul.hpp
 * \brief
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_ALLTOALL_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_ALLTOALL_HPP

#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_matrix_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_cross_core_sync.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_gm_to_ub.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_ub_to_gm.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/tla_gemm_kernel_padding_matmul.hpp"

namespace Catlass::Gemm::Kernel {
template <class PrologueA, class PrologueB, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, bool HasBias>
class MatmulAlltoAllKernel : public CommBase {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template <bool condition, class mmad>
    struct BiasTypeHelper {
        using type = typename mmad::ElementBias;
    };

    template <class mmad>
    struct BiasTypeHelper<false, mmad> {
        using type = float;
    };

    template <class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template <>
    struct LayoutHelper<void> {
        using type = void;
    };

    using ElementBias = typename BiasTypeHelper<HasBias, BlockMmad>::type;
    using LayoutA = std::conditional_t<std::is_void_v<PrologueA>, LayoutWA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PrologueB>, LayoutWB, typename LayoutHelper<PrologueB>::type>;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrBias;
        GM_ADDR ptrC;
        LayoutC layoutC;
        int32_t pValue;
        int32_t rankSize;
        int32_t pipeDepth;
        bool transB;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrBias_, GM_ADDR ptrC_, LayoutC layoutC_, int32_t pValue_, int32_t rankSize_,
               int32_t pipeDepth_, bool transB_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrBias(ptrBias_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              pValue(pValue_),
              rankSize(rankSize_),
              pipeDepth(pipeDepth_),
              transB(transB_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    MatmulAlltoAllKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementBias> gmBias;
        if constexpr (HasBias) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }

        BlockMmad blockMmad(resource);

        int32_t coreIdx = AscendC::GetBlockIdx();
        int32_t coreNum = AscendC::GetBlockNum();

        int32_t mLoops = (params.problemShape.m() + L1TileShape::M - 1) / L1TileShape::M;

        int32_t commCount = (mLoops + params.pValue - 1) / params.pValue;
        for (int32_t commIdx = 0; commIdx < commCount; commIdx++) {
            uint64_t flagIdx = commIdx % params.pipeDepth;

            int32_t tailN = (params.problemShape.n() / params.rankSize) % L1TileShape::N;
            uint32_t nLoops = params.problemShape.n() / params.rankSize / L1TileShape::N;
            if (tailN) {
                nLoops += 1;
            }
            nLoops *= params.rankSize;

            int32_t coreLoops = mLoops * nLoops;

            uint32_t actualPValue = params.pValue;
            if (commIdx == commCount - 1) {
                actualPValue = mLoops - commIdx * params.pValue;
            }
            if (commIdx >= params.pipeDepth) {
                WaitEvent(flagIdx);
            }
            int64_t gmAPingpongSize = static_cast<int64_t>(L1TileShape::M) * params.pValue * params.problemShape.n();

            int32_t actualLoopNum = actualPValue * nLoops;

            int32_t nLoopPerRank = nLoops / params.rankSize;
            GemmCoord calSwizzleShape{actualPValue, nLoops, 1};
            BlockScheduler matmulBlockScheduler(calSwizzleShape,
                                                MakeCoord(static_cast<uint32_t>(1), static_cast<uint32_t>(1)));

            for (int32_t loopOffset = 0; loopOffset < actualLoopNum; loopOffset++) {
                int32_t loopIdx = commIdx * params.pValue * nLoops + loopOffset;
                if (loopIdx % coreNum != coreIdx) {
                    continue;
                }

                GemmCoord blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopOffset);
                uint32_t mActual = (commIdx == commCount - 1) && (blockIdxCoord.m() == (mLoops - 1) % params.pValue) &&
                                           (params.problemShape.m() % L1TileShape::M != 0) ?
                                       (params.problemShape.m() - commIdx * params.pValue * L1TileShape::M -
                                        blockIdxCoord.m() * L1TileShape::M) :
                                       L1TileShape::M;
                uint32_t nActual = (blockIdxCoord.n() % nLoopPerRank == (nLoopPerRank - 1)) ?
                                       (params.problemShape.n() / params.rankSize -
                                        blockIdxCoord.n() % nLoopPerRank * L1TileShape::N) :
                                       L1TileShape::N;
                GemmCoord actualBlockShape{mActual, nActual, params.problemShape.k()};

                MatrixCoord offsetA{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.k() * L1TileShape::K};
                MatrixCoord offsetC{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = params.layoutA.GetOffset(offsetA) + static_cast<int64_t>(commIdx) * L1TileShape::M *
                                                                            params.pValue * params.problemShape.k();
                int64_t gmOffsetB = (static_cast<int64_t>(blockIdxCoord.n() / nLoopPerRank) *
                                         (params.problemShape.n() / params.rankSize) +
                                     static_cast<int64_t>(blockIdxCoord.n() % nLoopPerRank) * L1TileShape::N) *
                                    (params.transB ? params.problemShape.k() : 1);

                int64_t rankOffset = gmAPingpongSize / params.rankSize;
                int64_t rankIdx = blockIdxCoord.n() / nLoopPerRank;
                int64_t rankOffsetInRank = blockIdxCoord.n() % nLoopPerRank;
                int64_t gmOffsetC =
                    static_cast<int64_t>(flagIdx) * L1TileShape::M * params.pValue * params.problemShape.n() +
                    rankIdx * rankOffset +
                    static_cast<int64_t>(blockIdxCoord.m()) * L1TileShape::M *
                        (params.problemShape.n() / params.rankSize) +
                    blockIdxCoord.n() % nLoopPerRank * L1TileShape::N;

                if constexpr (HasBias) {
                    int64_t gmOffsetBias =
                        (blockIdxCoord.n() / nLoopPerRank) * (params.problemShape.n() / params.rankSize) +
                        (blockIdxCoord.n() % nLoopPerRank) * L1TileShape::N;

                    blockMmad(gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC],
                              params.layoutC, gmBias[gmOffsetBias], actualBlockShape);
                } else {
                    bool isFirstBlock = (loopOffset < coreNum);
                    bool hasNextBlock = false;
                    GemmCoord nextBlockIdCoord;
                    GemmCoord nextActualBlockShape;
                    if (loopOffset + coreNum < actualLoopNum) {
                        hasNextBlock = true;
                        nextBlockIdCoord = matmulBlockScheduler.GetBlockCoord(loopOffset + coreNum);

                        uint32_t mActualNext = (commIdx == commCount - 1) &&
                                                       (nextBlockIdCoord.m() == (mLoops - 1) % params.pValue) &&
                                                       (params.problemShape.m() % L1TileShape::M != 0) ?
                                                   (params.problemShape.m() - commIdx * params.pValue * L1TileShape::M -
                                                    nextBlockIdCoord.m() * L1TileShape::M) :
                                                   L1TileShape::M;
                        uint32_t nActualNext = (nextBlockIdCoord.n() % nLoopPerRank == (nLoopPerRank - 1)) ?
                                                   (params.problemShape.n() / params.rankSize -
                                                    nextBlockIdCoord.n() % nLoopPerRank * L1TileShape::N) :
                                                   L1TileShape::N;
                        nextActualBlockShape = GemmCoord{mActualNext, nActualNext, params.problemShape.k()};
                    }
                    MatrixCoord offsetNextA{nextBlockIdCoord.m() * L1TileShape::M,
                                            nextBlockIdCoord.k() * L1TileShape::K};
                    MatrixCoord offsetNextB{nextBlockIdCoord.k() * L1TileShape::K,
                                            nextBlockIdCoord.n() * L1TileShape::N};
                    int64_t gmOffsetNextA = params.layoutA.GetOffset(offsetNextA) + static_cast<int64_t>(commIdx) *
                                                                                        L1TileShape::M * params.pValue *
                                                                                        params.problemShape.k();
                    int64_t gmOffsetNextB =
                        (static_cast<int64_t>(nextBlockIdCoord.n() / nLoopPerRank) *
                             (params.problemShape.n() / params.rankSize) +
                         static_cast<int64_t>(nextBlockIdCoord.n() % nLoopPerRank) * L1TileShape::N) *
                        (params.transB ? params.problemShape.k() : 1);

                    blockMmad(gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC],
                              params.layoutC, gmA[gmOffsetNextA], gmB[gmOffsetNextB], actualBlockShape,
                              nextActualBlockShape, isFirstBlock, hasNextBlock);
                }
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(flagIdx);
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_ALLTO_ALL_HPP
