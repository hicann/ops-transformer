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
 * \file allto_all_matmul.hpp
 * \brief
 */

#ifndef CATLASS_GEMM_KERNEL_ALLTOALL_ALLTO_ALL_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_ALLTOALL_ALLTO_ALL_MATMUL_HPP

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
class AlltoAllMatmulKernel {
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
        int32_t swizzlCount;
        int32_t swizzlDirect;
        int32_t rankSize; // 待分析是否删除
        int32_t pipeDepth;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrBias_, GM_ADDR ptrC_, LayoutC layoutC_, int32_t pValue_, int32_t swizzlCount_,
               int32_t swizzlDirect_, int32_t rankSize_, int32_t pipeDepth_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrBias(ptrBias_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              pValue(pValue_),
              swizzlCount(swizzlCount_),
              swizzlDirect(swizzlDirect_),
              rankSize(rankSize_),
              pipeDepth(pipeDepth_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    AlltoAllMatmulKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    inline __aicore__ void GetBlockIdx(int32_t loopIdx, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect,
                                       int32_t swizzlCount, int64_t &mIdx, int64_t &nIdx)
    {
        uint32_t inBatchIdx = loopIdx % (mLoop * nLoop);
        if (swizzlDirect == 0) {
            uint32_t tileBlockLoop = (mLoop + swizzlCount - 1) / swizzlCount;
            uint32_t tileBlockIdx = inBatchIdx / (swizzlCount * nLoop);
            uint32_t inTileBlockIdx = inBatchIdx % (swizzlCount * nLoop);
            uint32_t nRow = swizzlCount;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = mLoop - swizzlCount * tileBlockIdx;
            }
            mIdx = tileBlockIdx * swizzlCount + inTileBlockIdx % nRow;
            nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 != 0) {
                nIdx = nLoop - nIdx - 1;
            }
        } else if (swizzlDirect == 1) {
            uint32_t tileBlockLoop = (nLoop + swizzlCount - 1) / swizzlCount;
            uint32_t tileBlockIdx = inBatchIdx / (swizzlCount * mLoop);
            uint32_t inTileBlockIdx = inBatchIdx % (swizzlCount * mLoop);

            uint32_t nCol = swizzlCount;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = nLoop - swizzlCount * tileBlockIdx;
            }
            nIdx = tileBlockIdx * swizzlCount + inTileBlockIdx % nCol;
            mIdx = inTileBlockIdx / nCol;
            if (tileBlockIdx % 2 != 0) {
                mIdx = mLoop - mIdx - 1;
            }
        }
    }

    inline __aicore__ GemmCoord GetBlockIdCoord(int32_t loopOffset, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect,
                                                int32_t swizzlCount)
    {
        uint32_t kIdx = 0;
        int64_t mIdx, nIdx;
        GetBlockIdx(loopOffset, mLoop, nLoop, swizzlDirect, swizzlCount, mIdx, nIdx);
        return GemmCoord(static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), kIdx);
    }

    inline __aicore__ GemmCoord GetBlockLocCoord(GemmCoord blockIdxCoord)
    {
        return GemmCoord(blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N,
                         blockIdxCoord.k() * L1TileShape::K);
    }

    inline __aicore__ GemmCoord GetBlockSizeCoord(GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop,
                                                  int32_t mSize, int32_t nLoop, int32_t nSize, int32_t kSize)
    {
        uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : L1TileShape::M;
        uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : L1TileShape::N;
        uint32_t kActual = kSize;
        return GemmCoord(mActual, nActual, kActual);
    }

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
        uint32_t nLoops = (params.problemShape.n() + L1TileShape::N - 1) / L1TileShape::N;
        int64_t pingPongBlockSize = static_cast<int64_t>(L1TileShape::M) * params.pValue * params.problemShape.k();

        int32_t commCount = (mLoops + params.pValue - 1) / params.pValue;

        for (int32_t commIdx = 0; commIdx < commCount; commIdx++) {
            uint64_t flagIdx = commIdx % params.pipeDepth;
            int64_t peerMemBlockSt = static_cast<int64_t>(flagIdx) * pingPongBlockSize;

            uint32_t actualPValue = params.pValue;
            int32_t mLoopStart = commIdx * params.pValue;
            int32_t blockMSize = actualPValue * L1TileShape::M;

            if (commIdx == commCount - 1) {
                actualPValue = mLoops - commIdx * params.pValue;
                blockMSize = params.problemShape.m() - mLoopStart * L1TileShape::M;
            }

            WaitEvent(flagIdx);

            int32_t actualLoopNum = actualPValue * nLoops;

            for (int32_t loopOffset = 0; loopOffset < actualLoopNum; loopOffset++) {
                int32_t loopIdx = mLoopStart * nLoops + loopOffset;

                if (loopIdx % coreNum != coreIdx) {
                    continue;
                }

                GemmCoord blockIdxCoord =
                    GetBlockIdCoord(loopOffset, actualPValue, nLoops, params.swizzlDirect, params.swizzlCount);
                GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
                GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, actualPValue, blockMSize,
                                                             nLoops, params.problemShape.n(), params.problemShape.k());

                MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
                MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
                MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};

                int64_t gmOffsetA = params.layoutA.GetOffset(offsetA) + peerMemBlockSt;
                int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = static_cast<int64_t>(mLoopStart) * L1TileShape::M * params.problemShape.n() +
                                    params.layoutC.GetOffset(offsetC);

                if constexpr (HasBias) {
                    blockMmad(gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC],
                              params.layoutC, gmBias[blockLocCoord.n()], blockSizeCoord);
                } else {
                    bool isFirstBlock = (loopOffset < coreNum);
                    bool hasNextBlock = false;
                    GemmCoord nextBlockIdCoord;
                    GemmCoord nextBlockLocCoord;
                    GemmCoord nextBlockSizeCoord;
                    if (loopOffset + coreNum < actualLoopNum) {
                        hasNextBlock = true;
                        nextBlockIdCoord = GetBlockIdCoord(loopOffset + coreNum, actualPValue, nLoops,
                                                           params.swizzlDirect, params.swizzlCount);
                        nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                        nextBlockSizeCoord =
                            GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, actualPValue, blockMSize, nLoops,
                                              params.problemShape.n(), params.problemShape.k());
                    }
                    MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
                    MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
                    int64_t gmOffsetNextA = params.layoutA.GetOffset(offsetNextA) + peerMemBlockSt;
                    int64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);
                    blockMmad(gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC],
                              params.layoutC, gmA[gmOffsetNextA], gmB[gmOffsetNextB], blockSizeCoord,
                              nextBlockSizeCoord, isFirstBlock, hasNextBlock);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flagIdx);
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_ALLTOALL_MATMUL_HPP
