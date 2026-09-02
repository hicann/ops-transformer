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
 * \file matmul_reduce_scatter_v2_matmul.hpp
 * \brief
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_REDUCE_SCATTER_V2_MATMUL_REDUCE_SCATTER_AIV_MODE_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_REDUCE_SCATTER_V2_MATMUL_REDUCE_SCATTER_AIV_MODE_HPP

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_cross_core_sync.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_gm_to_ub.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_ub_to_gm.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/tla_gemm_kernel_padding_matmul.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_matrix_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tile/tla_gemm_tile_copy.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tile/tla_gemm_tile_mmad.hpp"
#include "matmul_reduce_scatter_aiv_mode_util.h"
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"
#include "mrs_v2_block_mmad_preload_fixpipe.h"

using namespace AscendC;
using namespace matmulReduceScatterV2_aivmode_tiling;
using namespace matmulReduceScatterV2_util;
namespace Catlass::Gemm::Kernel {
template <class PrologueA, class PrologueB, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, bool HasBias>
class MatmulReduceScatterAivMode : public CommBase {
public:
    using BlockMmad = BlockMmad_;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementScale = uint64_t;
    using ElementAInt8 = int8_t;
    using ElementBInt8 = int8_t;
    using ElementCHalf = half;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;
    using LayoutScale = typename layout::VectorLayout;
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
    using L0TileShape = typename BlockMmad::L0TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using FixpipeBlockMmad = Gemm::Block::FixpipeBlockMmad<Gemm::MmadAtlasA2Preload<false, false>, L1TileShape,
                                                           L0TileShape, LayoutA, LayoutB, LayoutC>;

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
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
        GM_ADDR ptrPeerMem;
        LayoutC layoutPeerMem;
        GM_ADDR ptrWorkSpace;
        int32_t pValue;
        int32_t swizzlCount;
        int32_t swizzlDirect;
        DequantType dequantType;
        int32_t rankIdx;
        int32_t rankSize;
        bool needFixpipe;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrBias_, GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrScale_, LayoutScale layoutScale_,
               GM_ADDR ptrPeerMem_, LayoutC layoutPeerMem_, GM_ADDR ptrWorkSpace_, int32_t pValue_,
               int32_t swizzlCount_, int32_t swizzlDirect_, DequantType dequantType_, int32_t rankIdx_,
               int32_t rankSize_, bool needFixpipe_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrBias(ptrBias_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              ptrScale(ptrScale_),
              layoutScale(layoutScale_),
              ptrPeerMem(ptrPeerMem_),
              layoutPeerMem(layoutPeerMem_),
              ptrWorkSpace(ptrWorkSpace_),
              pValue(pValue_),
              swizzlCount(swizzlCount_),
              swizzlDirect(swizzlDirect_),
              dequantType(dequantType_),
              rankIdx(rankIdx_),
              rankSize(rankSize_),
              needFixpipe(needFixpipe_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    MatmulReduceScatterAivMode() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    inline __aicore__ void InitArgs(Params const &params)
    {
        coreIdx = AscendC::GetBlockIdx();
        coreNum = AscendC::GetBlockNum();
        finalM = params.problemShape.m() / params.rankSize;
        mLoopPerRank = (finalM + L1TileShape::M - 1) / L1TileShape::M;
        mLoops = mLoopPerRank * params.rankSize;
        nLoops = (params.problemShape.n() + L1TileShape::N - 1) / L1TileShape::N;
        coreLoops = mLoops * nLoops;
        kAlign = Block512B<ElementA>::AlignUp(params.problemShape.k());
        loopNumPerComm = params.pValue * coreNum;
        calCount = (coreLoops + loopNumPerComm - 1) / loopNumPerComm;
    }

    inline __aicore__ GemmCoord GetBlockSizeCoord(GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop,
                                                  int32_t mSize, int32_t nLoop, int32_t nSize, int32_t kSize)
    {
        uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : L1TileShape::M;
        uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : L1TileShape::N;
        uint32_t kActual = kSize;
        return GemmCoord{mActual, nActual, kActual};
    }

    inline __aicore__ GemmCoord GetBlockLocCoord(GemmCoord blockIdxCoord)
    {
        return GemmCoord{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N,
                         blockIdxCoord.k() * L1TileShape::K};
    }

    inline __aicore__ GemmCoord GetBlockIdCoord(int32_t loopOffset, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect,
                                                int32_t swizzlCount)
    {
        uint32_t kIdx = 0;
        int64_t mIdx, nIdx;
        GetSwizzledBlockIdx(loopOffset, mLoop, nLoop, swizzlDirect, swizzlCount, mIdx, nIdx);
        return GemmCoord{static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), kIdx}; // idx在uint32_t范围内
    }

    inline __aicore__ void FixpipeMatmul(Params const &params)
    {
        AscendC::GlobalTensor<ElementAInt8> gmAInt8;
        AscendC::GlobalTensor<ElementBInt8> gmBInt8;
        AscendC::GlobalTensor<ElementScale> gmScale;
        AscendC::GlobalTensor<ElementCHalf> gmCHalf;
        AscendC::GlobalTensor<ElementCHalf> gmPeerMemHalf;
        gmAInt8.SetGlobalBuffer((__gm__ ElementAInt8 *)params.ptrA);
        gmBInt8.SetGlobalBuffer((__gm__ ElementBInt8 *)params.ptrB);
        gmCHalf.SetGlobalBuffer((__gm__ ElementCHalf *)params.ptrC);
        gmPeerMemHalf.SetGlobalBuffer((__gm__ ElementCHalf *)params.ptrPeerMem);
        gmScale.SetGlobalBuffer((__gm__ ElementScale *)params.ptrScale);

        FixpipeBlockMmad fixpipeBlockMmad(resource);
        int32_t blockSize = L1TileShape::M * L1TileShape::N;
        for (int32_t calIdx = 0; calIdx < calCount; calIdx++) {
            int32_t flagIdx = calIdx % MAX_BLOCK_COUNT;
            if (calIdx >= MAX_BLOCK_COUNT) {
                WaitEvent(flagIdx);
            }
            for (int32_t p = 0; p < params.pValue; p++) {
                int32_t loopIdx = calIdx * loopNumPerComm + p * coreNum + coreIdx;
                if (loopIdx >= coreLoops) {
                    break;
                }
                int32_t dstRankIdx = loopIdx % params.rankSize;
                int64_t gmABlockSt = static_cast<int64_t>(dstRankIdx) * finalM * kAlign;
                int32_t inRankIdx = loopIdx / params.rankSize;
                GemmCoord blockIdxCoord =
                    GetBlockIdCoord(inRankIdx, mLoopPerRank, nLoops, params.swizzlDirect, params.swizzlCount);
                GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
                GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, mLoopPerRank, finalM, nLoops,
                                                             params.problemShape.n(), params.problemShape.k());

                MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
                MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
                MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
                int64_t gmOffsetA = gmABlockSt + params.layoutA.GetOffset(offsetA);
                int64_t gmOffsetC;
                int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                LayoutC layoutGmDst;
                AscendC::GlobalTensor<ElementCHalf> gmDstHalf;
                if (dstRankIdx == params.rankIdx && params.dequantType == DequantType::PER_CHANNEL) {
                    layoutGmDst = params.layoutC;
                    gmDstHalf = gmCHalf;
                    gmOffsetC = params.layoutC.GetOffset(offsetC);
                } else {
                    layoutGmDst = params.layoutPeerMem;
                    gmDstHalf = gmPeerMemHalf;
                    gmOffsetC = (static_cast<int64_t>(flagIdx) * loopNumPerComm +
                                 dstRankIdx * (loopNumPerComm / params.rankSize) +
                                 (loopIdx % loopNumPerComm) / params.rankSize) *
                                blockSize;
                }

                bool hasNextBlock = false;
                bool isFirstBlock = loopIdx == coreIdx;
                GemmCoord nextBlockIdCoord;
                GemmCoord nextBlockLocCoord;
                GemmCoord nextBlockSizeCoord;
                int32_t nextLoopIdx = loopIdx + coreNum;
                int32_t nextDstRankIdx = nextLoopIdx % params.rankSize;
                int32_t nextInRankIdx = nextLoopIdx / params.rankSize;
                if (nextLoopIdx < coreLoops) {
                    hasNextBlock = true;
                    nextBlockIdCoord =
                        GetBlockIdCoord(nextInRankIdx, mLoopPerRank, nLoops, params.swizzlDirect, params.swizzlCount);
                    nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                    nextBlockSizeCoord = GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, mLoopPerRank, finalM,
                                                           nLoops, params.problemShape.n(), params.problemShape.k());
                }
                int64_t nextGmABlockSt = static_cast<int64_t>(nextDstRankIdx) * finalM * kAlign;
                MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
                MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
                int64_t gmOffsetNextA = nextGmABlockSt + params.layoutA.GetOffset(offsetNextA);
                int64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);

                int64_t gmOffsetScale = blockLocCoord.n();
                fixpipeBlockMmad(gmAInt8[gmOffsetA], params.layoutA, gmBInt8[gmOffsetB], params.layoutB,
                                 gmDstHalf[gmOffsetC], layoutGmDst, gmScale[gmOffsetScale], params.layoutScale,
                                 gmAInt8[gmOffsetNextA], gmBInt8[gmOffsetNextB], blockSizeCoord, nextBlockSizeCoord,
                                 isFirstBlock, hasNextBlock);
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(flagIdx);
        }
    }

    inline __aicore__ void Matmul(Params const &params)
    {
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementBias> gmBias;
        AscendC::GlobalTensor<ElementC> gmPeerMem;
        AscendC::GlobalTensor<ElementC> gmWorkSpace;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        if constexpr (HasBias) {
            gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);
        }
        gmPeerMem.SetGlobalBuffer((__gm__ ElementC *)params.ptrPeerMem);
        gmWorkSpace.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkSpace);

        BlockMmad blockMmad(resource);
        int32_t matmulBlockSize = L1TileShape::M * L1TileShape::N;
        for (int32_t matmulCalIdx = 0; matmulCalIdx < calCount; matmulCalIdx++) {
            int32_t matmulFlagIdx = matmulCalIdx % MAX_BLOCK_COUNT;
            if (matmulCalIdx >= MAX_BLOCK_COUNT) {
                WaitEvent(matmulFlagIdx);
            }
            for (int32_t p = 0; p < params.pValue; p++) {
                int32_t matmulLoopIdx = matmulCalIdx * loopNumPerComm + p * coreNum + coreIdx;
                if (matmulLoopIdx >= coreLoops) {
                    break;
                }
                int32_t matmulDstRankIdx = matmulLoopIdx % params.rankSize;
                int32_t matmulInRankIdx = matmulLoopIdx / params.rankSize;
                int64_t matmulGmABlockSt = static_cast<int64_t>(matmulDstRankIdx) * finalM * kAlign;
                GemmCoord matmulBlockIdxCoord =
                    GetBlockIdCoord(matmulInRankIdx, mLoopPerRank, nLoops, params.swizzlDirect, params.swizzlCount);
                GemmCoord matmulBlockLocCoord = GetBlockLocCoord(matmulBlockIdxCoord);
                GemmCoord matmulBlockSizeCoord =
                    GetBlockSizeCoord(matmulBlockIdxCoord, matmulBlockLocCoord, mLoopPerRank, finalM, nLoops,
                                      params.problemShape.n(), params.problemShape.k());

                MatrixCoord offsetA{matmulBlockLocCoord.m(), matmulBlockLocCoord.k()};
                MatrixCoord offsetB{matmulBlockLocCoord.k(), matmulBlockLocCoord.n()};
                MatrixCoord matmulOffsetC{matmulBlockLocCoord.m(), matmulBlockLocCoord.n()};
                int64_t matmulGmOffsetA = matmulGmABlockSt + params.layoutA.GetOffset(offsetA);
                int64_t matmulGmOffsetB = params.layoutB.GetOffset(offsetB);
                int64_t matmulGmOffsetC;
                LayoutC layoutGmDst;
                AscendC::GlobalTensor<ElementC> gmDst;
                if (std::is_same<ElementC, int32_t>::value) {
                    gmDst = gmWorkSpace;
                    layoutGmDst = params.layoutPeerMem;
                    matmulGmOffsetC = (static_cast<int64_t>(matmulFlagIdx) * loopNumPerComm +
                                       matmulDstRankIdx * (loopNumPerComm / params.rankSize) +
                                       (matmulLoopIdx % loopNumPerComm) / params.rankSize) *
                                      matmulBlockSize;
                } else if (matmulDstRankIdx == params.rankIdx) {
                    gmDst = gmC;
                    layoutGmDst = params.layoutC;
                    matmulGmOffsetC = params.layoutC.GetOffset(matmulOffsetC);
                } else {
                    gmDst = gmPeerMem;
                    layoutGmDst = params.layoutPeerMem;
                    matmulGmOffsetC = (static_cast<int64_t>(matmulFlagIdx) * loopNumPerComm +
                                       matmulDstRankIdx * (loopNumPerComm / params.rankSize) +
                                       (matmulLoopIdx % loopNumPerComm) / params.rankSize) *
                                      matmulBlockSize;
                }
                if constexpr (HasBias) {
                    blockMmad(gmA[matmulGmOffsetA], params.layoutA, gmB[matmulGmOffsetB], params.layoutB,
                              gmDst[matmulGmOffsetC], layoutGmDst, gmBias[matmulBlockLocCoord.n()],
                              matmulBlockSizeCoord);
                } else {
                    bool matmulIsFirstBlock = matmulLoopIdx == coreIdx;
                    bool matmulHasNextBlock = false;
                    GemmCoord nextBlockLocCoord;
                    GemmCoord matmulNextBlockSizeCoord;
                    GemmCoord nextBlockIdCoord;
                    int32_t nextLoopIdx = matmulLoopIdx + coreNum;
                    int32_t nextInRankIdx = nextLoopIdx / params.rankSize;
                    int32_t nextDstRankIdx = nextLoopIdx % params.rankSize;
                    if (nextLoopIdx < coreLoops) {
                        matmulHasNextBlock = true;
                        nextBlockIdCoord = GetBlockIdCoord(nextInRankIdx, mLoopPerRank, nLoops, params.swizzlDirect,
                                                           params.swizzlCount);
                        nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                        matmulNextBlockSizeCoord =
                            GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, mLoopPerRank, finalM, nLoops,
                                              params.problemShape.n(), params.problemShape.k());
                    }
                    int64_t nextGmABlockSt = static_cast<int64_t>(nextDstRankIdx) * finalM * kAlign;
                    MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
                    MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
                    int64_t matmulGmOffsetNextA = nextGmABlockSt + params.layoutA.GetOffset(offsetNextA);
                    int64_t matmulGmOffsetNextB = params.layoutB.GetOffset(offsetNextB);
                    blockMmad(gmA[matmulGmOffsetA], params.layoutA, gmB[matmulGmOffsetB], params.layoutB,
                              gmDst[matmulGmOffsetC], layoutGmDst, gmA[matmulGmOffsetNextA], gmB[matmulGmOffsetNextB],
                              matmulBlockSizeCoord, matmulNextBlockSizeCoord, matmulIsFirstBlock, matmulHasNextBlock);
                }
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(matmulFlagIdx);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);

        InitArgs(params);

        if (params.needFixpipe) {
            FixpipeMatmul(params);
        } else {
            Matmul(params);
        }
    }

private:
    int32_t coreIdx;
    int32_t coreNum;
    int32_t mLoops;
    int32_t nLoops;
    int32_t coreLoops;
    int32_t calCount;
    int32_t finalM;
    int32_t kAlign;
    int32_t mLoopPerRank;
    int32_t loopNumPerComm;
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};
} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_REDUCE_SCATTER_V2_HPP
