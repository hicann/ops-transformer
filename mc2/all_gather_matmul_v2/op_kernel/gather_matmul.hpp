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
 * \file gather_matmul.hpp
 * \brief
 */

#ifndef CATLASS_GEMM_KERNEL_ALLGATHER_MATMUL_HPP
#define CATLASS_GEMM_KERNEL_ALLGATHER_MATMUL_HPP

#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_matrix_coord.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_cross_core_sync.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_gm_to_ub.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_ub_to_gm.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/tla_gemm_kernel_padding_matmul.hpp"
#include "block_mmad_preload_fixpipe.h"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tile/tla_gemm_tile_copy.hpp"
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/tile/tla_gemm_tile_mmad.hpp"

using namespace AscendC;

namespace Catlass::Gemm::Kernel {
constexpr static int32_t AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID = 12;
constexpr static int32_t MAX_BLOCK_COUNT = 2;

template <class PrologueA, class PrologueB, class BlockMmad_>
class AllGatherMatmulV2 {
public:
    using BlockMmad = BlockMmad_;
    using DispatchPolicy = typename BlockMmad::DispatchPolicy;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementScale = uint64_t;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutScale = typename layout::VectorLayout;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using L0TileShape = typename BlockMmad::L0TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using ElementAInt8 = int8_t;
    using ElementBInt8 = int8_t;
    using ElementCHalf = half;
    using FixpipeBlockMmad =
        Gemm::Block::FixpipeBlockMmad<DispatchPolicy, L1TileShape, L0TileShape, LayoutA, LayoutB, LayoutC>;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
        GM_ADDR ptrPeerMem;
        LayoutA layoutPeerMem;
        GM_ADDR ptrWorkSpace;
        int32_t pValue;
        int32_t swizzlCount;
        int32_t swizzlDirect;
        int32_t rankIdx;
        int32_t rankSize;
        bool needFixpipe;
        bool accumWorkSpacePingPong;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrC_, LayoutC layoutC_, GM_ADDR ptrScale_, LayoutScale layoutScale_, GM_ADDR ptrPeerMem_,
               LayoutA layoutPeerMem_, GM_ADDR ptrWorkSpace_, int32_t pValue_, int32_t swizzlCount_,
               int32_t swizzlDirect_, int32_t rankIdx_, int32_t rankSize_, bool needFixpipe_,
               bool accumWorkSpacePingPong_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
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
              rankIdx(rankIdx_),
              rankSize(rankSize_),
              needFixpipe(needFixpipe_),
              accumWorkSpacePingPong(accumWorkSpacePingPong_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    AllGatherMatmulV2() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    inline __aicore__ void InitArgs(Params const &params)
    {
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        gmWorkSpace.SetGlobalBuffer((__gm__ ElementC *)params.ptrWorkSpace);
        gmPeerMem.SetGlobalBuffer((__gm__ ElementA *)params.ptrPeerMem);
        gmScale.SetGlobalBuffer((__gm__ ElementScale *)params.ptrScale);
        gmAInt8.SetGlobalBuffer((__gm__ ElementAInt8 *)params.ptrA);
        gmBInt8.SetGlobalBuffer((__gm__ ElementBInt8 *)params.ptrB);
        gmCHalf.SetGlobalBuffer((__gm__ ElementCHalf *)params.ptrC);
        gmPeerMemInt8.SetGlobalBuffer((__gm__ ElementAInt8 *)params.ptrPeerMem);

        outputTypeInt32 = std::is_same<ElementC, int32_t>::value;
        coreIdx = AscendC::GetBlockIdx();
        coreNum = AscendC::GetBlockNum();
        mLoops = (params.problemShape.m() + L1TileShape::M - 1) / L1TileShape::M;
        nLoops = (params.problemShape.n() + L1TileShape::N - 1) / L1TileShape::N;
        coreLoops = mLoops * nLoops;
        kAlign = Block512B<ElementA>::AlignUp(params.problemShape.k());
        pingPongSize = L1TileShape::M * kAlign * params.pValue * params.rankSize;
        calCount = (mLoops + params.pValue - 1) / params.pValue;
    }

    inline __aicore__ GemmCoord GetBlockIdCoord(int32_t loopOffset, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect,
                                                int32_t swizzlCount)
    {
        uint32_t kIdx = 0;
        int64_t mIdx, nIdx;
        GetBlockIdx(loopOffset, mLoop, nLoop, swizzlDirect, swizzlCount, mIdx, nIdx);
        return GemmCoord{static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), kIdx}; // idx在uint32_t范围内
    }

    inline __aicore__ GemmCoord GetBlockLocCoord(GemmCoord blockIdxCoord)
    {
        return GemmCoord{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N,
                         blockIdxCoord.k() * L1TileShape::K};
    }

    inline __aicore__ GemmCoord GetBlockSizeCoord(GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop,
                                                  int32_t mSize, int32_t nLoop, int32_t nSize, int32_t kSize)
    {
        uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : L1TileShape::M;
        uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : L1TileShape::N;
        uint32_t kActual = kSize;
        return GemmCoord{mActual, nActual, kActual};
    }

    // Per-block geometry shared by the local-block matmul loops below.
    struct LocalBlockInfo {
        GemmCoord blockLocCoord;
        GemmCoord blockSizeCoord;
        GemmCoord nextBlockSizeCoord;
        uint64_t gmOffsetA;
        uint64_t gmOffsetB;
        uint64_t gmOffsetC;
        uint64_t gmOffsetNextA;
        uint64_t gmOffsetNextB;
        bool isFirstBlock;
        bool hasNextBlock;
    };

    inline __aicore__ LocalBlockInfo GetLocalBlockInfo(int32_t loopIdx, Params const &params)
    {
        GemmCoord blockIdxCoord = GetBlockIdCoord(loopIdx, mLoops, nLoops, params.swizzlDirect, params.swizzlCount);
        GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
        GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, mLoops, params.problemShape.m(),
                                                     nLoops, params.problemShape.n(), params.problemShape.k());
        MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
        MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
        MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
        uint64_t dstSt = params.rankIdx * static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n();

        GemmCoord nextBlockLocCoord;
        GemmCoord nextBlockSizeCoord;
        int32_t nextLoopIdx = loopIdx + coreNum;
        bool hasNextBlock = false;
        if (nextLoopIdx < coreLoops) {
            hasNextBlock = true;
            GemmCoord nextBlockIdCoord =
                GetBlockIdCoord(nextLoopIdx, mLoops, nLoops, params.swizzlDirect, params.swizzlCount);
            nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
            nextBlockSizeCoord = GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, mLoops, params.problemShape.m(),
                                                   nLoops, params.problemShape.n(), params.problemShape.k());
        }
        MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
        MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};

        LocalBlockInfo info{};
        info.blockLocCoord = blockLocCoord;
        info.blockSizeCoord = blockSizeCoord;
        info.nextBlockSizeCoord = nextBlockSizeCoord;
        info.gmOffsetA = params.layoutA.GetOffset(offsetA);
        info.gmOffsetB = params.layoutB.GetOffset(offsetB);
        info.gmOffsetC = dstSt + params.layoutC.GetOffset(offsetC);
        info.gmOffsetNextA = params.layoutA.GetOffset(offsetNextA);
        info.gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);
        info.isFirstBlock = (loopIdx < coreNum);
        info.hasNextBlock = hasNextBlock;
        return info;
    }

    CATLASS_DEVICE
    void DoLocalMatmul(Params const &params)
    {
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementC> gmDst = outputTypeInt32 ? gmWorkSpace : gmC;
        for (int32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {
            if (loopIdx % coreNum != coreIdx) {
                continue;
            }
            LocalBlockInfo info = GetLocalBlockInfo(loopIdx, params);
            blockMmad(gmA[info.gmOffsetA], params.layoutA, gmB[info.gmOffsetB], params.layoutB, gmDst[info.gmOffsetC],
                      params.layoutC, gmA[info.gmOffsetNextA], gmB[info.gmOffsetNextB], info.blockSizeCoord,
                      info.nextBlockSizeCoord, info.isFirstBlock, info.hasNextBlock);
        }
    }

    CATLASS_DEVICE
    void DoLocalFixpipeMatmul(Params const &params)
    {
        FixpipeBlockMmad fixpipeBlockMmad(resource);

        for (int32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {
            if (loopIdx % coreNum != coreIdx) {
                continue;
            }
            LocalBlockInfo info = GetLocalBlockInfo(loopIdx, params);
            uint64_t gmOffsetScale = info.blockLocCoord.n();
            fixpipeBlockMmad(gmAInt8[info.gmOffsetA], params.layoutA, gmBInt8[info.gmOffsetB], params.layoutB,
                             gmCHalf[info.gmOffsetC], params.layoutC, gmScale[gmOffsetScale], params.layoutScale,
                             gmAInt8[info.gmOffsetNextA], gmBInt8[info.gmOffsetNextB], info.blockSizeCoord,
                             info.nextBlockSizeCoord, info.isFirstBlock, info.hasNextBlock);
        }
    }

    // Loop-scope values shared by the cross-rank (remote-block) matmul loops below.
    struct RemoteLoopContext {
        int32_t otherRankNum;
        int32_t actualPValue;
        int32_t blockM;
        int32_t calIdx;
        int32_t flagIdx;
        int32_t pingPongSt;
        int32_t loopNumInOtherRank;
        uint64_t blockSize;
        uint64_t outputBlockSize;
        uint64_t mnSize;
    };

    struct RemoteBlockInfo {
        GemmCoord blockLocCoord;
        GemmCoord blockSizeCoord;
        GemmCoord nextBlockSizeCoord;
        int64_t gmOffsetC;
        uint64_t gmOffsetA;
        uint64_t gmOffsetB;
        uint64_t gmOffsetNextA;
        uint64_t gmOffsetNextB;
        bool aIsLocal;
        bool aNextIsLocal;
        bool isFirstBlock;
        bool hasNextBlock;
    };

    inline __aicore__ RemoteBlockInfo GetRemoteBlockInfo(int32_t loopOffset, RemoteLoopContext const &ctx,
                                                         bool accumWorkSpacePingPong, Params const &params)
    {
        int32_t loopOffsetInBlock = loopOffset / ctx.otherRankNum;
        int32_t dstBlockIdx = loopOffset % ctx.otherRankNum;
        GemmCoord blockIdxCoord =
            GetBlockIdCoord(loopOffsetInBlock, ctx.actualPValue, nLoops, params.swizzlDirect, params.swizzlCount);
        GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
        GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, ctx.actualPValue, ctx.blockM, nLoops,
                                                     params.problemShape.n(), params.problemShape.k());
        MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
        MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
        MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};

        int64_t gmOffsetC;
        if (accumWorkSpacePingPong) {
            gmOffsetC = (static_cast<int64_t>(dstBlockIdx) + static_cast<int64_t>(ctx.flagIdx) * params.rankSize) *
                            ctx.outputBlockSize +
                        params.layoutC.GetOffset(offsetC);
        } else {
            gmOffsetC = dstBlockIdx * ctx.mnSize + ctx.calIdx * ctx.outputBlockSize + params.layoutC.GetOffset(offsetC);
        }

        bool aIsLocal = (dstBlockIdx == params.rankIdx);
        uint64_t gmOffsetA =
            aIsLocal ? (ctx.calIdx * ctx.blockSize + params.layoutA.GetOffset(offsetA)) :
                       (ctx.pingPongSt + dstBlockIdx * ctx.blockSize + params.layoutPeerMem.GetOffset(offsetA));

        GemmCoord nextBlockLocCoord;
        GemmCoord nextBlockSizeCoord;
        uint64_t gmOffsetNextA = 0;
        uint64_t gmOffsetNextB = 0;
        bool aNextIsLocal = false;
        bool hasNextBlock = false;
        int32_t nextLoopOffset = loopOffset + coreNum;
        if (nextLoopOffset < ctx.loopNumInOtherRank) {
            hasNextBlock = true;
            int32_t nextLoopOffsetInBlock = nextLoopOffset / ctx.otherRankNum;
            int32_t nextDstBlockIdx = nextLoopOffset % ctx.otherRankNum;
            GemmCoord nextBlockIdCoord = GetBlockIdCoord(nextLoopOffsetInBlock, ctx.actualPValue, nLoops,
                                                         params.swizzlDirect, params.swizzlCount);
            nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
            nextBlockSizeCoord = GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, ctx.actualPValue, ctx.blockM,
                                                   nLoops, params.problemShape.n(), params.problemShape.k());
            MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
            MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
            aNextIsLocal = (nextDstBlockIdx == params.rankIdx);
            gmOffsetNextA =
                aNextIsLocal ?
                    (ctx.calIdx * ctx.blockSize + params.layoutA.GetOffset(offsetNextA)) :
                    (ctx.pingPongSt + nextDstBlockIdx * ctx.blockSize + params.layoutPeerMem.GetOffset(offsetNextA));
            gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);
        }

        RemoteBlockInfo info{};
        info.blockLocCoord = blockLocCoord;
        info.blockSizeCoord = blockSizeCoord;
        info.nextBlockSizeCoord = nextBlockSizeCoord;
        info.gmOffsetC = gmOffsetC;
        info.gmOffsetA = gmOffsetA;
        info.gmOffsetB = params.layoutB.GetOffset(offsetB);
        info.gmOffsetNextA = gmOffsetNextA;
        info.gmOffsetNextB = gmOffsetNextB;
        info.aIsLocal = aIsLocal;
        info.aNextIsLocal = aNextIsLocal;
        info.isFirstBlock = (loopOffset < coreNum);
        info.hasNextBlock = hasNextBlock;
        return info;
    }

    inline __aicore__ RemoteLoopContext GetRemoteLoopContext(int32_t calIdx, Params const &params)
    {
        RemoteLoopContext ctx;
        ctx.otherRankNum = params.rankSize;
        ctx.blockM = params.pValue * L1TileShape::M;
        ctx.blockSize = static_cast<uint64_t>(ctx.blockM) * kAlign;
        ctx.outputBlockSize = static_cast<uint64_t>(ctx.blockM) * params.problemShape.n();
        ctx.mnSize = static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n();
        ctx.calIdx = calIdx;
        ctx.flagIdx = calIdx % MAX_BLOCK_COUNT;
        ctx.actualPValue = params.pValue;
        if (calIdx == calCount - 1) {
            ctx.actualPValue = mLoops - calIdx * params.pValue;
            ctx.blockM = params.problemShape.m() - calIdx * ctx.blockM;
        }
        ctx.pingPongSt = ctx.flagIdx * pingPongSize;
        ctx.loopNumInOtherRank = ctx.actualPValue * ctx.otherRankNum * nLoops;
        return ctx;
    }

    inline __aicore__ void FixpipeMatmul(Params const &params)
    {
        FixpipeBlockMmad fixpipeBlockMmad(resource);

        for (int32_t calIdx = 0; calIdx < calCount; calIdx++) {
            RemoteLoopContext ctx = GetRemoteLoopContext(calIdx, params);
            WaitEvent(ctx.flagIdx);
            int32_t loopSt = coreLoops + calIdx * params.pValue * nLoops * ctx.otherRankNum;
            for (int32_t loopOffset = 0; loopOffset < ctx.loopNumInOtherRank; loopOffset++) {
                int32_t loopIdx = loopSt + loopOffset;
                if (loopIdx % coreNum != coreIdx) {
                    continue;
                }
                RemoteBlockInfo info = GetRemoteBlockInfo(loopOffset, ctx, false, params);
                AscendC::GlobalTensor<ElementAInt8> gmAIn = info.aIsLocal ? gmAInt8 : gmPeerMemInt8;
                AscendC::GlobalTensor<ElementAInt8> gmANextIn = info.aNextIsLocal ? gmAInt8 : gmPeerMemInt8;
                uint64_t gmOffsetScale = info.blockLocCoord.n();
                fixpipeBlockMmad(gmAIn[info.gmOffsetA], params.layoutPeerMem, gmBInt8[info.gmOffsetB], params.layoutB,
                                 gmCHalf[info.gmOffsetC], params.layoutC, gmScale[gmOffsetScale], params.layoutScale,
                                 gmANextIn[info.gmOffsetNextA], gmBInt8[info.gmOffsetNextB], info.blockSizeCoord,
                                 info.nextBlockSizeCoord, info.isFirstBlock, info.hasNextBlock);
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(ctx.flagIdx);
        }
    }

    inline __aicore__ void Matmul(Params const &params)
    {
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementC> gmDst = outputTypeInt32 ? gmWorkSpace : gmC;
        for (int32_t calIdx = 0; calIdx < calCount; calIdx++) {
            RemoteLoopContext ctx = GetRemoteLoopContext(calIdx, params);
            WaitEvent(ctx.flagIdx);
            int32_t loopSt = coreLoops + calIdx * params.pValue * nLoops * ctx.otherRankNum;
            for (int32_t loopOffset = 0; loopOffset < ctx.loopNumInOtherRank; loopOffset++) {
                int32_t loopIdx = loopSt + loopOffset;
                if (loopIdx % coreNum != coreIdx) {
                    continue;
                }
                RemoteBlockInfo info = GetRemoteBlockInfo(loopOffset, ctx, params.accumWorkSpacePingPong, params);
                AscendC::GlobalTensor<ElementA> gmAIn = info.aIsLocal ? gmA : gmPeerMem;
                AscendC::GlobalTensor<ElementA> gmAInNext = info.aNextIsLocal ? gmA : gmPeerMem;
                blockMmad(gmAIn[info.gmOffsetA], params.layoutPeerMem, gmB[info.gmOffsetB], params.layoutB,
                          gmDst[info.gmOffsetC], params.layoutC, gmAInNext[info.gmOffsetNextA], gmB[info.gmOffsetNextB],
                          info.blockSizeCoord, info.nextBlockSizeCoord, info.isFirstBlock, info.hasNextBlock);
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(ctx.flagIdx);
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);

        InitArgs(params);

        if constexpr (std::is_same_v<ElementA, AscendC::int4b_t>) {
            Matmul(params);
        } else {
            if (params.needFixpipe) {
                FixpipeMatmul(params);
            } else {
                Matmul(params);
            }
        }
    }

private:
    int32_t coreIdx;
    int32_t coreNum;
    int32_t mLoops;
    int32_t nLoops;
    int32_t coreLoops;
    int32_t kAlign;
    int32_t pingPongSize;
    int32_t calCount;
    bool outputTypeInt32;
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
    AscendC::GlobalTensor<ElementA> gmA;
    AscendC::GlobalTensor<ElementB> gmB;
    AscendC::GlobalTensor<ElementC> gmC;
    AscendC::GlobalTensor<ElementC> gmWorkSpace;
    AscendC::GlobalTensor<ElementA> gmPeerMem;
    AscendC::GlobalTensor<ElementScale> gmScale;
    AscendC::GlobalTensor<ElementAInt8> gmAInt8;
    AscendC::GlobalTensor<ElementBInt8> gmBInt8;
    AscendC::GlobalTensor<ElementCHalf> gmCHalf;
    AscendC::GlobalTensor<ElementAInt8> gmPeerMemInt8;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_ALLGATHER_MATMUL_HPP
