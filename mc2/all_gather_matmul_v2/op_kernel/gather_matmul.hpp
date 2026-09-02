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

    CATLASS_DEVICE
    void DoLocalMatmul(Params const &params)
    {
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementC> gmDst = outputTypeInt32 ? gmWorkSpace : gmC;
        uint64_t dstSt = params.rankIdx * static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n();
        for (int32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {
            if (loopIdx % coreNum != coreIdx) {
                continue;
            }
            GemmCoord blockIdxCoord = GetBlockIdCoord(loopIdx, mLoops, nLoops, params.swizzlDirect, params.swizzlCount);
            GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
            GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, mLoops, params.problemShape.m(),
                                                         nLoops, params.problemShape.n(), params.problemShape.k());

            MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
            MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
            MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
            uint64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            uint64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            uint64_t gmOffsetC = dstSt + params.layoutC.GetOffset(offsetC);

            bool isFirstBlock = (loopIdx < coreNum);
            bool hasNextBlock = false;
            GemmCoord nextBlockIdCoord;
            GemmCoord nextBlockLocCoord;
            GemmCoord nextBlockSizeCoord;
            int32_t nextLoopIdx = loopIdx + coreNum;
            if (nextLoopIdx < coreLoops) {
                hasNextBlock = true;
                nextBlockIdCoord =
                    GetBlockIdCoord(nextLoopIdx, mLoops, nLoops, params.swizzlDirect, params.swizzlCount);
                nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                nextBlockSizeCoord =
                    GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, mLoops, params.problemShape.m(), nLoops,
                                      params.problemShape.n(), params.problemShape.k());
            }
            MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
            MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
            uint64_t gmOffsetNextA = params.layoutA.GetOffset(offsetNextA);
            uint64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);

            blockMmad(gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmDst[gmOffsetC], params.layoutC,
                      gmA[gmOffsetNextA], gmB[gmOffsetNextB], blockSizeCoord, nextBlockSizeCoord, isFirstBlock,
                      hasNextBlock);
        }
    }

    CATLASS_DEVICE
    void DoLocalFixpipeMatmul(Params const &params)
    {
        FixpipeBlockMmad fixpipeBlockMmad(resource);

        uint64_t dstSt = params.rankIdx * static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n();
        for (int32_t loopIdx = 0; loopIdx < coreLoops; loopIdx++) {
            if (loopIdx % coreNum != coreIdx) {
                continue;
            }
            GemmCoord blockIdxCoord = GetBlockIdCoord(loopIdx, mLoops, nLoops, params.swizzlDirect, params.swizzlCount);
            GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
            GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, mLoops, params.problemShape.m(),
                                                         nLoops, params.problemShape.n(), params.problemShape.k());

            MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
            MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
            MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
            uint64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            uint64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            uint64_t gmOffsetC = dstSt + params.layoutC.GetOffset(offsetC);

            bool isFirstBlock = (loopIdx < coreNum);
            bool hasNextBlock = false;
            GemmCoord nextBlockIdCoord;
            GemmCoord nextBlockLocCoord;
            GemmCoord nextBlockSizeCoord;
            int32_t nextLoopIdx = loopIdx + coreNum;
            if (nextLoopIdx < coreLoops) {
                hasNextBlock = true;
                nextBlockIdCoord =
                    GetBlockIdCoord(nextLoopIdx, mLoops, nLoops, params.swizzlDirect, params.swizzlCount);
                nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                nextBlockSizeCoord =
                    GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, mLoops, params.problemShape.m(), nLoops,
                                      params.problemShape.n(), params.problemShape.k());
            }
            MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
            MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
            uint64_t gmOffsetNextA = params.layoutA.GetOffset(offsetNextA);
            uint64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);

            uint64_t gmOffsetScale = blockLocCoord.n();
            fixpipeBlockMmad(gmAInt8[gmOffsetA], params.layoutA, gmBInt8[gmOffsetB], params.layoutB, gmCHalf[gmOffsetC],
                             params.layoutC, gmScale[gmOffsetScale], params.layoutScale, gmAInt8[gmOffsetNextA],
                             gmBInt8[gmOffsetNextB], blockSizeCoord, nextBlockSizeCoord, isFirstBlock, hasNextBlock);
        }
    }

    inline __aicore__ void FixpipeMatmul(Params const &params)
    {
        FixpipeBlockMmad fixpipeBlockMmad(resource);

        int32_t otherRankNum = params.rankSize;
        int32_t blockM = params.pValue * L1TileShape::M;
        uint64_t blockSize = static_cast<uint64_t>(blockM) * kAlign;
        uint64_t outputBlockSize = static_cast<uint64_t>(blockM) * params.problemShape.n();
        uint64_t mnSize = static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n();
        for (int32_t calIdx = 0; calIdx < calCount; calIdx++) {
            int32_t flagIdx = calIdx % MAX_BLOCK_COUNT;
            int32_t actualPValue = params.pValue;
            if (calIdx == calCount - 1) {
                actualPValue = mLoops - calIdx * params.pValue;
                blockM = params.problemShape.m() - calIdx * blockM;
            }

            WaitEvent(flagIdx);

            int32_t pingPongSt = flagIdx * pingPongSize;
            int32_t loopNumInOtherRank = actualPValue * otherRankNum * nLoops;
            int32_t loopSt = coreLoops + calIdx * params.pValue * nLoops * otherRankNum;
            for (int32_t loopOffset = 0; loopOffset < loopNumInOtherRank; loopOffset++) {
                int32_t loopIdx = loopSt + loopOffset;
                if (loopIdx % coreNum != coreIdx) {
                    continue;
                }
                int32_t loopOffsetInBlock = loopOffset / otherRankNum;
                int32_t dstBlockIdx = loopOffset % otherRankNum;
                GemmCoord blockIdxCoord =
                    GetBlockIdCoord(loopOffsetInBlock, actualPValue, nLoops, params.swizzlDirect, params.swizzlCount);
                GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
                GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, actualPValue, blockM, nLoops,
                                                             params.problemShape.n(), params.problemShape.k());
                MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
                MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
                MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
                uint64_t gmOffsetA = pingPongSt + dstBlockIdx * blockSize + params.layoutPeerMem.GetOffset(offsetA);
                uint64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = dstBlockIdx * mnSize + calIdx * outputBlockSize + params.layoutC.GetOffset(offsetC);

                AscendC::GlobalTensor<ElementAInt8> gmAIn = gmPeerMemInt8;
                if (dstBlockIdx == params.rankIdx) { // 从gmA里面取
                    gmAIn = gmAInt8;
                    gmOffsetA = calIdx * blockSize + params.layoutA.GetOffset(offsetA);
                }

                bool isFirstBlock = (loopOffset < coreNum);
                bool hasNextBlock = false;
                GemmCoord nextBlockIdCoord;
                GemmCoord nextBlockLocCoord;
                GemmCoord nextBlockSizeCoord;
                int32_t nextLoopOffset = loopOffset + coreNum;
                int32_t nextLoopOffsetInBlock = nextLoopOffset / otherRankNum;
                int32_t nextDstBlockIdx = nextLoopOffset % otherRankNum;
                if (nextLoopOffset < loopNumInOtherRank) {
                    hasNextBlock = true;
                    nextBlockIdCoord = GetBlockIdCoord(nextLoopOffsetInBlock, actualPValue, nLoops, params.swizzlDirect,
                                                       params.swizzlCount);
                    nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                    nextBlockSizeCoord = GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, actualPValue, blockM,
                                                           nLoops, params.problemShape.n(), params.problemShape.k());
                }
                MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
                MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
                uint64_t gmOffsetNextA =
                    pingPongSt + nextDstBlockIdx * blockSize + params.layoutPeerMem.GetOffset(offsetNextA);
                uint64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);

                AscendC::GlobalTensor<ElementAInt8> gmANextIn = gmPeerMemInt8;
                if (nextDstBlockIdx == params.rankIdx) { // 从gmA里面取
                    gmANextIn = gmAInt8;
                    gmOffsetNextA = calIdx * blockSize + params.layoutA.GetOffset(offsetNextA);
                }

                uint64_t gmOffsetScale = blockLocCoord.n();
                fixpipeBlockMmad(gmAIn[gmOffsetA], params.layoutPeerMem, gmBInt8[gmOffsetB], params.layoutB,
                                 gmCHalf[gmOffsetC], params.layoutC, gmScale[gmOffsetScale], params.layoutScale,
                                 gmANextIn[gmOffsetNextA], gmBInt8[gmOffsetNextB], blockSizeCoord, nextBlockSizeCoord,
                                 isFirstBlock, hasNextBlock);
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(flagIdx);
        }
    }

    inline __aicore__ void Matmul(Params const &params)
    {
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementC> gmDst = outputTypeInt32 ? gmWorkSpace : gmC;
        int32_t otherRankNum = params.rankSize;
        int32_t blockM = params.pValue * L1TileShape::M;
        uint64_t blockSize = static_cast<uint64_t>(blockM) * kAlign;
        uint64_t outputBlockSize = static_cast<uint64_t>(blockM) * params.problemShape.n();
        uint64_t mnSize = static_cast<uint64_t>(params.problemShape.m()) * params.problemShape.n();
        for (int32_t calIdx = 0; calIdx < calCount; calIdx++) {
            int32_t flagIdx = calIdx % MAX_BLOCK_COUNT;
            int32_t actualPValue = params.pValue;
            if (calIdx == calCount - 1) {
                actualPValue = mLoops - calIdx * params.pValue;
                blockM = params.problemShape.m() - calIdx * blockM;
            }

            WaitEvent(flagIdx);

            int32_t pingPongSt = flagIdx * pingPongSize;
            int32_t loopNumInOtherRank = actualPValue * otherRankNum * nLoops;
            int32_t loopSt = coreLoops + calIdx * params.pValue * nLoops * otherRankNum;
            for (int32_t loopOffset = 0; loopOffset < loopNumInOtherRank; loopOffset++) {
                int32_t loopIdx = loopSt + loopOffset;
                if (loopIdx % coreNum != coreIdx) {
                    continue;
                }
                int32_t loopOffsetInBlock = loopOffset / otherRankNum;
                int32_t dstBlockIdx = loopOffset % otherRankNum;
                GemmCoord blockIdxCoord =
                    GetBlockIdCoord(loopOffsetInBlock, actualPValue, nLoops, params.swizzlDirect, params.swizzlCount);
                GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
                GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, actualPValue, blockM, nLoops,
                                                             params.problemShape.n(), params.problemShape.k());
                MatrixCoord offsetA{blockLocCoord.m(), blockLocCoord.k()};
                MatrixCoord offsetB{blockLocCoord.k(), blockLocCoord.n()};
                MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
                uint64_t gmOffsetA = pingPongSt + dstBlockIdx * blockSize + params.layoutPeerMem.GetOffset(offsetA);
                uint64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = dstBlockIdx * mnSize + calIdx * outputBlockSize + params.layoutC.GetOffset(offsetC);
                if (params.accumWorkSpacePingPong) {
                    gmOffsetC = (static_cast<int64_t>(dstBlockIdx) + static_cast<int64_t>(flagIdx) * params.rankSize) *
                                    outputBlockSize +
                                params.layoutC.GetOffset(offsetC);
                }

                AscendC::GlobalTensor<ElementA> gmAIn = gmPeerMem;
                if (dstBlockIdx == params.rankIdx) { // 从gmA里面取
                    gmAIn = gmA;
                    gmOffsetA = calIdx * blockSize + params.layoutA.GetOffset(offsetA);
                }

                bool isFirstBlock = (loopOffset < coreNum);
                bool hasNextBlock = false;
                GemmCoord nextBlockIdCoord;
                GemmCoord nextBlockLocCoord;
                GemmCoord nextBlockSizeCoord;
                int32_t nextLoopOffset = loopOffset + coreNum;
                int32_t nextLoopOffsetInBlock = nextLoopOffset / otherRankNum;
                int32_t nextDstBlockIdx = nextLoopOffset % otherRankNum;
                if (nextLoopOffset < loopNumInOtherRank) {
                    hasNextBlock = true;
                    nextBlockIdCoord = GetBlockIdCoord(nextLoopOffsetInBlock, actualPValue, nLoops, params.swizzlDirect,
                                                       params.swizzlCount);
                    nextBlockLocCoord = GetBlockLocCoord(nextBlockIdCoord);
                    nextBlockSizeCoord = GetBlockSizeCoord(nextBlockIdCoord, nextBlockLocCoord, actualPValue, blockM,
                                                           nLoops, params.problemShape.n(), params.problemShape.k());
                }
                MatrixCoord offsetNextA{nextBlockLocCoord.m(), nextBlockLocCoord.k()};
                MatrixCoord offsetNextB{nextBlockLocCoord.k(), nextBlockLocCoord.n()};
                uint64_t gmOffsetNextA =
                    pingPongSt + nextDstBlockIdx * blockSize + params.layoutPeerMem.GetOffset(offsetNextA);
                uint64_t gmOffsetNextB = params.layoutB.GetOffset(offsetNextB);

                AscendC::GlobalTensor<ElementA> gmAInNext = gmPeerMem;
                if (nextDstBlockIdx == params.rankIdx) { // 从gmA里面取
                    gmAInNext = gmA;
                    gmOffsetNextA = calIdx * blockSize + params.layoutA.GetOffset(offsetNextA);
                }

                blockMmad(gmAIn[gmOffsetA], params.layoutPeerMem, gmB[gmOffsetB], params.layoutB, gmDst[gmOffsetC],
                          params.layoutC, gmAInNext[gmOffsetNextA], gmB[gmOffsetNextB], blockSizeCoord,
                          nextBlockSizeCoord, isFirstBlock, hasNextBlock);
            }
            FFTSCrossCoreSync<PIPE_FIX, 2>(flagIdx);
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
