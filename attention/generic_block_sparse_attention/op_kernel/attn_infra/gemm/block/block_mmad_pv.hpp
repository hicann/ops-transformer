/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_MMAD_PV_HPP
#define BLOCK_MMAD_PV_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/gbsa_coord.hpp"
#include "../../../attn_infra/arch/gbsa_cross_core_sync.hpp"
#include "../../../attn_infra/gemm/gbsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/gbsa_helper.hpp"
#include "../../../attn_infra/gbsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_tile_mmad.hpp"
#include "../../../gbsa_sparse_kv_geometry.hpp"
#include "../../../gbsa_sparse_pair_copy.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <bool PAGED_CACHE_FLAG_, bool ENABLE_UNIT_FLAG_, bool HAS_RESIDUAL_BLOCK_, class L1TileShape_,
          class L0TileShape_, class AType_, class BType_, class CType_, class BiasType_, class TileCopy_,
          class TileMmad_>
struct BlockMmad<MmadAtlasA2SFAIPV<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, HAS_RESIDUAL_BLOCK_>, L1TileShape_,
                 L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_, TileMmad_> {
public:
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2SFAIPV<PAGED_CACHE_FLAG_, ENABLE_UNIT_FLAG_, HAS_RESIDUAL_BLOCK_>;
    static constexpr bool HAS_RESIDUAL_BLOCK = DispatchPolicy::HAS_RESIDUAL_BLOCK;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    /// Construct
    __aicore__ inline BlockMmad(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
    {
        // Allocate L1 memory space
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(l1BufAddrStart + L1A_SIZE * 2);
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + L1A_SIZE * i);
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmad() {}

    __aicore__ inline void getBlockShape(GemmCoord& actualShape, uint32_t& nowNIdx, uint32_t& nLoop,
                                         uint32_t& stackSeqTile, uint32_t blockSize)
    {
        uint32_t nSplitSize = blockSize;
        if (nowNIdx == nLoop - 1) {
            nSplitSize = stackSeqTile - nowNIdx * blockSize;
        }
        actualShape[2] = nSplitSize;
    }

    __aicore__ inline void LoadP(AscendC::GlobalTensor<ElementA> gA, const LayoutAInL1& layoutAInL1,
                                 const LayoutA& layoutTileA, Arch::CrossCoreFlag softmaxFlag)
    {
        Arch::CrossCoreWaitFlag(softmaxFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
        copyGmToL1A(l1ATensor[l1PPingPongFlag], gA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementB> gB,
                                      AscendC::GlobalTensor<ElementC> gC, AscendC::GlobalTensor<int32_t> gBlockTable,
                                      AscendC::GlobalTensor<int32_t> gSelectIdx, LayoutA layoutA, LayoutB layoutB,
                                      LayoutC layoutC, GemmCoord actualOriShape, uint32_t nIdx, uint32_t nLoop,
                                      uint32_t& blockSize, uint32_t kvSeqlen, uint32_t strideKV, uint32_t blockStackNum,
                                      Arch::CrossCoreFlag softmaxFlag, uint32_t& y, uint32_t selectNum,
                                      uint32_t kvYBlockNum, uint32_t pageSize, uint64_t pageStride,
                                      uint32_t localExplicitBlockCount, uint32_t residualBlockId,
                                      uint64_t peerPageStride, GbsaSparsePairCopy::KvCopyInfo* kvCopyInfo = nullptr)
    {
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        uint32_t stackSeqTile = actualOriShape[2];
        GemmCoord actualShape{rowNum, embed, 0};

        // load V
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(stackSeqTile, embed);
        constexpr uint32_t copySubTileTokens = GbsaSparsePairCopy::KV_COPY_SUB_TILE_TOKENS;
        constexpr bool groupedV = L1TileShape::N == copySubTileTokens && L0TileShape::K == copySubTileTokens &&
                                  L1TileShape::K == GbsaSparsePairCopy::KV_COPY_MAX_TILE_TOKENS;
        // Arch22 tiling guarantees that Y divides 128.
        const uint32_t maxPlanRecords = copySubTileTokens / y;
        // Small M fits both 128-column accumulators in a single L0C slot.
        constexpr bool packL0C =
            L1TileShape::M * 2U * L0TileShape::N * sizeof(ElementAccumulator) <= L0C_PINGPONG_BUF_SIZE;
        constexpr uint32_t streamVMaxEmbed = packL0C ? 2U * L0TileShape::N : L0TileShape::N;
        constexpr bool streamVGeometry = L1TileShape::N == copySubTileTokens &&
                                         L1TileShape::K == 2U * copySubTileTokens &&
                                         copySubTileTokens % L0TileShape::K == 0U;
        const bool streamV = PAGED_CACHE_FLAG_ && streamVGeometry && stackSeqTile == 2U * copySubTileTokens &&
                             embed <= streamVMaxEmbed && kvCopyInfo != nullptr && kvCopyInfo[0].count > 0U &&
                             kvCopyInfo[0].count <= maxPlanRecords && kvCopyInfo[1].count > 0U &&
                             kvCopyInfo[1].count <= maxPlanRecords;
        // Both output panels use the same full P tile in L1.
        const auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, stackSeqTile);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        uint32_t nL1Loop = CeilDiv<L1TileShape::N>(stackSeqTile);

        for (uint32_t blockStackIdx = 0; blockStackIdx < nL1Loop; ++blockStackIdx) {
            if (streamV && blockStackIdx != 0U) {
                break;
            }
            getBlockShape(actualShape, blockStackIdx, nL1Loop, stackSeqTile, L1TileShape::N);
            uint32_t kActual = actualShape.k();
            uint32_t nActual = actualShape.n();

            uint32_t processSize = 0;
            uint32_t nBlockOffset = nIdx * blockSize + blockStackIdx * L1TileShape::N;
            const bool yFitsPage = y != 0U && pageSize == 128U && (pageSize % y) == 0U;
            const uint32_t firstSlot = nBlockOffset / y;
            const __gm__ int32_t* selectIdx = gSelectIdx.GetPhyAddr();
            const __gm__ int32_t* blockPages = nullptr;
            uint32_t cachedLogicalPage = 0xFFFFFFFFU;
            uint32_t cachedPhysicalPage = 0U;
            if constexpr (PAGED_CACHE_FLAG_) {
                blockPages = gBlockTable.GetPhyAddr();
            }
            uint32_t fastBlockCount = 0U;
            const bool copyGeometryOk = PAGED_CACHE_FLAG_ ? yFitsPage : (y != 0U && y <= 128U);
            if (L1TileShape::N == 128U && strideKV < 65536U && copyGeometryOk && (nBlockOffset % y) == 0U &&
                firstSlot < localExplicitBlockCount) {
                const uint32_t explicitAvail = localExplicitBlockCount - firstSlot;
                const uint32_t byTokens = kActual / y;
                fastBlockCount = byTokens < explicitAvail ? byTokens : explicitAvail;
                if (fastBlockCount != 0U &&
                    (static_cast<uint64_t>(selectIdx[firstSlot + fastBlockCount - 1U]) + 1U) * y > kvSeqlen) {
                    fastBlockCount = 0U;
                }
            }
            uint32_t currentSelectYIdx = firstSlot;
            uint32_t currentYoffset = nBlockOffset % y;
            uint32_t currentYIdx = 0U;
            uint64_t offsetInKV = 0;

            auto* subCopyInfo = kvCopyInfo == nullptr ? nullptr : kvCopyInfo + blockStackIdx;
            if (fastBlockCount != 0U && subCopyInfo != nullptr && subCopyInfo->count <= fastBlockCount) {
                for (uint32_t ri = 0; ri < subCopyInfo->count; ++ri) {
                    const auto& r = subCopyInfo->records[ri];
                    GbsaSparsePairCopy::Emit<false>(l1BTensor, gB, layoutBInL1,
                                                    blockStackIdx * L1TileShape::N + r.dstRow, r.source, r.rows,
                                                    r.ndNum, r.stride, nActual, strideKV);
                }
                processSize = fastBlockCount * y;
                currentSelectYIdx = firstSlot + fastBlockCount;
                currentYoffset = 0U;
            } else if (fastBlockCount != 0U) {
                if constexpr (PAGED_CACHE_FLAG_) {
                    GbsaSparsePairCopy::Load<false>(l1BTensor, gB, layoutBInL1, gSelectIdx, gBlockTable, firstSlot,
                                                    fastBlockCount, blockStackIdx * L1TileShape::N, nActual, strideKV,
                                                    pageStride, peerPageStride, y, kvYBlockNum, kvSeqlen);
                } else {
                    GbsaSparsePairCopy::LoadDense<false>(l1BTensor, gB, layoutBInL1, gSelectIdx, firstSlot,
                                                         fastBlockCount, blockStackIdx * L1TileShape::N, nActual,
                                                         strideKV, y);
                }
                processSize = fastBlockCount * y;
                currentSelectYIdx = firstSlot + fastBlockCount;
                currentYoffset = 0U;
            }
            if (processSize < kActual && currentSelectYIdx < selectNum) {
                currentYIdx = GbsaSparseGeometry::ReadSparseBlockId<HAS_RESIDUAL_BLOCK>(
                    selectIdx, currentSelectYIdx, localExplicitBlockCount, residualBlockId);
                offsetInKV = static_cast<uint64_t>(currentYIdx) * y + currentYoffset;
                while (processSize < kActual && currentSelectYIdx < selectNum && currentYIdx < kvYBlockNum &&
                       offsetInKV < kvSeqlen) {
                    uint32_t yAcutal =
                        (currentSelectYIdx == selectNum - 1 && currentYIdx == kvYBlockNum - 1 && kvSeqlen % y != 0) ?
                            (kvSeqlen - y * currentYIdx) :
                            y;
                    uint32_t remainingInYBlock = yAcutal - currentYoffset;
                    uint32_t remainingInNBlock = kActual - processSize;

                    uint32_t actualYSize = min(remainingInNBlock, remainingInYBlock);
                    if (actualYSize == 0) {
                        break;
                    }

                    auto layoutBTile = layoutB.GetTileLayout(MakeCoord(actualYSize, nActual));
                    MatrixCoord l1BTileCoord{blockStackIdx * L1TileShape::N + processSize, 0};
                    auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];

                    uint64_t sourceOffset = offsetInKV * strideKV;
                    if constexpr (PAGED_CACHE_FLAG_) {
                        const uint32_t pageIdx = static_cast<uint32_t>(offsetInKV / pageSize);
                        if (pageIdx != cachedLogicalPage) {
                            cachedLogicalPage = pageIdx;
                            cachedPhysicalPage = static_cast<uint32_t>(blockPages[pageIdx]);
                        }
                        sourceOffset =
                            static_cast<uint64_t>(cachedPhysicalPage) * pageStride + (offsetInKV % pageSize) * strideKV;
                    }
                    copyGmToL1B(l1BTile, gB[sourceOffset], layoutBInL1, layoutBTile);

                    processSize += actualYSize;
                    currentYoffset += actualYSize;
                    offsetInKV += actualYSize;

                    if (currentYoffset >= yAcutal) {
                        currentSelectYIdx++;
                        if (currentSelectYIdx >= selectNum) {
                            break;
                        }
                        currentYoffset = 0;
                        currentYIdx = GbsaSparseGeometry::ReadSparseBlockId<HAS_RESIDUAL_BLOCK>(
                            selectIdx, currentSelectYIdx, localExplicitBlockCount, residualBlockId);
                        offsetInKV = static_cast<uint64_t>(currentYIdx) * y;
                    }
                }
            }
            if constexpr (groupedV) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4 + blockStackIdx);
                // P shares MTE2 on Arch22. Queue it after the first V group so
                // MTE1 can start PV while the remaining V groups arrive.
                if (blockStackIdx == 0U) {
                    LoadP(gA, layoutAInL1, layoutA.GetTileLayout(MakeCoord(rowNum, stackSeqTile)), softmaxFlag);
                }
            }
        }

        if constexpr (!groupedV) {
            if (streamV) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            }
            LoadP(gA, layoutAInL1, layoutA.GetTileLayout(MakeCoord(rowNum, stackSeqTile)), softmaxFlag);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1PPingPongFlag);

        const uint32_t nL0Loop = CeilDiv<L0TileShape::N>(embed);
        const uint32_t kL0Loop = CeilDiv<L0TileShape::K>(stackSeqTile);
        const uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        for (uint32_t kL0Idx = 0U; kL0Idx < kL0Loop; ++kL0Idx) {
            const uint32_t kOffset = kL0Idx * L0TileShape::K;
            const uint32_t kL0Actual = min(L0TileShape::K, stackSeqTile - kOffset);
            const uint32_t l0AStage = l0APingPongFlag;
            const auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(rowNum, kL0Actual);
            auto l1ATile = l1ATensor[l1PPingPongFlag][layoutAInL1.GetOffset(MatrixCoord{0U, kOffset})];
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AStage);
            copyL1ToL0A(l0ATensor[l0AStage], l1ATile, layoutAInL0, layoutAInL1);
            if (kL0Idx + 1U == kL0Loop) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1PPingPongFlag);
            }
            if constexpr (groupedV) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4 + kOffset / copySubTileTokens);
            } else if (streamV && kOffset == copySubTileTokens) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
            }

            // Hold the P panel in L0A until all N panels finish reading it.
            for (uint32_t nL0Idx = 0U; nL0Idx < nL0Loop; ++nL0Idx) {
                const uint32_t nOffset = nL0Idx * L0TileShape::N;
                const uint32_t nL0Actual = min(L0TileShape::N, embed - nOffset);
                const uint32_t l0BStage = l0BPingPongFlag;
                const uint32_t l0CStage = packL0C ? l0CPingPongFlag : (l0CPingPongFlag + nL0Idx) % STAGES;
                const auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nL0Actual);
                auto l1BTile = l1BTensor[layoutBInL1.GetOffset(MatrixCoord{kOffset, nOffset})];
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BStage + 2U);
                copyL1ToL0B(l0BTensor[l0BStage], l1BTile, layoutBInL0, layoutBInL1);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                if (kL0Idx == 0U && (!packL0C || nL0Idx == 0U)) {
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CStage);
                }
                auto l0CTile = l0CTensor[l0CStage][packL0C ? nOffset * mRound : 0U];
                tileMmad(l0CTile, l0ATensor[l0AStage], l0BTensor[l0BStage], mRound, nL0Actual, kL0Actual, kL0Idx == 0U);
                if (streamV && kL0Idx == 0U && nL0Idx == 0U) {
                    const auto& plan = kvCopyInfo[1];
                    for (uint32_t ri = 0U; ri < plan.count; ++ri) {
                        const auto& r = plan.records[ri];
                        GbsaSparsePairCopy::Emit<false>(l1BTensor, gB, layoutBInL1, copySubTileTokens + r.dstRow,
                                                        r.source, r.rows, r.ndNum, r.stride, embed, strideKV);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID4);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BStage + 2U);
                l0BPingPongFlag = 1U - l0BStage;
            }
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AStage);
            l0APingPongFlag = 1U - l0AStage;
        }
        l1PPingPongFlag = 1U - l1PPingPongFlag;

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        const uint32_t nFixLoop = packL0C ? 1U : nL0Loop;
        for (uint32_t nFixIdx = 0U; nFixIdx < nFixLoop; ++nFixIdx) {
            const uint32_t nOffset = nFixIdx * L0TileShape::N;
            const uint32_t nActual = packL0C ? embed : min(L0TileShape::N, embed - nOffset);
            const uint32_t l0CStage = (l0CPingPongFlag + nFixIdx) % STAGES;
            auto layoutCTile = layoutC.GetTileLayout(MakeCoord(rowNum, RoundUp(nActual, 8U)));
            auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(rowNum, nActual));
            copyL0CToGm(gC[layoutC.GetOffset(MatrixCoord{0U, nOffset})], l0CTensor[l0CStage], layoutCTile, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CStage);
        }
        l0CPingPongFlag = (l0CPingPongFlag + nFixLoop) % STAGES;
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor;
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    uint32_t l1PPingPongFlag = 0;
    uint32_t l0CPingPongFlag = 0;
    uint32_t l0APingPongFlag = 0;
    uint32_t l0BPingPongFlag = 0;
};

////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_MMAD_SFAI_PV_HPP
