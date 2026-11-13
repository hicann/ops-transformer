/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @brief Transposed (DN) counterpart of the QK matmul: computes K * Q^T = S^T.
 * Q stays resident on L1 as the B matrix; K base tiles are gathered into L1 as
 * the A matrix one by one. S^T leaves L0C through a SPLIT_N fixpipe, so each
 * vector sub-block receives [kvTile, qPerSubCore] row-major in its UB.
 */
#ifndef GEMM_BLOCK_QK_TRANS_ARCH35_ABF16_C2UB_HPP
#define GEMM_BLOCK_QK_TRANS_ARCH35_ABF16_C2UB_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/arch/gbsa_cross_core_sync.hpp"
#include "../../../attn_infra/gbsa_coord.hpp"
#include "../../../attn_infra/gemm/gbsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/gbsa_helper.hpp"
#include "../../../attn_infra/gbsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/gbsa_tile_mmad.hpp"
#include "../../../tla/gbsa_tla_layout.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"
#include "../../../arch35/gbsa_layout_utils.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <bool PAGED_CACHE_FLAG_, bool HAS_RESIDUAL_BLOCK_, bool PRECOMPUTE_SPARSE_, class L1TileShape_,
          class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_, class TileCopy_,
          class TileMmad_>
struct BlockMmadTla<MmadAtlasA5UnionQK<PAGED_CACHE_FLAG_, HAS_RESIDUAL_BLOCK_, PRECOMPUTE_SPARSE_>, L1TileShape_,
                    L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5UnionQK<PAGED_CACHE_FLAG_, HAS_RESIDUAL_BLOCK_, PRECOMPUTE_SPARSE_>;
    static constexpr bool PAGED_CACHE_FLAG = DispatchPolicy::PAGED_CACHE_FLAG;
    static constexpr bool HAS_RESIDUAL_BLOCK = DispatchPolicy::HAS_RESIDUAL_BLOCK;
    static constexpr bool PRECOMPUTE_SPARSE = DispatchPolicy::PRECOMPUTE_SPARSE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    // ElementA_/ElementB_ keep the interface meaning (Q / K); the cube operands
    // are swapped: A operand carries K, B operand carries Q.
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / L0_STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / L0_STAGES;
    static constexpr uint32_t L0C_HALF_BUF_SIZE = ArchTag::L0C_SIZE / 2;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_HALF_BUF_SIZE / L0_STAGES;

    static constexpr uint32_t MAX_L1_STAGES = 3; // 编译期常量，为静态L1Tensor数组开辟准备。取一个buffer份数的极大值
    static constexpr uint32_t V0_V1_FLAG_ID_OFFSET = 16; // 核间同步mode4，AIC侧需要两个flagId分别对应两个AIV

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, Mm1L1TileHelper& mm1L1TileHelper)
    {
        // M/N roles swap against the helper: A side tiles KV, B side tiles Q.
        l1ATileM = mm1L1TileHelper.mm1L1TileN;
        l1BTileN = mm1L1TileHelper.mm1L1TileM;
        l1ATileK = mm1L1TileHelper.mm1L1TileKRight;
        l1BTileK = mm1L1TileHelper.mm1L1TileKLeft;
        l1ABufNum = mm1L1TileHelper.kL1BufNum;
        l1BBufNum = mm1L1TileHelper.qL1BufNum;

        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1KTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementB>(l1ATileM * l1ATileK * sizeof(ElementB) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1QTensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1ATileM * l1ATileK * sizeof(ElementB) * l1ABufNum + l1BTileK * l1BTileN * sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementB>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmadTla() {}

    template <class TensorQ>
    __aicore__ inline void loadQGM(TensorQ& gQTensor, GemmCoord actualOriShape)
    {
        // Q copied to L1 as the B matrix, layout [embed, qSTile].
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorQ>;
        CopyGmToL1B copyGmToL1B;
        uint32_t qSTile = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        auto l1BLayoutTla = tla::MakeLayout<ElementA, LayoutTagL1B>(embed, qSTile);
        auto l1BTensorTla = tla::MakeTensor(l1QTensor[0], l1BLayoutTla, Arch::PositionL1{});
        auto l1BTensorTlaTile = GetTile(l1BTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(embed, qSTile));
        auto gQTensorTlaTile = GetTile(gQTensor, tla::MakeCoord(0, 0), tla::MakeShape(embed, qSTile));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1B(l1BTensorTlaTile, gQTensorTlaTile);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    // Gather each token's GQA group into packed L1 columns [t*groupSize, ...).
    // BNSD heads are separated by the storage sequence capacity times D.
    template <class GlobalQ>
    __aicore__ inline void loadQGMTokenGroups(GlobalQ& gQ, const uint64_t* tokenOffsets, uint32_t tokenNum,
                                              uint32_t groupSize, uint32_t embed, uint64_t headStride)
    {
        using LayoutQ = layout::ColumnMajor;
        const uint32_t qSTile = tokenNum * groupSize;
        auto l1BLayoutTla = tla::MakeLayout<ElementA, LayoutTagL1B>(embed, qSTile);
        auto l1BTensorTla = tla::MakeTensor(l1QTensor[0], l1BLayoutTla, Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        for (uint32_t t = 0; t < tokenNum; ++t) {
            auto gmQTensor = tla::MakeTensor(
                gQ[tokenOffsets[t]], tla::MakeLayout<ElementA, LayoutQ>(headStride, groupSize), Arch::PositionGM{});
            using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<decltype(gmQTensor)>;
            CopyGmToL1B copyGmToL1B;
            auto l1Tile = GetTile(l1BTensorTla, tla::MakeCoord(0, t * groupSize), tla::MakeShape(embed, groupSize));
            auto gTile = GetTile(gmQTensor, tla::MakeCoord(0, 0), tla::MakeShape(embed, groupSize));
            // Serialize GM->L1 gathers: consecutive MTE2 into shared L1 without a
            // barrier can drop earlier token tiles when kvHeads>1 packs multiple
            // non-contiguous Q groups. (No MTE2_MTE2 hard-event on this arch.)
            if (t > 0U) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            }
            copyGmToL1B(l1Tile, gTile);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    __aicore__ inline uint32_t GetCurLoopCounter(uint32_t outterLoopItr, uint32_t curLoopNum, uint32_t curLoopItr)
    {
        return outterLoopItr * curLoopNum + curLoopItr;
    }

    // Gather one K L1 tile (kv rows x embed slice) into the A-side L1 buffer.
    // Mirrors SparseKL1TileNLoad of the non-transposed mmad with A/B swapped;
    // the precomputed path issues one layout-generic copy per offset record.
    template <class TensorK, class TensorL1A>
    __aicore__ inline void SparseKL1TileMLoad(TensorK& gKTensor, TensorL1A& l1ATensorTla,
                                              AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                              uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile, uint32_t blockShapeY,
                                              uint32_t l1ATileMAct, uint32_t l1ATileKAct,
                                              uint32_t kvSBaseTileInnerOffset, uint32_t embedInnerOffset,
                                              AscendC::GlobalTensor<int32_t> gBlockTable, uint32_t blockSize,
                                              uint64_t pageStride, uint64_t tokenStride,
                                              uint32_t localExplicitBlockCount, uint32_t residualBlockId,
                                              AscendC::GlobalTensor<uint64_t> kvOffsetInfo, uint32_t kvOffsetGroupWords)
    {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorK>;
        CopyGmToL1A copyGmToL1A;
        if constexpr (PRECOMPUTE_SPARSE) {
            constexpr uint32_t recordWords = PAGED_CACHE_FLAG ? 4U : 2U;
            auto groupInfo = kvOffsetInfo[kvSBaseTileInnerOffset / l1ATileM * kvOffsetGroupWords];
            // Headers and first record lines are refreshed/prefetched after V0 Ready.
            // Extract the physical base once, not on every scalar descriptor load.
            const __gm__ uint64_t* records = groupInfo.GetPhyAddr();
            const uint32_t blockCount = records[0];
            constexpr uint32_t cacheLineWords = 8U;
            const uint32_t recordEnd = 8U + blockCount * recordWords;
            if (embedInnerOffset == 0U) {
                for (uint32_t word = 2U * cacheLineWords; word < recordEnd; word += cacheLineWords) {
                    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                                      AscendC::DcciDst::CACHELINE_OUT>(groupInfo[word]);
                }
                // Warm all remaining descriptor lines once; the copy loop needs
                // no per-record cache-line/window bookkeeping.
                for (uint32_t word = 2U * cacheLineWords; word < recordEnd; word += cacheLineWords) {
                    AscendC::DataCachePreload(groupInfo[word], static_cast<int64_t>(0));
                }
            }
            const auto& srcData = gKTensor.data();
            uint32_t dealtLenAccum = 0;
            for (uint32_t i = 0; i < blockCount; ++i) {
                const uint32_t word = 8U + i * recordWords;
                const uint64_t sourceOffset = records[word];
                const uint64_t lengthOrMeta = records[word + (PAGED_CACHE_FLAG ? 2U : 1U)];
                const uint32_t curDealtLen = static_cast<uint32_t>(lengthOrMeta) >> 1U;
                const uint32_t ndNum = 1U + static_cast<uint32_t>(lengthOrMeta & 1U);
                const uint64_t srcMatrixStride = lengthOrMeta >> 32U;
                for (uint32_t r = 0; r < ndNum; ++r) {
                    auto l1ATensorTlaTile = GetTile(l1ATensorTla, tla::MakeCoord(dealtLenAccum, 0U),
                                                    tla::MakeShape(curDealtLen, l1ATileKAct));
                    auto gKSourceTensor =
                        tla::MakeTensor(srcData[sourceOffset + r * srcMatrixStride + embedInnerOffset],
                                        gKTensor.layout(), Arch::PositionGM{});
                    auto gKTensorTlaTile =
                        GetTile(gKSourceTensor, tla::MakeCoord(0, 0), tla::MakeShape(curDealtLen, l1ATileKAct));
                    copyGmToL1A(l1ATensorTlaTile, gKTensorTlaTile);
                    dealtLenAccum += curDealtLen;
                }
            }
            return;
        }
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile + kvSBaseTileInnerOffset;
        uint32_t yBlockInnerStartOffset = baseTileStartOffset % blockShapeY;
        uint32_t dealtLenAccum = 0;
        // Tile length has already been clipped to validKvTokens. Thus every
        // sparse index and token visited here is valid, including the last block.
        for (uint32_t gatheredYBlockIdx = baseTileStartOffset / blockShapeY; dealtLenAccum < l1ATileMAct;
             ++gatheredYBlockIdx) {
            const uint32_t curYBlockSize = min(blockShapeY - yBlockInnerStartOffset, l1ATileMAct - dealtLenAccum);
            const int32_t logicalBlockId = GbsaKernelArch35::GetSparseBlockId<HAS_RESIDUAL_BLOCK>(
                gSparseBlockIdx, gatheredYBlockIdx, localExplicitBlockCount, residualBlockId);
            const uint64_t oriStartOffset =
                static_cast<uint64_t>(logicalBlockId) * blockShapeY + yBlockInnerStartOffset;
            // Only PA can split one sparse block across several physical pages.
            for (uint32_t copied = 0; copied < curYBlockSize;) {
                uint32_t curDealtLen = curYBlockSize - copied;
                uint64_t sourceToken = oriStartOffset + copied;
                uint64_t pageOffset = 0;
                if constexpr (PAGED_CACHE_FLAG) {
                    const uint32_t pageInnerOffset = sourceToken % blockSize;
                    curDealtLen = min(curDealtLen, blockSize - pageInnerOffset);
                    const uint32_t physicalPage = gBlockTable.GetValue(sourceToken / blockSize);
                    pageOffset = static_cast<uint64_t>(physicalPage) * pageStride;
                    sourceToken = pageInnerOffset;
                }
                auto l1ATensorTlaTile = GetTile(l1ATensorTla, tla::MakeCoord(dealtLenAccum + copied, 0U),
                                                tla::MakeShape(curDealtLen, l1ATileKAct));
                // GM is already rebased to the batch/head or physical block, with zero tensor coordinates.
                const uint64_t sourceOffset = pageOffset + sourceToken * tokenStride + embedInnerOffset;
                auto gKSourceTensor =
                    tla::MakeTensor(gKTensor.data()[sourceOffset], gKTensor.layout(), Arch::PositionGM{});
                auto gKTensorTlaTile =
                    GetTile(gKSourceTensor, tla::MakeCoord(0, 0), tla::MakeShape(curDealtLen, l1ATileKAct));
                copyGmToL1A(l1ATensorTlaTile, gKTensorTlaTile);
                copied += curDealtLen;
            }
            dealtLenAccum += curYBlockSize;
            yBlockInnerStartOffset = 0;
        }
    }

    __aicore__ inline void PrefetchKvOffsetGroups(AscendC::GlobalTensor<uint64_t> info, uint32_t groupWords,
                                                  uint32_t begin, uint32_t end)
    {
        // Do not let a later DCCI drain earlier prefetch requests.
        for (uint32_t group = begin; group < end; ++group) {
            auto groupInfo = info[group * groupWords];
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(groupInfo);
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                              AscendC::DcciDst::CACHELINE_OUT>(groupInfo[8U]);
        }
        for (uint32_t group = begin; group < end; ++group) {
            auto groupInfo = info[group * groupWords];
            AscendC::DataCachePreload(groupInfo, static_cast<int64_t>(0));
            AscendC::DataCachePreload(groupInfo[8U], static_cast<int64_t>(0));
        }
    }

    template <class TensorK, class TensorC>
    __aicore__ inline void operator()(TensorK& gKTensor, TensorC& ubCTensor,
                                      AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape,
                                      uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile, uint32_t blockShapeY,
                                      uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages,
                                      Arch::CrossCoreFlag mm1ToSmFlag, uint64_t tokenStride,
                                      AscendC::GlobalTensor<int32_t> gBlockTable = {}, uint32_t blockSize = 0,
                                      uint64_t pageStride = 0, uint32_t localExplicitBlockCount = 0,
                                      uint32_t residualBlockId = 0, AscendC::GlobalTensor<uint64_t> kvOffsetInfo = {},
                                      uint32_t kvOffsetGroupWords = 0, uint32_t sparseReadyFlag = 0)
    {
        uint32_t qSTile = actualOriShape[0];
        uint32_t embed = actualOriShape[2];
        uint32_t curBaseTileSize = actualOriShape[1];

        auto l1QLayoutTla = tla::MakeLayout<ElementA, LayoutTagL1B>(embed, qSTile);
        auto l1QTensorTla = tla::MakeTensor(l1QTensor[0], l1QLayoutTla, Arch::PositionL1{});

        uint32_t mL1LoopNum = CeilDiv(curBaseTileSize, l1ATileM);
        uint32_t kL1LoopNum = CeilDiv(embed, l1ATileK);
        // Non-quant tiling guarantees groupSize <= 128 and D <= 256.
        // Fold the single N slice and keep both Q K-slices resident in L0B.
        const uint32_t nL0LoopNum = CeilDiv(qSTile, L0_TILE_N);
        uint32_t totalML0LoopNum = CeilDiv(kvSBaseTile, L0_TILE_M);
        uint32_t totalKL0LoopNum = CeilDiv(embed, L0_TILE_K);
        uint32_t fullML0LoopNum = CeilDiv(l1ATileM, L0_TILE_M);
        uint32_t fullKL0LoopNum = CeilDiv(l1ATileK, L0_TILE_K);

        const bool l1ToL0BNoRepeatFlag = PRECOMPUTE_SPARSE || ((nL0LoopNum == 1U) && (totalKL0LoopNum <= L0_STAGES));

        if constexpr (PRECOMPUTE_SPARSE) {
            // Consume round 0 once per attention tile, outside the L1/D loops.
            const uint32_t readyFlag =
                mL1LoopNum > 2U ? sparseReadyFlag + GbsaKernelArch35::KV_OFFSET_STAGES : sparseReadyFlag;
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(readyFlag);
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(readyFlag + V0_V1_FLAG_ID_OFFSET);
            PrefetchKvOffsetGroups(kvOffsetInfo, kvOffsetGroupWords, 0U, min(2U, mL1LoopNum));
        }

        // while splitting the base tile S to 2 AIVs,
        // the order of the elements in each row is expected to be preserved,
        // which means a row in l0C cannot be chunked and processed by dualMode FixPipe seperately.
        // therefore, FixPipe won't launch until each portion(chunked only by rows, based on nbuffer strategy)
        // of the base tile is ready on l0C
        for (uint32_t mL1Itr = 0; mL1Itr < mL1LoopNum; mL1Itr++) {
            uint32_t l1TileMAct = (mL1Itr == mL1LoopNum - 1) ? (curBaseTileSize - mL1Itr * l1ATileM) : l1ATileM;
            for (uint32_t kL1Itr = 0; kL1Itr < kL1LoopNum; ++kL1Itr) {
                uint32_t l1TileKAct = (kL1Itr == kL1LoopNum - 1) ? (embed - kL1Itr * l1ATileK) : l1ATileK;
                uint32_t kLoopCounterL1 = (gatheredKvSTileIdx * mL1LoopNum + mL1Itr) * kL1LoopNum + kL1Itr;
                // Non-quantized tiling allocates one or two K buffers.
                uint32_t l1ABufId =
                    PRECOMPUTE_SPARSE ? (kLoopCounterL1 & (l1ABufNum - 1U)) : (kLoopCounterL1 % l1ABufNum);
                uint32_t l1AEventId = l1ABufId + 1;
                auto l1ALayoutTla = tla::MakeLayout<ElementB, LayoutTagL1A>(l1TileMAct, l1TileKAct);
                auto l1ATensorTla = tla::MakeTensor(l1KTensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventId);
                SparseKL1TileMLoad(gKTensor, l1ATensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSBaseTile,
                                   blockShapeY, l1TileMAct, l1TileKAct, mL1Itr * l1ATileM, kL1Itr * l1ATileK,
                                   gBlockTable, blockSize, pageStride, tokenStride, localExplicitBlockCount,
                                   residualBlockId, kvOffsetInfo, kvOffsetGroupWords);
                if constexpr (PRECOMPUTE_SPARSE) {
                    if (mL1LoopNum > 2U && mL1Itr == 1U && kL1Itr == kL1LoopNum - 1U) {
                        // Round 0's K copies are already submitted. Prefetch round 1
                        // during their transfer/compute instead of on group 2's load path.
                        AscendC::CrossCoreWaitFlag<4, PIPE_S>(sparseReadyFlag);
                        AscendC::CrossCoreWaitFlag<4, PIPE_S>(sparseReadyFlag + V0_V1_FLAG_ID_OFFSET);
                        PrefetchKvOffsetGroups(kvOffsetInfo, kvOffsetGroupWords, 2U, mL1LoopNum);
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);

                uint32_t mL0LoopNum = CeilDiv(l1TileMAct, L0_TILE_M);
                uint32_t kL0LoopNum = CeilDiv(l1TileKAct, L0_TILE_K);
                for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                    uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (l1TileMAct - mL0Itr * L0_TILE_M) : L0_TILE_M;
                    uint32_t globalML0Itr = mL1Itr * fullML0LoopNum + mL0Itr;
                    uint32_t l0CLoopCounter = gatheredKvSTileIdx * totalML0LoopNum + globalML0Itr;
                    uint32_t l0CBufId = l0CLoopCounter % L0_STAGES;
                    uint32_t l0CEventId = l0CBufId;
                    auto l0CLayoutTla = tla::MakeLayoutL0C(l0TileMAct, qSTile);
                    auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});
                    for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
                        uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (qSTile - nL0Itr * L0_TILE_N) : L0_TILE_N;
                        auto l0CTensorTlaTile = GetTile(l0CTensorTla, tla::MakeCoord(0, nL0Itr * L0_TILE_N),
                                                        tla::MakeShape(l0TileMAct, l0TileNAct));
                        for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                            uint32_t l0TileKAct =
                                (kL0Itr == kL0LoopNum - 1) ? (l1TileKAct - kL0Itr * L0_TILE_K) : L0_TILE_K;
                            uint32_t globalKL0Itr = kL1Itr * fullKL0LoopNum + kL0Itr;
                            uint32_t l0ALoopCounter =
                                prefixSumL0AStages + globalML0Itr * totalKL0LoopNum + globalKL0Itr;
                            uint32_t l0BLoopCounter = prefixSumL0BStages + nL0Itr * totalKL0LoopNum + globalKL0Itr;
                            uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                            uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                            uint32_t l0AEventId = l0ABufId;
                            uint32_t l0BEventId = l0BBufId + 2;

                            auto l1ATensorTlaTile =
                                GetTile(l1ATensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, kL0Itr * L0_TILE_K),
                                        tla::MakeShape(l0TileMAct, l0TileKAct));
                            auto l0ALayoutTla = tla::MakeLayout<ElementB, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                            auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                            copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            if (nL0Itr == nL0LoopNum - 1U && mL0Itr == mL0LoopNum - 1U && kL0Itr == kL0LoopNum - 1U) {
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventId);
                            }

                            auto l1QTensorTlaTile =
                                GetTile(l1QTensorTla,
                                        tla::MakeCoord(kL1Itr * l1ATileK + kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                                        tla::MakeShape(l0TileKAct, l0TileNAct));
                            auto l0BLayoutTla = tla::MakeLayout<ElementA, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                            auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});
                            const bool firstM = (mL1Itr == 0U) && (mL0Itr == 0U);
                            const bool lastM = (mL1Itr == mL1LoopNum - 1U) && (mL0Itr == mL0LoopNum - 1U);
                            if (!l1ToL0BNoRepeatFlag || firstM) {
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                                copyL1ToL0B(l0BTensorTla, l1QTensorTlaTile);
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            }

                            if (!l1ToL0BNoRepeatFlag || firstM) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            }
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            if (nL0Itr == 0U && kL1Itr == 0U && kL0Itr == 0U) {
                                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                            }
                            bool initMmad = (kL1Itr == 0U) && (kL0Itr == 0U);
                            uint32_t l0TileMAligned = RoundUp(l0TileMAct, 16);
                            tileMmad(l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, l0TileMAligned, l0TileNAct,
                                     l0TileKAct, initMmad);
                            if (!l1ToL0BNoRepeatFlag || lastM) {
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        }
                    }

                    if (kL1Itr == kL1LoopNum - 1U) {
                        // All L1 M tiles share one S buffer and one release token.
                        // Acquire it only before the first Fixpipe of the whole KV tile.
                        if (globalML0Itr == 0U) {
                            WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
                        uint32_t mFixPAligned8 = RoundUp(l0TileMAct, 8);
                        uint32_t nFixPAligned8 = RoundUp(qSTile, 8);
                        // S covers the entire KV tile, including all L1 M slices.
                        const uint32_t sRowOffset = mL1Itr * l1ATileM + mL0Itr * L0_TILE_M;
                        auto ubCTensorTlaTile = GetTile(ubCTensor, tla::MakeCoord(sRowOffset, 0),
                                                        tla::MakeShape(mFixPAligned8, nFixPAligned8));
                        CopyL0CToDst copyL0CToDst;
                        copyL0CToDst(ubCTensorTlaTile, l0CTensorTla);
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                    }
                }
            }
        }
        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementB> l1KTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementA> l1QTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementB> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementA> l0BTensor[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1ABufNum;
    uint32_t l1BBufNum;
};
////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block
#endif
