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
 * @brief matmul implementation for single p&v base tile
 * This implementation is designed for the following senario:
 * A full p base tile is loaded to L1 from UB, no workspace transit
 * A full v base tile is loaded to L1 from GM，relavent instructions launched before p base tile crossCore wait
 * A full p*v base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_PV_ARCH35_ABF16_C2UB_HIF8_HPP
#define GEMM_BLOCK_PV_ARCH35_ABF16_C2UB_HIF8_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_arch35_utils.hpp"
#include "../../../attn_infra/gemm/block/bsa_block_mmad_pv_arch35_ABf16_C_to_UB.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_,
          class TileCopy_, class TileMmad_, bool transposedMm1>
struct BlockMmadTla<
    // if transposedMm1, P is actually nZ in UB, and nZ in L1,
    // and is transposed to zN while loaded to L0A.
    MmadAtlasA5BsaPVFp8<transposedMm1>, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_,
    TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5BsaPVFp8<transposedMm1>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
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

    // Plan C keeps PV unsplit along K. One full K tile is processed per MMAD,
    // while the two micro-slots preserve the MM1/PV L0 buffer and event mapping.
    static constexpr uint32_t L0_MICRO_STAGES = 2;
    static constexpr uint32_t L0_TILE_K_MICRO = L0_TILE_K;
    static constexpr uint32_t L0A_MICRO_BUF_SIZE = ArchTag::L0A_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static constexpr uint32_t L0B_MICRO_BUF_SIZE = ArchTag::L0B_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static constexpr uint32_t L0C_MICRO_BUF_SIZE = ArchTag::L0C_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    // A full K tile occupies one complete outer slot.
    static_assert(L0A_PINGPONG_BUF_SIZE >= L0_TILE_M * L0_TILE_K_MICRO * sizeof(ElementA),
                  "L0 outer slot too small for the whole K tile of A (Plan C)");
    static_assert(L0B_PINGPONG_BUF_SIZE >= L0_TILE_K_MICRO * L0_TILE_N * sizeof(ElementB),
                  "L0 outer slot too small for the whole K tile of B (Plan C)");
    static_assert(L0C_MICRO_BUF_SIZE >= L0_TILE_M * L0_TILE_N * sizeof(ElementAccumulator),
                  "L0C micro slot too small for the OTmp tile");
    static constexpr uint32_t L0A_EVENT_BASE = 0;
    static constexpr uint32_t L0B_EVENT_BASE = L0_STAGES * L0_MICRO_STAGES; // 4

    // Keep the computed event range within the MTE1<->M queue depth.
    static_assert(L0B_EVENT_BASE + L0_STAGES * L0_MICRO_STAGES <= AscendC::QUE_MAX_EVENT,
                  "Plan B MM2 needs 8 event ids on the MTE1<->M queue; QUE_MAX_EVENT is smaller");

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart,
                                   Mm2L1TileHelper& mm2L1TileHelper)
    {
        l1ATileM = mm2L1TileHelper.mm2L1TileM;
        l1BTileN = mm2L1TileHelper.mm2L1TileN;
        l1ATileK = mm2L1TileHelper.mm2L1TileKLeft;
        l1BTileK = mm2L1TileHelper.mm2L1TileKRight;
        l1ABufNum = mm2L1TileHelper.pL1BufNum;
        l1BBufNum = mm2L1TileHelper.vL1BufNum;
        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart + l1ATileM * l1ATileK *
                                                                                                  sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum +
                l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        // Address L0A/L0B by outer slot and micro-slot index.
        for (uint32_t i = 0; i < L0_STAGES * L0_MICRO_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_MICRO_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_MICRO_BUF_SIZE * i);
        }
        // Keep the L0C slot mapping aligned with MM1; MM2 uses the first micro-slot.
        for (uint32_t i = 0; i < L0_STAGES * L0_MICRO_STAGES; i++) {
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_MICRO_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmadTla() {}

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

    template <class TensorB, class TensorL1B>
    __aicore__ inline void SparseVBaseTileL1FullLoad(TensorB& gBTensor, TensorL1B& l1BTensorTla,
                                                     AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                                     uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen,
                                                     uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval,
                                                     uint32_t yBlockNumRsvd, uint32_t curBaseTileSize, uint32_t embed)
    {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile;
        uint32_t baseTileEndOffset = baseTileStartOffset + curBaseTileSize;
        // Traverse selected sparse blocks to fill the current tile.
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
        uint32_t dealtLenAccum = 0;

        while (dealtLenAccum < curBaseTileSize && gatheredYBlockIdx < yBlockNumRsvd && oriYBlockIdx < yBlockNumAval &&
               oriStartOffset < kvSeqlen) {
            uint32_t curYBlockSize = blockShapeY;
            if (oriYBlockIdx == yBlockNumAval - 1) {
                curYBlockSize = kvSeqlen - oriYBlockIdx * blockShapeY;
            }
            uint32_t gatheredEndOffset = min(gatheredYBlockIdx * blockShapeY + curYBlockSize, baseTileEndOffset);
            uint32_t curDealtLen = gatheredEndOffset - gatheredStartOffset;
            if (curDealtLen == 0) {
                break;
            }
            auto l1BTensorTlaTile =
                GetTile(l1BTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
            auto gBTensorTlaTile =
                GetTile(gBTensor, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(curDealtLen, embed));
            copyGmToL1B(l1BTensorTlaTile, gBTensorTlaTile);
            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < curBaseTileSize) {
                // Avoid reading the sparse index after the final iteration.
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
            }
        }
    }

    template <class TensorB, class TensorC>
    __aicore__ inline void operator()(TensorB& gBTensor, TensorC& ubCTensor,
                                      AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape,
                                      uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen, uint32_t kvSBaseTile,
                                      uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                                      uint64_t& l0Stage, Arch::CrossCoreFlag smToMm2Flag,
                                      Arch::CrossCoreFlag mm2ToReFlag)
    {
        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;

        const uint32_t bufId = static_cast<uint32_t>(l0Stage++ % L0_STAGES);
        // Derive MTE1/M event ids from the outer and micro-slot indices; L0C uses FIX_M.
        const uint32_t l0AMicroBase = L0A_EVENT_BASE + bufId * L0_MICRO_STAGES;
        const uint32_t l0BMicroBase = L0B_EVENT_BASE + bufId * L0_MICRO_STAGES;
        const uint32_t l0CEventId = bufId * L0_MICRO_STAGES;

        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        uint32_t curBaseTileSize = actualOriShape[2];

        uint32_t l1BBufId = gatheredKvSTileIdx % l1BBufNum;
        uint32_t l1BEventId = l1BBufId + 3;
        uint32_t l1ABufId = gatheredKvSTileIdx % l1ABufNum;

        auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(curBaseTileSize, embed);
        auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, curBaseTileSize);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[l1ABufId], l1ALayoutTla, Arch::PositionL1{});
        // Load the full V tile to L1 before the cross-core wait.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
        SparseVBaseTileL1FullLoad(gBTensor, l1BTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile,
                                  blockShapeY, yBlockNumAval, yBlockNumRsvd, curBaseTileSize, embed);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
        // Wait for online softmax to publish P.
        WaitCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
        uint32_t mL0LoopNum = CeilDiv(rowNum, L0_TILE_M);
        uint32_t nL0LoopNum = CeilDiv(embed, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(curBaseTileSize, L0_TILE_K);
        // Preserve column order when splitting OTmp between AIVs; issue Fixpipe after each L0C portion is ready.
        for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
            uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (embed - nL0Itr * L0_TILE_N) : L0_TILE_N;
            auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, l0TileNAct);
            auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CEventId], l0CLayoutTla, Arch::PositionL0C{});
            for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                auto l0CTensorTlaTile = GetTile(l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0),
                                                tla::MakeShape(l0TileMAct, l0TileNAct));
                for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                    uint32_t l0TileKAct =
                        (kL0Itr == kL0LoopNum - 1) ? (curBaseTileSize - kL0Itr * L0_TILE_K) : L0_TILE_K;
                    uint32_t kMicroLoopNum = CeilDiv(l0TileKAct, L0_TILE_K_MICRO);
                    for (uint32_t microK = 0; microK < kMicroLoopNum; microK++) {
                        uint32_t microKAct =
                            (microK == kMicroLoopNum - 1) ? (l0TileKAct - microK * L0_TILE_K_MICRO) : L0_TILE_K_MICRO;
                        uint32_t microKOffset = kL0Itr * L0_TILE_K + microK * L0_TILE_K_MICRO;
                        // Rotate within the allocated micro-slots for K tails and reuse.
                        const uint32_t l0AEventId = l0AMicroBase + microK % L0_MICRO_STAGES;
                        const uint32_t l0BSlot = bufId * L0_MICRO_STAGES + microK % L0_MICRO_STAGES;
                        const uint32_t l0BEventId = l0BMicroBase + microK % L0_MICRO_STAGES;
                        bool lastK = (kL0Itr == kL0LoopNum - 1) && (microK == kMicroLoopNum - 1);

                        // Reuse L0B across the K loop when possible.
                        auto l1BTensorTlaTile = GetTile(l1BTensorTla, tla::MakeCoord(microKOffset, nL0Itr * L0_TILE_N),
                                                        tla::MakeShape(microKAct, l0TileNAct));
                        auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(microKAct, l0TileNAct);
                        auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BSlot], l0BLayoutTla, Arch::PositionL0B{});

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                        copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                        if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && lastK) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                        }

                        auto l1ATensorTlaTile = GetTile(l1ATensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, microKOffset),
                                                        tla::MakeShape(l0TileMAct, microKAct));
                        auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, microKAct);
                        auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0AEventId], l0ALayoutTla, Arch::PositionL0A{});

                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                        // reverse crossCoreSync for P
                        if ((mL0Itr == mL0LoopNum - 1) && (nL0Itr == nL0LoopNum - 1) && lastK) {
                            SetCrossCoreSync<4, PIPE_MTE1>(smToMm2Flag);
                        }

                        bool initMmad = (kL0Itr == 0) && (microK == 0);
                        uint32_t l0TileMAligned = RoundUp(l0TileMAct, 16);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);

                        if (mL0Itr == 0 && kL0Itr == 0 && microK == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                        }
                        tileMmad(l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, l0TileMAligned, l0TileNAct, microKAct,
                                 initMmad);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                    }
                }
            }
            // fixpipe
            if (nL0Itr == 0) {
                // reverse crossCoreSync, do fixPipe only after ubCTensor is fully released
                WaitCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);

            uint32_t mFixPAligned = 0;
            if constexpr (DispatchPolicy::transposedMm1) {
                mFixPAligned = RoundUp(rowNum, 32);
            } else {
                mFixPAligned = RoundUp(rowNum, 8);
            }
            uint32_t nFixPAligned8 = RoundUp(l0TileNAct, 8);
            // SPLIT_M + NO_QUANT: one dual-destination Fixpipe serves both AIVs, the hardware
            // writing rows 0..m/2-1 into AIV0 and m/2..m-1 into AIV1 at the same UB offsets
            // given by ubCTensor's per-AIV stride. The dst shape therefore carries the full M
            // (it becomes mSize) and the full N of the chunk; the dequantization is done by the
            // rescale-O epilogue instead of by the Fixpipe.
            CopyL0CToDst copyL0CToDst;
            auto ubCTensorTlaTile =
                GetTile(ubCTensor, tla::MakeCoord(0, nL0Itr * L0_TILE_N), tla::MakeShape(mFixPAligned, nFixPAligned8));
            copyL0CToDst(ubCTensorTlaTile, l0CTensorTla);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
        }
        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(mm2ToReFlag);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES * L0_MICRO_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES * L0_MICRO_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES * L0_MICRO_STAGES];

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
#endif // GEMM_BLOCK_PV_ARCH35_ABF16_C2UB_HIF8_HPP
