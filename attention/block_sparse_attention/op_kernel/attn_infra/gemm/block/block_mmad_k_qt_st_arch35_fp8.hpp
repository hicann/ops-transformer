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
 * @brief matmul implementation for single q&k^t base tile
 * This implementation is designed for the following senario:
 * A full q base tile is loaded to L1 from GM at the very beginning,
 * and it remains persistent until each k base tile is dealt
 * A full q*k^t base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GEMM_BLOCK_K_QT_ST_ARCH35_HIF8_HPP
#define GEMM_BLOCK_K_QT_ST_ARCH35_HIF8_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_arch35_utils.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"

////////////////////////////////////////////////////////////////////

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_,
          class TileCopy_,
          class TileMmad_>
struct BlockMmadTla<MmadAtlasA5BsaQKFp8<true>, // FP8 specialization for performing K * Q^t = S^t
                    L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA5BsaQKFp8<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementQ = ElementA_;
    using ElementK = ElementB_;
    using ElementS = ElementC_;

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
    static constexpr bool FULL_QUANT_HIF8 = AscendC::IsSameType<ElementQ, hifloat8_t>::value;

    // Split the M dimension into two micro-slots inside each outer double-buffer slot.
    static constexpr uint32_t L0_MICRO_STAGES = 2;
    static constexpr uint32_t L0_TILE_M_MICRO = L0_TILE_M / L0_MICRO_STAGES;
    static constexpr uint32_t L0A_MICRO_BUF_SIZE = ArchTag::L0A_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static constexpr uint32_t L0B_MICRO_BUF_SIZE = ArchTag::L0B_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static constexpr uint32_t L0C_MICRO_BUF_SIZE = ArchTag::L0C_SIZE / (L0_STAGES * L0_MICRO_STAGES);
    static_assert(L0_TILE_M % L0_MICRO_STAGES == 0, "L0 tile M must split evenly into micro slots");
    static_assert(L0A_MICRO_BUF_SIZE >= L0_TILE_M_MICRO * L0_TILE_K * sizeof(ElementK),
                  "L0A micro slot too small for one micro M block");
    static_assert(L0C_MICRO_BUF_SIZE >= L0_TILE_M_MICRO * L0_TILE_N * sizeof(ElementAccumulator),
                  "L0C micro slot too small for one micro M block");
    // Q is shared by both micro-slots and must fit in the first slot.
    static_assert(L0B_MICRO_BUF_SIZE >= L0_TILE_K * L0_TILE_N * sizeof(ElementQ),
                  "L0B micro slice must hold one whole Q tile (qSTile == L0_TILE_N)");

    // Keep L0A and L0B event ranges disjoint; L0C uses a separate queue.
    static constexpr uint32_t L0A_EVENT_BASE = 0;
    static constexpr uint32_t L0B_EVENT_BASE = L0_STAGES * L0_MICRO_STAGES; // 4

    // Keep the computed event range within the hardware queue depth.
    static_assert(L0B_EVENT_BASE + L0_STAGES * L0_MICRO_STAGES <= AscendC::QUE_MAX_EVENT,
                  "Plan B MM1 event ids exceed the MTE1<->M queue depth (QUE_MAX_EVENT)");

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, Mm1L1TileHelper& mm1L1TileHelper)
    {
        l1QBufNum = mm1L1TileHelper.qL1BufNum;
        l1KBufNum = mm1L1TileHelper.kL1BufNum;
        l1ATileM = mm1L1TileHelper.mm1L1TileN;
        l1BTileN = mm1L1TileHelper.mm1L1TileM;
        l1ATileK = mm1L1TileHelper.mm1L1TileKRight;
        l1BTileK = mm1L1TileHelper.mm1L1TileKLeft;
        for (uint32_t i = 0; i < l1QBufNum; i++) {
            l1QTensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementQ>(l1BTileK * l1BTileN * sizeof(ElementQ) * i);
        }
        for (uint32_t i = 0; i < l1KBufNum; i++) {
            l1KTensor[i] = resource.l1Buf.template GetBufferByByte<ElementK>(
                l1BTileK * l1BTileN * sizeof(ElementQ) * l1QBufNum + l1ATileM * l1ATileK * sizeof(ElementK) * i);
        }
        // Address all L0 buffers by outer slot and micro-slot index.
        for (uint32_t i = 0; i < L0_STAGES * L0_MICRO_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementQ>(L0A_MICRO_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementK>(L0B_MICRO_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_MICRO_BUF_SIZE * i);
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmadTla() {}

    template <class TensorQ>
    __aicore__ inline void loadQGM(TensorQ& gQTensor, GemmCoord actualOriShape)
    {
        uint32_t qSTile = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        // Q copied to L1 as B matrix
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorQ>;
        CopyGmToL1B copyGmToL1B;
        auto l1BLayoutTla = tla::MakeLayout<ElementQ, LayoutTagL1B>(embed, qSTile);
        auto l1BTensorTla = tla::MakeTensor(l1QTensor[0], l1BLayoutTla, Arch::PositionL1{});
        auto l1BTensorTlaTile = GetTile(l1BTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(embed, qSTile));
        auto gQTensorTlaTile = GetTile(gQTensor, tla::MakeCoord(0, 0), tla::MakeShape(embed, qSTile));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        copyGmToL1B(l1BTensorTlaTile, gQTensorTlaTile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <class TensorK, class TensorL1K>
    __aicore__ inline void SparseKL1TileMLoad(TensorK& gKTensor, TensorL1K& l1KTensorTla,
                                              AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                              uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen, uint32_t kvSBaseTile,
                                              uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                                              uint32_t l1KTileMAct, uint32_t embed, uint32_t kvSBaseTileInnerOffset)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorK>;
        CopyGmToL1A copyGmToL1A;
        uint32_t baseTileStartOffset = gatheredKvSTileIdx * kvSBaseTile + kvSBaseTileInnerOffset;
        uint32_t baseTileEndOffset = baseTileStartOffset + l1KTileMAct;
        // Traverse sparse blocks to fill the requested K tile.
        uint32_t gatheredStartOffset = baseTileStartOffset;
        uint32_t gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
        uint32_t yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
        uint32_t oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
        uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
        uint32_t dealtLenAccum = 0;

        while (dealtLenAccum < l1KTileMAct && gatheredYBlockIdx < yBlockNumRsvd && oriYBlockIdx < yBlockNumAval &&
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
            // K copied to L1 as A matrix
            auto l1KTensorTlaTile =
                GetTile(l1KTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
            auto gKTensorTlaTile =
                GetTile(gKTensor, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(curDealtLen, embed));
            copyGmToL1A(l1KTensorTlaTile, gKTensorTlaTile);
            dealtLenAccum += curDealtLen;
            gatheredStartOffset += curDealtLen;
            gatheredYBlockIdx = gatheredStartOffset / blockShapeY;
            yBlockInnerStartOffset = gatheredStartOffset % blockShapeY;
            if (dealtLenAccum < l1KTileMAct) {
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
                oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStartOffset;
            }
        }
    }

    template <class TensorK, class TensorS>
    __aicore__ inline void operator()(TensorK& gKTensor, TensorS& ubSTensor,
                                      AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape,
                                      uint32_t gatheredKvSTileIdx, uint32_t kvSeqlen, uint32_t kvSBaseTile,
                                      uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
                                      uint64_t& l0Stage, Arch::CrossCoreFlag mm1ToSmFlag)
    {
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorS>;
        CopyL0CToDst copyL0CToDst;

        const uint32_t bufId = static_cast<uint32_t>(l0Stage++ % L0_STAGES);
        // Derive buffer indices and event ids from the outer slot.
        const uint32_t l0AMicroBase = L0A_EVENT_BASE + bufId * L0_MICRO_STAGES;
        const uint32_t l0BSlot = bufId * L0_MICRO_STAGES;
        const uint32_t l0BEventId = L0B_EVENT_BASE + l0BSlot;
        const uint32_t l0CMicroBase = bufId * L0_MICRO_STAGES;

        uint32_t qSTile = actualOriShape[0];
        uint32_t embed = actualOriShape[2];
        uint32_t kSTile = actualOriShape[1];

        // Q as Matrix B on L1, has shape [D, S1] in nZ
        auto l1QLayoutTla = tla::MakeLayout<ElementQ, LayoutTagL1B>(embed, qSTile);
        auto l1QTensorTla = tla::MakeTensor(l1QTensor[0], l1QLayoutTla, Arch::PositionL1{});

        uint32_t mL1LoopNum = CeilDiv(kSTile, l1ATileM);
        uint32_t nL0LoopNum = CeilDiv(qSTile, L0_TILE_N);
        uint32_t kL0LoopNum = CeilDiv(embed, L0_TILE_K);

        // Preserve column order when splitting S between AIVs; issue Fixpipe only after each L0C portion is ready.
        for (uint32_t mL1Itr = 0; mL1Itr < mL1LoopNum; mL1Itr++) {
            uint32_t l1TileMAct = (mL1Itr == mL1LoopNum - 1) ? (kSTile - mL1Itr * l1ATileM) : l1ATileM;
            uint32_t mLoopCounterL1 = GetCurLoopCounter(gatheredKvSTileIdx, mL1LoopNum, mL1Itr);
            uint32_t l1KBufId = mLoopCounterL1 % l1KBufNum;
            uint32_t l1AEventId = l1KBufId + 1;
            auto l1KLayoutTla = tla::MakeLayout<ElementK, LayoutTagL1A>(l1TileMAct, embed);
            auto l1KTensorTla = tla::MakeTensor(l1KTensor[l1KBufId], l1KLayoutTla, Arch::PositionL1{});
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventId);
            SparseKL1TileMLoad(gKTensor, l1KTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSeqlen, kvSBaseTile,
                               blockShapeY, yBlockNumAval, yBlockNumRsvd, l1TileMAct, embed, mL1Itr * l1ATileM);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventId);
            uint32_t mL0LoopNum = CeilDiv(l1TileMAct, L0_TILE_M);
            bool l1ToL0BNoRepeatFlag = (nL0LoopNum == 1) && (kL0LoopNum <= L0_STAGES);
            for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (l1TileMAct - mL0Itr * L0_TILE_M) : L0_TILE_M;
                uint32_t microLoopNum = CeilDiv(l0TileMAct, L0_TILE_M_MICRO);

                // Launch MMAD for all micro-slots before collecting Fixpipe results.
                for (uint32_t microIdx = 0; microIdx < microLoopNum; microIdx++) {
                    uint32_t microMAct =
                        (microIdx == microLoopNum - 1) ? (l0TileMAct - microIdx * L0_TILE_M_MICRO) : L0_TILE_M_MICRO;
                    uint32_t microMOffset = mL0Itr * L0_TILE_M + microIdx * L0_TILE_M_MICRO;
                    const uint32_t l0AEventId = l0AMicroBase + microIdx;
                    const uint32_t l0CMicroEventId = l0CMicroBase + microIdx;
                    auto l0CLayoutTla = tla::MakeLayoutL0C(microMAct, qSTile);
                    auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CMicroEventId], l0CLayoutTla, Arch::PositionL0C{});
                    for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
                        uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (qSTile - nL0Itr * L0_TILE_N) : L0_TILE_N;
                        auto l0CTensorTlaTile = GetTile(l0CTensorTla, tla::MakeCoord(0, nL0Itr * L0_TILE_N),
                                                        tla::MakeShape(microMAct, l0TileNAct));
                        for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                            uint32_t l0TileKAct = (kL0Itr == kL0LoopNum - 1) ? (embed - kL0Itr * L0_TILE_K) : L0_TILE_K;

                            auto l1KTensorTlaTile =
                                GetTile(l1KTensorTla, tla::MakeCoord(microMOffset, kL0Itr * L0_TILE_K),
                                        tla::MakeShape(microMAct, l0TileKAct));
                            auto l0ALayoutTla = tla::MakeLayout<ElementK, LayoutTagL0A>(microMAct, l0TileKAct);
                            auto l0ATensorTla =
                                tla::MakeTensor(l0ATensor[l0AEventId], l0ALayoutTla, Arch::PositionL0A{});

                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                            copyL1ToL0A(l0ATensorTla, l1KTensorTlaTile);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            if ((nL0Itr == nL0LoopNum - 1) && (microIdx == microLoopNum - 1) &&
                                (mL0Itr == mL0LoopNum - 1) && (kL0Itr == kL0LoopNum - 1)) {
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventId);
                            }

                            auto l1QTensorTlaTile =
                                GetTile(l1QTensorTla, tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                                        tla::MakeShape(l0TileKAct, l0TileNAct));
                            auto l0BLayoutTla = tla::MakeLayout<ElementQ, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                            auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BSlot], l0BLayoutTla, Arch::PositionL0B{});

                            // Load the shared Q tile only for the first micro-slot.
                            bool loadQ = l1ToL0BNoRepeatFlag && (mL1Itr == 0) && (mL0Itr == 0) && (microIdx == 0);
                            if (loadQ) {
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                                copyL1ToL0B(l0BTensorTla, l1QTensorTlaTile);
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            }

                            bool initMmad = (kL0Itr == 0);
                            uint32_t microMAligned = RoundUp(microMAct, 16);
                            if (loadQ) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            }
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            if (nL0Itr == 0 && kL0Itr == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CMicroEventId);
                            }
                            tileMmad(l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, microMAligned, l0TileNAct,
                                     l0TileKAct, initMmad);
                            if (l1ToL0BNoRepeatFlag && (mL1Itr == mL1LoopNum - 1) && (mL0Itr == mL0LoopNum - 1) &&
                                (microIdx == microLoopNum - 1)) {
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                        }
                    }
                    // Record MMAD completion; Fixpipe is collected in the second pass.
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CMicroEventId);
                }

                // Collect Fixpipe results and release each micro-slot for reuse.
                if (mL0Itr == 0) {
                    // reverse crossCoreSync, do fixPipe only after ubSTensor is fully released
                    WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
                }
                for (uint32_t microIdx = 0; microIdx < microLoopNum; microIdx++) {
                    uint32_t microMAct =
                        (microIdx == microLoopNum - 1) ? (l0TileMAct - microIdx * L0_TILE_M_MICRO) : L0_TILE_M_MICRO;
                    uint32_t microMOffset = mL0Itr * L0_TILE_M + microIdx * L0_TILE_M_MICRO;
                    const uint32_t l0CMicroEventId = l0CMicroBase + microIdx;
                    auto l0CLayoutTla = tla::MakeLayoutL0C(microMAct, qSTile);
                    auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CMicroEventId], l0CLayoutTla, Arch::PositionL0C{});
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CMicroEventId);

                    uint32_t mFixPAligned8 = RoundUp(microMAct, 8);
                    if constexpr (FULL_QUANT_HIF8) {
                        // NO_QUANT SPLIT_N: one dual-destination Fixpipe serves both AIVs. K * Q^T
                        // stores Q on the N axis of L0C, so the split is along N: the hardware writes
                        // columns 0..n/2-1 into AIV0 and n/2..n-1 into AIV1, both at the same UB
                        // offsets given by ubSTensor's per-AIV stride. nSize is the full per-block Q
                        // width, which is what ubSTensor now declares, and the dequantization is done
                        // by the online-softmax epilogue instead of by the Fixpipe.
                        uint32_t nFixPAligned32 = RoundUp(qSTile, 32);
                        auto ubSTensorTlaTile = GetTile(ubSTensor, tla::MakeCoord(microMOffset, 0),
                                                        tla::MakeShape(mFixPAligned8, nFixPAligned32));
                        copyL0CToDst(ubSTensorTlaTile, l0CTensorTla);
                    } else {
                        uint32_t nFixPAligned8 = RoundUp(qSTile, 8);
                        auto ubSTensorTlaTile = GetTile(ubSTensor, tla::MakeCoord(microMOffset, 0),
                                                        tla::MakeShape(mFixPAligned8, nFixPAligned8));
                        copyL0CToDst(ubSTensorTlaTile, l0CTensorTla);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CMicroEventId);
                }
            }
        }
        // crossCoreSync after all fixPipe move
        SetCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementQ> l1QTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementK> l1KTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementQ> l0ATensor[L0_STAGES * L0_MICRO_STAGES];
    AscendC::LocalTensor<ElementK> l0BTensor[L0_STAGES * L0_MICRO_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES * L0_MICRO_STAGES];

    TileMmad tileMmad;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;

    uint32_t l1ATileM;
    uint32_t l1BTileN;
    uint32_t l1ATileK;
    uint32_t l1BTileK;
    uint32_t l1QBufNum;
    uint32_t l1KBufNum;
};
////////////////////////////////////////////////////////////////////

} // namespace NpuArch::Gemm::Block
#endif // GEMM_BLOCK_K_QT_ST_ARCH35_HIF8_HPP
