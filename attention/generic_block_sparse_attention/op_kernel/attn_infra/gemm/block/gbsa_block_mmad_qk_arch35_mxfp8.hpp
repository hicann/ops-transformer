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
 * This implementation is designed for the following scenario:
 * A full q base tile is loaded to L1 from GM at the very beginning,
 * and it remains persistent until each k base tile is dealt
 * A full q*k^t base tile is loaded to UB from l0C, no workspace transit
 */
#ifndef GBSA_BLOCK_MMAD_QK_ARCH35_MXFP8_HPP
#define GBSA_BLOCK_MMAD_QK_ARCH35_MXFP8_HPP

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
#include "../tile_common/gbsa_mxfp8_tile.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"
#include "../../../arch35/gbsa_layout_utils.hpp"

////////////////////////////////////////////////////////////////////

enum class GbsaMxQkVariant : uint32_t {
    D192 = 0U,
    D192_PREFILL = 192U,
    D256 = 256U
};

namespace NpuArch::Gemm::Block {
////////////////////////////////////////////////////////////////////

// Keep the MXFP8 block geometry independent of non-quantized QK resources.
template <bool GENERAL>
struct GbsaMxfp8QKGeometry {};

template <>
struct GbsaMxfp8QKGeometry<true> {
protected:
    uint32_t mxValidD = 0;
    uint64_t mxQScaleHeadStrideHalf = 0;
    uint64_t mxKScaleTokenStrideHalf = 0;
};

template <bool PACKED>
struct GbsaMxfp8PackedQKGeometry {};

template <>
struct GbsaMxfp8PackedQKGeometry<true> {
protected:
    Tile::GbsaMxPackedCursor mxPackedCursor;
    AscendC::LocalTensor<uint8_t> mxPackedScale;
};

struct Mxfp8Mm1L1TileHelper {
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;

    __aicore__ inline Mxfp8Mm1L1TileHelper() {}

    __aicore__ inline Mxfp8Mm1L1TileHelper(uint32_t m, uint32_t n, uint32_t kl, uint32_t kr, uint32_t pbn, uint32_t vbn)
        : mm1L1TileM(m),
          mm1L1TileN(n),
          mm1L1TileKLeft(kl),
          mm1L1TileKRight(kr),
          qL1BufNum(pbn),
          kL1BufNum(vbn)
    {}
};

template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_,
          class TileCopy_, class TileMmad_>
struct BlockMmadTla<MmadAtlasA5GbsaQKMxfp8, L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_,
                    TileCopy_, TileMmad_> : protected GbsaMxfp8QKGeometry<TileCopy_::IS_GENERAL_MX>,
                                            protected GbsaMxfp8PackedQKGeometry<TileCopy_::IS_PACKED_MX> {
public:
    using DispatchPolicy = MmadAtlasA5GbsaQKMxfp8;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    static_assert(AscendC::IsSameType<ElementA, fp8_e4m3fn_t>::value &&
                      AscendC::IsSameType<ElementB, fp8_e4m3fn_t>::value &&
                      AscendC::IsSameType<ElementC, float>::value && TileCopy::IS_PACKED_MX && TileCopy::IS_GENERAL_MX,
                  "MXFP8 block requires packed FP8 inputs and FP32 accumulators");

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

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag>& resource, Mxfp8Mm1L1TileHelper& mm1L1TileHelper)
    {
        l1ATileM = mm1L1TileHelper.mm1L1TileM;
        l1BTileN = mm1L1TileHelper.mm1L1TileN;
        l1ATileK = mm1L1TileHelper.mm1L1TileKLeft;
        l1BTileK = mm1L1TileHelper.mm1L1TileKRight;
        l1ABufNum = mm1L1TileHelper.qL1BufNum;
        l1BBufNum = mm1L1TileHelper.kL1BufNum;

        for (uint32_t i = 0; i < l1ABufNum; i++) {
            l1ATensor[i] =
                resource.l1Buf.template GetBufferByByte<ElementA>(l1ATileM * l1ATileK * sizeof(ElementA) * i);
        }
        for (uint32_t i = 0; i < l1BBufNum; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum + l1BTileK * l1BTileN * sizeof(ElementB) * i);
        }
        {
            mxScaleA = resource.l1Buf.template GetBufferByByte<uint8_t>(Tile::GBSA_MX_SCALE_BASE +
                                                                        0U * Tile::GBSA_MX_SCALE_SLOT);
        }
        for (uint32_t i = 0; i < L0_STAGES; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i);
        }
    }

    __aicore__ inline void SetMxQScale(AscendC::GlobalTensor<uint8_t> scale)
    {
        mxGmA = scale;
    }
    // General MX uses actual GM D and a separate 64-aligned compute D. Scale
    // strides are measured in packed E8M0 pairs (uint16 elements).
    __aicore__ inline void SetMxGeometry(uint32_t validD, uint64_t qScaleHeadStrideHalf, uint64_t kScaleTokenStrideHalf)
    {
        {
            this->mxValidD = validD;
            this->mxQScaleHeadStrideHalf = qScaleHeadStrideHalf;
            this->mxKScaleTokenStrideHalf = kScaleTokenStrideHalf;
        }
    }

    __aicore__ inline void SetMxPacked(Tile::GbsaMxPackedCursor cursor, AscendC::LocalTensor<uint8_t> scale)
    {
        {
            this->mxPackedCursor = cursor;
            this->mxPackedScale = scale;
        }
    }

    /// Destructor
    __aicore__ inline ~BlockMmadTla() {}

    template <class TensorA>
    __aicore__ inline void loadQGM(TensorA& gATensor, GemmCoord actualOriShape)
    {
        using CopyGmToL1A = typename TileCopy_::template CopyGmToL1A<TensorA>;
        CopyGmToL1A copyGmToL1A;
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[1];
        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{});
        auto l1ATensorTlaTile = GetTile(l1ATensorTla, tla::MakeCoord(0, 0), tla::MakeShape(rowNum, embed));
        uint32_t copyEmbed = embed;
        {
            copyEmbed = this->mxValidD;
        }
        auto gATensorTlaTile = GetTile(gATensor, tla::MakeCoord(0, 0), tla::MakeShape(rowNum, copyEmbed));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);

        copyGmToL1A(l1ATensorTlaTile, gATensorTlaTile);
        {
            Tile::CopyGbsaMxQKScale(mxScaleA, mxGmA, rowNum, embed / 64U, this->mxQScaleHeadStrideHalf);
            copyL1ToL0A.SetScale(mxScaleA, embed / 64U);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag& crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs separately
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
        // in mode 4, AIC wait for 2 AIVs separately
        if constexpr (MODE == 4U) {
            uint16_t flagIdV0 = crossCoreFlag.id;
            uint16_t flagIdV1 = flagIdV0 + V0_V1_FLAG_ID_OFFSET;
            Arch::CrossCoreFlag crossCoreFlagV1(flagIdV1);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlagV1);
        }
    }

    template <class TensorB, class TensorL1B>
    __aicore__ inline void SparseKL1TileNLoad(TensorB& gBTensor, TensorL1B& l1BTensorTla,
                                              AscendC::GlobalTensor<int32_t> gSparseBlockIdx,
                                              uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile, uint32_t blockShapeY,
                                              uint32_t l1BTileNAct, uint32_t l1BTileKAct,
                                              uint32_t kvSBaseTileInnerOffset, uint32_t embedInnerOffset,
                                              AscendC::GlobalTensor<int32_t> gBlockTable, uint32_t blockSize,
                                              uint64_t pageStride, uint64_t tokenStride,
                                              uint32_t localExplicitBlockCount, uint32_t residualBlockId,
                                              AscendC::GlobalTensor<uint64_t> kvOffsetInfo, uint32_t kvOffsetGroupWords)
    {
        using CopyGmToL1B = typename TileCopy_::template CopyGmToL1B<TensorB>;
        CopyGmToL1B copyGmToL1B;

        auto cursor = this->mxPackedCursor;
        if (cursor.packedRecords != nullptr && kvSBaseTileInnerOffset == 0U && l1BTileNAct == cursor.packedRows) {
            AscendC::GlobalTensor<uint64_t> records;
            records.SetGlobalBuffer(const_cast<__gm__ uint64_t*>(cursor.packedRecords));
            Tile::CopyGbsaMxPackedRecords<false, ElementB, ArchTag>(
                gBTensor, l1BTensorTla, records, cursor.packedRecordCount, l1BTileKAct, embedInnerOffset);
            return;
        }
        cursor.Advance(kvSBaseTileInnerOffset);
        uint32_t token, rows, copied = 0;
        while (copied < l1BTileNAct && cursor.NextSpan(l1BTileNAct - copied, token, rows)) {
            const uint64_t sourceOffset = static_cast<uint64_t>(token) * tokenStride + embedInnerOffset;
            auto gm = tla::MakeTensor(gBTensor.data()[sourceOffset], gBTensor.layout(), Arch::PositionGM{});
            auto src = GetTile(gm, tla::MakeCoord(0, 0), tla::MakeShape(l1BTileKAct, rows));
            auto dst = GetTile(l1BTensorTla, tla::MakeCoord(0, copied), tla::MakeShape(l1BTileKAct, rows));
            copyGmToL1B(dst, src);
            copied += rows;
        }
        return;
    }

    template <class TensorB, class TensorC>
    __aicore__ inline void operator()(TensorB& gBTensor, TensorC& ubCTensor,
                                      AscendC::GlobalTensor<int32_t> gSparseBlockIdx, GemmCoord actualOriShape,
                                      uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile, uint32_t blockShapeY,
                                      uint64_t prefixSumL0AStages, uint64_t prefixSumL0BStages,
                                      Arch::CrossCoreFlag mm1ToSmFlag, uint64_t tokenStride,
                                      AscendC::GlobalTensor<int32_t> gBlockTable = {}, uint32_t blockSize = 0,
                                      uint64_t pageStride = 0, uint32_t localExplicitBlockCount = 0,
                                      uint32_t residualBlockId = 0, AscendC::GlobalTensor<uint64_t> kvOffsetInfo = {},
                                      uint32_t kvOffsetGroupWords = 0, uint32_t sparseReadyFlag = 0)
    {
        uint32_t rowNum = actualOriShape[0];
        uint32_t embed = actualOriShape[2];
        uint32_t curBaseTileSize = actualOriShape[1];

        auto l1ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL1A>(rowNum, embed);
        auto l1ATensorTla = tla::MakeTensor(l1ATensor[0], l1ALayoutTla, Arch::PositionL1{});

        // Packed dispatch has D192/256, <=128 query rows and a single full-D
        // L1 tile spanning the complete 128/256/512-token fragment. Express only
        // those proven constants; N loops still follow the runtime tile/tail.
        static_assert(L0_TILE_K == 128U || L0_TILE_K == 256U,
                      "Recheck packed D192/256 loop constants for a new L0 K shape");
        constexpr uint32_t packedKLoops = (256U + L0_TILE_K - 1U) / L0_TILE_K;
        uint32_t nL1LoopNum = 1U;
        uint32_t kL1LoopNum = 1U;
        // Packed tiling guarantees groupSize <= 128 and D <= 256.
        // Fold the single M slice and keep both Q K-slices resident in L0A.
        const uint32_t mL0LoopNum = 1U;
        uint32_t totalNL0LoopNum = CeilDiv(kvSBaseTile, L0_TILE_N);
        uint32_t totalKL0LoopNum = packedKLoops;
        uint32_t fullNL0LoopNum = CeilDiv(l1BTileN, L0_TILE_N);
        uint32_t fullKL0LoopNum = packedKLoops;

        const bool l1ToL0ANoRepeatFlag = true;

        // while splitting the base tile S to 2 AIVs,
        // the order of the elements in each column is expected to be preserved,
        // which means a column in l0C cannot be chunked and processed by dualMode FixPipe separately.
        // therefore, FixPipe won't launch until each portion(chunked only by columns, based on nbuffer strategy)
        // of the base tile is ready on l0C
        for (uint32_t nL1Itr = 0; nL1Itr < nL1LoopNum; nL1Itr++) {
            uint32_t l1TileNAct = (nL1Itr == nL1LoopNum - 1) ? (curBaseTileSize - nL1Itr * l1BTileN) : l1BTileN;
            uint32_t nL0LoopNum = CeilDiv(l1TileNAct, L0_TILE_N);
            for (uint32_t kL1Itr = 0; kL1Itr < kL1LoopNum; ++kL1Itr) {
                uint32_t l1TileKAct = (kL1Itr == kL1LoopNum - 1) ? (embed - kL1Itr * l1BTileK) : l1BTileK;
                uint32_t kLoopCounterL1 = (gatheredKvSTileIdx * nL1LoopNum + nL1Itr) * kL1LoopNum + kL1Itr;
                // Select the K slot without changing the established event mapping.
                uint32_t l1BBufId = kLoopCounterL1 % l1BBufNum;
                uint32_t l1BEventId = l1BBufId + 1;
                auto l1BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL1B>(l1TileKAct, l1TileNAct);
                auto l1BTensorTla = tla::MakeTensor(l1BTensor[l1BBufId], l1BLayoutTla, Arch::PositionL1{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);

                SparseKL1TileNLoad(gBTensor, l1BTensorTla, gSparseBlockIdx, gatheredKvSTileIdx, kvSBaseTile,
                                   blockShapeY, l1TileNAct, l1TileKAct, nL1Itr * l1BTileN, kL1Itr * l1BTileK,
                                   gBlockTable, blockSize, pageStride, tokenStride, localExplicitBlockCount,
                                   residualBlockId, kvOffsetInfo, kvOffsetGroupWords);

                {
                    // AIV packed all scale rows in exact MX NZ order. The K
                    // tile spans full D; each 16-token group starts on 32B.
                    const uint32_t groups = embed / 64U;
                    const uint32_t scaleOffset = nL1Itr * l1BTileN * groups * 2U;
                    copyL1ToL0B.SetScale(this->mxPackedScale[scaleOffset], groups);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventId);

                uint32_t kL0LoopNum = packedKLoops;
                for (uint32_t nL0Itr = 0; nL0Itr < nL0LoopNum; nL0Itr++) {
                    uint32_t l0TileNAct = (nL0Itr == nL0LoopNum - 1) ? (l1TileNAct - nL0Itr * L0_TILE_N) : L0_TILE_N;
                    uint32_t globalNL0Itr = nL1Itr * fullNL0LoopNum + nL0Itr;
                    uint32_t l0CLoopCounter = gatheredKvSTileIdx * totalNL0LoopNum + globalNL0Itr;
                    uint32_t l0CBufId = l0CLoopCounter % L0_STAGES;
                    uint32_t l0CEventId = l0CBufId;
                    auto l0CLayoutTla = tla::MakeLayoutL0C(rowNum, l0TileNAct);
                    auto l0CTensorTla = tla::MakeTensor(l0CTensor[l0CBufId], l0CLayoutTla, Arch::PositionL0C{});
                    for (uint32_t mL0Itr = 0; mL0Itr < mL0LoopNum; mL0Itr++) {
                        uint32_t l0TileMAct = (mL0Itr == mL0LoopNum - 1) ? (rowNum - mL0Itr * L0_TILE_M) : L0_TILE_M;
                        auto l0CTensorTlaTile = GetTile(l0CTensorTla, tla::MakeCoord(mL0Itr * L0_TILE_M, 0),
                                                        tla::MakeShape(l0TileMAct, l0TileNAct));
                        for (uint32_t kL0Itr = 0; kL0Itr < kL0LoopNum; kL0Itr++) {
                            uint32_t l0TileKAct =
                                (kL0Itr == kL0LoopNum - 1) ? (l1TileKAct - kL0Itr * L0_TILE_K) : L0_TILE_K;
                            uint32_t globalKL0Itr = kL1Itr * fullKL0LoopNum + kL0Itr;
                            uint32_t l0ALoopCounter = prefixSumL0AStages + mL0Itr * totalKL0LoopNum + globalKL0Itr;
                            uint32_t l0BLoopCounter =
                                prefixSumL0BStages + globalNL0Itr * totalKL0LoopNum + globalKL0Itr;
                            uint32_t l0ABufId = l0ALoopCounter % L0_STAGES;
                            uint32_t l0BBufId = l0BLoopCounter % L0_STAGES;
                            uint32_t l0AEventId = l0ABufId;
                            uint32_t l0BEventId = l0BBufId + 2;

                            auto l1BTensorTlaTile =
                                GetTile(l1BTensorTla, tla::MakeCoord(kL0Itr * L0_TILE_K, nL0Itr * L0_TILE_N),
                                        tla::MakeShape(l0TileKAct, l0TileNAct));
                            auto l0BLayoutTla = tla::MakeLayout<ElementB, LayoutTagL0B>(l0TileKAct, l0TileNAct);
                            auto l0BTensorTla = tla::MakeTensor(l0BTensor[l0BBufId], l0BLayoutTla, Arch::PositionL0B{});
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                            copyL1ToL0B(l0BTensorTla, l1BTensorTlaTile);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            if (nL0Itr == nL0LoopNum - 1U && mL0Itr == mL0LoopNum - 1U && kL0Itr == kL0LoopNum - 1U) {
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventId);
                            }

                            auto l1ATensorTlaTile =
                                GetTile(l1ATensorTla,
                                        tla::MakeCoord(mL0Itr * L0_TILE_M, kL1Itr * l1BTileK + kL0Itr * L0_TILE_K),
                                        tla::MakeShape(l0TileMAct, l0TileKAct));
                            auto l0ALayoutTla = tla::MakeLayout<ElementA, LayoutTagL0A>(l0TileMAct, l0TileKAct);
                            auto l0ATensorTla = tla::MakeTensor(l0ATensor[l0ABufId], l0ALayoutTla, Arch::PositionL0A{});
                            const bool firstN = (nL1Itr == 0U) && (nL0Itr == 0U);
                            const bool lastN = (nL1Itr == nL1LoopNum - 1U) && (nL0Itr == nL0LoopNum - 1U);
                            if (!l1ToL0ANoRepeatFlag || firstN) {
                                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                                copyL1ToL0A(l0ATensorTla, l1ATensorTlaTile);
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            }

                            if (!l1ToL0ANoRepeatFlag || firstN) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEventId);
                            }
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEventId);
                            if (mL0Itr == 0U && kL1Itr == 0U && kL0Itr == 0U) {
                                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);
                            }
                            bool initMmad = (kL1Itr == 0U) && (kL0Itr == 0U);
                            uint32_t l0TileMAligned = RoundUp(l0TileMAct, 16);
                            tileMmad(l0CTensorTlaTile, l0ATensorTla, l0BTensorTla, l0TileMAligned, l0TileNAct,
                                     l0TileKAct, initMmad);
                            if (!l1ToL0ANoRepeatFlag || lastN) {
                                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventId);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventId);
                        }
                    }

                    if (kL1Itr == kL1LoopNum - 1U) {
                        // All L1 N tiles share one S buffer and one release token.
                        // Acquire it only before the first Fixpipe of the whole KV tile.
                        if (globalNL0Itr == 0U) {
                            WaitCrossCoreSync<4, PIPE_FIX>(mm1ToSmFlag);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
                        using CopyL0CToDst = typename TileCopy_::template CopyL0CToDst<TensorC>;
                        {
                            uint32_t nFixPAligned8 = RoundUp(l0TileNAct, 8);
                            auto ubCTensorTlaTile = GetTile(ubCTensor, tla::MakeCoord(0, globalNL0Itr * L0_TILE_N),
                                                            tla::MakeShape(rowNum, nFixPAligned8));
                            CopyL0CToDst copyL0CToDst;
                            copyL0CToDst(ubCTensorTlaTile, l0CTensorTla);
                        }
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
    AscendC::LocalTensor<ElementA> l1ATensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor[MAX_L1_STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[L0_STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[L0_STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[L0_STAGES];

    AscendC::LocalTensor<uint8_t> mxScaleA;
    AscendC::GlobalTensor<uint8_t> mxGmA;
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

// Packed D192/256 Q stays in a disjoint L0A region for the whole query task.
// Grouped K lookahead uses independent L1 slots and their free/ready events.
template <class Base, GbsaMxQkVariant QkVariant = GbsaMxQkVariant::D192>
struct GbsaMxPackedLookaheadQK : public Base {
    // The variant controls actual MMAD geometry. D192 and D256 still share
    // D256-capacity buffers selected by Host tiling.
    static constexpr uint32_t ACTUAL_HEAD_DIM = QkVariant == GbsaMxQkVariant::D256 ? 256U : 192U;
    using ArchTag = typename Base::ArchTag;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementC = typename Base::ElementC;
    using TileCopy = typename Base::TileCopy;
    bool prefetched = false;
    bool qResident = false;
    bool residentQuery = false;
    bool reuseCsaQuery = false;
    static_assert(TileCopy::IS_PACKED_MX && Base::L0_STAGES == 2U &&
                      ((Base::L0_TILE_K == 256U && Base::L0_TILE_N == 128U) ||
                       (Base::L0_TILE_K == 128U && Base::L0_TILE_N == 256U)),
                  "Packed QK requires two32KiB L0B banks");

    __aicore__ inline GbsaMxPackedLookaheadQK(Arch::Resource<ArchTag>& resource, Mxfp8Mm1L1TileHelper& helper)
        : Base(resource, helper)
    {
        // Event4 is unused by the base QK/PV M_MTE1 protocols (0..3).
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        if (this->l1BBufNum == 3U) {
            // K slots 0/1 use events 1/2; PV owns events 3/4.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        }
    }

    __aicore__ inline ~GbsaMxPackedLookaheadQK()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        if (this->l1BBufNum == 3U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        }
    }

    __aicore__ inline void SetQueryTileCount(uint64_t tiles)
    {
        // A single D192 KV tile already loads Q only once in the base path.
        // Retain that path when there is no later tile to reuse the Q copy.
        reuseCsaQuery = tiles > 1U;
    }

    template <class TensorA>
    __aicore__ inline void loadQGM(TensorA& query, GemmCoord shape)
    {
        Base::loadQGM(query, shape);
        residentQuery = shape[1] == 256U || (shape[1] == 192U && reuseCsaQuery);
        if (residentQuery) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        }
        qResident = false;
        if constexpr (Base::L0_TILE_N == 128U) {
            if (this->l1BBufNum == 3U) {
                // Q is ready in L1 and the preceding query has released its
                // disjoint L0A region. Prepare it before waiting for K records.
                PrepareGroupedQuery(shape[0], shape[1]);
            }
        }
    }

    __aicore__ inline void PrepareGroupedQuery(uint32_t rows, uint32_t embed)
    {
        auto l1Q =
            tla::MakeTensor(this->l1ATensor[0], tla::MakeLayout<ElementA, typename Base::LayoutTagL1A>(rows, embed),
                            Arch::PositionL1{});
        auto l0Q =
            tla::MakeTensor(this->l0ATensor[0][QueryL0Offset()],
                            tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(rows, embed), Arch::PositionL0A{});
        auto srcQ = GetTile(l1Q, tla::MakeCoord(0, 0), tla::MakeShape(rows, embed));
        this->copyL1ToL0A(l0Q, srcQ);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        qResident = true;
    }

    __aicore__ inline uint32_t QueryL0Offset() const
    {
        // Up to M64 fits the existing upper 16 KiB. M128 reserves the upper
        // 32 KiB and PV places its two 16-KiB P slots in the lower half.
        return this->l1ATileM > 64U ? 32U * 1024U : 48U * 1024U;
    }

    __aicore__ inline void ReleaseQuery()
    {
        if (residentQuery) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        }
    }

    template <class TensorC>
    __aicore__ inline void EmitScore(TensorC& output, uint32_t rows, uint32_t columns, uint32_t slice, uint32_t bank,
                                     Arch::CrossCoreFlag& ready)
    {
        if (slice == 0U) {
            this->template WaitCrossCoreSync<4, PIPE_FIX>(ready);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(bank);
        auto c = tla::MakeTensor(this->l0CTensor[bank], tla::MakeLayoutL0C(rows, columns), Arch::PositionL0C{});
        using Copy = typename TileCopy::template CopyL0CToDst<TensorC>;
        if constexpr (AscendC::IsSameType<ElementC, float>::value) {
            auto dst =
                GetTile(output, tla::MakeCoord(0, slice * Base::L0_TILE_N), tla::MakeShape(rows, RoundUp(columns, 8U)));
            Copy copy;
            copy(dst, c);
        } else {
            const uint32_t perSubcore = RoundUp(rows, 2U) / 2U;
            auto dst = GetTile(output, tla::MakeCoord(0, slice * Base::L0_TILE_N),
                               tla::MakeShape(perSubcore, RoundUp(columns, 16U)));
            auto first = GetTile(c, tla::MakeCoord(0, 0), tla::MakeShape(perSubcore, columns));
            auto second = GetTile(c, tla::MakeCoord(perSubcore, 0), tla::MakeShape(perSubcore, columns));
            Copy copy0, copy1;
            copy0(dst, first, false);
            copy1(dst, second, true);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(bank);
    }

    __aicore__ inline bool HasPrefetched() const
    {
        return prefetched;
    }

    __aicore__ inline static uint32_t KSlotEvent(uint32_t slot)
    {
        return slot == 2U ? EVENT_ID5 : slot + 1U;
    }

    template <class TensorB>
    __aicore__ inline void PreloadGroup(TensorB& key, const Tile::GbsaMxPackedCursor& cursor, uint32_t columns,
                                        uint32_t embed, uint32_t group, uint32_t blockY, uint64_t tokenStride)
    {
        const uint32_t bank = group % this->l1BBufNum;
        const uint32_t event = KSlotEvent(bank);
        auto l1K =
            tla::MakeTensor(this->l1BTensor[bank],
                            tla::MakeLayout<ElementB, typename Base::LayoutTagL1B>(embed, columns), Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event);
        const auto saved = this->mxPackedCursor;
        this->mxPackedCursor = cursor.Group(group, columns);
        this->SparseKL1TileNLoad(key, l1K, {}, 0U, 128U, blockY, columns, embed, 0U, 0U, {}, 0U, 0U, tokenStride, 0U,
                                 0U, {}, 0U);
        this->mxPackedCursor = saved;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event);
    }

    // One online-softmax tile has one, two or four N128 QK MMADs at the actual D.
    // Each AIV has independent credits for its two groups. They occupy
    // disjoint L1 regions, so publishing group 2/3 need not wait for 0/1.
    // MTE1 returns each credit; the next tile reuses it in the same group order.
    template <uint32_t Groups, class TensorB, class TensorC>
    __aicore__ inline void ComputeGrouped(TensorB& key, TensorC& output, GemmCoord shape, uint32_t tile,
                                          uint32_t blockY, Arch::CrossCoreFlag ready, uint64_t tokenStride)
    {
        constexpr uint32_t embed = ACTUAL_HEAD_DIM;
        const uint32_t rows = shape[0], columns = shape[1];
        const auto cursor = this->mxPackedCursor;
        if (!prefetched) {
            PreloadGroup(key, cursor, min(128U, columns), embed, 0U, blockY, tokenStride);
        }
        prefetched = false;
        auto l0Q =
            tla::MakeTensor(this->l0ATensor[0][QueryL0Offset()],
                            tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(rows, embed), Arch::PositionL0A{});
        if (!qResident) {
            PrepareGroupedQuery(rows, embed);
        }
        // AIV1 publishes one empty group when KV128 has only group 0.
        constexpr uint32_t notifyGroups = Groups == 1U ? 2U : Groups;
        for (uint32_t group = 0U; group < notifyGroups; ++group) {
            const uint32_t scaleFlag = 10U + group / 2U + (group % 2U) * 16U;
            // Even empty tail groups exchange credits; neither AIV can be
            // left blocked while draining a task or moving to the next batch.
            AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(scaleFlag);
            if (group * 128U < columns) {
                const uint32_t n = min(128U, columns - group * 128U);
                const uint32_t bank = group % Base::L0_STAGES;
                const uint32_t kSlot = group % this->l1BBufNum;
                const uint32_t kEvent = KSlotEvent(kSlot), bEvent = bank + 2U;
                auto l1K = tla::MakeTensor(this->l1BTensor[kSlot],
                                           tla::MakeLayout<ElementB, typename Base::LayoutTagL1B>(embed, n),
                                           Arch::PositionL1{});
                auto l0K = tla::MakeTensor(this->l0BTensor[bank],
                                           tla::MakeLayout<ElementB, typename Base::LayoutTagL0B>(embed, n),
                                           Arch::PositionL0B{});
                auto c = tla::MakeTensor(this->l0CTensor[bank], tla::MakeLayoutL0C(rows, n), Arch::PositionL0C{});
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvent);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bEvent);
                this->copyL1ToL0B.SetScale(this->mxPackedScale[group * 128U * (embed / 32U)], embed / 64U);
                this->copyL1ToL0B(l0K, l1K);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEvent);
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(scaleFlag);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(bank);
                this->tileMmad(c, l0Q, l0K, RoundUp(rows, 16U), n, embed, true);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bEvent);
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(bank);
                EmitScore(output, rows, n, group, bank, ready);
                // Submit all remaining K groups after MMAD0. Group 3 reuses
                // slot 0 only after its MTE1 read releases the slot; MMAD0
                // continues from L0B. Each group retains its own ready event.
                if (group == 0U || this->l1BBufNum != 3U) {
                    const uint32_t begin = group + 1U;
                    const uint32_t end = this->l1BBufNum == 3U ? Groups : begin + 1U;
                    for (uint32_t next = begin; next < end && next * 128U < columns; ++next) {
                        PreloadGroup(key, cursor, min(128U, columns - next * 128U), embed, next, blockY, tokenStride);
                    }
                }
            } else {
                AscendC::CrossCoreSetFlag<4, PIPE_MTE1>(scaleFlag);
            }
        }
        qResident = true;
        this->template SetCrossCoreSync<4, PIPE_FIX>(ready);
    }

    template <class TensorB>
    __aicore__ inline void Preload(TensorB& key, const Tile::GbsaMxPackedCursor& cursor, uint32_t columns,
                                   uint32_t embed, uint32_t tile, uint32_t kvBaseTile, uint32_t blockY,
                                   uint64_t tokenStride)
    {
        if constexpr (Base::L0_TILE_N == 128U) {
            if (kvBaseTile == 512U || kvBaseTile == 256U || kvBaseTile == 128U) {
                PreloadGroup(key, cursor, min(128U, columns), embed, 0U, blockY, tokenStride);
                prefetched = true;
                return;
            }
        }
        const uint32_t bank = tile % this->l1BBufNum;
        const uint32_t event = bank + 1U;
        auto l1K =
            tla::MakeTensor(this->l1BTensor[bank],
                            tla::MakeLayout<ElementB, typename Base::LayoutTagL1B>(embed, columns), Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event);
        this->mxPackedCursor = cursor;
        this->SparseKL1TileNLoad(key, l1K, {}, tile, kvBaseTile, blockY, columns, embed, 0U, 0U, {}, 0U, 0U,
                                 tokenStride, 0U, 0U, {}, 0U);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event);
        prefetched = true;
    }

    template <class TensorB, class TensorC>
    __aicore__ inline void operator()(TensorB& key, TensorC& output, AscendC::GlobalTensor<int32_t> indices,
                                      GemmCoord shape, uint32_t tile, uint32_t kvBaseTile, uint32_t blockY,
                                      uint64_t prefixA, uint64_t prefixB, Arch::CrossCoreFlag ready,
                                      uint64_t tokenStride)
    {
        if constexpr (Base::L0_TILE_N == 128U) {
            if (kvBaseTile == 512U) {
                ComputeGrouped<4U>(key, output, shape, tile, blockY, ready, tokenStride);
                return;
            }
            if (kvBaseTile == 256U) {
                ComputeGrouped<2U>(key, output, shape, tile, blockY, ready, tokenStride);
                return;
            }
            if (kvBaseTile == 128U) {
                ComputeGrouped<1U>(key, output, shape, tile, blockY, ready, tokenStride);
                return;
            }
        }
        if (!residentQuery) {
            Base::operator()(key, output, indices, shape, tile, kvBaseTile, blockY, prefixA, prefixB, ready,
                             tokenStride);
            return;
        }
        // Retain the non-grouped helper's compile-time D192/256 geometry.
        constexpr uint32_t embed = ACTUAL_HEAD_DIM;
        const uint32_t rows = shape[0], columns = shape[1];
        const uint32_t slices = CeilDiv(columns, Base::L0_TILE_N);
        const uint32_t cStart = tile * CeilDiv(kvBaseTile, Base::L0_TILE_N);
        const uint32_t kBank = tile % this->l1BBufNum;
        const uint32_t kEvent = kBank + 1U;
        auto l1K =
            tla::MakeTensor(this->l1BTensor[kBank],
                            tla::MakeLayout<ElementB, typename Base::LayoutTagL1B>(embed, columns), Arch::PositionL1{});
        if (!prefetched)
            Preload(key, this->mxPackedCursor, columns, embed, tile, kvBaseTile, blockY, tokenStride);
        this->copyL1ToL0B.SetScale(this->mxPackedScale, embed / 64U);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvent);
        prefetched = false;

        constexpr uint32_t kSlices = (embed + Base::L0_TILE_K - 1U) / Base::L0_TILE_K;
        auto l1Q =
            tla::MakeTensor(this->l1ATensor[0], tla::MakeLayout<ElementA, typename Base::LayoutTagL1A>(rows, embed),
                            Arch::PositionL1{});
        for (uint32_t n = 0U; n < slices; ++n) {
            const uint32_t nAct = min(Base::L0_TILE_N, columns - n * Base::L0_TILE_N);
            const uint32_t cBank = (cStart + n) % Base::L0_STAGES;
            auto c = tla::MakeTensor(this->l0CTensor[cBank], tla::MakeLayoutL0C(rows, nAct), Arch::PositionL0C{});
            for (uint32_t k = 0U; k < kSlices; ++k) {
                const uint32_t kAct = min(Base::L0_TILE_K, embed - k * Base::L0_TILE_K);
                const uint32_t aBank = (prefixA + k) % Base::L0_STAGES;
                const uint32_t bBank = (prefixB + n * kSlices + k) % Base::L0_STAGES;
                const uint32_t bEvent = bBank + 2U;
                auto srcK =
                    GetTile(l1K, tla::MakeCoord(k * Base::L0_TILE_K, n * Base::L0_TILE_N), tla::MakeShape(kAct, nAct));
                auto l0K = tla::MakeTensor(this->l0BTensor[bBank],
                                           tla::MakeLayout<ElementB, typename Base::LayoutTagL0B>(kAct, nAct),
                                           Arch::PositionL0B{});
                // Q slices start at 48KiB; a second K slice starts at 56KiB.
                // D192: one K slice occupies 48..54KiB (at most 32 rows).
                // PV uses at most8KiB at0 and32KiB, so it cannot overwrite Q.
                auto l0Q = tla::MakeTensor(this->l0ATensor[0][48U * 1024U + k * 8192U],
                                           tla::MakeLayout<ElementA, typename Base::LayoutTagL0A>(rows, kAct),
                                           Arch::PositionL0A{});
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(bEvent);
                this->copyL1ToL0B(l0K, srcK);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                if (n + 1U == slices && k + 1U == kSlices) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kEvent);
                }
                if (n == 0U && !qResident) {
                    auto srcQ = GetTile(l1Q, tla::MakeCoord(0, k * Base::L0_TILE_K), tla::MakeShape(rows, kAct));
                    this->copyL1ToL0A(l0Q, srcQ);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(aBank);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(aBank);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(bEvent);
                if (k == 0U)
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cBank);
                // Wide K128 uses the standard accumulating MMAD and its
                // documented small-MN barriers. SingleK256 stays initialized.
                this->tileMmad(c, l0Q, l0K, RoundUp(rows, 16U), nAct, kAct, k == 0U);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(bEvent);
            }
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cBank);
            EmitScore(output, rows, nAct, n, cBank, ready);
        }
        qResident = true;
        this->template SetCrossCoreSync<4, PIPE_FIX>(ready);
    }
};

} // namespace NpuArch::Gemm::Block
#endif
