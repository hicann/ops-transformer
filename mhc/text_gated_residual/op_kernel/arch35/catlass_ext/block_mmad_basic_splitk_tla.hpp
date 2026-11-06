/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */
/**
 * BlockMmadTla specialization for MmadAscend950BasicSplitK.
 * Inner loop aligned to blaze block_mmad_matmul_basic_split_k.h using CANN tensor_api.
 */
#ifndef CATLASS_GEMM_BLOCK_BLOCK_MMAD_BASIC_SPLITK_TLA_HPP
#define CATLASS_GEMM_BLOCK_BLOCK_MMAD_BASIC_SPLITK_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "hc_dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "tile_copy_tensor_basic_splitk.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "tensor_api/tensor.h"

#ifndef C0_EXP_TILECOPY
#define C0_EXP_TILECOPY 0
#endif
#ifndef C0_EXP_MMAD
#define C0_EXP_MMAD 0
#endif
#ifndef C0_EXP_L0_EVENTS
#define C0_EXP_L0_EVENTS 0
#endif
#ifndef C0_EXP_L1_LAYOUT
#define C0_EXP_L1_LAYOUT 0
#endif
#ifndef C0_EXP_L1_PIPE
#define C0_EXP_L1_PIPE 0
#endif

namespace Catlass::Gemm::Block {

template <class ArchTag_, bool ENABLE_UNIT_FLAG_, uint32_t L0C_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_,
          uint32_t L0B_STAGES_, bool ENABLE_MM_LAYOUT_TRANSFORM_, bool ENABLE_INNER_SPLIT_K_, class L1TileShape_,
          class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_, class TileCopy_,
          class TileMmad_>
struct BlockMmadTla<MmadAscend950BasicSplitK<ArchTag_, ENABLE_UNIT_FLAG_, L0C_STAGES_, L1_STAGES_, L0A_STAGES_,
                                             L0B_STAGES_, ENABLE_MM_LAYOUT_TRANSFORM_, ENABLE_INNER_SPLIT_K_>,
                    L1TileShape_, L0TileShape_, ElementA_, ElementB_, ElementC_, ElementBias_, TileCopy_, TileMmad_> {
public:
    using DispatchPolicy = MmadAscend950BasicSplitK<ArchTag_, ENABLE_UNIT_FLAG_, L0C_STAGES_, L1_STAGES_, L0A_STAGES_,
                                                    L0B_STAGES_, ENABLE_MM_LAYOUT_TRANSFORM_, ENABLE_INNER_SPLIT_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = ElementA_;
    using LayoutA = typename TileCopy::LayoutA;
    using ElementB = ElementB_;
    using LayoutB = typename TileCopy::LayoutB;
    using ElementC = ElementC_;
    using LayoutC = typename TileCopy::LayoutC;
    using ElementBias = ElementBias_;
    using TileMmad = TileMmad_;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_MM_LAYOUT_TRANSFORM = DispatchPolicy::ENABLE_MM_LAYOUT_TRANSFORM;
    static constexpr uint32_t L1_STAGES = DispatchPolicy::L1_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;

    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});
    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape{});

    static constexpr uint16_t MTE1_MTE2_EVENT_ID_NUM = 4;
    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr uint16_t FIRST_FLAG = 1;
    static constexpr uint16_t SIXTH_FLAG = 6;
    static constexpr uint16_t SEVENTH_FLAG = 7;
    static constexpr uint64_t QUADRUPLE_BUFFER_COUNT = 4;
    static constexpr uint64_t DOUBLE_BUFFER_COUNT = 2;
    static constexpr uint8_t kUnitFlagKeep = 2;
    static constexpr uint8_t kUnitFlagUpdate = 3;

    using MakeLayoutAL1 =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<ElementA>>;
    using MakeLayoutBL1 =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<ElementB>>;
    using MakeLayoutBGm =
        AscendC::Te::FrameLayoutFormat<AscendC::Te::DNLayoutPtn, AscendC::Te::LayoutTraitDefault<ElementB>>;

    using PackedCopy = Gemm::Tile::PackedTileCopyTla<ArchTag, ElementA, layout::RowMajor, ElementB, layout::ColumnMajor,
                                                     ElementC, layout::RowMajor, ElementBias>;
    using PackedMmad = Gemm::Tile::TileMmadTla<ArchTag, ElementA, layout::zN>;
    using LayoutTagL1APacked = typename PackedCopy::LayoutTagL1A;
    using LayoutTagL1BPacked = typename PackedCopy::LayoutTagL1B;
    using LayoutTagL0APacked = typename PackedCopy::LayoutTagL0A;
    using LayoutTagL0BPacked = typename PackedCopy::LayoutTagL0B;

    CATLASS_DEVICE
    BlockMmadTla(Arch::Resource<ArchTag>& resource, uint32_t l1BufAddrStart = 0)
    {
        if ASCEND_IS_AIC {
            (void)resource.l1Buf.template GetBufferByByte<uint8_t>(0);
            (void)resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            (void)resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            (void)resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);

            uint64_t aL1One = static_cast<uint64_t>(L1_TILE_M) * L1_TILE_K * sizeof(ElementA);
            uint64_t bL1One = static_cast<uint64_t>(L1_TILE_N) * L1_TILE_K * sizeof(ElementB);
            constexpr uint64_t quarterL1 = ArchTag::L1_SIZE / QUADRUPLE_BUFFER_COUNT;
            constexpr uint64_t halfL0 = ArchTag::L0A_SIZE / DOUBLE_BUFFER_COUNT;
            constexpr uint64_t halfL0C = ArchTag::L0C_SIZE / DOUBLE_BUFFER_COUNT;
            for (uint32_t i = 0; i < L1_STAGES; ++i) {
                if constexpr (C0_EXP_L1_LAYOUT != 0) {
                    aL1Buffer_[i] = static_cast<uint64_t>(l1BufAddrStart) + aL1One * i;
                    bL1Buffer_[i] = static_cast<uint64_t>(l1BufAddrStart) + aL1One * L1_STAGES + bL1One * i;
                } else {
                    aL1Buffer_[i] =
                        static_cast<uint64_t>(l1BufAddrStart) + quarterL1 * (QUADRUPLE_BUFFER_COUNT / L1_STAGES) * i;
                    bL1Buffer_[i] = aL1Buffer_[i] + aL1One;
                }
                l1ALocal_[i] = resource.l1Buf.template GetBufferByByte<ElementA>(static_cast<uint32_t>(aL1Buffer_[i]));
                l1BLocal_[i] = resource.l1Buf.template GetBufferByByte<ElementB>(static_cast<uint32_t>(bL1Buffer_[i]));
            }
            l0ALocal_[0] = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
            l0ALocal_[1] = resource.l0ABuf.template GetBufferByByte<ElementA>(static_cast<uint32_t>(halfL0));
            l0BLocal_[0] = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
            l0BLocal_[1] = resource.l0BBuf.template GetBufferByByte<ElementB>(static_cast<uint32_t>(halfL0));
            l0CLocal_[0] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
            l0CLocal_[1] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(static_cast<uint32_t>(halfL0C));

            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            if constexpr (C0_EXP_L0_EVENTS != 0) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
            }
            if constexpr (ENABLE_MM_LAYOUT_TRANSFORM) {
                AscendC::SetMMLayoutTransform(true);
            }
            l0cPingPong_ = 0;
            l0PingPong_ = 0;
            abL1LoopCnt_ = 0;
        }
    }

    CATLASS_DEVICE
    ~BlockMmadTla()
    {
        if ASCEND_IS_AIC {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; ++i) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            if constexpr (C0_EXP_L0_EVENTS != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
            }
            if constexpr (ENABLE_MM_LAYOUT_TRANSFORM) {
                AscendC::SetMMLayoutTransform(false);
            }
        }
    }

    template <class TensorC>
    CATLASS_DEVICE void SynchronizeBlock()
    {}

    template <class TensorA, class TensorB, class TensorC, class TensorBias = EmptyClass>
    CATLASS_DEVICE void operator()(TensorA& tensorA, TensorB& tensorB, TensorC& tensorC, GemmCoord const& actualShape,
                                   TensorBias const& = {}, Callback const& = {}, Callback const& = {},
                                   uint64_t kPartIdx = 0, uint64_t kPartRound = 1)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<TensorC>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        typename TileCopy::CopyL1ToL0A copyL1ToL0A;
        typename TileCopy::CopyL1ToL0B copyL1ToL0B;
        CopyL0CToDst copyL0CToDst;
        using CopyGmToL1APacked = typename PackedCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1BPacked = typename PackedCopy::template CopyGmToL1B<TensorB>;
        using CopyL0CPacked = typename PackedCopy::template CopyL0CToDst<TensorC>;
        CopyGmToL1APacked copyGmToL1APacked;
        CopyGmToL1BPacked copyGmToL1BPacked;
        typename PackedCopy::CopyL1ToL0A copyL1ToL0APacked;
        typename PackedCopy::CopyL1ToL0B copyL1ToL0BPacked;
        CopyL0CPacked copyL0CPacked;
        PackedMmad packedMmad;
        (void)kPartIdx;
        (void)kPartRound;

        uint64_t curML1 = actualShape.m();
        uint64_t curNL1 = actualShape.n();
        uint64_t kFull = actualShape.k();
        uint64_t curM = Min(curML1, static_cast<uint64_t>(L0_TILE_M));
        uint64_t curN = Min(curNL1, static_cast<uint64_t>(L0_TILE_N));

        auto gmA = MakeTeGmA(tensorA, curML1, kFull);
        auto gmB = MakeTeGmB(tensorB, kFull, curNL1);
        auto gmC = MakeTeGmC(tensorC, curML1, curNL1);

        uint64_t splitK = kFull;
        uint64_t splitKRound = 1;
        uint64_t splitKTail = kFull;
        if constexpr (DispatchPolicy::ENABLE_INNER_SPLIT_K) {
            splitK = (kFull > DispatchPolicy::FP32_K_SWITCH_BASE) ? DispatchPolicy::FP32_SPLIT_K_BASE2 :
                                                                    DispatchPolicy::FP32_SPLIT_K_BASE1;
            splitKRound = kFull / splitK;
            splitKTail = (kFull % splitK) + splitK;
            if (splitKRound == 0) {
                splitKRound = 1;
                splitK = kFull;
                splitKTail = kFull;
            }
        }

        constexpr uint64_t halfL0C = ArchTag::L0C_SIZE / DOUBLE_BUFFER_COUNT;
        constexpr uint64_t halfL0 = ArchTag::L0A_SIZE / DOUBLE_BUFFER_COUNT;
        const bool enableL0cPingPong = (L0C_STAGES > 1);

        for (uint64_t splitIdx = 0; splitIdx < splitKRound; ++splitIdx) {
            uint64_t coordK = splitIdx * splitK;
            uint64_t blkK = (splitIdx + 1 == splitKRound) ? splitKTail : splitK;
            auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(coordK)),
                                      AscendC::Te::MakeShape(static_cast<int64_t>(curML1), static_cast<int64_t>(blkK)));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(coordK), static_cast<int64_t>(0)),
                                      AscendC::Te::MakeShape(static_cast<int64_t>(blkK), static_cast<int64_t>(curNL1)));

            uint64_t l0cOffset = (l0cPingPong_ & 1) * halfL0C;
            auto layoutL0C = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, 16>(curM, curN);
            auto tensorL0C = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);

            uint64_t kL1 = Min(blkK, static_cast<uint64_t>(L1_TILE_K));
            uint64_t kL1Iter = CeilDiv(blkK, kL1);

            if constexpr (C0_EXP_L1_PIPE != 0) {
                uint64_t curKL1_0 = (kL1Iter == 1) ? blkK : kL1;
                uint64_t l1Buf0 = abL1LoopCnt_ & (L1_STAGES - 1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1Buf0));
                this->CopyGmL1At(tensorA, tensorB, gmBlockA, gmBlockB, coordK, kL1, curM, curN, 0, l1Buf0, curKL1_0);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1Buf0));
                for (uint64_t iter0 = 0; iter0 < kL1Iter; ++iter0) {
                    uint64_t curKL1 = (iter0 + 1 == kL1Iter) ? (blkK - kL1 * iter0) : kL1;
                    uint64_t l1BufId = (abL1LoopCnt_ + iter0) & (L1_STAGES - 1);
                    if (iter0 + 1 < kL1Iter) {
                        uint64_t nextKL1 = (iter0 + 2 == kL1Iter) ? (blkK - kL1 * (iter0 + 1)) : kL1;
                        uint64_t l1Next = (abL1LoopCnt_ + iter0 + 1) & (L1_STAGES - 1);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1Next));
                        this->CopyGmL1At(tensorA, tensorB, gmBlockA, gmBlockB, coordK, kL1, curM, curN, iter0 + 1,
                                         l1Next, nextKL1);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1Next));
                    }
                    this->RunL0OnL1(tensorL0C, curM, curN, kL1Iter, halfL0, iter0, l1BufId, curKL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufId));
                }
                abL1LoopCnt_ += kL1Iter;
            } else {
                for (uint64_t iter0 = 0; iter0 < kL1Iter; ++iter0) {
                    uint64_t curKL1 = (iter0 + 1 == kL1Iter) ? (blkK - kL1 * iter0) : kL1;
                    uint64_t l1BufId = abL1LoopCnt_ & (L1_STAGES - 1);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufId));
                    this->CopyGmL1At(tensorA, tensorB, gmBlockA, gmBlockB, coordK, kL1, curM, curN, iter0, l1BufId,
                                     curKL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1BufId));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1BufId));
                    this->RunL0OnL1(tensorL0C, curM, curN, kL1Iter, halfL0, iter0, l1BufId, curKL1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<uint16_t>(l1BufId));
                    abL1LoopCnt_++;
                }
            }

            AscendC::PipeBarrier<PIPE_FIX>();
            bool doAtomic = (splitIdx != 0);
            if (!DispatchPolicy::ENABLE_INNER_SPLIT_K && kPartRound > 1) {
                doAtomic = (kPartIdx != 0);
            }
            if (doAtomic) {
                AscendC::SetAtomicAdd<float>();
            }
            if constexpr (C0_EXP_TILECOPY != 0) {
                auto layoutL0CP = tla::MakeLayoutL0C(static_cast<uint32_t>(curM), static_cast<uint32_t>(curN));
                auto tensorL0CP = tla::MakeTensor(l0CLocal_[l0cPingPong_ & 1], layoutL0CP, Arch::PositionL0C{});
                uint8_t unitFlagFix = ENABLE_UNIT_FLAG ? kUnitFlagUpdate : 0;
                copyL0CPacked(tensorC, tensorL0CP, unitFlagFix);
            } else {
                copyL0CToDst(gmC, tensorL0C);
            }
            bool lastSplit = (splitIdx + 1 == splitKRound);
            if (!DispatchPolicy::ENABLE_INNER_SPLIT_K && kPartRound > 1) {
                lastSplit = (kPartIdx + 1 == kPartRound);
            }
            if (lastSplit) {
                AscendC::DisableDmaAtomic();
            }
            if (enableL0cPingPong) {
                l0cPingPong_++;
            }
        }
    }

private:
    template <class TensorA, class TensorB, class TeBlockA, class TeBlockB>
    CATLASS_DEVICE void CopyGmL1At(TensorA& tensorA, TensorB& tensorB, TeBlockA& gmBlockA, TeBlockB& gmBlockB,
                                   uint64_t coordK, uint64_t kL1, uint64_t curM, uint64_t curN, uint64_t iter0,
                                   uint64_t l1BufId, uint64_t curKL1)
    {
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<TensorB>;
        using CopyGmToL1APacked = typename PackedCopy::template CopyGmToL1A<TensorA>;
        using CopyGmToL1BPacked = typename PackedCopy::template CopyGmToL1B<TensorB>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyGmToL1APacked copyGmToL1APacked;
        CopyGmToL1BPacked copyGmToL1BPacked;
        auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ElementA>(aL1Buffer_[l1BufId]), layoutAL1);
        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ElementB>(bL1Buffer_[l1BufId]), layoutBL1);
        if constexpr (C0_EXP_TILECOPY != 0) {
            auto layoutAL1P = tla::MakeLayout<ElementA, LayoutTagL1APacked>(static_cast<uint32_t>(curM),
                                                                            static_cast<uint32_t>(curKL1));
            auto tensorAL1P = tla::MakeTensor(l1ALocal_[l1BufId], layoutAL1P, Arch::PositionL1{});
            auto gmTileA =
                GetTile(tensorA, tla::MakeCoord(static_cast<uint32_t>(0), static_cast<uint32_t>(coordK + iter0 * kL1)),
                        tla::MakeShape(static_cast<uint32_t>(curM), static_cast<uint32_t>(curKL1)));
            copyGmToL1APacked(tensorAL1P, gmTileA);
            auto layoutBL1P = tla::MakeLayout<ElementB, LayoutTagL1BPacked>(static_cast<uint32_t>(curKL1),
                                                                            static_cast<uint32_t>(curN));
            auto tensorBL1P = tla::MakeTensor(l1BLocal_[l1BufId], layoutBL1P, Arch::PositionL1{});
            auto gmTileB =
                GetTile(tensorB, tla::MakeCoord(static_cast<uint32_t>(coordK + iter0 * kL1), static_cast<uint32_t>(0)),
                        tla::MakeShape(static_cast<uint32_t>(curKL1), static_cast<uint32_t>(curN)));
            copyGmToL1BPacked(tensorBL1P, gmTileB);
        } else {
            auto gmTileA =
                gmBlockA.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(iter0 * kL1)),
                               AscendC::Te::MakeShape(static_cast<int64_t>(curM), static_cast<int64_t>(curKL1)));
            copyGmToL1A(tensorAL1, gmTileA);
            auto gmTileB =
                gmBlockB.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(iter0 * kL1), static_cast<int64_t>(0)),
                               AscendC::Te::MakeShape(static_cast<int64_t>(curKL1), static_cast<int64_t>(curN)));
            copyGmToL1B(tensorBL1, gmTileB);
        }
        (void)tensorAL1;
        (void)tensorBL1;
    }

    template <class TensorL0C>
    CATLASS_DEVICE void RunL0OnL1(TensorL0C& tensorL0C, uint64_t curM, uint64_t curN, uint64_t kL1Iter, uint64_t halfL0,
                                  uint64_t iter0, uint64_t l1BufId, uint64_t curKL1)
    {
        typename TileCopy::CopyL1ToL0A copyL1ToL0A;
        typename TileCopy::CopyL1ToL0B copyL1ToL0B;
        typename PackedCopy::CopyL1ToL0A copyL1ToL0APacked;
        typename PackedCopy::CopyL1ToL0B copyL1ToL0BPacked;
        PackedMmad packedMmad;
        auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ElementA>(aL1Buffer_[l1BufId]), layoutAL1);
        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, ElementB>(bL1Buffer_[l1BufId]), layoutBL1);
        uint64_t kL0Iter = CeilDiv(curKL1, static_cast<uint64_t>(L0_TILE_K));
        constexpr bool enableL0PingPong = (L0A_STAGES > 1 && L0B_STAGES > 1);
        for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * L0_TILE_K) : L0_TILE_K;
            uint64_t l0Offset = enableL0PingPong ? halfL0 * (l0PingPong_ & 1) : 0;
            uint16_t ping = static_cast<uint16_t>(l0PingPong_ & 1);
            uint16_t mte1Flag = enableL0PingPong ? static_cast<uint16_t>(ping + SIXTH_FLAG) : SIXTH_FLAG;
            uint16_t l0AEvt = ping;
            uint16_t l0BEvt = static_cast<uint16_t>(ping + 2);

            auto layoutAL0 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<ElementA>>(
                    curM, curK0);
            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, ElementA>(l0Offset), layoutAL0);
            auto layoutBL0 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<ElementB>>(curK0,
                                                                                                                  curN);
            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, ElementB>(l0Offset), layoutBL0);

            if constexpr (C0_EXP_L0_EVENTS != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvt);
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1Flag);
            }
            if constexpr (C0_EXP_L1_PIPE != 0) {
                if (iter1 == 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<uint16_t>(l1BufId));
                }
            }
            if constexpr (C0_EXP_TILECOPY != 0) {
                auto layoutAL1P = tla::MakeLayout<ElementA, LayoutTagL1APacked>(static_cast<uint32_t>(curM),
                                                                                static_cast<uint32_t>(curKL1));
                auto tensorAL1P = tla::MakeTensor(l1ALocal_[l1BufId], layoutAL1P, Arch::PositionL1{});
                auto tileL1A = GetTile(
                    tensorAL1P, tla::MakeCoord(static_cast<uint32_t>(0), static_cast<uint32_t>(iter1 * L0_TILE_K)),
                    tla::MakeShape(static_cast<uint32_t>(curM), static_cast<uint32_t>(curK0)));
                auto layoutAL0P = tla::MakeLayout<ElementA, LayoutTagL0APacked>(static_cast<uint32_t>(curM),
                                                                                static_cast<uint32_t>(curK0));
                auto tensorAL0P = tla::MakeTensor(l0ALocal_[ping], layoutAL0P, Arch::PositionL0A{});
                copyL1ToL0APacked(tensorAL0P, tileL1A);
            } else {
                auto tensorBlockAL1 = tensorAL1.Slice(
                    AscendC::Te::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(iter1 * L0_TILE_K)),
                    AscendC::Te::MakeShape(static_cast<int64_t>(curM), static_cast<int64_t>(curK0)));
                copyL1ToL0A(tensorAL0, tensorBlockAL1);
            }
            if constexpr (C0_EXP_L0_EVENTS != 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0AEvt);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvt);
            }

            if constexpr (C0_EXP_TILECOPY != 0) {
                auto layoutBL1P = tla::MakeLayout<ElementB, LayoutTagL1BPacked>(static_cast<uint32_t>(curKL1),
                                                                                static_cast<uint32_t>(curN));
                auto tensorBL1P = tla::MakeTensor(l1BLocal_[l1BufId], layoutBL1P, Arch::PositionL1{});
                auto tileL1B = GetTile(
                    tensorBL1P, tla::MakeCoord(static_cast<uint32_t>(iter1 * L0_TILE_K), static_cast<uint32_t>(0)),
                    tla::MakeShape(static_cast<uint32_t>(curK0), static_cast<uint32_t>(curN)));
                auto layoutBL0P = tla::MakeLayout<ElementB, LayoutTagL0BPacked>(static_cast<uint32_t>(curK0),
                                                                                static_cast<uint32_t>(curN));
                auto tensorBL0P = tla::MakeTensor(l0BLocal_[ping], layoutBL0P, Arch::PositionL0B{});
                copyL1ToL0BPacked(tensorBL0P, tileL1B);
            } else {
                auto tensorBlockBL1 = tensorBL1.Slice(
                    AscendC::Te::MakeCoord(static_cast<int64_t>(iter1 * L0_TILE_K), static_cast<int64_t>(0)),
                    AscendC::Te::MakeShape(static_cast<int64_t>(curK0), static_cast<int64_t>(curN)));
                copyL1ToL0B(tensorBL0, tensorBlockBL1);
            }

            if constexpr (C0_EXP_L0_EVENTS != 0) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BEvt);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0AEvt);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BEvt);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(mte1Flag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(mte1Flag);
            }

            bool initCmatrix = (iter0 == 0 && iter1 == 0);
            uint8_t unitFlag = 0;
            if constexpr (ENABLE_UNIT_FLAG) {
                unitFlag = ((iter0 + 1 == kL1Iter && iter1 + 1 == kL0Iter) ? kUnitFlagUpdate : kUnitFlagKeep);
            }
            if constexpr (C0_EXP_MMAD != 0) {
                auto layoutAL0P = tla::MakeLayout<ElementA, LayoutTagL0APacked>(static_cast<uint32_t>(curM),
                                                                                static_cast<uint32_t>(curK0));
                auto tensorAL0P = tla::MakeTensor(l0ALocal_[ping], layoutAL0P, Arch::PositionL0A{});
                auto layoutBL0P = tla::MakeLayout<ElementB, LayoutTagL0BPacked>(static_cast<uint32_t>(curK0),
                                                                                static_cast<uint32_t>(curN));
                auto tensorBL0P = tla::MakeTensor(l0BLocal_[ping], layoutBL0P, Arch::PositionL0B{});
                auto layoutL0CP = tla::MakeLayoutL0C(static_cast<uint32_t>(curM), static_cast<uint32_t>(curN));
                auto tensorL0CP = tla::MakeTensor(l0CLocal_[l0cPingPong_ & 1], layoutL0CP, Arch::PositionL0C{});
                packedMmad(tensorL0CP, tensorAL0P, tensorBL0P, static_cast<uint32_t>(curM), static_cast<uint32_t>(curN),
                           static_cast<uint32_t>(curK0), initCmatrix, unitFlag);
            } else {
                constexpr auto mmadAtom =
                    AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
                AscendC::Te::MmadParams mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                                   static_cast<uint16_t>(curK0), unitFlag, initCmatrix};
                AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
            }
            if constexpr (C0_EXP_L0_EVENTS != 0) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEvt);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEvt);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1Flag);
            }
            l0PingPong_++;
        }
    }

    template <class TensorA>
    CATLASS_DEVICE auto MakeTeGmA(TensorA const& tensorA, uint64_t m, uint64_t k)
    {
        int64_t elemOff = tensorA.layout()(tensorA.coord());
        auto ptr = tensorA.data().GetPhyAddr(static_cast<uint64_t>(elemOff));
        int64_t ld = static_cast<int64_t>(tla::get<0>(tensorA.stride()));
        auto layout =
            AscendC::Te::MakePatternLayout<AscendC::Te::NDLayoutPtn, AscendC::Te::LayoutTraitDefault<ElementA>>(
                AscendC::Te::MakeShape(static_cast<int64_t>(m), static_cast<int64_t>(k)),
                AscendC::Te::MakeStride(ld, AscendC::Std::Int<1>{}));
        return AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr(ptr), layout);
    }

    template <class TensorB>
    CATLASS_DEVICE auto MakeTeGmB(TensorB const& tensorB, uint64_t k, uint64_t n)
    {
        int64_t elemOff = tensorB.layout()(tensorB.coord());
        auto ptr = tensorB.data().GetPhyAddr(static_cast<uint64_t>(elemOff));
        // DN / ColumnMajor: N-stride is parent K. Using tile K as ld (MakeLayoutBGm(k,n))
        // breaks multi-core SplitK: GetTile K-slice keeps stride=(1, originK).
        int64_t ld = static_cast<int64_t>(tla::get<1>(tensorB.stride()));
        if (ld <= 0) {
            ld = static_cast<int64_t>(k);
        }
        auto layoutB =
            AscendC::Te::MakePatternLayout<AscendC::Te::DNLayoutPtn, AscendC::Te::LayoutTraitDefault<ElementB>>(
                AscendC::Te::MakeShape(static_cast<int64_t>(k), static_cast<int64_t>(n)),
                AscendC::Te::MakeStride(AscendC::Std::Int<1>{}, ld));
        return AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr(ptr), layoutB);
    }

    template <class TensorC>
    CATLASS_DEVICE auto MakeTeGmC(TensorC const& tensorC, uint64_t m, uint64_t n)
    {
        int64_t elemOff = tensorC.layout()(tensorC.coord());
        auto ptr = tensorC.data().GetPhyAddr(static_cast<uint64_t>(elemOff));
        int64_t ld = static_cast<int64_t>(tla::get<0>(tensorC.stride()));
        auto layout =
            AscendC::Te::MakePatternLayout<AscendC::Te::NDLayoutPtn, AscendC::Te::LayoutTraitDefault<ElementC>>(
                AscendC::Te::MakeShape(static_cast<int64_t>(m), static_cast<int64_t>(n)),
                AscendC::Te::MakeStride(ld, AscendC::Std::Int<1>{}));
        return AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr(ptr), layout);
    }

    uint64_t aL1Buffer_[4]{0};
    uint64_t bL1Buffer_[4]{0};
    AscendC::LocalTensor<ElementA> l1ALocal_[4];
    AscendC::LocalTensor<ElementB> l1BLocal_[4];
    AscendC::LocalTensor<ElementA> l0ALocal_[2];
    AscendC::LocalTensor<ElementB> l0BLocal_[2];
    AscendC::LocalTensor<ElementAccumulator> l0CLocal_[2];
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
};

} // namespace Catlass::Gemm::Block

#endif
