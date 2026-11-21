/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "a2_fullb_common.hpp"
#include "a2_fullb_c0_as2.hpp"
#include "a2_fullb_reduce_v1.hpp"
#include "a2_fullb_c1.hpp"
namespace Catlass::Gemm::Kernel::SmallModelA2 {
template <class Element, uint32_t C1SwizzleDir, class ArchTag_ = Arch::AtlasA2,
          class Tiling = HcForwardMulSumTilingData, bool TwoWaves = false, bool Direct128 = false>
struct HcForwardSmallFullBKernel : HcA2FullBCommon<Element, C1SwizzleDir, true, ArchTag_, Tiling> {
    static constexpr bool PreloadSmall = true;
    using Base = HcA2FullBCommon<Element, C1SwizzleDir, true, ArchTag_, Tiling>;
    using Params = typename Base::Params;
    using ArchTag = typename Base::ArchTag;
    using LayoutTagC = typename Base::LayoutTagC;
    using DispatchPolicyC0 = typename Base::DispatchPolicyC0;
    using TileCopyC0 = typename Base::TileCopyC0;
    using BlockMmadC1 = typename Base::BlockMmadC1;
    using BlockMmadC1A1 = typename Base::BlockMmadC1A1;
    using BlockSchedulerC1 = typename Base::BlockSchedulerC1;
    using Base::kIsA2;
    using Base::L0B_STAGES_C0;
    using Base::L1M0;
    using Base::L1N0;
    using Base::L1K0;
    using Base::L1M1;
    using Base::L1N1;
    using Base::L1K1;
    using Base::L0K0;
    using Base::RunMmad;
    using Base::C0PreferN128;
    using Base::C0PreferN32;
    template <uint32_t kTileM = L1M0, uint32_t kTileN = L1N0, class TensorA, class TensorBDown, class TensorBGate,
              class TensorC, class TensorGate>
    CATLASS_DEVICE void RunSmallNCubesC0Direct(Arch::Resource<ArchTag>& resource, TensorA const& tensorA,
                                               TensorBDown const& tensorBDown, TensorBGate const& tensorBGate,
                                               TensorC const& tensorC0, TensorGate const& tensorGate, uint32_t row,
                                               uint32_t actualM, uint32_t rDim, uint32_t nConcat, uint32_t kDim,
                                               uint32_t blockIdx, uint32_t usedAic, bool useCombine, uint32_t rowC0 = 0)
    {
        static_assert(L0K0 * kTileN * static_cast<int64_t>(sizeof(Element)) * L0B_STAGES_C0 <= 64u * 1024u,
                      "C0 wide L0B");
        static_assert(kTileM * L0K0 * static_cast<int64_t>(sizeof(Element)) * 2u <= 64u * 1024u, "C0 wide L0A");
        using L1Wide = tla::Shape<tla::Int<kTileM>, tla::Int<kTileN>, tla::Int<L1K0>>;
        using L0Wide = tla::Shape<tla::Int<kTileM>, tla::Int<kTileN>, tla::Int<L0K0>>;
        using BlockMmadWide =
            Gemm::Block::BlockMmadTla<DispatchPolicyC0, L1Wide, L0Wide, Element, Element, Element, void, TileCopyC0>;
        BlockMmadWide mmad(resource);
        uint32_t mTiles = (actualM + kTileM - 1) / kTileM;
        uint32_t nTiles = (nConcat + kTileN - 1) / kTileN;
        // A2：纯 gate 的 N 片排在队尾。多出来的 loop 落在 AIC 0、1 上，这两次只算 gate。
        uint32_t firstGateN = (rDim + kTileN - 1) / kTileN;
        if (firstGateN > nTiles) {
            firstGateN = nTiles;
        }
        if (firstGateN > 0 && firstGateN * kTileN < rDim) {
            firstGateN = nTiles;
        }
        bool gateTail = kIsA2 && useCombine && firstGateN < nTiles;
        GemmCoord problem{actualM, nConcat, kDim};
        BlockSchedulerC1 sched(problem, MatrixCoord{kTileM, kTileN});
        uint32_t coreLoops = mTiles * nTiles;
        if (coreLoops == 0) {
            coreLoops = sched.GetCoreLoops();
        }
        uint32_t downTiles = mTiles * firstGateN;
        uint32_t gateTiles = coreLoops - downTiles;
        const uint32_t cDim = nConcat > rDim ? (nConcat - rDim) : 0;
        for (uint32_t loopIdx = blockIdx; loopIdx < coreLoops; loopIdx += usedAic) {
            uint32_t mIdx = 0;
            uint32_t nIdx = 0;
            if (gateTail) {
                // gate-only 的 N 片排到队首：分到它的核先把极小的 gate 算完再算 down，
                // down 均摊到其余核。按 stride 领循环时多出的循环固定落在低号核上，
                // 这些核 down 任务最少、且以 gate 开头。
                if (loopIdx < gateTiles) {
                    uint32_t g = loopIdx;
                    mIdx = g % mTiles;
                    nIdx = firstGateN + g / mTiles;
                } else {
                    uint32_t d = loopIdx - gateTiles;
                    mIdx = d % mTiles;
                    nIdx = d / mTiles;
                }
            } else {
                GemmCoord blockCoord = sched.GetBlockCoord(loopIdx);
                mIdx = blockCoord.m();
                nIdx = blockCoord.n();
            }
            uint32_t mOff = mIdx * kTileM;
            uint32_t nOff = nIdx * kTileN;
            uint32_t tileM = actualM - mOff < kTileM ? actualM - mOff : kTileM;
            uint32_t actualN = nConcat - nOff < kTileN ? nConcat - nOff : kTileN;
            if (nOff >= nConcat || actualN == 0 || tileM == 0) {
                continue;
            }
            if (nOff + actualN > nConcat) {
                actualN = nConcat - nOff;
            }
            if (nOff < rDim) {
                uint32_t nDown = actualN;
                if (nOff + nDown > rDim) {
                    nDown = rDim - nOff;
                }
                auto tileC = GetTile(tensorC0, tla::MakeCoord(mOff + rowC0, nOff), tla::MakeShape(tileM, nDown));
                auto tileA = GetTile(tensorA, tla::MakeCoord(mOff, 0), tla::MakeShape(tileM, kDim));
                auto tileB = GetTile(tensorBDown, tla::MakeCoord(0, nOff), tla::MakeShape(kDim, nDown));
                mmad(tileA, tileB, tileC, GemmCoord{tileM, nDown, kDim});
                if (useCombine && cDim > 0 && nOff + actualN > rDim) {
                    uint32_t nGate = (nOff + actualN) - rDim;
                    if (nGate > cDim) {
                        nGate = cDim;
                    }
                    RunMmad(mmad, tensorA, tensorBGate, tensorGate, mOff, 0, 0, tileM, nGate, kDim);
                }
            } else if (useCombine) {
                uint32_t colGate = nOff - rDim;
                if (colGate >= cDim) {
                    continue;
                }
                actualN = actualN < (cDim - colGate) ? actualN : (cDim - colGate);
                RunMmad(mmad, tensorA, tensorBGate, tensorGate, mOff, colGate, colGate, tileM, actualN, kDim);
            }
        }
        (void)row;
    }

    template <class Mmad, class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void RunSmallNCubesC1(Mmad& mmad, TensorA const& tensorA, TensorB const& tensorB,
                                         TensorC const& tensorC, uint32_t row, uint32_t actualM, uint32_t nOff,
                                         uint32_t actualN, uint32_t kDim)
    {
        auto tileB = GetTile(tensorB, tla::MakeCoord(0, nOff), tla::MakeShape(kDim, actualN));
        for (uint32_t mOff = 0; mOff < actualM; mOff += L1M1) {
            uint32_t tileM = actualM - mOff < L1M1 ? actualM - mOff : L1M1;
            auto tileA = GetTile(tensorA, tla::MakeCoord(row + mOff, 0), tla::MakeShape(tileM, kDim));
            auto tileC = GetTile(tensorC, tla::MakeCoord(row + mOff, nOff), tla::MakeShape(tileM, actualN));
            mmad(tileA, tileB, tileC, GemmCoord{tileM, actualN, kDim});
        }
    }

    template <class Mmad, class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void RunSmallC1Loops(Arch::Resource<ArchTag>& resource, TensorA const& tensorA,
                                        TensorB const& tensorB, TensorC const& tensorC, uint32_t row, uint32_t actualM,
                                        uint32_t nDim, uint32_t kDim, uint32_t blockIdx, uint32_t usedAic,
                                        uint32_t nSlice)
    {
        Mmad mmad(resource);
        for (uint32_t n = blockIdx; n < nSlice; n += usedAic) {
            uint32_t nOff = n * L1N1;
            uint32_t actualN = nDim - nOff < L1N1 ? nDim - nOff : L1N1;
            RunSmallNCubesC1(mmad, tensorA, tensorB, tensorC, row, actualM, nOff, actualN, kDim);
        }
    }

    // 一条行带且 K 已全部在 L1 时只用一级 L1A。mmad 建在 N 循环外，驻留跨 N 片保留。
    template <class TensorA, class TensorB, class TensorC>
    CATLASS_DEVICE void RunSmallC1(Arch::Resource<ArchTag>& resource, TensorA const& tensorA, TensorB const& tensorB,
                                   TensorC const& tensorC, uint32_t row, uint32_t actualM, uint32_t nDim, uint32_t kDim,
                                   uint32_t blockIdx, uint32_t usedAic, uint32_t nSlice)
    {
        if (actualM <= L1M1 && kDim <= L1K1) {
            RunSmallC1Loops<BlockMmadC1A1>(resource, tensorA, tensorB, tensorC, row, actualM, nDim, kDim, blockIdx,
                                           usedAic, nSlice);
        } else {
            RunSmallC1Loops<BlockMmadC1>(resource, tensorA, tensorB, tensorC, row, actualM, nDim, kDim, blockIdx,
                                         usedAic, nSlice);
        }
    }

    template <class Context>
    CATLASS_DEVICE void RunPreloadedC0(Context& c, uint32_t row, uint32_t rows, uint32_t slot)
    {
        using Policy = std::conditional_t<Direct128, Gemm::MmadTgrA2FullB<ArchTag, false, false, 1, false, 1, 1, 2, 2>,
                                          Gemm::MmadTgrA2SmallB<ArchTag, false, false, 1, false, 2, 1, 2, 2>>;
        static constexpr uint32_t TileM = Direct128 ? 128u : 64u;
        using L1 = tla::Shape<tla::Int<TileM>, tla::Int<160>, tla::Int<512>>;
        using L0 = tla::Shape<tla::Int<TileM>, tla::Int<160>, tla::Int<64>>;
        using Copy = Gemm::Tile::PackedTileCopyTla<ArchTag, Element, typename Base::LayoutTagA, Element,
                                                   typename Base::LayoutTagB, float, LayoutTagC, void>;
        using Mmad = Gemm::Block::BlockMmadA2FullBTla<Policy, L1, L0, Element, Element, float, void, Copy>;
        const uint32_t tasks = c.f_splitk * ((c.f_r + 159u) / 160u);
        for (uint32_t task = c.f_blockIdx; task < tasks; task += c.f_usedAic) {
            Mmad block(c.f_resource);
            const uint32_t k = task / 2, n = task % 2;
            const uint32_t ko = k * 1024, no = n * 160;
            auto b = GetTile(c.f_tensorWDown, tla::MakeCoord(ko, no), tla::MakeShape(1024u, 160u));
            auto bg = GetTile(c.f_tensorWGate, tla::MakeCoord(ko, 0), tla::MakeShape(1024u, c.f_channels));
            block.PreloadB(b);
            if (c.f_useCombine && n == 0)
                block.PreloadGate(bg);

            if (task == c.f_blockIdx) {
                HcAicWait(TwoWaves && ((row / c.f_mTask) & 1u) != 0u ? FLAG_V0_SECOND : FLAG_V0);
                // B is independent of V0. Gate the subsequent Xn DMA on V0 readiness.
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(7);
            }

            auto a = GetTile(c.f_tensorXn, tla::MakeCoord(row, ko), tla::MakeShape(rows, 1024u));
            const int64_t region = static_cast<int64_t>(c.f_mTask) * c.f_nWs * 10;
            auto ptr = reinterpret_cast<__gm__ float*>(c.f_ptrPartial) + slot * region;
            AscendC::GlobalTensor<float> gd, gg;
            gd.SetGlobalBuffer(ptr + static_cast<int64_t>(k) * rows * c.f_r);
            gg.SetGlobalBuffer(ptr + static_cast<int64_t>(10) * rows * c.f_r +
                               static_cast<int64_t>(k) * c.f_mTask * c.f_channels);
            auto td = tla::MakeTensor(gd, tla::MakeLayout<float, LayoutTagC>(rows, c.f_r), Arch::PositionGM{});
            auto tg = tla::MakeTensor(gg, tla::MakeLayout<float, LayoutTagC>(rows, c.f_channels), Arch::PositionGM{});
            auto out = GetTile(td, tla::MakeCoord(0, no), tla::MakeShape(rows, 160u));
            block(a, b, out, GemmCoord{rows, 160u, 1024u});
            if (c.f_useCombine && n == 0) {
                block.SelectGate(true);
                // Gate is computed and stored independently, never as concatenated down columns.
                block(a, bg, tg, GemmCoord{rows, c.f_channels, 1024u});
            }
        }
        if (c.f_blockIdx >= tasks)
            HcAicWait(TwoWaves && ((row / c.f_mTask) & 1u) != 0u ? FLAG_V0_SECOND : FLAG_V0);
    }

    static constexpr int64_t FLAG_V0_SECOND = 7;
    static constexpr int64_t FLAG_V1_SECOND = 6;

    template <class C>
    CATLASS_DEVICE void RunC1FullResident(C& c, uint32_t rows, uint32_t waves, uint32_t firstWave = 0u)
    {
        using C1 =
            Gemm::Block::BlockMmadC1A2FullBTla<typename Base::DispatchPolicyC1A1, typename Base::L1C1,
                                               typename Base::L0C1, Element, Element, Element, void,
                                               typename Base::TileCopyC1, typename Base::BlockMmadC1A1::TileMmad>;
        for (uint32_t w = 0; w < waves; ++w) {
            const uint32_t wave = firstWave + w;
            for (uint32_t n = c.f_blockIdx; n < c.f_nSliceC1; n += 2u * c.f_usedAic) {
                C1 block(c.f_resource);
                const uint32_t next = n + c.f_usedAic;
                auto b0 = GetTile(c.f_tensorWUp, tla::MakeCoord(0, n * L1N1), tla::MakeShape(c.f_r, L1N1));
                block.PreloadSlot(b0, 0u);
                if (next < c.f_nSliceC1) {
                    auto b1 = GetTile(c.f_tensorWUp, tla::MakeCoord(0, next * L1N1), tla::MakeShape(c.f_r, L1N1));
                    block.PreloadSlot(b1, 1u);
                }
                if (n == c.f_blockIdx) {
                    HcAicWait(TwoWaves && (wave & 1u) != 0u ? FLAG_V1_SECOND : FLAG_V1);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(7);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(7);
                }
                block.BeginWave();
                for (uint32_t slot = 0; slot < 2u && n + slot * c.f_usedAic < c.f_nSliceC1; ++slot) {
                    block.SelectB(slot);
                    const uint32_t off = (n + slot * c.f_usedAic) * L1N1;
                    RunSmallNCubesC1(block, c.f_tensorDown, c.f_tensorWUp, c.f_tensorUp, wave * c.f_mTask, rows, off,
                                     L1N1, c.f_r);
                }
            }
            if (c.f_blockIdx >= c.f_nSliceC1)
                HcAicWait(TwoWaves && (wave & 1u) != 0u ? FLAG_V1_SECOND : FLAG_V1);
            AscendC::PipeBarrier<PIPE_ALL>();
            HcAicSet((wave & 1u) == 0u ? FLAG_C1_0 : FLAG_C1_1);
        }
    }

    template <class Context>
    CATLASS_DEVICE void RunBody(Context& c)
    {
        using namespace AscendC;
        auto& params = c.f_params;
        auto& usedAic = c.f_usedAic;
        auto& mTask = c.f_mTask;
        auto& d = c.f_d;
        auto& r = c.f_r;
        auto& hidden = c.f_hidden;
        auto& channels = c.f_channels;
        auto& M = c.f_M;
        auto& eps = c.f_eps;
        auto& useCombine = c.f_useCombine;
        auto& szT = c.f_szT;
        auto& nConcat = c.f_nConcat;
        auto& nWs = c.f_nWs;
        auto& blockIdx = c.f_blockIdx;
        auto& resource = c.f_resource;
        auto& c0SlotBytes = c.f_c0SlotBytes;
        auto& ptrC0Core = c.f_ptrC0Core;
        auto& tensorXn = c.f_tensorXn;
        auto& tensorDown = c.f_tensorDown;
        auto& tensorWDown = c.f_tensorWDown;
        auto& tensorWGate = c.f_tensorWGate;
        auto& tensorWUp = c.f_tensorWUp;
        auto& tensorUp = c.f_tensorUp;
        auto& localTasks = c.f_localTasks;
        auto& blockV0 = c.f_blockV0;
        auto& blockV1 = c.f_blockV1;
        auto& blockV2 = c.f_blockV2;
        auto& nSliceC1 = c.f_nSliceC1;

        // c0dp：C0 把 down 直写用户 down_pre_act（未除 C，反向自行重算），gate 仍写 c0Slot。
        const bool c0dpSmall = params.ptrDownPreAct != nullptr;
        const int64_t c0SlotEls = c0SlotBytes / static_cast<int64_t>(sizeof(Element));
        uint32_t gateNSmall = channels == 0 ? 1u : channels;
        auto layoutC0Slot = tla::MakeLayout<Element, LayoutTagC>(mTask, r);
        auto layoutGateSlot = tla::MakeLayout<Element, LayoutTagC>(mTask, gateNSmall);
        GlobalTensor<Element> gmC0S0;
        gmC0S0.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(ptrC0Core));
        auto tensorC0S0 = tla::MakeTensor(gmC0S0, layoutC0Slot, Arch::PositionGM{});
        GlobalTensor<Element> gmC0S1;
        gmC0S1.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(ptrC0Core + c0SlotBytes));
        auto tensorC0S1 = tla::MakeTensor(gmC0S1, layoutC0Slot, Arch::PositionGM{});
        GlobalTensor<Element> gmGateS0;
        gmGateS0.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(
            ptrC0Core + static_cast<int64_t>(mTask) * r * static_cast<int64_t>(sizeof(Element))));
        auto tensorGateS0 = tla::MakeTensor(gmGateS0, layoutGateSlot, Arch::PositionGM{});
        GlobalTensor<Element> gmGateS1;
        gmGateS1.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(
            ptrC0Core + c0SlotBytes + static_cast<int64_t>(mTask) * r * static_cast<int64_t>(sizeof(Element))));
        auto tensorGateS1 = tla::MakeTensor(gmGateS1, layoutGateSlot, Arch::PositionGM{});
        GlobalTensor<Element> gmC0DpaSmall;
        if (c0dpSmall) {
            gmC0DpaSmall.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrDownPreAct));
        }
        auto tensorC0DpaSmall =
            tla::MakeTensor(gmC0DpaSmall, tla::MakeLayout<Element, LayoutTagC>(M, r), Arch::PositionGM{});
        for (uint32_t wave = 0; wave <= localTasks + 1; ++wave) {
            if (wave < localTasks) {
                uint32_t row = wave * mTask;
                uint32_t actualM = M - row < mTask ? M - row : mTask;
                uint32_t slot = wave & 1;
                if ASCEND_IS_AIV {
                    uint32_t aivIdx = 0;
                    uint32_t totalAiv = 1;
                    uint32_t v0Bytes = params.ptrInvRms != nullptr ? channels * sizeof(float) : d * szT;
                    HcAivRowRange(actualM, v0Bytes, usedAic, aivIdx, totalAiv);
                    if (aivIdx < totalAiv) {
                        uint32_t m = aivIdx;
                        int64_t off = static_cast<int64_t>(row + m) * d;
                        blockV0(resource, off, off, (totalAiv - aivIdx), d, channels, hidden, eps, useCombine, true,
                                static_cast<int64_t>(row + m) * channels);
                    }
                    HcAivSyncAll();
                    if constexpr (TwoWaves)
                        HcAivSet((wave & 1u) == 0u ? FLAG_V0 : FLAG_V0_SECOND);
                    else
                        HcAivSet(FLAG_V0);
                }
                if ASCEND_IS_AIC {
                    if constexpr (PreloadSmall) {
                        RunPreloadedC0(c, row, actualM, slot);
                    } else {
                        HcAicWait(FLAG_V0);
                        {
                            auto const& tensorC0 = c0dpSmall ? tensorC0DpaSmall : (slot == 0 ? tensorC0S0 : tensorC0S1);
                            auto const& tensorGate = slot == 0 ? tensorGateS0 : tensorGateS1;
                            auto tensorARow = GetTile(tensorXn, tla::MakeCoord(row, 0), tla::MakeShape(actualM, d));

                            if (C0PreferN128(actualM, nConcat, usedAic)) {
                                RunSmallNCubesC0Direct<64u, 128u>(resource, tensorARow, tensorWDown, tensorWGate,
                                                                  tensorC0, tensorGate, 0, actualM, r, nConcat, d,
                                                                  blockIdx, usedAic, useCombine, c0dpSmall ? row : 0u);
                            } else if (actualM > 128u) {
                                RunSmallNCubesC0Direct<64u, 80u>(resource, tensorARow, tensorWDown, tensorWGate,
                                                                 tensorC0, tensorGate, 0, actualM, r, nConcat, d,
                                                                 blockIdx, usedAic, useCombine, c0dpSmall ? row : 0u);
                            } else if (C0PreferN32(actualM, nConcat, usedAic)) {
                                RunSmallNCubesC0Direct<32u, 32u>(resource, tensorARow, tensorWDown, tensorWGate,
                                                                 tensorC0, tensorGate, 0, actualM, r, nConcat, d,
                                                                 blockIdx, usedAic, useCombine, c0dpSmall ? row : 0u);
                            } else {
                                RunSmallNCubesC0Direct(resource, tensorARow, tensorWDown, tensorWGate, tensorC0,
                                                       tensorGate, 0, actualM, r, nConcat, d, blockIdx, usedAic,
                                                       useCombine, c0dpSmall ? row : 0u);
                            }
                        }
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                    HcAicSet(slot == 0 ? FLAG_C0_0 : FLAG_C0_1);
                }
            }
            if (wave >= 1 && (wave - 1) < localTasks) {
                uint32_t prev = wave - 1;
                uint32_t prevSlot = prev & 1;
                uint32_t row = prev * mTask;
                uint32_t actualM = M - row < mTask ? M - row : mTask;
                if ASCEND_IS_AIV {
                    HcAivWait(prevSlot == 0 ? FLAG_C0_0 : FLAG_C0_1);
                    HcAivSyncAll();
                    if constexpr (PreloadSmall) {
                        const int64_t region = static_cast<int64_t>(mTask) * nWs * 10 * sizeof(float);
                        Epilogue::SmallModelA2::BlockHcReduceV1<Element>{}(
                            resource, c.f_ptrPartial + prevSlot * region,
                            c.f_ptrDown + static_cast<int64_t>(row) * r * sizeof(Element),
                            params.ptrInject + static_cast<int64_t>(row) * channels * sizeof(Element), actualM, r,
                            channels, useCombine, mTask * channels,
                            params.ptrDownPreAct == nullptr ?
                                nullptr :
                                params.ptrDownPreAct + static_cast<int64_t>(row) * r * sizeof(Element));
                    } else {
                        uint32_t aivIdx = 0;
                        uint32_t step = 1;
                        HcAivRowRange(actualM, ((r * szT) & 31u) != 0u ? szT : channels * szT, usedAic, aivIdx, step);
                        if (aivIdx < step) {
                            uint32_t m = aivIdx;
                            // c0dp：V1 从 down_pre_act 读 C0 原始输出（offC0 相对 dpa 基址），
                            // 不再单独写保存；非 c0dp 从 c0 槽读并保存 down_pre_act。
                            const int64_t c0SlotEls = c0SlotBytes / static_cast<int64_t>(sizeof(Element));
                            blockV1.template operator()<Arch::Resource<ArchTag>>(
                                resource,
                                c0dpSmall ? static_cast<int64_t>(row + m) * r :
                                            prevSlot * c0SlotEls + static_cast<int64_t>(m) * r,
                                c0dpSmall, static_cast<int64_t>(row + m) * r, static_cast<int64_t>(row + m) * channels,
                                (step - aivIdx), nWs, nConcat, r, channels, useCombine, true,
                                !c0dpSmall && params.ptrDownPreAct != nullptr, static_cast<int64_t>(row + m) * r,
                                prevSlot * c0SlotEls + static_cast<int64_t>(mTask) * r +
                                    static_cast<int64_t>(m) * channels);
                        }
                    }
                    HcAivSyncAll();
                    if constexpr (TwoWaves)
                        HcAivSet(prevSlot == 0u ? FLAG_V1 : FLAG_V1_SECOND);
                    else
                        HcAivSet(FLAG_V1);
                }
                if ASCEND_IS_AIC {
                    RunC1FullResident(c, actualM, 1u, prev); // The helper publishes the single completion flag.
                }
            }
            if (wave >= 2 && (wave - 2) < localTasks) {
                uint32_t prev2 = wave - 2;
                uint32_t c1Slot = prev2 & 1;
                uint32_t row = prev2 * mTask;
                uint32_t actualM = M - row < mTask ? M - row : mTask;
                if ASCEND_IS_AIV {
                    HcAivWait(c1Slot == 0 ? FLAG_C1_0 : FLAG_C1_1);
                    HcAivSyncAll();
                    uint32_t aivIdx = 0;
                    uint32_t totalAiv = 1;
                    HcAivRowRange(actualM, hidden * szT, usedAic, aivIdx, totalAiv);
                    if (aivIdx < totalAiv) {
                        uint32_t m = aivIdx;
                        int64_t off = static_cast<int64_t>(row + m) * d;
                        blockV2(resource, off, off, static_cast<int64_t>(row + m) * hidden, (totalAiv - aivIdx),
                                channels, hidden, true);
                    }
                }
            }
        }
        return;
    }
    CATLASS_DEVICE void operator()(Params const& params)
    {
        Base::template Execute<HcForwardSmallFullBKernel>(params);
    }
};
} // namespace Catlass::Gemm::Kernel::SmallModelA2
