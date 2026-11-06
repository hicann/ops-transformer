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
#include "hc_forward_mul_sum_kernel.hpp"
namespace Catlass::Gemm::Kernel {
template <class Element, uint32_t C1SwizzleDir, class ArchTag_ = Arch::AtlasA2,
          class Tiling = HcForwardMulSumTilingData>
struct HcForwardLargeKernel : HcForwardCommon<Element, C1SwizzleDir, false, ArchTag_, Tiling> {
    using Base = HcForwardCommon<Element, C1SwizzleDir, false, ArchTag_, Tiling>;
    using Params = typename Base::Params;
    using ArchTag = typename Base::ArchTag;
    using LayoutTagA = typename Base::LayoutTagA;
    using LayoutTagC = typename Base::LayoutTagC;
    using BlockMmadC0 = typename Base::BlockMmadC0;
    using BlockMmadC1 = typename Base::BlockMmadC1;
    using BlockSchedulerC1 = typename Base::BlockSchedulerC1;
    using Base::L1M0;
    using Base::L1N0;
    using Base::L1M1;
    using Base::L1N1;
    template <class TensorA, class TensorBDown, class TensorBGate, class TensorC, class TensorGate>
    CATLASS_DEVICE void RunLargeC0NConcat(Arch::Resource<ArchTag>& resource, TensorA const& tensorA,
                                          TensorBDown const& tensorBDown, TensorBGate const& tensorBGate,
                                          TensorC const& tensorC0, TensorGate const& tensorGate, uint32_t row,
                                          uint32_t actualM, uint32_t rDim, uint32_t channels, uint32_t kDim,
                                          bool useCombine, uint32_t rowC0 = 0)
    {
        // BlockMmad 把整块 actualN 放进一块 L0C。L0C 只按 L1N0 分配，N 超过 L1N0 时 FIXP 读越界。
        // rowC0：down 输出张量的行偏移（c0dp 时 down 直写 down_pre_act 全区域张量，按任务行寻址）。
        // gate 恒写槽内（mTask, channels）张量，坐标保持槽内 mOff，不吃 rowC0。
        BlockMmadC0 mmad(resource);
        for (uint32_t mOff = 0; mOff < actualM; mOff += L1M0) {
            uint32_t tileM = actualM - mOff < L1M0 ? actualM - mOff : L1M0;
            auto tileA = GetTile(tensorA, tla::MakeCoord(row + mOff, 0), tla::MakeShape(tileM, kDim));
            for (uint32_t nOff = 0; nOff < rDim; nOff += L1N0) {
                uint32_t tileN = rDim - nOff < L1N0 ? rDim - nOff : L1N0;
                auto tileB = GetTile(tensorBDown, tla::MakeCoord(0, nOff), tla::MakeShape(kDim, tileN));
                auto tileC = GetTile(tensorC0, tla::MakeCoord(rowC0 + mOff, nOff), tla::MakeShape(tileM, tileN));
                mmad(tileA, tileB, tileC, GemmCoord{tileM, tileN, kDim});
            }
            if (useCombine && channels > 0) {
                auto tileBg = GetTile(tensorBGate, tla::MakeCoord(0, 0), tla::MakeShape(kDim, channels));
                auto tileCi = GetTile(tensorGate, tla::MakeCoord(mOff, 0), tla::MakeShape(tileM, channels));
                mmad(tileA, tileBg, tileCi, GemmCoord{tileM, channels, kDim});
            }
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
        auto& nConcat = c.f_nConcat;
        auto& nWs = c.f_nWs;
        auto& blockIdx = c.f_blockIdx;
        auto& resource = c.f_resource;
        auto& xnSlotBytes = c.f_xnSlotBytes;
        auto& downSlotBytes = c.f_downSlotBytes;
        auto& c0SlotBytes = c.f_c0SlotBytes;
        auto& ptrXn = c.f_ptrXn;
        auto& ptrDown = c.f_ptrDown;
        auto& ptrC0Core = c.f_ptrC0Core;
        auto& tensorWDown = c.f_tensorWDown;
        auto& tensorWGate = c.f_tensorWGate;
        auto& tensorWUp = c.f_tensorWUp;
        auto& tensorUp = c.f_tensorUp;
        auto& localTasks = c.f_localTasks;
        auto& blockV0 = c.f_blockV0;
        auto& blockV1 = c.f_blockV1;
        auto& blockV2 = c.f_blockV2;

        // fused 大路径按核环形槽：xn 每核 3 槽、down 每核 2 槽、
        // c0 每核 2 槽。c0 槽内 down 段在前、gate 段紧随，两段交错不能并成单张量，按槽各建。
        // wave 循环内 C0/C1 只用坐标偏移寻址，不再逐 wave 重复 SetGlobalBuffer+MakeTensor。
        const bool c0DpLarge = params.ptrDownPreAct != nullptr;
        GlobalTensor<Element> gmXnRing;
        gmXnRing.SetGlobalBuffer(
            reinterpret_cast<__gm__ Element*>(ptrXn + static_cast<int64_t>(blockIdx) * 3u * xnSlotBytes));
        // c0 对环形槽里的 xn 只读一次，与 v2 一样不占 L2。这段只在大 shape 路径执行。
        gmXnRing.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        auto tensorXnRing =
            tla::MakeTensor(gmXnRing, tla::MakeLayout<Element, LayoutTagA>(3u * mTask, d), Arch::PositionGM{});
        GlobalTensor<Element> gmDownRing;
        gmDownRing.SetGlobalBuffer(
            reinterpret_cast<__gm__ Element*>(ptrDown + static_cast<int64_t>(blockIdx) * 2u * downSlotBytes));
        auto tensorDownRing =
            tla::MakeTensor(gmDownRing, tla::MakeLayout<Element, LayoutTagA>(2u * mTask, r), Arch::PositionGM{});
        uint32_t gateN = channels == 0 ? 1u : channels;
        auto layoutC0Slot = tla::MakeLayout<Element, LayoutTagC>(mTask, r);
        auto layoutGateSlot = tla::MakeLayout<Element, LayoutTagC>(mTask, gateN);
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
        // c0dp：down 直写 down_pre_act 全区域张量（(M, r)），按任务行 rowC0 寻址。
        GlobalTensor<Element> gmC0Dpa;
        if (c0DpLarge) {
            gmC0Dpa.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(params.ptrDownPreAct));
        }
        auto tensorC0Dpa = tla::MakeTensor(gmC0Dpa, tla::MakeLayout<Element, LayoutTagC>(M, r), Arch::PositionGM{});

        // fused 默认 V2 delay-2：V0[t] → C0[t] → V1[t-1] → C1[t-1] → V2[t-2]
        // 尾部多排空 2 拍（最后 V1/C1 与最后 V2 各一拍）。
        for (uint32_t wave = 0; wave <= localTasks + 1; ++wave) {
            uint32_t task = blockIdx + wave * usedAic;
            if (wave < localTasks) {
                uint32_t row = task * mTask;
                uint32_t actualM = M - row < mTask ? M - row : mTask;
                if ASCEND_IS_AIV {
                    int64_t offX = static_cast<int64_t>(row) * d;
                    int64_t offXn = (static_cast<int64_t>(blockIdx) * 3u * xnSlotBytes +
                                     static_cast<int64_t>(wave % 3u) * xnSlotBytes) /
                                    static_cast<int64_t>(sizeof(Element));
                    int64_t offInv = static_cast<int64_t>(row) * channels;
                    blockV0(resource, offX, offXn, actualM, d, channels, hidden, eps, useCombine, false, offInv);
                    HcAivSet(FLAG_V0);
                }
                if ASCEND_IS_AIC {
                    HcAicWait(FLAG_V0);
                    uint32_t slot = wave & 1;
                    // c0dp：down 直写 down_pre_act 全区域张量（rowC0=任务行，保存原始值未除 C），
                    // gate 仍写本核 c0 槽；两者均复用循环外一次构造的槽张量。
                    auto const& tensorC0 = c0DpLarge ? tensorC0Dpa : (slot == 0 ? tensorC0S0 : tensorC0S1);
                    auto const& tensorGate = slot == 0 ? tensorGateS0 : tensorGateS1;
                    RunLargeC0NConcat(resource, tensorXnRing, tensorWDown, tensorWGate, tensorC0, tensorGate,
                                      (wave % 3u) * mTask, actualM, r, channels, d, useCombine, c0DpLarge ? row : 0u);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    HcAicSet(slot == 0 ? FLAG_C0_0 : FLAG_C0_1);
                }
            }
            if (wave >= 1 && (wave - 1) < localTasks) {
                uint32_t prev = blockIdx + (wave - 1) * usedAic;
                uint32_t prevSlot = (wave - 1) & 1;
                uint32_t row = prev * mTask;
                uint32_t actualM = M - row < mTask ? M - row : mTask;
                if ASCEND_IS_AIV {
                    HcAivWait(prevSlot == 0 ? FLAG_C0_0 : FLAG_C0_1);
                    const int64_t c0SlotEls = c0SlotBytes / static_cast<int64_t>(sizeof(Element));
                    // c0dp：V1 从 down_pre_act 读 C0 原始输出（offC0 相对 dpa 基址），
                    // 不再单独写保存；否则从本核 c0 槽读。
                    const bool v1C0FromDpa = params.ptrDownPreAct != nullptr;
                    const int64_t offC0 = v1C0FromDpa ? static_cast<int64_t>(row) * r : prevSlot * c0SlotEls;
                    const int64_t offDown = (static_cast<int64_t>(blockIdx) * 2u + prevSlot) * mTask * r;
                    blockV1.template operator()<Arch::Resource<ArchTag>>(
                        resource, offC0, v1C0FromDpa, offDown, static_cast<int64_t>(row) * channels, actualM, nWs,
                        nConcat, r, channels, useCombine, false, false, 0,
                        prevSlot * c0SlotEls + static_cast<int64_t>(mTask) * r);
                    HcAivSet(FLAG_V1);
                }
                if ASCEND_IS_AIC {
                    HcAicWait(FLAG_V1);
                    GemmCoord problem{actualM, d, r};
                    BlockSchedulerC1 scheduler(problem, MatrixCoord{L1M1, L1N1});
                    uint32_t nTiles = scheduler.GetCoreLoops();
                    BlockMmadC1 mmadC1(resource);
                    for (uint32_t nTile = 0; nTile < nTiles; ++nTile) {
                        GemmCoord coord = scheduler.GetBlockCoord(nTile);
                        GemmCoord actual = scheduler.GetActualBlockShape(coord);
                        uint32_t nOff = coord.n() * L1N1;
                        auto tileA =
                            GetTile(tensorDownRing, tla::MakeCoord(prevSlot * mTask, 0), tla::MakeShape(actual.m(), r));
                        auto tileB = GetTile(tensorWUp, tla::MakeCoord(0, nOff), tla::MakeShape(r, actual.n()));
                        auto tileC =
                            GetTile(tensorUp, tla::MakeCoord(row, nOff), tla::MakeShape(actual.m(), actual.n()));
                        mmadC1(tileA, tileB, tileC, actual);
                    }

                    AscendC::PipeBarrier<PIPE_ALL>();
                    HcAicSet(prevSlot == 0 ? FLAG_C1_0 : FLAG_C1_1);
                }
            }
            if (wave >= 2 && (wave - 2) < localTasks) {
                uint32_t prev2 = blockIdx + (wave - 2) * usedAic;
                uint32_t c1Slot = (wave - 2) & 1;
                uint32_t row = prev2 * mTask;
                uint32_t actualM = M - row < mTask ? M - row : mTask;
                if ASCEND_IS_AIV {
                    HcAivWait(c1Slot == 0 ? FLAG_C1_0 : FLAG_C1_1);
                    int64_t offX = static_cast<int64_t>(row) * d;
                    int64_t offXn = (static_cast<int64_t>(blockIdx) * 3u * xnSlotBytes +
                                     static_cast<int64_t>((wave - 2) % 3u) * xnSlotBytes) /
                                    static_cast<int64_t>(sizeof(Element));
                    blockV2(resource, offX, offXn, static_cast<int64_t>(row) * hidden, actualM, channels, hidden,
                            false);
                }
            }
        }
    }
    CATLASS_DEVICE void operator()(Params const& params)
    {
        Base::template Execute<HcForwardLargeKernel>(params);
    }
};
} // namespace Catlass::Gemm::Kernel
