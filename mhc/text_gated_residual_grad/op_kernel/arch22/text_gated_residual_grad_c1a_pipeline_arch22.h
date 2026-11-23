/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEXT_GATED_RESIDUAL_GRAD_C1A_PIPELINE_ARCH22_H
#define TEXT_GATED_RESIDUAL_GRAD_C1A_PIPELINE_ARCH22_H

namespace TextGatedResidualGrad {

// Keep the NN mapping of output tiles to cores. Within each tile, accumulate four
// row strips over the same K panels, sharing B without rereading it from GM.
template <typename T>
class HcGradC1aPipeline {
    using Arch = Catlass::Arch::AtlasA2;
    using RowMajor = Catlass::layout::RowMajor;
    using AType = Catlass::Gemm::GemmType<T, RowMajor>;
    using BType = Catlass::Gemm::GemmType<T, RowMajor>;
    using CType = Catlass::Gemm::GemmType<float, RowMajor>;
    using Copy = Catlass::Gemm::Tile::TileCopy<Arch, AType, BType, CType>;
    using L1A = typename Copy::CopyL1ToL0A::LayoutSrc;
    using L1B = typename Copy::CopyL1ToL0B::LayoutSrc;
    using L0A = typename Copy::CopyL1ToL0A::LayoutDst;
    using L0B = typename Copy::CopyL1ToL0B::LayoutDst;
    using L0C = Catlass::layout::zN;
    using HardEvent = AscendC::HardEvent;
    static constexpr uint32_t SLOTS = 2;
    static constexpr uint32_t STRIPS = 4;
    static constexpr uint32_t ROWS = 32;
    static constexpr uint32_t COLS = 256;
    static constexpr uint32_t BASE_K = 64;
    static constexpr uint32_t PANEL_K = 256;
    static constexpr uint32_t TAIL_K = SLOTS * PANEL_K;
    static constexpr uint32_t L1A_BYTES = STRIPS * ROWS * PANEL_K * sizeof(T);
    static constexpr uint32_t L1B_BYTES = COLS * PANEL_K * sizeof(T);
    static constexpr uint32_t L0A_BYTES = ROWS * BASE_K * sizeof(T);
    static constexpr uint32_t L0B_BYTES = COLS * BASE_K * sizeof(T);
    static constexpr uint32_t L0C_BYTES = ROWS * COLS * sizeof(float);
    static_assert(SLOTS * (L1A_BYTES + L1B_BYTES) <= Arch::L1_SIZE);
    static_assert(ROWS * TAIL_K * sizeof(T) <= L1A_BYTES);
    static_assert(SLOTS * L0A_BYTES <= Arch::L0A_SIZE && SLOTS * L0B_BYTES <= Arch::L0B_SIZE);
    static_assert(STRIPS * L0C_BYTES <= Arch::L0C_SIZE);

public:
    __aicore__ explicit HcGradC1aPipeline(Catlass::Arch::Resource<Arch>& resource)
    {
        using namespace AscendC;
        for (uint32_t s = 0; s < SLOTS; ++s) {
            l1A_[s] = resource.l1Buf.template GetBufferByByte<T>(s * L1A_BYTES);
            l1B_[s] = resource.l1Buf.template GetBufferByByte<T>(SLOTS * L1A_BYTES + s * L1B_BYTES);
            l0A_[s] = resource.l0ABuf.template GetBufferByByte<T>(s * L0A_BYTES);
            l0B_[s] = resource.l0BBuf.template GetBufferByByte<T>(s * L0B_BYTES);
            SetFlag<HardEvent::MTE1_MTE2>(s);
            SetFlag<HardEvent::M_MTE1>(s);
            SetFlag<HardEvent::M_MTE1>(s + SLOTS);
        }
        for (uint32_t s = 0; s < STRIPS; ++s) {
            l0C_[s] = resource.l0CBuf.template GetBufferByByte<float>(s * L0C_BYTES);
            SetFlag<HardEvent::FIX_M>(s);
        }
    }

    __aicore__ ~HcGradC1aPipeline()
    {
        using namespace AscendC;
        for (uint32_t s = 0; s < SLOTS; ++s) {
            WaitFlag<HardEvent::MTE1_MTE2>(s);
            WaitFlag<HardEvent::M_MTE1>(s);
            WaitFlag<HardEvent::M_MTE1>(s + SLOTS);
        }
        for (uint32_t s = 0; s < STRIPS; ++s) {
            WaitFlag<HardEvent::FIX_M>(s);
        }
    }

    __aicore__ void operator()(const AscendC::GlobalTensor<T>& a, const RowMajor& layoutA,
                               const AscendC::GlobalTensor<T>& b, const RowMajor& layoutB,
                               const AscendC::GlobalTensor<float>& c, const RowMajor& layoutC,
                               const Catlass::GemmCoord& shape)
    {
        using namespace AscendC;
        const uint32_t strips = (shape.m() + ROWS - 1U) / ROWS;
        const uint32_t bodyK = shape.k() - TAIL_K;
        const auto aPanel = L1A::template MakeLayout<T>(shape.m(), PANEL_K);
        const auto bPanel = L1B::template MakeLayout<T>(PANEL_K, shape.n());
        uint32_t panel = 0;
        for (uint32_t k = 0; k < bodyK; k += PANEL_K, ++panel) {
            const uint32_t s = panel % SLOTS;
            WaitFlag<HardEvent::MTE1_MTE2>(s);
            CopyAToL1(l1A_[s], a[layoutA.GetOffset({0, k})], aPanel, layoutA.GetTileLayout({shape.m(), PANEL_K}));
            CopyBToL1(l1B_[s], b[layoutB.GetOffset({k, 0})], bPanel, layoutB.GetTileLayout({PANEL_K, shape.n()}));
            SetFlag<HardEvent::MTE2_MTE1>(s);
            if (panel != 0) {
                ComputeBodyPanel(s ^ 1U, aPanel, bPanel, shape, strips, panel == 1);
            }
        }
        ComputeBodyPanel((panel - 1U) % SLOTS, aPanel, bPanel, shape, strips, false);

        // Retain the last two B panels in L1, and stream those K values of A by
        // row strip. Keep the next strip's DMA queued alongside the current
        // strip's final MMADs and the previous strip's Fixpipe writeback.
        for (uint32_t s = 0; s < SLOTS; ++s) {
            WaitFlag<HardEvent::MTE1_MTE2>(s);
            CopyBToL1(l1B_[s], b[layoutB.GetOffset({bodyK + s * PANEL_K, 0})], bPanel,
                      layoutB.GetTileLayout({PANEL_K, shape.n()}));
            SetFlag<HardEvent::MTE2_MTE1>(s + SLOTS);
        }
        for (uint32_t s = 0; s < SLOTS && s < strips; ++s) {
            LoadTailA(s, a, layoutA, shape.m(), bodyK);
        }
        for (uint32_t strip = 0; strip < strips; ++strip) {
            const uint32_t aSlot = strip % SLOTS;
            const uint32_t m = StripRows(shape.m(), strip);
            const auto aTail = L1A::template MakeLayout<T>(m, TAIL_K);
            WaitFlag<HardEvent::MTE2_MTE1>(aSlot);
            for (uint32_t k = 0; k < TAIL_K; k += BASE_K) {
                const uint32_t bSlot = k / PANEL_K;
                if (strip == 0 && k % PANEL_K == 0) {
                    WaitFlag<HardEvent::MTE2_MTE1>(bSlot + SLOTS);
                }
                LoadB(l1B_[bSlot][bPanel.GetOffset({k % PANEL_K, 0})], bPanel, shape.n());
                Multiply(strip, l1A_[aSlot][aTail.GetOffset({0, k})], aTail, m, shape.n(), false);
                ReleaseB();
            }
            const bool loadNext = strip + SLOTS < strips;
            if (loadNext) {
                SetFlag<HardEvent::MTE1_MTE2>(aSlot);
            }
            SetFlag<HardEvent::M_FIX>(strip);
            WaitFlag<HardEvent::M_FIX>(strip);
            StoreC(c[layoutC.GetOffset({strip * ROWS, 0})], l0C_[strip], layoutC.GetTileLayout({m, shape.n()}),
                   L0C::MakeLayoutInL0C({m, shape.n()}));
            SetFlag<HardEvent::FIX_M>(strip);
            if (loadNext) {
                WaitFlag<HardEvent::MTE1_MTE2>(aSlot);
                LoadTailA(strip + SLOTS, a, layoutA, shape.m(), bodyK);
            }
        }
        for (uint32_t s = 0; s < SLOTS; ++s) {
            SetFlag<HardEvent::MTE1_MTE2>(s);
        }
    }

private:
    __aicore__ static uint32_t StripRows(uint32_t rows, uint32_t strip)
    {
        const uint32_t remaining = rows - strip * ROWS;
        return remaining < ROWS ? remaining : ROWS;
    }

    __aicore__ void LoadTailA(uint32_t strip, const AscendC::GlobalTensor<T>& a, const RowMajor& layout, uint32_t rows,
                              uint32_t k)
    {
        const uint32_t m = StripRows(rows, strip);
        const uint32_t s = strip % SLOTS;
        CopyAToL1(l1A_[s], a[layout.GetOffset({strip * ROWS, k})], L1A::template MakeLayout<T>(m, TAIL_K),
                  layout.GetTileLayout({m, TAIL_K}));
        AscendC::SetFlag<HardEvent::MTE2_MTE1>(s);
    }

    __aicore__ void ComputeBodyPanel(uint32_t s, const L1A& aPanel, const L1B& bPanel, const Catlass::GemmCoord& shape,
                                     uint32_t strips, bool first)
    {
        using namespace AscendC;
        WaitFlag<HardEvent::MTE2_MTE1>(s);
        for (uint32_t k = 0; k < PANEL_K; k += BASE_K) {
            LoadB(l1B_[s][bPanel.GetOffset({k, 0})], bPanel, shape.n());
            for (uint32_t strip = 0; strip < strips; ++strip) {
                const uint32_t row = strip * ROWS;
                Multiply(strip, l1A_[s][aPanel.GetOffset({row, k})], aPanel, StripRows(shape.m(), strip), shape.n(),
                         first && k == 0);
            }
            ReleaseB();
        }
        SetFlag<HardEvent::MTE1_MTE2>(s);
    }

    __aicore__ void LoadB(const AscendC::LocalTensor<T>& src, const L1B& layout, uint32_t n)
    {
        AscendC::WaitFlag<HardEvent::M_MTE1>(bSlot_ + SLOTS);
        CopyBToL0(l0B_[bSlot_], src, L0B::template MakeLayout<T>(BASE_K, n), layout);
    }

    __aicore__ void ReleaseB()
    {
        AscendC::SetFlag<HardEvent::M_MTE1>(bSlot_ + SLOTS);
        bSlot_ ^= 1U;
    }

    __aicore__ void Multiply(uint32_t strip, const AscendC::LocalTensor<T>& src, const L1A& layout, uint32_t m,
                             uint32_t n, bool reset)
    {
        using namespace AscendC;
        WaitFlag<HardEvent::M_MTE1>(aSlot_);
        CopyAToL0(l0A_[aSlot_], src, L0A::template MakeLayout<T>(m, BASE_K), layout);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        if (reset) {
            WaitFlag<HardEvent::FIX_M>(strip);
        }
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        Matmul(l0C_[strip], l0A_[aSlot_], l0B_[bSlot_], m, n, BASE_K, reset);
        SetFlag<HardEvent::M_MTE1>(aSlot_);
        aSlot_ ^= 1U;
    }

    AscendC::LocalTensor<T> l1A_[SLOTS];
    AscendC::LocalTensor<T> l1B_[SLOTS];
    AscendC::LocalTensor<T> l0A_[SLOTS];
    AscendC::LocalTensor<T> l0B_[SLOTS];
    AscendC::LocalTensor<float> l0C_[STRIPS];
    uint32_t aSlot_{0};
    uint32_t bSlot_{0};
    typename Copy::CopyGmToL1A CopyAToL1;
    typename Copy::CopyGmToL1B CopyBToL1;
    typename Copy::CopyL1ToL0A CopyAToL0;
    typename Copy::CopyL1ToL0B CopyBToL0;
    typename Copy::CopyL0CToGm StoreC;
    Catlass::Gemm::Tile::TileMmad<Arch, AType, BType, void> Matmul;
};
} // namespace TextGatedResidualGrad
#endif
