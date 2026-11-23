/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_DW_PANEL_ARCH22_H
#define TEXT_GATED_RESIDUAL_GRAD_DW_PANEL_ARCH22_H

#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

namespace TextGatedResidualGrad {

// Keep the NN tile footprint in L1. Its two output subtiles share B along M or A along N.
// Each subtile keeps its own FP32 accumulator and consumes every K slice in order.
template <typename T, uint32_t GroupM, uint32_t GroupN>
class HcGradDwPanel {
    using Arch = Catlass::Arch::AtlasA2;
    using RowMajor = Catlass::layout::RowMajor;
    using ColumnMajor = Catlass::layout::ColumnMajor;
    using AType = Catlass::Gemm::GemmType<T, ColumnMajor>;
    using BType = Catlass::Gemm::GemmType<T, RowMajor>;
    using CType = Catlass::Gemm::GemmType<float, RowMajor>;
    using Copy = Catlass::Gemm::Tile::TileCopy<Arch, AType, BType, CType>;
    using L1A = typename Copy::CopyL1ToL0A::LayoutSrc;
    using L1B = typename Copy::CopyL1ToL0B::LayoutSrc;
    using L0A = typename Copy::CopyL1ToL0A::LayoutDst;
    using L0B = typename Copy::CopyL1ToL0B::LayoutDst;
    using L0C = Catlass::layout::zN;

public:
    static constexpr uint32_t PANEL_K = 256;

private:
    static constexpr uint32_t BASE_K = 64;
    static constexpr uint32_t TILE_M = 128;
    static constexpr uint32_t TILE_N = 128;
    static constexpr uint32_t SLOTS = 2;
    static constexpr bool SPLIT_M = GroupM == 2 * TILE_M;
    static constexpr uint32_t L1A_BYTES = GroupM * PANEL_K * sizeof(T);
    static constexpr uint32_t L1B_BYTES = GroupN * PANEL_K * sizeof(T);
    static constexpr uint32_t L0A_BYTES = GroupM * BASE_K * sizeof(T);
    static constexpr uint32_t L0B_BYTES = GroupN * BASE_K * sizeof(T);
    static constexpr uint32_t L0C_BYTES = TILE_M * TILE_N * sizeof(float);
    static_assert((GroupM == 2 * TILE_M && GroupN == TILE_N) || (GroupM == TILE_M && GroupN == 2 * TILE_N));
    static_assert((L1A_BYTES + L1B_BYTES) * SLOTS <= Arch::L1_SIZE);
    static_assert(L0A_BYTES * SLOTS <= Arch::L0A_SIZE && L0B_BYTES * SLOTS <= Arch::L0B_SIZE);
    static_assert(L0C_BYTES * SLOTS <= Arch::L0C_SIZE);

    static constexpr L1A A_LAYOUT = L1A::template MakeLayout<T>(GroupM, PANEL_K);
    static constexpr L1B B_LAYOUT = L1B::template MakeLayout<T>(PANEL_K, GroupN);

public:
    __aicore__ explicit HcGradDwPanel(Catlass::Arch::Resource<Arch>& resource)
    {
        using namespace AscendC;
        for (uint32_t slot = 0; slot < SLOTS; ++slot) {
            l1A_[slot] = resource.l1Buf.template GetBufferByByte<T>(slot * L1A_BYTES);
            l1B_[slot] = resource.l1Buf.template GetBufferByByte<T>(SLOTS * L1A_BYTES + slot * L1B_BYTES);
            l0A_[slot] = resource.l0ABuf.template GetBufferByByte<T>(slot * L0A_BYTES);
            l0B_[slot] = resource.l0BBuf.template GetBufferByByte<T>(slot * L0B_BYTES);
            l0C_[slot] = resource.l0CBuf.template GetBufferByByte<float>(slot * L0C_BYTES);
            SetFlag<HardEvent::MTE1_MTE2>(slot);
            SetFlag<HardEvent::M_MTE1>(slot);
            SetFlag<HardEvent::FIX_M>(slot);
        }
    }

    __aicore__ ~HcGradDwPanel()
    {
        using namespace AscendC;
        if (hasPending_) {
            ComputePanel(nextL1_ ^ 1U);
        }
        for (uint32_t slot = 0; slot < SLOTS; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(slot);
            WaitFlag<HardEvent::M_MTE1>(slot);
            WaitFlag<HardEvent::FIX_M>(slot);
        }
    }

    __aicore__ void operator()(const AscendC::GlobalTensor<T>& a, const ColumnMajor& layoutA,
                               const AscendC::GlobalTensor<T>& b, const RowMajor& layoutB,
                               const AscendC::GlobalTensor<float>& c, const RowMajor& layoutC,
                               const Catlass::GemmCoord& shape)
    {
        using namespace AscendC;
        for (uint32_t k = 0; k < shape.k(); k += PANEL_K) {
            const uint32_t slot = nextL1_;
            WaitFlag<HardEvent::MTE1_MTE2>(slot);
            CopyAToL1(l1A_[slot], a[layoutA.GetOffset({0, k})], A_LAYOUT, layoutA.GetTileLayout({shape.m(), PANEL_K}));
            SetFlag<HardEvent::MTE2_MTE1>(slot);
            CopyBToL1(l1B_[slot], b[layoutB.GetOffset({k, 0})], B_LAYOUT, layoutB.GetTileLayout({PANEL_K, shape.n()}));
            SetFlag<HardEvent::MTE2_MTE1>(slot + SLOTS);

            // Load the next tile's first panel before consuming the current tile's last panel.
            if (hasPending_) {
                ComputePanel(slot ^ 1U);
            }
            hasPending_ = true;
            nextL1_ = slot ^ 1U;
            rows_ = shape.m();
            cols_ = shape.n();
            first_ = k == 0;
            last_ = k + PANEL_K == shape.k();
            if (last_) {
                output_ = c;
                outputLayout_ = layoutC;
            }
        }
    }

private:
    __aicore__ void ComputePanel(uint32_t slot)
    {
        using namespace AscendC;
        const uint32_t m0 = rows_ < TILE_M ? rows_ : TILE_M;
        const uint32_t n0 = cols_ < TILE_N ? cols_ : TILE_N;
        const uint32_t m1 = SPLIT_M ? rows_ - m0 : m0;
        const uint32_t n1 = SPLIT_M ? n0 : cols_ - n0;
        const bool hasSecond = SPLIT_M ? m1 != 0 : n1 != 0;
        for (uint32_t k = 0; k < PANEL_K; k += BASE_K) {
            const uint32_t l0 = nextL0_;
            auto& a0 = l0A_[l0];
            auto& b0 = l0B_[l0];
            WaitFlag<HardEvent::M_MTE1>(l0);
            if (k == 0) {
                WaitFlag<HardEvent::MTE2_MTE1>(slot);
            }
            CopyAToL0(a0, l1A_[slot][A_LAYOUT.GetOffset({0, k})], L0A::template MakeLayout<T>(m0, BASE_K), A_LAYOUT);
            if (k == 0) {
                WaitFlag<HardEvent::MTE2_MTE1>(slot + SLOTS);
            }
            CopyBToL0(b0, l1B_[slot][B_LAYOUT.GetOffset({k, 0})], L0B::template MakeLayout<T>(BASE_K, n0), B_LAYOUT);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

            const bool reset = first_ && k == 0;
            const bool store = last_ && k + BASE_K == PANEL_K;
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            Accumulate(0, a0, b0, m0, n0, 0, 0, reset, store);
            if (hasSecond) {
                if constexpr (SPLIT_M) {
                    auto a1 = a0[TILE_M * BASE_K];
                    CopyAToL0(a1, l1A_[slot][A_LAYOUT.GetOffset({TILE_M, k})], L0A::template MakeLayout<T>(m1, BASE_K),
                              A_LAYOUT);
                    AccumulateSecond(a1, b0, m1, n1, TILE_M, 0, reset, store);
                } else {
                    auto b1 = b0[BASE_K * TILE_N];
                    CopyBToL0(b1, l1B_[slot][B_LAYOUT.GetOffset({k, TILE_N})], L0B::template MakeLayout<T>(BASE_K, n1),
                              B_LAYOUT);
                    AccumulateSecond(a0, b1, m1, n1, 0, TILE_N, reset, store);
                }
            }
            if (k + BASE_K == PANEL_K) {
                SetFlag<HardEvent::MTE1_MTE2>(slot);
            }
            SetFlag<HardEvent::M_MTE1>(l0);
            nextL0_ = l0 ^ 1U;
        }
    }

    __aicore__ void AccumulateSecond(const AscendC::LocalTensor<T>& a, const AscendC::LocalTensor<T>& b, uint32_t m,
                                     uint32_t n, uint32_t row, uint32_t col, bool reset, bool store)
    {
        using namespace AscendC;
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
        Accumulate(1, a, b, m, n, row, col, reset, store);
    }

    __aicore__ void Accumulate(uint32_t slot, const AscendC::LocalTensor<T>& a, const AscendC::LocalTensor<T>& b,
                               uint32_t m, uint32_t n, uint32_t row, uint32_t col, bool reset, bool store)
    {
        using namespace AscendC;
        if (reset) {
            WaitFlag<HardEvent::FIX_M>(slot);
        }
        Matmul(l0C_[slot], a, b, m, n, BASE_K, reset);
        if (store) {
            SetFlag<HardEvent::M_FIX>(slot);
            WaitFlag<HardEvent::M_FIX>(slot);
            StoreC(output_[outputLayout_.GetOffset({row, col})], l0C_[slot], outputLayout_.GetTileLayout({m, n}),
                   L0C::MakeLayoutInL0C({m, n}));
            SetFlag<HardEvent::FIX_M>(slot);
        }
    }

    AscendC::LocalTensor<T> l1A_[SLOTS];
    AscendC::LocalTensor<T> l1B_[SLOTS];
    AscendC::LocalTensor<T> l0A_[SLOTS];
    AscendC::LocalTensor<T> l0B_[SLOTS];
    AscendC::LocalTensor<float> l0C_[SLOTS];
    AscendC::GlobalTensor<float> output_;
    RowMajor outputLayout_;
    uint32_t rows_{0};
    uint32_t cols_{0};
    uint32_t nextL1_{0};
    uint32_t nextL0_{0};
    bool first_{false};
    bool last_{false};
    bool hasPending_{false};
    typename Copy::CopyGmToL1A CopyAToL1;
    typename Copy::CopyGmToL1B CopyBToL1;
    typename Copy::CopyL1ToL0A CopyAToL0;
    typename Copy::CopyL1ToL0B CopyBToL0;
    typename Copy::CopyL0CToGm StoreC;
    Catlass::Gemm::Tile::TileMmad<Arch, AType, BType, void> Matmul;
};
} // namespace TextGatedResidualGrad
#endif
