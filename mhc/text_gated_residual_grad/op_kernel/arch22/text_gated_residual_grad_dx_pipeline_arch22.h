/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEXT_GATED_RESIDUAL_GRAD_DX_PIPELINE_ARCH22_H
#define TEXT_GATED_RESIDUAL_GRAD_DX_PIPELINE_ARCH22_H

namespace TextGatedResidualGrad {

template <typename T>
class HcGradDxPipeline {
    using Arch = Catlass::Arch::AtlasA2;
    using RowMajor = Catlass::layout::RowMajor;
    using L1 = Catlass::layout::zN;
    using L0C = Catlass::layout::zN;
    using Operand = Catlass::Gemm::GemmType<T, RowMajor>;
    using Result = Catlass::Gemm::GemmType<float, RowMajor>;
    using Copy = Catlass::Gemm::Tile::TileCopy<Arch, Operand, Operand, Result>;
    using HardEvent = AscendC::HardEvent;
    static constexpr uint32_t M = 128;
    static constexpr uint32_t N = 128;
    static constexpr uint32_t BASE_K = 64;
    static constexpr uint32_t K_PAD = 336;
    static constexpr uint32_t SLOTS = 2;
    static constexpr uint32_t LOAD_3D_M_EXTENSION_SHIFT = 16U;
    static constexpr uint32_t LOAD_3D_K_START_SHIFT = 32U;
    static constexpr uint32_t LOAD_3D_M_START_SHIFT = 48U;
    static constexpr uint32_t L1A_BYTES = M * K_PAD * sizeof(T);
    static constexpr uint32_t L1B_BYTES = SLOTS * N * K_PAD * sizeof(T);
    static constexpr uint32_t L0_BYTES = M * BASE_K * sizeof(T);
    static constexpr uint32_t L0C_BYTES = M * N * sizeof(float);
    static_assert(SLOTS * (L1A_BYTES + L1B_BYTES) <= Arch::L1_SIZE);
    static_assert(SLOTS * L0_BYTES <= Arch::L0A_SIZE && SLOTS * L0_BYTES <= Arch::L0B_SIZE);
    static_assert(SLOTS * L0C_BYTES <= Arch::L0C_SIZE);

public:
    __aicore__ explicit HcGradDxPipeline(Catlass::Arch::Resource<Arch>& resource)
    {
        using namespace AscendC;
        for (uint32_t slot = 0; slot < SLOTS; ++slot) {
            l1A_[slot] = resource.l1Buf.template GetBufferByByte<T>(slot * L1A_BYTES);
            l1B_[slot] = resource.l1Buf.template GetBufferByByte<T>(SLOTS * L1A_BYTES + slot * L1B_BYTES);
            l0A_[slot] = resource.l0ABuf.template GetBufferByByte<T>(slot * L0_BYTES);
            l0B_[slot] = resource.l0BBuf.template GetBufferByByte<T>(slot * L0_BYTES);
            l0C_[slot] = resource.l0CBuf.template GetBufferByByte<float>(slot * L0C_BYTES);
            SetFlag<HardEvent::MTE1_MTE2>(slot);
            SetFlag<HardEvent::M_MTE1>(slot);
            SetFlag<HardEvent::FIX_M>(slot);
        }
    }

    __aicore__ ~HcGradDxPipeline()
    {
        using namespace AscendC;
        if (pending_) {
            Compute(l1Slot_ ^ 1U);
        }
        for (uint32_t slot = 0; slot < SLOTS; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(slot);
            WaitFlag<HardEvent::M_MTE1>(slot);
            WaitFlag<HardEvent::FIX_M>(slot);
        }
    }

    __aicore__ void operator()(const AscendC::GlobalTensor<T>& a, const RowMajor& layoutA,
                               const AscendC::GlobalTensor<T>& b, const RowMajor& layoutB,
                               const AscendC::GlobalTensor<float>& c, const RowMajor& layoutC,
                               const Catlass::GemmCoord& shape)
    {
        using namespace AscendC;
        const auto slot = l1Slot_;
        WaitFlag<HardEvent::MTE1_MTE2>(slot);
        const auto aAddress = a.GetPhyAddr();
        if (aAddress != cachedAAddress_[slot] || cachedARows_[slot] != shape.m()) {
            copyA_(l1A_[slot], a, L1::template MakeLayout<T>(shape.m(), shape.k()),
                   layoutA.GetTileLayout({shape.m(), shape.k()}));
            cachedAAddress_[slot] = aAddress;
            cachedARows_[slot] = shape.m();
        }
        copyB_(l1B_[slot], b, L1::template MakeLayout<T>(shape.k(), shape.n()),
               layoutB.GetTileLayout({shape.k(), shape.n()}));
        SetFlag<HardEvent::MTE2_MTE1>(slot);
        if (pending_) {
            Compute(slot ^ 1U);
        }
        pendingC_ = c;
        pendingLayout_ = layoutC;
        m_ = shape.m();
        n_ = shape.n();
        k_ = shape.k();
        pending_ = true;
        l1Slot_ ^= 1U;
    }

private:
    __aicore__ void Compute(uint32_t l1Slot)
    {
        using namespace AscendC;
        // Load3D the same ND-to-NZ representation used by the A2 NN scheduler.
        // The B source is shared by both N subtiles, and A is loaded from GM once.
        constexpr uint8_t padding[4] = {0, 0, 0, 0};
        const uint32_t kAligned = (k_ + 15U) & ~15U;
        SetFmatrix(1, m_, padding, FmatrixMode::FMATRIX_LEFT);
        SetFmatrix(1, kAligned, padding, FmatrixMode::FMATRIX_RIGHT);
        WaitFlag<HardEvent::MTE2_MTE1>(l1Slot);
        for (uint32_t col = 0; col < n_; col += N) {
            const uint32_t cols = n_ - col < N ? n_ - col : N;
            const uint32_t sumSlot = l0CSlot_;
            for (uint32_t k = 0; k < k_; k += BASE_K) {
                const uint32_t depth = k_ - k < BASE_K ? k_ - k : BASE_K;
                const uint32_t depthAligned = (depth + 15U) & ~15U;
                const uint32_t slot = l0Slot_;
                WaitFlag<HardEvent::M_MTE1>(slot);

                LoadData3DParamsV2Pro loadA;
                loadA.channelSize = kAligned;
                loadA.extConfig = (static_cast<uint64_t>(k) << LOAD_3D_K_START_SHIFT) |
                                  (static_cast<uint64_t>(m_) << LOAD_3D_M_EXTENSION_SHIFT) | depthAligned;
                LoadData<T>(l0A_[slot], l1A_[l1Slot], loadA);

                LoadData3DParamsV2Pro loadB;
                loadB.channelSize = n_;
                loadB.extConfig = (static_cast<uint64_t>(k) << LOAD_3D_M_START_SHIFT) |
                                  (static_cast<uint64_t>(col) << LOAD_3D_K_START_SHIFT) |
                                  (static_cast<uint64_t>(depthAligned) << LOAD_3D_M_EXTENSION_SHIFT) | cols;
                loadB.fMatrixCtrl = true;
                LoadData<T>(l0B_[slot], l1B_[l1Slot], loadB);
                SetFlag<HardEvent::MTE1_M>(slot);
                if (k == 0) {
                    WaitFlag<HardEvent::FIX_M>(sumSlot);
                }
                WaitFlag<HardEvent::MTE1_M>(slot);
                MmadParams params;
                params.m = m_;
                params.n = cols;
                params.k = depth;
                params.cmatrixInitVal = k == 0;
                Mmad(l0C_[sumSlot], l0A_[slot], l0B_[slot], params);
                SetFlag<HardEvent::M_MTE1>(slot);
                l0Slot_ ^= 1U;
            }
            SetFlag<HardEvent::M_FIX>(sumSlot);
            WaitFlag<HardEvent::M_FIX>(sumSlot);
            store_(pendingC_[pendingLayout_.GetOffset({0, col})], l0C_[sumSlot],
                   pendingLayout_.GetTileLayout({m_, cols}), L0C::MakeLayoutInL0C({m_, cols}));
            SetFlag<HardEvent::FIX_M>(sumSlot);
            l0CSlot_ ^= 1U;
        }
        SetFlag<HardEvent::MTE1_MTE2>(l1Slot);
    }

    AscendC::LocalTensor<T> l1A_[SLOTS];
    AscendC::LocalTensor<T> l1B_[SLOTS];
    AscendC::LocalTensor<T> l0A_[SLOTS];
    AscendC::LocalTensor<T> l0B_[SLOTS];
    AscendC::LocalTensor<float> l0C_[SLOTS];
    AscendC::GlobalTensor<float> pendingC_;
    RowMajor pendingLayout_;
    const __gm__ T* cachedAAddress_[SLOTS]{nullptr, nullptr};
    uint32_t cachedARows_[SLOTS]{0, 0};
    uint32_t m_{0};
    uint32_t n_{0};
    uint32_t k_{0};
    uint32_t l1Slot_{0};
    uint32_t l0Slot_{0};
    uint32_t l0CSlot_{0};
    bool pending_{false};
    typename Copy::CopyGmToL1A copyA_;
    typename Copy::CopyGmToL1B copyB_;
    typename Copy::CopyL0CToGm store_;
};

} // namespace TextGatedResidualGrad
#endif
