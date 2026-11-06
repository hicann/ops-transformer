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
#pragma once

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "../hc_gm_copy.hpp"

namespace Catlass::Epilogue::Tile {

template <class ArchTag_, class ComputeType_>
struct TileGroupedRms {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const& xN, AscendC::LocalTensor<ElementCompute> const& grouped,
                    AscendC::LocalTensor<ElementCompute> const& work, AscendC::LocalTensor<ElementCompute> const& mean,
                    AscendC::LocalTensor<ElementCompute> const& reduceTmp,
                    AscendC::LocalTensor<ElementCompute> const& gamma, float eps, uint32_t channels, uint32_t hidden,
                    bool hasGamma)
    {
        using namespace AscendC;
        const uint32_t d = channels * hidden;
        PipeBarrier<PIPE_V>();
        Mul(work, grouped, grouped, d);
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < channels; ++c) {
            ReduceSum(mean[c], work[c * hidden], reduceTmp, static_cast<int32_t>(hidden));
        }
        PipeBarrier<PIPE_V>();
        Muls(mean, mean, static_cast<ElementCompute>(1.0f / static_cast<float>(static_cast<int32_t>(hidden))),
             channels);
        PipeBarrier<PIPE_V>();
        Adds(mean, mean, static_cast<ElementCompute>(eps), channels);
        PipeBarrier<PIPE_V>();
        Sqrt(work, mean, channels);
        PipeBarrier<PIPE_V>();
        Duplicate(mean, static_cast<ElementCompute>(1.0f), channels);
        PipeBarrier<PIPE_V>();
        Div(mean, mean, work, channels);
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < channels; ++c) {
            Muls(work[c * hidden], grouped[c * hidden], mean.GetValue(c), hidden);
        }
        PipeBarrier<PIPE_V>();
        if (hasGamma) {
            for (uint32_t c = 0; c < channels; ++c) {
                Mul(xN[c * hidden], work[c * hidden], gamma, hidden);
            }
        } else {
            DataCopy(xN, work, d);
        }
        PipeBarrier<PIPE_V>();
    }
};

template <class ArchTag_, class ComputeType_>
struct TileMulSumInject {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<float> const& logits, AscendC::LocalTensor<float> const& xN,
                    AscendC::LocalTensor<float> const& wGate, AscendC::LocalTensor<float> const& prod,
                    AscendC::LocalTensor<float> const& reduceTmp, uint32_t channels, uint32_t d, float invC)
    {
        using namespace AscendC;
        for (uint32_t c = 0; c < channels; ++c) {
            Mul(prod, xN, wGate[c * d], d);
            PipeBarrier<PIPE_V>();
            ReduceSum(logits[c], prod, reduceTmp, static_cast<int32_t>(d));
            PipeBarrier<PIPE_V>();
        }
        Muls(logits, logits, invC, channels);
        PipeBarrier<PIPE_V>();
        Muls(prod, logits, -1.0f, channels);
        PipeBarrier<PIPE_V>();
        Exp(prod, prod, channels);
        PipeBarrier<PIPE_V>();
        Adds(prod, prod, 1.0f, channels);
        PipeBarrier<PIPE_V>();
        Duplicate(reduceTmp, 1.0f, channels);
        PipeBarrier<PIPE_V>();
        Div(logits, reduceTmp, prod, channels);
        PipeBarrier<PIPE_V>();
        Muls(logits, logits, 2.0f, channels);
        PipeBarrier<PIPE_V>();
    }
};

// Two independent register streams expose instruction-level parallelism to the VF scheduler.
__simd_vf__ inline void HcSiluVf(__ubuf__ float* dst, uint32_t count, float scale, bool applyScale)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t vl = 64;
    for (uint32_t base = 0; base < count; base += 2u * vl) {
        uint32_t n0 = count - base;
        uint32_t n1 = n0 > vl ? n0 - vl : 0u;
        MaskReg m0 = UpdateMask<float>(n0);
        MaskReg m1 = UpdateMask<float>(n1);
        RegTensor<float> x0, x1, e0, e1;
        LoadAlign(x0, dst + base);
        if (base + vl < count)
            LoadAlign(x1, dst + base + vl);
        else
            Duplicate(x1, 0.0f, m0);
        if (applyScale) {
            Muls(x0, x0, scale, m0);
            Muls(x1, x1, scale, m1);
        }
        Muls(e0, x0, -1.0f, m0);
        Muls(e1, x1, -1.0f, m1);
        Exp(e0, e0, m0);
        Exp(e1, e1, m1);
        Adds(e0, e0, 1.0f, m0);
        Adds(e1, e1, 1.0f, m1);
        Div(x0, x0, e0, m0);
        Div(x1, x1, e1, m1);
        StoreAlign(dst + base, x0, m0);
        if (base + vl < count)
            StoreAlign(dst + base + vl, x1, m1);
    }
}

__simd_vf__ inline void HcGateVf(__ubuf__ float* dst, uint32_t count)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t vl = 64;
    for (uint32_t base = 0; base < count; base += 2u * vl) {
        uint32_t n0 = count - base;
        uint32_t n1 = n0 > vl ? n0 - vl : 0u;
        MaskReg m0 = UpdateMask<float>(n0);
        MaskReg m1 = UpdateMask<float>(n1);
        RegTensor<float> x0, x1, e0, e1, one;
        Duplicate(one, 1.0f, m0);
        LoadAlign(x0, dst + base);
        if (base + vl < count)
            LoadAlign(x1, dst + base + vl);
        else
            Duplicate(x1, 0.0f, m0);
        Muls(e0, x0, -1.0f, m0);
        Muls(e1, x1, -1.0f, m1);
        Exp(e0, e0, m0);
        Exp(e1, e1, m1);
        Adds(e0, e0, 1.0f, m0);
        Adds(e1, e1, 1.0f, m1);
        Div(x0, one, e0, m0);
        Div(x1, one, e1, m1);
        Muls(x0, x0, 2.0f, m0);
        Muls(x1, x1, 2.0f, m1);
        StoreAlign(dst + base, x0, m0);
        if (base + vl < count)
            StoreAlign(dst + base + vl, x1, m1);
    }
}

template <class ArchTag_, class ComputeType_, uint32_t COMPUTE_LENGTH_, bool EnableRegbase = true>
struct TileDivSilu {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<ElementCompute> const& dst, AscendC::LocalTensor<ElementCompute> const& expBuf,
                    uint32_t count, float invC, bool applyInvC = true)
    {
        using namespace AscendC;
        if (!EnableRegbase || (count % 64u) != 0u) {
            uint32_t offset = 0;
            while (offset < count) {
                uint32_t n = (count - offset) < COMPUTE_LENGTH ? (count - offset) : COMPUTE_LENGTH;
                auto d = dst[offset];
                auto e = expBuf[offset];
                if (applyInvC) {
                    Muls(d, d, static_cast<ElementCompute>(invC), n);
                    PipeBarrier<PIPE_V>();
                }
                Muls(e, d, static_cast<ElementCompute>(-1), n);
                PipeBarrier<PIPE_V>();
                Exp(e, e, n);
                PipeBarrier<PIPE_V>();
                Adds(e, e, static_cast<ElementCompute>(1), n);
                PipeBarrier<PIPE_V>();
                Div(d, d, e, n);
                PipeBarrier<PIPE_V>();
                offset += n;
            }
            return;
        }
        (void)expBuf;
        HcSiluVf(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), count, invC, applyInvC);
        PipeBarrier<PIPE_V>();
    }
};

template <class ArchTag_, class ComputeType_>
struct TileInjectPost {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    template <class Element>
    CATLASS_DEVICE void operator()(AscendC::LocalTensor<float> const& ubLogitsC,
                                   AscendC::LocalTensor<float> const& ubWork, GM_ADDR gmInjectRow, uint32_t channels,
                                   float invC)
    {
        using namespace AscendC;
        if (channels == 0 || gmInjectRow == nullptr) {
            return;
        }
        const uint32_t n = 16;
        if constexpr (!std::is_same_v<Element, float>) {
            auto rounded = ubWork.template ReinterpretCast<Element>();
            Cast(rounded, ubLogitsC, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_V>();
            Cast(ubLogitsC, rounded, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
        }
        Muls(ubWork, ubLogitsC, invC, n);
        PipeBarrier<PIPE_V>();
        if constexpr (!std::is_same_v<Element, float>) {
            auto rounded = ubLogitsC.template ReinterpretCast<Element>();
            Cast(rounded, ubWork, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_V>();
            Cast(ubWork, rounded, RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
        }
        Muls(ubLogitsC, ubWork, -1.0f, n);
        PipeBarrier<PIPE_V>();
        Exp(ubLogitsC, ubLogitsC, n);
        PipeBarrier<PIPE_V>();
        Adds(ubLogitsC, ubLogitsC, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Duplicate(ubWork, 1.0f, n);
        PipeBarrier<PIPE_V>();
        Div(ubWork, ubWork, ubLogitsC, n);
        PipeBarrier<PIPE_V>();
        Muls(ubWork, ubWork, 2.0f, n);
        PipeBarrier<PIPE_V>();
        GlobalTensor<Element> gInj;
        gInj.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(gmInjectRow));
        DataCopyExtParams padParams;
        padParams.blockCount = 1;
        padParams.blockLen = channels * static_cast<uint32_t>(sizeof(Element));
        padParams.srcStride = 0;
        padParams.dstStride = 0;
        if constexpr (std::is_same_v<Element, float>) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(gInj, ubWork, padParams);
        } else {
            auto srcH = ubLogitsC.template ReinterpretCast<Element>();
            Cast(srcH, ubWork, RoundMode::CAST_RINT, n);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(gInj, srcH, padParams);
        }
        // 下一行会立刻重写这块 UB。MTE3 没结束就返回的话，写回读到的是被踩过的数据。
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
};

// Preserve channel order and the original reciprocal -> product -> scale sequence.
__simd_vf__ inline void HcSigmoidMulAccVf(__ubuf__ float* out, __ubuf__ float* x, __ubuf__ float* negativeUp,
                                          uint32_t channels, uint32_t hidden, float invC)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t vl = 64;
    for (uint32_t base = 0; base < hidden; base += 2u * vl) {
        uint32_t n0 = hidden - base;
        uint32_t n1 = n0 > vl ? n0 - vl : 0u;
        MaskReg m0 = UpdateMask<float>(n0);
        MaskReg m1 = UpdateMask<float>(n1);
        RegTensor<float> acc0, acc1, e0, e1, x0, x1, one;
        Duplicate(acc0, 0.0f, m0);
        Duplicate(acc1, 0.0f, m1);
        Duplicate(one, 1.0f, m0);
        for (uint32_t c = 0; c < channels; ++c) {
            uint32_t off = c * hidden + base;
            LoadAlign(e0, negativeUp + off);
            LoadAlign(x0, x + off);
            if (base + vl < hidden) {
                LoadAlign(e1, negativeUp + off + vl);
                LoadAlign(x1, x + off + vl);
            } else {
                Duplicate(e1, 0.0f, m0);
                Duplicate(x1, 0.0f, m0);
            }
            Exp(e0, e0, m0);
            Exp(e1, e1, m1);
            Adds(e0, e0, 1.0f, m0);
            Adds(e1, e1, 1.0f, m1);
            Div(e0, one, e0, m0);
            Div(e1, one, e1, m1);
            Mul(e0, e0, x0, m0);
            Mul(e1, e1, x1, m1);
            Muls(e0, e0, invC, m0);
            Muls(e1, e1, invC, m1);
            Add(acc0, acc0, e0, m0);
            Add(acc1, acc1, e1, m1);
        }
        StoreAlign(out + base, acc0, m0);
        if (base + vl < hidden)
            StoreAlign(out + base + vl, acc1, m1);
    }
}

template <class ArchTag_, class ComputeType_, bool EnableRegbase = true>
struct TileSigmoidMulAcc {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;

    // ubTmp 进入时已经是 -up。ones 是 1 个 block 的 1，Counter 模式下 src0 步长为 0。
    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<float> const& mixedRow, AscendC::LocalTensor<float> const& ubOnes,
                    AscendC::LocalTensor<float> const& ubXn, AscendC::LocalTensor<float> const& ubTmp,
                    uint32_t channels, uint32_t hidden, float invC)
    {
        using namespace AscendC;
        // Unaligned channel starts retain the existing generic vector path.
        if (!EnableRegbase || (hidden % 64u) != 0u) {
            const uint32_t d = channels * hidden;
            Exp(ubTmp, ubTmp, d);
            PipeBarrier<PIPE_V>();
            Adds(ubTmp, ubTmp, 1.0f, d);
            PipeBarrier<PIPE_V>();
            // Counter 一次 Div：mask=d，repeat=1。src0 是 1 个 block 的 1，blk/rep 步长为 0。
            const BinaryRepeatParams divRep{1, 0, 1, 8, 0, 8};
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(d);
            Div<float, false>(ubTmp, ubOnes, ubTmp, MASK_PLACEHOLDER, 1, divRep);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();
            ResetMask();
            Mul(ubTmp, ubTmp, ubXn, d);
            PipeBarrier<PIPE_V>();
            Muls(ubTmp, ubTmp, invC, d);
            PipeBarrier<PIPE_V>();
            Duplicate(mixedRow, 0.0f, hidden);
            PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < channels; ++c) {
                uint32_t off = c * hidden;
                Add(mixedRow, mixedRow, ubTmp[off], hidden);
                PipeBarrier<PIPE_V>();
            }
            return;
        }
        (void)ubOnes;
        HcSigmoidMulAccVf(reinterpret_cast<__ubuf__ float*>(mixedRow.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ float*>(ubXn.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ float*>(ubTmp.GetPhyAddr()), channels, hidden, invC);
        PipeBarrier<PIPE_V>();
    }
};

} // namespace Catlass::Epilogue::Tile
