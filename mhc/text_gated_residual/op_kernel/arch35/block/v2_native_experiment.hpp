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
#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "../hc_gm_copy.hpp"
namespace Catlass::Epilogue::V2NativeExperiment {
constexpr AscendC::Reg::CastTrait UP = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                        AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait DOWN = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                          AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
template <class E>
__simd_vf__ inline void Compute(__ubuf__ E* out, __ubuf__ E* up, __ubuf__ E* xn, uint32_t channels, uint32_t width,
                                float invC)
{
    namespace R = AscendC::Reg;
    R::MaskReg m = R::CreateMask<float>();
    R::RegTensor<float> a0, a1, u0, u1, x0, x1, one;
    R::RegTensor<E> t0, t1;
    R::Duplicate(one, 1.0f, m);
#pragma unroll 1
    for (uint32_t j = 0; j < width / 128u; ++j) {
        uint32_t b = static_cast<uint32_t>(j) * 128u;
        R::Duplicate(a0, 0.0f, m);
        R::Duplicate(a1, 0.0f, m);
#pragma unroll 1
        for (uint32_t c = 0; c < channels; ++c) {
            uint32_t o = c * width + b;
            R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(t0, up + o);
            R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(t1, up + o + 64u);
            R::Cast<float, E, UP>(u0, t0, m);
            R::Cast<float, E, UP>(u1, t1, m);
            R::Muls(u0, u0, -1.0f, m);
            R::Muls(u1, u1, -1.0f, m);
            R::Exp(u0, u0, m);
            R::Exp(u1, u1, m);
            R::Adds(u0, u0, 1.0f, m);
            R::Adds(u1, u1, 1.0f, m);
            R::Div(u0, one, u0, m);
            R::Div(u1, one, u1, m);
            R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(t0, xn + o);
            R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(t1, xn + o + 64u);
            R::Cast<float, E, UP>(x0, t0, m);
            R::Cast<float, E, UP>(x1, t1, m);
            R::Mul(u0, u0, x0, m);
            R::Mul(u1, u1, x1, m);
            R::Muls(u0, u0, invC, m);
            R::Muls(u1, u1, invC, m);
            R::Add(a0, a0, u0, m);
            R::Add(a1, a1, u1, m);
        }
        R::Cast<E, float, DOWN>(t0, a0, m);
        R::Cast<E, float, DOWN>(t1, a1, m);
        R::StoreAlign<E, R::StoreDist::DIST_PACK_B32>(out + b, t0, m);
        R::StoreAlign<E, R::StoreDist::DIST_PACK_B32>(out + b + 64u, t1, m);
    }
}
template <class E>
__simd_vf__ inline void Accumulate(__ubuf__ float* acc, __ubuf__ E* up, __ubuf__ E* xn, __ubuf__ E* out, uint32_t width,
                                   float invC, bool first, bool last)
{
    namespace R = AscendC::Reg;
    auto m = R::CreateMask<float>();
    R::RegTensor<float> a, u, x, one;
    R::RegTensor<E> t;
    R::Duplicate(one, 1.0f, m);
    for (uint32_t b = 0; b < width; b += 64u) {
        if (first)
            R::Duplicate(a, 0.0f, m);
        else
            R::LoadAlign(a, acc + b);
        R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(t, up + b);
        R::Cast<float, E, UP>(u, t, m);
        R::Muls(u, u, -1.0f, m);
        R::Exp(u, u, m);
        R::Adds(u, u, 1.0f, m);
        R::Div(u, one, u, m);
        R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(t, xn + b);
        R::Cast<float, E, UP>(x, t, m);
        R::Mul(u, u, x, m);
        R::Muls(u, u, invC, m);
        R::Add(a, a, u, m);
        if (last) {
            R::Cast<E, float, DOWN>(t, a, m);
            R::StoreAlign<E, R::StoreDist::DIST_PACK_B32>(out + b, t, m);
        } else
            R::StoreAlign(acc + b, a, m);
    }
}

// Whole rows use constant ping-pong slots. Other widths use native VF stripes;
// only valid GM elements are copied, with UB padding for unmasked vector loads.
template <class E, uint32_t Slot>
CATLASS_DEVICE void Step(__ubuf__ uint8_t* u, __ubuf__ uint8_t* x, __ubuf__ uint8_t* y, __gm__ uint8_t* gu,
                         __gm__ uint8_t* gx, __gm__ uint8_t* gy, uint32_t row, uint32_t end, uint32_t inputBytes,
                         uint32_t outputBytes, uint32_t channels, uint32_t hidden, float invC)
{
    using namespace AscendC;
    constexpr uint32_t Next = Slot ^ 1u;
    constexpr event_t cur = Slot == 0u ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t next = Next == 0u ? EVENT_ID0 : EVENT_ID1;
    WaitFlag<HardEvent::MTE2_V>(cur);
    if (row + 1u < end) {
        WaitFlag<HardEvent::V_MTE2>(next);
        asc_copy_gm2ub_align(u + Next * inputBytes, gu + (static_cast<uint64_t>(row) + 1u) * inputBytes, 1u, inputBytes,
                             0u, 0u, true, 4u, inputBytes, inputBytes);
        asc_copy_gm2ub_align(x + Next * inputBytes, gx + (static_cast<uint64_t>(row) + 1u) * inputBytes, 1u, inputBytes,
                             0u, 0u, true, 4u, inputBytes, inputBytes);
        SetFlag<HardEvent::MTE2_V>(next);
    }
    WaitFlag<HardEvent::MTE3_V>(cur);
    asc_vf_call<Compute<E>>(reinterpret_cast<__ubuf__ E*>(y + Slot * outputBytes),
                            reinterpret_cast<__ubuf__ E*>(u + Slot * inputBytes),
                            reinterpret_cast<__ubuf__ E*>(x + Slot * inputBytes), channels, hidden, invC);
    SetFlag<HardEvent::V_MTE2>(cur);
    SetFlag<HardEvent::V_MTE3>(cur);
    WaitFlag<HardEvent::V_MTE3>(cur);
    asc_copy_ub2gm_align(gy + static_cast<uint64_t>(row) * outputBytes, y + Slot * outputBytes, 1u, outputBytes, 4u,
                         outputBytes, outputBytes);
    SetFlag<HardEvent::MTE3_V>(cur);
}

template <class E, class Resource>
CATLASS_DEVICE void Run(Resource& resource, GM_ADDR up, GM_ADDR xn, GM_ADDR mixed, uint32_t begin, uint32_t end,
                        uint32_t channels, uint32_t hidden)
{
    using namespace AscendC;
    constexpr uint32_t UB_BYTES = 248u * 1024u;
    const uint64_t inputBytes = static_cast<uint64_t>(channels) * hidden * sizeof(E);
    const uint64_t outputBytes = static_cast<uint64_t>(hidden) * sizeof(E);
    const float invC = 1.0f / static_cast<float>(channels);
    auto ub = resource.ubBuf.template GetBufferByByte<uint8_t>(0);
    PipeBarrier<PIPE_ALL>();
    if (hidden % 128u == 0u && 4u * inputBytes + 2u * outputBytes <= UB_BYTES) {
        auto u = reinterpret_cast<__ubuf__ uint8_t*>(ub.GetPhyAddr());
        auto x = u + 2u * inputBytes;
        auto y = u + 4u * inputBytes;
        auto gu = reinterpret_cast<__gm__ uint8_t*>(up);
        auto gx = reinterpret_cast<__gm__ uint8_t*>(xn);
        auto gy = reinterpret_cast<__gm__ uint8_t*>(mixed);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        asc_copy_gm2ub_align(u, gu + static_cast<uint64_t>(begin) * inputBytes, 1u, inputBytes, 0u, 0u, true, 4u,
                             inputBytes, inputBytes);
        asc_copy_gm2ub_align(x, gx + static_cast<uint64_t>(begin) * inputBytes, 1u, inputBytes, 0u, 0u, true, 4u,
                             inputBytes, inputBytes);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t row = begin; row < end; row += 2u) {
            Step<E, 0u>(u, x, y, gu, gx, gy, row, end, inputBytes, outputBytes, channels, hidden, invC);
            if (row + 1u < end)
                Step<E, 1u>(u, x, y, gu, gx, gy, row + 1u, end, inputBytes, outputBytes, channels, hidden, invC);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    } else {
        constexpr uint32_t W = 1024u;
        auto u = ub.template ReinterpretCast<E>();
        auto x = ub[W * sizeof(E)].template ReinterpretCast<E>();
        auto y = ub[2u * W * sizeof(E)].template ReinterpretCast<E>();
        auto acc = ub[3u * W * sizeof(E)].template ReinterpretCast<float>();
        GlobalTensor<E> gu, gx, gy;
        gu.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(up));
        gx.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(xn));
        gy.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(mixed));
        gu.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gx.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gy.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        for (uint32_t row = begin; row < end; ++row) {
            for (uint32_t b = 0; b < hidden;) {
                const uint32_t valid = hidden - b < W ? hidden - b : W;
                const uint32_t width = (valid + 127u) / 128u * 128u;
                Duplicate(u, static_cast<E>(0), width);
                Duplicate(x, static_cast<E>(0), width);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                for (uint32_t c = 0; c < channels; ++c) {
                    const uint64_t offset = (static_cast<uint64_t>(row) * channels + c) * hidden + b;
                    HcCopyGmToUb(u, gu[offset], valid);
                    HcCopyGmToUb(x, gx[offset], valid);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    asc_vf_call<Accumulate<E>>(
                        reinterpret_cast<__ubuf__ float*>(acc.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ E*>(u.GetPhyAddr()), reinterpret_cast<__ubuf__ E*>(x.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ E*>(y.GetPhyAddr()), width, invC, c == 0u, c + 1u == channels);
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                HcCopyUbToGm(gy[static_cast<uint64_t>(row) * hidden + b], y, valid);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                b += valid;
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}
} // namespace Catlass::Epilogue::V2NativeExperiment
