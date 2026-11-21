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
#include "catlass/arch/resource.hpp"
#include "kernel_operator.h"
namespace Catlass::Epilogue::SmallModelA1B4 {
constexpr AscendC::Reg::CastTrait V1FullDown = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait V1FullUp = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                              AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};
template <class E>
__simd_vf__ inline void HcV1FullVf(__ubuf__ float* down, __ubuf__ float* gate, __ubuf__ E* act, __ubuf__ E* inject,
                                   uint32_t n, uint32_t split, uint32_t nr, uint32_t gateSplit)
{
    namespace V = AscendC::Reg;
    V::RegTensor<float> sum, part, value, den, one;
    V::RegTensor<E> native;
    V::MaskReg mask;
    uint32_t remaining = n;
#pragma unroll 1
    for (uint16_t j = 0; j < static_cast<uint16_t>((n + 63u) / 64u); ++j) {
        mask = V::UpdateMask<float>(remaining);
        uint32_t off = static_cast<uint32_t>(j) * 64u;
        V::LoadAlign(sum, down + off);
#pragma unroll 1
        for (uint16_t k0 = 0; k0 < static_cast<uint16_t>(split - 1u); ++k0) {
            V::LoadAlign(part, down + (static_cast<uint32_t>(k0) + 1u) * n + off);
            V::Add(sum, sum, part, mask);
        }
        V::Cast<E, float, V1FullDown>(native, sum, mask);
        V::Cast<float, E, V1FullUp>(value, native, mask);
        V::Muls(value, value, 0.25f, mask);
        V::Cast<E, float, V1FullDown>(native, value, mask);
        V::Cast<float, E, V1FullUp>(value, native, mask);
        V::Muls(den, value, -1.0f, mask);
        V::Exp(den, den, mask);
        V::Adds(den, den, 1.0f, mask);
        V::Div(value, value, den, mask);
        V::Cast<E, float, V1FullDown>(native, value, mask);
        V::StoreAlign<E, V::StoreDist::DIST_PACK_B32>(act + off, native, mask);
    }
#pragma unroll 1
    for (uint16_t row = 0; row < static_cast<uint16_t>(nr); ++row) {
        uint32_t valid = 4u;
        mask = V::UpdateMask<float>(valid);
        uint32_t off = static_cast<uint32_t>(row) * 8u;
        V::LoadAlign(sum, gate + off);
#pragma unroll 1
        for (uint16_t k0 = 0; k0 < static_cast<uint16_t>(gateSplit - 1u); ++k0) {
            V::LoadAlign(part, gate + (static_cast<uint32_t>(k0) + 1u) * nr * 8u + off);
            V::Add(sum, sum, part, mask);
        }
        V::Cast<E, float, V1FullDown>(native, sum, mask);
        V::Cast<float, E, V1FullUp>(value, native, mask);
        V::Muls(value, value, 0.25f, mask);
        V::Cast<E, float, V1FullDown>(native, value, mask);
        V::Cast<float, E, V1FullUp>(value, native, mask);
        V::Muls(den, value, -1.0f, mask);
        V::Exp(den, den, mask);
        V::Adds(den, den, 1.0f, mask);
        V::Duplicate(one, 1.0f, mask);
        V::Div(value, one, den, mask);
        V::Muls(value, value, 2.0f, mask);
        V::Cast<E, float, V1FullDown>(native, value, mask);
        V::StoreAlign<E, V::StoreDist::DIST_PACK_B32>(inject + static_cast<uint32_t>(row) * 16u, native, mask);
    }
}
struct BlockReduceV1Regbase {
    template <class E, class Resource>
    CATLASS_DEVICE static void Run(Resource& resource, GM_ADDR partial, GM_ADDR gatePartial, GM_ADDR act,
                                   GM_ADDR inject, uint32_t M, uint32_t R, uint32_t split, uint32_t gateSplit,
                                   bool gate)
    {
        using namespace AscendC;
        constexpr uint32_t RawFloats = 192u * 1024u / sizeof(float), GateOffset = 196608u, ActOffset = 201984u,
                           InjectOffset = 207872u, UbBytes = 208384u;
        static_assert(UbBytes <= Catlass::Arch::Ascend950::UB_SIZE, "V1 full UB overflow");
        auto down = resource.ubBuf.template GetBufferByByte<float>(0);
        auto rawGate = resource.ubBuf.template GetBufferByByte<float>(GateOffset);
        auto out = resource.ubBuf.template GetBufferByByte<E>(ActOffset);
        auto gateOut = resource.ubBuf.template GetBufferByByte<E>(InjectOffset);
        uint32_t sub = GetSubBlockNum() == 0u ? 1u : GetSubBlockNum(), aivNum = GetBlockNum() * sub, id = GetBlockIdx();
        uint64_t elements = static_cast<uint64_t>(M) * R;
        uint32_t tile = (elements / aivNum + 63u) / 64u * 64u;
        if (tile == 0u)
            tile = 64u;
        uint32_t cap = RawFloats / split / 64u * 64u;
        constexpr uint32_t ActCapacity = (InjectOffset - ActOffset) / sizeof(E) / 64u * 64u;
        cap = cap < ActCapacity ? cap : ActCapacity;
        tile = tile < cap ? tile : cap;
        uint32_t groups = (M + 3u) / 4u, begin = groups * id / aivNum * 4u, end = groups * (id + 1u) / aivNum * 4u;
        end = end < M ? end : M;
        uint32_t nr = gate && inject != nullptr ? end - begin : 0u;
        GlobalTensor<float> dp, gp;
        dp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial));
        gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gatePartial));
        GlobalTensor<E> da, di;
        da.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(act));
        di.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(inject));
        PipeBarrier<PIPE_ALL>();
        const uint64_t stride = static_cast<uint64_t>(aivNum) * tile;
        const uint32_t passes = (elements + stride - 1u) / stride;
        const uint32_t gateRows = nr;
        for (uint32_t pass = 0; pass < passes; ++pass) {
            const uint64_t off = static_cast<uint64_t>(id) * tile + static_cast<uint64_t>(pass) * stride;
            const uint32_t n = off < elements ? (elements - off < tile ? elements - off : tile) : 0u;
            const uint32_t nr = pass == 0u ? gateRows : 0u;
            if (n != 0u) {
                DataCopyExtParams cp(split, n * sizeof(float), (elements - n) * sizeof(float), 0, 0);
                DataCopyPadExtParams<float> pad(false, 0, 0, 0);
                DataCopyPad(down, dp[off], cp, pad);
            }
            if (nr != 0u) {
                DataCopyExtParams cp(nr, 4u * sizeof(float), 28u * sizeof(float), 0, 0);
                DataCopyPadExtParams<float> pad(true, 0, 4, 0.0f);
                for (uint32_t k = 0; k < gateSplit; ++k)
                    DataCopyPad(rawGate[k * nr * 8u],
                                gp[static_cast<uint64_t>(k) * M * 32u + static_cast<uint64_t>(begin) * 32u], cp, pad);
            }
            if (n != 0u || nr != 0u) {
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                asc_vf_call<HcV1FullVf<E>>(reinterpret_cast<__ubuf__ float*>(down.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ float*>(rawGate.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ E*>(out.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ E*>(gateOut.GetPhyAddr()), n, split, nr,
                                           gateSplit);
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                if (n != 0u)
                    DataCopy(da[off], out, n);
                DataCopyExtParams cp(1, 4u * sizeof(E), 0, 0, 0);
                for (uint32_t row = 0; row < nr; ++row)
                    DataCopyPad(di[static_cast<uint64_t>(begin + row) * 4u], gateOut[row * 16u], cp);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace Catlass::Epilogue::SmallModelA1B4
