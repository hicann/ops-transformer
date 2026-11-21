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
#include "catlass/arch/resource.hpp"
#include "../hc_gm_copy.hpp"
namespace Catlass::Epilogue::LargeV0OneVf {
constexpr AscendC::Reg::CastTrait V0CastUp = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                              AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait V0CastDown = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <class E, bool HasGamma, bool SaveInv>
__simd_vf__ inline void HcV0NormVf(__ubuf__ E* dst, __ubuf__ E* src, __ubuf__ float* gamma, __ubuf__ float* invUb,
                                   uint32_t hidden, float eps)
{
    namespace R = AscendC::Reg;
    R::RegTensor<float> sum0, sum1, x0, x1, sq0, sq1, inv0, inv1, g0, g1, one, y0, y1;
    R::RegTensor<E> native0, native1;
    R::MaskReg mask = R::CreateMask<float>();
#pragma unroll 1
    for (uint16_t pair = 0; pair < 2u; ++pair) {
        uint32_t base0 = static_cast<uint32_t>(pair) * hidden * 2u, base1 = base0 + hidden;
        R::Duplicate(sum0, 0.0f, mask);
        R::Duplicate(sum1, 0.0f, mask);
#pragma unroll 1
        for (uint16_t j = 0; j < hidden / 64u; ++j) {
            uint32_t off = static_cast<uint32_t>(j) * 64u;
            R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(native0, src + base0 + off);
            R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(native1, src + base1 + off);
            R::Cast<float, E, V0CastUp>(x0, native0, mask);
            R::Cast<float, E, V0CastUp>(x1, native1, mask);
            R::Mul(sq0, x0, x0, mask);
            R::Mul(sq1, x1, x1, mask);
            R::Add(sum0, sum0, sq0, mask);
            R::Add(sum1, sum1, sq1, mask);
        }
        R::Reduce<R::ReduceType::SUM>(sum0, sum0, mask);
        R::Reduce<R::ReduceType::SUM>(sum1, sum1, mask);
        R::Duplicate(inv0, sum0, mask);
        R::Duplicate(inv1, sum1, mask);
        R::Muls(inv0, inv0, 1.0f / static_cast<float>(hidden), mask);
        R::Muls(inv1, inv1, 1.0f / static_cast<float>(hidden), mask);
        R::Adds(inv0, inv0, eps, mask);
        R::Adds(inv1, inv1, eps, mask);
        R::Sqrt(inv0, inv0, mask);
        R::Sqrt(inv1, inv1, mask);
        R::Duplicate(one, 1.0f, mask);
        R::Div(inv0, one, inv0, mask);
        R::Div(inv1, one, inv1, mask);
        if constexpr (SaveInv) {
            R::StoreAlign<float, R::StoreDist::DIST_FIRST_ELEMENT_B32>(invUb + pair * 2u, inv0, mask);
            R::StoreAlign<float, R::StoreDist::DIST_FIRST_ELEMENT_B32>(invUb + pair * 2u + 1u, inv1, mask);
        }
#pragma unroll 1
        for (uint16_t j = 0; j < hidden / 64u; ++j) {
            uint32_t off = static_cast<uint32_t>(j) * 64u;
            R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(native0, src + base0 + off);
            R::LoadAlign<E, R::LoadDist::DIST_UNPACK_B16>(native1, src + base1 + off);
            R::Cast<float, E, V0CastUp>(x0, native0, mask);
            R::Cast<float, E, V0CastUp>(x1, native1, mask);
            R::Mul(y0, x0, inv0, mask);
            R::Mul(y1, x1, inv1, mask);
            if constexpr (HasGamma) {
                R::LoadAlign(g0, gamma + base0 + off);
                R::LoadAlign(g1, gamma + base1 + off);
                R::Mul(y0, y0, g0, mask);
                R::Mul(y1, y1, g1, mask);
            }
            R::Cast<E, float, V0CastDown>(native0, y0, mask);
            R::Cast<E, float, V0CastDown>(native1, y1, mask);
            R::StoreAlign<E, R::StoreDist::DIST_PACK_B32>(dst + base0 + off, native0, mask);
            R::StoreAlign<E, R::StoreDist::DIST_PACK_B32>(dst + base1 + off, native1, mask);
        }
    }
}

template <class E>
struct BlockV0 {
    template <bool Gamma, bool SaveInv>
    CATLASS_DEVICE static void RunVf(AscendC::LocalTensor<E> output, AscendC::LocalTensor<E> input,
                                     AscendC::LocalTensor<float> gamma, AscendC::LocalTensor<float> inv,
                                     uint32_t hidden, float eps)
    {
        asc_vf_call<HcV0NormVf<E, Gamma, SaveInv>>(reinterpret_cast<__ubuf__ E*>(output.GetPhyAddr()),
                                                   reinterpret_cast<__ubuf__ E*>(input.GetPhyAddr()),
                                                   reinterpret_cast<__ubuf__ float*>(gamma.GetPhyAddr()),
                                                   reinterpret_cast<__ubuf__ float*>(inv.GetPhyAddr()), hidden, eps);
    }
    CATLASS_DEVICE static void Dispatch(AscendC::LocalTensor<E> output, AscendC::LocalTensor<E> input,
                                        AscendC::LocalTensor<float> gamma, AscendC::LocalTensor<float> inv,
                                        uint32_t hidden, float eps, bool hasGamma, bool saveInv)
    {
        if (hasGamma) {
            if (saveInv)
                RunVf<true, true>(output, input, gamma, inv, hidden, eps);
            else
                RunVf<true, false>(output, input, gamma, inv, hidden, eps);
        } else {
            if (saveInv)
                RunVf<false, true>(output, input, gamma, inv, hidden, eps);
            else
                RunVf<false, false>(output, input, gamma, inv, hidden, eps);
        }
    }
    template <class Resource>
    CATLASS_DEVICE void operator()(Resource& resource, GM_ADDR x, GM_ADDR xn, GM_ADDR gamma, uint32_t rows, uint32_t d,
                                   uint32_t channels, uint32_t hidden, float eps, bool hasGamma, bool disableSubSplit,
                                   GM_ADDR invRms, bool xnL2Disable, uint32_t taskRows = 0)
    {
        using namespace AscendC;
        static_assert(196640u <= Catlass::Arch::Ascend950::UB_SIZE, "V0 oneVF UB overflow");
        const uint32_t subs = GetSubBlockNum() == 0 ? 1 : GetSubBlockNum();
        const uint32_t splitRows = taskRows == 0 ? rows : taskRows;
        const uint32_t per = (splitRows + subs - 1u) / subs;
        const uint32_t begin = disableSubSplit ? 0u : GetSubBlockIdx() * per;
        const uint32_t end = disableSubSplit ? rows : (begin + per < rows ? begin + per : rows);
        if (begin >= end)
            return;
        const bool doubleRows = hidden == 2560u;
        const uint32_t rowBytes = d * sizeof(E);
        const uint32_t gammaOffset = doubleRows ? 4u * rowBytes : rowBytes;
        const uint32_t invOffset = gammaOffset + d * sizeof(float);
        auto gu = resource.ubBuf.template GetBufferByByte<float>(gammaOffset);
        GlobalTensor<E> gi, go;
        gi.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(x));
        go.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(xn));
        gi.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (xnL2Disable)
            go.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        GlobalTensor<float> gr;
        if (invRms != nullptr)
            gr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(invRms));
        if (hasGamma) {
            GlobalTensor<float> gg;
            gg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma));
            HcCopyGmToUb(gu, gg, d);
            SetFlag<HardEvent::MTE2_V>(3);
            WaitFlag<HardEvent::MTE2_V>(3);
            Adds(gu, gu, 1.0f, d);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopyExtParams invCopy{};
        invCopy.blockCount = 1;
        invCopy.blockLen = 4u * sizeof(float);
        if (doubleRows) {
            const event_t inputEvents[2] = {EVENT_ID0, EVENT_ID1};
            const event_t outputEvents[2] = {EVENT_ID4, EVENT_ID5};
            bool inputReleased[2] = {false, false};
            for (uint32_t s = 0; s < 2u; ++s)
                SetFlag<HardEvent::MTE3_V>(outputEvents[s]);
            for (uint32_t row = begin; row < end; ++row) {
                uint32_t stage = (row - begin) & 1u;
                if (inputReleased[stage]) {
                    WaitFlag<HardEvent::V_MTE2>(inputEvents[stage]);
                    inputReleased[stage] = false;
                }
                auto currentIn = resource.ubBuf.template GetBufferByByte<E>(stage * rowBytes);
                HcCopyGmToUb(currentIn, gi[static_cast<uint64_t>(row) * d], d);
                SetFlag<HardEvent::MTE2_V>(inputEvents[stage]);
                WaitFlag<HardEvent::MTE2_V>(inputEvents[stage]);
                WaitFlag<HardEvent::MTE3_V>(outputEvents[stage]);
                auto input = resource.ubBuf.template GetBufferByByte<E>(stage * rowBytes);
                auto output = resource.ubBuf.template GetBufferByByte<E>((2u + stage) * rowBytes);
                auto inv = resource.ubBuf.template GetBufferByByte<float>(invOffset + stage * 32u);
                Dispatch(output, input, gu, inv, hidden, eps, hasGamma, invRms != nullptr);
                SetFlag<HardEvent::V_MTE2>(inputEvents[stage]);
                inputReleased[stage] = true;
                SetFlag<HardEvent::V_MTE3>(outputEvents[stage]);
                WaitFlag<HardEvent::V_MTE3>(outputEvents[stage]);
                HcCopyUbToGm(go[static_cast<uint64_t>(row) * d], output, d);
                if (invRms != nullptr)
                    DataCopyPad(gr[static_cast<uint64_t>(row) * channels], inv, invCopy);
                SetFlag<HardEvent::MTE3_V>(outputEvents[stage]);
            }
            for (uint32_t s = 0; s < 2u; ++s) {
                WaitFlag<HardEvent::MTE3_V>(outputEvents[s]);
                if (inputReleased[s])
                    WaitFlag<HardEvent::V_MTE2>(inputEvents[s]);
            }
        } else {
            auto rowUb = resource.ubBuf.template GetBufferByByte<E>(0);
            auto inv = resource.ubBuf.template GetBufferByByte<float>(invOffset);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            for (uint32_t row = begin; row < end; ++row) {
                WaitFlag<HardEvent::MTE3_MTE2>(0);
                HcCopyGmToUb(rowUb, gi[static_cast<uint64_t>(row) * d], d);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                // Same UB pointer: entire pair reduction precedes the second pass;
                // both chunk loads precede 64-native-element PACK_B32 stores.
                Dispatch(rowUb, rowUb, gu, inv, hidden, eps, hasGamma, invRms != nullptr);
                SetFlag<HardEvent::V_MTE3>(4);
                WaitFlag<HardEvent::V_MTE3>(4);
                HcCopyUbToGm(go[static_cast<uint64_t>(row) * d], rowUb, d);
                if (invRms != nullptr)
                    DataCopyPad(gr[static_cast<uint64_t>(row) * channels], inv, invCopy);
                SetFlag<HardEvent::MTE3_MTE2>(0);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(0);
        }
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace Catlass::Epilogue::LargeV0OneVf
