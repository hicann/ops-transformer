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
namespace Catlass::Epilogue::SmallModelA1B4 {
constexpr AscendC::Reg::CastTrait V0CastUp = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                              AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait V0CastDown = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <class E, bool HasGamma>
__simd_vf__ inline void HcV0NormVf(__ubuf__ E* dst, __ubuf__ E* src, __ubuf__ float* gamma, float eps)
{
    namespace R = AscendC::Reg;
    R::RegTensor<float> sum0, sum1, x0, x1, sq0, sq1, inv0, inv1, g0, g1, one, y0, y1;
    R::RegTensor<E> native0, native1;
    R::MaskReg mask = R::CreateMask<float>();
#pragma unroll 1
    for (uint16_t pair = 0; pair < 2u; ++pair) {
        uint32_t base0 = static_cast<uint32_t>(pair) * 5120u, base1 = base0 + 2560u;
        R::Duplicate(sum0, 0.0f, mask);
        R::Duplicate(sum1, 0.0f, mask);
#pragma unroll 1
        for (uint16_t j = 0; j < 40u; ++j) {
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
        R::Muls(inv0, inv0, 1.0f / 2560.0f, mask);
        R::Muls(inv1, inv1, 1.0f / 2560.0f, mask);
        R::Adds(inv0, inv0, eps, mask);
        R::Adds(inv1, inv1, eps, mask);
        R::Sqrt(inv0, inv0, mask);
        R::Sqrt(inv1, inv1, mask);
        R::Duplicate(one, 1.0f, mask);
        R::Div(inv0, one, inv0, mask);
        R::Div(inv1, one, inv1, mask);
#pragma unroll 1
        for (uint16_t j = 0; j < 40u; ++j) {
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
struct BlockEpilogueHcV0Vf {
    template <class Resource>
    CATLASS_DEVICE void operator()(Resource& resource, GM_ADDR x, GM_ADDR xn, GM_ADDR gamma, uint32_t rows, float eps,
                                   bool hasGamma)
    {
        using namespace AscendC;
        if (rows == 0u)
            return;
        constexpr uint32_t D = 10240u, InOffset = 0u, OutOffset = 20480u, GammaOffset = 40960u, UbBytes = 81920u;
        static_assert(UbBytes <= Catlass::Arch::Ascend950::UB_SIZE, "V0 VF UB overflow");
        auto input = resource.ubBuf.template GetBufferByByte<E>(InOffset);
        auto output = resource.ubBuf.template GetBufferByByte<E>(OutOffset);
        auto gammaUb = resource.ubBuf.template GetBufferByByte<float>(GammaOffset);
        GlobalTensor<E> gi, go;
        gi.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(x));
        go.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(xn));
        gi.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (hasGamma) {
            GlobalTensor<float> gg;
            gg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma));
            HcCopyGmToUb(gammaUb, gg, D);
            SetFlag<HardEvent::MTE2_V>(3);
            WaitFlag<HardEvent::MTE2_V>(3);
            Adds(gammaUb, gammaUb, 1.0f, D);
            PipeBarrier<PIPE_ALL>();
        }
        SetFlag<HardEvent::MTE3_MTE2>(0);
        for (uint32_t row = 0; row < rows; ++row) {
            WaitFlag<HardEvent::MTE3_MTE2>(0);
            HcCopyGmToUb(input, gi[static_cast<uint64_t>(row) * D], D);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            if (hasGamma)
                asc_vf_call<HcV0NormVf<E, true>>(reinterpret_cast<__ubuf__ E*>(output.GetPhyAddr()),
                                                 reinterpret_cast<__ubuf__ E*>(input.GetPhyAddr()),
                                                 reinterpret_cast<__ubuf__ float*>(gammaUb.GetPhyAddr()), eps);
            else
                asc_vf_call<HcV0NormVf<E, false>>(reinterpret_cast<__ubuf__ E*>(output.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ E*>(input.GetPhyAddr()),
                                                  reinterpret_cast<__ubuf__ float*>(gammaUb.GetPhyAddr()), eps);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            HcCopyUbToGm(go[static_cast<uint64_t>(row) * D], output, D);
            SetFlag<HardEvent::MTE3_MTE2>(0);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        PipeBarrier<PIPE_ALL>();
    }
};
} // namespace Catlass::Epilogue::SmallModelA1B4
