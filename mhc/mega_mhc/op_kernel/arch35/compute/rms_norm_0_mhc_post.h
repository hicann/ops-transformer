/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_0_MHC_POST_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_0_MHC_POST_H

#include "kernel_operator.h"
#include "mega_mhc_post.h"
#include "rms_norm_common.h"

namespace MegaMhc::Compute {
using namespace AscendC;

/** Fuse RMSNorm_0 apply and Post without materializing x1 in UB.
 * x1[d]=Cast<T>(FP32(hOut[d])*invRms*gamma0[d]); output[n,d]=Cast<T>(sum_j
 * hRes[j,n]*FP32(residual[j,d])+hPost[n]*FP32(x1[d])). Preserve the x1 Cast before Post. Accumulate residual heads
 * 0,1,2,3 first, then add hPost*x1. Residual and output may alias: load all four source heads before each store. The
 * caller ensures MTE2_V readiness. HAS_ROW_SQUARES consumes precomputed RMS0 squares; ACCUMULATE_SQUARES produces
 * rounded Post squares for Mid. WITH_RMS0=false uses hOut directly and never reads gamma0 or RMS0 statistics. */
template <typename T, bool ACCUMULATE_SQUARES = false, bool HAS_ROW_SQUARES = false, bool WITH_RMS0 = true>
__aicore__ inline void ComputeRmsNorm0PostRow(__ubuf__ T* hOut, __ubuf__ float* gamma0, float epsilon,
                                              __ubuf__ T* residual, __ubuf__ float* hPost, __ubuf__ float* hRes,
                                              __ubuf__ T* output, uint32_t d, __ubuf__ float* sumScratch,
                                              __ubuf__ float* sumSquares = nullptr)
{
    const float meanScale = 1.0f / static_cast<float>(d);
    const uint16_t repeatTimes = static_cast<uint16_t>((d + kRegLaneRounding) / kRegFloatLanes);
    if constexpr (WITH_RMS0 && !HAS_ROW_SQUARES) {
        AccumulateRowSquares<T>(hOut, d, sumScratch);
    }
    __VEC_SCOPE__
    {
        Reg::MaskReg fullMask = Compute::MaskForCount(kRegFloatLanes);
        Reg::RegTensor<float> squares01;
        Reg::RegTensor<float> squares23;
        Reg::RegTensor<float> post0;
        Reg::RegTensor<float> post1;
        Reg::RegTensor<float> post2;
        Reg::RegTensor<float> post3;
        Reg::RegTensor<float> w00;
        Reg::RegTensor<float> w01;
        Reg::RegTensor<float> w02;
        Reg::RegTensor<float> w03;
        Reg::RegTensor<float> w10;
        Reg::RegTensor<float> w11;
        Reg::RegTensor<float> w12;
        Reg::RegTensor<float> w13;
        Reg::RegTensor<float> w20;
        Reg::RegTensor<float> w21;
        Reg::RegTensor<float> w22;
        Reg::RegTensor<float> w23;
        Reg::RegTensor<float> w30;
        Reg::RegTensor<float> w31;
        Reg::RegTensor<float> w32;
        Reg::RegTensor<float> w33;
        Reg::RegTensor<T> input;
        Reg::RegTensor<float> value;
        Reg::RegTensor<float> gamma;
        Reg::RegTensor<float> inverse;
        Reg::RegTensor<float> scalarInverse;
        Reg::RegTensor<float> sum0;
        Reg::RegTensor<float> sum1;
        Reg::RegTensor<float> sum2;
        Reg::RegTensor<float> sum3;
        Reg::RegTensor<float> term1;
        Reg::RegTensor<float> reduced;
        Reg::MaskReg mask;
        if constexpr (WITH_RMS0) {
            InverseRmsRegister(sumScratch, meanScale, epsilon, scalarInverse);
            Reg::Duplicate(inverse, scalarInverse, fullMask);
        }
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post0, hPost + 0U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post1, hPost + 1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post2, hPost + 2U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(post3, hPost + 3U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w00, hRes + 0U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w01, hRes + 1U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w02, hRes + 2U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w03, hRes + 3U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w10, hRes + 4U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w11, hRes + 5U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w12, hRes + 6U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w13, hRes + 7U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w20, hRes + 8U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w21, hRes + 9U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w22, hRes + 10U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w23, hRes + 11U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w30, hRes + 12U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w31, hRes + 13U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w32, hRes + 14U);
        Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(w33, hRes + 15U);
        auto residual0 = residual;
        auto residual1 = residual0 + d;
        auto residual2 = residual1 + d;
        auto residual3 = residual2 + d;
        auto output0 = output;
        auto output1 = output0 + d;
        auto output2 = output1 + d;
        auto output3 = output2 + d;
        if constexpr (ACCUMULATE_SQUARES) {
            Reg::Duplicate(squares01, 0.0f, fullMask);
            Reg::Duplicate(squares23, 0.0f, fullMask);
        }
        uint32_t remaining = d;
        for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
            const uint32_t column = repeat * kRegCastVectorLanes;
            mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(input, hOut + column);
            Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            if constexpr (WITH_RMS0) {
                Reg::LoadAlign<float>(gamma, gamma0 + column);
                Reg::Mul(value, value, inverse, mask);
                Reg::Mul(value, value, gamma, mask);
                Reg::Cast<T, float, kCastFloatToInput>(input, value, mask);
                Reg::Cast<float, T, kCastInputToFloat>(value, input, mask);
            }
            AccumulatePostSource<T, true>(residual0 + column, w00, w01, w02, w03, sum0, sum1, sum2, sum3, mask);
            AccumulatePostSource<T>(residual1 + column, w10, w11, w12, w13, sum0, sum1, sum2, sum3, mask);
            AccumulatePostSource<T>(residual2 + column, w20, w21, w22, w23, sum0, sum1, sum2, sum3, mask);
            AccumulatePostSource<T>(residual3 + column, w30, w31, w32, w33, sum0, sum1, sum2, sum3, mask);
            Reg::Mul(term1, value, post0, mask);
            Reg::Add(sum0, sum0, term1, mask);
            Reg::Mul(term1, value, post1, mask);
            Reg::Add(sum1, sum1, term1, mask);
            Reg::Mul(term1, value, post2, mask);
            Reg::Add(sum2, sum2, term1, mask);
            Reg::Mul(term1, value, post3, mask);
            Reg::Add(sum3, sum3, term1, mask);
            if constexpr (ACCUMULATE_SQUARES) {
                Reg::Cast<T, float, kCastFloatToInput>(input, sum0, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output0 + column, input, mask);
                Reg::Cast<float, T, kCastInputToFloat>(sum0, input, mask);
                Reg::Mul(sum0, sum0, sum0, fullMask);
                Reg::Cast<T, float, kCastFloatToInput>(input, sum1, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output1 + column, input, mask);
                Reg::Cast<float, T, kCastInputToFloat>(sum1, input, mask);
                Reg::MulAddDst(sum0, sum1, sum1, fullMask);
                Reg::Add(squares01, squares01, sum0, fullMask);
                Reg::Cast<T, float, kCastFloatToInput>(input, sum2, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output2 + column, input, mask);
                Reg::Cast<float, T, kCastInputToFloat>(sum2, input, mask);
                Reg::Mul(sum2, sum2, sum2, fullMask);
                Reg::Cast<T, float, kCastFloatToInput>(input, sum3, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output3 + column, input, mask);
                Reg::Cast<float, T, kCastInputToFloat>(sum3, input, mask);
                Reg::MulAddDst(sum2, sum3, sum3, fullMask);
                Reg::Add(squares23, squares23, sum2, fullMask);
            } else {
                Reg::Cast<T, float, kCastFloatToInput>(input, sum0, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output0 + column, input, mask);
                Reg::Cast<T, float, kCastFloatToInput>(input, sum1, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output1 + column, input, mask);
                Reg::Cast<T, float, kCastFloatToInput>(input, sum2, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output2 + column, input, mask);
                Reg::Cast<T, float, kCastFloatToInput>(input, sum3, mask);
                Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output3 + column, input, mask);
            }
        }
        if constexpr (ACCUMULATE_SQUARES) {
            Reg::Add(squares01, squares01, squares23, fullMask);
            Reg::Reduce<Reg::ReduceType::SUM>(reduced, squares01, fullMask);
            Reg::Store<float>(sumSquares, reduced, 1U);
        }
    }
}

} // namespace MegaMhc::Compute

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_RMS_NORM_0_MHC_POST_H
