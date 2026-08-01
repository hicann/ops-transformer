/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file lightning_indexer_v2_vector1.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_VECTOR1_H
#define LIGHTNING_INDEXER_V2_VECTOR1_H

#include "kernel_operator.h"
#include "common/lightning_indexer_v2_vector1_base.h"

namespace liV2Vector1 {

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<float> &out_,
                                              const LocalTensor<uint32_t> &in,
                                              const uint32_t topK,
                                              const uint32_t negInfBits)
{
    auto outBuf = (__local_mem__ float*)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t*)in.GetPhyAddr();

    const uint16_t repeatSize32 = 128;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> regIn[2];
        AscendC::MicroAPI::RegTensor<float> regOut[2];
        AscendC::MicroAPI::RegTensor<uint32_t> regNegInf;
        AscendC::MicroAPI::RegTensor<uint32_t> regZero;
        AscendC::MicroAPI::MaskReg maskInvalid[2];
        AscendC::MicroAPI::MaskReg maskAllB32 =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::Duplicate(regNegInf, negInfBits, maskAllB32);
        AscendC::MicroAPI::Duplicate(regZero, (uint32_t)0, maskAllB32);

        UIntSortConstCtx<float> uint32Ctx;
        InitUIntSortConstCtx(uint32Ctx, maskAllB32);
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);

            MicroAPI::Compare<uint32_t, CMPMODE::EQ>(maskInvalid[0], regIn[0], regZero, maskAllB32);
            MicroAPI::Compare<uint32_t, CMPMODE::EQ>(maskInvalid[1], regIn[1], regZero, maskAllB32);

            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            MicroAPI::Select((AscendC::MicroAPI::RegTensor<uint32_t>&)regOut[0], regNegInf,
                    (AscendC::MicroAPI::RegTensor<uint32_t>&)regOut[0], maskInvalid[0]);
            MicroAPI::Select((AscendC::MicroAPI::RegTensor<uint32_t>&)regOut[1], regNegInf,
                    (AscendC::MicroAPI::RegTensor<uint32_t>&)regOut[1], maskInvalid[1]);
            AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(outBuf + i * repeatSize32,
                                                                                          regOut[0],
                                                                                          maskAllB32);
            AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                outBuf + i * repeatSize32 + 64,
                regOut[1],
                maskAllB32);
        }
    }
}

template<typename W_T>
__aicore__ inline void MulWeightAndReduceSum(const LocalTensor<uint32_t> &out, // out    [S2Base]     [128   ] 2
                                             const LocalTensor<float> &qk, // q*k^t  [G, S2Base]  [64 128] 2
                                             const uint32_t qkVLStride,
                                             const LocalTensor<W_T> &weight, // w      [G]          [64    ] 1
                                             const int gSize) // G 64
{
    __local_mem__ W_T* weight_ = (__local_mem__ W_T*)weight.GetPhyAddr();

    constexpr uint32_t VL = 64; // vector length

    auto qk0 = (__local_mem__ float*)qk.GetPhyAddr();
    auto out0 = (__local_mem__ uint32_t*)out.GetPhyAddr();
    auto out1 = out0 + VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> brcGatherIndex;
        AscendC::MicroAPI::RegTensor<float> regQK[2];
        AscendC::MicroAPI::RegTensor<float> regW;
        AscendC::MicroAPI::RegTensor<float> regwBrc;
        AscendC::MicroAPI::RegTensor<float> regSum[2];

        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<float,
                                                                           AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg maskAll16 = AscendC::MicroAPI::CreateMask<W_T,
                                                                           AscendC::MicroAPI::MaskPattern::ALL>();

        FloatSortConstCtx<float> fp32Ctx;
        InitFloatSortConstCtx(fp32Ctx, maskAll);

        AscendC::MicroAPI::LoadAlign<W_T, AscendC::MicroAPI::LoadDist::DIST_NORM>(regW, weight_);

        AscendC::MicroAPI::Duplicate(regSum[0], 0.0f, maskAll);
        AscendC::MicroAPI::Duplicate(regSum[1], 0.0f, maskAll);

        for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); ++i) {
            AscendC::MicroAPI::Duplicate(brcGatherIndex, i);
            AscendC::MicroAPI::LoadAlign<float>(regQK[0], qk0 + 128 * i);
            AscendC::MicroAPI::LoadAlign<float>(regQK[1], qk0 + 128 * i + qkVLStride);
            AscendC::MicroAPI::Gather(regwBrc, regW, brcGatherIndex);

            AscendC::MicroAPI::Relu(regQK[0], regQK[0], maskAll);
            AscendC::MicroAPI::Relu(regQK[1], regQK[1], maskAll);

            AscendC::MicroAPI::MulAddDst(regSum[0], regQK[0], regwBrc, maskAll);
            AscendC::MicroAPI::MulAddDst(regSum[1], regQK[1], regwBrc, maskAll);
        }

        AscendC::MicroAPI::RegTensor<uint32_t> regOut[2];
        FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum[0], regSum[1], fp32Ctx, maskAll);

        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0, regOut[0], maskAll);
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1, regOut[1], maskAll);
    }
}
}

#endif