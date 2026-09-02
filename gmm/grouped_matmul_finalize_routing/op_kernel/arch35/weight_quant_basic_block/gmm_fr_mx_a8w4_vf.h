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
 * \file gmm_fr_mx_a8w4_vf.h
 * \brief
 */
#ifndef GMM_FR_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H
#define GMM_FR_WEIGHT_QUANT_BASIC_BLOCK_VF_MX_H

#include "../common/basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

namespace MicroAPI = AscendC::MicroAPI;
using AscendC::BLOCK_CUBE;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2::Arch35 {

__simd_vf__ inline void InitZeroVf(__ubuf__ float *ubAddr, uint16_t loopCount)
{
    MicroAPI::RegTensor<float> zeroVreg;
    MicroAPI::Duplicate(zeroVreg, 0);
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t loopIdx = 0; loopIdx < loopCount; loopIdx++) {
        MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(loopIdx, QUADRUPLE_BUFFER_NUM * VEC_MAX_ELEM_B32);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(ubAddr, zeroVreg, outAddrReg, preg);
    }
}

static constexpr MicroAPI::CastTrait CAST_B16_TO_B32_TRAIT = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
;

template <typename sharedInputType>
__simd_vf__ inline void CastAndMulWithSharedWeightVf(__ubuf__ float *dstAddr, __ubuf__ sharedInputType *srcAddr,
                                                     uint16_t loopCount, float weight)
{
    MicroAPI::RegTensor<sharedInputType> sharedInputB16Vreg;
    MicroAPI::RegTensor<float> sharedInputB32Vreg;
    MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t loopIdx = 0; loopIdx < loopCount; loopIdx++) {
        MicroAPI::AddrReg sharedInputAreg = MicroAPI::CreateAddrReg<sharedInputType>(loopIdx, VEC_MAX_ELEM_B32);
        MicroAPI::LoadAlign<sharedInputType, MicroAPI::LoadDist::DIST_UNPACK_B16>(sharedInputB16Vreg, srcAddr,
                                                                                  sharedInputAreg);
        MicroAPI::Cast<float, sharedInputType, CAST_B16_TO_B32_TRAIT>(sharedInputB32Vreg, sharedInputB16Vreg, maskAll);
        MicroAPI::Muls(sharedInputB32Vreg, sharedInputB32Vreg, weight, maskAll);
        MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(loopIdx, QUADRUPLE_BUFFER_NUM * VEC_MAX_ELEM_B32);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(dstAddr, sharedInputB32Vreg, outAddrReg,
                                                                        maskAll);
    }
}

template <typename biasType>
__simd_callee__ inline void FrMxA8W4BiasCompute(uint16_t biasLoopNum, __ubuf__ biasType *biasInUbAddr,
                                                __ubuf__ biasType *biasOutUbAddr)
{
    static constexpr biasType MX_BIAS_FACTOR = static_cast<biasType>(0.015625f);
    MicroAPI::RegTensor<biasType> biasVreg, biasFactorVreg;
    MicroAPI::MaskReg maskBiasAll = MicroAPI::CreateMask<biasType, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasFactorVreg, MX_BIAS_FACTOR,
                                                                             maskBiasAll);
    for (uint16_t loopBiasIdx = 0; loopBiasIdx < biasLoopNum; ++loopBiasIdx) {
        MicroAPI::AddrReg biasAreg = MicroAPI::CreateAddrReg<biasType>(loopBiasIdx, VEC_MAX_ELEM_B16);
        MicroAPI::LoadAlign<biasType, MicroAPI::LoadDist::DIST_NORM>(biasVreg, biasInUbAddr, biasAreg);
        MicroAPI::Mul<biasType, AscendC::MicroAPI::MaskMergeMode::ZEROING>(biasVreg, biasVreg, biasFactorVreg,
                                                                           maskBiasAll);
        MicroAPI::StoreAlign<biasType, MicroAPI::StoreDist::DIST_NORM_B16>(biasOutUbAddr, biasVreg, biasAreg,
                                                                           maskBiasAll);
    }
}

template <typename xType, typename wType, typename biasType, bool hasBias>
__simd_vf__ inline void FrAntiQuantMxA8W4NzNkVf(MxA8W4NzParams<xType, wType, biasType> mxA8W4NzParams)
{
    if constexpr (hasBias) {
        FrMxA8W4BiasCompute<biasType>(mxA8W4NzParams.biasLoopNum, mxA8W4NzParams.biasInUbAddr,
                                      mxA8W4NzParams.biasOutUbAddr);
    }
    MicroAPI::RegTensor<int8_t> wShrReg, wShlReg, wAndReg, wLoad, wShl, wShr0, wShr1, wSel, wAnd;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg pregVsel = MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShrReg, E2M1_SHIFT_RIGHT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wShlReg, SHIFT_LEFT_SIZE, preg);
    MicroAPI::Duplicate<int8_t, AscendC::MicroAPI::MaskMergeMode::ZEROING>(wAndReg, E2M1_AND_MASK, preg);

    for (uint16_t loopKIdx = 0; loopKIdx < mxA8W4NzParams.loopKNum; ++loopKIdx) {
        for (uint16_t innerLoopIdx = 0; innerLoopIdx < mxA8W4NzParams.innerLoopNum; ++innerLoopIdx) {
            // DIST_US_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7
            // Vd 0 1 0 1 2 3 2 3 4 5 4 5 6 7 6 7
            // 4bit物理地址位移 = 逻辑索引 >> 1
            MicroAPI::AddrReg aregWeightB8In = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, (C0_SIZE_B8 * mxA8W4NzParams.nRealSizeAlign) >> 1, innerLoopIdx, VECTOR_REG_WIDTH >> 1);
            MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                (MicroAPI::RegTensor<uint8_t> &)wLoad, (__ubuf__ uint8_t *&)mxA8W4NzParams.weightLowBitPhyAddr,
                aregWeightB8In);

            MicroAPI::ShiftRight(wShr0, wLoad, wShrReg, preg);
            MicroAPI::ShiftLeft(wShl, wLoad, wShlReg, preg);
            MicroAPI::ShiftRight(wShr1, wShl, wShrReg, preg);
            MicroAPI::Select(wSel, wShr1, wShr0, pregVsel);
            MicroAPI::And(wAnd, wSel, wAndReg, preg);

            MicroAPI::AddrReg aregWeightB8Out = MicroAPI::CreateAddrReg<uint8_t>(
                loopKIdx, mxA8W4NzParams.loopKDstStride, innerLoopIdx, mxA8W4NzParams.innerDstStride);
            MicroAPI::StoreAlign<uint8_t, MicroAPI::StoreDist::DIST_NORM_B8>(
                (__ubuf__ uint8_t *&)mxA8W4NzParams.weightHighBitPhyAddr, (MicroAPI::RegTensor<uint8_t> &)wAnd,
                aregWeightB8Out, preg);
        }
    }
}

__simd_vf__ inline void FrMulLogitsVf(uint16_t mSize, uint64_t nLoopCnt, __ubuf__ float *logitsUbAddr,
                                      __ubuf__ float *yUbAddr)
{
    MicroAPI::RegTensor<float> yFp32;
    MicroAPI::RegTensor<float> logits;
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(logits, logitsUbAddr + mIdx);

        for (uint16_t nLoopIdx = 0; nLoopIdx < nLoopCnt; ++nLoopIdx) {
            MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
                yFp32, yUbAddr + mIdx * 256 + nLoopIdx * VEC_MAX_ELEM_B32);

            MicroAPI::Muls(yFp32, yFp32, 64.0f, maskAll);
            MicroAPI::Mul(yFp32, yFp32, logits, maskAll);

            MicroAPI::AddrReg outAddrReg = MicroAPI::CreateAddrReg<float>(mIdx, 256, nLoopIdx, VEC_MAX_ELEM_B32);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM_B32>(yUbAddr, yFp32, outAddrReg, maskAll);
        }
    }
}
} // namespace WeightQuantBatchMatmulV2::Arch35
#endif
