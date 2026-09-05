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
 * \file vf_rms_norm.h
 * \brief
 */

#ifndef VF_RMS_NORM_H
#define VF_RMS_NORM_H
#include "kernel_tensor.h"

// repeatTimes——D轴的分块数
template <typename T, typename GammaType>
__simd_vf__ void RmsNormVFImpl(__ubuf__ T *inputBuf, __ubuf__ GammaType *gammaBuf, __ubuf__ T *outputBuf,
                               uint32_t repeatTimes, float reciprocal, float epsilon)
{
    Reg::RegTensor<T> vregSum;
    Reg::RegTensor<T> vregSumReduce;
    Reg::RegTensor<T> vregDiv;
    Reg::RegTensor<T> vregSquareRoot;

    Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskFirst = Reg::CreateMask<T, Reg::MaskPattern::VL1>();

    static constexpr Reg::CastTrait castTraitB162B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    Reg::Duplicate<T, T>(vregSum, 0.0f);

    for (uint32_t i = 0; i < repeatTimes; ++i) {
        Reg::RegTensor<T> vregX;
        Reg::RegTensor<T> vregXSquare;
        uint64_t loopOffset = i * FLOAT_REP_SIZE;

        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregX, inputBuf + loopOffset);
        Reg::Mul(vregXSquare, vregX, vregX, maskAll);
        Reg::Add(vregSum, vregXSquare, vregSum, maskAll);
    }

    Reg::Reduce<Reg::ReduceType::SUM, T, T, Reg::MaskMergeMode::ZEROING>(vregSumReduce, vregSum, maskAll);
    Reg::Muls<T, T, Reg::MaskMergeMode::ZEROING>(vregSumReduce, vregSumReduce, reciprocal, maskFirst);
    Reg::Adds<T, T, Reg::MaskMergeMode::ZEROING>(vregSumReduce, vregSumReduce, epsilon, maskFirst);
    Reg::Sqrt(vregSquareRoot, vregSumReduce, maskFirst);
    Reg::Duplicate<T, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(vregDiv, vregSquareRoot, maskAll);

    for (uint32_t i = 0; i < repeatTimes; ++i) {
        Reg::RegTensor<T> vregX;
        Reg::RegTensor<T> vregGammaCast;
        uint16_t loopOffset = i * FLOAT_REP_SIZE;

        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregX, inputBuf + loopOffset);
        Reg::LoadAlign<GammaType, Reg::LoadDist::DIST_NORM>(vregGammaCast, gammaBuf + loopOffset);

        Reg::Div(vregX, vregX, vregDiv, maskAll);
        Reg::Mul(vregX, vregX, vregGammaCast, maskAll);

        Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregX, maskAll);
    }
}

/**
 * @brief RmsNormVF 对一行进行rmsnorm
 * @param outputLocal 输出tensor [row, col]，row目前均为1
 * @param inputLocal 输入tensor [row, col]
 * @param gammaLocal gamma参数tensor [row, col]
 * @param rmsNormParams rmsNrom计算所需系数，包括
          row 行数  1
          col 列数，对应headSizeCq或headSizeCkv
          reciprocal ，1/N
          epsilon，防止除零极小数
 */
template <typename T, typename GammaType>
__aicore__ inline void RmsNormVF(const LocalTensor<T> outputLocal, const LocalTensor<T> inputLocal,
                                 const LocalTensor<GammaType> gammaLocal, float reciprocal, float epsilon, uint32_t row,
                                 uint32_t col)
{
    uint32_t cnt = row * col;
    uint32_t repeatTimes = (cnt + FLOAT_REP_SIZE - 1) / FLOAT_REP_SIZE;

    __ubuf__ T *inputBuf = (__ubuf__ T *)inputLocal.GetPhyAddr();
    __ubuf__ GammaType *gammaBuf = (__ubuf__ GammaType *)gammaLocal.GetPhyAddr();
    __ubuf__ T *outputBuf = (__ubuf__ T *)outputLocal.GetPhyAddr();

    RmsNormVFImpl<T, GammaType>(inputBuf, gammaBuf, outputBuf, repeatTimes, reciprocal, epsilon);
}

#endif
