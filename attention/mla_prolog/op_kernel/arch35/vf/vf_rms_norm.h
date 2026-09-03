/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

namespace MlaProlog {
constexpr uint64_t FLOAT_REP_SIZE = 64;

template <typename InType, typename GammaType, typename C, typename OutType>
__simd_vf__ void RmsNormVFImpl(__ubuf__ InType *inputBuf, __ubuf__ GammaType *gammaBuf, __ubuf__ OutType *outputBuf,
                               uint32_t cnt, uint32_t repeatTimes, const RmsNormParam rmsNormParams)
{
    Reg::RegTensor<C> vregSum;
    Reg::RegTensor<C> vregSumReduce;
    Reg::RegTensor<C> vregDiv;
    Reg::RegTensor<C> vregSquareRoot;

    Reg::MaskReg pregAll = Reg::CreateMask<C, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregFirst = Reg::CreateMask<C, Reg::MaskPattern::VL1>();

    static constexpr Reg::CastTrait castTraitB162B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    Reg::Duplicate<C, C>(vregSum, 0.0);

    for (uint16_t i = 0; i < uint16_t(repeatTimes); ++i) {
        Reg::RegTensor<C> vregXCast;
        Reg::RegTensor<C> vregXSquare;
        uint64_t loopOffset = i * FLOAT_REP_SIZE;

        Reg::LoadAlign<C, Reg::LoadDist::DIST_NORM>(vregXCast, inputBuf + loopOffset);
        Reg::Mul<C, Reg::MaskMergeMode::ZEROING>(vregXSquare, vregXCast, vregXCast, pregAll);
        Reg::Add<C, Reg::MaskMergeMode::ZEROING>(vregSum, vregXSquare, vregSum, pregAll);
    }

    Reg::Reduce<Reg::ReduceType::SUM, C, C, Reg::MaskMergeMode::ZEROING>(vregSumReduce, vregSum, pregAll);
    Reg::Muls<C, C, Reg::MaskMergeMode::ZEROING>(vregSumReduce, vregSumReduce, rmsNormParams.reciprocal, pregFirst);
    Reg::Adds<C, C, Reg::MaskMergeMode::ZEROING>(vregSumReduce, vregSumReduce, rmsNormParams.epsilon, pregFirst);
    Reg::Sqrt<C, Reg::MaskMergeMode::ZEROING>(vregSquareRoot, vregSumReduce, pregFirst);
    Reg::Duplicate<C, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(vregDiv, vregSquareRoot, pregAll);

    for (uint16_t i = 0; i < uint16_t(repeatTimes); ++i) {
        Reg::RegTensor<C> vregXCast;
        Reg::RegTensor<GammaType> vregGamma;
        Reg::RegTensor<C> vregGammaCast;
        uint16_t loopOffset = i * FLOAT_REP_SIZE;

        Reg::LoadAlign<C, Reg::LoadDist::DIST_NORM>(vregXCast, inputBuf + loopOffset);
        Reg::LoadAlign<GammaType, Reg::LoadDist::DIST_UNPACK_B16>(vregGamma, gammaBuf + loopOffset);
        Reg::Cast<C, GammaType, castTraitB162B32>(vregGammaCast, vregGamma, pregAll);

        Reg::Div<C, Reg::MaskMergeMode::ZEROING>(vregXCast, vregXCast, vregDiv, pregAll);
        Reg::Mul<C, Reg::MaskMergeMode::ZEROING>(vregXCast, vregXCast, vregGammaCast, pregAll);

        Reg::StoreAlign<OutType, Reg::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregXCast, pregAll);
    }
}

/**
 * @brief RmsNormVF 对一行进行rmsnorm
 * @param outputLocal 输出tensor [row, col]，row目前均为1
 * @param inputLocal 输入tensor [row, col]
 * @param gammaLocal gamma参数tensor [row, col]
 * @param rmsNormParams rmsNrom计算所需系数，包括
          row 行数
          col 列数，对应headSizeCq或headSizeCkv
          reciprocal ，1/N
          epsilon，防止除零极小数
 */
template <typename InType, typename GammaType, typename C, typename OutType>
__aicore__ inline void RmsNormVF(const LocalTensor<OutType> &outputLocal, const LocalTensor<InType> &inputLocal,
                                 const LocalTensor<GammaType> &gammaLocal, const RmsNormParam rmsNormParams)
{
    uint32_t cnt = rmsNormParams.row * rmsNormParams.col;
    uint32_t repeatTimes = (cnt + FLOAT_REP_SIZE - 1) / FLOAT_REP_SIZE;

    __ubuf__ InType *inputBuf = (__ubuf__ InType *)inputLocal.GetPhyAddr();
    __ubuf__ GammaType *gammaBuf = (__ubuf__ GammaType *)gammaLocal.GetPhyAddr();
    __ubuf__ OutType *outputBuf = (__ubuf__ OutType *)outputLocal.GetPhyAddr();

    RmsNormVFImpl<InType, GammaType, C, OutType>(inputBuf, gammaBuf, outputBuf, cnt, repeatTimes, rmsNormParams);

    if (unlikely(rmsNormParams.isScaleEnable)) {
        AscendC::PipeBarrier<PIPE_V>();
        Muls(outputLocal, outputLocal, rmsNormParams.scale, cnt);
    }
}
} // namespace MlaProlog
#endif
