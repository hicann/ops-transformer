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
 * \file vf_rms_norm_arch92.h
 * \brief
 */

#ifndef VF_RMS_NORM_ARCH92_H
#define VF_RMS_NORM_ARCH92_H
#include "kernel_tensor.h"
#include "../mla_prolog_comm_arch92.h"

namespace MlaProlog {
constexpr uint64_t FLOAT_REP_SIZE = 64;

template <typename InType, typename GammaType, typename C, typename OutType>
__simd_vf__ void RmsNormVFImpl(__ubuf__ InType* inputBuf, __ubuf__ GammaType* gammaBuf, __ubuf__ OutType* outputBuf,
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
 * ★迁移(false 开关, V2/MLA_TOKEN_BATCH): RmsNorm 两行交织版(row==2 专用)。
 * 与单行版的结构差异(数值路径不变——Mul+Add 保持原样):
 *   1. 两行独立累加器交织: 行0 的 Add 依赖链间隙被行1 的 Load/Mul 填充(官方双发·队列深度),
 *      单次 VF 调用处理 2 行(消一次 VF 发射——"VF 框架开销"family: 逐行调用 ~62c/次);
 *   2. pass2 的 gamma 一次 Load 供两行(gamma Load 次数减半);
 *   3. 归约/rsqrt 链两行镜像(各一组 pregFirst 标量链)。
 * 输入/输出为 [2, col] 紧凑布局(行距 = rmsNormParams.col)。
 * 调用方: RmsNormVF wrapper 按 row 路由(仅 RmsNormCq 合批路径传 row=2)。
 */
template <typename InType, typename GammaType, typename C, typename OutType>
__simd_vf__ void RmsNormVFImplRow2(__ubuf__ InType* inputBuf, __ubuf__ GammaType* gammaBuf, __ubuf__ OutType* outputBuf,
                                   uint32_t repeatTimes, const RmsNormParam rmsNormParams)
{
    Reg::RegTensor<C> vregSum0;
    Reg::RegTensor<C> vregSum1;
    Reg::RegTensor<C> vregSumReduce0;
    Reg::RegTensor<C> vregSumReduce1;
    Reg::RegTensor<C> vregSquareRoot0;
    Reg::RegTensor<C> vregSquareRoot1;
    Reg::RegTensor<C> vregDiv0;
    Reg::RegTensor<C> vregDiv1;

    Reg::MaskReg pregAll = Reg::CreateMask<C, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregFirst = Reg::CreateMask<C, Reg::MaskPattern::VL1>();

    static constexpr Reg::CastTrait castTraitB162B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    const uint32_t rowStride = rmsNormParams.col; // 紧凑布局行距

    Reg::Duplicate<C, C>(vregSum0, 0.0);
    Reg::Duplicate<C, C>(vregSum1, 0.0);

    for (uint16_t i = 0; i < uint16_t(repeatTimes); ++i) {
        Reg::RegTensor<C> vregX0;
        Reg::RegTensor<C> vregX1;
        Reg::RegTensor<C> vregXSquare0;
        Reg::RegTensor<C> vregXSquare1;
        uint64_t loopOffset = i * FLOAT_REP_SIZE;

        Reg::LoadAlign<C, Reg::LoadDist::DIST_NORM>(vregX0, inputBuf + loopOffset);
        Reg::LoadAlign<C, Reg::LoadDist::DIST_NORM>(vregX1, inputBuf + rowStride + loopOffset);
        Reg::Mul<C, Reg::MaskMergeMode::ZEROING>(vregXSquare0, vregX0, vregX0, pregAll);
        Reg::Mul<C, Reg::MaskMergeMode::ZEROING>(vregXSquare1, vregX1, vregX1, pregAll);
        Reg::Add<C, Reg::MaskMergeMode::ZEROING>(vregSum0, vregXSquare0, vregSum0, pregAll);
        Reg::Add<C, Reg::MaskMergeMode::ZEROING>(vregSum1, vregXSquare1, vregSum1, pregAll);
    }

    Reg::Reduce<Reg::ReduceType::SUM, C, C, Reg::MaskMergeMode::ZEROING>(vregSumReduce0, vregSum0, pregAll);
    Reg::Reduce<Reg::ReduceType::SUM, C, C, Reg::MaskMergeMode::ZEROING>(vregSumReduce1, vregSum1, pregAll);
    Reg::Muls<C, C, Reg::MaskMergeMode::ZEROING>(vregSumReduce0, vregSumReduce0, rmsNormParams.reciprocal, pregFirst);
    Reg::Muls<C, C, Reg::MaskMergeMode::ZEROING>(vregSumReduce1, vregSumReduce1, rmsNormParams.reciprocal, pregFirst);
    Reg::Adds<C, C, Reg::MaskMergeMode::ZEROING>(vregSumReduce0, vregSumReduce0, rmsNormParams.epsilon, pregFirst);
    Reg::Adds<C, C, Reg::MaskMergeMode::ZEROING>(vregSumReduce1, vregSumReduce1, rmsNormParams.epsilon, pregFirst);
    Reg::Sqrt<C, Reg::MaskMergeMode::ZEROING>(vregSquareRoot0, vregSumReduce0, pregFirst);
    Reg::Sqrt<C, Reg::MaskMergeMode::ZEROING>(vregSquareRoot1, vregSumReduce1, pregFirst);
    Reg::Duplicate<C, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(vregDiv0, vregSquareRoot0, pregAll);
    Reg::Duplicate<C, Reg::HighLowPart::LOWEST, Reg::MaskMergeMode::ZEROING>(vregDiv1, vregSquareRoot1, pregAll);

    for (uint16_t i = 0; i < uint16_t(repeatTimes); ++i) {
        Reg::RegTensor<C> vregX0;
        Reg::RegTensor<C> vregX1;
        Reg::RegTensor<GammaType> vregGamma;
        Reg::RegTensor<C> vregGammaCast;
        uint16_t loopOffset = i * FLOAT_REP_SIZE;

        Reg::LoadAlign<C, Reg::LoadDist::DIST_NORM>(vregX0, inputBuf + loopOffset);
        Reg::LoadAlign<C, Reg::LoadDist::DIST_NORM>(vregX1, inputBuf + rowStride + loopOffset);
        Reg::LoadAlign<GammaType, Reg::LoadDist::DIST_UNPACK_B16>(vregGamma, gammaBuf + loopOffset);
        Reg::Cast<C, GammaType, castTraitB162B32>(vregGammaCast, vregGamma, pregAll);

        Reg::Div<C, Reg::MaskMergeMode::ZEROING>(vregX0, vregX0, vregDiv0, pregAll);
        Reg::Div<C, Reg::MaskMergeMode::ZEROING>(vregX1, vregX1, vregDiv1, pregAll);
        Reg::Mul<C, Reg::MaskMergeMode::ZEROING>(vregX0, vregX0, vregGammaCast, pregAll);
        Reg::Mul<C, Reg::MaskMergeMode::ZEROING>(vregX1, vregX1, vregGammaCast, pregAll);

        Reg::StoreAlign<OutType, Reg::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregX0, pregAll);
        Reg::StoreAlign<OutType, Reg::StoreDist::DIST_NORM>(outputBuf + rowStride + loopOffset, vregX1, pregAll);
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
__aicore__ inline void RmsNormVF(const LocalTensor<OutType>& outputLocal, const LocalTensor<InType>& inputLocal,
                                 const LocalTensor<GammaType>& gammaLocal, const RmsNormParam rmsNormParams)
{
    uint32_t cnt = rmsNormParams.row * rmsNormParams.col;
    uint32_t repeatTimes = (cnt + FLOAT_REP_SIZE - 1) / FLOAT_REP_SIZE;

    __ubuf__ InType* inputBuf = (__ubuf__ InType*)inputLocal.GetPhyAddr();
    __ubuf__ GammaType* gammaBuf = (__ubuf__ GammaType*)gammaLocal.GetPhyAddr();
    __ubuf__ OutType* outputBuf = (__ubuf__ OutType*)outputLocal.GetPhyAddr();

    if (rmsNormParams.row == 2u) {
        const uint32_t rowRepeatTimes = (rmsNormParams.col + FLOAT_REP_SIZE - 1) / FLOAT_REP_SIZE;
        RmsNormVFImplRow2<InType, GammaType, C, OutType>(inputBuf, gammaBuf, outputBuf, rowRepeatTimes, rmsNormParams);
    } else {
        RmsNormVFImpl<InType, GammaType, C, OutType>(inputBuf, gammaBuf, outputBuf, cnt, repeatTimes, rmsNormParams);
    }

    if (unlikely(rmsNormParams.isScaleEnable)) {
        Muls(outputLocal, outputLocal, rmsNormParams.scale, cnt);
    }
}
} // namespace MlaProlog
#endif
