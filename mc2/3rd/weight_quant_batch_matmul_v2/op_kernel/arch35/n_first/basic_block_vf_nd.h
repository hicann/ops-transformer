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
 * \file basic_block_vf_nd.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_VF_ND_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_VF_ND_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Reg = AscendC::Reg;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::Reg::AddrReg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType>
struct LocalAddressParam {
    __local_mem__ xType *antiQuantScaleBasePhyAddr;
    __local_mem__ xType *antiQuantScaleBasePhyAddr1;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1;
    __local_mem__ wType *weightLowBitPhyAddr0;
    __local_mem__ wType *weightLowBitPhyAddr1;
    __local_mem__ xType *weightF16PhyAddr0;
    __local_mem__ xType *weightF16PhyAddr1;
};

template <typename xType>
struct CalculateParam {
    xType offsetValue;
    xType scaleValue;
    uint64_t ubLoop;
};

static constexpr Reg::CastTrait FP8_TO_FP32_TRAIT_0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                       Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr Reg::CastTrait FP8_TO_FP32_TRAIT_2 = {Reg::RegLayout::TWO, Reg::SatMode::UNKNOWN,
                                                       Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

static constexpr Reg::CastTrait FP32_TO_F16_ODD = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                   Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};
static constexpr Reg::CastTrait FP32_TO_F16_EVEN = {Reg::RegLayout::ONE, Reg::SatMode::NO_SAT,
                                                    Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_ROUND};

static constexpr Reg::CastTrait S8_TO_FP16_TRAIT_ODD = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                        Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
static constexpr Reg::CastTrait FP16_TO_BF16_TRAIT = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <typename xType, typename wType, bool hasAntiQuantOffset, Mc2QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdNkVfLoadScaleOffset(RegTensor<xType> &antiQuantScaleVreg,
                                                         RegTensor<xType> &antiQuantOffsetVreg,
                                                         LocalAddressParam<xType, wType> &localAddressParam,
                                                         const CalculateParam<xType> &calculateParam, uint16_t nIdx)
{
    if constexpr (hasAntiQuantOffset) {
        Reg::DataCopy<xType, Reg::LoadDist::DIST_BRC_B16>(antiQuantOffsetVreg,
                                                          localAddressParam.antiQuantOffsetBasePhyAddr + nIdx);
    }
    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        Reg::Duplicate(antiQuantScaleVreg, calculateParam.scaleValue);
    } else {
        Reg::DataCopy<xType, Reg::LoadDist::DIST_BRC_B16>(antiQuantScaleVreg,
                                                          localAddressParam.antiQuantScaleBasePhyAddr + nIdx);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize>
__aicore__ inline void AntiQuantFP8NdNkVfLoadWeight(RegTensor<wType> &weightF8Vreg0, RegTensor<wType> &weightF8Vreg1,
                                                    LocalAddressParam<xType, wType> &localAddressParam, uint16_t nIdx)
{
    // UNPK_B8 表示按照如下形式载入:
    // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
    // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    Reg::DataCopy<wType, Reg::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg0, localAddressParam.weightLowBitPhyAddr0 + nIdx * ubMte2InnerSize);
    Reg::DataCopy<wType, Reg::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg1, localAddressParam.weightLowBitPhyAddr1 + nIdx * ubMte2InnerSize);
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize, Mc2QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdNkVf(LocalAddressParam<xType, wType> &localAddressParam,
                                          const CalculateParam<xType> &calculateParam)
{
    RegTensor<xType> antiQuantScaleVreg, antiQuantOffsetVreg;
    RegTensor<wType> weightF8Vreg0, weightF8Vreg1;
    RegTensor<xType> weightF16Vreg0, weightF16Vreg1, weightF16Vreg2, weightF16Vreg3;
    RegTensor<float> weightF32Vreg0, weightF32Vreg1, weightF32Vreg2, weightF32Vreg3;
    MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < calculateParam.ubLoop; ubLoopNIdx++) {
        AntiQuantFP8NdNkVfLoadScaleOffset<xType, wType, hasAntiQuantOffset, antiQuantType>(
            antiQuantScaleVreg, antiQuantOffsetVreg, localAddressParam, calculateParam, ubLoopNIdx);
        AntiQuantFP8NdNkVfLoadWeight<xType, wType, hasAntiQuantOffset, ubMte2InnerSize>(weightF8Vreg0, weightF8Vreg1,
                                                                                        localAddressParam, ubLoopNIdx);
        // 奇数、偶数位置分散到2个fp32寄存器存储
        Reg::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
        Reg::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
        Reg::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
        Reg::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);

        Reg::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
        Reg::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
        Reg::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
        Reg::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);

        Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg2,
                                                       (RegTensor<uint16_t> &)weightF16Vreg2,
                                                       (RegTensor<uint16_t> &)weightF16Vreg3, maskAll);
        Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg0,
                                                       (RegTensor<uint16_t> &)weightF16Vreg0,
                                                       (RegTensor<uint16_t> &)weightF16Vreg1, maskAll);

        if constexpr (hasAntiQuantOffset) {
            if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
                Reg::Adds(weightF16Vreg0, weightF16Vreg0, calculateParam.offsetValue, maskAll);
                Reg::Adds(weightF16Vreg2, weightF16Vreg2, calculateParam.offsetValue, maskAll);
            } else {
                Reg::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
                Reg::Add(weightF16Vreg2, weightF16Vreg2, antiQuantOffsetVreg, maskAll);
            }
        }

        Reg::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
        Reg::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg, maskAll);

        Reg::DataCopy<xType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
        Reg::DataCopy<xType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            localAddressParam.weightF16PhyAddr1, weightF16Vreg2, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset>
__aicore__ inline void AntiQuantFP8NdKnVfLoadOffset(RegTensor<xType> &antiQuantOffsetVreg0,
                                                    RegTensor<xType> &antiQuantOffsetVreg1,
                                                    LocalAddressParam<xType, wType> &localAddressParam)
{
    if constexpr (hasAntiQuantOffset) {
        Reg::DataCopy<xType, Reg::LoadDist::DIST_NORM>(antiQuantOffsetVreg0,
                                                       localAddressParam.antiQuantOffsetBasePhyAddr);
        Reg::DataCopy<xType, Reg::LoadDist::DIST_NORM>(antiQuantOffsetVreg1,
                                                       localAddressParam.antiQuantOffsetBasePhyAddr1);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, Mc2QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdKnVfLoadScale(RegTensor<xType> &antiQuantScaleVreg0,
                                                   RegTensor<xType> &antiQuantScaleVreg1,
                                                   LocalAddressParam<xType, wType> &localAddressParam,
                                                   const CalculateParam<xType> &calculateParam)
{
    if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
        Reg::Duplicate(antiQuantScaleVreg0, calculateParam.scaleValue);
        Reg::Duplicate(antiQuantScaleVreg1, calculateParam.scaleValue);
    } else {
        Reg::DataCopy<xType, Reg::LoadDist::DIST_NORM>(antiQuantScaleVreg0,
                                                       localAddressParam.antiQuantScaleBasePhyAddr);
        Reg::DataCopy<xType, Reg::LoadDist::DIST_NORM>(antiQuantScaleVreg1,
                                                       localAddressParam.antiQuantScaleBasePhyAddr1);
    }
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize>
__aicore__ inline void AntiQuantFP8NdKnVfLoadWeight(RegTensor<wType> &weightF8Vreg0, RegTensor<wType> &weightF8Vreg1,
                                                    LocalAddressParam<xType, wType> &localAddressParam, uint16_t kIdx)
{
    // UNPK_B8 表示按照如下形式载入:
    // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
    // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    Reg::DataCopy<wType, Reg::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg0, localAddressParam.weightLowBitPhyAddr0 + kIdx * ubMte2InnerSize);
    Reg::DataCopy<wType, Reg::LoadDist::DIST_UNPACK_B8>(
        weightF8Vreg1, localAddressParam.weightLowBitPhyAddr1 + kIdx * ubMte2InnerSize);
}

template <typename xType, typename wType>
__aicore__ inline void AntiQuantFP8NdKnVfStoreWeight(RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg2,
                                                     LocalAddressParam<xType, wType> &localAddressParam, MaskReg &mask)
{
    Reg::DataCopy<xType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
        localAddressParam.weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, mask);
    Reg::DataCopy<xType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
        localAddressParam.weightF16PhyAddr1, weightF16Vreg2, WEIGHT_F16_UB_NZ_STRIDE, 1, mask);
}

template <typename xType, typename wType, bool hasAntiQuantOffset, uint32_t ubMte2InnerSize, Mc2QuantType antiQuantType>
__aicore__ inline void AntiQuantFP8NdKnVf(LocalAddressParam<xType, wType> &localAddressParam,
                                          const CalculateParam<xType> &calculateParam)
{
    RegTensor<xType> antiQuantScaleVreg0, antiQuantScaleVreg1, antiQuantOffsetVreg0, antiQuantOffsetVreg1;
    RegTensor<wType> weightF8Vreg0, weightF8Vreg1;
    RegTensor<xType> weightF16Vreg0, weightF16Vreg1, weightF16Vreg2, weightF16Vreg3;
    RegTensor<float> weightF32Vreg0, weightF32Vreg1, weightF32Vreg2, weightF32Vreg3;
    MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    AntiQuantFP8NdKnVfLoadOffset<xType, wType, hasAntiQuantOffset>(antiQuantOffsetVreg0, antiQuantOffsetVreg1,
                                                                   localAddressParam);
    AntiQuantFP8NdKnVfLoadScale<xType, wType, hasAntiQuantOffset, antiQuantType>(
        antiQuantScaleVreg0, antiQuantScaleVreg1, localAddressParam, calculateParam);
    for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < calculateParam.ubLoop; ubLoopKIdx++) {
        AntiQuantFP8NdKnVfLoadWeight<xType, wType, hasAntiQuantOffset, ubMte2InnerSize>(weightF8Vreg0, weightF8Vreg1,
                                                                                        localAddressParam, ubLoopKIdx);
        Reg::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg0, weightF8Vreg0, maskAll);
        Reg::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg1, weightF8Vreg0, maskAll);
        Reg::Cast<float, wType, FP8_TO_FP32_TRAIT_0>(weightF32Vreg2, weightF8Vreg1, maskAll);
        Reg::Cast<float, wType, FP8_TO_FP32_TRAIT_2>(weightF32Vreg3, weightF8Vreg1, maskAll);
        Reg::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg0, weightF32Vreg0, maskAll);
        Reg::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg1, weightF32Vreg1, maskAll);
        Reg::Cast<xType, float, FP32_TO_F16_ODD>(weightF16Vreg2, weightF32Vreg2, maskAll);
        Reg::Cast<xType, float, FP32_TO_F16_EVEN>(weightF16Vreg3, weightF32Vreg3, maskAll);
        Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg0,
                                                       (RegTensor<uint16_t> &)weightF16Vreg0,
                                                       (RegTensor<uint16_t> &)weightF16Vreg1, maskAll);
        Reg::Or<uint16_t, Reg::MaskMergeMode::ZEROING>((RegTensor<uint16_t> &)weightF16Vreg2,
                                                       (RegTensor<uint16_t> &)weightF16Vreg2,
                                                       (RegTensor<uint16_t> &)weightF16Vreg3, maskAll);
        if constexpr (hasAntiQuantOffset) {
            if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
                Reg::Adds(weightF16Vreg0, weightF16Vreg0, calculateParam.offsetValue, maskAll);
                Reg::Adds(weightF16Vreg2, weightF16Vreg2, calculateParam.offsetValue, maskAll);
            } else {
                Reg::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg0, maskAll);
                Reg::Add(weightF16Vreg2, weightF16Vreg2, antiQuantOffsetVreg1, maskAll);
            }
        }
        Reg::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg0, maskAll);
        Reg::Mul(weightF16Vreg2, weightF16Vreg2, antiQuantScaleVreg1, maskAll);
        AntiQuantFP8NdKnVfStoreWeight<xType, wType>(weightF16Vreg0, weightF16Vreg2, localAddressParam, maskAll);
    }
}

template <typename xType, typename wType>
__aicore__ inline void CastS8RegTensorToF16RegTensor(RegTensor<xType> &weightF16Vreg, RegTensor<wType> &weightS8Vreg,
                                                     RegTensor<half> &weightFp16Vreg, Reg::MaskReg &maskAll)
{
    // PART_EVEN 表示按照如下形式处理做cast：
    // Vn 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
    // Vd 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 .....
    if constexpr (!IsSameType<typename Reg::TypeGet<xType>::T, vector_f16>::value) {
        Reg::Cast<half, wType, S8_TO_FP16_TRAIT_ODD>(weightFp16Vreg, weightS8Vreg, maskAll);
        Reg::Cast<xType, half, FP16_TO_BF16_TRAIT>(weightF16Vreg, weightFp16Vreg, maskAll);
    } else {
        Reg::Cast<xType, wType, S8_TO_FP16_TRAIT_ODD>(weightF16Vreg, weightS8Vreg, maskAll);
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig>
__aicore__ inline void AddMulWeightF16RegTensorNdKn(RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg1,
                                                    RegTensor<xType> &antiQuantScaleVreg,
                                                    RegTensor<xType> &antiQuantOffsetVreg,
                                                    RegTensor<xType> &antiQuantScaleVreg1,
                                                    RegTensor<xType> &antiQuantOffsetVreg1, Reg::MaskReg &maskAll,
                                                    const xType &offsetValue)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        if constexpr (wqmmConfig.antiQuantType == Mc2QuantType::PER_TENSOR) {
            Reg::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue, maskAll);
            Reg::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue, maskAll);
        } else {
            Reg::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
            Reg::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg1, maskAll);
        }
    }

    Reg::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
    Reg::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg1, maskAll);
}

template <typename xType, typename wType>
__aicore__ inline void WeightF16NdRegTensorToNzUb(__local_mem__ xType *&weightF16PhyAddr0,
                                                  __local_mem__ xType *&weightF16PhyAddr1,
                                                  RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg1,
                                                  Reg::MaskReg &maskAll)
{
    Reg::DataCopy<xType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
        weightF16PhyAddr0, weightF16Vreg0, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
    Reg::DataCopy<xType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
        weightF16PhyAddr1, weightF16Vreg1, WEIGHT_F16_UB_NZ_STRIDE, 1, maskAll);
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig>
__aicore__ inline void AddMulWeightF16RegTensorNdNk(RegTensor<xType> &weightF16Vreg0, RegTensor<xType> &weightF16Vreg1,
                                                    RegTensor<xType> &antiQuantScaleVreg,
                                                    RegTensor<xType> &antiQuantOffsetVreg, Reg::MaskReg &maskAll,
                                                    const xType &scaleValue, const xType &offsetValue)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        if constexpr (wqmmConfig.antiQuantType == Mc2QuantType::PER_TENSOR) {
            Reg::Adds(weightF16Vreg0, weightF16Vreg0, offsetValue, maskAll);
            Reg::Adds(weightF16Vreg1, weightF16Vreg1, offsetValue, maskAll);
        } else {
            Reg::Add(weightF16Vreg0, weightF16Vreg0, antiQuantOffsetVreg, maskAll);
            Reg::Add(weightF16Vreg1, weightF16Vreg1, antiQuantOffsetVreg, maskAll);
        }
    }

    Reg::Mul(weightF16Vreg0, weightF16Vreg0, antiQuantScaleVreg, maskAll);
    Reg::Mul(weightF16Vreg1, weightF16Vreg1, antiQuantScaleVreg, maskAll);
}

template <typename xType, const WqmmConfig &wqmmConfig>
__aicore__ inline void NdNkLoadScaleOffset(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                           __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                           RegTensor<xType> &antiQuantScaleVreg, RegTensor<xType> &antiQuantOffsetVreg,
                                           xType scaleValue, uint64_t ubLoopNIdx)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        Reg::DataCopy<xType, Reg::LoadDist::DIST_BRC_B16>(antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr + ubLoopNIdx);
    }
    if constexpr (wqmmConfig.antiQuantType == Mc2QuantType::PER_TENSOR) {
        Reg::Duplicate(antiQuantScaleVreg, scaleValue);
    } else {
        Reg::DataCopy<xType, Reg::LoadDist::DIST_BRC_B16>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr + ubLoopNIdx);
    }
}

template <typename xType, const WqmmConfig &wqmmConfig>
__aicore__ inline void NdKnLoadScaleOffset(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                           __local_mem__ xType *antiQuantScaleBasePhyAddr1,
                                           __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                           __local_mem__ xType *antiQuantOffsetBasePhyAddr1,
                                           RegTensor<xType> &antiQuantScaleVreg, RegTensor<xType> &antiQuantScaleVreg1,
                                           RegTensor<xType> &antiQuantOffsetVreg,
                                           RegTensor<xType> &antiQuantOffsetVreg1, xType scaleValue)
{
    if constexpr (wqmmConfig.hasAntiQuantOffset) {
        Reg::DataCopy<xType, Reg::LoadDist::DIST_NORM>(antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr);
        Reg::DataCopy<xType, Reg::LoadDist::DIST_NORM>(antiQuantOffsetVreg1, antiQuantOffsetBasePhyAddr1);
    }
    if constexpr (wqmmConfig.antiQuantType == Mc2QuantType::PER_TENSOR) {
        Reg::Duplicate(antiQuantScaleVreg, scaleValue);
        Reg::Duplicate(antiQuantScaleVreg1, scaleValue);
    } else {
        Reg::DataCopy<xType, Reg::LoadDist::DIST_NORM>(antiQuantScaleVreg, antiQuantScaleBasePhyAddr);
        Reg::DataCopy<xType, Reg::LoadDist::DIST_NORM>(antiQuantScaleVreg1, antiQuantScaleBasePhyAddr1);
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantB8CommonNdKn(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                             __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                             __local_mem__ wType *weightLowBitPhyAddr0,
                                             __local_mem__ xType *weightF16PhyAddr0, xType scaleValue,
                                             xType offsetValue, uint64_t ubLoopK)
{
    __local_mem__ xType *antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<xType> antiQuantScaleVreg1;
        RegTensor<xType> antiQuantOffsetVreg1;
        RegTensor<half> weightFp16Vreg0;
        RegTensor<half> weightFp16Vreg1;
        RegTensor<wType> weightS8Vreg0;
        RegTensor<wType> weightS8Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

        NdKnLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantScaleBasePhyAddr1,
                                               antiQuantOffsetBasePhyAddr, antiQuantOffsetBasePhyAddr1,
                                               antiQuantScaleVreg, antiQuantScaleVreg1, antiQuantOffsetVreg,
                                               antiQuantOffsetVreg1, scaleValue);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < static_cast<uint16_t>(ubLoopK); ubLoopKIdx++) {
            // UNPK_B8 表示按照如下形式载入:
            // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
            // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            Reg::DataCopy<wType, Reg::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg0, weightLowBitPhyAddr0 + ubLoopKIdx * vecConfig.ubMte2InnerSize);
            Reg::DataCopy<wType, Reg::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg1, weightLowBitPhyAddr1 + ubLoopKIdx * vecConfig.ubMte2InnerSize);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg0, weightS8Vreg0, weightFp16Vreg0, maskAll);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg1, weightS8Vreg1, weightFp16Vreg1, maskAll);
            AddMulWeightF16RegTensorNdKn<xType, wType, wqmmConfig>(weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg,
                                                                   antiQuantOffsetVreg, antiQuantScaleVreg1,
                                                                   antiQuantOffsetVreg1, maskAll, offsetValue);
            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantB8CommonNdNk(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                             __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                             __local_mem__ wType *weightLowBitPhyAddr0,
                                             __local_mem__ xType *weightF16PhyAddr0, xType scaleValue,
                                             xType offsetValue, uint64_t ubLoopN)
{
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 1);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);

    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<wType> weightS8Vreg0;
        RegTensor<wType> weightS8Vreg1;
        RegTensor<half> weightFp16Vreg0;
        RegTensor<half> weightFp16Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < static_cast<uint16_t>(ubLoopN); ubLoopNIdx++) {
            NdNkLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
                                                   antiQuantScaleVreg, antiQuantOffsetVreg, scaleValue, ubLoopNIdx);
            // UNPK_B8 表示按照如下形式载入:
            // Vn 1 2 3 4 5 6 7 8 9 a b c d e f g .....
            // Vd 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 .....
            Reg::DataCopy<wType, Reg::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg0, weightLowBitPhyAddr0 + ubLoopNIdx * vecConfig.ubMte2InnerSize);
            Reg::DataCopy<wType, Reg::LoadDist::DIST_UNPACK_B8>(
                weightS8Vreg1, weightLowBitPhyAddr1 + ubLoopNIdx * vecConfig.ubMte2InnerSize);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg0, weightS8Vreg0, weightFp16Vreg0, maskAll);
            CastS8RegTensorToF16RegTensor<xType, wType>(weightF16Vreg1, weightS8Vreg1, weightFp16Vreg1, maskAll);
            AddMulWeightF16RegTensorNdNk<xType, wType, wqmmConfig>(weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg,
                                                                   antiQuantOffsetVreg, maskAll, scaleValue,
                                                                   offsetValue);
            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdNk(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                         __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                         __local_mem__ wType *weightLowBitPhyAddr0,
                                         __local_mem__ xType *weightF16PhyAddr0, xType scaleValue, xType offsetValue,
                                         uint64_t ubLoopN)
{
    // int4每次处理128个数即为64B, 256>>2=64
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg;
        RegTensor<xType> antiQuantOffsetVreg;
        RegTensor<int4x2_t> weightS4Vreg0;
        RegTensor<int4x2_t> weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0;
        RegTensor<xType> weightF16Vreg1;

        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        static constexpr Reg::CastTrait castS4ToF16Trait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                            Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < static_cast<uint16_t>(ubLoopN); ubLoopNIdx++) {
            NdNkLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantOffsetBasePhyAddr,
                                                   antiQuantScaleVreg, antiQuantOffsetVreg, scaleValue, ubLoopNIdx);
            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x
            Reg::DataCopy<int4x2_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg0,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

            Reg::DataCopy<int4x2_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg1,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + ubLoopNIdx * (vecConfig.ubMte2InnerSize >> 1)));

            // PART_P0 表示按照如下形式处理做cast：
            // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
            Reg::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
            Reg::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
            AddMulWeightF16RegTensorNdNk<xType, wType, wqmmConfig>(weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg,
                                                                   antiQuantOffsetVreg, maskAll, scaleValue,
                                                                   offsetValue);
            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

template <typename xType, typename wType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig>
__aicore__ inline void AntiQuantInt4NdKn(__local_mem__ xType *antiQuantScaleBasePhyAddr,
                                         __local_mem__ xType *antiQuantOffsetBasePhyAddr,
                                         __local_mem__ wType *weightLowBitPhyAddr0,
                                         __local_mem__ xType *weightF16PhyAddr0, xType scaleValue, xType offsetValue,
                                         uint64_t ubLoopK)
{
    __local_mem__ xType *antiQuantScaleBasePhyAddr1 = antiQuantScaleBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ xType *antiQuantOffsetBasePhyAddr1 = antiQuantOffsetBasePhyAddr + VEC_MAX_ELEM_B16;
    __local_mem__ wType *weightLowBitPhyAddr1 = weightLowBitPhyAddr0 + (VECTOR_REG_WIDTH >> 2);
    __local_mem__ xType *weightF16PhyAddr1 = weightF16PhyAddr0 + WEIGHT_F16_UB_NZ_STRIDE * (VECTOR_REG_WIDTH >> 1);
    __VEC_SCOPE__
    {
        RegTensor<xType> antiQuantScaleVreg, antiQuantOffsetVreg, antiQuantScaleVreg1, antiQuantOffsetVreg1;
        RegTensor<int4x2_t> weightS4Vreg0, weightS4Vreg1;
        RegTensor<xType> weightF16Vreg0, weightF16Vreg1;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        static constexpr Reg::CastTrait castS4ToF16Trait = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                            Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        NdKnLoadScaleOffset<xType, wqmmConfig>(antiQuantScaleBasePhyAddr, antiQuantScaleBasePhyAddr1,
                                               antiQuantOffsetBasePhyAddr, antiQuantOffsetBasePhyAddr1,
                                               antiQuantScaleVreg, antiQuantScaleVreg1, antiQuantOffsetVreg,
                                               antiQuantOffsetVreg1, scaleValue);

        for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < static_cast<uint16_t>(ubLoopK); ubLoopKIdx++) {
            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x
            Reg::DataCopy<int4x2_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg0,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr0 + ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1)));

            Reg::DataCopy<int4x2_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                weightS4Vreg1,
                (__local_mem__ int4x2_t *)(weightLowBitPhyAddr1 + ubLoopKIdx * (vecConfig.ubMte2InnerSize >> 1)));

            // PART_P0 表示按照如下形式处理做cast：
            // Vn 1 2 0 0 0 0 0 0 3 4 0 0 0 0 0 0
            // Vd 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
            Reg::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg0, weightS4Vreg0, maskAll);
            Reg::Cast<xType, int4x2_t, castS4ToF16Trait>(weightF16Vreg1, weightS4Vreg1, maskAll);
            AddMulWeightF16RegTensorNdKn<xType, wType, wqmmConfig>(weightF16Vreg0, weightF16Vreg1, antiQuantScaleVreg,
                                                                   antiQuantOffsetVreg, antiQuantScaleVreg1,
                                                                   antiQuantOffsetVreg1, maskAll, offsetValue);

            WeightF16NdRegTensorToNzUb<xType, wType>(weightF16PhyAddr0, weightF16PhyAddr1, weightF16Vreg0,
                                                     weightF16Vreg1, maskAll);
        }
    }
}

} // namespace Mc2WeightQuantBatchMatmulV2::Arch35
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_VF_ND_H
