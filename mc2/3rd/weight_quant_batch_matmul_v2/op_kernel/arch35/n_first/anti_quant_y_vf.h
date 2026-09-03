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
 * \file anti_quant_y_vf.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_ANTI_QUANT_Y_VF_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_ANTI_QUANT_Y_VF_H

#include "basic_block_config.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Reg = AscendC::Reg;
using AscendC::Reg::RegTensor;

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

template <typename yType>
struct LocalAddressYParam {
    __local_mem__ int32_t *yOriginPhyAddr;
    __local_mem__ float *cScalePhyAddr;
    __local_mem__ float *kScalePhyAddr;
    __local_mem__ float *biasPhyAddr;
    __local_mem__ yType *yPhyAddr;
};

static constexpr Reg::CastTrait C32_TO_FP32_TRAIT = {Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

static constexpr Reg::CastTrait FP32_TO_F16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                               AscendC::RoundMode::CAST_RINT};

template <typename yType, bool hasBias>
__aicore__ inline void AntiQuantYB32(LocalAddressYParam<yType> &localAddressParam, uint64_t nRealL0Size,
                                     uint64_t nRealVRegSize, uint16_t ubLoopN, uint16_t ubLoopM)
{
    __VEC_SCOPE__
    {
        RegTensor<int32_t> yOriginVreg;
        RegTensor<float> yVreg;
        RegTensor<float> antiQuantCScaleVreg;
        RegTensor<float> antiQuantKScaleVreg;
        RegTensor<float> biasVreg;
        RegTensor<yType> yF16Vreg;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();

        uint32_t nRealL0Temp = nRealL0Size;
        for (uint16_t nId = 0; nId < ubLoopN; nId++) {
            if constexpr (hasBias) {
                Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(biasVreg,
                                                               localAddressParam.biasPhyAddr + nId * nRealVRegSize);
            }
            Reg::DataCopy<float, Reg::LoadDist::DIST_NORM>(antiQuantCScaleVreg,
                                                           localAddressParam.cScalePhyAddr + nId * nRealVRegSize);

            Reg::MaskReg yResultMask = Reg::UpdateMask<float>(nRealL0Temp);
            for (uint16_t mId = 0; mId < ubLoopM; mId++) {
                uint64_t yOffset = nId * nRealVRegSize + mId * nRealL0Size;
                uint64_t ubYOffset = (nId * nRealVRegSize >> 1) + mId * nRealL0Size;
                Reg::DataCopy<int32_t, Reg::LoadDist::DIST_NORM>(yOriginVreg,
                                                                 localAddressParam.yOriginPhyAddr + yOffset);
                Reg::DataCopy<float, Reg::LoadDist::DIST_BRC_B32>(antiQuantKScaleVreg,
                                                                  localAddressParam.kScalePhyAddr + mId);

                Reg::Cast<float, int32_t, C32_TO_FP32_TRAIT>(yVreg, yOriginVreg, maskAll);
                Reg::Mul(yVreg, yVreg, antiQuantCScaleVreg, maskAll);
                Reg::Mul(yVreg, yVreg, antiQuantKScaleVreg, maskAll);
                if constexpr (hasBias) {
                    Reg::Add(yVreg, yVreg, biasVreg, maskAll);
                }

                Reg::Cast<yType, float, FP32_TO_F16>(yF16Vreg, yVreg, maskAll);

                Reg::DataCopy<yType, Reg::StoreDist::DIST_PACK_B32>(localAddressParam.yPhyAddr + ubYOffset * 2,
                                                                    yF16Vreg, yResultMask);
            }
        }
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_ANTI_QUANT_Y_VF_H
