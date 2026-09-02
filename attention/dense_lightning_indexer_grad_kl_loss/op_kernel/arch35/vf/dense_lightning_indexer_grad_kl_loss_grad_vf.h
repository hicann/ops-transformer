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
 * \file dense_lightning_indexer_grad_kl_loss_grad_vf.h
 * \brief DAV_3510 RegBase gradient elementwise helper.
 */
#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_GRAD_VF_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_GRAD_VF_H

#include "kernel_tensor.h"

namespace DenseDliGradKLLossArch35 {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr static CastTrait CAST_GRAD_FP32_TO_B16 = {
    RegLayout::ZERO,
    SatMode::SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_ROUND,
};

constexpr uint32_t GRAD_FP32_REG_ELEMENTS = 64;

__simd_vf__ inline void ComputeDwElementwiseVF(__ubuf__ float *dwElementUb, __ubuf__ float *deltaUb,
                                               __ubuf__ float *reluUb, uint32_t count)
{
    RegTensor<float> deltaReg;
    RegTensor<float> reluReg;
    RegTensor<float> dwElementReg;

    for (uint32_t offset = 0; offset < count; offset += GRAD_FP32_REG_ELEMENTS) {
        uint32_t validCount = count - offset < GRAD_FP32_REG_ELEMENTS ? count - offset : GRAD_FP32_REG_ELEMENTS;
        MaskReg tailMask = UpdateMask<float>(validCount);
        LoadAlign(deltaReg, deltaUb + offset);
        LoadAlign(reluReg, reluUb + offset);
        Mul(dwElementReg, deltaReg, reluReg, tailMask);
        StoreAlign(dwElementUb + offset, dwElementReg, tailMask);
    }
}

__aicore__ inline void ComputeDwElementwise(const LocalTensor<float> &dwElementTensor,
                                            const LocalTensor<float> &deltaTensor, const LocalTensor<float> &reluTensor,
                                            uint32_t count)
{
    __ubuf__ float *dwElementUb = reinterpret_cast<__ubuf__ float *>(dwElementTensor.GetPhyAddr());
    __ubuf__ float *deltaUb = reinterpret_cast<__ubuf__ float *>(deltaTensor.GetPhyAddr());
    __ubuf__ float *reluUb = reinterpret_cast<__ubuf__ float *>(reluTensor.GetPhyAddr());
    ComputeDwElementwiseVF(dwElementUb, deltaUb, reluUb, count);
}

template <typename OUT_T>
__simd_vf__ inline void ComputeReluGradElementwiseVF(__ubuf__ OUT_T *reluGradUb, __ubuf__ float *deltaUb,
                                                     __ubuf__ float *reluUb, __ubuf__ float *weightUb, uint32_t count)
{
    RegTensor<float> deltaReg;
    RegTensor<float> reluReg;
    RegTensor<float> weightReg;
    RegTensor<float> weightedReg;
    RegTensor<float> zeroReg;
    RegTensor<float> reluGradReg;
    RegTensor<OUT_T> reluGradOutReg;
    MaskReg reluMask;

    Duplicate(zeroReg, 0.0f);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weightReg, weightUb);
    for (uint32_t offset = 0; offset < count; offset += GRAD_FP32_REG_ELEMENTS) {
        uint32_t validCount = count - offset < GRAD_FP32_REG_ELEMENTS ? count - offset : GRAD_FP32_REG_ELEMENTS;
        MaskReg tailMask = UpdateMask<float>(validCount);
        LoadAlign(deltaReg, deltaUb + offset);
        LoadAlign(reluReg, reluUb + offset);
        Mul(weightedReg, deltaReg, weightReg, tailMask);
        Compare<float, CMPMODE::GT>(reluMask, reluReg, zeroReg, tailMask);
        Select(reluGradReg, weightedReg, zeroReg, reluMask);
        Cast<OUT_T, float, CAST_GRAD_FP32_TO_B16>(reluGradOutReg, reluGradReg, tailMask);
        StoreAlign<OUT_T, StoreDist::DIST_PACK_B32>(reluGradUb + offset, reluGradOutReg, tailMask);
    }
}

template <typename OUT_T>
__aicore__ inline void ComputeReluGradElementwise(const LocalTensor<OUT_T> &reluGradTensor,
                                                  const LocalTensor<float> &deltaTensor,
                                                  const LocalTensor<float> &reluTensor,
                                                  const LocalTensor<float> &weightTensor, uint32_t weightOffset,
                                                  uint32_t count)
{
    __ubuf__ OUT_T *reluGradUb = reinterpret_cast<__ubuf__ OUT_T *>(reluGradTensor.GetPhyAddr());
    __ubuf__ float *deltaUb = reinterpret_cast<__ubuf__ float *>(deltaTensor.GetPhyAddr());
    __ubuf__ float *reluUb = reinterpret_cast<__ubuf__ float *>(reluTensor.GetPhyAddr());
    __ubuf__ float *weightUb = reinterpret_cast<__ubuf__ float *>(weightTensor.GetPhyAddr()) + weightOffset;
    ComputeReluGradElementwiseVF<OUT_T>(reluGradUb, deltaUb, reluUb, weightUb, count);
}

template <typename OUT_T>
__simd_vf__ inline void ComputeGradElementwiseVF(__ubuf__ float *dwElementUb, __ubuf__ OUT_T *reluGradUb,
                                                 __ubuf__ float *deltaUb, __ubuf__ float *reluUb,
                                                 __ubuf__ float *weightUb, uint32_t count)
{
    RegTensor<float> deltaReg;
    RegTensor<float> reluReg;
    RegTensor<float> weightReg;
    RegTensor<float> dwElementReg;
    RegTensor<float> weightedReg;
    RegTensor<float> zeroReg;
    RegTensor<float> reluGradReg;
    RegTensor<OUT_T> reluGradOutReg;
    MaskReg reluMask;

    Duplicate(zeroReg, 0.0f);
    LoadAlign<float, LoadDist::DIST_BRC_B32>(weightReg, weightUb);
    for (uint32_t offset = 0; offset < count; offset += GRAD_FP32_REG_ELEMENTS) {
        uint32_t validCount = count - offset < GRAD_FP32_REG_ELEMENTS ? count - offset : GRAD_FP32_REG_ELEMENTS;
        MaskReg tailMask = UpdateMask<float>(validCount);
        LoadAlign(deltaReg, deltaUb + offset);
        LoadAlign(reluReg, reluUb + offset);

        Mul(dwElementReg, deltaReg, reluReg, tailMask);
        StoreAlign(dwElementUb + offset, dwElementReg, tailMask);

        Mul(weightedReg, deltaReg, weightReg, tailMask);
        Compare<float, CMPMODE::GT>(reluMask, reluReg, zeroReg, tailMask);
        Select(reluGradReg, weightedReg, zeroReg, reluMask);
        Cast<OUT_T, float, CAST_GRAD_FP32_TO_B16>(reluGradOutReg, reluGradReg, tailMask);
        StoreAlign<OUT_T, StoreDist::DIST_PACK_B32>(reluGradUb + offset, reluGradOutReg, tailMask);
    }
}

template <typename OUT_T>
__aicore__ inline void ComputeGradElementwise(const LocalTensor<float> &dwElementTensor,
                                              const LocalTensor<OUT_T> &reluGradTensor,
                                              const LocalTensor<float> &deltaTensor,
                                              const LocalTensor<float> &reluTensor,
                                              const LocalTensor<float> &weightTensor, uint32_t weightOffset,
                                              uint32_t count)
{
    __ubuf__ float *dwElementUb = reinterpret_cast<__ubuf__ float *>(dwElementTensor.GetPhyAddr());
    __ubuf__ OUT_T *reluGradUb = reinterpret_cast<__ubuf__ OUT_T *>(reluGradTensor.GetPhyAddr());
    __ubuf__ float *deltaUb = reinterpret_cast<__ubuf__ float *>(deltaTensor.GetPhyAddr());
    __ubuf__ float *reluUb = reinterpret_cast<__ubuf__ float *>(reluTensor.GetPhyAddr());
    __ubuf__ float *weightUb = reinterpret_cast<__ubuf__ float *>(weightTensor.GetPhyAddr()) + weightOffset;
    ComputeGradElementwiseVF<OUT_T>(dwElementUb, reluGradUb, deltaUb, reluUb, weightUb, count);
}
} // namespace DenseDliGradKLLossArch35

#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_GRAD_VF_H
