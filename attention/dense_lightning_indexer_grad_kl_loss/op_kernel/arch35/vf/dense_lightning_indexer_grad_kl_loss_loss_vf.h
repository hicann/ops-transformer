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
 * \file dense_lightning_indexer_grad_kl_loss_loss_vf.h
 * \brief DAV_3510 RegBase elementwise loss helper.
 */
#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_LOSS_VF_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_LOSS_VF_H

#include "kernel_tensor.h"

namespace DenseDliGradKLLossArch35 {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t LOSS_FP32_REG_ELEMENTS = 64;
constexpr float LOSS_MIN_VALUE = 1e-8f;

__simd_vf__ inline void ComputeLossElementwiseVF(__ubuf__ float *dstUb, __ubuf__ float *pUb, __ubuf__ float *syUb,
                                                 uint32_t count)
{
    RegTensor<float> pReg;
    RegTensor<float> pClampedReg;
    RegTensor<float> syReg;
    RegTensor<float> syClampedReg;
    RegTensor<float> minReg;
    RegTensor<float> logPReg;
    RegTensor<float> logSyReg;
    RegTensor<float> lossReg;

    Duplicate(minReg, LOSS_MIN_VALUE);
    for (uint32_t offset = 0; offset < count; offset += LOSS_FP32_REG_ELEMENTS) {
        uint32_t validCount = count - offset < LOSS_FP32_REG_ELEMENTS ? count - offset : LOSS_FP32_REG_ELEMENTS;
        MaskReg tailMask = UpdateMask<float>(validCount);
        LoadAlign(pReg, pUb + offset);
        LoadAlign(syReg, syUb + offset);
        Max(pClampedReg, minReg, pReg, tailMask);
        Max(syClampedReg, minReg, syReg, tailMask);
        Log(logPReg, pClampedReg, tailMask);
        Log(logSyReg, syClampedReg, tailMask);
        Sub(lossReg, logPReg, logSyReg, tailMask);
        Mul(lossReg, lossReg, pReg, tailMask);
        StoreAlign(dstUb + offset, lossReg, tailMask);
    }
}

__aicore__ inline void ComputeLossElementwise(const LocalTensor<float> &dstTensor, const LocalTensor<float> &pTensor,
                                              const LocalTensor<float> &syTensor, uint32_t count)
{
    __ubuf__ float *dstUb = reinterpret_cast<__ubuf__ float *>(dstTensor.GetPhyAddr());
    __ubuf__ float *pUb = reinterpret_cast<__ubuf__ float *>(pTensor.GetPhyAddr());
    __ubuf__ float *syUb = reinterpret_cast<__ubuf__ float *>(syTensor.GetPhyAddr());
    ComputeLossElementwiseVF(dstUb, pUb, syUb, count);
}
} // namespace DenseDliGradKLLossArch35

#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_LOSS_VF_H
