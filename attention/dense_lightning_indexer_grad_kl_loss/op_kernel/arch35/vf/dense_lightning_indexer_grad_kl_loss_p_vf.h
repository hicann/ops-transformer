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
 * \file dense_lightning_indexer_grad_kl_loss_p_vf.h
 * \brief DAV_3510 RegBase probability accumulation helper.
 */
#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_P_VF_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_P_VF_H

#include "kernel_tensor.h"

namespace DenseDliGradKLLossArch35 {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t P_FP32_REG_ELEMENTS = 64;

template <bool FIRST_HEAD>
__simd_vf__ inline void ComputePHeadVF(__ubuf__ float *dstUb, __ubuf__ float *scoreUb, __ubuf__ float *maxUb,
                                       __ubuf__ float *sumUb, float scale, uint32_t rowCount, uint32_t rowSize)
{
    RegTensor<float> scoreReg;
    RegTensor<float> scaledReg;
    RegTensor<float> maxReg;
    RegTensor<float> sumReg;
    RegTensor<float> subReg;
    RegTensor<float> probReg;
    RegTensor<float> accReg;

    for (uint32_t rowIdx = 0; rowIdx < rowCount; ++rowIdx) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(maxReg, maxUb + rowIdx);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(sumReg, sumUb + rowIdx);
        uint32_t rowOffset = rowIdx * rowSize;
        for (uint32_t columnOffset = 0; columnOffset < rowSize; columnOffset += P_FP32_REG_ELEMENTS) {
            uint32_t validCount =
                rowSize - columnOffset < P_FP32_REG_ELEMENTS ? rowSize - columnOffset : P_FP32_REG_ELEMENTS;
            MaskReg tailMask = UpdateMask<float>(validCount);
            uint32_t offset = rowOffset + columnOffset;
            LoadAlign(scoreReg, scoreUb + offset);
            Muls(scaledReg, scoreReg, scale, tailMask);
            Sub(subReg, scaledReg, maxReg, tailMask);
            Exp(probReg, subReg, tailMask);
            Div(probReg, probReg, sumReg, tailMask);
            if constexpr (!FIRST_HEAD) {
                LoadAlign(accReg, dstUb + offset);
                Add(probReg, accReg, probReg, tailMask);
            }
            StoreAlign(dstUb + offset, probReg, tailMask);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <bool FIRST_HEAD>
__aicore__ inline void ComputePHead(const LocalTensor<float> &dstTensor, const LocalTensor<float> &scoreTensor,
                                    const LocalTensor<float> &maxTensor, const LocalTensor<float> &sumTensor,
                                    float scale, uint32_t rowCount, uint32_t rowSize)
{
    __ubuf__ float *dstUb = reinterpret_cast<__ubuf__ float *>(dstTensor.GetPhyAddr());
    __ubuf__ float *scoreUb = reinterpret_cast<__ubuf__ float *>(scoreTensor.GetPhyAddr());
    __ubuf__ float *maxUb = reinterpret_cast<__ubuf__ float *>(maxTensor.GetPhyAddr());
    __ubuf__ float *sumUb = reinterpret_cast<__ubuf__ float *>(sumTensor.GetPhyAddr());
    ComputePHeadVF<FIRST_HEAD>(dstUb, scoreUb, maxUb, sumUb, scale, rowCount, rowSize);
}
} // namespace DenseDliGradKLLossArch35

#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_P_VF_H
