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
 * \file dense_lightning_indexer_grad_kl_loss_cast_vf.h
 * \brief DAV_3510 RegBase output cast helper.
 */
#ifndef DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_CAST_VF_H
#define DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_CAST_VF_H

#include "kernel_tensor.h"

namespace DenseDliGradKLLossArch35 {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr static CastTrait CAST_FP32_TO_B16 = {
    RegLayout::ZERO,
    SatMode::SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_ROUND,
};

constexpr uint32_t FP32_REG_ELEMENTS = 64;

template <typename OUT_T>
__simd_vf__ inline void CastFp32ToOutVF(__ubuf__ OUT_T *dstUb, __ubuf__ float *srcUb, uint32_t count)
{
    RegTensor<float> srcReg;
    RegTensor<OUT_T> dstReg;

    for (uint32_t offset = 0; offset < count; offset += FP32_REG_ELEMENTS) {
        uint32_t validCount = count - offset < FP32_REG_ELEMENTS ? count - offset : FP32_REG_ELEMENTS;
        MaskReg tailMask = UpdateMask<float>(validCount);
        LoadAlign(srcReg, srcUb + offset);
        if constexpr (IsSameType<OUT_T, float>::value) {
            StoreAlign<OUT_T, StoreDist::DIST_NORM_B32>(dstUb + offset, srcReg, tailMask);
        } else {
            Cast<OUT_T, float, CAST_FP32_TO_B16>(dstReg, srcReg, tailMask);
            StoreAlign<OUT_T, StoreDist::DIST_PACK_B32>(dstUb + offset, dstReg, tailMask);
        }
    }
}

template <typename OUT_T>
__aicore__ inline void CastFp32ToOut(const LocalTensor<OUT_T> &dstTensor, const LocalTensor<float> &srcTensor,
                                     uint32_t count)
{
    __ubuf__ OUT_T *dstUb = reinterpret_cast<__ubuf__ OUT_T *>(dstTensor.GetPhyAddr());
    __ubuf__ float *srcUb = reinterpret_cast<__ubuf__ float *>(srcTensor.GetPhyAddr());
    CastFp32ToOutVF<OUT_T>(dstUb, srcUb, count);
}
} // namespace DenseDliGradKLLossArch35

#endif // DENSE_LIGHTNING_INDEXER_GRAD_KL_LOSS_CAST_VF_H
