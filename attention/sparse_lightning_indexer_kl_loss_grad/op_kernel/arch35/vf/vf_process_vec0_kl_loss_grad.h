/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file vf_process_vec0_kl_loss_grad.h
 * \brief
 */
#ifndef VF_PROCESS_VEC0_KL_LOSS_GRAD_H
#define VF_PROCESS_VEC0_KL_LOSS_GRAD_H

#include "kernel_tensor.h"

namespace AscendC {
using namespace MicroAPI;

static constexpr uint32_t FLOAT_REP_SIZE_128 = 128;

template <typename INPUT_T>
__simd_vf__ inline void Ub2UbNd2Nz128(__ubuf__ INPUT_T *dstUb, __ubuf__ INPUT_T *srcUb, uint32_t m, uint32_t n)
{
    RegTensor<INPUT_T> vreg_input_x1;
    uint32_t blockStride = m;
    uint32_t repeatStride = 1;
    uint32_t repeatSize = (uint32_t)128ULL;
    MaskReg preg16 = UpdateMask<uint16_t>(repeatSize);
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x1, srcUb + i * n);
        StoreAlign<INPUT_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ INPUT_T *&)dstUb), vreg_input_x1, blockStride, repeatStride, preg16);
    }
}

template <typename INPUT_T>
__simd_vf__ inline void Ub2UbNd2Nz512(__ubuf__ INPUT_T *dstUb, __ubuf__ INPUT_T *srcUb, uint32_t m, uint32_t n)
{
    RegTensor<INPUT_T> vreg_input_x1;
    RegTensor<INPUT_T> vreg_input_x2;
    RegTensor<INPUT_T> vreg_input_x3;
    RegTensor<INPUT_T> vreg_input_x4;
    uint32_t blockStride = m;
    uint32_t repeatStride = 1;
    uint32_t repeatSize = (uint32_t)128ULL;
    MaskReg preg16 = UpdateMask<uint16_t>(repeatSize);
    __ubuf__ INPUT_T *dstUb2 = dstUb + m * FLOAT_REP_SIZE_128;
    __ubuf__ INPUT_T *dstUb3 = dstUb + m * FLOAT_REP_SIZE_128 * 2;
    __ubuf__ INPUT_T *dstUb4 = dstUb + m * FLOAT_REP_SIZE_128 * 3;
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x1, srcUb + i * n);
        LoadAlign(vreg_input_x2, srcUb + i * n + FLOAT_REP_SIZE_128);
        LoadAlign(vreg_input_x3, srcUb + i * n + FLOAT_REP_SIZE_128 * 2);
        LoadAlign(vreg_input_x4, srcUb + i * n + FLOAT_REP_SIZE_128 * 3);
        StoreAlign<INPUT_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ INPUT_T *&)dstUb), vreg_input_x1, blockStride, repeatStride, preg16);
        StoreAlign<INPUT_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ INPUT_T *&)dstUb2), vreg_input_x2, blockStride, repeatStride, preg16);
        StoreAlign<INPUT_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ INPUT_T *&)dstUb3), vreg_input_x3, blockStride, repeatStride, preg16);
        StoreAlign<INPUT_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ INPUT_T *&)dstUb4), vreg_input_x4, blockStride, repeatStride, preg16);
    }
}

template <typename INPUT_T>
__aicore__ inline void Ub2UbNd2Nz(const LocalTensor<INPUT_T> &dstTensor, const LocalTensor<INPUT_T> &srcTensor,
                                  uint32_t m, uint32_t n)
{
    __ubuf__ INPUT_T *dstUb = (__ubuf__ INPUT_T *)dstTensor.GetPhyAddr();
    __ubuf__ INPUT_T *srcUb = (__ubuf__ INPUT_T *)srcTensor.GetPhyAddr();
    if (n == 512) {
        Ub2UbNd2Nz512<INPUT_T>(dstUb, srcUb, m, n);
    }
    if (n == 64 || n == 128) {
        Ub2UbNd2Nz128<INPUT_T>(dstUb, srcUb, m, n);
    }
}
} // namespace AscendC

#endif // VF_PROCESS_VEC0_KL_LOSS_GRAD_H
