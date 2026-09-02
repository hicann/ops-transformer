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
 * \file vf_softmax_grad_front_cast.h
 */
#ifndef MY_SOFTMAX_GRAD_CAST_INTERFACE_H_
#define MY_SOFTMAX_GRAD_CAST_INTERFACE_H_
#include "kernel_tensor.h"
#include "vf_softmax_grad_front_cast_aligned256_f32.h"
#include "vf_softmax_grad_front_cast_aligned512_f32.h"
#include "vf_softmax_grad_front_cast_aligned768_f32.h"
#include "vf_softmax_grad_front_cast_aligned256_f16.h"
#include "vf_softmax_grad_front_cast_aligned512_f16.h"
#include "vf_softmax_grad_front_cast_aligned768_f16.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
/* **************************************************************************************************

SoftmaxGradFrontCast *
************************************************************************************************* /

@INGROUP SoftmaxGradFrontCast
brief compute :sum = reducesum(cast(grad) * cast(x))
param [out] dstTensor output LocalTensor
param [in] gradTensor input grad LocalTensor
param [in] srcTensor input src LocalTensor
*/
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCast(const LocalTensor<T> &dstTensor, const LocalTensor<T1> &gradTensor,
                                              const LocalTensor<T1> &srcTensor, uint32_t srcM, uint32_t realN = srcN)
{
    if constexpr (IsSameType<T1, float>::value) {
        if constexpr (srcN <= 256) {
            MySoftmaxGradFrontCastAligned256F32<T1, T, srcN, HEAD_DIM_ALIGN>(dstTensor, gradTensor, srcTensor, srcM,
                                                                             realN);
        } else if constexpr (srcN <= 512) {
            MySoftmaxGradFrontCastAligned512F32<T1, T, srcN, HEAD_DIM_ALIGN>(dstTensor, gradTensor, srcTensor, srcM,
                                                                             realN);
        } else if constexpr (srcN <= 768) {
            MySoftmaxGradFrontCastAligned768F32<T1, T, srcN, HEAD_DIM_ALIGN>(dstTensor, gradTensor, srcTensor, srcM,
                                                                             realN);
        }
    } else {
        if constexpr (srcN <= 256) {
            MySoftmaxGradFrontCastAligned256F16<T1, T, srcN, HEAD_DIM_ALIGN>(dstTensor, gradTensor, srcTensor, srcM,
                                                                             realN);
        } else if constexpr (srcN <= 512) {
            MySoftmaxGradFrontCastAligned512F16<T1, T, srcN, HEAD_DIM_ALIGN>(dstTensor, gradTensor, srcTensor, srcM,
                                                                             realN);
        } else if constexpr (srcN <= 768) {
            MySoftmaxGradFrontCastAligned768F16<T1, T, srcN, HEAD_DIM_ALIGN>(dstTensor, gradTensor, srcTensor, srcM,
                                                                             realN);
        }
    }
}
#else
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCast(const LocalTensor<T> &dstTensor, const LocalTensor<T1> &gradTensor,
                                              const LocalTensor<T1> &srcTensor, uint32_t srcM, uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // MY_SOFTMAX_GRAD_CAST_INTERFACE_H
