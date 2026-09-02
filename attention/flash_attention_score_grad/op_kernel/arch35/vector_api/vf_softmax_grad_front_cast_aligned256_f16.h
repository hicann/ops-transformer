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
#ifndef MY_SOFTMAX_GRAD_CAST_ALIGNED256_F16_INTERFACE_H_
#define MY_SOFTMAX_GRAD_CAST_ALIGNED256_F16_INTERFACE_H_
#include "kernel_tensor.h"
#include "vf_common_utils.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
/* **************************************************************************************************

SoftmaxGradFrontCast *
************************************************************************************************* /

@INGROUP SoftmaxGradFrontCast
brief compute :sum = reducesum(cast(grad) * cast(x))
param [out] dstTensor output LocalTensor
param [in] gradTensor input grad LocalTensor
param [in] srcTensor input src LocalTensor
*/
template <typename T1>
__simd_vf__ inline void CastAligned256F16VF128(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               const uint32_t fullExeSize, uint32_t reduceSize, uint32_t srcM)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregMul;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregMul2;

    RegTensor<half> vregSrcHalf;
    RegTensor<half> vregGradHalf;
    RegTensor<bfloat16_t> vregSrcBf;
    RegTensor<bfloat16_t> vregGradBf;

    RegTensor<float> vregAdd;
    RegTensor<float> vregReduceSum;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<T1, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        if constexpr (IsSameType<T1, half>::value) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1 *&)srcLocalInt),
                                                                   fullExeSize);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf, ((__ubuf__ T1 *&)gradLocalInt),
                                                                   fullExeSize);
            Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradHalf, pregFullExeB16);
        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1 *&)srcLocalInt),
                                                                   fullExeSize);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1 *&)gradLocalInt),
                                                                   fullExeSize);
            Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradBf, pregFullExeB16);
        }
        Mul(vregMul, vregGrad, vregSrc, pregTailExe);
        Mul(vregMul2, vregGrad1, vregSrc1, pregTailExe);
        Add(vregAdd, vregMul, vregMul2, pregFullExe);
        ReduceSum(vregReduceSum, vregAdd, pregTailExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, uint32_t srcN>
__simd_vf__ inline void CastAligned256F16VF256(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               const uint32_t fullExeSize, uint32_t reduceSize,
                                               uint64_t srcLocalIntTail, uint64_t gradLocalIntTail, uint32_t srcM)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregMul;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregMul2;

    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregSrc1Tail;
    RegTensor<float> vregGrad1Tail;
    RegTensor<float> vregMul2Tail;

    RegTensor<half> vregSrcHalf;
    RegTensor<half> vregGradHalf;
    RegTensor<bfloat16_t> vregSrcBf;
    RegTensor<bfloat16_t> vregGradBf;

    RegTensor<half> vregSrcHalfTail;
    RegTensor<half> vregGradHalfTail;
    RegTensor<bfloat16_t> vregSrcBfTail;
    RegTensor<bfloat16_t> vregGradBfTail;

    RegTensor<float> vregAdd;
    RegTensor<float> vregAddTail;
    RegTensor<float> vregAddLast;
    RegTensor<float> vregReduceSum;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<T1, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        if constexpr (IsSameType<T1, half>::value) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1 *&)srcLocalInt), srcN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf, ((__ubuf__ T1 *&)gradLocalInt), srcN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalfTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                                   srcN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalfTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                                   srcN);
            Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcHalfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcHalfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradHalfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradHalfTail, pregFullExeB16);
        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1 *&)srcLocalInt), srcN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1 *&)gradLocalInt), srcN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBfTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                                   srcN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBfTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                                   srcN);
            Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcBfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradBfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcBfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradBfTail, pregFullExeB16);
        }
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul2, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Mul(vregMul2Tail, vregGrad1Tail, vregSrc1Tail, pregTailExe);
        Add(vregAdd, vregMul, vregMul2, pregFullExe);
        Add(vregAddTail, vregMulTail, vregMul2Tail, pregTailExe);

        Add(vregAddLast, vregAdd, vregAddTail, pregFullExe);

        ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned256F16(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T1> &gradTensor,
                                                           const LocalTensor<T1> &srcTensor, uint32_t srcM,
                                                           uint32_t realN = srcN)
{
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t gradLocalInt = gradTensor.GetPhyAddr();

    if constexpr (srcN <= 128) {
        const uint32_t fullExeSize = srcN;
        uint32_t reduceSize = (realN + 1) >> 1;
        CastAligned256F16VF128<T1>(srcLocalInt, dstLocalInt, gradLocalInt, fullExeSize, reduceSize, srcM);
    } else if constexpr (srcN <= 256) {
        const uint32_t fullExeSize = 128;
        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
        uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

        uint32_t tailSize = realN > fullExeSize ? (realN % fullExeSize) : 0;
        uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
        if (realN == fullExeSize * 2) {
            reduceSize = (fullExeSize + 1) >> 1;
        }
        CastAligned256F16VF256<T1, srcN>(srcLocalInt, dstLocalInt, gradLocalInt, fullExeSize, reduceSize,
                                         srcLocalIntTail, gradLocalIntTail, srcM);
    }
}
#else
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned256F16(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T1> &gradTensor,
                                                           const LocalTensor<T1> &srcTensor, uint32_t srcM,
                                                           uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // MY_SOFTMAX_GRAD_CAST_ALIGNED256_F16_INTERFACE_H_
