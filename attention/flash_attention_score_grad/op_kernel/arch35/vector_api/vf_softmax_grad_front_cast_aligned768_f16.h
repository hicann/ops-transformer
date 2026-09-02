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
#ifndef MY_SOFTMAX_GRAD_CAST_ALIGNED768_F16_INTERFACE_H_
#define MY_SOFTMAX_GRAD_CAST_ALIGNED768_F16_INTERFACE_H_
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
template <typename T1, uint32_t HEAD_DIM_ALIGN>
__simd_vf__ inline void CastAligned768F16VF640(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               uint32_t reduceSize, uint32_t srcM, uint32_t realN)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregMul;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregMul2;

    RegTensor<float> vregSrc2;
    RegTensor<float> vregGrad2;
    RegTensor<float> vregMul3;
    RegTensor<float> vregSrc3;
    RegTensor<float> vregGrad3;
    RegTensor<float> vregMul4;

    RegTensor<float> vregSrc4;
    RegTensor<float> vregGrad4;
    RegTensor<float> vregMul5;
    RegTensor<float> vregSrc5;
    RegTensor<float> vregGrad5;
    RegTensor<float> vregMul6;

    RegTensor<float> vregSrc6;
    RegTensor<float> vregGrad6;
    RegTensor<float> vregMul7;
    RegTensor<float> vregSrc7;
    RegTensor<float> vregGrad7;
    RegTensor<float> vregMul8;

    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregSrc1Tail;
    RegTensor<float> vregGrad1Tail;
    RegTensor<float> vregMul2Tail;

    RegTensor<half> vregSrcHalf;
    RegTensor<half> vregGradHalf;
    RegTensor<half> vregSrcHalf1;
    RegTensor<half> vregGradHalf1;
    RegTensor<half> vregSrcHalf2;
    RegTensor<half> vregGradHalf2;
    RegTensor<half> vregSrcHalf3;
    RegTensor<half> vregGradHalf3;
    RegTensor<bfloat16_t> vregSrcBf;
    RegTensor<bfloat16_t> vregGradBf;
    RegTensor<bfloat16_t> vregSrcBf1;
    RegTensor<bfloat16_t> vregGradBf1;
    RegTensor<bfloat16_t> vregSrcBf2;
    RegTensor<bfloat16_t> vregGradBf2;
    RegTensor<bfloat16_t> vregSrcBf3;
    RegTensor<bfloat16_t> vregGradBf3;

    RegTensor<half> vregSrcHalfTail;
    RegTensor<half> vregGradHalfTail;
    RegTensor<bfloat16_t> vregSrcBfTail;
    RegTensor<bfloat16_t> vregGradBfTail;

    RegTensor<float> vregAdd;
    RegTensor<float> vregAdd1;
    RegTensor<float> vregAdd2;
    RegTensor<float> vregAdd3;
    RegTensor<float> vregAddTail;
    RegTensor<float> vregAddTemp;
    RegTensor<float> vregAddTemp1;
    RegTensor<float> vregAddLast;
    RegTensor<float> vregReduceSum;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<T1, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    const uint32_t fullExeSize = 128;
    uint64_t srcLocalInt1 = srcLocalInt + fullExeSize * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(T1);
    uint64_t srcLocalInt2 = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t gradLocalInt2 = gradLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t srcLocalInt3 = srcLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t gradLocalInt3 = gradLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 4 * sizeof(T1);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        if constexpr (IsSameType<T1, half>::value) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1 *&)srcLocalInt),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf, ((__ubuf__ T1 *&)gradLocalInt),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf1, ((__ubuf__ T1 *&)srcLocalInt1),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf1, ((__ubuf__ T1 *&)gradLocalInt1),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf2, ((__ubuf__ T1 *&)srcLocalInt2),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf2, ((__ubuf__ T1 *&)gradLocalInt2),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf3, ((__ubuf__ T1 *&)srcLocalInt3),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf3, ((__ubuf__ T1 *&)gradLocalInt3),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalfTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalfTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                                   HEAD_DIM_ALIGN);
            Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc2, vregSrcHalf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc3, vregSrcHalf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad2, vregGradHalf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad3, vregGradHalf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc4, vregSrcHalf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc5, vregSrcHalf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad4, vregGradHalf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad5, vregGradHalf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc6, vregSrcHalf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc7, vregSrcHalf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad6, vregGradHalf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad7, vregGradHalf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcHalfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcHalfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradHalfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradHalfTail, pregFullExeB16);
        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1 *&)srcLocalInt),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1 *&)gradLocalInt),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf1, ((__ubuf__ T1 *&)srcLocalInt1),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf1, ((__ubuf__ T1 *&)gradLocalInt1),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf2, ((__ubuf__ T1 *&)srcLocalInt2),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf2, ((__ubuf__ T1 *&)gradLocalInt2),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf3, ((__ubuf__ T1 *&)srcLocalInt3),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf3, ((__ubuf__ T1 *&)gradLocalInt3),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBfTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBfTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                                   HEAD_DIM_ALIGN);
            Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc2, vregSrcBf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad2, vregGradBf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc3, vregSrcBf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad3, vregGradBf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc4, vregSrcBf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad4, vregGradBf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc5, vregSrcBf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad5, vregGradBf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc6, vregSrcBf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad6, vregGradBf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc7, vregSrcBf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad7, vregGradBf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcBfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradBfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcBfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradBfTail, pregFullExeB16);
        }
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul2, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMul3, vregGrad2, vregSrc2, pregFullExe);
        Mul(vregMul4, vregGrad3, vregSrc3, pregFullExe);
        Mul(vregMul5, vregGrad4, vregSrc4, pregFullExe);
        Mul(vregMul6, vregGrad5, vregSrc5, pregFullExe);
        Mul(vregMul7, vregGrad6, vregSrc6, pregFullExe);
        Mul(vregMul8, vregGrad7, vregSrc7, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Mul(vregMul2Tail, vregGrad1Tail, vregSrc1Tail, pregTailExe);
        Add(vregAdd, vregMul, vregMul2, pregFullExe);
        Add(vregAdd1, vregMul3, vregMul4, pregFullExe);
        Add(vregAdd2, vregMul5, vregMul6, pregFullExe);
        Add(vregAdd3, vregMul7, vregMul8, pregFullExe);
        Add(vregAddTail, vregMulTail, vregMul2Tail, pregTailExe);
        Add(vregAddTemp, vregAdd, vregAddTail, pregFullExe);
        Add(vregAddTemp1, vregAdd1, vregAdd2, pregFullExe);
        Add(vregAdd, vregAddTemp, vregAdd3, pregFullExe);
        Add(vregAddLast, vregAddTemp1, vregAdd, pregFullExe);
        ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, uint32_t HEAD_DIM_ALIGN>
__simd_vf__ inline void CastAligned768F16VF768(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               uint32_t reduceSize, uint32_t srcM, uint32_t realN)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregMul;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregMul2;
    RegTensor<float> vregSrc2;
    RegTensor<float> vregGrad2;
    RegTensor<float> vregMul3;
    RegTensor<float> vregSrc3;
    RegTensor<float> vregGrad3;
    RegTensor<float> vregMul4;
    RegTensor<float> vregSrc4;
    RegTensor<float> vregGrad4;
    RegTensor<float> vregMul5;
    RegTensor<float> vregSrc5;
    RegTensor<float> vregGrad5;
    RegTensor<float> vregMul6;
    RegTensor<float> vregSrc6;
    RegTensor<float> vregGrad6;
    RegTensor<float> vregMul7;
    RegTensor<float> vregSrc7;
    RegTensor<float> vregGrad7;
    RegTensor<float> vregMul8;
    RegTensor<float> vregSrc8;
    RegTensor<float> vregGrad8;
    RegTensor<float> vregMul9;
    RegTensor<float> vregSrc9;
    RegTensor<float> vregGrad9;
    RegTensor<float> vregMul10;

    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregSrc1Tail;
    RegTensor<float> vregGrad1Tail;
    RegTensor<float> vregMul2Tail;

    RegTensor<half> vregSrcHalf;
    RegTensor<half> vregGradHalf;
    RegTensor<half> vregSrcHalf1;
    RegTensor<half> vregGradHalf1;
    RegTensor<half> vregSrcHalf2;
    RegTensor<half> vregGradHalf2;
    RegTensor<half> vregSrcHalf3;
    RegTensor<half> vregGradHalf3;
    RegTensor<half> vregSrcHalf4;
    RegTensor<half> vregGradHalf4;
    RegTensor<bfloat16_t> vregSrcBf;
    RegTensor<bfloat16_t> vregGradBf;
    RegTensor<bfloat16_t> vregSrcBf1;
    RegTensor<bfloat16_t> vregGradBf1;
    RegTensor<bfloat16_t> vregSrcBf2;
    RegTensor<bfloat16_t> vregGradBf2;
    RegTensor<bfloat16_t> vregSrcBf3;
    RegTensor<bfloat16_t> vregGradBf3;
    RegTensor<bfloat16_t> vregSrcBf4;
    RegTensor<bfloat16_t> vregGradBf4;

    RegTensor<half> vregSrcHalfTail;
    RegTensor<half> vregGradHalfTail;
    RegTensor<bfloat16_t> vregSrcBfTail;
    RegTensor<bfloat16_t> vregGradBfTail;

    RegTensor<float> vregAdd;
    RegTensor<float> vregAdd1;
    RegTensor<float> vregAdd2;
    RegTensor<float> vregAdd3;
    RegTensor<float> vregAdd4;
    RegTensor<float> vregAddTail;
    RegTensor<float> vregAddTemp;
    RegTensor<float> vregAddTemp1;
    RegTensor<float> vregAddTemp2;
    RegTensor<float> vregAddLast;
    RegTensor<float> vregReduceSum;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<T1, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    const uint32_t fullExeSize = 128;
    uint64_t srcLocalInt1 = srcLocalInt + fullExeSize * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(T1);
    uint64_t srcLocalInt2 = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t gradLocalInt2 = gradLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t srcLocalInt3 = srcLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t gradLocalInt3 = gradLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t srcLocalInt4 = srcLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t gradLocalInt4 = gradLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 5 * sizeof(T1);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 5 * sizeof(T1);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        if constexpr (IsSameType<T1, half>::value) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1 *&)srcLocalInt),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf, ((__ubuf__ T1 *&)gradLocalInt),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf1, ((__ubuf__ T1 *&)srcLocalInt1),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf1, ((__ubuf__ T1 *&)gradLocalInt1),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf2, ((__ubuf__ T1 *&)srcLocalInt2),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf2, ((__ubuf__ T1 *&)gradLocalInt2),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf3, ((__ubuf__ T1 *&)srcLocalInt3),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf3, ((__ubuf__ T1 *&)gradLocalInt3),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf4, ((__ubuf__ T1 *&)srcLocalInt4),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf4, ((__ubuf__ T1 *&)gradLocalInt4),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalfTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalfTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                                   HEAD_DIM_ALIGN);
            Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradHalf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc2, vregSrcHalf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc3, vregSrcHalf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad2, vregGradHalf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad3, vregGradHalf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc4, vregSrcHalf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc5, vregSrcHalf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad4, vregGradHalf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad5, vregGradHalf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc6, vregSrcHalf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc7, vregSrcHalf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad6, vregGradHalf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad7, vregGradHalf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc8, vregSrcHalf4, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc9, vregSrcHalf4, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad8, vregGradHalf4, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad9, vregGradHalf4, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcHalfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcHalfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradHalfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradHalfTail, pregFullExeB16);
        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1 *&)srcLocalInt),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1 *&)gradLocalInt),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf1, ((__ubuf__ T1 *&)srcLocalInt1),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf1, ((__ubuf__ T1 *&)gradLocalInt1),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf2, ((__ubuf__ T1 *&)srcLocalInt2),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf2, ((__ubuf__ T1 *&)gradLocalInt2),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf3, ((__ubuf__ T1 *&)srcLocalInt3),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf3, ((__ubuf__ T1 *&)gradLocalInt3),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf4, ((__ubuf__ T1 *&)srcLocalInt4),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf4, ((__ubuf__ T1 *&)gradLocalInt4),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBfTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                                   HEAD_DIM_ALIGN);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBfTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                                   HEAD_DIM_ALIGN);
            Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradBf, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc2, vregSrcBf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad2, vregGradBf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc3, vregSrcBf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad3, vregGradBf1, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc4, vregSrcBf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad4, vregGradBf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc5, vregSrcBf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad5, vregGradBf2, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc6, vregSrcBf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad6, vregGradBf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc7, vregSrcBf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad7, vregGradBf3, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrc8, vregSrcBf4, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGrad8, vregGradBf4, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc9, vregSrcBf4, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad9, vregGradBf4, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcBfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradBfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcBfTail, pregFullExeB16);
            Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradBfTail, pregFullExeB16);
        }
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul2, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMul3, vregGrad2, vregSrc2, pregFullExe);
        Mul(vregMul4, vregGrad3, vregSrc3, pregFullExe);
        Mul(vregMul5, vregGrad4, vregSrc4, pregFullExe);
        Mul(vregMul6, vregGrad5, vregSrc5, pregFullExe);
        Mul(vregMul7, vregGrad6, vregSrc6, pregFullExe);
        Mul(vregMul8, vregGrad7, vregSrc7, pregFullExe);
        Mul(vregMul9, vregGrad8, vregSrc8, pregFullExe);
        Mul(vregMul10, vregGrad9, vregSrc9, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Mul(vregMul2Tail, vregGrad1Tail, vregSrc1Tail, pregTailExe);
        Add(vregAdd, vregMul, vregMul2, pregFullExe);
        Add(vregAdd1, vregMul3, vregMul4, pregFullExe);
        Add(vregAdd2, vregMul5, vregMul6, pregFullExe);
        Add(vregAdd3, vregMul7, vregMul8, pregFullExe);
        Add(vregAdd4, vregMul9, vregMul10, pregFullExe);
        Add(vregAddTail, vregMulTail, vregMul2Tail, pregTailExe);

        Add(vregAddTemp, vregAdd, vregAddTail, pregFullExe);
        Add(vregAddTemp1, vregAdd1, vregAdd2, pregFullExe);
        Add(vregAddTemp2, vregAdd3, vregAdd4, pregFullExe);
        Add(vregAdd, vregAddTemp, vregAddTemp1, pregFullExe);
        Add(vregAddLast, vregAddTemp2, vregAdd, pregFullExe);
        ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned768F16(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T1> &gradTensor,
                                                           const LocalTensor<T1> &srcTensor, uint32_t srcM,
                                                           uint32_t realN = srcN)
{
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t gradLocalInt = gradTensor.GetPhyAddr();

    if constexpr (srcN == 640) {
        const uint32_t fullExeSize = 128;
        uint32_t tailSize = realN > fullExeSize * 4 ? (realN % fullExeSize) : 0;
        uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
        if (realN == fullExeSize * 5) {
            reduceSize = (fullExeSize + 1) >> 1;
        }
        CastAligned768F16VF640<T1, HEAD_DIM_ALIGN>(srcLocalInt, dstLocalInt, gradLocalInt, reduceSize, srcM, realN);
    } else if constexpr (srcN == 768) {
        const uint32_t fullExeSize = 128;
        uint32_t tailSize = realN > fullExeSize * 5 ? (realN % fullExeSize) : 0;
        uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
        if (realN == fullExeSize * 6) {
            reduceSize = (fullExeSize + 1) >> 1;
        }
        CastAligned768F16VF768<T1, HEAD_DIM_ALIGN>(srcLocalInt, dstLocalInt, gradLocalInt, reduceSize, srcM, realN);
    }
}
#else
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned768F16(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T1> &gradTensor,
                                                           const LocalTensor<T1> &srcTensor, uint32_t srcM,
                                                           uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // MY_SOFTMAX_GRAD_CAST_ALIGNED768_F16_INTERFACE_H_
