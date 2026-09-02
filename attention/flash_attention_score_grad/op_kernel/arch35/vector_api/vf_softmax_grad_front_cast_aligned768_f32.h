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
#ifndef MY_SOFTMAX_GRAD_CAST_ALIGNED768_F32_INTERFACE_H_
#define MY_SOFTMAX_GRAD_CAST_ALIGNED768_F32_INTERFACE_H_
#include "kernel_tensor.h"

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
__simd_vf__ inline void CastAligned768F32VF576(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               uint32_t srcM, uint32_t realN)
{
    const uint32_t fullExeSize = 64;
    uint64_t srcLocalInt1 = srcLocalInt + fullExeSize * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(T1);
    uint64_t srcLocalInt2 = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t gradLocalInt2 = gradLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t srcLocalInt3 = srcLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t gradLocalInt3 = gradLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t srcLocalInt4 = srcLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t gradLocalInt4 = gradLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t srcLocalInt5 = srcLocalInt + fullExeSize * 5 * sizeof(T1);
    uint64_t gradLocalInt5 = gradLocalInt + fullExeSize * 5 * sizeof(T1);
    uint64_t srcLocalInt6 = srcLocalInt + fullExeSize * 6 * sizeof(T1);
    uint64_t gradLocalInt6 = gradLocalInt + fullExeSize * 6 * sizeof(T1);
    uint64_t srcLocalInt7 = srcLocalInt + fullExeSize * 7 * sizeof(T1);
    uint64_t gradLocalInt7 = gradLocalInt + fullExeSize * 7 * sizeof(T1);
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 8 * sizeof(T1);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 8 * sizeof(T1);
    uint32_t tailSize = realN % fullExeSize;
    uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregSrc2;
    RegTensor<float> vregGrad2;
    RegTensor<float> vregSrc3;
    RegTensor<float> vregGrad3;
    RegTensor<float> vregSrc4;
    RegTensor<float> vregGrad4;
    RegTensor<float> vregSrc5;
    RegTensor<float> vregGrad5;
    RegTensor<float> vregSrc6;
    RegTensor<float> vregGrad6;
    RegTensor<float> vregSrc7;
    RegTensor<float> vregGrad7;
    RegTensor<float> vregMul;
    RegTensor<float> vregMul1;
    RegTensor<float> vregMul2;
    RegTensor<float> vregMul3;
    RegTensor<float> vregMul4;
    RegTensor<float> vregMul5;
    RegTensor<float> vregMul6;
    RegTensor<float> vregMul7;
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregAdd;
    RegTensor<float> vregAdd1;
    RegTensor<float> vregAdd2;
    RegTensor<float> vregAdd3;
    RegTensor<float> vregAdd4;
    RegTensor<float> vregAdd5;
    RegTensor<float> vregAdd6;
    RegTensor<float> vregAdd7;
    RegTensor<float> vregReduceSum;
    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1 *&)srcLocalInt2),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1 *&)gradLocalInt2),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1 *&)srcLocalInt3),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1 *&)gradLocalInt3),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1 *&)srcLocalInt4),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1 *&)gradLocalInt4),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1 *&)srcLocalInt5),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1 *&)gradLocalInt5),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1 *&)srcLocalInt6),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1 *&)gradLocalInt6),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc7, ((__ubuf__ T1 *&)srcLocalInt7),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad7, ((__ubuf__ T1 *&)gradLocalInt7),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                               HEAD_DIM_ALIGN);
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
        Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
        Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
        Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
        Mul(vregMul6, vregGrad6, vregSrc6, pregFullExe);
        Mul(vregMul7, vregGrad7, vregSrc7, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Add(vregAdd, vregMul, vregMulTail, pregFullExe);
        Add(vregAdd1, vregMul1, vregMul7, pregFullExe);
        Add(vregAdd2, vregMul2, vregMul6, pregFullExe);
        Add(vregAdd3, vregMul3, vregMul5, pregFullExe);
        Add(vregAdd4, vregAdd, vregMul4, pregFullExe);
        Add(vregAdd5, vregAdd1, vregAdd3, pregFullExe);
        Add(vregAdd6, vregAdd4, vregAdd2, pregFullExe);
        Add(vregAdd7, vregAdd6, vregAdd5, pregFullExe);
        ReduceSum(vregReduceSum, vregAdd7, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__simd_vf__ inline void CastAligned768F32VF640(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               uint32_t srcM, uint32_t realN)
{
    const uint32_t fullExeSize = 64;
    uint64_t srcLocalInt1 = srcLocalInt + fullExeSize * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(T1);
    uint64_t srcLocalInt2 = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t gradLocalInt2 = gradLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t srcLocalInt3 = srcLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t gradLocalInt3 = gradLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t srcLocalInt4 = srcLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t gradLocalInt4 = gradLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t srcLocalInt5 = srcLocalInt + fullExeSize * 5 * sizeof(T1);
    uint64_t gradLocalInt5 = gradLocalInt + fullExeSize * 5 * sizeof(T1);
    uint64_t srcLocalInt6 = srcLocalInt + fullExeSize * 6 * sizeof(T1);
    uint64_t gradLocalInt6 = gradLocalInt + fullExeSize * 6 * sizeof(T1);
    uint64_t srcLocalInt7 = srcLocalInt + fullExeSize * 7 * sizeof(T1);
    uint64_t gradLocalInt7 = gradLocalInt + fullExeSize * 7 * sizeof(T1);
    uint64_t srcLocalInt8 = srcLocalInt + fullExeSize * 8 * sizeof(T1);
    uint64_t gradLocalInt8 = gradLocalInt + fullExeSize * 8 * sizeof(T1);
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 9 * sizeof(T1);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 9 * sizeof(T1);
    uint32_t tailSize = realN % fullExeSize;
    uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregSrc2;
    RegTensor<float> vregGrad2;
    RegTensor<float> vregSrc3;
    RegTensor<float> vregGrad3;
    RegTensor<float> vregSrc4;
    RegTensor<float> vregGrad4;
    RegTensor<float> vregSrc5;
    RegTensor<float> vregGrad5;
    RegTensor<float> vregSrc6;
    RegTensor<float> vregGrad6;
    RegTensor<float> vregSrc7;
    RegTensor<float> vregGrad7;
    RegTensor<float> vregSrc8;
    RegTensor<float> vregGrad8;
    RegTensor<float> vregMul;
    RegTensor<float> vregMul1;
    RegTensor<float> vregMul2;
    RegTensor<float> vregMul3;
    RegTensor<float> vregMul4;
    RegTensor<float> vregMul5;
    RegTensor<float> vregMul6;
    RegTensor<float> vregMul7;
    RegTensor<float> vregMul8;
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregAdd;
    RegTensor<float> vregAdd1;
    RegTensor<float> vregAdd2;
    RegTensor<float> vregAdd3;
    RegTensor<float> vregAdd4;
    RegTensor<float> vregAdd5;
    RegTensor<float> vregAdd6;
    RegTensor<float> vregAdd7;
    RegTensor<float> vregAdd8;
    RegTensor<float> vregReduceSum;
    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1 *&)srcLocalInt2),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1 *&)gradLocalInt2),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1 *&)srcLocalInt3),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1 *&)gradLocalInt3),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1 *&)srcLocalInt4),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1 *&)gradLocalInt4),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1 *&)srcLocalInt5),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1 *&)gradLocalInt5),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1 *&)srcLocalInt6),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1 *&)gradLocalInt6),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc7, ((__ubuf__ T1 *&)srcLocalInt7),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad7, ((__ubuf__ T1 *&)gradLocalInt7),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc8, ((__ubuf__ T1 *&)srcLocalInt8),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad8, ((__ubuf__ T1 *&)gradLocalInt8),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                               HEAD_DIM_ALIGN);
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
        Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
        Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
        Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
        Mul(vregMul6, vregGrad6, vregSrc6, pregFullExe);
        Mul(vregMul7, vregGrad7, vregSrc7, pregFullExe);
        Mul(vregMul8, vregGrad8, vregSrc8, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Add(vregAdd, vregMul, vregMulTail, pregFullExe);
        Add(vregAdd1, vregMul1, vregMul8, pregFullExe);
        Add(vregAdd2, vregMul2, vregMul7, pregFullExe);
        Add(vregAdd3, vregMul3, vregMul6, pregFullExe);
        Add(vregAdd4, vregMul4, vregMul5, pregFullExe);
        Add(vregAdd5, vregAdd, vregAdd4, pregFullExe);
        Add(vregAdd6, vregAdd1, vregAdd3, pregFullExe);
        Add(vregAdd7, vregAdd5, vregAdd2, pregFullExe);
        Add(vregAdd8, vregAdd7, vregAdd6, pregFullExe);
        ReduceSum(vregReduceSum, vregAdd8, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, uint32_t HEAD_DIM_ALIGN>
__simd_vf__ inline void CastAligned768F32VF704(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               uint32_t srcM, uint32_t realN)
{
    const uint32_t fullExeSize = 64;
    uint64_t srcLocalInt1 = srcLocalInt + fullExeSize * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(T1);
    uint64_t srcLocalInt2 = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t gradLocalInt2 = gradLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t srcLocalInt3 = srcLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t gradLocalInt3 = gradLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t srcLocalInt4 = srcLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t gradLocalInt4 = gradLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t srcLocalInt5 = srcLocalInt + fullExeSize * 5 * sizeof(T1);
    uint64_t gradLocalInt5 = gradLocalInt + fullExeSize * 5 * sizeof(T1);
    uint64_t srcLocalInt6 = srcLocalInt + fullExeSize * 6 * sizeof(T1);
    uint64_t gradLocalInt6 = gradLocalInt + fullExeSize * 6 * sizeof(T1);
    uint64_t srcLocalInt7 = srcLocalInt + fullExeSize * 7 * sizeof(T1);
    uint64_t gradLocalInt7 = gradLocalInt + fullExeSize * 7 * sizeof(T1);
    uint64_t srcLocalInt8 = srcLocalInt + fullExeSize * 8 * sizeof(T1);
    uint64_t gradLocalInt8 = gradLocalInt + fullExeSize * 8 * sizeof(T1);
    uint64_t srcLocalInt9 = srcLocalInt + fullExeSize * 9 * sizeof(T1);
    uint64_t gradLocalInt9 = gradLocalInt + fullExeSize * 9 * sizeof(T1);
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 10 * sizeof(T1);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 10 * sizeof(T1);
    uint32_t tailSize = realN % fullExeSize;
    uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregSrc2;
    RegTensor<float> vregGrad2;
    RegTensor<float> vregSrc3;
    RegTensor<float> vregGrad3;
    RegTensor<float> vregSrc4;
    RegTensor<float> vregGrad4;
    RegTensor<float> vregSrc5;
    RegTensor<float> vregGrad5;
    RegTensor<float> vregSrc6;
    RegTensor<float> vregGrad6;
    RegTensor<float> vregSrc7;
    RegTensor<float> vregGrad7;
    RegTensor<float> vregSrc8;
    RegTensor<float> vregGrad8;
    RegTensor<float> vregSrc9;
    RegTensor<float> vregGrad9;
    RegTensor<float> vregMul;
    RegTensor<float> vregMul1;
    RegTensor<float> vregMul2;
    RegTensor<float> vregMul3;
    RegTensor<float> vregMul4;
    RegTensor<float> vregMul5;
    RegTensor<float> vregMul6;
    RegTensor<float> vregMul7;
    RegTensor<float> vregMul8;
    RegTensor<float> vregMul9;
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregAdd;
    RegTensor<float> vregAdd1;
    RegTensor<float> vregAdd2;
    RegTensor<float> vregAdd3;
    RegTensor<float> vregAdd4;
    RegTensor<float> vregAdd5;
    RegTensor<float> vregAdd6;
    RegTensor<float> vregAdd7;
    RegTensor<float> vregAdd8;
    RegTensor<float> vregAdd9;
    RegTensor<float> vregReduceSum;
    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1 *&)srcLocalInt2),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1 *&)gradLocalInt2),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1 *&)srcLocalInt3),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1 *&)gradLocalInt3),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1 *&)srcLocalInt4),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1 *&)gradLocalInt4),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1 *&)srcLocalInt5),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1 *&)gradLocalInt5),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1 *&)srcLocalInt6),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1 *&)gradLocalInt6),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc7, ((__ubuf__ T1 *&)srcLocalInt7),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad7, ((__ubuf__ T1 *&)gradLocalInt7),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc8, ((__ubuf__ T1 *&)srcLocalInt8),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad8, ((__ubuf__ T1 *&)gradLocalInt8),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc9, ((__ubuf__ T1 *&)srcLocalInt9),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad9, ((__ubuf__ T1 *&)gradLocalInt9),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                               HEAD_DIM_ALIGN);
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
        Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
        Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
        Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
        Mul(vregMul6, vregGrad6, vregSrc6, pregFullExe);
        Mul(vregMul7, vregGrad7, vregSrc7, pregFullExe);
        Mul(vregMul8, vregGrad8, vregSrc8, pregFullExe);
        Mul(vregMul9, vregGrad9, vregSrc9, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Add(vregAdd, vregMul, vregMulTail, pregFullExe);
        Add(vregAdd1, vregMul1, vregMul9, pregFullExe);
        Add(vregAdd2, vregMul2, vregMul8, pregFullExe);
        Add(vregAdd3, vregMul3, vregMul7, pregFullExe);
        Add(vregAdd4, vregMul4, vregMul6, pregFullExe);
        Add(vregAdd5, vregAdd, vregMul5, pregFullExe);
        Add(vregAdd6, vregAdd1, vregAdd4, pregFullExe);
        Add(vregAdd7, vregAdd2, vregAdd3, pregFullExe);
        Add(vregAdd8, vregAdd5, vregAdd7, pregFullExe);
        Add(vregAdd9, vregAdd6, vregAdd8, pregFullExe);
        ReduceSum(vregReduceSum, vregAdd9, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, uint32_t HEAD_DIM_ALIGN>
__simd_vf__ inline void CastAligned768F32VF768(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               uint32_t srcM, uint32_t realN)
{
    const uint32_t fullExeSize = 64;
    uint64_t srcLocalInt1 = srcLocalInt + fullExeSize * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(T1);
    uint64_t srcLocalInt2 = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t gradLocalInt2 = gradLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t srcLocalInt3 = srcLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t gradLocalInt3 = gradLocalInt + fullExeSize * 3 * sizeof(T1);
    uint64_t srcLocalInt4 = srcLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t gradLocalInt4 = gradLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t srcLocalInt5 = srcLocalInt + fullExeSize * 5 * sizeof(T1);
    uint64_t gradLocalInt5 = gradLocalInt + fullExeSize * 5 * sizeof(T1);
    uint64_t srcLocalInt6 = srcLocalInt + fullExeSize * 6 * sizeof(T1);
    uint64_t gradLocalInt6 = gradLocalInt + fullExeSize * 6 * sizeof(T1);
    uint64_t srcLocalInt7 = srcLocalInt + fullExeSize * 7 * sizeof(T1);
    uint64_t gradLocalInt7 = gradLocalInt + fullExeSize * 7 * sizeof(T1);
    uint64_t srcLocalInt8 = srcLocalInt + fullExeSize * 8 * sizeof(T1);
    uint64_t gradLocalInt8 = gradLocalInt + fullExeSize * 8 * sizeof(T1);
    uint64_t srcLocalInt9 = srcLocalInt + fullExeSize * 9 * sizeof(T1);
    uint64_t gradLocalInt9 = gradLocalInt + fullExeSize * 9 * sizeof(T1);
    uint64_t srcLocalInt10 = srcLocalInt + fullExeSize * 10 * sizeof(T1);
    uint64_t gradLocalInt10 = gradLocalInt + fullExeSize * 10 * sizeof(T1);
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 11 * sizeof(T1);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 11 * sizeof(T1);
    uint32_t tailSize = realN % fullExeSize;
    uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregSrc2;
    RegTensor<float> vregGrad2;
    RegTensor<float> vregSrc3;
    RegTensor<float> vregGrad3;
    RegTensor<float> vregSrc4;
    RegTensor<float> vregGrad4;
    RegTensor<float> vregSrc5;
    RegTensor<float> vregGrad5;
    RegTensor<float> vregSrc6;
    RegTensor<float> vregGrad6;
    RegTensor<float> vregSrc7;
    RegTensor<float> vregGrad7;
    RegTensor<float> vregSrc8;
    RegTensor<float> vregGrad8;
    RegTensor<float> vregSrc9;
    RegTensor<float> vregGrad9;
    RegTensor<float> vregSrc10;
    RegTensor<float> vregGrad10;
    RegTensor<float> vregMul;
    RegTensor<float> vregMul1;
    RegTensor<float> vregMul2;
    RegTensor<float> vregMul3;
    RegTensor<float> vregMul4;
    RegTensor<float> vregMul5;
    RegTensor<float> vregMul6;
    RegTensor<float> vregMul7;
    RegTensor<float> vregMul8;
    RegTensor<float> vregMul9;
    RegTensor<float> vregMul10;
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregAdd;
    RegTensor<float> vregAdd1;
    RegTensor<float> vregAdd2;
    RegTensor<float> vregAdd3;
    RegTensor<float> vregAdd4;
    RegTensor<float> vregAdd5;
    RegTensor<float> vregAdd6;
    RegTensor<float> vregAdd7;
    RegTensor<float> vregAdd8;
    RegTensor<float> vregAdd9;
    RegTensor<float> vregAdd10;
    RegTensor<float> vregReduceSum;
    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1 *&)srcLocalInt2),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1 *&)gradLocalInt2),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1 *&)srcLocalInt3),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1 *&)gradLocalInt3),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1 *&)srcLocalInt4),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1 *&)gradLocalInt4),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1 *&)srcLocalInt5),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1 *&)gradLocalInt5),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1 *&)srcLocalInt6),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1 *&)gradLocalInt6),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc7, ((__ubuf__ T1 *&)srcLocalInt7),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad7, ((__ubuf__ T1 *&)gradLocalInt7),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc8, ((__ubuf__ T1 *&)srcLocalInt8),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad8, ((__ubuf__ T1 *&)gradLocalInt8),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc9, ((__ubuf__ T1 *&)srcLocalInt9),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad9, ((__ubuf__ T1 *&)gradLocalInt9),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc10, ((__ubuf__ T1 *&)srcLocalInt10),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad10, ((__ubuf__ T1 *&)gradLocalInt10),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                               HEAD_DIM_ALIGN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail),
                                                               HEAD_DIM_ALIGN);

        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
        Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
        Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
        Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
        Mul(vregMul6, vregGrad6, vregSrc6, pregFullExe);
        Mul(vregMul7, vregGrad7, vregSrc7, pregFullExe);
        Mul(vregMul8, vregGrad8, vregSrc8, pregFullExe);
        Mul(vregMul9, vregGrad9, vregSrc9, pregFullExe);
        Mul(vregMul10, vregGrad10, vregSrc10, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Add(vregAdd, vregMul, vregMulTail, pregFullExe);
        Add(vregAdd1, vregMul1, vregMul10, pregFullExe);
        Add(vregAdd2, vregMul2, vregMul9, pregFullExe);
        Add(vregAdd3, vregMul3, vregMul8, pregFullExe);
        Add(vregAdd4, vregMul4, vregMul7, pregFullExe);
        Add(vregAdd5, vregMul5, vregMul6, pregFullExe);
        Add(vregAdd6, vregAdd, vregAdd5, pregFullExe);
        Add(vregAdd7, vregAdd1, vregAdd4, pregFullExe);
        Add(vregAdd8, vregAdd2, vregAdd3, pregFullExe);
        Add(vregAdd9, vregAdd6, vregAdd8, pregFullExe);
        Add(vregAdd10, vregAdd9, vregAdd7, pregFullExe);
        ReduceSum(vregReduceSum, vregAdd10, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned768F32(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T1> &gradTensor,
                                                           const LocalTensor<T1> &srcTensor, uint32_t srcM,
                                                           uint32_t realN = srcN)
{
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t gradLocalInt = gradTensor.GetPhyAddr();

    if constexpr (srcN == 576) {
        // D=576 unroll8次
        CastAligned768F32VF576<T1, HEAD_DIM_ALIGN>(srcLocalInt, dstLocalInt, gradLocalInt, srcM, realN);
    } else if constexpr (srcN == 640) {
        // D=576 unroll9次
        CastAligned768F32VF640<T1, srcN, HEAD_DIM_ALIGN>(srcLocalInt, dstLocalInt, gradLocalInt, srcM, realN);
    } else if constexpr (srcN == 704) {
        // D=576 unroll10次
        CastAligned768F32VF704<T1, HEAD_DIM_ALIGN>(srcLocalInt, dstLocalInt, gradLocalInt, srcM, realN);
    } else if constexpr (srcN == 768) {
        // D=576 unroll11次
        CastAligned768F32VF768<T1, HEAD_DIM_ALIGN>(srcLocalInt, dstLocalInt, gradLocalInt, srcM, realN);
    }
}
#else
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned768F32(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T1> &gradTensor,
                                                           const LocalTensor<T1> &srcTensor, uint32_t srcM,
                                                           uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // MY_SOFTMAX_GRAD_CAST_ALIGNED768_F32_INTERFACE_H_
