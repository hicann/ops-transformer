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
#ifndef MY_SOFTMAX_GRAD_CAST_ALIGNED256_F32_INTERFACE_H_
#define MY_SOFTMAX_GRAD_CAST_ALIGNED256_F32_INTERFACE_H_
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
template <typename T1>
__simd_vf__ inline void CastAligned256F32VF64(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                              const uint32_t fullExeSize, uint32_t srcM, uint32_t realN)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregMul;

    RegTensor<float> vregAdd;
    RegTensor<float> vregReduceSum;

    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(realN);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), fullExeSize);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt), fullExeSize);
        Mul(vregMul, vregGrad, vregSrc, pregTailExe);
        ReduceSum(vregReduceSum, vregMul, pregTailExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, uint32_t srcN>
__simd_vf__ inline void CastAligned256F32VF128(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               uint64_t srcLocalIntTail, uint64_t gradLocalIntTail, uint32_t reduceSize,
                                               uint32_t srcM)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregMul;
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregAdd;
    RegTensor<float> vregReduceSum;
    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail), srcN);
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Add(vregAdd, vregMul, vregMulTail, pregFullExe);
        ReduceSum(vregReduceSum, vregAdd, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, uint32_t srcN>
__simd_vf__ inline void CastAligned256F32VF192(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               uint64_t srcLocalInt1, uint64_t gradLocalInt1, uint64_t srcLocalIntTail,
                                               uint64_t gradLocalIntTail, uint32_t reduceSize, uint32_t srcM)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregMul;
    RegTensor<float> vregMul1;
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregAdd;
    RegTensor<float> vregAdd1;
    RegTensor<float> vregReduceSum;
    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail), srcN);
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Add(vregAdd, vregMul, vregMulTail, pregFullExe);
        Add(vregAdd1, vregMul1, vregAdd, pregFullExe);
        ReduceSum(vregReduceSum, vregAdd1, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, uint32_t srcN>
__simd_vf__ inline void CastAligned256F32VF256(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                               uint64_t srcLocalInt1, uint64_t gradLocalInt1, uint64_t srcLocalInt2,
                                               uint64_t gradLocalInt2, uint64_t srcLocalIntTail,
                                               uint64_t gradLocalIntTail, uint32_t reduceSize, uint32_t srcM)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregSrc1;
    RegTensor<float> vregGrad1;
    RegTensor<float> vregSrc2;
    RegTensor<float> vregGrad2;
    RegTensor<float> vregMul;
    RegTensor<float> vregMul1;
    RegTensor<float> vregMul2;
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregGradTail;
    RegTensor<float> vregMulTail;
    RegTensor<float> vregAdd;
    RegTensor<float> vregAdd1;
    RegTensor<float> vregAdd2;
    RegTensor<float> vregReduceSum;
    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        // 手动unroll 128个数分64个数做mul和add
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1 *&)srcLocalInt), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1 *&)gradLocalInt), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1 *&)srcLocalInt1), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1 *&)gradLocalInt1), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1 *&)srcLocalInt2), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1 *&)gradLocalInt2), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1 *&)srcLocalIntTail), srcN);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail, ((__ubuf__ T1 *&)gradLocalIntTail), srcN);
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
        Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
        Add(vregAdd, vregMul, vregMulTail, pregFullExe);
        Add(vregAdd1, vregMul1, vregMul2, pregFullExe);
        Add(vregAdd2, vregAdd, vregAdd1, pregFullExe);
        ReduceSum(vregReduceSum, vregAdd2, pregFullExe);
        StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregReduceSum,
                                                                     uregReduceSum, 1);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned256F32(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T1> &gradTensor,
                                                           const LocalTensor<T1> &srcTensor, uint32_t srcM,
                                                           uint32_t realN = srcN)
{
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t gradLocalInt = gradTensor.GetPhyAddr();

    if constexpr (srcN == 64) {
        const uint32_t fullExeSize = 64;
        CastAligned256F32VF64<T1>(srcLocalInt, dstLocalInt, gradLocalInt, fullExeSize, srcM, realN);
    } else if constexpr (srcN == 128) {
        // D=128 unroll一次
        const uint32_t fullExeSize = 64;
        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
        uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

        uint32_t tailSize = realN % fullExeSize;
        uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;
        CastAligned256F32VF128<T1, srcN>(srcLocalInt, dstLocalInt, gradLocalInt, srcLocalIntTail, gradLocalIntTail,
                                         reduceSize, srcM);
    } else if constexpr (srcN == 192) {
        // D=192 unroll2次
        const uint32_t fullExeSize = 64;
        uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
        uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
        uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

        uint32_t tailSize = realN % fullExeSize;
        uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;
        CastAligned256F32VF192<T1, srcN>(srcLocalInt, dstLocalInt, gradLocalInt, srcLocalInt1, gradLocalInt1,
                                         srcLocalIntTail, gradLocalIntTail, reduceSize, srcM);
    } else if constexpr (srcN == 256) {
        // D=256 unroll3次
        const uint32_t fullExeSize = 64;
        uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
        uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

        uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
        uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
        uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

        uint32_t tailSize = realN % fullExeSize;
        uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;
        CastAligned256F32VF256<T1, srcN>(srcLocalInt, dstLocalInt, gradLocalInt, srcLocalInt1, gradLocalInt1,
                                         srcLocalInt2, gradLocalInt2, srcLocalIntTail, gradLocalIntTail, reduceSize,
                                         srcM);
    }
}
#else
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCastAligned256F32(const LocalTensor<T> &dstTensor,
                                                           const LocalTensor<T1> &gradTensor,
                                                           const LocalTensor<T1> &srcTensor, uint32_t srcM,
                                                           uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // MY_SOFTMAX_GRAD_CAST_ALIGNED256_F32_INTERFACE_H_
