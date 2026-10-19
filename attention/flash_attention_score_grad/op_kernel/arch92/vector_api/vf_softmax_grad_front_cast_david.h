/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_softmax_grad_front_cast_david.h
 */
#ifndef MY_SOFTMAX_GRAD_CAST_INTERFACE_DAVID_H_
#define MY_SOFTMAX_GRAD_CAST_INTERFACE_DAVID_H_
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32Odd = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
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
__aicore__ inline void MySoftmaxGradFrontCast(const LocalTensor<T>& dstTensor, const LocalTensor<T1>& gradTensor,
                                              const LocalTensor<T1>& srcTensor, uint32_t srcM, uint32_t realN = srcN)
{
    if constexpr (IsSameType<T1, float>::value) {
        uint64_t srcLocalInt = srcTensor.GetPhyAddr();
        uint64_t dstLocalInt = dstTensor.GetPhyAddr();
        uint64_t gradLocalInt = gradTensor.GetPhyAddr();
        // D=64 一次全载
        if constexpr (srcN == 64) {
            const uint32_t fullExeSize = 64;
            __VEC_SCOPE__
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
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt),
                                                                          fullExeSize);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          fullExeSize);
                    Mul(vregMul, vregGrad, vregSrc, pregTailExe);
                    ReduceSum(vregReduceSum, vregMul, pregTailExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 128) {
            // D=128 unroll一次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
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
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                    Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                    Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                    Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 192) {
            // D=192 unroll2次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
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
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                    Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                    Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                    Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                    Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                    Add(vregAdd1, vregMul1, vregAdd, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd1, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
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

            __VEC_SCOPE__
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
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1*&)gradLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                    Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                    Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                    Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                    Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                    Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                    Add(vregAdd1, vregMul1, vregMul2, pregFullExe);
                    Add(vregAdd2, vregAdd, vregAdd1, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd2, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 320) {
            // D=320 unroll4次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
            {
                RegTensor<float> vregSrc;
                RegTensor<float> vregGrad;
                RegTensor<float> vregSrc1;
                RegTensor<float> vregGrad1;
                RegTensor<float> vregSrc2;
                RegTensor<float> vregGrad2;
                RegTensor<float> vregSrc3;
                RegTensor<float> vregGrad3;
                RegTensor<float> vregMul;
                RegTensor<float> vregMul1;
                RegTensor<float> vregMul2;
                RegTensor<float> vregMul3;
                RegTensor<float> vregSrcTail;
                RegTensor<float> vregGradTail;
                RegTensor<float> vregMulTail;
                RegTensor<float> vregAdd;
                RegTensor<float> vregAdd1;
                RegTensor<float> vregAdd2;
                RegTensor<float> vregAdd3;
                RegTensor<float> vregReduceSum;
                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1*&)gradLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1*&)gradLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                    Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                    Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                    Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                    Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
                    Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                    Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                    Add(vregAdd1, vregMul1, vregMul3, pregFullExe);
                    Add(vregAdd2, vregAdd, vregMul2, pregFullExe);
                    Add(vregAdd3, vregAdd2, vregAdd1, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd3, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 384) {
            // D=384 unroll5次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

            uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
            {
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
                RegTensor<float> vregMul;
                RegTensor<float> vregMul1;
                RegTensor<float> vregMul2;
                RegTensor<float> vregMul3;
                RegTensor<float> vregMul4;
                RegTensor<float> vregSrcTail;
                RegTensor<float> vregGradTail;
                RegTensor<float> vregMulTail;
                RegTensor<float> vregAdd;
                RegTensor<float> vregAdd1;
                RegTensor<float> vregAdd2;
                RegTensor<float> vregAdd3;
                RegTensor<float> vregReduceSum;
                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1*&)gradLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1*&)gradLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1*&)srcLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1*&)gradLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                    Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                    Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                    Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                    Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
                    Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
                    Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                    Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                    Add(vregAdd1, vregMul1, vregMul4, pregFullExe);
                    Add(vregAdd2, vregMul2, vregMul3, pregFullExe);
                    Add(vregAdd3, vregAdd, vregAdd1, pregFullExe);
                    Add(vregAdd, vregAdd3, vregAdd2, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 448) {
            // D=448 unroll6次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

            uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

            uint64_t srcLocalInt5 = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
            uint64_t gradLocalInt5 = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
            {
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
                RegTensor<float> vregMul;
                RegTensor<float> vregMul1;
                RegTensor<float> vregMul2;
                RegTensor<float> vregMul3;
                RegTensor<float> vregMul4;
                RegTensor<float> vregMul5;
                RegTensor<float> vregSrcTail;
                RegTensor<float> vregGradTail;
                RegTensor<float> vregMulTail;
                RegTensor<float> vregAdd;
                RegTensor<float> vregAdd1;
                RegTensor<float> vregAdd2;
                RegTensor<float> vregAdd3;
                RegTensor<float> vregAdd4;
                RegTensor<float> vregAdd5;
                RegTensor<float> vregReduceSum;
                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1*&)gradLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1*&)gradLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1*&)srcLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1*&)gradLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1*&)srcLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1*&)gradLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                    Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                    Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                    Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                    Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
                    Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
                    Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
                    Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                    Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                    Add(vregAdd1, vregMul1, vregMul5, pregFullExe);
                    Add(vregAdd2, vregMul2, vregMul4, pregFullExe);
                    Add(vregAdd3, vregAdd, vregMul3, pregFullExe);
                    Add(vregAdd4, vregAdd3, vregAdd2, pregFullExe);
                    Add(vregAdd, vregAdd4, vregAdd1, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 512) {
            // D=512 unroll7次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

            uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

            uint64_t srcLocalInt5 = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
            uint64_t gradLocalInt5 = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

            uint64_t srcLocalInt6 = srcTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);
            uint64_t gradLocalInt6 = gradTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);

            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
            {
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
                RegTensor<float> vregMul;
                RegTensor<float> vregMul1;
                RegTensor<float> vregMul2;
                RegTensor<float> vregMul3;
                RegTensor<float> vregMul4;
                RegTensor<float> vregMul5;
                RegTensor<float> vregMul6;
                RegTensor<float> vregSrcTail;
                RegTensor<float> vregGradTail;
                RegTensor<float> vregMulTail;
                RegTensor<float> vregAdd;
                RegTensor<float> vregAdd1;
                RegTensor<float> vregAdd2;
                RegTensor<float> vregAdd3;
                RegTensor<float> vregAdd4;
                RegTensor<float> vregAdd5;
                RegTensor<float> vregReduceSum;
                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1*&)gradLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1*&)gradLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1*&)srcLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1*&)gradLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1*&)srcLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1*&)gradLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1*&)srcLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1*&)gradLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                    Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                    Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                    Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                    Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
                    Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
                    Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
                    Mul(vregMul6, vregGrad6, vregSrc6, pregFullExe);
                    Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                    Add(vregAdd, vregMul, vregMulTail, pregFullExe);
                    Add(vregAdd1, vregMul1, vregMul6, pregFullExe);
                    Add(vregAdd2, vregMul2, vregMul5, pregFullExe);
                    Add(vregAdd3, vregMul3, vregMul4, pregFullExe);
                    Add(vregAdd4, vregAdd, vregAdd3, pregFullExe);
                    Add(vregAdd, vregAdd1, vregAdd2, pregFullExe);
                    Add(vregAdd5, vregAdd, vregAdd4, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd5, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 576) {
            // D=576 unroll8次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

            uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

            uint64_t srcLocalInt5 = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
            uint64_t gradLocalInt5 = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

            uint64_t srcLocalInt6 = srcTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);
            uint64_t gradLocalInt6 = gradTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);

            uint64_t srcLocalInt7 = srcTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);
            uint64_t gradLocalInt7 = gradTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);

            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 8 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 8 * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
            {
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
                RegTensor<float> vregReduceSum;
                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1*&)gradLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1*&)gradLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1*&)srcLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1*&)gradLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1*&)srcLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1*&)gradLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1*&)srcLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1*&)gradLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc7, ((__ubuf__ T1*&)srcLocalInt7),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad7, ((__ubuf__ T1*&)gradLocalInt7),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
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
                    Add(vregAdd1, vregMul1, vregMul6, pregFullExe);
                    Add(vregAdd2, vregMul2, vregMul5, pregFullExe);
                    Add(vregAdd3, vregMul3, vregMul4, pregFullExe);
                    Add(vregAdd4, vregAdd, vregAdd3, pregFullExe);
                    Add(vregAdd4, vregAdd4, vregMul7, pregFullExe);
                    Add(vregAdd, vregAdd1, vregAdd2, pregFullExe);
                    Add(vregAdd5, vregAdd, vregAdd4, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd5, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 640) {
            // D=576 unroll8次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

            uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

            uint64_t srcLocalInt5 = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
            uint64_t gradLocalInt5 = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

            uint64_t srcLocalInt6 = srcTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);
            uint64_t gradLocalInt6 = gradTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);

            uint64_t srcLocalInt7 = srcTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);
            uint64_t gradLocalInt7 = gradTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);

            uint64_t srcLocalInt8 = srcTensor.GetPhyAddr() + fullExeSize * 8 * sizeof(T1);
            uint64_t gradLocalInt8 = gradTensor.GetPhyAddr() + fullExeSize * 8 * sizeof(T1);

            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 9 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 9 * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
            {
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
                RegTensor<float> vregReduceSum;
                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1*&)gradLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1*&)gradLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1*&)srcLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1*&)gradLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1*&)srcLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1*&)gradLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1*&)srcLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1*&)gradLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc7, ((__ubuf__ T1*&)srcLocalInt7),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad7, ((__ubuf__ T1*&)gradLocalInt7),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc8, ((__ubuf__ T1*&)srcLocalInt8),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad8, ((__ubuf__ T1*&)gradLocalInt8),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
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
                    Add(vregAdd1, vregMul1, vregMul6, pregFullExe);
                    Add(vregAdd2, vregMul2, vregMul5, pregFullExe);
                    Add(vregAdd3, vregMul3, vregMul4, pregFullExe);
                    Add(vregAdd4, vregAdd, vregAdd3, pregFullExe);
                    Add(vregAdd, vregAdd1, vregAdd2, pregFullExe);
                    Add(vregAdd5, vregAdd, vregAdd4, pregFullExe);
                    Add(vregAdd6, vregMul7, vregMul8, pregFullExe);
                    Add(vregAdd6, vregAdd6, vregAdd5, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd6, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 704) {
            // D=576 unroll8次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

            uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

            uint64_t srcLocalInt5 = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
            uint64_t gradLocalInt5 = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

            uint64_t srcLocalInt6 = srcTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);
            uint64_t gradLocalInt6 = gradTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);

            uint64_t srcLocalInt7 = srcTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);
            uint64_t gradLocalInt7 = gradTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);

            uint64_t srcLocalInt8 = srcTensor.GetPhyAddr() + fullExeSize * 8 * sizeof(T1);
            uint64_t gradLocalInt8 = gradTensor.GetPhyAddr() + fullExeSize * 8 * sizeof(T1);

            uint64_t srcLocalInt9 = srcTensor.GetPhyAddr() + fullExeSize * 9 * sizeof(T1);
            uint64_t gradLocalInt9 = gradTensor.GetPhyAddr() + fullExeSize * 9 * sizeof(T1);

            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 10 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 10 * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
            {
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
                RegTensor<float> vregReduceSum;
                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1*&)gradLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1*&)gradLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1*&)srcLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1*&)gradLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1*&)srcLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1*&)gradLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1*&)srcLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1*&)gradLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc7, ((__ubuf__ T1*&)srcLocalInt7),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad7, ((__ubuf__ T1*&)gradLocalInt7),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc8, ((__ubuf__ T1*&)srcLocalInt8),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad8, ((__ubuf__ T1*&)gradLocalInt8),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc9, ((__ubuf__ T1*&)srcLocalInt9),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad9, ((__ubuf__ T1*&)gradLocalInt9),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
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
                    Add(vregAdd1, vregMul1, vregMul6, pregFullExe);
                    Add(vregAdd2, vregMul2, vregMul5, pregFullExe);
                    Add(vregAdd3, vregMul3, vregMul4, pregFullExe);
                    Add(vregAdd4, vregAdd, vregAdd3, pregFullExe);
                    Add(vregAdd, vregAdd1, vregAdd2, pregFullExe);
                    Add(vregAdd5, vregAdd, vregAdd4, pregFullExe);
                    Add(vregAdd6, vregMul7, vregMul8, pregFullExe);
                    Add(vregAdd6, vregAdd6, vregMul9, pregFullExe);
                    Add(vregAdd6, vregAdd6, vregAdd5, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd6, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 768) {
            // D=576 unroll8次
            const uint32_t fullExeSize = 64;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

            uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

            uint64_t srcLocalInt5 = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
            uint64_t gradLocalInt5 = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

            uint64_t srcLocalInt6 = srcTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);
            uint64_t gradLocalInt6 = gradTensor.GetPhyAddr() + fullExeSize * 6 * sizeof(T1);

            uint64_t srcLocalInt7 = srcTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);
            uint64_t gradLocalInt7 = gradTensor.GetPhyAddr() + fullExeSize * 7 * sizeof(T1);

            uint64_t srcLocalInt8 = srcTensor.GetPhyAddr() + fullExeSize * 8 * sizeof(T1);
            uint64_t gradLocalInt8 = gradTensor.GetPhyAddr() + fullExeSize * 8 * sizeof(T1);

            uint64_t srcLocalInt9 = srcTensor.GetPhyAddr() + fullExeSize * 9 * sizeof(T1);
            uint64_t gradLocalInt9 = gradTensor.GetPhyAddr() + fullExeSize * 9 * sizeof(T1);

            uint64_t srcLocalInt10 = srcTensor.GetPhyAddr() + fullExeSize * 10 * sizeof(T1);
            uint64_t gradLocalInt10 = gradTensor.GetPhyAddr() + fullExeSize * 10 * sizeof(T1);

            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 11 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 11 * sizeof(T1);

            uint32_t tailSize = realN % fullExeSize;
            uint32_t reduceSize = tailSize == 0 ? fullExeSize : tailSize;

            __VEC_SCOPE__
            {
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
                RegTensor<float> vregReduceSum;
                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad, ((__ubuf__ T1*&)gradLocalInt),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad1, ((__ubuf__ T1*&)gradLocalInt1),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad2, ((__ubuf__ T1*&)gradLocalInt2),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad3, ((__ubuf__ T1*&)gradLocalInt3),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc4, ((__ubuf__ T1*&)srcLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad4, ((__ubuf__ T1*&)gradLocalInt4),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc5, ((__ubuf__ T1*&)srcLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad5, ((__ubuf__ T1*&)gradLocalInt5),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc6, ((__ubuf__ T1*&)srcLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad6, ((__ubuf__ T1*&)gradLocalInt6),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc7, ((__ubuf__ T1*&)srcLocalInt7),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad7, ((__ubuf__ T1*&)gradLocalInt7),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc8, ((__ubuf__ T1*&)srcLocalInt8),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad8, ((__ubuf__ T1*&)gradLocalInt8),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc9, ((__ubuf__ T1*&)srcLocalInt9),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad9, ((__ubuf__ T1*&)gradLocalInt9),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc10, ((__ubuf__ T1*&)srcLocalInt10),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGrad10, ((__ubuf__ T1*&)gradLocalInt10),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                          srcN);
                    DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradTail,
                                                                          ((__ubuf__ T1*&)gradLocalIntTail), srcN);
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
                    Add(vregAdd1, vregMul1, vregMul6, pregFullExe);
                    Add(vregAdd2, vregMul2, vregMul5, pregFullExe);
                    Add(vregAdd3, vregMul3, vregMul4, pregFullExe);
                    Add(vregAdd4, vregAdd, vregAdd3, pregFullExe);
                    Add(vregAdd, vregAdd1, vregAdd2, pregFullExe);
                    Add(vregAdd5, vregAdd, vregAdd4, pregFullExe);
                    Add(vregAdd6, vregMul7, vregMul8, pregFullExe);
                    Add(vregAdd7, vregMul9, vregMul10, pregFullExe);
                    Add(vregAdd7, vregAdd7, vregAdd6, pregFullExe);
                    Add(vregAdd7, vregAdd7, vregAdd5, pregFullExe);
                    ReduceSum(vregReduceSum, vregAdd7, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        }
    } else {
        uint64_t srcLocalInt = srcTensor.GetPhyAddr();
        uint64_t dstLocalInt = dstTensor.GetPhyAddr();
        uint64_t gradLocalInt = gradTensor.GetPhyAddr();

        if constexpr (srcN <= 128) {
            const uint32_t fullExeSize = srcN;
            uint32_t reduceSize = (realN + 1) >> 1;
            __VEC_SCOPE__
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
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              fullExeSize);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf, ((__ubuf__ T1*&)gradLocalInt), fullExeSize);
                        Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradHalf, pregFullExeB16);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              fullExeSize);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1*&)gradLocalInt),
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
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN <= 256) {
            const uint32_t fullExeSize = 128;
            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

            uint32_t tailSize = realN > fullExeSize ? (realN % fullExeSize) : 0;
            uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
            if (realN == fullExeSize * 2) {
                reduceSize = (fullExeSize + 1) >> 1;
            }

            __VEC_SCOPE__
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
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf,
                                                                              ((__ubuf__ T1*&)gradLocalInt), srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalfTail,
                                                                              ((__ubuf__ T1*&)srcLocalIntTail), srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalfTail,
                                                                              ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                        Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcHalfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcHalfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradHalfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradHalfTail, pregFullExeB16);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1*&)gradLocalInt),
                                                                              srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBfTail,
                                                                              ((__ubuf__ T1*&)srcLocalIntTail), srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBfTail,
                                                                              ((__ubuf__ T1*&)gradLocalIntTail), srcN);
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
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 384) {
            const uint32_t fullExeSize = 128;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);

            uint32_t tailSize = realN > fullExeSize * 2 ? (realN % fullExeSize) : 0;
            uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
            if (realN == fullExeSize * 3) {
                reduceSize = (fullExeSize + 1) >> 1;
            }

            __VEC_SCOPE__
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
                RegTensor<bfloat16_t> vregSrcBf;
                RegTensor<bfloat16_t> vregGradBf;
                RegTensor<bfloat16_t> vregSrcBf1;
                RegTensor<bfloat16_t> vregGradBf1;

                RegTensor<half> vregSrcHalfTail;
                RegTensor<half> vregGradHalfTail;
                RegTensor<bfloat16_t> vregSrcBfTail;
                RegTensor<bfloat16_t> vregGradBfTail;

                RegTensor<float> vregAdd;
                RegTensor<float> vregAdd1;
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
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf,
                                                                              ((__ubuf__ T1*&)gradLocalInt), srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf1,
                                                                              ((__ubuf__ T1*&)srcLocalInt1), srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalf1,
                                                                              ((__ubuf__ T1*&)gradLocalInt1), srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalfTail,
                                                                              ((__ubuf__ T1*&)srcLocalIntTail), srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradHalfTail,
                                                                              ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                        Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradHalf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregSrc2, vregSrcHalf1, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc3, vregSrcHalf1, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGrad2, vregGradHalf1, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad3, vregGradHalf1, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcHalfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcHalfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradHalfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradHalfTail, pregFullExeB16);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1*&)gradLocalInt),
                                                                              srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                              srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf1,
                                                                              ((__ubuf__ T1*&)gradLocalInt1), srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBfTail,
                                                                              ((__ubuf__ T1*&)srcLocalIntTail), srcN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBfTail,
                                                                              ((__ubuf__ T1*&)gradLocalIntTail), srcN);
                        Cast<float, T1, castTraitB162B32Even>(vregSrc, vregSrcBf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGrad, vregGradBf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc1, vregSrcBf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad1, vregGradBf, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregSrc2, vregSrcBf1, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGrad2, vregGradBf1, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc3, vregSrcBf1, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad3, vregGradBf1, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcBfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradBfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcBfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradBfTail, pregFullExeB16);
                    }
                    Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                    Mul(vregMul2, vregGrad1, vregSrc1, pregFullExe);
                    Mul(vregMul3, vregGrad2, vregSrc2, pregFullExe);
                    Mul(vregMul4, vregGrad3, vregSrc3, pregFullExe);
                    Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                    Mul(vregMul2Tail, vregGrad1Tail, vregSrc1Tail, pregTailExe);
                    Add(vregAdd, vregMul, vregMul2, pregFullExe);
                    Add(vregAdd1, vregMul3, vregMul4, pregFullExe);
                    Add(vregAddTail, vregMulTail, vregMul2Tail, pregTailExe);
                    Add(vregAddLast, vregAdd, vregAdd1, pregFullExe);
                    Add(vregAdd, vregAddLast, vregAddTail, pregFullExe);

                    ReduceSum(vregReduceSum, vregAdd, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 512) {
            const uint32_t fullExeSize = 128;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);

            uint32_t tailSize = realN > fullExeSize * 3 ? (realN % fullExeSize) : 0;
            uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
            if (realN == fullExeSize * 4) {
                reduceSize = (fullExeSize + 1) >> 1;
            }

            __VEC_SCOPE__
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
                RegTensor<bfloat16_t> vregSrcBf;
                RegTensor<bfloat16_t> vregGradBf;
                RegTensor<bfloat16_t> vregSrcBf1;
                RegTensor<bfloat16_t> vregGradBf1;
                RegTensor<bfloat16_t> vregSrcBf2;
                RegTensor<bfloat16_t> vregGradBf2;

                RegTensor<half> vregSrcHalfTail;
                RegTensor<half> vregGradHalfTail;
                RegTensor<bfloat16_t> vregSrcBfTail;
                RegTensor<bfloat16_t> vregGradBfTail;

                RegTensor<float> vregAdd;
                RegTensor<float> vregAdd1;
                RegTensor<float> vregAdd2;
                RegTensor<float> vregAddTail;
                RegTensor<float> vregAddTemp;
                RegTensor<float> vregAddLast;
                RegTensor<float> vregReduceSum;

                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                MaskReg pregFullExeB16 = CreateMask<T1, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    if constexpr (IsSameType<T1, half>::value) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf, ((__ubuf__ T1*&)gradLocalInt), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalf1, ((__ubuf__ T1*&)srcLocalInt1), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf1, ((__ubuf__ T1*&)gradLocalInt1), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalf2, ((__ubuf__ T1*&)srcLocalInt2), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf2, ((__ubuf__ T1*&)gradLocalInt2), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalfTail, ((__ubuf__ T1*&)srcLocalIntTail), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalfTail, ((__ubuf__ T1*&)gradLocalIntTail), HEAD_DIM_ALIGN);
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
                        Cast<float, T1, castTraitB162B32Even>(vregSrcTail, vregSrcHalfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregSrc1Tail, vregSrcHalfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Even>(vregGradTail, vregGradHalfTail, pregFullExeB16);
                        Cast<float, T1, castTraitB162B32Odd>(vregGrad1Tail, vregGradHalfTail, pregFullExeB16);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1*&)gradLocalInt),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBf1, ((__ubuf__ T1*&)gradLocalInt1), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBf2, ((__ubuf__ T1*&)gradLocalInt2), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcBfTail, ((__ubuf__ T1*&)srcLocalIntTail), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBfTail, ((__ubuf__ T1*&)gradLocalIntTail), HEAD_DIM_ALIGN);
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
                    Mul(vregMulTail, vregGradTail, vregSrcTail, pregTailExe);
                    Mul(vregMul2Tail, vregGrad1Tail, vregSrc1Tail, pregTailExe);
                    Add(vregAdd, vregMul, vregMul2, pregFullExe);
                    Add(vregAdd1, vregMul3, vregMul4, pregFullExe);
                    Add(vregAdd2, vregMul5, vregMul6, pregFullExe);
                    Add(vregAddTail, vregMulTail, vregMul2Tail, pregTailExe);
                    Add(vregAddTemp, vregAdd, vregAddTail, pregFullExe);
                    Add(vregAdd, vregAdd1, vregAdd2, pregFullExe);
                    Add(vregAddLast, vregAddTemp, vregAdd, pregFullExe);

                    ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 640) {
            const uint32_t fullExeSize = 128;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);

            uint32_t tailSize = realN > fullExeSize * 4 ? (realN % fullExeSize) : 0;
            uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
            if (realN == fullExeSize * 5) {
                reduceSize = (fullExeSize + 1) >> 1;
            }

            __VEC_SCOPE__
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
                RegTensor<float> vregAddLast;
                RegTensor<float> vregReduceSum;

                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                MaskReg pregFullExeB16 = CreateMask<T1, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    if constexpr (IsSameType<T1, half>::value) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf, ((__ubuf__ T1*&)gradLocalInt), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalf1, ((__ubuf__ T1*&)srcLocalInt1), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf1, ((__ubuf__ T1*&)gradLocalInt1), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalf2, ((__ubuf__ T1*&)srcLocalInt2), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf2, ((__ubuf__ T1*&)gradLocalInt2), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalf3, ((__ubuf__ T1*&)srcLocalInt3), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf3, ((__ubuf__ T1*&)gradLocalInt3), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalfTail, ((__ubuf__ T1*&)srcLocalIntTail), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalfTail, ((__ubuf__ T1*&)gradLocalIntTail), HEAD_DIM_ALIGN);
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
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1*&)gradLocalInt),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBf1, ((__ubuf__ T1*&)gradLocalInt1), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBf2, ((__ubuf__ T1*&)gradLocalInt2), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBf3, ((__ubuf__ T1*&)gradLocalInt3), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcBfTail, ((__ubuf__ T1*&)srcLocalIntTail), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBfTail, ((__ubuf__ T1*&)gradLocalIntTail), HEAD_DIM_ALIGN);
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
                    Add(vregAdd, vregAdd1, vregAdd2, pregFullExe);
                    Add(vregAdd, vregAdd, vregAdd3, pregFullExe);
                    Add(vregAddLast, vregAddTemp, vregAdd, pregFullExe);

                    ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        } else if constexpr (srcN == 768) {
            const uint32_t fullExeSize = 128;
            uint64_t srcLocalInt1 = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t gradLocalInt1 = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
            uint64_t srcLocalInt2 = srcTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t gradLocalInt2 = gradTensor.GetPhyAddr() + fullExeSize * 2 * sizeof(T1);
            uint64_t srcLocalInt3 = srcTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t gradLocalInt3 = gradTensor.GetPhyAddr() + fullExeSize * 3 * sizeof(T1);
            uint64_t srcLocalInt4 = srcTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t gradLocalInt4 = gradTensor.GetPhyAddr() + fullExeSize * 4 * sizeof(T1);
            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);
            uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * 5 * sizeof(T1);

            uint32_t tailSize = realN > fullExeSize * 5 ? (realN % fullExeSize) : 0;
            uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
            if (realN == fullExeSize * 6) {
                reduceSize = (fullExeSize + 1) >> 1;
            }

            __VEC_SCOPE__
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
                RegTensor<float> vregAddLast;
                RegTensor<float> vregReduceSum;

                MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
                MaskReg pregFullExeB16 = CreateMask<T1, MaskPattern::ALL>();
                UnalignReg uregReduceSum;
                MaskReg pregTailExe = UpdateMask<float>(reduceSize);

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    // 手动unroll 128个数分64个数做mul和add
                    if constexpr (IsSameType<T1, half>::value) {
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcHalf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf, ((__ubuf__ T1*&)gradLocalInt), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalf1, ((__ubuf__ T1*&)srcLocalInt1), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf1, ((__ubuf__ T1*&)gradLocalInt1), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalf2, ((__ubuf__ T1*&)srcLocalInt2), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf2, ((__ubuf__ T1*&)gradLocalInt2), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalf3, ((__ubuf__ T1*&)srcLocalInt3), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf3, ((__ubuf__ T1*&)gradLocalInt3), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalf4, ((__ubuf__ T1*&)srcLocalInt4), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalf4, ((__ubuf__ T1*&)gradLocalInt4), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcHalfTail, ((__ubuf__ T1*&)srcLocalIntTail), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradHalfTail, ((__ubuf__ T1*&)gradLocalIntTail), HEAD_DIM_ALIGN);
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
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf, ((__ubuf__ T1*&)srcLocalInt),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradBf, ((__ubuf__ T1*&)gradLocalInt),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf1, ((__ubuf__ T1*&)srcLocalInt1),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBf1, ((__ubuf__ T1*&)gradLocalInt1), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf2, ((__ubuf__ T1*&)srcLocalInt2),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBf2, ((__ubuf__ T1*&)gradLocalInt2), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf3, ((__ubuf__ T1*&)srcLocalInt3),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBf3, ((__ubuf__ T1*&)gradLocalInt3), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcBf4, ((__ubuf__ T1*&)srcLocalInt4),
                                                                              HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBf4, ((__ubuf__ T1*&)gradLocalInt4), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregSrcBfTail, ((__ubuf__ T1*&)srcLocalIntTail), HEAD_DIM_ALIGN);
                        DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                            vregGradBfTail, ((__ubuf__ T1*&)gradLocalIntTail), HEAD_DIM_ALIGN);
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
                    Add(vregAdd, vregAdd1, vregAdd2, pregFullExe);
                    Add(vregAdd4, vregAdd4, vregAdd3, pregFullExe);
                    Add(vregAdd, vregAdd, vregAdd4, pregFullExe);
                    Add(vregAddLast, vregAddTemp, vregAdd, pregFullExe);

                    ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
                    DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt),
                                                                                    vregReduceSum, uregReduceSum, 1);
                }
                vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
            }
        }
    }
}
#else
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCast(const LocalTensor<T>& dstTensor, const LocalTensor<T1>& gradTensor,
                                              const LocalTensor<T1>& srcTensor, uint32_t srcM, uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // MY_SOFTMAX_GRAD_CAST_INTERFACE_DAVID_H
