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
 * \file vf_anti_quant_softmax_grad_front_cast_david.h
 */
#ifndef MY_ANTI_QUANT_SOFTMAX_GRAD_CAST_INTERFACE_DAVID_H_
#define MY_ANTI_QUANT_SOFTMAX_GRAD_CAST_INTERFACE_DAVID_H_
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
constexpr static AscendC::MicroAPI::CastTrait castTraitB82B32Three = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitB82B32Two = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitB82B32One = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr static AscendC::MicroAPI::CastTrait castTraitB82B32Zero = {
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
template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void MyAntiQuantSoftmaxGradFrontCast(const LocalTensor<T>& dstTensor,
                                                       const LocalTensor<T1>& gradTensor,
                                                       const LocalTensor<T1>& srcTensor, float scaleOut, float scaleDx,
                                                       uint32_t srcM, uint32_t realN = srcN)
{
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t gradLocalInt = gradTensor.GetPhyAddr();

    if constexpr (srcN <= 256) {
        uint32_t fullExeSize = srcN;
        uint32_t reduceSize = (realN + 3) >> 2;
        __VEC_SCOPE__
        {
            RegTensor<T> vregSrc;
            RegTensor<T> vregGrad;
            RegTensor<T> vregMul;
            RegTensor<T> vregSrc1;
            RegTensor<T> vregGrad1;
            RegTensor<T> vregMul1;
            RegTensor<T> vregSrc2;
            RegTensor<T> vregGrad2;
            RegTensor<T> vregMul2;
            RegTensor<T> vregSrc3;
            RegTensor<T> vregGrad3;
            RegTensor<T> vregMul3;

            RegTensor<T1> vregSrcFp8;
            RegTensor<T1> vregGradFp8;

            RegTensor<T> vregAdd;
            RegTensor<T> vregAdd2;
            RegTensor<T> vregAddLast;
            RegTensor<T> vregReduceSum;

            MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
            MaskReg pregFullExeB8 = UpdateMask<float8_e5m2_t>(fullExeSize);
            UnalignReg uregReduceSum;
            MaskReg pregTailExe = UpdateMask<T>(reduceSize);
            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                // 手动unroll 128个数分成4组做mul和add
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8, ((__ubuf__ T1*&)srcLocalInt), srcN);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradFp8, ((__ubuf__ T1*&)gradLocalInt), srcN);
                Cast<T, T1, castTraitB82B32Zero>(vregSrc, vregSrcFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32One>(vregSrc1, vregSrcFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Two>(vregSrc2, vregSrcFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Three>(vregSrc3, vregSrcFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Zero>(vregGrad, vregGradFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32One>(vregGrad1, vregGradFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Two>(vregGrad2, vregGradFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Three>(vregGrad3, vregGradFp8, pregFullExeB8);
                Muls(vregSrc, vregSrc, scaleDx, pregTailExe);
                Muls(vregGrad, vregGrad, scaleOut, pregTailExe);
                Muls(vregSrc1, vregSrc1, scaleDx, pregTailExe);
                Muls(vregGrad1, vregGrad1, scaleOut, pregTailExe);
                Muls(vregSrc2, vregSrc2, scaleDx, pregTailExe);
                Muls(vregGrad2, vregGrad2, scaleOut, pregTailExe);
                Muls(vregSrc3, vregSrc3, scaleDx, pregTailExe);
                Muls(vregGrad3, vregGrad3, scaleOut, pregTailExe);
                Mul(vregMul, vregGrad, vregSrc, pregTailExe);
                Mul(vregMul1, vregGrad1, vregSrc1, pregTailExe);
                Mul(vregMul2, vregGrad2, vregSrc2, pregTailExe);
                Mul(vregMul3, vregGrad3, vregSrc3, pregTailExe);
                Add(vregAdd, vregMul, vregMul1, pregTailExe);
                Add(vregAdd2, vregMul2, vregMul3, pregTailExe);
                Add(vregAddLast, vregAdd, vregAdd2, pregTailExe);
                ReduceSum(vregReduceSum, vregAddLast, pregTailExe);
                vstus(uregReduceSum, 1, vregReduceSum, ((__ubuf__ float*&)dstLocalInt), POST_UPDATE);
            }
            vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
        }
    } else if constexpr (srcN <= 512) {
        uint32_t fullExeSize = 256;
        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * sizeof(T1);
        uint64_t gradLocalIntTail = gradTensor.GetPhyAddr() + fullExeSize * sizeof(T1);

        uint32_t tailSize = realN % fullExeSize;
        uint32_t reduceSize = tailSize == 0 ? fullExeSize : (tailSize + 3) >> 2;

        __VEC_SCOPE__
        {
            RegTensor<T> vregSrc;
            RegTensor<T> vregGrad;
            RegTensor<T> vregMul;
            RegTensor<T> vregSrc1;
            RegTensor<T> vregGrad1;
            RegTensor<T> vregMul1;
            RegTensor<T> vregSrc2;
            RegTensor<T> vregGrad2;
            RegTensor<T> vregMul2;
            RegTensor<T> vregSrc3;
            RegTensor<T> vregGrad3;
            RegTensor<T> vregMul3;
            RegTensor<T> vregSrcTail;
            RegTensor<T> vregGradTail;
            RegTensor<T> vregMul4;
            RegTensor<T> vregSrcTail1;
            RegTensor<T> vregGradTail1;
            RegTensor<T> vregMul5;
            RegTensor<T> vregSrcTail2;
            RegTensor<T> vregGradTail2;
            RegTensor<T> vregMul6;
            RegTensor<T> vregSrcTail3;
            RegTensor<T> vregGradTail3;
            RegTensor<T> vregMul7;

            RegTensor<T1> vregSrcFp8;
            RegTensor<T1> vregGradFp8;
            RegTensor<T1> vregSrcFp8Tail;
            RegTensor<T1> vregGradFp8Tail;

            RegTensor<T> vregAdd;
            RegTensor<T> vregAdd2;
            RegTensor<T> vregAdd3;
            RegTensor<T> vregAdd4;
            RegTensor<T> vregAdd5;
            RegTensor<T> vregAdd6;
            RegTensor<T> vregAddLast;
            RegTensor<T> vregReduceSum;

            MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
            MaskReg pregFullExeB8 = CreateMask<T1, MaskPattern::ALL>();
            UnalignReg uregReduceSum;
            MaskReg pregTailExe = UpdateMask<T>(reduceSize);

            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                // 手动unroll 128个数分成4组做mul和add
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8, ((__ubuf__ T1*&)srcLocalInt), 512);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradFp8, ((__ubuf__ T1*&)gradLocalInt), 512);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8Tail, ((__ubuf__ T1*&)srcLocalIntTail),
                                                                      512);
                DataCopy<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradFp8Tail,
                                                                      ((__ubuf__ T1*&)gradLocalIntTail), 512);
                Cast<T, T1, castTraitB82B32Zero>(vregSrc, vregSrcFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32One>(vregSrc1, vregSrcFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Two>(vregSrc2, vregSrcFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Three>(vregSrc3, vregSrcFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Zero>(vregGrad, vregGradFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32One>(vregGrad1, vregGradFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Two>(vregGrad2, vregGradFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Three>(vregGrad3, vregGradFp8, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Zero>(vregSrcTail, vregSrcFp8Tail, pregFullExeB8);
                Cast<T, T1, castTraitB82B32One>(vregSrcTail1, vregSrcFp8Tail, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Two>(vregSrcTail2, vregSrcFp8Tail, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Three>(vregSrcTail3, vregSrcFp8Tail, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Zero>(vregGradTail, vregGradFp8Tail, pregFullExeB8);
                Cast<T, T1, castTraitB82B32One>(vregGradTail1, vregGradFp8Tail, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Two>(vregGradTail2, vregGradFp8Tail, pregFullExeB8);
                Cast<T, T1, castTraitB82B32Three>(vregGradTail3, vregGradFp8Tail, pregFullExeB8);
                Muls(vregSrc, vregSrc, scaleDx, pregFullExe);
                Muls(vregGrad, vregGrad, scaleOut, pregFullExe);
                Muls(vregSrc1, vregSrc1, scaleDx, pregFullExe);
                Muls(vregGrad1, vregGrad1, scaleOut, pregFullExe);
                Muls(vregSrc2, vregSrc2, scaleDx, pregFullExe);
                Muls(vregGrad2, vregGrad2, scaleOut, pregFullExe);
                Muls(vregSrc3, vregSrc3, scaleDx, pregFullExe);
                Muls(vregGrad3, vregGrad3, scaleOut, pregFullExe);
                Muls(vregSrcTail, vregSrcTail, scaleDx, pregTailExe);
                Muls(vregGradTail, vregGradTail, scaleOut, pregTailExe);
                Muls(vregSrcTail1, vregSrcTail1, scaleDx, pregTailExe);
                Muls(vregGradTail1, vregGradTail1, scaleOut, pregTailExe);
                Muls(vregSrcTail2, vregSrcTail2, scaleDx, pregTailExe);
                Muls(vregGradTail2, vregGradTail2, scaleOut, pregTailExe);
                Muls(vregSrcTail3, vregSrcTail3, scaleDx, pregTailExe);
                Muls(vregGradTail3, vregGradTail3, scaleOut, pregTailExe);
                Mul(vregMul, vregGrad, vregSrc, pregFullExe);
                Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
                Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
                Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
                Mul(vregMul4, vregGradTail, vregSrcTail, pregTailExe);
                Mul(vregMul5, vregGradTail1, vregSrcTail1, pregTailExe);
                Mul(vregMul6, vregGradTail2, vregSrcTail2, pregTailExe);
                Mul(vregMul7, vregGradTail3, vregSrcTail3, pregTailExe);
                Add(vregAdd, vregMul, vregMul1, pregFullExe);
                Add(vregAdd2, vregMul2, vregMul3, pregFullExe);
                Add(vregAdd3, vregMul4, vregMul5, pregFullExe);
                Add(vregAdd4, vregMul6, vregMul7, pregFullExe);
                Add(vregAdd5, vregAdd, vregAdd2, pregFullExe);
                Add(vregAdd6, vregAdd3, vregAdd4, pregFullExe);
                Add(vregAddLast, vregAdd5, vregAdd6, pregFullExe);
                ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
                vstus(uregReduceSum, 1, vregReduceSum, ((__ubuf__ float*&)dstLocalInt), POST_UPDATE);
            }
            vstas(uregReduceSum, ((__ubuf__ float*&)dstLocalInt), 0, POST_UPDATE);
        }
    }
}
#else
template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void MyAntiQuantSoftmaxGradFrontCast(const LocalTensor<T>& dstTensor,
                                                       const LocalTensor<T1>& gradTensor,
                                                       const LocalTensor<T1>& srcTensor, float scaleOut, float scaleDx,
                                                       uint32_t srcM, uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // _MY_ANTI_QUANT_SOFTMAX_GRAD_CAST_INTERFACE_DAVID_H_
