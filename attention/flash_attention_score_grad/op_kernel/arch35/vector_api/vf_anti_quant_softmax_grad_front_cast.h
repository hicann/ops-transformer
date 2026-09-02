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
 * \file vf_anti_quant_softmax_grad_front_cast.h
 */
#ifndef MY_ANTI_QUANT_SOFTMAX_GRAD_CAST_INTERFACE_H_
#define MY_ANTI_QUANT_SOFTMAX_GRAD_CAST_INTERFACE_H_
#include "kernel_tensor.h"
#include "vf_common_utils.h"

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
template <typename T1, typename T, typename OUTDTYPE, uint32_t srcN>
__simd_vf__ inline void AntiQuantSoftmaxGradVF128(uint64_t dstLocalInt, uint64_t gradLocalInt, uint64_t srcLocalInt,
                                                  float scaleDx, uint32_t srcM, uint32_t reduceSize)
{
    RegTensor<T1> vregSrcFp8;
    RegTensor<OUTDTYPE> vregGradB16;

    RegTensor<T> vregSrc;
    RegTensor<T> vregSrc1;
    RegTensor<T> vregSrc2;
    RegTensor<T> vregSrc3;
    RegTensor<T> vregGrad;
    RegTensor<T> vregGrad1;
    RegTensor<T> vregMul;
    RegTensor<T> vregMul1;
    RegTensor<T> vregAdd;
    RegTensor<T> vregReduceSum;
    UnalignReg uregReduceSum;
    MaskReg pregFullExeB8 = CreateMask<T1, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<OUTDTYPE, MaskPattern::ALL>();
    MaskReg pregTailExe = UpdateMask<T>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8, ((__ubuf__ T1 *&)srcLocalInt), srcN);
        LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradB16, ((__ubuf__ OUTDTYPE *&)gradLocalInt),
                                                                     srcN);
        Cast<T, T1, castTraitB82B32Zero>(vregSrc, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32One>(vregSrc1, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32Two>(vregSrc2, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32Three>(vregSrc3, vregSrcFp8, pregFullExeB8);
        Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad, vregGradB16, pregFullExeB16);
        Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad1, vregGradB16, pregFullExeB16);
        Interleave(vregSrc, vregSrc2, vregSrc, vregSrc2);
        Interleave(vregSrc1, vregSrc3, vregSrc1, vregSrc3);
        Muls(vregSrc, vregSrc, scaleDx, pregTailExe);
        Muls(vregSrc1, vregSrc1, scaleDx, pregTailExe);
        Mul(vregMul, vregGrad, vregSrc, pregTailExe);
        Mul(vregMul1, vregGrad1, vregSrc1, pregTailExe);
        Add(vregAdd, vregMul, vregMul1, pregTailExe);
        ReduceSum(vregReduceSum, vregAdd, pregTailExe);
        vstus(uregReduceSum, 1, vregReduceSum, ((__ubuf__ float *&)dstLocalInt), POST_UPDATE);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, typename OUTDTYPE, uint32_t srcN>
__simd_vf__ inline void AntiQuantSoftmaxGradVF256(uint64_t dstLocalInt, uint64_t gradLocalInt, uint64_t srcLocalInt,
                                                  float scaleDx, uint32_t srcM, uint32_t reduceSize)
{
    const uint32_t fullExeSize = 128;
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * sizeof(OUTDTYPE);

    RegTensor<T1> vregSrcFp8;
    RegTensor<OUTDTYPE> vregGradB16;
    RegTensor<OUTDTYPE> vregGradB16Tail;

    RegTensor<T> vregSrc;
    RegTensor<T> vregSrc1;
    RegTensor<T> vregSrc2;
    RegTensor<T> vregSrc3;
    RegTensor<T> vregGrad;
    RegTensor<T> vregGrad1;
    RegTensor<T> vregGrad2;
    RegTensor<T> vregGrad3;

    RegTensor<T> vregMul;
    RegTensor<T> vregMul1;
    RegTensor<T> vregMul2;
    RegTensor<T> vregMul3;
    RegTensor<T> vregAdd;
    RegTensor<T> vregAdd2;
    RegTensor<T> vregAddLast;
    RegTensor<T> vregReduceSum;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregFullExeB8 = CreateMask<T1, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<OUTDTYPE, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<float>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8, ((__ubuf__ T1 *&)srcLocalInt), srcN);
        LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradB16, ((__ubuf__ OUTDTYPE *&)gradLocalInt),
                                                                     srcN);
        LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradB16Tail,
                                                                     ((__ubuf__ OUTDTYPE *&)gradLocalIntTail), srcN);

        Cast<T, T1, castTraitB82B32Zero>(vregSrc, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32One>(vregSrc1, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32Two>(vregSrc2, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32Three>(vregSrc3, vregSrcFp8, pregFullExeB8);
        Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad, vregGradB16, pregFullExeB16);
        Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad1, vregGradB16, pregFullExeB16);
        Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad2, vregGradB16Tail, pregFullExeB16);
        Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad3, vregGradB16Tail, pregFullExeB16);
        Interleave(vregSrc, vregSrc2, vregSrc, vregSrc2);
        Interleave(vregSrc1, vregSrc3, vregSrc1, vregSrc3);
        Muls(vregSrc, vregSrc, scaleDx, pregFullExe);
        Muls(vregSrc1, vregSrc1, scaleDx, pregFullExe);
        Muls(vregSrc2, vregSrc2, scaleDx, pregTailExe);
        Muls(vregSrc3, vregSrc3, scaleDx, pregTailExe);
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMul2, vregGrad2, vregSrc2, pregTailExe);
        Mul(vregMul3, vregGrad3, vregSrc3, pregTailExe);
        Add(vregAdd, vregMul, vregMul1, pregFullExe);
        Add(vregAdd2, vregMul2, vregMul3, pregFullExe);
        Add(vregAddLast, vregAdd, vregAdd2, pregFullExe);
        ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
        vstus(uregReduceSum, 1, vregReduceSum, ((__ubuf__ float *&)dstLocalInt), POST_UPDATE);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, typename OUTDTYPE, uint32_t srcN>
__simd_vf__ inline void AntiQuantSoftmaxGradVF384(uint64_t dstLocalInt, uint64_t gradLocalInt, uint64_t srcLocalInt,
                                                  float scaleDx, uint32_t srcM, uint32_t reduceSize)
{
    const uint32_t fullExeSize = 128;
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(OUTDTYPE);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 2 * sizeof(OUTDTYPE);

    RegTensor<T1> vregSrcFp8;
    RegTensor<T1> vregSrcFp8Tail;
    RegTensor<OUTDTYPE> vregGradDtype;
    RegTensor<OUTDTYPE> vregGradDtype1;
    RegTensor<OUTDTYPE> vregGradDtypeTail;

    RegTensor<T> vregSrc;
    RegTensor<T> vregSrc1;
    RegTensor<T> vregSrc2;
    RegTensor<T> vregSrc3;
    RegTensor<T> vregSrcTail;
    RegTensor<T> vregSrcTail1;
    RegTensor<T> vregSrcTail2;
    RegTensor<T> vregSrcTail3;

    RegTensor<T> vregGrad;
    RegTensor<T> vregGrad1;
    RegTensor<T> vregGrad2;
    RegTensor<T> vregGrad3;
    RegTensor<T> vregGradTail;
    RegTensor<T> vregGradTail1;

    RegTensor<T> vregMul;
    RegTensor<T> vregMul1;
    RegTensor<T> vregMul2;
    RegTensor<T> vregMul3;
    RegTensor<T> vregMul4;
    RegTensor<T> vregMul5;

    RegTensor<T> vregAdd;
    RegTensor<T> vregAdd2;
    RegTensor<T> vregAdd3;
    RegTensor<T> vregAdd4;
    RegTensor<T> vregAddLast;
    RegTensor<T> vregReduceSum;

    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregFullExeB8 = CreateMask<T1, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<OUTDTYPE, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<T>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8, ((__ubuf__ T1 *&)srcLocalInt), 512);
        LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8Tail, ((__ubuf__ T1 *&)srcLocalIntTail), 512);
        LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype,
                                                                     ((__ubuf__ OUTDTYPE *&)gradLocalInt), 512);
        LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype1,
                                                                     ((__ubuf__ OUTDTYPE *&)gradLocalInt1), 512);
        LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtypeTail,
                                                                     ((__ubuf__ OUTDTYPE *&)gradLocalIntTail), 512);

        Cast<T, T1, castTraitB82B32Zero>(vregSrc, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32One>(vregSrc1, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32Two>(vregSrc2, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32Three>(vregSrc3, vregSrcFp8, pregFullExeB8);
        Cast<T, T1, castTraitB82B32Zero>(vregSrcTail, vregSrcFp8Tail, pregFullExeB8);
        Cast<T, T1, castTraitB82B32One>(vregSrcTail1, vregSrcFp8Tail, pregFullExeB8);
        Cast<T, T1, castTraitB82B32Two>(vregSrcTail2, vregSrcFp8Tail, pregFullExeB8);
        Cast<T, T1, castTraitB82B32Three>(vregSrcTail3, vregSrcFp8Tail, pregFullExeB8);

        Interleave(vregSrc, vregSrc2, vregSrc, vregSrc2);
        Interleave(vregSrc1, vregSrc3, vregSrc1, vregSrc3);
        Interleave(vregSrcTail, vregSrcTail2, vregSrcTail, vregSrcTail2);
        Interleave(vregSrcTail1, vregSrcTail3, vregSrcTail1, vregSrcTail3);

        Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad, vregGradDtype, pregFullExeB16);
        Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad1, vregGradDtype, pregFullExeB16);
        Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad2, vregGradDtype1, pregFullExeB16);
        Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad3, vregGradDtype1, pregFullExeB16);
        Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGradTail, vregGradDtypeTail, pregFullExeB16);
        Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGradTail1, vregGradDtypeTail, pregFullExeB16);

        Muls(vregSrc, vregSrc, scaleDx, pregFullExe);
        Muls(vregSrc1, vregSrc1, scaleDx, pregFullExe);
        Muls(vregSrc2, vregSrc2, scaleDx, pregFullExe);
        Muls(vregSrc3, vregSrc3, scaleDx, pregFullExe);
        Muls(vregSrcTail, vregSrcTail, scaleDx, pregTailExe);
        Muls(vregSrcTail1, vregSrcTail1, scaleDx, pregTailExe);
        Mul(vregMul, vregGrad, vregSrc, pregFullExe);
        Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
        Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
        Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
        Mul(vregMul4, vregGradTail, vregSrcTail, pregTailExe);
        Mul(vregMul5, vregGradTail1, vregSrcTail1, pregTailExe);
        Add(vregAdd, vregMul, vregMul1, pregFullExe);
        Add(vregAdd2, vregMul2, vregMul3, pregFullExe);
        Add(vregAdd3, vregMul4, vregMul5, pregFullExe);
        Add(vregAdd4, vregAdd, vregAdd2, pregFullExe);
        Add(vregAddLast, vregAdd3, vregAdd4, pregFullExe);
        ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
        vstus(uregReduceSum, 1, vregReduceSum, ((__ubuf__ float *&)dstLocalInt), POST_UPDATE);
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, typename OUTDTYPE, uint32_t srcN, bool OUTDTYPE_IS_B16>
__simd_vf__ inline void AntiQuantSoftmaxGradVF512(uint64_t dstLocalInt, uint64_t gradLocalInt, uint64_t srcLocalInt,
                                                  float scaleDx, uint32_t srcM, uint32_t reduceSize)
{
    const uint32_t fullExeSize = 128;
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(OUTDTYPE);
    uint64_t gradLocalInt2 = gradLocalInt + fullExeSize * 2 * sizeof(OUTDTYPE);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 3 * sizeof(OUTDTYPE);

    RegTensor<T> vregSrc;
    RegTensor<T> vregSrc1;
    RegTensor<T> vregSrc2;
    RegTensor<T> vregSrc3;
    RegTensor<T> vregSrcTail;
    RegTensor<T> vregSrcTail1;
    RegTensor<T> vregSrcTail2;
    RegTensor<T> vregSrcTail3;

    RegTensor<T> vregGrad;
    RegTensor<T> vregGrad1;
    RegTensor<T> vregGrad2;
    RegTensor<T> vregGrad3;
    RegTensor<T> vregGrad4;
    RegTensor<T> vregGrad5;
    RegTensor<T> vregGradTail;
    RegTensor<T> vregGradTail1;

    RegTensor<T> vregMul;
    RegTensor<T> vregMul1;
    RegTensor<T> vregMul2;
    RegTensor<T> vregMul3;
    RegTensor<T> vregMul4;
    RegTensor<T> vregMul5;
    RegTensor<T> vregMul6;
    RegTensor<T> vregMul7;

    RegTensor<T1> vregSrcFp8;
    RegTensor<T1> vregSrcFp8Tail;
    RegTensor<OUTDTYPE> vregGradDtype;
    RegTensor<OUTDTYPE> vregGradDtype1;
    RegTensor<OUTDTYPE> vregGradDtype2;
    RegTensor<OUTDTYPE> vregGradDtypeTail;

    RegTensor<T> vregAdd;
    RegTensor<T> vregAdd1;
    RegTensor<T> vregAdd2;
    RegTensor<T> vregAdd3;
    RegTensor<T> vregAdd4;
    RegTensor<T> vregAdd5;
    RegTensor<T> vregAddLast;
    RegTensor<T> vregReduceSum;

    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregFullExeB8 = CreateMask<T1, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<OUTDTYPE, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<T>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        if constexpr (OUTDTYPE_IS_B16) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8, ((__ubuf__ T1 *&)srcLocalInt), 512);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8Tail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                                   512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype1,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt1), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype2,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt2), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtypeTail,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalIntTail), 512);
            Cast<T, T1, castTraitB82B32Zero>(vregSrc, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32One>(vregSrc1, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Two>(vregSrc2, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Three>(vregSrc3, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Zero>(vregSrcTail, vregSrcFp8Tail, pregFullExeB8);
            Cast<T, T1, castTraitB82B32One>(vregSrcTail1, vregSrcFp8Tail, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Two>(vregSrcTail2, vregSrcFp8Tail, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Three>(vregSrcTail3, vregSrcFp8Tail, pregFullExeB8);
            Interleave(vregSrc, vregSrc2, vregSrc, vregSrc2);
            Interleave(vregSrc1, vregSrc3, vregSrc1, vregSrc3);
            Interleave(vregSrcTail, vregSrcTail2, vregSrcTail, vregSrcTail2);
            Interleave(vregSrcTail1, vregSrcTail3, vregSrcTail1, vregSrcTail3);
            Muls(vregSrc, vregSrc, scaleDx, pregFullExe);
            Muls(vregSrc1, vregSrc1, scaleDx, pregFullExe);
            Muls(vregSrc2, vregSrc2, scaleDx, pregFullExe);
            Muls(vregSrc3, vregSrc3, scaleDx, pregFullExe);
            Muls(vregSrcTail, vregSrcTail, scaleDx, pregFullExe);
            Muls(vregSrcTail1, vregSrcTail1, scaleDx, pregFullExe);
            Muls(vregSrcTail2, vregSrcTail2, scaleDx, pregTailExe);
            Muls(vregSrcTail3, vregSrcTail3, scaleDx, pregTailExe);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad, vregGradDtype, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad1, vregGradDtype, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad2, vregGradDtype1, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad3, vregGradDtype1, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad4, vregGradDtype2, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad5, vregGradDtype2, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGradTail, vregGradDtypeTail, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGradTail1, vregGradDtypeTail, pregFullExeB16);

            Mul(vregMul, vregGrad, vregSrc, pregFullExe);
            Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
            Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
            Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
            Mul(vregMul4, vregGrad4, vregSrcTail, pregFullExe);
            Mul(vregMul5, vregGrad5, vregSrcTail1, pregFullExe);
            Mul(vregMul6, vregGradTail, vregSrcTail2, pregTailExe);
            Mul(vregMul7, vregGradTail1, vregSrcTail3, pregTailExe);

            Add(vregAdd, vregMul, vregMul1, pregFullExe);
            Add(vregAdd1, vregMul2, vregMul3, pregFullExe);
            Add(vregAdd2, vregMul4, vregMul5, pregFullExe);
            Add(vregAdd3, vregMul6, vregMul7, pregFullExe);
            Add(vregAdd4, vregAdd, vregAdd1, pregFullExe);
            Add(vregAdd5, vregAdd2, vregAdd3, pregFullExe);
            Add(vregAddLast, vregAdd4, vregAdd5, pregFullExe);

            ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
            vstus(uregReduceSum, 1, vregReduceSum, ((__ubuf__ float *&)dstLocalInt), POST_UPDATE);
        }
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, typename OUTDTYPE, uint32_t srcN, bool OUTDTYPE_IS_B16>
__simd_vf__ inline void AntiQuantSoftmaxGradVF640(uint64_t dstLocalInt, uint64_t gradLocalInt, uint64_t srcLocalInt,
                                                  float scaleDx, uint32_t srcM, uint32_t reduceSize)
{
    const uint32_t fullExeSize = 128;
    uint64_t srcLocalInt1 = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(OUTDTYPE);
    uint64_t gradLocalInt2 = gradLocalInt + fullExeSize * 2 * sizeof(OUTDTYPE);
    uint64_t gradLocalInt3 = gradLocalInt + fullExeSize * 3 * sizeof(OUTDTYPE);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 4 * sizeof(OUTDTYPE);

    RegTensor<T> vregSrc;
    RegTensor<T> vregSrc1;
    RegTensor<T> vregSrc2;
    RegTensor<T> vregSrc3;
    RegTensor<T> vregSrc4;
    RegTensor<T> vregSrc5;
    RegTensor<T> vregSrc6;
    RegTensor<T> vregSrc7;
    RegTensor<T> vregSrcTail;
    RegTensor<T> vregSrcTail1;
    RegTensor<T> vregSrcTail2;
    RegTensor<T> vregSrcTail3;

    RegTensor<T> vregGrad;
    RegTensor<T> vregGrad1;
    RegTensor<T> vregGrad2;
    RegTensor<T> vregGrad3;
    RegTensor<T> vregGrad4;
    RegTensor<T> vregGrad5;
    RegTensor<T> vregGrad6;
    RegTensor<T> vregGrad7;
    RegTensor<T> vregGradTail;
    RegTensor<T> vregGradTail1;

    RegTensor<T> vregMul;
    RegTensor<T> vregMul1;
    RegTensor<T> vregMul2;
    RegTensor<T> vregMul3;
    RegTensor<T> vregMul4;
    RegTensor<T> vregMul5;
    RegTensor<T> vregMul6;
    RegTensor<T> vregMul7;
    RegTensor<T> vregMul8;
    RegTensor<T> vregMul9;

    RegTensor<T1> vregSrcFp8;
    RegTensor<T1> vregSrcFp81;
    RegTensor<T1> vregSrcFp8Tail;
    RegTensor<OUTDTYPE> vregGradDtype;
    RegTensor<OUTDTYPE> vregGradDtype1;
    RegTensor<OUTDTYPE> vregGradDtype2;
    RegTensor<OUTDTYPE> vregGradDtype3;
    RegTensor<OUTDTYPE> vregGradDtypeTail;

    RegTensor<T> vregAdd;
    RegTensor<T> vregAdd1;
    RegTensor<T> vregAdd2;
    RegTensor<T> vregAdd3;
    RegTensor<T> vregAdd4;
    RegTensor<T> vregAdd5;
    RegTensor<T> vregAdd6;
    RegTensor<T> vregAddLast;
    RegTensor<T> vregReduceSum;

    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregFullExeB8 = CreateMask<T1, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<OUTDTYPE, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<T>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        if constexpr (OUTDTYPE_IS_B16) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8, ((__ubuf__ T1 *&)srcLocalInt), 512);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp81, ((__ubuf__ T1 *&)srcLocalInt1), 512);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8Tail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                                   512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype1,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt1), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype2,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt2), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype3,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt3), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtypeTail,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalIntTail), 512);

            Cast<T, T1, castTraitB82B32Zero>(vregSrc, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32One>(vregSrc1, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Two>(vregSrc2, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Three>(vregSrc3, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Zero>(vregSrc4, vregSrcFp81, pregFullExeB8);
            Cast<T, T1, castTraitB82B32One>(vregSrc5, vregSrcFp81, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Two>(vregSrc6, vregSrcFp81, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Three>(vregSrc7, vregSrcFp81, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Zero>(vregSrcTail, vregSrcFp8Tail, pregFullExeB8);
            Cast<T, T1, castTraitB82B32One>(vregSrcTail1, vregSrcFp8Tail, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Two>(vregSrcTail2, vregSrcFp8Tail, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Three>(vregSrcTail3, vregSrcFp8Tail, pregFullExeB8);

            Interleave(vregSrc, vregSrc2, vregSrc, vregSrc2);
            Interleave(vregSrc1, vregSrc3, vregSrc1, vregSrc3);
            Interleave(vregSrc4, vregSrc6, vregSrc4, vregSrc6);
            Interleave(vregSrc5, vregSrc7, vregSrc5, vregSrc7);
            Interleave(vregSrcTail, vregSrcTail2, vregSrcTail, vregSrcTail2);
            Interleave(vregSrcTail1, vregSrcTail3, vregSrcTail1, vregSrcTail3);

            Muls(vregSrc, vregSrc, scaleDx, pregFullExe);
            Muls(vregSrc1, vregSrc1, scaleDx, pregFullExe);
            Muls(vregSrc2, vregSrc2, scaleDx, pregFullExe);
            Muls(vregSrc3, vregSrc3, scaleDx, pregFullExe);
            Muls(vregSrc4, vregSrc4, scaleDx, pregFullExe);
            Muls(vregSrc5, vregSrc5, scaleDx, pregFullExe);
            Muls(vregSrc6, vregSrc6, scaleDx, pregFullExe);
            Muls(vregSrc7, vregSrc7, scaleDx, pregFullExe);
            Muls(vregSrcTail, vregSrcTail, scaleDx, pregTailExe);
            Muls(vregSrcTail1, vregSrcTail1, scaleDx, pregTailExe);

            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad, vregGradDtype, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad1, vregGradDtype, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad2, vregGradDtype1, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad3, vregGradDtype1, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad4, vregGradDtype2, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad5, vregGradDtype2, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad6, vregGradDtype3, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad7, vregGradDtype3, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGradTail, vregGradDtypeTail, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGradTail1, vregGradDtypeTail, pregFullExeB16);

            Mul(vregMul, vregGrad, vregSrc, pregFullExe);
            Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
            Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
            Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
            Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
            Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
            Mul(vregMul6, vregGrad6, vregSrc6, pregFullExe);
            Mul(vregMul7, vregGrad7, vregSrc7, pregFullExe);
            Mul(vregMul8, vregGradTail, vregSrcTail, pregTailExe);
            Mul(vregMul9, vregGradTail1, vregSrcTail1, pregTailExe);

            Add(vregAdd, vregMul, vregMul1, pregFullExe);
            Add(vregAdd1, vregMul2, vregMul3, pregFullExe);
            Add(vregAdd2, vregMul4, vregMul5, pregFullExe);
            Add(vregAdd3, vregMul6, vregMul7, pregFullExe);
            Add(vregAdd4, vregMul8, vregMul9, pregFullExe);

            Add(vregAdd5, vregAdd, vregAdd1, pregFullExe);
            Add(vregAdd6, vregAdd2, vregAdd3, pregFullExe);
            Add(vregAdd, vregAdd4, vregAdd5, pregFullExe);
            Add(vregAddLast, vregAdd6, vregAdd, pregFullExe);
            ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
            vstus(uregReduceSum, 1, vregReduceSum, ((__ubuf__ float *&)dstLocalInt), POST_UPDATE);
        }
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, typename OUTDTYPE, uint32_t srcN, bool OUTDTYPE_IS_B16>
__simd_vf__ inline void AntiQuantSoftmaxGradVF768(uint64_t dstLocalInt, uint64_t gradLocalInt, uint64_t srcLocalInt,
                                                  float scaleDx, uint32_t srcM, uint32_t reduceSize)
{
    const uint32_t fullExeSize = 128;
    uint64_t srcLocalInt1 = srcLocalInt + fullExeSize * 2 * sizeof(T1);
    uint64_t srcLocalIntTail = srcLocalInt + fullExeSize * 4 * sizeof(T1);
    uint64_t gradLocalInt1 = gradLocalInt + fullExeSize * sizeof(OUTDTYPE);
    uint64_t gradLocalInt2 = gradLocalInt + fullExeSize * 2 * sizeof(OUTDTYPE);
    uint64_t gradLocalInt3 = gradLocalInt + fullExeSize * 3 * sizeof(OUTDTYPE);
    uint64_t gradLocalInt4 = gradLocalInt + fullExeSize * 4 * sizeof(OUTDTYPE);
    uint64_t gradLocalIntTail = gradLocalInt + fullExeSize * 5 * sizeof(OUTDTYPE);

    RegTensor<T> vregSrc;
    RegTensor<T> vregSrc1;
    RegTensor<T> vregSrc2;
    RegTensor<T> vregSrc3;
    RegTensor<T> vregSrc4;
    RegTensor<T> vregSrc5;
    RegTensor<T> vregSrc6;
    RegTensor<T> vregSrc7;
    RegTensor<T> vregSrcTail;
    RegTensor<T> vregSrcTail1;
    RegTensor<T> vregSrcTail2;
    RegTensor<T> vregSrcTail3;

    RegTensor<T> vregGrad;
    RegTensor<T> vregGrad1;
    RegTensor<T> vregGrad2;
    RegTensor<T> vregGrad3;
    RegTensor<T> vregGrad4;
    RegTensor<T> vregGrad5;
    RegTensor<T> vregGrad6;
    RegTensor<T> vregGrad7;
    RegTensor<T> vregGrad8;
    RegTensor<T> vregGrad9;
    RegTensor<T> vregGradTail;
    RegTensor<T> vregGradTail1;

    RegTensor<T> vregMul;
    RegTensor<T> vregMul1;
    RegTensor<T> vregMul2;
    RegTensor<T> vregMul3;
    RegTensor<T> vregMul4;
    RegTensor<T> vregMul5;
    RegTensor<T> vregMul6;
    RegTensor<T> vregMul7;
    RegTensor<T> vregMul8;
    RegTensor<T> vregMul9;
    RegTensor<T> vregMul10;
    RegTensor<T> vregMul11;

    RegTensor<T1> vregSrcFp8;
    RegTensor<T1> vregSrcFp81;
    RegTensor<T1> vregSrcFp8Tail;
    RegTensor<OUTDTYPE> vregGradDtype;
    RegTensor<OUTDTYPE> vregGradDtype1;
    RegTensor<OUTDTYPE> vregGradDtype2;
    RegTensor<OUTDTYPE> vregGradDtype3;
    RegTensor<OUTDTYPE> vregGradDtype4;
    RegTensor<OUTDTYPE> vregGradDtypeTail;

    RegTensor<T> vregAdd;
    RegTensor<T> vregAdd1;
    RegTensor<T> vregAdd2;
    RegTensor<T> vregAdd3;
    RegTensor<T> vregAdd4;
    RegTensor<T> vregAdd5;
    RegTensor<T> vregAdd6;
    RegTensor<T> vregAdd7;
    RegTensor<T> vregAdd8;
    RegTensor<T> vregAddLast;
    RegTensor<T> vregReduceSum;

    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregFullExeB8 = CreateMask<T1, MaskPattern::ALL>();
    MaskReg pregFullExeB16 = CreateMask<OUTDTYPE, MaskPattern::ALL>();
    UnalignReg uregReduceSum;
    MaskReg pregTailExe = UpdateMask<T>(reduceSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        if constexpr (OUTDTYPE_IS_B16) {
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8, ((__ubuf__ T1 *&)srcLocalInt), 512);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp81, ((__ubuf__ T1 *&)srcLocalInt1), 512);
            LoadAlign<T1, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcFp8Tail, ((__ubuf__ T1 *&)srcLocalIntTail),
                                                                   512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype1,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt1), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype2,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt2), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype3,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt3), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtype4,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalInt4), 512);
            LoadAlign<OUTDTYPE, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregGradDtypeTail,
                                                                         ((__ubuf__ OUTDTYPE *&)gradLocalIntTail), 512);
            Cast<T, T1, castTraitB82B32Zero>(vregSrc, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32One>(vregSrc1, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Two>(vregSrc2, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Three>(vregSrc3, vregSrcFp8, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Zero>(vregSrc4, vregSrcFp81, pregFullExeB8);
            Cast<T, T1, castTraitB82B32One>(vregSrc5, vregSrcFp81, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Two>(vregSrc6, vregSrcFp81, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Three>(vregSrc7, vregSrcFp81, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Zero>(vregSrcTail, vregSrcFp8Tail, pregFullExeB8);
            Cast<T, T1, castTraitB82B32One>(vregSrcTail1, vregSrcFp8Tail, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Two>(vregSrcTail2, vregSrcFp8Tail, pregFullExeB8);
            Cast<T, T1, castTraitB82B32Three>(vregSrcTail3, vregSrcFp8Tail, pregFullExeB8);

            Interleave(vregSrc, vregSrc2, vregSrc, vregSrc2);
            Interleave(vregSrc1, vregSrc3, vregSrc1, vregSrc3);
            Interleave(vregSrc4, vregSrc6, vregSrc4, vregSrc6);
            Interleave(vregSrc5, vregSrc7, vregSrc5, vregSrc7);
            Interleave(vregSrcTail, vregSrcTail2, vregSrcTail, vregSrcTail2);
            Interleave(vregSrcTail1, vregSrcTail3, vregSrcTail1, vregSrcTail3);

            Muls(vregSrc, vregSrc, scaleDx, pregFullExe);
            Muls(vregSrc1, vregSrc1, scaleDx, pregFullExe);
            Muls(vregSrc2, vregSrc2, scaleDx, pregFullExe);
            Muls(vregSrc3, vregSrc3, scaleDx, pregFullExe);
            Muls(vregSrc4, vregSrc4, scaleDx, pregFullExe);
            Muls(vregSrc5, vregSrc5, scaleDx, pregFullExe);
            Muls(vregSrc6, vregSrc6, scaleDx, pregFullExe);
            Muls(vregSrc7, vregSrc7, scaleDx, pregFullExe);
            Muls(vregSrcTail, vregSrcTail, scaleDx, pregFullExe);
            Muls(vregSrcTail1, vregSrcTail1, scaleDx, pregFullExe);
            Muls(vregSrcTail2, vregSrcTail2, scaleDx, pregTailExe);
            Muls(vregSrcTail3, vregSrcTail3, scaleDx, pregTailExe);

            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad, vregGradDtype, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad1, vregGradDtype, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad2, vregGradDtype1, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad3, vregGradDtype1, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad4, vregGradDtype2, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad5, vregGradDtype2, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad6, vregGradDtype3, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad7, vregGradDtype3, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGrad8, vregGradDtype4, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGrad9, vregGradDtype4, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Even>(vregGradTail, vregGradDtypeTail, pregFullExeB16);
            Cast<T, OUTDTYPE, castTraitB162B32Odd>(vregGradTail1, vregGradDtypeTail, pregFullExeB16);

            Mul(vregMul, vregGrad, vregSrc, pregFullExe);
            Mul(vregMul1, vregGrad1, vregSrc1, pregFullExe);
            Mul(vregMul2, vregGrad2, vregSrc2, pregFullExe);
            Mul(vregMul3, vregGrad3, vregSrc3, pregFullExe);
            Mul(vregMul4, vregGrad4, vregSrc4, pregFullExe);
            Mul(vregMul5, vregGrad5, vregSrc5, pregFullExe);
            Mul(vregMul6, vregGrad6, vregSrc6, pregFullExe);
            Mul(vregMul7, vregGrad7, vregSrc7, pregFullExe);
            Mul(vregMul8, vregGrad8, vregSrcTail, pregFullExe);
            Mul(vregMul9, vregGrad9, vregSrcTail1, pregFullExe);
            Mul(vregMul10, vregGradTail, vregSrcTail2, pregTailExe);
            Mul(vregMul11, vregGradTail1, vregSrcTail3, pregTailExe);

            Add(vregAdd, vregMul, vregMul1, pregFullExe);
            Add(vregAdd1, vregMul2, vregMul3, pregFullExe);
            Add(vregAdd2, vregMul4, vregMul5, pregFullExe);
            Add(vregAdd3, vregMul6, vregMul7, pregFullExe);
            Add(vregAdd4, vregMul8, vregMul9, pregFullExe);
            Add(vregAdd5, vregMul10, vregMul11, pregFullExe);
            Add(vregAdd6, vregAdd, vregAdd1, pregFullExe);
            Add(vregAdd7, vregAdd2, vregAdd3, pregFullExe);
            Add(vregAdd8, vregAdd4, vregAdd5, pregFullExe);
            Add(vregAdd, vregAdd6, vregAdd7, pregFullExe);
            Add(vregAddLast, vregAdd, vregAdd8, pregFullExe);
            ReduceSum(vregReduceSum, vregAddLast, pregFullExe);
            vstus(uregReduceSum, 1, vregReduceSum, ((__ubuf__ float *&)dstLocalInt), POST_UPDATE);
        }
    }
    vstas(uregReduceSum, ((__ubuf__ float *&)dstLocalInt), 0, POST_UPDATE);
}

template <typename T1, typename T, typename OUTDTYPE, uint32_t srcN>
__aicore__ inline void MyAntiQuantSoftmaxGradFrontCast(const LocalTensor<T> &dstTensor,
                                                       const LocalTensor<OUTDTYPE> &gradTensor,
                                                       const LocalTensor<T1> &srcTensor, float scaleDx, uint32_t srcM,
                                                       uint32_t realN = srcN)
{
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t gradLocalInt = gradTensor.GetPhyAddr();
    const bool OUTDTYPE_IS_B16 = IsSameType<OUTDTYPE, half>::value || IsSameType<OUTDTYPE, bfloat16_t>::value;

    if constexpr (srcN <= 128) {
        const uint32_t fullExeSize = 128;
        uint32_t reduceSize = realN >> 1;
        AntiQuantSoftmaxGradVF128<T1, T, OUTDTYPE, srcN>(dstLocalInt, gradLocalInt, srcLocalInt, scaleDx, srcM,
                                                         reduceSize);
    } else if constexpr (srcN <= 256) {
        const uint32_t fullExeSize = 128;
        uint32_t tailSize = realN > fullExeSize ? (realN % fullExeSize) : 0;
        uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
        if (realN == fullExeSize * 2) {
            reduceSize = (fullExeSize + 1) >> 1;
        }
        AntiQuantSoftmaxGradVF256<T1, T, OUTDTYPE, srcN>(dstLocalInt, gradLocalInt, srcLocalInt, scaleDx, srcM,
                                                         reduceSize);
    } else if constexpr (srcN <= 384) {
        const uint32_t fullExeSize = 128;
        uint32_t tailSize = realN > fullExeSize * 2 ? (realN % fullExeSize) : 0;
        uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
        if (realN == fullExeSize * 3) {
            reduceSize = (fullExeSize + 1) >> 1;
        }
        AntiQuantSoftmaxGradVF384<T1, T, OUTDTYPE, srcN>(dstLocalInt, gradLocalInt, srcLocalInt, scaleDx, srcM,
                                                         reduceSize);
    } else if constexpr (srcN <= 512) {
        const uint32_t fullExeSize = 128;
        uint32_t tailSize = realN > fullExeSize * 3 ? (realN % fullExeSize) : 0;
        uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
        if (realN == fullExeSize * 4) {
            reduceSize = (fullExeSize + 1) >> 1;
        }
        AntiQuantSoftmaxGradVF512<T1, T, OUTDTYPE, srcN, OUTDTYPE_IS_B16>(dstLocalInt, gradLocalInt, srcLocalInt,
                                                                          scaleDx, srcM, reduceSize);
    } else if constexpr (srcN <= 640) {
        const uint32_t fullExeSize = 128;
        uint32_t tailSize = realN > fullExeSize * 4 ? (realN % fullExeSize) : 0;
        uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
        if (realN == fullExeSize * 5) {
            reduceSize = (fullExeSize + 1) >> 1;
        }
        AntiQuantSoftmaxGradVF640<T1, T, OUTDTYPE, srcN, OUTDTYPE_IS_B16>(dstLocalInt, gradLocalInt, srcLocalInt,
                                                                          scaleDx, srcM, reduceSize);
    } else if constexpr (srcN <= 768) {
        const uint32_t fullExeSize = 128;
        uint32_t tailSize = realN > fullExeSize * 5 ? (realN % fullExeSize) : 0;
        uint32_t reduceSize = (tailSize == 0) ? 0 : (tailSize + 1) >> 1;
        if (realN == fullExeSize * 6) {
            reduceSize = (fullExeSize + 1) >> 1;
        }
        AntiQuantSoftmaxGradVF768<T1, T, OUTDTYPE, srcN, OUTDTYPE_IS_B16>(dstLocalInt, gradLocalInt, srcLocalInt,
                                                                          scaleDx, srcM, reduceSize);
    }
}
#else
template <typename T1, typename T, typename OUTDTYPE, uint32_t srcN>
__aicore__ inline void MyAntiQuantSoftmaxGradFrontCast(const LocalTensor<T> &dstTensor,
                                                       const LocalTensor<OUTDTYPE> &gradTensor,
                                                       const LocalTensor<T1> &srcTensor, float scaleDx, uint32_t srcM,
                                                       uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // _MY_ANTI_QUANT_SOFTMAX_GRAD_CAST_INTERFACE_H_
