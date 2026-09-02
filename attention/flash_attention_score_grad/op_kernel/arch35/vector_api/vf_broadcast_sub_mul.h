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
 * \file vf_broadcast_sub_mul.h
 */
#ifndef MY_BROADCAST_SUB_MUL_INTERFACE_H
#define MY_BROADCAST_SUB_MUL_INTERFACE_H
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
/* **************************************************************************************************

SUB *
************************************************************************************************* /
/
@INGROUP BroadcastSubMul
brief compute :sub = (x - broadcast(grad)) * sfm
param [out] dstTensor output LocalTensor
param [in] gradTensor input grad LocalTensor
param [in] srcTensor input src LocalTensor
*/
template <typename T, uint32_t srcN, const bool IS_DETER_OLD>
__simd_vf__ inline void BroadcastSubMulVF64(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t dstLocalIntZero,
                                            uint64_t gradLocalInt, uint64_t sfmLocalInt, uint32_t srcM, uint32_t realN)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregSub;

    RegTensor<float> vregMul;
    RegTensor<float> vregSfm;

    RegTensor<float> vregAdd;
    RegTensor<float> vregReduceSum;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregTailExe = UpdateMask<float>(realN);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregGrad, ((__ubuf__ float *&)gradLocalInt), 1);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float *&)srcLocalInt), 128);
        Sub(vregSub, vregSrc, vregGrad, pregTailExe);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSfm, ((__ubuf__ float *&)sfmLocalInt), 128);
        Mul(vregMul, vregSub, vregSfm, pregTailExe);
        StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregMul, 128,
                                                                   pregFullExe);
        if constexpr (IS_DETER_OLD) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
            Duplicate(vregMul, 0);
            StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ float *&)dstLocalIntZero, vregMul, 128,
                                                                       pregFullExe);
        }
    }
}

template <typename T, uint32_t srcN, const bool IS_DETER_OLD>
__simd_vf__ inline void BroadcastSubMulVF128(uint64_t srcLocalInt, uint64_t dstLocalInt, uint64_t gradLocalInt,
                                             uint64_t sfmLocalInt, uint32_t realTailSize, uint32_t srcM)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregGrad;
    RegTensor<float> vregSub;

    RegTensor<float> vregMul;
    RegTensor<float> vregSfm;

    RegTensor<float> vregAdd;
    RegTensor<float> vregReduceSum;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregTailExe = UpdateMask<float>(realTailSize);

    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregGrad, ((__ubuf__ float *&)gradLocalInt), 1);
        // 主块
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float *&)srcLocalInt), 64);
        Sub(vregSub, vregSrc, vregGrad, pregFullExe);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSfm, ((__ubuf__ float *&)sfmLocalInt), 64);
        Mul(vregMul, vregSub, vregSfm, pregFullExe);
        StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregMul, 64,
                                                                   pregFullExe);
        // 尾块
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float *&)srcLocalInt), 64);
        Sub(vregSub, vregSrc, vregGrad, pregTailExe);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSfm, ((__ubuf__ float *&)sfmLocalInt), 64);
        Mul(vregMul, vregSub, vregSfm, pregTailExe);
        StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalInt), vregMul, 64,
                                                                   pregFullExe);
    }
}

template <typename T, uint16_t srcN, const bool IS_DETER_OLD = 0, const bool IS_FP8_INPUT = false>
__aicore__ inline void BroadcastSubMul(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                       const LocalTensor<T> &gradTensor, const LocalTensor<T> &sfmTensor, uint32_t srcM,
                                       uint32_t realN = srcN)
{
    const uint16_t fullExeSize = 64;
    uint16_t loopTimes = CeilDivision(srcN, fullExeSize);
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t dstLocalIntZero = dstTensor.GetPhyAddr() + fullExeSize * sizeof(float);
    uint64_t gradLocalInt = gradTensor.GetPhyAddr();
    uint64_t sfmLocalInt = sfmTensor.GetPhyAddr();

    if constexpr (IS_FP8_INPUT) {
        loopTimes = CeilDivision(realN, fullExeSize) - 1;
        uint32_t tailSize = realN % fullExeSize;
        uint32_t realTailSize = tailSize == 0 ? fullExeSize : tailSize;
        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * sizeof(float) * loopTimes;
        uint64_t sfmLocalIntTail = sfmTensor.GetPhyAddr() + fullExeSize * sizeof(float) * loopTimes;
        uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + fullExeSize * sizeof(float) * loopTimes;
        __VEC_SCOPE__
        {
            RegTensor<float> vregSrc;
            RegTensor<float> vregGrad;
            RegTensor<float> vregSub;
            RegTensor<float> vregMul;
            RegTensor<float> vregSfm;

            // tail reg
            RegTensor<float> vregSrcTail;
            RegTensor<float> vregSubTail;
            RegTensor<float> vregMulTail;
            RegTensor<float> vregSfmTail;

            MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregTailExe = UpdateMask<float>(realTailSize);

            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregGrad, ((__ubuf__ float *&)gradLocalInt), 1);
                for (uint16_t n = 0; n < loopTimes; n++) {
                    DataCopy(vregSrc, ((__ubuf__ float *&)srcLocalInt + m * srcN + n * fullExeSize));
                    Sub(vregSub, vregSrc, vregGrad, pregFullExe);
                    DataCopy(vregSfm, ((__ubuf__ float *&)sfmLocalInt + m * srcN + n * fullExeSize));
                    Mul(vregMul, vregSub, vregSfm, pregFullExe);
                    DataCopy(((__ubuf__ float *&)dstLocalInt + m * srcN + n * fullExeSize), vregMul, pregFullExe);
                }
                // 尾块
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail,
                                                                         ((__ubuf__ float *&)srcLocalIntTail), srcN);
                Sub(vregSubTail, vregSrcTail, vregGrad, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSfmTail,
                                                                         ((__ubuf__ float *&)sfmLocalIntTail), srcN);
                Mul(vregMulTail, vregSubTail, vregSfmTail, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float *&)dstLocalIntTail),
                                                                         vregMulTail, srcN, pregFullExe);
            }
        }
    } else if constexpr (srcN == 64) {
        BroadcastSubMulVF64<T, srcN, IS_DETER_OLD>(srcLocalInt, dstLocalInt, dstLocalIntZero, gradLocalInt, sfmLocalInt,
                                                   srcM, realN);
    } else if constexpr (srcN == 128) {
        uint32_t tailSize = realN % fullExeSize;
        uint32_t realTailSize = tailSize == 0 ? fullExeSize : tailSize;
        BroadcastSubMulVF128<T, srcN, IS_DETER_OLD>(srcLocalInt, dstLocalInt, gradLocalInt, sfmLocalInt, realTailSize,
                                                    srcM);
    }
}
#else
template <typename T, uint16_t srcN, const bool IS_DETER_OLD = 0, const bool IS_FP8_INPUT = false>
__aicore__ inline void BroadcastSubMul(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                                       const LocalTensor<T> &gradTensor, const LocalTensor<T> &sfmTensor, uint32_t srcM,
                                       uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // MY_BROADCAST_SUB_MUL_INTERFACE_H
