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
 * \file vf_broadcast_sub_mul_david.h
 */
#ifndef MY_BROADCAST_SUB_MUL_INTERFACE_DAVID_H
#define MY_BROADCAST_SUB_MUL_INTERFACE_DAVID_H
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
template <typename T, uint32_t srcN, const bool IS_DETER = 0>
__aicore__ inline void BroadcastSubMul(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& gradTensor, const LocalTensor<T>& sfmTensor, uint32_t srcM,
                                       uint32_t realN = srcN)
{
    const uint32_t fullExeSize = 64;
    uint16_t loopTimes = CeilDivision(srcN, fullExeSize);
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t dstLocalIntZero = dstTensor.GetPhyAddr() + fullExeSize * sizeof(float);
    uint64_t gradLocalInt = gradTensor.GetPhyAddr();
    uint64_t sfmLocalInt = sfmTensor.GetPhyAddr();

    if constexpr (srcN == 64) {
        __VEC_SCOPE__
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
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregGrad, ((__ubuf__ float*&)gradLocalInt), 1);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float*&)srcLocalInt), 128);
                Sub(vregSub, vregSrc, vregGrad, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSfm, ((__ubuf__ float*&)sfmLocalInt), 128);
                Mul(vregMul, vregSub, vregSfm, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt), vregMul, 128,
                                                                         pregFullExe);
                if constexpr (IS_DETER) { // 确定性计算需要将64~128的数据补零， 否则会有脏数据inf
                    Duplicate(vregMul, 0);
                    DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>((__ubuf__ float*&)dstLocalIntZero, vregMul,
                                                                             128, pregFullExe);
                }
            }
        }
    } else if constexpr (srcN == 128) {
        uint32_t tailSize = realN % fullExeSize;
        uint32_t realTailSize = tailSize == 0 ? fullExeSize : tailSize;
        __VEC_SCOPE__
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
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                    vregGrad, ((__ubuf__ float*&)gradLocalInt), 1);
                // 主块
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float*&)srcLocalInt), 64);
                Sub(vregSub, vregSrc, vregGrad, pregFullExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSfm, ((__ubuf__ float*&)sfmLocalInt), 64);
                Mul(vregMul, vregSub, vregSfm, pregFullExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt), vregMul, 64,
                                                                         pregFullExe);
                // 尾块
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float*&)srcLocalInt), 64);
                Sub(vregSub, vregSrc, vregGrad, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSfm, ((__ubuf__ float*&)sfmLocalInt), 64);
                Mul(vregMul, vregSub, vregSfm, pregTailExe);
                DataCopy<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(((__ubuf__ float*&)dstLocalInt), vregMul, 64,
                                                                         pregFullExe);
            }
        }
    }
}
#else
template <typename T, uint32_t srcN, const bool IS_DETER = 0>
__aicore__ inline void BroadcastSubMul(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                       const LocalTensor<T>& gradTensor, const LocalTensor<T>& sfmTensor, uint32_t srcM,
                                       uint32_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // MY_BROADCAST_SUB_MUL_INTERFACE_DAVID_H
