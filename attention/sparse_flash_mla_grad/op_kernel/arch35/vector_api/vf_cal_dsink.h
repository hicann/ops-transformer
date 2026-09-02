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
 * \file vf_cal_sink.h
 */
#ifndef CAL_DSINK_INTERFACE_H
#define CAL_DSINK_INTERFACE_H
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
param [in] pTensor input p LocalTensor
param [in] dpTensor input dp LocalTensor
*/

template <typename T>
__simd_vf__ inline void CalculateDsinkVF(uint64_t dstLocalInt, uint64_t pLocalInt, uint64_t dpLocalInt,
                                         uint64_t softmaxSinkLocalInt, uint64_t pLocalIntTail, uint64_t dpLocalIntTail,
                                         uint16_t loopTimes, uint16_t fullExeSize, uint32_t realTailSize,
                                         uint64_t realM)
{
    RegTensor<T> vregSrc;
    RegTensor<T> vregP;
    RegTensor<T> vregDp;
    RegTensor<T> vregPTail;
    RegTensor<T> vregDpTail;
    RegTensor<T> vregSoftmaxSink;

    RegTensor<T> vregMul;
    RegTensor<T> vregMul1;

    RegTensor<T> vregReduceSum;
    RegTensor<T> vregMulTail;
    RegTensor<T> vregMulTail1;
    RegTensor<T> vregReduceSumTail;
    RegTensor<T> vregRes;
    UnalignRegForStore uregRes;

    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregAccu = CreateMask<T, MaskPattern::VL1>();
    MaskReg pregTailExe = UpdateMask<T>(realTailSize);
    int32_t srcN = 128;
    for (uint16_t m = 0; m < static_cast<uint16_t>(realM); m++) {
        Duplicate(vregRes, 0);
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregSoftmaxSink, ((__ubuf__ T *&)softmaxSinkLocalInt), 1);
        for (uint16_t n = 0; n < loopTimes; n++) {
            LoadAlign(vregP, ((__ubuf__ T *&)pLocalInt + m * srcN + n * fullExeSize)); ////todo srcN
            LoadAlign(vregDp, ((__ubuf__ T *&)dpLocalInt + m * srcN + n * fullExeSize));
            Mul(vregMul, vregP, vregDp, pregFullExe);
            Mul(vregMul1, vregMul, vregSoftmaxSink, pregFullExe);
            Muls(vregMul1, vregMul1, -1, pregFullExe);
            Reduce<MicroAPI::ReduceType::SUM>(vregReduceSum, vregMul1, pregFullExe);
            Add(vregRes, vregRes, vregReduceSum, pregAccu);
        }

        // 尾块
        LoadAlign(vregPTail, ((__ubuf__ T *&)pLocalIntTail + m * srcN));
        LoadAlign(vregDpTail, ((__ubuf__ T *&)dpLocalIntTail + m * srcN));
        Mul(vregMulTail, vregPTail, vregDpTail, pregTailExe);
        Mul(vregMulTail1, vregMulTail, vregSoftmaxSink, pregTailExe);
        Muls(vregMulTail1, vregMulTail1, -1, pregFullExe);
        Reduce<MicroAPI::ReduceType::SUM>(vregReduceSumTail, vregMulTail1, pregTailExe);
        Add(vregRes, vregRes, vregReduceSumTail, pregAccu);
        uint64_t dstOffset = dstLocalInt + 4 * m;
        StoreUnAlign<T>(((__ubuf__ T *&)dstOffset), vregRes, uregRes, 1);
        StoreUnAlignPost<T>(((__ubuf__ T *&)dstOffset), uregRes, 0);
    }
}

template <typename T>
__aicore__ inline void CalculateDsink(const LocalTensor<T> &dstTensor, const LocalTensor<T> &pTensor,
                                      const LocalTensor<T> &dpTensor, const LocalTensor<T> &softmaxSinkTensor,
                                      int64_t realN, uint64_t realM)
{
    const uint16_t fullExeSize = 64;
    uint16_t loopTimes = CeilDivision(realN, fullExeSize) - 1;
    uint16_t tailSize = realN % fullExeSize;
    uint16_t realTailSize = tailSize == 0 ? fullExeSize : tailSize;
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t pLocalInt = pTensor.GetPhyAddr();
    uint64_t dpLocalInt = dpTensor.GetPhyAddr();
    uint64_t softmaxSinkLocalInt = softmaxSinkTensor.GetPhyAddr();
    uint64_t pLocalIntTail = pTensor.GetPhyAddr() + loopTimes * fullExeSize * sizeof(T);
    uint64_t dpLocalIntTail = dpTensor.GetPhyAddr() + loopTimes * fullExeSize * sizeof(T);

    CalculateDsinkVF<T>(dstLocalInt, pLocalInt, dpLocalInt, softmaxSinkLocalInt, pLocalIntTail, dpLocalIntTail,
                        loopTimes, fullExeSize, realTailSize, realM);
}
#else
template <typename T>
__aicore__ inline void CalculateDsink(const LocalTensor<T> &dstTensor, const LocalTensor<T> &pTensor,
                                      const LocalTensor<T> &dpTensor, const LocalTensor<T> &softmaxSinkTensor,
                                      int64_t realN, uint64_t realM)
{}
#endif
} // namespace AscendC

#endif // CAL_DSINK_INTERFACE_H
