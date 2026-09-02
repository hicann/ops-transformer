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
#ifndef CAL_SINK_INTERFACE_H
#define CAL_SINK_INTERFACE_H
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

template <typename T, uint16_t srcN>
__simd_vf__ inline void CalculateSinkVF(T sinkScale, uint64_t dstLocalInt, uint64_t pLocalInt, uint64_t dpLocalInt,
                                        uint64_t pLocalIntTail, uint64_t dpLocalIntTail, uint64_t maxLocalInt,
                                        uint64_t sumLocalInt, uint16_t srcM, uint16_t loopTimes, uint16_t fullExeSize,
                                        uint32_t realTailSize)
{
    RegTensor<T> vregSrc;
    RegTensor<T> vregP;
    RegTensor<T> vregDp;
    RegTensor<T> vregPTail;
    RegTensor<T> vregDpTail;

    RegTensor<T> vregMax;
    RegTensor<T> vregSum;
    RegTensor<T> vregSink;

    RegTensor<T> vregMul;
    RegTensor<T> vregSub;
    RegTensor<T> vregDiv;
    RegTensor<T> vregReduceSum;
    RegTensor<T> vregMulTail;
    RegTensor<T> vregSubTail;
    RegTensor<T> vregDivTail;
    RegTensor<T> vregReduceSumTail;
    RegTensor<T> vregRes;
    UnalignRegForStore uregRes;

    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregAccu = CreateMask<T, MaskPattern::VL1>();
    MaskReg pregTailExe = UpdateMask<T>(realTailSize);

    Duplicate(vregSink, sinkScale);
    Duplicate(vregRes, 0);
    for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregMax, ((__ubuf__ T *&)maxLocalInt), 8);
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
            vregSum, ((__ubuf__ T *&)sumLocalInt), 8);
        for (uint16_t n = 0; n < loopTimes; n++) {
            LoadAlign(vregP, ((__ubuf__ T *&)pLocalInt + m * srcN + n * fullExeSize));
            LoadAlign(vregDp, ((__ubuf__ T *&)dpLocalInt + m * srcN + n * fullExeSize));
            Mul(vregMul, vregP, vregDp, pregFullExe);
            ExpSub(vregSub, vregSink, vregMax, pregFullExe);
            Div(vregDiv, vregSub, vregSum, pregFullExe);
            Mul(vregMul, vregMul, vregDiv, pregFullExe);
            Reduce<MicroAPI::ReduceType::SUM>(vregReduceSum, vregMul, pregFullExe);
            Add(vregRes, vregRes, vregReduceSum, pregAccu);
        }
        // 尾块
        LoadAlign(vregPTail, ((__ubuf__ T *&)pLocalIntTail + m * srcN));
        LoadAlign(vregDpTail, ((__ubuf__ T *&)dpLocalIntTail + m * srcN));
        Mul(vregMulTail, vregPTail, vregDpTail, pregTailExe);
        ExpSub(vregSubTail, vregSink, vregMax, pregFullExe);
        Div(vregDivTail, vregSubTail, vregSum, pregFullExe);
        Mul(vregMulTail, vregMulTail, vregDivTail, pregTailExe);
        Reduce<MicroAPI::ReduceType::SUM>(vregReduceSumTail, vregMulTail, pregTailExe);
        Add(vregRes, vregRes, vregReduceSumTail, pregAccu);
    }
    StoreUnAlign<T>(((__ubuf__ T *&)dstLocalInt), vregRes, uregRes, 1);
    StoreUnAlignPost<T>(((__ubuf__ T *&)dstLocalInt), uregRes, 0);
}

template <typename T, uint16_t srcN>
__aicore__ inline void CalculateSink(const LocalTensor<T> &dstTensor, const LocalTensor<T> &pTensor,
                                     const LocalTensor<T> &dpTensor, const LocalTensor<T> &maxTensor,
                                     const LocalTensor<T> &sumTensor, T sinkScale, uint16_t srcM, uint16_t realN = srcN)
{
    const uint16_t fullExeSize = 64;
    uint16_t loopTimes = CeilDivision(realN, fullExeSize) - 1;
    uint16_t tailSize = realN % fullExeSize;
    uint16_t realTailSize = tailSize == 0 ? fullExeSize : tailSize;
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t pLocalInt = pTensor.GetPhyAddr();
    uint64_t dpLocalInt = dpTensor.GetPhyAddr();
    uint64_t pLocalIntTail = pTensor.GetPhyAddr() + loopTimes * fullExeSize * sizeof(T);
    uint64_t dpLocalIntTail = dpTensor.GetPhyAddr() + loopTimes * fullExeSize * sizeof(T);
    uint64_t maxLocalInt = maxTensor.GetPhyAddr();
    uint64_t sumLocalInt = sumTensor.GetPhyAddr();

    CalculateSinkVF<T, srcN>(sinkScale, dstLocalInt, pLocalInt, dpLocalInt, pLocalIntTail, dpLocalIntTail, maxLocalInt,
                             sumLocalInt, srcM, loopTimes, fullExeSize, realTailSize);
}
#else
template <typename T, uint16_t srcN>
__aicore__ inline void CalculateSink(const LocalTensor<T> &dstTensor, const LocalTensor<T> &pTensor,
                                     const LocalTensor<T> &dpTensor, const LocalTensor<T> &maxTensor,
                                     const LocalTensor<T> &sumTensor, T sinkScale, uint16_t srcM, uint16_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // CAL_SINK_INTERFACE_H
