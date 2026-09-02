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
 * \file vf_post_reduce_sink.h
 */
#ifndef POST_REDUCE_SINK_INTERFACE_H
#define POST_REDUCE_SINK_INTERFACE_H
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
__simd_vf__ inline void ReduceSinkVF(uint64_t dstLocalInt, uint64_t srcLocalInt, uint64_t srcLocalIntTail,
                                     uint16_t loopTimes, uint16_t fullExeSize, uint32_t realTailSize)
{
    RegTensor<T> vregSrc;
    RegTensor<T> vregSrcTail;
    RegTensor<T> vregReduceSum;
    RegTensor<T> vregReduceSumTail;
    RegTensor<T> vregRes;
    UnalignRegForStore uregRes;

    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregAccu = CreateMask<T, MaskPattern::VL1>();
    MaskReg pregTailExe = UpdateMask<T>(realTailSize);

    Duplicate(vregRes, 0);
    for (uint16_t n = 0; n < loopTimes; n++) {
        LoadAlign(vregSrc, ((__ubuf__ T *&)srcLocalInt + n * fullExeSize));
        Reduce<MicroAPI::ReduceType::SUM>(vregReduceSum, vregSrc, pregFullExe);
        Add(vregRes, vregRes, vregReduceSum, pregAccu);
    }
    // 尾块
    LoadAlign(vregSrcTail, ((__ubuf__ T *&)srcLocalIntTail));
    Reduce<MicroAPI::ReduceType::SUM>(vregReduceSumTail, vregSrcTail, pregTailExe);
    Add(vregRes, vregRes, vregReduceSumTail, pregAccu);
    StoreUnAlign<T>(((__ubuf__ T *&)dstLocalInt), vregRes, uregRes, 1);
    StoreUnAlignPost<T>(((__ubuf__ T *&)dstLocalInt), uregRes, 0);
}

template <typename T>
__aicore__ inline void ReduceSink(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, uint64_t pingSize)
{
    const uint16_t fullExeSize = 64;
    uint16_t loopTimes = static_cast<uint16_t>(((pingSize + fullExeSize - 1) / fullExeSize) - 1);
    uint16_t tailSize = pingSize % fullExeSize;
    uint16_t realTailSize = tailSize == 0 ? fullExeSize : tailSize;
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + loopTimes * fullExeSize * sizeof(T);

    ReduceSinkVF<T>(dstLocalInt, srcLocalInt, srcLocalIntTail, loopTimes, fullExeSize, realTailSize);
}
#else
template <typename T>
__aicore__ inline void ReduceSink(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, uint64_t pingSize)
{}
#endif
} // namespace AscendC

#endif // POST_REDUCE_SINK_INTERFACE_H
