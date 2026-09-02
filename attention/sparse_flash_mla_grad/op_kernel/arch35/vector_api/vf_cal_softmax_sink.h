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
 * \file vf_cal_softmax_sink.h
 */
#ifndef CAL_SOFTMAX_SINK_INTERFACE_H
#define CAL_SOFTMAX_SINK_INTERFACE_H
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;
/* **************************************************************************************************

 *
************************************************************************************************* /
/
@INGROUP BroadcastSubMul
brief compute exp(sink - lse)
param [out] dstTensor output LocalTensor
param [in] lseTensor input lse LocalTensor
param [in] sinkTensor input sink LocalTensor
*/

template <typename T>
__simd_vf__ inline void CalculateSoftMaxSinkVF(uint64_t dstLocalInt, uint64_t lseLocalInt, uint64_t sinkLocalInt,
                                               uint32_t realM)
{
    RegTensor<T> vregLse;
    RegTensor<T> vregSink;
    RegTensor<T> vregRes;
    UnalignRegForStore uregRes;

    MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregTail = UpdateMask<T>(realM);

    Duplicate(vregRes, 0);
    LoadAlign(vregLse, ((__ubuf__ T *&)lseLocalInt));
    LoadAlign(vregSink, ((__ubuf__ T *&)sinkLocalInt));
    ExpSub(vregRes, vregSink, vregLse, pregTail);

    StoreAlign<T>(((__ubuf__ T *&)dstLocalInt), vregRes, pregTail);
}

template <typename T>
__aicore__ inline void CalculateSoftMaxSink(const LocalTensor<T> &dstTensor, const LocalTensor<T> &lseTensor,
                                            const LocalTensor<T> &sinkTensor, uint32_t realM)
{
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t lseLocalInt = lseTensor.GetPhyAddr();
    uint64_t sinkLocalInt = sinkTensor.GetPhyAddr();

    CalculateSoftMaxSinkVF<T>(dstLocalInt, lseLocalInt, sinkLocalInt, realM);
}
#else
template <typename T>
__aicore__ inline void CalculateSoftMaxSink(const LocalTensor<T> &dstTensor, const LocalTensor<T> &lseTensor,
                                            const LocalTensor<T> &sinkTensor, uint32_t realM)
{}
#endif
} // namespace AscendC

#endif // CAL_SOFTMAX_SINK_INTERFACE_H
