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
 * \file vf_sink.h
 * \brief
 */
#ifndef VF_SINK_H
#define VF_SINK_H

#include "kernel_tensor.h"

namespace AscendC {
using namespace MicroAPI;

template <typename T>
__simd_vf__ inline void ComputePsinkPartialVF(uint64_t dstLocalInt, uint64_t sinkLocalInt, uint64_t maxLocalInt,
                                              uint64_t sumLocalInt, uint64_t sinkLocalIntTail, uint64_t maxLocalIntTail,
                                              uint64_t sumLocalIntTail, uint32_t loopTimes, uint32_t tailSize)
{
    RegTensor<float> vregSink;
    RegTensor<float> vregMax;
    RegTensor<float> vregSum;
    RegTensor<float> vregSinkTail;
    RegTensor<float> vregMaxTail;
    RegTensor<float> vregSumTail;
    RegTensor<float> vregExp;
    RegTensor<float> vregDiv;
    RegTensor<float> vregReduceSum;
    RegTensor<float> vregDst;

    UnalignRegForStore uregRes;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregAccu = CreateMask<T, MaskPattern::VL1>();
    MaskReg pregTailExe = UpdateMask<float>(tailSize);

    Duplicate(vregDst, 0.0f);
    for (uint16_t k = 0; k < static_cast<uint16_t>(loopTimes); k++) {
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSink, ((__ubuf__ float *&)sinkLocalInt), 64);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregMax, ((__ubuf__ float *&)maxLocalInt), 64);
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSum, ((__ubuf__ float *&)sumLocalInt), 64);
        ExpSub(vregExp, vregSink, vregMax, pregFullExe);
        Div(vregDiv, vregExp, vregSum, pregFullExe);
        Reduce<MicroAPI::ReduceType::SUM>(vregReduceSum, vregDiv, pregFullExe);
        Add(vregDst, vregDst, vregReduceSum, pregAccu);
    }
    LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSinkTail, ((__ubuf__ float *&)sinkLocalIntTail), 64);
    LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregMaxTail, ((__ubuf__ float *&)maxLocalIntTail), 64);
    LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSumTail, ((__ubuf__ float *&)sumLocalIntTail), 64);
    ExpSub(vregExp, vregSinkTail, vregMaxTail, pregTailExe);
    Div(vregDiv, vregExp, vregSumTail, pregTailExe);
    Reduce<MicroAPI::ReduceType::SUM>(vregReduceSum, vregDiv, pregTailExe);
    Add(vregDst, vregDst, vregReduceSum, pregAccu);
    StoreUnAlign<T>(((__ubuf__ T *&)dstLocalInt), vregDst, uregRes, 1);
    StoreUnAlignPost<T>(((__ubuf__ T *&)dstLocalInt), uregRes, 0);
}

template <typename T>
__aicore__ inline void ComputePsinkPartial(const LocalTensor<T> &dstTensor, const LocalTensor<T> &sinkTensor,
                                           const LocalTensor<T> &maxTensor, const LocalTensor<T> &sumTensor,
                                           uint32_t realK)
{
    const uint32_t fullExeSize = 64;
    uint32_t loopTimes = (realK + fullExeSize - 1) / fullExeSize - 1;
    uint32_t tailSize = realK % fullExeSize == 0 ? fullExeSize : realK % fullExeSize;
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t sinkLocalInt = sinkTensor.GetPhyAddr();
    uint64_t maxLocalInt = maxTensor.GetPhyAddr();
    uint64_t sumLocalInt = sumTensor.GetPhyAddr();
    uint64_t sinkLocalIntTail = sinkTensor.GetPhyAddr() + loopTimes * fullExeSize * sizeof(float);
    uint64_t maxLocalIntTail = maxTensor.GetPhyAddr() + loopTimes * fullExeSize * sizeof(float);
    uint64_t sumLocalIntTail = sumTensor.GetPhyAddr() + loopTimes * fullExeSize * sizeof(float);

    ComputePsinkPartialVF<T>(dstLocalInt, sinkLocalInt, maxLocalInt, sumLocalInt, sinkLocalIntTail, maxLocalIntTail,
                             sumLocalIntTail, loopTimes, tailSize);
}
} // namespace AscendC

#endif // VF_SINK_H
