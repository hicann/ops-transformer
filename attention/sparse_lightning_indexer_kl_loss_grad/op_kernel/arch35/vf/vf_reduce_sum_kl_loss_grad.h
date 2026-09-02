/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file vf_reduce_sum_kl_loss_grad.h
 * \brief
 */
#ifndef VF_REDUCE_SUM_KL_LOSS_GRAD_H
#define VF_REDUCE_SUM_KL_LOSS_GRAD_H

#include "kernel_tensor.h"

namespace AscendC {
template <typename T>
__simd_vf__ inline void ReduceSumBasicVF(uint64_t dstLocalInt, uint64_t srcLocalInt, uint64_t srcLocalIntTail,
                                         uint32_t loopTimes, uint32_t tailSize)
{
    RegTensor<float> vregSrc;
    RegTensor<float> vregSrcTail;
    RegTensor<float> vregDst;
    RegTensor<float> vregReduceSum;

    UnalignRegForStore uregRes;

    MaskReg pregFullExe = CreateMask<float, MaskPattern::ALL>();
    MaskReg pregAccu = CreateMask<T, MaskPattern::VL1>();
    MaskReg pregTailExe = UpdateMask<float>(tailSize);

    Duplicate(vregDst, 0.0f);
    for (uint16_t k = 0; k < static_cast<uint16_t>(loopTimes); k++) {
        LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ float *&)srcLocalInt), 64);
        Reduce<MicroAPI::ReduceType::SUM>(vregReduceSum, vregSrc, pregFullExe);
        Add(vregDst, vregDst, vregReduceSum, pregAccu);
    }
    LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ float *&)srcLocalIntTail), 64);
    Reduce<MicroAPI::ReduceType::SUM>(vregReduceSum, vregSrcTail, pregTailExe);
    Add(vregDst, vregDst, vregReduceSum, pregAccu);
    StoreUnAlign<T>(((__ubuf__ T *&)dstLocalInt), vregDst, uregRes, 1);
    StoreUnAlignPost<T>(((__ubuf__ T *&)dstLocalInt), uregRes, 0);
}

template <typename T, typename OUT_T>
__aicore__ inline void ReduceSumVf(const LocalTensor<OUT_T> &dstTensor, const LocalTensor<T> &srcTensor, uint32_t realK)
{
    const uint32_t fullExeSize = 64;
    uint32_t loopTimes = (realK + fullExeSize - 1) / fullExeSize - 1;
    uint32_t tailSize = realK % fullExeSize == 0 ? fullExeSize : realK % fullExeSize;
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + loopTimes * fullExeSize * 4;

    ReduceSumBasicVF<T>(dstLocalInt, srcLocalInt, srcLocalIntTail, loopTimes, tailSize);
}
} // namespace AscendC

#endif // VF_CAST_DUP_H
