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
 * \file vf_outer_add.h
 * \brief
 */

#ifndef VF_OUTER_ADD_H
#define VF_OUTER_ADD_H

#include "kernel_tensor.h"

namespace AscendC {

template <typename T>
__simd_vf__ void OuterAddVFImpl(__ubuf__ T *outputBuf, __ubuf__ T *deltaBuf, __ubuf__ T *kInputBuf,
                                __ubuf__ T *stateInBuf, const uint32_t floatRepSize, uint16_t dLoops, uint32_t dTail,
                                const uint16_t row, const uint32_t col)
{
    AscendC::Reg::RegTensor<T> vregKInput;
    AscendC::Reg::RegTensor<T> vregDelta;
    AscendC::Reg::RegTensor<T> vregStateIn;
    AscendC::Reg::MaskReg pregAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t j = 0; j < dLoops; j++) {
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(vregKInput, kInputBuf + j * floatRepSize);
        for (uint16_t i = 0; i < row; i++) {
            uint16_t loopOffset = i * col + j * floatRepSize;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                vregDelta, deltaBuf + i); // 使用DIST_BRC_B32模式搬运将首元素broadcast到整个寄存器
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(vregStateIn, stateInBuf + loopOffset);
            AscendC::Reg::MulAddDst<T, AscendC::Reg::MaskMergeMode::ZEROING>(vregStateIn, vregKInput, vregDelta,
                                                                             pregAll);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregStateIn,
                                                                            pregAll);
        }
    }

    if (dTail > 0) {
        AscendC::Reg::MaskReg tailMask = AscendC::Reg::UpdateMask<T>(dTail);
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(vregKInput, kInputBuf + dLoops * floatRepSize);
        for (uint16_t i = 0; i < row; i++) {
            uint16_t loopOffset = i * col + dLoops * floatRepSize;
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B32>(
                vregDelta, deltaBuf + i); // 使用DIST_BRC_B32模式搬运将首元素broadcast到整个寄存器
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(vregStateIn, stateInBuf + loopOffset);
            AscendC::Reg::MulAddDst<T, AscendC::Reg::MaskMergeMode::ZEROING>(vregStateIn, vregKInput, vregDelta,
                                                                             tailMask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(outputBuf + loopOffset, vregStateIn,
                                                                            tailMask);
        }
    }
}
template <typename T>
__aicore__ inline void OuterAddVF(const LocalTensor<T> &outputUb, const LocalTensor<T> &deltaInUb,
                                  const LocalTensor<T> &kInUb, const LocalTensor<T> &stateInUb, const uint16_t row,
                                  const uint32_t col)
{
    __ubuf__ T *deltaBuf = (__ubuf__ T *)deltaInUb.GetPhyAddr();
    __ubuf__ T *kInputBuf = (__ubuf__ T *)kInUb.GetPhyAddr();
    __ubuf__ T *stateInBuf = (__ubuf__ T *)stateInUb.GetPhyAddr();
    __ubuf__ T *outputBuf = (__ubuf__ T *)outputUb.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64; // 一个寄存器能够存放64个FP32
    uint16_t dLoops = col / floatRepSize;
    uint32_t dTail = col % floatRepSize;
    OuterAddVFImpl<T>(outputBuf, deltaBuf, kInputBuf, stateInBuf, floatRepSize, dLoops, dTail, row, col);
}

} // namespace AscendC

#endif // VF_OUTER_ADD_H
