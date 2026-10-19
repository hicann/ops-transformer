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
 * \file vf_ds_abs_reduce_max_david.h
 */
#ifndef DS_ABS_REDUCE_MAX_INTERFACE_DAVID_H_
#define DS_ABS_REDUCE_MAX_INTERFACE_DAVID_H_
#include "kernel_tensor.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace MicroAPI;

template <typename T, uint32_t srcN>
__aicore__ inline void DsAbsReduceMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t srcM,
                                      uint32_t realN = srcN)
{
    if constexpr (srcN == 64) {
        const uint32_t fullExeSize = 64;
        uint64_t srcLocalInt = srcTensor.GetPhyAddr();
        uint64_t dstLocalInt = dstTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<T> vregInput;
            RegTensor<T> vregInputAbs;
            RegTensor<T> vregInputMax;
            RegTensor<T> vregMax;

            MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
            MaskReg pregTailExe = UpdateMask<T>(realN);
            MaskReg pregLen1 = CreateMask<T, MicroAPI::MaskPattern::VL1>();
            UnalignReg uregReduceSum;

            Duplicate(vregMax, 0.0f, pregFullExe);
            for (uint16_t i = 0; i < static_cast<uint16_t>(srcM); i++) {
                Duplicate(vregInputMax, 0.0f, pregFullExe);
                DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregInput, ((__ubuf__ T*&)srcLocalInt), 128);
                Abs(vregInputAbs, vregInput, pregTailExe);
                ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputMax, vregInputAbs, pregTailExe);
                Max<T, MicroAPI::MaskMergeMode::ZEROING>(vregMax, vregMax, vregInputMax, pregLen1);
            }
            vstus(uregReduceSum, 1, vregMax, ((__ubuf__ T*&)dstLocalInt), POST_UPDATE);
            vstas(uregReduceSum, ((__ubuf__ T*&)dstLocalInt), 0, POST_UPDATE);
        }
    } else {
        const uint32_t bytesOfType = sizeof(T);
        const uint32_t regBytes = 256;
        const uint32_t fullExeSize = regBytes / bytesOfType;
        uint16_t repeatTimes = srcN / fullExeSize - 1;
        uint32_t tailSize = realN % fullExeSize == 0 ? fullExeSize : realN % fullExeSize;

        uint64_t srcLocalInt = srcTensor.GetPhyAddr();
        uint64_t dstLocalInt = dstTensor.GetPhyAddr();
        uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + fullExeSize * bytesOfType * repeatTimes;

        __VEC_SCOPE__
        {
            RegTensor<T> vregInput;
            RegTensor<T> vregInputTail;
            RegTensor<T> vregInputAbs;
            RegTensor<T> vregInputTailAbs;
            RegTensor<T> vregInputMax;
            RegTensor<T> vregInputTailMax;
            RegTensor<T> vregMax;
            RegTensor<T> vregMaxTmp;
            MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
            MaskReg pregTailExe = UpdateMask<T>(tailSize);
            MaskReg pregLen1 = CreateMask<T, MicroAPI::MaskPattern::VL1>();
            UnalignReg uregReduceSum;
            Duplicate(vregInputMax, 0.0f, pregFullExe);
            Duplicate(vregMax, 0.0f, pregFullExe);
            for (uint16_t i = 0; i < static_cast<uint16_t>(srcM); i++) {
                for (uint16_t j = 0; j < repeatTimes; j++) {
                    DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregInput, ((__ubuf__ T*&)srcLocalInt), srcN);
                    Abs(vregInputAbs, vregInput, pregFullExe);
                }
                // tailLoop
                DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vregInputTail, ((__ubuf__ T*&)srcLocalIntTail),
                                                                     srcN);
                Abs(vregInputTailAbs, vregInputTail, pregTailExe);
                Max<T, MicroAPI::MaskMergeMode::ZEROING>(vregMaxTmp, vregInputAbs, vregInputTailAbs, pregFullExe);
                Max<T, MicroAPI::MaskMergeMode::ZEROING>(vregInputMax, vregMaxTmp, vregInputMax, pregFullExe);
            }
            ReduceMax<T, MicroAPI::MaskMergeMode::ZEROING>(vregMax, vregInputMax, pregFullExe);

            vstus(uregReduceSum, 1, vregMax, ((__ubuf__ T*&)dstLocalInt), POST_UPDATE);
            vstas(uregReduceSum, ((__ubuf__ T*&)dstLocalInt), 0, POST_UPDATE);
        }
    }
}
#else
template <typename T, uint32_t srcN>
__aicore__ inline void DsAbsReduceMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint64_t srcM,
                                      uint64_t realN = srcN)
{}
#endif
} // namespace AscendC

#endif // _DS_ABS_REDUCE_MAX_INTERFACE_DAVID_H_
