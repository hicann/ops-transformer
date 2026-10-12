/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file non_zero_big_mask_dim1.h
 * \brief
 */

#ifndef CANN_NON_ZERO_BIG_MASK_DIM1_H
#define CANN_NON_ZERO_BIG_MASK_DIM1_H

#include "non_zero_big_mask.h"

namespace NonZero {
using namespace AscendC;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

template <typename T1, typename T2>
class NonZeroBigMaskDim1 : public NonZeroBigMask<T1, T2> {
public:
    __aicore__ inline NonZeroBigMaskDim1(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace,
                                const mNonZeroTiling* tilingData, TPipe* tPipe);
    template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
    __aicore__ inline void Process(CLS_NAME* objPtr);
    __aicore__ inline void ComputeOutput(LocalTensor<int32_t>& yUb);
};
template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMaskDim1<T1, T2>::ComputeOutput(LocalTensor<int32_t>& yUb)
{
    this->CastOutput(yUb, this->processSize_ * (this->shapeDim_ + 1), this->arNum_, this->shapeDim_);
    this->outQueY_.EnQue(yUb);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMaskDim1<T1, T2>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace,
                                                        const mNonZeroTiling* tilingData, TPipe* tPipe)
{
    this->InitBase(x, y, outShape, workspace, tilingData, tPipe);
}

template <typename T1, typename T2>
template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
__aicore__ inline void NonZeroBigMaskDim1<T1, T2>::Process(CLS_NAME* objPtr)
{
    this->template ProcessBase<CLS_NAME, ComputeOutputPtr>(objPtr);
}
} // namespace NonZero

#endif // CANN_NON_ZERO_BIG_MASK_DIM1_H
