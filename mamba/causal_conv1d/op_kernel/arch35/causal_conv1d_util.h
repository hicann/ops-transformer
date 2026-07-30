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
 * \file causal_conv1d_util.h
 */

#ifndef CAUSAL_CONV1D_COMMON_H
#define CAUSAL_CONV1D_COMMON_H

#include "kernel_operator.h"

namespace CausalConv1dUtil {

__aicore__ inline int32_t CurrSlot(int32_t t, int32_t width) { return (t + width - 1) % (width + 1); }

__aicore__ inline int32_t NextSlot(int32_t t, int32_t width) { return (t + width) % (width + 1); }

__aicore__ inline constexpr bool IsFnInputModeKey(uint32_t inputModeKey)
{
    return (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN) ||
           (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH);
}

__aicore__ inline constexpr bool IsUpdateInputModeKey(uint32_t inputModeKey)
{
    return (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN) ||
           (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH);
}

__aicore__ inline constexpr bool IsVarlenInputModeKey(uint32_t inputModeKey)
{
    return (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN) ||
           (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN);
}

struct CalcBufPool {
    AscendC::LocalTensor<float> weight;
    AscendC::LocalTensor<float> bias;

    __aicore__ inline CalcBufPool() = default;

    __aicore__ static inline CalcBufPool Init(AscendC::LocalTensor<float> calcBuf, int32_t width, int32_t rowStride)
    {
        CalcBufPool pool;
        pool.weight = calcBuf;
        pool.bias = calcBuf[width * rowStride];
        return pool;
    }
};

using AscendC::SetFlag;
using AscendC::WaitFlag;

template <AscendC::HardEvent EVENT>
__aicore__ inline void SetEvent(AscendC::HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<EVENT>(eventId);
    WaitFlag<EVENT>(eventId);
}

} // namespace CausalConv1dUtil

#endif // CAUSAL_CONV1D_COMMON_H
