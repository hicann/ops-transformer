/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_AIV_SYNC_COMMON_H
#define MC2_AIV_SYNC_COMMON_H

#include "mc2_aiv_data_copy_common.h"

namespace Mc2AivSync {

using namespace AscendC;

constexpr uint64_t AIV_SYNC_MODE = 0U;
constexpr uint64_t AIC_SYNC_MODE = 2U;

template <pipe_t pipe, uint64_t mode>
inline __aicore__ void FFTSCrossCoreSync(uint64_t flagId)
{
    AscendC::CrossCoreSetFlag<mode, pipe>(flagId);
}

inline __aicore__ void SetAndWaitAivSync(uint64_t flagId, int32_t pipeDepth = 2)
{
    FFTSCrossCoreSync<PIPE_MTE3, AIV_SYNC_MODE>(flagId + pipeDepth);
    WaitEvent(flagId + pipeDepth);
}

inline __aicore__ void SetAicSync(uint64_t flagId)
{
    FFTSCrossCoreSync<PIPE_MTE3, AIC_SYNC_MODE>(flagId);
}

__attribute__((always_inline)) inline __aicore__ void SetBuffFlagByAdd(__gm__ int32_t *buff,
                                                                       TBuf<AscendC::TPosition::VECCALC> &uBuf,
                                                                       int32_t flag)
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> ubTensor = uBuf.AllocTensor<int32_t>();
    ubTensor(0) = flag;
    PipeBarrier<PIPE_ALL>();
    SetAtomicAdd<int32_t>();
    PipeBarrier<PIPE_ALL>();
    Mc2AivDataCopy::CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    PipeBarrier<PIPE_ALL>();
}

} // namespace Mc2AivSync

#endif // MC2_AIV_SYNC_COMMON_H
