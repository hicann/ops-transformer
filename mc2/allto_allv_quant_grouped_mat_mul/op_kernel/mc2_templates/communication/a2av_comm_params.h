/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef A2AV_COMM_PARAMS_H
#define A2AV_COMM_PARAMS_H

#include "../common/a2av_common_tiling.h"

namespace MC2KernelTemplate {

static constexpr uint8_t SEND_OFFSET_BY_RAW_ARRAY = 0;
static constexpr uint8_t SEND_OFFSET_ACCUMULATIVE = 1;

struct A2avCommParams {
    uint64_t sendCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t sendOffset[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t recvCnt[MAX_EP_RANK_SIZE] = {0UL};
    uint64_t recvOffset[MAX_EP_RANK_SIZE] = {0UL};
};

__aicore__ inline void CalcA2avCommBeforeParams(
    A2avCommParams &params,
    const uint64_t *rawSendCounts, const uint64_t *rawRecvCounts,
    uint32_t rankDim, uint32_t e, uint32_t startExpertIdx, uint32_t expertNum,
    uint64_t axis,
    uint64_t &sendOffsetLastSum, uint64_t &recvOffsetLastSum)
{
    for (uint64_t i = 0UL; i < rankDim; i++) {
        params.sendCnt[i] = 0UL;
        params.recvCnt[i] = 0UL;
        for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
            params.sendCnt[i] += static_cast<uint64_t>(rawSendCounts[expertIdx + i * e]) * axis;
            params.recvCnt[i] += static_cast<uint64_t>(rawRecvCounts[expertIdx + i * e]) * axis;
        }
    }

    params.sendOffset[0] = 0UL;
    for (uint32_t j = 0U; j < startExpertIdx; j++) {
        params.sendOffset[0] += static_cast<uint64_t>(rawSendCounts[j]) * axis;
    }
    for (uint32_t i = 1U; i < rankDim; i++) {
        params.sendOffset[i] = 0UL;
        for (uint32_t k = 0U; k < i; k++) {
            for (uint32_t j = 0U; j < e; j++) {
                params.sendOffset[i] += static_cast<uint64_t>(rawSendCounts[j + k * e]) * axis;
            }
        }
        for (uint32_t j = 0U; j < startExpertIdx; j++) {
            params.sendOffset[i] += static_cast<uint64_t>(rawSendCounts[j + i * e]) * axis;
        }
    }

    for (uint32_t i = 0U; i < rankDim; i++) {
        if ((startExpertIdx == 0U) && (i == 0U)) {
            params.recvOffset[i] = 0UL;
            recvOffsetLastSum += params.recvCnt[0];
        } else {
            params.recvOffset[i] = recvOffsetLastSum;
            recvOffsetLastSum += params.recvCnt[i];
        }
    }
}

__aicore__ inline void CalcA2avCommAfterParams(
    A2avCommParams &params,
    const uint64_t *rawSendCounts, const uint64_t *rawRecvCounts,
    uint32_t rankDim, uint32_t e, uint32_t startExpertIdx, uint32_t expertNum,
    uint64_t axis, uint64_t &sendOffsetLastSum, uint64_t &recvOffsetLastSum)
{
    for (uint64_t i = 0UL; i < rankDim; i++) {
        params.sendCnt[i] = 0UL;
        params.recvCnt[i] = 0UL;
        for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
            params.sendCnt[i] += static_cast<uint64_t>(rawSendCounts[expertIdx + i * e]) * axis;
            params.recvCnt[i] += static_cast<uint64_t>(rawRecvCounts[expertIdx + i * e]) * axis;
        }
    }

    uint64_t batchStartOffset = 0UL;
    for (uint64_t i = 0UL; i < startExpertIdx; i++) {
        for (uint64_t j = 0UL; j < rankDim; j++) {
            batchStartOffset += static_cast<uint64_t>(rawSendCounts[i + j * e]);
        }
    }
    batchStartOffset *= axis;
    params.sendOffset[0] = batchStartOffset;
    uint64_t rankRunningOffset = 0UL;
    for (uint64_t i = 1UL; i < rankDim; i++) {
        for (uint64_t expertIdx = startExpertIdx; expertIdx < startExpertIdx + expertNum; expertIdx++) {
            rankRunningOffset += static_cast<uint64_t>(rawSendCounts[expertIdx + (i - 1) * e]) * axis;
        }
        params.sendOffset[i] = batchStartOffset + rankRunningOffset;
    }
    params.recvOffset[0] = 0UL;
    for (uint64_t i = 0UL; i < startExpertIdx; i++) {
        params.recvOffset[0] += static_cast<uint64_t>(rawRecvCounts[i]) * axis;
    }
    for (uint64_t i = 1UL; i < rankDim; i++) {
        params.recvOffset[i] = params.recvOffset[i - 1];
        for (uint64_t j = 0UL; j < e; j++) {
            params.recvOffset[i] += static_cast<uint64_t>(rawRecvCounts[startExpertIdx + (i - 1) * e + j]) * axis;
        }
    }
}

} // namespace MC2KernelTemplate

#endif // A2AV_COMM_PARAMS_H