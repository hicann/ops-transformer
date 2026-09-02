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
 * \file moe_init_routing_v2_arch35_common.h
 * \brief
 */
#ifndef MOE_INIT_ROUTING_V2_ARCH35_COMMON_H
#define MOE_INIT_ROUTING_V2_ARCH35_COMMON_H

#include "kernel_operator.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define THREAD_NUM 2048
#endif
namespace MoeInitRoutingV2 {
using namespace AscendC;
constexpr int64_t SPLIT_N = 0;
constexpr int64_t SPLIT_K = 1;
constexpr float MIN_FP32 = -3.4e38;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;

constexpr int64_t ASSIST_NUM = 256;
constexpr int64_t ASSIST_INDEX_NUM = 32;

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;

constexpr int64_t MAX_EXPERT_NUM = 5120;
constexpr int64_t DROPLESS_MODE = 0;
constexpr int64_t DROP_PAD_MODE = 1;
constexpr int64_t EXERPT_TOKENS_COUNT = 2;
constexpr int64_t EXERPT_TOKENS_CUMSUM = 1;
constexpr int64_t EXERPT_TOKENS_NONE = 0;
constexpr int64_t EXERPT_TOKENS_BEFORE_CAPACITY = 1;

constexpr int32_t MAX_THREAD_DIM_Y_NUM = 8;

constexpr int32_t BLOCK_B32_SIZE = 8;
constexpr int32_t REPEAT_B32_SIZE = 64;
constexpr int32_t CONSTANT_TWO = 2;
constexpr int32_t CONSTANT_THREE = 3;
constexpr int32_t CONSTANT_FOUR = 4;
constexpr int32_t CONSTANT_FIVE = 5;
constexpr int32_t CONSTANT_SIX = 6;
constexpr int32_t CONSTANT_SEVEN = 7;

const __gm__ int32_t assist[256] = {
    0,  0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0, 2,  0, 0, 0, 0, 0, 0, 0, 3,  0, 0, 0, 0, 0, 0, 0,
    4,  0, 0, 0, 0, 0, 0, 0, 5,  0, 0, 0, 0, 0, 0, 0, 6,  0, 0, 0, 0, 0, 0, 0, 7,  0, 0, 0, 0, 0, 0, 0,
    8,  0, 0, 0, 0, 0, 0, 0, 9,  0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0,
    12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0,
    16, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0,
    20, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0,
    24, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0,
    28, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0};

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

__aicore__ inline constexpr uint32_t GetVRegSize()
{
    return 256U;
}

__aicore__ inline void ArithProgressionPerf(const LocalTensor<int32_t> &dst, const int32_t firstValue,
                                            const int32_t diffValue, int32_t countAlign)
{
    // countAlign must be eight aligned
    countAlign = (countAlign + CONSTANT_SEVEN) / BLOCK_B32_SIZE * BLOCK_B32_SIZE;
    dst.SetValue(0, firstValue);
    dst.SetValue(1, firstValue + diffValue * 1);
    dst.SetValue(CONSTANT_TWO, firstValue + diffValue * CONSTANT_TWO);
    dst.SetValue(CONSTANT_THREE, firstValue + diffValue * CONSTANT_THREE);
    dst.SetValue(CONSTANT_FOUR, firstValue + diffValue * CONSTANT_FOUR);
    dst.SetValue(CONSTANT_FIVE, firstValue + diffValue * CONSTANT_FIVE);
    dst.SetValue(CONSTANT_SIX, firstValue + diffValue * CONSTANT_SIX);
    dst.SetValue(CONSTANT_SEVEN, firstValue + diffValue * CONSTANT_SEVEN);
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
    if (countAlign > BLOCK_B32_SIZE) {
        int32_t offset;
        if (countAlign > REPEAT_B32_SIZE) {
            for (int32_t i = 1; i < BLOCK_B32_SIZE; i++) {
                offset = i * BLOCK_B32_SIZE;
                Adds(dst[offset], dst, diffValue * offset, BLOCK_B32_SIZE, 1, {1, 1, 1, 1});
            }
            int32_t loopTimes = (countAlign + REPEAT_B32_SIZE - 1) / REPEAT_B32_SIZE;
            for (int32_t i = 1; i < loopTimes - 1; i++) {
                offset = i * REPEAT_B32_SIZE;
                Adds(dst[offset], dst, diffValue * offset, REPEAT_B32_SIZE, 1, {1, 1, 1, 1});
            }
            offset = (loopTimes - 1) * REPEAT_B32_SIZE;
            Adds(dst[offset], dst, diffValue * offset, countAlign - offset, 1, {1, 1, 1, 1});
        } else {
            for (int32_t i = 1; i < countAlign / BLOCK_B32_SIZE; i++) {
                offset = i * BLOCK_B32_SIZE;
                Adds(dst[offset], dst, diffValue * offset, BLOCK_B32_SIZE, 1, {1, 1, 1, 1});
            }
        }
    }
}

template <typename T>
__aicore__ inline void InitGmValue(GlobalTensor<T> gm, LocalTensor<T> tensor, int64_t count, T value)
{
    int64_t ubSize = tensor.GetSize();
    int64_t loops = Ceil(count, ubSize);
    int64_t perLoopCount = Ceil(count, loops);
    int64_t lastLoopCount = count - (loops - 1) * perLoopCount;
    Duplicate(tensor, value, perLoopCount);
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
    DataCopyExtParams intriParams{1, static_cast<uint32_t>(perLoopCount * sizeof(T)), 0, 0, 0};
    for (int64_t loop = 0; loop < loops - 1; loop++) {
        DataCopyPad(gm[loop * perLoopCount], tensor, intriParams);
    }
    intriParams.blockLen = static_cast<uint32_t>(lastLoopCount * sizeof(T));
    DataCopyPad(gm[(loops - 1) * perLoopCount], tensor, intriParams);
}

} // namespace MoeInitRoutingV2
#endif // MOE_INIT_ROUTING_V2_ARCH35_COMMON_H
