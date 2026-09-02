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
 * \file moe_token_permute_with_routing_map_common.h
 * \brief
 */
#ifndef MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_COMMON_H
#define MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_COMMON_H

#include "kernel_operator.h"

namespace MoeTokenPermute {
using namespace AscendC;
constexpr int64_t SPLIT_N = 0;
constexpr int64_t SPLIT_K = 1;
constexpr float MIN_FP32 = -3.4e38f;
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;

constexpr int64_t ASSIST_NUM = 256;
constexpr int64_t ASSIST_INDEX_NUM = 32;
constexpr int64_t SORT32_ALIGN_ELEMENT = 32;
constexpr int64_t SCATTER_UB_SIZE = 192 * 1024;
constexpr int64_t SCATTER_PER_LOOP_ALIGN = 512;

__aicore__ inline int64_t CalcScatterPerLoopMaxRows(int64_t useCoreNum)
{
    int64_t perLoopMaxRows = (SCATTER_UB_SIZE - ASSIST_NUM * static_cast<int64_t>(sizeof(float)) -
                              SCATTER_PER_LOOP_ALIGN * static_cast<int64_t>(sizeof(float))) /
                             (SORT32_ALIGN_ELEMENT * 2) / 2;
    return (perLoopMaxRows / SCATTER_PER_LOOP_ALIGN) * SCATTER_PER_LOOP_ALIGN;
}

// 稀疏 block 布局：assist[i*8]=i，与 moe_v2 assist 一致，供 scatter 按 idx*8 取值。
const __gm__ int32_t scatterAssist[256] = {
    0,  0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0, 2,  0, 0, 0, 0, 0, 0, 0, 3,  0, 0, 0, 0, 0, 0, 0,
    4,  0, 0, 0, 0, 0, 0, 0, 5,  0, 0, 0, 0, 0, 0, 0, 6,  0, 0, 0, 0, 0, 0, 0, 7,  0, 0, 0, 0, 0, 0, 0,
    8,  0, 0, 0, 0, 0, 0, 0, 9,  0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0,
    12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0,
    16, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0,
    20, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0,
    24, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0,
    28, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0};

constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;

constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;
constexpr int64_t INT32DIVINT8 = 4;

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

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

template <typename T>
__aicore__ inline void DataCopyPadCustom(LocalTensor<T> inLocal, GlobalTensor<T> dstGm,
                                         DataCopyExtParams tokenCopyParams, DataCopyPadExtParams<T> padParams)
{
#ifndef __CCE_KT_TEST__
    DataCopyPad(inLocal, dstGm, tokenCopyParams, padParams);
#endif
}

template <typename T>
__aicore__ inline void DataCopyPadCustom(GlobalTensor<T> dstGm, LocalTensor<T> inLocal,
                                         DataCopyExtParams tokenCopyParams)
{
#ifndef __CCE_KT_TEST__
    DataCopyPad(dstGm, inLocal, tokenCopyParams);
#endif
}

template <typename T>
__aicore__ inline void DataCopyPadCustom(GlobalTensor<T> dstGm, LocalTensor<T> inLocal, DataCopyParams tokenCopyParams)
{
#ifndef __CCE_KT_TEST__
    DataCopyPad(dstGm, inLocal, tokenCopyParams);
#endif
}
} // namespace MoeTokenPermute
  // namespace MoeTokenPermute
#endif // MOE_TOKEN_PERMUTE_WITH_ROUTING_MAP_COMMON_H
