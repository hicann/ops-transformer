/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Pipeline ABI and stats flush directly adapted from reference PR 13974.
#ifndef ENGRAM_GATE_PIPELINE_ARCH35_H
#define ENGRAM_GATE_PIPELINE_ARCH35_H
#include "engram_gate_arch35.h"
namespace EngramGateOps {
constexpr event_t BUF_EVT = static_cast<event_t>(0);      // MTE2->V 就绪基 id, 实际 id = BUF_EVT + slot
constexpr event_t REL_EVT = static_cast<event_t>(0);      // V->MTE2 slot 释放基 id
constexpr event_t OUT_EVT = static_cast<event_t>(4);      // V->MTE3 copy-out 守卫 (专用)
constexpr event_t OUT_DONE_EVT = static_cast<event_t>(5); // MTE3->V out slot 完成基 id, 实际 id = OUT_DONE_EVT + slot
constexpr event_t STATS_WINDOW_EVT = static_cast<event_t>(7); // V<->MTE3 统计量窗口搬出
constexpr uint32_t DBUF_SLOTS = 2;                            // 输入数据双缓冲 slot 数

constexpr uint32_t UB_ALIGN = 32; // UB 缓冲起始偏移对齐
constexpr uint32_t SINGLE_BUFFER_SLOT = 1;
constexpr int32_t ONE_HEAD_PER_TASK = 1;
constexpr int32_t HEADS_PER_PAIR = 2;
constexpr int32_t STATS_OUTPUT_COUNT = 4;
constexpr uint32_t STATS_TASK_FLOATS = STATS_OUTPUT_COUNT * STATS_BLOCK_FLOATS;
constexpr uint32_t STATS_TASK_BYTES = STATS_TASK_FLOATS * sizeof(float);
constexpr int32_t STATS_MAX_WINDOW_TASKS = 64;
constexpr uint32_t STATS_WINDOW_GROWTH_FACTOR = 2;

__aicore__ inline uint32_t AlignUp32(const uint32_t bytes)
{
    return (bytes + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
}

// BF16 -> FP32: ZERO 布局, ZEROING 合并 (解包 + 升精度)。
template <bool TWO_HEAD, bool PAIR_MODE>
__aicore__ inline void EngramGateCopyStats(__ubuf__ float* statsUb, __gm__ float* dotOut, __gm__ float* gateOut,
                                           __gm__ float* rstdXOut, __gm__ float* rstdKOut, int64_t row0, int64_t stride,
                                           uint32_t tasks)
{
    constexpr uint32_t heads = TWO_HEAD ? HEADS_PER_PAIR : ONE_HEAD_PER_TASK;
    constexpr uint32_t groups = PAIR_MODE ? HEADS_PER_PAIR : ONE_HEAD_PER_TASK;
    constexpr uint32_t blockBytes = heads * sizeof(float);
    constexpr uint32_t srcStride = (groups * STATS_TASK_BYTES - UB_ALIGN) / UB_ALIGN;
    const int64_t firstRow = TWO_HEAD ? HEADS_PER_PAIR * row0 : row0;
    const int64_t rowStride = TWO_HEAD ? HEADS_PER_PAIR * static_cast<int64_t>(stride) : stride;
    __gm__ float* outputs[STATS_OUTPUT_COUNT] = {dotOut, gateOut, rstdXOut, rstdKOut};

    SetFlag<HardEvent::V_MTE3>(STATS_WINDOW_EVT);
    WaitFlag<HardEvent::V_MTE3>(STATS_WINDOW_EVT);
    for (uint32_t group = 0; group < groups && group < tasks; ++group) {
        const uint32_t blocks = (tasks - group + groups - 1) / groups;
        const int64_t dstStride = (rowStride - heads) * sizeof(float);
        for (uint32_t stat = 0; stat < STATS_OUTPUT_COUNT; ++stat) {
            AscendC::CopyUbufToGmAlignV2<float, AscendC::PaddingMode::Normal>(
                outputs[stat] + firstRow + group, statsUb + group * STATS_TASK_FLOATS + stat * STATS_BLOCK_FLOATS,
                blocks, blockBytes, srcStride, dstStride, true, 0);
        }
    }
    // 窗口会立即复用，必须等待所有统计量搬出后再允许下一次 VF 写入。
    SetFlag<HardEvent::MTE3_V>(STATS_WINDOW_EVT);
    WaitFlag<HardEvent::MTE3_V>(STATS_WINDOW_EVT);
}

class EngramGatePipelineBase {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR kv, GM_ADDR w, GM_ADDR out, GM_ADDR dot, GM_ADDR gate, GM_ADDR rx,
                                GM_ADDR rk, const EngramGateTilingData* td)
    {
        hiddenStates_ = x;
        kv_ = kv;
        weight_ = w;
        output_ = out;
        dot_ = dot;
        gate_ = gate;
        rstdX_ = rx;
        rstdK_ = rk;
        tilingData_ = td;
    }

protected:
    GM_ADDR hiddenStates_ = nullptr;
    GM_ADDR kv_ = nullptr;
    GM_ADDR weight_ = nullptr;
    GM_ADDR output_ = nullptr;
    GM_ADDR dot_ = nullptr;
    GM_ADDR gate_ = nullptr;
    GM_ADDR rstdX_ = nullptr;
    GM_ADDR rstdK_ = nullptr;
    const EngramGateTilingData* tilingData_ = nullptr;
};
} // namespace EngramGateOps
#endif
