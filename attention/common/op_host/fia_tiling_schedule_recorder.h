/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_tiling_schedule_recorder.h
 * \brief 记录一次 tiling 过程中设置的 schedule mode，供 tiling 结果缓存快照使用
 */

#ifndef FIA_TILING_SCHEDULE_RECORDER_H
#define FIA_TILING_SCHEDULE_RECORDER_H

#include <cstdint>

namespace optiling {
// gert::TilingContext 只提供 SetScheduleMode，没有对应的 getter。
// tiling 结果缓存无法直接从 context 快照 schedule mode，因此在各处调用
// context->SetScheduleMode 时同步记录到 thread_local 状态中，缓存在同一线程
// 的 tiling 执行前后读取，从而在缓存命中时能够完整恢复 schedule mode。
class FiaTilingScheduleRecorder {
public:
    static void Record(const uint32_t scheduleMode)
    {
        auto &state = ThreadState();
        state.set = true;
        state.scheduleMode = scheduleMode;
    }

    static void Reset()
    {
        auto &state = ThreadState();
        state.set = false;
        state.scheduleMode = 0U;
    }

    // 返回 true 表示本次 tiling 过程中记录过 schedule mode
    static bool Get(uint32_t &scheduleMode)
    {
        auto &state = ThreadState();
        if (!state.set) {
            return false;
        }
        scheduleMode = state.scheduleMode;
        return true;
    }

private:
    struct State {
        bool set = false;
        uint32_t scheduleMode = 0U;
    };

    static State &ThreadState()
    {
        static thread_local State state;
        return state;
    }
};
} // namespace optiling

#endif // FIA_TILING_SCHEDULE_RECORDER_H
