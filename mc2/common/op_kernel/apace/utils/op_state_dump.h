/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_STATE_DUMP_H
#define OP_STATE_DUMP_H

// 使用约定: 调用方(成员声明/include/Init/DoDump/SetOpStateDump)无需包裹 #if MC2_DFX_ENABLE，
// 本头文件已内置开关——DFX 关闭时 Init/DoDump 编译为空操作，可直接无条件调用。

#include "op_state_dump_struct.h"

#if defined(__CCE_AICORE__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

// DoDump 更新域掩码: 按位或组合任意待更新域。
// DUMP_FIELD_TURN 与 DUMP_FIELD_TURN_INC 互斥，同时指定时 TURN_INC 优先。
enum DumpField : uint8_t {
    DUMP_FIELD_NONE = 0,
    DUMP_FIELD_TURN = 1,      // 写 execTurn (显式轮次值)
    DUMP_FIELD_POSITION = 2,  // 写 execPosition
    DUMP_FIELD_PHASE = 4,     // 写 commPhase
    DUMP_FIELD_COMMIT = 8,    // 递增 commCommitCount
    DUMP_FIELD_WAIT = 16,     // 递增 commWaitCount
    DUMP_FIELD_TURN_INC = 32, // execTurn 槽内自增1 (轮次推进，调用方无需维护计数)
    // 常用组合: 每轮首个打点——轮次自增+写位置/阶段
    DUMP_FIELD_EXEC_INC = DUMP_FIELD_TURN_INC | DUMP_FIELD_POSITION | DUMP_FIELD_PHASE,
    // 常用组合: 同轮次内后续打点——只写位置/阶段，不动轮次
    DUMP_FIELD_STEP = DUMP_FIELD_POSITION | DUMP_FIELD_PHASE,
    // 常用组合: 写显式轮次+位置/阶段
    DUMP_FIELD_EXEC = DUMP_FIELD_TURN | DUMP_FIELD_POSITION | DUMP_FIELD_PHASE,
};

namespace Mc2Kernel {

#if MC2_DFX_ENABLE
class OpStateDump {
public:
    __aicore__ inline OpStateDump() = default;

    __aicore__ inline void Init(GM_ADDR workspaceGM, const Utils::DfxWorkspaceLayoutInfo *wsLayout, uint32_t aicCoreNum)
    {
        opStateDumpBase_ = nullptr;
        slotIdx_ = 0;

        if (wsLayout == nullptr || workspaceGM == nullptr) {
            return;
        }

        // 遍历 workspace 段表，定位 STATE_DUMP 段。
        // 注意: kernel 收到的 workspaceGM 已跳过 LIB_API 段(由框架管理)，
        // 但段表中的 offset 是从 0 累加的(含 LIB_API)，因此需减去 libApiSize
        // 得到用户段实际偏移，否则基地址越界导致 GM 写入崩溃。
        uint64_t libApiSize = 0;
        for (uint32_t i = 0; i < wsLayout->segCount && i < Utils::MAX_WORKSPACE_SEGMENTS; i++) {
            if (wsLayout->segments[i].type == Utils::WS_SEG_LIB_API) {
                libApiSize = wsLayout->segments[i].size;
            }
            if (wsLayout->segments[i].type == Utils::WS_SEG_STATE_DUMP) {
                uint64_t userOffset = wsLayout->segments[i].offset - libApiSize;
                opStateDumpBase_ = reinterpret_cast<__gm__ Utils::StateDumpPerCore *>(workspaceGM + userOffset);
                break;
            }
        }

        if (opStateDumpBase_ == nullptr) {
            return;
        }

        if ASCEND_IS_AIC {
            slotIdx_ = AscendC::GetBlockIdx();
        } else {
            slotIdx_ = aicCoreNum + AscendC::GetBlockIdx();
        }

        __gm__ Utils::StateDumpPerCore *mySlot = &opStateDumpBase_[slotIdx_];
        mySlot->magicNum = 0x5A5A5A5AU;
        mySlot->coreId = static_cast<uint16_t>(slotIdx_);
        mySlot->execTurn = 0;
        mySlot->execPosition = 0;
        mySlot->commPhase = static_cast<uint8_t>(Utils::RT_PHASE_COMM_INIT);
        mySlot->commCommitCount = 0;
        mySlot->commWaitCount = 0;

        FlushCache();
    }

    // 唯一打点接口: 按 fields 掩码(DUMP_FIELD_*)更新对应域，未选中的域保持 GM 中原值。
    // 每轮首个打点用 DUMP_FIELD_EXEC_INC(轮次自增)，同轮后续用 DUMP_FIELD_STEP(不动轮次)，
    // 轮次值需显式指定时用 DUMP_FIELD_EXEC 并传 turn; GM 中 execTurn 恒为当前轮次(tile 序号)。
    // commitDelta/waitDelta 为计数增量(默认 1)，需一次递增多次时传入 N;
    __aicore__ inline void DoDump(uint8_t fields, uint8_t position = 0, uint8_t phase = 0, uint8_t turn = 0,
                                  uint8_t commitDelta = 1, uint8_t waitDelta = 1)
    {
        if (opStateDumpBase_ == nullptr) {
            return;
        }
        __gm__ Utils::StateDumpPerCore *mySlot = &opStateDumpBase_[slotIdx_];
        if (fields & DUMP_FIELD_TURN_INC) {
            mySlot->execTurn++;
        } else if (fields & DUMP_FIELD_TURN) {
            mySlot->execTurn = turn;
        }
        if (fields & DUMP_FIELD_POSITION) {
            mySlot->execPosition = position;
        }
        if (fields & DUMP_FIELD_PHASE) {
            mySlot->commPhase = phase;
        }
        if (fields & DUMP_FIELD_COMMIT) {
            mySlot->commCommitCount += commitDelta;
        }
        if (fields & DUMP_FIELD_WAIT) {
            mySlot->commWaitCount += waitDelta;
        }
        FlushCache();
    }

private:
    __aicore__ inline void FlushCache()
    {
        AscendC::GlobalTensor<uint8_t> cacheLine;
        cacheLine.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(&opStateDumpBase_[slotIdx_]),
                                  Utils::STATE_DUMP_PER_CORE_SIZE);
        DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            cacheLine);
    }

    __gm__ Utils::StateDumpPerCore *opStateDumpBase_ = nullptr;
    uint32_t slotIdx_ = 0;
};
#else  // !MC2_DFX_ENABLE
class OpStateDump {
public:
    __aicore__ inline OpStateDump() = default;
    __aicore__ inline void Init(GM_ADDR, const Utils::DfxWorkspaceLayoutInfo *, uint32_t) {}
    __aicore__ inline void DoDump(uint8_t, uint8_t = 0, uint8_t = 0, uint8_t = 0, uint8_t = 1, uint8_t = 1) {}
};
#endif // MC2_DFX_ENABLE

} // namespace Mc2Kernel
#endif // __CCE_AICORE__
#endif // OP_STATE_DUMP_H
