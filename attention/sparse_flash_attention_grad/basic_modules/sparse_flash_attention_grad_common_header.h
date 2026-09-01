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
 * \file common_header.h
 * \brief
 */
#pragma once
#include <limits>
#include <type_traits>
#include "kernel_operator.h"
#include "kernel_event.h"
#include "kernel_tensor.h"
#include "kernel_macros.h"

namespace SFAG_BASIC {

constexpr static uint64_t MAX_CORE_NUM = 24;
constexpr static uint64_t PER_LOOP_BLOCK_SIZE = 128;
constexpr static uint64_t PING_PONG_BUFFER = 2;
constexpr static uint64_t SCATTER_BUFFER_NUM = 3;

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define PIPE_BARRIER(pipe) AscendC::PipeBarrier<pipe>()

/////////////////////////////////////////////////////
// common struct
/////////////////////////////////////////////////////

template <class TILING_CLASS, typename T1, const uint32_t INPUT_LAYOUT = 0, const uint32_t ATTEN_ENABLE = 1,
          const bool HAS_ROPE = false, const bool IS_BSND = false, const bool IS_DETERMINISTIC = false,
          const bool KV_MERGE = false, typename... Args>
struct SFAG_TYPE {
    using tiling_class = TILING_CLASS;
    using t1 = T1;
    static constexpr uint32_t atten_enable = ATTEN_ENABLE;
    static constexpr bool has_rope = HAS_ROPE;
    static constexpr bool is_bsnd = IS_BSND;
    static constexpr bool is_deterministic = IS_DETERMINISTIC;
    static constexpr bool kv_merge = KV_MERGE;
};

struct RunInfo {
    int64_t task;
    int64_t curS1;
    int64_t curS2;
    int64_t sumGmOffset;
    int64_t blkCntOffset;
    int64_t queryGmOffset;
    int64_t queryRopeGmOffset;
    int64_t keyGmOffset;
    int64_t keyRopeGmOffset;
    int64_t dyGmOffset;
    int64_t valueGmOffset;
    int64_t indicesGmOffset;
    int64_t mm12GmOffset;
    int64_t mm345GmOffset;
    int64_t mm3OutGmOffset;
    int64_t mm4OutGmOffset;
    int64_t mm5OutGmOffset;
    int64_t actualSelCntOffset;
    int64_t scatterTaskId;
    int64_t s1Index;
    int64_t s1Begin;
    int64_t s1End;
    int64_t actualSelectedBlockCount;
    int64_t changeS1 = false;
    int64_t selectedKGmOffset;
    int64_t lastBlockSize;
    bool isLastBasicBlock;
    bool valid = false;
    bool isSmallS2 = false;
    bool noReload = false;
    event_t gatherMte2WaitMte3;
    event_t gatherMte2WaitMte3Pong;
    event_t scatterVWaitMte3;
    event_t scatterVWaitMte3Pong;
    event_t scatterTmpKMte2WaitV;
    event_t scatterTmpVMte2WaitV;
    event_t scatterTmpKMte2WaitVPong;
    event_t scatterTmpVMte2WaitVPong;
};

struct Nz2NdInfo {
    int64_t ndFirstAxisRealSize = 0;
    int64_t ndFirstAxisBaseSize = 0;
    int64_t ndFirstAxisLoopSize = 0;
    int64_t ndLastAxis = 0;
    int64_t loopIdx = 0;
    int64_t vecCoreOffset = 0;
};

/////////////////////////////////////////////////////
// hardware
/////////////////////////////////////////////////////
enum class ArchType {
    ASCEND_V220,
    ASCEND_V200,
    ASCEND_M200
};

template <ArchType ArchTag>
struct HardwareInfo {
    static uint32_t const l2BW = 5;
    static uint32_t const hbmBW = 1;
    static uint32_t const supportMix = 0;
    static uint32_t const l1Size = 512 * 1024;
    static uint32_t const l0ASize = 64 * 1024;
    static uint32_t const l0BSize = 64 * 1024;
    static uint32_t const l0CSize = 128 * 1024;
    static uint32_t const l2Size = 192 * 1024 * 1024;
    static uint32_t const biasSize = 1024;
    static uint32_t const fixBufSize = 7 * 1024;
    static uint32_t const ubSize = 192 * 1024;
    static uint32_t const fractalSize = 512;
    static uint32_t const l1l0BlockSize = 32;
    static uint32_t const btBlockSize = 64;
    static uint32_t const fbBlockSize = 128;
};

/////////////////////////////////////////////////////
// mem
/////////////////////////////////////////////////////
enum class BufferType {
    ASCEND_UB,
    ASCEND_CB,
    ASCEND_L0A,
    ASCEND_L0B,
    ASCEND_L0C,
    ASCEND_MAX
};

template <ArchType ArchTag>
struct AsdopsBuffer {
public:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    __aicore__ AsdopsBuffer()
    {
        constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {
            HardwareInfo<ArchTag>::ubSize, HardwareInfo<ArchTag>::l1Size, HardwareInfo<ArchTag>::l0ASize,
            HardwareInfo<ArchTag>::l0BSize, HardwareInfo<ArchTag>::l0CSize};
#ifdef __DAV_C220_VEC__
        tensor[(uint32_t)BufferType::ASCEND_UB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        tensor[(uint32_t)BufferType::ASCEND_UB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
#elif __DAV_C220_CUBE__
        tensor[(uint32_t)BufferType::ASCEND_CB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        tensor[(uint32_t)BufferType::ASCEND_CB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
        tensor[(uint32_t)BufferType::ASCEND_L0A].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        tensor[(uint32_t)BufferType::ASCEND_L0A].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        tensor[(uint32_t)BufferType::ASCEND_L0B].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        tensor[(uint32_t)BufferType::ASCEND_L0B].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        tensor[(uint32_t)BufferType::ASCEND_L0C].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
        tensor[(uint32_t)BufferType::ASCEND_L0C].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
#else
        tensor[(uint32_t)BufferType::ASCEND_UB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        tensor[(uint32_t)BufferType::ASCEND_UB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        tensor[(uint32_t)BufferType::ASCEND_CB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        tensor[(uint32_t)BufferType::ASCEND_CB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
        tensor[(uint32_t)BufferType::ASCEND_L0A].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        tensor[(uint32_t)BufferType::ASCEND_L0A].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        tensor[(uint32_t)BufferType::ASCEND_L0B].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        tensor[(uint32_t)BufferType::ASCEND_L0B].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        tensor[(uint32_t)BufferType::ASCEND_L0C].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
        tensor[(uint32_t)BufferType::ASCEND_L0C].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
#endif
    };
#pragma GCC diagnostic pop

    template <BufferType BufferType_, typename DstDataType>
    __aicore__ AscendC::LocalTensor<DstDataType> GetBuffer(const uint32_t offset) const
    {
        return tensor[(uint32_t)BufferType_][offset].template ReinterpretCast<DstDataType>();
    }

public:
    AscendC::LocalTensor<uint8_t> tensor[(uint32_t)BufferType::ASCEND_MAX];
};

/*
 * common function
 */
template <class T>
__aicore__ inline T AlignTo(const T n, const T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a < b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

} // namespace SFAG_BASIC
