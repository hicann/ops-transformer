/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SYNC_H
#define MEGA_MOE_SYNC_H

#if !(defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
#include "c_api/asc_simd.h"
#endif
#include "mega_moe_arch_policy.h"
#include <type_traits>

namespace MegaMoeImpl {

// This load bypasses Scalar DCache and adds no barrier or polling backoff.
template <typename T>
__aicore__ inline T GmRead(__gm__ T* address)
{
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int64_t> ||
                      std::is_same_v<T, uint64_t>,
                  "GM scalar reads require 32-bit or 64-bit integers");
#if defined(MEGA_MOE_ARCH92_DEVICE)
    return AscendC::ReadGmByPassDCache(address);
#else
    if constexpr (sizeof(T) == sizeof(uint64_t)) {
        return static_cast<T>(ld_dev(reinterpret_cast<__gm__ uint64_t*>(address), 0));
    } else {
        return static_cast<T>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(address), 0));
    }
#endif
}

template <typename T, typename V>
__aicore__ inline void GmWrite(__gm__ T* address, V value)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::WriteGmByPassDCache(address, static_cast<T>(value));
#else
    asc_store_dev(address, static_cast<T>(value));
#endif
}

enum class SaturationStrategy : uint8_t {
    USE_API = 0,
    USE_GLOBAL = 1
};

__aicore__ inline SaturationStrategy GetSaturationStrategy()
{
#if defined(MEGA_MOE_ARCH92_DEVICE) || (defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
    return static_cast<SaturationStrategy>(AscendC::GetCtrlSpr<60, 60>());
#else
    return static_cast<SaturationStrategy>((static_cast<uint64_t>(asc_get_ctrl()) >> 60U) & 1U);
#endif
}

__aicore__ inline void SetSaturationStrategy(SaturationStrategy strategy)
{
#if defined(MEGA_MOE_ARCH92_DEVICE) || (defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
    AscendC::SetCtrlSpr<60, 60>(static_cast<int64_t>(strategy));
#else
    constexpr uint64_t STRATEGY_MASK = uint64_t{1} << 60U;
    const uint64_t currentCtrl = static_cast<uint64_t>(asc_get_ctrl());
    asc_set_ctrl((currentCtrl & ~STRATEGY_MASK) | (static_cast<uint64_t>(strategy) << 60U));
#endif
}

__aicore__ inline int64_t GetDeviceCycle()
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    return AscendC::GetSystemCycle();
#else
    return asc_get_system_cycle();
#endif
}

__aicore__ inline void CleanInvalidateGmCache()
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    dcci(reinterpret_cast<__gm__ int64_t*>(0), cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
#else
    asc_dcci_entire_out();
#endif
}

__aicore__ inline void CleanInvalidateGmCacheLine(__gm__ void* address)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(address));
    AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        global);
#else
    asc_dcci_single(address);
#endif
}

__aicore__ inline void GmDataBarrier()
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    dsb(mem_dsb_t::DSB_DDR);
#else
    asc_sync_data_barrier(mem_dsb_t::DSB_DDR);
#endif
}

// Callers select AIC/AIV execution and use pipelines available on that core.
template <pipe_t Producer, pipe_t Consumer>
__aicore__ inline void PipeNotify(uint16_t eventId)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::TQueSync<Producer, Consumer> sync;
    sync.SetFlag(static_cast<event_t>(eventId));
#else
    asc_sync_notify(Producer, Consumer, static_cast<event_t>(eventId));
#endif
}

template <pipe_t Producer, pipe_t Consumer>
__aicore__ inline void PipeWait(uint16_t eventId)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::TQueSync<Producer, Consumer> sync;
    sync.WaitFlag(static_cast<event_t>(eventId));
#else
    asc_sync_wait(Producer, Consumer, static_cast<event_t>(eventId));
#endif
}

template <pipe_t Producer, pipe_t Consumer, int32_t EventId>
__aicore__ inline void PipeFence()
{
    PipeNotify<Producer, Consumer>(EventId);
    PipeWait<Producer, Consumer>(EventId);
}

template <pipe_t Pipe>
__aicore__ inline void PipeBarrier()
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::PipeBarrier<Pipe>();
#else
    static_assert(Pipe != PIPE_V && Pipe != PIPE_S, "arch35 has no V/S pipe_barrier instruction");
    asc_sync_pipe(Pipe);
#endif
}

// Mode 4: one AIC and one of its AIVs. AIC adds 16 to select AIV1.
// Events are counted; GMM producers must retain their 15-tile credit limit.
template <pipe_t Pipe>
__aicore__ inline void SyncIntraArrive(uint16_t eventId)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::CrossCoreSetFlag<4, Pipe>(eventId);
#else
    asc_sync_intra_arrive(Pipe, eventId);
#endif
}

template <pipe_t Pipe>
__aicore__ inline void SyncIntraWait(uint16_t eventId)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::CrossCoreWaitFlag<4, Pipe>(eventId);
#else
    asc_sync_intra_wait(Pipe, eventId);
#endif
}

// Mode 0: all cores of the caller's kind in this kernel launch.
template <pipe_t Pipe>
__aicore__ inline void SyncInterArrive(uint16_t eventId)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::CrossCoreSetFlag<0, Pipe>(eventId);
#else
    // 9.1.0 的 asc_sync_block_arrive 使用 mode 2，不能替代这里的 mode 0。
    const uint64_t config = 1U | (static_cast<uint64_t>(eventId & 0xFU) << 8U);
    ffts_cross_core_sync(Pipe, config);
#endif
}

template <pipe_t Pipe>
__aicore__ inline void SyncInterWait(uint16_t eventId)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::CrossCoreWaitFlag<0, Pipe>(eventId);
#else
    asc_sync_inter_wait(Pipe, eventId);
#endif
}

#if !defined(MEGA_MOE_ARCH92_DEVICE)
// Reserved event IDs used by the full-core barrier protocols.
constexpr uint16_t ALL_AIC_BARRIER_EVENT = 11;
constexpr uint16_t AIV_ARRIVE_EVENT = 12;
constexpr uint16_t AIV_RELEASE_EVENT = 13;
constexpr uint16_t ALL_AIV_BARRIER_EVENT = 14;
// On AIC, this offset selects the second paired AIV for intra-block events.
constexpr uint16_t AIV1_EVENT_OFFSET = 16;
#endif

// Call only when every participating core reaches the barrier; drain GMM ACKs first.
__aicore__ inline void AllVectorBarrier()
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::SyncAll<true>();
#else
    if constexpr (g_coreType == AscendC::AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncInterArrive<PIPE_MTE3>(ALL_AIV_BARRIER_EVENT);
        SyncInterWait<PIPE_S>(ALL_AIV_BARRIER_EVENT);
    }
#endif
}

__aicore__ inline void AllCoreBarrier()
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    AscendC::SyncAll<false>();
#else
    PipeBarrier<PIPE_ALL>();
    if constexpr (g_coreType == AscendC::AIC) {
        asc_sync_intra_wait(PIPE_S, AIV_ARRIVE_EVENT);
        asc_sync_intra_wait(PIPE_S, AIV_ARRIVE_EVENT + AIV1_EVENT_OFFSET);
        SyncInterArrive<PIPE_FIX>(ALL_AIC_BARRIER_EVENT);
        asc_sync_inter_wait(PIPE_S, ALL_AIC_BARRIER_EVENT);
        asc_sync_intra_arrive(PIPE_S, AIV_RELEASE_EVENT);
        asc_sync_intra_arrive(PIPE_S, AIV_RELEASE_EVENT + AIV1_EVENT_OFFSET);
    } else {
        asc_sync_intra_arrive(PIPE_MTE3, AIV_ARRIVE_EVENT);
        asc_sync_intra_wait(PIPE_S, AIV_RELEASE_EVENT);
    }
#endif
}

} // namespace MegaMoeImpl
#endif
