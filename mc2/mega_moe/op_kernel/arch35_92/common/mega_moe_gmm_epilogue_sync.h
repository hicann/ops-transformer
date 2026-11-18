/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MOE_GMM_EPILOGUE_SYNC_H
#define MEGA_MOE_GMM_EPILOGUE_SYNC_H

#include "mega_moe_sync.h"
#include "kernel_operator.h"
#include "mega_moe_arch_policy.h"
#include "mega_moe_constants.h"

namespace MegaMoeImpl {

constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t GMM_AIV_FLAG_OFFSET = 16U;

// 每条同步链路各允许 AIC 领先 15 个逻辑 tile，计数跨 expert/wave 保留。
// 每个 tile 的所有 GM 输出完成后通知消费者，消费者处理完所有子 tile 后通知 AIC。
constexpr uint32_t GMM_MAX_PENDING_TILES = 15;

// Mode 4 flag IDs: each pair holds a producer-ready and consumer-ACK event.
enum class GmmEventPair : uint16_t {
    MOE_GMM1 = 8,
    MOE_GMM2 = 10,
    SHARED_GMM1 = 12,
};

// 轮询 GM 中的 int32 flag 直至等于期望值，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagEquals(__gm__ int32_t* flagAddr, int32_t expectedValue,
                                             int64_t pollBackoffCycles = GM_FLAG_POLL_BACKOFF_CYCLES)
{
    while (GmRead(flagAddr) != expectedValue) {
        int64_t startCycle = GetDeviceCycle();
        while (GetDeviceCycle() - startCycle < pollBackoffCycles) {
        }
    }
}

// 轮询 GM 中的 int32 计数直至不小于目标值，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagAtLeast(__gm__ int32_t* flagAddr, int32_t targetValue)
{
    while (GmRead(flagAddr) < targetValue) {
        int64_t startCycle = GetDeviceCycle();
        while (GetDeviceCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    SyncIntraArrive<PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    SyncIntraWait<PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    SyncIntraArrive<PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    SyncIntraWait<PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

// Non-owning view of the caller's UB state. State advances at the same notify
// sites as before and remains alive across expert/wave calls.
class Gmm1UbActivationSync {
public:
    __aicore__ explicit inline Gmm1UbActivationSync(uint16_t& pingpongIdx)
        : pingpongIdx_(pingpongIdx)
    {}

    __aicore__ inline Gmm1UbActivationSync(int32_t& submittedTiles, uint16_t& pingpongIdx)
        : pingpongIdx_(pingpongIdx),
          submittedTiles_(&submittedTiles)
    {}

    __aicore__ inline void WaitForActivation()
    {
        // 首次填满双缓冲后，复用槽位前等待 Activation 完成。
        if (*submittedTiles_ >= static_cast<int32_t>(DOUBLE_BUFFER)) {
            WaitForVector(pingpongIdx_);
        }
    }

    __aicore__ inline void NotifyActivation()
    {
        NotifyVector(pingpongIdx_);
        ++(*submittedTiles_);
        pingpongIdx_ = 1U - pingpongIdx_;
    }

    __aicore__ inline void WaitForGmm1()
    {
        WaitForCube(pingpongIdx_);
    }

    __aicore__ inline void NotifyGmm1()
    {
        NotifyCube(pingpongIdx_);
        pingpongIdx_ = 1U - pingpongIdx_;
    }

    __aicore__ static inline void EndSync(int32_t vecSetSyncCom, uint16_t pingpongIdx)
    {
        if (vecSetSyncCom == 0) {
            return;
        }
        if constexpr (g_coreType == AscendC::AIC) {
            if (vecSetSyncCom == 1) {
                WaitForVector(1U - pingpongIdx);
            } else {
                WaitForVector(pingpongIdx);
                WaitForVector(1U - pingpongIdx);
            }
        }
    }

private:
    uint16_t& pingpongIdx_;
    int32_t* submittedTiles_ = nullptr; // Required only for AIC submission.
};

// GMM1 -> Activation：A8W8/A4W4 使用 AIV0，A8W4 使用 AIV1。
class Gmm1ActivationSync {
public:
    __aicore__ explicit inline Gmm1ActivationSync(uint16_t activationSubBlockIdx,
                                                  GmmEventPair eventPair = GmmEventPair::MOE_GMM1)
        : aivFlagOffset_(activationSubBlockIdx * GMM_AIV_FLAG_OFFSET),
          gmm1FinishedFlag_(static_cast<uint16_t>(eventPair)),
          activationFinishedFlag_(gmm1FinishedFlag_ + 1U)
    {}

    // AIC：提交前预留额度。累计第 16 个 tile 起，每次消费一个 ACK；已到达的 ACK 可直接通过。
    // PIPE_S gates submission itself. PIPE_MTE2 would only stall loads and cannot
    // enforce this bound for work that can advance without another MTE2 load.
    __aicore__ inline void WaitForActivation()
    {
        if (pendingTiles_ == GMM_MAX_PENDING_TILES) {
            SyncIntraWait<PIPE_S>(activationFinishedFlag_ + aivFlagOffset_);
            --pendingTiles_;
        }
        ++pendingTiles_;
    }

    // AIC：整个交织 tile 的 Fixpipe 已提交，由 FIX 通知 Activation 读取 GM。
    __aicore__ inline void NotifyActivation()
    {
        SyncIntraArrive<PIPE_FIX>(gmm1FinishedFlag_ + aivFlagOffset_);
    }

    // AIV：在 MTE2 读取 GMM1 结果前等待。
    __aicore__ inline void WaitForGmm1()
    {
        SyncIntraWait<PIPE_MTE2>(gmm1FinishedFlag_);
    }

    // AIV：整个逻辑 tile 的 Activation/量化输出写回后，归还一个额度。
    __aicore__ inline void NotifyGmm1()
    {
        SyncIntraArrive<PIPE_MTE3>(activationFinishedFlag_);
    }

    // 主流程末尾统一排空；独立事件允许 ACK 跨过后续共享计算保留。
    __aicore__ inline void EndSync()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            while (pendingTiles_ != 0U) {
                SyncIntraWait<PIPE_S>(activationFinishedFlag_ + aivFlagOffset_);
                --pendingTiles_;
            }
        }
    }

private:
    // Mode 4：AIC 的 flagId 加 16 选择 AIV1；AIV 使用本地 flagId。
    const uint16_t aivFlagOffset_;
    const uint16_t gmm1FinishedFlag_;
    const uint16_t activationFinishedFlag_;
    uint32_t pendingTiles_ = 0; // 已提交的 tile 数减去 AIC 已消费的 Activation 完成通知数。
};

// GMM2 -> 非量化 Combine：arch35 使用 AIV1，arch92 使用唯一 AIV，事件和计数独立于 GMM1/Activation。
template <uint16_t CombineSubBlockIdx>
class Gmm2CombineSyncForSubBlock {
private:
    static constexpr uint16_t GMM2_FINISHED_FLAG = static_cast<uint16_t>(GmmEventPair::MOE_GMM2);
    static constexpr uint16_t COMBINE_FINISHED_FLAG = GMM2_FINISHED_FLAG + 1U;
    static constexpr uint16_t COMBINE_AIV_FLAG_OFFSET = CombineSubBlockIdx * GMM_AIV_FLAG_OFFSET;

public:
    // AIC：提交前预留额度。累计第 16 个 tile 起，每次消费一个 ACK；已到达的 ACK 可直接通过。
    // PIPE_S gates submission itself. PIPE_MTE2 would only stall loads and cannot
    // enforce this bound for work that can advance without another MTE2 load.
    __aicore__ inline void WaitForCombine()
    {
        if (pendingTiles_ == GMM_MAX_PENDING_TILES) {
            SyncIntraWait<PIPE_S>(COMBINE_FINISHED_FLAG + COMBINE_AIV_FLAG_OFFSET);
            --pendingTiles_;
        }
        ++pendingTiles_;
    }

    __aicore__ inline void NotifyCombine()
    {
        SyncIntraArrive<PIPE_FIX>(GMM2_FINISHED_FLAG + COMBINE_AIV_FLAG_OFFSET);
    }

    __aicore__ inline void WaitForGmm2()
    {
        SyncIntraWait<PIPE_MTE2>(GMM2_FINISHED_FLAG);
    }

    // Combine AIV 输出写回后，归还一个额度。
    __aicore__ inline void NotifyGmm2()
    {
        SyncIntraArrive<PIPE_MTE3>(COMBINE_FINISHED_FLAG);
    }

    __aicore__ inline void EndSync()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            while (pendingTiles_ != 0U) {
                SyncIntraWait<PIPE_S>(COMBINE_FINISHED_FLAG + COMBINE_AIV_FLAG_OFFSET);
                --pendingTiles_;
            }
        }
    }

private:
    uint32_t pendingTiles_ = 0;
};

#if defined(MEGA_MOE_ARCH92_DEVICE)
using Gmm2CombineSync = Gmm2CombineSyncForSubBlock<0U>;
#else
using Gmm2CombineSync = Gmm2CombineSyncForSubBlock<1U>;
#endif

} // namespace MegaMoeImpl
#endif
