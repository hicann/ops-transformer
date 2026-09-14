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

#include "kernel_operator.h"
#include "mega_moe_constants.h"

namespace MegaMoeImpl {

// 每条同步链路各允许 AIC 领先 15 个逻辑 tile，计数跨 expert/wave 保留。
// 每个 tile 的所有 GM 输出完成后通知消费者，消费者处理完所有子 tile 后通知 AIC。
constexpr uint32_t GMM_MAX_PENDING_TILES = 15;

// Mode 4 flag IDs: each pair holds a producer-ready and consumer-ACK event.
enum class GmmEventPair : uint16_t {
    MOE_GMM1 = 8,
    MOE_GMM2 = 10,
    SHARED_GMM1 = 12,
};
constexpr uint16_t GMM_AIV_FLAG_OFFSET = 16U;

// 轮询 GM 中的 int32 flag 直至等于期望值，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagEquals(__gm__ int32_t *flagAddr, int32_t expectedValue,
                                             int64_t pollBackoffCycles = GM_FLAG_POLL_BACKOFF_CYCLES)
{
    while (AscendC::ReadGmByPassDCache(flagAddr) != expectedValue) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < pollBackoffCycles) {
        }
    }
}

// 轮询 GM 中的 int32 计数直至不小于目标值，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagAtLeast(__gm__ int32_t *flagAddr, int32_t targetValue)
{
    while (AscendC::ReadGmByPassDCache(flagAddr) < targetValue) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}

constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

// Non-owning view of the caller's UB state. State advances at the same notify
// sites as before and remains alive across expert/wave calls.
class Gmm1UbActivationSync {
public:
    __aicore__ explicit inline Gmm1UbActivationSync(uint16_t &pingpongIdx)
        : pingpongIdx_(pingpongIdx)
    {}

    __aicore__ inline Gmm1UbActivationSync(int32_t &submittedTiles, uint16_t &pingpongIdx)
        : pingpongIdx_(pingpongIdx),
          submittedTiles_(&submittedTiles)
    {}

    __aicore__ inline void WaitForActivation()
    {
        if (*submittedTiles_ >= 2) {
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
    uint16_t &pingpongIdx_;
    int32_t *submittedTiles_ = nullptr; // Required only for AIC submission.
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
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(activationFinishedFlag_ + aivFlagOffset_);
            --pendingTiles_;
        }
        ++pendingTiles_;
    }

    // AIC：整个交织 tile 的 Fixpipe 已提交，由 FIX 通知 Activation 读取 GM。
    __aicore__ inline void NotifyActivation()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(gmm1FinishedFlag_ + aivFlagOffset_);
    }

    // AIV：在 MTE2 读取 GMM1 结果前等待。
    __aicore__ inline void WaitForGmm1()
    {
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(gmm1FinishedFlag_);
    }

    // AIV：整个逻辑 tile 的 Activation/量化输出写回后，归还一个额度。
    __aicore__ inline void NotifyGmm1()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(activationFinishedFlag_);
    }

    // MTE 主流程末尾统一排空；独立事件允许 ACK 跨过后续共享计算保留。
    __aicore__ inline void EndSync()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            while (pendingTiles_ != 0U) {
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(activationFinishedFlag_ + aivFlagOffset_);
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

// GM path facade: hardware ACKs and Layered GM flags keep their original
// protocols. AddressInfo is referenced because each problem updates its GM bases.
// Prefetch uses per-tile expert tags; non-prefetch uses the caller's sequence.
template <bool TopkWeightsPrefetch, typename AddressInfo>
class Gmm1GmActivationSync {
public:
    __aicore__ explicit inline Gmm1GmActivationSync(const AddressInfo &addresses, int32_t *sequence = nullptr)
        : addresses_(addresses),
          sequence_(sequence)
    {}

    __aicore__ inline void WaitForActivation()
    {
        if (addresses_.gmm1ActivationSync != nullptr) {
            addresses_.gmm1ActivationSync->WaitForActivation();
        }
    }

    __aicore__ inline void NotifyActivation(uint32_t expertIdx, uint32_t loopIdx)
    {
        if (addresses_.gmm1ActivationSync != nullptr) {
            addresses_.gmm1ActivationSync->NotifyActivation();
            if constexpr (!TopkWeightsPrefetch) {
                ++(*sequence_);
            }
        } else {
            AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
            if constexpr (TopkWeightsPrefetch) {
                __gm__ int32_t *status = addresses_.gmm1TileStatus + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
                AscendC::WriteGmByPassDCache(status, static_cast<int32_t>(expertIdx + 1));
            } else {
                AscendC::WriteGmByPassDCache(addresses_.gmmToEpilogueFlag, ++(*sequence_));
            }
        }
    }

    __aicore__ inline void WaitForGmm1(uint32_t expertIdx = 0U, uint32_t loopIdx = 0U)
    {
        if constexpr (!TopkWeightsPrefetch) {
            expectedSequence_ = *sequence_ + 1;
        }
        if (addresses_.gmm1ActivationSync != nullptr) {
            addresses_.gmm1ActivationSync->WaitForGmm1();
        } else if constexpr (TopkWeightsPrefetch) {
            __gm__ int32_t *status = addresses_.gmm1TileStatus + static_cast<uint64_t>(loopIdx) * INT_CACHELINE;
            int32_t roundTag = static_cast<int32_t>(expertIdx + 1);
            WaitUntilGmFlagEquals(status, roundTag);
        } else {
            WaitUntilGmFlagAtLeast(addresses_.gmmToEpilogueFlag, expectedSequence_);
        }
    }

    __aicore__ inline void NotifyGmm1()
    {
        if constexpr (!TopkWeightsPrefetch) {
            *sequence_ = expectedSequence_;
        }
        if (addresses_.gmm1ActivationSync != nullptr) {
            addresses_.gmm1ActivationSync->NotifyGmm1();
        }
    }

private:
    const AddressInfo &addresses_;
    int32_t *sequence_;
    int32_t expectedSequence_ = 0;
};

// Layered prefetch's existing final GM handshake, separate from per-tile ACKs.
class Gmm1GmCompletionSync {
public:
    __aicore__ inline Gmm1GmCompletionSync(__gm__ int32_t *flag, int32_t tag, uint32_t subBlockIdx,
                                           uint32_t activationSubBlockIdx)
        : flag_(flag),
          tag_(tag),
          subBlockIdx_(subBlockIdx),
          activationSubBlockIdx_(activationSubBlockIdx)
    {}

    __aicore__ inline void EndSync()
    {
        if constexpr (g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == activationSubBlockIdx_) {
                AscendC::WriteGmByPassDCache(flag_, tag_);
            }
        } else {
            WaitUntilGmFlagEquals(flag_, tag_);
        }
    }

private:
    __gm__ int32_t *flag_;
    int32_t tag_;
    uint32_t subBlockIdx_;
    uint32_t activationSubBlockIdx_;
};

// GMM2 -> 非量化 Combine：固定使用 AIV1，事件和计数独立于 GMM1/Activation。
class Gmm2CombineSync {
public:
    // AIC：提交前预留额度。累计第 16 个 tile 起，每次消费一个 ACK；已到达的 ACK 可直接通过。
    // PIPE_S gates submission itself. PIPE_MTE2 would only stall loads and cannot
    // enforce this bound for work that can advance without another MTE2 load.
    __aicore__ inline void WaitForCombine()
    {
        if (pendingTiles_ == GMM_MAX_PENDING_TILES) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(COMBINE_FINISHED_FLAG + GMM_AIV_FLAG_OFFSET);
            --pendingTiles_;
        }
        ++pendingTiles_;
    }

    __aicore__ inline void NotifyCombine()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(GMM2_FINISHED_FLAG + GMM_AIV_FLAG_OFFSET);
    }

    __aicore__ inline void WaitForGmm2()
    {
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(GMM2_FINISHED_FLAG);
    }

    // AIV1：Combine 输出写回后，归还一个额度。
    __aicore__ inline void NotifyGmm2()
    {
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(COMBINE_FINISHED_FLAG);
    }

    __aicore__ inline void EndSync()
    {
        if constexpr (g_coreType == AscendC::AIC) {
            while (pendingTiles_ != 0U) {
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(COMBINE_FINISHED_FLAG + GMM_AIV_FLAG_OFFSET);
                --pendingTiles_;
            }
        }
    }

private:
    static constexpr uint16_t GMM2_FINISHED_FLAG = static_cast<uint16_t>(GmmEventPair::MOE_GMM2);
    static constexpr uint16_t COMBINE_FINISHED_FLAG = GMM2_FINISHED_FLAG + 1U;
    uint32_t pendingTiles_ = 0;
};

} // namespace MegaMoeImpl
#endif
