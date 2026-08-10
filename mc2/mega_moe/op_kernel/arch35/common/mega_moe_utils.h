/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_utils.h
 * \brief MegaMoe 调度使用的通用 host/device 辅助函数。
 */

#ifndef MEGA_MOE_UTILS_H
#define MEGA_MOE_UTILS_H

#include <cstdint>

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "mega_moe_constants.h"
#include "mega_moe_types.h"

namespace MegaMoeImpl {

using namespace AscendC;

constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;

struct WorkRange {
    uint32_t start = 0;
    uint32_t count = 0;
};

__aicore__ inline WorkRange TilingByJobContext(uint32_t totalLen, uint32_t jobIndex,
                                               uint32_t totalJobs, uint32_t align = ALIGN_32)
{
    if (totalJobs == 0U || jobIndex >= totalJobs) {
        return {};
    }
    uint32_t lenPerJob = Ops::Base::CeilDiv(totalLen, totalJobs);
    uint32_t alignedLenPerJob = Ops::Base::CeilAlign(lenPerJob, align);
    uint32_t jobOffset = jobIndex * alignedLenPerJob;
    if (jobOffset >= totalLen) {
        return {};
    }
    uint32_t jobLen = jobOffset + alignedLenPerJob > totalLen ? totalLen - jobOffset : alignedLenPerJob;
    return {jobOffset, jobLen};
}

__aicore__ inline GroupSyncSlotLayout CalcGroupSyncSlotLayout(uint32_t expertTokenCount, uint32_t logicalCoreCount)
{
    uint32_t groupCount = Ops::Base::CeilDiv(expertTokenCount, COMBINE_TOKEN_GROUP_SIZE);
    uint32_t totalSyncSlotCount = groupCount > logicalCoreCount ? groupCount : logicalCoreCount;
    return {totalSyncSlotCount / groupCount, totalSyncSlotCount % groupCount};
}

__aicore__ inline __gm__ int32_t *GetCombineSyncCounterAddress(__gm__ int32_t *expertCounterBase,
                                                               uint32_t localSyncSlotIndex)
{
    return expertCounterBase + static_cast<uint64_t>(localSyncSlotIndex) * INT_CACHELINE;
}

__aicore__ inline void GetGroupSyncSlotRange(uint32_t groupIndex, const GroupSyncSlotLayout &slotLayout,
                                             uint32_t &firstSyncSlot, uint32_t &syncSlotCount)
{
    syncSlotCount = slotLayout.baseSlotCountPerGroup + (groupIndex < slotLayout.extraSlotGroupCount ? 1U : 0U);
    uint32_t precedingExtraSlotCount =
        groupIndex < slotLayout.extraSlotGroupCount ? groupIndex : slotLayout.extraSlotGroupCount;
    firstSyncSlot = groupIndex * slotLayout.baseSlotCountPerGroup + precedingExtraSlotCount;
}

__aicore__ inline void ComputeCombineGroupsForCore(uint32_t logicalCoreId, uint32_t groupCount,
                                                   uint32_t logicalCoreCount, uint32_t &firstGroupIndex,
                                                   uint32_t &groupStride, uint32_t &coreIndexWithinGroup,
                                                   uint32_t &coresAssignedToGroup)
{
    uint32_t totalSyncSlotCount = groupCount > logicalCoreCount ? groupCount : logicalCoreCount;
    uint32_t baseSlotCountPerGroup = totalSyncSlotCount / groupCount;
    uint32_t extraSlotGroupCount = totalSyncSlotCount % groupCount;
    uint32_t slotBoundaryAfterExtraSlotGroups = extraSlotGroupCount * (baseSlotCountPerGroup + 1U);

    if (logicalCoreId < slotBoundaryAfterExtraSlotGroups) {
        coresAssignedToGroup = baseSlotCountPerGroup + 1U;
        firstGroupIndex = logicalCoreId / coresAssignedToGroup;
        coreIndexWithinGroup = logicalCoreId % coresAssignedToGroup;
    } else {
        uint32_t slotOffsetAfterExtraSlotGroups = logicalCoreId - slotBoundaryAfterExtraSlotGroups;
        coresAssignedToGroup = baseSlotCountPerGroup;
        firstGroupIndex = extraSlotGroupCount + slotOffsetAfterExtraSlotGroups / coresAssignedToGroup;
        coreIndexWithinGroup = slotOffsetAfterExtraSlotGroups % coresAssignedToGroup;
    }

    groupStride = groupCount < logicalCoreCount ? groupCount : logicalCoreCount;
}

// 返回当前 wave 的专家数；不足正常 wave 一半的尾块合并到前一个 wave。
__aicore__ inline uint32_t GetWaveExpertCount(uint32_t waveBegin, uint32_t expertCount, uint32_t expertsPerWave)
{
    if (waveBegin >= expertCount) {
        return 0U;
    }
    uint32_t remainingExperts = expertCount - waveBegin;
    if (expertsPerWave == 0U || remainingExperts <= expertsPerWave) {
        return remainingExperts;
    }
    uint32_t tailExperts = remainingExperts - expertsPerWave;
    bool mergeSmallTail = tailExperts < expertsPerWave && 2U * tailExperts < expertsPerWave;
    return mergeSmallTail ? remainingExperts : expertsPerWave;
}

// 连续均衡分配 token，前 totalTokens % workerCount 个任务各多处理一个 token。
__aicore__ inline WorkRange GetBalancedTokenRange(uint32_t totalTokens, uint32_t workerIdx, uint32_t workerCount)
{
    if (workerCount == 0U || workerIdx >= workerCount) {
        return {};
    }
    uint32_t base = totalTokens / workerCount;
    uint32_t remainder = totalTokens % workerCount;
    uint32_t extraBefore = workerIdx < remainder ? workerIdx : remainder;
    return {workerIdx * base + extraBefore, base + static_cast<uint32_t>(workerIdx < remainder)};
}

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
// 使用调用方准备的全零 UB Tensor，清理当前 AIV 任务负责的连续 GM 区域。
template <int32_t JobAlignment>
__aicore__ inline void ResetWorkspaceRegion(const AivJobContext &job, GM_ADDR regionPtr, int32_t elementCount,
                                            int32_t batchElementCount, LocalTensor<int32_t> &resetTensor)
{
    WorkRange range = TilingByJobContext(static_cast<uint32_t>(elementCount), job.jobIndex,
                                         job.totalJobs, static_cast<uint32_t>(JobAlignment));
    GlobalTensor<int32_t> regionGm;
    regionGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(regionPtr));
    for (uint32_t offset = 0; offset < range.count; offset += static_cast<uint32_t>(batchElementCount)) {
        uint32_t batchCount = range.count - offset < static_cast<uint32_t>(batchElementCount) ?
                                  range.count - offset :
                                  static_cast<uint32_t>(batchElementCount);
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(regionGm[range.start + offset], resetTensor, copyParams);
    }
}

#endif

__aicore__ inline void NotifyCube(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void WaitForVector(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}

__aicore__ inline void NotifyVector(uint16_t value = 0)
{
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

__aicore__ inline void WaitForCube(uint16_t value = 0)
{
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

template <bool IsPingPong = false>
__aicore__ inline void EndSync(int32_t vecSetSyncCom, uint16_t pingpongIdx = 0)
{
    if (vecSetSyncCom == 0) {
        return;
    }
    if constexpr (g_coreType == AIC) {
        if constexpr (IsPingPong) {
            if (vecSetSyncCom == 1) {
                WaitForVector(1U - pingpongIdx);
            } else {
                WaitForVector(pingpongIdx);
                WaitForVector(1U - pingpongIdx);
            }
        } else {
            WaitForVector();
        }
    }
}

__aicore__ inline void EndGMM2Sync(int32_t &vecSetSyncCom, uint16_t gmm2PingPongIdx)
{
    if constexpr (g_coreType == AIV) {
        return;
    }
    if (vecSetSyncCom <= 0) {
        return;
    } else if (vecSetSyncCom == 1) {
        WaitForVector();
    } else {
        WaitForVector(gmm2PingPongIdx);
        WaitForVector(1U - gmm2PingPongIdx);
    }
}

__aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen, int32_t &coreOffset, int32_t align = ALIGN_32)
{
    int32_t coreIdx = GetBlockIdx();
    int32_t coreNum = GetBlockNum() * 2;
    int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
    int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
    coreLen = lenPerCoreAlign;
    coreOffset = coreIdx * lenPerCoreAlign;
    if (coreOffset + coreLen >= totalLen) {
        coreLen = totalLen - coreOffset;
    }
    if (coreOffset >= totalLen) {
        coreLen = 0;
    }
}

__aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sigAddr, int32_t compareValue)
{
    do {
        if (ReadGmByPassDCache(sigAddr) == compareValue) {
            return;
        }
    } while (true);
}

__aicore__ inline uint64_t GetUrmaCommHandle(__gm__ Mc2MoeContext *mc2Context, uint32_t rankId, uint32_t epRankId)
{
    uint32_t index = rankId > epRankId ? rankId - 1U : rankId;
    return mc2Context->hcommHandle[index];
}

inline GM_ADDR winRankAddr_[HCCL_MAX_RANK_SIZE];

__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(winRankAddr_[rankId] + offset);
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

template <typename ElementType>
__aicore__ inline GM_ADDR GetExpertWeightAddr(GM_ADDR tensorListAddr, bool isPerExpertWeightTensor,
                                              uint32_t expertIdx, uint64_t elementOffset)
{
    uint16_t tensorIdx = isPerExpertWeightTensor ? static_cast<uint16_t>(expertIdx) : 0U;
    GM_ADDR tensorAddr = GetTensorAddr(tensorIdx, tensorListAddr);
    if (isPerExpertWeightTensor) {
        return tensorAddr;
    }
    return tensorAddr + elementOffset * sizeof(ElementType);
}

template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&value)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(value));
}

template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T &&value)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(value)));
}

template <AscendC::HardEvent event, int32_t eventId>
__aicore__ inline void SyncFuncStatic()
{
    AscendC::SetFlag<event>(static_cast<event_t>(eventId));
    AscendC::WaitFlag<event>(static_cast<event_t>(eventId));
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_UTILS_H
