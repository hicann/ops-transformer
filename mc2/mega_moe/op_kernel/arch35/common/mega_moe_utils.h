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
#include "mega_moe_gmm_epilogue_sync.h"
#include "mega_moe_types.h"

namespace MegaMoeImpl {

using namespace AscendC;

constexpr uint32_t UINT64_BYTE_OFFSET_SHIFT = 3U;

__aicore__ inline __gm__ int32_t* GetSyncCountAddress(GM_ADDR rankSyncBase, uint32_t aivCoreIdx)
{
    return reinterpret_cast<__gm__ int32_t*>(rankSyncBase + RANK_SYNC_COUNTER_OFFSET_BYTES +
                                             static_cast<uint64_t>(aivCoreIdx) * RANK_SYNC_COUNTER_SLOT_BYTES);
}

// 推进指定物理 AIV 的同步轮次，仅负责计数，不表示量化数据已写回。
__aicore__ inline void IncrementSyncCount(GM_ADDR rankSyncBase, uint32_t aivCoreIdx)
{
    auto* sequenceAddr = GetSyncCountAddress(rankSyncBase, aivCoreIdx);
    uint32_t nextSequence = static_cast<uint32_t>(ReadGmByPassDCache(sequenceAddr)) + 1U;
    WriteGmByPassDCache(sequenceAddr, static_cast<int32_t>(nextSequence));
}

__aicore__ inline uint32_t GetSyncRoundTag(int32_t syncCount)
{
    // & 0x7F：保留同步计数的低 7 位，用于区分轮次；计数相差 128 时标签相同。
    // | 0x80：将标签的最高位设为 1，使其落在 [0x80, 0xFF]，避开接收区的零初值。
    // << 24：将标签放入 count 的高 8 位，低 24 位留给数量；接收时掩出高 8 位直接比较。
    return ((static_cast<uint32_t>(syncCount) & 0x7FU) | 0x80U) << 24U;
}

/*
 * 对一段连续 int32 数据原地执行 inclusive prefix-scan。
 *
 * 路由计数全程保持 int32 精确累加。本实现专用于一段连续的一维数据，不需要通用多维 CumSum 的
 * 维度转换和额外临时 UB：每个向量寄存器内使用 Hillis-Steele scan，相邻寄存器通过上一寄存器
 * 末值 carry 串接。
 *
 * 本函数是 VF 内部的可复用计算单元，其他 VF 可直接调用，不会产生额外的 VF 启动。
 */
// InputMask 可在加载时去除输入中的标记位，默认保留完整 int32 数值。
template <int32_t InputMask = -1>
__simd_callee__ inline void InclusivePrefixSumInt32(__ubuf__ int32_t* values, uint32_t elementCount)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = GetVecLen() / sizeof(int32_t);
    AscendC::Reg::RegTensor<int32_t> prefixReg;
    AscendC::Reg::RegTensor<int32_t> shiftedReg;
    AscendC::Reg::RegTensor<int32_t> carryReg;
    AscendC::Reg::RegTensor<int32_t> laneIndexReg;
    AscendC::Reg::RegTensor<int32_t> shiftedIndexReg;
    AscendC::Reg::RegTensor<int32_t> inputMaskReg;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Arange(laneIndexReg, 0);
    AscendC::Reg::Duplicate(carryReg, 0, fullMask);
    if constexpr (InputMask != -1) {
        AscendC::Reg::Duplicate(inputMaskReg, InputMask, fullMask);
    }
    for (uint32_t vectorOffset = 0U; vectorOffset < elementCount; vectorOffset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = elementCount - vectorOffset;
        validCount = validCount < ELEMENTS_PER_VECTOR ? validCount : ELEMENTS_PER_VECTOR;
        uint32_t maskCount = validCount;
        AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<int32_t>(maskCount);
        AscendC::Reg::LoadAlign(prefixReg, values + vectorOffset);
        if constexpr (InputMask != -1) {
            AscendC::Reg::And(prefixReg, prefixReg, inputMaskReg, activeMask);
        }

        for (uint32_t shift = 1U; shift < validCount; shift <<= 1U) {
            AscendC::Reg::Adds(shiftedIndexReg, laneIndexReg, -static_cast<int32_t>(shift), fullMask);
            AscendC::Reg::Maxs(shiftedIndexReg, shiftedIndexReg, 0, fullMask);
            AscendC::Reg::Gather(shiftedReg, prefixReg,
                                 reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(shiftedIndexReg));
            AscendC::Reg::MaskReg shiftedLaneMask;
            AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::GE>(shiftedLaneMask, laneIndexReg,
                                                                  static_cast<int32_t>(shift), activeMask);
            AscendC::Reg::Add<int32_t, AscendC::Reg::MaskMergeMode::MERGING>(prefixReg, prefixReg, shiftedReg,
                                                                             shiftedLaneMask);
        }

        AscendC::Reg::Add<int32_t, AscendC::Reg::MaskMergeMode::MERGING>(prefixReg, prefixReg, carryReg, activeMask);
        AscendC::Reg::StoreAlign(values + vectorOffset, prefixReg, activeMask);

        AscendC::Reg::Duplicate(shiftedIndexReg, static_cast<int32_t>(validCount - 1U), fullMask);
        AscendC::Reg::Gather(carryReg, prefixReg,
                             reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(shiftedIndexReg));
    }
}

__aicore__ inline WorkRange TilingByJobContext(uint32_t totalLen, uint32_t jobIndex, uint32_t totalJobs,
                                               uint32_t align = ALIGN_32)
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

__aicore__ inline __gm__ int32_t* GetCombineSyncCounterAddress(__gm__ int32_t* expertCounterBase,
                                                               uint32_t localSyncSlotIndex)
{
    return expertCounterBase + static_cast<uint64_t>(localSyncSlotIndex) * INT_CACHELINE;
}

__aicore__ inline void GetGroupSyncSlotRange(uint32_t groupIndex, const GroupSyncSlotLayout& slotLayout,
                                             uint32_t& firstSyncSlot, uint32_t& syncSlotCount)
{
    syncSlotCount = slotLayout.baseSlotCountPerGroup + (groupIndex < slotLayout.extraSlotGroupCount ? 1U : 0U);
    uint32_t precedingExtraSlotCount =
        groupIndex < slotLayout.extraSlotGroupCount ? groupIndex : slotLayout.extraSlotGroupCount;
    firstSyncSlot = groupIndex * slotLayout.baseSlotCountPerGroup + precedingExtraSlotCount;
}

/*
 * 计算单个专家贡献的独立 M 分组数。不同专家使用不同权重，因此各专家不足
 * rowsPerMGroup 的尾块也必须分别占用一个分组。
 */
__aicore__ inline uint32_t GetMGroupCountForRows(uint64_t rowCount, uint32_t rowsPerMGroup)
{
    if (rowCount == 0U || rowsPerMGroup == 0U) {
        return 0U;
    }
    return static_cast<uint32_t>((rowCount - 1U) / rowsPerMGroup + 1U);
}

// 当前 AIC 未分到该 Wave problem 的任何 tile 时，补偿推进 rolling 游标并返回 true。
__aicore__ inline bool HandleWaveProblemWithoutWork(uint32_t problemTileCount, const BlockJobContext& blockJob,
                                                    uint32_t& startBlockIdx)
{
    if constexpr (g_coreType == AIC) {
        if (problemTileCount < blockJob.totalJobs) {
            uint32_t firstOwnedTile =
                (blockJob.jobIndex < startBlockIdx ? blockJob.jobIndex + blockJob.totalJobs : blockJob.jobIndex) -
                startBlockIdx;
            if (firstOwnedTile >= problemTileCount) {
                startBlockIdx = (startBlockIdx + problemTileCount) % blockJob.totalJobs;
                return true;
            }
        }
    }
    return false;
}

// 计算剩余分组预算允许当前 Wave 在本专家内推进到的结束行偏移。
__aicore__ inline uint32_t GetWaveEndRowOffsetInExpert(uint64_t expertRowCount, uint32_t currentRowOffsetInExpert,
                                                       uint32_t remainingMGroupCount, uint32_t rowsPerMGroup)
{
    if (remainingMGroupCount == 0U || rowsPerMGroup == 0U || currentRowOffsetInExpert >= expertRowCount) {
        return currentRowOffsetInExpert;
    }
    uint64_t maxWaveEndRowOffset =
        static_cast<uint64_t>(currentRowOffsetInExpert) + static_cast<uint64_t>(remainingMGroupCount) * rowsPerMGroup;
    return static_cast<uint32_t>(expertRowCount < maxWaveEndRowOffset ? expertRowCount : maxWaveEndRowOffset);
}

// 连续均衡分配工作项，前 totalWorkItems % job.totalJobs 个任务各多处理一项。
__aicore__ inline WorkRange GetBalancedWorkRange(uint32_t totalWorkItems, const AivJobContext& job)
{
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs) {
        return {};
    }
    uint32_t base = totalWorkItems / job.totalJobs;
    uint32_t remainder = totalWorkItems % job.totalJobs;
    uint32_t extraBefore = job.jobIndex < remainder ? job.jobIndex : remainder;
    return {job.jobIndex * base + extraBefore, base + static_cast<uint32_t>(job.jobIndex < remainder)};
}

// 以全局任务前缀轮转“多一个任务”的首 owner；Dispatch/Combine 共用这一公式。
__aicore__ inline WorkRange GetRotatedBalancedWorkRange(uint32_t totalWorkItems, uint32_t workerIdx,
                                                        uint32_t workerCount, uint64_t globalWorkPrefix)
{
    if (workerCount == 0U || workerIdx >= workerCount) {
        return {};
    }
    uint32_t firstOwner = static_cast<uint32_t>(globalWorkPrefix % workerCount);
    uint32_t logicalWorkerIdx = workerIdx >= firstOwner ? workerIdx - firstOwner : workerIdx + workerCount - firstOwner;
    return GetBalancedWorkRange(totalWorkItems, {.jobIndex = logicalWorkerIdx, .totalJobs = workerCount});
}

// 按 strideBytes 字节间距清零 int32；range 以逻辑元素为单位，strideBytes 至少为 4 且为 4 的倍数。
// 调用方提供已填零、非空且 32B 对齐的 UB Tensor，并保护其 MTE3 生命周期。
__aicore__ inline void ResetWorkspaceRegionWithStride(const WorkRange& range, GM_ADDR regionPtr, uint64_t strideBytes,
                                                      LocalTensor<int32_t>& resetTensor)
{
    GlobalTensor<int32_t> regionGm;
    regionGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(regionPtr));
    const uint32_t batchCapacity = resetTensor.GetSize() < UINT16_MAX ? resetTensor.GetSize() : UINT16_MAX;
    for (uint32_t offset = 0U; offset < range.count;) {
        const uint32_t remaining = range.count - offset;
        const uint32_t batchCount = remaining < batchCapacity ? remaining : batchCapacity;
        const uint64_t dstElement = (static_cast<uint64_t>(range.start) + offset) * (strideBytes / sizeof(int32_t));
        DataCopyExtParams copyParams{static_cast<uint16_t>(batchCount), sizeof(int32_t), 0U,
                                     static_cast<int64_t>(strideBytes - sizeof(int32_t)), 0U};
        DataCopyPad<int32_t, PaddingMode::Compact>(regionGm[dstElement], resetTensor, copyParams);
        offset += batchCount;
    }
}

// 使用调用方准备的全零 UB Tensor 清理指定范围；range 以 int32 元素为单位。
__aicore__ inline void ResetWorkspaceRegion(const WorkRange& range, GM_ADDR regionPtr, int32_t batchElementCount,
                                            LocalTensor<int32_t>& resetTensor)
{
    GlobalTensor<int32_t> regionGm;
    regionGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(regionPtr));
    for (uint32_t offset = 0; offset < range.count; offset += static_cast<uint32_t>(batchElementCount)) {
        uint32_t batchCount = range.count - offset < static_cast<uint32_t>(batchElementCount) ?
                                  range.count - offset :
                                  static_cast<uint32_t>(batchElementCount);
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(batchCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(regionGm[range.start + offset], resetTensor, copyParams);
    }
}

// 按任务划分对齐范围，再复用范围版本执行清零。
template <int32_t JobAlignment>
__aicore__ inline void ResetWorkspaceRegion(const AivJobContext& job, GM_ADDR regionPtr, int32_t elementCount,
                                            int32_t batchElementCount, LocalTensor<int32_t>& resetTensor)
{
    const WorkRange range = TilingByJobContext(static_cast<uint32_t>(elementCount), job.jobIndex, job.totalJobs,
                                               static_cast<uint32_t>(JobAlignment));
    ResetWorkspaceRegion(range, regionPtr, batchElementCount, resetTensor);
}

__aicore__ inline void TilingByCore(int32_t totalLen, int32_t& coreLen, int32_t& coreOffset, int32_t align = ALIGN_32)
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

// 普通路径沿用 [expert][block] 的 32B 槽；Wave 一次发布完整表后使用
// [block][expert] 连续布局，减少逐专家元数据同步。
__aicore__ inline uint64_t GetExpertCountWorkspaceOffset(const BlockWorkspaceContext& workspace, uint32_t expertCount,
                                                         uint32_t expertIdx, bool useBlockMajorLayout)
{
    if (useBlockMajorLayout) {
        uint32_t blockMajorStride = Ops::Base::CeilAlign(expertCount, static_cast<uint32_t>(INT_CACHELINE));
        return static_cast<uint64_t>(workspace.blockIdx) * blockMajorStride + expertIdx;
    }
    return static_cast<uint64_t>(expertIdx) * INT32_PER_256B * workspace.blockNum +
           static_cast<uint64_t>(INT32_PER_256B) * workspace.blockIdx;
}

__aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t* sigAddr, int32_t compareValue)
{
    do {
        if (ReadGmByPassDCache(sigAddr) == compareValue) {
            return;
        }
    } while (true);
}

__aicore__ inline uint64_t GetUrmaCommHandle(__gm__ Mc2MoeContext* mc2Context, uint32_t rankId, uint32_t epRankId)
{
    uint32_t index = rankId > epRankId ? rankId - 1U : rankId;
    return mc2Context->hcommHandle_[index];
}

inline GM_ADDR g_winRankAddr_[HCCL_MAX_RANK_SIZE];

__aicore__ inline GM_ADDR GetRankWinAddrWithOffset(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(g_winRankAddr_[rankId] + offset);
}

__aicore__ inline GM_ADDR GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> UINT64_BYTE_OFFSET_SHIFT);
    return reinterpret_cast<GM_ADDR>(*(retPtr + index));
}

template <typename ElementType>
__aicore__ inline GM_ADDR GetExpertWeightAddr(GM_ADDR tensorListAddr, bool isPerExpertWeightTensor, uint32_t expertIdx,
                                              uint64_t elementOffset)
{
    uint16_t tensorIdx = isPerExpertWeightTensor ? static_cast<uint16_t>(expertIdx) : 0U;
    GM_ADDR tensorAddr = GetTensorAddr(tensorIdx, tensorListAddr);
    if (isPerExpertWeightTensor) {
        return tensorAddr;
    }
    return tensorAddr + elementOffset * sizeof(ElementType);
}

template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& value)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(value));
}

template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& value)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(value)));
}

__aicore__ inline ExpertLoopState CreateExpertLoopState(const MoeStageCommonConfig& context)
{
    ExpertLoopState state;
    Get<N_VALUE>(state.problemShape) = context.gmm1OutputDim;
    Get<K_VALUE>(state.problemShape) = context.tokenHiddenDim;
    return state;
}

// 按专家顺序推进连续 token 行偏移，并用当前专家的 token 数更新 M 维度。
__aicore__ inline void UpdateExpertLoopState(ExpertLoopState& state, uint32_t expertIdx, uint64_t expertTokenCount)
{
    if (expertIdx != state.expertIdx) {
        state.globalTokenStartIndex += Get<M_VALUE>(state.problemShape);
    }
    state.expertIdx = expertIdx;
    Get<M_VALUE>(state.problemShape) = static_cast<int64_t>(expertTokenCount);
}

// 按当前 Wave 的剩余容量推进专家 token 位置，返回本次推进覆盖的 token 数。
template <uint32_t TileM>
__aicore__ inline uint32_t AdvanceExpertTokenPositionInWave(uint32_t expertTokenCount, uint32_t waveMGroupTarget,
                                                            uint32_t& waveMGroupCount, ExpertTokenPosition& position)
{
    uint32_t expertRemainingTokenCount = expertTokenCount - position.tokenIndexInExpert;
    uint32_t waveRemainingTokenCapacity = (waveMGroupTarget - waveMGroupCount) * TileM;
    uint32_t sliceTokenCount =
        expertRemainingTokenCount < waveRemainingTokenCapacity ? expertRemainingTokenCount : waveRemainingTokenCapacity;
    waveMGroupCount += Ops::Base::CeilDiv(sliceTokenCount, TileM);
    position.tokenIndexInExpert += sliceTokenCount;
    position.globalTokenIndex += sliceTokenCount;
    if (position.tokenIndexInExpert >= expertTokenCount) {
        ++position.expertIdx;
        position.tokenIndexInExpert = 0U;
    }
    return sliceTokenCount;
}

// 轮询 GM 中的 int32 ready flag，并在两次读取之间加入短暂退避。
__aicore__ inline void WaitUntilGmFlagIsNonZero(__gm__ int32_t* flagAddr)
{
    while (AscendC::ReadGmByPassDCache(flagAddr) == 0) {
        int64_t startCycle = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - startCycle < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}

/*
 * 等待 Dispatch 发布当前 expert/block 的 token 总数。这里只同步专家级元数据；每个 256-row group
 * 的 Dispatch 数据就绪依赖由 WaitForGmm1InputReady 处理。
 */
__aicore__ inline void WaitForMoeExpertTokenCountReady(GM_ADDR tokenCountReadyFlagWorkspace,
                                                       const BlockWorkspaceContext& countWorkspace, uint32_t expertIdx)
{
    __gm__ int32_t* tokenCountReadyFlag = reinterpret_cast<__gm__ int32_t*>(tokenCountReadyFlagWorkspace) +
                                          static_cast<uint64_t>(expertIdx) * countWorkspace.blockNum * INT_CACHELINE +
                                          static_cast<uint64_t>(countWorkspace.blockIdx) * INT_CACHELINE;
    WaitUntilGmFlagIsNonZero(tokenCountReadyFlag);
}

// 从当前 block 的 [block][expert] workspace 行读取专家 token 数。
__aicore__ inline uint32_t GetExpertTokenCountFromWorkspace(GM_ADDR expertTokenCountWorkspace,
                                                            const BlockWorkspaceContext& countWorkspace,
                                                            uint32_t expertCount, uint32_t expertIdx)
{
    uint64_t countSlotIndex = GetExpertCountWorkspaceOffset(countWorkspace, expertCount, expertIdx, true);
    __gm__ int32_t* expertTokenCountAddr =
        reinterpret_cast<__gm__ int32_t*>(expertTokenCountWorkspace) + countSlotIndex;
    return static_cast<uint32_t>(AscendC::ReadGmByPassDCache(expertTokenCountAddr));
}

/*
 * 在当前 WAVE 的剩余容量内规划下一个非空专家 slice。
 * 单次调用最多返回一个非空专家范围；空专家会被跳过。函数只更新当前 WAVE 的 M-group 计数，
 * 不修改调用方持有的 Dispatch 位置；调用方在 Dispatch 完成后使用返回范围的 end 提交进度。
 */
template <uint32_t TileM>
__aicore__ inline ExpertTokenRange PlanNextExpertTokenRangeInWave(GM_ADDR expertTokenCountWorkspace,
                                                                  const BlockWorkspaceContext& countWorkspace,
                                                                  uint32_t expertCount, uint32_t waveMGroupTarget,
                                                                  uint32_t& waveMGroupCount,
                                                                  const ExpertTokenPosition& position)
{
    ExpertTokenPosition plannedPosition = position;
    while (plannedPosition.expertIdx < expertCount && waveMGroupCount < waveMGroupTarget) {
        uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(expertTokenCountWorkspace, countWorkspace,
                                                                     expertCount, plannedPosition.expertIdx);
        // 空专家或已完成专家不占用 WAVE 容量，继续寻找下一个非空专家。
        if (expertTokenCount == 0U || plannedPosition.tokenIndexInExpert >= expertTokenCount) {
            ++plannedPosition.expertIdx;
            plannedPosition.tokenIndexInExpert = 0U;
            continue;
        }

        ExpertTokenRange range{plannedPosition, plannedPosition};
        AdvanceExpertTokenPositionInWave<TileM>(expertTokenCount, waveMGroupTarget, waveMGroupCount, plannedPosition);
        range.end = plannedPosition;
        return range;
    }
    return {plannedPosition, plannedPosition};
}

// 出口跨卡握手：物理 AIV 分别推进自己的 sequence，入口已在 MoE 量化后推进一次。
__aicore__ inline int32_t CrossRankHandshakeInWorldSize(GM_ADDR rankSyncInWorldPtr, uint32_t rankId, uint32_t worldSize,
                                                        const AivJobContext& syncJob, __gm__ int32_t* syncCount)
{
    auto* syncRank = reinterpret_cast<__gm__ int32_t*>(rankSyncInWorldPtr);
    const int32_t count = static_cast<int32_t>(static_cast<uint32_t>(ReadGmByPassDCache(syncCount)) + 1U);
    for (uint32_t rankIdx = syncJob.jobIndex; rankIdx < worldSize; rankIdx += syncJob.totalJobs) {
        auto* remoteSyncAddr = reinterpret_cast<__gm__ int32_t*>(g_winRankAddr_[rankIdx]) + rankId * INT_CACHELINE;
        WriteGmByPassDCache(remoteSyncAddr, count);
        GmSignalWaitBarrier(syncRank + rankIdx * INT_CACHELINE, count);
    }
    WriteGmByPassDCache(syncCount, count);
    return count;
}

// Caller selects participating AIV cores; this entry performs full-AIV synchronization.
__aicore__ inline void CrossRankSyncInWorldSize(GM_ADDR rankSyncInWorldPtr, uint32_t rankId, uint32_t worldSize,
                                                const AivJobContext& aivJob)
{
    auto* syncCount = GetSyncCountAddress(rankSyncInWorldPtr, aivJob.jobIndex);
    CrossRankHandshakeInWorldSize(rankSyncInWorldPtr, rankId, worldSize, aivJob, syncCount);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

template <AscendC::HardEvent event, int32_t eventId>
__aicore__ inline void SyncFuncStatic()
{
    AscendC::SetFlag<event>(static_cast<event_t>(eventId));
    AscendC::WaitFlag<event>(static_cast<event_t>(eventId));
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_UTILS_H
