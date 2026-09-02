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
 * \file moe_distribute_combine_a2.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_A2_H
#define MOE_DISTRIBUTE_COMBINE_A2_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "utils/std/algorithm.h"
#include "../moe_distribute_combine_tiling.h"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_a2_adump.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/arch22/moe_distribute_a2_constant.h"

namespace MoeDistributeCombineA2Impl {
// 缓冲区相关常量
constexpr uint8_t BUFFER_NUM = 2; // 多缓冲区数量
constexpr uint32_t PING_IDX = 0;  // Ping缓冲区索引
constexpr uint32_t PONG_IDX = 1;  // Pong缓冲区索引

// 内存对齐相关常量
constexpr uint32_t BLOCK_SIZE = 32;                               // Block字节数
constexpr uint32_t B32_PER_BLOCK = BLOCK_SIZE / sizeof(uint32_t); // 每个Block的32位整数数量
constexpr uint32_t B64_PER_BLOCK = BLOCK_SIZE / sizeof(uint64_t); // 每个Block的64位整数数量
constexpr uint32_t REPEAT_BYTES = 256;                            // REPEAT字节数
constexpr uint64_t MB_SIZE = 1024 * 1024;                         // MB大小

// batchWrite接口相关常量
constexpr uint32_t BATCH_WRITE_ITEM_OFFSET = 8 * 1024; // batchWriteInfo结构体地址相对于windowOut最后1M的偏移
constexpr uint32_t BATCH_WRITE_ITEM_SIZE = 32;         // BatchWriteItem结构体大小
constexpr uint32_t U64_PER_ITEM = BATCH_WRITE_ITEM_SIZE / sizeof(uint64_t); // BatchWriteItem结构体占64位整数数量
constexpr uint32_t U32_PER_ITEM = BATCH_WRITE_ITEM_SIZE / sizeof(uint32_t); // BatchWriteItem结构体占32位整数数量

// 其他常量
constexpr uint32_t TOKEN_OFFSET = 32;          // 标志位与开头TokenNum之间的偏移量
constexpr uint32_t SKIP_OFFSET = 32;           // 数据区与收发标志位之间的跳过字节数
constexpr uint32_t FLAG_VALUE = 0xFFFFFFFF;    // 标志值
constexpr uint32_t A2_RANK_NUM_PER_SERVER = 8; // A2单节点Rank数量
constexpr uint32_t MAX_NUM_EXPERTS_PER_TILE =
    16384; // 统计从各专家发回的token数量时，每轮最多处理的专家索引的数量，超过该数量时需要分多轮统计
constexpr uint32_t MAX_ROUTED_TOKENS_PER_TILE =
    512; // WindowCopy时每轮最多处理的bs * topk的数量，超过该数量时需要分多轮进行WindowCopy

using namespace AscendC;

template <typename T>
inline __aicore__ T RoundUp(const T val, const T align)
{
    return (val + align - 1) / align * align;
}

struct TaskInfo {
    uint32_t startTaskId{0};
    uint32_t endTaskId{0};
    uint32_t taskNum{0};

    __aicore__ inline void SplitCore(uint32_t taskNumTotal, uint32_t aivNum, uint32_t aivId)
    {
        const uint32_t baseNum = taskNumTotal / aivNum;
        const uint32_t remainder = taskNumTotal % aivNum;

        const bool hasExtraTask = aivId < remainder;
        taskNum = baseNum + static_cast<uint32_t>(hasExtraTask);
        startTaskId = baseNum * aivId + (hasExtraTask ? aivId : remainder);
        endTaskId = startTaskId + taskNum;
    }
};

// 直方图统计，统计src中小于max的每个值的数量，如果isSetMask为true，则只统计mask中为true的位置
template <bool isSetMask = false, typename T, typename U>
__aicore__ inline void Histograms(const LocalTensor<T> &dst, const LocalTensor<U> &src, const LocalTensor<bool> &mask,
                                  U max, int32_t count)
{
#pragma unroll 8
    for (uint32_t i = 0; i < count; ++i) {
        if constexpr (isSetMask) {
            if (!mask(i)) {
                continue;
            }
        }
        U value = src(i);
        if (value < max) {
            dst(value) += 1;
        }
    }
}

template <typename T>
__aicore__ inline void CopyGm2Ub(const LocalTensor<T> &dstTensor, const GlobalTensor<T> &srcTensor, uint32_t count)
{
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>((count) * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
    DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyUb2Gm(const GlobalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, uint32_t count)
{
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>((count) * sizeof(T)), 0, 0, 0};
    DataCopyPad(dstTensor, srcTensor, dataCopyParams);
}

#define TemplateMC2TypeA2Class typename ExpandXType, typename ExpandIdxType
#define TemplateMC2TypeA2Func ExpandXType, ExpandIdxType
template <TemplateMC2TypeA2Class>
class MoeDistributeCombineA2 {
public:
    __aicore__ inline MoeDistributeCombineA2(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount,
                                GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void AllocTensorForComm();
    __aicore__ inline void AllocTensorForWindowCopy();
    __aicore__ inline void CalRecvCountsAndOffsets();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void CalTokenActiveMask();
    __aicore__ inline void ProcessMoeAndCopyExpert(int32_t eventId, uint32_t tokenIdx, uint32_t expertOffset);
    __aicore__ inline void SingleServerDispatch(LocalTensor<ExpandIdxType> &sendCountInfo);
    __aicore__ inline void MultiServerDispatch(LocalTensor<ExpandIdxType> &sendCountInfo);
    __aicore__ inline uint32_t GetRankTokenNumAndDataCopy2WindowOut(LocalTensor<ExpandIdxType> &sendCountInfo,
                                                                    GlobalTensor<ExpandXType> &rankWindowOut,
                                                                    uint32_t rankId);
    __aicore__ inline void ConstructBatchWriteInfo(LocalTensor<ExpandIdxType> &sendCountInfo);
    __aicore__ inline void CopyStatusToADump();

    GlobalTensor<ExpandXType> expandXGlobal_;
    GlobalTensor<ExpandIdxType> expertIdsGlobal_;
    GlobalTensor<ExpandIdxType> expandIdxGlobal_;
    GlobalTensor<ExpandIdxType> sendCountGlobal_;
    GlobalTensor<float> topkWeightsGlobal_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    GlobalTensor<ExpandXType> rankWindow_; // 用于存对端window的变量
    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXType> localInWindow_;
    GlobalTensor<uint32_t> bufferIdGlobal_;  // win区状态位置拷入相关参数
    GlobalTensor<uint64_t> workspaceGlobal_; // 存储batchWriteInfo结构体信息
    GlobalTensor<uint32_t> flagGlobal_;
    GlobalTensor<bool> xActiveMaskGlobal_;
    GlobalTensor<ExpandXType> oriXGlobal_; // Dispatch时输入的原始token数据，用于copyExpert或constExpert的计算
    GlobalTensor<int32_t> performanceInfoI32Global_;
    GlobalTensor<uint32_t> expertRecvCountGlobal_;
    GlobalTensor<uint32_t> expertWindowOffsetGlobal_;

    LocalTensor<ExpandXType> xLocal_[BUFFER_NUM];
    LocalTensor<uint64_t> batchWriteU64Local_;
    LocalTensor<uint32_t> batchWriteU32Local_;
    LocalTensor<uint32_t> flagLocal_;
    LocalTensor<int32_t> sendCountLocal_;
    LocalTensor<uint32_t> recvCountLocal_;
    LocalTensor<uint32_t> expertWindowOffsetLocal_;
    LocalTensor<float> topkSumFloatLocal_;
    LocalTensor<ExpandXType> topkSumLocal_;
    LocalTensor<float> tokenFloatLocal_;
    LocalTensor<ExpandIdxType> expertIdsLocal_;
    LocalTensor<float> topkWeightsLocal_;
    LocalTensor<ExpandIdxType> expandIdxLocal_;
    LocalTensor<bool> expertMaskLocal_;
    LocalTensor<int32_t> performanceInfoI32Local_;
    LocalTensor<uint32_t> numRecvTokensPerRankLocal_;
    LocalTensor<uint8_t> aDumpTensor_;
    LocalTensor<bool> isVisitedLocal_;

    GM_ADDR adumpStatusGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR oriXGM_;
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t coreIdx_{0};
    uint32_t moeExpertNum_{0};
    uint32_t localMoeExpertNum_{0}; // 每张卡的专家数
    uint32_t zeroExpertNum_{0};
    uint32_t copyExpertNum_{0};
    uint32_t constExpertNum_{0};
    uint64_t rankSizeOnWin_{0};
    uint64_t dataOffsetOnWin_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t halfWinSize_{0};
    uint32_t dataSpaceSize_{0};
    uint32_t bufferId_{0};

    bool isInputTokenMaskFlag_{false};
    bool isInputExpertMaskFlag_{false};
    bool needPerformanceInfo_{false};
    bool isSingleServer_{false};

    TaskInfo tokenTaskInfo_;
    TaskInfo worldSizeTaskInfo_;

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
    Mc2A2Kernel::MoeDistributeA2ADump aDump_;
};

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR sendCount, GM_ADDR scales, GM_ADDR xActiveMask,
    GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR performanceInfo,
    GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeCombineA2TilingData *tilingData)
{
    oriXGM_ = oriX;
    rankId_ = tilingData->moeDistributeCombineInfo.epRankId;
    axisBS_ = tilingData->moeDistributeCombineInfo.bs;
    axisH_ = tilingData->moeDistributeCombineInfo.h;
    axisK_ = tilingData->moeDistributeCombineInfo.k;
    aivNum_ = tilingData->moeDistributeCombineInfo.aivNum;
    moeExpertNum_ = tilingData->moeDistributeCombineInfo.moeExpertNum;
    zeroExpertNum_ = tilingData->moeDistributeCombineInfo.zeroExpertNum;
    copyExpertNum_ = tilingData->moeDistributeCombineInfo.copyExpertNum;
    constExpertNum_ = tilingData->moeDistributeCombineInfo.constExpertNum;
    worldSize_ = tilingData->moeDistributeCombineInfo.epWorldSize;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineInfo.isTokenMask;
    isInputExpertMaskFlag_ = tilingData->moeDistributeCombineInfo.isExpertMask;
    isSingleServer_ = worldSize_ <= A2_RANK_NUM_PER_SERVER;
    auto contextGM = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)contextGM;
    hccl_.InitV2(contextGM, tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeCombineA2TilingData, mc2CcTiling));
    halfWinSize_ = winContext_->winSize / 2;
    dataSpaceSize_ = halfWinSize_ - Mc2A2Kernel::ADUMP_WIN_SIZE;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    bufferIdGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + Mc2A2Kernel::FULLMESH_BUFFERID_ADDR));
    bufferId_ = bufferIdGlobal_.GetValue(0);
    adumpStatusGM_ = windowInGM_ + Mc2A2Kernel::ADUMP_STATUS_COMBINE_ADDR;
    windowInGM_ = windowInGM_ + Mc2A2Kernel::ADUMP_WIN_SIZE + halfWinSize_ * bufferId_;
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_) + Mc2A2Kernel::ADUMP_WIN_SIZE + halfWinSize_ * bufferId_;
    coreIdx_ = GetBlockIdx();
    expandXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGlobal_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    sendCountGlobal_.SetGlobalBuffer((__gm__ int32_t *)sendCount);
    topkWeightsGlobal_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    workspaceGlobal_.SetGlobalBuffer((__gm__ uint64_t *)(hccl_.GetWindowsOutAddr(rankId_) + halfWinSize_ * bufferId_ +
                                                         Mc2A2Kernel::BATCH_WRITE_ITEM_OFFSET));
    expertRecvCountGlobal_.SetGlobalBuffer((__gm__ uint32_t *)workspaceGM);
    expertWindowOffsetGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(workspaceGM + moeExpertNum_ * sizeof(uint32_t)));
    performanceInfoI32Global_.SetGlobalBuffer((__gm__ int32_t *)performanceInfo);
    xActiveMaskGlobal_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    oriXGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)oriX);
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    rankSizeOnWin_ = dataSpaceSize_ / worldSize_ / BLOCK_SIZE * BLOCK_SIZE;
    dataOffsetOnWin_ = rankId_ * rankSizeOnWin_;
    axisHExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    needPerformanceInfo_ = performanceInfo != nullptr;
    worldSizeTaskInfo_.SplitCore(worldSize_, aivNum_, coreIdx_);
    uint32_t aDumpSize =
        Std::max(RoundUp(Mc2A2Kernel::H_POS + 1U, B32_PER_BLOCK), B32_PER_BLOCK * 2U) * sizeof(uint32_t);
    uint32_t aDumpUbAddr = AscendC::TOTAL_UB_SIZE - aDumpSize;
    aDumpTensor_ = LocalTensor<uint8_t>{TPosition::LCM, aDumpUbAddr, aDumpSize};
    GM_ADDR aDumpAddr = hccl_.GetWindowsInAddr(rankId_) + Mc2A2Kernel::ADUMP_DATA_COMBINE_ADDR_START;
    aDump_.Init(aDumpAddr, coreIdx_, rankId_, worldSize_, rankId_, moeExpertNum_, worldSize_,
                tilingData->moeDistributeCombineInfo.globalBs, bufferId_, Mc2A2Kernel::IS_FULLMESH, aivNum_,
                aDumpTensor_, axisH_, axisBS_);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::AllocTensorForComm()
{
    xLocal_[PING_IDX] = LocalTensor<ExpandXType>{TPosition::LCM, 0, axisH_};
    xLocal_[PONG_IDX] = LocalTensor<ExpandXType>{TPosition::LCM, axisHExpandXTypeSize_, axisH_};
    uint32_t batchWriteLocalAddr = axisHExpandXTypeSize_ * BUFFER_NUM;
    uint32_t batchWriteU64LocalEleNum = U64_PER_ITEM * Ceil(worldSize_, aivNum_);
    uint32_t batchWriteU32LocalEleNum = U64_PER_ITEM * Ceil(worldSize_, aivNum_);
    batchWriteU64Local_ = LocalTensor<uint64_t>{TPosition::LCM, batchWriteLocalAddr, batchWriteU64LocalEleNum};
    batchWriteU32Local_ = LocalTensor<uint32_t>{TPosition::LCM, batchWriteLocalAddr, batchWriteU32LocalEleNum};
    uint32_t flagLocalAddr = batchWriteLocalAddr + batchWriteU64LocalEleNum * sizeof(uint64_t);
    flagLocal_ = LocalTensor<uint32_t>{TPosition::LCM, flagLocalAddr, B32_PER_BLOCK};
    uint32_t sendCountLocalAddr = flagLocalAddr + BLOCK_SIZE;
    uint32_t sendCountLocalEleNum = RoundUp(moeExpertNum_, B32_PER_BLOCK);
    sendCountLocal_ = LocalTensor<ExpandIdxType>{TPosition::LCM, sendCountLocalAddr, sendCountLocalEleNum};

    uint32_t recvCountLocalAddr = sendCountLocalAddr + sendCountLocalEleNum * sizeof(ExpandIdxType);
    uint32_t moeExpertNumAlign = RoundUp(moeExpertNum_, B32_PER_BLOCK);
    recvCountLocal_ = LocalTensor<uint32_t>{TPosition::LCM, recvCountLocalAddr, moeExpertNumAlign};
    uint32_t expertWindowOffsetLocalAddr = recvCountLocalAddr + moeExpertNumAlign * sizeof(uint32_t);
    expertWindowOffsetLocal_ = LocalTensor<uint32_t>{TPosition::LCM, expertWindowOffsetLocalAddr, moeExpertNumAlign};

    uint32_t worldSizeAlign = RoundUp(worldSize_, B32_PER_BLOCK);
    uint32_t numRecvTokensPerRankAddr = expertWindowOffsetLocalAddr + moeExpertNumAlign * sizeof(uint32_t);
    numRecvTokensPerRankLocal_ = LocalTensor<uint32_t>{TPosition::LCM, numRecvTokensPerRankAddr, worldSizeAlign};

    uint32_t expertMaskLocalAddr = numRecvTokensPerRankAddr + worldSizeAlign * sizeof(uint32_t);
    expertMaskLocal_ = LocalTensor<bool>{TPosition::LCM, expertMaskLocalAddr, MAX_NUM_EXPERTS_PER_TILE};

    uint32_t expertIdsLocalAddr = expertMaskLocalAddr + MAX_NUM_EXPERTS_PER_TILE * sizeof(bool);
    expertIdsLocal_ = LocalTensor<int32_t>{TPosition::LCM, expertIdsLocalAddr, MAX_NUM_EXPERTS_PER_TILE};

    uint32_t performanceInfoAddr = expertIdsLocalAddr + MAX_NUM_EXPERTS_PER_TILE * sizeof(int32_t);
    uint32_t performanceInfoEleCount = RoundUp(worldSize_, B64_PER_BLOCK) * sizeof(int64_t) / sizeof(int32_t);
    performanceInfoI32Local_ = LocalTensor<int32_t>{TPosition::LCM, performanceInfoAddr, performanceInfoEleCount};

    uint32_t isVisitedAddr = performanceInfoAddr + performanceInfoEleCount * sizeof(int32_t);
    uint32_t isVisitedCount = RoundUp(worldSizeTaskInfo_.taskNum, BLOCK_SIZE);
    isVisitedLocal_ = LocalTensor<bool>{TPosition::LCM, isVisitedAddr, isVisitedCount};
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::CalTokenActiveMask()
{
    if (!isInputTokenMaskFlag_) {
        return;
    }
    uint32_t xActiveMaskAlignSize = RoundUp(axisBS_, BLOCK_SIZE);
    auto xActiveMaskLocal = LocalTensor<bool>{TPosition::LCM, 0, xActiveMaskAlignSize};
    auto xActiveMaskHalfLocal = LocalTensor<half>{TPosition::LCM, xActiveMaskAlignSize, xActiveMaskAlignSize};
    auto sharedTmpBuffer = xActiveMaskLocal.ReinterpretCast<half>();
    auto xActiveMaskInt8Local = xActiveMaskLocal.ReinterpretCast<int8_t>();
    CopyGm2Ub(xActiveMaskLocal, xActiveMaskGlobal_, axisBS_);
    SyncFunc<HardEvent::MTE2_V>();
    Cast(xActiveMaskHalfLocal, xActiveMaskInt8Local, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    // sharedTmpBuffer仅用于占位，结果通过GetAccVal获取
    ReduceSum(sharedTmpBuffer, xActiveMaskHalfLocal, sharedTmpBuffer, axisBS_);
    axisBS_ = static_cast<int32_t>(AscendC::GetAccVal<half>());
}

template <TemplateMC2TypeA2Class>
__aicore__ inline uint32_t MoeDistributeCombineA2<TemplateMC2TypeA2Func>::GetRankTokenNumAndDataCopy2WindowOut(
    LocalTensor<ExpandIdxType> &sendCountInfo, GlobalTensor<ExpandXType> &rankWindowOut, uint32_t rankId)
{
    uint32_t rankTokenNum = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    int32_t eventId = 0;
    for (uint32_t expertId = 0; expertId < localMoeExpertNum_; ++expertId) {
        uint32_t preCount = 0;
        if (expertId != 0 || rankId != 0) {
            preCount = static_cast<uint32_t>(sendCountInfo(expertId * worldSize_ + rankId - 1));
        }
        uint32_t startTokenIdx = preCount * axisH_;
        uint32_t tokenNum = sendCountInfo(expertId * worldSize_ + rankId) - preCount;
        for (uint32_t tokenId = 0; tokenId < tokenNum; ++tokenId) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(xLocal_[eventId], expandXGlobal_[startTokenIdx], axisH_);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(rankWindowOut[rankTokenNum * axisH_], xLocal_[eventId], axisH_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
            eventId ^= 1;
            startTokenIdx += axisH_;
            rankTokenNum++;
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    return rankTokenNum;
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::SingleServerDispatch(
    LocalTensor<ExpandIdxType> &sendCountInfo)
{
    int32_t eventId = 0;
    GlobalTensor<uint32_t> tokenNumInWindow;
    LocalTensor<uint32_t> tokenNumTensor = aDumpTensor_.ReinterpretCast<uint32_t>();
    for (uint32_t dstRankId = worldSizeTaskInfo_.startTaskId; dstRankId < worldSizeTaskInfo_.endTaskId; ++dstRankId) {
        localOutWindow_.SetGlobalBuffer(
            (__gm__ ExpandXType *)(windowOutGM_ + dstRankId * rankSizeOnWin_ + TOKEN_OFFSET));
        tokenNumInWindow.SetGlobalBuffer((__gm__ uint32_t *)(hccl_.GetWindowsInAddr(dstRankId) +
                                                             Mc2A2Kernel::ADUMP_WIN_SIZE + halfWinSize_ * bufferId_ +
                                                             dataOffsetOnWin_));

        uint32_t rankTokenNum = GetRankTokenNumAndDataCopy2WindowOut(sendCountInfo, localOutWindow_, dstRankId);
        SyncFunc<HardEvent::MTE3_S>();
        tokenNumTensor(B32_PER_BLOCK) = rankTokenNum;
        SyncFunc<HardEvent::S_MTE3>();
        CopyUb2Gm(tokenNumInWindow, tokenNumTensor[B32_PER_BLOCK], 1);
        GlobalTensor<ExpandXType> dstGlobal;
        dstGlobal.SetGlobalBuffer((__gm__ ExpandXType *)(hccl_.GetWindowsInAddr(dstRankId) +
                                                         Mc2A2Kernel::ADUMP_WIN_SIZE + halfWinSize_ * bufferId_ +
                                                         dataOffsetOnWin_ + TOKEN_OFFSET));
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t tokenId = 0; tokenId < rankTokenNum; ++tokenId) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(xLocal_[eventId], localOutWindow_[tokenId * axisH_], axisH_);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(dstGlobal[tokenId * axisH_], xLocal_[eventId], axisH_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
            eventId ^= 1;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        PipeBarrier<PIPE_MTE3>();
        flagLocal_(0) = FLAG_VALUE;
        flagGlobal_.SetGlobalBuffer(
            (__gm__ uint32_t *)dstGlobal.GetPhyAddr(rankTokenNum * axisH_ + SKIP_OFFSET / sizeof(ExpandXType)));
        CopyUb2Gm(flagGlobal_, flagLocal_, 1);
        aDump_.UpdateFullmeshEpRankSendOrRecv(dstRankId, Mc2A2Kernel::IS_SEND);
        aDump_.UpdateFullmeshTokenSendOrRecv(dstRankId, rankTokenNum, Mc2A2Kernel::IS_SEND);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::ConstructBatchWriteInfo(
    LocalTensor<ExpandIdxType> &sendCountInfo)
{
    GlobalTensor<uint32_t> tokenNumOutWindow;
    LocalTensor<uint32_t> tokenNumTensor = aDumpTensor_.ReinterpretCast<uint32_t>();
    for (uint32_t dstRankId = worldSizeTaskInfo_.startTaskId; dstRankId < worldSizeTaskInfo_.endTaskId; ++dstRankId) {
        localOutWindow_.SetGlobalBuffer(
            (__gm__ ExpandXType *)(windowOutGM_ + dstRankId * rankSizeOnWin_ + TOKEN_OFFSET));
        tokenNumOutWindow.SetGlobalBuffer((__gm__ uint32_t *)(windowOutGM_ + dstRankId * rankSizeOnWin_));
        uint32_t rankTokenNum = GetRankTokenNumAndDataCopy2WindowOut(sendCountInfo, localOutWindow_, dstRankId);
        SyncFunc<HardEvent::MTE3_S>();
        tokenNumTensor(B32_PER_BLOCK) = rankTokenNum;
        SyncFunc<HardEvent::S_MTE3>();
        CopyUb2Gm(tokenNumOutWindow, tokenNumTensor[B32_PER_BLOCK], 1);
        flagGlobal_.SetGlobalBuffer(
            (__gm__ uint32_t *)(localOutWindow_.GetPhyAddr(rankTokenNum * axisH_) + SKIP_OFFSET / sizeof(ExpandXType)));
        flagGlobal_(0) = FLAG_VALUE;
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            flagGlobal_);

        uint32_t rankIdOffset = dstRankId - worldSizeTaskInfo_.startTaskId;
        batchWriteU64Local_(rankIdOffset * U64_PER_ITEM) = (uint64_t)(tokenNumOutWindow.GetPhyAddr());
        batchWriteU64Local_(rankIdOffset * U64_PER_ITEM + 1) =
            (uint64_t)(hccl_.GetWindowsInAddr(dstRankId) + Mc2A2Kernel::ADUMP_WIN_SIZE + halfWinSize_ * bufferId_ +
                       dataOffsetOnWin_);
        batchWriteU64Local_(rankIdOffset * U64_PER_ITEM + 2) =
            rankTokenNum * axisH_ + (SKIP_OFFSET + TOKEN_OFFSET) / sizeof(ExpandXType) + 2;
        batchWriteU32Local_(rankIdOffset * U32_PER_ITEM + 6) = HcclDataType::HCCL_DATA_TYPE_FP16;
        batchWriteU32Local_(rankIdOffset * U32_PER_ITEM + 7) = dstRankId;
        aDump_.UpdateFullmeshEpRankSendOrRecv(dstRankId, Mc2A2Kernel::IS_SEND);
        aDump_.UpdateFullmeshTokenSendOrRecv(dstRankId, rankTokenNum, Mc2A2Kernel::IS_SEND);
    }
    SyncFunc<HardEvent::S_MTE3>();
    DataCopy(workspaceGlobal_[worldSizeTaskInfo_.startTaskId * U64_PER_ITEM], batchWriteU64Local_,
             worldSizeTaskInfo_.taskNum * U64_PER_ITEM);
    SyncFunc<HardEvent::MTE3_S>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::MultiServerDispatch(
    LocalTensor<ExpandIdxType> &sendCountInfo)
{
    if (coreIdx_ == 0) {
        HcclHandle handleId = hccl_.BatchWrite<true>((GM_ADDR)(workspaceGlobal_.GetPhyAddr()), worldSize_);
        bufferIdGlobal_(0) = bufferId_ ^ 1;
    }
    if (rankId_ >= worldSizeTaskInfo_.startTaskId && rankId_ < worldSizeTaskInfo_.endTaskId) {
        localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowOutGM_ + dataOffsetOnWin_ + TOKEN_OFFSET));
        localInWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(windowInGM_ + dataOffsetOnWin_ + TOKEN_OFFSET));
        GlobalTensor<uint32_t> tokenNumInWindow;
        LocalTensor<uint32_t> tokenNumTensor = aDumpTensor_.ReinterpretCast<uint32_t>();
        tokenNumInWindow.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + dataOffsetOnWin_));
        uint32_t rankIdOffset = rankId_ - worldSizeTaskInfo_.startTaskId;
        uint64_t rankTokenNum =
            (batchWriteU64Local_(rankIdOffset * 4 + 2) - (SKIP_OFFSET + TOKEN_OFFSET) / sizeof(ExpandXType) - 2) /
            axisH_;
        SyncFunc<HardEvent::MTE3_S>();
        tokenNumTensor(B32_PER_BLOCK) = rankTokenNum;
        SyncFunc<HardEvent::S_MTE3>();
        CopyUb2Gm(tokenNumInWindow, tokenNumTensor[B32_PER_BLOCK], 1);
        int32_t eventId = 0;
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t tokenId = 0; tokenId < rankTokenNum; ++tokenId) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(xLocal_[eventId], localOutWindow_[tokenId * axisH_], axisH_);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(localInWindow_[tokenId * axisH_], xLocal_[eventId], axisH_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
            eventId ^= 1;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        flagGlobal_.SetGlobalBuffer(
            (__gm__ uint32_t *)localInWindow_.GetPhyAddr(rankTokenNum * axisH_ + SKIP_OFFSET / sizeof(ExpandXType)));
        flagGlobal_(0) = FLAG_VALUE;
        DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            flagGlobal_);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::AlltoAllDispatch()
{
    aDump_.UpdateFullmeshRankTaskInfo(worldSizeTaskInfo_.startTaskId, worldSizeTaskInfo_.taskNum, Mc2A2Kernel::IS_SEND);
    if (worldSizeTaskInfo_.taskNum == 0) {
        SyncAll<true>();
        return;
    }
    DataCopy(sendCountLocal_, sendCountGlobal_, RoundUp(moeExpertNum_, B32_PER_BLOCK));
    SyncFunc<HardEvent::MTE2_S>();

    if (isSingleServer_) {
        SingleServerDispatch(sendCountLocal_);
        SyncAll<true>();
        if (coreIdx_ == 0) {
            bufferIdGlobal_(0) = bufferId_ ^ 1;
        }
    } else {
        ConstructBatchWriteInfo(sendCountLocal_);
        SyncAll<true>();
        MultiServerDispatch(sendCountLocal_);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::CalRecvCountsAndOffsets()
{
    TaskInfo taskInfo;
    taskInfo.SplitCore(axisBS_ * axisK_, aivNum_, coreIdx_);
    uint32_t expertTileNum = CeilDiv(taskInfo.taskNum, MAX_NUM_EXPERTS_PER_TILE);
    uint32_t numExpertsPerTail = taskInfo.taskNum - (expertTileNum - 1) * MAX_NUM_EXPERTS_PER_TILE;

    Duplicate(recvCountLocal_, 0u, moeExpertNum_);
    Duplicate(expertWindowOffsetLocal_, 0u, moeExpertNum_);

    SyncFunc<HardEvent::V_MTE3>();
    if (coreIdx_ == aivNum_ - 1) {
        CopyUb2Gm(expertRecvCountGlobal_, recvCountLocal_, moeExpertNum_);
    }
    SyncAll<true>();
    for (uint32_t i = 0u; i < expertTileNum; i++) {
        uint32_t tileEleCount = (i == expertTileNum - 1) ? numExpertsPerTail : MAX_NUM_EXPERTS_PER_TILE;
        uint64_t EleOffset = taskInfo.startTaskId + i * MAX_NUM_EXPERTS_PER_TILE;
        CopyGm2Ub(expertIdsLocal_, expertIdsGlobal_[EleOffset], tileEleCount);
        if (isInputExpertMaskFlag_) {
            CopyGm2Ub(expertMaskLocal_, xActiveMaskGlobal_[EleOffset], tileEleCount);
        }
        SyncFunc<HardEvent::MTE2_S>();
        if (isInputExpertMaskFlag_) {
            Histograms<true>(recvCountLocal_, expertIdsLocal_, expertMaskLocal_, (int32_t)moeExpertNum_, tileEleCount);
        } else {
            Histograms<false>(recvCountLocal_, expertIdsLocal_, expertMaskLocal_, (int32_t)moeExpertNum_, tileEleCount);
        }
        SyncFunc<HardEvent::S_MTE2>();
    }
    SyncFunc<HardEvent::S_MTE3>();

    SetAtomicAdd<int32_t>();
    CopyUb2Gm(expertRecvCountGlobal_, recvCountLocal_, moeExpertNum_);
    SetAtomicNone();
    SyncAll<true>();

    CopyGm2Ub(recvCountLocal_, expertRecvCountGlobal_, moeExpertNum_);
    SyncFunc<HardEvent::MTE2_S>();
    for (uint32_t i = 0u; i < worldSizeTaskInfo_.taskNum; ++i) {
        uint32_t prefixSum = 0;
        uint32_t expertBaseOffset = i * localMoeExpertNum_;
        uint32_t recvCountBaseOffset = (worldSizeTaskInfo_.startTaskId + i) * localMoeExpertNum_;
        for (uint32_t j = 0u; j < localMoeExpertNum_; ++j) {
            expertWindowOffsetLocal_(expertBaseOffset + j) = prefixSum;
            prefixSum += recvCountLocal_(recvCountBaseOffset + j);
        }
        numRecvTokensPerRankLocal_(i) = prefixSum;
    }
    SyncFunc<HardEvent::S_MTE3>();
    CopyUb2Gm(expertWindowOffsetGlobal_[worldSizeTaskInfo_.startTaskId * localMoeExpertNum_], expertWindowOffsetLocal_,
              worldSizeTaskInfo_.taskNum * localMoeExpertNum_);
    SyncAll<true>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::CopyStatusToADump()
{
    // AdumpFlag, realTokenNum, recvTokenNum, realFlag, recvFlag
    uint32_t countPerRank = 5U;
    LocalTensor<int32_t> aDumpTensor = LocalTensor<int32_t>{TPosition::LCM, 0U, B32_PER_BLOCK * countPerRank};
    aDumpTensor.SetValue(0, Mc2A2Kernel::DUMP_FLAG_VALUE);
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    DataCopyExtParams copyDumpParams{static_cast<uint16_t>(countPerRank), static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
                                     0};
    DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    GlobalTensor<int32_t> aDumpStatusGM;
    aDumpStatusGM.SetGlobalBuffer((__gm__ int32_t *)(adumpStatusGM_));
    GlobalTensor<int32_t> tokenGlobal;
    GlobalTensor<int32_t> flagGlobal;
    for (uint32_t rankId = worldSizeTaskInfo_.startTaskId; rankId < worldSizeTaskInfo_.endTaskId; rankId++) {
        if (isVisitedLocal_(rankId - worldSizeTaskInfo_.startTaskId)) {
            continue;
        }
        uint32_t realTokenNum = numRecvTokensPerRankLocal_(rankId - worldSizeTaskInfo_.startTaskId);
        aDumpTensor.SetValue(B32_PER_BLOCK, realTokenNum);
        GM_ADDR wAddr = windowInGM_ + rankSizeOnWin_ * rankId;
        tokenGlobal.SetGlobalBuffer((__gm__ int32_t *)wAddr);
        DataCopyPad(aDumpTensor[B32_PER_BLOCK * 2], tokenGlobal, copyFlagParams, padParams); // tokenNum
        wAddr =
            windowInGM_ + rankSizeOnWin_ * rankId + TOKEN_OFFSET + SKIP_OFFSET + realTokenNum * axisHExpandXTypeSize_;
        flagGlobal.SetGlobalBuffer((__gm__ int32_t *)wAddr);
        DataCopyPad(aDumpTensor[B32_PER_BLOCK * 3], flagGlobal, copyFlagParams, padParams); // target token flag
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        int32_t recvTokenNum = aDumpTensor(B32_PER_BLOCK * 2);
        if (recvTokenNum <= axisBS_ * axisK_) {
            wAddr = windowInGM_ + rankSizeOnWin_ * rankId + TOKEN_OFFSET + SKIP_OFFSET +
                    recvTokenNum * axisHExpandXTypeSize_;
            flagGlobal.SetGlobalBuffer((__gm__ int32_t *)wAddr);
            DataCopyPad(aDumpTensor[B32_PER_BLOCK * 4], flagGlobal, copyFlagParams, padParams);
        } else {
            aDumpTensor.SetValue(B32_PER_BLOCK * 4, 0U);
        }
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(aDumpStatusGM[rankId * countPerRank], aDumpTensor, copyDumpParams); // 设置AdumpFlag
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::WaitDispatch()
{
    aDump_.UpdateFullmeshRankTaskInfo(worldSizeTaskInfo_.startTaskId, worldSizeTaskInfo_.taskNum, Mc2A2Kernel::IS_RECV);
    if (unlikely(needPerformanceInfo_) && worldSizeTaskInfo_.taskNum > 0) {
        Duplicate<int32_t>(performanceInfoI32Local_, 0, worldSize_ * sizeof(int64_t) / sizeof(int32_t));
        SyncFunc<HardEvent::V_S>();
    }
    uint32_t waitFlagNum = 0;
    auto &&isVisitedU32 = isVisitedLocal_.template ReinterpretCast<uint32_t>();
    Duplicate<uint32_t>(isVisitedU32, uint32_t(0), isVisitedU32.GetSize());
    SyncFunc<AscendC::HardEvent::V_S>();

    bool unDump = true;
    int64_t startTime = GetCurrentTimestampUs();
    while (waitFlagNum < worldSizeTaskInfo_.taskNum) {
        for (uint32_t rankId = worldSizeTaskInfo_.startTaskId; rankId < worldSizeTaskInfo_.endTaskId; ++rankId) {
            GM_ADDR wAddr = windowInGM_ + rankSizeOnWin_ * rankId + SKIP_OFFSET + TOKEN_OFFSET +
                            numRecvTokensPerRankLocal_(rankId - worldSizeTaskInfo_.startTaskId) * axisHExpandXTypeSize_;
            flagGlobal_.SetGlobalBuffer((__gm__ uint32_t *)wAddr);
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(flagGlobal_);
            uint32_t flag = flagGlobal_(0);
            if (flag != FLAG_VALUE) {
                continue;
            }
            isVisitedLocal_(rankId - worldSizeTaskInfo_.startTaskId) = true;
            waitFlagNum++;
            flagGlobal_(0) = 0;
            // 重要：要下DCCI保证清零写进去，避免下一次判断时又判断生效，重复累计waitFlagNum
            DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(flagGlobal_);
            if (unlikely(needPerformanceInfo_)) {
                RecordRankCommDuration(performanceInfoI32Local_, rankId, startTime);
            }
            aDump_.UpdateFullmeshEpRankSendOrRecv(rankId, Mc2A2Kernel::IS_RECV);
            aDump_.UpdateFullmeshTokenSendOrRecv(
                rankId, numRecvTokensPerRankLocal_(rankId - worldSizeTaskInfo_.startTaskId), Mc2A2Kernel::IS_RECV);
        }
        if (unDump && (GetCurrentTimestampUs() - startTime) > Mc2A2Kernel::TIMEOUT) {
            CopyStatusToADump();
            unDump = false;
        }
    }
    if (unlikely(needPerformanceInfo_) && worldSizeTaskInfo_.taskNum > 0) {
        AscendC::SetAtomicAdd<int32_t>();
        CopyUb2Gm(performanceInfoI32Global_, performanceInfoI32Local_, worldSize_ * sizeof(int64_t) / sizeof(int32_t));
        AscendC::SetAtomicNone();
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::Process()
{
    if ASCEND_IS_AIV {
        AllocTensorForComm();
        CalTokenActiveMask();
        AlltoAllDispatch();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_FULLMESH_COMBINE_SEND);
        CalRecvCountsAndOffsets();
        WaitDispatch();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_FULLMESH_COMBINE_RECV);
        AllocTensorForWindowCopy();
        LocalWindowCopy();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_END);
        hccl_.Finalize();
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::AllocTensorForWindowCopy()
{
    xLocal_[PING_IDX] = LocalTensor<ExpandXType>{TPosition::LCM, 0, axisH_};
    xLocal_[PONG_IDX] = LocalTensor<ExpandXType>{TPosition::LCM, axisHExpandXTypeSize_, axisH_};
    uint32_t tokenFloatLocalAddr = axisHExpandXTypeSize_ * BUFFER_NUM;
    tokenFloatLocal_ = LocalTensor<float>{TPosition::LCM, tokenFloatLocalAddr, axisH_};
    uint32_t topkSumFloatLocalAddr = tokenFloatLocalAddr + axisH_ * sizeof(float);
    topkSumFloatLocal_ = LocalTensor<float>{TPosition::LCM, topkSumFloatLocalAddr, axisH_};
    uint32_t topkSumLocalAddr = topkSumFloatLocalAddr + axisH_ * sizeof(float);
    topkSumLocal_ = LocalTensor<ExpandXType>{TPosition::LCM, topkSumLocalAddr, axisH_};
    uint32_t expertWindowOffsetLocalAddr = topkSumLocalAddr + axisH_ * sizeof(float);
    uint32_t moeExpertNumAlign = RoundUp(moeExpertNum_, B32_PER_BLOCK);
    expertWindowOffsetLocal_ = LocalTensor<uint32_t>{TPosition::LCM, expertWindowOffsetLocalAddr, moeExpertNumAlign};
    uint32_t topkWeightsLocalAddr = expertWindowOffsetLocalAddr + moeExpertNumAlign * sizeof(uint32_t);
    topkWeightsLocal_ = LocalTensor<float>{TPosition::LCM, topkWeightsLocalAddr, MAX_ROUTED_TOKENS_PER_TILE};
    uint32_t expandIdxLocalAddr = topkWeightsLocalAddr + MAX_ROUTED_TOKENS_PER_TILE * sizeof(float);
    expandIdxLocal_ = LocalTensor<int32_t>{TPosition::LCM, expandIdxLocalAddr, MAX_ROUTED_TOKENS_PER_TILE};
    uint32_t expertIdsLocalAddr = expandIdxLocalAddr + MAX_ROUTED_TOKENS_PER_TILE * sizeof(int32_t);
    expertIdsLocal_ = LocalTensor<int32_t>{TPosition::LCM, expertIdsLocalAddr, MAX_ROUTED_TOKENS_PER_TILE};
    uint32_t expertMaskLocalAddr = expertIdsLocalAddr + MAX_ROUTED_TOKENS_PER_TILE * sizeof(int32_t);
    expertMaskLocal_ = LocalTensor<bool>{TPosition::LCM, expertMaskLocalAddr, MAX_ROUTED_TOKENS_PER_TILE};
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::LocalWindowCopy()
{
    tokenTaskInfo_.SplitCore(axisBS_, aivNum_, coreIdx_);
    if (tokenTaskInfo_.taskNum == 0) {
        return;
    }
    uint32_t maxTokenNumPerTile = MAX_ROUTED_TOKENS_PER_TILE / axisK_;
    uint32_t tileNum = CeilDiv(tokenTaskInfo_.taskNum, maxTokenNumPerTile);
    uint32_t lastTileTokenCount = tokenTaskInfo_.taskNum - (tileNum - 1) * maxTokenNumPerTile;
    CopyGm2Ub(expertWindowOffsetLocal_, expertWindowOffsetGlobal_, moeExpertNum_);

    for (uint32_t ti = 0; ti < tileNum; ++ti) {
        uint32_t tileTokenCount = (ti == tileNum - 1) ? lastTileTokenCount : maxTokenNumPerTile;
        uint32_t tileStart = tokenTaskInfo_.startTaskId + ti * maxTokenNumPerTile;
        uint32_t tileEleCount = tileTokenCount * axisK_;
        SyncFunc<HardEvent::S_MTE2>();
        CopyGm2Ub(topkWeightsLocal_, topkWeightsGlobal_[tileStart * axisK_], tileEleCount);
        CopyGm2Ub(expandIdxLocal_, expandIdxGlobal_[tileStart * axisK_], tileEleCount);
        CopyGm2Ub(expertIdsLocal_, expertIdsGlobal_[tileStart * axisK_], tileEleCount);
        if (isInputExpertMaskFlag_) {
            CopyGm2Ub(expertMaskLocal_, xActiveMaskGlobal_[tileStart * axisK_], tileEleCount);
        }
        SyncFunc<HardEvent::MTE2_S>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        int32_t eventId = 0;
        for (uint32_t j = 0; j < tileTokenCount; ++j) {
            uint32_t tokenIdx = tileStart + j;
            Duplicate(topkSumFloatLocal_, 0.0f, axisH_);
            for (uint32_t k = 0; k < axisK_; ++k) {
                uint32_t expertOffset = j * axisK_ + k;
                if (isInputExpertMaskFlag_ && !expertMaskLocal_(expertOffset)) {
                    continue;
                }
                int32_t expertId = expertIdsLocal_(expertOffset);
                const bool isMoeExpert = expertId < moeExpertNum_;
                const bool isCopyExpert = expertId >= moeExpertNum_ + zeroExpertNum_ &&
                                          expertId < moeExpertNum_ + zeroExpertNum_ + copyExpertNum_;
                if (isMoeExpert || isCopyExpert) {
                    ProcessMoeAndCopyExpert(eventId, tokenIdx, expertOffset);
                    eventId ^= 1;
                }
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            Cast(topkSumLocal_, topkSumFloatLocal_, AscendC::RoundMode::CAST_RINT, axisH_);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(expandOutGlobal_[tokenIdx * axisH_], topkSumLocal_, axisH_);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeCombineA2<TemplateMC2TypeA2Func>::ProcessMoeAndCopyExpert(int32_t eventId,
                                                                                              uint32_t tokenIdx,
                                                                                              uint32_t expertOffset)
{
    GM_ADDR wAddr;
    float topkWeight = topkWeightsLocal_(expertOffset);
    int32_t expertId = expertIdsLocal_(expertOffset);
    if (expertId < moeExpertNum_) {
        uint32_t rank = expertId / localMoeExpertNum_;
        wAddr = (__gm__ uint8_t *)(windowInGM_) + rankSizeOnWin_ * rank + TOKEN_OFFSET +
                (expertWindowOffsetLocal_(expertId) + expandIdxLocal_(expertOffset)) * axisHExpandXTypeSize_;
    } else {
        wAddr = (__gm__ uint8_t *)(oriXGM_) + tokenIdx * axisHExpandXTypeSize_;
    }
    rankWindow_.SetGlobalBuffer((__gm__ ExpandXType *)wAddr);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    DataCopy(xLocal_[eventId], rankWindow_, axisH_);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(tokenFloatLocal_, xLocal_[eventId], AscendC::RoundMode::CAST_NONE, axisH_);
    SetFlag<HardEvent::V_MTE2>(eventId);
    PipeBarrier<PIPE_V>();
    Muls(tokenFloatLocal_, tokenFloatLocal_, topkWeight, axisH_);
    PipeBarrier<PIPE_V>();
    Add(topkSumFloatLocal_, topkSumFloatLocal_, tokenFloatLocal_, axisH_);
}
} // namespace MoeDistributeCombineA2Impl
#endif // MOE_DISTRIBUTE_COMBINE_A2_H
