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
 * \file moe_distribute_dispatch_a2.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_A2_H
#define MOE_DISTRIBUTE_DISPATCH_A2_H
#include <climits>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "utils/std/algorithm.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_dispatch_tiling.h"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "moe_distribute_a2_adump.h"
#include "moe_distribute_a2_constant.h"

namespace MoeDistributeDispatchA2Impl {
constexpr static uint8_t BUFFER_NUM = 2;
constexpr static uint32_t PING_IDX = 0;
constexpr static uint32_t PONG_IDX = 1;
constexpr static uint32_t UB_ALIGN = 32; // UB按32字节对齐
constexpr static uint32_t BITS32_PER_BLOCK = 8;
constexpr static uint32_t BITS16_PER_BLOCK = 16;
constexpr static uint32_t BITS32_PER_REPEAT = 64;
constexpr static uint32_t BITS16_PER_REPEAT = 128;
constexpr static uint32_t BW_ITEM_SIZE = 32; // = sizeof(BatchWriteItem)
constexpr static uint32_t U64_PER_ITEM = BW_ITEM_SIZE / sizeof(uint64_t);
constexpr static uint32_t U32_PER_ITEM = BW_ITEM_SIZE / sizeof(uint32_t);
constexpr static uint32_t SKIP_OFFSET = 512;
constexpr static int32_t FLAG_VALUE = 0xFFFFFFFF;
constexpr uint32_t A2_RANK_NUM_PER_SERVER = 8;
constexpr static uint32_t BITS_PER_U32 = 32;  // 单个uint32_t的bit数
constexpr static uint32_t VEC_UB_ALIGN = 256; // Vector指令进行Repeat时的粒度

template <typename T1, typename T2>
__aicore__ inline T2 RoundUp(const T1 val, const T2 align)
{
    return (static_cast<T2>(val) + align - 1) / align * align;
}

struct TaskInfo {
    uint32_t startTaskId{0};
    uint32_t endTaskId{0};
    uint32_t taskNum{0};

    __aicore__ inline void SplitCore(uint32_t taskNumTotal, uint32_t aivNum, uint32_t aivId)
    {
        if (unlikely(aivNum == 0)) {
            startTaskId = endTaskId = taskNum = 0;
            return;
        }

        const uint32_t baseNum = taskNumTotal / aivNum;
        const uint32_t remainder = taskNumTotal % aivNum;

        const bool hasExtraTask = aivId < remainder;
        taskNum = baseNum + static_cast<uint32_t>(hasExtraTask);
        startTaskId = baseNum * aivId + (hasExtraTask ? aivId : remainder);
        endTaskId = startTaskId + taskNum;
    }
};

#define TemplateMC2TypeA2Class \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, \
        bool WithScalePayload
#define TemplateMC2TypeA2Func XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, WithScalePayload

using namespace AscendC;
template <TemplateMC2TypeA2Class>
class MoeDistributeDispatchA2 {
public:
    __aicore__ inline MoeDistributeDispatchA2(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut,
                                GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AllocTensor();
    __aicore__ inline void InitStatusTensor();
    __aicore__ inline void IndexSort();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void GetStatusCumSum();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void ConstructBatchWriteInfo();
    __aicore__ inline void ReorderTokens();
    __aicore__ inline void QuantProcess(uint32_t expertIndex, TEventID eventId);
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void ZeroComputeExpertMaskCal();
    __aicore__ inline void CalValidTokenCount();
    __aicore__ inline void CleanUpFlags();
    __aicore__ inline void CopyPerformanceInfo();
    __aicore__ inline void SingleServerSendToMoeExpert();
    __aicore__ inline void CopyStatusToADump(LocalTensor<bool> isVisited, LocalTensor<int32_t> dataFlagLocal);

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<int32_t> windowInstatusTensor_;
    GlobalTensor<uint32_t> batchWriteInfoTensor_;
    GlobalTensor<int32_t> sendStatusTensor_;
    GlobalTensor<uint32_t> bufferChosenGlobal_;
    GlobalTensor<int8_t> xActiveMaskGMTensor_;
    GlobalTensor<int32_t> performanceInfoI32GMTensor_;

    LocalTensor<XType> xTensor_[BUFFER_NUM];
    LocalTensor<ExpandXOutType> xOutTensor_[BUFFER_NUM];
    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<uint64_t> batchWriteU64Tensor_;
    LocalTensor<uint32_t> batchWriteU32Tensor_;
    LocalTensor<uint32_t> expertCumsumTensor_;
    LocalTensor<int32_t> validExpIndexTensor_;
    LocalTensor<uint16_t> zeroExpertMaskTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> xFloatTensor_;
    LocalTensor<int32_t> flagTensor_;
    LocalTensor<int8_t> xActiveMaskTensor_;
    LocalTensor<half> xActiveMaskHalfTensor_;
    LocalTensor<int32_t> epRecvCountsTempLocal_;
    LocalTensor<int32_t> epRecvCountsOutLocal_;
    LocalTensor<int32_t> performanceInfoI32Tensor_;
    LocalTensor<uint8_t> aDumpTensor_;

    GM_ADDR expandIdxOutGM_;
    GM_ADDR expertTokenNumsOutGM_;
    GM_ADDR epRecvCountsGM_;
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    GM_ADDR batchWriteInfo_;
    GM_ADDR adumpStatusGM_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t expertIdsCnt_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t aivId_{0};
    uint32_t moeExpertNum_{0}; // moe专家卡数, 等于worldSize_ - 共享专家卡数
    uint32_t hCommuSize_{0};
    uint32_t axisHCommu_{0};
    uint32_t localMoeExpertNum_{0};
    uint32_t localMoeExpertNumAlign_{0};
    uint32_t dataSizePerRank_{0};
    uint32_t dataSize_{0};
    uint32_t bufferChosen_{0};
    uint32_t totalSize_{0};
    uint64_t activeMaskBsCnt_{0};
    uint32_t expertTokenNumsType_{0};
    int32_t zeroComputeExpertNum_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    uint32_t statusEntryCount_{0};
    bool isTokenMaskFlag_{false};
    bool isExpertMaskFlag_{false};
    uint32_t performanceInfoSize_{0};
    bool needPerformanceInfo_{false};
    bool isSingleServer_{false};
    uint32_t dataOffset_{0}; // 数据空间起始偏移
    uint32_t maxMoeExpertNum_{0};
    uint32_t globalBs_{0};
    TaskInfo worldTaskInfo_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    __gm__ HcclOpResParam *winContext_{nullptr};
    Mc2A2Kernel::MoeDistributeA2ADump aDump_;
};

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR performanceInfo, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR epRecvCountsOut,
    GM_ADDR workspaceGM, TPipe *pipe, GM_ADDR tilingGM)
{
    tpipe_ = pipe;
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchA2TilingData, tilingData, tilingGM);

    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

    hccl_.InitV2(contextGM0, &tilingData);
    hccl_.SetCcTilingV2(offsetof(MoeDistributeDispatchA2TilingData, mc2CcTiling));

    winContext_ = (__gm__ HcclOpResParam *)contextGM0;
    rankId_ = tilingData.moeDistributeDispatchInfo.epRankId;
    windowInGM_ = hccl_.GetWindowsInAddr(rankId_);
    windowOutGM_ = hccl_.GetWindowsOutAddr(rankId_);

    axisBS_ = tilingData.moeDistributeDispatchInfo.bs;
    axisH_ = tilingData.moeDistributeDispatchInfo.h;
    axisK_ = tilingData.moeDistributeDispatchInfo.k;
    aivNum_ = tilingData.moeDistributeDispatchInfo.aivNum;
    worldSize_ = tilingData.moeDistributeDispatchInfo.epWorldSize;
    expertTokenNumsType_ = tilingData.moeDistributeDispatchInfo.expertTokenNumsType;
    isTokenMaskFlag_ = tilingData.moeDistributeDispatchInfo.isTokenMask;
    isExpertMaskFlag_ = tilingData.moeDistributeDispatchInfo.isExpertMask;
    zeroComputeExpertNum_ = tilingData.moeDistributeDispatchInfo.zeroComputeExpertNum;
    maxMoeExpertNum_ = tilingData.moeDistributeDispatchInfo.maxMoeExpertNum;
    globalBs_ = tilingData.moeDistributeDispatchInfo.globalBs;
    totalSize_ = winContext_->winSize / BUFFER_NUM;
    dataSize_ = totalSize_ - Mc2A2Kernel::ADUMP_WIN_SIZE;
    dataSizePerRank_ = dataSize_ / worldSize_ / UB_ALIGN * UB_ALIGN;
    moeExpertNum_ = tilingData.moeDistributeDispatchInfo.moeExpertNum;
    localMoeExpertNum_ = moeExpertNum_ / worldSize_;
    aivId_ = GetBlockIdx();
    expertIdsCnt_ = axisBS_ * axisK_;
    localMoeExpertNumAlign_ = (localMoeExpertNum_ + BITS32_PER_BLOCK - 1) / BITS32_PER_BLOCK * BITS32_PER_BLOCK;

    bufferChosenGlobal_.SetGlobalBuffer((__gm__ uint32_t *)(windowInGM_ + Mc2A2Kernel::FULLMESH_BUFFERID_ADDR));
    bufferChosen_ = bufferChosenGlobal_(0);

    adumpStatusGM_ = windowInGM_ + Mc2A2Kernel::ADUMP_STATUS_DISPATCH_ADDR;
    windowInGM_ = windowInGM_ + Mc2A2Kernel::ADUMP_WIN_SIZE + totalSize_ * bufferChosen_;
    windowOutGM_ = windowOutGM_ + Mc2A2Kernel::ADUMP_WIN_SIZE + totalSize_ * bufferChosen_;

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOut),
                                        worldSize_ * axisBS_ * localMoeExpertNum_ * axisH_);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)(dynamicScalesOut));
    windowInstatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowInGM_));
    sendStatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(windowOutGM_));

    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ int8_t *)xActiveMask);
    expandIdxOutGM_ = expandIdxOut;
    expertTokenNumsOutGM_ = expertTokenNumsOut;
    epRecvCountsGM_ = epRecvCountsOut;

    hCommuSize_ = axisH_ * sizeof(ExpandXOutType) + ((StaticQuant || DynamicQuant || WithScalePayload) ? UB_ALIGN : 0);
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);

    batchWriteInfo_ = workspaceGM;
    batchWriteInfoTensor_.SetGlobalBuffer((__gm__ uint32_t *)(batchWriteInfo_), worldSize_ * U32_PER_ITEM);

    if constexpr (StaticQuant || DynamicQuant || WithScalePayload) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    }

    needPerformanceInfo_ = performanceInfo != nullptr;
    if (unlikely(needPerformanceInfo_)) {
        performanceInfoSize_ = worldSize_;
        performanceInfoI32GMTensor_.SetGlobalBuffer((__gm__ int32_t *)performanceInfo);
    }

    isSingleServer_ = worldSize_ <= A2_RANK_NUM_PER_SERVER;
    worldTaskInfo_.SplitCore(worldSize_, aivNum_, aivId_);
    uint32_t aDumpSize =
        Std::max(RoundUp(Mc2A2Kernel::H_POS + 1U, BITS32_PER_BLOCK), BITS32_PER_BLOCK * 2U) * sizeof(uint32_t);
    uint32_t aDumpUbAddr = AscendC::TOTAL_UB_SIZE - aDumpSize;
    aDumpTensor_ = LocalTensor<uint8_t>{TPosition::LCM, aDumpUbAddr, aDumpSize};
    GM_ADDR aDumpAddr = hccl_.GetWindowsInAddr(rankId_) + Mc2A2Kernel::ADUMP_DATA_DISPATCH_ADDR_START;
    aDump_.Init(aDumpAddr, aivId_, rankId_, worldSize_, rankId_, moeExpertNum_, worldSize_, globalBs_, bufferChosen_,
                Mc2A2Kernel::IS_FULLMESH, aivNum_, aDumpTensor_, axisH_, axisBS_);
    aDump_.UpdateFullmeshRankTaskInfo(worldTaskInfo_.startTaskId, worldTaskInfo_.taskNum, Mc2A2Kernel::IS_SEND);
    aDump_.UpdateFullmeshRankTaskInfo(worldTaskInfo_.startTaskId, worldTaskInfo_.taskNum, Mc2A2Kernel::IS_RECV);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::AllocTensor()
{
    uint32_t xPongAddr = WithScalePayload ? hCommuSize_ : axisH_ * sizeof(XType);
    uint32_t xLength = WithScalePayload ? hCommuSize_ * BUFFER_NUM : axisH_ * sizeof(XType) + hCommuSize_;
    xTensor_[PING_IDX] = LocalTensor<XType>{TPosition::LCM, 0, axisH_};
    xTensor_[PONG_IDX] = LocalTensor<XType>{TPosition::LCM, xPongAddr, axisH_};
    xOutTensor_[PING_IDX] = LocalTensor<ExpandXOutType>{TPosition::LCM, 0, axisHCommu_};
    xOutTensor_[PONG_IDX] = LocalTensor<ExpandXOutType>{TPosition::LCM, xPongAddr, axisHCommu_};
    // statusEntry contains the number of tokens received by each expert
    // and a flag indicating whether the recv buffer has received data.
    statusEntryCount_ = RoundUp(localMoeExpertNum_ + 1, BITS32_PER_BLOCK);
    dataOffset_ = statusEntryCount_ * sizeof(uint32_t);
    uint32_t statusWorldSize =
        RoundUp(statusEntryCount_ * worldSize_ * sizeof(int32_t), VEC_UB_ALIGN) / sizeof(int32_t);
    statusTensor_ = LocalTensor<int32_t>{TPosition::LCM, 0, statusWorldSize};

    uint32_t batchWriteU32Size = U32_PER_ITEM * worldTaskInfo_.taskNum;
    uint32_t batchWriteU64Size = U64_PER_ITEM * worldTaskInfo_.taskNum;
    batchWriteU32Tensor_ = LocalTensor<uint32_t>{TPosition::LCM, 0, batchWriteU32Size};
    batchWriteU64Tensor_ = LocalTensor<uint64_t>{TPosition::LCM, 0, batchWriteU64Size};

    uint32_t statusWorldLength = statusWorldSize * sizeof(int32_t);
    uint32_t batchWriteLength = batchWriteU32Size * sizeof(uint32_t);
    uint32_t expertCumsumAddr = Std::max(xLength, Std::max(statusWorldLength, batchWriteLength));
    uint32_t expertCumsumSize = RoundUp(moeExpertNum_ + 1, BITS32_PER_BLOCK);
    expertCumsumTensor_ = LocalTensor<uint32_t>{TPosition::LCM, expertCumsumAddr, expertCumsumSize};

    uint32_t expertIdsAddr = expertCumsumAddr + expertCumsumSize * sizeof(uint32_t);
    uint32_t expertIdsSize = RoundUp(expertIdsCnt_, BITS32_PER_BLOCK);
    uint32_t expertIdsLength = expertIdsSize * sizeof(int32_t);
    expertIdsTensor_ = LocalTensor<int32_t>{TPosition::LCM, expertIdsAddr, expertIdsSize};

    uint32_t expertCountAddr = expertIdsAddr + expertIdsLength;
    expertCountTensor_ = LocalTensor<int32_t>{TPosition::LCM, expertCountAddr, expertIdsSize};

    uint32_t xfloatOrFlagAddr = expertCountAddr + expertIdsLength;
    flagTensor_ = LocalTensor<int32_t>{TPosition::LCM, xfloatOrFlagAddr, BITS32_PER_BLOCK};
    xFloatTensor_ = LocalTensor<float>{TPosition::LCM, xfloatOrFlagAddr, axisH_};

    uint32_t smoothScalesAddr = xfloatOrFlagAddr + axisH_ * sizeof(float);
    smoothScalesTensor_ = LocalTensor<float>{TPosition::LCM, smoothScalesAddr, axisH_};

    if (unlikely(needPerformanceInfo_)) {
        uint32_t performanceInfoI32Addr = smoothScalesAddr + axisH_ * sizeof(float);
        performanceInfoI32Tensor_ = LocalTensor<int32_t>{
            TPosition::LCM, performanceInfoI32Addr,
            RoundUp(performanceInfoSize_ * static_cast<uint32_t>(sizeof(int64_t)), UB_ALIGN) / sizeof(int32_t)};
    }

    uint32_t validExpIndexAddr = AscendC::TOTAL_UB_SIZE - expertIdsLength - aDumpTensor_.GetSize() * sizeof(uint8_t);
    validExpIndexTensor_ = LocalTensor<int32_t>{TPosition::LCM, validExpIndexAddr, expertIdsSize};

    uint32_t xActiveMaskAlignSize = RoundUp(isExpertMaskFlag_ ? expertIdsCnt_ : axisBS_, 128);
    xActiveMaskTensor_ = LocalTensor<int8_t>{TPosition::LCM, expertCountAddr, xActiveMaskAlignSize};
    uint32_t xActiveMaskHalfAddr = expertCountAddr + xActiveMaskAlignSize;
    xActiveMaskHalfTensor_ = LocalTensor<half>{TPosition::LCM, xActiveMaskHalfAddr, xActiveMaskAlignSize};

    uint32_t zeroExpertMaskSize = RoundUp(expertIdsCnt_, BITS32_PER_REPEAT) / (CHAR_BIT * sizeof(uint16_t));
    zeroExpertMaskTensor_ = LocalTensor<uint16_t>{TPosition::LCM, xActiveMaskHalfAddr, zeroExpertMaskSize};
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::CalValidTokenCount()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    } else if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }
    CreateVecIndex(validExpIndexTensor_, 0, RoundUp(expertIdsCnt_, BITS32_PER_BLOCK));
    if (sendToMoeExpTokenCnt_ > 0 && zeroComputeExpertNum_ > 0) {
        ZeroComputeExpertMaskCal();
    }
    PipeBarrier<PIPE_V>();
    if (sendToMoeExpTokenCnt_ == 0 || (!isExpertMaskFlag_ && zeroComputeExpertNum_ <= 0)) {
        return;
    }
    if (isExpertMaskFlag_) {
        if (zeroComputeExpertNum_ > 0) {
            auto &&xActiveMaskU16Tensor_ = xActiveMaskTensor_.ReinterpretCast<uint16_t>();
            And(xActiveMaskU16Tensor_, xActiveMaskU16Tensor_, zeroExpertMaskTensor_,
                Ceil(sendToMoeExpTokenCnt_, BITS16_PER_BLOCK));
            PipeBarrier<PIPE_V>();
        }
        GatherMask(validExpIndexTensor_, validExpIndexTensor_, xActiveMaskTensor_.ReinterpretCast<uint32_t>(), true,
                   sendToMoeExpTokenCnt_, {1, 1, 0, 0}, sendToMoeExpTokenCnt_);
    } else { // !isExpertMaskFlag_ && zeroComputeExpertNum_ > 0
        GatherMask(validExpIndexTensor_, validExpIndexTensor_, zeroExpertMaskTensor_.ReinterpretCast<uint32_t>(), true,
                   sendToMoeExpTokenCnt_, {1, 1, 0, 0}, sendToMoeExpTokenCnt_);
    }
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_MTE2>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::TokenActiveMaskCal()
{
    auto &&sharedTmpBuffer = xActiveMaskTensor_.ReinterpretCast<half>();
    auto xActiveMaskParams = DataCopyExtParams{1U, axisBS_, 0U, 0U, 0U};
    auto xActiveMaskCopyPadParams = DataCopyPadExtParams<int8_t>{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor_, xActiveMaskGMTensor_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(xActiveMaskHalfTensor_, xActiveMaskTensor_, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    ReduceSum(sharedTmpBuffer, xActiveMaskHalfTensor_, sharedTmpBuffer, axisBS_);
    activeMaskBsCnt_ = static_cast<int32_t>(AscendC::GetAccVal<half>());
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ExpertActiveMaskCal()
{
    auto maskCopyPadParams = DataCopyPadExtParams<int8_t>{false, 0U, 0U, 0U};
    auto maskParams = DataCopyExtParams{1U, expertIdsCnt_, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor_, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(xActiveMaskHalfTensor_, xActiveMaskTensor_, RoundMode::CAST_NONE, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    CompareScalar(xActiveMaskTensor_.template ReinterpretCast<uint8_t>(), xActiveMaskHalfTensor_, static_cast<half>(1),
                  AscendC::CMPMODE::EQ, RoundUp(expertIdsCnt_, BITS16_PER_REPEAT));
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ZeroComputeExpertMaskCal()
{
    auto expertIdsCntParams = DataCopyExtParams{1, sendToMoeExpTokenCnt_ * sizeof(int32_t), 0U, 0U, 0U};
    auto expertIdsCntCopyPadParams = DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Mins(expertIdsTensor_, expertIdsTensor_, static_cast<int32_t>(moeExpertNum_), sendToMoeExpTokenCnt_);
    PipeBarrier<PIPE_V>();
    CompareScalar(zeroExpertMaskTensor_.template ReinterpretCast<uint8_t>(), expertIdsTensor_,
                  static_cast<int32_t>(moeExpertNum_), AscendC::CMPMODE::EQ,
                  RoundUp(sendToMoeExpTokenCnt_, BITS32_PER_REPEAT));
    PipeBarrier<PIPE_V>();
    Not(zeroExpertMaskTensor_, zeroExpertMaskTensor_, Ceil(sendToMoeExpTokenCnt_, BITS16_PER_BLOCK));
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::QuantProcess(uint32_t expertIndex,
                                                                                    TEventID eventId)
{
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    WaitFlag<HardEvent::MTE2_V>(eventId);
    Cast(xFloatTensor_, xTensor_[PING_IDX], RoundMode::CAST_NONE, axisH_);
    SetFlag<HardEvent::V_MTE2>(eventId);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSmoothScaleExist) {
        DataCopy(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], axisH_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(xFloatTensor_, xFloatTensor_, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<float> &absTensor = smoothScalesTensor_;
    LocalTensor<float> &maxTensor = smoothScalesTensor_;
    Abs(absTensor, xFloatTensor_, axisH_);
    PipeBarrier<PIPE_V>();
    ReduceMax(maxTensor, absTensor, absTensor, axisH_, false);
    SyncFunc<AscendC::HardEvent::V_S>();
    float dynamicScale = float(127.0) / maxTensor.GetValue(0);
    SyncFunc<AscendC::HardEvent::S_V>();

    Muls(xFloatTensor_, xFloatTensor_, dynamicScale, axisH_);
    PipeBarrier<PIPE_V>();
    auto &&xHalfTensor = xFloatTensor_.ReinterpretCast<half>();
    auto &&xIntTensor = xFloatTensor_.ReinterpretCast<int32_t>();
    Cast(xIntTensor, xFloatTensor_, RoundMode::CAST_RINT, axisH_);
    SetDeqScale((half)1.0f);
    PipeBarrier<PIPE_V>();
    Cast(xHalfTensor, xIntTensor, RoundMode::CAST_ROUND, axisH_);
    PipeBarrier<PIPE_V>();
    Cast(xOutTensor_[1], xHalfTensor, RoundMode::CAST_TRUNC, axisH_);
    SetFlag<HardEvent::V_MTE3>(eventId);
    auto &&xOutFloatTensor = xOutTensor_[PONG_IDX].template ReinterpretCast<float>();
    dynamicScale = 1 / dynamicScale;
    xOutFloatTensor.SetValue(axisH_ / sizeof(float), dynamicScale);
    SetFlag<HardEvent::S_MTE3>(eventId);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::InitStatusTensor()
{
    Duplicate<int32_t>(statusTensor_, 0, worldSize_ * statusEntryCount_);
    uint32_t statusEntryBlockNum = statusEntryCount_ / BITS32_PER_BLOCK;
    uint32_t statusEntryNumPerRepeat = AscendC::DEFAULT_REPEAT_STRIDE / statusEntryBlockNum;
    constexpr uint64_t maskTemplate[] = {
        0x0ULL,                 // placeholder
        0x8080808080808080ULL,  // Flag last element of each block when StatusEntry size is 1 blocks
        0x8000800080008000ULL,  // Flag last element of every 2 blocks when StatusEntry size is 2 blocks
        0x800000800000ULL,      // Flag last element of blocks 3 & 6 when StatusEntry size is 3 blocks
        0x8000000080000000ULL,  // Flag last element of blocks 4 & 8 when StatusEntry size is 4 blocks
        0x8000000000ULL,        // Flag last element of 5th block when StatusEntry size is 5 blocks
        0x800000000000ULL,      // Flag last element of 6th block when StatusEntry size is 6 blocks
        0x80000000000000ULL,    // Flag last element of 7th block when StatusEntry size is 7 blocks
        0x8000000000000000ULL}; // Flag last element of 8th block when StatusEntry size is 8 blocks
    uint32_t repeatTime, dstRepeatStride, statusEntryOffset;
    uint64_t mask[2] = {0};
    if (statusEntryNumPerRepeat > 0) {
        repeatTime = Ceil(worldSize_, statusEntryNumPerRepeat);
        dstRepeatStride = statusEntryNumPerRepeat * statusEntryBlockNum;
        statusEntryOffset = 0;
        mask[0] = maskTemplate[statusEntryBlockNum];
    } else {
        repeatTime = worldSize_;
        dstRepeatStride = statusEntryBlockNum;
        statusEntryOffset = (statusEntryBlockNum - AscendC::DEFAULT_REPEAT_STRIDE) * BITS32_PER_BLOCK;
        mask[0] = maskTemplate[AscendC::DEFAULT_REPEAT_STRIDE];
    }
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_[statusEntryOffset], FLAG_VALUE, mask, repeatTime, 1, dstRepeatStride);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::IndexSort()
{
    uint32_t activeExpertIds = activeMaskBsCnt_ * axisK_;
    auto copyExpertIdsParams = DataCopyExtParams{1, static_cast<uint32_t>(activeExpertIds * sizeof(int32_t)), 0, 0, 0};
    auto padParams = DataCopyPadExtParams<int32_t>{false, 0, 0, 0};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, copyExpertIdsParams, padParams);
    Duplicate(expertCountTensor_, 0, RoundUp(activeExpertIds, BITS32_PER_BLOCK));
    InitStatusTensor();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    SyncFunc<AscendC::HardEvent::V_S>();
    // 统计给每个专家发的token数
    for (uint32_t tokenIndex = 0; tokenIndex < sendToMoeExpTokenCnt_; ++tokenIndex) {
        int32_t expertIdx = validExpIndexTensor_(tokenIndex);
        int32_t expertId = expertIdsTensor_(expertIdx);
        int32_t rankId = expertId / localMoeExpertNum_;
        int32_t expertOffsetInRank = expertId % localMoeExpertNum_;
        expertCountTensor_(expertIdx) = statusTensor_(rankId * statusEntryCount_ + expertOffsetInRank);
        statusTensor_(rankId * statusEntryCount_ + expertOffsetInRank)++;
    }
    expertCumsumTensor_(0) = 0;
    for (uint32_t expertId = 1; expertId < moeExpertNum_; expertId++) {
        int32_t rankId = (expertId - 1) / localMoeExpertNum_;
        int32_t expertOffsetInRank = (expertId - 1) % localMoeExpertNum_;
        uint32_t count = statusTensor_(rankId * statusEntryCount_ + expertOffsetInRank);
        uint32_t preSum = expertCumsumTensor_(expertId - 1);
        expertCumsumTensor_(expertId) = count + preSum;
    }
    expertCumsumTensor_(moeExpertNum_) = sendToMoeExpTokenCnt_;

    if (aivId_ == aivNum_ - 1) {
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> expandIdxGMTensor;
        expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t *)expandIdxOutGM_);
        DataCopyPad(expandIdxGMTensor, expertCountTensor_, copyExpertIdsParams);
        DataCopy(windowInstatusTensor_[rankId_ * dataSizePerRank_ / sizeof(int32_t)],
                 statusTensor_[rankId_ * statusEntryCount_], statusEntryCount_);
        Duplicate<int32_t>(flagTensor_, FLAG_VALUE, BITS32_PER_BLOCK);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        for (uint32_t rankId = 0; rankId < worldSize_; rankId++) {
            uint64_t rankOffset = rankId * dataSizePerRank_ / sizeof(int32_t);
            DataCopy(sendStatusTensor_[rankOffset], statusTensor_[rankId * statusEntryCount_], statusEntryCount_);
            uint32_t startExpertId = rankId * localMoeExpertNum_;
            uint32_t tokenCount =
                expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
            uint64_t dataFlagOffset =
                rankOffset + (dataOffset_ + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
            DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(sendStatusTensor_[dataFlagOffset], flagTensor_, copyFlagParams);
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ReorderTokens()
{
    TaskInfo tokenTaskInfo;
    tokenTaskInfo.SplitCore(sendToMoeExpTokenCnt_, aivNum_, aivId_);
    GlobalTensor<ExpandXOutType> sendTokensGlobal;
    constexpr auto event = (StaticQuant || DynamicQuant) ? HardEvent::V_MTE2 : HardEvent::MTE3_MTE2;
    const uint32_t scaleOffset = axisH_ * sizeof(ExpandXOutType) / sizeof(float);
    SetFlag<event>(EVENT_ID0);
    SetFlag<event>(EVENT_ID1);
    for (uint32_t tokenIndex = tokenTaskInfo.startTaskId; tokenIndex < tokenTaskInfo.endTaskId; ++tokenIndex) {
        int32_t eventId = (tokenIndex & 1) ? EVENT_ID0 : EVENT_ID1;
        int32_t expertIdx = validExpIndexTensor_(tokenIndex);
        int32_t expertId = expertIdsTensor_(expertIdx);
        int32_t rankId = expertId / localMoeExpertNum_;
        int32_t startExpertId = rankId * localMoeExpertNum_;
        uint32_t expertOffset = expertCumsumTensor_(expertId) - expertCumsumTensor_(startExpertId);
        int32_t tokenOffset = expertCountTensor_(expertIdx);
        sendTokensGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(windowOutGM_ + rankId * dataSizePerRank_ + dataOffset_));
        if constexpr (WithScalePayload) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            uint32_t srcTokenIndex = expertIdx / axisK_;
            DataCopy(xOutTensor_[eventId], xGMTensor_[srcTokenIndex * axisH_], axisH_);
            auto xOutFloatTensor = xOutTensor_[eventId].template ReinterpretCast<float>();
            DataCopyExtParams scaleCopyParams{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> scalePadParams{false, 0U, 0U, 0U};
            DataCopyPad(xOutFloatTensor[scaleOffset], scalesGMTensor_[srcTokenIndex], scaleCopyParams, scalePadParams);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(sendTokensGlobal[(expertOffset + tokenOffset) * axisHCommu_], xOutTensor_[eventId], axisHCommu_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        } else if constexpr (StaticQuant || DynamicQuant) {
            WaitFlag<HardEvent::V_MTE2>(eventId);
            DataCopy(xTensor_[PING_IDX], xGMTensor_[expertIdx / axisK_ * axisH_], axisH_);
            SetFlag<HardEvent::MTE2_V>(eventId);
            QuantProcess(expertId, eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::S_MTE3>(eventId);
            DataCopy(sendTokensGlobal[(expertOffset + tokenOffset) * axisHCommu_], xOutTensor_[PONG_IDX], axisHCommu_);
        } else {
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(xTensor_[eventId], xGMTensor_[expertIdx / axisK_ * axisH_],
                     axisH_); // 约束对齐 expertIdx / axisK_ * axisH_
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(sendTokensGlobal[(expertOffset + tokenOffset) * axisHCommu_], xTensor_[eventId], axisHCommu_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
    }
    WaitFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID1);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::SingleServerSendToMoeExpert()
{
    if (aivId_ == 0) {
        bufferChosenGlobal_(0) = bufferChosen_ ^ 1;
    }
    for (uint32_t rankIndex = worldTaskInfo_.startTaskId; rankIndex < worldTaskInfo_.endTaskId; ++rankIndex) {
        uint32_t startExpertId = rankIndex * localMoeExpertNum_;
        uint32_t currentIndex = rankIndex - worldTaskInfo_.startTaskId;
        uint32_t tokenCount =
            expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
        GM_ADDR rankGM = (__gm__ uint8_t *)(hccl_.GetWindowsInAddr(rankIndex) + Mc2A2Kernel::ADUMP_WIN_SIZE +
                                            totalSize_ * bufferChosen_ + (dataSizePerRank_ * rankId_));
        GM_ADDR localBuf = (__gm__ uint8_t *)(windowOutGM_ + dataSizePerRank_ * rankIndex);

        GlobalTensor<ExpandXOutType> currRankWindowInGlobal;
        GlobalTensor<ExpandXOutType> currRankWindowOutGlobal;
        currRankWindowInGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)rankGM);
        currRankWindowOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)localBuf);

        DataCopy(xOutTensor_[PING_IDX], currRankWindowOutGlobal,
                 statusEntryCount_ * sizeof(int32_t) / sizeof(ExpandXOutType));
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopy(currRankWindowInGlobal, xOutTensor_[PING_IDX],
                 statusEntryCount_ * sizeof(int32_t) / sizeof(ExpandXOutType));
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();

        currRankWindowInGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(rankGM + dataOffset_));
        currRankWindowOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(localBuf + dataOffset_));

        SyncFunc<AscendC::HardEvent::S_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t currTokenIdx = 0; currTokenIdx < tokenCount; currTokenIdx++) {
            TEventID eventId = (currTokenIdx & 1) ? EVENT_ID0 : EVENT_ID1;
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(xOutTensor_[eventId], currRankWindowOutGlobal[currTokenIdx * axisHCommu_], axisHCommu_);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(currRankWindowInGlobal[currTokenIdx * axisHCommu_], xOutTensor_[eventId], axisHCommu_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        currRankWindowInGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(rankGM + dataOffset_ + tokenCount * hCommuSize_ + SKIP_OFFSET));
        currRankWindowOutGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(localBuf + dataOffset_ + tokenCount * hCommuSize_ + SKIP_OFFSET));

        DataCopyExtParams copySkipOffsetFlagParams{1, static_cast<uint32_t>(sizeof(uint32_t)), 0, 0, 0};
        DataCopyPadExtParams<ExpandXOutType> padParams{false, 0, 0, 0};

        DataCopyPad(xOutTensor_[PING_IDX], currRankWindowOutGlobal, copySkipOffsetFlagParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(currRankWindowInGlobal, xOutTensor_[PING_IDX], copySkipOffsetFlagParams);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        aDump_.UpdateFullmeshEpRankSendOrRecv(rankIndex, Mc2A2Kernel::IS_SEND);
        aDump_.UpdateFullmeshTokenSendOrRecv(rankIndex, tokenCount, Mc2A2Kernel::IS_SEND);
    }
    PipeBarrier<PIPE_ALL>(); // 与WaitDispatch的LocalTensor高度重叠
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::ConstructBatchWriteInfo()
{
    uint32_t batchWriteDataType = static_cast<uint32_t>(AscendC::HcclDataType::HCCL_DATA_TYPE_INT8);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    for (uint32_t rankIndex = worldTaskInfo_.startTaskId; rankIndex < worldTaskInfo_.endTaskId; ++rankIndex) {
        uint32_t startExpertId = rankIndex * localMoeExpertNum_;
        uint32_t currentIndex = rankIndex - worldTaskInfo_.startTaskId;
        uint32_t tokenCount =
            expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
        GM_ADDR rankGM = (__gm__ uint8_t *)(hccl_.GetWindowsInAddr(rankIndex) + Mc2A2Kernel::ADUMP_WIN_SIZE +
                                            totalSize_ * bufferChosen_ + (dataSizePerRank_ * rankId_));
        GM_ADDR localBuf = (__gm__ uint8_t *)(windowOutGM_ + dataSizePerRank_ * rankIndex);
        uint64_t batchWriteDataSize = dataOffset_ + tokenCount * hCommuSize_ + sizeof(int32_t) + SKIP_OFFSET;
        batchWriteU64Tensor_(currentIndex * U64_PER_ITEM) = (uint64_t)localBuf;
        batchWriteU64Tensor_(currentIndex * U64_PER_ITEM + 1) = (uint64_t)rankGM;
        batchWriteU64Tensor_(currentIndex * U64_PER_ITEM + 2) = batchWriteDataSize;
        batchWriteU32Tensor_(currentIndex * U32_PER_ITEM + 6) = batchWriteDataType;
        batchWriteU32Tensor_(currentIndex * U32_PER_ITEM + 7) = rankIndex;
        aDump_.UpdateFullmeshEpRankSendOrRecv(rankIndex, Mc2A2Kernel::IS_SEND);
        aDump_.UpdateFullmeshTokenSendOrRecv(rankIndex, tokenCount, Mc2A2Kernel::IS_SEND);
    }

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(batchWriteInfoTensor_[worldTaskInfo_.startTaskId * U32_PER_ITEM], batchWriteU32Tensor_,
             worldTaskInfo_.taskNum * U32_PER_ITEM);
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::SendToMoeExpert()
{
    if (aivId_ == 0) {
        HcclHandle batchWriteResult = hccl_.BatchWrite<true>(batchWriteInfo_, worldSize_);
        bufferChosenGlobal_(0) = bufferChosen_ ^ 1;
    }
    if (aivId_ == aivNum_ - 1) {
        uint32_t startExpertId = rankId_ * localMoeExpertNum_;
        uint32_t tokenCount =
            expertCumsumTensor_(startExpertId + localMoeExpertNum_) - expertCumsumTensor_(startExpertId);
        GlobalTensor<ExpandXOutType> currRankWindowInGlobal;
        GlobalTensor<ExpandXOutType> currRankWindowOutGlobal;
        currRankWindowInGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(windowInGM_ + rankId_ * dataSizePerRank_ + dataOffset_));
        currRankWindowOutGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(windowOutGM_ + rankId_ * dataSizePerRank_ + dataOffset_));
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t currTokenIdx = 0; currTokenIdx < tokenCount; currTokenIdx++) {
            int32_t eventId = (currTokenIdx & 1) ? EVENT_ID0 : EVENT_ID1;
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(xOutTensor_[eventId], currRankWindowOutGlobal[currTokenIdx * axisHCommu_], axisHCommu_);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(currRankWindowInGlobal[currTokenIdx * axisHCommu_], xOutTensor_[eventId], axisHCommu_);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        uint64_t dataFlagOffset =
            (rankId_ * dataSizePerRank_ + dataOffset_ + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        windowInstatusTensor_(dataFlagOffset) = FLAG_VALUE;
        DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
            windowInstatusTensor_[dataFlagOffset]);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::CopyStatusToADump(
    LocalTensor<bool> isVisited, LocalTensor<int32_t> dataFlagLocal)
{
    DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    GlobalTensor<int32_t> aDumpStatusGM;
    aDumpStatusGM.SetGlobalBuffer((__gm__ int32_t *)(adumpStatusGM_));

    uint32_t countPerRank = statusEntryCount_ + 2U;
    for (uint32_t rankId = worldTaskInfo_.startTaskId; rankId < worldTaskInfo_.endTaskId; rankId++) {
        if (isVisited(rankId - worldTaskInfo_.startTaskId)) {
            continue;
        }
        dataFlagLocal.SetValue(0, Mc2A2Kernel::DUMP_FLAG_VALUE);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(aDumpStatusGM[rankId * countPerRank], dataFlagLocal, copyFlagParams); // 设置AdumpFlag
        DataCopy(aDumpStatusGM[rankId * countPerRank + 1U], statusTensor_[rankId * statusEntryCount_],
                 statusEntryCount_);
        int32_t statusFlag = statusTensor_(rankId * statusEntryCount_ + statusEntryCount_ - 1);
        if (statusFlag != FLAG_VALUE) {
            continue;
        }
        uint32_t tokenCount = 0;
        for (int32_t expertOffset = 0; expertOffset < localMoeExpertNum_; expertOffset++) {
            tokenCount += statusTensor_(rankId * statusEntryCount_ + expertOffset);
        }
        if (tokenCount > globalBs_) {
            continue;
        }
        uint64_t dataFlagOffset =
            (rankId * dataSizePerRank_ + dataOffset_ + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        DataCopyPad(dataFlagLocal, windowInstatusTensor_[dataFlagOffset], copyFlagParams, padParams);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(aDumpStatusGM[(rankId + 1) * countPerRank - 1U], dataFlagLocal, copyFlagParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::WaitDispatch()
{
    if (unlikely(needPerformanceInfo_)) {
        // 避免没有被分配任务的核未初始化performanceInfoI32Tensor_
        Duplicate<int32_t>(performanceInfoI32Tensor_, 0, performanceInfoI32Tensor_.GetSize());
        SyncFunc<AscendC::HardEvent::V_S>();
    }

    if (worldTaskInfo_.taskNum == 0) {
        SyncAll<true>();
        return;
    }

    DataCopyExtParams copyFlagParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    auto dataFlagLocal =
        LocalTensor<int32_t>{TPosition::LCM, statusTensor_.GetSize() * sizeof(int32_t), BITS32_PER_BLOCK};
    auto isVisited = LocalTensor<bool>{
        TPosition::LCM, statusTensor_.GetSize() * sizeof(int32_t) + dataFlagLocal.GetSize() * sizeof(int32_t),
        RoundUp(worldTaskInfo_.taskNum, UB_ALIGN)};
    auto &&isVisitedU32 = isVisited.template ReinterpretCast<uint32_t>();
    Duplicate<uint32_t>(isVisitedU32, uint32_t(0), isVisitedU32.GetSize());
    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();

    bool unDump = true;
    uint32_t recvFlagNum = 0;
    int64_t startTime = GetCurrentTimestampUs();
    while (recvFlagNum < worldTaskInfo_.taskNum) {
        for (uint32_t rankId = worldTaskInfo_.startTaskId; rankId < worldTaskInfo_.endTaskId; rankId++) {
            if (isVisited(rankId - worldTaskInfo_.startTaskId)) {
                continue;
            }
            DataCopy(statusTensor_[rankId * statusEntryCount_],
                     windowInstatusTensor_[rankId * dataSizePerRank_ / sizeof(int32_t)], statusEntryCount_);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            int32_t statusFlag = statusTensor_(rankId * statusEntryCount_ + statusEntryCount_ - 1);
            if (statusFlag != FLAG_VALUE) {
                continue;
            }
            uint32_t tokenCount = 0;
            for (int32_t expertOffset = 0; expertOffset < localMoeExpertNum_; expertOffset++) {
                tokenCount += statusTensor_(rankId * statusEntryCount_ + expertOffset);
            }
            uint64_t dataFlagOffset =
                (rankId * dataSizePerRank_ + dataOffset_ + tokenCount * hCommuSize_ + SKIP_OFFSET) / sizeof(int32_t);
            DataCopyPad(dataFlagLocal, windowInstatusTensor_[dataFlagOffset], copyFlagParams, padParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            int32_t dataFlag = dataFlagLocal(0);
            if (dataFlag != FLAG_VALUE) {
                continue;
            }
            isVisited(rankId - worldTaskInfo_.startTaskId) = true;
            recvFlagNum++;
            windowInstatusTensor_(dataFlagOffset) = 0;
            if (unlikely(needPerformanceInfo_)) {
                auto srcRankId = rankId;
                RecordRankCommDuration(performanceInfoI32Tensor_, srcRankId, startTime);
            }
            aDump_.UpdateFullmeshEpRankSendOrRecv(rankId, Mc2A2Kernel::IS_RECV);
            aDump_.UpdateFullmeshTokenSendOrRecv(rankId, tokenCount, Mc2A2Kernel::IS_RECV);
        }
        if (unDump && (GetCurrentTimestampUs() - startTime) > Mc2A2Kernel::TIMEOUT) {
            CopyStatusToADump(isVisited, dataFlagLocal);
            unDump = false;
        }
    }
    SyncAll<true>();
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::GetStatusCumSum()
{
    const uint32_t statusSize = statusEntryCount_ * sizeof(int32_t);
    uint32_t srcStrideU32 = dataSizePerRank_ - statusSize;
    DataCopyExtParams copyStatusParams{static_cast<uint16_t>(worldSize_), statusSize, srcStrideU32, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyPad(statusTensor_, windowInstatusTensor_, copyStatusParams, padParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    uint32_t epRecvCountsTempAddr = statusTensor_.GetSize() * sizeof(int32_t);
    uint32_t epRecvCountsTempSize = localMoeExpertNumAlign_ * worldSize_;
    epRecvCountsTempLocal_ = LocalTensor<int32_t>{TPosition::LCM, epRecvCountsTempAddr, epRecvCountsTempSize};
    uint32_t epRecvCountsOutAddr = epRecvCountsTempAddr + epRecvCountsTempSize * sizeof(int32_t);
    uint32_t worldSizeAlign = RoundUp(worldSize_, BITS32_PER_BLOCK);
    uint32_t epRecvCountsOutSize = localMoeExpertNum_ * worldSizeAlign;
    epRecvCountsOutLocal_ = LocalTensor<int32_t>{TPosition::LCM, epRecvCountsOutAddr, epRecvCountsOutSize};
    uint16_t srcStrideU16 = (statusEntryCount_ - localMoeExpertNumAlign_) / BITS32_PER_BLOCK;
    uint16_t worldSizeU16 = (uint16_t)worldSize_;
    DataCopyParams copyParamsMultiple{worldSizeU16, static_cast<uint16_t>(localMoeExpertNumAlign_ / BITS32_PER_BLOCK),
                                      srcStrideU16, 0};
    DataCopy(epRecvCountsTempLocal_, statusTensor_, copyParamsMultiple);
    PipeBarrier<PIPE_V>();
    for (uint32_t rankIndex = 1; rankIndex < worldSize_; ++rankIndex) {
        uint32_t statusOffset = rankIndex * localMoeExpertNumAlign_;
        Add(epRecvCountsTempLocal_[statusOffset], epRecvCountsTempLocal_[statusOffset - localMoeExpertNumAlign_],
            epRecvCountsTempLocal_[statusOffset], localMoeExpertNum_);
        PipeBarrier<PIPE_V>();
    }
    uint32_t patternAddr = epRecvCountsOutAddr + epRecvCountsOutSize * sizeof(int32_t);
    uint32_t patternSize = localMoeExpertNumAlign_;
    if (worldSizeAlign != worldSize_) {
        // worldSize_不为8的倍数时，epRecvCounts 需要多进行一次 GatherMask。
        // 此段代码计算这次额外操作所需的 Pattern 内存大小上限：每个长度为 worldSizeAlign 的对齐块中，只取前 worldSize_
        // 个元素进行采集。
        uint32_t requiredPattern = RoundUp(((worldSizeAlign + BITS_PER_U32 - 1) / BITS_PER_U32), BITS32_PER_BLOCK);
        patternSize = requiredPattern > patternSize ? requiredPattern : patternSize;
    }
    auto patternLocal = LocalTensor<uint32_t>{TPosition::LCM, patternAddr, patternSize};
    Duplicate<uint32_t>(patternLocal, 0, patternSize);
    SyncFunc<AscendC::HardEvent::V_S>();
    srcStrideU16 = localMoeExpertNumAlign_ * sizeof(int32_t) / UB_ALIGN;
    int32_t previousSum = 0;
    uint64_t rsvdCnt = 0;
    uint32_t mask4Gather = localMoeExpertNumAlign_;
    uint32_t patternCount = (localMoeExpertNum_ + BITS_PER_U32 - 1) / BITS_PER_U32;
    uint32_t remainingMoeExperts = localMoeExpertNum_;
    for (uint32_t patternIdx = 0; patternIdx < patternCount; ++patternIdx) {
        patternLocal(patternIdx) = 1;
        uint32_t currentExpertBatchSize = remainingMoeExperts > BITS_PER_U32 ? BITS_PER_U32 : remainingMoeExperts;
        uint32_t expertBatchStartIdx = patternIdx * BITS_PER_U32;
        for (uint32_t expertIndex = expertBatchStartIdx; expertIndex < expertBatchStartIdx + currentExpertBatchSize;
             ++expertIndex) {
            SyncFunc<AscendC::HardEvent::S_V>();
            GatherMask(epRecvCountsOutLocal_[expertIndex * worldSizeAlign], epRecvCountsTempLocal_, patternLocal, true,
                       mask4Gather, {1, worldSizeU16, srcStrideU16, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();
            Adds(epRecvCountsOutLocal_[expertIndex * worldSizeAlign],
                 epRecvCountsOutLocal_[expertIndex * worldSizeAlign], previousSum, worldSize_);
            SyncFunc<AscendC::HardEvent::V_S>();
            previousSum = epRecvCountsOutLocal_(expertIndex * worldSizeAlign + worldSize_ - 1);
            patternLocal(patternIdx) = patternLocal(patternIdx) << 1;
        }
        remainingMoeExperts -= currentExpertBatchSize;
        patternLocal(patternIdx) = 0;
    }
    if (worldSizeAlign != worldSize_) {
        mask4Gather = worldSizeAlign;
        uint16_t repeatTimes = static_cast<uint16_t>(localMoeExpertNum_);
        uint16_t src0RepeatStride = static_cast<uint16_t>(worldSizeAlign * sizeof(int32_t) / UB_ALIGN);

        patternCount = (worldSize_ + BITS_PER_U32 - 1) / BITS_PER_U32;
        Duplicate<uint32_t>(patternLocal, UINT32_MAX, patternCount);
        SyncFunc<AscendC::HardEvent::V_S>();
        patternLocal(patternCount - 1) = (1 << (worldSize_ % BITS_PER_U32)) - 1;
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(epRecvCountsOutLocal_, epRecvCountsOutLocal_, patternLocal, true, mask4Gather,
                   {1, repeatTimes, src0RepeatStride, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>(); // localWindowCopy需要用到
    }

    if (aivId_ == aivNum_ - 1) {
        uint32_t expertTokenNumsW64Addr = patternAddr + localMoeExpertNumAlign_ * sizeof(uint32_t);
        auto expertTokenNumsW64Local = LocalTensor<int32_t>{TPosition::LCM, expertTokenNumsW64Addr,
                                                            localMoeExpertNum_ * (sizeof(int64_t) / sizeof(int32_t))};
        if (expertTokenNumsType_ == 0) {
            for (int i = 0; i < localMoeExpertNum_; i++) {
                expertTokenNumsW64Local(i * 2) = epRecvCountsOutLocal_((i + 1) * worldSize_ - 1);
                expertTokenNumsW64Local(i * 2 + 1) = 0;
            }
        } else {
            uint32_t tokenCountOffset = (worldSize_ - 1) * localMoeExpertNumAlign_;
            for (int i = 0; i < localMoeExpertNum_; i++) {
                expertTokenNumsW64Local(i * 2) = epRecvCountsTempLocal_(tokenCountOffset + i);
                expertTokenNumsW64Local(i * 2 + 1) = 0;
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> expertTokenNumsGlobal;
        expertTokenNumsGlobal.SetGlobalBuffer((__gm__ int32_t *)(expertTokenNumsOutGM_));
        DataCopyExtParams copyPadParams{1, static_cast<uint32_t>(localMoeExpertNum_ * sizeof(int64_t)), 0, 0, 0};
        DataCopyPad(expertTokenNumsGlobal, expertTokenNumsW64Local, copyPadParams);

        GlobalTensor<int32_t> epRecvCountsGlobal;
        epRecvCountsGlobal.SetGlobalBuffer((__gm__ int32_t *)(epRecvCountsGM_));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyExtParams epRecvCountPadParams{1, static_cast<uint32_t>(moeExpertNum_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(epRecvCountsGlobal, epRecvCountsOutLocal_, epRecvCountPadParams);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::LocalWindowCopy()
{
    GlobalTensor<ExpandXOutType> currRankWindowGlobal;
    uint32_t xOutTensor0Addr = epRecvCountsOutLocal_.GetPhyAddr() + epRecvCountsOutLocal_.GetSize() * sizeof(int32_t);
    uint32_t xOutTensor1Addr = xOutTensor0Addr + axisHCommu_ * sizeof(ExpandXOutType);
    uint32_t dynamicScalesAddr = xOutTensor1Addr + axisHCommu_ * sizeof(ExpandXOutType);
    const uint32_t scaleOffset = axisH_ * sizeof(ExpandXOutType) / sizeof(float);
    uint32_t tokenNum = epRecvCountsOutLocal_(localMoeExpertNum_ * worldSize_ - 1);
    auto dynamicScalesTensor = LocalTensor<float>{TPosition::LCM, dynamicScalesAddr, tokenNum};
    xOutTensor_[0] = LocalTensor<ExpandXOutType>{TPosition::LCM, xOutTensor0Addr, axisHCommu_};
    xOutTensor_[1] = LocalTensor<ExpandXOutType>{TPosition::LCM, xOutTensor1Addr, axisHCommu_};
    for (uint32_t rankId = worldTaskInfo_.startTaskId; rankId < worldTaskInfo_.endTaskId; rankId++) {
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowInGM_) + rankId * dataSizePerRank_ + dataOffset_;
        currRankWindowGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr));
        uint32_t currRankDataOffset = 0;
        uint32_t currRankStatusOffset = rankId * statusEntryCount_;

        for (uint32_t j = 0; j < localMoeExpertNum_; j++) {
            // 将数据从Window拷贝到UB
            uint32_t currTokensCount = statusTensor_(currRankStatusOffset + j);
            uint32_t currTokensOffset = epRecvCountsOutLocal_(j * worldSize_ + rankId) - currTokensCount;
            uint32_t dynamicScalesLocalIdx = 0;
            SyncFunc<AscendC::HardEvent::S_MTE2>();
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t k = 0; k < currTokensCount; k++) {
                int32_t eventId = (k & 1) ? EVENT_ID0 : EVENT_ID1;
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                DataCopy(xOutTensor_[eventId], currRankWindowGlobal[(currRankDataOffset + k) * axisHCommu_],
                         axisHCommu_);
                SetFlag<HardEvent::MTE2_MTE3>(eventId);
                if constexpr (DynamicQuant || WithScalePayload) {
                    PipeBarrier<PIPE_ALL>();
                    auto &&dynamicScalesLocal = xOutTensor_[eventId].template ReinterpretCast<float>();
                    dynamicScalesTensor.SetValue(dynamicScalesLocalIdx++, dynamicScalesLocal(scaleOffset));
                    PipeBarrier<PIPE_ALL>();
                }
                WaitFlag<HardEvent::MTE2_MTE3>(eventId);
                DataCopy(expandXOutGMTensor_[(currTokensOffset + k) * axisH_], xOutTensor_[eventId], axisH_);
                SetFlag<HardEvent::MTE3_MTE2>(eventId);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            currRankDataOffset += currTokensCount;
            PipeBarrier<PIPE_ALL>();
            if constexpr (DynamicQuant || WithScalePayload) {
                DataCopyExtParams scalesCopyParams{1U, static_cast<uint32_t>(dynamicScalesLocalIdx * sizeof(float)), 0U,
                                                   0U, 0U};
                DataCopyPad(dynamicScalesOutGMTensor_[currTokensOffset], dynamicScalesTensor, scalesCopyParams);
            }
        }
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::CleanUpFlags()
{
    if (aivId_ == 0) {
        constexpr uint32_t multiMachineMaxStatusSize = 512U; // 多机，固定清理512Bytes，可最大支持单卡127个Moe专家
        // 单机，根据单卡理论最大MoeExpertNum专家计算，尾部还存在一个flag
        uint32_t singleMachineMaxStatusSize = RoundUp((maxMoeExpertNum_ / worldSize_ + 1) * sizeof(uint32_t), UB_ALIGN);
        uint32_t statusSize = isSingleServer_ ? singleMachineMaxStatusSize : multiMachineMaxStatusSize;
        Duplicate<int32_t>(statusTensor_, 0, worldSize_ * statusSize / sizeof(int32_t));
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        uint32_t dstStrideU32 = dataSizePerRank_ - statusSize;
        auto copyStatusParams = DataCopyExtParams{static_cast<uint16_t>(worldSize_), statusSize, 0, dstStrideU32, 0};
        DataCopyPad(windowInstatusTensor_, statusTensor_, copyStatusParams);
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::CopyPerformanceInfo()
{
    if (unlikely(needPerformanceInfo_)) {
        AscendC::SetAtomicAdd<int32_t>();
        AscendC::DataCopyPad(performanceInfoI32GMTensor_, performanceInfoI32Tensor_,
                             {1, static_cast<uint32_t>(performanceInfoSize_ * sizeof(int64_t)), 0, 0, 0});
        AscendC::SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    }
}

template <TemplateMC2TypeA2Class>
__aicore__ inline void MoeDistributeDispatchA2<TemplateMC2TypeA2Func>::Process()
{
    if ASCEND_IS_AIV {
        AllocTensor();
        CalValidTokenCount();
        IndexSort();
        ReorderTokens();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_FULLMESH_DISPATCH_REORDERTOKEN);
        if (isSingleServer_) {
            SyncAll<true>();
            SingleServerSendToMoeExpert();
        } else {
            ConstructBatchWriteInfo();
            SyncAll<true>();
            SendToMoeExpert();
        }
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_FULLMESH_DISPATCH_SEND);
        WaitDispatch();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_FULLMESH_DISPATCH_RECV);
        CopyPerformanceInfo(); // 避免performanceInfoI32Tensor_被复用，提前搬出性能打点数据
        GetStatusCumSum();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_FULLMESH_DISPATCH_GET_STATUS_CUM_SUM);
        LocalWindowCopy();
        SyncAll<true>();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_FULLMESH_DISPATCH_LOCAL_WINDOW_COPY);
        CleanUpFlags();
        aDump_.RunPosRecord(Mc2A2Kernel::RUNPOS_END);
        hccl_.Finalize();
    }
}
} // namespace MoeDistributeDispatchA2Impl
#endif // MOE_DISTRIBUTE_DISPATCH_A2_H
