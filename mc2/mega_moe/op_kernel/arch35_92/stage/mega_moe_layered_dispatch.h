/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_DISPATCH_H
#define MEGA_MOE_LAYERED_DISPATCH_H

#include "adv_api/transpose/confusion_transpose.h"

#include "../mega_moe_vf/mega_moe_layered_vf.h"

// Internal member definitions; included by mega_moe_layered.h after the class declaration.
namespace MegaMoeImpl {

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchRounds(uint32_t fixedBefore,
                                                                                          uint32_t dispatchFixedCost)
{
    uint32_t dispatchAvailUb = 0U;
    if (COMM_WORK_UB_LIMIT > fixedBefore && COMM_WORK_UB_LIMIT - fixedBefore > dispatchFixedCost) {
        dispatchAvailUb = COMM_WORK_UB_LIMIT - fixedBefore - dispatchFixedCost;
    }
    uint32_t maxDispatchRoundSize = dispatchAvailUb * 8U / 65U;
    maxDispatchRoundSize = (maxDispatchRoundSize / 256U) * 256U;
    if (maxDispatchRoundSize == 0U) {
        maxDispatchRoundSize = 256U;
    }
    if (maskRouteCapacity_ <= static_cast<uint64_t>(maxDispatchRoundSize)) {
        dispatchRoundSendTotalNum_ = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(ALIGN_256)));
    } else {
        uint32_t minDispatchRounds = static_cast<uint32_t>(
            Ops::Base::CeilDiv(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(maxDispatchRoundSize)));
        uint32_t evenDispatchRoundSize = static_cast<uint32_t>(Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(minDispatchRounds)),
            static_cast<int64_t>(ALIGN_256)));
        dispatchRoundSendTotalNum_ =
            (evenDispatchRoundSize <= maxDispatchRoundSize) ? evenDispatchRoundSize : maxDispatchRoundSize;
    }
    if (dispatchRoundSendTotalNum_ == 0U) {
        dispatchRoundSendTotalNum_ = 256U;
    }
    dispatchTotalRounds_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(dispatchRoundSendTotalNum_)));
    dispatchRoundMaskAlignSize_ = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(dispatchRoundSendTotalNum_) / 8, static_cast<int64_t>(ALIGN_32)));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchIndexBuffers(
    uint32_t fixedBefore)
{
    uint32_t validTopkIndexTensorAddr = fixedBefore;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(dispatchRoundSendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = reinterpret_cast<__ubuf__ int32_t*>(validTopkIndexTensorAddr);
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(dispatchRoundSendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ = reinterpret_cast<__ubuf__ int32_t*>(topkIndexTensorAddr);
    // gatherMaskTensor_ 仅容纳单轮 mask 位。
    uint32_t gatherMaskTensorAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t gatherMaskTensorSize = dispatchRoundMaskAlignSize_;
    gatherMaskTensor_ = reinterpret_cast<__ubuf__ uint8_t*>(gatherMaskTensorAddr);
    gatherMaskInt32Tensor_ = reinterpret_cast<__ubuf__ uint32_t*>(gatherMaskTensorAddr);
    return gatherMaskTensorAddr + gatherMaskTensorSize;
}

/*
 * 绑定二级接收的复制缓冲、专家计数暂存区和后续 metadata 的 UB 起点。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchReceiveScratch(
    uint32_t copyBufferUbOffset, uint32_t copyBufferTotalBytes, uint32_t expertCountBufferBytes)
{
    const uint32_t copyBufferStrideBytes = mxQuantTokenScaleAlignBytes_;
    for (int32_t index = 0; index < DISPATCH_COPY_BUFFER_COUNT; ++index) {
        copyTmpTensors_[index] = reinterpret_cast<__ubuf__ ActivationType*>(
            copyBufferUbOffset + static_cast<uint32_t>(index) * copyBufferStrideBytes);
    }
    // 整表建表和 ExpertTokenNumCopyOut 共用每专家一个计数的暂存区。
    // 每专家一个 int32_t，缓冲字节数按 32B 对齐。
    const uint32_t expertCountUbOffset = copyBufferUbOffset + copyBufferTotalBytes;
    expertTokenNumsOutTensor_ = reinterpret_cast<__ubuf__ int32_t*>(expertCountUbOffset);
    // 后续 Dispatch 接收从该地址申请 metadata 临时区。
    ubBufferUsedAddr_ = expertCountUbOffset + expertCountBufferBytes;
}

// =================================================================================================
// DispatchBuffInit：Dispatch 接收、计数发布和 ExpertTokenNumCopyOut 使用的 buffer。
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // Dispatch 阶段 UB 布局：Hcomm 私有区、前缀和、分轮索引/掩码、
    // 本地搬运缓冲、专家计数输出和元数据区依次排列。
    // 整批 Dispatch 在 WQE 区上方预留持久的来源就绪缓存。
    comm_.BindBatchBuffer(LocalTensor<uint8_t>(TPosition::VECCALC, COMM_WORK_UB_LIMIT, LAYERED_HCOMM_BATCH_UB_BYTES),
                          LAYERED_HCOMM_BATCH_UB_BYTES);
    uint32_t cumsumInfoTensorAddr = ALIGN_512;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ = reinterpret_cast<__ubuf__ int32_t*>(cumsumInfoTensorAddr);
    // 分轮次参数计算：validTopkIndexTensor_ + topkIndexTensor_ + gatherMaskTensor_ 均为 round-sized
    // 专家计数直接从 GM 读取，不再加载完整 mask slot 到 UB。
    // round 张量: validTopkIndex(R*4) + topkIndex(R*4) + gatherMask(R/8) = 8R + R/8 = 65R/8
    // 预留本地搬运缓冲、专家计数输出和 256 行 metadata。
    // DispatchPrepare 使用的量化缓冲生命周期不重叠，不计入本阶段。
    uint32_t copyBufferTotalBytes = static_cast<uint32_t>(DISPATCH_COPY_BUFFER_COUNT) * mxQuantTokenScaleAlignBytes_;
    uint32_t expertCountBufferBytes =
        Ops::Base::CeilAlign(static_cast<int64_t>(moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    uint32_t dispatchFixedCost = copyBufferTotalBytes + expertCountBufferBytes + L1_TILE_M_256 * ALIGN_32;
    uint32_t fixedBefore = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    InitDispatchRounds(fixedBefore, dispatchFixedCost);
    uint32_t copyBufferUbOffset = InitDispatchIndexBuffers(fixedBefore);
    InitDispatchReceiveScratch(copyBufferUbOffset, copyBufferTotalBytes, expertCountBufferBytes);
    {
        LocalTensor<int32_t> values(TPosition::VECCALC, reinterpret_cast<uint64_t>(cumsumInfoTensor_),
                                    (cumsumInfoTensorSize / sizeof(int32_t)));
        Duplicate<int32_t>(values, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    }
    PipeBarrier<PIPE_ALL>();
}

// =================================================================================================
// DispatchPrepareBuffInit：一级 Dispatch 前置量化使用的双缓冲。
// SendMask/reset 和本阶段之间已有全核同步，Wave 内 DispatchBuffInit 会在前置阶段结束后复用同一 UB。
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchPrepareBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    uint32_t xInTensorSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t xInTensorAddr1 = ALIGN_512;
    uint32_t xInTensorAddr2 = xInTensorAddr1 + xInTensorSize;
    xInTensor1_ = reinterpret_cast<__ubuf__ bfloat16_t*>(xInTensorAddr1);
    xInTensor2_ = reinterpret_cast<__ubuf__ bfloat16_t*>(xInTensorAddr2);

    uint32_t mxTempTensorAddr = xInTensorAddr2 + xInTensorSize;
    mxTempTensor_ = reinterpret_cast<__ubuf__ uint16_t*>(mxTempTensorAddr);

    uint32_t xOutTensorSize = mxQuantTokenScaleAlignBytes_;
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    xOutTensor1_ = reinterpret_cast<__ubuf__ ActivationType*>(xOutTensorAddr1);
    xOutTensor2_ = reinterpret_cast<__ubuf__ ActivationType*>(xOutTensorAddr2);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitResetBatch()
{
    const auto& workspace = params_.workspace.layout;
    // 工作区统一管理连续排布的 Activation、Dispatch、计数和 Combine 标志。
    uint64_t totalFlagInt32 = (workspace.reset.flags.sizeBytes / SIZE_INT_32);
    int64_t tokenGroupResetSize = static_cast<int64_t>(moeExpertPerRank_) * blockAivNum_ * INT_CACHELINE;
    totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > tokenGroupResetSize) ?
                         static_cast<int64_t>(totalFlagInt32) :
                         tokenGroupResetSize;
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchElementCount_ = resetNumPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                  static_cast<int32_t>(resetNumPerCore) :
                                  DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount_), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);
    resetBatchElementCount_ = resetTensorSize / sizeof(int32_t);
    return resetTensorSize;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitSendMaskRounds(uint32_t resetTensorSize,
                                                                                          uint32_t countTensorSize)
{
    // 分轮次参数计算：UB 预算扣除固定开销后，求解最大 roundSendTotalNum_
    // round 张量: topkIds(R*4) + 2*mask(CeilAlign(R/8,32))
    //            = 4R + R/4  (R 为 256 对齐时 CeilAlign(R/8,32)=R/8)
    // 解: R <= available * 4 / 17，再 floor-align 到 256（保证 GM 写 32B 对齐）
    uint32_t fixedUbCosts = ALIGN_512 + resetTensorSize + countTensorSize;
    uint32_t availableUb = COMM_WORK_UB_LIMIT - fixedUbCosts;
    uint32_t maxRoundSize = availableUb * 4U / 17U;
    if (maxRoundSize == 0) {
        maxRoundSize = 256U;
    }
    maxRoundSize = (maxRoundSize / 256U) * 256U;
    if (static_cast<uint64_t>(sendTotalNum_) <= static_cast<uint64_t>(maxRoundSize)) {
        roundSendTotalNum_ = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(ALIGN_256)));
    } else {
        uint32_t minRounds = static_cast<uint32_t>(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(maxRoundSize)));
        uint32_t evenRoundSize = static_cast<uint32_t>(Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(minRounds)),
            static_cast<int64_t>(ALIGN_256)));
        roundSendTotalNum_ = (evenRoundSize <= maxRoundSize) ? evenRoundSize : maxRoundSize;
    }
    if (roundSendTotalNum_ == 0) {
        roundSendTotalNum_ = 256U;
    }
    roundCompareCount_ = roundSendTotalNum_;
    roundMaskAlignSize_ = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(roundCompareCount_) / 8, static_cast<int64_t>(ALIGN_32)));
    roundMaskSlotSize_ = roundMaskAlignSize_;
    totalRounds_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(roundSendTotalNum_)));
}

// ======================================================================================
// SendAndQuantBuffInit：ComputeExpertMasks 与 ResetState 的 UB 地址及容量准备
// --------------------------------------------------------------------------------------
//   大 bs 场景下 sendTotalNum 可能超出 UB 256KB 限制，因此按 roundSendTotalNum_ 分轮次
//   分配 UB 张量（topkIds / sendMask / count），每轮仅加载一部分 topkIds，
//   在 UB 中计算部分 mask 后拼写到 workspace 对应偏移，最终合并为完整 mask slot。
// ======================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendAndQuantBuffInit()
{
    // 输入准备与 DispatchBuffInit 生命期不重叠，可复用同一 UB 地址。
    // 减去通信 scratch、清零和专家 count 区后，按 4R + R/4 字节计算单轮容量。
    LocalTensor<uint8_t> hcommTensor(TPosition::VECCALC, 0, ALIGN_512);
    comm_.Init(hcommTensor, ALIGN_512);

    uint32_t resetTensorSize = InitResetBatch();
    // 每核负责的专家 count 独立保存在 32B 槽中，供 VF 累加、MTE3 搬出。
    const uint32_t countSlots = Ops::Base::CeilDiv(worldSize_ * moeExpertPerRank_, blockAivNum_);
    const uint32_t countTensorSize =
        Ops::Base::CeilAlign(countSlots * static_cast<uint32_t>(ALIGN_32), static_cast<uint32_t>(ALIGN_256));
    InitSendMaskRounds(resetTensorSize, countTensorSize);
    // Tensor用处：ComputeExpertMasks 每轮搬运本卡 topkIds 的一个子段；
    // Tensor大小：roundCompareCount_ 个 int32（256B 对齐）；
    uint32_t topkIdsTensorAddr = ALIGN_512;
    uint32_t topkIdsTensorSize = roundCompareCount_ * sizeof(int32_t);
    topkIdsTensor_ = reinterpret_cast<__ubuf__ int32_t*>(topkIdsTensorAddr);

    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = reinterpret_cast<__ubuf__ int32_t*>(resetTensorAddr);
    // Tensor用处：ComputeExpertMasks 每轮存储部分 mask 位；DOUBLE_BUFFER 双缓冲；
    // Tensor大小：roundMaskSlotSize_（仅本轮 mask 位图，不包含 count）；
    uint32_t sendMaskAddr = resetTensorAddr + resetTensorSize;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] =
            reinterpret_cast<__ubuf__ uint8_t*>(sendMaskAddr + static_cast<uint32_t>(index) * roundMaskSlotSize_);
    }

    sendMaskCounts_ =
        reinterpret_cast<__ubuf__ int32_t*>(sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * roundMaskSlotSize_);
}

// ===============================================================================================
// ResetState：准备零缓冲，清理本卡计算同步标志、通信队列及 ready 状态。
// ===============================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetState()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    const AivJobContext job{aivCoreIdx_, blockAivNum_};

    LocalTensor<int32_t> resetTensor(TPosition::VECCALC, reinterpret_cast<uint64_t>(resetTensor_),
                                     resetBatchElementCount_);
    Duplicate<int32_t>(resetTensor, 0, resetBatchElementCount_);
    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
    ResetWorkspaceRegion<1>(job, workspace.sync.activationReady.get_ptr(buffer),
                            (workspace.reset.flags.sizeBytes / SIZE_INT_32), resetBatchElementCount_, resetTensor_);

    // 同一 AIV owner 清零目标 Server 的队列 count 和 Combine 列表行头。
    const int64_t tokenListStride =
        CalcServerCombineTokenListStrideBytes(params_.tilingData->compute.numMaxTokensPerRank);
    for (uint32_t targetServer = aivCoreIdx_; targetServer < serverNum_; targetServer += blockAivNum_) {
        auto countPtr = reinterpret_cast<__gm__ int32_t*>(workspace.urma.relaySendQueue.get_ptr(buffer) +
                                                          DispatchRelayQueueServerOffset(targetServer));
        GmWrite(countPtr, int32_t(0));
        if (IsCombineServerDedupEnabled()) {
            auto listCount = reinterpret_cast<__gm__ int32_t*>(workspace.urma.serverCombine.tokenIds.get_ptr(buffer) +
                                                               targetServer * tokenListStride);
            GmWrite(listCount, int32_t(0));
        }
    }
    // CCL ready 区连续清零；普通模式的两个 Combine ready 区大小为 0。
    const auto readyBytes =
        peermem.dispatchFlag.sizeBytes + peermem.sourceReady.sizeBytes + peermem.contributionReady.sizeBytes;
    const auto readyRange = GetBalancedWorkRange(static_cast<int32_t>(readyBytes / sizeof(int32_t)), job);
    ResetWorkspaceRegion(readyRange, peermem.dispatchFlag.get_ptr(commBuffer), resetBatchElementCount_, resetTensor_);
    // Server Combine 的连续 token 状态区。
    if (IsCombineServerDedupEnabled()) {
        const auto completedCountRange =
            GetBalancedWorkRange(static_cast<int32_t>(workspace.reset.serverCombine.sizeBytes / sizeof(int32_t)), job);
        ResetWorkspaceRegion(completedCountRange, workspace.urma.serverCombine.completedCounts.get_ptr(buffer),
                             resetBatchElementCount_, resetTensor_);
    }
}

// ==================================================
// ExpertTokenNumCopyOut：输出本卡各 MoE 专家收到的 token 总数（不包含共享专家）。
// ==================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ExpertTokenNumCopyOut()
{
    // 归属核 Wave 在 Dispatch、GMM 和 Combine 之间复用 AIV1 UB，输出前必须恢复完整前缀和。
    CopyGmToUb(cumsumInfoTensor_, cumsumGm_, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)),
               GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
    PipeNotify<PIPE_MTE2, PIPE_S>(0);
    PipeWait<PIPE_MTE2, PIPE_S>(0);
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    expertTokenNumsOutTensor_[0] = cumsumInfoTensor_[lastRankIdx];
    for (int32_t expertIdx = 1; expertIdx < moeExpertPerRank_; expertIdx++) {
        int32_t cur = cumsumInfoTensor_[expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx];
        int32_t prev = cumsumInfoTensor_[(expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx];
        expertTokenNumsOutTensor_[expertIdx] = cur - prev;
    }
    PipeFence<PIPE_S, PIPE_MTE3, SYNC_EVENT_ID2>();

    CopyUbToGm(expertTokenNumsOut_, expertTokenNumsOutTensor_,
               static_cast<uint32_t>(moeExpertPerRank_ * sizeof(int32_t)));
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(0);
    PipeWait<PIPE_MTE3, PIPE_MTE2>(0);
}

// 发送区按 [目标卡][专家] 排列，本卡专家直接写入接收区的 [本源卡][专家]。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline GM_ADDR MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GetSendMaskSlot(uint32_t expertId) const
{
    const uint32_t targetRank = expertId / moeExpertPerRank_;
    GM_ADDR maskBase = targetRank == rankId_ ?
                           params_.communication.layout.maskRecv.get_ptr(params_.communication.base) :
                           params_.workspace.layout.urma.mask.get_ptr(params_.workspace.base);
    return maskBase + static_cast<uint64_t>(expertId) * maskSlotSize_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyExpertMaskCounts(
    const WorkRange& expertRange)
{
    const uint32_t expertEnd = expertRange.start + expertRange.count;
    const uint32_t localExpertBegin = rankId_ * moeExpertPerRank_;
    const uint32_t localExpertEnd = localExpertBegin + moeExpertPerRank_;
    uint32_t expertBegin = expertRange.start;
    // 发送暂存区按全局专家连续排列，仅进入、离开本卡专家范围时切换 GM 区域。
    while (expertBegin < expertEnd) {
        uint32_t copyEnd = expertEnd; // 默认搬完本核剩余 count。
        if (expertBegin < localExpertBegin && localExpertBegin < copyEnd) {
            copyEnd = localExpertBegin; // 搬到本卡专家区起点，下一段切换到接收区。
        } else if (expertBegin < localExpertEnd && localExpertEnd < copyEnd) {
            copyEnd = localExpertEnd; // 搬到本卡专家区终点，下一段切回发送暂存区。
        }
        const WorkRange copyRange{expertBegin, copyEnd - expertBegin};
        const uint64_t ubOffset = static_cast<uint64_t>(copyRange.start - expertRange.start) * ALIGN_32;
        auto* source = reinterpret_cast<__ubuf__ uint8_t*>(sendMaskCounts_) + ubOffset;
        auto* destination = reinterpret_cast<__gm__ uint8_t*>(GetSendMaskSlot(copyRange.start) + maskAlignSize_);
        const CopyRows copyParams{static_cast<uint16_t>(copyRange.count), sizeof(int32_t), maskSlotSize_, ALIGN_32};
        CopyUbToGm(destination, source, copyParams);
        expertBegin = copyEnd;
    }
}

/*
 * 全部 AIV 均分 [目标 rank × 本地专家]，各核写互不重叠的完整 mask 槽。
 * 每轮 topk 只搬入一次，复用于本核全部专家；跨轮 count 保存在独立 UB 槽中。
 * UB = 512B 通信 scratch + topk + reset 零缓冲 + 双 mask 缓冲 + count 槽。
 * InitSendMaskRounds 将 count 槽一并计入预算，所有缓冲均位于 COMM_WORK_UB_LIMIT 以下。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ComputeExpertMasks(const WorkRange& expertRange)
{
    constexpr uint32_t COUNT_STRIDE = ALIGN_32 / sizeof(int32_t);
    LocalTensor<int32_t> counts(TPosition::VECCALC, reinterpret_cast<uint64_t>(sendMaskCounts_),
                                expertRange.count * COUNT_STRIDE);
    Duplicate<int32_t>(counts, 0, expertRange.count * COUNT_STRIDE);

    for (uint32_t roundIdx = 0U; roundIdx < totalRounds_; ++roundIdx) {
        const uint64_t roundStart = static_cast<uint64_t>(roundIdx) * roundSendTotalNum_;
        const uint32_t roundLen = static_cast<uint32_t>(
            sendTotalNum_ - roundStart < roundSendTotalNum_ ? sendTotalNum_ - roundStart : roundSendTotalNum_);
        const uint64_t byteOffset = roundStart / 8U;
        const uint32_t alignedMaskBytes =
            Ops::Base::CeilAlign(Ops::Base::CeilDiv(roundLen, 8U), static_cast<uint32_t>(ALIGN_32));
        const uint32_t writeBytes = static_cast<uint32_t>(
            maskAlignSize_ - byteOffset < alignedMaskBytes ? maskAlignSize_ - byteOffset : alignedMaskBytes);

        // 下一轮覆盖 topk 前，等待前轮 VF 读取完成；VF 自身屏蔽无效尾部，无需填充输入。
        PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID1>();
        CopyGmToUb(topkIdsTensor_, reinterpret_cast<__gm__ int32_t*>(params_.expertIdxGmAddr) + roundStart,
                   roundLen * sizeof(int32_t));
        PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID1>();

        for (uint32_t expertOffset = 0U; expertOffset < expertRange.count; ++expertOffset) {
            const uint32_t expertId = expertRange.start + expertOffset;
            const uint32_t bufferIdx = expertOffset % DOUBLE_BUFFER;
            // mutex 0/1 分别保护两块 mask 缓冲，首次可直接获取，复用时由硬件等待前次搬出。
            asc_lock(PIPE_V, bufferIdx);
            CallVf<ComputeExpertMaskVF>(topkIdsTensor_, sendMaskTensor_[bufferIdx],
                                        sendMaskCounts_ + expertOffset * COUNT_STRIDE, expertId, roundLen);
            asc_unlock(PIPE_V, bufferIdx);
            asc_lock(PIPE_MTE3, bufferIdx);
            CopyUbToGm(reinterpret_cast<__gm__ uint8_t*>(GetSendMaskSlot(expertId) + byteOffset),
                       sendMaskTensor_[bufferIdx], writeBytes);
            asc_unlock(PIPE_MTE3, bufferIdx);
        }
    }
    // count 由 VF 写入 UB，等待 V 完成后搬出；源槽保持到本卡交接同步结束。
    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID3>();
    CopyExpertMaskCounts(expertRange);
}

// 固定 channel owner 发送其他计算核准备的完整 [mask | count] 槽，锁继续由 Sync1 释放。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendExpertMasks()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const uint64_t rankMaskBytes = static_cast<uint64_t>(moeExpertPerRank_) * maskSlotSize_;
    for (uint32_t targetRank = aivCoreIdx_; targetRank < worldSize_; targetRank += blockAivNum_) {
        if (targetRank == rankId_) {
            continue;
        }
        Channel channel = comm_.GetChannel(targetRank);
        comm_.Lock(channel);
        const uint64_t srcOffset = static_cast<uint64_t>(targetRank) * rankMaskBytes;
        const uint64_t dstOffset = peermem.maskRecv.offsetBytes + static_cast<uint64_t>(rankId_) * rankMaskBytes;
        comm_.WriteNbiChunked(channel, GetRankWinAddrWithOffset(targetRank, dstOffset),
                              workspace.urma.mask.get_ptr(buffer) + srcOffset, rankMaskBytes);
    }
    // mask 发送完成的等待统一放在首次跨卡同步前，与中间的本卡准备工作重叠。
}

// ======================================================================
// LoadTopkWeightsToUb：prefetch 或 Server Combine 携带权重，否则仅做 MTE2_V 同步
// ======================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTopkWeightsToUb(
    __ubuf__ ActivationType* xOutTensor, int32_t currentOffset, int32_t index, TEventID event)
{
    if (RecordHasWeights()) {
        QuantProcessConfig config{mxQuantTokenAlignBytes_, mxQuantScaleAlignBytes_, mxQuantTokenScaleAlignBytes_,
                                  mxQuantScaleNumAlignPerToken_};
        QuantProcessScratch<ActivationType> scratch{};
        scratch.mxTempTensor = mxTempTensor_;
        uint32_t tokenIndex = currentOffset + index;
        GM_ADDR tokenTopkWeightsAddr =
            params_.probsGmAddr + static_cast<uint64_t>(tokenIndex) * topK_ * sizeof(TopkWeightsType);
        PrefetchTopkWeights<TopkWeightsType>(tokenTopkWeightsAddr, topK_, config, scratch, xOutTensor, event);
    } else {
        // Without weight prefetch, this event still waits for the input token copy.
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchRelayQueueServerOffset(
    uint32_t targetServer) const
{
    return static_cast<uint64_t>(targetServer) * dispatchRelayQueueBytesPerServer_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RelayTokenOffset(uint32_t sourceServer,
                                                                                            uint32_t tokenId) const
{
    return (static_cast<uint64_t>(sourceServer) *
                static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank) +
            tokenId) *
           static_cast<uint64_t>(relayRecordBytes_);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RelayFlagOffset(uint32_t sourceServer,
                                                                                           uint32_t tokenId) const
{
    if (IsCombineServerDedupEnabled()) {
        return sourceServer * CalcServerCombineReadyRowBytes(params_.tilingData->compute.numMaxTokensPerRank) +
               static_cast<uint64_t>(tokenId) * sizeof(uint64_t);
    }
    return (static_cast<uint64_t>(sourceServer) *
                static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank) +
            tokenId) *
           sizeof(uint64_t);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::QuantizeTokenInUb(
    __ubuf__ bfloat16_t* input, __ubuf__ ActivationType* output, __ubuf__ uint16_t* scratch)
{
    __ubuf__ bfloat16_t* srcAddr = reinterpret_cast<__ubuf__ bfloat16_t*>(input);
    __ubuf__ uint16_t* maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(scratch);
    __ubuf__ uint16_t* halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(
        (scratch + (Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32)))));
    __ubuf__ int8_t* outDataAddr = reinterpret_cast<__ubuf__ int8_t*>(output);
    __ubuf__ uint16_t* mxScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>((output + (mxQuantTokenAlignBytes_)));
    Quant::ComputeMaxExp(srcAddr, maxExpAddr, k_);
    Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, mxQuantScaleNumAlignPerToken_);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, k_);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, k_);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::QuantizeLocalTokensToRelay()
{
    const auto& peermem = params_.communication.layout;
    WorkRange tokenRange = GetBalancedWorkRange(m_, {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_});
    uint32_t tokenNumInCore = tokenRange.count;
    uint32_t tokenStart = tokenRange.start;
    if (tokenNumInCore == 0U) {
        return;
    }

    __gm__ bfloat16_t* inputGm = nullptr;
    __gm__ uint8_t* quantOutputGm = nullptr;
    inputGm = reinterpret_cast<__gm__ bfloat16_t*>(params_.aGmAddr) + static_cast<uint64_t>(tokenStart) * k_;

    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
    for (uint32_t index = 0U; index < tokenNumInCore; ++index) {
        bool useFirstBuffer = (index & 1U) == 0U;
        TEventID event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ bfloat16_t* xInTensor = useFirstBuffer ? xInTensor1_ : xInTensor2_;
        __ubuf__ ActivationType* xOutTensor = useFirstBuffer ? xOutTensor1_ : xOutTensor2_;
        uint32_t tokenIdx = tokenStart + index;

        PipeWait<PIPE_MTE3, PIPE_MTE2>(event);
        CopyGmToUb(xInTensor, (inputGm + (static_cast<uint64_t>(index) * k_)),
                   static_cast<uint16_t>(k_ * sizeof(bfloat16_t)), GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
        LoadTopkWeightsToUb(xOutTensor, tokenStart, index, event);

        QuantizeTokenInUb(xInTensor, xOutTensor, mxTempTensor_);

        PipeNotify<PIPE_V, PIPE_MTE3>(event);
        PipeWait<PIPE_V, PIPE_MTE3>(event);
        uint64_t relayOffset = RelayTokenOffset(serverId_, tokenIdx);
        GM_ADDR recordAddr = GetRankWinAddrWithOffset(rankId_, peermem.dispatchReceive.offsetBytes) + relayOffset;
        quantOutputGm = reinterpret_cast<__gm__ uint8_t*>(recordAddr);
        __ubuf__ uint8_t* xOutBytesTensor = reinterpret_cast<__ubuf__ uint8_t*>(xOutTensor);
        CopyUbToGm(quantOutputGm, xOutBytesTensor, mxQuantTokenScaleAlignBytes_);
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(event);
    }
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::EnqueueDispatchRelayToken(uint32_t targetServer,
                                                                                                 uint32_t tokenIdx)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    GM_ADDR queueBase = workspace.urma.relaySendQueue.get_ptr(buffer) + DispatchRelayQueueServerOffset(targetServer);
    __gm__ int32_t* countPtr = reinterpret_cast<__gm__ int32_t*>(queueBase);
    int32_t slotIdx = AtomicAdd(countPtr, int32_t(1));
    uint64_t metaOffset = ALIGN_32 + static_cast<uint64_t>(slotIdx) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(queueBase + metaOffset);
    GmWrite(metaPtr, static_cast<int32_t>(tokenIdx));
}

template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildDispatchRelayCandidates(
    const LocalTensor<int32_t>& topkIds, const LocalTensor<uint32_t>& masks, uint32_t tokenCount,
    const WorkRange& serverRange)
{
    CallVf<BuildDispatchRelayCandidatesByServerVF<CombineServerDedup>>(
        reinterpret_cast<__ubuf__ int32_t*>(topkIds.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(masks.GetPhyAddr()), tokenCount, topK_,
        static_cast<int32_t>(worldSize_ * moeExpertPerRank_),
        static_cast<int32_t>(moeExpertPerRank_ * rankNumPerServer_), static_cast<int32_t>(serverId_), serverRange.start,
        serverRange.count);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::EnqueueDispatchRelayCandidates(
    const LocalTensor<uint32_t>& masks, const WorkRange& tokenRange, const WorkRange& serverRange)
{
    constexpr uint32_t ROW_ELEMENTS = AscendC::GetVecLen() / sizeof(int32_t);
    for (uint32_t token = 0; token < tokenRange.count; ++token) {
        for (uint32_t index = 0; index < serverRange.count; ++index) {
            uint32_t bitmapValue = masks.GetValue(token * ROW_ELEMENTS + index);
            if (bitmapValue == 0U) {
                continue;
            }
            EnqueueDispatchRelayToken(serverRange.start + index, tokenRange.start + token);
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::TransposeDispatchRelayBitmaps(
    const LocalTensor<uint32_t>& masks, const LocalTensor<uint32_t>& transposedMasks, uint32_t tokenCount)
{
    constexpr uint32_t ROW_ELEMENTS = AscendC::GetVecLen() / sizeof(uint32_t);
    constexpr uint32_t BLOCK_ELEMENTS = ALIGN_32 / sizeof(uint32_t);
    const uint32_t alignedCount = Ops::Base::CeilAlign(tokenCount, BLOCK_ELEMENTS);
    // 场景 13：[1, alignedCount, 64] -> [1, 64, alignedCount]，每个 Server 行首按 32B 对齐。
    ConfusionTransposeTiling tiling{};
    tiling.param0 = 1U;
    tiling.param1 = alignedCount;
    tiling.param2 = ROW_ELEMENTS;
    // 950 的 021 路径额外 scratch 为 0；传有效 UB 视图，避免无 scratch 重载申请栈缓冲。
    Transpose<uint32_t>(transposedMasks, masks, transposedMasks.template ReinterpretCast<uint8_t>(),
                        TransposeType::TRANSPOSE_ND2ND_021, tiling);
    return alignedCount;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishDispatchRelayBitmaps(
    const LocalTensor<uint32_t>& masks, const WorkRange& tokenRange, const WorkRange& serverRange, uint32_t rowElements)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    // GM 保留 8B ready 槽；Compact 只写低 32 位，高位由入口整区清零保证。
    const DataCopyExtParams copyParams{static_cast<uint16_t>(tokenRange.count), sizeof(uint32_t), 0U, sizeof(uint32_t),
                                       0U};
    __gm__ uint32_t* readyGlobal = nullptr;
    for (uint32_t index = 0; index < serverRange.count; ++index) {
        const uint32_t server = serverRange.start + index;
        readyGlobal = reinterpret_cast<__gm__ uint32_t*>(peermem.sourceReady.get_ptr(commBuffer) +
                                                         RelayFlagOffset(server, tokenRange.start));
        CopyUbToGm(readyGlobal, reinterpret_cast<__ubuf__ uint32_t*>((masks[index * rowElements]).GetPhyAddr()),
                   CopyRows{static_cast<uint16_t>(tokenRange.count), sizeof(uint32_t),
                            (sizeof(uint32_t)) + (sizeof(uint32_t)), sizeof(uint32_t)});
        if (server == serverId_) {
            readyGlobal = reinterpret_cast<__gm__ uint32_t*>(peermem.dispatchFlag.get_ptr(commBuffer) +
                                                             RelayFlagOffset(server, tokenRange.start));
            CopyUbToGm(readyGlobal, reinterpret_cast<__ubuf__ uint32_t*>((masks[index * rowElements]).GetPhyAddr()),
                       CopyRows{static_cast<uint16_t>(tokenRange.count), sizeof(uint32_t),
                                (sizeof(uint32_t)) + (sizeof(uint32_t)), sizeof(uint32_t)});
        }
    }
}

/*
 * 当前 UB：[0,512B) 通信 scratch；[512B,184KiB) 建表；[184KiB,248KiB) 保留给 WQE。
 * 普通模式 topkIds/masks 两块，每批 367 token；去重模式增加独立转置区，每批 240 token。
 * 每块每 token 预留 256B；去重三块共 180KiB，转置后按 Server 行批量 MTE 发布。
 * 输入每批只搬一次；每段位图完成标量消费及 MTE 发布后，复用同一输出缓冲。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareDispatchRelayQueues()
{
    constexpr uint32_t ROW_ELEMENTS = AscendC::GetVecLen() / sizeof(int32_t);
    constexpr uint32_t ROW_BYTES = ROW_ELEMENTS * sizeof(int32_t);
    constexpr uint32_t BLOCK_ELEMENTS = ALIGN_32 / sizeof(uint32_t);
    constexpr uint32_t NORMAL_BATCH_TOKENS = (COMM_WORK_UB_LIMIT - ALIGN_512) / (2U * ROW_BYTES);
    constexpr uint32_t DEDUP_BATCH_TOKENS =
        (COMM_WORK_UB_LIMIT - ALIGN_512) / (3U * ROW_BYTES) / BLOCK_ELEMENTS * BLOCK_ELEMENTS;
    static_assert(NORMAL_BATCH_TOKENS > 0U && NORMAL_BATCH_TOKENS <= UINT16_MAX);
    static_assert(DEDUP_BATCH_TOKENS > 0U && DEDUP_BATCH_TOKENS <= UINT16_MAX);
    const bool combineServerDedup = IsCombineServerDedupEnabled();
    const uint32_t batchTokens = combineServerDedup ? DEDUP_BATCH_TOKENS : NORMAL_BATCH_TOKENS;
    const uint32_t bufferBytes = batchTokens * ROW_BYTES;
    LocalTensor<int32_t> topkIds(TPosition::VECCALC, ALIGN_512, batchTokens * ROW_ELEMENTS);
    LocalTensor<uint32_t> masks(TPosition::VECCALC, ALIGN_512 + bufferBytes, batchTokens * ROW_ELEMENTS);
    LocalTensor<uint32_t> transposedMasks;
    if (combineServerDedup) {
        transposedMasks =
            LocalTensor<uint32_t>(TPosition::VECCALC, ALIGN_512 + 2U * bufferBytes, batchTokens * ROW_ELEMENTS);
    }
    // 全部物理 AIV 处理互不重叠的 token 区间。
    const auto tokenRange = GetBalancedWorkRange(m_, {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_});
    if (combineServerDedup) {
        BuildDispatchRelayQueues<true>(topkIds, masks, transposedMasks, tokenRange, batchTokens);
    } else {
        BuildDispatchRelayQueues<false>(topkIds, masks, transposedMasks, tokenRange, batchTokens);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildDispatchRelayQueues(
    const LocalTensor<int32_t>& topkIds, const LocalTensor<uint32_t>& masks,
    const LocalTensor<uint32_t>& transposedMasks, const WorkRange& tokenRange, uint32_t batchTokens)
{
    constexpr uint32_t ROW_ELEMENTS = AscendC::GetVecLen() / sizeof(int32_t);
    __gm__ int32_t* topkIdsGlobal = nullptr;
    topkIdsGlobal = reinterpret_cast<__gm__ int32_t*>(params_.expertIdxGmAddr);
    uint32_t rowBlocks = (topK_ * sizeof(int32_t) + ALIGN_32 - 1U) / ALIGN_32;
    // 预置缓冲可用事件，循环内固定消费并补回。
    PipeNotify<PIPE_S, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
    PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
    if constexpr (CombineServerDedup) {
        PipeNotify<PIPE_MTE3, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
    }
    for (uint32_t tokenOffset = 0; tokenOffset < tokenRange.count; tokenOffset += batchTokens) {
        const WorkRange batchRange{tokenRange.start + tokenOffset, (tokenRange.count - tokenOffset < batchTokens) ?
                                                                       tokenRange.count - tokenOffset :
                                                                       batchTokens};
        // 覆盖 topkIds 前，等待输入缓冲可复用。
        PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
        // GM 行紧凑，UB 每行补齐到 256B；dstStride 以 32B 块计。
        CopyGmToUb(
            reinterpret_cast<__ubuf__ int32_t*>((topkIds).GetPhyAddr()),
            (topkIdsGlobal + (static_cast<uint64_t>(batchRange.start) * topK_)),
            CopyRows{static_cast<uint16_t>(batchRange.count), static_cast<uint32_t>(topK_ * sizeof(int32_t)),
                     static_cast<uint32_t>(topK_ * sizeof(int32_t)),
                     static_cast<uint32_t>(
                         (Ops::Base::CeilAlign(static_cast<uint32_t>(static_cast<uint32_t>(topK_ * sizeof(int32_t))),
                                               static_cast<uint32_t>(ALIGN_32))) +
                         static_cast<uint32_t>(
                             static_cast<uint32_t>(ROW_ELEMENTS * sizeof(int32_t) / ALIGN_32 - rowBlocks)) *
                             ALIGN_32)});
        PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
        for (uint32_t serverStart = 0; serverStart < serverNum_; serverStart += ROW_ELEMENTS) {
            const WorkRange serverRange{
                serverStart, (serverNum_ - serverStart < ROW_ELEMENTS) ? serverNum_ - serverStart : ROW_ELEMENTS};
            // 覆盖 masks 前，等待标量读取结束。
            PipeWait<PIPE_S, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
            BuildDispatchRelayCandidates<CombineServerDedup>(topkIds, masks, batchRange.count, serverRange);
            PipeNotify<PIPE_V, PIPE_S>(static_cast<event_t>(SYNC_EVENT_ID2));
            if constexpr (CombineServerDedup) {
                // 当前 VF 可与上段搬出重叠；仅转置覆盖输出前等待上段 MTE3 读完。
                PipeWait<PIPE_MTE3, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
                // 950 的 V 流水依赖由硬件保证，转置直接读取当前 VF 的输出。
                const uint32_t rowElements = TransposeDispatchRelayBitmaps(masks, transposedMasks, batchRange.count);
                PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
                PublishDispatchRelayBitmaps(transposedMasks, batchRange, serverRange, rowElements);
                PipeNotify<PIPE_MTE3, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
            }
            PipeWait<PIPE_V, PIPE_S>(static_cast<event_t>(SYNC_EVENT_ID2));
            EnqueueDispatchRelayCandidates(masks, batchRange, serverRange);
            PipeNotify<PIPE_S, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
        }
        PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
    }
    // 回收最后一轮可用事件；空 token 区间也回收入口预置事件。
    PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
    PipeWait<PIPE_S, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
    if constexpr (CombineServerDedup) {
        PipeWait<PIPE_MTE3, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
        // 上面的 Wait 只阻塞 V；Scalar 返回前仍需确认全部 GM ready 发布完成。
        PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishLocalDispatchRelayCount()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    const uint32_t capacity = static_cast<uint32_t>(params_.tilingData->compute.numMaxTokensPerRank);
    auto countPtr = reinterpret_cast<__gm__ int32_t*>(workspace.urma.relaySendQueue.get_ptr(buffer) +
                                                      DispatchRelayQueueServerOffset(serverId_));
    const uint64_t encodedCount = static_cast<uint64_t>(GmRead(countPtr)) + 1U;
    const uint64_t countOffset = RelayFlagOffset(serverId_, capacity);
    GmWrite(reinterpret_cast<__gm__ uint64_t*>(peermem.sourceReady.get_ptr(commBuffer) + countOffset), encodedCount);
    GmWrite(reinterpret_cast<__gm__ uint64_t*>(peermem.dispatchFlag.get_ptr(commBuffer) + countOffset), encodedCount);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchBulkReadyFlagOffset(
    uint32_t sourceServer) const
{
    const uint32_t readySlot =
        IsCombineServerDedupEnabled() ? static_cast<uint32_t>(params_.tilingData->compute.numMaxTokensPerRank) : 0U;
    return RelayFlagOffset(sourceServer, readySlot);
}

/*
 * 准备目标 Server 独立的 GM 通知源；普通模式写入 1，Server Combine 模式写入有效数量加 1。
 * 通知 DMA 读取期间不再改写该地址。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline GM_ADDR MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareDispatchReadySource(
    uint32_t targetServer)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    GM_ADDR queueHeader = workspace.urma.relaySendQueue.get_ptr(buffer) + DispatchRelayQueueServerOffset(targetServer);
    GM_ADDR readySource = queueHeader;
    if (IsCombineServerDedupEnabled()) {
        // Combine 仍按有效路由 token 数建表，不能把全量 m_ 或布尔 1 作为尾部 count。
        const uint64_t encodedCount =
            static_cast<uint64_t>(GmRead(reinterpret_cast<__gm__ int32_t*>(queueHeader))) + 1U;
        readySource = peermem.sourceReady.get_ptr(commBuffer) + DispatchBulkReadyFlagOffset(targetServer);
        GmWrite(reinterpret_cast<__gm__ uint64_t*>(readySource), encodedCount);
    } else {
        // 普通模式不再构造队列，复用每个目标独立的 32B header 保存稳定通知源。
        GmWrite(reinterpret_cast<__gm__ uint64_t*>(readySource), uint64_t(1));
    }
    // 通知 DMA 直接读 GM；在途期间 sourceReady 尾槽和普通模式 header 都不再改写。
    GmDataBarrier();
    return readySource;
}

/*
 * 一级 Dispatch：向单个远端 Server 的同卡号中继发送全量记录和可选 Combine 位图。
 * 最后提交来源就绪通知，整批完成等待留给后续依赖或出口。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendDispatchToRemoteServer(uint32_t targetServer,
                                                                                                  uint32_t relayRank)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    GM_ADDR readySource = PrepareDispatchReadySource(targetServer);
    Channel channel = comm_.GetChannel(relayRank);
    BatchHandle batchHandle = comm_.MakeBatchHandle(channel, GetRankWinAddrWithOffset(relayRank, 0));
    const uint64_t relayOffset = RelayTokenOffset(serverId_, 0U);
    GM_ADDR dataSource = GetRankWinAddrWithOffset(rankId_, peermem.dispatchReceive.offsetBytes) + relayOffset;
    GM_ADDR dataTarget = GetRankWinAddrWithOffset(relayRank, peermem.dispatchReceive.offsetBytes) + relayOffset;
    const uint64_t dataBytes = static_cast<uint64_t>(m_) * static_cast<uint64_t>(relayRecordBytes_);
    uint32_t pendingWqes = comm_.WriteRelaxedNbiChunked(batchHandle, dataTarget, dataSource, dataBytes);
    if (IsCombineServerDedupEnabled()) {
        // 未命中和实际 m_ 以外的位图槽由入口 reset 保证为零；只发位图，不混入尾部 count。
        const uint64_t capacity = static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank);
        GM_ADDR bitmapSource = peermem.sourceReady.get_ptr(commBuffer) + RelayFlagOffset(targetServer, 0U);
        GM_ADDR bitmapTarget =
            GetRankWinAddrWithOffset(relayRank, peermem.dispatchFlag.offsetBytes) + RelayFlagOffset(serverId_, 0U);
        const uint64_t bitmapBytes = capacity * sizeof(uint64_t);
        pendingWqes = comm_.WriteRelaxedNbiChunked(batchHandle, bitmapTarget, bitmapSource, bitmapBytes, pendingWqes);
    }
    if (pendingWqes != 0U) {
        comm_.BatchCommit(batchHandle);
    }
    GM_ADDR readyTarget =
        GetRankWinAddrWithOffset(relayRank, peermem.dispatchFlag.offsetBytes) + DispatchBulkReadyFlagOffset(serverId_);
    // 尾通知生成 CQE，使后续依赖或 Wave 出口的 Drain 能收口全部前序无 CQE 写入。
    comm_.WriteStrongNbiWithCqe(batchHandle, readyTarget, readySource, sizeof(uint64_t));
    comm_.BatchCommit(batchHandle);
}

/*
 * 一级 Dispatch 入口：由固定归属核向其负责的远端 Server 提交整批发送。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::StartInterServerDispatch()
{
    if (IsCombineServerDedupEnabled() && IsChannelOwner(rankId_)) {
        PublishLocalDispatchRelayCount();
    }
    for (uint32_t targetServer = 0U; targetServer < serverNum_; ++targetServer) {
        const uint32_t relayRank = targetServer * rankNumPerServer_ + rankIdInServer_;
        if (targetServer == serverId_ || relayRank >= worldSize_ || !IsChannelOwner(relayRank)) {
            continue;
        }
        SendDispatchToRemoteServer(targetServer, relayRank);
    }
}

/*
 * 等待远端 Server 的整批数据到达指定 relay，并缓存确认结果；调用方只传入远端来源。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::EnsureDispatchSourceReady(uint32_t relayRank,
                                                                                                 uint32_t srcServer)
{
    const uint32_t srcRankInServer = relayRank % rankNumPerServer_;
    const uint32_t srcRank = srcServer * rankNumPerServer_ + srcRankInServer;
    // 每个 uint32_t 缓存 32 个来源 rank；商确定字下标，余数确定该字内的 bit。
    const uint32_t cacheWord = srcRank / 32U;
    const uint32_t cacheBit = uint32_t(1) << (srcRank % 32U);
    // bit 为 1 表示已确认该来源整批到达；为 0 仅表示尚未确认。
    const uint32_t readySources = dispatchSourceReadyBitsUb_[cacheWord];
    if ((readySources & cacheBit) != 0U) {
        return;
    }

    const auto& peermem = params_.communication.layout;
    const uint64_t readyOffset = DispatchBulkReadyFlagOffset(srcServer);
    GM_ADDR readyAddr = GetRankWinAddrWithOffset(relayRank, peermem.dispatchFlag.offsetBytes) + readyOffset;
    if (relayRank == rankId_) {
        WaitUntilGmFlagIsNonZero(reinterpret_cast<__gm__ uint64_t*>(readyAddr));
    } else {
        const auto& workspace = params_.workspace.layout;
        const auto& buffer = params_.workspace.base;
        GM_ADDR snapshotAddr = workspace.urma.remoteReadySnapshot.get_ptr(buffer) +
                               static_cast<uint64_t>(blockIdx_) * dispatchRelayFlagSnapshotBytesPerBlock_;
        Channel channel = comm_.GetChannel(relayRank);
        // 只使用独立 GM 快照，不覆盖尚待消费的 Gather 结果或批次 metadata。
        // Drain 确认本次 DMA 已完成；GmRead 绕过 Scalar DCache 读取最新标志。
        do {
            comm_.ReadNbi(channel, snapshotAddr, readyAddr, sizeof(uint64_t));
            comm_.Drain(channel);
        } while (GmRead(reinterpret_cast<__gm__ uint64_t*>(snapshotAddr)) == 0U);
    }
    // 仅在等待成功后置位，后续专家和 wave 复用确认结果。
    dispatchSourceReadyBitsUb_[cacheWord] = readySources | cacheBit;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTokenFromLocalRelay(uint32_t srcServer,
                                                                                               uint32_t tokenIndex,
                                                                                               int32_t bufferIdx,
                                                                                               uint32_t copyInNum)
{
    const auto& peermem = params_.communication.layout;
    uint64_t remoteCopyOffset = RelayTokenOffset(srcServer, tokenIndex);
    GM_ADDR localRecordAddr = GetRankWinAddrWithOffset(rankId_, peermem.dispatchReceive.offsetBytes) + remoteCopyOffset;
    __gm__ ActivationType* relayRecordsGm = reinterpret_cast<__gm__ ActivationType*>(localRecordAddr);
    CopyGmToUb(copyTmpTensors_[bufferIdx], relayRecordsGm, (copyInNum) * sizeof(ActivationType));
    PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID1>();
}

// 一次构建完整专家计数表并备份前缀和，同时统计计算 wave 数。
// 全部写回完成后仅发布本 block 的整表就绪标志。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareExpertTokenCountTable()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    gmmWaveCount_ = 0U;
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1U || moeExpertPerRank_ == 0U) {
        return;
    }

    // GM 按 [源卡][专家] 保存 count，先紧凑搬入，再重排成前缀和使用的 [专家][源卡]。
    // 复用接收 metadata 区（256 * 32B）；最多 2048 个 int32 count，不增加 UB 预算。
    auto* rankCounts = reinterpret_cast<__ubuf__ int32_t*>(ubBufferUsedAddr_);
    // maskRecv 基址 512B 对齐、maskAlignSize_ 32B 对齐；blockCount <= 2048，满足 Compact Copy。
    auto* countGm = reinterpret_cast<__gm__ int32_t*>(peermem.maskRecv.get_ptr(commBuffer) + maskAlignSize_);
    PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID2>();
    CopyGmToUb(rankCounts, countGm,
               CopyRows{static_cast<uint16_t>(worldSize_ * moeExpertPerRank_), sizeof(int32_t), maskSlotSize_,
                        sizeof(int32_t)},
               GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
    PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();

    const uint32_t countElements = worldSize_ * moeExpertPerRank_;
    LocalTensor<int32_t> rankCountsTensor(TPosition::VECCALC, reinterpret_cast<uint64_t>(rankCounts), countElements);
    LocalTensor<int32_t> expertCountsTensor(TPosition::VECCALC, reinterpret_cast<uint64_t>(cumsumInfoTensor_),
                                            countElements);
    ConfusionTransposeTiling tiling{};
    tiling.param0 = 1U;
    tiling.param1 = worldSize_;
    tiling.param2 = moeExpertPerRank_;
    // 950 的 021 路径无需额外 scratch，源和目标使用两块独立 UB。
    Transpose<int32_t>(expertCountsTensor, rankCountsTensor, expertCountsTensor.template ReinterpretCast<uint8_t>(),
                       TransposeType::TRANSPOSE_ND2ND_021, tiling);

    // 复用 MTE 的 count 表计算：前缀和 + 专家差分 + maxOutputSize clamp。
    // URMA mask count 高 8 位为 0，0x00FFFFFF 的 epoch mask 在当前场景下是无操作。
    ComputeExpertCountTables(cumsumInfoTensor_, expertTokenNumsOutTensor_, moeExpertPerRank_, worldSize_,
                             static_cast<uint32_t>(maxOutputSize_));
    PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID2>();

    const BlockWorkspaceContext countWorkspace{blockIdx_, aicNum_};
    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, moeExpertPerRank_, 0U, true);
    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
    CopyUbToGm(expertRecvCountsGm_ + countOffset, expertTokenNumsOutTensor_,
               moeExpertPerRank_ * static_cast<uint32_t>(sizeof(int32_t)));
    CopyUbToGm(cumsumGm_, cumsumInfoTensor_, worldSize_ * moeExpertPerRank_ * static_cast<uint32_t>(sizeof(int32_t)));
    // 两张表写回完成后才发布就绪标志；预取权重由后续接收和消费位置的同步独立保护。
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();

    __gm__ int32_t* tokenCountReadyFlag =
        reinterpret_cast<__gm__ int32_t*>(workspace.sync.countsReady.get_ptr(buffer)) +
        static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    GmWrite(tokenCountReadyFlag, static_cast<int32_t>(1));

    // 缓存仅由本核 AIV1 消费，先发布整表就绪，再用 VF 完成元数据统计，避免延后其他角色读取计数。
    // VF 复用 metadata 区前缀：result[0] 为 totalMGroups，result[8] 为最后非空专家下标。
    auto* waveMetaTensor = reinterpret_cast<__ubuf__ int32_t*>(ubBufferUsedAddr_);
    CallVf<ComputeLayeredWaveMetaVF>(expertTokenNumsOutTensor_, waveMetaTensor, moeExpertPerRank_);
    PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID2>();

    const int32_t totalMGroups = waveMetaTensor[0];
    const int32_t lastIndex = waveMetaTensor[8];
    const uint32_t lastNonEmptyExpertEnd = lastIndex >= 0 ? static_cast<uint32_t>(lastIndex) + 1U : 0U;
    if (totalMGroups <= 0) {
        gmmWaveCount_ = 1U; // 全部专家为空时，规划器仍需一个元数据 wave 完成遍历。
        return;
    }
    const uint64_t totalMGroups64 = static_cast<uint64_t>(totalMGroups);
    gmmWaveCount_ = (totalMGroups64 - 1U) / mGroupsPerWave_ + 1U;
    // 最后一个满 wave 后仍有空专家时，保留规划器产生的元数据尾 wave。
    if (totalMGroups64 % mGroupsPerWave_ == 0U && lastNonEmptyExpertEnd < moeExpertPerRank_) {
        ++gmmWaveCount_;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateLocalRelayWeight(int32_t copyIdx,
                                                                                              int32_t outIdx,
                                                                                              TEventID eventId)
{
    if constexpr (TopkWeightsPrefetch) {
        PipeWait<PIPE_MTE2, PIPE_S>(eventId);
        __ubuf__ ActivationType* weightBuf = copyTmpTensors_[outIdx];
        uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
        __ubuf__ int32_t* bufWeightsInt32 = reinterpret_cast<__ubuf__ int32_t*>((weightBuf + (weightOffsetInUb)));
        int32_t topkIndex = (metaInfoTensor_ + (copyIdx * INT32_PER_256B))[TOPK_INDEX];
        int32_t weightBits = bufWeightsInt32[static_cast<uint32_t>(topkIndex)];
        (metaInfoTensor_ + (copyIdx * INT32_PER_256B))[WEIGHT_INDEX] = weightBits;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadLocalRelayToken(
    uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx, uint32_t copyInNum, TEventID eventId)
{
    PipeWait<PIPE_MTE3, PIPE_MTE2>(eventId);
    LoadTokenFromLocalRelay(srcServer, tokenIndex, bufferIdx, copyInNum);
    PipeNotify<PIPE_MTE2, PIPE_MTE3>(eventId);
    if constexpr (TopkWeightsPrefetch) {
        PipeNotify<PIPE_MTE2, PIPE_S>(eventId);
    }
}

// ============================================================================
// CopyTokensFromLocalRelay：本卡中转使用 UB 多 buffer 搬运 token 与 scale
// ----------------------------------------------------------------------------
//   prime: 发出前 DISPATCH_COPY_BUFFER_COUNT 个 token 的 MTE2。
//   steady: 每轮执行 MTE3_out[i] + MTE2_in[i + DISPATCH_COPY_BUFFER_COUNT]，循环复用槽位。
//   drain: 收尾不再发新 MTE2，只等待 MTE3 完成。
// ============================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyTokensFromLocalRelay(
    int32_t rowDstOffsetInCore, uint32_t srcServer, int32_t copyNum, int64_t widthA, int64_t widthAScale,
    uint32_t copyInNum)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    constexpr TEventID kBufEvents[DISPATCH_COPY_BUFFER_COUNT] = {EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    __gm__ ActivationType* recvDataGm = nullptr;
    __gm__ QuantScaleOutType* recvScaleGm = nullptr;
    recvDataGm = reinterpret_cast<__gm__ ActivationType*>(workspace.routed.gmm1Input.data.get_ptr(buffer) +
                                                          static_cast<uint64_t>(rowDstOffsetInCore) * widthA *
                                                              sizeof(ActivationType));
    recvScaleGm = reinterpret_cast<__gm__ QuantScaleOutType*>(workspace.routed.gmm1Input.scale.get_ptr(buffer) +
                                                              static_cast<uint64_t>(rowDstOffsetInCore) * widthAScale *
                                                                  sizeof(QuantScaleOutType));

    for (int32_t bufferIdx = 0; bufferIdx < DISPATCH_COPY_BUFFER_COUNT; ++bufferIdx) {
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(kBufEvents[bufferIdx]);
    }

    int32_t primeCount = copyNum < DISPATCH_COPY_BUFFER_COUNT ? copyNum : DISPATCH_COPY_BUFFER_COUNT;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t tokenIndex = (metaInfoTensor_ + (primeIdx * INT32_PER_256B))[TOKEN_ID];
        TEventID eventId = kBufEvents[primeIdx];
        LoadLocalRelayToken(srcServer, static_cast<uint32_t>(tokenIndex), primeIdx, copyInNum, eventId);
    }

    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t outIdx = copyIdx % DISPATCH_COPY_BUFFER_COUNT;
        TEventID eventId = kBufEvents[outIdx];
        PipeWait<PIPE_MTE2, PIPE_MTE3>(eventId);

        UpdateLocalRelayWeight(copyIdx, outIdx, eventId);
        __ubuf__ ActivationType* tokenScaleBuf = copyTmpTensors_[outIdx];
        __ubuf__ QuantScaleOutType* scaleBuf =
            reinterpret_cast<__ubuf__ QuantScaleOutType*>((tokenScaleBuf + (mxQuantTokenAlignBytes_)));
        CopyUbToGm((recvDataGm + (copyIdx * widthA)), tokenScaleBuf,
                   static_cast<uint16_t>(widthA * sizeof(ActivationType)));
        CopyUbToGm((recvScaleGm + (copyIdx * widthAScale)), scaleBuf,
                   static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)));
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(eventId);

        int32_t nextIdx = copyIdx + DISPATCH_COPY_BUFFER_COUNT;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = (metaInfoTensor_ + (nextIdx * INT32_PER_256B))[TOKEN_ID];
            // 等待本轮 MTE3 完成后再复用 outIdx 槽。
            LoadLocalRelayToken(srcServer, static_cast<uint32_t>(tokenIndex), outIdx, copyInNum, eventId);
        }
    }

    for (int32_t bufferIdx = 0; bufferIdx < DISPATCH_COPY_BUFFER_COUNT; ++bufferIdx) {
        PipeWait<PIPE_MTE3, PIPE_MTE2>(kBufEvents[bufferIdx]);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishDispatchRows(uint32_t expertIdx,
                                                                                           int32_t globalRowStart,
                                                                                           int32_t rowCount)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    if (rowCount <= 0) {
        return;
    }
    int32_t priorExpertRows = expertIdx == 0U ? 0 : cumsumInfoTensor_[expertIdx * worldSize_ - 1U];
    int32_t localRow = globalRowStart - priorExpertRows;
    int32_t remainingRows = rowCount;
    __gm__ int32_t* expertFlagBase = workspace.sync.dispatchReady.template get_ptr<int32_t>(buffer) +
                                     static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_;
    while (remainingRows > 0) {
        int32_t waveIdx = localRow / static_cast<int32_t>(L1_TILE_M_256);
        int32_t rowsInWave = static_cast<int32_t>(L1_TILE_M_256) - localRow % static_cast<int32_t>(L1_TILE_M_256);
        if (rowsInWave > remainingRows) {
            rowsInWave = remainingRows;
        }
        atomicAdd(expertFlagBase + static_cast<int64_t>(waveIdx) * INT_CACHELINE, rowsInWave);
        localRow += rowsInWave;
        remainingRows -= rowsInWave;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::FindDispatchSegmentEnd(
    uint32_t segmentStart, uint32_t batchTokenCount)
{
    __ubuf__ int32_t* firstRecord = (metaInfoTensor_ + (segmentStart * INT32_PER_256B));
    uint32_t expertIdx = static_cast<uint32_t>(firstRecord[DISPATCH_BATCH_EXPERT_ID]);
    int32_t dstRowStart = firstRecord[DISPATCH_BATCH_DST_ROW];
    uint32_t segmentEnd = segmentStart + 1U;
    while (segmentEnd < batchTokenCount) {
        __ubuf__ int32_t* nextRecord = (metaInfoTensor_ + (segmentEnd * INT32_PER_256B));
        if (static_cast<uint32_t>(nextRecord[DISPATCH_BATCH_EXPERT_ID]) != expertIdx ||
            nextRecord[DISPATCH_BATCH_DST_ROW] != dstRowStart + static_cast<int32_t>(segmentEnd - segmentStart)) {
            break;
        }
        ++segmentEnd;
    }
    return segmentEnd;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveRemoteDispatchBatch(
    uint32_t relayRank, uint32_t batchTokenCount)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    if (batchTokenCount == 0U) {
        return;
    }

    Channel channel = comm_.GetChannel(relayRank);
    BatchHandle batchHandle = comm_.MakeBatchHandle(channel, GetRankWinAddrWithOffset(relayRank, 0));
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        __ubuf__ int32_t* record = (metaInfoTensor_ + (index * INT32_PER_256B));
        uint32_t srcServer = static_cast<uint32_t>(record[DISPATCH_BATCH_SRC_SERVER]);
        uint32_t tokenIndex = static_cast<uint32_t>(record[TOKEN_ID]);
        uint32_t dstRow = static_cast<uint32_t>(record[DISPATCH_BATCH_DST_ROW]);
        GM_ADDR remoteRecordAddr = GetRankWinAddrWithOffset(relayRank, peermem.dispatchReceive.offsetBytes) +
                                   RelayTokenOffset(srcServer, tokenIndex);
        GM_ADDR tokenDstAddr = workspace.routed.gmm1Input.data.get_ptr(buffer) +
                               static_cast<uint64_t>(dstRow) * widthA * sizeof(ActivationType);
        GM_ADDR scaleDstAddr = workspace.routed.gmm1Input.scale.get_ptr(buffer) +
                               static_cast<uint64_t>(dstRow) * widthAScale * sizeof(QuantScaleOutType);
        comm_.ReadNbi(batchHandle, tokenDstAddr, remoteRecordAddr, widthA * sizeof(ActivationType));
        comm_.ReadNbi(batchHandle, scaleDstAddr, remoteRecordAddr + mxQuantTokenAlignBytes_,
                      widthAScale * sizeof(QuantScaleOutType));
        if constexpr (TopkWeightsPrefetch) {
            GM_ADDR weightDstAddr =
                workspace.routing.tokenWeights.get_ptr(buffer) + static_cast<uint64_t>(dstRow) * weightAlignBytes_;
            comm_.ReadNbi(batchHandle, weightDstAddr,
                          remoteRecordAddr + mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, weightAlignBytes_);
        }
    }
    comm_.BatchCommit(batchHandle);

    if constexpr (TopkWeightsPrefetch) {
        // 预取路径提交后立即读取刚收回的权重，必须先等这批 token/scale/weight 读取完成。
        comm_.Drain(batchHandle);
        for (uint32_t index = 0U; index < batchTokenCount; ++index) {
            __ubuf__ int32_t* record = (metaInfoTensor_ + (index * INT32_PER_256B));
            uint32_t dstRow = static_cast<uint32_t>(record[DISPATCH_BATCH_DST_ROW]);
            uint32_t topkIndex = static_cast<uint32_t>(record[TOPK_INDEX]);
            __gm__ int32_t* weightGmI32 = reinterpret_cast<__gm__ int32_t*>(
                workspace.routing.tokenWeights.get_ptr(buffer) + static_cast<uint64_t>(dstRow) * weightAlignBytes_);
            record[WEIGHT_INDEX] = GmRead(weightGmI32 + topkIndex);
        }
    }

    // 每个 expert/rank 段内的 metadata 连续。先写完全部 metadata，再发布专家就绪计数。
    // 元数据及预取权重由标量更新，搬出前需保证这些 UB 写入已完成。
    PipeFence<PIPE_S, PIPE_MTE3, SYNC_EVENT_ID3>();
    uint32_t segmentStart = 0U;
    while (segmentStart < batchTokenCount) {
        __ubuf__ int32_t* firstRecord = (metaInfoTensor_ + (segmentStart * INT32_PER_256B));
        int32_t dstRowStart = firstRecord[DISPATCH_BATCH_DST_ROW];
        uint32_t segmentEnd = FindDispatchSegmentEnd(segmentStart, batchTokenCount);
        CopyUbToGm((metadataGm_ + (static_cast<uint64_t>(dstRowStart) * INT32_PER_256B)),
                   (metaInfoTensor_ + (segmentStart * INT32_PER_256B)),
                   ((segmentEnd - segmentStart) * INT32_PER_256B) * sizeof(int32_t));
        segmentStart = segmentEnd;
    }
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID3>();

    if constexpr (!TopkWeightsPrefetch) {
        // 非预取时 metadata 由标量构造，与这批 token/scale 读取相互独立；搬出与通信等待重叠，
        // 发布 ready 前汇合两者。
        comm_.Drain(batchHandle);
    }

    segmentStart = 0U;
    while (segmentStart < batchTokenCount) {
        __ubuf__ int32_t* firstRecord = (metaInfoTensor_ + (segmentStart * INT32_PER_256B));
        uint32_t expertIdx = static_cast<uint32_t>(firstRecord[DISPATCH_BATCH_EXPERT_ID]);
        int32_t dstRowStart = firstRecord[DISPATCH_BATCH_DST_ROW];
        uint32_t segmentEnd = FindDispatchSegmentEnd(segmentStart, batchTokenCount);
        PublishDispatchRows(expertIdx, dstRowStart, static_cast<int32_t>(segmentEnd - segmentStart));
        segmentStart = segmentEnd;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyLocalDispatchTokens(int32_t dstRow,
                                                                                               int32_t srcRank,
                                                                                               int32_t tokenOffset,
                                                                                               int32_t tokenCount)
{
    if (tokenCount <= 0) {
        return;
    }
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                          MXFP_MULTI_BASE_SIZE; // 输出 token-scale 长度,紧密排列
    uint32_t copyInNum = mxQuantTokenScaleAlignBytes_ / sizeof(ActivationType);
    if constexpr (!TopkWeightsPrefetch) {
        // mode4 一级记录仍保留中继归约所需权重，二级只搬 token 和 scale。
        copyInNum =
            Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32)) /
            sizeof(ActivationType);
    }
    uint32_t srcServer = static_cast<uint32_t>(srcRank) / rankNumPerServer_;

    for (int32_t i = 0; i < tokenCount; ++i) {
        int32_t topkIndex = validTopkIndexTensor_[tokenOffset + i];
        int32_t tokenIndex = topkIndex / topK_;
        (metaInfoTensor_ + (i * INT32_PER_256B))[RANK_ID] = srcRank;
        (metaInfoTensor_ + (i * INT32_PER_256B))[TOKEN_ID] = tokenIndex;
        (metaInfoTensor_ + (i * INT32_PER_256B))[TOPK_INDEX] = topkIndex % topK_;
    }

    CopyTokensFromLocalRelay(dstRow, srcServer, tokenCount, widthA, widthAScale, copyInNum);
    PipeFence<PIPE_S, PIPE_MTE3, SYNC_EVENT_ID3>();
    CopyUbToGm((metadataGm_ + (dstRow * INT32_PER_256B)), metaInfoTensor_,
               (tokenCount * INT32_PER_256B) * sizeof(int32_t));
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID3>();
}

// 加载完整 cumsum 表到 UB，并初始化 dispatch metadata 临时区。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareDispatchReceive()
{
    CopyGmToUb(cumsumInfoTensor_, cumsumGm_, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)),
               GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
    PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID2>();
    metaInfoTensor_ = reinterpret_cast<__ubuf__ int32_t*>(ubBufferUsedAddr_);
}

/*
 * 计算当前专家与通信 wave 的交集，返回全局紧凑行号区间 [beginRow, endRow)。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline DispatchRowRange MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CalcDispatchExpertSlice(
    uint32_t expertIdx, uint32_t expertBegin, const ExpertTokenRange& range)
{
    const int32_t expertStart = expertIdx == 0U ? 0 : cumsumInfoTensor_[expertIdx * worldSize_ - 1U];
    const int32_t expertEndRow = cumsumInfoTensor_[(expertIdx + 1U) * worldSize_ - 1U];
    DispatchRowRange expertRows;
    expertRows.beginRow =
        expertStart + static_cast<int32_t>(expertIdx == expertBegin ? range.begin.tokenIndexInExpert : 0U);
    expertRows.endRow = expertIdx == range.end.expertIdx ?
                            expertStart + static_cast<int32_t>(range.end.tokenIndexInExpert) :
                            expertEndRow;
    expertRows.endRow = expertRows.endRow < static_cast<int32_t>(maxOutputSize_) ? expertRows.endRow :
                                                                                   static_cast<int32_t>(maxOutputSize_);
    return expertRows;
}

// 搬入本轮 mask 并 GatherMask，返回本轮命中的 topk 行数。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadAndGatherDispatchRound(
    __gm__ uint8_t* rankMaskBasePtr, uint32_t roundIdx)
{
    uint64_t roundStart = static_cast<uint64_t>(roundIdx) * static_cast<uint64_t>(dispatchRoundSendTotalNum_);
    uint32_t roundLen = (roundIdx + 1U < dispatchTotalRounds_) ? dispatchRoundSendTotalNum_ :
                                                                 static_cast<uint32_t>(maskRouteCapacity_ - roundStart);
    uint32_t topkIndexTensorElemCount =
        static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(roundLen * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32))) /
        sizeof(int32_t);
    uint32_t curRoundMaskBytes =
        (roundLen == dispatchRoundSendTotalNum_) ?
            dispatchRoundMaskAlignSize_ :
            static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<int64_t>(Ops::Base::CeilDiv(roundLen, 8U)),
                                                       static_cast<int64_t>(ALIGN_32)));
    uint64_t roundMaskByteOffset = roundStart / 8U;
    auto* roundMaskSrc = rankMaskBasePtr + roundMaskByteOffset;
    PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID1>();
    CopyGmToUb(gatherMaskTensor_, roundMaskSrc, curRoundMaskBytes);
    PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID1>();
    LocalTensor<int32_t> indices(TPosition::VECCALC, reinterpret_cast<uint64_t>(topkIndexTensor_),
                                 topkIndexTensorElemCount);
    CreateVecIndex(indices, static_cast<int32_t>(roundStart), topkIndexTensorElemCount);

    uint64_t roundTokenCount = 0U;
    LocalTensor<uint32_t> rankMaskSlice(TPosition::VECCALC, reinterpret_cast<uint64_t>(gatherMaskInt32Tensor_),
                                        curRoundMaskBytes / sizeof(uint32_t));
    LocalTensor<int32_t> selected(TPosition::VECCALC, reinterpret_cast<uint64_t>(validTopkIndexTensor_),
                                  topkIndexTensorElemCount);
    GatherMask(selected, indices, rankMaskSlice, true, roundLen, {1, 1, 0, 0}, roundTokenCount);
    PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID4>();
    return roundTokenCount;
}

/*
 * 将本轮命中行裁剪到通信 wave 的 [sourceBegin, sourceEnd)，同时返回扫描游标的推进量。
 * mask 容量尾部可能残留上次调用数据，有效行数只取 rankTokenCount 的剩余量。
 */
__aicore__ inline DispatchRoundSlice ClipDispatchRoundToWave(int32_t accumulatedRows, uint64_t roundTokenCount,
                                                             int32_t rankTokenCount, int32_t sourceBegin,
                                                             int32_t sourceEnd)
{
    const int32_t remainingRankRows = rankTokenCount - accumulatedRows;
    DispatchRoundSlice roundSlice;
    roundSlice.validRowCount = static_cast<int32_t>(roundTokenCount);
    if (roundSlice.validRowCount > remainingRankRows) {
        roundSlice.validRowCount = remainingRankRows;
    }
    roundSlice.sourceRowBegin = accumulatedRows > sourceBegin ? accumulatedRows : sourceBegin;
    const int32_t roundEnd = accumulatedRows + roundSlice.validRowCount;
    const int32_t copyEnd = roundEnd < sourceEnd ? roundEnd : sourceEnd;
    roundSlice.copyRowCount = copyEnd - roundSlice.sourceRowBegin;
    roundSlice.gatheredOffset = roundSlice.sourceRowBegin - accumulatedRows;
    return roundSlice;
}

// 本地 relay：按 256 行 chunk 搬入，并发布对应 group 的 GMM1 ready。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyLocalDispatchRound(
    uint32_t expertIdx, uint32_t srcRank, int32_t rowStart, int32_t copyBegin, int32_t roundCopyCount,
    int32_t gatheredOffset)
{
    for (int32_t chunkStart = 0; chunkStart < roundCopyCount; chunkStart += static_cast<int32_t>(L1_TILE_M_256)) {
        int32_t chunkRows = roundCopyCount - chunkStart;
        if (chunkRows > static_cast<int32_t>(L1_TILE_M_256)) {
            chunkRows = static_cast<int32_t>(L1_TILE_M_256);
        }
        int32_t dstRow = rowStart + copyBegin + chunkStart;
        CopyLocalDispatchTokens(dstRow, static_cast<int32_t>(srcRank), gatheredOffset + chunkStart, chunkRows);
        PublishDispatchRows(expertIdx, dstRow, chunkRows);
    }
}

// 远端 relay：逐 token 填 metadata，攒满批次后立即读取到本地 workspace。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendRemoteDispatchRound(
    uint32_t relayRank, uint32_t expertIdx, uint32_t srcServer, uint32_t srcRank, int32_t rowStart, int32_t copyBegin,
    int32_t roundCopyCount, int32_t gatheredOffset, uint32_t& remoteBatchTokenCount)
{
    for (int32_t tokenOffset = 0; tokenOffset < roundCopyCount; ++tokenOffset) {
        int32_t topkIndex = validTopkIndexTensor_[gatheredOffset + tokenOffset];
        int32_t dstRow = rowStart + copyBegin + tokenOffset;
        auto* record = metaInfoTensor_ + remoteBatchTokenCount * INT32_PER_256B;
        record[RANK_ID] = static_cast<int32_t>(srcRank);
        record[TOKEN_ID] = topkIndex / static_cast<int32_t>(topK_);
        record[TOPK_INDEX] = topkIndex % static_cast<int32_t>(topK_);
        record[WEIGHT_INDEX] = 0;
        record[DISPATCH_BATCH_DST_ROW] = dstRow;
        record[DISPATCH_BATCH_EXPERT_ID] = static_cast<int32_t>(expertIdx);
        record[DISPATCH_BATCH_SRC_SERVER] = static_cast<int32_t>(srcServer);
        ++remoteBatchTokenCount;
        if (remoteBatchTokenCount == DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) {
            ReceiveRemoteDispatchBatch(relayRank, remoteBatchTokenCount);
            remoteBatchTokenCount = 0U;
        }
    }
}

/*
 * 按 mask 轮次接收单个来源 rank 在当前专家、通信 wave 内的行。
 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsLocalServer>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveDispatchFromSourceRank(
    uint32_t relayRank, uint32_t expertIdx, uint32_t srcServer, __gm__ uint8_t* expertMaskBase,
    const DispatchRowRange& expertRows, uint32_t& remoteBatchTokenCount)
{
    const uint32_t srcRank = srcServer * rankNumPerServer_ + relayRank % rankNumPerServer_;
    const int32_t rowStart =
        (srcRank == 0U && expertIdx == 0U) ? 0 : cumsumInfoTensor_[expertIdx * worldSize_ + srcRank - 1U];
    if (rowStart >= static_cast<int32_t>(maxOutputSize_)) {
        return;
    }
    auto* rankMaskBasePtr = expertMaskBase + static_cast<uint64_t>(srcRank) * moeExpertPerRank_ * maskSlotSize_;
    auto* rankCountPtr = reinterpret_cast<__gm__ int32_t*>(rankMaskBasePtr + maskAlignSize_);
    const int32_t rankTokenCount = GmRead(rankCountPtr);
    if (rankTokenCount <= 0) {
        return;
    }
    const int32_t sourceBegin = expertRows.beginRow > rowStart ? expertRows.beginRow - rowStart : 0;
    const int32_t sourceEnd =
        expertRows.endRow < rowStart + rankTokenCount ? expertRows.endRow - rowStart : rankTokenCount;
    if (sourceBegin >= sourceEnd) {
        return;
    }
    const bool isRemoteRelay = relayRank != rankId_;
    int32_t accumulatedRows = 0;
    for (uint32_t roundIdx = 0U; roundIdx < dispatchTotalRounds_ && accumulatedRows < sourceEnd; ++roundIdx) {
        const uint64_t roundTokenCount = LoadAndGatherDispatchRound(rankMaskBasePtr, roundIdx);
        if (roundTokenCount == 0U) {
            continue;
        }
        const DispatchRoundSlice roundSlice =
            ClipDispatchRoundToWave(accumulatedRows, roundTokenCount, rankTokenCount, sourceBegin, sourceEnd);
        if (roundSlice.copyRowCount > 0) {
            // 本 Server 的记录在进入 wave 前已量化并经过跨卡同步，只等待远端来源。
            if constexpr (!IsLocalServer) {
                EnsureDispatchSourceReady(relayRank, srcServer);
            }
            if (isRemoteRelay) {
                AppendRemoteDispatchRound(relayRank, expertIdx, srcServer, srcRank, rowStart, roundSlice.sourceRowBegin,
                                          roundSlice.copyRowCount, roundSlice.gatheredOffset, remoteBatchTokenCount);
            } else {
                CopyLocalDispatchRound(expertIdx, srcRank, rowStart, roundSlice.sourceRowBegin, roundSlice.copyRowCount,
                                       roundSlice.gatheredOffset);
            }
        }
        accumulatedRows += roundSlice.validRowCount;
    }
}

/*
 * 固定专家的 wave 交集与 mask 基址，再按当前阶段遍历本 Server 或远端 Server 来源。
 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsLocalServer>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveDispatchExpertFromServers(
    uint32_t relayRank, uint32_t expertIdx, const ExpertTokenRange& range, uint32_t& remoteBatchTokenCount)
{
    const DispatchRowRange expertRows = CalcDispatchExpertSlice(expertIdx, range.begin.expertIdx, range);
    if (expertRows.beginRow >= expertRows.endRow) {
        return;
    }
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    auto* expertMaskBase = reinterpret_cast<__gm__ uint8_t*>(peermem.maskRecv.get_ptr(commBuffer) +
                                                             static_cast<uint64_t>(expertIdx) * maskSlotSize_);
    const uint32_t srcServerBegin = IsLocalServer ? serverId_ : 0U;
    const uint32_t srcServerEnd = IsLocalServer ? serverId_ + 1U : serverNum_;
    for (uint32_t srcServer = srcServerBegin; srcServer < srcServerEnd; ++srcServer) {
        if (!IsLocalServer && srcServer == serverId_) {
            continue;
        }
        ReceiveDispatchFromSourceRank<IsLocalServer>(relayRank, expertIdx, srcServer, expertMaskBase, expertRows,
                                                     remoteBatchTokenCount);
    }
}

/*
 * 在一个通信 wave 内完成本 Server 或远端 Server 阶段，覆盖全部归属 relay 和专家。
 * 每个 relay 的尾批在本阶段内完成，两个阶段之间不保留未消费的 metadata。
 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsLocalServer>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveDispatchServerPhase(
    const ExpertTokenRange& range)
{
    const uint32_t expertBegin = range.begin.expertIdx;
    const uint32_t expertEnd = range.end.expertIdx + (range.end.tokenIndexInExpert != 0U ? 1U : 0U);
    for (uint32_t srcRankInServer = 0U; srcRankInServer < rankNumPerServer_; ++srcRankInServer) {
        const uint32_t relayRank = serverId_ * rankNumPerServer_ + srcRankInServer;
        if (relayRank >= worldSize_ || !IsChannelOwner(relayRank)) {
            continue;
        }
        uint32_t remoteBatchTokenCount = 0U;
        for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
            ReceiveDispatchExpertFromServers<IsLocalServer>(relayRank, expertIdx, range, remoteBatchTokenCount);
        }
        // 本 Server 的尾批在开始任何远端来源等待前完成；批次不跨 relay 或阶段。
        if (relayRank != rankId_ && remoteBatchTokenCount != 0U) {
            ReceiveRemoteDispatchBatch(relayRank, remoteBatchTokenCount);
        }
    }
}

// =================================================================================================
// ReceiveDispatchTokenRange：接收一个通信 wave 覆盖的 dispatch token 行区间。
//   - 按 relay 通道、专家、源 rank、dispatch 轮裁剪；
//   - 本地 relay 直接搬入，远端 relay 攒 metadata 后走 ReceiveRemoteDispatchBatch；
//   - 按 256 行 group 发布 GMM1 dispatch ready flag。
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveDispatchTokenRange(
    const ExpertTokenRange& range)
{
    const uint32_t expertBegin = range.begin.expertIdx;
    const uint32_t expertEnd = range.end.expertIdx + (range.end.tokenIndexInExpert != 0U ? 1U : 0U);
    if (expertBegin >= expertEnd || expertBegin >= moeExpertPerRank_) {
        return;
    }
    PrepareDispatchReceive();
    ReceiveDispatchServerPhase<true>(range);
    ReceiveDispatchServerPhase<false>(range);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline auto MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitSharedExpertInputBuffers()
    -> QuantProcessScratch<ActivationType>
{
    uint32_t xInAddr = ALIGN_512;
    uint32_t xInSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    __ubuf__ bfloat16_t* xInBuf0 = reinterpret_cast<__ubuf__ bfloat16_t*>(xInAddr);
    __ubuf__ bfloat16_t* xInBuf1 = reinterpret_cast<__ubuf__ bfloat16_t*>(xInAddr + xInSize);
    uint32_t mxTempAddr = xInAddr + xInSize * 2;
    __ubuf__ uint16_t* mxTempBuf = reinterpret_cast<__ubuf__ uint16_t*>(mxTempAddr);
    uint32_t xOutAddr = mxTempAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t xOutSize =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    __ubuf__ ActivationType* xOutBuf0 = reinterpret_cast<__ubuf__ ActivationType*>(xOutAddr);
    __ubuf__ ActivationType* xOutBuf1 = reinterpret_cast<__ubuf__ ActivationType*>(xOutAddr + xOutSize);
    QuantProcessScratch<ActivationType> scratch{};
    scratch.xInTensor0 = xInBuf0;
    scratch.xInTensor1 = xInBuf1;
    scratch.xOutTensor0 = xOutBuf0;
    scratch.xOutTensor1 = xOutBuf1;
    scratch.mxTempTensor = mxTempBuf;
    return scratch;
}

// ===============================================================
// SharedExpertCopyInput：从原始 bf16 输入量化后写入共享专家专用缓冲区
//   源: aGmAddr [bs × h] bf16（layered URMA 模式下 quantTokenScale 区域未分配，需直接从原始输入量化）
//   目标: sharedExpertInputDataPtr [bs × h] fp8 连续, sharedExpertInputScalePtr [bs × scaleN] 连续
//   AIV 执行，在 AIC GMM1 开始前调用
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SharedExpertCopyInput()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    int32_t currentNum;
    int32_t currentOffset;
    TilingByCore(m_, currentNum, currentOffset, 1);

    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    auto scratch = InitSharedExpertInputBuffers();
    __gm__ bfloat16_t* inputGm = nullptr;
    __gm__ ActivationType* dataOutputGm = nullptr;
    __gm__ QuantScaleOutType* scaleOutputGm = nullptr;
    dataOutputGm = workspace.shared.gmm1Input.data.template get_ptr<ActivationType>(buffer);
    scaleOutputGm = workspace.shared.gmm1Input.scale.template get_ptr<QuantScaleOutType>(buffer);

    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        int32_t tokenIdx = currentOffset + index;
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInBuf = (index % DOUBLE_BUFFER == 0) ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutBuf = (index % DOUBLE_BUFFER == 0) ? scratch.xOutTensor0 : scratch.xOutTensor1;

        inputGm = reinterpret_cast<__gm__ bfloat16_t*>(params_.aGmAddr +
                                                       static_cast<uint64_t>(tokenIdx) * k_ * sizeof(bfloat16_t));
        PipeWait<PIPE_MTE3, PIPE_MTE2>(event);
        CopyGmToUb(xInBuf, inputGm, static_cast<uint16_t>(k_ * sizeof(bfloat16_t)),
                   GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);

        QuantizeTokenInUb(xInBuf, xOutBuf, scratch.mxTempTensor);

        PipeNotify<PIPE_V, PIPE_MTE3>(event);
        PipeWait<PIPE_V, PIPE_MTE3>(event);
        __ubuf__ QuantScaleOutType* bufScale =
            reinterpret_cast<__ubuf__ QuantScaleOutType*>((xOutBuf + (mxQuantTokenAlignBytes_)));
        CopyUbToGm((dataOutputGm + (tokenIdx * widthA)), xOutBuf,
                   static_cast<uint16_t>(widthA * sizeof(ActivationType)));
        CopyUbToGm((scaleOutputGm + (tokenIdx * widthAScale)), bufScale,
                   static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)));
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(event);
    }
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_LAYERED_DISPATCH_H
