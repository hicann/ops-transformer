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
    validTopkIndexTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(dispatchRoundSendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIndexTensorSize / sizeof(int32_t));
    // gatherMaskTensor_ 仅容纳单轮 mask 位。
    uint32_t gatherMaskTensorAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t gatherMaskTensorSize = dispatchRoundMaskAlignSize_;
    gatherMaskTensor_ =
        LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr, gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ =
        LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr, gatherMaskTensorSize / sizeof(uint32_t));
    return gatherMaskTensorAddr + gatherMaskTensorSize;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchCopyBuffers(
    uint32_t copyTmpBaseAddr, uint32_t relayFlagTensorSize, uint32_t dispatchWorkTensorSize,
    uint32_t expertTokenNumsOutTensorSize)
{
    uint32_t tokenScaleSize = mxQuantTokenScaleAlignBytes_;
    uint32_t copyTmpBufferSize = tokenScaleSize;
    for (int32_t index = 0; index < DISPATCH_COPY_BUFFER_COUNT; ++index) {
        copyTmpTensors_[index] = LocalTensor<ActivationType>(
            TPosition::VECCALC, copyTmpBaseAddr + static_cast<uint32_t>(index) * copyTmpBufferSize,
            copyTmpBufferSize / sizeof(ActivationType));
    }
    relayFlagTensor_ =
        LocalTensor<uint64_t>(TPosition::VECCALC, copyTmpBaseAddr, relayFlagTensorSize / sizeof(uint64_t));
    // 中继标志快照和本地搬运缓冲的生命期不重叠，因此复用同一段 UB。
    // 一级量化已经在 Wave 前完成，Dispatch 阶段不再保留旧的单 token 量化和稀疏槽位 UB。
    uint32_t dispatchWorkTensorEnd = copyTmpBaseAddr + dispatchWorkTensorSize;
    // Tensor用处：ExpertTokenNumCopyOut 函数中本卡各专家收到的tokenCnt数；
    // Tensor大小：moeExpertPerRank_ * sizeof(int32_t) 对齐至32字节；
    uint32_t expertTokenNumsOutTensorAddr = dispatchWorkTensorEnd;
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                                                     expertTokenNumsOutTensorSize / sizeof(int32_t));
    // 后续 Dispatch 接收从该地址申请 metadata 临时区。
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
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

    // Dispatch 阶段 UB 布局：Hcomm 私有区、专家计数/前缀和、分轮索引/掩码、
    // 中继标志快照/本地搬运缓冲、专家计数输出和元数据区依次排列。
    // 按最大 256 行元数据预留后，通信工作区不得越过 184 KiB 处的 Hcomm WQE 专用区。
    LocalTensor<uint8_t> hcommTensor = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512);
    hcomm_.Init(hcommTensor, ALIGN_512 / sizeof(uint8_t));
    hcommBatchWqeTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, LAYERED_HCOMM_BATCH_UB_OFFSET,
                                                LAYERED_HCOMM_BATCH_UB_BYTES / sizeof(uint8_t));
    uint32_t expertTokenCntTensorAddr = ALIGN_512;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    expertTokenCntTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // 分轮次参数计算：validTopkIndexTensor_ + topkIndexTensor_ + gatherMaskTensor_ 均为 round-sized
    // 专家计数直接从 GM 读取，不再加载完整 mask slot 到 UB。
    // round 张量: validTopkIndex(R*4) + topkIndex(R*4) + gatherMask(R/8) = 8R + R/8 = 65R/8
    // copyTmpTensors_ 与 relayFlagTensor_ 复用地址，只计两者中的较大值；此外预留专家计数输出和
    // 256 行 metadata。DispatchPrepare 使用的量化缓冲生命周期不重叠，不计入本阶段。
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(DISPATCH_COPY_BUFFER_COUNT) * mxQuantTokenScaleAlignBytes_;
    uint32_t relayFlagTensorSize =
        Ops::Base::CeilAlign(static_cast<uint32_t>(DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY * sizeof(uint64_t)),
                             static_cast<uint32_t>(ALIGN_32));
    uint32_t expertTokenNumsOutTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    uint32_t dispatchWorkTensorSize = relayFlagTensorSize > copyTmpTotalSize ? relayFlagTensorSize : copyTmpTotalSize;
    uint32_t dispatchFixedCost = dispatchWorkTensorSize + expertTokenNumsOutTensorSize + L1_TILE_M_256 * ALIGN_32;
    uint32_t fixedBefore = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    InitDispatchRounds(fixedBefore, dispatchFixedCost);
    uint32_t copyTmpBaseAddr = InitDispatchIndexBuffers(fixedBefore);
    InitDispatchCopyBuffers(copyTmpBaseAddr, relayFlagTensorSize, dispatchWorkTensorSize, expertTokenNumsOutTensorSize);
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
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
    xInTensor1_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInTensorAddr1, xInTensorSize / sizeof(bfloat16_t));
    xInTensor2_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, xInTensorAddr2, xInTensorSize / sizeof(bfloat16_t));

    uint32_t mxTempTensorAddr = xInTensorAddr2 + xInTensorSize;
    mxTempTensor_ =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, MX_QUANT_TEMP_UB_BYTES / sizeof(uint16_t));

    uint32_t xOutTensorSize = mxQuantTokenScaleAlignBytes_;
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    xOutTensor1_ =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    xOutTensor2_ =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));

    uint32_t readyFlagTensorAddr = xOutTensorAddr2 + xOutTensorSize;
    relayReadyFillTensor_ =
        LocalTensor<uint64_t>(TPosition::VECCALC, readyFlagTensorAddr, DISPATCH_READY_FLAG_BATCH_TOKENS);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitResetBatch()
{
    // 计算 resetTensor 大小（与原逻辑一致）
    uint64_t totalFlagInt32 =
        static_cast<uint64_t>(moeExpertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_) +
         static_cast<uint64_t>(INT_CACHELINE) * static_cast<uint64_t>(aicNum_)); // 64 * (16 + 256 + 16 * 28) = 46080
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        totalFlagInt32 += static_cast<uint64_t>(aicNum_) * INT_CACHELINE;
    }
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
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitSendMaskRounds(uint32_t resetTensorSize)
{
    // 分轮次参数计算：UB 预算扣除固定开销后，求解最大 roundSendTotalNum_
    // round 张量: topkIds(R*4) + 2*maskSlot(CeilAlign(R/8,32)+32) + gatherOut(R*4)
    //            = 8R + R/4 + 64  (R 为 256 对齐时 CeilAlign(R/8,32)=R/8)
    // 解: R <= (available - 64) * 4 / 33，再 floor-align 到 256（保证 GM 写 32B 对齐）
    uint32_t fixedUbCosts = ALIGN_512 + resetTensorSize;
    uint32_t availableUb = COMM_WORK_UB_LIMIT - fixedUbCosts;
    uint32_t maxRoundSize = (availableUb - 64U) * 4U / 33U;
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
    roundCompareCount_ = roundSendTotalNum_; // 16K
    roundMaskAlignSize_ = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(roundCompareCount_) / 8, static_cast<int64_t>(ALIGN_32)));
    roundMaskSlotSize_ = roundMaskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    totalRounds_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(roundSendTotalNum_)));
}

// ======================================================================================
// SendAndQuantBuffInit：SendMaskCal & ResetFlagList localTensor申请
// --------------------------------------------------------------------------------------
//   大 bs 场景下 sendTotalNum 可能超出 UB 256KB 限制，因此按 roundSendTotalNum_ 分轮次
//   分配 UB 张量（topkIds / sendMask / sendGatherOut），每轮仅加载一部分 topkIds，
//   在 UB 中计算部分 mask 后拼写到 workspace 对应偏移，最终合并为完整 mask slot。
// ======================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // 输入准备与 DispatchBuffInit 生命期不重叠，可复用同一 UB 地址。
    // 减去 Hcomm 和分批清零区后，按 8R + R/4 + 64 字节计算单轮容量，避免清零区随 GM 总量增长。
    LocalTensor<uint8_t> hcommTensor = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
    hcomm_.Init(hcommTensor, ALIGN_512);
    hcommBatchWqeTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, LAYERED_HCOMM_BATCH_UB_OFFSET,
                                                LAYERED_HCOMM_BATCH_UB_BYTES / sizeof(uint8_t));

    uint32_t resetTensorSize = InitResetBatch();
    InitSendMaskRounds(resetTensorSize);
    // Tensor用处：SendMaskCal 每轮搬运本卡 topkIds 的一个子段；
    // Tensor大小：roundCompareCount_ 个 int32（256B 对齐）；
    uint32_t topkIdsTensorAddr = ALIGN_512;
    uint32_t topkIdsTensorSize = roundCompareCount_ * sizeof(int32_t);
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    // Tensor用处：SendMaskCal 每轮存储部分 mask 位；DOUBLE_BUFFER 双缓冲；
    // Tensor大小：roundMaskSlotSize_（每轮部分 [mask|count] 槽位大小）；
    uint32_t sendMaskAddr = resetTensorAddr + resetTensorSize;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] = LocalTensor<uint8_t>(
            TPosition::VECCALC, sendMaskAddr + static_cast<uint32_t>(index) * roundMaskSlotSize_, roundMaskSlotSize_);
    }

    // Tensor用处：SendMaskCal 每轮 GatherMask 计 count 的废弃输出 scratch；
    // Tensor大小：roundCompareCount_ 个 int32（256B 对齐）；
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * roundMaskSlotSize_;
    uint32_t sendGatherOutSize = roundCompareCount_ * sizeof(int32_t);
    sendGatherOutTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
}

// ===============================================================================================
// ResetFlagList：清理本卡 workspace 中连续排布的 GMM/dispatch flag 和 AIC-AIV1 ready sequence。
// ===============================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetFlagList()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // workSpace Flag 清零
    // 总数 = ActivationToGmm2(moeExpertPerRank * INT_CACHELINE)
    //        + DispatchToGmm1(moeExpertPerRank * dispatchFlagSlotsPerExpert_)
    //        + SendCntCalToUpdParams(moeExpertPerRank * aicNum_ * INT_CACHELINE)
    //        + GmmToEpilogue(aicNum_ * INT_CACHELINE, specialized A8W4/A4W4 only)
    int32_t flagNum =
        static_cast<int32_t>(moeExpertPerRank_) * (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_ +
                                                   static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(aicNum_));
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        flagNum += static_cast<int32_t>(aicNum_) * static_cast<int32_t>(INT_CACHELINE);
    }
    AivJobContext job{aivCoreIdx_, blockAivNum_};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    ResetWorkspaceRegion<1>(job, params_.workspaceInfo.flagActivationToGmm2Ptr, flagNum, resetBatchElementCount_,
                            resetTensor_);
    // URMA Layered 的量化与非量化 Combine 都使用 token-group counter。
    ResetGmm2CombineSyncCounters();
    if constexpr (TopkWeightsPrefetch) {
        int32_t statusElementCount =
            (static_cast<int32_t>(moeExpertPerRank_) * static_cast<int32_t>(params_.tilingData->maxTilesPerExpert) +
             1) *
            INT_CACHELINE;
        ResetWorkspaceRegion<1>(job, params_.workspaceInfo.gmm1TileStatusPtr, statusElementCount,
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
    DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                {1U, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    expertTokenNumsOutTensor_.SetValue(0, cumsumInfoTensor_.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < moeExpertPerRank_; expertIdx++) {
        int32_t cur = cumsumInfoTensor_.GetValue(expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx);
        int32_t prev = cumsumInfoTensor_.GetValue((expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx);
        expertTokenNumsOutTensor_.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut_, expertTokenNumsOutTensor_, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildExpertMaskRound(
    int32_t curExpertId, uint32_t roundIdx, uint64_t roundStart, uint32_t roundLen, uint32_t curRoundCompareCount,
    uint64_t &totalSendCnt)
{
    // Phase 1: 加载本轮 topkIds 子段；用非法 expert id 填充对齐尾部。
    Duplicate<int32_t>(topkIdsTensor_, -1, roundCompareCount_);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    GlobalTensor<int32_t> roundSrcGlobal;
    roundSrcGlobal.SetGlobalBuffer(
        (__gm__ int32_t *)(params_.expertIdxGmAddr + static_cast<uint64_t>(roundStart) * sizeof(int32_t)));
    DataCopyExtParams roundLoadParams{1U, static_cast<uint32_t>(roundLen * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> roundLoadPad{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor_, roundSrcGlobal, roundLoadParams, roundLoadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    // Phase 2: CompareScalar → 本轮部分 mask + GatherMask 累加 count
    LocalTensor<uint8_t> maskBuf = sendMaskTensor_[roundIdx % DOUBLE_BUFFER];
    LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
    CompareScalar(maskBuf, topkIdsTensor_, curExpertId, AscendC::CMPMODE::EQ, curRoundCompareCount);
    uint64_t roundSendCnt = 0;
    GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true, static_cast<uint32_t>(roundLen), {1, 1, 0, 0},
               roundSendCnt);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    totalSendCnt += roundSendCnt;
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildExpertMask(uint32_t curRankId,
                                                                                           int32_t curExpertId,
                                                                                           uint64_t srcOffset,
                                                                                           uint64_t dstOffset)
{
    uint64_t totalSendCnt = 0;
    for (uint32_t roundIdx = 0; roundIdx < totalRounds_; ++roundIdx) {
        uint64_t roundStart = static_cast<uint64_t>(roundIdx) * static_cast<uint64_t>(roundSendTotalNum_);
        uint64_t roundLen64 =
            (roundIdx + 1 < totalRounds_) ? static_cast<uint64_t>(roundSendTotalNum_) : sendTotalNum_ - roundStart;
        uint32_t roundLen = static_cast<uint32_t>(roundLen64);
        uint32_t curRoundCompareCount =
            (roundLen == roundSendTotalNum_) ?
                roundCompareCount_ :
                static_cast<uint32_t>(
                    Ops::Base::CeilAlign(static_cast<int64_t>(roundLen) * static_cast<int64_t>(sizeof(int32_t)),
                                         static_cast<int64_t>(ALIGN_256)) /
                    static_cast<int64_t>(sizeof(int32_t)));
        uint32_t curRoundMaskAlignSize =
            (roundLen == roundSendTotalNum_) ?
                roundMaskAlignSize_ :
                static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<int64_t>(curRoundCompareCount) / 8,
                                                           static_cast<int64_t>(ALIGN_32)));

        BuildExpertMaskRound(curExpertId, roundIdx, roundStart, roundLen, curRoundCompareCount, totalSendCnt);
        LocalTensor<uint8_t> maskBuf = sendMaskTensor_[roundIdx % DOUBLE_BUFFER];
        // Phase 2b: 将本轮部分 mask 写入 workspace 或 local win 对应字节偏移
        // roundByteOffset = roundStart/8，因 roundSendTotalNum_ 为 256 对齐，该偏移 32B 对齐
        uint64_t roundByteOffset = static_cast<uint64_t>(roundStart) / 8U;
        uint32_t writeBytes =
            (static_cast<uint64_t>(curRoundMaskAlignSize) <= static_cast<uint64_t>(maskAlignSize_) - roundByteOffset) ?
                curRoundMaskAlignSize :
                static_cast<uint32_t>(static_cast<uint64_t>(maskAlignSize_) - roundByteOffset);
        DataCopyExtParams partialMaskCopyParams{1U, writeBytes, 0U, 0U, 0U};

        if (curRankId == rankId_) {
            GlobalTensor<uint8_t> winDstGlobal;
            winDstGlobal.SetGlobalBuffer(
                (__gm__ uint8_t *)(GetRankWinAddrWithOffset(rankId_, dstOffset + roundByteOffset)));
            DataCopyPad(winDstGlobal, maskBuf, partialMaskCopyParams);
        } else {
            GlobalTensor<uint8_t> wsDstGlobal;
            wsDstGlobal.SetGlobalBuffer(
                (__gm__ uint8_t *)(params_.workspaceInfo.maskSlotPtr + srcOffset + roundByteOffset));
            DataCopyPad(wsDstGlobal, maskBuf, partialMaskCopyParams);
        }
        PipeBarrier<PIPE_ALL>();
    }
    return totalSendCnt;
}

// ======================================================================================================
// SendMaskCal：对本卡 topk 按通信域内所有专家id计算mask位，并发送至目标专家卡
// ------------------------------------------------------------------------------------------------------
//   大 bs 场景下 sendTotalNum 可能超出 UB 限制，因此按 roundSendTotalNum_ 分轮次计算：
//   Phase 1: 对每个全局专家，逐轮加载 topkIds 子段 → CompareScalar → 部分 mask；
//   Phase 2: 每轮部分 mask 拼写到 workspace/win 对应字节偏移，GatherMask 累加本轮 count；
//   Phase 3: 所有轮次完成后，写入 total count，URMA 发送完整 mask slot。
// ======================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendMaskCal()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t totalExperts = static_cast<int32_t>(worldSize_);
    // 按卡处理，只使用前 worldSize 个核。
    for (uint32_t curRankId = aivCoreIdx_; curRankId < static_cast<uint32_t>(totalExperts); curRankId += blockAivNum_) {
        ChannelHandle maskChannel{};
        if (curRankId != rankId_) {
            maskChannel = GetUrmaCommHandle(mc2Context_, curRankId, rankId_);
            // SendMask 与紧随其后的第一次全卡同步由同一物理 AIV 负责，锁跨两个阶段复用，
            // 避免在中间没有换核时提前 Unlock 刷新通道 counter。
            hcomm_.Lock(maskChannel);
        }
        for (uint32_t expertIdIndex = 0; expertIdIndex < moeExpertPerRank_; ++expertIdIndex) { // 按专家处理
            int32_t curExpertId = static_cast<int32_t>(curRankId * moeExpertPerRank_ + expertIdIndex);
            uint64_t srcOffset = static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) +
                                                       static_cast<int32_t>(curRankId)) *
                                 static_cast<uint64_t>(maskSlotSize_);
            uint64_t dstOffset =
                maskWinOffset_ + static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) +
                                                       static_cast<int32_t>(rankId_)) *
                                     static_cast<uint64_t>(maskSlotSize_);
            uint64_t totalSendCnt = BuildExpertMask(curRankId, curExpertId, srcOffset, dstOffset);
            // Phase 3: 写入 total count 并发送完整 mask slot
            if (curRankId == rankId_) {
                __gm__ int32_t *winCountPtr =
                    reinterpret_cast<__gm__ int32_t *>(GetRankWinAddrWithOffset(rankId_, dstOffset + maskAlignSize_));
                WriteGmByPassDCache(winCountPtr, static_cast<int32_t>(totalSendCnt));
                PipeBarrier<PIPE_ALL>();
            } else {
                __gm__ int32_t *wsCountPtr =
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.maskSlotPtr + srcOffset + maskAlignSize_);
                WriteGmByPassDCache(wsCountPtr, static_cast<int32_t>(totalSendCnt));
                PipeBarrier<PIPE_ALL>();
                GM_ADDR remoteDataAddr = GetRankWinAddrWithOffset(curRankId, dstOffset);
                GM_ADDR localGmAddr = params_.workspaceInfo.maskSlotPtr + srcOffset;
                hcomm_.WriteNbi(maskChannel, remoteDataAddr, localGmAddr, maskSlotSize_);
            }
        }
    }
    // SendMask 使用 direct WriteNbi，不创建批量 handle。先让本 AIV 负责的所有远端通道
    // 都拥有在途请求，再统一 Drain；锁继续保留给紧随其后的 CrossRankSync 复用。
    for (uint32_t curRankId = aivCoreIdx_; curRankId < static_cast<uint32_t>(totalExperts); curRankId += blockAivNum_) {
        if (curRankId == rankId_) {
            continue;
        }
        ChannelHandle maskChannel = GetUrmaCommHandle(mc2Context_, curRankId, rankId_);
        hcomm_.Drain(maskChannel);
    }
}

// ======================================================================
// LoadTopkWeightsToUb：权重搬运到UB（TopkWeightsPrefetch=0 时仅做 MTE2_V 同步）
// ======================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTopkWeightsToUb(
    const LocalTensor<ActivationType> &xOutTensor, int32_t currentOffset, int32_t index, TEventID event)
{
    if constexpr (TopkWeightsPrefetch) {
        QuantProcessConfig config{mxQuantTokenAlignBytes_, mxQuantScaleAlignBytes_, mxQuantTokenScaleAlignBytes_,
                                  mxQuantScaleNumAlignPerToken_};
        QuantProcessScratch<ActivationType> scratch{};
        scratch.mxTempTensor = mxTempTensor_;
        MegaMoeImpl::LoadTopkWeightsToUb<TopkWeightsType, ActivationType, TopkWeightsPrefetch>(
            params_, config, scratch, xOutTensor, currentOffset + index, event);
    } else {
        // Without weight prefetch, this event still waits for the input token copy.
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
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
    return (static_cast<uint64_t>(sourceServer) * static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank) +
            tokenId) *
           static_cast<uint64_t>(relayRecordBytes_);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RelayFlagOffset(uint32_t sourceServer,
                                                                                           uint32_t tokenId) const
{
    return (static_cast<uint64_t>(sourceServer) * static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank) +
            tokenId) *
           sizeof(uint64_t);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetContiguousGm(GM_ADDR dstAddr,
                                                                                         uint64_t sizeBytes)
{
    if (sizeBytes == 0 || resetBatchElementCount_ == 0) {
        return;
    }
    GlobalTensor<int32_t> dstGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dstAddr));

    uint64_t totalElements = sizeBytes / sizeof(int32_t);
    uint64_t elementsPerCore = Ops::Base::CeilDiv(totalElements, static_cast<uint64_t>(blockAivNum_));
    elementsPerCore = Ops::Base::CeilAlign(elementsPerCore, static_cast<uint64_t>(INT32_PER_256B));
    uint64_t coreOffset = static_cast<uint64_t>(aivCoreIdx_) * elementsPerCore;
    if (coreOffset >= totalElements) {
        return;
    }

    uint64_t coreElements = totalElements - coreOffset;
    coreElements = coreElements < elementsPerCore ? coreElements : elementsPerCore;
    for (uint64_t resetOffset = 0; resetOffset < coreElements; resetOffset += resetBatchElementCount_) {
        uint64_t remainingElements = coreElements - resetOffset;
        uint32_t currentElements =
            static_cast<uint32_t>(remainingElements < static_cast<uint64_t>(resetBatchElementCount_) ?
                                      remainingElements :
                                      static_cast<uint64_t>(resetBatchElementCount_));
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(currentElements * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(dstGm[coreOffset + resetOffset], resetTensor_, copyParams);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetDispatchState()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 队列在全 AIV 构建完成后一次性消费，只需清零每个目标 Server 的最终 count。
    for (uint32_t targetServer = aivCoreIdx_; targetServer < serverNum_; targetServer += blockAivNum_) {
        __gm__ int32_t *countPtr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchRelaySendQueuePtr +
                                                                      DispatchRelayQueueServerOffset(targetServer));
        WriteGmByPassDCache(countPtr, int32_t(0));
    }
    // 远端 ready flag 仍是一级数据到达 relay 后供二级 Dispatch 使用的完成协议。
    ResetContiguousGm(params_.peermemInfo.dispatchFlagPtr,
                      static_cast<uint64_t>(serverNum_) *
                          static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank) * sizeof(uint64_t));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::QuantizeTokenInUb(
    const LocalTensor<bfloat16_t> &input, const LocalTensor<ActivationType> &output,
    const LocalTensor<uint16_t> &scratch)
{
    __ubuf__ bfloat16_t *srcAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(input.GetPhyAddr());
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(scratch.GetPhyAddr());
    __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        scratch[Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr());
    __ubuf__ int8_t *outDataAddr = reinterpret_cast<__ubuf__ int8_t *>(output.GetPhyAddr());
    __ubuf__ uint16_t *mxScaleAddr =
        reinterpret_cast<__ubuf__ uint16_t *>(output[mxQuantTokenAlignBytes_].GetPhyAddr());
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
    WorkRange tokenRange = GetBalancedTokenRange(m_, aivCoreIdx_, blockAivNum_);
    uint32_t tokenNumInCore = tokenRange.count;
    uint32_t tokenStart = tokenRange.start;
    if (tokenNumInCore == 0U) {
        return;
    }

    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<uint8_t> workspaceDstGlobal;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params_.aGmAddr) +
                                    static_cast<uint64_t>(tokenStart) * k_);

    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(k_ * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0U; index < tokenNumInCore; ++index) {
        bool useFirstBuffer = (index & 1U) == 0U;
        TEventID event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<bfloat16_t> xInTensor = useFirstBuffer ? xInTensor1_ : xInTensor2_;
        LocalTensor<ActivationType> xOutTensor = useFirstBuffer ? xOutTensor1_ : xOutTensor2_;
        uint32_t tokenIdx = tokenStart + index;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor[static_cast<uint64_t>(index) * k_], xCopyInParams, xCopyInPadParams);
        LoadTopkWeightsToUb(xOutTensor, tokenStart, index, event);

        QuantizeTokenInUb(xInTensor, xOutTensor, mxTempTensor_);

        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        uint64_t relayOffset = RelayTokenOffset(serverId_, tokenIdx);
        GM_ADDR recordAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) + relayOffset;
        workspaceDstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(recordAddr));
        LocalTensor<uint8_t> xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(workspaceDstGlobal, xOutBytesTensor, {1U, mxQuantTokenScaleAlignBytes_, 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SetLocalRelayReadyFlags()
{
    uint32_t baseTokenNum = m_ / blockAivNum_;
    uint32_t tokenRemainder = m_ % blockAivNum_;
    uint32_t tokenNumInCore = baseTokenNum + static_cast<uint32_t>(aivCoreIdx_ < tokenRemainder);
    uint32_t tokenStart = aivCoreIdx_ * baseTokenNum + ((aivCoreIdx_ < tokenRemainder) ? aivCoreIdx_ : tokenRemainder);
    if (tokenNumInCore == 0U) {
        return;
    }

    Duplicate<uint64_t>(relayReadyFillTensor_, uint64_t(1), DISPATCH_READY_FLAG_BATCH_TOKENS);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    GlobalTensor<uint64_t> readyFlagGlobal;
    readyFlagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(params_.peermemInfo.dispatchFlagPtr) +
                                    static_cast<uint64_t>(serverId_) * params_.tilingData->numMaxTokensPerRank +
                                    tokenStart);
    for (uint32_t offset = 0U; offset < tokenNumInCore; offset += DISPATCH_READY_FLAG_BATCH_TOKENS) {
        uint32_t remaining = tokenNumInCore - offset;
        uint32_t currentTokenCount =
            remaining < DISPATCH_READY_FLAG_BATCH_TOKENS ? remaining : DISPATCH_READY_FLAG_BATCH_TOKENS;
        DataCopyPad(readyFlagGlobal[offset], relayReadyFillTensor_,
                    {1U, currentTokenCount * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U});
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendTokenToDispatchRelayQueues(
    GlobalTensor<int32_t> &topkIdsGlobal, uint32_t tokenIdx)
{
    uint32_t targetServers[DISPATCH_MAX_TOPK];
    uint32_t targetServerCount = 0U;
    for (uint32_t topkIdx = 0; topkIdx < topK_; ++topkIdx) {
        int32_t globalExpertId = topkIdsGlobal.GetValue(tokenIdx * topK_ + topkIdx);
        if (globalExpertId < 0 || globalExpertId >= static_cast<int32_t>(worldSize_ * moeExpertPerRank_)) {
            continue;
        }
        uint32_t targetRank = static_cast<uint32_t>(globalExpertId) / moeExpertPerRank_;
        uint32_t targetServer = targetRank / rankNumPerServer_;
        if (targetServer == serverId_) {
            continue;
        }
        bool duplicatedServer = false;
        for (uint32_t index = 0U; index < targetServerCount; ++index) {
            if (targetServers[index] == targetServer) {
                duplicatedServer = true;
                break;
            }
        }
        if (duplicatedServer) {
            continue;
        }
        targetServers[targetServerCount++] = targetServer;
    }

    for (uint32_t index = 0U; index < targetServerCount; ++index) {
        uint32_t targetServer = targetServers[index];
        __gm__ int32_t *countPtr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchRelaySendQueuePtr +
                                                                      DispatchRelayQueueServerOffset(targetServer));
        int32_t slotIdx = AtomicAdd(countPtr, int32_t(1));
        uint64_t metaOffset = DispatchRelayQueueServerOffset(targetServer) + ALIGN_32 +
                              static_cast<uint64_t>(slotIdx) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
        __gm__ int32_t *metaPtr =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchRelaySendQueuePtr + metaOffset);
        WriteGmByPassDCache(metaPtr, static_cast<int32_t>(tokenIdx));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildDispatchRelayQueues()
{
    // 全部物理 AIV 扫描不重叠的 token 区间；每个 token 在核内先折叠为唯一目标 Server 集合。
    uint32_t computeIdx = aivCoreIdx_;
    uint32_t baseTokenNum = m_ / blockAivNum_;
    uint32_t tokenRemainder = m_ % blockAivNum_;
    uint32_t tokenNumInCore = baseTokenNum + static_cast<uint32_t>(computeIdx < tokenRemainder);
    uint32_t tokenStart = computeIdx * baseTokenNum + ((computeIdx < tokenRemainder) ? computeIdx : tokenRemainder);
    uint32_t tokenEnd = tokenStart + tokenNumInCore;
    GlobalTensor<int32_t> topkIdsGlobal;
    topkIdsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.expertIdxGmAddr));
    for (uint32_t tokenIdx = tokenStart; tokenIdx < tokenEnd; ++tokenIdx) {
        AppendTokenToDispatchRelayQueues(topkIdsGlobal, tokenIdx);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendDispatchRelayData(
    uint32_t targetServer, uint32_t relayRank, HcommBatchHandle &batchHandle, int32_t slot)
{
    uint64_t metaOffset = DispatchRelayQueueServerOffset(targetServer) + ALIGN_32 +
                          static_cast<uint64_t>(slot) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    __gm__ int32_t *srcMetaPtr =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchRelaySendQueuePtr + metaOffset);
    int32_t tokenIdx = ReadGmByPassDCache(srcMetaPtr);
    uint64_t relayOffset = RelayTokenOffset(serverId_, static_cast<uint32_t>(tokenIdx));
    GM_ADDR srcAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) + relayOffset;
    GM_ADDR dstAddr = GetRankWinAddrWithOffset(relayRank, dispatchWinOffset_) + relayOffset;
    hcomm_.WriteNbi<DISPATCH_DATA_WQE_CONFIG>(batchHandle, dstAddr, srcAddr, mxQuantTokenScaleAlignBytes_);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendDispatchRelayFlag(
    uint32_t targetServer, uint32_t relayRank, HcommBatchHandle &batchHandle, int32_t slot, bool firstFlag)
{
    uint64_t metaOffset = DispatchRelayQueueServerOffset(targetServer) + ALIGN_32 +
                          static_cast<uint64_t>(slot) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    __gm__ int32_t *srcMetaPtr =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchRelaySendQueuePtr + metaOffset);
    int32_t tokenIdx = ReadGmByPassDCache(srcMetaPtr);
    uint64_t flagOffset = RelayFlagOffset(serverId_, static_cast<uint32_t>(tokenIdx));
    GM_ADDR localFlagAddr = GetRankWinAddrWithOffset(rankId_, dispatchFlagWinOffset_) + flagOffset;
    GM_ADDR remoteFlagAddr = GetRankWinAddrWithOffset(relayRank, dispatchFlagWinOffset_) + flagOffset;
    if (firstFlag) {
        hcomm_.WriteNbi<DISPATCH_FIRST_FLAG_WQE_CONFIG>(batchHandle, remoteFlagAddr, localFlagAddr, sizeof(uint64_t));
    } else {
        hcomm_.WriteNbi<DISPATCH_FLAG_WQE_CONFIG>(batchHandle, remoteFlagAddr, localFlagAddr, sizeof(uint64_t));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CommitDispatchRelayBatch(
    uint32_t targetServer, uint32_t relayRank, HcommBatchHandle &batchHandle, int32_t batchStart,
    uint32_t batchTokenCount)
{
    // 先提交本批全部 token 数据，再单独组装并提交 flag。首条 flag 使用 odr=6 建立批次边界；
    // 后续 flag 使用 odr=5。三种 WQE 均关闭 CQE，这一配置不外溢到二级读和 Combine。
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        AppendDispatchRelayData(targetServer, relayRank, batchHandle, batchStart + static_cast<int32_t>(index));
    }
    hcomm_.BatchCommit(batchHandle);
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        AppendDispatchRelayFlag(targetServer, relayRank, batchHandle, batchStart + static_cast<int32_t>(index),
                                index == 0U);
    }
    hcomm_.BatchCommit(batchHandle);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendDispatchRelayQueues()
{
    for (uint32_t targetServer = 0; targetServer < serverNum_; ++targetServer) {
        if (targetServer == serverId_) {
            continue;
        }
        uint32_t relayRank = targetServer * rankNumPerServer_ + rankIdInServer_;
        // 每个目标 Server 只对应一个一级中继 rank，统一归属映射保证仅该 AIV1 访问通道。
        // token 队列仍由所有计算 block 共享。
        if (relayRank >= worldSize_ || !IsChannelOwner(relayRank)) {
            continue;
        }

        ChannelHandle channel = GetUrmaCommHandle(mc2Context_, relayRank, rankId_);
        HcommBatchHandle batchHandle = hcomm_.MakeBatchHandle(
            channel, hcommBatchWqeTensor_, LAYERED_HCOMM_BATCH_UB_BYTES, GetRankWinAddrWithOffset(relayRank, 0));
        // 队列已经由全部 AIV 构建并同步完成，[0, count) 可直接按连续 slot 分批发送。
        __gm__ int32_t *countPtr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchRelaySendQueuePtr +
                                                                      DispatchRelayQueueServerOffset(targetServer));
        int32_t tokenCount = ReadGmByPassDCache(countPtr);
        for (int32_t batchStart = 0; batchStart < tokenCount;
             batchStart += static_cast<int32_t>(DISPATCH_SEND_BATCH_TOKEN_CAPACITY)) {
            int32_t remaining = tokenCount - batchStart;
            uint32_t batchTokenCount =
                static_cast<uint32_t>(remaining < static_cast<int32_t>(DISPATCH_SEND_BATCH_TOKEN_CAPACITY) ?
                                          remaining :
                                          static_cast<int32_t>(DISPATCH_SEND_BATCH_TOKEN_CAPACITY));
            CommitDispatchRelayBatch(targetServer, relayRank, batchHandle, batchStart, batchTokenCount);
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTokenFromLocalRelay(uint32_t srcServer,
                                                                                               uint32_t tokenIndex,
                                                                                               int32_t bufferIdx,
                                                                                               uint32_t copyInNum)
{
    __gm__ uint64_t *readyFlag = reinterpret_cast<__gm__ uint64_t *>(params_.peermemInfo.dispatchFlagPtr +
                                                                     RelayFlagOffset(srcServer, tokenIndex));
    while (ReadGmByPassDCache(readyFlag) != uint64_t(1)) {
    }

    uint64_t remoteCopyOffset = RelayTokenOffset(srcServer, tokenIndex);
    GM_ADDR localRecordAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) + remoteCopyOffset;
    GlobalTensor<ActivationType> relayGlobalTensor;
    relayGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(localRecordAddr));
    DataCopy(copyTmpTensors_[bufferIdx], relayGlobalTensor, copyInNum);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
}

// ==================================================================================================
// 计算并发布当前专家接收的 token 数。
// --------------------------------------------------------------------------------------------------
//   Phase 1: 逐 rank 从本卡 win 加载单个 [mask|count] 槽位，提取 count 并累加 cumsum；
//   Phase 2: 写 expertRevNumsGlobalTensor_ + AtomicAdd 通知 AIC;
// ==================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishExpertTokenCount(uint32_t expertIdx)
{
    uint64_t tokenCount = 0U;
    uint32_t expertRankOffset = expertIdx * worldSize_;
    // 逐 rank 直接从 GM 读取 count 并累计全局 cumsum。
    for (uint32_t srcRank = 0U; srcRank < worldSize_; ++srcRank) {
        __gm__ int32_t *rankCountPtr = reinterpret_cast<__gm__ int32_t *>(
            params_.peermemInfo.maskRecvPtr + static_cast<uint64_t>(expertIdx) * worldSize_ * maskSlotSize_ +
            static_cast<uint64_t>(srcRank) * maskSlotSize_ + maskAlignSize_);
        int32_t rankTokenCount = ReadGmByPassDCache(rankCountPtr);
        tokenCount += static_cast<uint64_t>(rankTokenCount);
        cumsumRevCntInRank_ += static_cast<uint64_t>(rankTokenCount);
        cumsumInfoTensor_.SetValue(expertRankOffset + srcRank, static_cast<int32_t>(cumsumRevCntInRank_));
    }

    // 仅写当前专家的 cumsum slice；随后发布 token 数和 AIC ready flag。
    expertTokenCntTensor_.SetValue(0, static_cast<int32_t>(tokenCount));
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(expertRevNumsGlobalTensor_[expertIdx * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_],
                      expertTokenCntTensor_, INT32_PER_256B);
    uint32_t cumsumCopyBytes = static_cast<uint32_t>(worldSize_ * sizeof(int32_t));
    if (cumsumCopyBytes % static_cast<uint32_t>(ALIGN_32) == 0U) {
        DataCopyPad(cumsumInfoGlobalTensor_[expertRankOffset], cumsumInfoTensor_[expertRankOffset],
                    {1U, cumsumCopyBytes, 0U, 0U, 0U});
    } else {
        // Packed expert slices are not UB-aligned when worldSize_ is not a multiple of 8.
        uint64_t cumsumStride =
            Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), ALIGN_32);
        __gm__ int32_t *cumsumDst =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr + cumsumStride * blockIdx_) +
            expertRankOffset;
        for (uint32_t srcRank = 0U; srcRank < worldSize_; ++srcRank) {
            WriteGmByPassDCache(cumsumDst + srcRank, cumsumInfoTensor_.GetValue(expertRankOffset + srcRank));
        }
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID2>();
    PipeBarrier<PIPE_ALL>();
    if constexpr (TopkWeightsPrefetch) {
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    }

    __gm__ int32_t *tokenCountReadyFlag =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
        static_cast<uint64_t>(expertIdx) * aicNum_ * INT_CACHELINE + static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    AscendC::AtomicAdd(tokenCountReadyFlag, static_cast<int32_t>(1));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateLocalRelayWeight(int32_t copyIdx,
                                                                                              int32_t outIdx,
                                                                                              TEventID eventId)
{
    if constexpr (TopkWeightsPrefetch) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        LocalTensor<ActivationType> weightBuf = copyTmpTensors_[outIdx];
        uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
        LocalTensor<int32_t> bufWeightsInt32 = weightBuf[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t topkIndex = metaInfoTensor_[copyIdx * INT32_PER_256B].GetValue(TOPK_INDEX);
        int32_t weightBits = bufWeightsInt32.GetValue(static_cast<uint32_t>(topkIndex));
        metaInfoTensor_[copyIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadLocalRelayToken(
    uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx, uint32_t copyInNum, TEventID eventId)
{
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    LoadTokenFromLocalRelay(srcServer, tokenIndex, bufferIdx, copyInNum);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
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
    constexpr TEventID kBufEvents[DISPATCH_COPY_BUFFER_COUNT] = {EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        params_.workspaceInfo.dispatchRevDataPtr +
        static_cast<uint64_t>(rowDstOffsetInCore) * widthA * sizeof(ActivationType)));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr +
        static_cast<uint64_t>(rowDstOffsetInCore) * widthAScale * sizeof(QuantScaleOutType)));

    for (int32_t bufferIdx = 0; bufferIdx < DISPATCH_COPY_BUFFER_COUNT; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }

    int32_t primeCount = copyNum < DISPATCH_COPY_BUFFER_COUNT ? copyNum : DISPATCH_COPY_BUFFER_COUNT;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t tokenIndex = metaInfoTensor_[primeIdx * INT32_PER_256B].GetValue(TOKEN_ID);
        TEventID eventId = kBufEvents[primeIdx];
        LoadLocalRelayToken(srcServer, static_cast<uint32_t>(tokenIndex), primeIdx, copyInNum, eventId);
    }

    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t outIdx = copyIdx % DISPATCH_COPY_BUFFER_COUNT;
        TEventID eventId = kBufEvents[outIdx];
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

        UpdateLocalRelayWeight(copyIdx, outIdx, eventId);
        LocalTensor<ActivationType> tokenScaleBuf = copyTmpTensors_[outIdx];
        LocalTensor<QuantScaleOutType> scaleBuf =
            tokenScaleBuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScaleBuf,
                    {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], scaleBuf,
                    {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

        int32_t nextIdx = copyIdx + DISPATCH_COPY_BUFFER_COUNT;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = metaInfoTensor_[nextIdx * INT32_PER_256B].GetValue(TOKEN_ID);
            // 等待本轮 MTE3 完成后再复用 outIdx 槽。
            LoadLocalRelayToken(srcServer, static_cast<uint32_t>(tokenIndex), outIdx, copyInNum, eventId);
        }
    }

    for (int32_t bufferIdx = 0; bufferIdx < DISPATCH_COPY_BUFFER_COUNT; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishDispatchRows(uint32_t expertIdx,
                                                                                           int32_t globalRowStart,
                                                                                           int32_t rowCount)
{
    if (rowCount <= 0) {
        return;
    }
    int32_t priorExpertRows = expertIdx == 0U ? 0 : cumsumInfoTensor_.GetValue(expertIdx * worldSize_ - 1U);
    int32_t localRow = globalRowStart - priorExpertRows;
    int32_t remainingRows = rowCount;
    __gm__ int32_t *expertFlagBase = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagDispatchToGmm1Ptr) +
                                     static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_;
    while (remainingRows > 0) {
        int32_t waveIdx = localRow / static_cast<int32_t>(L1_TILE_M_256);
        int32_t rowsInWave = static_cast<int32_t>(L1_TILE_M_256) - localRow % static_cast<int32_t>(L1_TILE_M_256);
        if (rowsInWave > remainingRows) {
            rowsInWave = remainingRows;
        }
        AtomicAdd(expertFlagBase + static_cast<int64_t>(waveIdx) * INT_CACHELINE, rowsInWave);
        localRow += rowsInWave;
        remainingRows -= rowsInWave;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::FindDispatchSegmentEnd(
    uint32_t segmentStart, uint32_t batchTokenCount)
{
    LocalTensor<int32_t> firstRecord = metaInfoTensor_[segmentStart * INT32_PER_256B];
    uint32_t expertIdx = static_cast<uint32_t>(firstRecord.GetValue(DISPATCH_BATCH_EXPERT_ID));
    int32_t dstRowStart = firstRecord.GetValue(DISPATCH_BATCH_DST_ROW);
    uint32_t segmentEnd = segmentStart + 1U;
    while (segmentEnd < batchTokenCount) {
        LocalTensor<int32_t> nextRecord = metaInfoTensor_[segmentEnd * INT32_PER_256B];
        if (static_cast<uint32_t>(nextRecord.GetValue(DISPATCH_BATCH_EXPERT_ID)) != expertIdx ||
            nextRecord.GetValue(DISPATCH_BATCH_DST_ROW) !=
                dstRowStart + static_cast<int32_t>(segmentEnd - segmentStart)) {
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
    if (batchTokenCount == 0U) {
        return;
    }

    GM_ADDR scratchAddr = params_.workspaceInfo.dispatchRemoteReadyFlagSnapshotPtr +
                          static_cast<uint64_t>(blockIdx_) * dispatchRelayFlagSnapshotBytesPerBlock_;
    GlobalTensor<uint64_t> scratchFlagGlobalTensor;
    scratchFlagGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(scratchAddr));
    DataCopyPadExtParams<uint64_t> flagCopyPadParams{true, 0U, 0U, 0U};
    ChannelHandle channel = GetUrmaCommHandle(mc2Context_, relayRank, rankId_);

    // 批次可以跨专家和源 Server，但不能跨 relay 通道。ready flag 按 token-id 的 256 槽窗口读取，
    // 只处理当前批次实际命中的窗口。token id 属于源 Rank，上界必须用全卡容量而非本卡 m_。
    for (uint32_t srcServer = 0U; srcServer < serverNum_; ++srcServer) {
        for (uint64_t windowStart64 = 0U;
             windowStart64 < static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank);
             windowStart64 += DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) {
            uint32_t windowStart = static_cast<uint32_t>(windowStart64);
            uint64_t windowEnd64 = windowStart64 + DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY;
            if (windowEnd64 > static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank)) {
                windowEnd64 = static_cast<uint64_t>(params_.tilingData->numMaxTokensPerRank);
            }
            uint32_t windowEnd = static_cast<uint32_t>(windowEnd64);
            bool windowHasToken = false;
            for (uint32_t index = 0U; index < batchTokenCount; ++index) {
                LocalTensor<int32_t> record = metaInfoTensor_[index * INT32_PER_256B];
                if (static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_SRC_SERVER)) != srcServer) {
                    continue;
                }
                uint32_t tokenIndex = static_cast<uint32_t>(record.GetValue(TOKEN_ID));
                if (tokenIndex >= windowStart && tokenIndex < windowEnd) {
                    windowHasToken = true;
                    break;
                }
            }
            if (!windowHasToken) {
                continue;
            }

            uint64_t flagSnapshotBytes = static_cast<uint64_t>(windowEnd - windowStart) * sizeof(uint64_t);
            DataCopyExtParams flagCopyParams{1U, static_cast<uint32_t>(flagSnapshotBytes), 0U, 0U, 0U};
            GM_ADDR remoteFlagAddr =
                GetRankWinAddrWithOffset(relayRank, dispatchFlagWinOffset_) + RelayFlagOffset(srcServer, windowStart);
            bool windowTokensReady = false;
            while (!windowTokensReady) {
                hcomm_.ReadNbi(channel, scratchAddr, remoteFlagAddr, flagSnapshotBytes);
                hcomm_.Drain(channel);
                SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID1>();
                DataCopyPad(relayFlagTensor_, scratchFlagGlobalTensor, flagCopyParams, flagCopyPadParams);
                SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID1>();

                windowTokensReady = true;
                for (uint32_t index = 0U; index < batchTokenCount; ++index) {
                    LocalTensor<int32_t> record = metaInfoTensor_[index * INT32_PER_256B];
                    if (static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_SRC_SERVER)) != srcServer) {
                        continue;
                    }
                    uint32_t tokenIndex = static_cast<uint32_t>(record.GetValue(TOKEN_ID));
                    if (tokenIndex >= windowStart && tokenIndex < windowEnd &&
                        relayFlagTensor_.GetValue(tokenIndex - windowStart) != uint64_t(1)) {
                        windowTokensReady = false;
                        break;
                    }
                }
            }
        }
    }

    HcommBatchHandle batchHandle = hcomm_.MakeBatchHandle(channel, hcommBatchWqeTensor_, LAYERED_HCOMM_BATCH_UB_BYTES,
                                                          GetRankWinAddrWithOffset(relayRank, 0));
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        LocalTensor<int32_t> record = metaInfoTensor_[index * INT32_PER_256B];
        uint32_t srcServer = static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_SRC_SERVER));
        uint32_t tokenIndex = static_cast<uint32_t>(record.GetValue(TOKEN_ID));
        uint32_t dstRow = static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_DST_ROW));
        GM_ADDR remoteRecordAddr =
            GetRankWinAddrWithOffset(relayRank, dispatchWinOffset_) + RelayTokenOffset(srcServer, tokenIndex);
        GM_ADDR tokenDstAddr =
            params_.workspaceInfo.dispatchRevDataPtr + static_cast<uint64_t>(dstRow) * widthA * sizeof(ActivationType);
        GM_ADDR scaleDstAddr = params_.workspaceInfo.dispatchRevScalePtr +
                               static_cast<uint64_t>(dstRow) * widthAScale * sizeof(QuantScaleOutType);
        hcomm_.ReadNbi(batchHandle, tokenDstAddr, remoteRecordAddr, widthA * sizeof(ActivationType));
        hcomm_.ReadNbi(batchHandle, scaleDstAddr, remoteRecordAddr + mxQuantTokenAlignBytes_,
                       widthAScale * sizeof(QuantScaleOutType));
        if constexpr (TopkWeightsPrefetch) {
            GM_ADDR weightDstAddr =
                params_.workspaceInfo.dispatchRevWeightsPtr + static_cast<uint64_t>(dstRow) * weightAlignBytes_;
            hcomm_.ReadNbi(batchHandle, weightDstAddr,
                           remoteRecordAddr + mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, weightAlignBytes_);
        }
    }
    hcomm_.BatchCommit(batchHandle);
    hcomm_.Drain(batchHandle);

    if constexpr (TopkWeightsPrefetch) {
        for (uint32_t index = 0U; index < batchTokenCount; ++index) {
            LocalTensor<int32_t> record = metaInfoTensor_[index * INT32_PER_256B];
            uint32_t dstRow = static_cast<uint32_t>(record.GetValue(DISPATCH_BATCH_DST_ROW));
            uint32_t topkIndex = static_cast<uint32_t>(record.GetValue(TOPK_INDEX));
            __gm__ int32_t *weightGmI32 = reinterpret_cast<__gm__ int32_t *>(
                params_.workspaceInfo.dispatchRevWeightsPtr + static_cast<uint64_t>(dstRow) * weightAlignBytes_);
            record.SetValue(WEIGHT_INDEX, ReadGmByPassDCache(weightGmI32 + topkIndex));
        }
    }

    // 每个 expert/rank 段内的 metadata 连续。先写完全部 metadata，再发布专家就绪计数。
    uint32_t segmentStart = 0U;
    while (segmentStart < batchTokenCount) {
        LocalTensor<int32_t> firstRecord = metaInfoTensor_[segmentStart * INT32_PER_256B];
        int32_t dstRowStart = firstRecord.GetValue(DISPATCH_BATCH_DST_ROW);
        uint32_t segmentEnd = FindDispatchSegmentEnd(segmentStart, batchTokenCount);
        DataCopy(metaInfoGlobalTensor_[static_cast<uint64_t>(dstRowStart) * INT32_PER_256B],
                 metaInfoTensor_[segmentStart * INT32_PER_256B], (segmentEnd - segmentStart) * INT32_PER_256B);
        segmentStart = segmentEnd;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();

    segmentStart = 0U;
    while (segmentStart < batchTokenCount) {
        LocalTensor<int32_t> firstRecord = metaInfoTensor_[segmentStart * INT32_PER_256B];
        uint32_t expertIdx = static_cast<uint32_t>(firstRecord.GetValue(DISPATCH_BATCH_EXPERT_ID));
        int32_t dstRowStart = firstRecord.GetValue(DISPATCH_BATCH_DST_ROW);
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
    uint32_t copyInNum = mxQuantTokenScaleAlignBytes_;
    uint32_t srcServer = static_cast<uint32_t>(srcRank) / rankNumPerServer_;

    for (int32_t i = 0; i < tokenCount; ++i) {
        int32_t topkIndex = validTopkIndexTensor_.GetValue(tokenOffset + i);
        int32_t tokenIndex = topkIndex / topK_;
        metaInfoTensor_[i * INT32_PER_256B].SetValue(RANK_ID, srcRank);
        metaInfoTensor_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        metaInfoTensor_[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
    }

    CopyTokensFromLocalRelay(dstRow, srcServer, tokenCount, widthA, widthAScale, copyInNum);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    DataCopy(metaInfoGlobalTensor_[dstRow * INT32_PER_256B], metaInfoTensor_, tokenCount * INT32_PER_256B);
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
}

// 每个 relay 通道建立一条跨专家接收流；完整批次固定为 256 token，仅宏 Wave 尾批不足 256。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveDispatchExpertRange(uint32_t expertBegin,
                                                                                                  uint32_t expertEnd)
{
    if (expertBegin >= expertEnd || expertBegin >= moeExpertPerRank_) {
        return;
    }
    expertEnd = expertEnd < moeExpertPerRank_ ? expertEnd : moeExpertPerRank_;
    DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                {1U, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();

    metaInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, ubBufferUsedAddr_, L1_TILE_M_256 * INT32_PER_256B);
    for (uint32_t srcRankInServer = 0U; srcRankInServer < rankNumPerServer_; ++srcRankInServer) {
        uint32_t relayRank = serverId_ * rankNumPerServer_ + srcRankInServer;
        if (relayRank >= worldSize_ || !IsChannelOwner(relayRank)) {
            continue;
        }
        bool isRemoteRelay = relayRank != rankId_;
        uint32_t remoteBatchTokenCount = 0U;

        for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
            for (uint32_t srcServer = 0U; srcServer < serverNum_; ++srcServer) {
                uint32_t srcRank = srcServer * rankNumPerServer_ + srcRankInServer;
                if (srcRank >= worldSize_) {
                    continue;
                }

                int32_t rowStart = (srcRank == 0U && expertIdx == 0U) ?
                                       0 :
                                       cumsumInfoTensor_.GetValue(expertIdx * worldSize_ + srcRank - 1U);
                if (rowStart >= static_cast<int32_t>(maxOutputSize_)) {
                    continue;
                }
                __gm__ uint8_t *rankMaskBasePtr = reinterpret_cast<__gm__ uint8_t *>(
                    params_.peermemInfo.maskRecvPtr + static_cast<uint64_t>(expertIdx) * worldSize_ * maskSlotSize_ +
                    static_cast<uint64_t>(srcRank) * maskSlotSize_);
                __gm__ int32_t *rankCountPtr = reinterpret_cast<__gm__ int32_t *>(rankMaskBasePtr + maskAlignSize_);
                const int32_t rankTokenCount = ReadGmByPassDCache(rankCountPtr);
                if (rankTokenCount <= 0) {
                    continue;
                }

                int32_t accumulatedRows = 0;
                for (uint32_t roundIdx = 0U; roundIdx < dispatchTotalRounds_ && accumulatedRows < rankTokenCount;
                     ++roundIdx) {
                    uint64_t roundStart =
                        static_cast<uint64_t>(roundIdx) * static_cast<uint64_t>(dispatchRoundSendTotalNum_);
                    uint32_t roundLen = (roundIdx + 1U < dispatchTotalRounds_) ?
                                            dispatchRoundSendTotalNum_ :
                                            static_cast<uint32_t>(maskRouteCapacity_ - roundStart);
                    uint32_t topkIndexTensorElemCount =
                        static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<int64_t>(roundLen * sizeof(int32_t)),
                                                                   static_cast<int64_t>(ALIGN_32))) /
                        sizeof(int32_t);
                    uint32_t curRoundMaskBytes = (roundLen == dispatchRoundSendTotalNum_) ?
                                                     dispatchRoundMaskAlignSize_ :
                                                     static_cast<uint32_t>(Ops::Base::CeilAlign(
                                                         static_cast<int64_t>(Ops::Base::CeilDiv(roundLen, 8U)),
                                                         static_cast<int64_t>(ALIGN_32)));
                    uint64_t roundMaskByteOffset = roundStart / 8U;
                    GlobalTensor<uint8_t> roundMaskSrc;
                    roundMaskSrc.SetGlobalBuffer(rankMaskBasePtr + roundMaskByteOffset);
                    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
                    DataCopy(gatherMaskTensor_, roundMaskSrc, curRoundMaskBytes);
                    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
                    CreateVecIndex(topkIndexTensor_, static_cast<int32_t>(roundStart), topkIndexTensorElemCount);

                    uint64_t roundTokenCount = 0U;
                    LocalTensor<uint32_t> rankMaskSlice = gatherMaskInt32Tensor_[0];
                    GatherMask(validTopkIndexTensor_, topkIndexTensor_, rankMaskSlice, true, roundLen, {1, 1, 0, 0},
                               roundTokenCount);
                    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
                    if (roundTokenCount == 0U) {
                        continue;
                    }

                    // 发送方只覆写真实 bs 的 mask 前缀，容量尾部可能保留上次调用数据。
                    // mask 命中下标单调递增，因此每轮只消费 count 剩余量的前缀即可忽略尾部残留。
                    const int32_t remainingRankRows = rankTokenCount - accumulatedRows;
                    int32_t validRoundCount = static_cast<int32_t>(roundTokenCount);
                    if (validRoundCount > remainingRankRows) {
                        validRoundCount = remainingRankRows;
                    }
                    int32_t roundCopyCount = validRoundCount;
                    if (rowStart + accumulatedRows + roundCopyCount > static_cast<int32_t>(maxOutputSize_)) {
                        roundCopyCount = static_cast<int32_t>(maxOutputSize_) - rowStart - accumulatedRows;
                    }
                    if (roundCopyCount <= 0) {
                        accumulatedRows += validRoundCount;
                        continue;
                    }

                    if (!isRemoteRelay) {
                        for (int32_t chunkStart = 0; chunkStart < roundCopyCount;
                             chunkStart += static_cast<int32_t>(L1_TILE_M_256)) {
                            int32_t chunkRows = roundCopyCount - chunkStart;
                            if (chunkRows > static_cast<int32_t>(L1_TILE_M_256)) {
                                chunkRows = static_cast<int32_t>(L1_TILE_M_256);
                            }
                            int32_t dstRow = rowStart + accumulatedRows + chunkStart;
                            CopyLocalDispatchTokens(dstRow, static_cast<int32_t>(srcRank), chunkStart, chunkRows);
                            PublishDispatchRows(expertIdx, dstRow, chunkRows);
                        }
                    } else {
                        for (int32_t tokenOffset = 0; tokenOffset < roundCopyCount; ++tokenOffset) {
                            int32_t topkIndex = validTopkIndexTensor_.GetValue(tokenOffset);
                            int32_t dstRow = rowStart + accumulatedRows + tokenOffset;
                            LocalTensor<int32_t> record = metaInfoTensor_[remoteBatchTokenCount * INT32_PER_256B];
                            record.SetValue(RANK_ID, static_cast<int32_t>(srcRank));
                            record.SetValue(TOKEN_ID, topkIndex / static_cast<int32_t>(topK_));
                            record.SetValue(TOPK_INDEX, topkIndex % static_cast<int32_t>(topK_));
                            record.SetValue(WEIGHT_INDEX, 0);
                            record.SetValue(DISPATCH_BATCH_DST_ROW, dstRow);
                            record.SetValue(DISPATCH_BATCH_EXPERT_ID, static_cast<int32_t>(expertIdx));
                            record.SetValue(DISPATCH_BATCH_SRC_SERVER, static_cast<int32_t>(srcServer));
                            ++remoteBatchTokenCount;
                            if (remoteBatchTokenCount == DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) {
                                ReceiveRemoteDispatchBatch(relayRank, remoteBatchTokenCount);
                                remoteBatchTokenCount = 0U;
                            }
                        }
                    }
                    accumulatedRows += validRoundCount;
                }
            }
        }

        // 扫描完当前宏 Wave 后，仅通道尾批允许不足 256 token。
        if (isRemoteRelay && remoteBatchTokenCount != 0U) {
            ReceiveRemoteDispatchBatch(relayRank, remoteBatchTokenCount);
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline auto MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitSharedExpertInputBuffers()
    -> QuantProcessScratch<ActivationType>
{
    uint32_t xInAddr = ALIGN_512;
    uint32_t xInSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    LocalTensor<bfloat16_t> xInBuf0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAddr, xInSize / sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> xInBuf1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAddr + xInSize, xInSize / sizeof(bfloat16_t));
    uint32_t mxTempAddr = xInAddr + xInSize * 2;
    LocalTensor<uint16_t> mxTempBuf =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempAddr, MX_QUANT_TEMP_UB_BYTES / sizeof(uint16_t));
    uint32_t xOutAddr = mxTempAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t xOutSize =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    LocalTensor<ActivationType> xOutBuf0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutAddr, xOutSize / sizeof(ActivationType));
    LocalTensor<ActivationType> xOutBuf1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutAddr + xOutSize, xOutSize / sizeof(ActivationType));
    return {xInBuf0, xInBuf1, xOutBuf0, xOutBuf1, mxTempBuf};
}

// ===============================================================
// SharedExpertCopyInput：从原始 bf16 输入量化后写入共享专家专用缓冲区
//   源: aGmAddr [bs × h] bf16（layered URMA 模式下 quantTokenScalePtr 未填充，需直接从原始输入量化）
//   目标: sharedExpertInputDataPtr [bs × h] fp8 连续, sharedExpertInputScalePtr [bs × scaleN] 连续
//   AIV 执行，在 AIC GMM1 开始前调用
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SharedExpertCopyInput()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t currentNum;
    int32_t currentOffset;
    TilingByCore(m_, currentNum, currentOffset, 1);

    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    auto scratch = InitSharedExpertInputBuffers();
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<ActivationType> dataDstGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleDstGlobalTensor;
    dataDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params_.workspaceInfo.sharedExpertInputDataPtr));
    scaleDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleOutType *>(params_.workspaceInfo.sharedExpertInputScalePtr));

    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(k_ * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        int32_t tokenIdx = currentOffset + index;
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInBuf = (index % DOUBLE_BUFFER == 0) ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutBuf = (index % DOUBLE_BUFFER == 0) ? scratch.xOutTensor0 : scratch.xOutTensor1;

        srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            params_.aGmAddr + static_cast<uint64_t>(tokenIdx) * k_ * sizeof(bfloat16_t)));
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInBuf, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);

        QuantizeTokenInUb(xInBuf, xOutBuf, scratch.mxTempTensor);

        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        LocalTensor<QuantScaleOutType> bufScale =
            xOutBuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(dataDstGlobalTensor[tokenIdx * widthA], xOutBuf,
                    {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleDstGlobalTensor[tokenIdx * widthAScale], bufScale,
                    {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareDispatch(uint32_t firstWaveExpertCount)
{
    // 全量量化和跨全部专家的目标 Server 队列已在 Wave 前由全部 AIV 完成。
    // Wave 入口仅发布专家计数并一次性提交一级通信，一级批次不由计算 Wave 截断。
    for (uint32_t expertIdx = 0U; expertIdx < firstWaveExpertCount; ++expertIdx) {
        PublishExpertTokenCount(expertIdx);
    }
    // 先提交一级大批次，再计算后续专家计数；BatchCommit 后网络传输可与本地计数重叠。
    SendDispatchRelayQueues();
    for (uint32_t expertIdx = firstWaveExpertCount; expertIdx < moeExpertPerRank_; ++expertIdx) {
        PublishExpertTokenCount(expertIdx);
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_LAYERED_DISPATCH_H
