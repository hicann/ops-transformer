/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_COMBINE_H
#define MEGA_MOE_LAYERED_COMBINE_H

// Internal member definitions; included by mega_moe_layered.h after the class declaration.
namespace MegaMoeImpl {

// =============================================
// ResetGmm2CombineSyncCounters：重置 GMM2→Combine 同步计数器
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetGmm2CombineSyncCounters()
{
    if constexpr (g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(params_.tilingData->combineSyncSlotCountPerExpert *
                                                     moeExpertPerRank_ * static_cast<uint64_t>(INT_CACHELINE));
        int32_t coreLen, coreOffset;
        TilingByCore(totalCounters, coreLen, coreOffset);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.gmm2CombineSyncCounterPtr);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
        for (int32_t resetElementOffset = 0; resetElementOffset < coreLen;
             resetElementOffset += resetBatchElementCount_) {
            int32_t currentBatchElementCount = coreLen - resetElementOffset < resetBatchElementCount_ ?
                                                   coreLen - resetElementOffset :
                                                   resetBatchElementCount_;
            DataCopyExtParams counterCopyParams{1U, static_cast<uint32_t>(currentBatchElementCount * sizeof(int32_t)),
                                                0U, 0U, 0U};
            DataCopyPad(gmm2CombineSyncCounterGm[coreOffset + resetElementOffset], resetTensor_, counterCopyParams);
        }
    }
}

// =============================================
// InitCombineBuffers：初始化 Combine 所需的 buffer 大小
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitCombineBuffers()
{
    if constexpr (g_coreType == AIV) {
        // Combine 仅在当前 AIV1 完成 Dispatch 张量使用后启动，并从 UB 首地址重建视图。
        // Hcomm 使用前 512 字节；Combine metadata 和计算临时区从其后开始复用 UB。
        LocalTensor<uint8_t> hcommTensor = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
        hcomm_.Init(hcommTensor, ALIGN_512);
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GetCombineRankRange(uint32_t expertIdx,
                                                                                           uint32_t dstRank,
                                                                                           uint32_t &rowStart,
                                                                                           uint32_t &tokenCount)
{
    uint64_t expertBegin = static_cast<uint64_t>(expertIdx) * worldSize_;
    uint64_t rankEndIndex = expertBegin + dstRank;
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        cumsumInfoGlobalTensor_[rankEndIndex]);
    int32_t rankEnd = cumsumInfoGlobalTensor_.GetValue(rankEndIndex);

    int32_t expertStart = 0;
    if (expertBegin != 0U) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            cumsumInfoGlobalTensor_[expertBegin - 1U]);
        expertStart = cumsumInfoGlobalTensor_.GetValue(expertBegin - 1U);
    }
    int32_t rankStart = expertStart;
    if (dstRank != 0U) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            cumsumInfoGlobalTensor_[rankEndIndex - 1U]);
        rankStart = cumsumInfoGlobalTensor_.GetValue(rankEndIndex - 1U);
    }
    rowStart = static_cast<uint32_t>(rankStart - expertStart);
    tokenCount = static_cast<uint32_t>(rankEnd - rankStart);
    return tokenCount != 0U;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineRank(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State, uint32_t expertIdx,
    CombineImpl::LayeredCombineBatchState &batchState)
{
    constexpr bool IsQuantized = CombineQuantMode != COMBINE_NO_QUANT;
    uint32_t rowStart = 0U;
    uint32_t tokenCount = 0U;
    if (!GetCombineRankRange(expertIdx, batchState.dstRankId, rowStart, tokenCount)) {
        return;
    }
    AscendC::SetCtrlSpr<60, 60>(0);

    uint32_t mExpert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
    GroupSyncSlotLayout slotLayout = CalcGroupSyncSlotLayout(mExpert, blockNum_);
    __gm__ int32_t *expertCounterBase = reinterpret_cast<__gm__ int32_t *>(gmmAddrInfo.gmm2CombineSyncCounter);

    uint32_t processedCount = 0U;
    while (processedCount < tokenCount) {
        uint32_t currentRow = rowStart + processedCount;
        uint32_t targetGroup = currentRow / L1_TILE_M_256;
        uint32_t firstSyncSlot = 0;
        uint32_t syncSlotCount = 0;
        GetGroupSyncSlotRange(targetGroup, slotLayout, firstSyncSlot, syncSlotCount);
        __gm__ int32_t *counterAddr = GetCombineSyncCounterAddress(expertCounterBase, firstSyncSlot);
        if (AscendC::ReadGmByPassDCache(counterAddr) < static_cast<int32_t>(nTilesPerGroup)) {
            continue;
        }

        uint32_t groupEndRow = (targetGroup + 1U) * L1_TILE_M_256;
        groupEndRow = groupEndRow < mExpert ? groupEndRow : mExpert;
        uint32_t batchCount = groupEndRow - currentRow;
        uint32_t remainingTokens = tokenCount - processedCount;
        batchCount = batchCount < remainingTokens ? batchCount : remainingTokens;

        int64_t offset = ALIGN_512;
        LocalTensor<int32_t> metaInfoTensor(TPosition::VECIN, offset, batchCount * META_INFO_SIZE);
        offset += batchCount * META_INFO_SIZE * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> metaInfoGm;
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.metaInfoPtr +
            static_cast<uint64_t>(gmm2State.expertBeforeCnt + currentRow) * META_INFO_SIZE * sizeof(int32_t)));
        SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID1>();
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, batchCount * META_INFO_SIZE);

        CombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t, true, IsQuantized>(
            currentRow, batchCount, k_, expertIdx, rankId_, gmmAddrInfo.gmm2OutGlobal, params_, metaInfoTensor,
            combineUbTensorSize_, offset, quantTokenSizeBytes, batchState);
        processedCount += batchCount;
    }
}

// 按目标 rank 跨专家积累 Combine 批次，并在宏 Wave 尾提交不足 256-token 的尾批。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineExpertRange(
    ExpertLoopState waveBeginState, uint32_t expertBegin, uint32_t expertEnd)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1U || expertBegin >= expertEnd) {
        return;
    }

    // 每个目标 rank 在一个计算宏 Wave 内只创建一个 batch handle，WQE 可跨专家积累。
    // 满 256 token 提交，宏 Wave 尾不足 256 的部分立即 flush，保证下游不会为凑包等待。
    for (uint32_t dstRank = 0U; dstRank < worldSize_; ++dstRank) {
        if (!IsChannelOwner(dstRank)) {
            continue;
        }
        CombineImpl::LayeredCombineBatchState batchState{};
        batchState.dstRankId = dstRank;
        if (dstRank != rankId_) {
            ChannelHandle channel = GetUrmaCommHandle(mc2Context_, dstRank, rankId_);
            batchState.batchHandle = hcomm_.MakeBatchHandle(channel, hcommBatchWqeTensor_, LAYERED_HCOMM_BATCH_UB_BYTES,
                                                            GetRankWinAddrWithOffset(dstRank, 0));
        }

        ExpertLoopState combineState = waveBeginState;
        GMMAddrInfo combineAddrInfo{};
        for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
            if (!UpdateGroupParams<AddrUpdateMode::GMM2>(combineState, expertIdx)) {
                continue;
            }
            UpdateGlobalBuffer<AddrUpdateMode::GMM2>(combineAddrInfo, combineState);
            ProcessCombineRank(combineAddrInfo, combineState, expertIdx, batchState);
        }
        if (dstRank != rankId_ && batchState.pendingTokenCount != 0U) {
            hcomm_.BatchCommit(batchState.batchHandle);
            batchState.pendingTokenCount = 0U;
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitUnpermuteWeightChunk(
    uint32_t dataResBufAlign, uint32_t dataResFp32BufAlign)
{
    uint32_t fixedUbBeforeTopK = dataResBufAlign + dataResFp32BufAlign;
    uint32_t scaleUbCost = 0;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        // 预算公式必须与 UnpermuteBuffInit 的张量一致：每 32 个元素一个 scale，再扩展为两份。
        uint32_t scaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(k_), static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        scaleUbCost = Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DEQUANT_SCALE_EXPAND),
                                           static_cast<uint32_t>(ALIGN_32)) +
                      Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(float) * DEQUANT_SCALE_EXPAND),
                                           static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t fixedUbCost = fixedUbBeforeTopK + scaleUbCost;
    uint32_t availUb = fixedUbCost < SEND_MASK_UB_LIMIT ? SEND_MASK_UB_LIMIT - fixedUbCost : 0U;
    uint32_t perTokenBytes = topK_ * sizeof(float);
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        perTokenBytes += topK_ * sizeof(bfloat16_t);
    }
    constexpr uint32_t bufferCount = Std::IsSame<TopkWeightsType, bfloat16_t>::value ? 2U : 1U;
    constexpr uint32_t alignmentSlackBytes = bufferCount * (static_cast<uint32_t>(ALIGN_32) - 1U);
    uint32_t chunkUb = availUb > alignmentSlackBytes ? availUb - alignmentSlackBytes : 0U;
    topKWeightsChunkLen_ = (perTokenBytes > 0) ? chunkUb / perTokenBytes : m_;
    if (topKWeightsChunkLen_ == 0) {
        topKWeightsChunkLen_ = 1;
    }
    if (topKWeightsChunkLen_ > m_) {
        topKWeightsChunkLen_ = m_;
    }
}

// =============================================
// UnpermuteBuffInit：Unpermute中使用的buffer申请
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteBuffInit()
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)),
                                                    static_cast<uint32_t>(ALIGN_32));
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    InitUnpermuteWeightChunk(dataResBufAlign, dataResFp32BufAlign);
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(topKWeightsChunkLen_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    // Tensor用处：Unpermute 函数用于存储mte2搬入token；
    // Tensor大小：大小为3 *
    // 单个token长度，2块是用于mte2搬运的doubleBuffer，1块是用于存储累加计算Cast完的输出结果，用于搬出；
    uint32_t dataResAddr = 0;
    uint32_t dataResSize = dataResBufAlign / sizeof(bfloat16_t);
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResSize);
    // Tensor用处：Unpermute 函数用于存储token Cast 目的Tensor；
    // Tensor大小：dataResTensor_开设大小乘以BF16_TO_FP32；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    uint32_t dataResFp32Size = dataResFp32BufAlign / sizeof(float);
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32Size);
    // Tensor用处：用于存储topKWeight；
    // Tensor大小：m_ * topK_ * sizeof(float) align到32字节对齐；
    uint32_t topKWeightsAddr = dataResFp32Addr + dataResFp32BufAlign;
    uint32_t topKWeightsSize = topKWeightsBufAlign / sizeof(float);
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr, topKWeightsSize);
    uint32_t tempAddr = topKWeightsAddr + topKWeightsBufAlign;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(k_), static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）
        // Tensor大小：scaleNum * sizeof(bfloat16_t) * DEQUANT_SCALE_EXPAND，用于 scale 扩展。
        uint32_t bf16ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DEQUANT_SCALE_EXPAND),
                                 static_cast<uint32_t>(ALIGN_32));
        bf16ScaleTensor_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）
        // Tensor大小：scaleNum * sizeof(float) * DEQUANT_SCALE_EXPAND，用于 scale 扩展。
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(
            static_cast<uint32_t>(scaleNum * sizeof(float) * DEQUANT_SCALE_EXPAND), static_cast<uint32_t>(ALIGN_32));
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }
    topKWeightsTempAddr_ = tempAddr;
}

// ===============================================================
// UnpermuteSharedExpert：共享专家结果累加到当前 token 的 fp32 累加器
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteSharedExpert(int32_t tokenIdx)
{
    LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
    LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
    LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
    LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer((__gm__ bfloat16_t *)params_.workspaceInfo.sharedExpertResultPtr);
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        auto event = (sharedIdx % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (sharedIdx % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (sharedIdx % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
        WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        DataCopy(dataInBf16, sharedResult[(sharedIdx * m_ + tokenIdx) * k_], k_);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
        Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
        Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadUnpermuteWeights(int32_t chunkStart,
                                                                                            int32_t chunkTokenCnt)
{
    if constexpr (!TopkWeightsPrefetch) {
        if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
            GlobalTensor<float> topKWeightsGlobalTensor_;
            topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float *)params_.probsGmAddr);
            DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(chunkTokenCnt * topK_ * sizeof(float)), 0U, 0U,
                                            0U};
            DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
            DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_[chunkStart * topK_], copyParams, copyPadParams);
            SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
        }
        if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
            uint32_t tempBufAlign = Ops::Base::CeilAlign(
                static_cast<uint32_t>(topKWeightsChunkLen_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
            LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, topKWeightsTempAddr_,
                                              tempBufAlign / sizeof(bfloat16_t));
            GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
            topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t *)params_.probsGmAddr);
            DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(chunkTokenCnt * topK_ * sizeof(bfloat16_t)), 0U,
                                            0U, 0U};
            DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
            DataCopyPad(tempLocal, topkWeightsGlobalTensor[chunkStart * topK_], copyParams, copyPadParams);
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
            Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, chunkTokenCnt * topK_);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadUnpermuteExpertInput(
    const GlobalTensor<bfloat16_t> &expandedX, int32_t tokenIdx, int32_t expId, TEventID event,
    LocalTensor<bfloat16_t> &dataInBf16, LocalTensor<float> &dataInFp32)
{
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * k_], k_);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
        Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
    } else {
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantTokenSize = k_ + nScale;
        uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
        WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * quantEleNum], quantEleNum);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        using Fp8Type =
            typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
        Mxfp8::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_, fp32ScaleTensor_, nScale,
                                                 k_);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AccumulateUnpermuteExperts(
    const GlobalTensor<bfloat16_t> &expandedX, int32_t tokenIdx, int32_t localIdx)
{
    LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
    LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
    LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
    LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
    for (int32_t expId = 0; expId < topK_; ++expId) {
        auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
        LoadUnpermuteExpertInput(expandedX, tokenIdx, expId, event, dataInBf16, dataInFp32);
        if constexpr (TopkWeightsPrefetch) {
            if (expId == 0) {
                DataCopy(dataResFp32Tensor_, dataInFp32, k_);
            } else {
                Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
            }
        } else {
            float expScale = topKWeightsTensor_.GetValue(localIdx * topK_ + expId);
            if (expId == 0) {
                Muls(dataResFp32Tensor_, dataInFp32, expScale, k_);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, k_);
                Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
            }
        }
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

// ===============================================================
// Unpermute：对于各个专家还回来token的后处理，进行对应scale相乘与累加
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t *)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t *)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t chunkStart = coreOffset; chunkStart < coreLen + coreOffset;) {
        int32_t chunkEnd = chunkStart + static_cast<int32_t>(topKWeightsChunkLen_);
        if (chunkEnd > coreLen + coreOffset) {
            chunkEnd = coreLen + coreOffset;
        }
        int32_t chunkTokenCnt = chunkEnd - chunkStart;
        LoadUnpermuteWeights(chunkStart, chunkTokenCnt);
        for (int32_t tokenIdx = chunkStart; tokenIdx < chunkEnd; tokenIdx++) {
            int32_t localIdx = tokenIdx - chunkStart;
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
            AccumulateUnpermuteExperts(expandedX, tokenIdx, localIdx);
            // 共享专家结果累加（直接加，不乘 topk_weight）
            if (sharedExpertNum_ > 0) {
                UnpermuteSharedExpert(tokenIdx);
            }
            // fp32 -> bf16
            Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[tokenIdx * k_], dataResTensor_, k_);
        }
        chunkStart = chunkEnd;
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_LAYERED_COMBINE_H
