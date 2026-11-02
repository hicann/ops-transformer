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

struct LayeredDequantScaleBufferBytes {
    uint32_t bf16;
    uint32_t fp32;
};

/*
 * 按公共 DeQuantMxFp8 的实际存储轮次计算两块 scale scratch 的字节数，包含尾轮无效元素。
 * 设 VL 为向量寄存器字节数：DIST_INTLV_B16/B32 忽略 mask，每轮交织写出 2 * VL 字节。
 * BF16 阶段每轮转换 VL / sizeof(bfloat16_t) 个原始 scale，并交织复制为两份。
 * FP32 阶段的输入已有 scaleNum * 2 个 BF16 元素，每轮转换 VL / sizeof(float) 个，再复制为两份。
 * 因此先将每阶段转换后的单份数据字节数向上对齐到 VL（计算完整轮数），再乘 2（双寄存器存储）。
 * 不能先乘 2 再仅按 VL 对齐：例如 VL=256B、scaleNum=34，BF16 阶段一轮实际写 512B，
 * Align256(34 * 2B) * 2 = 512B，而 Align256(34 * 2B * 2) 只有 256B。
 */
__aicore__ inline LayeredDequantScaleBufferBytes CalcLayeredDequantScaleBufferBytes(uint32_t scaleNum)
{
    constexpr uint32_t vectorBytes = AscendC::VECTOR_REG_WIDTH;
    uint32_t bf16Bytes = Ops::Base::CeilAlign(scaleNum * static_cast<uint32_t>(sizeof(bfloat16_t)), vectorBytes) * 2U;
    uint32_t fp32Bytes = Ops::Base::CeilAlign(scaleNum * 2U * static_cast<uint32_t>(sizeof(float)), vectorBytes) * 2U;
    return {bf16Bytes, fp32Bytes};
}

/**
 * 绑定 Combine 消费的 GMM2 输出与组同步计数地址。
 */
__aicore__ inline void UpdateCombineGlobalBuffer(GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                 const WorkspaceLayout& workspace, GM_ADDR workspaceBase,
                                                 uint32_t combineSyncSlotCountPerExpert)
{
    UpdateMoeExpertCombineGlobalBuffer(workspace, workspaceBase, gmmAddrInfo, state);
    gmmAddrInfo.gmm2CombineSyncCounter =
        (__gm__ int32_t*)workspace.urma.sync.combineCounters.get_ptr(workspaceBase) +
        static_cast<uint64_t>(state.expertIdx) * combineSyncSlotCountPerExpert * static_cast<uint64_t>(INT_CACHELINE);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitCombineBuffers()
{
    if constexpr (g_coreType == AIV) {
        // Combine 仅在当前 AIV1 完成 Dispatch 张量使用后启动，并从 UB 首地址重建视图。
        // Hcomm 使用前 512 字节；Combine metadata 和计算临时区从其后开始复用 UB。
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t quantTokenSizeBytes = static_cast<uint32_t>(CalcCombineTokenBytes(k_, sizeof(bfloat16_t), true));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GetCombineRankRange(uint32_t expertIdx,
                                                                                           uint32_t originRank,
                                                                                           uint32_t& rowStart,
                                                                                           uint32_t& tokenCount)
{
    // originRank 指 token 原始输入卡，用于查询结果行范围；实际发送目标由 batchState 保存。
    uint64_t expertBegin = static_cast<uint64_t>(expertIdx) * worldSize_;
    uint64_t rankEndIndex = expertBegin + originRank;
    CleanInvalidateGmCache();
    int32_t rankEnd = cumsumGm_[rankEndIndex];

    int32_t expertStart = 0;
    if (expertBegin != 0U) {
        CleanInvalidateGmCache();
        expertStart = *(cumsumGm_ + (expertBegin - 1U));
    }
    int32_t rankStart = expertStart;
    if (originRank != 0U) {
        CleanInvalidateGmCache();
        rankStart = cumsumGm_[rankEndIndex - 1U];
    }
    rowStart = static_cast<uint32_t>(rankStart - expertStart);
    tokenCount = static_cast<uint32_t>(rankEnd - rankStart);
    return tokenCount != 0U;
}

/* 处理当前专家的一段结果行，等待对应 GMM2 分组完成，再构造普通写或去重贡献请求。 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CombineTokensInRange(
    const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& gmm2State, uint32_t expertIdx, const WorkRange& rowRange,
    CombineBatchState& batchState)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    constexpr bool IsQuantized = CombineQuantMode != COMBINE_NO_QUANT;

    uint32_t mExpert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
    GroupSyncSlotLayout slotLayout = CalcGroupSyncSlotLayout(mExpert, blockNum_);
    __gm__ int32_t* expertCounterBase = reinterpret_cast<__gm__ int32_t*>(gmmAddrInfo.gmm2CombineSyncCounter);

    uint32_t processedCount = 0U;
    while (processedCount < rowRange.count) {
        uint32_t currentRow = rowRange.start + processedCount;
        uint32_t targetGroup = currentRow / L1_TILE_M_256;
        uint32_t firstSyncSlot = 0;
        uint32_t syncSlotCount = 0;
        GetGroupSyncSlotRange(targetGroup, slotLayout, firstSyncSlot, syncSlotCount);
        __gm__ int32_t* counterAddr = GetCombineSyncCounterAddress(expertCounterBase, firstSyncSlot);
        if (GmRead(counterAddr) < static_cast<int32_t>(nTilesPerGroup)) {
            continue;
        }

        uint32_t groupEndRow = (targetGroup + 1U) * L1_TILE_M_256;
        groupEndRow = groupEndRow < mExpert ? groupEndRow : mExpert;
        uint32_t batchCount = groupEndRow - currentRow;
        uint32_t remainingTokens = rowRange.count - processedCount;
        batchCount = batchCount < remainingTokens ? batchCount : remainingTokens;

        // 两种模式共用同一 metadata 基址；普通 Group 与去重 Group 都自行搬入 metadata 并绑定 UB。
        GM_ADDR metaInfoAddr =
            workspace.routing.receivedTokenMetadata.get_ptr(buffer) +
            static_cast<uint64_t>(gmm2State.globalTokenStartIndex + currentRow) * META_INFO_SIZE * sizeof(int32_t);
        if constexpr (CombineServerDedup) {
            CombineImpl::CombineServerDedupTokenGroup(currentRow, batchCount, k_, expertIdx, rankId_,
                                                      gmmAddrInfo.gmm2OutGlobal, metaInfoAddr, params_, comm_,
                                                      batchState);
        } else {
            CombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t, IsQuantized>(
                currentRow, batchCount, k_, expertIdx, rankId_, gmmAddrInfo.gmm2OutGlobal, metaInfoAddr,
                combineUbTensorSize_, params_, comm_, batchState);
        }
        processedCount += batchCount;
    }
}

/*
 * 处理当前专家发往实际 targetRank 的结果，expertRowRange 限定当前通信 Wave 的专家行范围。
 * 普通模式处理该输入卡；Server 去重模式处理各 Server 中与中继同编号的原始输入卡。
 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CombineTokensOfExpert(
    const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& gmm2State, uint32_t expertIdx,
    const WorkRange& expertRowRange, CombineBatchState& batchState)
{
    const uint32_t targetRank = batchState.dstRankId;
    // 处理当前专家中属于一张原始输入卡的结果，共用当前实际目标通道的批次。
    auto combineTokensFromRank = [&](uint32_t originRank) __attribute__((cce_aicore))
    {
        WorkRange rowRange;
        if (!GetCombineRankRange(expertIdx, originRank, rowRange.start, rowRange.count)) {
            return;
        }
        // 一个专家可能跨多个通信 Wave；仅发送来源卡范围与本 Wave 的交集。
        const uint32_t rowEnd = rowRange.start + rowRange.count;
        const uint32_t waveEnd = expertRowRange.start + expertRowRange.count;
        const uint32_t begin = rowRange.start > expertRowRange.start ? rowRange.start : expertRowRange.start;
        const uint32_t end = rowEnd < waveEnd ? rowEnd : waveEnd;
        if (begin >= end) {
            return;
        }
        CombineTokensInRange<CombineServerDedup>(gmmAddrInfo, gmm2State, expertIdx, WorkRange{begin, end - begin},
                                                 batchState);
    };
    if constexpr (!CombineServerDedup) {
        combineTokensFromRank(targetRank);
        return;
    }

    // 本 Server 中继负责各 Server 内与其同编号的原始输入卡。
    const uint32_t rankInServer = targetRank % rankNumPerServer_;
    for (uint32_t originServer = 0; originServer < serverNum_; ++originServer) {
        const uint32_t originRank = originServer * rankNumPerServer_ + rankInServer;
        if (originRank < worldSize_) {
            combineTokensFromRank(originRank);
        }
    }
}

/*
 * 向实际 targetRank 处理一个通信 Wave 的 Combine；保留完整专家状态，并逐专家裁剪行范围。
 * 跨计算 Wave、专家和输入卡积累同一通道的请求，满批提交，通信 Wave 结束时提交尾批。
 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineToTargetRank(
    ExpertLoopState waveBeginState, const ExpertTokenRange& range, uint32_t targetRank)
{
    CombineBatchState batchState{};
    batchState.dstRankId = targetRank;
    if (targetRank != rankId_) {
        Channel channel = comm_.GetChannel(targetRank);
        batchState.batchHandle = comm_.MakeBatchHandle(channel, GetRankWinAddrWithOffset(targetRank, 0));
    }
    ExpertLoopState combineState = waveBeginState;
    GMMAddrInfo combineAddrInfo{};
    const uint32_t expertBegin = range.begin.expertIdx;
    const uint32_t expertEnd = range.end.expertIdx + (range.end.tokenIndexInExpert != 0U ? 1U : 0U);
    for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
        const uint32_t rowBegin = expertIdx == expertBegin ? range.begin.tokenIndexInExpert : 0U;
        // 首专家从中间行继续时，waveBeginState 已保留该专家的完整形状及基址。
        if (rowBegin == 0U) {
            const BlockWorkspaceContext countWorkspace{blockIdx_, aicNum_};
            const uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
                params_.workspace.layout.routing.recvCounts.get_ptr(params_.workspace.base), countWorkspace,
                moeExpertPerRank_, expertIdx);
            UpdateExpertLoopState(combineState, expertIdx, expertTokenCount);
            if (expertTokenCount == 0U) {
                continue;
            }
        }
        const uint32_t expertRows = Get<M_VALUE>(combineState.problemShape);
        const uint32_t rowEnd = expertIdx == range.end.expertIdx ? range.end.tokenIndexInExpert : expertRows;
        if (rowBegin >= rowEnd) {
            continue;
        }
        // Combine 按消费范围绑定 GMM2 输出与组同步计数，地址由累计行数和专家序号直接计算。
        UpdateCombineGlobalBuffer(combineAddrInfo, combineState, params_.workspace.layout, params_.workspace.base,
                                  static_cast<uint32_t>(params_.tilingData->compute.combineSyncSlotCountPerExpert));
        CombineTokensOfExpert<CombineServerDedup>(combineAddrInfo, combineState, expertIdx,
                                                  WorkRange{rowBegin, rowEnd - rowBegin}, batchState);
    }
    if (targetRank != rankId_ && batchState.pendingTokenCount != 0U) {
        comm_.BatchCommit(batchState.batchHandle);
        batchState.pendingTokenCount = 0U;
    }
}

/* 按实际目标卡分配通信 Wave 的 Combine；去重模式的目标为本 Server 内中继，普通模式为源卡。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineTokenRange(
    const ExpertLoopState& waveBeginState, const ExpertTokenRange& range)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    const uint32_t expertBegin = range.begin.expertIdx;
    const uint32_t expertEnd = range.end.expertIdx + (range.end.tokenIndexInExpert != 0U ? 1U : 0U);
    if (subBlockIdx_ != 1U || expertBegin >= expertEnd) {
        return;
    }

    // 每个实际目标卡在一个通信 Wave 中共用批次；普通模式回源卡，去重模式发往本 Server 中继。
    const bool combineServerDedup = IsCombineServerDedupEnabled();
    const WorkRange targetRange =
        combineServerDedup ? WorkRange{serverId_ * rankNumPerServer_, rankNumPerServer_} : WorkRange{0U, worldSize_};
    for (uint32_t targetRank = targetRange.start;
         targetRank < targetRange.start + targetRange.count && targetRank < worldSize_; ++targetRank) {
        if (!IsChannelOwner(targetRank)) {
            continue;
        }
        if (combineServerDedup) {
            ProcessCombineToTargetRank<true>(waveBeginState, range, targetRank);
        } else {
            ProcessCombineToTargetRank<false>(waveBeginState, range, targetRank);
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
        uint32_t scaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(k_), static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        auto scaleBytes = CalcLayeredDequantScaleBufferBytes(scaleNum);
        scaleUbCost = scaleBytes.bf16 + scaleBytes.fp32;
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
    dataResTensor_ = reinterpret_cast<__ubuf__ bfloat16_t*>(dataResAddr);
    // Tensor用处：Unpermute 函数用于存储token Cast 目的Tensor；
    // Tensor大小：dataResTensor_开设大小乘以BF16_TO_FP32；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    uint32_t dataResFp32Size = dataResFp32BufAlign / sizeof(float);
    dataResFp32Tensor_ = reinterpret_cast<__ubuf__ float*>(dataResFp32Addr);
    // Tensor用处：用于存储topKWeight；
    // Tensor大小：m_ * topK_ * sizeof(float) align到32字节对齐；
    uint32_t topKWeightsAddr = dataResFp32Addr + dataResFp32BufAlign;
    uint32_t topKWeightsSize = topKWeightsBufAlign / sizeof(float);
    topKWeightsTensor_ = reinterpret_cast<__ubuf__ float*>(topKWeightsAddr);
    uint32_t tempAddr = topKWeightsAddr + topKWeightsBufAlign;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(k_), static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        auto scaleBytes = CalcLayeredDequantScaleBufferBytes(scaleNum);
        // 按实际双寄存器存储范围预留，尾轮不能只按有效 scale 长度分配。
        uint32_t bf16ScaleBufAlign = scaleBytes.bf16;
        bf16ScaleTensor_ = reinterpret_cast<__ubuf__ bfloat16_t*>(tempAddr);
        tempAddr += bf16ScaleBufAlign;
        uint32_t fp32ScaleBufAlign = scaleBytes.fp32;
        fp32ScaleTensor_ = reinterpret_cast<__ubuf__ float*>(tempAddr);
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
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    __ubuf__ bfloat16_t* dataIn0Bf16 = (dataResTensor_ + (k_));
    __ubuf__ bfloat16_t* dataIn1Bf16 = (dataResTensor_ + (k_ * 2));
    __ubuf__ float* dataIn0Fp32 = (dataResFp32Tensor_ + (k_));
    __ubuf__ float* dataIn1Fp32 = (dataResFp32Tensor_ + (k_ * 2));
    __gm__ bfloat16_t* sharedResult = workspace.shared.gmm2Output.template get_ptr<bfloat16_t>(buffer);
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        auto event = (sharedIdx % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (sharedIdx % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (sharedIdx % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
        PipeWait<PIPE_V, PIPE_MTE2>(event);
        CopyGmToUb(dataInBf16, (sharedResult + ((sharedIdx * m_ + tokenIdx) * k_)), (k_) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
        PipeNotify<PIPE_S, PIPE_V>(event);
        PipeWait<PIPE_S, PIPE_V>(event);
        {
            LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), k_);
            LocalTensor<bfloat16_t> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInBf16), k_);
            Cast(dst, src, RoundMode::CAST_NONE, k_);
        }

        {
            LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResFp32Tensor_), k_);
            LocalTensor<float> rhs(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), k_);
            Add(dst, dst, rhs, k_);
        }
        PipeNotify<PIPE_V, PIPE_MTE2>(event);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadUnpermuteWeights(int32_t chunkStart,
                                                                                            int32_t chunkTokenCnt)
{
    if constexpr (!TopkWeightsPrefetch) {
        if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
            __gm__ float* topkWeightsGm_ = (__gm__ float*)params_.probsGmAddr;

            CopyGmToUb(topKWeightsTensor_, (topkWeightsGm_ + (chunkStart * topK_)),
                       static_cast<uint32_t>(chunkTokenCnt * topK_ * sizeof(float)));
            PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID2>();
        }
        if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
            uint32_t tempBufAlign = Ops::Base::CeilAlign(
                static_cast<uint32_t>(topKWeightsChunkLen_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
            __ubuf__ bfloat16_t* tempLocal = reinterpret_cast<__ubuf__ bfloat16_t*>(topKWeightsTempAddr_);
            __gm__ bfloat16_t* topkWeightsGm = (__gm__ bfloat16_t*)params_.probsGmAddr;

            CopyGmToUb(tempLocal, (topkWeightsGm + (chunkStart * topK_)),
                       static_cast<uint32_t>(chunkTokenCnt * topK_ * sizeof(bfloat16_t)));
            PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
            {
                LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(topKWeightsTensor_),
                                       chunkTokenCnt * topK_);
                LocalTensor<bfloat16_t> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(tempLocal),
                                            chunkTokenCnt * topK_);
                Cast(dst, src, RoundMode::CAST_NONE, chunkTokenCnt * topK_);
            }
            PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID2>();
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadUnpermuteExpertInput(
    __gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t expId, TEventID event, __ubuf__ bfloat16_t* dataInBf16,
    __ubuf__ float* dataInFp32)
{
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        PipeWait<PIPE_V, PIPE_MTE2>(event);
        CopyGmToUb(dataInBf16, (expandedX + ((tokenIdx * topK_ + expId) * k_)), (k_) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
        PipeNotify<PIPE_S, PIPE_V>(event);
        PipeWait<PIPE_S, PIPE_V>(event);
        {
            LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), k_);
            LocalTensor<bfloat16_t> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInBf16), k_);
            Cast(dst, src, RoundMode::CAST_NONE, k_);
        }
    } else {
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        // 与发送端及 Peermem 容量使用相同的 padded record stride，包含真正的 scale 区。
        uint32_t quantTokenSize = static_cast<uint32_t>(CalcCombineTokenBytes(k_, sizeof(bfloat16_t), true));
        uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
        PipeWait<PIPE_V, PIPE_MTE2>(event);
        CopyGmToUb(dataInBf16, (expandedX + ((tokenIdx * topK_ + expId) * quantEleNum)),
                   (quantEleNum) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
        using Fp8Type =
            typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
        PipeNotify<PIPE_S, PIPE_V>(event);
        PipeWait<PIPE_S, PIPE_V>(event);
        Mxfp8::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_, fp32ScaleTensor_, nScale,
                                                 k_);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AccumulateUnpermuteExperts(
    __gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t localIdx)
{
    __ubuf__ bfloat16_t* dataIn0Bf16 = (dataResTensor_ + (k_));
    __ubuf__ bfloat16_t* dataIn1Bf16 = (dataResTensor_ + (k_ * 2));
    __ubuf__ float* dataIn0Fp32 = (dataResFp32Tensor_ + (k_));
    __ubuf__ float* dataIn1Fp32 = (dataResFp32Tensor_ + (k_ * 2));
    for (int32_t expId = 0; expId < topK_; ++expId) {
        auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
        LoadUnpermuteExpertInput(expandedX, tokenIdx, expId, event, dataInBf16, dataInFp32);

        if constexpr (TopkWeightsPrefetch) {
            if (expId == 0) {
                CopyUbToUb(dataResFp32Tensor_, dataInFp32, (k_) * sizeof(float));
            } else {
                {
                    LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResFp32Tensor_), k_);
                    LocalTensor<float> rhs(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), k_);
                    Add(dst, dst, rhs, k_);
                }
            }
        } else {
            float expScale = topKWeightsTensor_[localIdx * topK_ + expId];
            if (expId == 0) {
                {
                    LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResFp32Tensor_), k_);
                    LocalTensor<float> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), k_);
                    Muls(dst, src, expScale, k_);
                }
            } else {
                {
                    LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), k_);
                    Muls(dst, dst, expScale, k_);
                }

                {
                    LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResFp32Tensor_), k_);
                    LocalTensor<float> rhs(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), k_);
                    Add(dst, dst, rhs, k_);
                }
            }
        }
        PipeNotify<PIPE_V, PIPE_MTE2>(event);
    }
}

// ===============================================================
// Unpermute：对于各个专家还回来token的后处理，进行对应scale相乘与累加
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Unpermute()
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    __gm__ bfloat16_t* expandedX = (__gm__ bfloat16_t*)peermem.combineSend.get_ptr(commBuffer);
    __gm__ bfloat16_t* output = (__gm__ bfloat16_t*)params_.y2GmAddr;
    PipeNotify<PIPE_V, PIPE_MTE2>(EVENT_ID0);
    PipeNotify<PIPE_V, PIPE_MTE2>(EVENT_ID1);
    for (int32_t chunkStart = coreOffset; chunkStart < coreLen + coreOffset;) {
        int32_t chunkEnd = chunkStart + static_cast<int32_t>(topKWeightsChunkLen_);
        if (chunkEnd > coreLen + coreOffset) {
            chunkEnd = coreLen + coreOffset;
        }
        int32_t chunkTokenCnt = chunkEnd - chunkStart;
        LoadUnpermuteWeights(chunkStart, chunkTokenCnt);
        for (int32_t tokenIdx = chunkStart; tokenIdx < chunkEnd; tokenIdx++) {
            int32_t localIdx = tokenIdx - chunkStart;
            AccumulateUnpermuteExperts(expandedX, tokenIdx, localIdx);
            // 共享专家结果累加（直接加，不乘 topk_weight）
            if (sharedExpertNum_ > 0) {
                UnpermuteSharedExpert(tokenIdx);
            }
            // 输出区 [0,k) 在最终 Cast 处复用；复用等待放在 Cast 前，让输入搬入和 FP32 累加
            // 与上一 token 的搬出重叠。
            PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID2>();
            // fp32 -> bf16

            {
                LocalTensor<bfloat16_t> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResTensor_), k_);
                LocalTensor<float> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResFp32Tensor_), k_);
                Cast(dst, src, RoundMode::CAST_RINT, k_);
            }
            PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID3>();
            CopyUbToGm((output + (tokenIdx * k_)), dataResTensor_, (k_) * sizeof(bfloat16_t));
        }
        chunkStart = chunkEnd;
    }
    // 出口显式等待最后一次输出搬出完成。
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID3>();
    PipeWait<PIPE_V, PIPE_MTE2>(EVENT_ID0);
    PipeWait<PIPE_V, PIPE_MTE2>(EVENT_ID1);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_LAYERED_COMBINE_H
