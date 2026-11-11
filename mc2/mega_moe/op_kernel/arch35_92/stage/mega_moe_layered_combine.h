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
    const uint8_t CombineQuantMode = params_.tilingData->combineQuantMode;
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;

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
            CombineImpl::CombineTokenGroup<bfloat16_t>(CombineQuantMode, currentRow, batchCount, k_, expertIdx, rankId_,
                                                       gmmAddrInfo.gmm2OutGlobal, metaInfoAddr, combineUbTensorSize_,
                                                       params_, comm_, batchState);
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

/* 按 Host 配置绑定普通 Unpermute 的输入 ring、累加区和权重区。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline MegaMoeUnpermuteBufferConfig MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteBuffInit()
{
    const uint8_t CombineQuantMode = params_.tilingData->combineQuantMode;
    const auto* tilingData = params_.tilingData;
    const auto bufferConfig = aivCoreIdx_ < tilingData->unpermuteFullTokenChunkCoreCount ?
                                  tilingData->unpermuteConfigForFullTokenChunk :
                                  tilingData->unpermuteConfigForTailTokenChunk;
    if (bufferConfig.tokensPerBatch == 0) {
        return bufferConfig;
    }
    const uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    const uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    const uint32_t dataResBytes = (bufferConfig.inputBufferCount + 1U) * bf16SlotBytes;
    uint32_t dataResFp32Bytes = fp32SlotBytes;
    if (CombineQuantMode != COMBINE_NO_QUANT) {
        dataResFp32Bytes += bufferConfig.inputBufferCount * fp32SlotBytes;
    }
    // BF16 槽 0 用于输出，其余槽用于输入；FP32 槽 0 累加，通信量化另有输入槽。
    dataResTensor_ = reinterpret_cast<__ubuf__ bfloat16_t*>(0U);
    dataResFp32Tensor_ = reinterpret_cast<__ubuf__ float*>(dataResBytes);
    uint32_t tempAddr = dataResBytes + dataResFp32Bytes;
    topKWeightsTensor_ = reinterpret_cast<__ubuf__ float*>(tempAddr);
    tempAddr += bufferConfig.topKWeightsBufferBytes;
    topKWeightsTempAddr_ = tempAddr;
    tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    if (CombineQuantMode != COMBINE_NO_QUANT) {
        const uint32_t scaleNum = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        const auto scaleBytes = CalcLayeredDequantScaleBufferBytes(scaleNum);
        bf16ScaleTensor_ = reinterpret_cast<__ubuf__ bfloat16_t*>(tempAddr);
        tempAddr += scaleBytes.bf16;
        fp32ScaleTensor_ = reinterpret_cast<__ubuf__ float*>(tempAddr);
    }
    return bufferConfig;
}

// ===============================================================
// UnpermuteSharedExpert：共享专家结果累加到当前 token 的 fp32 累加器
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::WaitForSharedExpertResult(
    uint32_t sharedExpertIdx, uint32_t tokenIdx)
{
    const uint32_t tokenGroupCount = Ops::Base::CeilDiv(m_, COMBINE_TOKEN_GROUP_SIZE);
    auto* expertCounters =
        params_.workspace.layout.urma.sync.sharedGmm2Tiles.template get_ptr<int32_t>(params_.workspace.base) +
        static_cast<uint64_t>(sharedExpertIdx) * tokenGroupCount * INT_CACHELINE;
    auto* counter = GetCombineSyncCounterAddress(expertCounters, tokenIdx / COMBINE_TOKEN_GROUP_SIZE);
    const uint32_t outputTileCount = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(L1_TILE_N));
    // FIX 写完该组的全部输出列块后，才允许 Unpermute 读取共享结果。
    WaitUntilGmFlagEquals(counter, static_cast<int32_t>(outputTileCount));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteSharedExpert(
    int32_t tokenIdx, uint32_t& readySharedGroup, const MegaMoeUnpermuteBufferConfig& bufferConfig,
    uint32_t& inputBufferIdx)
{
    const uint32_t tokenGroup = static_cast<uint32_t>(tokenIdx) / COMBINE_TOKEN_GROUP_SIZE;
    const bool waitForGroup = tokenGroup != readySharedGroup;
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    __gm__ bfloat16_t* sharedResult = workspace.shared.gmm2Output.template get_ptr<bfloat16_t>(buffer);
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (waitForGroup) {
            WaitForSharedExpertResult(sharedIdx, static_cast<uint32_t>(tokenIdx));
        }
        auto event = static_cast<TEventID>(inputBufferIdx);
        auto dataInBf16 = dataResTensor_ + (inputBufferIdx + 1U) * bufferConfig.bf16SlotElementCount;
        PipeWait<PIPE_V, PIPE_MTE2>(event);
        CopyGmToUb(dataInBf16, (sharedResult + ((sharedIdx * m_ + tokenIdx) * k_)), (k_) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
        AccumulateUnpermuteExpertVF<bfloat16_t, false, false>(dataResFp32Tensor_, dataInBf16,
                                                              static_cast<__ubuf__ float*>(nullptr), k_);
        PipeNotify<PIPE_V, PIPE_MTE2>(event);
        inputBufferIdx = (inputBufferIdx + 1U) % static_cast<uint32_t>(bufferConfig.inputBufferCount);
    }
    readySharedGroup = tokenGroup; // 全部共享专家已就绪，后续 token 复用本组结果。
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadUnpermuteExpertInput(
    __gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t expId, TEventID event, __ubuf__ bfloat16_t* dataInBf16,
    __ubuf__ float* dataInFp32)
{
    const uint8_t CombineQuantMode = params_.tilingData->combineQuantMode;
    if (CombineQuantMode == COMBINE_NO_QUANT) {
        PipeWait<PIPE_V, PIPE_MTE2>(event);
        CopyGmToUb(dataInBf16, (expandedX + ((tokenIdx * topK_ + expId) * k_)), (k_) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
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
        PipeNotify<PIPE_S, PIPE_V>(event);
        PipeWait<PIPE_S, PIPE_V>(event);
        if (CombineQuantMode == MXFP8_E4M3_COMM_QUANT) {
            Mxfp8::DeQuantMxFp8<fp8_e4m3fn_t, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_, fp32ScaleTensor_,
                                                          nScale, k_);
        } else {
            Mxfp8::DeQuantMxFp8<fp8_e5m2_t, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_, fp32ScaleTensor_,
                                                        nScale, k_);
        }
    }
}

// 仅为 VF 输入类型及原有标量开关保留独立特化。
template <typename InputType>
__aicore__ inline void AccumulateLayeredUnpermuteInput(__ubuf__ float* accum, __ubuf__ InputType* input,
                                                       __ubuf__ float* weight, uint32_t elementCount, bool applyWeight,
                                                       bool initAccum)
{
    SelectRuntimeBool(
        applyWeight, [&](auto weightTag) __attribute__((cce_aicore)) {
            if (initAccum) {
                AccumulateUnpermuteExpertVF<InputType, decltype(weightTag)::value, true>(accum, input, weight,
                                                                                         elementCount);
            } else {
                AccumulateUnpermuteExpertVF<InputType, decltype(weightTag)::value, false>(accum, input, weight,
                                                                                          elementCount);
            }
        });
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AccumulateUnpermuteExperts(
    __gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t localIdx, const MegaMoeUnpermuteBufferConfig& bufferConfig,
    uint32_t& inputBufferIdx)
{
    const bool TopkWeightsPrefetch = params_.tilingData->isTopkWeightsPrefetch;
    const uint8_t CombineQuantMode = params_.tilingData->combineQuantMode;
    for (int32_t expId = 0; expId < topK_; ++expId) {
        auto event = static_cast<TEventID>(inputBufferIdx);
        auto dataInBf16 = dataResTensor_ + (inputBufferIdx + 1U) * bufferConfig.bf16SlotElementCount;
        __ubuf__ float* dataInFp32 = nullptr;
        if (CombineQuantMode != COMBINE_NO_QUANT) {
            dataInFp32 = dataResFp32Tensor_ + (inputBufferIdx + 1U) * bufferConfig.fp32SlotElementCount;
        }
        LoadUnpermuteExpertInput(expandedX, tokenIdx, expId, event, dataInBf16, dataInFp32);

        __ubuf__ float* weightUb = nullptr;
        if (!TopkWeightsPrefetch) {
            weightUb = topKWeightsTensor_ + localIdx * topK_ + expId;
        }
        // 首槽与后续槽分别实例化，避免 VF 的每次向量迭代都判断初始化状态。
        if (CombineQuantMode == COMBINE_NO_QUANT) {
            AccumulateLayeredUnpermuteInput(dataResFp32Tensor_, dataInBf16, weightUb, k_, !TopkWeightsPrefetch,
                                            expId == 0);
        } else {
            AccumulateLayeredUnpermuteInput(dataResFp32Tensor_, dataInFp32, weightUb, k_, !TopkWeightsPrefetch,
                                            expId == 0);
        }
        PipeNotify<PIPE_V, PIPE_MTE2>(event);
        inputBufferIdx = (inputBufferIdx + 1U) % static_cast<uint32_t>(bufferConfig.inputBufferCount);
    }
}

/* 单个权重 chunk：搬入并转换权重、逐 token 累加、转换输出并搬出。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteTokenChunk(
    const WorkRange& tokenRange, const MegaMoeUnpermuteBufferConfig& bufferConfig, uint32_t& inputBufferIdx,
    uint32_t& readySharedGroup)
{
    const bool TopkWeightsPrefetch = params_.tilingData->isTopkWeightsPrefetch;
    // 权重区和输出区各使用一个核内 mutex，与输入 ring 的 Set/Wait 事件编号独立。
    constexpr uint8_t UNPERMUTE_WEIGHT_MUTEX_ID = 0U;
    constexpr uint8_t UNPERMUTE_OUTPUT_MUTEX_ID = 1U;
    const int32_t chunkStart = static_cast<int32_t>(tokenRange.start);
    const int32_t chunkTokenCnt = static_cast<int32_t>(tokenRange.count);
    const int32_t chunkEnd = chunkStart + chunkTokenCnt;
    __gm__ bfloat16_t* expandedX =
        params_.communication.layout.combineSend.template get_ptr<bfloat16_t>(params_.communication.base);
    __gm__ bfloat16_t* output = reinterpret_cast<__gm__ bfloat16_t*>(params_.y2GmAddr);
    if (!TopkWeightsPrefetch) {
        constexpr bool isBf16Weights = Std::IsSame<TopkWeightsType, bfloat16_t>::value;
        const uint32_t weightCount = chunkTokenCnt * topK_;
        __gm__ TopkWeightsType* weightsGm = reinterpret_cast<__gm__ TopkWeightsType*>(params_.probsGmAddr);
        __ubuf__ TopkWeightsType* weightsUb = isBf16Weights ?
                                                  reinterpret_cast<__ubuf__ TopkWeightsType*>(topKWeightsTempAddr_) :
                                                  reinterpret_cast<__ubuf__ TopkWeightsType*>(topKWeightsTensor_);
        // 等上一批最后一次权重读取完成，再覆盖权重区及 BF16 转换暂存区。
        asc_lock(PIPE_MTE2, UNPERMUTE_WEIGHT_MUTEX_ID);
        CopyGmToUb(weightsUb, weightsGm + chunkStart * topK_,
                   weightCount * static_cast<uint32_t>(sizeof(TopkWeightsType)));
        asc_unlock(PIPE_MTE2, UNPERMUTE_WEIGHT_MUTEX_ID);
        // 等权重搬入，保护转换及本批乘权；最后一个 token 乘权后释放。
        asc_lock(PIPE_V, UNPERMUTE_WEIGHT_MUTEX_ID);
        if constexpr (isBf16Weights) {
            LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(topKWeightsTensor_),
                                   chunkTokenCnt * topK_);
            LocalTensor<bfloat16_t> src(TPosition::VECCALC, topKWeightsTempAddr_, chunkTokenCnt * topK_);
            Cast(dst, src, RoundMode::CAST_NONE, chunkTokenCnt * topK_);
        }
    }
    for (int32_t tokenIdx = chunkStart; tokenIdx < chunkEnd; tokenIdx++) {
        int32_t localIdx = tokenIdx - chunkStart;
        AccumulateUnpermuteExperts(expandedX, tokenIdx, localIdx, bufferConfig, inputBufferIdx);
        if (!TopkWeightsPrefetch && tokenIdx + 1 == chunkEnd) {
            // 最后一次乘权后释放，下一批权重搬入可与共享累加和输出 Cast 重叠。
            asc_unlock(PIPE_V, UNPERMUTE_WEIGHT_MUTEX_ID);
        }
        // 共享专家结果累加（直接加，不乘 topk_weight）
        if (sharedExpertNum_ > 0) {
            UnpermuteSharedExpert(tokenIdx, readySharedGroup, bufferConfig, inputBufferIdx);
        }
        // 输出区 [0,k) 在最终 Cast 处复用；复用等待放在 Cast 前，让输入搬入和 FP32 累加
        // 与上一 token 的搬出重叠。
        asc_lock(PIPE_V, UNPERMUTE_OUTPUT_MUTEX_ID);
        // fp32 -> bf16
        {
            LocalTensor<bfloat16_t> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResTensor_), k_);
            LocalTensor<float> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResFp32Tensor_), k_);
            Cast(dst, src, RoundMode::CAST_RINT, k_);
        }
        asc_unlock(PIPE_V, UNPERMUTE_OUTPUT_MUTEX_ID);
        asc_lock(PIPE_MTE3, UNPERMUTE_OUTPUT_MUTEX_ID);
        CopyUbToGm((output + (tokenIdx * k_)), dataResTensor_, (k_) * sizeof(bfloat16_t));
        asc_unlock(PIPE_MTE3, UNPERMUTE_OUTPUT_MUTEX_ID);
    }
}

// ===============================================================
// Unpermute：对于各个专家还回来token的后处理，进行对应scale相乘与累加
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Unpermute(
    const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    const WorkRange tokenRange = GetBalancedWorkRange(m_, {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_});
    if (tokenRange.count == 0U) {
        return;
    }
    const uint32_t tokenEnd = tokenRange.start + tokenRange.count;
    const uint32_t batchTokens = bufferConfig.tokensPerBatch;
    // 普通专家、共享专家和后续 token 共用游标，跨权重 chunk 也不重置。
    uint32_t readySharedGroup = UINT32_MAX; // 本次调用内跨权重 chunk 缓存；首 token 必须等待。
    uint32_t inputBufferIdx = 0U;
    // 输入 ring 使用事件 0..5；权重区由 mutex 保护，不占用预留事件 6/7。
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(bufferIdx));
    }
    for (uint32_t chunkStart = tokenRange.start; chunkStart < tokenEnd; chunkStart += batchTokens) {
        const uint32_t remainingTokens = tokenEnd - chunkStart;
        const WorkRange chunkRange{chunkStart, batchTokens < remainingTokens ? batchTokens : remainingTokens};
        UnpermuteTokenChunk(chunkRange, bufferConfig, inputBufferIdx, readySharedGroup);
    }
    // 最后一次输出搬出完成，也保证其前序权重 V mutex 已释放。
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID3>();
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        PipeWait<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(bufferIdx));
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_LAYERED_COMBINE_H
