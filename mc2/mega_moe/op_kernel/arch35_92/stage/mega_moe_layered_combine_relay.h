/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_COMBINE_RELAY_H
#define MEGA_MOE_LAYERED_COMBINE_RELAY_H

#include "../mega_moe_vf/mega_moe_layered_vf.h"

namespace MegaMoeImpl {

/*
 * 筛选指定来源 Server 中一批有效 token ID，ID 和位图分别压紧写入输出缓冲，并返回有效数量。
 * 输入和输出缓冲由调用方复用，内部等待前批读取结束，返回前确认筛选完成。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GatherCombineTokenIds(
    uint32_t originServer, const WorkRange& tokenRange, const LocalTensor<int32_t>& tokenBitmaps,
    const LocalTensor<int32_t>& tokenIds, const LocalTensor<int32_t>& selectedTokenIds,
    const LocalTensor<int32_t>& selectedTokenBitmaps, const LocalTensor<uint8_t>& compareMask)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    __gm__ int32_t* dispatchFlags = nullptr;
    dispatchFlags = reinterpret_cast<__gm__ int32_t*>(peermem.dispatchFlag.get_ptr(commBuffer) +
                                                      RelayFlagOffset(originServer, tokenRange.start));
    // 前批 V_S 已确认向量读取结束；S_MTE2 将此边界传递到下一次输入覆盖。
    PipeFence<PIPE_S, PIPE_MTE2, SYNC_EVENT_ID2>();
    CopyGmToUb(reinterpret_cast<__ubuf__ int32_t*>((tokenBitmaps).GetPhyAddr()), dispatchFlags,
               CopyRows{static_cast<uint16_t>(tokenRange.count), sizeof(int32_t), (sizeof(int32_t)) + (sizeof(int32_t)),
                        sizeof(int32_t)});
    PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
    const uint32_t compareCount = Ops::Base::CeilAlign(tokenRange.count * static_cast<uint32_t>(sizeof(int32_t)),
                                                       static_cast<uint32_t>(ALIGN_256)) /
                                  sizeof(int32_t);
    CompareScalar(compareMask, tokenBitmaps, int32_t(0), CMPMODE::NE, compareCount);
    CreateVecIndex(tokenIds, static_cast<int32_t>(tokenRange.start), compareCount);
    auto gatherMaskPattern = compareMask.template ReinterpretCast<uint32_t>();
    uint64_t selectedTokenCount = 0U;
    // 位图比较和 ID 生成可与前批搬出重叠，仅覆盖筛选输出前等待 MTE3。
    PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID2>();
    // Counter 模式仅消费本批实际数量，尾部补齐 lane 的脏值不进入列表。
    GatherMask(selectedTokenIds, tokenIds, gatherMaskPattern, true, tokenRange.count, {1, 1, 0, 0}, selectedTokenCount);
    GatherMask(selectedTokenBitmaps, tokenBitmaps, gatherMaskPattern, true, tokenRange.count, {1, 1, 0, 0},
               selectedTokenCount);
    PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID2>();
    return selectedTokenCount;
}

/*
 * 每个 Server 行由一个 AIV0 顺序构建，固定 1024-token 批避免 UB 随 BS 增长。
 * UB 从 512B 起：位图、原始 ID、筛选 ID、筛选位图各 4KiB，比较掩码 128B，共 16512B。
 * W4 路径由调用方通过 V_S 等待 Prologue 收尾，覆盖空列表和无建表任务的核；
 * A8W8 的 Activation 返回前已等待输出搬运完成。UB 复用不依赖与旧缓冲区域错开。
 * 只在列表数量与行尾 count 一致且搬出完成后发布行头；未收齐时重新扫描。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildServerCombineLists()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    constexpr uint32_t BATCH_TOKENS = 1024U;
    constexpr uint32_t TOKEN_LIST_HEADER_ELEMENTS = 1U; // 行首保存有效 token 数量 + 1。
    constexpr uint32_t ID_BYTES = BATCH_TOKENS * sizeof(int32_t);
    constexpr uint32_t MASK_BYTES = BATCH_TOKENS / 8U;
    static_assert(ALIGN_512 + 4U * ID_BYTES + MASK_BYTES <= COMM_WORK_UB_LIMIT);
    LocalTensor<int32_t> tokenBitmaps(TPosition::VECCALC, ALIGN_512, BATCH_TOKENS);
    LocalTensor<int32_t> tokenIds(TPosition::VECCALC, ALIGN_512 + ID_BYTES, BATCH_TOKENS);
    LocalTensor<int32_t> selectedTokenIds(TPosition::VECCALC, ALIGN_512 + 2U * ID_BYTES, BATCH_TOKENS);
    LocalTensor<int32_t> selectedTokenBitmaps(TPosition::VECCALC, ALIGN_512 + 3U * ID_BYTES, BATCH_TOKENS);
    LocalTensor<uint8_t> compareMask(TPosition::VECCALC, ALIGN_512 + 4U * ID_BYTES, MASK_BYTES);
    const uint64_t maxTokensPerRank = static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank);
    // 先发布本核负责的全部 Server 列表，再进入贡献归约。
    for (uint32_t originServer = blockIdx_; originServer < serverNum_; originServer += blockNum_) {
        auto tokenCountAddr = reinterpret_cast<__gm__ uint64_t*>(peermem.dispatchFlag.get_ptr(commBuffer) +
                                                                 RelayFlagOffset(originServer, maxTokensPerRank));
        uint64_t encodedTokenCount = 0U;
        while ((encodedTokenCount = GmRead(tokenCountAddr)) == 0U) {
        }
        const uint32_t expectedTokenCount = static_cast<uint32_t>(encodedTokenCount - 1U);
        auto tokenListPtr =
            reinterpret_cast<__gm__ int32_t*>(workspace.urma.serverCombine.tokenIds.get_ptr(buffer) +
                                              originServer * CalcServerCombineTokenListStrideBytes(maxTokensPerRank));
        __gm__ int32_t* tokenList = nullptr;
        tokenList = tokenListPtr;
        __gm__ int32_t* bitmapList = nullptr;
        bitmapList = reinterpret_cast<__gm__ int32_t*>(workspace.urma.serverCombine.tokenBitmaps.get_ptr(buffer) +
                                                       static_cast<uint64_t>(originServer) * maxTokensPerRank *
                                                           sizeof(uint32_t));
        uint32_t collectedTokenCount = 0U;
        while (collectedTokenCount != expectedTokenCount) {
            collectedTokenCount = 0U;
            for (uint64_t batchTokenStart = 0; batchTokenStart < maxTokensPerRank; batchTokenStart += BATCH_TOKENS) {
                const WorkRange tokenRange{static_cast<uint32_t>(batchTokenStart),
                                           static_cast<uint32_t>(maxTokensPerRank - batchTokenStart < BATCH_TOKENS ?
                                                                     maxTokensPerRank - batchTokenStart :
                                                                     BATCH_TOKENS)};
                const uint64_t selectedTokenCount =
                    GatherCombineTokenIds(originServer, tokenRange, tokenBitmaps, tokenIds, selectedTokenIds,
                                          selectedTokenBitmaps, compareMask);
                if (selectedTokenCount != 0U) {
                    // 筛选结果由 V 写入，MTE3 搬出前等待其完成。
                    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
                    CopyUbToGm((tokenList + (TOKEN_LIST_HEADER_ELEMENTS + collectedTokenCount)),
                               reinterpret_cast<__ubuf__ int32_t*>((selectedTokenIds).GetPhyAddr()),
                               static_cast<uint32_t>(selectedTokenCount * sizeof(int32_t)));
                    CopyUbToGm((bitmapList + (collectedTokenCount)),
                               reinterpret_cast<__ubuf__ int32_t*>((selectedTokenBitmaps).GetPhyAddr()),
                               static_cast<uint32_t>(selectedTokenCount * sizeof(uint32_t)));
                    collectedTokenCount += static_cast<uint32_t>(selectedTokenCount);
                }
            }
        }
        // 重扫时由 MTE3_V 保护输出缓冲复用，仅发布行头前等待全部列表搬出。
        PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();
        GmWrite(tokenListPtr, static_cast<int32_t>(collectedTokenCount + 1U));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::IsServerContributionsReady(
    uint32_t originServer, const LocalTensor<uint32_t>& tokenIdsTensor,
    const LocalTensor<uint32_t>& expectedTopkBitmapsTensor, uint32_t batchTokenCount, uint32_t workUbOffset) const
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    constexpr uint32_t BATCH_TOKENS = 256U;
    constexpr uint32_t READY_ROW_ELEMENTS = 32U;
    constexpr uint32_t INPUT_ELEMENTS = BATCH_TOKENS * READY_ROW_ELEMENTS;
    constexpr uint32_t INPUT_BYTES = INPUT_ELEMENTS * sizeof(uint32_t);
    LocalTensor<uint32_t> contributionReadyTensor(TPosition::VECIN, workUbOffset, INPUT_ELEMENTS);
    LocalTensor<uint8_t> arrivedMaskTensor(TPosition::VECCALC, workUbOffset + INPUT_BYTES, INPUT_ELEMENTS / 8U);
    LocalTensor<uint32_t> pendingTensor(TPosition::VECCALC, workUbOffset + INPUT_BYTES + INPUT_ELEMENTS / 8U,
                                        ALIGN_32 / sizeof(uint32_t));
    // 前轮向量读取结束后，MTE2 才能覆盖整批 ready 工作区。
    PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID2>();
    for (uint32_t tokenIndex = 0; tokenIndex < batchTokenCount; ++tokenIndex) {
        const uint64_t tokenSlot = static_cast<uint64_t>(originServer) *
                                       static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank) +
                                   tokenIdsTensor.GetValue(tokenIndex);
        __gm__ uint32_t* contributionReadyGlobalTensor = nullptr;
        contributionReadyGlobalTensor = reinterpret_cast<__gm__ uint32_t*>(
            peermem.contributionReady.get_ptr(commBuffer) + tokenSlot * topK_ * sizeof(uint64_t));
        CopyGmToUb(reinterpret_cast<__ubuf__ uint32_t*>(
                       (contributionReadyTensor[tokenIndex * READY_ROW_ELEMENTS]).GetPhyAddr()),
                   contributionReadyGlobalTensor,
                   CopyRows{static_cast<uint16_t>(topK_), static_cast<uint32_t>(sizeof(uint32_t)),
                            (static_cast<uint32_t>(sizeof(uint32_t))) + (static_cast<uint32_t>(sizeof(uint32_t))),
                            static_cast<uint32_t>(sizeof(uint32_t))});
    }
    PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
    const uint32_t compareCount =
        Ops::Base::CeilAlign(batchTokenCount * READY_ROW_ELEMENTS * static_cast<uint32_t>(sizeof(uint32_t)),
                             static_cast<uint32_t>(ALIGN_256)) /
        sizeof(uint32_t);
    // 每 token 固定 32 个输入对应一个 u32 掩码；未加载 topK 尾位由 expected 掩盖。
    // 奇数尾批比较补齐到 256B，多比较的一行不参与结果判断。
    CompareScalar(arrivedMaskTensor, contributionReadyTensor, uint32_t(1), CMPMODE::EQ, compareCount);
    CheckServerContributionsVF(reinterpret_cast<__ubuf__ uint32_t*>(arrivedMaskTensor.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(expectedTopkBitmapsTensor.GetPhyAddr()),
                               reinterpret_cast<__ubuf__ uint32_t*>(pendingTensor.GetPhyAddr()), batchTokenCount);
    PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID2>();
    return pendingTensor.GetValue(0U) == 0U;
}

template <TemplateMegaMoeLayeredTypeClass>
/* 等待各 Server 的 token 列表发布，将有效 count 缓存到 UB，并返回总 token 数。 */
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::WaitForServerCombineTokenCounts(
    const LocalTensor<uint32_t>& tokenCountsTensor, uint64_t tokenListStrideBytes) const
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    uint64_t totalTokenCount = 0U;
    // 所有行头发布后列表长度才稳定，空列表的 encodedCount 为 1。
    for (uint32_t originServer = 0; originServer < serverNum_; ++originServer) {
        auto tokenListPtr =
            reinterpret_cast<__gm__ int32_t*>(workspace.urma.serverCombine.tokenIds.get_ptr(buffer) +
                                              static_cast<uint64_t>(originServer) * tokenListStrideBytes);
        int32_t encodedCount;
        do {
            encodedCount = GmRead(tokenListPtr);
        } while (encodedCount == 0);
        const uint32_t count = static_cast<uint32_t>(encodedCount - 1);
        tokenCountsTensor.SetValue(originServer, count);
        totalTokenCount += count;
    }
    return totalTokenCount;
}

/* 中继阶段由 AIV0 建表并归约，AIV1 等待各 Server 归约完成后回传。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineRelay()
{
    if (subBlockIdx_ == 0U) {
        if constexpr (MoeQuantConfig::AXW_MODE != AxWMode::A8W8) {
            // 无建表任务或空列表也会直接由 Scalar 缓存 count，先完成 Prologue 的旧 UB 访问。
            PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID2>();
        }
        BuildServerCombineLists();
        ReduceRelayContributions();
    } else {
        SendServerCombineResults();
    }
}

/* 分配 AIV0 中继归约任务和 UB，按 Server 列表范围分批处理有效 token。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReduceRelayContributions()
{
    constexpr uint32_t BATCH_TOKENS = 256U;
    constexpr uint32_t BATCH_BYTES = BATCH_TOKENS * sizeof(uint32_t);
    const uint64_t maxTokensPerRank = static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank);
    const uint64_t tokenListStrideBytes = CalcServerCombineTokenListStrideBytes(maxTokensPerRank);
    const uint32_t tokenCountsBytes =
        Ops::Base::CeilAlign(serverNum_ * static_cast<uint32_t>(sizeof(uint32_t)), static_cast<uint32_t>(ALIGN_32));
    // 从 512B 起连续放 count、256 个 token ID、256 个位图和复用工作区，均按 32B 对齐。
    // 最大 Server 数、H 和权重缓冲下共到 88704B，贡献检查与整 token 归约交替使用工作区。
    LocalTensor<uint32_t> tokenCountsTensor(TPosition::VECCALC, ALIGN_512, serverNum_);
    LocalTensor<uint32_t> tokenIdsTensor(TPosition::VECCALC, ALIGN_512 + tokenCountsBytes, BATCH_TOKENS);
    LocalTensor<uint32_t> expectedTopkBitmapsTensor(TPosition::VECCALC, ALIGN_512 + tokenCountsBytes + BATCH_BYTES,
                                                    BATCH_TOKENS);
    const uint32_t workUbOffset = ALIGN_512 + tokenCountsBytes + 2U * BATCH_BYTES;
    const uint64_t totalTokenCount = WaitForServerCombineTokenCounts(tokenCountsTensor, tokenListStrideBytes);
    // total <= Server 数 * B <= epWorldSize * B，已有 host 容量校验保证不超过 INT32_MAX。
    const auto tokenRange =
        GetBalancedWorkRange(static_cast<uint32_t>(totalTokenCount), {.jobIndex = blockIdx_, .totalJobs = blockNum_});
    if (tokenRange.count == 0U) {
        return;
    }
    ReduceRelayContributionsInRange(tokenRange, tokenCountsTensor, tokenIdsTensor, expectedTopkBitmapsTensor,
                                    tokenListStrideBytes, workUbOffset);
}

/* 将本核连续任务范围映射到各 Server 列表，按固定容量分批归约。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReduceRelayContributionsInRange(
    const WorkRange& tokenRange, const LocalTensor<uint32_t>& tokenCountsTensor,
    const LocalTensor<uint32_t>& tokenIdsTensor, const LocalTensor<uint32_t>& expectedTopkBitmapsTensor,
    uint64_t tokenListStrideBytes, uint32_t workUbOffset)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    constexpr uint32_t TOKEN_LIST_HEADER_ELEMENTS = 1U;
    const uint64_t maxTokensPerRank = static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank);
    const uint32_t batchTokens = tokenIdsTensor.GetSize();
    WorkRange serverTokenRange{0U, 0U};
    for (uint32_t originServer = 0; originServer < serverNum_; ++originServer) {
        if (serverTokenRange.start >= (tokenRange.start + tokenRange.count)) {
            break;
        }
        serverTokenRange.count = tokenCountsTensor.GetValue(originServer);
        const uint32_t serverTokenEnd = serverTokenRange.start + serverTokenRange.count;
        if (serverTokenEnd <= tokenRange.start) {
            serverTokenRange.start += serverTokenRange.count;
            continue;
        }
        auto tokenListPtr =
            reinterpret_cast<__gm__ int32_t*>(workspace.urma.serverCombine.tokenIds.get_ptr(buffer) +
                                              static_cast<uint64_t>(originServer) * tokenListStrideBytes);
        __gm__ uint32_t* tokenIdsGlobalTensor = nullptr;
        tokenIdsGlobalTensor = reinterpret_cast<__gm__ uint32_t*>(tokenListPtr + TOKEN_LIST_HEADER_ELEMENTS);
        __gm__ uint32_t* expectedTopkBitmapsGlobalTensor = nullptr;
        expectedTopkBitmapsGlobalTensor =
            reinterpret_cast<__gm__ uint32_t*>(workspace.urma.serverCombine.tokenBitmaps.get_ptr(buffer)) +
            static_cast<uint64_t>(originServer) * maxTokensPerRank;
        const uint32_t ownedStart =
            tokenRange.start > serverTokenRange.start ? tokenRange.start : serverTokenRange.start;
        const uint32_t ownedEnd =
            serverTokenEnd < tokenRange.start + tokenRange.count ? serverTokenEnd : tokenRange.start + tokenRange.count;
        const WorkRange ownedRange{ownedStart - serverTokenRange.start, ownedEnd - ownedStart};
        for (uint32_t batchOffset = 0U; batchOffset < ownedRange.count; batchOffset += batchTokens) {
            const uint32_t remaining = ownedRange.count - batchOffset;
            const WorkRange batchRange{ownedRange.start + batchOffset,
                                       remaining < batchTokens ? remaining : batchTokens};
            ReduceRelayContributionsForBatch(originServer, batchRange, tokenIdsGlobalTensor,
                                             expectedTopkBitmapsGlobalTensor, tokenIdsTensor, expectedTopkBitmapsTensor,
                                             workUbOffset);
        }
        if (originServer != serverId_ && ownedRange.count != 0U) {
            auto completedCountPtr = reinterpret_cast<__gm__ int32_t*>(
                workspace.urma.serverCombine.completedCounts.get_ptr(buffer) +
                static_cast<uint64_t>(originServer) * INT_CACHELINE * sizeof(int32_t));
            // 各 token 的 MTE3_S 已确认结果写回，再汇总本核负责的完成数量。
            AtomicAdd(completedCountPtr, static_cast<int32_t>(ownedRange.count));
        }
        serverTokenRange.start += serverTokenRange.count;
    }
}

/* 搬入中继归约的一批 token 信息，等待本批贡献全部到齐，再按列表顺序归约。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReduceRelayContributionsForBatch(
    uint32_t originServer, const WorkRange& batchRange, __gm__ uint32_t* tokenIdsGlobalTensor,
    __gm__ uint32_t* expectedTopkBitmapsGlobalTensor, const LocalTensor<uint32_t>& tokenIdsTensor,
    const LocalTensor<uint32_t>& expectedTopkBitmapsTensor, uint32_t workUbOffset)
{
    // 上批 Scalar 已结束缓存读写，整批搬入连续 token ID 和对应位图。
    PipeFence<PIPE_S, PIPE_MTE2, SYNC_EVENT_ID2>();
    CopyGmToUb(reinterpret_cast<__ubuf__ uint32_t*>((tokenIdsTensor).GetPhyAddr()),
               (tokenIdsGlobalTensor + (batchRange.start)), batchRange.count * static_cast<uint32_t>(sizeof(uint32_t)));
    CopyGmToUb(reinterpret_cast<__ubuf__ uint32_t*>((expectedTopkBitmapsTensor).GetPhyAddr()),
               (expectedTopkBitmapsGlobalTensor + (batchRange.start)),
               batchRange.count * static_cast<uint32_t>(sizeof(uint32_t)));
    PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID2>();
    // 整批搬入并比较所有 token 的 ready，全部到齐后再按列表顺序归约。
    while (!IsServerContributionsReady(originServer, tokenIdsTensor, expectedTopkBitmapsTensor, batchRange.count,
                                       workUbOffset)) {
    }
    for (uint32_t tokenIndex = 0; tokenIndex < batchRange.count; ++tokenIndex) {
        ReduceRelayContributionsForToken(originServer, tokenIdsTensor.GetValue(tokenIndex),
                                         expectedTopkBitmapsTensor.GetValue(tokenIndex), workUbOffset);
    }
}

/*
 * 按 topK 槽递增归约一个原始输入 token；需要中继乘权时一次搬入权重，FP32 累加后转 BF16。
 * 工作区依次为可选权重、BF16 贡献、FP32 输入和累加值，各区按 32B 对齐。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReduceRelayContributionsForToken(
    uint32_t originServer, uint32_t tokenIdx, uint32_t expectedTopkBitmap, uint32_t workUbOffset)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    const uint64_t capacity = static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank);
    const uint64_t tokenBytes = static_cast<uint64_t>(k_) * sizeof(bfloat16_t);
    const uint64_t tokenSlot = static_cast<uint64_t>(originServer) * capacity + tokenIdx;
    LocalTensor<float> topkWeightsTensor;
    if constexpr (!TopkWeightsPrefetch) {
        topkWeightsTensor = LocalTensor<float>(TPosition::VECIN, workUbOffset, topK_);
        workUbOffset +=
            Ops::Base::CeilAlign(topK_ * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        __gm__ float* topkWeightsGlobalTensor = nullptr;
        topkWeightsGlobalTensor = reinterpret_cast<__gm__ float*>(peermem.dispatchReceive.get_ptr(commBuffer) +
                                                                  RelayTokenOffset(originServer, tokenIdx) +
                                                                  mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_);
        PipeFence<PIPE_S, PIPE_MTE2, SYNC_EVENT_ID2>();
        CopyGmToUb(reinterpret_cast<__ubuf__ float*>((topkWeightsTensor).GetPhyAddr()), topkWeightsGlobalTensor,
                   topK_ * static_cast<uint32_t>(sizeof(float)));
        PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID2>();
    }
    const uint32_t contributionBytes =
        Ops::Base::CeilAlign(k_ * static_cast<uint32_t>(sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    const uint32_t floatBytes =
        Ops::Base::CeilAlign(k_ * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<bfloat16_t> contributionTensor(TPosition::VECIN, workUbOffset, k_);
    LocalTensor<float> inputTensor(TPosition::VECCALC, workUbOffset + contributionBytes, k_);
    LocalTensor<float> sumTensor(TPosition::VECCALC, workUbOffset + contributionBytes + floatBytes, k_);
    __gm__ bfloat16_t* contributionGlobalTensor = nullptr;
    contributionGlobalTensor =
        reinterpret_cast<__gm__ bfloat16_t*>(peermem.contribution.get_ptr(commBuffer) + tokenSlot * topK_ * tokenBytes);
    // 跨 Server 结果复用当前 token 的槽 0，整 token 读完全部有效贡献后才覆盖。
    __gm__ bfloat16_t* resultGlobalTensor = nullptr;
    resultGlobalTensor = reinterpret_cast<__gm__ bfloat16_t*>(
        originServer == serverId_ ? peermem.combineReceive.get_ptr(commBuffer) +
                                        (static_cast<uint64_t>(tokenIdx) * serverNum_ + serverId_) * tokenBytes :
                                    peermem.contribution.get_ptr(commBuffer) + tokenSlot * topK_ * tokenBytes);
    AccumulateRelayContributions(expectedTopkBitmap, contributionGlobalTensor, contributionTensor, inputTensor,
                                 sumTensor, topkWeightsTensor);
    Cast(contributionTensor, sumTensor, RoundMode::CAST_RINT, k_);
    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
    CopyUbToGm(resultGlobalTensor, reinterpret_cast<__ubuf__ bfloat16_t*>((contributionTensor).GetPhyAddr()),
               k_ * static_cast<uint32_t>(sizeof(bfloat16_t)));
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();
}

/* 按 topK 槽递增搬入有效贡献，按需乘权并在 FP32 中累加。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AccumulateRelayContributions(
    uint32_t expectedTopkBitmap, __gm__ bfloat16_t* contributionGlobalTensor,
    const LocalTensor<bfloat16_t>& contributionTensor, const LocalTensor<float>& inputTensor,
    const LocalTensor<float>& sumTensor, const LocalTensor<float>& topkWeightsTensor)
{
    Duplicate(sumTensor, 0.0f, k_);
    for (uint32_t topk = 0; topk < topK_; ++topk) {
        if ((expectedTopkBitmap & (uint32_t{1} << topk)) == 0U) {
            continue;
        }
        PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID2>();
        CopyGmToUb(reinterpret_cast<__ubuf__ bfloat16_t*>((contributionTensor).GetPhyAddr()),
                   (contributionGlobalTensor + (static_cast<uint64_t>(topk) * k_)),
                   k_ * static_cast<uint32_t>(sizeof(bfloat16_t)));
        PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
        Cast(inputTensor, contributionTensor, RoundMode::CAST_NONE, k_);
        if constexpr (!TopkWeightsPrefetch) {
            Muls(inputTensor, inputTensor, topkWeightsTensor.GetValue(topk), k_);
        }
        Add(sumTensor, sumTensor, inputTensor, k_);
    }
}

/* 固定 channel owner 按 Server 顺序等待归约完成，再按列表分批回传。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendServerCombineResults()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    constexpr uint32_t BATCH_TOKENS = 256U;
    constexpr uint32_t TOKEN_LIST_HEADER_ELEMENTS = 1U;
    const uint64_t capacity = static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank);
    LocalTensor<uint32_t> tokenIdsTensor(TPosition::VECIN, ALIGN_512, BATCH_TOKENS);
    for (uint32_t targetServer = 0; targetServer < serverNum_; ++targetServer) {
        const uint32_t targetRank = targetServer * rankNumPerServer_ + rankIdInServer_;
        if (targetRank == rankId_ || targetRank >= worldSize_ || !IsChannelOwner(targetRank)) {
            continue;
        }
        auto tokenListPtr = reinterpret_cast<__gm__ int32_t*>(workspace.urma.serverCombine.tokenIds.get_ptr(buffer) +
                                                              static_cast<uint64_t>(targetServer) *
                                                                  CalcServerCombineTokenListStrideBytes(capacity));
        int32_t encodedCount;
        do {
            encodedCount = GmRead(tokenListPtr);
        } while (encodedCount == 0);
        const uint32_t count = static_cast<uint32_t>(encodedCount - 1);
        if (count == 0U) {
            continue;
        }
        auto completedCountPtr =
            reinterpret_cast<__gm__ int32_t*>(workspace.urma.serverCombine.completedCounts.get_ptr(buffer) +
                                              static_cast<uint64_t>(targetServer) * INT_CACHELINE * sizeof(int32_t));
        while (GmRead(completedCountPtr) != static_cast<int32_t>(count)) {
        }
        __gm__ uint32_t* tokenIdsGlobalTensor = nullptr;
        tokenIdsGlobalTensor = reinterpret_cast<__gm__ uint32_t*>(tokenListPtr + TOKEN_LIST_HEADER_ELEMENTS);
        SendRelayResultsToTargetRank(targetServer, targetRank, count, BATCH_TOKENS, tokenIdsGlobalTensor,
                                     tokenIdsTensor);
    }
}

/* 已完成的一个 Server 列表只提交一次，按批追加请求，由阶段出口统一 Drain。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendRelayResultsToTargetRank(
    uint32_t targetServer, uint32_t targetRank, uint32_t count, uint32_t batchTokens,
    __gm__ uint32_t* tokenIdsGlobalTensor, const LocalTensor<uint32_t>& tokenIdsTensor)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    const uint64_t capacity = static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank);
    const uint64_t tokenBytes = static_cast<uint64_t>(k_) * sizeof(bfloat16_t);
    const auto channel = comm_.GetChannel(targetRank);
    auto batch = comm_.MakeBatchHandle(channel, GetRankWinAddrWithOffset(targetRank, 0));
    for (uint32_t entry = 0; entry < count; entry += batchTokens) {
        const uint32_t remaining = count - entry;
        const uint32_t batchCount = remaining < batchTokens ? remaining : batchTokens;
        PipeFence<PIPE_S, PIPE_MTE2, SYNC_EVENT_ID2>();
        CopyGmToUb(reinterpret_cast<__ubuf__ uint32_t*>((tokenIdsTensor).GetPhyAddr()),
                   (tokenIdsGlobalTensor + (entry)), batchCount * static_cast<uint32_t>(sizeof(uint32_t)));
        PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID2>();
        for (uint32_t index = 0; index < batchCount; ++index) {
            const uint32_t tokenIdx = tokenIdsTensor.GetValue(index);
            const uint64_t txSlot = static_cast<uint64_t>(targetServer) * capacity + tokenIdx;
            const uint64_t rxSlot = static_cast<uint64_t>(tokenIdx) * serverNum_ + serverId_;
            comm_.WriteNbi(
                batch, GetRankWinAddrWithOffset(targetRank, peermem.combineReceive.offsetBytes + rxSlot * tokenBytes),
                peermem.contribution.get_ptr(commBuffer) + txSlot * topK_ * tokenBytes, k_ * sizeof(bfloat16_t));
        }
        comm_.BatchCommit(batch);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteServerCombine()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    const uint32_t floatBytes =
        Ops::Base::CeilAlign(k_ * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<float> inputTensor(TPosition::VECIN, ALIGN_512, k_);
    LocalTensor<float> sumTensor(TPosition::VECCALC, ALIGN_512 + floatBytes, k_);
    LocalTensor<bfloat16_t> bf16Tensor(TPosition::VECIN, ALIGN_512 + 2U * floatBytes, k_);
    __gm__ OutputType* outputGlobalTensor = nullptr;
    outputGlobalTensor = reinterpret_cast<__gm__ OutputType*>(params_.y2GmAddr);
    LocalTensor<OutputType> outputTensor = bf16Tensor.template ReinterpretCast<OutputType>();
    __gm__ bfloat16_t* combineResultsGlobalTensor = nullptr;
    combineResultsGlobalTensor = reinterpret_cast<__gm__ bfloat16_t*>(peermem.combineReceive.get_ptr(commBuffer));
    __gm__ bfloat16_t* sharedResultsGlobalTensor = nullptr;
    sharedResultsGlobalTensor = reinterpret_cast<__gm__ bfloat16_t*>(workspace.shared.gmm2Output.get_ptr(buffer));
    const WorkRange tokenRange = GetBalancedWorkRange(m_, {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_});
    uint32_t readySharedGroup = UINT32_MAX; // 每次调用重新开始，不复用上轮就绪状态。
    for (uint32_t tokenOffset = 0U; tokenOffset < tokenRange.count; ++tokenOffset) {
        const uint32_t token = tokenRange.start + tokenOffset;
        Duplicate(sumTensor, 0.0f, k_);
        PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID2>();
        // 预置 bf16 缓冲可搬入事件，Server 和共享专家循环共用此链。
        PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
        for (uint32_t server = 0; server < serverNum_; ++server) {
            auto bitmapPtr = reinterpret_cast<__gm__ uint64_t*>(peermem.sourceReady.get_ptr(commBuffer) +
                                                                RelayFlagOffset(server, token));
            if (static_cast<uint32_t>(GmRead(bitmapPtr)) == 0U) {
                continue;
            }
            PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
            CopyGmToUb(reinterpret_cast<__ubuf__ bfloat16_t*>((bf16Tensor).GetPhyAddr()),
                       (combineResultsGlobalTensor + ((static_cast<uint64_t>(token) * serverNum_ + server) * k_)),
                       k_ * static_cast<uint32_t>(sizeof(bfloat16_t)));
            PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
            Cast(inputTensor, bf16Tensor, RoundMode::CAST_NONE, k_);
            PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
            Add(sumTensor, sumTensor, inputTensor, k_);
        }
        const uint32_t tokenGroup = token / COMBINE_TOKEN_GROUP_SIZE;
        const bool waitForGroup = tokenGroup != readySharedGroup;
        for (uint32_t expert = 0; expert < sharedExpertNum_; ++expert) {
            if (waitForGroup) {
                WaitForSharedExpertResult(expert, token);
            }
            PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
            CopyGmToUb(reinterpret_cast<__ubuf__ bfloat16_t*>((bf16Tensor).GetPhyAddr()),
                       (sharedResultsGlobalTensor + ((static_cast<uint64_t>(expert) * m_ + token) * k_)),
                       k_ * static_cast<uint32_t>(sizeof(bfloat16_t)));
            PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
            Cast(inputTensor, bf16Tensor, RoundMode::CAST_NONE, k_);
            PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
            Add(sumTensor, sumTensor, inputTensor, k_);
        }
        readySharedGroup = tokenGroup;
        // 回收最后一个事件；没有输入时回收入口预置事件。
        PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
        Cast(outputTensor, sumTensor, RoundMode::CAST_RINT, k_);
        PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
        CopyUbToGm((outputGlobalTensor + (static_cast<uint64_t>(token) * k_)),
                   reinterpret_cast<__ubuf__ OutputType*>((outputTensor).GetPhyAddr()),
                   k_ * static_cast<uint32_t>(sizeof(OutputType)));
    }
    // 返回前等待本核最后一次输出搬运完成。
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();
}

} // namespace MegaMoeImpl
#endif
