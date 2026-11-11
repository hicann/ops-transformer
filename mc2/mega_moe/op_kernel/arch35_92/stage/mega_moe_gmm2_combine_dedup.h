/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GMM2_COMBINE_DEDUP_H
#define MEGA_MOE_GMM2_COMBINE_DEDUP_H

#include "../common/mega_moe_sync.h"
#include "mega_moe_gmm2_combine.h"

namespace MegaMoeImpl {

// ============================== combine 去重 ==============================
// 干什么：基线 combine 里，同一 token 在本卡的多个专家输出行要逐行发回源卡、由源卡累加。
// combine 去重把这些行在本卡先合并成一行再发，发送量从「组内行数」降到 1。
// 怎么跑：GMM2 的输出本来就全量落在 workspace（gmm2MmadRes）里，combine 作为一个独立
// 阶段按专家从小到大消费。prepass 建表时保证了组内成员按专家升序、组内最大专家的那行
// 是 LAST（合并驱动行）——所以轮到专家 e 时，e 组内的其它成员必然已经算完落盘，
// 等一个 WaitWaveGmm2Ready(e) 就等于成员全就绪，不需要任何跨专家的等待标记。
// 怎么合：提前乘权开启时，各成员已在激活量化阶段乘权；关闭时，在合并阶段按成员的 topk 权重乘加。
// 合并使用 fp32 整行累加，加完一次 Cast、发到 (token, minTopkIdx) 槽。
// 三类行：PLAIN（不在组里）走基线的行环直发；SKIP（组内非 LAST）直接跳过；LAST 走上面
// 的合并发送。desc 描述表按批装载，每批只花一次 MTE2。
// ==========================================================================

// 成员行/累加按段处理：段长固定，缓冲不随 h 增长，避免整行 fp32 双缓冲挤爆 UB。
constexpr uint32_t WAVE_COMBINE_DEDUP_SEG_ELEMS = 4096U;
constexpr uint32_t WAVE_COMBINE_DEDUP_DESC_UB_BYTES = 12U * 1024U;
// tkw=0 乘权用：LAST 行 revWeights 槽（整 token topK 个 fp32 位型权重）的 UB 装载区，
// topK ≤ 32（host 校验）→ 128B 定长。
constexpr uint32_t WAVE_COMBINE_DEDUP_WEIGHT_UB_BYTES = 128U;
/*
 * 成员乒乓装载事件（MTE2_V/V_MTE2）与合并发送 ping/pong 事件（V_MTE3/MTE3_V）均用 6/7 双 id；
 * 与行环 slot 事件（MTE3_MTE2 的 0..5）分属不同 pipe 对或不同 id，互不冲突。
 */
constexpr int32_t WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE = 6;
constexpr int32_t WAVE_COMBINE_DEDUP_MERGE_EVENT_BASE = 6;
/*
 * desc 批区与权重区的外移段基址：紧随 GMM2-ready 检查区（scan/reduce/sum 三段）之后按 512B
 * 对齐起排，止于 combine meta-info 区之前。把这两段固定开销挪出
 * [WAVE_COMBINE_UB_BASE, WAVE_COMBINE_UB_LIMIT) 窗口，窗口让位给随 h 增长的行宽类缓冲
 * （合并发送 ping/pong 占两份行宽后，容量上限仍有 h ≤ 9216）。
 */
constexpr uint32_t WAVE_COMBINE_DEDUP_SIDE_UB_BASE = static_cast<uint32_t>(
    (WAVE_GMM2_READY_SUM_UB_ADDR + INT_CACHELINE * sizeof(int32_t) + ALIGN_512 - 1) / ALIGN_512 * ALIGN_512);
static_assert(WAVE_COMBINE_DEDUP_SIDE_UB_BASE + WAVE_COMBINE_DEDUP_DESC_UB_BYTES + WAVE_COMBINE_DEDUP_WEIGHT_UB_BYTES <=
                  META_INFO_TENSOR_ADDR,
              "dedup desc/weight side region overruns combine meta-info base");

struct WaveCombineDedupScratch {
    __ubuf__ bfloat16_t* mergeSendTensor = nullptr; // 整行 bf16 合并结果 ×2 ping/pong（发送源，行间解串行）
    uint32_t mergeRowSequence = 0U;                 // LAST 行序号（选 ping/pong 半区与事件 id）
    __ubuf__ bfloat16_t* memberTensor = nullptr; // 2 × 段长 bf16 成员装载乒乓
    __ubuf__ float* memberFp32Tensor = nullptr;  // 段长 fp32 成员转换暂存
    __ubuf__ float* accumTensor = nullptr;       // 段长 fp32 累加
    __ubuf__ int32_t* descTensor = nullptr;      // rowDesc 批装载
    __ubuf__ int32_t* weightTensor = nullptr;    // tkw=0：LAST 行整 token 权重段装载（行内自闭环）
};

// 去重 combine 用到的 GM 视图：rowDesc 表、GMM2 结果（绝对基址/本专家段基址）与远端发送基偏移。
struct WaveCombineDedupGmViews {
    __gm__ int32_t* rowDescGm = nullptr;
    __gm__ bfloat16_t* gmm2AbsGm = nullptr;
    __gm__ bfloat16_t* gmm2OutGm = nullptr;
    uint64_t gmRemoteBaseOffset = 0;
};

/*
 * 成员乒乓（V_MTE2 事件 6/7）与合并发送（MTE3_V 事件 6/7）走「先挂再等」协议：进 wave 前先
 * Set 一次（本函数），之后每次复写缓冲前无条件 Wait、用完再 Set，wave 结束由
 * DrainCombineRowBuffersDedup 无条件 Wait 收尾。这样任何时刻每个事件 id 上恰有一个在途
 * Set：首轮 Wait 等到的是这里预挂的 Set（不死等），结束时全部被 Drain 消费（无悬挂残留）。
 * 调用本函数的开关条件必须与调用 Drain 的条件完全一致，否则一边挂一边不收，直接死等。
 */
__aicore__ inline void ArmWaveCombineDedupEvents()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE));
    PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE + 1));
    PipeNotify<PIPE_MTE3, PIPE_V>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MERGE_EVENT_BASE));
    PipeNotify<PIPE_MTE3, PIPE_V>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MERGE_EVENT_BASE + 1));
}

/*
 * 在行环之后依次落位去重的窗口内各段：合并发送整行 ping/pong → 成员乒乓 → fp32 转换 → 累加；
 * desc 批区与权重区固定落在窗口外的外移段（WAVE_COMBINE_DEDUP_SIDE_UB_BASE 起）。
 */
__aicore__ inline void LayoutWaveCombineDedupScratch(WaveCombineDedupScratch& dedupScratch, uint32_t addr,
                                                     uint32_t rowStrideBytes)
{
    constexpr uint32_t segBf16Bytes = WAVE_COMBINE_DEDUP_SEG_ELEMS * sizeof(bfloat16_t);
    constexpr uint32_t segFp32Bytes = WAVE_COMBINE_DEDUP_SEG_ELEMS * sizeof(float);
    dedupScratch.mergeSendTensor = reinterpret_cast<__ubuf__ bfloat16_t*>(addr);
    addr += 2U * rowStrideBytes;
    dedupScratch.memberTensor = reinterpret_cast<__ubuf__ bfloat16_t*>(addr);
    addr += 2U * segBf16Bytes;
    dedupScratch.memberFp32Tensor = reinterpret_cast<__ubuf__ float*>(addr);
    addr += segFp32Bytes;
    dedupScratch.accumTensor = reinterpret_cast<__ubuf__ float*>(addr);
    dedupScratch.descTensor = reinterpret_cast<__ubuf__ int32_t*>(WAVE_COMBINE_DEDUP_SIDE_UB_BASE);
    dedupScratch.weightTensor =
        reinterpret_cast<__ubuf__ int32_t*>(WAVE_COMBINE_DEDUP_SIDE_UB_BASE + WAVE_COMBINE_DEDUP_DESC_UB_BYTES);
}

/*
 * 去重专属 UB 规划（[WAVE_COMBINE_UB_BASE, WAVE_COMBINE_UB_LIMIT) 内动态切分）：
 * 行环(n×行宽) + 合并发送整行 ping/pong + 成员乒乓/fp32/累加三段（段长固定）；
 * desc 批区与权重区在窗口外的外移段，不占本窗口。
 * host 容量门已保证 h 满足容量约束（4×行宽 + 固定 49152B ≤ 122880B，即 h ≤ 9216）。
 */
__aicore__ inline WaveCombineBufferConfig InitWaveCombineBuffersDedup(const MoeStageCommonConfig& common,
                                                                      WaveCombineScratch& scratch,
                                                                      WaveCombineDedupScratch& dedupScratch)
{
    WaveCombineBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }
    bufferConfig.rowBytes = common.tokenHiddenDim * static_cast<uint32_t>(sizeof(bfloat16_t));
    bufferConfig.rowStrideBytes = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(bufferConfig.rowBytes), static_cast<uint64_t>(ALIGN_32)));
    bufferConfig.slotStrideBytes = bufferConfig.rowStrideBytes;
    constexpr uint32_t segBf16Bytes = WAVE_COMBINE_DEDUP_SEG_ELEMS * sizeof(bfloat16_t);
    constexpr uint32_t segFp32Bytes = WAVE_COMBINE_DEDUP_SEG_ELEMS * sizeof(float);
    /*
     * 窗口内固定开销：合并发送整行 ping/pong（2×行宽）+ 成员乒乓（2×段长 bf16）
     * + fp32 转换与累加（2×段长 fp32）；desc 批区与权重区在窗口外的外移段，不计入。
     */
    const uint32_t fixedBytes = 2U * bufferConfig.rowStrideBytes + 2U * segBf16Bytes + 2U * segFp32Bytes;
    // 纵深防御：host 门失效（h 超界）时无符号减法会回绕成巨大预算→ring 钳到 6→静默写穿
    // ready-scan 区；此处先钳零，让越界表现为固定的 min-ring 布局而非任意写穿。
    constexpr uint32_t totalBudgetBytes = WAVE_COMBINE_UB_LIMIT - WAVE_COMBINE_UB_BASE;
    const uint32_t ringBudgetBytes = fixedBytes < totalBudgetBytes ? totalBudgetBytes - fixedBytes : 0U;
    bufferConfig.rowBufferCount = ringBudgetBytes / bufferConfig.slotStrideBytes;
    bufferConfig.rowBufferCount = bufferConfig.rowBufferCount < WAVE_COMBINE_MIN_ROW_BUFFER_COUNT ?
                                      WAVE_COMBINE_MIN_ROW_BUFFER_COUNT :
                                      bufferConfig.rowBufferCount;
    bufferConfig.rowBufferCount = bufferConfig.rowBufferCount > WAVE_COMBINE_MAX_ROW_BUFFER_COUNT ?
                                      WAVE_COMBINE_MAX_ROW_BUFFER_COUNT :
                                      bufferConfig.rowBufferCount;
    uint32_t rowRingBytes = bufferConfig.rowBufferCount * bufferConfig.slotStrideBytes;
    scratch.rowBufferTensor = reinterpret_cast<__ubuf__ bfloat16_t*>(WAVE_COMBINE_UB_BASE);
    scratch.metaInfoTensor = reinterpret_cast<__ubuf__ int32_t*>(META_INFO_TENSOR_ADDR);
    LayoutWaveCombineDedupScratch(dedupScratch, WAVE_COMBINE_UB_BASE + rowRingBytes, bufferConfig.rowStrideBytes);
    return bufferConfig;
}

// tkw=0：把 LAST 行（本行）revWeights 槽的整 token 权重段一次连续读回 weightTensor。
// 权重段由 dispatch 侧落盘；读取贯穿整组累加，放行事件（S_MTE2）由调用方在组尾配平。
__aicore__ inline void LoadDedupMergeWeightBits(WaveCombineDedupScratch& dedupScratch, const Params& params,
                                                uint32_t descRowBase, uint32_t memberCount, TEventID mergeEvent)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    const uint32_t topK = static_cast<uint32_t>(params.tilingData->compute.topK);
    const uint32_t weightAlignBytes =
        Ops::Base::CeilAlign(topK * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    const uint64_t lastRow = static_cast<uint64_t>(static_cast<uint32_t>(
        dedupScratch.descTensor[descRowBase + DEDUP_ROW_DESC_HEADER_INT32 + (memberCount - 1U) * 3U]));
    __gm__ int32_t* revWeightsGm = nullptr;
    revWeightsGm =
        reinterpret_cast<__gm__ int32_t*>(workspace.routing.tokenWeights.get_ptr(buffer) + lastRow * weightAlignBytes);
    CopyGmToUb(dedupScratch.weightTensor, revWeightsGm, topK * static_cast<uint32_t>(sizeof(float)));
    PipeNotify<PIPE_MTE2, PIPE_S>(mergeEvent);
    PipeWait<PIPE_MTE2, PIPE_S>(mergeEvent);
}

// 取第 memberIdx 个成员的合并权重：desc 给出该成员的 topkIndex，从整 token 权重段按位取回 float。
__aicore__ inline float GetDedupMemberWeight(const WaveCombineDedupScratch& dedupScratch, const Params& params,
                                             uint32_t descRowBase, uint32_t memberIdx)
{
    const int32_t memberTopkIdx =
        dedupScratch.descTensor[descRowBase + DEDUP_ROW_DESC_HEADER_INT32 + memberIdx * 3U + 1U];
    const int32_t weightBits =
        dedupScratch.weightTensor[static_cast<uint32_t>(memberTopkIdx) % params.tilingData->compute.topK];
    return *reinterpret_cast<const float*>(&weightBits);
}

/*
 * 发起第 issueIdx 个成员行在当前 h 段的半区装载：V_MTE2=半区可复写（Arm 预挂首轮）、
 * MTE2_V=装载完成；半区按 issueIdx 奇偶轮换。
 */
__aicore__ inline void IssueDedupMemberLoad(WaveCombineDedupScratch& dedupScratch, __gm__ bfloat16_t* gmm2AbsGm,
                                            uint32_t descRowBase, uint32_t issueIdx, uint32_t h, uint32_t segBase,
                                            uint32_t segBytes)
{
    const uint64_t issueRow = static_cast<uint64_t>(
        static_cast<uint32_t>(dedupScratch.descTensor[descRowBase + DEDUP_ROW_DESC_HEADER_INT32 + issueIdx * 3U]));
    const uint32_t issueBuf = issueIdx & 1U;
    const TEventID issueEvent =
        static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE + static_cast<int32_t>(issueBuf));
    // 等该半区上一次的 Cast 读完（首两轮等到的是预挂 Set），才能复写发起装载。
    PipeWait<PIPE_V, PIPE_MTE2>(issueEvent);
    auto* issueUb = dedupScratch.memberTensor + issueBuf * WAVE_COMBINE_DEDUP_SEG_ELEMS;
    CopyGmToUb(issueUb, gmm2AbsGm + issueRow * h + segBase, segBytes);
    PipeNotify<PIPE_MTE2, PIPE_V>(issueEvent);
}

/*
 * 消费第 memberIdx 个成员已装载的半区：fp32 累加进 accumTensor（tkw=0 先按成员权重乘；
 * memberIdx=0 直写初始化累加区），消费完释放半区（V_MTE2）。
 */
__aicore__ inline void ConsumeDedupMemberSegment(WaveCombineDedupScratch& dedupScratch, const Params& params,
                                                 uint32_t descRowBase, uint32_t memberIdx, bool mulWeights,
                                                 uint32_t segLen)
{
    const uint32_t buf = memberIdx & 1U;
    const TEventID memberEvent =
        static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE + static_cast<int32_t>(buf));
    PipeWait<PIPE_MTE2, PIPE_V>(memberEvent);
#if defined(MEGA_MOE_ARCH92_DEVICE)
    LocalTensor<bfloat16_t> memberUb(
        TPosition::VECIN, reinterpret_cast<uint64_t>(dedupScratch.memberTensor + buf * WAVE_COMBINE_DEDUP_SEG_ELEMS),
        segLen);
    LocalTensor<float> accum(TPosition::VECCALC, reinterpret_cast<uint64_t>(dedupScratch.accumTensor), segLen);
    LocalTensor<float> memberFp32(TPosition::VECCALC, reinterpret_cast<uint64_t>(dedupScratch.memberFp32Tensor),
                                  segLen);
    if (memberIdx == 0U) {
        Cast(accum, memberUb, RoundMode::CAST_NONE, segLen);
        if (mulWeights) {
            Muls(accum, accum, GetDedupMemberWeight(dedupScratch, params, descRowBase, 0U), segLen);
        }
    } else {
        Cast(memberFp32, memberUb, RoundMode::CAST_NONE, segLen);

        if (mulWeights) {
            Muls(memberFp32, memberFp32, GetDedupMemberWeight(dedupScratch, params, descRowBase, memberIdx), segLen);
        }
        Add(accum, accum, memberFp32, segLen);
    }
#else
    if constexpr (g_coreType == AIV) {
        auto* memberUb = dedupScratch.memberTensor + buf * WAVE_COMBINE_DEDUP_SEG_ELEMS;
        auto* accum = dedupScratch.accumTensor;
        if (mulWeights) {
            float memberWeight = GetDedupMemberWeight(dedupScratch, params, descRowBase, memberIdx);
            if (memberIdx == 0U) {
                AccumulateBf16ExpertSlotVF<true, true>(accum, memberUb, memberWeight, segLen);
            } else {
                AccumulateBf16ExpertSlotVF<false, true>(accum, memberUb, memberWeight, segLen);
            }
        } else {
            if (memberIdx == 0U) {
                AccumulateBf16ExpertSlotVF<true, false>(accum, memberUb, 0.0f, segLen);
            } else {
                AccumulateBf16ExpertSlotVF<false, false>(accum, memberUb, 0.0f, segLen);
            }
        }
    }
#endif

    PipeNotify<PIPE_V, PIPE_MTE2>(memberEvent);
}

/*
 * 对一个 h 段读齐组内全部成员并 fp32 累加进 accumTensor（tkw=0 先按成员权重乘）。
 * 前瞻双缓冲：先发起成员 i+1 的装载再消费成员 i，MTE2 与 V 重叠（原写法为发起→立等→计算
 * 的全串行，成员装载时间全部裸露在关键路径上）。首轮只发起、末轮只消费，各自单列在主循环外。
 * 事件协议不变：任何时刻每个事件 id 恰一个在途 Set，Arm/Drain 配平不变（成员乒乓事件跨行
 * 存活，ArmWaveCombineDedupEvents 已预挂首轮 Set，此处直接 Wait）。
 * fp32 累加顺序严格保持 memberIdx 升序（desc 表序）——与 golden 全序对齐契约一致，勿动。
 */
__aicore__ inline void AccumulateDedupMemberSegment(WaveCombineDedupScratch& dedupScratch, __gm__ bfloat16_t* gmm2AbsGm,
                                                    const Params& params, uint32_t descRowBase, uint32_t memberCount,
                                                    bool mulWeights, uint32_t h, uint32_t segBase, uint32_t segLen)
{
    if (memberCount == 0U) {
        return;
    }
    const uint32_t segBytes = segLen * static_cast<uint32_t>(sizeof(bfloat16_t));
    // 首轮：只发起成员 0 的装载。
    IssueDedupMemberLoad(dedupScratch, gmm2AbsGm, descRowBase, 0U, h, segBase, segBytes);
    for (uint32_t memberIdx = 1U; memberIdx < memberCount; ++memberIdx) {
        IssueDedupMemberLoad(dedupScratch, gmm2AbsGm, descRowBase, memberIdx, h, segBase, segBytes);
        ConsumeDedupMemberSegment(dedupScratch, params, descRowBase, memberIdx - 1U, mulWeights, segLen);
    }
    // 末轮：只消费最后一个成员。
    ConsumeDedupMemberSegment(dedupScratch, params, descRowBase, memberCount - 1U, mulWeights, segLen);
}

/*
 * 合并驱动行（LAST）：按段读齐组内全部成员（含本行）→ fp32 累加 → Cast 进整行发送缓冲，
 * 一次发到 (tokenIdx*topK + minTopkIdx) 槽。成员行号是全局接收行号，直接对 gmm2MmadRes 绝对基址寻址。
 */
__aicore__ inline void SendWaveCombineTokenDedupMerge(const MoeStageCommonConfig& common,
                                                      WaveCombineDedupScratch& dedupScratch, const Params& params,
                                                      __gm__ bfloat16_t* gmm2AbsGm, uint64_t gmRemoteBaseOffset,
                                                      __ubuf__ int32_t* tokenMetaInfo, uint32_t descRowBase)
{
    const uint32_t h = common.tokenHiddenDim;
    const int32_t flags = dedupScratch.descTensor[descRowBase];
    const uint32_t memberCount = static_cast<uint32_t>(flags & DEDUP_ROW_MEMBER_COUNT_MASK);
    const int32_t minTopkIdx = dedupScratch.descTensor[descRowBase + 1U];
    CombineImpl::CombineTokenRoute route = CombineImpl::LoadCombineTokenRoute(tokenMetaInfo, 0U);
    __gm__ bfloat16_t* gmRemoteD =
        reinterpret_cast<__gm__ bfloat16_t*>(GetRankWinAddrWithOffset(route.dstRankId, gmRemoteBaseOffset));
    uint64_t gmDstRowOffset =
        (static_cast<uint64_t>(route.tokenIdx) * params.tilingData->compute.topK + static_cast<uint64_t>(minTopkIdx)) *
        h;

    const uint32_t mergeBuf = dedupScratch.mergeRowSequence & 1U;
    dedupScratch.mergeRowSequence++;
    const TEventID mergeEvent =
        static_cast<TEventID>(WAVE_COMBINE_DEDUP_MERGE_EVENT_BASE + static_cast<int32_t>(mergeBuf));
    // 半区步长与 LayoutWaveCombineDedupScratch 的 rowStrideBytes 同源（CeilAlign(h*2B,32)/2B 元素）。
    const uint32_t mergeStrideElems =
        static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<uint64_t>(h) * sizeof(bfloat16_t), static_cast<uint64_t>(ALIGN_32))) /
        sizeof(bfloat16_t);
    auto* mergeSendUb = dedupScratch.mergeSendTensor + mergeBuf * mergeStrideElems;
    /*
     * 等本半区上一次发送（MTE3 读）完成（首两轮等到的是预挂 Set），才能复写；另一半区的
     * 发送与本行的累加/Cast 并行——跨卡 MTE3 写延迟不再卡住下一行计算。
     */
    PipeWait<PIPE_MTE3, PIPE_V>(mergeEvent);
    const bool mulWeights = !params.tilingData->isTopkWeightsPrefetch;
    if (mulWeights) {
        LoadDedupMergeWeightBits(dedupScratch, params, descRowBase, memberCount, mergeEvent);
    }
    for (uint32_t segBase = 0U; segBase < h; segBase += WAVE_COMBINE_DEDUP_SEG_ELEMS) {
        const uint32_t segLen = h - segBase < WAVE_COMBINE_DEDUP_SEG_ELEMS ? h - segBase : WAVE_COMBINE_DEDUP_SEG_ELEMS;
        AccumulateDedupMemberSegment(dedupScratch, gmm2AbsGm, params, descRowBase, memberCount, mulWeights, h, segBase,
                                     segLen);
#if defined(MEGA_MOE_ARCH92_DEVICE)
        {
            LocalTensor<bfloat16_t> dst(TPosition::VECIN, reinterpret_cast<uint64_t>(mergeSendUb + segBase), segLen);
            LocalTensor<float> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(dedupScratch.accumTensor), segLen);
            Cast(dst, src, RoundMode::CAST_RINT, segLen);
        }
#else
        if constexpr (g_coreType == AIV) {
            CastFp32ToBf16VF(mergeSendUb + segBase, dedupScratch.accumTensor, segLen);
        }
#endif
    }
    if (mulWeights) {
        // 权重区读取全部结束，放行下一组的装载（与 LoadDedupMergeWeightBits 的装载配平，行内自闭环）。
        PipeNotify<PIPE_S, PIPE_MTE2>(mergeEvent);
        PipeWait<PIPE_S, PIPE_MTE2>(mergeEvent);
    }
    PipeNotify<PIPE_V, PIPE_MTE3>(mergeEvent);
    PipeWait<PIPE_V, PIPE_MTE3>(mergeEvent);
    CopyUbToGm(gmRemoteD + gmDstRowOffset, mergeSendUb, h * static_cast<uint32_t>(sizeof(bfloat16_t)));
    PipeNotify<PIPE_MTE3, PIPE_V>(mergeEvent);
}

/*
 * 排空去重 combine 的全部在途事件：行环 slot（复用基础 Drain）+ 成员乒乓 + 合并发送。
 * 与 ArmWaveCombineDedupEvents 严格配对：每个事件 id 无条件 Wait 掉最后一个在途 Set。
 */
__aicore__ inline void DrainCombineRowBuffersDedup(uint32_t& issuedRowCount, uint32_t rowBufferCount)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    DrainCombineRowBuffers(issuedRowCount, rowBufferCount);
    PipeWait<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE));
    PipeWait<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE + 1));
    PipeWait<PIPE_MTE3, PIPE_V>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MERGE_EVENT_BASE));
    PipeWait<PIPE_MTE3, PIPE_V>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MERGE_EVENT_BASE + 1));
}

#ifndef MEGA_MOE_ARCH92_DEVICE
// prepass 槽 0（done 计数：AIV1 低 16 位、AIV0 参与档另有高 16 位）到齐后整张 desc 表方可读；
// combine 消费点远晚于 prepass 收尾，此处等待实际≈0，仅作契约兜底。
__aicore__ inline void WaitDedupPrepassAllDone(const Params& params)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    WaitUntilGmFlagEquals(
        reinterpret_cast<__gm__ int32_t*>(workspace.mte.sync.dedupReady.get_ptr(buffer)),
        params.tilingData->compute.hiddenDim <= DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM ?
            (static_cast<int32_t>(MegaMoeHardwarePolicy::GetBlockNum()) |
             (static_cast<int32_t>(MegaMoeHardwarePolicy::GetBlockNum()) << DEDUP_PREPASS_DONE_AIV0_SHIFT)) :
            static_cast<int32_t>(MegaMoeHardwarePolicy::GetBlockNum()));
}
#endif

#ifdef MEGA_MOE_ARCH92_DEVICE
// Arch92 单 AIV 配比，只等待低位完成计数。
__aicore__ inline void WaitDedupPrepassAllDone(const Params& params)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    WaitUntilGmFlagEquals(reinterpret_cast<__gm__ int32_t*>(workspace.mte.sync.dedupReady.get_ptr(buffer)),
                          static_cast<int32_t>(MegaMoeHardwarePolicy::GetBlockNum()));
}
#endif

__aicore__ inline WaveCombineDedupGmViews CreateWaveCombineDedupGmViews(const MoeStageCommonConfig& common,
                                                                        const Params& params,
                                                                        const ExpertLoopState& state)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    const auto& peermem = params.communication.layout;
    WaveCombineDedupGmViews views;
    views.rowDescGm = reinterpret_cast<__gm__ int32_t*>(workspace.mte.dedupRowDescriptors.get_ptr(buffer));
    // Combine 去重的 GMM2 结果全量保存，跨专家和跨 wave 的成员按逻辑行读取。
    GM_ADDR combineData = workspace.routed.gmm2Output.get_ptr(buffer);
    views.gmm2AbsGm = reinterpret_cast<__gm__ bfloat16_t*>(combineData);
    views.gmm2OutGm = reinterpret_cast<__gm__ bfloat16_t*>(
        combineData + static_cast<uint64_t>(state.globalTokenStartIndex) * common.tokenHiddenDim * sizeof(bfloat16_t));
    views.gmRemoteBaseOffset = static_cast<uint64_t>(peermem.combineSend.offsetBytes);
    return views;
}

// 对已装载的一批 desc/metaInfo 逐行分流：SKIP 行跳过（贡献由组内末行合并发送）、PLAIN 行走
// 基线行环发送、LAST 行走合并发送。
__aicore__ inline void ProcessWaveCombineDedupBatch(const MoeStageCommonConfig& common,
                                                    const WaveCombineBufferConfig& bufferConfig,
                                                    WaveCombineScratch& scratch, WaveCombineDedupScratch& dedupScratch,
                                                    const Params& params, __gm__ bfloat16_t* gmm2OutGm,
                                                    __gm__ bfloat16_t* gmm2AbsGm, uint64_t gmRemoteBaseOffset,
                                                    uint32_t strideInt32, uint32_t batchTokenNum,
                                                    uint32_t batchTokenLocalBase, uint32_t& rowSequence)
{
    for (uint32_t batchTokenIdx = 0U; batchTokenIdx < batchTokenNum; ++batchTokenIdx) {
        const int32_t flags = dedupScratch.descTensor[batchTokenIdx * strideInt32];
        const bool inGroup = (flags & DEDUP_ROW_FLAG_GROUP) != 0;
        if (inGroup && (flags & DEDUP_ROW_FLAG_LAST) == 0) {
            continue; // SKIP：贡献由组内末行合并发送
        }
        __ubuf__ int32_t* tokenMetaInfo = (scratch.metaInfoTensor + (batchTokenIdx * META_INFO_SIZE));
        if (!inGroup) {
            uint32_t slot = rowSequence % bufferConfig.rowBufferCount;
            uint32_t tokenLocal = batchTokenLocalBase + batchTokenIdx;
            if (rowSequence < bufferConfig.rowBufferCount) {
                SendWaveCombineToken<false>(COMBINE_NO_QUANT, common, bufferConfig, scratch, params, gmm2OutGm,
                                            gmRemoteBaseOffset, tokenLocal, tokenMetaInfo, slot);
            } else {
                SendWaveCombineToken<true>(COMBINE_NO_QUANT, common, bufferConfig, scratch, params, gmm2OutGm,
                                           gmRemoteBaseOffset, tokenLocal, tokenMetaInfo, slot);
            }
            ++rowSequence;
            continue;
        }
        SendWaveCombineTokenDedupMerge(common, dedupScratch, params, gmm2AbsGm, gmRemoteBaseOffset, tokenMetaInfo,
                                       batchTokenIdx * strideInt32);
    }
}

// 去重版专家粒度 combine：desc 与 metaInfo 按批装载（批宽 = desc 区容量），逐行按 PLAIN/SKIP/LAST
// 分流。等待只有一处 WaitWaveGmm2Ready(当前专家)——专家序推进使成员就绪成为结构性保证。
template <bool IncludeAiv0 = false>
__aicore__ inline void RunWaveCombineStageDedup(const MoeStageCommonConfig& common, const AivJobContext& job,
                                                const WaveCombineBufferConfig& bufferConfig,
                                                WaveCombineScratch& scratch, WaveCombineDedupScratch& dedupScratch,
                                                const Params& params, const ExpertLoopState& state, uint32_t expertIdx,
                                                uint32_t& rowSequence)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    uint32_t currentExpertTokenNum = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
    WorkRange currentCoreTokenRange = GetWaveCombineOwnedRange<IncludeAiv0>(
        job, currentExpertTokenNum, static_cast<uint64_t>(state.globalTokenStartIndex));
    if (currentCoreTokenRange.count == 0U) {
        return;
    }
    WaitWaveGmm2Ready(job, params, expertIdx);
    WaitDedupPrepassAllDone(params);

    const uint32_t strideInt32 =
        static_cast<uint32_t>(CalcDedupRowDescStrideInt32(static_cast<int64_t>(params.tilingData->compute.topK)));
    uint32_t batchCap = WAVE_COMBINE_DEDUP_DESC_UB_BYTES / (strideInt32 * static_cast<uint32_t>(sizeof(int32_t)));
    batchCap = batchCap > WAVE_COMBINE_META_INFO_TOKEN_CAPACITY ? WAVE_COMBINE_META_INFO_TOKEN_CAPACITY : batchCap;
    WaveCombineDedupGmViews gmViews = CreateWaveCombineDedupGmViews(common, params, state);
    __gm__ int32_t* rowDescGm = gmViews.rowDescGm;
    __gm__ bfloat16_t* gmm2AbsGm = gmViews.gmm2AbsGm;
    __gm__ bfloat16_t* gmm2OutGm = gmViews.gmm2OutGm;
    const uint64_t gmRemoteBaseOffset = gmViews.gmRemoteBaseOffset;
    const uint64_t globalRowBase = static_cast<uint64_t>(state.globalTokenStartIndex) + currentCoreTokenRange.start;

    for (uint32_t processedTokenNum = 0U; processedTokenNum < currentCoreTokenRange.count;) {
        uint32_t batchTokenNum = currentCoreTokenRange.count - processedTokenNum;
        batchTokenNum = batchTokenNum > batchCap ? batchCap : batchTokenNum;
        PreloadWaveCombineMetaInfo(params, scratch, globalRowBase + processedTokenNum, batchTokenNum, 0U);
        // 上一批 desc 的标量读先于本批覆盖。
        PipeFence<PIPE_S, PIPE_MTE2, SYNC_EVENT_ID4>();
        CopyGmToUb(dedupScratch.descTensor, (rowDescGm + ((globalRowBase + processedTokenNum) * strideInt32)),
                   (batchTokenNum * strideInt32) * sizeof(int32_t));
        PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID4>();
        const uint32_t batchTokenLocalBase = currentCoreTokenRange.start + processedTokenNum;
        ProcessWaveCombineDedupBatch(common, bufferConfig, scratch, dedupScratch, params, gmm2OutGm, gmm2AbsGm,
                                     gmRemoteBaseOffset, strideInt32, batchTokenNum, batchTokenLocalBase, rowSequence);
        processedTokenNum += batchTokenNum;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_DEDUP_H
