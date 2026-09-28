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

#include "mega_moe_gmm2_combine.h"

namespace MegaMoeImpl {

// ============================== combine 去重 ==============================
// 干什么：基线 combine 里，同一 token 在本卡的多个专家输出行要逐行发回源卡、由源卡累加。
// combine 去重把这些行在本卡先合并成一行再发，发送量从「组内行数」降到 1。
// 怎么跑：GMM2 的输出本来就全量落在 workspace（gmm2MmadRes）里，combine 作为一个独立
// 阶段按专家从小到大消费。prepass 建表时保证了组内成员按专家升序、组内最大专家的那行
// 是 LAST（合并驱动行）——所以轮到专家 e 时，e 组内的其它成员必然已经算完落盘，
// 等一个 WaitWaveGmm2Ready(e) 就等于成员全就绪，不需要任何跨专家的等待标记。
// 怎么合：每行在激活层已经乘过自己的权重（去重强制 prefetch 语义），合并就是 fp32 整行
// 相加，加完一次 Cast、发到 (token, minTopkIdx) 槽。
// 三类行：PLAIN（不在组里）走基线的行环直发；SKIP（组内非 LAST）直接跳过；LAST 走上面
// 的合并发送。desc 描述表按批装载，每批只花一次 MTE2。
// ==========================================================================

// 成员行/累加按段处理：段长固定，缓冲不随 h 增长，避免整行 fp32 双缓冲挤爆 UB。
constexpr uint32_t WAVE_COMBINE_DEDUP_SEG_ELEMS = 4096U;
constexpr uint32_t WAVE_COMBINE_DEDUP_DESC_UB_BYTES = 12U * 1024U;
// tkw=0 乘权用：LAST 行 revWeights 槽（整 token topK 个 fp32 位型权重）的 UB 装载区，
// topK ≤ 32（host 校验）→ 128B 定长。
constexpr uint32_t WAVE_COMBINE_DEDUP_WEIGHT_UB_BYTES = 128U;
// 成员乒乓装载事件（MTE2_V/V_MTE2）用 6/7，合并发送事件（V_MTE3/MTE3_V）用 6；
// 与行环 slot 事件（MTE3_MTE2 的 0..5）分属不同 pipe 对或不同 id，互不冲突。
constexpr int32_t WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE = 6;
constexpr int32_t WAVE_COMBINE_DEDUP_MERGE_EVENT = 6;

struct WaveCombineDedupScratch {
    LocalTensor<bfloat16_t> mergeSendTensor; // 整行 bf16 合并结果（发送源）
    LocalTensor<bfloat16_t> memberTensor;    // 2 × 段长 bf16 成员装载乒乓
    LocalTensor<float> memberFp32Tensor;     // 段长 fp32 成员转换暂存
    LocalTensor<float> accumTensor;          // 段长 fp32 累加
    LocalTensor<int32_t> descTensor;         // rowDesc 批装载
    LocalTensor<int32_t> weightTensor;       // tkw=0：LAST 行整 token 权重段装载（行内自闭环）
};

// 去重 combine 用到的 GM 视图：rowDesc 表、GMM2 结果（绝对基址/本专家段基址）与远端发送基偏移。
struct WaveCombineDedupGmViews {
    GlobalTensor<int32_t> rowDescGm;
    GlobalTensor<bfloat16_t> gmm2AbsGm;
    GlobalTensor<bfloat16_t> gmm2OutGm;
    uint64_t gmRemoteBaseOffset = 0;
};

// 成员乒乓（V_MTE2 事件 6/7）与合并发送（MTE3_V 事件 6）走「先挂再等」协议：进 wave 前先
// Set 一次（本函数），之后每次复写缓冲前无条件 Wait、用完再 Set，wave 结束由
// DrainCombineRowBuffersDedup 无条件 Wait 收尾。这样任何时刻每个事件 id 上恰有一个在途
// Set：首轮 Wait 等到的是这里预挂的 Set（不死等），结束时全部被 Drain 消费（无悬挂残留）。
// 调用本函数的开关条件必须与调用 Drain 的条件完全一致，否则一边挂一边不收，直接死等。
__aicore__ inline void ArmWaveCombineDedupEvents()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE));
    SetFlag<HardEvent::V_MTE2>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE + 1));
    SetFlag<HardEvent::MTE3_V>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MERGE_EVENT));
}

// 在行环之后依次落位去重各 UB 段：合并发送整行 → 成员乒乓 → fp32 转换 → 累加 → desc 批区 → 权重区。
__aicore__ inline void LayoutWaveCombineDedupScratch(WaveCombineDedupScratch &dedupScratch, uint32_t addr,
                                                     uint32_t rowStrideBytes)
{
    constexpr uint32_t segBf16Bytes = WAVE_COMBINE_DEDUP_SEG_ELEMS * sizeof(bfloat16_t);
    constexpr uint32_t segFp32Bytes = WAVE_COMBINE_DEDUP_SEG_ELEMS * sizeof(float);
    dedupScratch.mergeSendTensor = LocalTensor<bfloat16_t>(TPosition::VECIN, addr, rowStrideBytes / sizeof(bfloat16_t));
    addr += rowStrideBytes;
    dedupScratch.memberTensor = LocalTensor<bfloat16_t>(TPosition::VECIN, addr, 2U * WAVE_COMBINE_DEDUP_SEG_ELEMS);
    addr += 2U * segBf16Bytes;
    dedupScratch.memberFp32Tensor = LocalTensor<float>(TPosition::VECCALC, addr, WAVE_COMBINE_DEDUP_SEG_ELEMS);
    addr += segFp32Bytes;
    dedupScratch.accumTensor = LocalTensor<float>(TPosition::VECCALC, addr, WAVE_COMBINE_DEDUP_SEG_ELEMS);
    addr += segFp32Bytes;
    dedupScratch.descTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, addr, WAVE_COMBINE_DEDUP_DESC_UB_BYTES / sizeof(int32_t));
    addr += WAVE_COMBINE_DEDUP_DESC_UB_BYTES;
    dedupScratch.weightTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, addr, WAVE_COMBINE_DEDUP_WEIGHT_UB_BYTES / sizeof(int32_t));
}

// 去重专属 UB 规划（[WAVE_COMBINE_UB_BASE, WAVE_COMBINE_UB_LIMIT) 内动态切分）：
// 行环(n×行宽) + 合并发送整行 + 成员乒乓/fp32/累加三段（段长固定）+ desc 批区。
// host 收益门已保证 h 满足容量约束（3×行宽 + 固定 60KiB ≤ 120KiB，即 h ≤ 10240）。
template <bool IncludeAiv0 = false>
__aicore__ inline WaveCombineBufferConfig InitWaveCombineBuffersDedup(const MoeStageCommonConfig &common,
                                                                      WaveCombineScratch &scratch,
                                                                      WaveCombineDedupScratch &dedupScratch)
{
    WaveCombineBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }
    if (!IncludeAiv0 && GetSubBlockIdx() != 1U) {
        return bufferConfig;
    }
    bufferConfig.rowBytes = common.tokenHiddenDim * static_cast<uint32_t>(sizeof(bfloat16_t));
    bufferConfig.rowStrideBytes = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(bufferConfig.rowBytes), static_cast<uint64_t>(ALIGN_32)));
    bufferConfig.slotStrideBytes = bufferConfig.rowStrideBytes;
    constexpr uint32_t segBf16Bytes = WAVE_COMBINE_DEDUP_SEG_ELEMS * sizeof(bfloat16_t);
    constexpr uint32_t segFp32Bytes = WAVE_COMBINE_DEDUP_SEG_ELEMS * sizeof(float);
    const uint32_t fixedBytes = bufferConfig.rowStrideBytes /* mergeSend */ + 2U * segBf16Bytes /* member 乒乓 */ +
                                2U * segFp32Bytes /* memberFp32 + accum */ + WAVE_COMBINE_DEDUP_DESC_UB_BYTES +
                                WAVE_COMBINE_DEDUP_WEIGHT_UB_BYTES;
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
    scratch.rowBufferTensor =
        LocalTensor<bfloat16_t>(TPosition::VECIN, WAVE_COMBINE_UB_BASE, rowRingBytes / sizeof(bfloat16_t));
    scratch.metaInfoTensor = LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR,
                                                  WAVE_COMBINE_META_INFO_TOKEN_CAPACITY * META_INFO_SIZE);
    LayoutWaveCombineDedupScratch(dedupScratch, WAVE_COMBINE_UB_BASE + rowRingBytes, bufferConfig.rowStrideBytes);
    return bufferConfig;
}

// tkw=0：把 LAST 行（本行）revWeights 槽的整 token 权重段一次连续读回 weightTensor。
// 权重段由 dispatch 侧落盘；读取贯穿整组累加，放行事件（S_MTE2）由调用方在组尾配平。
__aicore__ inline void LoadDedupMergeWeightBits(WaveCombineDedupScratch &dedupScratch, const Params &params,
                                                uint32_t descRowBase, uint32_t memberCount, TEventID mergeEvent)
{
    const uint32_t topK = params.tilingData->topK;
    const uint32_t weightAlignBytes =
        Ops::Base::CeilAlign(topK * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    const uint64_t lastRow = static_cast<uint64_t>(static_cast<uint32_t>(
        dedupScratch.descTensor.GetValue(descRowBase + DEDUP_ROW_DESC_HEADER_INT32 + (memberCount - 1U) * 3U)));
    GlobalTensor<int32_t> revWeightsGm;
    revWeightsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.dispatchRevWeightsPtr + lastRow * weightAlignBytes));
    DataCopyPad(dedupScratch.weightTensor, revWeightsGm, {1U, topK * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U},
                {false, 0U, 0U, 0U});
    SetFlag<HardEvent::MTE2_S>(mergeEvent);
    WaitFlag<HardEvent::MTE2_S>(mergeEvent);
}

// 取第 memberIdx 个成员的合并权重：desc 给出该成员的 topkIndex，从整 token 权重段按位取回 float。
__aicore__ inline float GetDedupMemberWeight(const WaveCombineDedupScratch &dedupScratch, const Params &params,
                                             uint32_t descRowBase, uint32_t memberIdx)
{
    const int32_t memberTopkIdx =
        dedupScratch.descTensor.GetValue(descRowBase + DEDUP_ROW_DESC_HEADER_INT32 + memberIdx * 3U + 1U);
    const int32_t weightBits =
        dedupScratch.weightTensor.GetValue(static_cast<uint32_t>(memberTopkIdx) % params.tilingData->topK);
    return *reinterpret_cast<const float *>(&weightBits);
}

// 对一个 h 段读齐组内全部成员并 fp32 累加进 accumTensor（tkw=0 先按成员权重乘）。
// 成员乒乓事件跨行存活：ArmWaveCombineDedupEvents 已预挂首轮 Set，此处直接 Wait。
__aicore__ inline void AccumulateDedupMemberSegment(WaveCombineDedupScratch &dedupScratch,
                                                    GlobalTensor<bfloat16_t> &gmm2AbsGm, const Params &params,
                                                    uint32_t descRowBase, uint32_t memberCount, bool mulWeights,
                                                    uint32_t h, uint32_t segBase, uint32_t segLen)
{
    DataCopyExtParams segGm2UbParams{1U, segLen * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> segPad{false, 0U, 0U, 0U};
    for (uint32_t memberIdx = 0U; memberIdx < memberCount; ++memberIdx) {
        const uint64_t memberRow = static_cast<uint64_t>(static_cast<uint32_t>(
            dedupScratch.descTensor.GetValue(descRowBase + DEDUP_ROW_DESC_HEADER_INT32 + memberIdx * 3U)));
        const uint32_t buf = memberIdx & 1U;
        TEventID memberEvent = static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE + static_cast<int32_t>(buf));
        // 等上一次用这个半区的 Cast 读完（首轮等到的是预挂 Set），才能复写半区。
        WaitFlag<HardEvent::V_MTE2>(memberEvent);
        LocalTensor<bfloat16_t> memberUb = dedupScratch.memberTensor[buf * WAVE_COMBINE_DEDUP_SEG_ELEMS];
        DataCopyPad(memberUb, gmm2AbsGm[memberRow * h + segBase], segGm2UbParams, segPad);
        SetFlag<HardEvent::MTE2_V>(memberEvent);
        WaitFlag<HardEvent::MTE2_V>(memberEvent);
        if (memberIdx == 0U) {
            Cast(dedupScratch.accumTensor, memberUb, RoundMode::CAST_NONE, segLen);
            if (mulWeights) {
                PipeBarrier<PIPE_V>();
                Muls(dedupScratch.accumTensor, dedupScratch.accumTensor,
                     GetDedupMemberWeight(dedupScratch, params, descRowBase, 0U), segLen);
            }
        } else {
            Cast(dedupScratch.memberFp32Tensor, memberUb, RoundMode::CAST_NONE, segLen);
            PipeBarrier<PIPE_V>();
            if (mulWeights) {
                Muls(dedupScratch.memberFp32Tensor, dedupScratch.memberFp32Tensor,
                     GetDedupMemberWeight(dedupScratch, params, descRowBase, memberIdx), segLen);
                PipeBarrier<PIPE_V>();
            }
            Add(dedupScratch.accumTensor, dedupScratch.accumTensor, dedupScratch.memberFp32Tensor, segLen);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(memberEvent);
    }
}

// 合并驱动行（LAST）：按段读齐组内全部成员（含本行）→ fp32 累加 → Cast 进整行发送缓冲，
// 一次发到 (tokenIdx*topK + minTopkIdx) 槽。成员行号是全局接收行号，直接对 gmm2MmadRes 绝对基址寻址。
__aicore__ inline void SendWaveCombineTokenDedupMerge(const MoeStageCommonConfig &common,
                                                      WaveCombineDedupScratch &dedupScratch, const Params &params,
                                                      GlobalTensor<bfloat16_t> &gmm2AbsGm, uint64_t gmRemoteBaseOffset,
                                                      LocalTensor<int32_t> &tokenMetaInfo, uint32_t descRowBase)
{
    const uint32_t h = common.tokenHiddenDim;
    const int32_t flags = dedupScratch.descTensor.GetValue(descRowBase);
    const uint32_t memberCount = static_cast<uint32_t>(flags & DEDUP_ROW_MEMBER_COUNT_MASK);
    const int32_t minTopkIdx = dedupScratch.descTensor.GetValue(descRowBase + 1U);
    CombineImpl::CombineTokenRoute route = CombineImpl::LoadCombineTokenRoute(tokenMetaInfo, 0U);
    GlobalTensor<bfloat16_t> gmRemoteD;
    gmRemoteD.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(GetRankWinAddrWithOffset(route.dstRankId, gmRemoteBaseOffset)));
    uint64_t gmDstRowOffset =
        (static_cast<uint64_t>(route.tokenIdx) * params.tilingData->topK + static_cast<uint64_t>(minTopkIdx)) * h;

    constexpr TEventID mergeEvent = static_cast<TEventID>(WAVE_COMBINE_DEDUP_MERGE_EVENT);
    // 等上一合并行发送（MTE3 读 mergeSend）完成（首轮等到的是预挂 Set），才能复写发送缓冲。
    WaitFlag<HardEvent::MTE3_V>(mergeEvent);
    const bool mulWeights = params.tilingData->topkWeightsPrefetch == 0;
    if (mulWeights) {
        LoadDedupMergeWeightBits(dedupScratch, params, descRowBase, memberCount, mergeEvent);
    }
    for (uint32_t segBase = 0U; segBase < h; segBase += WAVE_COMBINE_DEDUP_SEG_ELEMS) {
        const uint32_t segLen = h - segBase < WAVE_COMBINE_DEDUP_SEG_ELEMS ? h - segBase : WAVE_COMBINE_DEDUP_SEG_ELEMS;
        AccumulateDedupMemberSegment(dedupScratch, gmm2AbsGm, params, descRowBase, memberCount, mulWeights, h, segBase,
                                     segLen);
        Cast(dedupScratch.mergeSendTensor[segBase], dedupScratch.accumTensor, RoundMode::CAST_RINT, segLen);
        PipeBarrier<PIPE_V>(); // 下一段的累加 Cast 复写 accum 前，本段回写 Cast 已读完
    }
    if (mulWeights) {
        // 权重区读取全部结束，放行下一组的装载（与 LoadDedupMergeWeightBits 的装载配平，行内自闭环）。
        SetFlag<HardEvent::S_MTE2>(mergeEvent);
        WaitFlag<HardEvent::S_MTE2>(mergeEvent);
    }
    SetFlag<HardEvent::V_MTE3>(mergeEvent);
    WaitFlag<HardEvent::V_MTE3>(mergeEvent);
    DataCopyExtParams ub2GmParams{1U, h * static_cast<uint32_t>(sizeof(bfloat16_t)), 0U, 0U, 0U};
    DataCopyPad(gmRemoteD[gmDstRowOffset], dedupScratch.mergeSendTensor, ub2GmParams);
    SetFlag<HardEvent::MTE3_V>(mergeEvent);
}

// 排空去重 combine 的全部在途事件：行环 slot（复用基础 Drain）+ 成员乒乓 + 合并发送。
// 与 ArmWaveCombineDedupEvents 严格配对：每个事件 id 无条件 Wait 掉最后一个在途 Set。
__aicore__ inline void DrainCombineRowBuffersDedup(uint32_t &issuedRowCount, uint32_t rowBufferCount)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    DrainCombineRowBuffers(issuedRowCount, rowBufferCount);
    WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE));
    WaitFlag<HardEvent::V_MTE2>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MEMBER_EVENT_BASE + 1));
    WaitFlag<HardEvent::MTE3_V>(static_cast<TEventID>(WAVE_COMBINE_DEDUP_MERGE_EVENT));
}

// prepass 槽 0（done 计数：AIV1 低 16 位、AIV0 参与档另有高 16 位）到齐后整张 desc 表方可读；
// combine 消费点远晚于 prepass 收尾，此处等待实际≈0，仅作契约兜底。
__aicore__ inline void WaitDedupPrepassAllDone(const Params &params)
{
    WaitUntilGmFlagEquals(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.dedupPrepassReadyPtr),
                          params.tilingData->hiddenDim <= DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM ?
                              (static_cast<int32_t>(GetBlockNum()) | (static_cast<int32_t>(GetBlockNum()) << 16)) :
                              static_cast<int32_t>(GetBlockNum()));
}

__aicore__ inline WaveCombineDedupGmViews CreateWaveCombineDedupGmViews(const MoeStageCommonConfig &common,
                                                                        const Params &params,
                                                                        const ExpertLoopState &state)
{
    WaveCombineDedupGmViews views;
    views.rowDescGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.dedupRowDescPtr));
    views.gmm2AbsGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.workspaceInfo.gmm2MmadResPtr));
    views.gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
        params.workspaceInfo.gmm2MmadResPtr +
        static_cast<uint64_t>(state.globalTokenStartIndex) * common.tokenHiddenDim * sizeof(bfloat16_t)));
    views.gmRemoteBaseOffset =
        static_cast<uint64_t>(params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr);
    return views;
}

// 对已装载的一批 desc/metaInfo 逐行分流：SKIP 行跳过（贡献由组内末行合并发送）、PLAIN 行走
// 基线行环发送、LAST 行走合并发送。
__aicore__ inline void ProcessWaveCombineDedupBatch(const MoeStageCommonConfig &common,
                                                    const WaveCombineBufferConfig &bufferConfig,
                                                    WaveCombineScratch &scratch, WaveCombineDedupScratch &dedupScratch,
                                                    const Params &params, GlobalTensor<bfloat16_t> &gmm2OutGm,
                                                    GlobalTensor<bfloat16_t> &gmm2AbsGm, uint64_t gmRemoteBaseOffset,
                                                    uint32_t strideInt32, uint32_t batchTokenNum,
                                                    uint32_t batchTokenLocalBase, uint32_t &rowSequence)
{
    for (uint32_t batchTokenIdx = 0U; batchTokenIdx < batchTokenNum; ++batchTokenIdx) {
        const int32_t flags = dedupScratch.descTensor.GetValue(batchTokenIdx * strideInt32);
        const bool inGroup = (flags & DEDUP_ROW_FLAG_GROUP) != 0;
        if (inGroup && (flags & DEDUP_ROW_FLAG_LAST) == 0) {
            continue; // SKIP：贡献由组内末行合并发送
        }
        LocalTensor<int32_t> tokenMetaInfo = scratch.metaInfoTensor[batchTokenIdx * META_INFO_SIZE];
        if (!inGroup) {
            uint32_t slot = rowSequence % bufferConfig.rowBufferCount;
            uint32_t tokenLocal = batchTokenLocalBase + batchTokenIdx;
            if (rowSequence < bufferConfig.rowBufferCount) {
                SendWaveCombineToken<COMBINE_NO_QUANT, false>(common, bufferConfig, scratch, params, gmm2OutGm,
                                                              gmRemoteBaseOffset, tokenLocal, tokenMetaInfo, slot);
            } else {
                SendWaveCombineToken<COMBINE_NO_QUANT, true>(common, bufferConfig, scratch, params, gmm2OutGm,
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
__aicore__ inline void RunWaveCombineStageDedup(const MoeStageCommonConfig &common, const AivJobContext &job,
                                                const WaveCombineBufferConfig &bufferConfig,
                                                WaveCombineScratch &scratch, WaveCombineDedupScratch &dedupScratch,
                                                const Params &params, const ExpertLoopState &state, uint32_t expertIdx,
                                                uint32_t &rowSequence)
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
        static_cast<uint32_t>(CalcDedupRowDescStrideInt32(static_cast<int64_t>(params.tilingData->topK)));
    uint32_t batchCap = WAVE_COMBINE_DEDUP_DESC_UB_BYTES / (strideInt32 * static_cast<uint32_t>(sizeof(int32_t)));
    batchCap = batchCap > WAVE_COMBINE_META_INFO_TOKEN_CAPACITY ? WAVE_COMBINE_META_INFO_TOKEN_CAPACITY : batchCap;
    WaveCombineDedupGmViews gmViews = CreateWaveCombineDedupGmViews(common, params, state);
    GlobalTensor<int32_t> &rowDescGm = gmViews.rowDescGm;
    GlobalTensor<bfloat16_t> &gmm2AbsGm = gmViews.gmm2AbsGm;
    GlobalTensor<bfloat16_t> &gmm2OutGm = gmViews.gmm2OutGm;
    const uint64_t gmRemoteBaseOffset = gmViews.gmRemoteBaseOffset;
    const uint64_t globalRowBase = static_cast<uint64_t>(state.globalTokenStartIndex) + currentCoreTokenRange.start;

    for (uint32_t processedTokenNum = 0U; processedTokenNum < currentCoreTokenRange.count;) {
        uint32_t batchTokenNum = currentCoreTokenRange.count - processedTokenNum;
        batchTokenNum = batchTokenNum > batchCap ? batchCap : batchTokenNum;
        PreloadWaveCombineMetaInfo(params, scratch, globalRowBase + processedTokenNum, batchTokenNum, 0U);
        // 上一批 desc 的标量读先于本批覆盖。
        SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID4>();
        DataCopy(dedupScratch.descTensor, rowDescGm[(globalRowBase + processedTokenNum) * strideInt32],
                 batchTokenNum * strideInt32);
        SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID4>();
        const uint32_t batchTokenLocalBase = currentCoreTokenRange.start + processedTokenNum;
        ProcessWaveCombineDedupBatch(common, bufferConfig, scratch, dedupScratch, params, gmm2OutGm, gmm2AbsGm,
                                     gmRemoteBaseOffset, strideInt32, batchTokenNum, batchTokenLocalBase, rowSequence);
        processedTokenNum += batchTokenNum;
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_GMM2_COMBINE_DEDUP_H
